#!/usr/bin/env python3
"""Regression: the node pack against the product-label-refinement engine's own run.py, on the same job and images.

  python -I tools/regress.py                          # the demo job, pack run in-process (needs torch)
  python3 -P tools/regress.py --server http://127.0.0.1:8188 [--header "K: V"]   # pack run on a ComfyUI server
  python -I tools/regress.py --job JOB.json [--scene PATH] [--reference PATH] [--workflow WF.api.json] [--psd]
  python -I tools/regress.py --job JOB.json --draft [--quad "X,Y ..."] [--jitter 3] [--outline hand|traced]
                             [--rough-pairs] [--mask PATH|hand [--no-quad]] [--no-occluders]
                             [--label auto|off|cylinder|flat]

Runs the engine's scripts/run.py (--skill-dir, default ~/Documents/Claude/product-label-refinement) with --job-dir and --output
inside --work, never at the job's own output path; or pass --skill-out to compare with an existing engine output.
Prints the pixel differences (max, count above 0, 1 and --levels) and which report sections match; with --psd
both sides also write the layered PSD and their psd reports are compared.
Exit 0 when fewer than --max-px pixels differ by more than --levels, nothing outside the work box differs and
(with --psd) the PSDs have the same layers and flatten to their PNGs within 2 levels.

--draft scores Auto Draft against a hand-measured job. From the hand job it makes the rough input a person would
give: the quad (where the outline's bounding box corners land under the hand fit, each corner --jitter px off),
the product's outline on the packshot (--outline hand; traced: the draft traces it), the hand job's occluders 2 px
off, optionally its landmarks as rough pairs and a product mask (hand: its generated outline minus its occluders,
what a good SAM 3 mask gives). A label job (the hand job's model is cylinder, or its shape "generated") drafts in
Auto Draft's label mode (--label auto): the quad is then the generated label's 4 corners (the hand job's
qa/label_outline.json, else its measured outline's sides extended), each --jitter px off. The pack runs workflows/product-label-refinement-draft.api.json on that input and the engine runs
run.py draft on the same; their draft jobs must be identical and their images must match as above. The draft is
then scored against the hand job (run by the engine):
  edge       how far the draft puts the product's outline from where the hand fit puts it, scene px (median, max)
  placement  how far any point of the product (its print) lands from the hand fit's position, scene px
  geometry   pixels more than --levels from the hand job's image when the hand job runs with the draft's
             measurements (work box, fit, occluders, old silhouette) and its own grade, finish and fill
  vs hand    pixels of the draft's own image more than --levels from the hand job's image, and each run's changed px
  audit      each run's audit verdict (clean, WARN, FAIL)
A draft whose run stops (FILL STOPPED) passes when the engine stops with the same message, and is scored as stopped.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image

import comfy_run as cr

PACK = Path(__file__).resolve().parent.parent
SKILL = Path("~/Documents/Claude/product-label-refinement").expanduser()
JITTER = np.array([(1.0, -0.67), (-0.67, 1.0), (1.0, 0.67), (-1.0, -0.67)])   # x --jitter: a rough click per corner


def resolve(p, base):
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else Path(base) / p


def result_image(out, wf):
    """The Save Image output fed by Composite Behind's image."""
    entry = json.loads((out / "history.json").read_text())
    for nid, node in wf.items():
        src = node["inputs"].get("images")
        if node["class_type"] == "SaveImage" and cr.is_link(src) and \
                wf[src[0]]["class_type"] == "ImagePostCompositeBehind" and src[1] == 0:
            img = entry["outputs"][nid]["images"][0]
            return out / img.get("subfolder", "") / img["filename"]
    sys.exit("no Save Image node takes Composite Behind's image")


def run_skill(a, args, what, may_stop=False):
    """-> run.py's stdout; with may_stop, (stdout, the message it stopped with or None)"""
    cmd = [a.skill_python, "-P", str(Path(a.skill_dir).expanduser() / "scripts" / "run.py")] + [str(v) for v in args]
    print(f"{what:10s} " + " ".join(cmd[2:4]) + " ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode and not may_stop:
        sys.exit(f"{what} run failed:\n{r.stdout}{r.stderr}")
    if may_stop:
        return r.stdout, stop_line(r.stderr) if r.returncode else None
    return r.stdout


def stop_line(text):
    """The line a run stopped with (run.py's sys.exit text, the pack's node error)."""
    lines = [l.strip() for l in text.splitlines() if l.strip() and "Warning" not in l and not l.strip().startswith("return ")]
    stops = [l for l in lines if "STOPPED" in l or l.startswith("failed ")]
    line = (stops or lines or ["?"])[-1]
    return line.split("): ", 1)[-1] if line.startswith("failed ") else line   # failed     node ID (CLASS): message


def run_pack(a, args, out, may_stop=False):
    """The pack, through the same API workflow an agent would submit -> (prompt, output image, report.json); with
    may_stop, a run that stopped at a node gives (prompt, None, None, its message): what ran is saved already"""
    args = ["--workflow", a.workflow, "--out", str(out)] + [str(v) for v in args]
    if a.server:
        cmd = [sys.executable, "-P", str(PACK / "tools" / "comfy_run.py"), "--server", a.server] + args
        for h in a.header:
            cmd += ["--header", h]
        print(f"pack       {a.server}")
    else:
        cmd = [sys.executable, "-I", str(PACK / "tools" / "offline_run.py")] + args
        print("pack       in-process (offline_run.py)")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode and not (may_stop and r.returncode in (1, 3) and (out / "history.json").exists()):
        sys.exit(f"pack run failed:\n{r.stdout}{r.stderr}")
    wf = json.loads((out / "prompt.json").read_text())
    if r.returncode:
        return wf, None, None, stop_line(r.stdout + r.stderr)
    res = wf, result_image(out, wf), json.loads((out / "report.json").read_text())
    return res + (None,) if may_stop else res


def compare(pack_out, skill_out, pack_rep, levels):
    A = np.asarray(Image.open(pack_out).convert("RGB")).astype(np.int16)
    B = np.asarray(Image.open(skill_out).convert("RGB")).astype(np.int16)
    if A.shape != B.shape:
        sys.exit(f"size mismatch: pack {A.shape} vs skill {B.shape}")
    d = np.abs(A - B).max(2)
    x0, y0, x1, y1 = pack_rep["roi"]
    inside = np.zeros(d.shape, bool)
    inside[y0:y1, x0:x1] = True
    return {"pack_output": str(pack_out), "skill_output": str(skill_out), "max_level_diff": int(d.max()),
            "px_diff_gt0": int((d > 0).sum()), "px_diff_gt1": int((d > 1).sum()),
            f"px_diff_gt{levels}": int((d > levels).sum()), "px_diff_outside_roi": int((d[~inside] > 0).sum()),
            "pack_outside_roi_changed": pack_rep["change"]["outside_roi_changed"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill-dir", default=str(SKILL))
    ap.add_argument("--job", help="default: this pack's examples/demo/job.json")
    ap.add_argument("--scene", help="default: the job's scene")
    ap.add_argument("--reference", help="default: the job's reference")
    ap.add_argument("--workflow", help="default: workflows/product-label-refinement.api.json (with --draft: "
                    "product-label-refinement-draft.api.json; a job with matte ink, polarity light: "
                    "product-label-refinement-ink.api.json, its old_silhouette mask file into the template's mask loader)")
    ap.add_argument("--psd", action="store_true", help="also write and compare the layered PSDs")
    ap.add_argument("--server", help="run the pack on this ComfyUI server instead of in-process")
    ap.add_argument("--header", action="append", default=[])
    ap.add_argument("--skill-python", default=sys.executable)
    ap.add_argument("--skill-out", help="existing skill output to compare with (skips the skill run)")
    ap.add_argument("--work", help="scratch folder (default: a new temp folder)")
    ap.add_argument("--levels", type=int, default=8)
    ap.add_argument("--max-px", type=int, default=20)
    d = ap.add_argument_group("--draft: score Auto Draft against the hand-measured job")
    d.add_argument("--draft", action="store_true")
    d.add_argument("--quad", help="the rough quad (default: the hand fit's, --jitter px off)")
    d.add_argument("--jitter", type=float, default=3.0, help="px each derived quad corner is off (default 3)")
    d.add_argument("--outline", choices=("hand", "traced"), default="hand")
    d.add_argument("--rough-pairs", action="store_true", help="also give the hand job's landmarks, --jitter px off")
    d.add_argument("--mask", help="the product's mask (white = product), or 'hand': the hand job's generated outline "
                                  "minus its occluders")
    d.add_argument("--no-quad", action="store_true", help="with --mask: no quad, the mask alone")
    d.add_argument("--no-occluders", action="store_true")
    d.add_argument("--label", choices=("auto", "off", "cylinder", "flat"), default="auto",
                   help="Auto Draft's label mode (auto: cylinder for a cylinder hand job, flat for one in the generated "
                        "label's shape, else off)")
    d.add_argument("--name", help="the row's name (default: the job folder's)")
    a = ap.parse_args()
    job_p = Path(a.job).expanduser().resolve() if a.job else PACK / "examples" / "demo" / "job.json"
    job = json.loads(job_p.read_text())
    m = job.get("matte")
    light = isinstance(m, dict) and m.get("type") == "ink" and m.get("polarity") == "light"
    a.workflow = a.workflow or str(PACK / "workflows" / ("product-label-refinement-draft.api.json" if a.draft else
                                                         "product-label-refinement-ink.api.json" if light else
                                                         "product-label-refinement.api.json"))
    osil = job.get("old_silhouette")
    matte_file = resolve(osil["path"], job_p.parent) if isinstance(osil, dict) and osil.get("type") == "mask" else None
    scene = Path(a.scene).expanduser() if a.scene else resolve(job["scene"], job_p.parent)
    reference = Path(a.reference).expanduser() if a.reference else resolve(job["reference"], job_p.parent)
    work = Path(a.work).expanduser().resolve() if a.work else Path(tempfile.mkdtemp(prefix="imagepost-regress-"))
    work.mkdir(parents=True, exist_ok=True)
    if a.draft:
        return draft_main(a, job_p, job, scene, reference, work)

    # 1. the skill
    if a.skill_out:
        skill_out, skill_rep = Path(a.skill_out).expanduser(), None
    else:
        skill_out = work / "skill" / "out.png"   # always inside --work: never the job's own output path
        run_skill(a, [job_p, "--job-dir", work / "skill", "--output", skill_out, "--scene", scene] +
                  (["--psd"] if a.psd else []), "skill")
        skill_rep = json.loads((work / "skill" / "report.json").read_text())

    # 2. the pack, through the same API workflow an agent would submit
    out = work / "pack"
    _, pack_out, pack_rep = run_pack(a, ["--job", job_p, "--scene", scene, "--reference", reference] +
                                     (["--mask", matte_file] if matte_file else []) + (["--psd"] if a.psd else []), out)

    # 3. compare
    res = compare(pack_out, skill_out, pack_rep, a.levels)
    if skill_rep:
        res["report_sections_identical"] = {k: pack_rep.get(k) == skill_rep.get(k)
                                            for k in ("align", "grade", "finish", "fill", "change") + (("matte",) if light else ())}
    psd_ok = True
    if a.psd:
        texts = sorted(out.glob("ImagePostSavePSD-*.txt"))
        pack_psd = json.loads(texts[0].read_text())["psd"] if texts else None
        skill_psd = skill_rep.get("psd") if skill_rep else None
        res["psd"] = {"pack": pack_psd and {k: pack_psd[k] for k in ("layers", "layers_vs_png")},
                      "skill": skill_psd and {k: skill_psd[k] for k in ("layers", "layers_vs_png")}}
        psd_ok = bool(pack_psd) and pack_psd["layers_vs_png"]["max_levels"] <= 2 and \
            (skill_psd is None or pack_psd["layers"] == skill_psd["layers"])
        res["psd"]["pass"] = psd_ok
    ok = res[f"px_diff_gt{a.levels}"] < a.max_px and res["px_diff_outside_roi"] == 0 and \
        res["pack_outside_roi_changed"] == 0 and psd_ok
    res["pass"] = bool(ok)
    (work / "regress.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"{'PASS' if ok else 'FAIL'}: {res[f'px_diff_gt{a.levels}']} px differ by more than {a.levels} levels "
          f"(limit {a.max_px}); work folder {work}")
    sys.exit(0 if ok else 1)


# ---- --draft ----

def _imgpost():
    """The vendored engine (geometry, outline, masks, autofit), for the rough input and the scores."""
    if str(PACK) not in sys.path:
        sys.path.insert(0, str(PACK))
    from imgpost import autofit, geometry, images, masks, outline
    return autofit, geometry, images, masks, outline


def label_corners(job, job_p, approx):
    """The generated label's 4 sharp corners (sides extended), TL TR BR BL as on the packshot: the hand job's
    measure.py label output when it is there, else lines through its measured outline's sides 20 to 45 % of the way
    from each corner, intersected; else (no measured outline) `approx`, the hand fit's box corners."""
    lo = Path(job_p).parent / "qa" / "label_outline.json"
    if lo.exists():
        return np.asarray(json.loads(lo.read_text())["corners_sharp"], float)
    osil = job.get("old_silhouette")
    if not (isinstance(osil, dict) and osil.get("measured")):
        return np.asarray(approx, float)
    af, geo, _, _, _ = _imgpost()
    M, A, out = af.resample(np.asarray(osil["measured"], float), 0.5, closed=True), np.asarray(approx, float), []
    for i in range(4):
        c, Ls = A[i], []
        for o in (A[i - 1], A[(i + 1) % 4]):
            Ln = float(np.linalg.norm(o - c))
            t = (o - c) / Ln
            s_, off = (M - c) @ t, (M - c) @ np.array([-t[1], t[0]])
            Ls.append(geo.line_through(M[(s_ > 0.2 * Ln) & (s_ < 0.45 * Ln) & (np.abs(off) < 0.08 * Ln)]))
        out.append(np.linalg.solve(np.array([Ls[0][:2], Ls[1][:2]]), -np.array([Ls[0][2], Ls[1][2]])))
    return np.array(out)


def rough_input(a, job, scene, reference, work, job_p=None):
    """The rough input a person would give for this product, from the hand job -> (dict for the pack's Auto Draft
    node, run.py draft arguments, the hand fit, the hand outline polygon in reference px)"""
    af, geo, im, mk, ol = _imgpost()
    Hh = geo.fit(job["align"])[0]
    P = np.asarray(ol.outline_polygon(job["reference_outline"]), float)
    if a.outline == "hand":
        Q = P
    else:
        r_u8, r_a, _ = im.load_u8(reference)
        Q = np.asarray(af.trace_outline(af.reference_mask(r_u8, r_a)), float) if r_a is not None or \
            af.on_plain_background(r_u8) else np.array([(0, 0), (r_u8.shape[1] - 1, r_u8.shape[0] - 1)], float)
    (bx0, by0), (bx1, by1) = Q.min(0), Q.max(0)
    fmt = lambda pts: " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    node, cli = {"quad": "", "pairs": "", "occluders": "", "outline": "", "mask_skip": ""}, []   # the template's demo input off
    model = job["align"].get("model", "homography")
    label = a.label if a.label != "auto" else "cylinder" if model == "cylinder" else \
        "flat" if job.get("shape") == "generated" else "off"
    box = geo.apply_H(Hh, [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)])
    if a.quad:
        node["quad"] = a.quad
    elif not a.no_quad:
        node["quad"] = fmt((label_corners(job, job_p, box) if label != "off" else box) + a.jitter * JITTER)
    if node["quad"]:
        cli += ["--quad"] + node["quad"].split()
    if a.outline == "hand":
        node["outline"] = fmt(P)
        cli += ["--outline", node["outline"]]
    if a.rough_pairs:
        pairs = [f"{p['ref'][0]:.2f},{p['ref'][1]:.2f}={p['scene'][0] + dx:.2f},{p['scene'][1] + dy:.2f}"
                 for p, (dx, dy) in zip(job["align"]["points"], np.tile(a.jitter * JITTER, (8, 1)))]
        node["pairs"] = "\n".join(pairs)
        cli += ["--pairs"] + pairs
    occ = []
    for o in [] if a.no_occluders else job.get("occluders", []):
        if "polyline" in o:
            occ.append(f"{o['side']}: " + fmt([(x + 2, y - 2) for x, y in o["polyline"]]))
        elif "polygon" in o:
            occ.append("polygon: " + fmt([(x + 1.5, y - 1.5) for x, y in o["polygon"]]))
    if occ:
        node["occluders"] = "\n".join(occ)
        for o in occ:
            cli += ["--occluder", o.replace(": ", ":", 1)]
    node["model"] = model if model in ("homography", "affine") else "homography"
    node["label"] = label
    if label != "off":
        cli += ["--label", label]
    gs = job.get("grade", {})
    node["light"] = "cylinder" if gs.get("shading") else "relight" if gs.get("relight") else "off"
    cli += ["--model", node["model"]] + (["--cylinder"] if node["light"] == "cylinder" else
                                         ["--no-relight"] if node["light"] == "off" else [])
    mask = None
    if a.mask == "hand":
        # what a good SAM 3 mask of the generated product gives: its outline, minus what stands in front of it
        s_u8 = im.load_u8(scene)[0]
        osil = job["old_silhouette"]
        poly = osil["measured"] if isinstance(osil, dict) and osil.get("measured") else \
            osil if isinstance(osil, list) else geo.map_polygon(Hh, P).tolist()
        H, W = s_u8.shape[:2]
        x0, y0, x1, y1 = (int(round(v)) for v in job["roi"])
        roi = [max(0, x0), max(0, y0), min(W, x1), min(H, y1)]
        M = np.zeros((H, W), np.uint8)
        M[roi[1]:roi[3], roi[0]:roi[2]] = 255 * ((mk.poly_cover(roi, poly, ss=2) > 0.5) &
                                                 (mk.visibility(roi, job.get("occluders", [])) > 0.5))
        mask = work / "mask.png"
        Image.fromarray(M).save(mask)
    elif a.mask:
        mask = Path(a.mask).expanduser().resolve()
    if mask:
        cli += ["--mask", mask]
    return node, cli, mask, Hh, P


def scores(job, draft_job, Hh, P, ref_shape):
    """edge: distance between the outline as the draft fit and as the hand fit draw it (both ways round); placement:
    distance between where the two fits put each point inside the outline. Scene px."""
    from scipy.spatial import cKDTree
    af, geo, _, _, _ = _imgpost()
    from imgpost import raster
    Hd = geo.fit(draft_job["align"])[0]
    Pd = af.resample(P, 1.0 / max(af.local_scale(geo.as_matrix(Hh, P.mean(0)), P.mean(0)), 1e-6), closed=True)   # ~1 scene px
    A, B = geo.apply_H(Hh, Pd), geo.apply_H(Hd, Pd)
    A, B = A[np.isfinite(A).all(1)], B[np.isfinite(B).all(1)]
    edge = np.r_[cKDTree(af.resample(A, 0.25, closed=True)).query(B)[0],
                 cKDTree(af.resample(B, 0.25, closed=True)).query(A)[0]]
    ys, xs = np.nonzero(raster.fill_polygon(ref_shape[0], ref_shape[1], [tuple(q) for q in P]) > 0)
    k = max(1, len(xs) // 4000)
    R = np.c_[xs[::k], ys[::k]].astype(float)
    pa, pb = geo.apply_H(Hh, R), geo.apply_H(Hd, R)
    ok = np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
    pl = np.linalg.norm(pa[ok] - pb[ok], axis=1)
    r2 = lambda v: round(float(v), 2)
    return {"edge_px": {"median": r2(np.median(edge)), "max": r2(edge.max())},
            "placement_px": {"median": r2(np.median(pl)), "max": r2(pl.max())}}


def verdict(rep):
    found = rep.get("audit") or []
    return "FAIL" if any(f["level"] == "FAIL" for f in found) else "WARN" if any(f["level"] == "WARN" for f in found) \
        else "clean"


def geometry_job(job, draft_job, scene, reference):
    """The hand job with the draft's measurements (work box, fit, occluders, and the old silhouette unless the hand job
    measured the generated label's own outline) and everything else, the grade included, as the hand job has it."""
    hy = dict(job, scene=str(scene), reference=str(reference))
    for k in ("roi", "align", "occluders"):
        hy[k] = draft_job[k]
    if job.get("shape") != "generated" and not isinstance(job["old_silhouette"], dict):
        hy["old_silhouette"] = draft_job["old_silhouette"]
    return hy


def draft_main(a, job_p, job, scene, reference, work):
    node, cli, mask, Hh, P = rough_input(a, job, scene, reference, work, job_p)

    # 1. the engine: run.py draft on the rough input, then the draft job in full; and the hand job
    sd = work / "skill-draft"
    run_skill(a, ["draft", scene, reference] + cli + ["--job-dir", sd, "--force"], "skill")
    _, skill_stop = run_skill(a, [sd / "job.json", "--job-dir", sd, "--output", sd / "out.png"], "skill", may_stop=True)
    run_skill(a, [job_p, "--job-dir", work / "hand", "--output", work / "hand" / "out.png", "--scene", scene], "hand")
    skill_job = json.loads((sd / "job.json").read_text())
    hand_rep = json.loads((work / "hand" / "report.json").read_text())

    # 2. the pack: Auto Draft on the same rough input, through the draft workflow
    out = work / "pack"
    sets = []
    for k, v in node.items():
        sets += ["--set", f"ImagePostAutoDraft.{k}={v}"]
    wf, pack_out, pack_rep, pack_stop = run_pack(a, ["--scene", scene, "--reference", reference] + sets +
                                                 (["--mask", mask] if mask else []), out, may_stop=True)
    entry = json.loads((out / "history.json").read_text())
    files = [f for nid, n in wf.items() if n["class_type"] == "ImagePostAutoDraft"
             for f in entry["outputs"].get(nid, {}).get("files", [])]
    if not files:
        sys.exit("the Auto Draft node saved no draft job")
    pack_job = json.loads((out / files[0].get("subfolder", "") / files[0]["filename"]).read_text())

    # 3. pack against engine: the same draft job, then the same image (or the same stop)
    res = {"draft_jobs_identical": pack_job == {k: v for k, v in skill_job.items() if k not in ("scene", "reference", "output")}}
    if skill_stop or pack_stop:
        res.update(stopped={"skill": skill_stop, "pack": pack_stop})
        ok = res["draft_jobs_identical"] and skill_stop is not None and skill_stop == pack_stop
    else:
        res.update(compare(pack_out, sd / "out.png", pack_rep, a.levels))
        ok = res["draft_jobs_identical"] and res[f"px_diff_gt{a.levels}"] < a.max_px and res["px_diff_outside_roi"] == 0 \
            and res["pack_outside_roi_changed"] == 0

    # 4. the draft against the hand job
    from imgpost import images as im
    sc = scores(job, pack_job, Hh, P, im.load_u8(reference)[0].shape)
    hand_img = np.asarray(Image.open(work / "hand" / "out.png").convert("RGB")).astype(np.int16)
    gd = work / "geometry"
    gd.mkdir(exist_ok=True)
    (gd / "job.json").write_text(json.dumps(geometry_job(job, pack_job, scene, reference), indent=2))
    _, geo_stop = run_skill(a, [gd / "job.json", "--job-dir", gd, "--output", gd / "out.png", "--scene", scene],
                            "geometry", may_stop=True)

    def vs_hand(p):
        return int((np.abs(hand_img - np.asarray(Image.open(p).convert("RGB")).astype(np.int16)).max(2) > a.levels).sum())

    res["score"] = dict(sc, **{
        "vs_hand_px": vs_hand(pack_out) if pack_out else pack_stop,
        "vs_hand_geometry_px": vs_hand(gd / "out.png") if not geo_stop else geo_stop,
        "changed_px": {"draft": pack_rep["change"]["changed_px"] if pack_rep else None,
                       "hand": hand_rep["change"]["changed_px"]},
        "audit": {"draft": verdict(pack_rep) if pack_rep else "stopped", "hand": verdict(hand_rep)},
        "sides_kept": len(pack_job["align"]["lines"]),
        "trusted_points": sum(1 for q in pack_job["align"]["points"] if q["w"] >= 1.0),
        "input": {"quad": bool(node["quad"]), "jitter": a.jitter if node["quad"] and not a.quad else None,
                  "outline": a.outline, "pairs": a.rough_pairs, "mask": a.mask, "occluders": bool(node["occluders"]),
                  "label": node["label"]}})
    res["pass"] = bool(ok)
    (work / "regress.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    s = res["score"]
    fmt = lambda v: f"{v:,}" if isinstance(v, int) else "stopped"
    print(f"row        | {a.name or job_p.parent.name} | {s['edge_px']['median']} / {s['edge_px']['max']} | "
          f"{s['placement_px']['median']} / {s['placement_px']['max']} | {fmt(s['vs_hand_geometry_px'])} | "
          f"{fmt(s['vs_hand_px'])} of {s['changed_px']['hand']:,} | {s['audit']['draft']} (hand {s['audit']['hand']}) |")
    same = f"both stopped: {skill_stop}" if skill_stop and skill_stop == pack_stop else \
        f"{res.get(f'px_diff_gt{a.levels}')} px differ by more than {a.levels} levels (limit {a.max_px})"
    print(f"{'PASS' if ok else 'FAIL'}: draft jobs {'identical' if res['draft_jobs_identical'] else 'DIFFER'}, {same}; "
          f"work folder {work}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
