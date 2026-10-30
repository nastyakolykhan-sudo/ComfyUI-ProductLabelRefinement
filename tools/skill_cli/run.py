#!/usr/bin/env python3
"""product-label-refinement, stills runner. Run with `python3 -P` (keeps user site-packages, keeps the cwd off sys.path).

  run.py init SCENE REFERENCE [--job-dir DIR]
      Create the job folder (default SCENE_DIR/product-label-refinement/SCENE_STEM/) with a job.json skeleton.
  run.py draft SCENE REFERENCE --quad X,Y X,Y X,Y X,Y [--pairs RX,RY=SX,SY ...] [--occluder SIDE:X,Y X,Y ...]
               [--mask MASK.png [--mask-skip "X,Y ..."]] [--suggest] [--model homography|affine]
               [--outline X0,Y0,X1,Y1 | "X,Y X,Y ..."] [--cylinder] [--no-relight] [--label cylinder|flat]
               [--job-dir DIR] [--force]
      A first job.json from rough input (imgpost/draft.py, which the ComfyUI pack's Auto Draft node also runs),
      then the align stage: the packshot's outline traced, landmark pairs refined, edge lines proposed per straight
      side, the old silhouette and occluders snapped to edges, colour and relight defaults set. --quad: where the
      corners of the packshot outline's bounding box land in the scene (TL TR BR BL, a few px is fine); with
      --mask alone the mask's box stands in for it. Check qa/align_overlay.png and qa/draft_points.png before grading.
      --label cylinder|flat: the label only (a bottle's or jar's label; flat: a tube's panels). --outline is then the
      label on the packshot, --quad the generated label's 4 corners (TL TR BR BL as on the packshot) and --mask, if
      given, the generated label's mask: its own outline becomes the shape and its sides the fit's edges.
      [--ink light|dark [--print-box X0,Y0,X1,Y1]] [--bow "t:b | t | t/w | t/matte ..."]: print straight on a
      coloured body, lighter or darker than it (imgpost/ink.py, the mask is the matte the print stays inside) and the
      bow of a body seen from above or below (align.bow, measured from the mask); job-spec.md, matte and align.bow.
  run.py JOB.json [--stage align|grade|all] [--job-dir DIR] [--output PATH] [--scene PATH] [--psd]
      align  fit the warp            -> qa/align_overlay.png, report.json
      grade  + light/colour fit      -> qa/grade_preview.png
      all    + finish, leftover fill, composite -> output image, qa/compare.png, qa/edges.png, qa/before_after.png
             --psd (or "psd": true)  + the output as a layered .psd next to it (original, fill, product + mask)
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True   # keep __pycache__ out of the skill folder
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

from imgpost import fill as fl
from imgpost import finish as fn
from imgpost import geometry as geo
from imgpost import grade as gr
from imgpost import images as im
from imgpost import masks as mk
from imgpost import outline as ol
from imgpost import psd
from imgpost import qa
from imgpost import warp as wp


def cmd_init(a):
    scene, ref = im.resolve(a.scene).resolve(), im.resolve(a.reference).resolve()
    s, _, _ = im.load_u8(scene)
    r, ra, _ = im.load_u8(ref)
    jd = Path(a.job_dir).expanduser().resolve() if a.job_dir else scene.parent / "product-label-refinement" / scene.stem
    for sub in ("work", "qa"):
        (jd / sub).mkdir(parents=True, exist_ok=True)
    jp = jd / "job.json"
    if jp.exists():
        print(f"job.json already exists, left untouched: {jp}")
    else:
        outline = {"type": "alpha"} if ra is not None else \
            {"type": "rounded_rect", "x0": 0, "y0": 0, "x1": r.shape[1] - 1, "y1": r.shape[0] - 1, "r_top": 0, "r_bottom": 0}
        job = {
            "scene": str(scene), "reference": str(ref),
            "output": str(scene.with_name(scene.stem + "_refined.png")),
            "scene_size": [s.shape[1], s.shape[0]], "reference_size": [r.shape[1], r.shape[0]],
            "roi": None,
            "reference_outline": outline,
            "align": {"model": "homography", "points": [], "y_only": [], "lines": []},
            "old_silhouette": [],
            "occluders": [],
            "grade": {"gain": "white_level", "white_percentile": 92, "window": 41, "erode": 4, "deep_erode": 20},
            "finish": {"blur": "auto", "blur_probe": None, "grain": "auto", "edge_softness": 0.45, "rolloff": 0.85},
            "fill": {"dilate": 2},
        }
        jp.write_text(json.dumps(job, indent=2))
    print(f"scene      {scene}  {s.shape[1]}x{s.shape[0]}")
    print(f"reference  {ref}  {r.shape[1]}x{r.shape[0]}{'  (has transparency)' if ra is not None else ''}")
    print(f"job        {jp}")


def _pts(text):
    return [tuple(float(v) for v in p.split(",")) for p in text.replace(";", " ").split()]


def cmd_draft(a):
    from imgpost import draft as dr
    scene_p, ref_p = im.resolve(a.scene).resolve(), im.resolve(a.reference).resolve()
    s_u8, _, _ = im.load_u8(scene_p)
    r_u8, r_a, _ = im.load_u8(ref_p)
    jd = Path(a.job_dir).expanduser().resolve() if a.job_dir else scene_p.parent / "product-label-refinement" / scene_p.stem
    jp = jd / "job.json"
    if jp.exists() and not a.force:
        sys.exit(f"job.json already exists, left untouched: {jp} (--force replaces it)")
    if a.quad and len(_pts(" ".join(a.quad))) != 4:
        sys.exit("--quad needs 4 points: TL TR BR BL of the outline's bounding box in the scene" +
                 (" (with --label: the generated label's corners)" if a.label else ""))
    occluders = []
    for spec in a.occluder:
        side, _, txt = spec.partition(":")
        if side not in ("left", "right", "above", "below", "polygon"):
            sys.exit(f"--occluder {spec!r}: start with left:, right:, above:, below: or polygon:")
        occluders.append((side, _pts(txt)))
    outline = None
    if a.outline and len(_pts(a.outline)) >= 3:
        outline = _pts(a.outline)
    elif a.outline:
        outline = tuple(float(v) for v in a.outline.split(","))
    mask = im.load_u8(im.resolve(a.mask))[0].max(axis=2) > 127 if a.mask else None
    for sub in ("work", "qa"):
        (jd / sub).mkdir(parents=True, exist_ok=True)
    try:
        d = dr.draft(s_u8, r_u8, r_a, quad=_pts(" ".join(a.quad)) if a.quad else None,
                     pairs=[(_pts(r)[0], _pts(sc)[0]) for r, sc in (q.split("=") for q in a.pairs)],
                     occluders=occluders, mask=mask, mask_skip=_pts(a.mask_skip) if a.mask_skip else None,
                     model=a.model, outline=outline, cylinder=a.cylinder, relight=not a.no_relight, suggest=a.suggest,
                     label=a.label)
        if a.ink or a.bow:
            dr.print_on_body(d, s_u8, mask=mask, ink=a.ink,
                             print_box=[float(v) for v in a.print_box.split(",")] if a.print_box else None, bow=a.bow)
    except dr.DraftError as e:
        sys.exit(str(e))
    if a.ink in ("light", "dark") and a.mask:
        d["job"]["old_silhouette"] = {"type": "mask", "path": str(im.resolve(a.mask).resolve())}
    body = d["job"]
    job = {"about": body["about"], "scene": str(scene_p), "reference": str(ref_p),
           "output": str(scene_p.with_name(scene_p.stem + "_refined.png")),
           **{k: v for k, v in body.items() if k != "about"}}
    jp.write_text(json.dumps(job, indent=2))
    if d["sheet"]:
        d["sheet"].save(jd / "qa" / "draft_points.png")

    print(f"job        {jp}")
    for line in dr.summary(d):
        print(line)
    cmd_run(argparse.Namespace(job=str(jp), stage="align", job_dir=None, output=None, scene=None, psd=False))
    print(f"qa         {jd / 'qa' / 'draft_points.png'}")


def _print_align(rep):
    print(f"align ({rep['model']}{' + residual' if 'residual' in rep else ''}), scene px")
    for p in rep["points"]:
        print(f"  point   {p['label'][:34]:34s} dx {p['dx']:6.2f}  dy {p['dy']:6.2f}")
    for key in ("y_only", "x_only"):
        for p in rep[key]:
            print(f"  {key:7s} {p['label'][:34]:34s} d  {p['d']:6.2f}")
    for l in rep["lines"]:
        print(f"  line    {l['label'][:34]:34s} max {l['max_px']:5.2f}  mean {l['mean_px']:6.2f}")
    r = rep.get("residual")
    if r:
        print(f"residual  sigma {r['sigma']} px, max shift {r['max_shift_px']} px, {r['controls']} edge points"
              + (f", {r['dropped_n']} stray left out ({', '.join(r['dropped'])})" if r["dropped_n"] else ""))
        print("  after   " + ", ".join(f"{l['label'][:24]} {l['max_px']}" for l in r["lines_after"]))
    b = rep.get("bow")
    if b:
        print(f"bow       {b['b']}: the front {b['front_px']:+} px, the limbs {b['limb_px']:+} px along the axis" +
              ("; lines after: " + ", ".join(f"{l['label'][:24]} {l['max_px']}" for l in b["lines_after"]) if b["lines_after"] else ""))


def cmd_run(a):
    jp = Path(a.job).expanduser().resolve()
    job = json.loads(jp.read_text())
    jd = Path(a.job_dir).expanduser().resolve() if a.job_dir else jp.parent
    work, qad = jd / "work", jd / "qa"
    work.mkdir(parents=True, exist_ok=True)
    qad.mkdir(parents=True, exist_ok=True)
    scene_p, ref_p = im.resolve(a.scene or job["scene"], jp.parent), im.resolve(job["reference"], jp.parent)
    out_p = im.resolve(a.output or job.get("output") or scene_p.with_name(scene_p.stem + "_refined.png"), jp.parent)
    if out_p.resolve() == scene_p.resolve():
        sys.exit("output path is the original scene: refusing to overwrite it")
    todo = [k for k in ("roi", "old_silhouette") if not job.get(k)]
    if len(job.get("align", {}).get("points", [])) < 3:
        todo.append("align.points (3 or more)")
    if todo:
        sys.exit("job.json is missing: " + ", ".join(todo))

    scene_u8, scene_a, info = im.load_u8(scene_p)
    ref_u8, ref_a, _ = im.load_u8(ref_p)
    Hs, Ws = scene_u8.shape[:2]
    x0, y0, x1, y1 = roi = [max(0, int(round(job["roi"][0]))), max(0, int(round(job["roi"][1]))),
                            min(Ws, int(round(job["roi"][2]))), min(Hs, int(round(job["roi"][3])))]
    rep = {"job": str(jp), "scene": str(scene_p), "reference": str(ref_p), "output": str(out_p), "roi": roi, "stage": a.stage}

    def save_report():
        (jd / "report.json").write_text(json.dumps(rep, indent=2))

    osil = job["old_silhouette"]
    scene_matte = None
    if isinstance(osil, dict) and osil.get("type") == "mask":
        # the generated product's own matte (SAM 3), scene-sized
        scene_matte = _load_mask(im.resolve(osil["path"], jp.parent), (Hs, Ws), "old_silhouette")
    align = job["align"]
    if align.get("bow"):
        # a round body seen from above or below: centre, axis, half-widths (and b "auto" at the seams) from the scene
        from imgpost import bow as bw
        bm = align["bow"].get("mask")
        bmask = _load_mask(im.resolve(bm, jp.parent), (Hs, Ws), "align.bow.mask") if bm else scene_matte
        try:
            filled_bow, bow_rep = bw.resolve(align["bow"], scene_u8, bmask)
        except ValueError as e:
            sys.exit(str(e))
        align = dict(align, bow=filled_bow)
        rep["bow_measured"] = dict(bow_rep, **{k: filled_bow[k] for k in ("centre", "axis", "b")})

    # 1. geometry: warp the reference into the work box
    H, rep["align"] = geo.fit(align)
    if hasattr(H, "to_json"):
        (work / "mapping.json").write_text(json.dumps(H.to_json(), indent=1))
    else:
        np.save(work / "H.npy", H)
    refmask = ol.RefMask(job["reference_outline"], ref_u8, ref_a)
    crop = im.to_float(scene_u8[y0:y1, x0:x1])
    S_lin = im.srgb_to_lin(crop)
    ref_lin = im.srgb_to_lin(im.to_float(ref_u8))
    ref_flat = job.get("grade", {}).get("ref_flatten")
    if ref_flat:
        # a cylinder's label: the packshot's studio falloff toward its limbs comes out, the scene's light goes in
        ry, rx = np.mgrid[0:ref_u8.shape[0], 0:ref_u8.shape[1]].astype(float)
        inside = refmask.sample(rx, ry) > 0.5
        if hasattr(H, "axis_x"):                 # a cylinder's paper level is read between its limbs, not past them
            inside &= np.abs(rx - H.axis_x) <= H.r - 1.0
        if hasattr(H, "axis_x"):
            from imgpost import audit as au
            for msg in au.limb_guard(H, ref_lin, inside):
                print(msg)
                rep["align"].setdefault("warnings", []).append(msg)
        ref_lin = gr.flatten(ref_lin, inside, "columns" if ref_flat == "columns" else "field", columns=hasattr(H, "axis_x"))
        pc = job.get("grade", {}).get("paper_clean")
        if pc:
            # the packshot's own paper (grain, banding, the falloff at its edges) gives way to the scene's
            ref_lin = gr.clean_paper(ref_lin, inside, 0.06 if pc is True else float(pc))
        del ry, rx
    pre = job.get("prefilter", 0.6)
    if pre == "auto":
        # anti-alias only when the scene shows the packshot smaller; enlarging needs no blur before sampling
        from imgpost import autofit as af
        P = np.asarray(refmask.polygon, float) if refmask.polygon else np.array(refmask.bbox(ref_u8.shape)).reshape(2, 2)
        pre = 0.6 if af.local_scale(H, P.mean(0)) < 1.0 else 0.0
    W_lin, a_new, (U, V) = wp.warp(ref_lin, H, roi, refmask, ss=job.get("supersample", 4), prefilter=pre)
    a_body = None
    if job["reference_outline"].get("body"):
        # parts of the outline where the real product shows its own body (a tube's grey between two labels): not
        # composited; they show the scene's body colour (filled in step 4)
        a_body = np.zeros_like(a_new)
        for poly in job["reference_outline"]["body"]:
            bm = ol.RefMask({"type": "polygon", "pts": poly}, ref_u8)
            a_body = 1.0 - (1.0 - a_body) * (1.0 - wp.bleed_cover(H, roi, bm))
    occ = job.get("occluders", [])
    vis = mk.visibility(roi, occ)
    if scene_matte is not None:
        old = scene_matte[y0:y1, x0:x1].copy()
    elif osil == "auto" or isinstance(osil, dict):
        # the generated product's outline taken from the fit (it follows the generated edges, with the residual
        # correction): stray snapped points can't bulge it, and a label replaced edge to edge needs no fill ring.
        # With "measured" (the snapped generated outline), slivers where the generated product sticks out up to
        # max_dev px join it (and get filled); vertices further out are strays and are dropped
        oc = osil if isinstance(osil, dict) else {}
        old = a_new >= oc.get("cover", 0.9)
        if oc.get("measured"):
            from scipy.ndimage import distance_transform_edt
            dout = distance_transform_edt(~old)
            Mv = np.asarray(oc["measured"], float)
            cx = np.clip(np.round(Mv[:, 0] - x0).astype(int), 0, old.shape[1] - 1)
            cy = np.clip(np.round(Mv[:, 1] - y0).astype(int), 0, old.shape[0] - 1)
            keep_v = dout[cy, cx] <= float(oc.get("max_dev", 3.0))
            if keep_v.sum() >= 3:
                old |= mk.poly_cover(roi, Mv[keep_v].tolist(), ss=4) > 0.5
    else:
        old = mk.poly_cover(roi, osil, ss=4) > 0.5   # pixel centres inside, PIL edge bleed < 1/4 px
    gen_cov = a_bleed = None
    if job.get("shape") == "generated":
        # a label on a product the job keeps: its visible shape is the generated label's own (measured from inside:
        # rounded corners, a rim of glass, a narrower stretch), and the real label, bled past its edge, fills it
        oc = osil if isinstance(osil, dict) else {}
        if not oc.get("measured"):
            sys.exit("shape 'generated' needs old_silhouette {\"type\": \"auto\", \"measured\": [[x, y], ...]}")
        if refmask.bleed <= 0:
            sys.exit("shape 'generated' needs reference_outline.bleed (ref px) so the real label reaches the shape's edge")
        gen_cov = mk.poly_cover(roi, oc["measured"], ss=4)        # all of it: measured with measure.py label, checked
        from scipy.ndimage import distance_transform_edt
        real = a_new >= 0.5
        old = gen_cov > 0.5
        a_bleed = wp.bleed_cover(H, roi, refmask)
        if hasattr(H, "axis_x") and real.any():
            # the bleed runs in packshot px and stops at the cylinder's silhouette (past it the bottle isn't there),
            # so a generated label reaching a few px past the fitted silhouette would come out cut there: in scene
            # px the paper runs on as far as the bleed would on the front
            from imgpost import autofit as af
            r_sc = refmask.bleed * af.local_scale(H, np.asarray(refmask.polygon, float).mean(0)) if refmask.polygon else 0.0
            if r_sc > 0:
                a_bleed = np.maximum(a_bleed, np.clip(r_sc + 0.5 - distance_transform_edt(~real), 0.0, 1.0))
        rep["shape"] = {"generated_px": int(old.sum()),
                        "real_past_shape_px": round(float(distance_transform_edt(~old)[real].max()), 2) if real.any() else 0.0,
                        "shape_past_real_px": round(float(distance_transform_edt(~real)[old].max()), 2) if old.any() else 0.0,
                        "unreached_px": int((old & (a_bleed < 0.5) & (vis > 0.5)).sum())}
    outline_scene = geo.map_polygon(H, refmask.polygon).tolist() if refmask.polygon else None
    qa.overlay_align(crop, im.lin_to_srgb(W_lin), a_new, roi, outline_scene,
                     [mk.occluder_polygon(roi, o) for o in occ]).save(qad / "align_overlay.png")
    _print_align(rep["align"])
    if "shape" in rep:
        q = rep["shape"]
        print(f"shape      the generated label's own outline: the real label reaches {q['real_past_shape_px']} px past it "
              f"(cut there), it reaches {q['shape_past_real_px']} px past the real one (bled paper)" +
              (f"; {q['unreached_px']} px the bleed can't reach (filled)" if q["unreached_px"] else ""))
    from imgpost import autofit as af
    from imgpost import ink as ik
    if refmask.polygon and af.on_plain_background(ref_u8) and not ik.on_body(job.get("matte")):
        # a measured outline that takes in the packshot's anti-aliased backdrop composites as a light rim (print on
        # the body lays on only print inside its print box: no outline shows)
        fr = [f for f in af.outline_fringe(ref_u8, refmask.polygon, near=0.5) if f["px"] >= 5]
        if hasattr(H, "axis_x"):                  # a cylinder's outline runs past its limb on purpose (wrap): not a rim
            fr = [f for f in fr if min(abs(f["from"][0] - H.axis_x), abs(f["to"][0] - H.axis_x)) < H.r - 1.0]
        if fr:
            rep["align"]["fringe"] = fr
            print(f"fringe     the outline takes in the packshot's backdrop on {len(fr)} stretch(es), worst "
                  f"{fr[0]['worst']:+.2f} px from {fr[0]['from']} to {fr[0]['to']}: a light rim. "
                  "measure.py fringe JOB, re-measure those sides (pitfall 45)")
    if a.stage == "align":
        save_report()
        print(f"qa         {qad / 'align_overlay.png'}")
        return
    if ik.on_body(job.get("matte")):
        return _run_on_body(a, job, rep, save_report, qad, out_p, scene_u8, scene_a, info, ref_u8, ref_lin, H, refmask,
                            roi, old, vis, pre)

    # 2. light and colour from the generated product
    gs = job.get("grade", {})
    Y, X = np.mgrid[y0:y1, x0:x1].astype(float)
    er = gs.get("erode", 4)
    inner = binary_erosion(old & (vis > 0.99), iterations=er)
    both = binary_erosion((a_new > 0.99) & old & (vis > 0.99), iterations=er)
    if a_body is not None:
        both &= a_body < 0.5
    if both.sum() < 200:
        sys.exit("the real and generated products barely overlap: check align and old_silhouette")
    gain = gs.get("gain", "white_level")
    if gain == "white_level":
        g = gr.white_level_gain(S_lin, inner, binary_erosion(inner, iterations=gs.get("deep_erode", 20)), X, Y,
                                gs.get("white_percentile", 92), gs.get("window", 41))
    elif gain == "luma_ratio":
        g = gr.luma_ratio_gain(S_lin, W_lin, both, X, Y, gs.get("sigma", 6.0), gs.get("match_hue"), gs.get("flat", 0.08))
    else:
        g = np.ones(S_lin.shape[:2])
    rep["grade"] = {"gain": gain, "gain_range": [round(float(g[both].min()), 3), round(float(g[both].max()), 3)]}
    if gs.get("colour", "scene") == "reference":
        # keep the real product's colours; only light, white balance and black level come from the scene
        def box_lin(u8, box):
            bx0, by0, bx1, by1 = [int(round(v)) for v in box]
            return np.median(im.srgb_to_lin(im.to_float(u8[by0:by1, bx0:bx1])).reshape(-1, 3), 0)
        if gs.get("white_field") not in (None, False):
            # the generated label's own paper, per pixel: the scene's light with its falloff and tint (bounce light)
            wf = gs["white_field"] if isinstance(gs["white_field"], dict) else {}
            ys_, xs_ = np.nonzero(old)
            size = int(wf.get("size", 2 * max(2, round(min(np.ptp(ys_), np.ptp(xs_)) / 24)) + 1))
            white = gr.paper_field(S_lin, inner, size, wf.get("sigma", max(1.0, size / 10)))
            white = white * wf.get("scale", 1.0)
            rep["grade"]["white_field"] = {"size": size}
        elif gs.get("white_box"):
            shape = {"luma_ratio": lambda: gr.luma_ratio_gain(S_lin, W_lin, both, X, Y, gs.get("sigma", 6.0),
                                                              gs.get("match_hue"), gs.get("flat", 0.08)),
                     "white_level": lambda: g / np.median(g[both]),
                     "none": lambda: np.ones(S_lin.shape[:2])}[gs.get("falloff", "luma_ratio")]()
            white = box_lin(scene_u8, gs["white_box"])[None, None, :] * shape[..., None] * gs.get("white_scale", 1.0)
        elif gain == "white_level":
            white = g[..., None] * gr.white_chroma(S_lin, inner, g)[None, None, :]
        else:
            sys.exit("grade.colour 'reference' needs grade.white_box, or gain 'white_level' on a product with real white areas")
        black = box_lin(scene_u8, gs["black_box"]) if gs.get("black_box") else \
            np.minimum(np.percentile(S_lin[both], 1, axis=0), gs.get("black_cap", 0.02))
        ref_white = np.ones(3) if ref_flat else box_lin(ref_u8, gs["ref_white_box"]) if gs.get("ref_white_box") else np.ones(3)
        graded = gr.keep_reference_colour(W_lin, white, black, ref_white)
        rep["grade"].update({"colour": "reference", "white_point_lin": [round(float(v), 4) for v in np.median(white[both], 0)],
                             "black_point_lin": [round(float(v), 4) for v in black],
                             "ref_white_lin": [round(float(v), 4) for v in ref_white]})
    else:
        curves = gr.tone_curves(S_lin, W_lin, g, both)
        graded = gr.apply_grade(W_lin, g, curves)
        rep["grade"].update({"colour": "scene", "curves_rgb": [{k: v for k, v in c.items() if k != "_p"} for c in curves]})
        pw = gs.get("protect_white")
        if pw:
            # near-clipped packshot whites carry none of its light falloff: never dim them below the median light
            lo, hi = pw if isinstance(pw, (list, tuple)) else (0.88, 0.98)
            w = gr.clip_weight(W_lin, lo, hi)
            gm = float(np.median(g[both]))
            graded = graded * (1.0 + w * (np.maximum(g, gm) / g - 1.0))[..., None]
            rep["grade"]["protect_white"] = {"range": [lo, hi], "px": int((w[both] > 0.5).sum())}

    sh = gs.get("shading")
    if sh:
        # residual shading across the product (cylinders): keeps the generated can's round light, not its print
        centers, prof = gr.shading_profile(S_lin, graded, U, both, sh.get("bins", 32), sh.get("smooth", 1.0), sh.get("match_hue"))
        graded = gr.apply_profile(graded, U, centers, prof, sh.get("strength", 1.0), tuple(sh.get("clip", (0.4, 2.5))))
        rep["grade"]["shading_profile"] = {"range": [round(float(np.exp(prof.min())), 3), round(float(np.exp(prof.max())), 3)],
                                           "bins": len(prof)}
        if sh.get("x_range") and (sh.get("limb") or sh.get("spec") or sh.get("shadow")):
            graded = gr.cylinder_light(graded, U, sh["x_range"], sh.get("limb", 0.0), sh.get("spec"), sh.get("white", 0.9), sh.get("shadow"))

    rl = gs.get("relight")
    if rl:
        # the source's light the smooth grade missed: cast shadows, falloff, glows (+ optional glossy sheen)
        if rl.get("debug"):
            np.savez_compressed(work / "relight_debug.npz", S_lin=S_lin, graded=graded, both=both, W_lin=W_lin, a_new=a_new)
        R, SH, rinfo = gr.relight_map(S_lin, graded, both, W_lin=W_lin,
                                      **{k: v for k, v in rl.items() if k not in ("about", "strength", "clip", "debug")})
        fmul = np.clip(np.exp(R), *tuple(rl.get("clip", (0.25, 3.0))))
        # veil (white, equal on all channels): lit gloss is part of the material's look and takes the light;
        # unlit haze is a reflection and goes on top
        if SH is not None and rinfo.get("veil") == "unlit":
            relit = graded * fmul[..., None] + SH[..., None]
        else:
            relit = (graded + (SH[..., None] if SH is not None else 0.0)) * fmul[..., None]
        st = float(rl.get("strength", 1.0))
        graded = graded + st * (relit - graded)
        rep["grade"]["relight"] = dict(rinfo, range=[round(float(fmul[both].min()), 3), round(float(fmul[both].max()), 3)],
                                       sheen_max=round(float(SH[both].max()), 3) if SH is not None else None)
        qa.relight_sheet(crop, fmul, SH, both).save(qad / "relight_map.png")

    def med(img):
        return [int(v) for v in np.round(np.median(im.lin_to_srgb(img[both]), 0) * 255)]

    rep["grade"]["median_srgb_overlap"] = {"scene": med(S_lin), "graded_reference": med(graded)}
    prev = im.lin_to_srgb(graded) * a_new[..., None] + crop * (1 - a_new[..., None])
    qa.before_after(scene_u8[y0:y1, x0:x1], im.to_u8(prev)).save(qad / "grade_preview.png")
    if a.stage == "grade":
        save_report()
        print(json.dumps(rep["grade"], indent=1))
        print(f"qa         {qad / 'grade_preview.png'}")
        return

    # 3. finish: highlight roll-off, softness and grain matched to the frame
    fs = job.get("finish", {})
    if fs.get("edge") == "bleed":
        # the product's own colour runs on past its edge, so the blur and the sharpen can't pull the packshot's
        # backdrop into it (a light or dark rim along the outline)
        from scipy.ndimage import distance_transform_edt
        full = a_new >= 0.98
        if full.any() and (~full).any():
            iy, ix = distance_transform_edt(~full, return_distances=False, return_indices=True)
            graded = graded[iy, ix]
    card = im.lin_to_srgb(gr.rolloff(graded, fs.get("rolloff", 0.85)))
    Ls = im.lum(crop)
    probe = fs.get("blur_probe")
    pb = (probe[0] - x0, probe[1] - y0, probe[2] - x0, probe[3] - y0) if probe else fn.auto_probe(Ls, both)
    s_scene, s_card = fn.edge_sigma(Ls, pb), fn.edge_sigma(im.lum(card), pb)
    sharp = None
    if fs.get("sharpen") == "auto" and np.isfinite(s_scene + s_card) and s_card > 1.05 * s_scene:
        # a packshot softer than the frame (a small screenshot enlarged): unsharp mask to the scene's edge blur
        card, r_sh, amt = fn.sharpen_to(card, pb, s_scene, im.lum, a_new)
        sharp = {"radius": round(r_sh, 2), "amount": round(amt, 3), "edge_sigma_before": round(s_card, 3)}
        s_card = fn.edge_sigma(im.lum(card), pb)
    auto_blur = float(np.sqrt(max(s_scene ** 2 - s_card ** 2, 0.0))) if np.isfinite(s_scene + s_card) else 0.0
    blur = auto_blur if fs.get("blur", "auto") == "auto" else float(fs["blur"])
    card = fn.blur_rgb(card, blur)
    grain = fs.get("grain", "auto")
    if grain == "auto":
        flat = fn.flat_mask(Ls, both)
        add = np.sqrt(np.clip(fn.mad_std(fn.highpass(crop)[flat]) ** 2 - fn.mad_std(fn.highpass(card)[flat]) ** 2, 0, None))
    else:
        add = np.full(3, float(grain))
    if add.max() > 0.002:
        card = card + fn.make_grain(card.shape[:2], add, a_new > 0.5, seed=11)
    rep["finish"] = {"blur_px": round(blur, 3), "blur_auto_estimate": round(auto_blur, 3),
                     "edge_sigma_scene": round(s_scene, 3), "edge_sigma_reference": round(s_card, 3),
                     "blur_probe": [pb[0] + x0, pb[1] + y0, pb[2] + x0, pb[3] + y0],
                     "grain_added": [round(float(v), 4) for v in add]}
    if sharp:
        rep["finish"]["sharpen"] = sharp

    # 4. fill the background where the generated product showed and the real one doesn't
    fls = job.get("fill", {})
    dil = int(fls.get("dilate", 2))
    old_dil = binary_dilation(old, iterations=dil) if dil > 0 else old.copy()   # scipy: iterations 0 = until it fills everything
    visb = vis > 0.5
    mt = job.get("matte")
    ink_mode = mt == "ink" or (isinstance(mt, dict) and mt.get("type") == "ink")
    F = old_dil & visb & ((a_new < 0.98) | ink_mode)    # an ink matte covers only its ink: clear all generated print
    if gen_cov is not None:
        # the generated shape: fill only where the bled label can't reach. Its edge pixels share the new label's edge
        # (both are the generated outline's coverage), so the scene's own blend stays under them: a fill there would
        # show its background and grain through the anti-aliasing as specks
        F = (gen_cov > 0.5) & visb & (np.clip(2 * a_bleed, 0, 1) < 0.98)
    D = binary_dilation(F) & ~old_dil & visb
    # specks boxed in between the product and an occluder (a few px) take their colour from whatever borders them
    orphans = fl.anchor_orphans(F, D, int(fls.get("orphan_px", 16)))
    if orphans:
        D = D | orphans["ring"]
    try:
        base, nfill = fl.harmonic_fill(crop, F, D, offset=(x0, y0))
    except fl.FillError as e:
        rep["fill"] = {"error": str(e)}
        save_report()
        sys.exit(f"FILL STOPPED, leftover regions with no background around them: {e}\n"
                 "Fix: extend reference_outline under the occluder ('extend') or correct the occluder polyline, then re-run.")
    ring = binary_dilation(D, iterations=3) & ~old_dil & visb
    gstd = fn.mad_std(fn.highpass(crop)[ring]) if ring.any() else np.zeros(3)
    if nfill:
        base[F] = np.clip(base[F] + fn.make_grain(F.shape, gstd, F, seed=7)[F], 0, 1)
    rep["fill"] = {"pixels": int(nfill), "grain_std": [round(float(v), 4) for v in gstd]}
    if orphans:
        rep["fill"]["orphans"] = {"regions": orphans["n"], "px": orphans["px"]}
    if gen_cov is not None:
        # the outline's anti-aliased pixels hold generated paper already: under the new edge they show the background
        # nearby (its local mean, no grain) so the new paper isn't counted twice (a light rim)
        from scipy.ndimage import uniform_filter
        a_sh = np.clip(gaussian_filter(gen_cov, fs.get("edge_softness", 0.45)), 0, 1)
        bgpx = (gen_cov <= 0.02) & (vis > 0.99)
        wsum = uniform_filter(bgpx.astype(float), 5)
        bmean = np.stack([uniform_filter(base[..., c] * bgpx, 5) for c in range(3)], -1) / np.maximum(wsum, 1e-6)[..., None]
        rim = (gen_cov > 0.02) & (a_sh < 0.98) & (vis > 0.99) & ~F & (wsum > 0.04)
        base[rim] = bmean[rim]
        rep["fill"]["edge_px"] = int(rim.sum())
    if a_body is not None:
        # a body region is part of the label: it ends where the label (its visible shape) ends
        a_body = a_body * (np.clip(gaussian_filter(gen_cov, fs.get("edge_softness", 0.45)), 0, 1) if gen_cov is not None
                           else np.clip(gaussian_filter(a_new, fs.get("edge_softness", 0.45)), 0, 1))
    if a_body is not None and (a_body > 0.02).any():
        # where the real product shows its body: the scene's body colour (the generated image drew it close by)
        paper_l = float(np.percentile(im.lum(crop)[inner], 90)) if inner.any() else 1.0
        sig = float(fls.get("body_sigma", 20))
        B, Mb = gr.body_field(crop, a_body > 0.5, paper_l, sig)
        if B is None:
            save_report()
            sys.exit("reference_outline.body: no pixels of the product's body colour near the body region; check it")
        bstd = fn.mad_std(fn.highpass(crop)[Mb]) if Mb.sum() > 50 else np.zeros(3)
        B = np.clip(B + fn.make_grain(B.shape[:2], bstd, a_body > 0.02, seed=5), 0, 1)
        base = base * (1 - a_body[..., None]) + B * a_body[..., None]
        rep["fill"]["body"] = {"px": int((a_body > 0.5).sum()), "sigma": sig,
                               "colour_srgb": [int(v) for v in np.round(np.median(B[a_body > 0.5], 0) * 255)]}

    # 5. composite behind the occluders and save next to the original
    if gen_cov is not None:
        alpha = np.clip(gaussian_filter(gen_cov, fs.get("edge_softness", 0.45)), 0, 1) * np.clip(2 * a_bleed, 0, 1) * vis
    else:
        alpha = np.clip(gaussian_filter(a_new, fs.get("edge_softness", 0.45)), 0, 1) * vis
    if a_body is not None:
        alpha = alpha * (1.0 - a_body)
    if ink_mode:
        # print only: the packshot's ink as coverage in the ink's colour, over the scene's own surface (the fill has
        # taken the generated print away), so a patch of packshot background never shows its edge
        if gs.get("colour") != "reference" or not ref_flat:
            sys.exit("matte 'ink' needs grade.colour 'reference' and grade.ref_flatten")
        mcfg = mt if isinstance(mt, dict) else {}
        Lw = im.lum(W_lin)
        ink_level = float(mcfg.get("ink", np.percentile(Lw[a_new > 0.5], 0.5)))
        ink_lin = gr.keep_reference_colour(np.full_like(W_lin, ink_level), white, black, ref_white)
        wL, kL = im.lum(white), im.lum(ink_lin)
        cov = (wL - im.lum(im.srgb_to_lin(np.clip(card, 0, 1)))) / np.maximum(wL - kL, 1e-4)
        t0 = float(mcfg.get("floor", 0.06))                  # paper grain and grain added stay out of the matte
        cov = np.clip((cov - t0) / (1 - t0), 0, 1)
        card = im.lin_to_srgb(ink_lin)
        alpha = alpha * cov
        rep["matte"] = {"type": "ink", "ink_level": round(ink_level, 4), "coverage_px": int((cov * (a_new > 0.5)).sum())}
    out = base * (1 - alpha[..., None]) + card * alpha[..., None]
    if not np.isfinite(out).all():
        save_report()
        sys.exit("composite contains NaN or inf: nothing saved")
    res = scene_u8.copy()
    res[y0:y1, x0:x1] = im.to_u8(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    im.save_u8(out_p, res, scene_a, info)
    if a.psd or job.get("psd"):
        rep["psd"] = psd.write_refined(out_p.with_suffix(".psd"), scene_u8, scene_a, info, res, base, F, card, alpha, roi)

    # 6. checks and QA sheets
    from imgpost import audit as au
    squeeze = au.squeeze_info(H, refmask, ref_u8.shape, ref_lin)
    cut = None
    if gen_cov is not None:
        # the real print falling outside the generated shape is cut off: the generated label is too small there
        Lw = im.lum(W_lin)
        cut = (a_new > 0.5) & (gen_cov < 0.5) & (vis > 0.5) & (Lw < 0.45 * np.percentile(Lw[a_new > 0.5], 90))
    findings, asheet = au.run(scene_u8[y0:y1, x0:x1], res[y0:y1, x0:x1], alpha, old, vis, F, (x0, y0),
                              rep.get("align"), squeeze, cut, ink=ink_mode, shape=gen_cov is not None)
    rep["audit"] = findings
    if asheet:
        asheet.save(qad / "audit.png")
    chg = np.abs(res.astype(np.int16) - scene_u8.astype(np.int16)).max(2) > 0
    inside = np.zeros_like(chg)
    inside[y0:y1, x0:x1] = True
    ys, xs = np.nonzero(chg)
    rep["change"] = {"changed_px": int(chg.sum()), "pct_of_frame": round(100 * float(chg.mean()), 3),
                     "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
                     "outside_roi_changed": int((chg & ~inside).sum())}
    before, after = scene_u8[y0:y1, x0:x1], res[y0:y1, x0:x1]
    qa.before_after(before, after).save(qad / "before_after.png")
    rx0, ry0, rx1, ry1 = refmask.bbox(ref_u8.shape, pad=10)
    qa.compare_sheet(before, after, ref_u8[ry0:ry1, rx0:rx1]).save(qad / "compare.png")
    tiles = qa.edge_tiles(before, after, ((alpha > 0.03) & (alpha < 0.97)) | F, (x0, y0))
    if tiles:
        tiles.save(qad / "edges.png")
    save_report()

    c, f = rep["change"], rep["finish"]
    print(f"saved      {out_p}")
    print(f"changed    {c['changed_px']:,} px ({c['pct_of_frame']}% of frame), bbox {c['bbox']}, outside work box: {c['outside_roi_changed']}")
    print(f"grade      gain {gain} {rep['grade']['gain_range']}, median sRGB scene {rep['grade']['median_srgb_overlap']['scene']}"
          f" vs graded {rep['grade']['median_srgb_overlap']['graded_reference']}")
    print(f"finish     blur {f['blur_px']} px (auto estimate {f['blur_auto_estimate']}), grain added {f['grain_added']}" +
          (f", sharpened (radius {f['sharpen']['radius']}, amount {f['sharpen']['amount']}: edge blur "
           f"{f['sharpen']['edge_sigma_before']} -> {f['edge_sigma_reference']} px, scene {f['edge_sigma_scene']})" if "sharpen" in f else ""))
    print(f"fill       {nfill} px, grain std {rep['fill']['grain_std']}" +
          (f"; {orphans['n']} speck(s), {orphans['px']} px, boxed in by an occluder: filled from their border" if orphans else ""))
    for f in findings:
        if f["level"] != "INFO":
            print(f"audit      {f['level']:4s} {f['check']}: {f['text']}" + (f" at {f['box']}" if f.get("box") else ""))
    worst = "FAIL" if any(f["level"] == "FAIL" for f in findings) else "WARN" if any(f["level"] == "WARN" for f in findings) else "clean"
    print(f"audit      {worst}" + (f": look at {qad / 'audit.png'} before delivering" if worst != "clean" else "") +
          (f" ({sum(f['level'] == 'INFO' for f in findings)} info in report.json)" if any(f["level"] == "INFO" for f in findings) else ""))
    if "psd" in rep:
        p, v = rep["psd"], rep["psd"]["layers_vs_png"]
        print(f"psd        {p['path']}  {p['mb']} MB, layers {', '.join(l['name'] for l in p['layers'])} (+ mask);"
              f" layers vs PNG: max {v['max_levels']} level(s) on {v['px_differ']} px")
    print(f"qa         {qad}/compare.png  edges.png  before_after.png  align_overlay.png  grade_preview.png")


def _load_mask(p, hw, name):
    from PIL import Image
    m = np.asarray(Image.open(p).convert("L")) > 127
    if m.shape != tuple(hw):
        sys.exit(f"{name} {p} is {m.shape[1]}x{m.shape[0]}: expected the scene's size ({hw[1]}x{hw[0]})")
    return m


def _run_on_body(a, job, rep, save_report, qad, out_p, scene_u8, scene_a, info, ref_u8, ref_lin, H, refmask, roi, old,
                 vis, pre):
    """Steps 2-6 for print straight on a coloured body (matte ink, polarity light or dark; imgpost/ink.py): the
    packshot's print as coverage against its body light, the frame's garbled print filled from its body, the true
    print laid on in the frame's own print colour. old is the generated product's matte (old_silhouette as a mask)."""
    from imgpost import audit as au
    from imgpost import ink as ik
    x0, y0, x1, y1 = roi
    cfg = ik.config(job["matte"])
    pol = cfg["polarity"]
    crop_u8 = scene_u8[y0:y1, x0:x1]
    try:
        bow = job["align"].get("bow") or {}
        prep = ik.prepare(ref_u8, refmask, cfg, H, roi, crop_u8, ss=job.get("supersample", 4), prefilter=pre,
                          body={k: bow[k] for k in ("centre", "axis", "half_widths")} if bow.get("half_widths") else None,
                          bow_auto=bow.get("from") == "matte")
        cov, in_band, crep = prep["cover"], prep["band"], prep["report"]
        filled, inkd, frep = ik.garbled(crop_u8, old, cov, in_band, cfg, vis, prep["box"], prep["inks"])
    except ik.InkError as e:
        save_report()
        sys.exit(str(e))
    out_box, alpha = ik.composite(filled, crop_u8, inkd)
    rep["matte"] = dict({"type": "ink", "polarity": pol}, **crep, **frep)
    if a.stage == "grade":
        qa.before_after(crop_u8, out_box).save(qad / "grade_preview.png")
        save_report()
        print(json.dumps(rep["matte"], indent=1))
        print(f"qa         {qad / 'grade_preview.png'}")
        return
    res = scene_u8.copy()
    res[y0:y1, x0:x1] = out_box
    out_p.parent.mkdir(parents=True, exist_ok=True)
    im.save_u8(out_p, res, scene_a, info)
    F = inkd["fill"]
    base, card, alpha_psd = ik.layers(filled, out_box, inkd)
    if a.psd or job.get("psd"):
        rep["psd"] = psd.write_refined(out_p.with_suffix(".psd"), scene_u8, scene_a, info, res, base, F, card, alpha_psd, roi)

    # checks: the audit (ink: the print lies on the product's own surface) plus where the keyframe changed pixels
    findings, asheet = au.run(crop_u8, out_box, alpha, old, vis, F, (x0, y0), rep.get("align"),
                              au.squeeze_info(H, refmask, ref_u8.shape, ref_lin), None, ink=True, shape=False)
    for f in findings:
        if f["check"] == "fill":            # the garbled print is as thick as the generated letters, under the new print
            f.update(level="INFO", text=f["text"] + " (the garbled print, under the new print: see before_after)")
        elif f["check"] == "edges":         # no outline is composited: the lines only place the print
            f.update(level="INFO", text=f["text"] + f" (matte ink {pol}: no outline shows; check the print's placement)")
        elif f["check"] == "leftover" and pol == "dark":   # the new edge is a letter's, and the body beyond it is the
            f.update(level="INFO", text=f["text"] + " (matte ink dark: the body round the print is the generated one)")
    findings += ik.check(crop_u8, out_box, inkd, cov, cfg, (x0, y0))
    findings += ik.soft_check(crep, cov, prep["ref_sum"], alpha, (x0, y0))
    findings += ik.dark_checks(crop_u8, out_box, inkd, (x0, y0))
    rep["audit"] = findings
    if findings:
        asheet = au.sheet(crop_u8, out_box, findings, (x0, y0))
    if asheet:
        asheet.save(qad / "audit.png")
    chg = (res != scene_u8).any(-1)
    ys, xs = np.nonzero(chg)
    rep["change"] = {"changed_px": int(chg.sum()), "pct_of_frame": round(100 * float(chg.mean()), 3),
                     "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
                     "outside_roi_changed": 0}
    rep["outside_matte_changed"] = int((chg[y0:y1, x0:x1] & ~old).sum())
    qa.before_after(crop_u8, out_box).save(qad / "before_after.png")
    rx0, ry0, rx1, ry1 = refmask.bbox(ref_u8.shape, pad=10)
    qa.compare_sheet(crop_u8, out_box, ref_u8[ry0:ry1, rx0:rx1]).save(qad / "compare.png")
    tiles = qa.edge_tiles(crop_u8, out_box, F | ((alpha > 0.03) & (alpha < 0.97)), (x0, y0))
    if tiles:
        tiles.save(qad / "edges.png")
    save_report()
    m, c = rep["matte"], rep["change"]
    print(f"saved      {out_p}")
    print(f"changed    {c['changed_px']:,} px ({c['pct_of_frame']}% of frame), bbox {c['bbox']}, outside the matte: "
          f"{rep['outside_matte_changed']}")
    rg = m.get("register")
    if rg:
        print(f"register   " + (f"the print moved {rg['shift_px']} px, scale {rg['scale_xy']}, {rg['rotation_deg']} deg" +
                                (f", bow {rg['bow_px'][0]:+} to {rg['bow_px'][1]:+} px (top to bottom)" if "bow_px" in rg else "") + " onto the frame's print "
                                f"(correlation {rg['ncc_before']} -> {rg['ncc_after']})" if rg["kept"]
                                else f"not applied: {rg['why']}"))
    print(f"ink        {pol}: packshot {'closing' if pol == 'dark' else 'opening'} {m['open']} px, {m['print_px']:,} "
          f"print px; garbled print {m['garbled_px']:,} px filled over {m['filled_px']:,} px" +
          (f" (frame closing {m['frame_open_auto']['size']} px)" if "frame_open_auto" in m else "") +
          f"; ink = body x {m['ink_k']}")
    for f in findings:
        if f["level"] != "INFO":
            print(f"audit      {f['level']:4s} {f['check']}: {f['text']}" + (f" at {f['box']}" if f.get("box") else ""))
    worst = "FAIL" if any(f["level"] == "FAIL" for f in findings) else "WARN" if any(f["level"] == "WARN" for f in findings) else "clean"
    print(f"audit      {worst}" + (f": look at {qad / 'audit.png'} before delivering" if worst != "clean" else ""))
    if "psd" in rep:
        p, v = rep["psd"], rep["psd"]["layers_vs_png"]
        print(f"psd        {p['path']}  {p['mb']} MB; layers vs PNG: max {v['max_levels']} level(s) on {v['px_differ']} px")
    print(f"qa         {qad}/compare.png  edges.png  before_after.png  align_overlay.png")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "draft":
        ap = argparse.ArgumentParser(prog="run.py draft")
        ap.add_argument("cmd")
        ap.add_argument("scene")
        ap.add_argument("reference")
        ap.add_argument("--quad", nargs="+", help="TL TR BR BL of the packshot outline's bounding box in the scene, X,Y each")
        ap.add_argument("--pairs", nargs="*", default=[], help="rough landmark pairs RX,RY=SX,SY")
        ap.add_argument("--occluder", action="append", default=[], help="SIDE:X,Y X,Y ... or polygon:X,Y X,Y ...")
        ap.add_argument("--suggest", action="store_true", help="also add packshot corners that match the scene well")
        ap.add_argument("--mask", help="the product's mask in the scene (measure.py mask): fit the outline to its contour")
        ap.add_argument("--mask-skip", help="\"X,Y X,Y ...\": part of the outline (reference px) not to fit to the mask, "
                        "e.g. an edge where the mask runs on into a part the job keeps")
        ap.add_argument("--model", choices=("homography", "affine"), default="homography")
        ap.add_argument("--outline", help="the product on a packshot that isn't on a plain background, or shows "
                        "other items: X0,Y0,X1,Y1 (a box) or \"X,Y X,Y X,Y ...\" (a polygon)")
        ap.add_argument("--cylinder", action="store_true", help="cans and bottles: shading template instead of relight")
        ap.add_argument("--no-relight", action="store_true")
        ap.add_argument("--label", choices=("cylinder", "flat"), help="the label only: on a bottle or jar (cylinder), "
                        "or on a flat face (a tube's panels); --quad is then the generated label's 4 corners")
        ap.add_argument("--ink", choices=("light", "dark"), help="print straight on a coloured body, lighter or darker "
                        "than it: matte ink with that polarity (the packshot's print on the frame's own body); with "
                        "--mask, the mask is the matte it stays inside")
        ap.add_argument("--print-box", help="X0,Y0,X1,Y1: the print's rows on the packshot, inside its frosted rims, "
                        "seams and lid edge (--ink)")
        ap.add_argument("--bow", help="a round body seen from above or below (needs --mask): \"t:b ...\" given, or "
                        "seams to measure: t, t/w (a ring of half-width w), t/matte (the matte's end); t along the "
                        "product's axis from its mask's centre, e.g. \"-150:0 146.5\"")
        ap.add_argument("--job-dir")
        ap.add_argument("--force", action="store_true")
        cmd_draft(ap.parse_args())
        return
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        ap = argparse.ArgumentParser(prog="run.py init")
        ap.add_argument("cmd")
        ap.add_argument("scene")
        ap.add_argument("reference")
        ap.add_argument("--job-dir")
        cmd_init(ap.parse_args())
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job")
    ap.add_argument("--stage", choices=("align", "grade", "all"), default="all")
    ap.add_argument("--job-dir", help="where work/, qa/ and report.json go (default: the job file's folder)")
    ap.add_argument("--output", help="override job.output")
    ap.add_argument("--scene", help="override job.scene: chain products by passing the previous job's output")
    ap.add_argument("--psd", action="store_true", help="also write the output as a layered PSD next to it (same as \"psd\": true)")
    cmd_run(ap.parse_args())


if __name__ == "__main__":
    main()
