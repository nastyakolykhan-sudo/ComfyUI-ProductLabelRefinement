#!/usr/bin/env python3
"""The one workflow (workflows/product-label-refinement-all.api.json) against the separate templates on a live
ComfyUI server: each case run both ways with the same inputs must give the same outputs, 0 px apart, with the same
reports.

  python -P tools/regress_all.py --server http://127.0.0.1:8188 [--video] [--keep DIR]

Cases (the pack's demos):
  ink-draft   the ink demo from its frame, mask and packshot     = product-label-refinement-ink-draft
  ink-job     the ink demo's measured job                        = product-label-refinement-ink
  paper-draft the stills demo from its rough input (PSD too)     = product-label-refinement-draft
  paper-job   the stills demo's measured job (PSD too)           = product-label-refinement
  video       (--video, minutes on a CPU) the video demo's clip, mask video and keyframe = product-label-refinement-video
The one workflow is given the product mask in every case (an empty one for the paper demo, with Auto Draft's mask fit
off), so SAM 3 doesn't run; its triage sheet has no template to match and isn't compared. Runs go through tools/comfy_run.py. Needs numpy and Pillow, and PyAV for --video (a
ComfyUI Python has them all). Refine (Video) keeps its tracks by its inputs, so the second video run re-renders only.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

PACK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACK / "tools"))
from build_workflows import DRAFT_INPUT, INK_DRAFT_INPUT  # noqa: E402

WF = PACK / "workflows"
ALL = WF / "product-label-refinement-all.api.json"
VOLATILE = {"job", "path", "_source", "seconds", "work", "qa_plot", "log"}


def cases(empty_mask):
    ink, demo, vid = PACK / "examples" / "ink_demo", PACK / "examples" / "demo", PACK / "examples" / "video_demo"
    ink_in = ["--reference", ink / "inkdemo_packshot.png", "--mask", ink / "inkdemo_mask.png"]
    paper_off = ["--mask", empty_mask, "--set", "8.value=false", "--psd"]
    paper_draft = ["--set", "24.ink=off", "--set", "24.model=homography", "--set", "24.light=relight",
                   "--set", f"24.quad={DRAFT_INPUT['quad']}", "--set", f"24.pairs={DRAFT_INPUT['pairs']}",
                   "--set", f"24.occluders={DRAFT_INPUT['occluders']}", "--set", "24.print_box=", "--set", "24.bow="]
    assert INK_DRAFT_INPUT["print_box"] and INK_DRAFT_INPUT["bow"]    # the one workflow holds the ink demo's
    return {
        "ink-draft": (["--workflow", WF / "product-label-refinement-ink-draft.api.json", "--scene", ink / "inkdemo_frame.png", *ink_in],
                      ["--source", ink / "inkdemo_frame.png", *ink_in]),
        "ink-job": (["--workflow", WF / "product-label-refinement-ink.api.json", "--scene", ink / "inkdemo_frame.png", *ink_in,
                     "--job", ink / "job.json"],
                    ["--source", ink / "inkdemo_frame.png", *ink_in, "--job", ink / "job.json"]),
        "paper-draft": (["--workflow", WF / "product-label-refinement-draft.api.json", "--scene", demo / "imagepost_demo_scene.png",
                         "--reference", demo / "imagepost_demo_packshot.png", "--psd"],
                        ["--source", demo / "imagepost_demo_scene.png", "--reference", demo / "imagepost_demo_packshot.png",
                         *paper_off, *paper_draft]),
        "paper-job": (["--workflow", WF / "product-label-refinement.api.json", "--scene", demo / "imagepost_demo_scene.png",
                       "--reference", demo / "imagepost_demo_packshot.png", "--job", demo / "job.json", "--psd"],
                      ["--source", demo / "imagepost_demo_scene.png", "--reference", demo / "imagepost_demo_packshot.png",
                       *paper_off, "--job", demo / "job.json"]),
        "video": (["--workflow", WF / "product-label-refinement-video.api.json", "--image", f"1={vid / 'labelvideo_demo_clip.mov'}",
                   "--image", f"3={vid / 'labelvideo_demo_mask.mov'}", "--image", f"6={vid / 'labelvideo_demo_keyframe.png'}"],
                  ["--source", vid / "labelvideo_demo_clip.mov", "--mask", vid / "labelvideo_demo_mask.mov",
                   "--keyframe", vid / "labelvideo_demo_keyframe.png", "--set", "6.value=-1"]),
    }


def run(server, args, out, timeout):
    cmd = [sys.executable, "-P", str(PACK / "tools" / "comfy_run.py"), "--server", server, "--out", str(out),
           "--timeout", str(timeout), *map(str, args)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"comfy_run failed ({p.returncode}):\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")


def outputs(root):
    """-> {kind: path}: the result image, report(s), draft job, QA images by name, PSD, lossless master."""
    out = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.name in ("history.json", "prompt.json"):
            continue
        rel = p.relative_to(root)
        stem = re.sub(r"_\d{5}_?$", "", p.stem)
        if rel.parts[0] != "product-label-refinement" and p.suffix != ".json" or "triage" in p.stem or "triage" in rel.parts:
            continue                                  # inputs echoed back by the loaders' previews; the one workflow's triage
        if p.suffix == ".json" and rel.parts[0] != "product-label-refinement":
            out[p.name] = p                           # report.json, video_report.json
        elif "qa" in rel.parts[:-1]:
            out[f"qa/{stem}{p.suffix}"] = p
        elif "video" in rel.parts[:-1]:
            if p.suffix == ".mov":
                out["master.mov"] = p                 # (the QA plot: the template only previews it; the mp4: H.264)
        elif p.suffix == ".json":
            out["draft.json"] = p
        elif p.suffix in (".png", ".psd"):
            out[f"result{p.suffix}"] = p              # the refined image (named after the scene in the templates)
    return out


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def mov_frames(p):
    import av
    with av.open(str(p)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def compare(a, b):
    """-> list of (name, verdict)"""
    rows = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            rows.append((k, f"only in {'the template' if k in a else 'the one workflow'}"))
        elif k.endswith(".png"):
            x, y = np.asarray(Image.open(a[k])), np.asarray(Image.open(b[k]))
            d = int(np.abs(x.astype(int) - y.astype(int)).max()) if x.shape == y.shape else -1
            rows.append((k, "same" if d == 0 else f"DIFF (max {d})" if d > 0 else "DIFF (size)"))
        elif k.endswith(".json"):
            same = clean(json.loads(a[k].read_text())) == clean(json.loads(b[k].read_text()))
            rows.append((k, "same" if same else "DIFF"))
        elif k.endswith(".mov"):
            x, y = mov_frames(a[k]), mov_frames(b[k])
            d = max((int(np.abs(f.astype(int) - g.astype(int)).max()) for f, g in zip(x, y)), default=0)
            rows.append((k, f"same ({len(x)} frames)" if len(x) == len(y) and d == 0 else f"DIFF ({len(x)} vs {len(y)}, max {d})"))
        else:
            rows.append((k, "same" if a[k].read_bytes() == b[k].read_bytes() else "DIFF (bytes)"))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--video", action="store_true", help="also the video demo (minutes on a CPU)")
    ap.add_argument("--only", nargs="+", help="these cases only")
    ap.add_argument("--keep", help="keep the runs here")
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()
    work = Path(a.keep).expanduser() if a.keep else Path(tempfile.mkdtemp(prefix="plr-regress-all-"))
    work.mkdir(parents=True, exist_ok=True)
    scene = Image.open(PACK / "examples" / "demo" / "imagepost_demo_scene.png")
    empty = work / "empty_mask.png"
    Image.new("L", scene.size, 0).save(empty)
    names = a.only or [c for c in cases(empty) if c != "video" or a.video]
    failed = 0
    for name in names:
        tpl, one = cases(empty)[name]
        run(a.server, tpl, work / name / "template", a.timeout)
        run(a.server, ["--workflow", ALL, *one], work / name / "all", a.timeout)
        rows = compare(outputs(work / name / "template"), outputs(work / name / "all"))
        bad = [r for r in rows if not r[1].startswith("same")]
        failed += bool(bad)
        print(f"{name:12s} {'PASS' if not bad else 'FAIL'}: {len(rows)} outputs compared")
        for k, v in rows:
            print(f"    {k:28s} {v}")
    print(f"runs in {work}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
