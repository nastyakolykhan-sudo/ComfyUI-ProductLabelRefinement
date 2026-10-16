#!/usr/bin/env python3
"""The video nodes against the skill: Product Label: Refine (Video), called in-process the way ComfyUI calls it (IMAGE and MASK
tensors in), against the skill's own `video.py --refine` on the same clip, mask and keyframe(s). The frames, matte and
changed mask must be identical, and the reports equal (paths and timings aside).

  python -P tools/regress_video.py --comfy COMFYUI_DIR [--clip C --mask M --keyframe K ...] [--settings S.json]
                                   [--weights alltracker.pth] [--keep DIR]

Without --clip it runs the demo (examples/video_demo, drawn by tools/make_video_demo.py). Needs a Python with torch and
ComfyUI's dependencies (the video nodes import folder_paths), ffmpeg on PATH, and the skill folder (imgpost.lock.json's).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy", required=True, help="a ComfyUI checkout (for folder_paths)")
    ap.add_argument("--clip")
    ap.add_argument("--mask")
    ap.add_argument("--keyframe", nargs="+")
    ap.add_argument("--keyframe-frame", nargs="+")
    ap.add_argument("--settings")
    ap.add_argument("--weights", default=os.environ.get("PLR_ALLTRACKER", "~/Documents/models/trackers/alltracker/alltracker.pth"))
    ap.add_argument("--keep")
    a = ap.parse_args()
    demo = PACK / "examples" / "video_demo"
    clip = Path(a.clip or demo / "labelvideo_demo_clip.mov").expanduser()
    mask = Path(a.mask or demo / "labelvideo_demo_mask.mov").expanduser()
    kfs = [Path(k).expanduser() for k in (a.keyframe or [demo / "labelvideo_demo_keyframe.png"])]
    weights = Path(a.weights).expanduser()
    lock = json.loads((PACK / "imgpost.lock.json").read_text())
    skill = Path(lock["skill"]).expanduser()
    tmp = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="labelvideo-regress-"))
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PLR_ALLTRACKER=str(weights))

    # 1. the skill on files
    settings = json.loads(Path(a.settings).read_text()) if a.settings else {}
    settings.setdefault("track", {})["backend"] = "alltracker"
    sp = tmp / "settings.json"
    sp.write_text(json.dumps(settings))
    cmd = [sys.executable, "-P", str(skill / "scripts" / "video.py"), "--refine", "--clip", str(clip), "--mask", str(mask),
           "--keyframe", *map(str, kfs), "--settings", str(sp), "--out", str(tmp / "skill"), "--workers", "1"]
    if a.keyframe_frame:
        cmd += ["--keyframe-frame", *a.keyframe_frame]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"the skill's video.py --refine failed:\n{r.stderr[-3000:]}")

    # 2. the pack, in-process, with the same decoded frames as tensors
    sys.path.insert(0, str(Path(a.comfy).expanduser()))
    import torch
    import folder_paths
    trackers = tmp / "models_trackers"
    trackers.mkdir(exist_ok=True)
    if not (trackers / "alltracker.pth").exists():
        os.symlink(weights, trackers / "alltracker.pth")
    folder_paths.add_model_folder_path("trackers", str(trackers))
    folder_paths.set_temp_directory(str(tmp / "comfy_temp"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("productlabel_video_nodes", PACK / "video_nodes.py")
    nodes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nodes)
    sys.path.insert(0, str(PACK))
    from imgpost import images as im
    from imgvideo import vio
    info = vio.probe(clip)
    fr = vio.read(clip, 0, info["frames"], info)
    mk = vio.read(mask, 0, info["frames"])[..., 0] > 127
    kf = np.stack([im.load_u8(k)[0] for k in kfs])
    frames_t = torch.from_numpy(fr.astype(np.float32) / 255.0)
    node = nodes.ProductLabelVideoRefine()
    kfr = ", ".join(a.keyframe_frame) if a.keyframe_frame else "auto"
    out, matte, changed, report, qa = node.refine(frames_t, torch.from_numpy(mk.astype(np.float32)),
                                                  torch.from_numpy(kf.astype(np.float32) / 255.0), "alltracker.pth",
                                                  keyframe_frames=kfr, settings=json.dumps(settings), fps=info["fps"])
    pack_frames = nodes._u8(out)
    pack_matte = matte.numpy() > 0.5
    pack_changed = changed.numpy() > 0.5
    pack_report = json.loads(report)

    # 3. compare
    stem = clip.stem
    sk = tmp / "skill"
    sk_frames = vio.read(sk / f"{stem}_refined_lossless.mov", 0, info["frames"], info)
    sk_matte = vio.read(sk / f"{stem}_matte.mov", 0, info["frames"])[..., 0] > 127
    sk_changed = vio.read(sk / f"{stem}_changed.mov", 0, info["frames"])[..., 0] > 127
    sk_report = json.loads((sk / f"{stem}_report.json").read_text())
    d = np.abs(pack_frames.astype(np.int16) - sk_frames.astype(np.int16)).max(-1)
    untouched = np.abs(pack_frames.astype(np.int16) - fr.astype(np.int16)).max(-1)[~pack_changed]

    def strip(rep):
        return {k: v for k, v in rep.items() if k not in ("work", "seconds")}
    ok = d.max() == 0 and np.array_equal(pack_matte, sk_matte) and np.array_equal(pack_changed, sk_changed) and \
        strip(pack_report) == strip(sk_report) and untouched.max(initial=0) == 0
    print(f"frames: {int((d > 0).sum())} px differ (max {int(d.max())} levels); matte identical "
          f"{np.array_equal(pack_matte, sk_matte)}; changed identical {np.array_equal(pack_changed, sk_changed)}; "
          f"reports equal {strip(pack_report) == strip(sk_report)}; pixels outside 'changed' left as input: "
          f"{untouched.max(initial=0) == 0}; refined {int(pack_changed.any((1, 2)).sum())} of {info['frames']} frames")
    print("PASS" if ok else "FAIL")
    if not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
