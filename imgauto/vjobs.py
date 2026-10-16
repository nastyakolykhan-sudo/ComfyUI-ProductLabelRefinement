"""vjob.json for each shot, with the parameters a person used to type measured from the clip and the keyframe.

The defaults are the ones the review rounds converged on (pitfalls V18-V28): print-only tracks, affine tube camera
with the mesh, shear_max 0.9, limb snap and edge feather on tubes, no sparkle re-add (V19, V26). Frames a sparkle
trail crosses (spark_glow, V22) come from the refix loop's QA flags: measured up front, print and background
glitter swamp the trail (white print reads as sparkle, and SAM cuts the trail out of the matte).
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

TRACK = {"on": "print", "print_thresh": 0.06, "points": 900, "erode": 4, "crop": "auto", "margin": 40}
FIT = {"wmin": 0.5, "reject": 2.5, "half": 3, "grid": 7, "camera": "affine", "min_inliers": 0.5, "max_rms": 6.0,
       "max_rigid": 12.0, "shear_max": 0.9, "residual": {"sigma": "auto", "lam": 1.0}}
RENDER = {"blur": "auto", "shutter": "auto", "inset": 1.5, "supersample": 4, "print_thresh": 0.03,
          "limb_snap": {"step": 24, "max_px": 30}, "feather": {"theta_deg": 10, "v_px": 24}}
QA = {"step": 2.0, "ncc_min": 0.6, "ncc_drop": 0.08, "lock_min": 0.3, "unseen_max": 0.02}


def track_res(w, h):
    """CoTracker's working size for a w x h crop: about half, as [960, 512] for 1080 x 1920 (rows, columns)."""
    return [max(256, int(h * 0.5) // 8 * 8), max(256, int(w * 0.474) // 8 * 8)]


def has_holes(stats, frames, share=0.001):
    """The matte has holes inside it (something in front: a sparkle trail, a prop) on any of the frames."""
    return any(stats[t].get("holes", 0) > share for t in frames)


def build(clip, mask, shot, anchor, words, crop_size, polarity="both", tophat=9, body_width=300,
          sparks=(), holes=False, ends_at_cut=False, out_dir="work"):
    """One vjob dict. shot: [a, b) in the clip; anchor: a video.py anchor dict; crop_size: (w, h)."""
    render = dict(RENDER, polarity=polarity, print_tophat=int(tophat) | 1, ease=[2, 0 if ends_at_cut else 2])
    if sparks:
        render["spark_glow"] = sorted(int(t) for t in sparks)
    else:
        render["sparkles"] = False
    if holes:
        render["occluder_feather"] = 20
    return {
        "about": "Written by autopilot.py; every field it measured is listed in ../decisions.json.",
        "clip": str(clip), "mask": str(mask), "shot": [int(shot[0]), int(shot[1])],
        "anchor": anchor,
        "track": dict(TRACK, res=track_res(*crop_size)),
        "fit": dict(FIT),
        "render": render,
        "qa": dict(QA, rms_max=round(0.005 * body_width, 2), words=[w.lower() for w in words]),
        "output": {"refined": f"{out_dir}/shot_refined.mp4", "lossless": f"{out_dir}/shot_lossless.mov",
                   "side_by_side": f"{out_dir}/shot_side_by_side.mp4", "qa_plot": f"{out_dir}/shot_qa_plot.png"},
    }


def keyframe_anchor(keyframe, frame, band, model="tube"):
    return {"keyframe": str(keyframe), "frame": int(frame), "band": [int(band[0]), int(band[1])], "model": model,
            "theta_lim": 75}


def reacquire_anchor(texture_job, frame):
    return {"texture_job": str(texture_job), "reacquire": int(frame), "scales": [1.0]}


def chain_anchor(texture_job, job, frame):
    return {"texture_job": str(texture_job), "chain": {"job": str(job), "frame": int(frame)}}


def write(vjob, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vjob, indent=1))
    return path
