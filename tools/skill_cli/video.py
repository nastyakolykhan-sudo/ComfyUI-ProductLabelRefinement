#!/usr/bin/env python3
"""product-label-refinement, video runner: the true label re-projected onto a product through a shot, by tracking.

  python -P video.py VJOB.json --stage track|fit|still|render|qa|merge|review|stitch|all [--range A:B] [--workers N]

--stage stitch takes a stitch job (no shot of its own): several refined keyframes on one unrolled label, for other
shots' anchor.texture_job (imgvideo/stitch.py has the vjob fields). qa/fit.json's "stops" says why the held run ends
at each side (shot edge, cut, product left, morph, misfit, fast motion), for choosing chain or re-acquire next.
  python -P video.py --refine --clip CLIP --mask MASK --keyframe KF.png [KF2.png ...] --out DIR [--settings S.json]
                     [--keyframe-frame N|auto ...]     (the ComfyUI node's contract on files: imgvideo/pipeline.py refine)

The truth is either an artwork or packshot with an ordinary stills job (run.py) measured on one frame of the shot
(anchor.job), or a corrected keyframe: one frame of the clip fixed by hand or by a model (anchor.keyframe), whose
own pixels are then the label space. CoTracker3 tracks the product from the anchor frame through the clip; every
frame gets the stills engine's own fit (imgpost.geometry.fit) to the tracks, optionally with its Residual on top (a mesh
warp for surfaces a rigid model can't follow), smoothed with zero lag; the frames the tracks hold around the anchor
(until a cut, a morph or the product leaving) are the shot that gets refined, every other frame stays as it is. The
label's print is warped into each frame (imgpost.warp), lit from the frame, given its softness and grain, and
composited inside the product's matte. --range A:B renders those frames only (a test, into work/).

VJOB.json (paths relative to its folder):
  clip            the source video
  mask            optional: the product's matte as a video (white = product, same size and frame count). Without
                  it, a colour key (key) stands in.
  shot            [start, stop) frames to consider (default: the whole clip)
  anchor          {"frame": N, "job": "anchor/job.json"}   a stills job (run.py) on clip frame N, or
                  {"keyframe": "reference.png", "frame": "auto" | N, "band": [t0, t1], "model": "homography" | "tube",
                  "theta_lim": 75}   a corrected keyframe; band is the label's stretch along the product's axis in px
                  from the matte's centre (cap, pump and crimp outside it); "tube" unrolls the keyframe onto a surface
                  of revolution (a tapered tube, a bottle), so print that spins across it stays on it; or
                  {"texture_job": "../job", "reacquire": N, "roll_search": [-4, 4, 0.25]}   another shot: the keyframe's
                  unrolled label from that job, found again on frame N of this shot by correlating the print (spin, axial
                  offset, scale; roll with roll_search, or better "roll_from_print": {"box": [u0, v0, u1, v1]}: the frame's
                  text lines level with the keyframe's); "scales": [1.0] keeps the silhouette's own scale (geometry from
                  the body, not from garbled print drawn at another size); "pitch_sag_px": s | "auto" tilts it toward the
                  camera (a re-acquired tube has no pitch: s = how far the source's text lines bow toward the base
                  mid-label beyond this job's render, measured strip by strip along the axis; "auto" measures it on
                  the anchor frame and corrects until the label's bow matches, cached in work/pitch.json;
                  "pitch_rows": [v0, v1] picks the label rows, default the texture's densest lines of print); "texture_rotate_deg": d turns the keyframe's
                  print in label space (scipy's sense: the angle that levels its lines) where this shot's generator drew
                  it level on the body; or {"texture_job": ..., "chain": {"job": "../job", "frame": N}}
                  where there is no cut: that job's own mapping on frame N (the shot's range starts at N) carries on
  key             product body colour: {"hue": [lo, hi] degrees, "sat": 0.27, "val": 0.35}
  track           {"step": "auto" | px, "points": 900, "erode": 4, "crop": [x0, y0, x1, y1] or "auto", "margin": 60,
                   "on": "all" | "print" (glossy bodies: their reflections don't move with the surface), "res": [h, w]}
  fit             {"wmin": 0.5, "reject": 2.5, "half": 3, "perspective": false, "camera": "ortho" | "affine" (tube),
                   "residual": {"sigma": "auto", "lam": 1},
                   "grid": 5, "min_inliers": 0.5, "max_rms": 3.0, "shear_max": deg (affine tube: a frame whose shear jumps
                   past it from its neighbours, a morph frame, takes their shape and a similarity fit to its tracks),
                   "morph_rigid": [clip frames] (of those, the ones where the product turns: a rigid camera through
                   their own tracks instead, which keeps the spin), "morph_skip": [clip frames] (a trend the test
                   flags at the end of a held run, where the tracks still follow the print: keep their own fit)}
  artwork         keyframe only: {"body_sigma": 12, "thresh": 0.12, "core": 0.25, "edge_band": 3}
  render          {"blur": "auto" | px (plus per-frame softness measured on the source), "shutter": 0..1 (motion
                   blur along the tracks; "auto" = none, generated frames rarely have it), "white_box": [x0, y0, x1,
                   y1] on the anchor frame (a white part of the product), "inset": 1.5, "polarity": "light" | "dark" |
                   "both", "supersample": 4, "print_tophat": 7, "print_thresh": 0.03, "spark_thresh": 0.05,
                   "spark_reach": 7, "ink_gain": "auto" | [r, g, b], "crf": 12, "ease": [frames in, frames out] (2, 2):
                   the fix fades in and out where the held shot ends with the product still in view, "sparkles": true
                   (false: no light re-added from the source, for shots no sparkle crosses; [frames]: only on these,
                   clip frames and half-open runs mixed, [98, [100, 119]]: glitter or a trail in front of the label
                   there, as the review's glitter check lists them in "sparkles_frames"), "spark_glow": true | [frames]
                   (sparkle trails keep their soft haze; a list: only on the frames a trail crosses; true with a
                   sparkles list: on those frames), "occluder_feather": px
                   (holes the matte cuts inside the label for something in front: fade into the source around them,
                   so its spilled haze stays), "feather": {"theta_deg":
                   10, "v_px": 24} (keyframe texture: the label fades into the source at its sides and band ends),
                   "limb_snap": {"step": 24, "max_px": 30} (tube: the label's sides snapped onto the matte's per row;
                   a real tube isn't a straight taper; "hold": true keeps each side's fit to the rows it measured
                   (an occluder in front); "axial": "top" | "bottom": on morph frames, also the product's
                   end along the axis where the frames with good tracks put it), "body_snap": true | {...} (tube:
                   the label placed by the product's body, not by the generated print it tracks, V39: per frame the
                   label keeps the keyframe frame's proportion of the body along its axis and its print the
                   keyframe's angle to the body's centreline; Tube v -> p + f v plus a turn, median-filtered over the
                   shot, clamped; light and sparkles still read through the tracked mapping; the limb snap runs after
                   it. Keys (fit.BODY_SNAP): "top" / "bottom": the edge that bounds the body above / below the label,
                   "auto" (the matte's end if within reach, else the stronger luminance rise or fall, decided on the
                   reference frame) | "matte" | "rise" | "fall" | "edge" | "dark" | "light" (the darkest / brightest
                   line, a cap seam) | "none"; "reach": [0.37, 0.42] (search above / below, shares of the label's
                   length; then each frame only within "window": 0.5 x the gap the reference predicts + 2 % of the
                   label), "inside": 0.05 (a dark / light line may sit this far inside the label), "across": 0.65,
                   "median": 5, "outlier_k": 4, "stretch": [0.9, 1.1], "roll_max": 4 deg, "shift_max": 0.15 (of the
                   label's length), "ref_frame": clip frame (default the anchor: the keyframe's frame), "frames":
                   [clip frames] (apply only there; measured over the whole shot)), "alpha": "label" | "print"
                   (keyframe texture, light print: "label" puts the relit label band over the source; "print" changes
                   only the print, carried onto the source's own body, so no patch outline can show, V40,
                   imgvideo/printalpha.py; with "print", unless the settings say otherwise: soft_cap 0.3, smear false,
                   feather v_px 4, and the sparkle re-add off, always: it ghosts the garbled print), "print_alpha":
                   {"ink": "keyframe" (the new print is the keyframe's own ink, carried through each frame's mapping)
                   | "render" (the render's top-hat), "carry": true (false: paste the render's pixels), "grow": 3,
                   "sigma": 1.5, "near": 14 (garbled print this close to the new print goes), "ink_levels": 4,
                   "ink_p90": 8, "ink_inset": 6, "presence": "match" (the print's presence, its mean lift over its
                   footprint, brought up to the source's per frame, smoothed, never lowered; "off") with
                   "presence_target": 1.0 (1.25 for thin true copy against a bold garbled print) and "max_gain": 2.5,
                   "shutter": 0.5 (motion blur along the label's own motion, at most "max_blur" 3 px, only from
                   "min_speed" 8 px/frame, and only as soft as the source's print is: none at "sharp_full" 0.9 of the
                   keyframe frame's sharpness; 0 = no blur), ... (printalpha.DEFAULTS)}, "glue": true | {"sigma": 4,
                   "edge": "line" | "mean" | [start, end], "rms_max": 0.9, "dev_max": 6} (V42, imgvideo/glue.py:
                   the placement carried frame to frame along the product's own motion, Lucas-Kanade on the source's
                   pixels, and pulled back onto the placement over sigma frames, zero-lag; the keyframe's frame stays
                   exact; the limb snap is inside it), "level": true | {"sigma": 4, "offset": "base" | deg,
                   "base_mad_max": 0.5, "gate": true, ...} (V45, imgvideo/level.py: after the glue, the label sheared
                   along the body's axis so its lines keep the keyframe frame's level against the body, offset "base"
                   levelling the keyframe's own lines with its base too; only where the straightened base holds
                   steady), "soft_cap": px (the still stage's
                   per-frame softness at most this; a restored label stays sharp), "smear": "auto" (as measured) |
                   false | [clip frames] (only there: motion blur the source clearly shows), "frames": [clip frames] (a
                   patch job: renders and delivers only these; merge it first, before the approved job, which keeps
                   the rest)}
  qa              {"step": 2.5, "ncc_min": 0.6, "ncc_drop": 0.08, "lock_min": 0.3, "rms_max": 1.5, "unseen_max": 0.02,
                   "text_boxes": [{"box": [u0, v0, u1, v1], "rotate": k} or {"centre": [u, v], "size": [w, h],
                   "angle": deg}, reference px, optional "ink"], "words": [expected copy, default: OCR of the truth]}
  review          the automatic checks' thresholds (imgvideo/review.py DEFAULTS: edges, ghost, shear, bend, body,
                  glitter, vanish, switch), plus "cuts": [frames] to list the cuts by hand, and "looks": {...}
                  (imgvideo/lookchecks.py LOOKS: swim, ends, level, sharp, tilt, ghost, spots, each flag with the
                  setting to try in review.actions) or false; they run on a lossless deliverable in qa and merge
  output          {"refined": mp4, "lossless": mov (QuickTime PNG: untouched pixels exact), "side_by_side": mp4,
                   "qa_plot": png}; defaults next to the clip
  merge, deliver  several shots of one clip: "merge": [the other shots' vjob.json], "deliver": {the same keys as
                  output}; --stage merge writes one clip from each shot's lossless render (one job per shot)
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")     # CoTracker's 5-D grid_sample isn't on MPS yet
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", category=RuntimeWarning)

# the job and the stages live in imgvideo/pipeline.py (the ComfyUI pack calls them on arrays); this is the command line
from imgvideo.pipeline import (  # noqa: E402,F401
    poly_mask,
    FileMedia,
    ArrayMedia,
    _InfoMedia,
    Job,
    stage_track,
    stage_fit,
    _fit,
    _white,
    stage_still,
    _font,
    _sbs_frame,
    _render_context,
    _render_one,
    _render_worker_init,
    _render_worker,
    render_workers,
    stage_render,
    stage_qa,
    stage_merge,
    _review,
    _delivered,
    _apply_review,
    _write_review,
    stage_review,
    plot_qa,
)


def refine_files(a):
    """--refine: imgvideo.pipeline.refine on files. The clip, its product matte (a video, white = product) and the
    corrected keyframes (images, the clip's size) in; into --out: {clip}_refined_lossless.mov (QuickTime PNG: the input
    outside the changed pixels, exactly; comp this), {clip}_refined.mp4 (H.264, to review), {clip}_matte.mov (the matte
    to paste through, patched frames included), {clip}_changed.mov (exactly the pixels that changed),
    {clip}_report.json, {clip}_label-qa.png, and work/ (the shots' caches: a re-run with the same inputs skips tracking)."""
    import json
    import shutil

    import numpy as np

    from imgpost import images as im
    from imgvideo import vio
    from imgvideo.pipeline import refine
    if not (a.clip and a.mask and a.keyframe and a.out):
        raise SystemExit("--refine needs --clip, --mask, --keyframe (one or more) and --out")
    clip, mask, out = Path(a.clip).expanduser(), Path(a.mask).expanduser(), Path(a.out).expanduser()
    info = vio.probe(clip)
    frames = vio.read(clip, 0, info["frames"], info)
    masks = vio.read(mask, 0, info["frames"])[..., 0] > 127
    kfs = [im.load_u8(Path(k).expanduser())[0] for k in a.keyframe]
    kfr = None if not a.keyframe_frame else [None if x == "auto" else int(x) for x in a.keyframe_frame]
    settings = json.loads(Path(a.settings).expanduser().read_text()) if a.settings else {}
    out.mkdir(parents=True, exist_ok=True)
    r = refine(frames, masks, kfs, settings, kfr, fps=info["fps"], work=out / "work", workers=a.workers)
    stem = clip.stem
    W = vio.Writer(out / f"{stem}_refined_lossless.mov", info, audio_from=clip, lossless=True)
    Wr = vio.Writer(out / f"{stem}_refined.mp4", info, audio_from=clip)
    for f in r["frames"]:
        W.write(f)
        Wr.write(f)
    W.close()
    Wr.close()
    for name in ("matte", "changed"):
        Wm = vio.Writer(out / f"{stem}_{name}.mov", info, lossless=True)
        for m in r[name]:
            Wm.write(np.repeat((m.astype(np.uint8) * 255)[..., None], 3, -1))
        Wm.close()
    (out / f"{stem}_report.json").write_text(json.dumps(r["report"], indent=1, default=str))
    if r["qa_plot"]:
        shutil.copy2(r["qa_plot"], out / f"{stem}_label-qa.png")
    print(f"saved      {out / (stem + '_refined_lossless.mov')}\n           (+ _refined.mp4, _matte.mov, _changed.mov, "
          f"_report.json{', _label-qa.png' if r['qa_plot'] else ''})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vjob", nargs="?")
    ap.add_argument("--stage", default="all", choices=["track", "fit", "still", "render", "qa", "merge", "review", "stitch",
                                                       "all"])
    ap.add_argument("--range", help="A:B clip frames to render as a test (into work/)")
    ap.add_argument("--notes", action="store_true", help="review: also write the checks into the round's notes.md")
    ap.add_argument("--workers", type=int, help="render: processes (default PLR_WORKERS, else the cores less two, at most "
                    "8; 1 = in this process); the output is the same bit for bit with any number")
    ap.add_argument("--refine", action="store_true", help="the ComfyUI node's contract on files (refine_files): --clip, "
                    "--mask and --keyframe in, the refined clip, matte and report into --out; no vjob")
    ap.add_argument("--clip", help="refine: the clip (full frames, or a workflow's crops)")
    ap.add_argument("--mask", help="refine: the product's matte as a video, same size and frame count")
    ap.add_argument("--keyframe", nargs="+", help="refine: corrected keyframes (images the clip's size), one per shot")
    ap.add_argument("--keyframe-frame", nargs="+", help="refine: each keyframe's clip frame, or auto (matched)")
    ap.add_argument("--settings", help="refine: JSON of vjob sections merged over what is measured (see refine())")
    ap.add_argument("--out", help="refine: the output folder")
    a = ap.parse_args()
    if a.refine:
        return refine_files(a)
    if not a.vjob:
        ap.error("a vjob.json, or --refine")
    if a.stage == "stitch":                  # a stitch job has no shot of its own (imgvideo/stitch.py)
        from imgvideo.stitch import stitch
        return stitch(a.vjob)
    J = Job(a.vjob)
    rng = tuple(int(x) for x in a.range.split(":")) if a.range else None
    stages = ["track", "fit", "still", "render", "qa"] if a.stage == "all" else [a.stage]
    for s in stages:
        if s == "render":
            stage_render(J, rng, a.workers)
        elif s == "qa" and rng:
            continue
        elif s == "review":
            stage_review(J, a.notes)
        else:
            {"track": stage_track, "fit": stage_fit, "still": stage_still, "qa": stage_qa, "merge": stage_merge}[s](J)


if __name__ == "__main__":
    main()
