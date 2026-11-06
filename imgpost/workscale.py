"""Work scale (pitfall 113): a small still is resized up before the fix, so the new letters are drawn with pixels
enough to read (a 1,024 px frame's small print is 5 px tall). The draft's measures are tuned at a frame's own size (a
label's corners came out 400 px off on the same frame resized 4x), so the draft runs on the frame at its own size and
its job is scaled up to the work size: every scene-px field times the factor, and the ink's scene-px reaches with it
(matte.px_scale: the garbled print's fill and its inset, ink.config).

  resized_by(u8)             a frame already resized up (a chained product's: the previous one's result): 4, 2 or 1
  down(scene_u8, masks, f)   the frame (Lanczos) and its masks (area, > 0.5) at 1 / f
  up_job(job, f)             the job's scene px times f (pixel centres kept: x -> f x + (f - 1) / 2)
"""
import numpy as np
from PIL import Image

RESIZED_DB = {4: 42.0, 2: 48.0}   # a frame resized up f x (Lanczos) comes back this close (PSNR, dB) from a round trip
                                   # down and up by f: 46-48 dB at 4x (a 1,024 px frame, and with a label fixed on it),
                                   # 53-54 at 2x; native generated stills 30-36 at 4x, 39-44 at 2x


def _round_trip_db(u8, f):
    h, w = u8.shape[:2]
    h2, w2 = h // f * f, w // f * f
    im = Image.fromarray(np.ascontiguousarray(u8[:h2, :w2]))
    back = np.asarray(im.resize((w2 // f, h2 // f), Image.LANCZOS).resize((w2, h2), Image.LANCZOS)).astype(np.float64)
    mse = float(((u8[:h2, :w2].astype(np.float64) - back) ** 2).mean())
    return 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-9))


def resized_by(u8):
    """How many times a frame was already resized up (4, 2 or 1): a chained product's frame is the previous product's
    work-size result, so its draft runs at that frame's own size too. -> (factor, {f: dB})"""
    db = {}
    for f in (4, 2):
        db[f] = round(_round_trip_db(u8, f), 1)
        if db[f] >= RESIZED_DB[f]:
            return f, db
    return 1, db


def down(scene_u8, masks, f):
    """-> (scene at 1 / f, [mask at 1 / f or None])"""
    if f == 1:
        return scene_u8, list(masks)
    h, w = scene_u8.shape[:2]
    size = (max(1, int(round(w / f))), max(1, int(round(h / f))))
    s = np.asarray(Image.fromarray(scene_u8).resize(size, Image.LANCZOS))
    out = []
    for m in masks:
        if m is None:
            out.append(None)
            continue
        a = Image.fromarray((np.asarray(m, bool) * 255).astype(np.uint8)).resize(size, Image.BOX)
        out.append(np.asarray(a) > 127)
    return s, out


def _p(v, f):
    return round(float(v) * f + 0.5 * (f - 1), 3)


def _pts(P, f):
    return [[_p(x, f) for x in q] for q in P]


def up_job(job, f):
    """The job's scene px (work box, size, landmark and edge points, limbs, bow, measured outlines, occluders, the
    label's outline, the fill's and the grade's reaches) times f, in place; matte.px_scale = f. -> job"""
    if f == 1:
        return job
    job["scene_size"] = [int(v * f) for v in job["scene_size"]]
    job["roi"] = [int(round(v * f)) for v in job["roi"]]
    al = job.get("align") or {}
    for k in ("points", "lines", "limbs"):
        for it in al.get(k) or []:
            if "scene" in it:
                s = it["scene"]
                it["scene"] = _pts(s, f) if s and isinstance(s[0], list) else [_p(x, f) for x in s]
    bow = al.get("bow")
    if isinstance(bow, dict):
        if isinstance(bow.get("centre"), list):
            bow["centre"] = [_p(x, f) for x in bow["centre"]]
        for k in ("half_widths", "b"):
            if isinstance(bow.get(k), list):
                bow[k] = [[round(float(x) * f, 3) for x in q] for q in bow[k]]
    for o in job.get("occluders") or []:
        if isinstance(o, dict) and isinstance(o.get("polygon"), list):
            o["polygon"] = _pts(o["polygon"], f)
    osil = job.get("old_silhouette")
    if isinstance(osil, list):
        job["old_silhouette"] = _pts(osil, f)
    elif isinstance(osil, dict) and isinstance(osil.get("measured"), list):
        osil["measured"] = _pts(osil["measured"], f)
    m = job.get("matte")
    if isinstance(m, dict):
        if isinstance(m.get("label"), list):
            m["label"] = _pts(m["label"], f)
        m["px_scale"] = f
    fl = job.get("fill")
    if isinstance(fl, dict) and isinstance(fl.get("dilate"), (int, float)):
        fl["dilate"] = int(round(fl["dilate"] * f))
    g = job.get("grade")
    if isinstance(g, dict):
        for k in ("erode", "deep_erode", "window"):
            if isinstance(g.get(k), (int, float)) and not isinstance(g.get(k), bool):
                g[k] = int(round(g[k] * f))
    job["work_scale"] = f
    return job
