#!/usr/bin/env python3
"""Build the ink demo: a made-up frosted bottle with white print straight on its coloured body, a generated-looking
frame showing a garbled copy of it from slightly above, the bottle's mask in that frame, and the job.

  python3 -P tools/make_ink_demo.py [--out examples/ink_demo]

Writes four files:
  inkdemo_packshot.png  the real product: the bottle's clean white print on its coral body, front view, plain backdrop
  inkdemo_frame.png     the "generated" frame: the same bottle smaller, turned 3 degrees, seen from slightly above (its
                        seams and print bow), lit from the left, with frosted rims, its print garbled (misspelled,
                        a px or two off), softened and grained, on a mint backdrop
  inkdemo_mask.png      the bottle's mask in the frame (what SAM 3 gives)
  job.json              the measurements that fix it: matte ink (polarity light) and align.bow
Everything is drawn from code with fixed seeds, so the demo shows no real product or brand. The job's landmarks
and bow come from the geometry that placed the bottle; on a real image they are measured (Auto Draft, run.py draft).
Needs numpy, scipy and Pillow 10.1 or later.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, map_coordinates

PACK = Path(__file__).resolve().parent.parent
SS = 4
REF_W, REF_H = 600, 1100
FRAME_W, FRAME_H = 768, 1024
BODY = (150, 330, 450, 1000)            # body box in the packshot: shoulder seam at 330, base at 1000
COLLAR = (228, 262, 372, 330)
CAP = (238, 96, 362, 262)
R_BASE = 34
PRINT_BOX = [176, 396, 432, 924]        # the print's rows and columns on the packshot, inside the frosted rims
SCALE, TURN = 0.4, np.radians(3.0)
CENTRE = (384.0, 560.0)                 # where the packshot's body centre lands in the frame
BOW = [[-140.0, 4.0], [130.0, 7.0]]     # px along the bottle's axis from the mask's centre: seen from above, lower bows more
ROI_PAD = 40


def srgb_to_lin(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def font(size):
    return ImageFont.load_default(size=size * SS)


def S(*v):
    return [int(round(c * SS)) for c in v]


def silhouette():
    """The bottle's coverage on the packshot (0..1), body only and all of it"""
    m = Image.new("L", (REF_W * SS, REF_H * SS), 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle(S(*BODY), radius=R_BASE * SS, fill=255)
    d.rectangle(S(BODY[0], BODY[1], BODY[2], BODY[1] + 2 * R_BASE), fill=255)          # square shoulders
    body = np.asarray(m.reduce(SS), float) / 255.0
    d.rectangle(S(*COLLAR), fill=255)
    d.rounded_rectangle(S(*CAP), radius=10 * SS, fill=255)
    return body, np.asarray(m.reduce(SS), float) / 255.0


def draw_print(version):
    """White print on the packshot's frame (coverage 0..1). generated: misspelled, a px or two off, other sizes"""
    real = version == "real"
    m = Image.new("L", (REF_W * SS, REF_H * SS), 0)
    d = ImageDraw.Draw(m)
    word = "SAMPLE STUDIO" if real else "SAMPEL STUIDO"
    wm = Image.new("L", (560 * SS, 70 * SS), 0)
    ImageDraw.Draw(wm).text(S(280, 35), word, font=font(50 if real else 48), fill=255, anchor="mm")
    wm = wm.rotate(90, expand=True)
    m.paste(wm, S(180 if real else 181, 400 if real else 398), wm)
    dx, dy = (0, 0) if real else (1, 1)
    for text, size, y in ((("Light Print", 40, 560), ("on a Body", 40, 606), ("frosted sample bottle", 18, 676),
                           ("white ink, 30 ml", 18, 702), ("made from code", 18, 728)) if real else
                          (("Ligth Pirnt", 39, 561), ("on a Bdoy", 41, 607), ("forsted smaple botle", 18, 677),
                           ("whtie ink, 03 ml", 18, 703), ("mdae form cdoe", 18, 729))):
        d.text(S(252 + dx, y + dy), text, font=font(size), fill=255, anchor="lm")
    d.ellipse(S(342 + dx, 820 + dy, 392 + dx, 870 + dy), outline=255, width=5 * SS)
    d.line(S(355 + dx, 845 + dy, 379 + dx, 845 + dy), fill=255, width=5 * SS)
    return np.asarray(m.reduce(SS), float) / 255.0


def body_colour(base, cap):
    """The bottle's colours on the packshot (sRGB float): the body front-lit (a little darker at its limbs), the
    cap pale"""
    x = (np.arange(REF_W) - 300.0) / 150.0
    shade = 0.92 + 0.08 * np.sqrt(np.clip(1 - x ** 2, 0, 1))
    img = np.array(base)[None, None] * shade[None, :, None] * np.ones((REF_H, 1, 1))
    return np.where((np.arange(REF_H) < COLLAR[1])[:, None, None], np.array(cap)[None, None], img)


def mapping():
    """packshot px -> frame px (affine), and its inverse"""
    c, s = np.cos(TURN), np.sin(TURN)
    A = SCALE * np.array([[c, -s], [s, c]])
    b = np.array(CENTRE) - A @ np.array([300.0, (BODY[1] + BODY[3]) / 2.0])
    return A, b


def frame_coords(bow=None):
    """For every frame pixel, the packshot point it shows (inverse affine after the bow)"""
    A, b = mapping()
    Y, X = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(float)
    if bow is not None:
        d = bow.shift(X, Y)
        X, Y = X - d * bow.axis[0], Y - d * bow.axis[1]
    Ai = np.linalg.inv(A)
    u = Ai[0, 0] * (X - b[0]) + Ai[0, 1] * (Y - b[1])
    v = Ai[1, 0] * (X - b[0]) + Ai[1, 1] * (Y - b[1])
    return u, v


def sample(img, u, v, order=1):
    """img sampled at packshot coords, supersampled 3 x 3 per frame pixel"""
    if img.ndim == 2:
        return map_coordinates(img, [v, u], order=order, mode="constant")
    return np.stack([map_coordinates(img[..., k], [v, u], order=order, mode="constant") for k in range(img.shape[2])], -1)


def render(bow):
    """-> frame sRGB uint8, mask bool, from the packshot's drawings through the affine + bow"""
    body_r, all_r = silhouette()
    gen = draw_print("generated")
    u, v = frame_coords(bow)
    # 3 x 3 supersampling: offsets in frame px mapped through the local affine
    A, _ = mapping()
    Ai = np.linalg.inv(A)
    acc_all = acc_body = acc_print = 0.0
    n = 0
    for oy in (-1 / 3, 0, 1 / 3):
        for ox in (-1 / 3, 0, 1 / 3):
            du, dv = Ai @ np.array([ox, oy])
            acc_all = acc_all + sample(all_r, u + du, v + dv)
            acc_body = acc_body + sample(body_r, u + du, v + dv)
            acc_print = acc_print + sample(gen, u + du, v + dv)
            n += 1
    cov_all, cov_body, cov_print = acc_all / n, acc_body / n, acc_print / n
    rng = np.random.default_rng(5)
    Y, X = np.mgrid[0:FRAME_H, 0:FRAME_W].astype(float)
    bg = srgb_to_lin(np.array([158, 212, 192]) / 255.0)[None, None] * (0.85 + 0.25 * (1 - Y / FRAME_H))[..., None]
    # the body: coral, lit from the left, frosted rims and a bright band under the shoulder
    s_across = ((u - 300.0) / 150.0)
    lit = 0.80 + 0.30 * np.clip(1 - (s_across + 0.35) ** 2, 0, 1)
    rim = np.exp(-((np.abs(s_across) - 0.93) / 0.035) ** 2) * 0.35
    shoulder = np.exp(-((v - 348.0) / 7.0) ** 2) * 0.25 * (np.abs(s_across) < 0.9)
    body = srgb_to_lin(np.array([240, 138, 120]) / 255.0)[None, None] * lit[..., None] + (rim + shoulder)[..., None] * 0.45
    cap = srgb_to_lin(np.array([236, 230, 226]) / 255.0)[None, None] * (0.85 + 0.15 * lit)[..., None]
    prod = np.where((v < COLLAR[1] + 0.5)[..., None], cap, body)
    ink = np.clip(body * np.array([1.18, 1.65, 1.75])[None, None], 0, 1)
    prod = prod * (1 - cov_print[..., None] * (cov_body > 0.5)[..., None]) + ink * (cov_print * (cov_body > 0.5))[..., None]
    out = bg * (1 - cov_all[..., None]) + prod * cov_all[..., None]
    out = np.stack([gaussian_filter(out[..., k], 0.6) for k in range(3)], -1)
    out = lin_to_srgb(out) + rng.normal(0, 0.008, out.shape)
    return np.clip(np.round(out * 255), 0, 255).astype(np.uint8), cov_all > 0.5


def build_job(mask, bow):
    A, b = mapping()

    def f(p):
        return [round(float(v), 2) for v in A @ np.asarray(p, float) + b]

    roi_ys, roi_xs = np.nonzero(mask)
    roi = [int(roi_xs.min()) - ROI_PAD, int(roi_ys.min()) - ROI_PAD, int(roi_xs.max()) + ROI_PAD + 1,
           int(roi_ys.max()) + ROI_PAD + 1]
    marks = [("wordmark top", (205, 414)), ("wordmark bottom", (205, 946)), ("heading start", (254, 560)),
             ("heading end", (420, 560)), ("seal centre", (367, 845))]
    return {
        "about": "Ink demo (tools/make_ink_demo.py): a made-up frosted bottle whose white print came out garbled in "
                 "a generated frame, seen from slightly above. The landmarks come from the geometry that placed the "
                 "bottle (the affine part; the bow lies over it); on a real image they are measured.",
        "scene": "inkdemo_frame.png", "reference": "inkdemo_packshot.png", "output": "inkdemo_frame_refined.png",
        "scene_size": [FRAME_W, FRAME_H], "reference_size": [REF_W, REF_H], "roi": roi,
        "reference_outline": {"type": "polygon", "pts": [[BODY[0], BODY[1]], [BODY[2], BODY[1]], [BODY[2], BODY[3]],
                                                         [BODY[0], BODY[3]]]},
        "align": {"model": "affine", "points": [{"label": l, "ref": list(p), "scene": f(p), "w": 1} for l, p in marks],
                  "y_only": [], "lines": [],
                  "bow": {"b": bow.profile.tolist(), "centre": [round(float(v), 4) for v in bow.centre],
                          "axis": [round(float(v), 8) for v in bow.axis],
                          "half_widths": [[float(t), float(w)] for t, w in bow.half_widths]}},
        "old_silhouette": {"type": "mask", "path": "inkdemo_mask.png"},
        "occluders": [],
        "matte": {"type": "ink", "polarity": "light", "print_box": PRINT_BOX},
        "grade": {}, "finish": {}, "fill": {},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(PACK / "examples" / "ink_demo"))
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PACK))
    from imgpost import bow as bw, geometry as geo

    body_r, all_r = silhouette()
    real = draw_print("real")
    pack = body_colour(np.array([236, 146, 128]) / 255.0, np.array([238, 236, 232]) / 255.0)
    pack = pack * (1 - real[..., None] * (body_r[..., None] > 0.5)) + real[..., None] * (body_r[..., None] > 0.5)
    pack = pack * all_r[..., None] + np.array([248, 248, 246])[None, None] / 255.0 * (1 - all_r[..., None])
    Image.fromarray(np.clip(np.round(pack * 255), 0, 255).astype(np.uint8)).save(out / "inkdemo_packshot.png", optimize=True)

    # the bow's axis and widths from the bottle's mask without the bow (its matte barely moves under it)
    _, mask0 = render(None)
    c, ax, n = bw.body_axis(mask0)
    bow = geo.Bow(np.eye(3), c, ax, bw.half_widths(mask0, c, ax, n), BOW)
    frame, mask = render(bow)
    Image.fromarray(frame).save(out / "inkdemo_frame.png", optimize=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out / "inkdemo_mask.png", optimize=True)
    (out / "job.json").write_text(json.dumps(build_job(mask, bow), indent=2) + "\n")
    for name in ("inkdemo_packshot.png", "inkdemo_frame.png", "inkdemo_mask.png", "job.json"):
        print(f"wrote {out / name}  {(out / name).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
