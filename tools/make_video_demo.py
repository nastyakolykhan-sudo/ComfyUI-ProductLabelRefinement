#!/usr/bin/env python3
"""Draw the label-video demo from code: a made-up tube turning and drifting in front of a soft backdrop, its label
"generated" with garbled print (wrong letters, wobbling lines), the product's matte, and one corrected keyframe (the
same frame with the true print). Nothing shows a real product or brand. Same libraries, same bytes.

  python3 -P make_video_demo.py OUT_DIR [--frames 40] [--size 432x768] [--keyframe 20]

-> OUT_DIR/labelvideo_demo_clip.mov (QuickTime PNG), labelvideo_demo_mask.mov, labelvideo_demo_keyframe.png
   (frame --keyframe with the true print); with --truth also labelvideo_demo_truth.mov (every frame, true print)
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TRUE_LINES = [("SAMPLE", 64), ("everyday", 34), ("hydrating serum", 30), ("for demo use only", 22), ("30 ml  1.0 fl oz", 22)]
GARBLED = [("SMAPLE", 64), ("evreyady", 34), ("hydarting srum", 30), ("fro dmeo ues olny", 22), ("3O ml  1.O fI oz", 22)]


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def label_texture(lines, w=720, h=460, seed=0, wobble=0.0):
    """The label unrolled: 360 degrees of the tube across w px, its height h px. Print in a dark ink on cream paper,
    a band of stripes and a small emblem, so the tracker has detail to hold."""
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (w, h), (243, 236, 222))
    d = ImageDraw.Draw(im)
    x0 = w // 2 - 120                                     # the front of the tube faces the camera at u = w/2
    y = 46
    for text, size in lines:                              # each line centred on the front of the tube
        f = font(size)
        tw = d.textlength(text, font=f)
        dx, dy = (int(round(rng.normal(0, wobble))) for _ in range(2)) if wobble else (0, 0)
        d.text((w / 2 - tw / 2 + dx, y + dy), text, fill=(38, 52, 92), font=f)
        y += int(size * 1.35)
    for k in range(9):                                    # stripes round the back
        d.rectangle([k * w // 9 + 6, h - 70, k * w // 9 + 40, h - 52], fill=(196, 86, 64))
    d.ellipse([x0 - 70, 52, x0 - 18, 104], outline=(38, 52, 92), width=5)
    d.line([x0 - 62, 78, x0 - 26, 78], fill=(38, 52, 92), width=4)
    return np.asarray(im)


def render(t, n, W, H, tex):
    """Frame t: the tube (radius, height) turning about its axis, drifting and tilting a little, lit from the left."""
    ph = 0.22 * np.sin(2 * np.pi * t / n)                 # turn, radians
    cx = W * (0.5 + 0.06 * np.sin(2 * np.pi * t / n * 0.7))
    cy = H * (0.52 + 0.02 * np.cos(2 * np.pi * t / n))
    tilt = np.deg2rad(4.0 * np.sin(2 * np.pi * t / n * 0.5))
    R, Lh, cap = W * 0.17, H * 0.46, H * 0.09
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    xr = (xx - cx) * np.cos(tilt) + (yy - cy) * np.sin(tilt)        # tube coordinates
    yr = -(xx - cx) * np.sin(tilt) + (yy - cy) * np.cos(tilt)
    xp = xx + 4.0 * t                                     # the backdrop pans: every frame differs outside the product
    bg = np.stack([200 + 30 * (yy / H), 182 + 25 * (xp / W) + 6 * np.sin(xp / W * 5.0), 196 + 20 * (yy / H)], -1)
    body = (np.abs(xr) < R) & (yr > -Lh / 2) & (yr < Lh / 2)
    capm = (np.abs(xr) < R * 0.86) & (yr <= -Lh / 2) & (yr > -Lh / 2 - cap)
    s = np.clip(xr / R, -1, 1)
    theta = np.arcsin(s)                                   # angle on the tube, 0 facing the camera
    shade = 0.55 + 0.45 * np.cos(theta + 0.5)              # light from the left
    u = ((theta + ph) / (2 * np.pi) + 0.5) % 1.0 * tex.shape[1]
    v = (yr + Lh / 2) / Lh * tex.shape[0]
    ui = np.clip(u.astype(int), 0, tex.shape[1] - 1)
    vi = np.clip(v.astype(int), 0, tex.shape[0] - 1)
    out = bg.copy()
    out[body] = tex[vi[body], ui[body]] * shade[body, None]
    out[capm] = np.array([70, 74, 82]) * (0.6 + 0.4 * np.cos(theta[capm] + 0.5))[:, None]
    gl = np.exp(-((s + 0.35) / 0.08) ** 2)[..., None] * 60                    # a soft gloss streak
    out = np.where(body[..., None], out + gl, out)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), body | capm


def write_mov(path, frames, fps=24):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                          "-i", "-", "-c:v", "png", "-pix_fmt", "rgb24", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError("ffmpeg failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--size", default="432x768")
    ap.add_argument("--keyframe", type=int, default=20)
    ap.add_argument("--truth", action="store_true", help="also write every frame with the true print (to check by eye)")
    a = ap.parse_args()
    W, H = (int(x) for x in a.size.split("x"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    true_tex = label_texture(TRUE_LINES)
    clip, truth, masks = [], [], []
    for t in range(a.frames):
        g = label_texture(GARBLED, seed=t, wobble=1.6)          # the generator redraws the print every frame
        f, m = render(t, a.frames, W, H, g)
        ft, _ = render(t, a.frames, W, H, true_tex)
        clip.append(f)
        truth.append(ft)
        masks.append(np.repeat((m * 255).astype(np.uint8)[..., None], 3, -1))
    write_mov(out / "labelvideo_demo_clip.mov", clip)
    write_mov(out / "labelvideo_demo_mask.mov", masks)
    if a.truth:
        write_mov(out / "labelvideo_demo_truth.mov", truth)
    Image.fromarray(truth[a.keyframe]).save(out / "labelvideo_demo_keyframe.png")
    print(f"wrote {a.frames} frames {W}x{H} and keyframe {a.keyframe} to {out}")


if __name__ == "__main__":
    main()
