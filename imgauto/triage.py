"""Does each product need fixing at all? (2026-10-18; the user: "check whether the product even needs fixing ... in
one input there could be several products, and some of them could have hallucinations, and others might look
completely fine. So no need to waste time and resources on the products that don't have any issues in them.")

Run per product (and per shot in a video) on its clearest frames, before any tracking, keyframe or generation. Each
frame's product is straightened first (its axis upright: the body's two-pass axis, as the label checks do), and so is
the reference photo's product, scaled to the frame's size. Then:

  sheet   per frame: the frame's print band next to the reference's at the frame's size, both straightened, in
          colour, enlarged. The agent reads it and calls each product (or shot) "fix" or "fine".
  hints   optional (--ocr, --glyph; about 10 s a frame): OCR of both (words of 3+ letters spelled against the copy)
          and a letter-scale match. They
          don't vote: on the first real job OCR read 0-0.23 of the copy on the CORRECTED serum and tube (soft, curved,
          low-contrast white print), the same as on the garbled source, against 0.33-0.62 off the reference; the
          glyph match failed the same way (the headline was nearly right in both, only the small copy differed).

A product is skipped only on the agent's "fine": leaving a garbled label costs more than a render. The look is one
sheet per product (seconds), against minutes of tracking and rendering per product saved when it's fine.

python -P -m imgauto.triage FRAME.png MASK.png REFERENCE.png [--copy "word word ..."] [--sheet OUT.jpg] [--ocr] [--glyph]
  (one frame, a test; PYTHONPATH=scripts)
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from . import verify

TH = {"min_ceiling": 0.3}


def axis(M):
    """Centre, unit axis direction (down the body), half-width: two passes (principal axis of the lower 60 %, then the
    limbs' midpoints across the body's middle in that frame), so a tilted body's rounded ends don't lean it."""
    ys, xs = np.nonzero(M)
    P = np.stack([xs, ys], 1).astype(float)
    c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    d = V[:, np.argmax(w)]
    d = d if d[1] >= 0 else -d
    a = (P - c) @ d
    Pb = P[a > a.min() + 0.4 * np.ptp(a)]
    c = Pb.mean(0)
    w, V = np.linalg.eigh(np.cov((Pb - c).T))
    d = V[:, np.argmax(w)]
    d = d if d[1] >= 0 else -d
    for _ in range(2):
        n = np.array([d[1], -d[0]])
        a, x = (Pb - c) @ d, (Pb - c) @ n
        lo, hi = np.quantile(a, 0.2), np.quantile(a, 0.8)
        at, mid, half = [], [], []
        for b0 in np.arange(lo, hi, 2.0):
            s = (a >= b0) & (a < b0 + 2.0)
            if s.sum() > 4:
                at.append(b0 + 1.0)
                mid.append(0.5 * (x[s].min() + x[s].max()))
                half.append(0.5 * (x[s].max() - x[s].min()))
        if len(at) < 4:
            break
        k, m0 = np.polyfit(at, mid, 1)
        c, d = c + m0 * n, d + k * n
        d = d / np.linalg.norm(d)
    return c, d, (float(np.median(half)) if half else 0.5 * np.ptp(x))


def straighten(img, M, step=1.0, rgb=False):
    """The product upright, at 1/step px: (gray 0..1 (or RGB 0..1 with rgb), mask, length along the axis in source
    px)."""
    c, d, hw = axis(M)
    n = np.array([d[1], -d[0]])
    ys, xs = np.nonzero(M)
    al = (np.stack([xs, ys], 1) - c) @ d
    A = np.arange(al.min() - 4, al.max() + 4, step)
    C = np.arange(-hw - 6, hw + 6, step)
    X = c[0] + A[:, None] * d[0] + C[None, :] * n[0]
    Y = c[1] + A[:, None] * d[1] + C[None, :] * n[1]
    if rgb:
        im3 = img if img.ndim == 3 else np.repeat(img[..., None], 3, -1)
        sc = 255.0 if im3.max() > 1.5 else 1.0
        G = np.stack([ndi.map_coordinates(im3[..., k].astype(np.float32) / sc, [Y, X], order=1) for k in range(3)], -1)
    else:
        g = img.astype(np.float32).mean(-1) / 255.0 if img.ndim == 3 else img.astype(np.float32)
        G = ndi.map_coordinates(g, [Y, X], order=1)
    m = ndi.map_coordinates(M.astype(np.float32), [Y, X], order=1) > 0.5
    return G, m, float(np.ptp(al))


def _bp(g, lo, hi):
    return ndi.gaussian_filter(g, lo) - ndi.gaussian_filter(g, hi)


def _ncc(a, b, m):
    a, b = a[m], b[m]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def _place(R, shape, dy, dx, rot, sc):
    """R resampled into an array of `shape`: scaled by sc and turned by rot (deg) about R's centre, put at the
    target's centre, shifted (dy, dx)."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = (H - 1) / 2 + dy, (W - 1) / 2 + dx
    t = np.radians(rot)
    u, v = (xx - cx), (yy - cy)
    xr = (np.cos(t) * u + np.sin(t) * v) / sc + (R.shape[1] - 1) / 2
    yr = (-np.sin(t) * u + np.cos(t) * v) / sc + (R.shape[0] - 1) / 2
    return ndi.map_coordinates(R, [yr, xr], order=1, cval=np.nan)


def glyphs(F, Fm, R, Rm):
    """Align the reference print R to the frame's F (straightened, same scale) on the layout band, then compare the
    letters. -> {layout, glyphs, ceiling, shift, rot, scale}"""
    inner = ndi.binary_erosion(Fm, iterations=4)
    Fl, Ff = _bp(F, 3.0, 8.0), _bp(F, 0.7, 2.0)
    best = (-2, None)
    for sc in (0.96, 1.0, 1.04):
        for rot in (-2.0, 0.0, 2.0):
            Rl = _bp(np.nan_to_num(_place(np.where(Rm, R, np.nan), F.shape, 0, 0, rot, sc), nan=float(np.nanmean(R))), 3.0, 8.0)
            for dy in range(-12, 13, 3):
                for dx in range(-6, 7, 3):
                    Rs = np.roll(np.roll(Rl, dy, 0), dx, 1)
                    v = _ncc(Fl, Rs, inner)
                    if v > best[0]:
                        best = (v, (dy, dx, rot, sc))
    layout, (dy, dx, rot, sc) = best
    # refine to 1 px, then the letters
    Rp = np.nan_to_num(_place(np.where(Rm, R, np.nan), F.shape, 0, 0, rot, sc), nan=float(np.nanmean(R)))
    g = -2.0
    for ddy in range(dy - 2, dy + 3):
        for ddx in range(dx - 2, dx + 3):
            g = max(g, _ncc(Ff, _bp(np.roll(np.roll(Rp, ddy, 0), ddx, 1), 0.7, 2.0), inner))
    # the ceiling: the reference against itself after a frame-like softening (0.8 px) and a half-px shift
    Rb = ndi.shift(ndi.gaussian_filter(Rp, 0.8), (0.5, 0.5), order=1)
    ceil = _ncc(_bp(Rb, 0.7, 2.0), _bp(Rp, 0.7, 2.0), inner)
    return {"layout": round(layout, 3), "glyphs": round(g, 3), "ceiling": round(ceil, 3),
            "shift": [dy, dx], "rot": rot, "scale": sc}


def read_words(G, m):
    """OCR words on a straightened product: both polarities, upright and turned 90 deg either way."""
    inner = ndi.binary_erosion(m, iterations=3)
    L = G - ndi.gaussian_filter(G, 12)
    lo, hi = np.percentile(L[inner], [2, 98]) if inner.any() else (L.min(), L.max())
    N = np.clip((L - lo) / max(hi - lo, 1e-4), 0, 1)
    N = np.where(inner, N, 0.5)
    words = set()
    for img in (N, 1 - N):
        for k in (0, 1, 3):
            w = verify.ocr(np.rot90(img, k), psm=11, scale=3)
            if w:
                words |= w
    return words


def text_score(copy, words):
    if not copy or words is None:
        return None
    long_ = [w for w in copy if len(verify._norm_word(w)) >= 3]
    if not long_:
        return None
    sp = verify.spelling(long_, words)
    return round(sum(v == "exact" for v in sp.values()) / len(sp), 3)


def frame_verdict(text, ceiling, gl):
    """Always 'look': the agent decides from the sheet. OCR is a hint. On the first real job it read 0-0.23 of the copy on
    the CORRECTED serum and tube (soft, curved, low-contrast white print) against 0.33-0.62 off the reference at the
    same size, the same as on the garbled source: OCR can't tell them apart at frame scale. The glyph match failed
    the same way (the headline was nearly right in both; only the small copy differed)."""
    ratio = None if (text is None or ceiling is None or ceiling < TH["min_ceiling"]) else text / max(ceiling, 1e-6)
    g = None if gl.get("glyphs") is None else round(gl["glyphs"] / max(gl["ceiling"], 1e-6), 3)
    return "look", (None if ratio is None else round(ratio, 3)), g


def measure(frame, M, ref_rgb, ref_mask, copy=(), glyph=False, ocr=False):
    """One frame of one product -> the measures and its verdict. glyph: also the letter-scale match (reported only;
    a few seconds a frame)."""
    F, Fm, L = straighten(frame, M)
    Rg, Rm, Lr = straighten(ref_rgb, ref_mask, step=Lr_step(ref_mask, L))
    gl = glyphs(F, Fm, Rg, Rm) if glyph else {"layout": None, "glyphs": None, "ceiling": None}
    tf = tr = None
    if ocr:                                      # hints only (about 10 s a frame): off by default
        tf, tr = text_score(copy, read_words(F, Fm)), text_score(copy, read_words(Rg, Rm))
    v, ratio, grel = frame_verdict(tf, tr, gl)
    # the sheet: the print band of both, in colour (the rows where the reference has print, with a margin)
    Fc = straighten(frame, M, rgb=True)[0]
    Rc = straighten(ref_rgb, ref_mask, step=Lr_step(ref_mask, L), rgb=True)[0]
    th = (Rg - ndi.grey_opening(Rg, size=(15, 15))) > 0.03
    th |= (ndi.grey_closing(Rg, size=(15, 15)) - Rg) > 0.03            # dark print too
    rows = np.nonzero((th & ndi.binary_erosion(Rm, iterations=4)).any(1))[0]
    band = (max(int(np.percentile(rows, 1)) - 10, 0), int(np.percentile(rows, 99)) + 10) if len(rows) > 10 else (0, len(Rg))
    return {"verdict": v, "text": tf, "text_ceiling": tr, "text_ratio": ratio, "glyphs_rel": grel, **gl,
            "_crops": (Fc, Fm, Rc, Rm, band)}


def Lr_step(ref_mask, frame_len):
    """The reference's sampling step that brings its product to the frame's length."""
    ys, xs = np.nonzero(ref_mask)
    c, d, hw = axis(ref_mask)
    al = (np.stack([xs, ys], 1) - c) @ d
    return float(np.ptp(al)) / max(frame_len, 1.0)


def combine(verdicts, agent=None):
    """A product's (or shot's) verdict: the agent's ("fix" | "fine"), else "look" (waiting for it)."""
    return agent if agent in ("fix", "fine") else "look"


def sheet(rows, out, h=520):
    """Per measured frame: the frame's print band | the reference's at the frame's size, both straightened, in colour,
    enlarged to h px tall, with the OCR hints. What the agent judges fix or fine from."""
    tiles = []
    for lab, r in rows:
        F, Fm, R, Rm, (b0, b1) = r["_crops"]
        pair = []
        for g in (F, R):
            g = g[b0:min(b1, len(g))]
            im = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8))
            im = im.resize((max(1, im.width * h // max(im.height, 1)), h), Image.LANCZOS)
            pair.append(im)
        t = Image.new("RGB", (pair[0].width + pair[1].width + 6, h + 20), (20, 20, 20))
        t.paste(pair[0], (0, 20))
        t.paste(pair[1], (pair[0].width + 6, 20))
        hint = (f"   (OCR hint: {r['text']} of the copy; the reference at this size {r['text_ceiling']})"
                if r.get("text") is not None else "")
        ImageDraw.Draw(t).text((3, 4), f"{lab} | reference{hint}", fill=(255, 220, 0))
        tiles.append(t)
    per_row = 2
    rws = [tiles[i:i + per_row] for i in range(0, len(tiles), per_row)]
    W = max(sum(t.width + 10 for t in rw) for rw in rws)
    H = sum(max(t.height for t in rw) + 10 for rw in rws)
    s = Image.new("RGB", (W, H), (20, 20, 20))
    y = 0
    for rw in rws:
        x = 0
        for t in rw:
            s.paste(t, (x, y))
            x += t.width + 10
        y += max(t.height for t in rw) + 10
    s.save(out, quality=90)


def main():
    args = sys.argv[1:]
    fr, mk, rf = args[:3]
    copy = args[args.index("--copy") + 1].split() if "--copy" in args else []
    frame = np.asarray(Image.open(fr).convert("RGB"))
    M = np.asarray(Image.open(mk).convert("L")) > 127
    ref = Image.open(rf)
    rgba = np.asarray(ref.convert("RGBA"))
    if (rgba[..., 3] < 250).mean() > 0.05:
        Rm = rgba[..., 3] > 127
    else:
        rgb = rgba[..., :3].astype(np.float32)
        border = np.r_[rgb[:3].reshape(-1, 3), rgb[-3:].reshape(-1, 3)]
        Rm = np.abs(rgb - np.median(border, 0)).max(-1) > 25
        Rm = ndi.binary_opening(Rm, iterations=2)
        lab, n = ndi.label(Rm)
        Rm = lab == (1 + int(np.argmax(ndi.sum(Rm, lab, range(1, n + 1))))) if n else Rm
    r = measure(frame, M, rgba[..., :3], Rm, copy, glyph="--glyph" in args, ocr="--ocr" in args)
    print(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}))
    if "--sheet" in args:
        sheet([("frame", r)], args[args.index("--sheet") + 1])


if __name__ == "__main__":
    main()
