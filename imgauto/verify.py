"""Checks on a corrected keyframe before the engine uses it, and the vjob parameters measured from it.

A corrected keyframe (a generative edit, or a hand fix) is trusted only when:
- it is a frame of the clip: least difference outside the grown matte, well clear of the runner-up;
- the edit stayed on the label: what changed lies inside the matte, and the product's outline didn't move;
- its text is the copy, spelled exactly. The engine's QA recall forgives one edit on long words (OCR slips at a few
  px per letter), so "niacynamide" would pass it. Here a near miss is "unconfirmed": a second reader (Qwen-VL on
  Swordfish) must read the word exactly, or the attempt fails.
"""
import numpy as np
from scipy import ndimage as ndi

from imgvideo import qa as vqa
from imgvideo import surface
from imgvideo.keyframe import product_axis


def find_frame(F2, M2, ref2, grow=5):
    """The clip frame (1/2 scale) the keyframe (same scale) was made from -> (frame, mean |diff| outside the grown
    matte, the runner-up's)."""
    ref = ref2.astype(np.int16)
    d = []
    for f, m in zip(F2, M2):
        out = ~ndi.binary_dilation(m, iterations=grow)
        d.append(float(np.abs(f.astype(np.int16) - ref)[out].mean()) if out.any() else np.inf)
    o = np.argsort(d)
    return int(o[0]), round(d[o[0]], 3), round(d[o[1]], 3)


def edit_region(ref, src, mask, thresh=12, grow=3):
    """Where the correction changed the frame: -> (changed inside the matte, share of the change outside it)."""
    diff = np.abs(ref.astype(np.int16) - src.astype(np.int16)).max(-1) > thresh
    diff = ndi.binary_opening(diff, iterations=1)
    inside = diff & ndi.binary_dilation(mask, iterations=grow)
    return inside, float((diff & ~inside).sum()) / max(int(diff.sum()), 1)


def label_band(ref, src, mask, pad=24, q=(1, 99)):
    """The label's stretch along the matte's axis, px from the matte's centroid (video.py's anchor.band), measured
    as where the correction changed the print. -> ([t0, t1], info)"""
    changed, outside = edit_region(ref, src, mask)
    c, ax, n = product_axis(mask)
    ys, xs = np.nonzero(changed)
    if len(xs) < 200:
        return None, {"changed_px": int(len(xs)), "outside_share": round(outside, 4)}
    t = (np.c_[xs, ys] - c) @ ax
    tm = (np.c_[np.nonzero(mask)[1], np.nonzero(mask)[0]] - c) @ ax
    t0 = max(float(np.percentile(t, q[0])) - pad, float(tm.min()))
    t1 = min(float(np.percentile(t, q[1])) + pad, float(tm.max()))
    return [int(round(t0)), int(round(t1))], {"changed_px": int(len(xs)), "outside_share": round(outside, 4),
                                              "matte_extent": [int(tm.min()), int(tm.max())]}


def body_band(ref, mask, step=8, gap=2, span=5, lum_step=20.0, width_step=0.06, print_min=0.15, trim=4,
              ref_len=1200.0, changed=None):
    """The label's stretch along the matte's axis from the product's own profile, the way it is measured by hand
    (workflow.md, Keyframe mode): per strip of t, the matte's width, brightness and print energy. A cap, pump, collar
    or crimp is a step in width or body colour that persists: the mean over `span` strips before it against the mean
    over `span` strips after (a smooth gradient along a lit body isn't one, nor is a line of print: caps, collars
    and crimps step 25-65 levels, text lines under 15). The band is the segment between steps that carries the most
    print, trimmed `trim` px at both ends.
    The strips were calibrated on products about 1,400-1,700 px long. A shorter product (its matte under `ref_len`)
    takes _short_band: strips, trim and print filter scaled by length / ref_len (V36: on a 370 px serum the 40 px
    windows read its frosted gradient as a step, and on a tapered tube every window pair as a width step).
    changed (optional, bool like mask): the keyframe's changed pixels; info["changed_in_band"] is the share of them
    inside the band. A print-only keyframe (the stills route) under 0.9 has a band that misses print.
    -> ([t0, t1], profile summary)"""
    c, ax, n = product_axis(mask)
    ys, xs = np.nonzero(mask)
    P = np.c_[xs, ys].astype(float)
    t, q = (P - c) @ ax, (P - c) @ n
    k = float(min(1.0, (t.max() - t.min()) / ref_len))
    if k < 1.0:
        out = _short_band(ref, mask, t, q, ys, xs, k, step, gap, span, lum_step, width_step, print_min, trim)
    else:
        out = _long_band(ref, t, q, ys, xs, step, gap, span, lum_step, width_step, print_min, trim)
    if out[0] is None:
        return out
    band, info = out
    info["scale"] = round(k, 3)
    if changed is not None and changed.any():
        cy, cx = np.nonzero(changed)
        tc = (np.c_[cx, cy] - c) @ ax
        info["changed_in_band"] = round(float(((tc >= band[0]) & (tc <= band[1])).mean()), 3)
    return band, info


def _strips(ref, t, q, ys, xs, step, k):
    g = ref.astype(np.float32).mean(-1)
    pb = np.abs(ndi.gaussian_filter(g, max(0.5, k)) - ndi.gaussian_filter(g, max(1.0, 4.0 * k)))
    gv, pv = g[ys, xs], pb[ys, xs]
    rows = []
    for t0 in np.arange(t.min(), t.max(), step):
        sel = (t >= t0) & (t < t0 + step)
        if sel.sum() < 10:
            continue
        lo, hi = np.percentile(q[sel], [3, 97])
        mid = sel & (np.abs(q - 0.5 * (lo + hi)) < 0.35 * (hi - lo))
        rows.append((t0, hi - lo, float(np.median(gv[sel])), float(pv[mid].mean()) if mid.any() else 0.0))
    return np.array(rows)


def _long_band(ref, t, q, ys, xs, step, gap, span, lum_step, width_step, print_min, trim):
    R = _strips(ref, t, q, ys, xs, step, 1.0)
    w, lum = R[:, 1], R[:, 2]
    has_print = R[:, 3] >= print_min * np.percentile(R[:, 3], 95)
    brk = np.zeros(len(R), bool)
    for i in range(span, len(R) - span - gap + 1):        # a step between strip i-1 and i+gap
        lb, la = lum[i - span:i].mean(), lum[i + gap:i + gap + span].mean()
        wb, wa = w[i - span:i].mean(), w[i + gap:i + gap + span].mean()
        brk[i] = abs(la - lb) > lum_step or abs(wa - wb) > width_step * wb
    idx = np.nonzero(brk)[0]
    cuts = [0]
    for i in idx:                                          # one cut per run of flagged strips, at its middle
        if cuts[-1] and i - cuts[-1] <= gap + 1:
            continue
        run = idx[(idx >= i) & (idx <= i + span)]
        cuts.append(int(run.mean() + gap / 2 + 0.5))
    cuts.append(len(R))
    segs = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= 2]
    if not segs:
        return None, {}
    a, b = max(segs, key=lambda s: (int(has_print[s[0]:s[1]].sum()), s[1] - s[0]))
    t0 = float(R[a, 0]) + trim
    t1 = float(R[b - 1, 0] + step) - trim
    return [int(round(t0)), int(round(t1))], {"matte_extent": [int(t.min()), int(t.max())],
                                              "body_width_px": round(float(np.median(w[a:b])), 1),
                                              "segments": [[int(R[x, 0]), int(R[y - 1, 0] + step)] for x, y in segs]}


def _short_band(ref, mask, t, q, ys, xs, k, step, gap, span, lum_step, width_step, print_min, trim):
    """body_band on a product shorter than the calibration (k = length / ref_len < 1): strips of max(2, step k) px,
    so a frosted gradient or a tube's taper stays under the thresholds inside a window pair. A luminance step must
    persist into the next window on both sides (clipped at the matte's ends: a crimp or cap there has little room),
    so a bold line of print, gone again a window later, isn't one. A segment is a run of strips between transitions,
    the transitions' own strips left out (the band starts where the shoulder ends). On the first real job's serum
    and tube (370-410 px) the bands came within 1-24 px of the hand's and held 97-100 % of the keyframes' changed
    pixels (the unscaled strips: 73 % and 38 %); the video demo's bold-text label 62 % (67 %)."""
    step, trim = max(2.0, step * k), max(1.0, trim * k)
    R = _strips(ref, t, q, ys, xs, step, k)
    w, lum = R[:, 1], R[:, 2]
    N = len(R)
    brk = np.zeros(N, bool)
    for i in range(1, N - gap):
        b0, a1 = max(0, i - span), min(N, i + gap + span)
        if i - b0 < 2 or a1 - (i + gap) < 2:
            continue
        wb, wa = w[b0:i].mean(), w[i + gap:a1].mean()
        lb, la = lum[b0:i].mean(), lum[i + gap:a1].mean()
        step_l = abs(la - lb) > lum_step
        if step_l:
            b00, a2 = max(0, b0 - span), min(N, a1 + span)
            if a2 - a1 >= 2:
                step_l &= abs(lum[a1:a2].mean() - lb) > lum_step
            if b0 - b00 >= 2:
                step_l &= abs(la - lum[b00:b0].mean()) > lum_step
        brk[i] = step_l or abs(wa - wb) > width_step * wb
    lab, nl = ndi.label(~brk)
    has_print = R[:, 3] >= print_min * np.percentile(R[:, 3], 95)
    segs = [np.nonzero(lab == j)[0] for j in range(1, nl + 1)]
    segs = [s for s in segs if len(s) >= 2]
    if not segs:
        return None, {}
    best = max(segs, key=lambda s: (int(has_print[s].sum()), len(s)))
    t0 = float(R[best[0], 0]) + trim
    t1 = float(R[best[-1], 0] + step) - trim
    return [int(round(t0)), int(round(t1))], {"matte_extent": [int(t.min()), int(t.max())],
                                              "body_width_px": round(float(np.median(w[best])), 1),
                                              "segments": [[int(R[s[0], 0]), int(R[s[-1], 0] + step)] for s in segs]}


def print_polarity(ref, mask, changed):
    """Whether the corrected print is lighter or darker than the body around it: 'light', 'dark' or 'both'."""
    g = ref.astype(np.float32).mean(-1)
    body = ndi.median_filter(g, size=25)
    d = (g - body)[changed & mask]
    if d.size < 100:
        return "both"
    lt, dk = float((d > 12).mean()), float((d < -12).mean())
    if lt > 2 * dk:
        return "light"
    if dk > 2 * lt:
        return "dark"
    return "both"


def stroke_width(img, mask, changed=None, thresh=8.0, erode=10):
    """Typical print stroke width in px: the median run of print across rows and columns, away from the limbs (where
    the body's own edge reads as a stroke). video.py's render.print_tophat is about twice it (+1)."""
    g = img.astype(np.float32).mean(-1)
    p = (np.abs(g - ndi.median_filter(g, size=25)) > thresh) & ndi.binary_erosion(mask, iterations=erode)
    runs = []
    for line in list(p[::5]) + list(p.T[::5]):
        x = np.diff(np.r_[0, line.astype(np.int8), 0])
        s, e = np.nonzero(x == 1)[0], np.nonzero(x == -1)[0]
        runs += list((e - s)[(e - s) <= 30])
    return float(np.median(runs)) if runs else 4.0


def _norm_word(w):
    return vqa._fold(w.lower())


def ocr(gray, psm=11, scale=3):
    """Words of 3+ letters tesseract reads in a 0..1 image (dark print on light), lower case, accents folded."""
    import shutil, subprocess, tempfile
    from pathlib import Path
    from PIL import Image
    if shutil.which("tesseract") is None:
        return None
    img = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8))
    img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.png"
        img.save(p)
        out = subprocess.run(["tesseract", str(p), "stdout", "--psm", str(psm)], capture_output=True, text=True).stdout
    return {_norm_word(w) for w in vqa.WORD.findall(out)}


def _local_contrast(L, sigma=10.0):
    """Print against its own body: the body's shading and colour gradient taken out, strokes at unit contrast."""
    d = L - ndi.gaussian_filter(L, sigma)
    s = np.sqrt(ndi.gaussian_filter(d * d, 3 * sigma)) + 1e-3
    return np.clip(0.5 + 0.18 * d / s, 0, 1)


def read_label(ref, mask, band, theta_lim=60.0):
    """The keyframe's label unrolled onto a flat grid through a tube fitted to the matte, contrast-normalised, read by
    tesseract in both polarities and all four orientations (brand names often run along the axis).
    -> (set of words read, tube info)"""
    T, W, Hh, info = surface.from_keyframe(mask, band, theta_lim=75.0)
    tex, _ = surface.unroll(ref, T, W, Hh, theta_lim)
    du = int(T.Ru * np.radians(theta_lim))
    L = tex[:, max(0, int(T.u0) - du): int(T.u0) + du].astype(np.float32).mean(-1) / 255.0
    N = _local_contrast(L)
    words = set()
    for img in (N, 1.0 - N):
        for rot in range(4):
            got = ocr(np.rot90(img, rot))
            if got is None:
                return None, info
            words |= got
    return words, info


def spelling(copy, words):
    """Each copy word: 'exact', 'near' (one edit: an OCR slip or a misspelling, unconfirmed) or 'missing'."""
    out = {}
    for w in copy:
        wn = _norm_word(w)
        if wn in words:
            out[w] = "exact"
        elif len(wn) >= 5 and any(vqa._near(wn, g) for g in words):
            out[w] = "near"
        else:
            out[w] = "missing"
    return out


def verdict(spell, min_exact=0.6):
    """Pass when no word is missing outright from the readable ones, most are exact, and every near miss is listed for
    the second reader. -> ('pass' | 'unconfirmed' | 'fail', summary)"""
    n = max(len(spell), 1)
    ex = sum(v == "exact" for v in spell.values()) / n
    near = [w for w, v in spell.items() if v == "near"]
    miss = [w for w, v in spell.items() if v == "missing"]
    summary = {"exact": round(ex, 2), "near": near, "missing": miss}
    if ex >= min_exact and not near:
        return "pass", summary
    if ex >= min_exact:
        return "unconfirmed", summary
    return "fail", summary


def ocr_boxes(gray, psm=11, scale=3):
    """Words with boxes: [(word, x0, y0, x1, y1, conf)] in the input image's px (tesseract TSV)."""
    import shutil, subprocess, tempfile
    from pathlib import Path
    from PIL import Image
    if shutil.which("tesseract") is None:
        return None
    img = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8))
    img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.png"
        img.save(p)
        out = subprocess.run(["tesseract", str(p), "stdout", "--psm", str(psm), "tsv"], capture_output=True,
                             text=True).stdout
    res = []
    for line in out.splitlines()[1:]:
        c = line.split("\t")
        if len(c) < 12 or not c[11].strip():
            continue
        for w in vqa.WORD.findall(c[11]):
            x, y, ww, hh = (int(v) / scale for v in c[6:10])
            res.append((_norm_word(w), x, y, x + ww, y + hh, float(c[10])))
    return res


def _unrotate_box(b, r, H, W):
    """A box (x0, y0, x1, y1) found in np.rot90(A, r) back in A's coordinates (A is H x W)."""
    x0, y0, x1, y1 = b
    if r == 0:
        return x0, y0, x1, y1
    if r == 1:
        return W - y1, x0, W - y0, x1
    if r == 2:
        return W - x1, H - y1, W - x0, H - y0
    return y0, H - x1, y1, H - x0


def text_boxes(ref, mask, band, copy, theta_lim=75.0, read_lim=60.0, pad=0.8):
    """video.py's qa.text_boxes for a keyframe: boxes around its text, in label px of the label_texture.png the
    engine unrolls from the same keyframe, matte and band. Words are read in four orientations and both inks; only
    words of the label text count; nearby words of one orientation and ink merge into a block.
    -> [{"box": [u0, v0, u1, v1], "rotate": k, "ink": "light" | "dark", "words": [...]}]"""
    T, W, Hh, _ = surface.from_keyframe(mask, band, theta_lim=theta_lim)
    tex, _ = surface.unroll(ref, T, W, Hh, theta_lim)
    du = int(T.Ru * np.radians(read_lim))
    u_off = max(0, int(T.u0) - du)
    L = tex[:, u_off: int(T.u0) + du].astype(np.float32).mean(-1) / 255.0
    N = _local_contrast(L)
    H, Wd = N.shape
    want = {_norm_word(w) for w in copy}
    found = []
    for ink, img in (("dark", N), ("light", 1.0 - N)):
        for r in range(4):
            for w, *b, conf in ocr_boxes(np.rot90(img, r)) or []:
                if w in want or (len(w) >= 5 and any(vqa._near(w, c) for c in want)):
                    found.append((r, ink, w, _unrotate_box(b, r, H, Wd)))
    blocks = []
    for r, _, w, b in sorted(found, key=lambda f: f[0]):
        ink = "any"                                 # measured from the pixels once the blocks are merged
        h = (b[3] - b[1]) if r in (0, 2) else (b[2] - b[0])
        grown = (b[0] - pad * h, b[1] - pad * h, b[2] + pad * h, b[3] + pad * h)
        for blk in blocks:
            B = blk["box"]
            if blk["rotate"] == r and blk["ink"] == ink and not (grown[2] < B[0] or B[2] < grown[0] or
                                                                  grown[3] < B[1] or B[3] < grown[1]):
                blk["box"] = [min(B[0], b[0]), min(B[1], b[1]), max(B[2], b[2]), max(B[3], b[3])]
                blk["words"].append(w)
                break
        else:
            blocks.append({"box": list(b), "rotate": r, "ink": ink, "words": [w]})
    out = []
    for blk in blocks:                              # the same words read twice: keep the block with more words
        x0, y0, x1, y1 = blk["box"]
        sub = L[int(max(0, y0)): int(y1) + 1, int(max(0, x0)): int(x1) + 1]
        dv = sub - np.median(sub)
        blk["ink"] = "light" if np.percentile(dv, 98) > -np.percentile(dv, 2) else "dark"
        m = 6
        out.append({"box": [int(x0 + u_off - m), int(max(0, y0 - m)), int(x1 + u_off + m), int(min(Hh, y1 + m))],
                    "rotate": blk["rotate"] if blk["rotate"] < 3 else -1, "ink": blk["ink"],
                    "words": sorted(set(blk["words"]))})
    out.sort(key=lambda b: -len(b["words"]))
    keep, seen = [], set()
    for b in out:
        if set(b["words"]) - seen:
            keep.append(b)
            seen |= set(b["words"])
    return keep
