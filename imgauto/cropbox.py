"""Crop windows around a product in the master, and the exact paste back.

The window logic is ComfyUI-MaskAspectCrop's ImageCropByMaskToAspect (padding in px around each frame's mask box,
"adaptive" = the standard aspect closest to the batch's envelope), with two differences that keep paste-back exact:
- native scale: the node resizes every crop to a base resolution, and a resampled crop can't go back bit for bit;
- one window per shot ("fixed"), integer offsets, even sizes (H.264 previews). When the product travels so far that
  a fixed window would be much larger than the product's own box, "follow" keeps the size and moves the window by
  whole pixels along a smoothed path.
"""
import math

import numpy as np
from scipy import ndimage as ndi

ASPECTS = {"16:9": 16 / 9, "4:3": 4 / 3, "1:1": 1.0, "3:4": 3 / 4, "9:16": 9 / 16, "21:9": 21 / 9}


def box(mask, k=1):
    """Bounding box (x0, y0, x1, y1), exclusive, in full-res px of a matte at 1/k scale, or None."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return xs.min() * k, ys.min() * k, (xs.max() + 1) * k, (ys.max() + 1) * k


def _closest(ratio):
    return min(ASPECTS, key=lambda n: abs(math.log(ratio) - math.log(ASPECTS[n])))


def _fit(w, h, r, W, H):
    """Smallest w' x h' with aspect r containing w x h, scaled down to fit W x H (the node's _fit_to_ratio)."""
    cw, ch = (w, w / r) if w / h > r else (h * r, h)
    if cw > W or ch > H:
        s = min(W / cw, H / ch)
        cw, ch = cw * s, ch * s
    return int(round(cw)), int(round(ch))


def _even(v, lo=2):
    return max(lo, int(v) // 2 * 2)


def _place(c, size, limit):
    """Integer start of a window of `size` centred at c, kept inside [0, limit)."""
    return int(min(max(round(c - size / 2), 0), limit - size))


def shot_window(boxes, W, H, padding=33, aspect="adaptive", follow_ratio=1.6):
    """One window for a shot's frames. boxes: per frame (x0, y0, x1, y1) or None.
    -> {"mode": "fixed" | "follow", "aspect", "size": [w, h], "offsets": [[x, y] per frame]} (every frame gets an
    offset; frames without a box hold the nearest one's)."""
    have = [b for b in boxes if b is not None]
    if not have:
        return None
    pad = [(max(b[0] - padding, 0), max(b[1] - padding, 0), min(b[2] + padding, W), min(b[3] + padding, H))
           for b in have]
    env_w = max(b[2] - b[0] for b in pad)
    env_h = max(b[3] - b[1] for b in pad)
    name = _closest(env_w / env_h) if aspect == "adaptive" else aspect
    r = ASPECTS[name]
    U = (min(b[0] for b in pad), min(b[1] for b in pad), max(b[2] for b in pad), max(b[3] for b in pad))
    uw, uh = _fit(U[2] - U[0], U[3] - U[1], r, W, H)
    fw, fh = _fit(env_w, env_h, r, W, H)
    mode = "fixed" if uw * uh <= follow_ratio * fw * fh else "follow"
    w, h = (uw, uh) if mode == "fixed" else (fw, fh)
    w, h = _even(min(w, W)), _even(min(h, H))
    cs = []
    last = None
    for b in boxes:
        if b is not None:
            last = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        cs.append(last)
    first = next(c for c in cs if c is not None)
    cs = [c if c is not None else first for c in cs]
    if mode == "fixed":
        x, y = _place((U[0] + U[2]) / 2, w, W), _place((U[1] + U[3]) / 2, h, H)
        offs = [[x, y]] * len(boxes)
    else:
        C = np.array(cs, float)
        C = np.c_[ndi.gaussian_filter1d(C[:, 0], 2.0, mode="nearest"), ndi.gaussian_filter1d(C[:, 1], 2.0, mode="nearest")]
        offs = [[_place(cx, w, W), _place(cy, h, H)] for cx, cy in C]
    contains = all(b is None or (o[0] <= b[0] and o[1] <= b[1] and b[2] <= o[0] + w and b[3] <= o[1] + h)
                   for b, o in zip(boxes, offs))
    return {"mode": mode, "aspect": name, "size": [w, h], "offsets": offs, "contains_product": bool(contains)}


def crop(frame, win, i):
    x, y = win["offsets"][i]
    w, h = win["size"]
    return frame[y:y + h, x:x + w]


def paste(master, refined, source, win, i, keep=None):
    """Frame i's refined crop back into the master frame: only the pixels the fix changed (refined != its source
    crop), and of those only where `keep` (crop-sized bool, e.g. the product's matte minus the mattes in front of it)
    allows. -> (new master frame, changed-pixel mask in master px)"""
    x, y = win["offsets"][i]
    w, h = win["size"]
    out = master.copy()
    ch = (refined != source).any(-1)
    if keep is not None:
        ch &= keep
    reg = out[y:y + h, x:x + w]
    reg[ch] = refined[ch]
    full = np.zeros(master.shape[:2], bool)
    full[y:y + h, x:x + w] = ch
    return out, full
