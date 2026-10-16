"""Shots, cuts and trims for one product, measured from the clip and the product's matte.

Generated video breaks in four ways, and only one of them is a cut (pitfall V15):
- a cut: another camera set-up, so the background's layout changes; tracks can't cross it and the next shot needs
  its own anchor;
- a pose snap: the product jumps or turns for a frame inside one set-up (the background stays); CoTracker tracks
  through it and the engine's impact and shear handling (V2, V18) treats the frame;
- a morph frame: a blend of two poses; the frames on either side of it still match each other;
- continuous motion that a colour histogram mistakes for a cut (a box rising into view): the product carries on.

The evidence at each boundary t|t+1:
- p1: band-passed correlation of the product between t and t+1, aligned on the matte's centroid;
- p2: the same across one frame (t-1 with t+1, t with t+2): a morph frame breaks p1, not p2;
- layout: correlation of the background's coarse layout (1/8 scale, blurred, product masked out). Generated
  backgrounds are too smooth for fine correlation, but a new set-up moves walls, floors and props;
- iou: the mattes' overlap (a product that jumps away).

The classifier only has to be sure about cuts: anything it isn't sure of stays inside the shot, and the engine's
held run (video.py fit) has the last word. Where that run breaks early, the next shot starts there (autopilot.py).
"""
import numpy as np
from scipy import ndimage as ndi

DEFAULTS = {
    "p_cont": 0.45,          # p1 at or above: the product carries on
    "p_bridge": 0.45,        # p2 at or above: a morph frame inside the shot
    "layout_cut": 0.35,      # background layout correlation below: a new set-up (a cut)
    "layout_same": 0.65,     # at or above: the same set-up, a pose snap; between: uncertain (kept in the shot)
    "iou_cut": 0.30,         # matte overlap below: the product jumped away (a cut)
    "min_area": 0.004,       # matte area as a share of the frame: below, the product isn't there
    "min_width_px": 48,      # the product's width across its axis, full-res px: below, its print can't be read
    "print_min": 0.25,       # print energy against the product's own 90th percentile: below, no label to fix
    "min_run": 3,            # shorter runs of fixable frames are left alone (one or two frames can't be tracked)
    "likely_product": 0.6,   # share of the product's usual area that its colours cover on a frame with no matte
}


def _band(g, lo=1.0, hi=6.0):
    return ndi.gaussian_filter(g, lo) - ndi.gaussian_filter(g, hi)


def _ncc(a, b, m):
    if m.sum() < 50:
        return float("nan")
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-6))


def _aligned_ncc(B, M, s, t):
    """Product correlation between frames s and t, t shifted onto s by the mattes' centroids."""
    a, b = M[s], M[t]
    if a.sum() < 100 or b.sum() < 100:
        return float("nan")
    d = np.round(np.array(ndi.center_of_mass(a)) - np.array(ndi.center_of_mass(b))).astype(int)
    bt = np.roll(B[t], tuple(d), (0, 1))
    m = ndi.binary_erosion(a & np.roll(b, tuple(d), (0, 1)), iterations=2)
    return _ncc(B[s], bt, m)


def downscale(x, f):
    """Area-average by an integer factor (frames T x H x W [x C], or booleans by any-of)."""
    T, h, w = x.shape[:3]
    y = x[:, :h // f * f, :w // f * f].reshape(T, h // f, f, w // f, f, *x.shape[3:])
    return y.any((2, 4)) if x.dtype == bool else y.mean((2, 4)).astype(np.float32)


def frame_stats(F, M, k):
    """Per-frame measurements of the product (F, M at 1/k scale). Lengths are reported in full-res px."""
    T, h, w = M.shape
    G = F.astype(np.float32).mean(-1)
    yx = np.mgrid[:h, :w].reshape(2, -1).T[:, ::-1].astype(float)          # (x, y) per pixel
    out = []
    for t in range(T):
        m = M[t]
        a = int(m.sum())
        st = {"t": t, "area": a / (h * w)}
        if a < 200:
            out.append(st)
            continue
        ys, xs = np.nonzero(m)
        P = np.c_[xs, ys].astype(float)
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        ax = vt[0] * (np.sign(vt[0][1]) or 1.0)
        n = np.array([ax[1], -ax[0]])
        tt, qq = (P - c) @ ax, (P - c) @ n
        length = float(np.ptp(tt))
        width = float(np.ptp(np.clip(qq, *np.percentile(qq, [2, 98]))))
        filled = ndi.binary_fill_holes(m)
        core = ndi.binary_erosion(m, iterations=max(2, int(width * 0.06)))
        g = G[t]
        sharp = float(np.sqrt(np.mean(_band(g, 0.7, 2.0)[core] ** 2))) if core.any() else 0.0
        # print: strokes a few px wide, in the middle 70 % of the width (limbs and their gloss left out)
        Q = ((yx - c) @ n).reshape(h, w)
        mid = core & (np.abs(Q) < 0.35 * width)
        pb = np.abs(_band(g, 1.0, 4.0))
        pr = float(np.mean(pb[mid])) if mid.any() else 0.0
        # where the print sits across the product: 0 = centred on the axis (facing the camera), 1 = at a limb
        off = float(abs((pb[mid] * Q[mid]).sum() / (pb[mid].sum() + 1e-9)) / (0.35 * width + 1e-9)) if mid.any() else 1.0
        # glints: small bright blobs on the product (a sparkle burst); print shows on every frame of a shot alike,
        # so only a frame well above its shot's median counts (keypick)
        gtop = g - ndi.grey_opening(g, size=7)
        lab, nb = ndi.label((gtop > 30) & (g > 200) & ndi.binary_dilation(m, iterations=6))
        sizes = np.bincount(lab.ravel())[1:] if nb else np.array([])
        st["glints"] = int(((sizes >= 2) & (sizes <= 40)).sum())
        st.update({"centre": (c * k).round(1).tolist(), "axis": ax.round(4).tolist(),
                   "length_px": round(length * k, 1), "width_px": round(width * k, 1),
                   "holes": round(float((filled & ~m).sum()) / a, 4), "sharp": round(sharp, 3),
                   "print": round(pr, 3), "print_offset": round(min(off, 1.0), 3)})
        out.append(st)
    return out


def boundaries(F4, M4, F8, M8):
    """Per boundary t|t+1: p1, p2, iou and skip (t with t+2, across frame t+1) at 1/4 scale, layout at 1/8."""
    T = len(M4)
    B = np.stack([_band(g) for g in F4.astype(np.float32).mean(-1)])
    L = np.stack([ndi.gaussian_filter(g, 2.0) for g in F8.astype(np.float32).mean(-1)])
    out = []
    for t in range(T - 1):
        a, b = M4[t], M4[t + 1]
        iou = float((a & b).sum() / max((a | b).sum(), 1))
        p1 = _aligned_ncc(B, M4, t, t + 1)
        bridge = [_aligned_ncc(B, M4, s, u) for s, u in ((t - 1, t + 1), (t, t + 2)) if s >= 0 and u < T]
        bridge = [x for x in bridge if not np.isnan(x)]
        bg = ~ndi.binary_dilation(M8[t] | M8[t + 1], iterations=3)
        skip = _aligned_ncc(B, M4, t, t + 2) if t + 2 < T else float("nan")
        out.append({"t": t, "p1": round(p1, 3), "p2": round(max(bridge), 3) if bridge else float("nan"),
                    "skip": round(skip, 3), "layout": round(_ncc(L[t], L[t + 1], bg), 3), "iou": round(iou, 3)})
    return out


def classify(b, cfg=DEFAULTS):
    """'continuous', 'morph', 'snap' (pose jump in one set-up), 'cut', 'uncertain', or 'absent' (no matte)."""
    p1, p2, lay = b["p1"], b["p2"], b["layout"]
    if np.isnan(p1):
        return "absent"
    if p1 >= cfg["p_cont"]:
        return "continuous"
    if not np.isnan(p2) and p2 >= cfg["p_bridge"]:
        return "morph"
    if lay < cfg["layout_cut"] or b["iou"] < cfg["iou_cut"]:
        return "cut"
    return "snap" if lay >= cfg["layout_same"] else "uncertain"


def product_colours(F, M, bins=8):
    """Colour model of the product against everything else: P(product | colour) per RGB bin."""
    q = (F // (256 // bins)).astype(np.int32)
    idx = (q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]
    fg = np.bincount(idx[M], minlength=bins ** 3).astype(float)
    bg = np.bincount(idx[~M], minlength=bins ** 3).astype(float)
    fg /= max(fg.sum(), 1)
    bg /= max(bg.sum(), 1)
    return fg / (fg + bg + 1e-9), bins


def likely_product(F, model, area_ref):
    """On a frame with no matte: the largest blob of product colours, as a share of the product's usual area."""
    p, bins = model
    q = (F // (256 // bins)).astype(np.int32)
    lik = p[(q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]] > 0.75
    lik = ndi.binary_opening(lik, iterations=2)
    lab, n = ndi.label(lik)
    if not n:
        return 0.0
    return float(np.bincount(lab.ravel())[1:].max()) / max(area_ref, 1)


def fixable(st, ref_print, cfg=DEFAULTS):
    """Why a frame's label isn't worth fixing, or None when it is."""
    if st["area"] < cfg["min_area"]:
        return "no matte"
    if st["width_px"] < cfg["min_width_px"]:
        return "small"
    if st["print"] < cfg["print_min"] * ref_print:
        return "no print"
    return None


def segment(stats, bounds, likely=None, cfg=DEFAULTS):
    """Shots split at cuts, each with its fixable runs (the trims), its morph frames, its uncertain boundaries
    and the frames it skips with the reason. likely[t]: the product's colours on a frame with no matte (V13)."""
    T = len(stats)
    prints = [s["print"] for s in stats if "print" in s]
    ref = float(np.percentile(prints, 90)) if prints else 0.0
    why = [fixable(s, ref, cfg) if "print" in s else "no matte" for s in stats]
    if likely is not None:
        why = ["no matte, product visible" if w == "no matte" and likely[t] >= cfg["likely_product"] else w
               for t, w in enumerate(why)]
    kinds = [classify(b, cfg) for b in bounds]
    cuts = [t + 1 for t, kd in enumerate(kinds) if kd == "cut"]
    edges = [0] + cuts + [T]
    out = []
    for s0, s1 in zip(edges[:-1], edges[1:]):
        if all(why[t] == "no matte" for t in range(s0, s1)):
            continue
        runs, r0 = [], None
        for t in range(s0, s1 + 1):
            ok = t < s1 and why[t] is None
            if ok and r0 is None:
                r0 = t
            if not ok and r0 is not None:
                if t - r0 >= cfg["min_run"]:
                    runs.append([r0, t])
                r0 = None
        skipped = {}
        for t in range(s0, s1):
            if not any(a <= t < b for a, b in runs):
                skipped.setdefault(why[t] or "short run", []).append(t)
        inner = range(s0, s1 - 1)
        out.append({"shot": [s0, s1],
                    "starts_with": kinds[s0 - 1] if s0 > 0 else "clip start",
                    "trims": runs,
                    "morph": [t + 1 for t in inner if kinds[t] == "morph"],
                    "snap": [t + 1 for t in inner if kinds[t] == "snap"],
                    "uncertain": [t + 1 for t in inner if kinds[t] == "uncertain"],
                    "skipped": {k: ranges(v) for k, v in skipped.items()}})
    return out, kinds, why


def ranges(ts):
    """[3, 4, 5, 9] -> '3-5, 9'"""
    ts = sorted(ts)
    out, i = [], 0
    while i < len(ts):
        j = i
        while j + 1 < len(ts) and ts[j + 1] == ts[j] + 1:
            j += 1
        out.append(f"{ts[i]}-{ts[j]}" if j > i else f"{ts[i]}")
        i = j + 1
    return ", ".join(out)


def analyse(F, M, k=2, cfg=DEFAULTS):
    """Everything above, from frames (T x H x W x 3 uint8) and mattes (T x H x W bool) at 1/k of full resolution,
    k in (1, 2, 4); the thresholds were calibrated at k = 2. Lengths in the stats are full-res px.
    -> (stats, bounds, shots, kinds, why)"""
    if k not in (1, 2, 4):
        raise ValueError("analyse: k must be 1, 2 or 4")
    stats = frame_stats(F, M, k)
    F4, M4 = downscale(F, 4 // k), downscale(M, 4 // k)
    F8, M8 = downscale(F, 8 // k), downscale(M, 8 // k)
    bounds = boundaries(F4, M4, F8, M8)
    have = [s for s in stats if s["area"] >= cfg["min_area"]]
    likely = None
    if have:
        model = product_colours(F8, M8)
        area_ref = float(np.median([s["area"] for s in have])) * M8.shape[1] * M8.shape[2]
        likely = [likely_product(F8[t], model, area_ref) if s["area"] < cfg["min_area"] else 1.0
                  for t, s in enumerate(stats)]
        for s, lk in zip(stats, likely):
            s["likely"] = round(lk, 3)
    shots, kinds, why = segment(stats, bounds, likely, cfg)
    return stats, bounds, shots, kinds, why
