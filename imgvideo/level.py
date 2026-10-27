"""render.level: the label's lines level with the product (pitfall V45; ported from the first real run's job scripts
level_fix.py, body_axis.py and check_level.py, 2026-10-17; the user: the label is "higher on the left side and lower on
the right" against the bottle's bottom edge, "especially well" seen "if straightening the product").

The generator can draw the print slanted on the body over a run of frames. The label is tracked on that print, and
the body snap and the glue turn the label's axis onto the body's but keep its lines: a shear. Per frame, on the body's
own upright axis (body_axis: two passes, so a tilted body's rounded base and shoulder don't lean it):
  level   the label's horizontal (the mapping's u direction across the copy block, `at` and `span`) against the
          body's level (the normal to its axis), degrees (+: falls toward the body's right)
  target  the keyframe frame's level (the median over it and `window` frames each side), plus `offset`: degrees, or
          "base": minus the keyframe's own lines against its base, measured on the keyframe image (its print's skew
          by point projection against the matte's bottom edge, both straightened; bowed lines at the axis, V61), so
          the label sits level with the base on every frame, the keyframe's own included
  smooth  the difference over time: a robust local line (Huber weights, `sigma` frames), so the matte's axis noise
          doesn't come through
  shear   the label sheared along the body's axis about the copy block's centre by minus that: its lines turn to the
          target, its verticals (on the limbs already) stay. A smooth image-space Residual over the mapping (after
          the glue)
It runs where the body has a base to level against: the matte's bottom edge, straightened, holds steady over the shot
(its frame-to-frame scatter under `base_mad_max` deg). A squeeze tube's cap end doesn't (it scattered 1.4 deg on the
first real run, a bottle's base 0.15): no correction there, said so in the log and the report ("gate": false runs it
anyway).

The same measures read the delivered frames for the review's level check (lookchecks): `measure` is check_level.py's.
"""
import numpy as np
from scipy.ndimage import binary_erosion, grey_closing, grey_opening, map_coordinates, median_filter

from imgpost import geometry as geo

LEVEL = {
    "sigma": 4.0,          # frames: the level's difference from the target, smoothed over this
    "window": 3,           # frames each side of the keyframe's: the target's median (and the base's for "base")
    "offset": "base",      # deg added to the keyframe's level, or "base": the keyframe's own lines level with its base
    "at": [0.5, 0.75],     # the copy block's centre, as a share of the label's width and height (label px)
    "span": 0.4,           # +- this share of the label's width: the chord its horizontal is read along
    "base_mad_max": 0.5,   # deg: the straightened base's frame-to-frame scatter above this: no base to level against
    "offset_max": 3.0,     # deg: a keyframe offset measured beyond this isn't believed ("base" falls back to 0)
    "min_deg": 0.02,       # deg: smaller corrections leave the frame's mapping as it is
    "gate": True,          # false: level even where the base scatters
}


SQUAT = 3.0     # the matte's eigenvalue ratio under this (about 1.7 x as tall as wide), or wider than tall: squat
ROUND = 1.3     # ... and under this, near round: its axis is the image vertical (imgauto triage TH)
BOW_PX = 6.0    # px: the print's lines bowing more than this across the copy block are read at the body's axis (V61)
BOW_K = 0.004   # 1/px: the bow searched, y - k x^2 (the jar's lines from above: 0.0004-0.0006)


def settings(spec):
    """render.level (true or a dict) merged over LEVEL."""
    c = dict(LEVEL, **(spec if isinstance(spec, dict) else {}))
    if not (c["offset"] == "base" or isinstance(c["offset"], (int, float))):
        raise ValueError(f"render.level.offset {c['offset']!r}: degrees or \"base\"")
    return c


def lin_lum(u8):
    x = u8.astype(np.float32) / 255.0
    x = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return x @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def body_axis(M, lower=0.6, mid=(0.2, 0.8)):
    """The product's axis from its matte, robust to tilt. Horizontal image rows cut a tilted body obliquely near its
    rounded base and shoulder, so their midpoints lean the axis (4 deg off on half of a tilted bottle's frames). Two
    passes: the principal axis of the matte's lower `lower` share along its own length (the body without a pump or
    cap); then, straightened on it, a line through the limbs' midpoints across the body's middle (`mid` of its length),
    twice. A squat matte (a jar, a balm: eigenvalue ratio under SQUAT, or wider than tall) starts from the axis nearest
    the image vertical, and the second pass keeps it (V53, V61). -> (centre point, unit axis direction pointing down the
    body, angle from image vertical in degrees, half-width px)"""
    ys, xs = np.nonzero(M)
    P = np.stack([xs, ys], 1).astype(float)
    c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    d = V[:, np.argmax(w)]
    squat = w.max() < SQUAT * w.min() or abs(d[0]) > abs(d[1])
    if squat:
        # a squat product (a jar, a balm: about as wide as tall, or wider): its largest axis can run across the body,
        # and the lower 60 % along that is one side of it. The axis nearest the image vertical instead, the vertical
        # itself on a near-round matte, kept by the second pass (imgauto's triage, V53; V61)
        d = np.array([0.0, 1.0]) if w.max() < ROUND * w.min() else V[:, np.argmax(np.abs(V[1]))]
    if d[1] < 0:
        d = -d
    a = (P - c) @ d
    keep = a > a.min() + (1 - lower) * np.ptp(a)
    Pb = P[keep]
    c = Pb.mean(0)
    if not squat:
        w, V = np.linalg.eigh(np.cov((Pb - c).T))
        d = V[:, np.argmax(w)]
    d = d if d[1] > 0 else -d
    for _ in range(2):
        n = np.array([d[1], -d[0]])
        a, x = (Pb - c) @ d, (Pb - c) @ n
        lo, hi = np.quantile(a, mid[0]), np.quantile(a, mid[1])
        mids, at, half = [], [], []
        for b0 in np.arange(lo, hi, 2.0):
            s = (a >= b0) & (a < b0 + 2.0)
            if s.sum() > 4:
                mids.append(0.5 * (x[s].min() + x[s].max()))
                half.append(0.5 * (x[s].max() - x[s].min()))
                at.append(b0 + 1.0)
        k, m0 = np.polyfit(at, mids, 1)
        c = c + m0 * n
        d = d + k * n
        d = d / np.linalg.norm(d)
    return c, d, float(np.degrees(np.arctan2(d[0], d[1]))), float(np.median(half))


def straighten(M, F=None, axis=None):
    """The matte (and the frame's linear luminance) resampled with the body's axis vertical, 1 px steps. -> (L or
    None, matte, half-width, axis angle)"""
    c, d, ang, hw = axis if axis is not None else body_axis(M)
    n = np.array([d[1], -d[0]])
    ys, xs = np.nonzero(M)
    al = (np.stack([xs, ys], 1) - c) @ d
    A = np.arange(al.min() - 10, al.max() + 10)
    C = np.arange(-hw - 15, hw + 15)
    X = c[0] + A[:, None] * d[0] + C[None, :] * n[0]
    Y = c[1] + A[:, None] * d[1] + C[None, :] * n[1]
    L = map_coordinates(lin_lum(F), [Y, X], order=1) if F is not None else None
    m = map_coordinates(M.astype(np.float32), [Y, X], order=1) > 0.5
    return L, m, hw, ang


def base_line(m):
    """The body's bottom edge on a straightened matte: its lowest row per column over the middle 70 %, a robust line.
    -> (slope, intercept), or None"""
    W = m.shape[1]
    mid = np.arange(int(0.15 * W), int(0.85 * W))
    if len(mid) < 4:
        return None
    bot = np.array([np.nonzero(m[:, c])[0].max() if m[:, c].any() else np.nan for c in mid], float)
    ok = np.isfinite(bot)
    if ok.sum() < 4:
        return None
    k, c = np.polyfit(mid[ok], bot[ok], 1)
    r = bot[ok] - (k * mid[ok] + c)
    keep = np.abs(r) < 3 * 1.4826 * np.median(np.abs(r)) + 0.5
    return tuple(np.polyfit(mid[ok][keep], bot[ok][keep], 1))


def base_deg(M, axis=None):
    """The base's angle against the body's level, degrees (+: lower on the right), or None."""
    _, m, _, _ = straighten(M, axis=axis)
    b = base_line(m)
    return None if b is None else float(np.degrees(np.arctan(b[0])))


def scatter(x):
    """Frame-to-frame scatter of a per-frame angle: 1.4826 x the median |x - its 5-frame running median|."""
    x = np.asarray(x, float)
    if len(x) < 5:
        return None
    return float(1.4826 * np.median(np.abs(x - median_filter(x, 5, mode="nearest"))))


def print_mask(L, inner, polarity="light", thresh=0.03):
    """The print's strokes on straightened luminance: a top-hat 15 (light ink) or a black top-hat (dark ink)."""
    th = L - grey_opening(L, size=(15, 15)) if polarity != "dark" else grey_closing(L, size=(15, 15)) - L
    return (th > thresh) & inner, th


def _sharpness(q, wv, kern):
    """measure's profile sharpness (0.25 px bins from q.min() - 1, smoothed), by bincount"""
    h = np.bincount(((q - q.min() + 1.0) * 4.0).astype(np.int64), weights=wv)
    return float((np.convolve(h, kern, mode="same") ** 2).sum())


def _bowed(yy, xr, wv, kern, ang):
    """The print's lines through a parabola about the body's axis (x = xr), where they bow more than BOW_PX across the
    block: their angle (deg) at the axis, or None (straight enough: the straight read stands)."""
    x2 = xr ** 2
    dk = BOW_K / 40
    t_ = np.radians(ang)
    ks = np.arange(-BOW_K, BOW_K + 1e-12, dk)
    k0 = ks[int(np.argmax([_sharpness((yy - k * x2) * np.cos(t_) - xr * np.sin(t_), wv, kern) for k in ks]))]
    if abs(k0) * x2.max() <= BOW_PX:
        return None

    def search(ks, angs):
        best = (-1.0, 0.0, 0.0)
        for k in ks:
            yk = yy - k * x2
            for a in angs:
                t_ = np.radians(a)
                v = _sharpness(yk * np.cos(t_) - xr * np.sin(t_), wv, kern)
                if v > best[0]:
                    best = (v, float(a), float(k))
        return best[1], best[2]
    a, k = search(np.arange(k0 - 10 * dk, k0 + 10 * dk + 1e-12, dk), np.arange(-6.0, 6.0001, 0.25))
    a, _ = search(np.arange(k - dk, k + dk + 1e-12, dk / 10), np.arange(max(-6.0, a - 0.3), min(6.0, a + 0.3) + 1e-9, 0.05))
    return round(a, 2)


def measure(F, M, polarity="light"):
    """Is the label level with the product (check_level.py)? The frame straightened on the body's axis; there:
      text  the copy's lines: the angle within +-6 deg (0.05 steps) whose row profile of the print's own pixel
            positions (projected, 0.25 px bins, smoothed 1 px: image rotation and whole-pixel bins both bias toward 0)
            is sharpest, over the copy block (the print right of its first quarter: a vertical wordmark stays out);
            lines that bow over BOW_PX across the block (a squat product seen from above), their angle at the body's
            axis through a parabola (V61)
      base  the matte's bottom edge (base_line)
      gaps  the bottom print line to the base at the copy block's two ends (px)
    -> {"text_deg", "base_deg", "level_deg" (text - base, +: the label's right side lower than the body's),
        "gap_left_px", "gap_right_px", "axis_deg", "block_px" (the copy block's width)}, or None"""
    L, m, hw, axis = straighten(M, F)
    inner = binary_erosion(m, iterations=6)
    ink, th = print_mask(L, inner, polarity)
    cols = np.nonzero(ink.any(0))[0]
    if len(cols) < 20:
        return None
    c_lo = cols.min() + int(0.25 * (cols.max() - cols.min()))
    c_hi = cols.max() - 3
    blk = ink[:, c_lo:c_hi]
    rws = np.nonzero(blk.any(1))[0]
    if len(rws) < 10:
        return None
    r0, r1 = rws.min(), rws.max()
    yy, xx = np.nonzero(blk[r0:r1 + 1])
    wv = th[r0:r1 + 1, c_lo:c_hi][yy, xx]
    best, ang = -1.0, 0.0
    kern = np.exp(-0.5 * (np.arange(-8, 9) / 4.0) ** 2)
    for a in np.arange(-6.0, 6.0001, 0.05):
        t_ = np.radians(a)
        q = yy * np.cos(t_) - xx * np.sin(t_)
        h, _ = np.histogram(q, bins=np.arange(q.min() - 1, q.max() + 1, 0.25), weights=wv)
        h = np.convolve(h, kern, mode="same")
        v = float((h ** 2).sum())
        if v > best:
            best, ang = v, a
    text = float(ang)                              # + : the lines fall to the right
    # bowed lines (a squat product seen from above: the jar's sag 40-50 px across the copy block): a straight line reads
    # the chord of each arc over the block, degrees off its fall (V61). Read where each arc runs level on a round body,
    # at the axis, through a parabola: q = (y - k x^2) cos a - x sin a, x from the axis column
    xr = (xx + c_lo - (hw + 15.0)).astype(float)
    bow = _bowed(yy, xr, wv, kern, text)
    if bow is not None:
        text = bow
    b = base_line(m)
    if b is None:
        return None
    k, c = b
    base = float(np.degrees(np.arctan(k)))

    def low(c0, c1):
        rr = np.nonzero(ink[:, c0:c1].any(1))[0]
        return rr.max() if len(rr) else np.nan
    w = max((c_hi - c_lo) // 5, 4)
    gl = (k * (c_lo + w / 2) + c) - low(c_lo, c_lo + w)
    gr = (k * (c_hi - w / 2) + c) - low(c_hi - w, c_hi)
    return {"text_deg": round(text, 2), "base_deg": round(base, 2), "level_deg": round(text - base, 2),
            "gap_left_px": round(float(gl), 1), "gap_right_px": round(float(gr), 1), "axis_deg": round(axis, 2),
            "block_px": int(c_hi - c_lo)}


def mapping_level(H, d, uv, du):
    """The label's horizontal at uv (label px) against the body's level, degrees (+: falls toward the body's right)."""
    n = np.array([d[1], -d[0]])
    v = (geo.apply_H(H, uv + [[du, 0]]) - geo.apply_H(H, uv - [[du, 0]]))[0]
    return float(np.degrees(np.arctan2(v @ d, v @ n)))


def robust_smooth(tt, y, sigma):
    """A zero-lag local line per frame: Gaussian weights over `sigma` frames, Huber weights on frames (4 passes)."""
    out = np.empty_like(y)
    for i, t in enumerate(tt):
        w0 = np.exp(-0.5 * ((tt - t) / sigma) ** 2)
        X = np.c_[np.ones(len(tt)), tt - t]
        w = w0.copy()
        for _ in range(4):
            beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
            r = np.abs(y - X @ beta)
            sc = 1.4826 * np.median(r[w0 > 0.05]) + 0.05
            w = w0 * np.minimum(1.0, 1.5 * sc / np.maximum(r, 1e-9))
        out[i] = beta[0]
    return out


def copy_point(LP, cfg):
    """The copy block's centre (label px, 1 x 2) and the half-chord its horizontal is read along."""
    (u0, v0), (u1, v1) = LP.min(0), LP.max(0)
    at = cfg["at"]
    return np.array([[u0 + at[0] * (u1 - u0), v0 + at[1] * (v1 - v0)]]), float(cfg["span"]) * (u1 - u0)


def keyframe_level(kf_img, mattes, k, window, polarity="light"):
    """The keyframe's own lines against its base: the print's skew on the keyframe image (on its frame's matte), the
    base the median over the keyframe's frame and `window` frames each side (mattes: {frame: bool matte}). -> (deg,
    report) or (None, why)"""
    if k not in mattes:
        return None, "no matte on the keyframe's frame"
    m = measure(kf_img, mattes[k], polarity)
    if m is None:
        return None, "no copy block found on the keyframe"
    bases = [b for t in range(k - window, k + window + 1) if t in mattes for b in [base_deg(mattes[t])] if b is not None]
    if not bases:
        return None, "no base found round the keyframe's frame"
    base = float(np.median(bases))
    return m["text_deg"] - base, {"text_deg": m["text_deg"], "base_deg": round(base, 2),
                                  "level_deg": round(m["text_deg"] - base, 2)}


def level(maps, matte, LP, k, cfg=None, kf_img=None, polarity="light"):
    """maps {frame: the mapping (label px -> frame px), glued or placed}; matte(t) the product's matte on frame t; LP
    the label's outline (label px); k the keyframe's frame (in maps); kf_img the keyframe image (offset "base").
    -> ({frame: levelled mapping} for the frames it moves, report)"""
    c = settings(cfg)
    ts = sorted(t for t in maps if matte(t).any())
    rep = {"keyframe": k, "sigma": c["sigma"]}
    if k not in ts:
        return {}, dict(rep, skipped=f"no matte on the keyframe's frame {k}")
    M = {t: matte(t) for t in ts}
    ax = {t: body_axis(M[t]) for t in ts}
    bd = [base_deg(M[t], ax[t]) for t in ts]
    bdv = [b for b in bd if b is not None]
    sc = scatter(bdv)
    rep["base_scatter_deg"] = None if sc is None else round(sc, 3)
    if c["gate"] and (sc is None or sc > float(c["base_mad_max"])):
        return {}, dict(rep, skipped=f"no steady base to level against: the straightened base scatters "
                                     f"{'?' if sc is None else round(sc, 2)} deg frame to frame (base_mad_max "
                                     f"{c['base_mad_max']})")
    uv, du = copy_point(LP, c)
    lv = np.array([mapping_level(maps[t], ax[t][1], uv, du) for t in ts])
    ref = float(np.median([lv[i] for i, t in enumerate(ts) if abs(t - k) <= int(c["window"])]))
    off, kfm = c["offset"], None
    if off == "base":
        if kf_img is None:
            off, kfm = 0.0, {"skipped": "no keyframe image (a re-acquired shot): offset 0"}
        else:
            kl, kfm = keyframe_level(kf_img, M, k, int(c["window"]), polarity)
            if kl is None:
                off, kfm = 0.0, {"skipped": f"{kfm}: offset 0"}
            elif abs(kl) > float(c["offset_max"]):
                off, kfm = 0.0, dict(kfm, skipped=f"the keyframe's lines read {kl:+.2f} deg off its base, past offset_max "
                                                  f"{c['offset_max']}: offset 0")
            else:
                off = -kl
    off = float(off)
    target = ref + off
    tt = np.array(ts, float)
    dl = robust_smooth(tt, lv - target, float(c["sigma"]))
    rep.update(keyframe_level_deg=round(ref, 3), offset_deg=round(off, 3), **({"keyframe_vs_base": kfm} if kfm else {}),
               per_frame={t: {"level_vs_target_deg": round(float(lv[i] - target), 2),
                              "correction_deg": round(float(-dl[i]), 2)} for i, t in enumerate(ts)})
    cc = np.abs(dl)
    rep["correction_deg_median"], rep["correction_deg_max"] = round(float(np.median(cc)), 2), round(float(cc.max()), 2)
    (u0, v0), (u1, v1) = LP.min(0), LP.max(0)
    mu, mv = 0.12 * (u1 - u0), 0.08 * (v1 - v0)
    G = np.stack(np.meshgrid(np.linspace(u0 - mu, u1 + mu, 9), np.linspace(v0 - mv, v1 + mv, 13)), -1).reshape(-1, 2)
    out = {}
    for i, t in enumerate(ts):
        a = -np.radians(dl[i])
        if abs(a) < np.radians(float(c["min_deg"])):
            continue
        H = maps[t]
        d = ax[t][1]
        n = np.array([d[1], -d[0]])
        P = geo.apply_H(H, G)
        pc = geo.apply_H(H, uv)[0]
        # a shear along the axis: a point's shift along it grows with its distance across the body, so the label's
        # lines turn by a while lines along the axis stay put
        D = (np.tan(a) * ((P - pc) @ n))[:, None] * d[None, :]
        out[t] = geo.Residual(H, P, D, float(np.ptp(P, axis=0).max() / 6.0), 1e-3)
    rep["moved"] = len(out)
    return out, rep
