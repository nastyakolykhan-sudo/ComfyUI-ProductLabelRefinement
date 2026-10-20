"""align.bow measured on the scene: a round body seen from above or below bows every line across it (pitfall 95).

The product's matte gives the body's centre, axis and half-width along it; the body's own level edges (a shoulder or
cap seam, a collar, a base) give the bow: each is traced column by column across the body and fitted as
t(s) = t0 + b * sqrt(1 - s^2 / w^2), t along the axis, s across it, w the matte's half-width at that seam.
geometry.Bow lays the profile [[t, b], ...] over the fitted mapping (fit() reads job align.bow).
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from . import images as im


def body_axis(mask):
    """Centre and unit axis (pointing down the image) of a matte, and its normal. The principal axis decides whether
    the body stands or lies (a near-round matte, eigenvalues within 1.5x, stands); the axis is then the midline of
    the matte's two sides: the midpoints of its extent across, row by row (column by column lying) over the middle
    70 % of its length, fitted as a line. A body about as wide as it is tall has no clear principal axis: a jar seen
    from above came out 1.07 deg off it, and its print ran skewed against the lid (its sides' midline: -0.1 deg)."""
    ys, xs = np.nonzero(mask)
    P = np.c_[xs, ys].astype(float)
    c = P.mean(0)
    _, sv, vt = np.linalg.svd(P - c, full_matrices=False)
    ax = vt[0] * (np.sign(vt[0][1]) or 1.0)
    upright = abs(ax[1]) >= abs(ax[0]) or sv[0] < 1.5 * sv[1]
    along, across = (ys, xs) if upright else (xs, ys)
    lo, hi = np.percentile(along, [15, 85])
    keys = np.arange(int(np.ceil(lo)), int(hi) + 1)
    if len(keys) >= 10:
        order = np.argsort(along, kind="stable")
        a_s, x_s = along[order], across[order]
        i0, i1 = np.searchsorted(a_s, keys, "left"), np.searchsorted(a_s, keys, "right")
        ok = i1 > i0
        mid = np.array([(x_s[a:b].min() + x_s[a:b].max()) / 2.0 for a, b in zip(i0[ok], i1[ok])])
        k1, k0 = np.polyfit(keys[ok].astype(float), mid, 1)
        d = np.array([k1, 1.0]) if upright else np.array([1.0, k1])
        ax = d / np.linalg.norm(d)
        ax = ax * (np.sign(ax[1]) or 1.0)
        ca = c[1] if upright else c[0]
        c = np.array([k0 + k1 * ca, ca]) if upright else np.array([ca, k0 + k1 * ca])
    return c, ax, np.array([ax[1], -ax[0]])


def _project(mask, c, ax, n):
    """The matte's pixels as (t along the axis, s across it) from c (written out: numpy's matmul on some BLAS
    builds warns falsely on these)"""
    ys, xs = np.nonzero(mask)
    dx, dy = xs - c[0], ys - c[1]
    return dx * ax[0] + dy * ax[1], dx * n[0] + dy * n[1]


def half_widths(mask, c, ax, n, step=10.0):
    """[[t, w], ...]: the matte's half-width across the axis every `step` px along it (its pixels within 1 px of t)."""
    t, q = _project(mask, c, ax, n)
    out = []
    for tt in np.arange(np.ceil(t.min() / step) * step, t.max() + 1e-6, step):
        sel = np.abs(t - tt) < 1.0
        if sel.sum() >= 2:
            out.append([float(tt), round(float(q[sel].max() - q[sel].min()) / 2.0, 2)])
    return out


def seam(img_u8, mask, t0, c, ax, n, w, search=6.0, reach=0.85, step=0.25):
    """One level edge across the body near t0 (px along the axis): its position per column s (every px across,
    |s| <= reach * w), the strongest luminance step along the axis within +-search px, signed as the median column's.
    -> (s, t) arrays of the columns found"""
    L = gaussian_filter(im.lum(im.to_float(img_u8)), 0.7)
    ss = np.arange(-np.floor(reach * w), np.floor(reach * w) + 1)
    tt = np.arange(t0 - search, t0 + search + 1e-9, step)
    X = c[0] + tt[None, :] * ax[0] + ss[:, None] * n[0]
    Y = c[1] + tt[None, :] * ax[1] + ss[:, None] * n[1]
    prof = map_coordinates(L, [Y, X], order=1, mode="nearest")
    inside = map_coordinates(mask.astype(float), [Y, X], order=1, mode="constant") > 0.5
    g = np.gradient(prof, step, axis=1)
    sign = float(np.sign(np.median(g[np.arange(len(ss)), np.abs(g).argmax(1)]))) or 1.0
    g = g * sign
    k = g.argmax(1)
    ok = (k > 0) & (k < len(tt) - 1) & (g[np.arange(len(ss)), k] > 0.25 * np.median(g.max(1)))
    ok &= inside.any(1)
    i = np.arange(len(ss))[ok]
    k = k[ok]
    gm, g0, gp = g[i, k - 1], g[i, k], g[i, k + 1]
    off = 0.5 * (gm - gp) / np.where(np.abs(gm - 2 * g0 + gp) > 1e-12, gm - 2 * g0 + gp, 1e-12)
    return ss[i], tt[k] + np.clip(off, -1, 1) * step


def fit_seam(s, t, w, outlier=1.5):
    """t(s) = t0 + b * sqrt(1 - s^2 / w^2), least squares with outliers (> outlier px) dropped -> (t0, b, info)"""
    keep = np.ones(len(s), bool)
    for _ in range(3):
        A = np.c_[np.ones(keep.sum()), np.sqrt(np.clip(1 - (s[keep] / w) ** 2, 0, 1))]
        (t0, b), *_ = np.linalg.lstsq(A, t[keep], rcond=None)
        r = t - (t0 + b * np.sqrt(np.clip(1 - (s / w) ** 2, 0, 1)))
        new = np.abs(r) <= outlier
        if new.sum() < 6 or (new == keep).all():
            break
        keep = new
    return float(t0), float(b), {"columns": int(keep.sum()), "of": int(len(s)),
                                 "rms_px": round(float(np.sqrt(np.mean(r[keep] ** 2))), 2)}


def parse(text):
    """A bow written as text (run.py draft --bow, Auto Draft's bow) -> align.bow without the matte's numbers, or None.
    Tokens, t in px along the product's axis from its matte's centre (+ down the image):
      t:b       b given at t ("-150:0" a squeezed tube's flat crimp)
      t         measure the level edge (seam) across the body near t
      t/w       ... on a narrower ring of half-width w (a collar)
      t/matte   ... the matte's own end near t (a base against the background)
    All given: {"b": [[t, b], ...]}; else {"b": "auto", "seams": [...]}."""
    toks = str(text or "").replace(",", " ").split()
    if not toks:
        return None
    seams, given = [], True
    for tok in toks:
        try:
            if ":" in tok:
                t, b = tok.split(":")
                seams.append([float(t), float(b)])
            elif "/" in tok:
                t, w = tok.split("/")
                seams.append({"t": float(t), "edge": "matte"} if w.strip().lower() == "matte" else {"t": float(t), "w": float(w)})
                given = False
            else:
                seams.append(float(tok))
                given = False
        except ValueError:
            raise ValueError(f"bow {tok!r}: write t:b (given), t (a seam to measure), t/w (a seam on a ring of "
                             "half-width w) or t/matte (the matte's end)") from None
    return {"b": sorted(seams)} if given else {"b": "auto", "seams": seams}


def matte_ends(mask):
    """The bow of a round body seen from above or below, from its matte's own two ends (SAM 3's outline is cleaner
    than any seam inside the body): the top end's outline is the far side of the top face and the base's the near
    side of the base, both arcs of the same view, so the bow is the base's, at both ends with its sign (+: seen from
    above). A jar's measured 49 px at its top and 53 at its base, its lid seam about 66: a print registered without
    them bowed 108 to 120 and ran skewed. -> ([[t_top, b], [t_base, b]], report) or (None, report) when an end
    can't be read"""
    c, ax, n = body_axis(mask)
    hw = np.asarray(half_widths(mask, c, ax, n), float)
    if len(hw) < 4:
        return None, {"why": "the matte is too small"}
    t_top, t_base = float(hw[0, 0]), float(hw[-1, 0])
    out, rep = [], {}
    for name, t0 in (("top", t_top + 5.0), ("base", t_base - 5.0)):
        w = float(np.interp(t0 + (30.0 if name == "top" else -30.0), hw[:, 0], hw[:, 1]))
        s, t = contour(mask, t0, c, ax, n, w)
        if len(s) < 6:
            return None, {"why": f"the matte's {name} end has too few columns"}
        _, b, info = fit_seam(s, t, w)
        out.append([t0, b])
        rep[name] = dict(info, t=round(t0, 1), b=round(b, 2), half_width=round(w, 1))
    sgn = np.sign(out[1][1]) or 1.0
    prof = [[out[0][0], round(sgn * abs(out[0][1]), 2)], [out[1][0], round(sgn * abs(out[1][1]), 2)]]
    return prof, rep


def contour(mask, t0, c, ax, n, w, reach=0.85, depth=30.0):
    """The matte's own end near t0 (a base or a top seen against the background): per column across the body, the
    matte's last pixel along the axis past t0 - depth (first pixel before t0 + depth for t0 < 0). -> (s, t)"""
    T, S = _project(mask, c, ax, n)
    sgn = 1.0 if t0 >= 0 else -1.0
    sel = (T - (t0 - sgn * depth)) * sgn > 0
    cols = np.round(S[sel]).astype(int)
    Ts = T[sel] * sgn
    keys = np.unique(cols)
    keys = keys[np.abs(keys) <= reach * w]
    return keys.astype(float), np.array([Ts[cols == k].max() for k in keys]) * sgn


def resolve(bow, img_u8=None, mask=None):
    """Complete a job's align.bow from the scene: centre, axis and half_widths from the product's matte (scene-sized,
    bool) when missing; b "auto" measured at each of `seams`, px along the axis from the matte's centre:
      t                      a level edge across the body (shoulder or cap seam): the luminance step, fitted with
                             the matte's half-width there;
      {"t": t, "w": w}       a seam on a narrower ring (a collar: w its half-width), fitted with w and scaled to the
                             matte's half-width at t (b grows with the width at the same view);
      {"t": t, "edge": "matte"}  the body's end against the background (a base): the matte's own contour;
      [t, b]                 given, e.g. [crimp_t, 0] for a squeezed tube's flat crimp.
    On the first real job's box (two products printed on it, seen from above) these measured 11.7 at a bottle's
    shoulder (collar w 45) and 16 at its base against the hand's 10 and 15, and 7.7 at a tube's cap seam against 10:
    a start, checked by eye.
    -> (bow with numbers only, report)"""
    out = {k: v for k, v in bow.items() if k not in ("mask", "seams")}      # "from" ("matte") stays
    rep = {}
    need = [k for k in ("centre", "axis", "half_widths") if out.get(k) is None]
    if need or out.get("b") == "auto":
        if mask is None or not mask.any():
            raise ValueError("align.bow: " + ", ".join(need or ["b auto"]) + " are measured from the product's matte: "
                             "give align.bow.mask (or old_silhouette as a mask)")
        c, ax, n = body_axis(mask)
        if out.get("centre") is None:
            out["centre"] = c.tolist()
        if out.get("axis") is None:
            out["axis"] = ax.tolist()
        c, ax = np.asarray(out["centre"], float), np.asarray(out["axis"], float)
        ax = ax / np.linalg.norm(ax)
        n = np.array([ax[1], -ax[0]])
        if out.get("half_widths") is None:
            out["half_widths"] = half_widths(mask, c, ax, n)
    if out.get("b") == "auto":
        if img_u8 is None or not bow.get("seams"):
            raise ValueError("align.bow.b 'auto' needs align.bow.seams: [t, ...] px along the axis of the body's level "
                             "edges (shoulder, collar, cap seam, base), and the scene")
        tw = np.asarray(out["half_widths"], float)
        prof, rep["seams"] = [], []
        for sm in bow["seams"]:
            if isinstance(sm, (list, tuple)):
                prof.append([float(sm[0]), float(sm[1])])
                rep["seams"].append({"t": float(sm[0]), "b": float(sm[1]), "given": True})
                continue
            sm = sm if isinstance(sm, dict) else {"t": sm}
            t_s = float(sm["t"])
            w_t = float(np.interp(t_s, tw[:, 0], tw[:, 1]))
            if sm.get("edge") == "matte":
                w = float(np.interp(t_s - 30.0 * np.sign(t_s or 1.0), tw[:, 0], tw[:, 1]))
                s, t = contour(mask, t_s, c, ax, n, w)
                scale = 1.0
            else:
                w = float(sm.get("w", w_t))
                s, t = seam(img_u8, mask, t_s, c, ax, n, w)
                scale = w_t / w
            if len(s) < 6:
                raise ValueError(f"align.bow: the level edge near t {t_s} wasn't found across the body "
                                 f"({len(s)} columns); check its t, or give [t, b]")
            t0, b, info = fit_seam(s, t, w)
            prof.append([t_s, round(b * scale, 2)])
            rep["seams"].append(dict(info, t=t_s, b=round(b * scale, 2), b_ring=round(b, 2), t0=round(t0, 2),
                                     half_width=round(w, 2), edge=sm.get("edge", "seam")))
        out["b"] = sorted(prof)
    return out, rep
