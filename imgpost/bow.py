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


def long_lines(L, inside, frame, size=19, span=0.75, rms=1.5, tilt=5.0):
    """The long, level, smooth lines of dark print across a body (ink.line_bows on its print map: rules, a band's
    edges, full-width lines of text): at least `span` of the body's width (its median half-width, frame
    half_widths), rms under `rms` px, tilt under `tilt` deg; one line of text read as two runs a few px apart keeps
    the longer. L: linear luminance; inside: the body (bool); frame: line_bows's. -> lines, top to bottom"""
    from . import ink as ik
    W = float(np.median(np.asarray(frame["half_widths"], float)[:, 1]))
    P = ik.print_map(L, inside, size, True)[0] * inside
    out = []
    for l in sorted((l for l in ik.line_bows(P, inside, frame)
                     if l["s_span"] >= span * 2 * W and l["rms"] < rms and abs(l["tilt_deg"]) < tilt),
                    key=lambda l: l["t"]):
        if out and l["t"] - out[-1]["t"] < 6:
            if l["n"] > out[-1]["n"]:
                out[-1] = l
        else:
            out.append(l)
    return out


def pair_lines(a, b, reach=40.0, tol=8.0):
    """Each line of b paired with a's line at its place in the layout: one offset along the axis for all (the one most
    lines meet within 6 px; the fit follows the outline, the generator drew the print a few px off), then the nearest
    within `tol` px. -> (offset px, [(line of a, line of b), ...])"""
    if not a or not b:
        return 0.0, []
    ta = np.array([l["t"] for l in a])
    best = (-1, 0.0, 0.0)
    for off in sorted({round(float(x - l["t"]), 1) for x in ta for l in b if abs(x - l["t"]) <= reach}):
        d = [np.abs(ta - (l["t"] + off)).min() for l in b]
        hits, miss = sum(x <= 6 for x in d), float(np.sum(np.minimum(d, 6)))
        if hits > best[0] or (hits == best[0] and miss < best[2]):
            best = (hits, off, miss)
    off = best[1]
    pairs = []
    for l in b:
        j = int(np.argmin(np.abs(ta - (l["t"] + off))))
        if abs(ta[j] - (l["t"] + off)) <= tol:
            pairs.append((a[j], l))
    return off, pairs


def from_print(scene_u8, mask, ref_u8, refmask, H, size=19, span=0.75, rms=1.5, tilt=5.0):
    """The body's bow as the generated product draws it, for a packshot warped whole (pitfall 101: a squeeze tube seen
    from above, warped flat by an affine fit: "the label doesn't follow the curved tube's shape"). The long, level,
    smooth lines across the body (ink.line_bows on its dark print: rules, a band's edges, full-width lines of text; at
    least `span` of the body's width, rms under `rms` px, tilt under `tilt` deg) in the frame and in the packshot as
    the fit warps it, both in the frame's body frame. Each packshot line is paired with the frame's line at its place
    in the layout (one offset for all: the fit follows the outline, the generator drew the print a few px off), so a
    rule meets the rule and a line of text the same line, its letters' own lean on both sides. A pair's bow less the
    packshot line's own is what the warp must add there. The matte's ends pair with the warped packshot outline's
    where they keep the lines' trend (lower parts bow more seen from above): a tube's flat crimp does; a base SAM 3
    cut flat along the cap's reflection, under a cap whose own bottom curves 25 px, doesn't. Those knots are smoothed:
    the bow over the body's half-width there, fitted as a monotone run down the body (a round section seen from above
    bows more the lower it sits, relative to its width), each knot weighted by its lines' length and smoothness.
    mask: the product's matte (bool, scene-sized); refmask: outline.RefMask; H: the draft's fit (reference -> scene).
    -> (profile [[t, b], ...] for align.bow, or None; report)"""
    from . import warp as wp
    c, ax, n = body_axis(mask)
    hw = half_widths(mask, c, ax, n)
    if len(hw) < 4:
        return None, {"why": "the matte is too small"}
    ys, xs = np.nonzero(mask)
    pad = int(size) + 4
    x0, y0 = max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0)
    x1, y1 = min(int(xs.max()) + pad + 1, mask.shape[1]), min(int(ys.max()) + pad + 1, mask.shape[0])
    frame = {"centre": [float(c[0]) - x0, float(c[1]) - y0], "axis": [float(ax[0]), float(ax[1])], "half_widths": hw}

    gen = long_lines(im.lum(im.srgb_to_lin(im.to_float(scene_u8[y0:y1, x0:x1]))), mask[y0:y1, x0:x1], frame, size,
                     span, rms, tilt)
    rgb, cov, _ = wp.warp(im.srgb_to_lin(im.to_float(ref_u8)), H, (x0, y0, x1, y1), refmask, ss=2)
    pk = long_lines(im.lum(rgb), cov > 0.5, frame, size, span, rms, tilt)
    rep = {"frame_lines": [{"t": l["t"], "b": l["b"], "rms": l["rms"]} for l in gen],
           "packshot_lines": [{"t": l["t"], "b": l["b"], "rms": l["rms"]} for l in pk]}
    if not gen or not pk:
        return None, dict(rep, why="no long level line in the " + ("frame's product" if not gen else "warped packshot"))
    tg = np.array([l["t"] for l in gen])
    off, pairs = pair_lines(gen, pk)
    knots = [[p["t"], g["b"] - p["b"], min(g["n"], p["n"]) / (np.hypot(g["rms"], p["rms"]) + 0.25) ** 2,
              f"line {g['t']:+.0f}"] for g, p in pairs]   # [t, b, weight, what]
    rep.update(offset_px=off, pairs=[[k[0], round(k[1], 2), k[3]] for k in knots])
    # the matte's ends, where they keep the lines' trend (seen from above, lower parts bow more)
    ends, _ = matte_ends(mask)
    pends, _ = matte_ends(cov > 0.5)
    used = []
    if ends is not None and pends is not None and len(knots) >= 2:
        fb = [gen[int(np.argmin(np.abs(tg - (k[0] + off))))]["b"] for k in knots]
        rising = fb[-1] >= fb[0]
        for i, name in ((0, "top"), (1, "base")):
            t_end, b_end = ends[i]
            keeps = (b_end <= min(fb)) if (i == 0) == rising else (b_end >= max(fb))
            if keeps:
                knots.append([t_end, b_end - pends[i][1], 40.0 / 0.3 ** 2, f"matte {name}"])
                used.append(name)
    rep["matte_ends"] = {"frame": ends, "packshot": pends, "used": used}
    if not knots:
        return None, dict(rep, why="no packshot line pairs with one of the frame's")
    knots.sort(key=lambda k: k[0])
    t = np.array([k[0] for k in knots]); b = np.array([k[1] for k in knots]); wt = np.array([k[2] for k in knots])
    # seen from above, a round section's bow over its half-width grows down the body (from below it shrinks): that
    # ratio, fitted as a monotone run (pooled where a knot breaks it: a line of text's letters lean a few px)
    hwa = np.asarray(hw, float)
    half = np.interp(t, hwa[:, 0], hwa[:, 1])
    r = b / half
    gr = np.array([l["b"] for l in gen]) / np.interp(tg, hwa[:, 0], hwa[:, 1])
    rising = len(gr) < 2 or np.polyfit(tg, gr, 1)[0] >= 0
    fit = _monotone(r if rising else -r, wt) * (1 if rising else -1)
    prof = [[round(float(tt), 1), round(float(f * h), 2)] for tt, f, h in zip(t, fit, half)]
    rep.update(knots=[[round(float(k[0]), 1), round(float(k[1]), 2), k[3]] for k in knots], rising=bool(rising),
               profile=prof)
    return prof, rep


def _monotone(y, w):
    """Weighted least squares fit of y that never decreases (pool adjacent violators)."""
    blocks = []                                         # [value, weight, count]
    for v, ww in zip(np.asarray(y, float), np.asarray(w, float)):
        blocks.append([v, ww, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    return np.concatenate([[v] * c for v, _, c in blocks])


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
