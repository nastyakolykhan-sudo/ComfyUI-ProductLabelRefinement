"""Copy rows (2026-11-20): on the ink route, a block of the packshot's copy sized to the generated print's own rows.

The ink route lays the packshot's whole print through one mapping (the fit, and the registration over it when kept), so
the print keeps the packshot's proportions. A generator can draw one block of the copy smaller than the packshot lays it
out while drawing the rest at about its size (a tube's lines of copy drawn ~20 % narrower and ~10 % shorter than the
packshot's, its vertical wordmark not): one registration can't follow both, and the new lines ran past the generated
rows and off the body's edge (the user: "too stretched horizontally... keep it the same while making it legible").

Here each block of three or more lines of text in a column (textline's lines and blocks, read on the packshot) is
compared with the generated print pulled back into the packshot's frame (textline.pull_back): the block's two scales
(along its rows and down the label, about its first line's start) and its shift that correlate it best with the
generated print. A block whose lines run COPY_LONG or more longer than the generated rows, when that fit is sure, goes
on at the fitted size; any other block, and the rest of the print, stays as the ink route laid it (so a block the
generator drew at the packshot's size is untouched).

The check (copy_long, ink.dark_checks): the new block's lines against the generated rows under them, across the body;
over COPY_LONG longer: FAIL.

Pure numpy + scipy; arrays are packshot px unless named otherwise."""
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import correlate

COPY_MIN_LINES = 3        # a block of this many lines of text or more is copy that can be sized on its own
COPY_LONG = 1.10          # its lines this many times the generated rows' length or longer: sized to them (and FAIL)
COPY_SURE = 0.6           # ... when the fit correlates this well (along the rows: and better than as laid)
COPY_GAIN = 0.05          # ... a block's rows: this much better than as laid
COPY_OFF = 0.75           # ... or a block off its generated rows by this many line heights (down the label or along
                          # the rows) goes onto them, at the fitted size
SCALES_ROW = np.round(np.arange(0.70, 1.1001, 0.02), 3)    # along the rows
SCALES_DOWN = np.round(np.arange(0.80, 1.1001, 0.02), 3)   # down the label
COPY_ASPECT = 0.15        # the two scales at most this share apart (the letters keep about their shape)
RUN_LEVEL = 0.25          # a generated row's run: its columns over this share of the row's print
REACH = 2.0               # line heights: the block's shift tried each way
EDGE_SCENE = 0.5          # the true print's widest strokes (scene px): the frame's print read this far inside the body
EDGE_IN = 1.0             # line heights: the generated print read this far inside the band (its edge is the body's
                          # outline, read as print)


def frame_print(crop_u8, in_band, cfg, stroke):
    """The frame's print map in the work box (textline.place_print_level's, at the frame's own size)."""
    from . import images as im
    from . import ink
    dark = cfg["polarity"] == "dark"
    Lf = im.lum(im.srgb_to_lin(im.to_float(crop_u8)))
    B = in_band > 0.5
    so = max(7, int(np.ceil(1.5 * stroke)) | 1) if stroke else 15
    # clear of the body's edge: its rim's glow reads as print (a run of "print" to the limb on every row)
    Bi = ndi.binary_erosion(B, iterations=max(2, int(round(EDGE_SCENE * (stroke or 6.0)))))
    f_px = int(cfg.get("_px_scale") or 1)
    if f_px > 1:
        H_, W_ = Lf.shape
        h_, w_ = -(-H_ // f_px), -(-W_ // f_px)
        Lp = np.pad(Lf, ((0, h_ * f_px - H_), (0, w_ * f_px - W_)), mode="edge")
        Bp = np.pad(B, ((0, h_ * f_px - H_), (0, w_ * f_px - W_)), mode="edge")
        Ls = Lp.reshape(h_, f_px, w_, f_px).mean((1, 3))
        Bs = Bp.reshape(h_, f_px, w_, f_px).mean((1, 3)) > 0.5
        so_s = max(7, int(np.ceil(so / f_px)) | 1)
        Pq = ink.print_map(Ls, Bs, so_s, dark)[0] * Bs
        return ndi.zoom(Pq, f_px, order=1)[:H_, :W_] * Bi
    return ink.print_map(Lf, B, so, dark)[0] * Bi


def _ncc_map(G, T):
    """Zero-mean normalised correlation of template T at each valid place on G."""
    n = T.size
    Tz = T - T.mean()
    tn = np.sqrt((Tz * Tz).sum())
    if tn <= 1e-9:
        return None
    one = np.ones_like(T)
    s1 = correlate(G, one, mode="valid", method="fft")
    s2 = correlate(G * G, one, mode="valid", method="fft")
    z = correlate(G, Tz, mode="valid", method="fft")
    var = np.maximum(s2 - s1 * s1 / n, 1e-9)
    return z / (np.sqrt(var) * tn)


def _scaled(Rb, sx, sy, ax, ay):
    """Rb scaled by sx along the rows and sy down, about (ax, ay): out(x, y) = Rb(ax + (x - ax) / sx, ay + (y - ay) / sy)."""
    return ndi.affine_transform(Rb, np.diag([1.0 / sy, 1.0 / sx]), offset=(ay * (1 - 1 / sy), ax * (1 - 1 / sx)),
                                order=1, mode="constant")


def _fit_block(Rb, G, box, h, ax, ay):
    """The block's best (ncc, sx, sy, dx, dy) on G within REACH line heights; and the block as laid (1, 1, 0, 0)."""
    x0, y0, x1, y1 = box
    r = int(np.ceil(REACH * h))
    best, before = (-2.0, 1.0, 1.0, 0.0, 0.0), -2.0
    H_, W_ = G.shape
    for sy in SCALES_DOWN:
        for sx in SCALES_ROW:
            if abs(np.log(sx / sy)) > np.log(1 + COPY_ASPECT) + 1e-9:
                continue
            Ts = _scaled(Rb, sx, sy, ax, ay)
            # the template: the scaled block's box with a margin of r (empty print round it counts too)
            ty, tx = np.nonzero(Ts > 0.02)
            if len(ty) < 20:
                continue
            a0, a1 = max(ty.min() - r, 0), min(ty.max() + 1 + r, H_)
            b0, b1 = max(tx.min() - r, 0), min(tx.max() + 1 + r, W_)
            T = Ts[a0:a1, b0:b1]
            g0, g1 = max(a0 - r, 0), min(a1 + r, H_)
            c0, c1 = max(b0 - r, 0), min(b1 + r, W_)
            Gw = G[g0:g1, c0:c1]
            if Gw.shape[0] < T.shape[0] or Gw.shape[1] < T.shape[1]:
                continue
            z = _ncc_map(Gw, T)
            if z is None:
                continue
            iy, ix = np.mgrid[0:z.shape[0], 0:z.shape[1]]
            dy_, dx_ = g0 + iy - a0, c0 + ix - b0
            z = np.where((np.abs(dy_) <= r) & (np.abs(dx_) <= r), z, -2.0)
            k = np.unravel_index(int(np.argmax(z)), z.shape)
            if z[k] > best[0]:
                best = (float(z[k]), float(sx), float(sy), float(dx_[k]), float(dy_[k]))
            if sx == 1.0 and sy == 1.0:
                at0 = (dy_ == 0) & (dx_ == 0)
                if at0.any():
                    before = float(z[at0][0])
    return best, before


def _cols(M, x0, x1, t, b):
    a_, b_ = max(int(np.floor(t)), 0), max(int(np.ceil(b)) + 1, 0)
    out = np.zeros(max(x1 - x0, 1))
    if b_ > a_:
        xa, xb = max(x0, 0), min(x1, M.shape[1])
        if xb > xa:
            out[xa - x0:xb - x0] = M[a_:b_, xa:xb].sum(0)
    return out


def _fit_rows(R, G, lines, h, ax, sy, dy, ay, scales=None):
    """Along the rows: the block's scale about its anchor ax and its shift, from each line's column profile (its own
    rows on the packshot) against the generated print's (the rows it lands on, sy and dy down the label), summed over
    the lines (line ends and word gaps: a blurred block's 2-D correlation is ruled by its rows, and a block running
    15 % past its generated lines read 1.06). -> (sx, dx, ncc, ncc at 1 and 0)"""
    pad = int(np.ceil(3 * h))
    x0 = int(min(L["x0"] for L in lines)) - pad
    x1 = int(max(L["x1"] for L in lines)) + pad
    xs = np.arange(x0, x1, dtype=float)
    T, Gs = [], []
    for L in lines:
        T.append(_cols(R, x0, x1, L["top"], L["bot"]))
        Gs.append(_cols(G, x0, x1, ay + dy + sy * (L["top"] - ay), ay + dy + sy * (L["bot"] - ay)))
    Gc = np.concatenate(Gs)
    reach = REACH * h

    def score(sx, dx):
        t = np.concatenate([np.interp((xs - ax - dx) / sx + ax, xs, p, left=0.0, right=0.0) for p in T])
        return _ncc1(t, Gc)
    best = (-2.0, 1.0, 0.0)
    for sx in (SCALES_ROW if scales is None else scales):
        if abs(np.log(sx / sy)) > np.log(1 + COPY_ASPECT) + 1e-9:
            continue
        for dx in np.arange(-reach, reach + 1e-6, 1.0):
            c = score(sx, dx)
            if c > best[0]:
                best = (c, float(sx), float(dx))
    c0 = score(1.0, 0.0)
    return best[1], best[2], best[0], c0


def _ncc1(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else -1.0


def _gen_rows(P, lines, A, band, h):
    """The generated rows under each placed line: its rows (as placed by A), the generated print's column profile over
    them, and its run along them (columns over RUN_LEVEL of the profile's top, joined across gaps under a line height,
    the run overlapping the new line most); and whether that run ends inside the band, a line height clear of its
    edge (one reaching it may run on out of sight). -> [(k, new length, generated length, inside)]"""
    out = []
    bx = np.nonzero(band.any(0))[0]
    for L in lines:
        x0n = A[0, 0] * L["x0"] + A[0, 2]
        x1n = A[0, 0] * L["x1"] + A[0, 2]
        t = A[1, 1] * L["top"] + A[1, 2]
        b = A[1, 1] * L["bot"] + A[1, 2]
        a_, b_ = max(int(np.floor(t)), 0), int(np.ceil(b)) + 1
        if b_ <= a_:
            continue
        cols = ndi.uniform_filter1d(P[a_:b_, :].sum(0), 3)
        top = np.percentile(cols[int(max(x0n, 0)):int(x1n) + 1], 90) if x1n > x0n else 0.0
        if top <= 0:
            continue
        on = np.nonzero(cols > RUN_LEVEL * top)[0]
        if not len(on):
            continue
        runs, s_ = [], on[0]
        for p, q in zip(on[:-1], on[1:]):
            if q - p > h:
                runs.append((s_, p))
                s_ = q
        runs.append((s_, on[-1]))
        ov = [min(e, x1n) - max(s0, x0n) for s0, e in runs]
        j = int(np.argmax(ov))
        if ov[j] <= 0:
            continue
        gs, ge = runs[j]
        inside = gs > bx.min() + h and ge < bx.max() - h
        out.append((int(L["k"]), round(float(x1n - x0n), 1), float(ge - gs + 1), bool(inside)))
    return out


def fit(cov_r, band, Hm, roi, crop_u8, in_band, cfg, stroke):
    """The packshot's blocks of copy against the generated print. -> ([{"mask": packshot bool, "A": 3 x 3}] to lay
    at their fitted size, report)"""
    from . import textline as tl
    P_s = frame_print(crop_u8, in_band, cfg, stroke)
    P_b = tl.pull_back(P_s, Hm, roi, band)
    tl_r = tl.text_lines(cov_r, band, sharp=True)
    lr = tl_r["lines"]
    comp = tl._comp_lines(cov_r, tl_r)
    out, rep = [], {"blocks": []}
    for blk in tl._blocks(lr):
        if len(blk) < COPY_MIN_LINES:
            continue
        lines0 = sorted([lr[k] for k in blk], key=lambda L: L["cy"])
        h = float(np.median([L["h"] for L in lines0]))
        how = tl._alignment(lines0)
        mask_full = np.isin(comp, [k + 1 for k in blk])
        # the block's neighbourhood only (the scales tried and the reach, the body's edge round it), its own px
        X0, X1 = min(L["x0"] for L in lines0), max(L["x1"] for L in lines0)
        Y0, Y1 = min(L["top"] for L in lines0), max(L["bot"] for L in lines0)
        er = max(1, int(round(EDGE_IN * h)))
        m = int(np.ceil(REACH * h + 0.35 * max(X1 - X0, Y1 - Y0) + 3 * h)) + er + 4
        wy0, wx0 = max(int(Y0) - m, 0), max(int(X0) - m, 0)
        wy1, wx1 = min(int(np.ceil(Y1)) + m, cov_r.shape[0]), min(int(X1) + m, cov_r.shape[1])
        win = (slice(wy0, wy1), slice(wx0, wx1))
        lines = [dict(L, x0=L["x0"] - wx0, x1=L["x1"] - wx0, top=L["top"] - wy0, bot=L["bot"] - wy0,
                      cy=L["cy"] - wy0) for L in lines0]
        ax = float(np.mean([tl._anchor(L, how) for L in lines]))
        ay = float(lines[0]["top"])
        sig = max(0.25 * h, 1.0)
        Rb = ndi.gaussian_filter(cov_r[win] * mask_full[win], sig)
        x0b, x1b = min(L["x0"] for L in lines), max(L["x1"] for L in lines)
        y0b, y1b = min(L["top"] for L in lines), max(L["bot"] for L in lines)
        bw = np.pad(band, er + 1)[wy0:wy1 + 2 * er + 2, wx0:wx1 + 2 * er + 2]
        inner = ndi.binary_erosion(bw, iterations=er)[er + 1:-(er + 1), er + 1:-(er + 1)]
        G = ndi.gaussian_filter(P_b[win] * inner, sig)
        # down the label: the 2-D fit (rows and their pitch); along the rows: the lines' column profiles
        (ncc2, sx2, sy, dx2, dy), before2 = _fit_block(Rb, G, (x0b, y0b, x1b, y1b), h, ax, ay)
        sx, dx, ncc, before = _fit_rows(Rb, G, lines, h, ax, sy, dy, ay)
        # long: its lines run COPY_LONG or more past the generated rows (the column fit sure of it); off: it sits
        # COPY_OFF line heights or more off them (the 2-D fit sure of it: the rows correlate much better there)
        long_ = ncc >= COPY_SURE and ncc > before and sx <= 1.0 / COPY_LONG
        off = ncc2 >= COPY_SURE and ncc2 >= before2 + COPY_GAIN and max(abs(dx2), abs(dy)) >= COPY_OFF * h
        sure = long_ or off
        if off and not long_:
            # on its rows at its own size along them (its lines run as long as the generated ones)
            sx = 1.0
            dx = _fit_rows(Rb, G, lines, h, ax, sy, dy, ay, scales=(1.0,))[1]
        A = np.array([[sx, 0.0, ax + dx - sx * ax], [0.0, sy, ay + dy - sy * ay], [0.0, 0.0, 1.0]])
        Tw = np.array([[1.0, 0.0, wx0], [0.0, 1.0, wy0], [0.0, 0.0, 1.0]])
        A_full = Tw @ A @ np.linalg.inv(Tw)
        r = {"lines": len(blk), "line_h": round(h, 1), "box": [int(X0), int(Y0), int(X1), int(Y1)],
             "align": how, "scale_xy": [sx, sy], "shift": [dx, dy], "ncc_laid": round(before, 3),
             "ncc_fit": round(ncc, 3), "block_2d": {"scale_xy": [sx2, sy], "shift": [dx2, dy], "ncc_laid": round(before2, 3),
                                                    "ncc_fit": round(ncc2, 3)}, "sure": bool(sure), "long": bool(long_), "off_rows": bool(off),
             "sized": bool(sure)}
        Pi = P_b[win] * inner
        r["rows_laid"] = _gen_rows(Pi, lines, np.eye(3), bw[er + 1:-(er + 1), er + 1:-(er + 1)], h)
        if sure:
            out.append({"mask": ndi.binary_dilation(mask_full, iterations=1) & (cov_r > 0), "A": A_full})
            r["rows_sized"] = _gen_rows(Pi, lines, A, bw[er + 1:-(er + 1), er + 1:-(er + 1)], h)
            # the check's measure: the block as it now goes on, fitted again along its rows
            Ai = np.linalg.inv(A)
            Rs = ndi.affine_transform(Rb, np.array([[Ai[1, 1], Ai[1, 0]], [Ai[0, 1], Ai[0, 0]]]),
                                      offset=(Ai[1, 2], Ai[0, 2]), order=1, mode="constant")
            Ls = [dict(L, x0=A[0, 0] * L["x0"] + A[0, 2], x1=A[0, 0] * L["x1"] + A[0, 2],
                       top=A[1, 1] * L["top"] + A[1, 2], bot=A[1, 1] * L["bot"] + A[1, 2]) for L in lines]
            ay2 = A[1, 1] * ay + A[1, 2]
            sx2, dx2, n2, b2 = _fit_rows(Rs, G, Ls, h * sy, A[0, 0] * ax + A[0, 2], 1.0, 0.0, ay2)
            r["after"] = {"scale_x": sx2, "shift_x": dx2, "ncc_laid": round(b2, 3), "ncc_fit": round(n2, 3)}
            fin = (sx2, n2)
        else:
            fin = (sx, ncc)
        # the new lines against the generated rows, as they go on: 1 / the fit's scale along the rows (when it
        # correlates COPY_SURE or better)
        r["long_ratio"] = round(1.0 / fin[0], 3) if fin[1] >= COPY_SURE else None
        rep["blocks"].append(r)
    return out, rep


def lay(cov_r, band, Hm, roi, kw, fits):
    """The print laid again with the sized blocks on their own placement (textline.LineMap over the mapping) and the
    rest as before. -> cover in the work box"""
    from . import ink
    from . import textline as tl
    kw = dict(kw, extra=None)
    rest = np.ones(cov_r.shape)
    for f in fits:
        rest = np.where(f["mask"], 0.0, rest)
    cov = ink.warp_cover(cov_r * rest, band, Hm, roi, **kw)[0]
    for f in fits:
        cov = cov + ink.warp_cover(cov_r * f["mask"], band, tl.LineMap(Hm, f["A"]), roi, **kw)[0]
    return np.clip(cov, 0, 1)


def check(rep):
    """copy_long: a block of copy whose new lines run COPY_LONG or more longer than the generated rows under them (the
    fit sure of it), as it goes on: FAIL (the user, 2026-11-20: "too stretched horizontally over the tube... keep it
    the same while making it legible"). -> audit findings"""
    out = []
    for r in (rep or {}).get("blocks") or []:
        q = r.get("long_ratio")
        if q is None:
            continue
        bad = q >= COPY_LONG
        out.append({"check": "copy_long", "level": "FAIL" if bad else "INFO", "share": q,
                    "text": f"the new copy's {r['lines']} lines run {100 * (q - 1):+.0f} % against the generated rows "
                            f"under them" + ((f" (sized to them from {100 * (1 / r['scale_xy'][0] - 1):+.0f} %)" if r.get("long")
                                              else f" (moved onto them, {r['shift'][1]:+.0f} px down the label)")
                                             if r.get("sized") else "") +
                            (f": over {100 * (COPY_LONG - 1):.0f} %, the copy runs past its generated rows (off the "
                             "body's edge)" if bad else "")})
    return out


def ratio(rows):
    """The new lines' length against the generated rows under them: the median over the rows the generated print
    ends inside the band (a run to its edge may go on out of sight), else over all. -> ratio or None"""
    ins = [n / g for _, n, g, i in rows if i and g > 0]
    alls = [n / g for _, n, g, i in rows if g > 0]
    v = ins if len(ins) >= 2 else alls
    return float(np.median(v)) if len(v) >= 2 else None
