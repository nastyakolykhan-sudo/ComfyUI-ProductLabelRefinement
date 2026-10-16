"""Automatic review checks: the failures the user kept finding by eye in review rounds 2-5 of example1 and example2
(2026-10-15/16), flagged per frame before delivery. Each was calibrated on those rounds: it fires on the frames the user
marked and stays quiet on the frames they approved (references/video/workflow.md, "Automatic checks").

  edges   the label's edge against the product. A seam: the output steps across the label's edge (its sides, under
          the cap or crimp, at the base) where the source doesn't, held for a third of a second. Or a gap: the
          label's own limb sits inside the body's silhouette, leaving a strip of the source between them.
  ghost   the old garbled print showing through the new label (an ease blend, a relight or sparkle re-add that
          follows a misregistered source).
  shear   a shear spike against the neighbouring frames of the shot (a morph frame the fit followed).
  vanish  the fix stops while the product and its print are still in view (occlusion, an exit, lost tracks).
  switch  a hard switch at a fix's end (no ease into the source), or the label jumping against the body where two
          shots meet without a cut.
  bend    the label's side snapped onto a matte edge that isn't the product's limb (an occluder's straight cut, an edge
          shared with something in front), so the print bends along it (V32).

Inputs are the delivered frames themselves (source, output, matte) plus, per refined frame, the mapping that rendered
it and its job's artwork and still label. Nothing here changes a render."""
import math

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import (binary_dilation, binary_erosion, binary_fill_holes, distance_transform_edt, gaussian_filter,
                           map_coordinates)

from imgpost import geometry as geo
from imgpost import images as im

from . import fit as vf
from . import labelspace as ls

# defaults, calibrated on the example rounds (override per job with vjob "review": {...})
DEFAULTS = {
    "seam_levels": 5.5,          # p75 of the excess step across the label's edge, sRGB levels, on two sides at once,
    "seam_strong": 7.0,          # or on one side above this (each a 3-frame median) ...
    "seam_run_s": 0.25,          # ... held this long, so a one-frame blip doesn't count
    "gap_px": 12.0,              # the label's limb inside the matte: p90 over its rows, at least this many px
    "gap_frac": 0.045,           # ... and this share of the label's width
    "ghost_pc": 0.3,             # median partial correlation of the output with the old print, given the truth
    "ghost_cluster": 6,          # or this many adjacent blocks (32 label px each) above 0.5: a local double image
    "ghost_min_blocks": 8,       # blocks where the source's print is legible enough to tell
    "shear_deg": 0.9,            # shear jump against the shot's frames either side (V18's threshold)
    "shear_deg_end": 1.8,        # ... at a shot's first or last frames, where the neighbours are on one side only
    "spike_frac": 0.03,          # shape off the neighbours' average beyond a similarity, share of the label's width
    "vanish_matte": 0.25,        # an unrefined frame still shows the product: matte at least this share ...
    "vanish_detail": 0.6,        # ... and print detail at least this share of the refined frames next to it (median of 5)
    "switch_fix": 0.65,          # the boundary frame's fix at least this share of its run's: not eased
    "jump_px": 3.0,              # the old print, read through both frames' mappings, moves this far in label space
    "bend_slope": 0.6,           # a matte edge running this steep to the axis (px across per px along) over 2 row pairs
    "bend_ends": 0.15,           # ... away from the band's ends (where a crimp or a cap steps the outline on purpose),
    "bend_moved": 4.0,           # ... on a side the snap moved this far: the snap followed an occluder, not the limb
    "cut_hist": 0.25,            # colour-histogram distance above it (whole frame), or a matte overlap below cut_iou:
    "cut_iou": 0.5,              # a cut (cuts measured 0.28-0.42, continuous boundaries up to 0.20); "cuts": [frames]
}                                # lists them by hand (a cut between t - 1 and t is frame t)


# ---------------------------------------------------------------- image measures

def srgb_lum(u8):
    return (0.2126 * u8[..., 0] + 0.7152 * u8[..., 1] + 0.0722 * u8[..., 2]).astype(np.float32)


def lin_lum(u8):
    return im.lum(im.srgb_to_lin(im.to_float(u8))).astype(np.float32)


def label_region(ch, min_px=300):
    """Changed px -> the label's region: gaps closed, holes filled, specks dropped."""
    if ch.sum() < min_px:
        return np.zeros_like(ch)
    r = binary_fill_holes(ndi.binary_closing(ch, iterations=3))
    lab, n = ndi.label(r)
    if n > 1:
        sz = ndi.sum(r, lab, range(1, n + 1))
        r = np.isin(lab, 1 + np.nonzero(sz >= max(min_px, 0.05 * sz.max()))[0])
    return r


def _axes(region):
    """Centre, long axis (pointing down the frame), its normal (pointing left), and the region's extent along it."""
    ys, xs = np.nonzero(region)
    c = np.array([xs.mean(), ys.mean()])
    X = np.c_[xs - c[0], ys - c[1]].astype(np.float64)
    cov = np.array([[np.mean(X[:, 0] ** 2), np.mean(X[:, 0] * X[:, 1])], [np.mean(X[:, 0] * X[:, 1]), np.mean(X[:, 1] ** 2)]])
    a = np.linalg.eigh(cov)[1][:, 1]
    a = -a if a[1] < 0 else a
    n = np.array([a[1], -a[0]])
    n = -n if n[0] > 0 else n
    s = X[:, 0] * a[0] + X[:, 1] * a[1]
    return c, a, n, float(s.min()), float(s.max())


def seam_steps(src_u8, out_u8, matte, lab, k0=2, k1=6, edge=4, every=3):
    """How far the output steps across the label's edge beyond the source's own step there (sRGB levels), per side
    relative to the label's axis: {side: p75}. Only edge points well inside the matte (its own edge isn't a seam)."""
    dm = distance_transform_edt(matte)
    bd = lab & ~binary_erosion(lab) & (dm > edge + k1)
    ys, xs = np.nonzero(bd)
    if len(xs) < 20:
        return {}
    ys, xs = ys[::every], xs[::every]
    f = gaussian_filter(lab.astype(np.float32), 2.0)
    gy, gx = ndi.sobel(f, 0)[ys, xs], ndi.sobel(f, 1)[ys, xs]
    nn = np.hypot(gx, gy) + 1e-9
    nx, ny = -gx / nn, -gy / nn                                      # outward
    Lo, Ls = srgb_lum(out_u8), srgb_lum(src_u8)
    ks = np.arange(k0, k1 + 1, dtype=np.float32)

    def side_mean(img, sgn):
        X = xs[:, None] + sgn * ks[None] * nx[:, None]
        Y = ys[:, None] + sgn * ks[None] * ny[:, None]
        return map_coordinates(img, [Y.ravel(), X.ravel()], order=1, mode="nearest").reshape(X.shape).mean(1)
    ex = np.clip(np.abs(side_mean(Lo, -1) - side_mean(Lo, 1)) - np.abs(side_mean(Ls, -1) - side_mean(Ls, 1)), 0, None)
    c, a, n, s0, s1 = _axes(lab)
    P = np.c_[xs - c[0], ys - c[1]].astype(np.float64)
    al, ac = P[:, 0] * a[0] + P[:, 1] * a[1], P[:, 0] * n[0] + P[:, 1] * n[1]
    span = s1 - s0
    side = np.where(al < s0 + 0.04 * span, "top", np.where(al > s1 - 0.04 * span, "bottom", np.where(ac > 0, "left", "right")))
    return {sd: round(float(np.percentile(ex[side == sd], 75)), 2) for sd in ("left", "right", "top", "bottom")
            if (side == sd).sum() >= 10}


def detail(u8, m, sigma=1.2):
    """Print detail: std of the high-passed linear luminance over m."""
    if m.sum() < 200:
        return 0.0
    L = lin_lum(u8)
    return float((L - gaussian_filter(L, sigma))[m].std())


def hist_dist(a, b, bins=8):
    def h(x):
        q = (x[::4, ::4] // (256 // bins)).reshape(-1, 3).astype(np.int32)
        return np.bincount(q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2], minlength=bins ** 3) / len(q)
    return 0.5 * float(np.abs(h(a) - h(b)).sum())


def matte_overlap(m0, m1):
    """Overlap of two mattes with their centroids aligned (1 = the same silhouette)."""
    if m0.sum() < 100 or m1.sum() < 100:
        return 0.0
    y0, x0 = ndi.center_of_mass(m0)
    y1, x1 = ndi.center_of_mass(m1)
    sh = ndi.shift(m1.astype(np.float32), (y0 - y1, x0 - x1), order=0) > 0.5
    return float((m0 & sh).sum() / max((m0 | sh).sum(), 1))


def frame_measures(src_u8, out_u8, matte, prev=None):
    """The image measures of one delivered frame (prev: the previous frame's (src, matte) for the cut test)."""
    ch = np.any(out_u8 != src_u8, -1)
    r = {"changed_px": int(ch.sum()), "matte_px": int(matte.sum()), "detail": round(detail(src_u8, binary_erosion(matte, iterations=4)), 6)}
    lab = label_region(ch)
    if lab.sum() > 2000:
        r["seam"] = seam_steps(src_u8, out_u8, matte, lab)
        r["fix_levels"] = round(float(np.abs(srgb_lum(out_u8) - srgb_lum(src_u8))[lab].mean()), 3)
    if prev is not None:
        r["hist"] = round(hist_dist(prev[0], src_u8), 4)
        r["overlap"] = round(matte_overlap(prev[1], matte), 3)
    return r


# ---------------------------------------------------------------- mapping measures

def _base(H):
    while hasattr(H, "base"):
        H = H.base
    return H


def limb_rows(H, matte, v_range, step=24.0, reach=60.0):
    """Per label row, through the point facing the camera: (v, left gap, right gap, label width) px, + = the label's
    own limb stops inside the matte (a strip of the source between them), - = it overhangs. Tubes only."""
    Hb = _base(H)
    if not hasattr(Hb, "Ru"):
        return None
    d = np.cross(Hb.A[0], Hb.A[1])
    ths = np.linspace(-np.pi, np.pi, 1441)
    m = matte.astype(np.float32)
    out = []
    for v in np.arange(v_range[0], v_range[1] + 1e-6, step):
        nd = np.sin(ths) * d[0] - Hb.k * d[1] + np.cos(ths) * d[2]
        i = np.nonzero(np.diff(np.sign(nd)))[0]
        if len(i) < 2:
            continue
        cand = ths[i]
        thL, thR = cand[np.argmin(np.sin(cand))], cand[np.argmax(np.sin(cand))]
        if thR < thL:
            thR += 2 * np.pi
        thc = 0.5 * (thL + thR)
        pL, pc, pR = H.forward(np.c_[Hb.u0 + Hb.Ru * np.array([thL, thc, thR]), np.full(3, v)])
        n = (pR - pL) / max(np.linalg.norm(pR - pL), 1e-6)
        sL, sR = float((pL - pc) @ n), float((pR - pc) @ n)
        ss = np.arange(sL - reach, sR + reach, 0.5)
        q = pc[None] + ss[:, None] * n[None]
        if (q < 1).any() or (q[:, 0] > m.shape[1] - 2).any() or (q[:, 1] > m.shape[0] - 2).any():
            continue
        ins = map_coordinates(m, [q[:, 1], q[:, 0]], order=1) > 0.5
        c = int(np.argmin(np.abs(ss)))
        if not ins[c]:
            continue
        lo = hi = c
        while lo > 0 and ins[lo - 1]:
            lo -= 1
        while hi < len(ss) - 1 and ins[hi + 1]:
            hi += 1
        if lo == 0 or hi == len(ss) - 1:
            continue
        out.append((float(v), float(sL - ss[lo]), float(ss[hi] - sR), sR - sL))
    return np.array(out) if len(out) >= 4 else None


def limb_gap(H, matte, v_range):
    """-> (p90 over rows of the larger side's gap, label width) px, or None."""
    L = limb_rows(H, matte, v_range)
    if L is None:
        return None
    g = np.r_[ndi.median_filter(L[:, 1], 3, mode="nearest"), ndi.median_filter(L[:, 2], 3, mode="nearest")]
    return round(float(np.percentile(g, 90)), 2), round(float(np.median(L[:, 3])), 1)


def occluder_edge(H_render, H_tracked, matte, v_range, slope=0.6, ends=0.15, moved=4.0, reach_px=50.0, step=24.0):
    """Does a side of the rendered label follow a matte edge that isn't the product's limb? Per label row, the matte's
    edge against the tracked (rigid) mapping's limb: a limb changes slowly along the axis (a flaring tube 4 px in 24
    rows); an occluder's cut steps or runs diagonally. -> {"run": 2+ consecutive steep row pairs, "any": one, "side",
    "slope", "moved": px the snap moved that side, "v": rows} on the worst side, or None (no snap, not a tube)"""
    if H_render is H_tracked:
        return None
    Lt = limb_rows(H_tracked, matte, v_range, step=step, reach=80.0)
    Lr = limb_rows(H_render, matte, v_range, step=step, reach=80.0)
    if Lt is None or Lr is None:
        return None
    sc = float(np.linalg.norm(_base(H_tracked).A[:, 1]))        # scene px per label px along the axis
    lo, hi = v_range[0] + ends * (v_range[1] - v_range[0]), v_range[1] - ends * (v_range[1] - v_range[0])
    rr = {v: (a, b) for v, a, b, _ in Lr}
    best = None
    for side, name in ((1, "left"), (2, "right")):
        e = {v: g for v, g in zip(Lt[:, 0], Lt[:, side]) if lo <= v <= hi}
        mv = max([abs(rr[v][side - 1] - e[v]) for v in e if v in rr] or [0.0])
        if mv <= moved:
            continue
        vs = sorted(e)
        run, any_, longest, smax, rows = 0, False, 0, 0.0, []
        for v1, v2 in zip(vs[:-1], vs[1:]):
            st = abs(e[v2] - e[v1]) / max((v2 - v1) * sc, 1e-6)
            if st > slope and v2 - v1 <= step + 1 and abs(e[v1]) <= reach_px and abs(e[v2]) <= reach_px:
                run += 1
                any_ = True
                smax = max(smax, st)
                rows += [v1, v2]
                longest = max(longest, run)
            else:
                run = 0
        if any_ and (best is None or (longest, smax) > (best["longest"], best["slope"])):
            best = {"longest": longest, "slope": round(smax, 2), "side": name, "moved": round(mv, 1),
                    "v": [float(min(rows)), float(max(rows))]}
    if best is None:
        return {"run": False, "any": False}
    return dict(best, run=best["longest"] >= 2, any=True)


def shear_of(H):
    """An affine tube's shear (deg) and axis-length ratio off 1 (V18's measure), or None."""
    return vf.shear_deg(_base(H))


def shape_points(H, G):
    """The label grid G mapped into the frame, and which points face the camera."""
    P = geo.apply_H(H, G)
    if hasattr(H, "inverse"):
        u, v, ok = H.inverse(P[:, 0], P[:, 1])
        ok = np.ones(len(G), bool) if ok is None else ok
        return P, ok & (np.abs(u - G[:, 0]) < 1.0) & (np.abs(v - G[:, 1]) < 1.0)
    return P, np.ones(len(G), bool)


def label_grid(outline, n=9):
    """n x n label points over the inner part of the label (reference px)."""
    (u0, v0), (u1, v1) = outline.min(0), outline.max(0)
    U, V = np.meshgrid(np.linspace(u0 + 0.15 * (u1 - u0), u1 - 0.15 * (u1 - u0), n), np.linspace(v0 + 0.05 * (v1 - v0), v1 - 0.05 * (v1 - v0), n))
    return np.c_[U.ravel(), V.ravel()]


def _sim_resid(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    Dm = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ Dm @ Vt
    s = np.trace(np.diag(D) @ Dm) / max((Xc ** 2).sum(1).mean(), 1e-12)
    return np.linalg.norm(s * Xc @ R.T + my - Y, axis=1)


def _hp(L, s=1.5):
    return L - gaussian_filter(L, s)


class Ghost:
    """The old print showing through the new label, in label space: per block, the partial correlation of the output's
    high-passed print with the source's, given the truth's. Blocks count only where the source's print is legible
    (at least half the truth's strength), unlike the truth there (correlation under 0.4: where the old print already
    matches, a ghost can't be told apart and doesn't show), away from sparkles (the source brighter than its own still
    label) and from the label's own fade into the source (feather)."""

    def __init__(self, art, ref_u8, still_med, still_step, feather=None, Ru=100.0, step=2.0, block=16, spark=0.03):
        self.art, self.block, self.spark = art, block, spark
        self.grid = ls.Grid(art, step)
        rx0, ry0, rx1, ry1 = art.stats["crop"]
        ref = map_coordinates(im.lum(im.srgb_to_lin(im.to_float(ref_u8[ry0:ry1, rx0:rx1]))), [self.grid.v, self.grid.u], order=1)
        self.ref_hp = _hp(ref)
        self.still = None
        if still_med is not None:
            g4v, g4u = self.grid.v / still_step, self.grid.u / still_step
            st = np.stack([map_coordinates(np.nan_to_num(still_med[..., c], nan=-1.0), [g4v, g4u], order=1, mode="nearest")
                           for c in range(3)], -1)
            st[st[..., 0] < 0] = np.nan
            self.still = st
        fu, fv = ((float(feather.get("theta_deg", 10)) * math.pi / 180 * Ru + 8, float(feather.get("v_px", 24)) + 8)
                  if feather else (8.0, 8.0))
        P = art.polygon
        g = self.grid
        self.inner = g.inside & (g.u > P[:, 0].min() + fu) & (g.u < P[:, 0].max() - fu) & (g.v > P[:, 1].min() + fv) & \
            (g.v < P[:, 1].max() - fv)

    def __call__(self, out_u8, src_u8, matte, H_out, H_src, min_rs=0.4):
        g, b = self.grid, self.block
        ro, vo = g.unwarp(out_u8, self.art.local(H_out), matte)
        rs, vs = g.unwarp(src_u8, self.art.local(H_out), matte)
        m = vo & vs & self.inner
        if self.still is not None:          # light passing in front is the source's on purpose
            rs2, _ = g.unwarp(src_u8, self.art.local(H_src), matte)
            ok = np.isfinite(self.still[..., 0])
            sp = ok & (im.lum(rs2) - np.where(ok, im.lum(np.nan_to_num(self.still)), 0) > self.spark)
            m &= ~binary_dilation(sp, iterations=2)
        if m.sum() < 500:
            return None
        Lo, Ls, Lr = _hp(im.lum(ro)), _hp(im.lum(rs)), self.ref_hp
        h, w = m.shape
        rows = []
        for i in range(0, h - b + 1, b):
            for j in range(0, w - b + 1, b):
                mm = m[i:i + b, j:j + b]
                if mm.mean() < 0.6:
                    continue
                o, s_, r = (x[i:i + b, j:j + b][mm] for x in (Lo, Ls, Lr))
                rows.append((i // b, j // b, o, s_, r))
        if not rows:
            return None
        E = np.array([x[3].std() for x in rows])
        Er = np.array([x[4].std() for x in rows])
        thr = max(np.percentile(E, 50), 0.5 * np.percentile(Er, 75))
        pc, ij = [], []
        for (bi, bj, o, s_, r), es in zip(rows, E):
            if es < thr or (r.std() > 0 and np.corrcoef(r, s_)[0, 1] > min_rs):
                continue
            X = np.c_[r - r.mean(), np.ones(len(r))]

            def resid(y):
                y = y - y.mean()
                return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
            a_, b_ = resid(o), resid(s_)
            d = np.sqrt((a_ ** 2).sum() * (b_ ** 2).sum())
            pc.append(float((a_ * b_).sum() / d) if d > 0 else 0.0)
            ij.append((bi, bj))
        if not pc:
            return {"blocks": 0}
        pc = np.array(pc)
        gh = np.zeros((h // b + 1, w // b + 1), bool)
        for (bi, bj), x in zip(ij, pc):
            gh[bi, bj] = x > 0.5
        lab, n = ndi.label(gh)
        cl = int(np.max(ndi.sum(gh, lab, range(1, n + 1)))) if n else 0
        return {"blocks": int(len(pc)), "pc": round(float(np.median(pc)), 3), "cluster": cl}


def label_jump(art, H0, H1, src0, src1, m0, m1, step=2.0):
    """The source's own print read back through each frame's mapping: how far it moves in label space between the two
    frames (label px), and that in scene px. A mapping that follows the body keeps it in place."""
    g = ls.Grid(art, step)
    r0, v0 = g.unwarp(src0, art.local(H0), m0)
    r1, v1 = g.unwarp(src1, art.local(H1), m1)
    m = v0 & v1 & g.inside
    if m.sum() < 500:
        return None
    A, B = _hp(im.lum(r0), 3.0), _hp(im.lum(r1), 3.0)
    A, B = np.where(m, A, 0.0), np.where(m, B, 0.0)
    win = np.outer(np.hanning(A.shape[0]), np.hanning(A.shape[1]))
    FA, FB = np.fft.fft2(A * win), np.fft.fft2(B * win)
    R = FB * np.conj(FA)
    r = np.fft.ifft2(R / (np.abs(R) + 1e-9)).real
    i, j = np.unravel_index(np.argmax(r), r.shape)
    dy = (i if i <= r.shape[0] // 2 else i - r.shape[0]) * step
    dx = (j if j <= r.shape[1] // 2 else j - r.shape[1]) * step
    c = art.polygon.mean(0) + art.off
    M = geo._local_affine(H0, c)
    d_scene = M[:2, :2] @ np.array([dx, dy])
    return round(float(np.hypot(dx, dy)), 2), round(float(np.linalg.norm(d_scene)), 2)


# ---------------------------------------------------------------- the checks

def _runs(frames):
    out = []
    for t in sorted(frames):
        if out and t == out[-1][1] + 1:
            out[-1][1] = t
        else:
            out.append([t, t])
    return out


def _fmt_runs(frames):
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in _runs(set(frames)))


def evaluate(F, fps=24.0, cfg=None):
    """F: clip frame -> measures (frame_measures, plus from the mappings: "shot" (a key per shot: frames of one shot
    share it), "gap" (p90, width), "shear" (deg, ratio), "pts" (shape points, front), "ghost", "jump_px" (to the
    previous frame, where two shots meet without a cut)). -> {frame: [(check, why)]}, summary"""
    c = dict(DEFAULTS, **(cfg or {}))
    T = sorted(F)
    refined = {t for t in T if F[t].get("changed_px", 0) > 300}
    flags = {t: [] for t in T}

    def cut(t):                                              # a cut between t - 1 and t
        if c.get("cuts") is not None:
            return t in c["cuts"] or F.get(t, {}).get("hist") is None
        r = F.get(t, {})
        return r.get("hist") is None or r["hist"] > c["cut_hist"] or r.get("overlap", 1.0) < c["cut_iou"]

    # edges: seams held for a quarter of a second (on two sides, or strongly on one), or the label's limb inside the body
    run_n = max(2, int(math.ceil(c["seam_run_s"] * fps)))

    def seam3(t):
        out = {}
        for sd in ("left", "right", "top", "bottom"):
            x = [F[u]["seam"][sd] for u in (t - 1, t, t + 1) if u in F and sd in F[u].get("seam", {})]
            if len(x) >= 2:
                out[sd] = float(np.median(x))
        return out
    sm = {t: seam3(t) for t in refined}
    hit = {t for t, v in sm.items() if v and (max(v.values()) > c["seam_strong"] or
                                              sum(x > c["seam_levels"] for x in v.values()) >= 2)}
    for a, b in _runs(hit):
        if b - a + 1 >= run_n:
            for t in range(a, b + 1):
                sd = max(sm[t], key=sm[t].get)
                flags[t].append(("edges", f"seam at the label's {sd} edge ({sm[t][sd]:.1f} levels)"))
    gap_hit = {t for t in refined if F[t].get("gap") and F[t]["gap"][0] > max(c["gap_px"], c["gap_frac"] * F[t]["gap"][1])}
    for a, b in _runs(gap_hit):
        if b - a + 1 >= run_n:
            for t in range(a, b + 1):
                flags[t].append(("edges", f"the label's side stops {F[t]['gap'][0]:.0f} px inside the product's edge"))
    # ghost
    for t in refined:
        g = F[t].get("ghost")
        if g and g.get("blocks", 0) >= c["ghost_min_blocks"] and (g["pc"] > c["ghost_pc"] or g["cluster"] >= c["ghost_cluster"]):
            flags[t].append(("ghost", f"the old print shows through the new label (partial correlation {g['pc']}, "
                                      f"{g['cluster']} adjacent blocks)"))
    # shear and shape spikes, against the frames either side in the same shot
    # a morph frame the fit followed is a spike that returns: the frames either side of it (one or two frames) agree
    # with each other more than with it. A trend or a kink is the product turning (V25), however fast
    def same(s, t):
        return s in refined and F[s].get("shot") == F[t].get("shot") and F[s].get("shear") is not None

    def S(s):
        return F[s]["shear"][0]
    for t in refined:
        if F[t].get("shear") is None or "shot" not in F[t]:
            continue
        d = None
        for L in (1, 2):
            for a in range(t - L + 1, t + 1):
                b = a + L - 1
                # the frames either side, and one more beyond each (a shot's first or last frame is no anchor)
                if not all(same(s, t) for s in range(a - 2, b + 3)):
                    continue
                s0, s1 = S(a - 1), S(b + 1)
                line = {s: s0 + (s1 - s0) * (s - a + 1) / (L + 1) for s in range(a, b + 1)}
                h = max(abs(S(s) - line[s]) for s in line)
                if abs(s1 - s0) < h:
                    d = max(d or 0.0, abs(S(t) - line[t]))
        if d is not None and d > c["shear_deg"]:
            flags[t].append(("shear", f"shear {d:.1f} deg off the frames around it"))
            continue
        # a shot's first or last frames: against the line through the next two
        for s1, s2 in ((t + 1, t + 2), (t - 1, t - 2)):
            if not same(2 * t - s1, t) and same(s1, t) and same(s2, t):
                e = abs(S(t) - (2 * S(s1) - S(s2)))
                if e > c["shear_deg_end"]:
                    flags[t].append(("shear", f"shear {e:.1f} deg off the frames after it" if s1 > t else
                                              f"shear {e:.1f} deg off the frames before it"))
    for t in refined:
        if not all(u in refined and F[u].get("pts") is not None and F[u].get("shot") == F[t].get("shot") for u in (t - 1, t, t + 1)):
            continue
        (A, oa), (B, ob), (C, oc) = F[t - 1]["pts"], F[t]["pts"], F[t + 1]["pts"]
        ok = oa & ob & oc
        if ok.sum() < 6 or not F[t].get("gap"):
            continue
        sp = float(_sim_resid(B[ok], 0.5 * (A[ok] + C[ok])).max())
        if sp > c["spike_frac"] * F[t]["gap"][1]:
            flags[t].append(("shear", f"shape {sp:.0f} px off its neighbours' (beyond moving, turning and scaling)"))
    # bend: a side snapped onto an occluder's edge; a run that holds it extends to the frames next to it in the shot
    # that show the same step
    core = {t for t in refined if (F[t].get("occl") or {}).get("run")}
    bend = set(core)
    for t in sorted(core):
        for step in (1, -1):
            u = t + step
            while u in refined and u not in bend and F[u].get("shot") == F[t].get("shot") and (F[u].get("occl") or {}).get("any"):
                bend.add(u)
                u += step
    for t in sorted(bend):
        o = F[t]["occl"]
        flags[t].append(("bend", f"the label's {o['side']} side follows a matte edge running {o['slope']:.1f} px across per px "
                                 f"along (rows {o['v'][0]:.0f}-{o['v'][1]:.0f}): an occluder's cut, not the product's limb; "
                                 f"the snap moved that side {o['moved']:.0f} px, so the print bends there"))
    # vanish and switch: where a refined run meets frames left as source, with no cut between
    vanish = {}
    for a, b in _runs(refined):
        fix = float(np.median([F[t]["fix_levels"] for t in range(a, b + 1) if "fix_levels" in F[t]] or [0.0]))
        near = {b: range(max(a, b - 4), b + 1), a: range(a, min(b, a + 4) + 1)}     # the run's frames next to each end
        for edge_t, step in ((b, 1), (a, -1)):
            t, ref = edge_t + step, F[edge_t]
            det = float(np.median([F[u]["detail"] for u in near[edge_t]]))
            hidden = []
            while t in F and t not in refined and not (cut(t) if step > 0 else cut(t + 1)):
                r = F[t]
                if r["matte_px"] < c["vanish_matte"] * ref["matte_px"] or r["detail"] < c["vanish_detail"] * det:
                    break
                hidden.append(t)
                t += step
            if not hidden:
                continue
            for u in hidden:
                vanish.setdefault(u, edge_t)
            if fix > 0 and ref.get("fix_levels", 0) >= c["switch_fix"] * fix:
                flags[edge_t].append(("switch", f"hard switch: the fix {'ends' if step > 0 else 'starts'} here at "
                                                f"{100 * ref['fix_levels'] / fix:.0f} % strength, the label still in view "
                                                f"on {hidden[0]}"))
    for u, e in vanish.items():
        flags[u].append(("vanish", f"not refined, though the product and its print are in view (the fix "
                                   f"{'ends' if u > e else 'starts'} at {e})"))
    shot_detail = {}
    for t in refined:
        shot_detail.setdefault(F[t].get("shot"), []).append(F[t]["detail"])
    for t in refined:
        j = F[t].get("jump_px")
        if j is None or j <= c["jump_px"] or (t - 1) not in refined or cut(t):
            continue
        # only where both frames show legible print (a smear gives the correlation nothing to hold)
        need = 0.5 * float(np.median(shot_detail.get(F[t].get("shot"), [0.0])))
        if F[t]["detail"] >= need and F[t - 1]["detail"] >= need:
            flags[t].append(("switch", f"the label jumps {j:.1f} px against the product from {t - 1} (another shot's fix)"))
    # mask dropouts inside a shot (V13): the product leaves the matte between frames with no cut
    empty = [t for t in T if F[t]["matte_px"] == 0 and (t - 1) in F and F[t - 1]["matte_px"] > 5000 and not cut(t)]
    flags = {t: v for t, v in flags.items() if v}
    by = {}
    for t, v in flags.items():
        for k, _ in v:
            by.setdefault(k, []).append(t)
    summary = {"checks": {k: _fmt_runs(v) for k, v in sorted(by.items())}, "flagged": len(flags),
               "cuts": [t for t in T[1:] if cut(t)],
               "refined": len(refined), "mask_empty_after": empty, "thresholds": c}
    return flags, summary


def notes_block(flags, summary, title="Automatic checks"):
    """A markdown section for the round's notes.md."""
    names = {"edges": "Edges (seam or gap at the label's sides, under the cap or crimp, at the base)",
             "ghost": "Old print showing through the new label",
             "shear": "Shear or shape spikes against the neighbouring frames",
             "vanish": "Label not refined while the product is in view",
             "switch": "Hard switch at a fix's end, or a jump where two shots meet",
             "bend": "The label bends along an occluder's edge (snapped onto it, not onto the product's limb)"}
    out = [f"## {title}", "",
           f"Run on the delivered lossless file ({summary['refined']} refined frames). Each check is calibrated on the "
           "frames marked in earlier review rounds; a flag means: look at that frame at 3-4x before sending."]
    if not flags:
        out.append("")
        out.append("No frame flagged.")
    else:
        out.append("")
        for k in ("edges", "ghost", "shear", "bend", "vanish", "switch"):
            if k in summary["checks"]:
                ex = next(why for t in sorted(flags) for kk, why in flags[t] if kk == k)
                out.append(f"- **{names[k]}:** frames {summary['checks'][k]} (e.g. {ex}).")
    if summary.get("mask_empty_after"):
        out.append(f"- **Mask empty** right after frames with the product in it: {_fmt_runs(summary['mask_empty_after'])}.")
    return "\n".join(out) + "\n"


MARK0, MARK1 = "<!-- automatic-checks -->", "<!-- /automatic-checks -->"


def write_notes(path, block):
    """Put the block into notes.md between markers (replacing an earlier run's), after anything written by hand."""
    from pathlib import Path
    p = Path(path)
    txt = p.read_text() if p.exists() else ""
    new = f"{MARK0}\n{block}{MARK1}\n"
    if MARK0 in txt and MARK1 in txt:
        a, b = txt.index(MARK0), txt.index(MARK1) + len(MARK1)
        txt = txt[:a] + new.rstrip("\n") + txt[b:]
    else:
        txt = (txt.rstrip("\n") + "\n\n" if txt else "") + new
    p.write_text(txt)
