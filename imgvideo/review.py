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
  body    the label off the product's body (V39): its top and bottom edges off the keyframe's proportion of the body
          (the gap to the collar seam or cap above, to the body's end below) or its axis turned off the body's
          centreline, held for a quarter of a second. The tracks follow the generated print, which the generator can
          draw longer, lower or slanted on the body; render.body_snap fixes it. Tubes only.
  glitter small bright specks in front of the label (glitter, sparkles) in the source that the output lost (V37): the
          new label went in over them, so the glitter seems to pass behind it. render.sparkles on those frames (with
          spark_glow) puts them back.

Inputs are the delivered frames themselves (source, output, matte) plus, per refined frame, the mapping that rendered
it and its job's artwork and still label. Nothing here changes a render."""
import json
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
                                 # lists them by hand (a cut between t - 1 and t is frame t)
    "body_px": 1.0,              # the label's top or bottom edge this far off the keyframe's proportion of the body
    "body_frac": 0.0055,         # (or this share of the label's length, if more: 1 px on the serum's 188 px label),
    "body_roll": 0.5,            # or its axis this many deg off the body's centreline (each a 5-frame median) ...
    "body_run_s": 0.25,          # ... held this long
    "slide_px": 4.0,             # the label turned round the product this far off the keyframe's place on the body (label
                                 # px at its middle, 5-frame median, held body_run_s): the jar's label slid 13 px with its
                                 # generated print across the camera move (V59)
    "glitter_tophat": 5,         # specks: a white top-hat this wide (px) ...
    "glitter_levels": 25.0,      # ... above this (sRGB levels), at most glitter_px across,
    "glitter_px": 6,
    "glitter_excess": 12.0,      # brighter by this (levels) than the frames around it there: light moving across,
    "glitter_print_px": 3,       # and this far off the truth's print (a morphing garbled print reads as specks)
    "glitter_lost": 2,           # a frame where the output lost this many of the source's specks ...
    "glitter_share": 0.5,        # ... and at least this share of them, while the product's surroundings show
    "glitter_ring": 2,           # at least this many specks too (glitter in front of the scene), and the label's lost
    "glitter_density": 3.0,      # specks are at most this many times denser than those (else: the surface shimmers)
    "glitter_pad": 2,            # sparkles_frames: the flagged runs padded by this many frames, and merged across
    "glitter_gap": 4,            # gaps up to this, within a shot
    "shimmer_levels": 3.0,       # shimmer: the light on the label's body (off its print) blinking from frame to frame
    "shimmer_block": 9,          # this many levels more than the source's at the same label points (temporal 2nd
    "shimmer_frac": 0.005,       # difference, block means over this many px), on this share of the body ...
    "shimmer_run": 3,            # ... for this many frames in a row: the tube's body boiled in soft patches round its
                                 # text (0.6% of it a frame, median), 0.06% with render.shade_local (V58)
    "step_share": 0.15,          # step: the product's step shown twice at the label's end on this share of its columns
    "step_run": 3,               # ... for this many frames in a row: the tube's curved keyframe 38-49% on every frame,
                                 # 0 with render.end_steps; v3's tube and the jar 0-1% (V60)
}


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

    def exclude(self, glass):
        """Leave out where the render keeps the frame's own glass (render.clear_glass, V56): the frame's surface there
        is the source on purpose, so it correlates with the source's."""
        if glass is not None:
            self.inner = self.inner & ~(glass(self.grid.u, self.grid.v) > 0.5)

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


def doubled(art, out_u8, src_u8, matte, H, polarity="light", clear=2, edge=4, tophat=15, thresh=0.03, piece=3, ends=32,
            reach=10, tile=48, match=0.4, ink=0.25, glass=None, keep=False, H_src=None):
    """The source's own print beside the label's (V54): the generator's lines drift against the tracked label, so
    wherever the label lets the source through over print (its end fade, a band that ends on a line) the frame's own
    line shows a few px off the label's: the one workflow's jar, its second-last line doubled at its band's
    lower end away from the keyframe. The ghost check reads only the label's inner part and misses it there.
    Measured inside the label's area (where its texture goes, the matte eroded `edge` px): print in the output (a
    top-hat of the job's polarity, `tophat` px at `thresh` linear) that the source shows there too, more than `clear`
    px off the label's own strokes (the keyframe's, through the mapping that rendered the frame), and explained by
    the label's own print moved: per `tile` px tile round it, the shift within `reach` px that best lays the label's
    print onto the source's print (Dice), kept when the two cover each other to `match` (the dark reflections
    of a chrome lid or a glass base showing through a fade are thin marks too, but nothing of the label moved).
    Pieces of `piece` px or more. glass: the render's clear-glass weight (render.clear_glass, V56): the frame's own glass
    it keeps is left out (its dark reflections beside the letters aren't a doubled line; "glass_px" counts them),
    except the frame's print where the tracking puts the label's (H_src, when the render placed the label elsewhere:
    render.body_snap.turn holds it off the generator's print, V59), off the label's own strokes: the generator's line
    left showing beside the label's ("glass_line_px", counted in "px").
    -> {"px", "at": [y, x] of the largest piece, "v": [label rows it spans], "by_end":
    px within `ends` rows of the label's top, its bottom, or inside, "band_v": [the label's rows]} or None (keep: plus
    "mask": (roi, the px), for a look)"""
    from scipy.signal import correlate

    from imgpost import warp as wp
    from .render import work_box
    if not matte.any():
        return None
    Hl = art.local(H)
    x0, y0, x1, y1 = roi = work_box(Hl, art, out_u8.shape[:2], pad=4)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    W, a_lab, (Ug, V) = wp.warp(art.pack, Hl, roi, art.refmask, ss=2, prefilter=0)
    lab = (W[..., 2] > 0.5) & (a_lab > 0.5) & binary_erosion(matte[y0:y1, x0:x1], iterations=edge)
    if lab.sum() < 200:
        return None

    def depth(L):
        th = []
        if polarity in ("light", "both"):
            th.append(L - ndi.grey_opening(L, size=(tophat, tophat)))
        if polarity in ("dark", "both"):
            th.append(ndi.grey_closing(L, size=(tophat, tophat)) - L)
        return np.max(th, 0)

    def prt(L):
        return depth(L) > thresh
    lp = W[..., 0] > 0.2                                     # the label's print
    own = lp.copy()
    if getattr(art, "texture", None) is not None:
        # the label's own strokes as the keyframe shows them, its body's grain, frost and highlights too
        own |= prt(im.lum(wp.warp(art.texture, Hl, roi, art.refmask, ss=2, prefilter=0)[0]))
    d_src, d_out = depth(lin_lum(src_u8[y0:y1, x0:x1])), depth(lin_lum(out_u8[y0:y1, x0:x1]))
    sp = d_src > thresh
    cand = (d_out > thresh) & sp & lab & ~binary_dilation(own, iterations=clear)
    # the source's print as the label's own print moved, tile by tile round the candidates
    h, w = cand.shape
    Lpad = np.pad(lp, reach)
    pred = np.zeros_like(cand)
    for ty in range(0, h, tile):
        for tx in range(0, w, tile):
            r0, r1, c0, c1 = ty, min(ty + tile, h), tx, min(tx + tile, w)
            if not cand[r0:r1, c0:c1].any():
                continue
            S = sp[r0:r1, c0:c1].astype(np.float32)
            Lw = Lpad[r0:r1 + 2 * reach, c0:c1 + 2 * reach].astype(np.float32)
            # every shift at once: how well the moved print and the source's print cover each other (Dice; a few
            # moved px that all land on print don't make a match)
            num = correlate(Lw, S, mode="valid", method="direct")
            den = correlate(Lw, np.ones_like(S), mode="valid", method="direct")
            sc = np.where(den >= 30, 2.0 * num / np.maximum(den + S.sum(), 1), 0.0)
            i, j = np.unravel_index(int(np.argmax(sc)), sc.shape)
            if sc[i, j] >= match:
                pred[r0:r1, c0:c1] |= Lpad[r0 + i:r1 + i, c0 + j:c1 + j]
    g = cand & binary_dilation(pred, iterations=1)
    ref = None
    if ink and (lp & lab).sum() > 50:
        # ink, not shading (V55): a round body's limb darkens (or a highlight brightens) over a band the top-hat takes
        # for print, and the relight carries it into the output as it should; next to the label's letters near a limb
        # it passes the moved-print test too. A doubled line is the generator's ink, as deep as the label's own: kept
        # within 2 px of the source's px at least `ink` times the depth of the label's print in this frame (p75)
        ref = float(np.percentile(d_out[lp & lab], 75))
        g &= binary_dilation(d_src >= ink * ref, iterations=2)
    n_glass = None
    if glass is not None:
        # where the render keeps the frame's own glass (render.clear_glass, V56) what shows is the frame's surface, its
        # print near the label's letters filled: its dark reflections beside the letters aren't a doubled line
        kept = glass(np.nan_to_num(Ug), np.nan_to_num(V)) > 0.5
        n_glass = int((g & kept).sum())
        g &= ~kept
        if H_src is not None:
            Ws = wp.warp(art.pack, art.local(H_src), roi, art.refmask, ss=2, prefilter=0)[0][..., 0] > 0.2
            gl = (d_out > thresh) & sp & kept & lab & ~binary_dilation(own, iterations=clear) & \
                binary_dilation(Ws, iterations=2)
            if ref is not None:
                gl &= binary_dilation(d_src >= ink * ref, iterations=2)
            g |= gl
    lb, n = ndi.label(g, np.ones((3, 3)))
    if n:
        sz = ndi.sum(g, lb, range(1, n + 1))
        g = np.isin(lb, 1 + np.nonzero(sz >= piece)[0])
    P = art.polygon
    v0, v1 = float(P[:, 1].min()), float(P[:, 1].max())
    r = {"px": int(g.sum()), "band_v": [round(v0, 1), round(v1, 1)]}
    if n_glass is not None:
        r["glass_px"] = n_glass
        if H_src is not None:
            r["glass_line_px"] = int((g & kept).sum())
    if r["px"]:
        lb, n = ndi.label(g, np.ones((3, 3)))
        sz = ndi.sum(g, lb, range(1, n + 1))
        cy, cx = ndi.center_of_mass(g, lb, int(np.argmax(sz)) + 1)
        vg = V[g]
        r.update(at=[int(round(cy)) + y0, int(round(cx)) + x0],
                 v=[round(float(np.percentile(vg, 5)), 1), round(float(np.percentile(vg, 95)), 1)],
                 by_end={"top": int((vg <= v0 + ends).sum()), "bottom": int((vg >= v1 - ends).sum()),
                         "inside": int(((vg > v0 + ends) & (vg < v1 - ends)).sum())})
    if keep:
        r["mask"] = (roi, g)
        r["ink_ref"] = ref
    return r


class Glitter:
    """Small bright specks in front of the label (glitter, sparkles) in the source that the output lost (V37), in label
    space: the source and the output both read through the source's own (tracked) mapping, every label px, so they're
    compared at the same frame points. A speck: a `tophat` px white top-hat of the sRGB luminance above `levels`, at
    most max_px across, and brighter by `excess` levels than the median of the shot's frames around it (2 each side) at
    the same label point and within `reach` px: light that moves across the label, not its print, which stays on it
    (the shot's own still label is too coarse to tell print from specks). Only off the truth's print (dilated by
    print_px): where the generator morphs its garbled print from frame to frame (a morph frame), its strokes read as
    specks too. Lost: the output keeps under half of its top-hat. Glitter in front of the scene shows around the
    product too, a shimmering surface (a metallic tube's sparkle that twinkles from frame to frame) doesn't: the same
    specks are counted in a ring outside the product's matte (`ring` px), per area, for evaluate() to compare. add()
    every refined frame in clip order, then finish(): each frame is measured as soon as the frames after it are in,
    and only 5 frames are kept (a large label is a million points a frame). Dark print (polarity "dark" or "both"):
    only off the frame's own dark print too (a 7 px black top-hat over `levels`, dilated 3 px): the gaps between the
    strokes of a garbled seal the generator morphs from frame to frame, beside the truth's, read as specks (V56).
    The same rows give the shimmer measure (V58): the light on the label's body blinking from frame to frame against
    the source's at the same points (each one's temporal 2nd difference, block means over `shimmer_block` px; the
    body px where the output's is `shimmer_levels` above the source's). For it the output is read through the mapping
    it was rendered with (add()'s H_out): where the render holds the label off the tracks (body_snap.turn, V59), its
    print read through the source's mapping moves across the body and reads as blinking light. -> {frame: measures}"""

    def __init__(self, art, inset=3, print_px=3, polarity="light", **measure_kw):
        self.art, self.grid = art, ls.Grid(art, 1.0)
        self.dark = polarity in ("dark", "both")
        prt = map_coordinates(art.pack[..., 0], [self.grid.v, self.grid.u], order=1) > 0.1
        self.inside = binary_erosion(self.grid.inside, iterations=int(inset)) & \
            ~(binary_dilation(prt, iterations=int(print_px)) if print_px > 0 else prt)
        self.rows, self.out, self.kw = {}, {}, measure_kw

    def _flush(self, upto):
        for T in sorted(self.rows):
            if T <= upto and T not in self.out:
                self.out[T] = self.measures(T, **self.kw)
        for T in [T for T in self.rows if T < upto - 2]:      # no frame still to measure needs it
            del self.rows[T]

    def finish(self):
        self._flush(float("inf"))
        return self.out

    @staticmethod
    def _levels(lin):
        return srgb_lum(np.clip(im.lin_to_srgb(np.clip(lin, 0, None)), 0, 1) * 255.0)

    def add(self, T, shot, src_u8, out_u8, matte, H_src, ring=(4, 40), H_out=None):
        tophat, levels, max_px = (self.kw.get(k, d) for k, d in (("tophat", 5), ("levels", 25.0), ("max_px", 6)))
        Hl = self.art.local(H_src)
        rs, vs = self.grid.unwarp(src_u8, Hl, matte)
        ro, _ = self.grid.unwarp(out_u8, Hl, matte)
        rr, vr = self.grid.unwarp(out_u8, self.art.local(H_out), matte) if H_out is not None else (None, None)
        # specks around the product: a ring outside its matte (no frame-to-frame reference: the background moves)
        ys, xs = np.nonzero(matte)
        n_ring = a_ring = 0
        if len(ys):
            y0, y1 = max(ys.min() - ring[1] - 3, 0), ys.max() + ring[1] + 4
            x0, x1 = max(xs.min() - ring[1] - 3, 0), xs.max() + ring[1] + 4
            d = distance_transform_edt(~matte[y0:y1, x0:x1])
            rg = (d > ring[0]) & (d <= ring[1])
            L = srgb_lum(src_u8[y0:y1, x0:x1])
            lab, n = ndi.label(rg & (L - ndi.grey_opening(L, size=(tophat, tophat)) > levels), structure=np.ones((3, 3)))
            n_ring = sum((a.stop - a.start) <= max_px and (b.stop - b.start) <= max_px for a, b in ndi.find_objects(lab))
            a_ring = int(rg.sum())
        Lo = self._levels(ro)
        self.rows[T] = (shot, self._levels(rs), Lo, vs & self.inside, int(n_ring), a_ring,
                        Lo if rr is None else self._levels(rr), vs & self.inside if vr is None else vr & vs & self.inside)
        self._flush(T - 2)

    def measures(self, T, tophat=5, levels=25.0, max_px=6, excess=12.0, reach=3, shimmer_levels=3.0, shimmer_block=9):
        import warnings
        shot, Ls, Lo, v, n_ring, a_ring, Lr, vr = self.rows[T]
        nb = [self.rows[u] for u in (T - 2, T - 1, T + 1, T + 2) if u in self.rows and self.rows[u][0] == shot]
        if len(nb) < 2 or v.sum() < 50:
            return None
        shim = None
        p, q = self.rows.get(T - 1), self.rows.get(T + 1)
        if p is not None and q is not None and p[0] == shot and q[0] == shot:
            okk = vr & p[7] & q[7]
            if okk.sum() >= 500:
                B = int(shimmer_block)
                wsum = ndi.uniform_filter(okk.astype(np.float32), B)

                def blk(L, Lp, Lq):
                    d2 = np.where(okk, np.abs(L - 0.5 * (Lp + Lq)), 0.0).astype(np.float32)
                    return ndi.uniform_filter(d2, B) / np.maximum(wsum, 1e-6)
                good = okk & (wsum > 0.5)
                ex = blk(Lr, p[6], q[6]) - blk(Ls, p[1], q[1])
                shim = {"area": int(good.sum()), "px": int((good & (ex > shimmer_levels)).sum())}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = np.nanmedian(np.stack([np.where(n[3], n[1], np.nan) for n in nb]), 0)
        ok = v & np.isfinite(ref)
        ref_d = ndi.grey_dilation(np.where(np.isfinite(ref), ref, -1e9).astype(np.float32), size=(reach, reach))
        ths = Ls - ndi.grey_opening(Ls, size=(tophat, tophat))
        tho = Lo - ndi.grey_opening(Lo, size=(tophat, tophat))
        cand = ok & (ths > levels) & (Ls - ref_d > excess)
        if self.dark:
            cand &= ~binary_dilation(ndi.grey_closing(Ls, size=(7, 7)) - Ls > levels, iterations=3)
        lab, n = ndi.label(cand, structure=np.ones((3, 3)))
        base = {"area": int(ok.sum()), "ring": n_ring, "ring_area": a_ring, "shimmer": shim}
        if not n:
            return dict(base, specks=0, lost=0, lost_px=0)
        idx = np.arange(1, n + 1)
        small = np.array([(a.stop - a.start) <= max_px and (b.stop - b.start) <= max_px for a, b in ndi.find_objects(lab)])
        ps, po = ndi.maximum(ths, lab, idx), ndi.maximum(tho, lab, idx)
        lost = small & (po < 0.5 * ps)
        return dict(base, specks=int(small.sum()), lost=int(lost.sum()), lost_px=int(ndi.sum(cand, lab, idx)[lost].sum()))


def end_doubled(art, out_u8, src_u8, matte, H, reach=24, every=4, thresh=12.0, sep=(3, 16), span=0.7):
    """The product's step shown twice at the label's ends (V60): a band edge the keyframe curves with the body, cut by
    the label's straight end, showed as a dark line inside the label with the frame's own band edge a few px past
    it. Columns across the middle `span` of the label every `every` px, through the mapping that rendered the frame
    (its surface runs on past the label): the luminance (sRGB levels) along the axis within `reach` px of each end,
    the same image points in the output and the source. A column is doubled where the output steps twice the same
    way (`thresh` levels per px, `sep` px apart) and the source doesn't. -> {"top": share, "bottom": share} of the
    columns inside the matte, or None"""
    from imgpost import geometry as geo
    P = art.polygon
    if P is None:
        return None
    u0, u1 = float(P[:, 0].min()), float(P[:, 0].max())
    v0, v1 = float(P[:, 1].min()), float(P[:, 1].max())
    um = 0.5 * (u0 + u1)
    us = np.arange(um - 0.5 * span * (u1 - u0), um + 0.5 * span * (u1 - u0) + 1e-6, every)
    Hl = art.local(H)
    Lo, Ls = srgb_lum(out_u8), srgb_lum(src_u8)
    h, w = matte.shape
    out = {}
    for end, vc in (("top", v0), ("bottom", v1)):
        vs = np.arange(vc - reach, vc + reach + 1e-6, 1.0)
        U, V = np.meshgrid(us, vs)
        XY = geo.apply_H(Hl, np.c_[U.ravel(), V.ravel()])
        x, y = XY[:, 0].reshape(U.shape), XY[:, 1].reshape(U.shape)
        ok = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
        xi, yi = np.clip(np.round(x).astype(int), 0, w - 1), np.clip(np.round(y).astype(int), 0, h - 1)
        ok &= matte[yi, xi]
        cols = ok.all(0)
        if cols.sum() < 5:
            continue
        def steps(L):
            prof = map_coordinates(L, [y, x], order=1, mode="nearest")[:, cols]
            d = 0.5 * (prof[2:] - prof[:-2])
            n = []
            for j in range(d.shape[1]):
                dj = d[:, j]
                pk = [i for i in range(1, len(dj) - 1) if abs(dj[i]) >= thresh and abs(dj[i]) >= abs(dj[i - 1]) and
                      abs(dj[i]) >= abs(dj[i + 1])]
                pairs = any(sep[0] <= b - a <= sep[1] and np.sign(dj[a]) == np.sign(dj[b]) for a in pk for b in pk if b > a)
                n.append((len(pk), pairs))
            return n
        so, ss_ = steps(Lo), steps(Ls)
        dbl = [po and no > ns for (no, po), (ns, ps) in zip(so, ss_)]
        out[end] = round(float(np.mean(dbl)), 3)
    return out or None


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
    previous frame, where two shots meet without a cut), "body" (the label against the body: fit.body_offsets and
    the gaps)). -> {frame: [(check, why)]}, summary (with "geometry": the body measures per frame, and
    "sparkles_frames": the glitter check's runs, ready for render.sparkles)"""
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
    # body: the label's top and bottom edges or its axis off the keyframe's proportion of the body (V39), a 5-frame
    # median within the shot, held for body_run_s
    by_shot = {}
    for t in refined:
        if F[t].get("body") is not None:
            by_shot.setdefault(F[t].get("shot"), []).append(t)
    body_run = max(2, int(math.ceil(c["body_run_s"] * fps)))
    geometry = {}
    for shot, ts in by_shot.items():
        ts = sorted(ts)
        med = {}
        for k in ("top_px", "bottom_px", "roll_deg"):
            x = np.array([F[t]["body"][k] for t in ts], float)
            x = np.where(np.isfinite(x), x, np.nanmedian(x) if np.isfinite(x).any() else 0.0)
            med[k] = ndi.median_filter(x, 5, mode="nearest")
        lim = {t: max(c["body_px"], c["body_frac"] * (F[t]["body"].get("label_px") or 0.0)) for t in ts}
        off = {t: (abs(med["top_px"][i]) > lim[t] or abs(med["bottom_px"][i]) > lim[t] or
                   abs(med["roll_deg"][i]) > c["body_roll"]) for i, t in enumerate(ts)}
        for a, b in _runs([t for t in ts if off[t]]):
            if b - a + 1 >= body_run:
                for t in range(a, b + 1):
                    i = ts.index(t)
                    flags[t].append(("body", f"the label sits off the body: top {med['top_px'][i]:+.1f} px, bottom "
                                             f"{med['bottom_px'][i]:+.1f} px off the keyframe's proportion, axis "
                                             f"{med['roll_deg'][i]:+.2f} deg off the body's centreline"))
        # slide: the label turned round the product off the keyframe's place on the body (the print it follows sliding
        # across the body, V59)
        x = np.array([F[t]["body"].get("turn_px") if F[t]["body"].get("turn_px") is not None else np.nan for t in ts], float)
        med["turn_px"] = ndi.median_filter(np.where(np.isfinite(x), x, np.nanmedian(x) if np.isfinite(x).any() else 0.0),
                                           5, mode="nearest") if len(x) else x
        for a, b in _runs([t for i, t in enumerate(ts) if np.isfinite(x).any() and abs(med["turn_px"][i]) > c["slide_px"]]):
            if b - a + 1 >= body_run:
                for t in range(a, b + 1):
                    i = ts.index(t)
                    flags[t].append(("slide", f"the label turned {med['turn_px'][i]:+.1f} px round the product off the "
                                              f"keyframe's place on its body: if the product doesn't turn on its axis "
                                              f"there, render.body_snap.turn holds it"))
        for i, t in enumerate(ts):
            geometry[t] = dict(F[t]["body"], top_px_med=round(float(med["top_px"][i]), 2),
                               bottom_px_med=round(float(med["bottom_px"][i]), 2),
                               roll_deg_med=round(float(med["roll_deg"][i]), 3),
                               **({"turn_px_med": round(float(med["turn_px"][i]), 2)} if np.isfinite(x).any() else {}))
    # glitter: specks in front of the label in the source that the output lost (V37)
    for t in refined:
        g = F[t].get("glitter")
        if g and g["lost"] >= c["glitter_lost"] and g["lost"] >= c["glitter_share"] * g["specks"] and \
                g.get("ring", 0) >= c["glitter_ring"] and \
                g["lost"] / max(g.get("area", 1), 1) <= c["glitter_density"] * g["ring"] / max(g.get("ring_area", 1), 1):
            flags[t].append(("glitter", f"{g['lost']} of the source's {g['specks']} specks in front of the label lost "
                                        f"(glitter seems to pass behind it): render.sparkles on this frame"))
    # shimmer: the light on the label's body blinking from frame to frame against the source's (V58)
    def shim(t):
        s = (F[t].get("glitter") or {}).get("shimmer")
        return s["px"] / max(s["area"], 1) if s else 0.0
    for a, b in _runs([t for t in refined if shim(t) > c["shimmer_frac"]]):
        if b - a + 1 >= c["shimmer_run"]:
            for t in range(a, b + 1):
                flags[t].append(("shimmer", f"the light on {100 * shim(t):.1f}% of the label's body blinks "
                                            f"{c['shimmer_levels']:g}+ levels more than the source's from frame to frame "
                                            f"(render.shade_local reads it round each patch)"))
    # step: the product's step (a band's edge) shown twice at the label's end (V60)
    def stp(t):
        e = F[t].get("endstep") or {}
        return max(e.values()) if e else 0.0
    for a, b in _runs([t for t in refined if stp(t) >= c["step_share"]]):
        if b - a + 1 >= c["step_run"]:
            for t in range(a, b + 1):
                e = F[t]["endstep"]
                end = max(e, key=e.get)
                flags[t].append(("step", f"the product's step shows twice at the label's {end} end ({100 * e[end]:.0f}% of "
                                         f"it): the keyframe's edge inside the label beside the frame's own"))
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
    if geometry:
        summary["geometry"] = {str(t): geometry[t] for t in sorted(geometry)}
    if "glitter" in by:
        # the runs for render.sparkles: padded, merged across short gaps, never across a shot's ends
        runs = []
        for a, b in _runs(by["glitter"]):
            sh = F[a].get("shot")
            a, b = max(a - c["glitter_pad"], sh[0] if sh else 0), min(b + c["glitter_pad"], (sh[1] - 1) if sh else b)
            if runs and runs[-1][2] == sh and a - runs[-1][1] <= c["glitter_gap"]:
                runs[-1][1] = max(runs[-1][1], b)
            else:
                runs.append([a, b, sh])
        summary["sparkles_frames"] = [[a, b + 1] for a, b, _ in runs]
    return flags, summary


def notes_block(flags, summary, title="Automatic checks"):
    """A markdown section for the round's notes.md."""
    names = {"edges": "Edges (seam or gap at the label's sides, under the cap or crimp, at the base)",
             "ghost": "Old print showing through the new label",
             "shear": "Shear or shape spikes against the neighbouring frames",
             "vanish": "Label not refined while the product is in view",
             "switch": "Hard switch at a fix's end, or a jump where two shots meet",
             "bend": "The label bends along an occluder's edge (snapped onto it, not onto the product's limb)",
             "body": "The label off the product's body (its ends off the keyframe's proportion, or turned off its axis)",
             "slide": "The label turned round the product off its place on the body (print sliding, or the product turning)",
             "glitter": "Glitter in front of the label lost (it seems to pass behind the label)",
             "shimmer": "The light on the label's body blinks from frame to frame (soft patches boil)",
             "step": "The product's step (a band's edge) shown twice at the label's end",
             "swim": "The label slides against the product on near-still frames",
             "ends": "The label slides on a held run's first or last frames (the glue's edge)",
             "level": "The label's lines off the product's base (straightened)",
             "lines": "The label's lines turned against the frame's own print",
             "sharp": "The print softer than the source's own print",
             "tilt": "The print's side edge turned or shifted against the product's limb",
             "spots": "Dark spots on the label's body",
             "double": "The source's own print beside the label's (doubled lines)"}
    out = [f"## {title}", "",
           f"Run on the delivered lossless file ({summary['refined']} refined frames). Each check is calibrated on the "
           "frames marked in earlier review rounds; a flag means: look at that frame at 3-4x before sending."]
    if not flags:
        out.append("")
        out.append("No frame flagged.")
    else:
        out.append("")
        for k in ("edges", "step", "ghost", "shear", "bend", "body", "slide", "glitter", "shimmer", "vanish", "switch", "swim",
                  "ends", "level", "lines", "sharp", "tilt", "spots", "double"):
            if k in summary["checks"]:
                ex = next(why for t in sorted(flags) for kk, why in flags[t] if kk == k)
                out.append(f"- **{names[k]}:** frames {summary['checks'][k]} (e.g. {ex}).")
    if summary.get("actions"):
        out.append("")
        out.append("To try next (the engine's settings):")
        for a in summary["actions"]:
            out.append(f"- {a['check']}: {a['why']} -> `{json.dumps(a['try'])}`")
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
