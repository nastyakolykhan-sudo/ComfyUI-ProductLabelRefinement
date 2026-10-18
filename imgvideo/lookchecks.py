"""The look checks: the user's notes from the first real run's review rounds (2026-10-17), read off the delivered frames,
with pass/fail flags and the setting an agent tries next (summary["actions"]). Ported from that run's job scripts
(check_swim.py, check_level.py, check_sharp.py, check_body.py, check_ghost.py, check_spots.py), same measures.

  swim    the label slides against the product on near-still frames (the user: "it looks like it's fluid and lives
          its own life"). Per pair of frames, the shift of the body (the matte without the label) and of the label
          (the changed pixels), each the best whole-region match of high-passed luminance (masked SSD, sub-pixel by a
          parabola); swim = |label - body|. The source's own print is read the same way on the same region: the
          label may swim as much as the garbled print did, not more. Near-still pairs only (the body moves at most
          still_px): on fast moves the measure is blur, not glue.
  ends    the same on the first and last pairs of a run the label holds: the glue's smoothing at a run's ends (edge
          "line" extrapolates the next frames' trend, "mean" doesn't; "mean" helped one product and hurt another).
  level   the label's lines against the product's base, both straightened on the body's axis (level.measure: skew by
          point projection, 1 px profile smoothing). Against 0 where render.level levelled to the base, else against
          the keyframe frame's own level. Only where the base holds steady (base_mad_max), as render.level.
  sharp   the print's edge strength against the keyframe frame's (the user: "the label gets too blurry at some
          points"), under sharp_floor and softer than the source's own print there by sharp_margin (a source that is
          soft too gets matching motion blur on purpose).
  tilt    the print's side edge against the product's limb (the vertical wordmark, if there is one), in the body's own
          frame, against the median over the keyframe's frame and ref_window frames each side: turned (deg) or
          shifted (px).
  ghost   on frames with the sparkle re-add (render.sparkles as a list of frames, whole-band compositing): the garbled
          print coming back through it (V37), the re-add render against one without it, measured in the render
          (render.json ghost_px). Print-only compositing keeps the re-add off: nothing to measure there (on an approved
          print-only render the same measure counts only the source's glitter, 45-250 px a frame).
  spots   small marks a print-only composite leaves: changed pixels darker than the source's own body by spot_levels,
          clear of the output's print by spot_clear px (letter fringes are not spots), in pieces under spot_max_px.
"""
import math

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, grey_opening, median_filter

from . import level as lvl
from .glue import edges
from .printalpha import edge_strength, lin_lum

LOOKS = {
    "still_px": 3.0,         # px: a pair is near-still when the body moves at most this far
    "swim_search": 8,        # px: the shift search around the matte centroids' shift
    "swim_excess": 0.35,     # px: the label's swim over the source print's on a near-still pair
    "swim_pairs": 2,         # ... on this many pairs within swim_window pairs: flagged
    "swim_window": 6,
    "ends_frames": 4,        # the pairs within this many frames of a held run's first and last frames: "ends"
    "level_deg": 1.0,        # deg: the lines off the base (or the keyframe's level), 5-frame median ...
    "level_run_s": 0.25,     # ... held this long
    "base_mad_max": 0.5,     # deg: the straightened base's frame-to-frame scatter above this: level not judged
    "base_jump_deg": 1.5,    # deg: a frame's base this far off its 15-frame running median (a sparkle trail splitting
                             # the matte read 21 deg) isn't judged
    "sharp_floor": 0.7,      # the print's edge strength under this share of the keyframe frame's ...
    "sharp_margin": 0.1,     # ... and under the source's own share there less this
    "tilt_deg": 1.0,         # deg: the print's side edge turned against the limb (5-frame median) ...
    "side_px": 4.0,          # ... or shifted (px, at the body's scale), on frames whose print is crisp (sharp_floor):
                             # a gross check. Read off a print-only composite it scatters (approved renders: p90
                             # 0.6 deg, 2 px); the body check (V39, through the mappings) is the fine one
    "tilt_run_s": 0.25,      # ... held this long
    "ref_window": 3,         # frames each side of the keyframe's: the tilt reference
    "ghost_px": 60,          # old print px the re-add brings back on a frame (it brought 33-330; real specks 6-41)
    "spot_levels": 12.0,     # levels darker than the source's body ...
    "spot_max_px": 60,       # ... in pieces at most this big ...
    "spot_clear": 4,         # ... this far from the output's print (dilation, px): a letter's fringe isn't a spot
    "spot_min_px": 4,        # ... and at least this big
    "spot_body": 15,         # px: the grey opening of the source that gives its body under the garbled print
}


# ---------------------------------------------------------------------------------------------------------- swim

def _hp(x):
    return gaussian_filter(x, 1.0) - gaussian_filter(x, 6.0)


def shift(a, b, m, mx, c=(0, 0)):
    """The (dx, dy) that best maps a onto b (b(p + d) ~ a(p)) over mask m: masked SSD over integer shifts within mx of
    c, then a parabola through the minimum's neighbours. -> (shift, on the search's edge) or None"""
    ys, xs = np.nonzero(m)
    if len(ys) < 50:
        return None
    va = a[ys, xs]
    H, W = b.shape
    E = np.full((2 * mx + 1, 2 * mx + 1), np.inf)
    cx, cy = int(round(c[0])), int(round(c[1]))
    for dy in range(-mx, mx + 1):
        yy = ys + dy + cy
        okY = (yy >= 0) & (yy < H)
        for dx in range(-mx, mx + 1):
            xx = xs + dx + cx
            ok = okY & (xx >= 0) & (xx < W)
            if ok.sum() > 0.8 * len(ys):
                E[dy + mx, dx + mx] = np.mean((b[yy[ok], xx[ok]] - va[ok]) ** 2)
    if not np.isfinite(E).any():
        return None
    iy, ix = np.unravel_index(np.argmin(E), E.shape)

    def par(e0, e1, e2):
        d = e0 - 2 * e1 + e2
        return 0.0 if not np.isfinite(d) or d <= 0 else 0.5 * (e0 - e2) / d
    fx = par(E[iy, ix - 1], E[iy, ix], E[iy, ix + 1]) if 0 < ix < 2 * mx else 0.0
    fy = par(E[iy - 1, ix], E[iy, ix], E[iy + 1, ix]) if 0 < iy < 2 * mx else 0.0
    edge = ix in (0, 2 * mx) or iy in (0, 2 * mx)
    return np.array([ix - mx + fx + cx, iy - mx + fy + cy]), edge


def swim_pair(s0, s1, o0, o1, m0, m1, region, still_px=3.0, mx=8):
    """One pair of frames: the body's shift (source), and the label's against it in the output and in the source on the
    same region (the output's changed pixels on the first frame). -> {"body": [dx, dy], "swim_px", "swim_src_px"}, or
    None (too little to match, or the body moves more than still_px)"""
    ys, xs = np.nonzero(m0 | m1)
    if not len(ys):
        return None
    y0, y1 = max(ys.min() - 20, 0), min(ys.max() + 21, m0.shape[0])
    x0, x1 = max(xs.min() - 20, 0), min(xs.max() + 21, m0.shape[1])
    sl = (slice(y0, y1), slice(x0, x1))
    c0 = np.array(np.nonzero(m0[sl]))[::-1].mean(1)
    c1 = np.array(np.nonzero(m1[sl]))[::-1].mean(1)
    cs = c1 - c0
    R = region[sl]
    body = binary_dilation(m0[sl], iterations=3) & ~binary_dilation(R, iterations=8)
    lab = binary_dilation(R, iterations=2) & m0[sl]
    A, B = _hp(lin_lum(s0[sl])), _hp(lin_lum(s1[sl]))
    db = shift(A, B, body, mx, cs)
    if db is None or db[1] or np.hypot(*db[0]) > still_px:
        return None
    ds = shift(A, B, lab, mx, cs)
    Ao, Bo = _hp(lin_lum(o0[sl])), _hp(lin_lum(o1[sl]))
    # the output's body is the source's outside the label (only the label changes): the same body shift
    do = shift(Ao, Bo, lab, mx, cs)
    if ds is None or do is None:
        return None
    return {"body": [round(float(v), 2) for v in db[0]], "swim_px": round(float(np.hypot(*(do[0] - db[0]))), 2),
            "swim_src_px": round(float(np.hypot(*(ds[0] - db[0]))), 2)}


# ---------------------------------------------------------------------------------------------------------- tilt

def tilt_measure(frame, M, rim=6, step=4.0):
    """check_body.py's measure: the print's left edge (its leftmost stroke per 4 px step along the axis) against the
    body's left limb, in the body's frame. -> {"tilt_deg", "side_px", "top_px", "bot_px", "body_px", "axis_deg"} or
    None"""
    ys, xs = np.nonzero(M)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    L = lin_lum(frame[y0:y1, x0:x1])
    Mi = binary_erosion(M[y0:y1, x0:x1], iterations=rim)
    ink = ((L - grey_opening(L, size=(15, 15))) > 0.03) & Mi
    iy, ix = np.nonzero(ink)
    if len(iy) < 200:
        return None
    rows = np.arange(np.percentile(iy, 2), np.percentile(iy, 98)).astype(int)
    m = M[y0:y1, x0:x1]
    mids, ok = [], []
    for r in rows:
        c = np.nonzero(m[r])[0]
        if len(c) > 10:
            mids.append(0.5 * (c.min() + c.max()))
            ok.append(r)
    if len(ok) < 4:
        return None
    a, b = np.polyfit(ok, mids, 1)
    th = np.arctan(a)
    d = np.array([np.sin(th), np.cos(th)])
    n = np.array([np.cos(th), -np.sin(th)])
    c0 = np.array([b, 0.0])
    my, mx_ = np.nonzero(m)
    P = np.stack([mx_, my], 1) - c0
    Q = np.stack([ix, iy], 1) - c0
    ma, mc = P @ d, P @ n
    qa, qc = Q @ d, Q @ n
    lim, lft, at = [], [], []
    for lo in np.arange(qa.min(), qa.max(), step):
        s = (qa >= lo) & (qa < lo + step)
        sm = (ma >= lo) & (ma < lo + step)
        if s.sum() >= 2 and sm.sum() >= 2:
            lim.append(mc[sm].min())
            lft.append(qc[s].min())
            at.append(lo + step / 2)
    if len(at) < 8:
        return None
    lim, lft, at = map(np.array, (lim, lft, at))
    gap = lft - lim
    col = gap < np.percentile(gap, 50) + 6
    if col.sum() < 8:
        return None
    k, c = np.polyfit(at[col], gap[col], 1)
    r = gap[col] - (k * at[col] + c)
    keep = np.abs(r) < 3 * 1.4826 * np.median(np.abs(r)) + 0.5
    k, c = np.polyfit(at[col][keep], gap[col][keep], 1)
    return {"tilt_deg": float(np.degrees(np.arctan(k))), "side_px": float(np.median(gap[col][keep])),
            "top_px": float(qa.min() - ma.min()), "bot_px": float(ma.max() - qa.max()),
            "body_px": float(ma.max() - ma.min()), "axis_deg": float(np.degrees(th))}


# --------------------------------------------------------------------------------------------- ghost and spots

def ghost_px(src, clean, out, M, levels=4.0):
    """check_ghost.py: old garbled print (the source's top-hat 7 at 0.03) away from the new print (the clean render's
    top-hat 15 at 0.03, dilated 3 px) where the output is brighter than the clean render by `levels`, inside the matte
    eroded 4 px. -> px"""
    m = binary_erosion(M, iterations=4)
    if not m.any():
        return 0
    ys, xs = np.nonzero(m)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    Ls, Lc = lin_lum(src[sl]), lin_lum(clean[sl])
    old = (Ls - grey_opening(Ls, size=(7, 7))) > 0.03
    new = binary_dilation((Lc - grey_opening(Lc, size=(15, 15))) > 0.03, iterations=3)
    brighter = (out[sl].astype(np.int16) - clean[sl].astype(np.int16)).mean(-1) > levels
    return int((old & ~new & brighter & m[sl]).sum())


def spots(src, out, changed, cfg=None):
    """check_spots.py, calibrated: changed px darker than the source's own body round them (a 7 px grey opening of the
    source) by spot_levels, clear of the output's print (top-hat 15 at 0.03, dilated spot_clear px: the first version
    took a letter's dark fringe, 2 px out, for a spot), in pieces of spot_min_px to spot_max_px. The body is a
    spot_body px opening of the source (7 px kept garbled glyphs wider than that). -> [(y, x, px)]"""
    c = dict(LOOKS, **(cfg or {}))
    if not changed.any():
        return []
    ys, xs = np.nonzero(changed)
    y0, y1 = max(ys.min() - 8, 0), min(ys.max() + 9, changed.shape[0])
    x0, x1 = max(xs.min() - 8, 0), min(xs.max() + 9, changed.shape[1])
    sl = (slice(y0, y1), slice(x0, x1))
    Lo = lin_lum(out[sl])
    pr = binary_dilation((Lo - grey_opening(Lo, size=(15, 15))) > 0.03, iterations=int(c["spot_clear"]))
    # the body under the source's garbled print: a 15 px opening (the composite clamps the body to a 13 px one; a 7 px
    # opening kept a garbled glyph wider than that, a heart, and read its removal as a spot)
    body = grey_opening(src[sl].astype(np.float32).mean(-1), size=(int(c["spot_body"]), int(c["spot_body"])))
    dark = (body - out[sl].astype(np.float32).mean(-1) >= float(c["spot_levels"])) & changed[sl] & ~pr
    lab, n = ndi.label(dark, np.ones((3, 3)))
    if not n:
        return []
    ar = ndi.sum(dark, lab, range(1, n + 1))
    cm = ndi.center_of_mass(dark, lab, range(1, n + 1))
    return [(int(round(y)) + y0, int(round(x)) + x0, int(s)) for (y, x), s in zip(cm, ar)
            if c["spot_min_px"] <= s <= c["spot_max_px"]]


# -------------------------------------------------------------------------------------------- per-frame measures

def frame_looks(src, out, M, changed, polarity="light"):
    """The per-frame measures: print sharpness of the output and the source, the print against the limb (tilt), the
    lines against the base (level), the base alone, spots. -> dict"""
    r = {}
    if not M.any():
        return r
    r["sharp_out"] = edge_strength(out, M)
    r["sharp_src"] = edge_strength(src, M)
    t = tilt_measure(out, M)
    if t is not None:
        r["tilt"] = t
    try:
        lv = lvl.measure(out, M, polarity)
    except (ValueError, np.linalg.LinAlgError):
        lv = None
    if lv is not None:
        r["level"] = lv
    try:
        r["base_deg"] = lvl.base_deg(M)
    except (ValueError, np.linalg.LinAlgError):
        pass
    sp = spots(src, out, changed)
    if sp:
        r["spots"] = sp
    return r


# --------------------------------------------------------------------------------------------------- evaluate

def _runs(ts):
    out = []
    for t in sorted(ts):
        if out and t == out[-1][1] + 1:
            out[-1][1] = t
        else:
            out.append([t, t])
    return out


def evaluate(F, shots, fps=24.0, cfg=None):
    """F {clip frame: {"looks": frame_looks(...), "swim": swim_pair(t, t + 1) or None, "ghost_px": or None}} over the
    refined frames; shots {key: {"frames": [clip frames], "keyframe": frame, "level_target": "base" | None, "glue":
    render.glue, "level_on": bool}}. -> ({frame: [(check, why)]}, summary {"looks": per-check numbers, "actions":
    [{"check", "frames", "try": settings}]})"""
    c = dict(LOOKS, **(cfg or {}))
    flags, summ, actions = {}, {}, []

    def flag(t, k, why):
        flags.setdefault(t, []).append((k, why))

    def act(k, frames, why, try_):
        actions.append({"check": k, "frames": [[a, b + 1] for a, b in _runs(frames)], "why": why, "try": try_})
    for key, sh in shots.items():
        ts = sorted(t for t in sh["frames"] if t in F)
        if not ts:
            continue
        kf = sh["keyframe"]
        S = {}
        # swim: near-still pairs; a run of excess pairs is flagged, and the pairs at a held run's ends on their own
        sw = {t: F[t]["swim"] for t in ts if F[t].get("swim")}
        if sw:
            ex = {t: s["swim_px"] - s["swim_src_px"] for t, s in sw.items()}
            bad = sorted(t for t, e in ex.items() if e > c["swim_excess"])
            held = _runs(ts)
            first = {t for a, b in held for t in sw if a <= t < a + c["ends_frames"]}
            last = {t for a, b in held for t in sw if b - c["ends_frames"] <= t < b}
            ends = first | last
            hit = [t for t in bad if t not in ends and
                   sum(abs(u - t) <= c["swim_window"] // 2 for u in bad if u not in ends) >= c["swim_pairs"]]
            end_hit = [t for t in bad if t in ends]
            for t in hit:
                flag(t, "swim", f"the label slides {sw[t]['swim_px']:.2f} px against the product to {t + 1} "
                                f"(the source's print {sw[t]['swim_src_px']:.2f}; the body moves "
                                f"{np.hypot(*sw[t]['body']):.1f} px)")
            g = sh.get("glue")
            if hit:
                act("swim", hit, "the label swims on near-still frames",
                    {"render": {"glue": True}} if not g else {"render": {"glue": dict(g if isinstance(g, dict) else {},
                                                                                      sigma=6)}})
            for t in end_hit:
                flag(t, "ends", f"at the run's end the label slides {sw[t]['swim_px']:.2f} px against the product to "
                                f"{t + 1} (the source's print {sw[t]['swim_src_px']:.2f})")
            if end_hit:
                # flip the glue's edge at the end that swims, keep the other (one product wanted "mean" at its start,
                # another "line" there and "mean" at its end)
                e = list(edges(g.get("edge", "line") if isinstance(g, dict) else "line"))
                flip = {"line": "mean", "mean": "line"}
                if any(t in first for t in end_hit):
                    e[0] = flip[e[0]] if g else "mean"
                if any(t in last for t in end_hit):
                    e[1] = flip[e[1]] if g else "mean"
                act("ends", end_hit, "the label swims on a held run's " + " and ".join(
                    w for w, s_ in (("first pairs", first), ("last pairs", last)) if any(t in s_ for t in end_hit)),
                    {"render": {"glue": dict(g if isinstance(g, dict) else {}, edge=e[0] if e[0] == e[1] else e)}})
            v = np.array(list(ex.values()))
            S["swim"] = {"pairs": len(sw), "median_px": round(float(np.median([s["swim_px"] for s in sw.values()])), 2),
                         "source_median_px": round(float(np.median([s["swim_src_px"] for s in sw.values()])), 2),
                         "excess_p90_px": round(float(np.percentile(v, 90)), 2), "flagged": len(hit),
                         "ends_flagged": len(end_hit),
                         "ends": {str(t): [sw[t]["swim_px"], sw[t]["swim_src_px"]] for t in sorted(ends)}}
        # level: against the base, where the base holds steady
        bd = [F[t]["looks"].get("base_deg") for t in ts]
        bdv = [b for b in bd if b is not None]
        sc = lvl.scatter(bdv)
        lv0 = {t: F[t]["looks"]["level"] for t in ts if F[t]["looks"].get("level")}
        lv, raw = {}, {t: m["level_deg"] for t, m in lv0.items()}
        if lv0:
            tl0 = sorted(lv0)
            bm = median_filter(np.array([lv0[t]["base_deg"] for t in tl0]), 15, mode="nearest")
            lv = {t: lv0[t]["level_deg"] for t, b in zip(tl0, bm)
                  if abs(lv0[t]["base_deg"] - b) <= c["base_jump_deg"] and abs(lv0[t]["text_deg"]) < 5.9}
        if sc is None or sc > c["base_mad_max"] or len(lv) < 5:
            S["level"] = {"judged": False, "base_scatter_deg": None if sc is None else round(sc, 3)}
        else:
            ref = [lv[t] for t in lv if abs(t - kf) <= c["ref_window"]]
            kfl = float(np.median(ref)) if ref else None
            to_base = sh.get("level_target") == "base"
            target = 0.0 if to_base else kfl
            S["level"] = {"judged": True, "base_scatter_deg": round(sc, 3), "target": "base" if to_base else "keyframe",
                          "keyframe_level_deg": None if kfl is None else round(kfl, 2), "frames": len(raw),
                          "not_judged": sorted(set(raw) - set(lv))}
            k0 = [raw[t] for t in raw if abs(t - kf) <= c["ref_window"]]
            if k0:
                # check_level.py's numbers: every measured frame against the keyframe frame's level
                v0 = np.array(list(raw.values())) - float(np.median(k0))
                S["level"]["vs_keyframe"] = {"median_deg": round(float(np.median(v0)), 2),
                                             "median_abs_deg": round(float(np.median(np.abs(v0))), 2),
                                             "p90_abs_deg": round(float(np.percentile(np.abs(v0), 90)), 2)}
            if target is not None:
                tl = sorted(lv)
                x = median_filter(np.array([lv[t] - target for t in tl]), 5, mode="nearest")
                run = max(2, int(math.ceil(c["level_run_s"] * fps)))
                off = [t for t, v in zip(tl, x) if abs(v) > c["level_deg"]]
                hit = [t for a, b in _runs(off) if b - a + 1 >= run for t in range(a, b + 1) if t in lv]
                for t in hit:
                    flag(t, "level", f"the label's lines run {x[tl.index(t)]:+.1f} deg against the "
                                     f"{'base' if to_base else 'keyframe frame' + chr(39) + 's level'} "
                                     f"(+: lower on the right)")
                if hit:
                    act("level", hit, "the label's lines off the base",
                        {"render": {"level": True}} if not sh.get("level_on") else {"render": {"level": {"sigma": 2}}})
                S["level"].update(median_deg=round(float(np.median(x)), 2), median_abs_deg=round(float(np.median(np.abs(x))), 2),
                                  p90_abs_deg=round(float(np.percentile(np.abs(x), 90)), 2), flagged=len(hit))
                if not to_base and kfl is not None and abs(kfl) > c["level_deg"] / 2:
                    act("keyframe_level", [kf], f"the keyframe's own lines run {kfl:+.2f} deg against its base",
                        {"render": {"level": {"offset": "base"}}})
        # sharp: against the keyframe frame's own, and the source's
        ko, ks = (F.get(kf, {}).get("looks") or {}).get("sharp_out"), (F.get(kf, {}).get("looks") or {}).get("sharp_src")
        if ko:
            ro = {t: F[t]["looks"]["sharp_out"] / ko for t in ts if F[t]["looks"].get("sharp_out")}
            rs = {t: F[t]["looks"]["sharp_src"] / ks for t in ts if ks and F[t]["looks"].get("sharp_src")}
            soft = [t for t, r in ro.items() if r < c["sharp_floor"] and r < rs.get(t, 1.0) - c["sharp_margin"]]
            for t in soft:
                flag(t, "sharp", f"the print is softer than the source's here: {ro[t]:.2f} of the keyframe frame's "
                                 f"sharpness (the source {rs.get(t, float('nan')):.2f})")
            if soft:
                act("sharp", soft, "the print softer than the source's", {"render": {"soft_cap": 0.15,
                                                                                     "print_alpha": {"shutter": 0}}})
            v = np.array(list(ro.values()))
            S["sharp"] = {"median": round(float(np.median(v)), 3), "p10": round(float(np.percentile(v, 10)), 3),
                          "under_floor": [t for t, r in ro.items() if r < c["sharp_floor"]], "flagged": len(soft)}
        # tilt: the print's side edge against the limb, against the keyframe's frame and its neighbours
        tm = {t: F[t]["looks"]["tilt"] for t in ts if F[t]["looks"].get("tilt")}
        ref = [tm[t] for t in tm if abs(t - kf) <= c["ref_window"]]
        crisp = {t for t in tm if not ko or (F[t]["looks"].get("sharp_out") or 0) / ko >= c["sharp_floor"]}
        if ref and len(tm) >= 5:
            k = {q: float(np.median([m[q] for m in ref])) for q in ("tilt_deg", "side_px", "body_px")}
            tl = sorted(tm)
            dt = np.array([tm[t]["tilt_deg"] - k["tilt_deg"] for t in tl])
            ds = np.array([tm[t]["side_px"] - k["side_px"] * tm[t]["body_px"] / k["body_px"] for t in tl])
            mt, ms = median_filter(dt, 5, mode="nearest"), median_filter(ds, 5, mode="nearest")
            run = max(2, int(math.ceil(c["tilt_run_s"] * fps)))
            off = [t for t, a, b in zip(tl, mt, ms) if t in crisp and (abs(a) > c["tilt_deg"] or abs(b) > c["side_px"])]
            hit = [t for a, b in _runs(off) if b - a + 1 >= run for t in range(a, b + 1) if t in tm]
            for t in hit:
                i = tl.index(t)
                flag(t, "tilt", f"the print's side edge {mt[i]:+.2f} deg, {ms[i]:+.1f} px off the product's limb "
                                f"against the keyframe's")
            if hit:
                act("tilt", hit, "the print turned or shifted against the limb", {"render": {"body_snap": True}})
            S["tilt"] = {"tilt_median_abs_deg": round(float(np.median(np.abs(dt))), 2),
                         "tilt_p90_abs_deg": round(float(np.percentile(np.abs(dt), 90)), 2),
                         "side_median_abs_px": round(float(np.median(np.abs(ds))), 2),
                         "side_p90_abs_px": round(float(np.percentile(np.abs(ds), 90)), 2), "flagged": len(hit)}
        # ghost (measured in the render) and spots
        gp = {t: F[t]["ghost_px"] for t in ts if F[t].get("ghost_px") is not None}
        if gp:
            gh = [t for t, v in gp.items() if v > c["ghost_px"]]
            for t in gh:
                flag(t, "ghost", f"the sparkle re-add brings {gp[t]} px of the old print back")
            if gh:
                act("ghost", gh, "the sparkle re-add brings the old print back", {"render": {"alpha": "print"}})
            S["ghost"] = {"median_px": int(np.median(list(gp.values()))), "max_px": int(max(gp.values())),
                          "flagged": len(gh)}
        sp = {t: F[t]["looks"]["spots"] for t in ts if F[t]["looks"].get("spots")}
        for t, v in sp.items():
            flag(t, "spots", f"{len(v)} dark spot{'s' if len(v) > 1 else ''} on the label's body (at " +
                 ", ".join(f"{x},{y}" for y, x, _ in v[:3]) + "): look")
        S["spots"] = {"frames": len(sp), "flagged": len(sp)}
        summ[str(key)] = S
    return flags, {"looks": summ, "actions": actions}
