"""Per-frame surface fit to the tracks, with the stills engine's own mapping (imgpost geometry.fit: homography or cylinder) or a tube.

Each track gets its label coordinates (u, v) once, from the anchor frame's mapping (a stills job measured on
that frame). Every frame is then a still fit whose points are the tracks: geometry.fit with the anchor job's model.
Smoothing works on control points (a grid on the label mapped by each frame's fit), so it is the same for any model,
and refits the model to the smoothed points. It is zero-lag (a centred local quadratic) and never runs across an
impact (a kink in the motion: a bounce, a hit) or a cut, where any smoothing would pull the label off the product."""
import numpy as np

from imgpost import geometry as geo


def _rebuild_residual(state):
    obj = object.__new__(geo.Residual)
    obj.__dict__.update(state)
    return obj


# geometry.Residual forwards unknown attributes to its base through __dict__, which unpickling hasn't filled yet:
# pickle it by its state instead (fit.pkl holds the per-frame mappings)
import copyreg  # noqa: E402
copyreg.pickle(geo.Residual, lambda R: (_rebuild_residual, (dict(R.__dict__),)))


def label_coords(H0, xy):
    """Anchor-frame scene px -> label (reference) px, and whether the point is on the modelled surface."""
    if hasattr(H0, "inverse"):
        u, v, ok = H0.inverse(xy[:, 0], xy[:, 1])
        return np.c_[u, v], (np.ones(len(xy), bool) if ok is None else ok)
    q = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(H0).T
    return q[:, :2] / q[:, 2:3], np.ones(len(xy), bool)


def _geo_fit(spec):
    """geometry.fit, but a perspective cylinder whose camera lands behind the product (the perspective terms are tiny
    and their sign is noise on a nearly affine view) is refitted affine: the engine would see its back half."""
    if spec.get("model") == "tube":
        from .surface import fit as tube_fit
        return tube_fit(spec)
    H, rep = geo.fit(spec)
    if spec.get("perspective") and isinstance(H, geo.Cylinder):
        cam = H._camera()
        if cam is not None and cam[2] < 0:
            H, rep = geo.fit(dict(spec, perspective=False))
    return H, rep


def _spec(base, uv, xy, w):
    s = {k: v for k, v in base.items() if k in ("model", "cylinder", "tube", "camera", "perspective", "perspective_reg")}
    s["points"] = [{"ref": [float(a), float(b)], "scene": [float(c), float(d)], "w": float(e)}
                   for (a, b), (c, d), e in zip(uv, xy, w)]
    return s


def with_residual(H, uv, xy, residual):
    """The stills engine's Residual (a smooth scene-space RBF correction) through the points' misfit: a mesh warp on top of
    the rigid model, for surfaces it can't follow (a squeezed tube, a bottle drawn a little differently per frame).
    residual: {"sigma": px | "auto" (a quarter of the points' smaller extent), "lam": 1.0}"""
    if not residual:
        return H
    q = geo.apply_H(H, uv)
    sig = residual.get("sigma", "auto")
    if sig == "auto":
        sig = 0.25 * float(min(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    return geo.Residual(H, q, xy - q, float(sig), float(residual.get("lam", 1.0)))


def shear_deg(H):
    """How far an affine tube's camera is from a rigid one: the angle between its around and along axes off 90
    degrees, and their length ratio off 1 (None for other models)."""
    A = getattr(H, "A", None)
    if A is None or not hasattr(H, "Ru"):
        return None
    a, b = A[:, 0], A[:, 1]
    ang = 90.0 - np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
    return float(ang), float(np.linalg.norm(a) / np.linalg.norm(b) - 1.0)


def fit_frame(base, uv, xy, w, wmin=0.5, reject=2.5, iters=2, residual=None):
    """One frame: the model fitted to the visible tracks, outliers (residual above `reject` px and 3x the median)
    dropped and refitted, then the residual correction when asked.
    -> (mapping, residual px per track of the rigid model (nan where not used), inlier mask)"""
    use = w >= wmin
    if use.sum() < 8:
        return None, np.full(len(uv), np.nan), use
    for _ in range(iters + 1):
        H, _ = _geo_fit(_spec(base, uv[use], xy[use], np.sqrt(w[use])))
        r = np.linalg.norm(geo.apply_H(H, uv) - xy, axis=1)
        new = (w >= wmin) & (r <= max(reject, 3.0 * np.median(r[use])))
        if new.sum() < 8 or (new == use).all():
            break
        use = new
    return with_residual(H, uv[use], xy[use], residual), np.where(use, r, np.nan), use


def control_grid(H0, uv, n=5):
    """n x n label points spanning the tracked part of the label (inner 90 %), front-facing on a cylinder."""
    lo, hi = np.percentile(uv, 5, axis=0), np.percentile(uv, 95, axis=0)
    if hasattr(H0, "axis_x"):
        lo[0], hi[0] = max(lo[0], H0.axis_x - 0.8 * H0.r), min(hi[0], H0.axis_x + 0.8 * H0.r)
    g = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)), -1).reshape(-1, 2)
    # only where the label is tracked (a tilted label leaves its box's corners empty)
    from scipy.spatial import cKDTree
    step = float(min((hi - lo) / max(n - 1, 1)))
    near = cKDTree(uv).query(g)[0] <= 0.75 * step
    return g[near] if near.sum() >= 6 else g


def impacts(C, k=5.0, floor=3.0):
    """Frames where the motion kinks (control-point acceleration far above the clip's usual): smoothing stops there."""
    a = np.full(len(C), 0.0)
    a[1:-1] = np.linalg.norm(C[2:] - 2 * C[1:-1] + C[:-2], axis=-1).mean(1)
    return np.nonzero(a > max(floor, k * np.median(a[1:-1]) if len(C) > 2 else floor))[0]


def smooth_controls(C, q, breaks=(), half=3):
    """Centred local quadratic per control coordinate (T x K x 2), tricube x frame-quality weights; windows end at
    breaks (impact frames, which keep their own fit). Zero lag: a linear or quadratic motion passes unchanged."""
    T = len(C)
    seg = np.zeros(T, int)
    for b in sorted(breaks):
        seg[b:] += 1
        seg[b + 1:] += 1                       # the break frame is a segment of its own
    out = C.copy()
    for t in range(T):
        idx = np.array([s for s in range(max(0, t - half), min(T, t + half + 1)) if seg[s] == seg[t]])
        if len(idx) < 3:
            continue
        dt = (idx - t).astype(float)
        wt = (1 - (np.abs(dt) / (half + 1)) ** 3) ** 3 * np.maximum(q[idx], 1e-3)
        deg = 2 if len(idx) >= 5 else 1
        B = np.vander(dt, deg + 1, increasing=True) * np.sqrt(wt)[:, None]
        Y = C[idx].reshape(len(idx), -1) * np.sqrt(wt)[:, None]
        out[t] = np.linalg.lstsq(B, Y, rcond=None)[0][0].reshape(C.shape[1:])
    return out


def model_spec(base, G, P):
    s = {k: v for k, v in base.items() if k in ("model", "cylinder", "tube", "camera", "perspective", "perspective_reg")}
    s["points"] = [{"ref": g.tolist(), "scene": p.tolist()} for g, p in zip(G, P)]
    return s


def fit_points(base, G, P, residual=None):
    """The model (plus the residual correction) through control points: the per-frame mapping after smoothing."""
    return with_residual(_geo_fit(model_spec(base, G, P))[0], G, P, residual)


def sub_mappings(base, G, Cs, t, shutter, step_px=0.75, max_n=12, residual=None):
    """Mappings across frame t's shutter (a fraction of the frame interval, centred on t): the smoothed control
    points moved toward the previous and the next frame, refitted. As many as the motion needs for samples about
    step_px apart (1 when the label barely moves)."""
    T = len(Cs)
    nxt, prv = Cs[min(t + 1, T - 1)] - Cs[t], Cs[max(t - 1, 0)] - Cs[t]
    span = 0.5 * shutter * (np.linalg.norm(nxt, axis=1).max() + np.linalg.norm(prv, axis=1).max())
    n = int(np.clip(np.ceil(span / step_px) + 1, 1, max_n))
    if n == 1 or shutter <= 0:
        return [fit_points(base, G, Cs[t], residual)], 0.0
    out = []
    for s in np.linspace(-0.5 * shutter, 0.5 * shutter, n):
        P = Cs[t] + (s * nxt if s >= 0 else -s * prv)
        out.append(fit_points(base, G, P, residual))
    return out, float(span)


def _stop_reason(t_out, t_in, valid, ok, tracks, vis, q, fin, rig, min_inliers, max_rms, max_rigid):
    """Why the held run ends where it does, at one end: the first frame past it (t_out; None at the shot range's own
    edge) against the run's last frame (t_in). Checked in this order, every criterion that fails listed:
      cut           the tracks jump at once: their median move from t_in is over 5x the run's usual frame-to-frame
                    move (and 20 px), or their visibility halves from one frame to the next;
      product left  too few tracks hold (occluded, out of frame, turned away) without a jump;
      morph         the rigid model can't follow the tracks (the generator redraws the product);
      misfit        only the final mapping misses (the mesh or smoothing can't follow).
    -> {"frame", "reason", "failed", "values"}: what an autopilot needs to choose between chaining the next shot
    (the product continues) and re-acquiring the label on it (a cut)."""
    if t_out is None:
        return {"frame": None, "reason": "shot edge", "failed": [], "values": {}}
    trk = tracks[:, ok] if ok is not None else tracks
    vv = vis[:, ok] if ok is not None else vis
    both = (vv[t_out] > 0.5) & (vv[t_in] > 0.5)
    jump = float(np.median(np.linalg.norm(trk[t_out][both] - trk[t_in][both], axis=1))) if both.sum() >= 8 else float("inf")
    steps = []
    for a_, b_ in zip(valid[:-1], valid[1:]):
        m = (vv[a_] > 0.5) & (vv[b_] > 0.5)
        if m.sum() >= 8:
            steps.append(float(np.median(np.linalg.norm(trk[b_][m] - trk[a_][m], axis=1))))
    usual = float(np.median(steps)) if steps else 0.0
    vis_in, vis_out = float((vv[t_in] > 0.5).mean()), float((vv[t_out] > 0.5).mean())
    failed = []
    if (np.isfinite(jump) and jump > max(5.0 * usual, 20.0)) or vis_out < 0.5 * vis_in:
        failed.append("cut")
    if q[t_out] < min_inliers:
        failed.append("product left")
    if rig[t_out] > max_rigid:
        failed.append("morph")
    if fin[t_out] > max_rms:
        failed.append("misfit")
    def r3(x):
        return round(float(x), 3) if np.isfinite(x) else None
    return {"frame": int(t_out), "reason": failed[0] if failed else "misfit", "failed": failed,
            "values": {"inliers": r3(q[t_out]), "min_inliers": min_inliers, "rms_px": r3(fin[t_out]), "max_rms": max_rms,
                       "rigid_rms_px": r3(rig[t_out]), "max_rigid": max_rigid, "track_jump_px": r3(jump),
                       "usual_step_px": r3(usual), "visible_before": r3(vis_in), "visible_after": r3(vis_out)}}


def fit_shot(base, H0, xy0, tracks, vis, wmin=0.5, reject=2.5, half=3, cuts=(), residual=None, ngrid=5, anchor=None,
             min_inliers=0.5, max_rms=3.0, max_rigid=12.0, shear_max=None, morph_rigid=(), morph_skip=()):
    """All frames of a shot. -> dict with raw and smoothed mappings, per-frame errors, impact frames, and `valid`:
    the frames around the anchor the tracks hold (enough inliers, misfit of the final mapping below max_rms, of the
    rigid model alone below max_rigid) until a cut, a morph or the product leaving breaks them."""
    uv, ok = label_coords(H0, xy0)
    T = len(tracks)
    raw, used, rig, fin = [], [], [], []
    for t in range(T):
        H, r, u = fit_frame(base, uv[ok], tracks[t][ok], vis[t][ok], wmin, reject, residual=residual)
        raw.append(H)
        used.append(u)
        rig.append(float(np.sqrt(np.nanmean(r ** 2))) if np.isfinite(r).any() else np.inf)
        fin.append(float(np.sqrt(np.mean(np.sum((geo.apply_H(H, uv[ok][u]) - tracks[t][ok][u]) ** 2, 1))))
                   if H is not None and u.any() else np.inf)
    good = [t for t in range(T) if raw[t] is not None]
    for t in range(T):                          # a frame without enough tracks borrows its nearest fit
        if raw[t] is None:
            raw[t] = raw[min(good, key=lambda s: abs(s - t))]
    G = control_grid(H0, uv[ok], ngrid)
    C = np.stack([geo.apply_H(H, G) for H in raw])
    q = np.array([u.mean() for u in used])
    good_t = (q >= min_inliers) & (np.array(fin) <= max_rms) & (np.array(rig) <= max_rigid)
    a = int(anchor if anchor is not None else np.argmax(q))
    lo, hi = a, a
    while lo > 0 and good_t[lo - 1]:
        lo -= 1
    while hi < T - 1 and good_t[hi + 1]:
        hi += 1
    valid = list(range(lo, hi + 1))
    stops = {"start": _stop_reason(lo - 1 if lo > 0 else None, lo, valid, ok, tracks, vis, q, fin, rig,
                                   min_inliers, max_rms, max_rigid),
             "end": _stop_reason(hi + 1 if hi < T - 1 else None, hi, valid, ok, tracks, vis, q, fin, rig,
                                 min_inliers, max_rms, max_rigid)}
    cuts = set(cuts) | ({lo - 1} if lo > 0 else set()) | ({hi + 1} if hi < T - 1 else set())
    br = sorted(set(impacts(C).tolist()) | cuts)
    Cs = smooth_controls(C, q, br, half)
    sm = [fit_points(base, G, Cs[t], residual) for t in range(T)]
    # smoothing must not argue with the tracks: generated video judders (the whole product jumps back and forth by
    # many px between frames), and a frame whose smoothed mapping fits its tracks worse than its own fit keeps its own
    kept_raw = []
    for t in range(T):
        u = used[t]
        if raw[t] is None or not u.any():
            continue
        e_raw = np.sqrt(np.mean(np.sum((geo.apply_H(raw[t], uv[ok][u]) - tracks[t][ok][u]) ** 2, 1)))
        e_sm = np.sqrt(np.mean(np.sum((geo.apply_H(sm[t], uv[ok][u]) - tracks[t][ok][u]) ** 2, 1)))
        if e_sm > e_raw + max(0.3, 0.25 * e_raw):
            sm[t] = raw[t]
            Cs[t] = C[t]
            kept_raw.append(t)
    # a morph frame (the generator blends two poses) drags an affine camera into a shear spike, and the print would
    # shear with it: where the final mapping's shear (or one axis' stretch) jumps past shear_max degrees (shear_max
    # / 40) from the frames around it, the frame takes its shape from the frames either side (control points
    # interpolated in time) and only its position, scale and in-plane turn from its own tracks (a similarity fit,
    # which can't shear). No mesh correction there: it would follow the morph. Validity stays the affine fit's.
    morph, morph_source = [], {}
    if shear_max is not None and base.get("camera", "ortho") == "affine":
        S = [shear_deg(getattr(sm[t], "base", sm[t])) for t in range(T)]
        if all(x is not None for x in S):
            S = np.array(S)
            for t in range(T):
                nb = [s for s in range(max(0, t - 4), min(T, t + 5)) if s in valid]
                if t not in valid or len(nb) < 4 or t == a or t in morph_skip:   # the anchor frame is the truth's own
                    continue
                d = np.abs(S[t] - np.median(S[nb], 0))
                if d[0] > shear_max or d[1] > shear_max / 40.0:
                    morph.append(t)
            ok_t = [t for t in valid if t not in morph]
            for t in morph:
                if t in morph_rigid:
                    # a product turning through bad tracks (an occluder rising, print sliding): a rigid camera
                    # (scaled orthographic: spin, tilt, scale, no shear or stretch) through its own tracks keeps
                    # the turn the neighbours' shape would freeze
                    u = used[t]
                    Hr, _ = _geo_fit(_spec(dict(base, camera="ortho"), uv[ok][u], tracks[t][ok][u], np.sqrt(vis[t][ok][u])))
                    morph_source[t] = sm[t]
                    sm[t], Cs[t] = Hr, geo.apply_H(Hr, G)
                    continue
                p_ = max([s for s in ok_t if s < t], default=None)
                n_ = min([s for s in ok_t if s > t], default=None)
                if p_ is None and n_ is None:
                    continue
                if p_ is None or n_ is None:
                    Ci = Cs[p_ if n_ is None else n_]
                else:
                    Ci = Cs[p_] + (Cs[n_] - Cs[p_]) * (t - p_) / (n_ - p_)
                Hi = _geo_fit(model_spec(base, G, Ci))[0]
                u = used[t]
                X, Y = geo.apply_H(Hi, uv[ok][u]), tracks[t][ok][u]
                wt = np.sqrt(vis[t][ok][u])
                # weighted similarity X -> Y (Umeyama, no reflection)
                mx, my = (wt[:, None] * X).sum(0) / wt.sum(), (wt[:, None] * Y).sum(0) / wt.sum()
                Xc, Yc = X - mx, Y - my
                Sxy = (wt[:, None] * Yc).T @ Xc / wt.sum()
                Us, Ds, Vt = np.linalg.svd(Sxy)
                Dm = np.diag([1.0, np.sign(np.linalg.det(Us @ Vt))])
                Rm = Us @ Dm @ Vt
                sc = np.trace(np.diag(Ds) @ Dm) / ((wt * (Xc ** 2).sum(1)).sum() / wt.sum())
                M, tv = sc * Rm, my - sc * Rm @ mx
                morph_source[t] = sm[t]            # where the source's own print is (render reads its light there)
                sm[t] = type(Hi)(Hi.u0, Hi.Ru, Hi.R, Hi.k, M @ Hi.A, M @ Hi.a + tv)
                Cs[t] = geo.apply_H(sm[t], G)
    err = []
    for t in range(T):
        u = used[t]
        r = np.linalg.norm(geo.apply_H(sm[t], uv[ok][u]) - tracks[t][ok][u], axis=1)
        err.append({"frame": t, "inliers": int(u.sum()), "of": int(ok.sum()), "valid": t in valid,
                    "rigid_rms_px": round(rig[t], 3) if np.isfinite(rig[t]) else None,
                    "rms_px": round(float(np.sqrt(np.mean(r ** 2))), 3) if len(r) else None,
                    "p90_px": round(float(np.percentile(r, 90)), 3) if len(r) else None,
                    "smooth_shift_px": round(float(np.linalg.norm(Cs[t] - C[t], axis=1).max()), 3)})
    return {"raw": raw, "smooth": sm, "uv": uv, "on_surface": ok, "controls": G, "C_raw": C, "C_smooth": Cs,
            "impacts": br, "errors": err, "valid": valid, "kept_raw": kept_raw, "morph": morph, "stops": stops,
            "morph_source": morph_source}


def silhouette_snap(H, matte, v_range, step=24.0, max_px=30.0, inner=(-1.0, -0.6, -0.2, 0.2, 0.6, 1.0), lam=0.02,
                    hold=False, front="mean"):
    """A tube's mapping corrected so its limbs land on the product's matte. The rigid tube tapers linearly, but a
    squeezed tube or a generated bottle doesn't: its real sides drift 10-25 px from the model's at the label's ends,
    and the source shows between them as an edge. Per label row: the matte's two edges along the row (through the
    point facing the camera); each side's offset from the model's limb, fitted along the label by a robust quadratic
    (rows an occluder cuts, or edges further than max_px, drop out); then a smooth scene-space correction (the
    engine's Residual) that maps the model's [left, right] limbs onto the matte's, linearly across the row (so the
    print in the middle moves only by the row's mean offset). front "track" (V57): the print facing the camera stays
    where the tracks put it: the two sides' common offset is the cylinder's axis off by it with the print held, a
    correction of (1 - cos theta) times it across the row (none at the front, all of it at the limbs); their difference
    stays linear (the radius). On a squat jar seen from above the fit's axis drifted 6-9 px from the matte's centre
    while its print sat on the tracks, and the mean offset carried the print off the frame's own by as much.
    -> (mapping, report)"""
    from scipy.ndimage import map_coordinates
    Hb = H
    while hasattr(Hb, "base"):
        Hb = Hb.base
    if not hasattr(Hb, "Ru"):
        return H, {"snap": "not a tube"}
    d = np.cross(Hb.A[0], Hb.A[1])
    ths = np.linspace(-np.pi, np.pi, 1441)
    m = matte.astype(float)
    rows, eL, eR, geo_rows = [], [], [], []
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
        ss = np.arange(sL - max_px - 2, sR + max_px + 2, 0.5)
        q = pc[None] + ss[:, None] * n[None]
        if (q < 1).any() or (q[:, 0] > m.shape[1] - 2).any() or (q[:, 1] > m.shape[0] - 2).any():
            continue                                   # the row leaves the frame
        val = map_coordinates(m, [q[:, 1], q[:, 0]], order=1)
        ins = val > 0.5
        c = int(np.argmin(np.abs(ss)))
        if not ins[c]:
            continue
        lo = c
        while lo > 0 and ins[lo - 1]:
            lo -= 1
        hi = c
        while hi < len(ss) - 1 and ins[hi + 1]:
            hi += 1
        if lo == 0 or hi == len(ss) - 1:
            continue                                   # the matte runs past the search: no edge there
        # sub-pixel crossings of 0.5
        el = ss[lo - 1] + (0.5 - val[lo - 1]) / max(val[lo] - val[lo - 1], 1e-6) * 0.5
        er = ss[hi] + (val[hi] - 0.5) / max(val[hi] - val[hi + 1], 1e-6) * 0.5
        rows.append(v)
        eL.append(el - sL)
        eR.append(er - sR)
        geo_rows.append((v, thL, thR, thc, n))
    rep = {"rows": len(rows)}
    if len(rows) < 6:
        return H, dict(rep, snap="too few rows")
    rows, eL, eR = np.array(rows), np.array(eL), np.array(eR)

    def robust_quad(y):
        ok = np.abs(y) <= max_px
        for _ in range(3):
            if ok.sum() < 5:
                return None, ok
            c = np.polyfit(rows[ok], y[ok], 2)
            r = y - np.polyval(c, rows)
            mad = 1.4826 * np.median(np.abs(r[ok] - np.median(r[ok]))) + 0.5
            ok = (np.abs(y) <= max_px) & (np.abs(r) <= 3.0 * mad)
        return c, ok

    cL, okL = robust_quad(eL)
    cR, okR = robust_quad(eR)

    def held(c, ok):
        # hold (opt-in): only between the rows the fit used (an occluder hides the rest: a quadratic run past them
        # can swing by 100 px), and never past max_px. Without it the quadratic runs on, as renders made before it did
        if c is None:
            return lambda v: 0.0 * np.asarray(v, float)
        if not hold:
            return lambda v: np.polyval(c, v)
        lo, hi = rows[ok].min(), rows[ok].max()
        return lambda v: np.clip(np.polyval(c, np.clip(v, lo, hi)), -max_px, max_px)

    fL, fR = held(cL, okL), held(cR, okR)
    C, D = [], []
    for v, thL, thR, thc, n in geo_rows:
        a, b = float(fL(v)), float(fR(v))
        xs = inner if front != "track" else sorted(set(inner) | {0.0})
        for x in xs:                        # x: -1 at the left limb, +1 at the right; screen-linear in x
            th = thc + np.arcsin(np.clip(x, -1, 1)) * (thR - thL) / np.pi
            C.append(H.forward([[Hb.u0 + Hb.Ru * th, v]])[0])
            if front == "track":
                D.append((0.5 * (a + b) * (1 - np.sqrt(max(1 - x * x, 0.0))) + 0.5 * (b - a) * x) * n)
            else:
                D.append((a * (1 - x) / 2 + b * (1 + x) / 2) * n)
    C, D = np.array(C), np.array(D)
    span = np.linalg.norm(C[1:] - C[:-1], axis=1)
    sig = float(max(np.median(span[span > 0]) * 1.5, 6.0)) if (span > 0).any() else 12.0
    Hs = geo.Residual(H, C, D, sig, lam)
    Hs.__dict__["limb_snap"] = True                 # (print_mapping: the mapping under it orients the print)
    if front == "track":
        rep["front"] = "track"
    rep.update(left_px=[round(float(fL(rows.min())), 1), round(float(fL(np.median(rows))), 1), round(float(fL(rows.max())), 1)],
               right_px=[round(float(fR(rows.min())), 1), round(float(fR(np.median(rows))), 1), round(float(fR(rows.max())), 1)],
               rows_used=[int(okL.sum()), int(okR.sum())], sigma=round(sig, 1))
    return Hs, rep


def axis_ends(H, matte, v_mid, reach=600.0):
    """Where the product's matte ends along the label's axis (through the point facing the camera), in label v:
    (top, bottom), or None. The ends (a crimp, a cap) are landmarks along the axis the print can't give on a blurred
    or morphing frame."""
    from scipy.ndimage import map_coordinates
    Hb = H
    while hasattr(Hb, "base"):
        Hb = Hb.base
    if not hasattr(Hb, "Ru"):
        return None
    d = np.cross(Hb.A[0], Hb.A[1])
    uc = Hb.u0 + Hb.Ru * np.arctan2(d[0], d[2])
    vs = np.arange(v_mid - 1200.0 - reach, v_mid + 1200.0 + reach, 1.0)
    P = H.forward(np.c_[np.full(len(vs), uc), vs])
    ok = (P[:, 0] > 0) & (P[:, 1] > 0) & (P[:, 0] < matte.shape[1] - 1) & (P[:, 1] < matte.shape[0] - 1)
    ins = np.zeros(len(vs), bool)
    ins[ok] = map_coordinates(matte.astype(float), [P[ok, 1], P[ok, 0]], order=1) > 0.5
    c = int(np.argmin(np.abs(vs - v_mid)))
    if not ins[c]:
        return None
    lo = hi = c
    while lo > 0 and ins[lo - 1]:
        lo -= 1
    while hi < len(vs) - 1 and ins[hi + 1]:
        hi += 1
    if lo == 0 or hi == len(vs) - 1 or not ok[lo - 1] or not ok[hi + 1]:
        return None                                   # the product runs off the frame: no end there
    return float(vs[lo]), float(vs[hi])


# ---------------------------------------------------------------- the label against the body (V39)

BODY_SNAP = {"top": "auto", "bottom": "auto", "reach": [0.37, 0.42], "across": 0.65, "inside": 0.05, "window": 0.5,
             "median": 5, "outlier_k": 4.0, "stretch": [0.9, 1.1], "roll_max": 4.0, "shift_max": 0.15, "turn": False,
             "turn_max": 0.08}


def _edge_profile(p, ax, lum, reach):
    """The frame's luminance from p along ax, every 0.25 px up to reach px, lightly smoothed: (s, its gradient)."""
    from scipy.ndimage import gaussian_filter1d, map_coordinates
    s = np.arange(0, reach, 0.25)
    g = gaussian_filter1d(map_coordinates(lum, [p[1] + s * ax[1], p[0] + s * ax[0]], order=1), 1.0)
    return s, np.gradient(g)


def _edge_along(p, ax, lum, matte, reach, kind, inside=0.0, lo=None):
    """px from p along ax to the body's edge: "matte" the matte's end (the first sample off it), "rise" / "fall" the
    strongest brightening / darkening of the frame's luminance (a white collar or cap seam / a dark one), "edge"
    either, "dark" / "light" the darkest / brightest line (a cap's seam line), searched from `inside` px back inside
    the label. lo: search only from there (px) on, up to reach (a window round the expected gap). -> px (negative:
    inside the label), or None (no matte end within reach, or the matte already ends before lo)"""
    from scipy.ndimage import gaussian_filter1d, map_coordinates
    if kind == "matte":
        s0 = 0.0 if lo is None else max(0.0, lo)
        s = np.arange(s0, reach, 0.25)
        if not len(s):
            return None
        on = map_coordinates(matte, [p[1] + s * ax[1], p[0] + s * ax[0]], order=1) >= 0.5
        if lo is not None and not on[0]:
            return None
        off = np.nonzero(~on)[0]
        return float(s[off[0]]) if len(off) else None
    if kind in ("dark", "light"):
        s = np.arange(-inside if lo is None else lo, reach, 0.25)
        if not len(s):
            return None
        g = gaussian_filter1d(map_coordinates(lum, [p[1] + s * ax[1], p[0] + s * ax[0]], order=1), 0.8)
        return float(s[int(np.argmin(g) if kind == "dark" else np.argmax(g))])
    s, d = _edge_profile(p, ax, lum, reach)
    if lo is not None:
        k = s >= lo
        if not k.any():
            return None
        s, d = s[k], d[k]
    i = int(np.argmax(d if kind == "rise" else -d if kind == "fall" else np.abs(d)))
    return float(s[i])


def body_edges(H, lum, matte, v_range, uc, reach=(0.37, 0.42)):
    """Which edge bounds the body above and below the label (body_snap's "auto", decided on the reference frame):
    "matte" where the matte ends within reach (a crimp, a base), else "rise" or "fall", the stronger of the frame's
    brightening and darkening there (a white collar, a dark cap). -> {"top": kind, "bottom": kind}"""
    v0, v1 = v_range
    p = geo.apply_H(H, np.array([[uc, v0], [uc, v1]]))
    L = float(np.linalg.norm(p[1] - p[0]))
    ax = (p[1] - p[0]) / max(L, 1e-6)
    out = {}
    for side, q, d, r in (("top", p[0], -ax, reach[0] * L), ("bottom", p[1], ax, reach[1] * L)):
        if _edge_along(q, d, lum, matte, r, "matte") is not None:
            out[side] = "matte"
        else:
            g = _edge_profile(q, d, lum, r)[1]
            out[side] = "rise" if g.max() >= -g.min() else "fall"
    return out


def print_mapping(H):
    """The mapping that orients a rendered label's print: under the limb snap (which only moves its rows sideways onto
    the matte, shearing the label's sides, never turning its text lines), else H itself."""
    return H.base if getattr(H, "__dict__", {}).get("limb_snap") else H


def body_measure(H, lum, matte, v_range, uc, top="rise", bottom="matte", reach=(0.37, 0.42), across=0.65, inside=0.05,
                 H_print=None, ref=None, window=0.5):
    """The label against the product's body on one frame, along the label's axis (the line u = uc through its band
    v_range). lum: the frame's luminance (mean of R, G, B, levels); matte: the product's matte (0/1 float).
    -> {"top": px from the label's top edge up to the body's edge above it (a collar seam or a cap: `top`), "bot": px
    from its bottom edge down to the body's end (`bottom`), "L": the label's length, px, "roll": deg from the print's
    axis (H_print's, default print_mapping(H)) to the body's centreline (the matte limbs' midpoints across the label's
    middle 60 %; + = the body runs clockwise from the print on screen: turning the label by +roll puts it on the
    body), "pivot": the label's middle, px}, or None (no body end within reach).
    reach: how far to search above / below, as shares of the label's length; across: half the search across (widened
    while the body runs past it, V61); inside:
    how far back inside the label a "dark" / "light" line is searched (share of its length). ref: the reference
    frame's measure: then each edge is searched only round the gap the reference's proportion predicts for this frame
    (+- window x that gap + 2 % of the label's length), so a frame can't jump to another edge (a cap's top, a
    reflection) or to a split in the matte."""
    from scipy.ndimage import map_coordinates
    v0, v1 = v_range
    p = geo.apply_H(H, np.array([[uc, v0], [uc, v1], [uc, 0.5 * (v0 + v1)]]))
    L = float(np.linalg.norm(p[1] - p[0]))
    if L < 4:
        return None
    ax = (p[1] - p[0]) / L
    n = np.array([ax[1], -ax[0]])
    win = [(None, reach[0] * L), (None, reach[1] * L)]
    if ref is not None:
        for i, k in enumerate(("top", "bot")):
            e = ref[k] * L / ref["L"]
            w = max(3.0, window * abs(e) + 0.02 * L)
            win[i] = (e - w, e + w)
    tp = _edge_along(p[0], -ax, lum, matte, win[0][1], top, inside * L, win[0][0]) if top not in (None, "none") else 0.0
    bt = _edge_along(p[1], ax, lum, matte, win[1][1], bottom, inside * L, win[1][0]) if bottom not in (None, "none") else 0.0
    if tp is None or bt is None:
        return None

    def sections(R):
        S_ = np.arange(-R, R + 0.5, 0.5)
        return S_, [np.nonzero(map_coordinates(matte, [c[1] + S_ * n[1], c[0] + S_ * n[0]], order=1) > 0.5)[0]
                    for c in [p[0] + (p[1] - p[0]) * fr for fr in np.linspace(0.2, 0.8, 9)] + [p[2]]]
    # the search across: `across` x the label's length, widened while the body runs on past its end. A squat product
    # is wider than that (a jar's body, 0.6 x its label's length each side): sections cut at the window's end leaned
    # the centreline halfway toward the label's own axis (V61)
    R = max(60.0, np.round(across * L * 2) / 2)
    S_, ons = sections(R)
    for _ in range(4):
        if not any(len(on) >= 10 and (on.min() == 0 or on.max() == len(S_) - 1) for on in ons):
            break
        R = np.round(1.5 * R * 2) / 2
        S_, ons = sections(R)
    mids = []
    for fr, on in zip(np.linspace(0.2, 0.8, 9), ons[:9]):
        c = p[0] + (p[1] - p[0]) * fr
        if len(on) >= 10:
            mids.append(c + n * 0.5 * (S_[on.min()] + S_[on.max()]))
    roll = np.nan
    if len(mids) >= 5:
        m = np.array(mids)
        cc = m - m.mean(0)
        bax = np.linalg.eigh(cc.T @ cc)[1][:, 1]
        q = geo.apply_H(print_mapping(H) if H_print is None else H_print, np.array([[uc, v0], [uc, v1]]))
        pa = (q[1] - q[0]) / max(float(np.linalg.norm(q[1] - q[0])), 1e-6)
        bax = bax if bax @ pa > 0 else -bax
        roll = float(np.degrees(np.arctan2(pa[0] * bax[1] - pa[1] * bax[0], pa @ bax)))
    # the label point under the body's centreline at the label's middle (its u, through the tube): the label's turn
    # against the body (body_snap.turn, V59)
    u_body = np.nan
    Hb = H
    while hasattr(Hb, "base"):
        Hb = Hb.base
    on = ons[9]
    if len(on) >= 10 and hasattr(Hb, "inverse"):
        cm = p[2] + n * 0.5 * (S_[on.min()] + S_[on.max()])
        uu, _, okk = Hb.inverse(np.array([cm[0]]), np.array([cm[1]]))
        if okk is None or bool(np.asarray(okk).ravel()[0]):
            u_body = float(np.asarray(uu).ravel()[0])
    return {"top": tp, "bot": bt, "L": L, "roll": roll, "pivot": p[2].tolist(), "u_body": u_body}


def body_offsets(m, ref):
    """One frame's measure against the reference frame's (the keyframe's): px the label's top and bottom edges sit
    off where the reference's proportion of the body puts them (+ = the gap is larger: the label too low at the top,
    too high at the bottom), the roll off the reference's (deg), and the label's share of the body over the
    reference's (stretch)."""
    B, Bk = m["top"] + m["L"] + m["bot"], ref["top"] + ref["L"] + ref["bot"]
    return {"top_px": m["top"] - B * ref["top"] / Bk, "bottom_px": m["bot"] - B * ref["bot"] / Bk,
            "roll_deg": m["roll"] - ref["roll"] if np.isfinite(m["roll"]) and np.isfinite(ref["roll"]) else np.nan,
            "stretch": (m["L"] / B) / (ref["L"] / Bk),
            "turn_px": m.get("u_body", np.nan) - ref.get("u_body", np.nan)}


def body_corrections(meas, ref, v_range, median=5, outlier_k=4.0, stretch=(0.9, 1.1), roll_max=4.0, shift_max=0.15,
                     turn=False, turn_max=None, orbit=None):
    """Per frame, the tube correction that puts the label back at the reference frame's proportion of the body and its
    print at the reference's angle to the body's centreline: {t: (f, p, roll_deg, du)} for Tube v -> p + f v (label
    px), a turn by roll_deg about the label's middle, and (turn) a turn round the axis by du label px that keeps the
    label point under the body's centreline the reference's (body_snapped), or, with `orbit` ({t: label px}, the
    scene's turn about the product's axis since the reference frame, imgvideo.orbit), the reference's turned by the
    orbit: the label turns with the camera's real orbit and only the print's own slide is taken out (V65).
    meas: {t: body_measure}. Each series is cleared of outliers (a sparkle trail splitting the matte) against a
    `median`-frame running median, then median-filtered, then clamped (stretch, roll_max deg, shift_max of the
    label's length). A frame measured as None takes its neighbours' correction."""
    from scipy import ndimage as ndi
    v0, v1 = v_range
    ts = sorted(meas)
    raw = []
    for t in ts:
        m = meas[t]
        if m is None:                                    # not measured here: its neighbours' correction
            raw.append([np.nan, np.nan, np.nan, np.nan])
            continue
        o = body_offsets(m, ref)
        sig = m["L"] / (v1 - v0)
        vt = v0 - o["top_px"] / sig                      # where the label's top should go, in the old label v
        vb = v1 + o["bottom_px"] / sig
        f = (vb - vt) / (v1 - v0)
        du = -(o["turn_px"] - (orbit.get(t, 0.0) if orbit else 0.0)) if turn else 0.0
        raw.append([f, vt - f * v0, o["roll_deg"], du])   # (+roll: onto the body's axis)
    arr = np.array(raw, float).reshape(-1, 4)
    if not len(arr):
        return {}
    for j in range(4):
        x = arr[:, j]
        if not np.isfinite(x).any():
            arr[:, j] = 0.0 if j else 1.0
            continue
        med = ndi.median_filter(np.nan_to_num(x, nan=np.nanmedian(x)), median, mode="nearest")
        bad = ~np.isfinite(x) | (np.abs(x - med) > outlier_k * 1.4826 * np.nanmedian(np.abs(x - med)) + 1e-9)
        x = np.where(bad, med, x)
        arr[:, j] = ndi.median_filter(x, median, mode="nearest")
    arr[:, 0] = np.clip(arr[:, 0], *stretch)
    arr[:, 1] = np.clip(arr[:, 1], -shift_max * (v1 - v0), shift_max * (v1 - v0))
    arr[:, 2] = np.clip(arr[:, 2], -roll_max, roll_max)
    if turn_max is not None:
        arr[:, 3] = np.clip(arr[:, 3], -turn_max, turn_max)
    return {t: (float(a[0]), float(a[1]), float(a[2]), float(a[3])) for t, a in zip(ts, arr)}


def body_snapped(H, p, f, roll_deg, pivot, du=0.0):
    """H with its tube's label moved along the axis and stretched (Tube v -> p + f v), turned by roll_deg about pivot
    (scene px) and turned round the axis by du label px (u0 + du); a Residual on top (the fit's) keeps its scene-space
    correction."""
    import copy
    from .surface import Tube
    if hasattr(H, "base"):
        R = copy.copy(H)
        R.__dict__["base"] = body_snapped(H.base, p, f, roll_deg, pivot, du)
        return R
    A = H.A.copy()
    a = H.a + H.A[:, 1] * p
    A[:, 1] = A[:, 1] * f
    c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    Rm = np.array([[c, -s], [s, c]])
    pv = np.asarray(pivot, float)
    return Tube(H.u0 + du, H.Ru, H.R + H.k * p, H.k * f, Rm @ A, Rm @ (a - pv) + pv)
