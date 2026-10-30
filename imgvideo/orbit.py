"""The camera's orbit round a product that stands on a surface (V65), from the scene itself: the turn hold
(render.body_snap.turn) can't tell the product's print sliding round it from the camera orbiting it, but the ground
the product stands on turns about the product's axis with the orbit, and only with it.

Points on the ground round the product's base (a ring between `ring` base radii, in front of the base's far edge)
are tracked with the shot's tracker from the anchor frame, read in the ground plane normalised by each frame's base
ellipse (from the matte: the half width over its lowest rows, the bottom edge fitted as an ellipse arc), and the
rotation about the base's centre is fitted per frame (2D Procrustes). On the one workflow's jar the ground turned
5.9 deg over the push-in, and the generator's print turned with it within 1.7 label px rms."""
import numpy as np
from scipy import ndimage as ndi

ORBIT = {
    "step": 8,              # px: the query grid on the anchor frame
    "ring": [1.05, 1.9],    # the ground between these base radii (in the ground plane), off the product's matte
    "front": -0.3,          # ... from this far behind the base's centre line (the far side is hidden by the product)
    "res_max": 640,         # px: the tracker works at this longer side at most
    "min_points": 200,      # a frame is read with this many visible points or more
    "max_residual": 0.05,   # ... and the rotation fits them within this (base radii, median)
    "min_ratio": 0.12,      # the base ellipse's minor / major axis: under it the ground is seen edge on
}


def base_ellipse(m):
    """(xc, yc, a, b) of the product's base seen from above, from its matte: a = the half width over the lowest rows
    (75-90 % down the matte), the bottom edge y(x) = yc + b sqrt(1 - ((x - xc) / a)^2) fitted over its middle 80 %.
    None when the matte is empty or touches the frame's bottom (the base isn't seen)."""
    ys, xs = np.nonzero(m)
    if len(ys) < 500 or ys.max() >= m.shape[0] - 2:
        return None
    y0, y1 = ys.min(), ys.max()
    rows = range(int(y1 - 0.25 * (y1 - y0)), int(y1 - 0.10 * (y1 - y0)))
    L = np.array([np.nonzero(m[y])[0][[0, -1]] for y in rows if m[y].any()], float)
    if len(L) < 3:
        return None
    xc, a = float(L.mean()), float(0.5 * (L[:, 1] - L[:, 0]).max())
    cols = np.arange(int(xc - 0.8 * a), int(xc + 0.8 * a))
    cols = cols[(cols >= 0) & (cols < m.shape[1])]
    has = m[:, cols].any(0)
    cols = cols[has]
    if len(cols) < 10:
        return None
    yb = np.array([np.nonzero(m[:, x])[0].max() for x in cols], float)
    s = np.sqrt(np.clip(1 - ((cols - xc) / a) ** 2, 0, 1))
    (yc, b), *_ = np.linalg.lstsq(np.c_[np.ones_like(s), s], yb, rcond=None)
    return xc, float(yc), a, float(b)


def scene_orbit(frames, masks, anchor, cfg=None, track=None):
    """The orbit per frame: {"phi": [rad, the ground's turn about the product's axis from the anchor frame, + = its
    front moving left in the image], "residual": [base radii], "points": [n], "ellipse": [[xc, yc, a, b]],
    "b_a": the anchor's minor / major}, or {"skipped": why}. track: the tracker (imgvideo.track.track_points)."""
    c = dict(ORBIT, **(cfg or {}))
    if track is None:
        from .track import track_points as track
    T = len(frames)
    E = [base_ellipse(np.asarray(masks[t])) for t in range(T)]
    if E[anchor] is None:
        return {"skipped": "the base isn't seen on the anchor frame (no matte, or it runs off the frame's bottom)"}
    xc, yc, a, b = E[anchor]
    if b / a < c["min_ratio"]:
        return {"skipped": f"the ground is seen edge on (the base ellipse {b / a:.2f} of its width)"}
    h, w = np.asarray(frames[0]).shape[:2]
    st = int(c["step"])
    yy, xx = np.mgrid[st // 2:h:st, st // 2:w:st]
    X, Y = (xx - xc) / a, (yy - yc) / b
    r = np.hypot(X, Y)
    off = ~ndi.binary_dilation(np.asarray(masks[anchor]), iterations=8)[yy, xx]
    sel = off & (r > c["ring"][0]) & (r < c["ring"][1]) & (Y > c["front"]) & (yy < h - 8)
    q = np.c_[xx[sel], yy[sel]].astype(np.float32)
    if len(q) < c["min_points"]:
        return {"skipped": f"too little ground round the base ({len(q)} points)"}
    crop = (max(int(xc - (c["ring"][1] + 0.3) * a), 0), max(int(yc - (c["ring"][1] + 0.3) * b) - int(0.2 * a), 0),
            min(int(xc + (c["ring"][1] + 0.3) * a), w), min(int(yc + (c["ring"][1] + 0.3) * b) + 8, h))
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    s = min(1.0, float(c["res_max"]) / max(cw, ch))
    res = (max(int(ch * s) // 8 * 8, 64), max(int(cw * s) // 8 * 8, 64))
    tr, vis, _ = track(frames, q, anchor, crop, res=res)
    P0 = np.c_[(q[:, 0] - xc) / a, (q[:, 1] - yc) / b]
    phi, resid, npts = [], [], []
    for t in range(T):
        e = E[t]
        ok = vis[t] > 0.5
        if e is None or ok.sum() < c["min_points"]:
            phi.append(np.nan); resid.append(np.nan); npts.append(int(ok.sum()))
            continue
        P1 = np.c_[(tr[t, :, 0] - e[0]) / e[2], (tr[t, :, 1] - e[1]) / e[3]]
        p0, p1 = P0[ok], P1[ok]
        f = np.arctan2(np.sum(p0[:, 0] * p1[:, 1] - p0[:, 1] * p1[:, 0]), np.sum(p0[:, 0] * p1[:, 0] + p0[:, 1] * p1[:, 1]))
        R = np.array([[np.cos(f), -np.sin(f)], [np.sin(f), np.cos(f)]])
        rr = float(np.median(np.linalg.norm(p0 @ R.T - p1, axis=1)))
        bad = rr > c["max_residual"]
        phi.append(np.nan if bad else float(f)); resid.append(rr); npts.append(int(ok.sum()))
    phi = np.array(phi, float)
    if np.isfinite(phi).sum() < max(3, T // 2):
        return {"skipped": "the ground's turn didn't fit on most frames", "residual": [None if not np.isfinite(x) else
                                                                                      round(x, 4) for x in resid]}
    # frames that didn't fit: from their neighbours (the orbit is a camera move, smooth)
    idx = np.arange(T)
    okf = np.isfinite(phi)
    phi = np.interp(idx, idx[okf], phi[okf])
    return {"phi": phi.tolist(), "residual": [None if not np.isfinite(x) else round(float(x), 4) for x in resid],
            "points": npts, "ellipse": [None if e is None else [round(float(v), 2) for v in e] for e in E],
            "b_a": round(b / a, 3), "anchor": int(anchor), "res": list(res), "queries": int(len(q))}
