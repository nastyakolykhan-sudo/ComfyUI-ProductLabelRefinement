"""render.glue: the label glued to the product's own motion (pitfall V42; ported from the first real run's job script
stabilize.py, 2026-10-17; the user: "in some frames that don't change much in terms of motion the label moves too
much ... I want the label to be better locked on the product, now it looks like it's fluid and lives its own life").

The engine places the label per frame from geometry: tracks on the garbled print, a per-frame mesh warp, a 7-frame
smoothing that keeps a frame's own fit where the smoothed one fits worse, the body snap and the limb snap onto a SAM
matte. Each is right on average and noisy frame to frame: on near-still frames the label moved 0.4-2.4 px against the
body while the source's own print moved 0.1-0.4. Here the label moves the way the product moves:
  1. motion   frame t -> t+1, an affine fitted to the source's own pixels over the label's area (the print, garbled or
              not, the body's shading and the limbs: all glued to the surface), by robust Lucas-Kanade (high-passed
              luminance, Huber weights, two pyramid levels, started from the placement's own step). A fit with a
              robust rms over rms_max or a median deviation from the placement's step over dev_max px falls back to
              that step
  2. carry    the label's control grid goes from frame to frame along that motion, out from the keyframe's frame
              (exact there) both ways
  3. offset   the placement's offset from the carried grid, per frame, smoothed over time: a local linear fit with
              Gaussian weights (sigma frames) and Huber weights on frames (a one-frame jump of the placement, a sparkle
              trail splitting the matte, gets none), zero-lag. edge "mean": a local mean within 2 sigma of the run's
              ends instead (a line extrapolates the next frames' trend there). Added back: the label moves with the
              product frame to frame and lands where the engine places it over a few frames. The edge can be set
              per end ([start, end]): the review's "ends" check reads the first and last pairs and says which to change
The glued mapping is the placement plus a smooth image-space correction (geometry.Residual through a dense grid over
the label and a margin). The keyframe's frame stays exact."""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, gaussian_filter, map_coordinates, sobel, zoom

from imgpost import geometry as geo

GLUE = {
    "sigma": 4.0,       # frames: the placement's offset from the carried motion, smoothed over this
    "edge": "line",     # "line" | "mean": the smoothing within 2 sigma of the run's ends; [first, last]: per end
    "rms_max": 0.9,     # a motion fit's robust rms (over the template's spread) above this falls back
    "dev_max": 6.0,     # px: a motion fit off the placement's own step by more (median) falls back
    "min_px": 400,      # the label's area on a frame under this: the placement's step
}


def edges(spec):
    """render.glue.edge: "line" | "mean", or [the run's start, its end]. -> (start, end)"""
    e = [spec, spec] if isinstance(spec, str) else list(spec)
    if len(e) != 2 or any(x not in ("line", "mean") for x in e):
        raise ValueError(f"render.glue.edge {spec!r}: \"line\", \"mean\" or [start, end] of those")
    return e[0], e[1]


def lum(u8):
    x = u8.astype(np.float32) / 255.0
    x = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return x @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def hp(x):
    return gaussian_filter(x, 0.8) - gaussian_filter(x, 5.0)


def affine_fit(P, Q):
    """Least-squares affine P -> Q: (A 2x2, b)."""
    X = np.c_[P, np.ones(len(P))]
    M, *_ = np.linalg.lstsq(X, Q, rcond=None)
    return M[:2].T, M[2]


def lk_affine(T, I, mask, A0, b0, iters=30):
    """Affine (A, b) with I(A x + b) ~ T(x) over mask: Gauss-Newton with Huber weights. -> (A, b, robust rms, used)"""
    ys, xs = np.nonzero(mask)
    if len(ys) < 200:
        return None
    X = np.c_[xs, ys].astype(np.float64)
    c = X.mean(0)
    Xc = X - c
    t = T[ys, xs].astype(np.float64)
    Iy, Ix = sobel(I, 0) / 8.0, sobel(I, 1) / 8.0
    A, b = A0.copy(), b0 + A0 @ c - c               # work about the region's centre: W(x) = A (x - c) + c + b'
    p = np.r_[A.ravel() - np.eye(2).ravel(), b]
    for _ in range(iters):
        A = np.eye(2) + p[:4].reshape(2, 2)
        W = Xc @ A.T + c + p[4:]
        iw = map_coordinates(I, [W[:, 1], W[:, 0]], order=1, mode="nearest")
        gx = map_coordinates(Ix, [W[:, 1], W[:, 0]], order=1, mode="nearest")
        gy = map_coordinates(Iy, [W[:, 1], W[:, 0]], order=1, mode="nearest")
        e = iw - t
        s = 1.4826 * np.median(np.abs(e)) + 1e-6
        w = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(e), 1e-9))
        J = np.c_[gx * Xc[:, 0], gx * Xc[:, 1], gy * Xc[:, 0], gy * Xc[:, 1], gx, gy]
        H = (J * w[:, None]).T @ J + 1e-9 * np.eye(6)
        dp = -np.linalg.solve(H, (J * w[:, None]).T @ e)
        p = p + dp
        if np.abs(dp[4:]).max() < 1e-3 and np.abs(dp[:4]).max() < 1e-5:
            break
    A = np.eye(2) + p[:4].reshape(2, 2)
    b = p[4:] + c - A @ c
    return A, b, float(np.sqrt(np.mean((w * e) ** 2)) / (np.std(t) + 1e-9)), len(ys)


def motion(F0, F1, mask, A0, b0):
    """Frame-to-frame affine of the label's area, coarse (half size) then fine."""
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - 40, 0), min(ys.max() + 41, F0.shape[0])
    x0, x1 = max(xs.min() - 40, 0), min(xs.max() + 41, F0.shape[1])
    T, I = hp(lum(F0[y0:y1, x0:x1])), hp(lum(F1[y0:y1, x0:x1]))
    m = mask[y0:y1, x0:x1]
    o = np.array([x0, y0], float)
    b0c = b0 + A0 @ o - o                           # the same affine in crop coordinates
    Ts, Is, ms = zoom(T, 0.5, order=1), zoom(I, 0.5, order=1), zoom(m.astype(np.float32), 0.5, order=0) > 0.5
    r = lk_affine(Ts, Is, ms, A0, b0c * 0.5)
    if r is not None:
        A0, b0c = r[0], r[1] * 2.0
    r = lk_affine(T, I, m, A0, b0c)
    if r is None:
        return None
    A, bc, rms, n = r
    return A, bc + o - A @ o, rms, n


def label_mask(H, LP, shape, matte):
    poly = geo.map_polygon(H, LP)
    im = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in poly], fill=255)
    return (np.asarray(im) > 127) & binary_erosion(matte, iterations=2)


def grid(LP):
    """The control grid: 9 x 13 points over the label and a margin round it (the correction must cover every rendered
    pixel), and which of them lie on the label. -> (G N x 2, inl N bool)"""
    (u0, v0), (u1, v1) = LP.min(0), LP.max(0)
    mu, mv = 0.12 * (u1 - u0), 0.08 * (v1 - v0)
    G = np.stack(np.meshgrid(np.linspace(u0 - mu, u1 + mu, 9), np.linspace(v0 - mv, v1 + mv, 13)), -1).reshape(-1, 2)
    inl = (G[:, 0] >= u0) & (G[:, 0] <= u1) & (G[:, 1] >= v0) & (G[:, 1] <= v1)
    return G, inl


def glue(maps, frame, matte, LP, k, cfg=None):
    """maps {frame: the placement's mapping (label px -> frame px)}; frame(t), matte(t): the source frame and the
    product's matte; LP the label's outline (label px); k the keyframe's frame. -> ({frame: glued grid points}, the
    grid G, its label points inl, {frame: the placement's grid points}, report)"""
    c = dict(GLUE, **(cfg or {}))
    sigma_t = float(c["sigma"])
    ts = sorted(maps)
    if k not in maps:
        raise ValueError(f"render.glue: the keyframe's frame {k} has no placement")
    G, inl = grid(LP)
    P = {t: geo.apply_H(maps[t], G) for t in ts}
    steps = {}                                       # t -> (A, b, ok) for t -> t+1
    for t in ts[:-1]:
        if t + 1 not in maps:
            continue
        A0, b0 = affine_fit(P[t], P[t + 1])
        Ft = frame(t)
        m = label_mask(maps[t], LP, Ft.shape[:2], matte(t))
        r = motion(Ft, frame(t + 1), m, A0, b0) if m.sum() > int(c["min_px"]) else None
        ok = False
        if r is not None:
            A, b, rms, n = r
            pred = P[t] @ A.T + b
            dev = float(np.median(np.linalg.norm(pred - P[t + 1], axis=1)))
            ok = rms < float(c["rms_max"]) and dev < float(c["dev_max"])
            steps[t] = (A, b, ok, round(rms, 3), round(dev, 2))
        if not ok:
            steps[t] = (A0, b0, False, None if r is None else round(r[2], 3), None)
    # carry from the keyframe's frame both ways along the product's motion (the placement's own step where the motion
    # fit failed), then add back the placement's offset from the carry, smoothed: zero-lag, robust (Huber), so slow
    # corrections and the motion fit's own drift land, and one-frame jumps (a sparkle trail splitting the matte) don't
    C = {k: P[k].copy()}
    rep = {}
    for t in ts:
        if t > k and t - 1 in C:
            A, b, ok, *_ = steps.get(t - 1, (None, None, False))
            if not ok:
                A, b = affine_fit(P[t - 1], P[t])
                rep[t] = {"motion": "fallback"}
            C[t] = C[t - 1] @ A.T + b
    for t in reversed(ts):
        if t < k and t + 1 in C:
            A, b, ok, *_ = steps.get(t, (None, None, False))
            if not ok:
                A, b = affine_fit(P[t], P[t + 1])
                rep[t] = {"motion": "fallback"}
            Ai = np.linalg.inv(A)
            C[t] = (C[t + 1] - b) @ Ai.T
    tt = np.array([t for t in ts if t in C])
    e0, e1 = edges(c["edge"])
    D = np.stack([P[t] - C[t] for t in tt])                       # frames x points x 2
    Ds = np.empty_like(D)
    for i, t in enumerate(tt):
        w0 = np.exp(-0.5 * ((tt - t) / sigma_t) ** 2)
        w0[np.abs(tt - t) > 3 * sigma_t] = 0
        edge = (e0 == "mean" and t - tt[0] < 2 * sigma_t) or (e1 == "mean" and tt[-1] - t < 2 * sigma_t)
        X = np.ones((len(tt), 1)) if edge else np.c_[np.ones(len(tt)), tt - t]
        w = w0.copy()
        for _ in range(3):                                         # Huber weights on the frames
            sw = w[:, None]
            beta = np.linalg.lstsq(X * np.sqrt(sw), (D.reshape(len(tt), -1)) * np.sqrt(sw), rcond=None)[0]
            r = np.median(np.linalg.norm((D - (X @ beta).reshape(D.shape)), axis=2), axis=1)
            sc = 1.4826 * np.median(r[w0 > 0.05]) + 0.15
            w = w0 * np.minimum(1.0, 1.5 * sc / np.maximum(r, 1e-9))
        Ds[i] = beta[0].reshape(D.shape[1:])
        rep.setdefault(int(t), {})["placement_jitter_px"] = round(float(np.median(np.linalg.norm(D[i] - Ds[i], axis=1))), 2)
        rep[int(t)]["trusted"] = bool(w[i] / max(w0[i], 1e-9) > 0.5)
    ik = int(np.nonzero(tt == k)[0][0])
    fix = D[ik] - Ds[ik]                                           # the keyframe's frame stays exact
    S = {}
    for i, t in enumerate(tt):
        S[int(t)] = C[int(t)] + Ds[i] + fix * np.exp(-0.5 * ((t - k) / (2 * sigma_t)) ** 2)
    for t in ts:
        if t in S:
            rep.setdefault(t, {})["correction_px"] = round(float(np.linalg.norm(S[t] - P[t], axis=1).max()), 2)
            if t in steps:
                rep[t]["motion_rms"], rep[t]["motion_dev_px"] = steps[t][3], steps[t][4]
    return S, G, inl, P, rep, steps


def glued_mappings(maps, P, S, k):
    """The glued mapping per frame: the placement plus the image-space correction S - P through the grid points.
    -> {frame: geometry.Residual}"""
    sig = float(np.ptp(P[k], axis=0).max() / 6.0)
    return {t: geo.Residual(maps[t], P[t], S[t] - P[t], sig, 1e-3) for t in S}


def summary(rep, steps, ts, s0, kf, sigma):
    """stabilize.json's summary (clip frames)."""
    have = [t for t in ts if "correction_px" in rep.get(t, {})]
    corr = np.array([rep[t]["correction_px"] for t in have])
    return {"keyframe": kf, "sigma_frames": sigma, "frames": len(have),
            "motion_fallbacks": [s0 + t for t in ts[:-1] if t in steps and not steps[t][2]],
            "untrusted_placements": [s0 + t for t in ts if rep.get(t, {}).get("trusted") is False],
            "placement_jitter_px_median": round(float(np.median([rep[t]["placement_jitter_px"] for t in have])), 2),
            "correction_px_median": round(float(np.median(corr)), 2), "correction_px_max": round(float(corr.max()), 2)}
