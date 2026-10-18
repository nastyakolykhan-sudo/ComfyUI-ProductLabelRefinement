"""Print on a surface of revolution (a tapered lip tube, a cone; a cylinder with no taper) under an affine camera.

Label px (u, v): v runs along the axis, u unrolls the surface by angle, theta = (u - u0) / Ru (Ru: label px per
radian, the keyframe's radius where the label starts). The radius tapers linearly, r(v) = R + k v (R: at v = 0). Surface point P = (r sin theta, v, r cos theta), z toward the
camera; scene px = A P + a. Spin about the axis, tilt and turn all live in A, so the per-frame fit is linear in
(A, a). Same interface as imgpost.geometry's mappings (forward, inverse), so imgpost.warp and geometry.apply_H and
map_polygon take it as they are. Used when a keyframe is unrolled into label space: print that spins into view
which the keyframe never showed has no texture, and the inverse says where."""
import numpy as np


class Tube:
    def __init__(self, u0, Ru, R, k, A, a):
        self.u0, self.Ru, self.R, self.k = float(u0), float(Ru), float(R), float(k)
        self.A = np.asarray(A, float).reshape(2, 3)
        self.a = np.asarray(a, float).reshape(2)

    def surface(self, pts):
        P = np.atleast_2d(np.asarray(pts, float))
        th = (P[:, 0] - self.u0) / self.Ru
        r = np.maximum(self.R + self.k * P[:, 1], 1e-6)
        return np.c_[r * np.sin(th), P[:, 1], r * np.cos(th)]

    def forward(self, pts):
        return self.surface(pts) @ self.A.T + self.a

    def inverse(self, x, y):
        """Scene px -> label px and a mask of rays that meet the front of the surface."""
        shape = np.shape(x)
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        A = self.A
        d = np.cross(A[0], A[1])                       # the viewing ray (A d = 0), pointing at the camera
        P0 = np.linalg.pinv(A) @ np.vstack([x - self.a[0], y - self.a[1]])     # 3 x N, one point on each ray
        px, py, pz = P0
        k, R = self.k, self.R
        qa = d[0] ** 2 + d[2] ** 2 - (k * d[1]) ** 2
        qb = 2.0 * (px * d[0] + pz * d[2] - k * d[1] * (R + k * py))
        qc = px ** 2 + pz ** 2 - (R + k * py) ** 2
        disc = qb * qb - 4.0 * qa * qc
        ok = disc >= 0.0
        sq = np.sqrt(np.clip(disc, 0.0, None))
        lam = (-qb + np.sign(qa) * sq) / (2.0 * qa)     # the hit nearer the camera
        X, Y, Z = px + lam * d[0], py + lam * d[1], pz + lam * d[2]
        ok &= (R + k * Y) > 0
        th = np.arctan2(X, Z)
        u, v = self.u0 + self.Ru * th, Y
        return u.reshape(shape), v.reshape(shape), ok.reshape(shape)

    def limb_points(self, side, v):
        """Scene points of the silhouette on `side` at label rows v (where the ray grazes the surface)."""
        v = np.atleast_1d(np.asarray(v, float))
        d = np.cross(self.A[0], self.A[1])
        sgn = 1.0 if side == "right" else -1.0
        # normal of the surface at angle th: (sin th, -k, cos th) up to scale; grazing when normal . d = 0
        out = []
        for vv in v:
            ths = np.linspace(-np.pi, np.pi, 721)
            nd = np.sin(ths) * d[0] - self.k * d[1] + np.cos(ths) * d[2]
            i = np.nonzero(np.diff(np.sign(nd)))[0]
            cand = ths[i] if len(i) else np.array([sgn * np.pi / 2])
            th = cand[np.argmax(sgn * np.sin(cand))]
            out.append(self.forward([[self.u0 + self.Ru * th, vv]])[0])
        return np.array(out)

    def to_json(self):
        return {"model": "tube", "u0": self.u0, "Ru": self.Ru, "R": self.R, "k": self.k, "A": self.A.round(8).tolist(),
                "a": self.a.round(6).tolist()}


def fit(spec):
    """geometry.fit for model "tube": spec["tube"] = {"u0", "Ru", "R", "k"}, spec["points"] with "ref" (label px), "scene",
    optional "w". Weighted linear least squares for (A, a). -> (Tube, report)"""
    tb = spec["tube"]
    P = spec["points"]
    uv = np.array([p["ref"] for p in P], float)
    xy = np.array([p["scene"] for p in P], float)
    w = np.sqrt(np.array([p.get("w", 1.0) for p in P], float))[:, None]
    T = Tube(tb["u0"], tb["Ru"], tb["R"], tb["k"], np.zeros((2, 3)), np.zeros(2))
    P3 = T.surface(uv)
    S = np.c_[P3, np.ones(len(uv))]
    M = np.linalg.lstsq(S * w, xy * w, rcond=None)[0]            # 4 x 2: the affine camera, a start
    T.A, T.a = M[:3].T.copy(), M[3].copy()
    if spec.get("camera", "ortho") == "ortho":
        # scaled orthographic: A = s x the top two rows of a rotation. Over the front of a tube Z hardly varies, so
        # an affine camera trades its depth column (spin, tilt) against its offset and the label jitters by px
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        U_, sv, Vt = np.linalg.svd(T.A, full_matrices=False)
        R2 = U_ @ Vt
        Rm = np.vstack([R2, np.cross(R2[0], R2[1])])
        if np.linalg.det(Rm) < 0:
            Rm[2] *= -1
        x0 = np.r_[sv.mean(), Rotation.from_matrix(Rm).as_rotvec(), T.a]

        def make(p):
            return p[0] * Rotation.from_rotvec(p[1:4]).as_matrix()[:2], p[4:6]

        def res(p):
            A, a = make(p)
            return ((P3 @ A.T + a - xy) * w).ravel()
        sol = least_squares(res, x0, x_scale="jac")
        T.A, T.a = make(sol.x)
    r = np.linalg.norm(T.forward(uv) - xy, axis=1)
    return T, {"model": "tube", "rms_px": round(float(np.sqrt(np.mean(r ** 2))), 3)}


def from_keyframe(mask, t_range, theta_lim=75.0, step=4.0):
    """The tube as the keyframe shows it: axis and taper from the matte's width along the label's band, spin 0 (the
    keyframe's middle faces the camera), no tilt toward the camera. -> (Tube in label px, label width, label height,
    band info). Label px: v = 0 at the band's start; u0 in the middle of a strip theta in [-90, 90] degrees."""
    from .keyframe import product_axis
    c, ax, n = product_axis(mask)
    ys, xs = np.nonzero(mask)
    Pm = np.c_[xs, ys].astype(float)
    t, q = (Pm - c) @ ax, (Pm - c) @ n
    rows = []
    for tt in np.arange(t_range[0], t_range[1] + 1e-6, step):
        sel = np.abs(t - tt) < 1.0
        if sel.sum() >= 2:
            rows.append((tt, q[sel].min(), q[sel].max()))
    rows = np.array(rows)
    mid = 0.5 * (rows[:, 1] + rows[:, 2])                       # the axis, refined: the middle of the silhouette
    half = 0.5 * (rows[:, 2] - rows[:, 1])
    v = rows[:, 0] - t_range[0]
    kq, q0 = np.polyfit(v, mid, 1)                              # axis drift across the band (a tilt left in the PCA)
    k, R0 = np.polyfit(v, half, 1)
    ax2 = ax + kq * n
    ax2 = ax2 / np.linalg.norm(ax2)
    n2 = np.array([ax2[1], -ax2[0]])
    origin = c + t_range[0] * ax + q0 * n
    W = int(np.ceil(np.pi * R0)) + 1
    Hh = int(np.ceil(t_range[1] - t_range[0])) + 1
    T = Tube(0.5 * (W - 1), R0, R0, k, np.c_[n2, ax2, np.zeros(2)], origin)
    info = {"theta_lim": theta_lim, "R0": round(float(R0), 2), "taper": round(float(k), 5),
            "fit_half_width_rms": round(float(np.sqrt(np.mean((half - (R0 + k * v)) ** 2))), 2),
            "axis": ax2.round(5).tolist(), "origin": origin.round(2).tolist()}
    return T, W, Hh, info


def unroll(key_u8, T, W, Hh, theta_lim):
    """The keyframe read onto the label grid (label px) through the tube, and where it shows the label well enough
    (within theta_lim of facing the camera). -> (texture uint8 Hh x W x 3, polygon of the seen strip)"""
    from scipy.ndimage import map_coordinates
    v, u = np.mgrid[0:Hh, 0:W].astype(float)
    P = T.forward(np.c_[u.ravel(), v.ravel()])
    tex = np.stack([map_coordinates(key_u8[..., c].astype(float), [P[:, 1], P[:, 0]], order=3, mode="nearest")
                    for c in range(3)], -1).reshape(Hh, W, 3)
    du = T.Ru * np.radians(theta_lim)
    poly = [[T.u0 - du, 0.0], [T.u0 + du, 0.0], [T.u0 + du, Hh - 1.0], [T.u0 - du, Hh - 1.0]]
    return np.clip(tex + 0.5, 0, 255).astype(np.uint8), poly


def unroll_mask(mask, T, W, Hh):
    """The keyframe's matte on the label grid: where the unrolled label is really on the product."""
    from scipy.ndimage import map_coordinates
    v, u = np.mgrid[0:Hh, 0:W].astype(float)
    P = T.forward(np.c_[u.ravel(), v.ravel()])
    return (map_coordinates(mask.astype(float), [P[:, 1], P[:, 0]], order=1, mode="constant").reshape(Hh, W) > 0.5)


def _masked_ncc(a, ma, b, mb, max_du, max_dv, c_du=0, c_dv=0):
    """Normalized cross-correlation of a (where ma) against b (where mb) over integer shifts (du, dv) within max of
    (c_du, c_dv), b shifted: a[v, u] ~ b[v + dv, u + du]. Brute force on small images. -> (du, dv, score, map)"""
    H, W = a.shape
    best = (c_du, c_dv, -np.inf)
    smap = np.full((2 * max_dv + 1, 2 * max_du + 1), np.nan)
    for j, dv in enumerate(range(c_dv - max_dv, c_dv + max_dv + 1)):
        for i, du in enumerate(range(c_du - max_du, c_du + max_du + 1)):
            ya0, ya1 = max(0, -dv), min(H, H - dv)
            xa0, xa1 = max(0, -du), min(W, W - du)
            if ya1 - ya0 < 8 or xa1 - xa0 < 8:
                continue
            A = a[ya0:ya1, xa0:xa1]
            B = b[ya0 + dv:ya1 + dv, xa0 + du:xa1 + du]
            m = ma[ya0:ya1, xa0:xa1] & mb[ya0 + dv:ya1 + dv, xa0 + du:xa1 + du]
            n = m.sum()
            if n < 0.15 * ma.sum():
                continue
            x, y = A[m] - A[m].mean(), B[m] - B[m].mean()
            d = np.sqrt((x * x).sum() * (y * y).sum())
            sc = float((x * y).sum() / d) if d > 0 else -np.inf
            smap[j, i] = sc
            if sc > best[2]:
                best = (du, dv, sc)
    return best[0], best[1], best[2], smap


def _rolled(T, deg):
    """The same tube turned in the image plane by deg about its band's middle (the matte's axis can be off when
    part of the product is hidden or the matte includes a pump or cap that isn't on the axis)."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    Rm = np.array([[c, -s], [s, c]])
    mid = T.forward([[T.u0, 0.0]])[0]
    return Tube(T.u0, T.Ru, T.R, T.k, Rm @ T.A, Rm @ (T.a - mid) + mid)


def reacquire(tex_u8, valid, tube_k, mask_k_area, frame_u8, mask_f, band_k, theta_lim=75.0, down=4, max_spin=80.0,
              max_dv=60.0, scales=(0.85, 0.88, 0.91, 0.94, 0.97, 1.0, 1.03, 1.06), rolls=(0.0,)):
    """The keyframe's label found again on a frame of another shot (no tracks cross the cut). The tube is fitted to
    the frame's matte; the frame is read onto the keyframe's label grid with no spin; there the bottle's spin is a
    shift along u and an axial offset a shift along v, found by masked correlation of band-passed print (the frame's
    print is garbled but keeps the layout). -> (Tube in the keyframe's label px for that frame, report)"""
    from scipy.ndimage import gaussian_filter, map_coordinates
    from imgpost import images as im
    s0 = float(np.sqrt(mask_f.sum() / max(mask_k_area, 1)))
    Tf, _, _, info = from_keyframe(mask_f, [band_k[0] * s0, band_k[1] * s0], theta_lim)
    best = None
    for f_sc in scales:                    # coarse: quarter resolution, the whole spin and offset range
        H, rep = _reacquire_at(tex_u8, valid, tube_k, frame_u8, mask_f, Tf, Tf.R / tube_k["R"] * f_sc, theta_lim, down,
                               max_spin, max_dv)
        if best is None or rep["verify_ncc"] > best[1]["verify_ncc"]:
            best = (H, rep, f_sc)
    H, rep, f_sc = best
    roll = 0.0
    d2 = max(1, down // 2)
    if len(rolls) > 1:                     # roll: the print's own orientation, not the matte's axis; at half resolution,
        rbest = None                       # where a degree turns a text line's ends by px (at a quarter it blurs away)
        for r in rolls:
            Hr, rr = _reacquire_at(tex_u8, valid, tube_k, frame_u8, mask_f, _rolled(Tf, r), Tf.R / tube_k["R"] * f_sc, theta_lim,
                                   d2, max_spin, max_dv, around=(rep["_du"] * down // d2, rep["_dv"] * down // d2, 3))
            if rbest is None or rr["verify_ncc"] > rbest[1]["verify_ncc"]:
                rbest = (Hr, rr, float(r))
        roll = rbest[2]
        rep = dict(rep, _du=rbest[1]["_du"] * d2 // down, _dv=rbest[1]["_dv"] * d2 // down)
    Tr = _rolled(Tf, roll) if roll else Tf
    fine = None
    for f2 in (f_sc - 0.015, f_sc, f_sc + 0.015):     # fine: half resolution, around the coarse answer
        Hf, rf = _reacquire_at(tex_u8, valid, tube_k, frame_u8, mask_f, Tr, Tf.R / tube_k["R"] * f2, theta_lim, d2,
                               max_spin, max_dv, around=(rep["_du"] * down // d2, rep["_dv"] * down // d2, 3))
        if fine is None or rf["verify_ncc"] > fine[1]["verify_ncc"]:
            fine = (Hf, rf)
    if fine[1]["verify_ncc"] >= rep["verify_ncc"] - 0.02:
        H, rep = fine
    rep = {k: v for k, v in rep.items() if not k.startswith("_")}
    rep["roll_deg"] = roll
    rep["silhouette"] = info
    return H, rep


def _reacquire_at(tex_u8, valid, tube_k, frame_u8, mask_f, Tf, sig, theta_lim, down, max_spin, max_dv, around=None):
    from scipy.ndimage import gaussian_filter, map_coordinates
    from imgpost import images as im
    n_f, ax_f = Tf.A[:, 0], Tf.A[:, 1]

    def mapping(phi, v_off):
        c, s = np.cos(phi), np.sin(phi)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])             # spin about the axis
        A0 = sig * np.c_[n_f, ax_f, np.zeros(2)]
        return Tube(tube_k["u0"], tube_k["Ru"], tube_k["R"], tube_k["k"], A0 @ Ry, Tf.a - A0[:, 1] * v_off)

    Hh, W = tex_u8.shape[:2]
    v, u = np.mgrid[0:Hh:down, 0:W:down].astype(float)
    th = (u - tube_k["u0"]) / tube_k["Ru"]
    lim = np.radians(theta_lim)

    def bandpass(L):
        return gaussian_filter(L, 1.0) - gaussian_filter(L, 4.0)

    def lum_blur(img, sd):                                    # luminance, low-passed before it is sampled sparsely
        return gaussian_filter(im.lum(im.srgb_to_lin(im.to_float(img))), sd)

    K = bandpass(lum_blur(tex_u8, down / 2)[::down, ::down])
    fL = lum_blur(frame_u8, max(down / 2 * sig, 0.5))
    mK = valid[::down, ::down] & (np.abs(th) <= lim)
    T0 = mapping(0.0, 0.0)
    P = T0.forward(np.c_[u.ravel(), v.ravel()])
    F = bandpass(map_coordinates(fL, [P[:, 1], P[:, 0]], order=1, mode="nearest").reshape(u.shape))
    mF = (map_coordinates(mask_f.astype(float), [P[:, 1], P[:, 0]], order=1).reshape(u.shape) > 0.5) & (np.abs(th) <= lim)
    from scipy.ndimage import binary_erosion
    mK, mF = binary_erosion(mK, iterations=2), binary_erosion(mF, iterations=2)
    if around is None:
        du, dv, sc, _ = _masked_ncc(F, mF, K, mK, int(np.radians(max_spin) * tube_k["Ru"] / down), int(max_dv / down))
    else:
        du, dv, sc, _ = _masked_ncc(F, mF, K, mK, around[2], around[2], around[0], around[1])
    # F[v, u] (frame, no spin) ~ K[v + dv, u + du]: the frame's front shows the keyframe's label du further round
    cands = []
    for sgn in (1.0, -1.0):
        phi = sgn * du * down / tube_k["Ru"]
        H = mapping(phi, dv * down)
        P = H.forward(np.c_[u.ravel(), v.ravel()])
        Fs = bandpass(map_coordinates(fL, [P[:, 1], P[:, 0]], order=1, mode="nearest").reshape(u.shape))
        # where this mapping shows the label: front-facing in the frame
        ub, vb, ok = H.inverse(P[:, 0], P[:, 1])
        seen = (np.abs(ub - u.ravel()) < down) & (np.abs(vb - v.ravel()) < down) & ok
        m = mK & seen.reshape(u.shape) & (map_coordinates(mask_f.astype(float), [P[:, 1], P[:, 0]], order=1).reshape(u.shape) > 0.5)
        x, y = Fs[m] - Fs[m].mean(), K[m] - K[m].mean()
        cands.append((float((x * y).sum() / max(np.sqrt((x * x).sum() * (y * y).sum()), 1e-9)), sgn, H))
    score, sgn, H = max(cands, key=lambda c: c[0])
    rep = {"scale": round(float(sig), 4), "spin_deg": round(float(np.degrees(sgn * du * down / tube_k["Ru"])), 1),
           "axial_offset_px": round(float(dv * down), 1), "shift_ncc": round(sc, 3), "verify_ncc": round(score, 3),
           "other_sign_ncc": round(min(c[0] for c in cands), 3), "_du": int(du), "_dv": int(dv)}
    return H, rep


def line_angle(L, lo=-6.0, hi=6.0, step=0.25):
    """The angle (degrees) that makes a block of text's lines level: the rotation with the most varied row sums of
    its band-passed luminance (lines of print give peaks and troughs). Positive: the lines slope down to the right."""
    from scipy.ndimage import gaussian_filter, rotate
    B = gaussian_filter(L, 0.8) - gaussian_filter(L, 6.0)
    best = (0.0, -np.inf)
    for a in np.arange(lo, hi + 1e-9, step):
        v = float(np.var(rotate(B, a, reshape=False, order=1, mode="constant").sum(1)))
        if v > best[1]:
            best = (float(a), v)
    return best[0]


def roll_from_print(H, frame_u8, mask_f, tex_u8, box, rolls=np.arange(-4.0, 4.01, 0.25)):
    """Turn a re-acquired mapping in the image plane until the frame's own print, read back onto the label grid,
    has its lines at the same slope as the keyframe's. Roll is what the eye compares against the bottle, and blurred
    or garbled print hardly changes a correlation for a degree or two. box: [u0, v0, u1, v1] label px around lines
    of text. -> (rolled mapping, report)"""
    from scipy.ndimage import map_coordinates
    from imgpost import images as im
    u0, v0, u1, v1 = box
    vv, uu = np.mgrid[v0:v1, u0:u1].astype(float)
    target = line_angle(im.lum(im.srgb_to_lin(im.to_float(tex_u8[v0:v1, u0:u1]))))
    Lf = im.lum(im.srgb_to_lin(im.to_float(frame_u8)))
    centre = H.forward([[0.5 * (u0 + u1), 0.5 * (v0 + v1)]])[0]
    best = None
    for r in rolls:
        c, s = np.cos(np.radians(r)), np.sin(np.radians(r))
        Rm = np.array([[c, -s], [s, c]])
        Hr = Tube(H.u0, H.Ru, H.R, H.k, Rm @ H.A, Rm @ (H.a - centre) + centre)
        P = Hr.forward(np.c_[uu.ravel(), vv.ravel()])
        Lb = map_coordinates(Lf, [P[:, 1], P[:, 0]], order=1, mode="nearest").reshape(uu.shape)
        seen = map_coordinates(mask_f.astype(float), [P[:, 1], P[:, 0]], order=1).reshape(uu.shape) > 0.5
        if seen.mean() < 0.7:
            continue
        err = abs(line_angle(np.where(seen, Lb, np.median(Lb[seen]))) - target)
        if best is None or err < best[1] - 1e-9 or (abs(err - best[1]) < 1e-9 and abs(r) < abs(best[0])):
            best = (float(r), err, Hr)
    if best is None:
        return H, {"roll_deg": 0.0, "note": "the box isn't seen on this frame"}
    return best[2], {"roll_deg": best[0], "line_angle_error_deg": round(best[1], 2), "keyframe_line_angle_deg": target}


def text_rows(tex_u8, polygon, frac=0.35):
    """The label rows with the most lines of print (label v [v0, v1]): the window, frac of the band long, where the
    band-passed luminance varies most across rows (lines of text give peaks and troughs)."""
    from scipy.ndimage import gaussian_filter, gaussian_filter1d
    from imgpost import images as im
    P = np.asarray(polygon, float)
    u0, u1, v0, v1 = int(P[:, 0].min()), int(P[:, 0].max()), int(P[:, 1].min()), int(P[:, 1].max())
    L = im.lum(im.srgb_to_lin(im.to_float(tex_u8)))[v0:v1, u0:u1]
    B = gaussian_filter(L, 1.0) - gaussian_filter(L, 8.0)
    e = gaussian_filter1d(np.abs(np.diff(B, axis=0)).mean(1), 3)
    n = max(int(frac * (v1 - v0)), 20)
    c = np.convolve(e, np.ones(n) / n, mode="valid")
    j = int(np.argmax(c))
    return float(v0 + j), float(v0 + j + n)


def line_bow(sample, H, matte, rows, n_strips=9, reach=0.6):
    """How far the print's lines bow toward the base mid-label, in px, measured strip by strip (V28): along the
    product's axis, in n_strips strips across the width (to reach x the half-width), each strip's band-passed
    profile against the centre strip's; offset(x) = c + b x + s sqrt(1 - x^2/R^2) gives the bow s (+ = a smile) and
    the linear term b (roll). sample(xy) -> luminance at image points. rows: label v range with lines of print.
    -> (s, b in px per px, offsets)"""
    from scipy.ndimage import map_coordinates, gaussian_filter1d
    Hb = H
    while hasattr(Hb, "base"):
        Hb = Hb.base
    A = Hb.A
    ax = A[:, 1] / np.linalg.norm(A[:, 1])
    n = np.array([ax[1], -ax[0]])
    d = np.cross(A[0], A[1])
    uc = Hb.u0 + Hb.Ru * np.arctan2(d[0], d[2])
    p0, p1 = H.forward([[uc, rows[0]], [uc, rows[1]]])
    c = 0.5 * (p0 + p1)
    s_ = np.arange(-600, 601, 0.5)
    q = c[None] + s_[:, None] * n[None]
    m = map_coordinates(matte.astype(float), [q[:, 1], q[:, 0]], order=1) > 0.5
    ii = np.nonzero(m)[0]
    if not len(ii):
        return None, None, None
    R = 0.5 * (s_[ii.max()] - s_[ii.min()])
    cen = 0.5 * (s_[ii.max()] + s_[ii.min()])
    L = float(np.linalg.norm(p1 - p0))
    w = np.arange(-40, L + 40, 0.5)
    xs = np.linspace(-reach, reach, n_strips) * R
    prof = []
    for x in xs:
        acc = 0.0
        for dx in np.linspace(-0.05, 0.05, 7) * R:
            pts = p0[None] + (cen + x + dx) * n[None] + w[:, None] * ax[None]
            acc = acc + sample(pts)
        pr = acc - gaussian_filter1d(acc, 12)
        prof.append(gaussian_filter1d(pr, 1.0))
    ref = prof[len(xs) // 2]
    offs = []
    for pr in prof:
        lags = np.arange(-40, 41)
        cc = [float(np.dot(np.roll(pr, -k)[80:-80], ref[80:-80])) for k in lags]
        j = int(np.argmax(cc))
        sub = 0.0
        if 0 < j < len(lags) - 1:
            y0, y1, y2 = cc[j - 1], cc[j], cc[j + 1]
            den = y0 - 2 * y1 + y2
            sub = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        offs.append((lags[j] + sub) * 0.5)
    offs = np.array(offs)
    X = np.c_[np.ones(len(xs)), xs, np.sqrt(1 - (xs / R) ** 2)]
    coef, *_ = np.linalg.lstsq(X, offs, rcond=None)
    return float(coef[2]), float(coef[1]), offs


def texture_sampler(tex_u8, H):
    """sample(xy) for line_bow: the texture's luminance where each image point meets the label through H (a render
    of the label's print alone, without light): rays off the label read the texture's mean."""
    from scipy.ndimage import map_coordinates
    from imgpost import images as im
    L = im.lum(im.srgb_to_lin(im.to_float(tex_u8)))
    mean = float(L.mean())

    def f(pts):
        u, v, ok = H.inverse(pts[:, 0], pts[:, 1])
        out = map_coordinates(L, [v, u], order=1, mode="constant", cval=mean)
        if ok is not None:
            out = np.where(ok, out, mean)
        return out
    return f


def image_sampler(frame_u8):
    from scipy.ndimage import map_coordinates
    from imgpost import images as im
    L = im.lum(im.srgb_to_lin(im.to_float(frame_u8)))
    return lambda pts: map_coordinates(L, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
