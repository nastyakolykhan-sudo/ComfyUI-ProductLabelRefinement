"""The source's optics, measured so the new print matches them: softness from the product's silhouette, and per
frame how much softer the label is than in a typical frame of the shot. Generated video smears and morphs some
transition frames (a fall, a bounce) without physical motion blur along the tracks, so it is read frame by frame."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import least_squares
from scipy.special import erf

from imgpost import images as im


def _profile_sigma(L, p, n, half=6.0, step=0.25):
    s = np.arange(-half, half + step, step)
    xs, ys = p[0] + s * n[0], p[1] + s * n[1]
    z = map_coordinates(L, [ys, xs], order=1, mode="nearest")
    f = lambda q: q[0] + q[1] * 0.5 * (1 + erf((s - q[2]) / (np.sqrt(2) * max(abs(q[3]), 0.05)))) - z
    sol = least_squares(f, [z[0], z[-1] - z[0], 0.0, 1.0])
    ok = abs(sol.x[1]) > 0.04 and abs(sol.x[2]) < half - 2
    return abs(sol.x[3]) if ok else np.nan


def limb_sigma(frame_u8, H, rows, n=16):
    """Gaussian-equivalent blur (px) across a cylinder's silhouette (both limbs), median of n profiles each side."""
    if not hasattr(H, "limb_points"):
        return float("nan")
    L = im.lum(im.srgb_to_lin(im.to_float(frame_u8)))
    out = []
    for side in ("left", "right"):
        v = np.linspace(rows[0], rows[1], n + 2)
        P = H.limb_points(side, v)
        for i in range(1, len(P) - 1):
            t = P[i + 1] - P[i - 1]
            nrm = np.array([t[1], -t[0]]) / max(np.linalg.norm(t), 1e-9)
            out.append(_profile_sigma(L, P[i], nrm))
    return float(np.nanmedian(out))


def hp_ncc(a, b, m, sigma=2.5):
    """Correlation of the high-passed luminance of a and b over mask m."""
    A, B = im.lum(a), im.lum(b)
    A, B = A - gaussian_filter(A, sigma), B - gaussian_filter(B, sigma)
    A, B = A[m] - A[m].mean(), B[m] - B[m].mean()
    d = np.sqrt((A * A).sum() * (B * B).sum())
    return float((A * B).sum() / d) if d > 0 else float("nan")



def hp_energy(L, m, sigma=1.2):
    """Detail energy: std of the high-passed luminance over m."""
    h = L - gaussian_filter(L, sigma)
    return float(h[m].std())


def softness_curve(L, m, sigmas=np.arange(0.0, 4.01, 0.25)):
    """How a frame's detail energy falls as it is blurred: (sigmas, energy / energy unblurred)."""
    e0 = hp_energy(L, m)
    return sigmas, np.array([hp_energy(gaussian_filter(L, s), m) / e0 if s > 0 else 1.0 for s in sigmas])


def extra_blur(ratio, curve, floor=0.85):
    """The Gaussian (px) that brings detail energy down to `ratio` of a typical frame's: 0 above `floor` (a pose
    further away or turned reads a little softer without being blurred)."""
    sig, f = curve
    if ratio >= floor:
        return 0.0
    return float(np.interp(ratio, f[::-1], sig[::-1]))


def line_blur(img, d, length):
    """img averaged over shifts along the unit direction d spanning `length` px, centred (a box motion blur)."""
    from scipy.ndimage import shift
    if length < 0.5:
        return img
    n = int(np.ceil(length)) + 1
    acc = 0.0
    for o in np.linspace(-0.5 * length, 0.5 * length, n):
        acc = acc + shift(img, (o * d[1], o * d[0]) + ((0.0,) * (img.ndim - 2)), order=1, mode="nearest")
    return acc / n


def _dir_energy(L, m, d):
    gy, gx = np.gradient(gaussian_filter(L, 0.7))
    par = (gx * d[0] + gy * d[1])[m]
    per = (-gx * d[1] + gy * d[0])[m]
    return float(np.mean(par ** 2)), float(np.mean(per ** 2))


def smear_length(L_t, m_t, L_a, m_a, d, speed, d_a=None, lengths=(0, 1, 2, 3, 4, 6, 8, 11, 15, 20, 26, 34)):
    """Motion smear on frame t along the label's direction of travel d, measured on the source: how much more detail
    it has lost along d than across d, compared with the anchor frame (same print, no motion) along d_a, the same
    direction relative to the label there (the product may have turned in the frame between them), matched by
    smearing the anchor along d_a. Capped at the frame's own motion. -> length px (0: the frame isn't smeared)"""
    d_a = d if d_a is None else d_a
    pt, qt = _dir_energy(L_t, m_t, d)
    ratio_t = pt / max(qt, 1e-12)
    pa, qa = _dir_energy(L_a, m_a, d_a)
    if ratio_t >= 0.85 * pa / max(qa, 1e-12):
        return 0.0
    best, err = 0.0, np.inf
    for Ln in lengths:
        if Ln > speed + 1:
            break
        p, q = _dir_energy(line_blur(L_a, d_a, Ln), m_a, d_a)
        e = abs(np.log(max(p / max(q, 1e-12), 1e-12)) - np.log(max(ratio_t, 1e-12)))
        if e < err:
            best, err = float(Ln), e
    return best
