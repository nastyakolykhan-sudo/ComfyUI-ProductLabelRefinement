"""Several products in one master: depth order per frame from their mattes, then each product's fix pasted back
to front, each only inside its own matte minus the mattes in front of it.

Depth from two mattes that meet: the one in front cuts into the other's silhouette, so its matte intrudes into the
other's convex hull more than the other way round. Products that don't meet keep any order. Fallback when the
evidence is even: the matte whose bottom edge is lower is nearer (products standing on one surface)."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, Delaunay


def _hull_mask(m):
    ys, xs = np.nonzero(m)
    if len(xs) < 10:
        return m.copy()
    P = np.c_[xs, ys][:: max(1, len(xs) // 3000)]
    try:
        hull = Delaunay(P[ConvexHull(P).vertices])
    except Exception:
        return m.copy()
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    gy, gx = np.mgrid[y0:y1, x0:x1]
    out = np.zeros_like(m)
    out[y0:y1, x0:x1] = hull.find_simplex(np.c_[gx.ravel(), gy.ravel()]).reshape(gy.shape) >= 0
    return out


def in_front(a, b, touch_px=3):
    """+1 when matte a is in front of matte b, -1 when behind, 0 when they don't meet."""
    if not (ndi.binary_dilation(a, iterations=touch_px) & b).any():
        return 0
    ia = (a & _hull_mask(b)).sum()               # a cutting into b's silhouette
    ib = (b & _hull_mask(a)).sum()
    if abs(int(ia) - int(ib)) > 0.002 * min(a.sum(), b.sum()):
        return 1 if ia > ib else -1
    ya, yb = np.nonzero(a)[0].max(), np.nonzero(b)[0].max()
    return 1 if ya > yb else -1


def depth_order(mattes):
    """Products back to front for one frame. mattes: {name: bool frame-sized matte}. -> [names], evidence"""
    names = [n for n, m in mattes.items() if m.any()]
    score = {n: 0 for n in names}
    ev = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = in_front(mattes[a], mattes[b])
            if r:
                score[a] += r
                score[b] -= r
                ev.append([a, b] if r > 0 else [b, a])         # [front, behind]
    return sorted(names, key=lambda n: score[n]), ev


def keep_masks(mattes, order):
    """Per product, where its patch may land: its own matte minus every matte in front of it."""
    out, front = {}, np.zeros_like(next(iter(mattes.values())))
    for n in reversed(order):
        out[n] = mattes[n] & ~front
        front |= mattes[n]
    return out
