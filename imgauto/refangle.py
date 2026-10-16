"""How closely a frame shows the product at the reference photo's viewing angle.

Where it does, the stills engine (run.py: the real photo warped onto the frame) can make the keyframe with exact
text, and a generative edit is only the fallback. Two measures, after a similarity that matches the two silhouettes'
centroid, axis and length:
- shape: the silhouettes' overlap. Tilt toward the camera, the cap or pump's proportions and the taper all change it;
- layout: band-passed correlation of the print. Generated print is garbled but keeps its layout, so this says
  which face is turned to the camera (spin) and checks roll.
A reference photo may hold several objects (a tube next to its box): each is tried and the best one kept.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def objects(path, min_share=0.02, tol=25, colours=None, scale=1, neck=14):
    """The objects on a packshot (transparent or plain background): -> [(mask, gray)] per object, largest first.
    colours: the clip's product colour model (shots.product_colours); objects are then cut to the pixels in the
    product's colours first, which parts a tube from the box it touches. scale: keep every scale-th pixel. Objects
    joined by a neck under 2 x neck px are split (each pixel goes to the nearest eroded core)."""
    im = Image.open(path)
    rgba = np.asarray(im.convert("RGBA")).astype(np.float32)
    rgb, a = rgba[..., :3], rgba[..., 3]
    if (a < 250).mean() > 0.05:
        fg = a > 127
    else:
        border = np.r_[rgb[:3].reshape(-1, 3), rgb[-3:].reshape(-1, 3), rgb[:, :3].reshape(-1, 3), rgb[:, -3:].reshape(-1, 3)]
        bg = np.median(border, 0)
        fg = np.abs(rgb - bg).max(-1) > tol
    rgb, fg = rgb[::scale, ::scale], fg[::scale, ::scale]
    if colours is not None:
        p, bins = colours
        q = (rgb // (256 // bins)).astype(np.int32)
        like = p[(q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]] > 0.5
        like = ndi.binary_closing(like, iterations=6)
        if (fg & like).sum() > 0.2 * fg.sum() * 0.25:
            fg = fg & like
    fg = ndi.binary_fill_holes(ndi.binary_opening(fg, iterations=2))
    lab, n = ndi.label(ndi.binary_erosion(fg, iterations=neck))  # objects touching at a thin neck come apart
    if n:
        _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab = np.where(fg, lab[iy, ix], 0)
    sizes = np.bincount(lab.ravel())[1:]
    gray = rgb.mean(-1)
    out = [(lab == i + 1, gray) for i in np.argsort(-sizes) if sizes[i] >= min_share * fg.size]
    return out


def shape_iou(ref_mask, target):
    """Silhouette overlap of a reference object with a clip matte after the centroid / axis / length similarity."""
    cf, tf, lf = _pose(target)
    cr, tr, lr = _pose(ref_mask)
    s = lr / max(lf, 1e-6)
    r = tr - tf
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]]) * s
    mr = ndi.affine_transform(ref_mask.astype(np.float32), R, cr - R @ cf, output_shape=target.shape, order=1) > 0.5
    return float((target & mr).sum() / max((target | mr).sum(), 1))


def product_object(path, target, colours=None, scale=1, necks=(6, 14, 24, 36)):
    """The object on a reference photo that is the clip's product: of the objects found with each neck size (a tube
    beside its box, or joined to the tube printed on the box, comes apart at some neck), the one whose silhouette
    best matches `target` (a clip matte at any scale). -> (mask, gray, iou) or None"""
    best = None
    for nk in necks:
        for m, g in objects(path, colours=colours, scale=scale, neck=nk):
            iou = shape_iou(m, target)
            if best is None or iou > best[2]:
                best = (m, g, iou)
    return best


def _pose(m):
    ys, xs = np.nonzero(m)
    P = np.c_[ys, xs].astype(float)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    ax = vt[0] * (np.sign(vt[0][0]) or 1.0)
    return c, float(np.arctan2(ax[1], ax[0])), float(np.ptp((P - c) @ ax))


def _band(g, s, lo=2.0, hi=10.0):
    """Text blocks and lines, not strokes: garbled print keeps the layout, not the letters."""
    return ndi.gaussian_filter(g, lo * s) - ndi.gaussian_filter(g, hi * s)


def match(frame_gray, frame_mask, ref_mask, ref_gray, rots=(-6, -3, 0, 3, 6), shifts=range(-8, 9, 4)):
    """-> (shape overlap 0..1, print layout correlation -1..1) of a reference object against a frame's product.
    The silhouettes' centroid, axis and length give the similarity; a small search over rotation (deg) and shift
    (px) absorbs the axis error a pump or cap puts into it. Work happens in the product's box only."""
    if frame_mask.sum() < 200:
        return 0.0, 0.0
    ys, xs = np.nonzero(frame_mask)
    pad = 24
    y0, x0 = max(ys.min() - pad, 0), max(xs.min() - pad, 0)
    y1, x1 = ys.max() + pad + 1, xs.max() + pad + 1
    fm, fg = frame_mask[y0:y1, x0:x1], frame_gray[y0:y1, x0:x1]
    cf, tf, lf = _pose(fm)
    cr, tr, lr = _pose(ref_mask)
    s = lr / max(lf, 1e-6)
    gf = _band(fg, 1.0)
    rb = _band(ref_gray, max(s, 1.0))
    core = ndi.binary_erosion(fm, iterations=12)                 # silhouette edges out: they'd dominate
    best_iou, best_lay = 0.0, -1.0
    for dr in rots:
        r = tr - tf + np.radians(dr)
        R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]]) * s     # frame px (y, x) -> reference px
        off = cr - R @ cf
        mr = ndi.affine_transform(ref_mask.astype(np.float32), R, off, output_shape=fm.shape, order=1) > 0.5
        gr = ndi.affine_transform(rb, R, off, output_shape=fm.shape, order=1)
        if dr == 0:
            best_iou = float((fm & mr).sum() / max((fm | mr).sum(), 1))
        for dy in shifts:
            for dx in shifts:
                g2 = np.roll(gr, (dy, dx), (0, 1))
                m = core & np.roll(ndi.binary_erosion(mr, iterations=12), (dy, dx), (0, 1))
                if m.sum() < 100:
                    continue
                x, y = gf[m] - gf[m].mean(), g2[m] - g2[m].mean()
                best_lay = max(best_lay, float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-6)))
    return best_iou, max(best_lay, 0.0)


def angle_scores(F, M, frames, refs):
    """Per frame: the best reference object's {"shape", "layout", "angle", "ref"}; angle = shape x max(layout, 0)
    rescaled so a full silhouette overlap with clearly matching print reads about 1."""
    out = {}
    for t in frames:
        g = F[t].astype(np.float32).mean(-1)
        best = {"shape": 0.0, "layout": 0.0, "angle": 0.0, "ref": None}
        for name, objs in refs.items():
            for j, (rm, rg) in enumerate(objs):
                iou, lay = match(g, M[t], rm, rg)
                ang = float(np.clip(iou, 0, 1) ** 2 * np.clip(lay / 0.5, 0, 1))
                if ang > best["angle"] or best["ref"] is None:
                    best = {"shape": round(iou, 3), "layout": round(lay, 3), "angle": round(ang, 3), "ref": f"{name}#{j}"}
        out[t] = best
    return out
