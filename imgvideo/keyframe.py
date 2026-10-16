"""A corrected keyframe as the label's truth: one frame of the shot, fixed by hand or by a model, in scene px.

The keyframe's own pixels are the label space (the mapping on the keyframe is the identity). Its print is read
against its own body (a smooth body field with the print taken out), in either polarity and any number of inks:
each print pixel's colour comes from the ink cores near it, its coverage from where it sits between body and ink.
The keyframe's light is divided out of the ink (ink over the keyframe's body luminance), so each frame relights it
with its own body light and the keyframe's shading never travels with the label."""
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

from imgpost import images as im
from imgpost import outline as ol

from .render import normconv, shift_ref


def find_keyframe(frames, masks, ref_u8, grow=10):
    """The source frame the corrected keyframe was made from: least mean difference outside the (grown) mask.
    -> (index, mean |diff| there, the runner-up's)"""
    ref = ref_u8.astype(np.int16)
    d = []
    for f, m in zip(frames, masks):
        out = ~binary_dilation(m, iterations=grow)
        d.append(float(np.abs(f.astype(np.int16) - ref)[out].mean()) if out.any() else np.inf)
    d = np.array(d)
    o = np.argsort(d)
    return int(o[0]), round(float(d[o[0]]), 3), round(float(d[o[1]]), 3)


def product_axis(mask):
    """Centre and unit axis (pointing down the image) of a matte, and its normal."""
    ys, xs = np.nonzero(mask)
    P = np.c_[xs, ys].astype(float)
    c = P.mean(0)
    ax = np.linalg.svd(P - c, full_matrices=False)[2][0]
    ax = ax * (np.sign(ax[1]) or 1.0)
    return c, ax, np.array([ax[1], -ax[0]])


def label_band(mask, t_range, step=6.0, inset=1.0, centre=None, axis=None):
    """The label as the matte's extent across its axis between t_range (px along the axis from its centre):
    -> (polygon [[x, y], ...] inset px inside the matte's sides, wide quad over the same band, axis info)"""
    c, ax, n = product_axis(mask)
    if centre is not None:
        c = np.asarray(centre, float)
    if axis is not None:
        ax = np.asarray(axis, float) / np.linalg.norm(axis)
        n = np.array([ax[1], -ax[0]])
    ys, xs = np.nonzero(mask)
    P = np.c_[xs, ys].astype(float)
    t, q = (P - c) @ ax, (P - c) @ n
    left, right = [], []
    for tt in np.arange(t_range[0], t_range[1] + 1e-6, step):
        sel = np.abs(t - tt) < 1.0
        if sel.sum() < 2:
            continue
        lo, hi = q[sel].min() + inset, q[sel].max() - inset
        left.append(c + tt * ax + lo * n)
        right.append(c + tt * ax + hi * n)
    poly = np.vstack([left, right[::-1]])
    w = float(np.ptp(q))
    t0, t1 = t_range
    quad = np.array([c + t0 * ax + (q.min() - w) * n, c + t0 * ax + (q.max() + w) * n,
                     c + t1 * ax + (q.max() + w) * n, c + t1 * ax + (q.min() - w) * n])
    return poly.round(2).tolist(), quad.round(2).tolist(), {"centre": c.round(2).tolist(), "axis": ax.round(5).tolist()}


class KeyframeArtwork:
    """The keyframe's label prepared once, cropped to its outline: ink coverage (0..1), the keyframe's body luminance
    (its light, print taken out), the ink's colour per pixel, and the band quad for unseen-area counts. Same interface
    as render.Artwork (pack, ink_rgb / ink_field, refmask, polygon, off, local)."""

    def __init__(self, key_u8, polygon, quad=None, body_sigma=12.0, thresh=0.12, core=0.25, edge_band=3.0, pad=24,
                 valid=None, stroke=21):
        """valid: optional bool image (label px, the size of key_u8) where the label is really on the product (an
        unrolled keyframe's matte: a tube the surface model fits a few px wide shows background past its limb)."""
        full = ol.RefMask({"type": "polygon", "pts": polygon}, key_u8)
        rx0, ry0, rx1, ry1 = full.bbox(key_u8.shape, pad=pad)
        self.off = (rx0, ry0)
        crop = key_u8[ry0:ry1, rx0:rx1]
        self.refmask = ol.RefMask({"type": "polygon", "pts": [[x - rx0, y - ry0] for x, y in polygon]}, crop)
        self.polygon = np.asarray(self.refmask.polygon, float)
        self.quad = None if quad is None else np.asarray(quad, float) - (rx0, ry0)
        lin = im.srgb_to_lin(im.to_float(crop))
        ry, rx = np.mgrid[0:lin.shape[0], 0:lin.shape[1]].astype(float)
        inside = self.refmask.sample(rx, ry) > 0.5
        if valid is not None:
            inside &= valid[ry0:ry1, rx0:rx1]
        deep = distance_transform_edt(inside) > edge_band            # the silhouette's own rim isn't print
        lg = np.log(lin + 1e-3)
        # print is thin: strokes narrower than `stroke` px, lighter or darker than the body around them (a broad
        # specular streak or the darkening toward a limb is light, not print)
        L = im.lum(lin)
        op_, cl_ = ndi.grey_opening(L, size=(stroke, stroke)), ndi.grey_closing(L, size=(stroke, stroke))
        thin = deep & (((L - op_) / np.maximum(op_, 1e-3) > thresh) | ((cl_ - L) / np.maximum(cl_, 1e-3) > thresh))
        body = inside & ~binary_dilation(thin, iterations=2)
        for _ in range(2):
            B = np.stack([normconv(lin[..., c], body, body_sigma)[0] for c in range(3)], -1)
            d = np.abs(lg - np.log(B + 1e-3)).max(-1)
            prt = binary_dilation(thin, iterations=1) & deep & (d > thresh)
            body = inside & ~binary_dilation(prt, iterations=2)
        B = np.stack([normconv(lin[..., c], body, body_sigma)[0] for c in range(3)], -1)
        d = np.abs(lg - np.log(B + 1e-3)).max(-1)
        prt = binary_dilation(thin, iterations=1) & deep & (d > thresh)
        cores = prt & (d > np.maximum(core, np.percentile(d[prt], 50) if prt.any() else core))
        # each print pixel's ink: the colour of the ink cores near it (several inks keep their own colour)
        I = np.stack([normconv(lin[..., c], cores, 1.5)[0] for c in range(3)], -1)
        I2 = np.stack([normconv(lin[..., c], cores, 6.0)[0] for c in range(3)], -1)
        w1 = normconv(np.ones(lin.shape[:2]), cores, 1.5)[1]
        I = np.where((w1 > 0.05)[..., None], I, I2)
        dv = I - B
        cov = ((lin - B) * dv).sum(-1) / np.maximum((dv * dv).sum(-1), 1e-6)
        zone = binary_dilation(prt, iterations=2) & deep
        self.cov = np.clip(cov, 0, 1) * zone
        self.Lb = np.maximum(im.lum(B), 1e-4)
        self.ink_rgb = None
        self.ink_field = np.where(zone[..., None], I, B)
        self.pack = np.stack([self.cov, self.Lb, inside.astype(float)], -1)
        # texture mode (render): the whole label, high frequencies from here, light and colour from each frame
        self.texture = lin
        self.texture_mode = True
        self.stats = {"print_px": int(prt.sum()), "core_px": int(cores.sum()), "crop": [int(rx0), int(ry0), int(rx1), int(ry1)],
                      "dark_ink_px": int((cores & (im.lum(I) < im.lum(B))).sum()),
                      "light_ink_px": int((cores & (im.lum(I) >= im.lum(B))).sum())}

    def local(self, H):
        return shift_ref(H, *self.off)
