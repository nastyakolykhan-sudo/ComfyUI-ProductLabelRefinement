"""Label space: a frame read back onto the artwork's own grid through the frame's mapping (the unwarp).

Used twice: the source's still label (the median over the shot, so light that moves across the label stands out),
and QA (each output frame scored against the artwork where the label is seen)."""
import numpy as np
from scipy.ndimage import map_coordinates

from imgpost import geometry as geo
from imgpost import images as im


class Grid:
    """Label points every `step` reference px over the artwork crop (local reference px)."""

    def __init__(self, art, step):
        h, w = art.pack.shape[:2]
        self.step = float(step)
        self.v, self.u = np.mgrid[0:h:step, 0:w:step].astype(float)
        self.shape = self.u.shape
        self.inside = map_coordinates(art.pack[..., 2], [self.v, self.u], order=1) > 0.5

    def unwarp(self, frame_u8, H_local, key=None):
        """-> (linear RGB on the grid, visible: front-facing, on the label, inside the product's matte)"""
        P = geo.apply_H(H_local, np.c_[self.u.ravel(), self.v.ravel()])
        x, y = P[:, 0], P[:, 1]
        vis = self.inside.ravel().copy()
        if hasattr(H_local, "inverse"):                 # the far side maps back elsewhere: not seen
            ub, vb, ok = H_local.inverse(x, y)
            vis &= (ok if ok is not None else True) & (np.abs(ub - self.u.ravel()) < 0.75 * self.step) & \
                (np.abs(vb - self.v.ravel()) < 0.75 * self.step)
        h, w = frame_u8.shape[:2]
        vis &= (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
        if key is not None:
            xi, yi = np.clip(np.round(x).astype(int), 0, w - 1), np.clip(np.round(y).astype(int), 0, h - 1)
            vis &= key[yi, xi]
        lin = im.srgb_to_lin(im.to_float(frame_u8))
        rgb = np.stack([map_coordinates(lin[..., c], [y, x], order=1, mode="nearest") for c in range(3)], -1)
        return rgb.reshape(*self.shape, 3), vis.reshape(self.shape)


def still_label(stack, vis, min_frames=5):
    """Per label point, the median over the frames that see it -> (median linear RGB, how many frames saw it)."""
    S = np.where(vis[..., None], stack, np.nan)
    n = vis.sum(0)
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(S, axis=0)
    med[n < min_frames] = np.nan
    return med, n


def sampler(grid, med):
    """A function (U, V) local reference px -> the still label's linear RGB there (the frame's own pixel where the
    still label has no data: None when it has none at all)."""
    ok = np.isfinite(med[..., 0])
    if not ok.any():
        return lambda U, V: None
    filled = med.copy()
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(~ok, return_distances=False, return_indices=True)
    filled = filled[tuple(idx)]

    def f(U, V):
        gi, gj = V / grid.step, U / grid.step
        return np.stack([map_coordinates(filled[..., c], [gi, gj], order=1, mode="nearest") for c in range(3)], -1)
    return f
