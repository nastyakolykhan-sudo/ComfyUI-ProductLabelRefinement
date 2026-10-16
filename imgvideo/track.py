"""Dense point tracks on the product through a shot, plus a colour key for its body.

Trackers (track.backend): "alltracker", the default (AllTracker, MIT code and weights: imgvideo/alltracker, weights
from alltracker_weights()), or "cotracker" (CoTracker3 offline through torch.hub: CC BY-NC 4.0, non-commercial use
only; never for client work; kept to reproduce jobs tracked before 0.3.0).

Tracks are queried on one anchor frame and run both ways from it, so every track shares the anchor frame's
mapping to the artwork. Visibility is kept soft (CoTracker's visibility x confidence), not thresholded."""
import numpy as np
from scipy import ndimage as ndi


def hsv(rgb):
    """-> hue (degrees), saturation, value (0..1) of uint8 RGB."""
    c = rgb.astype(np.float32) / 255.0
    mx, mn = c.max(-1), c.min(-1)
    d = np.maximum(mx - mn, 1e-6)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, (mx - mn) / np.maximum(mx, 1e-6), mx


def product_key(rgb, hue=(-14, 28), sat=0.27, val=0.35, near=None, open_px=2, region=None, min_px=40):
    """Bool mask of the product's body colour: hue between hue[0] and hue[1] degrees (wrapping through 0), with
    saturation and value floors; the largest connected component, or the one nearest `near` (x, y). With `region`
    (bool, where the label is), every component with min_px or more inside it: an occluder crossing the product
    (a sparkle trail, a hand) splits it into pieces that all belong to it."""
    h, s, v = hsv(rgb)
    lo, hi = hue
    dh = (h - lo) % 360
    m = (dh <= (hi - lo) % 360) & (s >= sat) & (v >= val)
    m = ndi.binary_opening(m, iterations=open_px)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    if region is not None:
        cnt = ndi.sum(region, lab, range(1, n + 1))
        keep = np.r_[False, cnt >= min_px]
        if keep.any():
            return ndi.binary_fill_holes(keep[lab])
    if near is None:
        k = int(np.argmax(ndi.sum(m, lab, range(1, n + 1)))) + 1
    else:
        ys, xs = np.nonzero(m)
        j = np.argmin((xs - near[0]) ** 2 + (ys - near[1]) ** 2)
        k = lab[ys[j], xs[j]]
    return ndi.binary_fill_holes(lab == k)


def grid_queries(mask, step=5, erode=4):
    """(N, 2) x, y on a regular grid inside the eroded mask."""
    m = ndi.binary_erosion(mask, iterations=erode) if erode else mask
    ys, xs = np.mgrid[0:mask.shape[0]:step, 0:mask.shape[1]:step]
    ok = m[ys, xs]
    return np.c_[xs[ok], ys[ok]].astype(float)


def load_cotracker(device=None):
    import torch
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline", trust_repo=True, verbose=False)
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), device


def cotracker(frames, xy, anchor, crop, res=None, model=None, device=None):
    """Track points xy (full-frame px on frame `anchor`) through frames (T x H x W x 3 uint8), inside the crop box
    [x0, y0, x1, y1]. res: (h, w) the tracker works at (default the model's 384 x 512).
    -> tracks T x N x 2 (full-frame px), visibility T x N (0..1, visibility x confidence), and the tracker's own
    estimate of the query points on the anchor frame (N x 2): its offset from the queries is a per-point bias that
    every other frame carries too."""
    import torch
    import torch.nn.functional as F
    if model is None:
        model, device = load_cotracker(device)
    device = device or next(model.parameters()).device
    x0, y0, x1, y1 = crop
    H, W = y1 - y0, x1 - x0
    rh, rw = res or tuple(model.interp_shape)
    vid = torch.from_numpy(np.ascontiguousarray(frames[:, y0:y1, x0:x1])).permute(0, 3, 1, 2).float()
    vid = F.interpolate(vid, (rh, rw), mode="bilinear", align_corners=True)[None].to(device)
    sx, sy = (rw - 1) / (W - 1), (rh - 1) / (H - 1)
    q = np.c_[np.full(len(xy), anchor, float), (xy[:, 0] - x0) * sx, (xy[:, 1] - y0) * sy]
    T = vid.shape[1]

    def run(v, qq):
        with torch.no_grad():
            out = model.model(video=v, queries=torch.from_numpy(qq).float()[None].to(device), iters=6)
        tr, vis, conf = out[0][0], out[1][0], out[2][0] if out[2] is not None else None
        w = vis * conf if conf is not None else vis
        return tr.float().cpu().numpy(), w.float().cpu().numpy()

    tr, w = run(vid, q)
    if anchor > 0:                       # frames before the anchor: the same queries on the reversed clip
        qb = q.copy()
        qb[:, 0] = T - 1 - anchor
        trb, wb = run(vid.flip(1), qb)
        tr[:anchor], w[:anchor] = trb[::-1][:anchor], wb[::-1][:anchor]
    est = tr[anchor] / np.array([sx, sy]) + np.array([x0, y0])     # the tracker's own reading of the query frame
    tr[anchor], w[anchor] = q[:, 1:], 1.0
    tr = tr / np.array([sx, sy]) + np.array([x0, y0])
    return tr, np.clip(w, 0.0, 1.0), est


TRACKER_LICENCES = {"alltracker": "MIT (code and weights)", "cotracker": "CC BY-NC 4.0 (non-commercial use only)"}


def alltracker_weights():
    """AllTracker's checkpoint: $PLR_ALLTRACKER, else ~/Documents/models/trackers/alltracker/alltracker.pth (on a
    server: the volume's models folder). Never downloaded here."""
    import os
    from pathlib import Path
    p = Path(os.environ.get("PLR_ALLTRACKER", "~/Documents/models/trackers/alltracker/alltracker.pth")).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"AllTracker weights not found at {p}: get alltracker.pth (MIT) from "
                                "huggingface.co/aharley/alltracker and point PLR_ALLTRACKER at it")
    return p


def load_alltracker(device=None, weights=None):
    import torch
    from .alltracker import Net
    model = Net(16, init_weights=False)
    sd = torch.load(weights or alltracker_weights(), map_location="cpu", weights_only=True)
    model.load_state_dict(sd["model"] if "model" in sd else sd, strict=True)
    if device is None:                   # not MPS: some of its pooling isn't implemented there
        device = "cuda" if torch.cuda.is_available() else "cpu"
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval(), device


def alltracker(frames, xy, anchor, crop, res=None, model=None, device=None):
    """cotracker()'s contract with AllTracker: dense flow from the anchor frame to every frame of the shot (forward,
    and on the reversed clip for frames before the anchor), sampled at the query points. res: (h, w) the tracker
    works at, rounded down to multiples of 8 (default the crop's own size, at most 1024 on its longer side).
    -> tracks T x N x 2 (full-frame px), visibility T x N (0..1, visible x confident), the tracker's own reading of
    the queries on the anchor frame (N x 2; the flow there is zero, so it is the queries)"""
    import torch
    import torch.nn.functional as F
    if model is None:
        model, device = load_alltracker(device)
    device = device or next(model.parameters()).device
    x0, y0, x1, y1 = crop
    H, W = y1 - y0, x1 - x0
    if res:
        rh, rw = res
    else:
        s = min(1.0, 1024.0 / max(H, W))
        rh, rw = H * s, W * s
    rh, rw = max(8, int(rh) // 8 * 8), max(8, int(rw) // 8 * 8)
    vid = torch.from_numpy(np.ascontiguousarray(frames[:, y0:y1, x0:x1])).permute(0, 3, 1, 2).float()
    vid = F.interpolate(vid, (rh, rw), mode="bilinear", align_corners=False, antialias=True)[None].to(device)
    sx, sy = rw / W, rh / H
    q = np.c_[(xy[:, 0] - x0 + 0.5) * sx - 0.5, (xy[:, 1] - y0 + 0.5) * sy - 0.5]
    gy, gx = torch.meshgrid(torch.arange(rh, dtype=torch.float32), torch.arange(rw, dtype=torch.float32), indexing="ij")
    grid = torch.stack([gx, gy])[None]
    with torch.no_grad():
        fl, vc, _, _ = model.forward_sliding(vid[:, anchor:], iters=4, sw=None, is_training=False)
        traj, visc = fl[0].float().cpu() + grid, vc[0].float().cpu()
        if anchor > 0:
            bfl, bvc, _, _ = model.forward_sliding(vid[:, :anchor + 1].flip([1]), iters=4, sw=None, is_training=False)
            traj = torch.cat([(bfl[0].float().cpu() + grid).flip([0])[:-1], traj])
            visc = torch.cat([bvc[0].float().cpu().flip([0])[:-1], visc])

    def sample(maps):
        g = torch.from_numpy(np.c_[2 * q[:, 0] / (rw - 1) - 1, 2 * q[:, 1] / (rh - 1) - 1]).float()
        g = g[None, :, None, :].expand(maps.shape[0], -1, -1, -1)
        return F.grid_sample(maps, g, mode="bilinear", align_corners=True)[..., 0].permute(0, 2, 1).numpy()
    tr, vcs = sample(traj), sample(visc)
    vis = np.clip(vcs[..., 0], 0.0, 1.0) * np.clip(vcs[..., 1], 0.0, 1.0)
    tr = np.stack([(tr[..., 0] + 0.5) / sx - 0.5 + x0, (tr[..., 1] + 0.5) / sy - 0.5 + y0], -1)
    est = tr[anchor].copy()
    tr[anchor], vis[anchor] = xy, 1.0
    return tr, vis, est


def track_points(frames, xy, anchor, crop, res=None, backend="alltracker"):
    """The shot's tracks with track.backend's tracker (cotracker()'s contract)."""
    if backend == "alltracker":
        return alltracker(frames, xy, anchor, crop, res=res)
    if backend == "cotracker":
        return cotracker(frames, xy, anchor, crop, res=res)
    raise ValueError(f"track.backend {backend!r}: one of {sorted(TRACKER_LICENCES)}")
