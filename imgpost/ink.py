"""Light print straight on a coloured body: `matte: {"type": "ink", "polarity": "light", "print_box": [...]}`.

The dark-ink matte (run.py) measures coverage down from the paper's white, so it can't lift white print off a coral
or pink bottle (pitfall 92). Here the body stays the frame's and only print changes:
  1. cover   the packshot's print as coverage above its own body light (a grey opening wider than its strokes takes
             the print off the body), inside the reference outline (the band) and the print box only: a frosted
             body's own highlights (the band under a shoulder, limb rims) are bright thin structures too (pitfall 93).
             Warped into the work box through the job's mapping, supersampled.
  2. garbled the frame's own print (brighter than its body: a top-hat) near the true print's footprint and inset
             from the matte, filled harmonically from the body around it, with the frame's grain.
  3. ink     the true print in the frame's own print colour, lit by its body: ink = body x k per channel, k from the
             generated print's most-covered pixels, composited inside the matte (the scene's product matte, the
             job's old_silhouette) and nowhere else.
It reproduces the first real job's approved keyframes (a frosted serum and a squeezed tube, 2026-10-16), made by a
job-level prototype, within 1 level.
Work-box arrays throughout; the matte M is the generated product's own matte (SAM 3) over the work box.
"""
import numpy as np
from scipy import ndimage as ndi

from . import fill as ipfill
from . import images as im

DEFAULTS = {
    "open": "auto",       # ref px: the grey opening that reads the packshot's body light (wider than its strokes)
    "floor": 0.08,        # packshot coverage below this is body, not print
    "blur": 0.4,          # scene px: softness of the warped coverage
    "frame_open": 7,      # scene px: the grey opening that reads the frame's body light
    "gen_thresh": 0.025,  # linear luminance above the frame's body that counts as its (garbled) print
    "gen_near": 5,        # scene px: garbled print only this near the true print's footprint (null: anywhere in the band)
    "gen_inset": 4,       # scene px: ... and this far inside the matte (with gen_near)
    "gen_dilate": 2,      # scene px: the fill reaches this far past the garbled print
    "ink_pct": 90,        # percentile of frame / body over the garbled print's core that sets the ink's colour
    "seed": 7,            # the grain over the fill
}


class InkError(ValueError):
    pass


def light(matte):
    """True for a light-ink matte (print lighter than the body it sits on)."""
    return isinstance(matte, dict) and matte.get("type") == "ink" and matte.get("polarity") == "light"


def config(matte):
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in matte.items() if k in DEFAULTS})
    cfg["print_box"] = matte.get("print_box")
    return cfg


def band_mask(refmask, shape):
    """The reference outline at reference pixels, edges inclusive (PIL's polygon fill). The approved keyframes were
    made so, and the garbled print's fill set (and with it the grain draw) depends on the band's edge pixels."""
    H, W = shape[:2]
    if refmask.polygon is None:                      # alpha, key_white: the outline's own mask
        Y, X = np.mgrid[0:H, 0:W].astype(float)
        return refmask.sample(X, Y) > 0.5
    from PIL import Image, ImageDraw
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in refmask.polygon], fill=255)
    return np.asarray(m) > 127


def stroke_open(Lr, band, box):
    """An opening wider than the packshot's widest print stroke (ref px, odd): 1.5 x the widest stroke, at least 7.
    The strokes: luminance above a wide opening (an eighth of the band's width), over half its 99.5th percentile,
    inside the print box; components of 20 px or more. -> (size, widest stroke px)"""
    xs = np.nonzero(band.any(0))[0]
    big = int(max(15, (xs.max() - xs.min()) // 8)) | 1
    body = ndi.grey_opening(np.where(band, Lr, ndi.grey_dilation(Lr * band, size=big)), size=(big, big))
    top = (Lr - body) * box
    sel = box & band
    if not sel.any() or np.percentile(top[sel], 99.5) <= 0:
        return 2 * (big // 2) + 1, 0.0
    ink = top > 0.5 * np.percentile(top[sel], 99.5)
    lab, n = ndi.label(ink)
    if n == 0:
        return 2 * (big // 2) + 1, 0.0
    idx = np.arange(1, n + 1)
    half = ndi.maximum(ndi.distance_transform_edt(ink), lab, idx)
    half = half[ndi.sum(ink, lab, idx) >= 20]
    widest = 2.0 * float(half.max()) if len(half) else 0.0
    return max(7, int(np.ceil(1.5 * widest)) | 1), widest


def ref_cover(ref_u8, refmask, cfg):
    """1. The packshot's print as coverage (reference px, 0..1) -> (cover, band, report)."""
    band = band_mask(refmask, ref_u8.shape)
    if not band.any():
        raise InkError("matte ink (light): the reference outline covers no packshot pixels")
    keep = np.ones(band.shape, bool)
    if cfg.get("print_box"):
        bx0, by0, bx1, by1 = [int(round(v)) for v in cfg["print_box"]]
        keep = np.zeros(band.shape, bool)
        keep[by0:by1, bx0:bx1] = True
    Lr = im.lum(im.srgb_to_lin(im.to_float(ref_u8)))
    rep = {}
    s = cfg["open"]
    if s == "auto":
        s, widest = stroke_open(Lr, band, keep)
        rep["open_auto"] = {"size": s, "widest_stroke_px": round(widest, 1)}
    s = int(s)
    # work on the band's box plus 3 openings: every step below reaches at most 2.5 openings, so this is exact
    ys, xs = np.nonzero(band)
    p = 3 * s + 8
    y0, y1 = max(0, ys.min() - p), min(band.shape[0], ys.max() + 1 + p)
    x0, x1 = max(0, xs.min() - p), min(band.shape[1], xs.max() + 1 + p)
    L, B = Lr[y0:y1, x0:x1], band[y0:y1, x0:x1]
    body = ndi.gaussian_filter(ndi.grey_opening(np.where(B, L, ndi.grey_dilation(L * B, size=s)), size=(s, s)), s / 4)
    white = float(np.percentile(L[B], 99.8))
    cov = np.clip((L - body) / np.maximum(white - body, 1e-3), 0, 1)
    cov[cov < cfg["floor"]] = 0
    cov *= B
    cov *= keep[y0:y1, x0:x1]
    full = np.zeros(band.shape)
    full[y0:y1, x0:x1] = cov
    rep.update({"open": s, "ref_white_lin": round(white, 4), "print_px": int((full > 0.5).sum())})
    return full, band, rep


def warp_cover(cov_r, band, H, roi, ss=4, prefilter=0.6, blur=0.4):
    """2. The coverage and the band warped into the work box (ss x ss samples per pixel) -> (cover, in_band)."""
    x0, y0, x1, y1 = roi
    w, h = x1 - x0, y1 - y0
    xs = x0 + (np.arange(w * ss) + 0.5) / ss - 0.5
    ys = y0 + (np.arange(h * ss) + 0.5) / ss - 0.5
    X, Y = np.meshgrid(xs, ys)
    ok = None
    if hasattr(H, "inverse"):
        u, v, ok = H.inverse(X, Y)
    else:
        q = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)] @ np.linalg.inv(H).T
        u, v = (q[:, 0] / q[:, 2]).reshape(X.shape), (q[:, 1] / q[:, 2]).reshape(X.shape)
    del X, Y
    if ok is None:
        ok = np.ones(u.shape, bool)
    c = ndi.map_coordinates(ndi.gaussian_filter(cov_r, prefilter), [v, u], order=1, mode="constant") * ok
    b = ndi.map_coordinates(band.astype(float), [v, u], order=1, mode="constant") * ok

    def down(z):
        return z.reshape(h, ss, w, ss).mean(axis=(1, 3))

    cov = down(c)
    if blur > 0:
        cov = ndi.gaussian_filter(cov, blur)
    return cov, down(b)


def garbled(crop_u8, M, cover, in_band, cfg, vis=None):
    """3. The frame's garbled print removed: -> (filled linear rgb, info for composite(), report).
    M: the generated product's matte over the work box (bool). info carries k (the ink's colour), the composite's
    alpha and region, and M."""
    Fl = im.srgb_to_lin(im.to_float(crop_u8))
    inner = ndi.binary_erosion(M, iterations=1)
    work = inner & (in_band > 0.5)
    if not work.any():
        raise InkError("matte ink (light): the warped reference outline doesn't overlap the product's matte: check "
                       "align and old_silhouette (the matte)")
    Lf = im.lum(Fl)
    so = int(cfg["frame_open"])
    top = Lf - ndi.grey_opening(np.where(M, Lf, 0), size=(so, so))
    gen = work & (top > cfg["gen_thresh"])
    if vis is not None:
        gen &= vis > 0.5
    if cfg["gen_near"] is not None:
        # only garbled print near where the true print goes (its layout is the true one's, a few px off), inset from
        # the matte: a frosted body's own highlights (the band under the shoulder, the limb rims, the base) stay
        near = ndi.binary_dilation(cover > 0.1, iterations=int(cfg["gen_near"]))
        gen &= near & ndi.binary_erosion(M, iterations=int(cfg["gen_inset"]))
    if not gen.any():
        raise InkError("matte ink (light): no garbled print brighter than the frame's body inside the band "
                       "(gen_thresh, frame_open): is the print light on this product?")
    core = work & (top > np.percentile(top[gen], 75))
    dil = int(cfg["gen_dilate"])
    fillm = (ndi.binary_dilation(gen, iterations=dil) if dil > 0 else gen) & ndi.binary_erosion(M, iterations=2)
    known = M & ~fillm
    try:
        filled = ipfill.harmonic_fill(Fl, fillm, known)[0]
    except ipfill.FillError as e:
        raise InkError(f"matte ink (light): garbled print with no body around it: {e}") from e
    # the frame's grain over the filled pixels (MAD of the high-pass on the body it kept)
    hp = Fl - ndi.gaussian_filter(Fl, (1.2, 1.2, 0))
    sel = known & work
    sd = 1.4826 * np.median(np.abs(hp[sel] - np.median(hp[sel], 0)), 0)
    rng = np.random.default_rng(int(cfg["seed"]))
    filled[fillm] += rng.normal(0, 1, (int(fillm.sum()), 3)) * sd
    # the ink's colour from the generated print's fully covered pixels: k = frame / body, per channel
    k = np.maximum(np.percentile(Fl[core] / np.maximum(filled[core], 1e-4), cfg["ink_pct"], axis=0), 1.0)
    a = np.clip(cover, 0, 1) * inner * np.clip(in_band, 0, 1)
    if vis is not None:
        a = a * vis
    info = {"polarity": "light", "k": k, "alpha": a, "region": work | fillm, "matte": M.copy(), "fill": fillm}
    rep = {"filled_px": int(fillm.sum()), "garbled_px": int(gen.sum()), "grain_sd_lin": sd.round(5).tolist(),
           "ink_k": k.round(3).tolist()}
    return filled, info, rep


def composite(filled, crop_u8, info):
    """The true print laid on the filled body: -> (work box uint8, alpha). Pixels outside the matte stay the scene's,
    and inside it only the band and the fill change."""
    Fl = im.srgb_to_lin(im.to_float(crop_u8))
    a = info["alpha"]
    ink = np.clip(filled * info["k"], 0, 1)
    out_l = filled * (1 - a[..., None]) + ink * a[..., None]
    out_l = np.where(info["region"][..., None], out_l, Fl)
    out = np.clip(np.round(im.lin_to_srgb(np.clip(out_l, 0, 1)) * 255), 0, 255).astype(np.uint8)
    M = info["matte"]
    out[~M] = crop_u8[~M]
    return out, a


def check(crop_u8, out_u8, info, cover, cfg, offset):
    """The keyframe may change pixels only on print (pitfall 93): changed pixels away from the true print's footprint
    (any coverage, grown by gen_near + gen_dilate + 1 px), or on the matte's rim (4 px) outside it, are the body's
    own highlights taken for print. On the first real job's approved keyframes every changed pixel lies within 6 px
    of the true print (with gen_near). -> audit findings ("ink_edges")."""
    changed = (out_u8 != crop_u8).any(-1)
    reach = int(cfg["gen_near"] if cfg["gen_near"] is not None else DEFAULTS["gen_near"]) + int(cfg["gen_dilate"]) + 1
    foot = ndi.binary_dilation(cover > 1e-3, iterations=reach)
    M = info["matte"]
    rim = M & ~ndi.binary_erosion(M, iterations=4)
    out = []
    for name, sel, what in (("rim", changed & rim & ~foot, "on the matte's rim, away from the print"),
                            ("away", changed & ~rim & ~foot, "away from the true print")):
        lab, n = ndi.label(sel)
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            px = int((lab[sl] == i).sum())
            ox, oy = offset
            out.append({"check": "ink_edges", "level": "WARN" if px >= 12 else "INFO", "px": px,
                        "box": [sl[1].start + ox, sl[0].start + oy, sl[1].stop - 1 + ox, sl[0].stop - 1 + oy],
                        "text": f"{px} px changed {what}: a body highlight taken for print? (gen_near, print_box)"})
    return sorted(out, key=lambda f: -f["px"])[:8]


def layers(filled, out_u8, info):
    """For the PSD: the filled body and the ink in sRGB, and an sRGB-space alpha that flattens to `out_u8` (the
    composite blends in linear light). -> (base, card, alpha)"""
    base = im.lin_to_srgb(np.clip(filled, 0, 1))
    card = im.lin_to_srgb(np.clip(filled * info["k"], 0, 1))
    out = out_u8.astype(float) / 255.0
    lb, lc, lo = im.lum(base), im.lum(card), im.lum(out)
    alpha = np.where(lc - lb > 1e-3, np.clip((lo - lb) / np.maximum(lc - lb, 1e-3), 0, 1), 0.0) * info["region"]
    return base, card, alpha
