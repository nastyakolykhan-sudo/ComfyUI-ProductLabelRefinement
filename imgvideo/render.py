"""Re-project the artwork into a frame: the stills engine's warp, then relight, optics and composite against that frame.

Print on a body (white ink on frosted glass, a tube's print): the stills engine's ink matte, per frame. The source's own
pixels stay wherever both the source and the artwork show bare body, so its light, speculars and grain are kept as
they are. The source's print (garbled or soft) is taken out with the stills engine's harmonic fill from the body around it,
and the artwork's ink goes on top: its coverage warped by the frame's mapping (averaged over sub-frame mappings for
motion blur), its colour the artwork's ink lit by the source's body light (body luminance ratio, low-pass). Light
that moves across the label (sparkles, glints passing in front) is added back over the new print."""
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, grey_opening

from imgpost import fill as fl
from imgpost import grade as gr
from imgpost import finish as fn
from imgpost import geometry as geo
from imgpost import images as im
from imgpost import outline as ol
from imgpost import warp as wp


def normconv(x, m, sigma):
    """Gaussian of x over the pixels in m only, spread into the gaps (normalized convolution) -> (field, weight)."""
    den = gaussian_filter(m.astype(float), sigma)
    return gaussian_filter(np.where(m, x, 0.0), sigma) / np.maximum(den, 1e-6), den


def shift_ref(H, ox, oy):
    """The same mapping for a reference cropped at (ox, oy): reference px (u', v') = (u - ox, v - oy)."""
    if isinstance(H, geo.Residual):                         # the scene-space correction stays; its base moves
        import copy
        R = copy.copy(H)
        R.__dict__["base"] = shift_ref(H.base, ox, oy)
        return R
    from .surface import Tube
    if isinstance(H, Tube):
        return Tube(H.u0 - ox, H.Ru, H.R + H.k * oy, H.k, H.A, H.a + H.A[:, 1] * oy)
    if isinstance(H, geo.Cylinder):
        k = 1.0 + H.c[1] * oy
        return geo.Cylinder(H.axis_x - ox, H.r, H.A / k, (H.a + H.A[:, 1] * oy) / k, H.c / k, H.wrap)
    T = np.array([[1.0, 0, ox], [0, 1.0, oy], [0, 0, 1.0]])
    return H @ T


class Artwork:
    """The artwork prepared once, cropped to the label's outline: ink coverage (0..1), the body's own luminance with
    the print taken out, and the ink's colour. polarity "light": ink brighter than the body (white print on glass);
    "dark": ink darker (black text on a white label)."""

    def __init__(self, ref_u8, ref_a, outline_spec, polarity="light", tophat=31, thresh=0.08, prefilter=0.6, pad=48,
                 edge_band=16):
        full = ol.RefMask(outline_spec, ref_u8, ref_a)
        rx0, ry0, rx1, ry1 = full.bbox(ref_u8.shape, pad=pad)
        self.off = (rx0, ry0)
        crop_u8 = ref_u8[ry0:ry1, rx0:rx1]
        spec = dict(outline_spec)
        if spec.get("type") == "polygon":
            spec["pts"] = [[x - rx0, y - ry0] for x, y in spec["pts"]]
        elif spec.get("type") not in ("alpha", "key_white"):
            raise ValueError("imgvideo: reference_outline must be a polygon, alpha or key_white for now")
        self.refmask = ol.RefMask(spec, crop_u8, None if ref_a is None else ref_a[ry0:ry1, rx0:rx1])
        lin = im.srgb_to_lin(im.to_float(crop_u8))
        L = im.lum(lin)
        ry, rx = np.mgrid[0:L.shape[0], 0:L.shape[1]].astype(float)
        inside = self.refmask.sample(rx, ry) > 0.5
        sgn = 1.0 if polarity == "light" else -1.0
        # print = strokes narrower than the top-hat, brighter (or darker) than the body around them
        th = sgn * (L - grey_opening(L, size=(tophat, tophat))) if sgn > 0 else \
            -(L - ndi.grey_closing(L, size=(tophat, tophat)))
        # a packshot's lit rim (and its anti-aliased edge) is a bright line along the outline too, but it isn't print
        solid = inside if ref_a is None else inside & (ref_a[ry0:ry1, rx0:rx1] > 127)
        ink = inside & (th > thresh) & (distance_transform_edt(solid) > edge_band)
        body = inside & ~binary_dilation(ink, iterations=3)
        Lb, wb = normconv(L, body, 12.0)
        Lb2, _ = normconv(L, body, 48.0)
        Lb = np.where(wb > 0.2, Lb, Lb2)
        core = ink & (th > np.percentile(th[ink], 60))
        self.ink_rgb = np.median(lin[core], 0)
        il = float(im.lum(self.ink_rgb))
        self.cov = np.clip((L - Lb) / np.where(np.abs(il - Lb) > 1e-3, il - Lb, 1e-3), 0, 1) * \
            binary_dilation(ink, iterations=2)
        self.Lb = Lb
        pack = np.stack([self.cov, Lb, inside.astype(float)], -1)
        self.pack = np.stack([gaussian_filter(pack[..., i], prefilter) for i in range(3)], -1) if prefilter else pack
        self.polygon = None if self.refmask.polygon is None else np.asarray(self.refmask.polygon, float)
        self.stats = {"ink_px": int(ink.sum()), "ink_rgb_lin": [round(float(v), 4) for v in self.ink_rgb],
                      "crop": [int(rx0), int(ry0), int(rx1), int(ry1)]}

    def local(self, H):
        return shift_ref(H, *self.off)


def work_box(H_local, art, shape, pad=14):
    P = geo.map_polygon(H_local, art.polygon) if art.polygon is not None else None
    h, w = shape
    x0, y0 = np.floor(P.min(0)).astype(int) - pad
    x1, y1 = np.ceil(P.max(0)).astype(int) + pad
    return [max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))]


def silhouette(key, inset=1.5, soft=0.7):
    """The product's matte inset by `inset` px with a soft edge (0..1)."""
    d = distance_transform_edt(key) - inset
    return np.clip(d / max(2 * soft, 1e-3) + 0.5, 0, 1)


def render(frame_u8, H, Hsub, art, key, cfg, label_median=None, seed=0, glass=None):
    """One frame. H: the frame's mapping (reference px -> scene px); Hsub: the mappings across the shutter (motion
    blur; [H] for none); key: bool product matte, full frame; glass: a keyframe label's clear-glass weight (glass_map).
    -> (output frame uint8, info dict, layers for QA)"""
    Hl = art.local(H)
    roi = work_box(Hl, art, frame_u8.shape[:2])
    x0, y0, x1, y1 = roi
    ss = int(cfg.get("supersample", 4))
    crop = im.to_float(frame_u8[y0:y1, x0:x1])
    S_lin = im.srgb_to_lin(crop)
    W, a_lab, (U, V) = wp.warp(art.pack, Hl, roi, art.refmask, ss=ss, prefilter=0)
    cov = W[..., 0] if len(Hsub) <= 1 else np.mean(
        [wp.warp(art.pack, art.local(h), roi, art.refmask, ss=max(2, ss // 2), prefilter=0)[0][..., 0] for h in Hsub], 0)
    sil = silhouette(key[y0:y1, x0:x1], cfg.get("inset", 1.5))
    R = np.clip(a_lab, 0, 1) * sil
    act = R > 0.5
    # static softness of the frame (lens, encoder): the new print's edges match it
    if cfg.get("blur", 0) > 0:
        cov = gaussian_filter(cov, float(cfg["blur"]))
    cov = cov * R
    # the source's own print: strokes narrower than the top-hat, brighter (darker) than the body around them
    Ls = im.lum(S_lin)
    sgn = 1.0 if cfg.get("polarity", "light") == "light" else -1.0
    th_s = int(cfg.get("print_tophat", 7))
    top = Ls - grey_opening(Ls, size=(th_s, th_s)) if sgn > 0 else ndi.grey_closing(Ls, size=(th_s, th_s)) - Ls
    if cfg.get("polarity", "light") == "both":             # light and dark print (several inks)
        top = np.maximum(Ls - grey_opening(Ls, size=(th_s, th_s)), ndi.grey_closing(Ls, size=(th_s, th_s)) - Ls)
    P_src = binary_dilation(top > cfg.get("print_thresh", 0.03), iterations=1)
    # a glass or plastic rim along the silhouette is a thin bright line too, but it isn't print
    P_src &= distance_transform_edt(key[y0:y1, x0:x1]) > float(cfg.get("edge_band", 4.0))
    if getattr(art, "texture_mode", False):              # a keyframe's whole label, relit (no ink, no fill)
        info = {"roi": roi, "src_print_px": int((P_src & act).sum()), "blur_px": round(float(cfg.get("blur", 0)), 3)}
        return _render_texture(frame_u8, crop, S_lin, Hl, Hsub, art, R, act, P_src, a_lab, sil, U, V, roi, cfg, info,
                               label_median, seed, key, glass)
    P_ink = binary_dilation(cov > 0.04, iterations=1)
    F = (P_src | P_ink) & act
    D = binary_dilation(F) & ~F & act
    info = {"roi": roi, "fill_px": int(F.sum()), "src_print_px": int((P_src & act).sum()), "blur_px": round(float(cfg.get("blur", 0)), 3)}
    base = crop.copy()
    if F.any():
        try:
            base, _ = fl.harmonic_fill(crop, F, D, offset=(x0, y0))
        except fl.FillError:                                     # a hole with no body around it: never stop a shot
            for c in range(3):
                base[..., c] = np.where(F, normconv(crop[..., c], ~F & act, 4.0)[0], crop[..., c])
            info["fill_fallback"] = True
        ring = binary_dilation(F, iterations=3) & ~F & act
        gstd = fn.mad_std(fn.highpass(crop)[ring]) if ring.sum() > 50 else np.zeros(3)
        rng_grain = fn.make_grain(F.shape, gstd, F, seed=seed)
        base[F] = np.clip(base[F] + rng_grain[F], 0, 1)
        info["grain_std"] = [round(float(v), 4) for v in gstd]
    Fs = np.clip(gaussian_filter(F.astype(float), 0.6), 0, 1) * R
    base = crop * (1 - Fs[..., None]) + base * Fs[..., None]
    base_lin = im.srgb_to_lin(base)
    # the ink lit like the source's body: body luminance ratio, source over artwork, low-pass
    Lb_src, _ = normconv(im.lum(base_lin), act, float(cfg.get("light_sigma", 2.0)))
    light = np.clip(Lb_src / np.maximum(W[..., 1], 1e-4), *cfg.get("light_clip", (0.2, 3.0)))
    # white balance: the scene's white (a white part of the product, its chroma) over the artwork's
    wb = np.asarray(cfg.get("white_chroma", [1, 1, 1]), float) / np.asarray(cfg.get("ref_white_chroma", [1, 1, 1]), float)
    if getattr(art, "ink_field", None) is not None:          # a keyframe: each print pixel keeps its own ink
        if not hasattr(art, "ink_pack"):
            art.ink_pack = art.ink_field
        ink_rgb = wp.warp(art.ink_pack, Hl, roi, art.refmask, ss=max(2, ss // 2), prefilter=0)[0]
    else:
        ink_rgb = art.ink_rgb[None, None, :]
    ink_lin = ink_rgb * (wb * np.asarray(cfg.get("ink_gain", [1, 1, 1]), float))[None, None, :] * light[..., None]
    ink = im.lin_to_srgb(np.clip(ink_lin, 0, None))
    ink = np.clip(ink + fn.make_grain(ink.shape[:2], info.get("grain_std", np.zeros(3)), act, seed=seed + 7), 0, 1)
    out = base * (1 - cov[..., None]) + ink * cov[..., None]
    # light moving across the label (sparkles in front): the source's excess over its own still label, re-added
    # where the print changed (elsewhere the source pixel is already there)
    if label_median is not None:
        Mu = label_median(U, V)
        if Mu is not None:
            exc = S_lin - Mu
            e = im.lum(exc)
            e_hp = e - gaussian_filter(e, float(cfg.get("spark_sigma", 4.0)))
            # brighter than the still label anywhere close by: the source's own print wobbling a px or two (its
            # strokes are in the still label next to where they are now) is not light passing in front
            n = int(cfg.get("spark_reach", 7))
            e_mx = Ls - ndi.grey_dilation(im.lum(Mu), size=(n, n))
            t = float(cfg.get("spark_thresh", 0.05))
            g = np.clip((np.minimum(np.minimum(e, e_hp), e_mx) - t) / t, 0, 1) * R
            spark = np.clip(exc, 0, None) * g[..., None]
            wgt = np.maximum(Fs, cov)
            out = im.lin_to_srgb(im.srgb_to_lin(out) + spark * wgt[..., None])
            info["spark_px"] = int((g * wgt > 0.25).sum())
    if getattr(art, "unroll", False) and hasattr(Hl, "inverse"):
        # unrolled label: rays that meet the label's band on the surface, against where the keyframe left texture
        Yg, Xg = np.mgrid[y0:y1, x0:x1].astype(float)
        uu, vv, okk = Hl.inverse(Xg, Yg)
        okk = np.ones_like(Xg, bool) if okk is None else okk
        v0, v1 = art.polygon[:, 1].min(), art.polygon[:, 1].max()
        band = okk & (vv >= v0) & (vv <= v1)
        unseen = band & (sil > 0.5) & ~binary_dilation(a_lab > 0.5, iterations=2)
        info["unseen_px"] = int(unseen.sum())
        info["label_px"] = int((band & (sil > 0.5)).sum())
    elif getattr(art, "quad", None) is not None:
        # the product's label at this frame (its matte across the label's band) that the keyframe never saw: no
        # texture there, so the source stays (counted, not invented)
        from imgpost.masks import poly_cover
        band = poly_cover(roi, geo.map_polygon(Hl, art.quad).tolist(), ss=2) > 0.5
        seen = a_lab > 0.5
        unseen = band & (sil > 0.5) & ~binary_dilation(seen, iterations=2)
        info["unseen_px"] = int(unseen.sum())
        info["label_px"] = int((band & (sil > 0.5)).sum())
    res = frame_u8.copy()
    res[y0:y1, x0:x1] = im.to_u8(out)
    info["ink_px"] = int((cov > 0.5).sum())
    return res, info, {"roi": roi, "R": R, "cov": cov, "F": F, "P_src": P_src & act, "U": U, "V": V}


CLEAR_GLASS = {"dist": 0.2, "ref_px": 64, "spread": 1.5, "soft": 1.0, "near": 10}


def glass_map(art, med, step, band, fit_band, spec=True):
    """Clear glass under a keyframe's label (render.clear_glass, V56): in the rows the label's band was stretched over
    past the body's own band (anchor.fit_band) to carry the print the keyframe corrected (V54), the still label's cells
    whose bare surface (the label's print 2 cells round taken out) isn't the body's hue: chromaticity (log r/g, log b/g
    of linear RGB) more than `dist` from the body's just inside the stretched end (`ref_px` rows). Hue, not brightness:
    a limb's shading is the body darker (with brightness the jar's own body cells spread 0.3-0.6, and the tube's pink
    stretched rows read as glass). Measured: the jar's stretched rows 91 % over 0.2, the tube's none. A jar's last lines sit where
    its pink body meets a clear glass base: carried from the keyframe, the glass there (and the keyframe's fills over
    the garbled print in it) showed as soft smudges against the frame's own sharp reflections, and the glitter in
    front was lost. Cells under the print take their bare neighbours' call (`spread` cells); the weight is smoothed by
    `soft` cells. -> a function (U, V) local reference px -> weight 0..1, or None (no stretched rows, no glass)"""
    from scipy.ndimage import map_coordinates
    from . import labelspace as ls
    c = dict(CLEAR_GLASS, **(spec if isinstance(spec, dict) else {}))
    oy = art.off[1]
    f0, f1 = fit_band[0] - band[0] - oy, fit_band[1] - band[0] - oy      # the body's band, in local reference rows
    if not (fit_band[0] > band[0] or fit_band[1] < band[1]):
        return None
    grid = ls.Grid(art, step)
    v = grid.v
    ok = grid.inside & np.isfinite(med[..., 0])
    fit = (v >= f0) & (v <= f1)
    stretch = ok & ~fit
    if not stretch.any():
        return None
    near = binary_dilation(map_coordinates(art.pack[..., 0], [grid.v, grid.u], order=1) > 0.05, iterations=2)
    bare = ok & ~near
    ends = np.zeros_like(fit)
    if fit_band[1] < band[1]:
        ends |= (v > f1 - float(c["ref_px"])) & fit
    if fit_band[0] > band[0]:
        ends |= (v < f0 + float(c["ref_px"])) & fit
    body = bare & ends
    if body.sum() < 20:
        return None
    lm = np.log(np.maximum(med, 1e-4))
    ch = np.stack([lm[..., 0] - lm[..., 1], lm[..., 2] - lm[..., 1]], -1)
    d = np.max(np.abs(ch - np.median(ch[body], 0)), -1)
    g = (stretch & bare & (d > float(c["dist"]))).astype(float)
    if not g.any():
        return None
    gs, wt = normconv(g, stretch & bare, float(c["spread"]))
    W = np.where(bare, g, np.where(wt > 0.05, gs, 0.0)) * stretch
    W = np.clip(gaussian_filter(W, float(c["soft"])), 0, 1) * stretch

    def weight(U, Vv):
        return map_coordinates(W, [Vv / step, U / step], order=1, mode="constant", cval=0.0)
    weight.cells = int((W > 0.5).sum())
    weight.near = int(c["near"])
    return weight


def _fill_print(crop, F, act, roi):
    """crop with the px in F filled from the surface round them (harmonic, else normalized convolution), blended in
    over a 0.6 px edge."""
    if not F.any():
        return crop
    D = binary_dilation(F) & ~F & act
    try:
        base, _ = fl.harmonic_fill(crop, F, D, offset=(roi[0], roi[1]))
    except fl.FillError:
        base = crop.copy()
        for ch in range(3):
            base[..., ch] = np.where(F, normconv(crop[..., ch], ~F & act, 4.0)[0], crop[..., ch])
    Fs = np.clip(gaussian_filter(F.astype(float), 0.6), 0, 1)
    return crop * (1 - Fs[..., None]) + base * Fs[..., None]


def _clear_glass(comp, crop, out, alpha, Wk, P_src, act, sil, a_lab, U, V, roi, glass, info, Wk_src=None):
    """Over clear glass (glass_map's weight) the frame keeps its own surface: its print (the generator's line, within
    `near` px of the label's) is filled from the glass round it, and only the label's print goes on, relit as the
    label's (render.clear_glass, V56). Wk_src: the label's print where the tracking puts it, when the render places
    it elsewhere (render.body_snap.turn holds it off the generator's print, V59): the frame's print is filled near
    either."""
    x0, y0 = roi[0], roi[1]
    gw = glass(np.nan_to_num(U), np.nan_to_num(V)) * (a_lab > 0.5) * (sil > 0)
    gw = gw * np.clip(4.0 * alpha, 0, 1)                    # the label's own fades and holes still apply
    on = (gw > 0.02) & act
    if not on.any():
        return comp
    cov = np.clip(Wk, 0, 1) * sil
    lab = cov > 0.2
    if Wk_src is not None:
        lab = lab | (np.clip(Wk_src, 0, 1) * sil > 0.2)
    F = (binary_dilation(cov > 0.04, iterations=1) |
         (P_src & binary_dilation(lab, iterations=int(getattr(glass, "near", 10))))) & on
    base = _fill_print(crop, F, act, roi)
    a_ink = np.clip(cov * alpha, 0, 1)
    comp_g = base * (1 - a_ink[..., None]) + out * a_ink[..., None]
    info["clear_glass_px"] = int((gw > 0.5).sum())
    info["clear_glass_fill_px"] = int(F.sum())
    return comp * (1 - gw[..., None]) + comp_g * gw[..., None]


def print_rows(art, strip=4, line=0.25, min_px=3):
    """Where the label's own print starts and ends along its rows (local reference px), or None: its coverage over 0.2
    where the texture is on the product, counted in strips of `strip` rows; a strip is print when it holds `line` x
    the median printed strip (and min_px) or more, so the odd dark speck in a glass base under the last line (the
    keyframe's relit refraction) doesn't count. Cached on the artwork."""
    if not hasattr(art, "_print_rows"):
        p = (art.pack[..., 0] > 0.2) & (art.pack[..., 2] > 0.5)
        n = p.sum(1)
        k = int(np.ceil(len(n) / strip)) * strip
        h = np.pad(n, (0, k - len(n))).reshape(-1, strip).sum(1)
        ok = h > 0
        dense = np.nonzero(ok & (h >= max(min_px, line * float(np.median(h[ok])) if ok.any() else min_px)))[0]
        art._print_rows = (float(dense[0] * strip), float(min(dense[-1] * strip + strip, len(n)) - 1)) \
            if len(dense) else None
    return art._print_rows


def print_fade(art, fv, gap):
    """The end fade's length at the label's top and bottom (feather.print_gap): at most fv rows, ending gap px before
    the label's first (last) print row, at least 1 px. -> (top px, bottom px)"""
    pr = print_rows(art)
    if pr is None:
        return fv, fv
    vlo, vhi = art.polygon[:, 1].min(), art.polygon[:, 1].max()
    return (float(np.clip(pr[0] - gap - vlo, 1.0, fv)), float(np.clip(vhi - gap - pr[1], 1.0, fv)))


SHADE_PRINT = {"size": 61, "contrast": 0.5, "grow": 2, "rim": 0.15}


def shade_print(S_lin, T_lin, act, polarity, spec=True, key=None):
    """Print in the frame or in the label (render.shade_print): px under `contrast` x the body round them (a grey
    closing `size` px wide; light print: over 1 / contrast x a grey opening), grown `grow` px, away from the product's
    limbs (`rim` x the matte's half-width in from its edge, `key` the matte). Black print on a light body sits at a
    tenth of it. The limbs stay in the light's read: a packshot keyframe's darker limb is light the frame corrects
    (masked, the tube's right limb came out black). -> bool mask"""
    c = dict(SHADE_PRINT, **(spec if isinstance(spec, dict) else {}))
    k = int(c["size"])
    out = np.zeros(act.shape, bool)
    for img in (S_lin, T_lin):
        L = np.where(act, im.lum(img), np.nan)
        Lf = np.nan_to_num(L, nan=float(np.nanmedian(L)) if np.isfinite(L).any() else 0.0)
        if polarity in ("dark", "both"):
            out |= Lf < float(c["contrast"]) * ndi.grey_closing(Lf, size=(k, k))
        if polarity in ("light", "both"):
            out |= Lf > ndi.grey_opening(Lf, size=(k, k)) / float(c["contrast"])
    out = binary_dilation(out & act, iterations=int(c["grow"]))
    if key is not None and key.any():
        d = distance_transform_edt(key)
        out &= d > float(c["rim"]) * float(d.max())
    return out


def _shade(img, m, sigma):
    """Low-pass of img (h x w x 3) over the pixels in m, filled into the gaps from further out."""
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        a, wa = normconv(img[..., c], m, sigma)
        b, _ = normconv(img[..., c], m, 4 * sigma)
        out[..., c] = np.where(wa > 0.15, a, b)
    return np.maximum(out, 1e-4)


def _ratio_field(S_lin, T, m, block=8, med=5, sigma=10.0, local=False):
    """The frame's light over the keyframe's, per channel, as a smooth field: log(S / T) block medians over m, then a
    median across blocks, so print in either image (the frame's garbled, the keyframe's correct one) and small
    glints are outvoted by the bare surface around them; filled into the gaps, smoothed, back to pixels.
    local (render.shade_local, V58): the median across blocks takes the bare blocks round each one only, and the
    smoothing weighs each block by how much of it is bare. Without it the blocks print covers were first filled with
    the frame's overall median, which near lines of print outvoted the bare blocks round them by an amount that
    changed as the frame's own lines morphed: the label's body boiled in soft patches round its text."""
    from scipy.ndimage import median_filter, zoom
    h, w = m.shape
    r = np.log(np.maximum(S_lin, 1e-4)) - np.log(np.maximum(T, 1e-4))
    out = np.empty_like(r)
    sg = max(sigma / block, 0.7)
    for c in range(3):
        g, cnt = _block_stat_nan(r[..., c], m, block)
        ok = np.isfinite(g)
        if local and ok.any():
            gl = _nan_median(g, med)
            wgt = np.where(ok, np.clip(cnt / float(block * block), 0.0, 1.0), 0.0)
            v = np.where(ok, gl, 0.0)
            f, wt = gaussian_filter(v * wgt, sg) / np.maximum(gaussian_filter(wgt, sg), 1e-6), gaussian_filter(wgt, sg)
            f2 = gaussian_filter(v * wgt, 4 * sg) / np.maximum(gaussian_filter(wgt, 4 * sg), 1e-6)
            f = np.where(wt > 0.1, f, f2)
        else:
            gm = np.where(ok, g, np.nanmedian(g) if ok.any() else 0.0)
            gm = np.where(ok, median_filter(gm, size=med, mode="nearest"), gm)
            f, wt = normconv(gm, ok, sg)
            f2, _ = normconv(gm, ok, 4 * sg)
            f = np.where(wt > 0.15, f, f2)
        out[..., c] = gaussian_filter(zoom(f, block, order=1)[:h, :w], block / 2)
    return np.exp(out)


def _nan_median(g, size):
    """Median of the finite values in a size x size window round each cell (NaN where there are none)."""
    import warnings
    k = size // 2
    p = np.pad(g, k, mode="constant", constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(p, (size, size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win.reshape(*g.shape, -1), axis=-1)


def _block_stat_nan(v, valid, block):
    """Block medians of v over valid pixels (NaN where a block has under a quarter of its pixels valid)."""
    import warnings
    h, w = v.shape
    hb, wb = -(-h // block), -(-w // block)
    pad = np.full((hb * block, wb * block), np.nan)
    pad[:h, :w] = np.where(valid, v, np.nan)
    b = pad.reshape(hb, block, wb, block).transpose(0, 2, 1, 3).reshape(hb, wb, -1)
    cnt = np.isfinite(b).sum(-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.where(cnt[..., None] >= block * block // 4, b, np.nan), axis=-1)
    return med, cnt


def _render_texture(frame_u8, crop, S_lin, Hl, Hsub, art, R, act, P_src, a_lab, sil, U, V, roi, cfg, info, label_median,
                    seed, key, glass=None):
    """A corrected keyframe's whole label, relit: out = keyframe texture x (frame's low-pass shading / keyframe's
    low-pass shading), both read on bare surface (print taken out), so the keyframe's own light never travels with
    the label and each frame's light, colour and reflections come from the frame. Where the keyframe never saw the
    label (it turned into view) the frame stays as it is."""
    x0, y0, x1, y1 = roi
    ss = int(cfg.get("supersample", 4))
    T = wp.warp(art.texture, Hl, roi, art.refmask, ss=ss, prefilter=0)[0]
    if len(Hsub) > 1:                                        # motion blur across the shutter
        T = np.mean([wp.warp(art.texture, art.local(h), roi, art.refmask, ss=max(2, ss // 2), prefilter=0)[0] for h in Hsub], 0)
    if cfg.get("smear"):                                     # the source's own smear along the motion on this frame
        from .optics import line_blur
        Ln, dx, dy = cfg["smear"]
        T = line_blur(T, np.array([dx, dy]), float(Ln))
        info["smear_px"] = float(Ln)
    Wk = wp.warp(art.pack, Hl, roi, art.refmask, ss=max(2, ss // 2), prefilter=0)[0][..., 0]
    Pk = Wk > 0.2                                            # keyframe's print
    # (clear glass, V59) the label's print where the tracking puts it, when the render places it elsewhere
    Wk_s = wp.warp(art.pack, art.local(cfg["source_H"]), roi, art.refmask, ss=max(2, ss // 2), prefilter=0)[0][..., 0] \
        if (glass is not None or cfg.get("held_turn")) and cfg.get("source_H") is not None else None
    occ = None
    if cfg.get("occluder_feather"):
        # something in front (a sparkle trail the matte cuts out) leaves a hole inside the label; only holes well
        # inside it count (a limb strip the model overhangs isn't one)
        lab = a_lab > 0.5
        occ = lab & ~key[y0:y1, x0:x1] & (distance_transform_edt(lab) > float(cfg.get("occluder_depth", 6.0)))
        occ = occ if occ.any() else None
    Ts, act_s, Us, Vs = T, act, U, V
    if cfg.get("source_H") is not None:
        # a morph frame: the print goes on rigid, but the frame's light and sparkles are read against the label where
        # the source's own print sits (its tracks), or the misregistered garbled print reads as light
        Ws_, a_s, (Us, Vs) = wp.warp(art.texture, art.local(cfg["source_H"]), roi, art.refmask, ss=ss, prefilter=0)
        Ts, act_s = Ws_, act & (a_s > 0.5)
    act_r = act_s
    if cfg.get("shade_print"):
        # the frame's light read on bare surface only (V54): the source's own print and the label's left out, at any
        # stroke width. The block medians outvote thin print, but a bold logo the generator drew bigger and elsewhere
        # (a tube's logo letter, 30-40 px strokes) wins its blocks: its darkness travelled with the label as a shadow,
        # and the label's own letter, over the source's bare body, came out with a halo
        sp = shade_print(S_lin, Ts, act_s, cfg.get("polarity", "light"), cfg["shade_print"], key[y0:y1, x0:x1])
        act_r = act_s & ~sp
        info["shade_print_px"] = int((sp & act_s).sum())
    out_lin = T * _ratio_field(S_lin, Ts, act_r, int(cfg.get("shade_block", 8)), int(cfg.get("shade_median", 5)),
                               float(cfg.get("shade_sigma", 10.0)), bool(cfg.get("shade_local")))
    if cfg.get("blur", 0) > 0:
        out_lin = np.stack([gaussian_filter(out_lin[..., c], float(cfg["blur"])) for c in range(3)], -1)
    out = im.lin_to_srgb(gr.rolloff(np.clip(out_lin, 0, None), cfg.get("rolloff", 0.95)))
    ring = act & ~binary_dilation(P_src, iterations=2)
    gstd = fn.mad_std(fn.highpass(crop)[ring]) - fn.mad_std(fn.highpass(out)[ring]) if ring.sum() > 50 else np.zeros(3)
    gstd = np.clip(gstd, 0, None)
    out = np.clip(out + fn.make_grain(out.shape[:2], gstd, act, seed=seed), 0, 1)
    if label_median is not None and cfg.get("sparkles", True):   # light passing in front (sparkles), re-added
        Mu = label_median(Us, Vs)
        if Mu is not None:
            exc = S_lin - Mu
            e = im.lum(exc)
            e_hp = e - gaussian_filter(e, float(cfg.get("spark_sigma", 4.0)))
            n = int(cfg.get("spark_reach", 7))
            e_mx = im.lum(S_lin) - ndi.grey_dilation(im.lum(Mu), size=(n, n))
            t = float(cfg.get("spark_thresh", 0.05))
            g = np.clip((np.minimum(np.minimum(e, e_hp), e_mx) - t) / t, 0, 1) * R
            if cfg.get("spark_glow"):
                # dense sparkle trails glow: their soft haze is light too, not only the bright cores. The haze is the
                # low-passed excess well above the still label, near cores; everything is feathered, so the trail
                # meets the new print with the source's own falloff and no cut-out edge
                gs = float(cfg.get("spark_glow_sigma", 3.0))
                e_lp = gaussian_filter(e, gs)
                tg = float(cfg.get("spark_glow_thresh", 0.04))
                # both ramps start at zero (a threshold or a hard "near" mask cuts the haze's faint tail off in a step
                # the eye reads as a cut-out edge)
                src_ = g if occ is None else np.maximum(g, occ)    # a trail the matte cuts out spills haze too
                near = np.clip(gaussian_filter(src_, 3 * gs) / 0.05, 0, 1)
                glow = np.clip(e_lp / (2.0 * tg), 0, 1) * near
                g = np.clip(gaussian_filter(np.maximum(g, glow), 1.5), 0, 1) * R
            out = im.lin_to_srgb(im.srgb_to_lin(out) + np.clip(exc, 0, None) * g[..., None])
            info["spark_px"] = int((g > 0.25).sum())
    alpha = np.clip(gaussian_filter(R, float(cfg.get("edge_softness", 0.6))), 0, 1) * (sil > 0)
    fe = cfg.get("feather")
    if fe and art.polygon is not None:     # fade the label into the source over its last degrees and rows
        ulo, uhi = art.polygon[:, 0].min(), art.polygon[:, 0].max()
        vlo, vhi = art.polygon[:, 1].min(), art.polygon[:, 1].max()
        fu = max(float(fe.get("theta_deg", 10.0)) * np.pi / 180.0 * getattr(getattr(Hl, "base", Hl), "Ru", 100.0), 1.0)
        fv = max(float(fe.get("v_px", 20.0)), 1.0)
        fv0 = fv1 = fv
        if fe.get("print_gap") is not None:
            # the end fade stops print_gap px short of the label's own print (V54): faded over a line, the label lets
            # the frame's own line through beside its own, and the generator's lines drift against the tracked label
            fv0, fv1 = print_fade(art, fv, float(fe["print_gap"]))
            info["feather_v_px"] = [round(fv0, 1), round(fv1, 1)]
        def ss(x):
            x = np.clip(x, 0, 1)
            return x * x * (3 - 2 * x)
        wf = ss((U - ulo) / fu) * ss((uhi - U) / fu) * ss((V - vlo) / fv0) * ss((vhi - V) / fv1)
        alpha = alpha * wf
        info["feather"] = True
    if occ is not None:
        # the hole's haze spills into the matte, so a hard switch to the new label at its edge reads as a cut-out:
        # fade into the source over occluder_feather px there
        x = np.clip((distance_transform_edt(~occ) - 1.0) / float(cfg["occluder_feather"]), 0, 1)
        alpha = alpha * x * x * (3 - 2 * x)
        info["occluder_px"] = int(occ.sum())
    if cfg.get("held_turn") and Wk_s is not None:
        # (V59) held off its tracks by the body snap's turn, the label fades over the frame's own print a few px off
        # its letters (a seal's ring near the limb echoed beside the label's): where the label lets the frame
        # through, the frame's print near the letters where the tracking puts them is filled from the surface
        # round it
        Ff = P_src & binary_dilation(np.clip(Wk_s, 0, 1) * sil > 0.2, iterations=10) & (alpha < 0.98) & (sil > 0)
        crop = _fill_print(crop, Ff, act, roi)
        info["held_fill_px"] = int(Ff.sum())
    comp = crop * (1 - alpha[..., None]) + out * alpha[..., None]
    if glass is not None:
        comp = _clear_glass(comp, crop, out, alpha, Wk, P_src, act, sil, a_lab, U, V, roi, glass, info, Wk_s)
    if getattr(art, "unroll", False) and hasattr(Hl, "inverse"):
        Yg, Xg = np.mgrid[y0:y1, x0:x1].astype(float)
        uu, vv, okk = Hl.inverse(Xg, Yg)
        okk = np.ones_like(Xg, bool) if okk is None else okk
        v0, v1 = art.polygon[:, 1].min(), art.polygon[:, 1].max()
        band = okk & (vv >= v0) & (vv <= v1)
        unseen = band & (sil > 0.5) & ~binary_dilation(a_lab > 0.5, iterations=2)
        info["unseen_px"] = int(unseen.sum())
        info["label_px"] = int((band & (sil > 0.5)).sum())
    res = frame_u8.copy()
    res[y0:y1, x0:x1] = im.to_u8(comp)
    info.update(ink_px=int(Pk.sum()), fill_px=0, grain_std=[round(float(v), 4) for v in gstd])
    return res, info, {"roi": roi, "R": R, "cov": Pk.astype(float), "F": np.zeros_like(act), "P_src": P_src & act, "U": U, "V": V}
