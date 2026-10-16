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


def render(frame_u8, H, Hsub, art, key, cfg, label_median=None, seed=0):
    """One frame. H: the frame's mapping (reference px -> scene px); Hsub: the mappings across the shutter (motion
    blur; [H] for none); key: bool product matte, full frame. -> (output frame uint8, info dict, layers for QA)"""
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
                               label_median, seed, key)
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


def _shade(img, m, sigma):
    """Low-pass of img (h x w x 3) over the pixels in m, filled into the gaps from further out."""
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        a, wa = normconv(img[..., c], m, sigma)
        b, _ = normconv(img[..., c], m, 4 * sigma)
        out[..., c] = np.where(wa > 0.15, a, b)
    return np.maximum(out, 1e-4)


def _ratio_field(S_lin, T, m, block=8, med=5, sigma=10.0):
    """The frame's light over the keyframe's, per channel, as a smooth field: log(S / T) block medians over m, then a
    median across blocks, so print in either image (the frame's garbled, the keyframe's correct one) and small
    glints are outvoted by the bare surface around them; filled into the gaps, smoothed, back to pixels."""
    from scipy.ndimage import median_filter, zoom
    h, w = m.shape
    r = np.log(np.maximum(S_lin, 1e-4)) - np.log(np.maximum(T, 1e-4))
    out = np.empty_like(r)
    for c in range(3):
        g, cnt = _block_stat_nan(r[..., c], m, block)
        ok = np.isfinite(g)
        gm = np.where(ok, g, np.nanmedian(g) if ok.any() else 0.0)
        gm = np.where(ok, median_filter(gm, size=med, mode="nearest"), gm)
        f, wt = normconv(gm, ok, max(sigma / block, 0.7))
        f2, _ = normconv(gm, ok, 4 * max(sigma / block, 0.7))
        f = np.where(wt > 0.15, f, f2)
        out[..., c] = gaussian_filter(zoom(f, block, order=1)[:h, :w], block / 2)
    return np.exp(out)


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
                    seed, key):
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
    Pk = wp.warp(art.pack, Hl, roi, art.refmask, ss=max(2, ss // 2), prefilter=0)[0][..., 0] > 0.2    # keyframe's print
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
    out_lin = T * _ratio_field(S_lin, Ts, act_s, int(cfg.get("shade_block", 8)), int(cfg.get("shade_median", 5)),
                               float(cfg.get("shade_sigma", 10.0)))
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
        def ss(x):
            x = np.clip(x, 0, 1)
            return x * x * (3 - 2 * x)
        wf = ss((U - ulo) / fu) * ss((uhi - U) / fu) * ss((V - vlo) / fv) * ss((vhi - V) / fv)
        alpha = alpha * wf
        info["feather"] = True
    if occ is not None:
        # the hole's haze spills into the matte, so a hard switch to the new label at its edge reads as a cut-out:
        # fade into the source over occluder_feather px there
        x = np.clip((distance_transform_edt(~occ) - 1.0) / float(cfg["occluder_feather"]), 0, 1)
        alpha = alpha * x * x * (3 - 2 * x)
        info["occluder_px"] = int(occ.sum())
    comp = crop * (1 - alpha[..., None]) + out * alpha[..., None]
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
