"""Print-only compositing (render.alpha "print", pitfall V40): only the print changes, on the source's own body.

Texture mode relights the keyframe's whole label band (body and print) and puts it over the source, so where the relight
misses (frosted bodies, haze, a limb) the band's outline and its 8 px relight blocks show on the plain body: the user's
"visible square of the label composited" on the first real run. The keyframe is the source's own body plus the true
print, so only the print needs to change: the new print, and the garbled print it replaces. Everywhere else the
source stays. Per frame, inside the render's changed pixels:
  print  = the new print and, within `near` px of it, the source's garbled print (a top-hat 7 at 0.03 linear; specks
           further away are sparkles and highlights: kept), away from the silhouette's rim, where the frames differ;
           a garbled stroke counts only where the source is brighter than the render and the render is no darker
           than the source's body there less 6 levels
  alpha  = smooth(dilate(print, grow px), sigma), 0 outside the changed pixels
  target = (carry) the source's body (no brighter than a 13 px opening of it, +3 %: every light stroke in the patch
           goes, found or not) under a soft matte of the render's ink, in the render's ink colour scaled by the two
           bodies' brightness, in linear light; none of the render's body comes through. (carry false: the render's
           own pixels, which showed its relit body as a darker block by the tube's limb)
  out    = source x (1 - alpha) + target x alpha
The new print is the keyframe's own ink (ink "keyframe"): where the keyframe is 4+ levels brighter than its source frame,
a bright stroke (top-hat 15 at 0.03) 6+ px inside the matte, in components whose 90th-percentile gain is 8+ levels, in
label space through the keyframe frame's mapping, dilated 2 px, carried onto each frame through that frame's mapping.
The render's top-hat misses faint ink (a pale copy in shade, a frame the generator drew blank), and the keyframe's body
light can't come through as print. ink "render": the render's top-hat 15 at 0.03, 3+ px inside the change and 6+
levels over the source (the first real run's frosted bottle).
Light print on a body only (white ink): a dark print needs the top-hats and gains turned round.

Ported from the first real run's job script (ink_alpha.py, 2026-10-17), step for step: on the same render it gives
the same pixels."""
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, grey_opening

from imgpost import geometry as geo

DEFAULTS = {
    "ink": "keyframe",      # the new print: the keyframe's own ink ("keyframe") or the render's top-hat ("render")
    "carry": True,          # the target: the print carried onto the source's body (false: the render's pixels)
    "grow": 3,              # px the print is dilated by
    "sigma": 1.5,           # px the alpha is smoothed by
    "near": 14,             # px: garbled print this close to the new print is replaced; further out it stays
    "inner": 3.0,           # px (ink "render"): the new print this far inside the changed pixels
    "gain": 6.0,            # levels (ink "render"): the new print this much brighter than the source
    "edge_band": 4.0,       # px: away from the silhouette's rim (a lit rim isn't print)
    "thresh": 0.03,         # linear: the top-hats' threshold
    "ink_levels": 4.0,      # levels (ink "keyframe"): the keyframe this much brighter than its source frame
    "ink_p90": 8.0,         # levels (ink "keyframe"): a component of ink rises this far somewhere (90th percentile)
    "ink_inset": 6.0,       # px (ink "keyframe"): strokes this far inside the keyframe frame's matte
    "lost_gain": 0.12,      # the lost-print measure: print whose gain over its body is this much lower in the output
    "presence": "match",    # "match": the print's presence (mean lift over its footprint) brought up to the source's
    "presence_target": 1.0,  # x the source's presence (1.25 for thin true copy against a bold garbled print)
    "max_gain": 2.5,        # presence gain at most
    "presence_median": 9,   # frames: the running median over the measured ratios
    "presence_sigma": 2.0,  # frames: the Gaussian after it
    "shutter": 0.5,         # motion blur: the shutter as a fraction of the frame interval (0: none)
    "max_blur": 3.0,        # px: the blur's length at most (6 px smeared a 5 px copy past reading)
    "min_speed": 8.0,       # px/frame: slower isn't a rapid move: no blur
    "sharp_full": 0.9,      # the source's print at this share of the keyframe frame's sharpness or more: no blur
    "sharp_span": 0.25,     # ... the full max_blur at sharp_full - sharp_span
    "sharp_gate": True,     # false: blur by speed alone (over-blurs generated video, which jumps between crisp frames)
}


def settings(spec):
    """render.print_alpha merged over DEFAULTS."""
    c = dict(DEFAULTS, **(spec or {}))
    if c["ink"] not in ("keyframe", "render"):
        raise ValueError(f"render.print_alpha.ink {c['ink']!r}: \"keyframe\" or \"render\"")
    if c["presence"] not in ("off", "match"):
        raise ValueError(f"render.print_alpha.presence {c['presence']!r}: \"off\" or \"match\"")
    return c


def lin_lum(u8):
    x = u8.astype(np.float32) / 255.0
    x = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return x @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def _lin(u8):
    x = u8.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055) * 255.0


def keyframe_ink(kf_u8, src_u8, matte, H, label_shape, c):
    """The keyframe's own ink in label space: (N x 2 label px, their count). kf_u8 / src_u8: the keyframe and its
    source frame (full frames); matte: the product's matte on that frame (bool) or None; H: the mapping the render
    placed the label with on that frame (label px -> frame px); label_shape: (h, w) of the label texture."""
    from scipy.ndimage import map_coordinates
    h, w = label_shape
    g = np.stack(np.meshgrid(np.arange(float(w)), np.arange(float(h))), -1).reshape(-1, 2)
    p = geo.apply_H(H, g)
    K = kf_u8.astype(np.float32).mean(-1)
    S = src_u8.astype(np.float32).mean(-1)
    # ink: a bright stroke on the keyframe (top-hat 15), ink_levels above its source frame, ink_inset px inside the
    # matte (the keyframe build insets the ink 4 px; at the limbs the two decodes differ along the dark edge)
    Lk = lin_lum(kf_u8)
    stroke = (Lk - grey_opening(Lk, size=(15, 15))) > float(c["thresh"])
    if matte is not None:
        stroke &= distance_transform_edt(matte) > float(c["ink_inset"])
    smp = lambda I, o=1: map_coordinates(I, [p[:, 1], p[:, 0]], order=o, mode="nearest")
    gain = (smp(K) - smp(S)).reshape(h, w)
    added = (gain > float(c["ink_levels"])) & (smp(stroke.astype(np.float32), 0) > 0.5).reshape(h, w)
    # letters stand well above the source somewhere (90th percentile 13-17 levels on a pale tube); a
    # stroke-like sliver the keyframe build left at a limb (4.7) is not ink
    lab, n = ndi.label(added, np.ones((3, 3)))
    if n:
        p90 = np.array([np.percentile(gain[lab == i], 90) for i in range(1, n + 1)])
        added = np.isin(lab, 1 + np.nonzero(p90 >= float(c["ink_p90"]))[0])
    ink = binary_dilation(added, iterations=2)
    return g[ink.ravel()], int(ink.sum())


def ink_mask(H, pts, shape):
    """The keyframe's ink (label px) put onto a frame through its mapping: a bool mask of the frame's size."""
    p = np.round(geo.apply_H(H, pts)).astype(int)
    ok = (p[:, 0] >= 0) & (p[:, 0] < shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < shape[0])
    m = np.zeros(shape, bool)
    m[p[ok, 1], p[ok, 0]] = True
    return ndi.binary_closing(binary_dilation(m, iterations=1), iterations=1)


def _body(lin, inside, size):
    """The body under light strokes: a grey opening per channel that the outside of the matte can't pull down."""
    out = []
    for ch in range(3):
        x = np.where(inside, lin[..., ch], 1.0)
        out.append(gaussian_filter(grey_opening(x, size=(size, size)), 1.5))
    return np.stack(out, -1)


def transfer(s, r, inside, gain=1.0, blur=None):
    """The print carried onto the source's own body: the render's ink as a gain over the render's own body (light
    print), on the source with its garbled strokes filled from the source's body around them. Nothing of the render's
    body comes through, so no patch outline can show (the render's relight ran 15-20 levels darker than the source by
    a tube's limb, and a pasted patch around the copy showed as a darker block). gain: the print's presence matched to
    the source's (presence); blur: (direction, length px) of a line blur along the label's own motion."""
    sl, rl = _lin(s), _lin(r)
    rb, bs = _body(rl, inside, 15), _body(sl, inside, 13)
    # the source's body: light strokes clamped to the body around them (3 % over it), so garbled print too faint to be
    # found (a ghost word over the serum's copy) goes as well; grain and shading below it stay
    sb = np.minimum(sl, bs * 1.03)
    # the ink: a soft matte of the render's print (its brightness over its body, against the strokes' own peak), in the
    # render's ink colour scaled by how much brighter the source's body is than the render's there. A per-channel gain
    # or an added excess shifted the hue (white on the serum's redder limb read cool)
    lum = lambda x: x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    Lr, Lb = lum(rl), lum(rb)
    peak = ndi.grey_dilation(Lr, size=(7, 7))
    cv = np.clip((Lr - Lb) / np.maximum(peak - Lb, 1e-4), 0, 1) * np.clip((peak - Lb - 0.004) / 0.008, 0, 1)
    scale = gaussian_filter(np.clip(lum(bs) / np.maximum(Lb, 1e-4), 0.5, 2.0), 2.0)[..., None]
    # the ink's own colour, unmixed from the render's pixel (render = body (1 - c) + ink c): a thin stroke's edge pixel
    # carries the ink at partial coverage, and taking the pixel as the ink colour weakened it twice (a thin copy lifted
    # 0.7x the render's)
    ink = np.clip(rb + (rl - rb) / np.maximum(cv, 0.05)[..., None], 0, 1) * scale
    if gain != 1.0:     # the print's presence matched to the source's (thin true copy against a bold garbled one)
        ink = np.clip(bs + gain * (ink - bs), 0, 1)
    if blur is not None:                            # motion blur along the label's own motion (shutter x speed)
        from .optics import line_blur
        d, L = blur
        cb = line_blur(cv, d, L)
        ink = line_blur(ink * cv[..., None], d, L) / np.maximum(cb, 1e-4)[..., None]
        cv = cb
    cv = cv[..., None]
    return _srgb(sb * (1 - cv) + ink * cv)


def presence(s, r, new, key):
    """The print's presence: its mean lift over the body (display levels, top-hat 15) across a footprint wide enough
    for a bold garbled print as well (the true print dilated 4 px), in the source and in the render. Thin true copy
    reads fainter than a bold garbled print at the same peak brightness. -> source / render, or None"""
    from scipy.ndimage import binary_erosion
    fp = binary_dilation(new, iterations=4) & binary_erosion(key, iterations=4)
    if fp.sum() < 200:
        return None
    w = np.array([0.2126, 0.7152, 0.0722], np.float32)

    def lift(u8):
        L = u8.astype(np.float32) @ w
        return np.maximum(L - grey_opening(L, size=(15, 15)), 0)[fp].mean()
    a, b = float(lift(s)), float(lift(r))
    return a / b if b > 0.3 else None


def _crop(ch):
    ys, xs = np.nonzero(ch)
    return (max(ys.min() - 24, 0), min(ys.max() + 25, ch.shape[0]), max(xs.min() - 24, 0), min(xs.max() + 25, ch.shape[1]))


def _new_print(s, r, chc, keyc, c, inkc):
    """The new print in a crop: the render's top-hat 15 (ink "render") or the keyframe's ink. -> (new, rim, differs)"""
    thresh = float(c["thresh"])
    Lr = lin_lum(r)
    new = (Lr - grey_opening(Lr, size=(15, 15))) > thresh
    rim = distance_transform_edt(keyc) > float(c["edge_band"])
    differs = gaussian_filter(np.abs(s.astype(np.float32) - r.astype(np.float32)).max(-1), 0.7) > 3.0
    new &= rim & differs
    # the new print, not the render's own edges or the keyframe's body light: at least `inner` px inside the changed
    # pixels (a band end rendered hard reads as a 1 px bright line on its edge) and brighter than the source by `gain`
    # levels (light print; a relit highlight of the keyframe's body changes by a few levels only)
    new &= distance_transform_edt(chc) > float(c["inner"])
    new &= gaussian_filter((r.astype(np.float32) - s.astype(np.float32)).mean(-1), 0.7) > float(c["gain"])
    if inkc is not None:
        # the keyframe's own ink is the new print, whatever its contrast here (white on a pale tube in shade, faded by
        # the limb feather, a frame the generator drew blank: the render's top-hat misses those letters)
        new = inkc & rim & (distance_transform_edt(chc) > 1)
    return new, rim, differs


def measure_presence(src, ref, ch, key, c, ink=None):
    """The print's presence on one frame, source over render (presence), or None."""
    y0, y1, x0, x1 = _crop(ch)
    s, r = src[y0:y1, x0:x1], ref[y0:y1, x0:x1]
    new, _, _ = _new_print(s, r, ch[y0:y1, x0:x1], key[y0:y1, x0:x1], c, None if ink is None else ink[y0:y1, x0:x1])
    return presence(s, r, new, key[y0:y1, x0:x1])


def presence_gains(raw, frames, c):
    """Per-frame presence gains from the measured ratios {clip frame: source / render}: a running median over
    presence_median frames, a Gaussian over presence_sigma, interpolated onto every frame, times presence_target,
    clamped to [1, max_gain] (never fainter than the render). -> {clip frame: gain}"""
    from scipy.ndimage import median_filter
    if not raw:
        return {}
    tk = np.array(sorted(raw))
    gv = median_filter(np.array([raw[t] for t in tk]), size=int(c["presence_median"]), mode="nearest")
    gv = gaussian_filter(gv, float(c["presence_sigma"]), mode="nearest")
    return {t: float(np.clip(float(c["presence_target"]) * np.interp(t, tk, gv), 1.0, float(c["max_gain"])))
            for t in frames}


def edge_strength(frame, M, rim=6):
    """The print's edge strength on a frame (check_sharp): the 95th percentile of the linear luminance's gradient over
    the strokes (top-hat 15 at 0.03, inside the matte eroded `rim` px) and 2 px round them, over the strokes' median
    contrast, so a fainter frame isn't read as a softer one. None with under 100 stroke px."""
    from scipy.ndimage import binary_erosion, sobel
    ys, xs = np.nonzero(M)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    L = lin_lum(frame[y0:y1, x0:x1])
    th = L - grey_opening(L, size=(15, 15))
    Mi = binary_erosion(M[y0:y1, x0:x1], iterations=rim)
    ink = (th > 0.03) & Mi
    if ink.sum() < 100:
        return None
    g = np.hypot(sobel(L, 0), sobel(L, 1)) / 8.0
    near = binary_erosion(Mi, iterations=1) & ~binary_erosion(~ink, iterations=2)   # the strokes and 2 px around them
    return float(np.percentile(g[near], 95) / max(np.median(th[ink]), 1e-4))


def label_motion(P, inl):
    """The label's own motion per frame, px per frame: the median over the label's grid points (P {frame: N x 2}, inl
    the points inside the label) of the centred difference. -> {frame: [vx, vy]} (rounded to 1/1000 px)"""
    tl = sorted(P)
    mot = {}
    for i, t in enumerate(tl):
        a_, b_ = tl[max(i - 1, 0)], tl[min(i + 1, len(tl) - 1)]
        if b_ > a_:
            v = np.median((P[b_][inl] - P[a_][inl]) / (b_ - a_), axis=0)
            mot[t] = [round(float(v[0]), 3), round(float(v[1]), 3)]
    return mot


def motion_blur(v, sharp, c):
    """The line blur for one frame: v the label's motion (px/frame), sharp the source's print sharpness against the
    keyframe frame's (edge_strength ratio) or None. Length shutter x speed, at most max_blur, and only as soft as the
    source is: none at sharp_full (0.9) of the keyframe's sharpness or more, max_blur at sharp_full - sharp_span (0.65)
    (generated video jumps up to 85 px between crisp frames). -> (direction, length px) or None (under 1 px, or under
    min_speed px/frame)"""
    v = np.array(v, float)
    sp = float(np.hypot(*v))
    L = min(float(c["shutter"]) * sp, float(c["max_blur"]))
    if sharp is not False:
        s_ = 1.0 if sharp is None else sharp
        L = min(L, float(c["max_blur"]) * float(np.clip((float(c["sharp_full"]) - s_) / float(c["sharp_span"]), 0, 1)))
    return (v / max(sp, 1e-6), L) if L >= 1.0 and sp >= float(c["min_speed"]) else None


def composite(src, ref, ch, key, c, ink=None, gain=1.0, blur=None):
    """One frame. src: the source frame; ref: the render (uint8, full frames); ch: the render's changed pixels; key: the
    product's matte (bool); ink: the keyframe's ink on this frame (ink_mask) or None (the render's top-hat); gain: the
    print's presence gain (presence_gains); blur: motion_blur's (direction, length) or None. -> (output uint8,
    {"alpha_px", "print_px", "lost_print_px", "ink_px"})"""
    y0, y1, x0, x1 = _crop(ch)
    s, r, chc = src[y0:y1, x0:x1], ref[y0:y1, x0:x1], ch[y0:y1, x0:x1]
    Ls, Lr = lin_lum(s), lin_lum(r)
    old = (Ls - grey_opening(Ls, size=(7, 7))) > float(c["thresh"])
    new, rim, differs = _new_print(s, r, chc, key[y0:y1, x0:x1], c, None if ink is None else ink[y0:y1, x0:x1])
    # the garbled print sits around the true print; bright specks further away are sparkles and highlights: kept
    old &= rim & differs & (distance_transform_edt(~new) <= float(c["near"]))
    # garbled print to remove is brighter in the source than in the render; a highlight the render keeps (or brightens:
    # the keyframe's own body light) is not
    old &= gaussian_filter((s.astype(np.float32) - r.astype(np.float32)).mean(-1), 0.7) > 3.0
    # and removing it puts the body back: the render there is no darker than the source's body around the stroke (a
    # sparkle by the limb over the render's dark limb line came out as a dark dot)
    sm = s.astype(np.float32).mean(-1)
    old &= r.astype(np.float32).mean(-1) >= grey_opening(sm, size=(7, 7)) - 6.0
    pr = old | new
    extra = int(np.ceil(0.5 * blur[1])) if blur is not None else 0       # room for the blurred print
    reg = binary_dilation(pr, iterations=int(c["grow"]) + extra) & chc
    a = np.clip(gaussian_filter(reg.astype(np.float32), float(c["sigma"])) * 1.5, 0, 1) * chc
    tgt = r.astype(np.float32)
    if c["carry"]:
        tgt = transfer(s, r, key[y0:y1, x0:x1], gain, blur).astype(np.float32)
    o = np.clip(s * (1 - a[..., None]) + tgt * a[..., None] + 0.5, 0, 255).astype(np.uint8)
    out = src.copy()
    out[y0:y1, x0:x1] = o
    # print the render has and the output lacks, as gains over each image's own body (the carried print sits on the
    # source's body, which can be darker than the render's): light print, so the output darker than the render. A
    # motion blur lowers the gain on purpose: not measured there
    Lo = lin_lum(o)
    pl = ink[y0:y1, x0:x1] if ink is not None else (Lr - grey_opening(Lr, size=(15, 15))) > 0.012
    gr = Lr / np.maximum(grey_opening(Lr, size=(15, 15)), 1e-4)
    go = Lo / np.maximum(grey_opening(Lo, size=(15, 15)), 1e-4)
    lost = pl & chc & (gr - go > float(c["lost_gain"])) if blur is None else np.zeros_like(chc)
    return out, {"alpha_px": int((a > 0.02).sum()), "print_px": int(pr.sum()), "lost_print_px": int(lost.sum()),
                 "ink_px": int((pl & chc).sum())}
