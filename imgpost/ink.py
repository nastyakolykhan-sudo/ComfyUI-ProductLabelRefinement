"""Print straight on a coloured body: `matte: {"type": "ink", "polarity": "light" | "dark", "print_box": [...]}`.

The paper-based ink matte (run.py) measures coverage down from the paper's white, so it can't lift white print off a
coral or pink bottle (pitfall 92), and a label fit needs a physical label's edges, which dark print straight on a jar's
body doesn't have (pitfall 97). Here the body stays the frame's and only print changes:
  1. cover   the packshot's print as coverage against its own body light, inside the reference outline (the band) and
             the print box only. Light print: above the body (a grey opening wider than its strokes takes the print off
             the body); dark print: below it (a grey closing). A frosted body's own highlights (the band under a
             shoulder, limb rims) and a jar's seams and lid edge are thin structures of either polarity too (pitfall
             93): the print box keeps them out. Warped into the work box through the job's mapping, supersampled.
  2. garbled the frame's own print (brighter or darker than its body: a top-hat) near the true print's footprint and
             inset from the matte, filled harmonically from the body around it, with the frame's grain.
  3. ink     the true print in the frame's own print colour, lit by its body: ink = body x k per channel, k from the
             generated print's most-covered pixels (above 1 for light print, below for dark), composited inside the
             matte (the scene's product matte, the job's old_silhouette) and nowhere else.
Light print reproduces the first real job's approved keyframes (a frosted serum and a squeezed tube, 2026-10-16), made
by a job-level prototype, within 1 level. Dark print was added for a pink jar seen from above (2026-10-18).
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
    "gen_glow": None,     # scene px (light print): the fill also takes the garbled strokes' soft glow this far out
    "glow_thresh": 0.008,  # linear luminance above the body light (off every light stroke) that counts as glow
    "ink_pct": 90,        # percentile of frame / body over the garbled print's core that sets the ink's colour
    "seed": 7,            # the grain over the fill
    "gen_rel": None,      # ... and this share of the body light above (light) or below (dark) it, too
    "gen_grow": False,    # whole garbled strokes: every one that touches the zone above or the print box
    "register": False,    # snap the true print onto the frame's own print (an affine over the fit, scene px)
    "reg_reach": "auto",  # scene px: how far the frame's print is looked for round the true print (auto: a tenth
                          # of the body's width)
    "reg_scale": 0.15,    # the scales tried: 1 +- this
    "reg_min": 0.3,       # the registered print maps' correlation needed to keep it
    "reg_bow": False,     # ... with a bow profile over it, fitted to the generated print (loose: off by default)
    "bow_from": None,     # "print": the body's bow from the generated print's own lines (line_bows), unless the job's
                          # bow was given; kept only where the print registers better with it (pitfall 100)
    "reg_model": "affine",  # "scale": stretched along and across the body's axis only (no turn or shear: the print's
                            # lines stay parallel to the body's level edges); "affine": any
    "reg_aspect": None,   # the registration's two scales at most this share apart (text only: 0.03; the letters keep
                          # their shape, a white bottle's came out 8.5 % wider and 5 % shorter, pitfall 112)
    "sharpen": None,      # "auto": the warped print's edges sharpened to the frame's own print's
    "sharpen_max": 1.5,   # ... by an unsharp mask of at most this amount (more rings and aliases)
    "inks": None,         # "packshot": each ink in its packshot colour (a white bottle's black text and yellow bar),
                          # none darker than the frame's own darkest; None: one ink, the frame's (k)
    "text": False,        # text only (pitfall 111): letters only, line by line onto the generated lines, each in the
                          # generated line's own ink; the generated rules, bands, edges and lone marks stay
}
# dark print straight on a jar or a bottle (2026-10-18, pitfall 97): the frame's strokes are as wide as the packshot's,
# often wider than 7 px (a wordmark), and a closing narrower than a stroke leaves its middle unfilled, so the frame's
# closing follows its widest garbled stroke; a jar's body shades within a closing that wide, so garbled print is also
# 15 % darker than its body; a fit to the silhouette of a jar seen from above places the print tens of px off the
# generated print, so the print is registered onto it, and garbled strokes are taken whole; a 1,024 px packshot's
# print came out softer than the generator's (edge blur 0.95 px against 0.62), so it is sharpened to the frame's
DARK = dict(DEFAULTS, frame_open="auto", gen_rel=0.15, gen_grow=True, register=True, reg_model="scale",
            bow_from="print", sharpen="auto", inks="packshot")
# light print, drafted from the frame, the mask and the packshot alone (pitfall 105): a fit to the silhouette put a
# serum's print 30 px off the generated print, and the garbled print, filled only near the true print's footprint,
# showed beside it on every frame. It is registered onto the generated print as dark print is (scaled along and across
# the body's axis only); a print already on it moves by the registration's own residual. A generated frame's light print
# fades into its body over several px (pitfall 108): the fill also takes that glow
LIGHT = dict(DEFAULTS, register=True, reg_model="scale", gen_glow=6)
POLARITIES = ("light", "dark")
TEXT_ASPECT = 0.03     # text only: the registration's scales at most this share apart (reg_aspect)


class InkError(ValueError):
    pass


def on_body(matte):
    """True for print straight on the body: matte ink with polarity light (lighter than the body) or dark, or mixed
    (both: text only, each line in its own polarity)."""
    return isinstance(matte, dict) and matte.get("type") == "ink" and (
        matte.get("polarity") in POLARITIES or (matte.get("polarity") == "mixed" and bool(matte.get("text"))))


def _sub(cfg, pol):
    """A mixed print's config for one polarity's pass (text only)."""
    base = dict(DARK if pol == "dark" else LIGHT)
    base.update({k: v for k, v in cfg.items() if k in DEFAULTS and k not in ("frame_open", "gen_rel", "gen_grow",
                                                                             "register", "reg_model", "bow_from",
                                                                             "sharpen", "inks", "gen_glow")})
    base.update(polarity=pol, print_box=cfg.get("print_box"), label=cfg.get("label"), text=True, inks=None,
                gen_grow=False)
    if base.get("reg_aspect") is None:
        base["reg_aspect"] = TEXT_ASPECT
    f = cfg.get("_px_scale") or 1
    for k in ("frame_open", "gen_glow"):              # the pass's own defaults, at the work scale too
        if f != 1 and isinstance(base.get(k), int) and not isinstance(base.get(k), bool):
            base[k] = int(round(base[k] * f))
    base["_px_scale"] = f
    return base


# the route's first name, from when it took light print only: the pack's flow nodes (Print Type) call it
light = on_body


SCENE_PX = ("frame_open", "gen_inset", "gen_dilate", "gen_near", "gen_glow", "reg_reach")   # scene px reaches


def config(matte):
    cfg = dict(DARK if matte.get("polarity") == "dark" else LIGHT)
    cfg.update({k: v for k, v in matte.items() if k in DEFAULTS})
    f = matte.get("px_scale") or 1
    if f != 1:
        # the frame resized up before the fix (work scale, pitfall 113): the garbled print's reaches with it (the old
        # letters' fill missed their 4x strokes: pale halos round the new letters), the warp's softness not (the new
        # letters are drawn crisp at the work size)
        for k in SCENE_PX:
            if isinstance(cfg.get(k), (int, float)) and not isinstance(cfg.get(k), bool):
                cfg[k] = int(round(cfg[k] * f)) if isinstance(cfg[k], int) else cfg[k] * f
    cfg["_px_scale"] = f
    cfg["print_box"] = matte.get("print_box")
    cfg["polarity"] = matte.get("polarity", "light")
    cfg["label"] = matte.get("label")
    if cfg.get("text"):
        # the generated print's own inks, line by line, and its own blocks: no packshot inks, no whole blocks grown;
        # the letters keep their shape (the lines are placed one by one after the registration)
        cfg.update(inks=None, gen_grow=False)
        if cfg.get("reg_aspect") is None:
            cfg["reg_aspect"] = TEXT_ASPECT
    return cfg


def _extend(L, inside):
    """L outside `inside` replaced by its nearest inside value: a grey closing or opening near the edge then reads
    only the inside."""
    iy, ix = ndi.distance_transform_edt(~inside, return_distances=False, return_indices=True)
    return L[iy, ix]


def _tophat(L, inside, size, dark):
    """How far each pixel stands out of its body: light print above a grey opening, dark print below its body light.
    The dark body is the local mean of the pixels that aren't print: the print found as 10 % below a closing, and a
    2 px ring round it left out too (a packshot's sharpening halos: a closing rides on them, and read a jar's plain body
    between its lines of text as 13 % ink). -> (top-hat, body)"""
    if dark:
        body, _ = _dark_body(L, inside, size)
        return body - L, body
    body = ndi.grey_opening(np.where(inside, L, ndi.grey_dilation(L * inside, size=size)), size=(size, size))
    return L - body, body


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


def box_mask(shape, cfg):
    """The print box on the packshot (bool), or None without one."""
    if not cfg.get("print_box"):
        return None
    bx0, by0, bx1, by1 = [int(round(v)) for v in cfg["print_box"]]
    m = np.zeros(shape[:2], bool)
    m[by0:by1, bx0:bx1] = True
    return m


def _dark_body(L, inside, size):
    """The body light under dark print, and the print's strokes with a 2 px rim (bool): the strokes 25 % below a
    closing, the body the local mean of the rest; then the strokes again, 25 % below that body (between dense lines of
    text a closing sits on the halos), and the body again without the halos. Print's core is near 100 % below its
    body; a packshot body's own soft shading reached 12 % under the wordmark. -> (body, strokes)"""
    ext = _extend(L, inside)
    body = ndi.grey_closing(ext, size=(size, size))
    halo = np.zeros(L.shape, bool)
    for _ in range(2):
        strokes = ndi.binary_dilation(body - ext > 0.25 * body, iterations=2)
        w = (inside & ~strokes & ~halo).astype(float)
        num, den = ndi.gaussian_filter(ext * w, size / 2), ndi.gaussian_filter(w, size / 2)
        body = np.where(den > 0.02, num / np.maximum(den, 1e-6), body)
        # the sharpening halos round the print, brighter than the body: left out of the second pass (in a coloured
        # body's weaker channels they lifted its mean between dense lines of text by a quarter)
        halo = ndi.binary_dilation(strokes, iterations=3) & (ext > 1.04 * body)
    return body, strokes


def _longest_run(m):
    """The longest run of True along each row of a 2-D bool array."""
    best = np.zeros(m.shape[0], int)
    cur = np.zeros(m.shape[0], int)
    for j in range(m.shape[1]):
        cur = np.where(m[:, j], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def _trim_edge_lines(cov, box, reach, share=0.75):
    """A seam, the lid's edge, a limb or a label's own edge the print box takes in near its edge: rows (columns)
    within `reach` px of the box's top and bottom (sides) holding one unbroken run of print over `share` of the box's
    width (height) are cleared (a jar's pink-to-glass seam right under its last line of text; a bottle label's edge
    2 px inside the box: both run its whole width or height). A line of text is broken into letters (counting its
    covered share instead cut a last line's letters in half), and a rule over a wordmark spans 60 % of it.
    In place -> px cleared"""
    ys, xs = np.nonzero(box)
    if not len(ys):
        return 0
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    sub = cov[y0:y1 + 1, x0:x1 + 1]
    before = int((sub > 0.5).sum())
    for axis in (1, 0):
        m = sub > 0.3
        run = _longest_run(m if axis == 1 else m.T) / float(m.shape[1] if axis == 1 else m.shape[0])
        n = len(run)
        for i in list(range(min(reach, n))) + list(range(max(0, n - reach), n)):
            if run[i] > share:
                if axis == 1:
                    sub[i, :] = 0
                else:
                    sub[:, i] = 0
    return before - int((sub > 0.5).sum())


def _dark_body_rgb(R, inside, size):
    """_dark_body per channel: the strokes found in any channel (a yellow ink is dark only in blue), the body light per
    channel the local mean of the rest. -> (body h x w x 3, strokes)"""
    strokes = np.zeros(R.shape[:2], bool)
    for c in range(3):
        strokes |= _dark_body(R[..., c], inside, size)[1]
    w = (inside & ~strokes).astype(float)
    den = ndi.gaussian_filter(w, size / 2)
    body = np.stack([ndi.gaussian_filter(_extend(R[..., c], inside) * w, size / 2) for c in range(3)], -1) / \
        np.maximum(den, 1e-6)[..., None]
    return np.where(den[..., None] > 0.02, body, 1.0), strokes


def stroke_open(Lr, band, box, dark=False, inner=False):
    """An opening wider than the packshot's widest print stroke (ref px, odd): 1.5 x the widest stroke, at least 7.
    The strokes: luminance above a wide opening (an eighth of the band's width; below a closing for dark print), over
    half its 99.5th percentile, inside the print box; components of 20 px or more. -> (size, widest stroke px)"""
    xs = np.nonzero(band.any(0))[0]
    big = int(max(15, (xs.max() - xs.min()) // 8)) | 1
    return _widest(_tophat(Lr, band, big, dark)[0] * box, box & band, big, inner)


def _widest(top, sel, big, inner=False):
    """-> (1.5 x the widest stroke in top-hat `top` over `sel`, odd, at least 7; the widest stroke px)"""
    if not sel.any() or np.percentile(top[sel], 99.5) <= 0:
        return 2 * (big // 2) + 1, 0.0
    ink = top > 0.5 * np.percentile(top[sel], 99.5)
    if inner:
        ink &= sel                     # text only: the band's own marks (a letter touching dark glass past it isn't wide)
    lab, n = ndi.label(ink)
    if n == 0:
        return 2 * (big // 2) + 1, 0.0
    idx = np.arange(1, n + 1)
    half = ndi.maximum(ndi.distance_transform_edt(ink), lab, idx)
    keep = ndi.sum(ink, lab, idx) >= 20
    if inner:
        # text only: only letters' strokes (marks in lines of text, textline): a label's shaded side, its border, a
        # band or a part of the product read as wide dark marks too (a dropper bottle's shaded label: 59 px)
        from . import textline as tl
        t = tl.text_lines(ink.astype(float), sel, thresh=0.5, sharp=True)
        letters = t["text"]
        keep &= ndi.maximum(letters, lab, idx) > 0
    half = half[keep]
    widest = 2.0 * float(half.max()) if len(half) else 0.0
    return max(7, int(np.ceil(1.5 * widest)) | 1), widest


def ref_cover(ref_u8, refmask, cfg):
    """1. The packshot's print as coverage (reference px, 0..1) -> (cover, band, report)."""
    band = band_mask(refmask, ref_u8.shape)
    if not band.any():
        raise InkError(f"matte ink ({cfg['polarity']}): the reference outline covers no packshot pixels")
    keep = np.ones(band.shape, bool)
    if cfg.get("print_box"):
        bx0, by0, bx1, by1 = [int(round(v)) for v in cfg["print_box"]]
        keep = np.zeros(band.shape, bool)
        keep[by0:by1, bx0:bx1] = True
    rep = {}
    Lr = im.lum(im.srgb_to_lin(im.to_float(ref_u8)))
    dark = cfg["polarity"] == "dark"
    inks = dark and cfg.get("inks") == "packshot"
    s = cfg["open"]
    if s == "auto" or dark:
        if inks:
            # several inks: the widest stroke in any channel (a yellow bar is a stroke in blue only)
            Rl = im.srgb_to_lin(im.to_float(ref_u8))
            s_a, widest = max(stroke_open(Rl[..., c], band, keep, True) for c in range(3))
        else:
            s_a, widest = stroke_open(Lr, band, keep, dark, inner=bool(cfg.get("text")))
    if s == "auto":
        s = s_a
        rep["open_auto"] = {"size": s, "widest_stroke_px": round(widest, 1)}
    if dark:
        rep["widest_stroke_px"] = round(widest, 1)       # register() sizes the frame's closing from it
    s = int(s)
    # work on the band's box plus 3 openings: every step below reaches at most 2.5 openings, so this is exact
    ys, xs = np.nonzero(band)
    p = 3 * s + 8
    y0, y1 = max(0, ys.min() - p), min(band.shape[0], ys.max() + 1 + p)
    x0, x1 = max(0, xs.min() - p), min(band.shape[1], xs.max() + 1 + p)
    L, B = Lr[y0:y1, x0:x1], band[y0:y1, x0:x1]
    K = keep[y0:y1, x0:x1]
    if dark:
        # the print as coverage below the body light, down to the print's own black (its darkest pixels in the box:
        # a silver lid or a jar's glass base can be darker than its print, and stay out), on its strokes only: a body
        # darkening toward a seam, the lid or a limb faster than the mean follows stays body
        body, strokes = _dark_body(L, B, s)
        black = float(np.percentile(L[B & K] if (B & K).any() else L[B], 0.2))
        cov = np.clip((body - L) / np.maximum(body - black, 1e-3), 0, 1) * strokes
        if cfg.get("print_box"):
            # the body's own straight lines (a label's edge, a seam, a lid's rim) run on past the print box; print
            # lies inside it (a rule over a wordmark too): a long level or upright run of dark structure that reaches
            # a closing past the box is cleared where it enters it (a toner label's edge 2 px inside the box showed
            # as a silver line down the bottle's side)
            # on the dark pixels themselves, not the strokes' 2 px rim, which joins a line of letters into a run
            n_run = max(40, 2 * s)
            dark_px = B & (body - L > 0.25 * body)
            lines = ndi.binary_opening(dark_px, structure=np.ones((n_run, 1), bool)) | \
                ndi.binary_opening(dark_px, structure=np.ones((1, n_run), bool))
            lab, n = ndi.label(lines)
            if n:
                far = lines & ~ndi.binary_dilation(K, iterations=s)
                struct = np.isin(lab, np.unique(lab[far & (lab > 0)]))
                # with its light anti-aliased rim, but not into the dark pixels beside it (a line of text's feet)
                struct |= ndi.binary_dilation(struct, iterations=2) & ~dark_px
                rep["structure_px"] = int((struct & K & (cov > 0.5)).sum())
                cov[struct] = 0
        if inks:
            # another ink (a yellow bar absorbs nine tenths of the blue and a quarter of the luminance): its
            # coverage is its absorption in its strongest channel, down to that channel's black in the box
            _, Cm, oth, _ = _channel_maps(im.srgb_to_lin(im.to_float(ref_u8))[y0:y1, x0:x1], B, K, s)
            if (oth & B & K).any():
                cov = np.where(oth, Cm, cov)
                rep["other_ink_ref_px"] = int((oth & B & K & (Cm > 0.5)).sum())
    else:
        body = ndi.gaussian_filter(ndi.grey_opening(np.where(B, L, ndi.grey_dilation(L * B, size=s)), size=(s, s)), s / 4)
        white = float(np.percentile(L[B], 99.8))
        cov = np.clip((L - body) / np.maximum(white - body, 1e-3), 0, 1)
    cov[cov < cfg["floor"]] = 0
    cov *= B
    cov *= K
    if dark and cfg.get("print_box"):
        rep["edge_lines_px"] = _trim_edge_lines(cov, K & B, s)
    if dark:
        # print always has a core; a patch that never reaches 40 % is the body's shading (a lid's shadow in the box's
        # corner, sharpened into a mark)
        lab, n = ndi.label(cov > 0)
        if n:
            faint = np.isin(lab, np.nonzero(ndi.maximum(cov, lab, np.arange(1, n + 1)) < 0.4)[0] + 1)
            rep["faint_px"] = int(faint.sum())
            cov[faint] = 0
    full = np.zeros(band.shape)
    full[y0:y1, x0:x1] = cov
    rep.update({"open": s, "ref_black_lin": round(black, 4)} if dark else {"open": s, "ref_white_lin": round(white, 4)})
    rep["print_px"] = int((full > 0.5).sum())
    return full, band, rep


def warp_cover(cov_r, band, H, roi, ss=4, prefilter=0.6, blur=0.4, extra=None):
    """2. The coverage and the band warped into the work box (ss x ss samples per pixel) -> (cover, in_band), and
    with `extra` ({name: reference-px map, 2-D or with channels}: the print box, the inks' colours) those too, as the
    mean of each pixel's samples -> (cover, in_band, {name: map})."""
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
    if extra is not None:
        out = {}
        for name, m in extra.items():
            m = np.asarray(m, float)
            ch = [m] if m.ndim == 2 else [m[..., i] for i in range(m.shape[2])]
            ws = [down(ndi.map_coordinates(z, [v, u], order=1, mode="nearest") * ok) for z in ch]
            out[name] = ws[0] if m.ndim == 2 else np.stack(ws, -1)
        return cov, down(b), out
    return cov, down(b)


class Corrected:
    """A fitted mapping with a correction laid over it in scene px (register()): an affine A, and a bow along the
    body's axis (geometry.Bow's, one number) -> scene -> reference through the correction's inverse, then the
    mapping's."""

    def __init__(self, H, A, bow=None):
        from .geometry import Bow
        self.H, self.A = H, np.asarray(A, float)
        self.Ai = np.linalg.inv(self.A)
        self.bow = Bow(self.A, bow["centre"], bow["axis"], bow["half_widths"], bow["b"]) if bow and bow.get("b") else None

    def forward(self, pts):
        """reference px -> scene px: the mapping's, then the correction (text only pulls the frame's print back)."""
        from .geometry import apply_H
        q = apply_H(self.H, pts)
        if self.bow is not None:
            return self.bow.forward(q)
        return q @ self.A[:2, :2].T + self.A[:2, 2]

    def inverse(self, X, Y):
        if self.bow is not None:
            x, y, _ = self.bow.inverse(X, Y)
        else:
            Ai = self.Ai
            x = Ai[0, 0] * X + Ai[0, 1] * Y + Ai[0, 2]
            y = Ai[1, 0] * X + Ai[1, 1] * Y + Ai[1, 2]
        if hasattr(self.H, "inverse"):
            return self.H.inverse(x, y)
        Hi = np.linalg.inv(self.H)
        d = Hi[2, 0] * x + Hi[2, 1] * y + Hi[2, 2]
        return (Hi[0, 0] * x + Hi[0, 1] * y + Hi[0, 2]) / d, (Hi[1, 0] * x + Hi[1, 1] * y + Hi[1, 2]) / d, None


def _channel_maps(Rl, B, K, s):
    """Per channel over a crop: the body light, the strokes in any channel, the transmittance T against the body, the
    coverage in the strongest channel (down to each channel's black in the box K) and the other inks' patches.
    -> (T, Cm, other, cosine to the main ink)"""
    bodyc, strokes_c = _dark_body_rgb(Rl, B, s)
    sel = B & K if (B & K).any() else B
    blk = np.array([np.percentile(Rl[..., c][sel], 0.2) for c in range(3)])
    Cm = np.clip((bodyc - Rl) / np.maximum(bodyc - blk, 1e-3), 0, 1).max(-1) * strokes_c
    T = np.clip(Rl / np.maximum(bodyc, 1e-4), 0, 1)
    oth, cos = _other_inks(T, Cm)
    return T, Cm, oth, cos


def _other_inks(T, C):
    """Patches of another ink than the main one, from the transmittance T (h x w x 3) against the body and the
    coverage C in the strongest channel. The main ink: the hue of absorption of the darkest well-covered pixels.
    Another ink is a patch, not a pixel: 12 or more pixels covered 0.55 or more (a yellow bar 0.9, a 12 px yellow disc
    round a badge's letters 0.6) whose absorption's hue is off the main ink's (cosine under 0.97, median under 0.9)
    and that has a colour of its own (median chroma 0.25: a yellow's is 0.85), with its 2 px rim. A black letter's JPEG
    colour fringes and a coloured body read darker between dense lines of text (the halos lift its mean) stay out.
    -> (bool mask, hue cosine to the main ink)"""
    A = 1.0 - T
    An = A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-6)
    top = C > 0.9 * C.max() if (C > 0.9).sum() < 20 else C > 0.9
    dark = top & (im.lum(T) <= np.percentile(im.lum(T)[top], 20))
    main = np.median(An[dark], 0)
    main /= max(np.linalg.norm(main), 1e-6)
    cos = (An * main).sum(-1)
    cand = (C >= 0.55) & (cos < 0.97)
    lab, n = ndi.label(cand)
    oth = np.zeros(C.shape, bool)
    if n:
        idx = np.arange(1, n + 1)
        chroma = T.max(-1) - T.min(-1)
        big = (ndi.sum(cand, lab, idx) >= 12) & (ndi.median(cos, lab, idx) < 0.9) & \
            (ndi.median(chroma, lab, idx) >= 0.25)
        oth = np.isin(lab, idx[big])
        oth = ndi.binary_dilation(oth, iterations=2) & (C > 0) & (cos < 0.97) | oth
    return oth, cos


def ref_inks(ref_u8, band, cov_r, cfg, s):
    """Each dark ink's own colour on the packshot (a white bottle's black text and its yellow bar), as transmittance
    against the packshot's body light per channel. -> H x W x 5 (linear) or None:
      0-2  the ink's colour: one per connected patch of print (the median over its well-covered pixels, so a foil's
           studio-light gradient goes: an ink is a flat colour), spread to its surroundings (nearest);
      3    1 for the main ink (the darkest's hue of absorption), 0 for another (_other_inks: a patch with a hue and
           a colour of its own, and its rim). The main ink takes the frame's own ink colour (k) and its coverage as
           alpha, since a thin stroke's packshot colour is mixed with the body (anti-aliased) and reads greyer than
           the generator's black;
      4    1 / the patch's peak coverage for another ink: a light yellow absorbs 60 % of the blue at most, and is
           still a solid ink (its alpha 1), 1 for the main ink.
    s: ref_cover's closing (its report's "open")."""
    if (cov_r > 0.5).sum() < 20:
        return None
    Rl = im.srgb_to_lin(im.to_float(ref_u8))
    s = int(s)
    # ref_cover's crop (the band's box plus 3 closings), so both read the same inks
    ys, xs = np.nonzero(band)
    p = 3 * s + 8
    y0, y1 = max(0, ys.min() - p), min(band.shape[0], ys.max() + 1 + p)
    x0, x1 = max(0, xs.min() - p), min(band.shape[1], xs.max() + 1 + p)
    B, C = band[y0:y1, x0:x1], cov_r[y0:y1, x0:x1]
    K = box_mask(band.shape, cfg)[y0:y1, x0:x1] if cfg.get("print_box") else np.ones(B.shape, bool)
    T, _, oth, _ = _channel_maps(Rl[y0:y1, x0:x1], B, K, s)
    wm = np.where(oth, 0.0, 1.0)
    Tk = np.ones_like(T)
    scale = np.ones(C.shape)
    lab, n = ndi.label(C > 0)
    other = (C > 0) & (wm < 0.5)
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        o = other[sl] & (lab[sl] == i)
        if o.sum() < 4:
            continue
        cmax = float(np.percentile(C[sl][o], 95))
        well = o & (C[sl] >= 0.6 * cmax)
        Tk[sl][well] = np.median(T[sl][well], 0)
        scale[sl][lab[sl] == i] = 1.0 / max(cmax, 0.05)
    has = np.zeros(C.shape, bool)
    has[(C > 0) & (wm < 0.5)] = True
    has &= (Tk < 1).any(-1)
    iy, ix = ndi.distance_transform_edt(~has, return_distances=False, return_indices=True) if has.any() else (None, None)
    full = np.ones(Rl.shape[:2] + (5,))
    if iy is not None:
        full[y0:y1, x0:x1, :3] = Tk[iy, ix]
    full[y0:y1, x0:x1, 3] = wm
    full[y0:y1, x0:x1, 4] = scale
    return full


def frame_stroke(crep, cover, cov_r):
    """The true print's widest stroke in scene px: the packshot's (ref_cover's report) times the warp's scale (the
    square root of the coverage's area ratio), or None."""
    w = crep.get("widest_stroke_px") or crep.get("open_auto", {}).get("widest_stroke_px")
    if not w or cov_r.sum() <= 0:
        return None
    return float(w) * float(np.sqrt(cover.sum() / cov_r.sum()))


def print_map(L, inside, size, dark):
    """How strongly each pixel reads as print, 0..1: its top-hat as a share of the body light (dark print) or of the
    headroom above it (light). -> (map, top-hat, body)"""
    top, body = _tophat(L, inside, size, dark)
    return np.clip(top / np.maximum(body if dark else 1.0 - body, 1e-3), 0, 1), top, body


def _ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def body_frame(mask, offset=(0, 0)):
    """The body's centre (scene px), axis and half-widths along it from a mask of it (bool; offset: where the mask's
    origin lies in the scene), for register()'s bow."""
    from . import bow as bw
    c, ax, n = bw.body_axis(mask)
    return {"centre": (c + np.asarray(offset, float)).tolist(), "axis": ax.tolist(),
            "half_widths": bw.half_widths(mask, c, ax, n)}


def prepare(ref_u8, refmask, cfg, H, roi, crop_u8, ss=4, prefilter=0.6, body=None, bow_auto=False):
    if cfg.get("polarity") == "mixed":
        return _prepare_mixed(ref_u8, refmask, cfg, H, roi, crop_u8, ss, prefilter, body, bow_auto)
    return _prepare(ref_u8, refmask, cfg, H, roi, crop_u8, ss, prefilter, body, bow_auto)


MIXED_SAME = 0.5     # a line found in both polarities (a light letter's dark rim) goes to the one covering more of it


def _prepare_mixed(ref_u8, refmask, cfg, H, roi, crop_u8, ss, prefilter, body, bow_auto):
    """Mixed print, text only: the dark lines and the light lines each prepared on their own (their own covers,
    registrations and line placements); a line both found goes to the polarity covering more of its box.
    -> prepare()'s dict, the cover the two together, text {"mixed", "dark", "light"} with each pass's own."""
    out = {}
    for pol in ("dark", "light"):
        out[pol] = _prepare(ref_u8, refmask, _sub(cfg, pol), H, roi, crop_u8, ss, prefilter, body, bow_auto)
    cd, cl = out["dark"]["cover"], out["light"]["cover"]
    # a print found both ways keeps one: per pixel, the polarity whose cover is stronger
    both = (cd > 0.02) & (cl > 0.02)
    lab, n = ndi.label(ndi.binary_dilation((cd > 0.1) | (cl > 0.1), iterations=1))
    if n and both.any():
        idx = np.arange(1, n + 1)
        sd, sl_ = ndi.sum(cd, lab, idx), ndi.sum(cl, lab, idx)
        drop_d = np.isin(lab, idx[(ndi.sum(both, lab, idx) > 0) & (sl_ > sd)])
        drop_l = np.isin(lab, idx[(ndi.sum(both, lab, idx) > 0) & (sd >= sl_)])
        cd = np.where(drop_d, 0.0, cd)
        cl = np.where(drop_l, 0.0, cl)
    rd, rl = out["dark"]["report"], out["light"]["report"]
    rep = {"mixed_print": {"dark": rd, "light": rl}, "open": [rd.get("open"), rl.get("open")],
           "print_px": int(rd.get("print_px", 0)) + int(rl.get("print_px", 0))}
    return {"cover": np.clip(cd + cl, 0, 1), "band": out["dark"]["band"], "box": None, "inks": None, "report": rep,
            "ref_sum": out["dark"]["ref_sum"] + out["light"]["ref_sum"],
            "text": {"mixed": True, "dark": out["dark"]["text"], "light": out["light"]["text"], "cover_dark": cd,
                     "cover_light": cl}}


def _prepare(ref_u8, refmask, cfg, H, roi, crop_u8, ss=4, prefilter=0.6, body=None, bow_auto=False):
    """Step 1, for run.py and the pack's Warp Reference: the packshot's print as coverage, warped through the fit;
    for dark print also the print box and the inks' colours, the print registered onto the frame's own (with a bow
    over it: `body` is align.bow's centre, axis and half_widths, else the warped outline's) and sharpened to its
    edges. bow_auto: the fit's bow (H a geometry.Bow) was measured from the matte, not given: kept only if the print
    registers better with it. -> {"cover", "band", "box", "inks", "report", "ref_sum"} (work-box arrays; box and inks
    None for light print; ref_sum the packshot coverage's total, for soft_check)"""
    cov_r, band, crep = ref_cover(ref_u8, refmask, cfg)
    extra = {}
    if cfg["gen_grow"] and cfg.get("print_box"):
        extra["box"] = box_mask(ref_u8.shape, cfg)
    if cfg["inks"] == "packshot":
        inks_r = ref_inks(ref_u8, band, cov_r, cfg, crep["open"])
        if inks_r is not None:
            extra["inks"] = inks_r
    kw = dict(ss=ss, prefilter=prefilter, blur=float(cfg["blur"]), extra=extra or None)
    cov, in_band, *ex = warp_cover(cov_r, band, H, roi, **kw)
    ex = ex[0] if ex else {}
    Hfin = H
    if cfg["register"]:
        if body is None and (in_band > 0.5).any():
            body = body_frame(in_band > 0.5, roi[:2])
            if cfg.get("text"):
                # text only: the print's own frame, its columns as the fit maps them (a label's outline has no long
                # axis to go by: a square one's turned the print 4 deg)
                from .geometry import apply_H
                ry_, rx_ = np.nonzero(cov_r > 0.5)
                if len(rx_):
                    q = apply_H(H, np.array([[rx_.mean(), ry_.mean()], [rx_.mean(), ry_.mean() + 50.0]]))
                    d_ = (q[1] - q[0]) / max(np.linalg.norm(q[1] - q[0]), 1e-9)
                    body = dict(body, axis=d_.tolist())
        # where the fit sends the packshot's rows at the print's centre (the fit alone, without the bow)
        from .geometry import apply_H
        ry, rx = np.nonzero(cov_r > 0.5)
        rows = None
        if len(rx):
            q = apply_H(H.base if hasattr(H, "profile") and hasattr(H, "base") else H,
                        np.array([[rx.mean(), ry.mean()], [rx.mean() + 50.0, ry.mean()]]))
            rows = q[1] - q[0]
        stroke = frame_stroke(crep, cov, cov_r)
        if cfg.get("text"):
            rows = None          # text only: no level turn (a tilted label's rows and columns aren't square in the frame)
        A, Abow, crep["register"] = register(crop_u8, cov, in_band, cfg, roi[:2], stroke, body, rows)
        if bow_auto and hasattr(H, "base"):
            # a bow measured from the matte's ends (Auto Draft, no bow given) is kept only where the generated print
            # bows too: a jar's did, a bottle's print the generator drew flat didn't (its base still sags 20 px)
            cov0, _, *ex0 = warp_cover(cov_r, band, H.base, roi, **kw)
            A0, Abow0, rep0 = register(crop_u8, cov0, in_band, cfg, roi[:2], stroke, body, rows)
            with_bow = crep["register"].get("ncc_after", -1) if A is not None else -1
            without = rep0.get("ncc_after", -1) if A0 is not None else -1
            crep["register"]["bow_from_matte"] = {"with": round(float(with_bow), 3), "without": round(float(without), 3),
                                                  "kept": bool(with_bow >= without)}
            if without > with_bow:
                H, cov, ex, A, Abow = H.base, cov0, ex0, A0, Abow0
                crep["register"] = dict(rep0, bow_from_matte=crep["register"]["bow_from_matte"])
                ex = ex[0] if ex else {}
        if A is not None and cfg.get("bow_from") == "print" and (bow_auto or not hasattr(H, "profile")) and body:
            # the bow from the generated print's own lines, at the heights the registered print takes (pitfall 100):
            # the matte's ends (a lid's rim, a base) bowed 51-63 px, a jar's print lines 73-88
            from .geometry import Bow
            base = H.base if hasattr(H, "profile") and hasattr(H, "base") else H
            cov_a, _, *_ = warp_cover(cov_r, band, Bow(Corrected(base, A), body["centre"], body["axis"],
                                                       body["half_widths"], H.profile) if hasattr(H, "profile")
                                      else Corrected(H, A, Abow), roi, **kw)
            prof, crep["bow_from_print"] = print_bow(crop_u8, in_band > 0.5, cov_a, body, roi[:2],
                                                     max(7, int(np.ceil(1.5 * stroke)) | 1) if stroke else 15)
            if prof is not None:
                frame_wb = {"centre": (np.asarray(body["centre"], float) - np.asarray(roi[:2], float)).tolist(),
                            "axis": body["axis"], "half_widths": body["half_widths"]}
                before = (H, cov, ex, A, Abow, crep["register"])
                for step in range(2):
                    H = Bow(base, body["centre"], body["axis"], body["half_widths"], prof)
                    cov, _, *ex = warp_cover(cov_r, band, H, roi, **kw)
                    ex = ex[0] if ex else {}
                    A2, Abow2, rep2 = register(crop_u8, cov, in_band, cfg, roi[:2], stroke, body, rows)
                    if A2 is None:
                        break
                    A, Abow = A2, Abow2
                    crep["register"] = dict(rep2, first=crep["register"].get("first", crep["register"]))
                    if step:
                        break
                    # the packshot's own lines curve too (a jar's front photo: -5 to -14 px of bow), and the warp
                    # carries that in: the true print's lines measured as warped, the profile corrected by what they
                    # miss of the generated print's, once
                    cov_b, _, *_ = warp_cover(cov_r, band, Bow(Corrected(base, A), body["centre"], body["axis"],
                                                               body["half_widths"], prof), roi, **kw)
                    got = [l for l in line_bows(np.clip(cov_b, 0, 1), in_band > 0.5, frame_wb)
                           if abs(l["tilt_deg"]) < 5 and l["rms"] < 3
                           and prof[0][0] - 10 <= l["t"] <= prof[-1][0] + 10]
                    if not got:
                        break
                    tp = np.array([l["t"] for l in got])
                    want = np.interp(tp, [q[0] for q in prof], [q[1] for q in prof])
                    miss = want - np.array([l["b"] for l in got])
                    wt = np.array([l["n"] for l in got], float)
                    if len(got) >= 2 and np.ptp(tp) > 20:
                        A_ = np.c_[np.ones_like(tp), tp] * np.sqrt(wt)[:, None]
                        m0, m1 = np.linalg.lstsq(A_, miss * np.sqrt(wt), rcond=None)[0]
                    else:
                        m0, m1 = float(np.average(miss, weights=wt)), 0.0
                    prof = [[q[0], round(float(q[1] + m0 + m1 * q[0]), 2)] for q in prof]
                    crep["bow_from_print"]["packshot_lines"] = [{"t": l["t"], "b": l["b"]} for l in got]
                    crep["bow_from_print"]["profile_corrected"] = prof
                # kept only where the print registers better with it, as the matte's bow: a bottle's print the
                # generator drew flat gave 2-8 px of noise and no better fit (0.697 -> 0.696), a jar's 0.58 -> 0.64
                with_bow = crep["register"].get("ncc_after", -1) if crep["register"] is not before[5] else -1
                without = before[5].get("ncc_after", -1)
                keep = with_bow >= without + 0.005
                crep["bow_from_print"].update({"with": round(float(with_bow), 3), "without": round(float(without), 3),
                                               "kept": bool(keep)})
                if not keep:
                    H, cov, ex, A, Abow, crep["register"] = before
        if A is not None:
            # the band (the body) stays where the fit put it; only the print moves. The body's bow (align.bow) goes
            # over the correction, not under it: the correction's scale along the axis (0.87 on a jar seen from
            # above) flattened the bow by as much, and the top rule bowed less than the lid seam above it
            if hasattr(H, "profile") and hasattr(H, "base") and Abow is None:
                from .geometry import Bow
                Hc = Bow(Corrected(H.base, A), H.centre, H.axis, H.half_widths, H.profile)
            else:
                Hc = Corrected(H, A, Abow)
            cov, _, *ex = warp_cover(cov_r, band, Hc, roi, **kw)
            ex = ex[0] if ex else {}
            Hfin = Hc
        else:
            Hfin = H
    text = None
    if cfg.get("text"):
        # text only (pitfall 111): the packshot's letters, line by line, onto the generated print's own lines
        from . import textline as tl
        cov, text, crep["text"] = tl.place_print(cov_r, band, Hfin, roi, crop_u8, in_band, cfg, kw,
                                                 frame_stroke(crep, cov, cov_r))
        # text_fit: whether the letters' size and place were checked against the generated print (the registration
        # kept) or come from the label's outline alone
        rg = crep.get("register") or {}
        text["register"] = {"run": bool(cfg.get("register")), "kept": bool(rg.get("kept")),
                            "ncc_after": rg.get("ncc_after"), "why": rg.get("why")}
    if cfg["sharpen"] == "auto":
        cov, crep["sharpen"] = sharpen_cover(cov, crop_u8, in_band, cfg, frame_stroke(crep, cov, cov_r))
    return {"cover": cov, "band": in_band, "box": ex.get("box"), "inks": ex.get("inks"), "report": crep,
            "ref_sum": float(cov_r.sum()), "text": text}


def register(crop_u8, cover, in_band, cfg, offset, stroke=None, body=None, rows=None):
    """The true print snapped onto the frame's own (garbled) print: an affine correction in scene px laid over the
    fit (pitfall 97). The generator draws the layout roughly right on the body it drew, while a fit to the silhouette
    can place it tens of px off (a jar seen from above: its lid's top widens the matte, and its front is foreshortened).
    Both as print maps, blurred: the warped coverage, and the frame's print against its body light (the closing wider
    than `stroke`, the true print's widest stroke in scene px) within reg_reach of the true print. First the x and y
    scales (1 +- reg_scale about the print's centre) and the shift with the best normalised correlation, at half
    resolution; then the full affine with a gain and an offset by Gauss-Newton at 2 and 1 px of blur, and with
    `body` (centre, axis, half_widths in scene px: align.bow's, or body_frame() of the warped outline) the body's
    frame: with reg_model "scale" the print stretches only along and across its axis, and with reg_bow a bow profile
    goes on top (off for dark print: fitted to garbled print it came out twice the jar's own, pitfall 97). Kept when
    the correlation reaches reg_min and improves.
    -> (3 x 3 correction in scene px or None, the bow over it {centre, axis, half_widths, b} or None, report)"""
    from scipy.signal import correlate
    dark = cfg["polarity"] == "dark"
    Lf = im.lum(im.srgb_to_lin(im.to_float(crop_u8)))
    B = in_band > 0.5
    C = np.clip(cover, 0, 1)
    foot = C > 0.1
    if foot.sum() < 50 or not B.any():
        return None, None, {"kept": False, "why": "too little print to register"}
    xs = np.nonzero(B.any(0))[0]
    width = xs.max() - xs.min() + 1
    reach = cfg["reg_reach"]
    reach = max(8, int(round(0.1 * width))) if reach == "auto" else int(reach)
    zone = ndi.binary_erosion(B, iterations=int(cfg["gen_inset"])) & (ndi.distance_transform_edt(~foot) <= reach)
    if stroke:
        so = max(7, int(np.ceil(1.5 * stroke)) | 1)
    else:
        big = int(max(15, width // 8)) | 1
        so = _widest(_tophat(Lf, B, big, dark)[0] * zone, zone, big)[0]
    P = print_map(Lf, B, so, dark)[0] * zone
    fy, fx = np.nonzero(foot)
    c = np.array([fx.mean(), fy.mean()])
    rep = {"reach": reach, "closing" if dark else "opening": so}

    # 1. x and y scale and shift: the coverage's box as a template slid over the frame's print map (half resolution)
    Pb, Cb = ndi.gaussian_filter(P, 2.0)[::2, ::2], ndi.gaussian_filter(C, 2.0)[::2, ::2]
    c2, r2 = c / 2.0, reach / 2.0
    best = (-2.0, 1.0, 1.0, 0.0, 0.0)
    n_sc = int(round(cfg["reg_scale"] / 0.025))
    steps = 1.0 + 0.025 * np.arange(-n_sc, n_sc + 1)
    asp = cfg.get("reg_aspect")
    for sy in steps:
        for sx in steps:
            if asp is not None and abs(sx / sy - 1.0) > float(asp) + 1e-9:
                continue
            Cs = ndi.affine_transform(Cb, np.diag([1 / sy, 1 / sx]), offset=(c2[1] * (1 - 1 / sy), c2[0] * (1 - 1 / sx)),
                                      order=1)
            ty, tx = np.nonzero(Cs > 0.02)
            if len(ty) < 20:
                continue
            r0, r1, c0, c1 = ty.min(), ty.max() + 1, tx.min(), tx.max() + 1
            T = Cs[r0:r1, c0:c1]
            if T.shape[0] > Pb.shape[0] or T.shape[1] > Pb.shape[1]:
                continue
            z = correlate(Pb, T, mode="valid", method="fft")
            E = correlate(Pb * Pb, np.ones_like(T), mode="valid", method="fft")
            sc_n = z / np.sqrt(np.maximum(E, 1e-9) * (T * T).sum())
            iy, ix = np.mgrid[0:sc_n.shape[0], 0:sc_n.shape[1]]
            sc_n[(np.abs(iy - r0) > r2) | (np.abs(ix - c0) > r2)] = -2
            k = np.unravel_index(sc_n.argmax(), sc_n.shape)
            if sc_n[k] > best[0]:
                best = (float(sc_n[k]), float(sx), float(sy), 2.0 * (k[1] - c0), 2.0 * (k[0] - r0))
    M = np.diag([best[1], best[2]])
    t = np.array([best[3], best[4]], float)

    # 2. the full affine, gain and offset (Gauss-Newton on the blurred maps)
    def warped(Pz, M, t, X, Y):
        ax = c[0] + M[0, 0] * X + M[0, 1] * Y + t[0]
        ay = c[1] + M[1, 0] * X + M[1, 1] * Y + t[1]
        return ndi.map_coordinates(Pz, [ay, ax], order=1, mode="constant"), ax, ay

    sup = ndi.binary_dilation(C > 0.02, iterations=4)
    sy_, sx_ = np.nonzero(sup)
    X, Y = sx_ - c[0], sy_ - c[1]
    fb = None
    if body is not None and cfg.get("reg_bow"):
        # the bow's shift along the axis per px of b at each print pixel (geometry.Bow's arc), and its change from
        # the print's top to its bottom: seen from above, lower lines bow more (one bow for all of a jar's print came
        # out 108 px, the lid seam's and the top rule's ~72, and the top rule ran skewed against the lid)
        from .geometry import Bow
        bw_ = Bow(np.eye(3), body["centre"], body["axis"], body["half_widths"], 1.0)
        fb = bw_.shift(sx_ + offset[0], sy_ + offset[1])
        tt = (sx_ + offset[0] - bw_.centre[0]) * bw_.axis[0] + (sy_ + offset[1] - bw_.centre[1]) * bw_.axis[1]
        t_lo, t_hi = np.percentile(tt, [2, 98])
        tc, th = 0.5 * (t_lo + t_hi), max(0.5 * (t_hi - t_lo), 1.0)
        uu = np.clip((tt - tc) / th, -1, 1)
        fbx, fby = fb * bw_.axis[0], fb * bw_.axis[1]
    before = _ncc(ndi.gaussian_filter(P, 1.0)[sy_, sx_], ndi.gaussian_filter(C, 1.0)[sy_, sx_])
    g, b0, db, db1 = 1.0, 0.0, 0.0, 0.0

    def moved(t):
        return t if fb is None else (t[0] + (db + db1 * uu) * fbx, t[1] + (db + db1 * uu) * fby)

    # the body's axis and its normal: with reg_model "scale" the print only stretches along and across them (no turn,
    # no shear), so its lines stay parallel to the body's own level edges
    if body is not None:
        axv = np.asarray(body["axis"], float)
        axv = axv / np.linalg.norm(axv)
    else:
        axv = np.array([0.0, 1.0])
    nv = np.array([axv[1], -axv[0]])
    scale_only = cfg.get("reg_model", "affine") == "scale"
    turn = 0.0
    Rt = np.eye(2)
    if scale_only:
        up = abs(axv[1]) >= abs(axv[0])
        sn, sa = (best[1], best[2]) if up else (best[2], best[1])
        if rows is not None and body is not None:
            # level: the print's rows (where the fit sends the packshot's rows) turned square to the body's axis, a
            # fixed turn, not fitted (the fit to a jar's silhouette turned the print 1 deg against its lid seam)
            rd = np.asarray(rows, float) / np.linalg.norm(rows)
            turn = float(np.arctan2(rd[0] * nv[1] - rd[1] * nv[0], rd[0] * nv[0] + rd[1] * nv[1]))
            Rt = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        M = (sn * np.outer(nv, nv) + sa * np.outer(axv, axv)) @ Rt
    for sigma in (2.0, 1.0):
        Pz, Cv = ndi.gaussian_filter(P, sigma), ndi.gaussian_filter(C, sigma)[sy_, sx_]
        gy, gx = np.gradient(Pz)
        for _ in range(40):
            Pw, ax, ay = warped(Pz, M, moved(t), X, Y)
            gxw = ndi.map_coordinates(gx, [ay, ax], order=1, mode="constant")
            gyw = ndi.map_coordinates(gy, [ay, ax], order=1, mode="constant")
            r = Pw - (g * Cv + b0)
            if scale_only:
                Xr, Yr = Rt[0, 0] * X + Rt[0, 1] * Y, Rt[1, 0] * X + Rt[1, 1] * Y
                pn, pa = Xr * nv[0] + Yr * nv[1], Xr * axv[0] + Yr * axv[1]
                cols = [(gxw * nv[0] + gyw * nv[1]) * pn, (gxw * axv[0] + gyw * axv[1]) * pa]
            else:
                cols = [gxw * X, gxw * Y, gyw * X, gyw * Y]
            k = len(cols)
            cols += [gxw, gyw, -Cv, -np.ones_like(Cv)]
            if fb is not None:
                cols += [gxw * fbx + gyw * fby, (gxw * fbx + gyw * fby) * uu]
            d = np.linalg.lstsq(np.c_[tuple(cols)], -r, rcond=None)[0]
            if scale_only:
                sn, sa = sn + d[0], sa + d[1]
                if asp is not None:
                    # the letters keep their shape: the two scales' ratio held within reg_aspect
                    gm, rr = np.sqrt(abs(sn * sa)), np.clip(sn / sa, 1.0 / (1.0 + float(asp)), 1.0 + float(asp))
                    sn, sa = gm * np.sqrt(rr), gm / np.sqrt(rr)
                M = (sn * np.outer(nv, nv) + sa * np.outer(axv, axv)) @ Rt
            else:
                M = M + d[:4].reshape(2, 2)
            t = t + d[k:k + 2]
            g, b0 = g + d[k + 2], b0 + d[k + 3]
            if fb is not None:
                db, db1 = db + d[k + 4], db1 + d[k + 5]
            if np.abs(d[k:k + 2]).max() < 0.005 and np.abs(d[:k]).max() < 1e-5 and \
                    (fb is None or np.abs(d[k + 4:k + 6]).max() < 0.01):
                break
    after = _ncc(warped(ndi.gaussian_filter(P, 1.0), M, moved(t), X, Y)[0], ndi.gaussian_filter(C, 1.0)[sy_, sx_])
    sc = float(np.sqrt(abs(np.linalg.det(M))))
    if scale_only:
        rep["level_turn_deg"] = round(float(np.degrees(turn)), 2)
    rep.update({"ncc_before": round(before, 3), "ncc_after": round(after, 3), "shift_px": t.round(2).tolist(),
                "scale": round(sc, 4), "scale_xy": [round(float(np.hypot(*M[:, 0])), 4), round(float(np.hypot(*M[:, 1])), 4)],
                "rotation_deg": round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 2),
                "matrix": M.round(5).tolist(), "coarse": {"ncc": round(best[0], 3), "scale_xy": [best[1], best[2]],
                                                          "shift_px": [best[3], best[4]]}})
    if fb is not None:
        rep["bow_px"] = [round(float(db - db1), 2), round(float(db + db1), 2)]      # at the print's top, bottom
        rep["bow_t"] = [round(float(t_lo), 1), round(float(t_hi), 1)]
    why = None
    sxy = np.array(rep["scale_xy"])
    if after < cfg["reg_min"]:
        why = f"the print maps correlate {after:.2f} after it (reg_min {cfg['reg_min']}): the generated print isn't the true layout"
    elif after <= before:
        why = f"no better than the fit ({after:.2f} against {before:.2f})"
    elif np.abs(t).max() > 1.5 * reach or np.abs(sxy - 1).max() > cfg["reg_scale"] + 0.05:
        why = f"it moved too far ({t.round(1).tolist()} px, scale {sxy.round(3).tolist()})"
    if why:
        rep.update(kept=False, why=why)
        return None, None, rep
    rep["kept"] = True
    ca = c + np.asarray(offset, float)
    A = np.eye(3)
    A[:2, :2] = M
    A[:2, 2] = ca + t - M @ ca
    return A, (dict(body, b=[[float(t_lo), float(db - db1)], [float(t_hi), float(db + db1)]]) if fb is not None else None), rep


def edge_sigma(P, sel):
    """A print map's edge blur in px: a unit step blurred by sigma has its steepest gradient 1 / (sigma sqrt(2 pi)).
    The 90th percentile of the gradient over its mid-tone pixels in `sel`, against its 99th-percentile level."""
    gy, gx = np.gradient(P)
    g = np.hypot(gx, gy)
    on = sel & (P > 0.05)
    if on.sum() < 50:
        return float("nan")
    hi = float(np.percentile(P[on], 99))
    edge = sel & (P > 0.3 * hi) & (P < 0.7 * hi)
    if edge.sum() < 20:
        return float("nan")
    return float(hi / (np.sqrt(2 * np.pi) * np.percentile(g[edge], 90)))


def sharpen_cover(cover, crop_u8, in_band, cfg, stroke=None):
    """The warped print's edges to the frame's own print's blur: an unsharp mask (radius the coverage's own blur),
    its amount bisected until the edge blurs match, clipped to 0..1 (no halo either way). The frame's print is read
    within 3 px of the (registered) true print. -> (cover, report)"""
    dark = cfg["polarity"] == "dark"
    Lf = im.lum(im.srgb_to_lin(im.to_float(crop_u8)))
    B = in_band > 0.5
    near = B & ndi.binary_dilation(cover > 0.1, iterations=3)
    so = max(7, int(np.ceil(1.5 * stroke)) | 1) if stroke else 15
    P = print_map(Lf, B, so, dark)[0]
    s_f, s_c = edge_sigma(P, near), edge_sigma(cover, near)
    rep = {"edge_sigma_frame": round(s_f, 3), "edge_sigma_print": round(s_c, 3)}
    if not np.isfinite(s_f + s_c) or s_c <= 1.05 * s_f:
        return cover, dict(rep, amount=0.0)
    blurred = ndi.gaussian_filter(cover, s_c)

    def sharp(a):
        return np.clip(cover + a * (cover - blurred), 0, 1)

    lo, hi = 0.0, float(cfg.get("sharpen_max", 1.5))
    if edge_sigma(sharp(hi), near) > s_f:
        # more would ring and alias: a packshot this soft is too small for the frame (pitfall 97)
        out = sharp(hi)
        rep.update({"amount": hi, "radius": round(s_c, 3), "edge_sigma_after": round(edge_sigma(out, near), 3),
                    "capped": True})
        return out, rep
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if edge_sigma(sharp(mid), near) > s_f:
            lo = mid
        else:
            hi = mid
    out = sharp(hi)
    rep.update({"amount": round(hi, 3), "radius": round(s_c, 3), "edge_sigma_after": round(edge_sigma(out, near), 3)})
    return out, rep


def garbled(crop_u8, M, cover, in_band, cfg, vis=None, in_box=None, ink_rgb=None, text=None):
    if text is not None and text.get("mixed"):
        return _garbled_mixed(crop_u8, M, in_band, cfg, vis, text)
    return _garbled(crop_u8, M, cover, in_band, cfg, vis, in_box, ink_rgb, text)


def _garbled_mixed(crop_u8, M, in_band, cfg, vis, text):
    """Mixed print, text only: the generated dark letters removed and the dark lines' inks read, then the light ones
    on that result; one info for composite() (k per pixel: below 1 on the dark lines, above on the light)."""
    fd, idk, rd = _garbled(crop_u8, M, text["cover_dark"], in_band, _sub(cfg, "dark"), vis, None, None, text["dark"])
    mid = np.clip(np.round(im.lin_to_srgb(np.clip(fd, 0, 1)) * 255), 0, 255).astype(np.uint8)
    mid[~M] = crop_u8[~M]
    fl, il, rl = _garbled(mid, M, text["cover_light"], in_band, _sub(cfg, "light"), vis, None, None, text["light"])
    a_d, a_l = idk["alpha"], il["alpha"]
    kd = np.broadcast_to(np.asarray(idk["k"], float), fl.shape)
    kl = np.broadcast_to(np.asarray(il["k"], float), fl.shape)
    k = np.where((a_d >= a_l)[..., None], kd, kl)
    info = {"polarity": "mixed", "k": k, "alpha": np.clip(a_d + a_l, 0, 1), "region": idk["region"] | il["region"],
            "matte": M.copy(), "fill": idk["fill"] | il["fill"], "vis": vis,
            "frame_size": max(int(idk.get("frame_size") or 0), int(il.get("frame_size") or 0)),
            "grown": (idk.get("grown") if idk.get("grown") is not None else np.zeros(M.shape, bool)) |
                     (il.get("grown") if il.get("grown") is not None else np.zeros(M.shape, bool)),
            "text": text, "keep": idk.get("keep", np.zeros(M.shape, bool)) | il.get("keep", np.zeros(M.shape, bool)),
            "passes": [("dark", idk), ("light", il)]}
    if il.get("glow") is not None:
        info["glow"] = il["glow"]
    if idk.get("ink_abs") is not None or il.get("ink_abs") is not None:
        # a printed body: each pass's absolute inks (the ratio's ink where a pass has none)
        ad = idk["ink_abs"] if idk.get("ink_abs") is not None else np.clip(fd * kd, 0, 1)
        al = il["ink_abs"] if il.get("ink_abs") is not None else np.clip(fl * kl, 0, 1)
        info["ink_abs"] = np.where((a_d >= a_l)[..., None], ad, al)
    rep = {"mixed_fill": {"dark": rd, "light": rl}, "garbled_px": rd["garbled_px"] + rl["garbled_px"],
           "filled_px": rd["filled_px"] + rl["filled_px"], "ink_k": [rd["ink_k"], rl["ink_k"]]}
    return fl, info, rep


def _garbled(crop_u8, M, cover, in_band, cfg, vis=None, in_box=None, ink_rgb=None, text=None):
    """3. The frame's garbled print removed: -> (filled linear rgb, info for composite(), report).
    M: the generated product's matte over the work box (bool); in_box: the print box warped (warp_cover's extra
    "box"), with gen_grow; ink_rgb: ref_inks() warped (extra "inks"), each ink's own colour; text: prepare()'s text
    info (matte "text"): the generated letters are removed, and only they (its rules, bands, edges and lone marks
    stay), and each new line takes the ink of the generated line under it. info carries k (the ink's colour, per pixel
    with text), the composite's alpha and region, and M."""
    pol = cfg.get("polarity", "light")
    dark = pol == "dark"
    Fl = im.srgb_to_lin(im.to_float(crop_u8))
    inner = ndi.binary_erosion(M, iterations=1)
    work = inner & (in_band > 0.5)
    if not work.any():
        raise InkError(f"matte ink ({pol}): the warped reference outline doesn't overlap the product's matte: check "
                       "align and old_silhouette (the matte)")
    Lf = im.lum(Fl)
    zone = work
    if cfg["gen_near"] is not None:
        # only garbled print near where the true print goes (its layout is the true one's, a few px off), inset from
        # the matte: a frosted body's own highlights (the band under the shoulder, the limb rims, the base) stay
        zone = work & ndi.binary_dilation(cover > 0.1, iterations=int(cfg["gen_near"])) & \
            ndi.binary_erosion(M, iterations=int(cfg["gen_inset"]))
    rep = {}
    so = cfg["frame_open"]
    if so == "auto":
        # wider than the frame's widest garbled stroke: a closing narrower than a stroke leaves its middle as body
        xs = np.nonzero(M.any(0))[0]
        big = int(max(15, (xs.max() - xs.min()) // 8)) | 1 if len(xs) else 15
        # a second ink (ref_inks' main-ink weight under 0.5 on the true print): the widest stroke in any channel
        multi = ink_rgb is not None and bool((ink_rgb[..., 3][cover > 0.5] < 0.5).any())
        chans = [Fl[..., c] for c in range(3)] if dark and multi else [Lf]
        so, widest = max(_widest(_tophat(ch, M, big, dark)[0] * zone, zone, big) for ch in chans)
        rep["frame_open_auto"] = {"size": so, "widest_stroke_px": round(widest, 1)}
    so = int(so)
    if dark:
        body = _dark_body(Lf, M, so)[0]
        top = body - Lf
    else:
        body = ndi.grey_opening(np.where(M, Lf, 0), size=(so, so))
        top = Lf - body
    cand = top > cfg["gen_thresh"]
    if cfg["gen_rel"] is not None:
        cand &= top > cfg["gen_rel"] * (body if dark else 1.0 - body)
    gen = zone & cand
    grown = None
    tz = None
    if text is not None:
        # text only: the generated letters inside the band (any line of them: the packshot's print is the label's
        # whole text) and garbled marks near the new print, never the generated rules, bands, edges or lone marks
        f_px = cfg.get("_px_scale") or 1          # the work scale (pitfall 113): these reaches are scene px
        inset = ndi.binary_erosion(M, iterations=int(cfg["gen_inset"]))
        if cfg.get("label"):
            # the generated label's measured outline (label mode): its border and what lies past it stay
            from . import masks as mk_
            lab_in = mk_.poly_cover(text["roi"], cfg["label"], ss=2) > 0.5
            inset &= ndi.binary_erosion(lab_in, iterations=int(round(LABEL_INSET * f_px)))
        # every mark of the generated print in the band (the packshot's print is the label's whole text: garbled
        # lines it has no line for go too), but its rules, bands, edges and icons (marks in no line of text, larger
        # than its letters) unless the new print covers them
        keep = ndi.binary_dilation(text["gen_other"] | (text["gen_icon"] & ~ndi.binary_dilation(
            cover > 0.1, iterations=int(round(2 * f_px)))), iterations=int(round(f_px)))
        if text.get("gen_kept") is not None:
            # the generated print's own lines the gate kept (legible already, or not type): never filled
            keep |= text["gen_kept"]
        # whole marks: a generated letter or word reaching into the text area goes all of it (cut at the area's edge,
        # its rest fed the fill and smudged it)
        # only marks holding letters (or small lone marks) of the generated print: a band's faint end, a label's
        # shading and the paper's grain that read as print near the new print stay
        # marks labelled whole across the inset: a generated word crossing the label's measured outline (a last line
        # set close to a curved bottom edge) goes whole when most of it lies inside (pitfall 113: cut at the outline,
        # its lower half stayed under the new last word)
        body_in = ndi.binary_erosion(M, iterations=int(cfg["gen_inset"]))
        lab_all, n_g = ndi.label(work & body_in & cand & ~keep, structure=np.ones((3, 3), bool))
        lab_g = lab_all
        if n_g:
            tot = ndi.sum(np.ones_like(lab_all), lab_all, np.arange(1, n_g + 1))
            ins = ndi.sum(inset, lab_all, np.arange(1, n_g + 1))
            out_ = np.r_[False, ins < INSET_SHARE * tot]
            lab_g = np.where(out_[lab_all], 0, lab_all)
        # small lone marks only beside letters (a dot, an accent): elsewhere they are the body's glitter and specks
        letters = text["gen_text"] | (text["gen_lone"] & ~text["gen_icon"] &
                                      ndi.binary_dilation(text["gen_text"], iterations=int(round(3 * f_px))))
        hit = np.unique(lab_g[text["text_zone"] & letters & (lab_g > 0)])
        # a generated line goes whole once any of its letters go (a block placed a little lower left its top line's
        # first word above the text area)
        gl = text.get("gen_line")
        if gl is not None:
            lines_hit = np.unique(gl[np.isin(lab_g, hit[hit > 0]) & (gl > 0)])
            if len(lines_hit):
                hit = np.union1d(hit, np.unique(lab_g[np.isin(gl, lines_hit) & letters & (lab_g > 0)]))
                # ... its marks past the label's measured outline too, inside the product (a corner the draft took
                # for hidden round the limb cut a last line's end off: it stayed under the new last word)
                hit = np.union1d(hit, np.unique(lab_all[np.isin(gl, lines_hit) & letters & (lab_all > 0)]))
        # ... and any generated mark the new letters land on in the text area: new ink never goes over old (a
        # generated i drawn as one tall stroke stayed under the new i's dot, which came out black, pitfall 113)
        under = np.unique(lab_all[(cover > 0.1) & text["text_zone"] & (lab_all > 0)])
        hit = np.union1d(hit, under[under > 0])
        if text.get("gen_kept") is not None and not cfg.get("label"):
            # a printed body (the gate's text on the body): only the generated marks by the new letters go, never a
            # mark the gate kept or a mark no new line replaces (its whole text isn't the packshot's lines: a glossy
            # tube's legible wordmark was erased with nothing put back)
            lh = float(text.get("line_h_px") or 8.0)
            reach = int(max(2, round(NEAR_NEW * lh)))
            near_new = ndi.binary_dilation(cover > 0.1, iterations=reach)
            # ... and past each new line's ends along its own run (a garbled line runs longer than the copy)
            nl_ = text.get("new_line")
            if nl_ is not None:
                for j in np.unique(nl_[nl_ > 0]):
                    yy, xx = np.nonzero(nl_ == j)
                    if len(yy) < 8:
                        continue
                    P_ = np.c_[xx - xx.mean(), yy - yy.mean()]
                    ax = np.linalg.svd(P_, full_matrices=False)[2][0]
                    base_ = ndi.binary_dilation(nl_ == j, iterations=reach)
                    for t_ in np.linspace(-NEAR_RUN * lh, NEAR_RUN * lh, 9):
                        near_new |= ndi.shift(base_.astype(np.uint8), (t_ * ax[1], t_ * ax[0]), order=0) > 0
            ok = np.unique(lab_all[near_new & (lab_all > 0)])
            hit = np.intersect1d(hit, ok)
            if gl is not None and len(hit):
                # ... and the rest of a generated line most of which goes (its tail curving off the new line's)
                gone = np.isin(lab_all, hit) & (gl > 0)
                for g_ in np.unique(gl[gone]):
                    line_px = (gl == g_) & letters & (lab_all > 0)
                    if (gone & (gl == g_)).sum() >= LINE_WHOLE * max(line_px.sum(), 1):
                        hit = np.union1d(hit, np.unique(lab_all[line_px]))
        lab_g = np.where(np.isin(lab_all, hit[hit > 0]), lab_all, lab_g)
        tz = np.isin(lab_g, hit[hit > 0])
        gen = tz
        import os
        if os.environ.get("PLR_TEXT_DEBUG"):
            np.savez_compressed(os.environ["PLR_TEXT_DEBUG"].replace(".npz", "_wb.npz"), cand=cand, work=work,
                                inset=inset, keep=keep, zone=text["text_zone"], tz=tz, gen=gen,
                                other=text["gen_other"], icon=text["gen_icon"], cover=cover)
        rep["text_px"] = int(tz.sum())
        rep["kept_marks_px"] = int((keep & work).sum())
    elif cfg["gen_grow"]:
        # whole blocks of garbled print: its strokes grouped into words and blocks (gaps under a third of the frame's
        # closing merged: a badge's ring text, ring and icon), each taken whole where it touches the true print or
        # the print box and lies mostly inside the box: a garbled badge drawn higher than the true one goes entirely,
        # a last line of text touching a seam (and through it the glass below) and the matte's rim stay
        cand &= work & ndi.binary_erosion(M, iterations=int(cfg["gen_inset"]))
        # long level lines (runs 3 closings long, together over 40 % of the body's width) never grow: a seam, the
        # lid's edge or the base (a jar's pink-to-glass seam touches its last line of text and joins it to the glass
        # below), or a rule, which lies on the true print's own and is filled there
        bx = np.nonzero(work.any(0))[0]
        runs = ndi.binary_opening(cand, structure=np.ones((1, 3 * so), bool))
        lab, n = ndi.label(runs)
        wide = [i for i, sl in enumerate(ndi.find_objects(lab), 1) if sl[1].stop - sl[1].start > 0.4 * (bx.max() - bx.min())]
        lines = ndi.binary_dilation(np.isin(lab, wide), iterations=1) & cand if wide else np.zeros_like(cand)
        rep["line_px"] = int(lines.sum())
        cand_g = cand & ~lines
        lab, n = ndi.label(ndi.binary_dilation(cand_g, iterations=max(2, int(round(so / 3)))))
        lab = lab * cand_g
        grow = np.unique(lab[gen & (lab > 0)])
        if in_box is not None:
            boxm = in_box > 0.5
            grow = np.union1d(grow, np.unique(lab[cand_g & boxm]))
            idx = np.arange(1, n + 1)
            share = ndi.sum(boxm, lab, idx) / np.maximum(ndi.sum(cand_g, lab, idx), 1)
            grow = grow[share[grow - 1] >= 0.5] if len(grow) else grow
        grown = cand_g & np.isin(lab, grow)
        if in_box is not None and lines.any():
            # a long line lying almost wholly inside the print box is print (a bottle's coloured bar)
            ll, nl = ndi.label(lines)
            idx = np.arange(1, nl + 1)
            keep_l = idx[ndi.sum(in_box > 0.5, ll, idx) >= 0.8 * ndi.sum(lines, ll, idx)]
            grown |= np.isin(ll, keep_l)
        gen = (zone & cand) | grown
        rep["grown_px"] = int(gen.sum() - (zone & cand).sum())
    occ = None
    if vis is not None:
        gen &= vis > 0.5
        if (vis < 0.99).any():
            # an occluder's soft edge (a fish's rim, its fins, a snout's outline 3-4 px past SAM 3's mask) isn't
            # garbled print (pitfall 97; ink_rim checks it)
            occ = vis < 0.99
            gen &= ~ndi.binary_dilation(occ, iterations=4)
    if not gen.any():
        raise InkError(f"matte ink ({pol}): no garbled print {'darker' if dark else 'brighter'} than the frame's body "
                       f"inside the band (gen_thresh, frame_open): is the print {pol} on this product?")
    core = work & (top > np.percentile(top[gen], 75))
    dil = int(cfg["gen_dilate"])
    fillm = (ndi.binary_dilation(gen, iterations=dil) if dil > 0 else gen) & ndi.binary_erosion(M, iterations=2)
    if text is not None:
        fillm &= ~keep                 # the fill's reach stops at the label's own marks (a band under a wordmark)
    if dark:
        # the garbled strokes' faint edges (4 % below the body) and their sharpening halos (3 % above it), within
        # 3 px: as the fill's border they set its level, and the fill came out light, the body left between the
        # lines as orange blotches
        near3 = ndi.binary_dilation(gen, iterations=3) & ndi.binary_erosion(M, iterations=2)
        if text is not None:
            near3 &= ~keep
        fillm |= near3 & ((top > 0.04 * body) | (-top > 0.03 * body))
        rep["halo_px"] = int((near3 & (-top > 0.03 * body)).sum())
    glow = None
    if not dark and cfg.get("gen_glow"):
        # a generated frame's light print fades into its body over several px, past its top-hat and the fill's reach:
        # filled from there, the fill came out brighter than the body (a serum: +9.5 levels a px out, +4 at 3 px) and
        # each old letter showed as a soft ghost beside the new one. The glow: px within gen_glow of the fill and
        # glow_thresh above the body light, the body the local mean of the px off every light stroke by gen_glow
        # (the opening sits on the glow), joined to the fill
        r = int(cfg["gen_glow"])
        lit = M & (top > 0.5 * cfg["gen_thresh"])
        w = (M & ~ndi.binary_dilation(lit | fillm, iterations=r)).astype(float)
        den = ndi.gaussian_filter(w, 1.5 * r)
        bl = ndi.gaussian_filter(_extend(Lf, M) * w, 1.5 * r) / np.maximum(den, 1e-6)
        glow = ndi.binary_dilation(fillm, iterations=r) & ndi.binary_erosion(M, iterations=2) & (den > 0.02) & \
            (Lf - bl > cfg["glow_thresh"])
        if vis is not None:
            glow &= vis > 0.5
        lab, _ = ndi.label(glow | fillm)
        glow &= np.isin(lab, np.unique(lab[fillm]))
        glow &= ~fillm
        if text is not None:
            glow &= ~keep              # text only: the glow's fill stops at the product's own marks too
        fillm |= glow
        rep["glow_px"] = int(glow.sum())
    known = M & ~fillm
    if occ is not None:
        # the fill stays a px off the occluder and never borrows its colour (a fish's bright body showed as blotches
        # along its edge)
        fillm &= ~ndi.binary_dilation(occ, iterations=1)
        known &= ~occ
    try:
        filled = ipfill.harmonic_fill(Fl, fillm, known)[0]
    except ipfill.FillError as e:
        raise InkError(f"matte ink ({pol}): garbled print with no body around it: {e}") from e
    # the frame's grain over the filled pixels (MAD of the high-pass on the body it kept)
    hp = Fl - ndi.gaussian_filter(Fl, (1.2, 1.2, 0))
    sel = known & work
    sd = 1.4826 * np.median(np.abs(hp[sel] - np.median(hp[sel], 0)), 0)
    rng = np.random.default_rng(int(cfg["seed"]))
    filled[fillm] += rng.normal(0, 1, (int(fillm.sum()), 3)) * sd
    # the ink's colour from the generated print's fully covered pixels: k = frame / body, per channel (its brightest
    # ratios for light print, its darkest for dark)
    ratio = Fl[core] / np.maximum(filled[core], 1e-4)
    if dark:
        k = np.minimum(np.percentile(ratio, 100 - cfg["ink_pct"], axis=0), 1.0)
    else:
        k = np.maximum(np.percentile(ratio, cfg["ink_pct"], axis=0), 1.0)
    a = np.clip(cover, 0, 1) * inner * np.clip(in_band, 0, 1)
    if ink_rgb is not None:
        # another ink's alpha against its own peak coverage (a light yellow is solid where it's at its peak)
        wm = np.clip(ink_rgb[..., 3], 0, 1)
        a = np.clip(a * (wm + (1 - wm) * ink_rgb[..., 4]), 0, 1)
    if vis is not None:
        a = a * vis
    info = {"polarity": pol, "k": k, "alpha": a, "region": work | fillm, "matte": M.copy(), "fill": fillm}
    info.update(vis=vis, frame_size=so)            # for the checks (dark_checks)
    if ink_rgb is not None:
        # the main ink in the frame's own colour (k); any other ink in its packshot colour, relit by the frame's body
        # and no darker than k
        wm = np.clip(ink_rgb[..., 3:4], 0, 1)
        info["k"] = wm * k + (1 - wm) * np.maximum(np.clip(ink_rgb[..., :3], 0, 1), k)
        other = (a > 0.5) & (wm[..., 0] < 0.5)
        rep["other_ink_px"] = int(other.sum())
        if other.any():
            rep["other_ink_lin"] = np.median(info["k"][other], 0).round(3).tolist()
    if grown is not None:
        # the blocks of garbled print gen_grow took whole, with the fill's reach round them: check() reads them
        info["grown"] = ndi.binary_dilation(grown, iterations=dil + 3) & fillm
    if text is not None:
        # the generated letters taken away from the new print, with the fill's reach: check() reads them as print
        info["grown"] = ndi.binary_dilation(tz, iterations=dil + 3) & fillm
        info["keep"] = keep & work
        info["text_zone"] = text["text_zone"] & work
        info["px_scale"] = cfg.get("_px_scale") or 1
        info["k"], rep["text_inks"], KA = _line_inks(Fl, filled, gen, top, text, k, cfg, dark, a)
        info["text"] = text
        if KA is not None:
            # a printed body: the letters in the generated letters' own colour (as a ratio over the body, letters on
            # a near-black base came out speckled and fringed, the ratio 300-950)
            info["ink_abs"] = KA
    if glow is not None:
        info["glow"] = glow            # the garbled strokes' glow the fill took: check() reads it as theirs
    rep.update({"filled_px": int(fillm.sum()), "garbled_px": int(gen.sum()), "grain_sd_lin": sd.round(5).tolist(),
                "ink_k": k.round(3).tolist()})
    return filled, info, rep


def _line_inks(Fl, filled, gen, top, text, k, cfg, dark, a):
    """Text only: each new line's ink from the generated print under it (frame / filled body over the most covered
    garbled px within 2 px of the line, as k), a line with too few the nearest line's with enough, else the print's k.
    On a printed body (no label) the ink is absolute: the generated letters' own colour (their most covered px), a
    line with too few taking the nearest line's in the frame (a turned block's lines borrowed by row number across
    frames took a glint's and a blue ink). -> (k per pixel, h x w x 3; report; absolute ink per pixel or None)"""
    nl = text["new_line"]
    lr = text["lines_true"]
    absolute = not cfg.get("label") and text.get("gen_kept") is not None
    own, own_abs = {}, {}
    for L in lr:
        if len(L.get("members") or []) <= 2 and len(lr) > 1:
            continue        # a dot or a short mark on its own: too little generated print under it to read its ink (a
            # wordmark's i dot on bare paper came out black against its letters' dark brown, pitfall 113)
        foot = ndi.binary_dilation(nl == L["k"] + 1, iterations=2)
        sel = gen & foot
        if sel.sum() < 12:
            continue
        core = sel & (top >= np.percentile(top[sel], 75))
        ratio = Fl[core] / np.maximum(filled[core], 1e-4)
        own[L["k"]] = (np.minimum(np.percentile(ratio, 100 - cfg["ink_pct"], axis=0), 1.0) if dark else
                       np.maximum(np.percentile(ratio, cfg["ink_pct"], axis=0), 1.0))
        own_abs[L["k"]] = np.median(Fl[core], 0)
    cen = {}
    if absolute:
        for L in lr:
            ys, xs = np.nonzero(nl == L["k"] + 1)
            if len(ys):
                cen[L["k"]] = np.array([ys.mean(), xs.mean()])

    def nearest(j):
        if absolute and j in cen:
            have = [q for q in own if q in cen]
            if have:
                return min(have, key=lambda q: float(np.hypot(*(cen[q] - cen[j]))))
        return min(own, key=lambda q: abs(lr[q]["cy"] - lr[j]["cy"]))
    kl, ka = {}, {}
    by_k = {L["k"]: L for L in lr}
    for L in lr:
        j = L["k"]
        src = j if j in own else (nearest(j) if own else None)
        kl[j] = own[src] if src is not None else k
        if src is not None:
            ka[j] = own_abs[src]
    K = np.broadcast_to(np.asarray(k, float), Fl.shape).copy()
    KA = None
    if (nl > 0).any():
        iy, ix = ndi.distance_transform_edt(nl == 0, return_distances=False, return_indices=True)
        idn = nl[iy, ix]
        on = a > 0
        for j, kj in kl.items():
            K[on & (idn == j + 1)] = kj
        if absolute and ka:
            KA = np.zeros(Fl.shape)
            KA[...] = np.median(np.stack(list(ka.values())), 0)
            for j, kj in ka.items():
                KA[on & (idn == j + 1)] = kj
    rep = {"own": len(own), "lines": len(lr), "absolute": bool(KA is not None),
           "k_lines": {int(j): np.asarray(v).round(3).tolist() for j, v in kl.items()}}
    return K, rep, KA


def composite(filled, crop_u8, info):
    """The true print laid on the filled body: -> (work box uint8, alpha). Pixels outside the matte stay the scene's,
    and inside it only the band and the fill change."""
    Fl = im.srgb_to_lin(im.to_float(crop_u8))
    a = info["alpha"]
    ink = np.clip(info["ink_abs"], 0, 1) if info.get("ink_abs") is not None else np.clip(filled * info["k"], 0, 1)
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
    grown = info.get("grown")
    if grown is not None:
        foot |= grown
    if info.get("glow") is not None:
        foot |= ndi.binary_dilation(info["glow"], iterations=1)
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
    if grown is not None:
        # garbled print taken whole beyond the true print's footprint (gen_grow): print the generator drew elsewhere
        sel = changed & grown & ~ndi.binary_dilation(cover > 1e-3, iterations=reach)
        lab, n = ndi.label(sel)
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            px = int((lab[sl] == i).sum())
            if px >= 12:
                ox, oy = offset
                out.append({"check": "ink_edges", "level": "INFO", "px": px,
                            "box": [sl[1].start + ox, sl[0].start + oy, sl[1].stop - 1 + ox, sl[0].stop - 1 + oy],
                            "text": f"{px} px of garbled print removed away from the true print (a block mostly in the "
                                    "print box, taken whole: gen_grow); check the body there"})
    return sorted(out, key=lambda f: (f["level"] != "WARN", -f["px"]))[:8]


def soft_check(crep, cover, ref_sum, alpha, offset):
    """A packshot too small for the frame (pitfall 97): the frame shows its print over 1.25x its size (its small
    letters' detail is lost however sharp their edges get), or its edges stay over 10 % softer than the generated
    print's after the sharpen's cap. A WARN with the scale and the print's box. -> audit findings ("packshot")"""
    sh = crep.get("sharpen") or {}
    scale = float(np.sqrt(cover.sum() / max(ref_sum, 1e-6)))
    soft = sh.get("capped") and sh.get("edge_sigma_after", 0) > 1.1 * sh.get("edge_sigma_frame", 1e9)
    if scale <= 1.25 and not soft:
        return []
    ys, xs = np.nonzero(alpha > 0.5)
    box = [int(xs.min()) + offset[0], int(ys.min()) + offset[1], int(xs.max()) + offset[0], int(ys.max()) + offset[1]] \
        if len(xs) else None
    return [{"check": "packshot", "level": "WARN", "px": int((alpha > 0.5).sum()), "box": box, "scale": round(scale, 2),
             "text": f"the frame shows the packshot's print {scale:.2f}x its size" +
                     (f", its edges {sh['edge_sigma_after']} px soft after the sharpen's cap against the frame's "
                      f"{sh['edge_sigma_frame']}" if soft else "") +
                     ": its small print loses detail; a larger packshot gives a crisper print (look at it at 100 %)"}]


def _print_maps(src_u8, out_u8, M, size, dark=True):
    """The frame's and the result's print (dark, or light) as maps (0..1 against the body light), and the changed
    pixels."""
    Ls = im.lum(im.srgb_to_lin(im.to_float(src_u8)))
    Lo = im.lum(im.srgb_to_lin(im.to_float(out_u8)))
    Ps = print_map(Ls, M, size, dark)[0] * M
    Po = print_map(Lo, M, size, dark)[0] * M
    changed = np.abs(out_u8.astype(int) - src_u8.astype(int)).max(-1) > 12
    return Ps, Po, changed


def _robust_fit(x, y, deg, tol=2.5, iters=4):
    """Least squares polynomial with points over tol px (or 2.5 MAD) off it left out. -> (coefficients, inliers)"""
    keep = np.ones(len(x), bool)
    c = np.polyfit(x, y, deg)
    for _ in range(iters):
        r = y - np.polyval(c, x)
        lim = max(tol, 2.5 * 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep]))))
        new = np.abs(r) <= lim
        if new.sum() < deg + 8 or (new == keep).all():
            break
        keep = new
        c = np.polyfit(x[keep], y[keep], deg)
    return c, keep


def line_bows(P, M, frame=None, min_len=0.3):
    """Each line of print's own curve across a round body: its bow and its tilt (pitfall 100).
    P: a print map over the work box (0..1); M: the body's matte (bool, same box); frame: {centre, axis,
    half_widths} in work-box px (default: the matte's own, bow.body_axis). The print is unrolled into the body's frame
    (s across, t along the axis, 1 px steps); the lines are the peaks of its profile along t in the middle strip
    (|s| under a fifth of the width), each followed outward both ways in 12 px windows by its print's centre of mass,
    then fitted as t(s) = a + k s + b sqrt(1 - s^2 / w(t)^2): b is the line's bow (+: its ends higher, seen from above),
    atan(k) its tilt. A jar seen from above: its generated print's lines bowed 73-88 px, its lid's rim 63 and its base
    51; one bow from those flattened the print's lower lines ("doesn't respect the shape of the product").
    -> [{"t", "b", "tilt_deg", "s_span", "rms", "n", "h"}, ...] from top to bottom (h: the line's height in the middle
    strip, px: a rule or an edge a few, a line of text its letters' height)"""
    from . import bow as bw
    if frame is None:
        c, ax, n = bw.body_axis(M)
        hw = np.asarray(bw.half_widths(M, c, ax, n), float)
    else:
        c = np.asarray(frame["centre"], float)
        ax = np.asarray(frame["axis"], float)
        ax = ax / np.linalg.norm(ax)
        n = np.array([ax[1], -ax[0]])
        hw = np.asarray(frame["half_widths"], float)
    if len(hw) < 3:
        return []
    W = float(np.median(hw[:, 1]))
    pts = np.argwhere(P > 0.3)
    if len(pts) < 50:
        return []
    tt_all = (pts[:, 1] - c[0]) * ax[0] + (pts[:, 0] - c[1]) * ax[1]
    t0, t1 = int(np.floor(tt_all.min())) - 6, int(np.ceil(tt_all.max())) + 6
    S = np.arange(-int(0.95 * W), int(0.95 * W) + 1)
    T = np.arange(t0, t1 + 1)
    SS, TT = np.meshgrid(S, T)
    X, Y = c[0] + SS * n[0] + TT * ax[0], c[1] + SS * n[1] + TT * ax[1]
    R = ndi.map_coordinates(P, [Y, X], order=1, mode="constant")
    R *= ndi.map_coordinates(M.astype(float), [Y, X], order=1, mode="constant") > 0.5
    mid = np.abs(S) <= 0.2 * W
    prof = ndi.gaussian_filter1d(R[:, mid].mean(1), 1.5)
    if prof.max() <= 0:
        return []
    # the lines: runs of the profile over a quarter of its peak, split at their valleys
    on = prof > 0.25 * prof.max()
    lab, nl = ndi.label(on)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        ti = sl[0]
        if ti.stop - ti.start < 3:
            continue
        seg = prof[ti]
        tc = T[ti][int(np.argmax(seg))]
        half = max(3.0, (ti.stop - ti.start) / 2.0)
        # follow it outward from the middle: the centre of mass of its print in a window round the prediction
        pos, cen = [], []
        for direction in (1, -1):
            pred, prev_s = float(tc), 0.0
            slope = 0.0
            s0 = 0 if direction == 1 else -6
            for s_c in range(s0, int(direction * 0.95 * W), direction * 6):
                cols = (S >= s_c - 6) & (S < s_c + 6)
                rows = (T >= pred - half - 3) & (T <= pred + half + 3)
                win = R[np.ix_(rows, cols)]
                mass = win.sum()
                if mass < 0.15 * half * 12:
                    if abs(s_c) > 0.15 * W:
                        break
                    continue
                tm = float((win.sum(1) * T[rows]).sum() / mass)
                pos.append(s_c)
                cen.append(tm)
                if len(pos) >= 2:
                    slope = 0.7 * slope + 0.3 * (tm - pred) / max(abs(s_c - prev_s), 1)
                pred, prev_s = tm + slope * 6 * direction, s_c
        if len(pos) < 8:
            continue
        sv, tv = np.asarray(pos, float), np.asarray(cen, float)
        w = np.interp(tv, hw[:, 0], hw[:, 1])
        arc = np.sqrt(np.clip(1 - (sv / np.maximum(w, 1)) ** 2, 0, 1))
        A = np.c_[np.ones_like(sv), sv, arc]
        keep = np.ones(len(sv), bool)
        for _ in range(4):
            coef = np.linalg.lstsq(A[keep], tv[keep], rcond=None)[0]
            r = tv - A @ coef
            new = np.abs(r) <= max(2.0, 2.5 * 1.4826 * np.median(np.abs(r[keep])))
            if new.sum() < 8 or (new == keep).all():
                break
            keep = new
        span = float(sv[keep].max() - sv[keep].min())
        if span < min_len * 2 * W:
            continue
        # b: the arc's coefficient (+: lower in the middle, its ends higher: seen from above), as geometry.Bow draws it
        out.append({"t": round(float(coef[0] + coef[2]), 1), "b": round(float(coef[2]), 2),
                    "tilt_deg": round(float(np.degrees(np.arctan(coef[1]))), 2), "s_span": round(span, 1),
                    "rms": round(float(np.sqrt(np.mean(r[keep] ** 2))), 2), "n": int(keep.sum()),
                    "h": int(ti.stop - ti.start)})
    return out


def print_bow(crop_u8, M, cover, body, offset, size, pad=10.0):
    """The body's bow as the generated print draws it: line_bows on the frame's print map, the lines whose middle
    lies within the (registered) true print's height (pad px either side: a lid seam or a glass base just past it
    stay out), level (under 5 deg) and smooth (rms under 3 px); their bows fitted as b(t) = a + c t, weighted by
    their length. -> (profile [[t_top, b], [t_bottom, b]] or None, report)"""
    Lf = im.lum(im.srgb_to_lin(im.to_float(crop_u8)))
    P = print_map(Lf, M, size, True)[0] * M
    frame = {"centre": (np.asarray(body["centre"], float) - np.asarray(offset, float)).tolist(), "axis": body["axis"],
             "half_widths": body["half_widths"]}
    lines = line_bows(P, M, frame)
    c = np.asarray(frame["centre"], float)
    ax = np.asarray(body["axis"], float)
    ax = ax / np.linalg.norm(ax)
    ys, xs = np.nonzero(cover > 0.1)
    if not len(ys) or not lines:
        return None, {"lines": lines, "why": "no lines of print measured"}
    tc = (xs - c[0]) * ax[0] + (ys - c[1]) * ax[1]
    lo, hi = float(np.percentile(tc, 1)) - pad, float(np.percentile(tc, 99)) + pad
    use = [l for l in lines if lo <= l["t"] <= hi and abs(l["tilt_deg"]) < 5 and l["rms"] < 3]
    rep = {"lines": lines, "used": [l["t"] for l in use], "t_range": [round(lo, 1), round(hi, 1)]}
    if not use:
        return None, dict(rep, why="no level line of print within the print's height")
    t = np.array([l["t"] for l in use])                   # along the axis from the body's centre, as Bow's profile
    b = np.array([l["b"] for l in use])
    w = np.array([l["n"] for l in use], float)
    if len(use) >= 2 and np.ptp(t) > 20:
        A = np.c_[np.ones_like(t), t] * np.sqrt(w)[:, None]
        a0, a1 = np.linalg.lstsq(A, b * np.sqrt(w), rcond=None)[0]
    else:
        a0, a1 = float(np.average(b, weights=w)), 0.0
    t_top, t_bot = float(t.min()), float(t.max())
    prof = [[round(t_top, 1), round(float(a0 + a1 * t_top), 2)], [round(t_bot, 1), round(float(a0 + a1 * t_bot), 2)]]
    rep["profile"] = prof
    return prof, rep


def level_check(src_u8, out_u8, M, size, offset, max_deg=1.0):
    """The print's top edge against the generated print's (the user's note on a jar: "a bit skewed (top edge not
    aligned very well)", which the audit called clean). Per column across the body (in its own frame: its axis from
    the matte's sides), the topmost print in the result and in the frame, where the fix changed pixels; their
    difference fitted as a line, columns over 2 px off it left out (the layouts differ: a badge drawn higher). Its
    slope is the turn between the two prints' top lines. The generated print's top line ran with the jar's lid seam
    within 0.34 deg. Measured: v1's jar +1.58 deg (the skew the user saw), v2's +0.35; two bottles 0.0-0.55 (one
    approved as it was). -> audit findings ("ink_level"): WARN over max_deg, else one INFO with the numbers"""
    from . import bow as bw
    Ps, Po, changed = _print_maps(src_u8, out_u8, M, size)
    # only where the fix changed pixels (a jar's lid, untouched, has dark reflections above the print)
    inner = ndi.binary_erosion(M, iterations=4) & ndi.binary_dilation(changed, iterations=int(size))
    if inner.sum() < 100:
        return []
    c, ax, n = bw.body_axis(M)
    ys, xs = np.nonzero(inner)
    sv = (xs - c[0]) * n[0] + (ys - c[1]) * n[1]
    tv = (xs - c[0]) * ax[0] + (ys - c[1]) * ax[1]
    sb = np.round(sv).astype(int)
    lo = sb.min()

    def top(P):
        ink = P[ys, xs] > 0.5
        best = np.full(sb.max() - lo + 1, np.inf)
        np.minimum.at(best, sb[ink] - lo, tv[ink])
        return best

    ts, to = top(Ps), top(Po)
    ok = np.isfinite(ts) & np.isfinite(to)
    sc = np.arange(lo, sb.max() + 1)[ok]
    ts, to = np.where(ok, ts, 0.0), np.where(ok, to, 0.0)
    if len(sc) < 40:
        return []
    d = (to - ts)[ok]
    coef, keep = _robust_fit(sc.astype(float), d, 1, tol=2.0)
    if keep.sum() < 40:
        return []
    k = coef[0]
    span = float(sc[keep].max() - sc[keep].min())
    deg = float(np.degrees(np.arctan(k)))
    xs_k = c[0] + sc[keep] * n[0]
    box = [int(xs_k.min()) + offset[0], int(c[1] + np.median(ts[ok][keep]) - 6) + offset[1],
           int(xs_k.max()) + offset[0], int(c[1] + np.median(ts[ok][keep]) + 6) + offset[1]]
    text = (f"the print's top line turns {deg:+.2f} deg against the generated print's ({int(keep.sum())} columns over "
            f"{span:.0f} px: {abs(k) * span:.1f} px end to end)")
    if abs(deg) > max_deg:
        return [{"check": "ink_level", "level": "WARN", "px": int(keep.sum()), "box": box, "tilt_deg": round(deg, 2),
                 "text": text + ": skewed against the body's lines? (registration, align.bow, the body's axis)"}]
    return [{"check": "ink_level", "level": "INFO", "px": int(keep.sum()), "tilt_deg": round(deg, 2), "text": text}]


def shape_check(src_u8, out_u8, M, size, offset, min_px=4.0, share=0.1):
    """Each line's curve across the body against the generated print's (the user's note on a jar's v2: "the label
    still looks distorted compared to the original. like it doesn't respect the shape of the product", which ink_level,
    the top line's turn, passed at 0.35 deg). line_bows on both prints where the fix changed pixels; each of the
    frame's level, smooth lines matched to the result's nearest (within 12 px along the axis); a line is off when the
    two bows differ by more than max(min_px, share x the frame's bow). WARN when two lines or more and half of them
    are off, and the same way (the median difference over the tolerance at the median bow): the user saw every line
    flatter, where a bottle's generated lines wobble a few px either way. Measured: the jar's v2 (one bow from the
    matte's ends) 6 of 6 lines 12-44 px flatter than the frame's 73-88 (median -35), v1 (a fitted bow) 6 of 6 off
    (median +17), the fix 0 of 6 (0.4-5.1 px); the approved bottles 0-2 of 6 and 4-5 of 12 off, medians -2 to -3.
    -> audit findings ("ink_shape"): one WARN, else one INFO with the numbers"""
    from . import bow as bw
    Ps, Po, changed = _print_maps(src_u8, out_u8, M, size)
    region = ndi.binary_dilation(changed, iterations=int(size)) & M
    if region.sum() < 100:
        return []
    c, ax, n = bw.body_axis(M)
    frame = {"centre": c.tolist(), "axis": ax.tolist(), "half_widths": bw.half_widths(M, c, ax, n)}
    src = [l for l in line_bows(Ps * region, M, frame) if abs(l["tilt_deg"]) < 5 and l["rms"] < 3]
    res = line_bows(Po * region, M, frame)
    pairs = []
    for l in src:
        m = min(res, key=lambda q: abs(q["t"] - l["t"]), default=None)
        if m is not None and abs(m["t"] - l["t"]) <= 12:
            d = m["b"] - l["b"]
            pairs.append((l, m, d, abs(d) > max(min_px, share * abs(l["b"]))))
    if len(pairs) < 2:
        return []
    off = [p for p in pairs if p[3]]
    nums = ", ".join(f"{l['b']:+.0f}/{m['b']:+.0f}" for l, m, _, _ in pairs)
    text = (f"{len(off)} of {len(pairs)} lines of print curve unlike the generated print's (bow px, frame/result, "
            f"top to bottom: {nums})")
    d = float(np.median([p[2] for p in pairs]))
    if len(off) >= 2 and 2 * len(off) >= len(pairs) and \
            abs(d) > max(min_px, share * float(np.median([abs(p[0]["b"]) for p in pairs]))):
        ys, xs = np.nonzero(region)
        tv = (xs - c[0]) * ax[0] + (ys - c[1]) * ax[1]
        t0, t1 = min(p[0]["t"] for p in off) - 15, max(p[0]["t"] for p in off) + 15
        sel = (tv >= t0 - max(0.0, d) - abs(d)) & (tv <= t1 + abs(d))
        box = [int(xs[sel].min()) + offset[0], int(ys[sel].min()) + offset[1], int(xs[sel].max()) + offset[0],
               int(ys[sel].max()) + offset[1]] if sel.any() else None
        f = {"check": "ink_shape", "level": "WARN", "px": int(sum(p[1]["n"] for p in off)),
             "bow_px": [[round(p[0]["b"], 1), round(p[1]["b"], 1)] for p in pairs],
             "text": text + f": {'flatter' if d < 0 else 'more bowed'} by {abs(d):.0f} px (median), the body's "
                            "curve wrong at those heights? (align.bow, matte bow_from)"}
        return [dict(f, box=box)] if box else [f]
    return [{"check": "ink_shape", "level": "INFO", "px": len(pairs),
             "bow_px": [[round(p[0]["b"], 1), round(p[1]["b"], 1)] for p in pairs], "text": text}]


def rim_check(src_u8, out_u8, M, vis, size, offset):
    """Halos and lines along the print's and an occluder's edges (the user's note on a toner: "issue around the
    edges of the fish and on the right edge - we see a bit of silver line"), which the audit didn't flag:
      occluder  dark structure removed right at an occluder's edge (the object reaches past the occluder: a fish's eye
                ring filled as garbled print), or the body changed in its 3 px rim without print either side (the
                fill took the occluder's colour);
      line      a long straight line of new ink (1.5 closings or more, a few px thick) near the body's edge, where the
                frame had no print: the packshot's label edge or a seam laid on as print.
    -> audit findings ("ink_rim"), WARN from 12 px"""
    Ps, Po, changed = _print_maps(src_u8, out_u8, M, size)
    out = []
    ox, oy = offset

    def add(sel, what, minpx=12):
        lab, n = ndi.label(sel)
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            px = int((lab[sl] == i).sum())
            if px >= minpx:
                out.append({"check": "ink_rim", "level": "WARN", "px": px,
                            "box": [sl[1].start + ox, sl[0].start + oy, sl[1].stop - 1 + ox, sl[0].stop - 1 + oy],
                            "text": f"{px} px {what}"})

    if vis is not None and (vis < 0.5).any():
        occ = vis < 0.5
        removed = changed & (Ps > 0.3) & (Po < 0.15) & M
        lab, n = ndi.label(removed)
        touch = np.unique(lab[ndi.binary_dilation(occ, iterations=2) & (lab > 0)])
        add(np.isin(lab, touch) & (lab > 0), "of dark structure removed right at an occluder's edge: the object reaches "
            "past the occluder (grow it, add its fins or rim)")
        rim = ndi.binary_dilation(occ, iterations=3) & ~occ & M
        diff = np.abs(out_u8.astype(int) - src_u8.astype(int)).max(-1)
        add(rim & (diff > 20) & (Ps < 0.2) & (Po < 0.2), "of the body changed along an occluder's edge with no print "
            "either side: the fill took the occluder's colour?")
    # a thin line is anti-aliased: 0.12-0.42 of the body light across a toner's 2 px label edge
    n_run = max(30, int(1.5 * size))
    newink = changed & (Po > 0.2) & (Ps < 0.1) & M
    lines = ndi.binary_opening(newink, structure=np.ones((n_run, 1), bool)) | \
        ndi.binary_opening(newink, structure=np.ones((1, n_run), bool))
    near_edge = M & ~ndi.binary_erosion(M, iterations=2 * int(size))
    lab, n = ndi.label(lines)
    # not the edge of print the frame had beside it (a bar's edge 1 px off the generated bar's)
    beside = ndi.binary_dilation(Ps > 0.3, iterations=2)
    thin = [i for i, sl in enumerate(ndi.find_objects(lab), 1)
            if min(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) <= 6 and (lab[sl] == i)[near_edge[sl]].any()
            and beside[sl][lab[sl] == i].mean() < 0.2]
    add(np.isin(lab, thin), "of a straight line of new ink near the body's edge where the frame had no print: a "
        "label's edge, a seam or a rim laid on as print? (print_box)")
    return sorted(out, key=lambda f: -f["px"])[:6]


LABEL_INSET = 3      # text only: px kept clear inside the generated label's measured outline (its border, its shading)
INSET_SHARE = 0.5    # ... a generated mark goes whole when this share of it lies inside (a word crossing the outline)
DOUBLE_FAIL = 0.25   # double_check: the generated print left beside the new print, as a share of the new print's px


def double_check(src_u8, out_u8, M, size, offset, dark=True, share=DOUBLE_FAIL):
    """The generated print left showing beside the new print: print in the result, unchanged from the frame and
    within reach of the new strokes (8 px, or 8 % of the new print's extent), counted against the new print's own px.
    Where the new print lands on the generated print, the fill takes the garbled strokes under and around it; landed
    beside it, the garbled print shows next to the new one on the keyframe and every frame of the clip. FAIL past
    `share`. Measured: light print on a serum placed 33 px sideways of its generated print 0.60; registered 0.007; an
    approved jar 0.046. -> audit findings ("ink_double")"""
    Ps, Po, changed = _print_maps(src_u8, out_u8, M, size, dark)
    inner = ndi.binary_erosion(M, iterations=4)
    new = (Po > 0.3) & changed & inner
    if new.sum() < 50:
        return []
    ys, xs = np.nonzero(new)
    span = max(np.ptp(ys), np.ptp(xs))
    near = ndi.distance_transform_edt(~new) <= max(8.0, 0.08 * span)
    old = (Po > 0.3) & (Ps > 0.3) & ~ndi.binary_dilation(changed, iterations=2) & inner & near
    v = float(old.sum() / new.sum())
    text = (f"the generated print still shows beside the new print: {int(old.sum())} px, {100 * v:.1f} % of the new "
            f"print's {int(new.sum())} px")
    if v > share:
        oy, ox = np.nonzero(old)
        box = [int(ox.min()) + offset[0], int(oy.min()) + offset[1], int(ox.max()) + offset[0], int(oy.max()) + offset[1]]
        return [{"check": "ink_double", "level": "FAIL", "px": int(old.sum()), "box": box, "share": round(v, 3),
                 "text": text + f": over {100 * share:.0f} %, the new print sits beside the generator's, not on it "
                                "(the fit, the registration)"}]
    return [{"check": "ink_double", "level": "INFO", "px": int(old.sum()), "share": round(v, 3), "text": text}]


TEXT_HUE_FAIL = 25.0    # text_hue: a new line's letters this far (dE76, medians) from the generated letters they replace
TEXT_KEPT_WARN = 0.6    # text_kept: a line the gate kept as legible under this score (textline.LEGIBLE 0.5) is worth a look
LINE_WHOLE = 0.5        # text on a printed body: a generated line with this share of its letters gone goes whole
NEAR_RUN = 2.0          # ... and within this many line heights past a new line's ends, along its run
NEAR_NEW = 0.6          # text on a printed body: generated marks within this many new-line heights of the new letters go
TEXT_CHANGE_LEVELS = 2   # text_only: a pixel changed by more than this (8-bit) outside the letters and the fill moved
TEXT_ONLY_PX = 20        # ... FAIL from this many such px (the composite's own linear-light round trip moves 0-1)
TEXT_LEFT_FAIL = 0.15    # text_left: generated letters left in the text area, as a share of the new letters' px
TEXT_PLACED_MIN = 0.85  # text_missing: FAIL when less of the packshot's letters went on (cut at an edge, left out)
TEXT_OFF_FAIL = 0.4      # text_lines: FAIL when this share of the new lines (by their px) sits off the generated text
TEXT_COURSE_FAIL = 0.35  # text_course: the new lines run off their generated rows by this many deg (median by letters,
                         # each line's straight slope against its row's along its own run): FAIL. Calibrated on the user's
                         # calls: v11's squeeze tube 1.20 ("still skewed a bit"), v13's right dropper bottle 0.40 ("lines
                         # aren't aligned very well"); fixed 0.11 and 0.17. An approved bottle read 0.41 in v13: the gate
                         # is stricter than the eye (pitfall 115)
TEXT_COURSE_WARN = 0.6   # ... one line this far: WARN
TEXT_BOW_WARN = 0.15     # ... or bent against its row by this many line heights (v13's arched top line: 0.19): WARN
TEXT_INK_SHARE = 0.5     # text_ink: new letters' px darker than this share of the generated letters' darkest ink (2nd
                         # percentile; dark print) ...
TEXT_INK_PX = 12         # ... in pieces this big (px at 1x): FAIL
TEXT_COURSE_MIN_H = 8    # px: lines smaller than this (a 1,024 px frame's small print) aren't measured
# text_fit: FAIL where the text job's print didn't register onto the generated print (the
#   letters' size and place from the label's outline alone): the user's two "needs work" stills of v11-v12 on text
#   mode both had it (a right dropper bottle "a bit distorted, especially the logo and upper text": correlation 0.03,
#   letters 8-21 % narrow; a white bottle "applied completely wrong": 0.02); the three approved registered (0.32-0.46),
#   pitfall 114
TEXT_RIM_FAIL = 10       # text_rim: new letters' px (packshot px) taken from the packshot label's own edge (within
                         # max(3, 0.3 letter heights) of its outline): FAIL (the user, 2026-11-06: "I can see the label
                         # corner in lower left": a corner arc of 26 px drawn inside the label; every other label 0,
                         # pitfall 114)


def dark_any(passes):
    return any(pol == "dark" for pol, _ in passes)


def _course(Ps, Po, nl):
    """text_course's measure (pitfall 115): each new line (nl: the work box's new line map) 4 line heights long and
    TEXT_COURSE_MIN_H tall or more, along its own run (its principal axis, so a tilted label reads alike), the centre of
    the output's print (Po) and of the generated print (Ps) across the run, in bins of half a line height; their
    difference's straight slope (deg) and bow (line heights). -> [(line, deg, bow, px)]"""
    out = []
    for k in np.unique(nl[nl > 0]):
        m = nl == k
        ys, xs = np.nonzero(m)
        if len(xs) < 40:
            continue
        X = np.c_[xs - xs.mean(), ys - ys.mean()].astype(float)
        ev, ec = np.linalg.eigh(X.T @ X)
        d, n = ec[:, 1], ec[:, 0]
        d = -d if d[0] < 0 else d
        h = np.sqrt(ev[0] / len(xs)) * 3.5
        if np.sqrt(ev[1] / len(xs)) * 3.46 < 4 * h or h < TEXT_COURSE_MIN_H:
            continue
        reg = ndi.binary_dilation(m, iterations=max(1, int(0.35 * h)))
        ry, rx = np.nonzero(reg)
        t = (rx - xs.mean()) * d[0] + (ry - ys.mean()) * d[1]
        s_ = (rx - xs.mean()) * n[0] + (ry - ys.mean()) * n[1]
        b = np.floor((t - t.min()) / (0.5 * h)).astype(int)
        cen, mass = [], []
        for P in (Po, Ps):
            w = P[ry, rx]
            mass.append(np.bincount(b, weights=w))
            cen.append(np.bincount(b, weights=w * s_) / np.maximum(mass[-1], 1e-9))
        ok = (mass[0] > 0.3 * np.median(mass[0])) & (mass[1] > 0.3 * np.median(mass[1]))
        if ok.sum() < 6:
            continue
        tc = ((np.arange(len(ok)) + 0.5) * 0.5 * h + t.min())[ok]
        diff = (cen[0] - cen[1])[ok]
        deg = float(np.degrees(np.arctan(np.polyfit(tc, diff, 1)[0])))
        bow = float(np.polyfit(tc - tc.mean(), diff, 2)[0] * (np.ptp(tc) / 2) ** 2 / h)
        out.append((int(k) - 1, round(deg, 2), round(bow, 3), int(m.sum())))
    return out


def _lab_u8(u8):
    from .decide import _lab
    return _lab(u8)


def text_checks(crop_u8, out_u8, info, offset):
    """Text only (pitfall 111; the user's 2026-11-04 notes: "only make the texts on the label legible", "the line should
    match the product color like it does in the original"): every check a whole-label or whole-product route failed.
      text_only   pixels changed outside the new letters (2 px round them) and the fill under the generated letters:
                  the product's own colour, finish, outline or lines moved. FAIL.
      text_keep   the label's own marks the text step kept (its rules, bands, border, icons) changed. FAIL.
      text_left   generated letters still showing in the text area, beside the new ones. FAIL past TEXT_LEFT_FAIL.
      text_lines  new lines whose rows don't match the generated print's (off the generated text): WARN; FAIL past
                  TEXT_OFF_FAIL of the new letters (the generated text isn't where the true lines go).
    -> audit findings"""
    out = []
    ox, oy = offset
    src = crop_u8.astype(int)
    changed = np.abs(out_u8.astype(int) - src).max(-1) > TEXT_CHANGE_LEVELS
    a = info["alpha"]
    f_px = info.get("px_scale") or 1                    # the work scale (pitfall 113): the reaches are scene px

    def it(n):
        return max(1, int(round(n * f_px)))
    letters = ndi.binary_dilation(a > 0.02, iterations=it(2))
    allowed = letters | ndi.binary_dilation(info["fill"], iterations=it(1))

    def box(m):
        ys, xs = np.nonzero(m)
        return [int(xs.min()) + ox, int(ys.min()) + oy, int(xs.max()) + ox, int(ys.max()) + oy]

    away = changed & ~allowed
    n = int(away.sum())
    out.append({"check": "text_only", "level": "FAIL" if n >= TEXT_ONLY_PX else "INFO", "px": n,
                "box": box(away) if n else None,
                "text": f"{n} px changed outside the new letters and the fill under the generated ones" +
                        (": the product's own colour, finish or lines moved" if n >= TEXT_ONLY_PX else "")})
    keep = info.get("keep")
    if keep is not None:
        k = changed & keep & ~ndi.binary_dilation(a > 0.02, iterations=it(1))
        hit = keep & (a > 0.3)
        nk, nh = int(k.sum()), int(hit.sum())
        lvl = "FAIL" if nk >= TEXT_ONLY_PX or nh >= 4 * TEXT_ONLY_PX else "INFO"
        out.append({"check": "text_keep", "level": lvl, "px": nk + nh, "box": box(k | hit) if nk + nh else None,
                    "text": f"the label's own marks (its rules, bands, border, icons): {nk} px changed, {nh} px under "
                            "new letters" + (": a line or band of the generated label moved or was written over"
                                             if lvl == "FAIL" else "")})
    passes = info.get("passes") or [(info.get("polarity"), info)]
    new_px = max(int((a > 0.3).sum()), 1)
    left_px, off_px, tot_px, off_lines = 0, 0, 0, []
    for pol, inf in passes:
        tx = inf.get("text") or {}
        zone = inf.get("text_zone")
        if zone is not None and tx.get("gen_text") is not None:
            # the whole band, not only the text area: generated letters left anywhere on the label
            zone = zone | (tx["gen_text"] & ndi.binary_erosion(inf["matte"], iterations=it(4)))
        if zone is not None:
            size = int(inf.get("frame_size") or 15)
            Ps, Po, _ = _print_maps(crop_u8, out_u8, inf["matte"], size, pol == "dark")
            foot = ndi.binary_dilation(inf["alpha"] > 0.02, iterations=it(2))
            lk = zone & (Po > 0.35) & (Ps > 0.35) & ~changed & ~foot & ~ndi.binary_dilation(keep, iterations=it(1)) \
                if keep is not None else zone & (Po > 0.35) & (Ps > 0.35) & ~changed & ~foot
            left_px += int(lk.sum())
        place = tx.get("place") or {}
        nl = tx.get("new_line")
        for L in tx.get("lines_true", []):
            r = place.get(L["k"], {})
            px = int((nl == L["k"] + 1).sum()) if nl is not None else 1
            tot_px += px
            if not r.get("on_text", True):
                off_px += px
                off_lines.append(f"{pol} line {L['k']} (rows match {r.get('rows_ncc')})")
    # text_course: each new line along its generated row (the user, 2026-11-10: "lines aren't aligned very well, they
    # should be straightened to match the top and bottom edge of the label"; "generated rows are actually pretty lined
    # up almost all the time", pitfall 115)
    co = []
    for pol, inf in passes:
        tx = inf.get("text") or {}
        if tx.get("new_line") is None:
            continue
        size = int(inf.get("frame_size") or 15)
        Ps_, Po_, _ = _print_maps(crop_u8, out_u8, inf["matte"], size, pol == "dark")
        off = {k for k, r in (tx.get("place") or {}).items() if not r.get("on_text", True)}     # text_lines' own
        co += [(pol,) + r for r in _course(Ps_, Po_, tx["new_line"]) if r[0] not in off]
    if co:
        dv = np.array([abs(r[2]) for r in co])
        wv = np.array([r[4] for r in co], float)
        o_ = np.argsort(dv)
        med = float(dv[o_][np.searchsorted(np.cumsum(wv[o_]), 0.5 * wv.sum())])
        wa = max(co, key=lambda r: abs(r[2]))
        wb = max(co, key=lambda r: abs(r[3]))
        lvl = "FAIL" if med > TEXT_COURSE_FAIL else \
            "WARN" if abs(wa[2]) > TEXT_COURSE_WARN or abs(wb[3]) > TEXT_BOW_WARN else "INFO"
        out.append({"check": "text_course", "level": lvl, "deg": round(med, 2), "worst_deg": wa[2],
                    "worst_bow_lh": wb[3], "lines": [{"pass": p_, "line": k, "deg": d, "bow_lh": bw}
                                                     for p_, k, d, bw, _ in co],
                    "text": f"the new lines run {med:.2f} deg off their generated rows (median by letters; worst "
                            f"{abs(wa[2]):.2f}, {wa[0]} line {wa[1]}; most bent {abs(wb[3]):.2f} line heights, "
                            f"{wb[0]} line {wb[1]})" +
                            (": the text runs off the generated lines (skewed against the label)" if lvl == "FAIL"
                             else ": a line off or bent against its generated row" if lvl == "WARN" else "")})
    # text_ink: new letters inked twice (over generated ink the fill didn't take): darker than anything the generator
    # drew (a wordmark's i dot came out black, 8 levels, against the generated ink's 45, pitfall 113)
    stack, stack_lines = 0, []
    if not dark_any(passes):
        pass
    else:
        lum = out_u8.astype(float).mean(-1)
        for pol, inf in passes:
            tx = inf.get("text") or {}
            nl = tx.get("new_line")
            if nl is None or pol != "dark":
                continue
            core = (inf["alpha"] > 0.6)
            gt = tx.get("gen_text")
            src_l = crop_u8.astype(float).mean(-1)
            gsel = gt & (src_l < 128) if gt is not None else None
            if gsel is None or gsel.sum() < 30:
                continue
            ref = float(np.percentile(src_l[gsel], 2))       # the generated letters' darkest ink
            for k in np.unique(nl[nl > 0]):
                m = core & (nl == k)
                if m.sum() < 30:
                    continue
                lab_, n_ = ndi.label(m & (lum < TEXT_INK_SHARE * ref))
                if n_:
                    sz = ndi.sum(np.ones_like(lab_), lab_, np.arange(1, n_ + 1))
                    big = int(sz[sz >= TEXT_INK_PX * f_px * f_px].sum())
                    if big:
                        stack += big
                        stack_lines.append(int(k) - 1)
    out.append({"check": "text_ink", "level": "FAIL" if stack else "INFO", "px": stack,
                "text": (f"{stack} px of new letters far darker than the generated ink (lines {stack_lines}): inked twice, "
                         "over generated print the fill didn't take" if stack else "the new letters' ink is even")})
    # text_fit: the letters' size and place checked against the generated print (the registration kept)
    unreg = [(pol, (inf.get("text") or {}).get("register") or {}) for pol, inf in passes]
    unreg = [(pol, r) for pol, r in unreg if r.get("run") and not r.get("kept")]
    out.append({"check": "text_fit", "level": "FAIL" if unreg else "INFO",
                "text": ("the " + ", ".join(pol for pol, _ in unreg) + " print didn't register onto the generated "
                         "print (" + "; ".join(str(r.get("why")) for _, r in unreg) + "): the letters' size and place "
                         "come from the label's outline alone, unchecked (distorted letters, lines off the label)")
                if unreg else "the letters' size and place registered onto the generated print"})
    # text_rim: new letters taken from the packshot label's own edge (its outline, a rounded corner's shading)
    rim_px = sum(float((inf.get("text") or {}).get("rim_px") or 0.0) for _, inf in passes)
    out.append({"check": "text_rim", "level": "FAIL" if rim_px >= TEXT_RIM_FAIL else "INFO", "px": round(rim_px, 1),
                "text": f"{rim_px:.0f} px of the new letters come from the packshot label's own edge" +
                        (": the label's outline or corner drawn as print inside the generated label"
                         if rim_px >= TEXT_RIM_FAIL else "")})
    v = left_px / float(new_px)
    out.append({"check": "text_left", "level": "FAIL" if v > TEXT_LEFT_FAIL else "INFO", "px": left_px,
                "share": round(v, 3), "text": f"generated letters still showing in the text area: {left_px} px, "
                                             f"{100 * v:.1f} % of the new letters' {new_px} px" +
                                             (": garbled text left beside the new" if v > TEXT_LEFT_FAIL else "")})
    shares = [((inf.get("text") or {}).get("placed_share"), pol) for pol, inf in passes]
    for v_, pol in shares:
        if v_ is None:
            continue
        out.append({"check": "text_missing", "level": "FAIL" if v_ < TEXT_PLACED_MIN else "INFO", "share": round(v_, 3),
                    "text": f"{100 * v_:.0f} % of the packshot's {pol} letters went on" +
                            (f" (under {100 * TEXT_PLACED_MIN:.0f} %): part of the copy is cut off at the label's or "
                             "the frame's edge, or left out" if v_ < TEXT_PLACED_MIN else "")})
    # text_hue (a printed body): new letters off the generated letters' colour (a turned block's line took a glint's
    # ink: yellow, lavender)
    hue = []
    src_lab = _lab_u8(crop_u8)
    out_lab = _lab_u8(out_u8)
    for pol, inf in passes:
        tx = inf.get("text") or {}
        nl = tx.get("new_line")
        if tx.get("gen_kept") is None or nl is None or inf.get("grown") is None:
            continue
        a_p = inf["alpha"]
        lit = pol != "dark"
        for j in np.unique(nl[nl > 0]):
            mine = (nl == j) & (a_p > 0.5)
            old_ = inf["grown"] & ndi.binary_dilation(nl == j, iterations=it(2)) & inf["matte"]
            if mine.sum() < 12 or old_.sum() < 12:
                continue
            L_old = src_lab[..., 0][old_]
            core = old_.copy()
            core[old_] = (L_old >= np.percentile(L_old, 75)) if lit else (L_old <= np.percentile(L_old, 25))
            L_new = out_lab[..., 0][mine]
            mcore = mine.copy()
            mcore[mine] = (L_new >= np.percentile(L_new, 75)) if lit else (L_new <= np.percentile(L_new, 25))
            de = float(np.linalg.norm(np.median(out_lab[mcore], 0) - np.median(src_lab[core], 0)))
            if de > TEXT_HUE_FAIL:
                hue.append(f"{pol} line {int(j) - 1} (dE {de:.0f})")
    if any((inf.get("text") or {}).get("gen_kept") is not None for _, inf in passes):
        out.append({"check": "text_hue", "level": "FAIL" if hue else "INFO",
                    "text": ("new lines off the generated letters' colour: " + ", ".join(hue[:6])) if hue else
                            "the new lines in the generated letters' colour"})
    # text_kept (a printed body's gate): lines kept as generated because they already read as the packshot's, the
    # nearer the cut the likelier a letter of them is still wrong (a headline drawn with one wrong letter: 0.56)
    near = []
    for pol, inf in passes:
        for k, es in ((inf.get("text") or {}).get("gated") or {}).items():
            for e in (es if isinstance(es, list) else [es]):
                if e.get("why") == "legible" and (e.get("legible") or 0) < TEXT_KEPT_WARN:
                    near.append(f"{pol} line {k} ({e['legible']})")
    if near:
        out.append({"check": "text_kept", "level": "WARN",
                    "text": f"{len(near)} line(s) kept as generated, reading only partly like the packshot's (under "
                            f"{TEXT_KEPT_WARN:g}): " + ", ".join(near[:6]) + ": check their letters against the copy"})
    if off_lines:
        sh = off_px / float(max(tot_px, 1))
        out.append({"check": "text_lines", "level": "FAIL" if sh > TEXT_OFF_FAIL else "WARN", "share": round(sh, 3),
                    "text": f"{len(off_lines)} new line(s), {100 * sh:.0f} % of the new letters, sit off the generated "
                            "text (their rows don't match the generated print's: placed with their block): " +
                            ", ".join(off_lines[:6]) + (": the generated text isn't where the true lines go"
                                                        if sh > TEXT_OFF_FAIL else "")})
    return out


def dark_checks(crop_u8, out_u8, info, offset):
    """The checks the user's notes on print on the body became (pitfalls 97-100, 105): ink_place (the new print off
    the generated print along the body) and ink_double (the generated print left beside it), either polarity; for
    dark print also ink_level (the print's top line skewed against the generated print's), ink_shape (its lines
    curving unlike the generated print's) and ink_rim (an occluder's edge cut, a straight line of new ink near the
    body's edge). For run.py and the pack's QA Sheet. -> audit findings"""
    if info.get("text") is not None:
        return text_checks(crop_u8, out_u8, info, offset)
    if info.get("polarity") not in POLARITIES:
        return []
    size = int(info.get("frame_size") or 19)
    if info.get("polarity") == "light":
        return place_check(crop_u8, out_u8, info["matte"], size, offset, dark=False) + \
            double_check(crop_u8, out_u8, info["matte"], size, offset, dark=False)
    return place_check(crop_u8, out_u8, info["matte"], size, offset) + \
        double_check(crop_u8, out_u8, info["matte"], size, offset) + \
        level_check(crop_u8, out_u8, info["matte"], size, offset) + \
        shape_check(crop_u8, out_u8, info["matte"], size, offset) + \
        rim_check(crop_u8, out_u8, info["matte"], info.get("vis"), size, offset)


MIN_MATCH = 0.5      # place_check: the two prints' profiles correlate this well or better for a FAIL


def place_check(src_u8, out_u8, M, size, offset, max_px=12.0, max_share=0.10, dark=True):
    """Print on the body is registered onto the generated print, so it lands on it: the result's new print against the frame's
    own print along the body's axis (their row profiles across the body, correlated over shifts up to the new print's
    height). Off by over max_px, or max_share of that height, the label sits off the generated print and the
    generator's own print shows beside it: a FAIL, which stops the clip (a misplaced keyframe would carry through
    every frame); with the profiles matching under MIN_MATCH (another layout) a WARN. Measured: a squat jar whose mask
    fit was squeezed into two thirds of its mask, +144 px over a 229 px print (match 0.78); the same jar fitted right,
    -2 px (0.85); light print on a serum placed by its silhouette alone, 30 px beside its generated print, the
    garbled print showing on every frame. -> audit findings ("ink_place")"""
    from . import bow as bw
    Ps, Po, changed = _print_maps(src_u8, out_u8, M, size, dark)
    inner = ndi.binary_erosion(M, iterations=4)
    ys, xs = np.nonzero(inner)
    if len(ys) < 200:
        return []
    c, ax, _n = bw.body_axis(M)
    t = np.round((xs - c[0]) * ax[0] + (ys - c[1]) * ax[1]).astype(int)
    lo = int(t.min())
    pn = np.bincount(t - lo, weights=Po[ys, xs] * changed[ys, xs])
    pg = np.bincount(t - lo, weights=Ps[ys, xs], minlength=len(pn))
    rows = np.nonzero(pn > 0.05 * pn.max())[0] if pn.max() > 0 else []
    if len(rows) < 10 or pg.max() <= 0:
        return []
    span = int(rows.max() - rows.min())
    best, shift = -1.0, 0
    for d in range(-span, span + 1):
        a, b = pn[max(0, -d):len(pn) - max(0, d)], pg[max(0, d):len(pg) - max(0, -d)]
        v = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
        if v > best:
            best, shift = v, d
    lim = max(max_px, max_share * span)
    text = (f"the new print sits {shift:+d} px along the body from the generated print it replaces (its {span} px of rows; "
            f"profile match {best:.2f})")
    if abs(shift) > lim:
        yy, xx = np.nonzero(changed & inner)
        box = [int(xx.min()) + offset[0], int(yy.min()) + offset[1], int(xx.max()) + offset[0], int(yy.max()) + offset[1]]
        # a weak profile match (the generator drew another layout) can't tell where the print should be: a WARN
        return [{"check": "ink_place", "level": "FAIL" if best >= MIN_MATCH else "WARN", "px": int(abs(shift)), "box": box,
                 "shift_px": int(shift),
                 "text": text + f": over {lim:.0f} px, the label is off the product's print and the generator's own shows "
                                "beside it (the outline's fit to the mask, the registration's reach)"}]
    return [{"check": "ink_place", "level": "INFO", "px": int(abs(shift)), "shift_px": int(shift), "text": text}]


def layers(filled, out_u8, info):
    """For the PSD: the filled body and the ink in sRGB, and an sRGB-space alpha that flattens to `out_u8` (the
    composite blends in linear light). -> (base, card, alpha)"""
    base = im.lin_to_srgb(np.clip(filled, 0, 1))
    card = im.lin_to_srgb(np.clip(info["ink_abs"], 0, 1) if info.get("ink_abs") is not None else
                          np.clip(filled * info["k"], 0, 1))
    out = out_u8.astype(float) / 255.0
    lb, lc, lo = im.lum(base), im.lum(card), im.lum(out)
    pol = info.get("polarity")
    dark = pol == "dark" or pol == "mixed"
    if pol == "dark":                          # the ink below the body
        lb, lc, lo = -lb, -lc, -lo
    elif pol == "mixed":                       # per pixel: the dark lines' ink below, the light lines' above
        sg = np.where(np.asarray(info["k"], float).reshape(out.shape).mean(-1) < 1.0, -1.0, 1.0)
        lb, lc, lo = sg * lb, sg * lc, sg * lo
    alpha = np.where(lc - lb > 1e-3, np.clip((lo - lb) / np.maximum(lc - lb, 1e-3), 0, 1), 0.0) * info["region"]
    if dark:
        # one alpha can't blend a body toward near-black in sRGB the way the composite does in linear light (14
        # levels off): the ink layer's colour is solved per pixel so the layers flatten to the output
        a3 = alpha[..., None]
        solved = np.clip((out - (1 - a3) * base) / np.maximum(a3, 1e-6), 0, 1)
        card = np.where(a3 > 0.02, solved, card)
    return base, card, alpha
