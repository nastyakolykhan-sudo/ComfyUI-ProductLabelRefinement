"""Auto Draft's settings decided from its inputs (the frame, the product's mask and the packshot), for a run with no
one in the loop: the print's polarity and box on the packshot (this module), the route and the fit (draft.auto).

The print on a packshot is its glyphs: small marks that stand out of the body's own light (above it for light print,
below it for dark), inside the product's outline, away from the outline's edge. A part of the product that differs
from the body (a lid, a cap, a glass base, a coloured band) or a rim reaches the outline, and so does everything joined
to it; a seam or a lid's shadow runs along such a part. Neither is print."""
import numpy as np
from scipy import ndimage as ndi

from . import images as im
from . import ink

EDGE_PX = 3          # a mark this close to the outline belongs to a part of the product, not the print
PART_PX = 4          # ... and so does a mark this close to such a part (a lid's shadow line under it)
MIN_GLYPH = 20       # px: smaller marks are noise
POLARITY_RATIO = 2.0  # one polarity's glyphs outnumber the other's this many times, or the polarity is mixed


def glyphs(ref_u8, band, dark):
    """The packshot's print marks of one polarity inside `band` (the outline's mask, bool). -> (glyph mask, number of
    glyphs, the parts' mask)"""
    Lr = im.lum(im.srgb_to_lin(im.to_float(ref_u8)))
    xs = np.nonzero(band.any(0))[0]
    big = int(max(15, (xs.max() - xs.min()) // 8)) | 1
    top, _body = ink._tophat(Lr, band, big, dark)
    p = np.percentile(top[band], 99.5)
    if p <= 0:
        return np.zeros(band.shape, bool), 0, np.zeros(band.shape, bool)
    marks = (top > 0.5 * p) & band
    lab, n = ndi.label(marks)
    edge = band & ~ndi.binary_erosion(band, iterations=EDGE_PX)
    touch = np.unique(lab[edge & marks])
    parts = np.isin(lab, touch[touch > 0])
    near = set(np.unique(lab[ndi.binary_dilation(parts, iterations=PART_PX) & marks]).tolist())
    size = ndi.sum(marks, lab, np.arange(1, n + 1))
    keep = [i for i in range(1, n + 1) if size[i - 1] >= MIN_GLYPH and i not in near]
    return np.isin(lab, keep), len(keep), parts


def polarity(ref_u8, band):
    """Light print (lighter than the body it sits on) or dark: the polarity whose glyphs outnumber the other's.
    -> ("light" | "dark", {"light": n, "dark": n, "mixed": bool})"""
    nl, nd = glyphs(ref_u8, band, False)[1], glyphs(ref_u8, band, True)[1]
    pol = "light" if nl > nd else "dark"
    hi, lo = max(nl, nd), min(nl, nd)
    return pol, {"light_glyphs": nl, "dark_glyphs": nd, "mixed": hi < POLARITY_RATIO * max(lo, 1)}


def print_box(ref_u8, band, dark):
    """The print's box on the packshot [x0, y0, x1, y1]: its glyphs' extent with a few px to spare (0.8 % of the
    outline's width, 4 px at least), kept off the product's parts. -> (box or None, {"glyphs", "pad_px"})"""
    G, n, parts = glyphs(ref_u8, band, dark)
    if not n:
        return None, {"glyphs": 0}
    ys, xs = np.nonzero(G)
    bx = np.nonzero(band.any(0))[0]
    pad = max(4, int(round(0.008 * (bx.max() - bx.min()))))
    gx0, gy0, gx1, gy1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    x0, y0, x1, y1 = gx0 - pad, gy0 - pad, gx1 + pad, gy1 + pad
    # each side's spare px stop short of a part's marks (a lid's rim above, a glass base below)
    py, px = np.nonzero(parts)
    rows = (py >= gy0) & (py < gy1)          # beside the print (a corner's part belongs to the side above or below)
    cols = (px >= gx0) & (px < gx1)
    left, right = px[rows & (px >= x0) & (px < gx0)], px[rows & (px >= gx1) & (px < x1)]
    top, bottom = py[cols & (py >= y0) & (py < gy0)], py[cols & (py >= gy1) & (py < y1)]
    x0 = max(x0, int(left.max()) + 1) if len(left) else x0
    x1 = min(x1, int(right.min())) if len(right) else x1
    y0 = max(y0, int(top.max()) + 1) if len(top) else y0
    y1 = min(y1, int(bottom.min())) if len(bottom) else y1
    H, W = band.shape
    box = [int(max(x0, 0)), int(max(y0, 0)), int(min(x1, W)), int(min(y1, H))]
    return box, {"glyphs": n, "pad_px": pad}


# the packshot's outline, fitted to the product's mask, overlaps it this much or more: the frame shows the product from
# the packshot's view (front on, as packshots are taken)
VIEW_IOU = 0.97
VIEW_BORDER = 0.01   # within this of VIEW_IOU the call is close: the report says so


def view_match(scene_u8, mask, ref_u8, ref_alpha=None, outline=None):
    """How well the packshot's silhouette, fitted to the product's mask by an affine fit (Auto Draft's own mask fit),
    covers the mask. -> {"iou", "mean_px": the fit's mean residual per outline sample, "outline": the draft's}"""
    from . import draft as dr, geometry as geo, outline as ol, raster
    d = dr.draft(scene_u8, ref_u8, ref_alpha, mask=mask, model="affine", outline=outline)
    P = np.asarray(ol.outline_polygon(d["job"]["reference_outline"]), float)
    W = raster.fill_polygon(mask.shape[0], mask.shape[1], [tuple(q) for q in geo.apply_H(d["H"], P)]) > 0
    iou = float((W & mask).sum() / max((W | mask).sum(), 1))
    return {"iou": round(iou, 4), "outline": d["job"]["reference_outline"], "P": P}


def settings(scene_u8, mask, ref_u8, ref_alpha=None, outline=None):
    """Auto Draft's route and fit from its inputs, for any of model, ink, print_box, bow left at "auto":
      polarity   the packshot's print, lighter or darker than its body (its glyphs, polarity()).
      view       the packshot's silhouette, fitted to the product's mask, covers it (IoU >= VIEW_IOU): the frame shows
                 the product as the packshot does, front on. Else it's seen from another angle (from above, a lid's
                 top in the mask; turned).
      model      affine where the view matches (a silhouette pins no perspective, and the bow is measured on top);
                 homography where it doesn't.
      ink        light print: on the body (ink light; the frame's body and light kept). Dark print where the view
                 matches: the packshot warped whole by the fit (ink off), its print placed by the product's own
                 silhouette, not by the generated print, which the generator draws off its place. Dark print seen
                 from another angle: on the body (ink dark), the print registered onto the generated print, which
                 sits on the body as the view draws it.
      print_box  on the body: the print's glyphs' extent on the packshot (print_box()).
      bow        warped whole: "print" (the generated product's long lines against the packshot's, as warped);
                 on the body: from the matte's two ends ("matte"; dark print keeps it only where the print registers
                 better, and also reads each line of the generated print).
    -> {"model", "ink", "print_box", "bow", "why": [lines], "measures": {...}, "close": bool}"""
    if mask is None:
        raise ValueError("auto settings need the product's mask (SAM 3's, or a given one)")
    vm = view_match(scene_u8, mask, ref_u8, ref_alpha, outline)
    from . import outline as ol, raster
    band = raster.fill_polygon(ref_u8.shape[0], ref_u8.shape[1], [tuple(q) for q in vm["P"]]) > 0
    pol, prep = polarity(ref_u8, band)
    front = vm["iou"] >= VIEW_IOU
    close = abs(vm["iou"] - VIEW_IOU) < VIEW_BORDER
    why = [f"view: the packshot's silhouette fitted to the mask covers it {vm['iou']:.3f} (front on at "
           f"{VIEW_IOU:g} or more): {'the packshot' + chr(39) + 's view' if front else 'another angle'}" +
           (" (close to the limit: check the route)" if close else ""),
           f"print: {pol} ({prep['light_glyphs']} light glyphs, {prep['dark_glyphs']} dark on the packshot" +
           (", mixed: check the polarity)" if prep["mixed"] else ")")]
    out = {"model": "affine" if front else "homography"}
    if pol == "dark" and front:
        out.update(ink="off", print_box=None, bow="print")
        why.append("route: dark print seen front on: the packshot warped whole, placed by the product's silhouette "
                   "(ink off, bow from the generated product's long lines)")
    else:
        box, brep = print_box(ref_u8, band, pol == "dark")
        out.update(ink=pol, print_box=box, bow="matte")
        why.append(f"route: {pol} print on the body (ink {pol}), print box {box} ({brep.get('glyphs', 0)} glyphs, "
                   f"{brep.get('pad_px', 0)} px spare), bow from the matte's ends" +
                   (", registered onto the generated print" if pol == "dark" else ""))
    why.append(f"fit: {out['model']}")
    out.update(why=why, close=close or prep["mixed"],
               measures={"view_iou": vm["iou"], "polarity": pol, **prep})
    return out


def resolve(scene_u8, mask, ref_u8, ref_alpha=None, outline=None, model="auto", ink="auto", print_box="auto",
            bow="auto"):
    """Auto Draft's model, ink, print_box and bow with each "auto" decided (settings()) and the rest as given.
    print_box "auto" with ink given: that polarity's glyph box; bow "auto": warped whole "print", on the body the
    matte's ends (dark print: the engine's own default, None here; light: the profile, measured now).
    -> (model, ink or None, print_box or None, bow (text, spec dict or None), notes: the lines for the report)"""
    from . import bow as bw
    if mask is None:
        raise ValueError("auto (model, ink, print_box or bow) needs the product's mask (connect product_mask)")
    s = settings(scene_u8, mask, ref_u8, ref_alpha, outline)
    notes = [f"auto       {w}" for w in s["why"]]
    if s["close"]:
        notes.append("auto       WARN a close call (see above): look at the keyframe before the clip")
    model = s["model"] if model == "auto" else model
    ink = s["ink"] if ink == "auto" else (ink or "off")
    if print_box == "auto":
        if ink == s["ink"]:
            print_box = s["print_box"]
        elif ink in ("light", "dark"):
            band = view_match(scene_u8, mask, ref_u8, ref_alpha, outline)["P"]
            from . import raster
            band = raster.fill_polygon(ref_u8.shape[0], ref_u8.shape[1], [tuple(q) for q in band]) > 0
            print_box = _print_box(ref_u8, band, ink == "dark")[0]
        else:
            print_box = None
    if bow == "auto":
        bow = "print" if ink == "off" else "matte"
    if bow == "matte":
        if ink == "light":
            prof, _rep = bw.matte_ends(mask)
            bow = {"b": prof, "from": "matte"} if prof is not None and max(abs(prof[0][1]), abs(prof[1][1])) >= 1.0 \
                else None
            notes.append("auto       bow " + ("from the matte's ends: " + ", ".join(f"t {t:g}: {b:+g}" for t, b in prof)
                                             if bow else "none: the matte's ends are level (seen from level)"))
        else:
            bow = None                       # dark print: the engine's own (the matte's ends, kept where it registers)
    return model, (None if ink == "off" else ink), print_box, bow, notes


_print_box = print_box
