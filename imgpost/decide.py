"""Auto Draft's settings decided from its inputs (the frame, the product's mask and the packshot), for a run with no
one in the loop: the print's polarity and box on the packshot (this module), the route and the fit (draft.auto).

The print on a packshot is its glyphs: small marks that stand out of the body's own light (above it for light print,
below it for dark), inside the product's outline, away from the outline's edge. A part of the product that differs
from the body (a lid, a cap, a glass base, a coloured band) or a rim reaches the outline, and so does everything joined
to it; a seam or a lid's shadow runs along such a part. Neither is print."""
import json

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
      ink        mixed print (glyphs both lighter and darker than the body, neither 2:1 ahead): warped whole (ink
                 off), whatever the view: an ink route lays one polarity and would read the other ink as body or
                 as garbled print. Light print: on the body (ink light; the frame's body and light kept). Dark print where the view
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
    if prep["mixed"]:
        out.update(ink="off", print_box=None, bow="print")
        why.append("route: print both lighter and darker than the body (mixed): an ink route lays one polarity, so the "
                   "packshot is warped whole, placed by the product's silhouette (ink off, bow from the generated "
                   "product's long lines)")
    elif pol == "dark" and front:
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


# a label laid over the body (pitfall 106): SAM 3's "label" on the packshot, and the body past its two ends
LABEL_SAME = 10.0     # dE: the body past both ends of a label is one material within this (a frosted bottle's 1.4, a
                      # tube round its panels 1.4; a jar's lid above its print and glass base below: 18.2, two parts)
LABEL_FIELD = 25.0    # dE: ... or the label's field stands off the body past both its ends by this much (white paper on
                      # amber glass 55-87, though the glass reads lighter at a thin shoulder than at a thick base: 37-45
                      # apart; a jar's printed band against its lid and glass base 11-16)
LABEL_GAP = 0.02      # the windows past the label's ends, along the product's axis: from this share of its length ...
LABEL_REACH = 0.08    # ... to this
LABEL_MIN = 0.01      # a window holding less of the product than this share: the print runs to the product's end
LABEL_FIND = 0.03     # SAM 3's label covers at least this share of the product, else there's none
LABEL_SIDE = 0.04     # the body's own colour beside the label at its rows, per side, over this share of the body's
                      # width: a panel on a face (flat); else the label wraps round the body (cylinder)


def _lab(u8):
    x = im.srgb_to_lin(im.to_float(u8))
    xyz = x @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    f = xyz / np.array([0.9505, 1.0, 1.089])
    f = np.where(f > 0.008856, np.cbrt(f), 7.787 * f + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def paper_label(ref_u8, band, ref_label):
    """Is the packshot's print on a label laid over the body (a paper label, printed panels) rather than straight on
    it? ref_label: SAM 3's "label" on the packshot (bool), taken within the product's outline (band). A label when the
    product continues past both of its ends along the axis, and either is one material there (LABEL_SAME: a frosted
    bottle above and below its white label, a tube round its panels) or the label's field stands well off the body on
    both sides (LABEL_FIELD: white paper on amber glass, which reads lighter at its shoulder than at its base). Print
    that runs to an end of the product (a serum's to its base, a tube's from its crimp) or a band between two
    different parts not much unlike it (a lid above, a glass base below) is on the body.
    -> (kind "cylinder" | "flat" | None, the label's box on the packshot [x0, y0, x1, y1] or None, measures)"""
    from . import bow as bw
    L = ref_label & band
    if L.sum() < LABEL_FIND * band.sum():
        return None, None, {"label": "none found on the packshot"}
    c, ax, n = bw.body_axis(band)
    ys, xs = np.nonzero(band)
    T = np.full(band.shape, np.nan)
    T[ys, xs] = (xs - c[0]) * ax[0] + (ys - c[1]) * ax[1]
    length = float(np.nanmax(T) - np.nanmin(T))
    t0, t1 = np.percentile(T[L], [1, 99])
    g, d = LABEL_GAP * length, LABEL_REACH * length
    lab = _lab(ref_u8)
    ends, meas = [], {"label_share": round(float(L.sum() / band.sum()), 3)}
    for name, win in (("before", band & ~L & (T >= t0 - d) & (T <= t0 - g)),
                      ("after", band & ~L & (T >= t1 + g) & (T <= t1 + d))):
        share = float(win.sum() / band.sum())
        meas[name + "_share"] = round(share, 3)
        ends.append(np.median(lab[win], 0) if share >= LABEL_MIN else None)
    if any(e is None for e in ends):
        meas["why"] = "the print runs to an end of the product"
        return None, None, meas
    same = float(np.linalg.norm(ends[0] - ends[1]))
    field = np.median(lab[ndi.binary_erosion(L, iterations=3)], 0)
    off = min(float(np.linalg.norm(field - e)) for e in ends)
    meas.update(ends_dE=round(same, 1), field_dE=round(off, 1))
    if same > LABEL_SAME and off < LABEL_FIELD:
        meas["why"] = "the parts past its two ends differ (a lid, a base) and the print's field isn't laid over them"
        return None, None, meas
    # flat or round: the body's own colour beside the label at its middle rows (a panel's margin) or not (it wraps)
    body = 0.5 * (ends[0] + ends[1])
    mid = band & (T >= t0 + 0.3 * (t1 - t0)) & (T <= t0 + 0.7 * (t1 - t0))
    beside = mid & ~ndi.binary_dilation(L, iterations=2) & (np.linalg.norm(lab - body, axis=-1) <= LABEL_SAME)
    width = float(mid.sum()) / max(1, len(np.unique(np.round(T[mid]))))
    side = float(beside.sum()) / max(1, len(np.unique(np.round(T[mid])))) / 2.0 / max(width, 1.0)
    meas["beside_share"] = round(side, 3)
    kind = "flat" if side > LABEL_SIDE else "cylinder"
    ly, lx = np.nonzero(L)
    box = [int(lx.min()), int(ly.min()), int(lx.max()) + 1, int(ly.max()) + 1]
    meas.update(kind=kind, box=box)
    return kind, box, meas


def scene_label(mask, label_mask, min_share=0.05):
    """The generated label on the frame: SAM 3's labels within the product's mask, as one label (the convex hull of
    the pieces, kept inside the mask: a tube's two printed panels make one). -> bool mask or None"""
    from scipy.spatial import ConvexHull
    from . import raster
    Lm = label_mask & mask
    if Lm.sum() < max(200, min_share * mask.sum()):
        return None
    ys, xs = np.nonzero(Lm)
    P = np.c_[xs, ys].astype(float)
    try:
        hull = P[ConvexHull(P).vertices]
    except Exception:                                # a degenerate piece: the pieces themselves
        return Lm
    return (raster.fill_polygon(mask.shape[0], mask.shape[1], [tuple(q) for q in hull]) > 0) & mask


def label_auto(scene_u8, mask, ref_u8, ref_alpha=None, outline=None, ref_label=None, label_mask=None):
    """Auto Draft's label mode decided (label "auto"): the packshot's print on a label laid over its body
    (paper_label) and SAM 3's label found on the frame within the product. -> (kind or None, the label's box on the
    packshot, the generated label's mask, the model for a flat label, notes)"""
    if ref_label is None or label_mask is None or mask is None:
        return None, None, None, None, []
    vm = view_match(scene_u8, mask, ref_u8, ref_alpha, outline)
    from . import raster
    band = raster.fill_polygon(ref_u8.shape[0], ref_u8.shape[1], [tuple(q) for q in vm["P"]]) > 0
    kind, box, meas = paper_label(ref_u8, band, ref_label)
    if kind is None:
        return None, None, None, None, [f"auto       label: off ({meas.get('why', meas.get('label'))}; "
                                        f"{json.dumps(meas)})"]
    sl = scene_label(mask, label_mask)
    if sl is None:
        return None, None, None, None, [
            f"auto       label: the packshot's print is on a label ({kind}, {json.dumps(meas)}), but SAM 3 found no "
            "label on the frame within the product: drafted without label mode",
            "auto       WARN a close call (see above): look at the keyframe before the clip"]
    model = "affine" if vm["iou"] >= VIEW_IOU else "homography"
    return kind, box, sl, model, [
        f"auto       label: {kind} (the packshot's print is on a label laid over the body, the same past both its "
        f"ends; {json.dumps(meas)}): the label only, its outline on the packshot {box}, the generated label SAM 3's "
        f"within the product ({int(sl.sum())} px)"]


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
