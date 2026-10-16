"""A first job from rough input: run.py draft and the ComfyUI pack's Auto Draft node both call draft().

Rough input in, a job to check and finish out: the packshot's outline traced (or given), landmark pairs refined,
edge lines proposed per straight side whose generated edge agrees, the old silhouette and the occluders snapped
to edges, the work box, colour and relight defaults set. With a product mask (SAM, measure.py mask) the outline is
fitted to the mask's contour instead of to image edges. Integer coordinates are pixel centres. Images are uint8
H x W x 3; a mask is bool H x W, scene-sized.
"""
import numpy as np

from . import autofit as af
from . import geometry as geo
from . import images as im
from . import outline as ol
from . import raster

ABOUT = "drafted from rough input: check every measurement before grading"
ROUGH_W, ROUGH_W_HELD = 0.15, 0.05   # the rough corners' weight, and once edges and landmarks hold the fit


class DraftError(ValueError):
    """The rough input can't make a draft; the message says what to give."""


def _pt_list(pts):
    return [tuple(float(v) for v in p) for p in pts]


def _hull_quad(mask):
    """The mask's convex hull cut down to 4 corners (each step drops the corner whose loss costs least area), in
    order round it (TL TR BR BL), turned to the order nearest upright."""
    from scipy.ndimage import binary_erosion
    from scipy.spatial import ConvexHull
    ys, xs = np.nonzero(mask & ~binary_erosion(mask))
    pts = np.c_[xs, ys].astype(float)
    hull = pts[ConvexHull(pts).vertices]
    while len(hull) > 4:
        a, b, c = np.roll(hull, 1, 0), hull, np.roll(hull, -1, 0)
        lost = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
        hull = np.delete(hull, int(np.argmin(lost)), 0)
    cen = hull.mean(0)
    q = hull[np.argsort(np.arctan2(hull[:, 1] - cen[1], hull[:, 0] - cen[0]))]   # clockwise on screen (y down)
    top = [abs(np.arctan2(*(np.roll(q, -k, 0)[1] - np.roll(q, -k, 0)[0])[::-1])) for k in range(4)]
    return np.roll(q, -int(np.argmin(top)), 0)


def mask_starts(mask, rmask, poly_ref, model="homography"):
    """Rough placements from a product mask alone, for the mask fit to start from: the packshot outline's bounding box
    onto the mask's box, and onto the mask's hull cut down to 4 corners (perspective: a box seen from above); then,
    when both shapes are elongated, the outline turned by the two masks' principal axes, scaled to the mask's area at
    its centroid (a tube at an angle). Every turn is the one nearer upright: a label's contour fits as well upside
    down. -> list of 3 x 3 matrices"""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise DraftError("the product mask is empty")
    P = np.asarray(poly_ref, float)
    (bx0, by0), (bx1, by1) = P.min(0), P.max(0)
    box = [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]
    x0, y0, x1, y1 = xs.min() - 0.5, ys.min() - 0.5, xs.max() + 0.5, ys.max() + 0.5
    starts = [af.fit_points(box, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], model=model),
              af.fit_points(box, _hull_quad(mask), model=model)]

    def moments(m):
        yy, xx = np.nonzero(m)
        w, V = np.linalg.eigh(np.cov(np.c_[xx, yy].astype(float).T))
        return len(xx), np.array([xx.mean(), yy.mean()]), w, V[:, 1]

    na, ca, wa, va = moments(rmask)
    nb, cb, wb, vb = moments(mask)
    if wa[1] > 1.2 * wa[0] and wb[1] > 1.2 * wb[0]:   # an axis is only defined on an elongated shape
        s = float(np.sqrt(nb / max(na, 1)))
        t = float(np.arctan2(vb[1], vb[0]) - np.arctan2(va[1], va[0]))
        t = (t + np.pi / 2) % np.pi - np.pi / 2          # the direction nearer upright
        R = s * np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        M = np.eye(3)
        M[:2, :2], M[:2, 2] = R, cb - R @ ca
        starts.append(M)
    return starts


def _mask_score(fit):
    """How well a mask fit sits on the contour: mean |residual| per outline sample, capped at 5 px (5 where none)."""
    r = np.abs(fit["residual"])
    return float(np.mean(np.where(np.isfinite(r), np.minimum(r, 5.0), 5.0)))


def draft(scene_u8, ref_u8, ref_alpha=None, quad=None, pairs=(), occluders=(), mask=None, mask_skip=None,
          model="homography", outline=None, cylinder=False, relight=True, suggest=False, about=ABOUT):
    """-> dict:
      job        the draft job without its file paths (scene, reference, output): scene_size, reference_size, roi,
                 reference_outline, align, old_silhouette, occluders, grade, finish, fill
      H          the draft's fit (reference px -> scene px)
      points     the landmark pairs as refined (dicts with label, ref, scene, w), corners the 4 rough corners or the
                 mask-fit outline points; props the proposed edge lines (kept or not, and why)
      sil_found  share of the old silhouette snapped to the generated edge; mask_note the mask fit's last note
      orient     the fit's lean and slope (af.orientation); white_box grade.ref_white_box or None; notes
      sheet      review tiles (packshot under the fit | scene) per landmark pair, a PIL image or None
    quad: 4 (x, y), where the corners of the packshot outline's bounding box land in the scene (TL TR BR BL, a few px
    off is fine). pairs: rough landmark pairs [((rx, ry), (sx, sy))]. occluders: [(side, [(x, y), ...])], side
    left / right / above / below (the occluder occupies that side of the polyline) or polygon. mask: the product's
    mask in the scene, bool; without quad and pairs the fit starts from its box and its principal axis. mask_skip: a
    polygon in reference px whose outline points aren't fitted to the mask (a part the mask runs on into).
    outline: None (traced from a packshot on a plain background or its alpha), a polygon [(x, y), ...] or a box
    (x0, y0, x1, y1) in reference px. Raises DraftError when the input can't make a draft."""
    scene, ref = im.to_float(scene_u8), im.to_float(ref_u8)
    Hs, Ws = scene_u8.shape[:2]
    notes = []

    # 1. the real outline
    if outline is not None and len(outline) >= 3 and np.ndim(outline[0]) == 1:
        outline = {"type": "polygon", "about": "given to the draft", "pts": [list(q) for q in _pt_list(outline)]}
    elif outline is not None:
        x0, y0, x1, y1 = (float(v) for v in outline)
        outline = {"type": "rounded_rect", "x0": x0, "y0": y0, "x1": x1, "y1": y1, "r_top": 0, "r_bottom": 0}
    elif ref_alpha is not None or af.on_plain_background(ref_u8):
        outline = {"type": "polygon", "about": "traced from the packshot by the draft: check it at the corners",
                   "pts": af.trace_outline(af.reference_mask(ref_u8, ref_alpha))}
    else:
        outline = {"type": "rounded_rect", "x0": 0, "y0": 0, "x1": ref_u8.shape[1] - 1, "y1": ref_u8.shape[0] - 1,
                   "r_top": 0, "r_bottom": 0}
        notes.append("packshot isn't on a plain background: the outline is the whole image, measure the real one")
    P = np.asarray(ol.outline_polygon(outline), float)
    rmask = raster.fill_polygon(ref_u8.shape[0], ref_u8.shape[1], [tuple(q) for q in P]) > 0
    bx0, by0 = P.min(0)
    bx1, by1 = P.max(0)
    box = np.array([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)])

    # 2. rough fit from the quad, the pairs alone, or the mask's box
    pairs = [(tuple(float(v) for v in r), tuple(float(v) for v in s)) for r, s in pairs]
    if mask is not None and mask.shape != scene_u8.shape[:2]:
        raise DraftError(f"the product mask is {mask.shape[1]}x{mask.shape[0]}, the scene is {Ws}x{Hs}")
    if quad is not None:
        quad = _pt_list(quad)
        if len(quad) != 4:
            raise DraftError("the quad needs 4 points: TL TR BR BL of the outline's bounding box in the scene")
        H0 = af.fit_points(box, quad, model=model)
    elif len(pairs) >= (3 if model == "affine" else 4):
        H0 = af.fit_points([p for p, _ in pairs], [q for _, q in pairs], model=model)
    elif mask is not None:
        H0 = None                                        # the mask alone: started from its box and its axes (2b)
    else:
        raise DraftError("give a quad, a product mask, or at least 4 landmark pairs (3 with model affine)")

    # 2b. a product mask: fit the outline to its contour, then work from that fit
    mask_note = None
    if mask is not None:
        skip_poly = _pt_list(mask_skip) if mask_skip else None
        skip = (lambda D: af.inside_polygon(D, skip_poly)) if skip_poly else None
        if H0 is not None:
            fit = af.fit_to_mask(mask, P, H0, model=model, skip=skip)
        else:
            fit = min((af.fit_to_mask(mask, P, h, model=model, skip=skip) for h in mask_starts(mask, rmask, P, model)),
                      key=_mask_score)
        H0 = fit["H"]
        mask_note = fit["notes"][-1]

    # 3. landmarks: refined pairs, optional suggestions, rough corners as weak placeholders
    points = []
    for i, r in enumerate(af.refine_points(scene, ref, H0, pairs)):
        points.append({"label": f"landmark {i}" + (f", ncc {r['ncc']}" if r["trusted"] else ": weak match, read it by hand"),
                       "ref": [round(v, 2) for v in r["ref"]], "scene": r["scene"], "w": 1.0 if r["trusted"] else 0.3})
    if suggest:
        for i, m in enumerate(af.suggest_landmarks(scene, ref, rmask, H0)):
            points.append({"label": f"suggested {i}, ncc {m['ncc']}", "ref": m["ref"], "scene": m["scene"], "w": 1.0})
    if mask is not None:
        # 8 points spread evenly along the outline (by length: picked by vertex, a 4-corner outline gave 3 distinct ones)
        Q = af.resample(P, float(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum()) / 8.0, closed=True)[:8]
        corners = [{"label": f"outline point {i} (fitted to the mask contour)", "ref": [round(float(v), 2) for v in q],
                    "scene": [round(float(v), 2) for v in geo.apply_H(H0, [q])[0]], "w": 1.0} for i, q in enumerate(Q)]
    else:
        corners = [{"label": f"rough {n} corner: replace with a landmark", "ref": [round(float(v), 2) for v in b],
                    "scene": [round(float(v), 2) for v in q], "w": ROUGH_W}
                   for n, b, q in zip(("top left", "top right", "bottom right", "bottom left"), box, geo.apply_H(H0, box))]

    def fit(lines):
        return geo.fit({"model": model, "points": points + corners, "lines": lines})[0]

    # 4. edge lines: straight sides whose generated edge agrees, re-proposed once under the refined fit
    props = af.propose_lines(scene, P, H0)
    H1 = fit([{"ref": l["ref"], "scene": l["scene"], "n": 12, "w": 2.0} for l in props if l["kept"]])
    props = af.propose_lines(scene, P, H1)
    lines = [{"label": f"outline side {i} (draft: {l['found']:.0%} of it on the edge, max {l['resid']} px)",
              "ref": l["ref"], "n": 12, "w": 2.0, "scene": l["scene"]} for i, l in enumerate(props) if l["kept"]]
    H = fit(lines)
    # weak points that disagree with the edges are ignored; placeholders fade once edges and landmarks hold the fit
    changed = False
    if points:
        for q, r in zip(points, geo.apply_H(H, [q["ref"] for q in points]) - np.array([q["scene"] for q in points])):
            if q["w"] < 1.0 and float(np.hypot(*r)) > 4.0:
                q["w"], q["label"] = 0.0, f"{q['label']} (off the fitted edges by {float(np.hypot(*r)):.0f} px: ignored)"
                changed = True
    if mask is None and (len(lines) >= 2 or sum(q["w"] >= 1.0 for q in points) >= 3):
        for c in corners:
            c["w"] = ROUGH_W_HELD
        changed = True
    if changed:
        H = fit(lines)

    # 5. old silhouette, 6. occluders
    sil, sil_found = af.snap_silhouette(scene, P, H)
    occ = []
    for i, (side, rough) in enumerate(occluders):
        rough = _pt_list(rough)
        if side == "polygon":
            S, T, _ = af.snap_polyline(scene, rough, "inside", step=6.0, closed=True, pick="first")
            occ.append({"label": f"occluder {i} (snapped by draft: {np.isfinite(T).all(1).mean():.0%})",
                        "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in np.where(np.isfinite(T), T, S)]})
        elif side in ("left", "right", "above", "below"):
            S, T, _ = af.snap_polyline(scene, rough, side, step=8.0, pick="first")
            occ.append({"label": f"occluder {i} (snapped by draft: {np.isfinite(T).all(1).mean():.0%})", "side": side,
                        "polyline": [[round(float(x), 2), round(float(y), 2)] for x, y in np.where(np.isfinite(T), T, S)]})
        else:
            raise DraftError(f"occluder {i}: its side is left, right, above, below or polygon, not {side!r}")

    # 7. work box, grade, finish, fill
    allp = np.vstack([geo.apply_H(H, P), sil])
    roi = [max(0, int(np.floor(allp[:, 0].min())) - 40), max(0, int(np.floor(allp[:, 1].min())) - 40),
           min(Ws, int(np.ceil(allp[:, 0].max())) + 41), min(Hs, int(np.ceil(allp[:, 1].max())) + 41)]
    grade = {"gain": "white_level", "white_percentile": 92, "window": 41, "erode": 4, "deep_erode": 20}
    wb = af.white_box(ref_u8, rmask)
    if wb:
        grade.update({"colour": "reference", "ref_white_box": wb})
    else:
        notes.append("no white area on the packshot: colour 'scene'; set grade.white_box on a nearby white surface "
                     "and colour 'reference' if the generated colours are off")
    if cylinder:
        grade["shading"] = {"bins": 20, "smooth": 2.0, "match_hue": 25, "clip": [0.4, 1.6],
                            "x_range": [round(float(bx0), 2), round(float(bx1), 2)], "limb": 0.65,
                            "about": "draft: measure the generated highlight's position and add spec / shadow"}
    elif relight:
        grade["relight"] = {"sigma": 14}
    job = {
        "about": about,
        "scene_size": [Ws, Hs], "reference_size": [ref_u8.shape[1], ref_u8.shape[0]],
        "roi": roi, "reference_outline": outline,
        "align": {"model": model, "points": points + corners, "y_only": [], "lines": lines},
        "old_silhouette": [[round(float(x), 2), round(float(y), 2)] for x, y in sil],
        "occluders": occ, "grade": grade,
        "finish": {"blur": "auto", "blur_probe": None, "grain": "auto", "edge_softness": 0.45, "rolloff": 0.85},
        "fill": {"dilate": 2},
    }
    return {"job": job, "H": H, "points": points, "corners": corners, "props": props, "sil_found": sil_found,
            "mask_note": mask_note, "orient": af.orientation(H, (bx0, by0, bx1, by1)), "white_box": wb, "notes": notes,
            "sheet": af.match_sheet(scene, ref, H, [p for p in points])}


def summary(d):
    """The draft's lines run.py prints (and the Auto Draft node shows): outline, points, sides, silhouette,
    orientation, grade, notes."""
    job, points = d["job"], d["points"]
    outline, grade, wb = job["reference_outline"], job["grade"], d["white_box"]
    out = [f"outline    {outline['type']}" + (f", {len(outline['pts'])} points traced" if outline["type"] == "polygon" else "")]
    trusted = sum(1 for q in points if q["w"] >= 1.0)
    if d["mask_note"] is not None:
        out.append(f"mask fit   {d['mask_note']}; 8 outline points from the fit (w 1)")
        out.append(f"points     {trusted} trusted, {len(points) - trusted} weak landmark pairs")
    else:
        out.append(f"points     {trusted} trusted, {len(points) - trusted} weak, plus 4 rough corners as placeholders (w 0.15)")
    for l in d["props"]:
        out.append(f"side       {'kept ' if l['kept'] else 'left '} {l['why']}")
    out.append(f"silhouette {d['sil_found']:.0%} snapped to the generated edge")
    o = d["orient"]
    out.append(f"orient     lean left {o['lean_left']:+.3f} right {o['lean_right']:+.3f} (fan {o['fan']:+.3f}), "
               f"slope top {o['slope_top']:+.3f} bottom {o['slope_bottom']:+.3f}")
    out.append(f"grade      colour {grade.get('colour', 'scene')}" + (f", ref_white_box {wb}" if wb else "") +
               (", shading template" if "shading" in grade else ", relight on" if "relight" in grade else ""))
    for n in d["notes"]:
        out.append(f"note       {n}")
    return out
