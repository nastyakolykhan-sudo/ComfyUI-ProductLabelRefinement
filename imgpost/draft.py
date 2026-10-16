"""A first job from rough input: run.py draft and the ComfyUI pack's Auto Draft node both call draft().

Rough input in, a job to check and finish out: the packshot's outline traced (or given), landmark pairs refined,
edge lines proposed per straight side whose generated edge agrees, the old silhouette and the occluders snapped
to edges, the work box, colour and relight defaults set. With a product mask (SAM, measure.py mask) the outline is
fitted to the mask's contour instead of to image edges. Integer coordinates are pixel centres. Images are uint8
H x W x 3; a mask is bool H x W, scene-sized.

Label mode (label "cylinder" or "flat"): the label only, by the stills recipe for labels on bottles, jars and tubes
(measuring.md "Labels on bottles and jars"). The quad is the generated label's 4 corners; its own outline is measured
(af.label_outline, each side walked along the curve it bows to) and becomes the visible shape ("generated"); its
sides become the fit's lines, the side where it meets the bottle's silhouette a limb. "cylinder" takes the bottle's
axis and radius from the packshot's silhouette at the label's height, the drop shadow left out.
"""
import numpy as np

from scipy.ndimage import gaussian_filter1d, minimum_filter

from . import autofit as af
from . import geometry as geo
from . import images as im
from . import masks as mk
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


def _snap_occluders(scene, occluders):
    """The rough occluders snapped to their own edges, as job entries."""
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
    return occ


def _mask_score(fit):
    """How well a mask fit sits on the contour: mean |residual| per outline sample, capped at 5 px (5 where none)."""
    r = np.abs(fit["residual"])
    return float(np.mean(np.where(np.isfinite(r), np.minimum(r, 5.0), 5.0)))


def draft(scene_u8, ref_u8, ref_alpha=None, quad=None, pairs=(), occluders=(), mask=None, mask_skip=None,
          model="homography", outline=None, cylinder=False, relight=True, suggest=False, about=ABOUT, label=None):
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
    (x0, y0, x1, y1) in reference px. label: None (or "off"), "cylinder" or "flat": the label mode (label_draft).
    Raises DraftError when the input can't make a draft."""
    if label not in (None, "off"):
        return label_draft(scene_u8, ref_u8, ref_alpha, quad=quad, outline=outline, kind=label, pairs=pairs,
                           occluders=occluders, mask=mask, model=model, about=about)
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
    occ = _snap_occluders(scene, occluders)

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


# ---- label mode ----

LABEL_KINDS = ("cylinder", "flat")
CORNERS = ("TL", "TR", "BR", "BL")
SIDE_TOL = 12.0      # deg: an outline sample belongs to a side when its normal is this close to the side's
LIMB_TOL = 4.0       # px: a packshot label side this close to the bottle's limb (or past it) wraps on round it
SQUEEZE_OK = 0.75    # print at its furthest shown this wide or wider passed review (audit.py squeeze: 0.72 warns)
SMALL = 250.0        # px: a generated label smaller than this (square root of its area) takes the small-frame settings
_ARRIVE = {"TL": ("left", "top"), "TR": ("top", "right"), "BR": ("right", "bottom"), "BL": ("bottom", "left")}
_ENDS = {"top": ("TL", "TR"), "right": ("TR", "BR"), "bottom": ("BR", "BL"), "left": ("BL", "TL")}


def _area(Q):
    Q = np.asarray(Q, float)
    return 0.5 * float(np.sum(Q[:, 0] * np.roll(Q[:, 1], -1) - np.roll(Q[:, 0], -1) * Q[:, 1]))


def _meet(L1, L2):
    return np.linalg.solve(np.array([L1[:2], L2[:2]]), -np.array([L1[2], L2[2]]))


def _outline_runs(P):
    """An upright packshot label's outline as its 4 sides: per side the longest run of outline samples (1 px apart)
    whose outward normal is within SIDE_TOL of up, right, down or left; top runs left to right, right top to bottom,
    bottom right to left, left bottom to top."""
    Q = af.resample(np.asarray(P, float), 1.0, closed=True)
    T = np.roll(Q, -2, 0) - np.roll(Q, 2, 0)
    T /= np.maximum(np.linalg.norm(T, axis=1), 1e-9)[:, None]
    N = np.c_[T[:, 1], -T[:, 0]] if _area(Q) > 0 else np.c_[-T[:, 1], T[:, 0]]          # outward
    runs = {}
    for name, d, ahead in (("top", (0, -1), lambda r: r[-1, 0] - r[0, 0]), ("right", (1, 0), lambda r: r[-1, 1] - r[0, 1]),
                           ("bottom", (0, 1), lambda r: r[0, 0] - r[-1, 0]), ("left", (-1, 0), lambda r: r[0, 1] - r[-1, 1])):
        ok = N @ np.array(d, float) >= np.cos(np.radians(SIDE_TOL))
        if not ok.any() or ok.all():
            raise DraftError(f"the label's outline on the packshot has no clear {name} side: give the label upright, "
                             "with its 4 sides")
        best, cur = [], []
        for i in (np.arange(len(Q)) + int(np.argmin(ok))) % len(Q):     # from a sample off the side: no run wraps round
            if ok[i]:
                cur.append(i)
            else:
                best, cur = (cur if len(cur) > len(best) else best), []
        r = Q[cur if len(cur) > len(best) else best]
        runs[name] = r if ahead(r) >= 0 else r[::-1]
    return runs


def label_frame(P):
    """The packshot label: its sides' straight parts (rounded corners cut off), the sharp corners TL TR BR BL (the
    sides extended) and each corner's radius (from where the straight runs end)."""
    R = _outline_runs(P)
    end = lambda r, first: geo.line_through(r[:max(4, len(r) // 4)] if first else r[-max(4, len(r) // 4):])
    C = {c: _meet(end(R[a], False), end(R[b], True)) for c, (a, b) in _ARRIVE.items()}
    rad = {c: max(0.0, (float(np.linalg.norm(R[a][-1] - C[c])) + float(np.linalg.norm(R[b][0] - C[c]))) / 2
                  / (1 - np.sin(np.radians(SIDE_TOL))) - 0.5) for c, (a, b) in _ARRIVE.items()}
    sides = {}
    for k, (c0, c1) in _ENDS.items():
        r = R[k]
        keep = (np.linalg.norm(r - C[c0], axis=1) >= rad[c0]) & (np.linalg.norm(r - C[c1], axis=1) >= rad[c1])
        sides[k] = r[keep] if keep.sum() >= 2 else r
    return {"sides": sides, "corners": C, "radii": rad}


def _snap_sides(ref, F, search=3.0, step=2.0):
    """The packshot label's sides moved onto the packshot's own edges: outlines are drawn 0.5 to 1.2 px inside them,
    and a bottle's fit turns the print on less. Per sample the nearest edge within `search` px along the normal;
    samples more than 1 px off their side's median offset take the median."""
    cen = np.mean(list(F["corners"].values()), axis=0)
    out = {}
    for k, S in F["sides"].items():
        S = af.resample(np.asarray(S, float), step)
        T = np.gradient(S, axis=0)
        T /= np.maximum(np.linalg.norm(T, axis=1), 1e-9)[:, None]
        N = np.c_[T[:, 1], -T[:, 0]]
        N = np.where((((S - cen) * N).sum(1) < 0)[:, None], -N, N)
        hit = af.snap_points(ref, S, N, search=search, pick="nearest")[0]
        off = ((hit - S) * N).sum(1)
        ok = np.isfinite(off)
        if ok.sum() >= max(3, 0.3 * len(S)):
            med = float(np.median(off[ok]))
            ok &= np.abs(off - med) <= 1.0
            S = np.where(ok[:, None], hit, S + med * N)
        out[k] = S
    return out


def _row_cross(P, y):
    Q = np.vstack([P, P[:1]])
    xs = [x0 + (y - y0) * (x1 - x0) / (y1 - y0) for (x0, y0), (x1, y1) in zip(Q[:-1], Q[1:])
          if (y0 - y) * (y1 - y) <= 0 and y0 != y1]
    return (min(xs), max(xs)) if xs else (None, None)


def _steps(L, x0, x1, g_min=2.0, c_min=5.0):
    """Steps along a luminance row (0..255) between x0 and x1 -> [(x at the 50 % level, contrast, 10-90 % width)]."""
    s = gaussian_filter1d(L, 0.8)
    g = np.gradient(s)
    out = []
    for x in range(max(2, x0), min(len(L) - 3, x1)):
        if not (abs(g[x]) >= abs(g[x - 1]) and abs(g[x]) > abs(g[x + 1]) and abs(g[x]) >= g_min):
            continue
        a = b = x
        while a > 1 and np.sign(g[a - 1]) == np.sign(g[x]) and abs(g[a - 1]) > 0.15 * abs(g[x]):
            a -= 1
        while b < len(L) - 2 and np.sign(g[b + 1]) == np.sign(g[x]) and abs(g[b + 1]) > 0.15 * abs(g[x]):
            b += 1
        lo, c = L[a - 1], float(L[b + 1] - L[a - 1])
        if abs(c) < c_min:
            continue
        xs, seg = np.arange(a - 1, b + 2), L[a - 1:b + 2]

        def cross(f):
            k = np.nonzero(np.diff(np.sign(seg - (lo + f * c))))[0]
            return float(xs[k[0]] + (lo + f * c - seg[k[0]]) / (seg[k[0] + 1] - seg[k[0]])) if len(k) else float(x)
        out.append((cross(0.5), c, abs(cross(0.9) - cross(0.1))))
    return out


def bottle_limbs(ref_u8, ref_alpha, P, rows=24, soft=4.0):
    """The bottle's silhouette on the packshot at the label's height, per side: walking out from the label's side,
    the last step before the backdrop. A drop shadow's edge is soft (10-90 % wider than `soft` px) and a strong step
    up to the backdrop: the limb is then the last sharp step inside the shadow, where the label or glass meets it
    (pitfall 64). On a cut-out packshot the limb is where its alpha crosses half. -> dict(axis_x, radius, limbs {side: x}, spread {side: px over the rows}, shadow {side: bool})"""
    cut = ref_alpha is not None and ref_alpha.min() < 255   # a cut-out: the silhouette is its alpha, no shadow
    if cut:
        L, bgd = ref_alpha.astype(float), ref_alpha < 128
    else:
        bg, d = af.background(ref_u8)
        bgd = np.linalg.norm(ref_u8.astype(float) - bg, axis=-1) <= af._bg_tol(d)
        L = ref_u8.astype(float) @ af.LUMA
    P = np.asarray(P, float)
    y0, y1 = P[:, 1].min(), P[:, 1].max()
    found, shadow = {"left": [], "right": []}, {"left": 0, "right": 0}
    for y in np.linspace(y0 + 0.15 * (y1 - y0), y1 - 0.15 * (y1 - y0), rows):
        yi = int(round(y))
        xl, xr = _row_cross(P, yi)
        if xl is None:
            continue
        for side, xs, sgn in (("left", xl, -1), ("right", xr, 1)):
            run = (lambda x_: bgd[yi, x_:x_ + 4].all()) if sgn > 0 else (lambda x_: bgd[yi, max(x_ - 3, 0):x_ + 1].all())
            x = int(round(xs)) - sgn * 6                     # from inside the label's side, out to 4 px of backdrop
            while 3 < x < L.shape[1] - 4 and not run(x):
                x += sgn
            lo, hi = (x - 3, int(round(xs)) + 12) if sgn < 0 else (int(round(xs)) - 12, x + 3)
            st = sorted(_steps(L[yi], max(lo, 1), min(hi, L.shape[1] - 2)), key=lambda t: sgn * t[0])   # in -> out
            if not st:
                continue
            last = st[-1]
            if not cut and last[2] > soft and len(st) > 1 and sgn * last[1] >= 25.0:      # a drop shadow's edge
                shadow[side] += 1
                inner = [t for t in st[:-1] if t[2] <= 2.5 * soft]
                if inner:
                    found[side].append(inner[-1][0])
            else:
                found[side].append(last[0])
    out = {"limbs": {}, "spread": {}, "shadow": {}}
    for side in ("left", "right"):
        v = np.asarray(found[side])
        if len(v) < 4:
            raise DraftError(f"the bottle's {side} edge wasn't found on the packshot at the label's height: is the "
                             "packshot on a plain backdrop, and the outline the label's?")
        v = v[np.abs(v - np.median(v)) <= 2.0]
        out["limbs"][side] = round(float(np.median(v)), 2)
        out["spread"][side] = round(float(np.ptp(v)), 2)
        out["shadow"][side] = shadow[side] > rows // 3
    out["axis_x"] = round((out["limbs"]["left"] + out["limbs"]["right"]) / 2, 2)
    out["radius"] = round((out["limbs"]["right"] - out["limbs"]["left"]) / 2, 2)
    return out


def _limb_floor(ref_u8, ref_alpha, outline, axis_x, r, wrap, base=80.0):
    """cylinder.limb_min: the smallest silhouette angle at which the packshot's print reaching furthest round, on a
    side the label wraps past, still shows SQUEEZE_OK of its width (cos(a + 90 - L) / cos(a), the audit's squeeze
    check), never under `base`. -> (limb_min, print reach per side in deg)"""
    from . import audit as au
    reach = au.print_reach(axis_x, r, ol.RefMask(outline, ref_u8, ref_alpha), ref_u8.shape,
                           im.srgb_to_lin(im.to_float(ref_u8))) or {}
    need = base
    for side in wrap:
        if side in reach:
            a = reach[side]
            need = max(need, a + 90.0 - float(np.degrees(np.arccos(SQUEEZE_OK * np.cos(np.radians(a))))))
    return float(np.ceil(need * 2) / 2), {k: round(v, 1) for k, v in reach.items()}


def _trim(S, d0, d1, every=2):
    S = np.asarray(S, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(S, axis=0), axis=1))]
    return S[(s >= d0) & (s <= s[-1] - d1)][::every]


def _at_x(R, x):
    """y of a side (polyline across x) at column x, straight on past its ends."""
    R = np.asarray(R, float)
    R = R[np.argsort(R[:, 0])]
    if R[0, 0] <= x <= R[-1, 0]:
        return float(np.interp(x, R[:, 0], R[:, 1]))
    k = slice(0, min(6, len(R))) if x < R[0, 0] else slice(max(0, len(R) - 6), len(R))
    a, b = np.polyfit(R[k, 0], R[k, 1], 1)
    return float(a * x + b)


def _mask_corners(mask, M):
    """A label's sharp corners from its mask: the hull cut to 4 corners (nearest upright), then per corner the lines
    through both sides' contour 20 to 45 % of the way along, intersected (a hull corner sits on a rounded corner's arc)."""
    A = _hull_quad(mask)
    out = []
    for i in range(4):
        c, Ls = A[i], []
        for o in (A[i - 1], A[(i + 1) % 4]):
            Ln = float(np.linalg.norm(o - c))
            t = (o - c) / Ln
            s, off = (M - c) @ t, (M - c) @ np.array([-t[1], t[0]])
            sel = (s > 0.2 * Ln) & (s < 0.45 * Ln) & (np.abs(off) < 0.08 * Ln)
            if sel.sum() < 3:
                break
            Ls.append(geo.line_through(M[sel]))
        out.append(_meet(*Ls) if len(Ls) == 2 else c)
    return np.array(out)


def _mask_guides(M, corners, slack=3.0):
    """Each side's curve from the mask's contour (corner i to i + 1): a cubic bow off the chord fitted to the contour
    15 to 85 % of the way along, its ends held within `slack` px of the corners. The guides af.label_outline walks."""
    C = np.asarray(corners, float)
    cen = C.mean(0)
    G = []
    for i in range(4):
        a, b = C[i], C[(i + 1) % 4]
        L = float(np.linalg.norm(b - a))
        t = (b - a) / L
        n = np.array([t[1], -t[0]])
        n = -n if n @ ((a + b) / 2 - cen) < 0 else n
        u, off = (M - a) @ t / L, (M - a) @ n
        sel = (u > 0.15) & (u < 0.85) & (np.abs(off) < max(8.0, 0.1 * L))
        uu = np.linspace(0.0, 1.0, max(3, int(np.ceil(L / 2.0)) + 1))
        if sel.sum() >= 6:
            v, y = u[sel], off[sel]
            A = np.c_[1 - v, v, v * (1 - v), v * v * (1 - v)]
            reg = np.diag([1.0 / slack, 1.0 / slack, 0.0, 0.0]) * np.sqrt(len(v)) * 0.3
            c = np.linalg.lstsq(np.vstack([A, reg]), np.r_[y, 0, 0, 0, 0], rcond=None)[0]
            c[:2] = np.clip(c[:2], -slack, slack)
            o = (1 - uu) * c[0] + uu * c[1] + uu * (1 - uu) * (c[2] + c[3] * uu)
        else:
            o = np.zeros_like(uu)
        G.append(a[None] + (uu * L)[:, None] * t[None] + o[:, None] * n[None])
    return G


def label_draft(scene_u8, ref_u8, ref_alpha=None, quad=None, outline=None, kind="cylinder", pairs=(), occluders=(),
                mask=None, model="homography", about=ABOUT):
    """draft()'s label mode: the label only, measured by the recipe for labels on bottles, jars and tubes.
    quad: the generated label's 4 corners in the scene, in order round it (TL TR BR BL as on the packshot), a few px
    off at most; or None with mask, the generated label's mask. outline: the label on the packshot, upright, a polygon
    or a box (reference px). kind "cylinder": a label on a bottle or jar (align.model cylinder; axis and radius from
    the packshot's silhouette at the label's height); "flat": a label on a flat or flattened face (a tube's panels),
    with `model`. pairs: landmark pairs, refined under the fit and added. -> draft()'s dict, plus "label": what was
    measured (sides' support and bow, radii, cylinder, wrap, limb_min, print reach)."""
    if kind not in LABEL_KINDS:
        raise DraftError(f"label is cylinder, flat or off, not {kind!r}")
    if outline is None or not len(outline):
        raise DraftError("label mode needs the label's outline on the packshot (reference px: a box x0,y0,x1,y1 or a "
                         "polygon): traced, the packshot's outline is the whole product")
    scene, ref = im.to_float(scene_u8), im.to_float(ref_u8)
    Hs, Ws = scene_u8.shape[:2]
    notes = []

    # 1. the packshot's label: its sides on the packshot's own edges
    if len(outline) >= 3 and np.ndim(outline[0]) == 1:
        spec_o = {"type": "polygon", "about": "the label on the packshot, given to the draft",
                  "pts": [list(q) for q in _pt_list(outline)]}
    else:
        x0, y0, x1, y1 = (float(v) for v in outline)
        spec_o = {"type": "rounded_rect", "x0": x0, "y0": y0, "x1": x1, "y1": y1, "r_top": 0, "r_bottom": 0}
    P = np.asarray(ol.outline_polygon(spec_o), float)
    F = label_frame(P)
    C, rad = F["corners"], F["radii"]
    S = _snap_sides(ref, F)
    ref_q = np.array([C[c] for c in CORNERS])

    # 2. the bottle: its limbs at the label's height; a label side at a limb wraps on round it
    cyl, wrap, limb_min, reach = None, set(), None, {}
    if kind == "cylinder":
        cyl = bottle_limbs(ref_u8, ref_alpha, P)
        if max(C["TR"][0], C["BR"][0]) >= cyl["limbs"]["right"] - LIMB_TOL:
            wrap.add("right")
        if min(C["TL"][0], C["BL"][0]) <= cyl["limbs"]["left"] + LIMB_TOL:
            wrap.add("left")
        for side in ("left", "right"):
            if cyl["shadow"][side]:
                notes.append(f"the packshot's drop shadow touches the bottle's {side} side: its limb is taken where "
                             f"the label or glass meets the shadow (x {cyl['limbs'][side]}), check it (pitfall 64)")
            if cyl["spread"][side] > 2.5:
                notes.append(f"the bottle's {side} limb varies {cyl['spread'][side]} px over the label's rows: check "
                             "axis_x and radius")
        limb_min, reach = _limb_floor(ref_u8, ref_alpha, spec_o, cyl["axis_x"], cyl["radius"], wrap)

    # 3. the generated label's corners: the quad, or the mask's
    M = None
    if mask is not None:
        if mask.shape != scene_u8.shape[:2]:
            raise DraftError(f"the label's mask is {mask.shape[1]}x{mask.shape[0]}, the scene is {Ws}x{Hs}")
        if not mask.any():
            raise DraftError("the label's mask is empty")
        M = af.resample(np.asarray(af.trace_outline(mask), float), 1.0, closed=True)
    if quad is not None:
        q = np.asarray(_pt_list(quad), float)
        if len(q) != 4:
            raise DraftError("the quad needs the generated label's 4 corners: TL TR BR BL, in the packshot label's order")
    elif M is not None:
        q = _mask_corners(mask, M)
    else:
        raise DraftError("label mode needs the generated label's 4 corners as the quad (TL TR BR BL, as on the "
                         "packshot), or its mask")
    occ = _snap_occluders(scene, occluders)
    # sharp corners: where the label wraps round the silhouette, and where an occluder hides one
    sharp = ({1, 2} if "right" in wrap else set()) | ({0, 3} if "left" in wrap else set())
    vis = None
    if occ:
        vb = [int(np.floor(q[:, 0].min())) - 80, int(np.floor(q[:, 1].min())) - 80,
              int(np.ceil(q[:, 0].max())) + 81, int(np.ceil(q[:, 1].max())) + 81]
        vis = minimum_filter(mk.visibility(vb, occ, ss=2), size=7)      # 3 px round an occluder count as under it
        for i, c in enumerate(q):
            xi, yi = int(round(c[0])) - vb[0], int(round(c[1])) - vb[1]
            if vis[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].min() < 0.5:
                sharp.add(i)

    # 4. its own outline: each side walked along the curve it bows to (found in the image, or the mask's contour)
    outer = [i for i, side in ((1, "right"), (3, "left")) if side in wrap]
    try:
        if M is not None:
            lo = af.label_outline(scene, q, search=3.5, sharp=sorted(sharp), guides=_mask_guides(M, q), outer=outer)
        else:
            lo = af.label_outline(scene, q, sharp=sorted(sharp), guides="bow", outer=outer)
    except ValueError as e:
        raise DraftError(f"the generated label's outline: {e}") from e
    CS = np.asarray(lo["sharp"], float)
    S1, S2, S3, S4 = (np.asarray(v, float) for v in lo["sides"])
    for i, (name, sup) in enumerate(zip(("top", "right", "bottom", "left"), lo["support"])):
        if sup < 0.6:
            notes.append(f"the generated label's {name} side: {sup:.0%} of its samples on the curve, check it at 8x")

    # 5. its sides become the fit's edges. A scene edge may be shorter than the packshot's segment, never longer:
    # trimmed by the packshot's corner radius in scene px + 2 where a rounded corner starts, 4 px at a limb
    sc = float(np.sqrt(abs(_area(CS)) / max(abs(_area(ref_q)), 1e-9)))
    tr = lambda c, limb=False: 4.0 if limb else max(2.0, sc * rad[c] + 2.0)
    E = {"top": _trim(S1, tr("TL", "left" in wrap), tr("TR", "right" in wrap)),
         "right": _trim(S2, 2, 2) if "right" in wrap else _trim(S2, tr("TR"), tr("BR")),
         "bottom": _trim(S3[::-1], tr("BL", "left" in wrap), tr("BR", "right" in wrap)),
         "left": _trim(S4[::-1], 2, 2) if "left" in wrap else _trim(S4[::-1], tr("TL"), tr("BL"))}
    if vis is not None:                  # edge samples under or next to an occluder are the occluder's edge
        for k, pts in E.items():
            if len(pts):
                xi = np.clip(np.round(pts[:, 0]).astype(int) - vb[0], 0, vis.shape[1] - 1)
                yi = np.clip(np.round(pts[:, 1]).astype(int) - vb[1], 0, vis.shape[0] - 1)
                E[k] = pts[vis[yi, xi] > 0.5]
    for k, pts in E.items():
        if len(pts) < 3:
            raise DraftError(f"the generated label's {k} side has too little edge left to fit (an occluder over it?)")
    rt = lambda A: [[round(float(x), 2), round(float(y), 2)] for x, y in A]
    lines, limbs, corners = [], [], []
    if kind == "cylinder":
        ax, r = cyl["axis_x"], cyl["radius"]
        top, bot = np.asarray(S["top"], float), np.asarray(S["bottom"], float)
        for side, lx in (("right", ax + r), ("left", ax - r)):       # the top and bottom run on to a limb they wrap past
            if side in wrap:
                sg = 1 if side == "right" else -1
                ext = []
                for R in (top, bot):
                    R = R[sg * R[:, 0] <= sg * lx]
                    R = R[np.argsort(R[:, 0])]
                    ext.append(np.vstack([R, [lx, _at_x(R, lx)]]) if sg > 0 else np.vstack([[lx, _at_x(R, lx)], R]))
                top, bot = ext
        lines.append({"label": "label top edge", "ref": rt(top), "curve": True, "w": 2.0, "scene": rt(E["top"])})
        lines.append({"label": "label bottom edge", "ref": rt(bot), "curve": True, "w": 2.0, "scene": rt(E["bottom"])})
        for side in ("left", "right"):
            if side in wrap:
                lx = ax - r if side == "left" else ax + r
                limbs.append({"label": f"label {side} edge = the bottle's silhouette", "side": side,
                              "rows": [round(_at_x(top, lx), 2), round(_at_x(bot, lx), 2)], "w": 2.0, "scene": rt(E[side])})
            else:
                Rs = S[side]
                lines.append({"label": f"label {side} edge", "ref": rt([Rs[0], Rs[-1]]), "w": 2.0, "scene": rt(E[side])})
        for i, c in enumerate(CORNERS):
            side = "left" if c in ("TL", "BL") else "right"
            if side in wrap:             # a corner at the limb: where the edge meets the silhouette, roughly
                lx = ax - r if side == "left" else ax + r
                ref_pt, w = [lx, _at_x(top if c[0] == "T" else bot, lx)], 0.1
            else:
                ref_pt, w = C[c], 0.3
            corners.append({"label": f"label {c} corner (sides extended): a weak seed", "ref": [round(float(v), 2) for v in ref_pt],
                            "scene": [round(float(v), 2) for v in CS[i]], "w": w})
        align = {"model": "cylinder", "cylinder": {"axis_x": cyl["axis_x"], "radius": cyl["radius"]}}
        if wrap:
            align["cylinder"]["wrap"] = 20
        align["cylinder"]["limb_min"] = limb_min
    else:
        for k in ("top", "right", "bottom", "left"):
            Rs = S[k]
            lines.append({"label": f"label {k} edge", "ref": rt([Rs[0], Rs[-1]]), "w": 2.0, "scene": rt(E[k])})
        for i, c in enumerate(CORNERS):
            corners.append({"label": f"label {c} corner (sides extended)" + (": hidden, a weak seed" if i in sharp else ""),
                            "ref": [round(float(v), 2) for v in C[c]], "scene": [round(float(v), 2) for v in CS[i]],
                            "w": 0.3 if i in sharp else 1.0})
        align = {"model": model}
    align.update({"points": corners, "lines": lines})
    if limbs:
        align["limbs"] = limbs
    align["residual"] = {}
    try:
        H = geo.fit(align)[0]
    except ValueError as e:
        raise DraftError(f"the label's fit: {e}") from e

    # 6. landmark pairs, refined under that fit
    points = []
    for i, rp in enumerate(af.refine_points(scene, ref, H, [(tuple(map(float, a_)), tuple(map(float, b_))) for a_, b_ in pairs])):
        points.append({"label": f"landmark {i}" + (f", ncc {rp['ncc']}" if rp["trusted"] else ": weak match, read it by hand"),
                       "ref": [round(v, 2) for v in rp["ref"]], "scene": rp["scene"], "w": 1.0 if rp["trusted"] else 0.3})
    if points:
        align["points"] = points + corners
        H = geo.fit(align)[0]

    # 7. shape, work box, grade, finish, fill
    sil = np.asarray(lo["polygon"], float)
    small = float(np.sqrt(abs(_area(CS)))) < SMALL      # 1024 px frames: no fill ring, a tighter white level
    m = 16 if small else 40
    allp = np.vstack([sil, geo.map_polygon(H, P)])
    roi = [max(0, int(np.floor(allp[:, 0].min())) - m), max(0, int(np.floor(allp[:, 1].min())) - m),
           min(Ws, int(np.ceil(allp[:, 0].max())) + m + 1), min(Hs, int(np.ceil(allp[:, 1].max())) + m + 1)]
    spec_o["bleed"] = 10
    # the paper grade needs paper: on print straight on a coloured bottle, flattening by columns streaks it
    paper = af.white_box(ref_u8, raster.fill_polygon(ref_u8.shape[0], ref_u8.shape[1], [tuple(p_) for p_ in P]) > 0)
    if paper:
        grade = {"gain": "white_level", "white_percentile": 92, "window": 15 if small else 41, "erode": 2 if small else 4,
                 "deep_erode": 6 if small else 20, "colour": "reference", "ref_flatten": "field", "white_field": {},
                 "paper_clean": True}
    else:
        grade = {"gain": "white_level", "white_percentile": 92, "window": 41, "erode": 4, "deep_erode": 20}
        notes.append("no white paper on the packshot's label (print on the bottle itself?): the plain grade, colour "
                     "'scene', instead of the paper grade; check the colour and set grade.white_box if it's off")
    job = {
        "about": about,
        "scene_size": [Ws, Hs], "reference_size": [ref_u8.shape[1], ref_u8.shape[0]],
        "roi": roi, "reference_outline": spec_o, "shape": "generated", "align": align,
        "old_silhouette": {"type": "auto", "measured": rt(sil)},
        "occluders": occ, "grade": grade, "prefilter": "auto",
        "finish": {"blur": "auto", "blur_probe": None, "grain": "auto", "edge_softness": 0.45, "rolloff": 0.85,
                   "sharpen": "auto", "edge": "bleed"},
        "fill": {"dilate": 0 if small else 2},
    }
    (bx0, by0), (bx1, by1) = P.min(0), P.max(0)
    info = {"kind": kind, "support": lo["support"], "bows": lo.get("bows"), "radii": lo["radii"], "sharp": sorted(sharp),
            "from": "mask" if M is not None and quad is None else "quad", "guides": "mask" if M is not None else "bow",
            "cylinder": cyl, "wrap": sorted(wrap), "limb_min": limb_min, "reach": reach, "scale": round(sc, 3),
            "paper": paper}
    return {"job": job, "H": H, "points": points, "corners": corners, "props": [], "sil_found": None, "mask_note": None,
            "orient": af.orientation(H, (bx0, by0, bx1, by1)), "white_box": None, "notes": notes,
            "sheet": af.match_sheet(scene, ref, H, points) if points else None, "label": info}


def summary(d):
    """The draft's lines run.py prints (and the Auto Draft node shows): outline, points, sides, silhouette,
    orientation, grade, notes."""
    if d.get("label"):
        return _label_summary(d)
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


def _label_summary(d):
    job, info, points = d["job"], d["label"], d["points"]
    a = job["align"]
    sides = ("top", "right", "bottom", "left")
    out = [f"label      {info['kind']}: the generated label's outline measured from the {info['from']} "
           f"({'its contour' if info['guides'] == 'mask' else 'each side along the curve it bows to'}), the visible shape",
           "sides      " + ", ".join(f"{n} {sup:.0%}" + (f" bow {b:+.1f} px" if b is not None else "")
                                    for n, sup, b in zip(sides, info["support"], info["bows"] or [None] * 4))
           + " (share of samples on the curve)",
           "corners    " + ", ".join(f"{c} " + (f"r {r} px" if r else "sharp") for c, r in zip(CORNERS, info["radii"]))]
    cyl = info["cylinder"]
    if cyl:
        sh = [s_ for s_ in ("left", "right") if cyl["shadow"][s_]]
        out.append(f"cylinder   axis_x {cyl['axis_x']}, radius {cyl['radius']} (limbs {cyl['limbs']['left']} / "
                   f"{cyl['limbs']['right']}" + (f"; drop shadow on the {', '.join(sh)}: limb where the label or "
                                                 "glass meets it" if sh else "") + ")")
        out.append(f"wrap       {', '.join(info['wrap']) or 'none'}" + (
            f"; limb_min {info['limb_min']} (print reaches " +
            ", ".join(f"{k_} {v_} deg" for k_, v_ in info["reach"].items()) + ")" if info["reach"] else
            f"; limb_min {info['limb_min']}"))
    trusted = sum(1 for q in points if q["w"] >= 1.0)
    out.append(f"fit        {a['model']} + residual: {len(a['lines'])} lines" +
               (f", {len(a.get('limbs', []))} limb" + ("s" if len(a.get("limbs", [])) > 1 else "") if a.get("limbs") else "") +
               f", 4 corners as weak seeds; {trusted} trusted, {len(points) - trusted} weak landmark pairs")
    o = d["orient"]
    out.append(f"orient     lean left {o['lean_left']:+.3f} right {o['lean_right']:+.3f} (fan {o['fan']:+.3f}), "
               f"slope top {o['slope_top']:+.3f} bottom {o['slope_bottom']:+.3f}")
    out.append("grade      " + ("colour reference, ref_flatten field, white_field, paper_clean (the label's paper at "
                                f"{info['paper']})" if info["paper"] else "white level, colour scene (no paper)") +
               f"; finish edge bleed, sharpen auto; prefilter auto; fill dilate {job['fill']['dilate']}")
    for n in d["notes"]:
        out.append(f"note       {n}")
    return out
