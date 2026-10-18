"""Refining a keyframe: the zoomed crop the refinement works on, the check every attempt must pass, and the user's
attempt ladder with its log.

- Zoomed crop: CropRotate's geometry, filled in from the matte (centre, the product's axis turned upright, the
  product's extent plus a margin), sampled up so the refinement has pixels for legible text. An edit comes back at
  any size and is composited into the frame only inside the grown matte (CropRotateComposite's job), so the frame
  is untouched outside it by construction.
- The check: the frame unchanged outside the grown matte; the product's outline unmoved (edge strength along the
  matte's boundary correlates with the source's); the print's layout kept (text blocks correlate with the source's
  garbled print, so re-acquisition finds it); the label text spelled exactly (OCR on the unrolled label, a near miss
  counts as a fail); and the agent's own zoomed reading, recorded in the attempt log.
- The ladder, as the user runs it by hand: Nano Banana, Nano Banana with a new seed, Ideogram, then alternating with
  new seeds, 7 attempts at most. Then it stops and reports what failed and what to change; it never moves on to
  another frame by itself. The stills route (the reference photo warped on by run.py) comes first where the angle
  matches and isn't counted.
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

from imgvideo.keyframe import product_axis

from . import verify

LADDER = [("nano-banana-2", 0), ("nano-banana-2", 1), ("ideogram-4.5", 0), ("nano-banana-2", 2),
          ("ideogram-4.5", 1), ("nano-banana-2", 3), ("ideogram-4.5", 2)]
MAX_ATTEMPTS = 7
CHECKS = {"outline_min": 0.65, "layout_min": 0.30, "outside_max_px": 0}   # outline: accepted fixes 0.74-0.94, a 3 px shift 0.30-0.62
ADVICE = {
    "spelling": "the text: check the label copy, quote it word for word in the prompt, or use a frame where the "
                "label is larger (more pixels per letter)",
    "outline": "the geometry: the edit moved or redrew the product; strengthen the prompt's keep-the-shape wording, "
               "or use a frame where the product is larger and less turned",
    "layout": "the layout: the edit redrew where the text sits; give a reference photo at this frame's angle, or "
              "pick a frame nearer the reference's angle",
    "outside": "a bug: pixels outside the matte changed after compositing",
    "agent": "what the agent saw when reading the zoomed label (see its note)",
}


# ---------------------------------------------------------------- the zoomed crop (CropRotate geometry)

def crop_geometry(mask, margin=0.08):
    """CropRotate's frame for the product: {"cx", "cy", "w", "h", "angle"} in source px, angle in degrees,
    positive = clockwise (the product's axis turned upright)."""
    c, ax, n = product_axis(mask)
    ys, xs = np.nonzero(mask)
    P = np.c_[xs, ys].astype(float) - c
    t, q = P @ ax, P @ n
    tm, qm = 0.5 * (t.min() + t.max()), 0.5 * (q.min() + q.max())
    centre = c + tm * ax + qm * n
    L, Wd = np.ptp(t) * (1 + 2 * margin), np.ptp(q) * (1 + 2 * margin)
    angle = float(np.degrees(np.arctan2(ax[0], ax[1])))       # axis (pointing down) against the image's +y
    # an output pixel at offset d samples c + R(-angle) d, so the crop's down direction R(-angle)(0, 1) is the axis
    return {"cx": round(float(centre[0]), 2), "cy": round(float(centre[1]), 2), "w": round(float(Wd), 2),
            "h": round(float(L), 2), "angle": round(angle, 3)}


def quad(mask):
    """The matte's oriented bounding box as run.py draft's --quad: TL TR BR BL in scene px, upright in the product's
    own frame (top = the end the axis starts from)."""
    g = crop_geometry(mask, margin=0.0)
    a = np.radians(g["angle"])
    R = np.array([[np.cos(-a), -np.sin(-a)], [np.sin(-a), np.cos(-a)]])       # crop offset -> scene offset
    hw, hh = g["w"] / 2, g["h"] / 2
    c = np.array([g["cx"], g["cy"]])
    return [(c + R @ np.array(d)).round(1).tolist() for d in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]


def label_quad(mask, band, full=0.95, step=4.0, limb_inset=0.0):
    """The label's 4 corners (run.py draft --label's --quad, or --outline on the packshot): within the band, the stretch
    where the body is at full width (a rounded base or shoulder narrows it and isn't label), as a rectangle along the
    axis at the median left and right edges. Measured the same way on a frame and on a packshot, the two rectangles
    correspond. band: px along the axis from the matte's centroid (verify.body_band). limb_inset: px trimmed at both
    limbs (0 for label mode: its quad stays on the silhouette; the packshot outline's wrapping sides belong on the
    photo's limbs). -> TL TR BR BL in image px"""
    c, ax, n = product_axis(mask)
    ys, xs = np.nonzero(mask)
    P = np.c_[xs, ys].astype(float) - c
    t, q = P @ ax, P @ n
    rows = []
    for tt in np.arange(band[0], band[1] + 1e-6, step):
        sel = np.abs(t - tt) < step / 2
        if sel.sum() > 4:
            rows.append((tt, np.percentile(q[sel], 1), np.percentile(q[sel], 99)))
    R = np.array(rows)
    w = R[:, 2] - R[:, 1]
    ok = np.nonzero(w >= full * np.median(w))[0]
    t0, t1 = R[ok[0], 0], R[ok[-1], 0]
    lo, hi = float(np.median(R[ok, 1])) + limb_inset, float(np.median(R[ok, 2])) - limb_inset
    return [(c + tt * ax + qq * n).round(1).tolist() for tt, qq in ((t0, lo), (t0, hi), (t1, hi), (t1, lo))]


PAPER_MIN = 0.10   # a paper label's largest white area: 0.17-0.58 of the label (7 hand label jobs); print on a body 0.003-0.008


def paper_gate(rgb, label_mask, min_share=PAPER_MIN, min_l=0.9, max_sat=0.08):
    """Is the packshot's label paper (label mode fits it) or print straight on the body? The largest connected area
    of near-white, unsaturated pixels must cover min_share of the label: a white letter or two (a lip tube's
    white print: a 14 px letter passed the old test, imgpost's white_box alone) isn't paper. -> (white box for the grade or
    None, info)"""
    from imgpost import autofit
    f = rgb.astype(float) / 255.0
    L = f @ np.array([0.2126, 0.7152, 0.0722])
    ok = label_mask & (L >= min_l) & ((f.max(2) - f.min(2)) <= max_sat)
    lab, n = ndi.label(ok)
    largest = int(np.bincount(lab.ravel())[1:].max()) if n else 0
    share = largest / max(int(label_mask.sum()), 1)
    info = {"paper_share": round(share, 4), "white_share": round(float(ok.sum()) / max(int(label_mask.sum()), 1), 4),
            "min_share": min_share}
    if share < min_share:
        return None, info
    return autofit.white_box(rgb, label_mask), info


def coverage(geom, mask):
    """Share of the matte inside the crop's rectangle (1.0 for a crop_geometry of that matte)."""
    ys, xs = np.nonzero(mask)
    a = np.radians(geom["angle"])
    dx, dy = xs - geom["cx"], ys - geom["cy"]
    ox = np.cos(a) * dx - np.sin(a) * dy
    oy = np.sin(a) * dx + np.cos(a) * dy
    return float(((np.abs(ox) <= geom["w"] / 2) & (np.abs(oy) <= geom["h"] / 2)).mean())


def _grid(geom, out_w, out_h):
    """Source (x, y) for every output pixel of the crop."""
    a = np.radians(geom["angle"])
    sx, sy = geom["w"] / out_w, geom["h"] / out_h
    v, u = np.mgrid[0:out_h, 0:out_w].astype(float)
    dx, dy = (u + 0.5) * sx - geom["w"] / 2, (v + 0.5) * sy - geom["h"] / 2
    ca, sa = np.cos(-a), np.sin(-a)                           # output offset d samples c + R(-angle) d
    return geom["cx"] + ca * dx - sa * dy, geom["cy"] + sa * dx + ca * dy


def crop(img, geom, long_side=2048):
    """The zoomed, upright crop (uint8), its long side `long_side` px."""
    s = long_side / max(geom["w"], geom["h"])
    ow, oh = max(8, int(round(geom["w"] * s))), max(8, int(round(geom["h"] * s)))
    X, Y = _grid(geom, ow, oh)
    out = np.stack([map_coordinates(img[..., ch].astype(np.float32), [Y, X], order=3, mode="constant", cval=0)
                    for ch in range(3)], -1)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def uncrop(frame, edit, geom, mask, grow=6, feather=3.0):
    """An edited crop (any size) back into the frame, only inside the matte grown by `grow` px, feathered inside
    that boundary. Outside the grown matte the frame is returned untouched. -> (frame, alpha)"""
    h, w = frame.shape[:2]
    a = np.radians(geom["angle"])
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    dx, dy = xx - geom["cx"], yy - geom["cy"]
    ca, sa = np.cos(a), np.sin(a)                             # inverse of _grid's rotation
    ox, oy = ca * dx - sa * dy + geom["w"] / 2, sa * dx + ca * dy + geom["h"] / 2
    eh, ew = edit.shape[:2]
    U, V = ox * ew / geom["w"] - 0.5, oy * eh / geom["h"] - 0.5
    inside = (U >= 0) & (U <= ew - 1) & (V >= 0) & (V <= eh - 1)
    grown = ndi.binary_dilation(mask, iterations=grow) & inside
    alpha = np.clip(ndi.distance_transform_edt(grown) / max(feather, 1e-3), 0, 1)
    sel = alpha > 0
    vals = np.stack([map_coordinates(edit[..., ch].astype(np.float32), [V[sel], U[sel]], order=3, mode="nearest")
                     for ch in range(3)], -1)
    out = frame.astype(np.float32).copy()
    out[sel] = out[sel] * (1 - alpha[sel, None]) + vals * alpha[sel, None]
    res = frame.copy()
    res[sel] = np.clip(out[sel] + 0.5, 0, 255).astype(np.uint8)
    return res, alpha


# ---------------------------------------------------------------- checking one attempt

def _edges(g):
    return np.hypot(ndi.sobel(g, 0), ndi.sobel(g, 1))


def outline_score(attempt, source, mask, band=6):
    """Edge strength along the matte's boundary, attempt against source: about 1 when the product's outline stayed
    where it was."""
    ring = ndi.binary_dilation(mask, iterations=band) & ~ndi.binary_erosion(mask, iterations=band)
    ea = _edges(ndi.gaussian_filter(attempt.astype(np.float32).mean(-1), 1.0))[ring]
    es = _edges(ndi.gaussian_filter(source.astype(np.float32).mean(-1), 1.0))[ring]
    ea, es = ea - ea.mean(), es - es.mean()
    return float((ea * es).sum() / (np.sqrt((ea * ea).sum() * (es * es).sum()) + 1e-6))


def layout_score(attempt, source, mask, erode=12):
    """Text-block correlation of the attempt's print with the source's garbled print inside the matte."""
    m = ndi.binary_erosion(mask, iterations=erode)
    band = lambda img: (ndi.gaussian_filter(img.astype(np.float32).mean(-1), 2.0)
                        - ndi.gaussian_filter(img.astype(np.float32).mean(-1), 10.0))
    a, s = band(attempt)[m], band(source)[m]
    a, s = a - a.mean(), s - s.mean()
    return float((a * s).sum() / (np.sqrt((a * a).sum() * (s * s).sum()) + 1e-6))


def check(attempt, source, mask, copy, band=None, grow=6, checks=CHECKS):
    """Every automatic check of one refined keyframe (full-frame, already composited). -> dict with "failed" (list)"""
    outside = int((np.abs(attempt.astype(np.int16) - source.astype(np.int16)).max(-1) > 0)[
        ~ndi.binary_dilation(mask, iterations=grow)].sum())
    res = {"outside_px": outside, "outline": round(outline_score(attempt, source, mask), 3),
           "layout": round(layout_score(attempt, source, mask), 3)}
    band = band or verify.body_band(attempt, mask)[0]
    res["band"] = band
    failed = []
    if copy and band:
        words, _ = verify.read_label(attempt, mask, band)
        sp = verify.spelling(copy, words or set())
        v, summ = verify.verdict(sp)
        res["spelling"] = {"verdict": v, **summ}
        if v == "fail":
            failed.append("spelling")
        elif v == "unconfirmed":                 # a near miss: an OCR slip or a misspelling; the agent decides
            res["needs_agent"] = summ["near"]
    if outside > checks["outside_max_px"]:
        failed.append("outside")
    if res["outline"] < checks["outline_min"]:
        failed.append("outline")
    if res["layout"] < checks["layout_min"]:
        failed.append("layout")
    res["failed"] = failed
    return res


# ---------------------------------------------------------------- the ladder and its log

class Ladder:
    """One keyframe's attempts, logged in attempts.json next to the crops:
    {"frame", "geometry", "attempts": [{"n", "model", "seed", "file", "checks", "agent", "verdict", "why", "time"}]}.
    The agent's reading goes in "agent": {"verdict": "pass" | "fail", "note"}; an attempt passes only when the
    automatic checks pass and the agent's reading does (or the run has no agent)."""

    def __init__(self, folder, frame, geometry, base_seed=46, agent=True):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "attempts.json"
        self.agent = agent
        if self.path.exists():
            self.log = json.loads(self.path.read_text())
        else:
            self.log = {"frame": int(frame), "geometry": geometry, "base_seed": base_seed, "attempts": []}
            self.save()

    def save(self):
        self.path.write_text(json.dumps(self.log, indent=1))

    def passed(self):
        return next((a for a in self.log["attempts"] if a["verdict"] == "pass"), None)

    def next(self):
        """(n, model, seed) of the next attempt, or None when one passed or all 7 are spent."""
        if self.passed():
            return None
        n = len(self.log["attempts"])
        if n >= MAX_ATTEMPTS:
            return None
        model, k = LADDER[n]
        return n + 1, model, self.log["base_seed"] + k

    def record(self, n, model, seed, file, checks, agent=None):
        auto_ok = not checks["failed"]
        ag = agent if agent is not None else ({"verdict": "pending"} if self.agent else {"verdict": "n/a"})
        if not auto_ok and ag["verdict"] == "pending":
            ag = {"verdict": "not needed", "note": "the automatic checks failed"}
        if ag["verdict"] == "n/a" and checks.get("needs_agent"):
            ag = {"verdict": "fail", "note": f"near misses {checks['needs_agent']} with no agent to read them "
                                              "(the Qwen-VL tiebreak is planned)"}
        verdict = ("pass" if auto_ok and ag["verdict"] in ("pass", "n/a") else
                   "pending" if auto_ok and ag["verdict"] == "pending" else "fail")
        why = list(checks["failed"]) + (["agent"] if ag["verdict"] == "fail" else [])
        self.log["attempts"].append({"n": n, "model": model, "seed": seed, "file": str(file), "checks": checks,
                                     "agent": ag, "verdict": verdict, "why": why,
                                     "time": time.strftime("%Y-%m-%d %H:%M")})
        self.save()
        return verdict

    def set_agent(self, n, verdict, note=""):
        """The agent's zoomed reading of attempt n; settles a pending attempt."""
        a = self.log["attempts"][n - 1]
        a["agent"] = {"verdict": verdict, "note": note}
        if verdict == "fail":
            a["verdict"], a["why"] = "fail", sorted(set(a["why"]) | {"agent"})
        elif not a["checks"]["failed"]:
            a["verdict"] = "pass"
        self.save()

    def report(self):
        """After the last attempt: what failed, how often, and what to change. None while attempts remain."""
        if self.passed() or len(self.log["attempts"]) < MAX_ATTEMPTS:
            return None
        counts = {}
        for a in self.log["attempts"]:
            for w in a["why"]:
                counts[w] = counts.get(w, 0) + 1
        lines = [f"frame {self.log['frame']}: no attempt passed in {MAX_ATTEMPTS} "
                 f"({', '.join(f'{m} seed {s}' for m, s in [(a['model'], a['seed']) for a in self.log['attempts']])})."]
        for w, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {w} failed {c} of {MAX_ATTEMPTS}: change {ADVICE.get(w, w)}.")
        lines.append("Stopped here, as set: change the settings, the prompt or the inputs, then rerun this keyframe.")
        return "\n".join(lines)


def save_png(arr, path):
    Image.fromarray(arr).save(path)
    return path
