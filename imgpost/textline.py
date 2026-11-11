"""Text only (2026-11-04): a label's letters found as lines of text, told from its other marks, and each line of the
true print placed on the generated print's own line.

A generated label keeps its shape, outline, lines, bands and colours; only its letters change (pitfall 111). The
letters are marks of one polarity against the surface they sit on (ink.py's print maps), grouped into lines in the
packshot's frame, where lines of text run level: marks whose rows overlap and that sit close along the row. What
isn't a line of text stays the generated product's own:
  rule     a mark long along the row and thin across it, nearly solid (a band, a rule, a label's edge or a seam);
  edge     a mark touching the print's band edge (a part of the product, a label's border);
  lone     a mark in no line: a line of one mark that isn't a word (an icon, a badge's ring, a glint).
The generated print is read in the same frame (the frame's print map pulled back through the fit), so a line of the
true print and a line of the generated one compare row for row.

Placement: each true line goes onto the generated line under it (matched in order down the label, by their rows and
heights, columns kept apart by their extents): its height and rows from the generated line's, its run along the row
from where both start (or centre, or end: the block's own alignment on the packshot); lines of one size on the packshot
keep one size. A true line with no generated line under it follows the block's placement (the median of its matched
neighbours') and is reported.

Pure numpy + scipy; arrays are packshot px unless named otherwise."""
import numpy as np
from scipy import ndimage as ndi

MARK_THRESH = 0.35     # a print map (0..1 coverage) above this is a mark
GEN_THRESH = 0.35     # the generated print's marks (lower merges its blurred lines into long marks)
MIN_MARK = 4           # px: smaller marks are noise (at the packshot's scale; an i's dot at 1,000 px is 6-12)
RULE_LEN = 0.35        # a rule runs over this share of the band's width ...
RULE_THIN = 0.12       # ... and is this thin across against its length
RULE_SOLID = 0.6       # ... and covers this share of its box (a line of letters merged by blur covers less)
ROW_OVERLAP = 0.45     # two marks share a line when their rows overlap by this share of the smaller's height
ROW_GAP = 3.0          # ... and the gap between them along the row is at most this many line heights (a
                       # column beside it starts 4-6 away)
ROW_SIZE = 2.2         # ... and are within this many times of each other's height (the line's core's)
WORD_ASPECT = 1.8      # a line of one mark is a word (merged letters) when this many times as long as high
GATE = True            # garbled type only (2026-11-10): lines already legible and marks that aren't type stay generated
TURNS = True           # blocks of lines running at a turn on the packshot read in their own frame (2026-11-10)


def printed_body(cfg):
    """Text straight on the product's body, no label (a body printed all over: a sleeve, a glossy tube): its print
    holds graphics and type set every way, so the gate, the turned blocks and the grouping beside big marks apply. A
    label's text (label mode) keeps the reading its approvals were made on."""
    return bool(cfg.get("text")) and not cfg.get("label")


def marks(P, sel, thresh=MARK_THRESH, min_px=MIN_MARK):
    """The marks of a print map inside `sel`: -> (label image, [{"i", "x0", "y0", "x1", "y1", "px", "cy", "h", "w"}])
    with x1, y1 exclusive."""
    m = (P > thresh) & sel
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    out = []
    if not n:
        return lab, out
    size = ndi.sum(m, lab, np.arange(1, n + 1))
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None or size[i - 1] < min_px:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        out.append({"i": i, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "px": int(size[i - 1]),
                    "cy": 0.5 * (y0 + y1), "h": y1 - y0, "w": x1 - x0})
    return lab, out


def _band_width(sel):
    xs = np.nonzero(sel.any(0))[0]
    return float(xs.max() - xs.min() + 1) if len(xs) else 1.0


LONG_ON_ROWS = 0.4     # a long generated mark is the label's own (a band) when under this share of it lies on the new
                       # lines' rows (a curved band under a wordmark 0.1-0.3); over it, blurred lines of small type
EDGE_LONG = 3.0        # a mark touching the band's edge is the edge's (a label's border, its shading, a part of the
                       # product) when this many typical mark heights long; a compact one is a letter cut by the band
RIM_HUG = 0.8          # a mark by the band's rim with this share of its px within max(3, 0.3 typ) px of the rim follows
                       # the label's outline (its printed edge, a rounded corner's shading: a 2 px arc at a packshot
                       # label's corner joined a line of text and was drawn inside the generated label, pitfall 114) ...
RIM_HUG_LEN = 1.5      # ... when this many typical mark heights long (smaller: a letter's dot or serif cut by the rim)
EDGE_TALL = 2.8        # a mark by the rim this many typical heights tall is no line of type (a holder's or a cloth's
                       # shade at a label's corner joined the last line and was painted over, pitfall 114)
SHARP_RULE = 2.5       # a sharp print's single mark this many times as long as high, even and gapless (_even), is a
                       # rule, not a word (a band's lighter end: 3.4); the generated print's blur smooths words too


def _even(lab, m):
    """A long mark with no gaps along it, an even thickness and smooth edges (a band, a rule, a band's lighter end,
    curved or not), not a word: merged letters leave gaps or notches and their tops jump between the x-height and
    the ascenders at every letter."""
    sub = lab[m["y0"]:m["y1"], m["x0"]:m["x1"]] == m["i"]
    has = sub.any(0)
    if has.mean() < 0.98:
        return False
    rows = np.arange(sub.shape[0])[:, None]
    top = np.where(sub, rows, sub.shape[0]).min(0)[has].astype(float)
    bot = np.where(sub, rows, -1).max(0)[has].astype(float)
    t = bot - top + 1
    jump = max(np.percentile(np.abs(np.diff(top)), 90), np.percentile(np.abs(np.diff(bot)), 90)) if len(t) > 2 else 0
    return bool(np.std(t) <= 0.2 * np.mean(t) and jump <= 1.0)


BAND_SPAN = 0.6        # a band runs across this share of the label at least (a line of type blurred solid: 0.47) ...
BAND_FILL = 0.85       # ... fills this share of its own columns' extent (letters merged by blur leave gaps) ...
BAND_EVEN = 0.35       # ... at a thickness this even (sd / mean over its columns)


def _band_like(lab, m):
    """A long mark solid down its own columns at an even thickness: a band or rule a little turned or bowed, whose
    box a turn of 1 deg leaves only 58 % full (pitfall 113: a 4x frame's band failed RULE_SOLID and went as text)."""
    sub = lab[m["y0"]:m["y1"], m["x0"]:m["x1"]] == m["i"]
    has = sub.any(0)
    if has.mean() < 0.95:
        return False
    rows = np.arange(sub.shape[0])[:, None]
    top = np.where(sub, rows, sub.shape[0]).min(0)[has]
    bot = np.where(sub, rows, -1).max(0)[has]
    ext = (bot - top + 1).astype(float)
    fill = float(sub.sum()) / max(ext.sum(), 1.0)
    return bool(fill >= BAND_FILL and ext.std() <= BAND_EVEN * ext.mean())


def classify(lab, ms, sel, sharp=False):
    """Each mark's kind before lines: "rule" (long, thin, solid: a band, a rule, an edge line), "edge" (touching the
    band's edge, long: EDGE_LONG) or None (a candidate letter). In place: sets m["kind"]."""
    bw = _band_width(sel)
    rim = sel & ~ndi.binary_erosion(sel, iterations=2)
    touch = set(np.unique(lab[rim]).tolist())
    typ = float(np.median([m["h"] for m in ms])) if ms else 1.0
    dist = ndi.distance_transform_edt(sel)
    near_r, hug_r = max(4.0, 0.6 * typ), max(3.0, 0.3 * typ)

    def by_rim(m):
        sub = lab[m["y0"]:m["y1"], m["x0"]:m["x1"]] == m["i"]
        d = dist[m["y0"]:m["y1"], m["x0"]:m["x1"]][sub]
        if not len(d) or d.min() > near_r:
            return False
        return (float((d <= hug_r).mean()) >= RIM_HUG and max(m["w"], m["h"]) >= RIM_HUG_LEN * typ) or \
            m["h"] >= EDGE_TALL * typ
    for m in ms:
        long_ = max(m["w"], m["h"])
        thin = min(m["w"], m["h"])
        solid = m["px"] / float(m["w"] * m["h"])
        if m["w"] >= RULE_LEN * bw and thin <= RULE_THIN * long_ and solid >= RULE_SOLID:
            m["kind"] = "rule"
        elif sharp and m["w"] >= SHARP_RULE * m["h"] and m["w"] >= 0.1 * bw and _even(lab, m):
            m["kind"] = "rule"
        elif m["w"] >= BAND_SPAN * bw and thin <= RULE_THIN * long_ and _band_like(lab, m):
            m["kind"] = "rule"                      # a band turned or bowed a little: solid down its own columns
        elif m["h"] >= RULE_LEN * bw and m["w"] <= RULE_THIN * m["h"] and solid >= RULE_SOLID:
            m["kind"] = "rule"                      # upright: a label's side edge, a seam
        elif m["i"] in touch and max(m["w"], m["h"]) >= EDGE_LONG * typ:
            m["kind"] = "edge"
        elif by_rim(m):
            m["kind"] = "edge"
        else:
            m["kind"] = None
    return ms


BAND_FULL = 0.8        # a long generated mark short of this share of the label's width ...
IN_BLOCK = 0.6         # ... with this share of it inside a block's text area is blurred type, not a band
RULE_LANDING = 0.5     # a short true line with this share of its cover on the generated rules and bands: left out
LOCAL_N = 4            # a turned block's lines join a mark by the rows of their last this many marks (a curved line)
TINY = 0.6             # a mark beside a line over ROW_SIZE times its height waits to join it (a full stop) only when
                       # under this share of the typical mark's height
SMALL = 0.5            # a mark under this share of the typical mark's height is a dot, an accent or a comma: it joins
                       # the line beside it after the lines are formed


def _core(L, by_i):
    """A line's core rows: the median top and bottom of its full-height marks."""
    mem = [by_i[i] for i in L["members"]]
    hs = np.array([m["h"] for m in mem], float)
    full = [m for m in mem if m["h"] >= 0.6 * np.median(hs)] or mem
    return float(np.median([m["y0"] for m in full])), float(np.median([m["y1"] for m in full]))


def group(ms, local=False, beside=False):
    """Candidate letters (kind None) grouped into lines: a mark joins the line whose core rows (its full-height marks'
    median top and bottom) it shares (ROW_OVERLAP) and that it sits next to along the row (ROW_GAP line heights); small
    marks (dots, accents, commas: under SMALL of the typical height) then join the line beside them.
    -> [line {"members": [mark i], "x0", "y0", "x1", "y1", "top", "bot", "h", "cy", "n"}], sorted down the label.
    Marks in no line of text (a lone mark that isn't a word) get kind "lone"; the rest kind "text" and m["line"]."""
    cand = [m for m in ms if m["kind"] is None]
    by_i = {m["i"]: m for m in ms}
    if not cand:
        return []
    typ = float(np.median([m["h"] for m in cand]))
    big = sorted([m for m in cand if m["h"] >= SMALL * typ], key=lambda m: (m["x0"], m["cy"]))
    small = [m for m in cand if m["h"] < SMALL * typ]
    lines = []
    for m in big:
        best, bo = None, 0.0
        tiny = False
        for L in lines:
            ct, cb = L["core"]
            if local and len(L["members"]) > LOCAL_N:
                # a turned block's line may curve: its rows where it has got to (its last marks), not its median
                last = [by_i[i] for i in L["members"][-LOCAL_N:]]
                ct, cb = float(np.median([q["y0"] for q in last])), float(np.median([q["y1"] for q in last]))
            ov = min(cb, m["y1"]) - max(ct, m["y0"])
            hh = min(cb - ct, m["h"])
            if hh <= 0 or ov < ROW_OVERLAP * hh:
                continue
            if m["h"] / max(cb - ct, 1.0) > ROW_SIZE:
                continue                    # a mark of another size: a shaded patch, a badge beside a line
            if m["h"] / max(cb - ct, 1.0) < 1.0 / ROW_SIZE:
                # a full stop after a wordmark: it joins the line beside it later; a letter of the label's typical
                # size beside a big figure or a glint starts its own line (deferred, a line of small type beside a
                # figure was left as lone marks and erased)
                tiny = tiny or not beside or m["h"] < TINY * typ
                continue
            gap = m["x0"] - L["x1"]
            if gap > ROW_GAP * max(cb - ct, 1):
                continue
            score = ov / float(hh) - 0.1 * max(gap, 0) / max(cb - ct, 1)
            if best is None or score > bo:
                best, bo = L, score
        if best is None and tiny:
            small.append(m)
            continue
        if best is None:
            best = {"members": [], "x0": m["x0"], "x1": m["x1"]}
            lines.append(best)
        best["members"].append(m["i"])
        best["x0"], best["x1"] = min(best["x0"], m["x0"]), max(best["x1"], m["x1"])
        best["core"] = _core(best, by_i)
    for m in small:
        cx = 0.5 * (m["x0"] + m["x1"])
        best, bd = None, None
        for L in lines:
            ct, cb = L["core"]
            h = cb - ct
            if not (L["x0"] - 0.5 * h <= cx <= L["x1"] + 0.5 * h):
                continue
            if not (ct - 0.9 * h <= m["cy"] <= cb + 0.6 * h):
                continue
            d = abs(m["cy"] - 0.5 * (ct + cb))
            if best is None or d < bd:
                best, bd = L, d
        if best is None:
            m["kind"] = "lone"
        else:
            best["members"].append(m["i"])
    out = []
    for L in lines:
        mem = [by_i[i] for i in L["members"]]
        full = [m for m in mem if m["h"] >= SMALL * typ]
        if len(full) == 1 and full[0]["w"] < WORD_ASPECT * full[0]["h"]:
            for m in mem:
                m["kind"] = "lone"
            continue
        top, bot = L["core"]
        for m in mem:
            m["kind"] = "text"
        out.append({"members": [m["i"] for m in mem], "x0": int(min(m["x0"] for m in mem)),
                    "x1": int(max(m["x1"] for m in mem)), "y0": int(min(m["y0"] for m in mem)),
                    "y1": int(max(m["y1"] for m in mem)), "top": top, "bot": bot, "h": bot - top,
                    "cy": 0.5 * (top + bot), "n": len(mem), "px": int(sum(m["px"] for m in mem))})
    out.sort(key=lambda L: (L["cy"], L["x0"]))
    for k, L in enumerate(out):
        L["k"] = k
        for i in L["members"]:
            by_i[i]["line"] = k
    return out


def text_lines(P, sel, thresh=MARK_THRESH, min_px=MIN_MARK, sharp=False, local=False, beside=False):
    """A print map's lines of text and its other marks. -> {"lab": marks' label image, "marks": [...], "lines": [...],
    "text": bool mask of the letters, "other": bool mask of the marks that aren't (rules, edges, lone marks),
    "line_map": int image, the line number + 1 of each letter's px (0 elsewhere)}"""
    lab, ms = marks(P, sel, thresh, min_px)
    classify(lab, ms, sel, sharp)
    lines = group(ms, local, beside)
    lut = np.zeros(lab.max() + 1, np.int32)
    rule = np.zeros(lab.max() + 1, bool)
    edge = np.zeros(lab.max() + 1, bool)
    lone = np.zeros(lab.max() + 1, bool)
    for m in ms:
        if m.get("kind") == "text":
            lut[m["i"]] = m["line"] + 1
        elif m.get("kind") == "lone":
            lone[m["i"]] = True
        elif m.get("kind") == "edge":
            edge[m["i"]] = True
        else:
            rule[m["i"]] = True
    line_map = lut[lab]
    # an edge mark is the band's edge only near it: a letter blurred into a label's shaded border stays a letter
    typ = float(np.median([m["h"] for m in ms])) if ms else 1.0
    near_rim = ndi.distance_transform_edt(sel) <= max(4.0, 0.6 * typ)
    other = rule[lab] | (edge[lab] & near_rim)
    return {"lab": lab, "marks": ms, "lines": lines, "text": line_map > 0, "other": other, "lone": lone[lab],
            "edge": edge[lab], "line_map": line_map}


def summary(tl):
    kinds = {}
    for m in tl["marks"]:
        kinds[m.get("kind") or "?"] = kinds.get(m.get("kind") or "?", 0) + 1
    return {"lines": len(tl["lines"]), "marks": kinds,
            "heights": [round(L["h"], 1) for L in tl["lines"]]}


# ---- the frame's print in the packshot's frame, and back ----

def _inverse(Hm, X, Y):
    """scene px -> reference px through a fitted mapping (a 3 x 3 or an object with inverse()) -> (u, v, ok)"""
    if hasattr(Hm, "inverse"):
        u, v, ok = Hm.inverse(X, Y)
        return u, v, (np.ones(np.shape(u), bool) if ok is None else ok)
    q = np.c_[np.ravel(X), np.ravel(Y), np.ones(np.size(X))] @ np.linalg.inv(Hm).T
    return (q[:, 0] / q[:, 2]).reshape(np.shape(X)), (q[:, 1] / q[:, 2]).reshape(np.shape(X)), \
        np.ones(np.shape(X), bool)


def pull_back(P_wb, Hm, roi, band):
    """A work-box map (scene px) read at the packshot's band pixels through the mapping (reference -> scene), bilinear.
    -> reference-px map (0 off the band)"""
    from .geometry import apply_H
    ys, xs = np.nonzero(band)
    q = apply_H(Hm, np.c_[xs, ys].astype(float))
    out = np.zeros(band.shape)
    out[ys, xs] = ndi.map_coordinates(P_wb, [q[:, 1] - roi[1], q[:, 0] - roi[0]], order=1, mode="constant")
    return out


def push(M_ref, Hm, roi, order=0):
    """A reference-px map read at each work-box pixel's centre through the mapping's inverse (nearest for labels,
    bilinear with order 1). -> work-box map"""
    x0, y0, x1, y1 = roi
    X, Y = np.meshgrid(np.arange(x0, x1, dtype=float), np.arange(y0, y1, dtype=float))
    u, v, ok = _inverse(Hm, X, Y)
    out = ndi.map_coordinates(M_ref.astype(float) if order else M_ref, [v, u], order=order, mode="constant")
    return np.where(ok, out, 0)


class LineMap:
    """One line's own placement over the fitted mapping: A (3 x 3, reference px -> reference px) first, then the
    mapping. inverse() for ink.warp_cover."""

    def __init__(self, Hm, A):
        self.Hm, self.A = Hm, np.asarray(A, float)
        self.Ai = np.linalg.inv(self.A)

    def inverse(self, X, Y):
        u, v, ok = _inverse(self.Hm, X, Y)
        Ai = self.Ai
        return Ai[0, 0] * u + Ai[0, 1] * v + Ai[0, 2], Ai[1, 0] * u + Ai[1, 1] * v + Ai[1, 2], ok

    def forward(self, pts):
        from .geometry import apply_H
        p = np.atleast_2d(np.asarray(pts, float))
        return apply_H(self.Hm, p @ self.A[:2, :2].T + self.A[:2, 2])


class BentLineMap(LineMap):
    """A line bent onto its generated row's course (pitfall 115): a vertical offset b(u) in the packshot's rows, per
    packshot column u of the line, before its placement A and the mapping (the row's curve less its slope, which the
    turn carries)."""

    def __init__(self, Hm, A, bend):
        super().__init__(Hm, A)
        self.bend = bend                      # callable: u (packshot px, before A) -> rows

    def inverse(self, X, Y):
        u, v, ok = super().inverse(X, Y)
        return u, v - self.bend(u), ok

    def forward(self, pts):
        p = np.atleast_2d(np.asarray(pts, float)).copy()
        p[:, 1] = p[:, 1] + self.bend(p[:, 0])
        return super().forward(p)


class StraightLineMap:
    """A line laid straight (pitfall 113): its LineMap's local affine at the line's centre c (reference px), so a fit
    that bows rows round a cylinder (a squeeze tube's flat panel fitted as one) lays it as straight as the generator
    drew it."""

    def __init__(self, lm, c, d=4.0):
        self.lm = lm
        c = np.asarray(c, float)
        self.c = c
        f0 = lm.forward([c])[0]
        jx = (lm.forward([c + [d, 0]])[0] - lm.forward([c - [d, 0]])[0]) / (2 * d)
        jy = (lm.forward([c + [0, d]])[0] - lm.forward([c - [0, d]])[0]) / (2 * d)
        self.J = np.c_[jx, jy]
        self.Ji = np.linalg.inv(self.J)
        self.f0 = f0

    def forward(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        return self.f0 + (p - self.c) @ self.J.T

    def inverse(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        dx, dy = X - self.f0[0], Y - self.f0[1]
        u = self.c[0] + self.Ji[0, 0] * dx + self.Ji[0, 1] * dy
        v = self.c[1] + self.Ji[1, 0] * dx + self.Ji[1, 1] * dy
        return u, v, np.ones(np.shape(u), bool)


def _path_sharpness(L, mp, P_s, origin, n=48):
    """How sharp the generated print's row profile is along a true line laid by mapping mp (over its run, a line
    height above and below): var / mean^2 of the row sums. P_s: the frame's print map in the work box at origin."""
    h = max(L["h"], 1.0)
    us = np.linspace(L["x0"], L["x1"], n)
    vs = np.arange(L["top"] - h, L["bot"] + h + 1e-6, 0.5)
    U, V = np.meshgrid(us, vs)
    q = mp.forward(np.c_[U.ravel(), V.ravel()])
    xs, ys = q[:, 0] - origin[0], q[:, 1] - origin[1]
    vals = ndi.map_coordinates(P_s, [ys, xs], order=1, mode="constant").reshape(U.shape)
    prof = vals.sum(1)
    m = prof.mean()
    return float(prof.var() / (m * m)) if m > 1e-9 else 0.0


STRAIGHT_GAIN = 1.1    # a line goes straight when the generated print's rows are this much sharper along it


def measure_rows(lines, M, pad=0.35, wide=1.0):
    """Each line's rows from its row profile over its columns in map M (both prints blurred alike): the run of rows at
    half its peak, within its marks' rows and pad line heights round them (a neighbouring line's rows stay out). The
    profile over `wide` line heights round them is kept for refine(), and so are its column profile over its rows.
    In place: L["top"], L["bot"], L["h"], L["cy"], L["prof"] (first row, profile), L["cols"] (first column, profile)."""
    H_, W_ = M.shape
    for L in lines:
        h = max(L["y1"] - L["y0"], 2)
        x0, x1 = max(0, L["x0"]), min(W_, L["x1"])
        n0, n1 = max(0, int(L["y0"] - pad * h)), min(H_, int(np.ceil(L["y1"] + pad * h)))
        prof = M[n0:n1, x0:x1].sum(1)
        if prof.max() <= 0:
            continue
        pk = int(np.argmax(prof))
        a = b = pk
        while a - 1 >= 0 and prof[a - 1] >= 0.5 * prof[pk]:
            a -= 1
        while b + 1 < len(prof) and prof[b + 1] >= 0.5 * prof[pk]:
            b += 1
        L["top"], L["bot"] = float(n0 + a) - 0.5, float(n0 + b) + 0.5
        L["h"], L["cy"] = L["bot"] - L["top"], 0.5 * (L["top"] + L["bot"])
        w0, w1 = max(0, int(L["y0"] - wide * h)), min(H_, int(np.ceil(L["y1"] + wide * h)))
        L["prof"] = (w0, M[w0:w1, x0:x1].sum(1))
        c0, c1 = max(0, int(x0 - 2 * h)), min(W_, int(x1 + 2 * h))
        L["cols"] = (c0, M[int(round(L["top"])):int(round(L["bot"])) + 1, c0:c1].sum(0))
        L["c_own"] = (x0, x1)


REFINE_SCALES = np.arange(0.80, 1.2501, 0.02)


def _ncc1(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else -1.0


def refine(r, g, scale=None):
    """A matched pair's scale (or the given one) and rows (the true line's row profile, scaled about its centre and shifted, against the
    generated line's), then its shift along the row at that scale (their column profiles). -> {"scale", "dy", "dx",
    "ncc_rows", "ncc_cols"} (reference px)"""
    if "prof" not in r or "prof" not in g:
        return None
    r0, rp = r["prof"]
    g0, gp = g["prof"]
    yr = r0 + np.arange(len(rp), dtype=float)
    yg = g0 + np.arange(len(gp), dtype=float)
    h = max(r["h"], g["h"], 1.0)
    best = (-2.0, 1.0, 0.0)
    for s_ in (REFINE_SCALES if scale is None else [scale]):
        for dy in np.arange(-0.8 * h, 0.8 * h + 0.01, 0.25):
            # the true profile seen at the generated rows: y = cy_r + dy + s (yr - cy_r)
            t = np.interp((yg - r["cy"] - dy) / s_ + r["cy"], yr, rp, left=0.0, right=0.0)
            c = _ncc1(t, gp)
            if c > best[0]:
                best = (c, float(s_), float(dy))
    c_r, s_, dy = best
    # along the row: the true columns scaled about the line's start, against the generated line's
    c0, cp = r["cols"]
    q0, qp = g["cols"]
    xr = c0 + np.arange(len(cp), dtype=float)
    xg = q0 + np.arange(len(qp), dtype=float)
    ax = float(r["x0"])
    bx = (-2.0, 0.0)
    for dx in np.arange(-2.0 * h, 2.0 * h + 0.01, 0.5):
        t = np.interp((xg - ax - dx) / s_ + ax, xr, cp, left=0.0, right=0.0)
        c = _ncc1(t, qp)
        if c > bx[0]:
            bx = (c, float(dx))
    return {"scale": s_, "dy": dy, "dx": bx[1], "ncc_rows": round(c_r, 3), "ncc_cols": round(bx[0], 3)}


# ---- matching the true lines to the generated ones, and placing them ----

MATCH_ROWS = 0.9       # a true line and a generated one match only with their centres this many line heights apart
MATCH_SIZE = 1.7       # ... their heights this many times apart at most
MATCH_SPAN = 0.3       # ... and their runs along the row overlapping by this share of the shorter
SIZE_CLASS = 0.12      # true lines this near in height are one size of type: they keep one size
SCALE_MAX = 0.08       # a size class scales by at most 1 +- this onto the generated lines (the fit and the
                       # registration set the type's size; blurred small type reads up to 20 % large)
SCALE_AGREE = 2        # ... and only when this many of its matched lines agree (rows correlating 0.7 or more)
MIN_GAP = 0.4          # placed lines in a block keep this share of their gap on the packshot


def _span_overlap(a, b):
    ov = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
    return ov / float(max(1, min(a["x1"] - a["x0"], b["x1"] - b["x0"])))


def match(lr, lg):
    """The true lines matched one to one with the generated lines under them, keeping their order down the label
    within a column. -> {true line k: generated line k}"""
    cand = []
    for r in lr:
        for g in lg:
            h = max(r["h"], g["h"], 1.0)
            dc = abs(r["cy"] - g["cy"]) / h
            ratio = max(r["h"], g["h"]) / max(min(r["h"], g["h"]), 1.0)
            ov = _span_overlap(r, g)
            if dc > MATCH_ROWS or ratio > MATCH_SIZE or ov < MATCH_SPAN:
                continue
            cand.append((dc + 0.6 * np.log(ratio) + 0.5 * (1 - min(ov, 1.0)), r["k"], g["k"]))
    cand.sort()
    pairs, used = {}, set()
    for c, rk, gk in cand:
        if rk in pairs or gk in used:
            continue
        # order: no crossing with a pair already taken in the same column (their runs overlap)
        r, g = lr[rk], lg[gk]
        cross = False
        for rk2, gk2 in pairs.items():
            r2, g2 = lr[rk2], lg[gk2]
            if _span_overlap(r, r2) < 0.2:
                continue
            if (r["cy"] - r2["cy"]) * (g["cy"] - g2["cy"]) < 0:
                cross = True
                break
        if not cross:
            pairs[rk] = gk
            used.add(gk)
    return pairs


BLOCK_GAP = 1.5        # a line starts a new block below a gap of more than this many line heights


def _blocks(lr):
    """True lines in blocks: lines whose runs along the row overlap, each within BLOCK_GAP line heights of the block's
    last line above it (a label's top lines, its wordmark and its lower lines are three blocks). -> [[k, ...], ...]"""
    blocks = []
    for L in sorted(lr, key=lambda L: L["cy"]):
        for b in blocks:
            last = lr[b[-1]]
            if _span_overlap(L, last) >= 0.3 and L["top"] - last["bot"] <= BLOCK_GAP * max(L["h"], last["h"]):
                b.append(L["k"])
                break
        else:
            blocks.append([L["k"]])
    return blocks


def _alignment(lines):
    """A block's alignment on the packshot: the edge its lines line up on ("x0" left, "xc" centre, "x1" right)."""
    if len(lines) < 2:
        return "x0"
    sd = {"x0": np.std([L["x0"] for L in lines]), "x1": np.std([L["x1"] for L in lines]),
          "xc": np.std([0.5 * (L["x0"] + L["x1"]) for L in lines])}
    return min(sd, key=sd.get)


def _anchor(L, how):
    return 0.5 * (L["x0"] + L["x1"]) if how == "xc" else float(L[how])


def place(lr, lg, pairs):
    """Each true line's placement (reference px -> reference px: a scale about its start and its centre row, and a
    shift) onto the generated line it matched (refine(): the row and column profiles). A size of type (true lines
    within SIZE_CLASS of one height) keeps one scale, its matched lines' median (1 +- SCALE_MAX); each line its own
    rows at that scale; a block (a column of lines) one shift along the row, its matched lines' median, so its
    alignment holds. A line with no match: its class's scale (or all matched lines'), its rows shifted as its matched
    neighbours' in the block. -> ({k: 3 x 3}, {k: report})"""
    order = sorted(lr, key=lambda L: L["h"])
    cls, cur = {}, []
    for L in order:
        if cur and L["h"] > (1 + SIZE_CLASS) * lr[cur[0]]["h"]:
            for k in cur:
                cls[k] = cur[0]
            cur = []
        cur.append(L["k"])
    for k in cur:
        cls[k] = cur[0]
    free = {k: refine(lr[k], lg[g]) for k, g in pairs.items()}
    sc = {}
    for k, f in free.items():
        if f and f["ncc_rows"] >= 0.7:
            sc.setdefault(cls[k], []).append(f["scale"])
    all_s = [v for vs in sc.values() for v in vs]
    s_all = float(np.clip(np.median(all_s), 1 - SCALE_MAX, 1 + SCALE_MAX)) if len(all_s) >= SCALE_AGREE else 1.0
    s_cls = {c: float(np.clip(np.median(v), 1 - SCALE_MAX, 1 + SCALE_MAX)) for c, v in sc.items()
             if len(v) >= SCALE_AGREE}
    A, rep, dy, dxs = {}, {}, {}, {}
    for k, g in pairs.items():
        s_ = s_cls.get(cls[k], s_all)
        f = refine(lr[k], lg[g], s_)
        if f and f["ncc_rows"] >= 0.3:
            dy[k] = f["dy"]
            dxs[k] = (f["dx"], f["ncc_cols"])
        rep[k] = {"scale": round(s_, 3), "matched": g, "fit": f}
    for blk in _blocks(lr):
        got = [dxs[k] for k in blk if k in dxs and dxs[k][1] >= 0.3]
        dx = float(np.median([d for d, _ in got])) if got else 0.0
        matched = [k for k in blk if k in dy]
        for k in blk:
            L = lr[k]
            s_ = s_cls.get(cls[k], s_all)
            if k in dy:
                dy_k = dy[k]
            elif matched:
                ys = np.array([lr[m]["cy"] for m in matched])
                ds = np.array([dy[m] for m in matched])
                o = np.argsort(ys)
                dy_k = float(np.interp(L["cy"], ys[o], ds[o]))
            else:
                dy_k = 0.0
            r = rep.setdefault(k, {"scale": round(s_, 3), "matched": None})
            if k in pairs and k not in dy:
                r["matched_but_weak"] = True
            ax, ay = float(L["x0"]), L["cy"]
            M = np.eye(3)
            M[0, 0] = M[1, 1] = s_
            M[0, 2] = ax + dx - s_ * ax
            M[1, 2] = ay + dy_k - s_ * ay
            A[k] = M
            r.update(dx=round(dx, 2), dy=round(dy_k, 2), placed="own rows" if k in dy else "between neighbours")
        _keep_apart(lr, blk, A, rep)
    return A, rep


def _keep_apart(lr, blk, A, rep, rounds=20):
    """Lines of a block, placed, keep MIN_GAP of their packshot gap: two that come closer move together (both take
    their mean shift down the label) until none do. In place."""
    def rows(k):
        M = A[k]
        return M[1, 1] * lr[k]["top"] + M[1, 2], M[1, 1] * lr[k]["bot"] + M[1, 2]
    order = sorted(blk, key=lambda k: lr[k]["cy"])
    for _ in range(rounds):
        moved = False
        for a, b in zip(order, order[1:]):
            if _span_overlap(lr[a], lr[b]) < 0.2:
                continue
            gap0 = lr[b]["top"] - lr[a]["bot"]
            _, abot = rows(a)
            btop, _ = rows(b)
            if btop - abot < MIN_GAP * max(gap0, 0.0) - 1e-6 or btop < abot:
                m = 0.5 * (A[a][1, 2] - (1 - A[a][1, 1]) * lr[a]["cy"] + A[b][1, 2] - (1 - A[b][1, 1]) * lr[b]["cy"])
                for k in (a, b):
                    A[k][1, 2] = m + (1 - A[k][1, 1]) * lr[k]["cy"]
                    rep[k]["dy"] = round(float(m), 2)
                    rep[k]["kept_apart"] = True
                moved = True
        if not moved:
            break


# ---- the true print placed line by line (ink.prepare with matte "text") ----

def _comp_lines(cov_r, tl_r):
    """The packshot coverage of each line of text: its letters' marks (at the marks' own level: fainter coverage joins
    a letter to a label's shaded edge or a band) grown by their anti-aliased rim, up to 2 px, within the coverage.
    -> int image of line number + 1 (0: no line, a rule, an edge, a lone mark)"""
    out = tl_r["line_map"].astype(np.int32)
    if not (out > 0).any():
        return out
    d, (iy, ix) = ndi.distance_transform_edt(out == 0, return_distances=True, return_indices=True)
    other = ndi.binary_dilation(tl_r["other"] | tl_r["edge"] | tl_r["lone"], iterations=1)
    rim = (out == 0) & (cov_r > 0) & (d <= 2.0) & ~other
    out[rim] = out[iy, ix][rim]
    return out


def _roi_of(Hm, box, roi, pad):
    """The scene work-box sub-box a reference box [x0, y0, x1, y1] maps into, padded and clipped. -> roi or None"""
    from .geometry import apply_H
    x0, y0, x1, y1 = box
    xs = np.linspace(x0, x1, 9)
    ys = np.linspace(y0, y1, 9)
    pts = np.array([[x, y] for x in xs for y in ys], float)
    q = apply_H(Hm, pts)
    a0 = int(np.floor(q[:, 0].min() - pad)), int(np.floor(q[:, 1].min() - pad))
    a1 = int(np.ceil(q[:, 0].max() + pad)) + 1, int(np.ceil(q[:, 1].max() + pad)) + 1
    r = [max(a0[0], roi[0]), max(a0[1], roi[1]), min(a1[0], roi[2]), min(a1[1], roi[3])]
    return r if r[2] > r[0] and r[3] > r[1] else None


def ref_per_scene(Hm, cov_r):
    """Packshot px per scene px at the print's centre (the mapping's local scale)."""
    from .geometry import _local_affine
    ys, xs = np.nonzero(cov_r > 0.5)
    c = np.array([xs.mean(), ys.mean()]) if len(xs) else np.array(cov_r.shape[::-1], float) / 2
    J = _local_affine(LineMap(Hm, np.eye(3)), c)[:2, :2]
    return float(1.0 / np.sqrt(max(abs(np.linalg.det(J)), 1e-9)))


class Turned90:
    """The packshot read a quarter turn round (its text runs up or down it: a tube lying on its side): text-frame px
    (np.rot90 of the packshot's arrays) to the packshot's, then the fit. forward() / inverse() as a fit's."""

    def __init__(self, Hm, width):
        self.Hm, self.W = Hm, float(width)

    def forward(self, pts):
        from .geometry import apply_H
        p = np.atleast_2d(np.asarray(pts, float))
        return apply_H(self.Hm, np.c_[self.W - 1 - p[:, 1], p[:, 0]])

    def inverse(self, X, Y):
        x, y, ok = _inverse(self.Hm, X, Y)
        return y, self.W - 1 - x, ok


def _profile_score(M):
    """How sharply a print's lines stand apart along the rows: the squared coefficient of variation of its row
    profile (lines and the gaps between them), over the band's rows that hold print."""
    prof = M.sum(1)
    on = prof > 0
    if on.sum() < 4:
        return 0.0
    v = prof[np.nonzero(on)[0].min():np.nonzero(on)[0].max() + 1]
    return float(v.var() / max(v.mean() ** 2, 1e-12))


def orientation(cov_r, band):
    """Whether the packshot's lines of text run along its rows (0) or along its columns (1: a quarter turn), from
    which way its letters' projection profile separates into lines more sharply (the classic orientation test: lines
    of text along the rows give a row profile of peaks and gaps, their column profile is flat by comparison). Marks
    in no line (rules, edges, lone marks) are left out. -> (k, {"rows": score, "columns": score})"""
    t = text_lines(cov_r, band, sharp=True)
    T = cov_r * (_comp_lines(cov_r, t) > 0) if t["lines"] else cov_r * band
    t1 = text_lines(np.rot90(cov_r), np.rot90(band), sharp=True)
    T1 = np.rot90(cov_r) * (_comp_lines(np.rot90(cov_r), t1) > 0) if t1["lines"] else np.rot90(cov_r * band)
    a, b = _profile_score(T), _profile_score(T1)
    return (1 if b > ORIENT_RATIO * a else 0), {"rows": round(a, 3), "columns": round(b, 3)}


ORIENT_RATIO = 1.5     # the quarter turn only when its profile separates lines this much more sharply


class Rotated:
    """A block of the packshot read turned by theta deg (its lines of text run at theta on it: a slanted headline
    block): text-frame px to the packshot's (x = R x' + t), then the fit. forward() / inverse() as a fit's."""

    def __init__(self, Hm, R, t):
        self.Hm, self.R, self.t = Hm, np.asarray(R, float), np.asarray(t, float)

    def to_ref(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        return p @ self.R.T + self.t

    def forward(self, pts):
        from .geometry import apply_H
        return apply_H(self.Hm, self.to_ref(pts))

    def inverse(self, X, Y):
        x, y, ok = _inverse(self.Hm, X, Y)
        dx, dy = x - self.t[0], y - self.t[1]
        R = self.R
        return R[0, 0] * dx + R[1, 0] * dy, R[0, 1] * dx + R[1, 1] * dy, ok


def _turn_frame(shape, band, theta):
    """The text frame of lines running at theta deg on the packshot (image rows down): R, t and the frame's shape
    covering the band's box. -> (R, t, (h, w))"""
    th = np.deg2rad(theta)
    c, s_ = np.cos(th), np.sin(th)
    R = np.array([[c, -s_], [s_, c]])
    ys, xs = np.nonzero(band)
    P = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.max(), ys.max()], [xs.min(), ys.max()]], float)
    q = P @ R                        # R^T p for each corner (rows)
    off = q.min(0) - 2
    size = np.ceil(q.max(0) - off).astype(int) + 3
    return R, R @ off, (int(size[1]), int(size[0]))


def _turn(M, R, t, shape, order=1):
    """An array read in the text frame: out[y', x'] = M[R (x', y') + t]."""
    # (row, col) order: [y, x] = [[c, s], [-s, c]] [y', x'] + [ty, tx]
    Mrc = np.array([[R[1, 1], R[1, 0]], [R[0, 1], R[0, 0]]])
    return ndi.affine_transform(M.astype(float) if order else M, Mrc, offset=[t[1], t[0]], output_shape=shape,
                                order=order, mode="constant")


def _back(Mt, R, t, shape, order=0):
    """A text-frame array read back on the packshot: out[y, x] = Mt[R^T ((x, y) - t)]."""
    Ri = R.T
    Mrc = np.array([[Ri[1, 1], Ri[1, 0]], [Ri[0, 1], Ri[0, 0]]])
    o = -Ri @ t
    return ndi.affine_transform(Mt.astype(float) if order else Mt, Mrc, offset=[o[1], o[0]], output_shape=shape,
                                order=order, mode="constant")


BLOCK_TURNS = tuple(a for a in range(-30, 31, 2) if abs(a) >= 4)   # deg: the turns a block of lines is read at
TURN_GAIN = 1.5        # a block's lines read at a turn hold this many times the marks of the longest level piece
TURN_MIN = 4           # ... and this many marks at least
TURN_BIN = 10.0        # deg: blocks whose turns lie this close are read in one frame (their lines keep their own turn)
TURN_SPREAD = 2.0      # a turned line's letters' centres spread over this many times more rows read level than turned


def angle_groups(cov_r, band):
    """Blocks of the packshot whose lines of text run at a turn (a slanted headline block): read level, a slanted
    line falls apart into short level pieces; read at its own turn it is one long line. Each level mark goes to the
    turn at which its line holds the most marks, when that is TURN_GAIN times its longest level piece (by marks, the
    level line it is in) and TURN_MIN marks or more. -> (level text_lines, [{"theta", "marks": level mark ids}], report)"""
    t0 = text_lines(cov_r, band, sharp=True, beside=True)
    lab0 = t0["lab"]
    n_lv = np.zeros(lab0.max() + 1, int)
    for L in t0["lines"]:
        n_lv[L["members"]] = len(L["members"])
    best_n = np.zeros(lab0.max() + 1, int)
    best_th = np.zeros(lab0.max() + 1, float)
    cands = []
    ys, xs = np.nonzero(band)
    sub = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    for th in BLOCK_TURNS:
        R, t, shp = _turn_frame(cov_r.shape, band, th)
        C = _turn(cov_r, R, t, shp)
        B = _turn(band, R, t, shp) > 0.5
        tt = text_lines(C, B, sharp=True, local=True, beside=True)
        by_t = {m["i"]: m for m in tt["marks"]}
        good = []
        for L in tt["lines"]:
            if len(L["members"]) < TURN_MIN or not type_like(L, tt)[0]:
                continue
            # its letters run on one row at this turn, and spread over rows read level (a level word read turned:
            # the other way round)
            ms = [by_t[i] for i in L["members"]]
            cx = np.array([0.5 * (m["x0"] + m["x1"]) for m in ms])
            cy = np.array([0.5 * (m["y0"] + m["y1"]) for m in ms])
            lv = np.c_[cx, cy] @ R.T + t
            if np.std(lv[:, 1]) < TURN_SPREAD * np.std(cy) + 0.25 * L["h"]:
                continue
            good.append(L)
        if not good:
            continue
        lut = np.zeros(tt["lab"].max() + 1, np.int32)
        for j, L in enumerate(good, 1):
            lut[L["members"]] = j
        back = _back(lut[tt["lab"]], R, t, cov_r.shape)
        a, b = lab0[sub].ravel(), back[sub].ravel()
        on = (a > 0) & (b > 0)
        if not on.any():
            continue
        # each turned line's level marks (the level marks under its letters), and its length against theirs read
        # level: the median of their level lines' (a curved headline's first word sat in a level line with the
        # next word's end and an icon)
        for j, L in enumerate(good, 1):
            mine = np.unique(a[on & (b == j)])
            if not len(mine):
                continue
            n = len(L["members"])
            if n < TURN_GAIN * max(float(np.median(np.maximum(n_lv[mine], 1))), 1.0):
                continue
            cands.append((n, -abs(th), th, mine))
    cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
    for n, _, th, mine in cands:
        free = mine[best_n[mine] == 0]
        if len(free) < TURN_MIN:
            continue
        best_n[free] = n
        best_th[free] = th
    ids = np.nonzero(best_n > 0)[0]
    groups = []
    for i in ids:
        for g in groups:
            if abs(g["theta"] - best_th[i]) <= TURN_BIN:
                g["marks"].append(int(i))
                g["thetas"].append(float(best_th[i]))
                break
        else:
            groups.append({"theta": float(best_th[i]), "marks": [int(i)], "thetas": [float(best_th[i])]})
    for g in groups:
        g["theta"] = float(np.median(g["thetas"]))
        del g["thetas"]
    groups = [g for g in groups if len(g["marks"]) >= TURN_MIN]
    rep = [{"theta": round(g["theta"], 1), "marks": len(g["marks"])} for g in groups]
    return t0, groups, rep


def _merge_placed(parts, shape):
    """place_print_level()'s results for several frames (level, then each turned block), as one: the text maps of the
    block a work-box px lies in (its text zone), the level frame's elsewhere; line numbers renumbered in order."""
    cov = np.zeros(shape)
    info, rep = None, None
    off = 0
    own = np.zeros(shape, bool)
    for j, (c, inf, rp, theta) in enumerate(parts):
        cov += c
        nl = np.where(inf["new_line"] > 0, inf["new_line"] + off, 0)
        gl = np.where(inf["gen_line"] > 0, inf["gen_line"] + off, 0)
        lr = [dict(L, k=L["k"] + off, theta=theta) for L in inf["lines_true"]]
        place = {k + off: v for k, v in (inf.get("place") or {}).items()}
        gated = {k + off: v for k, v in (inf.get("gated") or {}).items()}
        if info is None:
            info = dict(inf, new_line=nl, gen_line=gl, lines_true=lr, place=place, gated=gated,
                        placed_px=float(c.sum()), expect_px=float(c.sum()) / max(inf["placed_share"], 1e-6))
            rep = dict(rp, turns=[])
        else:
            z = inf["text_zone"] & ~own
            for key in ("gen_text", "gen_other", "gen_lone", "gen_icon"):
                info[key] = np.where(z, inf[key], info[key])
            info["gen_line"] = np.where(z, gl, info["gen_line"])
            info["new_line"] = np.where(nl > 0, nl, info["new_line"])
            info["text_zone"] = info["text_zone"] | inf["text_zone"]
            for key in ("gen_kept", "gen_kept_type", "gen_kept_gfx"):
                info[key] = info[key] | inf[key]
            info["lines_true"] = info["lines_true"] + lr
            info["place"].update(place)
            info["gated"].update(gated)
            info["rim_px"] = info["rim_px"] + inf["rim_px"]
            info["placed_px"] += float(c.sum())
            info["expect_px"] += float(c.sum()) / max(inf["placed_share"], 1e-6)
            rep["turns"].append({"theta": theta, "lines": rp.get("lines"), "gated": rp.get("gated"), "legible": rp.get("legible"),
                                 "true": rp.get("true"), "off_text": rp.get("off_text")})
        if j:
            own |= inf["text_zone"]
        off += len(inf["lines_true"])
    info["placed_share"] = info["placed_px"] / max(info["expect_px"], 1e-6)
    rep["placed_share"] = round(info["placed_share"], 3)
    return np.clip(cov, 0, 1), info, rep


def place_print(cov_r, band, Hm, roi, crop_u8, in_band, cfg, kw, stroke=None):
    """place_print_level() in the frame where the packshot's lines of text run level (orientation()): a quarter turn
    round when they run up its columns; the cover comes back in the work box either way. -> as place_print_level()"""
    k, score = orientation(cov_r, band)
    if k:
        C, Bd, Hb = (np.ascontiguousarray(np.rot90(cov_r)), np.ascontiguousarray(np.rot90(band)),
                     Turned90(Hm, cov_r.shape[1]))
    else:
        C, Bd, Hb = cov_r, band, Hm
    groups, grep = [], []
    if TURNS and printed_body(cfg):
        t0, groups, grep = angle_groups(C, Bd)
    if not groups:
        cov, info, rep = place_print_level(C, Bd, Hb, roi, crop_u8, in_band, cfg, kw, stroke)
    else:
        # the level frame without the turned blocks' marks, then each turned block in its own frame (its marks only)
        lab0 = t0["lab"]
        turned = np.isin(lab0, [i for g in groups for i in g["marks"]])
        turned = ndi.binary_dilation(turned, iterations=2) & (C > 0)
        parts = [place_print_level(np.where(turned, 0.0, C), Bd, Hb, roi, crop_u8, in_band, cfg, kw, stroke) + (0.0,)]
        for g in groups:
            R, t, shp = _turn_frame(C.shape, Bd, g["theta"])
            mine = ndi.binary_dilation(np.isin(lab0, g["marks"]), iterations=2) & (C > 0)
            Cg = _turn(np.where(mine, C, 0.0), R, t, shp)
            Bg = _turn(Bd, R, t, shp) > 0.5
            parts.append(place_print_level(Cg, Bg, Rotated(Hb, R, t), roi, crop_u8, in_band, cfg, kw, stroke,
                                           local=True) + (g["theta"],))
        cov, info, rep = _merge_placed(parts, parts[0][0].shape)
    if info.get("gen_kept") is not None and info["gen_kept"].any():
        cov, rep["on_kept_dropped"] = _drop_on_kept(cov, info)
        # graphics are kept where no new letter goes: a generated mark under a new line there was its garbled
        # letters run into a ray or a glint
        info["gen_kept_gfx"] = info["gen_kept_gfx"] & ~ndi.binary_dilation(cov > 0.05, iterations=2)
        info["gen_kept"] = info["gen_kept_type"] | info["gen_kept_gfx"]
    rep["orientation"] = dict(score, turned=bool(k), blocks=grep)
    return cov, info, rep


ON_KEPT = 0.3          # a placed line with this share of its letters on generated marks the gate kept (the legible
                       # rest of its word, read in another frame or part) stays as generated too


def _drop_on_kept(cov, info):
    """Placed lines landing on the generated print the gate kept: taken off again (one word isn't half redrawn over
    itself). -> (cover, [line numbers dropped])"""
    nl, kept = info["new_line"], info["gen_kept_type"]
    out = []
    for j in np.unique(nl[nl > 0]):
        m = nl == j
        if (m & kept).sum() >= ON_KEPT * m.sum():
            cov = np.where(ndi.binary_dilation(m, iterations=1), 0.0, cov)
            nl[m] = 0
            info["gen_kept_type"] = info["gen_kept_type"] | m
            out.append(int(j) - 1)
    return cov, out


def place_print_level(cov_r, band, Hm, roi, crop_u8, in_band, cfg, kw, stroke=None, local=False):
    """The packshot's print, letters only, placed line by line onto the generated print (ink.prepare's text step).
    cov_r: the packshot's print as coverage (ref_cover); Hm: the fit with the print's registration over it; kw:
    warp_cover's arguments. The frame's print map is read inside the warped band and pulled back to the packshot's
    frame, where both prints' lines run level; the rules, edge marks and lone marks of either stay out.
    -> (cover in the work box, info for garbled(): the generated letters, the generated other marks, the generated and
    the new line maps in the work box, the pairs; report)"""
    from . import images as im
    from . import ink
    dark = cfg["polarity"] == "dark"
    pb = printed_body(cfg)
    tl_r = text_lines(cov_r, band, sharp=True, local=local, beside=pb)
    comp = _comp_lines(cov_r, tl_r)
    Lf = im.lum(im.srgb_to_lin(im.to_float(crop_u8)))
    B = in_band > 0.5
    so = max(7, int(np.ceil(1.5 * stroke)) | 1) if stroke else 15
    f_px = int(cfg.get("_px_scale") or 1)
    if f_px > 1:
        # a frame resized up for the fix (work scale, pitfall 113): its print read at the frame's own size, as the
        # marks' kinds were tuned (at 4x a wordmark's descender ran into the band under it, and the band went with
        # the letters), then laid back over the work size
        H_, W_ = Lf.shape
        h_, w_ = -(-H_ // f_px), -(-W_ // f_px)
        Lp = np.pad(Lf, ((0, h_ * f_px - H_), (0, w_ * f_px - W_)), mode="edge")
        Bp = np.pad(B, ((0, h_ * f_px - H_), (0, w_ * f_px - W_)), mode="edge")
        Ls = Lp.reshape(h_, f_px, w_, f_px).mean((1, 3))
        Bs = Bp.reshape(h_, f_px, w_, f_px).mean((1, 3)) > 0.5
        so_s = max(7, int(np.ceil(so / f_px)) | 1)
        Pq = ink.print_map(Ls, Bs, so_s, dark)[0] * Bs
        P_s = ndi.zoom(Pq, f_px, order=1)[:H_, :W_] * B
    else:
        P_s = ink.print_map(Lf, B, so, dark)[0] * B
    P_b = pull_back(P_s, Hm, roi, band)
    tl_g = text_lines(P_b, band, thresh=GEN_THRESH, local=local, beside=pb)
    lr, lg = tl_r["lines"], tl_g["lines"]
    # both prints blurred alike for their rows: the frame's print edge blur (scene px) in packshot px, over the
    # packshot's own
    s_ps = ref_per_scene(Hm, cov_r)
    near = B & ndi.binary_dilation(P_s > 0.3, iterations=2)
    sig_f = ink.edge_sigma(P_s, near)
    sig_r = ink.edge_sigma(cov_r, band)
    sig = float(np.sqrt(max((sig_f * s_ps) ** 2 - (sig_r if np.isfinite(sig_r) else 0.0) ** 2, 0.0))) \
        if np.isfinite(sig_f) else 0.0
    R_b = ndi.gaussian_filter(cov_r * (comp > 0), sig) if sig > 0.3 else cov_r * (comp > 0)
    A, prep = place_blocks(lr, R_b, P_b)
    bends = row_courses(lr, R_b, P_b, A, prep)
    pairs = {}
    import os
    if os.environ.get("PLR_TEXT_DEBUG"):
        os.environ["PLR_TEXT_DEBUG"] = os.environ["PLR_TEXT_DEBUG"].split("@")[0].replace(".npz", "") + \
            f"@{cfg['polarity']}.npz"
        np.savez_compressed(os.environ["PLR_TEXT_DEBUG"], P_b=P_b, R_b=R_b, cov_r=cov_r, band=band,
                            gl=tl_g["line_map"], go=tl_g["other"], rl=tl_r["line_map"], ro=tl_r["other"])
    x0, y0, x1, y1 = roi
    cov = np.zeros((y1 - y0, x1 - x0))
    new_line = np.zeros(cov.shape, np.int32)
    best = np.zeros(cov.shape)
    kw = dict(kw, extra=None)
    # the generated label's own rules and bands in the work box: a short true "line" (one or two marks) that lands
    # mostly on them is the packshot's own band end read as a word, not text (pitfall 113: a grey smudge on the
    # generated band's end)
    on_rules = push(tl_g["other"], Hm, roi) > 0
    dropped, straight = [], []
    leg = {}
    placed_h = []
    gate = GATE and printed_body(cfg)
    gated = {}
    kept_wb = np.zeros(cov.shape, bool)
    kept_gfx = np.zeros(cov.shape, bool)
    gen_lab = ndi.label(P_s > GEN_THRESH, structure=np.ones((3, 3), bool))[0] if gate else None
    for L in lr:
        k = L["k"]
        ck = cov_r * (comp == k + 1)
        ys, xs = np.nonzero(ck > 0)
        if not len(ys):
            continue
        lm = LineMap(Hm, A[k])
        if k in bends:
            # onto its generated row's own course: turned (A) and bent as that row runs (pitfall 115)
            lm = BentLineMap(Hm, A[k], lambda uu, c_=bends[k]: np.polyval(c_, np.asarray(uu, float)))
        else:
            # curved as the fit lays it, or straight where the generator drew the line straight (a flat panel fitted
            # round a cylinder bowed the lines up at their ends, the user: "skewed", pitfall 113)
            sm = StraightLineMap(lm, [0.5 * (L["x0"] + L["x1"]), L["cy"]])
            s_c, s_s = _path_sharpness(L, lm, P_s, roi[:2]), _path_sharpness(L, sm, P_s, roi[:2])
            if s_s > STRAIGHT_GAIN * s_c:
                lm = sm
                straight.append(k)
        sub = _roi_of(lm, [xs.min() - 2, ys.min() - 2, xs.max() + 3, ys.max() + 3], roi, 4)
        if sub is None:
            continue
        c, _ = ink.warp_cover(ck, band, lm, sub, **kw)
        sl = (slice(sub[1] - y0, sub[3] - y0), slice(sub[0] - x0, sub[2] - x0))
        if len(L.get("members") or []) <= 2 and (c * on_rules[sl]).sum() >= RULE_LANDING * max(c.sum(), 1e-9):
            dropped.append(k)
            continue
        hs_ = float(np.ptp(np.nonzero(c > 0.3)[0]) + 1) if (c > 0.3).any() else 4.0
        leg[k] = legibility(c, P_s, sl, hs_)
        if gate:
            # garbled type only: a line the generator already drew legibly, or a "line" that isn't type (glints,
            # rays, icons on a fully printed body), stays as generated, its footprint kept from the fill; a line
            # whose parts stand apart (a word, an icon after it) is decided part by part
            segs = segments(L, tl_r)
            dec = []
            for sg in segs:
                if len(segs) == 1:
                    cs, sls, lg_ = c, sl, leg[k]
                else:
                    mk = (comp == k + 1) & (ndi.binary_dilation(np.isin(tl_r["lab"], sg["members"]), iterations=2))
                    cs, _ = ink.warp_cover(ck * mk, band, lm, sub, **kw)
                    sls = sl
                    hs2 = float(np.ptp(np.nonzero(cs > 0.3)[0]) + 1) if (cs > 0.3).any() else hs_
                    lg_ = legibility(cs, P_s, sl, hs2)
                is_type, feats = type_like(sg, tl_r)
                if len(sg["members"]) == 1:
                    m1 = tl_r["marks"][[q["i"] for q in tl_r["marks"]].index(sg["members"][0])]
                    is_type = is_type and m1["w"] >= WORD_ASPECT * m1["h"]
                why = None if is_type else "not type"
                if why is None and lg_ is not None and lg_[0] >= LEGIBLE:
                    why = "legible"
                dec.append((sg, cs, lg_, why, feats))
            if any(d[3] for d in dec):
                c = np.zeros_like(c)
                for sg, cs, lg_, why, feats in dec:
                    if not why:
                        c = c + cs
                        continue
                    gated.setdefault(k, []).append(
                        {"why": why, "legible": lg_[0] if lg_ else None, "x": [sg["x0"], sg["x1"]],
                         **{q: feats[q] for q in ("stroke_w", "h_rel_med", "fill_w")}})
                    dy_, dx_ = (lg_[2] if lg_ else (0, 0))
                    fp = ndi.shift((cs > 0.2).astype(np.uint8), (dy_, dx_), order=0) > 0
                    reach = fp
                    if why == "not type":
                        # graphics the generator drew its own way (an icon off the packshot's place): the marks near
                        reach = ndi.binary_dilation(fp, iterations=int(max(2, round(0.3 * hs_))))
                    if why == "legible":
                        kept_wb[sls] |= _kept_marks(reach, gen_lab[sls]) | fp
                    else:
                        kept_gfx[sls] |= _kept_marks(reach, gen_lab[sls], inside=True,
                                                     min_px=GFX_SIZE * fp.sum() / max(len(sg["members"]), 1))
                if c.sum() <= 0:
                    continue
        placed_h.append(hs_)
        cov[sl] += c
        win = c > best[sl]
        new_line[sl] = np.where(win & (c > 0.02), k + 1, new_line[sl])
        best[sl] = np.maximum(best[sl], c)
    cov = np.clip(cov, 0, 1)
    if gated:
        # the kept marks' own soft edges and glow with them (filled, a halo showed round a kept headline)
        e_ = int(round(KEPT_EDGE * (cfg.get("_px_scale") or 1)))
        kept_wb = ndi.binary_dilation(kept_wb, iterations=e_)
        kept_gfx = ndi.binary_dilation(kept_gfx, iterations=e_)
    # text_rim's measure: the placed lines' packshot letters within max(3, 0.3 letter heights) of the packshot label's
    # own outline (its printed edge or a corner's shading read as a letter, pitfall 114)
    typ_r = float(np.median([m["h"] for m in tl_r["marks"]])) if tl_r["marks"] else 1.0
    rim_r = ndi.distance_transform_edt(band) <= max(3.0, 0.3 * typ_r)
    placed_k = [L["k"] + 1 for L in lr if L["k"] not in dropped and L["k"] not in gated]
    rim_px = float((cov_r * np.isin(comp, placed_k) * rim_r).sum())
    # how much of the packshot's letters went on: the new cover's px against the true letters' px at each line's
    # scale (packshot px per scene px squared), lines cut off at the band's or the work box's edge falling short
    expect = sum(float((cov_r * (comp == L["k"] + 1)).sum()) * (A[L["k"]][0, 0] / s_ps) ** 2 for L in lr
                 if L["k"] not in gated)
    placed = float(cov.sum()) / max(expect, 1e-6)
    # where text is: each block of true lines' box, a line height to spare along the row (garbled lines run longer)
    # and most of a pitch above and below, inside the band clear of its edge (a label's own border, its shaded edge)
    zone_r = np.zeros(band.shape, bool)
    inner_b = ndi.binary_erosion(band, iterations=3)
    from . import raster
    bxs = np.nonzero(band.any(0))[0]
    for blk in _blocks(lr):
        ls = sorted([lr[k] for k in blk], key=lambda L: L["cy"])
        hm = float(np.median([L["h"] for L in ls]))
        cys = [L["cy"] for L in ls]
        pitch = float(np.median(np.diff(cys))) if len(cys) > 1 else 1.5 * hm
        # along the row to the band's ends (a generated line can run on past the true one, a curved one drop below
        # it at its end), from most of a pitch above the block's top line to below its last, as placed (turned)
        top, bot = ls[0], ls[-1]
        a, b = float(bxs.min()), float(bxs.max())
        y0_, y1_ = top["top"] - 0.8 * pitch, bot["bot"] + 0.8 * pitch
        corners = np.array([[a, y0_, 1], [b, y0_, 1], [b, y1_, 1], [a, y1_, 1]], float)
        q = np.r_[(A[top["k"]] @ corners[:2].T).T[:, :2], (A[bot["k"]] @ corners[2:].T).T[:, :2]]
        zone_r |= raster.fill_polygon(band.shape[0], band.shape[1], [tuple(v) for v in q]) > 0
    zone_r &= inner_b
    # the generated print's long marks off the rows the new lines take (turned with their block, across the band) are
    # the label's own lines and bands, wherever their edges run (a curved band reaching the label's border): kept
    rows_r = np.zeros(band.shape, bool)
    a_, b_ = float(bxs.min()), float(bxs.max())
    for L in lr:
        corners = np.array([[a_, L["top"], 1], [b_, L["top"], 1], [b_, L["bot"], 1], [a_, L["bot"], 1]], float)
        q = (A[L["k"]] @ corners.T).T[:, :2]
        rows_r |= raster.fill_polygon(band.shape[0], band.shape[1], [tuple(v) for v in q]) > 0
    bw_ = _band_width(band)
    long_ids = []
    for m in tl_g["marks"]:
        if m["w"] < RULE_LEN * bw_ and m["h"] < RULE_LEN * bw_:
            continue
        # a long mark mostly off the new lines' rows is a band or a line of the label; mostly on them, it is lines of
        # small type blurred into one mark, and goes whole
        sel_m = tl_g["lab"][m["y0"]:m["y1"], m["x0"]:m["x1"]] == m["i"]
        share = float((rows_r[m["y0"]:m["y1"], m["x0"]:m["x1"]] & sel_m).sum()) / max(1, int(sel_m.sum()))
        if share < LONG_ON_ROWS:
            long_ids.append(m["i"])
    # a "rule" short of the label's width lying inside a block's text area is a line of small type blurred solid
    # (pitfall 113: kept as a rule, the new lines went over it), not a band: bands run across the label between blocks
    for m in tl_g["marks"]:
        if m.get("kind") != "rule" or m["w"] >= BAND_FULL * bw_:
            continue
        sel_m = tl_g["lab"][m["y0"]:m["y1"], m["x0"]:m["x1"]] == m["i"]
        inzone = float((zone_r[m["y0"]:m["y1"], m["x0"]:m["x1"]] & sel_m).sum()) / max(1, int(sel_m.sum()))
        if inzone >= IN_BLOCK:
            whole = tl_g["lab"] == m["i"]
            tl_g["other"] = tl_g["other"] & ~whole
            tl_g["text"] = tl_g["text"] | whole
            long_ids = [i for i in long_ids if i != m["i"]]
    if long_ids:
        # pixel by pixel: a band the generated wordmark's descender runs into is one mark with it; its letters' part
        # lies on the new lines' rows and goes, the band's stays
        med_h = float(np.median([L["h"] for L in lr])) if lr else 4.0
        on_rows = ndi.binary_dilation(rows_r, iterations=max(1, int(round(0.15 * med_h))))
        tl_g["other"] = tl_g["other"] | (np.isin(tl_g["lab"], long_ids) & ~on_rows)
    text_zone = push(zone_r, Hm, roi) > 0
    gen_text = push(ndi.binary_dilation(tl_g["text"], iterations=1), Hm, roi) > 0
    if cfg.get("label"):
        # a label fitted in label mode: its measured outline keeps its border (ink.garbled); marks touching the
        # packshot outline's edge are letters the generator set nearer the label's edge
        tl_g["other"] = tl_g["other"] & ~(tl_g["edge"] & ~np.isin(tl_g["lab"], long_ids if long_ids else [-1]))
    gen_other = push(tl_g["other"], Hm, roi) > 0
    # icons: the generated print's lone marks (in no line of text) larger than the true print's letters
    med_h = float(np.median([L["h"] for L in lr])) if lr else 0.0
    icon_ids = []
    for m in tl_g["marks"]:
        if m.get("kind") != "lone" or max(m["h"], m["w"]) < 1.2 * med_h:
            continue
        # a lone mark on the rows a new line takes is a word blurred into one blob, not an icon
        sel_m = tl_g["lab"][m["y0"]:m["y1"], m["x0"]:m["x1"]] == m["i"]
        on = float((rows_r[m["y0"]:m["y1"], m["x0"]:m["x1"]] & sel_m).sum()) / max(1, int(sel_m.sum()))
        # ... nor is one with generated letters beside it in its own row (a line's first word blurred into a blob)
        beside = any(t.get("kind") == "text" and min(t["y1"], m["y1"]) - max(t["y0"], m["y0"]) >= 0.5 * min(t["h"], m["h"])
                     and max(t["x0"] - m["x1"], m["x0"] - t["x1"]) <= 2.0 * m["h"] for t in tl_g["marks"])
        if on < 0.5 and not beside:
            icon_ids.append(m["i"])
    gen_icon = push(np.isin(tl_g["lab"], icon_ids) if icon_ids else np.zeros(band.shape, bool), Hm, roi) > 0
    if os.environ.get("PLR_TEXT_DEBUG"):
        np.savez_compressed(os.environ["PLR_TEXT_DEBUG"].replace(".npz", "_gen.npz"), lab=tl_g["lab"],
                            other=tl_g["other"], edge=tl_g["edge"], lone=tl_g["lone"], text=tl_g["text"],
                            icons=np.isin(tl_g["lab"], icon_ids) if icon_ids else np.zeros(band.shape, bool),
                            rows=rows_r, P_b=P_b)
    if os.environ.get("PLR_TEXT_DEBUG"):
        np.savez_compressed(os.environ["PLR_TEXT_DEBUG"].replace(".npz", "_new.npz"), new_line=new_line, roi=np.array(roi))
    gen_lone = push(tl_g["lone"], Hm, roi) > 0
    gen_line = push(tl_g["line_map"], Hm, roi)
    info = {"gen_text": gen_text, "gen_other": gen_other, "gen_lone": gen_lone, "gen_icon": gen_icon,
            "line_h_px": float(np.median(placed_h)) if placed_h else None,
            "text_zone": text_zone, "roi": list(roi), "gen_kept": kept_wb | kept_gfx, "gen_kept_type": kept_wb,
            "gen_kept_gfx": kept_gfx, "gated": gated,
            "gen_line": gen_line,
            "new_line": new_line,
            "pairs": pairs, "lines_true": lr, "lines_gen": lg, "place": prep, "placed_share": placed, "rim_px": rim_px}
    rep = {"true": summary(tl_r), "generated": summary(tl_g), "placed_share": round(placed, 3), "matched": len(pairs),
           "rim_px": round(rim_px, 1),
           "blur": {"frame_px": round(float(sig_f), 3), "packshot_px": round(float(sig_r), 3),
                    "packshot_per_scene_px": round(s_ps, 3), "rows_blur_px": round(sig, 3)},
           "off_text": [k for k in range(len(lr)) if not prep.get(k, {}).get("on_text")],
           "on_rules_dropped": dropped, "straight": straight, "legible": leg, "gated": gated,
           "lines": [dict(k=L["k"], rows=[round(L["top"], 1), round(L["bot"], 1)], x=[L["x0"], L["x1"]],
                          **prep.get(L["k"], {})) for L in lr],
           "lines_generated": [dict(k=g["k"], rows=[round(g["top"], 1), round(g["bot"], 1)], x=[g["x0"], g["x1"]])
                               for g in lg]}
    rep.pop("matched", None)
    return cov, info, rep


LEGIBLE = 0.5          # a true line the generated print already draws (its letters' column structure correlating
                       # this well, legibility()) stays as generated: a fully printed sleeve's wordmark 0.71, headline
                       # 0.87, figure 0.88 against its garbled lines' 0.06-0.39
TYPE_STROKE = 0.35     # a true "line" whose marks' stroke (twice the widest inscribed radius, px-weighted) is this
                       # share of its core height or more is graphics (glints 0.44, rays 0.41, icons 0.52), not type
                       # (garbled lines 0.18-0.30)
TYPE_CORE = 0.5        # ... or whose marks' median height is under this share of its core (its core is a few big
                       # marks: rays, icons among their specks 0.17-0.2; type 0.93-1.18)
TYPE_MARKS = 3         # ... or holds fewer marks than this (a glossy body's specks paired into "lines": 2-3 specks) ...
TYPE_LEN = 1.5         # ... or runs shorter than this many core heights (specks: under one; a word of three letters 1.6)
TYPE_FILL = 0.35       # ... or whose marks fill under this share of their boxes (a glint's points 0.26; type 0.49-0.64)


SEG_GAP = 0.6          # a line's marks this many core heights apart or more are parts decided apart (the gate)


KEPT_EDGE = 2          # px (scene, at 1x): the kept marks' soft edges kept with them
KEPT_INSIDE = 0.6      # graphics kept: a generated mark this much inside their reach
GFX_SIZE = 0.3         # ... and at least this share of the graphic's mean mark area
KEPT_SHARE = 0.3       # a generated mark is a kept line's when this share of it lies under the line's letters


def _kept_marks(fp, lab, inside=False, min_px=0.0):
    """The generated print's marks (labels in the same window) a kept true line's letters lie on, whole: the share of
    each under the footprint KEPT_SHARE or more, or most of the footprint on it. inside (graphics, a footprint reaching
    round the true marks): only marks mostly inside it (KEPT_INSIDE; a garbled line merged into one long mark beside
    an icon was kept whole, and the new line over it dropped)."""
    ids = np.unique(lab[fp & (lab > 0)])
    if not len(ids):
        return fp & False
    tot = ndi.sum(np.ones_like(lab), lab, ids)
    on = ndi.sum(fp, lab, ids)
    if inside:
        # ... of a graphic's size: garbled letters by a ray lie inside its reach too
        keep = ids[(on >= KEPT_INSIDE * tot) & (tot >= min_px)]
        return np.isin(lab, keep)
    keep = ids[(on >= KEPT_SHARE * tot) | (on >= 0.5 * max(fp.sum(), 1))]
    return np.isin(lab, keep)


def segments(L, tl_r):
    """A true line's parts: its marks along the row, split where they stand SEG_GAP core heights apart.
    -> [{"members", "x0", "x1", "h"}]"""
    by_i = {m["i"]: m for m in tl_r["marks"]}
    ms = sorted([by_i[i] for i in L["members"]], key=lambda m: m["x0"])
    out, cur, right = [], [], None
    for m in ms:
        if cur and m["x0"] - right > SEG_GAP * L["h"]:
            out.append(cur)
            cur = []
        cur.append(m)
        right = max(right, m["x1"]) if right is not None and len(cur) > 1 else m["x1"]
    if cur:
        out.append(cur)
    return [{"members": [m["i"] for m in sg], "x0": min(m["x0"] for m in sg), "x1": max(m["x1"] for m in sg),
             "h": L["h"]} for sg in out]


def type_like(L, tl_r):
    """Whether a true line is a line of type (its marks letter-shaped, its core their height), from line_feats()."""
    f = line_feats(L, tl_r)
    return (f["stroke_w"] < TYPE_STROKE and f["h_rel_med"] >= TYPE_CORE and f["fill_w"] >= TYPE_FILL and
            (f["n"] >= TYPE_MARKS and f["len_h"] >= TYPE_LEN or f["n"] == 1)), f


def _ncc_shift(T, G, r):
    """The best normalised correlation of T against G shifted by up to r px each way (G is T's window padded by r)."""
    h, w = T.shape
    t = T - T.mean()
    tn = np.sqrt((t * t).sum())
    if tn < 1e-9:
        return -1.0, (0, 0)
    best, at = -1.0, (0, 0)
    for dy in range(0, 2 * r + 1):
        for dx in range(0, 2 * r + 1):
            g = G[dy:dy + h, dx:dx + w]
            g = g - g.mean()
            gn = np.sqrt((g * g).sum())
            if gn < 1e-9:
                continue
            v = float((t * g).sum() / (tn * gn))
            if v > best:
                best, at = v, (dy - r, dx - r)
    return best, at


def line_feats(L, tl_r):
    """The shape of a true line's marks: heights against its core, fill, stroke, holes."""
    by_i = {m["i"]: m for m in tl_r["marks"]}
    lab = tl_r["lab"]
    hs, fills, strokes, holes, pxs = [], [], [], [], []
    for i in L["members"]:
        m = by_i[i]
        sub = lab[m["y0"]:m["y1"], m["x0"]:m["x1"]] == i
        hs.append(m["h"] / max(L["h"], 1.0))
        fills.append(m["px"] / float(m["w"] * m["h"]))
        dt = ndi.distance_transform_edt(np.pad(sub, 1))
        strokes.append(2 * float(dt.max()) / max(L["h"], 1.0))
        filled = ndi.binary_fill_holes(sub)
        holes.append(int(ndi.label(filled & ~sub)[1]))
        pxs.append(m["px"])
    w = np.asarray(pxs, float)
    return {"n": len(hs), "len_h": round((L["x1"] - L["x0"]) / max(L["h"], 1.0), 2),
            "h_rel_med": round(float(np.median(hs)), 2), "h_rel_max": round(float(np.max(hs)), 2),
            "fill_w": round(float(np.average(fills, weights=w)), 2),
            "stroke_w": round(float(np.average(strokes, weights=w)), 3),
            "holes": int(sum(holes)), "big_px_share": round(float(w[np.asarray(hs) > 1.5].sum() / w.sum()), 2)}


def legibility(c, P_s, sl, h):
    """How well the generated print already draws a placed true line: the placed line's cover c (work-box slice sl)
    against the generated print map P_s over the same window, each with its row band taken out (the column
    structure: letters' shapes, which a garbled line of the same size and run doesn't share), shifted up to a quarter
    line height. -> (score, raw score, shift)"""
    ys, xs = np.nonzero(c > 0.05)
    if not len(ys):
        return None
    r = int(max(2, round(0.25 * h)))
    pad = int(max(2, round(0.4 * h)))
    H0, W0 = P_s.shape
    oy, ox = sl[0].start, sl[1].start
    a0, a1 = oy + ys.min() - pad, oy + ys.max() + 1 + pad
    b0, b1 = ox + xs.min() - pad, ox + xs.max() + 1 + pad
    if a0 - r < 0 or b0 - r < 0 or a1 + r > H0 or b1 + r > W0:
        a0, b0 = max(a0, r), max(b0, r)
        a1, b1 = min(a1, H0 - r), min(b1, W0 - r)
        if a1 - a0 < 3 or b1 - b0 < 3:
            return None
    T = np.zeros((a1 - a0, b1 - b0))
    cy0, cx0 = a0 - oy, b0 - ox
    ch, cw = c.shape
    ty0, tx0 = max(0, -cy0), max(0, -cx0)
    sy0, sx0 = max(0, cy0), max(0, cx0)
    sy1, sx1 = min(ch, cy0 + T.shape[0]), min(cw, cx0 + T.shape[1])
    T[ty0:ty0 + sy1 - sy0, tx0:tx0 + sx1 - sx0] = c[sy0:sy1, sx0:sx1]
    G = P_s[a0 - r:a1 + r, b0 - r:b1 + r]
    raw, at = _ncc_shift(T, G, r)
    sx = max(1.0, 0.6 * h)
    Tc = T - ndi.gaussian_filter1d(T, sx, axis=1)
    Gc = G - ndi.gaussian_filter1d(G, sx, axis=1)
    col, at2 = _ncc_shift(Tc, Gc, r)
    return round(col, 3), round(raw, 3), list(at2)


# ---- placement v2: each block of true lines onto the generated print's own rows ----

SLIDE = 0.75           # a line slides at most this many of its block's line pitches down or up the label
SMOOTH = 0.35          # the cost of two neighbouring lines sliding apart, per line height (against a row match of 1)
STEP = 0.5             # px (reference) between the slides tried
ON_TEXT = 0.5          # a placed line whose rows correlate this well with the generated print's sits on its text
BLOCK_SCALES = (0.94, 0.97, 1.0, 1.03, 1.06)   # a block's size tried round the fit's
LINE_TURNS = tuple(np.arange(-20.0, 20.01, 1.0))  # deg: each line's turn tried round the fit's (coarse, then 0.25 deg
                                                  # round the best): a leaning bottle's generated lines ran 7-15 deg off
                                                  # its label fit's, a wordmark 12 deg (pitfall 112)
TURN_CONF = 1.5        # a line's own turn counts when its row profile is this many times sharper at its best turn
                       # than at the median turn tried (the turn is visible in its rows) ...
TURN_LEN = 4.0         # ... and it runs this many line heights long (a short word's rows are sharp at many turns)
TURN_SPREAD = 1.5      # deg: a line's own turn stays within this of the label's trend (a robust line in its height
                       # down the label: the generated lines' fan); lines whose own turn isn't visible take the trend


def _rows_of(M, x0, x1, y0, y1):
    H_, W_ = M.shape
    x0, x1 = max(0, int(x0)), min(W_, int(x1))
    y0i, y1i = int(np.floor(y0)), int(np.ceil(y1))
    out = np.zeros(max(y1i - y0i, 1))
    a, b = max(0, y0i), min(H_, y1i)
    if b > a and x1 > x0:
        out[a - y0i:b - y0i] = M[a:b, x0:x1].sum(1)
    return y0i, out


def _cols_of(M, x0, x1, y0, y1):
    H_, W_ = M.shape
    y0, y1 = max(0, int(round(y0))), min(H_, int(round(y1)) + 1)
    x0i, x1i = int(np.floor(x0)), int(np.ceil(x1))
    out = np.zeros(max(x1i - x0i, 1))
    a, b = max(0, x0i), min(W_, x1i)
    if b > a and y1 > y0:
        out[a - x0i:b - x0i] = M[y0:y1, a:b].sum(0)
    return x0i, out


def _line_scores(L, R, G, s, dx, slides, pad=0.6):
    """How well one true line's rows (R: the true print blurred alike) match the generated print's (G) at each slide,
    the line scaled by s about its start and centre row and moved dx along the row. -> NCC per slide"""
    h = max(L["h"], 1.0)
    t0, tp = _rows_of(R, L["x0"], L["x1"], L["top"] - pad * h, L["bot"] + pad * h)
    yt = t0 + np.arange(len(tp), dtype=float)
    # generated columns the scaled line covers
    gx0, gx1 = L["x0"] + dx, L["x0"] + dx + s * (L["x1"] - L["x0"])
    out = np.full(len(slides), -1.0)
    for i, dy in enumerate(slides):
        # the generated rows where the scaled line's window lands
        a = L["cy"] + dy + s * (yt[0] - L["cy"])
        b = L["cy"] + dy + s * (yt[-1] - L["cy"])
        g0, gp = _rows_of(G, gx0, gx1, a, b + 1)
        yg = g0 + np.arange(len(gp), dtype=float)
        tt = np.interp((yg - L["cy"] - dy) / s + L["cy"], yt, tp, left=0.0, right=0.0)
        out[i] = _ncc1(tt, gp)
    return out


def _block_dx(lines, R, G, s, reach):
    """One shift along the row for a block: the true block's column profile (scaled by s about its lines' common
    start) against the generated print's over the same rows. -> (dx, ncc)"""
    x0 = min(L["x0"] for L in lines)
    x1 = max(L["x1"] for L in lines)
    y0 = min(L["top"] for L in lines)
    y1 = max(L["bot"] for L in lines)
    c0, cp = _cols_of(R, x0 - reach, x1 + reach, y0, y1)
    q0, qp = _cols_of(G, x0 - 2 * reach, x1 + 2 * reach, y0, y1)
    xr = c0 + np.arange(len(cp), dtype=float)
    xg = q0 + np.arange(len(qp), dtype=float)
    best = (-2.0, 0.0)
    for dx in np.arange(-reach, reach + 0.01, STEP):
        t = np.interp((xg - x0 - dx) / s + x0, xr, cp, left=0.0, right=0.0)
        c = _ncc1(t, qp)
        if c > best[0]:
            best = (c, float(dx))
    return best[1], best[0]


def _viterbi(lines, scores, slides, gaps):
    """The slides of a block's lines (down the label) maximising their row matches less SMOOTH per line height of
    two neighbours sliding apart, with each placed gap at least MIN_GAP of its packshot gap. -> (slide index per
    line, total)"""
    n, m = len(lines), len(slides)
    sl = np.asarray(slides)
    acc = scores[0].copy()
    back = np.zeros((n, m), int)
    for k in range(1, n):
        h = max(lines[k]["h"], lines[k - 1]["h"], 1.0)
        cost = SMOOTH * np.abs(sl[None, :] - sl[:, None]) / h           # [prev, cur]
        g0 = gaps[k - 1]
        # placed gap: g0 + (dy_cur - dy_prev) must stay >= MIN_GAP * g0
        bad = (g0 + sl[None, :] - sl[:, None]) < MIN_GAP * max(g0, 0.0) - 1e-6
        tot = acc[:, None] - cost - 1e3 * bad
        back[k] = np.argmax(tot, 0)
        acc = tot[back[k], np.arange(m)] + scores[k]
    idx = np.zeros(n, int)
    idx[-1] = int(np.argmax(acc))
    for k in range(n - 1, 0, -1):
        idx[k - 1] = back[k, idx[k]]
    return idx, float(acc.max())


def _line_box(L, G, rad):
    c = np.array([0.5 * (L["x0"] + L["x1"]), L["cy"]])
    r = 0.5 * np.hypot(L["x1"] - L["x0"], L["h"]) + rad
    return c, (int(max(c[0] - r, 0)), int(max(c[1] - r, 0)), int(min(c[0] + r + 1, G.shape[1])),
               int(min(c[1] + r + 1, G.shape[0])))


def _turn_sharpness(L, G, th):
    """How sharp the generated print's row profile is over a true line's run and the rows round it (its neighbours'
    included: the local run of the text), read along a turn by th about the line's centre: var / mean^2."""
    h = max(L["h"], 1.0)
    c = np.array([0.5 * (L["x0"] + L["x1"]), L["cy"]])
    # every turn resampled alike (bilinear at a fixed half-pixel offset): read untouched at 0 deg, the level frame was
    # always the sharpest, and a tube's lines 1.4 deg off read 0 (pitfall 113)
    t = np.radians(th)
    ys, xs = np.mgrid[int(np.floor(L["top"] - 1.5 * h)):int(np.ceil(L["bot"] + 1.5 * h)) + 1,
                      int(L["x0"]):int(L["x1"]) + 1].astype(float)
    xs, ys = xs + 0.5, ys + 0.5
    X = c[0] + np.cos(t) * (xs - c[0]) - np.sin(t) * (ys - c[1])
    Y = c[1] + np.sin(t) * (xs - c[0]) + np.cos(t) * (ys - c[1])
    p = ndi.map_coordinates(G, [Y, X], order=1, mode="constant").sum(1)
    p = ndi.gaussian_filter1d(p, 0.5)
    m = p.mean()
    return float(p.var() / (m * m)) if m > 1e-9 else 0.0


def line_turns(lr, R, G):
    """Each true line's turn onto the generated print (pitfall 112): the turn about the line's centre at which the
    generated print's rows there are sharpest (a turned line of text smears its row profile; a normalised match of
    the rows can't tell, it matched a level product's lines at 6-13 deg), coarse over LINE_TURNS then 0.25 deg round
    the best; a robust line of the turns in the lines' height down the label (the generated lines' fan: a leaning
    bottle's ran 12 deg at the top, 2 at the bottom); each line its own turn within TURN_SPREAD of that trend where
    its rows show it (TURN_CONF, TURN_LEN), else the trend. R: the true print blurred alike, read for its own lines' turn.
    -> ({k: deg}, {k: report})"""
    own, conf, tru = {}, {}, {}
    for L in lr:
        co = np.array([_turn_sharpness(L, G, th) for th in LINE_TURNS])
        t0 = LINE_TURNS[int(np.argmax(co))]
        fine = np.arange(t0 - 1.0, t0 + 1.01, 0.25)
        cf = np.array([_turn_sharpness(L, G, th) for th in fine])
        own[L["k"]] = float(fine[int(np.argmax(cf))])
        med = float(np.median(co))
        conf[L["k"]] = float(cf.max() / med) if med > 0 else 0.0
        # the packshot's own line, read the same way: a photo turned against its label's outline (1 deg on a tube's
        # packshot) has lines that aren't level in its frame, and a line laid level over the generated one leaned
        # by as much (the user: "the text below the grey line is still skewed a bit", pitfall 113)
        tt = np.arange(-3.0, 3.01, 0.1)
        ct = np.array([_turn_sharpness(L, R, th) for th in tt])
        tru[L["k"]] = float(tt[int(np.argmax(ct))])
    # the photo's turn is one for its label: the median over its long lines (a short word's rows are sharp at many)
    lt = [tru[L["k"]] for L in lr if (L["x1"] - L["x0"]) >= TURN_LEN * max(L["h"], 1.0)]
    t_photo = float(np.median(lt)) if lt else 0.0
    for L in lr:
        tru[L["k"]] = t_photo
        own[L["k"]] -= t_photo
    ks = [L["k"] for L in lr]
    ys = np.array([L["cy"] for L in lr])
    th = np.array([own[k] for k in ks])
    long_ = np.array([(L["x1"] - L["x0"]) >= TURN_LEN * max(L["h"], 1.0) for L in lr])
    w = np.array([np.log(max(conf[k], 1e-9) / TURN_CONF) for k in ks]).clip(0, None) * long_
    ok = w > 0
    if ok.sum() >= 3 and np.ptp(ys[ok]) > 0:
        X = np.c_[np.ones(len(ys)), ys - ys[ok].mean()]
        wt = w.copy()
        for _ in range(6):                                  # Huber weights: a line read at a wrong turn gets few
            beta = np.linalg.lstsq(X[ok] * np.sqrt(wt[ok])[:, None], th[ok] * np.sqrt(wt[ok]), rcond=None)[0]
            r = np.abs(th - X @ beta)
            sc = 1.4826 * np.median(r[ok]) + 0.5
            wt = w * np.minimum(1.0, 1.5 * sc / np.maximum(r, 1e-9))
        trend = X @ beta
    elif ok.any():
        trend = np.full(len(ys), float(np.average(th[ok], weights=w[ok])))
    else:
        trend = np.zeros(len(ys))
    out, rep = {}, {}
    for i, k in enumerate(ks):
        t = float(trend[i] + np.clip(th[i] - trend[i], -TURN_SPREAD, TURN_SPREAD)) if ok[i] else float(trend[i])
        out[k] = t
        rep[k] = {"own_deg": round(own[k], 2), "own_sharp": round(conf[k], 2), "trend_deg": round(float(trend[i]), 2),
                  "own_counts": bool(ok[i]), "packshot_deg": round(tru[k], 2)}
    return out, rep


def _robust_quad(u, e, it=4, order=2):
    """Quadratic (order 1: straight) e(u), refitted on the samples within 2.5 robust sigmas (+1 px) -> 3 coefficients
    (np.polyval)"""
    keep = np.ones(len(u), bool)
    c = np.zeros(3)
    for _ in range(it):
        if keep.sum() < 6:
            break
        c = np.polyfit(u[keep], e[keep], order if np.ptp(u[keep]) > 20 else 1)
        if len(c) == 2:
            c = np.r_[0.0, c]
        r = np.abs(e - np.polyval(c, u))
        keep = r <= 2.5 * 1.4826 * np.median(r[keep]) + 1.0
    return c


COURSE_LEN = 4.0       # a line's own course onto its generated row: lines this many line heights long or more ...
COURSE_COVER = 0.6     # ... with generated ink under this share of their columns
COURSE_WIN = 0.4       # rows read this share of a line height past the line's own, above and below
COURSE_TURN = 2.5      # deg: the course's turn at most this, its offset and bow at most COURSE_SHIFT line heights
COURSE_SHIFT = 0.35


def _course_of(L, R, G, A):
    """One placed line's course against its generated row (pitfall 115): column by column over the line's run, the
    centre row of the true letters (R) and of the generated print (G) where the placement A lays those rows; their
    difference as a robust quadratic in the line's own columns. -> (coefficients, report) or (None, report)"""
    h = max(L["h"], 1.0)
    us = np.arange(int(np.ceil(L["x0"])), int(np.floor(L["x1"])) + 1, dtype=float)
    vs = np.arange(L["top"] - COURSE_WIN * h, L["bot"] + COURSE_WIN * h + 1e-6, 0.5)
    U, V = np.meshgrid(us, vs)
    rt = ndi.map_coordinates(R, [V, U], order=1, mode="constant")
    q = np.c_[U.ravel(), V.ravel(), np.ones(U.size)] @ np.asarray(A, float).T
    gt = ndi.map_coordinates(G, [q[:, 1].reshape(U.shape), q[:, 0].reshape(U.shape)], order=1, mode="constant")
    mt, mg = rt.sum(0), gt.sum(0)
    ok = (mt > 0.15 * np.median(mt[mt > 0])) & (mg > 0.15 * np.median(mg[mg > 0])) if (mt > 0).any() and (mg > 0).any() \
        else np.zeros(len(us), bool)
    rep = {"cover": round(float(ok.mean()), 2)}
    if ok.mean() < COURSE_COVER or np.ptp(us) < COURSE_LEN * h:
        return None, rep
    ct = (rt * vs[:, None]).sum(0) / np.maximum(mt, 1e-9)
    cg = (gt * vs[:, None]).sum(0) / np.maximum(mg, 1e-9)
    c = _robust_quad(us[ok], (cg - ct)[ok])
    rep["fit_rms"] = round(float(np.sqrt(np.mean((np.polyval(c, us[ok]) - (cg - ct)[ok]) ** 2))), 2)
    return c, rep


def row_courses(lr, R, G, A, prep):
    """Each long line laid onto its own generated row's course (pitfall 115; the user, 2026-11-10: "generated rows are
    actually pretty lined up almost all the time ... they can be a good reference for mapping"): one turn per line
    left a block's top line arched over a straighter generated row, the last lines bent into the line above, and a
    wordmark off the band its generated row runs with. The course's slope at the line's centre turns the line (A, in
    place); its offset and curve bend it (rows per packshot column, BentLineMap). Turn, offset and bow capped
    (COURSE_TURN, COURSE_SHIFT). -> {k: bend coefficients (np.polyval, rows by packshot column)}"""
    bends = {}
    for L in lr:
        k = L["k"]
        if k not in A:
            continue
        if not prep.get(k, {}).get("on_text", True):
            # a line off the generated text (the generated block has fewer lines): its "row" is its neighbours'
            prep.setdefault(k, {})["course"] = {"off_text": True}
            continue
        c, r = _course_of(L, R, G, A[k])
        prep.setdefault(k, {})["course"] = r
        if c is None:
            continue
        h = max(L["h"], 1.0)
        uc = 0.5 * (L["x0"] + L["x1"])
        half = 0.5 * (L["x1"] - L["x0"])
        lim = COURSE_SHIFT * h
        c = np.array(c, float)
        c[0] = np.clip(c[0], -lim / max(half * half, 1.0), lim / max(half * half, 1.0))      # the bow
        d1 = float(np.clip(2 * c[0] * uc + c[1], -np.tan(np.radians(COURSE_TURN)), np.tan(np.radians(COURSE_TURN))))
        e0 = float(np.clip(np.polyval(c, uc), -lim, lim))
        turn = float(np.degrees(np.arctan(d1)))
        A[k] = np.asarray(A[k], float) @ _rot(np.array([uc, L["cy"]]), turn)
        # the bend: the course's offset and curve about the centre, its slope taken by the turn
        bends[k] = np.array([c[0], -2 * c[0] * uc, c[0] * uc * uc + e0], float)
        r.update(turn_deg=round(turn, 2), offset_px=round(e0, 2), bow_px=round(float(c[0] * half * half), 2))
        prep[k]["turn_deg"] = round(float(prep[k].get("turn_deg", 0.0)) + turn, 2)
    return bends


def place_blocks(lr, R, G):
    """Each block of true lines (a column: lines whose runs overlap) placed on the generated print G, both in the
    packshot's frame (R the true print blurred alike): each line turned as the generated line it lands on runs
    (line_turns), the block's size (BLOCK_SCALES round the fit's) and one shift along the row (column profiles), each
    line's own slide down the label (row profiles, _viterbi: in order, kept apart, sliding together unless the rows
    say otherwise). -> ({k: 3 x 3 reference px}, {k: report})"""
    A, rep = {}, {}
    turns, trep = line_turns(lr, R, G)
    for blk in _blocks(lr):
        lines = sorted([lr[k] for k in blk], key=lambda L: L["cy"])
        hs = np.array([L["h"] for L in lines])
        cys = np.array([L["cy"] for L in lines])
        pitch = float(np.median(np.diff(cys))) if len(lines) > 1 else 1.5 * float(hs.max())
        pitch = max(pitch, float(np.median(hs)))
        D = SLIDE * pitch
        slides = np.arange(-D, D + 1e-6, STEP)
        gaps = [lines[i + 1]["top"] - lines[i]["bot"] for i in range(len(lines) - 1)]
        reach = 1.5 * float(np.median(hs))
        c = np.array([0.5 * (min(L["x0"] for L in lines) + max(L["x1"] for L in lines)), float(np.mean(cys))])
        pad = D + 2 * reach + 2 * float(np.max(hs))
        x0b, x1b = min(L["x0"] for L in lines), max(L["x1"] for L in lines)
        y0b, y1b = min(L["top"] for L in lines), max(L["bot"] for L in lines)
        rad = 0.5 * np.hypot(x1b - x0b, y1b - y0b) + pad
        box = (int(max(c[0] - rad, 0)), int(max(c[1] - rad, 0)), int(min(c[0] + rad + 1, G.shape[1])),
               int(min(c[1] + rad + 1, G.shape[0])))
        th_b = float(np.median([turns[L["k"]] for L in lines]))
        # the generated print read along each line's own turn (a block's lines can fan), all about the block's centre:
        # turned about their own centres, a short line and a long one under it closed up at their left ends
        Gl = {L["k"]: (c, _turned(G, c, turns[L["k"]], box)) for L in lines}
        Gb = _turned(G, c, th_b, box)

        def trial(s_):
            dx, cx = _block_dx(lines, R, Gb, s_, reach)
            sc = np.array([_line_scores(L, R, Gl[L["k"]][1], s_, dx, slides) for L in lines])
            idx, tot = _viterbi(lines, sc, slides, gaps)
            # a size off the fit's needs the rows to say so
            tot -= 0.5 * len(lines) * abs(np.log(s_)) / 0.06
            return (tot, s_, dx, cx, idx, sc)
        tot, s_, dx, cx, idx, sc = max((trial(s_) for s_ in BLOCK_SCALES), key=lambda b: b[0])
        for i, L in enumerate(lines):
            dy = float(slides[idx[i]])
            on = float(sc[i, idx[i]])
            M = np.eye(3)
            M[0, 0] = M[1, 1] = s_
            M[0, 2] = L["x0"] + dx - s_ * L["x0"]
            M[1, 2] = L["cy"] + dy - s_ * L["cy"]
            A[L["k"]] = _rot(Gl[L["k"]][0], turns[L["k"]]) @ M
            rep[L["k"]] = dict({"scale": round(s_, 3), "dx": round(dx, 2), "dy": round(dy, 2),
                                "turn_deg": round(float(turns[L["k"]]), 2), "rows_ncc": round(on, 3),
                                "on_text": bool(on >= ON_TEXT), "block": blk[0], "block_cols_ncc": round(cx, 3)},
                               turn=trep[L["k"]])
    return A, rep


def _rot(c, deg):
    """3 x 3: a turn by deg about point c (reference px; + turns clockwise on the image, y down)."""
    t = np.radians(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = c - R @ c
    return M


def _turned(G, c, deg, box=None):
    """The generated print read through a turn by deg about c: Gt(p) = G(c + R (p - c)), so a true line laid level
    on Gt lies turned on G; only inside box [x0, y0, x1, y1] (0 elsewhere) when given."""
    if abs(deg) < 1e-9:
        return G
    t = np.radians(deg)
    x0, y0, x1, y1 = box if box is not None else (0, 0, G.shape[1], G.shape[0])
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(float)
    X = c[0] + np.cos(t) * (xs - c[0]) - np.sin(t) * (ys - c[1])
    Y = c[1] + np.sin(t) * (xs - c[0]) + np.cos(t) * (ys - c[1])
    out = np.zeros_like(G)
    out[y0:y1, x0:x1] = ndi.map_coordinates(G, [Y, X], order=1, mode="constant")
    return out
