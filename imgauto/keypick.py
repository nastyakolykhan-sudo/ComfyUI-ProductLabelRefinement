"""Which frame of a shot to correct: the keyframe the generative edit or the stills job works on, and the anchor the
engine tracks from in both directions.

Each term is normalised within the shot (1 = the shot's best), so the weights trade like for like:
- size: the product's width x length, the label's area on screen (the edit and OCR need pixels);
- sharp: fine detail inside the matte (a soft or smeared anchor softens the whole shot, V17);
- print: print-stroke energy in the middle of the product (the label is legible, not turned away or blurred);
- frontal: the print centred on the product's axis (the label faces the camera; past ~75 deg it is unusable, V6);
- stable: not a morph frame. A morph frame disagrees with both neighbours while they still agree with each other
  (t-1 with t+1 beats t with either); a product that spins or moves fast lowers all three alike and isn't penalised.
Occlusion (holes in the matte, or a matte smaller than its neighbours'), a sparkle burst (glints well above the
shot's median) and the trim's first and last frames are penalised rather than excluded, so a shot always gets a pick and the report says why it is weak.

A bonus (x up to 1 + ANGLE_BONUS) goes to frames that show the product at a reference photo's viewing angle
(refangle.py): there the stills engine can make the keyframe from the real photo with exact text, and the
generative edit is only the fallback. Only the shot's best candidates are checked (it costs a warp per frame).
"""
import numpy as np

WEIGHTS = {"size": 0.30, "sharp": 0.25, "print": 0.15, "frontal": 0.15, "stable": 0.15}
PENALTY = {"occluded": 0.6, "edge": 0.8, "unstable": 0.5, "sparkles": 0.7}
ANGLE_BONUS = 0.25
STILLS_ANGLE = 0.6           # angle match at or above: the stills engine can make this keyframe from the photo


def _norm(x):
    x = np.asarray(x, float)
    top = np.nanpercentile(x, 95) if np.isfinite(x).any() else 1.0
    return np.clip(np.nan_to_num(x / max(top, 1e-9)), 0.0, 1.0)


def score(stats, bounds, frames, open_ends=()):
    """Scores for the given frames (a trim): -> list of {t, score, terms, flags} sorted best first. open_ends: trim
    ends that aren't transitions (the clip's first or last frame), which aren't penalised as edges."""
    frames = list(frames)
    S = [stats[t] for t in frames]
    p1 = {b["t"]: b["p1"] for b in bounds}
    skip = {b["t"] + 1: b["skip"] for b in bounds}           # frame u: u-1 with u+1
    morph = [np.nan_to_num(skip.get(t, np.nan) - np.nanmax([p1.get(t - 1, np.nan), p1.get(t, np.nan)]))
             if np.isfinite(skip.get(t, np.nan)) else 0.0 for t in frames]
    stab = 1.0 - np.clip(np.array(morph) * 2.5, 0.0, 1.0)
    terms = {"size": _norm([s["width_px"] * s["length_px"] for s in S]),
             "sharp": _norm([s["sharp"] for s in S]),
             "print": _norm([s["print"] for s in S]),
             "frontal": np.clip(1.0 - np.array([s["print_offset"] for s in S]), 0.0, 1.0),
             "stable": stab}
    areas = np.array([s["area"] for s in S])
    glints = np.array([s.get("glints", 0) for s in S], float)
    g_med = float(np.median(glints)) if len(glints) else 0.0
    out = []
    for i, t in enumerate(frames):
        lo, hi = max(0, i - 3), min(len(frames), i + 4)
        nb = np.median(np.r_[areas[lo:i], areas[i + 1:hi]]) if hi - lo > 1 else areas[i]
        occl = max(S[i]["holes"], 1.0 - areas[i] / max(nb, 1e-9))
        sc = float(np.prod([max(terms[k][i], 1e-3) ** w for k, w in WEIGHTS.items()]))
        flags = []
        if occl > 0.03:
            flags.append("occluded")
        if i in (0, len(frames) - 1) and t not in open_ends:
            flags.append("edge")
        if stab[i] < 0.5:
            flags.append("unstable")
        if glints[i] > max(10.0, 3.0 * g_med):          # a sparkle burst across the product on this frame
            flags.append("sparkles")
        for f in flags:
            sc *= PENALTY[f]
        out.append({"t": t, "score": round(sc, 3), "terms": {k: round(float(v[i]), 2) for k, v in terms.items()},
                    "flags": flags})
    return sorted(out, key=lambda r: -r["score"])


def pick(stats, bounds, trims, n_frames=None, angle=None, top=12):
    """The shot's keyframe over all its trims: -> (best, ranking). The margin to the runner-up is the
    self-check: under 5 % the contact sheet shows both. angle(frames) -> {t: refangle score}: applied to the
    `top` best frames, which are then re-ranked."""
    frames = [t for a, b in trims for t in range(a, b)]
    if not frames:
        return None, []
    ranking = score(stats, bounds, frames, open_ends=(0, (n_frames or len(stats)) - 1))
    if angle is not None:
        cand = ranking[:top]
        ang = angle([r["t"] for r in cand])
        for r in cand:
            a = ang.get(r["t"], {})
            r["angle"] = a
            r["base_score"] = r["score"]
            r["score"] = round(r["score"] * (1 + ANGLE_BONUS * a.get("angle", 0.0)), 3)
        ranking = sorted(cand, key=lambda r: -r["score"]) + ranking[top:]
    best = dict(ranking[0])
    best["margin"] = round(1.0 - ranking[1]["score"] / best["score"], 3) if len(ranking) > 1 and best["score"] else 1.0
    return best, ranking


def rank_of(ranking, t):
    """Where frame t sits in a ranking: -> (1-based rank, its entry) or (None, None)."""
    for i, r in enumerate(ranking):
        if r["t"] == t:
            return i + 1, r
    return None, None


def suggest(stats, bounds, runs, n_frames=None, angle=None):
    """A suggested keyframe for each run of frames, e.g. the runs a first keyframe doesn't cover.
    runs: [[a, b), ...] frame ranges. -> [{"run", "pick", "score", "margin", "flags", "angle"}] (None picks skipped)"""
    out = []
    for a, b in runs:
        best, _ = pick(stats, bounds, [[a, b]], n_frames, angle=angle)
        if best:
            out.append({"run": [a, b], "pick": best["t"], "score": best["score"], "margin": best["margin"],
                        "flags": best["flags"], "angle": best.get("angle", {})})
    return out
