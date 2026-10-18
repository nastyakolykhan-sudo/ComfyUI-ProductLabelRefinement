"""After a shot job has run: which of its fixable frames the engine's held run didn't cover, and what to do with each
uncovered run.

The engine has the last word on where tracks hold (qa/fit.json `valid`). A trim's frames outside the held run, and
frames where the label turns into view past what the keyframe saw (qa/qa.json `unseen_frac` above `qa.unseen_max`),
are uncovered. Each uncovered run gets a plan:
- "keyframe": the generator redrew the product at the run's edge (the silhouette, aligned on centroid, axis and
  length, overlaps the last held frame's under 0.9: a slanted crimp, the print lower), or the print turned
  away; tracks and print correlation won't register it, so it needs its own refined keyframe (V27);
- "chain": the product carries on across the edge (no cut, no redraw): the next job chains from the last held frame
  (anchor.chain, V15);
- "reacquire": after a cut, the keyframe's texture found again on the run's best frame.
Every plan carries the suggested keyframe frame (keypick.suggest), so a run that fails its chain or re-acquisition
goes straight to step 4. A stop reason from the engine (request R2) will replace the edge tests once it exists.
"""
import json
from pathlib import Path

import numpy as np

from . import keypick, refangle
from .shots import ranges


def runs_of(ts):
    """Sorted frames -> [[a, b), ...] runs."""
    out = []
    for t in sorted(ts):
        if out and t == out[-1][1]:
            out[-1][1] = t + 1
        else:
            out.append([t, t + 1])
    return out


STOP_PLAN = {"cut": "reacquire", "morph": "keyframe"}      # imgvideo 0.4.1 (R2); every other reason: chain


def job_stops(job_dir, offset):
    """The engine's reasons a run job's held run ended (qa/fit.json "stops", imgvideo 0.4.1, R2), frames in master
    frames: {"start"|"end": {"frame", "reason", "failed", "values"}}; {} for an older job."""
    fj = Path(job_dir) / "qa" / "fit.json"
    if not fj.exists():
        return {}
    return {k: dict(v, frame=None if v.get("frame") is None else offset + v["frame"])
            for k, v in (json.loads(fj.read_text()).get("stops") or {}).items()}


def job_coverage(job_dir, offset):
    """Held frames and unseen frames of a run job, in master frames. -> (held set, unseen set) or None"""
    jd = Path(job_dir)
    fj, qj = jd / "qa" / "fit.json", jd / "qa" / "qa.json"
    if not fj.exists():
        return None
    held = {offset + t for t in json.loads(fj.read_text()).get("valid", [])}
    unseen = set()
    if qj.exists():
        q = json.loads(qj.read_text())
        lim = q.get("thresholds", {}).get("unseen_max", 0.02)
        unseen = {offset + r["frame"] for r in q.get("frames", []) if (r.get("unseen_frac") or 0) > lim}
    return held, unseen


def redrawn(M, a, b, iou_min=0.9):
    """The product's silhouette changes shape between frames a and b (mattes at any one scale), beyond motion."""
    if not M[a].any() or not M[b].any():
        return False
    return refangle.shape_iou(M[b], M[a]) < iou_min


def plan(shot, kinds, held, unseen, stats, bounds, M, n_frames, angle=None, stops=None):
    """The uncovered runs of one shot and a plan for each. stops: the engine's reasons the held run ended (R2); where
    one applies to a run's edge it decides (cut: re-acquire, morph: a keyframe of the new pose, else chain), and the
    measured tests below are the fallback. -> {"held", "uncovered": [{run, plan, why, edge, suggest}]}"""
    want = {t for a, b in shot["trims"] for t in range(a, b)}
    gaps = (want - held) | (want & unseen)
    out = []
    for a, b in runs_of(gaps):
        edge = a - 1 if (a - 1) in held else (b if b in held else None)
        stop = None
        if stops and edge is not None:
            stop = stops.get("end") if edge < a else stops.get("start")
        if stop and stop.get("reason") and stop["reason"] != "shot edge":
            p = STOP_PLAN.get(stop["reason"], "chain")
            why = f"the engine's held run stopped there: {stop['reason']}"
            if set(range(a, b)) <= unseen and p == "chain":
                p, why = "keyframe", why + ", and the label turns into view past what the keyframe saw"
        elif edge is None:
            p, why = "keyframe", "no held frame next to it"
        elif edge < a and kinds[a - 1] == "cut" or edge >= b and kinds[b - 1] == "cut":
            p, why = "reacquire", "a cut at its edge"
        elif redrawn(M, edge, a if edge < a else b - 1):
            p, why = "keyframe", "the product is redrawn across the edge (silhouette changes shape)"
        elif set(range(a, b)) <= unseen:
            p, why = "keyframe", "the label turns into view past what the keyframe saw"
        else:
            p, why = "chain", f"carries on from held frame {edge}"
        sug = keypick.suggest(stats, bounds, [[a, b]], n_frames, angle=angle)
        out.append({"run": [a, b], "plan": p, "why": why, "edge": edge, "suggest": sug[0] if sug else None})
    return {"held": ranges(sorted(held & set(range(*shot["shot"])))), "uncovered": out}
