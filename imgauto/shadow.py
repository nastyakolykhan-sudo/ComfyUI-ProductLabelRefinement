"""Shadow mode: the autopilot's decisions next to the ones made by hand on the same project, read from the hand jobs
(9-build/*/vjob.json, their work/anchor.json and qa/fit.json) and from the last delivered lossless round."""
import json
import re
from pathlib import Path

import numpy as np

from . import media
from .shots import ranges


def hand_jobs(project):
    """Every hand-run shot job: {name, shot, kind, frame, held, impacts, patch}."""
    out = []
    for vj in sorted((project / "9-build").glob("*/vjob.json")):
        if vj.parent.name in ("auto", "reviews"):
            continue
        v = json.loads(vj.read_text())
        a = v.get("anchor", {})
        if "keyframe" in a:
            kind, frame = "keyframe", None
            aj = vj.parent / "work" / "anchor.json"
            if aj.exists():
                frame = json.loads(aj.read_text()).get("frame")
        elif "chain" in a:
            kind, frame = "chain", a["chain"]["frame"]
        elif "reacquire" in a:
            kind, frame = "reacquire", a["reacquire"]
        else:
            kind, frame = "artwork job", a.get("frame")
        fj = vj.parent / "qa" / "fit.json"
        held, impacts = None, []
        if fj.exists():
            f = json.loads(fj.read_text())
            if isinstance(f.get("valid"), list) and f["valid"]:
                held = [min(f["valid"]), max(f["valid"]) + 1]
            impacts = f.get("impacts", [])
        out.append({"name": vj.parent.name, "shot": v.get("shot"), "kind": kind, "frame": frame, "held": held,
                    "impacts": impacts, "patch": v.get("render", {}).get("frames"), "band": a.get("band"),
                    "render": {k: v.get("render", {}).get(k) for k in ("polarity", "print_tophat", "spark_glow",
                                                                       "sparkles", "occluder_feather")}})
    return latest(out)


def _version(name):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m and not name.endswith(("example1", "example2")) else 1


def latest(jobs):
    """Superseded rounds out: of hand jobs on overlapping shots (patch jobs aside), the highest version suffix wins
    (shotC5 over shotC4, shotB3 over shotB)."""
    keep = []
    for j in jobs:
        if j["patch"]:
            keep.append(j)
            continue
        rivals = [k for k in jobs if not k["patch"] and k is not j and k["shot"] and j["shot"]
                  and k["shot"][0] < j["shot"][1] and j["shot"][0] < k["shot"][1] and k["kind"] != "keyframe"
                  and j["kind"] != "keyframe"]
        if all(_version(j["name"]) >= _version(k["name"]) for k in rivals):
            keep.append(j)
    return keep


def delivered(project, source, info, chunk=24):
    """The latest review round's lossless render against the source, decoded exactly as the engine does (full res,
    the source's matrix): -> (round, frames the fix changed)"""
    from imgvideo import vio
    rv = sorted((project / "5-reviews").glob("v*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1)
    for d in reversed(rv):
        movs = sorted(d.glob("*_refined_lossless.mov"))
        if movs:
            ch = []
            for t0 in range(0, info["frames"], chunk):
                t1 = min(t0 + chunk, info["frames"])
                R, S = vio.read(movs[0], t0, t1), vio.read(source, t0, t1, info)
                ch += [t0 + i for i in range(min(len(R), len(S))) if (R[i] != S[i]).any()]
            return d.name, ch
    return None, []


def _frames(rs):
    return {t for a, b in rs for t in range(a, b)}


def compare(auto, hand, delivered_round, delivered_frames, n_frames):
    """Rows of the comparison table: [topic, autopilot, by hand, verdict]."""
    rows = []
    a_cuts = auto["cuts"]
    starts = sorted({j["shot"][0] for j in hand if j["kind"] in ("keyframe", "reacquire") and j["shot"][0] > 0
                     and not j["patch"]})
    chains = sorted({j["frame"] for j in hand if j["kind"] == "chain"})
    hand_cuts = [s for s in starts if s not in chains]
    ends = {j["shot"][1] for j in hand if j["shot"] and not j["patch"]} - set(starts) - {n_frames}
    agree = sorted(set(a_cuts) & set(hand_cuts))
    at_end = sorted((set(a_cuts) - set(hand_cuts)) & ends)
    rows.append(["Cuts (new anchor needed)", ", ".join(map(str, a_cuts)) or "none",
                 ", ".join(map(str, hand_cuts)) or "none",
                 "same" if set(a_cuts) == set(hand_cuts) else
                 f"agree on {agree or 'none'}; auto only {sorted(set(a_cuts) - set(hand_cuts) - set(at_end)) or '-'}"
                 + (f" (plus {at_end}, where the hand jobs end)" if at_end else "")
                 + f"; hand only {sorted(set(hand_cuts) - set(a_cuts)) or '-'}"])
    rows.append(["Continuous boundaries chained by hand", "-", ", ".join(f"{c}|{c + 1}" for c in chains) or "none",
                 "; ".join(f"{c}|{c + 1}: auto says {auto['kinds'][c]}" for c in chains if c < len(auto["kinds"]))
                 or "-"])
    af = _frames(auto["trims"])
    df = set(delivered_frames)
    rows.append([f"Frames to fix (auto trims) vs refined in {delivered_round}", ranges(sorted(af)) or "none",
                 ranges(sorted(df)) or "none",
                 f"both {len(af & df)}; auto only {ranges(sorted(af - df)) or '-'}; "
                 f"hand only {ranges(sorted(df - af)) or '-'}"])
    rows.append(["Frames with no matte", auto["no_matte"] or "none", auto.get("hand_no_matte", "-"), "-"])
    key = [j for j in hand if j["kind"] == "keyframe" and j["frame"] is not None]
    if key and auto.get("keyframe_rank"):
        kr = auto["keyframe_rank"]
        rows.append(["Keyframe (the corrected frame)", f"{kr['auto']} (score {kr['auto_score']})",
                     f"{key[0]['frame']} (score {kr['user_score']}, rank {kr['user_rank']}/{kr['of']})",
                     "same" if kr["auto"] == key[0]["frame"] else
                     f"{abs(kr['auto'] - key[0]['frame'])} frames apart; terms auto {kr['auto_terms']}, "
                     f"user {kr['user_terms']}"])
    for j in hand:
        if j["kind"] == "reacquire" and not j["patch"]:
            p = next((s for s in auto["picks"] if s["shot"][0] <= j["frame"] < s["shot"][1]), None)
            rows.append([f"Re-acquisition frame, hand job {j['name']} {j['shot']}",
                         f"{p['pick']} (shot {p['shot']})" if p else "no auto shot holds it", str(j["frame"]),
                         "same" if p and p["pick"] == j["frame"] else
                         (f"{abs(p['pick'] - j['frame'])} frames apart" if p else "-")])
    if key and key[0]["band"] and auto.get("band"):
        hb, ab = key[0]["band"], auto["band"]
        rows.append(["Label band on the keyframe (px along the axis)", str(ab), str(hb),
                     f"ends {ab[0] - hb[0]:+d} / {ab[1] - hb[1]:+d} px"])
    if key:
        imp = sorted(set(key[0]["impacts"]))
        span = set(range(*(key[0]["held"] or key[0]["shot"])))
        am = sorted(set(auto["morph_snap"]) & span)
        un = sorted(set(auto.get("uncertain", [])) & span)
        ev, cur = [], []
        for t in imp:                                # the engine marks both sides of a jump: group into events
            if cur and t - cur[-1] > 1:
                ev.append(cur)
                cur = []
            cur.append(t)
        if cur:
            ev.append(cur)
        hit = lambda e, fs: any(u in e or u - 1 in e for u in fs)
        found = sum(hit(e, am) for e in ev)
        found_u = sum(hit(e, am + un) for e in ev)
        extra = [u for u in am if not any(u in e or u - 1 in e for e in ev)]
        rows.append([f"Morph / snap boundaries in the keyframe shot vs the engine's impacts",
                     f"{ranges(am) or 'none'} (uncertain: {ranges(un) or 'none'})", ranges(imp) or "none",
                     f"{found} of {len(ev)} jump events (with uncertain: {found_u}); not in the engine's list: "
                     f"{ranges(extra) or 'none'}"])
        hr = key[0]["render"]
        ar = auto.get("render", {})
        for j in hand:                               # a patch job's settings belong to the shot it patches
            if j["patch"] and j["shot"] == key[0]["shot"]:
                hr = {k: (v if v is not None else hr.get(k)) for k, v in j["render"].items()}
        for f in ("polarity", "print_tophat", "spark_glow", "occluder_feather"):
            if ar.get(f) is not None or hr.get(f) is not None:
                rows.append([f"render.{f} (keyframe shot)", str(ar.get(f)), str(hr.get(f)),
                             "same" if ar.get(f) == hr.get(f) else "differs"])
    for k in auto["picks"]:
        if k.get("angle"):
            rows.append([f"Reference-angle match, shot {k['shot']}",
                         f"pick {k['pick']}: angle {k['angle'].get('angle')} (shape {k['angle'].get('shape')}, "
                         f"layout {k['angle'].get('layout')}) → {k['route']}", "-", "-"])
    if auto.get("verify"):
        v = auto["verify"]
        rows.append(["Keyframe verification (OCR of the unrolled label)", f"corrected: {v['corrected']}",
                     f"source frame: {v['source']}", "separates" if v["corrected"].startswith("pass")
                     and not v["source"].startswith("pass") else "check"])
    return rows


def table(rows):
    out = ["| Decision | Autopilot | By hand | Verdict |", "|---|---|---|---|"]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |")
    return "\n".join(out)
