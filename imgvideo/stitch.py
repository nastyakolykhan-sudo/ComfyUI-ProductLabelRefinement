"""Several refined keyframes of one product stitched into one unrolled label (R4): the label texture another shot's
job points at with anchor.texture_job, when no single keyframe saw all of the label a clip shows (another shot, a
spin, the back). Flat artwork is never assumed: the keyframes are the truth.

A stitch job's vjob.json:
  clip, mask      the clip and the product's matte (the keyframes' frames are read from them)
  stitch          {"base": a keyframe job (its folder or vjob.json: anchor.keyframe, model tube),
                   "keyframes": [{"image": a refined frame of the clip, the clip's size, "frame": its clip frame,
                                  "pitch_sag_px": "auto" (default) | px | 0}, ...],
                   "theta_lim": 75 (texture kept within this of facing the camera, per keyframe),
                   "agree_min": 0.6 (band-passed print correlation on the overlap with what is already stitched),
                   "min_overlap": 0.02 (share of the label's band; below it a keyframe can't be checked)}

The label grid is the base keyframe's, widened to the tube's whole circumference (u0 moves; the anchor.json written
here says where). Each further keyframe is found on the base's label by re-acquisition (spin, axial offset, scale:
its print is true, so the correlation is high), tilted to its own pitch (the bow of its print, V28), and read onto
the grid where it faces the camera within theta_lim and inside its matte. Each keyframe's own shading (block-median
luminance) is divided out, so no keyframe's light is stitched in: the render takes light from each frame anyway.
On the overlap with what is already stitched, its print must agree (band-passed correlation at least agree_min); a
keyframe that disagrees is reported and left out, never averaged in. The rest blend by viewing angle: cos theta,
falling to 0 at theta_lim.

Writes work/label_texture.png, work/label_valid.png and work/anchor.json (the base's, with the widened grid's tube
and label outline, and a "stitch" report), qa/stitch.json and qa/stitch_coverage.png (the texture over a strip that
says which keyframe each column comes from)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, median_filter

from imgpost import images as im

from . import surface


def _ramp(c, lim):
    cl = np.cos(np.radians(lim))
    return np.clip((c - cl) / (1.0 - cl), 0.0, 1.0)


def _shading(L, w, block=24):
    """The texture's slow light: block medians of log luminance over the bare label (w > 0, print strokes left out:
    a bold wordmark fills most of its blocks and would read as shadow, and dividing it out would pale the print),
    filled and smoothed."""
    from scipy.ndimage import binary_dilation
    h, wd = L.shape
    nb_y, nb_x = max(1, h // block), max(1, wd // block)
    g = np.full((nb_y, nb_x), np.nan)
    lg = np.log(np.maximum(L, 1e-4))
    bp = np.abs(gaussian_filter(lg, 1.0) - gaussian_filter(lg, 6.0))
    mad = float(np.median(bp[w > 0.05])) if (w > 0.05).any() else 0.0
    bare = (w > 0.05) & ~binary_dilation(bp > 3.0 * max(mad, 1e-3), iterations=3)
    for i in range(nb_y):
        for j in range(nb_x):
            m = bare[i * block:(i + 1) * block, j * block:(j + 1) * block]
            if m.sum() > block * block * 0.15:
                g[i, j] = np.median(lg[i * block:(i + 1) * block, j * block:(j + 1) * block][m])
    ok = np.isfinite(g)
    if not ok.any():
        return np.ones_like(L)
    g = np.where(ok, g, np.nanmedian(g))
    g = median_filter(g, size=3, mode="nearest")
    up = np.kron(g, np.ones((block, block)))
    up = np.pad(up, ((0, max(0, h - up.shape[0])), (0, max(0, wd - up.shape[1]))), mode="edge")[:h, :wd]
    return np.exp(gaussian_filter(up, block / 2.0))


def _bandpass(L):
    return gaussian_filter(L, 1.5) - gaussian_filter(L, 8.0)


def _ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def stitch(path):
    from .pipeline import FileMedia, auto_pitch, pitched
    path = Path(path).expanduser().resolve()
    jd, v = path.parent, json.loads(path.read_text())
    (jd / "work").mkdir(exist_ok=True)
    (jd / "qa").mkdir(exist_ok=True)
    sc = v["stitch"]
    media = FileMedia(im.resolve(v["clip"], jd), im.resolve(v["mask"], jd))
    bd = im.resolve(sc["base"], jd)
    bd = bd.parent if bd.suffix == ".json" else bd
    km = json.loads((bd / "work" / "anchor.json").read_text())
    if km["spec"].get("model") != "tube" or "tube" not in km:
        raise SystemExit("stitch: the base must be a keyframe job unrolled onto a tube (anchor.model tube)")
    tex_b = im.load_u8(bd / "work" / "label_texture.png")[0]
    valid_b = np.asarray(Image.open(bd / "work" / "label_valid.png")) > 127
    lim = float(sc.get("theta_lim", km["spec"].get("theta_lim", 75.0)))
    agree_min, min_overlap = float(sc.get("agree_min", 0.6)), float(sc.get("min_overlap", 0.02))
    tb = km["tube"]
    Ru = float(tb["Ru"])
    Hh, Wb = tex_b.shape[:2]
    W = int(np.ceil(2 * np.pi * Ru)) + 2
    shift = int(round(W / 2.0 - tb["u0"]))
    u0n = tb["u0"] + shift
    P_b = np.asarray(km["label_polygon"], float)
    vlo, vhi = float(P_b[:, 1].min()), float(P_b[:, 1].max())
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(Hh, dtype=float))
    band = (vv >= vlo) & (vv <= vhi)

    def flat(T, w, S=None):
        if S is None:
            S = _shading(im.lum(im.srgb_to_lin(T)), w)
        return T / np.maximum(S, 1e-4)[..., None]

    # the base, on the widened grid
    T_b = np.zeros((Hh, W, 3))
    w_b = np.zeros((Hh, W))
    c0, c1 = max(0, shift), min(W, shift + Wb)
    T_b[:, c0:c1] = im.to_float(tex_b)[:, c0 - shift:c1 - shift]
    th_b = (uu - u0n) / Ru
    w_b[:, c0:c1] = valid_b[:, c0 - shift:c1 - shift]
    w_b *= _ramp(np.cos(th_b), lim) * band
    S_b = _shading(im.lum(im.srgb_to_lin(T_b)), w_b)       # the base's light, filled over the whole grid
    layers = [{"name": "base", "frame": int(km["frame"]), "T": flat(T_b, w_b, S_b), "w": w_b, "status": "base"}]
    kb = int(km["frame"])
    mask_b_area = int(media.masks(kb, kb + 1)[0].sum())
    report = {"base": str(bd), "base_frame": kb, "grid": {"width": W, "u0": round(u0n, 2), "shift": shift}, "keyframes": []}

    for i, k in enumerate(sc.get("keyframes", [])):
        t = int(k["frame"])
        img = im.load_u8(im.resolve(k["image"], jd))[0]
        mf = media.masks(t, t + 1)[0]
        if img.shape[:2] != mf.shape:
            raise SystemExit(f"stitch: keyframe {k['image']} isn't the clip's size")
        H, rep = surface.reacquire(tex_b, valid_b, tb, mask_b_area, img, mf, km["spec"]["band"], lim)
        entry = {"image": str(k["image"]), "frame": t, "reacquire": {kk: rep[kk] for kk in ("scale", "spin_deg", "axial_offset_px",
                                                                                           "verify_ncc") if kk in rep}}
        if rep.get("verify_ncc", 0.0) < float(sc.get("found_min", 0.4)):
            # a true print correlates far better than this once found: the tube fit or the search went wrong (an
            # occluder cutting the matte is the usual cause). Don't guess: report it, leave it out.
            entry.update(status=f"left out: not found on the base's label (print correlation {rep.get('verify_ncc')}; "
                                f"an occluder cutting the matte, or a different product pose)", agreement=None)
            report["keyframes"].append(entry)
            print(f"stitch     keyframe {i + 1} (frame {t}): {entry['status']}")
            continue
        ps = k.get("pitch_sag_px", "auto")
        if ps == "auto":
            sag, prep = auto_pitch(H, tex_b, km["label_polygon"], img, mf)
            entry["pitch"] = prep
        else:
            sag = float(ps or 0.0)
        if sag:
            H = pitched(H, sag, km["label_polygon"], quiet=True)
        acc_w = sum(l["w"] for l in layers if l["status"] in ("base", "stitched"))
        acc_T = sum(l["T"] * l["w"][..., None] for l in layers if l["status"] in ("base", "stitched")) / np.maximum(acc_w, 1e-6)[..., None]
        A_bp = _bandpass(im.lum(im.srgb_to_lin(acc_T)))

        def unroll_k(du, dv):
            """the keyframe read onto the widened grid (base label px = grid u - shift), moved by (du, dv) label px"""
            Pq = H.forward(np.c_[(uu - shift - du).ravel(), (vv - dv).ravel()])
            T_ = np.stack([map_coordinates(im.to_float(img)[..., c], [Pq[:, 1], Pq[:, 0]], order=1, mode="constant")
                           for c in range(3)], -1).reshape(Hh, W, 3)
            ins = map_coordinates(mf.astype(float), [Pq[:, 1], Pq[:, 0]], order=1, mode="constant").reshape(Hh, W) > 0.5
            d = np.cross(H.A[0], H.A[1])
            th_k = np.angle(np.exp(1j * ((uu - shift - du - H.u0) / H.Ru - np.arctan2(d[0], d[2]))))
            w_ = _ramp(np.cos(th_k), lim) * ins * band
            return T_, w_
        T_k, w_k = unroll_k(0.0, 0.0)
        T_kf = flat(T_k, w_k)
        # re-acquisition leaves 1-3 label px (it searches at a quarter, then half resolution): refine the shift on the
        # overlap before judging agreement, or two true prints a stroke apart read as disagreeing
        O = (acc_w > 0.3) & (w_k > 0.3)
        best = (0.0, 0.0)
        if O.sum() > 500:
            B_bp = _bandpass(im.lum(im.srgb_to_lin(T_kf)))
            sc_ = {}
            for dv in range(-12, 13, 2):
                for du in range(-12, 13, 2):
                    Bs = np.roll(np.roll(B_bp, dv, 0), du, 1)
                    Os = O & np.roll(np.roll(O, dv, 0), du, 1)
                    if Os.sum() > 500:
                        sc_[(du, dv)] = _ncc(A_bp[Os], Bs[Os])
            if sc_:
                best = max(sc_, key=sc_.get)
                for dv in range(best[1] - 1, best[1] + 2):          # then to the pixel round the coarse best
                    for du in range(best[0] - 1, best[0] + 2):
                        if (du, dv) in sc_:
                            continue
                        Bs = np.roll(np.roll(B_bp, dv, 0), du, 1)
                        Os = O & np.roll(np.roll(O, dv, 0), du, 1)
                        if Os.sum() > 500:
                            sc_[(du, dv)] = _ncc(A_bp[Os], Bs[Os])
                best = max(sc_, key=sc_.get)
            if best != (0, 0):
                T_k, w_k = unroll_k(float(best[0]), float(best[1]))
                T_kf = flat(T_k, w_k)
        entry["refined_shift_px"] = [int(best[0]), int(best[1])]
        O = (acc_w > 0.3) & (w_k > 0.3)
        share = float(O.sum() / max(band.sum(), 1))
        entry["overlap_share"] = round(share, 4)
        entry["new_coverage_share"] = round(float(((w_k > 0.3) & (acc_w <= 0.05)).sum() / max(band.sum(), 1)), 4)
        if share < min_overlap:
            entry.update(status="stitched (unchecked: no overlap)", agreement=None)
        else:
            ncc = _ncc(A_bp[O], _bandpass(im.lum(im.srgb_to_lin(T_kf)))[O])
            entry["agreement"] = round(ncc, 3)
            entry["status"] = "stitched" if ncc >= agree_min else "left out: its print disagrees on the overlap"
        layers.append({"name": f"keyframe {i + 1}", "frame": t, "T": T_kf, "w": w_k,
                       "status": "stitched" if entry["status"].startswith("stitched") else "left out"})
        report["keyframes"].append(entry)
        print(f"stitch     keyframe {i + 1} (frame {t}): print correlation {rep.get('verify_ncc')}, spin {rep.get('spin_deg')} deg"
              + (f", pitch {entry['pitch']['sag_px']:+.1f} px" if "pitch" in entry else "")
              + f"; overlap {100 * share:.1f} % of the band, agreement {entry['agreement']}, new coverage "
              f"{100 * entry['new_coverage_share']:.1f} %: {entry['status']}")

    use = [l for l in layers if l["status"] in ("base", "stitched")]
    # each label pixel comes from one keyframe: the one that sees it best (viewing-angle weight times its print's
    # detail energy against the base's: a keyframe softened by its shot's blur or motion shouldn't win over a sharp
    # one), feathered only where the winner changes. Averaging two prints a few px apart would ghost them, and an
    # agreeing keyframe still isn't pixel-exact (re-acquisition, pitch)
    def detail(l):
        m = l["w"] > 0.3
        return float(np.mean(np.abs(_bandpass(im.lum(im.srgb_to_lin(l["T"])))[m]))) if m.any() else 0.0
    d0 = max(detail(use[0]), 1e-6)
    for l in use:
        l["sharp"] = float(np.clip(detail(l) / d0, 0.05, 2.0))
    score = np.stack([l["w"] * l["sharp"] for l in use])
    win = np.argmax(score, 0)
    Wsum = sum(l["w"] for l in use)
    soft = []
    for i, l in enumerate(use):
        soft.append(gaussian_filter((win == i).astype(float), 4.0) * (l["w"] > 0.02))
    Ssum = sum(soft)
    T = sum(l["T"] * m[..., None] for l, m in zip(use, soft)) / np.maximum(Ssum, 1e-9)[..., None]
    report["from_each_keyframe"] = {l["name"]: round(float(((win == i) & (Wsum > 0.02) & band).sum() / max(band.sum(), 1)), 4)
                                    for i, l in enumerate(use)}
    report["sharpness"] = {l["name"]: round(l["sharp"], 3) for l in use}
    # the base's own light back on: where the base wins the texture is the base keyframe's exactly, and every other
    # keyframe's print sits under the same light
    T = T * S_b[..., None]
    valid = Wsum > 0.02
    T = np.where(valid[..., None], T, 0.0)
    cols = valid[band.any(1)].mean(0) >= 0.5 if band.any() else valid.mean(0) >= 0.5
    # the covered run of columns containing the base's middle
    c = int(round(u0n))
    lo = c
    while lo > 0 and cols[lo - 1]:
        lo -= 1
    hi = c
    while hi < W - 1 and cols[hi + 1]:
        hi += 1
    poly = [[float(lo), vlo], [float(hi), vlo], [float(hi), vhi], [float(lo), vhi]]
    out_km = dict(km, tube=dict(tb, u0=u0n), label_polygon=poly, stitch=report)
    Image.fromarray(np.clip(T * 255 + 0.5, 0, 255).astype(np.uint8)).save(jd / "work" / "label_texture.png")
    Image.fromarray(valid.astype(np.uint8) * 255).save(jd / "work" / "label_valid.png")
    (jd / "work" / "anchor.json").write_text(json.dumps(out_km))
    report.update(label_outline_u=[lo, hi], coverage_deg=round(float((hi - lo) / Ru * 180 / np.pi), 1),
                  base_coverage_deg=round(float((P_b[:, 0].max() - P_b[:, 0].min()) / Ru * 180 / np.pi), 1))
    (jd / "qa" / "stitch.json").write_text(json.dumps(report, indent=1))
    # coverage sheet: the texture over a strip coloured by the keyframe each column mostly comes from
    pal = np.array([[60, 120, 220], [235, 105, 50], [30, 175, 120], [235, 160, 0], [170, 80, 200], [120, 120, 120]], np.uint8)
    dom = np.argmax(np.stack([l["w"][band.any(1)].mean(0) for l in use]), 0)
    strip = np.repeat(pal[dom % len(pal)][None], 24, 0)
    strip[:, ~cols] = 255
    sheet = np.concatenate([np.clip(T * 255, 0, 255).astype(np.uint8), np.full((6, W, 3), 255, np.uint8), strip], 0)
    Image.fromarray(sheet).save(jd / "qa" / "stitch_coverage.png")
    print(f"stitch     {len(use) - 1} of {len(sc.get('keyframes', []))} keyframes stitched onto the base (frame {kb}); the label now spans "
          f"{report['coverage_deg']} deg of the tube (the base alone: {report['base_coverage_deg']} deg)")
    print(f"saved      {jd / 'work' / 'label_texture.png'}\n           {jd / 'qa' / 'stitch.json'}\n           {jd / 'qa' / 'stitch_coverage.png'}")
    return report
