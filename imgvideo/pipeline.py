"""The video runner's job and stages (track, fit, still, render, qa, merge, review), on files or in memory.

scripts/video.py is the command line over these; a port (the ComfyUI pack) calls them on arrays: Job({"vjob": ...,
"dir": a work folder}, media=ArrayMedia(frames, masks, fps), keyframes={...}) keeps the work/ caches in that folder,
renders into Job.sink (the refined frames by clip frame) and writes no video. The vjob format is video.py's (its
docstring documents every key). Same code, same pixels: the fixtures (scripts/video_regress.py) hold both at 0 px.
"""
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion, map_coordinates

from imgpost import geometry as geo
from imgpost import images as im
from imgpost.raster import fill_polygon

from . import fit as vf
from . import glue as gl
from . import labelspace as ls
from . import level as lvl
from . import lookchecks as lk
from . import optics as op
from . import printalpha as pa
from . import render as rd
from . import track as tk
from . import vio


def poly_mask(shape, P):
    """Bool matte of a polygon (scene px) over a frame."""
    h, w = shape[:2]
    P = np.asarray(P, float)
    x0, y0 = np.maximum(np.floor(P.min(0)).astype(int) - 1, 0)
    x1, y1 = np.minimum(np.ceil(P.max(0)).astype(int) + 2, (w, h))
    out = np.zeros((h, w), bool)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = fill_polygon(y1 - y0, x1 - x0, P - (x0, y0))
    return out


class FileMedia:
    """A clip and its product matte as video files (the CLI)."""
    def __init__(self, clip, mask=None):
        self.clip, self.mask = clip, mask
        self.info = vio.probe(clip)
        self.has_masks = mask is not None

    def frames(self, a, b):
        return vio.read(self.clip, a, b, self.info)

    def masks(self, a, b):
        return vio.read(self.mask, a, b)[..., 0] > 127


class ArrayMedia:
    """Frames (T x H x W x 3 uint8) and the product's mattes (T x H x W bool) in memory: the ComfyUI node's input."""
    def __init__(self, frames, masks=None, fps=24.0):
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(f"frames: expected T x H x W x 3 uint8, got {frames.shape} {frames.dtype}")
        if masks is not None and masks.shape != frames.shape[:3]:
            raise ValueError(f"masks: expected {frames.shape[:3]} to match the frames, got {masks.shape}")
        from fractions import Fraction
        self._f, self._m = frames, None if masks is None else masks.astype(bool, copy=False)
        f = Fraction(float(fps)).limit_denominator(1001)
        self.info = {"w": frames.shape[2], "h": frames.shape[1], "fps": float(fps), "fps_str": f"{f.numerator}/{f.denominator}",
                     "frames": frames.shape[0], "matrix": "bt709", "range": "tv", "audio": False}
        self.has_masks = masks is not None

    def frames(self, a, b):
        return self._f[a:b]

    def masks(self, a, b):
        return self._m[a:b]


class _InfoMedia:
    """A render worker's media: the clip's description only (each task brings its own frame and matte)."""
    def __init__(self, info):
        self.info, self.has_masks = info, True


def pitched(H, sag, polygon, quiet=False):
    """A tube tilted toward the camera (V28). A re-acquired mapping has no pitch of its own (it comes from the matte's
    outline), so circles round the product stay straight lines. sag: px the label's line through the middle of its
    band bows toward the base at the point facing the camera, as the source's own print lines show it (+ = a smile,
    the camera looking down). It pivots on the point facing the camera (where re-acquisition registered the print):
    that point stays, the sides lift."""
    from .surface import Tube
    P = np.asarray(polygon, float)
    r_mid = H.R + H.k * 0.5 * (P[:, 1].min() + P[:, 1].max())
    ax = H.A[:, 1] / np.linalg.norm(H.A[:, 1])
    A = H.A.copy()
    A[:, 2] = A[:, 2] + ax * (sag / r_mid)
    a = H.a - ax * sag
    if not quiet:
        print(f"pitched    the re-acquired tube toward the camera: {sag:+g} px of sag at the label's middle")
    return Tube(H.u0, H.Ru, H.R, H.k, A, a)


def auto_pitch(H, label_img, polygon, frame, matte, rows=None):
    """The sag (pitched) that makes the label's print, through H, bow like the lines in `frame` (V28): both measured
    strip by strip (surface.line_bow), corrected by secant steps to within 1 px. -> (sag, report)"""
    from . import surface
    rows = rows or list(surface.text_rows(label_img, polygon))
    target = surface.line_bow(surface.image_sampler(frame), H, matte, rows)[0]

    def bow(sag):
        Hp = pitched(H, sag, polygon, quiet=True)
        return surface.line_bow(surface.texture_sampler(label_img, Hp), Hp, matte, rows)[0]
    steps = [(0.0, bow(0.0))]
    sag = target - steps[0][1]
    for _ in range(5):
        b = bow(sag)
        steps.append((sag, b))
        if abs(target - b) <= 1.0:
            break
        (s0, b0), (s1, b1) = steps[-2], steps[-1]
        g = (b1 - b0) / (s1 - s0) if abs(s1 - s0) > 1e-6 and abs(b1 - b0) > 1e-6 else 1.0
        sag = s1 + (target - b1) / (g if 0.2 < g < 3.0 else 1.0)
    sag, got = min(steps, key=lambda sb: abs(target - sb[1]))
    # a camera can't look down a tube by more than about 30 degrees and still read its label: past that the bow
    # came from garbled print, not from pitch
    P = np.asarray(polygon, float)
    lim = 0.5 * (H.R + H.k * 0.5 * (P[:, 1].min() + P[:, 1].max()))
    clamped = abs(sag) > lim
    sag = float(np.clip(sag, -lim, lim))
    return sag, {"sag_px": round(sag, 2), "source_bow_px": round(float(target), 2), "render_bow_px": round(float(got), 2),
                 "rows": [round(float(r), 1) for r in rows], "steps": [[round(a, 2), round(b, 2)] for a, b in steps],
                 **({"clamped_to_px": round(lim, 1)} if clamped else {})}


class Job:
    """One shot job. `path` is its vjob.json (the CLI: the clip and the mask are video files), or {"vjob": dict, "dir":
    folder} with `media` (ArrayMedia: frames and mattes in memory; nothing is written but the work/ caches) and
    `keyframes` ({anchor.keyframe name: uint8 image})."""
    def __init__(self, path, media=None, keyframes=None):
        if isinstance(path, dict):
            self.path, self.dir, self.v = None, Path(path["dir"]), path["vjob"]
        else:
            self.path = Path(path).expanduser().resolve()
            self.dir = self.path.parent
            self.v = json.loads(self.path.read_text())
        self.work, self.qa = self.dir / "work", self.dir / "qa"
        self.work.mkdir(parents=True, exist_ok=True)
        self.qa.mkdir(exist_ok=True)
        self.keyframes = keyframes or {}
        if media is None:
            self.clip = im.resolve(self.v["clip"], self.dir)
            self.mask_path = im.resolve(self.v["mask"], self.dir) if self.v.get("mask") else None
            media = FileMedia(self.clip, self.mask_path)
        else:
            self.clip = self.mask_path = None
        self.media, self.info, self.has_masks = media, media.info, media.has_masks
        self.sink = None if self.clip is not None else {}       # in memory: the refined frames, by clip frame
        self.s0, self.s1 = self.v.get("shot", [0, self.info["frames"]])
        if self.v.get("render"):
            self.v["render"] = print_defaults(self.v["render"])
        self._frames = self._masks = None
        anc = self.v["anchor"]
        self.keyframe = bool(anc.get("keyframe") or anc.get("texture_job"))
        self.tube = self.keyframe and (anc.get("model") == "tube" or bool(anc.get("texture_job")))
        self.valid_path = self.work / "label_valid.png"
        self._valid = None
        self.H_reacq = None
        if anc.get("texture_job"):
            # another shot of the clip: the keyframe's unrolled label from the job that made it, found again on a
            # frame of this shot (tracks don't cross cuts)
            tj = im.resolve(anc["texture_job"], self.dir)
            self.kmeta = json.loads((tj / "work" / "anchor.json").read_text())
            self.kf_path = im.resolve(self.kmeta["spec"]["keyframe"], tj)
            self.label_img = im.load_u8(tj / "work" / "label_texture.png")[0]
            self.valid_path = tj / "work" / "label_valid.png"
            if anc.get("texture_rotate_deg"):
                # the keyframe's print can sit a degree or two off its own body's axis (the generator drew it so);
                # another shot's generator drew it level: turn the print, keep the geometry on the silhouette
                from scipy.ndimage import rotate
                r = float(anc["texture_rotate_deg"])
                self.label_img = np.clip(np.stack([rotate(self.label_img[..., c].astype(float), r, reshape=False, order=3,
                                                          mode="nearest") for c in range(3)], -1) + 0.5, 0, 255).astype(np.uint8)
                self._valid = rotate((np.asarray(Image.open(self.valid_path)) > 127).astype(float), r, reshape=False,
                                     order=1, mode="constant") > 0.5
            self.anchor = int(anc["chain"]["frame"]) if anc.get("chain") else int(anc["reacquire"])
            tb = self.kmeta["tube"]
            self.ajob = {"align": {"model": "tube", "tube": {k: tb[k] for k in ("u0", "Ru", "R", "k")}},
                         "reference": str(tj / "work" / "label_texture.png"),
                         "reference_outline": {"type": "polygon", "pts": self.kmeta["label_polygon"]}}
            self.ajob_path = self.path
            self.H_reacq = self._chain(anc) if anc.get("chain") else self._reacquire(anc, tj)
            if anc.get("pitch_sag_px") == "auto":
                self.H_reacq = self._pitched(self.H_reacq, self._auto_pitch(anc))
            elif anc.get("pitch_sag_px"):
                self.H_reacq = self._pitched(self.H_reacq, float(anc["pitch_sag_px"]))
        elif self.keyframe:
            if str(anc["keyframe"]) in self.keyframes:           # in memory
                self.kf_path, self.kf = Path(str(anc["keyframe"])), self.keyframes[str(anc["keyframe"])]
            else:
                self.kf_path = im.resolve(anc["keyframe"], self.dir)
                self.kf = im.load_u8(self.kf_path)[0]
            self.kmeta = self._keyframe_meta(anc)
            self.anchor = self.kmeta["frame"]
            self.ajob_path = self.path
            if self.tube:                # label space: the keyframe unrolled onto the tube
                self.label_img = im.load_u8(self.work / "label_texture.png")[0]
                tb = self.kmeta["tube"]
                self.ajob = {"align": {"model": "tube", "tube": {k: tb[k] for k in ("u0", "Ru", "R", "k")}},
                             "reference": str(self.work / "label_texture.png"),
                             "reference_outline": {"type": "polygon", "pts": self.kmeta["label_polygon"]}}
            else:                        # label space: the keyframe's own pixels
                self.label_img = self.kf
                P = np.asarray(self.kmeta["polygon"], float)
                (x0, y0), (x1, y1) = P.min(0), P.max(0)
                ident = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                self.ajob = {"align": {"model": "homography", "points": [{"ref": p, "scene": p} for p in ident]},
                             "reference": str(self.kf_path), "reference_outline": {"type": "polygon", "pts": self.kmeta["polygon"]}}
        else:
            self.anchor = int(anc["frame"])
            self.ajob_path = im.resolve(anc["job"], self.dir)
            self.ajob = json.loads(self.ajob_path.read_text())
        if self.clip is None:                # in memory: no deliverables; the QA plot goes into qa/
            self.out = self.lossless = self.sbs = None
            self.plot = self.qa / "label-qa.png"
            return
        stem = self.clip.stem
        out = self.v.get("output") or {}
        self.out = im.resolve(out.get("refined", self.clip.with_name(stem + "_refined.mp4")), self.dir)
        self.lossless = im.resolve(out["lossless"], self.dir) if out.get("lossless") else None
        self.sbs = im.resolve(out.get("side_by_side", self.clip.with_name(stem + "_side-by-side.mp4")), self.dir)
        self.plot = im.resolve(out.get("qa_plot", self.clip.with_name(stem + "_label-qa.png")), self.dir)

    def valid(self):
        """Where the unrolled keyframe is really on the product (label px)."""
        return self._valid if self._valid is not None else np.asarray(Image.open(self.valid_path)) > 127

    def owned(self, res):
        """Shot frames this job delivers: those the tracks hold, or only render.frames of them (a patch job that
        redoes a few frames of an approved render and leaves the rest to it)."""
        only = self.v.get("render", {}).get("frames")
        V = res["valid"]
        return [t for t in V if t + self.s0 in set(only)] if only else list(V)

    def spec(self):
        """What a render worker rebuilds this job from: its vjob path, or the in-memory job."""
        if self.path is not None:
            return str(self.path)
        return {"job": {"vjob": self.v, "dir": str(self.dir)}, "info": self.info, "keyframes": self.keyframes}

    def output(self):
        """The shot as rendered: the refined file (CLI), or the source with the refined frames kept in memory."""
        if self.sink is None:
            return vio.read(self.out, self.s0, self.s1, self.info)
        return np.stack([self.sink.get(t + self.s0, f) for t, f in enumerate(self.frames())])

    def frames(self):
        if self._frames is None:
            self._frames = self.media.frames(self.s0, self.s1)
        return self._frames

    def masks(self):
        if self._masks is None and self.has_masks:
            self._masks = self.media.masks(self.s0, self.s1)
        return self._masks

    def _keyframe_meta(self, anc):
        """Which frame the keyframe was made from, and the label's outline on it (cached in work/anchor.json)."""
        from . import keyframe as kf
        cache = self.work / "anchor.json"
        spec = {"keyframe": str(anc["keyframe"]), "frame": anc.get("frame", "auto"), "band": anc["band"],
                "model": anc.get("model", "homography"), "theta_lim": anc.get("theta_lim", 75.0)}
        if cache.exists():
            m = json.loads(cache.read_text())
            if m.get("spec") == spec:
                return m
        if not self.has_masks:
            raise SystemExit("anchor.keyframe needs the product's matte (vjob mask)")
        if anc.get("frame", "auto") == "auto":
            k, d, d2 = kf.find_keyframe(self.frames(), self.masks(), self.kf)
            k += self.s0
        else:
            k, d, d2 = int(anc["frame"]), None, None
        poly, quad, axis = kf.label_band(self.masks()[k - self.s0], anc["band"])
        m = {"spec": spec, "frame": int(k), "match_outside_mask": d, "runner_up": d2, "polygon": poly, "quad": quad, **axis}
        if spec["model"] == "tube":
            from . import surface
            T, W, Hh, info = surface.from_keyframe(self.masks()[k - self.s0], anc["band"], spec["theta_lim"])
            tex, lpoly = surface.unroll(self.kf, T, W, Hh, spec["theta_lim"])
            Image.fromarray(tex).save(self.work / "label_texture.png")
            Image.fromarray(surface.unroll_mask(self.masks()[k - self.s0], T, W, Hh).astype(np.uint8) * 255).save(
                self.work / "label_valid.png")
            m.update(tube=T.to_json(), label_polygon=lpoly, tube_info=info)
            print(f"unrolled   the keyframe onto a tube: radius {info['R0']} px, taper {info['taper']} px/px (half-width "
                  f"misfit {info['fit_half_width_rms']} px), label {W} x {Hh} px, texture within {spec['theta_lim']} deg of "
                  f"facing the camera")
        cache.write_text(json.dumps(m))
        print(f"keyframe   {self.kf_path.name} is frame {k}" + (f" (mean |diff| outside the matte {d} levels; next best "
              f"frame {d2})" if d is not None else ""))
        return m

    def _chain(self, anc):
        """No cut between the shots: the other job's own mapping on a frame both share is this shot's anchor, so the
        label carries on from it (its spin, roll and position) instead of being found again."""
        cdir = im.resolve(anc["chain"]["job"], self.dir)
        cv = json.loads((cdir / "vjob.json").read_text())
        r = pickle.load(open(cdir / "work" / "fit.pkl", "rb"))
        H = r["smooth"][int(anc["chain"]["frame"]) - int(cv.get("shot", [0])[0])]
        print(f"chained   frame {anc['chain']['frame']} takes {cdir.name}'s mapping as this shot's anchor")
        return H

    def _pitched(self, H, sag, quiet=False):
        return pitched(H, sag, self.kmeta["label_polygon"], quiet)

    def _auto_pitch(self, anc):
        """pitch_sag_px "auto" (V28): the sag that makes the label's print bow like the source's own lines on the anchor
        frame. The source's bow is measured strip by strip; so is a render of the label's print alone through the
        pitched mapping; the sag is corrected by secant steps until they agree within 1 px (the render's bow isn't the
        sag itself: example2's 32 px rendered as 22). Cached in work/pitch.json."""
        from . import surface
        cache = self.work / "pitch.json"
        rows = anc.get("pitch_rows") or list(surface.text_rows(self.label_img, self.kmeta["label_polygon"]))
        key = {"frame": self.anchor, "rows": [round(float(r), 1) for r in rows], "tube": self.H_reacq.to_json()}
        if cache.exists():
            c = json.loads(cache.read_text())
            if c.get("key") == key:
                return float(c["sag_px"])
        fr = self.media.frames(self.anchor, self.anchor + 1)[0]
        mf = self.media.masks(self.anchor, self.anchor + 1)[0]
        sag, rep = auto_pitch(self.H_reacq, self.label_img, self.kmeta["label_polygon"], fr, mf, rows)
        cache.write_text(json.dumps(dict(rep, key=key)))
        print(f"pitch      auto: the source's lines bow {rep['source_bow_px']:+.1f} px (label rows {rows[0]:.0f}-{rows[1]:.0f}); "
              f"a sag of {sag:+.1f} px makes the label's bow {rep['render_bow_px']:+.1f} px")
        return sag

    def _reacquire(self, anc, tj):
        """The keyframe's label found on this shot's anchor frame (cached in work/reacquire.json)."""
        from . import surface
        cache = self.work / "reacquire.json"
        key = {"frame": self.anchor, "texture_job": str(tj), "roll_search": anc.get("roll_search"),
               "roll_from_print": anc.get("roll_from_print")}
        for k in ("scales", "texture_rotate_deg"):           # (opt-in keys stay out of older jobs' cache keys)
            if anc.get(k) is not None:
                key[k] = anc[k]
        if cache.exists():
            c = json.loads(cache.read_text())
            if c.get("key") == key:
                t = c["tube"]
                return surface.Tube(t["u0"], t["Ru"], t["R"], t["k"], t["A"], t["a"])
        kf_frame = int(self.kmeta["frame"])
        mk = self.media.masks(kf_frame, kf_frame + 1)[0]
        fr = self.media.frames(self.anchor, self.anchor + 1)[0]
        mf = self.media.masks(self.anchor, self.anchor + 1)[0]
        valid = self.valid()
        rl = anc.get("roll_search")                  # [lo, hi, step] degrees
        rolls = tuple(np.round(np.arange(rl[0], rl[1] + 1e-9, rl[2]), 3)) if rl else (0.0,)
        kw = {"scales": tuple(anc["scales"])} if anc.get("scales") else {}   # [1.0]: the silhouette's own scale
        H, rep = surface.reacquire(self.label_img, valid, self.kmeta["tube"], mk.sum(), fr, mf, self.kmeta["spec"]["band"],
                                   self.kmeta["spec"]["theta_lim"], rolls=rolls, **kw)
        if anc.get("roll_from_print"):               # {"box": [u0, v0, u1, v1]} label px around lines of text
            H, rr = surface.roll_from_print(H, fr, mf, self.label_img, anc["roll_from_print"]["box"])
            rep["roll_from_print"] = rr
            rep["roll_deg"] = rr.get("roll_deg", 0.0)
        cache.write_text(json.dumps({"key": key, "tube": H.to_json(), "report": rep}))
        print(f"reacquired the keyframe's label on frame {self.anchor}: scale {rep['scale']}, spin {rep['spin_deg']} deg, "
              f"axial offset {rep['axial_offset_px']} px, roll {rep.get('roll_deg', 0.0)} deg, print correlation {rep['verify_ncc']}")
        return H

    def H0(self):
        if self.H_reacq is not None:
            return self.H_reacq
        if self.tube:
            from .surface import Tube
            tb = self.kmeta["tube"]
            return Tube(tb["u0"], tb["Ru"], tb["R"], tb["k"], tb["A"], tb["a"])
        return geo.fit(self.ajob["align"])[0]

    def base(self):
        """The anchor's model for the per-frame fits; fit.perspective adds projective terms to a cylinder (a product
        moving nearer and further, or turning toward the camera, foreshortens more than an affine camera allows)."""
        fc = self.v.get("fit", {})
        return dict(self.ajob["align"], perspective=bool(fc.get("perspective", self.ajob["align"].get("perspective", False))),
                    perspective_reg=float(fc.get("perspective_reg", 0.0)), camera=fc.get("camera", "ortho"))

    def artwork(self):
        if self.keyframe:
            from .keyframe import KeyframeArtwork
            if self.tube:
                valid = self.valid()
                art = KeyframeArtwork(self.label_img, self.kmeta["label_polygon"], None, valid=valid, **self.v.get("artwork", {}))
                art.unroll = True
                return art, self.label_img
            return KeyframeArtwork(self.kf, self.kmeta["polygon"], self.kmeta["quad"], **self.v.get("artwork", {})), self.kf
        ref_u8, ref_a, _ = im.load_u8(im.resolve(self.ajob["reference"], self.ajob_path.parent))
        return rd.Artwork(ref_u8, ref_a, self.ajob["reference_outline"], self.v.get("render", {}).get("polarity", "light")), ref_u8

    def outline(self):
        return np.asarray(self.ajob["reference_outline"]["pts"], float)

    def key(self, t, H):
        """The product's matte on shot frame t: the mask video, or every piece of the body colour under the label."""
        if self.has_masks:
            return self.masks()[t]
        frame = self.frames()[t]
        k = self.v.get("key", {})
        region = binary_dilation(poly_mask(frame.shape, geo.map_polygon(H, self.outline())), iterations=4)
        c = geo.apply_H(H, [self.outline().mean(0)])[0]
        return tk.product_key(frame, tuple(k.get("hue", (-14, 28))), k.get("sat", 0.27), k.get("val", 0.35), near=tuple(c),
                              region=region)


def stage_track(J):
    fr = J.frames()
    a = J.anchor - J.s0
    H0 = J.H0()
    tc = J.v.get("track", {})
    key = J.key(a, H0)
    if J.keyframe:                       # the label and a little around it (the cap or pump isn't on its surface)
        key = key & binary_dilation(poly_mask(key.shape, geo.map_polygon(H0, J.outline())), iterations=8)
    erode = tc.get("erode", 4)
    if tc.get("on", "all") == "print":
        # only on the print: on a glossy body, points on bare surface follow its reflections, which stay with the
        # light while the product turns; the print (garbled or not) stays with the surface
        from scipy.ndimage import grey_closing, grey_opening
        L = im.lum(im.srgb_to_lin(im.to_float(fr[a])))
        n = int(J.v.get("render", {}).get("print_tophat", 7))
        top = np.maximum(L - grey_opening(L, size=(n, n)), grey_closing(L, size=(n, n)) - L)
        key = binary_dilation((top > tc.get("print_thresh", 0.02)) & binary_erosion(key, iterations=erode), iterations=2)
        erode = 0
    step = tc.get("step", "auto")
    if step == "auto":
        area = (binary_erosion(key, iterations=erode) if erode else key).sum()      # iterations=0 erodes everything
        step = max(3, int(round(np.sqrt(area / tc.get("points", 900)))))
    xy = tk.grid_queries(key, step, erode)
    crop = tc.get("crop", "auto")
    if crop == "auto":                   # the product's reach over the shot, plus a margin
        boxes = []
        for t in range(0, len(fr), 2):
            k = J.masks()[t] if J.has_masks else J.key(t, H0)
            ys, xs = np.nonzero(k)
            if len(xs) > 0.3 * key.sum():
                boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
        B = np.array(boxes)
        m = tc.get("margin", 60)
        crop = [max(0, int(B[:, 0].min()) - m), max(0, int(B[:, 1].min()) - m),
                min(J.info["w"], int(B[:, 2].max()) + m), min(J.info["h"], int(B[:, 3].max()) + m)]
    t0 = time.time()
    tracks, vis, est = tk.track_points(fr, xy, a, crop, res=tc.get("res"), backend=tc.get("backend", "alltracker"))
    np.savez_compressed(J.work / "tracks.npz", xy=xy, tracks=tracks.astype(np.float32), vis=vis.astype(np.float32),
                        anchor=a, crop=crop, est=est.astype(np.float32))
    print(f"track      {len(xy)} points (every {step} px) from frame {J.anchor}, crop {crop}, {len(fr)} frames in "
          f"{time.time() - t0:.0f} s; visibility mean {vis.mean():.3f}, lowest frame {vis.mean(1).min():.3f}")


def stage_fit(J):
    d = np.load(J.work / "tracks.npz")
    H0 = J.H0()
    xy0, tracks, vis = d["xy"], d["tracks"].astype(float), d["vis"].astype(float)
    uv, ok = vf.label_coords(H0, xy0)
    P = J.outline()
    if J.keyframe:                       # tracks on the label only (label px)
        lab = poly_mask(J.label_img.shape, P)
        xi = np.clip(np.round(uv[:, 0]).astype(int), 0, lab.shape[1] - 1)
        yi = np.clip(np.round(uv[:, 1]).astype(int), 0, lab.shape[0] - 1)
        keep = ok & lab[yi, xi]
    else:
        keep = ok & (uv[:, 1] >= P[:, 1].min()) & (uv[:, 1] <= P[:, 1].max())
    if hasattr(H0, "axis_x"):
        keep &= np.abs(uv[:, 0] - H0.axis_x) < 0.97 * H0.r          # grazing tracks slide along the limb
    fc = J.v.get("fit", {})
    res = vf.fit_shot(J.base(), H0, xy0[keep], tracks[:, keep], vis[:, keep], fc.get("wmin", 0.5), fc.get("reject", 2.5),
                      fc.get("half", 3), residual=fc.get("residual"), ngrid=fc.get("grid", 5), anchor=J.anchor - J.s0,
                      min_inliers=fc.get("min_inliers", 0.5), max_rms=fc.get("max_rms", 3.0), max_rigid=fc.get("max_rigid", 12.0),
                      shear_max=fc.get("shear_max"), morph_rigid=[int(t) - J.s0 for t in fc.get("morph_rigid", [])],
                      morph_skip=[int(t) - J.s0 for t in fc.get("morph_skip", [])])
    pickle.dump(res, open(J.work / "fit.pkl", "wb"))
    E = res["errors"]
    for e in E:
        e["frame"] += J.s0
    stops = _confirm_cuts(J, res)
    res["stops"] = stops
    pickle.dump(res, open(J.work / "fit.pkl", "wb"))
    stops = {k: dict(v, frame=None if v["frame"] is None else v["frame"] + J.s0) for k, v in stops.items()}
    (J.qa / "fit.json").write_text(json.dumps({"impacts": [int(i) + J.s0 for i in res["impacts"]],
                                               "valid": [int(t) + J.s0 for t in res["valid"]], "stops": stops,
                                               "frames": E}, indent=0))
    V = res["valid"]
    rms = np.array([E[t]["rms_px"] for t in V])
    rig = np.array([E[t]["rigid_rms_px"] for t in V])
    sh = np.array([E[t]["smooth_shift_px"] for t in V])
    print(f"fit        {keep.sum()} tracks on the label; the tracks hold frames {V[0] + J.s0}-{V[-1] + J.s0} ({len(V)} of "
          f"{len(E)}); there reprojection rms median {np.median(rms):.2f} px, p95 {np.percentile(rms, 95):.2f}, worst "
          f"{rms.max():.2f} (frame {V[int(rms.argmax())] + J.s0})" +
          (f"; rigid model alone {np.median(rig):.2f} median" if fc.get("residual") else "") +
          f"; smoothing moved the label {np.median(sh):.2f} px (median), {sh.max():.2f} max; "
          f"{sum(t in res['kept_raw'] for t in V)} frames kept their own fit (smoothing fought their tracks: judder)" +
          (f"; morph frames (a shear spike: shape from the frames either side) {[t + J.s0 for t in res['morph']]}"
           if res.get("morph") else "") + "; the run stops: " + _stops_text(stops))


def _frame_hist(f, bins=8):
    q = (f // (256 // bins)).astype(np.int32)
    h = np.bincount(((q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]).ravel(), minlength=bins ** 3).astype(float)
    return h / h.sum()


def _confirm_cuts(J, res):
    """A track jump is a cut only if the picture jumps too: the whole frame's colour histogram changes far more
    between the two frames than it does from frame to frame inside the run. A product moving fast (a blurred entry)
    jumps its tracks without one; a box rising into view (V15) changes the histogram without a track jump. The
    frame evidence goes into each stop's values; a "cut" without it becomes the next reason that fails (or
    "fast motion")."""
    stops = res.get("stops")
    if not stops:
        return stops or {}
    V = res["valid"]
    fr = J.frames()
    H = {}

    def hist(t):
        if t not in H:
            H[t] = _frame_hist(fr[t])
        return H[t]
    steps = [0.5 * np.abs(hist(a) - hist(b)).sum() for a, b in zip(V[:-1], V[1:])][::max(1, len(V) // 40)]
    usual = float(np.median(steps)) if steps else 0.0
    out = {}
    for k, v in stops.items():
        if v["frame"] is None:
            out[k] = v
            continue
        t_in = V[0] if k == "start" else V[-1]
        d = float(0.5 * np.abs(hist(v["frame"]) - hist(t_in)).sum())
        spike = d > max(0.2, 4.0 * usual)
        failed = list(v["failed"])
        if "cut" in failed and not spike:
            failed.remove("cut")
            failed.append("fast motion")
        reason = failed[0] if failed else "misfit"
        out[k] = dict(v, reason=reason, failed=failed,
                      values=dict(v["values"], frame_change=round(d, 3), usual_frame_change=round(usual, 3)))
    return out


def _stops_text(stops):
    out = []
    for k, v in stops.items():
        where = "at the shot edge" if v["frame"] is None else "at %d (%s)" % (v["frame"], v["reason"])
        out.append("%s %s" % (k, where))
    return ", ".join(out)


def _fit(J):
    r = pickle.load(open(J.work / "fit.pkl", "rb"))
    r.setdefault("valid", list(range(len(r["errors"]))))     # fits written before the shot extent existed
    r.setdefault("kept_raw", [])
    return r


def _white(J, fr):
    """The scene's white balance: the chroma of render.white_box (a white part of the product) on the anchor frame."""
    rc = J.v.get("render", {})
    if not rc.get("white_box"):
        return {}
    x0, y0, x1, y1 = rc["white_box"]
    w = np.median(im.srgb_to_lin(im.to_float(fr[J.anchor - J.s0][y0:y1, x0:x1])).reshape(-1, 3), 0)
    return {"white_chroma": (w / im.lum(w)).tolist()}


def stage_still(J):
    """The source's still label (median over the shot in label space), its softness (static and per frame) and, for
    an artwork, the ink's colour and level as the source renders its print."""
    fr = J.frames()
    res = _fit(J)
    V = res["valid"]
    art, _ = J.artwork()
    grid = ls.Grid(art, 4.0)
    stack, vis = [], []
    for t in V:
        rgb, v = grid.unwarp(fr[t], art.local(res["smooth"][t]), J.key(t, res["smooth"][t]))
        stack.append(rgb.astype(np.float32))
        vis.append(v)
    med, n = ls.still_label(np.stack(stack), np.stack(vis))
    np.savez_compressed(J.work / "still.npz", med=med.astype(np.float32), n=n, step=grid.step)
    rc = J.v.get("render", {})
    optics = {}
    a = J.anchor - J.s0
    P = J.outline()
    sig = op.limb_sigma(fr[a], res["smooth"][a], (P[:, 1].min() + 40, P[:, 1].max() - 40))
    optics["limb_sigma_px"] = round(sig, 3) if np.isfinite(sig) else None
    if J.keyframe:                       # a keyframe carries the shot's softness already
        optics["blur"] = 0.0
    elif np.isfinite(sig):
        optics["blur"] = round(float(np.clip(np.sqrt(max(sig ** 2 - 0.35 ** 2, 0.0)), 0, 1.5)), 3)
    else:
        optics["blur"] = 0.6
    # per frame: how much softer the label is than in the anchor frame (a keyframe) or a typical frame (artwork)
    Cs = res["C_smooth"]
    en, curve = {}, None
    for t in V:
        H = res["smooth"][t]
        roi = rd.work_box(art.local(H), art, fr[t].shape[:2], pad=4)
        x0, y0, x1, y1 = roi
        L = im.lum(im.srgb_to_lin(im.to_float(fr[t][y0:y1, x0:x1])))
        _, a_lab, _ = rd.wp.warp(art.pack, art.local(H), roi, art.refmask, ss=1, prefilter=0)
        m = binary_erosion(J.key(t, H)[y0:y1, x0:x1], iterations=4) & (a_lab > 0.9)
        if m.sum() < 500:
            continue
        en[t] = op.hp_energy(L, m)
        if t == a:
            curve = op.softness_curve(L, m)
    e_ref = en.get(a) if J.keyframe and a in en else float(np.median(list(en.values())))
    # motion smear along each frame's direction of travel, as the source shows it (generated frames often have none)
    smear = {}
    if a in V:
        Ha = res["smooth"][a]
        ra = rd.work_box(art.local(Ha), art, fr[a].shape[:2], pad=4)
        La = im.lum(im.srgb_to_lin(im.to_float(fr[a][ra[1]:ra[3], ra[0]:ra[2]])))
        ma = binary_erosion(J.key(a, Ha)[ra[1]:ra[3], ra[0]:ra[2]], iterations=4)
        Tn = len(Cs)
        for t in V:
            vel = 0.5 * (Cs[min(t + 1, Tn - 1)] - Cs[max(t - 1, 0)]).mean(0)
            sp = float(np.linalg.norm(vel))
            if sp < 3.0:
                continue
            H = res["smooth"][t]
            roi = rd.work_box(art.local(H), art, fr[t].shape[:2], pad=4)
            L = im.lum(im.srgb_to_lin(im.to_float(fr[t][roi[1]:roi[3], roi[0]:roi[2]])))
            m = binary_erosion(J.key(t, H)[roi[1]:roi[3], roi[0]:roi[2]], iterations=4)
            if m.sum() < 500 or ma.sum() < 500:
                continue
            d = vel / sp

            def axis_angle(Hm):           # the label's axis on screen
                q = geo.apply_H(Hm, [[P[:, 0].mean(), P[:, 1].min()], [P[:, 0].mean(), P[:, 1].max()]])
                return np.arctan2(*(q[1] - q[0])[::-1])
            rot = axis_angle(Ha) - axis_angle(H)
            d_a = np.array([d[0] * np.cos(rot) - d[1] * np.sin(rot), d[0] * np.sin(rot) + d[1] * np.cos(rot)])
            Ln = op.smear_length(L, m, La, ma, d, sp, d_a)
            if Ln > 0:
                smear[str(t + J.s0)] = [Ln, round(float(d[0]), 4), round(float(d[1]), 4)]
    optics["smear"] = smear
    soft = {str(t + J.s0): round(op.extra_blur(e / e_ref, curve), 3) for t, e in en.items()} if curve is not None else {}
    optics["soft"] = {k: v for k, v in soft.items() if v > 0}
    sm = optics["soft"]
    gains = []
    if not J.keyframe:
        # the ink's colour and level: the source's own print where it overlaps the artwork's boldest ink (the
        # generator renders the print in the shot's look even where it garbles it), per channel, slowest frames
        T = len(Cs)
        speed = np.array([np.linalg.norm(Cs[min(t + 1, T - 1)] - Cs[max(t - 1, 0)], axis=1).max() / 2 for t in range(T)])
        rcg = dict(rc, blur=optics["blur"], ink_gain=[1.0, 1.0, 1.0])
        rcg.update(_white(J, fr))
        for t in [int(t) for t in np.argsort(speed[V])[:8]]:
            t = V[t]
            H = res["smooth"][t]
            out, _, lay = rd.render(fr[t], H, [H], art, J.key(t, H), rcg, seed=t)
            x0, y0, x1, y1 = lay["roi"]
            m = (lay["cov"] > 0.85) & lay["P_src"]
            if m.sum() < 30:
                continue
            s_lin, o_lin = (im.srgb_to_lin(im.to_float(f[y0:y1, x0:x1])) for f in (fr[t], out))
            ls_, lo_ = im.lum(s_lin), im.lum(o_lin)
            gs = np.median(s_lin[m & (ls_ >= np.percentile(ls_[m], 90))], 0)
            go = np.median(o_lin[m & (lo_ >= np.percentile(lo_[m], 90))], 0)
            gains.append(gs / np.maximum(go, 1e-4))
    optics["ink_gain"] = np.clip(np.median(gains, 0), 0.5, 2.0).round(3).tolist() if gains else [1.0, 1.0, 1.0]
    (J.work / "optics.json").write_text(json.dumps(optics, indent=1))
    print(f"still      label median over {len(V)} frames (seen by {int(np.median(n[n > 0]))} frames median); static blur "
          f"{optics['blur']} px" + (f" ({optics['limb_sigma_px']} px at the silhouette)" if optics["limb_sigma_px"] else "") +
          f"; smeared along the motion on {len(smear)} frames" + (f" ({' '.join(f'{k}:{v[0]:g}' for k, v in smear.items())} px)" if smear else "") +
          f"; softer than {'the keyframe' if J.keyframe else 'usual'} on {len(sm)} frames" +
          (f" ({' '.join(f'{k}:{v:g}' for k, v in list(sm.items())[:12])} px extra blur)" if sm else "") +
          (f"; ink gain {optics['ink_gain']} from the source's print on {len(gains)} frames" if gains else ""))


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _sbs_frame(src, out, box, scale, t, shot):
    x0, y0, x1, y1 = box
    w, h = int(round((x1 - x0) * scale)), int(round((y1 - y0) * scale))
    a = Image.fromarray(src[y0:y1, x0:x1]).resize((w, h), Image.LANCZOS)
    b = Image.fromarray(out[y0:y1, x0:x1]).resize((w, h), Image.LANCZOS)
    c = Image.new("RGB", (2 * w + 8, h + 44), (252, 252, 251))
    c.paste(a, (0, 44))
    c.paste(b, (w + 8, 44))
    d = ImageDraw.Draw(c)
    f = _font(26)
    d.text((10, 8), "original", fill=(11, 11, 11), font=f)
    d.text((w - 120, 8), f"frame {t}", fill=(82, 81, 78), font=f)
    d.text((w + 18, 8), "refined" if shot else "not processed", fill=(11, 11, 11) if shot else (82, 81, 78), font=f)
    return np.asarray(c)


def _render_context(J, rc, optics, shutter, ax_ref, res=None, art=None):
    """What every frame's render needs, built once: in the main process, or in each render worker."""
    res = _fit(J) if res is None else res
    art = J.artwork()[0] if art is None else art
    still = None
    if (J.work / "still.npz").exists():
        s = np.load(J.work / "still.npz")
        still = ls.sampler(ls.Grid(art, float(s["step"])), s["med"].astype(float))
    Pv = J.outline()[:, 1]
    return {"res": res, "art": art, "rc": rc, "optics": optics, "shutter": shutter, "still": still, "ax_ref": ax_ref,
            "Pv": Pv, "vmid": 0.5 * (Pv.min() + Pv.max()), "morph": set(res.get("morph", [])),
            "ax_mode": rc["limb_snap"].get("axial") if isinstance(rc.get("limb_snap"), dict) else None,
            "residual": J.v.get("fit", {}).get("residual"), "body": rc.get("_body") or {},
            "sparkles": frame_set(rc["sparkles"]) if isinstance(rc.get("sparkles"), list) else None,
            "glow": frame_set(rc["spark_glow"]) if isinstance(rc.get("spark_glow"), list) else None,
            "print": rc.get("_print"), "glue": rc.get("_glue"), "level": rc.get("_level")}


def frame_set(spec):
    """Clip frames from a settings list: frames (98) and half-open runs ([98, 119]: 98-118), mixed freely."""
    out = set()
    for x in spec or []:
        if isinstance(x, (list, tuple)):
            out.update(range(int(x[0]), int(x[1])))
        else:
            out.add(int(x))
    return out


PRINT_RENDER = {"soft_cap": 0.3, "smear": False}   # what render.alpha "print" sets unless the settings do (V40)
PRINT_FEATHER_V_PX = 4


def print_defaults(render, given=None):
    """render.alpha "print" (V40) and the defaults that go with it, wherever `given` (the settings as written; default:
    `render` itself) doesn't set them: the still stage's per-frame softness capped at 0.3 px and no motion smear (a
    restored label stays sharp and legible; a false 26 px smear happens), the label's end fade (feather.v_px) 4 px (the
    24 px default dimmed print that starts inside it: the tube's top N), and the sparkle re-add off, always (it ghosts
    the garbled print where that morphs away from its still median, V37; the source's glitter stays anyway, off the
    letters). What it sets goes into render._defaults ({key: {"value", "by"}}), which the render logs and reports.
    -> the render settings (`render` itself when alpha isn't "print")"""
    if not isinstance(render, dict) or render.get("alpha", "label") != "print":
        return render
    given = render if given is None else (given or {})
    out = dict(render)
    added = {}
    by = "render.alpha \"print\""
    for k, v in PRINT_RENDER.items():
        if k not in given:
            if out.get(k) != v:
                added[k] = {"value": v, "by": by}
            out[k] = v
    if isinstance(out.get("feather"), dict) and "v_px" not in (given.get("feather") or {}):
        if out["feather"].get("v_px") != PRINT_FEATHER_V_PX:
            added["feather.v_px"] = {"value": PRINT_FEATHER_V_PX, "by": by}
        out["feather"] = dict(out["feather"], v_px=PRINT_FEATHER_V_PX)
    if out.get("sparkles", True) is not False:
        if "sparkles" in out:                # asked for: said so in the render's log
            out["_sparkles_asked"] = out["sparkles"]
        out["sparkles"] = False
        added["sparkles"] = {"value": False, "by": by}
    if added:
        out["_defaults"] = dict(out.get("_defaults") or {}, **added)
    return out


# the render stages and looks the measured job (imgauto vjobs) switches on that a run's settings may not name: the
# render logs and reports which of them came from a default rather than the settings
DEFAULT_KEYS = ("body_snap", "glue", "level", "alpha", "limb_snap", "soft_cap", "smear")


def defaults_on(render, given, by):
    """{key: {"value", "by"}}: the DEFAULT_KEYS `render` has on that `given` (the settings as written) doesn't set; by:
    who set them, a text or {key: text}."""
    given = given or {}
    return {k: {"value": render[k], "by": by.get(k, by.get("*")) if isinstance(by, dict) else by} for k in DEFAULT_KEYS
            if k in render and k not in given and render[k] not in (None, "label") and
            not (k == "smear" and render[k] == "auto")}


def defaults_text(d):
    def v(x):
        return "on" if x is True else "off" if x is False else json.dumps(x)
    return "; ".join(f"{k} {v(e['value'])} ({e['by']})" for k, e in d.items())


def render_optics(optics, rc):
    """optics.json (the still stage's measures) as the render uses them: the per-frame softness capped at
    render.soft_cap px (none: as measured), the motion smear only where render.smear keeps it ("auto", the default: as
    measured; false: none; [clip frames | [a, b)]: only there, for motion blur the source clearly shows)."""
    o = dict(optics)
    cap = rc.get("soft_cap")
    if cap is not None:
        o["soft"] = {k: min(float(v), float(cap)) for k, v in o.get("soft", {}).items()}
    sm = rc.get("smear", "auto")
    if sm is False:
        o["smear"] = {}
    elif isinstance(sm, list):
        keep = frame_set(sm)
        o["smear"] = {k: v for k, v in o.get("smear", {}).items() if int(k) in keep}
    elif sm not in ("auto", True):
        raise ValueError(f"render.smear {sm!r}: \"auto\", false or a list of clip frames")
    return o


def _print_setup(J, C, fr, frames):
    """render.alpha "print": its settings (printalpha.DEFAULTS under render.print_alpha) and the keyframe's own ink in
    label space, read on the keyframe's frame through the mapping the render places there. Where that can't be read (a
    re-acquired shot: its keyframe's frame is in another shot), the render's own print stands in (ink "render").
    -> (settings, ink label px N x 2 or None, report)"""
    rc = C["rc"]
    if rc.get("polarity", "light") != "light":
        raise ValueError(f"render.alpha \"print\" carries light print (white ink on a body) only for now; this job's "
                         f"print polarity is {rc.get('polarity')!r}")
    c = pa.settings(rc.get("print_alpha"))
    rep, pts = {"ink": c["ink"]}, None
    if c["ink"] == "keyframe":
        a = J.anchor - J.s0
        anc = J.v["anchor"]
        why = None
        if not anc.get("keyframe") or anc.get("texture_job"):
            why = "the keyframe's own frame isn't in this shot (texture_job)"
        elif a not in frames:
            why = f"the tracks don't hold the keyframe's frame {J.anchor}"
        else:
            H = _place(C, J, a)[0]
            pts, n = pa.keyframe_ink(J.kf, fr[a], J.key(a, H), H, J.label_img.shape[:2], c)
            rep["ink_px"] = n
            if not n:
                pts, why = None, "no ink found on the keyframe (ink_levels, ink_p90)"
        if why:
            rep.update(ink="render", ink_fallback=why)
    return c, pts, rep


def _axial(C, J, t, H):
    """A reconstructed (morph) frame's position along the axis, from the product's ends (limb_snap "axial"). -> (H,
    the shift in label px or None)"""
    rc, ax_ref, ax_mode = C["rc"], C["ax_ref"], C["ax_mode"]
    if ax_ref is not None and t in C["morph"]:
        # a reconstructed frame's position along the axis came from blurred or bad tracks: the product's end
        # (a crimp, a cap) lands where the frames with good tracks put it. Motion blur lengthens the matte at
        # both ends, so half of the extra length comes off the measured end
        e = vf.axis_ends(H, J.key(t, H), C["vmid"])
        if e is not None:
            ext = max(0.0, 0.5 * ((e[1] - e[0]) - (ax_ref[1] - ax_ref[0])))
            vm, vr = (e[0] + ext, ax_ref[0]) if ax_mode == "top" else (e[1] - ext, ax_ref[1])
            c = vm - vr
            if abs(c) <= float(rc["limb_snap"].get("axial_max", 60.0)):
                return rd.shift_ref(H, 0.0, c), round(float(c), 1)
    return H, None


def _limb(C, J, t, H):
    """The label's sides onto the product's matte (render.limb_snap). -> (mapping, report or None)"""
    rc, Pv = C["rc"], C["Pv"]
    if not rc.get("limb_snap"):
        return H, None
    # the model's straight taper misses a real tube's sides by 10-25 px at the label's ends; light and sparkles stay
    # read through the tracked mapping
    sc_ = {k: v for k, v in (rc["limb_snap"] if isinstance(rc["limb_snap"], dict) else {}).items()
           if k in ("step", "max_px", "inner", "lam", "hold")}
    return vf.silhouette_snap(H, J.key(t, H), (Pv.min(), Pv.max()), **sc_)


def _place(C, J, t):
    """Where the render puts the label on shot frame t: the tracked mapping, moved along the axis on a morph frame
    (limb_snap axial), placed by the product's body (render.body_snap) and snapped onto the limbs (limb_snap).
    -> (the mapping, the tracked mapping, the axial shift or None, the body correction or None, the limb snap's report
    or None). With render.glue the glued mapping (the placement carried along the product's own motion, V42); with
    render.level that mapping levelled with the product's base (V45)."""
    H0_ = C["res"]["smooth"][t]
    lv = C.get("level")
    if lv and t in lv:
        return lv[t], H0_, None, C["body"].get(t), None
    g = C.get("glue")
    if g and t in g:
        return g[t], H0_, None, C["body"].get(t), None
    H, sh = _axial(C, J, t, H0_)
    bs = C["body"].get(t)
    if bs is not None:
        # the label placed by the product's body, not by the generated print it tracks (render.body_snap, V39): along
        # the axis and turned onto the body's centreline; the limb snap then puts its sides on the matte
        H = vf.body_snapped(H, bs[1], bs[0], bs[2], bs[3])
    H, snap_rep = _limb(C, J, t, H)
    return H, H0_, sh, bs, snap_rep


def _render_one(C, J, t, frame, extras=None):
    """Shot frame t, from its source frame. extras (render.alpha "print"): {"gain": the print's presence gain, "blur":
    (direction, px) or None}, or {"measure": True} (the presence pre-pass: -> source / render presence or None).
    -> (output, info, motion span px, motion samples, the mapping that placed the print where it isn't the tracked one,
    else None)"""
    res, rc, optics, art = C["res"], C["rc"], C["optics"], C["art"]
    Hs, span = vf.sub_mappings(J.base(), res["controls"], res["C_smooth"], t, C["shutter"], residual=C["residual"])
    rct = dict(rc, blur=float(np.hypot(rc["blur"], optics.get("soft", {}).get(str(t + J.s0), 0.0))),
               smear=optics.get("smear", {}).get(str(t + J.s0)))
    if t in res.get("morph_source", {}):
        rct["source_H"] = res["morph_source"][t]
    H, H0_, sh, bs, snap_rep = _place(C, J, t)
    if sh is not None:
        rct["axial_shift"] = sh
    placed = None
    if H is not H0_:
        rct.setdefault("source_H", H0_)
        placed = H
    if C["glow"] is not None:                                # the glow only on the frames a sparkle trail crosses
        rct["spark_glow"] = t + J.s0 in C["glow"]
    if C["sparkles"] is not None:                            # the re-add only on these frames (glitter in front, V37)
        rct["sparkles"] = t + J.s0 in C["sparkles"]
    out, info, _ = rd.render(frame, H, Hs, art, J.key(t, H), rct, label_median=C["still"], seed=t + J.s0)
    pr = C["print"]
    if pr is not None:
        # render.alpha "print" (V40): only the print changes, carried onto the source's own body
        ch = np.any(out != frame, axis=-1)
        ex = extras or {}
        if ex.get("measure"):
            return pa.measure_presence(frame, out, ch, J.key(t, H), pr["c"], pa.ink_mask(H, pr["pts"], ch.shape)
                                       if pr["pts"] is not None else None) if ch.any() else None
        if ch.any():
            ink = pa.ink_mask(H, pr["pts"], ch.shape) if pr["pts"] is not None else None
            g, bl = float(ex.get("gain", 1.0)), ex.get("blur")
            out, pi = pa.composite(frame, out, ch, J.key(t, H), pr["c"], ink, gain=g, blur=bl)
            if ink is None:                  # (lost print is measured against the keyframe's own ink only)
                pi.pop("lost_print_px")
                pi.pop("ink_px")
            info["print_alpha"] = dict(pi, render_px=int(ch.sum()), presence_gain=round(g, 3),
                                       blur_px=round(float(bl[1]), 2) if bl is not None else 0.0)
    elif C["sparkles"] is not None and rct.get("sparkles"):
        # the review's ghost check on a sparkle re-add frame (V37: the re-add can bring the garbled print back): this
        # render against one without the re-add, which only exists here
        clean, _, _ = rd.render(frame, H, Hs, art, J.key(t, H), dict(rct, sparkles=False), label_median=C["still"],
                                seed=t + J.s0)
        info["ghost_px"] = lk.ghost_px(frame, clean, out, J.key(t, H))
    if snap_rep:
        info["snap"] = {k: v for k, v in snap_rep.items() if k in ("left_px", "right_px", "snap")}
    if "axial_shift" in rct:
        info["axial_shift"] = rct["axial_shift"]
    if bs is not None:
        info["body_snap"] = {"stretch": round(bs[0], 5), "shift": round(bs[1], 3), "roll_deg": round(bs[2], 3)}
    return out, info, span, len(Hs), placed


def _body_snap(J, C, fr, frames):
    """render.body_snap (V39): per frame, the correction that keeps the label at the reference frame's (the keyframe's)
    proportion of the product's body along its axis and on the body's centreline, measured on the mapping the render
    would place (the tracked one, snapped onto the limbs), median-filtered over the shot. -> ({shot frame: (f, p,
    roll_deg, pivot)} for the frames it moves, the report for render.json)"""
    rc = C["rc"]
    bc = dict(vf.BODY_SNAP, **(rc["body_snap"] if isinstance(rc["body_snap"], dict) else {}))
    P = J.outline()
    (u0, v0), (u1, v1) = P.min(0), P.max(0)
    uc, vr = 0.5 * (u0 + u1), (float(v0), float(v1))
    a = int(bc.get("ref_frame", J.anchor)) - J.s0
    if a not in frames:
        return {}, {"skipped": f"the tracks don't hold the reference frame {a + J.s0} (body_snap.ref_frame)"}

    def placed(t):
        H, _ = _axial(C, J, t, C["res"]["smooth"][t])
        return _limb(C, J, t, H)[0], H

    def measure(t, top, bottom, ref=None):
        # the gaps where the label's ends are rendered (snapped onto the limbs); the roll of its print (the mapping
        # under the limb snap, which shifts rows sideways but never turns the text lines)
        H, Hp = placed(t)
        return vf.body_measure(H, fr[t].astype(float).mean(-1), J.key(t, H).astype(float), vr, uc, top, bottom,
                               bc["reach"], bc["across"], bc["inside"], H_print=Hp, ref=ref, window=bc["window"])
    kinds = {"top": bc["top"], "bottom": bc["bottom"]}
    if "auto" in kinds.values():                           # decided once, on the reference frame
        Hk = placed(a)[0]
        auto = vf.body_edges(Hk, fr[a].astype(float).mean(-1), J.key(a, Hk).astype(float), vr, uc, bc["reach"])
        kinds = {k: auto[k] if v == "auto" else v for k, v in kinds.items()}
    ref = measure(a, kinds["top"], kinds["bottom"])
    if ref is None:
        return {}, {"skipped": f"no body measure on the reference frame {a + J.s0} (body_snap.top / bottom, reach)",
                    "edges": kinds}
    # every other frame round the gaps the reference's proportion predicts for it (None: its neighbours' correction)
    meas = {t: (ref if t == a else measure(t, kinds["top"], kinds["bottom"], ref)) for t in sorted(frames)}
    corr = vf.body_corrections(meas, ref, vr, bc["median"], bc["outlier_k"], bc["stretch"], bc["roll_max"],
                               bc["shift_max"])
    corr[a] = (1.0, 0.0, 0.0)                              # the keyframe's frame is the truth
    only = frame_set(bc["frames"]) if bc.get("frames") else None

    def pivot(t):
        return meas[t]["pivot"] if meas[t] is not None else \
            geo.apply_H(placed(t)[0], np.array([[uc, 0.5 * (vr[0] + vr[1])]]))[0].tolist()
    body = {t: (f, p, r, pivot(t)) for t, (f, p, r) in corr.items()
            if (f, p, r) != (1.0, 0.0, 0.0) and (only is None or t + J.s0 in only)}
    off = {t: vf.body_offsets(m, ref) for t, m in meas.items() if m is not None}
    meas_n = sum(m is not None for m in meas.values())

    def rng(k):
        x = [v[k] for v in off.values() if np.isfinite(v[k])]
        return [round(float(min(x)), 2), round(float(max(x)), 2)] if x else None
    rep = {"reference_frame": a + J.s0, "edges": kinds, "measured": meas_n, "moved": len(body),
           "reference": {"top_gap_px": round(ref["top"], 2), "bottom_gap_px": round(ref["bot"], 2),
                         "label_px": round(ref["L"], 2), "roll_deg": round(ref["roll"], 3)},
           "before": {"top_px": rng("top_px"), "bottom_px": rng("bottom_px"), "roll_deg": rng("roll_deg"),
                      "stretch": rng("stretch")},
           "stretch": [round(min(c[0] for c in corr.values()), 4), round(max(c[0] for c in corr.values()), 4)],
           "roll_deg": [round(min(c[2] for c in corr.values()), 3), round(max(c[2] for c in corr.values()), 3)]}
    print(f"body snap  {len(body)} of {len(frames)} frames placed by the body (reference frame {a + J.s0}, edges "
          f"{kinds['top']} above / {kinds['bottom']} below): before, the label's top {rep['before']['top_px']} px and "
          f"bottom {rep['before']['bottom_px']} px off the keyframe's proportion, roll {rep['before']['roll_deg']} deg; "
          f"stretch {rep['stretch']}, roll {rep['roll_deg']} deg")
    return body, rep


def _glue(J, C, fr, res):
    """render.glue (V42): the placement on every frame (_place on the frames the tracks hold, the fit's own mapping on
    the rest, as the job script read them), glued to the product's motion (imgvideo.glue). -> ({shot frame: glued
    mapping}, the report, {shot frame: the glued grid points}, which grid points are on the label)"""
    rc = C["rc"]
    gc = dict(gl.GLUE, **(rc["glue"] if isinstance(rc["glue"], dict) else {}))
    V = set(res["valid"])
    maps = {t: (_place(C, J, t)[0] if t in V else H) for t, H in enumerate(res["smooth"])}
    a = J.anchor - J.s0
    S, G, inl, P, rep, steps = gl.glue(maps, lambda t: fr[t], lambda t: J.key(t, maps[t]), J.outline(), a, gc)
    glued = {t: H for t, H in gl.glued_mappings(maps, P, S, a).items() if t in V}
    summ = gl.summary(rep, steps, sorted(maps), J.s0, J.anchor, gc["sigma"])
    summ["per_frame"] = {t + J.s0: v for t, v in rep.items()}
    e0, e1 = gl.edges(gc["edge"])
    print(f"glue       {len(glued)} frames moved with the product's own motion (sigma {gc['sigma']} frames, edge "
          f"{e0 if e0 == e1 else e0 + ' at the start, ' + e1 + ' at the end'}): correction median {summ['correction_px_median']} px, max {summ['correction_px_max']}; "
          f"placement jitter median {summ['placement_jitter_px_median']} px; motion fits fell back on "
          f"{summ['motion_fallbacks'] or 'no frames'}")
    return glued, summ, S, inl


def _level(J, C, fr, res):
    """render.level (V45): the label's lines level with the product's base, over the mapping the render places (glued,
    or as placed) on the frames the tracks hold (the fit's own mapping on the rest, as the job script read them).
    -> ({shot frame: levelled mapping}, the report)"""
    rc = C["rc"]
    V = set(res["valid"])
    maps = {t: (_place(C, J, t)[0] if t in V else H) for t, H in enumerate(res["smooth"])}
    anc = J.v["anchor"]
    kf = J.kf if anc.get("keyframe") and not anc.get("texture_job") else None
    lev, rep = lvl.level(maps, lambda t: J.key(t, maps[t]), J.outline(), J.anchor - J.s0, rc["level"], kf_img=kf,
                         polarity=rc.get("polarity", "light"))
    lev = {t: H for t, H in lev.items() if t in V}
    rep = dict(rep, keyframe=J.anchor, moved=len(lev))
    if "per_frame" in rep:
        rep["per_frame"] = {t + J.s0: v for t, v in rep["per_frame"].items()}
    if rep.get("skipped"):
        print(f"level      not levelled: {rep['skipped']}")
    else:
        kb = rep.get("keyframe_vs_base") or {}
        print(f"level      the label's lines levelled on {len(lev)} frames (target: the keyframe frame's level "
              f"{rep['keyframe_level_deg']:+.2f} deg, offset {rep['offset_deg']:+.2f}" +
              (f", the keyframe's own lines {kb['level_deg']:+.2f} deg against its base" if "level_deg" in kb else "") +
              (f"; {kb['skipped']}" if "skipped" in kb else "") +
              f"): correction median {rep['correction_deg_median']} deg, max {rep['correction_deg_max']}; the base "
              f"scatters {rep['base_scatter_deg']} deg frame to frame")
    return lev, rep


def _print_frames(J, C, fr, res, todo, pc, glue_pts):
    """render.alpha "print": the per-frame inputs of the presence gain and the motion blur. The label's motion is the
    centred difference of its grid points (glued, or as placed); the source's print sharpness is edge_strength
    against the keyframe frame's. -> ({clip frame: [vx, vy]}, {clip frame: sharpness} or None, report)"""
    rep = {}
    mot, ssharp = {}, None
    if float(pc["shutter"]) > 0:
        if glue_pts is not None:
            Sg, inl = glue_pts
            mot = {t + J.s0: v for t, v in pa.label_motion(Sg, inl).items()}
        else:
            G, inl = gl.grid(J.outline())
            V = set(res["valid"])
            P = {t: geo.apply_H(_place(C, J, t)[0] if t in V else H, G) for t, H in enumerate(res["smooth"])}
            mot = {t + J.s0: v for t, v in pa.label_motion(P, inl).items()}
        if pc["sharp_gate"]:
            a = J.anchor - J.s0
            k = pa.edge_strength(fr[a], J.key(a, res["smooth"][a])) if a in set(res["valid"]) else None
            if k is None:
                rep["blur_off"] = "the keyframe frame's print sharpness can't be measured: no motion blur"
                mot = {}
            else:
                ssharp = {}
                for t in todo:
                    m = J.key(t, res["smooth"][t])
                    e = pa.edge_strength(fr[t], m) if m.any() else None
                    if e is not None:
                        ssharp[t + J.s0] = round(e / k, 3)
    return mot, ssharp, rep


_RW = {}


def _render_worker_init(path, rc, optics, shutter, ax_ref):
    """A render worker: the job and its render context, rebuilt from the vjob (frames and mattes come per task)."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        J = Job(path) if isinstance(path, str) else Job(path["job"], media=_InfoMedia(path["info"]),
                                                        keyframes=path["keyframes"])
    J._masks = {}
    _RW.update(J=J, C=_render_context(J, rc, optics, shutter, ax_ref))


def _render_worker(task):
    t, frame, matte, *extras = task
    J = _RW["J"]
    J._masks = {t: matte}                  # J.key(t, ...) reads the frame's matte from here
    return _render_one(_RW["C"], J, t, frame, extras[0] if extras else None)


def render_workers(n=None):
    """How many processes render frames: --workers, else PLR_WORKERS, else the cores less two (at most 8). Every frame
    renders from its own inputs and seed, so the output is the same bit for bit with any number."""
    n = n if n is not None else int(os.environ.get("PLR_WORKERS", "0") or 0)
    return n if n > 0 else max(1, min(8, (os.cpu_count() or 2) - 2))


def stage_render(J, rng=None, workers=None):
    fr = J.frames()
    res = _fit(J)
    V = set(res["valid"])
    own = set(J.owned(res))
    art, _ = J.artwork()
    rc = dict(J.v.get("render", {}))
    if rc.get("_defaults"):
        print(f"defaults   on by default, not in the job's settings: {defaults_text(rc['_defaults'])}")
    optics = json.loads((J.work / "optics.json").read_text()) if (J.work / "optics.json").exists() else {}
    optics = render_optics(optics, rc)                         # render.soft_cap, render.smear (V40)
    if rc.get("blur", "auto") == "auto":
        rc["blur"] = optics.get("blur", 0.6)
    shutter = rc.get("shutter", "auto")
    shutter = 0.0 if shutter == "auto" else float(shutter)
    rc.update(_white(J, fr))
    if rc.get("ink_gain", "auto") == "auto":
        rc["ink_gain"] = optics.get("ink_gain", [1.0, 1.0, 1.0])
    lo, hi = (0, len(fr)) if rng is None else (rng[0] - J.s0, rng[1] - J.s0)
    test = rng is not None
    # one fixed close-up box over the product's reach in the shot, for the side-by-side
    P = np.vstack([geo.map_polygon(art.local(res["smooth"][t]), art.polygon) for t in sorted(V)])
    cx0, cy0 = P.min(0) - 70
    cx1, cy1 = P.max(0) + 50
    cy0 -= 0.2 * (cy1 - cy0)                                   # room for the pump / cap above the label
    box = [int(max(0, cx0)) // 2 * 2, int(max(0, cy0)) // 2 * 2, int(min(J.info["w"], cx1)) // 2 * 2, int(min(J.info["h"], cy1)) // 2 * 2]
    scale = min(1300.0 / (box[3] - box[1]), 1.5)
    if test:
        out_p, sbs_p, ll_p = (J.work / f"test_{rng[0]}-{rng[1]}_refined.mp4", J.work / f"test_{rng[0]}-{rng[1]}_side-by-side.mp4",
                              J.work / f"test_{rng[0]}-{rng[1]}_lossless.mov" if J.lossless else None)
        start = rng[0]
    else:
        out_p, sbs_p, ll_p, start = J.out, J.sbs, J.lossless, 0
    sinfo = dict(J.info)
    sw, sh = 2 * int(round((box[2] - box[0]) * scale)) + 8, int(round((box[3] - box[1]) * scale)) + 44
    sinfo.update(w=sw + sw % 2, h=sh + sh % 2)
    if J.sink is None:
        W = vio.Writer(out_p, J.info, crf=rc.get("crf", 12), audio_from=None if test else J.clip, start=start)
        Wl = vio.Writer(ll_p, J.info, audio_from=None if test else J.clip, start=start, lossless=True) if ll_p else None
        Ws = vio.Writer(sbs_p, sinfo, crf=18)
    else:                                    # in memory: the refined frames go into J.sink, nothing is written
        W = Wl = Ws = None
    infos, t0 = {}, time.time()
    # ease the fix in and out over a few frames where the held shot starts or ends with the product still in view
    n_in, n_out = rc.get("ease", [2, 2])
    Vs = sorted(V)
    ease = {}
    if Vs[0] > 0:                        # inside the shot range: the tracks were lost, the label is still in view
        for i in range(n_in):
            ease[Vs[0] + i] = min(ease.get(Vs[0] + i, 1.0), (i + 1) / (n_in + 1))
    if Vs[-1] < len(fr) - 1:             # (a shot range's own ends are cuts: the next shot has its own fix)
        for i in range(n_out):
            ease[Vs[-1] - i] = min(ease.get(Vs[-1] - i, 1.0), (i + 1) / (n_out + 1))

    def emit(src, out, t, shot):
        if J.sink is not None:
            if shot:
                J.sink[t] = out
            return
        W.write(out)
        if Wl:
            Wl.write(out)
        s = _sbs_frame(src, out, box, scale, t, shot)
        Ws.write(np.pad(s, ((0, sinfo["h"] - s.shape[0]), (0, sinfo["w"] - s.shape[1]), (0, 0)), constant_values=252))

    if not test:                                               # frames before the range considered: as they are
        for t, f in enumerate(J.media.frames(0, J.s0) if J.s0 > 0 else []):
            emit(f, f, t, False)
    snapped = {}
    Pv = J.outline()[:, 1]
    vmid = 0.5 * (Pv.min() + Pv.max())
    morph = set(res.get("morph", []))
    ax_mode = rc["limb_snap"].get("axial") if isinstance(rc.get("limb_snap"), dict) else None
    ax_ref = None
    if ax_mode and morph:
        # the product's ends along the axis on the frames whose own tracks hold: where the reconstructed (morph)
        # frames put them too
        E = [vf.axis_ends(res["smooth"][t], J.key(t, res["smooth"][t]), vmid) for t in sorted(V) if t not in morph]
        E = np.array([e for e in E if e is not None])
        if len(E) >= 3:
            ax_ref = np.median(E, 0)

    C = _render_context(J, rc, optics, shutter, ax_ref, res=res, art=art)
    body_rep = None
    if rc.get("body_snap"):
        # measured over every frame the tracks hold (not only a test range or a patch job's frames), so a range renders
        # as the full shot does
        body, body_rep = _body_snap(J, C, fr, V)
        rc["_body"] = C["body"] = body                         # (the workers rebuild their context from rc)
    glue_rep = glue_pts = None
    if rc.get("glue"):
        # the label glued to the product's own motion (V42), over the placement the body snap and limb snap made
        glued, glue_rep, Sg, inl = _glue(J, C, fr, res)
        rc["_glue"] = C["glue"] = glued
        glue_pts = (Sg, inl)
    level_rep = None
    if rc.get("level"):
        # the label's lines level with the product's base (V45), on top of the glue (or the placement)
        lev, level_rep = _level(J, C, fr, res)
        if lev:
            rc["_level"] = C["level"] = lev
            glue_pts = None                                    # the label's motion: through the levelled mappings
    print_rep = None
    if rc.get("alpha", "label") == "print":
        # print-only compositing (V40): the keyframe's ink read once, here, after the body snap placed the keyframe frame
        pc, pts, print_rep = _print_setup(J, C, fr, V)
        rc["_print"] = C["print"] = {"c": pc, "pts": pts}
        if rc.get("_sparkles_asked") is not None:
            print(f"print      the sparkle re-add asked for (render.sparkles {rc['_sparkles_asked']}) stays off with "
                  f"render.alpha \"print\": it ghosts the garbled print (V37); the source's glitter stays off the letters")
    elif rc.get("alpha", "label") != "label":
        raise ValueError(f"render.alpha {rc['alpha']!r}: \"label\" (the whole label band) or \"print\" (only the print)")
    todo = [t for t in range(lo, hi) if t in own]
    mot, ssharp, gains = {}, None, {}
    if print_rep is not None:
        # the presence gain and the motion blur's inputs (V43, V44)
        mot, ssharp, more = _print_frames(J, C, fr, res, todo, pc, glue_pts)
        print_rep.update(more)
    nw = render_workers(workers)
    pool = None
    if nw > 1 and len(todo) >= 3 and J.has_masks:
        # frames render in parallel processes, in order on the way out
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=min(nw, len(todo)), mp_context=mp.get_context("spawn"),
                                   initializer=_render_worker_init, initargs=(J.spec(), rc, optics, shutter, ax_ref))
        mk_all = J.masks()

    def frames_out(ts, extras_for):
        """(t, _render_one's result) for ts, in order: in the render processes (2 queued per process) or here."""
        if pool is None:
            for t in ts:
                yield t, _render_one(C, J, t, fr[t], extras_for(t))
            return
        pending, queue = {}, iter(ts)

        def submit():
            t = next(queue, None)
            if t is not None:
                pending[t] = pool.submit(_render_worker, (t, fr[t], mk_all[t], extras_for(t)))
        for _ in range(2 * nw):
            submit()
        for t in ts:
            r = pending.pop(t).result()
            submit()
            yield t, r

    if print_rep is not None and pc["presence"] == "match":
        # the print's presence, source over render, on every frame first (a render pass of its own), then smoothed
        raw = {t + J.s0: v for t, v in frames_out(todo, lambda t: {"measure": True}) if v is not None}
        gains = pa.presence_gains(raw, [t + J.s0 for t in todo], pc)
        print_rep["presence_measured"] = len(raw)

    def extras_for(t):
        if print_rep is None:
            return None
        T_ = t + J.s0
        bl = None
        if float(pc["shutter"]) > 0 and T_ in mot:
            bl = pa.motion_blur(mot[T_], ssharp.get(T_) if ssharp is not None else False, pc)
        return {"gain": gains.get(T_, 1.0), "blur": bl}
    rendered = frames_out(todo, extras_for)

    def one(t):
        t_, r = next(rendered)
        assert t_ == t
        return r

    for t in range(lo, hi):
        if t not in own:                                       # outside the shot the tracks hold: as it is
            emit(fr[t], fr[t], t + J.s0, False)
            continue
        out, info, span, n_hs, placed = one(t)
        if placed is not None:
            snapped[t + J.s0] = placed
        f = ease.get(t, 1.0)
        if f < 1.0:                    # easing in or out where the shot the tracks hold ends with the label in view
            ch = np.any(out != fr[t], axis=-1)
            mix = np.clip(fr[t] * (1.0 - f) + out.astype(float) * f + 0.5, 0, 255).astype(np.uint8)
            out = np.where(ch[..., None], mix, fr[t])
            info["ease"] = round(f, 3)
        info.update(motion_samples=n_hs, motion_px=round(span, 2))
        infos[t + J.s0] = info
        emit(fr[t], out, t + J.s0, True)
    if not test and J.s1 < J.info["frames"]:                     # frames after: as they are
        for t, f in enumerate(J.media.frames(J.s1, J.info["frames"])):
            emit(f, f, J.s1 + t, False)
    if pool is not None:
        pool.shutdown()
    if W:
        W.close()
        Ws.close()
    if Wl:
        Wl.close()
    unseen = {t: i["unseen_px"] / max(i["label_px"], 1) for t, i in infos.items() if "unseen_px" in i}
    if not test:                             # the mappings that rendered each frame, for QA to read the output through
        pickle.dump(snapped, open(J.work / "render_maps.pkl", "wb"))
    a = J.anchor - J.s0
    anchor_info = None
    if J.keyframe and a not in own and a in V and not test:
        # a patch job doesn't deliver the anchor frame, but QA measures the keyframe's own edge strip on it
        _, ai, _, _, _ = _render_one(C, J, a, fr[a])
        anchor_info = {k: ai[k] for k in ("unseen_px", "label_px") if k in ai}
    if print_rep is not None:
        pf = {t: i["print_alpha"] for t, i in infos.items() if "print_alpha" in i}
        print_rep["frames"] = len(pf)
        print_rep.update(presence=pc["presence"], presence_target=pc["presence_target"], shutter=pc["shutter"])
        gs = [v["presence_gain"] for v in pf.values()]
        if gs:
            print_rep["presence_gain"] = [round(min(gs), 3), round(float(np.median(gs)), 3), round(max(gs), 3)]
        bl = {t: v["blur_px"] for t, v in pf.items() if v.get("blur_px")}
        print_rep["blur_frames"] = sorted(bl)
        if bl:
            print_rep["blur_px_max"] = max(bl.values())
        if pf:
            print_rep["alpha_share_median"] = round(float(np.median([v["alpha_px"] / max(v["render_px"], 1)
                                                                     for v in pf.values()])), 3)
            lp = {t: v["lost_print_px"] for t, v in pf.items() if "lost_print_px" in v}
            if lp:
                # print the render has and the output lacks (keyframe ink only): over 30 px on a frame is worth a look.
                # Bright print near white reads lower in contrast on every frame (a frosted bottle's white print: about
                # half of the ink px, all letters there): read a frame against the shot's median share
                print_rep["lost_print_px_max"] = max(lp.values())
                print_rep["lost_print_frames"] = [t for t, v in lp.items() if v > 30]
                print_rep["lost_print_share_median"] = round(float(np.median(
                    [pf[t]["lost_print_px"] / max(pf[t]["ink_px"], 1) for t in lp])), 4)
    tune = {k: rc[k] for k in ("soft_cap", "smear") if k in rc}
    (J.work / ("render_test.json" if test else "render.json")).write_text(json.dumps(
        {"blur": rc["blur"], "shutter": shutter, "white_chroma": rc.get("white_chroma"), "ink_gain": rc.get("ink_gain"),
         **tune, "frames": infos, **({"anchor": anchor_info} if anchor_info else {}),
         **({"body_snap": body_rep} if body_rep else {}), **({"glue": glue_rep} if glue_rep else {}),
         **({"level": level_rep} if level_rep else {}), **({"print_alpha": print_rep} if print_rep else {}),
         **({"defaults": rc["_defaults"]} if rc.get("_defaults") else {})}, indent=0))
    n = len(infos)
    print(f"render     {n} frames refined, {hi - lo - n} left as they are, in {time.time() - t0:.0f} s on {nw if pool else 1} "
          f"process{'es' if pool and nw > 1 else ''} (blur {rc['blur']} px); "
          f"fill median {int(np.median([i['fill_px'] for i in infos.values()]))} px/frame, ink "
          f"{int(np.median([i['ink_px'] for i in infos.values()]))} px; sparkle px max "
          f"{max(i.get('spark_px', 0) for i in infos.values())}" +
          (f"; label the keyframe never saw: median {100 * np.median(list(unseen.values())):.2f} %, max "
           f"{100 * max(unseen.values()):.2f} % (frame {max(unseen, key=unseen.get)}), above 1 % on "
           f"{sum(v > 0.01 for v in unseen.values())} frames" if unseen else ""))
    if print_rep is not None:
        print(f"print      only the print changed on {print_rep['frames']} frames (ink: {print_rep['ink']}" +
              (f", {print_rep['ink_px']} keyframe ink px" if "ink_px" in print_rep else "") +
              (f"; {print_rep['ink_fallback']}" if "ink_fallback" in print_rep else "") + ")" +
              (f", {100 * print_rep['alpha_share_median']:.0f} % of the render's change (median)"
               if "alpha_share_median" in print_rep else "") +
              (f"; print lost against the render over 30 px on {_ranges(print_rep['lost_print_frames']) or 'no frames'} "
               f"(max {print_rep['lost_print_px_max']} px; median share of the ink "
               f"{100 * print_rep['lost_print_share_median']:.1f} %)" if "lost_print_px_max" in print_rep else "") +
              (f"; presence gain {print_rep['presence_gain']} (min, median, max)" if "presence_gain" in print_rep else "") +
              f"; motion blur on {_ranges(print_rep['blur_frames']) or 'no frames'}" +
              (f" (at most {print_rep['blur_px_max']} px)" if "blur_px_max" in print_rep else ""))
    if J.sink is None:
        print(f"saved      {out_p}" + (f"\n           {ll_p}" if ll_p else "") + f"\n           {sbs_p}")


def stage_qa(J):
    from . import qa
    res = _fit(J)
    V = J.owned(res)
    art, ref_u8 = J.artwork()
    qc = J.v.get("qa", {})
    rx0, ry0, rx1, ry1 = art.stats["crop"]
    ref_lin = im.srgb_to_lin(im.to_float(ref_u8[ry0:ry1, rx0:rx1]))

    def local(b):
        b = dict(b)
        if "centre" in b:
            b["centre"] = [b["centre"][0] - rx0, b["centre"][1] - ry0]
        else:
            b["box"] = [b["box"][0] - rx0, b["box"][1] - ry0, b["box"][2] - rx0, b["box"][3] - ry0]
        return b
    pol = J.v.get("render", {}).get("polarity", "light")
    sc = qa.Scorer(art, ref_lin, qc.get("step", 2.5), [local(b) for b in qc.get("text_boxes", [])], pol, qc.get("words"))
    s = np.load(J.work / "still.npz")
    g4 = ls.Grid(art, float(s["step"]))
    med = s["med"].astype(float)
    from scipy.ndimage import map_coordinates                  # the still label resampled onto the QA grid
    still = np.stack([map_coordinates(np.nan_to_num(med[..., c], nan=-1.0), [sc.grid.v / g4.step, sc.grid.u / g4.step],
                                      order=1, mode="nearest") for c in range(3)], -1)
    still[still[..., 0] < 0] = np.nan
    src = J.frames()
    out = J.output()
    rj = json.loads((J.work / "render.json").read_text())
    rinfo = rj["frames"]
    # the keyframe itself leaves its oblique edge strip (past theta_lim) as source: count only what goes beyond it
    ka = rinfo.get(str(J.anchor)) or rj.get("anchor", {})
    base_unseen = ka["unseen_px"] / max(ka["label_px"], 1) if "unseen_px" in ka else 0.0
    rows = []
    maps = pickle.load(open(J.work / "render_maps.pkl", "rb")) if (J.work / "render_maps.pkl").exists() else {}
    for t in V:
        H = art.local(res["smooth"][t])
        k = J.key(t, res["smooth"][t])
        a, _, _ = sc.score(src[t], H, k, still=still)
        Ho = art.local(maps[t + J.s0]) if t + J.s0 in maps else H
        b, rgb, _ = sc.score(out[t], Ho, k)
        e, ri = res["errors"][t], rinfo.get(str(t + J.s0), {})
        rows.append({"frame": t + J.s0, "ncc_src": a["ncc"], "ncc_out": b["ncc"], "lock": a.get("lock"),
                     "ocr_src": a.get("ocr"), "ocr_out": b.get("ocr"), "ocr_words_out": b.get("ocr_words"),
                     "vis_frac": b["vis_frac"], "rms_px": e["rms_px"], "p90_px": e["p90_px"],
                     "unseen_frac": round(max(ri["unseen_px"] / max(ri["label_px"], 1) - base_unseen, 0.0), 4)
                     if "unseen_px" in ri else None})
        if t == J.anchor - J.s0:
            Image.fromarray(im.to_u8(im.lin_to_srgb(np.nan_to_num(rgb)))).save(J.qa / "unwarp_anchor.png")
    ncc_min, rms_max, lock_min = qc.get("ncc_min", 0.6), qc.get("rms_max", 1.5), qc.get("lock_min", 0.3)
    unseen_max = qc.get("unseen_max", 0.02)
    soft = render_optics(json.loads((J.work / "optics.json").read_text()), J.v.get("render", {})).get("soft", {}) \
        if (J.work / "optics.json").exists() else {}
    nv = [r["ncc_out"] for r in rows if r["ncc_out"] is not None]
    ncc_floor = max(ncc_min, float(np.median(nv)) - qc.get("ncc_drop", 0.08)) if nv else ncc_min
    for r in rows:
        why = []
        if r["ncc_out"] is None or r["ncc_out"] < ncc_floor:
            sb = soft.get(str(r["frame"]))
            why.append(f"print match {r['ncc_out']}" + (f" (softened {sb} px to match the source's own softness here)" if sb else ""))
        if r["lock"] is not None and r["lock"] < lock_min:
            why.append(f"lock {r['lock']} (the source label morphs here, or the registration slipped)")
        if r["rms_px"] is not None and r["rms_px"] > rms_max:
            why.append(f"reprojection {r['rms_px']} px")
        if r["ocr_out"] == 0 and sc.ceiling:
            why.append("OCR read none of the copy")
        if r["unseen_frac"] is not None and r["unseen_frac"] > unseen_max:
            why.append(f"{100 * r['unseen_frac']:.1f} % of the label turned into view past what the keyframe shows (source kept there)")
        r["flag"] = "; ".join(why)
    # nothing may change outside the product's matte: count it, exactly on the lossless file
    outside = {}
    for name, path in (("refined_mp4", J.out), ("lossless_mov", J.lossless)) if J.sink is None else (("frames", None),):
        if J.sink is None and (path is None or not Path(path).exists()):
            continue
        o = out if name in ("refined_mp4", "frames") else vio.read(path, J.s0, J.s1, J.info)
        mk = J.masks() if J.has_masks else None
        ch, lv = 0, 0
        for t in range(len(src)):
            d = np.abs(o[t].astype(np.int16) - src[t].astype(np.int16)).max(-1)
            m = ~mk[t] if mk is not None else np.ones(d.shape, bool)
            ch += int((d[m] > 0).sum())
            lv = max(lv, int(d[m].max()) if m.any() else 0)
        outside[name] = {"px_changed": ch, "max_levels": lv}
    rep = {"truth_words": sc.truth, "ocr_ceiling": sc.ceiling, "valid": [int(t) + J.s0 for t in V],
           "keyframe_edge_strip": round(base_unseen, 4),
           "outside_matte": outside,
           "thresholds": {"ncc_min": round(ncc_floor, 3), "lock_min": lock_min, "rms_max": rms_max, "unseen_max": unseen_max},
           "impacts": [int(i) + J.s0 for i in res["impacts"]], "frames": rows}
    if J.lossless is not None and Path(J.lossless).exists() and _delivered(J.lossless):
        # the automatic review checks, on the delivered lossless file (they need its exact changed pixels); a shot job
        # that feeds a merge gets them in the merge, on the clip as delivered
        flags, summary = _review(J, [J], J.lossless)
        _apply_review(rows, rep, flags, summary)
        block = _write_review(flags, summary, Path(J.lossless), J.qa / "checks.json")
        print(block.rstrip())
    flagged = [r for r in rows if r["flag"]]
    (J.qa / "qa.json").write_text(json.dumps(rep, indent=0, ensure_ascii=False))
    plot_qa(rows, rep, J.plot)

    def med_(k):
        v = [r[k] for r in rows if r[k] is not None]
        return round(float(np.median(v)), 3) if v else None
    print(f"qa         {len(rows)} refined frames; print match vs truth median: source {med_('ncc_src')} -> refined "
          f"{med_('ncc_out')}; OCR word recall source {med_('ocr_src')} -> refined {med_('ocr_out')} (the truth itself at "
          f"this scale: {sc.ceiling}); lock {med_('lock')}; flagged {len(flagged)}" +
          (f": {', '.join(str(r['frame']) for r in flagged[:16])}" if flagged else "") +
          f"; outside the matte {outside}")
    print(f"saved      {J.plot}\n           {J.qa / 'qa.json'}")


def stage_merge(J):
    """Several shot jobs of one clip into one set of deliverables (J.v["merge"]: the other jobs' vjob.json paths).
    Each frame comes from the job whose shot holds it, as rendered losslessly; every other frame is the source."""
    parts = [J] + [Job(im.resolve(p, J.dir)) for p in J.v["merge"]]
    owner = {}
    for P in parts:                        # a frame two shots share (a chained anchor) stays with the first job
        for t in P.owned(_fit(P)):
            owner.setdefault(t + P.s0, P)
    dl = J.v["deliver"]
    out_p, ll_p, sbs_p, plot_p = (im.resolve(dl[k], J.dir) for k in ("refined", "lossless", "side_by_side", "qa_plot"))
    # one close-up box over every part's label
    boxes = []
    for P in parts:
        art, _ = P.artwork()
        res = _fit(P)
        Q = np.vstack([geo.map_polygon(art.local(res["smooth"][t]), art.polygon) for t in res["valid"]])
        boxes.append(np.r_[Q.min(0), Q.max(0)])
    B = np.array(boxes)
    cx0, cy0, cx1, cy1 = B[:, 0].min() - 70, B[:, 1].min() - 70, B[:, 2].max() + 50, B[:, 3].max() + 50
    cy0 -= 0.2 * (cy1 - cy0)
    box = [int(max(0, cx0)) // 2 * 2, int(max(0, cy0)) // 2 * 2, int(min(J.info["w"], cx1)) // 2 * 2, int(min(J.info["h"], cy1)) // 2 * 2]
    scale = min(1300.0 / (box[3] - box[1]), 1.5)
    sinfo = dict(J.info)
    sw, sh = 2 * int(round((box[2] - box[0]) * scale)) + 8, int(round((box[3] - box[1]) * scale)) + 44
    sinfo.update(w=sw + sw % 2, h=sh + sh % 2)
    W = vio.Writer(out_p, J.info, crf=J.v.get("render", {}).get("crf", 12), audio_from=J.clip)
    Wl = vio.Writer(ll_p, J.info, audio_from=J.clip, lossless=True)
    Ws = vio.Writer(sbs_p, sinfo, crf=18)
    src_all = vio.read(J.clip, 0, J.info["frames"], J.info)
    cache = {}
    for t in range(J.info["frames"]):
        P = owner.get(t)
        if P is None:
            out = src_all[t]
        else:
            if P.lossless not in cache:
                cache.clear()
                cache[P.lossless] = vio.read(P.lossless, 0, J.info["frames"], J.info)
            out = cache[P.lossless][t]
        W.write(out)
        Wl.write(out)
        s_ = _sbs_frame(src_all[t], out, box, scale, t, P is not None)
        Ws.write(np.pad(s_, ((0, sinfo["h"] - s_.shape[0]), (0, sinfo["w"] - s_.shape[1]), (0, 0)), constant_values=252))
    W.close()
    Wl.close()
    Ws.close()
    rows, impacts = [], []
    for P in parts:
        q = json.loads((P.qa / "qa.json").read_text())
        rows += [r for r in q["frames"] if owner.get(r["frame"]) is P]
        impacts += q["impacts"]
        if P is J:
            rep = q
    rows.sort(key=lambda r: r["frame"])
    rep = dict(rep, frames=rows, impacts=sorted(set(impacts)))
    for r in rows:                         # the parts' own review flags give way to the delivered clip's
        r["flag"] = r.get("flag_qa", r["flag"])
        r.pop("review", None)
    flags, summary = _review(J, parts, ll_p)
    _apply_review(rows, rep, flags, summary)
    block = _write_review(flags, summary, ll_p, plot_p.with_name(plot_p.stem + "_checks.json"))
    plot_qa(rows, rep, plot_p)
    # nothing may change outside the matte: the lossless file, every frame
    ll = vio.read(ll_p, 0, J.info["frames"], J.info)
    mk = vio.read(J.mask_path, 0, J.info["frames"])[..., 0] > 127 if J.mask_path else None
    d = np.abs(ll.astype(np.int16) - src_all.astype(np.int16)).max(-1)
    outside = int((d[~mk] > 0).sum()) if mk is not None else None
    shots = ", ".join(f"{min(t for t in owner if owner[t] is P)}-{max(t for t in owner if owner[t] is P)}" for P in parts)
    print(f"merge      {len(owner)} of {J.info['frames']} frames refined (shots {shots}), the rest as they are; "
          f"lossless vs source outside the matte: {outside} px changed; flagged {sum(bool(r['flag']) for r in rows)}")
    print(block.rstrip())
    print(f"saved      {ll_p}\n           {out_p}\n           {sbs_p}\n           {plot_p}")


def _review(J, parts, ll_path, arrays=None):
    """The automatic review checks (imgvideo.review) on a delivered lossless file: J's shot (one job) or the whole clip
    (a merge of parts, each frame read through the part that delivers it). -> (flags, summary)"""
    from . import review as rv
    # a delivery is checked over the whole clip (the fix stopping where the product is still in view counts there);
    # anything else only over the job's shot
    if arrays is not None:                   # in memory: (source, refined, mattes) over the whole clip
        lo, hi = 0, J.info["frames"]
        src, out, base_mk = arrays
    else:
        lo, hi = (0, J.info["frames"]) if len(parts) > 1 or _delivered(ll_path) else (J.s0, J.s1)
        src = vio.read(J.clip, lo, hi, J.info)
        out = vio.read(ll_path, lo, hi, J.info)
        base_mk = vio.read(J.mask_path, lo, hi)[..., 0] > 127 if J.mask_path else None
    owner = {}
    for P in parts:
        for t in P.owned(_fit(P)):
            owner.setdefault(t + P.s0, P)
    ctx = {}
    cf = dict(rv.DEFAULTS, **(J.v.get("review") or {}))
    looks_cfg = cf.get("looks", {})                  # the look checks (lookchecks): false switches them off
    LF = {}

    def context(P):
        if id(P) not in ctx:
            res = _fit(P)
            maps = pickle.load(open(P.work / "render_maps.pkl", "rb")) if (P.work / "render_maps.pkl").exists() else {}
            art, ref_u8 = P.artwork()
            st = np.load(P.work / "still.npz") if (P.work / "still.npz").exists() else None
            Ru = getattr(rv._base(res["smooth"][min(max(P.anchor - P.s0, 0), len(res["smooth"]) - 1)]), "Ru", 100.0)
            gh = rv.Ghost(art, ref_u8, None if st is None else st["med"].astype(float), 4.0 if st is None else float(st["step"]),
                          P.v.get("render", {}).get("feather"), Ru)
            Pv = P.outline()[:, 1]
            (u0, _), (u1, _) = P.outline().min(0), P.outline().max(0)
            ctx[id(P)] = dict(res=res, maps=maps, art=art, ghost=gh, G=rv.label_grid(P.outline()), v=(Pv.min(), Pv.max()),
                              shot=(P.s0, P.s1, P.anchor), outline=P.outline(), uc=0.5 * (u0 + u1),
                              glitter=rv.Glitter(art, print_px=int(cf["glitter_print_px"]), tophat=int(cf["glitter_tophat"]),
                                                 levels=float(cf["glitter_levels"]), max_px=int(cf["glitter_px"]),
                                                 excess=float(cf["glitter_excess"])))
            ctx[id(P)]["body"] = _body_reference(P, ctx[id(P)], lambda T: src[T - lo] if lo <= T < hi else
                                                 P.media.frames(T, T + 1)[0])
            rj = json.loads((P.work / "render.json").read_text()) if (P.work / "render.json").exists() else {}
            ctx[id(P)]["rinfo"], ctx[id(P)]["rlevel"] = rj.get("frames", {}), rj.get("level")
        return ctx[id(P)]
    F, prev = {}, None
    for i in range(hi - lo):
        T = lo + i
        P = owner.get(T)
        if P is not None:
            c = context(P)
            t = T - P.s0
            H = c["res"]["smooth"][t]
            m = P.key(t, H)
        else:
            m = base_mk[i] if base_mk is not None else np.zeros(src.shape[1:3], bool)
        r = rv.frame_measures(src[i], out[i], m, prev)
        if P is not None and r["changed_px"] > 300:
            Ho = c["maps"].get(T, H)
            r.update(shot=c["shot"], gap=rv.limb_gap(Ho, m, c["v"]), shear=rv.shear_of(H), pts=rv.shape_points(H, c["G"]),
                     ghost=c["ghost"](out[i], src[i], m, Ho, c["res"].get("morph_source", {}).get(t, H)),
                     occl=rv.occluder_edge(Ho, H, m, c["v"], cf["bend_slope"], cf["bend_ends"], cf["bend_moved"]))
            c["glitter"].add(T, c["shot"], src[i], out[i], m, c["res"].get("morph_source", {}).get(t, H))
            b = c["body"]
            if b is not None:
                mb = vf.body_measure(Ho, src[i].astype(float).mean(-1), m.astype(float), c["v"], c["uc"], b["edges"]["top"],
                                     b["edges"]["bottom"], b["reach"], b["across"], b["inside"],
                                     ref=None if T == b["frame"] else b["ref"], window=b["window"])
                if mb is not None:
                    r["body"] = _body_row(mb, b["ref"])
            if looks_cfg is not False:
                # the look checks' per-frame measures (the user's notes from the first real run, V42-V45)
                pol = (P.v.get("render") or {}).get("polarity", "light")
                chg = np.any(out[i] != src[i], axis=-1)
                LF[T] = {"looks": lk.frame_looks(src[i], out[i], m, chg, pol),
                         "ghost_px": c["rinfo"].get(str(T), {}).get("ghost_px")}
                if i > 0 and owner.get(T - 1) is P and (T - 1) in LF:
                    lc = dict(lk.LOOKS, **(looks_cfg or {}))
                    LF[T - 1]["swim"] = lk.swim_pair(src[i - 1], src[i], out[i - 1], out[i], prev[1], m,
                                                     np.any(out[i - 1] != src[i - 1], axis=-1), lc["still_px"],
                                                     int(lc["swim_search"]))
            Q = owner.get(T - 1)
            if i > 0 and Q is not None and Q is not P and context(Q)["shot"] != c["shot"] and \
                    np.allclose(context(Q)["outline"], c["outline"]):
                # two shots meet: does the label move with the product across it (the old print, read through both
                # frames' tracked mappings, stays put in label space)
                Hq = context(Q)["res"]["smooth"][T - 1 - Q.s0]
                j = rv.label_jump(c["art"], Hq, H, src[i - 1], src[i], prev[1], m)
                if j:
                    r["jump_px"] = j[1]
        prev = (src[i], m)
        F[T] = r
    for c in ctx.values():                            # glitter: each frame against the frames around it
        for T, g in c["glitter"].finish().items():
            F[T]["glitter"] = g
    if os.environ.get("PLR_REVIEW_DUMP"):             # the raw measures, to re-tune the checks without re-measuring
        pickle.dump({"F": F, "fps": J.info["fps"], "cfg": J.v.get("review")}, open(os.environ["PLR_REVIEW_DUMP"], "wb"))
    flags, summary = rv.evaluate(F, J.info["fps"], J.v.get("review"))
    if LF:
        # the look checks over each part's frames, merged into the review's flags (with what to try next)
        shots = {}
        for P in parts:
            if id(P) not in ctx:
                continue
            c = ctx[id(P)]
            rl = c["rlevel"] or {}
            on = bool(rl) and not rl.get("skipped")
            kb = rl.get("keyframe_vs_base") or {}
            shots[f"{c['shot'][0]}-{c['shot'][1]}"] = {
                "frames": [T for T, Q in owner.items() if Q is P and T in LF], "keyframe": P.anchor,
                "glue": (P.v.get("render") or {}).get("glue"), "level_on": on,
                "level_target": "base" if on and "level_deg" in kb and not kb.get("skipped") else None}
        lflags, lsum = lk.evaluate(LF, shots, J.info["fps"], looks_cfg or None)
        for t, v in lflags.items():
            flags.setdefault(t, []).extend(v)
        by = {}
        for t, v in flags.items():
            for k, _ in v:
                by.setdefault(k, []).append(t)
        summary.update(checks={k: rv._fmt_runs(v) for k, v in sorted(by.items())}, flagged=len(flags), **lsum)
    if all((P.v.get("render") or {}).get("alpha") == "print" for P in parts):
        # print-only compositing keeps the source's body, its glitter included, everywhere but on the letters, and the
        # re-add stays off there (it ghosts the garbled print, V37): render.sparkles isn't the remedy
        for v in flags.values():
            for i, (k, txt) in enumerate(v):
                if k == "glitter":
                    v[i] = (k, txt.replace("render.sparkles on this frame", "specks on the new letters (the re-add is "
                                                                             "off under render.alpha print: look)"))
        summary.pop("sparkles_frames", None)
    return flags, summary


def _body_reference(P, c, frame_at):
    """The body check's reference (V39): the label against the body on the job's keyframe frame (render.body_snap's
    ref_frame), through the mapping that rendered it, and the edges that bound the body there. -> {"ref", "edges",
    "reach", "across"}, or None (not a tube, or no body end found)"""
    res = c["res"]
    bs = P.v.get("render", {}).get("body_snap")
    bc = dict(vf.BODY_SNAP, **(bs if isinstance(bs, dict) else {}))
    a = int(bc.get("ref_frame", P.anchor))
    from .review import _base
    if not P.s0 <= a < P.s1 or not hasattr(_base(res["smooth"][a - P.s0]), "Ru"):
        return None
    H = c["maps"].get(a, res["smooth"][a - P.s0])
    lum, mt = frame_at(a).astype(float).mean(-1), P.key(a - P.s0, H).astype(float)
    edges = {"top": bc["top"], "bottom": bc["bottom"]}
    if "auto" in edges.values():
        auto = vf.body_edges(H, lum, mt, c["v"], c["uc"], bc["reach"])
        edges = {k: auto[k] if v == "auto" else v for k, v in edges.items()}
    ref = vf.body_measure(H, lum, mt, c["v"], c["uc"], edges["top"], edges["bottom"], bc["reach"], bc["across"], bc["inside"])
    return None if ref is None else {"ref": ref, "frame": a, "edges": edges, "reach": bc["reach"], "across": bc["across"],
                                     "inside": bc["inside"], "window": bc["window"]}


def _body_row(m, ref):
    """One frame's label against the body, for the report: the gaps (px), and the offsets from the keyframe's
    proportion (px, deg, stretch)."""
    o = vf.body_offsets(m, ref)

    def r(x, n=2):
        return round(float(x), n) if np.isfinite(x) else None
    return {"top_gap_px": r(m["top"]), "bottom_gap_px": r(m["bot"]), "label_px": r(m["L"]), "top_px": r(o["top_px"]),
            "bottom_px": r(o["bottom_px"]), "roll_deg": r(o["roll_deg"], 3), "stretch": r(o["stretch"], 4)}


def _delivered(path):
    """A round's deliverable: {project}/5-reviews/v{N}/..."""
    import re
    rnd = Path(path).parent
    return rnd.parent.name == "5-reviews" and re.fullmatch(r"v\d+", rnd.name) is not None


def _apply_review(rows, rep, flags, summary):
    """The review flags into the per-frame QA rows (their reasons join the QA's own) and the report."""
    for r in rows:
        r["flag_qa"] = r["flag"]
        why = [w for _, w in flags.get(r["frame"], [])]
        if why:
            r["review"] = why
            r["flag"] = "; ".join([r["flag"]] + why if r["flag"] else why)
    rep["review"] = dict(summary, frames={str(t): [k for k, _ in v] for t, v in sorted(flags.items())})


def _write_review(flags, summary, ll_path, json_path):
    """checks.json beside the QA, and the "Automatic checks" section of the round's notes.md when the lossless file is
    a round's deliverable ({project}/5-reviews/v{N}/). -> the section's markdown"""
    from . import review as rv
    json_path.write_text(json.dumps({"summary": summary, "frames": {str(t): v for t, v in sorted(flags.items())}}, indent=0))
    block = rv.notes_block(flags, summary)
    if _delivered(ll_path):
        rv.write_notes(Path(ll_path).parent / "notes.md", block)
    return block


def stage_review(J, notes=False):
    """The automatic review checks on what a job (or a merge, with "deliver") already delivered, without rendering:
    checks.json beside its QA plot, the section printed; notes.md only with --notes."""
    if J.v.get("merge"):
        parts = [J] + [Job(im.resolve(p, J.dir)) for p in J.v["merge"]]
        ll, plot = im.resolve(J.v["deliver"]["lossless"], J.dir), im.resolve(J.v["deliver"]["qa_plot"], J.dir)
    else:
        parts, ll, plot = [J], J.lossless, J.plot
    if ll is None or not Path(ll).exists():
        raise SystemExit("review: no lossless deliverable to check (output.lossless / deliver.lossless)")
    from . import review as rv
    flags, summary = _review(J, parts, ll)
    jp = Path(plot).with_name(Path(plot).stem + "_checks.json")
    jp.write_text(json.dumps({"summary": summary, "frames": {str(t): v for t, v in sorted(flags.items())}}, indent=0))
    block = rv.notes_block(flags, summary)
    if notes:
        _write_review(flags, summary, ll, jp)
    print(block.rstrip())
    print(f"saved      {jp}")


def plot_qa(rows, rep, path):
    """The QA plot (PNG). -> True, or False when matplotlib isn't installed (a server may not have it): no plot."""
    try:
        import matplotlib
    except ImportError:
        print("qa plot   skipped: matplotlib isn't installed")
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid_c, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    s1, s2, s3, s4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    f = np.array([r["frame"] for r in rows])

    def col(k):
        return np.array([np.nan if r[k] is None else r[k] for r in rows], float)
    has_ocr = np.isfinite(col("ocr_out")).any()
    has_unseen = np.isfinite(col("unseen_frac")).any()
    flagged = f[[bool(r["flag"]) for r in rows]]
    panels = [("Print match: correlation with the truth (label space)", [("refined", "ncc_out", s1), ("source", "ncc_src", s2)],
               rep["thresholds"]["ncc_min"])]
    if has_ocr:
        panels.append(("OCR: share of the truth's words read back (dashed: the truth itself at this scale)",
                       [("refined", "ocr_out", s1), ("source", "ocr_src", s2)], rep["ocr_ceiling"]))
    panels += [("Registration: reprojection error of the tracks (px)", [("rms", "rms_px", s3), ("90th percentile", "p90_px", s4)],
                rep["thresholds"]["rms_max"]),
               ("Registration lock: source label vs its own still label", [("lock", "lock", s3)], rep["thresholds"]["lock_min"])]
    if has_unseen:
        panels.append(("Label turned into view past what the keyframe shows (share of the visible label; source kept)",
                       [("unseen", "unseen_frac", s3)], rep["thresholds"]["unseen_max"]))
    checks = (rep.get("review") or {}).get("frames") or {}
    n_ax = len(panels) + (1 if rep.get("review") else 0)
    fig, axs = plt.subplots(n_ax, 1, figsize=(11, 2.3 * n_ax + 0.6), sharex=True, facecolor=surf)
    axs = np.atleast_1d(axs)
    if rep.get("review"):
        # the automatic review checks: one row per check, a mark on each flagged frame (refined or not)
        ax = axs[-1]
        names = ["edges", "ghost", "shear", "bend", "vanish", "switch", "swim", "ends", "level", "sharp", "tilt", "spots"]
        ax.set_facecolor(surf)
        for i, k in enumerate(names):
            xs = [int(t) for t, v in checks.items() if k in v]
            ax.scatter(xs, [i] * len(xs), marker="|", s=160, color=(s2 if k in ("vanish", "switch") else s1), lw=2, zorder=3)
        ax.set_yticks(range(len(names)), names)
        ax.set_ylim(-0.6, len(names) - 0.4)
        if not checks:
            ax.text(0.5, 0.5, "no frame flagged", transform=ax.transAxes, ha="center", va="center", color=ink2, fontsize=9)
        ax.set_title("Automatic checks: flagged frames (refined or left as source)", loc="left", fontsize=10, color=ink)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid_c)
        ax.tick_params(colors=ink2, labelsize=8)
        ax.grid(axis="x", color=grid_c, lw=0.6)
    for ax, (title, series, thr) in zip(axs, panels):
        ax.set_facecolor(surf)
        for x in flagged:
            ax.axvspan(x - 0.5, x + 0.5, color="#f3d6d6", lw=0, zorder=0)
        for lab, k, c in series:
            ax.plot(f, col(k), color=c, lw=1.6, label=lab, zorder=3)
        if thr is not None:
            ax.axhline(thr, color=ink2, lw=1, ls=(0, (4, 3)), zorder=2)
            ax.text(f[-1] + 0.5, thr, " ceiling" if "OCR" in title else " threshold", color=ink2, fontsize=8, va="center")
        for x in rep["impacts"]:
            if f.min() <= x <= f.max():
                ax.axvline(x, ymin=0, ymax=0.04, color=ink2, lw=0.8)
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.grid(axis="y", color=grid_c, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid_c)
        ax.tick_params(colors=ink2, labelsize=8)
        if len(series) > 1:
            ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=len(series), labelcolor=ink)
    axs[-1].set_xlabel("frame (QA panels: refined frames only; ticks: impact frames, kept unsmoothed; shaded: flagged)", color=ink2,
                       fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=surf)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------------------------------------------------
# The contract the ComfyUI node and `video.py --refine` share: a clip, its product matte and corrected keyframes in;
# the refined clip, the matte to paste it through and a report out. The decisions a person made between video.py runs
# (shots, the keyframe's frame, the band, print polarity and stroke) come from imgauto (the autopilot's measurements,
# its stable API); every shot is then this module's job on arrays, merged in keyframe order.

def _half(x):
    """Half size by area: uint8 frames (rounded), or a matte (more than half of each 2 x 2 block). What imgauto's
    measurements were calibrated on (the autopilot reads its clips at half size)."""
    T, h, w = x.shape[:3]
    y = x[:, :h // 2 * 2, :w // 2 * 2].reshape(T, h // 2, 2, w // 2, 2, *x.shape[3:])
    if x.dtype == bool:
        return y.mean((2, 4)) > 0.5
    return (y.mean((2, 4)) + 0.5).astype(np.uint8)


def _merge(base, over):
    """over's keys into base, recursively for dicts."""
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _ranges(ts):
    """[3, 4, 5, 9] -> [[3, 6], [9, 10]] (half-open runs)"""
    out = []
    for t in sorted(ts):
        if out and t == out[-1][1]:
            out[-1][1] = t + 1
        else:
            out.append([t, t + 1])
    return out


def _digest(*parts):
    import hashlib
    h = hashlib.blake2b(digest_size=12)
    for p in parts:
        h.update(p.tobytes() if isinstance(p, np.ndarray) else json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _key_colours(rgb, m):
    """The product body's colour key (video.py's "key": hue range, saturation and value floors) from its matte on one
    frame, or None when the body is too grey to key by colour."""
    h, s, v = tk.hsv(rgb[m])
    if len(h) < 500 or np.median(s) < 0.15:
        return None
    w = s > np.percentile(s, 30)                 # the body's own colour (white or dark print left out)
    a = np.deg2rad(h[w])
    mean = float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))
    off = (h[w] - mean + 180.0) % 360.0 - 180.0
    lo, hi = np.percentile(off, [3, 97])
    return {"hue": [round(mean + lo - 6.0, 1), round(mean + hi + 6.0, 1)],
            "sat": round(float(max(0.08, 0.7 * np.percentile(s[w], 5))), 3),
            "val": round(float(max(0.08, 0.7 * np.percentile(v[w], 5))), 3)}


def _patch_matte(J, f, cfg):
    """Frames the tracks hold whose matte is (nearly) empty while the product is in view (a segmentation gap): the
    body's colour key there, placed by the fitted mapping (video.py's matte when there is no mask video). The key's
    colours come from settings "key", else from the matte on the keyframe's frame f. Patches J.media's mattes in
    place. -> ({clip frame: px}, the key used or the reason there is none)"""
    if cfg.get("matte_patch") is False:
        return {}, "off (settings matte_patch false)"
    res = _fit(J)
    M, F = J.media._m, J.media._f
    area = np.array([int(M[t + J.s0].sum()) for t in res["valid"]])
    ref = float(np.median(area[area > 0])) if (area > 0).any() else 0.0
    low = [t for t, a in zip(res["valid"], area) if a < 0.25 * ref]
    if not low:
        return {}, None
    k = cfg.get("key") or _key_colours(F[f], M[f])
    if k is None:
        return {}, "the body is too grey to key by colour: give the frames a matte (or settings key)"
    out = {}
    P = J.outline()
    for t in low:
        H = res["smooth"][t]
        frame = F[t + J.s0]
        region = binary_dilation(poly_mask(frame.shape, geo.map_polygon(H, P)), iterations=4)
        c = geo.apply_H(H, [P.mean(0)])[0]
        m = tk.product_key(frame, tuple(k["hue"]), k.get("sat", 0.27), k.get("val", 0.35), near=tuple(c), region=region)
        if m.sum() >= 0.25 * ref:            # a plausible body; anything less stays as segmented (and is reported)
            M[t + J.s0] = m
            out[t + J.s0] = int(m.sum())
    return out, k


def _staged(J, stamps, k_track, k_fit, still_key, f, cfg):
    """track and fit only when their inputs changed since the last run in this folder (the work/ caches are video.py's
    own), then the matte patch, then still (when its inputs, the patched mattes included, changed). -> patch"""
    old = json.loads(stamps.read_text()) if stamps.exists() else {}
    stale = False
    for name, fn, key in (("track", stage_track, k_track), ("fit", stage_fit, k_fit)):
        stale = stale or old.get(name) != key
        if stale:
            fn(J)
            old[name] = key
            stamps.write_text(json.dumps(old))
        else:
            print(f"{name:<10} unchanged since the last run: kept")
    patch = _patch_matte(J, f, cfg)
    k_still = _digest(k_fit, still_key, J.media._m[J.s0:J.s1])
    if stale or old.get("still") != k_still:
        stage_still(J)
        old["still"] = k_still
        stamps.write_text(json.dumps(old))
    else:
        print("still      unchanged since the last run: kept")
    return patch


SETTINGS_SECTIONS = ("track", "fit", "render", "qa", "review", "artwork", "key")
# render settings the still label doesn't depend on
RENDER_ONLY = ("body_snap", "sparkles", "spark_glow", "alpha", "print_alpha", "soft_cap", "smear", "_sparkles_asked",
               "glue", "level", "_defaults")


def refine(frames, masks, keyframes, settings=None, keyframe_frames=None, fps=24.0, work=None, workers=None):
    """frames     T x H x W x 3 uint8: the clip (full frames, or the crops a workflow made)
    masks      T x H x W bool: the product's matte, same count and size
    keyframes  [H x W x 3 uint8, ...]: corrected frames of the clip (the label fixed), one per shot to cover
    settings   video.py's vjob sections (track, fit, render, qa, review, artwork, key) merged over the measured ones,
               plus "anchor" (band, model, theta_lim: the same for every keyframe, or a list, one per keyframe),
               "shots" ([[a, b] or null per keyframe]) and "words" (the label's copy, for OCR in QA)
    keyframe_frames  [frame or None per keyframe]; None: matched outside the matte (video.py's anchor.frame "auto")
    work       folder for each shot's work/ caches; reused across runs (track, fit and still run again only when their
               inputs change); a temp folder when None
    workers    render processes (video.py --workers; 1 = in this process)
    -> {"frames": T x H x W x 3 uint8 (the input outside the changed pixels), "matte": T x H x W bool (the input matte,
        plus any patched frames), "changed": T x H x W bool, "report": dict, "qa_plot": path or None, "work": path}"""
    import tempfile

    from imgauto import coverage as acov
    from imgauto import shots as ash
    from imgauto import verify as aver
    from imgauto import vjobs as avj

    from . import __version__
    from .keyframe import find_keyframe
    t_start = time.time()
    cfg = dict(settings or {})
    frames = np.ascontiguousarray(frames)
    masks = np.array(masks, dtype=bool)                     # a copy: the matte patch writes into it
    media = ArrayMedia(frames, masks, fps)
    T = frames.shape[0]
    keyframes = [np.ascontiguousarray(k) for k in keyframes]
    for i, k in enumerate(keyframes):
        if k.shape != frames.shape[1:] or k.dtype != np.uint8:
            raise ValueError(f"keyframe {i} is {k.shape[1]} x {k.shape[0]} {k.dtype}, the frames are {frames.shape[2]} x "
                             f"{frames.shape[1]} uint8: map a cropped keyframe back onto its frame first")
    kfr = list(keyframe_frames) if keyframe_frames is not None else [None] * len(keyframes)
    if len(kfr) != len(keyframes):
        raise ValueError(f"keyframe_frames: {len(kfr)} given for {len(keyframes)} keyframes")
    root = Path(work) if work else Path(tempfile.mkdtemp(prefix="label-refine-"))
    root.mkdir(parents=True, exist_ok=True)

    F2, M2 = _half(frames), _half(masks)
    stats, bounds, sh, kinds, why = ash.analyse(F2, M2)
    cuts = [t + 1 for t, kd in enumerate(kinds) if kd == "cut"]
    print(f"shots      {len(sh)} (cuts at {cuts or 'none'}); fixable {sum(b - a for s in sh for a, b in s['trims'])} of {T} "
          f"frames")

    def per_keyframe(key, i):
        v = cfg.get(key)
        return v[i] if isinstance(v, list) and v and isinstance(v[0], (list, dict, type(None))) else v

    decisions, jobs, patched_all = [], [], {}
    words = [w.lower() for w in cfg.get("words", [])]
    for i, kf in enumerate(keyframes):
        d = {"keyframe": i}
        decisions.append(d)
        if kfr[i] is None:
            f, dd, d2 = find_keyframe(frames, masks, kf)
            d.update(frame=int(f), frame_from="matched outside the matte", match_outside_mask=dd, runner_up=d2)
        else:
            f = int(kfr[i])
            d.update(frame=f, frame_from="given")
        if not 0 <= f < T:
            raise ValueError(f"keyframe {i}: frame {f} is outside the clip (0-{T - 1})")
        m = masks[f]
        if m.sum() < 200:
            d["skipped"] = "the matte is (nearly) empty on the keyframe's frame"
            continue
        anc = dict(per_keyframe("anchor", i) or {})
        if anc.get("band"):
            band, binfo = [int(x) for x in anc["band"]], {"from": "settings"}
        else:
            band, binfo = aver.body_band(kf, m)
            binfo = dict(binfo, **{"from": "measured (product profile)"})
        if not band:
            d["skipped"] = "no label band found on the keyframe (give anchor.band in settings)"
            continue
        changed, outside = aver.edit_region(kf, frames[f], m)
        polarity = (cfg.get("render") or {}).get("polarity") or aver.print_polarity(kf, m, changed)
        stroke = aver.stroke_width(frames[f], m, changed) or 4
        boxes = []                           # QA's OCR boxes, in the unrolled keyframe's label px, when the copy is given
        tube = anc.get("model", "tube") == "tube"
        if words and tube and not (cfg.get("qa") or {}).get("text_boxes"):
            try:
                boxes = [{k: v for k, v in b.items() if k != "words"}
                         for b in aver.text_boxes(kf, m, band, words, theta_lim=float(anc.get("theta_lim", 75.0)))]
            except Exception as e:           # (tesseract missing, or no text read)
                d["text_boxes"] = f"not measured: {e}"
        sh_i = per_keyframe("shots", i)
        if sh_i:
            a, b = int(sh_i[0]), int(sh_i[1])
            shot_from = "settings"
        else:
            s = next((s for s in sh if s["shot"][0] <= f < s["shot"][1]), None)
            trims = (s or {}).get("trims") or [[f, f + 1]]
            a, b = min(trims[0][0], f), max(trims[-1][1], f + 1)
            shot_from = "shot analysis (the fixable frames around the keyframe, cut to cut)"
        ys, xs = np.nonzero(masks[a:b].any(0))
        crop = (int(xs.max() - xs.min()) + 80, int(ys.max() - ys.min()) + 80)
        name = f"keyframe-{i}-{_digest(kf)[:10]}"
        anchor = _merge(avj.keyframe_anchor(name, f, band), {k: v for k, v in anc.items() if k != "band"})
        vj = avj.build("(memory)", "(memory)", [a, b], anchor, words, crop, polarity=polarity, tophat=2 * stroke + 1,
                       body_width=binfo.get("body_width_px", 300),
                       holes=avj.has_holes([{**s, "holes": s.get("holes", 0)} for s in stats], range(a, b)),
                       ends_at_cut=b < T and kinds[b - 1] == "cut")
        if boxes:
            vj["qa"]["text_boxes"] = boxes
            d["text_boxes"] = len(boxes)
        if not tube:                          # a flat face: the tube's limb snap and edge feather don't apply
            for k in ("limb_snap", "feather"):
                vj["render"].pop(k, None)
            vj["fit"].pop("camera", None)
        vj = _merge(vj, {k: cfg[k] for k in SETTINGS_SECTIONS if k in cfg})
        # what the measured job switched on that the settings don't name (logged by the render, in the report)
        dflt = defaults_on(vj["render"], cfg.get("render"), {"*": "the measured job", **{
            k: "the measured job, round products" for k in ("body_snap", "glue", "level")}})
        if dflt:
            vj["render"]["_defaults"] = dflt
        # render.alpha "print": its defaults over the measured job's (feather.v_px 4 unless the settings give one)
        vj["render"] = print_defaults(vj["render"], cfg.get("render") or {})
        vj["about"] = "written by imgvideo.pipeline.refine (in memory)"
        d.update(shot=[a, b], shot_from=shot_from, band=band, band_info=binfo, polarity=polarity, stroke_px=stroke,
                 edit_outside_matte=round(outside, 4))
        jd = root / f"shot{i}_{a}-{b}"
        J = Job({"vjob": vj, "dir": str(jd)}, media=media, keyframes={name: kf})
        k_track = _digest(frames[a:b], masks[a:b], kf, vj["anchor"], vj["track"], vj["shot"])
        k_fit = _digest(k_track, vj["fit"])
        # (the still label doesn't read the render's per-frame placement and re-add settings: changing them re-renders only)
        r_still = {k_: v_ for k_, v_ in vj["render"].items() if k_ not in RENDER_ONLY}
        patched, key = _staged(J, jd / "stamps.json", k_track, k_fit, [r_still, vj.get("artwork"), cfg.get("key"),
                                                                       cfg.get("matte_patch")], f, cfg)
        if patched or isinstance(key, str):
            d["matte_patch"] = {"frames": [[int(a_), int(b_)] for a_, b_ in _ranges(patched)], "key": key}
        patched_all.update(patched)
        stage_render(J, workers=workers)
        stage_qa(J)
        res = _fit(J)
        d.update(held=[int(min(res["valid"])) + a, int(max(res["valid"])) + a], refined=len(J.sink))
        rj = json.loads((J.work / "render.json").read_text())
        if rj.get("body_snap"):
            d["body_snap"] = rj["body_snap"]
        if rj.get("print_alpha"):
            d["print_alpha"] = rj["print_alpha"]
        if rj.get("glue"):
            d["glue"] = {k: v for k, v in rj["glue"].items() if k != "per_frame"}
        if rj.get("level"):
            d["level"] = {k: v for k, v in rj["level"].items() if k != "per_frame"}
        if rj.get("defaults"):
            d["defaults"] = rj["defaults"]
        jobs.append(J)

    # merge: each frame from the first keyframe's shot that refined it; everything else is the input
    out = frames.copy()
    owner = {}
    for J in jobs:
        for t, fr in sorted(J.sink.items()):
            if t not in owner:
                owner[t] = J
                out[t] = fr
    changed = np.any(out != frames, axis=-1)
    outside = int((changed & ~masks).sum())
    rows, impacts, rep = [], [], None
    for J in jobs:
        q = json.loads((J.qa / "qa.json").read_text())
        rows += [r for r in q["frames"] if owner.get(r["frame"]) is J]
        impacts += q["impacts"]
        rep = rep or q
    plot = None
    review = geometry = None
    if jobs:
        rows.sort(key=lambda r: r["frame"])
        rep = dict(rep, frames=rows, impacts=sorted(set(impacts)))
        flags, summary = _review(jobs[0], jobs, None, arrays=(frames, out, masks))
        geometry = summary.pop("geometry", None)
        _apply_review(rows, rep, flags, summary)
        review = rep.get("review")
        if rows:
            plot = root / "label-qa.png"
            if not plot_qa(rows, rep, plot):
                plot = None

    # what is left, shot by shot: imgauto's coverage plan for each run still to fix (chain, re-acquire or a new keyframe,
    # with the reason and the frame to correct). Two kinds: frames left as they are (not_covered), and frames refined
    # where part of the label turned into view past what the keyframe saw, so the source is kept on that part
    # (partly_covered).
    covered = set(owner)
    unseen = set()
    for J in jobs:
        q = json.loads((J.qa / "qa.json").read_text())
        lim = q.get("thresholds", {}).get("unseen_max", 0.02)
        unseen |= {r["frame"] for r in q.get("frames", []) if (r.get("unseen_frac") or 0) > lim and owner.get(r["frame"]) is J}
    not_covered, partly_covered = [], []
    for s_ in sh:
        for u in acov.plan(s_, kinds, covered, unseen, stats, bounds, M2, T)["uncovered"]:
            sg = u.get("suggest") or {}
            entry = {"plan": u["plan"], "why": u["why"], "suggested_keyframe": sg.get("pick"),
                     "suggestion_score": sg.get("score")}
            a_, b_ = (int(x) for x in u["run"])
            for refined in (False, True):
                for r0, r1 in _ranges([t for t in range(a_, b_) if (t in covered) == refined]):
                    (partly_covered if refined else not_covered).append({"frames": [int(r0), int(r1)], **entry})
    not_covered.sort(key=lambda e: e["frames"])
    partly_covered.sort(key=lambda e: e["frames"])
    report = {
        "engine": {"imgvideo": __version__},
        "frames": T, "size": [frames.shape[2], frames.shape[1]], "fps": fps,
        "cuts": cuts,
        "keyframes": decisions,
        "covered": [[int(a), int(b)] for a, b in _ranges(covered)],
        "not_covered": not_covered,           # left as they are
        "partly_covered": partly_covered,     # refined; part of the label keeps the source (past what the keyframe saw)
        "needs_mask": [[int(a), int(b)] for a, b in _ranges([t for t, w in enumerate(why) if w == "no matte, product visible"])],
        "flagged": {str(r["frame"]): r["flag"] for r in rows if r.get("flag")},
        "review": review,
        # per refined frame, the label against the product's body (V39): the gaps to the edge above (collar seam,
        # cap) and to the body's end below, px; how far its top and bottom sit off the keyframe's proportion of the
        # body (px, + = the gap is larger) and its axis off the body's centreline (deg), raw and as the body check
        # reads them (a 5-frame median); its share of the body against the keyframe's (stretch)
        "geometry": geometry,
        "outside_matte_px": outside,
        "matte_patched": [[int(a), int(b)] for a, b in _ranges(patched_all)],
        "work": str(root),
        "seconds": round(time.time() - t_start, 1),
    }
    (root / "report.json").write_text(json.dumps(report, indent=1, default=str))
    def spans(es):
        return ", ".join(f"{e['frames'][0]}-{e['frames'][1] - 1}" for e in es) or "none"
    print(f"refine     {len(covered)} of {T} frames refined by {len(jobs)} keyframe(s); left as they are: "
          f"{spans(not_covered)}; refined with part of the label kept from the source (it turned past what the "
          f"keyframe saw): {spans(partly_covered)}; flagged {len(report['flagged'])}; {outside} px changed outside "
          f"the matte; {report['seconds']} s")
    return {"frames": out, "matte": masks, "changed": changed, "report": report, "qa_plot": plot, "work": root}
