"""Flow nodes for the one workflow that covers every case (workflows/product-label-refinement-all.*): one input for an
image or a clip, optional files, and the switches that run only the branch a job needs.

- Load Source takes an image or a video from ComfyUI's input folder and says which it is (is_video).
- Load Optional is a file that may be left at "(none)": a product mask (else SAM 3), a corrected keyframe (else the
  stills branch makes it). `given` says whether it was set.
- Switch passes one of two lazy inputs on, so the other branch never runs. Run If passes its input on or blocks
  everything downstream of it, output nodes included (a still never reaches the video nodes).
- Print Type reads the job: print on the body (ink light or dark) or paper, the two stills graphs.
- Pick Frame takes the keyframe's frame (and mask) out of the clip; Shot takes the keyframe frame's shot out of an
  edit (between the hard cuts round it), so SAM 3 and Refine work on that shot only, and Splice puts the refined shot
  back, so the output is the whole clip with only that shot changed; Paste-Back Mask is where a run changed pixels, so
  a straightened crop goes back into its frame only there (and not at all when nothing was straightened).
- Video Settings adds the product's copy to Refine (Video)'s settings JSON as `words`.
- Label Triage shows the product's print next to the packshot's, straightened and at the same size, for the agent's
  call: fix or fine. A fine product skips the fix (the workflow's "Fix this product" boolean).

Images and videos are decoded by ComfyUI's own loaders (Load Image, and Load Video's VideoFromFile), so the pixels are
the ones the separate templates get from the same files.
"""
import json
import math
import os

import torch

try:
    import folder_paths
except ImportError:                          # outside ComfyUI (tools/build_workflows.py)
    folder_paths = None

from .nodes import JOB, _stages

CATEGORY = "image/product-label-refinement/flow"
NONE = "(none)"
ANY = "*"
_UNSET = object()                            # an optional input that isn't connected (not passed at all)


def _input_files(kinds):
    if folder_paths is None:
        return []
    d = folder_paths.get_input_directory()
    files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
    return sorted(folder_paths.filter_files_content_types(files, kinds))


def _is_video(name):
    """By its extension; an annotated name ("refined_00001.mov [output]": a run's output, the next product's source)
    is read without its annotation."""
    return bool(folder_paths.filter_files_content_types([folder_paths.annotated_filepath(name)[0]], ["video"]))


def _load(name):
    """-> (frames IMAGE [T, H, W, 3], fps, audio or None, is_video). A video goes through VideoFromFile like Load Video
    and Get Video Components; an image through Load Image itself."""
    if _is_video(name):
        from comfy_api.latest import InputImpl
        c = InputImpl.VideoFromFile(folder_paths.get_annotated_filepath(name)).get_components()
        return c.images, float(c.frame_rate), c.audio, True
    import nodes as comfy_nodes              # ComfyUI's nodes.py
    image, _alpha = comfy_nodes.LoadImage().load_image(name)
    return image, 24.0, None, False


def _changed(name):
    if not name or name == NONE:
        return ""
    p = folder_paths.get_annotated_filepath(name)
    try:
        st = os.stat(p)
    except OSError:
        return float("nan")
    return f"{p}:{st.st_size}:{st.st_mtime_ns}"


def _valid(name, optional):
    if optional and name in ("", NONE):
        return True
    if not folder_paths.exists_annotated_filepath(name):
        return f"no such file in ComfyUI's input folder: {name}"
    return True


def _truthy(v):
    """True, a non-empty string (job text), a non-zero number, or anything given (an image, a job)."""
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (bool, int, float)):
        return bool(v)
    if isinstance(v, (list, tuple, dict)):
        return len(v) > 0
    return True


class ProductLabelLoadSource:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source": (_input_files(["image", "video"]), {"image_upload": True, "tooltip": (
            "The generated image or clip to fix, from ComfyUI's input folder (an agent uploads it with POST "
            "/upload/image, any file type, and gives its name, with the subfolder: plr/clip.mov). A video runs the "
            "video branch: SAM 3 through the clip, the stills fix on the keyframe's frame, then Refine (Video).")})}}

    RETURN_TYPES = ("IMAGE", "FLOAT", "AUDIO", "BOOLEAN", "INT")
    RETURN_NAMES = ("frames", "fps", "audio", "is_video", "frame_count")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = ("An image or a video from the input folder: frames (one for an image), fps and audio (a video's; 24 "
                   "and none for an image), and is_video, which picks the branch.")

    @classmethod
    def IS_CHANGED(cls, source):
        return _changed(source)

    @classmethod
    def VALIDATE_INPUTS(cls, source):
        return _valid(source, False)

    def load(self, source):
        frames, fps, audio, is_video = _load(source)
        return (frames, fps, audio, is_video, int(frames.shape[0]))


class ProductLabelLoadOptional:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"file": ([NONE] + _input_files(["image", "video"]), {"image_upload": True, "tooltip": (
            "An image or a video from the input folder, or (none). As a mask: white = product (red channel), one "
            "per frame for a clip.")})}}

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("image", "mask", "given")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = ("A file that may be left at (none): image (all frames), mask (their red channel, as Load Image (as "
                   "Mask) and Image To Mask read it) and given. Feed `given` to a Switch, so what the file replaces "
                   "runs only without it.")

    @classmethod
    def IS_CHANGED(cls, file):
        return _changed(file)

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        return _valid(file, True)

    def load(self, file):
        if file in ("", NONE):
            return (None, None, False)
        frames, _fps, _audio, _video = _load(file)
        return (frames, frames[..., 0].clone(), True)


class ProductLabelSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"when": (ANY, {"tooltip": "True, a non-empty string (a job's text), a non-zero number, "
                                                       "or anything given"})},
                "optional": {"if_true": (ANY, {"lazy": True}), "if_false": (ANY, {"lazy": True})}}

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "pick"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Passes if_true or if_false on. Only the one passed on is computed, so the other branch never runs; an "
                   "input left unconnected passes nothing (none).")

    def check_lazy_status(self, when, if_true=_UNSET, if_false=_UNSET):
        name, v = ("if_true", if_true) if _truthy(when) else ("if_false", if_false)
        return [name] if v is None else []

    def pick(self, when, if_true=None, if_false=None):
        return (if_true if _truthy(when) else if_false,)


class ProductLabelRunIf:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (ANY, {"lazy": True}),
                             "when": (ANY, {"tooltip": "True, a non-empty string, a non-zero number, or anything given"}),
                             "run_when": (["true", "false"], {"tooltip": "false: run when `when` is false or empty"})}}

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "gate"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Passes value on when `when` is as run_when says, else blocks every node downstream, output nodes "
                   "included, without an error. value isn't computed when blocked.")

    @staticmethod
    def _open(when, run_when):
        return _truthy(when) == (run_when == "true")

    def check_lazy_status(self, value, when, run_when):
        return ["value"] if self._open(when, run_when) and value is None else []

    def gate(self, value, when, run_when):
        if not self._open(when, run_when):
            from comfy_execution.graph_utils import ExecutionBlocker
            return (ExecutionBlocker(None),)
        return (value,)


class ProductLabelPrintType:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job": (JOB,)}}

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("light", "print")
    FUNCTION = "route"
    CATEGORY = CATEGORY
    DESCRIPTION = ("light (the output's first name; true for both polarities): the job lays print straight on the "
                   "body, lighter or darker than it (matte ink with polarity light or dark: Auto Draft's ink light or "
                   "dark). Its graph skips Grade To Scene and Match Finish and keeps the print inside the product's "
                   "mask. Otherwise paper (a label, or print on paper or a flat panel), the graded composite.")

    def route(self, job):
        m = job.get("matte")
        on_body = bool(_stages().ik.on_body(m))
        return (on_body, f"{m.get('polarity', 'light')} print on the body" if on_body else "paper")


class ProductLabelPickFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"index": ("INT", {"default": 0, "min": -1, "max": 1_000_000, "tooltip": (
                    "The frame, from 0. Clamped to the batch: an image is always its frame 0, -1 is frame 0.")})},
                "optional": {"images": ("IMAGE",), "masks": ("MASK",)}}

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("image", "mask", "index")
    FUNCTION = "pick"
    CATEGORY = CATEGORY
    DESCRIPTION = "One frame of a batch of images and/or masks, as a batch of 1, and the frame it took."

    def pick(self, index, images=None, masks=None):
        if masks is not None and masks.dim() == 2:
            masks = masks[None]
        n = (images if images is not None else masks).shape[0] if (images is not None or masks is not None) else 1
        k = min(max(int(index), 0), n - 1)
        return (None if images is None else images[k:k + 1], None if masks is None else masks[k:k + 1], k)



def find_cuts(frames, chunk=16):
    """-> the frames (from 1) that start a new shot: hard cuts, where the whole picture changes at once. Each frame is
    shrunk to 64x64; a cut is a frame whose thumbnail differs from the previous one by 0.06 or more on average (0-1),
    8x the clip's median frame-to-frame change, and whose colour histogram changes too (L1 0.3 or more of 2). On edits
    with hard cuts the cuts measured 0.15-0.25 and everything else under 0.03, fades and push-ins included."""
    T = frames.shape[0]
    if T < 3:
        return []
    th = []
    for i in range(0, T, chunk):
        x = frames[i:i + chunk, ..., :3].float().movedim(-1, 1)
        th.append(torch.nn.functional.interpolate(x, size=(64, 64), mode="area").clamp(0, 1).cpu())
    t = torch.cat(th)
    q = (t * 7.999).long()
    idx = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    h = torch.stack([torch.bincount(i.flatten(), minlength=512).float() for i in idx]) / (64 * 64)
    dh = (h[1:] - h[:-1]).abs().sum(1)
    dp = (t[1:] - t[:-1]).abs().mean((1, 2, 3))
    floor = max(0.06, 8 * float(dp.median()))
    return [int(i) + 1 for i in range(T - 1) if float(dp[i]) >= floor and float(dh[i]) >= 0.3]


class ProductLabelShot:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "frames": ("IMAGE", {"tooltip": "The whole clip (an edit with several shots is fine)"}),
                    "keyframe_index": ("INT", {"default": 0, "min": -1, "max": 1_000_000, "tooltip": (
                        "The keyframe frame in the whole clip (from 0). -1 (a given keyframe whose frame Refine finds): "
                        "the whole clip, unless start and end are set.")}),
                    "start": ("INT", {"default": -1, "min": -1, "max": 1_000_000, "tooltip": (
                        "The shot's first frame in the clip; -1: found, the cut before the keyframe frame (or 0)")}),
                    "end": ("INT", {"default": -1, "min": -1, "max": 1_000_000, "tooltip": (
                        "The shot's last frame in the clip; -1: found, the frame before the next cut (or the last)")})},
                "optional": {"masks": ("MASK", {"tooltip": "A given mask video, one per frame of the clip: cut to the "
                                                           "same shot"})}}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING")
    RETURN_NAMES = ("frames", "masks", "start", "keyframe_index", "shot")
    FUNCTION = "cut"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The keyframe frame's shot out of a clip, so SAM 3 and Refine (Video) work on that shot only: between "
                   "the hard cuts around the keyframe frame (found from the frames), or from start to end. "
                   "keyframe_index: the keyframe frame counted from the shot's start; start: where the shot goes back "
                   "(Splice); shot: JSON with start, end, the cuts found and the clip's length. Refine's report counts "
                   "frames from the shot's start: add start for the clip's frame. An image is its own shot.")

    def cut(self, frames, keyframe_index, start=-1, end=-1, masks=None):
        T, k = int(frames.shape[0]), int(keyframe_index)
        auto = (start < 0 or end < 0) and k >= 0
        cuts = find_cuts(frames) if auto else []
        if k >= T:
            raise ValueError(f"the keyframe frame {k} is past the clip's last frame ({T - 1})")
        s = start if start >= 0 else max([c for c in cuts if c <= k], default=0) if k >= 0 else 0
        e = end if end >= 0 else min([c - 1 for c in cuts if c > k], default=T - 1) if k >= 0 else T - 1
        s, e = min(s, T - 1), min(e, T - 1)
        if e < s:
            raise ValueError(f"the shot runs from frame {s} to {e}: end before start")
        if k >= 0 and not s <= k <= e:
            raise ValueError(f"the keyframe frame {k} is outside the shot ({s}-{e})")
        if masks is not None:
            masks = masks[None] if masks.dim() == 2 else masks
            if masks.shape[0] == T:
                masks = masks[s:e + 1]
        shot = {"start": s, "end": e, "frames": e - s + 1, "keyframe_in_shot": k - s if k >= 0 else -1,
                "clip_frames": T, "cuts": cuts, "found": {"start": start < 0, "end": end < 0} if auto else False}
        return (frames[s:e + 1], masks, s, k - s if k >= 0 else -1, json.dumps(shot))


def _sam3_detect(model, images, conditioning, threshold=0.33):
    """Core SAM3_Detect on a batch of frames (one union mask per frame), as the workflow's own detect node runs it."""
    from comfy_extras.nodes_sam3 import SAM3_Detect
    o = SAM3_Detect.execute(model=model, image=images, conditioning=conditioning, threshold=threshold,
                            refine_iterations=2, individual_masks=False)
    m = (getattr(o, "result", None) or getattr(o, "args", None) or o)[0]
    return m[None] if m.dim() == 2 else m


class ProductLabelFindKeyframe:
    # candidates per shot: these fractions of the way through it
    PROBES = (0.25, 0.5, 0.75)
    MIN_AREA = 0.002                         # a detection smaller than this share of the frame isn't the product
    TIE = 0.01                               # keypick scores this close to the best are a tie
    CLOSE = 0.7                              # another shot scoring this share of the best one's: a warning
    ALSO = 0.5                               # ... this share: reported as likely showing the product too (not fixed)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "frames": ("IMAGE", {"tooltip": "The whole clip (an image: its frame 0)"}),
                    "keyframe_frame": ("INT", {"default": -1, "min": -1, "max": 1_000_000, "tooltip": (
                        "The keyframe frame as set: 0 or more is kept; -1 is found (with a corrected keyframe, Refine "
                        "finds its frame)")}),
                    "keyframe_given": ("BOOLEAN", {"default": False, "tooltip": "A corrected keyframe is given"}),
                    "mask_given": ("BOOLEAN", {"default": False, "tooltip": "A product mask video is given"})},
                "optional": {
                    "masks": ("MASK", {"lazy": True, "tooltip": "The given mask video (one per frame of the clip)"}),
                    "model": ("MODEL", {"lazy": True, "tooltip": "SAM 3"}),
                    "conditioning": ("CONDITIONING", {"lazy": True, "tooltip": "SAM 3's text prompt"}),
                    "reference": ("IMAGE", {"lazy": True, "tooltip": "The packshot"}),
                    "reference_mask": ("MASK", {"lazy": True, "tooltip": "The packshot's outline (Load Image's "
                                                                         "convention: 1 = transparent)"})}}

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("keyframe_frame", "report")
    FUNCTION = "find"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "The keyframe frame: as set (0 or more), or found (-1). Found: the clip's shots (between hard cuts); in each, "
        "SAM 3 on three frames; the shot whose product matches the packshot best (silhouette, print layout, colour, "
        "size on screen); in that shot, SAM 3's track and the frame where the product shows largest, sharpest and "
        "most frontal, off morph frames and the shot's ends (the engine's keypick). With a corrected keyframe, -1 "
        "passes on (Refine finds its frame); an image is frame 0. report: JSON with every shot's score, the pick and "
        "its runner-up; a margin under 5 % or a weak match is flagged.")

    def _auto(self, frames, keyframe_frame, keyframe_given):
        return keyframe_frame < 0 and not keyframe_given and frames.shape[0] > 1

    def check_lazy_status(self, frames, keyframe_frame, keyframe_given, mask_given, masks=_UNSET, model=_UNSET,
                          conditioning=_UNSET, reference=_UNSET, reference_mask=_UNSET):
        if not self._auto(frames, keyframe_frame, keyframe_given):
            return []
        need = ["reference", "reference_mask"] + (["masks"] if mask_given else ["model", "conditioning"])
        got = {"masks": masks, "model": model, "conditioning": conditioning, "reference": reference,
               "reference_mask": reference_mask}
        return [k for k in need if got[k] is None]

    def find(self, frames, keyframe_frame, keyframe_given, mask_given, masks=None, model=None, conditioning=None,
             reference=None, reference_mask=None):
        if not self._auto(frames, keyframe_frame, keyframe_given):
            k = int(keyframe_frame) if frames.shape[0] > 1 or keyframe_frame < 0 else 0
            why = ("set" if keyframe_frame >= 0 else "the corrected keyframe's (Refine finds it)") \
                if frames.shape[0] > 1 else "an image"
            rep = json.dumps({"keyframe_frame": k, "from": why})
            return {"ui": {"text": [rep]}, "result": (k, rep)}
        import numpy as np
        from .video_nodes import _engine, _u8
        _engine()
        from imgauto import keypick, refangle, shots as ash
        T = int(frames.shape[0])
        cuts = find_cuts(frames)
        bounds = list(zip([0] + cuts, [c - 1 for c in cuts] + [T - 1]))
        ref = _u8(reference[:1])[0]
        rm = reference_mask[None] if reference_mask.dim() == 2 else reference_mask
        Rm = rm[0].detach().cpu().numpy() < 0.5 if rm.shape[1:3] == reference.shape[1:3] and bool((rm > 0.5).any()) \
            else _reference_matte(ref, None)
        if masks is not None:
            masks = masks[None] if masks.dim() == 2 else masks
        probes = sorted({min(e, max(s, round(s + (e - s) * q))) for s, e in bounds for q in self.PROBES})
        idx = torch.tensor(probes)
        pm = masks[idx] if mask_given else _sam3_detect(model, frames[idx], conditioning)
        shots_rep = []
        for s, e in bounds:
            cands = []
            for j, t in enumerate(probes):
                if not s <= t <= e:
                    continue
                M = pm[j].detach().cpu().numpy() > 0.5
                fr = _u8(frames[t:t + 1])[0]
                cands.append(dict(t=t, **_match_packshot(fr, M, ref, Rm, refangle, self.MIN_AREA)))
            best = max(cands, key=lambda c: c["score"])
            shots_rep.append({"shot": [s, e], "score": best["score"], "best": best["t"], "probes": cands})
        ranked = sorted(shots_rep, key=lambda r: -r["score"])
        top = ranked[0]
        if top["score"] <= 0:
            raise ValueError("no shot shows the product (SAM 3 found nothing like the packshot on any shot's probes): "
                             "check the SAM 3 prompt, or set the keyframe frame")
        s, e = top["shot"]
        # the shot's masks: the given ones, else SAM 3 tracked through it from its best probe
        if mask_given:
            sm = masks[s:e + 1]
        else:
            from .video_nodes import ProductLabelVideoSAM3BothWays
            kt = top["best"] - s
            km = _sam3_detect(model, frames[top["best"]:top["best"] + 1], conditioning)
            sm = ProductLabelVideoSAM3BothWays().track(frames[s:e + 1], model, km, kt, conditioning=conditioning,
                                                       detection_threshold=0.33, max_objects=1, detect_interval=1)[0]
        F = _u8(frames[s:e + 1])
        M = sm.detach().cpu().numpy() > 0.5
        n = e - s + 1
        F2, M2 = ash.downscale(F, 2), ash.downscale(M, 2)
        stats, bnds, _sh, _kinds, _why = ash.analyse(F2, M2)
        # the shot's ends are cuts (penalised as edges) unless they are the clip's own first or last frame
        ranking = keypick.score(stats, bnds, range(n), open_ends=[t for t, c in ((0, s == 0), (n - 1, e == T - 1)) if c])
        top_s = ranking[0]["score"]
        # frames keypick can't tell apart (within TIE of the best): the one nearest the shot's middle, so the label's
        # track runs the shortest way to either end (the track's error grows with the distance from the keyframe)
        near = [r for r in ranking if r["score"] >= (1.0 - self.TIE) * top_s]
        best = min(near, key=lambda r: (abs(r["t"] - (n - 1) / 2), -r["score"]))
        k = s + int(best["t"])
        warn = []
        if best["flags"]:
            warn.append(f"the pick is flagged {', '.join(best['flags'])} (keypick): check the keyframe frame")
        close = [r for r in ranked[1:] if r["score"] >= self.CLOSE * top["score"]]
        for r in close:
            warn.append(f"shot {r['shot'][0]}-{r['shot'][1]} matches the packshot nearly as well ({r['score']:.3f} vs "
                        f"{top['score']:.3f}): if it shows another product of the same kind, name what tells this one "
                        "apart in the SAM 3 prompt (its colour) or set the keyframe frame; if it shows this product "
                        "too, run again with the keyframe frame set there")
        also = [r["shot"] for r in ranked[1:] if r["score"] >= self.ALSO * top["score"]]
        rep = {"keyframe_frame": k, "from": "found", "shot": [s, e], "also_likely": also,
               "pick": {**best, "t": k, "best_scored": s + int(ranking[0]["t"]),
                        "tied": [s + int(r["t"]) for r in sorted(near, key=lambda r: r["t"])]},
               "shots": shots_rep, "warnings": warn}
        text = json.dumps(rep, default=float)
        print(f"keyframe   frame {k} (shot {s}-{e}, score {top['score']:.2f}; " +
              ", ".join(f"{r['shot'][0]}-{r['shot'][1]} {r['score']:.2f}" for r in shots_rep) + ")" +
              "".join(f"\nkeyframe   WARN {w}" for w in warn))
        return {"ui": {"text": [text]}, "result": (k, text)}


def _ab_hist(rgb, m, bins=16, reach=48.0):
    """The colours of the masked pixels as a smoothed CIELAB a*b* histogram (lightness left out: the scene's light
    and the packshot's differ; the darkest and brightest pixels too: print and highlights carry no hue)."""
    import numpy as np
    from scipy import ndimage as ndi
    x = rgb[m].astype(np.float64) / 255.0
    x = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    xyz = x @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    f = xyz / np.array([0.9505, 1.0, 1.089])
    f = np.where(f > 0.008856, np.cbrt(f), 7.787 * f + 16 / 116)
    L, a, b = 116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])
    k = (L > 15) & (L < 97)
    h, _, _ = np.histogram2d(a[k], b[k], bins=bins, range=[[-reach, reach], [-reach, reach]])
    h = ndi.gaussian_filter(h, 1.0)
    return h / max(h.sum(), 1e-9)


def _match_packshot(frame, M, ref, Rm, refangle, min_area, max_area=0.6):
    """How well SAM 3's product on a frame matches the packshot's: -> {"area", "shape", "colour", "score"}.
    shape: the silhouettes' overlap after their centroid, axis and length are matched (refangle); colour: the
    Bhattacharyya overlap of the two products' a*b* colour histograms; score: shape x colour x the square root of the
    product's share of the frame (bigger on screen is better to fix from). The print's layout isn't scored: seen from
    above or with the generator's own print, it reads low on the right product. A detection over max_area of the
    frame is the scene, not a product."""
    import numpy as np
    area = float(M.mean())
    if not min_area <= area <= max_area:
        return {"area": round(area, 4), "shape": 0.0, "colour": 0.0, "score": 0.0}
    iou = refangle.shape_iou(Rm, M)
    col = float(np.sqrt(_ab_hist(frame, M) * _ab_hist(ref, Rm)).sum())
    score = iou * col * np.sqrt(area)
    return {"area": round(area, 4), "shape": round(float(iou), 3), "colour": round(col, 3), "score": round(float(score), 4)}


class ProductLabelSplice:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames": ("IMAGE", {"tooltip": "The whole clip"}),
                             "shot_frames": ("IMAGE", {"tooltip": "The shot, refined (Refine (Video)'s frames)"}),
                             "start": ("INT", {"default": 0, "min": 0, "max": 1_000_000, "tooltip": "Shot's start"})},
                "optional": {"shot": ("STRING", {"forceInput": True, "tooltip": "Shot's JSON, passed on"})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "shot")
    FUNCTION = "splice"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("The refined shot back into the whole clip at start: the clip with only that shot changed, to save "
                   "and to run the next product on. shot: the Shot's JSON, shown, once the shot is refined.")

    def splice(self, frames, shot_frames, start, shot=""):
        n, T = int(shot_frames.shape[0]), int(frames.shape[0])
        if start + n > T or shot_frames.shape[1:3] != frames.shape[1:3]:
            raise ValueError(f"a {n}-frame shot at {start} doesn't fit the {T}-frame clip "
                             f"({tuple(shot_frames.shape[1:3])} vs {tuple(frames.shape[1:3])})")
        if start == 0 and n == T:
            out = shot_frames                        # the whole clip was the shot: nothing to splice
        else:
            out = torch.cat([frames[:start], shot_frames.to(frames.dtype), frames[start + n:]])
        return {"ui": {"text": [shot]} if shot else {}, "result": (out, shot)}


class ProductLabelMaskOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "masks": ("MASK",),
                             "opacity": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("overlay",)
    FUNCTION = "draw"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The mask on its frames, to check it fits the product: tinted green inside, its edge in red (2 px). "
                   "One mask for every frame, or one per frame.")

    def draw(self, images, masks, opacity):
        masks = masks[None] if masks.dim() == 2 else masks
        if masks.shape[1:3] != images.shape[1:3]:
            raise ValueError(f"the mask is {tuple(masks.shape[1:3])}, the frames {tuple(images.shape[1:3])}")
        out = []
        for t in range(images.shape[0]):
            m = (masks[min(t, masks.shape[0] - 1)].float() > 0.5).float()[None, None]
            edge = (torch.nn.functional.max_pool2d(m, 5, stride=1, padding=2) - m)[0, 0, ..., None]
            m = m[0, 0, ..., None]
            im = images[t, ..., :3].float()
            tint = torch.tensor([0.1, 0.9, 0.3], dtype=im.dtype)
            im = im * (1 - opacity * m) + tint * (opacity * m)
            out.append(im * (1 - edge) + torch.tensor([1.0, 0.1, 0.1], dtype=im.dtype) * edge)
        return (torch.stack(out),)


class ProductLabelPackshotMatte:
    USES = ["auto: alpha, else the backdrop's trace where SAM 3 agrees, else SAM 3", "alpha, else SAM 3"]
    # the backdrop's trace and SAM 3's mask overlap this much or more: the trace is kept (exact to the pixel on a plain
    # backdrop); less, and the trace has taken in something round the product (a drop shadow, a reflection)
    AGREE = 0.98

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reference": ("IMAGE",),
                             "mask": ("MASK", {"tooltip": "The packshot Load Image's MASK (1 = transparent)"}),
                             "use": (cls.USES, {"tooltip": (
                                 "A packshot without alpha (a product on a white or black backdrop). auto: the engine's "
                                 "own trace of the backdrop where SAM 3's outline agrees with it, else SAM 3's (the trace "
                                 "took in a drop shadow or a reflection). alpha, else SAM 3: always SAM 3's outline "
                                 "(Label Triage's sheet)")})},
                "optional": {"sam_mask": ("MASK", {"lazy": True, "tooltip": "SAM 3 on the packshot (1 = product)"})}}

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "source")
    FUNCTION = "matte"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The packshot's outline as Load Image's MASK (1 = transparent): its alpha when it has one. Else, with "
                   "auto, the engine's own trace of the backdrop (Load Image's mask passed on as it is) where SAM 3's "
                   "mask of the packshot overlaps it by 98 % or more, and SAM 3's mask where it doesn't: the trace has "
                   "then taken in a drop shadow or a reflection round the product. With 'alpha, else SAM 3', SAM 3's "
                   "mask. source: alpha, sam3 or none (the trace), and for auto why.")

    @staticmethod
    def _alpha(reference, mask):
        m = mask[None] if mask.dim() == 2 else mask
        return m.shape[1:3] == reference.shape[1:3] and bool((m > 0.5).any())

    def check_lazy_status(self, reference, mask, use, sam_mask=_UNSET):
        need = not self._alpha(reference, mask)
        return ["sam_mask"] if need and sam_mask is None else []

    def matte(self, reference, mask, use, sam_mask=None):
        if self._alpha(reference, mask):
            return {"ui": {"text": ["alpha"]}, "result": (mask, "alpha")}
        s = None
        if sam_mask is not None:
            s = sam_mask[None] if sam_mask.dim() == 2 else sam_mask
            s = (s[:1] > 0.5) if s.shape[1:3] == reference.shape[1:3] and bool((s[:1] > 0.5).any()) else None
        src = self._decide(reference, s) if use == self.USES[0] else "sam3" if s is not None else "none"
        out = (1.0 - s.float(), src) if src.startswith("sam3") else (mask, src)
        return {"ui": {"text": [src]}, "result": out}

    def _decide(self, reference, s):
        """auto: the engine's trace (as Auto Draft draws it from a packshot on a plain backdrop) against SAM 3's."""
        from .nodes import _u8
        from .imgpost import autofit as af, raster
        import numpy as np
        ref = _u8(reference[:1], "reference")
        if not af.on_plain_background(ref):
            return "sam3 (the packshot isn't on a plain backdrop)" if s is not None else \
                "none (WARN: no plain backdrop and no SAM 3 outline: the engine takes the whole image as the outline)"
        if s is None:
            return "none (SAM 3 found nothing on the packshot: the backdrop's trace)"
        P = np.asarray(af.trace_outline(af.reference_mask(ref, None)), float)
        tr = raster.fill_polygon(ref.shape[0], ref.shape[1], [tuple(q) for q in P]) > 0
        sm = s[0].detach().cpu().numpy()
        iou = float((tr & sm).sum() / max((tr | sm).sum(), 1))
        extra, miss = int((tr & ~sm).sum()), int((sm & ~tr).sum())
        if iou >= self.AGREE:
            return f"none (the backdrop's trace: SAM 3 agrees, IoU {iou:.3f})"
        return (f"sam3 (the backdrop's trace and SAM 3 overlap {iou:.3f}: the trace takes in {extra} px SAM 3 leaves "
                f"out, a drop shadow or a reflection, and misses {miss} px)")


class ProductLabelSAM3Points:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"points": ("STRING", {"default": "", "tooltip": (
            "Clicks on the keyframe frame, in its px: x,y on the product (several allowed), -x,y on something that "
            "isn't it. Empty: SAM 3 finds the product from the text prompt. Given: SAM 3 segments what was clicked and "
            "the text prompt is left out (SAM 3 would add its own detections to the clicked object), e.g. to pick one "
            "product of several alike in a group shot. One click on the product's body works best: on real frames, "
            "two clicks or a -x,y next to one shrank the mask to part of the label or emptied it. Check the mask "
            "image (triage/mask) after every change.")})}}

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("positive_coords", "negative_coords", "given")
    FUNCTION = "parse"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Clicks for SAM3 Detect: 'x,y x,y -x,y' -> its positive_coords and negative_coords (JSON), and "
                   "whether any were given (then the text prompt is switched off).")

    def parse(self, points):
        import re
        pos, neg = [], []
        for sign, x, y in re.findall(r"([+-]?)\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)", points):
            (neg if sign == "-" else pos).append({"x": float(x), "y": float(y)})
        if neg and not pos:
            raise ValueError("SAM 3 points: give at least one x,y on the product (only -x,y given)")
        return (json.dumps(pos) if pos else "", json.dumps(neg) if neg else "", bool(pos))

class ProductLabelPasteBackMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"before": ("IMAGE", {"tooltip": "The crop as it went in (Image Reframe's image)"}),
                             "after": ("IMAGE", {"tooltip": "The crop as it came out (Composite Behind's image)"}),
                             "crop_data": ("CROP_DATA",),
                             "grow": ("INT", {"default": 2, "min": 0, "max": 64, "tooltip": "px added round the change"})}}

    RETURN_TYPES = ("MASK", "BOOLEAN")
    RETURN_NAMES = ("mask", "straightened")
    FUNCTION = "mask"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Where a straightened crop goes back into its frame: 1 where the run changed it (8 bit, any channel), "
                   "grown by `grow` px, for Image Reframe Composite's mask, so nothing else is resampled. straightened "
                   "is false when the crop is the whole frame, unturned: then the result is the frame itself and no "
                   "paste-back is needed.")

    def mask(self, before, after, crop_data, grow):
        if before.shape[:3] != after.shape[:3]:
            raise ValueError(f"before is {tuple(before.shape[:3])}, after {tuple(after.shape[:3])}: the same size needed")
        b = torch.round(before[..., :3].float().clamp(0, 1) * 255)
        a = torch.round(after[..., :3].float().clamp(0, 1) * 255)
        m = (a != b).any(-1).float()
        if grow > 0:
            m = torch.nn.functional.max_pool2d(m[:, None], 2 * grow + 1, stride=1, padding=grow)[:, 0]
        c = crop_data
        whole = (c["width"] == c["source_width"] and c["height"] == c["source_height"]
                 and c["center_x"] * 2 == c["source_width"] and c["center_y"] * 2 == c["source_height"])
        return (m, not (whole and c["angle"] % 360 == 0))


class ProductLabelFrameMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK", {"tooltip": "Image Reframe's mask: 1 in the padding, else the mask it cropped"}),
                             "crop_data": ("CROP_DATA",)}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "frame_mask"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The product mask in a straightened frame: Image Reframe's mask with its padding taken out. A turned "
                   "frame reaches past the image at its corners, and Image Reframe marks that padding 1; the product "
                   "isn't there, so Auto Draft must see 0. The padding is recomputed from crop_data the way Image Reframe "
                   "computes it (an exact pixel copy when the frame is unturned on whole px, else bilinear coverage).")

    def frame_mask(self, mask, crop_data):
        c = crop_data
        cx, cy, w, h = c["center_x"], c["center_y"], c["width"], c["height"]
        src_w, src_h = c["source_width"], c["source_height"]
        m = mask.float()
        if m.dim() == 2:
            m = m[None]
        x0, y0 = cx - w / 2, cy - h / 2
        if c["angle"] % 360 == 0 and float(x0).is_integer() and float(y0).is_integer():
            x0, y0 = int(x0), int(y0)
            cover = torch.zeros((h, w), device=m.device)
            ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(src_w, x0 + w), min(src_h, y0 + h)
            if ix1 > ix0 and iy1 > iy0:
                cover[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = 1
        else:
            a = math.radians(c["angle"])
            cos_a, sin_a = math.cos(a), math.sin(a)
            xs = torch.arange(w, device=m.device, dtype=torch.float32) + 0.5 - w / 2
            ys = torch.arange(h, device=m.device, dtype=torch.float32) + 0.5 - h / 2
            dy, dx = torch.meshgrid(ys, xs, indexing="ij")
            sx, sy = cx + cos_a * dx + sin_a * dy, cy - sin_a * dx + cos_a * dy
            grid = torch.stack((sx / src_w * 2 - 1, sy / src_h * 2 - 1), dim=-1).unsqueeze(0)
            ones = torch.ones((1, 1, src_h, src_w), device=m.device)
            cover = torch.nn.functional.grid_sample(ones, grid, mode="bilinear", padding_mode="zeros",
                                                    align_corners=False)[0, 0]
        return ((m - (1 - cover)).clamp(0, 1),)


class ProductLabelVideoSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "settings": ("STRING", {"default": "", "multiline": True, "tooltip": (
                "Refine (Video)'s settings JSON: the engine's vjob sections (track, fit, render, qa, review, key), anchor, "
                "shots, words, matte_patch. Empty: everything measured.")}),
            "product_copy": ("STRING", {"default": "", "multiline": True, "tooltip": (
                "The label's copy, read off the packshot, as it is printed: it becomes the settings' `words` (QA reads "
                "it back with OCR, and the text boxes come from it) unless the settings give words.")})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("settings", "product_copy")
    FUNCTION = "merge"
    CATEGORY = CATEGORY
    DESCRIPTION = "Refine (Video)'s settings JSON with the product's copy added as `words`; the copy itself, for Label Triage."

    def merge(self, settings, product_copy):
        words = product_copy.split()
        if not words:
            return (settings, product_copy)
        cfg = json.loads(settings) if settings.strip() else {}
        if not isinstance(cfg, dict):
            raise ValueError("settings: a JSON object")
        cfg.setdefault("words", words)
        return (json.dumps(cfg), product_copy)


class ProductLabelTriage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frame": ("IMAGE", {"tooltip": "One frame: the image, or the clip's keyframe frame"}),
                             "mask": ("MASK", {"tooltip": "The product's mask on that frame (SAM 3)"}),
                             "reference": ("IMAGE", {"tooltip": "The packshot"})},
                "optional": {"reference_mask": ("MASK", {"tooltip": "The packshot's Load Image MASK: a transparent "
                                                                    "packshot's alpha is its outline"}),
                             "product_copy": ("STRING", {"default": "", "multiline": True,
                                                         "tooltip": "The label's copy, for the OCR hint"}),
                             "ocr": ("BOOLEAN", {"default": False, "tooltip": "OCR hints (tesseract; about 10 s): "
                                                 "they don't decide, at frame scale they can't tell right from garbled"}),
                             "glyph": ("BOOLEAN", {"default": False, "tooltip": "A letter-scale match, reported only"})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("sheet", "measures")
    FUNCTION = "triage"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = ("Does the product need fixing at all? Its print band on the frame next to the packshot's at the same "
                   "size, both straightened, in colour, enlarged (the engine's imgauto triage). The agent (or a person) "
                   "looks and calls it fix or fine; a fine product skips the fix (the workflow's Fix this product). "
                   "measures: the triage's JSON, or its error; it never stops the run.")

    def triage(self, frame, mask, reference, reference_mask=None, product_copy="", ocr=False, glyph=False):
        import numpy as np
        from PIL import Image
        from .video_nodes import _engine, _u8
        _engine()                                    # the vendored imgauto, imgvideo, imgpost on sys.path
        from imgauto import triage as tg
        try:
            fr, ref = _u8(frame[:1])[0], _u8(reference[:1])[0]
            m = mask[None] if mask.dim() == 2 else mask
            M = m[0].detach().cpu().numpy() > 0.5
            if M.shape != fr.shape[:2]:
                raise ValueError(f"the mask is {M.shape[1]}x{M.shape[0]}, the frame {fr.shape[1]}x{fr.shape[0]}")
            if not M.any():
                raise ValueError("the product's mask is empty")
            Rm = _reference_matte(ref, reference_mask)
            r = tg.measure(fr, M, ref, Rm, product_copy.split(), glyph=glyph, ocr=ocr)
            path = os.path.join(folder_paths.get_temp_directory(), f"plr_triage_{os.getpid()}_{id(r)}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tg.sheet([("frame", r)], path)
            sheet = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0)[None]
            os.remove(path)
            meas = {k: v for k, v in r.items() if not k.startswith("_")}
        except Exception as e:                       # a look the agent can't take must not stop the fix
            sheet = torch.full((1, 64, 64, 3), 0.1)
            meas = {"verdict": "look", "error": f"{type(e).__name__}: {e}"}
        text = json.dumps(meas)
        return {"ui": {"text": [text]}, "result": (sheet, text)}


def _reference_matte(ref, reference_mask):
    """The packshot's product: its alpha (Load Image's MASK, 1 = transparent) when it has one, else everything that
    stands off the border's colour (the largest part), as the engine's triage reads a packshot file."""
    import numpy as np
    from scipy import ndimage as ndi
    if reference_mask is not None:
        a = reference_mask[0].detach().cpu().numpy() if reference_mask.dim() == 3 else reference_mask.detach().cpu().numpy()
        if a.shape == ref.shape[:2] and (a > 0.5).mean() > 0.05:
            return a < 0.5
    rgb = ref.astype(np.float32)
    border = np.r_[rgb[:3].reshape(-1, 3), rgb[-3:].reshape(-1, 3)]
    Rm = ndi.binary_opening(np.abs(rgb - np.median(border, 0)).max(-1) > 25, iterations=2)
    lab, n = ndi.label(Rm)
    return lab == (1 + int(np.argmax(ndi.sum(Rm, lab, range(1, n + 1))))) if n else Rm


NODE_CLASS_MAPPINGS = {
    "ProductLabelLoadSource": ProductLabelLoadSource,
    "ProductLabelLoadOptional": ProductLabelLoadOptional,
    "ProductLabelSwitch": ProductLabelSwitch,
    "ProductLabelRunIf": ProductLabelRunIf,
    "ProductLabelPrintType": ProductLabelPrintType,
    "ProductLabelPickFrame": ProductLabelPickFrame,
    "ProductLabelShot": ProductLabelShot,
    "ProductLabelFindKeyframe": ProductLabelFindKeyframe,
    "ProductLabelSplice": ProductLabelSplice,
    "ProductLabelMaskOverlay": ProductLabelMaskOverlay,
    "ProductLabelPackshotMatte": ProductLabelPackshotMatte,
    "ProductLabelSAM3Points": ProductLabelSAM3Points,
    "ProductLabelPasteBackMask": ProductLabelPasteBackMask,
    "ProductLabelFrameMask": ProductLabelFrameMask,
    "ProductLabelVideoSettings": ProductLabelVideoSettings,
    "ProductLabelTriage": ProductLabelTriage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProductLabelLoadSource": "Product Label: Load Source (Image or Video)",
    "ProductLabelLoadOptional": "Product Label: Load Optional File",
    "ProductLabelSwitch": "Product Label: Switch",
    "ProductLabelRunIf": "Product Label: Run If",
    "ProductLabelPrintType": "Product Label: Print Type",
    "ProductLabelPickFrame": "Product Label: Pick Frame",
    "ProductLabelShot": "Product Label: Shot (between cuts)",
    "ProductLabelFindKeyframe": "Product Label: Find Keyframe Frame",
    "ProductLabelSplice": "Product Label: Splice Shot Back",
    "ProductLabelMaskOverlay": "Product Label: Mask Overlay",
    "ProductLabelPackshotMatte": "Product Label: Packshot Matte",
    "ProductLabelSAM3Points": "Product Label: SAM 3 Points",
    "ProductLabelPasteBackMask": "Product Label: Paste-Back Mask",
    "ProductLabelFrameMask": "Product Label: Frame Mask",
    "ProductLabelVideoSettings": "Product Label: Video Settings",
    "ProductLabelTriage": "Product Label: Label Triage",
}
