"""Flow nodes for the one workflow that covers every case (workflows/product-label-refinement-all.*): one input for an
image or a clip, optional files, and the switches that run only the branch a job needs.

- Load Source takes an image or a video from ComfyUI's input folder and says which it is (is_video).
- Load Optional is a file that may be left at "(none)": a product mask (else SAM 3), a corrected keyframe (else the
  stills branch makes it). `given` says whether it was set.
- Work Size resizes a small still (and its given mask) up before the fix, so the new letters have pixels enough to
  read (pitfall 113); clips pass as they are.
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


def work_factor(h, w, work_size):
    """The whole factor (1-4) a still's long side is resized by to come near work_size: round(work_size / long side),
    1 when it is that size or more (or work_size is 0)."""
    long_ = max(int(h), int(w))
    if work_size <= 0 or long_ <= 0 or long_ >= work_size:
        return 1
    return max(1, min(4, int(work_size / long_ + 0.5)))


class ProductLabelWorkSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames": ("IMAGE",), "is_video": ("BOOLEAN", {"forceInput": True}),
                             "work_size": ("INT", {"default": 4096, "min": 0, "max": 8192, "step": 64, "tooltip": (
                                 "A still whose long side is under this is resized up (Lanczos, a whole factor up to "
                                 "4) to come near it before the fix, so the new letters are drawn with enough pixels "
                                 "to read; the output keeps that size. 0: off. Clips are never resized.")})},
                "optional": {"masks": ("MASK",)}}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("frames", "masks", "factor", "draft_scale")
    FUNCTION = "resize"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The size the fix works at. A small generated still has a label too few pixels tall to read once "
                   "its letters are right (a 1,024 px frame: 5 px small print), so it is resized up first, a plain "
                   "Lanczos resize (no upscaling model), and the fix draws the packshot's letters at that size. A "
                   "given product mask is resized with it. Clips and stills already at work_size pass as they are. "
                   "draft_scale (for Auto Draft's work_scale): the factor times how many times the frame was already "
                   "resized up (a chained product's frame is the previous product's result), so the draft measures "
                   "at the frame's own size.")

    def resize(self, frames, is_video, work_size=4096, masks=None):
        n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        if is_video or n != 1 or int(work_size) <= 0:
            return (frames, masks, 1, 1)
        import numpy as np
        from PIL import Image
        from .imgpost import workscale
        f = work_factor(h, w, int(work_size))
        a = np.clip(frames[0].cpu().numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)
        before, _ = workscale.resized_by(a)
        if f == 1:
            return (frames, masks, 1, before)
        big = np.asarray(Image.fromarray(a).resize((w * f, h * f), Image.LANCZOS)).astype(np.float32) / 255.0
        out = torch.from_numpy(big)[None]
        mk = masks
        if masks is not None:
            mk = torch.nn.functional.interpolate(masks[:, None].float(), size=(h * f, w * f), mode="bilinear",
                                                 align_corners=False)[:, 0]
            mk = (mk > 0.5).float()
        return (out, mk, f, f * before)


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
                                                           "same shot"}),
                             "keyframe_report": ("STRING", {"forceInput": True, "tooltip": (
                                 "Find Keyframe Frame's report: where it found the product lost and found again "
                                 "inside a shot (splits), taken as cuts for a found run's shot (its keyframe frame "
                                 "not the report's own); the keyframe's own shot stays whole (V83)")})}}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING")
    RETURN_NAMES = ("frames", "masks", "start", "keyframe_index", "shot")
    FUNCTION = "cut"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The keyframe frame's shot out of a clip, so SAM 3 and Refine (Video) work on that shot only: between "
                   "the hard cuts around the keyframe frame (found from the frames), or from start to end. "
                   "keyframe_index: the keyframe frame counted from the shot's start; start: where the shot goes back "
                   "(Splice); shot: JSON with start, end, the cuts found and the clip's length. Refine's report counts "
                   "frames from the shot's start: add start for the clip's frame. An image is its own shot.")

    def cut(self, frames, keyframe_index, start=-1, end=-1, masks=None, keyframe_report=None):
        T, k = int(frames.shape[0]), int(keyframe_index)
        auto = (start < 0 or end < 0) and k >= 0
        cuts = find_cuts(frames) if auto else []
        splits = []
        if auto and keyframe_report:
            try:
                kr_ = json.loads(keyframe_report) or {}
                # the keyframe's own shot stays whole (refined as before); a found run's shot ends at its splits
                if int(kr_.get("keyframe_frame", -1)) != k:
                    splits = [int(x) for x in kr_.get("splits") or [] if 0 < int(x) < T]
            except (TypeError, ValueError, AttributeError):
                splits = []
        # the product lost and found again inside a shot (Find Keyframe Frame): a shot of its own, as at a cut
        edges = sorted(set(cuts) | set(splits))
        if k >= T:
            raise ValueError(f"the keyframe frame {k} is past the clip's last frame ({T - 1})")
        s = start if start >= 0 else max([c for c in edges if c <= k], default=0) if k >= 0 else 0
        e = end if end >= 0 else min([c - 1 for c in edges if c > k], default=T - 1) if k >= 0 else T - 1
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
                "clip_frames": T, "cuts": cuts, "found": {"start": start < 0, "end": end < 0} if auto else False,
                **({"splits": splits} if splits else {})}
        return (frames[s:e + 1], masks, s, k - s if k >= 0 else -1, json.dumps(shot))


def _sam3_detect(model, images, conditioning, threshold=0.33):
    """Core SAM3_Detect on a batch of frames (one union mask per frame), as the workflow's own detect node runs it."""
    from comfy_extras.nodes_sam3 import SAM3_Detect
    o = SAM3_Detect.execute(model=model, image=images, conditioning=conditioning, threshold=threshold,
                            refine_iterations=2, individual_masks=False)
    m = (getattr(o, "result", None) or getattr(o, "args", None) or o)[0]
    return m[None] if m.dim() == 2 else m


CANDIDATES = 6           # SAM 3's detections of a one-object prompt scored against the packshot
PICK_MIN_AREA = 0.0002   # a detection smaller than this share of the frame isn't a candidate
PICK_MARGIN = 0.06       # another detection replaces SAM 3's likeliest only when its match is this much higher (a
                         # clear bottle over an amber one asked for: 0.74 vs 0.88; two alike bottles, left and right:
                         # 0.85 vs 0.87, the prompt decides)
PICK_FLOOR = 0.5         # the pick matching the packshot less than this is another product: stop (the product asked
                         # for scored 0.53-0.91 on its own frames, partly hidden or opened included; a shelf neighbour
                         # the prompt fit better than the product: 0.35)


def _sam3_candidates(model, image, conditioning, threshold=0.33, refine_iterations=2, n=CANDIDATES):
    """SAM 3's detections of a one-object prompt on one frame, each on its own, best SAM 3 score first, up to n.
    -> (masks [N, H, W] bool, scores), or None when the prompt asks for several objects or kinds (name:N, commas)."""
    if not conditioning:
        return None
    meta = conditioning[0][1]
    multi = meta.get("sam3_multi_cond")          # only there for several prompts or name:N (one prompt: plain)
    if multi is not None and (len(multi) != 1 or multi[0].get("max_detections", 1) != 1):
        return None
    entry = multi[0] if multi else {"cond": conditioning[0][0], "attention_mask": meta.get("attention_mask")}
    cond = [[conditioning[0][0], {**meta, "sam3_multi_cond": [{**entry, "max_detections": n}]}]]
    from comfy_extras.nodes_sam3 import SAM3_Detect
    o = SAM3_Detect.execute(model=model, image=image[:1], conditioning=cond, threshold=threshold,
                            refine_iterations=refine_iterations, individual_masks=True)
    r = getattr(o, "result", None) or getattr(o, "args", None) or o
    boxes = r[1][0] if r[1] else []
    return r[0].detach().cpu().numpy() > 0.5, [round(float(b["score"]), 3) for b in boxes]


def _pick_product(model, image, conditioning, ref, Rm, threshold=0.33, refine_iterations=2):
    """The product on one frame: of SAM 3's detections of the prompt, the one that matches the packshot best (its
    silhouette x colour, from _match_packshot), not only SAM 3's likeliest. A group shot can hold a product the
    prompt fits better (another box beside a battery pack). -> (mask [H, W] bool, report dict). A prompt for several
    objects or kinds keeps SAM 3's own union."""
    import numpy as np
    from .video_nodes import _engine, _u8
    _engine()
    from imgauto import refangle
    c = _sam3_candidates(model, image, conditioning, threshold, refine_iterations)
    if c is None:
        m = _sam3_detect(model, image[:1], conditioning, threshold)[0].detach().cpu().numpy() > 0.5
        return m, {"from": "SAM 3's union (the prompt asks for several objects or kinds)"}
    masks, scores = c
    H, W = image.shape[1:3]
    if not len(masks):
        return np.zeros((H, W), bool), {"from": "SAM 3 found nothing", "candidates": []}
    fr = _u8(image[:1])[0]
    cands = [{"sam3": s, **_match_packshot(fr, m, ref, Rm, refangle, PICK_MIN_AREA)} for m, s in zip(masks, scores)]
    # which product it is: silhouette x colour. Size is left out here (a bigger neighbour isn't more this product);
    # it only weighs which shot to fix from
    for c in cands:
        c["match"] = round(c["shape"] * c["colour"], 4)
    # SAM 3's likeliest stays unless another is clearly more this product: two of a kind look alike to the packshot,
    # and there the prompt's own words (left, right, a colour) decide
    best = max(range(len(cands)), key=lambda j: (cands[j]["match"], -j))
    i = best if cands[best]["match"] > cands[0]["match"] + PICK_MARGIN else 0
    rep = {"from": "the packshot's best match of SAM 3's detections", "pick": i, "candidates": cands}
    if i:
        rep["note"] = (f"SAM 3's likeliest detection ({cands[0]['sam3']}) matches the packshot less "
                       f"({cands[0]['match']}) than its #{i + 1} ({cands[i]['sam3']}: {cands[i]['match']}): took #{i + 1}")
    elif best:
        rep["note"] = (f"SAM 3's #{best + 1} matches the packshot about as well ({cands[best]['match']} vs "
                       f"{cands[0]['match']}): kept SAM 3's likeliest, as the prompt words it")
    return masks[i], rep


class ProductLabelSAM3Product:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "model": ("MODEL", {"tooltip": "SAM 3"}),
                    "image": ("IMAGE", {"tooltip": "The keyframe frame (an image: the image)"}),
                    "threshold": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "refine_iterations": ("INT", {"default": 2, "min": 0, "max": 5})},
                "optional": {
                    "conditioning": ("CONDITIONING", {"tooltip": "SAM 3's text prompt (none when points are given)"}),
                    "positive_coords": ("STRING", {"forceInput": True}),
                    "negative_coords": ("STRING", {"forceInput": True}),
                    "reference": ("IMAGE", {"tooltip": "The packshot"}),
                    "reference_mask": ("MASK", {"tooltip": "The packshot's outline (Load Image's convention: 1 = "
                                                           "transparent)"})}}

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "report")
    FUNCTION = "detect"
    CATEGORY = CATEGORY
    DESCRIPTION = ("SAM 3 on the keyframe frame: the product. Of SAM 3's detections of the prompt (up to 6), the one "
                   "whose silhouette and colour match the packshot best, so a group shot's neighbour the prompt also "
                   "fits isn't fixed instead. Points (SAM 3 points node) segment what was clicked; a prompt for "
                   "several objects or kinds (name:N, commas) keeps SAM 3's union. Stops when even the best match is "
                   "below 0.5: none of the detections is the packshot's product (the prompt fit a neighbour). report: "
                   "each detection's scores and the pick.")

    def detect(self, model, image, threshold, refine_iterations, conditioning=None, positive_coords=None,
               negative_coords=None, reference=None, reference_mask=None):
        points = bool(json.loads(positive_coords or "[]") or json.loads(negative_coords or "[]"))
        if points or conditioning is None or reference is None:
            from comfy_extras.nodes_sam3 import SAM3_Detect
            o = SAM3_Detect.execute(model=model, image=image[:1], conditioning=conditioning, threshold=threshold,
                                    refine_iterations=refine_iterations, individual_masks=False,
                                    positive_coords=positive_coords, negative_coords=negative_coords)
            m = (getattr(o, "result", None) or getattr(o, "args", None) or o)[0]
            rep = json.dumps({"from": "the points given" if points else "SAM 3 (no packshot to pick by)"})
            return {"ui": {"text": [rep]}, "result": (m[None] if m.dim() == 2 else m, rep)}
        from .video_nodes import _u8
        ref = _u8(reference[:1])[0]
        rm = reference_mask[None] if reference_mask is not None and reference_mask.dim() == 2 else reference_mask
        Rm = rm[0].detach().cpu().numpy() < 0.5 if rm is not None and rm.shape[1:3] == reference.shape[1:3] and \
            bool((rm > 0.5).any()) else _reference_matte(ref, None)
        m, rep = _pick_product(model, image, conditioning, ref, Rm, threshold, refine_iterations)
        text = json.dumps(rep, default=float)
        print("product    " + rep.get("note", rep["from"]))
        if rep.get("candidates"):
            c = rep["candidates"][rep["pick"]]
            if c["match"] < PICK_FLOOR:
                # a product the frame cuts off (a close-up of its body) matches the packshot only on its part in
                # frame: judged there before it's called another product
                cs = _cut_shape(Rm, m)
                if cs is not None:
                    c["shape_in_frame"] = cs
                    c["match"] = max(c["match"], round(cs * c["colour"], 4))
                    text = json.dumps(rep, default=float)
            if c["match"] < PICK_FLOOR:
                raise ValueError(
                    f"none of SAM 3's {len(rep['candidates'])} detections of the prompt is the packshot's product: the "
                    f"best matches it {c['match']} (silhouette {c['shape']} x colour {c['colour']}; below {PICK_FLOOR} "
                    f"is another product). Name the product as the frame shows it in the SAM 3 prompt, or give points "
                    f"on it. Detections: {text}")
        return {"ui": {"text": [text]}, "result": (torch.from_numpy(m.astype("float32"))[None], text)}


class ProductLabelFindKeyframe:
    # candidates per shot: these fractions of the way through it
    PROBES = (0.25, 0.5, 0.75)
    MIN_AREA = 0.002                         # a detection smaller than this share of the frame isn't the product
    TIE = 0.01                               # keypick scores this close to the best are a tie
    CLOSE = 0.7                              # another shot scoring this share of the best one's: a warning
    ALSO = 0.5                               # ... this share: likely showing the product too, refined from its own
                                             # keyframe frame (Every Shot)
    SHORT = 6                                # a likely shot under this many frames (a quarter second at 24 fps) is a
                                             # transition (a whip pan, a flash frame): passes as generated (a coloured
                                             # pump's edit: 1-frame shots of a pan between its shots)
    WHOLE = 0.90                             # a frame whose product outline matches the packshot's under this share of
                                             # the shot's (90th percentile) shows part of it only (a hand, a holder)
    PARTIAL = 0.6                            # ... and keypick's score is multiplied by this (as an occluded frame)
    LOST_IOU = 0.30                          # a product lost and found again overlapping where it was lost under this
                                             # (the engine's iou_cut; a tube drawn again on its box's front: 0.0, the
                                             # same tube's bounce: 0.28) ...
    LOST_SHAPE = 0.90                        # ... or its silhouette changed under this (imgauto.coverage's redrawn;
                                             # 0.84 on the box's front, a bounce 0.98): a shot of its own (V83)
    STEP = 3                                 # where the track holds no run of the product for SHORT frames or more,
                                             # SAM 3 looks for it again on every STEP-th frame (V83: a serum's track
                                             # lost it under its box's sleeve and never took up the serum drawn on the
                                             # box's front, which SAM 3 finds on its own: match 0.5 or more)

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
        "SAM 3 on three frames (its detection that matches the packshot best); the shot whose product matches the "
        "packshot best (silhouette, colour, size on screen); in that shot, SAM 3's track and, of the frames where the "
        "product is in view, the one where it shows largest, sharpest and most frontal, off morph frames and the "
        "shot's ends (the engine's keypick), and whole: a frame whose product outline matches the packshot's well "
        "under the shot's own (a hand placing it, something in front of it) is passed over. With a corrected keyframe, -1 "
        "passes on (Refine finds its frame); an image is frame 0. report: JSON with every shot's score, the pick and "
        "its runner-up; a margin under 5 % or a weak match is flagged.")

    @staticmethod
    def _entry(r, p):
        s = r["shot"][0]
        return {"shot": r["shot"], "keyframe_frame": p["k"], "score": r["score"],
                "in_view": {"first": s + p["view"][0], "last": s + p["view"][-1], "frames": len(p["view"]), "of": p["n"]}}

    def _pick(self, frames, s, e, T, best_t, probes, pm, mask_given, masks, model, conditioning, Rm):
        """The keyframe frame of shot s-e (clip frames, e included): SAM 3 tracked through it from its best probe (or
        the given masks), and of the frames where the product is in view the one keypick scores best, off partial
        frames, the most central of a tie. -> {"k", "view", "n", "best", "near", "ranking"}, or why there is none."""
        import numpy as np
        from imgauto import keypick, refangle
        # the shot's masks: the given ones, else SAM 3 tracked through it from its best probe
        tr = self._track(frames, s, e, best_t, pm[probes.index(best_t)] if not mask_given else None, mask_given,
                         masks, model, conditioning)
        M2, stats, bnds, n = (tr[x] for x in ("M2", "stats", "bounds", "n"))
        # the candidates: the frames where the product is in view. A shot can show it for part of its length only (it
        # slides behind something, into a box, out of the frame); there the track is empty and nothing is measured
        view = [t for t in range(n) if "width_px" in stats[t] and stats[t]["area"] >= self.MIN_AREA]
        if not view:
            return (f"SAM 3's track through shot {s}-{e} doesn't hold the product on any frame: check the SAM 3 "
                    "prompt, or set the keyframe frame")
        # the shot's ends are cuts (penalised as edges) unless they are the clip's own first or last frame; so are the
        # frames where the product comes into view or leaves it
        ranking = keypick.score(stats, bnds, view, open_ends=[t for t, c in ((0, s == 0), (n - 1, e == T - 1)) if c])
        # whole: the product's outline against the packshot's on each frame in view, against the shot's own (its 90th
        # percentile). A hand placing the product or something in front of it for many frames on end isn't seen by
        # keypick's occlusion (it compares a frame with its neighbours, as hidden as it is), and a keyframe there leaves
        # the hidden part out of the label's draft (a spray bottle's trigger under the hand: 0.64 of the free frames'
        # 0.98, and Auto Draft stopped). Under WHOLE of the shot's: flagged partial, its score times PARTIAL
        Rk = max(1, int(round(max(Rm.shape) / max(M2.shape[1:]))))
        Rs = Rm[::Rk, ::Rk]
        whole = {t: refangle.shape_iou(Rs, M2[t]) for t in view}
        ref_w = float(np.percentile(list(whole.values()), 90))
        for r in ranking:
            w = whole.get(r["t"])
            if w is None or ref_w <= 0:
                continue
            r["terms"]["whole"] = round(w / ref_w, 2)
            if w < self.WHOLE * ref_w:
                r["flags"].append("partial")
                r["score"] = round(r["score"] * self.PARTIAL, 3)
        ranking.sort(key=lambda r: -r["score"])
        top_s = ranking[0]["score"]
        # frames keypick can't tell apart (within TIE of the best): the one nearest the middle of the product's time in
        # view, so the label's track runs the shortest way to either end (its error grows with the distance)
        near = [r for r in ranking if r["score"] >= (1.0 - self.TIE) * top_s]
        mid = (view[0] + view[-1]) / 2
        best = min(near, key=lambda r: (abs(r["t"] - mid), -r["score"]))
        return {"k": s + int(best["t"]), "view": view, "n": n, "best": best, "near": near, "ranking": ranking,
                "track": tr}

    def _track(self, frames, s, e, t0, mask0, mask_given, masks, model, conditioning):
        """SAM 3 tracked through shot s-e (clip frames, e included) both ways from frame t0's mask (or the given
        masks), and the engine's analysis of it at half size. -> {"M" (full size), "M2", "stats", "bounds", "shots",
        "why", "n", "trims" (the engine's fixable runs, [a, b) from the shot's start)}"""
        from .video_nodes import _u8
        from imgauto import shots as ash
        if mask_given:
            sm = masks[s:e + 1]
        else:
            from .video_nodes import ProductLabelVideoSAM3BothWays
            km = torch.from_numpy(mask0.astype("float32"))[None]
            sm = ProductLabelVideoSAM3BothWays().track(frames[s:e + 1], model, km, t0 - s, conditioning=conditioning,
                                                       detection_threshold=0.33, max_objects=1, detect_interval=1)[0]
        F = _u8(frames[s:e + 1])
        M = sm.detach().cpu().numpy() > 0.5
        F2, M2 = ash.downscale(F, 2), ash.downscale(M, 2)
        stats, bnds, sh, _kinds, why = ash.analyse(F2, M2)
        return {"M": M, "M2": M2, "stats": stats, "bounds": bnds, "shots": sh, "why": why, "n": e - s + 1,
                "trims": sorted(tuple(x) for sh_ in sh for x in sh_["trims"])}

    def _unheld(self, tr):
        """The runs of a shot (from its start, [a, b)) where its track holds no fixable run of the product, SHORT
        frames or more."""
        held = {t for a, b in tr["trims"] for t in range(a, b)}
        runs, t, n = [], 0, tr["n"]
        while t < n:
            if t in held:
                t += 1
                continue
            u = t
            while u < n and u not in held:
                u += 1
            if u - t >= self.SHORT:
                runs.append((t, u))
            t = u
        return runs

    def _lost_found(self, tr):
        """Where SAM 3's track through a shot loses the product (frames with no matte between two of the engine's
        fixable runs) and finds it again elsewhere or redrawn, with no cut: a wipe, an occlusion, the product drawn
        again on its packaging. The keyframe's label can't be carried across (no tracks cross the lost frames, and the
        product found is drawn anew), so the found run is a shot of its own (V83). A product found again where it
        was lost, its shape kept (a hand passing in front), carries on. -> every gap between two of the engine's runs:
        [{"at": the next run's first frame, "lost": [first, last], "why", "found_frames", "iou", "shape", "split"}]
        (frames from the shot's start)"""
        from imgauto import refangle
        M2, trims, why = tr["M2"], tr["trims"], tr["why"]
        out = []
        for (a0, a1), (b0, b1) in zip(trims, trims[1:]):
            gap = {"at": int(b0), "lost": [int(a1), int(b0 - 1)],
                   "why": sorted({why[t] or "short run" for t in range(a1, b0)}), "found_frames": int(b1 - b0)}
            x, y = M2[a1 - 1], M2[b0]
            if x.any() and y.any():
                gap["iou"] = round(float((x & y).sum() / max((x | y).sum(), 1)), 3)
                gap["shape"] = round(float(refangle.shape_iou(y, x)), 3)
            gap["split"] = bool(b1 - b0 >= self.SHORT and any(w.startswith("no matte") for w in gap["why"]) and
                                "iou" in gap and (gap["iou"] < self.LOST_IOU or gap["shape"] < self.LOST_SHAPE))
            out.append(gap)
        return out

    def _search(self, frames, s, tr, model, conditioning, ref, Rm):
        """The runs of a shot where its track holds no run of the product (SHORT frames or more), searched again: SAM 3
        on every STEP-th frame, its detection that matches the packshot best (PICK_FLOOR or more is the product). A
        product found on SHORT frames or more, away from where the track last held it (overlapping it under
        LOST_IOU), is a shot of its own from the first frame it is found on (V83). -> [{"run", "probes": {frame:
        match}, "at", "lost", "iou", "shape", "split", "by", "probe", "mask"}] (clip frames; mask: the probe's
        detection, full size)"""
        from imgauto import refangle
        M = tr["M"]
        held = {t for a, b in tr["trims"] for t in range(a, b)}
        runs = self._unheld(tr)
        cache, out = {}, []
        area = M.shape[1] * M.shape[2]

        def det(t):
            if t not in cache:
                m, rep = _pick_product(model, frames[s + t:s + t + 1], conditioning, ref, Rm)
                c = rep["candidates"][rep["pick"]] if rep.get("candidates") else None
                ok = c is not None and c["match"] >= PICK_FLOOR and m.sum() >= self.MIN_AREA * area
                cache[t] = (ok, m, round(float(c["match"]), 3) if c else None)
            return cache[t]

        for r0, r1 in runs:
            ts = list(range(r0 + 1, r1, self.STEP))
            groups = []
            for t in ts:
                if det(t)[0]:
                    if groups and t - groups[-1][-1] <= self.STEP:
                        groups[-1].append(t)
                    else:
                        groups.append([t])
            entry = {"run": [s + r0, s + r1 - 1], "probes": {str(s + t): cache[t][2] for t in ts}}
            out.append(entry)
            for g in groups:
                if g[-1] - g[0] + 1 < self.SHORT:
                    continue
                f = g[0]
                while f - 1 >= r0 and det(f - 1)[0]:
                    f -= 1
                if f == 0:
                    continue                 # found from the shot's first frame: nothing to split off
                near = r0 - 1 if r0 - 1 in held else (r1 if r1 in held else None)
                mid = g[len(g) // 2]
                y = cache[g[0]][1]
                gap = {"run": entry["run"], "probes": entry["probes"], "at": s + f, "lost": [s + r0, s + f - 1],
                       "found_frames": g[-1] - f + 1, "by": "search", "probe": s + mid, "mask": cache[mid][1]}
                if near is not None and M[near].any():
                    x = M[near]
                    gap["iou"] = round(float((x & y).sum() / max((x | y).sum(), 1)), 3)
                    gap["shape"] = round(float(refangle.shape_iou(y[::2, ::2], x[::2, ::2])), 3)
                gap["split"] = near is None or gap.get("iou", 0.0) < self.LOST_IOU
                out[-1] = gap
        return out

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
        # each probe's product: the given mask, else SAM 3's detection that matches the packshot best
        pm = [m.detach().cpu().numpy() > 0.5 for m in masks[torch.tensor(probes)]] if mask_given else \
            [_pick_product(model, frames[t:t + 1], conditioning, ref, Rm)[0] for t in probes]
        shots_rep = []
        for s, e in bounds:
            cands = []
            for j, t in enumerate(probes):
                if not s <= t <= e:
                    continue
                M = pm[j]
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
        main = self._pick(frames, s, e, T, top["best"], probes, pm, mask_given, masks, model, conditioning, Rm)
        if isinstance(main, str):
            raise ValueError(main)
        k, view, n, best, near, ranking = (main[x] for x in ("k", "view", "n", "best", "near", "ranking"))
        warn = []
        if best["flags"]:
            warn.append(f"the pick is flagged {', '.join(best['flags'])} (keypick): check the keyframe frame")
        # the other shots likely showing the product too. Each long enough to hold a label gets its own keyframe frame,
        # picked as the best shot's is, and is refined too (Every Shot); a shorter one is a transition (a whip pan, a
        # flash frame) and passes as generated
        others = [r for r in ranked[1:] if r["score"] >= self.ALSO * top["score"]]
        also = [r["shot"] for r in others if r["shot"][1] - r["shot"][0] + 1 >= self.SHORT]
        passing = [r["shot"] for r in others if r["shot"][1] - r["shot"][0] + 1 < self.SHORT]
        every = [self._entry(top, main)]
        left = []
        for r in sorted((r for r in others if r["shot"] in also), key=lambda r: r["shot"][0]):
            p = self._pick(frames, r["shot"][0], r["shot"][1], T, r["best"], probes, pm, mask_given, masks, model,
                           conditioning, Rm)
            if isinstance(p, str):
                left.append({"shot": r["shot"], "why": p})
            else:
                every.append(self._entry(r, p))
        # the product lost and found again inside the keyframe's shot (no cut: a wipe, an occlusion, the product drawn
        # again on its packaging): the found run is refined as a shot of its own, from its own keyframe frame (V83),
        # inside the keyframe's shot, which stays whole and is refined as before; the found run fills only the frames
        # it leaves as they are (Every Shot, Refine). Shot reads the splits from this report for the found run's bounds
        # Only where the keyframe frame's track shows a hint (a gap between two runs, or 6 frames or more it holds no
        # run on) is the shot tracked again from the keyframe frame itself, as its own SAM 3 track will be, so the
        # split keeps every frame that track holds (a probe's track can jump to the drawn product early, while the
        # real one still shows); a clip with no hint runs as before
        gaps, kt = [], None
        if self._lost_found(main["track"]) or self._unheld(main["track"]):
            if mask_given:
                kt = main["track"]
            else:
                km, _ = _pick_product(model, frames[k:k + 1], conditioning, ref, Rm)
                kt = self._track(frames, s, e, k, km, False, None, model, conditioning)
            gaps = [dict(x, at=s + x["at"], lost=[s + y for y in x["lost"]], by="track") for x in self._lost_found(kt)]
            if not mask_given:
                gaps += self._search(frames, s, kt, model, conditioning, ref, Rm)
        lost = sorted((x for x in gaps if x.get("split")), key=lambda x: x["at"])
        s0 = s
        if lost:
            splits = [x["at"] for x in lost]
            segs = list(zip([s] + splits, [x - 1 for x in splits] + [e]))
            ms = next(a for a, b in segs if a <= k <= b)
            for a, b in segs:
                if a == ms:
                    continue
                hit = next((x for x in lost if x["at"] == a and x.get("probe") is not None), None)
                if hit is not None:
                    pt, pmask = hit["probe"], hit["mask"]        # where SAM 3 found it again
                else:
                    tr = [(s0 + x0, s0 + x1) for x0, x1 in kt["trims"] if a <= s0 + x0 and s0 + x1 - 1 <= b]
                    if not tr:
                        left.append({"shot": [a, b], "why": "no run of frames there holds the product's label"})
                        continue
                    x0, x1 = max(tr, key=lambda r: r[1] - r[0])
                    pt = (x0 + x1 - 1) // 2
                    pmask = kt["M"][pt - s0]
                p = self._pick(frames, a, b, T, pt, probes + [pt], pm + [pmask], mask_given, masks,
                               model, conditioning, Rm)
                if isinstance(p, str):
                    left.append({"shot": [a, b], "why": p})
                    continue
                also.append([a, b])
                every.append(dict(self._entry({"shot": [a, b], "score": top["score"]}, p),
                                  **{"from": "lost and found again"}))
            print(f"keyframe   lost and found again in shot {s0}-{segs[-1][1]}: " + ", ".join(
                f"lost {x['lost'][0]}-{x['lost'][1]}, found at {x['at']} by its {x['by']} (overlap {x.get('iou')}, "
                f"silhouette {x.get('shape')})" for x in lost))
        for r in left:
            warn.append(f"shot {r['shot'][0]}-{r['shot'][1]} likely shows the product but isn't refined: {r['why']}")
        done = [x["shot"] for x in every[1:]]
        close = [r for r in ranked[1:] if r["score"] >= self.CLOSE * top["score"] and r["shot"] in done]
        for r in close:
            warn.append(f"shot {r['shot'][0]}-{r['shot'][1]} matches the packshot nearly as well ({r['score']:.3f} vs "
                        f"{top['score']:.3f}): it is refined too, from its own keyframe frame. If it shows another "
                        "product of the same kind, name what tells this one apart in the SAM 3 prompt (its colour)")
        rep = {"keyframe_frame": k, "from": "found", "shot": [s, e], "also_likely": also,
               "every_shot": every, "not_refined": left, "passing": passing,
               "in_view": {"first": s0 + view[0], "last": s0 + view[-1], "frames": len(view), "of": n},
               "pick": {**best, "t": k, "best_scored": s0 + int(ranking[0]["t"]),
                        "tied": [s0 + int(r["t"]) for r in sorted(near, key=lambda r: r["t"])]},
               "shots": shots_rep, "warnings": warn,
               **({"gaps": [{k_: v for k_, v in x.items() if k_ != "mask"} for x in gaps]} if gaps else {}),
               **({"lost_found": [{k_: v for k_, v in x.items() if k_ not in ("mask", "probes")} for x in lost],
                   "splits": [x["at"] for x in lost]} if lost else {})}
        text = json.dumps(rep, default=float)
        print(f"keyframe   frame {k} (shot {s}-{e}, the product in view on {len(view)} of its {n} frames; "
              f"score {top['score']:.2f}; " +
              ", ".join(f"{r['shot'][0]}-{r['shot'][1]} {r['score']:.2f}" for r in shots_rep) + ")" +
              "".join(f"\nkeyframe   also shot {x['shot'][0]}-{x['shot'][1]}: frame {x['keyframe_frame']}" for x in every[1:]) +
              (f"\nkeyframe   passing (under {self.SHORT} frames): " + ", ".join(f"{a}-{b}" for a, b in passing)
               if passing else "") +
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


def _cut_shape(Rm, M, k=4):
    """The silhouettes' overlap where the product runs off the frame (a close-up cutting off its cap): the packshot's
    silhouette posed by the product's width across its axis (not its length, the frame cuts that) and its end in
    frame, the frame cutting the packshot as it cuts the product; the best of a few scales round the width's. -> 0..1,
    or None when the product doesn't touch the frame's edge. A product seen whole matches by refangle.shape_iou."""
    import numpy as np
    from scipy import ndimage as ndi
    Mk = M[::k, ::k]
    b = 2
    edge = np.zeros_like(Mk)
    edge[:b] = edge[-b:] = True
    edge[:, :b] = edge[:, -b:] = True
    if (Mk & edge).sum() < 4 or Mk.sum() < 200 or Rm.sum() < 200:
        return None

    def pose(m):
        ys, xs = np.nonzero(m)
        P = np.c_[ys, xs].astype(float)
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        ax = vt[0] * (np.sign(vt[0][0]) or 1.0)
        n = np.array([-ax[1], ax[0]])
        return P, c, ax, n, (P - c) @ ax, (P - c) @ n

    P, c, ax, n, u, v = pose(Mk)
    _, cr, ar, nr, ur, vr = pose(Rm)
    yb, xb = np.nonzero(Mk & edge)
    ub = ((np.c_[yb, xb] - c) @ ax).mean()
    cut_low = abs(ub - u.min()) < abs(ub - u.max())      # the frame cuts the product's low end along its axis
    base = (np.percentile(np.abs(v), 98) / max(np.percentile(np.abs(vr), 98), 1e-6))
    best = 0.0
    for f in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15):
        sc = base * f
        off = u.max() - sc * ur.max() if cut_low else u.min() - sc * ur.min()
        A = (np.outer(ar, ax) + np.outer(nr, n)) / sc
        d = cr - A @ c - (off / sc) * ar
        mr = ndi.affine_transform(Rm.astype(np.float32), A, d, output_shape=Mk.shape, order=1) > 0.5
        best = max(best, float((Mk & mr).sum() / max((Mk | mr).sum(), 1)))
    return round(best, 3)


def _audit_fails(report):
    """The FAIL entries of QA Sheet's report (its text)."""
    try:
        rep = json.loads(report)
    except (TypeError, ValueError):
        rep = {}
    return [f for f in rep.get("audit") or [] if f.get("level") == "FAIL"]


class ProductLabelAuditName:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"report": ("STRING", {"forceInput": True, "tooltip": "QA Sheet's report (its text)"}),
                             "prefix": ("STRING", {"default": "product-label-refinement/refined"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename_prefix",)
    FUNCTION = "name"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The refined image's file name: the prefix, with _AUDIT-FAIL added when its audit has a FAIL. The "
                   "image and its QA sheets are still saved, so the run's outputs show what failed (qa/audit); the name "
                   "keeps it from passing for a finished image.")

    def name(self, report, prefix):
        fails = _audit_fails(report)
        out = prefix + "_AUDIT-FAIL" if fails else prefix
        return {"ui": {"text": ["; ".join(f"{f['check']}: {f['text']}" for f in fails) or "audit: no FAIL"]},
                "result": (out,)}


class ProductLabelReviewName:
    SHARE = 0.5                  # a check over more than this share of the refined frames
    WRONG = ("double", "ghost")  # the print wrong on the frame (doubled, the old print through it): over SHARE, FAIL
    GHOST = 0.15                 # ghost alone over this share, FAIL, where the label is rendered (V79: approved clips 0-0.05
                                 # in every run; a coloured pump's 0.47 let the old print through). With the keyframe
                                 # carried (kf_kept reported) ghost reads the carried keyframe as the truth: what it
                                 # carries can't show there, and KF_OLD reads it on the keyframe instead
    KF_OLD = 0.2                 # with the keyframe carried, its old print left round the new over this share: FAIL (V79:
                                 # the rejected coloured pump keyframe 0.346, fixed 0.088; approved 0.041-0.139)
    CARRIED = 0.6                # under this share of the keyframe's corrected print in the rendered band: FAIL (V74)
    KF_KEPT = 0.9                # under this share of the keyframe's edit delivered on its own frame (the edit carry,
                                 # V76: a surface short of a shoulder left its line ends garbled; carried 0.97): FAIL
    CARRIED_DEFECT = ("spots",)  # the keyframe's whole edit carried (kf_kept reported): a keyframe defect repeats on every
                                 # frame; over SHARE, FAIL (V78: dark fill spots on 119 of 120; the spray clip 13 of 104)
    EDGE = ("drop", "drift", "twitch")  # the label left out next to its run while the product carries on, or turning
                                        # off the body where a run starts or ends (V81: the user saw 120-121 drop
                                        # and 123 drift on a three-shot clip), or off the product's own tracks for a
                                        # frame and back (V82: the user saw a tube's label jump at 99-104): any, FAIL
    LEFT = 6                           # the engine's fixable frames (the product in view, its print readable) left
                                        # as generated, this many in a row or more: FAIL (V83: a tube and a serum
                                        # drawn again on their box's front, 25 frames left; shorter runs are a
                                        # shot's edge frames, V81)
    HELD = ("slide", "creep", "fade", "lock")   # the label off the print it lies on, held (the check's own run), creeping on
                                        # a still product (V71: the user saw 0.6 px over a still shot), or fading over
                                        # the old print where the product leaves right after (V72), or (the edit
                                        # carry) the new print drifting against the generated print (V77): any, FAIL

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"report": ("STRING", {"forceInput": True, "tooltip": "Refine (Video)'s report (its text)"}),
                             "prefix": ("STRING", {"default": "product-label-refinement/video/refined"}),
                             "save_summary": ("BOOLEAN", {"default": True, "tooltip": (
                                 "Save the review's verdict and checks as a JSON file under the name (one of the "
                                 "clip's two names is enough)")})},
                "optional": {"keyframe_report": ("STRING", {"forceInput": True, "tooltip": (
                    "Find Keyframe Frame's report: other shots that likely show the product too (also_likely); any "
                    "this run leaves as generated is a _REVIEW-FAIL")}),
                             "every_shot": ("STRING", {"forceInput": True, "tooltip": (
                                 "Every Shot's shot JSON: which of those shots were refined (each judged on its own) "
                                 "and why any was left")})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename_prefix",)
    FUNCTION = "name"
    CATEGORY = CATEGORY
    DESCRIPTION = ("A clip's file name from Refine's own review. _REVIEW-FAIL: the print is wrong on most of the refined "
                   "frames (doubled, or the old print showing through), the label slid off the print it lies on, or it crept on a "
                   "still product. "
                   "_REVIEW-CHECK: other checks (light flicker, a step at the label's end, edges, ...) flag most of the "
                   "refined frames: look before delivering. They read approved clips as flagged too, so they don't "
                   "call a clip failed. The clip is always saved; with save_summary, the verdict and each check's "
                   "frames go in a JSON file beside it (a platform returns files, not text).")

    @classmethod
    def _unrefined(cls, not_covered, off=0):
        """Refine's runs of fixable frames left as generated (its report's not_covered), LEFT or more in a row, in clip
        frames: [(first, last)]."""
        out = []
        for x in not_covered or []:
            try:
                a, b = (int(v) for v in x["frames"])
            except (TypeError, ValueError, KeyError):
                continue
            if b - a >= cls.LEFT:
                out.append((a + off, b - 1 + off))
        return out

    @classmethod
    def verdict(cls, rv, keyframe=None, shots=None, every=None, not_covered=None):
        """-> (verdict PASS | CHECK | FAIL, reasons, frames per check) from Refine's review summary (and Find Keyframe
        Frame's report: the shots it found likely showing the product too; Every Shot's: which of them were refined).
        shots: Refine's report's shots (Every Shot: each refined on its own), each judged on its own review, the clip
        taking the worst."""
        if shots:
            off = int((every or {}).get("start") or 0)
            order = {"PASS": 0, "CHECK": 1, "FAIL": 2}
            worst, reasons, per = "PASS", [], {}
            for q in shots:
                a, b = (int(x) + off for x in q["frames"])
                v, rs, pr = cls._one((q.get("report") or {}).get("review") or {})
                worst = max(worst, v, key=order.get)
                reasons += [f"shot {a}-{b - 1}: {r}" for r in rs if v == "FAIL" or worst != "FAIL"]
                for k, ts in pr.items():
                    per.setdefault(k, []).extend(t + a for t in ts)
            also = cls._left(keyframe, every)
            if also:
                reasons.append(also)
                worst = "FAIL"
            gone = cls._unrefined(not_covered, off)
            if gone:
                reasons.append(cls._gone(gone, per))
                worst = "FAIL"
            if worst == "FAIL":
                reasons = [r for r in reasons if "flagged: look before delivering" not in r]
            return worst, reasons, per
        v, reasons, per = cls._one(rv)
        also = cls._left(keyframe, every)
        gone = cls._unrefined(not_covered, int((every or {}).get("start") or 0))
        if also or gone:
            reasons = ([] if v != "FAIL" else reasons) + ([also] if also else []) + \
                ([cls._gone(gone, per)] if gone else [])
            return "FAIL", reasons, per
        return v, reasons, per

    @staticmethod
    def _gone(gone, per):
        """The FAIL reason for fixable frames left as generated (and their frames under the check 'left')."""
        per.setdefault("left", []).extend(t for a, b in gone for t in range(a, b + 1))
        return ("left: the product shows with a readable print on frames " + ", ".join(f"{a}-{b}" for a, b in gone) +
                " but they are left as generated (no keyframe's label reaches them: the product lost and found "
                "again, or a run no shot covers)")

    @staticmethod
    def _left(keyframe, every):
        """The shots likely showing the product that this run leaves as generated, as a FAIL reason (or "")."""
        also = (keyframe or {}).get("also_likely") or []
        done = [x["shot"] for x in ((every or {}).get("shots") or [])]
        why = {tuple(x["shot"]): x.get("why", "") for x in ((every or {}).get("left") or [])}
        left = [s for s in also if not any(not (s[1] < d[0] or s[0] > d[1]) for d in done)]
        if not left:
            return ""
        # (V79: a pump bottle's last 49 frames, two shots past the keyframe's, kept their garbled print)
        return "the product likely shows on other shots too, left as generated: " + ", ".join(
            f"{a}-{b}" + (f" ({why[(a, b)][:200]})" if why.get((a, b)) else "") for a, b in left)

    @classmethod
    def _one(cls, rv):
        """One refined run's verdict from its review summary."""
        n, frames = int(rv.get("refined") or 0), rv.get("frames") or {}
        per = {}
        for t, ks in frames.items():
            for k in ks:
                per.setdefault(k, []).append(int(t))
        reasons = []
        wrong = sum(any(k in cls.WRONG for k in ks) for ks in frames.values())
        if n and wrong > cls.SHARE * n:
            reasons.append(f"the print wrong on {wrong} of {n} refined frames (" +
                           ", ".join(f"{k} {len(per[k])}" for k in cls.WRONG if k in per) + ")")
        if n and rv.get("kf_kept") is None and "ghost" in per and len(per["ghost"]) > cls.GHOST * n and \
                not wrong > cls.SHARE * n:
            # the old print through the new on more than GHOST of the frames (V79)
            reasons.append(f"ghost on {len(per['ghost'])} of {n} refined frames: the old print shows through the new")
        if rv.get("kf_old") is not None and float(rv["kf_old"]) > cls.KF_OLD:
            reasons.append(f"kf_old: the keyframe carried onto every frame left {100 * float(rv['kf_old']):.0f} % of the "
                           "area round its new print lit by the old print (it shows under the new letters)")
        for k in cls.HELD + cls.EDGE:
            if k in per:
                reasons.append(f"{k} on {len(per[k])} frames ({(rv.get('checks') or {}).get(k, '')})")
        if rv.get("carried") is not None and float(rv["carried"]) < cls.CARRIED:
            # the rendered band holds too little of the keyframe's corrected print: the rest stays garbled on every
            # frame (V74: a sleeve's band was its base, 0.45)
            reasons.append(f"carried: the clip renders {100 * float(rv['carried']):.0f} % of the keyframe's corrected "
                           "print (the rest of its lines stay garbled)")
        if rv.get("kf_kept") is not None and float(rv["kf_kept"]) < cls.KF_KEPT:
            # the keyframe's whole edit carried (V76): on the keyframe's own frame the clip must deliver the keyframe
            reasons.append(f"kf_kept: on the keyframe's own frame the clip delivers {100 * float(rv['kf_kept']):.0f} % "
                           "of the keyframe's edit (the rest keeps the source)")
        if rv.get("kf_kept") is not None:
            for k in cls.CARRIED_DEFECT:
                if n and len(per.get(k, ())) > cls.SHARE * n:
                    # the keyframe carried onto every frame (V76, V78): what's wrong on it is wrong on every frame
                    reasons.append(f"{k} on {len(per[k])} of {n} refined frames: the keyframe's own {k}, carried onto "
                                   "every frame")
        f = int(rv.get("flagged") or 0)
        if reasons:
            return "FAIL", reasons, per
        if n and f > cls.SHARE * n:
            return "CHECK", [f"{f} of {n} refined frames flagged: look before delivering"], per
        return "PASS", [], per

    def name(self, report, prefix, save_summary=True, keyframe_report=None, every_shot=None):
        try:
            full = json.loads(report)
            rv = full.get("review") or {}
        except (TypeError, ValueError, AttributeError):
            full, rv = {}, {}
        try:
            kr = json.loads(keyframe_report) if keyframe_report else {}
        except (TypeError, ValueError):
            kr = {}
        try:
            ev = (json.loads(every_shot) or {}).get("every_shot") if every_shot else None
        except (TypeError, ValueError, AttributeError):
            ev = None
        if isinstance(ev, dict) and ev.get("start") is None:
            ev = dict(ev, start=0)
        shots = full.get("shots")
        if full.get("found"):
            # runs found again inside a shot (V83): each judged on its own review, beside the shot's
            shots = (shots or [{"frames": [0, int(full.get("frames") or 0)], "report": {"review": rv}}]) + \
                [{"frames": q["frames"], "report": q.get("report") or {}} for q in full["found"]]
        v, reasons, per = self.verdict(rv, kr if isinstance(kr, dict) else {}, shots, ev, full.get("not_covered"))
        out = prefix + {"FAIL": "_REVIEW-FAIL", "CHECK": "_REVIEW-CHECK"}.get(v, "")
        n, f = int(rv.get("refined") or 0), int(rv.get("flagged") or 0)
        checks = ", ".join(f"{k} ({x})" for k, x in (rv.get("checks") or {}).items())
        text = (f"review: {v}, {f} of {n} refined frames flagged" + (f": {checks}" if checks else "") +
                ("".join(f"\n  {r}" for r in reasons)) + (f" -> {out.rpartition('/')[2]}" if v != "PASS" else ""))
        ui = {"text": [text]}
        if save_summary:
            from .nodes import _output_file
            path, fname, sub = _output_file(out + "_summary", "json")
            with open(path, "w") as fh:
                json.dump({"verdict": v, "reasons": reasons, "refined": n, "flagged": f, "checks": rv.get("checks") or {},
                           "frames_per_check": {k: len(x) for k, x in sorted(per.items())},
                           **{k: rv[k] for k in ("carried", "kf_kept", "kf_old") if rv.get(k) is not None},
                           **({"also_likely": kr["also_likely"]} if isinstance(kr, dict) and kr.get("also_likely") else {}),
                           **({"every_shot": {"refined": [x["shot"] for x in ev.get("shots") or []],
                                              "left": ev.get("left") or []}} if isinstance(ev, dict) else {}),
                           **({"shots": [{"frames": q["frames"], "verdict": self._one((q.get("report") or {}).get(
                               "review") or {})[0]} for q in shots]} if shots else {}),
                           "rule": {"FAIL": f"{'/'.join(self.WRONG)} on more than {self.SHARE:.0%} of the refined "
                                            f"frames, or ghost on more than {self.GHOST:.0%} (the label rendered), or kf_old over {self.KF_OLD:g} (the keyframe "
                                            f"carried), or the product likely on other shots left as generated, "
                                            f"or {self.LEFT} or more fixable frames in a row left as generated (left), "
                                            f"or any {'/'.join(self.HELD + self.EDGE)}, or (the keyframe carried) "
                                            f"{'/'.join(self.CARRIED_DEFECT)} on more than {self.SHARE:.0%}, or carried under "
                                            f"{self.CARRIED:g}, or kf_kept under {self.KF_KEPT:g}",
                                    "CHECK": f"any checks on more than {self.SHARE:.0%} of the refined frames"}},
                          fh, indent=2)
            ui["files"] = [{"filename": fname, "subfolder": sub, "type": "output"}]
        return {"ui": ui, "result": (out,)}


class ProductLabelAuditGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"keyframe": ("IMAGE", {"tooltip": "The refined keyframe the stills branch made"}),
                             "report": ("STRING", {"forceInput": True, "tooltip": "QA Sheet's report (its text)"})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("keyframe",)
    FUNCTION = "gate"
    CATEGORY = CATEGORY
    DESCRIPTION = ("The keyframe on to Refine (Video) only when its audit has no FAIL; else the run stops here with each "
                   "FAIL and why. A keyframe off the product's print (ink_place), cut along an occluder or squeezed "
                   "would be carried through every frame of the shot. The keyframe itself is still saved, to look at.")

    def gate(self, keyframe, report):
        fails = _audit_fails(report)
        if fails:
            raise ValueError("the keyframe's audit failed, so the clip isn't rendered (a wrong keyframe goes into every "
                             "frame): " + "; ".join(f"{f['check']}: {f['text']}" for f in fails) +
                             ". Look at qa/audit and the refined keyframe; fix it with Render the clip off, then queue "
                             "again.")
        return (keyframe,)


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


class ShotFailed:
    """What a node of another shot's run passes on instead of its outputs when it (or a node before it) stopped: that
    shot is left as generated, the others go on (Every Shot)."""
    def __init__(self, why):
        self.why = why

    def __repr__(self):
        return f"ShotFailed({self.why!r})"


GUARD = "ProductLabelGuarded/"


def _guarded(ct):
    """The class name of a guarded twin of node class ct, registered on first use: it runs ct's own function, and
    where that raises, or any input is a ShotFailed, it passes a ShotFailed on through every output (so a stop in one
    shot's run doesn't stop the job). Every Shot runs each other shot's nodes as these twins."""
    import nodes as core
    name = GUARD + ct
    if name in core.NODE_CLASS_MAPPINGS:
        return name
    base = core.NODE_CLASS_MAPPINGS[ct]
    try:
        from comfy_api.internal import _ComfyNodeInternal
        v3 = issubclass(base, _ComfyNodeInternal)
    except ImportError:
        v3 = False
    n_out = len(base.RETURN_TYPES)

    class Twin:
        DEPRECATED = True                    # kept out of the node search: only Every Shot places these
        RETURN_TYPES = tuple(base.RETURN_TYPES)
        RETURN_NAMES = tuple(getattr(base, "RETURN_NAMES", None) or base.RETURN_TYPES)
        FUNCTION = "run"
        OUTPUT_NODE = bool(getattr(base, "OUTPUT_NODE", False))
        CATEGORY = CATEGORY + "/guarded"

        @classmethod
        def INPUT_TYPES(cls):
            return base.INPUT_TYPES()

        def check_lazy_status(self, **kw):
            if any(isinstance(v, ShotFailed) for v in kw.values()) or v3 or not hasattr(base, "check_lazy_status"):
                return []
            try:
                return base().check_lazy_status(**kw)
            except Exception:
                return []

        def run(self, **kw):
            bad = next((v for v in kw.values() if isinstance(v, ShotFailed)), None)
            if bad is not None:
                return tuple([bad] * n_out)
            try:
                if v3:
                    o = base.execute(**kw)
                    r = getattr(o, "result", None) or getattr(o, "args", None) or o
                    return tuple(r)
                o = getattr(base(), base.FUNCTION)(**kw)
                return o
            except Exception as e:           # this shot's run stops here; the job goes on
                print(f"every shot WARN {ct} stopped: {str(e)[:300]}")
                return tuple([ShotFailed(f"{ct}: {e}")] * n_out)

    Twin.__name__ = "Guarded_" + "".join(ch if ch.isalnum() else "_" for ch in ct)
    core.NODE_CLASS_MAPPINGS[name] = Twin
    core.NODE_DISPLAY_NAME_MAPPINGS[name] = f"{ct} (guarded, Every Shot)"
    return name


MAX_SHOTS = 8                # Every Shot: the most other shots refined besides the keyframe's (more: left, named)


def _is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], (int, float))


class ProductLabelEveryShot:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "frames": ("IMAGE", {"tooltip": "The whole clip"}),
                    "keyframe_report": ("STRING", {"forceInput": True, "tooltip": (
                        "Find Keyframe Frame's report: every_shot, the other shots likely showing the product and each "
                        "one's keyframe frame")}),
                    "shot_frames": ("IMAGE", {"tooltip": "The keyframe frame's shot (Shot's frames)"}),
                    "shot_masks": ("MASK", {"tooltip": "The product's masks through that shot"}),
                    "keyframe": ("IMAGE", {"tooltip": "The keyframe going on to Refine (past the Audit Gate)"}),
                    "keyframe_index": ("INT", {"forceInput": True, "tooltip": "The keyframe frame in its shot"}),
                    "start": ("INT", {"forceInput": True, "tooltip": "Shot's start"}),
                    "shot": ("STRING", {"forceInput": True, "tooltip": "Shot's JSON"})},
                "optional": {
                    "template_keyframe": ("IMAGE", {"lazy": True, "tooltip": (
                        "The stills branch's refined frame, before the Audit Gate: what makes a keyframe from the "
                        "keyframe frame, run again for each other shot (never read itself)")}),
                    "template_report": ("STRING", {"lazy": True, "forceInput": True, "tooltip": (
                        "QA Sheet's report for that keyframe (its audit), run again for each other shot")})},
                "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "STRING", "INT", "STRING", "STRING", "MASK")
    RETURN_NAMES = ("frames", "masks", "keyframes", "keyframe_index", "keyframe_frames", "start", "shot", "shots",
                    "found_masks")
    FUNCTION = "every"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Every shot the product is in, for Refine (Video). Find Keyframe Frame names the other shots that likely show "
        "the product (long enough to hold a label) and each one's keyframe frame. For each, the graph from that frame "
        "to its keyframe and its masks runs again on that frame: SAM 3 on the frame, its shot between the cuts, SAM "
        "3 through it, the stills branch (Auto Draft, the text-only fix) and its audit. A shot whose keyframe fails "
        "its audit is left as generated (and named); the rest go on: the clip from the first shot's start to the last "
        "one's end, the masks of each shot (empty between them), the keyframes, their frames and the shots, so "
        "Refine refines each shot on its own from its own keyframe. One shot: its inputs pass on unchanged. Each "
        "other shot's keyframe is saved under shots/. A run where Find Keyframe Frame found the product lost and found "
        "again inside a shot (V83) gets its keyframe the same way and goes to Refine as a found run (shots: [start, "
        "end, \"found\"], its masks on found_masks): the shot itself is refined as before, the run fills the frames "
        "it leaves as they are.")

    def check_lazy_status(self, **kw):
        return []

    def every(self, frames, keyframe_report, shot_frames, shot_masks, keyframe, keyframe_index, start, shot,
              template_keyframe=None, template_report=None, dynprompt=None, unique_id=None):
        try:
            kr = json.loads(keyframe_report)
        except (TypeError, ValueError):
            kr = {}
        extras = ((kr.get("every_shot") or [])[1:] if isinstance(kr, dict) else [])[:MAX_SHOTS]
        one = (shot_frames, shot_masks, keyframe, keyframe_index, "auto", start, shot, "", None)
        if not extras:
            return {"ui": {"text": ["one shot: " + shot]}, "result": one}
        from comfy_execution.graph_utils import GraphBuilder
        me = dynprompt.get_node(unique_id)["inputs"]
        ends = {k: me.get(k) for k in ("template_keyframe", "template_report", "shot_masks", "shot")}
        if not all(_is_link(v) for v in ends.values()) or not _is_link(me.get("keyframe_report")):
            print("every shot WARN the templates aren't connected: only the keyframe's shot is refined")
            return {"ui": {"text": ["one shot (the templates aren't connected): " + shot]}, "result": one}
        src = me["keyframe_report"][0]               # Find Keyframe Frame: its frame output is each shot's frame
        # the template: every node between the keyframe frame and the ends (the keyframe, its audit, the shot's masks
        # and bounds); nodes that don't depend on the frame (the packshot, SAM 3, the settings) are shared
        up, dep = set(), {}

        def walk(n):
            if n in up:
                return
            up.add(n)
            for v in dynprompt.get_node(n)["inputs"].values():
                if _is_link(v):
                    walk(v[0])

        def depends(n):
            if n not in dep:
                dep[n] = False
                dep[n] = any(_is_link(v) and (v[0] == src or depends(v[0]))
                             for v in dynprompt.get_node(n)["inputs"].values())
            return dep[n]

        for v in ends.values():
            walk(v[0])
        clone = sorted(n for n in up if n != src and depends(n))
        g = GraphBuilder()
        gather = {"frames": me["frames"], "keyframe_report": me["keyframe_report"], "main_frames": me["shot_frames"],
                  "main_masks": me["shot_masks"], "main_keyframe": me["keyframe"], "main_shot": me["shot"]}
        for j, x in enumerate(extras, 1):
            k = int(x["keyframe_frame"])
            for n in clone:
                node = g.node(_guarded(dynprompt.get_node(n)["class_type"]), f"s{j}_{n}")
                node.set_override_display_id(n)
            for n in clone:
                node = g.lookup_node(f"s{j}_{n}")
                for name, v in dynprompt.get_node(n)["inputs"].items():
                    if _is_link(v) and v[0] == src:
                        node.set_input(name, k if int(v[1]) == 0 else v)
                    elif _is_link(v) and v[0] in clone:
                        node.set_input(name, g.lookup_node(f"s{j}_{v[0]}").out(int(v[1])))
                    else:
                        node.set_input(name, v)

            def out(key):
                v = ends[key]
                return g.lookup_node(f"s{j}_{v[0]}").out(int(v[1])) if v[0] in clone else v

            a, b = x["shot"]
            nm = g.node(_guarded("ProductLabelAuditName"), f"s{j}_name", report=out("template_report"),
                        prefix=f"product-label-refinement/shots/keyframe_{a}-{b}")
            g.node(_guarded("SaveImage"), f"s{j}_save", images=out("template_keyframe"), filename_prefix=nm.out(0))
            gather.update({f"keyframe_{j}": out("template_keyframe"), f"report_{j}": out("template_report"),
                           f"masks_{j}": out("shot_masks"), f"shot_{j}": out("shot")})
        gn = g.node("ProductLabelEveryShotGather", "gather", **gather)
        print(f"every shot {len(extras)} other shot(s): " + ", ".join(
            f"{x['shot'][0]}-{x['shot'][1]} from frame {x['keyframe_frame']}" for x in extras) +
              f" ({len(clone)} nodes run again for each)")
        return {"result": tuple(gn.out(i) for i in range(len(self.RETURN_TYPES))), "expand": g.finalize()}


class ProductLabelEveryShotGather:
    @classmethod
    def INPUT_TYPES(cls):
        opt = {}
        for j in range(1, MAX_SHOTS + 1):
            opt[f"keyframe_{j}"] = ("IMAGE",)
            opt[f"report_{j}"] = ("STRING", {"forceInput": True})
            opt[f"masks_{j}"] = ("MASK",)
            opt[f"shot_{j}"] = ("STRING", {"forceInput": True})
        return {"required": {"frames": ("IMAGE",), "keyframe_report": ("STRING", {"forceInput": True}),
                             "main_frames": ("IMAGE",), "main_masks": ("MASK",), "main_keyframe": ("IMAGE",),
                             "main_shot": ("STRING", {"forceInput": True})},
                "optional": opt}

    RETURN_TYPES = ProductLabelEveryShot.RETURN_TYPES
    RETURN_NAMES = ProductLabelEveryShot.RETURN_NAMES
    FUNCTION = "gather"
    CATEGORY = CATEGORY
    DESCRIPTION = "Every Shot's own: the shots' keyframes and masks put together for Refine (Video)."

    def gather(self, frames, keyframe_report, main_frames, main_masks, main_keyframe, main_shot, **kw):
        kr = json.loads(keyframe_report)
        ms = json.loads(main_shot)
        T = int(frames.shape[0])
        k0 = int(kr["keyframe_frame"])
        shots = [{"shot": [int(ms["start"]), int(ms["end"])], "keyframe_frame": k0, "masks": main_masks,
                  "keyframe": main_keyframe[:1], "from": "the keyframe's shot"}]
        left = [dict(x, why="not refined: " + x["why"]) for x in kr.get("not_refined") or []]
        found = []                           # runs found again inside a shot (V83): refined as found runs
        extras = (kr.get("every_shot") or [])[1:]
        for j, x in enumerate(extras, 1):
            if j > MAX_SHOTS:
                left.append({"shot": x["shot"], "why": f"over {MAX_SHOTS} other shots"})
                continue
            bad = next((kw.get(f"{k}_{j}") for k in ("keyframe", "report", "masks", "shot")
                        if isinstance(kw.get(f"{k}_{j}"), ShotFailed)), None)
            if bad is not None:
                left.append({"shot": x["shot"], "keyframe_frame": x["keyframe_frame"],
                             "why": "its run stopped: " + bad.why[:600]})
                continue
            if kw.get(f"keyframe_{j}") is None or kw.get(f"shot_{j}") is None:
                left.append({"shot": x["shot"], "why": "its keyframe wasn't made"})
                continue
            sj = json.loads(kw[f"shot_{j}"])
            a, b = int(sj["start"]), int(sj["end"])
            fails = _audit_fails(kw.get(f"report_{j}"))
            if fails:
                # a keyframe that fails its audit would go into every frame of its shot: the shot stays as generated
                left.append({"shot": [a, b], "keyframe_frame": x["keyframe_frame"], "why": "its keyframe's audit failed: " +
                             "; ".join(f"{f['check']}: {f['text']}" for f in fails)})
                continue
            again = x.get("from") == "lost and found again"
            inside = any(y["shot"][0] <= a and b <= y["shot"][1] for y in shots)
            if not (again and inside) and any(not (b < y["shot"][0] or a > y["shot"][1]) for y in shots + found):
                left.append({"shot": [a, b], "why": "overlaps a shot already refined"})
                continue
            m = kw.get(f"masks_{j}")
            m = m[None] if m is not None and m.dim() == 2 else m
            if m is None or m.shape[0] != b - a + 1:
                left.append({"shot": [a, b], "why": "its masks don't cover its frames"})
                continue
            (found if again and inside else shots).append({
                "shot": [a, b], "keyframe_frame": int(x["keyframe_frame"]), "masks": m,
                "keyframe": kw[f"keyframe_{j}"][:1],
                "from": "lost and found again inside the shot (V83)" if again and inside else
                        "its own keyframe frame (stills branch)"})
        shots.sort(key=lambda y: y["shot"][0])
        found.sort(key=lambda y: y["shot"][0])
        info = {"shots": [{k: v for k, v in y.items() if k in ("shot", "keyframe_frame", "from")}
                          for y in shots + found], "left": left, "clip_frames": T}
        for y in left:
            print(f"every shot WARN {y['shot'][0]}-{y['shot'][1]} left as generated: {y['why']}")
        if len(shots) == 1 and not found:
            # only the keyframe's shot: exactly the one-shot inputs
            text = json.dumps(dict(ms, every_shot=info))
            return {"ui": {"text": [text]}, "result": (main_frames, main_masks, main_keyframe, k0 - int(ms["start"]),
                                                       "auto", int(ms["start"]), text, "", None)}
        s0, e0 = shots[0]["shot"][0], max(y["shot"][1] for y in shots)
        n = e0 - s0 + 1
        masks = torch.zeros((n,) + tuple(main_masks.shape[-2:]), dtype=main_masks.dtype)
        for y in shots:
            a, b = y["shot"]
            masks[a - s0:b - s0 + 1] = y["masks"].to(masks.dtype)
        kfs = torch.cat([y["keyframe"].to(frames.dtype) for y in shots + found])
        rel = [[y["shot"][0] - s0, y["shot"][1] - s0 + 1] for y in shots] + \
              [[y["shot"][0] - s0, y["shot"][1] - s0 + 1, "found"] for y in found]
        fm = torch.cat([y["masks"].to(main_masks.dtype) for y in found]) if found else None
        info.update(start=s0, end=e0, frames=n)
        text = json.dumps({"start": s0, "end": e0, "frames": n, "keyframe_in_shot": -1, "clip_frames": T,
                           "cuts": ms.get("cuts", []), "every_shot": info})
        print("every shot refining " + ", ".join(f"{y['shot'][0]}-{y['shot'][1]} (keyframe {y['keyframe_frame']})"
                                                 for y in shots) +
              "".join(f", found again {y['shot'][0]}-{y['shot'][1]} (keyframe {y['keyframe_frame']})" for y in found) +
              f"; frames {s0}-{e0} go to Refine")
        return {"ui": {"text": [text]}, "result": (frames[s0:e0 + 1], masks, kfs, -1,
                                                   ", ".join(str(y["keyframe_frame"] - s0) for y in shots + found), s0,
                                                   text, json.dumps(rel), fm)}


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
    "ProductLabelWorkSize": ProductLabelWorkSize,
    "ProductLabelSwitch": ProductLabelSwitch,
    "ProductLabelRunIf": ProductLabelRunIf,
    "ProductLabelPrintType": ProductLabelPrintType,
    "ProductLabelPickFrame": ProductLabelPickFrame,
    "ProductLabelShot": ProductLabelShot,
    "ProductLabelFindKeyframe": ProductLabelFindKeyframe,
    "ProductLabelAuditGate": ProductLabelAuditGate,
    "ProductLabelReviewName": ProductLabelReviewName,
    "ProductLabelAuditName": ProductLabelAuditName,
    "ProductLabelSplice": ProductLabelSplice,
    "ProductLabelMaskOverlay": ProductLabelMaskOverlay,
    "ProductLabelPackshotMatte": ProductLabelPackshotMatte,
    "ProductLabelSAM3Points": ProductLabelSAM3Points,
    "ProductLabelSAM3Product": ProductLabelSAM3Product,
    "ProductLabelPasteBackMask": ProductLabelPasteBackMask,
    "ProductLabelFrameMask": ProductLabelFrameMask,
    "ProductLabelVideoSettings": ProductLabelVideoSettings,
    "ProductLabelTriage": ProductLabelTriage,
    "ProductLabelEveryShot": ProductLabelEveryShot,
    "ProductLabelEveryShotGather": ProductLabelEveryShotGather,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProductLabelLoadSource": "Product Label: Load Source (Image or Video)",
    "ProductLabelLoadOptional": "Product Label: Load Optional File",
    "ProductLabelWorkSize": "Product Label: Work Size (small stills resized up)",
    "ProductLabelSwitch": "Product Label: Switch",
    "ProductLabelRunIf": "Product Label: Run If",
    "ProductLabelPrintType": "Product Label: Print Type",
    "ProductLabelPickFrame": "Product Label: Pick Frame",
    "ProductLabelShot": "Product Label: Shot (between cuts)",
    "ProductLabelFindKeyframe": "Product Label: Find Keyframe Frame",
    "ProductLabelAuditGate": "Product Label: Audit Gate",
    "ProductLabelReviewName": "Product Label: Review Name",
    "ProductLabelAuditName": "Product Label: Audit Name",
    "ProductLabelSplice": "Product Label: Splice Shot Back",
    "ProductLabelMaskOverlay": "Product Label: Mask Overlay",
    "ProductLabelPackshotMatte": "Product Label: Packshot Matte",
    "ProductLabelSAM3Points": "Product Label: SAM 3 Points",
    "ProductLabelSAM3Product": "Product Label: SAM 3 Product (picked by the packshot)",
    "ProductLabelPasteBackMask": "Product Label: Paste-Back Mask",
    "ProductLabelFrameMask": "Product Label: Frame Mask",
    "ProductLabelVideoSettings": "Product Label: Video Settings",
    "ProductLabelTriage": "Product Label: Label Triage",
    "ProductLabelEveryShot": "Product Label: Every Shot (a keyframe per shot the product is in)",
    "ProductLabelEveryShotGather": "Product Label: Every Shot Gather",
}
