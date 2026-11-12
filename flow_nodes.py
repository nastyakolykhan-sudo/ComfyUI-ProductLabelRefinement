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
    ALSO = 0.5                               # ... this share: reported as likely showing the product too (not fixed)
    WHOLE = 0.90                             # a frame whose product outline matches the packshot's under this share of
                                             # the shot's (90th percentile) shows part of it only (a hand, a holder)
    PARTIAL = 0.6                            # ... and keypick's score is multiplied by this (as an occluded frame)

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
        # the shot's masks: the given ones, else SAM 3 tracked through it from its best probe
        if mask_given:
            sm = masks[s:e + 1]
        else:
            from .video_nodes import ProductLabelVideoSAM3BothWays
            kt = top["best"] - s
            km = torch.from_numpy(pm[probes.index(top["best"])].astype("float32"))[None]
            sm = ProductLabelVideoSAM3BothWays().track(frames[s:e + 1], model, km, kt, conditioning=conditioning,
                                                       detection_threshold=0.33, max_objects=1, detect_interval=1)[0]
        F = _u8(frames[s:e + 1])
        M = sm.detach().cpu().numpy() > 0.5
        n = e - s + 1
        F2, M2 = ash.downscale(F, 2), ash.downscale(M, 2)
        stats, bnds, _sh, _kinds, _why = ash.analyse(F2, M2)
        # the candidates: the frames where the product is in view. A shot can show it for part of its length only (it
        # slides behind something, into a box, out of the frame); there the track is empty and nothing is measured
        view = [t for t in range(n) if "width_px" in stats[t] and stats[t]["area"] >= self.MIN_AREA]
        if not view:
            raise ValueError(f"SAM 3's track through shot {s}-{e} doesn't hold the product on any frame: check the SAM 3 "
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
               "in_view": {"first": s + view[0], "last": s + view[-1], "frames": len(view), "of": n},
               "pick": {**best, "t": k, "best_scored": s + int(ranking[0]["t"]),
                        "tied": [s + int(r["t"]) for r in sorted(near, key=lambda r: r["t"])]},
               "shots": shots_rep, "warnings": warn}
        text = json.dumps(rep, default=float)
        print(f"keyframe   frame {k} (shot {s}-{e}, the product in view on {len(view)} of its {n} frames; "
              f"score {top['score']:.2f}; " +
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
    CARRIED = 0.6                # under this share of the keyframe's corrected print in the rendered band: FAIL (V74)
    KF_KEPT = 0.9                # under this share of the keyframe's edit delivered on its own frame (the edit carry,
                                 # V76: a surface short of a shoulder left its line ends garbled; carried 0.97): FAIL
    HELD = ("slide", "creep", "fade")   # the label off the print it lies on, held (the check's own run), creeping on
                                        # a still product (V71: the user saw 0.6 px over a still shot), or fading over
                                        # the old print where the product leaves right after (V72): any, FAIL

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"report": ("STRING", {"forceInput": True, "tooltip": "Refine (Video)'s report (its text)"}),
                             "prefix": ("STRING", {"default": "product-label-refinement/video/refined"}),
                             "save_summary": ("BOOLEAN", {"default": True, "tooltip": (
                                 "Save the review's verdict and checks as a JSON file under the name (one of the "
                                 "clip's two names is enough)")})}}

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
    def verdict(cls, rv):
        """-> (verdict PASS | CHECK | FAIL, reasons, frames per check) from Refine's review summary"""
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
        for k in cls.HELD:
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
        f = int(rv.get("flagged") or 0)
        if reasons:
            return "FAIL", reasons, per
        if n and f > cls.SHARE * n:
            return "CHECK", [f"{f} of {n} refined frames flagged: look before delivering"], per
        return "PASS", [], per

    def name(self, report, prefix, save_summary=True):
        try:
            rv = json.loads(report).get("review") or {}
        except (TypeError, ValueError, AttributeError):
            rv = {}
        v, reasons, per = self.verdict(rv)
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
                           **{k: rv[k] for k in ("carried", "kf_kept") if rv.get(k) is not None},
                           "rule": {"FAIL": f"{'/'.join(self.WRONG)} on more than {self.SHARE:.0%} of the refined "
                                            f"frames, or any {'/'.join(self.HELD)}, or carried under "
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
}
