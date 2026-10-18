"""Flow nodes for the one workflow that covers every case (workflows/product-label-refinement-all.*): one input for an
image or a clip, optional files, and the switches that run only the branch a job needs.

- Load Source takes an image or a video from ComfyUI's input folder and says which it is (is_video).
- Load Optional is a file that may be left at "(none)": a product mask (else SAM 3), a corrected keyframe (else the
  stills branch makes it). `given` says whether it was set.
- Switch passes one of two lazy inputs on, so the other branch never runs. Run If passes its input on or blocks
  everything downstream of it, output nodes included (a still never reaches the video nodes).
- Print Type reads the job: light ink (print on the body) or paper / dark ink, the two stills graphs.
- Pick Frame takes the keyframe's frame (and mask) out of the clip; Paste-Back Mask is where a run changed pixels, so
  a straightened crop goes back into its frame only there (and not at all when nothing was straightened).
- Video Settings adds the product's copy to Refine (Video)'s settings JSON as `words`.
- Label Triage shows the product's print next to the packshot's, straightened and at the same size, for the agent's
  call: fix or fine. A fine product skips the fix (the workflow's "Fix this product" boolean).

Images and videos are decoded by ComfyUI's own loaders (Load Image, and Load Video's VideoFromFile), so the pixels are
the ones the separate templates get from the same files.
"""
import json
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
    return bool(folder_paths.filter_files_content_types([name], ["video"]))


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
    DESCRIPTION = ("light: the job lays light print straight on the body (matte ink, polarity light: Auto Draft's ink "
                   "light). Its graph skips Grade To Scene and Match Finish and keeps the print inside the product's "
                   "mask. Otherwise paper or dark ink, the graded composite.")

    def route(self, job):
        light = bool(_stages().ik.light(job.get("matte")))
        return (light, "light ink on the body" if light else "paper or dark ink")


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


class ProductLabelPasteBackMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"before": ("IMAGE", {"tooltip": "The crop as it went in (Crop & Rotate's image)"}),
                             "after": ("IMAGE", {"tooltip": "The crop as it came out (Composite Behind's image)"}),
                             "crop_rotate": ("CROP_ROTATE",),
                             "grow": ("INT", {"default": 2, "min": 0, "max": 64, "tooltip": "px added round the change"})}}

    RETURN_TYPES = ("MASK", "BOOLEAN")
    RETURN_NAMES = ("mask", "straightened")
    FUNCTION = "mask"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Where a straightened crop goes back into its frame: 1 where the run changed it (8 bit, any channel), "
                   "grown by `grow` px, for Crop & Rotate Composite's mask, so nothing else is resampled. straightened "
                   "is false when the crop is the whole frame, unturned: then the result is the frame itself and no "
                   "paste-back is needed.")

    def mask(self, before, after, crop_rotate, grow):
        if before.shape[:3] != after.shape[:3]:
            raise ValueError(f"before is {tuple(before.shape[:3])}, after {tuple(after.shape[:3])}: the same size needed")
        b = torch.round(before[..., :3].float().clamp(0, 1) * 255)
        a = torch.round(after[..., :3].float().clamp(0, 1) * 255)
        m = (a != b).any(-1).float()
        if grow > 0:
            m = torch.nn.functional.max_pool2d(m[:, None], 2 * grow + 1, stride=1, padding=grow)[:, 0]
        c = crop_rotate
        whole = (c["width"] == c["source_width"] and c["height"] == c["source_height"]
                 and c["center_x"] * 2 == c["source_width"] and c["center_y"] * 2 == c["source_height"])
        return (m, not (whole and c["angle"] % 360 == 0))


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
    "ProductLabelPasteBackMask": ProductLabelPasteBackMask,
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
    "ProductLabelPasteBackMask": "Product Label: Paste-Back Mask",
    "ProductLabelVideoSettings": "Product Label: Video Settings",
    "ProductLabelTriage": "Product Label: Label Triage",
}
