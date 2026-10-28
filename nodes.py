"""Product Label Refinement nodes: deterministic product compositing (the product-label-refinement engine) as ComfyUI nodes.

Data conventions
- `scene` is the full generated frame. The job's roi is the work box; nothing outside it changes.
- Every other IMAGE and MASK passed between Product Label Refinement nodes covers the work box only. Mask inputs also take
  a scene-sized mask (mask editor, segmentation) and crop it to the work box.
- Product layers (warped, graded, product) are sRGB like any IMAGE but keep highlight headroom above 1.0,
  which Match Finish rolls off. Don't clamp them in between.
- Conversion happens here, at the node boundary; stages.py runs the skill's own imgpost code in float64.
"""
import hashlib
import json
import os
import re

import numpy as np
import torch

JOB = "IMAGEPOST_JOB"
HOMOGRAPHY = "IMAGEPOST_HOMOGRAPHY"
COORDS = "IMAGEPOST_COORDS"   # reference coordinates of every work-box pixel, float64 (grade.shading reads them)
INK = "IMAGEPOST_INK"         # the job's matte 'ink': the grade's white field and ink colour (dark ink on paper), or
                              # Fill Leftovers' ink colour and matte (print on the body, light or dark); None otherwise
CATEGORY = "image/product-label-refinement"

COLOUR_CHOICES = ["job", "scene", "reference"]
SHADING_CHOICES = ["job", "off"]
RELIGHT_CHOICES = ["job", "off"]
MODEL_CHOICES = ["homography", "affine"]
LIGHT_CHOICES = ["relight", "cylinder", "off"]
LABEL_CHOICES = ["off", "cylinder", "flat"]
INK_CHOICES = ["off", "light", "dark"]


def _stages():
    from . import stages   # imgpost pulls in scipy: load it on first use, not at ComfyUI startup
    return stages


# ---- node boundary: ComfyUI tensors <-> the arrays run.py works on ----

def _frame(image, name):
    """IMAGE [1, H, W, C] -> float32 H x W x 3."""
    if image.dim() != 4:
        raise ValueError(f"{name}: expected an IMAGE [B, H, W, C], got shape {tuple(image.shape)}")
    if image.shape[0] != 1:
        raise ValueError(f"{name}: Product Label Refinement works on one image per job, got a batch of {image.shape[0]}")
    a = image[0].detach().to("cpu", torch.float32).numpy()
    return np.repeat(a, 3, 2) if a.shape[2] == 1 else a[..., :3]


def _u8(image, name):
    """IMAGE -> uint8. Recovers the file's exact values from LoadImage's x / 255."""
    return np.clip(np.round(_frame(image, name).astype(np.float64) * 255), 0, 255).astype(np.uint8)


def _box_view(arr, box, scene_hw, name):
    """Work-box-sized arrays pass through; scene-sized ones are cropped to the work box."""
    x0, y0, x1, y1 = box
    if arr.shape[:2] == (y1 - y0, x1 - x0):
        return arr
    if arr.shape[:2] == tuple(scene_hw):
        return arr[y0:y1, x0:x1]
    raise ValueError(f"{name} is {arr.shape[1]}x{arr.shape[0]}: expected the work box ({x1 - x0}x{y1 - y0}) "
                     f"or the scene ({scene_hw[1]}x{scene_hw[0]})")


def _layer(image, box, scene_hw, name):
    return _box_view(_frame(image, name), box, scene_hw, name).astype(np.float64)


def _mask(mask, box, scene_hw, name):
    m = mask.detach().to("cpu", torch.float32)
    if m.dim() == 4 and m.shape[1] == 1:
        m = m[:, 0]
    if m.dim() == 3:
        if m.shape[0] != 1:
            raise ValueError(f"{name}: Product Label Refinement works on one mask per job, got a batch of {m.shape[0]}")
        m = m[0]
    if m.dim() != 2:
        raise ValueError(f"{name}: expected a MASK [B, H, W], got shape {tuple(mask.shape)}")
    return _box_view(m.numpy(), box, scene_hw, name).astype(np.float64)


def _to_layer(lin):
    """Linear float64 -> sRGB layer, values above 1 kept (imgpost's lin_to_srgb only clips below 0)."""
    from .imgpost import images as im
    return im.lin_to_srgb(lin)


def _from_layer(srgb):
    """Inverse of _to_layer. imgpost's srgb_to_lin clips at 1, which would cut the highlight headroom."""
    s = np.clip(srgb, 0, None)
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def _image_out(arr):
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None]


def _mask_out(arr):
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None]


def _pil_out(im):
    return torch.from_numpy(np.asarray(im.convert("RGB")).astype(np.float32) / 255.0)[None]


def _placeholder():
    """Stands in for a QA image the run didn't make (no edges to show, no relight)."""
    return torch.full((1, 64, 64, 3), 0.1)


def _output_file(prefix, ext):
    """-> (absolute path, filename, subfolder) in ComfyUI's output folder, numbered like Save Image's files."""
    import folder_paths   # ComfyUI's module (tools/offline_run.py provides a stand-in)
    root = os.path.abspath(folder_paths.get_output_directory())
    sub, _, stem = prefix.strip().replace("\\", "/").rpartition("/")
    stem = stem or "product-label-refinement"
    folder = os.path.abspath(os.path.join(root, sub))
    if os.path.commonpath([root, folder]) != root:
        raise ValueError(f"filename_prefix {prefix!r} points outside ComfyUI's output folder")
    os.makedirs(folder, exist_ok=True)
    pat = re.compile(re.escape(stem) + r"_(\d{5})_\." + re.escape(ext))
    taken = [int(m.group(1)) for f in os.listdir(folder) if (m := pat.fullmatch(f))]
    name = f"{stem}_{max(taken, default=0) + 1:05}_.{ext}"
    return os.path.join(folder, name), name, sub


def _scene(job, scene, name="scene"):
    """-> (scene uint8, work box [x0, y0, x1, y1], (H, W))"""
    u8 = _u8(scene, name)
    Hs, Ws = u8.shape[:2]
    if job.get("scene_size") and list(job["scene_size"]) != [Ws, Hs]:
        w, h = job["scene_size"]
        raise ValueError(f"the job was measured on a {w}x{h} scene but `{name}` is {Ws}x{Hs}: "
                         "connect the image the job was measured on")
    return u8, _stages().work_box(job, Hs, Ws), (Hs, Ws)


def _reference(job, reference, reference_mask=None):
    """-> (reference uint8, alpha uint8 or None). reference_mask is LoadImage's MASK (1 = transparent)."""
    u8 = _u8(reference, "reference")
    if job.get("reference_size") and list(job["reference_size"]) != [u8.shape[1], u8.shape[0]]:
        w, h = job["reference_size"]
        raise ValueError(f"the job was measured on a {w}x{h} reference but `reference` is {u8.shape[1]}x{u8.shape[0]}")
    alpha = None
    if reference_mask is not None:
        m = reference_mask.detach().to("cpu", torch.float32)
        m = m[0] if m.dim() == 3 else m
        if tuple(m.shape) == u8.shape[:2]:   # LoadImage returns a 64x64 placeholder when the file has no alpha
            alpha = np.clip(np.round((1.0 - m.numpy().astype(np.float64)) * 255), 0, 255).astype(np.uint8)
    return u8, alpha


def _refmask(job, ref_u8, ref_a):
    try:
        return _stages().reference_mask(job, ref_u8, ref_a)
    except ValueError as e:
        if job["reference_outline"].get("type") == "alpha":
            raise ValueError(f"{e}: connect the reference LoadImage's MASK to `reference_mask`") from e
        raise


def _shape(job, box, coords):
    """shape 'generated': the generated outline's coverage and Warp Reference's bleed coverage (stages.fill and
    stages.composite), or None for the real outline"""
    if job.get("shape") != "generated":
        return None
    if coords is None or coords.get("bleed") is None:
        raise ValueError("the job's shape 'generated' needs Warp Reference's coords (the bled label's coverage): connect them")
    if list(coords["box"]) != list(box):
        raise ValueError(f"coords were computed for work box {coords['box']}, not {box}: re-run Warp Reference")
    return {"cover": _stages().generated_cover(job, box), "bleed": coords["bleed"],
            "edge_softness": job.get("finish", {}).get("edge_softness", 0.45)}


def _body(job, box, coords, key):
    """reference_outline.body: Warp Reference's body coverage ('body_raw' for the grade's fit masks, 'body' for the
    fill and the composite), or None"""
    if not job["reference_outline"].get("body"):
        return None
    if coords is None or coords.get(key) is None:
        raise ValueError("the job's reference_outline.body needs Warp Reference's coords (the body coverage): connect them")
    if list(coords["box"]) != list(box):
        raise ValueError(f"coords were computed for work box {coords['box']}, not {box}: re-run Warp Reference")
    return coords[key]


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _numbers(text):
    return [float(v) for v in _NUMBER.findall(text or "")]


def _points(text, name):
    """"x,y x,y ..." (any separators: spaces, commas, semicolons, new lines) -> [(x, y), ...]"""
    v = _numbers(text)
    if len(v) % 2:
        raise ValueError(f"{name}: {len(v)} numbers, expected x,y pairs")
    return list(zip(v[0::2], v[1::2]))


def _quad(text):
    """4 numbers: a box x0,y0,x1,y1 around the product; 8: TL TR BR BL corners. -> 4 corners or None"""
    v = _numbers(text)
    if not v:
        return None
    if len(v) == 4:
        x0, y0, x1, y1 = v
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if len(v) == 8:
        return list(zip(v[0::2], v[1::2]))
    raise ValueError(f"quad: {len(v)} numbers; give a box x0,y0,x1,y1 or 4 corners x,y (TL TR BR BL)")


_PAIR = re.compile(r"([-+]?[\d.]+)\s*,\s*([-+]?[\d.]+)\s*=\s*([-+]?[\d.]+)\s*,\s*([-+]?[\d.]+)")


def _pairs(text):
    """"rx,ry=sx,sy" entries (reference px = rough scene px) -> [((rx, ry), (sx, sy)), ...]"""
    text = text or ""
    rest = _PAIR.sub(" ", text)
    if re.search(r"[\d=]", rest):
        raise ValueError(f"pairs: can't read {rest.strip()!r}: give rx,ry=sx,sy entries")
    return [((float(a), float(b)), (float(c), float(d))) for a, b, c, d in _PAIR.findall(text)]


def _occluders(text):
    """One per line (or separated by ";"): "left: x,y x,y ..." (right, above, below: the side the occluder occupies)
    or "polygon: x,y ..."."""
    out = []
    for line in re.split(r"[\n;]", text or ""):
        if not line.strip():
            continue
        side, colon, pts = line.partition(":")
        side = side.strip().lower()
        if not colon or side not in ("left", "right", "above", "below", "polygon"):
            raise ValueError(f"occluders: {line.strip()!r}: start with left:, right:, above:, below: or polygon:")
        out.append((side, _points(pts, f"occluder {side}")))
    return out


def _outline(text):
    """4 numbers: a box in reference px; 6 or more: a polygon. -> box tuple, polygon list or None (traced)"""
    v = _numbers(text)
    if not v:
        return None
    if len(v) == 4:
        return tuple(v)
    return _points(text, "outline")


def _product_mask(mask, hw):
    """A scene-sized MASK (SAM, the mask editor) -> bool mask, or None for LoadImage's 64x64 placeholder or an empty
    mask."""
    if mask is None:
        return None, None
    m = mask.detach().to("cpu", torch.float32)
    if m.dim() == 4 and m.shape[1] == 1:
        m = m[:, 0]
    if m.dim() == 3:
        if m.shape[0] != 1:
            raise ValueError(f"product_mask: a batch of {m.shape[0]} masks; pick the one product to draft")
        m = m[0]
    if tuple(m.shape) == (64, 64) and tuple(hw) != (64, 64) and not bool((m > 0.5).any()):
        return None, None
    if tuple(m.shape) != tuple(hw):
        raise ValueError(f"product_mask is {m.shape[1]}x{m.shape[0]}: expected the scene's size ({hw[1]}x{hw[0]})")
    b = m.numpy() > 0.5
    if not b.any():
        return None, "product_mask is empty: ignored"
    return b, None


def _box_masks(job, box, hw, product_mask, old_silhouette, occluder_mask):
    a_new = _mask(product_mask, box, hw, "product_mask")
    old = _mask(old_silhouette, box, hw, "old_silhouette") > 0.5
    vis = 1.0 - _mask(occluder_mask, box, hw, "occluder_mask")
    return a_new, old, vis


# ---- nodes ----

class ImagePostAutoDraft:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene": ("IMAGE",),
                "reference": ("IMAGE",),
                "quad": ("STRING", {"default": "", "tooltip": (
                    "Where the product sits in the scene, roughly (scene px): TL TR BR BL corners of the packshot "
                    "outline's bounding box as x,y x,y x,y x,y, or a box x0,y0,x1,y1 around it. A few px off is fine. "
                    "Leave it empty when product_mask is connected and nothing stands in front of the product, or give 4 "
                    "landmark pairs instead. With label on: the generated label's own 4 corners, in the packshot "
                    "label's order (TL TR BR BL), within about 3 px.")}),
                "model": (MODEL_CHOICES, {"default": "homography", "tooltip": (
                    "homography: perspective. affine: no perspective (a small or flat-on product).")}),
                "light": (LIGHT_CHOICES, {"default": "relight", "tooltip": (
                    "relight: grade.relight on (cast shadows, falloff, glows). cylinder: the shading template for "
                    "cans and bottles instead (measure the highlight and add spec / shadow). off: neither.")}),
                "suggest": ("BOOLEAN", {"default": False, "tooltip": (
                    "Also add packshot corners that match the scene well as landmarks.")}),
                "filename_prefix": ("STRING", {"default": "product-label-refinement/draft", "tooltip": (
                    "The draft job.json is saved here inside ComfyUI's output folder, numbered like Save Image's files.")}),
            },
            "optional": {
                "product_mask": ("MASK", {"tooltip": (
                    "The generated product's mask, scene-sized (SAM 3, the mask editor): the outline is fitted to its "
                    "contour instead of to image edges. Without a quad the fit starts from the mask itself (its box, hull "
                    "corners and main axis): give the quad as well when something stands in front of the product.")}),
                "reference_mask": ("MASK", {"tooltip": "The reference Load Image's MASK: a transparent packshot's outline is its alpha"}),
                "pairs": ("STRING", {"default": "", "multiline": True, "tooltip": (
                    "Rough landmark pairs, reference px = scene px: rx,ry=sx,sy, one per line. Each is refined by colour "
                    "correlation and kept only when trusted.")}),
                "occluders": ("STRING", {"default": "", "multiline": True, "tooltip": (
                    "Objects in front of the product, one per line, snapped to their own edge: left: x,y x,y ... "
                    "(the occluder occupies that side: left, right, above, below) or polygon: x,y x,y ...")}),
                "outline": ("STRING", {"default": "", "tooltip": (
                    "The product on the packshot (reference px): a box x0,y0,x1,y1 or a polygon x,y x,y ... Empty: traced "
                    "from a packshot on a plain background (or its alpha). Give it when the packshot shows more than "
                    "the product. With label on: the label on the packshot, upright (needed).")}),
                "mask_skip": ("STRING", {"default": "", "tooltip": (
                    "A polygon in reference px whose outline isn't fitted to product_mask (where the mask runs on "
                    "into a part the job keeps, such as a cap).")}),
                "label": (LABEL_CHOICES, {"default": "off", "tooltip": (
                    "The label only (run.py draft --label). cylinder: a label on a bottle or jar, mapped round the "
                    "bottle (its axis and radius from the packshot's silhouette at the label's height). flat: a label "
                    "on a flat or flattened face, such as a tube's panels (model as set). The generated label's own "
                    "outline is measured and kept as its shape; its sides become the fit's edges. quad is then the "
                    "label's 4 corners, outline the label on the packshot, product_mask a mask of the label; light is "
                    "not used (the generated label's paper carries the light).")}),
                "ink": (INK_CHOICES, {"default": "off", "tooltip": (
                    "Print straight on a coloured body, no paper label. light: lighter than the body (white print on a "
                    "frosted bottle or tube). dark: darker than it (black print on a pink jar or a white bottle, and "
                    "its coloured bars): the print is also snapped onto where the generator drew it (its place, scale "
                    "and bow) and sharpened to its edges. The packshot's print is laid on the frame's own body in the "
                    "frame's own print colour (job matte ink). Run it with the ink template: product_mask also goes "
                    "into Fill Leftovers' old_silhouette, the matte the print stays inside.")}),
                "print_box": ("STRING", {"default": "", "tooltip": (
                    "With ink: the print's box on the packshot x0,y0,x1,y1 (reference px), from the first to the "
                    "last line of print with a few px to spare, inside the body's frosted rims, seams and lid edge. "
                    "Only print inside it is laid on, so the packshot's own highlights, seams and rims don't go on "
                    "as ink.")}),
                "bow": ("STRING", {"default": "", "tooltip": (
                    "A round body seen from above or below: its print bows like its seams (align.bow; needs "
                    "product_mask). t along the product's axis from the mask's centre, px, + down the image: "
                    "t:b given (-150:0 for a squeezed tube's flat crimp); t a seam measured across the body; t/w a "
                    "seam on a narrower ring of half-width w (a collar); t/matte the mask's own end (a base). "
                    "Example: \"-150:0 146.5\". print: the bow the generated product's own long lines draw (its "
                    "rules, a band's edges), less the warped packshot's, for a packshot warped whole (a squeeze "
                    "tube seen from above). Empty: no bow (seen from level).")}),
            },
        }

    RETURN_TYPES = (JOB, "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("job", "job_json", "draft_points", "report")
    FUNCTION = "draft"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "A first job from rough input, in place of Load Job (run.py draft): traces the packshot's outline, refines "
        "landmark pairs, proposes edge lines from the outline's straight sides where the generated edge agrees, "
        "snaps the old silhouette and the occluders to edges, and sets the work box, colour and relight. With "
        "product_mask (SAM 3) the outline is fitted to the mask. With label on, the label only: the generated "
        "label's own outline as its shape, its sides as the fit's edges, round the bottle (cylinder) or flat. With ink "
        "light or dark (and print_box, bow), print straight on a coloured body: the ink template makes a keyframe from the "
        "frame, its product mask and the packshot. On a clean scene the draft is close to final; on a cluttered shelf "
        "it is a scaffold to finish by hand. job_json and the saved file are the job to edit and "
        "load into Load Job; draft_points shows each landmark pair (packshot under the fit | scene)."
    )

    def draft(self, scene, reference, quad, model, light, suggest, filename_prefix, product_mask=None,
              reference_mask=None, pairs="", occluders="", outline="", mask_skip="", label="off", ink="off",
              print_box="", bow=""):
        su8 = _u8(scene, "scene")
        ru8, ra = _reference({}, reference, reference_mask)
        mask, note = _product_mask(product_mask, su8.shape[:2])
        pb = _numbers(print_box)
        if pb and len(pb) != 4:
            raise ValueError(f"print_box: {len(pb)} numbers; give x0,y0,x1,y1 in reference px")
        job, text, sheet = _stages().draft(su8, ru8, ra, quad=_quad(quad), pairs=_pairs(pairs),
                                           occluders=_occluders(occluders), mask=mask,
                                           mask_skip=_points(mask_skip, "mask_skip") or None, model=model,
                                           outline=_outline(outline), cylinder=light == "cylinder",
                                           relight=light == "relight", suggest=suggest,
                                           label=None if label == "off" else label,
                                           ink=None if ink == "off" else ink, print_box=pb or None, bow=bow.strip() or None)
        if note:
            text = f"note       {note}\n{text}"
        if ink in ("light", "dark"):
            text += (f"\nnote       ink {ink}: connect product_mask to Fill Leftovers' old_silhouette (the matte the "
                     "print stays inside) and Fill Leftovers' ink to Composite Behind")
        job_text = json.dumps(job, indent=2)
        path, name, sub = _output_file(filename_prefix, "json")
        with open(path, "w") as f:
            f.write(job_text + "\n")
        return {"ui": {"text": [text], "files": [{"filename": name, "subfolder": sub, "type": "output"}]},
                "result": (job, job_text, _pil_out(sheet) if sheet else _placeholder(), text)}


class ImagePostLoadJob:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_json": ("STRING", {"default": "", "multiline": True,
                                        "tooltip": "job.json text in the product-label-refinement skill's format. Ignored when job_path is set."}),
                "job_path": ("STRING", {"default": "",
                                        "tooltip": "Path to a job.json on the machine running ComfyUI. Overrides job_json."}),
            },
        }

    RETURN_TYPES = (JOB,)
    RETURN_NAMES = ("job",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Loads an product-label-refinement job (all measurements: work box, landmarks, edge lines, outlines, occluders, "
        "grade/finish/fill settings). The job's scene, reference and output paths are not read: images come "
        "from Load Image nodes and the result goes to a Save Image node."
    )

    @classmethod
    def IS_CHANGED(cls, job_json, job_path):
        if job_path.strip():
            try:
                with open(os.path.expanduser(job_path.strip()), "rb") as f:
                    return hashlib.sha256(f.read()).hexdigest()
            except OSError:
                return float("nan")
        return job_json

    def load(self, job_json, job_path):
        source = None
        if job_path.strip():
            source = os.path.abspath(os.path.expanduser(job_path.strip()))
            with open(source) as f:
                text = f.read()
        elif job_json.strip():
            text = job_json
        else:
            raise ValueError("give the job: paste job.json into job_json or set job_path")
        try:
            job = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"the job is not valid JSON: {e}") from e
        _stages().check_job(job)
        if source:
            job["_source"] = source
        return (job,)


class ImagePostJobMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job": (JOB,), "scene": ("IMAGE",)},
                "optional": {"product_mask": ("MASK", {"tooltip": "Warp Reference's product_mask: needed when the job's "
                                                       "old_silhouette is 'auto' (the fitted outline)"})}}

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("old_silhouette", "occluder_mask")
    FUNCTION = "masks"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Rasterises the job's old_silhouette (the generated product's outline) and occluders (objects in front, "
        "anti-aliased) over the work box. Any scene- or work-box-sized MASK can stand in for either output."
    )

    def masks(self, job, scene, product_mask=None):
        _, box, hw = _scene(job, scene)
        a_new = _mask(product_mask, box, hw, "product_mask") if product_mask is not None else None
        old, vis = _stages().scene_masks(job, box, a_new)
        return (_mask_out(old), _mask_out(1.0 - vis))


class ImagePostFitWarp:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job": (JOB,)}}

    RETURN_TYPES = (HOMOGRAPHY, "STRING")
    RETURN_NAMES = ("homography", "report")
    FUNCTION = "fit"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Fits the reference -> scene mapping (homography, affine, or a cylinder for labels on bottles, plus the "
        "optional residual edge correction) to the job's align points, y_only/x_only rows, edge lines and curves "
        "and limbs. Shows the residual table; accept visible edge lines at 1.5 px or less."
    )

    def fit(self, job):
        S = _stages()
        H, rep = S.fit_warp(job)
        return {"ui": {"text": [S.align_table(rep)]}, "result": (H, json.dumps({"align": rep}))}


class ImagePostWarpReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"job": (JOB,), "homography": (HOMOGRAPHY,), "scene": ("IMAGE",), "reference": ("IMAGE",)},
            "optional": {"reference_mask": ("MASK", {"tooltip": "The reference Load Image's MASK, for reference_outline type 'alpha'"})},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", COORDS)
    RETURN_NAMES = ("warped", "product_mask", "align_overlay", "coords")
    FUNCTION = "warp"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Warps the reference into the work box (supersampled, in linear light) and its real outline into the "
        "product mask. align_overlay: reference at 50% over the scene, real outline in green, occluders in red. "
        "coords: where each work-box pixel samples the reference (for grade.shading), and with shape 'generated' "
        "the bled label's coverage (for Fill Leftovers and Composite Behind), and with matte ink on the body the packshot's "
        "print as coverage (for Fill Leftovers; dark print registered onto the frame's own print and sharpened to it). "
        "Warns when the measured outline takes in the packshot's backdrop (a light rim in the composite)."
    )

    def warp(self, job, homography, scene, reference, reference_mask=None):
        su8, box, _ = _scene(job, scene)
        ru8, ra = _reference(job, reference, reference_mask)
        refmask = _refmask(job, ru8, ra)
        H = homography if hasattr(homography, "forward") else np.asarray(homography, float)
        W_lin, a_new, (U, V), overlay, info = _stages().warp(job, H, refmask, su8, ru8, box)
        _fr, warn = _stages().fringe(refmask, ru8, H) if not _stages().ik.on_body(job.get("matte")) else ([], "")
        result = (_image_out(_to_layer(W_lin)), _mask_out(a_new), _pil_out(overlay),
                  {"box": box, "u": U, "v": V, "squeeze": info["squeeze"], "bleed": info["bleed"], "cut": info["cut"],
                   "shape": info["shape"], "body": info["body"], "body_raw": info["body_raw"], "ink": info["ink"]})
        lines = [warn or ("outline clear of the packshot's backdrop" if not info["ink"] else
                          "outline: the band the print is read in (no outline shows with matte ink on the body)")] + \
            info["warnings"]
        if info["ink"]:
            r = info["ink"]["report"]
            pol = job["matte"].get("polarity", "light")
            rg = r.get("register")
            if rg:
                lines.append("register   " + (f"the print moved {rg['shift_px']} px, scale {rg['scale_xy']}, "
                                              f"{rg['rotation_deg']} deg" + (f", bow {rg['bow_px'][0]:+} to {rg['bow_px'][1]:+} px (top to bottom)" if "bow_px" in rg
                                                                             else "") +
                                              f" onto the frame's print (correlation {rg['ncc_before']} -> "
                                              f"{rg['ncc_after']})" if rg["kept"] else f"not applied: {rg['why']}"))
            lines.append(f"ink        {pol}: packshot {'closing' if pol == 'dark' else 'opening'} {r['open']} px, "
                         f"{r['print_px']:,} print px (coords carry the warped print for Fill Leftovers)")
        q = info["shape"]
        if q:
            lines.append(f"shape      the generated label's own outline: the real label reaches {q['real_past_shape_px']} px "
                         f"past it (cut there), it reaches {q['shape_past_real_px']} px past the real one (bled paper)" +
                         (f"; {q['unreached_px']} px the bleed can't reach (filled)" if q["unreached_px"] else ""))
        text = "\n".join(lines)
        return {"ui": {"text": [text]}, "result": result}


class ImagePostGradeToScene:
    # New grade options: port the branch from run.py into stages.grade; the job's grade section reaches it as is.
    # A switch on the node is a combo whose first choice is "job", appended after the existing ones (UI workflows
    # store widget values by position), like `colour` and `shading`.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job": (JOB,), "scene": ("IMAGE",), "warped": ("IMAGE",), "product_mask": ("MASK",),
                "old_silhouette": ("MASK",), "occluder_mask": ("MASK",),
                "colour": (COLOUR_CHOICES, {"default": "job", "tooltip": (
                    "job: grade.colour from the job (default scene). scene: tone curves fitted to the generated "
                    "product. reference: keep the real product's colours, taking only light, white balance and "
                    "black level from the scene (grade.white_box, black_box, ref_white_box in the job).")}),
                "shading": (SHADING_CHOICES, {"default": "job", "tooltip": (
                    "job: apply grade.shading from the job when it has one (cylinders: the generated product's "
                    "residual light across the reference x, plus optional limb darkening and specular streak). "
                    "off: skip it, to compare.")}),
            },
            "optional": {
                "reference": ("IMAGE", {"tooltip": "Needed when the job sets grade.ref_white_box"}),
                "coords": (COORDS, {"tooltip": "Warp Reference's coords. Needed when the job sets grade.shading"}),
                "relight": (RELIGHT_CHOICES, {"default": "job", "tooltip": (
                    "job: apply grade.relight from the job when it has one (the source's cast shadows, falloff, "
                    "glows, and the gloss or wrap haze with sheen). off: skip it, to compare.")}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "IMAGE", INK)
    RETURN_NAMES = ("graded", "grade_preview", "report", "relight_map", "ink")
    FUNCTION = "grade"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Carries the scene's light and colour onto the warped reference: a smooth light falloff (gain "
        "white_level, luma_ratio or none), then per-channel tone curves (colour scene) or the real product's own "
        "colours between scene white and black points (colour reference), then grade.shading for cylinders and "
        "grade.relight for the light the smooth grade misses (cast shadows, glows, gloss). relight_map shows "
        "where relight changed the light (blue darker, red brighter); a dark placeholder when it didn't run. "
        "ink: for a job with matte 'ink' (print only, over the scene's own surface), connect it to Composite Behind."
    )

    def grade(self, job, scene, warped, product_mask, old_silhouette, occluder_mask, colour, shading,
              reference=None, coords=None, relight="job"):
        su8, box, hw = _scene(job, scene)
        gs = dict(job.get("grade", {}))
        if colour != "job":
            gs["colour"] = colour
        if shading == "off":
            gs.pop("shading", None)
        if relight == "off":
            gs.pop("relight", None)
        a_new, old, vis = _box_masks(job, box, hw, product_mask, old_silhouette, occluder_mask)
        ru8 = _reference(job, reference)[0] if reference is not None else None
        if coords is not None and list(coords["box"]) != list(box):
            raise ValueError(f"coords were computed for work box {coords['box']}, not {box}: re-run Warp Reference")
        graded, rep, preview, sheet, ink = _stages().grade(gs, su8, ru8, box, _from_layer(_layer(warped, box, hw, "warped")),
                                                           a_new, old, vis, None if coords is None else coords["u"],
                                                           job.get("matte"), _body(job, box, coords, "body_raw"))
        return (_image_out(_to_layer(graded)), _pil_out(preview), json.dumps({"grade": rep}),
                _pil_out(sheet) if sheet else _placeholder(), ink)


class ImagePostMatchFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job": (JOB,), "scene": ("IMAGE",), "graded": ("IMAGE",), "product_mask": ("MASK",),
                             "old_silhouette": ("MASK",), "occluder_mask": ("MASK",)},
                "optional": {"coords": (COORDS, {"tooltip": "Warp Reference's coords: needed when the job sets reference_outline.body"})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("product", "report")
    FUNCTION = "finish"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Rolls off highlights, then matches the frame's softness (finish.blur, auto from edge steepness in "
        "blur_probe) and grain (finish.grain, auto from MAD noise in flat areas)."
    )

    def finish(self, job, scene, graded, product_mask, old_silhouette, occluder_mask, coords=None):
        su8, box, hw = _scene(job, scene)
        a_new, old, vis = _box_masks(job, box, hw, product_mask, old_silhouette, occluder_mask)
        card, rep = _stages().finish(job.get("finish", {}), job.get("grade", {}), su8, box,
                                     _from_layer(_layer(graded, box, hw, "graded")), a_new, old, vis,
                                     _body(job, box, coords, "body_raw"))
        return (_image_out(card), json.dumps({"finish": rep}))


class ImagePostFillLeftovers:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job": (JOB,), "scene": ("IMAGE",), "product_mask": ("MASK",),
                             "old_silhouette": ("MASK",), "occluder_mask": ("MASK",)},
                "optional": {"coords": (COORDS, {"tooltip": "Warp Reference's coords: needed when the job sets shape "
                                                       "'generated' or matte ink on the body"})}}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", INK)
    RETURN_NAMES = ("background", "fill_mask", "report", "ink")
    FUNCTION = "fill"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Harmonic (membrane) fill of the background where the generated product showed and the real one doesn't, "
        "with matched grain. Stops with FILL STOPPED when a leftover region has no background around it: extend "
        "reference_outline under the occluder or correct the occluder polyline. "
        "With matte ink on the body (print straight on a coloured body, light or dark): the frame's own garbled print "
        "is filled from its body near where the true print goes, inside old_silhouette (connect the product's mask: "
        "SAM 3, Load Image (as Mask)); background is the body without its print and ink, for Composite Behind, the "
        "print's colour."
    )

    def fill(self, job, scene, product_mask, old_silhouette, occluder_mask, coords=None):
        su8, box, hw = _scene(job, scene)
        a_new, old, vis = _box_masks(job, box, hw, product_mask, old_silhouette, occluder_mask)
        S = _stages()
        if S.ik.on_body(job.get("matte")):
            if coords is not None and list(coords["box"]) != list(box):
                raise ValueError(f"coords were computed for work box {coords['box']}, not {box}: re-run Warp Reference")
            filled, F, rep, ink = S.light_fill(job, su8, box, old, vis, None if coords is None else coords.get("ink"))
            return (_image_out(_to_layer(filled)), _mask_out(F), json.dumps({"matte": rep}), ink)
        bd = _body(job, box, coords, "body")
        base, F, rep = _stages().fill(job.get("fill", {}), su8, box, a_new, old, vis, job.get("matte"),
                                      _shape(job, box, coords),
                                      None if bd is None else {"cover": bd, "erode": job.get("grade", {}).get("erode", 4)})
        return (_image_out(base), _mask_out(F), json.dumps({"fill": rep}), None)


class ImagePostCompositeBehind:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"job": (JOB,), "scene": ("IMAGE",), "background": ("IMAGE",), "product": ("IMAGE",),
                             "product_mask": ("MASK",), "occluder_mask": ("MASK",)},
                "optional": {"ink": (INK, {"tooltip": "The job's matte 'ink': Grade To Scene's ink (dark ink on paper), "
                                                     "or Fill Leftovers' ink (print on the body, light or dark)"}),
                             "coords": (COORDS, {"tooltip": "Warp Reference's coords: needed when the job sets shape 'generated'"})}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "matte")
    FUNCTION = "composite"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Composites the product over the filled background behind the occluders (matte = product mask softened by "
        "finish.edge_softness, times visibility) and pastes the work box into the scene, quantised to 8 bit like "
        "the skill's output. Pixels outside the work box are passed through untouched. With matte ink on the body: "
        "the packshot's print in Fill Leftovers' ink colour over its background, inside the product's matte only "
        "(product isn't read)."
    )

    def composite(self, job, scene, background, product, product_mask, occluder_mask, ink=None, coords=None):
        su8, box, hw = _scene(job, scene)
        if _stages().ik.on_body(job.get("matte")):
            # print on the body: background is Fill Leftovers' body (its print removed); product isn't read
            out, alpha = _stages().light_composite(su8, box, _from_layer(_layer(background, box, hw, "background")), ink)
            x0, y0, x1, y1 = box
            image = scene[:1].detach().to("cpu", torch.float32).clone()
            image[0, y0:y1, x0:x1, :3] = torch.from_numpy(out.astype(np.float32) / 255.0)
            return (image, _mask_out(alpha))
        out, alpha = _stages().composite(job.get("finish", {}), _layer(background, box, hw, "background"),
                                         _layer(product, box, hw, "product"),
                                         _mask(product_mask, box, hw, "product_mask"),
                                         1.0 - _mask(occluder_mask, box, hw, "occluder_mask"), ink, job.get("matte"),
                                         _shape(job, box, coords), _body(job, box, coords, "body"))
        x0, y0, x1, y1 = box
        image = scene[:1].detach().to("cpu", torch.float32).clone()
        image[0, y0:y1, x0:x1, :3] = torch.from_numpy(out.astype(np.float32) / 255.0)
        return (image, _mask_out(alpha))


class ImagePostQASheet:
    @classmethod
    def INPUT_TYPES(cls):
        report = ("STRING", {"forceInput": True})
        return {
            "required": {"job": (JOB,), "before": ("IMAGE",), "after": ("IMAGE",), "reference": ("IMAGE",),
                         "matte": ("MASK",), "fill_mask": ("MASK",)},
            "optional": {"reference_mask": ("MASK",), "align_report": report, "grade_report": report,
                         "finish_report": report, "fill_report": report,
                         "old_silhouette": ("MASK", {"tooltip": "Job Masks' old_silhouette: with it the run audits itself"}),
                         "occluder_mask": ("MASK",), "coords": (COORDS, {"tooltip": "Warp Reference's coords (squeeze check)"}),
                         "ink": (INK, {"tooltip": "Fill Leftovers' ink (matte ink on the body): with it the audit also "
                                                  "says where the keyframe changed pixels away from print, and whether "
                                                  "the packshot is too small for the frame"})},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("compare", "edges", "before_after", "report", "audit")
    FUNCTION = "qa"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "QA sheets: compare (before, after, reference), edges (4x before|after tiles along every boundary and "
        "fill) and before_after, plus the run's report.json (stage reports merged, change counts including "
        "outside_roi_changed). With old_silhouette (and occluder_mask, coords) it also audits the run: a missed "
        "occluder, a rim of the generated product, a thick fill, print pressed into a bottle's edge, edges off. A FAIL "
        "blocks delivery; look at every WARN tile on the audit sheet."
    )

    def qa(self, job, before, after, reference, matte, fill_mask, reference_mask=None,
           align_report=None, grade_report=None, finish_report=None, fill_report=None,
           old_silhouette=None, occluder_mask=None, coords=None, ink=None):
        S = _stages()
        su8, box, hw = _scene(job, before, "before")
        res = _u8(after, "after")
        if res.shape != su8.shape:
            raise ValueError(f"`after` is {res.shape[1]}x{res.shape[0]} but `before` is {su8.shape[1]}x{su8.shape[0]}")
        ru8, ra = _reference(job, reference, reference_mask)
        rep = {"job": job["_source"]} if job.get("_source") else {}
        rep.update({"roi": box, "stage": "all"})
        for part in (align_report, grade_report, finish_report, fill_report):
            if part:
                rep.update(json.loads(part))
        if coords is not None and coords.get("shape"):
            rep["shape"] = coords["shape"]
        rep["change"] = S.change_report(res, su8, box)
        before_after, compare, tiles = S.qa_sheets(res, su8, ru8, _refmask(job, ru8, ra),
                                                   _mask(matte, box, hw, "matte"),
                                                   _mask(fill_mask, box, hw, "fill_mask") > 0.5, box)
        edges = _pil_out(tiles) if tiles else _placeholder()
        audit_img, verdict = _placeholder(), ""
        if old_silhouette is not None:
            vis = 1.0 - _mask(occluder_mask, box, hw, "occluder_mask") if occluder_mask is not None else np.ones(
                (box[3] - box[1], box[2] - box[0]))
            found, sheet, verdict = S.audit(res, su8, _mask(matte, box, hw, "matte"), _mask(old_silhouette, box, hw,
                                            "old_silhouette") > 0.5, vis, _mask(fill_mask, box, hw, "fill_mask") > 0.5,
                                            box, rep.get("align"), None if coords is None else coords.get("squeeze"),
                                            None if coords is None else coords.get("cut"), S.ink_mode(job.get("matte")),
                                            job.get("shape") == "generated")
            if S.ik.on_body(job.get("matte")):
                pol = job["matte"].get("polarity", "light")
                for f in found:
                    if f["check"] == "fill":    # the garbled print is as thick as the generated letters, under the new print
                        f.update(level="INFO", text=f["text"] + " (the garbled print, under the new print: see before_after)")
                    elif f["check"] == "edges":     # no outline is composited: the lines only place the print
                        f.update(level="INFO", text=f["text"] + f" (matte ink {pol}: no outline shows; check the print's "
                                                                "placement)")
                    elif f["check"] == "leftover" and pol == "dark":   # the new edge is a letter's, and the body
                        f.update(level="INFO", text=f["text"] + " (matte ink dark: the body round the print is the "
                                                                "generated one)")
                if isinstance(ink, dict) and ink.get("polarity") in S.ik.POLARITIES:
                    found += S.light_check(job, res, su8, ink, box)
                sheet = S.au.sheet(su8[box[1]:box[3], box[0]:box[2]], res[box[1]:box[3], box[0]:box[2]], found,
                                   (box[0], box[1])) if found else None
                worst = "FAIL" if any(f["level"] == "FAIL" for f in found) else "WARN" if any(
                    f["level"] == "WARN" for f in found) else "clean"
                verdict = "\n".join([f"audit      {f['level']:4s} {f['check']}: {f['text']}" +
                                      (f" at {f['box']}" if f.get("box") else "") for f in found if f["level"] != "INFO"] +
                                     [f"audit      {worst}" + (": look at the audit sheet before delivering"
                                                              if worst != "clean" else "")])
            rep["audit"] = found
            rep["audit_verdict"] = verdict
            audit_img = _pil_out(sheet) if sheet else _placeholder()
        text = json.dumps(rep, indent=2)
        return {"ui": {"text": [text]}, "result": (_pil_out(compare), edges, _pil_out(before_after), text, audit_img)}


class ImagePostSavePSD:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "job": (JOB,), "scene": ("IMAGE",), "image": ("IMAGE",), "background": ("IMAGE",), "fill_mask": ("MASK",),
            "product": ("IMAGE",), "matte": ("MASK",),
            "filename_prefix": ("STRING", {"default": "product-label-refinement/refined", "tooltip": (
                "Folder and name inside ComfyUI's output folder; files are numbered like Save Image's.")}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Writes the result as a layered PSD for finishing by hand (run.py --psd): the original scene, the "
        "background fill (opaque only where the fill ran) and the product with matte x occluders as its layer "
        "mask. With every layer visible it flattens to the PNG within 2 levels (report: layers_vs_png). "
        "scene: the frame before this job; image: Composite Behind's image."
    )

    def save(self, job, scene, image, background, fill_mask, product, matte, filename_prefix):
        su8, box, hw = _scene(job, scene)
        res = _u8(image, "image")
        if res.shape != su8.shape:
            raise ValueError(f"`image` is {res.shape[1]}x{res.shape[0]} but `scene` is {su8.shape[1]}x{su8.shape[0]}")
        path, name, sub = _output_file(filename_prefix, "psd")
        rep = _stages().write_psd(path, su8, res, _layer(background, box, hw, "background"),
                                  _mask(fill_mask, box, hw, "fill_mask") > 0.5, _layer(product, box, hw, "product"),
                                  _mask(matte, box, hw, "matte"), box)
        rep["path"] = f"{sub}/{name}" if sub else name   # inside ComfyUI's output folder
        text = json.dumps({"psd": rep}, indent=2)
        return {"ui": {"text": [text], "files": [{"filename": name, "subfolder": sub, "type": "output"}]}}


NODE_CLASS_MAPPINGS = {
    "ImagePostLoadJob": ImagePostLoadJob,
    "ImagePostAutoDraft": ImagePostAutoDraft,
    "ImagePostJobMasks": ImagePostJobMasks,
    "ImagePostFitWarp": ImagePostFitWarp,
    "ImagePostWarpReference": ImagePostWarpReference,
    "ImagePostGradeToScene": ImagePostGradeToScene,
    "ImagePostMatchFinish": ImagePostMatchFinish,
    "ImagePostFillLeftovers": ImagePostFillLeftovers,
    "ImagePostCompositeBehind": ImagePostCompositeBehind,
    "ImagePostQASheet": ImagePostQASheet,
    "ImagePostSavePSD": ImagePostSavePSD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePostLoadJob": "Product Label: Load Job",
    "ImagePostAutoDraft": "Product Label: Auto Draft",
    "ImagePostJobMasks": "Product Label: Job Masks",
    "ImagePostFitWarp": "Product Label: Fit Warp",
    "ImagePostWarpReference": "Product Label: Warp Reference",
    "ImagePostGradeToScene": "Product Label: Grade To Scene",
    "ImagePostMatchFinish": "Product Label: Match Finish",
    "ImagePostFillLeftovers": "Product Label: Fill Leftovers",
    "ImagePostCompositeBehind": "Product Label: Composite Behind",
    "ImagePostQASheet": "Product Label: QA Sheet",
    "ImagePostSavePSD": "Product Label: Save PSD",
}
