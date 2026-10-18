"""Product Label video nodes: a video's hallucinated product label replaced by the true one, deterministic and
frame-exact.

Product Label: Refine (Video) has the same contract as the engine's command line (video.py --refine) on files:
  in   frames (IMAGE batch), the product's mask (MASK batch, same count and size), corrected keyframe(s) (IMAGE)
  out  frames (same count, size and order: outside the changed pixels the input tensor itself, bit for bit),
       matte (the mask plus any frames it patched), changed (exactly the pixels that changed), report (JSON),
       qa_image (the QA plot)
Full frames (recommended) or a workflow's crops: the contract is the same. A keyframe corrected on a zoomed crop comes
back onto its frame through keyframe_bbox (ImageCropByMaskToAspect's bbox) and keyframe_index.

The engine (imgvideo and imgauto here, next to the stills nodes' imgpost) is a byte-identical copy of the skill's, pinned
in imgpost.lock.json (tools/sync_imgpost.py for imgpost, tools/sync_video_engine.py for imgvideo and imgauto). The point tracker is AllTracker (MIT, code vendored in imgvideo/alltracker);
its weights come from models/trackers/ and are never downloaded here.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import folder_paths
except ImportError:                          # outside ComfyUI (tools/offline_run.py, tools/build_workflows.py)
    folder_paths = None

PACK = Path(__file__).resolve().parent
CATEGORY = "image/product-label-refinement/video"
if not hasattr(folder_paths, "models_dir"):  # (a stand-in folder_paths, as tools/offline_run.py installs)
    folder_paths = None
TRACKERS_DIR = os.path.join(folder_paths.models_dir, "trackers") if folder_paths else None


def _tracker_dirs():
    """models/trackers, plus a `trackers` folder beside every place the standard model folders come from. A server
    that mounts its models from a volume through extra_model_paths.yaml (main.py loads it before custom nodes) lists
    checkpoints, sams, ... there but not this new folder type, so the volume's trackers/ would otherwise go unseen."""
    dirs = [TRACKERS_DIR]
    for kind in ("checkpoints", "sams", "diffusion_models", "loras", "vae", "upscale_models", "text_encoders"):
        try:
            paths = folder_paths.get_folder_paths(kind)
        except Exception:                    # (a folder type this ComfyUI doesn't know)
            continue
        dirs += [os.path.join(os.path.dirname(os.path.normpath(p)), "trackers") for p in paths]
    return list(dict.fromkeys(dirs))


if folder_paths:
    for _d in _tracker_dirs():
        folder_paths.add_model_folder_path("trackers", _d)


def _engine():
    """The vendored engine as top-level packages (imgvideo imports `imgpost` by name: the same files the stills nodes
    import relatively), loaded on first use."""
    if str(PACK) not in sys.path:
        sys.path.insert(0, str(PACK))
    import imgauto
    import imgpost
    import imgvideo
    for mod in (imgpost, imgvideo, imgauto):
        if Path(mod.__file__).resolve().parent.parent != PACK:
            raise RuntimeError(f"another copy of {mod.__name__} is already loaded ({mod.__file__}); the video nodes need "
                               f"this pack's pinned copy in {PACK}")
    from imgvideo import pipeline
    return pipeline


def _trackers():
    try:
        names = [f for f in folder_paths.get_filename_list("trackers") if f.lower().startswith("alltracker") and f.endswith(".pth")]
    except Exception:                        # (no folder_paths outside ComfyUI)
        names = []
    return names or ["alltracker.pth (missing: put it in models/trackers)"]


def _u8(images):
    """IMAGE [B, H, W, C] -> uint8 B x H x W x 3; exact for frames decoded from 8 bits (x / 255)."""
    a = images.detach().to("cpu", torch.float32).numpy()[..., :3].astype(np.float64)
    return np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)


def _keyframes_to_frames(kf, frames, masks, bbox, index, kfr):
    """Keyframes the frames' size pass through; zoomed crops go back onto their frames through the crop windows."""
    T, H, W = frames.shape[:3]
    if kf.shape[1:3] == (H, W):
        return list(kf), kfr
    if bbox is None:
        raise ValueError(f"the keyframes are {kf.shape[2]} x {kf.shape[1]}, the frames {W} x {H}: connect keyframe_bbox "
                         "(the crop windows the keyframes were cut with, e.g. ImageCropByMaskToAspect's bbox)")
    if any(k is None for k in kfr):
        raise ValueError("a cropped keyframe needs its frame: set keyframe_index (one keyframe) or keyframe_frames")
    from PIL import Image
    from scipy.ndimage import binary_dilation
    out = []
    for crop, t in zip(kf, kfr):
        x0, y0, x1, y1 = (int(round(v)) for v in bbox[t])
        full = frames[t].copy()
        patch = np.asarray(Image.fromarray(crop).resize((x1 - x0, y1 - y0), Image.LANCZOS))
        keep = binary_dilation(masks[t], iterations=8)[y0:y1, x0:x1]       # the corrected product, not the crop's backdrop
        region = full[y0:y1, x0:x1]
        region[keep] = patch[keep]
        out.append(full)
    return out, kfr


def _parse_frames(text, n):
    text = (text or "auto").strip()
    if text in ("", "auto"):
        return [None] * n
    vals = [None if v.strip() == "auto" else int(v) for v in text.replace(";", ",").split(",") if v.strip()]
    if len(vals) != n:
        raise ValueError(f"keyframe_frames lists {len(vals)} frames for {n} keyframes")
    return vals


MODELS = ["tube (round products)", "flat (experimental: one face of a box, a flat label)"]
COMPOSITING = ["settings (render.alpha, else measured)", "print (only the print changes)", "label (the whole label band)"]


class ProductLabelVideoRefine:
    """frames + mask + corrected keyframe(s) in -> refined frames + matte + changed + report + QA image out."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "The clip: full frames (recommended) or a workflow's crops."}),
                "mask": ("MASK", {"tooltip": "The product's mask, one per frame, the frames' size (SAM 3, ...)."}),
                "keyframes": ("IMAGE", {"tooltip": "Corrected keyframe(s): a frame of the clip with the label fixed, one "
                                                   "per shot to cover. Frame-sized, or a zoomed crop with keyframe_bbox."}),
                "tracker": (_trackers(), {"tooltip": "AllTracker weights in models/trackers (MIT)."}),
            },
            "optional": {
                "keyframe_index": ("INT", {"default": -1, "min": -1, "max": 1_000_000,
                                           "tooltip": "The keyframe's frame in the batch (LoadVideoTrim's keyframe_index); "
                                                      "-1: matched automatically (one keyframe)."}),
                "keyframe_frames": ("STRING", {"default": "auto", "tooltip": "Several keyframes: their frames, e.g. "
                                                                             "'0, 85' (or auto)."}),
                "keyframe_bbox": ("BBOX", {"tooltip": "The crop windows the keyframes were cut with "
                                                      "(ImageCropByMaskToAspect's bbox)."}),
                "product": ("IMAGE", {"tooltip": "The product photo: context only, never used as texture."}),
                "settings": ("STRING", {"default": "", "multiline": True,
                                        "tooltip": "JSON: the engine's vjob sections (track, fit, render, qa, review, key) "
                                                   "and anchor, shots, words, matte_patch, merged over what is measured."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "processes": ("INT", {"default": 1, "min": 1, "max": 16,
                                      "tooltip": "Render processes. Above 1 the engine runs in its own process (same "
                                                 "pixels, faster)."}),
                "model": (MODELS, {"default": MODELS[0],
                                   "tooltip": "The label's surface. tube: bottles, jars, tubes, lipsticks (anything round). "
                                              "flat (experimental, not yet tested on a real clip): one face of a box or a "
                                              "flat label; run one node per visible face, each with that face's mask and "
                                              "keyframe, frames chained."}),
                "compositing": (COMPOSITING, {"default": COMPOSITING[0],
                                              "tooltip": "print: only the print changes, carried onto the source's own "
                                                         "body (no patch outline can show; light print on a body); then, "
                                                         "unless settings say otherwise, soft_cap 0.3 px, no smear, end "
                                                         "fade 4 px, no sparkle re-add. label: the whole relit label band "
                                                         "over the source. print and label override render.alpha in "
                                                         "the settings JSON; settings: its render.alpha, else the "
                                                         "measured default (print for light print, label for dark)."}),
                "soft_cap": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 3.0, "step": 0.05,
                                       "tooltip": "The most extra blur (px) the label gets to match a soft source frame "
                                                  "(render.soft_cap): a restored label stays sharp. -1: the engine's "
                                                  "default (0.3 with print compositing, uncapped with label) or the "
                                                  "settings JSON's; 0 or more overrides render.soft_cap there."}),
                "glue": ("BOOLEAN", {"default": False,
                                     "tooltip": "Glue the label to the product's own motion (render.glue): it moves "
                                                "frame to frame with the product's pixels and lands where the fit "
                                                "places it over a few frames, so it doesn't swim on near-still frames. "
                                                "Off: the settings JSON's render.glue, else the default (on for round "
                                                "products since engine 0.5.0; {\"render\": {\"glue\": false}} turns "
                                                "it off). The report's review says when an end of the run wants "
                                                "glue.edge \"mean\"."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("frames", "matte", "changed", "report", "qa_image")
    FUNCTION = "refine"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Replaces a hallucinated label through a clip with the true one from corrected keyframe(s): tracks "
                   "the product, fits its surface per frame, re-projects the label with the frame's light, softness "
                   "and grain. Pixels outside the changed area come back exactly as they went in.")

    def refine(self, frames, mask, keyframes, tracker, keyframe_index=-1, keyframe_frames="auto", keyframe_bbox=None,
               product=None, settings="", fps=24.0, processes=1, model=None, compositing=None, soft_cap=-1.0,
               glue=False):
        pl = _engine()
        if mask.dim() == 2:
            mask = mask[None]
        if frames.shape[0] != mask.shape[0] or frames.shape[1:3] != mask.shape[1:3]:
            raise ValueError(f"mask: {tuple(mask.shape)} doesn't match the frames {tuple(frames.shape[:3])}")
        fr = _u8(frames)
        mk = mask.detach().cpu().numpy() > 0.5
        kf = _u8(keyframes)
        kfr = _parse_frames(keyframe_frames, kf.shape[0])
        if keyframe_index >= 0:
            if kf.shape[0] != 1:
                raise ValueError("keyframe_index is for one keyframe: use keyframe_frames for several")
            kfr = [int(keyframe_index)]
        kfs, kfr = _keyframes_to_frames(kf, fr, mk, keyframe_bbox, keyframe_index, kfr)
        cfg = json.loads(settings) if settings and settings.strip() else {}
        if not isinstance(cfg, dict):
            raise ValueError("settings: a JSON object")
        tr = dict(cfg.get("track") or {})
        if tr.get("backend", "alltracker") != "alltracker":
            raise ValueError(f"track.backend {tr['backend']!r}: this pack runs AllTracker only (commercially licensed)")
        cfg["track"] = dict(tr, backend="alltracker")
        if model and model.startswith("flat") and not (isinstance(cfg.get("anchor"), dict) and cfg["anchor"].get("model")):
            cfg["anchor"] = dict(cfg.get("anchor") or {}, model="homography")
        rnd = dict(cfg.get("render") or {})
        if compositing and not compositing.startswith("settings"):     # the widget over render.alpha
            rnd["alpha"] = compositing.split()[0]
        if soft_cap is not None and soft_cap >= 0:                     # the widget over render.soft_cap
            rnd["soft_cap"] = float(soft_cap)
        if glue:                                                       # the widget turns render.glue on
            rnd["glue"] = rnd.get("glue") or True
        if rnd:
            cfg["render"] = rnd
        weights = folder_paths.get_full_path("trackers", tracker)
        if not weights or not os.path.exists(weights):
            raise FileNotFoundError(f"tracker weights not found: put alltracker.pth (MIT, huggingface.co/aharley/"
                                    f"alltracker) in {TRACKERS_DIR}")
        os.environ["PLR_ALLTRACKER"] = weights
        root = Path(folder_paths.get_temp_directory()) / "label-video" / _digest(fr, mk, *kfs)
        root.mkdir(parents=True, exist_ok=True)
        if processes > 1:
            r = _refine_in_process_of_its_own(root, fr, mk, kfs, cfg, kfr, fps, processes, weights)
        else:
            r = pl.refine(fr, mk, kfs, cfg, kfr, fps=fps, work=root / "work", workers=1)
        changed = torch.from_numpy(r["changed"])
        out = frames.clone()
        if changed.any():                    # only the changed pixels are written; the rest is the input tensor
            rgb = out[..., :3]
            rgb[changed] = torch.from_numpy(r["frames"][r["changed"]]).to(out.dtype) / 255.0
        qa = r.get("qa_plot")
        if qa and Path(qa).exists():
            from PIL import Image
            qa_img = torch.from_numpy(np.asarray(Image.open(qa).convert("RGB"), dtype=np.float32) / 255.0)[None]
        else:
            qa_img = torch.zeros((1, 64, 64, 3))
        matte = torch.from_numpy(r["matte"].astype(np.float32))
        return (out, matte, changed.to(torch.float32), json.dumps(r["report"], indent=1, default=str), qa_img)


def _digest(*arrays):
    import hashlib
    h = hashlib.blake2b(digest_size=10)
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _refine_in_process_of_its_own(root, fr, mk, kfs, cfg, kfr, fps, processes, weights):
    """refine() in a child Python (video_worker.py): its render workers can spawn there, which they can't from inside
    ComfyUI (a spawned worker re-imports the main module, ComfyUI's main.py)."""
    io = root / "io"
    io.mkdir(exist_ok=True)
    np.savez(io / "inputs.npz", frames=fr, masks=mk, **{f"kf{i}": k for i, k in enumerate(kfs)})
    (io / "args.json").write_text(json.dumps({"settings": cfg, "kfr": kfr, "fps": fps, "workers": processes,
                                             "n_kf": len(kfs), "work": str(root / "work")}))
    env = dict(os.environ, PLR_ALLTRACKER=weights)
    p = subprocess.run([sys.executable, "-P", str(PACK / "video_worker.py"), str(io)], env=env, capture_output=True, text=True)
    sys.stdout.write(p.stdout[-6000:])
    if p.returncode:
        raise RuntimeError(f"label video worker failed:\n{p.stderr[-4000:]}")
    z = np.load(io / "outputs.npz")
    meta = json.loads((io / "outputs.json").read_text())
    return {"frames": z["frames"], "matte": z["matte"], "changed": z["changed"], "report": meta["report"],
            "qa_plot": meta["qa_plot"]}


class ProductLabelVideoSaveLossless:
    """The comp file: QuickTime PNG (exact RGB), a PNG sequence, or ProRes 4444 (for editors; 12-bit YUV, not exact).
    Never H.264."""

    FORMATS = ["mov (PNG, exact RGB)", "png sequence", "prores 4444 (12-bit YUV, not exact)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",),
                             "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                             "filename_prefix": ("STRING", {"default": "product-label-refinement/video/refined"}),
                             "format": (cls.FORMATS,)},
                "optional": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = CATEGORY

    def save(self, images, fps, filename_prefix, format, audio=None):
        out_dir, name, counter, sub, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(),
                                                                          images.shape[2], images.shape[1])
        frames = _u8(images)
        if format.startswith("png sequence"):
            from PIL import Image
            d = Path(out_dir) / f"{name}_{counter:05}"
            d.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(frames):
                Image.fromarray(f).save(d / f"{name}_{i:05}.png", compress_level=4)
            return {"ui": {"text": [str(d)]}}
        path = Path(out_dir) / f"{name}_{counter:05}.mov"
        _write_mov(path, frames, fps, prores=format.startswith("prores"), audio=audio)
        return {"ui": {"text": [str(path)]}}


def _write_mov(path, frames, fps, prores=False, audio=None):
    import av
    from fractions import Fraction
    rate = Fraction(fps).limit_denominator(1001)
    with av.open(str(path), mode="w") as c:
        if prores:                           # 4444, 12-bit 4:4:4: visually lossless for editors, not bit-exact
            s = c.add_stream("prores_ks", rate=rate, options={"profile": "4", "vendor": "apl0"})
            s.pix_fmt = "yuv444p10le"
        else:                                # QuickTime PNG: the RGB values exactly
            s = c.add_stream("png", rate=rate)
            s.pix_fmt = "rgb24"
        s.width, s.height = frames.shape[2], frames.shape[1]
        a = None
        if audio is not None:
            wav = audio["waveform"][0].detach().cpu().numpy()[:2]          # channels x samples
            lay = "stereo" if wav.shape[0] > 1 else "mono"
            a = c.add_stream("pcm_s16le", rate=int(audio["sample_rate"]), layout=lay)
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
        if a is not None:
            pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
            af = av.AudioFrame.from_ndarray(np.ascontiguousarray(pcm.T).reshape(1, -1), format="s16", layout=lay)
            af.sample_rate = int(audio["sample_rate"])
            for p in a.encode(af):
                c.mux(p)
            for p in a.encode():
                c.mux(p)


class ProductLabelVideoSAM3BothWays:
    """The keyframe's SAM 3 mask propagated through the clip both ways: ComfyUI core's SAM3_VideoTrack (which seeds
    the batch's first frame and runs forward) run forward from the keyframe and backward on the reversed frames up to
    it, stitched into one mask per frame. Needs ComfyUI 0.19.3 or newer (core SAM 3 nodes)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "model": ("MODEL", {"tooltip": "The SAM 3 checkpoint's MODEL."}),
                             "keyframe_mask": ("MASK", {"tooltip": "The product's mask on the keyframe (SAM3 Detect)."}),
                             "keyframe_index": ("INT", {"default": 0, "min": 0, "max": 1_000_000})},
                "optional": {"conditioning": ("CONDITIONING", {"tooltip": "The text prompt, to re-detect while tracking."}),
                             "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "max_objects": ("INT", {"default": 1, "min": 1, "max": 64}),
                             "detect_interval": ("INT", {"default": 1, "min": 1, "max": 1000})}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "track"
    CATEGORY = CATEGORY

    def track(self, images, model, keyframe_mask, keyframe_index, conditioning=None, detection_threshold=0.5,
              max_objects=1, detect_interval=1):
        try:
            from comfy_extras.nodes_sam3 import SAM3_TrackToMask, SAM3_VideoTrack
        except ImportError as e:
            raise RuntimeError("SAM 3 video tracking needs ComfyUI 0.19.3 or newer (core SAM 3 nodes)") from e
        T = images.shape[0]
        k = min(max(int(keyframe_index), 0), T - 1)
        km = keyframe_mask[k:k + 1] if keyframe_mask.shape[0] == T else keyframe_mask[:1]

        def result(o):
            return getattr(o, "result", None) or getattr(o, "args", None) or o

        def run(imgs):
            td = result(SAM3_VideoTrack.execute(images=imgs, model=model, initial_mask=km, conditioning=conditioning,
                                                detection_threshold=detection_threshold, max_objects=max_objects,
                                                detect_interval=detect_interval))[0]
            return result(SAM3_TrackToMask.execute(track_data=td, object_indices=""))[0]
        fwd = run(images[k:])
        if k == 0:
            masks = fwd
        else:
            bwd = run(torch.flip(images[:k + 1], [0]))
            masks = torch.cat([torch.flip(bwd, [0])[:-1], fwd.to(bwd.dtype)], 0)
        if masks.shape[0] != T:
            raise RuntimeError(f"SAM 3 returned {masks.shape[0]} masks for {T} frames")
        return (masks,)


NODE_CLASS_MAPPINGS = {
    "ProductLabelVideoRefine": ProductLabelVideoRefine,
    "ProductLabelVideoSaveLossless": ProductLabelVideoSaveLossless,
    "ProductLabelVideoSAM3BothWays": ProductLabelVideoSAM3BothWays,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProductLabelVideoRefine": "Product Label: Refine (Video)",
    "ProductLabelVideoSaveLossless": "Product Label: Save Lossless (Video)",
    "ProductLabelVideoSAM3BothWays": "Product Label: SAM 3 Track From Keyframe",
}
