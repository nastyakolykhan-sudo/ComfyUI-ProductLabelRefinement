#!/usr/bin/env python3
"""Build the template workflows in both formats from one definition and the nodes' own INPUT_TYPES/RETURN_TYPES.

  python -I tools/build_workflows.py [--job examples/demo/job.json]

Writes workflows/product-label-refinement.api.json (what tools/comfy_run.py submits) and workflows/product-label-refinement.json (what the
ComfyUI editor loads), with the job embedded in Load Job; and workflows/product-label-refinement-draft.{api.json,json}, the same
graph started by Auto Draft from the demo's rough input instead; and workflows/product-label-refinement-ink.{api.json,json}
and -ink-draft.{api.json,json}, the ink graph (print straight on a coloured body, matte ink with polarity light or
dark): a keyframe from a frame, its product mask and the packshot, with the ink demo's measured job in Load Job, or
started by Auto Draft from the frame, the mask and the packshot alone; and workflows/product-label-refinement-all.
{api.json,json}, the one workflow for every case (an image or a clip; flow_nodes.py switches the branches), with
ComfyUI's core and SAM 3 nodes and Image Reframe (ComfyUI-MaskAspectCrop) as the frontend 1.55 stores them. Each UI
file is converted back the way the frontend does (links by slot, widgets by position) and checked to give its API
file again. Re-run it after changing a
node's inputs or outputs. Needs a Python with torch (the pack imports it), like tools/offline_run.py.
"""
import argparse
import importlib.util
import json
import sys
import uuid
from pathlib import Path

sys.dont_write_bytecode = True
PACK = Path(__file__).resolve().parent.parent
WIDGET = {"INT", "FLOAT", "STRING", "BOOLEAN"}


def L(nid, i):
    return [str(nid), i]


def definition(job_text):
    """node id -> (class_type, title, inputs, UI position)"""
    out = "product-label-refinement/imagepost_demo_scene_refined"
    return {
        1: ("LoadImage", "Scene (generated image)", {"image": "imagepost_demo_scene.png"}, (0, 0)),
        2: ("LoadImage", "Reference (real product)", {"image": "imagepost_demo_packshot.png"}, (0, 390)),
        3: ("ImagePostLoadJob", "Load Job", {"job_json": job_text, "job_path": ""}, (0, 780)),
        4: ("ImagePostJobMasks", "Job Masks", {"job": L(3, 0), "scene": L(1, 0), "product_mask": L(6, 1)}, (470, 180)),
        5: ("ImagePostFitWarp", "Fit Warp", {"job": L(3, 0)}, (470, 0)),
        6: ("ImagePostWarpReference", "Warp Reference", {"job": L(3, 0), "homography": L(5, 0), "scene": L(1, 0),
                                                         "reference": L(2, 0), "reference_mask": L(2, 1)}, (470, 400)),
        7: ("ImagePostGradeToScene", "Grade To Scene", {"job": L(3, 0), "scene": L(1, 0), "warped": L(6, 0),
                                                        "product_mask": L(6, 1), "old_silhouette": L(4, 0),
                                                        "occluder_mask": L(4, 1), "colour": "job", "shading": "job",
                                                        "reference": L(2, 0), "coords": L(6, 3), "relight": "job"}, (900, 0)),
        8: ("ImagePostMatchFinish", "Match Finish", {"job": L(3, 0), "scene": L(1, 0), "graded": L(7, 0),
                                                     "product_mask": L(6, 1), "old_silhouette": L(4, 0),
                                                     "occluder_mask": L(4, 1), "coords": L(6, 3)}, (1330, 260)),
        9: ("ImagePostFillLeftovers", "Fill Leftovers", {"job": L(3, 0), "scene": L(1, 0), "product_mask": L(6, 1),
                                                         "old_silhouette": L(4, 0), "occluder_mask": L(4, 1),
                                                         "coords": L(6, 3)}, (1330, 0)),
        10: ("ImagePostCompositeBehind", "Composite Behind", {"job": L(3, 0), "scene": L(1, 0), "background": L(9, 0),
                                                              "product": L(8, 0), "product_mask": L(6, 1),
                                                              "occluder_mask": L(4, 1), "ink": L(7, 4),
                                                              "coords": L(6, 3)}, (1330, 500)),
        11: ("ImagePostQASheet", "QA Sheet", {"job": L(3, 0), "before": L(1, 0), "after": L(10, 0), "reference": L(2, 0),
                                              "matte": L(10, 1), "fill_mask": L(9, 1), "reference_mask": L(2, 1),
                                              "align_report": L(5, 1), "grade_report": L(7, 2),
                                              "finish_report": L(8, 1), "fill_report": L(9, 2),
                                              "old_silhouette": L(4, 0), "occluder_mask": L(4, 1), "coords": L(6, 3)}, (1760, 0)),
        12: ("SaveImage", "Save refined image", {"images": L(10, 0), "filename_prefix": out}, (1330, 760)),
        13: ("SaveImage", "Save QA compare", {"images": L(11, 0), "filename_prefix": "product-label-refinement/qa/compare"}, (2220, 0)),
        14: ("SaveImage", "Save QA edges", {"images": L(11, 1), "filename_prefix": "product-label-refinement/qa/edges"}, (2220, 360)),
        15: ("SaveImage", "Save QA before_after", {"images": L(11, 2), "filename_prefix": "product-label-refinement/qa/before_after"}, (2220, 720)),
        16: ("SaveImage", "Save QA align_overlay", {"images": L(6, 2), "filename_prefix": "product-label-refinement/qa/align_overlay"}, (470, 700)),
        17: ("SaveImage", "Save QA grade_preview", {"images": L(7, 1), "filename_prefix": "product-label-refinement/qa/grade_preview"}, (900, 420)),
        18: ("SaveImage", "Save QA relight_map", {"images": L(7, 3), "filename_prefix": "product-label-refinement/qa/relight_map"}, (900, 780)),
        20: ("SaveImage", "Save QA audit", {"images": L(11, 4), "filename_prefix": "product-label-refinement/qa/audit"}, (2220, 1080)),
        19: ("ImagePostSavePSD", "Save PSD", {"job": L(3, 0), "scene": L(1, 0), "image": L(10, 0),
                                                           "background": L(9, 0), "fill_mask": L(9, 1), "product": L(8, 0),
                                                           "matte": L(10, 1), "filename_prefix": out}, (1760, 580)),
    }


# the demo's rough input, as a person would give it: the card's corners read a few px off, the mug traced roughly,
# two landmark pairs clicked near the header band's corners
DRAFT_INPUT = {
    "quad": "375,168 610,189 613,563 367,580",
    "pairs": "40,230=373,276\n560,230=609,288",
    "occluders": "polygon: 593,428 593,662 650,666 705,662 705,428 650,424",
}


def definition_draft():
    """The template with Auto Draft (node 3) in place of Load Job, and its review sheet saved."""
    nodes = definition("")
    nodes[3] = ("ImagePostAutoDraft", "Auto Draft", {
        "scene": L(1, 0), "reference": L(2, 0), "quad": DRAFT_INPUT["quad"], "model": "homography", "light": "relight",
        "suggest": False, "filename_prefix": "product-label-refinement/draft", "reference_mask": L(2, 1), "pairs": DRAFT_INPUT["pairs"],
        "occluders": DRAFT_INPUT["occluders"], "outline": "", "mask_skip": "", "label": "off", "ink": "off",
        "print_box": "", "bow": ""}, (0, 780))
    nodes[21] = ("SaveImage", "Save QA draft_points", {"images": L(3, 2), "filename_prefix": "product-label-refinement/qa/draft_points"},
                 (0, 1800))
    return nodes


def definition_ink(job_text):
    """The ink graph (matte ink on the body, polarity light or dark; imgpost/ink.py): the packshot's print laid on the
    frame's own body. The product mask (SAM 3) is the matte the print stays inside: it goes into Fill Leftovers' old_silhouette.
    No Grade or Finish: the body stays the frame's, the ink takes the frame's own print colour (Fill Leftovers' ink)."""
    out = "product-label-refinement/inkdemo_frame_keyframe"
    return {
        1: ("LoadImage", "Frame (generated)", {"image": "inkdemo_frame.png"}, (0, 0)),
        2: ("LoadImage", "Packshot (real product)", {"image": "inkdemo_packshot.png"}, (0, 390)),
        22: ("LoadImageMask", "Product mask (SAM 3)", {"image": "inkdemo_mask.png", "channel": "red"}, (0, 1180)),
        3: ("ImagePostLoadJob", "Load Job", {"job_json": job_text, "job_path": ""}, (0, 780)),
        4: ("ImagePostJobMasks", "Job Masks", {"job": L(3, 0), "scene": L(1, 0), "product_mask": L(6, 1)}, (470, 180)),
        5: ("ImagePostFitWarp", "Fit Warp", {"job": L(3, 0)}, (470, 0)),
        6: ("ImagePostWarpReference", "Warp Reference", {"job": L(3, 0), "homography": L(5, 0), "scene": L(1, 0),
                                                         "reference": L(2, 0), "reference_mask": L(2, 1)}, (470, 400)),
        9: ("ImagePostFillLeftovers", "Fill Leftovers (garbled print)", {
            "job": L(3, 0), "scene": L(1, 0), "product_mask": L(6, 1), "old_silhouette": L(22, 0),
            "occluder_mask": L(4, 1), "coords": L(6, 3)}, (900, 0)),
        10: ("ImagePostCompositeBehind", "Composite Behind", {"job": L(3, 0), "scene": L(1, 0), "background": L(9, 0),
                                                              "product": L(6, 0), "product_mask": L(6, 1),
                                                              "occluder_mask": L(4, 1), "ink": L(9, 3),
                                                              "coords": L(6, 3)}, (1330, 0)),
        11: ("ImagePostQASheet", "QA Sheet", {"job": L(3, 0), "before": L(1, 0), "after": L(10, 0), "reference": L(2, 0),
                                              "matte": L(10, 1), "fill_mask": L(9, 1), "reference_mask": L(2, 1),
                                              "align_report": L(5, 1), "fill_report": L(9, 2),
                                              "old_silhouette": L(22, 0), "occluder_mask": L(4, 1), "coords": L(6, 3),
                                              "ink": L(9, 3)}, (1760, 0)),
        12: ("SaveImage", "Save keyframe", {"images": L(10, 0), "filename_prefix": out}, (1330, 420)),
        13: ("SaveImage", "Save QA compare", {"images": L(11, 0), "filename_prefix": "product-label-refinement/qa/compare"}, (2220, 0)),
        14: ("SaveImage", "Save QA edges", {"images": L(11, 1), "filename_prefix": "product-label-refinement/qa/edges"}, (2220, 360)),
        15: ("SaveImage", "Save QA before_after", {"images": L(11, 2), "filename_prefix": "product-label-refinement/qa/before_after"}, (2220, 720)),
        16: ("SaveImage", "Save QA align_overlay", {"images": L(6, 2), "filename_prefix": "product-label-refinement/qa/align_overlay"}, (470, 700)),
        17: ("SaveImage", "Save QA garbled print removed", {"images": L(9, 0), "filename_prefix": "product-label-refinement/qa/print_removed"}, (900, 420)),
        20: ("SaveImage", "Save QA audit", {"images": L(11, 4), "filename_prefix": "product-label-refinement/qa/audit"}, (2220, 1080)),
    }


# the ink demo's input, as an agent would give it: the print's box on the packshot (its rows inside the frosted rims)
# and the bow read off the frame, a seam under the shoulder and the base's own end, measured from the mask
INK_DRAFT_INPUT = {"print_box": "176,396,432,924", "bow": "-140 130/matte"}


def definition_ink_draft():
    """The ink graph started by Auto Draft from the frame, its product mask and the packshot alone (node 3)."""
    nodes = definition_ink("")
    nodes[3] = ("ImagePostAutoDraft", "Auto Draft", {
        "scene": L(1, 0), "reference": L(2, 0), "quad": "", "model": "affine", "light": "off", "suggest": False,
        "filename_prefix": "product-label-refinement/draft", "product_mask": L(22, 0), "reference_mask": L(2, 1),
        "pairs": "", "occluders": "", "outline": "", "mask_skip": "", "label": "off", "ink": "light",
        "print_box": INK_DRAFT_INPUT["print_box"], "bow": INK_DRAFT_INPUT["bow"]}, (0, 780))
    nodes[21] = ("SaveImage", "Save QA draft_points", {"images": L(3, 2), "filename_prefix": "product-label-refinement/qa/draft_points"},
                 (0, 1560))
    return nodes


def definition_all():
    """One workflow for every case (the graph above, the ink graph and Refine (Video)), started by Auto Draft or a
    given job, on an image or a clip. Switches (lazy) and Run If gates (an ExecutionBlocker) run only the branch a
    case needs: an image never reaches the video nodes; a job with print on the body (ink light or dark) never runs
    Grade To Scene or Match Finish; a
    given keyframe skips the stills branch; a given job skips Auto Draft; a given mask skips SAM 3. It is set up as
    the ink demo (examples/ink_demo) with SAM 3 on: set the product mask to inkdemo_mask.png to run it without SAM 3."""
    SW, RUN = "ProductLabelSwitch", "ProductLabelRunIf"
    qa = "product-label-refinement/qa/"
    light, job = L(27, 0), L(26, 0)
    nodes = {
        # inputs: what an agent or a person sets
        1: ("ProductLabelLoadSource", "Source: the image or clip to fix", {"source": "inkdemo_frame.png"}, (0, 0)),
        2: ("LoadImage", "Packshot (the real product)", {"image": "inkdemo_packshot.png"}, (0, 400)),
        3: ("ProductLabelLoadOptional", "Product mask (optional: (none) = SAM 3)", {"file": "(none)"}, (0, 800)),
        4: ("ProductLabelLoadOptional", "Corrected keyframe (optional, clips: (none) = the stills branch makes it)",
            {"file": "(none)"}, (0, 1200)),
        5: ("PrimitiveStringMultiline", "Job JSON (optional: empty = Auto Draft)", {"value": ""}, (0, 1600)),
        6: ("PrimitiveInt", "Keyframe frame (clips; -1 = find the given keyframe's frame)", {"value": 0}, (0, 1880)),
        7: ("ProductLabelVideoSettings", "Refine settings JSON and the product's copy (clips)",
            {"settings": "", "product_copy": ""}, (0, 2040)),
        8: ("PrimitiveBoolean", "Auto Draft fits to the product mask", {"value": True}, (0, 2400)),
        9: ("PrimitiveBoolean", "Fix this product (false: it looks fine on the triage sheet; only the sheet runs)",
            {"value": True}, (0, 2560)),
        84: ("PrimitiveBoolean", "Render the clip (false: only the keyframe, to check it before the clip)",
             {"value": True}, (0, 0)),
        # the keyframe frame's shot (an edit: between the cuts round it), for SAM 3's track and Refine
        19: ("ProductLabelShot", "Shot: the keyframe frame's, between cuts (start/end -1 = found)", {
            "frames": L(85, 0), "keyframe_index": L(6, 0), "start": -1, "end": -1, "masks": L(3, 1)}, (0, 0)),
        # SAM 3: the product's mask on the keyframe frame (an image: the image), then through the shot both ways
        10: ("CheckpointLoaderSimple", "SAM 3.1 checkpoint", {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"}, (480, 0)),
        11: ("CLIPTextEncode", "SAM 3 prompt: the product (comma-separated; name:N for N objects)",
             {"text": "bottle", "clip": L(10, 1)}, (480, 160)),
        12: ("ProductLabelPickFrame", "Keyframe frame", {"index": L(6, 0), "images": L(1, 0)}, (480, 420)),
        86: ("ProductLabelSAM3Points", "SAM 3 points on the keyframe frame (optional: x,y on the product, -x,y off it)",
             {"points": ""}, (0, 0)),
        87: (SW, "SAM 3 text prompt: left out when points are given", {"when": L(86, 2), "if_false": L(11, 0)}, (0, 0)),
        13: ("SAM3_Detect", "SAM 3 on the keyframe frame", {
            "model": L(10, 0), "image": L(12, 0), "threshold": 0.33, "refine_iterations": 2, "individual_masks": False,
            "conditioning": L(87, 0), "positive_coords": L(86, 0), "negative_coords": L(86, 1)}, (480, 600)),
        14: ("ProductLabelVideoSAM3BothWays", "SAM 3 through the shot, both ways from the keyframe", {
            "images": L(19, 0), "model": L(10, 0), "keyframe_mask": L(13, 0), "keyframe_index": L(19, 3),
            "conditioning": L(87, 0), "detection_threshold": 0.33, "max_objects": 1, "detect_interval": 1}, (480, 880)),
        15: (SW, "SAM 3 masks: the clip's, else the image's", {"when": L(1, 3), "if_true": L(14, 0),
                                                              "if_false": L(13, 0)}, (480, 1200)),
        16: (SW, "Product mask: the given one, else SAM 3", {"when": L(3, 2), "if_true": L(19, 1), "if_false": L(15, 0)},
             (480, 1340)),
        18: (SW, "Keyframe frame's mask: the given one, else SAM 3 on that frame", {
            "when": L(3, 2), "if_true": L(3, 1), "if_false": L(13, 0)}, (480, 1620)),
        17: ("ProductLabelPickFrame", "Keyframe frame's mask", {"index": L(6, 0), "masks": L(18, 0)}, (480, 1480)),
        # the packshot's outline when it has no alpha: SAM 3 (always for triage; for the fix when 83 says so)
        81: ("SAM3_Detect", "SAM 3 on the packshot (only when it has no alpha)", {
            "model": L(10, 0), "image": L(2, 0), "threshold": 0.33, "refine_iterations": 2, "individual_masks": False,
            "conditioning": L(11, 0)}, (0, 0)),
        82: ("ProductLabelPackshotMatte", "Packshot outline for triage: alpha, else SAM 3", {
            "reference": L(2, 0), "mask": L(2, 1), "use": "alpha, else SAM 3", "sam_mask": L(81, 0)}, (0, 0)),
        83: ("ProductLabelPackshotMatte", "Packshot outline for the fix (alpha only: the engine traces the backdrop)", {
            "reference": L(2, 0), "mask": L(2, 1), "use": "alpha only (else the engine traces the packshot's background)",
            "sam_mask": L(81, 0)}, (0, 0)),
        # triage: does the product need fixing at all? (the agent looks at the sheet and sets node 9)
        70: ("ProductLabelTriage", "Label Triage: the print on the frame | the packshot's", {
            "frame": L(12, 0), "mask": L(17, 1), "reference": L(2, 0), "reference_mask": L(82, 0),
            "product_copy": L(7, 1), "ocr": False, "glyph": False}, (0, 0)),
        71: ("SaveImage", "Save the triage sheet", {"images": L(70, 0),
                                                    "filename_prefix": "product-label-refinement/triage/sheet"}, (0, 0)),
        76: ("ProductLabelMaskOverlay", "SAM 3's mask on the keyframe frame (check it fits)", {
            "images": L(12, 0), "masks": L(17, 1), "opacity": 0.4}, (0, 0)),
        77: ("SaveImage", "Save the mask check", {"images": L(76, 0),
                                                  "filename_prefix": "product-label-refinement/triage/mask"}, (0, 0)),
        72: (RUN, "Fix: only when the product needs it", {"value": L(12, 0), "when": L(9, 0), "run_when": "true"},
             (0, 0)),
        73: (RUN, "Video: only when the product needs fixing", {"value": L(1, 0), "when": L(9, 0), "run_when": "true"},
             (0, 0)),
        85: (RUN, "Clip: only when Render the clip is on", {"value": L(73, 0), "when": L(84, 0), "run_when": "true"},
             (0, 0)),
        74: (RUN, "Refine's mask: only when the product needs fixing (triage never tracks the clip)", {
            "value": L(16, 0), "when": L(85, 0), "run_when": "true"}, (0, 0)),
        # stills: the image, or the clip's keyframe frame; Auto Draft, or the given job
        20: (RUN, "Stills branch: only without a given keyframe", {"value": L(72, 0), "when": L(4, 2),
                                                                  "run_when": "false"}, (960, 0)),
        21: ("ImageReframe", "Straighten: frame and turn (optional; all 0 = the whole frame)", {
            "input_image": L(20, 0), "image": "", "center_x": 0.0, "center_y": 0.0, "frame_width": 0, "frame_height": 0,
            "rotation": 0.0, "aspect_ratio": "free", "padding_color": "grey", "mask": L(17, 1)}, (960, 140)),
        88: ("ProductLabelFrameMask", "Straightened product mask (the frame's padding left out)", {
            "mask": L(21, 2), "crop_data": L(21, 3)}, (960, 680)),
        22: (RUN, "Auto Draft: only without a given job", {"value": L(21, 1), "when": L(5, 0), "run_when": "false"},
             (960, 700)),
        23: (SW, "Auto Draft's product mask (on, off)", {"when": L(8, 0), "if_true": L(88, 0)}, (960, 840)),
        24: ("ImagePostAutoDraft", "Auto Draft", {
            "scene": L(22, 0), "reference": L(2, 0), "quad": "", "model": "affine", "light": "off", "suggest": False,
            "filename_prefix": "product-label-refinement/draft", "product_mask": L(23, 0), "reference_mask": L(83, 0),
            "pairs": "", "occluders": "", "outline": "", "mask_skip": "", "label": "off", "ink": "light",
            "print_box": INK_DRAFT_INPUT["print_box"], "bow": INK_DRAFT_INPUT["bow"]}, (960, 980)),
        25: ("ImagePostLoadJob", "Load Job (the given job)", {"job_json": L(5, 0), "job_path": ""}, (960, 1800)),
        26: (SW, "Job: the given one, else the draft", {"when": L(5, 0), "if_true": L(25, 0), "if_false": L(24, 0)},
             (960, 2000)),
        27: ("ProductLabelPrintType", "Print type (from the job)", {"job": job}, (960, 2140)),
        28: ("ImagePostJobMasks", "Job Masks", {"job": job, "scene": L(21, 1), "product_mask": L(30, 1)}, (1500, 180)),
        29: ("ImagePostFitWarp", "Fit Warp", {"job": job}, (1500, 0)),
        30: ("ImagePostWarpReference", "Warp Reference", {"job": job, "homography": L(29, 0), "scene": L(21, 1),
                                                          "reference": L(2, 0), "reference_mask": L(83, 0)}, (1500, 400)),
        31: ("ImagePostGradeToScene", "Grade To Scene (paper graph)", {
            "job": job, "scene": L(21, 1), "warped": L(30, 0), "product_mask": L(30, 1), "old_silhouette": L(28, 0),
            "occluder_mask": L(28, 1), "colour": "job", "shading": "job", "reference": L(2, 0), "coords": L(30, 3),
            "relight": "job"}, (1500, 700)),
        32: ("ImagePostMatchFinish", "Match Finish (paper graph)", {
            "job": job, "scene": L(21, 1), "graded": L(31, 0), "product_mask": L(30, 1), "old_silhouette": L(28, 0),
            "occluder_mask": L(28, 1), "coords": L(30, 3)}, (1500, 1080)),
        33: (SW, "Old silhouette: the product mask (print on the body), else the job's", {
            "when": light, "if_true": L(88, 0), "if_false": L(28, 0)}, (1980, 0)),
        34: ("ImagePostFillLeftovers", "Fill Leftovers", {
            "job": job, "scene": L(21, 1), "product_mask": L(30, 1), "old_silhouette": L(33, 0),
            "occluder_mask": L(28, 1), "coords": L(30, 3)}, (1980, 140)),
        35: (SW, "Product layer: the warped print (print on the body), else graded", {
            "when": light, "if_true": L(30, 0), "if_false": L(32, 0)}, (1980, 420)),
        36: (SW, "Ink: Fill Leftovers' (print on the body), else Grade To Scene's", {
            "when": light, "if_true": L(34, 3), "if_false": L(31, 4)}, (1980, 560)),
        37: ("ImagePostCompositeBehind", "Composite Behind", {
            "job": job, "scene": L(21, 1), "background": L(34, 0), "product": L(35, 0), "product_mask": L(30, 1),
            "occluder_mask": L(28, 1), "ink": L(36, 0), "coords": L(30, 3)}, (1980, 700)),
        38: (SW, "Grade report (paper graph)", {"when": light, "if_false": L(31, 2)}, (1980, 1000)),
        39: (SW, "Finish report (paper graph)", {"when": light, "if_false": L(32, 1)}, (1980, 1140)),
        40: ("ImagePostQASheet", "QA Sheet (its text: report.json)", {
            "job": job, "before": L(21, 1), "after": L(37, 0), "reference": L(2, 0), "matte": L(37, 1),
            "fill_mask": L(34, 1), "reference_mask": L(83, 0), "align_report": L(29, 1), "grade_report": L(38, 0),
            "finish_report": L(39, 0), "fill_report": L(34, 2), "old_silhouette": L(33, 0), "occluder_mask": L(28, 1),
            "coords": L(30, 3), "ink": L(34, 3)}, (2460, 0)),
        41: ("ProductLabelPasteBackMask", "Paste-back mask (straightened crops)", {
            "before": L(21, 1), "after": L(37, 0), "crop_data": L(21, 3), "grow": 2}, (2460, 560)),
        42: ("ImageReframeComposite", "Straightened crop back into its frame", {
            "frame_image": L(37, 0), "original_image": L(21, 0), "paste": "crop onto original", "feather": 0,
            "resize_source": True, "mask": L(41, 0), "crop_data": L(21, 3)},
            (2460, 760)),
        43: (SW, "Refined frame: pasted back (straightened), else as it is", {
            "when": L(41, 1), "if_true": L(42, 0), "if_false": L(37, 0)}, (2460, 980)),
        44: ("SaveImage", "Save refined image (a clip: the refined keyframe)", {
            "images": L(43, 0), "filename_prefix": "product-label-refinement/refined"}, (2940, 0)),
        45: ("SaveImage", "Save QA compare", {"images": L(40, 0), "filename_prefix": qa + "compare"}, (2940, 360)),
        46: ("SaveImage", "Save QA edges", {"images": L(40, 1), "filename_prefix": qa + "edges"}, (2940, 720)),
        47: ("SaveImage", "Save QA before_after", {"images": L(40, 2), "filename_prefix": qa + "before_after"}, (2940, 1080)),
        48: ("SaveImage", "Save QA audit", {"images": L(40, 4), "filename_prefix": qa + "audit"}, (2940, 1440)),
        49: ("SaveImage", "Save QA align_overlay", {"images": L(30, 2), "filename_prefix": qa + "align_overlay"}, (3420, 0)),
        50: ("SaveImage", "Save QA draft_points", {"images": L(24, 2), "filename_prefix": qa + "draft_points"}, (3420, 360)),
        51: (RUN, "Grade preview: paper only", {"value": L(31, 1), "when": light, "run_when": "false"}, (3420, 720)),
        52: ("SaveImage", "Save QA grade_preview", {"images": L(51, 0), "filename_prefix": qa + "grade_preview"}, (3420, 860)),
        53: (RUN, "Relight map: paper only", {"value": L(31, 3), "when": light, "run_when": "false"}, (3420, 1220)),
        54: ("SaveImage", "Save QA relight_map", {"images": L(53, 0), "filename_prefix": qa + "relight_map"}, (3420, 1360)),
        55: (RUN, "Garbled print removed: print on the body only", {"value": L(34, 0), "when": light, "run_when": "true"},
             (3900, 0)),
        56: ("SaveImage", "Save QA garbled print removed", {"images": L(55, 0), "filename_prefix": qa + "print_removed"},
             (3900, 140)),
        57: (RUN, "Product layer for the PSD: paper only", {"value": L(32, 0), "when": light, "run_when": "false"},
             (3900, 500)),
        58: ("ImagePostSavePSD", "Save PSD (paper)", {
            "job": job, "scene": L(21, 1), "image": L(37, 0), "background": L(34, 0), "fill_mask": L(34, 1),
            "product": L(57, 0), "matte": L(37, 1), "filename_prefix": "product-label-refinement/refined"}, (3900, 640)),
        # video: Refine (Video) through the clip from the keyframe (the stills branch's, or the given one)
        60: (RUN, "Video branch: only for a clip", {"value": L(19, 0), "when": L(1, 3), "run_when": "true"}, (960, 2300)),
        61: (SW, "Keyframe: the given one, else the stills branch's", {"when": L(4, 2), "if_true": L(4, 0),
                                                                       "if_false": L(43, 0)}, (960, 2440)),
        62: ("ProductLabelVideoRefine", "Product Label: Refine (Video)", {
            "frames": L(60, 0), "mask": L(74, 0), "keyframes": L(61, 0), "tracker": "alltracker.pth",
            "keyframe_index": L(19, 3), "keyframe_frames": "auto", "settings": L(7, 0), "fps": L(1, 1), "processes": 1,
            "model": "tube (round products)", "compositing": "settings (render.alpha, else measured)", "soft_cap": -1.0,
            "glue": False}, (1500, 2300)),
        75: ("ProductLabelSplice", "The refined shot back into the clip (its text: shot.json)", {
            "frames": L(1, 0), "shot_frames": L(62, 0), "start": L(19, 2), "shot": L(19, 4)}, (0, 0)),
        63: ("ProductLabelVideoSaveLossless", "Save the lossless master (QuickTime PNG)", {
            "images": L(75, 0), "fps": L(1, 1), "filename_prefix": "product-label-refinement/video/refined",
            "format": "mov (PNG, exact RGB)", "audio": L(1, 2)}, (1980, 2300)),
        64: ("CreateVideo", "Review copy", {"images": L(75, 0), "fps": L(1, 1), "audio": L(1, 2), "bit_depth": "auto",
                                            "color_space": "sRGB", "codec": "none"}, (1980, 2560)),
        65: ("SaveVideo", "Save the refined clip (mp4, to review)", {
            "video": L(64, 0), "filename_prefix": "product-label-refinement/video/review", "format": "auto",
            "format.codec": "auto", "codec": "auto"}, (2460, 2300)),
        66: ("SaveImage", "Save the clip's QA plot", {"images": L(62, 4),
                                                      "filename_prefix": "product-label-refinement/video/qa_plot"},
             (2460, 2620)),
        67: ("PreviewAny", "Refine report (JSON)", {"source": L(62, 3)}, (2940, 2300)),
        78: ("ProductLabelMaskOverlay", "The shot's masks on its frames (check SAM 3's track)", {
            "images": L(60, 0), "masks": L(74, 0), "opacity": 0.4}, (0, 0)),
        79: ("CreateVideo", "Mask check video", {"images": L(78, 0), "fps": L(1, 1), "bit_depth": "auto",
                                                 "color_space": "sRGB", "codec": "none"}, (0, 0)),
        80: ("SaveVideo", "Save the mask check (mp4)", {
            "video": L(79, 0), "filename_prefix": "product-label-refinement/video/masks", "format": "auto",
            "format.codec": "auto", "codec": "auto"}, (0, 0)),
    }
    pos = column_layout(ALL_COLUMNS, {k: v[0] for k, v in nodes.items()})
    return {k: (ct, title, inp, pos[k]) for k, (ct, title, inp, _) in nodes.items()}


# the one workflow's layout: columns left to right, nodes top to bottom (heights as the frontend draws them)
ALL_COLUMNS = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 84], [19, 10, 11, 86, 87, 12, 13, 18, 17, 14, 15, 16], [81, 82, 83, 70, 71, 76, 77, 72, 73, 85, 74],
               [20, 21, 88, 22, 23, 24], [25, 26, 27, 29, 28, 30],
               [31, 32, 33, 34, 35, 36], [37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48], [49, 50, 51, 52, 53, 54],
               [55, 56, 57, 58], [60, 61, 62], [75, 63, 64, 65], [66, 67, 78, 79, 80]]
ALL_HEIGHTS = {1: 374, 21: 720, 24: 852, 31: 246, 70: 300}
ALL_WIDTHS = {4: 510, 6: 439, 11: 504, 33: 416, 35: 397, 43: 400}


def column_layout(columns, cls, gap=70, col_gap=110):
    """-> {node id: (x, y)}; cls: {node id: class_type}, for the sizes in SIZE."""
    pos, x = {}, 0
    for col in columns:
        y, width = 0, 0
        for nid in col:
            w, h = SIZE.get(cls[nid], (380, 220))
            w, h = ALL_WIDTHS.get(nid, w), ALL_HEIGHTS.get(nid, h)
            pos[nid] = (x, y)
            y += h + gap
            width = max(width, w)
        x += width + col_gap
    return pos


# UI-only notes and groups of the one workflow (the API file has neither)
ALL_NOTES = {
    90: ("What to set", (-600, 0), (480, 980), (
        "ONE WORKFLOW FOR EVERY CASE\n\n"
        "Source: an image or a clip (Load Source). An image runs the stills branch only; a clip runs SAM 3 through "
        "the keyframe frame's shot, the stills branch on the keyframe frame, then Refine (Video) on that shot, spliced "
        "back into the clip. An edit with several shots: the Shot node finds the cuts round the keyframe frame (or "
        "set its start and end); run each product's shot, then the next product on the output.\n\n"
        "Packshot: the real product photo.\n"
        "Product mask: (none) = SAM 3 with the prompt; or a mask file (white = product; a clip: one per frame).\n"
        "Corrected keyframe (clips): (none) = the stills branch makes it; or a frame fixed elsewhere.\n"
        "Fix this product: false = the product looks fine on the triage sheet: only SAM 3 on the keyframe frame and "
        "the sheet run.\n"
        "Job JSON: empty = Auto Draft from its inputs; or a measured job.\n"
        "Keyframe frame (clips): the frame the stills branch fixes, from 0.\n"
        "Refine settings JSON and product copy (clips).\n"
        "SAM 3 prompt: the product, e.g. 'pump bottle'.\n"
        "Straighten: Image Reframe's frame and rotation (all 0: off).\n"
        "Print type: Auto Draft's ink (off: a paper label, or print on paper or a flat panel; light or dark: print "
        "straight on a round body, lighter or darker than it, with no paper; dark also snaps the print onto the "
        "generated print).\n\n"
        "Outputs: the refined image (a clip: the refined keyframe), QA sheets and report.json (QA Sheet's text); "
        "for a clip, the lossless master (.mov), the mp4 review copy, the QA plot and the Refine report.\n\n"
        "Only the needed branch runs: Switch and Run If nodes skip the rest.")),
}

ALL_GROUPS = [
    ("Inputs: set these", [1, 2, 3, 4, 5, 6, 7, 8, 9, 84], "#3f789e"),
    ("SAM 3: the product's mask (skipped when a mask is given); the keyframe frame's shot",
     [19, 10, 11, 86, 87, 12, 13, 14, 15, 16, 17, 18], "#8A8"),
    ("Triage: does the product need fixing? (the agent sets Fix this product); the mask check; the packshot's outline",
     [81, 82, 83, 70, 71, 76, 77, 72, 73, 85, 74], "#2a7"),
    ("Stills: the image, or the clip's keyframe frame (Auto Draft, or the given job)",
     [20, 21, 88, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], "#b58b2a"),
    ("Paper or print on the body, composite and QA", [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], "#a1309b"),
    ("Stills outputs", [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58], "#444"),
    ("Video: Refine (Video) through the shot, spliced back into the clip (clips only)",
     [60, 61, 62, 75, 63, 64, 65, 66, 67, 78, 79, 80], "#335"),
]


SIZE = {"LoadImage": (400, 360), "LoadImageMask": (400, 360), "SaveImage": (400, 330), "ImagePostLoadJob": (420, 560), "ImagePostAutoDraft": (460, 760),
        "ImagePostQASheet": (420, 520), "ImagePostFitWarp": (380, 140), "ImagePostSavePSD": (420, 300),
        "ProductLabelLoadSource": (400, 360), "ProductLabelLoadOptional": (400, 360), "ProductLabelSwitch": (380, 100),
        "ProductLabelRunIf": (380, 100), "ProductLabelPrintType": (380, 80), "ProductLabelPickFrame": (380, 120),
        "ProductLabelShot": (400, 200), "ProductLabelSplice": (400, 240), "ProductLabelMaskOverlay": (380, 120),
        "ProductLabelPackshotMatte": (400, 140), "ProductLabelSAM3Points": (400, 100),
        "PrimitiveStringMultiline": (400, 240), "PrimitiveInt": (400, 100), "PrimitiveBoolean": (400, 80),
        "ProductLabelVideoSettings": (400, 320), "CLIPTextEncode": (400, 200), "CheckpointLoaderSimple": (400, 120),
        "SAM3_Detect": (400, 240), "ProductLabelVideoSAM3BothWays": (400, 260), "ImageReframe": (440, 720), "ProductLabelFrameMask": (380, 80),
        "ProductLabelVideoRefine": (420, 440), "PreviewAny": (420, 300), "CreateVideo": (380, 180), "SaveVideo": (400, 300)}

# Nodes of ComfyUI core and other packs, as the frontend (1.55) serializes them: (slot inputs, widgets in
# widgets_values order, RETURN_TYPES, RETURN_NAMES). Widgets in FRONTEND_ONLY aren't inputs in the API file.
FOREIGN = {
    "LoadImage": ([], ["image", "upload"], ("IMAGE", "MASK"), ("IMAGE", "MASK")),
    "SaveImage": ([("images", "IMAGE")], ["filename_prefix"], (), ()),
    "LoadImageMask": ([], ["image", "channel", "upload"], ("MASK",), ("MASK",)),
    "CheckpointLoaderSimple": ([], ["ckpt_name"], ("MODEL", "CLIP", "VAE"), ("MODEL", "CLIP", "VAE")),
    "CLIPTextEncode": ([("clip", "CLIP")], ["text"], ("CONDITIONING",), ("CONDITIONING",)),
    "SAM3_Detect": ([("model", "MODEL"), ("image", "IMAGE"), ("conditioning", "CONDITIONING"),
                     ("bboxes", "BOUNDING_BOX"), ("positive_coords", "STRING"), ("negative_coords", "STRING")],
                    ["threshold", "refine_iterations", "individual_masks"], ("MASK", "BOUNDING_BOX"), ("masks", "bboxes")),
    "ImageReframe": ([("input_image", "IMAGE"), ("mask", "MASK")],
                     ["image", "upload", "center_x", "center_y", "frame_width", "frame_height", "rotation", "aspect_ratio",
                      "padding_color", "image_reframe_editor"],
                     ("IMAGE", "IMAGE", "MASK", "CROP_DATA"), ("original_image", "cropped_image", "mask", "crop_data")),
    "ImageReframeComposite": ([("frame_image", "IMAGE"), ("original_image", "IMAGE"), ("mask", "MASK"),
                               ("crop_data", "CROP_DATA")], ["paste", "feather", "resize_source"], ("IMAGE",), ("image",)),
    "PrimitiveStringMultiline": ([], ["value"], ("STRING",), ("STRING",)),
    "PrimitiveInt": ([], ["value", "control_after_generate"], ("INT",), ("INT",)),
    "PrimitiveBoolean": ([], ["value"], ("BOOLEAN",), ("BOOLEAN",)),
    "CreateVideo": ([("images", "IMAGE"), ("audio", "AUDIO")], ["fps", "bit_depth", "color_space", "codec"],
                    ("VIDEO",), ("VIDEO",)),
    "SaveVideo": ([("video", "VIDEO")], ["filename_prefix", "format", "format.codec", "codec"], ("VIDEO",), ("video",)),
    "PreviewAny": ([("source", "*")], [], ("STRING",), ("STRING",)),
}
FRONTEND_ONLY = {"upload": "image", "control_after_generate": "fixed", "image_reframe_editor": ""}
UI_ONLY = {"Note"}
WIDGET_PLACEHOLDER = {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}


def load_pack():
    spec = importlib.util.spec_from_file_location("comfyui_imagepost", PACK / "__init__.py",
                                                  submodule_search_locations=[str(PACK)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.NODE_CLASS_MAPPINGS


def node_spec(classes, ct):
    """-> (slot inputs [(name, type)], widget names, RETURN_TYPES, RETURN_NAMES)"""
    if ct in FOREIGN:
        return FOREIGN[ct]
    c = classes[ct]
    it = c.INPUT_TYPES()
    slots, widgets = [], []
    for sec in ("required", "optional"):
        for name, (t, *opt) in it.get(sec, {}).items():
            o = opt[0] if opt else {}
            if (isinstance(t, list) or t in WIDGET) and not o.get("forceInput"):
                widgets.append(name)
                if o.get("image_upload"):         # the frontend's upload button follows its combo
                    widgets.append("upload")
            else:
                slots.append((name, t))
    return slots, widgets, c.RETURN_TYPES, getattr(c, "RETURN_NAMES", c.RETURN_TYPES)


def widget_type(classes, ct, name):
    """The socket type of a widget turned input (a link into it)."""
    if ct in classes:
        for sec in ("required", "optional"):
            t = classes[ct].INPUT_TYPES().get(sec, {}).get(name)
            if t:
                return "COMBO" if isinstance(t[0], list) else t[0]
    return {"value": {"PrimitiveStringMultiline": "STRING", "PrimitiveInt": "INT"}.get(ct, "*")}.get(name, "*")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", default=str(PACK / "examples" / "demo" / "job.json"))
    ap.add_argument("--ink-job", default=str(PACK / "examples" / "ink_demo" / "job.json"))
    a = ap.parse_args()
    job_text = Path(a.job).expanduser().read_text()
    json.loads(job_text)
    ink_text = Path(a.ink_job).expanduser().read_text()
    json.loads(ink_text)
    classes = load_pack()
    build("product-label-refinement", definition(job_text), classes)
    build("product-label-refinement-draft", definition_draft(), classes)
    build("product-label-refinement-ink", definition_ink(ink_text), classes)
    build("product-label-refinement-ink-draft", definition_ink_draft(), classes)
    build("product-label-refinement-all", definition_all(), classes, ALL_NOTES, ALL_GROUPS)


def build(wf_name, nodes_def, classes, notes=None, groups=None):
    api = {str(k): {"inputs": inp, "class_type": ct, "_meta": {"title": title}}
           for k, (ct, title, inp, _) in nodes_def.items()}
    (PACK / "workflows" / f"{wf_name}.api.json").write_text(json.dumps(api, indent=2) + "\n")

    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for v in api[n]["inputs"].values():
            if isinstance(v, list):
                visit(v[0])
        order.append(n)

    for n in sorted(api, key=int):
        visit(n)
    specs = {nid: node_spec(classes, api[nid]["class_type"]) for nid in api}
    nodes, links = {}, []
    for nid in sorted(api, key=int):
        n = api[nid]
        slots, widgets, rt, rn = specs[nid]
        for name in n["inputs"]:
            if name not in widgets and name not in [s for s, _ in slots]:
                sys.exit(f"node {nid} ({n['class_type']}): {name} is not an input of the node")
        wv = [n["inputs"].get(w, FRONTEND_ONLY.get(w)) for w in widgets]
        for i, (w, v) in enumerate(zip(widgets, wv)):   # a widget fed by a link keeps a value of its own type
            if isinstance(v, list):
                wv[i] = WIDGET_PLACEHOLDER.get(widget_type(classes, n["class_type"], w))
        inputs = [{"name": s, "type": t, "link": None} for s, t in slots]
        inputs += [{"name": w, "type": widget_type(classes, n["class_type"], w), "widget": {"name": w}, "link": None}
                   for w in widgets if isinstance(n["inputs"].get(w), list)]
        nodes[nid] = {"id": int(nid), "type": n["class_type"], "pos": list(nodes_def[int(nid)][3]),
                      "size": list(SIZE.get(n["class_type"], (380, 220))), "flags": {}, "order": order.index(nid),
                      "mode": 0, "inputs": inputs,
                      "outputs": [{"name": o, "type": t, "links": None, "slot_index": i}
                                  for i, (o, t) in enumerate(zip(rn, rt))],
                      "title": n["_meta"]["title"], "properties": {"Node name for S&R": n["class_type"]},
                      "widgets_values": wv}
    for nid in sorted(api, key=int):
        for slot, inp in enumerate(nodes[nid]["inputs"]):
            v = api[nid]["inputs"].get(inp["name"])
            if v is None:
                continue
            src, idx = v
            t, got = inp["type"], specs[src][2][idx]
            if got != t and "*" not in (got, t) and not (t == "COMBO" and isinstance(got, list)):
                sys.exit(f"node {nid}.{inp['name']} expects {t}, linked to node {src} output {idx} ({got})")
            lid = len(links) + 1
            links.append([lid, int(src), idx, int(nid), slot, t])
            inp["link"] = lid
            out = nodes[src]["outputs"][idx]
            out["links"] = (out["links"] or []) + [lid]
    for nid, (title, pos, size, text) in (notes or {}).items():
        nodes[str(nid)] = {"id": nid, "type": "Note", "pos": list(pos), "size": list(size), "flags": {}, "order": len(order),
                           "mode": 0, "inputs": [], "outputs": [], "title": title, "properties": {},
                           "widgets_values": [text], "color": "#432", "bgcolor": "#653"}
    ui_groups = []
    for gi, (title, members, colour) in enumerate(groups or [], 1):
        boxes = [(nodes[str(m)]["pos"], nodes[str(m)]["size"]) for m in members]
        x0 = min(p[0] for p, _ in boxes) - 20
        y0 = min(p[1] for p, _ in boxes) - 80
        x1 = max(p[0] + s[0] for p, s in boxes) + 20
        y1 = max(p[1] + s[1] for p, s in boxes) + 20
        ui_groups.append({"id": gi, "title": title, "bounding": [x0, y0, x1 - x0, y1 - y0], "color": colour,
                          "font_size": 24, "flags": {}})
    ui = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"imagepost/{wf_name}")), "revision": 0,
          "last_node_id": max(int(n) for n in nodes), "last_link_id": len(links),
          "nodes": [nodes[n] for n in sorted(nodes, key=int)], "links": links, "groups": ui_groups, "config": {},
          "extra": {"ds": {"scale": 0.55, "offset": [40, 40]}}, "version": 0.4}
    (PACK / "workflows" / f"{wf_name}.json").write_text(json.dumps(ui, indent=1) + "\n")

    # round trip: UI -> API the way the frontend does it (links by slot, widgets by position)
    lk = {l[0]: l for l in links}
    back = {}
    for n in ui["nodes"]:
        if n["type"] in UI_ONLY:
            continue
        slots, widgets, _, _ = specs[str(n["id"])]
        inputs = {w: v for w, v in zip(widgets, n["widgets_values"]) if w not in FRONTEND_ONLY and v is not None}
        for inp in n["inputs"]:
            if inp["link"] is not None:
                l = lk[inp["link"]]
                inputs[inp["name"]] = [str(l[1]), l[2]]
        back[str(n["id"])] = {"inputs": inputs, "class_type": n["type"]}
    for k, v in api.items():
        if back.get(k) != {"inputs": v["inputs"], "class_type": v["class_type"]}:
            sys.exit(f"round trip differs at node {k}")
    if set(back) != set(api):
        sys.exit("round trip differs: node ids")
    print(f"wrote workflows/{wf_name}.api.json and workflows/{wf_name}.json: {len(api)} nodes, {len(links)} links; "
          "UI converts back to the API file exactly")


if __name__ == "__main__":
    main()
