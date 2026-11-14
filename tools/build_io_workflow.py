#!/usr/bin/env python3
"""Build the deployable version of the one workflow: Agentic Product Label Refinement, in both formats.

  python -P tools/build_io_workflow.py

Writes workflows/agentic-product-label-refinement.{api.json,json}: the graph of product-label-refinement-all
(tools/build_workflows.py definition_all) with its inputs and outputs as WorkflowIO nodes (comfy-workflow-io-nodes),
so a platform that builds a form from them (an app, an API) can run it:

- Source (one upload, an image or a clip) through Workflow IO Video Input and core Get Video Components: an image
  decodes as a one-frame video; the branch (is_video) is its frame count above 1 (core Get Image Size + Math
  Expression). It replaces Load Source, the file-name loader.
- Packshot through Workflow IO Image Input (image and alpha mask, as Load Image).
- Product (SAM 3's prompt) and Label Text (the product's copy) through Workflow IO Text Input.
- refined_image (the refined still, or a clip's refined keyframe) and refined_clip (the mp4) through Workflow IO
  Image/Video Output, their file names from Audit Name / Review Name (_AUDIT-FAIL, _REVIEW-FAIL, _REVIEW-CHECK).
- A clip whose keyframe fails its audit: the Audit Gate (which stops the run with an error, so a platform returns no
  file) is replaced by a Run If that skips the clip (core Compare Text: Audit Name's name ends with _AUDIT-FAIL). The
  job succeeds and returns refined_image_AUDIT-FAIL, no refined_clip: the same as a still that fails.

Everything else is the one workflow as it is: the agent's options (mask, corrected keyframe, job, keyframe frame,
Refine settings, triage, Render the clip) stay as nodes at their defaults, and the QA files are saved beside the
outputs. Re-run it after tools/build_workflows.py changes definition_all. Needs a Python with torch (the pack
imports it), like tools/build_workflows.py.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_workflows as bw  # noqa: E402

NAME = "agentic-product-label-refinement"
META = ["required", "description", "example", "advanced", "group", "order"]
FILE_IN = ["param_name", "value", "accepted_formats", "max_size_mb"] + META
MULTI = "FLOAT,INT,BOOLEAN"
# comfy-workflow-io-nodes and the core nodes this file adds, as the frontend serializes them (see bw.FOREIGN)
bw.FOREIGN.update({
    "WorkflowIOVideoInput": ([], FILE_IN, ("VIDEO",), ("video",)),
    "WorkflowIOImageInput": ([], FILE_IN, ("IMAGE", "MASK"), ("image", "mask")),
    "WorkflowIOTextInput": ([], ["param_name", "value"] + META, ("STRING",), ("value",)),
    "WorkflowIOImageOutput": ([("images", "IMAGE"), ("filename", "STRING")],
                              ["output_name", "filename_template", "format", "include"], ("IMAGE",), ("images",)),
    "WorkflowIOVideoOutput": ([("video", "VIDEO"), ("filename", "STRING")],
                              ["output_name", "filename_template", "format", "include"], ("VIDEO",), ("video",)),
    "GetVideoComponents": ([("video", "VIDEO")], [], ("IMAGE", "AUDIO", "FLOAT", "COMBO", "COMBO"),
                           ("images", "audio", "fps", "bit_depth", "color_space")),
    "GetImageSize": ([("image", "IMAGE")], [], ("INT", "INT", "INT"), ("width", "height", "batch_size")),
    "ComfyMathExpression": ([("values.a", MULTI), ("values.b", MULTI)], ["expression"], ("FLOAT", "INT", "BOOLEAN"),
                            ("FLOAT", "INT", "BOOL")),
    "StringCompare": ([], ["string_a", "string_b", "mode", "case_sensitive"], ("BOOLEAN",), ("BOOLEAN",)),
})
bw.SIZE.update({"WorkflowIOVideoInput": (400, 420), "WorkflowIOImageInput": (400, 420),
                "WorkflowIOTextInput": (400, 380), "WorkflowIOImageOutput": (400, 200),
                "WorkflowIOVideoOutput": (400, 200), "GetVideoComponents": (380, 140), "GetImageSize": (380, 120),
                "ComfyMathExpression": (380, 160), "StringCompare": (380, 200)})

FORMATS_SOURCE = "mp4,mov,webm,png,jpg,jpeg,webp"
FORMATS_IMAGE = "png,jpg,jpeg,webp"
# Load Source's outputs (frames, fps, audio, is_video, frame_count) -> what replaces each
SOURCE_OUT = {0: (1, 0), 1: (1, 2), 2: (1, 1), 3: (102, 2), 4: (101, 2)}


def file_input(name, formats, size_mb, desc, group, order):
    return {"param_name": name, "value": "", "accepted_formats": formats, "max_size_mb": size_mb, "required": True,
            "description": desc, "example": "", "advanced": False, "group": group, "order": order}


def text_input(name, value, desc, example, group, order):
    return {"param_name": name, "value": value, "required": False, "description": desc, "example": example,
            "advanced": False, "group": group, "order": order}


def definition():
    nodes = bw.definition_all()
    for nid, (ct, title, inp, pos) in list(nodes.items()):      # Load Source's outputs, from their new sources
        for k, v in inp.items():
            if isinstance(v, list) and v[0] == "1":
                inp[k] = bw.L(*SOURCE_OUT[v[1]])
    x0, y0 = nodes[1][3]
    left = x0 - 940
    nodes.update({
        100: ("WorkflowIOVideoInput", "Source", file_input(
            "Source", FORMATS_SOURCE, 500, "The generated image or clip whose label text needs fixing.", "Inputs", 0),
            (left, 0)),
        1: ("GetVideoComponents", "Source frames, audio and frame rate (an image: one frame)",
            {"video": bw.L(100, 0)}, (left + 470, 400)),
        101: ("GetImageSize", "Source frame count", {"image": bw.L(1, 0)}, (left + 470, 0)),
        102: ("ComfyMathExpression", "A clip: more than one frame", {"expression": "a > 1",
                                                                      "values.a": bw.L(101, 2)}, (left + 470, 190)),
        2: ("WorkflowIOImageInput", "Packshot", file_input(
            "Packshot", FORMATS_IMAGE, 100, "A clear photo of the real product with its label facing the camera.",
            "Inputs", 1), (left, 490)),
        103: ("WorkflowIOTextInput", "Product", text_input(
            "Product", "product", "What the product is, in a word or two.", "pump bottle", "Product", 2),
            (left, 980)),
        104: ("WorkflowIOTextInput", "Label Text", text_input(
            "Label Text", "", "The words printed on the real label, as they read; clips use them to check the result.",
            "GLOW SERUM\nDaily radiance drops\n30 ml", "Product", 3), (left, 1430)),
    })
    ct, title, inp, pos = nodes[11]                     # SAM 3's prompt: the product, from Product
    nodes[11] = (ct, title, {**inp, "text": bw.L(103, 0)}, pos)
    ct, title, inp, pos = nodes[7]                      # the product's copy, from Label Text
    nodes[7] = (ct, title, {**inp, "product_copy": bw.L(104, 0)}, pos)
    ct, title, inp, pos = nodes[93]
    nodes[93] = (ct, title, {**inp, "prefix": "refined_image"}, pos)
    ct, title, inp, pos = nodes[95]
    nodes[95] = (ct, title, {**inp, "prefix": "refined_clip"}, pos)
    # a keyframe that fails its audit: skip the clip (the job succeeds with the _AUDIT-FAIL keyframe) instead of the
    # Audit Gate's error, which a platform reports as a failed job with no file at all
    ct, title, inp, pos = nodes[91]
    assert ct == "ProductLabelAuditGate"
    gx, gy = pos
    nodes[105] = ("StringCompare", "Keyframe's audit failed: its name ends with _AUDIT-FAIL", {
        "string_a": bw.L(93, 0), "string_b": "_AUDIT-FAIL", "mode": "Ends With", "case_sensitive": True},
        (gx, gy - 260))
    nodes[91] = ("ProductLabelRunIf", "Keyframe on to the clip only when its audit has no FAIL (else no clip, no error)",
                 {"value": inp["keyframe"], "when": bw.L(105, 0), "run_when": "false"}, pos)
    ct, title, inp, pos = nodes[44]                     # the refined still (a clip: its refined keyframe)
    nodes[44] = ("WorkflowIOImageOutput", "refined_image (a clip: its refined keyframe; _AUDIT-FAIL when its audit fails)",
                 {"images": inp["images"], "output_name": "refined_image", "filename_template": "", "format": "png",
                  "include": True, "filename": bw.L(93, 0)}, pos)
    ct, title, inp, pos = nodes[65]                     # the refined clip
    nodes[65] = ("WorkflowIOVideoOutput", "refined_clip (_REVIEW-FAIL / _REVIEW-CHECK from its review)",
                 {"video": inp["video"], "output_name": "refined_clip", "filename_template": "", "format": "mp4",
                  "include": True, "filename": bw.L(95, 0)}, pos)
    return nodes


NOTES = {
    90: ("About this workflow", (-2200, 0), (560, 1250), (
        "AGENTIC PRODUCT LABEL REFINEMENT\n\n"
        "Makes the garbled text on a product's label legible, in an AI-generated image or a single-shot clip, from "
        "the real product's packshot. The generated product keeps its shape, position, label outline, colours and "
        "light; only the letters change. Results a check finds wrong are named, never passed: _AUDIT-FAIL on the "
        "image (a clip whose keyframe fails: that keyframe comes back, no clip, no error), _REVIEW-FAIL (wrong) or "
        "_REVIEW-CHECK (look first) on the clip.\n\n"
        "Inputs (WorkflowIO): Source (an image or a clip), Packshot, Product (what it is, e.g. pump bottle: SAM 3 "
        "finds it), Label Text (optional: the label's words, for clips).\n"
        "Outputs: refined_image (a clip: its refined keyframe) and refined_clip (mp4). The QA sheets, report.json, "
        "the clip's lossless master and its review summary are saved beside them.\n\n"
        "A clip is fixed in one shot: the shot the product is seen best in. For an edit with several shots, trim it "
        "to the shot first.\n\n"
        "The rest is the one workflow (product-label-refinement-all) at its defaults: an agent can still set the "
        "product mask, a corrected keyframe, a job, the keyframe frame or Refine settings in a run copy.\n\n"
        "## Node packs\n"
        "- ComfyUI-ProductLabelRefinement: https://github.com/nastyakolykhan-sudo/ComfyUI-ProductLabelRefinement "
        "@ 87d88fca4d6222fd1523cb7e2914e8c739e90b84\n"
        "- ComfyUI-MaskAspectCrop (ImageReframe, ImageReframeComposite): "
        "https://github.com/nastyakolykhan-sudo/ComfyUI-MaskAspectCrop @ 10b47888f08a297da1ba2f03f45ace3ab9594ad1\n"
        "- comfy-workflow-io-nodes (WorkflowIO*): https://github.com/aw93silverside/comfy-workflow-io-nodes "
        "@ 11e07cc007b2a81b4e4228d750c6192d14c179cd\n"
        "- Core ComfyUI 0.42.5 or later (SAM3_Detect, GetVideoComponents, ComfyMathExpression).\n\n"
        "## Models\n"
        "- checkpoints/sam3.1_multiplex_fp16.safetensors (Comfy-Org/sam3.1)\n"
        "- trackers/alltracker.pth (aharley/alltracker)")),
}


def groups():
    out = [("Inputs and outputs (WorkflowIO)", [100, 1, 101, 102, 2, 103, 104], "#3f789e"),
           ("Agent options (defaults; set in a run copy)", [3, 4, 5, 6, 7, 8, 9, 84], "#3f789e")]
    for title, members, colour in bw.ALL_GROUPS[1:]:
        out.append((title, members + [105] if 91 in members else members, colour))
    return out


def main():
    classes = bw.load_pack()
    nodes = definition()
    bw.build(NAME, nodes, classes, NOTES, groups())


if __name__ == "__main__":
    main()
