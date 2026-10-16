# ComfyUI-ProductLabelRefinement

Fix a hallucinated product in an AI-generated still by compositing the real product from its packshot, inside ComfyUI. The nodes warp the packshot onto the generated product's perspective and carry over the scene's light and colour. That includes cast shadows, falloff, gloss and haze, and the round light on cans and bottles. They match the frame's softness and grain, put the product behind whatever stands in front of it, and fill in the background where the generated product was bigger. Every run ends with QA sheets, a report, and a layered PSD if you want to finish by hand.

It is deterministic compositing with no models. Every measurement lives in a job file (`job.json`): the work box, landmark points, edge lines, outlines, foreground objects, and the grade, finish and fill settings. The same job gives the same pixels every time. That lets an agent or a script drive it through ComfyUI's API, check the report and QA sheets, edit the job and run it again.

A job doesn't have to be measured by hand first. **Auto Draft** makes one inside ComfyUI from rough input: the scene, the packshot, where the product roughly sits (four corners or a box), and optionally a product mask from SAM 3. On a clean scene the draft is close to final; on a cluttered shelf it is a scaffold to finish by hand. [Starting from rough input](#starting-from-rough-input-auto-draft) has the scores.

**Video too.** Product Label: Refine (Video) carries a corrected keyframe's label through a whole clip: track the product, fit its surface on every frame, re-project the true label with each frame's light, and change nothing outside the changed pixels. See [Video: the label through a clip](#video-the-label-through-a-clip).

**Renamed from ComfyUI-ImagePost (2026-10-16).** The pack now matches the product-label-refinement skill it ports. GitHub redirects the old URL, so existing clones keep pulling. The node IDs (`ImagePostLoadJob`, `ImagePostAutoDraft`, ...) are unchanged, so saved workflows still load; only the display names ("Product Label: ..."), the menu category, the template file names and the default output folder (`output/product-label-refinement/`) changed. To rename an existing clone's remote: `git remote set-url origin https://github.com/nastyakolykhan-sudo/ComfyUI-ProductLabelRefinement`.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nastyakolykhan-sudo/ComfyUI-ProductLabelRefinement
```

It needs no extra dependencies: numpy, scipy and Pillow (10.1 or later) ship with ComfyUI. Restart ComfyUI. The nodes are under **image › product-label-refinement**.

The video nodes also need AllTracker's weights in `ComfyUI/models/trackers/` and, optionally, matplotlib for their QA plot. See the [Swordfish checklist](#swordfish-checklist).

## Try the demo

1. Copy `examples/demo/imagepost_demo_scene.png` and `examples/demo/imagepost_demo_packshot.png` into `ComfyUI/input/`.
2. Load `workflows/product-label-refinement.json` and queue it.
3. The refined image lands in `output/product-label-refinement/`, the QA sheets in `output/product-label-refinement/qa/`, and the layered PSD next to the image.

To start from rough input instead, load `workflows/product-label-refinement-draft.json`: Auto Draft takes the place of Load Job, with the card's corners read a few px off, the mug traced roughly and two landmarks clicked near the header band. It also appears in ComfyUI's template browser, under ComfyUI-ProductLabelRefinement. The draft job it made is saved as `output/product-label-refinement/draft_00001_.json`.

The demo is a made-up product card. In the "generated" scene its text is garbled ("PRDOUCT", "sapmle lable"), its emblem and grid are wrong, its corners are square and its top is too tall. A mug stands in front of its lower right corner and casts a shadow across it. The packshot is the clean card. `tools/make_demo.py` draws all three files from code, so they show no real product or brand. `examples/demo/job.json` is the worked example of a job.

## Nodes

There is one node per step. The math is the product-label-refinement engine's (`imgpost/`, see [Maintaining](#maintaining)), split at the node boundaries in `stages.py`.

| Node | Inputs | Outputs | Step |
|---|---|---|---|
| **Load Job** | `job_json` (text) or `job_path` | `job` | reads and checks the job |
| **Auto Draft** | scene, reference, `quad`, `model`, `light`, `suggest`, `filename_prefix`, `product_mask`?, `reference_mask`?, `pairs`, `occluders`, `outline`, `mask_skip` | `job`, `job_json`, `draft_points`, `report` | a first job from rough input, in place of Load Job; saves it as JSON ([Starting from rough input](#starting-from-rough-input-auto-draft)) |
| **Job Masks** | job, scene | `old_silhouette`, `occluder_mask` | rasterises the generated outline and the foreground objects |
| **Fit Warp** | job | `homography`, `report` | homography, affine or cylinder fit (labels on bottles), with the optional residual edge correction; shows the residual table |
| **Warp Reference** | job, homography, scene, reference, `reference_mask`? | `warped`, `product_mask`, `align_overlay`, `coords` | supersampled warp in linear light; warns when the outline takes in the packshot's backdrop (a light rim) or a bottle's measured edge sits past its drop shadow |
| **Grade To Scene** | job, scene, warped, masks, `colour`, `shading`, `reference`?, `coords`?, `relight`? | `graded`, `grade_preview`, `report`, `relight_map`, `ink` | light falloff, colour, the scene's paper field, cylinder shading, relight; `ink` feeds Composite Behind for jobs with `matte: ink` |
| **Match Finish** | job, scene, graded, masks | `product`, `report` | highlight roll-off, blur and grain match |
| **Fill Leftovers** | job, scene, masks, `coords`? | `background`, `fill_mask`, `report` | harmonic fill; stops with FILL STOPPED. With `shape: generated` it needs Warp Reference's `coords` |
| **Composite Behind** | job, scene, background, product, product_mask, occluder_mask, `ink`?, `coords`? | `image`, `matte` | composite and paste into the scene; with `matte: ink`, only the print goes over the scene's own surface; with `shape: generated`, the generated product's own outline is the matte (needs `coords`) |
| **QA Sheet** (audit) | ..., `old_silhouette`?, `occluder_mask`?, `coords`? | ..., `audit` | with the masks connected the run audits itself: a missed occluder, a rim of the generated product, a thick fill, print pressed into a bottle's edge, edges off; FAIL blocks delivery |
| **QA Sheet** | job, before, after, reference, matte, fill_mask, reports? | `compare`, `edges`, `before_after`, `report` | QA sheets and `report.json` |
| **Save PSD** | job, scene, image, background, fill_mask, product, matte, `filename_prefix` | (file) | layered PSD for hand finishing |

`?` marks optional inputs. The masks inputs are `product_mask`, `old_silhouette` and `occluder_mask`.

A switch set to `job` does what the job says, so the switches only ever override it:

- `colour`: `job` uses `grade.colour` (default `scene`). `scene` fits tone curves to the generated product. `reference` keeps the real product's colours and takes only light, white balance and black level from the scene (`grade.white_box`, `black_box`, `ref_white_box`). Use `reference` whenever the generated colours are off, as in the demo. It needs `reference` connected when the job has `ref_white_box`.
- `shading`: `job` applies `grade.shading` for cans and bottles: the generated product's round light, highlight streaks and core shadow. It needs `coords` from Warp Reference. `off` skips it, for comparison.
- `relight`: `job` applies `grade.relight`, which brings back the light the smooth grade misses: cast shadows with their crisp edges, falloff and glows, plus gloss or wrap haze with `sheen`. `relight_map` shows what it changed (blue darker, red brighter, sheen in white). `off` skips it, for comparison.

Save PSD writes three layers, bottom to top:
- `original scene`;
- `background fill`, opaque only where the fill ran;
- `product`, with matte × foreground visibility as its layer mask.

With every layer visible it flattens to the PNG within 2 levels (its report: `layers_vs_png`). In the editor it runs on every queue, so mute it (Ctrl+M) when you don't need a PSD. `tools/comfy_run.py` runs it only with `--psd`.

## Starting from rough input (Auto Draft)

Auto Draft runs the engine's `run.py draft` (`imgpost/draft.py`) and hands its job to the other nodes, in place of Load Job. It:

- traces the packshot's outline (on a plain background, or from its alpha), unless you give `outline`;
- places it from the rough input and refines each landmark pair by colour correlation, keeping only trusted ones;
- proposes edge lines from the outline's straight sides where the generated edge is straight and in place;
- snaps the old silhouette and each occluder to its edge;
- sets the work box, colour (`reference` when the packshot has a white area) and relight.

With `label` set to `cylinder` (a label on a bottle or jar) or `flat` (a label on a flat or flattened face, such as a tube's panels), it drafts the label only, by the engine's recipe for labels (its `references/stills/measuring.md`, "Labels on bottles and jars"):

- measures the generated label's own outline from its 4 corners, each side walked along the curve it bows to (or along `product_mask`'s contour), and keeps it as the label's visible shape (`shape: generated`, the real label bled 10 px past it);
- on a bottle, takes the axis and radius from the packshot's silhouette at the label's height, leaving out a drop shadow: the limb is where the label or glass meets the shadow;
- where the label wraps round the bottle (a side of `outline` within 4 px of a limb, or past it), redraws that side on the limb, 2 px past it, so the real label reaches the silhouette even when `outline` was drawn short of it (a product mask's tilted rectangle had left strips the fill couldn't box);
- makes the label's sides the fit's edges: its top and bottom as curves, the side where it meets the bottle's silhouette as a limb, plus a residual correction; the packshot's sides are snapped onto the packshot's own edges first;
- sets `cylinder.limb_min` from how far round the packshot's print reaches, so the print isn't pressed into the silhouette;
- grades and finishes it as the recipe does (colour `reference`, `ref_flatten: field`, `white_field`, clean paper, edge bleed, auto sharpen and prefilter) when the packshot's label shows white paper; on print straight on a coloured bottle, the plain white-level grade with the scene's colour. `light` is not used.

Inputs, all in pixels (scene px unless said otherwise; numbers separated by spaces, commas or new lines):

- `quad`: where the packshot outline's bounding box corners land in the scene, `x,y x,y x,y x,y` (top left, top right, bottom right, bottom left). A box `x0,y0,x1,y1` around the product also works when the product stands upright and faces the camera. A few px off is fine. Pick it on the scene the node shows ([below](#picking-the-quad-on-the-scene)) or type it. With `label` on: the generated label's own 4 corners, in the packshot label's order, within about 3 px (where a side is rounded, the sides extended).
- `product_mask` (optional): the generated product's mask, scene-sized, from SAM 3, the mask editor, or any node with a MASK output. The outline is then fitted to the mask's contour instead of to image edges. With a mask, `quad` may stay empty: the fit starts from the mask's box, its hull's corners and its main axis, and keeps the start that sits best on the contour. That works when nothing stands in front of the product; when something does, give the quad too.
- `pairs` (optional): rough landmark pairs, `rx,ry=sx,sy` (reference px = scene px), one per line. Points on garbled generated print usually come back untrusted and are ignored.
- `occluders` (optional): objects in front, one per line: `left: x,y x,y ...` (the occluder fills that side of the line: `left`, `right`, `above` or `below`) or `polygon: x,y x,y ...`. Each is snapped to the occluder's own edge.
- `outline` (optional, reference px): the product on the packshot, as a box or a polygon, when the packshot shows more than the product (a box and its tube, a label on a bottle). With `label` on it is needed: the label on the packshot, upright.
- `mask_skip` (optional, reference px): a polygon whose part of the outline isn't fitted to the mask (where the mask runs on into a cap the job keeps).
- `model`: `homography` or `affine`. `light`: `relight`, `cylinder` (a shading template for cans and bottles) or `off`. `suggest`: also add packshot corners that match the scene well as landmarks.
- `label` (optional, last): `off`, `cylinder` or `flat`, as above. With `product_mask` it is a mask of the generated label: it gives each side's curve, and with `quad` empty its corners too.

Outputs:

- `job`: feeds the other nodes, exactly like Load Job's.
- `job_json`: the job as text. Paste it into Load Job to finish it by hand, or into the engine's job folder; add `scene` and `reference` paths to run it with `run.py`.
- `draft_points`: each landmark pair as packshot under the fit, then scene.
- `report`: the draft's summary plus the align table. The node shows it, and each run also saves the job as `{filename_prefix}_NNNNN_.json` in the output folder.

Check `align_overlay` and the report before trusting a draft: a side marked `left` means the generated edge disagreed and the fit doesn't use it. The rough corners stay in the job as weak placeholders (`w` 0.15 or 0.05); replace them with landmarks you read.

### Picking the quad on the scene

Auto Draft shows the scene under its widgets: the file of the Load Image that feeds `scene` (through Reroutes), or another node's preview once the graph has run. The picker (`web/quad_picker.js`) only edits the `quad` text, so the node's inputs, saved workflows, the API templates and `tools/comfy_run.py` work as before.

- Click the corners in order: top left, top right, bottom right, bottom left. The quad is written as `x,y x,y x,y x,y` in scene px when the fourth corner is placed.
- Drag a corner to adjust it. The last corner clicked or dragged moves 1 px per arrow key (Shift: 10 px).
- The wheel zooms about the pointer, up to 16 screen px per scene px. Middle-drag or Shift-drag pans, and so does dragging empty space once the four corners are placed. **Fit** shows the whole scene; **Clear** empties the quad to pick it again.
- A typed quad shows on the scene, and a box `x0,y0,x1,y1` shows as its four corners (drag one to make it a quad). The header turns amber when the corners cross or run anticlockwise, which means they were picked out of order.
- For a product without corners, such as a bottle, place the corners of a box around it. For a hidden corner (behind the mug in the demo), follow the two sides that meet there.

Zoom in on the corners before you queue. On the demo, corners clicked on the whole scene came within 2 to 3 px of the hand fit, and the draft left out a second side of the outline (its generated edge 2.3 px from where the rough fit put it). The two bottom corners were then dragged onto the card at 7× zoom, which put them on the hand fit's corners. The draft then kept the same sides as with the template's typed quad. It scored 0.9 / 2.4 px on the edge and 0.9 / 2.6 px on placement (engine 0.10.0, `regress.py --draft --outline traced --quad ...`). At 0.10.1, whose outline tracing moves this packshot's outline by under 1 px, a quad 3 px off at every corner with the same traced outline scores 0.4 / 4.0 and 1.7 / 4.1.

### How close a draft gets

Each draft was scored against the hand-measured job for the same product (`tools/regress.py --draft`, below). The rough input is what a person would give: the quad 3 to 4 px off at every corner, the product's outline on the packshot, and the occluders 2 px off. The mask is the hand-measured generated outline minus what stands in front of it, which stands in for a good SAM 3 mask; no real SAM masks were scored.

- **Edge**: how far the draft puts the product's outline from where the hand fit puts it (median / max, scene px).
- **Placement**: how far any point of the product, its print, lands from the hand fit's position (median / max).
- **Geometry vs hand**: the hand job run with the draft's measurements (work box, fit, occluders, old silhouette) and its own grade: pixels more than 8 levels from the hand job's image, and their share of the pixels the hand job changed.
- **Draft vs hand**: the same for the draft as it comes, with its default grade (the hand jobs tune theirs).
- **Audit**: the draft run's own verdict, with the hand job's in brackets. The hand jobs' WARNs were checked as false positives before delivery.
- **With mask**: edge and placement with the mask added to the quad. **Mask alone**: with no quad.

All columns but the last two use the quad and no mask. The six other rows' mask columns are from 0.9.0 (their code is unchanged and their quad rows score the same). The seven label rows draft with `label` on (`regress.py --draft` sets it from the hand job), their quad the generated label's corners 3 to 4 px off. Results on 2026-10-16, engine 0.10.2 for the six bottle-label rows (all three columns; 0.10.2 redraws a wrapping side of the packshot outline on the limb) and 0.10.1 for the panels' mask columns (the other rows are unchanged from 0.9.0):

| Product | Edge | Placement | Geometry vs hand | Draft vs hand | Audit | With mask: edge, placement | Mask alone: edge, placement |
|---|---|---|---|---|---|---|---|
| the demo card, a mug in front (1024 × 768) | 0.5 / 2.8 | 0.7 / 2.8 | 12,324 (13%) | 12,349 (13%) | WARN (WARN) | 0.6 / 7.2; 3.5 / 7.2 | 1.2 / 19.2; 6.8 / 19.6 |
| a box on a plain studio set (2286 × 4096) | 1.4 / 4.7 | 14.4 / 19.8 | 161,719 (32%) | 471,067 (93%) | WARN (WARN) | 1.0 / 8.1; 10.1 / 13.7 | 0.5 / 1.5; 1.4 / 1.9 |
| a lip tube on the same set: cap kept, 10 glints in front | 1.1 / 3.7 | 3.7 / 6.0 | 46,746 (24%) | 111,497 (56%) | WARN (WARN) | 1.3 / 2.0; 2.8 / 3.8 | 2.4 / 76.9; 38.3 / 76.7 |
| a tube's printed panels in a mesh bag (4096²), label `flat` | 0.6 / 2.3 | 0.9 / 2.4 | 40,304 (11%) | 61,441 (16%) | clean (WARN) | 0.6 / 2.3; 0.9 / 2.4 | 0.6 / 2.3; 0.8 / 2.4 |
| a dropper bottle's label in the same bag, bottle kept, label `cylinder` | 0.4 / 3.1 | 0.5 / 2.8 | 30,531 (12%) | 31,278 (12%) | WARN (WARN) | 0.4 / 3.1; 0.8 / 2.8 | 0.4 / 2.5; 1.2 / 2.7 |
| a bottle's label in the same bag, label `cylinder` | 0.3 / 3.5 | 10.0 / 12.4 | 99,356 (17%) | 101,490 (17%) | WARN (WARN) | 0.3 / 4.0; 6.2 / 8.1 | 0.3 / 4.0; 8.3 / 10.8 |
| a bottle label in a cup carrier (1024²), label `cylinder` | 0.1 / 0.2 | 0.2 / 0.3 | 1,261 (11%) | 1,515 (13%) | clean (clean) | 0.1 / 0.9; 1.0 / 1.3 | 0.1 / 0.6; 3.2 / 3.8 |
| a frosted bottle's label in the same carrier, label `cylinder` | 0.2 / 0.7 | 2.9 / 3.5 | 3,410 (20%) | 3,481 (20%) | clean (clean) | 0.1 / 0.7; 0.3 / 0.6 | 0.3 / 1.8; 2.5 / 3.0 |
| a bottle label in the same carrier, its wall over a corner, label `cylinder` | 0.1 / 0.7 | 0.3 / 0.6 | 1,865 (17%) | 2,145 (19%) | clean (clean) | 0.1 / 1.0; 0.3 / 0.9 | 0.1 / 0.5; 0.3 / 0.6 |
| print on a bottle on a video's anchor frame (1080 × 1920), label `cylinder` | 0.8 / 9.1 | 5.7 / 10.7 | 7,709 (34%) | 7,491 (33%) | WARN (FAIL) | 1.0 / 4.4; 8.6 / 10.3 | 0.9 / 5.3; 10.2 / 12.2 |
| a blister card on a cluttered shelf, behind a box and the shelf lip (4096²) | 0.8 / 2.8 | 1.9 / 5.3 | 74,776 (52%) | 117,754 (81%) | WARN (WARN) | 3.3 / 18.7; 13.2 / 20.8 (stopped) | 5.3 / 75.1; 35.3 / 75.5 (stopped) |
| a text block printed on a tube's body (ink matte) | 0.6 / 2.8 | 1.2 / 2.7 | 10,987 (24%) | 41,231 (90%) | WARN (WARN) | 10.4 / 40.5; 20.0 / 40.4 | 20.5 / 43.1; 34.4 / 43.0 |
| a user-fixed render as the reference (identity fit) | 0.7 / 2.2 | 1.1 / 2.7 | 34,757 (33%) | 100,092 (95%) | WARN (WARN) | 3.5 / 6.3; 2.9 / 6.1 | 4.6 / 51.8; 23.5 / 51.6 |

What it shows:

- **Flat products with a quad: close to final.** The demo card, a tube's printed panels (`label: flat`) and the blister card on the cluttered shelf (given its occluders) land within 0.5 to 0.8 px of the hand outline (median; 2.3 to 2.8 px at worst), with their print within 2 px (5.3 px at worst). The shelf card is no worse than the clean scenes once the quad and the occluders are given; what the shelf costs is the occluders to trace and the edges the draft can't use (only one of its sides passed).
- **A tall box needs landmarks or a mask.** With no landmarks the box's perspective comes from four corners each 3 px off, and its print slides up to 20 px along its long sides while its outline stays within 5 px. A mask alone fixes that: outline within 1.5 px and print within 1.9 px at worst.
- **A mask helps where nothing stands in front.** With the quad, a mask brings the tube's worst edge from 3.7 to 2.0 px; in label mode it brings the frosted label's print from 2.9 to 0.3 px and the large bottle label's from 10 to 6. Alone, it fails where something stands in front: 19 px on the demo (the mug), 77 px on the tube (the cap and glints), and on the shelf card the run stops with FILL STOPPED (the mask's contour along the box in front boxes in a sliver). Give the quad as well whenever the product is partly hidden.
- **Labels on bottles, with `label` on: close to final.** Four of the six paper labels land within 0.2 to 0.9 px of the hand fit's print (median; 2.8 px at worst) and the frosted one within 2.9 px, against 2 to 53 px for the flat draft. Their pixels against the hand job's image fell from 52 to 94 % to 12 to 20 % of what the hand job changed, and every audit is as clean as the hand job's or cleaner (the carrier label that FAILed over the wall is now clean). With a mask of the label the frosted label comes within 0.3 px.
- **Where a label draft stays off.** The other bottle label in the bag (4096 px) has its print 10 px round the bottle from the hand fit's (6 with a mask, 8 with a mask alone), and the frosted label's 2.9 px comes from half a pixel on its bottom edge. In the cylinder model a label's width and its turn about the axis trade off, and the edges barely tell them apart (the engine's pitfall 89): the hand fits sit on that ridge too (that label's own measured outline moves its hand fit 7.5 px). Both results read as the real label; check the print's margins against the packshot. From a mask alone, the first carrier label's corners come out about 1 px from the hand outline's, and that turns its print 4 px round the bottle (1.3 px at 0.10.0, from a slightly different trace of the same mask): give the quad with the mask on small labels. The anchor frame isn't a paper label: the print is on the bottle, and its outline on the packshot is a box whose top and bottom aren't edges in either image (12 and 17 packshot px off the collar line and the base), so it lands 5.7 px off. It has no white paper either, so the draft keeps the plain grade there (the paper grade streaked it). The hand job placed it by landmarks on the generated wordmark; give pairs there, or measure the print's position by hand.
- **Pixels.** Even with the hand job's grade, the closest drafts (the demo, the panels, three of the labels) differ from its image by more than 8 levels on 11 to 16 % of the pixels it changed, the rest on 17 to 52 %: a 1 px shift moves every print edge. A text block on a tube's body and a fixed render as the reference are in the table for completeness; neither is a case for a draft.

## Video: the label through a clip

Correct one frame of a clip, by hand or with an image model given the real product as reference. **Product Label: Refine (Video)** carries that label through the shot:
1. it tracks the product (AllTracker);
2. fits its surface on every frame;
3. re-projects the true label with each frame's light, softness and grain;
4. composites it inside the product's mask.

The text is right on every frame, the timing is the source's, and the pixels outside the changed area come back exactly as they went in. It's the product-label-refinement skill's video engine (`imgvideo` 0.3.1), and the skill's own `video.py --refine` gives the same pixels (see Regression).

### Swordfish checklist

1. **ComfyUI 0.19.3 or newer.**
   - The video nodes themselves run on 0.18.2.
   - SAM 3 (`SAM3_Detect`, `SAM3_VideoTrack`) and the SAM 3 helper below need 0.19.3, as does a label-fix workflow that segments with SAM 3.
2. **AllTracker's weights:** `ComfyUI/models/trackers/alltracker.pth`.
   - That's the `models/trackers` folder, or a `trackers` folder beside the standard model folders wherever they're mounted from: a volume that `extra_model_paths.yaml` maps for `checkpoints`, `sams` and the rest is searched too, so `trackers/alltracker.pth` at that volume's top level is found without editing the yaml. On Swordfish, add it from the Models page with the HuggingFace button (repo `aharley/alltracker`, file `alltracker.pth`, custom path `trackers`), then restart ComfyUI and reload the page.
   - Get them from `https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth`: 66,005,722 bytes, sha256 `ffd9ebcfb6d206d594b646999a150540f92c049cf9b2bf940facf7123f62aa1d`, MIT.
   - The node lists what's in that folder under `tracker` and never downloads anything.
3. **SAM 3.1:** `ComfyUI/models/checkpoints/sam3.1_multiplex_fp16.safetensors`, the checkpoint the label-fix workflow already loads. Its licence is Meta's SAM License.
4. **pip:**
   - `matplotlib` (optional): without it the QA plot is skipped and everything else runs.
   - `torchvision`, `einops` and `av` ship with ComfyUI.
   - `tesseract-ocr` (apt, optional): OCR in the QA when the settings give the label's copy (`words`).
5. **First run:**
   1. Copy `examples/video_demo/labelvideo_demo_clip.mov`, `labelvideo_demo_mask.mov` and `labelvideo_demo_keyframe.png` into `ComfyUI/input/`.
   2. Load `workflows/product-label-refinement-video.json` and queue it.

   Expected:
   - 40 of 40 frames refined;
   - the report says the keyframe is frame 20;
   - 0 px changed outside the matte;
   - the comp file in `output/product-label-refinement/video/refined_00001.mov`, plus an mp4 review copy.
6. **A real clip:** the same template with your clip, its mask video (white = product) and the corrected keyframe. Set `processes` to 8 on a server (same pixels, faster render).
   - **Inputs from a label-fix workflow:**
     - `frames`: the frames (after any resize);
     - `mask`: the video masks (SAM 3);
     - `keyframes`: the corrected keyframe;
     - `keyframe_bbox`: ImageCropByMaskToAspect's `bbox`, when the keyframe was corrected on a zoomed crop;
     - `keyframe_index`: LoadVideoTrim's `keyframe_index`.
   - **Out:** `frames` goes straight to SaveVideo and Save Lossless; nothing needs pasting back. With crops instead of full frames, paste back through `changed`.
   - **Then read the report:**
     - `not_covered` and `partly_covered` say what to do about each stretch (`chain`, `reacquire`, or a new `keyframe` at the suggested frame): `not_covered` holds frames left as they are, `partly_covered` frames refined with part of the label kept from the source because it turned past what the keyframe saw;
     - `flagged` lists frames to look at;
     - `review` has the automatic checks.

### Video nodes (image › product-label-refinement › video)

| Node | In | Out |
|---|---|---|
| **Product Label: Refine (Video)** | `frames` (IMAGE batch), `mask` (MASK batch), `keyframes` (IMAGE, one per shot), `tracker`; optional `keyframe_index`, `keyframe_frames`, `keyframe_bbox`, `product` (context only), `settings` (JSON), `fps`, `processes`, `model` (tube, or flat: experimental) | `frames` (same count, size and order; outside the changed pixels the input itself), `matte` (the mask plus patched frames), `changed`, `report` (JSON), `qa_image` |
| **Product Label: Save Lossless (Video)** | `images`, `fps`, `filename_prefix`, `format` (QuickTime PNG: exact; PNG sequence; ProRes 4444: for editors, not exact), `audio`? | the file in `output/` (never H.264) |
| **Product Label: SAM 3 Track From Keyframe** | `images`, `model` (SAM 3), `keyframe_mask`, `keyframe_index`; optional `conditioning`, `detection_threshold`, `max_objects`, `detect_interval` | `masks`: the keyframe's mask propagated forward from it and backward on the reversed frames, one per frame. ComfyUI core's `SAM3_VideoTrack` seeds only the first frame. **First test on Swordfish** (it needs 0.19.3 and the SAM 3.1 checkpoint) |

**What Refine measures by itself:**
- shots and cuts;
- each keyframe's frame;
- the label's band along the product;
- print polarity and stroke;
- the shot each keyframe covers;
- a colour-key patch for frames where the mask has a gap;
- the stretches it left as they are (`not_covered`) and the stretches refined with part of the label kept from the source (`partly_covered`), each with a plan and a suggested frame.

**`settings`** (JSON) overrides any of that:
- the engine's vjob sections: `track`, `fit`, `render`, `qa`, `review`, `artwork`, `key`;
- `anchor` (band, model);
- `shots` (`[[a, b] or null]` per keyframe);
- `words` (the label's copy, for OCR);
- `matte_patch: false`.

Typical overrides after a first look at the report: `{"render": {"spark_glow": [70, 71, 72], "occluder_feather": 20}}` for a sparkle trail across the label.

**Product types.**
- **`model: tube`** (the default): anything round, such as bottles, jars, tubes and lipsticks. Every delivered example so far was a tube.
- **`model: flat`** (experimental: it runs end to end but hasn't been tried on a real clip): one flat face. For a box, run one Refine node per visible face, each with that face's mask and its own keyframe, chaining `frames` out of one into `frames` of the next.
- **Small or metallic print** (lipsticks): track on the print (`{"track": {"on": "print"}}`), and correct the keyframe on a zoomed crop (`keyframe_bbox`).

**Re-runs** on the same frames, mask and keyframes keep the tracks, fit and still label from the last run (in ComfyUI's temp folder), so a render-only change re-renders without re-tracking.

**Licences:**
- AllTracker: code vendored in `imgvideo/alltracker` (MIT, LICENSE beside it); its weights are MIT too.
- SAM 3: Meta's SAM License; check its use restrictions for your case.
- No CoTracker: non-commercial, and refused here.

## Conventions

- **Work box.** `scene` is the whole generated frame, and the job's `roi` is the work box. Every other IMAGE or MASK passed between Product Label Refinement nodes covers only the work box. Composite Behind pastes it back, and every pixel outside the box passes through untouched.
- **Masks.** Mask inputs also accept a scene-sized mask, which they crop to the work box, so a mask from the mask editor or another node can stand in. `old_silhouette` is thresholded at 0.5. `occluder_mask` is 1 where something is in front of the product.
- **Product layers.** `warped`, `graded` and `product` are sRGB like any IMAGE, but values above 1.0 are kept as highlight headroom, and Match Finish rolls them off. Don't clamp them in between.
- **One image per job.** Batches larger than 1 raise an error.
- **Images come from Load Image nodes.** The job's `scene`, `reference` and `output` paths aren't read here. When the job has `scene_size` or `reference_size`, the connected images must match, which catches a wrong wire. A transparent packshot (`reference_outline` type `alpha`) needs the reference Load Image's MASK connected to `reference_mask`.
- **Output.** Composite Behind quantises the work box to 8 bit, so Save Image writes the same pixels the engine saves. Save Image and Save PSD don't keep the scene's ICC profile or alpha channel; the engine's own runner does.
- **Several products in one frame.** Run one job per product, back to front, and feed each run's image in as the next run's scene.

## Running it from an agent (ComfyUI API)

`tools/comfy_run.py` uses only the standard library. It uploads the images, swaps in a job (or sets the draft's input), queues the prompt, waits, then downloads everything:

```bash
python3 tools/comfy_run.py --server http://127.0.0.1:8188 \
  --workflow workflows/product-label-refinement.api.json \
  --scene examples/demo/imagepost_demo_scene.png --reference examples/demo/imagepost_demo_packshot.png \
  --job examples/demo/job.json --out ./ip-run
```

- `--scene` and `--reference` upload to `input/product-label-refinement/` and set the Load Image nodes that feed Warp Reference. `--image NODE_ID=PATH` targets any other Load Image node.
- `--job` replaces the job in every Load Job node. Without it, the workflow's embedded job runs.
- `--set NODE.INPUT=VALUE` sets a widget value on node `NODE`, given as an id or as a class name for every node of that class. With the draft workflow, that is how the rough input goes in.
- `--mask PATH` uploads a product mask (white = product, scene-sized) and adds a Load Image (as Mask) node that feeds every Auto Draft node's `product_mask`.
- `--stage align` runs only the fit, warp and `align_overlay`. `--stage grade` adds `grade_preview` and `relight_map`. Use them to check the alignment, then the grade, before a full run.
- `--psd` also runs Save PSD, as does `"psd": true` in the job.
- `--out` gets the outputs in the server's folder layout: images, QA sheets and the PSD. It also gets `report.json` (the QA node's report), `ImagePostFitWarp-5.txt` (the residual table), `ImagePostSavePSD-19.txt`, `prompt.json` and `history.json`.
- Exit codes: 0 done, 2 prompt rejected (validation errors printed), 3 node failed (for example `FILL STOPPED ...`, printed with the node), 4 timeout.
- For a server behind an auth proxy, pass `--header "Name: value"` (repeatable).

A draft from rough input, run through to the refined image:

```bash
python3 tools/comfy_run.py --server http://127.0.0.1:8188 \
  --workflow workflows/product-label-refinement-draft.api.json --scene scene.png --reference packshot.png \
  --set "ImagePostAutoDraft.quad=372,170 612,186 610,566 370,578" \
  --set "ImagePostAutoDraft.pairs=" --set "ImagePostAutoDraft.occluders=polygon: 593,428 593,662 705,662 705,428" \
  [--mask product_mask.png] [--stage align] --out ./ip-draft
```

Clear the template's demo `pairs` and `occluders` (as above) unless you set your own. `--out` then also holds `ImagePostAutoDraft-3.txt` (the draft's summary and align table) and `product-label-refinement/draft_00001_.json`, the draft job to check, edit and run with `--job` through `workflows/product-label-refinement.api.json`.

The same steps with curl. Upload both images:

```bash
curl -s -F "image=@scene.png" -F "subfolder=product-label-refinement" -F "overwrite=true" http://127.0.0.1:8188/upload/image
```

Point nodes 1 (scene) and 2 (reference) at `product-label-refinement/<name>`, then queue the prompt:

```bash
jq '{prompt: (.["1"].inputs.image = "product-label-refinement/scene.png" | .["2"].inputs.image = "product-label-refinement/packshot.png")}' \
  workflows/product-label-refinement.api.json > prompt.json
curl -s -H "Content-Type: application/json" -d @prompt.json http://127.0.0.1:8188/prompt
```

This returns `{"prompt_id": "..."}`. Poll until the entry appears, then read `outputs`:

```bash
curl -s http://127.0.0.1:8188/history/<prompt_id>
```

In `outputs`:
- each Save Image node lists `images`;
- the QA node lists `text`, which holds `report.json`;
- Save PSD lists `files`.

Fetch each file:

```bash
curl -s -o refined.png "http://127.0.0.1:8188/view?filename=imagepost_demo_scene_refined_00001_.png&subfolder=product-label-refinement&type=output"
```

A failed run has `status.status_str: "error"`, with the node and `exception_message` under `status.messages`.

## Writing a job

`examples/demo/job.json` shows every section. The main ones:

- `roi`: the work box in scene px. The product plus about 40 px of background.
- `reference_outline`: the real product's outline in packshot px. Types: `rounded_rect`, `profile`, `polygon`, `key_white`, `alpha`. `extend` lets an edge run on behind a foreground object.
- `align`:
  - `points`: 3 or more landmarks read on both images, such as wordmark or panel corners.
  - `lines`: visible outline edges.
  - `y_only` / `x_only`: rows or columns of repeated items.
  - `model`: `homography`, `affine`, or `cylinder` for print on a bottle or jar (`cylinder`, `limbs`, `"curve": true` lines).
  - `residual`: a smooth correction onto the generated edges a rigid fit can't follow.
  - `cylinder.limb_min`, `"one_sided": true` lines: keep a label from being turned until its print is pressed into a bottle's edge.
- `old_silhouette`: the generated product's outline in scene px. The fill restores the background inside it wherever the real product doesn't reach. `auto` (or `{"type": "auto", "measured": [...]}`) takes the fitted outline, plus measured slivers where the generated product sticks out; Job Masks then needs Warp Reference's product_mask.
- `reference_outline.body`: polygons where the real product shows its own body (a tube's band between two labels). They take the scene's own product colour; Grade To Scene, Match Finish, Fill Leftovers and Composite Behind then need Warp Reference's `coords`.
- `grade.paper_clean` and `finish.edge: bleed`: the packshot's paper grain, banding and edge falloff give way to the scene's paper; no rim along the outline.
- `shape: generated` (labels on a bottle the job keeps): the visible product takes the generated label's measured outline (`old_silhouette.measured`), and the real label, bled past its edge (`reference_outline.bleed`, reference px), fills it. Its corners, a narrower stretch, glass beside it and a wall over a corner stay as the frame drew them. The QA Sheet then also reports `shape` and audits `cut` (real print falling outside the shape).
- `occluders`: objects in front, each one of:
  - a `polyline` plus a `side`;
  - a `polygon`;
  - either of those with `soften` for an out-of-focus edge.
- `grade`: `gain` (`white_level`, `luma_ratio`, `none`), `colour`, `shading`, `relight`. On printed packaging whose generated print is laid out differently, `luma_ratio` with `match_hue` reads the light only where both images show the same ink; `protect_white` keeps the packshot's clipped whites from being dimmed. For labels, `ref_flatten` takes the packshot's own light off its paper and `white_field` lays the generated label's paper colour (light, falloff, tint) under it.
- `finish`: `blur` and `grain` (`auto` or a value), `edge_softness`, `rolloff`, `sharpen: auto` (a packshot softer than the frame).
- `fill`: `dilate`, `orphan_px`.
- `prefilter`: `auto` for a small packshot enlarged in the frame. `matte`: `ink` for text printed on a coloured body.
- `psd`: true to also write the layered PSD.

Measure landmarks and edges by colour change, not by the strongest gradient, and accept a fit at 1.5 px or less on the visible edge lines. The product-label-refinement engine's job spec documents every field.

## Regression

**Video.** `tools/regress_video.py` calls Product Label: Refine (Video) the way ComfyUI does (tensors in), on the same decoded frames as the skill's `video.py --refine`, and compares frames, matte, changed mask and report.

| Run (2026-10-16, imgvideo 0.3.0) | Result |
|---|---|
| The video demo (40 frames, 432 x 768, one keyframe): the node in-process vs `video.py --refine`, with and without the label's copy (`words`) | identical: frames 0 px, matte, changed mask and report; pixels outside `changed` are the input's |
| The video template on a live ComfyUI 0.18.2 (CPU, no matplotlib), the demo | success in 2 min: 40 of 40 frames refined, keyframe found at frame 20, 0 px changed outside the matte |
| A real 169-frame 1080 x 1920 clip with one corrected keyframe, the template on a live ComfyUI 0.18.2, `processes: 8` (CPU) | success in 20 min: frames 0-121 refined; the shot after the cut (122-138) reported as "reacquire, a cut at its edge", suggested keyframe 136; 0 px changed outside the matte. ComfyUI's video decode differs from ffmpeg's by a few levels, so a few flags land on other frames than in the CLI run |

**Stills.**

`tools/regress.py` runs the engine's own runner (`scripts/run.py`) and this pack on the same job and images, then compares them. The runner writes into a scratch folder, never to the job's own output. The check passes when fewer than 20 pixels differ by more than 8 levels and nothing outside the work box differs. With `--psd`, both sides also write the PSD, and their layers must match and flatten within 2 levels.

```bash
python -I tools/regress.py --psd                                    # the demo, in-process; needs a Python with torch
python3 tools/regress.py --server http://127.0.0.1:8188             # the pack on a live ComfyUI server
python -I tools/regress.py --job path/to/job.json [--scene ...] [--reference ...] [--psd]
```

In-process results on 2026-10-08:

| Run | Result |
|---|---|
| The demo, with `--psd` | identical pixels; the PSDs are byte-identical |
| 17 production jobs from one 4096 × 4096 frame: cards, boxes and a pouch, glossy and plastic-wrapped packs with relight and sheen, six cans with cylinder shading, a partly hidden pack fitted to its print's perspective, a user-supplied fixed render | all pass. 8 jobs identical, 8 within 1 level on 1 or 2 pixels, 1 with a 12 × 11 px patch up to 8 levels. The 3 PSDs checked have the engine's layers and flatten within 1 level |
| The same 17 jobs chained back to front through the pack alone, against the frame the engine delivered | within 1 level: 734 of 16.8 million pixels differ |
| Labels on bottles (engine 0.5.0): cylinder fits with and without the residual correction at 4096 and 1024 px, a flattened packshot under the scene's paper field, auto sharpen, and a text block with the ink matte | all pass, within 1 level on 0 to 10 pixels |
| Labels in their generated shape (engine 0.7.0): three 1024 px labels with `shape: generated`, one behind a wall | all pass, within 1 level on 0 to 3 pixels; the `shape` and `audit` reports match the engine's |
| Clean paper, body regions, top text placed by the tube (engine 0.8.0): four 4096 px jobs (two bottle labels, a tube's panels with a body band, an ink-matte text block) | all pass, within 1 level; the `fill`, `shape` and `audit` reports match the engine's |
| Engine 0.9.0 (Auto Draft, 2026-10-16): the demo and 12 production jobs, against `run.py` | all pass: 12 jobs within 1 level on 0 to 9,053 px, the shelf card within 8 levels on 1,634 px (the relight patch above); nothing outside the work boxes |
| Engine 0.10.0 (Auto Draft's label mode, 2026-10-16): the demo and 12 production jobs, against `run.py`; the 7 label jobs drafted with `label` on, with the quad, quad and mask, and mask alone | all pass: 12 jobs within 1 level on 0 to 9,053 px, the shelf card within 8 levels on 1,634 px; the 21 label drafts' jobs identical to the engine's and their images within 8 levels everywhere; the other 6 jobs' drafts score as at 0.9.0 |
| Engine 0.10.1 (outline tracing walks the boundary; a dropper bottle's pump head had made the traced outline zig-zag): the 7 label jobs with a quad and a mask and with a mask alone, the demo with a mask alone and with a traced outline | all 16 drafts' jobs identical to the engine's, images within 1 level; the render path doesn't trace outlines, so the stills fixture is byte-identical and the demo with `--psd` passes as before |
| Engine 0.10.2 (label mode redraws a wrapping side of the packshot outline on the limb, 2026-10-16): the 6 bottle-label jobs with a quad, a quad and a mask, and a mask alone | all 18 drafts' jobs identical to the engine's, images within 8 levels everywhere; placement as before or closer, except the 4096 px bottle label with a mask alone (5.4 to 8.3 px, its turn round the bottle) |
| Auto Draft against `run.py draft` (`--draft`): the same 13 jobs, each with a quad, a quad and a mask, and a mask alone | all 39 pass: identical draft jobs; images within 1 level, or both runs stop with the same FILL STOPPED (the shelf card with a mask) |

ComfyUI passes images between nodes in 32-bit floats, where the engine keeps 64-bit, so a value can land on the other side of a threshold. The 12 × 11 px patch is where relight's fine pass kept a slightly different region. Nothing outside the work boxes differed in any run.

### Scoring a draft

`--draft` scores Auto Draft against a hand-measured job:

```bash
python -I tools/regress.py --job path/to/job.json --draft [--jitter 3] [--outline hand|traced] [--rough-pairs] \
  [--mask PATH|hand [--no-quad]] [--no-occluders] [--label auto|off|cylinder|flat]
```

From the hand job it builds the rough input a person would give:
- the quad: where the hand fit puts the outline's bounding box corners, each corner `--jitter` px off;
- the outline on the packshot (`--outline traced` lets the draft trace it);
- the hand job's occluders, 2 px off;
- with `--rough-pairs`, its landmarks, `--jitter` px off;
- with `--mask hand`, a mask of its generated outline minus its occluders, or `--mask PATH` for a real one;
- for a label job (a cylinder hand job, or one in the generated label's shape), `label` on (`--label auto`, the default), and the quad is the generated label's 4 corners (the hand job's `qa/label_outline.json`, else its measured outline's sides extended), each `--jitter` px off.

The pack runs `workflows/product-label-refinement-draft.api.json` on that input, and the engine runs `run.py draft` on the same. Their draft jobs must be identical and their images must match as above, or the check fails. Then the draft is scored against the hand job: edge, placement, pixels against the hand job's image, and both audit verdicts (the table above). It prints a `row` line for that table and writes everything to `regress.json`.

`tools/offline_run.py` runs any API workflow in-process, with no server. It uses ComfyUI's own LoadImage and SaveImage conversions and validates inputs and link types the way ComfyUI does. `regress.py` uses it, and it takes the same arguments as `comfy_run.py`.

## Maintaining

Video: `imgvideo/` and `imgauto/` are the skill's too, pinned under `video_engine` in `imgpost.lock.json`:

```bash
python3 -P tools/sync_video_engine.py              # check; --update copies the skill's imgvideo/ and imgauto/ here
python -P tools/regress_video.py --comfy PATH/TO/ComfyUI   # Refine (Video) against the skill's video.py --refine, bit for bit
python3 -P tools/make_video_demo.py examples/video_demo    # redraw the video demo (byte-identical)
```

Stills:

`imgpost/` is a byte-identical copy of the product-label-refinement engine's library. The pack vendors it rather than importing it so the math stays identical and the pack stays self-contained on any server. `imgpost.lock.json` records the sha256 of every vendored file and of the runner version that `stages.py` ports.

```bash
python3 tools/sync_imgpost.py                    # check: local edits, engine changes, runner changes
python3 tools/sync_imgpost.py --update           # copy the engine's imgpost/ here and re-pin
python3 tools/sync_imgpost.py --mark-ported      # after porting a runner change into stages.py
python -I tools/build_workflows.py               # rebuild workflows/ after changing a node's inputs or outputs
python3 tools/make_demo.py                       # redraw examples/demo/ (byte-identical with the same libraries)
```

Never edit `imgpost/` here. To pick up an engine change:
1. Change the engine.
2. Run `--update`.
3. Port any runner diff into `stages.py`.
4. Run `tools/regress.py`.
5. Commit.

## Extending

- **New grade options.** Port the branch from the runner into `stages.grade`; the job's `grade` section passes it through unchanged. To add a switch on the node, add an optional combo whose first choice is `job`, after the existing ones, as `relight` was. UI workflows store widget values by position.
- **Measuring helpers.** `imgpost/autofit.py` holds the engine's measuring helpers: they snap a rough polyline to the colour edge below a pixel, refine a rough landmark by correlation with a trust verdict, trace a packshot's outline, fit an outline to a mask, and propose edge lines and the old silhouette from a rough placement. `imgpost/draft.py` puts them together, for the engine's `run.py draft` and for Auto Draft. Mask inputs already take scene-sized masks, so a mask from another node can replace Job Masks' outputs. Score any change to the draft with `tools/regress.py --draft` against hand-measured jobs before trusting it.

## Not covered yet

- Print straight on a bottle (no label edges, like the anchor frame): `label: cylinder` needs the label's top and bottom to be edges in both images. Without them, read landmarks.
- The turn of a label round its bottle isn't pinned by the edges (above): a 4096 px label can land 5 to 11 px round from where a hand fit puts it.
- Batch mode: one job across many frames.
- A GPU path. Everything runs the engine's numpy and scipy code on the CPU, about 4 to 15 s per product.
