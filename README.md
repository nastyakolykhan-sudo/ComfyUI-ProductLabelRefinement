# ComfyUI-ProductLabelRefinement

Fix a hallucinated product in an AI-generated still by compositing the real product from its packshot, inside ComfyUI. The nodes warp the packshot onto the generated product's perspective and carry over the scene's light and colour. That includes cast shadows, falloff, gloss and haze, and the round light on cans and bottles. They match the frame's softness and grain, put the product behind whatever stands in front of it, and fill in the background where the generated product was bigger. Every run ends with QA sheets, a report, and a layered PSD if you want to finish by hand.

It is deterministic compositing with no models. Every measurement lives in a job file (`job.json`): the work box, landmark points, edge lines, outlines, foreground objects, and the grade, finish and fill settings. The same job gives the same pixels every time. That lets an agent or a script drive it through ComfyUI's API, check the report and QA sheets, edit the job and run it again.

A job doesn't have to be measured by hand first. **Auto Draft** makes one inside ComfyUI from rough input: the scene, the packshot, where the product roughly sits (four corners or a box), and optionally a product mask from SAM 3. On a clean scene the draft is close to final; on a cluttered shelf it is a scaffold to finish by hand. [Starting from rough input](#starting-from-rough-input-auto-draft) has the scores.

**Print straight on a coloured body.** White print on a frosted bottle or a squeezed tube, or black print on a pink jar or a white bottle, has no paper to measure it against. The ink templates lay the packshot's print on the frame's own body, in the frame's own print colour, and bow it on a body seen from above or below; dark print is also snapped onto where the generator drew it. They make a keyframe from a frame, its SAM 3 matte and the packshot. See [Print straight on a body](#print-straight-on-a-body-the-ink-templates).

**Video too.** Product Label: Refine (Video) carries a corrected keyframe's label through a whole clip: track the product, fit its surface on every frame, re-project the true label with each frame's light, and change nothing outside the changed pixels. See [Video: the label through a clip](#video-the-label-through-a-clip).

**One workflow for every case.** `workflows/product-label-refinement-all.json` takes an image or a clip and runs only the branch the case needs: a still, a keyframe with print on the body (light or dark), a measured job, or a clip through SAM 3 and Refine (Video). It is the one to deploy; the other templates stay as they are. See [One workflow for every case](#one-workflow-for-every-case).

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

For print straight on a body, copy `examples/ink_demo/inkdemo_frame.png`, `inkdemo_packshot.png` and `inkdemo_mask.png` into `ComfyUI/input/` and load `workflows/product-label-refinement-ink-draft.json` (Auto Draft from the frame, the mask and the packshot) or `product-label-refinement-ink.json` (the measured `examples/ink_demo/job.json` in Load Job). The keyframe lands in `output/product-label-refinement/`. The ink demo is a made-up frosted bottle whose white print came out garbled ("SAMPEL STUIDO", "Ligth Pirnt on a Bdoy"), seen from slightly above; `tools/make_ink_demo.py` draws it.

For the one workflow, copy the ink demo's three files, load `workflows/product-label-refinement-all.json`, set **Product mask** to `inkdemo_mask.png` (or leave it at (none) with the SAM 3.1 checkpoint installed: the prompt is `bottle`) and queue it. It gives the ink-draft template's keyframe, as `output/product-label-refinement/refined_00001_.png`. Set **Source** to `labelvideo_demo_clip.mov`, **Product mask** to `labelvideo_demo_mask.mov`, **Corrected keyframe** to `labelvideo_demo_keyframe.png` and **Keyframe frame** to -1 for the video template's result.

The demo is a made-up product card. In the "generated" scene its text is garbled ("PRDOUCT", "sapmle lable"), its emblem and grid are wrong, its corners are square and its top is too tall. A mug stands in front of its lower right corner and casts a shadow across it. The packshot is the clean card. `tools/make_demo.py` draws all three files from code, so they show no real product or brand. `examples/demo/job.json` is the worked example of a job.

## Nodes

There is one node per step. The math is the product-label-refinement engine's (`imgpost/`, see [Maintaining](#maintaining)), split at the node boundaries in `stages.py`.

| Node | Inputs | Outputs | Step |
|---|---|---|---|
| **Load Job** | `job_json` (text) or `job_path` | `job` | reads and checks the job |
| **Auto Draft** | scene, reference, `quad`, `model`, `light`, `suggest`, `filename_prefix`, `product_mask`?, `reference_mask`?, `pairs`, `occluders`, `outline`, `mask_skip`, `label`, `ink`, `print_box`, `bow` | `job`, `job_json`, `draft_points`, `report` | a first job from rough input, in place of Load Job; saves it as JSON ([Starting from rough input](#starting-from-rough-input-auto-draft)); `ink`, `print_box` and `bow` for [print straight on a body](#print-straight-on-a-body-the-ink-templates) |
| **Job Masks** | job, scene | `old_silhouette`, `occluder_mask` | rasterises the generated outline and the foreground objects |
| **Fit Warp** | job | `homography`, `report` | homography, affine or cylinder fit (labels on bottles), with the optional residual edge correction; shows the residual table |
| **Warp Reference** | job, homography, scene, reference, `reference_mask`? | `warped`, `product_mask`, `align_overlay`, `coords` | supersampled warp in linear light; warns when the outline takes in the packshot's backdrop (a light rim) or a bottle's measured edge sits past its drop shadow; with `matte` ink on the body (light or dark), `coords` also carry the packshot's print as coverage (dark print registered onto the frame's own and sharpened to it) |
| **Grade To Scene** | job, scene, warped, masks, `colour`, `shading`, `reference`?, `coords`?, `relight`? | `graded`, `grade_preview`, `report`, `relight_map`, `ink` | light falloff, colour, the scene's paper field, cylinder shading, relight; `ink` feeds Composite Behind for jobs with `matte: ink` |
| **Match Finish** | job, scene, graded, masks | `product`, `report` | highlight roll-off, blur and grain match |
| **Fill Leftovers** | job, scene, masks, `coords`? | `background`, `fill_mask`, `report`, `ink` | harmonic fill; stops with FILL STOPPED. With `shape: generated` it needs Warp Reference's `coords`. With `matte` ink on the body it removes the frame's garbled print from its body (old_silhouette: the product's mask) and `ink` carries the print's colour to Composite Behind |
| **Composite Behind** | job, scene, background, product, product_mask, occluder_mask, `ink`?, `coords`? | `image`, `matte` | composite and paste into the scene; with `matte: ink`, only the print goes over the scene's own surface (ink on the body: Fill Leftovers' `ink`, and `product` isn't read); with `shape: generated`, the generated product's own outline is the matte (needs `coords`) |
| **QA Sheet** (audit) | ..., `old_silhouette`?, `occluder_mask`?, `coords`?, `ink`? | ..., `audit` | with the masks connected the run audits itself: a missed occluder, a rim of the generated product, a thick fill, print pressed into a bottle's edge, edges off, lines across a round product curving unlike the generated product's (`bow`, a fit with no curve of its own); FAIL blocks delivery. With Fill Leftovers' `ink` (ink on the body) it also flags pixels changed away from the print (`ink_edges`) and a packshot too small for the frame (`packshot`); for dark print, a top line turned against the generated print's (`ink_level`), lines curving unlike the generated print's (`ink_shape`) and cuts along an occluder's edge or a stray line of new ink (`ink_rim`) |
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

## Print straight on a body (the ink templates)

For print straight on a body, with no paper around it. Light print sits on a darker body (white print on a frosted bottle, a pink tube); dark print sits on a lighter one (black print on a pink jar or a white bottle, with its coloured bars and badges). The engine's paper-based ink matte can't lift either: it measures coverage down from the paper's white. A whole-band composite shows seams and the packshot body's colour, and a label fit needs a label's edges. Here the body, its lid and its reflections stay the frame's and only print changes (the engine's `matte: {"type": "ink", "polarity": "light" | "dark"}`, `imgpost/ink.py`):

1. **Warp Reference** reads the packshot's print as coverage against its own body light (above it for light print, below it for dark), inside the job's `reference_outline` (the band of body it maps) and its `print_box`, and warps it. Dark print is then registered onto the frame's own print: scale, shift, affine and bow, so it sits where the generator drew it. It is also sharpened to the frame's print.
2. **Fill Leftovers** finds the frame's garbled print (brighter or darker than its body) where the true print goes, inset from the product's matte, and fills it from the body around it with the frame's grain. Dark print's garbled strokes are taken whole, in blocks. `background` is the frame without its print.
3. **Composite Behind** lays the true print on in the frame's own print colour (the body times k per channel, k from the garbled print's most-covered pixels), inside the matte and nowhere else. A second dark ink (a yellow bar, a badge's disc) keeps its packshot colour, relit by the frame's body.

Grade To Scene and Match Finish aren't used. The product's mask (SAM 3) goes into Fill Leftovers' and QA Sheet's `old_silhouette`: it is the matte the print stays inside.

Two templates:
- `workflows/product-label-refinement-ink-draft.json`: **Auto Draft** from the frame, its mask and the packshot. Its inputs:
  - `ink`: light or dark.
  - `print_box`: the print's box on the packshot, from its first line to its last with a few px to spare, inside the frosted rims, the seams and the lid's edge. Without it the packshot's own highlights and seams go on as print.
  - `bow`.
- `workflows/product-label-refinement-ink.json`: the same graph with a measured job in **Load Job** (the ink demo's).

**The bow.** A round body seen from above or below bows every line of print across it, like its seams; a frontal packshot warped flat reads as a printed card. `bow` (the job's `align.bow`) is a profile along the product's axis, t in px from its mask's centre (+ down the image): seen from above, lower parts bow more, and a squeezed tube is flat at its crimp. Write it as `t:b` (given: `-150:0 146:10`), or as seams for Auto Draft to measure across the body: `t` (a level edge), `t/w` (a seam on a narrower ring of half-width w, a collar), `t/matte` (the mask's own end, a base). Or `print`: Auto Draft reads the bow off the generated product's own long lines (its rules, a band's edges, full-width lines of text), pairs each with the same line of the packshot as the fit warps it, and takes the difference, smoothed so the bow over the body's width grows down the body (engine 0.13.1). It's for a packshot warped whole: a squeeze tube seen from above, warped flat by an affine fit, read as a flat card ("the label doesn't follow the curved tube's shape"); with `print` its rules bow +6 and +9.5 px and its band and cap seam sag 7 and 16 px, against the frame's 6.9, 9.6, 9 and 17. The mask's flat end (a tube's crimp) holds the bow at 0 there; a base the mask cut flat along a reflection is left out. Empty for a product seen from level.

**How close it gets.** On a real 1080 × 1920 job (a frosted serum bottle and a squeezed lip tube with white print on their bodies, and both printed on a box seen from above), the four keyframes the user approved, made by a job-level script, came out of the engine within 1 level on 1 pixel and out of this pack within 1 level on 1 pixel (the box's two prints with their measured bows included). From a frame, its mask and the packshot alone (the ink-draft template), the tube's print landed within 1 px of the hand-fitted keyframe (median, 2.1 px at worst), and the bottle's 4.9 px off: the hand fit followed the generated print, which the generator drew lower and turned on the body, and Auto Draft fits the packshot to the mask's silhouette. Rough pairs on garbled print are too weak to correct it. Where the print must sit exactly on the generated print, measure its anchors and load the job. Bows measured at the seams came within 1.5 to 2.5 px of the hand values: check the outer print lines at 6x.

**Dark print** (`ink` dark; engine imgpost 0.12.0, each line's bow 0.13.0). A fit to the mask's silhouette can't place print on a round body seen from above. On a pink jar 12 deg from above, its print came out 30 px high and 15 % tall: the mask takes in the lid's top, and the jar's front is foreshortened. So dark print is registered onto the frame's own print: an x and y scale and shift by correlation, then a refinement by Gauss-Newton. The generator draws the layout roughly right. The print stretches only along and across the body's axis, after one fixed turn that sets its rows square to it, so its lines stay parallel to the body's own level edges. A free affine with a fitted bow ran the jar's top rule 1.8 deg against its lid seam; this keeps 0.35 deg, as the generated print's own.

`bow` is optional with `ink` dark. Without it, Auto Draft measures the body's bow from the mask's two ends, and the run keeps that bow only where the print registers better with it: on the jar it did, and on two bottles whose print the generator drew flat it didn't. Then the run measures each line of the frame's own print across the body and takes the bow from those (`matte.bow_from_print`), kept again only where the print registers better. The jar's lines bow 73-88 px, more than its lid's rim (63) and base (51): one bow from the mask's ends flattened the lower lines ("doesn't respect the shape of the product"). A packshot photographed front-on has its own lines curving 5-14 px the other way, so the bow is corrected once by what the warped packshot's lines miss. On two bottles whose print the generator drew flat it changed nothing. Warp Reference prints the move (`register`), and Fill Leftovers' report has it under `matte.register`.

Three more steps keep it clean:
- The packshot's body light is the local mean of its non-print pixels, with the sharpening halos left out (a closing rode on them and washed the box in 13 % grey).
- Seams and label edges the box takes in at its edge are cleared.
- The body's own straight lines (a label's edge, a seam, a lid's rim) run on past the print box and are cleared where they enter it.
- The cover is sharpened to the frame's print (a 1,024 px packshot's 0.95 px edge blur against the generator's 0.62), and a second ink keeps its own flat colour (a white bottle's gold bar, its foil gradient gone).
- Objects in front (Auto Draft's `occluders`) stay as they are: garbled print isn't looked for within 2 px of one, and the fill never takes its colour. Trace a SAM 3 click's mask for its outline, and add a polygon over what it leaves out (a fish's thin fins).

QA Sheet adds a `packshot` WARN when the frame shows the packshot's print over 1.25x its size: an 800 px packshot shown 1.5x keeps its small print soft, so ask for a larger one. Three more checks come from the user's reviews of the dark-print keyframes, which the audit had called clean:
- `ink_level`: the print's top line turned more than 1 deg against the generated print's. It fires on the first jar (+1.58 deg, "a bit skewed") and not on the fixed one (+0.35).
- `ink_shape`: each line's bow against the generated print's line at the same height. A line is off past 4 px or 10 % of its bow; it warns when half the lines or more are off the same way. It fires on the jar's second delivery (every line about 36 px flatter, "doesn't respect the shape of the product") and its first (16 px more bowed), not on the fixed one or the two bottles.
- `ink_rim`: dark structure removed at an occluder's edge, the body changed along it, or a straight line of new ink near the body's edge. It fires on the first toner (its "silver line" and the fish's eye) and not on the fixed one.

On the real frames tested from a SAM 3 mask, the pack's keyframes are bit-identical to `run.py draft` plus `run.py`, with the same `matte` report:
- the jar (1080 × 1350): clean audit, lid untouched, its top rule parallel to the lid seam;
- a white pump bottle with a gold bar: the text corrected, a `packshot` WARN;
- a white toner bottle with a fish in front of its wordmark (the fish as occluders): corrected, the fish untouched, a `packshot` WARN.

## Video: the label through a clip

Correct one frame of a clip, by hand or with an image model given the real product as reference. **Product Label: Refine (Video)** carries that label through the shot:
1. it tracks the product (AllTracker);
2. fits its surface on every frame;
3. re-projects the true label with each frame's light, softness and grain;
4. composites it inside the product's mask.

The text is right on every frame, the timing is the source's, and the pixels outside the changed area come back exactly as they went in. It's the product-label-refinement skill's video engine (`imgvideo` 0.5.0), and the skill's own `video.py --refine` gives the same pixels (see Regression).

### Swordfish checklist

1. **ComfyUI 0.19.3 or newer.**
   - The video nodes themselves run on 0.18.2.
   - SAM 3 (`SAM3_Detect`, `SAM3_VideoTrack`) and the SAM 3 helper below need 0.19.3, as does a label-fix workflow that segments with SAM 3.
2. **AllTracker's weights:** `ComfyUI/models/trackers/alltracker.pth`.
   - That's the `models/trackers` folder, or a `trackers` folder beside the standard model folders wherever they're mounted from: a volume that `extra_model_paths.yaml` maps for `checkpoints`, `sams` and the rest is searched too, so `trackers/alltracker.pth` at that volume's top level is found without editing the yaml. On Swordfish, add it from the Models page with the HuggingFace button (repo `aharley/alltracker`, file `alltracker.pth`, custom path `trackers`), then restart ComfyUI and reload the page.
   - Get them from `https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth`: 66,005,722 bytes, sha256 `ffd9ebcfb6d206d594b646999a150540f92c049cf9b2bf940facf7123f62aa1d`, MIT.
   - The node lists what's in that folder under `tracker` and never downloads anything.
3. **SAM 3.1:** `ComfyUI/models/checkpoints/sam3.1_multiplex_fp16.safetensors`, the checkpoint the label-fix workflow already loads. Its licence is Meta's SAM License.
   - The one workflow (`product-label-refinement-all`) also needs ComfyUI-MaskAspectCrop (Crop & Rotate) installed.
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
| **Product Label: Refine (Video)** | `frames` (IMAGE batch), `mask` (MASK batch), `keyframes` (IMAGE, one per shot), `tracker`; optional `keyframe_index`, `keyframe_frames`, `keyframe_bbox`, `product` (context only), `settings` (JSON), `fps`, `processes`, `model` (tube, or flat: experimental), `compositing` (settings, print or label: `render.alpha`), `soft_cap` (px, -1 = the engine's default: `render.soft_cap`) | `frames` (same count, size and order; outside the changed pixels the input itself), `matte` (the mask plus patched frames), `changed`, `report` (JSON), `qa_image` |
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

**Fixes the report asks for** (engine imgvideo 0.4.0). Run again with the same frames, mask and keyframes; only the render runs again. Both fixes fit in one `settings` object:
- **`{"render": {"body_snap": true}}`**: the label is placed by the product's body instead of by the generated print it tracks. Where the generator draws the print longer, lower or slanted on the body than on the keyframe's frame, the tracked label follows it. With this setting, on every frame the label keeps the keyframe frame's share of the body and its print keeps its angle to the body's centreline. The share is measured from the label to the collar seam or cap above and to the body's end below.
  - **When:** the report's `review.checks` has `body`, or `geometry` shows the label's top or bottom more than 1 px, or its roll more than 0.5°, off the keyframe's.
  - **Edges:** found automatically. On a squeezed tube, set them: `{"body_snap": {"top": "matte", "bottom": "dark"}}` (the crimp's top and the cap seam).
  - **Other keys:** `reach`, `stretch`, `roll_max`, `ref_frame`, `frames` (see the engine's `fit.BODY_SNAP`). The per-frame corrections are in each keyframe's `body_snap` in the report.
- **`{"render": {"sparkles": [[98, 119]], "spark_glow": true}}`**: light from the source (glitter, sparkles) is re-added in front of the label, only on these frames. The list takes clip frames, or half-open runs `[a, b]`.
  - **When:** `review.checks` has `glitter` (specks in front of the label that the output lost).
  - `review.sparkles_frames` already holds the runs, padded, ready to paste.

**Print-only compositing** (engine imgvideo 0.4.2, 0.4.3): `compositing: print` on the node, or `{"render": {"alpha": "print"}}`. Only the print changes, carried onto the source's own body, so no patch outline can show around the label. Without it, Refine picks the mode from the print (the measured defaults since 2026-10-17): `print` for light print, `label` (the whole relit label band over the source) for dark print. Where the relight misses (frosted glass, haze, a tube's limb), the band's outline shows as a lighter or darker square on the plain body.
- **How:** the new print is the keyframe's own ink, carried through each frame's mapping. The garbled print within 14 px of it goes. The target is the source's body with its light strokes clamped, under a soft matte of the render's ink in its own colour. On the first real clip, the body inside the patch stayed within 1 level of the source (median), and nothing changed outside the matte.
- **What it turns on with it** unless `settings` says otherwise:
  - `soft_cap` 0.3: the label is never blurred more than 0.3 px to match a soft source frame (a restored label stays sharp and legible);
  - `smear: false`: no motion smear. A false smear happens: 26 px on a sharp tube. `"smear": [frames]` keeps the measured smear on frames where the source clearly shows it;
  - `feather.v_px` 4: the label's end fade. The 24 px default dimmed print that starts inside it;
  - the sparkle re-add off, always: it ghosts the garbled print. The source's glitter stays anyway, everywhere but on the letters.
- **Presence** (0.4.3, on by default with `print`): a thin true typeface reads fainter than the bold garbled print it replaces, at the same brightness. The print's presence (how much it lifts the body across its footprint) is brought up to the source's on every frame, smoothed over a few frames, and never lowered. `print_alpha.presence_target` 1.25 puts thin copy a quarter above the source's. `presence: "off"` turns it off.
- **Motion blur on fast moves** (0.4.3, on by default with `print`): the print blurs along the label's own motion (`shutter` 0.5 of the frame's travel), at most 3 px, from 8 px/frame up, and only as soft as the source's own print is on that frame. Generated video jumps up to 85 px between frames that are themselves crisp. `shutter: 0` turns it off.
- **Glue** (0.4.3, `glue` on the node, or `{"render": {"glue": true}}`): the label moves frame to frame with the product's own pixels and lands where the fit places it over a few frames, so it doesn't swim on frames where the product barely moves. On the first real clip, swim on near-still frames went from 1.5-2.4 px to about 0.3 px, the source's own level. The keyframe's frame stays exact. Keys: `sigma` (4 frames), `edge` (`line`, or `mean` where the run's first frames swing), `rms_max`, `dev_max`. The report's `glue` lists the motion fits that fell back.
- **Knobs:** `render.print_alpha`: `ink` (`keyframe`, or `render`: the render's own print), `near`, `grow`, `sigma`, `carry` (false: paste the render's pixels), `ink_levels`, `ink_p90`, `ink_inset` (see the engine's `printalpha.DEFAULTS`).
- **The report:** each keyframe's `print_alpha` gives the ink px, the frames, the share of the render's change that was kept (median), and `lost_print_frames`. Those are frames where print the render had is missing from the output by over 30 px. Read them against `lost_print_share_median`: bright white print loses some contrast on every frame once it sits on the brighter source body (about half the ink on a frosted bottle with white print, with every letter there). A frame far above the median is the one to look at.
- **Limits:** light print on a body only (white ink on a bottle or a tube). A dark-print job raises an error. With `print`, the review's glitter check no longer suggests `render.sparkles`.

**On by default for round products** (engine imgvideo 0.5.0; `model: tube`). A first real clip's review notes, now defaults with their checks, so a run needs no settings beyond bands, shots and words:
- **Glue** (`render.glue`): on. Turn it off with `{"render": {"glue": false}}`. `edge` can now be set per end of the held run, `["line", "mean"]` (start, end): `mean` steadied one product's first frames and another's last ones, and unsteadied that other's first frames: the review's `ends` check says which end to flip.
- **Level** (`render.level`): the label's lines run level with the product's base, judged with the product straightened. The generator can draw its print slanted on the body for a run of frames; the tracked label inherits the slant as a shear, which the body snap (it turns the label's verticals) doesn't remove. Per frame, the label's horizontal across the copy block is read against the body's own axis (found in two passes, robust to a tilted product). The keyframe frame's level is the target, smoothed robustly over 4 frames, and the label is sheared along the axis so its lines turn and its verticals stay.
  - `offset: "base"` (the default) also levels the keyframe's own lines with its base, measured on the keyframe image. Give degrees instead, or `0` to keep the keyframe's level.
  - **A squat product seen from above** (0.9.0; a jar about as wide as tall): its axis is the one nearest the image vertical, not its widest extent (which ran across the jar, so the level read the lines against one of its sides); lines that bow more than 6 px across the copy block are read where they run level, at the body's axis; and the body snap searches for the product's sides until both fall inside its search (a jar's body is wider than the search used to reach, which turned the label up to 0.9° on frames where it was 0.5° off). On a jar's push-in the lines now stay within 0.2° of the frame's own print. No setting: tall products read as before.
  - **It runs only where the product has a steady base:** the matte's bottom edge, straightened, must scatter under `base_mad_max` 0.5° frame to frame. A squeeze tube's cap end scattered 1.4° and is skipped (the log and the report say so); `"gate": false` levels anyway. Other keys: `sigma`, `window`, `at` and `span` (where the copy block is, as shares of the label), `offset_max`, `min_deg` (the engine's `level.LEVEL`).
- **What came from a default:** the render's log has a `defaults` line, and each keyframe's `defaults` in the report lists every stage the measured job switched on that the `settings` don't name, with who set it.

**Dark print, whole-band compositing** (engine imgvideo 0.6.0): away from the keyframe the generator's lines drift against the tracked label, so wherever the label let the source through over print, the frame's own line showed a few px off the label's (a jar's second-last line doubled under its end fade; a tube's generated logo letter left a shadow through the relight). On by default with `label` compositing:
- **The band carries every line the keyframe corrected:** the measured band is stretched over lines of print the keyframe put in or took out past it, plus up to 32 px of room that stops at the keyframe's next body step (a base, a cap). The tube is still fitted on the measured band (`anchor.fit_band`). `{"anchor": {"over_print": false}}` keeps the measured band; the report's `band_info.over_print` says what moved.
- **`feather.print_gap` 2:** the label's end fade stops 2 px short of its own print (shorter than `v_px` where print runs close to an end).
- **`shade_print`:** the frame's light is read on bare surface only, both prints left out, away from the limbs: a bold logo drawn bigger and elsewhere no longer travels with the label as a shadow.
- **`clear_glass`** (0.7.0): where the stretched band runs over clear glass (a jar's last lines on its glass base: cells of the still label whose hue isn't the body's), the frame keeps its own glass. Its print near the label's letters is filled, and only the label's print goes on. Before, the keyframe's glass came along as soft smudges over the frame's sharp reflections. `{"render": {"clear_glass": false}}` turns it off.
- **`limb_snap.front: "track"`** (0.7.0): the limb snap moves the label's sides onto the matte but keeps the print facing the camera on its tracks. Before, the two sides' common offset moved the whole print: 6-7 px off the frame's own on a jar seen from above. `"front": "mean"` is the old behaviour.
- **`shade_local`** (0.8.0): the frame's light is read from the bare surface round each patch of the label. Before, patches the print covered took the frame's overall light, which near lines of text changed as the generator redrew them: a tube's body boiled in soft patches round its text (the light blinking 3-5 levels from frame to frame). `{"render": {"shade_local": false}}` is the old read.
- **`steady_light`** (0.10.0): that light is held steady over a few frames. A read pass first takes each frame's light on the label's own cells; each cell is then smoothed over 5 frames against its frame's overall level, which is added back, so a scene's real flicker carries on. Before, the light was read frame by frame, and where the generator's own lettering covered the bare surface, its faint strokes, redrawn every frame, made blobs blink round the label's text (a tube: 2+ levels on 0.33 % of the label a frame, now none). The read pass doubles the render's time. `{"render": {"steady_light": false}}` is the frame-by-frame light.
- **`{"render": {"body_snap": {"turn": true}}}`** (0.8.0, a setting, off by default): the body snap also holds the label's turn on the body, at the keyframe frame's place (at most 8 % of the label's width). The generated print can slide round the product as the camera moves, and the tracks follow it: a jar's label slid 13 px across the jar over a push-in, now within ±1.3 px. Only for a product that doesn't turn on its axis: one that does turns its print with it, and its matte looks the same (the review's `slide` flags both; look first). While it holds the label, the frame's own print near the letters where the tracking puts them is filled wherever the frame shows (the label's fades, clear glass the band keeps), so the generator's lines can't show beside the label's. Since 0.9.0 that fill stays inside the label's area: past it, a glass base's rim below the last line was filled flat.

**The look checks** (0.5.0, in `review`): the same review notes as pass/fail flags, each with the setting to try next in `review.actions` (`{"check", "frames", "why", "try"}`, ready to merge into `settings`):
- `swim`: the label slides against the product on near-still frames (the body moving 3 px or less), 0.35 px more than the generated print did on two or more pairs close together. Try: glue.
- `ends`: the same on the first or last 4 frames of a held run. Try: the other glue `edge` at that end.
- `level`: the label's lines more than 1° off the base (with `level` on) or off the keyframe's level, held for a quarter of a second, where the base is steady. Try: `level`.
- `lines` (0.9.0): the label's lines turned against the frame's own print (the same read on both, off their difference on the keyframe's frame), over 0.3° and 3 px end to end across the copy block, held for a quarter of a second, where the base is steady and the frame's print runs level with it as on the keyframe's frame (where the generator slants its print, the label follows the body and `level` judges it). Try: `level` when it's off; otherwise look.
- `sharp`: the print under 0.7 of the keyframe frame's sharpness and 0.1 softer than the source's own print there. Try: a lower `soft_cap`, no motion blur.
- `tilt`: the print's side edge turned over 1° or shifted over 4 px against the product's limb, against the keyframe's frame, on crisp frames: a gross check (the `body` check is the fine one). Try: `body_snap`.
- `ghost` (the sparkle re-add on a list of frames, with `label` compositing): the re-add brings over 60 px of the old print back on a frame. Try: `print` compositing (the re-add stays off there).
- `spots` (light print): dark marks (4-60 px, 12 levels under the source's body without its print) on the label's body, 4 px clear of the letters: look. On dark print the letters themselves are dark marks, so it isn't judged there (0.6.0).
- `double` (0.6.0; 0.7.0: the frame's own glass the render keeps is left out, as in `ghost`, except, since 0.8.0, the frame's print where the tracking puts the label's letters when the label is held elsewhere): the source's own print beside the label's: print in the output that the source shows there too, off the label's own strokes and explained by the label's print moved a few px, 150 px or more as a 5-frame median. Since 0.6.1 only ink counts (px near the source's print at least a quarter as deep as the label's own): a round body's soft limb band beside the letters no longer reads as doubled print. Try (at the label's ends): `feather.print_gap`, then more `anchor.over_print.room`; at the sides: look.
`review.looks` has the numbers per shot (swim against the source's, level, sharpness, tilt). `{"review": {"looks": false}}` skips them.

**What the report says about the label against the body:** `geometry` has one entry per refined frame:
- the gaps to the edge above and to the body's end below (px);
- how far the label's top and bottom sit off the keyframe's proportion (px; `*_med` are the 5-frame medians the `body` check reads);
- the print's roll against the body's centreline (deg);
- its share of the body against the keyframe's (`stretch`);
- its turn round the body off the keyframe frame's place (`turn_px`, label px at its middle; 0.8.0).

`review.checks` now covers edges, step, ghost, shear, bend, body, slide, glitter, shimmer, vanish and switch, plus the look checks above (swim, ends, level, lines, sharp, tilt, spots, double):
- `slide` (0.8.0): the label turned round the product off its place on the body, over 4 px as a 5-frame median for a quarter of a second. Look first: if the product doesn't turn on its axis there, `body_snap.turn` holds the label.
- `step` (0.8.0): the product's step (a band's edge) shown twice at the label's end: the output steps twice the same way 3-16 px apart where the source steps once, across 15 % of the end, 3 frames in a row. Since 0.8.0 a body step inside the label's last rows (a keyframe that curves the band's edge with the body) isn't taken for print, so the label's end fade runs its full length over it.
- `shimmer` (0.8.0, recalibrated 0.10.0): the light on the label's body blinking from frame to frame (the output read through the mapping that rendered it) more than the frame's own light: the source's at the same points where the source is bare, else the frame's bare-source p90 (where the source's own print sits, redrawn every frame, it blinks far more than any light). Over 0.2 % of the body 2 or more levels above, 3 frames in a row; the frame's own glass the render keeps is left out. The flag names the fix (`steady_light`).

**`body` and `glitter` are calibrated on one product so far:** a frosted serum bottle with white print. On older approved test clips they still flag some frames: a close-up bottle whose gap to the collar seam swings as it bounces, and a shimmering metallic tube. On a new product type, look at the flagged frames before turning the fix on. On dark print (0.7.0) a glitter speck must also be clear of the frame's own dark print: the gaps in a garbled seal the generator morphs from frame to frame were read as lost glitter.

**Product types.**
- **`model: tube`** (the default): anything round, such as bottles, jars, tubes and lipsticks. Every delivered example so far was a tube.
- **`model: flat`** (experimental: it runs end to end but hasn't been tried on a real clip): one flat face. For a box, run one Refine node per visible face, each with that face's mask and its own keyframe, chaining `frames` out of one into `frames` of the next.
- **Small or metallic print** (lipsticks): track on the print (`{"track": {"on": "print"}}`), and correct the keyframe on a zoomed crop (`keyframe_bbox`).

**Re-runs** on the same frames, mask and keyframes keep the tracks, fit and still label from the last run (in ComfyUI's temp folder), so a render-only change re-renders without re-tracking.

**Licences:**
- AllTracker: code vendored in `imgvideo/alltracker` (MIT, LICENSE beside it); its weights are MIT too.
- SAM 3: Meta's SAM License; check its use restrictions for your case.
- No CoTracker: non-commercial, and refused here.

## One workflow for every case

`workflows/product-label-refinement-all.json` (and `.api.json` for the API) is one graph for every case the separate templates cover. The source is an image or a clip, and switches run only the branch the case needs:

| The case | What runs |
|---|---|
| An image, a paper label or print on paper (`ink` off) | Auto Draft (or the given job), then the graph of `product-label-refinement-draft` (Grade To Scene, Match Finish) |
| An image, print straight on the body (`ink` light or dark) | Auto Draft (or the given job), then the graph of `product-label-refinement-ink-draft` (no grade; the product mask is the matte the print stays inside; dark print snapped onto the generated print) |
| A measured job | Load Job instead of Auto Draft (Auto Draft doesn't run) |
| A clip | SAM 3 on the keyframe frame, then through its shot both ways; the stills branch above on the keyframe frame; then Refine (Video) on that shot with that keyframe, spliced back into the clip |
| An edit (several shots, several products) | the Shot node finds the hard cuts round the keyframe frame, so SAM 3 and Refine work on that shot only, and the output is the whole edit with that shot refined. One run per product shot: the next product runs on the last run's output (`refined_#####.mov [output]`, no download) |
| A clip and a corrected keyframe made elsewhere | SAM 3 and Refine (Video); the stills branch doesn't run |
| A product mask given (image or clip) | SAM 3 doesn't run |
| `Fix this product` false (the product looks fine on the triage sheet) | only SAM 3 on the keyframe frame and Label Triage's sheet: seconds |

**Triage first.** Every run also draws Label Triage's sheet: the product's print on the frame (an image, or the clip's keyframe frame) next to the packshot's at the same size, both straightened, in colour (the engine's `imgauto/triage.py`). Run once with **Fix this product** off for each product in the input: the sheet and `triage.json` come back in seconds. Turn it on only for the products whose print is wrong; a product that looks fine never costs a draft, a track or a render. OCR on the sheet is an optional hint (`ocr`): at frame scale it can't tell a right print from a garbled one, so the call is the agent's (or yours).

An image never reaches the video nodes, a job with print on the body never runs Grade To Scene or Match Finish, and nothing in a skipped branch runs or fails.

**What it needs** (on top of the pack): ComfyUI 0.19.3 or newer (core SAM 3 nodes; tested on 0.41.0 with frontend 1.55.17), [ComfyUI-MaskAspectCrop](https://github.com/nastyakolykhan-sudo/ComfyUI-MaskAspectCrop) for Crop & Rotate, the SAM 3.1 checkpoint `models/checkpoints/sam3.1_multiplex_fp16.safetensors`, and AllTracker's weights for clips (see the [Swordfish checklist](#swordfish-checklist)).

**Same pixels as the separate templates.** Run with the same inputs, each case gives what its template gives, 0 px apart, with the same reports (`tools/regress_all.py`, see [Regression](#regression)):
- the ink demo: Auto Draft (= `product-label-refinement-ink-draft`) and its job (= `product-label-refinement-ink`);
- the stills demo: Auto Draft from its rough input (= `product-label-refinement-draft`) and its job (= `product-label-refinement`), the PSD included;
- the video demo with its keyframe (= `product-label-refinement-video`): the lossless master frame for frame;
- the video demo end to end, with a packshot of its tube: the keyframe the stills branch makes on frame 20 is the draft template's on that frame, and the clip is the video template's given that keyframe.

In these checks the product mask was given, so SAM 3 itself didn't run; its part of the graph is the label-fix workflow's (SAM3 Detect on the keyframe frame, then SAM 3 Track From Keyframe).

**Straightening** resamples what it changes: the crop is turned out of the frame and the changed pixels back into it (bicubic, as Crop & Rotate does), with a 2 px rim round them. On the ink demo turned upright by 3°, the print landed within 3 levels of the unturned run (median; 7 at the 95th percentile), and 39 px of the bottle's rim outside its mask changed. Leave it off unless the stills fix needs the label upright.

### Its inputs

The ids are the API file's (`workflows/product-label-refinement-all.api.json`), and the UI file uses the same ones.

| Node | Input | What to give |
|---|---|---|
| 1 Load Source | `source` | the image or clip to fix, by its name in `input/` (with a subfolder: `plr/clip.mov`). A video runs the video branch |
| 2 Packshot (Load Image) | `image` | the real product. A transparent PNG's alpha is its outline |
| 3 Product mask (Load Optional File) | `file` | `(none)`: SAM 3 finds it. Or a mask file, white = product, the source's size; a mask video for a clip, one frame per frame |
| 4 Corrected keyframe (Load Optional File) | `file` | clips: `(none)`: the stills branch makes it from the keyframe frame. Or a frame of the clip fixed elsewhere (by hand, an image model), the clip's size |
| 5 Job JSON (String, multiline) | `value` | empty: Auto Draft drafts the job. Or a measured job.json's text: Load Job runs it, and Auto Draft doesn't run |
| 6 Keyframe frame (Int) | `value` | clips: the frame the stills branch fixes (from 0, in the whole clip); its shot is the one refined. `-1` with a given keyframe: Refine finds its frame, in the whole clip (or between the Shot's `start` and `end`). An image ignores it |
| 83 Packshot outline for the fix | `use` | `alpha only` (default): a packshot without alpha is traced on its backdrop by the engine. `alpha, else SAM 3`: SAM 3's mask of the packshot (the prompt of node 11), for a product the backdrop hides (a white jar on white). Triage (82) always uses SAM 3 for a packshot without alpha |
| 19 Shot | `start`, `end` | `-1` (default): found, the hard cuts round the keyframe frame (a one-shot clip is all one shot). Or the shot's first and last frame in the clip |
| 7 Video Settings | `settings` | Refine (Video)'s settings JSON (below); empty: everything measured |
| 7 Video Settings | `product_copy` | the label's copy, as printed: it becomes the settings' `words` (OCR in the QA, text boxes) unless the settings give words, and Label Triage's OCR hint |
| 70 Label Triage | `ocr`, `glyph` | off: optional hints (tesseract; a letter-scale match), reported only |
| 9 Fix this product (Boolean) | `value` | `true` (default): fix it. `false`: the triage sheet says it looks fine; only SAM 3 on the keyframe frame and the sheet run |
| 84 Render the clip (Boolean) | `value` | clips: `true` (default): the keyframe, then SAM 3's track and Refine through the shot. `false`: the keyframe only (seconds), to check it before the clip; queue again with `true` and ComfyUI reuses the keyframe |
| 8 Auto Draft fits to the product mask (Boolean) | `value` | `true`: Auto Draft fits the outline to the mask (print on the body needs it). `false`: a paper draft from `quad` and `pairs` only, as the draft template |
| 86 SAM 3 points | `points` | empty: the text prompt finds the product. `x,y` (keyframe frame px): SAM 3 segments the product clicked and the text prompt is left out (it would add its own detections). For a group shot where the prompt picks the wrong product (3 of 9 on the test edits). One click on the body: two clicks or a `-x,y` shrank or emptied the mask on real frames. Check `triage/mask` |
| 11 SAM 3 prompt (CLIP Text Encode) | `text` | what the product is, e.g. `pump bottle`. Comma-separated prompts are joined into one mask; `name:N` takes N objects of a kind (1 by default) |
| 13 SAM 3 on the keyframe frame | `threshold`, `refine_iterations` | 0.33 and 2, as the label-fix workflow |
| 14 SAM 3 through the shot | `detection_threshold`, `max_objects`, `detect_interval` | 0.33, 1, 1 |
| 21 Straighten (Crop & Rotate) | `center_x`, `center_y`, `crop_width`, `crop_height`, `rotation` | a frame round the product and the turn that puts its label upright (degrees, + turns the image clockwise behind the frame). All 0: off, the whole frame, and nothing is resampled. When on, the stills branch works on the crop (Auto Draft's `quad`, `pairs`, `occluders` and the job are in the crop's px) and only the pixels it changed go back into the frame |
| 24 Auto Draft | `ink` | the print type: `off` for a paper label or print on paper or a flat panel; `light` or `dark` for print straight on a round body with no paper, lighter or darker than it (white print on a frosted bottle; black print on a jar or a white bottle). With `ink` set, give `print_box` |
| 24 Auto Draft | `quad`, `model`, `light`, `label`, `outline`, `pairs`, `occluders`, `mask_skip`, `print_box`, `bow`, `suggest` | as on the node ([Starting from rough input](#starting-from-rough-input-auto-draft), [Print straight on a body](#print-straight-on-a-body-the-ink-templates)). The template holds the ink demo's |
| 62 Refine (Video) | `processes`, `model`, `compositing`, `soft_cap`, `glue`, `keyframe_frames`, `tracker` | as on the node ([Video nodes](#video-nodes-image--product-label-refinement--video)); `processes` 8 on a server |

### Its outputs

| Node | File (in `output/`) | What |
|---|---|---|
| 71 | `product-label-refinement/triage/sheet_#####_.png` | the triage sheet: the print on the frame next to the packshot's |
| 77 | `product-label-refinement/triage/mask_#####_.png` | SAM 3's mask (or the given one) on the keyframe frame, tinted green with a red edge: check it fits the product before a fix |
| 70 Label Triage | (its text) | `triage.json`: the triage's measures (the call is the agent's), or why it couldn't measure |
| 44 | `product-label-refinement/refined_#####_.png` | the refined image (lossless PNG); for a clip, the refined keyframe |
| 40 QA Sheet | (its text) | `report.json`: the stills run's report and audit |
| 45-50 | `product-label-refinement/qa/{compare,edges,before_after,audit,align_overlay,draft_points}_#####_.png` | the QA sheets |
| 52, 54 | `qa/grade_preview`, `qa/relight_map` | the paper graph only |
| 56 | `qa/print_removed` | print on the body only: the body without its garbled print |
| 58 Save PSD | `product-label-refinement/refined_#####_.psd` | the paper graph only; `tools/comfy_run.py` runs it with `--psd` |
| 24 Auto Draft | `product-label-refinement/draft_#####_.json` (and its text) | the drafted job: edit it and give it back as the Job JSON |
| 63 | `product-label-refinement/video/refined_#####.mov` | clips: the lossless master of the whole clip with the refined shot spliced in (QuickTime PNG, exact RGB; its ui lists it under `files`); the next product's source |
| 65 | `product-label-refinement/video/review_#####_.mp4` | clips: the refined clip, to review |
| 66 | `product-label-refinement/video/qa_plot_#####_.png` | clips: Refine's QA plot |
| 80 | `product-label-refinement/video/masks_#####_.mp4` | clips: the shot's masks on its frames (SAM 3's track, or the given masks): check them through the shot |
| 67 Refine report | (its text) | clips: Refine's report JSON (`not_covered`, `partly_covered`, `flagged`, `review`, `geometry`). Its frames count from the shot's start |
| 75 Splice | (its text) | clips: `shot.json`, the refined shot (`start`, `end`, `keyframe_in_shot`, the `cuts` found, `clip_frames`): add `start` to a frame of Refine's report for the clip's frame |

### The flow nodes (image › product-label-refinement › flow)

| Node | In | Out |
|---|---|---|
| **Load Source (Image or Video)** | `source`: a file in `input/` | `frames` (one for an image), `fps`, `audio`, `is_video`, `frame_count`. Images go through Load Image, videos through Load Video's decoder: the same pixels as those nodes |
| **Load Optional File** | `file`, or `(none)` | `image` (every frame), `mask` (their red channel, as Load Image (as Mask) and Image To Mask read it), `given` |
| **Switch** | `when`; `if_true`, `if_false` (lazy) | the one picked; only it is computed. `when` is true for true, a non-empty string, a non-zero number or anything given |
| **Run If** | `value` (lazy), `when`, `run_when` | `value`, or a block: every node after it, output nodes included, is skipped without an error |
| **Print Type** | `job` | `light` (true for both polarities): the job lays print straight on the body (`matte` ink, polarity light or dark); `print` names it |
| **Pick Frame** | `index`; `images`?, `masks`? | that frame (clamped), and its index |
| **Shot (between cuts)** | `frames`, `keyframe_index`, `start`, `end`; `masks`? | the keyframe frame's shot: between the hard cuts round it (each frame shrunk to 64x64; a cut changes it by 0.06 or more on average and 8x the clip's median, and changes its colours), or `start` to `end`; its masks (a given mask video cut the same), `start`, the keyframe frame in the shot, and `shot` (JSON) |
| **Mask Overlay** | `images`, `masks`, `opacity` | the masks on their frames, tinted green with a red edge |
| **Packshot Matte** | `reference`, `mask`, `use`; `sam_mask`? (lazy) | the packshot's outline as Load Image's MASK: its alpha; else SAM 3's mask of it (computed only then) with `alpha, else SAM 3`; else Load Image's mask as it is (the engine traces the backdrop); and its `source` |
| **Splice Shot Back** | `frames`, `shot_frames`, `start`; `shot`? | the clip with the shot replaced at `start` (the shot itself when it is the whole clip); shows `shot` |
| **Paste-Back Mask** | `before`, `after`, `crop_rotate`, `grow` | where a straightened crop changed (8 bit), grown by `grow` px, for Crop & Rotate Composite; `straightened` false when the crop is the whole frame unturned |
| **Video Settings** | `settings`, `product_copy` | the settings JSON with the copy as `words`, and the copy |
| **Label Triage** | `frame`, `mask`, `reference`; `reference_mask`?, `product_copy`?, `ocr`, `glyph` | `sheet` (the print on the frame next to the packshot's, straightened, same size), `measures` (JSON; an error there never stops the run) |

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

### The one workflow over the API

`tools/comfy_run.py` drives `workflows/product-label-refinement-all.api.json` with the same flags:
- `--source PATH` uploads the image or clip into Load Source (1); `--scene` does the same when the workflow has no scene Load Image;
- `--reference PATH`: the packshot (2);
- `--mask PATH`: the product mask (3); without it SAM 3 runs;
- `--keyframe PATH`: a corrected keyframe (4);
- `--job PATH`: the job's text into the Job JSON (5); Auto Draft doesn't run;
- `--set ID.INPUT=VALUE` for everything else (the tables above), `--image ID=PATH` for any loader;
- the next product of an edit, on the last run's output without downloading it: `--set "1.source=product-label-refinement/video/refined_00001.mov [output]"` (the master's name from `history.json`).

`--out` then holds `triage.json` and the triage sheet, `report.json` (the stills report), `video_report.json` (Refine's), `shot.json` (the shot refined), every image, the draft job, the lossless `.mov` and the review mp4.

```bash
# triage: is the bottle's print wrong at all? (only SAM 3 and the sheet run)
python3 tools/comfy_run.py --server $URL --workflow workflows/product-label-refinement-all.api.json \
  --source examples/ink_demo/inkdemo_frame.png --reference examples/ink_demo/inkdemo_packshot.png \
  --set "11.text=bottle" --set 9.value=false --out ./triage

# a still with light print on the body: the ink demo, SAM 3 finding the bottle
python3 tools/comfy_run.py --server $URL --workflow workflows/product-label-refinement-all.api.json \
  --source examples/ink_demo/inkdemo_frame.png --reference examples/ink_demo/inkdemo_packshot.png \
  --set "11.text=bottle" --set 24.ink=light --set "24.print_box=176,396,432,924" --set "24.bow=-140 130/matte" --out ./run1

# a clip: the stills branch fixes frame 20, then Refine carries it through (the settings and copy optional)
python3 tools/comfy_run.py --server $URL --workflow workflows/product-label-refinement-all.api.json \
  --source clip.mov --reference packshot.png --set "11.text=serum bottle" --set 6.value=20 \
  --set 24.ink=off --set 24.print_box= --set 24.bow= --set 62.processes=8 \
  --set '7.settings={"render": {"body_snap": true}}' --set "7.product_copy=SAMPLE everyday hydrating serum" --out ./run2

# the same clip with a keyframe corrected elsewhere and a patched mask video: SAM 3 and the stills branch don't run
python3 tools/comfy_run.py --server $URL --workflow workflows/product-label-refinement-all.api.json \
  --source clip.mov --mask clip_mask.mov --keyframe keyframe.png --set 6.value=-1 --out ./run3
```

Without the tool: upload each file with `POST /upload/image` (any file type, videos too), set the same inputs by node id in the API file (`.["1"].inputs.source`, `.["3"].inputs.file`, `.["5"].inputs.value`, ...), `POST /prompt`, then read `/history/<prompt_id>`. Each Save Image lists `images`, Save Lossless lists `files`, SaveVideo lists its mp4, and the QA Sheet (40) and the Refine report (67) list `text`. Fetch files with `/view?filename=...&subfolder=...&type=output`.

An edit with three products in three shots and a group shot is six fix runs at most (the group shot: one run per product, back to front, its SAM 3 prompt naming that product), each on the last one's output; triage first and skip the fine ones.

Packshots on a backdrop: a drop shadow on a black or white backdrop gets traced with the product (the audit says the outline runs past the generated product, and the shadow lands in the frame). Set 83 to `alpha, else SAM 3`. Comparing a master with its source, decode the source as the workflow does (ComfyUI's VideoFromFile, as Load Source): PyAV's default rgb24 decode of an H.264 source differs by up to 29 levels, so it reports changes that aren't there.

The loop for an agent: triage each product first (`9.value=false`, one run per product, the SAM 3 prompt naming it) and fix only those whose sheet shows a wrong print; then run with Auto Draft; read `report.json` (`audit_verdict`, `align`) and the audit sheet; fix the draft job (`draft_#####_.json`) and run it again as the Job JSON. For a clip, iterate on the keyframe with Render the clip off (84 `false`: seconds a run), check the mask check (`triage/mask`) and the refined keyframe, then queue the same inputs with 84 `true` (the keyframe is reused; SAM 3's track and Refine run), check `video/masks` and the review mp4, then read `video_report.json`: `not_covered` and `partly_covered` say which frame to fix next (run again with that keyframe frame), and `review.actions` give settings to try.

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

Every pull request runs these checks on GitHub (`.github/workflows/checks.yml`, see [Maintaining](#maintaining)). Run them by hand for what a pull request can't cover, such as production jobs.

**Video.** `tools/regress_video.py` calls Product Label: Refine (Video) the way ComfyUI does (tensors in), on the same decoded frames as the skill's `video.py --refine`, and compares frames, matte, changed mask and report.

| Run (2026-10-16, imgvideo 0.3.0) | Result |
|---|---|
| The video demo (40 frames, 432 x 768, one keyframe): the node in-process vs `video.py --refine`, with and without the label's copy (`words`) | identical: frames 0 px, matte, changed mask and report; pixels outside `changed` are the input's |
| The video template on a live ComfyUI 0.18.2 (CPU, no matplotlib), the demo | success in 2 min: 40 of 40 frames refined, keyframe found at frame 20, 0 px changed outside the matte |
| A real 169-frame 1080 x 1920 clip with one corrected keyframe, the template on a live ComfyUI 0.18.2, `processes: 8` (CPU) | success in 20 min: frames 0-121 refined; the shot after the cut (122-138) reported as "reacquire, a cut at its edge", suggested keyframe 136; 0 px changed outside the matte. ComfyUI's video decode differs from ffmpeg's by a few levels, so a few flags land on other frames than in the CLI run |

**The one workflow.** `tools/regress_all.py` runs each case through `workflows/product-label-refinement-all.api.json` and through its own template on a live server, with the same inputs (via `tools/comfy_run.py`), and compares every output: images to 0 px, the reports, the draft job, the PSD and the lossless master frame for frame.

```bash
python -P tools/regress_all.py --server http://127.0.0.1:8188 [--video] [--keep DIR]
```

| Run (2026-10-18, ComfyUI 0.41.0, frontend 1.55.17, CPU) | Result |
|---|---|
| ink-draft, ink-job, paper-draft (PSD too), paper-job (PSD too) | all PASS: every image 0 px, the same `report.json`, draft job and PSD |
| video: the demo's clip, mask video and keyframe | PASS: the lossless master identical on 40 of 40 frames, the same Refine report |
| The video demo end to end (by hand): the stills branch on frame 20 from a packshot of the demo's tube, then Refine, against the draft template on that frame and the video template given its keyframe | identical keyframe (0 px, same report and draft job), identical clip (40 of 40 frames, the same report); 3.5 min |
| The UI file loaded in the frontend | no missing nodes; its own API export equals the API file |
| Triage only (`Fix this product` off) on the ink demo and on the video demo's frame 20 | 1 s each: only the sheet and `triage.json`; the sheets show the garbled print next to the true one |

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
| Engine 0.11.0 (print straight on a body: `matte` ink light, `align.bow`, 2026-10-16): the demo with `--psd`, the stills fixture and 10 production jobs (the dark ink matte among them); the ink demo; the four light-ink keyframes of a real job (two products in a tray, both printed on a box with a bow) | all 16 pass: the production jobs within 1 level on 0 to 9,053 px as before, the light-ink jobs within 1 level on 0 or 1 px with the same `matte` report; the engine's own outputs for the 10 production jobs are byte-identical to 0.10.2's. The ink-draft template on the ink demo gives the same draft job and image as `run.py draft` |
| Engine 0.12.0 (dark print on a body: `matte` ink dark, 2026-10-18): the engine's own outputs, before and after, for 15 jobs (the stills fixture, the ink demo, the four light-ink keyframes, 9 production jobs with the paper ink matte, cylinders and labels); through the pack, the demo with `--psd`, the ink demo, two light-ink keyframes and two dark-ink keyframes of real frames (a pink jar, a white bottle with a gold bar); the ink-draft template with `ink` dark on both frames | the 15 engine outputs byte-identical (images, PSDs, QA sheets, reports); all pack runs pass at 0 px, with identical `matte` reports; the ink-draft keyframes bit-identical to `run.py draft` plus `run.py` (the draft jobs differ only in `grade`, which the ink route doesn't read) |
| Engine 0.13.0 (dark print: each line's bow from the frame's own print, and the `ink_shape` check, 2026-10-20): the engine's own outputs for the same 15 jobs; the approved dark-ink keyframes of a white bottle and a toner; through the pack, the demo with `--psd`, the ink demo, a light-ink keyframe and the three dark-ink keyframes (the jar, the bottle, the toner) | the 15 engine outputs byte-identical; the bottle and the toner byte-identical to their approved keyframes (the print's bow didn't register better, so it isn't used); the jar's lines now bow 72-85 px against the generated print's 73-88 (one bow from the mask's ends gave 39-61); all pack runs pass, at most 2 px by 1 level, with identical `matte` reports |
| Engine 0.13.1 (Auto Draft `bow` print for a packshot warped whole, the audit's `bow` check, a `bow` given in text reaches the job again, 2026-10-25): the engine's own outputs for the same 15 jobs, before (0.13.0) and after; the jar's v3 keyframe; a squeeze tube's keyframe (frame 130) with its draft's fit, flat and bowed; Auto Draft against `run.py draft` on the tube with `bow` print | the 15 outputs, PSDs, QA sheets and reports byte-identical (the new check stays quiet on all of them); the jar's keyframe byte-identical; the tube's rules bow +6.1 and +9.5 px and its band and cap seam sag 7 and 16 px, against the frame's 6.9, 9.6, 9 and 17 (flat: 0, 0, -3), and the audit's `bow` warns on the flat one only; Auto Draft's job identical to `run.py draft`'s. Not run on a ComfyUI server (no local install that day): the vendored engine is the skill's, byte for byte |

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

### The routine: a branch, a pull request, the checks, one click

Each engine sync or pack change is made on its own branch, in its own worktree, and reaches `main` through a pull request. Nobody commits to `main` directly and no two sessions share a working tree, so a push never has to sort out whose files are whose.

1. **Branch.** One worktree per change, cut from `origin/main`:
   ```bash
   cd ~/Documents/Swordfish/ComfyUI-ProductLabelRefinement
   git fetch origin
   git worktree add ../plr-worktrees/TOPIC -b TOPIC origin/main
   ```
   Work and commit there. An engine sync is a change like any other: in its worktree, run `tools/sync_imgpost.py --update` (and `--mark-ported` once `run.py` is ported into `stages.py`) or `tools/sync_video_engine.py --update`, and commit everything they change: the vendored packages, `tools/skill_cli/` and `imgpost.lock.json`.
2. **Pull request.** Push the branch (`git push -u origin TOPIC`) and open a pull request against `main`. Without the gh CLI, the REST API opens it: `POST /repos/nastyakolykhan-sudo/ComfyUI-ProductLabelRefinement/pulls` with `head`, `base`, `title` and `body`.
3. **Checks.** GitHub Actions runs `.github/workflows/checks.yml` on the pull request, on CPU, in parallel jobs (about a quarter of an hour; the video jobs take longest):

   | Job | Runs |
   |---|---|
   | checks | `tools/brand_check.py` on the change; the pins (`tools/sync_imgpost.py`, `tools/sync_video_engine.py`); `tools/regress.py --psd` on the demo |
   | one workflow, CASE (5 jobs) | `tools/regress_all.py --only CASE` on a ComfyUI 0.41.0 started in the job: ink-draft, ink-job, paper-draft, paper-job, video |
   | video node (2 jobs) | `tools/regress_video.py` on the video demo: on ffmpeg 5.0.1, and on Ubuntu's own ffmpeg 6.1 |

   A red job names the step that failed; its runs are kept as an artifact for 7 days. The same checks run on `main` after each merge.
4. **Merge.** The user merges the pull request with one click once the checks are green.
5. **Swordfish pulls `main`.** Save the machine again so it takes the new version, then start an instance (the swordfish skill, "Update a machine"). A running instance doesn't pick the update up.
6. **Clean up.** `git worktree remove ../plr-worktrees/TOPIC`, then `git branch -d TOPIC`.

If `main` moves while a pull request is open and both touch the same lines, GitHub shows a conflict: merge `origin/main` into the branch in its worktree, resolve, push, and the checks run again. A conflict in `imgpost.lock.json` is resolved by re-running the sync on the merged tree.

**What the checks compare against.** GitHub has no skill folder, so each job rebuilds the skill's layout from what the pins cover: `imgpost/`, `imgvideo/`, `imgauto/`, and the skill's two command lines in `tools/skill_cli/` (the `run.py` that `stages.py` ports, and `video.py`). `regress.py` and `regress_video.py` then run unchanged: the nodes against the pinned engine. The sync checks compare the vendored copies with `imgpost.lock.json`. Two things stay local, because they need the skill folder or client files: whether the pack has fallen behind the skill (the same sync commands with the skill present), and regressions on production jobs.

**Brand names.** `tools/brand_check.py` scans the lines a branch adds, the names of the files it adds and its commit messages against a private list of terms (regular expressions, one per line, case-insensitive). The list is the repository secret `BRAND_TERMS` (Settings › Secrets and variables › Actions), never a file here: the check prints where a term matched and its number in the list, not the term, since the logs of a public repository are public. Locally: `python3 -P tools/brand_check.py --terms FILE` (`--all`: every tracked file). A pull request from a fork doesn't get the secret, so its check fails.

**Models and ffmpeg in the checks.** AllTracker (MIT, huggingface.co/aharley/alltracker) and SAM 3.1 (SAM License, huggingface.co/Comfy-Org/sam3.1) are fetched at pinned revisions, checked against their sha256 and cached between runs. The one workflow's triage runs SAM 3 on the packshot in every case (the demo packshots have no alpha); the given product masks keep it off the frames. The jobs run on ffmpeg 5.0.1 (a static build, pinned by sha256 and cached), the build the fixtures and templates are made on; the engine runs on ffmpeg 6 to 9 too (imgvideo 0.10.1), so the second video node job runs on Ubuntu's own ffmpeg 6.1. Changing the ComfyUI version, torch, ffmpeg or a model is a change to the env block at the top of `checks.yml`, made on a branch like any other.

### Tools

Video: `imgvideo/` and `imgauto/` are the skill's too, pinned under `video_engine` in `imgpost.lock.json` with the skill's `video.py` (`tools/skill_cli/video.py`):

```bash
python3 -P tools/sync_video_engine.py              # check; --update copies the skill's imgvideo/, imgauto/ and video.py here
python -P tools/regress_video.py --comfy PATH/TO/ComfyUI   # Refine (Video) against the skill's video.py --refine, bit for bit
python3 -P tools/make_video_demo.py examples/video_demo    # redraw the video demo (byte-identical)
```

Stills:

`imgpost/` is a byte-identical copy of the product-label-refinement engine's library. The pack vendors it rather than importing it so the math stays identical and the pack stays self-contained on any server. `imgpost.lock.json` records the sha256 of every vendored file and of the runner version that `stages.py` ports; `tools/skill_cli/run.py` is that runner.

```bash
python3 tools/sync_imgpost.py                    # check: local edits, engine changes, runner changes
python3 tools/sync_imgpost.py --update           # copy the engine's imgpost/ here and re-pin
python3 tools/sync_imgpost.py --mark-ported      # after porting a runner change into stages.py (copies the runner too)
python -I tools/build_workflows.py               # rebuild workflows/ (the one workflow too) after changing a node's inputs or outputs
python -P tools/regress_all.py --server URL --video   # the one workflow against each template, on a live server
python3 -P tools/brand_check.py --terms FILE     # brand names in the branch's changes (--all: every tracked file)
python3 tools/make_demo.py                       # redraw examples/demo/ (byte-identical with the same libraries)
python3 -P tools/make_ink_demo.py                # redraw examples/ink_demo/ (the light-ink demo; uses imgpost/)
```

Never edit `imgpost/` here. To pick up an engine change:
1. Change the engine.
2. In the change's worktree, run `--update`.
3. Port any runner diff into `stages.py`, then `--mark-ported`.
4. Run `tools/regress.py` (the pull request runs it too).
5. Commit, push the branch and open a pull request (above).

## Extending

- **New grade options.** Port the branch from the runner into `stages.grade`; the job's `grade` section passes it through unchanged. To add a switch on the node, add an optional combo whose first choice is `job`, after the existing ones, as `relight` was. UI workflows store widget values by position.
- **Measuring helpers.** `imgpost/autofit.py` holds the engine's measuring helpers: they snap a rough polyline to the colour edge below a pixel, refine a rough landmark by correlation with a trust verdict, trace a packshot's outline, fit an outline to a mask, and propose edge lines and the old silhouette from a rough placement. `imgpost/draft.py` puts them together, for the engine's `run.py draft` and for Auto Draft. Mask inputs already take scene-sized masks, so a mask from another node can replace Job Masks' outputs. Score any change to the draft with `tools/regress.py --draft` against hand-measured jobs before trusting it.

## Not covered yet

- Light print straight on a bottle, placed exactly on the generated print from a mask alone: the light route places it by the silhouette (above); dark print is registered onto the generated print. `label: cylinder` needs the label's top and bottom to be edges in both images, which print on a body doesn't have.
- Print on a body from a small packshot: the frame shows an 800 px packshot's small print soft however its edges are sharpened (QA Sheet's `packshot` WARN).
- A clip with dark print on a body: the keyframe comes out right, but away from it Refine (Video) leaves a ghost of the frame's own print where the generated print drifts against the tracked label (a jar's 49-frame shot through the one workflow, 2026-10-18: a line of text doubled 8 to 40 frames from the keyframe), and its spots check flags dark print on every frame.
- The turn of a label round its bottle isn't pinned by the edges (above): a 4096 px label can land 5 to 11 px round from where a hand fit puts it.
- Batch mode: one job across many frames.
- A GPU path. Everything runs the engine's numpy and scipy code on the CPU, about 4 to 15 s per product.
