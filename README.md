# ComfyUI-ImagePost

Fix a hallucinated product in an AI-generated still by compositing the real product from its packshot, inside ComfyUI. The nodes warp the packshot onto the generated product's perspective and carry over the scene's light and colour. That includes cast shadows, falloff, gloss and haze, and the round light on cans and bottles. They match the frame's softness and grain, put the product behind whatever stands in front of it, and fill in the background where the generated product was bigger. Every run ends with QA sheets, a report, and a layered PSD if you want to finish by hand.

It is deterministic compositing with no models. Every measurement lives in a job file (`job.json`): the work box, landmark points, edge lines, outlines, foreground objects, and the grade, finish and fill settings. The same job gives the same pixels every time. That lets an agent or a script drive it through ComfyUI's API, check the report and QA sheets, edit the job and run it again.

A job doesn't have to be measured by hand first. **Auto Draft** makes one inside ComfyUI from rough input: the scene, the packshot, where the product roughly sits (four corners or a box), and optionally a product mask from SAM 3. On a clean scene the draft is close to final; on a cluttered shelf it is a scaffold to finish by hand. [Starting from rough input](#starting-from-rough-input-auto-draft) has the scores.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/nastyakolykhan-sudo/ComfyUI-ImagePost
```

It needs no extra dependencies: numpy, scipy and Pillow (10.1 or later) ship with ComfyUI. Restart ComfyUI. The nodes are under **image › image-post**.

## Try the demo

1. Copy `examples/demo/imagepost_demo_scene.png` and `examples/demo/imagepost_demo_packshot.png` into `ComfyUI/input/`.
2. Load `workflows/image-post.json` and queue it.
3. The refined image lands in `output/image-post/`, the QA sheets in `output/image-post/qa/`, and the layered PSD next to the image.

To start from rough input instead, load `workflows/image-post-draft.json`: Auto Draft takes the place of Load Job, with the card's corners read a few px off, the mug traced roughly and two landmarks clicked near the header band. It also appears in ComfyUI's template browser, under ComfyUI-ImagePost. The draft job it made is saved as `output/image-post/draft_00001_.json`.

The demo is a made-up product card. In the "generated" scene its text is garbled ("PRDOUCT", "sapmle lable"), its emblem and grid are wrong, its corners are square and its top is too tall. A mug stands in front of its lower right corner and casts a shadow across it. The packshot is the clean card. `tools/make_demo.py` draws all three files from code, so they show no real product or brand. `examples/demo/job.json` is the worked example of a job.

## Nodes

There is one node per step. The math is the image-post engine's (`imgpost/`, see [Maintaining](#maintaining)), split at the node boundaries in `stages.py`.

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

Inputs, all in pixels (scene px unless said otherwise; numbers separated by spaces, commas or new lines):

- `quad`: where the packshot outline's bounding box corners land in the scene, `x,y x,y x,y x,y` (top left, top right, bottom right, bottom left). A box `x0,y0,x1,y1` around the product also works when the product stands upright and faces the camera. A few px off is fine.
- `product_mask` (optional): the generated product's mask, scene-sized, from SAM 3, the mask editor, or any node with a MASK output. The outline is then fitted to the mask's contour instead of to image edges. With a mask, `quad` may stay empty: the fit starts from the mask's box, its hull's corners and its main axis, and keeps the start that sits best on the contour. That works when nothing stands in front of the product; when something does, give the quad too.
- `pairs` (optional): rough landmark pairs, `rx,ry=sx,sy` (reference px = scene px), one per line. Points on garbled generated print usually come back untrusted and are ignored.
- `occluders` (optional): objects in front, one per line: `left: x,y x,y ...` (the occluder fills that side of the line: `left`, `right`, `above` or `below`) or `polygon: x,y x,y ...`. Each is snapped to the occluder's own edge.
- `outline` (optional, reference px): the product on the packshot, as a box or a polygon, when the packshot shows more than the product (a box and its tube, a label on a bottle).
- `mask_skip` (optional, reference px): a polygon whose part of the outline isn't fitted to the mask (where the mask runs on into a cap the job keeps).
- `model`: `homography` or `affine`. `light`: `relight`, `cylinder` (a shading template for cans and bottles) or `off`. `suggest`: also add packshot corners that match the scene well as landmarks.

Outputs:

- `job`: feeds the other nodes, exactly like Load Job's.
- `job_json`: the job as text. Paste it into Load Job to finish it by hand, or into the engine's job folder; add `scene` and `reference` paths to run it with `run.py`.
- `draft_points`: each landmark pair as packshot under the fit, then scene.
- `report`: the draft's summary plus the align table. The node shows it, and each run also saves the job as `{filename_prefix}_NNNNN_.json` in the output folder.

Check `align_overlay` and the report before trusting a draft: a side marked `left` means the generated edge disagreed and the fit doesn't use it. The rough corners stay in the job as weak placeholders (`w` 0.15 or 0.05); replace them with landmarks you read.

### How close a draft gets

Each draft was scored against the hand-measured job for the same product (`tools/regress.py --draft`, below). The rough input is what a person would give: the quad 3 to 4 px off at every corner, the product's outline on the packshot, and the occluders 2 px off. The mask is the hand-measured generated outline minus what stands in front of it, which stands in for a good SAM 3 mask; no real SAM masks were scored.

- **Edge**: how far the draft puts the product's outline from where the hand fit puts it (median / max, scene px).
- **Placement**: how far any point of the product, its print, lands from the hand fit's position (median / max).
- **Geometry vs hand**: the hand job run with the draft's measurements (work box, fit, occluders, old silhouette) and its own grade: pixels more than 8 levels from the hand job's image, and their share of the pixels the hand job changed.
- **Draft vs hand**: the same for the draft as it comes, with its default grade (the hand jobs tune theirs).
- **Audit**: the draft run's own verdict, with the hand job's in brackets. The hand jobs' WARNs were checked as false positives before delivery.
- **With mask**: edge and placement with the mask added to the quad. **Mask alone**: with no quad.

All columns but the last two use the quad and no mask. Results on 2026-10-16, engine 0.9.0:

| Product | Edge | Placement | Geometry vs hand | Draft vs hand | Audit | With mask: edge, placement | Mask alone: edge, placement |
|---|---|---|---|---|---|---|---|
| the demo card, a mug in front (1024 × 768) | 0.5 / 2.8 | 0.7 / 2.8 | 12,324 (13%) | 12,349 (13%) | WARN (WARN) | 0.6 / 7.2; 3.5 / 7.2 | 1.2 / 19.2; 6.8 / 19.6 |
| a box on a plain studio set (2286 × 4096) | 1.4 / 4.7 | 14.4 / 19.8 | 161,719 (32%) | 471,067 (93%) | WARN (WARN) | 1.0 / 8.1; 10.1 / 13.7 | 0.5 / 1.5; 1.4 / 1.9 |
| a lip tube on the same set: cap kept, 10 glints in front | 1.1 / 3.7 | 3.7 / 6.0 | 46,746 (24%) | 111,497 (56%) | WARN (WARN) | 1.3 / 2.0; 2.8 / 3.8 | 2.4 / 76.9; 38.3 / 76.7 |
| a tube's printed panels in a mesh bag (4096²) | 1.4 / 3.5 | 2.0 / 3.5 | 45,193 (12%) | 200,557 (52%) | WARN (WARN) | 2.6 / 4.1; 2.4 / 3.9 | 2.6 / 4.6; 2.4 / 4.3 |
| a dropper bottle's label in the same bag, bottle kept | 2.3 / 7.3 | 32.8 / 41.6 | 80,919 (31%) | 181,497 (70%) | WARN (WARN) | 2.2 / 8.1; 34.2 / 42.6 | 1.9 / 7.6; 33.3 / 41.0 |
| a bottle's label in the same bag | 1.2 / 6.3 | 53.4 / 67.0 | 144,076 (24%) | 325,509 (55%) | WARN (WARN) | 0.9 / 10.4; 54.1 / 71.3 | 1.2 / 16.5; 57.8 / 73.8 |
| a bottle label in a cup carrier (1024²) | 0.5 / 23.4 | 5.2 / 23.9 | 5,520 (46%) | 9,819 (82%) | WARN (clean) | 0.4 / 3.7; 3.1 / 3.8 | 0.4 / 2.3; 3.3 / 4.0 |
| a frosted bottle's label in the same carrier | 0.4 / 8.4 | 7.1 / 9.3 | 6,595 (39%) | 16,015 (94%) | WARN (clean) | 0.4 / 10.2; 4.2 / 10.0 | 0.4 / 8.5; 3.5 / 8.6 |
| a bottle label in the same carrier, its wall over a corner | 0.7 / 4.2 | 4.0 / 5.1 | 4,386 (39%) | 9,954 (89%) | FAIL (clean) | 0.5 / 3.5; 2.6 / 4.1 | 0.5 / 3.7; 2.7 / 4.7 |
| a bottle label on a video's anchor frame (1080 × 1920) | 0.7 / 4.4 | 9.5 / 11.1 | 10,944 (48%) | 10,944 (48%) | WARN (FAIL) | 0.7 / 3.2; 9.7 / 11.4 | 0.5 / 3.9; 8.8 / 10.2 |
| a blister card on a cluttered shelf, behind a box and the shelf lip (4096²) | 0.8 / 2.8 | 1.9 / 5.3 | 74,776 (52%) | 117,754 (81%) | WARN (WARN) | 3.3 / 18.7; 13.2 / 20.8 (stopped) | 5.3 / 75.1; 35.3 / 75.5 (stopped) |
| a text block printed on a tube's body (ink matte) | 0.6 / 2.8 | 1.2 / 2.7 | 10,987 (24%) | 41,231 (90%) | WARN (WARN) | 10.4 / 40.5; 20.0 / 40.4 | 20.5 / 43.1; 34.4 / 43.0 |
| a user-fixed render as the reference (identity fit) | 0.7 / 2.2 | 1.1 / 2.7 | 34,757 (33%) | 100,092 (95%) | WARN (WARN) | 3.5 / 6.3; 2.9 / 6.1 | 4.6 / 51.8; 23.5 / 51.6 |

What it shows:

- **Flat products with a quad: close to final.** The demo card, a tube's printed panels and the blister card on the cluttered shelf (given its occluders) land within 0.5 to 1.4 px of the hand outline (median; 2.8 to 3.5 px at worst), with their print within 2 px (5.3 px at worst). The shelf card is no worse than the clean scenes once the quad and the occluders are given; what the shelf costs is the occluders to trace and the edges the draft can't use (only one of its sides passed).
- **A tall box needs landmarks or a mask.** With no landmarks the box's perspective comes from four corners each 3 px off, and its print slides up to 20 px along its long sides while its outline stays within 5 px. A mask alone fixes that: outline within 1.5 px and print within 1.9 px at worst.
- **A mask helps where nothing stands in front.** With the quad, a mask brings the tube's worst edge from 3.7 to 2.0 px and one carrier label's from 23 to 3.7 px; on the frosted label it doesn't help (8.4 to 10.2 px). Alone, it fails where something stands in front: 19 px on the demo (the mug), 77 px on the tube (the cap and glints), and on the shelf card the run stops with FILL STOPPED (the mask's contour along the box in front boxes in a sliver). Give the quad as well whenever the product is partly hidden.
- **Labels on bottles: a scaffold.** The draft fits a flat homography. A label's outline lands within a pixel or two (median), but its print, which the hand jobs wrap round a cylinder, is 4 to 10 px off on the small labels and 33 to 53 px on the large ones. One label draft FAILs its audit (the print over the carrier wall and the glass beside it). Finish these with the cylinder model and the generated label's own shape, by hand.
- **Pixels.** Even with the hand job's grade, the closest drafts (the demo, the panels) differ from its image by more than 8 levels on 12 to 13 % of the pixels it changed, the rest on 24 to 52 %: a 1 px shift moves every print edge. A text block on a tube's body and a fixed render as the reference are in the table for completeness; neither is a case for a draft.

## Conventions

- **Work box.** `scene` is the whole generated frame, and the job's `roi` is the work box. Every other IMAGE or MASK passed between Image Post nodes covers only the work box. Composite Behind pastes it back, and every pixel outside the box passes through untouched.
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
  --workflow workflows/image-post.api.json \
  --scene examples/demo/imagepost_demo_scene.png --reference examples/demo/imagepost_demo_packshot.png \
  --job examples/demo/job.json --out ./ip-run
```

- `--scene` and `--reference` upload to `input/image-post/` and set the Load Image nodes that feed Warp Reference. `--image NODE_ID=PATH` targets any other Load Image node.
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
  --workflow workflows/image-post-draft.api.json --scene scene.png --reference packshot.png \
  --set "ImagePostAutoDraft.quad=372,170 612,186 610,566 370,578" \
  --set "ImagePostAutoDraft.pairs=" --set "ImagePostAutoDraft.occluders=polygon: 593,428 593,662 705,662 705,428" \
  [--mask product_mask.png] [--stage align] --out ./ip-draft
```

Clear the template's demo `pairs` and `occluders` (as above) unless you set your own. `--out` then also holds `ImagePostAutoDraft-3.txt` (the draft's summary and align table) and `image-post/draft_00001_.json`, the draft job to check, edit and run with `--job` through `workflows/image-post.api.json`.

The same steps with curl. Upload both images:

```bash
curl -s -F "image=@scene.png" -F "subfolder=image-post" -F "overwrite=true" http://127.0.0.1:8188/upload/image
```

Point nodes 1 (scene) and 2 (reference) at `image-post/<name>`, then queue the prompt:

```bash
jq '{prompt: (.["1"].inputs.image = "image-post/scene.png" | .["2"].inputs.image = "image-post/packshot.png")}' \
  workflows/image-post.api.json > prompt.json
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
curl -s -o refined.png "http://127.0.0.1:8188/view?filename=imagepost_demo_scene_refined_00001_.png&subfolder=image-post&type=output"
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

Measure landmarks and edges by colour change, not by the strongest gradient, and accept a fit at 1.5 px or less on the visible edge lines. The image-post engine's job spec documents every field.

## Regression

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
| Auto Draft against `run.py draft` (`--draft`): the same 13 jobs, each with a quad, a quad and a mask, and a mask alone | all 39 pass: identical draft jobs; images within 1 level, or both runs stop with the same FILL STOPPED (the shelf card with a mask) |

ComfyUI passes images between nodes in 32-bit floats, where the engine keeps 64-bit, so a value can land on the other side of a threshold. The 12 × 11 px patch is where relight's fine pass kept a slightly different region. Nothing outside the work boxes differed in any run.

### Scoring a draft

`--draft` scores Auto Draft against a hand-measured job:

```bash
python -I tools/regress.py --job path/to/job.json --draft [--jitter 3] [--outline hand|traced] [--rough-pairs] \
  [--mask PATH|hand [--no-quad]] [--no-occluders]
```

From the hand job it builds the rough input a person would give:
- the quad: where the hand fit puts the outline's bounding box corners, each corner `--jitter` px off;
- the outline on the packshot (`--outline traced` lets the draft trace it);
- the hand job's occluders, 2 px off;
- with `--rough-pairs`, its landmarks, `--jitter` px off;
- with `--mask hand`, a mask of its generated outline minus its occluders, or `--mask PATH` for a real one.

The pack runs `workflows/image-post-draft.api.json` on that input, and the engine runs `run.py draft` on the same. Their draft jobs must be identical and their images must match as above, or the check fails. Then the draft is scored against the hand job: edge, placement, pixels against the hand job's image, and both audit verdicts (the table above). It prints a `row` line for that table and writes everything to `regress.json`.

`tools/offline_run.py` runs any API workflow in-process, with no server. It uses ComfyUI's own LoadImage and SaveImage conversions and validates inputs and link types the way ComfyUI does. `regress.py` uses it, and it takes the same arguments as `comfy_run.py`.

## Maintaining

`imgpost/` is a byte-identical copy of the image-post engine's library. The pack vendors it rather than importing it so the math stays identical and the pack stays self-contained on any server. `imgpost.lock.json` records the sha256 of every vendored file and of the runner version that `stages.py` ports.

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

- Labels on bottles in the draft. Auto Draft fits a flat homography, so a label wrapped round a bottle lands 30 to 60 px off in its print even when its outline fits within a few px. Those jobs still need the cylinder model, measured by hand.
- Picking the quad on the canvas. It is typed as numbers (or comes from a mask); there is no point picker on the node yet.
- Batch mode: one job across many frames.
- A GPU path. Everything runs the engine's numpy and scipy code on the CPU, about 4 to 15 s per product.
