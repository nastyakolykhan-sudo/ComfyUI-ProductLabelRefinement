# imgvideo changelog

The video engine's version is `imgvideo.__version__`. It is developed in the ComfyUI-ProductLabelRefinement pack; the product-label-refinement skill carries a copy (`tools/sync_engine.py`, which records each package's version and files in `imgpost.lock.json`). The skill's `python -P scripts/video_regress.py --manifest` prints both.

Every release passes the video fixtures (`examples/video/regression.json`) at 0 px. Bump the minor version for any change that alters a render (then re-baseline the fixtures and say which), and the patch version for one that doesn't.

## 0.21.0 (2026-11-20)

Blend frames stay on their tracks (pitfall V82). A morph frame (a blend of two poses) borrowed its neighbours' shape
unless its own fit held the tracks 1.5 px better; on a bouncing tube the borrowed shape moved the label 2.0-3.2 px off
the tube's own tracks (held within 0.27-0.36 px) for a frame and back, which the user saw as the label jumping.

- **`fit.MORPH_OWN` `tight_px` 0.5, `shift_px` 1.0**: a morph frame whose own fit holds its tracks within 0.5 px keeps
  it where the borrowed shape would move the label 1 px or more. `morph_test` entries gain `shift_px`, and `kept`
  reads "own fit (V82)" where this rule decided. Renders change on such frames and, through glue and the still
  label's median over every frame's fit, slightly on the rest of the run: the peach set's tube (11 frames switched)
  and its approved serum (2, 39, 118: 1.15-1.45 px), the other test clips identical.
- **Review check `twitch`** (pipeline adds each frame's `own_fit` to the review's measures; review.DEFAULTS
  `twitch_own_px` 0.5, `twitch_px` 1.0): a morph frame left on its borrowed shape that way. The pack's Review Name
  fails a clip on it (with drop and drift).
- The 8 fixtures unchanged (0 px).

## 0.20.0 (2026-11-17)

A shot's edges (pitfall V81). The shots Every Shot gives are refined to their edges, and two new review checks fail a
clip whose label drops out next to its run or turns off the body where a run starts or ends. Single-shot clips render
as before; the checks read every clip.

- **`fit.fit_shot(..., carry=False, carry_gate=None)`, `fit.carry_step`, `fit.CARRY`**: past the held run, frame by
  frame for as long as the tracks and the matte allow (no frame count), the control points are moved frame by frame by the similarity the product's own tracks follow
  (visible on both frames: 24 and a fifth of them at least; inliers 0.6; median residual at most 3 px or 5 % of the
  frame's move; scale 0.8-1.25); `carry_gate` (pipeline's `_carry_gate`, the review's drop limits on the matte) stops
  it where the product's matte shrinks under 0.35 of the run's end frame's or overlaps the frame before under 0.45 (a
  cut, a wipe). The result gains `carried`, `carry` (each tried frame's measures or why it stopped) and `held` (the
  tracks' own run). Off by default (`fit.carry` false); Refine (Video)'s shots path turns it on.
- **Refine (Video) `shots`**: each shot runs over its whole range (`settings.shots` [[0, n]]): the shot analysis's own
  cuts read a camera move that swaps the background behind the product as a cut (pump clip: 118|119, 119|120) and
  stopped the label short; the tracks decide where it holds.
- **The print-following routes (V78, V80) keep each frame's own fit**: `fit.shear_max` off there (unless the settings
  give it) and the fit run again: a morph frame's borrowed shape turned the label 4.2 deg off the body on a shot's
  third frame (the tracks missed by 2.8 px; its own fit 0.8 px). Reported as `keyframes[i].printed_body.fit`.
- **The morph test on every route** (`fit.MORPH_OWN`, the fit's `morph_test`; `fit_shot(..., whole=)`, pipeline's
  `_whole`): a morph frame with the product whole in frame whose own fit holds its tracks (1 px) keeps it where the borrowed shape misses them by 1.5 px more or turns the label's axis 1.5 deg off it.
  Measured: the pump frame +1.98 px (axis 0.08 deg); the approved serum's and tube's morph frames +0.29 to +1.16 px,
  within 0.46 deg, kept (their fits byte-identical to 0.19.0's). A product cut off by the frame keeps the borrowed
  shape: its tracks hold part of the label (an approved spray bottle entering from the top: its own fit moved the print
  on that frame, release 39).
- **The print-following routes' morph frames, measured** (the pump's first shot, the 0.19 engine's borrowed shapes
  against v21's own fits, track rms): frame 3 4.00 -> 0.65 px, 40 1.29 -> 0.12, 58 2.73 -> 0.42, 68 1.93 -> 0.50, 119
  4.37 -> 0.50; the shot's median 0.43 -> 0.42.
- **Review `drop` and `drift`** (`review.evaluate`; DEFAULTS `drop_*`, `drift_*`; `frame_measures` gains `iou`, the
  mattes' overlap as they lie): `drop`, a frame left as generated next to a refined run, for as long as the same
  product and its print carry on (matte 0.35 of the run's end frame's, print detail half of the run's end frames',
  mattes overlapping 0.45 and their silhouettes 0.5 frame to frame; a camera move or a pan doesn't stop it, unlike
  `vanish`'s cut); `drift`, on a run's first or last 4 frames the label's axis off the body 1.5 deg off the run's frames
  round it (the body check's 5-frame median hides a one-frame turn). Review Name: any of either FAILs the clip.
  Calibration: the v20 pump clip `drop` 120, 121, 143-144, 163-165 (mattes 0.45-1.0, detail 0.69-0.99, overlaps
  0.58-0.93), `drift` 123 (2.2 deg; the run's own frames 0.1-0.9); the approved serum and tube clips none (the tube's
  close-up before its run: mattes overlapping 0.29 at the cut; `vanish` still reads it).

## 0.19.0 (2026-11-16)

Every shot the product is in (pitfall V80). The one workflow refines each shot that likely shows the product from its
own keyframe; the engine gains the shot merge and a second trigger for the route that carries the keyframe's edit.
Single-shot clips and the fixtures render as before.

- **`pipeline.merge_shots(results, shots, frames, masks, work)`**: shots refined one by one (refine() on each shot's
  own frames, exactly a run of that shot alone) put back into the clip; the report keeps each shot's own report under
  `shots` and merges covered, not_covered, partly_covered, flagged, keyframes, cuts and the review (refined and flagged
  summed, each check's frames in clip frames; kf_kept and carried the lowest, kf_old the highest); the QA plots stacked.
- **The route where the old light print is wide** (`OLD_WIDE` 14 px, `old_print_wide`): on the body route's trigger
  check, where the keyframe frame's widest old stroke (`body_open`'s measure) is over 14 px (imgpost.ink WIDE_FROM 2 x
  frame_open 7), the job takes the slide route (follow the print, the keyframe's whole edit, body_floor, body_open,
  lock off) even when the print doesn't slide; reported as keyframes[i].printed_body {"by": "the old print's strokes
  are wider than the clip's removal reads"}. Measured: a coloured pump bottle in its box 23.3 px (the body route:
  double and ghost on 17 of 17 frames); the approved frosted pump 4.5, tube 10.0 (unchanged).

## 0.18.0 (2026-11-13)

Light print on a round body whose generated print slides on it: the label follows the print and carries the
keyframe's whole edit (pitfall V78). Only where the body route would hold the label off the generated print past the
compositing's reach: the fixtures and the approved clips render as before.

- **The route decided from the shot** (`pipeline._print_on_body`, `PRINT_SLIDE` held 0.75): after the fit, the label
  is placed as the body route places it (body snap, glue, level, limb snap; no render) and measured against the
  tracked label at its inner grid points facing the camera, p90 per frame, then p90 over the shot. Over 0.75 x
  render.print_alpha `near` (14 px: how far from the new print the old print is replaced) the job takes V76's route
  on its own surface: render.print_alpha {ink "edit", presence off}, body_snap, glue, level, limb_snap and feather
  off (reported as keyframes[i].printed_body {"by": "the generated print slides on the body"} and
  keyframes[i].print_on_body). Measured: the coloured pump bottle 15.1 px; the approved frosted pump and tube 7.4,
  6.8. The print's slide along and round the body as body_snap measures it (shares of the label) didn't separate
  them (the approved pump slid more: its bottom edge read a steady perspective offset).
- **render.body_floor** (2.0 levels, set by that route only): where the output differs from its source frame, no
  channel darker than the source's body under its light print (a 15 px grey opening per channel) less 2 levels.
  Carried with the print, the keyframe's fill beside a wordmark sat 10-20 levels darker than another frame's body.
- **print_alpha.lock off** on this route (0.17.0's print lock reads the generated graphics round each line; a plain
  body has none, and the lock read the body the print slides on: `lock` on 58 of 120 frames).
- **Review Name:** with the keyframe's whole edit carried (kf_kept reported), `spots` on more than half the refined
  frames is _REVIEW-FAIL (a keyframe defect repeated on every frame; the pump before body_floor: 119 of 120; the
  spray clip 13 of 104).

- **render.body_open** (pitfall V79): the body under the old light print read through 1.5 x its widest stroke near the
  keyframe's edit, at least 15 px (`pipeline.body_open`; a blurred logo 23 px wide: 37 px), by the body floor and the
  spots check. Through 15 px the floor lifted a clean fill back to the old letters (kf_kept 0.894).
- **Review:** `kf_old` (`review.keyframe_old_left`): with the keyframe carried, the share of its area round the new
  print still lit by old print; Review Name FAILs over 0.2 (a rejected keyframe 0.346, its fix 0.082; approved 0.041,
  0.111, a printed sleeve 0.137). Review Name also FAILs `ghost` over 15 % where the label is rendered, and other
  shots Find Keyframe Frame found likely showing the product (`also_likely`, its report wired to Review Name) left as
  generated.

## 0.17.0 (2026-11-12)

The edit carry (0.16.0) locks each new line onto the generated print round it, frame by frame (pitfall V77; the
user's v16 note: "the text on the shoulder drifts a bit while the hand moves out of screen"). Only on that route:
the fixtures and the approved clips render as before.

- **Print lock** (`printalpha.print_lock`, render.print_alpha `lock_ring` 24, `lock_max` 3.0): per edited line, the
  keyframe carried onto the frame is registered onto the source in a ring round the line (Lucas-Kanade, high-passed
  luminance, 3 steps) and read there. The tracks drifted 0.2-0.6 px off the print after the hand left (the shoulder
  line crept back over 20 still frames) and the smoothed fit missed the generator's judder (+-0.9 px a frame while
  the hand moved the bottle). With the lock the shoulder lines sit on the generated lines within 0.05 px from the
  frame the bottle settles. A line whose ring holds under 5 % print (`lock_print`: a plain body, nothing to lock to)
  keeps its tracked place. `render.print_alpha.lock` false turns the lock and its check off (a route where the label
  already follows a print that slides on a plain body).
- **Check `lock`** (lookchecks.lock_pair, LOOKS lock_px 0.6, lock_frames 12): per line, the new letters' motion
  against the kept print round them, summed; flagged where it spreads over 0.6 px within 12 frames; lines whose ring
  holds under 5 % print are left out (a coloured pump's plain body read its sliding print as drift). The pack's
  Review Name calls any `lock` _REVIEW-FAIL.

## 0.16.0 (2026-11-11)

A clip on a body printed all over carries its keyframe's whole edit, on the generated print's own place, over a
surface that follows the body's outline (pitfall V76). Only where the band takes every corrected line (0.15.0's
`all_print`): the fixtures' frozen jobs and the approved clips render as before.

- **The profile surface** (`surface.Profile`, `profile_from_keyframe`): a tube whose centre and radius follow the
  keyframe matte's own outline row by row over the whole label band. The tube fitted on the base's short band and
  extrapolated up a narrower, off-centre shoulder (a cone of 76 px radius where the body is 90-130 px wide) cut the
  shoulder's line ends out of the unrolled texture.
- **The edit carry** (`render.print_alpha.ink "edit"`, `printalpha.keyframe_edit`, `keyframe_on`, `carry_keyframe`):
  the keyframe's whole edit, any colour, is the print; per frame the keyframe is sampled once through the frame's and
  the keyframe frame's mappings, under the render's low-pass light.
- **Follow the print**: on that route body_snap, glue, level, limb_snap and feather are off (they moved the label up to
  39 px off the generated print).
- **Review**: `kf_kept` (the keyframe's edit delivered on its own frame; the pack fails a clip under 0.9); `double` on
  the edit route reads the keyframe carried onto the frame as the label's own print, and only old print the output
  shows within reach of what the clip changed.
- **Flat model** (anchor.model "homography"): `fit.body_snapped` took the homography (a numpy array, whose own `.base`
  is None) for a wrapped mapping and crashed; it now leaves any non-tube mapping as it is.

## 0.15.0 (2026-11-10)

A keyframe whose corrected print lies mostly off the measured label band now renders over all of it (pitfall V74).
Unchanged for the fixtures' frozen jobs and the approved clips' keyframes (their bands measured the same).

- **The band takes every corrected line when most of them lie off it** (`BAND_OVER_PRINT["all_print"]`, 0.6):
  `band_over_print` stretched the profile's band only over corrected lines next to it; on a body printed all over (a
  spray bottle's sleeve, its copy up the shoulder and on the black base) the profile's band was the base (0.45 of the
  corrected print), and the clip carried only the base's lines, the rest left garbled on every frame, the keyframe's
  own included. Under 0.6 of the corrected print inside the stretched band, the band runs from the first corrected
  line to the last (plus `room`, within the matte's ends); the tube is still fitted on the profile's band.

## 0.14.0 (2026-11-05)

Renders with render.glue change on frames where the product moves under 1 px a frame (pitfall V71). Fixtures 8/8 at
0 px (their frozen jobs don't glue; none re-baselined).

- **The glue's smoothing runs on the product's motion clock** (`render.glue.still_px`, 1.0 px a frame): a frame
  advances the clock by the carried grid's median motion over still_px, at most one. On a still product the
  placement's smoothed offset and the keyframe's own fix (eased out over 2 sigma) hold, so the label holds: on frames
  counted, a pump bottle's label crept 0.59 px over a still run after its keyframe and into a cut (the user saw it);
  on the clock 0.03 px. `render.level` smooths on the same clock. `still_px: 0` = frames.
- **No ease where the product leaves** (V72): a held run that stops for a cut or because the product leaves the view
  holds full strength to its last frame (an ease there laid the label over the old print, doubled, before a wipe).
  Review check `fade` flags an ease on a run's last 3 frames (under 85 % of the fix on the 6 frames before) where the
  product is gone on the next frame.
- **Review check `creep`** (lookchecks): the label's shift against the still product summed over each run of still
  pairs (body under `creep_still` 0.5 px), over `creep_px` 0.3 px flags the run; pairs where the fix's strength
  changes (an ease, a fade at a run's end) don't count. The pack's Review Name calls any creep _REVIEW-FAIL.

## 0.13.1 (2026-11-02)

No render changes (the review only; fixtures 8/8 at 0 px). Pitfall V69.

- **`slide` reads the label's turn against the frame's own print** (the surface the label lies on: its tracked
  mapping), not the keyframe frame's place on the body, where no orbit was measured. A product that turns on its axis
  turns its print, and the label must turn with it: a drawn tube turning +-12.6 deg read 15 px of slide; now its
  label sits within 0.36 px of its print (90th percentile), and two products of an approved edit 0.46 and 0.52 px.
  Under a measured orbit, nothing changes (the orbit at the print's rate, V66: the rejected placement still flags).
  `print_px` (the print's turn) is measured on every frame, not only with an orbit.
  - **Gap:** a generated print sliding across a product that doesn't move, with no ground in view, is followed by the
    label and no longer flagged; on a plain round body that looks like the product spinning.

## 0.13.0 (2026-10-31)

The user's goal (2026-10-31): inputs in, the right output out, with no per-case setting (pitfall V67). The turn hold
was a setting each test case needed by hand: on for a squat product the camera orbits, off for a squeeze tube on a
push-in. Renders change only where the measured job (`refine()`: Refine (Video), `video.py --refine`) now decides
the turn on. Fixtures 8/8 at 0 px (their frozen jobs don't go through `refine()`; none re-baselined).

- **`render.body_snap.turn: "auto"`, the measured job's default** (unless the settings give `turn`; logged in
  render._defaults as `body_snap.turn`).
  - The body snap measures the print the label tracks round the body first, as with no turn (`turn_px`, label px
    at its middle, against the keyframe frame's place).
  - Within the review's `slide` limit (4 px): the label follows its print (as `false`). Past it: the scene's orbit
    is measured on the ground round the product's base; found, the label turns with it (as `true`, V65-V66); not
    found (no ground in view), it follows its print and `slide` flags the frames.
  - The decision is written into the render settings, so the workers and the review read the turn taken; the
    report's `body_snap.turn_auto` has the print's slide, the limit and why.
  - The orbit is measured only when the print slides past the limit (it took about 4 minutes on a CPU for a
    49-frame shot), not on every shot that has the turn on.
  - Evidence: a squeeze tube on a push-in (its print 2.2 px round the body: off, as its approved settings); a squat
    jar on a push-in with an orbit (32 px, the orbit measured: on, as its approved settings); the video demo, a
    bottle turning on its axis with no ground in view (15 px, no orbit: off, its output as before).
- **Removed: settings that only brought back what failed in testing** (the user, 2026-10-31: "don't keep the
  approaches that didn't work"). `refine()` rejects them with the reason (`pipeline.REMOVED`, `removed_settings`):
  `shade_print`, `shade_local`, `steady_light` or `clear_glass` false, `limb_snap.front` "mean", `anchor.over_print`
  false, and `body_snap.turn` "hold" (with its code: `true` now always follows a measured orbit, and holds the
  keyframe's place only where none is measured). The defaults are unchanged; a job that doesn't name these renders
  as before, so the fixtures and both approved test renders stand. Jobs run straight from a vjob (`video.py`) keep
  the engine's own defaults.

## 0.12.0 (2026-10-30)

The user on the one workflow's jar after v7 (pitfall V66): "it looks better now, but still is sliding a little".
Renders change only where `render.body_snap.turn: true` follows a measured orbit. Fixtures 8/8 at 0 px (none sets
the turn hold; none re-baselined).

- **The turn follows the orbit at the frame's print's rate** (`fit.orbit_gain`; `body_snap.orbit_gain`, default
  `[0.9, 1.3]`, a number pins the rate).
  - The orbit gives the turn its timing. The print gives its rate: the weighted median (weights orbit²) of the
    print's turn over the orbit's, on the frames the orbit has turned at least a quarter of its reach. The result is
    clamped to the bounds.
  - Jar: the print turned 1.12 times the measured orbit, steadily over the shot. v7's label, turned by the orbit
    alone, ran up to 3.9 px behind the frame's print at 140 (tracked on the print itself). Now it's 1.2 px at most.
  - Why the rate differs:
    - The label's model projects the body orthographically, so a turn moves its front slower than the real jar's in
      perspective. The camera is about 13 jar radii away (from the cap's and the base's ellipses), so about 8 %.
    - The ground's fit reads the turn a little low. A free translation in the fit reads 10 % more.
  - render.json's `body_snap.orbit` has `gain` and `print_rate`.
- **A product standing on the ground, seen from above, keeps the print's place along the body.** With the orbit
  followed, `auto` edges hold no gap (`none` above and below). Explicit edges still hold them.
  - The matte's top is the far rim of the lid's top face. It moves down the body as the camera comes closer: 3.6 px
    against the keyframe's proportion at 140.
  - The base's front reads through the wet ledge's reflections (±2 px).
  - The print itself fits the jar: its rise and fall follow a point between the cap seam and the base front
    (0.47 of the way; its points sit at 0.45) within 0.55 px rms.
  - Held to the matte's proportion, v7's label sat 2.4 px off the print at 140. Now it's within 0.2 px.
- **Review `slide`:** where the orbit is measured, the label's turn is judged against the orbit at the frame's own
  print's rate (the body rows' `print_px`, measured through the tracked mapping; `orbit_gain` in the geometry),
  whatever rate the render took. The limit is `slide_orbit_px` 2.0 there (`slide_px` 4.0 elsewhere).
  - v7's placement (`orbit_gain` 1.0, matte edges) flags on 140-151, off by up to 3.3 px. The fix is clean.
- **Review `body`:** the same rule for its edges, so a product standing on the ground, seen from above, is judged on
  its roll only.

## 0.11.0 (2026-10-30)

The user on the one workflow's jar after v5 (pitfall V65): "in the video camera slightly pushes in and orbits around
the jar to the right, but in the output it looks like the label is staying at the same place ignoring camera's
orbiting motion". `render.body_snap.turn: true` now holds the label to the scene's orbit; renders change only where
it is set. Fixtures 8/8 at 0 px (none sets the turn hold; none re-baselined).

- **`imgvideo.orbit`** (new): the camera's orbit about a product that stands on a surface, from the ground round its
  base.
  - Points on the ground between 1.05 and 1.9 base radii, off the product, are tracked from the anchor frame with
    the shot's tracker (AllTracker at 640 px at most).
  - They're read in the ground plane, normalised by each frame's base ellipse (`base_ellipse`: the matte's half
    width over its lowest rows, its bottom edge fitted as an ellipse arc).
  - The rotation about the base's centre is fitted per frame (2D Procrustes). A frame fitting worse than 0.05 base
    radii takes its neighbours'.
  - Skipped when the base runs off the frame, the ground is seen edge on (minor/major under 0.12), or fewer than 200
    points track.
  - Jar: 5.9° over the push-in, residual 1 % of the base radius, 3,800 points.
  - The pipeline keeps it in the work folder (`orbit.json`) and measures it whenever `body_snap.turn` is set.
- **`render.body_snap.turn`:**
  - `true`: the label point under the body's centreline is held at the keyframe frame's, turned by the orbit
    (`fit.body_corrections(orbit=)`, Ru × the orbit since the reference frame). The label turns with the camera,
    and only the print's own slide is taken out. Where no ground is measured, it falls back to the plain hold.
  - `"hold"`: the plain hold, as 0.8.0-0.10.1 did with `true`.
  - render.json's body_snap has `orbit`.
  - Jar: the generator's print turned with the orbit within 1.7 label px rms (its own slide at most 3.9 px, at the
    shot's start), so the plain hold threw away about 33 label px of real orbit.
- **render.level reads the label's own print** (`level.level(read=)`): with whole-band compositing, each frame's
  label as its mapping renders it (the keyframe warped onto the frame, no relight) is read with `level.measure`, as
  the review reads it. Frames it can't read take the mapping's read, offset to the rest. render.json's level has
  `read`.
  - The mapping's rows read the same whether the jar's label was held or turned with the orbit (0.94° vs 0.97° at
    140), yet the turned label's lines came out 2.0° against the frame's print's 0.5° (the hold's: 0.6°).
  - The keyframe's lines carry the jar's bow in the label: the surface model sags less than the jar seen from above.
    So turning the label 27 px brings a sloped stretch of each line to the axis, which the rows don't show.
  - Read from the print, the turned label's lines are within 0.25° of the frame's print (`lines` 0.05° median,
    1.9 px at most). Held, nothing changes (0.05°).
  - This was also the residual turn behind v3's `level` flags: v3 followed the tracks.
  - Print-only compositing (the approved serum) keeps the rows' read.
- **`level.mapping_level(c=)`:** the rows' read is taken where the copy row crosses the body's axis in that frame, no
  longer at the copy block's centre (a fixed label point).
- **Review `slide`:** where the render measured the orbit, the label's turn is judged against its place turned with
  the orbit (`orbit_px` in the body rows, `turn_vs_orbit` in the geometry). Otherwise it's judged against the
  keyframe's place, as before. The jar with the plain hold: flagged. Turned with the orbit: clean.

## 0.10.1 (2026-10-29)

ffmpeg 6 to 9 as well as 5.0.1 (pitfall V64). No render changes: on ffmpeg 5.0.1 the fixtures pass at 0 px (8/8),
the decode commands are byte-identical (plus one `ffmpeg -version` per process), and the review mp4 is byte-identical
(its command gains `setparams`).

- **An RGB source on ffprobe 6+:** `vio.probe` passed ffprobe's `color_space` on as the matrix, and on to x264 as
  `-colorspace`/`-color_primaries`/`-color_trc`. For an RGB source (PNG in a .mov, like the pack's video demo)
  ffprobe 5.0.1 leaves it out, so the engine fell back to bt709; ffprobe 6.1 reports "gbr", which x264 refuses
  ("Unable to parse option value"), so `video.py --refine` died writing the review mp4 (BrokenPipe). The matrix is
  now bt709 wherever ffprobe names none (`vio.NO_MATRIX`: missing, "unknown", "gbr"), the tags and the decode scaler
  5.0.1 effectively used. The range is unchanged (ffprobe reports "pc" for these sources on 5.0.1 too).
- **ffmpeg 9 removed `-vsync`:** `vio.passthrough()` (graphic-motion-designer 2.17.1's, verbatim) picks
  `-fps_mode passthrough` from 5.1 on and `-vsync 0` below, read once from `ffmpeg -version`; `vio.read` and
  imgauto's `media.read_small` send it where `-vsync 0` was (imgauto 0.4.4).
- **ffmpeg 9 wrote the review mp4's primaries and transfer as "unknown":** it takes them from the frames, not from
  `-color_primaries`/`-color_trc`, and frames from the pipe carry none (from a file input it takes the source's, e.g.
  iec61966-2-1). The Writer's filter now ends with `setparams` (colorspace, primaries, trc = the matrix), so 9 tags it
  as 5.0.1 does. On 5.0.1 and 6.1.1 the mp4 is byte-identical with or without it; on 9.0.2 only the tags change (the
  decoded pixels are the same).
- Tested on this Mac against 5.0.1 and real 6.1.1 and 9.0.2 builds (evermeet.cx, Intel under Rosetta):
  - the fixtures pass at 0 px on all three (8/8): renders are bit-identical across the three;
  - on 6.1.1 and 9.0.2, 0.10.0 fails as reported: "gbr" on 6.1.1, `Unrecognized option 'vsync'` on 9.0.2;
  - on 6.1.1, 0.10.1's probe, frames, small reads and mp4 bytes equal 5.0.1's (the demo clip and mask, example1's
    source and keyed mask); on 9.0.2 the frames and tags do;
  - the same results with stand-ins on 5.0.1 (an ffprobe reporting "gbr", an ffmpeg reporting 9.0.2 that rejects
    `-vsync`).
- Not changed: on 9.0.2, `media.read_small(gray=True)` differs from 5.0.1 (9.0.2's scaler range-expands a YUV source's gray, up to 18 levels; mask edges at 127/128 flip). It feeds only autopilot.py's analysis, never a render;
  RGB small reads are identical.

## 0.10.0 (2026-10-28)

The user on the one workflow's v4 tube (pitfall V63): "the compositing looks better now, but there's still some
flickering throughout the video. much subtle than before, but still noticable". refine() renders change with
whole-band compositing (one new default, `steady_light`); job files keep what they say, and the fixtures pass at
0 px (8/8, none re-baselined).

- **`render.steady_light`** (refine()'s default with whole-band compositing; `pipeline.STEADY_LIGHT`: `frames` 5,
  `sigma` 1.0, `step` 4): the label's light held steady over a few frames.
  - A read pass first renders every frame up to the relight and bins its log light field into label cells
    (`render.light_bin`, `step` px).
  - `pipeline.steady_light` then takes each cell against its frame's overall level, medians it over `frames`
    frames, smooths it with a Gaussian of `sigma` frames, and adds the frame's level back, so a scene's real flicker
    carries on.
  - The render relights from those cells (`render.light_apply`), each px at its label point; where a frame saw no
    cell, its own field.
  - render.json has `steady_light` and frames `steady_light_px`. In `RENDER_ONLY`. The read pass doubles the
    render's time.
  - Why: where the generator's own lettering (a different layout) covers the bare surface, its faint strokes,
    redrawn every frame, leaked into each frame's light read, so blobs 20-30 px wide blinked 2-5 levels round the
    label's text.
  - Tube (v4's node inputs): body px blinking 2+ levels over the frame's bare light, median 0.33 % a frame → 0
    (a perfect render, the keyframe warped through the same mappings with no relight: 0). Light-patch 2nd difference
    at 32 px, p99: 0.79 → 0.26 levels (the frame's own bare light: 0.53). Print sharpness, swim and level unchanged.
- **Review `shimmer` recalibrated** (`Glitter.measures`; `shimmer_levels` 3.0 → 2.0, `shimmer_frac` 0.005 → 0.002):
  - The reference is the source's at the same cells only where the source is bare there. Where its own print sits
    (either polarity, a 15 px top-hat over 12 levels, grown 3 px), the reference is the frame's bare-source p90.
  - Against the source's own value there (its print redrawn every frame blinks far more than any light), v4's
    blobs went unflagged.
  - The measure reports `ref`.
- Tried and dropped: excluding fainter print from the light read (`shade_print` contrast 0.8, grown 3 px) blinked
  far more (2.2 % of the body a frame).

## 0.9.0 (2026-10-27)

The one workflow's jar (pitfall V61; t2, shot 140-188, keyframe 180): its lines ran 0.5-0.75° more turned than the
frame's own print on 140-163 with 0.8.0's turn hold. Three measures read a squat product seen from above across its
body; each now reads it, and refine()'s placement follows. Not settings: they change renders only where a section
of body snap's search ran past its end, where the matte is squat, or where the print's lines bow. Fixtures 8/8 at 0 px
(none runs body snap or level; none re-baselined). The serum (V45) takes the old path on every frame.

- **`fit.body_measure` widens its search across the body** (×1.5, up to 4 times, from `across` × the label's length)
  while a section's matte run reaches the window's end. On the jar (body about 0.6 × its label's length each side)
  the far limb lay past the window on 140-152: body snap's roll read +0.9° where the label's axis was 0.46° off the
  body, and the turn hold's `u_body` sat off the centreline by half the cut. It feeds body snap (roll, turn) and the
  review's `body` and `slide`. Serum: its sections end 55 px or more inside the window, unchanged.
- **`level.body_axis` on a squat matte** (eigenvalue ratio under `SQUAT` 3, or wider than tall): the axis nearest the
  image vertical (the vertical itself under `ROUND` 1.3), kept by the second pass, as imgauto's triage does since
  V53. On the jar the first pass ran across it (89.7° from vertical): render.level read the rows against a side and
  sheared the label's verticals instead of turning its lines; the review's `level` read a limb as the base ("the
  keyframe's own lines run −5.57° against its base"). Serum: eigenvalue ratio 10.7-11.2, unchanged.
- **`level.measure` reads bowed lines at the axis**: lines bowing more than `BOW_PX` 6 px across the copy block (a
  parabola search, `BOW_K` 0.004) are read where each arc runs level on a round body, at the body's axis: q = (y − k
  x²) cos a − x sin a, coarse to fine. The jar's lines sag 41-52 px: read straight its keyframe's were −1.98° off
  its base, bowed −0.03° (render.level's offset "base" +0.01°, not a 2° turn of every frame). The result has
  `block_px`, the copy block's width. Serum keyframes: bow 1.4-2.4 px, the straight read (offset 0.94° as before).
  - Jar (offline: the node inputs, v3's keyframe, `body_snap.turn`): the label's rows within 0.18° of the frame's own
    print on every frame (0.8.0: 0.75°, 28 frames over 0.3°), its verticals within 0.24° of the body's centreline
    (0.43°); body roll medians ±0.13° (−0.35 to +0.30), `turn_px` within ±0.4; swim 0.34 px (source 0.77), sharp
    1.009, `double` median 93 px (75; flagged at 150), 0 px outside the matte. The review flags nothing (0.8.0:
    `level` 140-146).
- **Fills stay inside the label's area** (`render._fill_print`, pitfall V62): the turn hold's fill of the frame's print near
  the tracked letters (0.8.0) was bounded by the matte, so a glass base's rim within 10 px of the tracked last line,
  past the label's band, was filled flat (v4's jar, 152-170); where it had nothing inside the label to fill from, the
  fallback's normalized convolution had no support and wrote 0 (a black 1 px dash on the render box's last row, 152 and
  170 of v5's first render). Now the held fill takes `act` (the label's area) and the fallback keeps the frame where its
  weight is under 1e-3. Only renders with `body_snap.turn` change; the clear-glass fill was inside `act` already.
- **Review `lines`** (lookchecks; `lines_deg` 0.3, `lines_px` 3, `lines_print_deg` 0.5): the label's lines against the
  frame's own print, `level.measure` on both on one straightening, off their difference on the keyframe's frame,
  5-frame median held `level_run_s`; flagged over 0.3° and 3 px end to end across the copy block, where the base is
  steady and the frame's print holds level with it as on the keyframe's frame (a generator slanting its print, V45, is
  `level`'s). `frame_looks` adds `level_src`. Jar on 0.8.0: 141-164 (0.65°, 5.0 px); fixed: clean (0.15°); the
  approved serum v4: clean (2.1 px at most), its `level` flags 3-8 as before.

## 0.8.0 (2026-10-25)

The user's notes on the one workflow's round v3 (pitfalls V58, V59): the tube "isn't composited very well -- there's
some flickering happening" and the jar "slightly drifts during the camera motion -- it just has to be just a little
bit more glued". refine() renders change with whole-band compositing (one new default, `shade_local`); the turn
hold is a setting (`body_snap.turn`), and the clear glass's fill reaches where the tracking puts the letters. Job
files keep what they say, and the fixtures pass at 0 px (8/8, none re-baselined).

- **`render.shade_local`** (refine()'s default with whole-band compositing; `pipeline.print_gap_default`): the
  relight's median across blocks (`render._ratio_field`) takes the bare blocks round each one only (`_nan_median`),
  and the smoothing weighs each block by how much of it is bare. Before, the blocks print covered were filled with
  the frame's overall median, which near lines of text outvoted the bare blocks round them by an amount that changed
  as the generator's lines morphed: the tube's body boiled in soft patches round its text. In `RENDER_ONLY`.
  - Tube (keyframe B, offline): label-space flicker (p90 per px) median 0.76 -> 0.65 levels, 95th percentile 3.93 ->
    2.61 (source 0.85 / 6.92). TUBE_DEFAULTS
- **`render.body_snap.turn`** (a setting, off by default; `turn_max` 0.08 of the label's width): the body snap holds
  the label's turn on the body as well as its ends and axis. Not a default: a product that really turns on its axis
  turns its print with it, and the matte can't tell that from the generator's print sliding round the body. The
  pack's drawn demo tube turns ±12.6°, and with the hold on its label stayed put (±15 label px held). `fit.body_measure` adds `u_body` (the label's u under the matte's centreline at the label's middle),
  `body_offsets` `turn_px`, `body_corrections(turn=, turn_max=)` a fourth value du (5-frame median, outliers
  dropped, clamped), and `body_snapped(du=)` shifts the cylinder's u0. render.json's body_snap info and the report
  carry `turn_px`.
  - Jar (offline, v3's keyframe, `turn: true`): the print within ±1.3 px of its place on the body across 140-188 (source and v3:
    +10.5 to −3.5 px); swim median 0.32 px (source 0.77), sharp 0.995, `double` unflagged, 0 px outside the matte.
    (`level` read 13 -> 7 frames, 140-146, but on this jar it measured against a side, not the base: the lines still
    ran 0.5-0.75° off the frame's own print on 141-164; 0.9.0, V61.)
- **The frame's print near the tracked letters** (`render._clear_glass(Wk_src=)`, `render._fill_print`): where the
  render places the label off its tracked mapping (`source_H`), over clear glass the frame's print is filled near the
  label's letters at either place. Held off the generator's print by the turn, the jar's last line ended past the
  label's as a stray "GY" (140-164). And while the turn is held (`held_turn`, set per frame by the pipeline), the
  frame's print near the tracked letters is filled wherever the label lets the frame through (its fades): the jar's
  seal ring, near the limb, echoed 10-15 px right of the label's on 140-145 (`double` flagged 140-145).
  render.json frames have `held_fill_px`. Without the turn hold the fades are as before; the glass fill also reaches
  the tracked letters whenever the snaps place the label off its tracks (a px or two).
- **A body step at the label's end isn't print** (`render.end_steps`, in `print_rows`; pitfall V60): cells within
  16 rows of the label's top or bottom whose hue is off the body's (chromaticity over 0.3 from the bare body inside
  that end, in runs 15 px long and 3 rows tall), with the ink touching them. The tube's curved keyframe (Auto Draft
  bow "print") brought its band edge, higher at the sides, into the label's straight end: counted as print, it cut
  the end fade from 24 px to 1, and the label showed the keyframe's band edge as a dark line with the frame's own
  edge a few px below. Now the fade is 24 px again and the edge shows once. The flat tube keyframe and the jar: the
  same print rows as before.
- **Review `step`** (`review.end_doubled`, `step_share` 0.15, `step_run` 3): across the label's ends, the same image
  points in the output and the source (through the render mapping, which runs on past the label), columns where the
  output steps twice the same way 3-16 px apart and the source doesn't. The curved tube before the fix: 38-49 % of
  the bottom end on every frame; after it, v3's tube and the jar: 0-1 %.
- **Review `slide`** (`review.evaluate`, `slide_px` 4.0): the label's turn off its place on the body (`turn_px`,
  5-frame median) held `body_run_s`. Jar: v3's placement flagged 140-173; with the turn held, clean. It fires on a
  product that turns on its axis too, so the flag says to look first: if the product doesn't turn there,
  `body_snap.turn` holds the label. Geometry rows add `turn_px_med`.
- **Review `shimmer`** (`Glitter.measures`, `shimmer_levels` 3.0, `shimmer_block` 9, `shimmer_frac` 0.005,
  `shimmer_run` 3): per frame, the temporal 2nd difference of the label's body (off its print) in the output, read
  through its render mapping (`Glitter.add(H_out=)`), against the source's through its tracked mapping, block means;
  flagged when over 0.5 % of the body blinks 3+ levels more than the source's for 3 frames in a row. Tube without
  `shade_local`: 0.6 % of its body a frame (median); with it 0.06 %. Jar 0.06-0.12 %. Read through the tracked
  mapping instead, a held label's print moving across the body counted as blinking light (the jar flagged 1-33).
- **Review `double` in kept glass** (`review.doubled(H_src=)`): the frame's print in kept glass where the tracking
  puts the label's letters, off the label's own strokes, counts again (`glass_line_px`). Jar with the old fill:
  flagged 140-171 (226 px median); with the fix 75 px median, max 123, unflagged.

## 0.7.0 (2026-10-20)

The jar of the one workflow's round v2 (pitfalls V56, V57; the user: "fix the label shape and the clear-glass base").
refine() renders change with whole-band compositing (two new defaults); job files keep what they say, and the
fixtures pass at 0 px (8/8, none re-baselined).

- **`render.clear_glass`** (refine()'s default with whole-band compositing; `pipeline.print_gap_default`): in the
  rows the label's band was stretched over past the body's own band (V54's over-print reach, anchor.fit_band), the
  still label's cells whose hue isn't the body's are clear glass (`render.glass_map`). Hue is chromaticity, log r/g
  and log b/g, more than 0.2 (`dist`) from the body's in the 64 rows just inside the band's end; the label's print
  2 cells round is left out and takes its neighbours' call. There the frame keeps its own surface
  (`render._clear_glass`): its print within 10 px (`near`) of the label's letters is filled from the glass around it,
  and only the label's print goes on, relit as the label's. `render.json` frames have `clear_glass_px` and
  `clear_glass_fill_px`. In `RENDER_ONLY`.
  - Jar (the node's inputs, offline): the glass below the last line is 5.2 levels from the source with 0.85 of its
    edge energy (v2: 15.0 and 0.52), and the reflection bars beside "ALOE VERA"'s left end are back. Tube: no glass
    cells (pink in its stretched rows), so its glass map is None.
- **`render.limb_snap.front: "track"`** (refine()'s default with whole-band compositing; `fit.silhouette_snap`;
  "mean", the old behaviour, stays the function's default): the two sides' common offset is corrected as the
  cylinder's axis off by it with the print held, (1 − cos θ) × the offset across the row, none at the front. Their
  difference stays linear. Jar: the print at 140 −0.7 px from the frame's own (was +7.1), the tilt drift to 180
  −0.08° (was +0.36°), swim excess p90 0.09 px (was 0.27), `level` flagged on 16 frames (was 20). Tube: within
  0.4 px of before; sharp 1.036 (1.03), swim excess 0.10 (0.11), level 0.13° (0.15°), nothing flagged either way.
- **Review: the kept glass is the frame's own.** `review.doubled(glass=)` and `Ghost.exclude(glass)` leave out
  where the render keeps the frame's glass (its dark reflections beside the letters read as the source's print).
  `doubled` reports `glass_px`.
- **Review: glitter on dark print.** `Glitter(polarity=)`: on dark print a speck must also be clear of the frame's
  own dark print (a 7 px black top-hat over `levels`, dilated 3 px). On the jar the 47 flagged frames were the pink
  gaps of the generator's smaller, morphing seal beside the true one, with no sparkle there; now 0.
- **Jar, all together** (offline, the node's inputs, the v2 keyframe): flagged `level` on 140-155 only (v2:
  glitter 47 frames, level 20). `double` median 26 / max 79 px, unflagged. Sharp 0.976 (v2 0.979). Swim 0.66 px
  (source 0.66). 0 px outside the matte.

## 0.6.1 (2026-10-19)

The tube residuals 0.6.0 left (pitfall V55). No render changes (fixtures 8/8 at 0 px); the review's `double` changes.

- **Not the engine's:** the wordmark ends at the limbs and the darker right limb came from the keyframe Refine was
  given: node 83 at `alpha only`, so CQ-351's drop shadow was traced into the packshot's outline (V48). Its print was
  218 px wide where the generator's is 238-243 (the line ends 29 px short of the source's) and a black strip (40
  levels) lay on the right limb; Refine carried the strip as a bright stripe then a grey band (74-94 levels where the
  source reads 175-190). With 83 `alpha, else SAM 3` (keyframe t2-tube-B: print 242 px, limb pink) the same frames and
  masks give a pink limb (172-195 levels), no doubled line ends, no `edges` seam.
- **Review: `double` reads ink, not shading** (`review.doubled(ink=0.25)`). With the right keyframe it still flagged
  17-20 and 36-48 (median 133 px, max 219): a round body's soft limb band (the left limb at the C, the limbs past the
  last line's ends), which the top-hat takes for print and which sits next to the label's letters, so it passed the
  moved-print test. Those pieces are 0.03-0.065 deep (linear) against the label's ink at 0.38 (p75 of the output's
  top-hat on the label's print); the generator's ink reads 0.6-0.94. A px now counts within 2 px of a source px at
  least `ink` x that depth (0 = as 0.6.0; `keep` adds `ink_ref`).
  - Tube, keyframe B: 17 -> 0 flagged (median 52, max 111). Tube, the 0.6.0 run's keyframe: 23 -> 11 flagged (10-21:
    the source's wordmark ends and the strip's top, kept). Jar v1's own frames: 49 of 49 flagged at 0.2-0.4. Off it
    reproduces 0.6.0's numbers exactly. It only removes px, so 0.6.0's quiet runs stay quiet.

## 0.6.0 (2026-10-18)

The frame's own print doubling the label on dark print (pitfall V54), from the one workflow's local test: the t2 jar
(shot 140-188, a dark-print keyframe at 180) showed "soin auto-réhydratant 100H" twice away from the keyframe, and the
t2 tube (shot 91-139, a packshot keyframe at 130) a shadow of the generator's big C and its last line under the
label's. refine() renders change (three defaults below); vjob files keep what they say, and the fixtures pass at 0 px
(8/8, none re-baselined).

- **The label carries every line the keyframe corrected** (`band_over_print`, refine() only; `anchor.over_print:
  false` or `{BAND_OVER_PRINT keys}`). The measured band (the product profile's) is stretched over lines of print the
  keyframe corrected past it: its changed px on its own strokes, or on the source frame's strokes it took out (whole
  letters, `letter_min` 2.5-`letter` 48 px, in lines running `line_len` 0.25 of the body's width), a top-hat 2 x stroke
  + 1 at 0.03, per 4 px strip along the axis, a strip being print at 0.25 x the band's median strip; lines 8 px apart are
  one block. Then `room` 32 px past the last line (the end fade plus the drift), stopping 2 px before the keyframe's
  body steps 20 levels off the body under that line, within the matte less 4 px. The report's `band_info.over_print`
  has it.
  - **`anchor.fit_band`:** the tube is still fitted on the profile's band; the label (texture, outline) spans
    `anchor.band`. Fitted down to the jar's rounded foot the tube missed the silhouette by 21.6 px (1.9 on the band).
  - Jar: [-216, 247] -> [-216, 324] (the corrected "ALOE VERA" line ended at 292; the glass rim the keyframe relit
    under it, slivers and long lines, doesn't count; counting every changed px reached the jar's foot). Tube:
    [-409, 183] -> [-409, 205] (the generated last line at 191 is in; the packshot's tan base at 208 stops the room).
- **`render.feather.print_gap: px`** (refine()'s default 2 with whole-band compositing; `pipeline.print_gap_default`):
  each end's fade stops that far short of the label's own print (`render.print_rows`: coverage over 0.2 in 4-row
  strips holding 0.25 x the median printed strip), shorter than `v_px` where print runs close to the end, at least
  1 px. render.json frames have `feather_v_px` [top, bottom]: the jar's 24 / 11.
- **`render.shade_print: true | {...}`** (refine()'s default with whole-band compositing): the frame's light (the
  ratio field) is read on bare surface only, the source's print and the label's left out: px under 0.5 x a 61 px grey
  closing (light print: over 2 x an opening), grown 2 px, more than 0.15 x the matte's half-width in from its edge
  (`render.SHADE_PRINT`). The block medians outvote thin print, but the tube's generated C (30-40 px strokes, drawn
  bigger and elsewhere) won its blocks: its darkness travelled with the label as a shadow and the label's own C got
  a halo. Masked at the limbs too, the packshot keyframe's darker limb came out black; the rim keeps them in the read.
  render.json frames have `shade_print_px`. In `RENDER_ONLY` (the still label doesn't read it).
- **Defaults logged:** `feather.print_gap` and `shade_print` go into `render._defaults` with what set them.
- **Review: `double`** (`review.doubled`, a look check with an action): print in the output (a top-hat 15 of the job's
  polarity at 0.03) that the source shows there too, inside the label's area (texture extent, matte eroded 4 px), more
  than 2 px off the label's own strokes (its print coverage, and the keyframe texture's own thin marks: grain, frost,
  highlights), and explained by the label's own print moved: per 48 px tile round it, the shift within 10 px with the
  best Dice overlap of the label's print and the source's, kept at 0.4 (the dark reflections of a chrome lid or a
  glass base showing through a fade are thin marks too, but nothing of the label moved). Pieces of 3 px or more.
  Flagged at `LOOKS["double_px"]` 150 px as a 5-frame median; `review.looks` has the px per frame and per end (top,
  bottom, inside); the action (hits mostly at the ends) is `feather.print_gap` 2, else `anchor.over_print.room` 48.
  The ghost check reads only the label's inner part (it leaves the feather out) and was quiet on v1.
- **Review: `spots` on light print only.** Dark print's own letters are dark marks on the body: it flagged every frame
  of the dark-print jar and tube (V52). `review.looks` says `{"judged": false, "why"}` there.
- **Checked** (local CPU, ComfyUI 0.41.0's decode, the shots' SAM 3.1 masks re-run, 6-7 render processes):
  - **Jar, the fixes off** (`anchor.over_print: false`, `feather.print_gap: null`; renders like v1, not bit for bit:
    the masks and tracks were re-made, body-snap ranges within 0.1 px): `double` flags 49 of 49 frames. On v1's own
    frames, through that run's mappings: 426-639 px a frame (median 527), 49 of 49 flagged, nearly all at the label's
    lower end (140: 472 of 526 px; 160: 505 of 525; 188: 554 of 561), action `feather.print_gap` 2.
  - **Jar, the defaults:** "soin auto-réhydratant 100H" single on every frame, ALOE VERA the keyframe's; `double`
    0-90 px (median 14), 0 flagged; QA print match vs the keyframe 0.93 -> 0.978; 0 px outside the matte; no spots
    flag. Left flagged as before: glitter (the specks in front) and level on the first frames (the keyframe's own lines
    run 6.7° off its base: offset 0).
  - **Tube, the defaults:** the C clean (no shadow, no halo), the generated last line gone, no tan line at the band's
    end; `double` 5-frame median over 150 on 10-31 and 40: the generated wordmark and rule lines are wider than the
    packshot's and their ends show through the side fades at the limbs (look; no action). `edges`: an 8.9-level step
    at the band's new lower end. Its right limb keeps the packshot keyframe's darker shading, as before.
  - **Example rounds (light / both inks, approved frames):** example1 key4 on v7 at most 68 px a frame, shotB5 102,
    example2 shotA4 and shotC5 under 64 (frames delivered by another job aside): quiet.
  - `video_regress.py`: 8/8 PASS at 0 px.

## 0.5.0 (2026-10-17)

peach-glaze-set's v2-v4 job steps and checks in the engine, on by default for a Refine run (its `9-build/level_fix.py`,
`body_axis.py`, `check_swim.py`, `check_level.py`, `check_sharp.py`, `check_body.py`, `check_ghost.py`,
`check_spots.py`; pitfalls V42, V45). Renders from vjob files without `render.level` are unchanged (the fixtures pass
at 0 px). refine() renders of round products change: imgauto 0.4.1's vjobs switch glue and level on.

- **`render.level: true | {...}`** (`imgvideo/level.py`, V45): the label's lines level with the product, after the
  glue. level_fix.py step for step: per frame, the mapping's u direction across the copy block (`at` [0.5, 0.75] of
  the label, chord `span` 0.4 of its width) against the normal to the body's two-pass upright axis (body_axis.py);
  the target is the keyframe frame's level (median over `window` 3 frames each side) plus `offset`; the difference is
  smoothed by a robust local line (`sigma` 4); the label is sheared along the axis about the copy block's centre as a
  geometry.Residual over the glued (or placed) mapping. `_place` returns it; the motion blur's motion is read through
  it.
  - `offset: "base"` (default): minus the keyframe's own lines against its base, measured on the keyframe image
    (check_level.py's skew by point projection on the copy block, against the median base over the keyframe's frame
    and 3 each side); a number: degrees; past `offset_max` 3° it falls back to 0.
  - **Gate:** it runs only where the straightened matte's bottom edge holds steady: frame-to-frame scatter (1.4826 x
    the median distance from a 5-frame running median) under `base_mad_max` 0.5°. Serum 0.156°, squeeze tube 1.377°
    (skipped; its mapping shears ±2° as well, but nothing measures that against a base). `gate: false` forces it.
  - render.json and the report's keyframe entries have `level` (target, offset, the keyframe's lines vs its base, the
    base scatter, corrections; or `skipped` and why).
- **`render.glue.edge`** takes `[start, end]` per end of the run ("line" or "mean" each). A string is both, as before.
- **The look checks** (`imgvideo/lookchecks.py`) in the review, on the delivered frames, merged into `review.checks`
  and the flags; `review.looks` has the numbers per shot, `review.actions` the setting to try per flagged check
  (`{"check", "frames", "why", "try"}`); notes.md lists them. `review.looks: false` skips them.
  - `swim` / `ends`: check_swim.py's measure on near-still pairs (the body moving ≤ `still_px` 3, the search centred
    on the matte centroids' shift): the label's swim against the source print's on the same region. `swim`: 0.35 px
    over it on 2+ pairs within 6; `ends`: any such pair in a held run's first or last 4 frames; the action flips that
    end's glue edge.
  - `level`: check_level.py's measure per frame (the copy's skew by point projection, 0.25 px bins smoothed 1 px,
    against the straightened base). Judged where the base scatters under 0.5°, frames whose base jumps 1.5° off its
    15-frame running median left out; over 1° off the target (0 with `level`'s offset "base", else the keyframe
    frame's level), a 5-frame median held 0.25 s. `looks.level.vs_keyframe` has check_level.py's own numbers.
  - `sharp`: check_sharp.py's edge strength, output and source, each against its keyframe frame's: under 0.7 and 0.1
    under the source's.
  - `tilt`: check_body.py's measure with `--ref-window 3`; flagged over 1° or 4 px (5-frame median, 0.25 s, crisp
    frames only): read off a print-only composite it scatters to p90 0.65° on an approved render.
  - `ghost`: check_ghost.py's measure on sparkle re-add frames (render.sparkles a frame list, `alpha: "label"`): the
    render against one without the re-add, in `_render_one` (render.json `ghost_px`); over 60 px.
  - `spots`: check_spots.py calibrated: 4 px clear of the output's print (the job script took letter fringes, 2 px
    out, for spots on 125-127 frames), the body a 15 px opening of the source (7 px kept a garbled heart and read its
    removal as a spot), 4-60 px pieces.
- **Defaults are logged:** `print_defaults` and refine() record what they switch on that the settings don't name in
  `render._defaults`; the render prints a `defaults` line, render.json and the report's keyframe entries have
  `defaults` ({key: {"value", "by"}}).
- `RENDER_ONLY` has `level` and `_defaults` (the still label doesn't depend on them).
- **Checked** (peach-glaze-set, local CPU, 6 render processes, about 8.5 min per product):
  - **Port, render.level:** on the serum's glued fit (fit_stab4) with `offset` -0.94, the levelled mappings equal
    level_fix.py's fit_level2 to 2e-13 px; rendered with alpha print (presence, motion blur and soft cap at their
    defaults), the engine gives `serum_local/run12/serum_v4_lossless.mov` **bit for bit** (0 px differ). With the
    default `offset: "base"` (it measures the keyframe's fall as 0.92° where v4 passed 0.94): 249,852 px differ by 1
    level, 8,677 by 2, 199 by 3-31 (motion-blurred frames).
  - **Port, tube:** v2's glued fit (fit_stab3) with the defaults: `level` skips it (base scatter 1.377°), and the
    engine gives `tube_local/run8/tube_v2_lossless.mov` **bit for bit**.
  - **The look checks on the delivered files** give the job scripts' numbers: serum v4 sharp 1.393 / p10 1.021, level
    keyframe 0.54°, vs the keyframe -0.73 / 0.8 / p90 2.0 (check_level_v4), tilt 0.35 / p90 0.65 (check_body
    --ref-window 3); tube v2 sharp 1.039 / 0.748 and the same 10 frames under 0.7 (none flagged: the source is soft
    there). Flags: serum v4 `level` 3-8 (the run's start, under-corrected by σ 4); tube v2 `ends` 138-139 (0.43 and
    0.62 px while the body moved 0.03).
  - **Engine only, the defaults** (the tracked fit; settings: bands, shots, words, and for the tube its body-snap
    edges and `presence_target` 1.25), then `review.actions` applied once:
    - serum: the review flagged `ends` 0-1 (0.61 px, the source 0.0) and asked for `glue.edge ["mean", "line"]`;
      re-run with it, 0.29 px (v4 0.29). Swim median on near-still pairs 0.15 (v4 0.14, the source 0.14); level vs
      the keyframe -0.74 / 0.81 / 2.0, keyframe 0.54 (v4 the same to 0.01); sharp 1.398 / 1.017 (v4 1.393 / 1.021);
      tilt 0.37 / 0.68 (v4 0.35 / 0.65); no spots; `level` 3-8 left flagged as in v4.
    - tube: swim 0.22 (v2 0.23, the source 0.20); sharp 1.033 / 0.752 (v2 1.039 / 0.748), the same frames under the
      floor; tilt 0.14 / 0.36 (v2 0.14 / 0.44); level not judged (v2 neither); `ends` 139 (0.80 px; v2 0.62) asked
      for `["line", "mean"]`: re-run with it, the last pairs 136-139 read 0.07 / 0.09 / 0.09 / 0.18 px (were 0.12 / 0.22 / 0.37 / 0.80) and the start stays 0.16, swim median 0.20 = the source's; `swim` 55-56 (0.72 and 0.64 px, the source 0.25 and 0.21; v2 0.56 and 0.51, just under the 0.35 px excess) asked for `glue.sigma` 6: re-run with it, no flag left (swim median 0.21, the source 0.20; ends 0.06-0.23 px; sharp 1.025 / 0.764; tilt 0.17 / 0.46). Two rounds of actions, no hand settings.
    - Pixels are not v4's or v2's: the engine's body snap places the label by its own measure, not body_snap.py's.
  - The fixtures pass at 0 px (8/8, the patch fixture with its re-add frames' ghost renders included). The pack's
    `tools/regress_video.py` (the node against `video.py --refine` on the demo, where `level` runs: its synthetic
    base scatters 0.0°) passes: 0 px, matte, changed and reports equal.

## 0.4.3 (2026-10-17)

peach-glaze-set v2's job-level steps as render settings (its `9-build/ink_alpha.py` v2 and `stabilize.py`; pitfalls
V42-V44). Renders without `alpha: "print"` or `glue` are unchanged (the fixtures pass at 0 px).
`alpha: "print"` renders change: the carry fix below, and presence and motion blur are on by default with it.

- **Carry fix** (`printalpha.transfer`): the ink's colour is unmixed from the render's pixel before it goes under the
  soft matte, as ink = clip(rb + (rl − rb) / max(c, 0.05)) × scale. The old version used the render's pixel itself,
  so a thin stroke's partly covered edge pixels were weakened twice: the tube's copy lifted 0.7× the render's.
  0.4.2's outputs (and v1's carry outputs) no longer reproduce bit for bit; v2 is the reference now.
- **`print_alpha.presence: "match"`** (default; V43): the print's presence is its mean top-hat-15 lift in display
  levels, over the true print dilated 4 px and the matte eroded 4 px.
  - It's measured as source over render on every frame, in a render pass of its own.
  - Then a running median over 9 frames and a Gaussian over 2 frames, times `presence_target` (1.0; 1.25 for thin
    true copy against a bold garbled print), clamped to [1, `max_gain` 2.5].
  - It's applied as ink = body + gain × (ink − body). `"off"`: none.
- **Motion blur** (`print_alpha.shutter` 0.5, default; V44): a line blur of the soft matte and of ink × matte along
  the label's own motion.
  - The motion is the centred difference of the label's grid points: glued, or as placed.
  - Length = min(shutter × speed, `max_blur` 3 px), only from `min_speed` 8 px/frame.
  - It's no softer than the source's own print on that frame: × clip((0.9 − sharpness) / 0.25, 0, 1), where
    sharpness is `printalpha.edge_strength` (check_sharp.py's measure) against the keyframe frame's. Speed alone
    over-blurs generated video, which jumps 85 px between crisp frames.
  - The alpha grows by ceil(L / 2) px there, and lost print isn't measured on blurred frames. `shutter: 0`: none.
- **`render.glue: true | {...}`** (`imgvideo/glue.py`, V42): stabilize.py step for step. The placement (body snap,
  limb snap) is carried frame to frame along an affine Lucas-Kanade fit of the source's own pixels over the label's
  area. It falls back to the placement's own step where rms > 0.9 or the median deviation is over 6 px. The
  placement's offset from the carry is added back smoothed: a zero-lag, Huber-weighted local line over `sigma` 4
  frames, or `edge: "mean"` near the run's ends. The result is a geometry.Residual over the placement; the keyframe's
  frame stays exact. render.json and the report's keyframe entries have `glue`: fallbacks, jitter and corrections.
- `_render_one(..., extras)`: the per-frame presence gain and blur. The render workers take them with each frame.
- The node: a `glue` input (appended).
- **Checked:**
  - Port against the job scripts on the same inputs: `printalpha` = ink_alpha.py v2 (presence, carry with gain and
    blur, the blur rule) bit for bit on tube frames 4, 44 and 104. `glue` = stabilize.py on run5's placement: the
    glued mappings 0.0 px apart; the summary, the per-frame report and motion.json are equal.
  - The engine reproduces v2's tube (`tube_local/run8/tube_v2_lossless.mov`, the master's tube) with v1's fit.
    Settings: run5's + `{"alpha": "print", "glue": true, "body_snap": false, "print_alpha": {"presence_target":
    1.25}}`. 40 frames differ, 1,305 px, all but one by 1-2 levels (max 3). v2's glue fit (fit_stab3) sits 0.0075 px
    from the engine's. The presence gains (1.054-1.435) and the blur frames match.
  - **Engine only**, as a Swordfish Refine runs it: the tracked fit, `render.body_snap`, glue, alpha print and its
    defaults (serum: glue `{"edge": "mean"}`; tube: `presence_target` 1.25, body_snap matte/dark). No job-level
    step. Both match v2 on the run's checks:
    - swim median against the body: serum 0.39 px (v2 0.48, the source 0.42), tube 0.40 (v2 0.41, the source
      0.38);
    - sharpness median: serum 1.384 (v2 1.231), tube 1.033 (v2 1.039), with the same intentionally blurred tube
      frames under the floor;
    - 0 px outside the matte; 5-6 min per product on 6 CPU processes.
  - The fixtures pass at 0 px (8/8).
- Note: since 2026-10-17 imgauto's vjobs (refine()'s measured job) default to `body_snap: true`, `soft_cap` 0.3,
  `smear: false` and `alpha: "print"` for light print. `refine()` runs without settings change with it; vjob files
  don't.

## 0.4.2 (2026-10-17)

New render options, all opt-in: renders without them are unchanged (the fixtures pass at 0 px). They move
peach-glaze-set v1's job-level compositing (`9-build/ink_alpha.py`, the optics edit in `run_snapped.py`; pitfall V40)
into the render, so a Refine run needs no local post-step.

- **`render.alpha: "print"`** (`imgvideo/printalpha.py`; the default `"label"` is the relit band, as before): only the
  print changes, carried onto the source's own body, so no patch outline can show (the user's "visible square of the
  label composited"). Per frame, after the texture render, inside its changed pixels:
  - alpha = blur(dilate(new print ∪ the source's garbled print within 14 px of it, 3 px), σ 1.5). A garbled stroke
    counts only where the source is brighter than the render and the render is at least the source's body less
    6 levels there (a sparkle over the render's dark limb line became a dark dot).
  - The new print (`print_alpha.ink: "keyframe"`, the default) is the keyframe's own ink, read once on the keyframe's
    frame through the mapping the render places there: the keyframe 4+ levels over its source frame, a top-hat-15
    stroke 6+ px inside the matte, in components whose 90th-percentile gain is 8+ levels, dilated 2 px. It is carried
    onto each frame through that frame's mapping. The render's top-hat misses faint ink (a pale copy in shade, a
    frame the generator drew blank: the tube's 4). `"render"`: the render's top-hat 15 instead, 3+ px inside the
    change and 6+ levels over the source (peach-glaze-set's serum in v1). A re-acquired shot (`texture_job`) has no
    keyframe frame of its own, so it falls back to `"render"` and says so.
  - target (`carry: true`) = min(source, open13(source) × 1.03) under a soft matte of the render's ink, in the
    render's ink colour scaled by lum(source body) / lum(render body), in linear light: never the render's body.
  - out = source × (1 − alpha) + target × alpha.
  - Light print on a body only: other polarities raise an error.
  - Knobs: `render.print_alpha` (`printalpha.DEFAULTS`).
  - The report: render.json and each keyframe's `print_alpha` in `refine()`'s report give the ink, the keyframe ink
    px, the frames, the share of the render's change kept (median) and `lost_print_frames` (keyframe ink only): print
    the render had that the output lacks by over 30 px, as gains over each image's own body. Per frame it's in
    render.json's `frames[t].print_alpha`. `lost_print_share_median` is the shot's median share of the ink. Bright
    print near white reads lower in contrast on every frame once carried onto a brighter body: on peach-glaze-set's
    serum about half of the ink px on every frame, v1 included, with every letter there. Read a frame against that
    median. On faint print (the tube) it stays at 1 % or less.
- **With `alpha: "print"`, unless the settings set them:**
  - `soft_cap` 0.3;
  - `smear: false`;
  - `feather.v_px` 4: the 24 px end fade dimmed the tube's top N into the garbled print under it. `refine()` replaces
    the measured job's 24 unless the settings give `v_px`. A vjob file keeps the v_px it writes;
  - the sparkle re-add is off, always. It ghosts the garbled print where that morphs away from its still median
    (V37), and the source's glitter stays anyway, everywhere but on the letters. A `sparkles` setting is logged and
    ignored. The review's glitter check then names the frames but no longer suggests `render.sparkles`
    (`sparkles_frames` is left out).
- **`render.soft_cap: px`**: the still stage's per-frame softness (optics.json `soft`) is capped at this when the
  render reads it. optics.json keeps the measured values. QA's softness note reads the capped ones.
- **`render.smear: "auto" | false | [clip frames]`**: the still stage's motion smear as measured (the default), none,
  or only on those frames (motion blur the source clearly shows). A false smear happens: 26 px on peach-glaze-set's
  sharp tube, 110-111.
- `_render_one`'s placement (axial shift, body snap, limb snap) is `_place()` now, shared with the print setup.
  Renders unchanged.
- `refine()`'s still-label cache key leaves out `alpha`, `print_alpha`, `soft_cap` and `smear` too. A `feather`
  change still recomputes the still label, so switching to `"print"` without a `v_px` recomputes it once.
- **Checked on peach-glaze-set v1** (`video.py --refine` on copies of the jobs' work folders, with v1's
  body-snapped fit; optics.json as measured, no edits):
  - tube run5 settings + `{"alpha": "print"}` reproduce `tube_local/run5/tube_carry_lossless.mov` exactly, all 198
    frames (0 px, the same changed mask). The lost-print report is the job script's too: 40 frames over 30 px, at
    most 68 px.
  - serum run4 settings + `{"alpha": "print", "print_alpha": {"ink": "render"}, "feather": {"v_px": 24}}` reproduce
    `serum_local/run4/serum_carry_lossless.mov` exactly, 0 px.
  - The serum with the default ink (the keyframe's) differs from v1 by 1-2 levels on most changed px, about 250 px a
    frame over 2 levels, and looks the same at 3x.
  - **Engine only, as a Swordfish Refine would run it:** the tracked fit, `render.body_snap` (the tube:
    `{"top": "matte", "bottom": "dark"}`), `alpha: "print"` and its defaults, with no job-level step at all. The
    project's own checks match v1:
    - sharpness (`check_sharp.py`): serum median 1.576 against v1's 1.587, tube 1.115 against 1.121; no held frame
      is under the 0.7 floor;
    - body (`check_body.py --ref-window 3`, serum): tilt median 0.33° against v1's 0.44°, side and end gaps the same;
    - 0 px outside the matte;
    - the render took 27-34 s on 6 processes.
  - The fixtures pass at 0 px (8/8).

## 0.4.1 (2026-10-17)

New options and a new stage, all opt-in. No render changes: the fixtures pass at 0 px. These are the autopilot's
requests R1, R2 and R4. R3 (flat-artwork re-acquisition) is held back: the user's rule is no flat artwork.

- **`anchor.pitch_sag_px: "auto"`** (R1, V28): a re-acquired shot's pitch is measured instead of set by hand.
  1. The source's text lines bow is measured strip by strip on the anchor frame (`surface.line_bow`).
  2. The same is measured on the label's print alone through the pitched mapping (`surface.texture_sampler`).
  3. The sag is corrected by secant steps until the two agree within 1 px; it is clamped to half the label's radius
     (garbled print can bow any way).
  4. `pitch_rows: [v0, v1]` picks the label rows; the default is the texture's densest lines of print
     (`surface.text_rows`). The result is cached in `work/pitch.json`.
  On example2's last shot it found 41.9 px (40 by hand in round 5); the label's bow is then 29.1 px against the
  source's 29.2. `pipeline.pitched()` and `pipeline.auto_pitch()` are module functions now (Job._pitched calls them).
- **`qa/fit.json` → `stops`** (R2): why the held run ends at each side, every failing criterion and its values listed.
  - `shot edge`;
  - `cut`: the tracks jump (over 5x the run's usual step and 20 px) or their visibility halves, *and* the frame's
    colour histogram jumps (over 4x its usual change and 0.2);
  - `product left`: too few tracks hold;
  - `morph`: the rigid model misses by more than `max_rigid`;
  - `misfit`;
  - `fast motion`: a track jump with no picture change.

  On example1's keyframe shot it reads 75 cut, 105 morph (the generator redraws the tube: re-acquire, don't chain).
  Shot B ends at 33 on misfit, and example2's drop runs to its shot edges. `fit.pkl` carries the same under `stops`.
- **`--stage stitch`** (R4, `imgvideo/stitch.py`): several refined keyframes on one unrolled label, for other shots'
  `anchor.texture_job`. The base keyframe job's grid is widened to the tube's whole circumference (u0 and the label
  outline in the written `anchor.json` say where).
  - Each further keyframe is re-acquired on it, refined by a shift search on the overlap (±12 px), given its own
    pitch (auto), and read on where it faces within `theta_lim` and inside its matte.
  - Its print must agree on the overlap with what is stitched (band-passed correlation at least `agree_min` 0.6), or
    it is left out and reported. Under `found_min` 0.4 it is reported as not found.
  - Each label pixel comes from the keyframe that sees it best (viewing angle times print sharpness), feathered over
    about 4 px where that changes: averaging two prints a few px apart ghosts them (V41).
  - Shading is estimated on bare label only, and the base's light is put back. Where the base wins, the texture is
    the base's own (example1: median 0, 95th percentile 2 levels).
  - Writes `work/label_texture.png`, `label_valid.png`, `anchor.json`, `qa/stitch.json` and
    `qa/stitch_coverage.png`.
  - Tested on example1: two refined frames stitched (agreement 0.84 and 0.63, the second softened); a garbled source
    frame and box-occluded frames reported as not found. A shot re-acquired on the result renders without ghosting.

## 0.4.0 (2026-10-16)

New render options, all opt-in: renders without them are unchanged (the fixtures pass at 0 px). They turn the two
fixes peach-glaze-set needed at job level (pitfalls V37, V39) into settings the Refine node can set, and two new review
checks say when to set them, so an agent can decide from the report alone.

- **`render.body_snap`** (V39): the label placed by the product's body, not by the generated print it tracks. Per
  frame, on the mapping the render would place (tracked, limb-snapped):
  - the body along the label's axis: the edge above the label (a collar seam or a cap: the strongest luminance rise or
    fall, or the matte's end) and the body's end below it;
  - the body's centreline from the matte limbs' midpoints across the label.

  Tube v -> p + f v puts the label's top and bottom at the keyframe frame's proportion of that body, and a turn about
  the label's middle puts its print at the keyframe's angle to the centreline. The roll is measured on the mapping
  under the limb snap, which orients the text lines; the snap only shifts rows sideways. Each series is cleared of
  outliers, median-filtered over 5 frames and clamped (stretch 0.9-1.1, roll 4°, shift 15 % of the label). The tracked
  mapping stays the light and sparkle source (`source_H`, V19), and QA and the review still read the source through it.
  The limb snap runs after. `true`, or a dict with the keys in `fit.BODY_SNAP`:
  - `top` and `bottom`: `auto` (the matte's end within reach, else the stronger rise or fall, decided once on the
    reference frame), `matte`, `rise`, `fall`, `edge`, `dark` or `light` (the darkest or brightest line: a cap seam),
    or `none`;
  - `reach`, `inside`, `across`, `median`, `outlier_k`, `stretch`, `roll_max`, `shift_max`;
  - `ref_frame` (default: the keyframe's frame);
  - `frames` (apply only there).

  Each frame's correction is in `render.json` (`frames[t].body_snap`), with a summary (`body_snap`: edges, the
  reference gaps, the offsets before, the stretch and roll ranges), and `refine()`'s report carries the summary per
  keyframe. Set up the way `body_snap.py` (peach-glaze-set's job-level prototype) was, it reproduces that script's
  placement to 0.002 px on all 139 frames. The prototype, though, turned the label the wrong way (its run2 doubled the
  roll: frame 68 went from 0.61° to 0.97°). It also read roll on the snapped maps, where a 1° turn of the print shows
  as only 0.15-0.25°. The engine fixes both.
- **`render.sparkles: [frames]`** (V37): the re-add only on those frames (clip frames, or half-open runs `[a, b]`),
  with `spark_glow: true` on the same frames. Glitter across a drop is now one run, not a patch job plus a merge.
  `spark_glow` lists take runs too.
- **Review checks** (`review.py`), in the report's `review`:
  - `body`: the label's top or bottom edge off the keyframe's proportion of the body by more than 1 px (or 0.55 % of
    the label's length, if more), or its print more than 0.5° off the body's centreline. Each is a 5-frame median,
    held 0.25 s. Tubes only. Each frame's edges are searched only round the gap the reference's proportion predicts
    (± half that gap + 2 % of the label), so a frame can't jump to another edge or to a split in the matte. On the
    serum's default render it flags 0-121 (the label's top 2.4 px and bottom 5.6 px off, medians), which is what the
    user saw.
  - `glitter`: small bright specks in front of the label in the source that the output lost (at least 2 on a frame,
    and half of its specks). It works in label space, with source and output read through the source's tracked
    mapping. A speck is a 5 px top-hat above 25 levels, at most 6 px across, 12 levels brighter than the median of
    the 2 frames each side at that label point (light moving across the label, not print), and 3 px or more off the
    truth's print (a morphing garbled print reads as specks: frame 123). The product's surroundings must show specks
    too (2 or more in a 4-40 px ring outside the matte), no more than 3 times sparser than the label's lost ones:
    glitter in front of the scene does, a shimmering surface doesn't (example1's metallic tube twinkles from frame to
    frame). The review summary's `sparkles_frames` gives
    the runs padded by 2 frames and merged across gaps up to 4, ready for `render.sparkles`. On the serum without the
    re-add it flags 69-70 (the sparkle trail) and 102-103, 106-108, 113 (the glitter on the drop); with
    `sparkles: [[98, 119]]` only 113 stays.
- **`refine()`'s report** has `geometry`: per refined frame, the label against the body. It gives the gaps to the edge
  above and to the body's end (px), the offsets from the keyframe's proportion (px, raw and as the check reads them),
  the print's roll against the body's centreline (deg) and the stretch. The agent no longer needs `render_maps.pkl`.
- **On the serum, as an agent would run it:**
  - Pass 1 (defaults): the report flags `body` on 0-121 (the label's top 2.4 px and bottom 5.6 px off, medians: what
    the user saw) and `glitter` on 69-70, 103, 106-108 and 113-114, with `sparkles_frames` [[67, 73], [101, 117]].
  - Pass 2 (those settings: `body_snap: true`, `sparkles` = `sparkles_frames`, `spark_glow: true`; render only):
    no `body` flag (5-frame-median gaps at most 0.44 / 0.81 px, roll 0.32°) and `glitter` left on 69 and 113 only (2
    specks each, below the re-add's own `spark_thresh`). Flagged frames went from 121 to 9; the rest are the
    existing shear flags.
- `fit.silhouette_snap` tags the mapping it returns (`limb_snap`), so `fit.print_mapping` finds the mapping under it.
  `render_maps.pkl` from earlier versions lacks the tag, so the body check reads roll on the snapped map there.
- `refine()`'s still-label cache key leaves out `body_snap`, `sparkles` and `spark_glow`: the still label doesn't read
  them, so changing them re-renders only. Caches from 0.3.x recompute the still label once.

## 0.3.1 (2026-10-16)

Renders unchanged (the fixtures pass at 0 px).

- **`refine()`'s report** splits what is left in two:
  - `not_covered`: frames left as they are;
  - `partly_covered`: frames refined, but part of the label keeps the source's pixels because it turned into view past what the keyframe saw.

  The summary line says the same. Before, both were under `not_covered`, which read as a contradiction next to "40 of 40 frames refined".

## 0.3.0 (2026-10-16)

Renders from a stored fit are bit-identical to 0.2.0 (the fixtures pass at 0 px). New tracks differ: the default
tracker is now AllTracker.

- **AllTracker replaces CoTracker3** (`track.backend`, default `"alltracker"`). CoTracker3's code and weights are
  CC BY-NC 4.0 (non-commercial only); AllTracker's are MIT. The model code is vendored in `imgvideo/alltracker/`
  (source commit e755313, two edits listed in its `__init__.py`). Its weights (`alltracker.pth`, 66 MB) live outside
  the skill: `$PLR_ALLTRACKER`, else `~/Documents/models/trackers/alltracker/alltracker.pth`. It runs on CPU or CUDA
  (not MPS). In a bake-off on the six delivered shots with their delivered settings it held the same frames, fitted
  3-5x tighter (reprojection 0.08-0.59 px median vs 0.38-1.58), matched CoTracker's print match and lock, and produced
  fewer shear spikes (example2 shot A: 3 shear flags vs 12). `"cotracker"` stays only to reproduce old jobs.
- **`imgvideo/pipeline.py`**: video.py's job and stages, moved unchanged in logic, now run on files (the CLI) or on
  arrays in memory (`Job({"vjob": ..., "dir": ...}, media=ArrayMedia(frames, masks, fps), keyframes={...})`: the
  refined frames go into `Job.sink`, no video is written). Checked stage by stage against 0.2.0 on two fixture jobs
  (fit, still, optics, render, render maps, qa identical; the in-memory render equal to the file render).
- **`refine()` and `video.py --refine`**: the ComfyUI node's contract. Clip, product matte and corrected keyframe(s) in;
  the refined clip (the input outside the changed pixels), the matte, the changed pixels, a report and the QA plot
  out. The decisions a person made between runs come from `imgauto` (the autopilot's stable API, read-only here):
  shots and cuts, the band (`verify.body_band`: within 23 px of the hand bands; the edit-region band was 437-585 px
  off), print polarity and stroke, the vjob defaults, QA's OCR boxes when the label's copy is given (`verify.text_boxes`), and for what is left `coverage.plan` (imgauto 0.2.0: chain, re-acquire or a new keyframe, with the reason and the frame to correct). Each keyframe is
  matched to its frame by `keyframe.find_keyframe`. Frames the tracks hold with an empty matte get the body's colour
  key (`matte_patch`). Shots merge in keyframe order; the review checks run on the merged clip. Track, fit and still
  are cached per shot by their inputs, so a render-only change re-renders in place.
- The QA plot is skipped, with a note, where matplotlib isn't installed (a ComfyUI server); QA itself still runs.

## 0.2.0 (2026-10-16)

Renders are bit-identical to 0.1.0 for every delivered job (the fixtures).

- **Review checks** (`review.py`), the failures found by eye in review rounds 2-5, flagged per frame:
  - `edges`: a seam at the label's sides, under the cap or crimp, or at the base; or the label's limb inside the body;
  - `ghost`: the old print showing through the new label;
  - `shear`: a shear spike against the shot's neighbouring frames;
  - `vanish`: the fix stops while the product and its print are in view;
  - `switch`: a hard switch at a fix's end, or a jump where two shots meet without a cut;
  - `bend`: the label's side snapped onto an occluder's edge instead of the product's limb (the print bends along it), from round 6.

  They run in `--stage qa` (on a lossless deliverable), in `--stage merge`, and in the new `--stage review` (existing deliverables, no render). They write `checks.json`, mark the qa.json rows and a new panel of the QA plot, and write an "Automatic checks" section into the round's `notes.md`. Thresholds sit in `review.DEFAULTS`; a vjob's `review` block overrides them. They never change a render.
- **Parallel render:** `--workers N` (or `PLR_WORKERS`; default: the cores less two, at most 8). Frames render in spawned processes from the same inputs and seeds, so the output is bit-identical to a single process (checked on the fixtures). 4-4.5x faster with 8 workers on 16-17-frame ranges.
- `stage_render`'s per-frame body is now `video._render_one`, with its set-up in `video._render_context`, so workers (and a port) call the same code.
- `scripts/video_regress.py` and the fixtures: one range per shot type, compared bit for bit with the delivered lossless rounds. They run from frozen copies of their jobs' inputs (`{project}/9-build/fixtures/`, `tracks.npz` included, `--freeze` to re-baseline), so a tracker swap re-baselines only the tracks and fit.

## 0.1.0 (2026-10-15 to 2026-10-16)

The version string stayed 0.1.0 while the engine grew through review rounds 1-6 of the two example clips. Everything after round 1 is opt-in, so a job renders as it did when it was delivered.

- **Round 1:**
  - tracks on the print (CoTracker3, `track.on`);
  - a tube under an affine camera, with the stills engine's Residual mesh;
  - keyframe unroll;
  - texture-mode relight and still label, per-frame softness and smear;
  - QA (V1-V9).
- **Round 2:** re-acquisition across cuts (`anchor.texture_job` + `reacquire`), robust relight (V7), easing (`render.ease`, V10), `--stage merge` (V14).
- **Round 3:** chaining without a cut (`anchor.chain`, V15), `anchor.roll_from_print` (V16).
- **Round 4:**
  - morph frames (`fit.shear_max`, `morph_source`, V18-V19);
  - `render.sparkles`, `spark_glow`, `occluder_feather` (V22), `feather` (V21);
  - silhouette geometry (`anchor.scales`, `texture_rotate_deg`, V20);
  - patch jobs (`render.frames`, V23);
  - `render.limb_snap` with `axial` (V24), `fit.morph_skip` / `morph_rigid` (V25).
- **Round 5:** `anchor.pitch_sag_px` (V28), `limb_snap.hold` (V30).
