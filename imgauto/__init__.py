"""Autopilot for the video label fix: the decisions a person made between video.py runs (shots, trims, crops,
keyframes, which anchor each shot gets), made from measurements, each with the evidence it rests on.

It reads the engine (imgvideo, imgpost) and never changes it: every fix still goes through video.py and run.py.
Owned by the autopilot session; the ComfyUI port calls it and vendors it. The stable API (signatures kept across
versions; references/video/autopilot.md lists them with their units):
  shots.analyse(F, M, k=2, cfg=DEFAULTS) -> (stats, bounds, shots, kinds, why)
  shots.classify(bound, cfg=DEFAULTS) -> "continuous" | "morph" | "snap" | "cut" | "uncertain" | "absent"
  keypick.pick(stats, bounds, trims, n_frames=None, angle=None, top=12) -> (best, ranking)
  keypick.suggest(stats, bounds, runs, n_frames=None, angle=None) -> [suggestion per run]
  refangle.objects(path, colours=None, scale=1) -> [(mask, gray)]
  refangle.angle_scores(F, M, frames, refs) -> {t: {"shape", "layout", "angle", "ref"}}
  verify.body_band(ref, mask, changed=None) -> ([t0, t1], info)   the automatic anchor.band (scaled to the
                                                          product's length; info.changed_in_band)
  verify.find_frame(F, M, ref) -> (frame, diff, runner_up)
  verify.read_label(img, mask, band) -> (words, tube info);  verify.spelling(copy, words);  verify.verdict(spelling)
  verify.print_polarity(ref, mask, changed);  verify.stroke_width(img, mask)
  verify.text_boxes(ref, mask, band, copy) -> video.py qa.text_boxes;  verify.ocr_boxes(gray)
  refangle.product_object(path, target, colours=None, scale=1) -> (mask, gray, iou);  refangle.shape_iou(a, b)
  keyfix.crop_geometry / crop / uncrop / quad / coverage;  keyfix.check(attempt, source, mask, copy);  keyfix.Ladder
  keyfix.paper_gate(rgb, label_mask) -> (white box or None, info)   paper label, or print on the body
  labeltext.crops(path, out_dir, stem, colours=None, target=None) -> [paths];  labeltext.tesseract_words(paths)
  coverage.job_coverage(job_dir, offset);  coverage.job_stops(job_dir, offset);
  coverage.plan(shot, kinds, held, unseen, stats, bounds, M, n_frames, angle=None, stops=None)
  vjobs.reacquire_anchor(texture_job, frame, pitch="auto")
  cropbox.box / shot_window / crop / paste;  composite.depth_order / keep_masks
Depends on numpy, scipy, Pillow and imgvideo (vio, qa, surface, keyframe.product_axis); tesseract for OCR.
0.2.0 adds keyfix, labeltext, coverage, the text boxes and the product object; 0.1.0's signatures are unchanged.
0.3.0: body_band scales its strips to products under 1,200 px long, its steps persistent (V36), and takes the keyframe's changed pixels;
keyfix.paper_gate needs a white area of a tenth of the label (a white letter isn't paper). Bands of longer products
are unchanged.0.4.0: imgvideo 0.4.1's R1/R2/R4 wired in: coverage.job_stops and plan(stops=) (the engine's stop reasons), and
vjobs.reacquire_anchor(pitch="auto"); additions only.
0.4.1: vjobs' tube jobs also switch on imgvideo 0.5.0's render.glue and render.level (a refine() render changes; the
signatures don't).
"""
__version__ = "0.4.1"
