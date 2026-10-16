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
  verify.body_band(ref, mask) -> ([t0, t1], info)            the automatic anchor.band
  verify.find_frame(F, M, ref) -> (frame, diff, runner_up)
  verify.read_label(img, mask, band) -> (words, tube info);  verify.spelling(copy, words);  verify.verdict(spelling)
  verify.print_polarity(ref, mask, changed);  verify.stroke_width(img, mask)
  verify.text_boxes(ref, mask, band, copy) -> video.py qa.text_boxes;  verify.ocr_boxes(gray)
  refangle.product_object(path, target, colours=None, scale=1) -> (mask, gray, iou);  refangle.shape_iou(a, b)
  keyfix.crop_geometry / crop / uncrop / quad / coverage;  keyfix.check(attempt, source, mask, copy);  keyfix.Ladder
  labeltext.crops(path, out_dir, stem, colours=None, target=None) -> [paths];  labeltext.tesseract_words(paths)
  coverage.job_coverage(job_dir, offset);  coverage.plan(shot, kinds, held, unseen, stats, bounds, M, n_frames)
  cropbox.box / shot_window / crop / paste;  composite.depth_order / keep_masks
Depends on numpy, scipy, Pillow and imgvideo (vio, qa, surface, keyframe.product_axis); tesseract for OCR.
0.2.0 adds keyfix, labeltext, coverage, the text boxes and the product object; 0.1.0's signatures are unchanged."""
__version__ = "0.2.0"
