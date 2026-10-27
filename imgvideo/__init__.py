"""imgvideo: product-label-refinement for video clips. Tracks the product's label surface through a shot, fits
the stills engine's mappings (homography, cylinder) or a tube per frame, and re-projects the true label into every frame.

The geometry, warp, grade and finish code is the stills engine's (imgpost): this package only adds time.
Coordinate convention as in imgpost: integer coordinates are pixel centres, x right, y down.
"""
__version__ = "0.9.0"     # CHANGELOG.md; a port pins it with the files' sha256 (video_regress.py --manifest)
