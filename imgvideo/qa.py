"""Per-frame QA in label space: each frame unwarped onto the artwork's grid and scored against the artwork.

  ncc   correlation of the high-passed luminance with the artwork's, where the label is seen (print structure: light
        and grade differences drop out). The same score on the source frame is the baseline.
  lock  the source's own label against its still label (the shot's median): how well the mapping holds the surface.
        It doesn't use the render, so it is the independent check of the geometry.
  ocr   tesseract on the unwarped text blocks: share of the artwork's words read back (when tesseract is installed).
Note: ncc reads the output through the same mapping that rendered it, so it checks the render (print fidelity,
blur, relight, occlusion), not the registration; reprojection error and lock check the registration."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates

from imgpost import images as im

from . import labelspace as ls
from .optics import hp_ncc

WORD = re.compile(r"[A-Za-zÀ-ÿ]{3,}")


def ocr_words(gray, rotate=0, scale=4):
    """Words (3+ letters, lower case) tesseract reads in a float image of dark print on light (0..1)."""
    if shutil.which("tesseract") is None:
        return None
    g = np.rot90(gray, rotate)
    img = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8))
    img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.png"
        img.save(p)
        out = subprocess.run(["tesseract", str(p), "stdout", "--psm", "6"], capture_output=True, text=True).stdout
    return [w.lower() for w in WORD.findall(out)]


def _ink_gray(rgb, polarity="light"):
    """Label-space linear RGB -> print dark on light, contrast stretched, for OCR."""
    L = im.lin_to_srgb(im.lum(rgb))
    lo, hi = np.nanpercentile(L, 2), np.nanpercentile(L, 99.5)
    g = (L - lo) / max(hi - lo, 1e-3)
    return 1 - g if polarity == "light" else g


def _near(a, b):
    """One edit apart (a substitution, an insertion or a deletion): OCR's rn/m, l/i slips at a few px per letter."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


def _fold(w):
    """Accents off (açaí = acai): OCR at a few px per letter rarely keeps them."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", w) if not unicodedata.combining(c))


def recall(truth, words):
    """Share of the truth's words read back: exact (accents folded), or one edit apart for words of 6+ letters."""
    tw, got = {_fold(w) for w in truth}, {_fold(w) for w in words}
    hit = [w for w in tw if w in got or (len(w) >= 6 and any(_near(w, g) for g in got))]
    return len(hit) / max(len(tw), 1)


def _crop(img, step, tb):
    """A text box out of an image on a grid of `step` reference px: {"box": [u0, v0, u1, v1], "rotate": k} axis-aligned,
    or {"centre": [u, v], "size": [w, h], "angle": deg} along a tilted baseline (degrees from +x, y down), sampled
    every `step` px. Coordinates in local reference px."""
    if "centre" in tb:
        (cx, cy), (w, h), a = tb["centre"], tb["size"], np.radians(tb.get("angle", 0.0))
        d, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        S, R = np.meshgrid(np.arange(-w / 2, w / 2, step), np.arange(-h / 2, h / 2, step))
        X, Y = cx + S * d[0] + R * n[0], cy + S * d[1] + R * n[1]
        return map_coordinates(img, [Y / step, X / step], order=1, mode="nearest")
    u0, v0, u1, v1 = (int(round(x / step)) for x in tb["box"])
    return np.rot90(img[v0:v1, u0:u1], int(tb.get("rotate", 0)))


class Scorer:
    def __init__(self, art, ref_lin_crop, step, text_boxes=(), polarity="light", words=None):
        """text_boxes: see _crop, each with an optional "ink": "light" | "dark" (default: polarity, "both" = light).
        The truth is `words` when given, else what tesseract reads in the artwork's boxes at its own resolution."""
        self.grid = ls.Grid(art, step)
        self.polarity = polarity
        blur = np.stack([gaussian_filter(ref_lin_crop[..., c], 0.5 * step) for c in range(3)], -1)
        self.ref = np.stack([map_coordinates(blur[..., c], [self.grid.v, self.grid.u], order=1) for c in range(3)], -1)
        ink = map_coordinates(art.pack[..., 0], [self.grid.v, self.grid.u], order=1) > 0.2
        self.region = self.grid.inside & binary_dilation(ink, iterations=3)   # print and the body right around it
        self.boxes = list(text_boxes)
        self.truth = [w.lower() for w in words] if words else self._read_full(ref_lin_crop)
        # what OCR can read of the artwork itself at the frame's scale: the most a frame can score
        fs = self._read(self.ref, np.ones(self.grid.shape, bool)) if self.truth else None
        self.ceiling = round(recall(self.truth, fs), 3) if fs is not None else None

    def _ink(self, tb):
        k = tb.get("ink", self.polarity)
        return "light" if k == "both" else k

    def _read_full(self, ref_lin):
        if not self.boxes:
            return None
        words = []
        L = im.lum(ref_lin)
        for tb in self.boxes:
            w = ocr_words(_ink_gray(_crop(L, 1.0, tb)[..., None] * np.ones(3), self._ink(tb)), 0, scale=2)
            if w is None:
                return None
            words += w
        return words

    def _read(self, rgb, vis):
        if not self.boxes:
            return None
        words = []
        L = np.where(vis, im.lum(rgb), np.nan)
        for tb in self.boxes:
            c = _crop(np.nan_to_num(L, nan=np.nanmedian(L)), self.grid.step, tb)
            w = ocr_words(_ink_gray(c[..., None] * np.ones(3), self._ink(tb)), 0)
            if w is None:
                return None
            words += w
        return words

    def score(self, frame_u8, H_local, key, still=None, ocr=True):
        rgb, vis = self.grid.unwarp(frame_u8, H_local, key)
        m = vis & self.region
        out = {"vis_frac": round(float(m.sum() / max(self.region.sum(), 1)), 3),
               "ncc": round(hp_ncc(rgb, self.ref, m), 4) if m.sum() > 200 else None}
        if still is not None:
            ok = m & np.isfinite(still[..., 0])
            out["lock"] = round(hp_ncc(rgb, np.where(np.isfinite(still), still, 0), ok), 4) if ok.sum() > 200 else None
        if ocr and self.truth:
            words = self._read(rgb, vis)
            if words is not None:
                out["ocr"] = round(recall(self.truth, words), 3)
                out["ocr_words"] = " ".join(words[:24])
        return out, rgb, vis
