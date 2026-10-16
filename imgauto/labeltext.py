"""The label's text, read from the reference photos.

The agent reads it: the photos are cut down to the product (split from props by the clip's own product colours),
turned upright and zoomed, with close-up bands for the small print, so every word has enough pixels. Tesseract reads
the same crops as a cross-check; the words it can't confirm are listed. Qwen-VL, when wired, only breaks a tie
between the two (or reads when no agent runs).
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import keyfix, refangle, verify


def crops(path, out_dir, stem, colours=None, target=None, long_side=2400, bands=3, overlap=0.25):
    """Zoomed, upright crops of the product on one reference photo: the whole product, then `bands` overlapping
    close-ups along its length at twice the zoom. target: a clip matte, to pick the product among the photo's objects
    (refangle.product_object). -> [paths]"""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    k = max(1, int(round(max(rgb.shape[:2]) / 1000)))           # find the product at about 1000 px
    if target is not None:
        po = refangle.product_object(path, target, colours=colours, scale=k)
        objs = [po[:2]] if po else []
    else:
        objs = refangle.objects(path, colours=colours, scale=k)
    if not objs:
        return []
    geom = keyfix.crop_geometry(objs[0][0], margin=0.04)
    geom = dict(geom, cx=geom["cx"] * k + (k - 1) / 2, cy=geom["cy"] * k + (k - 1) / 2, w=geom["w"] * k,
                h=geom["h"] * k)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    whole = keyfix.crop(rgb, geom, long_side)
    paths = [keyfix.save_png(whole, out_dir / f"{stem}.png")]
    big = keyfix.crop(rgb, geom, 2 * long_side)
    h = big.shape[0]
    step = h / (bands - (bands - 1) * overlap)
    for j in range(bands):
        y0 = int(j * step * (1 - overlap))
        paths.append(keyfix.save_png(big[y0: min(h, int(y0 + step))], out_dir / f"{stem}_band{j + 1}.png"))
    return paths


def tesseract_words(img_paths, threads=6):
    """Every word tesseract reads on the crops (local contrast, four orientations). The polarity is chosen once, on
    the first crop: the one that reads more words."""
    from concurrent.futures import ThreadPoolExecutor
    norm = []
    for p in img_paths:
        L = np.asarray(Image.open(p).convert("L")).astype(np.float32) / 255.0
        norm.append(verify._local_contrast(L, sigma=14.0))
    if not norm:
        return set()
    first = [verify.ocr(img, psm=11, scale=1) for img in (norm[0], 1.0 - norm[0])]
    if first[0] is None:
        return None
    flip = len(first[1]) > len(first[0])
    jobs = [(1.0 - n if flip else n, r) for n in norm for r in range(4)]
    with ThreadPoolExecutor(threads) as ex:
        got = list(ex.map(lambda jr: verify.ocr(np.rot90(jr[0], jr[1]), psm=11, scale=1), jobs))
    return set().union(*[g for g in got if g])


def cross_check(copy, words):
    """The agent's copy against tesseract's reading: -> {"confirmed", "near", "unconfirmed"} word lists. A word
    tesseract misses is common at small sizes; a near miss is worth a second look."""
    sp = verify.spelling(copy, words or set())
    return {"confirmed": [w for w, v in sp.items() if v == "exact"],
            "near": [w for w, v in sp.items() if v == "near"],
            "unconfirmed": [w for w, v in sp.items() if v == "missing"]}


def write(out_dir, refs, copy, words):
    res = {"crops": [str(p) for p in refs], "tesseract": sorted(words) if words is not None else None,
           "copy": copy, "copy_source": "products.json" if copy else None}
    if copy:
        res["cross_check"] = cross_check(copy, words)
    else:
        res["request"] = ("Read the label from these crops, word for word, and put the words in products.json "
                          "\"copy\" for this product; tesseract's reading is listed for comparison.")
    (Path(out_dir) / "text.json").write_text(json.dumps(res, indent=1))
    return res
