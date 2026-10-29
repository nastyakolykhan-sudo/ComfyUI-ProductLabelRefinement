"""Reduced-size decoding for analysis: a whole master at 1/k scale fits in memory where full frames don't."""
import subprocess

import numpy as np

from imgvideo import vio


def read_small(path, k=2, info=None, gray=False):
    """Every frame at 1/k size (area-averaged), RGB uint8, or one channel when gray."""
    info = info or vio.probe(path)
    w, h = info["w"] // k, info["h"] // k
    fmt = "gray" if gray else "rgb24"
    vf = (f"scale=in_color_matrix={info['matrix']}:in_range={info['range']}:flags=area,"
          f"scale={w}:{h}:flags=area,format={fmt}")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf, *vio.passthrough(), "-f", "rawvideo",
                          "-pix_fmt", fmt, "-"], capture_output=True, check=True).stdout
    shape = (-1, h, w) if gray else (-1, h, w, 3)
    return np.frombuffer(raw, np.uint8).reshape(shape)


def read_masks(path, k=2, info=None):
    """A matte video as booleans at 1/k size (white = product)."""
    return read_small(path, k, info, gray=True) > 127
