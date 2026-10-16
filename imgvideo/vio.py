"""Video I/O through ffmpeg pipes. Frames are RGB uint8. Decode and encode use the same matrix and range (the
source's, bt709 by default), so pixels the job doesn't touch come back as they were, up to the encoder's loss."""
import json
import subprocess

import numpy as np


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,r_frame_rate,nb_frames,color_space,color_range",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"]
    v = next(s for s in st if s["codec_type"] == "video")
    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    return {"w": int(v["width"]), "h": int(v["height"]), "fps": num / den, "fps_str": v["r_frame_rate"],
            "frames": int(v.get("nb_frames", 0)), "matrix": v.get("color_space") or "bt709",
            "range": "pc" if v.get("color_range") == "pc" else "tv",
            "audio": any(s["codec_type"] == "audio" for s in st)}


def read(path, start=0, stop=None, info=None):
    """Frames [start, stop) as a T x H x W x 3 uint8 array."""
    info = info or probe(path)
    w, h = info["w"], info["h"]
    stop = info["frames"] if stop is None else stop
    vf = (f"select='between(n\\,{start}\\,{stop - 1})',"
          f"scale=in_color_matrix={info['matrix']}:in_range={info['range']}:flags=bicubic+accurate_rnd+full_chroma_int,"
          "format=rgb24")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf, "-vsync", "0", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


class Writer:
    """H.264 (yuv420p, crf 12 by default) with the source's matrix, range and fps; optional audio from a file.
    lossless: QuickTime PNG instead (RGB, exact)."""

    def __init__(self, path, info, crf=12, audio_from=None, start=0, lossless=False):
        w, h = info["w"], info["h"]
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", info["fps_str"], "-i", "-"]
        if lossless:                   # RGB as decoded, bit for bit (QuickTime PNG): untouched pixels stay exact
            if audio_from:
                cmd += ["-ss", f"{start / info['fps']:.6f}", "-i", str(audio_from), "-map", "0:v", "-map", "1:a?",
                        "-c:a", "copy", "-shortest"]
            cmd += ["-c:v", "png", "-pix_fmt", "rgb24", str(path)]
            self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            return
        if audio_from:
            cmd += ["-ss", f"{start / info['fps']:.6f}", "-i", str(audio_from), "-map", "0:v", "-map", "1:a?",
                    "-c:a", "copy", "-shortest"]
        cmd += ["-vf", f"scale=out_color_matrix={info['matrix']}:out_range={info['range']}:flags=bicubic+accurate_rnd",
                "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-colorspace", info["matrix"], "-color_primaries", info["matrix"], "-color_trc", info["matrix"],
                "-color_range", info["range"], "-movflags", "+faststart", str(path)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, rgb):
        self.p.stdin.write(np.ascontiguousarray(rgb, np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError("ffmpeg failed while encoding")
