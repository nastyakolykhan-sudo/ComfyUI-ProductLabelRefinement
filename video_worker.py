"""Runs the engine's refine() in its own Python, for Product Label: Refine (Video) with processes > 1: render workers can spawn
from here (their main module is this guarded script), not from inside ComfyUI.

  python -P video_worker.py IO_DIR      # IO_DIR/inputs.npz + args.json in; outputs.npz + outputs.json out
"""
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))          # imgvideo, imgpost, imgauto: this pack's
    import numpy as np
    from imgvideo.pipeline import refine

    io = Path(sys.argv[1])
    z = np.load(io / "inputs.npz")
    a = json.loads((io / "args.json").read_text())
    r = refine(z["frames"], z["masks"], [z[f"kf{i}"] for i in range(a["n_kf"])], a["settings"], a["kfr"], fps=a["fps"],
               work=a["work"], workers=a["workers"])
    np.savez(io / "outputs.npz", frames=r["frames"], matte=r["matte"], changed=r["changed"])
    (io / "outputs.json").write_text(json.dumps({"report": r["report"], "qa_plot": str(r["qa_plot"]) if r["qa_plot"] else None},
                                                default=str))
