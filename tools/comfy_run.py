#!/usr/bin/env python3
"""Submit an Image Post workflow (API format) to a ComfyUI server and download what it produced.

  python3 -P tools/comfy_run.py --server http://127.0.0.1:8188 \\
      --workflow workflows/image-post.api.json \\
      --scene path/to/scene.png --reference path/to/packshot.png \\
      [--job path/to/job.json] [--stage align|grade|all] [--psd] --out ./ip-run
  python3 -P tools/comfy_run.py --workflow workflows/image-post-draft.api.json --scene ... --reference ... \\
      --set "ImagePostAutoDraft.quad=X,Y X,Y X,Y X,Y" [--mask product_mask.png] --out ./ip-draft

--scene / --reference upload the files (POST /upload/image, into input/image-post/) and point the Load Image
nodes that feed Warp Reference at them; --image NODE_ID=PATH does the same for any Load Image node.
--job replaces the job text in every Load Job node. --set NODE.INPUT=VALUE sets a widget value on a node, NODE being
its id or its class_type (every node of that class): the Auto Draft node's quad, pairs, occluders, outline.
--mask PATH adds a Load Image (as Mask) node, red channel (white = product), feeding every Auto Draft node's
product_mask. --stage prunes the graph like run.py --stage
(align: fit, warp, overlay; grade: + grade preview and relight map; all: everything).
Save PSD nodes run only with --psd or "psd": true in the job, like run.py --psd.
--out receives every output file (server folder layout: images, the PSD), report.json (from the QA node), the
other nodes' ui text as CLASS-ID.txt, and history.json. Exit codes: 0 done, 2 prompt rejected, 3 execution
error, 4 timeout. Stdlib only. Servers behind an auth proxy: --header "Name: value" (repeatable).
"""
import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

STAGE_NODES = {
    "align": {"ImagePostLoadJob", "ImagePostAutoDraft", "ImagePostJobMasks", "ImagePostFitWarp", "ImagePostWarpReference"},
}
STAGE_NODES["grade"] = STAGE_NODES["align"] | {"ImagePostGradeToScene"}


# ---- workflow editing (shared with offline_run.py) ----

def load_workflow(path):
    wf = json.loads(Path(path).expanduser().read_text())
    if "nodes" in wf and "links" in wf:
        sys.exit(f"{path} is a UI workflow: use the API-format file (*.api.json, or 'Export (API)' in ComfyUI)")
    return wf


def is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


def loaders(wf, input_name):
    """Load Image nodes feeding Warp Reference's `input_name` (scene or reference)."""
    ids = {n["inputs"][input_name][0] for n in wf.values()
           if n["class_type"] == "ImagePostWarpReference" and is_link(n["inputs"].get(input_name))}
    return sorted(i for i in ids if wf[i]["class_type"] == "LoadImage")


def image_targets(wf, scene=None, reference=None, images=(), mask=None):
    """-> {load image node id: local path}. mask: adds the product mask's loader (add_mask)."""
    out = {}
    for role, path in (("scene", scene), ("reference", reference)):
        if path:
            ids = loaders(wf, role)
            if len(ids) != 1:
                sys.exit(f"--{role}: expected one Load Image node feeding Warp Reference's {role}, found {ids}; use --image ID=PATH")
            out[ids[0]] = path
    for spec in images:
        nid, _, path = spec.partition("=")
        if nid not in wf or wf[nid]["class_type"] != "LoadImage":
            sys.exit(f"--image {spec}: node {nid} is not a Load Image node")
        out[nid] = path
    if mask:
        out[add_mask(wf)] = mask
    for nid, path in out.items():
        p = Path(path).expanduser()
        if not p.is_file():
            sys.exit(f"image not found: {p}")
        out[nid] = p
    return out


def set_job(wf, job_path):
    text = Path(job_path).expanduser().read_text()
    json.loads(text)   # fail here, not on the server
    n = 0
    for node in wf.values():
        if node["class_type"] == "ImagePostLoadJob":
            node["inputs"]["job_json"], node["inputs"]["job_path"] = text, ""
            n += 1
    if not n:
        sys.exit("--job: the workflow has no Image Post Load Job node")


def set_inputs(wf, specs):
    """--set NODE.INPUT=VALUE: a widget value on node NODE (an id) or on every node of class NODE. A value replacing a
    number or a boolean is read as JSON; anything else stays text."""
    for spec in specs:
        key, eq, value = spec.partition("=")
        target, dot, name = key.rpartition(".")
        if not eq or not dot or not target or not name:
            sys.exit(f"--set {spec!r}: expected NODE.INPUT=VALUE (NODE: a node id or a class_type)")
        ids = [target] if target in wf else [nid for nid, n in wf.items() if n["class_type"] == target]
        if not ids:
            sys.exit(f"--set {spec!r}: the workflow has no node {target!r}")
        for nid in ids:
            old = wf[nid]["inputs"].get(name)
            if is_link(old):
                sys.exit(f"--set {spec!r}: input {name} of node {nid} is a link, not a value")
            if isinstance(old, (bool, int, float)):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    sys.exit(f"--set {spec!r}: {name} takes a number or true/false")
            wf[nid]["inputs"][name] = value


def add_mask(wf):
    """--mask: a Load Image (as Mask) node, red channel, feeding every Auto Draft node's product_mask -> its id."""
    drafts = [n for n in wf.values() if n["class_type"] == "ImagePostAutoDraft"]
    if not drafts:
        sys.exit("--mask: the workflow has no Auto Draft node")
    nid = str(max(int(k) for k in wf) + 1)
    wf[nid] = {"inputs": {"image": "", "channel": "red"}, "class_type": "LoadImageMask", "_meta": {"title": "Product mask"}}
    for n in drafts:
        n["inputs"]["product_mask"] = [nid, 0]
    return nid


def psd_wanted(wf, flag=False):
    """run.py's rule: --psd, or "psd": true in the job (read from the Load Job text; a job_path is server-side)."""
    if flag:
        return True
    for node in wf.values():
        if node["class_type"] == "ImagePostLoadJob" and node["inputs"].get("job_json", "").strip():
            try:
                return bool(json.loads(node["inputs"]["job_json"]).get("psd"))
            except (json.JSONDecodeError, AttributeError):
                return False
    return False


def keep_psd(wf, keep):
    """Drop the Save PSD nodes unless a PSD is wanted."""
    return wf if keep else {nid: n for nid, n in wf.items() if n["class_type"] != "ImagePostSavePSD"}


def prune(wf, stage):
    """Keep only nodes whose Image Post ancestors all belong to the stage (run.py --stage)."""
    if stage == "all":
        return wf
    allowed, memo = STAGE_NODES[stage], {}

    def ok(nid):
        if nid not in memo:
            memo[nid] = True   # cycle guard
            node = wf[nid]
            memo[nid] = (not node["class_type"].startswith("ImagePost") or node["class_type"] in allowed) and \
                all(ok(v[0]) for v in node["inputs"].values() if is_link(v))
        return memo[nid]

    return {nid: node for nid, node in wf.items() if ok(nid)}


# ---- server ----

class Server:
    def __init__(self, url, headers=()):
        self.url = url.rstrip("/")
        self.headers = dict(h.split(":", 1) for h in headers)
        self.headers = {k.strip(): v.strip() for k, v in self.headers.items()}

    def request(self, path, data=None, content_type=None):
        req = urllib.request.Request(self.url + path, data=data, headers=dict(self.headers))
        if content_type:
            req.add_header("Content-Type", content_type)
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    def json(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        return json.loads(self.request(path, data, "application/json" if data else None))

    def upload(self, path, subfolder="image-post"):
        boundary = uuid.uuid4().hex
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parts = []
        for name, value in (("overwrite", "true"), ("subfolder", subfolder), ("type", "input")):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
                     f'Content-Type: {ctype}\r\n\r\n'.encode() + path.read_bytes() + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        r = json.loads(self.request("/upload/image", b"".join(parts), f"multipart/form-data; boundary={boundary}"))
        return f"{r['subfolder']}/{r['name']}" if r.get("subfolder") else r["name"]


def save_outputs(entry, wf, fetch, out):
    """Write images and files (PSD), ui text and history.json; -> list of written paths."""
    written = []
    for nid, ui in entry.get("outputs", {}).items():
        cls = wf.get(nid, {}).get("class_type", "node")
        for img in ui.get("images", []) + ui.get("files", []):
            dest = out / img.get("subfolder", "") / img["filename"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(fetch(img))
            written.append(dest)
        if ui.get("text"):
            text = "\n".join(ui["text"])
            dest = out / ("report.json" if cls == "ImagePostQASheet" else f"{cls}-{nid}.txt")
            dest.write_text(text)
            written.append(dest)
    (out / "history.json").write_text(json.dumps(entry, indent=1))
    return written


def summary(out):
    rep = out / "report.json"
    if rep.exists():
        r = json.loads(rep.read_text())
        c = r.get("change", {})
        print(f"changed    {c.get('changed_px', 0):,} px, bbox {c.get('bbox')}, outside work box: {c.get('outside_roi_changed')}")
        for l in r.get("align", {}).get("lines", []):
            print(f"line       {l['label'][:40]:40s} max {l['max_px']:5.2f}")
        if r.get("audit_verdict"):
            print(r["audit_verdict"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--scene")
    ap.add_argument("--reference")
    ap.add_argument("--image", action="append", default=[], metavar="NODE_ID=PATH")
    ap.add_argument("--job", help="job.json to run instead of the one embedded in the workflow")
    ap.add_argument("--set", action="append", default=[], metavar="NODE.INPUT=VALUE", help="a widget value, e.g. the Auto Draft quad")
    ap.add_argument("--mask", help="the product's mask (white = product, scene-sized) for the Auto Draft nodes")
    ap.add_argument("--stage", choices=("align", "grade", "all"), default="all")
    ap.add_argument("--psd", action="store_true", help="also run the Save PSD nodes (same as \"psd\": true in the job)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--header", action="append", default=[], metavar="'Name: value'")
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)   # keep progress and errors in order when piped

    wf = load_workflow(a.workflow)
    if a.job:
        set_job(wf, a.job)
    set_inputs(wf, a.set)
    targets = image_targets(wf, a.scene, a.reference, a.image, a.mask)
    wf = prune(keep_psd(wf, psd_wanted(wf, a.psd)), a.stage)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    srv = Server(a.server, a.header)

    for nid, path in targets.items():
        if nid in wf:
            wf[nid]["inputs"]["image"] = srv.upload(path)
            print(f"uploaded   {path} -> {wf[nid]['inputs']['image']}")
    (out / "prompt.json").write_text(json.dumps(wf, indent=1))
    try:
        r = srv.json("/prompt", {"prompt": wf, "client_id": uuid.uuid4().hex})
    except urllib.error.HTTPError as e:
        print(f"prompt rejected ({e.code}):\n{e.read().decode(errors='replace')}", file=sys.stderr)
        sys.exit(2)
    pid = r["prompt_id"]
    print(f"queued     {pid}")

    t0, entry = time.time(), None
    while time.time() - t0 < a.timeout:
        h = srv.json(f"/history/{pid}")   # the entry appears once the prompt has finished or failed
        if pid in h:
            entry = h[pid]
            break
        time.sleep(1.0)
    if entry is None:
        print(f"timed out after {a.timeout:.0f} s (prompt {pid} may still be running)", file=sys.stderr)
        sys.exit(4)

    def fetch(img):
        q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                    "type": img.get("type", "output")})
        return srv.request(f"/view?{q}")

    written = save_outputs(entry, wf, fetch, out)
    status = entry.get("status", {})
    if status.get("status_str") == "error":
        for kind, msg in status.get("messages", []):
            if kind == "execution_error":
                print(f"failed     node {msg.get('node_id')} ({msg.get('node_type')}): {msg.get('exception_message')}",
                      file=sys.stderr)
        sys.exit(3)
    print(f"done       {time.time() - t0:.1f} s")
    for p in written:
        print(f"saved      {p}")
    summary(out)


if __name__ == "__main__":
    main()
