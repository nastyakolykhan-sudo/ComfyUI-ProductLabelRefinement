#!/usr/bin/env python3
"""Keep the video nodes' engine pinned to the product-label-refinement skill: imgvideo (with imgvideo/alltracker) and
imgauto, byte for byte, next to the stills nodes' imgpost (tools/sync_imgpost.py keeps that one).

  python3 -P tools/sync_video_engine.py              # check (default); exit 1 when anything moved
  python3 -P tools/sync_video_engine.py --update     # copy the skill's imgvideo/, imgauto/ and video.py here, re-pin

The pin is imgpost.lock.json's "video_engine" entry: each package's version and the sha256 of every file, plus the
imgpost version the video engine was synced with (the stills sync must keep imgpost at the skill's version too),
and video.py's sha256: tools/skill_cli/video.py is the skill's video.py, which the pull request checks run
tools/regress_video.py against, since GitHub has no skill folder.
After --update: run tools/regress_video.py, then commit.
"""
import argparse
import datetime
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent.parent
LOCK = PACK / "imgpost.lock.json"
PACKAGES = ("imgvideo", "imgauto")
CLI = PACK / "tools" / "skill_cli"


def files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in (".py", "") and "__pycache__" not in p.parts}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


def version(pkg_dir):
    m = re.search(r'__version__\s*=\s*"([^"]+)"', (pkg_dir / "__init__.py").read_text())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill-dir", help="default: the lock's skill path")
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    lock = json.loads(LOCK.read_text())
    skill = Path(a.skill_dir or lock["skill"]).expanduser()
    if a.update:
        if not (skill / "scripts").is_dir():
            sys.exit(f"skill not found: {skill}")
        ve = {"synced": datetime.date.today().isoformat(), "imgpost_version": version(skill / "scripts" / "imgpost"),
              "packages": {}}
        for pkg in PACKAGES:
            src, dst = skill / "scripts" / pkg, PACK / pkg
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.md"))
            ve["packages"][pkg] = {"version": version(dst), "files": files(dst)}
        shutil.copy2(skill / "scripts" / "video.py", CLI / "video.py")
        ve["video_py"] = sha(CLI / "video.py")
        lock["video_engine"] = ve
        LOCK.write_text(json.dumps(lock, indent=2) + "\n")
        print("synced " + ", ".join(f"{p} {v['version']} ({len(v['files'])} files)" for p, v in ve["packages"].items()) +
              f" (with imgpost {ve['imgpost_version']})")
        if ve["imgpost_version"] != lock.get("imgpost_version"):
            print(f"the pack's imgpost is {lock.get('imgpost_version')}: run tools/sync_imgpost.py --update too")
        return
    ve = lock.get("video_engine")
    if not ve:
        sys.exit("no video_engine pin: run --update")
    problems = []
    for pkg, pin in ve["packages"].items():
        here = files(PACK / pkg)
        edited = sorted(n for n in set(here) | set(pin["files"]) if here.get(n) != pin["files"].get(n))
        if edited:
            problems.append(f"{pkg}/ differs from the lock (edited here?): " + ", ".join(edited))
        src = skill / "scripts" / pkg
        if src.is_dir():
            there = files(src)
            moved = sorted(n for n in set(there) | set(pin["files"]) if there.get(n) != pin["files"].get(n))
            if moved:
                problems.append(f"the skill's {pkg} changed since the pin ({ve['synced']}): " + ", ".join(moved) + "  -> --update")
    if sha(CLI / "video.py") != ve.get("video_py"):
        problems.append("tools/skill_cli/video.py differs from the lock's video_py (edited here?)  -> --update")
    if (skill / "scripts").is_dir() and sha(skill / "scripts" / "video.py") != ve.get("video_py"):
        problems.append(f"the skill's video.py changed since the pin ({ve['synced']})  -> --update")
    if ve["imgpost_version"] != lock.get("imgpost_version"):
        problems.append(f"the video engine was synced with imgpost {ve['imgpost_version']}, the pack has "
                        f"{lock.get('imgpost_version')}")
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print(f"video engine pinned {ve['synced']}: " + ", ".join(f"{p} {v['version']}" for p, v in ve["packages"].items()) +
          f", with imgpost {ve['imgpost_version']}; the pack and the skill match")


if __name__ == "__main__":
    main()
