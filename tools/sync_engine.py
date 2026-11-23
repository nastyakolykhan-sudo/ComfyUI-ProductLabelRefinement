#!/usr/bin/env python3
"""The engine lives here, in the pack; the product-label-refinement skill carries a byte-identical copy.

  python3 -P tools/sync_engine.py              # check (default); exit 1 when the skill's copy differs from the pack's
  python3 -P tools/sync_engine.py --update     # copy the pack's engine into the skill and record it in the lock

The engine is three packages: imgpost (stills, with imgpost/stages.py, the steps both run.py and the nodes call),
imgvideo (video, with imgvideo/alltracker) and imgauto (the automatic decisions). Edit them here, on a branch of the
pack; never in the skill (its copy is overwritten by --update) and never in silverside-comfy-nodes (a mirror).
imgpost.lock.json records the skill's folder and what was last copied there (each package's version and the sha256 of
every file), so --check can tell the pack moving on (--update) from the skill being edited by hand (an error).
After --update: run tools/check_all.sh, then commit the pack and the skill.
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
PACKAGES = ("imgpost", "imgvideo", "imgauto")
SKIP = {"__pycache__", ".DS_Store"}


def files(root):
    root = Path(root)
    if not root.is_dir():
        return {}
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and not SKIP & set(p.relative_to(root).parts)}


def version(root):
    m = re.search(r'__version__\s*=\s*"([^"]+)"', (Path(root) / "__init__.py").read_text())
    return m.group(1) if m else None


def diff(a, b):
    return sorted(n for n in set(a) | set(b) if a.get(n) != b.get(n))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill-dir", help="default: the lock's skill path")
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    lock = json.loads(LOCK.read_text())
    skill = Path(a.skill_dir or lock["skill"]).expanduser()
    if not (skill / "scripts").is_dir():
        sys.exit(f"skill not found: {skill} (--skill-dir, or fix the lock's \"skill\")")

    if a.update:
        rec = {}
        for pkg in PACKAGES:
            src, dst = PACK / pkg, skill / "scripts" / pkg
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*SKIP))
            rec[pkg] = {"version": version(src), "files": files(dst)}
            print(f"{pkg} {rec[pkg]['version']}: {len(rec[pkg]['files'])} files -> {dst}")
        lock = {"skill": lock["skill"] if not a.skill_dir else str(skill), "synced": datetime.date.today().isoformat(),
                "packages": rec}
        LOCK.write_text(json.dumps(lock, indent=1) + "\n")
        return

    problems = []
    for pkg in PACKAGES:
        here, there = files(PACK / pkg), files(skill / "scripts" / pkg)
        pinned = lock.get("packages", {}).get(pkg, {}).get("files", {})
        if not diff(here, there):
            continue
        by_hand = diff(there, pinned)
        if by_hand:
            problems.append(f"{pkg}: the skill's copy was edited by hand since {lock.get('synced')}: "
                            + ", ".join(by_hand) + " -> move those edits into the pack, then --update")
        moved = diff(here, pinned)
        if moved:
            problems.append(f"{pkg}: the pack moved on since {lock.get('synced')}: " + ", ".join(moved) + " -> --update")
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("engine in the skill = the pack's: " +
          ", ".join(f"{p} {version(PACK / p)}" for p in PACKAGES) + f" (synced {lock.get('synced')})")


if __name__ == "__main__":
    main()
