#!/usr/bin/env python3
"""No client, product or brand names in this public pack: scan a branch's changes against a private list of terms.

  python3 -P tools/brand_check.py [--base origin/main] [--terms FILE]   # the branch's changes since it left --base
  python3 -P tools/brand_check.py --all [--terms FILE]                  # every tracked file, once in a while

The terms are regular expressions, one per line (blank lines and lines starting with # skipped), matched
case-insensitively; write \\bcap\\b, not cap, so capture doesn't match. They come from --terms, else the BRAND_TERMS
environment variable (the pull request checks pass the repository secret of that name). The list never goes into the
repository: a public file of client names would be the leak this check exists to stop.

Scanned: the lines the branch adds, the names of the files it adds or renames, and its commit messages (with --all:
every tracked text file and its name). A match prints where it is and the term's number in the list, never the
term or the line, since pull request logs are public. Exit 1 on a match, 2 without terms.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent.parent


def git(*args):
    return subprocess.run(["git", "-C", str(PACK), *args], capture_output=True, check=True).stdout.decode("utf-8", "replace")


def load_terms(path):
    text = Path(path).expanduser().read_text() if path else os.environ.get("BRAND_TERMS", "")
    lines = [l.strip() for l in text.splitlines()]
    return [re.compile(l, re.I) for l in lines if l and not l.startswith("#")]


def hits(terms, text):
    return [i + 1 for i, t in enumerate(terms) if t.search(text)]


def added_lines(base):
    """-> [(path, line number, text)] for every line the branch adds since it left base (binary files skipped)"""
    out, path, n = [], None, 0
    for l in git("diff", "--no-color", "--no-ext-diff", "-U0", "-M", "--diff-filter=ACMR", base, "HEAD").splitlines():
        if l.startswith("+++ "):
            path = l[6:] if l.startswith("+++ b/") else None
        elif l.startswith("@@"):
            n = int(re.match(r"@@ -\S+ \+(\d+)", l).group(1))
        elif l.startswith("+") and path:
            out.append((path, n, l[1:]))
            n += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main", help="the branch's changes since it left this (default origin/main)")
    ap.add_argument("--all", action="store_true", help="every tracked file instead of the branch's changes")
    ap.add_argument("--terms", help="a file of terms (default: the BRAND_TERMS environment variable)")
    a = ap.parse_args()
    terms = load_terms(a.terms)
    if not terms:
        print("no terms: pass --terms FILE or set BRAND_TERMS (in the pull request checks, the repository secret "
              "BRAND_TERMS; a pull request from a fork doesn't get it)")
        sys.exit(2)
    found = []
    if a.all:
        for name in git("ls-files", "-z").split("\0"):
            if not name:
                continue
            found += [f"file name {name}: term {i}" for i in hits(terms, name)]
            data = (PACK / name).read_bytes() if (PACK / name).is_file() else b""
            if b"\0" in data[:8192]:
                continue                                    # binary
            for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                found += [f"{name}:{n}: term {i}" for i in hits(terms, line)]
        scope = "every tracked file"
    else:
        base = git("merge-base", a.base, "HEAD").strip()
        for st in git("diff", "--name-status", "-M", "--diff-filter=ACR", base, "HEAD").splitlines():
            name = st.split("\t")[-1]
            found += [f"file name {name}: term {i}" for i in hits(terms, name)]
        for path, n, line in added_lines(base):
            found += [f"{path}:{n}: term {i}" for i in hits(terms, line)]
        for c in git("log", "--format=%h%x00%B%x01", f"{base}..HEAD").split("\x01"):
            if c.strip():
                h, msg = c.strip().split("\0", 1)
                found += [f"commit {h} message: term {i}" for i in hits(terms, msg)]
        scope = f"the changes since {base[:7]} ({a.base})"
    if found:
        print("\n".join(found))
        print(f"FAIL: {len(found)} brand-name match(es) in {scope}; term N is the Nth term of the private list")
        sys.exit(1)
    print(f"PASS: no brand names in {scope} ({len(terms)} terms)")


if __name__ == "__main__":
    main()
