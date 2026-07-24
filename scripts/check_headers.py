#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Enforce the two-line copyright / SPDX header on the core Python package.

Every tracked ``.py`` file in scope must begin with the copyright + SPDX header::

    # Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
    # SPDX-License-Identifier: GPL-3.0-or-later

(An optional ``#!`` shebang and/or a PEP 263 ``coding`` line may precede it; the
header is then required immediately after.) A file that already carries this block
block under a different copyright year -- or the pre-rename ``OptArena authors``
wording -- is accepted as-is and never rewritten, so ``--fix`` never stacks a
second notice on top of an existing one. ``--fix`` inserts the canonical 2021
header only into files that have no such block at all.

Scope -- the CORE package only: ``hpcagent_bench/`` (EXCLUDING ``benchmarks/`` and
the ``numpy_translators/`` distribution), plus ``scripts/`` and ``tests/``. Ported
kernels under ``benchmarks/`` and the translator's generated sources carry their
own provenance and are a separate, deferred header pass -- the same dirs the format
hook skips.

Run standalone with no arguments and the tool discovers the scope via
``git ls-files``; pre-commit instead passes the staged files as positional
arguments (already narrowed by the hook's path filter). ``--fix`` inserts the
header in place; without it the tool only reports, exiting 1 when any in-scope
file is missing the header (the offenders and the fix command are printed).
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SPDX_LINE = "# SPDX-License-Identifier: GPL-3.0-or-later"
# What --fix writes into a headerless file (the canonical year).
HEADER = (
    "# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.",
    SPDX_LINE,
)
# A header is PRESENT when an ETH-Zurich copyright line is immediately followed by the
# SPDX line. The year and the authors-org phrase are matched loosely so a pre-existing
# notice -- any year, and the pre-rename ``OptArena authors`` wording -- counts as headered
# and is left untouched rather than restamped or stacked under a second copy.
COPYRIGHT_RE = re.compile(r"^# Copyright \d{4} ETH Zurich and the [\w.-]+ authors\.$")

# Included roots and the sub-prefixes carved back out (their headers are deferred).
SCOPE_PREFIXES = ("hpcagent_bench/", "scripts/", "tests/")
EXCLUDE_PREFIXES = ("hpcagent_bench/benchmarks/", "hpcagent_bench/numpy_translators/")

CODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=]")


def in_scope(rel):
    posix = rel.replace("\\", "/")
    if not posix.endswith(".py"):
        return False
    if any(posix.startswith(p) for p in EXCLUDE_PREFIXES):
        return False
    return any(posix.startswith(p) for p in SCOPE_PREFIXES)


def tracked_python():
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True)
    return [ln for ln in out.stdout.splitlines() if ln.strip()] if out.returncode == 0 else []


def prefix_len(lines):
    """Number of leading lines (shebang, then optional coding) the header goes after."""
    idx = 0
    if idx < len(lines) and lines[idx].startswith("#!"):
        idx += 1
    if idx < len(lines) and CODING_RE.match(lines[idx]):
        idx += 1
    return idx


def has_header(lines):
    idx = prefix_len(lines)
    seg = [line.rstrip("\n") for line in lines[idx:idx + 2]]
    return len(seg) == 2 and COPYRIGHT_RE.match(seg[0]) is not None and seg[1] == SPDX_LINE


def insert_header(path):
    """Insert the header after any shebang/coding lines. Returns True when it wrote."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    if has_header([ln.rstrip("\n") for ln in lines]):
        return False
    idx = prefix_len([ln.rstrip("\n") for ln in lines])
    # Preserve the newline style already used at the insertion point.
    newline = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
    block = [line + newline for line in HEADER]
    path.write_text("".join(lines[:idx] + block + lines[idx:]), encoding="utf-8")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix", action="store_true", help="insert the header in place instead of failing")
    ap.add_argument("files", nargs="*", help="explicit files to check (default: all tracked in-scope .py)")
    args = ap.parse_args(argv)

    candidates = args.files if args.files else tracked_python()
    targets = [rel for rel in sorted(set(candidates)) if in_scope(rel) and (REPO_ROOT / rel).is_file()]

    offenders = []
    for rel in targets:
        path = REPO_ROOT / rel
        if args.fix:
            if insert_header(path):
                offenders.append(rel)
        else:
            lines = path.read_text(encoding="utf-8").splitlines()
            if not has_header(lines):
                offenders.append(rel)

    if args.fix:
        print(f"check-headers: inserted the header into {len(offenders)} of {len(targets)} in-scope file(s)")
        return 0
    if not offenders:
        print(f"check-headers: {len(targets)} in-scope file(s) OK")
        return 0
    print(f"check-headers: {len(offenders)} of {len(targets)} in-scope file(s) missing the copyright/SPDX header:\n")
    for rel in offenders:
        print(f"  {rel}")
    print("\nFix with:  python scripts/check_headers.py --fix")
    return 1


if __name__ == "__main__":
    sys.exit(main())
