#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pre-commit guard: reject a commit that stages an oversized file.

The repo keeps only source + small fixtures; large binaries (datasets, compiled
artifacts, model dumps) belong out of git. This hook fails the commit when any
staged file exceeds ``--max-kb`` kilobytes (default 500), so a stray artifact is
caught before it lands rather than after it bloats history.

Cross-platform by construction: pure ``pathlib``/``os.stat`` with no shell, so it
runs identically on macOS, WSL, and Linux (``language: system`` in
``.pre-commit-config.yaml`` -- no network fetch of a remote hook repo).

pre-commit passes the staged files as positional arguments; run standalone with no
arguments and the checker falls back to ``git diff --cached`` to find them itself.

Exit status: 0 when every checked file is within the limit, 1 when one or more
exceed it (each offender and its size are printed).
"""
import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_MAX_KB = 500
BYTES_PER_KB = 1024


def staged_files():
    """Return the repo's currently-staged file paths (added / copied / modified)."""
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                         capture_output=True,
                         text=True)
    if out.returncode != 0:
        return []
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def oversized(paths, max_bytes):
    """Yield ``(path, size_bytes)`` for each existing regular file over ``max_bytes``."""
    for rel in paths:
        path = Path(rel)
        if not path.is_file():  # deletions / submodules / gone paths
            continue
        size = path.stat().st_size
        if size > max_bytes:
            yield rel, size


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-kb", type=int, default=DEFAULT_MAX_KB, help="size limit in KiB (default: 500)")
    ap.add_argument("files", nargs="*", help="files to check (default: the staged set)")
    args = ap.parse_args(argv)

    candidates = args.files if args.files else staged_files()
    max_bytes = args.max_kb * BYTES_PER_KB
    offenders = sorted(oversized(candidates, max_bytes))

    if not offenders:
        return 0

    print(f"error: {len(offenders)} staged file(s) exceed the {args.max_kb} KiB limit:\n", file=sys.stderr)
    for rel, size in offenders:
        print(f"  {rel}  ({size / BYTES_PER_KB:.0f} KiB)", file=sys.stderr)
    print("\nKeep large artifacts out of git (see .gitignore), or raise --max-kb deliberately.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
