#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pre-commit guard: reject a commit that stages an oversized file.

The repo keeps only source + small fixtures; large binaries (datasets, compiled
artifacts, model dumps) belong out of git. This hook fails the commit when a staged
file exceeds its limit, so a stray artifact is caught before it lands rather than
after it bloats history.

Two limits, because the two kinds of file are not the same problem: ``--max-kb``
(default 500) for a binary blob, ``--max-text-kb`` (default 1024) for anything that
decodes as UTF-8. A binary at 500 KiB is already the thing being guarded against; a
source module at 500 KiB is just a big module.

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
#: Hand-written text gets its own, larger ceiling. What the hook is for is a BINARY blob -- a
#: dataset, a model dump, a compiled artifact -- and 500 KiB of those is already pathological,
#: while ``lowering.py`` is legitimately half a megabyte of source and grows by a few hundred
#: bytes per fix. Capping both at one number means a routine source edit fails the commit for a
#: reason that has nothing to do with the file it is guarding against; a generated .py table is
#: still caught, just at a threshold no hand-written module reaches by accident.
DEFAULT_MAX_TEXT_KB = 1024
BYTES_PER_KB = 1024


def staged_files():
    """Return the repo's currently-staged file paths (added / copied / modified)."""
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                         capture_output=True,
                         text=True)
    if out.returncode != 0:
        return []
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def is_text(path):
    """True when the file decodes as UTF-8 -- the property that separates source from a blob.

    Sniffed rather than keyed off the extension: a ``.py`` can be a generated megabyte of table
    and a fixture with no extension can be ordinary text, so the name is the wrong thing to ask.
    Reads a prefix, not the whole file, because the caller is already handling large ones.
    """
    with path.open("rb") as handle:
        head = handle.read(8192)
    if b"\0" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return len(head) == 8192  # a multi-byte char straddling the cut, not a binary file
    return True


def oversized(paths, max_bytes, max_text_bytes):
    """Yield ``(path, size_bytes, limit_bytes)`` for each existing regular file over its limit."""
    for rel in paths:
        path = Path(rel)
        if not path.is_file():  # deletions / submodules / gone paths
            continue
        size = path.stat().st_size
        limit = max_text_bytes if is_text(path) else max_bytes
        if size > limit:
            yield rel, size, limit


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-kb", type=int, default=DEFAULT_MAX_KB, help="binary size limit in KiB (default: 500)")
    ap.add_argument("--max-text-kb",
                    type=int,
                    default=DEFAULT_MAX_TEXT_KB,
                    help="size limit in KiB for UTF-8 text (default: 1024)")
    ap.add_argument("files", nargs="*", help="files to check (default: the staged set)")
    args = ap.parse_args(argv)

    candidates = args.files if args.files else staged_files()
    offenders = sorted(oversized(candidates, args.max_kb * BYTES_PER_KB, args.max_text_kb * BYTES_PER_KB))

    if not offenders:
        return 0

    print(f"error: {len(offenders)} staged file(s) exceed their size limit:\n", file=sys.stderr)
    for rel, size, limit in offenders:
        print(f"  {rel}  ({size / BYTES_PER_KB:.0f} KiB, limit {limit / BYTES_PER_KB:.0f} KiB)", file=sys.stderr)
    print("\nKeep large artifacts out of git (see .gitignore), or raise --max-kb deliberately.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
