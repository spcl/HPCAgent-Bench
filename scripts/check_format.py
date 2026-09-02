#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Enforce the repo formatters on changed source files (column limit 120).

Routing by extension:
  * Python  (.py)            -> ruff format ([tool.ruff] line-length = 120)
  * C / C++ (.c .cc .cpp ...) -> clang-format (.clang-format, ColumnLimit 120)
  * Fortran (.f90 .F90 ...)   -> fprettify  (.fprettify.rc, line-length 120)

Scope: by default the files changed versus ``--base`` (merge-base with
``origin/main``) plus any staged/working-tree edits, so the CI ``format-check``
job and a local run agree. ``--all`` checks every tracked source file.

Generated sources are never style-gated (``*_generated.*``). NATIVE kernel references under
``hpcagent_bench/benchmarks/`` are not either: those C / C++ / Fortran files are faithful
transcriptions of an upstream source and reformatting them breaks the line-level correspondence
with it. Their Python siblings ARE gated -- see :data:`SKIP_PREFIXES`.

Python goes through ONE ``ruff format`` invocation for the whole file list (0.4s over 684 files,
against 105s for a yapf process per file). C / C++ / Fortran keep the process-per-file thread pool,
because neither clang-format nor fprettify reports which of a batch it would rewrite.

Exit status: 0 when every checked file is already formatted; 1 when one or more
need reformatting (the offenders and the fix command are printed); 2 on a setup
error (a needed formatter is missing). ``--fix`` reformats in place instead.
"""

import argparse
import json
import concurrent.futures
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: ``[tool.ruff] line-length`` -- passed explicitly so a run from another directory cannot pick up
#: a different project's configuration.
PY_LINE_LENGTH = 120

PY_EXT = {".py"}
CPP_EXT = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx"}
FORT_EXT = {".f", ".f90", ".f03", ".f08", ".f95", ".for"}  # matched case-insensitively

# Prefixes skipped for the NATIVE languages only. hpcagent_bench/benchmarks carries the C, C++ and
# Fortran references, which are transcribed from an upstream source: reformatting them breaks the
# line-level correspondence with the source they were ported from, which is provenance rather than
# a style backlog. The numpy kernels in those same directories are ordinary hand-written Python and
# are formatted like the rest of the repo.
SKIP_PREFIXES = ("hpcagent_bench/benchmarks/",)
SKIP_NAME_MARKERS = ("_generated.",)


def _run(cmd):
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)


def _git_lines(args):
    out = _run(["git", *args])
    return [ln for ln in out.stdout.splitlines() if ln.strip()] if out.returncode == 0 else []


def _ref_exists(ref):
    return _run(["git", "rev-parse", "--verify", "--quiet", ref]).returncode == 0


def changed_files(base):
    """Files changed vs ``base`` (merge-base form) plus staged + working-tree edits."""
    files = set()
    if base and _ref_exists(base):
        files.update(_git_lines(["diff", "--name-only", "--diff-filter=ACMRT", f"{base}...HEAD"]))
    else:
        print(f"note: base ref {base!r} not found; checking working-tree + staged changes only", file=sys.stderr)
    files.update(_git_lines(["diff", "--name-only", "--diff-filter=ACMRT", "HEAD"]))
    files.update(_git_lines(["diff", "--name-only", "--diff-filter=ACMRT", "--cached"]))
    return files


def all_tracked_files():
    return set(_git_lines(["ls-files"]))


def is_skipped(rel, lang):
    posix = rel.replace(os.sep, "/")
    name = posix.rsplit("/", 1)[-1]
    if any(m in name for m in SKIP_NAME_MARKERS):
        return True
    return lang != "py" and any(posix.startswith(p) for p in SKIP_PREFIXES)


def classify(rel):
    ext = Path(rel).suffix.lower()
    if ext in PY_EXT:
        return "py"
    if ext in CPP_EXT:
        return "cpp"
    if ext in FORT_EXT:
        return "fortran"
    return None


# Each checker returns True when the file NEEDS formatting (an offender). With
# fix=True it additionally applies the formatter in place -- so the caller's count
# of "True"s is accurate in both modes (check: how many fail; fix: how many were
# reformatted).
def _needs_format_cpp(path, fix):
    needs = _run(["clang-format", "--dry-run", "-Werror", path]).returncode != 0
    if fix and needs:
        _run(["clang-format", "-i", path])
    return needs


def _needs_format_fortran(path, fix):
    cfg = str(REPO_ROOT / ".fprettify.rc")
    needs = bool(_run(["fprettify", "--config", cfg, "--diff", path]).stdout.strip())
    if fix and needs:
        _run(["fprettify", "--config", cfg, path])
    return needs


CHECKERS = {"cpp": (_needs_format_cpp, "clang-format"), "fortran": (_needs_format_fortran, "fprettify")}

#: The formatter each language is gated by, for the "missing tool" check and the offender report.
TOOLS = {"py": "ruff", "cpp": "clang-format", "fortran": "fprettify"}


def ruff_offenders(rels, fix):
    """Files ``ruff format`` would rewrite, reformatted in place first when ``fix``.

    ``--check`` names them (``Would reformat: <path>``) without writing, and is what the report
    needs in both modes: ruff's own fix output counts what it changed but does not name it.
    """
    if not rels:
        return []
    check = ["ruff", "format", "--check", "--output-format", "json", "--line-length", str(PY_LINE_LENGTH), *rels]
    out = _run(check)
    # json, not the prose. ruff's concise/full output changed shape between releases -- 0.16.5 emits
    # "unformatted: File would be reformatted" with the path on a following "--> path:line:col"
    # line, so a parser keyed on "Would reformat:" finds NOTHING and the gate passes every file
    # silently, in check AND in --fix. Measured on 0.16.5: prose parser 0 offenders, json parser 1.
    try:
        offenders = sorted({entry["filename"] for entry in json.loads(out.stdout or "[]")})
    except json.JSONDecodeError:
        # `--output-format` is PREVIEW-GATED on the formatter, so a ruff that does not honour it
        # prints prose here ("1 file already formatted") and the parse above dies -- taking the
        # pre-commit hook down with it on 0.15.10. The per-file EXIT CODE is the one shape that has
        # not moved across releases, so fall back to it: a process per file is slower than the one
        # batched call, but it is only reached on a ruff whose json this cannot read.
        offenders = sorted(
            rel
            for rel in rels
            if _run(["ruff", "format", "--check", "--line-length", str(PY_LINE_LENGTH), rel]).returncode != 0
        )
    if fix and offenders:
        _run(["ruff", "format", "--line-length", str(PY_LINE_LENGTH), *offenders])
    return offenders


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main", help="git ref to diff against (default: origin/main)")
    ap.add_argument("--all", action="store_true", help="check every tracked source file, not just changed ones")
    ap.add_argument("--fix", action="store_true", help="reformat offending files in place instead of failing")
    ap.add_argument("files", nargs="*", help="explicit files to check (overrides --base/--all)")
    ap.add_argument(
        "--jobs",
        type=int,
        default=min(32, (os.cpu_count() or 1) + 4),
        help="formatter processes to run at once (default: one per core, capped)",
    )
    args = ap.parse_args(argv)

    if args.files:
        candidates = set(args.files)
    elif args.all:
        candidates = all_tracked_files()
    else:
        candidates = changed_files(args.base)

    # Group existing, in-scope files by language.
    by_lang = {"py": [], "cpp": [], "fortran": []}
    for rel in sorted(candidates):
        lang = classify(rel)
        if lang is None or is_skipped(rel, lang) or not (REPO_ROOT / rel).is_file():
            continue
        by_lang[lang].append(rel)

    # Fail fast if a needed formatter is missing.
    missing = [TOOLS[lang] for lang, files in by_lang.items() if files and shutil.which(TOOLS[lang]) is None]
    if missing:
        print(
            f"error: missing formatter(s): {', '.join(sorted(set(missing)))} (pip install ruff fprettify clang-format)",
            file=sys.stderr,
        )
        return 2

    # One formatter PROCESS per file, so the cost is process spawns, not python: 664 in-scope files
    # took 105s serially and timed out pre-commit runs outright. Threads, not processes -- every
    # worker is blocked in subprocess.run with the GIL released, and a thread pool needs no pickling
    # and keeps the offender list in one address space. Each file is independent: the formatters
    # read and rewrite one path, so concurrent workers never touch the same file.
    offenders = [(rel, "ruff") for rel in ruff_offenders(by_lang["py"], args.fix)]
    work = [(lang, rel) for lang in ("cpp", "fortran") for rel in by_lang[lang]]
    if work:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(CHECKERS[lang][0], str(REPO_ROOT / rel), args.fix): (lang, rel) for lang, rel in work
            }
            hits = {futures[f] for f in concurrent.futures.as_completed(futures) if f.result()}
        # Report in the deterministic language-then-path order the serial loop produced, NOT in
        # completion order, so the same tree always prints the same list.
        offenders += [(rel, CHECKERS[lang][1]) for lang, rel in work if (lang, rel) in hits]

    n_checked = sum(len(f) for f in by_lang.values())
    if not offenders:
        print(f"format-check: {n_checked} changed source file(s) OK" + (" (reformatted in place)" if args.fix else ""))
        return 0
    if args.fix:
        print(f"format-check: reformatted {len(offenders)} file(s) in place")
        return 0

    print(f"format-check: {len(offenders)} of {n_checked} changed source file(s) need formatting:\n")
    for rel, tool in offenders:
        print(f"  [{tool}] {rel}")
    print("\nFix with:  python scripts/check_format.py --fix")
    return 1


if __name__ == "__main__":
    sys.exit(main())
