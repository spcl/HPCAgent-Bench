#!/usr/bin/env python
"""Enforce the repo formatters on changed source files (column limit 120).

Routing by extension:
  * Python  (.py)            -> yapf       (.style.yapf, column_limit = 120)
  * C / C++ (.c .cc .cpp ...) -> clang-format (.clang-format, ColumnLimit 120)
  * Fortran (.f90 .F90 ...)   -> fprettify  (.fprettify.rc, line-length 120)

Scope: by default the files changed versus ``--base`` (merge-base with
``origin/main``) plus any staged/working-tree edits, so the CI ``format-check``
job and a local run agree. ``--all`` checks every tracked source file.

Kernel ports / generated references are NOT style-gated (they are faithful ports
or machine-emitted): anything under ``optarena/benchmarks/`` or matching
``*_generated.*`` is skipped (the ``.yapfignore`` policy). ``NumpyTranslators/``
is also skipped -- it is a separate distribution with its own style policy.

Exit status: 0 when every checked file is already formatted; 1 when one or more
need reformatting (the offenders and the fix command are printed); 2 on a setup
error (a needed formatter is missing). ``--fix`` reformats in place instead.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PY_EXT = {".py"}
CPP_EXT = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh", ".hxx"}
FORT_EXT = {".f", ".f90", ".f03", ".f08", ".f95", ".for"}  # matched case-insensitively

# Ported / generated sources are not hand-formatted (faithfulness over style),
# matching the .yapfignore policy. A path is skipped when it sits under one of
# these prefixes or its name marks it as generated. NumpyTranslators is a SEPARATE
# distribution (its own pyproject, excluded from packaging) and carries its own
# style policy, so the repo-root configs do not govern it.
SKIP_PREFIXES = ("optarena/benchmarks/", "optarena/NumpyTranslators/")
SKIP_NAME_MARKERS = ("_generated.", )


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


def is_skipped(rel):
    posix = rel.replace(os.sep, "/")
    if any(posix.startswith(p) for p in SKIP_PREFIXES):
        return True
    name = posix.rsplit("/", 1)[-1]
    return any(m in name for m in SKIP_NAME_MARKERS)


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
def _needs_format_py(path, fix):
    needs = bool(_run(["yapf", "--diff", path]).stdout.strip())
    if fix and needs:
        _run(["yapf", "-i", path])
    return needs


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


CHECKERS = {
    "py": (_needs_format_py, "yapf"),
    "cpp": (_needs_format_cpp, "clang-format"),
    "fortran": (_needs_format_fortran, "fprettify")
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main", help="git ref to diff against (default: origin/main)")
    ap.add_argument("--all", action="store_true", help="check every tracked source file, not just changed ones")
    ap.add_argument("--fix", action="store_true", help="reformat offending files in place instead of failing")
    ap.add_argument("files", nargs="*", help="explicit files to check (overrides --base/--all)")
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
        if is_skipped(rel) or not (REPO_ROOT / rel).is_file():
            continue
        lang = classify(rel)
        if lang is not None:
            by_lang[lang].append(rel)

    # Fail fast if a needed formatter is missing.
    missing = [
        tool for lang, files in by_lang.items() if files for _, tool in [CHECKERS[lang]] if shutil.which(tool) is None
    ]
    if missing:
        print(
            f"error: missing formatter(s): {', '.join(sorted(set(missing)))} "
            f"(pip install yapf fprettify clang-format)",
            file=sys.stderr)
        return 2

    offenders = []
    for lang, files in by_lang.items():
        check, tool = CHECKERS[lang]
        for rel in files:
            if check(str(REPO_ROOT / rel), args.fix):
                offenders.append((rel, tool))

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
