#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unified YAML-style gate for every hpcagent_bench-owned YAML file.

ONE house style for all of HPCAgent-Bench's own YAML (manifests, the taxonomy
vocabularies, the env/compiler config, the global config):

  1. parses as YAML,
  2. a one-line ``#`` header comment on line 1 (says what the file is / how it
     is generated),
  3. structural indentation in multiples of two spaces,
  4. no tab characters,
  5. no trailing whitespace,
  6. exactly one trailing newline (LF).

Third-party-schema YAML is NOT ours to restyle and is skipped: GitHub Actions
workflows (``.github/``) and docker-compose (``*compose*``). Benchmark manifests
additionally have their SCHEMA checked by ``scripts/check_manifest_structure.py``
(which loads each one through ``BenchSpec.from_yaml``); this gate is the repo-wide
style layer on top, and knows nothing about any file's schema.

    python tests/check_yaml_style.py            # CI / standalone: every tracked owned YAML
    python tests/check_yaml_style.py --fix       # auto-fix the mechanical ones
    python tests/check_yaml_style.py --fix a.yaml b.yaml   # pre-commit: only the given files

``--fix`` only touches the safe, lossless bits (trailing whitespace, final
newline) -- it never re-dumps a file, so inline comments and authored key order
are preserved. Missing headers, tabs and odd indentation are reported for a
human to fix (re-indenting or wording a header is not safe to automate).

Wired into ``.pre-commit-config.yaml`` as ``hpcagent_bench-yaml-style``, which passes the
staged files as positional args (already narrowed by the hook's broad ``\\.ya?ml$`` regex;
:data:`SKIP` is what actually excludes third-party schemas, same as every other hook here
whose script owns its own skip policy). With no files given -- standalone, or the pytest
gate in ``tests/test_yaml_style.py`` -- it scans every tracked owned YAML instead.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent

#: Path fragments whose YAML follows a foreign schema we must not restyle.
SKIP = ("/.github/", "compose")


def owned_yaml() -> list[pathlib.Path]:
    """Every tracked ``*.yaml`` / ``*.yml`` that is hpcagent_bench's own to style (the
    standalone-scan fallback used when no explicit files are given)."""
    out = subprocess.run(["git", "ls-files", "*.yaml", "*.yml"], cwd=REPO, capture_output=True, text=True, check=True)
    return in_scope(out.stdout.split())


def in_scope(rels: list[str]) -> list[pathlib.Path]:
    """Filter candidate repo-relative paths down to the ones this gate owns (drops the
    third-party-schema :data:`SKIP` list). Shared by :func:`owned_yaml`'s standalone
    tree scan and ``main``'s pre-commit staged-file args, so both apply the same policy."""
    return [REPO / rel for rel in rels if not any(s in f"/{rel}" for s in SKIP)]


def violations(path: pathlib.Path) -> list[str]:
    """Style problems in ``path`` (empty list == conforms)."""
    text = path.read_text()
    probs: list[str] = []
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as e:  # noqa: PERF203
        return [f"does not parse: {str(e).splitlines()[0]}"]
    lines = text.splitlines()
    if not lines or not lines[0].lstrip().startswith("#"):
        probs.append("missing a '#' header comment on line 1")
    for i, line in enumerate(lines, 1):
        if "\t" in line:
            probs.append(f"L{i}: tab character")
        if line != line.rstrip():
            probs.append(f"L{i}: trailing whitespace")
        stripped = line.lstrip(" ")
        # Only STRUCTURAL lines must align to 2 spaces; a comment-only line may
        # be column-aligned to a value (intentional) and is exempt.
        if stripped and not stripped.startswith("#"):
            indent = len(line) - len(stripped)
            if indent % 2:
                probs.append(f"L{i}: odd indent ({indent} spaces)")
    if not text.endswith("\n"):
        probs.append("no final newline")
    if text.endswith("\n\n"):
        probs.append("more than one trailing newline")
    return probs


def fix(path: pathlib.Path) -> bool:
    """Apply the lossless fixes (trailing whitespace, single final newline).
    Returns ``True`` if the file changed."""
    text = path.read_text()
    fixed = "\n".join(line.rstrip() for line in text.splitlines())
    fixed = fixed.rstrip("\n") + "\n" if fixed else fixed
    if fixed != text:
        path.write_text(fixed)
        return True
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix", action="store_true", help="auto-fix trailing whitespace + final newline (lossless)")
    ap.add_argument("files",
                    nargs="*",
                    help="files to check (default: every tracked owned YAML); "
                    "pre-commit passes the staged files here")
    args = ap.parse_args(argv)

    # A staged DELETION is still passed as a positional arg by pre-commit; drop paths that
    # no longer exist (the standalone `git ls-files` scan never produces one, so this only
    # matters on the explicit-files branch).
    files = [p for p in in_scope(args.files) if p.is_file()] if args.files else owned_yaml()
    changed = 0
    bad: dict[pathlib.Path, list[str]] = {}
    for f in files:
        if args.fix and fix(f):
            changed += 1
        probs = violations(f)
        if probs:
            bad[f] = probs

    if args.fix:
        print(f"yaml-style: fixed {changed} file(s)")
    for f, probs in sorted(bad.items()):
        print(f"  {f.relative_to(REPO)}:")
        for p in probs:
            print(f"    - {p}")
    print(f"yaml-style: {len(files)} files | {len(bad)} with violations")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
