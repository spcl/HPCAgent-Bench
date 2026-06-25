#!/usr/bin/env python3
"""Unified YAML-style gate for every optarena-owned YAML file.

ONE house style for all of OptArena's own YAML (manifests, the taxonomy
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
additionally have their canonical key order enforced by the manifest
canonicalizer; this gate is the repo-wide style layer on top.

    python tests/check_yaml_style.py            # CI: report violations, exit 1
    python tests/check_yaml_style.py --fix       # auto-fix the mechanical ones

``--fix`` only touches the safe, lossless bits (trailing whitespace, final
newline) -- it never re-dumps a file, so inline comments and authored key order
are preserved. Missing headers, tabs and odd indentation are reported for a
human to fix (re-indenting or wording a header is not safe to automate).
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
    """Every tracked ``*.yaml`` / ``*.yml`` that is optarena's own to style."""
    out = subprocess.run(["git", "ls-files", "*.yaml", "*.yml"],
                         cwd=REPO, capture_output=True, text=True, check=True)
    files = []
    for rel in out.stdout.split():
        if any(s in f"/{rel}" for s in SKIP):
            continue
        files.append(REPO / rel)
    return files


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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix", action="store_true",
                    help="auto-fix trailing whitespace + final newline (lossless)")
    args = ap.parse_args(argv)

    files = owned_yaml()
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
