#!/usr/bin/env python
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fail on a Markdown link to a repo path that does not exist.

Scans README.md, CONTRIBUTING.md and docs/**/*.md (fenced code blocks skipped) for inline links
``[text](path)`` and reference definitions ``[label]: path``, and resolves each relative path
against the file that holds it. External URLs and bare ``#anchor`` links are ignored, a
``#fragment`` is stripped. Sphinx (``sphinx-build -W``) checks anchors between docs pages.

Output: ``file:line: broken link -> target``, one per hit; exit 1 when any is reported.
"""

import re
import sys
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT_FILES = ("README.md", "CONTRIBUTING.md")
INLINE = re.compile(r"\]\(<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\)")
DEFINITION = re.compile(r"^\s*\[[^\]]+\]:\s*<?(\S+?)>?\s*$")
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|#|//)", re.IGNORECASE)


def markdown_files() -> Iterator[Path]:
    for name in ROOT_FILES:
        yield REPO / name
    yield from sorted((REPO / "docs").rglob("*.md"))


def targets(path: Path) -> Iterator[tuple[int, str]]:
    fenced = False
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        elif not fenced:
            found = INLINE.findall(line)
            if match := DEFINITION.match(line):
                found.append(match.group(1))
            yield from ((number, target) for target in found)


def broken(path: Path) -> Iterator[str]:
    for number, target in targets(path):
        if EXTERNAL.match(target):
            continue
        if not (path.parent / target.split("#")[0]).resolve().exists():
            yield f"{path.relative_to(REPO)}:{number}: broken link -> {target}"


def main() -> int:
    problems = [problem for path in markdown_files() if path.exists() for problem in broken(path)]
    if problems:
        print("\n".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
