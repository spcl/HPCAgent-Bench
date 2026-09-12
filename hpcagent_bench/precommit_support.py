# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared helpers for the scripts/check_*.py pre-commit hooks.

Each hook falls back to scanning the tracked tree when pre-commit hands it no positional files (a
standalone run, or ``--all-files``); that discovery and the autogen-marker check were three and two
independent copies respectively before this.
"""

from __future__ import annotations

import pathlib
import subprocess


def git_tracked(pattern: str | None = None, cwd: pathlib.Path | None = None) -> list[str]:
    """Tracked file paths, optionally filtered by a ``git ls-files`` pathspec/glob.

    Empty on any git failure (not a repo, no git on PATH) rather than raising -- a hook with
    nothing to scan reports a clean pass, and pre-commit's positional-file mode never calls this.
    """
    args = ["git", "ls-files"] + ([pattern] if pattern else [])
    out = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]


def is_generated_source(path: pathlib.Path) -> bool:
    """A generated sibling carries the autogen marker on its first line."""
    try:
        with path.open(encoding="utf-8") as handle:
            return "hpcagent_bench-autogen" in handle.readline()
    except OSError:
        return False
