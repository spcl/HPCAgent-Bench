# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The repository's files, for the tests that scan every one of them."""

import os
import pathlib
import subprocess


def tracked_files(root: pathlib.Path) -> list[str]:
    """``git ls-files`` of ``root``; where git cannot read it (a ``git archive`` export, or a worktree whose
    ``.git`` lies outside a container's mounts), every file under it but its dot-directories and bytecode, which
    is what such a tree holds."""
    done = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=False)
    if done.returncode == 0:
        return done.stdout.split()
    found: list[str] = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if not name.startswith(".") and name != "__pycache__"]
        found += [str(pathlib.Path(folder, name).relative_to(root)) for name in files]
    return found
