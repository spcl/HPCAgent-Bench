# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Enums are plain ``enum.Enum``: a string boundary converts with ``Kind(value)`` and writes ``.value``.

A ``StrEnum`` member compares equal to its string, so a caller passing ``"latest"`` and a CSV holding
``str(member)`` both keep working by accident; with a plain enum the conversion is explicit.
"""

import subprocess

from hpcagent_bench import paths

ROOT = paths.ROOT


def test_no_python_file_uses_str_enum() -> None:
    probed = subprocess.run(
        ["git", "grep", "-n", "-E", r"\bStrEnum\b", "--", "*.py", ":!tests/test_no_str_enum.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # rc 1 = "no matches" (the good case); anything else means the scan never ran -- a tarball
    # checkout or a locked index must fail loudly, not report success over an empty stdout.
    assert probed.returncode in (0, 1), f"git grep failed rc={probed.returncode}:\n{probed.stderr}"
    assert not probed.stdout, f"use enum.Enum, not StrEnum:\n{probed.stdout}"
