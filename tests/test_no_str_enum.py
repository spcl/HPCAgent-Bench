# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Enums are plain ``enum.Enum``: a string boundary converts with ``Kind(value)`` and writes ``.value``.

A ``StrEnum`` member compares equal to its string, so a caller passing ``"latest"`` and a CSV holding
``str(member)`` both keep working by accident; with a plain enum the conversion is explicit.
"""

import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_no_python_file_uses_str_enum() -> None:
    found = subprocess.run(
        ["git", "grep", "-n", "-E", r"\bStrEnum\b", "--", "*.py", ":!tests/test_no_str_enum.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert not found, f"use enum.Enum, not StrEnum:\n{found}"
