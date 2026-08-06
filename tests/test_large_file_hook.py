# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the oversized-file pre-commit guard."""

import importlib.util
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def load_checker():
    spec = importlib.util.spec_from_file_location("check_large_files", REPO / "scripts" / "check_large_files.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_byte_identical_rename_destination_is_detected(monkeypatch):
    checker = load_checker()
    output = "R100\0old/legacy.bin\0archive/legacy.bin\0"
    monkeypatch.setattr(
        checker.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout=output, stderr=""),
    )

    assert checker.unchanged_rename_destinations() == {"archive/legacy.bin"}


def test_modified_rename_is_not_exempt(monkeypatch):
    checker = load_checker()
    output = "R099\0old/legacy.bin\0archive/legacy.bin\0"
    monkeypatch.setattr(
        checker.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout=output, stderr=""),
    )

    assert checker.unchanged_rename_destinations() == set()
