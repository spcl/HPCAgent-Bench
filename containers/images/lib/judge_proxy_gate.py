#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build gate: the launch venv carries the exact pins of pyproject.toml's ``judge-proxy`` group. The group is the one
place these versions are written, so a bump there never has to touch an image."""

import importlib.metadata
import pathlib
import re
import sys
import tomllib

PYPROJECT = pathlib.Path("/opt/hpcagent-bench/pyproject.toml")
PIN = re.compile(r"([\w.-]+)(?:\[[\w,-]+\])?==(\S+)")


def main() -> int:
    pins = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["dependency-groups"]["judge-proxy"]
    matches = [PIN.fullmatch(pin) for pin in pins]
    if None in matches:
        sys.exit(f"judge-proxy pins are not all exact: {pins}")
    want = {match[1]: match[2] for match in matches if match is not None}
    got = {name: importlib.metadata.version(name) for name in want}
    if got != want:
        sys.exit(f"judge-proxy: pyproject.toml pins {want}, the venv holds {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
