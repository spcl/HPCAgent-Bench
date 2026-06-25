# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""CI lint: optimization flags (``-O3`` / ``-march=native`` / ``-ffast-math``)
must come from the central matrix (``optarena/flags.py``), never be
string-literal'd elsewhere in the harness or build scripts.

Allowlisted files legitimately contain the flag text and cannot route through
``flags.py``: the matrix itself; the CMake template + the CMake-emitting scripts
(CMake cannot import Python); the hardware-probe Makefile generators (HPL /
STREAM ship their own build recipes); and the off-limits ``NumpyTo*`` package.
Adding a file here requires a justification in this list.
"""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
_PATTERN = re.compile(r"-O3|-march=native|-ffast-math")
_SCAN_DIRS = ("optarena", "scripts")
_ALLOW = {
    "optarena/flags.py",                                       # the matrix itself
    "optarena/hardware_info/practical/flops_with_linpack.py",  # emits an HPL Makefile
    "optarena/hardware_info/practical/memory_with_stream.py",  # emits a STREAM build
    "scripts/emit_cpp_ports.py",                              # emits CMake text (TODO: route)
    "scripts/emit_c_variants.py",                             # emits CMake text (TODO: route)
    "scripts/pull_cpp.py",                                    # emits CMake text (TODO: route)
}


def _candidates():
    for d in _SCAN_DIRS:
        root = REPO / d
        if not root.is_dir():
            continue
        for ext in ("*.py", "*.cmake"):
            for p in root.rglob(ext):
                rel = p.relative_to(REPO).as_posix()
                if rel in _ALLOW or "/NumpyTo" in "/" + rel:
                    continue
                yield p, rel


def test_no_literal_opt_flags_outside_matrix():
    offenders = []
    for p, rel in _candidates():
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if _PATTERN.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "Literal optimization flags found outside optarena/flags.py -- route them "
        "through the matrix (or allowlist with a justification in this file):\n  "
        + "\n  ".join(offenders))
