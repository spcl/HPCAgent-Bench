#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Thin shim -- the logic now lives in the ``optarena`` CLI.

``python scripts/pluto_affine_survey.py`` is equivalent to ``optarena pluto-survey``
(dispatched to :func:`optarena.support.collect.pluto_survey.survey`). Run from the repo root so
the ``tests.numerical_oracle`` package the survey imports is on the path. Kept so the
documented script path keeps working after the fold into the package CLI.
"""
import sys

from optarena.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["pluto-survey", *sys.argv[1:]]))
