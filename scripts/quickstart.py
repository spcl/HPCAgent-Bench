# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Thin shim -- the logic now lives in the ``optarena`` CLI.

``python scripts/quickstart.py <args>`` is equivalent to ``optarena quickstart <args>``
(dispatched to :func:`optarena.collect.quickstart.quickstart`). Kept so the documented
script path keeps working after the fold into the package CLI.
"""
import sys

from optarena.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["quickstart", *sys.argv[1:]]))
