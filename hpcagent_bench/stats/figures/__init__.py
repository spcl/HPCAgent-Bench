# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The figure builders. Every script that draws something calls one of these.

A builder owns the LAYOUT and nothing else: it takes a table, asks
:mod:`hpcagent_bench.stats.summary` for the numbers, :mod:`hpcagent_bench.stats.palette` for the
colour and the shape, :mod:`hpcagent_bench.stats.style` for the ink, and
:mod:`hpcagent_bench.stats.rules` for whether what it is about to draw is allowed. A script that
computes its own statistic on the way to a figure has made a second definition of that number.

Submodules: :mod:`results` (the DB figures -- speed-up heatmap, distribution grid, per-sample
diagnostics) and :mod:`signed` (the signed-change axis: one row per arm, and the paired
control-to-treatment comparison).

The headless backend is selected HERE, in the package, so it is in force before any submodule binds
pyplot -- the one ordering matplotlib does not let a module fix for itself afterwards.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
