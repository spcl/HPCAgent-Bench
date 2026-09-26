# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""hpcagent_bench.stats.figures.results draws with named colours only."""

import inspect
import re

from hpcagent_bench.stats.figures import results as plotting


def test_result_figures_carry_no_literal_hue() -> None:
    """Every colour comes from a named palette or ``style.STAT_INK`` constant, never a bare hex
    string re-typed at a call site."""
    assert re.findall(r"#[0-9a-fA-F]{6}\b", inspect.getsource(plotting)) == []
