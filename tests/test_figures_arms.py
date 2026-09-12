# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The per-arm figures draw from the tables stats.arms writes, in registered colours."""

import logging
import pathlib

import pandas as pd
import pytest

from hpcagent_bench.stats import palette
from hpcagent_bench.stats.figures import arms


def arm_table() -> pd.DataFrame:
    """The columns ``stats.arms.per_arm_summary`` writes that the figure reads, keyed as it keys them."""
    rows = [
        ("llr40-qwen38-c", "c", 4.0, 3.0, 2.0, 4.5, 40, 30),
        ("llr40-qwen38-fortran", "fortran", 2.0, 1.5, 1.2, 2.4, 40, 22),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "arm",
            "language",
            "geomean_solved",
            "geomean_served",
            "geomean_solved_low",
            "geomean_solved_high",
            "n_served",
            "n_solved",
        ],
    )
    return frame.assign(baseline="numba").set_index(["arm", "baseline"])


def paired_table() -> pd.DataFrame:
    """``stats.arms.per_language_kernel`` keyed on ``(baseline, benchmark)``, one kernel with no Fortran answer."""
    frame = pd.DataFrame(
        {
            "baseline": "numba",
            "benchmark": ["k1", "k2", "k3"],
            "c_best_su": [2.0, 5.0, 1.5],
            "fortran_best_su": [3.0, 4.0, None],
        }
    )
    return frame.set_index(["baseline", "benchmark"])


def test_both_arm_figures_render_in_registered_language_colours(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=palette.LOG.name):
        arms.figure_arms(arm_table(), "numba", tmp_path)
        arms.figure_paired(paired_table(), "numba", tmp_path)
    assert "registry.yaml" not in caplog.text
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == [
        "per_arm_geomean_numba.pdf",
        "per_arm_geomean_numba.png",
        "per_kernel_c_vs_fortran_numba.pdf",
        "per_kernel_c_vs_fortran_numba.png",
    ]
