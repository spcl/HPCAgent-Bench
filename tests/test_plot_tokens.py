# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``statistics/plot_tokens.py`` -- what one cell of the token figure is a measure of.

A kernel's cost is the tokens the arm spent on it, the SUM over the tasks that ran it
(:func:`hpcagent_bench.stats.population.kernel_tokens`). That is the quantity `plot_arm_summary.py`,
`plot_score_change.py` and `statistics/paired_arms.py` all cost a kernel at, and a median over the
EPISODES inside a cell is a different number with a different unit -- so this file pins the cell to
the sum with a fixture whose two statistics cannot be mistaken for each other.
"""

import importlib.util
import pathlib
import sys
import types

import pandas as pd
import pytest
from matplotlib.figure import Figure

REPO = pathlib.Path(__file__).resolve().parents[1]


def load_script() -> types.ModuleType:
    """Import ``statistics/plot_tokens.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("plot_tokens", REPO / "statistics" / "plot_tokens.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plot_tokens = load_script()

#: Two ARMS of one model ran ``k1``, spending 100 and 700; one arm ran ``k2``, spending 50. The two
#: statistics are far apart on purpose: the kernel's cost is 800 and the median episode is 400.
TASK_ROWS = [
    ("qwen38-a", "k1", "r1", 100, 1_000),
    ("qwen38-b", "k1", "r2", 700, 2_000),
    ("qwen38-a", "k2", "r3", 50, 1_000),
]


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_root": "rr",
                "job": 1,
                "run_id": run_id,
                "record": "task",
                "arm": arm,
                "model": "qwen38",
                "benchmark": benchmark,
                "tokens": tokens,
                "ts_ms": ts_ms,
            }
            for arm, benchmark, run_id, tokens, ts_ms in TASK_ROWS
        ]
    )


def test_a_cell_is_the_kernels_spend_not_the_median_of_its_episodes() -> None:
    """The figure used to plot ``median_episode_tokens``, so the same kernel read 400 here and 800
    in every other figure and table of the same campaign."""
    cells = plot_tokens.cells(frame()).set_index("benchmark")

    assert cells.loc["k1", "kernel_tokens"] == 800.0, "the kernel's cost is the sum over its tasks"
    assert cells.loc["k1", "kernel_tokens"] != 400.0, "not the median over the episodes in the cell"
    assert cells.loc["k2", "kernel_tokens"] == 50.0
    assert "median_episode_tokens" not in cells.columns


def test_a_cell_keeps_the_episodes_its_cost_is_made_of() -> None:
    """The CSV beside the figure has to be re-analysable without the observations file, and the
    parts must add up to the published number."""
    cells = plot_tokens.cells(frame()).set_index("benchmark")

    assert cells.loc["k1", "episodes"] == [100.0, 700.0]
    assert cells.loc["k1", "n"] == 2
    assert sum(cells.loc["k1", "episodes"]) == cells.loc["k1", "kernel_tokens"]


def test_a_rerun_of_one_arm_is_not_charged_twice() -> None:
    """``kernel_tokens`` reduces an arm's repeats to its LATEST run, so resubmitting an arm does not
    make its kernels look more expensive. The episode list must be reduced with it, or the CSV's
    parts would no longer sum to the figure's number."""
    rows = frame()
    rerun = rows[rows["benchmark"] == "k1"].head(1).assign(run_id="r1-again", tokens=10, ts_ms=9_000)
    cells = plot_tokens.cells(pd.concat([rows, rerun], ignore_index=True)).set_index("benchmark")

    assert cells.loc["k1", "kernel_tokens"] == 10.0 + 700.0, "arm a's latest run replaces its first"
    assert cells.loc["k1", "episodes"] == [10.0, 700.0]
    assert sum(cells.loc["k1", "episodes"]) == cells.loc["k1", "kernel_tokens"]


def test_the_axis_names_the_quantity_the_cell_holds(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A figure whose axis says "per Episode" over per-kernel sums is a mislabelled figure, which is
    worse than either quantity on its own. ``draw`` closes the figure it wrote, so the label is read
    off it on the way out."""
    labels: list[str] = []
    closed = plot_tokens.plt.close

    def record(figure: Figure) -> None:
        labels.extend(axis.get_ylabel() for axis in figure.axes)
        closed(figure)

    monkeypatch.setattr(plot_tokens.plt, "close", record)
    out = plot_tokens.draw(plot_tokens.cells(frame()), "test arm", tmp_path / "tokens.pdf")

    assert out.is_file() and out.with_suffix(".png").is_file()
    assert labels == ["Tokens per Kernel"], labels
