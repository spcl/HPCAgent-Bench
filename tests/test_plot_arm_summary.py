# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``statistics/plot_arm_summary.py`` -- where an arm's point lands on the speedup axis.

Speedup is a ratio, so its "overall" value is the GEOMETRIC MEAN over kernels
(:func:`hpcagent_bench.stats.population.kernel_medians`), the same rule every other "overall
speedup" in this repo follows (:class:`~hpcagent_bench.stats.population.ArmAggregate`). A median
of per-kernel speedups equals the geomean only when the per-kernel values happen to be symmetric,
so the two statistics have to be told apart by an asymmetric fixture, not merely computed and
compared against each other.
"""

import importlib.util
import math
import pathlib
import sys
import types

import pandas as pd
import pytest

from hpcagent_bench import experiment_tags
from hpcagent_bench.stats import palette
from hpcagent_bench.stats import style as plotstyle

REPO = pathlib.Path(__file__).resolve().parents[1]


def load_script() -> types.ModuleType:
    """Import ``statistics/plot_arm_summary.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("plot_arm_summary", REPO / "statistics" / "plot_arm_summary.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plot = load_script()

#: Per-kernel speedups whose geomean and median disagree, at (or above) the interval floor
#: (summary.MIN_PAIRS_FOR_INTERVAL = 6) so ``rules.require_interval`` does not reject the whole
#: table for being too thin to say anything either way: five kernels flat at 1.0x, one at 1000x.
#: Median = 1.0x (log2 = 0); geomean = 1000**(1/6) (log2 = 1.66...).
ASYMMETRIC_SPEEDUPS: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1000.0)


def arm_frame(speedups: tuple[float, ...]) -> pd.DataFrame:
    """One arm, one kernel per speedup, one episode each -- the shape ``arm_points`` groups over."""
    rows = []
    for index, value in enumerate(speedups):
        kernel = f"k{index}"
        run = f"w{index}"
        common = {
            "arm": "demo-arm",
            "model": "qwen38",
            "language": "c",
            "condition": "",
            "benchmark": kernel,
            "run_root": run,
            "job": run,
            "run_id": run,
            "attempt_index": 1,
            "baseline": "numba",
            "suspect": 0,
            "timing_reduction": "mwd-v2",
        }
        rows.append(
            {
                **common,
                "record": "submission",
                "speedup": value,
                "ts_ms": 1,
                "tokens": None,
                "baseline_ns": 1000.0,
                "native_ns": 1000.0 / value,
            }
        )
        rows.append(
            {
                **common,
                "record": "task",
                "speedup": None,
                "ts_ms": 2,
                "tokens": 100.0,
                "tokens_fresh_input": 100.0,
                "tokens_cached_input": 0.0,
                "tokens_output": 0.0,
                "baseline_ns": 0.0,
                "native_ns": 0.0,
            }
        )
    return pd.DataFrame(rows)


def test_an_arm_points_speed_up_is_the_geomean_over_kernels_not_the_median() -> None:
    frame = arm_frame(ASYMMETRIC_SPEEDUPS)
    table = plot.arm_points(frame)
    assert len(table) == 1
    row = table.iloc[0]
    expected_geomean = math.prod(ASYMMETRIC_SPEEDUPS) ** (1.0 / len(ASYMMETRIC_SPEEDUPS))
    assert row.log2_speedup == pytest.approx(math.log2(expected_geomean))
    median_log2 = math.log2(sorted(ASYMMETRIC_SPEEDUPS)[len(ASYMMETRIC_SPEEDUPS) // 2])
    assert row.log2_speedup != pytest.approx(median_log2)


def test_an_arm_points_tokens_are_the_geomean_over_kernels() -> None:
    """Paper rule: the spend axis is the geomean of billed tokens over kernels, 100 on every kernel here."""
    frame = arm_frame(ASYMMETRIC_SPEEDUPS)
    table = plot.arm_points(frame)
    row = table.iloc[0]
    assert row.tokens == pytest.approx(100.0)


def test_an_arm_short_of_the_roster_is_not_drawn() -> None:
    """An arm missing a roster kernel would be scored over a smaller kernel set than its neighbours on
    the same axes, so it is dropped unless the caller explicitly includes incomplete arms."""
    complete = arm_frame(ASYMMETRIC_SPEEDUPS)
    short = arm_frame(ASYMMETRIC_SPEEDUPS[:-1]).assign(arm="short-arm")
    rows = pd.concat([complete, short], ignore_index=True)
    assert set(plot.eligible_rows(rows).arm) == {"demo-arm"}
    assert set(plot.eligible_rows(rows, include_incomplete=True).arm) == {"demo-arm", "short-arm"}


# ---------------------------------------------------------------------------
# The drawing conventions, pinned: colour is the packet, shape is the model, the measured value is
# on Y, the legend belongs to the FIGURE, and the grid is major only.


def two_condition_points() -> pd.DataFrame:
    """One (model, language) with the packet off and on -- ``draw_metric``'s two-condition shape."""
    return pd.DataFrame(
        [
            {"model": "qwen38", "language": "c", "condition": condition, **point}
            for condition, point in (
                ("", {"log2_speedup": 1.0, "log2_speedup_low": 0.8, "log2_speedup_high": 1.2}),
                ("cpf", {"log2_speedup": 1.6, "log2_speedup_low": 1.4, "log2_speedup_high": 1.8}),
            )
        ]
    )


def test_the_measured_value_is_on_the_y_axis_and_the_language_is_the_x_category() -> None:
    """A speedup is a measured quantity and never sits on X; the x slots are LANGUAGES, which are
    names, so they carry no scale and no grid of their own."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        plot.draw_metric(ax, two_condition_points(), "log2_speedup", "Speedup", log=False)
        assert ax.get_ylabel() == "Speedup"
        assert [tick.get_text() for tick in ax.get_xticklabels()] == [experiment_tags.language_name("c")]
        assert not [tick for tick in ax.xaxis.get_major_ticks() if tick.gridline.get_visible()]
    finally:
        plt.close(fig)


def test_the_treated_mark_wears_the_packet_colour_and_the_control_mark_the_control_colour() -> None:
    """The one colour rule: a packet's hue comes from ``palette.color`` and the control from
    ``palette.control_color``, so both mean the same thing in every figure in the repo."""
    import matplotlib.colors
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    fig, ax = plt.subplots()
    try:
        plot.draw_metric(ax, two_condition_points(), "log2_speedup", "Speedup", log=False)
        faces, edges = set(), set()
        for collection in (c for c in ax.collections if isinstance(c, PathCollection)):
            faces.update(matplotlib.colors.to_hex(rgba) for rgba in collection.get_facecolor())
            edges.update(matplotlib.colors.to_hex(rgba) for rgba in collection.get_edgecolor())
    finally:
        plt.close(fig)
    assert palette.color("cpf") in faces
    assert palette.control_color() in edges
    assert palette.model_color("qwen38") not in faces


def test_the_value_axis_carries_a_major_grid_and_no_minor_one() -> None:
    """A major grid and no minor one on a LINEAR value axis (``log=False``): ``style.value_axis``
    cannot know a linear axis' units, so it rules minors only on log axes (rule four)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        plot.draw_metric(ax, two_condition_points(), "log2_speedup", "Speedup", log=False)
        assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
        assert not [tick for tick in ax.yaxis.get_minor_ticks() if tick.gridline.get_visible()]
    finally:
        plt.close(fig)


def test_the_pair_figure_draws_one_legend_on_the_figure_and_none_on_its_axes() -> None:
    """Two panels, ONE key: both draw the same models in the same colours, and a legend per axes
    invites reading them as two different sets of series."""
    import matplotlib.pyplot as plt

    frame = two_condition_points().assign(
        tokens=[1000.0, 900.0], tokens_low=[900.0, 800.0], tokens_high=[1100.0, 1000.0]
    )
    fig, axes = plt.subplots(1, 2)
    try:
        for ax, (column, label, log) in zip(axes, (plot.SPEEDUP, plot.TOKENS), strict=True):
            plot.draw_metric(ax, frame, column, label, log)
        plotstyle.legend_below(fig, plot.handles_for(frame), y=0.005)
        assert len(fig.legends) == 1
        assert all(ax.get_legend() is None for ax in fig.axes)
    finally:
        plt.close(fig)
