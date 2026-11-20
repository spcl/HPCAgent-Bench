# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The speedup figure's two marks (scatter, bar) draw the same interval, and a framework column registered at
runtime plots in its own slot's colour and marker."""

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.patches import Rectangle

from hpcagent_bench import vocabulary
from hpcagent_bench.stats import palette
from hpcagent_bench.stats.figures import efficacy, kernel_speedups, marks
from tests import test_plot_score_change as fixtures

#: Kernels per setup: past ``summary.MIN_PAIRS_FOR_INTERVAL``, so every mark carries its interval.
KERNELS: int = 8


def synthetic_rows() -> list[efficacy.SetupRow]:
    """One (model, delivery) pair whose per-kernel speedups spread, so each interval has width."""
    control = fixtures.observation_rows(fixtures.BLIND_PAIR[1], 2.0, 100e3, kernels=KERNELS)
    treated = fixtures.observation_rows(fixtures.BLIND_PAIR[0], 4.0, 50e3, kernels=KERNELS)
    for row in control + treated:
        if row["row_kind"] == "submission":
            spread = 1.0 + 0.1 * int(str(row["kernel"])[1:])
            row["speedup"] = float(row["speedup"]) * spread  # pyright: ignore[reportArgumentType]
            row["native_ns"] = float(row["baseline_ns"]) / float(row["speedup"])  # pyright: ignore[reportArgumentType]
    frame = fixtures.plot.pair_frame(pd.DataFrame(control + treated), [fixtures.BLIND_PAIR], "no-score-tool")
    return efficacy.setup_rows(frame)


def drawn_speedup_row(mark: marks.Mark) -> tuple[list[efficacy.SetupRow], plt.Axes, efficacy.FigureConfig]:
    """The speedup row of :func:`synthetic_rows` drawn under ``mark``, intervals never cut."""
    rows = synthetic_rows()
    config = dataclasses.replace(efficacy.DEFAULT_CONFIG, mark=mark, interval_reach=2.0**20)
    unused_fig, ax = plt.subplots()
    efficacy.draw_measure_row(ax, rows, "speedup", "s", {}, config)
    return rows, ax, config


def expected_intervals(rows: list[efficacy.SetupRow], config: efficacy.FigureConfig) -> list[tuple[float, ...]]:
    """``(x, low, high)`` of every setup's speedup CI, at the x it is drawn at."""
    out: list[tuple[float, ...]] = []
    for x, row in zip(efficacy.column_x(rows), rows, strict=True):
        for point, dodge in ((row.control, -config.dodge), (row.treated, config.dodge)):
            unused_value, low, high = efficacy.measure_value(point, "speedup")
            out.append((x + dodge, low, high))
    return sorted(out)


def drawn_intervals(ax: plt.Axes) -> list[tuple[float, ...]]:
    """``(x, low, high)`` of every vertical interval line on ``ax``."""
    segments = [seg for coll in ax.collections if isinstance(coll, LineCollection) for seg in coll.get_segments()]
    return sorted((float(s[0][0]), float(min(s[:, 1])), float(max(s[:, 1]))) for s in segments if s[0][0] == s[1][0])


def flat(pairs: list[tuple[float, ...]]) -> list[float]:
    """``pairs`` as one flat list, which ``pytest.approx`` compares element by element."""
    return [float(value) for pair in pairs for value in pair]


def test_the_scatter_form_draws_a_marker_at_each_estimate_and_its_ci_as_the_error_bar() -> None:
    rows, ax, config = drawn_speedup_row(marks.Mark.SCATTER)
    expected = expected_intervals(rows, config)
    assert all(high > low for unused_x, low, high in expected), expected
    assert flat(drawn_intervals(ax)) == pytest.approx(flat(expected))
    values = sorted(efficacy.measure_value(point, "speedup")[0] for row in rows for point in (row.control, row.treated))
    offsets = [tuple(o) for c in ax.collections if isinstance(c, PathCollection) for o in c.get_offsets()]
    assert sorted({y for unused_x, y in offsets}) == pytest.approx(values)
    assert not [patch for patch in ax.patches if isinstance(patch, Rectangle) and patch.get_width() < 1.0]


def test_the_bar_form_draws_one_bar_per_estimate_and_its_ci_as_the_error_bar() -> None:
    rows, ax, config = drawn_speedup_row(marks.Mark.BAR)
    assert flat(drawn_intervals(ax)) == pytest.approx(flat(expected_intervals(rows, config)))
    bars = sorted(
        (patch.get_x() + patch.get_width() / 2.0, patch.get_y() + patch.get_height())
        for patch in ax.patches
        if isinstance(patch, Rectangle) and patch.get_width() == pytest.approx(config.bar_width)
    )
    expected = sorted(
        (x + dodge, efficacy.measure_value(point, "speedup")[0])
        for x, row in zip(efficacy.column_x(rows), rows, strict=True)
        for point, dodge in ((row.control, -config.dodge), (row.treated, config.dodge))
    )
    assert flat(bars) == pytest.approx(flat(expected))
    assert all(patch.get_y() == 0.0 for patch in ax.patches if isinstance(patch, Rectangle) and patch.get_width() < 1.0)


def test_a_mark_is_spelled_only_at_the_cli_edge() -> None:
    assert marks.Mark.texts() == ("scatter", "bar")
    assert marks.Mark.parse("bar") is marks.Mark.BAR
    with pytest.raises(ValueError, match="scatter"):
        marks.Mark.parse("line")


def test_a_framework_column_registered_at_runtime_gets_its_own_look_and_plots() -> None:
    key = "plugin_column_under_test"
    shipped = vocabulary.FRAMEWORKS.keys()
    entry = dataclasses.replace(vocabulary.FRAMEWORKS[shipped[-1]], display="Plugin Column", column=None, flavor=None)
    vocabulary.FRAMEWORKS.add(key, entry)
    try:
        look = palette.look("frameworks", key)
        assert look == palette.slot_look(vocabulary.FRAMEWORKS.slot(key) or 0)
        assert look.colour != palette.control_color()
        assert look not in {palette.look("frameworks", other) for other in shipped}
        canon = pd.DataFrame(
            [{"run": "r1", "column": column, "kernel": kernel, "median_ms": ms, "validated": 1}
             for column, kernel, ms in (("numba", "k1", 100.0), ("numba", "k2", 200.0),
                                        (key, "k1", 50.0), (key, "k2", 40.0))]
        )  # fmt: skip
        (row,) = kernel_speedups.kernel_rows(canon, None, ["k1", "k2"], [key], baseline="numba")
        assert (row.color, row.marker) == (look.colour, look.marker)
        fig = kernel_speedups.kernel_figure([row], ["k1", "k2"], baseline="numba")
        colours = {
            tuple(np.round(face, 6))
            for ax in fig.axes for coll in ax.collections if isinstance(coll, PathCollection)
            for face in coll.get_facecolors()
        }  # fmt: skip
        assert tuple(np.round(plt.matplotlib.colors.to_rgba(look.colour), 6)) in colours
    finally:
        vocabulary.FRAMEWORKS.entries.pop(key)
        vocabulary.FRAMEWORKS.orders.pop(key)


def test_a_slot_past_the_palette_changes_marker_not_to_grey() -> None:
    hues = len(palette.hues())
    first, wrapped = palette.slot_look(3), palette.slot_look(3 + hues)
    assert first.colour == wrapped.colour
    assert first.marker != wrapped.marker
    assert palette.slot_look(10**6).colour in palette.hues()


if __name__ == "__main__":
    test_the_scatter_form_draws_a_marker_at_each_estimate_and_its_ci_as_the_error_bar()
    test_the_bar_form_draws_one_bar_per_estimate_and_its_ci_as_the_error_bar()
    test_a_mark_is_spelled_only_at_the_cli_edge()
    test_a_framework_column_registered_at_runtime_gets_its_own_look_and_plots()
    test_a_slot_past_the_palette_changes_marker_not_to_grey()
