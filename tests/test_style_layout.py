# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The measured-layout layer of :mod:`hpcagent_bench.stats.style`: the protrusions every figure
module sizes its chrome from, the crowded-tick shrink, the mark boxes, the save-time placement of
:data:`~hpcagent_bench.stats.style.CLEAR_GID` labels, and the ratio formatters."""

import logging

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.text import Annotation

from hpcagent_bench.stats import style


@pytest.mark.parametrize(
    ("value", "want"),
    [
        pytest.param(6.34919, "6.3x", id="two-figures"),
        pytest.param(32.5, "32x", id="no-decimal-past-ten"),
        pytest.param(0.9166, "0.92x", id="below-one"),
        pytest.param(1.0, "1x", id="unity"),
        pytest.param(0.0, "", id="no-ratio"),
    ],
)
def test_a_ratio_beside_its_mark_prints_two_significant_figures(value: float, want: str) -> None:
    assert style.ratio_label(value) == want


def test_a_log2_axis_tick_reads_back_as_the_ratio() -> None:
    assert (style.log2_ratio_tick(2.0), style.log2_ratio_tick(-1.0)) == ("4x", "0.5x")


def test_a_longer_y_label_widens_the_measured_left_protrusion() -> None:
    """The left margin is measured from the Y labels; a fixed fraction clipped a long one."""
    widths = []
    for label in ("Speed-Up", "Token Cost, Billed\n(1, 0.1, 1)\n(lower is better)"):
        fig, ax = plt.subplots()
        ax.set_ylabel(label)
        ax.set_yticks([1, 10, 100], ["1x", "10x", "0.00391x"])
        fig.canvas.draw()
        widths.append(style.left_protrusion_in(fig, ax))
        plt.close(fig)
    assert widths[1] > widths[0] > 0.0


def test_rotated_long_names_deepen_the_measured_band_below() -> None:
    depths = []
    for name in ("C", "Ragged Segmented Reduction"):
        fig, ax = plt.subplots()
        ax.set_xticks([0], [name], rotation=90)
        fig.canvas.draw()
        depths.append(style.below_protrusion_in(fig, ax))
        plt.close(fig)
    assert depths[1] > depths[0]


def crowded_axes(names: list[str], width_in: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width_in, 1.0))
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_xticks(range(len(names)), names)
    fig.canvas.draw()
    return fig, ax


def test_crowded_tick_labels_step_down_until_no_two_touch() -> None:
    fig, ax = crowded_axes(["Fortran"] * 6, 1.4)
    size = style.shrink_crowded_ticks(fig, [ax], 9.0, 3.0)
    assert size < 9.0
    assert not style.crowded_ticks(ax, fig.canvas.get_renderer(), size / 3.0 * fig.dpi / 72.0)
    plt.close(fig)


def test_tick_labels_that_already_fit_keep_their_size() -> None:
    fig, ax = crowded_axes(["C", "C"], 4.0)
    assert style.shrink_crowded_ticks(fig, [ax], 7.0, 3.0) == 7.0
    plt.close(fig)


def test_labels_still_crowded_at_the_floor_are_reported(caplog: pytest.LogCaptureFixture) -> None:
    """An overprint at the floor is a layout the caller has to change; saying nothing hid it."""
    fig, ax = crowded_axes(["Ragged Segmented Reduction"] * 8, 1.0)
    with caplog.at_level(logging.WARNING, logger=style.LOG.name):
        assert style.shrink_crowded_ticks(fig, [ax], 7.0, 6.0) == 6.0
    assert "floor" in caplog.text
    plt.close(fig)


def test_mark_boxes_takes_an_errorbar_cap_drawn_from_python_lists_on_a_log_axis() -> None:
    """A cap's data arrives as mixed int/float lists, which stacked into an OBJECT array the log
    transform could not take, and the whole save failed."""
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.errorbar([0], [200_000], yerr=[[10_000], [500_000]], fmt="none", capsize=3.0)
    ax.scatter([1], [300_000])
    fig.canvas.draw()
    boxes = style.mark_boxes(ax)
    assert len(boxes) >= 3  # the scatter point, the bar, and its caps
    plt.close(fig)


def clear_label(ax: plt.Axes, x: float, y: float, text: str = "3.1x") -> Annotation:
    return ax.annotate(text, xy=(x, y), xytext=(0.0, 2.0), textcoords="offset points", ha="center",
                       va="bottom", gid=style.CLEAR_GID, annotation_clip=False)  # fmt: skip


def settled(fig: plt.Figure) -> None:
    style.settle_clear_labels(fig)
    fig.canvas.draw()


def test_a_label_drawn_over_a_mark_settles_off_it() -> None:
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 10)
    ax.scatter([0.0], [5.0], s=200)
    label = clear_label(ax, 0.0, 4.8)
    settled(fig)
    box = label.get_window_extent(fig.canvas.get_renderer())
    assert not any(box.overlaps(mark) for mark in style.mark_boxes(ax))
    plt.close(fig)


def test_a_label_above_the_frame_settles_inside_it() -> None:
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 10)
    label = clear_label(ax, 0.0, 9.9)
    settled(fig)
    renderer = fig.canvas.get_renderer()
    box, frame = label.get_window_extent(renderer), ax.get_window_extent(renderer)
    assert frame.y0 <= box.y0 and box.y1 <= frame.y1
    plt.close(fig)


def test_two_labels_drawn_at_one_place_settle_apart() -> None:
    """Each label is wider than a narrow column, so two neighbouring ones were raised to the same
    height and printed on top of each other."""
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 10)
    first, second = clear_label(ax, 0.0, 3.0, "6.3x"), clear_label(ax, 0.05, 3.0, "0.92x")
    settled(fig)
    renderer = fig.canvas.get_renderer()
    assert not first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer))
    plt.close(fig)
