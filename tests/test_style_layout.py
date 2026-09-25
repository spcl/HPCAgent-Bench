# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The measured-layout layer of :mod:`hpcagent_bench.stats.style`: the protrusions every figure
module sizes its chrome from, the crowded-tick shrink, the mark boxes, the save-time placement of
:data:`~hpcagent_bench.stats.style.CLEAR_GID` labels, the ratio formatters, and the minor ticks
every value axis shares (:func:`~hpcagent_bench.stats.style.minor_ticks`)."""

import logging
import math
from collections.abc import Callable

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from matplotlib.ticker import FixedLocator, Locator, LogLocator, MultipleLocator

from hpcagent_bench.stats import style


@pytest.mark.parametrize(
    ("value", "want"),
    [
        pytest.param(6.34919, "6.3x", id="one-decimal"),
        pytest.param(32.45, "32.5x", id="one-decimal-past-ten"),
        pytest.param(0.928, "0.9x", id="below-one"),
        pytest.param(1.0, "1.0x", id="unity"),
        pytest.param(0.04, "0.04x", id="below-a-tenth-keeps-a-figure"),
        pytest.param(0.0, "", id="no-ratio"),
    ],
)
def test_a_ratio_beside_its_mark_prints_one_decimal(value: float, want: str) -> None:
    """One decimal is what a reader quotes; below 0.1x one decimal would print a real slowdown as
    0.0x, so those keep a significant figure."""
    assert style.ratio_label(value) == want


def test_a_log2_axis_tick_reads_back_as_the_ratio() -> None:
    assert (style.log2_ratio_tick(2.0), style.log2_ratio_tick(-1.0)) == ("4x", "0.5x")


def test_a_longer_y_label_widens_the_measured_left_protrusion() -> None:
    """The left margin is measured from the Y labels; a fixed fraction clipped a long one."""
    widths = []
    for label in ("Speedup", "Token Cost, Billed\n(1, 0.1, 1)\n(lower is better)"):
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


def log2_of(*ratios: float) -> list[float]:
    """``ratios`` as the exponents a linear ``log2`` axis holds them at."""
    return [math.log2(ratio) for ratio in ratios]


#: (kind, majors, view, the minors drawn): one row per ratio, log2, token and count axis a figure
#: rules, all placed by :class:`~hpcagent_bench.stats.style.MinorLocator`.
MINOR_CASES = [
    pytest.param("ratio", FixedLocator([1.0, 2.0, 4.0, 8.0]), (1.0, 8.0),
                 [1.25, 1.5, 1.75, 2.5, 3.0, 3.5, 5.0, 6.0, 7.0], id="log2-one-octave"),
    pytest.param("ratio", FixedLocator([1.0, 2.0]), (1.0, 2.6), [1.25, 1.5, 1.75, 2.5], id="log2-past-the-last-major"),
    pytest.param("ratio", FixedLocator([1.0, 4.0, 16.0]), (1.0, 16.0), [2.0, 8.0], id="log2-two-octaves"),
    pytest.param("ratio", FixedLocator([0.125, 1.0, 8.0]), (0.1, 10.0), [0.25, 0.5, 2.0, 4.0], id="log2-three-octaves"),
    pytest.param("log2", MultipleLocator(1.0), (0.0, 2.0), log2_of(1.25, 1.5, 1.75, 2.5, 3.0, 3.5),
                 id="linear-log2-one-octave"),
    pytest.param("log2", MultipleLocator(2.0), (-2.0, 4.0), log2_of(0.5, 2.0, 8.0), id="linear-log2-two-octaves"),
    pytest.param("token", LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)), (1e5, 1e6),
                 [3e5, 4e5, 6e5, 7e5, 8e5, 9e5], id="token-1-2-5"),
    pytest.param("token", LogLocator(base=10.0, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)), (1e5, 1e6),
                 [4e5, 6e5, 8e5, 9e5], id="token-config-subs"),
    pytest.param("count", FixedLocator([0.0, 20.0, 40.0]), (-2.0, 42.0), [5.0, 10.0, 15.0, 25.0, 30.0, 35.0],
                 id="count-40"),
    pytest.param("count", FixedLocator([0.0, 5.0, 10.0]), (-0.5, 10.5), [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0],
                 id="count-10"),
    pytest.param("count", FixedLocator([0.0, 9.0]), (-0.5, 9.5), [3.0, 6.0], id="count-9-in-thirds"),
]  # fmt: skip


def ruled_axis(kind: style.MinorKind, majors: Locator, view: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    """An axes whose Y is on ``kind``'s scale with ``majors`` over ``view``, ruled by
    :func:`~hpcagent_bench.stats.style.minor_ticks`."""
    fig, ax = plt.subplots()
    if kind == "ratio":
        ax.set_yscale("log", base=2.0)
    elif kind == "token":
        ax.set_yscale("log")
    ax.yaxis.set_major_locator(majors)
    ax.set_ylim(*view)
    style.minor_ticks(ax.yaxis, kind)
    return fig, ax


@pytest.mark.parametrize(("kind", "majors", "view", "want"), MINOR_CASES)
def test_minor_ticks_fall_where_the_shared_rule_puts_them(
    kind: style.MinorKind, majors: Locator, view: tuple[float, float], want: list[float]
) -> None:
    """User, 2026-09-22: more minor ticks on every value axis. A ratio axis reads its octave spacing
    off the majors (one-octave majors take the quarters, wider ones every octave between), a token
    axis takes every whole multiple of a power of ten, a count axis whole-number parts of a step."""
    fig, ax = ruled_axis(kind, majors, view)
    assert list(ax.yaxis.get_minorticklocs()) == pytest.approx(want)
    plt.close(fig)


@pytest.mark.parametrize(("kind", "majors", "view", "want"), MINOR_CASES)
def test_no_minor_tick_carries_a_label(
    kind: style.MinorKind, majors: Locator, view: tuple[float, float], want: list[float]
) -> None:
    """A number at every minor doubles the axis' text, and matplotlib's own log minor formatter
    prints a scientific-notation 3x10^n beside plain majors."""
    del want
    fig, ax = ruled_axis(kind, majors, view)
    fig.canvas.draw()
    texts = [label.get_text() for label in ax.yaxis.get_minorticklabels()]
    assert texts and not any(texts)
    plt.close(fig)


@pytest.mark.parametrize(("kind", "majors", "view", "want"), MINOR_CASES)
def test_no_minor_tick_lands_on_a_major(
    kind: style.MinorKind, majors: Locator, view: tuple[float, float], want: list[float]
) -> None:
    """A minor on a major draws a second, lighter line through the labelled one. Read off the
    locator itself: the drawn minors cannot show it, since matplotlib strips a minor that sits on a
    major before drawing."""
    del want
    fig, ax = ruled_axis(kind, majors, view)
    located = [float(value) for value in ax.yaxis.get_minor_locator()()]
    assert located
    assert not [value for value in located if any(math.isclose(value, major) for major in ax.yaxis.get_majorticklocs())]
    plt.close(fig)


def test_a_count_step_no_whole_part_divides_gets_no_minor_ticks() -> None:
    """A count is a whole number of tasks: matplotlib's own rule split a 0/7 axis at 1.75, 3.5 and
    5.25 tasks, lines that mark no count at all."""
    fig, ax = ruled_axis("count", FixedLocator([0.0, 7.0]), (-0.5, 7.5))
    assert list(ax.yaxis.get_minorticklocs()) == []
    plt.close(fig)


@pytest.mark.parametrize(
    ("measure", "share"),
    [
        pytest.param(Line2D.get_markersize, 0.55, id="length"),
        pytest.param(Line2D.get_markeredgewidth, 0.6, id="width"),
    ],
)
def test_a_minor_tick_mark_is_a_fixed_share_of_a_major_one(measure: Callable[[Line2D], float], share: float) -> None:
    """The minor marks subdivide the labelled reference: 0.55x the major's length, 0.6x its width.
    matplotlib's own minors are already smaller, so only the exact share shows the rule is applied."""
    fig, ax = ruled_axis("ratio", FixedLocator([1.0, 2.0, 4.0]), (1.0, 4.0))
    fig.canvas.draw()
    major, minor = ax.yaxis.get_major_ticks()[0].tick1line, ax.yaxis.get_minor_ticks()[0].tick1line
    assert measure(minor) == pytest.approx(share * measure(major))
    plt.close(fig)


def test_minors_follow_majors_and_limits_set_after_the_axis_was_ruled() -> None:
    """A figure pins its majors and snaps its limits AFTER styling the axis; minors frozen at styling
    time would sit at the old spacing."""
    fig, ax = ruled_axis("ratio", FixedLocator([1.0, 2.0, 4.0]), (1.0, 4.0))
    ax.set_yticks([1.0, 4.0, 16.0])
    ax.set_ylim(1.0, 16.0)
    assert list(ax.yaxis.get_minorticklocs()) == pytest.approx([2.0, 8.0])
    plt.close(fig)
