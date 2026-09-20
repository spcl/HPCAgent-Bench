# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-kernel speed-up and per-kernel tokens, drawn compactly enough to sit in a paper.

One column per kernel, either the median with a bootstrap confidence interval (``ci``, the
default) or a boxplot of the per-episode values (``box``). Both modes read the SAME per-episode
population: :func:`hpcagent_bench.stats.population.graded_episode_rows` for speed-up (one row per
episode, its own last reportable submission) and :func:`hpcagent_bench.stats.population.episode_tokens`
for tokens (one row per episode, its own cumulative total) -- so switching ``--style`` never
changes which numbers are behind the figure, only how they are drawn.

MOST EXPERIMENTS RUN ONE EPISODE PER KERNEL, so a kernel's cell is a single value under either
style: ``ci`` draws it as a point (:func:`hpcagent_bench.stats.summary.median_ci` collapses to the
point below its interval floor, :data:`~hpcagent_bench.stats.summary.MIN_INTERVAL_SAMPLES` = 5) and
``box`` draws it as a point too (:data:`MIN_EPISODES_FOR_SPREAD` below). ``git-scicomp`` is the
exception with 3 episodes per kernel: this is DELIBERATELY still under the bootstrap floor, so
``ci`` keeps drawing a plain point there rather than a whisker width nobody would trust; ``box``
draws an honest box at n=3 instead, because a boxplot's quartiles are a real (if coarse) summary of
three raw numbers where a bootstrap interval of three is not. Pick ``box`` to see that spread.

THE SPEED-UP AXIS IS LOG2. A ratio axis on a linear scale reads a 2x slow-down as a small event and
a 2x speed-up as a large one; log2 puts them the same distance from the 1x line and the ticks are
labelled back into ratios (``1/4x .. 4x``) rather than left as the small integers a bare log2 axis
would show.

``--summary`` appends a narrow column to the right of the per-kernel panel, past a dashed
separator, carrying the OVERALL value over the plotted kernels (of their own per-kernel medians).
Speed-up is a ratio, so its overall value is the GEOMETRIC MEAN (:func:`hpcagent_bench.stats.summary.geomean_ci`)
-- never a median, which is not the geometric mean except when the per-kernel values happen to be
symmetric; this is the same rule :func:`hpcagent_bench.stats.population.kernel_medians` and
:class:`~hpcagent_bench.stats.population.ArmAggregate` already report an "overall speed-up" under.
Tokens are not a ratio, so their summary column stays the median.

``--layout stacked`` draws both metrics as one figure, speed-up over tokens, sharing the kernel (x)
axis: the two panels are given the SAME kernel order (:func:`shared_kernel_order`) so column ``i``
names one kernel in both, even when a kernel has one metric and not the other (an empty column at
its slot, never a re-packed one).

Sized for a double-column paper page (:data:`hpcagent_bench.stats.style.DOUBLE_COLUMN_WIDTH`, ~7.0in):
each panel is :data:`PANEL_HEIGHT_IN` (1.8in) tall, well inside the 1.6-2.0in a reviewer asked for.
"""

import dataclasses
import math
import pathlib
from collections.abc import Callable, Sequence
from typing import Literal

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hpcagent_bench import experiment_tags
from hpcagent_bench.stats import population
from hpcagent_bench.stats import style as plotstyle
from hpcagent_bench.stats import summary

#: The two drawing modes: the median (+ bootstrap CI) or the raw per-episode boxplot.
Style = Literal["ci", "box"]

#: Episodes per kernel at or above which ``box`` draws an actual box instead of a point. 3 is
#: git-scicomp's own episode count -- the smallest population a quartile spread still means
#: something for, as opposed to being the two endpoints wearing quartile marks.
MIN_EPISODES_FOR_SPREAD: int = 3

#: A single panel's height, inches. Comfortably inside the 1.6-2.0in a paper figure gets.
PANEL_HEIGHT_IN: float = 1.8

#: Extra inches below every panel for the rotated kernel-name ticks, and above for the title.
CHROME_IN: float = 1.15


@dataclasses.dataclass(frozen=True, slots=True)
class KernelCell:
    """One kernel's per-episode values for one metric, sorted."""

    kernel: str
    episodes: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.episodes)

    def median(self) -> float:
        return float(np.median(self.episodes)) if self.episodes else math.nan


def speedup_cells(frame: pd.DataFrame) -> list[KernelCell]:
    """One cell per kernel: every episode's own final speed-up (:func:`population.graded_episode_rows`)."""
    graded = frame[frame.record == "submission"]
    episodes = population.graded_episode_rows(graded)
    cells: list[KernelCell] = []
    for kernel, group in episodes.groupby("benchmark"):
        values = sorted(float(v) for v in group.speedup if v > 0)
        if values:
            cells.append(KernelCell(str(kernel), tuple(values)))
    return cells


def token_cells(frame: pd.DataFrame) -> list[KernelCell]:
    """One cell per kernel: every episode's own token total (:func:`population.episode_tokens`)."""
    episodes = population.episode_tokens(frame, by=("benchmark",))
    cells: list[KernelCell] = []
    for kernel, group in episodes.groupby("benchmark"):
        values = sorted(float(v) for v in group.tokens if v > 0)
        if values:
            cells.append(KernelCell(str(kernel), tuple(values)))
    return cells


def ordered_kernels(cells: Sequence[KernelCell]) -> list[str]:
    """Kernels ascending by their own median -- the order every panel here draws in."""
    return [cell.kernel for cell in sorted(cells, key=lambda cell: cell.median())]


def shared_kernel_order(speed: Sequence[KernelCell], tokens: Sequence[KernelCell]) -> list[str]:
    """One kernel order for BOTH panels of a stacked figure: speed-up's order, then any kernel
    tokens has and speed-up does not, appended -- so column ``i`` names one kernel in both panels."""
    primary = ordered_kernels(speed)
    seen = set(primary)
    extra = [kernel for kernel in ordered_kernels(tokens) if kernel not in seen]
    return primary + extra


def exp2_or_nan(value: float) -> float:
    """``2**value``, or NaN through NaN/inf -- for mapping a log2-space bootstrap end back to a ratio."""
    return 2.0**value if math.isfinite(value) else math.nan


def bootstrap_point(cell: KernelCell, log2_space: bool) -> tuple[float, float, float]:
    """``(median, low, high)`` for the ``ci`` style: bootstrap median CI, in log2 space for a ratio
    (so the interval is symmetric about the geometric centre) and in linear space for a count."""
    values = np.asarray(cell.episodes, dtype=np.float64)
    space = np.log2(values) if log2_space else values
    med, low, high, _ = summary.median_ci(space, drop=False, warn=False, min_n=summary.MIN_INTERVAL_SAMPLES)
    if not log2_space:
        return med, low, high
    return exp2_or_nan(med), exp2_or_nan(low), exp2_or_nan(high)


def kernel_medians(cells: Sequence[KernelCell]) -> np.ndarray:
    """The plotted kernels' own per-kernel medians -- what the ``--summary`` column reduces one level up."""
    return np.array([cell.median() for cell in cells if math.isfinite(cell.median())], dtype=np.float64)


def summary_point_speedup(cells: Sequence[KernelCell]) -> tuple[float, float, float]:
    """``(geomean, low, high)`` over the plotted kernels' own median speed-ups.

    Speed-up is a ratio, so its OVERALL value is the geometric mean
    (:func:`hpcagent_bench.stats.summary.geomean_ci`) -- never a median, which equals the geomean
    only when the per-kernel values happen to be symmetric. The same rule
    :func:`hpcagent_bench.stats.population.kernel_medians` reports an arm's speed-up under.
    """
    medians = kernel_medians(cells)
    medians = medians[medians > 0.0]
    if medians.size == 0:
        return math.nan, math.nan, math.nan
    interval = summary.geomean_ci(medians)
    return interval.point, interval.low, interval.high


def summary_point_tokens(cells: Sequence[KernelCell]) -> tuple[float, float, float]:
    """``(median, low, high)`` over the plotted kernels' own median tokens.

    Tokens are not a ratio, so the geomean rule above does not apply here: the summary column stays
    the bootstrap median, same as every per-kernel token cell.
    """
    medians = kernel_medians(cells)
    if medians.size == 0:
        return math.nan, math.nan, math.nan
    med, low, high, _ = summary.median_ci(medians, drop=False, warn=False, min_n=summary.MIN_INTERVAL_SAMPLES)
    return med, low, high


#: A metric's ``--summary`` reducer: the plotted kernels' own medians in, ``(point, low, high)`` out.
SummaryReducer = Callable[[Sequence["KernelCell"]], tuple[float, float, float]]


def speedup_yticks(cells: Sequence[KernelCell]) -> list[float]:
    """Powers of two spanning every plotted value, always at least ``1/4x .. 4x``."""
    values = [v for cell in cells for v in cell.episodes if math.isfinite(v) and v > 0]
    low, high = (min(values), max(values)) if values else (1.0, 1.0)
    low_exp = min(-2, math.floor(math.log2(low)))
    high_exp = max(2, math.ceil(math.log2(high)))
    return [2.0**exp for exp in range(low_exp, high_exp + 1)]


def speedup_tick_label(value: float) -> str:
    """``0.25 -> "0.25x"``, ``1.0 -> "1x"``, ``4.0 -> "4x"``: a log2 tick read back as a ratio.

    A ratio below 1 prints as a decimal (user, 2026-09-20). The earlier ``1/n`` spelling only ever
    worked for a whole reciprocal: it rounded, so a half-octave tick at 0.707 printed ``1/1x``, a
    ratio of one marking a point 30% below it. Decimals also let a tick land anywhere, which is
    what densifying the token-cost axis needs.
    """
    if value == 1.0:
        return "1x"
    if value > 1.0:
        return f"{value:g}x"
    return f"{float(f'{value:.3g}'):g}x"


def style_speedup_axis(ax: matplotlib.axes.Axes, cells: Sequence[KernelCell]) -> None:
    """Powers of two, read back as ratios, with a MAJOR grid on the value axis and nothing on the
    kernel axis -- the ticks are pinned here, so the grid is drawn beside them rather than through
    ``plotstyle.value_axis``, which would relocate them."""
    ax.set_yscale("log", base=2)
    ticks = speedup_yticks(cells)
    ax.set_yticks(ticks)
    ax.set_yticklabels([speedup_tick_label(tick) for tick in ticks], fontsize=plotstyle.TICK_PT * 0.55)
    ax.axhline(1.0, color=plotstyle.REFERENCE, linewidth=0.9, zorder=1)
    ax.grid(axis="y", which="major", color=plotstyle.RULE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def style_token_axis(ax: matplotlib.axes.Axes) -> None:
    ax.set_yscale("log")
    plotstyle.value_axis(ax, "y", log_base=10.0)


def draw_ci(
    ax: matplotlib.axes.Axes, cells: Sequence[KernelCell], x_of: dict[str, int], color: str, log2_space: bool
) -> None:
    for cell in cells:
        x = x_of[cell.kernel]
        med, low, high = bootstrap_point(cell, log2_space)
        if math.isfinite(low) and math.isfinite(high) and low != high:
            ax.vlines(x, low, high, color=color, linewidth=1.2, alpha=0.6, zorder=2)
        ax.plot([x], [med], marker="o", markersize=4.0, color=color, linestyle="none", zorder=3)


def draw_box(ax: matplotlib.axes.Axes, cells: Sequence[KernelCell], x_of: dict[str, int], color: str) -> None:
    """A real box for a kernel with :data:`MIN_EPISODES_FOR_SPREAD`+ episodes; a point otherwise --
    mixing the two in one panel is deliberate (see the module docstring)."""
    boxed = [cell for cell in cells if cell.n >= MIN_EPISODES_FOR_SPREAD]
    pointwise = [cell for cell in cells if cell.n < MIN_EPISODES_FOR_SPREAD]
    if boxed:
        artists = ax.boxplot(
            [list(cell.episodes) for cell in boxed],
            positions=[x_of[cell.kernel] for cell in boxed],
            widths=0.5,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
            medianprops={"color": "0.1", "linewidth": 1.0},
        )
        for box in artists["boxes"]:
            box.set(facecolor=color, edgecolor=color, alpha=0.55, linewidth=0.6)
        for part in ("whiskers", "caps"):
            for line in artists[part]:
                line.set(color=color, linewidth=0.6)
    for cell in pointwise:
        ax.plot(
            [x_of[cell.kernel]], [cell.median()], marker="o", markersize=4.0, color=color, linestyle="none", zorder=3
        )


#: Gap (in x-axis units) between the last kernel column and the dashed separator, and between the
#: separator and the summary column.
SUMMARY_GAP: float = 0.7


def draw_summary_column(
    ax: matplotlib.axes.Axes,
    cells: Sequence[KernelCell],
    n_kernels: int,
    color: str,
    reducer: SummaryReducer,
    label: str,
) -> float:
    """The dashed separator, the summary reducer's marker, and a small label ABOVE it naming its
    own statistic; returns the column's x position.

    The label is an annotation at the marker, never an x-axis TICK label: a stacked figure shares
    one x axis between two panels (:func:`figure_stacked`) and matplotlib shares the same tick
    label text for every row sharing that axis, so a per-panel tick label silently loses whichever
    panel drew first -- the top panel's "Geomean" was overwritten by the bottom panel's "Median".
    An annotation anchored to the panel's own data coordinates has no such sharing.
    """
    separator_x = n_kernels - 0.5 + SUMMARY_GAP
    summary_x = separator_x + SUMMARY_GAP
    ax.axvline(separator_x, color=plotstyle.RULE, linestyle=(0, (3, 3)), linewidth=1.0, zorder=1)
    point, low, high = reducer(cells)
    if math.isfinite(point):
        if math.isfinite(low) and math.isfinite(high) and low != high:
            ax.vlines(summary_x, low, high, color=color, linewidth=1.5, alpha=0.7, zorder=2)
        ax.plot([summary_x], [point], marker="D", markersize=5.0, color=color, linestyle="none", zorder=3)
    ax.annotate(
        label,
        xy=(summary_x, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=plotstyle.TICK_PT * 0.5,
        color=plotstyle.MUTED,
        annotation_clip=False,
    )
    return summary_x


def draw_panel(
    ax: matplotlib.axes.Axes,
    cells: Sequence[KernelCell],
    kernels: Sequence[str],
    style_: Style,
    color: str,
    log2_space: bool,
    ylabel: str,
    summary_reducer: SummaryReducer,
    summary_label: str,
    summary_column: bool,
    label_ticks: bool,
) -> None:
    """One metric's panel: its cells over ``kernels`` (a FIXED order, so a stacked figure's two
    panels share x), plus the optional summary column."""
    x_of = {kernel: i for i, kernel in enumerate(kernels)}
    present = [cell for cell in cells if cell.kernel in x_of]
    if style_ == "box":
        draw_box(ax, present, x_of, color)
    else:
        draw_ci(ax, present, x_of, color, log2_space)
    n = len(kernels)
    right_edge = float(n) - 0.4
    if summary_column:
        right_edge = draw_summary_column(ax, present, n, color, summary_reducer, summary_label) + 0.5
    ax.set_xlim(-0.6, right_edge)
    # Kernel names are the only x TICKS -- the summary column carries its own statistic as an
    # annotation (draw_summary_column), never a tick label, which a shared stacked x axis would
    # silently hand to the wrong panel (see that function's docstring).
    ax.set_xticks(range(n))
    if label_ticks:
        # The tick is the kernel's manifest NAME; ``kernels`` are the identifiers the columns and
        # the results table are keyed by (:func:`experiment_tags.kernel_display_name`).
        labels = [experiment_tags.kernel_display_name(kernel) for kernel in kernels]
        ax.set_xticklabels(labels, rotation=90, fontsize=plotstyle.TICK_PT * 0.5)
    else:
        ax.set_xticklabels([])
    if log2_space:
        style_speedup_axis(ax, present)
    else:
        style_token_axis(ax)
    ax.set_ylabel(ylabel, fontsize=plotstyle.LABEL_PT * 0.72)
    plotstyle.despine(ax)


@dataclasses.dataclass(frozen=True, slots=True)
class Metric:
    """One panel's identity: its cells, axis kind, label and its ``--summary`` reducer --
    everything :func:`draw_panel` needs besides the shared kernel order and drawing mode.

    ``summary_reducer`` and ``summary_label`` carry the statistic the SUMMARY COLUMN is under: the
    geomean for a ratio (speed-up), the median for a count (tokens) -- see
    :func:`summary_point_speedup` and :func:`summary_point_tokens`.
    """

    cells: tuple[KernelCell, ...]
    log2_space: bool
    ylabel: str
    color: str
    summary_reducer: SummaryReducer
    summary_label: str


def speedup_metric(cells: Sequence[KernelCell], ylabel: str, color: str) -> Metric:
    return Metric(tuple(cells), True, ylabel, color, summary_point_speedup, "Geomean")


def token_metric(cells: Sequence[KernelCell], ylabel: str, color: str) -> Metric:
    return Metric(tuple(cells), False, ylabel, color, summary_point_tokens, "Median")


def figure_one(
    metric: Metric, kernels: Sequence[str], style_: Style, summary_column: bool, title: str
) -> matplotlib.figure.Figure:
    """A single metric's panel as its own figure."""
    fig, ax = plt.subplots(figsize=(plotstyle.DOUBLE_COLUMN_WIDTH, PANEL_HEIGHT_IN + CHROME_IN))
    draw_panel(
        ax,
        metric.cells,
        kernels,
        style_,
        metric.color,
        metric.log2_space,
        metric.ylabel,
        metric.summary_reducer,
        metric.summary_label,
        summary_column,
        True,
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.86, bottom=0.40)
    plotstyle.title(fig, title)
    return fig


def figure_stacked(
    speed: Metric, tokens: Metric, kernels: Sequence[str], style_: Style, summary_column: bool, title: str
) -> matplotlib.figure.Figure:
    """Both metrics as one figure, speed-up over tokens, sharing the kernel axis."""
    height = 2 * PANEL_HEIGHT_IN + CHROME_IN
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(plotstyle.DOUBLE_COLUMN_WIDTH, height))
    draw_panel(
        axes[0],
        speed.cells,
        kernels,
        style_,
        speed.color,
        speed.log2_space,
        speed.ylabel,
        speed.summary_reducer,
        speed.summary_label,
        summary_column,
        False,
    )
    draw_panel(
        axes[1],
        tokens.cells,
        kernels,
        style_,
        tokens.color,
        tokens.log2_space,
        tokens.ylabel,
        tokens.summary_reducer,
        tokens.summary_label,
        summary_column,
        True,
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.90, bottom=0.30, hspace=0.12)
    plotstyle.title(fig, title)
    return fig


def cells_table(cells: Sequence[KernelCell], metric: str) -> pd.DataFrame:
    """The data table behind one metric's panel: one row per kernel, its n, median and episodes."""
    rows = [
        {
            "benchmark": cell.kernel,
            "metric": metric,
            "n": cell.n,
            "median": cell.median(),
            "episodes": list(cell.episodes),
        }
        for cell in cells
    ]
    return pd.DataFrame(rows, columns=["benchmark", "metric", "n", "median", "episodes"])


def save(fig: matplotlib.figure.Figure, out: pathlib.Path) -> pathlib.Path:
    return plotstyle.save(fig, out.with_suffix(""))
