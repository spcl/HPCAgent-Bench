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
    """One kernel's per-episode values for one metric, sorted.

    ``delivered`` False is the 2026-09-16 rule: a kernel the arm was SERVED and never verified an
    answer for still scores 1x and its tokens are still spent. Dropping it instead would report the
    arm's speed-up over the kernels it happened to solve, which is a different and always kinder
    number -- a 28-of-40 arm would read like a 40-of-40 one. Such a cell holds the single
    placeholder value and draws as a cross (:func:`draw_ci`, :func:`draw_box`).
    """

    kernel: str
    episodes: tuple[float, ...]
    delivered: bool = True
    #: The judge or a source audit disowned this answer, so its value is drawn but not believed
    #: (:func:`draw_flagged`). Distinct from ``delivered`` False: there IS a number here, and the
    #: point of showing it is that it is large.
    flagged: bool = False

    @property
    def n(self) -> int:
        return len(self.episodes)

    def median(self) -> float:
        return float(np.median(self.episodes)) if self.episodes else math.nan


def speedup_cells(frame: pd.DataFrame, served: bool = True) -> list[KernelCell]:
    """One cell per kernel: every episode's own final speed-up
    (:func:`population.graded_episode_rows`), plus -- under ``served`` -- one placeholder cell at
    :data:`~hpcagent_bench.stats.population.NOT_DELIVERED` for every kernel the frame was served and
    never answered.

    ``served=False`` draws the solved kernels alone, which is the right population only when the
    caller has already said so somewhere else on the page.
    """
    graded = frame[frame.record == "submission"]
    episodes = population.graded_episode_rows(graded)
    cells: list[KernelCell] = []
    for kernel, group in episodes.groupby("benchmark"):
        values = sorted(float(v) for v in group.speedup if v > 0)
        if values:
            cells.append(KernelCell(str(kernel), tuple(values)))
    if not served:
        return cells
    answered = {cell.kernel for cell in cells}
    unanswered = sorted(set(frame["benchmark"].dropna().astype(str)) - answered)
    return cells + [KernelCell(kernel, (population.NOT_DELIVERED,), False) for kernel in unanswered]


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
    """Kernels ascending by median, each named ONCE.

    A panel may carry several series over one kernel axis, so a kernel appears in ``cells`` once
    per series. Ordering the cells directly would then emit that kernel once per series and the
    axis would grow to the CELL count -- six series over forty kernels drew 201 columns. A kernel
    is ranked by the median of its cells' medians, which for one series is its own median and
    leaves a single-series panel in exactly the order it had.
    """
    grouped: dict[str, list[float]] = {}
    for cell in cells:
        value = cell.median()
        if math.isfinite(value):
            grouped.setdefault(cell.kernel, []).append(value)
    for cell in cells:
        grouped.setdefault(cell.kernel, [])
    return sorted(grouped, key=lambda kernel: float(np.median(grouped[kernel])) if grouped[kernel] else math.inf)


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


@dataclasses.dataclass(frozen=True, slots=True)
class Series:
    """One drawn population inside a panel: a model, a language, a device, or any combination.

    A panel used to carry exactly one, so colour was a panel-level argument. It is a SERIES
    property now, because the question "is this kernel hard, or is this model bad at it" needs
    several populations over one kernel axis to answer.
    """

    label: str
    cells: tuple[KernelCell, ...]
    color: str
    marker: str = "o"
    filled: bool = True


#: Total x width one kernel column's series are spread over. Below ~0.8 the intervals of adjacent
#: kernels start to touch and the reader loses which column a mark belongs to.
DODGE_SPAN: float = 0.62


def dodge_offsets(count: int) -> list[float]:
    """Symmetric x offsets for ``count`` series sharing one kernel column, centred on it.

    One series draws ON the column, not beside it, so a single-series figure is pixel-identical to
    what it was before series existed.
    """
    if count < 2:
        return [0.0]
    step = DODGE_SPAN / (count - 1)
    return [-DODGE_SPAN / 2.0 + i * step for i in range(count)]


def draw_ci(
    ax: matplotlib.axes.Axes,
    cells: Sequence[KernelCell],
    x_of: dict[str, int],
    color: str,
    log2_space: bool,
    offset: float = 0.0,
    marker: str = "o",
    filled: bool = True,
) -> None:
    for cell in cells:
        x = x_of[cell.kernel] + offset
        if not cell.delivered:
            draw_placeholder(ax, x, color)
            continue
        if cell.flagged:
            draw_flagged(ax, x, cell.median(), color)
            continue
        med, low, high = bootstrap_point(cell, log2_space)
        if math.isfinite(low) and math.isfinite(high) and low != high:
            ax.vlines(x, low, high, color=color, linewidth=1.0, alpha=0.6, zorder=2)
        ax.plot(
            [x], [med], marker=marker, markersize=3.6, color=color, linestyle="none", zorder=3,
            markerfacecolor=color if filled else "none", markeredgewidth=0.9,
        )  # fmt: skip


#: The mark for a disowned answer, and the superscript that separates it from an unanswered one.
FLAGGED_MARKER: str = "X"
FLAGGED_ANNOTATION: str = "*"


def draw_flagged(ax: matplotlib.axes.Axes, x: float, value: float, color: str) -> None:
    """A disowned answer at the value it claimed: a filled cross carrying a ``*``.

    An unanswered kernel is already a cross at 1x (:func:`draw_placeholder`), so a reader who has
    learnt that mark reads this one as its neighbour: no credit. The ``*`` is what says the two
    are not the same, and the value is drawn where it landed because the claim being far above the
    honest ceiling is the whole observation.
    """
    ax.plot(
        [x], [value], marker=FLAGGED_MARKER, markersize=5.0, markeredgewidth=1.4, color=color,
        linestyle="none", zorder=4,
    )  # fmt: skip
    ax.annotate(
        FLAGGED_ANNOTATION, (x, value), textcoords="offset points", xytext=(3.5, 2.0), color=color,
        fontsize=7.0, ha="left", va="bottom", zorder=4, annotation_clip=False,
    )  # fmt: skip


def draw_placeholder(ax: matplotlib.axes.Axes, x: float, color: str) -> None:
    """A served-and-never-answered kernel: a cross at the 1x placeholder, in the panel's own colour.
    A cross, not a dot, so a reader never reads it as a measured 1x."""
    ax.plot(
        [x], [population.NOT_DELIVERED], marker="x", markersize=4.5, markeredgewidth=1.1, color=color,
        linestyle="none", zorder=3,
    )  # fmt: skip


def draw_box(ax: matplotlib.axes.Axes, cells: Sequence[KernelCell], x_of: dict[str, int], color: str) -> None:
    """A real box for a kernel with :data:`MIN_EPISODES_FOR_SPREAD`+ episodes; a point otherwise --
    mixing the two in one panel is deliberate (see the module docstring)."""
    for cell in (cell for cell in cells if not cell.delivered):
        draw_placeholder(ax, x_of[cell.kernel], color)
    for cell in (cell for cell in cells if cell.delivered and cell.flagged):
        draw_flagged(ax, x_of[cell.kernel], cell.median(), color)
    cells = [cell for cell in cells if cell.delivered and not cell.flagged]
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
    offset: float = 0.0,
    marker: str = "D",
    filled: bool = True,
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
    summary_x = separator_x + SUMMARY_GAP + offset
    ax.axvline(separator_x, color=plotstyle.RULE, linestyle=(0, (3, 3)), linewidth=1.0, zorder=1)
    point, low, high = reducer(cells)
    if math.isfinite(point):
        if math.isfinite(low) and math.isfinite(high) and low != high:
            ax.vlines(summary_x, low, high, color=color, linewidth=1.3, alpha=0.7, zorder=2)
        ax.plot(
            [summary_x], [point], marker=marker, markersize=5.0, color=color, linestyle="none", zorder=3,
            markerfacecolor=color if filled else "none", markeredgewidth=1.1,
        )  # fmt: skip
    ax.annotate(
        label,
        xy=(separator_x + SUMMARY_GAP, 1.0),
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
    series: Sequence[Series],
    kernels: Sequence[str],
    style_: Style,
    log2_space: bool,
    ylabel: str,
    summary_reducer: SummaryReducer,
    summary_label: str,
    summary_column: bool,
    label_ticks: bool,
) -> None:
    """One metric's panel: every series over ``kernels`` (a FIXED order, so a stacked figure's two
    panels share x), plus the optional summary column.

    Several series in one column are spread by :func:`dodge_offsets` so their intervals stay
    readable; one series keeps the column's exact x, so a single-series panel is unchanged.
    """
    x_of = {kernel: i for i, kernel in enumerate(kernels)}
    offsets = dodge_offsets(len(series))
    present: list[KernelCell] = []
    for one, offset in zip(series, offsets, strict=True):
        cells = [cell for cell in one.cells if cell.kernel in x_of]
        present.extend(cells)
        if style_ == "box" and len(series) == 1:
            draw_box(ax, cells, x_of, one.color)
        else:
            draw_ci(ax, cells, x_of, one.color, log2_space, offset, one.marker, one.filled)
    n = len(kernels)
    right_edge = float(n) - 0.4
    if summary_column:
        for one, offset in zip(series, offsets, strict=True):
            cells = [cell for cell in one.cells if cell.kernel in x_of]
            right_edge = (
                draw_summary_column(
                    ax, cells, n, one.color, summary_reducer, summary_label, offset * 2.0, one.marker, one.filled
                )
                + 0.5
            )
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

    series: tuple[Series, ...]
    log2_space: bool
    ylabel: str
    summary_reducer: SummaryReducer
    summary_label: str

    @property
    def cells(self) -> tuple[KernelCell, ...]:
        """Every series' cells, for the callers that only need the kernel axis."""
        return tuple(cell for one in self.series for cell in one.cells)


def speedup_metric(cells: Sequence[KernelCell], ylabel: str, color: str) -> Metric:
    return Metric((Series("", tuple(cells), color),), True, ylabel, summary_point_speedup, "Geomean")


def token_metric(cells: Sequence[KernelCell], ylabel: str, color: str) -> Metric:
    return Metric((Series("", tuple(cells), color),), False, ylabel, summary_point_tokens, "Median")


def speedup_series_metric(series: Sequence[Series], ylabel: str) -> Metric:
    return Metric(tuple(series), True, ylabel, summary_point_speedup, "Geomean")


def figure_one(
    metric: Metric, kernels: Sequence[str], style_: Style, summary_column: bool, title: str
) -> matplotlib.figure.Figure:
    """A single metric's panel as its own figure."""
    fig, ax = plt.subplots(figsize=(plotstyle.DOUBLE_COLUMN_WIDTH, PANEL_HEIGHT_IN + CHROME_IN))
    draw_panel(
        ax,
        metric.series,
        kernels,
        style_,
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
        speed.series,
        kernels,
        style_,
        speed.log2_space,
        speed.ylabel,
        speed.summary_reducer,
        speed.summary_label,
        summary_column,
        False,
    )
    draw_panel(
        axes[1],
        tokens.series,
        kernels,
        style_,
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
