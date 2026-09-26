# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-kernel figures: one column per kernel, one mark per series, one summary slot per series.

The one per-kernel drawing API: the compiler figure
(:func:`hpcagent_bench.stats.figures.signed.llr40_figure`) and the paper's per-kernel figures both
draw through this module, so a column, a mark, a summary slot and a margin mean the same thing
everywhere. A caller turns its data into :class:`KernelCell` s (:func:`kernel_cells`), groups them
into :class:`Series`, picks one :class:`Metric` per panel and calls :func:`figure_panels`.

A cell's status decides its mark: measured (a point with an interval), undelivered (hollow,
crossed, at the value the failure left, 1x when it delivered nothing), pending ("?" at 1x) or
flagged (a disowned answer's cross with a ``*``). Undelivered and pending cells enter no summary.

The value axis is log2 for a speedup (:func:`style_speedup_axis`) and log10 for a token count
(:func:`style_token_axis`), both sized from the cells before a mark is drawn. The summary column
sits past a dashed separator, one slot per series (:func:`summary_slot_x`), each showing the
geometric mean over the solved kernels (:func:`summary_geomean`). The canvas is measured, not
fixed (:func:`fit_canvas`): margins come from what the chrome actually prints.
"""

import enum
import dataclasses
import logging
import math
import pathlib
import textwrap
from collections.abc import Callable, Mapping, Sequence

import matplotlib.artist
import matplotlib.axes
import matplotlib.figure
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hpcagent_bench import experiment_tags
from hpcagent_bench.stats import population, summary
from hpcagent_bench.stats import style as plotstyle

LOG = logging.getLogger(__name__)


#: The two drawing modes: the median (+ bootstrap CI), or the raw per-episode boxplot.
class Style(enum.Enum):
    CI = "ci"
    BOX = "box"


#: Episodes per kernel at or above which ``box`` draws an actual box instead of a point.
MIN_EPISODES_FOR_SPREAD: int = 3

#: A single panel's height, inches, for a figure authored at double-column width.
PANEL_HEIGHT_IN: float = 1.8
#: A single panel's height at print size (``width_in`` given): a text-width strip of forty kernels.
PRINT_PANEL_HEIGHT_IN: float = 1.12


#: Authoring scale: type cut to about half, kernel names and key at half tick size, print weights.
AUTHOR_TYPE: plotstyle.TypeScale = dataclasses.replace(
    plotstyle.AUTHOR_SCALE,
    tick_pt=plotstyle.AUTHOR_SCALE.tick_pt * 0.55,
    label_pt=plotstyle.AUTHOR_SCALE.label_pt * 0.72,
    legend_pt=plotstyle.AUTHOR_SCALE.tick_pt * 0.5,
    annotation_pt=plotstyle.AUTHOR_SCALE.tick_pt * 0.5,
    line_width=plotstyle.PRINT_SCALE.line_width,
)

#: How far the kernel names may shrink below their starting ``annotation_pt`` to fit the column pitch.
MIN_NAME_SCALE: float = 0.6


def min_text_pt(type_: plotstyle.TypeScale, size: float) -> float:
    """The smallest a fitted text of ``size`` may get: the shared print floor at print size,
    :data:`MIN_NAME_SCALE` of it at authoring size."""
    return plotstyle.PRINT_MIN_PT if type_ == plotstyle.PRINT_SCALE else size * MIN_NAME_SCALE


#: Line weights in points beside the scale's own ``line_width``.
REFERENCE_LINE_WIDTH: float = 0.9
GRID_LINE_WIDTH: float = 0.7
BOX_LINE_WIDTH: float = 0.6
SUMMARY_LINE_WIDTH: float = 1.3
CROSS_EDGE_WIDTH: float = 1.2
FLAGGED_EDGE_WIDTH: float = 1.4

#: Air between the canvas edge and the chrome :func:`fit_canvas` measures, in inches.
CHROME_PAD_IN: float = 0.04


@dataclasses.dataclass(frozen=True, slots=True)
class KernelCell:
    """One kernel's values for one series, sorted, and the status that decides its mark."""

    kernel: str
    episodes: tuple[float, ...]
    #: False: served but never verified. Still scores 1x, still spends tokens, enters no summary.
    delivered: bool = True
    #: Disowned by the judge or an audit: the value is drawn but not believed (:func:`draw_flagged`).
    flagged: bool = False
    #: A (low, high) the caller already has for this kernel. ``None`` lets ``ci`` bootstrap one.
    interval: tuple[float, float] | None = None
    #: Served but not attempted yet: drawn as a "?" at 1x. Also undelivered, so no summary either.
    pending: bool = False

    @property
    def n(self) -> int:
        return len(self.episodes)

    def median(self) -> float:
        return float(np.median(self.episodes)) if self.episodes else math.nan


def usable(value: float) -> bool:
    """Whether ``value`` is a finite positive number, the only kind a log axis can place."""
    return math.isfinite(value) and value > 0.0


def kernel_interval(low: float, high: float) -> tuple[float, float] | None:
    """``(low, high)`` when both ends are usable and ``low < high``, else no interval."""
    return (low, high) if usable(low) and usable(high) and low < high else None


def kernel_cells(
    values: Mapping[str, float],
    kernels: Sequence[str],
    fill: bool = True,
    delivered: Mapping[str, bool] | None = None,
    low: Mapping[str, float] | None = None,
    high: Mapping[str, float] | None = None,
    pending: frozenset[str] = frozenset(),
) -> tuple[KernelCell, ...]:
    """One single-value cell per kernel of ``kernels`` from a caller's own ``kernel -> value`` map.

    A kernel with no usable value is filled at :data:`~hpcagent_bench.stats.population.NOT_DELIVERED`
    as an undelivered cell under ``fill``, else left out. ``delivered`` marks present values that are
    placeholders all the same. ``low``/``high`` give a kernel its own interval where usable and
    ``low < high``. ``pending`` kernels become pending cells whatever ``values`` says.
    """
    delivered = delivered or {}
    low, high = low or {}, high or {}
    cells: list[KernelCell] = []
    for kernel in kernels:
        value = values.get(kernel, math.nan)
        if kernel in pending:
            cells.append(KernelCell(kernel, (population.NOT_DELIVERED,), delivered=False, pending=True))
        elif usable(value):
            interval = kernel_interval(low.get(kernel, math.nan), high.get(kernel, math.nan))
            cells.append(KernelCell(kernel, (float(value),), delivered.get(kernel, True), interval=interval))
        elif fill:
            cells.append(KernelCell(kernel, (population.NOT_DELIVERED,), delivered=False))
    return tuple(cells)


def ordered_kernels(cells: Sequence[KernelCell]) -> list[str]:
    """Kernels ascending by median, each named once (a panel may carry several series over one
    kernel axis, so a kernel appears in ``cells`` once per series)."""
    grouped: dict[str, list[float]] = {}
    for cell in cells:
        value = cell.median()
        if math.isfinite(value):
            grouped.setdefault(cell.kernel, []).append(value)
    for cell in cells:
        grouped.setdefault(cell.kernel, [])
    return sorted(grouped, key=lambda kernel: float(np.median(grouped[kernel])) if grouped[kernel] else math.inf)


def answer_cells(
    frame: pd.DataFrame, repeats: population.RepeatPolicy = population.RepeatPolicy.LATEST
) -> list[KernelCell]:
    """One single-value cell per SOLVED kernel: its final answer under ``repeats``
    (:func:`population.kernel_answers`), the policy the tables score a kernel by."""
    answers = population.kernel_answers(frame, repeats=repeats, policy=population.KernelPolicy.SOLVED)
    if "speedup" not in answers.columns:
        return []
    return [KernelCell(str(kernel), (float(value),)) for kernel, value in answers["speedup"].items() if value > 0]


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


def cell_point(cell: KernelCell, log2_space: bool) -> tuple[float, float, float]:
    """``(point, low, high)`` a measured cell draws: its own interval around its median when the
    caller gave one, else the bootstrap median CI over its episodes (:func:`bootstrap_point`)."""
    if cell.interval is not None:
        return cell.median(), *cell.interval
    return bootstrap_point(cell, log2_space)


def kernel_medians(cells: Sequence[KernelCell]) -> np.ndarray:
    """The plotted kernels' own per-kernel medians over the solved kernels only: undelivered,
    pending and flagged cells are drawn but never enter a summary."""
    return np.array(
        [cell.median() for cell in cells if cell.delivered and not cell.flagged and math.isfinite(cell.median())],
        dtype=np.float64,
    )


def summary_geomean(cells: Sequence[KernelCell]) -> tuple[float, float, float]:
    """``(geomean, low, high)`` over the plotted kernels' own medians, with its 95% log-t interval
    (:func:`hpcagent_bench.stats.summary.geomean_interval`)."""
    interval = summary.geomean_interval(kernel_medians(cells))
    return interval.point, interval.low, interval.high


#: A metric's summary reducer: the plotted kernels' cells in, ``(point, low, high)`` out.
SummaryReducer = Callable[[Sequence["KernelCell"]], tuple[float, float, float]]


def drawn_values(cells: Sequence[KernelCell]) -> list[float]:
    """Every usable value ``cells`` draw a mark at -- never an interval end, since a two-repeat
    t-interval can span tens of octaves and pinning the axis to it flattens every mark."""
    return [v for cell in cells for v in cell.episodes if usable(v)]


#: The most labelled powers of two a speedup axis carries; a wider range labels every second (or
#: third) octave instead.
MAX_SPEEDUP_TICKS: int = 7

#: Octaves of air past the outermost speedup ticks, so a mark sitting on one is never cut by the frame.
VALUE_PAD_OCTAVES: float = 0.35


def speedup_yticks(cells: Sequence[KernelCell], max_ticks: int = MAX_SPEEDUP_TICKS) -> list[float]:
    """Powers of two spanning every plotted value, always at least ``1/4x .. 4x`` and always 1x, in
    steps of as many octaves as keep the count at or under ``max_ticks``."""
    values = drawn_values(cells)
    low, high = (min(values), max(values)) if values else (1.0, 1.0)
    low_exp = min(-2, math.floor(math.log2(low)))
    high_exp = max(2, math.ceil(math.log2(high)))
    step = 1
    while True:
        first, last = math.floor(low_exp / step) * step, math.ceil(high_exp / step) * step
        if (last - first) // step + 1 <= max_ticks:
            return [2.0**exp for exp in range(first, last + 1, step)]
        step += 1


#: Decades of air past the smallest and the largest plotted token value.
TOKEN_PAD_DECADES: float = 0.08


def grid_125(low: float, high: float) -> list[float]:
    """The 1-2-5 values in ``[low, high]``: where :func:`plotstyle.value_axis` puts a log10 axis's
    labelled majors."""
    decades = range(math.floor(math.log10(low)) - 1, math.ceil(math.log10(high)) + 2)
    return [m * 10.0**k for k in decades for m in (1.0, 2.0, 5.0) if low <= m * 10.0**k <= high]


def token_limits(cells: Sequence[KernelCell]) -> tuple[float, float]:
    """Every plotted token value with a little air, widened to the next 1-2-5 values outward when
    that window would label fewer than two ticks."""
    values = drawn_values(cells)
    if not values:
        return 1.0, 10.0
    pad = 10.0**TOKEN_PAD_DECADES
    low, high = min(values) / pad, max(values) * pad
    if len(grid_125(low, high)) >= 2:
        return low, high
    return max(grid_125(low / 10.0, low)), min(grid_125(high, high * 10.0))


def style_speedup_axis(
    ax: matplotlib.axes.Axes,
    cells: Sequence[KernelCell],
    tick_pt: float = AUTHOR_TYPE.tick_pt,
    reference_color: str = plotstyle.REFERENCE,
) -> None:
    """Powers of two, read back as ratios, over limits pinned just past them, with a major grid and
    the shared minor ruling on the value axis and nothing on the kernel axis."""
    ax.set_yscale("log", base=2)
    ticks = speedup_yticks(cells)
    ax.set_yticks(ticks)
    ax.set_yticklabels([plotstyle.ratio_tick_label(tick) for tick in ticks], fontsize=tick_pt)
    plotstyle.minor_ticks(ax.yaxis, plotstyle.MinorKind.RATIO)
    pad = 2.0**VALUE_PAD_OCTAVES
    ax.set_ylim(ticks[0] / pad, ticks[-1] * pad)
    ax.axhline(1.0, color=reference_color, linewidth=REFERENCE_LINE_WIDTH, zorder=1)
    ax.grid(axis="y", which="major", color=plotstyle.RULE, linewidth=GRID_LINE_WIDTH, zorder=0)
    ax.set_axisbelow(True)


def style_token_axis(
    ax: matplotlib.axes.Axes, cells: Sequence[KernelCell], tick_pt: float = AUTHOR_TYPE.tick_pt
) -> None:
    """A log10 value axis over the plotted values (:func:`token_limits`), majors at 1-2-5."""
    ax.set_yscale("log")
    ax.set_ylim(*token_limits(cells))
    plotstyle.value_axis(ax, "y", log_base=10.0)
    ax.tick_params(axis="y", labelsize=tick_pt)


@dataclasses.dataclass(frozen=True, slots=True)
class Series:
    """One drawn population inside a panel: a model, a language, a device, a condition, a column.

    Colour is a series property, so several populations can share one kernel axis. A series with
    no cells still holds its dodge offset and summary slot, so it leaves every other series where a
    panel above it put them.
    """

    label: str
    cells: tuple[KernelCell, ...]
    color: str
    marker: str = "o"
    filled: bool = True


#: Total x width one kernel column's series are spread over by default.
DODGE_SPAN: float = 0.62


def dodge_offsets(count: int, span: float = DODGE_SPAN) -> list[float]:
    """Evenly spaced x offsets for ``count`` series sharing one kernel column, centred on it, the
    outermost two ``span`` apart (0 stacks every series on the column)."""
    if count < 2:
        return [0.0] * count
    step = span / (count - 1)
    return [-span / 2.0 + i * step for i in range(count)]


#: A mark's diameter in points where it has room, the smallest a crowded column may shrink it to,
#: and how much of the gap to its neighbour a mark may cover.
MARK_PT: float = 4.5
MIN_MARK_PT: float = 2.6
MARK_GAP_RATIO: float = 1.7


def mark_size(pitch_in: float, n_series: int, span: float = DODGE_SPAN) -> float:
    """A mark's area in points squared: :data:`MARK_PT` across where the marks have room, shrinking
    with the gap to the nearest neighbour down to :data:`MIN_MARK_PT` where they do not."""
    dodged = n_series > 1 and span > 0.0
    gap_pt = 72.0 * pitch_in * (span / (n_series - 1) if dodged else 1.0)
    return max(MIN_MARK_PT, min(MARK_PT, MARK_GAP_RATIO * gap_pt)) ** 2


def column_pitch_in(ax: matplotlib.axes.Axes) -> float:
    """Inches of ``ax``'s kernel axis per column, as it is laid out now."""
    low, high = ax.get_xlim()
    return float(ax.bbox.width) / ax.figure.dpi / (high - low)


#: Drawn under the marks' white halos (:data:`~hpcagent_bench.stats.style.FILL_Z`), never over them.
INTERVAL_Z: float = 2.0

#: The mark for a disowned answer, the superscript that separates it from an unanswered one, and the
#: mark's size in points.
FLAGGED_MARKER: str = "X"
FLAGGED_ANNOTATION: str = "*"
FLAGGED_MARK_PT: float = 5.0


def draw_flagged(
    ax: matplotlib.axes.Axes, x: float, value: float, color: str, annotation_pt: float = AUTHOR_TYPE.annotation_pt
) -> None:
    """A disowned answer at the value it claimed: a filled cross carrying a ``*`` so it is never
    read as an unanswered kernel's plain cross."""
    ax.plot(
        [x], [value], marker=FLAGGED_MARKER, markersize=FLAGGED_MARK_PT, markeredgewidth=FLAGGED_EDGE_WIDTH,
        color=color, linestyle="none", zorder=4,
    )  # fmt: skip
    ax.annotate(
        FLAGGED_ANNOTATION, (x, value), textcoords="offset points", xytext=(3.5, 2.0), color=color,
        fontsize=annotation_pt, ha="left", va="bottom", zorder=4, annotation_clip=False,
    )  # fmt: skip


def draw_status(
    ax: matplotlib.axes.Axes, cell: KernelCell, x: float, one: Series, size: float, type_: plotstyle.TypeScale
) -> bool:
    """Draw ``cell`` as the mark its status calls for (pending, undelivered, flagged) and say so; a
    measured cell is left to the caller's style."""
    if cell.pending:
        plotstyle.pending_mark(ax, x, population.NOT_DELIVERED, one.color, size=size)
    elif not cell.delivered:
        plotstyle.point_mark(ax, x, cell.median(), one.color, one.marker, filled=False, size=size, delivered=False)
    elif cell.flagged:
        draw_flagged(ax, x, cell.median(), one.color, type_.annotation_pt)
    else:
        return False
    return True


def draw_ci(
    ax: matplotlib.axes.Axes,
    one: Series,
    x_of: Mapping[str, int],
    log2_space: bool,
    offset: float = 0.0,
    size: float = MARK_PT**2,
    type_: plotstyle.TypeScale = AUTHOR_TYPE,
) -> None:
    """``one``'s cells over the kernels of ``x_of``: each measured cell a point with its interval
    (:func:`cell_point`), each other cell the mark its status calls for (:func:`draw_status`)."""
    for cell in one.cells:
        if cell.kernel not in x_of:
            continue
        x = x_of[cell.kernel] + offset
        if draw_status(ax, cell, x, one, size, type_):
            continue
        point, low, high = cell_point(cell, log2_space)
        if math.isfinite(low) and math.isfinite(high) and low < high:
            ax.vlines(x, low, high, color=one.color, linewidth=type_.line_width, alpha=0.6, zorder=INTERVAL_Z)
        plotstyle.point_mark(ax, x, point, one.color, one.marker, one.filled, size=size)


def draw_box(
    ax: matplotlib.axes.Axes,
    one: Series,
    x_of: Mapping[str, int],
    size: float = MARK_PT**2,
    type_: plotstyle.TypeScale = AUTHOR_TYPE,
) -> None:
    """A real box for a kernel with :data:`MIN_EPISODES_FOR_SPREAD`+ episodes; a point otherwise."""
    boxed: list[KernelCell] = []
    for cell in one.cells:
        if cell.kernel not in x_of or draw_status(ax, cell, x_of[cell.kernel], one, size, type_):
            continue
        if cell.n >= MIN_EPISODES_FOR_SPREAD:
            boxed.append(cell)
        else:
            plotstyle.point_mark(ax, x_of[cell.kernel], cell.median(), one.color, one.marker, one.filled, size=size)
    if boxed:
        box_cells(ax, boxed, [x_of[cell.kernel] for cell in boxed], one.color, type_)


def box_cells(
    ax: matplotlib.axes.Axes,
    cells: Sequence[KernelCell],
    positions: Sequence[int],
    color: str,
    type_: plotstyle.TypeScale,
) -> None:
    """One box per cell over its episodes at ``positions``, filled and outlined in ``color``."""
    artists = ax.boxplot(
        [list(cell.episodes) for cell in cells],
        positions=list(positions),
        widths=0.5,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
        medianprops={"color": "0.1", "linewidth": type_.line_width},
    )
    for box in artists["boxes"]:
        box.set(facecolor=color, edgecolor=color, alpha=0.55, linewidth=BOX_LINE_WIDTH)
    for line in [*artists["whiskers"], *artists["caps"]]:
        line.set(color=color, linewidth=BOX_LINE_WIDTH)


#: Gap (in x-axis units) between the last kernel column and the dashed separator, and between the
#: separator and the summary column.
SUMMARY_GAP: float = 0.7

#: X distance between consecutive series' summary marks, in kernel columns.
SUMMARY_SLOT: float = 1.0


def summary_separator_x(n_kernels: int) -> float:
    """The x position of the dashed separator between the kernels and the summary slots."""
    return n_kernels - 0.5 + SUMMARY_GAP


def summary_slot_x(n_kernels: int, slot: int) -> float:
    """The x position of the ``slot``-th series' summary mark, past the dashed separator."""
    return summary_separator_x(n_kernels) + SUMMARY_GAP + slot * SUMMARY_SLOT


def summary_centre_x(n_kernels: int, slots: int) -> float:
    """The x position centred under the summary column's ``slots`` slots."""
    return (summary_slot_x(n_kernels, 0) + summary_slot_x(n_kernels, max(slots, 1) - 1)) / 2.0


def draw_summary_column(
    ax: matplotlib.axes.Axes,
    n_kernels: int,
    slots: int,
    label: str,
    type_: plotstyle.TypeScale,
    annotate: bool = True,
) -> None:
    """The dashed separator and a small label above the column's slots naming its own statistic.

    The label is an annotation, never an x-axis tick label: a stacked figure shares one x axis
    between its panels, and matplotlib would drop a per-panel tick label to whichever panel drew
    last. Where every panel's statistic is the same, :func:`style_panel` names it once as a shared
    x tick instead and ``annotate`` is off.
    """
    ax.axvline(
        summary_separator_x(n_kernels),
        color=plotstyle.RULE,
        linestyle=(0, (3, 3)),
        linewidth=type_.line_width,
        zorder=1,
    )
    if not annotate:
        return
    centre = summary_centre_x(n_kernels, slots)
    ax.annotate(
        label, xy=(centre, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3), textcoords="offset points",
        ha="center", va="bottom", fontsize=type_.annotation_pt, color=plotstyle.MUTED, annotation_clip=False,
    )  # fmt: skip


def draw_summary_mark(
    ax: matplotlib.axes.Axes,
    cells: Sequence[KernelCell],
    x: float,
    one: Series,
    metric: "Metric",
    type_: plotstyle.TypeScale,
    size: float,
    value: bool = True,
) -> None:
    """One series' summary: the reducer's point with its interval, and -- when ``value`` -- the
    point's own value printed above it (settled clear of the marks at save time). Six or more
    series overprint their values in the narrow column; such a figure gives the numbers in its
    caption instead."""
    point, low, high = metric.summary_reducer(cells)
    if not math.isfinite(point):
        return
    if math.isfinite(low) and math.isfinite(high) and low < high:
        ax.vlines(x, low, high, color=one.color, linewidth=SUMMARY_LINE_WIDTH, alpha=0.7, zorder=INTERVAL_Z)
    plotstyle.point_mark(ax, x, point, one.color, one.marker, one.filled, size=size)
    if not value:
        return
    ax.annotate(
        metric.value_label(point), xy=(x, high if math.isfinite(high) else point), xytext=(0, 2),
        textcoords="offset points", rotation=90, ha="center", va="bottom", fontsize=type_.annotation_pt,
        color=one.color, gid=plotstyle.CLEAR_GID, annotation_clip=False,
    )  # fmt: skip


def kernel_tick_label(kernel: str) -> str:
    """The kernel's short manifest name (:func:`experiment_tags.kernel_short_display_name`), folded
    at the short-name limit onto as many lines as it needs, never cut."""
    name = experiment_tags.kernel_short_display_name(kernel)
    return "\n".join(textwrap.wrap(name, experiment_tags.SHORT_NAME_MAX, break_long_words=False))


def compact_tick_label(kernel: str) -> str:
    """:func:`kernel_tick_label` at print size: the compact name
    (:func:`experiment_tags.kernel_compact_display_name`), folded like it, never cut."""
    name = experiment_tags.kernel_compact_display_name(kernel)
    return "\n".join(textwrap.wrap(name, experiment_tags.COMPACT_NAME_MAX, break_long_words=False))


@dataclasses.dataclass(frozen=True, slots=True)
class Metric:
    """One panel's identity: its series, axis kind, label and summary reducer -- everything
    :func:`style_panel` and :func:`draw_marks` need besides the shared kernel order."""

    series: tuple[Series, ...]
    log2_space: bool
    ylabel: str
    summary_reducer: SummaryReducer
    summary_label: str
    #: How a summary point's own value is printed beside it.
    value_label: Callable[[float], str] = plotstyle.ratio_label
    #: The 1x line's colour on a ratio axis: neutral, or a named baseline's own colour.
    reference_color: str = plotstyle.REFERENCE

    @property
    def cells(self) -> tuple[KernelCell, ...]:
        """Every series' cells, for the callers that only need the kernel axis."""
        return tuple(cell for one in self.series for cell in one.cells)


def speedup_series_metric(series: Sequence[Series], ylabel: str, reference_color: str = plotstyle.REFERENCE) -> Metric:
    """A ratio panel: log2 axis, geomean summary, values printed as ratios."""
    return Metric(tuple(series), True, ylabel, summary_geomean, "Geomean", reference_color=reference_color)


def token_series_metric(series: Sequence[Series], ylabel: str) -> Metric:
    """A count panel: log10 axis, geomean summary, values printed with a magnitude suffix."""
    return Metric(tuple(series), False, ylabel, summary_geomean, "Geomean", plotstyle.decade_label)


def style_panel(
    ax: matplotlib.axes.Axes,
    metric: Metric,
    kernels: Sequence[str],
    summary_column: bool,
    label_ticks: bool,
    type_: plotstyle.TypeScale = AUTHOR_TYPE,
    slots: int = 0,
    tick_label: Callable[[str], str] = kernel_tick_label,
    summary_tick: bool = False,
) -> None:
    """Every piece of ``ax``'s chrome and none of its data: the kernel axis over ``kernels`` (a
    fixed order shared by a stacked figure's panels), the summary column and the value axis, sized
    from the cells. Drawn before any mark so :func:`fit_canvas` can measure the chrome first.

    ``slots`` reserves that many summary slots (default: one per series). ``summary_tick`` names
    the summary column with a shared x tick instead of a per-panel annotation: right only when
    every panel sharing the x axis reports the same statistic.
    """
    n = len(kernels)
    slots = slots or len(metric.series)
    ax.set_xlim(-0.6, summary_slot_x(n, slots - 1) + 0.6 if summary_column else n - 0.4)
    # A shared stacked x axis hands one panel's tick labels to all of them, so the summary tick is
    # only drawn when every panel's statistic is the same (see draw_summary_column).
    named_summary = summary_column and summary_tick
    ax.set_xticks([*range(n), summary_centre_x(n, slots)] if named_summary else list(range(n)))
    if label_ticks:
        names = [tick_label(kernel) for kernel in kernels]
        label_kernel_ticks(ax, names, metric.summary_label if named_summary else "", type_)
    else:
        ax.set_xticklabels([])
    if summary_column:
        draw_summary_column(ax, n, slots, metric.summary_label, type_, annotate=not named_summary)
    style_value_axis(ax, metric, kernels, type_)
    plotstyle.despine(ax)


def label_kernel_ticks(
    ax: matplotlib.axes.Axes, names: Sequence[str], summary: str, type_: plotstyle.TypeScale
) -> None:
    """The kernel ticks' ``names``, rotated, then -- when given -- ``summary`` horizontal under its column."""
    ax.set_xticklabels(
        [*names, summary] if summary else list(names), rotation=90, fontsize=type_.annotation_pt, linespacing=0.95
    )
    if summary:
        ax.get_xticklabels()[-1].set_rotation(0)


def style_value_axis(
    ax: matplotlib.axes.Axes, metric: Metric, kernels: Sequence[str], type_: plotstyle.TypeScale
) -> None:
    """``metric``'s value axis, limits from its cells over ``kernels``, and its label."""
    named = set(kernels)
    cells = [cell for cell in metric.cells if cell.kernel in named]
    if metric.log2_space:
        style_speedup_axis(ax, cells, type_.tick_pt, metric.reference_color)
    else:
        style_token_axis(ax, cells, type_.tick_pt)
    ax.set_ylabel(metric.ylabel, fontsize=type_.label_pt)


def draw_marks(
    ax: matplotlib.axes.Axes,
    metric: Metric,
    kernels: Sequence[str],
    style_: Style,
    summary_column: bool,
    size: float,
    span: float = DODGE_SPAN,
    type_: plotstyle.TypeScale = AUTHOR_TYPE,
    summary_values: bool = True,
) -> None:
    """Every series' cells over ``kernels``, spread by :func:`dodge_offsets`, plus each series'
    summary in its own slot, sized no smaller than the kernel marks."""
    x_of = {kernel: i for i, kernel in enumerate(kernels)}
    for one, offset in zip(metric.series, dodge_offsets(len(metric.series), span), strict=True):
        if style_ == Style.BOX and len(metric.series) == 1:
            draw_box(ax, one, x_of, size, type_)
        else:
            draw_ci(ax, one, x_of, metric.log2_space, offset, size, type_)
    if not summary_column:
        return
    summary_size = max(size, mark_size(column_pitch_in(ax), 1))
    for slot, one in enumerate(metric.series):
        cells = [cell for cell in one.cells if cell.kernel in x_of]
        x = summary_slot_x(len(kernels), slot)
        draw_summary_mark(ax, cells, x, one, metric, type_, summary_size, summary_values)


#: Marker size of a key entry, in points: sized to the key's own text, not to a figure's marks.
LEGEND_MARK_PT: float = 5.0


def status_handles(metrics: Sequence[Metric]) -> list[matplotlib.artist.Artist]:
    """The key entries for the status marks ``metrics`` actually draw: the undelivered cross and the
    pending "?", each only when some cell draws one."""
    cells = [cell for metric in metrics for cell in metric.cells]
    handles: list[matplotlib.artist.Artist] = []
    if any(not cell.delivered and not cell.pending for cell in cells):
        handles.append(
            matplotlib.lines.Line2D(
                [],
                [],
                marker="x",
                linestyle="none",
                color=plotstyle.MUTED,
                markeredgewidth=CROSS_EDGE_WIDTH,
                markersize=LEGEND_MARK_PT,
                label=plotstyle.NOT_DELIVERED_LABEL,
            )
        )
    if any(cell.pending for cell in cells):
        handles.append(plotstyle.pending_legend_mark(LEGEND_MARK_PT))
    return handles


def figure_one(
    metric: Metric,
    kernels: Sequence[str],
    style_: Style,
    summary_column: bool,
    title: str,
    width_in: float | None = None,
    legend: Sequence[matplotlib.artist.Artist] = (),
    panel_height_in: float | None = None,
    tick_label: Callable[[str], str] | None = None,
    summary_values: bool = True,
) -> matplotlib.figure.Figure:
    """A single metric's panel as its own figure (:func:`figure_panels` with one panel)."""
    return figure_panels(
        [metric], kernels, style_, summary_column, title, width_in, legend, panel_height_in=panel_height_in,
        tick_label=tick_label, summary_values=summary_values,
    )  # fmt: skip


def figure_panels(
    metrics: Sequence[Metric],
    kernels: Sequence[str],
    style_: Style,
    summary_column: bool,
    title: str,
    width_in: float | None = None,
    legend: Sequence[matplotlib.artist.Artist] = (),
    pitch_in: float | None = None,
    span: float = DODGE_SPAN,
    panel_height_in: float | None = None,
    tick_label: Callable[[str], str] | None = None,
    summary_values: bool = True,
) -> matplotlib.figure.Figure:
    """``metrics`` as panels stacked top to bottom on one kernel axis, names under the last.

    ``width_in`` draws at that width at print size, to be placed at scale 1.0; without it the
    figure is authored at double-column width, or -- given ``pitch_in`` -- as wide as that many
    inches per column plus its measured chrome. Every panel reserves the same number of summary
    slots, so a panel with fewer series still ends where the others do.
    """
    type_, width, panel_height_in, tick_label = size_defaults(width_in, panel_height_in, tick_label)
    fig, grid = plt.subplots(
        len(metrics), 1, sharex=True, figsize=(width, len(metrics) * panel_height_in), squeeze=False
    )
    fig.set_dpi(plotstyle.SAVE_DPI)  # measure at the dpi save() writes
    axes = [row[0] for row in grid]
    slots = max(len(metric.series) for metric in metrics)
    summary_tick = len({metric.summary_label for metric in metrics}) == 1
    for index, (ax, metric) in enumerate(zip(axes, metrics, strict=True)):
        style_panel(ax, metric, kernels, summary_column, index == len(axes) - 1, type_, slots, tick_label, summary_tick)
    fit_canvas(fig, axes, title, legend, type_, panel_height_in, pitch_in)
    size = mark_size(column_pitch_in(axes[0]), slots, span)
    for ax, metric in zip(axes, metrics, strict=True):
        draw_marks(ax, metric, kernels, style_, summary_column, size, span, type_, summary_values)
    return fig


def size_defaults(
    width_in: float | None, panel_height_in: float | None, tick_label: Callable[[str], str] | None
) -> tuple[plotstyle.TypeScale, float, float, Callable[[str], str]]:
    """``(type scale, width, panel height, tick label)``: print size when ``width_in`` is given,
    authored size otherwise, each overridden by the caller's own ``panel_height_in``/``tick_label``."""
    if width_in is None:
        return (
            AUTHOR_TYPE,
            plotstyle.DOUBLE_COLUMN_WIDTH,
            panel_height_in or PANEL_HEIGHT_IN,
            tick_label or kernel_tick_label,
        )
    return plotstyle.PRINT_SCALE, width_in, panel_height_in or PRINT_PANEL_HEIGHT_IN, tick_label or compact_tick_label


def split_in_two(text: str) -> str:
    """``text`` broken at the space nearest its middle: two balanced lines, or ``text`` itself when
    it has no space to break at."""
    spaces = [index for index, char in enumerate(text) if char == " "]
    if not spaces:
        return text
    cut = min(spaces, key=lambda index: abs(index - len(text) / 2.0))
    return f"{text[:cut]}\n{text[cut + 1 :]}"


def fit_ylabels(
    fig: matplotlib.figure.Figure,
    axes: Sequence[matplotlib.axes.Axes],
    panel_height_in: float,
    type_: plotstyle.TypeScale = AUTHOR_TYPE,
) -> None:
    """Keep every Y label within its own panel's height: broken onto two lines, then stepped down
    to its floor (:func:`min_text_pt`). A label still too tall at the floor is logged; the caller
    has to shorten it."""
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        label = ax.yaxis.label
        if not label.get_text() or label.get_window_extent(renderer).height / fig.dpi <= panel_height_in:
            continue
        label.set_text(split_in_two(label.get_text()))
        size = float(label.get_fontsize())
        floor = min_text_pt(type_, size)
        while label.get_window_extent(renderer).height / fig.dpi > panel_height_in and size > floor:
            size = max(floor, size - 0.25)
            label.set_fontsize(size)
        if label.get_window_extent(renderer).height / fig.dpi > panel_height_in:
            LOG.warning(
                "per_kernel: the Y label %r is taller than its panel at the %.2fpt floor", label.get_text(), size
            )


#: The least gap between two stacked panels, in inches; :func:`fit_canvas` widens it to whatever the
#: lower panel prints above its frame (its summary statistic).
STACK_GAP_IN: float = 0.2

#: The band above and below the panels while :func:`fit_canvas` measures, in inches, replaced by
#: the measured bands afterwards.
PROBE_BAND_IN: float = 0.5


def fit_canvas(
    fig: matplotlib.figure.Figure,
    axes: Sequence[matplotlib.axes.Axes],
    title: str,
    legend: Sequence[matplotlib.artist.Artist],
    type_: plotstyle.TypeScale,
    panel_height_in: float = PANEL_HEIGHT_IN,
    pitch_in: float | None = None,
) -> None:
    """Size the canvas around panels of ``panel_height_in`` each, every band measured: the left
    margin from the Y labels, the right one from the summary text, the gap and top band from what
    each panel prints above its frame, the bottom band from the kernel names and the key under
    them. ``pitch_in`` makes the canvas as wide as the kernel axis at that pitch plus the measured
    left margin, instead of fitting the axis into the canvas.
    """
    # Measure with every panel already at its final height: a rotated Y label is centred on its
    # frame, so a shorter probe frame would bill it for a band the final panel never draws.
    probe = len(axes) * panel_height_in + (len(axes) - 1) * STACK_GAP_IN + 2.0 * PROBE_BAND_IN
    fig.set_size_inches(float(fig.get_size_inches()[0]), probe)
    fig.subplots_adjust(
        top=1.0 - PROBE_BAND_IN / probe, bottom=PROBE_BAND_IN / probe, hspace=STACK_GAP_IN / panel_height_in
    )
    fig.canvas.draw()
    fit_ylabels(fig, axes, panel_height_in, type_)
    left = max(plotstyle.left_protrusion_in(fig, ax) for ax in axes) + CHROME_PAD_IN
    right = max(plotstyle.right_protrusion_in(fig, ax) for ax in axes) + CHROME_PAD_IN
    if pitch_in is not None:
        low, high = axes[0].get_xlim()
        fig.set_size_inches(left + (high - low) * pitch_in + right, float(fig.get_size_inches()[1]))
    width = float(fig.get_size_inches()[0])
    fig.subplots_adjust(left=left / width, right=1.0 - right / width)
    plotstyle.shrink_crowded_ticks(fig, [axes[-1]], type_.annotation_pt, min_text_pt(type_, type_.annotation_pt))
    names = plotstyle.below_protrusion_in(fig, axes[-1])
    above = [plotstyle.above_protrusion_in(fig, ax) for ax in axes]
    top = (plotstyle.TITLE_BAND_IN if title else 0.0) + max(above[0], 0.0) + CHROME_PAD_IN
    gap = max([STACK_GAP_IN, *(value + CHROME_PAD_IN for value in above[1:])])
    body = (axes[0].get_position().x0, axes[0].get_position().x1)
    key = (
        plotstyle.legend_below(fig, legend, y=0.005, fontsize=type_.legend_pt, markerscale=1.0, span=body)
        if legend
        else 0.0
    )
    bottom = names + CHROME_PAD_IN + (key + CHROME_PAD_IN if legend else 0.0)
    height = top + len(axes) * panel_height_in + (len(axes) - 1) * gap + bottom
    fig.set_size_inches(width, height)
    fig.subplots_adjust(top=1.0 - top / height, bottom=bottom / height, hspace=gap / panel_height_in)
    if title:
        plotstyle.title(fig, title)


def save(fig: matplotlib.figure.Figure, out: pathlib.Path, print_size: bool = False) -> pathlib.Path:
    """Write ``fig`` at its own canvas (:func:`fit_canvas` measured it), so a figure drawn at a
    ``width_in`` is placed at exactly that width rather than a tight crop of its ink; ``print_size``
    refuses type off the print scale (:func:`~hpcagent_bench.stats.style.print_type_violations`)."""
    return plotstyle.save(fig, out.with_suffix(""), fixed=True, print_size=print_size)
