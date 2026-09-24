# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-kernel figures: one column per kernel, one mark per series, one summary slot per series.

THE ONE PER-KERNEL DRAWING API. Every figure with a kernel axis draws through this module --
``statistics/plot_per_kernel.py``, the llr-focus40 comparison
(:mod:`hpcagent_bench.stats.figures.kernel_comparison`), the compiler figure
(:func:`hpcagent_bench.stats.figures.signed.llr40_figure`) and the repo-against-kernel ratios
(``statistics/plot_repo_vs_kernel.py``) -- so a column, a placeholder, a summary slot, a tick and
a margin mean the same thing in all of them. Two copies of this geometry drifted before: one
printed short names and one full names, one thinned its ticks and one did not, one summary sat in a
slot per series and one crammed every series into one column. A caller turns its own data into
:class:`KernelCell` s (:func:`kernel_cells` from a ``kernel -> value`` map), groups them into
:class:`Series`, picks one :class:`Metric` per panel and calls :func:`figure_panels`.

A CELL IS ONE KERNEL'S VALUES FOR ONE SERIES, and its status decides its mark:

* measured -- the series' own shape at the value, with an interval when there is one: the cell's
  own (a repeat interval, a min/max over tasks) or, in the ``ci`` style, a bootstrap median CI over
  its episodes, which collapses to the point below
  :data:`~hpcagent_bench.stats.summary.MIN_INTERVAL_SAMPLES` (5) -- so a one-episode-per-kernel
  experiment draws plain points, and git-scicomp's 3 episodes stay a point in ``ci`` and become a
  real box in ``box`` (:data:`MIN_EPISODES_FOR_SPREAD`), where three raw quartiles still mean
  something and a bootstrap interval of three does not;
* undelivered -- served and never verified (it still scores 1x and its tokens
  are still spent): the shape HOLLOW and CROSSED (:func:`~hpcagent_bench.stats.style.point_mark`)
  at the value the cell carries, the 1x placeholder for a missing answer or the ratio a one-sided
  failure left, so a reader never takes it for a measured 1x;
* pending -- not attempted yet: a "?" at 1x, so "not run" never reads as "failed";
* flagged -- disowned by the judge or an audit: a cross with a ``*`` at the value it claimed.

Every mark carries a white halo and is sized to the column pitch (:func:`mark_size`): a crowded
column shrinks its marks rather than merging them into one blot, down to the size at which the
undelivered cross still reads.

THE SPEED-UP AXIS IS LOG2. A ratio axis on a linear scale reads a 2x slow-down as a small event and
a 2x speed-up as a large one; log2 puts them the same distance from the 1x line, and the ticks are
labelled back into ratios (:func:`~hpcagent_bench.stats.style.ratio_tick_label`) at powers of two,
thinned to at most :data:`MAX_SPEEDUP_TICKS` (:func:`speedup_yticks`). Tokens are a magnitude, so
their axis is log10 with 1-2-5 majors (:func:`token_limits`). Both limits come from the cells, not
from autoscaling, so the chrome can be measured before a single mark is drawn.

THE SUMMARY COLUMN sits past a dashed separator, ONE SLOT PER SERIES (:func:`summary_slot_x`):
summaries that agree to a few percent, drawn in one column, hid all but the top mark. Each slot
carries the series' overall value with its interval and prints the value (tagged
:data:`~hpcagent_bench.stats.style.CLEAR_GID`, settled clear of the marks at save), over the
SOLVED kernels only (:func:`kernel_medians`): an undelivered placeholder's 1x and a disowned claim
are drawn, but neither is a measured speed-up. Speed-up is a ratio, so its overall
value is the GEOMETRIC MEAN (:func:`summary_point_speedup`) -- never a median, which equals the
geomean only when the values happen to be symmetric; tokens are not a ratio, so theirs is the
median (:func:`summary_point_tokens`). The statistic is named ABOVE the panel, never as an x tick,
which a stacked figure's shared axis would hand to the wrong panel (:func:`draw_summary_column`).

THE CANVAS IS MEASURED (:func:`fit_canvas`): the left margin from the Y labels, the top band and
the stack gap from what each panel prints above its frame, the bottom band from the rotated kernel
names (first stepped down to the pitch) plus the key under them. A fixed band is right for one
width and one label length; on any other it wastes the page or prints the key over the names. A
figure is drawn at :data:`~hpcagent_bench.stats.style.DOUBLE_COLUMN_WIDTH` at authoring size
(:data:`AUTHOR_TYPE`), at a stated ``width_in`` at print size (:data:`PRINT_TYPE`), or as wide as
a stated kernel pitch needs (:func:`roomy_pitch_in`); each panel is :data:`PANEL_HEIGHT_IN` tall.
"""

import dataclasses
import logging
import math
import pathlib
import textwrap
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

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

#: The two drawing modes: the median (+ bootstrap CI) or the raw per-episode boxplot.
Style = Literal["ci", "box"]

#: Episodes per kernel at or above which ``box`` draws an actual box instead of a point. 3 is
#: git-scicomp's own episode count -- the smallest population a quartile spread still means
#: something for, as opposed to being the two endpoints wearing quartile marks.
MIN_EPISODES_FOR_SPREAD: int = 3

#: A single panel's height, inches, for a figure authored at double-column width.
PANEL_HEIGHT_IN: float = 1.8
#: A single panel's height at print size (``width_in`` given): a text-width strip of forty kernels.
PRINT_PANEL_HEIGHT_IN: float = 1.12


@dataclasses.dataclass(frozen=True, slots=True)
class PanelType:
    """A per-kernel figure's type sizes, in points."""

    tick_pt: float
    label_pt: float
    #: The kernel names' starting size: :func:`fit_canvas` steps them down to the column pitch.
    name_pt: float
    legend_pt: float


#: Authoring sizes, for a figure drawn at the double-column width and scaled when it is placed.
AUTHOR_TYPE = PanelType(
    plotstyle.TICK_PT * 0.55, plotstyle.LABEL_PT * 0.72, plotstyle.TICK_PT * 0.5, plotstyle.TICK_PT * 0.5
)
#: Print sizes, for a figure drawn at the width it is placed at: the tick and label sizes every paper
#: figure starts from (:data:`hpcagent_bench.stats.style.PRINT_TICK_PT`), the key at tick size.
PRINT_TYPE = PanelType(
    plotstyle.PRINT_TICK_PT, plotstyle.PRINT_LABEL_PT, plotstyle.PRINT_TICK_PT, plotstyle.PRINT_LEGEND_PT
)

#: How far the kernel names may shrink below ``name_pt`` to fit the column pitch.
MIN_NAME_SCALE: float = 0.6


def min_text_pt(type_: PanelType, size: float) -> float:
    """The smallest a fitted text of ``size`` may get: the shared print floor at print size, where
    every figure of a page must agree, and :data:`MIN_NAME_SCALE` of it at authoring size."""
    return plotstyle.PRINT_MIN_PT if type_ == PRINT_TYPE else size * MIN_NAME_SCALE


#: Air between the canvas edge and the chrome :func:`fit_canvas` measures, in inches.
CHROME_PAD_IN: float = 0.04


@dataclasses.dataclass(frozen=True, slots=True)
class KernelCell:
    """One kernel's values for one series, sorted, and the status that decides its mark.

    ``delivered`` False: a kernel the arm was SERVED and never verified an
    answer for still scores 1x and its tokens are still spent. Dropping it instead would report the
    arm's speed-up over the kernels it happened to solve, which is a different and always kinder
    number -- a 28-of-40 arm would read like a 40-of-40 one. Such a cell carries the value its
    failure left (the 1x placeholder, or a ratio one side of which is that placeholder) and draws
    hollow and crossed; it never enters a summary (:func:`kernel_medians`).
    """

    kernel: str
    episodes: tuple[float, ...]
    delivered: bool = True
    #: The judge or a source audit disowned this answer, so its value is drawn but not believed
    #: (:func:`draw_flagged`). Distinct from ``delivered`` False: there IS a number here, and the
    #: point of showing it is that it is large.
    flagged: bool = False
    #: A (low, high) the caller already has for this kernel -- a confidence interval over its own
    #: repetitions, or the minimum and maximum over its tasks. ``None`` lets the ``ci`` style
    #: bootstrap one from ``episodes`` instead.
    interval: tuple[float, float] | None = None
    #: Served but not attempted yet: drawn as a "?" at 1x, so "not run" never reads as "failed".
    #: A pending cell is also undelivered, so it enters no summary either.
    pending: bool = False

    @property
    def n(self) -> int:
        return len(self.episodes)

    def median(self) -> float:
        return float(np.median(self.episodes)) if self.episodes else math.nan


def usable(value: float) -> bool:
    """Whether ``value`` is a finite positive number, the only kind a log axis can place."""
    return math.isfinite(value) and value > 0.0


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

    A kernel with no usable value is FILLED at :data:`~hpcagent_bench.stats.population.NOT_DELIVERED`
    as an undelivered cell under ``fill`` -- a speed-up, where "no verified answer" has a natural
    place -- and left out otherwise: no token count is a neutral cost, and a mark at the axis edge
    would read as the smallest spend. ``delivered`` marks the PRESENT values that are placeholders
    all the same (a ratio whose one side never delivered is a number, not a measurement). ``low``
    and ``high`` give a kernel its own interval where both are usable and ``low < high``.
    ``pending`` kernels become pending cells whatever ``values`` says.
    """
    delivered = delivered or {}
    low, high = low or {}, high or {}
    cells: list[KernelCell] = []
    for kernel in kernels:
        value = values.get(kernel, math.nan)
        if kernel in pending:
            cells.append(KernelCell(kernel, (population.NOT_DELIVERED,), delivered=False, pending=True))
        elif usable(value):
            ends = (low.get(kernel, math.nan), high.get(kernel, math.nan))
            interval = ends if usable(ends[0]) and usable(ends[1]) and ends[0] < ends[1] else None
            cells.append(KernelCell(kernel, (float(value),), delivered.get(kernel, True), interval=interval))
        elif fill:
            cells.append(KernelCell(kernel, (population.NOT_DELIVERED,), delivered=False))
    return tuple(cells)


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


def answer_cells(frame: pd.DataFrame, repeats: population.RepeatPolicy = "latest") -> list[KernelCell]:
    """One single-value cell per SOLVED kernel: its final answer under ``repeats``
    (:func:`population.kernel_answers`), the policy the tables score a kernel by. :func:`speedup_cells`
    keeps every episode instead, which is what the box style's spread needs and what a rerun kernel's
    stale first run must not contribute to."""
    answers = population.kernel_answers(frame, repeats=repeats, policy="solved")
    if "speedup" not in answers.columns:
        return []
    return [KernelCell(str(kernel), (float(value),)) for kernel, value in answers["speedup"].items() if value > 0]


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


def cell_point(cell: KernelCell, log2_space: bool) -> tuple[float, float, float]:
    """``(point, low, high)`` a measured cell draws: its own interval around its median when the
    caller gave one, else the bootstrap median CI over its episodes (:func:`bootstrap_point`)."""
    if cell.interval is not None:
        return cell.median(), *cell.interval
    return bootstrap_point(cell, log2_space)


def kernel_medians(cells: Sequence[KernelCell]) -> np.ndarray:
    """The plotted kernels' own per-kernel medians -- what the summary column reduces one level
    up -- over the SOLVED kernels only: an undelivered placeholder's 1x, a pending kernel and a
    disowned answer's claim are drawn, but none is a measured value, so none enters the summary."""
    return np.array(
        [cell.median() for cell in cells if cell.delivered and not cell.flagged and math.isfinite(cell.median())],
        dtype=np.float64,
    )


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


#: A metric's summary reducer: the plotted kernels' cells in, ``(point, low, high)`` out.
SummaryReducer = Callable[[Sequence["KernelCell"]], tuple[float, float, float]]


def drawn_values(cells: Sequence[KernelCell]) -> list[float]:
    """Every usable value ``cells`` draw a mark at -- the values a value axis has to span. Never an
    interval end: an interval is read against the axis the values set and is cut by the frame where
    it runs past it, since a two-repeat t-interval spans thirty octaves and pinning the axis to it
    flattened every mark onto one line."""
    return [v for cell in cells for v in cell.episodes if usable(v)]


#: The most labelled powers of two a speed-up axis carries. A wider range labels every second (or
#: third) octave instead: twelve octaves on a 1.8in panel printed their labels on top of each other.
MAX_SPEEDUP_TICKS: int = 7

#: Octaves of air past the outermost speed-up ticks, so a mark sitting on one is never cut by the
#: frame.
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
    that window would label fewer than two ticks: one number on an axis is nothing to read a value
    against. Whole decades instead left most of a one-decade panel empty."""
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
    the shared minor ruling (:func:`plotstyle.minor_ticks`, read off these majors) on the value axis
    and nothing on the kernel axis -- the ticks are pinned here, so the grid is drawn beside them
    rather than through ``plotstyle.value_axis``, which would relocate them."""
    ax.set_yscale("log", base=2)
    ticks = speedup_yticks(cells)
    ax.set_yticks(ticks)
    ax.set_yticklabels([plotstyle.ratio_tick_label(tick) for tick in ticks], fontsize=tick_pt)
    plotstyle.minor_ticks(ax.yaxis, "ratio")
    pad = 2.0**VALUE_PAD_OCTAVES
    ax.set_ylim(ticks[0] / pad, ticks[-1] * pad)
    ax.axhline(1.0, color=reference_color, linewidth=0.9, zorder=1)
    ax.grid(axis="y", which="major", color=plotstyle.RULE, linewidth=0.7, zorder=0)
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

    Colour is a SERIES property, because the question "is this kernel hard, or is this model bad at it" needs
    several populations over one kernel axis to answer. A series with no cells still holds its
    dodge offset and its summary slot, so a series that has nothing to show on one panel of a
    stacked figure (a compiler on the token panel) leaves every other series where the panel
    above put it.
    """

    label: str
    cells: tuple[KernelCell, ...]
    color: str
    marker: str = "o"
    filled: bool = True


#: Total x width one kernel column's series are spread over by default. Below ~0.8 the intervals of
#: adjacent kernels stay apart and the reader keeps which column a mark belongs to.
DODGE_SPAN: float = 0.62


def dodge_offsets(count: int, span: float = DODGE_SPAN) -> list[float]:
    """Evenly spaced x offsets for ``count`` series sharing one kernel column, centred on it, the
    outermost two ``span`` apart (0 stacks every series on the column).

    One series draws ON the column, not beside it; no series gets no offset, so a panel with no
    series zips against an equally empty list.
    """
    if count < 2:
        return [0.0] * count
    step = span / (count - 1)
    return [-span / 2.0 + i * step for i in range(count)]


#: A mark's diameter in points where it has room, the smallest a crowded column may shrink it to
#: (below which the undelivered cross stops reading), and how much of the gap to its neighbour a
#: mark may cover. Marks OVERLAP slightly by design: each carries a white halo, so a mark drawn over
#: its neighbour still shows its own edge.
MARK_PT: float = 4.5
MIN_MARK_PT: float = 2.6
MARK_GAP_RATIO: float = 1.7


def mark_size(pitch_in: float, n_series: int, span: float = DODGE_SPAN) -> float:
    """A mark's AREA in points squared: :data:`MARK_PT` across where the marks have room, shrinking
    with the gap to the nearest neighbour down to :data:`MIN_MARK_PT` where they do not.

    The neighbour is the next dodged series when the column is dodged, and the next column when it
    is not (``span`` 0 stacks the series and they differ by shape alone). A fixed size merged a
    crowded column into one blob and left the reader no colour edges to count series by, which is
    the one thing the dodge is for.
    """
    dodged = n_series > 1 and span > 0.0
    gap_pt = 72.0 * pitch_in * (span / (n_series - 1) if dodged else 1.0)
    return max(MIN_MARK_PT, min(MARK_PT, MARK_GAP_RATIO * gap_pt)) ** 2


def column_pitch_in(ax: matplotlib.axes.Axes) -> float:
    """Inches of ``ax``'s kernel axis per column, as it is laid out now."""
    low, high = ax.get_xlim()
    return float(ax.bbox.width) / ax.figure.dpi / (high - low)


#: Inches per kernel column for a figure whose WIDTH follows its kernels rather than the page (a
#: standalone render): the floor rotated names need at authoring size, the ceiling past which a
#: figure with many series stops printing at a readable scale, and what each extra dodged series
#: asks for in between.
MIN_PITCH_IN: float = 0.22
MAX_PITCH_IN: float = 0.34
SERIES_PITCH_IN: float = 0.05


def roomy_pitch_in(n_series: int, span: float = DODGE_SPAN) -> float:
    """The column pitch a standalone figure asks for: :data:`SERIES_PITCH_IN` between dodged
    neighbours, clamped to the name floor and the printable ceiling."""
    wanted = (n_series - 1) * SERIES_PITCH_IN / span if span > 0.0 else 0.0
    return min(MAX_PITCH_IN, max(MIN_PITCH_IN, wanted))


#: Drawn under the marks' white halos (:data:`~hpcagent_bench.stats.style.FILL_Z`), never over them:
#: the "connector under fill" order :func:`~hpcagent_bench.stats.style.point_mark` documents.
INTERVAL_Z: float = 2.0

#: The mark for a disowned answer, and the superscript that separates it from an unanswered one.
FLAGGED_MARKER: str = "X"
FLAGGED_ANNOTATION: str = "*"


def draw_flagged(ax: matplotlib.axes.Axes, x: float, value: float, color: str) -> None:
    """A disowned answer at the value it claimed: a filled cross carrying a ``*``.

    An unanswered kernel is already a crossed mark at 1x, so a reader who has learnt that mark reads
    this one as its neighbour: no credit. The ``*`` is what says the two are not the same, and the
    value is drawn where it landed because the claim being far above the honest ceiling is the whole
    observation.
    """
    ax.plot(
        [x], [value], marker=FLAGGED_MARKER, markersize=5.0, markeredgewidth=1.4, color=color,
        linestyle="none", zorder=4,
    )  # fmt: skip
    ax.annotate(
        FLAGGED_ANNOTATION, (x, value), textcoords="offset points", xytext=(3.5, 2.0), color=color,
        fontsize=7.0, ha="left", va="bottom", zorder=4, annotation_clip=False,
    )  # fmt: skip


def draw_status(ax: matplotlib.axes.Axes, cell: KernelCell, x: float, one: Series, size: float) -> bool:
    """Draw ``cell`` as the mark its status calls for (pending, undelivered, flagged) and say so; a
    measured cell is left to the caller's style. One place decides these marks, so the ``ci`` and
    the ``box`` style cannot disagree about what a failure looks like."""
    if cell.pending:
        plotstyle.pending_mark(ax, x, population.NOT_DELIVERED, one.color, size=size)
    elif not cell.delivered:
        plotstyle.point_mark(ax, x, cell.median(), one.color, one.marker, filled=False, size=size, delivered=False)
    elif cell.flagged:
        draw_flagged(ax, x, cell.median(), one.color)
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
) -> None:
    """``one``'s cells over the kernels of ``x_of``: each measured cell a point with its interval
    (:func:`cell_point`), each other cell the mark its status calls for (:func:`draw_status`)."""
    for cell in one.cells:
        if cell.kernel not in x_of:
            continue
        x = x_of[cell.kernel] + offset
        if draw_status(ax, cell, x, one, size):
            continue
        point, low, high = cell_point(cell, log2_space)
        if math.isfinite(low) and math.isfinite(high) and low < high:
            ax.vlines(x, low, high, color=one.color, linewidth=1.0, alpha=0.6, zorder=INTERVAL_Z)
        plotstyle.point_mark(ax, x, point, one.color, one.marker, one.filled, size=size)


def draw_box(ax: matplotlib.axes.Axes, one: Series, x_of: Mapping[str, int], size: float = MARK_PT**2) -> None:
    """A real box for a kernel with :data:`MIN_EPISODES_FOR_SPREAD`+ episodes; a point otherwise --
    mixing the two in one panel is deliberate (see the module docstring)."""
    boxed: list[KernelCell] = []
    for cell in one.cells:
        if cell.kernel not in x_of or draw_status(ax, cell, x_of[cell.kernel], one, size):
            continue
        if cell.n >= MIN_EPISODES_FOR_SPREAD:
            boxed.append(cell)
        else:
            plotstyle.point_mark(ax, x_of[cell.kernel], cell.median(), one.color, one.marker, one.filled, size=size)
    if not boxed:
        return
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
        box.set(facecolor=one.color, edgecolor=one.color, alpha=0.55, linewidth=0.6)
    for part in ("whiskers", "caps"):
        for line in artists[part]:
            line.set(color=one.color, linewidth=0.6)


#: Gap (in x-axis units) between the last kernel column and the dashed separator, and between the
#: separator and the summary column.
SUMMARY_GAP: float = 0.7

#: X distance between consecutive series' summary marks, in kernel columns. Each series gets a slot
#: of its own: summaries that agree to a few percent, drawn in one column, hid all but the top mark.
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
    ax: matplotlib.axes.Axes, n_kernels: int, slots: int, label: str, label_pt: float, annotate: bool = True
) -> None:
    """The dashed separator and a small label ABOVE the column's slots naming its own statistic.

    The label is an annotation, never an x-axis TICK label: a stacked figure shares one x axis
    between its panels (:func:`figure_panels`) and matplotlib shares the same tick label text for
    every row sharing that axis, so a per-panel tick label silently loses whichever panel drew first
    -- the top panel's "Geomean" was overwritten by the bottom panel's "Median". An annotation
    anchored to the panel's own data coordinates has no such sharing. Where every panel's statistic
    is the same, :func:`style_panel` names it once as an x tick instead and ``annotate`` is off.
    """
    ax.axvline(summary_separator_x(n_kernels), color=plotstyle.RULE, linestyle=(0, (3, 3)), linewidth=1.0, zorder=1)
    if not annotate:
        return
    centre = summary_centre_x(n_kernels, slots)
    ax.annotate(
        label, xy=(centre, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3), textcoords="offset points",
        ha="center", va="bottom", fontsize=label_pt, color=plotstyle.MUTED, annotation_clip=False,
    )  # fmt: skip


def draw_summary_mark(
    ax: matplotlib.axes.Axes,
    cells: Sequence[KernelCell],
    x: float,
    one: Series,
    metric: "Metric",
    value_pt: float,
    size: float,
    value: bool = True,
) -> None:
    """One series' summary: the reducer's point with its interval, and -- when ``value`` -- the
    point's own value printed above the interval in the series' colour, so the number a caption
    quotes is on the figure. The value settles clear of the marks and inside the frame at save time
    (:func:`~hpcagent_bench.stats.style.settle_clear_labels`). Six or more series overprint their
    values in the narrow summary column; such a figure gives the numbers in its caption instead."""
    point, low, high = metric.summary_reducer(cells)
    if not math.isfinite(point):
        return
    if math.isfinite(low) and math.isfinite(high) and low < high:
        ax.vlines(x, low, high, color=one.color, linewidth=1.3, alpha=0.7, zorder=INTERVAL_Z)
    plotstyle.point_mark(ax, x, point, one.color, one.marker, one.filled, size=size)
    if not value:
        return
    ax.annotate(
        metric.value_label(point), xy=(x, high if math.isfinite(high) else point), xytext=(0, 2),
        textcoords="offset points", rotation=90, ha="center", va="bottom", fontsize=value_pt, color=one.color,
        gid=plotstyle.CLEAR_GID, annotation_clip=False,
    )  # fmt: skip


def kernel_tick_label(kernel: str) -> str:
    """The kernel's short manifest name (:func:`experiment_tags.kernel_short_display_name`). A kernel
    with no short name falls back to its full name, folded at the short-name limit onto as many
    lines as it needs -- never cut: a truncated name ("2-D Jacobi stencil..") no longer names one
    kernel, and the band under the panel is measured from whatever depth the names take
    (:func:`fit_canvas`). A word longer than the limit keeps its own line whole."""
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
    :func:`style_panel` and :func:`draw_marks` need besides the shared kernel order.

    ``summary_reducer`` and ``summary_label`` carry the statistic the SUMMARY COLUMN is under: the
    geomean for a ratio (speed-up), the median for a count (tokens) -- see
    :func:`summary_point_speedup` and :func:`summary_point_tokens`.
    """

    series: tuple[Series, ...]
    log2_space: bool
    ylabel: str
    summary_reducer: SummaryReducer
    summary_label: str
    #: How a summary point's own value is printed beside it.
    value_label: Callable[[float], str] = plotstyle.ratio_label
    #: The 1x line's colour on a ratio axis: neutral, or the baseline's own colour where the line IS
    #: a named baseline.
    reference_color: str = plotstyle.REFERENCE

    @property
    def cells(self) -> tuple[KernelCell, ...]:
        """Every series' cells, for the callers that only need the kernel axis."""
        return tuple(cell for one in self.series for cell in one.cells)


def speedup_series_metric(series: Sequence[Series], ylabel: str, reference_color: str = plotstyle.REFERENCE) -> Metric:
    """A ratio panel: log2 axis, geomean summary, values printed as ratios."""
    return Metric(tuple(series), True, ylabel, summary_point_speedup, "Geomean", reference_color=reference_color)


def token_series_metric(series: Sequence[Series], ylabel: str) -> Metric:
    """A count panel: log10 axis, median summary, values printed with a magnitude suffix."""
    return Metric(tuple(series), False, ylabel, summary_point_tokens, "Median", plotstyle.decade_label)


def style_panel(
    ax: matplotlib.axes.Axes,
    metric: Metric,
    kernels: Sequence[str],
    summary_column: bool,
    label_ticks: bool,
    type_: PanelType = AUTHOR_TYPE,
    slots: int = 0,
    tick_label: Callable[[str], str] = kernel_tick_label,
    summary_tick: bool = False,
) -> None:
    """Every piece of ``ax``'s chrome and none of its data: the kernel axis over ``kernels`` (a FIXED
    order, so a stacked figure's panels share x), the summary column's separator and statistic, and
    the value axis with limits taken from the cells. Drawn before any mark so :func:`fit_canvas` can
    measure the chrome and the marks can then be sized to the pitch it leaves.

    ``slots`` reserves that many summary slots (default: one per series), so every panel of a
    stacked figure ends at the same x even when one carries fewer series. ``tick_label`` spells a
    kernel's tick (default :func:`kernel_tick_label`). ``summary_tick`` names the summary column
    with an x tick under it (horizontal, e.g. "Geomean") instead of an annotation above it: right
    only when every panel sharing the x axis reports the same statistic (:func:`figure_panels`).
    """
    n = len(kernels)
    slots = slots or len(metric.series)
    ax.set_xlim(-0.6, summary_slot_x(n, slots - 1) + 0.6 if summary_column else n - 0.4)
    # A shared stacked x axis hands one panel's tick labels to all of them (draw_summary_column's
    # docstring), so the summary tick is only drawn when every panel's statistic is the same.
    named_summary = summary_column and summary_tick
    ax.set_xticks([*range(n), summary_centre_x(n, slots)] if named_summary else list(range(n)))
    if label_ticks:
        # The tick is the kernel's short NAME; ``kernels`` are the identifiers the columns and the
        # results table are keyed by.
        names = [tick_label(kernel) for kernel in kernels] + ([metric.summary_label] if named_summary else [])
        ax.set_xticklabels(names, rotation=90, fontsize=type_.name_pt, linespacing=0.95)
        if named_summary:
            ax.get_xticklabels()[-1].set_rotation(0)
    else:
        ax.set_xticklabels([])
    if summary_column:
        draw_summary_column(ax, n, slots, metric.summary_label, type_.name_pt, annotate=not named_summary)
    named = set(kernels)
    cells = [cell for cell in metric.cells if cell.kernel in named]
    if metric.log2_space:
        style_speedup_axis(ax, cells, type_.tick_pt, metric.reference_color)
    else:
        style_token_axis(ax, cells, type_.tick_pt)
    ax.set_ylabel(metric.ylabel, fontsize=type_.label_pt)
    plotstyle.despine(ax)


def draw_marks(
    ax: matplotlib.axes.Axes,
    metric: Metric,
    kernels: Sequence[str],
    style_: Style,
    summary_column: bool,
    size: float,
    span: float = DODGE_SPAN,
    type_: PanelType = AUTHOR_TYPE,
    summary_values: bool = True,
) -> None:
    """Every series' cells over ``kernels``, spread by :func:`dodge_offsets`, plus each series'
    summary in its own slot (its value printed when ``summary_values``). One series keeps the column's exact x, so a single-series panel reads
    as the plain strip it always was. Summary marks sit alone in their slots, so they take the
    roomiest size a mark gets, never smaller than the kernel marks."""
    x_of = {kernel: i for i, kernel in enumerate(kernels)}
    for one, offset in zip(metric.series, dodge_offsets(len(metric.series), span), strict=True):
        if style_ == "box" and len(metric.series) == 1:
            draw_box(ax, one, x_of, size)
        else:
            draw_ci(ax, one, x_of, metric.log2_space, offset, size)
    if not summary_column:
        return
    summary_size = max(size, mark_size(column_pitch_in(ax), 1))
    for slot, one in enumerate(metric.series):
        cells = [cell for cell in one.cells if cell.kernel in x_of]
        x = summary_slot_x(len(kernels), slot)
        draw_summary_mark(ax, cells, x, one, metric, type_.name_pt, summary_size, summary_values)


def draw_panel(
    ax: matplotlib.axes.Axes,
    metric: Metric,
    kernels: Sequence[str],
    style_: Style,
    summary_column: bool,
    label_ticks: bool,
    type_: PanelType = AUTHOR_TYPE,
    span: float = DODGE_SPAN,
) -> None:
    """One metric's panel on an axes the caller laid out: its chrome (:func:`style_panel`), then its
    marks sized to the pitch the axes has now (:func:`draw_marks`). :func:`figure_panels` measures
    the canvas between the two instead, which is what a figure to be saved wants."""
    style_panel(ax, metric, kernels, summary_column, label_ticks, type_)
    size = mark_size(column_pitch_in(ax), len(metric.series), span)
    draw_marks(ax, metric, kernels, style_, summary_column, size, span, type_)


#: Marker size of a key entry, in points: sized to the key's own text, not to a figure's marks.
LEGEND_MARK_PT: float = 5.0


def status_handles(metrics: Sequence[Metric]) -> list[matplotlib.artist.Artist]:
    """The key entries for the status marks ``metrics`` actually draw: the undelivered cross, and the
    pending "?", each only when some cell draws one -- an entry for a mark that is not on the figure
    is one more thing to read and find nowhere.

    The entry shows the CROSS, not the hollow shape the mark also has: hollow is this repo's
    spelling for a control, so an entry that showed only that would name the wrong thing.
    """
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
                markeredgewidth=1.2,
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
    """``metrics`` as panels stacked top to bottom on ONE kernel axis, names under the last.

    ``width_in`` draws at that width at print size (:data:`PRINT_TYPE`), to be placed at scale 1.0;
    without it the figure is authored at :data:`~hpcagent_bench.stats.style.DOUBLE_COLUMN_WIDTH`
    (:data:`AUTHOR_TYPE`), or -- given ``pitch_in`` -- as wide as that many inches per column plus
    its measured chrome. ``legend`` is a key drawn under the names; ``span`` is the dodge
    (:func:`dodge_offsets`). Every panel reserves the same number of summary slots, so a panel with
    fewer series still ends where the others do. ``tick_label`` spells the kernel names; when every
    panel reports the same summary statistic, it is named once as an x tick under its column.

    At print size the panels default to :data:`PRINT_PANEL_HEIGHT_IN` and the names to
    :func:`compact_tick_label`; authored, to :data:`PANEL_HEIGHT_IN` and :func:`kernel_tick_label`.
    """
    type_ = AUTHOR_TYPE if width_in is None else PRINT_TYPE
    if panel_height_in is None:
        panel_height_in = PANEL_HEIGHT_IN if width_in is None else PRINT_PANEL_HEIGHT_IN
    if tick_label is None:
        tick_label = kernel_tick_label if width_in is None else compact_tick_label
    width = plotstyle.DOUBLE_COLUMN_WIDTH if width_in is None else width_in
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
    type_: PanelType = AUTHOR_TYPE,
) -> None:
    """Keep every Y label within its own panel's height: broken onto two lines, then stepped down
    to its floor (:func:`min_text_pt`). A rotated label taller than its panel runs past both ends
    of the frame, and in a stack the two panels' labels printed over each other in the gap. A label
    still too tall at the floor is reported: the caller has to shorten it."""
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

#: The band above and below the panels while :func:`fit_canvas` measures, in inches: room for the
#: chrome to be drawn and read, replaced by the measured bands afterwards.
PROBE_BAND_IN: float = 0.5


def fit_canvas(
    fig: matplotlib.figure.Figure,
    axes: Sequence[matplotlib.axes.Axes],
    title: str,
    legend: Sequence[matplotlib.artist.Artist],
    type_: PanelType,
    panel_height_in: float = PANEL_HEIGHT_IN,
    pitch_in: float | None = None,
) -> None:
    """Size the canvas around panels of ``panel_height_in`` each, every band MEASURED: the left
    margin from the Y labels (first fitted to the panel height, :func:`fit_ylabels`), the right one from the text past the last column (the summary's
    statistic, centred on its slots), the gap and top band from what each panel prints above its frame, the
    bottom band from the kernel names (stepped down to the column pitch first) and the key under
    them. Fixed fractions put the names over the key on a narrow page and wasted columns beside a
    short Y label. ``pitch_in`` makes the canvas as wide as the kernel axis at that pitch plus the
    measured left margin, instead of fitting the axis into the canvas.

    The value labels a summary prints are not measured: they settle inside the frame at save
    (:func:`~hpcagent_bench.stats.style.settle_clear_labels`) and need no band of their own.
    """
    # Measure with every panel already at its final height: a rotated Y label is centred on its
    # frame, so on the shorter frame subplots() starts with it reached past both ends and was billed
    # as a band the final panel never draws.
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
    plotstyle.shrink_crowded_ticks(fig, [axes[-1]], type_.name_pt, min_text_pt(type_, type_.name_pt))
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


def save(fig: matplotlib.figure.Figure, out: pathlib.Path, print_size: bool = False) -> pathlib.Path:
    """Write ``fig`` at its own canvas (:func:`fit_canvas` measured it), so a figure drawn at a
    ``width_in`` is placed at exactly that width rather than a tight crop of its ink; ``print_size``
    refuses type off the print scale (:func:`~hpcagent_bench.stats.style.print_type_violations`)."""
    return plotstyle.save(fig, out.with_suffix(""), fixed=True, print_size=print_size)
