# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared visual style for every figure: type sizes, tick and spine weight, grid colour, neutral
inks. Colour belongs to the entity and lives in :mod:`hpcagent_bench.stats.palette`.
"""

import enum
import dataclasses
import itertools
import logging
import math
import pathlib
from collections.abc import Iterator, Sequence
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.axis import Axis, YAxis
from matplotlib.backend_bases import RendererBase
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Annotation, Text
from matplotlib.ticker import FuncFormatter, Locator, LogLocator, MaxNLocator, NullFormatter
from matplotlib.transforms import Bbox, Transform

LOG = logging.getLogger(__name__)

# matplotlib's drawing calls end in an untyped ``**kwargs``: every call below suppresses that report.

#: Ink, in decreasing emphasis. Every text uses INK; MUTED is for non-text marks (legend swatch,
#: tick dash, an interval that would compete with the data).
INK: str = "#1c1c1e"
MUTED: str = "#6b6b70"
RULE: str = "#d6d6da"
#: Text that labels the chart rather than the data (quadrant captions and the like).
FAINT: str = "#a8a8ae"
#: The zero/parity reference line, darker than the grid because it is a statement.
REFERENCE: str = "#3a3a3e"
#: Minor grid colour and line weight (:func:`minor_ticks`), lighter and thinner than the major grid.
MINOR_RULE: str = "#e8e8ea"
MINOR_GRID_WIDTH: float = 0.25
#: A minor tick mark against a major one, as fractions of the major's length and line width.
MINOR_TICK_LENGTH: float = 0.55
MINOR_TICK_WIDTH: float = 0.6


@dataclasses.dataclass(frozen=True, slots=True)
class StatInk:
    """Colours of statistics and states, never of entities (palette.py owns those)."""

    median: str = "#3b6fd4"  # median bar; blue so the geomean tick reads against it
    geomean: str = "#d4772a"  # geomean tick; orange, far from the median blue


STAT_INK = StatInk()

# Every label uses Title Case, except articles/conjunctions/prepositions inside it; identifiers
# and mathtext keep their own spelling. Not enforced in code.

#: Type scale, in points, sized for print: a paper reproduces a figure at about half width, so
#: 13pt ticks reach the page around 6.5pt.
TITLE_PT: float = 20.0
#: Floor for the shrink in :func:`title`.
MIN_TITLE_PT: float = 8.0
SUBTITLE_PT: float = 13.0
LABEL_PT: float = 16.0
TICK_PT: float = 14.0
ANNOTATION_PT: float = 13.0

#: Type sizes at placed width (scale 1.0): ticks/category names at PRINT_TICK_PT, axis/panel
#: labels at PRINT_LABEL_PT, legends at PRINT_LEGEND_PT.
PRINT_TICK_PT: float = 7.0
PRINT_LABEL_PT: float = 8.0
PRINT_LEGEND_PT: float = 6.0
#: Floor for shrinking print text to fit; below it, change the layout instead.
PRINT_MIN_PT: float = 5.5

#: Default dpi for :func:`save`. Text measurement should set the figure to this dpi first: FreeType
#: hints differently at matplotlib's default 100 dpi.
SAVE_DPI: float = 200.0

#: Full text width of a double-column A4 page, in inches.
DOUBLE_COLUMN_WIDTH: float = 7.0

#: Per-paper page width budgets, in inches, so a figure drops in at scale 1.0 instead of being
#: rescaled by ``\includegraphics`` (ICLR: iclr2027_conference.sty; ACM: acmart.cls sigconf).
ICLR_TEXT_WIDTH_IN: float = 5.5
ACM_COLUMN_WIDTH_IN: float = 3.33
ACM_TEXT_WIDTH_IN: float = 7.0

#: A wrapfigure's body width/height including axis chrome; shared so wrap figures share one box.
ICLR_WRAP_WIDTH_IN: float = 0.45 * ICLR_TEXT_WIDTH_IN
PRINT_BODY_HEIGHT_IN: float = 1.9

#: Allowed drift between a saved paper figure's width and its placed width.
PLACED_WIDTH_RTOL: float = 0.01


@dataclasses.dataclass(frozen=True, slots=True)
class TypeScale:
    """One figure's type sizes (points) and the mark and line weights that go with them.

    :data:`AUTHOR_SCALE` is for a figure meant to be shrunk; :data:`PRINT_SCALE` for one drawn at
    its placed width. A figure module picks one and never mixes them.
    """

    tick_pt: float
    label_pt: float
    title_pt: float
    legend_pt: float
    annotation_pt: float
    line_width: float
    marker_size: float

    @property
    def hairline_width(self) -> float:
        """A secondary stroke (median tick, whisker, reference line): half the data line."""
        return self.line_width / 2.0

    @property
    def point_size(self) -> float:
        """A raw-sample dot or a flier: half a data mark, so it never reads as a summary."""
        return self.marker_size / 2.0


AUTHOR_SCALE = TypeScale(TICK_PT, LABEL_PT, SUBTITLE_PT, TICK_PT, ANNOTATION_PT, 1.6, 6.0)
PRINT_SCALE = TypeScale(PRINT_TICK_PT, PRINT_LABEL_PT, PRINT_LABEL_PT, PRINT_LEGEND_PT, PRINT_TICK_PT, 1.0, 4.0)


def text_sizes(fig: Figure) -> list[tuple[str, float]]:
    """``(text, size in points)`` of every visible, non-empty text the figure draws, legends included."""
    return [
        (artist.get_text(), float(artist.get_fontsize()))
        for artist in fig.findobj(Text)
        if artist.get_visible() and artist.get_text().strip()
    ]


def print_type_violations(fig: Figure) -> list[tuple[str, float]]:
    """The texts of ``fig`` outside the print range [:data:`PRINT_MIN_PT`, :data:`PRINT_LABEL_PT`]."""
    return [(text, size) for text, size in text_sizes(fig) if not PRINT_MIN_PT - 1e-6 <= size <= PRINT_LABEL_PT + 1e-6]


def apply() -> None:
    """Set the process-wide rcParams. Idempotent; call it before creating a figure."""
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK,
            "axes.labelsize": LABEL_PT,
            "axes.titlesize": LABEL_PT + 1,
            "axes.titlecolor": INK,
            "axes.grid": False,  # each plot opts in on one axis; a full grid is noise
            "axes.axisbelow": True,  # data over guides, never the reverse
            "grid.color": RULE,
            "grid.linewidth": 0.4,  # the grid is a guide, not a mark
            "xtick.color": MUTED,  # the tick dash stays a guide
            "ytick.color": MUTED,
            "xtick.labelcolor": INK,  # its number is text, and text is ink
            "ytick.labelcolor": INK,
            "xtick.labelsize": TICK_PT,
            "ytick.labelsize": TICK_PT,
            "legend.frameon": False,
            "legend.fontsize": LABEL_PT,
            "text.color": INK,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,  # embed as TrueType so the PDF's text stays selectable
            "ps.fonttype": 42,
        }
    )


def despine(ax: Axes, keep: tuple[str, ...] = ("top", "right", "left", "bottom")) -> None:
    """Colour the kept spines RULE and hide the rest. Defaults to a light grey four-sided frame."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
        if side in keep:
            ax.spines[side].set_color(RULE)


#: Where :func:`title` puts a figure title: top edge below the canvas top, plot area further down
#: by TITLE_BAND_IN.
TITLE_TOP_IN: float = 0.34
TITLE_GAP_IN: float = 0.30
TITLE_BAND_IN: float = TITLE_TOP_IN + TITLE_GAP_IN


def title(fig: Figure, text: str) -> float:
    """Centred title; returns the top of the plot area for ``tight_layout(rect=...)``."""
    width, height = (float(value) for value in fig.get_size_inches())
    # Work in inches, then convert: a figure fraction is a different gap at every figure height.
    top = 1.0 - (TITLE_TOP_IN / height)
    artist = fig.text(0.5, top, text, fontsize=TITLE_PT, color=INK, ha="center", va="top")  # pyright: ignore[reportUnknownMemberType]
    # A title wider than the canvas is clipped at both ends: shrink it to fit.
    size = TITLE_PT
    while size > MIN_TITLE_PT:
        box = artist.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        if box.width <= width:
            break
        size -= 0.5
        artist.set_fontsize(size)
    return max(0.5, top - TITLE_GAP_IN / height)


#: ``columnspacing`` and ``handlelength`` of a key that must fit two columns in a wrap figure.
COMPACT_KEY: dict[str, float] = {"columnspacing": 0.8, "handlelength": 1.2}


def legend_below(
    fig: Figure,
    handles: Sequence[Artist],
    ncol: int = 0,
    y: float = 0.0,
    fontsize: float = 0.0,
    markerscale: float = 1.4,
    span: tuple[float, float] | None = None,
    columnspacing: float = 1.6,
    handlelength: float = 2.0,
) -> float:
    """One legend, centred under the whole figure and wrapped to its width; never inside the axes.
    Returns the legend's height in inches, for the caller's bottom margin.

    The requested column count is a ceiling: a row too wide for the canvas drops a column until it
    fits. ``fontsize``/``markerscale`` override the defaults for a figure too short to afford them.
    ``span``, the plot body's (left, right) in figure fractions, makes the body's width the limit
    instead of the canvas, and centres the legend on it.
    """
    columns = ncol if ncol != 0 else min(len(handles), 5)
    left, right = span if span is not None else (0.0, 1.0)
    limit = (right - left) * float(fig.get_size_inches()[0])
    while True:
        legend = fig.legend(  # pyright: ignore[reportUnknownMemberType]
            handles=handles,
            loc="lower center" if y != 0.0 else "upper center",
            bbox_to_anchor=((left + right) / 2.0, y),
            ncol=columns,
            frameon=False,
            fontsize=fontsize if fontsize > 0.0 else LABEL_PT,
            markerscale=markerscale,
            handletextpad=0.5,
            columnspacing=columnspacing,
            handlelength=handlelength,
            borderaxespad=0.0,
            # matplotlib's default 0.4 padding reads as a blank band under the ticks.
            borderpad=0.1,
        )
        box = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        if columns <= 1 or box.width <= limit:
            # Fill the box: the fewest columns that give this many rows (12 entries: 4, not 5).
            full = -(-len(handles) // max(1, -(-len(handles) // columns)))
            if full < columns:
                legend.remove()
                columns = full
                continue
            return float(box.height)
        legend.remove()
        columns -= 1


def decade_label(value: float, position: int = 0) -> str:
    """A log-axis major as a plain number with a magnitude suffix: 500K, 1M, 2.5M.

    Not scientific notation: matplotlib's own formatter also declines to label anything but 1x/2x
    of a decade. Falls back to a plain number below a thousand.
    """
    if value <= 0:
        return ""
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= threshold:
            scaled = value / threshold
            # Trim a trailing ".0": "1M" reads as a round number, "1.0M" as a measurement.
            return f"{scaled:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{value:g}"


def ratio_tick_label(value: float) -> str:
    """A ratio tick at full precision: ``0.25 -> "0.25x"``, ``1.0 -> "1x"``, ``4.0 -> "4x"``."""
    if value == 1.0:
        return "1x"
    if value > 1.0:
        return f"{value:g}x"
    return f"{float(f'{value:.3g}'):g}x"


def ratio_tick(value: float, position: int = 0) -> str:
    """:func:`ratio_tick_label` as a :class:`~matplotlib.ticker.FuncFormatter` on a log ratio axis."""
    del position
    return ratio_tick_label(value)


def log2_ratio_tick(value: float, position: int = 0) -> str:
    """:func:`ratio_tick_label` for an axis that holds ``log2(ratio)`` (``+1`` is 2x, ``-1`` 0.5x)."""
    del position
    return ratio_tick_label(2.0**value)


def ratio_label(value: float) -> str:
    """A measured ratio to one decimal (``6.3x``, ``32.5x``), or one significant figure below
    0.1x so a real slowdown does not print as ``0.0x``."""
    if not math.isfinite(value) or value <= 0.0:
        return ""
    return f"{value:.1f}x" if value >= 0.1 else f"{value:.1g}x"


#: What a value axis holds, for minor-tick purposes (:func:`minor_ticks`): ``ratio`` (log2 axis in
#: ratio units, majors at powers of two), ``log2`` (linear ``log2(ratio)`` axis, majors at whole
#: exponents), ``token`` (log10 axis), ``count`` (linear count 0..N).
class MinorKind(enum.Enum):
    RATIO = "ratio"
    LOG2 = "log2"
    TOKEN = "token"
    COUNT = "count"


#: Minor positions inside one octave of a ratio axis, as multiples of the lower major (the
#: integers 5x, 6x, 7x between 4x and 8x).
OCTAVE_SUBS: tuple[float, ...] = (1.25, 1.5, 1.75)

#: Float tolerance for treating a minor as landing on a major.
OCTAVE_TOLERANCE: float = 1e-6


def ratio_minor_exponents(majors: Sequence[float], low: float, high: float) -> list[float]:
    """Minor ticks of a ratio axis inside ``[low, high]`` (majors, limits and result all in log2
    exponent units).

    Spacing is read off ``majors``: more than an octave apart gets a minor at every octave between
    them; one octave apart gets :data:`OCTAVE_SUBS` inside it. Fewer than two majors, or majors
    under an octave apart, get none.
    """
    exponents = sorted(set(majors))
    if len(exponents) < 2:
        return []
    step = min(b - a for a, b in itertools.pairwise(exponents))
    candidates = ratio_minor_candidates(step, range(math.floor(low), math.ceil(high) + 1))
    return [
        value for value in candidates
        if low <= value <= high
        and not any(math.isclose(value, exponent, abs_tol=OCTAVE_TOLERANCE) for exponent in exponents)
    ]  # fmt: skip


def ratio_minor_candidates(step: float, octaves: range) -> list[float]:
    """Minor exponents over ``octaves`` for majors ``step`` octaves apart."""
    if step > 1.0 + OCTAVE_TOLERANCE:
        return [float(octave) for octave in octaves]
    if step > 1.0 - OCTAVE_TOLERANCE:
        return [octave + math.log2(sub) for octave in octaves for sub in OCTAVE_SUBS]
    return []


def token_minor_values(majors: Sequence[float], low: float, high: float) -> list[float]:
    """Minor ticks of a log10 token axis inside ``[low, high]``: whole multiples of a power of ten
    that are not majors."""
    if low <= 0.0 or high <= 0.0:
        return []
    decades = range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
    return [
        value for value in (multiple * 10.0**decade for decade in decades for multiple in range(1, 10))
        if low <= value <= high and not any(math.isclose(value, major, rel_tol=1e-9) for major in majors)
    ]  # fmt: skip


#: Divisions tried for a count axis' major step, first whole-number division wins (quarters,
#: fifths, thirds, halves).
COUNT_DIVISIONS: tuple[int, ...] = (4, 5, 3, 2)


def count_minor_values(majors: Sequence[float], low: float, high: float) -> list[float]:
    """Minor ticks of a linear count axis inside ``[low, high]``: the major step split into the
    first of :data:`COUNT_DIVISIONS` that gives a whole step; none if no division is whole."""
    values = sorted(set(majors))
    if len(values) < 2:
        return []
    step = min(b - a for a, b in itertools.pairwise(values))
    minor = next((step / parts for parts in COUNT_DIVISIONS if float(step / parts).is_integer()), 0.0)
    if minor < 1.0:
        return []
    start = math.ceil(low / minor) * minor
    candidates = [start + index * minor for index in range(int((high - start) // minor) + 1)]
    return [value for value in candidates if not any(math.isclose(value, major) for major in values)]


def minor_positions(kind: MinorKind, majors: Sequence[float], low: float, high: float) -> list[float]:
    """The minors of a ``kind`` axis inside ``[low, high]``, majors and limits in the axis' own units."""
    if kind == MinorKind.COUNT:
        return count_minor_values(majors, low, high)
    if kind == MinorKind.TOKEN:
        return token_minor_values(majors, low, high)
    if kind == MinorKind.LOG2:
        return ratio_minor_exponents(majors, low, high)
    if low <= 0.0:
        return []
    positive = [math.log2(value) for value in majors if value > 0.0]
    return [2.0**exponent for exponent in ratio_minor_exponents(positive, math.log2(low), math.log2(high))]


class MinorLocator(Locator):
    """:func:`minor_positions` as a matplotlib locator, derived fresh from the axis' current majors
    and view on every draw."""

    def __init__(self, kind: MinorKind) -> None:
        self.kind: MinorKind = kind

    def __call__(self) -> Sequence[float]:
        if not isinstance(self.axis, Axis):
            return []
        low, high = (float(value) for value in self.axis.get_view_interval())
        return self.tick_values(low, high)

    def tick_values(self, vmin: float, vmax: float) -> Sequence[float]:
        if not isinstance(self.axis, Axis):
            return []
        low, high = sorted((float(vmin), float(vmax)))
        majors = [float(value) for value in self.axis.get_major_locator().tick_values(low, high)]
        return minor_positions(self.kind, majors, low, high)


def minor_ticks(axis: Axis, kind: MinorKind, color: str = MINOR_RULE, width: float = MINOR_GRID_WIDTH) -> None:
    """Unlabelled minor ticks and a light minor grid on the value axis ``axis``.

    Marks point the way majors do, shorter and thinner (:data:`MINOR_TICK_LENGTH`,
    :data:`MINOR_TICK_WIDTH`), and carry no label. Never used on a category axis.
    """
    axis.set_minor_locator(MinorLocator(kind))
    axis.set_minor_formatter(NullFormatter())
    vertical = isinstance(axis, YAxis)
    axis.set_tick_params(
        which="minor",
        length=MINOR_TICK_LENGTH * float(matplotlib.rcParams["ytick.major.size" if vertical else "xtick.major.size"]),
        width=MINOR_TICK_WIDTH * float(matplotlib.rcParams["ytick.major.width" if vertical else "xtick.major.width"]),
    )
    axis.grid(True, which="minor", color=color, linewidth=width, zorder=0)  # pyright: ignore[reportUnknownMemberType]


def value_axis(ax: Axes, axis: Literal["x", "y"] = "y", log_base: float = 10.0, major: bool = True) -> None:
    """Ticks and a major grid for the axis carrying the measured quantity; on a log axis, also the
    shared minor ruling (:func:`minor_ticks`).

    ``log_base`` is passed rather than read off the axis: matplotlib keeps it under a private name,
    and a wrong guess puts gridlines at the wrong ratios.
    """
    target: Axis = ax.yaxis if axis == "y" else ax.xaxis
    scale: str = ax.get_yscale() if axis == "y" else ax.get_xscale()
    if scale == "log":
        if major:
            # A decade gets 1, 2, 5 (more than one labelled tick even under two decades). A base-2
            # (ratio) axis gets 1 only: a 1.5 sub would label ticks a reader cannot place on a log2
            # grid by eye (:func:`ratio_tick_label`).
            subs = (1.0, 2.0, 5.0) if log_base == 10.0 else (1.0,)
            target.set_major_locator(LogLocator(base=log_base, subs=subs, numticks=20))
        if log_base == 10.0 and major:
            # Not LogFormatterSciNotation: it blanks a 5x10^n tick even with labelOnlyBase=False.
            target.set_major_formatter(FuncFormatter(decade_label))
        minor_ticks(target, MinorKind.RATIO if log_base == 2.0 else MinorKind.TOKEN)
    else:
        target.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
        target.grid(False, which="minor")  # pyright: ignore[reportUnknownMemberType]
    ax.grid(axis=axis, which="major", color=RULE, linewidth=0.7, zorder=0)  # pyright: ignore[reportUnknownMemberType]
    ax.set_axisbelow(True)


#: Draw order for a paired point mark: fill, then the connector between two conditions (through an
#: unfilled mark's white centre, unbroken), then the outline on top.
FILL_Z: float = 3.0
CONNECTOR_Z: float = 4.0
MARK_Z: float = 5.0

#: Fraction of a mark's area the not-delivered cross covers; the model shape still reads first.
CROSS_SCALE: float = 0.45

#: What the cross means, wherever a figure draws one: a 1x placeholder, not a measurement.
NOT_DELIVERED_LABEL: str = "No Verified Answer (Drawn at 1x)"

#: An entry not yet attempted (vs. one that ran and failed). Drawn only with ``--mark-pending``;
#: enters no summary.
PENDING_MARKER: str = "$?$"
PENDING_LABEL: str = "Pending"
#: A glyph fills less of its box than a shape does; scales the "?" up to a shape's size.
PENDING_SCALE: float = 2.2
#: The artist gid every pending mark carries, so a caller can find what was drawn as pending.
PENDING_GID: str = "pending"


def edge_width(size: float, widest: float) -> float:
    """A mark's edge line width in points: ``widest`` at full size, thinner on a small one."""
    return min(widest, 0.2 * math.sqrt(size))


#: Size factors so matplotlib's filled markers match a circle's ink at equal ``s`` (a triangle
#: otherwise covers about half a square's area). Shapes not listed keep ``s``.
MARKER_AREA_SCALE: dict[str, float] = {"s": 0.8, "D": 0.85, "d": 1.0, "<": 1.3, ">": 1.3, "^": 1.3, "v": 1.3, "*": 1.5}


def point_mark(
    ax: Axes,
    x: float,
    y: float,
    color: str,
    marker: str,
    filled: bool,
    size: float = 110.0,
    delivered: bool = True,
    clip: bool = True,
) -> None:
    """One point of a two-condition pair: a white disc under the mark (masks the grid and any
    connector), then the mark itself.

    ``delivered=False`` overlays a small cross on the model's own shape, always hollow: a cross in
    the series' own colour over a filled mark of that colour is invisible. ``clip=False`` lets a
    mark on the axis limit print whole instead of halved.
    """
    filled = filled and delivered
    size *= MARKER_AREA_SCALE.get(marker, 1.0) if isinstance(marker, str) else 1.0
    ax.scatter(  # pyright: ignore[reportUnknownMemberType]
        x, y, s=size, marker=marker, color="white", edgecolor="none", zorder=FILL_Z, clip_on=clip
    )
    ax.scatter(  # pyright: ignore[reportUnknownMemberType]
        x,
        y,
        s=size,
        marker=marker,
        color=color if filled else "none",
        edgecolor=color,
        linewidth=edge_width(size, 1.8),
        zorder=MARK_Z,
        clip_on=clip,
    )
    if not delivered:
        ax.scatter(  # pyright: ignore[reportUnknownMemberType]
            x,
            y,
            s=size * CROSS_SCALE,
            marker="x",
            color=color,
            linewidth=edge_width(size, 1.6),
            zorder=MARK_Z + 1.0,
            clip_on=clip,
        )


def pending_mark(
    ax: Axes, x: float, y: float, color: str, size: float = 110.0, transform: Transform | None = None
) -> None:
    """A :data:`PENDING_MARKER` in ``color`` at ``(x, y)``; data coordinates unless ``transform``
    says otherwise."""
    extra = {} if transform is None else {"transform": transform}
    ax.scatter(  # pyright: ignore[reportUnknownMemberType]
        x,
        y,
        s=size * PENDING_SCALE,
        marker=PENDING_MARKER,
        color=color,
        linewidth=0.0,
        zorder=MARK_Z,
        gid=PENDING_GID,
        **extra,
    )


def pending_legend_mark(markersize: float) -> Line2D:
    """The legend entry for :func:`pending_mark`."""
    return Line2D(
        [], [], marker=PENDING_MARKER, linestyle="none", color=MUTED, markersize=markersize, label=PENDING_LABEL
    )


def row_axis(ax: Axes, labels: Sequence[str]) -> None:
    """A categorical y axis with one named row per series, top row first, no grid.

    Limits keep half a row of air at each end so the topmost and bottommost marks are not clipped.
    """
    ax.set_yticks(range(len(labels)))  # pyright: ignore[reportUnknownMemberType]
    ax.set_yticklabels(list(labels), fontsize=LABEL_PT, color=INK)  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    despine(ax)


def right_label(ax: Axes, row: int, text: str, color: str = MUTED) -> None:
    """A short annotation just outside the right edge of ``row`` (e.g. an n), outside the frame so
    it is never mistaken for data on the value axis."""
    ax.annotate(  # pyright: ignore[reportUnknownMemberType]
        text,
        xy=(1.006, 1.0 - (row + 0.5) / max(len(ax.get_yticks()), 1)),
        xycoords="axes fraction",
        fontsize=ANNOTATION_PT,
        color=color,
        ha="left",
        va="center",
        annotation_clip=False,
    )


#: Per-suffix metadata that keeps a written figure a pure function of its content; otherwise PDF
#: and SVG stamp the time of the write.
UNDATED: dict[str, dict[str, None]] = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}

#: SVG element ids are hashed with a random salt unless one is fixed.
SVG_HASH_SALT: str = "hpcagent-bench"


#: The white margin a placed figure keeps left and right of its ink, in inches.
PLACED_SIDE_PAD_IN: float = 0.02


def fill_width(fig: Figure, pad_in: float = PLACED_SIDE_PAD_IN, rounds: int = 3) -> None:
    """Stretch the axes horizontally so their ink spans the canvas less ``pad_in`` per side.

    Figure-level artists (legends, figure texts) stay put; a figure with a figure-level label
    beside its axes must not call this.
    """
    width = float(fig.get_size_inches()[0])
    axes = [ax for ax in fig.axes if ax.get_visible()]
    for _ in range(rounds):
        renderer = fig.canvas.get_renderer()
        ink = Bbox.union([ax.get_tightbbox(renderer) for ax in axes])
        left, right = ink.x0 / fig.dpi, ink.x1 / fig.dpi
        if abs(left - pad_in) < 0.005 and abs(width - pad_in - right) < 0.005:
            return
        scale = (width - 2.0 * pad_in) / (right - left)
        for ax in axes:
            box = ax.get_position()
            x0 = (pad_in + (box.x0 * width - left) * scale) / width
            x1 = (pad_in + (box.x1 * width - left) * scale) / width
            ax.set_position((x0, box.y0, x1 - x0, box.height))


def placed_box(fig: Figure, width_in: float) -> Bbox:
    """The saved box of a paper figure placed at ``width_in``: the ink's own height, the canvas's width.

    A tight crop sets the width from the ink, so the page rescales the figure and its type by the
    ratio of the two. Refuses a canvas of another width, ink outside it, and text outside the print
    range (:func:`print_type_violations`).
    """
    width = float(fig.get_size_inches()[0])
    if abs(width - width_in) > PLACED_WIDTH_RTOL * width_in:
        raise ValueError(f"figure is {width:.3f}in wide, placed at {width_in:.3f}in")
    ink = fig.get_tightbbox(fig.canvas.get_renderer())
    slack = PLACED_WIDTH_RTOL * width_in
    if ink.x0 < -slack or ink.x1 > width + slack:
        raise ValueError(f"ink spans {ink.x0:.3f}..{ink.x1:.3f}in, outside the {width:.3f}in canvas")
    wrong = print_type_violations(fig)
    if wrong:
        raise ValueError(f"text outside {PRINT_MIN_PT:g}-{PRINT_LABEL_PT:g}pt: {wrong[:6]}")
    return Bbox.from_extents(0.0, ink.y0, width, ink.y1)


def save(
    fig: Figure,
    stem: pathlib.Path,
    formats: Sequence[str] = ("pdf", "png"),
    fixed: bool = False,
    dpi: float = SAVE_DPI,
    width_in: float = 0.0,
    print_size: bool = False,
) -> pathlib.Path:
    """Write ``fig`` under ``stem`` once per suffix in ``formats``, and close it. Returns ``stem``.

    A paper takes the PDF; a web page takes the PNG (at ``dpi``, 200 by default) or the SVG.
    ``fixed`` keeps the canvas at its figsize instead of cropping to the ink, which is what keeps
    two paired figures the same size: a tight box is sized by each figure's own legend. Closing
    matters in a loop -- matplotlib keeps every open figure alive, and a sweep that renders one per
    directory otherwise ends up holding all of them. Every file is written :data:`UNDATED`, so a
    rerun is byte-identical. ``width_in`` marks a paper figure placed at that width
    (:func:`placed_box`): it is saved exactly that wide, and refused if its type is off the print scale.
    ``print_size`` applies the same type check to a figure that sizes its own canvas (``fixed``).
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    settle_clear_labels(fig)
    wrong = print_type_violations(fig) if print_size else []
    if wrong:
        raise ValueError(f"text outside {PRINT_MIN_PT:g}-{PRINT_LABEL_PT:g}pt: {wrong[:6]}")
    box = placed_box(fig, width_in) if width_in > 0.0 else fig.bbox_inches if fixed else "tight"
    with plt.rc_context({"svg.hashsalt": SVG_HASH_SALT}):
        for suffix in formats:
            fig.savefig(  # pyright: ignore[reportUnknownMemberType]
                stem.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches=box, metadata=UNDATED.get(suffix)
            )
    plt.close(fig)
    return stem


# ---------------------------------------------------------------------------------------------
# Measured layout. Every figure module sizes its chrome from what its text MEASURES on the laid-out
# figure, never from a fixed fraction: a fixed band is right for one width and one label length,
# and on any other it either wastes the page or prints the chrome over the data.
# ---------------------------------------------------------------------------------------------


def left_protrusion_in(fig: Figure, ax: Axes) -> float:
    """How far ``ax``'s Y tick labels and axis label reach left of its frame, in inches."""
    renderer = fig.canvas.get_renderer()
    return max(0.0, ax.get_window_extent(renderer).x0 - ax.yaxis.get_tightbbox(renderer).x0) / fig.dpi


def below_protrusion_in(fig: Figure, ax: Axes) -> float:
    """How far everything ``ax`` draws (X tick labels, axis label, annotations under the frame)
    reaches below its frame, in inches."""
    renderer = fig.canvas.get_renderer()
    return max(0.0, ax.get_window_extent(renderer).y0 - ax.get_tightbbox(renderer).y0) / fig.dpi


def above_protrusion_in(fig: Figure, ax: Axes) -> float:
    """How far everything ``ax`` draws reaches above its frame, in inches: all three title slots
    (a ``loc="left"`` title is not ``ax.title``), and annotations placed over the frame, such as a
    panel name or a summary column's statistic."""
    renderer = fig.canvas.get_renderer()
    return max(0.0, ax.get_tightbbox(renderer).y1 - ax.get_window_extent(renderer).y1) / fig.dpi


def crowded_ticks(ax: Axes, renderer: RendererBase, gap: float) -> bool:
    """Whether two X tick labels printed on one line of ``ax`` come within ``gap`` pixels. The labels
    on one line are the ticks sharing a pad: a staggered axis prints two lines."""
    lines: dict[float, list[Bbox]] = {}
    for tick in ax.xaxis.get_major_ticks():
        if tick.label1.get_text():
            lines.setdefault(tick.get_pad(), []).append(tick.label1.get_window_extent(renderer))
    return any(
        left.x1 + gap > right.x0
        for boxes in lines.values()
        for left, right in itertools.pairwise(sorted(boxes, key=lambda box: box.x0))
    )


def shrink_crowded_ticks(fig: Figure, axes: Sequence[Axes], start_pt: float, floor_pt: float) -> float:
    """Step the X tick labels of ``axes`` down from ``start_pt``, all to one size, until no printed
    line is crowded (:func:`crowded_ticks`); returns the size they got. Labels still crowded at
    ``floor_pt`` stay at the floor and are reported: below it they are no longer readable, and an
    overprint is a layout the caller has to change.

    Two labels closer than a third of their type size read as one word ("OMPTriton").
    """
    renderer = fig.canvas.get_renderer()
    size = start_pt
    while True:
        for ax in axes:
            for label in ax.get_xticklabels():
                label.set_fontsize(size)
        gap = size / 3.0 * fig.dpi / 72.0
        if not any(crowded_ticks(ax, renderer, gap) for ax in axes):
            return size
        if size <= floor_pt:
            LOG.warning("style: tick labels still overlap at the %.2fpt floor; the axis needs more width", floor_pt)
            return size
        size = max(floor_pt, size - 0.25)


def mark_boxes(ax: Axes) -> list[Bbox]:
    """The display box of every mark and interval drawn on ``ax``: one per scatter point, sized by
    its own marker area, one per interval segment, and one per marker of a plotted line."""
    dpi = ax.figure.dpi
    boxes: list[Bbox] = []
    for collection in ax.collections:
        if isinstance(collection, PathCollection):
            boxes += scatter_boxes(collection, dpi)
        elif isinstance(collection, LineCollection):
            boxes += segment_boxes(collection)
    for line in ax.lines:
        boxes += line_marker_boxes(line, dpi)
    return boxes


def centred_box(x: float, y: float, radius: float) -> Bbox:
    """The square display box of half-side ``radius`` pixels around ``(x, y)``."""
    return Bbox.from_extents(x - radius, y - radius, x + radius, y + radius)


def scatter_boxes(collection: PathCollection, dpi: float) -> list[Bbox]:
    """One box per scatter point, sized by its own marker area; none for a sizeless collection."""
    if not len(collection.get_sizes()):
        return []
    centres = collection.get_offset_transform().transform(collection.get_offsets())
    sizes = np.broadcast_to(collection.get_sizes(), (len(centres),))
    return [centred_box(x, y, math.sqrt(size) / 2.0 * dpi / 72.0) for (x, y), size in zip(centres, sizes, strict=True)]


def segment_boxes(collection: LineCollection) -> list[Bbox]:
    """One box per non-empty interval segment."""
    transform = collection.get_transform()
    return [
        Bbox.from_extents(*ends.min(axis=0), *ends.max(axis=0))
        for ends in (transform.transform(segment) for segment in collection.get_segments() if len(segment))
    ]


def line_marker_boxes(line: Line2D, dpi: float) -> list[Bbox]:
    """One box per marker of a visible plotted line; none for a line drawn without markers."""
    if line.get_marker() in (None, "", "None", " ") or not line.get_visible():
        return []
    radius = line.get_markersize() / 2.0 * dpi / 72.0
    # A line's data may arrive as Python lists of mixed int/float (an errorbar's caps), which
    # stack into an OBJECT array that a log transform cannot take.
    xs, ys = (np.asarray(values, dtype=float) for values in line.get_data())
    return [centred_box(x, y, radius) for x, y in line.get_transform().transform(np.column_stack((xs, ys)))]


#: Tags an annotation :func:`settle_clear_labels` places clear of the marks, of the other tagged
#: labels, and inside its axes' frame.
CLEAR_GID: str = "hpcagent-clear-label"

#: Clearance a settled label keeps from the frame and from every mark, in points.
CLEAR_PAD_PT: float = 1.5


def settle_clear_labels(fig: Figure) -> None:
    """Place every :data:`CLEAR_GID` annotation clear of the marks, of the labels settled before it
    and of its axes' frame, moving it as little as possible.

    A label starts where its caller put it (above a mark, beside a bracket). It is shifted inside the
    frame sideways, raised until nothing under its span touches it, and held under the frame top; if
    holding it there lands it on a mark again, it slides sideways at that height, and failing that
    takes the highest clear gap below (:func:`clear_place`). Runs at
    save time, when every limit and margin is final: a label's offset is in points, so a place clear
    before the last ``subplots_adjust`` need not be clear after it.
    """
    tagged = clear_labels(fig)
    if not tagged:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = CLEAR_PAD_PT * fig.dpi / 72.0
    labels = {id(label) for ax, label in tagged}
    obstacles: dict[int, list[Bbox]] = {}
    for ax, label in tagged:
        frame = ax.get_window_extent(renderer)
        if id(ax) not in obstacles:
            obstacles[id(ax)] = panel_obstacles(ax, renderer, labels)
        taken = obstacles[id(ax)]
        drawn = label.get_window_extent(renderer)
        box = clear_place(drawn, taken, frame, pad)
        taken.append(box)
        x, y = label.xyann
        label.xyann = (x + (box.x0 - drawn.x0) * 72.0 / fig.dpi, y + (box.y0 - drawn.y0) * 72.0 / fig.dpi)


def clear_labels(fig: Figure) -> list[tuple[Axes, Annotation]]:
    """``(axes, annotation)`` of every non-empty :data:`CLEAR_GID` annotation on ``fig``."""
    return [
        (ax, text) for ax in fig.axes for text in ax.texts
        if isinstance(text, Annotation) and text.get_gid() == CLEAR_GID and text.get_text()
    ]  # fmt: skip


def panel_obstacles(ax: Axes, renderer: RendererBase, labels: set[int]) -> list[Bbox]:
    """What a settled label on ``ax`` must keep clear of: its marks and its texts other than the
    ``labels`` being settled, inside its frame."""
    frame = ax.get_window_extent(renderer)
    others = [text.get_window_extent(renderer) for text in ax.texts if id(text) not in labels and text.get_text()]
    return [box for box in mark_boxes(ax) + others if box.overlaps(frame)]


def clear_place(box: Bbox, taken: Sequence[Bbox], frame: Bbox, pad: float) -> Bbox:
    """Where :func:`settle_clear_labels` puts a label drawn at ``box``: inside ``frame``, clear of
    every box in ``taken`` by ``pad`` pixels, as near its drawn place as that allows."""

    box = box.translated(max(0.0, frame.x0 + pad - box.x0) - max(0.0, box.x1 + pad - frame.x1), 0.0)
    for attempt in range(len(taken) + 1):
        blocking = near_boxes(box, taken, pad)
        if not blocking:
            break
        box = box.translated(0.0, max(other.y1 for other in blocking) + pad - box.y0)
    held = box.translated(0.0, min(0.0, frame.y1 - pad - box.y1))
    if not near_boxes(held, taken, pad):
        return held
    for candidate in fallback_places(held, taken, frame, pad):
        if not near_boxes(candidate, taken, pad):
            return candidate
    LOG.warning("style: no clear place for the label %r inside its panel", box)
    return held


def fallback_places(held: Bbox, taken: Sequence[Bbox], frame: Bbox, pad: float) -> Iterator[Bbox]:
    """Where a label held under the frame top at ``held`` may go when it lands on a mark there,
    nearest first: sideways at that height, up to two label widths each way inside the frame; then
    down into each gap under a taken box, highest first, above the frame bottom."""
    step = held.width / 4.0
    for shift in (sign * step * k for k in range(1, 9) for sign in (1.0, -1.0)):
        candidate = held.translated(shift, 0.0)
        if candidate.x0 >= frame.x0 + pad and candidate.x1 <= frame.x1 - pad:
            yield candidate
    for other in sorted(taken, key=lambda other: -other.y0):
        candidate = held.translated(0.0, other.y0 - pad - held.y1)
        if candidate.y0 >= frame.y0 + pad:
            yield candidate


def near_boxes(candidate: Bbox, taken: Sequence[Bbox], pad: float) -> list[Bbox]:
    """The boxes of ``taken`` within ``pad`` pixels of ``candidate``."""
    return [
        other for other in taken
        if other.x0 < candidate.x1 + pad and other.x1 > candidate.x0 - pad
        and other.y0 < candidate.y1 + pad and other.y1 > candidate.y0 - pad
    ]  # fmt: skip


def right_protrusion_in(fig: Figure, ax: Axes) -> float:
    """How far everything ``ax`` draws reaches right of its frame, in inches: a label centred on the
    last column (a summary column's statistic) overruns it by half its own width, and a fixed right
    pad cut it off at the canvas edge."""
    renderer = fig.canvas.get_renderer()
    return max(0.0, ax.get_tightbbox(renderer).x1 - ax.get_window_extent(renderer).x1) / fig.dpi
