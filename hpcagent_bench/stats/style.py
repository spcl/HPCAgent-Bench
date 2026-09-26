# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The one visual style every figure in this repo is drawn in.

Figures from one project that do not look like one project make a reader work out, per figure,
what is ink and what is data. This fixes the parts that are never data -- type sizes, tick and
spine weight, grid colour, the neutral inks -- so a plot only has to decide what it is actually
showing. Colour is NOT here: it belongs to the entity, and :mod:`hpcagent_bench.stats.palette`
owns it.

Neutrals carry a slight cool bias rather than being a pure grey, so they sit under the palette's
blues without looking like a different rendering of the page.
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

#: Ink, in decreasing emphasis. EVERY PIECE OF TEXT IS :data:`INK` (never a series colour, never
#: grey: grey text goes illegible at column width); MUTED is for non-text marks (a neutral legend
#: swatch, a tick dash, an interval that would compete with the data).
INK: str = "#1c1c1e"
MUTED: str = "#6b6b70"
RULE: str = "#d6d6da"
#: For text that labels the CHART rather than the data -- quadrant captions and the like. Lighter
#: than MUTED so it sits behind the marks in the reading order instead of competing with them.
FAINT: str = "#a8a8ae"
#: The zero/parity reference. Darker than the grid because it is a statement, not a guide.
REFERENCE: str = "#3a3a3e"
#: The MINOR grid's colour and line weight (:func:`minor_ticks`): lighter and thinner than the major
#: grid (:data:`RULE` at 0.7pt), so it reads as a finer ruling of the same reference and never as a
#: second one.
MINOR_RULE: str = "#e8e8ea"
MINOR_GRID_WIDTH: float = 0.25
#: A minor tick MARK against a major one, as fractions of the major's length and line width.
MINOR_TICK_LENGTH: float = 0.55
MINOR_TICK_WIDTH: float = 0.6


@dataclasses.dataclass(frozen=True, slots=True)
class StatInk:
    """Colours of STATISTICS and states, never of entities (palette.py owns those)."""

    median: str = "#3b6fd4"  # median bar; blue so the geomean tick reads against it
    geomean: str = "#d4772a"  # geomean tick; orange, far from the median blue


STAT_INK = StatInk()

#: TEXT CASE, for every label a figure shows: Title Case, except articles, coordinating conjunctions
#: and prepositions ("and", "or", "of", "per", "over", "to", "vs") inside the label. Identifiers
#: (``numba``, ``oss120b``, ``lang-c``) and mathtext keep their own spelling, so no function enforces it.

#: Type scale, in points, sized for PRINT: a paper reproduces a figure at about half width, so 13pt
#: ticks reach the page around 6.5pt.
TITLE_PT: float = 20.0
#: Floor for the shrink in :func:`title`: below this the title is smaller than the tick labels.
MIN_TITLE_PT: float = 8.0
SUBTITLE_PT: float = 13.0
LABEL_PT: float = 16.0
TICK_PT: float = 14.0
ANNOTATION_PT: float = 13.0

#: Type sizes of a figure drawn at the width it is placed at (scale 1.0), in points: what the page
#: prints. Every figure module's paper mode starts from these, so two figures of one page agree.
#: Ticks, point labels and category names print at PRINT_TICK_PT, axis labels and panel names at
#: PRINT_LABEL_PT, legends at PRINT_LEGEND_PT (a two-column key must fit a 2.5in wrap figure).
PRINT_TICK_PT: float = 7.0
PRINT_LABEL_PT: float = 8.0
PRINT_LEGEND_PT: float = 6.0
#: The floor any print-size fitting (crowded category names, a legend squeezed into its band) may
#: shrink text to. Below it a figure changes its layout instead: text that shrinks per figure is
#: exactly what makes two figures on one page print at different sizes.
PRINT_MIN_PT: float = 5.5

#: The dpi every figure is written at (:func:`save`'s default). FreeType hints tighter at a low dpi,
#: so ``get_window_extent`` at matplotlib's default 100 dpi UNDERSTATES text width: a caller that
#: measures text against a figure sets the figure to this dpi first.
SAVE_DPI: float = 200.0

#: The full text width of a double-column A4 paper, in inches: the width of a paper figure (the
#: per-kernel and efficacy figures).
DOUBLE_COLUMN_WIDTH: float = 7.0

#: Per-paper page budgets, in inches, so a figure drops in at scale 1.0 instead of being shrunk by
#: ``\includegraphics`` -- shrinking a figure shrinks its type below what this module sets.
#: ``ICLR_TEXT_WIDTH_IN``: ``agentbench-paper/iclr2027_conference.sty`` line 49,
#: ``\textwidth 5.5 true in`` (single column, so this is the whole row's budget).
#: ``ACM_COLUMN_WIDTH_IN``/``ACM_TEXT_WIDTH_IN``: the mpr paper's ``acmart.cls`` (``sigconf``,
#: two columns) documented defaults -- confirm against that class file before a real figure there
#: is sized to it.
ICLR_TEXT_WIDTH_IN: float = 5.5
ACM_COLUMN_WIDTH_IN: float = 3.33
ACM_TEXT_WIDTH_IN: float = 7.0

#: A figure wrapped beside the text (``wrapfigure`` at ``0.45\textwidth``), and the height of its
#: plot body including the axis chrome; its legend adds its own height below. Every wrap figure uses
#: both, so two of them on one page have the same box.
ICLR_WRAP_WIDTH_IN: float = 0.45 * ICLR_TEXT_WIDTH_IN
PRINT_BODY_HEIGHT_IN: float = 1.9

#: How far a saved paper figure's width may differ from the width it is placed at. Beyond it the
#: ``\includegraphics`` width rescales the type set here.
PLACED_WIDTH_RTOL: float = 0.01


@dataclasses.dataclass(frozen=True, slots=True)
class TypeScale:
    """One figure's type sizes (points) and the mark and line weights that go with them.

    Type size and figure size are one decision: a figure drawn to be shrunk to half width needs
    twice the type of one drawn at its placed width. :data:`AUTHOR_SCALE` is the first,
    :data:`PRINT_SCALE` the second; a figure module picks one and never mixes them.
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
            "axes.grid": False,  # each plot opts in on ONE axis; a full grid is noise
            "axes.axisbelow": True,  # data over guides, never the reverse
            "grid.color": RULE,
            "grid.linewidth": 0.4,  # thinner than 0.6 (user, 2026-09-25): the grid is a guide, not a mark
            "xtick.color": MUTED,  # the tick DASH stays a guide
            "ytick.color": MUTED,
            "xtick.labelcolor": INK,  # its NUMBER is text, and text is ink
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


#: Where :func:`title` puts a figure title: its top edge this far below the canvas top, and the plot
#: area this much further down. A figure that reserves room for a title reserves TITLE_BAND_IN.
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
    """One legend, under the whole figure, centred, wrapped to the figure width. Never inside the
    axes. Returns the legend's height in inches, which the caller adds to its bottom margin.

    An in-axes legend has to be placed, and every placement is a bet that one corner stays empty.
    That bet loses whenever the data changes: the score-vs-cost figure put its key in the corner
    two points later occupied, and the per-kernel figures have data in every row by construction.
    Below the figure there is no corner to lose, and the legend is in the same place in every
    figure, which is the point of a shared style.

    The requested column count is a ceiling, not a promise: a row of long labels that does not fit
    the canvas is wrapped onto more rows until it does. A legend wider than the figure survives a
    ``bbox_inches="tight"`` save by widening the saved page, which is how the llr-focus40 PDF came
    out 13.7 inches wide and was then shrunk to the column by the includegraphics width, halving its
    type while the paper template's own width was the number the figure was built for.

    ``fontsize`` overrides :data:`LABEL_PT` for a figure whose height cannot afford it -- several
    SQUARE panels joined into one short row still budget the same fixed pixels for the legend as a
    full-height figure, and LABEL_PT alone would not fit. ``markerscale`` is the swatch's own size
    against the handle's: the 1.4 default enlarges a swatch so it reads beside authoring-scale type,
    and at 6.5pt type the same 1.4 makes the swatch taller than the row it sits in.

    ``columnspacing`` and ``handlelength`` (in font sizes) default to a full-width key; a wrap
    figure passes :data:`COMPACT_KEY` to fit two columns in 2.5in.

    ``span``, the plot body's (left, right) in figure fractions, centres the legend on the body
    instead of the canvas and makes the BODY's width the limit: a key never runs out past the panels
    under the Y labels, it drops a column first.
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
            # matplotlib's 0.4 pads the key box and read as a blank band under the ticks.
            borderpad=0.1,
        )
        box = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        if columns <= 1 or box.width <= limit:
            # FILL the box: the fewest columns that give this many rows (12 entries: 4, not 5).
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

    NOT scientific notation. A token count is a quantity a reader compares and quotes, and
    "$5\\times10^{5}$" makes them do the arithmetic before they can do either -- while
    matplotlib's own scientific formatter additionally declines to label anything but 1x and 2x of
    a decade, leaving a rule with a blank where its label belongs.

    Falls back to the plain number below a thousand, so an axis in units of seconds or ratios is
    not given a suffix it does not want.
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
    """A ratio tick at full precision: ``0.25 -> "0.25x"``, ``1.0 -> "1x"``, ``4.0 -> "4x"``.

    A ratio below 1 prints as a decimal, so a half-octave tick (0.707) reads correctly.
    """
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
    """A measured ratio printed beside its mark, to one decimal: ``6.3x``, ``32.5x``, ``0.9x``
    -- one decimal is what a reader quotes; below 0.1x it would print a real
    slowdown as ``0.0x``, so those keep one significant figure (``0.04x``). The tick spelling keeps
    full precision, which beside a mark reads ``6.34919x``."""
    if not math.isfinite(value) or value <= 0.0:
        return ""
    return f"{value:.1f}x" if value >= 0.1 else f"{value:.1g}x"


#: What a value axis holds, as far as its minor ticks care (:func:`minor_ticks`). ``ratio``: a log2
#: axis in ratio units, majors at powers of two. ``log2``: a LINEAR axis holding ``log2(ratio)``
#: (the efficacy speedup axes), majors at whole exponents. ``token``: a log10 token axis.
#: ``count``: a linear count from 0 to N (the efficacy success row).
class MinorKind(enum.Enum):
    RATIO = "ratio"
    LOG2 = "log2"
    TOKEN = "token"
    COUNT = "count"


#: Where the minors of a ONE-octave ratio axis sit inside each octave, as multiples of its lower
#: major: the quarters, i.e. the integers 5x, 6x, 7x between 4x and 8x.
OCTAVE_SUBS: tuple[float, ...] = (1.25, 1.5, 1.75)

#: How close, in octaves, two exponents are to count as one: majors come back from a locator as
#: floats, and a minor landing a rounding error off a major is still that major.
OCTAVE_TOLERANCE: float = 1e-6


def ratio_minor_exponents(majors: Sequence[float], low: float, high: float) -> list[float]:
    """The minor ticks of a ratio axis inside ``[low, high]``; majors, limits and result all in log2
    units (exponents).

    The spacing is read off ``majors``, never assumed: majors more than an octave apart get a minor
    at every octave between them (1x, 4x, 16x -> 2x, 8x); majors one octave apart get
    :data:`OCTAVE_SUBS` inside each octave (1x, 2x -> 1.25x, 1.5x, 1.75x). Fewer than two majors
    leave no spacing to read, and majors under an octave apart no power of two between them: both
    get no minors. A minor never lands on a major."""
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
    """Minor exponents over ``octaves`` for majors ``step`` octaves apart: every octave when the
    majors skip some, :data:`OCTAVE_SUBS` inside each when they are one apart, none when closer."""
    if step > 1.0 + OCTAVE_TOLERANCE:
        return [float(octave) for octave in octaves]
    if step > 1.0 - OCTAVE_TOLERANCE:
        return [octave + math.log2(sub) for octave in octaves for sub in OCTAVE_SUBS]
    return []


def token_minor_values(majors: Sequence[float], low: float, high: float) -> list[float]:
    """The minor ticks of a log10 token axis inside ``[low, high]``: every whole multiple 1..9 of a
    power of ten that is not a major (majors 100K, 200K, 500K, 1M -> 300K, 400K, 600K ... 900K)."""
    if low <= 0.0 or high <= 0.0:
        return []
    decades = range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
    return [
        value for value in (multiple * 10.0**decade for decade in decades for multiple in range(1, 10))
        if low <= value <= high and not any(math.isclose(value, major, rel_tol=1e-9) for major in majors)
    ]  # fmt: skip


#: The parts a count axis' major step is split into, first whole-number step wins: quarters, then
#: fifths, thirds, halves (0/20/40 -> every 5, 0/5/10 -> every 1, 0/9 -> every 3).
COUNT_DIVISIONS: tuple[int, ...] = (4, 5, 3, 2)


def count_minor_values(majors: Sequence[float], low: float, high: float) -> list[float]:
    """The minor ticks of a linear count axis inside ``[low, high]``: the major step split into the
    first of :data:`COUNT_DIVISIONS` that gives a WHOLE step. A count is a whole number of tasks, so
    a line at 4.5 of them marks nothing; a step no division splits evenly (a prime N) gets none."""
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
    """The minors of a ``kind`` axis inside ``[low, high]``, majors and limits in the axis' own
    units: :func:`ratio_minor_exponents` on the exponents of a ``ratio`` axis (mapped back to
    ratios) or directly on a ``log2`` one, :func:`token_minor_values` on a ``token`` one,
    :func:`count_minor_values` on a ``count`` one."""
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
    """:func:`minor_positions` as a matplotlib locator, derived at every draw from the axis' CURRENT
    majors and view: a caller that pins its majors or moves its limits after the axis was styled
    still gets minors that fit them."""

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
    """Unlabelled minor ticks and a light minor grid on the VALUE axis ``axis``, by the one rule every
    figure shares.

    ``ratio``/``log2``: by the spacing of the majors actually set (:func:`ratio_minor_exponents`).
    ``token``: every whole multiple of a power of ten that is not a major
    (:func:`token_minor_values`). ``count``: whole-number steps only (:func:`count_minor_values`;
    majors 0/20/40 -> every 5, 0/5/10 -> every 1).

    The marks point the way the majors do, shorter and thinner (:data:`MINOR_TICK_LENGTH`,
    :data:`MINOR_TICK_WIDTH` of the major's), and carry NO label: a number at every minor doubles
    the axis' text, and matplotlib's own log minor formatter prints a scientific-notation 3x10^n
    beside plain majors. The grid is :data:`MINOR_RULE` at ``width``, under everything. A category
    axis never takes this: a line between two names measures nothing.
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
    """Ticks and a major grid for the axis carrying the MEASURED quantity, plus, on a log axis, the
    shared minor ruling (:func:`minor_ticks`).

    A LOG axis gets unlabelled minor ticks and a light minor grid: a ratio (log2) axis by its majors'
    octave spacing, a token (log10)
    axis at every whole multiple of a power of ten. The majors stay the only labelled reference; the minors are shorter,
    lighter and unlabelled, so a reader places a mark between two labels without the panel turning
    into a texture. A LINEAR axis keeps its majors alone: whether it holds ``log2`` units or a
    0..N count is the caller's to say, to :func:`minor_ticks`. Applied to the value axis only -- the other axis
    carries names, where a guide line per category measures nothing.

    ``log_base`` is passed rather than sniffed off the axis: matplotlib keeps it on the scale
    object under a private name, and a wrong guess puts the lines at the wrong ratios -- which
    looks like a grid and reads as a lie.
    """
    target: Axis = ax.yaxis if axis == "y" else ax.xaxis
    scale: str = ax.get_yscale() if axis == "y" else ax.get_xscale()
    if scale == "log":
        # Majors go at 1, 2 and 5 per decade rather than at the decades alone. A panel spanning
        # less than two decades -- which most token axes here do -- gets exactly ONE labelled tick
        # under the default locator, and a single number on an axis is nothing to read a value
        # against. ``log_base`` other than 10 keeps the locator matplotlib gives a base-2 axis,
        # because a caller on one has usually pinned landmarks of its own that these would
        # overwrite.
        if major:
            # A decade gets 1, 2, 5, more than one labelled tick even under two decades. A base-2
            # (ratio) axis gets 1 ONLY: a 1.5 sub would label 1.5x, 3x, 6x, 0.75x, ... -- ticks a
            # reader cannot place on a log2 grid by eye (:func:`ratio_tick_label`). The caller widens its own limits (SC15 speedup/ratio axes always
            # do) so a narrow window still gets more than the one tick this alone would leave it.
            subs = (1.0, 2.0, 5.0) if log_base == 10.0 else (1.0,)
            target.set_major_locator(LogLocator(base=log_base, subs=subs, numticks=20))
        if log_base == 10.0 and major:
            # NOT LogFormatterSciNotation: it returns "" for a 5x10^n tick even with
            # labelOnlyBase=False. This labels every major it is given.
            target.set_major_formatter(FuncFormatter(decade_label))
        minor_ticks(target, MinorKind.RATIO if log_base == 2.0 else MinorKind.TOKEN)
    else:
        target.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
        target.grid(False, which="minor")  # pyright: ignore[reportUnknownMemberType]
    ax.grid(axis=axis, which="major", color=RULE, linewidth=0.7, zorder=0)  # pyright: ignore[reportUnknownMemberType]
    ax.set_axisbelow(True)


#: The three layers a paired point mark occupies: fill, then the connector between two conditions
#: (so it runs through an unfilled mark's white centre, unbroken), then the outline on top.
FILL_Z: float = 3.0
CONNECTOR_Z: float = 4.0
MARK_Z: float = 5.0

#: How much of a mark's area the NOT-DELIVERED cross covers: the model shape still reads first.
CROSS_SCALE: float = 0.45

#: What the cross means, wherever a figure draws one: a 1x placeholder, not a measurement. How a
#: summary treats it differs by figure, so the legend does not say.
NOT_DELIVERED_LABEL: str = "No Verified Answer (Drawn at 1x)"

#: The PENDING mark: an entry that has not been attempted yet, as opposed to one that ran and failed
#: (the cross). Drawn only when a figure is asked to (``--mark-pending``); it enters no summary.
PENDING_MARKER: str = "$?$"
PENDING_LABEL: str = "Pending"
#: A glyph fills less of its box than a shape does; this makes a "?" read at a shape's size.
PENDING_SCALE: float = 2.2
#: The artist id every pending mark carries, so a caller can find what was drawn as pending.
PENDING_GID: str = "pending"


def edge_width(size: float, widest: float) -> float:
    """A mark's line width in points: ``widest`` on a full-size mark, thinner on a small one, where a
    fixed edge fills a hollow mark and swallows its cross."""
    return min(widest, 0.2 * math.sqrt(size))


#: Size factors that give matplotlib's filled markers about the ink of a circle at one ``s``: at equal
#: ``s`` a triangle covers roughly half a square, so a blind-submission row read smaller than a
#: CPF row beside it. Shapes not listed (the circle, crosses, stars given as paths) keep ``s``.
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
    """One point of a two-condition pair, drawn as a white disc plus the mark itself.

    The white disc is drawn under a FILLED mark too. It masks the grid and every connector that
    does not end here, so the only line a reader sees inside a mark is that mark's own -- with a
    transparent centre, three models' connectors crossing one point read as a mesh.

    ``delivered`` False overlays a small cross on the model's own shape: the point is the 1x
    placeholder an episode that never verified an answer leaves behind, not a measured 1x. The
    SHAPE still names the model and the colour still names the intervention -- replacing the shape
    outright would cost the figure the one channel that survives greyscale. The placeholder is
    always drawn HOLLOW: a cross in the series' own colour laid over a FILLED mark of that colour
    is invisible, which drew 34 of 40 PPCG placeholders as if they were measured 1x results.

    ``clip`` False lets a mark on the axis limit print whole across the frame instead of halved.
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
    """A :data:`PENDING_MARKER` in the series' own colour at ``(x, y)``, in data coordinates unless
    ``transform`` says otherwise (a row with no value axis centres it on the axes)."""
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
    """A categorical y axis with one named row per series, top row first.

    The rows are NAMES, so the axis carries no grid and no minor ticks: a guide line per category
    measures nothing. Limits are set with half a row of air at each end so the topmost and
    bottommost marks are not clipped by the frame.
    """
    ax.set_yticks(range(len(labels)))  # pyright: ignore[reportUnknownMemberType]
    ax.set_yticklabels(list(labels), fontsize=LABEL_PT, color=INK)  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    despine(ax)


def right_label(ax: Axes, row: int, text: str, color: str = MUTED) -> None:
    """A short annotation just outside the right edge of ``row`` -- the n a reader needs at the mark.

    Outside the frame rather than inside it: an n printed among the points is one more thing on the
    value axis, and a reader who is estimating a position has to decide it is not data.
    """
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


#: Per suffix, the metadata that keeps a written figure a function of the figure alone: PDF and SVG
#: otherwise stamp the time of the write, so two renders of one table differ in every file.
UNDATED: dict[str, dict[str, None]] = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}

#: SVG element ids are hashed with a random salt unless one is fixed.
SVG_HASH_SALT: str = "hpcagent-bench"


#: The white margin a placed figure keeps left and right of its ink, in inches.
PLACED_SIDE_PAD_IN: float = 0.02


def fill_width(fig: Figure, pad_in: float = PLACED_SIDE_PAD_IN, rounds: int = 3) -> None:
    """Stretch the axes horizontally so their ink (tick labels and axis labels included) spans the
    canvas less ``pad_in`` per side. Figure-level artists (legends, figure texts) stay put, so a
    figure that places a figure-level label beside its axes must not call this."""
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
