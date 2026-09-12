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

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter

# matplotlib's drawing calls end in an untyped ``**kwargs``, so every call below suppresses the
# unknown-member report that fact produces; the arguments themselves are checked.

#: Ink, in decreasing emphasis. Text NEVER takes a series colour: a coloured mark beside a label
#: carries the identity, and a coloured label just makes the text harder to read.
INK: str = "#1c1c1e"
MUTED: str = "#6b6b70"
RULE: str = "#d6d6da"
#: For text that labels the CHART rather than the data -- quadrant captions and the like. Lighter
#: than MUTED so it sits behind the marks in the reading order instead of competing with them.
FAINT: str = "#a8a8ae"
#: The zero/parity reference. Darker than the grid because it is a statement, not a guide.
REFERENCE: str = "#3a3a3e"

#: TEXT CASE, for every label a figure shows: Title Case. Capitalise each word except articles,
#: coordinating conjunctions and prepositions ("and", "or", "of", "per", "over", "to", "vs"), and
#: always capitalise the first and last word. Identifiers keep their own spelling -- ``numba``,
#: ``oss120b`` and ``lang-c`` are names, not words, and title-casing them makes them wrong.
#:
#: Written down rather than enforced by a function because these strings carry mathtext
#: (``$\log_2$``) and identifiers, and a naive title-caser mangles both.

#: Minor-tick positions within a decade on a base-10 log axis, given majors at 1, 2 and 5. Each
#: major interval is subdivided: 1-2, 2-5 and 5-10 all get lines, at roughly even log spacing.
LOG10_MINOR_SUBS: tuple[float, ...] = (1.25, 1.5, 1.75, 2.5, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0)

#: Type scale, in points, sized for PRINT rather than for a screen.
#:
#: A figure in a paper is reproduced at roughly half the width it was authored at, so 8pt ticks
#: land near 4pt on the page -- below what most venues will accept and below what a reader can
#: comfortably read. These are set so the SMALLEST text survives that reduction: 13pt ticks reach
#: the page around 6.5pt, and the axis labels and title scale with them.
TITLE_PT: float = 20.0
SUBTITLE_PT: float = 13.0
LABEL_PT: float = 16.0
TICK_PT: float = 14.0
ANNOTATION_PT: float = 13.0


def apply() -> None:
    """Set the process-wide rcParams. Idempotent; call it before creating a figure."""
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": RULE,
            "axes.labelcolor": MUTED,
            "axes.labelsize": LABEL_PT,
            "axes.titlesize": LABEL_PT + 1,
            "axes.titlecolor": INK,
            "axes.grid": False,  # each plot opts in on ONE axis; a full grid is noise
            "axes.axisbelow": True,  # data over guides, never the reverse
            "grid.color": RULE,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
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


def title(fig: Figure, text: str, subtitle: str = "") -> float:
    """Centred title; returns the top of the plot area for ``tight_layout(rect=...)``.

    ``subtitle`` is accepted and IGNORED. It used to render a how-to-read sentence under the
    title, and that sentence belongs in the caption a paper already gives every figure -- printed
    above the axes it competes with the title and eats a chunk of the panel. Kept in the signature
    so the callers do not all have to change at once, and so a caller passing one is not silently
    dropping information it thought was displayed.
    """
    height = float(fig.get_size_inches()[1])
    # Work in inches, then convert: a fraction of a 4-inch figure is a different gap than the same
    # fraction of a 12-inch one, which is what made the fixed offsets collide.
    top = 1.0 - (0.34 / height)
    fig.text(0.5, top, text, fontsize=TITLE_PT, color=INK, ha="center", va="top")  # pyright: ignore[reportUnknownMemberType]
    return max(0.5, top - 0.30 / height)


def legend_below(fig: Figure, handles: Sequence[Artist], ncol: int = 0, y: float = 0.0) -> None:
    """One legend, under the whole figure, centred. Never inside the axes.

    An in-axes legend has to be placed, and every placement is a bet that one corner stays empty.
    That bet loses whenever the data changes: the score-vs-cost figure put its key in the corner
    two points later occupied, and the per-kernel figures have data in every row by construction.
    Below the figure there is no corner to lose, and the legend is in the same place in every
    figure, which is the point of a shared style.
    """
    fig.legend(  # pyright: ignore[reportUnknownMemberType]
        handles=handles,
        loc="lower center" if y != 0.0 else "upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol if ncol != 0 else min(len(handles), 5),
        frameon=False,
        fontsize=LABEL_PT,
        markerscale=1.4,
        handletextpad=0.5,
        columnspacing=1.6,
        borderaxespad=0.0,
    )


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


def value_axis(
    ax: Axes, axis: Literal["x", "y"] = "y", minor: bool = True, log_base: float = 10.0, major: bool = True
) -> None:
    """Ticks and grid for the axis carrying the MEASURED quantity.

    Majors get a labelled line, minors an unlabelled fainter one: reading a value off a chart is
    interpolation between ticks, and with only a handful of majors the reader is interpolating
    across a gap wide enough to be a guess. Applied to the value axis only -- the other axis
    carries kernel names, where a grid line per category is noise.

    ``log_base`` is passed rather than sniffed off the axis: matplotlib keeps it on the scale
    object under a private name, and a wrong guess puts the minor lines at the wrong ratios --
    which looks like a grid and reads as a lie.
    """
    target: Axis = ax.yaxis if axis == "y" else ax.xaxis
    scale: str = ax.get_yscale() if axis == "y" else ax.get_xscale()
    if scale == "log":
        # A log axis needs log-spaced minors, and AutoMinorLocator refuses one outright ("does not
        # work on logarithmic scales").
        #
        # Majors go at 1, 2 and 5 per decade rather than at the decades alone. A panel spanning
        # less than two decades -- which most token axes here do -- gets exactly ONE labelled tick
        # under the default locator, and a single number on an axis is nothing to read a value
        # against. ``log_base`` other than 10 keeps the plain decade majors, because a caller on a
        # base-2 axis has usually pinned landmarks of its own that these would overwrite.
        if log_base == 10.0 and major:
            target.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=20))
            # NOT LogFormatterSciNotation: even with labelOnlyBase=False it returns the empty
            # string for a 5x10^n tick, so the axis got a line and a gap where its label should be
            # -- which looks like a stray rule rather than a tick. This labels every major it is
            # given, which is the only contract a caller pinning majors can rely on.
            target.set_major_formatter(FuncFormatter(decade_label))
        if minor:
            # Every intermediate multiple in the decade (2x, 3x .. 9x), which is what a reader
            # interpolates a log value against. subs="all" sounds like more and gives FEWER: on a
            # base-10 axis it returns just the decades again, so the panel came out with almost no
            # minor grid at all while its linear neighbour was dense with it.
            # Subdivide each MAJOR interval, not each integer. The majors sit at 1, 2 and 5, and
            # the whole-number subs 3,4,6..9 leave the 1-to-2 interval with no minor line at all
            # while 2-to-5 and 5-to-10 get several -- a grid that is finer in some bands than
            # others, which is worse than a coarse one because the spacing stops meaning anything.
            # These fill all three intervals at roughly even spacing in LOG space, which is the
            # space the reader is interpolating in.
            subs: tuple[float, ...] = LOG10_MINOR_SUBS
            if log_base != 10.0:
                subs = tuple(float(n) for n in range(2, int(log_base))) or (2.0,)
            target.set_minor_locator(LogLocator(base=log_base, subs=subs, numticks=100))
            target.set_minor_formatter(NullFormatter())
    else:
        # Eight majors and ONE minor between them. Twelve majors subdivided four ways put a line
        # every 2 percent of the axis: at that spacing the grid stops being a reference and becomes
        # a texture, and the marks sit on top of it rather than against it.
        target.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
        if minor:
            target.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis=axis, which="major", color=RULE, linewidth=0.7, zorder=0)  # pyright: ignore[reportUnknownMemberType]
    if minor:
        # Dashed, so a minor line is never mistaken for a major one at a glance -- weight alone
        # does not separate them once a figure is reduced for print.
        ax.grid(  # pyright: ignore[reportUnknownMemberType]
            axis=axis, which="minor", color=RULE, linewidth=0.45, linestyle=(0, (2, 3)), alpha=0.9, zorder=0
        )
    ax.set_axisbelow(True)


#: The three layers a paired point mark occupies. A connector between two conditions is drawn
#: BETWEEN a mark's fill and its outline: an unfilled mark reads as a box with a white centre, and
#: the reader follows the dashed connector THROUGH that centre to the other condition. Putting the
#: connector under the fill breaks it at both ends, which is where the eye is trying to start.
#: The outline stays on top of the line, so the mark keeps its shape where the two cross.
FILL_Z: float = 3.0
CONNECTOR_Z: float = 4.0
MARK_Z: float = 5.0


def point_mark(ax: Axes, x: float, y: float, color: str, marker: str, filled: bool, size: float = 110.0) -> None:
    """One point of a two-condition pair, drawn as a white disc plus the mark itself.

    The white disc is drawn under a FILLED mark too. It masks the grid and every connector that
    does not end here, so the only line a reader sees inside a mark is that mark's own -- with a
    transparent centre, three models' connectors crossing one point read as a mesh.
    """
    ax.scatter(  # pyright: ignore[reportUnknownMemberType]
        x, y, s=size, marker=marker, color="white", edgecolor="none", zorder=FILL_Z
    )
    ax.scatter(  # pyright: ignore[reportUnknownMemberType]
        x,
        y,
        s=size,
        marker=marker,
        color=color if filled else "none",
        edgecolor=color,
        linewidth=1.8,
        zorder=MARK_Z,
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


def save(fig: Figure, stem: pathlib.Path) -> pathlib.Path:
    """Write ``fig`` as both PDF and SVG under ``stem``, and close it. Returns ``stem``.

    Two formats because the two consumers differ: a paper takes the PDF, and a web or slide build
    takes the SVG. Closing matters in a loop -- matplotlib keeps every open figure alive, and a
    sweep that renders one per directory otherwise ends up holding all of them.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg"):
        fig.savefig(stem.with_suffix(suffix))  # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)
    return stem
