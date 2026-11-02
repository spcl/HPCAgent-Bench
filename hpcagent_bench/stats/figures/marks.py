# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The options a speedup figure shares (:class:`Mode`, :class:`Mark`) and :func:`draw_estimate`, the one
primitive that draws a point estimate with its interval under either mark. Text spellings live in
:mod:`hpcagent_agent.coded` (the ``Coded`` base of every closed category)."""

import enum
import math
from typing import NamedTuple

from hpcagent_agent.coded import Coded
from matplotlib.axes import Axes
from matplotlib.typing import LineStyleType

from hpcagent_bench.stats import style
from hpcagent_bench.stats.palette import Look, darken

__all__ = [
    "BAR_INTERVAL_DARKEN",
    "Estimate",
    "IntervalStyle",
    "Mark",
    "Mode",
    "draw_estimate",
    "draw_interval",
]


class Mode(Coded):
    """What a speedup figure measures a setup against."""

    #: Each setup against the experiment baseline; the control drawn beside its treated setup.
    ABSOLUTE = enum.auto()
    #: Each treated setup as its change over its own control (the control is 1x).
    RELATIVE = enum.auto()


class Mark(Coded):
    """How a point estimate is drawn; its interval is drawn the same way under every mark."""

    #: A marker at the estimate.
    SCATTER = enum.auto()
    #: A bar from the axis reference to the estimate.
    BAR = enum.auto()


#: How many lightness steps (:func:`~hpcagent_bench.stats.palette.darken`) darker than its bar a bar's interval is.
BAR_INTERVAL_DARKEN: int = 2


class Estimate(NamedTuple):
    """One drawn point estimate and its interval, in axis units (``low``/``high`` NaN: no interval)."""

    x: float
    value: float
    low: float
    high: float


class IntervalStyle(NamedTuple):
    """How an interval line is drawn: its weight, dash, opacity and the size of the arrowhead at a cut end."""

    width: float
    linestyle: LineStyleType
    cap_pt: float
    alpha: float = 0.75


def draw_interval(
    ax: Axes, x: float, low: float, high: float, bounds: tuple[float, float], colour: str, line: IntervalStyle
) -> tuple[float, float]:
    """An interval as a vertical line cut to ``bounds``, with an arrowhead in ``colour`` at each cut end; returns
    the ends drawn."""
    bottom, top = max(low, bounds[0]), min(high, bounds[1])
    ax.vlines(x, bottom, top, color=colour, linewidth=line.width, alpha=line.alpha, linestyles=line.linestyle,
              zorder=style.CONNECTOR_Z)  # fmt: skip
    for cut, end, marker in ((low < bounds[0], bottom, "v"), (high > bounds[1], top, "^")):
        if cut:
            ax.plot([x], [end], marker=marker, markersize=line.cap_pt * 1.6, color=colour,
                    linestyle="none", clip_on=False, zorder=style.CONNECTOR_Z)  # fmt: skip
    return bottom, top


def draw_estimate(
    ax: Axes,
    estimate: Estimate,
    look: Look,
    filled: bool,
    mark: Mark,
    line: IntervalStyle,
    *,
    bounds: tuple[float, float] = (-math.inf, math.inf),
    size: float = 74.0,
    width: float = 0.24,
    base: float = 0.0,
) -> tuple[float, float]:
    """``estimate`` as ``mark`` in ``look``, with its interval (when both ends are finite) cut to ``bounds``.

    :data:`Mark.SCATTER` draws ``look.marker`` at the value (hollow unless ``filled``, area ``size``);
    :data:`Mark.BAR` a bar ``width`` wide from ``base`` to the value, its interval :data:`BAR_INTERVAL_DARKEN`
    steps darker. Returns the interval ends drawn, NaN when
    none is."""
    drawn = (math.nan, math.nan)
    if math.isfinite(estimate.low) and math.isfinite(estimate.high):
        # Over a filled bar of its own colour the interval would vanish: a bar's is drawn two steps darker.
        ink = darken(look.colour, BAR_INTERVAL_DARKEN) if mark is Mark.BAR else look.colour
        drawn = draw_interval(ax, estimate.x, estimate.low, estimate.high, bounds, ink, line)
    if mark is Mark.BAR:
        if not math.isfinite(estimate.value):
            return drawn
        ax.bar(
            estimate.x, estimate.value - base, width=width, bottom=base, color=look.colour if filled else "white",
            edgecolor=look.colour, linewidth=line.width, zorder=style.FILL_Z,
        )  # fmt: skip
    else:
        style.point_mark(ax, estimate.x, estimate.value, look.colour, look.marker, filled, size=size)
    return drawn
