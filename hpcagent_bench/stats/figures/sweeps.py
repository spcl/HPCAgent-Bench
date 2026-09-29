# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Slide and paper figures for measured sweeps: speedup against size, tile-size bars, a wavefront picture.

A sweep figure is a row of panels on one shared log-log speedup axis (:func:`speedup_panels`), or a row of
bar panels grouped by kernel (:func:`tile_bars`). Neither carries a title: the slide or the caption names
it. The hardware the numbers were measured on goes in a footer line (:func:`footer`). Every figure is
saved as PNG and PDF by :func:`save_figure`. Colour comes from :mod:`hpcagent_bench.stats.palette`,
type sizes and neutral inks from :mod:`hpcagent_bench.stats.style`.
"""

import pathlib
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FuncFormatter

from hpcagent_bench.stats import palette, style

__all__ = [
    "SLIDE_SIZE_IN",
    "BarPanel",
    "CurvePanel",
    "Panel",
    "Series",
    "VLine",
    "curve_panels",
    "footer",
    "save_figure",
    "score_histogram",
    "speedup_panels",
    "tile_bars",
    "wavefront_grid",
]

#: A 16:9 slide-friendly strip: full slide width, about a third of its height.
SLIDE_SIZE_IN: tuple[float, float] = (13.33, 5.4)
#: Type sizes of a slide figure (pt): ticks, axis labels, in-panel names, footer.
SLIDE_TICK_PT: float = 12.0
SLIDE_LABEL_PT: float = 13.0
SLIDE_FOOTER_PT: float = 10.0
LEGEND_PT: float = 12.0
#: Characters per footer line at :data:`SLIDE_FOOTER_PT` on a slide-width canvas.
FOOTER_CHARS: int = 150


@dataclass(frozen=True, slots=True)
class Series:
    """One curve: ``xs`` and ``ys`` in data units. ``line=False`` draws markers only (a reference implementation)."""

    label: str
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    color: str
    marker: str = "o"
    linestyle: str = "-"
    line: bool = True


@dataclass(frozen=True, slots=True)
class VLine:
    """A vertical marker at ``x`` (a crossover size), named in the legend."""

    x: float
    label: str
    color: str
    linestyle: str = ":"


@dataclass(frozen=True, slots=True)
class Panel:
    """One panel of a speedup row: its name (drawn inside, top left), curves and crossover markers."""

    name: str
    series: tuple[Series, ...]
    vlines: tuple[VLine, ...] = ()


@dataclass(frozen=True, slots=True)
class BarPanel:
    """One panel of a bar figure: ``groups`` maps a group name (a kernel) to one value per bar label; None = not run."""

    name: str
    groups: tuple[tuple[str, tuple[float | None, ...]], ...]


@dataclass(frozen=True, slots=True)
class CurvePanel:
    """One panel of :func:`curve_panels`: its own y label, curves and vertical markers; ``ylog`` for a log y axis."""

    name: str
    ylabel: str
    series: tuple[Series, ...]
    vlines: tuple[VLine, ...] = ()
    ylog: bool = False
    ylim: tuple[float, float] | None = None


def _ratio_formatter(value: float, position: int = 0) -> str:
    """A speedup tick: ``0.1x``, ``1x``, ``10x``, ``100x``."""
    del position
    return f"{value:g}x"


def _size_formatter(value: float, position: int = 0) -> str:
    """A size tick: ``1e4`` .. ``1e8`` as ``10^4``."""
    del position
    exponent = round(np.log10(value)) if value > 0 else 0
    return f"$10^{{{exponent}}}$" if abs(value - 10.0**exponent) < 1e-9 * value else ""


def _layout(fig: Figure, *, foot: bool) -> None:
    """Fixed chrome: axes above ``bottom``, then the shared x label, the legend and the footer beneath them."""
    fig.subplots_adjust(left=0.065, right=0.995, top=0.98, bottom=0.33 if foot else 0.27, wspace=0.07)


def footer(fig: Figure, text: str) -> None:
    """One muted line at the bottom of the figure: the hardware, image and commit behind the numbers."""
    wrapped = textwrap.fill(text, width=FOOTER_CHARS)
    fig.text(
        0.5, 0.005, wrapped, ha="center", va="bottom", fontsize=SLIDE_FOOTER_PT, color=style.MUTED, linespacing=1.3
    )  # pyright: ignore[reportUnknownMemberType]


def _legend_handles(panels: Sequence[Panel]) -> tuple[list[Artist], list[str]]:
    """One handle per distinct series label and per distinct crossover label, in first-seen order; an empty label is unlisted."""
    seen: dict[str, Artist] = {}
    for panel in panels:
        for series in panel.series:
            if series.label and series.label not in seen:
                seen[series.label] = Line2D(
                    [],
                    [],
                    color=series.color,
                    marker=series.marker,
                    linestyle=series.linestyle if series.line else "none",
                    linewidth=2.4,
                    markersize=7,
                    markerfacecolor=series.color if series.line else "none",
                    markeredgewidth=1.8,
                )
        for vline in panel.vlines:
            if vline.label and vline.label not in seen:
                seen[vline.label] = Line2D([], [], color=vline.color, linestyle=vline.linestyle, linewidth=2.0)
    return list(seen.values()), list(seen)


def speedup_panels(
    panels: Sequence[Panel],
    *,
    xlabel: str,
    ylabel: str,
    foot: str = "",
    size_in: tuple[float, float] = SLIDE_SIZE_IN,
    x_ticks: Sequence[float] = (1e4, 1e6, 1e8),
    reference: float = 1.0,
    x_formatter: Callable[[float, int], str] | None = None,
) -> Figure:
    """A row of log-log panels sharing one speedup axis: x = problem size, y = speedup over the baseline.

    The baseline sits at ``reference`` (a dashed rule). Panels carry their name inside, no title; the
    y label is drawn once, on the first panel; one legend sits below the row."""
    style.apply()
    with plt.rc_context({"xtick.labelsize": SLIDE_TICK_PT, "ytick.labelsize": SLIDE_TICK_PT}):
        fig, axes = plt.subplots(1, len(panels), figsize=size_in, sharey=True, squeeze=False)
    row = axes[0]
    for ax, panel in zip(row, panels, strict=True):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(reference, color=style.REFERENCE, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        for vline in panel.vlines:
            ax.axvline(vline.x, color=vline.color, linestyle=vline.linestyle, linewidth=2.0, zorder=2)
        for series in panel.series:
            ax.plot(
                series.xs,
                series.ys,
                color=series.color,
                marker=series.marker,
                linestyle=series.linestyle if series.line else "none",
                linewidth=2.2,
                markersize=6.5,
                markerfacecolor=series.color if series.line else "none",
                markeredgewidth=1.6,
                zorder=4,
            )
        ax.set_xticks(list(x_ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(x_formatter or _size_formatter))
        style.minor_ticks(ax.xaxis, style.MinorKind.TOKEN)
        style.minor_ticks(ax.yaxis, style.MinorKind.TOKEN)
        ax.yaxis.set_major_formatter(FuncFormatter(_ratio_formatter))
        ax.grid(axis="both", which="major", color=style.RULE, linewidth=0.7, zorder=0)
        ax.tick_params(labelsize=SLIDE_TICK_PT)
        ax.text(
            0.04,
            0.95,
            panel.name,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=SLIDE_LABEL_PT + 1,
            color=style.INK,
        )  # pyright: ignore[reportUnknownMemberType]
    row[0].set_ylabel(ylabel, fontsize=SLIDE_LABEL_PT)
    handles, labels = _legend_handles(panels)
    _layout(fig, foot=bool(foot))
    fig.text(0.5, 0.245 if foot else 0.19, xlabel, ha="center", va="center", fontsize=SLIDE_LABEL_PT, color=style.INK)  # pyright: ignore[reportUnknownMemberType]
    fig.legend(  # pyright: ignore[reportUnknownMemberType]
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085 if foot else 0.01),
        ncol=min(len(handles), 3),
        frameon=False,
        fontsize=LEGEND_PT,
        columnspacing=1.6,
        handlelength=2.2,
        borderaxespad=0.0,
    )
    if foot:
        footer(fig, foot)
    return fig


def curve_panels(
    panels: Sequence[CurvePanel],
    *,
    xlabel: str,
    x_ticks: Sequence[float],
    foot: str = "",
    size_in: tuple[float, float] = SLIDE_SIZE_IN,
    x_formatter: Callable[[float, int], str] | None = None,
    legend_ncol: int = 4,
) -> Figure:
    """A row of panels with a shared log x axis and a y axis of their own (a rate, a spread, a probability).

    Each series is drawn as connected marks; a series with an empty label stays out of the key. No title."""
    style.apply()
    fig, axes = plt.subplots(1, len(panels), figsize=size_in, squeeze=False)
    for ax, panel in zip(axes[0], panels, strict=True):
        ax.set_xscale("log")
        if panel.ylog:
            ax.set_yscale("log")
        for vline in panel.vlines:
            ax.axvline(vline.x, color=vline.color, linestyle=vline.linestyle, linewidth=2.0, zorder=2)
        for series in panel.series:
            ax.plot(
                series.xs,
                series.ys,
                color=series.color,
                marker=series.marker,
                linestyle=series.linestyle if series.line else "none",
                linewidth=2.2,
                markersize=6.5,
                markerfacecolor=series.color if series.line else "none",
                markeredgewidth=1.6,
                zorder=4,
            )
        ax.set_xticks(list(x_ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(x_formatter or _size_formatter))
        if panel.ylim is not None:
            ax.set_ylim(*panel.ylim)
        ax.grid(axis="both", which="major", color=style.RULE, linewidth=0.7, zorder=0)
        ax.tick_params(labelsize=SLIDE_TICK_PT)
        ax.set_ylabel(panel.ylabel, fontsize=SLIDE_LABEL_PT)
        ax.text(
            0.04,
            0.95,
            panel.name,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=SLIDE_LABEL_PT + 1,
            color=style.INK,
        )  # pyright: ignore[reportUnknownMemberType]
    handles, labels = _legend_handles([Panel(p.name, p.series, p.vlines) for p in panels])
    fig.subplots_adjust(left=0.07, right=0.995, top=0.98, bottom=0.33 if foot else 0.27, wspace=0.28)
    fig.text(0.5, 0.245 if foot else 0.19, xlabel, ha="center", va="center", fontsize=SLIDE_LABEL_PT, color=style.INK)  # pyright: ignore[reportUnknownMemberType]
    fig.legend(  # pyright: ignore[reportUnknownMemberType]
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085 if foot else 0.01),
        ncol=min(len(handles), legend_ncol),
        frameon=False,
        fontsize=LEGEND_PT,
        columnspacing=1.6,
        handlelength=2.2,
        borderaxespad=0.0,
    )
    if foot:
        footer(fig, foot)
    return fig


def score_histogram(
    log2_scores: Sequence[float],
    *,
    xlabel: str,
    note: str,
    foot: str = "",
    size_in: tuple[float, float] = (9.5, 5.2),
    bins: int = 61,
    limit: float = 1.5,
) -> Figure:
    """Histogram of task scores on a log2 axis (0 = a score of exactly 1x) with a marker at 1.0 and a note on it.

    ``log2_scores`` are log2 of the task scores; the bar at 0 holds every task credited exactly 1. Scores outside
    +-``limit`` are clipped into the outer bins. The y axis is the share of tasks."""
    style.apply()
    fig, ax = plt.subplots(1, 1, figsize=size_in)
    values = np.clip(np.asarray(log2_scores, dtype=float), -limit, limit)
    edges = np.linspace(-limit, limit, bins + 1)
    weights = np.full(values.shape, 100.0 / max(values.size, 1))
    ax.hist(values, bins=edges, weights=weights, color=palette.hues()[0], edgecolor="white", linewidth=0.4, zorder=3)  # pyright: ignore[reportUnknownMemberType]
    ax.axvline(0.0, color=style.INK, linestyle=(0, (4, 3)), linewidth=2.0, zorder=4)
    ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{2.0**t:.2g}x" for t in ticks])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p=0: f"{v:g}%"))
    ax.grid(axis="y", which="major", color=style.RULE, linewidth=0.7, zorder=0)
    ax.tick_params(labelsize=SLIDE_TICK_PT)
    ax.set_xlabel(xlabel, fontsize=SLIDE_LABEL_PT)
    ax.set_ylabel("Share of tasks", fontsize=SLIDE_LABEL_PT)
    ax.text(
        0.98,
        0.95,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=SLIDE_LABEL_PT,
        color=style.INK,
        linespacing=1.5,
    )  # pyright: ignore[reportUnknownMemberType]
    fig.subplots_adjust(left=0.1, right=0.985, top=0.97, bottom=0.22 if foot else 0.13)
    if foot:
        footer(fig, foot)
    return fig


def tile_bars(
    panels: Sequence[BarPanel],
    bar_labels: Sequence[str],
    *,
    ylabel: str,
    legend_title: str,
    foot: str = "",
    size_in: tuple[float, float] = SLIDE_SIZE_IN,
    colormap: str = "viridis",
) -> Figure:
    """Grouped bars, one panel per target: a group per kernel, a bar per tile size, colour = tile size.

    The y axis is a log speedup over the untiled doall of the same target (a dashed rule at 1x)."""
    style.apply()
    fig, axes = plt.subplots(1, len(panels), figsize=size_in, sharey=True, squeeze=False)
    count = len(bar_labels)
    colours = [
        palette.colormap_slot(colormap, round(255 * (0.05 + 0.85 * i / max(count - 1, 1)))) for i in range(count)
    ]
    width = 0.8 / count
    every = [v for panel in panels for _, values in panel.groups for v in values if v is not None]
    top, low = max(every, default=2.0), min(min(every, default=1.0), 1.0)
    for ax, panel in zip(axes[0], panels, strict=True):
        ax.set_yscale("log")
        for g, (_, values) in enumerate(panel.groups):
            for b, value in enumerate(values):
                if value is None:
                    continue
                x = g - 0.4 + width * (b + 0.5)
                ax.bar(x, value - 1.0, width * 0.92, bottom=1.0, color=colours[b], zorder=3)
        ax.axhline(1.0, color=style.REFERENCE, linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
        ax.set_xticks(range(len(panel.groups)))
        ax.set_xticklabels([name for name, _ in panel.groups], fontsize=SLIDE_TICK_PT)
        ticks = [t for t in (0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100) if low / 1.05 <= t <= top * 1.3]
        ax.set_yticks(ticks)
        ax.set_ylim(low / 1.1, top * 1.9)
        ax.yaxis.set_major_formatter(FuncFormatter(_ratio_formatter))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, p=0: ""))
        ax.grid(axis="y", which="major", color=style.RULE, linewidth=0.7, zorder=0)
        ax.tick_params(labelsize=SLIDE_TICK_PT)
        ax.text(
            0.04,
            0.95,
            panel.name,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=SLIDE_LABEL_PT + 1,
            color=style.INK,
        )  # pyright: ignore[reportUnknownMemberType]
    axes[0][0].set_ylabel(ylabel, fontsize=SLIDE_LABEL_PT)
    handles = [Rectangle((0, 0), 1, 1, color=colour) for colour in colours]
    _layout(fig, foot=bool(foot))
    fig.legend(  # pyright: ignore[reportUnknownMemberType]
        handles=handles,
        labels=list(bar_labels),
        title=legend_title,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085 if foot else 0.01),
        ncol=count,
        frameon=False,
        fontsize=LEGEND_PT,
        title_fontsize=LEGEND_PT,
        columnspacing=1.4,
        handlelength=1.2,
        borderaxespad=0.0,
    )
    if foot:
        footer(fig, foot)
    return fig


def wavefront_grid(
    n: int,
    code: Sequence[str],
    *,
    foot: str = "",
    size_in: tuple[float, float] = (13.33, 5.4),
    colormap: str = "viridis",
) -> Figure:
    """The slide picture of a wavefront: an ``n x n`` grid, every cell coloured by its anti-diagonal ``i + j``,
    an arrow along the diagonal order, and the CPF loop nest (sequential over diagonals, doall inside) beside it."""
    style.apply()
    fig, (grid, text) = plt.subplots(1, 2, figsize=size_in, gridspec_kw={"width_ratios": [1.0, 1.15]})
    diagonals = 2 * n - 1
    norm = Normalize(0, diagonals - 1)
    for i in range(n):
        for j in range(n):
            colour = palette.colormap_slot(colormap, round(255 * (0.05 + 0.9 * norm(i + j))))
            grid.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=colour, edgecolor="white", linewidth=1.5))
            grid.text(j + 0.5, n - 0.5 - i, str(i + j), ha="center", va="center", fontsize=SLIDE_TICK_PT, color="white")
    mid = n // 2
    for i in range(n):  # outline one anti-diagonal: the cells that run together
        j = mid - i + (n // 2) - 1
        if 0 <= j < n:
            grid.add_patch(
                Rectangle((j, n - 1 - i), 1, 1, facecolor="none", edgecolor=style.INK, linewidth=3.0, zorder=5)
            )
    grid.add_patch(
        FancyArrowPatch(
            (n + 0.3, n + 0.5),
            (n + 1.9, n - 1.1),
            arrowstyle="-|>",
            mutation_scale=26,
            color=style.INK,
            linewidth=2.6,
            clip_on=False,
        )
    )
    grid.text(n + 1.05, n + 0.35, "order", ha="left", va="bottom", fontsize=SLIDE_LABEL_PT, color=style.MUTED)
    grid.set_xlim(-1.4, n + 2.4)
    grid.set_ylim(-1.4, n + 1.4)
    grid.set_aspect("equal")
    grid.axis("off")
    grid.text(n / 2, -1.25, "column j", ha="center", va="top", fontsize=SLIDE_LABEL_PT, color=style.MUTED)
    grid.text(-1.25, n / 2, "row i", ha="right", va="center", fontsize=SLIDE_LABEL_PT, color=style.MUTED, rotation=90)
    text.axis("off")
    text.text(
        0.0,
        0.5,
        "\n".join(code),
        ha="left",
        va="center",
        family="monospace",
        fontsize=SLIDE_LABEL_PT + 1,
        color=style.INK,
        linespacing=1.55,
    )
    fig.tight_layout(rect=(0.0, 0.06 if foot else 0.0, 1.0, 1.0))
    if foot:
        footer(fig, foot)
    return fig


def save_figure(fig: Figure, stem: pathlib.Path, png_dpi: float = 200.0) -> tuple[pathlib.Path, pathlib.Path]:
    """Write ``<stem>.png`` (deck) and ``<stem>.pdf`` (paper); returns both paths. Closes the figure."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    fig.savefig(png, dpi=png_dpi, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
    fig.savefig(pdf, bbox_inches="tight", metadata={"CreationDate": None})  # pyright: ignore[reportUnknownMemberType]
    plt.close(fig)
    return png, pdf
