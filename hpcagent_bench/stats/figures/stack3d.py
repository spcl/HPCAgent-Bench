# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The 3D stacked-bar figure: per-arm geomean speed-up extruded over the kernel axis.

A 3D bar chart does not scale to this repo's full kernel corpus (forty-plus bars per arm read as a
skyline nobody can compare by eye, which is exactly the failure Hoefler and Belli's Rule 7
("visual encoding") warns against), so this figure takes a CURATED kernel subset
(:func:`representative_kernels`: the geomean's own worst, median and best kernels by default) rather
than drawing every kernel. It exists beside, not instead of, the per-kernel + geomean figure
(:mod:`hpcagent_bench.stats.figures.per_kernel`), which stays the source for "every kernel, read
precisely"; this one is for "the shape of the corpus at a glance" -- a small multiple in 3D.

ONE BAR PER (ARM, KERNEL): X is the kernel, Y is the arm (one row per (model, packet) pair,
labelled with both), Z is ``log2(geomean speed-up)`` over that arm's episodes of that kernel --
the same log2 ratio axis every other speed-up figure in this repo draws
(``docs/plotting.md`` rule 2), so ``0`` is the no-change plane and the bars read as a skyline
above and below it. COLOUR IS THE PACKET (:func:`hpcagent_bench.stats.palette.color`,
:func:`~hpcagent_bench.stats.palette.control_color` for no packet) -- the default convention
(``docs/plotting.md`` rule 4), since the arm axis already spells out which model a row is and does
not need a second channel for it.

Bars are drawn with :meth:`mpl_toolkits.mplot3d.Axes3D.bar3d`, one call per bar rather than per
arm, so a missing (arm, kernel) cell (an arm that never ran a kernel in the subset) draws nothing
instead of a zero-height bar that would misread as a measured 1x.
"""

import dataclasses
import pathlib
from collections.abc import Sequence

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

from hpcagent_bench import experiment_tags
from hpcagent_bench.stats import palette, summary
from hpcagent_bench.stats import style as plotstyle
from hpcagent_bench.stats.figures import per_kernel

#: How many kernels :func:`representative_kernels` keeps by default: the geomean's own worst,
#: median and best over the pooled frame, plus enough of the middle to read as a distribution
#: rather than three isolated points.
DEFAULT_KERNEL_COUNT: int = 12

#: Bar footprint as a fraction of one (kernel, arm) grid cell; below 1.0 so a bar's four vertical
#: edges stay visible against its neighbours instead of fusing into a wall.
BAR_FRACTION: float = 0.6

#: Figure size in inches (width, height): wide enough for a double-column page, tall enough that
#: the 3D axes' own perspective does not crush the Z (speed-up) extent to a sliver.
FIGSIZE_IN: tuple[float, float] = (7.2, 4.6)

#: The camera: looking slightly down (elevation) and rotated so the nearest kernel column does not
#: occlude the row behind it.
ELEVATION_DEG: float = 22.0
AZIMUTH_DEG: float = -60.0


@dataclasses.dataclass(frozen=True, slots=True)
class Bar:
    """One drawn bar: which (kernel, arm) cell, its height and its colour."""

    kernel: str
    arm: str
    packet: str
    log2_speedup: float
    color: str


def arm_kernel_geomean(
    frame: pd.DataFrame, arms: Sequence[str], kernels: Sequence[str]
) -> dict[tuple[str, str], float]:
    """``{(arm, kernel): geomean speed-up}`` over each arm's own episodes of each kernel.

    Reads through :func:`hpcagent_bench.stats.population.graded_episode_rows`
    (:func:`hpcagent_bench.stats.figures.per_kernel.speedup_cells` on the arm's own slice of
    ``frame``), the same population every other speed-up figure in this repo reduces, so a bar's
    height is never a second definition of a kernel's speed-up.
    """
    out: dict[tuple[str, str], float] = {}
    for arm in arms:
        cells = {cell.kernel: cell for cell in per_kernel.speedup_cells(frame[frame["arm"] == arm], served=False)}
        for kernel in kernels:
            cell = cells.get(kernel)
            if cell is None or not cell.episodes:
                continue
            out[(arm, kernel)] = summary.geomean(cell.episodes)
    return out


def representative_kernels(frame: pd.DataFrame, count: int = DEFAULT_KERNEL_COUNT) -> list[str]:
    """``count`` kernels spanning the pooled geomean's range: evenly spaced by RANK, so the subset
    shows the corpus's spread (its worst, its best, and points in between) rather than an
    alphabetical or a first-N slice that says nothing about the distribution."""
    cells = per_kernel.speedup_cells(frame, served=False)
    ranked = sorted(cells, key=lambda cell: cell.median())
    if len(ranked) <= count:
        return [cell.kernel for cell in ranked]
    positions = [round(i * (len(ranked) - 1) / (count - 1)) for i in range(count)]
    seen: dict[int, None] = dict.fromkeys(positions)  # de-dupe, keep order
    return [ranked[i].kernel for i in seen]


def bars(frame: pd.DataFrame, arms: Sequence[str], kernels: Sequence[str]) -> list[Bar]:
    """Every drawn :class:`Bar`, one per measured (arm, kernel) cell."""
    grid = arm_kernel_geomean(frame, arms, kernels)
    out: list[Bar] = []
    for arm in arms:
        packet = experiment_tags.packet_of(arm)
        color = palette.color(packet) if packet else palette.control_color()
        for kernel in kernels:
            value = grid.get((arm, kernel))
            if value is None or value <= 0:
                continue
            out.append(Bar(kernel, arm, packet, summary.log2_change(value), color))
    return out


def arm_label(arm: str) -> str:
    """One arm's row label: its model, plus its packet when it carries one."""
    model = experiment_tags.model_name(experiment_tags.model_of(arm))
    packet = experiment_tags.packet_of(arm)
    return f"{model} + {experiment_tags.packet_name(packet)}" if packet else model


def drawn_bars(
    frame: pd.DataFrame,
    arms: Sequence[str] | None = None,
    kernels: Sequence[str] | None = None,
    kernel_count: int = DEFAULT_KERNEL_COUNT,
) -> tuple[list[str], list[str], list[Bar]]:
    """``(arms, kernels, bars)`` for :func:`figure_stack3d` and its own data table -- ONE place
    that resolves the defaults and drops empty rows, so the figure and
    :func:`hpcagent_bench.stats.figures.stack3d.bars_table` can never disagree about which arms or
    kernels were actually drawn.

    ``arms`` defaults to every arm the frame carries a speed-up for; ``kernels`` defaults to
    :func:`representative_kernels`. An empty frame, or a frame with nothing drawable, returns three
    empty sequences rather than raising -- matching every other figure builder's "nothing to draw"
    contract in this repo (:func:`hpcagent_bench.stats.figures.scaling.curves`'s own docstring
    states the same rule for an empty-or-column-less frame).
    """
    if frame.empty:
        return [], [], []
    if arms is None:
        arms = list(dict.fromkeys(str(a) for a in frame.get("arm", pd.Series(dtype=str)) if a))
    if kernels is None:
        kernels = list(representative_kernels(frame, kernel_count))
    drawn = bars(frame, arms, kernels)
    if not drawn:
        return [], [], []
    # An auto-detected arm the frame names but never measured a speed-up for (e.g. a scaling-only
    # arm sharing the frame with per-kernel rows) draws no bar but would still claim a Y row -- an
    # empty row a reader cannot tell from "measured, near 1x". Narrow the axis to arms with at
    # least one drawn bar, in the order the caller (or the frame) gave them.
    measured = {bar.arm for bar in drawn}
    return [arm for arm in arms if arm in measured], list(kernels), drawn


def figure_stack3d(
    frame: pd.DataFrame,
    arms: Sequence[str] | None = None,
    kernels: Sequence[str] | None = None,
    kernel_count: int = DEFAULT_KERNEL_COUNT,
    title: str = "",
) -> matplotlib.figure.Figure | None:
    """The 3D stacked-bar figure over ``frame``'s own arms and a representative kernel subset
    (:func:`drawn_bars`). Returns ``None`` when nothing measured survives the selection.
    """
    arms, kernels, drawn = drawn_bars(frame, arms, kernels, kernel_count)
    if not drawn:
        return None

    kernel_index = {kernel: i for i, kernel in enumerate(kernels)}
    arm_index = {arm: i for i, arm in enumerate(arms)}

    fig = plt.figure(figsize=FIGSIZE_IN)
    fig.set_dpi(plotstyle.SAVE_DPI)
    ax = fig.add_axes((0.06, 0.20, 0.90, 0.68), projection="3d")
    ax.view_init(elev=ELEVATION_DEG, azim=AZIMUTH_DEG)
    ax.set_box_aspect((1.7, 1.0, 0.85))

    dx = dy = BAR_FRACTION
    for bar in drawn:
        x = kernel_index[bar.kernel]
        y = arm_index[bar.arm]
        z0, height = (0.0, bar.log2_speedup) if bar.log2_speedup >= 0 else (bar.log2_speedup, -bar.log2_speedup)
        ax.bar3d(
            x - dx / 2.0, y - dy / 2.0, z0, dx, dy, max(height, 1e-6),
            color=bar.color, edgecolor="white", linewidth=0.4, shade=True,
        )  # fmt: skip

    ax.set_xticks(list(kernel_index.values()))
    ax.set_xticklabels([per_kernel.kernel_tick_label(k) for k in kernels], rotation=55, ha="right", fontsize=6.5)
    ax.set_yticks(list(arm_index.values()))
    ax.set_yticklabels([arm_label(a) for a in arms], fontsize=7.0)
    ax.set_zlabel("Speed-Up (log2)", fontsize=8.0)
    z_values = [b.log2_speedup for b in drawn]
    span = max(1.0, max((abs(v) for v in z_values), default=1.0))
    ax.set_zlim(-span * 1.05, span * 1.05)
    ax.zaxis.set_major_formatter(FuncFormatter(plotstyle.log2_ratio_tick))
    ax.tick_params(axis="z", labelsize=7.0)
    ax.tick_params(axis="x", pad=-2)
    ax.tick_params(axis="y", pad=-2)

    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=palette.color(p) if p else palette.control_color(), markersize=8, label=experiment_tags.packet_name(p) if p else "No Packet")
        for p in dict.fromkeys(bar.packet for bar in drawn)
    ]  # fmt: skip
    plotstyle.legend_below(fig, handles, y=0.03)
    if title:
        plotstyle.title(fig, title)
    return fig


def bars_table(drawn: Sequence[Bar]) -> pd.DataFrame:
    """The data table behind the figure: one row per drawn bar."""
    return pd.DataFrame(
        {
            "arm": bar.arm,
            "benchmark": bar.kernel,
            "packet": bar.packet,
            "geomean_speedup": 2.0**bar.log2_speedup,
            "log2_speedup": bar.log2_speedup,
        }
        for bar in drawn
    )


def save(fig: matplotlib.figure.Figure, stem: pathlib.Path) -> pathlib.Path:
    """:func:`hpcagent_bench.stats.style.save`, ``fixed=True`` -- a 3D axes' tight bbox is
    unstable across renders (the legend and the rotated ticks probe slightly differently each
    time), so this figure keeps its authored canvas size instead of cropping to the ink."""
    return plotstyle.save(fig, stem, fixed=True)
