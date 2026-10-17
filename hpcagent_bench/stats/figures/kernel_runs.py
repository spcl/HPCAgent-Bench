# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every run (slot) of every model, per kernel, with the conditions it ran under stacked side by side.

The x axis holds one group per CONDITION (:func:`condition_of`: a served temperature ``t<T>``, a skill
packet, or the default), dashed rules between them and the condition named above, and in each group one
column per kernel; one condition is one group and no rule. In a column the models sit side by side in
their colours.
The three rows of the efficacy dot row (:mod:`hpcagent_bench.stats.figures.efficacy`: its labels, its row
heights, print type) share that axis:

- speedup: the final-grade speedup of every run as a dot, solved filled and unsolved hollow at 1x, with
  the box over the graded runs;
- token cost: every run's tokens as a dot with its box, priced by the caller;
- solved: solved over graded runs per cell, a census mark with ``solved/graded`` beside it.

The condition comes from the setup name: ``temperature3-<model>-c-t<T>`` served temperature ``T`` (condition
``t<T>``), ``llr40-<model>-c-cpf-src`` ran packet ``cpf-src``, and a setup with neither ran the default
(condition ``""``: the model's own temperature, :data:`DEFAULT_TEMPERATURE`, and no packet).

    fig = kernel_runs_figure(runs, kernels, conditions=parse_conditions("t0,default"))
"""

import enum
import math
import re
import statistics
import textwrap
from collections.abc import Sequence
from typing import cast

import matplotlib.artist
import matplotlib.axes
import matplotlib.collections
import matplotlib.figure
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hpcagent_bench import study_tags
from hpcagent_bench.stats import palette, population, reliability, summary
from hpcagent_bench.stats import style as plotstyle
from hpcagent_bench.stats.figures import efficacy
from hpcagent_bench.stats.figures import per_kernel as pk

__all__ = [
    "DEFAULT_TEMPERATURE",
    "TEMPERATURE_SUFFIX",
    "Mode",
    "column_key",
    "condition_label",
    "condition_of",
    "kernel_runs_figure",
    "order_conditions",
    "parse_conditions",
    "run_costs",
    "select_conditions",
    "temperature_label",
    "temperature_of",
]


class Mode(enum.Enum):
    """How a cell's runs are summarized under their dots: a box (median, quartiles, whiskers to the
    extremes), or a violin of their density with the median's 95% bootstrap interval."""

    BOX = "box"
    VIOLIN = "violin"


#: ``-t<T>`` closing a setup name: the temperature the job served.
TEMPERATURE_SUFFIX: re.Pattern[str] = re.compile(r"-t(?P<value>\d+(?:\.\d+)?)$")
#: What a setup without the suffix served: the model's ``generation_config.json`` value (1.0 for every
#: model temperature3 runs; the job logs it).
DEFAULT_TEMPERATURE: float = 1.0
#: The rows top to bottom, as the efficacy dot row names them.
MEASURES: tuple[str, ...] = ("speedup", "cost", "success")
#: How far apart a column's outermost two models sit, in columns: close, so one column reads as one group,
#: and the share of a model's step its box (or violin) takes, so neighbours never touch.
DODGE_SPAN: float = 0.34
BOX_STEP_SHARE: float = 0.6
VIOLIN_STEP_SHARE: float = 0.9
#: A run dot's and a solved-share mark's diameter in points: twenty runs share one narrow box at print size.
RUN_MARK_PT: float = 1.6
RATE_MARK_PT: float = 3.0
#: The solved row's axis in percent, and above which a cell's count is printed under its mark.
RATE_TICKS: tuple[float, ...] = (0.0, 50.0, 100.0)
RATE_HEADROOM: float = 8.0
COUNT_BELOW: float = 50.0
#: Key entries for an unsolved run and for a run still owed its final grade.
UNSOLVED_LABEL: str = "Unsolved Run (at 1x)"
#: The box: every run is drawn, so its whiskers reach the extremes rather than a Tukey fence.
BOX_LABEL: str = "Box: Median and Quartiles of the Graded Runs, Whiskers to Min and Max"
WHISKERS: tuple[float, float] = (0.0, 100.0)
#: The square one-column figure's shorter key entries and group titles (``T = 0``).
COMPACT_BOX_LABEL: str = "Box: Median, Quartiles, Min to Max"
COMPACT_VIOLIN_LABEL: str = "Violin: Run Density"
#: The violin mode's key entries, how many lightness steps darker than its model the interval is drawn,
#: and its line weights (interval, median tick) in points.
VIOLIN_LABEL: str = "Violin: Density of the Graded Runs"
CI_LABEL: str = f"Median with {summary.DEFAULT_CONFIDENCE:.0%} Bootstrap Interval"
CI_DARKEN: int = 2
CI_LINE_WIDTH: float = 1.8
MEDIAN_LINE_WIDTH: float = 1.2
#: The kernel names under the columns: horizontal, folded at this width onto at most :data:`NAME_LINES`.
NAME_WRAP: int = 10
NAME_LINES: int = 3
OWED_LABEL: str = "Not Final-Graded Yet"
#: The column key's separator between the condition and the kernel.
KEY_SEP: str = "|"
#: Up to this many condition groups the figure is one ACM column wide and square.
SQUARE_GROUPS: int = 2


#: How ``--conditions`` spells the default condition (no temperature suffix, no packet).
DEFAULT_CONDITION: str = "default"


def temperature_of(setup: str) -> float:
    """The sampling temperature ``setup`` served."""
    match = TEMPERATURE_SUFFIX.search(setup)
    return float(match.group("value")) if match else DEFAULT_TEMPERATURE


def condition_of(setup: str) -> str:
    """The condition ``setup`` ran under: ``t<T>`` for a served temperature, else its packet (``""`` the default)."""
    match = TEMPERATURE_SUFFIX.search(setup)
    return f"t{match.group('value')}" if match else study_tags.packet_of(setup)


def is_temperature(condition: str) -> bool:
    return TEMPERATURE_SUFFIX.search(f"-{condition}") is not None


def condition_temperature(condition: str) -> float:
    """The temperature a condition served: its ``t<T>``, or :data:`DEFAULT_TEMPERATURE` for any other."""
    return temperature_of(f"-{condition}")


def temperature_label(value: float, defaults: bool, compact: bool = False) -> str:
    """``Temperature = 1.5`` (``T = 1.5`` when ``compact``); ``defaults`` adds ``(Default)`` to the group the
    unsuffixed setups served."""
    text = f"{'T' if compact else 'Temperature'} = {value:g}"
    return f"{text} (Default)" if defaults else text


def condition_label(condition: str, temperatures: bool, compact: bool = False) -> str:
    """A group's title: ``Temperature = 0`` (``T = 0`` when ``compact``) for a temperature; the default as
    ``Temperature = 1 (Default)`` when ``temperatures`` (the figure compares temperatures), else as the
    packet control; a packet by its display name."""
    if is_temperature(condition):
        return temperature_label(condition_temperature(condition), False, compact)
    if condition == "" and temperatures:
        return temperature_label(DEFAULT_TEMPERATURE, True, compact)
    return study_tags.packet_name(condition)


def order_conditions(conditions: Sequence[str]) -> list[str]:
    """Temperatures (the default among them at its own value) by value, then packets in registry order."""
    temperatures = any(is_temperature(condition) for condition in conditions)
    packet_rank = {key: rank for rank, key in enumerate(study_tags.order("packets"))}

    def key(condition: str) -> tuple[int, float]:
        if is_temperature(condition) or (condition == "" and temperatures):
            return 0, condition_temperature(condition)
        return 1, float(packet_rank.get(condition, len(packet_rank)))

    return sorted(set(conditions), key=key)


def parse_conditions(text: str) -> tuple[str, ...]:
    """``"t0,default"`` -> ``("t0", "")``: comma-separated conditions, ``default`` naming the default."""
    return tuple("" if item == DEFAULT_CONDITION else item for item in text.split(",") if item)


def select_conditions(runs: pd.DataFrame, conditions: Sequence[str]) -> pd.DataFrame:
    """The rows of ``runs`` whose setup ran one of ``conditions``."""
    return runs.loc[runs["setup"].map(lambda setup: condition_of(str(setup)) in conditions)]


def column_key(condition: str, kernel: str) -> str:
    """One x column: ``kernel`` under ``condition``."""
    return f"{condition}{KEY_SEP}{kernel}"


def kernel_of_key(key: str) -> str:
    return key.split(KEY_SEP, 1)[1]


def run_costs(frame: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """``runs`` (:func:`population.designed_runs`) with each run's ``tokens``: its episode row's total
    (:func:`population.episode_tokens`), already priced by the caller; NaN where none was recorded."""
    tokens = population.episode_tokens(frame).loc[:, [*population.EPISODE_KEY, "tokens"]]
    return runs.merge(tokens, on=list(population.EPISODE_KEY), how="left")


def cost_cell(cell: pk.KernelCell, runs: pd.DataFrame) -> pk.KernelCell | None:
    """``cell``'s runs at their token cost, in the same order and states; ``None`` when no run has one."""
    tokens = {int(slot): float(cost) for slot, cost in zip(runs[population.SLOT_COLUMN], runs["tokens"], strict=True)}
    made = tuple(pk.Run(run.number, tokens.get(run.number, math.nan), run.state) for run in cell.runs)
    costed = tuple(run for run in made if pk.usable(run.value))
    if not costed:
        return None
    return pk.KernelCell(cell.kernel, tuple(sorted(run.value for run in costed)), runs=costed)


def model_series(
    runs: pd.DataFrame, kernels: Sequence[str], conditions: Sequence[str]
) -> tuple[tuple[pk.Series, ...], tuple[pk.Series, ...]]:
    """Per model (registry order) its speedup series and its cost series, cells keyed by :func:`column_key`."""
    models = palette.in_order({study_tags.model_of(str(setup)) for setup in runs["setup"]})
    score: list[pk.Series] = []
    cost: list[pk.Series] = []
    for model in models:
        speedups: list[pk.KernelCell] = []
        costs: list[pk.KernelCell] = []
        for condition in conditions:
            mine = runs.loc[
                (runs["setup"].map(study_tags.model_of) == model) & (runs["setup"].map(condition_of) == condition)
            ]
            for cell in pk.run_cells(mine, kernels):
                keyed = pk.KernelCell(column_key(condition, cell.kernel), cell.episodes, runs=cell.runs)
                speedups.append(keyed)
                priced = cost_cell(keyed, mine.loc[mine["kernel"] == cell.kernel])
                if priced is not None:
                    costs.append(priced)
        color, name = palette.model_color(model), study_tags.model_name(model)
        score.append(pk.Series(name, tuple(speedups), color))
        cost.append(pk.Series(name, tuple(costs), color))
    return tuple(score), tuple(cost)


def kernel_name(kernel: str) -> str:
    """The kernel's short manifest name folded onto at most :data:`NAME_LINES` horizontal lines."""
    name = study_tags.kernel_short_display_name(kernel)
    return "\n".join(textwrap.wrap(name, NAME_WRAP, max_lines=NAME_LINES, placeholder="."))


def cell_width(n_series: int, mode: "Mode") -> float:
    """One model's box or violin width: its share of the dodge step, a lone model's at the same share."""
    share = BOX_STEP_SHARE if mode == Mode.BOX else VIOLIN_STEP_SHARE
    return share * DODGE_SPAN / max(n_series - 1, 1)


def style_rate_panel(ax: matplotlib.axes.Axes, columns: Sequence[str], type_: plotstyle.TypeScale) -> None:
    """The solved row's chrome: the shared column axis with the kernel names, a 0-100 % axis."""
    ax.set_xlim(-0.6, len(columns) - 0.4)
    ax.set_xticks(list(range(len(columns))))
    ax.set_xticklabels(
        [kernel_name(kernel_of_key(key)) for key in columns], fontsize=type_.annotation_pt, linespacing=0.95
    )
    ax.set_ylim(0.0, 100.0 + RATE_HEADROOM)
    ax.set_yticks(list(RATE_TICKS))
    ax.tick_params(axis="y", labelsize=type_.tick_pt)
    ax.yaxis.grid(True, color=plotstyle.RULE, linewidth=pk.GRID_LINE_WIDTH, zorder=0)
    ax.set_ylabel(efficacy.MEASURE_LABELS["success"], fontsize=type_.label_pt)
    plotstyle.despine(ax)


def draw_groups(
    axes: Sequence[matplotlib.axes.Axes],
    conditions: Sequence[str],
    n_kernels: int,
    type_: plotstyle.TypeScale,
    compact: bool = False,
) -> None:
    """A dashed rule between consecutive condition groups on every row, the condition over its group on
    the top row; a lone condition is neither ruled nor named."""
    if len(conditions) < 2:
        return
    temperatures = any(is_temperature(condition) for condition in conditions)
    for index in range(1, len(conditions)):
        for ax in axes:
            ax.axvline(
                index * n_kernels - 0.5, color=plotstyle.MUTED, linestyle=(0, (3, 3)), linewidth=type_.line_width
            )
    for index, condition in enumerate(conditions):
        centre = index * n_kernels + (n_kernels - 1) / 2.0
        axes[0].annotate(
            condition_label(condition, temperatures, compact), xy=(centre, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=type_.label_pt,
            color=plotstyle.INK, annotation_clip=False,
        )  # fmt: skip


def draw_runs(ax: matplotlib.axes.Axes, cell: pk.KernelCell, x: float, color: str, width: float) -> None:
    """Every run of ``cell`` as a small dot across its box in run order at its value: solved filled,
    unsolved hollow, a run owed its final grade a "?"."""
    size = RUN_MARK_PT**2
    for run, dx in zip(cell.runs, pk.dodge_offsets(len(cell.runs), pk.RUN_SPREAD * width), strict=True):
        if run.state == population.RunState.OWED:
            plotstyle.pending_mark(ax, x + dx, run.value, plotstyle.MUTED, size=pk.OWED_SCALE * size)
        else:
            plotstyle.point_mark(ax, x + dx, run.value, color, "o", run.state == population.RunState.SOLVED, size)


def draw_violin(ax: matplotlib.axes.Axes, cell: pk.KernelCell, x: float, color: str, width: float) -> None:
    """A violin of ``cell``'s graded runs, its density estimated over their logarithms so it keeps its
    shape on the log axis; none when the runs are too few or all equal."""
    values = np.log10(np.asarray(cell.episodes, dtype=np.float64))
    if cell.n < pk.MIN_EPISODES_FOR_SPREAD or np.ptp(values) == 0.0:
        return
    parts = ax.violinplot([values], positions=[x], widths=width, showextrema=False)
    for body in cast("list[matplotlib.collections.PolyCollection]", parts["bodies"]):
        for path in body.get_paths():
            vertices = np.asarray(path.vertices, dtype=np.float64)
            vertices[:, 1] = np.power(10.0, vertices[:, 1])
            path.vertices = vertices
        body.set(facecolor=color, edgecolor=color, alpha=pk.RUN_BOX_ALPHA, linewidth=pk.BOX_LINE_WIDTH)


def draw_interval(
    ax: matplotlib.axes.Axes, cell: pk.KernelCell, x: float, color: str, width: float, log2: bool
) -> None:
    """``cell``'s median as a short tick with its bootstrap interval (:func:`per_kernel.bootstrap_point`) as
    a line through it, both darker than the model, over the run dots; the interval is withheld below
    :data:`~hpcagent_bench.stats.summary.MIN_INTERVAL_SAMPLES` runs."""
    median, low, high = pk.bootstrap_point(cell, log2)
    if not math.isfinite(median):
        return
    dark, z = palette.darken(color, CI_DARKEN), plotstyle.MARK_Z + 2.0
    if math.isfinite(low) and math.isfinite(high) and low < high:
        ax.vlines(x, low, high, color=dark, linewidth=CI_LINE_WIDTH, zorder=z)
    ax.hlines(median, x - width / 3.0, x + width / 3.0, color=dark, linewidth=MEDIAN_LINE_WIDTH, zorder=z)


def draw_cells(
    ax: matplotlib.axes.Axes,
    metric: pk.Metric,
    x_of: dict[str, int],
    type_: plotstyle.TypeScale,
    mode: Mode,
) -> None:
    """Per model and column the ``mode`` summary of the graded runs, every run a dot (:func:`draw_runs`)."""
    width = cell_width(len(metric.series), mode)
    for one, offset in zip(metric.series, pk.dodge_offsets(len(metric.series), DODGE_SPAN), strict=True):
        if mode == Mode.BOX:
            boxed = [cell for cell in one.cells if cell.n >= pk.MIN_EPISODES_FOR_SPREAD]
            positions = [x_of[cell.kernel] + offset for cell in boxed]
            pk.box_cells(ax, boxed, positions, one.color, type_, width, pk.RUN_BOX_ALPHA, WHISKERS)
        for cell in one.cells:
            x = x_of[cell.kernel] + offset
            if mode == Mode.VIOLIN:
                draw_violin(ax, cell, x, one.color, width)
            draw_runs(ax, cell, x, one.color, width)
            if mode == Mode.VIOLIN:
                draw_interval(ax, cell, x, one.color, width, metric.log2_space)


def draw_rates(
    ax: matplotlib.axes.Axes, series: Sequence[pk.Series], x_of: dict[str, int], type_: plotstyle.TypeScale
) -> None:
    """Each model's solved share of the graded runs per column with its ``solved/graded`` beside it,
    under the mark above :data:`COUNT_BELOW`; a cell whose runs are all owed draws nothing."""
    for one, offset in zip(series, pk.dodge_offsets(len(series), DODGE_SPAN), strict=True):
        for cell in one.cells:
            graded = [run for run in cell.runs if run.state != population.RunState.OWED]
            if not graded:
                continue
            solved = sum(run.state == population.RunState.SOLVED for run in graded)
            x, rate = x_of[cell.kernel] + offset, 100.0 * solved / len(graded)
            plotstyle.point_mark(ax, x, rate, one.color, "o", True, size=RATE_MARK_PT**2)
            below = rate > COUNT_BELOW
            ax.annotate(
                pk.run_count_label(cell.runs), xy=(x, rate), xytext=(0, -3 if below else 3),
                textcoords="offset points", ha="center", va="top" if below else "bottom", rotation=90,
                fontsize=type_.legend_pt, color=plotstyle.INK, annotation_clip=False,
            )  # fmt: skip


def key_entries(
    series: Sequence[pk.Series], states: set[population.RunState], mode: Mode, compact: bool = False
) -> list[matplotlib.artist.Artist]:
    """The models, the ``mode`` statistic, and the run states the figure actually draws; ``compact`` for the
    one-column figure."""
    mark = pk.LEGEND_MARK_PT
    entries: list[matplotlib.artist.Artist] = [
        matplotlib.lines.Line2D([], [], marker="o", linestyle="none", color=one.color, markersize=mark, label=one.label)
        for one in series
    ]
    if mode == Mode.BOX:
        summary_label = COMPACT_BOX_LABEL if compact else BOX_LABEL
    else:
        summary_label = COMPACT_VIOLIN_LABEL if compact else VIOLIN_LABEL
    entries.append(matplotlib.patches.Patch(facecolor=plotstyle.MUTED, alpha=pk.RUN_BOX_ALPHA, label=summary_label))
    if mode == Mode.VIOLIN:
        entries.append(matplotlib.lines.Line2D([], [], color=plotstyle.INK, linewidth=CI_LINE_WIDTH, label=CI_LABEL))
    if population.RunState.UNSOLVED in states:
        entries.append(
            matplotlib.lines.Line2D(
                [], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=plotstyle.MUTED,
                markersize=mark, label=UNSOLVED_LABEL,
            )
        )  # fmt: skip
    if population.RunState.OWED in states:
        owed = plotstyle.pending_legend_mark(mark)
        owed.set_label(OWED_LABEL)
        entries.append(owed)
    return entries


def kernel_runs_figure(
    runs: pd.DataFrame,
    kernels: Sequence[str],
    title: str = "",
    width_in: float | None = None,
    allow_owed: bool = False,
    cost_label: str = efficacy.MEASURE_LABELS["cost"],
    mode: Mode = Mode.BOX,
    conditions: Sequence[str] = (),
) -> matplotlib.figure.Figure:
    """The three rows over (condition, kernel) columns for ``runs`` (:func:`run_costs`), drawn at ``width_in``
    on the print scale, each cell's runs summarized by ``mode``. ``conditions`` picks and orders the groups
    (default: every condition the runs ran, :func:`order_conditions`). Without ``width_in``, up to
    :data:`SQUARE_GROUPS` groups make a square one-column figure, more a text-wide one. Refuses owed runs
    (:class:`reliability.OwedRunsError`) unless ``allow_owed``, which draws them as "?"."""
    if not allow_owed:
        reliability.cell_runs(runs)
    groups = list(conditions) or order_conditions([condition_of(str(setup)) for setup in runs["setup"]])
    columns = [column_key(condition, kernel) for condition in groups for kernel in kernels]
    score_series, cost_series = model_series(runs, kernels, groups)
    score = pk.speedup_series_metric(score_series, efficacy.MEASURE_LABELS["speedup"])
    cost = pk.token_series_metric(cost_series, cost_label)

    square = len(groups) <= SQUARE_GROUPS
    if width_in is None:
        width_in = plotstyle.ACM_COLUMN_WIDTH_IN if square else plotstyle.ACM_TEXT_WIDTH_IN
    type_ = plotstyle.PRINT_SCALE
    heights = [efficacy.DOT_ROW_HEIGHT_IN * efficacy.MEASURE_HEIGHT[measure] for measure in MEASURES]
    fig, grid = plt.subplots(
        len(MEASURES), 1, sharex=True, figsize=(width_in, sum(heights)), squeeze=False,
        gridspec_kw={"height_ratios": heights},
    )  # fmt: skip
    fig.set_dpi(plotstyle.SAVE_DPI)
    axes = [row[0] for row in grid]
    for ax, metric in zip(axes[:2], (score, cost), strict=True):
        pk.style_panel(ax, metric, columns, False, False, type_)
    style_rate_panel(axes[2], columns, type_)
    draw_groups(axes, groups, len(kernels), type_, square)
    states = {run.state for cell in score.cells for run in cell.runs}
    legend = key_entries(score_series, states, mode, square)
    pk.fit_canvas(fig, axes, title, legend, type_, statistics.fmean(heights), height_in=width_in if square else None)

    x_of = {key: index for index, key in enumerate(columns)}
    for ax, metric in zip(axes[:2], (score, cost), strict=True):
        draw_cells(ax, metric, x_of, type_, mode)
    draw_rates(axes[2], score_series, x_of, type_)
    return fig
