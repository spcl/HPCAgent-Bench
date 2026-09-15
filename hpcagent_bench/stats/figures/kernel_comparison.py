# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""llr-focus40: the DaCe canonicalize CPU column against every COMPLETE agent arm, per kernel.

TWO PANELS PER MODEL, sharing the 40-kernel row axis (:func:`hpcagent_bench.stats.style.row_axis`):
a speed-up panel (log2 ratio over Numba, the DaCe canon CPU column plus that model's own arms) over
a token panel (log10 spend, agents only -- the canon column has no tokens, since it runs no agent).
Every panel repeats the DaCe canon CPU column (the framework colour,
:func:`hpcagent_bench.stats.palette.framework_color`) beside that model's own arms (the condition
colour, :func:`hpcagent_bench.stats.palette.color`: grey control, orange CPF page, blue CPF as
source -- the same packet palette :mod:`scripts.plot_arm_summary` draws with).

Each panel carries a SUMMARY ROW below the kernel rows, past a dashed separator
(:data:`SUMMARY_ROW_GAP`): the row-axis analogue of
:func:`hpcagent_bench.stats.figures.per_kernel.draw_summary_column`, rotated onto rows because
kernels are ROWS here and COLUMNS there. Speed-up's summary is the GEOMETRIC MEAN
(:func:`summary_speedup`) -- the project-wide rule for an overall speed-up, never a median; tokens'
summary is the MEDIAN (:func:`summary_tokens`), since tokens are not a ratio. Both are named by an
annotation inside the panel's own right edge, never a y-axis tick label: every panel here shares its
y axis (``sharey=True``), and a per-panel tick label would silently lose whichever panel drew first --
the same trap :func:`per_kernel.draw_summary_column` documents for a shared X axis.

NO EFFICACY, PARETO OR COST-VS-SPEED FRAMING HERE. This figure reports what each arm cost and what
it bought, side by side, and leaves any tradeoff reading to the caption -- never a quadrant, a
frontier or the word "efficacy" in the axes themselves.

CONDITION COMES FROM THE ARM NAME, not the ``language``/``packet`` columns: the pre-regrade
extraction records them inconsistently for the SAME arm (some rows ``language=c, packet=''``,
others ``language='', packet='cpf'``), where the arm name itself is the one column every row of an
arm agrees on. :data:`ARM_PATTERN` is both the arm selector and the (model, condition) parser.

COMPLETENESS is roster coverage, not scoring: :func:`hpcagent_bench.stats.population.complete_arms`
keeps only arms with a recorded row for every kernel :data:`ARM_PATTERN` -- and a model whose arms
are ALL incomplete draws no panel at all, rather than an empty one.

PER-KERNEL VALUES ARE WHATEVER THE FRAMEWORK'S OWN POLICY ASSIGNS UNDER ``--repeats`` -- never
invented here (spec R3-R7). :func:`hpcagent_bench.stats.population.kernel_answers` is an arm's
verified answer per kernel: the LATEST run's answer for a rerun kernel (default, none when that run
verified nothing -- an earlier run's answer never stands in), the MEDIAN answer over runs that
repeat by design. A kernel with no verified answer simply has no row and draws no mark. Tokens are
:func:`hpcagent_bench.stats.population.kernel_tokens` under the SAME ``--repeats``: the latest
run's own TASK TOKEN TOTAL, or the median of the runs' task totals, bracketed by that median's own
minimum and maximum over the tasks (``Series.tokens_min``/``Series.tokens_max``, drawn as a
whisker). Tokens are NEVER summed over tasks (R6), and come only from ``record = task`` rows (T4).
The canon column is :func:`hpcagent_bench.stats.canon.kernel_speedups` over the deterministic
sweep: no episodes, no policy to pick, and no tokens spent.
"""

import dataclasses
import math
import pathlib
import re
from collections.abc import Callable, Iterable, Sequence

import matplotlib.artist
import matplotlib.axes
import matplotlib.figure
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hpcagent_bench import experiment_tags, packets
from hpcagent_bench.stats import canon, palette, population, summary
from hpcagent_bench.stats import style as plotstyle
from hpcagent_bench.stats.figures.per_kernel import speedup_tick_label

#: An arm this figure may draw, and its (model, condition) in one match: ``-c`` is the control
#: (condition ``""``), ``-c-cpf`` the CPF page, ``-c-cpfsrc`` CPF as source. C only -- Fortran has no
#: CPF spelling (mpr-artifacts/experiments/llr-focus40-cpf/README.md).
ARM_PATTERN: re.Pattern[str] = re.compile(r"^cpf-llr-focus40-(?P<model>[a-z0-9]+)-c(?:-(?P<condition>cpf|cpfsrc))?$")

#: Panel order within one model, control first.
CONDITION_ORDER: tuple[str, ...] = ("", "cpf", "cpfsrc")

#: The canon column this figure draws by default, and what it is measured against.
CANON_COLUMN: str = "dace_cpu_canonicalize"
CANON_BASELINE: str = "numba"

#: The canon series' fixed marker -- it has no model, so it never borrows one of theirs.
CANON_MARKER: str = "D"

#: A single small-multiple panel's inches (width, height contribution per kernel row). 0.27in/row
#: is what a 40-name kernel axis needs at :data:`ROW_LABEL_PT` to stop consecutive labels
#: overlapping -- :func:`hpcagent_bench.stats.style.row_axis` sizes them at the figure-wide
#: LABEL_PT (16pt), which 40 rows in a compact panel has no room for.
PANEL_WIDTH_IN: float = 1.85
ROW_HEIGHT_IN: float = 0.27

#: The kernel row labels' own font size -- smaller than :data:`hpcagent_bench.stats.style.LABEL_PT`,
#: which :func:`row_axis` applies but which a 40-row axis has no vertical room for.
ROW_LABEL_PT: float = 8.5

#: Where a "no verified answer" mark sits on the SPEED-UP panel -- the 1x reference line, since that
#: is what a served but unsolved kernel leaves standing under every scoring policy this repo has
#: (:data:`~hpcagent_bench.stats.population.NOT_DELIVERED`). HOLLOW, never filled: a real 1.0x
#: speed-up and "nothing to plot here" must not draw as one mark.
MISSING_MARKER_X: float = 1.0

#: The shared legend entry for a missing-answer mark, neutral ink since it names a STATUS, not one
#: series' identity -- a coloured entry would read as one more condition or model.
MISSING_LABEL: str = "No Verified Answer"

#: Rows of air between the last kernel row and the dashed separator, and between the separator and
#: the summary row -- the row-axis analogue of
#: :data:`~hpcagent_bench.stats.figures.per_kernel.SUMMARY_GAP`.
SUMMARY_ROW_GAP: float = 0.9


@dataclasses.dataclass(frozen=True, slots=True)
class Series:
    """One drawn series: a per-kernel speed-up and per-kernel token spend, already keyed to the
    kernels each has a value for. The canon series has an empty ``tokens`` -- it runs no agent.
    ``tokens_min``/``tokens_max`` bracket a ``--repeats median`` kernel's token value with the
    minimum and maximum over its tasks (R5); empty under ``--repeats latest``, where one task IS
    the value and there is nothing to bracket."""

    key: str
    label: str
    color: str
    marker: str
    model: str
    condition: str
    values: dict[str, float]
    tokens: dict[str, float]
    tokens_min: dict[str, float]
    tokens_max: dict[str, float]


def parse_arm(arm: str, pattern: re.Pattern[str] = ARM_PATTERN) -> tuple[str, str] | None:
    """``arm``'s (model, condition), or ``None`` when it is not one this figure draws."""
    match = pattern.fullmatch(arm)
    if match is None:
        return None
    return match.group("model"), match.group("condition") or ""


def candidate_arms(frame: pd.DataFrame, pattern: re.Pattern[str] = ARM_PATTERN) -> dict[str, tuple[str, str]]:
    """Every distinct arm of ``frame`` that ``pattern`` names: arm -> (model, condition)."""
    out: dict[str, tuple[str, str]] = {}
    for arm in frame["arm"].dropna().astype(str).unique():
        parsed = parse_arm(str(arm), pattern)
        if parsed is not None:
            out[str(arm)] = parsed
    return out


def arm_speedups(frame: pd.DataFrame, arm: str, repeats: population.RepeatPolicy = "latest") -> dict[str, float]:
    """``arm``'s verified final answer per kernel under ``repeats`` (:func:`population.kernel_answers`).

    A kernel with no verified answer is absent, never entered at any stand-in value: that is the
    framework's own "solved" population, applied here rather than reinvented.
    """
    subset = frame[frame["arm"].astype(str) == arm]
    answers = population.kernel_answers(subset, repeats=repeats)
    if "speedup" not in answers.columns:
        return {}
    return {str(kernel): float(value) for kernel, value in answers["speedup"].items() if value > 0}


def arm_tokens(
    frame: pd.DataFrame, arm: str, repeats: population.RepeatPolicy = "latest"
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """``arm``'s per-kernel token total under ``repeats`` (:func:`population.kernel_tokens`), plus
    the minimum and maximum over the tasks when ``repeats="median"`` (R5).

    Tokens come only from ``record = task`` rows (T4); a kernel with no task total is absent, never
    entered at any stand-in value, matching :func:`arm_speedups`. Under ``latest`` one task IS the
    kernel's value, so the range dicts come back empty -- there is nothing to bracket.
    """
    subset = frame[frame["arm"].astype(str) == arm]
    totals = population.kernel_tokens(subset, ("arm", "benchmark"), repeats=repeats)
    values = {str(kernel): float(value) for (_, kernel), value in totals.items() if value > 0}
    if repeats != "median" or not values:
        return values, {}, {}
    episodes = population.episode_tokens(subset, ("arm", "benchmark"))
    grouped = episodes.groupby("benchmark").tokens
    low = {str(kernel): float(value) for kernel, value in grouped.min().items() if str(kernel) in values}
    high = {str(kernel): float(value) for kernel, value in grouped.max().items() if str(kernel) in values}
    return values, low, high


def canon_speedups(
    canon_frame: pd.DataFrame, baseline: str = CANON_BASELINE, column: str = CANON_COLUMN
) -> dict[str, float]:
    """The canon column's per-kernel speed-up over ``baseline`` (:func:`hpcagent_bench.stats.canon.kernel_speedups`)."""
    return canon.kernel_speedups(canon.read_times(canon_frame), baseline, column)


def roster_of(canon_frame: pd.DataFrame) -> list[str]:
    """The 40 llr-focus40 kernels: every kernel the canon sweep names, sorted -- the same order
    :func:`hpcagent_bench.stats.canon.speedups` already reduces its ratios in."""
    return sorted({str(k) for k in canon_frame["kernel"].dropna().unique()})


def condition_label(condition: str) -> str:
    """This figure's display text for a condition tag (``""`` control, ``cpf``, ``cpfsrc``).

    The control reads "No Packet", never the registry's "No Skill Packet": this figure's treatments
    (CPF page, CPF as source) are not skills, and borrowing the skills experiments' wording for the
    control names the wrong thing (:func:`hpcagent_bench.packets.control_label`, gated on the
    treatment set rather than hardcoded here or in the registry).
    """
    if condition == "":
        return packets.control_label(CONDITION_ORDER[1:])
    return experiment_tags.names("packets").get(condition, condition)


def rank_condition(condition: str, order: Sequence[str] = CONDITION_ORDER) -> tuple[int, str]:
    """``order``'s conditions first, in their declared order, then anything else alphabetically.

    A condition axis need not be a skill packet: git-scicomp's arm names carry ``kernel``/``repo``,
    neither of which is in :data:`CONDITION_ORDER`. ``CONDITION_ORDER.index`` would raise on those;
    this is the same "known order first, unregistered last" tiebreak :func:`palette.in_order` and
    :mod:`scripts.plot_arm_summary`'s ``condition_order`` already use for model and packet axes.
    """
    return (order.index(condition), "") if condition in order else (len(order), condition)


def build_panels(
    observations: pd.DataFrame,
    roster: Sequence[str],
    pattern: re.Pattern[str] = ARM_PATTERN,
    canon_frame: pd.DataFrame | None = None,
    canon_column: str = CANON_COLUMN,
    canon_baseline: str = CANON_BASELINE,
    include_incomplete: bool = False,
    condition_order: Sequence[str] = CONDITION_ORDER,
    repeats: population.RepeatPolicy = "latest",
) -> tuple[dict[str, list[Series]], Series | None, dict[str, int]]:
    """model -> its arm series (condition order), the deterministic reference series shared by
    every panel (``None`` when the caller has no such column -- llr-focus40-cpf's DaCe canon,
    git-scicomp's has none), and the dropped arms with how many roster kernels each covered.

    A model every one of whose candidate arms was dropped for incomplete coverage gets no key at
    all -- :func:`figure` therefore draws it no panel, per the figure's own contract.
    """
    candidates = candidate_arms(observations, pattern)
    frame = observations[observations["arm"].astype(str).isin(candidates)]
    if include_incomplete:
        kept, dropped = list(candidates), {}
    else:
        kept, dropped = population.complete_arms(frame, roster)
    canon_mark = (
        Series(
            "canon",
            f"DaCe Canon CPU (vs {canon_baseline})",
            palette.framework_color(canon_column),
            CANON_MARKER,
            "",
            "",
            canon_speedups(canon_frame, canon_baseline, canon_column),
            {},
            {},
            {},
        )
        if canon_frame is not None
        else None
    )
    by_model: dict[str, list[Series]] = {}
    for arm in kept:
        model, condition = candidates[arm]
        values = arm_speedups(frame, arm, repeats)
        if not values:
            continue
        tokens, tokens_min, tokens_max = arm_tokens(frame, arm, repeats)
        series = Series(
            arm,
            condition_label(condition),
            palette.color(condition),
            palette.marker(model),
            model,
            condition,
            values,
            tokens,
            tokens_min,
            tokens_max,
        )
        by_model.setdefault(model, []).append(series)
    for model, series_list in by_model.items():
        series_list.sort(key=lambda series: rank_condition(series.condition, condition_order))
    ordered = [model for model in palette.in_order(by_model.keys(), "models") if model in by_model]
    panels = {model: by_model[model] for model in ordered}
    return panels, canon_mark, dropped


def value_ticks(values: Iterable[float]) -> list[float]:
    """Powers of two spanning every plotted speed-up, always at least ``1/4x .. 4x`` -- an X-axis
    twin of :func:`hpcagent_bench.stats.figures.per_kernel.speedup_yticks`."""
    finite = [v for v in values if math.isfinite(v) and v > 0]
    low, high = (min(finite), max(finite)) if finite else (1.0, 1.0)
    low_exp = min(-2, math.floor(math.log2(low)))
    high_exp = max(2, math.ceil(math.log2(high)))
    return [2.0**exp for exp in range(low_exp, high_exp + 1)]


def token_axis_limits(values: Iterable[float]) -> tuple[float, float]:
    """Decade-rounded ``(low, high)`` spanning every plotted token value -- the log10 twin of
    :func:`value_ticks`, computed ONCE for the whole figure and shared by every token panel so a
    position means the same spend everywhere (see :func:`style_speedup_x_axis`)."""
    finite = [v for v in values if math.isfinite(v) and v > 0]
    if not finite:
        return 1.0, 10.0
    low, high = min(finite), max(finite)
    return 10.0 ** math.floor(math.log10(low)), 10.0 ** math.ceil(math.log10(high))


def summary_speedup(values: Iterable[float]) -> float:
    """The geometric mean over ``values`` -- speed-up's overall value is ALWAYS the geometric mean
    (the project-wide reporting rule; :func:`hpcagent_bench.stats.summary.geomean`), never a median,
    which equals the geomean only when the values happen to be symmetric in log space."""
    return summary.geomean(list(values), unusable="drop")


def summary_tokens(values: Iterable[float]) -> float:
    """The median over ``values`` -- tokens are not a ratio, so the summary stays the median, same
    as every per-kernel token figure in this repo."""
    finite = [v for v in values if math.isfinite(v) and v > 0.0]
    return float(np.median(finite)) if finite else math.nan


def style_speedup_x_axis(ax: matplotlib.axes.Axes, ticks: Sequence[float]) -> None:
    """The log2 ratio axis and 1x reference line, same reading as :mod:`hpcagent_bench.stats.figures.per_kernel`,
    rotated onto X because the row axis here carries the kernel, not the value.

    ``ticks`` is computed ONCE for the whole figure (:func:`figure`) and passed to every panel,
    with the SAME explicit ``xlim`` set from it: a position then means the same ratio in every
    panel, where each panel picking its own from its own data let one model's wider spread (Kimi
    reaching 128x) autoscale past where the others stopped, so 8x sat at a different x in every
    panel of one figure.
    """
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([speedup_tick_label(tick) for tick in ticks], fontsize=plotstyle.TICK_PT * 0.55, rotation=90)
    ax.set_xlim(ticks[0] / 1.3, ticks[-1] * 1.3)
    ax.axvline(1.0, color=plotstyle.REFERENCE, linewidth=0.9, zorder=1)


def style_token_x_axis(ax: matplotlib.axes.Axes, limits: tuple[float, float]) -> None:
    """The log10 token axis, shared ``limits`` across every panel -- same discipline as
    :func:`style_speedup_x_axis`, base 10 since a token count is a magnitude, not a power-of-two
    ratio. Rotated majors at the same size as the speed-up panel's: a token axis spanning several
    decades gets a major every 1/2/5 within each (:func:`plotstyle.value_axis`), too many to stay
    horizontal at this panel's width without overlapping."""
    ax.set_xscale("log")
    ax.set_xlim(*limits)
    plotstyle.value_axis(ax, "x", log_base=10.0)
    ax.tick_params(axis="x", labelsize=plotstyle.TICK_PT * 0.55, rotation=90)


def summary_row_position(n_kernels: int) -> tuple[float, float]:
    """``(separator_y, summary_y)`` below the last kernel row -- the same two additions
    :func:`hpcagent_bench.stats.figures.per_kernel.draw_summary_column` makes on its column axis,
    rotated onto rows."""
    separator_y = n_kernels - 0.5 + SUMMARY_ROW_GAP
    return separator_y, separator_y + SUMMARY_ROW_GAP


def draw_summary_row(
    ax: matplotlib.axes.Axes,
    separator_y: float,
    summary_y: float,
    series_list: Sequence[Series],
    value_of: Callable[[Series], dict[str, float]],
    reducer: Callable[[Iterable[float]], float],
    label: str,
) -> None:
    """The dashed separator and one dodged mark per series at ``summary_y``, plus a small annotation
    naming the statistic -- INSIDE the panel's own right edge, never past it: with several model
    panels side by side and only a narrow gap between them, an annotation that crossed the edge fell
    under the next panel's opaque background (see the module docstring for why this is an
    annotation and never a y-axis tick label)."""
    ax.axhline(separator_y, color=plotstyle.RULE, linestyle=(0, (3, 3)), linewidth=1.0, zorder=1)
    n = len(series_list)
    offsets = np.linspace(-0.3, 0.3, n) if n > 1 else np.array([0.0])
    for offset, series in zip(offsets, series_list, strict=True):
        point = reducer(value_of(series).values())
        if math.isfinite(point):
            plotstyle.point_mark(ax, point, summary_y + offset, series.color, series.marker, filled=True, size=30.0)
    ax.annotate(
        label,
        xy=(0.985, summary_y),
        xycoords=("axes fraction", "data"),
        xytext=(0, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=plotstyle.TICK_PT * 0.5,
        color=plotstyle.MUTED,
        annotation_clip=False,
    )


#: Drawn under the mark's white halo (:data:`~hpcagent_bench.stats.style.FILL_Z`), never over it --
#: the same "connector under fill" order :func:`~hpcagent_bench.stats.style.point_mark` documents.
TOKEN_RANGE_Z: float = 2.0


def draw_panel(
    ax: matplotlib.axes.Axes,
    kernels: Sequence[str],
    series_list: Sequence[Series],
    value_of: Callable[[Series], dict[str, float]],
    missing_x: float,
    reducer: Callable[[Iterable[float]], float],
    summary_label: str,
    label_rows: bool,
    range_of: Callable[[Series], tuple[dict[str, float], dict[str, float]]] | None = None,
) -> None:
    """One panel, for ONE metric (:func:`value_of` reads it off each series): the kernel rows,
    dodged apart within each row, plus the summary row below them (:func:`draw_summary_row`).

    A kernel a series has no value for still draws: a HOLLOW mark in that series' own colour and
    shape, at ``missing_x`` -- present and legible rather than a gap a reader has to notice on their
    own, and hollow so it is never mistaken for a genuine value.

    ``range_of``, when given, reads a (minimum, maximum) pair per series off ``--repeats median``'s
    token spread (R5) and draws it as a thin whisker behind the mark -- absent under ``--repeats
    latest``, where a kernel has one task and nothing to bracket.
    """
    separator_y, summary_y = summary_row_position(len(kernels))
    plotstyle.row_axis(ax, kernels)
    ax.set_ylim(summary_y + 0.6, -0.5)
    if label_rows:
        ax.tick_params(axis="y", labelsize=ROW_LABEL_PT)
    else:
        ax.tick_params(axis="y", labelleft=False)
    n = len(series_list)
    offsets = np.linspace(-0.3, 0.3, n) if n > 1 else np.array([0.0])
    y_of = {kernel: i for i, kernel in enumerate(kernels)}
    for offset, series in zip(offsets, series_list, strict=True):
        values = value_of(series)
        low_of, high_of = range_of(series) if range_of is not None else ({}, {})
        for kernel in kernels:
            y = y_of[kernel] + offset
            value = values.get(kernel)
            if value is None or not math.isfinite(value) or value <= 0.0:
                plotstyle.point_mark(ax, missing_x, y, series.color, series.marker, filled=False, size=26.0)
                continue
            low, high = low_of.get(kernel), high_of.get(kernel)
            if low is not None and high is not None and math.isfinite(low) and math.isfinite(high) and low < high:
                ax.hlines(y, low, high, color=series.color, linewidth=1.1, alpha=0.55, zorder=TOKEN_RANGE_Z)
            plotstyle.point_mark(ax, value, y, series.color, series.marker, filled=True, size=26.0)
    draw_summary_row(ax, separator_y, summary_y, series_list, value_of, reducer, summary_label)


def legend_handles(
    canon_mark: Series | None, panels: dict[str, list[Series]], condition_order: Sequence[str] = CONDITION_ORDER
) -> list[matplotlib.artist.Artist]:
    """One legend for the whole figure: the optional reference mark, each condition present
    (colour), each model present (shape) -- the same two channels :mod:`scripts.plot_arm_summary`
    draws with."""
    conditions = sorted(
        {series.condition for series_list in panels.values() for series in series_list},
        key=lambda condition: rank_condition(condition, condition_order),
    )
    models = list(panels.keys())
    handles: list[matplotlib.artist.Artist] = []
    if canon_mark is not None:
        handles.append(
            matplotlib.lines.Line2D(
                [],
                [],
                marker=canon_mark.marker,
                linestyle="none",
                color=canon_mark.color,
                markersize=8,
                label=canon_mark.label,
            )
        )
    for condition in conditions:
        colour = palette.color(condition)
        handles.append(matplotlib.patches.Patch(facecolor=colour, edgecolor=colour, label=condition_label(condition)))
    for model in models:
        handles.append(
            matplotlib.lines.Line2D(
                [],
                [],
                marker=palette.marker(model),
                linestyle="none",
                color=plotstyle.MUTED,
                markersize=8,
                label=experiment_tags.model_name(model),
            )
        )
    handles.append(
        matplotlib.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=plotstyle.MUTED,
            markersize=8,
            label=MISSING_LABEL,
        )
    )
    return handles


#: One panel's data span, in kernel rows: the roster plus the summary row's own air
#: (:data:`SUMMARY_ROW_GAP`, twice) and the half-row margin :func:`~hpcagent_bench.stats.style.row_axis`
#: leaves top and bottom.
def panel_rows(n_rows: int) -> float:
    return n_rows + 2.0 * SUMMARY_ROW_GAP + 0.6


#: Inches of air between the two panel rows -- room for the speed-up panel's own rotated X tick
#: labels, which sit between it and the token panel below (:func:`figure` turns this into a
#: `hspace` FRACTION of one panel's height, since that is what `subplots_adjust` takes).
ROW_GAP_IN: float = 0.85

#: Inches reserved above the top panel row (the figure title, clear of the model-name titles) and
#: below the bottom one (its own rotated X tick labels, plus the legend).
TOP_MARGIN_IN: float = 1.15
BOTTOM_MARGIN_IN: float = 1.6


def figure_size(n_panels: int, n_rows: int, double_column: bool) -> tuple[float, float]:
    """A compact double-column insert (fixed width) or a standalone report (one width slot per
    panel). Height stacks TWO panel rows (speed-up over tokens) plus the gap between them."""
    panel_h = panel_rows(n_rows) * ROW_HEIGHT_IN
    height = 2.0 * panel_h + ROW_GAP_IN + TOP_MARGIN_IN + BOTTOM_MARGIN_IN
    if double_column:
        return plotstyle.DOUBLE_COLUMN_WIDTH, height
    return max(plotstyle.DOUBLE_COLUMN_WIDTH, PANEL_WIDTH_IN * n_panels + 1.2), height


def figure(
    panels: dict[str, list[Series]],
    canon_mark: Series | None,
    kernels: Sequence[str],
    double_column: bool,
    title: str,
    condition_order: Sequence[str] = CONDITION_ORDER,
) -> matplotlib.figure.Figure:
    """The whole small-multiples figure: for each model, a speed-up panel (with the canon column)
    ABOVE a token panel (agents only), sharing the kernel row axis.

    Two panel ROWS rather than laying every model's two panels out side by side: at
    :data:`~hpcagent_bench.stats.style.DOUBLE_COLUMN_WIDTH` three models times two panels each made
    every panel too narrow for its rotated tick labels to stay legible.
    """
    if not panels:
        raise ValueError("no model has a panel to draw (every candidate arm was incomplete)")
    plotstyle.apply()
    n_panels = len(panels)
    # ONE tick set (speed-up) / ONE limit pair (tokens) for every panel of that kind (see
    # style_speedup_x_axis): the union of every drawn value, canon included, across every panel --
    # never one panel's own values, or a position would not mean the same ratio/spend next door.
    all_series = (*((canon_mark,) if canon_mark is not None else ()), *(s for arms in panels.values() for s in arms))
    speedup_ticks = value_ticks(v for series in all_series for v in series.values.values())
    token_limits = token_axis_limits(
        v
        for series in all_series
        for values in (series.tokens, series.tokens_min, series.tokens_max)
        for v in values.values()
    )

    fig, axes = plt.subplots(
        2, n_panels, sharey=True, figsize=figure_size(n_panels, len(kernels), double_column), squeeze=False
    )
    for index, (model, arms) in enumerate(panels.items()):
        speedup_series = (*((canon_mark,) if canon_mark is not None else ()), *arms)
        speedup_ax, token_ax = axes[0][index], axes[1][index]
        style_speedup_x_axis(speedup_ax, speedup_ticks)
        draw_panel(
            speedup_ax,
            kernels,
            speedup_series,
            lambda s: s.values,
            MISSING_MARKER_X,
            summary_speedup,
            "Geomean",
            index == 0,
        )
        speedup_ax.set_title(experiment_tags.model_name(model), fontsize=plotstyle.LABEL_PT * 0.85, color=plotstyle.INK)
        style_token_x_axis(token_ax, token_limits)
        draw_panel(
            token_ax,
            kernels,
            arms,
            lambda s: s.tokens,
            token_limits[0],
            summary_tokens,
            "Median",
            index == 0,
            range_of=lambda s: (s.tokens_min, s.tokens_max),
        )
        if index == 0:
            # ONE label per panel ROW, on the leftmost column only: an axes ylabel per model column
            # would repeat it, and the figure title only names the whole figure -- neither says
            # which row is which metric.
            speedup_ax.set_ylabel(
                "Speed-Up vs Numba", fontsize=plotstyle.LABEL_PT * 0.7, color=plotstyle.MUTED, labelpad=26
            )
            token_ax.set_ylabel("Tokens Spent", fontsize=plotstyle.LABEL_PT * 0.7, color=plotstyle.MUTED, labelpad=26)

    height = figure_size(n_panels, len(kernels), double_column)[1]
    panel_h = panel_rows(len(kernels)) * ROW_HEIGHT_IN
    fig.subplots_adjust(
        left=0.30 / max(n_panels, 1) + 0.03,
        right=0.98,
        top=1.0 - TOP_MARGIN_IN / height,
        bottom=BOTTOM_MARGIN_IN / height,
        hspace=ROW_GAP_IN / panel_h,
        wspace=0.10,
    )
    plotstyle.legend_below(
        fig, legend_handles(canon_mark, panels, condition_order), y=0.005, fontsize=plotstyle.TICK_PT * 0.75
    )
    plotstyle.title(fig, title)
    return fig


#: ``table_rows``' ``status`` column: whether a KERNEL row carries a real speed-up or names a
#: kernel the series covers (roster-complete) but never verified. Blank on a ``summary`` row.
STATUS_VERIFIED: str = "verified"
STATUS_MISSING: str = "no_verified_answer"

#: ``table_rows``' ``row`` column: a per-kernel value, or the series' own overall summary.
ROW_KERNEL: str = "kernel"
ROW_SUMMARY: str = "summary"

#: Documents the per-kernel value rule directly on the written table, since the table is read apart
#: from this module's docstring.
TABLE_NOTE: str = (
    "# speedup: the canon row (if any) is a deterministic column's median_ms ratio; every arm row is "
    "population.kernel_answers' verified answer under --repeats. tokens: population.kernel_tokens "
    "under the same --repeats -- the latest task's own total under reruns (default), the median of "
    "the tasks' totals under designed repeats; tokens_min/tokens_max bracket that median with the "
    "tasks' own minimum and maximum (blank under --repeats latest, where one task IS the value); "
    "canon has none, tokens never summed over tasks. status=no_verified_answer: the series covers "
    "this roster kernel but never verified an answer for it; speedup is blank. row=kernel rows carry "
    "one roster kernel each; row=summary rows (kernel blank) carry one series' OVERALL statistic: "
    "statistic=geomean for speedup (the project rule for an overall speed-up), statistic=median for "
    "tokens; value is that statistic over the n_kernels roster kernels the series had a value for."
)


def as_count(value: float) -> int | float:
    """``value`` as a python ``int`` when it is exactly integral, else the float itself at full
    precision (N4).

    A token total read straight off one task (the latest-run value, or ``tokens_min``/``tokens_max``,
    always a raw task total) is a count and always integral; a ``--repeats median`` kernel's median
    over an even number of tasks need not be (N2's median, not N3's count), so it is left alone
    rather than rounded to fit.
    """
    return int(value) if float(value).is_integer() else value


def series_rows(kind: str, model: str, series: Series, kernels: Sequence[str]) -> list[dict[str, object]]:
    """One ``series``' per-kernel rows over ``kernels``: a real speed-up and token spend where it
    has them, blank otherwise -- so a missing kernel is a readable fact in the table, not a silently
    absent one. ``tokens_min``/``tokens_max`` are blank except under ``--repeats median``."""
    rows: list[dict[str, object]] = []
    for kernel in kernels:
        value = series.values.get(kernel)
        verified = value is not None and math.isfinite(value) and value > 0.0
        tokens = series.tokens.get(kernel)
        has_tokens = tokens is not None and math.isfinite(tokens) and tokens > 0.0
        low = series.tokens_min.get(kernel)
        high = series.tokens_max.get(kernel)
        has_range = has_tokens and low is not None and high is not None and math.isfinite(low) and math.isfinite(high)
        rows.append(
            {
                "kernel": kernel,
                "series": series.key,
                "kind": kind,
                "model": model,
                "condition": series.condition,
                "speedup": value if verified else "",
                "tokens": as_count(tokens) if has_tokens else "",
                "tokens_min": as_count(low) if has_range else "",
                "tokens_max": as_count(high) if has_range else "",
                "status": STATUS_VERIFIED if verified else STATUS_MISSING,
                "row": ROW_KERNEL,
                "statistic": "",
                "value": "",
                "n_kernels": "",
            }
        )
    return rows


def series_summary_rows(kind: str, model: str, series: Series, kernels: Sequence[str]) -> list[dict[str, object]]:
    """``series``' own overall rows, over the SAME roster ``kernels`` the per-kernel rows list:
    geomean speed-up, and median tokens when the series has any (canon does not)."""
    speed_values = [series.values[k] for k in kernels if k in series.values]
    rows: list[dict[str, object]] = []
    speed_point = summary_speedup(speed_values)
    if math.isfinite(speed_point):
        rows.append(
            {
                "kernel": "",
                "series": series.key,
                "kind": kind,
                "model": model,
                "condition": series.condition,
                "speedup": "",
                "tokens": "",
                "tokens_min": "",
                "tokens_max": "",
                "status": "",
                "row": ROW_SUMMARY,
                "statistic": "geomean",
                "value": speed_point,
                "n_kernels": len(speed_values),
            }
        )
    if series.tokens:
        token_values = [series.tokens[k] for k in kernels if k in series.tokens]
        token_point = summary_tokens(token_values)
        if math.isfinite(token_point):
            rows.append(
                {
                    "kernel": "",
                    "series": series.key,
                    "kind": kind,
                    "model": model,
                    "condition": series.condition,
                    "speedup": "",
                    "tokens": "",
                    "tokens_min": "",
                    "tokens_max": "",
                    "status": "",
                    "row": ROW_SUMMARY,
                    "statistic": "median",
                    "value": token_point,
                    "n_kernels": len(token_values),
                }
            )
    return rows


TABLE_COLUMNS: tuple[str, ...] = (
    "kernel",
    "series",
    "kind",
    "model",
    "condition",
    "speedup",
    "tokens",
    "tokens_min",
    "tokens_max",
    "status",
    "row",
    "statistic",
    "value",
    "n_kernels",
)


def table_rows(panels: dict[str, list[Series]], canon_mark: Series | None, kernels: Sequence[str]) -> pd.DataFrame:
    """One row per (series, roster kernel), plus one or two summary rows per series (see
    :func:`series_summary_rows`)."""
    rows: list[dict[str, object]] = []
    if canon_mark is not None:
        rows += series_rows("canon", "", canon_mark, kernels)
        rows += series_summary_rows("canon", "", canon_mark, kernels)
    for model, arms in panels.items():
        for series in arms:
            rows += series_rows("arm", model, series, kernels)
            rows += series_summary_rows("arm", model, series, kernels)
    return pd.DataFrame(rows, columns=list(TABLE_COLUMNS))


def save(fig: matplotlib.figure.Figure, out: pathlib.Path) -> pathlib.Path:
    return plotstyle.save(fig, out.with_suffix(""))
