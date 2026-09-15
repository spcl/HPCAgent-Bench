# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""llr-focus40: the DaCe canonicalize CPU column against every COMPLETE agent arm, per kernel.

One small-multiple panel per model, sharing the 40-kernel row axis (:func:`hpcagent_bench.stats.style.row_axis`):
each row is a kernel, each panel's x axis is the log2 speed-up over Numba
(:func:`hpcagent_bench.stats.figures.per_kernel.speedup_tick_label`, the same ticks and 1x reference
line the per-kernel figure draws), and every panel repeats the DaCe canon CPU column (the framework
colour, :func:`hpcagent_bench.stats.palette.framework_color`) beside that model's own arms (the
condition colour, :func:`hpcagent_bench.stats.palette.color`: grey control, orange CPF page, blue
CPF as source -- the same packet palette :mod:`scripts.plot_arm_summary` draws with).

CONDITION COMES FROM THE ARM NAME, not the ``language``/``packet`` columns: the pre-regrade
extraction records them inconsistently for the SAME arm (some rows ``language=c, packet=''``,
others ``language='', packet='cpf'``), where the arm name itself is the one column every row of an
arm agrees on. :data:`ARM_PATTERN` is both the arm selector and the (model, condition) parser.

COMPLETENESS is roster coverage, not scoring: :func:`hpcagent_bench.stats.population.complete_arms`
keeps only arms with a recorded row for every kernel :data:`ARM_PATTERN` -- and a model whose arms
are ALL incomplete draws no panel at all, rather than an empty one.

PER-KERNEL VALUES ARE WHATEVER THE FRAMEWORK'S OWN POLICY ASSIGNS -- never invented here.
:func:`hpcagent_bench.stats.population.kernel_answers` is an arm's best verified FINAL answer per
kernel (within an episode the last submission, across episodes the maximum); a kernel with no
verified answer simply has no row and draws no mark. The canon column is
:func:`hpcagent_bench.stats.canon.kernel_speedups` over the deterministic sweep: no episodes, no
policy to pick.
"""

import dataclasses
import math
import pathlib
import re
from collections.abc import Iterable, Sequence

import matplotlib.artist
import matplotlib.axes
import matplotlib.figure
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hpcagent_bench import experiment_tags, packets
from hpcagent_bench.stats import canon, palette, population
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
CHROME_IN: float = 1.9

#: The kernel row labels' own font size -- smaller than :data:`hpcagent_bench.stats.style.LABEL_PT`,
#: which :func:`row_axis` applies but which a 40-row axis has no vertical room for.
ROW_LABEL_PT: float = 8.5

#: Where a "no verified answer" mark sits -- the 1x reference line, since that is what a served but
#: unsolved kernel leaves standing under every scoring policy this repo has (:data:`~hpcagent_bench.stats.population.NOT_DELIVERED`).
#: HOLLOW, never filled: a real 1.0x speed-up and "nothing to plot here" must not draw as one mark.
MISSING_MARKER_X: float = 1.0

#: The shared legend entry for a missing-answer mark, neutral ink since it names a STATUS, not one
#: series' identity -- a coloured entry would read as one more condition or model.
MISSING_LABEL: str = "No Verified Answer"


@dataclasses.dataclass(frozen=True, slots=True)
class Series:
    """One drawn series: a per-kernel speed-up, already keyed to the kernels it has a value for."""

    key: str
    label: str
    color: str
    marker: str
    model: str
    condition: str
    values: dict[str, float]


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
        series = Series(
            arm, condition_label(condition), palette.color(condition), palette.marker(model), model, condition, values
        )
        by_model.setdefault(model, []).append(series)
    for model, series_list in by_model.items():
        series_list.sort(key=lambda series: rank_condition(series.condition, condition_order))
    ordered = [model for model in palette.in_order(by_model.keys(), "models") if model in by_model]
    panels = {model: by_model[model] for model in ordered}
    return panels, canon_mark, dropped


def value_ticks(values: Iterable[float]) -> list[float]:
    """Powers of two spanning every plotted value, always at least ``1/4x .. 4x`` -- an X-axis twin
    of :func:`hpcagent_bench.stats.figures.per_kernel.speedup_yticks`."""
    finite = [v for v in values if math.isfinite(v) and v > 0]
    low, high = (min(finite), max(finite)) if finite else (1.0, 1.0)
    low_exp = min(-2, math.floor(math.log2(low)))
    high_exp = max(2, math.ceil(math.log2(high)))
    return [2.0**exp for exp in range(low_exp, high_exp + 1)]


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


def draw_panel(
    ax: matplotlib.axes.Axes,
    kernels: Sequence[str],
    canon_mark: Series | None,
    arms: Sequence[Series],
    ticks: Sequence[float],
    label_rows: bool,
) -> None:
    """One model's panel: the optional reference mark plus its arms, dodged apart within each row.

    A kernel a series has no value for still draws: a HOLLOW mark in that series' own colour and
    shape, at the 1x line (:data:`MISSING_MARKER_X`) -- present and legible rather than a gap a
    reader has to notice on their own, and hollow so it is never mistaken for a genuine 1.0x answer.
    """
    series_list = (*((canon_mark,) if canon_mark is not None else ()), *arms)
    style_speedup_x_axis(ax, ticks)
    plotstyle.row_axis(ax, kernels)
    if label_rows:
        ax.tick_params(axis="y", labelsize=ROW_LABEL_PT)
    else:
        ax.tick_params(axis="y", labelleft=False)
    n = len(series_list)
    offsets = np.linspace(-0.3, 0.3, n) if n > 1 else np.array([0.0])
    y_of = {kernel: i for i, kernel in enumerate(kernels)}
    for offset, series in zip(offsets, series_list, strict=True):
        for kernel in kernels:
            y = y_of[kernel] + offset
            value = series.values.get(kernel)
            if value is None or not math.isfinite(value) or value <= 0.0:
                plotstyle.point_mark(ax, MISSING_MARKER_X, y, series.color, series.marker, filled=False, size=26.0)
            else:
                plotstyle.point_mark(ax, value, y, series.color, series.marker, filled=True, size=26.0)


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


def figure_size(n_panels: int, n_rows: int, double_column: bool) -> tuple[float, float]:
    """A compact double-column insert (fixed width) or a standalone report (one width slot per panel)."""
    height = n_rows * ROW_HEIGHT_IN + CHROME_IN
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
    """The whole small-multiples figure: one panel per model, sharing the kernel row axis."""
    if not panels:
        raise ValueError("no model has a panel to draw (every candidate arm was incomplete)")
    plotstyle.apply()
    n_panels = len(panels)
    # ONE tick set for every panel (see style_speedup_x_axis): the union of every drawn value,
    # canon included, across every panel -- never one panel's own values, or a position would not
    # mean the same ratio next door.
    all_series = (*((canon_mark,) if canon_mark is not None else ()), *(s for arms in panels.values() for s in arms))
    ticks = value_ticks(v for series in all_series for v in series.values.values())
    fig, axes = plt.subplots(
        1, n_panels, sharey=True, figsize=figure_size(n_panels, len(kernels), double_column), squeeze=False
    )
    for index, (model, arms) in enumerate(panels.items()):
        ax = axes[0][index]
        draw_panel(ax, kernels, canon_mark, arms, ticks, label_rows=index == 0)
        ax.set_title(experiment_tags.model_name(model), fontsize=plotstyle.LABEL_PT * 0.85, color=plotstyle.INK)
    # Top/bottom margins in INCHES, not a fixed fraction: the rotated x ticks and the legend below
    # need roughly the same number of inches at any row count, and a fixed fraction of a figure
    # that grows with the roster (:data:`ROW_HEIGHT_IN` per kernel) leaves a growing dead strip.
    height = figure_size(n_panels, len(kernels), double_column)[1]
    fig.subplots_adjust(
        left=0.22 / max(n_panels, 1) + 0.02, right=0.99, top=1.0 - 1.0 / height, bottom=1.1 / height, wspace=0.08
    )
    plotstyle.legend_below(
        fig, legend_handles(canon_mark, panels, condition_order), y=0.01, fontsize=plotstyle.TICK_PT * 0.75
    )
    plotstyle.title(fig, title)
    return fig


#: ``table_rows``' ``status`` column: whether a row carries a real speed-up or names a kernel the
#: series covers (roster-complete) but never verified.
STATUS_VERIFIED: str = "verified"
STATUS_MISSING: str = "no_verified_answer"

#: Documents the per-kernel value rule directly on the written table, since the table is read apart
#: from this module's docstring.
TABLE_NOTE: str = (
    "# speedup: the canon row (if any) is a deterministic column's median_ms ratio; every arm row is "
    "population.kernel_answers' best verified final answer. status=no_verified_answer: the series "
    "covers this roster kernel but never verified an answer for it; speedup is blank."
)


def series_rows(kind: str, model: str, series: Series, kernels: Sequence[str]) -> list[dict[str, object]]:
    """One ``series``' rows over ``kernels``: a real speed-up where it has one, a blank
    ``no_verified_answer`` row otherwise -- so a missing kernel is a readable fact in the table, not
    a silently absent one."""
    rows: list[dict[str, object]] = []
    for kernel in kernels:
        value = series.values.get(kernel)
        verified = value is not None and math.isfinite(value) and value > 0.0
        rows.append(
            {
                "kernel": kernel,
                "series": series.key,
                "kind": kind,
                "model": model,
                "condition": series.condition,
                "speedup": value if verified else "",
                "status": STATUS_VERIFIED if verified else STATUS_MISSING,
            }
        )
    return rows


def table_rows(panels: dict[str, list[Series]], canon_mark: Series | None, kernels: Sequence[str]) -> pd.DataFrame:
    """One row per (series, roster kernel): a real speed-up, or a ``no_verified_answer`` row."""
    rows: list[dict[str, object]] = []
    if canon_mark is not None:
        rows += series_rows("canon", "", canon_mark, kernels)
    for model, arms in panels.items():
        for series in arms:
            rows += series_rows("arm", model, series, kernels)
    return pd.DataFrame(rows, columns=["kernel", "series", "kind", "model", "condition", "speedup", "status"])


def save(fig: matplotlib.figure.Figure, out: pathlib.Path) -> pathlib.Path:
    return plotstyle.save(fig, out.with_suffix(""))
