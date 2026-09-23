# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""llr-focus40: the DaCe canonicalize CPU column against every COMPLETE agent arm, per kernel.

THIS MODULE IS THE FIGURE'S DATA SIDE: which arms, which kernels, which value per kernel, and the
table behind it. Every mark, axis, summary slot and margin is drawn by
:mod:`hpcagent_bench.stats.figures.per_kernel`, the one per-kernel drawing API, through
:func:`speedup_series`/:func:`token_series` -- so this figure's columns, placeholders and summary
read exactly like every other per-kernel figure's.

TWO PANELS SHARING ONE 40-KERNEL X AXIS: a speed-up panel (log2 ratio over the judge's baseline on
Y, the DaCe canon CPU column plus every model's arms) over a token panel (log10 spend on Y, agents
only -- the canon column runs no agent and spends nothing, so it keeps an EMPTY series there, which
leaves every arm at the same dodge offset and summary slot in both panels). The MEASURED quantity is
on Y in both, the kernel NAMES on X, drawn once at the foot of the figure.

ALL MODELS IN ONE PANEL, never a panel column per model. A 40-name kernel axis is this figure's
binding constraint, and a column per model divides the room each name gets by the number of models --
below what a legible label needs at any width a page can print. Colour names the intervention
(:func:`hpcagent_bench.stats.palette.color`: grey control, orange CPF page, blue CPF as source -- the
same packet palette :mod:`statistics.plot_arm_summary` draws with) and SHAPE names the model
(:func:`hpcagent_bench.stats.palette.marker`), so the two channels already separate what a per-model
column would have separated by position. The DaCe canon CPU column keeps the framework colour
(:func:`hpcagent_bench.stats.palette.framework_color`) and its own fixed shape (:data:`CANON_MARKER`).

NO EFFICACY, PARETO OR COST-VS-SPEED FRAMING HERE. This figure reports what each arm cost and what
it bought, side by side, and leaves any tradeoff reading to the caption -- never a quadrant, a
frontier or the word "efficacy" in the axes themselves.

CONDITION COMES FROM THE ARM NAME, not the ``language``/``packet`` columns: the pre-regrade
extraction records them inconsistently for the SAME arm (some rows ``language=c, packet=''``,
others ``language='', packet='cpf'``), where the arm name itself is the one column every row of an
arm agrees on. :data:`ARM_PATTERN` is both the arm selector and the (model, condition) parser.

COMPLETENESS is roster coverage, not scoring: :func:`hpcagent_bench.stats.population.complete_arms`
keeps only arms with a recorded row for every kernel :data:`ARM_PATTERN` -- and a model whose arms
are ALL incomplete draws nothing at all, rather than an empty slot.

PER-KERNEL VALUES ARE WHATEVER THE FRAMEWORK'S OWN POLICY ASSIGNS UNDER ``--repeats`` -- never
invented here (spec R3-R7). :func:`hpcagent_bench.stats.population.kernel_answers` is an arm's
verified answer per kernel: the LATEST run's answer for a rerun kernel (default, none when that run
verified nothing -- an earlier run's answer never stands in), the MEDIAN answer over runs that
repeat by design. A kernel with no verified answer has no value here; the speed-up panel enters it
at 1x, hollow and crossed, and the table names it ``status=no_verified_answer``. Tokens are
:func:`hpcagent_bench.stats.population.kernel_tokens` under the SAME ``--repeats``: the latest
run's own TASK TOKEN TOTAL, or the median of the runs' task totals, bracketed by that median's own
minimum and maximum over the tasks (``SeriesValues.tokens_min``/``tokens_max``, drawn as the
kernel's interval). Tokens are NEVER summed over tasks (R6), and come only from ``record = task``
rows (T4). The canon column is :func:`hpcagent_bench.stats.canon.kernel_speedups` over the
deterministic sweep: no episodes, no policy to pick, and no tokens spent.

EACH PANEL'S SUMMARY is per_kernel's: one slot per series past a dashed separator, the geometric
mean with its 95% interval for speed-up (the project-wide rule for an overall speed-up, never a
median) and the median for tokens (not a ratio), over the kernels the series solved -- the same
statistics, over the same cells, the table's ``row=summary`` rows carry (:func:`series_summary_rows`).
"""

import dataclasses
import math
import re
from collections.abc import Sequence

import matplotlib.artist
import matplotlib.figure
import matplotlib.lines
import matplotlib.patches
import numpy as np
import pandas as pd

from hpcagent_bench import experiment_tags, packets
from hpcagent_bench.stats import canon, palette, population
from hpcagent_bench.stats import style as plotstyle
from hpcagent_bench.stats.figures import per_kernel
from hpcagent_bench.stats.figures.results import DEFAULT_BASELINE

#: An arm this figure may draw, and its (model, condition) in one match: ``-c`` is the control
#: (condition ``""``), ``-c-cpf`` the CPF page, ``-c-cpfsrc`` CPF as source. C only -- Fortran has no
#: CPF spelling (mpr-artifacts/experiments/llr-focus40-cpf/README.md).
ARM_PATTERN: re.Pattern[str] = re.compile(r"^cpf-llr-focus40-(?P<model>[a-z0-9]+)-c(?:-(?P<condition>cpf|cpfsrc))?$")

#: Draw order within one model's own slot, control first.
CONDITION_ORDER: tuple[str, ...] = ("", "cpf", "cpfsrc")

#: The canon column this figure draws by default, and what it is measured against.
CANON_COLUMN: str = "dace_cpu_canonicalize"
CANON_BASELINE: str = "numba"

#: The canon series' fixed marker -- it has no model, so it never borrows one of theirs.
CANON_MARKER: str = "D"


@dataclasses.dataclass(frozen=True, slots=True)
class SeriesValues:
    """One series' per-kernel speed-up and token spend, keyed by the kernels it has a value for:
    the record the table is written from, and what :func:`speedup_series`/:func:`token_series` hand
    to :mod:`~hpcagent_bench.stats.figures.per_kernel` to draw.

    The canon series has an empty ``tokens`` -- it runs no agent. ``tokens_min``/``tokens_max``
    bracket a ``--repeats median`` kernel's token value with the minimum and maximum over its tasks
    (R5); empty under ``--repeats latest``, where one task IS the value and there is nothing to
    bracket. ``delivered`` names the PRESENT speed-ups that are placeholders all the same (a ratio
    one side of which never delivered, ``statistics/plot_repo_vs_kernel.py``); empty means every
    present value is a measurement.
    """

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
    delivered: dict[str, bool] = dataclasses.field(default_factory=dict)


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

    A kernel with no verified answer is absent from this dict, and :func:`speedup_series` enters it
    at :data:`~hpcagent_bench.stats.population.NOT_DELIVERED` as an undelivered cell, so one place
    decides how a non-delivery is drawn. ``policy="solved"`` is therefore explicit: the served
    policy would hand back the same placeholder a second time and the two would compete.
    """
    subset = frame[frame["arm"].astype(str) == arm]
    answers = population.kernel_answers(subset, repeats=repeats, policy="solved")
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
    :mod:`statistics.plot_arm_summary`'s ``condition_order`` already use for model and packet axes.
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
) -> tuple[dict[str, list[SeriesValues]], SeriesValues | None, dict[str, int]]:
    """model -> its arm series (condition order), the deterministic reference series shared by
    every panel (``None`` when the caller has no such column -- llr-focus40-cpf's DaCe canon,
    git-scicomp's has none), and the dropped arms with how many roster kernels each covered.

    A model every one of whose candidate arms was dropped for incomplete coverage gets no key at
    all -- :func:`figure` therefore draws it nothing, per the figure's own contract.
    """
    candidates = candidate_arms(observations, pattern)
    frame = observations[observations["arm"].astype(str).isin(candidates)]
    if include_incomplete:
        kept, dropped = list(candidates), {}
    else:
        kept, dropped = population.complete_arms(frame, roster)
    canon_mark = (
        SeriesValues(
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
    by_model: dict[str, list[SeriesValues]] = {}
    for arm in kept:
        model, condition = candidates[arm]
        values = arm_speedups(frame, arm, repeats)
        if not values:
            continue
        tokens, tokens_min, tokens_max = arm_tokens(frame, arm, repeats)
        series = SeriesValues(
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


def speedup_series(values: SeriesValues, kernels: Sequence[str]) -> per_kernel.Series:
    """``values``' speed-ups as a drawn series over ``kernels``. A kernel with no verified answer
    enters at 1x, hollow and crossed (:func:`~hpcagent_bench.stats.figures.per_kernel.kernel_cells`):
    that is what a served but unsolved kernel leaves standing under every scoring policy this repo
    has, and a real 1.0x and a kernel nobody answered must not draw as one mark."""
    cells = per_kernel.kernel_cells(values.values, kernels, delivered=values.delivered)
    return per_kernel.Series(values.label, cells, values.color, values.marker)


def token_series(values: SeriesValues, kernels: Sequence[str]) -> per_kernel.Series:
    """``values``' token spend as a drawn series over ``kernels``. A kernel with no task total draws
    nothing -- a missing total is no measurement (spec R7), and a mark at the axis edge would read as
    the smallest spend -- and a ``--repeats median`` kernel carries its tasks' minimum and maximum as
    its interval (R5)."""
    cells = per_kernel.kernel_cells(values.tokens, kernels, fill=False, low=values.tokens_min, high=values.tokens_max)
    return per_kernel.Series(values.label, cells, values.color, values.marker)


def legend_handles(
    canon_mark: SeriesValues | None,
    panels: dict[str, list[SeriesValues]],
    condition_order: Sequence[str] = CONDITION_ORDER,
) -> list[matplotlib.artist.Artist]:
    """The figure's identity key: the optional reference mark, each condition present (colour), each
    model present (shape) -- the same two channels :mod:`statistics.plot_arm_summary` draws with.
    The status marks' entries come from :func:`per_kernel.status_handles`, only when drawn."""
    conditions = sorted(
        {series.condition for series_list in panels.values() for series in series_list},
        key=lambda condition: rank_condition(condition, condition_order),
    )
    handles: list[matplotlib.artist.Artist] = []
    if canon_mark is not None:
        handles.append(
            matplotlib.lines.Line2D(
                [],
                [],
                marker=canon_mark.marker,
                linestyle="none",
                color=canon_mark.color,
                markersize=per_kernel.LEGEND_MARK_PT,
                label=canon_mark.label,
            )
        )
    for condition in conditions:
        colour = palette.color(condition)
        handles.append(matplotlib.patches.Patch(facecolor=colour, edgecolor=colour, label=condition_label(condition)))
    for model in panels:
        handles.append(
            matplotlib.lines.Line2D(
                [],
                [],
                marker=palette.marker(model),
                linestyle="none",
                color=plotstyle.MUTED,
                markersize=per_kernel.LEGEND_MARK_PT,
                label=experiment_tags.model_name(model),
            )
        )
    return handles


def speedup_label(baseline: str) -> str:
    """The speed-up panel's axis label, naming the denominator the JUDGE recorded.

    The baseline is a property of the data (:func:`hpcagent_bench.stats.figures.results.baseline_of`
    reads the column the judge stamped), never of the figure: llr-focus40 is graded against numba
    and scientific_computing against c-autopar, so a fixed "vs Numba" here labels a
    scientific_computing panel with a denominator no score in it ever saw. The canon series already
    names its own denominator the same way (:func:`build_panels`).
    """
    return f"Speed-Up vs {baseline}"


def two_panel_figure(
    drawn: Sequence[SeriesValues],
    kernels: Sequence[str],
    double_column: bool,
    title: str,
    speedup_ylabel: str,
    token_ylabel: str,
    legend: Sequence[matplotlib.artist.Artist],
) -> matplotlib.figure.Figure:
    """Speed-up over tokens for ``drawn`` on one kernel axis
    (:func:`~hpcagent_bench.stats.figures.per_kernel.figure_panels`): a page insert at
    :data:`~hpcagent_bench.stats.style.DOUBLE_COLUMN_WIDTH` under ``double_column``, otherwise as
    wide as the dodged marks want (:func:`~hpcagent_bench.stats.figures.per_kernel.roomy_pitch_in`).
    ``legend`` is the identity key; the status entries are appended for the marks actually drawn."""
    speed = per_kernel.speedup_series_metric([speedup_series(values, kernels) for values in drawn], speedup_ylabel)
    tokens = per_kernel.token_series_metric([token_series(values, kernels) for values in drawn], token_ylabel)
    return per_kernel.figure_panels(
        [speed, tokens],
        kernels,
        "ci",
        True,
        title,
        legend=[*legend, *per_kernel.status_handles([speed, tokens])],
        pitch_in=None if double_column else per_kernel.roomy_pitch_in(len(drawn)),
    )


def figure(
    panels: dict[str, list[SeriesValues]],
    canon_mark: SeriesValues | None,
    kernels: Sequence[str],
    double_column: bool,
    title: str,
    condition_order: Sequence[str] = CONDITION_ORDER,
    baseline: str = DEFAULT_BASELINE,
) -> matplotlib.figure.Figure:
    """The whole figure: a speed-up panel (every model's arms plus the canon column) ABOVE a token
    panel (agents only), both on ONE shared kernel axis.

    Every model in ONE panel rather than a panel column each: 40 rotated kernel names are what the
    width has to buy, and a column per model would divide the room each name gets by the number of
    models (see the module docstring). ``panels`` keeps its per-model shape -- it decides the draw
    ORDER, so a model's arms stay together inside a kernel column, and it is what the table and the
    legend are built from.
    """
    if not panels:
        raise ValueError("no model has a panel to draw (every candidate arm was incomplete)")
    plotstyle.apply()
    drawn = [*((canon_mark,) if canon_mark is not None else ()), *(s for arms in panels.values() for s in arms)]
    legend = legend_handles(canon_mark, panels, condition_order)
    return two_panel_figure(drawn, kernels, double_column, title, speedup_label(baseline), "Tokens Spent", legend)


# ---------------------------------------------------------------------------------------------
# llr40_model_figure -- the SAME 40-kernel dodge, coloured by MODEL instead of packet: one point
# per (kernel, model) under a single packet SELECTION for the whole figure (the control arm, one
# named packet, or the per-kernel median over every packet arm), rather than build_panels' one
# point per (kernel, model, packet). No canon column, no title (a paper caption carries it).
# ---------------------------------------------------------------------------------------------

#: The llr-focus40 skill packets this figure may select over, control first -- registry keys
#: (:func:`hpcagent_bench.experiment_tags.order`), not their arm-name spelling (:data:`LLR40_ARM_SUFFIX`).
LLR40_PACKETS: tuple[str, ...] = ("", "lang-skills", "perf-playbook-cpu", "cpf", "cpfsrc")

#: One packet's arm-name suffix in the llr-focus40-cpf campaign -- the token after ``-c``/``-fortran``,
#: never derived from :func:`~hpcagent_bench.experiment_tags.packet_spellings` because that table's
#: spelling for ``lang-skills`` ("skills") is shorter than the registry key itself.
LLR40_ARM_SUFFIX: dict[str, str] = {
    "": "",
    "lang-skills": "skills",
    "perf-playbook-cpu": "perf-playbook-cpu",
    "cpf": "cpf",
    "cpfsrc": "cpfsrc",
}

#: ``packet_mode`` values :func:`llr40_model_value` treats as "aggregate", not one arm.
LLR40_MEDIAN_MODE: str = "median"

#: The dodged mark's shape when no single packet names it (the control, or a "median" aggregate
#: over every packet) -- a plain circle, never a registered packet's own shape, so it cannot be
#: misread as that one packet.
LLR40_DEFAULT_MARKER: str = "o"


def llr40_arm(model: str, packet_mode: str, language: str = "c") -> str:
    """The llr-focus40-cpf arm name for (``model``, one packet of :data:`LLR40_PACKETS`);
    ``packet_mode=""`` is the bare control."""
    suffix = LLR40_ARM_SUFFIX[packet_mode]
    return f"cpf-llr-focus40-{model}-{language}" + (f"-{suffix}" if suffix else "")


def median_over_packets(per_packet: Sequence[dict[str, float]]) -> dict[str, float]:
    """The per-kernel MEDIAN over whichever packet arms have a value there -- never an average of
    averages: a kernel only one packet arm answered still gets that one value, not a value diluted
    by the packets that never verified it."""
    kernels = {kernel for one in per_packet for kernel in one}
    return {kernel: float(np.median([one[kernel] for one in per_packet if kernel in one])) for kernel in kernels}


def llr40_model_value(
    frame: pd.DataFrame, model: str, packet_mode: str, repeats: population.RepeatPolicy, language: str = "c"
) -> tuple[dict[str, float], dict[str, float]]:
    """``(speed-up, billed tokens)`` per kernel for one model under one packet selection.

    ``packet_mode`` is one of :data:`LLR40_PACKETS` (the bare control is ``""``) or
    :data:`LLR40_MEDIAN_MODE`, the per-kernel median over every packet arm that has a value there.
    ``frame`` must already be priced by the caller's chosen cost card (:func:`hpcagent_bench.stats.
    cost.priced`) -- this reads whatever ``tokens`` column that left behind, the same contract
    :func:`arm_tokens` already has.
    """
    if packet_mode == LLR40_MEDIAN_MODE:
        per_packet_speed = [arm_speedups(frame, llr40_arm(model, p, language), repeats) for p in LLR40_PACKETS]
        per_packet_tokens = [arm_tokens(frame, llr40_arm(model, p, language), repeats)[0] for p in LLR40_PACKETS]
        return median_over_packets(per_packet_speed), median_over_packets(per_packet_tokens)
    arm = llr40_arm(model, packet_mode, language)
    return arm_speedups(frame, arm, repeats), arm_tokens(frame, arm, repeats)[0]


def llr40_model_marker(packet_mode: str) -> str:
    """The one SHAPE every dodged mark wears: the packet's own registered shape
    (:func:`~hpcagent_bench.stats.palette.packet_marker`) for a single named packet,
    :data:`LLR40_DEFAULT_MARKER` for the control or the "median" aggregate, neither of which is a
    packet a reader could mistake this for."""
    if packet_mode in ("", LLR40_MEDIAN_MODE):
        return LLR40_DEFAULT_MARKER
    return palette.packet_marker(packet_mode)


def llr40_model_panels(
    frame: pd.DataFrame,
    models: Sequence[str],
    packet_mode: str,
    repeats: population.RepeatPolicy,
    language: str = "c",
) -> dict[str, list[SeriesValues]]:
    """One :class:`SeriesValues` per model -- COLOUR is the model (:func:`~hpcagent_bench.stats.palette.
    model_color`, the settled rule this figure's own row shares with :mod:`hpcagent_bench.stats.
    figures.efficacy`), SHAPE the packet selection (:func:`llr40_model_marker`): a model with no
    value under ``packet_mode`` (every packet arm incomplete or unverified) is silently absent, the
    same contract :func:`build_panels` already has."""
    marker_shape = llr40_model_marker(packet_mode)
    panels: dict[str, list[SeriesValues]] = {}
    for model in models:
        speed, tokens = llr40_model_value(frame, model, packet_mode, repeats, language)
        if not speed:
            continue
        panels[model] = [
            SeriesValues(
                model,
                experiment_tags.model_name(model),
                palette.model_color(model),
                marker_shape,
                model,
                packet_mode,
                speed,
                tokens,
                {},
                {},
            )
        ]
    return panels


def llr40_legend_handles(models: Sequence[str], packet_mode: str) -> list[matplotlib.artist.Artist]:
    """One legend entry per drawn model (colour) in the packet selection's own shape -- the
    colour-is-model mirror of :func:`legend_handles`."""
    marker_shape = llr40_model_marker(packet_mode)
    return [
        matplotlib.lines.Line2D(
            [],
            [],
            marker=marker_shape,
            linestyle="none",
            color=palette.model_color(model),
            markersize=per_kernel.LEGEND_MARK_PT,
            label=experiment_tags.model_name(model),
        )
        for model in palette.in_order(models)
    ]


def llr40_model_figure(
    frame: pd.DataFrame,
    roster: Sequence[str],
    models: Sequence[str],
    packet_mode: str = "",
    repeats: population.RepeatPolicy = "latest",
    language: str = "c",
    double_column: bool = True,
    baseline: str = DEFAULT_BASELINE,
) -> matplotlib.figure.Figure:
    """LLR40 by MODEL: one dodged, coloured-by-model mark per kernel (:func:`llr40_model_panels`),
    speed-up over billed token spend, on the SAME kernel axis :func:`figure` draws by packet -- the
    per-kernel twin of :mod:`hpcagent_bench.stats.figures.efficacy`'s geomean summary row. NO title:
    a paper caption carries it. The key sits under the kernel names in a band measured from them
    (:func:`per_kernel.fit_canvas`), so a long rotated name never runs through it.
    """
    panels = llr40_model_panels(frame, models, packet_mode, repeats, language)
    if not panels:
        raise ValueError(f"no model of {list(models)} has a value under packet_mode={packet_mode!r}")
    plotstyle.apply()
    drawn = [series for model_arms in panels.values() for series in model_arms]
    return two_panel_figure(
        drawn,
        sorted(roster),
        double_column,
        "",
        speedup_label(baseline),
        "Billed Tokens (1 input + 0.1 cached input + 1 output)",
        llr40_legend_handles(list(panels), packet_mode),
    )


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
    "one roster kernel each; row=summary rows (kernel blank) carry one series' OVERALL statistic, "
    "the one the figure's summary slot draws: statistic=geomean for speedup (the project rule for an "
    "overall speed-up) with its 95% log-t interval low/high, over the n_kernels roster kernels the "
    "series solved; statistic=median for tokens with its bootstrap interval (blank under 5 kernels), "
    "over the n_kernels roster kernels the series has a task total for."
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


def series_rows(kind: str, model: str, series: SeriesValues, kernels: Sequence[str]) -> list[dict[str, object]]:
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
                "low": "",
                "high": "",
                "n_kernels": "",
            }
        )
    return rows


def series_summary_rows(kind: str, model: str, series: SeriesValues, kernels: Sequence[str]) -> list[dict[str, object]]:
    """``series``' own overall rows, over the SAME roster ``kernels`` the per-kernel rows list and
    from the SAME cells the figure draws (:func:`speedup_series`, :func:`token_series`), reduced by
    the figure's own reducers -- so a summary slot and its row cannot disagree: the geomean speed-up
    over the solved kernels, and the median tokens when the series spends any (canon does not)."""
    rows: list[dict[str, object]] = []
    reductions = (
        ("geomean", speedup_series(series, kernels).cells, per_kernel.summary_point_speedup),
        ("median", token_series(series, kernels).cells, per_kernel.summary_point_tokens),
    )
    for statistic, cells, reducer in reductions:
        point, low, high = reducer(cells)
        if not math.isfinite(point):
            continue
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
                "statistic": statistic,
                "value": point,
                "low": low if math.isfinite(low) else "",
                "high": high if math.isfinite(high) else "",
                "n_kernels": len(per_kernel.kernel_medians(cells)),
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
    "low",
    "high",
    "n_kernels",
)


def table_rows(
    panels: dict[str, list[SeriesValues]], canon_mark: SeriesValues | None, kernels: Sequence[str]
) -> pd.DataFrame:
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
