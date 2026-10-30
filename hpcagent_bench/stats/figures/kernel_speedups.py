# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Speedup per kernel: canon-sweep columns (compilers) and agent setups on one log2 kernel axis
(:func:`kernel_figure`, :func:`kernel_comparison`), following SC15 rules 4, 5/7 and 12
(:mod:`hpcagent_bench.stats.rules`)."""

import dataclasses
import math
import pathlib
import re
from collections.abc import Collection, Mapping, Sequence

import matplotlib.figure
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs] -- pandas ships none
from matplotlib.artist import Artist
from matplotlib.lines import Line2D

from hpcagent_bench import packets, study_tags
from hpcagent_bench.harness import denominator
from hpcagent_bench.results.observation_columns import RowKind
from hpcagent_bench.stats import canon, palette, population, rules, style
from hpcagent_bench.stats.figures import per_kernel, setup_names
from hpcagent_bench.stats.summary import geomean_ci, signed_change, usable_ratios

__all__ = [
    "BASELINE",
    "DEAD_BAND",
    "PANEL_HEIGHT_IN",
    "SUMMARY_COLUMNS",
    "TABLE_COLUMNS",
    "TOKEN_SUMMARY_COLUMNS",
    "Row",
    "agent_kernel_row",
    "answer_ratios",
    "baseline_color",
    "baseline_label",
    "baseline_references",
    "canon_kernel_row",
    "canon_label",
    "distinct_canon_labels",
    "fallback_note",
    "kernel_comparison",
    "kernel_figure",
    "kernel_intervals",
    "kernel_metrics",
    "kernel_rows",
    "legend_handles",
    "pending_note",
    "row_color",
    "sign_test",
    "solved_ratios",
    "summary_table",
    "table",
    "token_summary_table",
    "unattempted_kernels",
    "write_tables",
]


#: Dead band of the sign test. Below 1% the two setups are the same code and the difference is jitter.
DEAD_BAND: float = 1.01


@dataclasses.dataclass(frozen=True, slots=True)
class Row:
    """One drawn row: its ratios per kernel, the costs behind them, and what it excluded.

    ``color``/``marker`` and the trailing five fields are set only by the per-kernel
    figure rows (:func:`canon_kernel_row`, :func:`agent_kernel_row`); plain TSVC rows leave them
    at their defaults.
    """

    framework: str
    label: str
    ratios: dict[str, float]
    numerator_ms: dict[str, float]
    denominator_ms: dict[str, float]
    excluded: str
    color: str | None = None
    marker: study_tags.Marker = "o"
    ratios_low: dict[str, float] = dataclasses.field(default_factory=dict)  # per-kernel CI, SC15 5/7
    ratios_high: dict[str, float] = dataclasses.field(default_factory=dict)
    tokens: dict[str, float] = dataclasses.field(default_factory=dict)  # per-kernel token spend
    delivered: dict[str, bool] = dataclasses.field(default_factory=dict)  # ratio vs 1x placeholder
    pending: frozenset[str] = frozenset()  # kernels not attempted yet (mark_pending only)
    filled: bool = True  # hollow = the no-packet control (palette.CONTROL_MARKER)


def sign_test(ratios: Mapping[str, float]) -> tuple[int, int]:
    """Kernels the numerator wins and loses, outside the :data:`DEAD_BAND`."""
    wins = sum(1 for v in ratios.values() if v > DEAD_BAND)
    losses = sum(1 for v in ratios.values() if v < 1.0 / DEAD_BAND)
    return wins, losses


def table(rows: Sequence[Row]) -> pd.DataFrame:
    """The figure's data table: one record per (row, kernel), ratio and both costs (SC15 rule 4)."""
    records = [
        {
            "row": row.label,
            "framework": row.framework,
            "kernel": kernel,
            "speedup": ratio,
            "numerator_ms": row.numerator_ms[kernel],
            "denominator_ms": row.denominator_ms[kernel],
            "signed_change": signed_change(ratio),
        }
        for row in rows
        for kernel, ratio in sorted(row.ratios.items())
    ]
    frame = pd.DataFrame.from_records(records, columns=TABLE_COLUMNS)
    return rules.require_costs(frame, "speedup", ("numerator_ms", "denominator_ms"))


#: Column order of the emitted data table, so a diff between two runs compares like with like.
TABLE_COLUMNS: tuple[str, ...] = (
    "row",
    "framework",
    "kernel",
    "speedup",
    "numerator_ms",
    "denominator_ms",
    "signed_change",
)


def solved_ratios(row: Row) -> dict[str, float]:
    """``row``'s ratios over the kernels it solved: excludes 1x placeholders (``delivered`` False).

    A row with no ``delivered`` flags (every TSVC or agent row) solved every kernel it has a ratio
    for.
    """
    return {kernel: ratio for kernel, ratio in row.ratios.items() if row.delivered.get(kernel, True)}


def summary_table(rows: Sequence[Row]) -> pd.DataFrame:
    """One record per drawn row, over the kernels it solved (:func:`solved_ratios`): geomean,
    interval, median, n and the sign test.

    Gated by :func:`hpcagent_bench.stats.rules.require_interval` (SC15 rules 5/7).
    """
    records: list[dict[str, object]] = []
    for row in rows:
        solved = solved_ratios(row)
        values = usable_ratios(list(solved.values()), label=row.label)
        if values.size == 0:
            records.append({"row": row.label, "framework": row.framework, "n": 0, "excluded": row.excluded})
            continue
        interval = geomean_ci(values)
        wins, losses = sign_test(solved)
        records.append(
            {
                "row": row.label,
                "framework": row.framework,
                "n": interval.n,
                "geomean": interval.point,
                "geomean_low": interval.low,
                "geomean_high": interval.high,
                "median": float(np.median(values)),
                "wins": wins,
                "losses": losses,
                "excluded": row.excluded,
            }
        )
    frame = pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS)
    return rules.require_interval(frame, "geomean", "geomean_low", "geomean_high")


#: Column order of the emitted summary table.
SUMMARY_COLUMNS: tuple[str, ...] = (
    "row",
    "framework",
    "n",
    "geomean",
    "geomean_low",
    "geomean_high",
    "median",
    "wins",
    "losses",
    "excluded",
)


#: The speedup denominator when none is named: every track times its kernels against Numba.
BASELINE: str = "numba"

#: Column order of the emitted token summary table.
TOKEN_SUMMARY_COLUMNS: tuple[str, ...] = (
    "row", "framework", "n", "gm_tokens", "gm_tokens_low", "gm_tokens_high",
)  # fmt: skip


def canon_kernel_row(
    canon_frame: pd.DataFrame,
    column: str,
    tag_kernels: Sequence[str],
    baseline: str = BASELINE,
    mark_pending: bool = False,
    baseline_fallback: str = "",
) -> Row:
    """One canon-sweep column's row against ``baseline``, restricted to ``tag``: a kernel without a validated result
    is filled at 1x and flagged undelivered (never dropped); ``mark_pending`` sets aside kernels without a canon row;
    ``baseline_fallback`` times a kernel the baseline did not verify against that column instead."""
    raced = canon.with_best_of(canon.read_times(canon_frame), baseline, baseline_references(baseline))
    times, substituted = canon.with_fallback(raced, baseline, baseline_fallback)
    substituted = substituted & set(tag_kernels)
    base, cur = times.get(baseline, {}), times.get(column, {})
    kernels = sorted(tag_kernels)
    pending = (
        unattempted_kernels(canon_frame, column, kernels, baseline, baseline_fallback) if mark_pending else frozenset()
    )
    ratios, delivered = canon.tag_speedups(times, baseline, column, [k for k in kernels if k not in pending])
    nan = math.nan
    numerator_ms = {k: base.get(k, nan) for k in ratios}
    denominator_ms = {k: cur.get(k, nan) for k in ratios}
    notes = [note for note in (pending_note(pending), fallback_note(substituted, baseline_fallback)) if note != "none"]
    return Row(
        column, canon_label(column), ratios, numerator_ms, denominator_ms, "; ".join(notes) or "none",
        palette.framework_color(column), palette.framework_marker(column), delivered=delivered, pending=pending,
    )  # fmt: skip


def unattempted_kernels(
    canon_frame: pd.DataFrame, column: str, kernels: Sequence[str], baseline: str, baseline_fallback: str
) -> frozenset[str]:
    """The ``kernels`` with no canon row yet for ``column`` or for ``baseline`` (or its fallback)."""
    status = canon.read_status(canon_frame)
    references = [canon.REFERENCE_COLUMNS.get(name, name) for name in baseline_references(baseline)]
    base_run = set[str]().union(*(status.get(name, {}).keys() for name in (baseline, *references))) | (
        status.get(baseline_fallback, {}).keys() if baseline_fallback else set()
    )
    attempted = status.get(column, {}).keys() & base_run
    return frozenset(k for k in kernels if k not in attempted)


def canon_label(column: str) -> str:
    """A canon column's legend label: its optimizer's standalone name, else its framework name."""
    optimizer = study_tags.canonical("optimizers", column)
    standalone = study_tags.names("optimizers")
    return standalone[optimizer] if optimizer in standalone else study_tags.names("frameworks").get(column, column)


def fallback_note(substituted: frozenset[str], fallback: str) -> str:
    """A row's ``excluded`` text: how many kernels were timed against ``fallback``."""
    return f"{len(substituted)} over {fallback}" if substituted else "none"


def pending_note(pending: frozenset[str]) -> str:
    """A row's ``excluded`` text: how many tag kernels it has not attempted yet."""
    return f"{len(pending)} pending" if pending else "none"


def distinct_canon_labels(rows: Sequence[Row]) -> list[Row]:
    """``rows`` with every label two of them share (one optimizer on two devices) replaced by the framework's own
    name."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.label] = counts.get(row.label, 0) + 1
    return [
        dataclasses.replace(row, label=study_tags.framework_name(row.framework)) if counts[row.label] > 1 else row
        for row in rows
    ]


def agent_kernel_row(
    frame: pd.DataFrame,
    setup: str,
    model: str,
    condition: str,
    tag_kernels: Sequence[str],
    pending: frozenset[str] = frozenset(),
    treatments: Sequence[str] = (),
) -> Row:
    """One setup's row, restricted to ``tag``: its final answer per kernel, plus each kernel's
    own confidence interval over every graded episode it ran (SC15 rules 5/7)."""
    subset = frame.loc[frame["setup"].astype(str) == setup]
    answers = population.kernel_answers(subset, policy=population.KernelPolicy.SOLVED)
    kernels = set(tag_kernels)
    ratios, numerator_ms, denominator_ms = answer_ratios(answers, kernels)
    ratios_low, ratios_high = kernel_intervals(subset, ratios.keys(), setup)
    raw_tokens, tokens_low, tokens_high = setup_names.setup_tokens(subset, setup)
    del tokens_low, tokens_high  # under "latest" both are empty; a repeat's own range is not this figure's concern
    tokens = {k: v for k, v in raw_tokens.items() if k in kernels}
    label = f"{study_tags.model_name(model)} - {setup_names.condition_label(condition, treatments)}"
    # Colour is the model's and shape the packet's, hollow for no packet, as every model x packet figure draws them
    # (palette.packet_marker).
    return Row(
        setup, label, ratios, numerator_ms, denominator_ms, pending_note(pending),
        palette.model_color(model), palette.packet_marker(condition), ratios_low, ratios_high, tokens,
        pending=pending, filled=bool(condition),
    )  # fmt: skip


def answer_ratios(
    answers: pd.DataFrame, kernels: set[str]
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Each ``kernels`` answer's positive finite speedup, with its baseline and native times in ms."""
    ratios: dict[str, float] = {}
    numerator_ms: dict[str, float] = {}
    denominator_ms: dict[str, float] = {}
    if "speedup" not in answers.columns:
        return ratios, numerator_ms, denominator_ms
    columns = zip(answers.index, answers["speedup"], answers["baseline_ns"], answers["native_ns"], strict=True)
    for name, speedup, baseline_ns, native_ns in columns:
        kernel = str(name)
        if kernel not in kernels or not per_kernel.usable(float(speedup)):
            continue
        ratios[kernel] = float(speedup)
        numerator_ms[kernel] = float(baseline_ns) / 1.0e6
        denominator_ms[kernel] = float(native_ns) / 1.0e6
    return ratios, numerator_ms, denominator_ms


def kernel_intervals(
    subset: pd.DataFrame, kernels: Collection[str], setup: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Each of ``kernels``' geomean-speedup CI over every graded episode of ``setup``, as (low, high)."""
    ratios_low: dict[str, float] = {}
    ratios_high: dict[str, float] = {}
    graded = subset.loc[subset["row_kind"] == RowKind.SUBMISSION.text] if "row_kind" in subset.columns else subset
    episodes = population.graded_episode_rows(graded, population.SUBMISSION_ORDER)
    if episodes.empty:
        return ratios_low, ratios_high
    for raw_kernel, group in episodes.groupby("kernel"):
        kernel = str(raw_kernel)
        if kernel not in kernels:
            continue
        values = usable_ratios(group["speedup"].tolist(), label=f"{setup}@{kernel}")
        if values.size == 0:
            continue
        interval = geomean_ci(values)
        ratios_low[kernel] = interval.low
        ratios_high[kernel] = interval.high
    return ratios_low, ratios_high


def kernel_rows(
    canon_frame: pd.DataFrame,
    observations: pd.DataFrame | None,
    tag_kernels: Sequence[str],
    canon_columns: Sequence[str],
    pattern: re.Pattern[str] | None = None,
    conditions: Collection[str] | None = None,
    baseline: str = BASELINE,
    mark_pending: bool = False,
    baseline_fallback: str = "",
) -> list[Row]:
    """The canon rows of ``canon_columns``, then every TAG-COMPLETE setup row ``pattern`` names, against
    ``baseline``; ``conditions`` filters setups (``""`` = control), ``observations=None`` draws canon rows alone, and
    ``mark_pending`` keeps incomplete setups with their missing kernels pending."""
    rows = distinct_canon_labels(
        [
            canon_kernel_row(canon_frame, column, tag_kernels, baseline, mark_pending, baseline_fallback)
            for column in canon_columns
        ]
    )
    if observations is None:
        return rows
    if pattern is None:
        raise ValueError("observations need a setup pattern (setup_names.setup_pattern)")
    candidates = setup_names.candidate_setups(observations, pattern)
    frame = observations.loc[observations["setup"].astype(str).isin(candidates)]
    kept, dropped = population.complete_setups(frame, tag_kernels)
    if mark_pending:
        kept = [*kept, *dropped]
    # A condition may be spelled with a packet alias (``cpf-src``); the setups carry the canonical key.
    wanted = None if conditions is None else {packets.canonical(condition) for condition in conditions}
    kept = [setup for setup in kept if wanted is None or candidates[setup][1] in wanted]
    treatments = sorted({candidates[setup][1] for setup in kept} - {""})
    by_model: dict[str, list[str]] = {}
    for setup in kept:
        by_model.setdefault(candidates[setup][0], []).append(setup)
    # One model drawn under several conditions: colour carries the condition too (the control's neutral colour, each
    # treatment its packet colour), since one model colour would leave shape alone to tell them apart.
    by_condition = len(by_model) == 1 and len(kept) > 1
    conditions_drawn = sorted({candidates[setup][1] for setup in kept}, key=setup_names.rank_condition)
    treatment_hues = iter(palette.hues())
    condition_colors = (
        {c: palette.control_color() if c == "" else next(treatment_hues) for c in conditions_drawn}
        if by_condition
        else {}
    )
    for model in palette.in_order(by_model.keys(), "models"):
        for setup in sorted(by_model[model], key=lambda a: setup_names.rank_condition(candidates[a][1])):
            served = set(frame.loc[frame["setup"].astype(str) == setup, "kernel"].astype(str))
            pending = frozenset(k for k in tag_kernels if k not in served)
            condition = candidates[setup][1]
            row = agent_kernel_row(frame, setup, model, condition, tag_kernels, pending, treatments)
            rows.append(dataclasses.replace(row, color=condition_colors[condition]) if by_condition else row)
    return rows


#: A panel's height in the per-kernel figure, inches: what 40 kernels need to read at the
#: text width the figure prints at, not what the canvas can spare.
PANEL_HEIGHT_IN: float = 1.5


def baseline_references(baseline: str) -> tuple[str, ...]:
    """The references a speedup denominator races: one framework, or every reference of a best-of
    denominator (:data:`~hpcagent_bench.harness.denominator.KINDS`, ``best-of(numba,c)``)."""
    named = denominator.of_stored(baseline)
    return (baseline,) if named is None else denominator.KINDS[named]


def baseline_label(baseline: str) -> str:
    """The speedup axis label, naming the baseline by its registry display name; a best-of denominator
    names each reference it races (``Speedup over Best of GCC and Numba``)."""
    names = [study_tags.framework_name(reference) for reference in baseline_references(baseline)]
    return f"Speedup over {names[0] if len(names) == 1 else 'Best of ' + ' and '.join(names)}"


def baseline_color(baseline: str) -> str:
    """The 1x reference line's colour: its framework's, or ink for a best-of denominator (no one framework)."""
    references = baseline_references(baseline)
    return palette.framework_color(references[0]) if len(references) == 1 else style.INK


def row_color(row: Row) -> str:
    """The colour ``row`` draws in: its own, or its framework's for a row that sets none."""
    return row.color or palette.framework_color(row.framework)


def kernel_metrics(
    rows: Sequence[Row], tag_kernels: Sequence[str], baseline: str = BASELINE, cost_label: str = "Billed Tokens"
) -> list[per_kernel.Metric]:
    """The compiler figure's panels over ``sorted(tag)``: speedup for every row (whiskers = per-kernel 95%
    intervals, placeholders crossed, pending as "?", the 1x line in the baseline's colour), and token cost only when
    some row spends tokens (``cost_label`` names the card)."""
    kernels = sorted(tag_kernels)
    speed = [
        per_kernel.Series(
            row.label,
            per_kernel.kernel_cells(
                row.ratios, kernels, delivered=row.delivered, low=row.ratios_low, high=row.ratios_high,
                pending=row.pending,
            ),
            row_color(row),
            row.marker,
            row.filled,
        )
        for row in rows
    ]  # fmt: skip
    metrics = [per_kernel.speedup_series_metric(speed, baseline_label(baseline), baseline_color(baseline))]
    if any(row.tokens for row in rows):
        tokens = [
            per_kernel.Series(
                row.label, per_kernel.kernel_cells(row.tokens, kernels, fill=False), row_color(row), row.marker,
                row.filled,
            )
            for row in rows
        ]  # fmt: skip
        metrics.append(per_kernel.token_series_metric(tokens, cost_label))
    return metrics


def legend_handles(rows: Sequence[Row], metrics: Sequence[per_kernel.Metric]) -> list[Artist]:
    """One legend entry per row (its colour, shape and ``row.label``) plus the status marks ``metrics`` draw."""
    handles: list[Artist] = [
        Line2D(
            [], [], marker=palette.marker_style(row.marker), linestyle="none", color=row_color(row),
            markersize=per_kernel.LEGEND_MARK_PT,
            markerfacecolor=row_color(row) if row.filled else "none", label=row.label,
        )
        for row in rows
    ]  # fmt: skip
    return handles + per_kernel.status_handles(metrics)


def kernel_figure(
    rows: Sequence[Row],
    tag_kernels: Sequence[str],
    title: str = "",
    baseline: str = BASELINE,
    offset: float = 0.0,
    panel_height_in: float = PANEL_HEIGHT_IN,
    cost_label: str = "Billed Tokens",
) -> matplotlib.figure.Figure:
    """The per-kernel figure: canon-sweep columns and setups on ONE kernel axis, a log2 speedup panel over a token
    panel when any row spends tokens, each with a geomean summary column, drawn at print size by
    :func:`per_kernel.figure_panels`. ``offset`` spreads a kernel's rows over that fraction of a column."""
    if not rows:
        raise ValueError("no row to draw")
    style.apply()
    metrics = kernel_metrics(rows, tag_kernels, baseline, cost_label)
    return per_kernel.figure_panels(
        metrics,
        sorted(tag_kernels),
        per_kernel.Style.CI,
        True,
        title,
        width_in=style.DOUBLE_COLUMN_WIDTH,
        legend=legend_handles(rows, metrics),
        span=offset,
        panel_height_in=panel_height_in,
    )


def token_summary_table(rows: Sequence[Row]) -> pd.DataFrame:
    """Each token-spending row's GEOMEAN spend over its served kernels with a task total, with the figure's 95%
    interval (blank under ``summary.MIN_PAIRS_FOR_INTERVAL`` kernels; all-blank fails Rule 5). Non-spending rows are
    ABSENT."""
    records: list[dict[str, object]] = []
    for row in rows:
        cells = per_kernel.kernel_cells(row.tokens, sorted(row.tokens), fill=False)
        point, low, high = per_kernel.summary_geomean(cells)
        if not math.isfinite(point):
            continue
        records.append(
            {
                "row": row.label,
                "framework": row.framework,
                "n": len(per_kernel.kernel_medians(cells)),
                "gm_tokens": point,
                "gm_tokens_low": low,
                "gm_tokens_high": high,
            }
        )
    frame = pd.DataFrame.from_records(records, columns=TOKEN_SUMMARY_COLUMNS)
    return rules.require_interval(frame, "gm_tokens", "gm_tokens_low", "gm_tokens_high")


def kernel_comparison(
    canon_frame: pd.DataFrame,
    observations: pd.DataFrame | None,
    tag_kernels: Sequence[str],
    out: pathlib.Path,
    canon_columns: Sequence[str],
    pattern: re.Pattern[str] | None = None,
    conditions: Collection[str] | None = None,
    baseline: str = BASELINE,
    title: str = "",
    dpi: float = 150.0,
    labels: Mapping[str, str] | None = None,
    offset: float = 0.0,
    mark_pending: bool = False,
    baseline_fallback: str = "",
    panel_height_in: float = PANEL_HEIGHT_IN,
    cost_label: str = "Billed Tokens",
) -> pathlib.Path:
    """Build the per-kernel rows (:func:`kernel_rows`), write their tables (:func:`write_tables`,
    :func:`token_summary_table`) and render the figure; ``labels`` renames rows by key, ``dpi`` defaults to 150."""
    rows = kernel_rows(
        canon_frame,
        observations,
        tag_kernels,
        canon_columns,
        pattern,
        conditions,
        baseline,
        mark_pending,
        baseline_fallback,
    )
    rows = [dataclasses.replace(row, label=(labels or {}).get(row.framework, row.label)) for row in rows]
    write_tables(rows, out)
    tokens = token_summary_table(rows)
    if not tokens.empty:
        tokens.to_csv(out.with_name(f"{out.name}-tokens-summary.csv"), index=False)
    fig = kernel_figure(rows, tag_kernels, title, baseline, offset, panel_height_in, cost_label)
    return style.save(fig, out, formats=("pdf", "png"), fixed=True, dpi=dpi)


def write_tables(rows: Sequence[Row], stem: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Write the per-kernel and per-row tables beside the figure (written first; the tables are what gets diffed);
    returns both paths."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    per_kernel = stem.with_name(f"{stem.name}-kernels.csv")
    per_row = stem.with_name(f"{stem.name}-summary.csv")
    table(rows).to_csv(per_kernel, index=False)
    summary_table(rows).to_csv(per_row, index=False)
    return per_kernel, per_row
