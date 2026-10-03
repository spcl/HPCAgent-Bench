# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Signed-change per-kernel figures of the llr40 compiler and agent comparison (:func:`llr40_figure`,
:func:`llr40_two_row_figure`; ``statistics/plot_llr40_compilers.py`` draws them).

The axis is the signed relative change (:func:`hpcagent_bench.stats.summary.signed_change`), not
the ratio: 2x faster sits at +1, 2x slower at -1. Follows Hoefler and Belli (SC15) rules 4 (report
costs), 5/7 (report intervals) and 12 (no line between unordered rows), checked by
:mod:`hpcagent_bench.stats.rules`.
"""

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

from hpcagent_bench import study_tags
from hpcagent_bench.stats import canon, palette, population, rules, style
from hpcagent_bench.stats.figures import llr40_setups, per_kernel
from hpcagent_bench.stats.summary import geomean_ci, signed_change, usable_ratios

__all__ = [
    "DEAD_BAND",
    "LLR40_BASELINE",
    "LLR40_CANON_COLUMNS",
    "LLR40_CONDITIONS",
    "LLR40_PANEL_HEIGHT_IN",
    "SUMMARY_COLUMNS",
    "TABLE_COLUMNS",
    "TOKEN_SUMMARY_COLUMNS",
    "Row",
    "agent_kernel_row",
    "answer_ratios",
    "canon_kernel_row",
    "canon_label",
    "distinct_canon_labels",
    "fallback_note",
    "kernel_intervals",
    "legend_handles",
    "llr40_baseline_label",
    "llr40_figure",
    "llr40_metrics",
    "llr40_rows",
    "llr40_two_row_figure",
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

    ``color``/``marker`` and the trailing five fields are set only by the llr40 compiler
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
    marker: str = "o"
    ratios_low: dict[str, float] = dataclasses.field(default_factory=dict)  # per-kernel CI, SC15 5/7
    ratios_high: dict[str, float] = dataclasses.field(default_factory=dict)
    tokens: dict[str, float] = dataclasses.field(default_factory=dict)  # per-kernel token spend
    delivered: dict[str, bool] = dataclasses.field(default_factory=dict)  # ratio vs 1x placeholder
    pending: frozenset[str] = frozenset()  # kernels not attempted yet (mark_pending only)


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


#: The baseline every llr40 compiler row is measured against.
LLR40_BASELINE: str = llr40_setups.CANON_BASELINE

#: The two canon-sweep columns this figure draws as their own rows: DaCe's parallel-CPU backend,
#: then its canonicalizing pass. A caller wanting the polyhedral baselines too (Pluto, PPCG-on-AMD)
#: passes its own ``canon_columns`` (as :data:`statistics.plot_llr40_compilers`'s CLI default does).
LLR40_CANON_COLUMNS: tuple[str, ...] = ("dace_cpu", "dace_cpu_canonicalize")

#: The two CPF conditions this figure draws, per model -- never the no-packet control.
LLR40_CONDITIONS: tuple[str, ...] = ("cpf", "cpfsrc")

#: Column order of the emitted token summary table.
TOKEN_SUMMARY_COLUMNS: tuple[str, ...] = (
    "row", "framework", "n", "gm_tokens", "gm_tokens_low", "gm_tokens_high",
)  # fmt: skip


def canon_kernel_row(
    canon_frame: pd.DataFrame,
    column: str,
    tag_kernels: Sequence[str],
    baseline: str = LLR40_BASELINE,
    mark_pending: bool = False,
    baseline_fallback: str = "",
) -> Row:
    """One canon-sweep column's row against ``baseline``, tag-complete (restricted to
    ``tag``, since a canon sweep commonly spans more kernels than one figure's).

    A tag kernel ``column`` produced no validated result for is filled at 1x, never dropped
    (:func:`hpcagent_bench.stats.canon.tag_speedups`), flagged via ``delivered`` so the figure
    draws it crossed; the summary slot leaves it out. ``mark_pending`` instead sets aside a kernel
    with no canon row yet for ``column`` or ``baseline``, landing in ``pending``. ``baseline_fallback``
    times a kernel ``baseline`` did not verify against that column instead.
    """
    times, substituted = canon.with_fallback(canon.read_times(canon_frame), baseline, baseline_fallback)
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
        palette.framework_color(column), palette.marker(column), delivered=delivered, pending=pending,
    )  # fmt: skip


def unattempted_kernels(
    canon_frame: pd.DataFrame, column: str, kernels: Sequence[str], baseline: str, baseline_fallback: str
) -> frozenset[str]:
    """The ``kernels`` with no canon row yet for ``column`` or for ``baseline`` (or its fallback)."""
    status = canon.read_status(canon_frame)
    base_run = status.get(baseline, {}).keys() | (
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
    """``rows`` with every label two of them share replaced by the framework's own name.

    A canon column is labelled by its optimizer, and the registry aliases both device variants of
    one optimizer to one name (``dace_cpu_canonicalize`` and ``dace_gpu_canonicalize`` are both
    "Canonical Parallel Form"). Only a shared label falls back to the ``frameworks`` name, which
    carries the device.
    """
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
    repeats: population.RepeatPolicy = population.RepeatPolicy.LATEST,
    pending: frozenset[str] = frozenset(),
) -> Row:
    """One CPF setup's row, restricted to ``tag``: its final answer per kernel, plus each kernel's
    own confidence interval over every graded episode it ran (SC15 rules 5/7)."""
    subset = frame.loc[frame["setup"].astype(str) == setup]
    answers = population.kernel_answers(subset, repeats=repeats, policy=population.KernelPolicy.SOLVED)
    kernels = set(tag_kernels)
    ratios, numerator_ms, denominator_ms = answer_ratios(answers, kernels)
    ratios_low, ratios_high = kernel_intervals(subset, ratios.keys(), setup)
    raw_tokens, tokens_low, tokens_high = llr40_setups.setup_tokens(subset, setup, repeats)
    del tokens_low, tokens_high  # under "latest" both are empty; a repeat's own range is not this figure's concern
    tokens = {k: v for k, v in raw_tokens.items() if k in kernels}
    label = f"{study_tags.model_name(model)} - {llr40_setups.condition_label(condition)}"
    return Row(
        setup, label, ratios, numerator_ms, denominator_ms, pending_note(pending),
        palette.color(condition), palette.marker(model), ratios_low, ratios_high, tokens, pending=pending,
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
    graded = subset.loc[subset["row_kind"] == "submission"] if "row_kind" in subset.columns else subset
    episodes = population.graded_episode_rows(graded, population.SUBMISSION_ORDER)
    if episodes.empty:
        return ratios_low, ratios_high
    for kernel, group in episodes.groupby("kernel"):
        kernel = str(kernel)
        if kernel not in kernels:
            continue
        values = usable_ratios(group["speedup"].tolist(), label=f"{setup}@{kernel}")
        if values.size == 0:
            continue
        interval = geomean_ci(values)
        ratios_low[kernel] = interval.low
        ratios_high[kernel] = interval.high
    return ratios_low, ratios_high


def llr40_rows(
    canon_frame: pd.DataFrame,
    observations: pd.DataFrame | None,
    tag_kernels: Sequence[str],
    baseline: str = LLR40_BASELINE,
    canon_columns: Sequence[str] = LLR40_CANON_COLUMNS,
    conditions: Sequence[str] = LLR40_CONDITIONS,
    pattern: re.Pattern[str] = llr40_setups.SETUP_PATTERN,
    repeats: population.RepeatPolicy = population.RepeatPolicy.LATEST,
    mark_pending: bool = False,
    baseline_fallback: str = "",
) -> list[Row]:
    """DaCe's own canon-sweep rows, then every model's TAG-COMPLETE CPF setup rows
    (:func:`~hpcagent_bench.stats.population.complete_setups`), all against ``baseline`` -- the
    llr40 compiler figure's row source. ``observations=None`` draws the canon rows alone: the
    experiment DB is not always reachable, and a figure with only the deterministic columns is still
    a real, if partial, answer -- never a raised error.

    ``mark_pending`` also keeps a setup that has not been served every tag kernel yet, its missing
    kernels in ``pending``, where the default drops it."""
    rows = distinct_canon_labels(
        [
            canon_kernel_row(canon_frame, column, tag_kernels, baseline, mark_pending, baseline_fallback)
            for column in canon_columns
        ]
    )
    if observations is None:
        return rows
    candidates = llr40_setups.candidate_setups(observations, pattern)
    frame = observations.loc[observations["setup"].astype(str).isin(candidates)]
    kept, dropped = population.complete_setups(frame, tag_kernels)
    if mark_pending:
        kept = [*kept, *dropped]
    by_model: dict[str, list[str]] = {}
    for setup in kept:
        model, condition = candidates[setup]
        if condition in conditions:
            by_model.setdefault(model, []).append(setup)
    for model in palette.in_order(by_model.keys(), "models"):
        for setup in sorted(by_model[model], key=lambda a: llr40_setups.rank_condition(candidates[a][1])):
            model_tag, condition = candidates[setup]
            served = set(frame.loc[frame["setup"].astype(str) == setup, "kernel"].astype(str))
            pending = frozenset(k for k in tag_kernels if k not in served)
            rows.append(agent_kernel_row(frame, setup, model_tag, condition, tag_kernels, repeats, pending))
    return rows


#: A panel's height in the llr40 compiler figure, inches: what 40 kernels need to read at the
#: text width the figure prints at, not what the canvas can spare.
LLR40_PANEL_HEIGHT_IN: float = 1.5


def llr40_baseline_label(baseline: str) -> str:
    """The speedup axis label, naming the baseline by its registry display name."""
    return f"Speedup over {study_tags.names('frameworks').get(baseline, baseline)}"


def row_color(row: Row) -> str:
    """The colour ``row`` draws in: its own, or its framework's for a row that sets none."""
    return row.color or palette.framework_color(row.framework)


def llr40_metrics(
    rows: Sequence[Row], tag_kernels: Sequence[str], baseline: str = LLR40_BASELINE
) -> list[per_kernel.Metric]:
    """The compiler figure's panels over ``sorted(tag)``, as :mod:`per_kernel` draws them.

    Speedup for every row: a kernel's own repeat interval (SC15 rules 5/7) as its whisker, a
    compiler's 1x placeholder crossed (``Row.delivered``), an unanswered agent kernel filled at 1x and
    crossed, a pending kernel as "?"; the 1x line wears the baseline's own colour, since it IS the
    baseline. Tokens spent only when some row spends any -- a compiler-only render has nothing to
    put there, and an empty panel is omitted rather than left blank; a row that spends none (a canon
    column) keeps its dodge offset and summary slot on it and draws nothing.
    """
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
        )
        for row in rows
    ]  # fmt: skip
    metrics = [
        per_kernel.speedup_series_metric(speed, llr40_baseline_label(baseline), palette.framework_color(baseline))
    ]
    if any(row.tokens for row in rows):
        tokens = [
            per_kernel.Series(
                row.label, per_kernel.kernel_cells(row.tokens, kernels, fill=False), row_color(row), row.marker
            )
            for row in rows
        ]
        metrics.append(per_kernel.token_series_metric(tokens, "Tokens spent"))
    return metrics


def legend_handles(rows: Sequence[Row], metrics: Sequence[per_kernel.Metric]) -> list[Artist]:
    """One legend entry per row -- its own colour and shape, and the display name :func:`llr40_rows`
    built into ``row.label`` -- plus the status marks ``metrics`` actually draw
    (:func:`per_kernel.status_handles`). The interval method and its n belong to the caption: every
    whisker on this figure is a 95% interval."""
    handles: list[Artist] = [
        Line2D(
            [], [], marker=row.marker, linestyle="none", color=row_color(row), markersize=per_kernel.LEGEND_MARK_PT,
            label=row.label,
        )
        for row in rows
    ]  # fmt: skip
    return handles + per_kernel.status_handles(metrics)


def llr40_figure(
    rows: Sequence[Row],
    tag_kernels: Sequence[str],
    title: str = "",
    baseline: str = LLR40_BASELINE,
    offset: float = 0.0,
    panel_height_in: float = LLR40_PANEL_HEIGHT_IN,
) -> matplotlib.figure.Figure:
    """The llr40 compiler figure: DaCe's own canon-sweep columns and every model's CPF setup on
    ONE kernel axis, a speedup panel (log2, ratio-labelled ticks) over a tokens-spent panel when
    any row spends tokens (:func:`llr40_metrics`), each with per_kernel's summary column past a
    dashed separator -- one slot per row, the geomean with its 95% interval on both panels, over
    the kernels the row solved, value printed.

    Drawn by :func:`per_kernel.figure_panels` at the size it prints (text width,
    :data:`~hpcagent_bench.stats.style.DOUBLE_COLUMN_WIDTH`, print type) with the kernels' short
    names and the key under them, every band measured. No title unless ``title`` names one: a
    paper's caption already does. Every row of a kernel sits at the SAME x (the optimizers differ by
    shape), unless ``offset`` spreads them over that fraction of a kernel column."""
    if not rows:
        raise ValueError("no row to draw")
    style.apply()
    metrics = llr40_metrics(rows, tag_kernels, baseline)
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
    """Each token-spending row's GEOMEAN spend over the kernels it was served and has a task total
    for, with the 95% log-t interval the figure's token summary slot draws
    (:func:`per_kernel.summary_geomean` over the same cells). The interval is blank under
    ``summary.MIN_PAIRS_FOR_INTERVAL`` kernels, and a table where EVERY row is that thin fails Rule 5
    (:func:`hpcagent_bench.stats.rules.require_interval`). A row that spends no tokens (a canon
    column) is ABSENT, never entered at zero."""
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


def llr40_two_row_figure(
    canon_frame: pd.DataFrame,
    observations: pd.DataFrame | None,
    tag_kernels: Sequence[str],
    out: pathlib.Path,
    baseline: str = LLR40_BASELINE,
    canon_columns: Sequence[str] = LLR40_CANON_COLUMNS,
    conditions: Sequence[str] = LLR40_CONDITIONS,
    pattern: re.Pattern[str] = llr40_setups.SETUP_PATTERN,
    repeats: population.RepeatPolicy = population.RepeatPolicy.LATEST,
    title: str = "",
    dpi: float = 150.0,
    labels: Mapping[str, str] | None = None,
    offset: float = 0.0,
    mark_pending: bool = False,
    baseline_fallback: str = "",
    panel_height_in: float = LLR40_PANEL_HEIGHT_IN,
) -> pathlib.Path:
    """Build the llr40 compiler rows, write their tables (Rule 4's costs, rules 5/7's
    intervals -- :func:`write_tables`, :func:`token_summary_table`) and render the two-panel
    figure. The ONE function a script calls; ``statistics/plot_llr40_compilers.py`` only parses args.
    ``labels`` renames a row by its framework or setup key (a paper's own name for a column); the
    tables carry the same names the legend does.
    ``dpi`` defaults to 150 -- this figure's own review/paper convention, not
    :func:`~hpcagent_bench.stats.style.save`'s general-purpose 200.
    """
    rows = llr40_rows(
        canon_frame,
        observations,
        tag_kernels,
        baseline,
        canon_columns,
        conditions,
        pattern,
        repeats,
        mark_pending,
        baseline_fallback,
    )
    rows = [dataclasses.replace(row, label=(labels or {}).get(row.framework, row.label)) for row in rows]
    write_tables(rows, out)
    tokens = token_summary_table(rows)
    if not tokens.empty:
        tokens.to_csv(out.with_name(f"{out.name}-tokens-summary.csv"), index=False)
    fig = llr40_figure(rows, tag_kernels, title, baseline, offset, panel_height_in)
    return style.save(fig, out, formats=("pdf", "png"), fixed=True, dpi=dpi)


def write_tables(rows: Sequence[Row], stem: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Write the per-kernel and per-row tables beside the figure. Returns both paths.

    Written BEFORE the figure is drawn, and diffed rather than the image: a figure whose numbers
    moved is a regression, and a figure whose pixels moved because the frame changed is not.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    per_kernel = stem.with_name(f"{stem.name}-kernels.csv")
    per_row = stem.with_name(f"{stem.name}-summary.csv")
    table(rows).to_csv(per_kernel, index=False)
    summary_table(rows).to_csv(per_row, index=False)
    return per_kernel, per_row
