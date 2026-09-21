# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The signed-change figures: one row per arm, and the paired comparison of two tools.

TWO QUESTIONS, TWO FIGURES, ONE READER. :func:`arms_figure` puts several arms on a COMMON
denominator and answers "how fast is each arm". :func:`paired_figure` answers the different
question an ablation is about: on a kernel BOTH tools compiled, which is faster -- so its
denominator is the other TOOL, per kernel, and each row is a paired comparison rather than an
independent arm. Dividing two geomeans taken over different kernel sets is not a speed-up of
anything: an arm that fails on the kernels it is bad at comes out ahead by attrition.

THE AXIS is the signed relative change (:func:`hpcagent_bench.stats.summary.signed_change`), not
the ratio: 2x faster sits at +1 and 2x slower at -1, equidistant and odd about zero.

THE RULES this figure is built to keep, from Hoefler and Belli (SC15), checked by
:mod:`hpcagent_bench.stats.rules` rather than by review:

* Rule 4 -- the emitted table carries the MILLISECONDS behind every ratio, not the ratio alone.
* Rules 5 and 7 -- every row's geomean carries its log-space t-interval, and the per-kernel cloud
  is drawn beside it, so a reader sees the spread the interval summarizes.
* Rule 12 -- nothing is joined by a line. The rows are arms, which have no order, so a line
  between them would claim a trend across an axis that has none.

Usage::

    python -m hpcagent_bench.stats.figures.signed <sweep-directory> [--out DIR]
"""

import argparse
import collections
import csv
import dataclasses
import math
import pathlib
import random
import re
import sys
from collections.abc import Callable, Mapping, Sequence

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs] -- pandas ships none
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from hpcagent_bench import experiment_tags, flags
from hpcagent_bench.stats import canon, palette, population, rules, style
from hpcagent_bench.stats.figures import kernel_comparison
from hpcagent_bench.stats.summary import DEFAULT_CONFIDENCE, geomean_ci, log2_change, signed_change, usable_ratios

#: Framework -> the name a reader knows it by. Insertion order is the order on the axis.
ARMS: dict[str, str] = {
    "dace_cpu_canonicalize": "dace canon",
    "dace_cpu": "dace main",
    "cc_llvm_autopar": "llvm + polly",
}

#: The speed-up DENOMINATOR: a SERIAL optimizing compile, not the interpreted reference.
#:
#: cc, not numpy, and not the llvm+polly arm either. numpy flatters an auto-parallelizer for reasons
#: that have nothing to do with parallelization, and polly is not defined on every kernel -- a
#: non-affine loop is outside a polyhedral tool, so using it as the divisor would drop exactly the
#: kernels it cannot handle and measure the others on its home ground. cc exists for every kernel,
#: so every arm keeps full n, and llvm+polly stays visible as an ARM instead of hiding in the
#: denominator.
BASELINE: str = "cc"

#: Kernels these figures are about. tsvc_2_5* sources are already under this prefix.
TSVC_PREFIX: str = "tsvc_2"

#: The NUMERATOR of every ratio on the paired figure: the arm the ablation is about.
REFERENCE: str = "dace_cpu_canonicalize"

#: Denominator -> the row label, for the paired figure. Insertion order is the order on the axis.
COMPARISONS: dict[str, str] = {"dace_cpu": "vs dace main", "cc_llvm_autopar": "vs llvm + polly"}

#: Dead band of the sign test. Below 1% the two arms are the same code and the difference is jitter.
DEAD_BAND: float = 1.01


@dataclasses.dataclass(frozen=True, slots=True)
class Arm:
    """One framework's usable TSVC timings in ms, and a tally of what was thrown away and why."""

    framework: str
    times: dict[str, float]
    rejected: collections.Counter[str]


@dataclasses.dataclass(frozen=True, slots=True)
class Row:
    """One drawn row: its ratios per kernel, the costs behind them, and what it excluded.

    ``color``/``marker`` and the trailing four fields are used only by the llr-focus40 compiler
    figure (:func:`llr40_rows`, :func:`llr40_figure`): the TSVC rows :func:`arm_rows` and
    :func:`paired_rows` build never set them, so ``draw()`` keeps colouring by
    :func:`~hpcagent_bench.stats.palette.framework_colors` and every mark keeps the plain circle it
    always drew. ``ratios_low``/``ratios_high`` are a per-kernel confidence bound on ``ratios`` OVER
    THE KERNEL'S OWN REPETITIONS (SC15 rules 5/7) -- empty for a deterministic column, which has
    none to bound. ``tokens`` is the per-kernel spend a canon column has none of. ``delivered`` says
    which of ``ratios`` are real measurements versus the :data:`~hpcagent_bench.stats.population.
    NOT_DELIVERED` 1x placeholder (:func:`~hpcagent_bench.stats.canon.roster_speedups`) -- empty for
    an agent row, whose ``ratios`` only ever holds delivered kernels already (``policy="solved"``),
    so every present value there reads as delivered by the same default an empty dict gives it.
    ``pending`` names the kernels the row has not attempted yet (only under ``mark_pending``): they
    carry no ratio, enter no summary, and draw as :data:`~hpcagent_bench.stats.style.PENDING_MARKER`.
    """

    framework: str
    label: str
    ratios: dict[str, float]
    numerator_ms: dict[str, float]
    denominator_ms: dict[str, float]
    excluded: str
    color: str | None = None
    marker: str = "o"
    ratios_low: dict[str, float] = dataclasses.field(default_factory=dict)
    ratios_high: dict[str, float] = dataclasses.field(default_factory=dict)
    tokens: dict[str, float] = dataclasses.field(default_factory=dict)
    delivered: dict[str, bool] = dataclasses.field(default_factory=dict)
    pending: frozenset[str] = frozenset()


def shard_paths(root: pathlib.Path, framework: str) -> list[pathlib.Path]:
    """The framework's CSVs: the unsharded file, the per-rank shards, or both, in a stable order."""
    single = root / f"{framework}.csv"
    return ([single] if single.is_file() else []) + sorted(root.glob(f"{framework}.rank*.csv"))


def read_arm(root: pathlib.Path, framework: str) -> Arm:
    """Concatenate the framework's shards into ``kernel -> ms``, tallying every rejected TSVC row.

    A row that crashed, that the harness did not validate, or that carries no timing is NOT a data
    point. Those rows are counted and reported, never plotted as 1.0 -- a miscompile scored as "no
    change" is the one failure mode that would flatter every arm equally. Only TSVC rows are
    counted as rejects: a non-TSVC row is out of SCOPE, not excluded, and mixing the two would
    report the corpus size as an exclusion count.
    """
    times: dict[str, float] = {}
    rejected: collections.Counter[str] = collections.Counter()
    for path in shard_paths(root, framework):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                kernel = row["kernel"]
                if not kernel.startswith(TSVC_PREFIX):
                    continue
                if row["status"] != "ok":
                    rejected[f"status={row['status']}"] += 1
                elif row["validated"] != "True":
                    rejected["not validated"] += 1
                elif not row["median_ms"] or float(row["median_ms"]) <= 0:
                    rejected["no timing"] += 1
                else:
                    # A kernel can appear twice across shards only if a rank was re-run; the fastest
                    # of the two is the one the sweep itself would report, so keep the minimum.
                    ms = float(row["median_ms"])
                    times[kernel] = min(times.get(kernel, ms), ms)
    return Arm(framework, times, rejected)


def tally(arm: Arm) -> str:
    """One phrase naming every row the arm lost and why; empty when it lost none."""
    return ", ".join(f"{n} {why}" for why, n in sorted(arm.rejected.items()))


def against_baseline(arm: Arm, baseline: Mapping[str, float]) -> dict[str, float]:
    """``kernel -> baseline_ms / arm_ms`` over the kernels BOTH the arm and the reference timed."""
    return {k: baseline[k] / ms for k, ms in sorted(arm.times.items()) if k in baseline}


def paired(reference: Mapping[str, float], other: Mapping[str, float]) -> dict[str, float]:
    """``kernel -> other_ms / reference_ms`` over the kernels BOTH timed; above 1 the reference wins.

    Restricted to the intersection so every point is one kernel measured twice. A kernel only one
    of the two tools compiled leaves the comparison entirely, in both directions.
    """
    return {k: other[k] / ms for k, ms in sorted(reference.items()) if k in other}


def sign_test(ratios: Mapping[str, float]) -> tuple[int, int]:
    """Kernels the numerator wins and loses, outside the :data:`DEAD_BAND`."""
    wins = sum(1 for v in ratios.values() if v > DEAD_BAND)
    losses = sum(1 for v in ratios.values() if v < 1.0 / DEAD_BAND)
    return wins, losses


def table(rows: Sequence[Row]) -> pd.DataFrame:
    """The figure's DATA TABLE: one record per (row, kernel), ratio and both costs.

    Rule 4 is kept here rather than asserted in a caption -- the ratio ships with the milliseconds
    it was taken over, so a reader can see that a 1.4x on a 3 ms kernel is not a 1.4x on a 3 s one.
    """
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


def summary_table(rows: Sequence[Row]) -> pd.DataFrame:
    """One record per drawn row: the geomean, its interval, the median, n and the sign test.

    :func:`hpcagent_bench.stats.rules.require_interval` gates it, so a row that reached the figure
    without an interval fails here (Rules 5 and 7) instead of being drawn as a bare point.
    """
    records: list[dict[str, object]] = []
    for row in rows:
        values = usable_ratios(list(row.ratios.values()), label=row.label)
        if values.size == 0:
            records.append({"row": row.label, "framework": row.framework, "n": 0, "excluded": row.excluded})
            continue
        interval = geomean_ci(values)
        wins, losses = sign_test(row.ratios)
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


def draw_row(ax: Axes, index: int, row: Row, color: str, jitter: random.Random) -> None:
    """One arm's cloud of kernels, its geomean with interval, and its median tick.

    The cloud is Rule 12 read the other way: the individual kernels are plotted because they are
    what the interval summarizes, and they are NOT joined to anything -- a kernel axis has no
    order, so there is no trend for a line to indicate.
    """
    values = usable_ratios(list(row.ratios.values()), label=row.label)
    if values.size == 0:
        style.right_label(ax, index, "no data")
        return
    ax.scatter(  # pyright: ignore[reportUnknownMemberType]
        [signed_change(v) for v in values.tolist()],
        [index - 0.22 + jitter.uniform(-0.10, 0.10) for _ in range(values.size)],
        s=11,
        color=color,
        alpha=0.45,
        linewidth=0,
        zorder=2,
    )
    interval = geomean_ci(values)
    centre = signed_change(interval.point)
    ax.errorbar(  # pyright: ignore[reportUnknownMemberType]
        centre,
        index + 0.24,
        xerr=[[centre - signed_change(interval.low)], [signed_change(interval.high) - centre]],
        fmt="o",
        markersize=5.5,
        color=color,
        ecolor=style.REFERENCE,
        elinewidth=1.1,
        capsize=3.5,
        zorder=4,
    )
    middle = signed_change(float(np.median(values)))
    ax.plot(  # pyright: ignore[reportUnknownMemberType]
        [middle, middle], [index + 0.10, index + 0.38], color=style.INK, linewidth=1.0, zorder=3
    )
    style.right_label(ax, index, f"n={values.size}")


def draw(rows: Sequence[Row], title: str, xlabel: str, stem: pathlib.Path) -> pathlib.Path:
    """Render the rows onto the signed axis and write the figure. Returns ``stem``."""
    style.apply()
    tall = 1.35 + 0.62 * len(rows)
    fig, ax = plt.subplots(figsize=(6.8, tall))
    fig.subplots_adjust(left=0.20, right=0.885, top=1.0 - 0.86 / tall, bottom=1.30 / tall)
    # Seeded, so the same CSVs draw the same cloud: an unseeded jitter makes two renders of one
    # dataset look like two measurements.
    jitter = random.Random(0)
    style.row_axis(ax, [row.label for row in rows])
    ax.set_xlabel(xlabel)
    colors = palette.framework_colors([row.framework for row in rows])
    for index, row in enumerate(rows):
        draw_row(ax, index, row, colors[row.framework], jitter)
    ax.axvline(0.0, color=style.REFERENCE, linewidth=1.0, zorder=1)  # pyright: ignore[reportUnknownMemberType]
    ax.margins(x=0.08)
    style.value_axis(ax, axis="x")
    handles: list[Line2D] = [
        Line2D([], [], marker="o", linestyle="none", color=style.MUTED, markersize=3.5, alpha=0.5),
        Line2D([], [], marker="o", linestyle="none", color=style.MUTED, markersize=5.5),
        Line2D([], [], color=style.INK, linewidth=1.0),
    ]
    for handle, label in zip(handles, ("One Kernel", f"Geomean, {DEFAULT_CONFIDENCE:.0%} t-Interval", "Median")):
        handle.set_label(label)
    style.legend_below(fig, handles, ncol=3, y=0.015)
    style.title(fig, title)
    return style.save(fig, stem, formats=("pdf", "svg"))


def arm_rows(root: pathlib.Path, arms: Mapping[str, str] = ARMS, baseline: str = BASELINE) -> list[Row]:
    """Every arm's ratios against the common ``baseline``, in ``arms`` order."""
    reference = read_arm(root, baseline)
    if not reference.times:
        raise SystemExit(
            f"no {baseline} baseline in {root}: looked for {baseline}.csv and {baseline}.rank*.csv. "
            f"Every speed-up here is a ratio against it, so there is nothing to plot without it."
        )
    rows: list[Row] = []
    for framework, label in arms.items():
        arm = read_arm(root, framework)
        ratios = against_baseline(arm, reference.times)
        missing = len(arm.times) - len(ratios)
        excluded = [tally(arm)] if tally(arm) else []
        if missing:
            excluded.append(f"{missing} not timed by {baseline}")
        rows.append(
            Row(
                framework,
                label,
                ratios,
                {k: reference.times[k] for k in ratios},
                {k: arm.times[k] for k in ratios},
                "; ".join(excluded) or "none",
            )
        )
    return rows


def paired_rows(
    root: pathlib.Path, reference: str = REFERENCE, comparisons: Mapping[str, str] = COMPARISONS
) -> list[Row]:
    """Each comparison arm's ratios against ``reference``, restricted to the shared kernels."""
    numerator = read_arm(root, reference)
    if not numerator.times:
        raise SystemExit(
            f"no {reference} arm in {root}: looked for {reference}.csv and {reference}.rank*.csv. "
            f"It is the numerator of every ratio here, so there is nothing to plot without it."
        )
    rows: list[Row] = []
    for framework, label in comparisons.items():
        arm = read_arm(root, framework)
        ratios = paired(numerator.times, arm.times)
        unpaired = len(numerator.times) - len(ratios)
        excluded = [f"{unpaired} {reference} kernels unpaired"] if unpaired else []
        if tally(arm):
            excluded.append(f"comparison arm lost {tally(arm)}")
        rows.append(
            Row(
                framework,
                label,
                ratios,
                {k: arm.times[k] for k in ratios},
                {k: numerator.times[k] for k in ratios},
                "; ".join(excluded) or "none",
            )
        )
    return rows


#: The baseline every llr-focus40 compiler row is measured against
#: (:data:`kernel_comparison.CANON_BASELINE`).
LLR40_BASELINE: str = kernel_comparison.CANON_BASELINE

#: The two canon-sweep columns this figure draws as their OWN rows, in draw order: DaCe's
#: parallel-CPU backend, then its canonicalizing pass over the same backend. The LIBRARY default --
#: a caller wanting the polyhedral compiler baselines too (Pluto, PPCG-on-AMD) passes its own
#: ``canon_columns`` (:data:`statistics.plot_llr40_compilers`'s own CLI default does exactly this,
#: 2026-09-20), rather than widening what every existing caller of this constant draws.
LLR40_CANON_COLUMNS: tuple[str, ...] = ("dace_cpu", "dace_cpu_canonicalize")

#: The two CPF conditions this figure draws, per model -- never the no-packet control, which
#: answers a different question and which :mod:`hpcagent_bench.stats.figures.kernel_comparison`
#: already draws on its own axis.
LLR40_CONDITIONS: tuple[str, ...] = ("cpf", "cpfsrc")

#: Column order of the emitted token summary table.
TOKEN_SUMMARY_COLUMNS: tuple[str, ...] = (
    "row", "framework", "n", "geomean_tokens", "geomean_tokens_low", "geomean_tokens_high",
)  # fmt: skip


def canon_kernel_row(
    canon_frame: pd.DataFrame,
    column: str,
    roster: Sequence[str],
    baseline: str = LLR40_BASELINE,
    mark_pending: bool = False,
    baseline_fallback: str = "",
) -> Row:
    """One canon-sweep column's row against ``baseline``, ROSTER-COMPLETE: a single deterministic
    ``median_ms`` per kernel (:func:`hpcagent_bench.stats.canon.read_times`), so ``ratios_low``/
    ``ratios_high`` and ``tokens`` stay empty -- a canon sweep has no repetition to bound and runs
    no agent to cost. A canon sweep commonly spans MORE kernels than one figure's roster
    (:data:`kernel_comparison.CANON_COLUMN` sweeps 40); restricting to ``roster`` keeps the summary
    column from geomeaning a population the panel never drew.

    A roster kernel ``column`` produced no validated result for -- declined, crashed, or never
    attempted -- is FILLED at 1x, never dropped (:func:`hpcagent_bench.stats.canon.roster_speedups`,
    the 2026-09-20 decision): a compiler baseline that cannot handle a kernel is no different from
    an agent that never delivered one, and ``delivered`` flags it the same way
    :data:`~hpcagent_bench.stats.population.DELIVERED_COLUMN` flags that placeholder for an agent
    row, so ``llr40_figure`` draws and geomeans both under the one existing convention.

    ``mark_pending`` separates a kernel with NO canon row yet for ``column`` or ``baseline`` (never
    attempted) from one that ran and failed: it leaves the ratios and the summary and lands in
    ``pending``.

    ``baseline_fallback`` times a kernel ``baseline`` did not verify by that column instead
    (:func:`hpcagent_bench.stats.canon.with_fallback`); the row's ``excluded`` names how many did.
    """
    times, substituted = canon.with_fallback(canon.read_times(canon_frame), baseline, baseline_fallback)
    substituted = substituted & set(roster)
    base, cur = times.get(baseline, {}), times.get(column, {})
    kernels = sorted(roster)
    ratios, delivered = canon.roster_speedups(times, baseline, column, kernels)
    pending: frozenset[str] = frozenset()
    if mark_pending:
        status = canon.read_status(canon_frame)
        base_run = status.get(baseline, {}).keys() | (
            status.get(baseline_fallback, {}).keys() if baseline_fallback else set()
        )
        attempted = status.get(column, {}).keys() & base_run
        pending = frozenset(k for k in kernels if k not in attempted)
        ratios = {k: v for k, v in ratios.items() if k not in pending}
        delivered = {k: v for k, v in delivered.items() if k not in pending}
    nan = math.nan
    numerator_ms = {k: base.get(k, nan) for k in ratios}
    denominator_ms = {k: cur.get(k, nan) for k in ratios}
    optimizer = experiment_tags.canonical("optimizers", column)
    standalone = experiment_tags.names("optimizers")
    label = (
        standalone[optimizer] if optimizer in standalone else experiment_tags.names("frameworks").get(column, column)
    )
    return Row(
        column, label, ratios, numerator_ms, denominator_ms,
        "; ".join(note for note in (pending_note(pending), fallback_note(substituted, baseline_fallback)) if note != "none")
        or "none",
        palette.framework_color(column), palette.marker(column), delivered=delivered, pending=pending,
    )  # fmt: skip


def fallback_note(substituted: frozenset[str], fallback: str) -> str:
    """A row's ``excluded`` text: how many kernels were timed against ``fallback``."""
    return f"{len(substituted)} over {fallback}" if substituted else "none"


def pending_note(pending: frozenset[str]) -> str:
    """A row's ``excluded`` text: how many roster kernels it has not attempted yet."""
    return f"{len(pending)} pending" if pending else "none"


def distinct_canon_labels(rows: Sequence[Row]) -> list[Row]:
    """``rows`` with every label that two of them SHARE replaced by the framework's own name.

    A canon column is labelled by its optimizer, and the registry aliases both device variants of
    one optimizer to one name on purpose (``dace_cpu_canonicalize`` and ``dace_gpu_canonicalize``
    are both "Canonical Parallel Form"). A figure drawing both then shows two rows under one legend
    entry with nothing to say which device is which. Only then does the label fall back to the
    ``frameworks`` name, which carries the device; a figure drawing one variant keeps the optimizer
    name it always had.
    """
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.label] = counts.get(row.label, 0) + 1
    return [
        dataclasses.replace(row, label=experiment_tags.framework_name(row.framework)) if counts[row.label] > 1 else row
        for row in rows
    ]


def agent_kernel_row(
    frame: pd.DataFrame,
    arm: str,
    model: str,
    condition: str,
    roster: Sequence[str],
    repeats: population.RepeatPolicy = "latest",
    pending: frozenset[str] = frozenset(),
) -> Row:
    """One CPF arm's row, restricted to ``roster``: its final answer per kernel (Rule 4's costs
    behind the ratio), plus each kernel's OWN confidence interval over every graded episode that
    kernel ran (rules 5/7) -- the geomean of that kernel's repetitions, degenerate to a point below
    two samples (:func:`~hpcagent_bench.stats.summary.geomean_ci`)."""
    subset = frame[frame["arm"].astype(str) == arm]
    answers = population.kernel_answers(subset, repeats=repeats, policy="solved")
    kernels = set(roster)
    ratios: dict[str, float] = {}
    numerator_ms: dict[str, float] = {}
    denominator_ms: dict[str, float] = {}
    if "speedup" in answers.columns:
        for kernel, row in answers.iterrows():
            kernel = str(kernel)
            if kernel not in kernels:
                continue
            speedup = float(row["speedup"])
            if not math.isfinite(speedup) or speedup <= 0.0:
                continue
            ratios[kernel] = speedup
            numerator_ms[kernel] = float(row["baseline_ns"]) / 1.0e6
            denominator_ms[kernel] = float(row["native_ns"]) / 1.0e6
    ratios_low: dict[str, float] = {}
    ratios_high: dict[str, float] = {}
    graded = subset[subset["record"] == "submission"] if "record" in subset.columns else subset
    episodes = population.graded_episode_rows(graded, population.SUBMISSION_ORDER)
    if not episodes.empty:
        for kernel, group in episodes.groupby("benchmark"):
            kernel = str(kernel)
            if kernel not in ratios:
                continue
            values = usable_ratios(group["speedup"].tolist(), label=f"{arm}@{kernel}")
            if values.size == 0:
                continue
            interval = geomean_ci(values)
            ratios_low[kernel] = interval.low
            ratios_high[kernel] = interval.high
    raw_tokens, tokens_low, tokens_high = kernel_comparison.arm_tokens(subset, arm, repeats)
    del tokens_low, tokens_high  # under "latest" both are empty; a repeat's own range is not this figure's concern
    tokens = {k: v for k, v in raw_tokens.items() if k in kernels}
    label = f"{experiment_tags.model_name(model)} - {kernel_comparison.condition_label(condition)}"
    return Row(
        arm, label, ratios, numerator_ms, denominator_ms, pending_note(pending),
        palette.color(condition), palette.marker(model), ratios_low, ratios_high, tokens, pending=pending,
    )  # fmt: skip


def llr40_rows(
    canon_frame: pd.DataFrame,
    observations: pd.DataFrame | None,
    roster: Sequence[str],
    baseline: str = LLR40_BASELINE,
    canon_columns: Sequence[str] = LLR40_CANON_COLUMNS,
    conditions: Sequence[str] = LLR40_CONDITIONS,
    pattern: re.Pattern[str] = kernel_comparison.ARM_PATTERN,
    repeats: population.RepeatPolicy = "latest",
    mark_pending: bool = False,
    baseline_fallback: str = "",
) -> list[Row]:
    """DaCe's own canon-sweep rows, then every model's ROSTER-COMPLETE CPF arm rows
    (:func:`~hpcagent_bench.stats.population.complete_arms`), all against ``baseline`` -- the
    llr-focus40 compiler figure's row source. ``observations=None`` draws the canon rows alone: the
    campaign DB is not always reachable, and a figure with only the deterministic columns is still
    a real, if partial, answer -- never a raised error.

    ``mark_pending`` also keeps an arm that has not been served every roster kernel yet, its missing
    kernels in ``pending``, where the default drops it."""
    rows = distinct_canon_labels(
        [
            canon_kernel_row(canon_frame, column, roster, baseline, mark_pending, baseline_fallback)
            for column in canon_columns
        ]
    )
    if observations is None:
        return rows
    candidates = kernel_comparison.candidate_arms(observations, pattern)
    frame = observations[observations["arm"].astype(str).isin(candidates)]
    kept, dropped = population.complete_arms(frame, roster)
    if mark_pending:
        kept = [*kept, *dropped]
    by_model: dict[str, list[str]] = {}
    for arm in kept:
        model, condition = candidates[arm]
        if condition in conditions:
            by_model.setdefault(model, []).append(arm)
    for model in palette.in_order(by_model.keys(), "models"):
        for arm in sorted(by_model[model], key=lambda a: kernel_comparison.rank_condition(candidates[a][1])):
            model_tag, condition = candidates[arm]
            served = set(frame.loc[frame["arm"].astype(str) == arm, "benchmark"].astype(str))
            pending = frozenset(k for k in roster if k not in served)
            rows.append(agent_kernel_row(frame, arm, model_tag, condition, roster, repeats, pending))
    return rows


def geomean_reducer(values: "Sequence[float]") -> float:
    """The RIGHTMOST summary column's statistic for EVERY llr-focus40 row, on both panels: the
    geomean over the kernels the row has a value for (SC15 Rule 4's "use the geometric mean for
    summarizing ratios" applied identically to a speed-up ratio and to a token count -- neither
    is summed, and a median would not carry the log-space interval Rule 5/7 asks for)."""
    usable = usable_ratios(list(values), warn=False)
    return float(geomean_ci(usable).point) if usable.size else math.nan


def geomean_interval_of(
    row_by_key: Mapping[str, Row], select: Callable[[Row], dict[str, float]]
) -> Callable[["kernel_comparison.Series"], tuple[float, float]]:
    """A :func:`kernel_comparison.draw_summary_column` ``interval_of`` reading the SAME geomean's
    95% log-space t-interval that :func:`geomean_reducer` places the point from -- point and
    interval are ONE statistic, never two independently computed numbers that could disagree."""

    def interval(series: "kernel_comparison.Series") -> tuple[float, float]:
        usable = usable_ratios(list(select(row_by_key[series.key]).values()), warn=False)
        if usable.size == 0:
            return math.nan, math.nan
        ci = geomean_ci(usable)
        return ci.low, ci.high

    return interval


def style_log2_speedup_y_axis(ax: Axes, ratios: Sequence[float], reference_color: str = style.REFERENCE) -> None:
    """The speed-up panel's Y axis: LOG2 GEOMETRY (every doubling the same distance apart,
    :func:`~hpcagent_bench.stats.summary.log2_change`), ticks and labels read back in RATIOS
    (:func:`~hpcagent_bench.stats.style.ratio_tick_label`) so the axis still READS
    as a speed-up while only its geometry is log2. A LINEAR signed change (:func:`~hpcagent_bench.stats.summary.signed_change`)
    stretches every multiple of the baseline the same amount, so one 75x outlier would sit 74 units
    from zero and swamp every other kernel's mark onto a sliver near it; log2 puts that same 75x
    at +6.2, the same distance a 2x-to-4x doubling gets."""
    ticks = kernel_comparison.value_ticks(ratios)
    positions = [log2_change(tick) for tick in ticks]
    ax.set_yticks(positions)
    ax.set_yticklabels([style.ratio_tick_label(tick) for tick in ticks], fontsize=style.TICK_PT * 0.6)
    ax.set_ylim(positions[0] - 0.3, positions[-1] + 0.3)
    ax.axhline(0.0, color=reference_color, linewidth=0.9, zorder=1)
    ax.grid(axis="y", which="major", color=style.RULE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


#: The llr-focus40 figure is drawn at the size it prints: one text-width row
#: (:data:`~hpcagent_bench.stats.style.DOUBLE_COLUMN_WIDTH`), every font in real points, so the page
#: never rescales it. The panel height is what 40 kernels need to read, not what the canvas can spare.
LLR40_PANEL_HEIGHT_IN: float = 1.5
LLR40_TEXT_PT: float = 6.5
LLR40_LEGEND_PT: float = 6.5

#: A drawn mark's diameter in points: small enough that two optimizers on one kernel at nearby
#: speed-ups still show both shapes.
LLR40_MARK_PT: float = 4.0

#: Length of an x tick in points: every kernel slot and the summary slot get one.
LLR40_TICK_LEN_PT: float = 2.5

#: A legend mark's size in points at :data:`LLR40_LEGEND_PT`, before the legend's own markerscale.
LLR40_LEGEND_MARK_PT: float = 4.0

#: Inches left of the value axis (tick labels, axis label), right of the summary values, above the
#: top panel, and between two panels.
LLR40_LEFT_IN: float = 0.62
LLR40_RIGHT_IN: float = 0.3
LLR40_TOP_IN: float = 0.08
LLR40_GAP_IN: float = 0.12

#: Inches between the rotated kernel labels and the legend under them.
LLR40_LEGEND_GAP_IN: float = 0.06

#: The value axis keeps every power of two up to this many ticks, then only every other one: 13
#: labels on a 1.45-inch axis touch.
LLR40_MAX_TICKS: int = 8

#: The summary slot's own x tick label.
GEOMEAN_TICK: str = "Geomean"


def llr40_baseline_label(baseline: str) -> str:
    """The speed-up axis label, naming the baseline by its registry display name."""
    return f"Speed-up over {experiment_tags.names('frameworks').get(baseline, baseline)}"


def thin_speedup_ticks(ax: Axes) -> None:
    """Every other power of two once the axis holds more than :data:`LLR40_MAX_TICKS`, always
    keeping 1x, the reference line the axis label names."""
    positions = [float(tick) for tick in ax.get_yticks()]
    if len(positions) > LLR40_MAX_TICKS:
        positions = [position for position in positions if round(position) % 2 == 0]
    ax.set_yticks(positions)
    ax.set_yticklabels([style.ratio_tick_label(2.0**position) for position in positions], fontsize=LLR40_TEXT_PT)


def legend_handles(rows: Sequence[Row], kernels: Sequence[str]) -> list[Line2D]:
    """One legend entry per row -- its own colour and shape, and the display name :func:`llr40_rows`
    built into ``row.label`` -- plus the undelivered cross
    (:data:`~hpcagent_bench.stats.style.NOT_DELIVERED_LABEL`) only when some kernel draws one. The
    interval method and its n belong to the caption: every whisker on this figure is a 95% interval."""
    handles: list[Line2D] = [
        Line2D(
            [], [], marker=row.marker, linestyle="none", color=row.color or style.MUTED, markersize=7, label=row.label
        )
        for row in rows
    ]
    if any(
        kernel not in row.pending and (kernel not in row.ratios or not row.delivered.get(kernel, True))
        for row in rows
        for kernel in kernels
    ):
        handles.append(
            Line2D(
                [],
                [],
                marker="x",
                linestyle="none",
                color=style.MUTED,
                markeredgewidth=1.2,
                markersize=7,
                label=style.NOT_DELIVERED_LABEL,
            )  # fmt: skip
        )
    if any(row.pending for row in rows):
        handles.append(style.pending_legend_mark(7))
    return handles


def llr40_figure(
    rows: Sequence[Row],
    roster: Sequence[str],
    title: str = "",
    baseline: str = LLR40_BASELINE,
    offset: float = 0.0,
    panel_height_in: float = LLR40_PANEL_HEIGHT_IN,
) -> matplotlib.figure.Figure:
    """The llr-focus40 compiler figure: DaCe's own canon-sweep columns and every model's CPF arm,
    ONE KERNEL AXIS shared by a speed-up panel (LOG2 geometry, ratio-labelled ticks) and, only when
    at least one row spends tokens, a tokens-spent panel below it -- each with a geomean-and-95%-
    interval summary column, values printed, past a dashed separator
    (:func:`kernel_comparison.draw_panel`/:func:`kernel_comparison.draw_summary_column`, reused
    here through their ``transform``/``interval_of`` hooks so the log2 axis and the geomean
    statistic are this module's own while the kernel-axis/dodge/summary-column geometry stays the
    one place that draws it). A compiler-only render spends no tokens, and a panel with nothing to
    draw is omitted rather than left blank.

    Printed at text width with the kernels' ``short-name``
    (:func:`~hpcagent_bench.experiment_tags.kernel_short_display_name`) and the legend below; the
    height follows the longest drawn label and the wrapped legend, measured, never estimated. No
    title unless ``title`` names one: a paper's caption already does.

    Every row of a kernel sits at the SAME x (the optimizers differ by shape), unless ``offset``
    spreads them over that fraction of a kernel slot. The summary slot carries a "Geomean" tick."""
    if not rows:
        raise ValueError("no row to draw")
    style.apply()
    kernels = sorted(roster)
    row_by_key = {row.framework: row for row in rows}
    speedup_series = [
        kernel_comparison.Series(
            row.framework, row.label, row.color or palette.framework_color(row.framework), row.marker, "", "",
            row.ratios, {}, {}, {},
        )
        for row in rows
    ]  # fmt: skip
    token_rows = [row for row in rows if row.tokens]
    has_tokens = bool(token_rows)
    token_series = [
        kernel_comparison.Series(
            row.framework, row.label, row.color or palette.framework_color(row.framework), row.marker, "", "",
            row.tokens, {}, {}, {},
        )
        for row in token_rows
    ]  # fmt: skip
    width = style.DOUBLE_COLUMN_WIDTH
    pitch = (width - LLR40_LEFT_IN - LLR40_RIGHT_IN) / kernel_comparison.panel_slots(len(kernels))
    del pitch  # the mark is sized in points, not by the slot: rows share one x by default
    size = LLR40_MARK_PT**2
    n_panels = 2 if has_tokens else 1
    fig, axes = plt.subplots(n_panels, 1, sharex=True, figsize=(width, 3.0), squeeze=False)
    speedup_ax = axes[0][0]
    token_ax = axes[1][0] if has_tokens else None
    # the 1x line IS the baseline, so it wears the baseline framework's own colour
    style_log2_speedup_y_axis(
        speedup_ax, [v for row in rows for v in row.ratios.values()], palette.framework_color(baseline)
    )
    kernel_comparison.draw_panel(
        speedup_ax, kernels, speedup_series, lambda s: s.values, 1.0, geomean_reducer, "",
        not has_tokens, size,
        range_of=lambda s: (row_by_key[s.key].ratios_low, row_by_key[s.key].ratios_high),
        delivered_of=lambda s: row_by_key[s.key].delivered,
        pending_of=lambda s: row_by_key[s.key].pending,
        interval_of=geomean_interval_of(row_by_key, lambda r: r.ratios),
        transform=log2_change, value_text=kernel_comparison.speedup_value_text, span=offset,
    )  # fmt: skip
    thin_speedup_ticks(speedup_ax)
    speedup_ax.set_ylabel(llr40_baseline_label(baseline), fontsize=LLR40_TEXT_PT, color=style.MUTED)
    if has_tokens and token_ax is not None:
        token_limits = kernel_comparison.token_axis_limits(v for row in token_rows for v in row.tokens.values())
        kernel_comparison.style_token_y_axis(token_ax, token_limits)
        token_ax.tick_params(axis="y", labelsize=LLR40_TEXT_PT)
        token_row_by_key = {row.framework: row for row in token_rows}
        kernel_comparison.draw_panel(
            token_ax, kernels, token_series, lambda s: s.values, token_limits[0], geomean_reducer, "",
            True, size, mark_missing=False,
            interval_of=geomean_interval_of(token_row_by_key, lambda r: r.tokens),
            value_text=style.decade_label, span=offset,
        )  # fmt: skip
        token_ax.set_ylabel("Tokens spent", fontsize=LLR40_TEXT_PT, color=style.MUTED)
    summary_x = kernel_comparison.summary_column_position(len(kernels))[1]
    for ax_row in axes:
        ax_row[0].set_xticks([*range(len(kernels)), summary_x])
        ax_row[0].tick_params(axis="x", length=LLR40_TICK_LEN_PT, width=0.6, color=style.MUTED)
    bottom_ax = axes[-1][0]
    bottom_ax.set_xticklabels(
        [*(experiment_tags.kernel_short_display_name(kernel) for kernel in kernels), GEOMEAN_TICK],
        rotation=90, fontsize=LLR40_TEXT_PT, color=style.INK,
    )  # fmt: skip
    for text in fig.texts + [child for ax in fig.axes for child in ax.texts]:
        text.set_fontsize(min(float(text.get_fontsize()), LLR40_TEXT_PT))
    renderer = fig.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue] -- Agg canvas
    to_inches = fig.dpi_scale_trans.inverted()
    labels_in = max(
        (label.get_window_extent(renderer).transformed(to_inches).height for label in bottom_ax.get_xticklabels()),
        default=0.0,
    )
    handles = legend_handles(rows, kernels)
    for handle in handles:
        handle.set_markersize(LLR40_LEGEND_MARK_PT)
    legend_in = style.legend_below(fig, handles, ncol=len(handles), y=0.0, fontsize=LLR40_LEGEND_PT)
    title_in = 0.0 if not title else 0.22
    bottom_in = labels_in + LLR40_LEGEND_GAP_IN + legend_in + 0.04
    height = title_in + LLR40_TOP_IN + n_panels * panel_height_in + (n_panels - 1) * LLR40_GAP_IN + bottom_in
    fig.set_size_inches(width, height)
    fig.subplots_adjust(
        left=LLR40_LEFT_IN / width,
        right=1.0 - LLR40_RIGHT_IN / width,
        top=1.0 - (title_in + LLR40_TOP_IN) / height,
        bottom=bottom_in / height,
        hspace=LLR40_GAP_IN / panel_height_in,
    )
    for legend in fig.legends:
        legend.set_bbox_to_anchor((0.5, 0.0), transform=fig.transFigure)
        legend.set_loc("lower center")
    if title:
        fig.text(
            0.5, 1.0 - 0.04 / height, title, fontsize=LLR40_LEGEND_PT + 1.0, color=style.INK, ha="center", va="top"
        )
    return fig


def token_summary_table(rows: Sequence[Row]) -> pd.DataFrame:
    """Each token-spending row's geomean spend and its 95% interval -- Rule 4's construction
    applied to a COST directly, never a ratio, since tokens are not one. A row that spends no
    tokens (a canon column) is ABSENT, never entered at zero."""
    records: list[dict[str, object]] = []
    for row in rows:
        if not row.tokens:
            continue
        values = usable_ratios(list(row.tokens.values()), label=row.label)
        if values.size == 0:
            continue
        interval = geomean_ci(values)
        records.append(
            {
                "row": row.label,
                "framework": row.framework,
                "n": interval.n,
                "geomean_tokens": interval.point,
                "geomean_tokens_low": interval.low,
                "geomean_tokens_high": interval.high,
            }
        )
    frame = pd.DataFrame.from_records(records, columns=TOKEN_SUMMARY_COLUMNS)
    return rules.require_interval(frame, "geomean_tokens", "geomean_tokens_low", "geomean_tokens_high")


def llr40_two_row_figure(
    canon_frame: pd.DataFrame,
    observations: pd.DataFrame | None,
    roster: Sequence[str],
    out: pathlib.Path,
    baseline: str = LLR40_BASELINE,
    canon_columns: Sequence[str] = LLR40_CANON_COLUMNS,
    conditions: Sequence[str] = LLR40_CONDITIONS,
    pattern: re.Pattern[str] = kernel_comparison.ARM_PATTERN,
    repeats: population.RepeatPolicy = "latest",
    title: str = "",
    dpi: float = 150.0,
    labels: Mapping[str, str] | None = None,
    offset: float = 0.0,
    mark_pending: bool = False,
    baseline_fallback: str = "",
    panel_height_in: float = LLR40_PANEL_HEIGHT_IN,
) -> pathlib.Path:
    """Build the llr-focus40 compiler rows, write their tables (Rule 4's costs, rules 5/7's
    intervals -- :func:`write_tables`, :func:`token_summary_table`) and render the two-panel
    figure. The ONE function a script calls; ``statistics/plot_llr40_compilers.py`` only parses args.
    ``labels`` renames a row by its framework or arm key (a paper's own name for a column); the
    tables carry the same names the legend does.
    ``dpi`` defaults to 150 -- this figure's own review/paper convention, not
    :func:`~hpcagent_bench.stats.style.save`'s general-purpose 200.
    """
    rows = llr40_rows(
        canon_frame,
        observations,
        roster,
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
    fig = llr40_figure(rows, roster, title, baseline, offset, panel_height_in)
    return style.save(fig, out, formats=("pdf", "png"), fixed=True, dpi=dpi)


def arms_figure(root: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    """The three arms on a common serial denominator, with their tables beside the figure."""
    rows = arm_rows(root)
    write_tables(rows, out)
    return draw(
        rows,
        f"TSVC Kernels, Signed Speed-Up Against Serial gcc {flags.OPT_LEVEL}",
        f"signed relative speed-up vs serial gcc {flags.OPT_LEVEL}\n"
        "$+1$ = 2$\\times$ faster, 0 = no change, $-1$ = 2$\\times$ slower",
        out,
    )


def paired_figure(root: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    """Canonicalized dace against each tool it is paired with, with its tables."""
    rows = paired_rows(root)
    write_tables(rows, out)
    return draw(
        rows,
        "TSVC Kernels, Canonicalized dace Against Each Tool It Is Paired With",
        "signed relative speed-up of dace canon\n$+1$ = 2$\\times$ faster, 0 = no change, $-1$ = 2$\\times$ slower",
        out,
    )


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Signed-change TSVC figures for one sweep directory.")
    parser.add_argument("sweep", type=pathlib.Path, help="directory of <framework>[.rank<N>].csv files")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="directory for the figures and tables")
    args = parser.parse_args(argv)
    # The source directory is in the file name: two sweeps of the same three arms are two
    # measurements, and one silently overwriting the other is how a stale figure reaches a paper.
    name = args.sweep.resolve().name
    out = args.out if args.out is not None else pathlib.Path("reproducibility/canon/figures")
    print(arms_figure(args.sweep, out / f"tsvc_signed_speedup_{name}"))
    print(paired_figure(args.sweep, out / f"tsvc_canon_paired_{name}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
