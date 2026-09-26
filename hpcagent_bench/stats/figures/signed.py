# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The signed-change figures: one row per arm, and the paired comparison of two tools.

TWO QUESTIONS, TWO FIGURES, ONE READER. :func:`arms_figure` puts several arms on a COMMON
denominator and answers "how fast is each arm". :func:`paired_figure` answers the different
question an ablation is about: on a kernel BOTH tools compiled, which is faster -- so its
denominator is the other TOOL, per kernel, and each row is a paired comparison rather than an
independent arm. Dividing two geomeans taken over different kernel sets is not a speedup of
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
from collections.abc import Collection, Mapping, Sequence

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs] -- pandas ships none
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from hpcagent_bench import experiment_tags, flags
from hpcagent_bench.stats import canon, palette, population, rules, style
from hpcagent_bench.stats.figures import llr40_arms, per_kernel
from hpcagent_bench.stats.summary import DEFAULT_CONFIDENCE, geomean_ci, signed_change, usable_ratios

#: Framework -> the name a reader knows it by. Insertion order is the order on the axis.
ARMS: dict[str, str] = {
    "dace_cpu_canonicalize": "dace canon",
    "dace_cpu": "dace main",
    "cc_llvm_autopar": "llvm + polly",
}

#: The speedup DENOMINATOR: a SERIAL optimizing compile, not the interpreted reference.
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

    ``color``/``marker`` and the trailing five fields are used only by the llr-focus40 compiler
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


def solved_ratios(row: Row) -> dict[str, float]:
    """``row``'s ratios over the kernels it SOLVED: a compiler's 1x placeholder (``delivered``
    False) is a row of the per-kernel table and a crossed mark on the figure, but no measurement,
    so no summary takes it. A row with no ``delivered`` flags (every TSVC row, every
    agent row) solved every kernel it has a ratio for."""
    return {kernel: ratio for kernel, ratio in row.ratios.items() if row.delivered.get(kernel, True)}


def summary_table(rows: Sequence[Row]) -> pd.DataFrame:
    """One record per drawn row, over the kernels it solved (:func:`solved_ratios`): the geomean,
    its interval, the median, n and the sign test -- the geomean and interval the figure's summary
    slot draws, so the two cannot disagree.

    :func:`hpcagent_bench.stats.rules.require_interval` gates it, so a row that reached the figure
    without an interval fails here (Rules 5 and 7) instead of being drawn as a bare point.
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


#: The signed figure's mark and line sizes, in points: a cloud kernel's AREA and its key entry's
#: size, the geomean mark, its interval's weight and cap, and the median tick and zero line.
CLOUD_MARK_AREA: float = 11.0
CLOUD_KEY_MARK_PT: float = 3.5
GEOMEAN_MARK_PT: float = 5.5
INTERVAL_LINE_WIDTH: float = 1.1
INTERVAL_CAP_PT: float = 3.5
RULE_LINE_WIDTH: float = 1.0


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
        s=CLOUD_MARK_AREA,
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
        markersize=GEOMEAN_MARK_PT,
        color=color,
        ecolor=style.REFERENCE,
        elinewidth=INTERVAL_LINE_WIDTH,
        capsize=INTERVAL_CAP_PT,
        zorder=4,
    )
    middle = signed_change(float(np.median(values)))
    ax.plot(  # pyright: ignore[reportUnknownMemberType]
        [middle, middle], [index + 0.10, index + 0.38], color=style.INK, linewidth=RULE_LINE_WIDTH, zorder=3
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
    ax.axvline(  # pyright: ignore[reportUnknownMemberType]
        0.0, color=style.REFERENCE, linewidth=RULE_LINE_WIDTH, zorder=1
    )
    ax.margins(x=0.08)
    style.value_axis(ax, axis="x")
    handles: list[Line2D] = [
        Line2D([], [], marker="o", linestyle="none", color=style.MUTED, markersize=CLOUD_KEY_MARK_PT, alpha=0.5),
        Line2D([], [], marker="o", linestyle="none", color=style.MUTED, markersize=GEOMEAN_MARK_PT),
        Line2D([], [], color=style.INK, linewidth=RULE_LINE_WIDTH),
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
            f"Every speedup here is a ratio against it, so there is nothing to plot without it."
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
#: (:data:`llr40_arms.CANON_BASELINE`).
LLR40_BASELINE: str = llr40_arms.CANON_BASELINE

#: The two canon-sweep columns this figure draws as their OWN rows, in draw order: DaCe's
#: parallel-CPU backend, then its canonicalizing pass over the same backend. The LIBRARY default --
#: a caller wanting the polyhedral compiler baselines too (Pluto, PPCG-on-AMD) passes its own
#: ``canon_columns`` (as :data:`statistics.plot_llr40_compilers`'s CLI default does).
LLR40_CANON_COLUMNS: tuple[str, ...] = ("dace_cpu", "dace_cpu_canonicalize")

#: The two CPF conditions this figure draws, per model -- never the no-packet control, which
#: answers a different question.
LLR40_CONDITIONS: tuple[str, ...] = ("cpf", "cpfsrc")

#: Column order of the emitted token summary table.
TOKEN_SUMMARY_COLUMNS: tuple[str, ...] = (
    "row", "framework", "n", "gm_tokens", "gm_tokens_low", "gm_tokens_high",
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
    (:data:`llr40_arms.CANON_COLUMN` sweeps 40); restricting to ``roster`` keeps the summary
    column from geomeaning a population the panel never drew.

    A roster kernel ``column`` produced no validated result for -- declined, crashed, or never
    attempted -- is FILLED at 1x, never dropped (:func:`hpcagent_bench.stats.canon.roster_speedups`):
    a compiler baseline that cannot handle a kernel is no different from
    an agent that never delivered one, and ``delivered`` flags it the same way
    :data:`~hpcagent_bench.stats.population.DELIVERED_COLUMN` flags that placeholder for an agent
    row, so ``llr40_figure`` draws it crossed at 1x under the one existing convention. The figure's
    summary slot leaves it out (solved kernels only, :func:`per_kernel.kernel_medians`); the
    ``-summary.csv`` :func:`summary_table` writes leaves it out the same way.

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
    pending = (
        unattempted_kernels(canon_frame, column, kernels, baseline, baseline_fallback) if mark_pending else frozenset()
    )
    ratios, delivered = canon.roster_speedups(times, baseline, column, [k for k in kernels if k not in pending])
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
    optimizer = experiment_tags.canonical("optimizers", column)
    standalone = experiment_tags.names("optimizers")
    return standalone[optimizer] if optimizer in standalone else experiment_tags.names("frameworks").get(column, column)


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
    repeats: population.RepeatPolicy = population.RepeatPolicy.LATEST,
    pending: frozenset[str] = frozenset(),
) -> Row:
    """One CPF arm's row, restricted to ``roster``: its final answer per kernel (Rule 4's costs
    behind the ratio), plus each kernel's OWN confidence interval over every graded episode that
    kernel ran (rules 5/7) -- the geomean of that kernel's repetitions, degenerate to a point below
    two samples (:func:`~hpcagent_bench.stats.summary.geomean_ci`)."""
    subset = frame.loc[frame["arm"].astype(str) == arm]
    answers = population.kernel_answers(subset, repeats=repeats, policy=population.KernelPolicy.SOLVED)
    kernels = set(roster)
    ratios, numerator_ms, denominator_ms = answer_ratios(answers, kernels)
    ratios_low, ratios_high = kernel_intervals(subset, ratios.keys(), arm)
    raw_tokens, tokens_low, tokens_high = llr40_arms.arm_tokens(subset, arm, repeats)
    del tokens_low, tokens_high  # under "latest" both are empty; a repeat's own range is not this figure's concern
    tokens = {k: v for k, v in raw_tokens.items() if k in kernels}
    label = f"{experiment_tags.model_name(model)} - {llr40_arms.condition_label(condition)}"
    return Row(
        arm, label, ratios, numerator_ms, denominator_ms, pending_note(pending),
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
    subset: pd.DataFrame, kernels: Collection[str], arm: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Each of ``kernels``' geomean-speedup CI over every graded episode of ``arm``, as (low, high)."""
    ratios_low: dict[str, float] = {}
    ratios_high: dict[str, float] = {}
    graded = subset.loc[subset["record"] == "submission"] if "record" in subset.columns else subset
    episodes = population.graded_episode_rows(graded, population.SUBMISSION_ORDER)
    if episodes.empty:
        return ratios_low, ratios_high
    for kernel, group in episodes.groupby("benchmark"):
        kernel = str(kernel)
        if kernel not in kernels:
            continue
        values = usable_ratios(group["speedup"].tolist(), label=f"{arm}@{kernel}")
        if values.size == 0:
            continue
        interval = geomean_ci(values)
        ratios_low[kernel] = interval.low
        ratios_high[kernel] = interval.high
    return ratios_low, ratios_high


def llr40_rows(
    canon_frame: pd.DataFrame,
    observations: pd.DataFrame | None,
    roster: Sequence[str],
    baseline: str = LLR40_BASELINE,
    canon_columns: Sequence[str] = LLR40_CANON_COLUMNS,
    conditions: Sequence[str] = LLR40_CONDITIONS,
    pattern: re.Pattern[str] = llr40_arms.ARM_PATTERN,
    repeats: population.RepeatPolicy = population.RepeatPolicy.LATEST,
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
    candidates = llr40_arms.candidate_arms(observations, pattern)
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
        for arm in sorted(by_model[model], key=lambda a: llr40_arms.rank_condition(candidates[a][1])):
            model_tag, condition = candidates[arm]
            served = set(frame.loc[frame["arm"].astype(str) == arm, "benchmark"].astype(str))
            pending = frozenset(k for k in roster if k not in served)
            rows.append(agent_kernel_row(frame, arm, model_tag, condition, roster, repeats, pending))
    return rows


#: A panel's height in the llr-focus40 compiler figure, inches: what 40 kernels need to read at the
#: text width the figure prints at, not what the canvas can spare.
LLR40_PANEL_HEIGHT_IN: float = 1.5


def llr40_baseline_label(baseline: str) -> str:
    """The speedup axis label, naming the baseline by its registry display name."""
    return f"Speedup over {experiment_tags.names('frameworks').get(baseline, baseline)}"


def row_color(row: Row) -> str:
    """The colour ``row`` draws in: its own, or its framework's for a row that sets none."""
    return row.color or palette.framework_color(row.framework)


def llr40_metrics(
    rows: Sequence[Row], roster: Sequence[str], baseline: str = LLR40_BASELINE
) -> list[per_kernel.Metric]:
    """The compiler figure's panels over ``sorted(roster)``, as :mod:`per_kernel` draws them.

    Speedup for every row: a kernel's own repeat interval (SC15 rules 5/7) as its whisker, a
    compiler's 1x placeholder crossed (``Row.delivered``), an unanswered agent kernel filled at 1x and
    crossed, a pending kernel as "?"; the 1x line wears the baseline's own colour, since it IS the
    baseline. Tokens spent only when some row spends any -- a compiler-only render has nothing to
    put there, and an empty panel is omitted rather than left blank; a row that spends none (a canon
    column) keeps its dodge offset and summary slot on it and draws nothing.
    """
    kernels = sorted(roster)
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
    roster: Sequence[str],
    title: str = "",
    baseline: str = LLR40_BASELINE,
    offset: float = 0.0,
    panel_height_in: float = LLR40_PANEL_HEIGHT_IN,
) -> matplotlib.figure.Figure:
    """The llr-focus40 compiler figure: DaCe's own canon-sweep columns and every model's CPF arm on
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
    metrics = llr40_metrics(rows, roster, baseline)
    return per_kernel.figure_panels(
        metrics,
        sorted(roster),
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
    roster: Sequence[str],
    out: pathlib.Path,
    baseline: str = LLR40_BASELINE,
    canon_columns: Sequence[str] = LLR40_CANON_COLUMNS,
    conditions: Sequence[str] = LLR40_CONDITIONS,
    pattern: re.Pattern[str] = llr40_arms.ARM_PATTERN,
    repeats: population.RepeatPolicy = population.RepeatPolicy.LATEST,
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
        f"TSVC Kernels, Signed Speedup Against Serial gcc {flags.OPT_LEVEL}",
        f"signed relative speedup vs serial gcc {flags.OPT_LEVEL}\n"
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
        "signed relative speedup of dace canon\n$+1$ = 2$\\times$ faster, 0 = no change, $-1$ = 2$\\times$ slower",
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
