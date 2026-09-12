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

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import pathlib
import random
import sys
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs] -- pandas ships none
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from hpcagent_bench.flags import OPT_LEVEL
from hpcagent_bench.stats import palette, rules, style
from hpcagent_bench.stats.summary import DEFAULT_CONFIDENCE, geomean_ci, signed_change, usable_ratios

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
    """One drawn row: its ratios per kernel, the costs behind them, and what it excluded."""

    framework: str
    label: str
    ratios: dict[str, float]
    numerator_ms: dict[str, float]
    denominator_ms: dict[str, float]
    excluded: str


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
    return style.save(fig, stem)


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


def arms_figure(root: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    """The three arms on a common serial denominator, with their tables beside the figure."""
    rows = arm_rows(root)
    write_tables(rows, out)
    return draw(
        rows,
        f"TSVC Kernels, Signed Speed-Up Against Serial gcc {OPT_LEVEL}",
        f"signed relative speed-up vs serial gcc {OPT_LEVEL}\n$+1$ = 2$\\times$ faster, 0 = no change, $-1$ = 2$\\times$ slower",
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
