# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The SDFG parallelism taxonomy, stacked bucket shares per DaCe column.

Reads the ``kernel_metrics`` table scripts/collect_parallelism.py writes (or the full table --
every row is filtered to ``parallelism.*`` here too), rebuilds one :class:`ParallelismRecord` per
kernel through :func:`hpcagent_bench.metrics.parallelism.read_records`, and draws one stacked bar
per column: the SEQUENTIAL_BUCKETS/PARALLEL_BUCKETS/scan taxonomy folded into four shares (parallel,
scan, timestep, residual) that sum to 100% of the column's loop-level constructs, each segment
labelled with its raw count. ``libnode`` sits outside that 100% (a library-node count, not a loop)
and is annotated beside the bar instead. Per-kernel parallelized/fully-parallelized counts, under
the default rate definition, are annotated under each bar.

The table (``<stem>.csv``) carries every named rate definition (:data:`RATE_DEFINITIONS`) for every
drawn column, spelling out numerator and denominator terms AND their raw counts -- a rate is never
written down as a bare percentage.

Usage:  python3 scripts/plot_parallelism.py --db parallelism.db --out figures
"""

import argparse
import csv
import pathlib
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

from hpcagent_bench.experiments import read_table
from hpcagent_bench.metrics import parallelism
from hpcagent_bench.stats import palette, style

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

#: The table scripts/collect_parallelism.py writes.
TABLE: str = "kernel_metrics"

#: Stack segments, bottom to top, and their colour. Not a palette.py ENTITY (framework/model/
#: language) -- a fixed categorical scheme for the taxonomy's own groups, the same precedent as
#: plot_canon_speedup.py's MEDIAN_HUE/GEOMEAN_HUE for a statistic rather than an entity.
SEGMENTS: tuple[tuple[str, str], ...] = (
    ("parallel", "#2f8f55"),
    ("scan", "#8a6fd4"),
    ("timestep", "#c9a227"),
    ("residual", "#c0392b"),
)

#: Columns drawn when --columns is not given: canonicalized first (the paper's headline pipeline).
DEFAULT_COLUMNS: tuple[str, ...] = ("dace_cpu_canonicalize", "dace_cpu")

TABLE_FIELDS: tuple[str, ...] = (
    "column",
    "rate",
    "default",
    "numerator_terms",
    "numerator",
    "denominator_terms",
    "denominator",
    "value",
    "parallelized",
    "fully_parallelized",
    "neither",
    "total_kernels",
)


def segment_counts(agg: dict[str, int]) -> dict[str, int]:
    """``agg``'s raw bucket totals (:func:`hpcagent_bench.metrics.parallelism.totals`), folded into
    the four stack segments. These four sum to ``total`` exactly: PARALLEL_BUCKETS + SEQUENTIAL_
    BUCKETS + ``scan`` is every bucket in :data:`hpcagent_bench.metrics.parallelism.BUCKETS`.
    """
    return {
        "parallel": sum(agg.get(bucket, 0) for bucket in parallelism.PARALLEL_BUCKETS),
        "scan": agg.get("scan", 0),
        "timestep": agg.get("timestep", 0),
        "residual": agg.get("residual", 0),
    }


def draw(
    columns: Sequence[str], by_column: dict[str, dict[str, parallelism.ParallelismRecord]], double_column: bool
) -> "tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]":
    import matplotlib.patches
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    style.apply()
    height = (0.85 if double_column else 1.25) * len(columns) + 1.5
    fig, ax = plt.subplots(figsize=(style.DOUBLE_COLUMN_WIDTH if double_column else 7.2, height))
    ypos = list(range(len(columns)))[::-1]
    bar_height = 0.5

    for y, column in zip(ypos, columns, strict=True):
        records = list(by_column[column].values())
        agg = parallelism.totals(records)
        seg = segment_counts(agg)
        total = sum(seg.values())
        left = 0.0
        for name, color in SEGMENTS:
            share = seg[name] / total if total else 0.0
            ax.barh(y, share, left=left, height=bar_height, color=color, zorder=2)
            if seg[name]:
                ax.text(
                    left + share / 2.0,
                    y,
                    str(seg[name]),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white",
                    zorder=3,
                )
            left += share
        counts = parallelism.benchmark_counts(records, parallelism.RATE_DEFINITIONS[parallelism.DEFAULT_RATE])
        ax.text(
            1.03,
            y,
            f"+{agg.get('libnode', 0)} libnode",
            va="center",
            ha="left",
            fontsize=7.5,
            color=style.MUTED,
        )
        ax.text(
            0.0,
            y - bar_height * 0.85,
            f"parallelized {counts.parallelized}/{counts.total}   fully parallelized {counts.fully_parallelized}/{counts.total}",
            va="top",
            ha="left",
            fontsize=7.0,
            color=style.MUTED,
        )

    ax.set_yticks(ypos)
    ax.set_yticklabels(columns, fontsize=9.0)
    for tick, column in zip(ax.get_yticklabels(), columns, strict=True):
        tick.set_color(palette.framework_color(column))
    ax.set_ylim(-1.0, len(columns) - 0.3)
    ax.set_xlim(0.0, 1.34)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("share of loop-level constructs (map/reduce/scan/loop)", color=style.MUTED, fontsize=9.0)
    style.despine(ax, keep=("bottom",))
    ax.tick_params(axis="both", length=0, colors=style.MUTED)
    ax.grid(axis="x", which="major", color=style.RULE, linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    handles = [matplotlib.patches.Patch(facecolor=color, label=name) for name, color in SEGMENTS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=4, frameon=False, fontsize=8.0)
    return fig, ax


def write_table(
    columns: Sequence[str], by_column: dict[str, dict[str, parallelism.ParallelismRecord]], path: pathlib.Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(TABLE_FIELDS)
        for column in columns:
            records = list(by_column[column].values())
            agg = parallelism.totals(records)
            for name, definition in parallelism.RATE_DEFINITIONS.items():
                computed = parallelism.rate(agg, definition)
                counts = parallelism.benchmark_counts(records, definition)
                value = "" if computed["value"] is None else f"{computed['value']:.6f}"
                writer.writerow(
                    [
                        column,
                        name,
                        name == parallelism.DEFAULT_RATE,
                        computed["numerator_terms"],
                        computed["numerator"],
                        computed["denominator_terms"],
                        computed["denominator"],
                        value,
                        counts.parallelized,
                        counts.fully_parallelized,
                        counts.neither,
                        counts.total,
                    ]
                )


def run(db: pathlib.Path, out_dir: pathlib.Path, columns: Sequence[str], double_column: bool, stem: str) -> int:
    frame = read_table(db, TABLE)
    by_column = parallelism.read_records(frame)
    present = [column for column in columns if by_column.get(column)]
    if not present:
        print(f"{db} holds none of {list(columns)}", file=sys.stderr)
        return 1

    fig, _ax = draw(present, by_column, double_column)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = style.save(fig, out_dir / stem)
    table_path = out_stem.with_suffix(".csv")
    write_table(present, by_column, table_path)

    print(f"{out_stem}.pdf / .png")
    print(f"{table_path}")
    for column in present:
        records = list(by_column[column].values())
        counts = parallelism.benchmark_counts(records, parallelism.RATE_DEFINITIONS[parallelism.DEFAULT_RATE])
        print(f"  {column}: {len(records)} kernels, parallelized {counts.parallelized}/{counts.total}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures"))
    ap.add_argument("--columns", default=",".join(DEFAULT_COLUMNS), help="comma-separated column names")
    ap.add_argument("--double-column", action="store_true", help="compact A4 insert, DOUBLE_COLUMN_WIDTH wide")
    ap.add_argument("--stem", default="parallelism", help="output filename stem (default: parallelism)")
    args = ap.parse_args(argv)
    columns = [c for c in args.columns.split(",") if c]
    return run(args.db, args.out, columns, args.double_column, args.stem)


if __name__ == "__main__":
    sys.exit(main())
