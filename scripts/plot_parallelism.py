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
    # Rotated column names are the only thing below the axis (rule one puts the value on Y), so
    # the bottom margin has to grow with the longest one, not with a fixed guess -- and the axes
    # need enough of their OWN height for the equally rotated y label, or it prints truncated.
    longest = max((len(c) for c in columns), default=8)
    left_in, top_in, axes_min_in = 0.85, 0.25, 4.4
    bottom_in = longest * style.TICK_PT * 0.6 / 72.0 + 0.9
    width = style.DOUBLE_COLUMN_WIDTH if double_column else max(4.4, 2.2 * len(columns) + 1.8)
    height = axes_min_in + bottom_in + top_in
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=left_in / width, right=0.97, top=1.0 - top_in / height, bottom=bottom_in / height)
    xs = list(range(len(columns)))
    bar_width = 0.5

    for x, column in zip(xs, columns, strict=True):
        records = list(by_column[column].values())
        agg = parallelism.totals(records)
        seg = segment_counts(agg)
        total = sum(seg.values())
        bottom = 0.0
        for name, color in SEGMENTS:
            share = seg[name] / total if total else 0.0
            ax.bar(x, share, bottom=bottom, width=bar_width, color=color, zorder=2)
            if seg[name]:
                ax.text(
                    x,
                    bottom + share / 2.0,
                    str(seg[name]),
                    ha="center",
                    va="center",
                    fontsize=style.ANNOTATION_PT,
                    color="white",
                    zorder=3,
                )
            bottom += share
        counts = parallelism.benchmark_counts(records, parallelism.RATE_DEFINITIONS[parallelism.DEFAULT_RATE])
        caption = (
            f"+{agg.get('libnode', 0)} libnode\n"
            f"Parallelized {counts.parallelized}/{counts.total}\n"
            f"Fully Parallelized {counts.fully_parallelized}/{counts.total}"
        )
        # Above the bar, never below: the column NAME is the only thing below the axis, so its
        # rotated tick label never collides with this three-line caption.
        ax.text(x, 1.04, caption, va="bottom", ha="center", fontsize=style.ANNOTATION_PT, color=style.MUTED)

    ax.set_xticks(xs)
    ax.set_xticklabels(columns, fontsize=style.TICK_PT, rotation=90)
    for tick, column in zip(ax.get_xticklabels(), columns, strict=True):
        tick.set_color(palette.framework_color(column))
    ax.set_xlim(-0.7, len(columns) - 0.3)
    ax.set_ylim(0.0, 1.62)
    style.value_axis(ax, "y", major=True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Share of Loop-Level Constructs", color=style.MUTED, fontsize=style.ANNOTATION_PT)
    style.despine(ax, keep=("left",))
    ax.tick_params(axis="x", length=0, colors=style.MUTED)

    handles = [matplotlib.patches.Patch(facecolor=color, label=name) for name, color in SEGMENTS]
    style.legend_below(fig, handles, ncol=4, y=0.02)
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
