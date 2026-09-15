# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collect one canon sweep's per-rank CSVs into a SQLite ``canon`` table.

Ported from the reproducibility artifact's ``collect_canon.py``. A canon sweep
(``experiments/submit-canon-llr40.sh``) shards its output by rank (``<column>.rank<N>.csv``,
written by ``experiments/canon_column.sh``) because two ranks appending to one file interleave
partial lines. The shards of one column are disjoint kernel sets, so concatenating them is the
whole merge.

Every row of the output names the run it came from. A speed-up is only meaningful against a
baseline measured on the SAME node under the SAME configuration -- the columns of one sweep share
a job, a node and a preset, and columns from two sweeps do not.

Usage:  python3 scripts/collect_canon.py --run-dir <sweep dir> --db <out.db> [--label <name>]
"""

import argparse
import contextlib
import csv
import pathlib
import sqlite3
import sys

#: Column order on every figure. cc is the baseline the others are divided by with --baseline cc,
#: and it stays in the table (as a constant 1.0 there) so a reader can see it was measured rather
#: than assumed.
COLUMNS: tuple[str, ...] = (
    "cc",
    "cc_autopar",
    "numba",
    "dace_cpu",
    "dace_cpu_canonicalize",
    "dace_gpu",
    "dace_gpu_canonicalize",
)

#: The table this script writes, and the one scripts/plot_canon_speedup.py reads back.
TABLE: str = "canon"

#: Column name -> SQL type. median_ms is nullable: a kernel a column never produced a time for
#: (a crash) is a row with no time, not a zero one.
SCHEMA: tuple[tuple[str, str], ...] = (
    ("run", "TEXT NOT NULL"),
    ("column", "TEXT NOT NULL"),
    ("kernel", "TEXT NOT NULL"),
    ("preset", "TEXT NOT NULL"),
    ("datatype", "TEXT NOT NULL"),
    ("median_ms", "REAL"),
    ("validated", "TEXT NOT NULL"),
)


def rows_for(run_dir: pathlib.Path, column: str, run: str) -> list[dict[str, str]]:
    """Every rank shard of ``column``, as tidy rows. A missing shard is not an error: a column
    that was not part of a sweep simply contributes nothing."""
    out: list[dict[str, str]] = []
    for shard in sorted(run_dir.glob(f"{column}.rank*.csv")):
        with shard.open() as fh:
            for row in csv.DictReader(fh):
                out.append(
                    {
                        "run": run,
                        "column": column,
                        "kernel": row["kernel"],
                        "preset": row.get("preset", ""),
                        "datatype": row.get("datatype", ""),
                        "median_ms": row.get("median_ms", ""),
                        "validated": row.get("validated", ""),
                    }
                )
    return out


def write_db(rows: list[dict[str, str]], db_path: pathlib.Path) -> None:
    """(Re)write ``db_path`` as a fresh SQLite file holding ``rows``, in the order given.

    ``db_path`` is removed first: a stale ``canon`` table from an earlier, differently-shaped
    sweep must never survive under the same file name.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    names = [name for name, _sqltype in SCHEMA]
    columns_sql = ", ".join(f"{name} {sqltype}" for name, sqltype in SCHEMA)
    placeholders = ", ".join(f":{name}" for name in names)
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.execute(f"CREATE TABLE {TABLE} ({columns_sql})")
        conn.executemany(
            f"INSERT INTO {TABLE} ({', '.join(names)}) VALUES ({placeholders})",
            [{**row, "median_ms": float(row["median_ms"]) if row["median_ms"] else None} for row in rows],
        )
        conn.commit()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=pathlib.Path, required=True)
    ap.add_argument("--db", type=pathlib.Path, required=True)
    ap.add_argument("--label", default=None, help="run name recorded in the table (default: dir name)")
    args = ap.parse_args(argv)

    if not args.run_dir.is_dir():
        print(f"no such run directory: {args.run_dir}", file=sys.stderr)
        return 2
    label = args.label or args.run_dir.name
    rows = [row for column in COLUMNS for row in rows_for(args.run_dir, column, label)]
    if not rows:
        print(f"{args.run_dir} holds no <column>.rank*.csv shards", file=sys.stderr)
        return 1

    rows.sort(key=lambda row: (row["column"], row["kernel"]))
    write_db(rows, args.db)
    per = {c: sum(1 for row in rows if row["column"] == c) for c in COLUMNS}
    print(f"{args.db}: {len(rows)} rows  " + "  ".join(f"{c}={n}" for c, n in per.items() if n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
