#!/usr/bin/env python3
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fold ONE column's canon CSV shards into the persistent, cross-run canon results DB.

``scripts/collect_canon.py`` REBUILDS a fresh ``--db`` from a whole sweep's directory -- the tool
the paper's reproducibility repos call once a campaign is done, and it must keep doing exactly
that for them. This is a different tool for a different moment: the per-job step
``experiments/canon_column.sh`` runs at the END OF EACH COLUMN'S Slurm job, while the sweep is
still in progress and other columns' jobs may still be writing beside this one, so the tiny
per-kernel score row survives after the run's DaCe build tree (``dacecache-<column>[_rank<N>]``,
routinely the bulk of a canon work dir) is deleted.

APPEND-only against a DB every column of every campaign shares: a ``(run, column, kernel, preset,
datatype)`` row is ``INSERT OR REPLACE``, so re-running this script after a partial write (or a
requeued job) never doubles a row, and rebuilding it here (as collect_canon.py does) would erase
every other column's rows the moment the first column's job finished.

    python3 scripts/merge_canon_results.py --run-dir <out_root> --column <col> --run <label> \\
        --db <persistent db> [--expected N]

``--expected``, when given, is an INDEPENDENTLY counted row total (the caller's own ``wc -l`` over
the same shards) that must equal what this script's own CSV parse finds -- two different counts of
the same claim, not one number trusted twice. Prints ``merged <n> rows (<m> expected)`` and exits 0
only when they agree; the caller (canon_column.sh) treats any other outcome as "do not delete this
column's work-dir artifacts."
"""

import argparse
import contextlib
import csv
import pathlib
import sqlite3
import sys

#: Mirrors scripts/collect_canon.py's TABLE/SCHEMA (same tidy shape, same reader contract) --
#: kept as a literal copy rather than an import so this script has no import-path dependency on
#: that one; the two are read by different consumers (a reproducibility repo's whole-sweep rebuild
#: vs. this per-job append) and must be free to diverge if either grows a column the other has no
#: use for.
TABLE = "canon"
SCHEMA: tuple[tuple[str, str], ...] = (
    ("run", "TEXT NOT NULL"),
    ("column", "TEXT NOT NULL"),
    ("kernel", "TEXT NOT NULL"),
    ("preset", "TEXT NOT NULL"),
    ("datatype", "TEXT NOT NULL"),
    ("median_ms", "REAL"),
    ("validated", "TEXT NOT NULL"),
)


def rows_for(run_dir: pathlib.Path, column: str, run: str) -> list[dict[str, object]]:
    """Every rank shard of ``column`` in ``run_dir``, as tidy rows. A column with no shard (a rank
    whose kernel share was empty writes none at all -- see canon_column.sh's zero-kernel-rank
    guard) yields an empty list, which is a fact, not an error."""
    out: list[dict[str, object]] = []
    for shard in sorted(run_dir.glob(f"{column}.rank*.csv")):
        with shard.open(newline="") as fh:
            for row in csv.DictReader(fh):
                out.append(
                    {
                        "run": run,
                        "column": column,
                        "kernel": row["kernel"],
                        "preset": row.get("preset", ""),
                        "datatype": row.get("datatype", ""),
                        "median_ms": float(row["median_ms"]) if row.get("median_ms") else None,
                        "validated": row.get("validated", ""),
                    }
                )
    return out


def merge(rows: list[dict[str, object]], db_path: pathlib.Path) -> int:
    """(Re)ensure ``canon`` at ``db_path`` and ``INSERT OR REPLACE`` every row of ``rows``.

    The unique index is what makes this idempotent: a second call with the same rows (a retried
    merge after a transient failure) overwrites the same primary key instead of duplicating it, so
    calling this twice for one column's shards is always safe."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    names = [name for name, _sqltype in SCHEMA]
    columns_sql = ", ".join(f"{name} {sqltype}" for name, sqltype in SCHEMA)
    placeholders = ", ".join(f":{name}" for name in names)
    with contextlib.closing(sqlite3.connect(db_path, timeout=30.0)) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} ({columns_sql})")
        conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS ux_{TABLE}_row "
            f"ON {TABLE}(run, column, kernel, preset, datatype)"
        )
        conn.executemany(
            f"INSERT OR REPLACE INTO {TABLE} ({', '.join(names)}) VALUES ({placeholders})",
            rows,
        )
        conn.commit()
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=pathlib.Path, required=True)
    ap.add_argument("--column", required=True)
    ap.add_argument("--run", required=True, help="the run/campaign label recorded on every row")
    ap.add_argument("--db", type=pathlib.Path, required=True)
    ap.add_argument(
        "--expected",
        type=int,
        default=None,
        help="an independently counted row total this script's own CSV parse must match",
    )
    args = ap.parse_args(argv)

    if not args.run_dir.is_dir():
        print(f"no such run directory: {args.run_dir}", file=sys.stderr)
        return 2

    rows = rows_for(args.run_dir, args.column, args.run)
    found = len(rows)
    if args.expected is not None and args.expected != found:
        print(
            f"merged 0 rows ({args.expected} expected, {found} found in "
            f"{args.column}.rank*.csv under {args.run_dir}): counts disagree, not merging",
            file=sys.stderr,
        )
        return 1

    merged = merge(rows, args.db) if rows else 0
    print(f"merged {merged} rows ({found if args.expected is None else args.expected} expected) into {args.db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
