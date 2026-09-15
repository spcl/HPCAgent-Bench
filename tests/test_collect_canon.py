# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collecting a canon sweep's per-rank CSVs into the ``canon`` table plot_canon_speedup.py reads.

The rows below are copied verbatim from the committed reproducibility-artifact table
(``paper_artifacts/experiments/canon/data/canon_llr40.csv``, sweep 631260), so a synthetic sweep
proves this collector reproduces that table's own numbers, not numbers that merely look plausible.
"""

import contextlib
import csv
import importlib.util
import pathlib
import sqlite3
import sys

from hpcagent_bench import paths

SPEC = importlib.util.spec_from_file_location("collect_canon", paths.ROOT / "scripts" / "collect_canon.py")
collect_canon = importlib.util.module_from_spec(SPEC)
# Registered BEFORE exec: a module loaded by path alone has no entry of its own to resolve through.
sys.modules[SPEC.name] = collect_canon
SPEC.loader.exec_module(collect_canon)

#: canon_llr40.csv, sweep canon-llr40-631260, row for (cc, wf_triangular).
CC_WF_TRIANGULAR = {
    "run": "canon-llr40-631260",
    "column": "cc",
    "kernel": "wf_triangular",
    "preset": "fuzzed",
    "datatype": "float64",
    "median_ms": 165.9901,
    "validated": "True",
}
#: canon_llr40.csv, sweep canon-llr40-631260, row for (numba, argmax_with_index).
NUMBA_ARGMAX = {
    "run": "canon-llr40-631260",
    "column": "numba",
    "kernel": "argmax_with_index",
    "preset": "fuzzed",
    "datatype": "float64",
    "median_ms": 364.1676,
    "validated": "True",
}
#: canon_llr40.csv, sweep canon-llr40-631260, row for (dace_cpu, fuse_diamond): it crashed, so the
#: committed table carries the row with an empty time and an empty validated flag.
DACE_CPU_FUSE_DIAMOND_CRASHED = {
    "run": "canon-llr40-631260",
    "column": "dace_cpu",
    "kernel": "fuse_diamond",
    "preset": "fuzzed",
    "datatype": "float64",
    "median_ms": None,
    "validated": "",
}


def write_shard(run_dir: pathlib.Path, column: str, rank: int, rows: list[dict]) -> None:
    shard = run_dir / f"{column}.rank{rank}.csv"
    with shard.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["kernel", "preset", "datatype", "median_ms", "validated"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "kernel": row["kernel"],
                    "preset": row["preset"],
                    "datatype": row["datatype"],
                    "median_ms": "" if row["median_ms"] is None else row["median_ms"],
                    "validated": row["validated"],
                }
            )


def read_db(db_path: pathlib.Path) -> list[dict]:
    with contextlib.closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(f"SELECT * FROM {collect_canon.TABLE} ORDER BY rowid")]


def test_a_synthetic_sweep_reproduces_the_committed_tables_own_rows(tmp_path: pathlib.Path) -> None:
    """cc/wf_triangular and numba/argmax_with_index are copied from the committed canon_llr40.csv
    (sweep 631260); the collector must reproduce them field for field, not just a plausible row."""
    run_dir = tmp_path / "sweep"
    run_dir.mkdir()
    write_shard(run_dir, "cc", 0, [CC_WF_TRIANGULAR])
    write_shard(run_dir, "numba", 0, [NUMBA_ARGMAX])
    db_path = tmp_path / "canon.db"

    rc = collect_canon.main(["--run-dir", str(run_dir), "--db", str(db_path), "--label", "canon-llr40-631260"])

    assert rc == 0
    rows = read_db(db_path)
    got = {(row["column"], row["kernel"]): row for row in rows}
    assert got[("cc", "wf_triangular")] == CC_WF_TRIANGULAR
    assert got[("numba", "argmax_with_index")] == NUMBA_ARGMAX


def test_a_crashed_kernel_keeps_its_row_with_no_time_not_a_zero_one(tmp_path: pathlib.Path) -> None:
    """fuse_diamond crashes under dace_cpu in the real sweep, and the committed table still carries
    the row -- an empty time and an empty validated flag, never a dropped row or a 0.0 time."""
    run_dir = tmp_path / "sweep"
    run_dir.mkdir()
    write_shard(run_dir, "dace_cpu", 0, [DACE_CPU_FUSE_DIAMOND_CRASHED])
    db_path = tmp_path / "canon.db"

    collect_canon.main(["--run-dir", str(run_dir), "--db", str(db_path)])

    rows = read_db(db_path)
    assert rows[0]["median_ms"] is None
    assert rows[0]["validated"] == ""


def test_rows_are_ordered_by_column_then_kernel_regardless_of_shard_write_order(tmp_path: pathlib.Path) -> None:
    """Two ranks race to write their shards; the table's row order must not depend on who won,
    since a rerun that happens to schedule ranks differently would otherwise reorder the table."""
    run_dir = tmp_path / "sweep"
    run_dir.mkdir()
    write_shard(run_dir, "numba", 1, [{**NUMBA_ARGMAX, "kernel": "zzz_last"}])
    write_shard(run_dir, "numba", 0, [NUMBA_ARGMAX])
    write_shard(run_dir, "cc", 0, [CC_WF_TRIANGULAR])
    db_path = tmp_path / "canon.db"

    collect_canon.main(["--run-dir", str(run_dir), "--db", str(db_path)])

    rows = read_db(db_path)
    assert [(row["column"], row["kernel"]) for row in rows] == [
        ("cc", "wf_triangular"),
        ("numba", "argmax_with_index"),
        ("numba", "zzz_last"),
    ]


def test_a_stale_table_from_an_earlier_differently_shaped_sweep_does_not_survive(tmp_path: pathlib.Path) -> None:
    """``--db`` pointed at an existing file must not silently append to, or fail against, a
    ``canon`` table a previous run with a different schema left behind."""
    db_path = tmp_path / "canon.db"
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.execute("CREATE TABLE canon (something_else TEXT)")
        conn.execute("INSERT INTO canon VALUES ('stale')")
        conn.commit()
    run_dir = tmp_path / "sweep"
    run_dir.mkdir()
    write_shard(run_dir, "cc", 0, [CC_WF_TRIANGULAR])

    collect_canon.main(["--run-dir", str(run_dir), "--db", str(db_path)])

    rows = read_db(db_path)
    assert len(rows) == 1
    assert rows[0]["kernel"] == "wf_triangular"
