# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Folding one canon column's CSV shards into the persistent, cross-job canon results DB
(scripts/merge_canon_results.py) -- the step canon_column.sh runs before it may delete a column's
work-dir artifacts (its DaCe build tree and per-rank shard DBs).

Unlike scripts/collect_canon.py (a whole-sweep REBUILD for the reproducibility repos), this tool
must never erase another column's already-merged rows, since sibling columns' jobs are still
writing while any one column's job finalizes -- that is the property every test below is really
checking, in one shape or another.
"""

import contextlib
import csv
import importlib.util
import pathlib
import sqlite3
import sys

from hpcagent_bench import paths

SPEC = importlib.util.spec_from_file_location("merge_canon_results", paths.ROOT / "scripts" / "merge_canon_results.py")
merge_canon_results = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = merge_canon_results
SPEC.loader.exec_module(merge_canon_results)

ROW = {
    "kernel": "wf_triangular",
    "preset": "fuzzed",
    "datatype": "float64",
    "median_ms": "165.9901",
    "validated": "True",
}


def write_shard(run_dir: pathlib.Path, column: str, rank: int, rows: list[dict]) -> None:
    shard = run_dir / f"{column}.rank{rank}.csv"
    with shard.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["kernel", "preset", "datatype", "median_ms", "validated"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows(db_path: pathlib.Path) -> list[dict]:
    with contextlib.closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM canon ORDER BY rowid")]


def test_a_columns_rows_land_in_the_shared_db_it_did_not_create(tmp_path: pathlib.Path) -> None:
    run_dir = tmp_path / "sweep"
    run_dir.mkdir()
    write_shard(run_dir, "cc", 0, [ROW])
    db_path = tmp_path / "canon.db"

    rc = merge_canon_results.main(["--run-dir", str(run_dir), "--column", "cc", "--run", "sweep-1", "--db", str(db_path)])

    assert rc == 0
    rows = read_rows(db_path)
    assert len(rows) == 1
    assert rows[0]["column"] == "cc"
    assert rows[0]["kernel"] == "wf_triangular"
    assert rows[0]["run"] == "sweep-1"


def test_merging_a_second_column_never_erases_the_first_columns_already_merged_rows(tmp_path: pathlib.Path) -> None:
    """The whole reason this tool exists instead of reusing collect_canon.py's rebuild: sibling
    columns' jobs finalize independently, sometimes minutes apart, and a rebuild-style merge would
    wipe out cc's row the moment numba's job finalizes."""
    run_dir = tmp_path / "sweep"
    run_dir.mkdir()
    write_shard(run_dir, "cc", 0, [ROW])
    db_path = tmp_path / "canon.db"
    merge_canon_results.main(["--run-dir", str(run_dir), "--column", "cc", "--run", "sweep-1", "--db", str(db_path)])

    write_shard(run_dir, "numba", 0, [{**ROW, "kernel": "argmax_with_index"}])
    rc = merge_canon_results.main(
        ["--run-dir", str(run_dir), "--column", "numba", "--run", "sweep-1", "--db", str(db_path)]
    )

    assert rc == 0
    rows = {(r["column"], r["kernel"]) for r in read_rows(db_path)}
    assert rows == {("cc", "wf_triangular"), ("numba", "argmax_with_index")}


def test_merging_the_same_column_twice_does_not_duplicate_its_rows(tmp_path: pathlib.Path) -> None:
    """A requeued job (or a retried finalize step after a transient sqlite lock) must overwrite the
    same row, not append a second one -- the unique index on (run, column, kernel, preset,
    datatype) is what makes the merge idempotent."""
    run_dir = tmp_path / "sweep"
    run_dir.mkdir()
    write_shard(run_dir, "cc", 0, [ROW])
    db_path = tmp_path / "canon.db"

    for _ in range(2):
        rc = merge_canon_results.main(
            ["--run-dir", str(run_dir), "--column", "cc", "--run", "sweep-1", "--db", str(db_path)]
        )
        assert rc == 0

    rows = read_rows(db_path)
    assert len(rows) == 1


def test_a_row_count_mismatch_against_the_callers_own_count_refuses_to_merge(tmp_path: pathlib.Path) -> None:
    """``--expected`` is the caller's OWN independent count (e.g. wc -l over the same shards); a
    disagreement with what this script's CSV parse finds means something is wrong with the shard
    (a truncated write, a half-flushed file) and must not be quietly merged as if it were clean."""
    run_dir = tmp_path / "sweep"
    run_dir.mkdir()
    write_shard(run_dir, "cc", 0, [ROW])
    db_path = tmp_path / "canon.db"

    rc = merge_canon_results.main(
        ["--run-dir", str(run_dir), "--column", "cc", "--run", "sweep-1", "--db", str(db_path), "--expected", "2"]
    )

    assert rc != 0
    assert not db_path.exists(), "a refused merge must not create or touch the destination DB at all"


def test_a_column_with_no_shards_is_zero_rows_not_an_error(tmp_path: pathlib.Path) -> None:
    """A rank whose kernel share was empty writes no CSV at all (canon_column.sh's zero-kernel-rank
    guard); a column that happens to have none of its ranks' shards present must be a clean 0-row
    merge the caller can still treat as safe-to-clean-up, not a crash."""
    run_dir = tmp_path / "sweep"
    run_dir.mkdir()
    db_path = tmp_path / "canon.db"

    rc = merge_canon_results.main(
        ["--run-dir", str(run_dir), "--column", "stubcol", "--run", "sweep-1", "--db", str(db_path), "--expected", "0"]
    )

    assert rc == 0


def test_a_crashed_kernel_keeps_its_row_with_no_time_not_a_zero_one(tmp_path: pathlib.Path) -> None:
    """Mirrors collect_canon.py's own contract: an empty median_ms field must decode to NULL, never
    to 0.0, or a crashed kernel would misread as an instant one."""
    run_dir = tmp_path / "sweep"
    run_dir.mkdir()
    write_shard(run_dir, "dace_cpu", 0, [{**ROW, "median_ms": "", "validated": ""}])
    db_path = tmp_path / "canon.db"

    merge_canon_results.main(["--run-dir", str(run_dir), "--column", "dace_cpu", "--run", "sweep-1", "--db", str(db_path)])

    rows = read_rows(db_path)
    assert rows[0]["median_ms"] is None
    assert rows[0]["validated"] == ""
