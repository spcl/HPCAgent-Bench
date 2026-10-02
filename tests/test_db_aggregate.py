# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-rank results DBs and their aggregation.

A distributed run cannot share one SQLite file: WAL needs a ``-shm`` mapping that Lustre/NFS do not
provide, and rollback-journal locking over them is unreliable. Each rank therefore writes its own
persistent shard and the shards are merged afterwards by natural key (:func:`results_db.merge`) --
on demand, by :func:`recording.ensure_aggregated`, so no caller has to remember an aggregation step.
"""

import contextlib
import os
import pathlib
import sqlite3
import tempfile
import time

import pytest

from hpcagent_bench.harness import recording, results_db
from tests.results_rows import attempts, submissions


def _seed(path: str, *, run: str, kernels: list[str], with_results: bool = True, language: str = "c") -> None:
    """Write one shard: the run's setup and identity, one credited and one failed grade per kernel.

    The grades carry no ``language`` of their own -- the identity a figure groups by is the setup the
    run belongs to -- so the shard has to hold that row or the merged DB describes grades nothing
    can attribute."""
    setup = run.split(".")[0]
    with contextlib.closing(recording.connect(path)) as conn:
        results_db.ensure_setup(conn, results_db.Setup(setup, language, "cpu", study="agg", model="stub-model"))
        run_id = results_db.ensure_run(conn, setup, run, None)
        for kernel in kernels:
            stamp = {"preset": "S", "datatype": "float64", "source_mode": "restricted", "baseline": "c"}
            credited = {"build_ok": 1, "correct": 1, "speedup": 1.5, "credited_speedup": 1.5}
            results_db.add_grade(conn, run_id, kernel, "submit", ts_ms=1, values=stamp | credited)
            failed = {"build_ok": 0, "correct": 0, "reason": "build"}
            results_db.add_grade(conn, run_id, kernel, "submit", ts_ms=2, values=stamp | failed)
            if with_results:
                # The framework ``results`` table belongs to another module's schema but lives in the
                # same file; aggregation must carry it even though recording.py never creates it.
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS results ("
                    "id INTEGER PRIMARY KEY, timestamp INTEGER, benchmark TEXT, preset TEXT, "
                    "framework TEXT, validated INTEGER, time REAL)"
                )
                conn.execute(
                    "INSERT INTO results(timestamp, benchmark, preset, framework, validated, time) "
                    "VALUES (?,?,?,?,?,?)",
                    (1, kernel, "S", "numpy", 1, 2.0),
                )
        conn.commit()


def _count(path: str, table: str) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def test_shard_paths_order_numerically(tmp_path) -> None:
    """Shard 10 must merge after shard 9, which a lexical sort gets wrong."""
    base = str(tmp_path / "hpcagent_bench.db")
    for shard in (0, 2, 9, 10):
        open(recording.shard_db_path(shard, base), "w").close()
    assert recording.shard_paths(base) == [recording.shard_db_path(s, base) for s in (0, 2, 9, 10)]


def test_shard_paths_ignores_the_base_and_unrelated_files(tmp_path) -> None:
    base = str(tmp_path / "hpcagent_bench.db")
    open(base, "w").close()
    open(str(tmp_path / "hpcagent_benchX.db"), "w").close()
    open(str(tmp_path / "other1.db"), "w").close()
    _seed(recording.shard_db_path(1, base), run="r1", kernels=["gemm"])
    assert recording.shard_paths(base) == [recording.shard_db_path(1, base)]


def test_aggregate_merges_every_table_and_reassigns_ids(tmp_path) -> None:
    base = str(tmp_path / "hpcagent_bench.db")
    _seed(recording.shard_db_path(0, base), run="r0", kernels=["gemm", "jacobi_2d"], language="c")
    _seed(recording.shard_db_path(1, base), run="r1", kernels=["gemm", "spmv"], language="fortran")

    recording.aggregate(base)

    # Every grade is its own natural key; the runs and setups keep one row per key.
    assert len(submissions(base)) == 4
    assert len(attempts(base)) == 4
    assert _count(base, "results") == 4
    assert _count(base, "runs") == 2

    ids = [row["id"] for row in submissions(base)]
    runs = {row["label"] for row in submissions(base)}
    with results_db.reading(base) as conn:
        tagged = sorted(
            tuple(row)
            for row in conn.execute("SELECT language, COUNT(*) FROM grades_flat WHERE credited_speedup > 0 GROUP BY 1")
        )
    # Both shards number their own rows from 1; the destination must reassign, not collide.
    assert len(set(ids)) == 4
    assert runs == {"r0", "r1"}
    # A merged grade must still reach its setup: the identity is on `setups`, so a merge that carried the
    # grades and dropped the identity would leave four rows nothing can group.
    assert tagged == [("c", 2), ("fortran", 2)]


def test_two_ranks_of_one_run_merge_instead_of_colliding(tmp_path: pathlib.Path) -> None:
    """A run is keyed by its job and label, and every rank of a run writes its own shard with that
    same row. Merged with a plain INSERT the second copy raises UNIQUE and takes the WHOLE merge
    down, so a multi-rank experiment would lose every table, not one row. The same grade seen by two
    ranks is one grade too."""
    base = str(tmp_path / "hpcagent_bench.db")
    for rank in (0, 1):
        _seed(recording.shard_db_path(rank, base), run="llr2-c.n0.p3.w1", kernels=["gemm"], language="c")

    recording.aggregate(base)

    assert _count(base, "runs") == 1
    assert len(submissions(base)) == 1
    with results_db.reading(base) as conn:
        assert [r[0] for r in conn.execute("SELECT language FROM setups")] == ["c"]


def test_aggregate_merges_a_run_id_shared_by_multiple_shards(tmp_path) -> None:
    """A run served by several ranks writes the SAME run into every rank's own shard, so the second
    shard's row for it must not collide with the first's -- it is the same fact."""
    base = str(tmp_path / "hpcagent_bench.db")
    _seed(recording.shard_db_path(0, base), run="shared", kernels=["gemm"])
    _seed(recording.shard_db_path(1, base), run="shared", kernels=["spmv"])

    recording.aggregate(base)

    assert _count(base, "runs") == 1
    with results_db.reading(base) as conn:
        rows = [tuple(r) for r in conn.execute("SELECT label, model, setup FROM runs JOIN setups USING (setup)")]
    assert rows == [("shared", "stub-model", "shared")]
    # The grades still concatenate; only the run's identity dedups.
    assert len(submissions(base)) == 2


def test_aggregate_survives_a_shard_missing_a_column(tmp_path) -> None:
    """Shards can be written by different code versions; one stale shard must not kill the merge."""
    base = str(tmp_path / "hpcagent_bench.db")
    current, stale = recording.shard_db_path(0, base), recording.shard_db_path(1, base)
    _seed(current, run="r0", kernels=["gemm"])
    _seed(stale, run="r1", kernels=["spmv"])

    conn = sqlite3.connect(stale)
    try:  # drop a column the other shard has, the way an older schema would
        conn.execute("ALTER TABLE results DROP COLUMN framework")
        conn.commit()
    finally:
        conn.close()

    recording.aggregate(base)

    assert _count(base, "results") == 2
    conn = sqlite3.connect(base)
    try:
        frameworks = {r[0] for r in conn.execute("SELECT framework FROM results")}
    finally:
        conn.close()
    # The row survives; only the column the stale shard could not supply is NULL.
    assert frameworks == {None, "numpy"}


def test_aggregate_is_idempotent(tmp_path) -> None:
    """Rebuilt from scratch, never appended to -- re-merging must not double the rows."""
    base = str(tmp_path / "hpcagent_bench.db")
    _seed(recording.shard_db_path(0, base), run="r0", kernels=["gemm"])
    _seed(recording.shard_db_path(1, base), run="r1", kernels=["spmv"])

    recording.aggregate(base)
    first = len(submissions(base))
    recording.aggregate(base)
    assert len(submissions(base)) == first == 2


def test_aggregate_carries_the_sources_inside_the_db(tmp_path) -> None:
    """A source is a row of the DB, so the merged file holds the text itself: no store beside it."""
    base = str(tmp_path / "hpcagent_bench.db")
    shard = recording.shard_db_path(0, base)
    _seed(shard, run="r0", kernels=["gemm"])
    with contextlib.closing(recording.connect(shard)) as conn:
        results_db.store_source(conn, 1, "host", "c", "optimize this")
        conn.commit()

    recording.aggregate(base)

    with results_db.reading(base) as conn:
        texts = conn.execute("SELECT s.text FROM grade_sources gs JOIN sources s USING (hash)").fetchall()
    assert [tuple(row) for row in texts] == [("optimize this",)]
    assert not list(tmp_path.glob("*_prompts"))


def test_ensure_aggregated_builds_when_the_aggregate_is_missing(tmp_path) -> None:
    base = str(tmp_path / "hpcagent_bench.db")
    _seed(recording.shard_db_path(0, base), run="r0", kernels=["gemm"])
    assert not os.path.exists(base)

    assert recording.ensure_aggregated(base) == base
    assert len(submissions(base)) == 1


def test_ensure_aggregated_rebuilds_when_a_shard_is_newer(tmp_path) -> None:
    """A shard that landed after the last merge must not be read through a stale aggregate."""
    base = str(tmp_path / "hpcagent_bench.db")
    _seed(recording.shard_db_path(0, base), run="r0", kernels=["gemm"])
    recording.ensure_aggregated(base)
    assert len(submissions(base)) == 1

    late = recording.shard_db_path(1, base)
    _seed(late, run="r1", kernels=["spmv"])
    os.utime(late, (time.time() + 10, time.time() + 10))

    recording.ensure_aggregated(base)
    assert len(submissions(base)) == 2


def test_ensure_aggregated_is_a_noop_without_shards(tmp_path) -> None:
    """A single-writer run keeps working untouched -- no aggregate is invented for it."""
    base = str(tmp_path / "hpcagent_bench.db")
    _seed(base, run="solo", kernels=["gemm"])
    before = os.path.getmtime(base)
    assert recording.ensure_aggregated(base) == base
    assert os.path.getmtime(base) == before
    assert len(submissions(base)) == 1


def test_a_single_writer_run_still_writes_a_shard(monkeypatch, tmp_path) -> None:
    """Nothing writes the base file. It is BOTH authoritative and derived otherwise -- the next
    merge erases it, and its mtime makes a stale aggregate look fresh."""
    from hpcagent_bench import config

    for name in ("HPCAGENT_BENCH_DB_SHARD", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK"):
        monkeypatch.delenv(name, raising=False)
    config.set_override("record.db_path", str(tmp_path / "hpcagent_bench.db"))
    config.set_override("record.allow_memory_db", True)
    try:
        assert recording.db_path() == recording.shard_db_path(0, recording.base_db_path())
    finally:
        config.clear_override("record.db_path")
        config.clear_override("record.allow_memory_db")


def test_db_shard_prefers_the_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_PROCID", "7")
    monkeypatch.setenv("HPCAGENT_BENCH_DB_SHARD", "2")
    assert recording.db_shard() == 2


def test_db_shard_falls_back_to_the_launcher_rank(monkeypatch) -> None:
    """A job that forgets the explicit variable still shards, rather than sharing one file."""
    monkeypatch.delenv("HPCAGENT_BENCH_DB_SHARD", raising=False)
    monkeypatch.setenv("SLURM_PROCID", "7")
    assert recording.db_shard() == 7


def test_db_shard_is_none_when_single_writer(monkeypatch) -> None:
    for name in ("HPCAGENT_BENCH_DB_SHARD", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK"):
        monkeypatch.delenv(name, raising=False)
    assert recording.db_shard() is None


def test_memory_backed_storage_is_refused(tmp_path) -> None:
    """Results on tmpfs vanish with the allocation and steal RAM from the kernel being measured."""
    from hpcagent_bench import config

    # Whichever of the host's temp locations is actually in RAM -- named by the probe rather than
    # hardcoded, since which mounts are tmpfs differs per host and per CI image.
    memory_dir = next(
        (
            d
            for d in (tmp_path, pathlib.Path(tempfile.gettempdir()), pathlib.Path("/dev/shm"))
            if recording.memory_backed_fstype(str(d)) is not None
        ),
        None,
    )
    assert memory_dir is not None, "no tmpfs under tmp_path, the temp dir or /dev/shm in /proc/mounts"
    config.set_override("record.db_path", str(memory_dir / "hpcagent_bench.db"))
    try:
        with pytest.raises(ValueError, match="memory-backed"):
            recording.base_db_path()
    finally:
        config.clear_override("record.db_path")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
