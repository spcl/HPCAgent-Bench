# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-rank results DBs and their aggregation.

A distributed run cannot share one SQLite file: WAL needs a ``-shm`` mapping that Lustre/NFS do not
provide, and rollback-journal locking over them is unreliable. Each rank therefore writes its own
persistent shard and the shards are merged afterwards -- on demand, by
:func:`recording.ensure_aggregated`, so no caller has to remember an aggregation step.
"""
import os
import pathlib
import sqlite3
import tempfile
import time
from typing import List

import pytest

from hpcagent_bench.harness import recording


def _seed(path: str, *, run: str, kernels: List[str], with_results: bool = True) -> None:
    """Write one shard: dimension rows, a prompt, one row in each id-bearing log table."""
    conn = recording.connect(path)
    try:
        for kernel in kernels:
            conn.execute(
                "INSERT OR REPLACE INTO benchmarks(name, track, kind, domain, dwarf, source) "
                "VALUES (?,?,?,?,?,?)", (kernel, "hpc", "dense", "linalg", "dense_la", None))
            conn.execute(
                "INSERT INTO submissions(run_id, ts, benchmark, preset, datatype, language, "
                "source_mode, optimizer, baseline, speedup) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run, 1, kernel, "S", "float64", "c", "restricted", "noop", "c", 1.5))
            conn.execute(
                "INSERT INTO attempts(run_id, ts, benchmark, preset, datatype, language, "
                "source_mode, build_ok, correct, reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run, 1, kernel, "S", "float64", "c", "restricted", 0, 0, "build"))
            if with_results:
                # The framework ``results`` table belongs to another module's schema but lives in the
                # same file; aggregation must carry it even though recording.py never creates it.
                conn.execute("CREATE TABLE IF NOT EXISTS results ("
                             "id INTEGER PRIMARY KEY, timestamp INTEGER, benchmark TEXT, preset TEXT, "
                             "framework TEXT, validated INTEGER, time REAL)")
                conn.execute(
                    "INSERT INTO results(timestamp, benchmark, preset, framework, validated, time) "
                    "VALUES (?,?,?,?,?,?)", (1, kernel, "S", "numpy", 1, 2.0))
        conn.commit()
    finally:
        conn.close()


def _count(path: str, table: str) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def test_shard_paths_order_numerically(tmp_path):
    """Shard 10 must merge after shard 9, which a lexical sort gets wrong."""
    base = str(tmp_path / "hpcagent_bench.db")
    for shard in (0, 2, 9, 10):
        open(recording.shard_db_path(shard, base), "w").close()
    assert recording.shard_paths(base) == [recording.shard_db_path(s, base) for s in (0, 2, 9, 10)]


def test_shard_paths_ignores_the_base_and_unrelated_files(tmp_path):
    base = str(tmp_path / "hpcagent_bench.db")
    open(base, "w").close()
    open(str(tmp_path / "hpcagent_benchX.db"), "w").close()
    open(str(tmp_path / "other1.db"), "w").close()
    _seed(recording.shard_db_path(1, base), run="r1", kernels=["gemm"])
    assert recording.shard_paths(base) == [recording.shard_db_path(1, base)]


def test_aggregate_merges_every_table_and_reassigns_ids(tmp_path):
    base = str(tmp_path / "hpcagent_bench.db")
    _seed(recording.shard_db_path(0, base), run="r0", kernels=["gemm", "jacobi_2d"])
    _seed(recording.shard_db_path(1, base), run="r1", kernels=["gemm", "spmv"])

    recording.aggregate(base)

    # Row logs concatenate; the dimension table dedups on its natural key (gemm seen by both shards).
    assert _count(base, "submissions") == 4
    assert _count(base, "attempts") == 4
    assert _count(base, "results") == 4
    assert _count(base, "benchmarks") == 3

    conn = sqlite3.connect(base)
    try:
        ids = [r[0] for r in conn.execute("SELECT id FROM submissions")]
        runs = {r[0] for r in conn.execute("SELECT run_id FROM submissions")}
    finally:
        conn.close()
    # Both shards number their own rows from 1; the destination must reassign, not collide.
    assert len(set(ids)) == 4
    assert runs == {"r0", "r1"}


def test_aggregate_survives_a_shard_missing_a_column(tmp_path):
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


def test_aggregate_is_idempotent(tmp_path):
    """Rebuilt from scratch, never appended to -- re-merging must not double the rows."""
    base = str(tmp_path / "hpcagent_bench.db")
    _seed(recording.shard_db_path(0, base), run="r0", kernels=["gemm"])
    _seed(recording.shard_db_path(1, base), run="r1", kernels=["spmv"])

    recording.aggregate(base)
    first = _count(base, "submissions")
    recording.aggregate(base)
    assert _count(base, "submissions") == first == 2


def test_aggregate_merges_the_prompt_store(tmp_path):
    """A copied ``prompts`` row whose file stayed beside the shard would be a dangling pointer."""
    base = str(tmp_path / "hpcagent_bench.db")
    shard = recording.shard_db_path(0, base)
    conn = recording.connect(shard)
    try:
        digest = recording.store_prompt(conn, "optimize this", "gemm", store_dir=str(recording.prompt_store_dir(shard)))
    finally:
        conn.close()

    recording.aggregate(base)

    assert _count(base, "prompts") == 1
    stored = recording.prompt_store_dir(base) / f"{digest[:2]}/{digest}.txt"
    assert stored.read_text() == "optimize this"


def test_ensure_aggregated_builds_when_the_aggregate_is_missing(tmp_path):
    base = str(tmp_path / "hpcagent_bench.db")
    _seed(recording.shard_db_path(0, base), run="r0", kernels=["gemm"])
    assert not os.path.exists(base)

    assert recording.ensure_aggregated(base) == base
    assert _count(base, "submissions") == 1


def test_ensure_aggregated_rebuilds_when_a_shard_is_newer(tmp_path):
    """A shard that landed after the last merge must not be read through a stale aggregate."""
    base = str(tmp_path / "hpcagent_bench.db")
    _seed(recording.shard_db_path(0, base), run="r0", kernels=["gemm"])
    recording.ensure_aggregated(base)
    assert _count(base, "submissions") == 1

    late = recording.shard_db_path(1, base)
    _seed(late, run="r1", kernels=["spmv"])
    os.utime(late, (time.time() + 10, time.time() + 10))

    recording.ensure_aggregated(base)
    assert _count(base, "submissions") == 2


def test_ensure_aggregated_is_a_noop_without_shards(tmp_path):
    """A single-writer run keeps working untouched -- no aggregate is invented for it."""
    base = str(tmp_path / "hpcagent_bench.db")
    _seed(base, run="solo", kernels=["gemm"])
    before = os.path.getmtime(base)
    assert recording.ensure_aggregated(base) == base
    assert os.path.getmtime(base) == before
    assert _count(base, "submissions") == 1


def test_a_single_writer_run_still_writes_a_shard(monkeypatch, tmp_path):
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


def test_a_pre_sharding_db_is_adopted_rather_than_erased(tmp_path):
    """The rebuild unlinks the destination, so a run from before sharding -- whose results ARE the
    base file -- has to become an input first, or reading the DB destroys it."""
    base = str(tmp_path / "hpcagent_bench.db")
    _seed(base, run="legacy", kernels=["gemm"])
    _seed(recording.shard_db_path(0, base), run="r0", kernels=["spmv"])

    recording.ensure_aggregated(base)

    assert _count(base, "submissions") == 2
    conn = sqlite3.connect(base)
    try:
        assert {r[0] for r in conn.execute("SELECT run_id FROM submissions")} == {"legacy", "r0"}
    finally:
        conn.close()


def test_adoption_happens_once(tmp_path):
    """The rebuilt aggregate is marked derived, so re-reading cannot adopt it as its own input and
    double every legacy row."""
    base = str(tmp_path / "hpcagent_bench.db")
    _seed(base, run="legacy", kernels=["gemm"])
    _seed(recording.shard_db_path(0, base), run="r0", kernels=["spmv"])

    recording.aggregate(base)
    recording.aggregate(base)
    recording.aggregate(base)

    assert _count(base, "submissions") == 2
    assert recording.user_version(base) == recording.DERIVED_MARK


def test_adoption_carries_the_prompt_store(tmp_path):
    """The store is named after the DB beside it, so a base adopted under a new name leaves its
    prompt rows pointing at files that are no longer there."""
    base = str(tmp_path / "hpcagent_bench.db")
    conn = recording.connect(base)
    try:
        digest = recording.store_prompt(conn, "legacy prompt", "gemm", store_dir=str(recording.prompt_store_dir(base)))
    finally:
        conn.close()
    _seed(recording.shard_db_path(0, base), run="r0", kernels=["spmv"])

    recording.aggregate(base)

    assert _count(base, "prompts") == 1
    assert (recording.prompt_store_dir(base) / f"{digest[:2]}/{digest}.txt").read_text() == "legacy prompt"


def test_db_shard_prefers_the_explicit_override(monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "7")
    monkeypatch.setenv("HPCAGENT_BENCH_DB_SHARD", "2")
    assert recording.db_shard() == 2


def test_db_shard_falls_back_to_the_launcher_rank(monkeypatch):
    """A job that forgets the explicit variable still shards, rather than sharing one file."""
    monkeypatch.delenv("HPCAGENT_BENCH_DB_SHARD", raising=False)
    monkeypatch.setenv("SLURM_PROCID", "7")
    assert recording.db_shard() == 7


def test_db_shard_is_none_when_single_writer(monkeypatch):
    for name in ("HPCAGENT_BENCH_DB_SHARD", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK"):
        monkeypatch.delenv(name, raising=False)
    assert recording.db_shard() is None


def test_memory_backed_storage_is_refused(tmp_path):
    """Results on tmpfs vanish with the allocation and steal RAM from the kernel being measured."""
    from hpcagent_bench import config
    # Whichever of the host's temp locations is actually in RAM -- named by the probe rather than
    # hardcoded, since which mounts are tmpfs differs per host and per CI image.
    memory_dir = next((d for d in (tmp_path, pathlib.Path(tempfile.gettempdir()))
                       if recording.memory_backed_fstype(str(d)) is not None), None)
    if memory_dir is None:
        pytest.skip("no memory-backed temp directory on this host")
    config.set_override("record.db_path", str(memory_dir / "hpcagent_bench.db"))
    try:
        with pytest.raises(ValueError, match="memory-backed"):
            recording.base_db_path()
    finally:
        config.clear_override("record.db_path")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
