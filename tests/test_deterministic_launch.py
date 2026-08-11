# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end run of scripts/submit_deterministic.sbatch at one rank, no scheduler, no container.

The pure-logic parts of a cluster job are already covered (tests/test_cluster_launch.py,
tests/test_preset_sweep.py, tests/test_db_aggregate.py), but nothing exercised the SCRIPT: the
preflights, the rank -> kernel shard argv, the per-rank DB routing and the trailing rollup only ever
ran on a real allocation, so a typo in any of them surfaced as a dead job hours into a queue.

One rank is the degenerate case of the same deployment -- the script's N=1 path -- so it needs no
MPI, no container and no allocation. ``srun`` is the only cluster-only dependency and it is replaced
by a shim that runs the step in-process as rank 0, which is what a one-task srun does.
"""
import os
import pathlib
import shutil
import sqlite3
import subprocess

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "submit_deterministic.sbatch"

#: Cheapest honest column: numpy needs no compiler, so this skips the dace and autopar preflights and
#: measures a real kernel rather than a stub.
FRAMEWORK = "numpy"
KERNEL = "gesummv"

#: srun's own flags all precede the command; consume them, then become the command. Exporting
#: SLURM_PROCID is the point -- recording.db_shard() reads it, so the rank writes its own shard.
SRUN_SHIM = """#!/bin/bash
while [[ "${1:-}" == --* ]]; do shift; done
export SLURM_PROCID=0
exec "$@"
"""


def _job_env(work: pathlib.Path) -> dict:
    """Environment for a one-rank, container-free run of the job script."""
    env = dict(os.environ)
    bindir = work / "bin"
    bindir.mkdir()
    shim = bindir / "srun"
    shim.write_text(SRUN_SHIM)
    shim.chmod(0o755)

    env.update(
        PATH=str(bindir) + os.pathsep + env["PATH"],
        # REPO is where the job cds and writes its results/ and dace cache; point it at the test's
        # work dir so the run does not touch the working tree.
        HPCAGENT_BENCH_REPO=str(work),
        FRAMEWORKS=FRAMEWORK,
        BENCH=KERNEL,
        PRESET="S",
        REPEAT="1",
        TIMEOUT="120",
        HPCAGENT_BENCH_RECORD_DB_PATH=str(work / "hpcagent_bench.db"),
        # tmp_path is tmpfs on many hosts and base_db_path refuses memory-backed storage outright.
        HPCAGENT_BENCH_RECORD_ALLOW_MEMORY_DB="1",
        MPLBACKEND="Agg",
        OMPI_MCA_pml="ob1",
        OMPI_MCA_btl="^openib",
        MPICH_NO_LOCAL="1",
        PMIX_MCA_gds="hash",
    )
    # Inherited scheduler state would make the script plan for an allocation that is not there.
    for name in ("SLURM_JOB_ID", "SLURM_JOB_NUM_NODES", "SLURM_PROCID", "HPCAGENT_BENCH_DB_SHARD",
                 "OMPI_COMM_WORLD_RANK", "PMI_RANK", "HPCAGENT_BENCH_EDF"):
        env.pop(name, None)
    return env


def _rows(db: pathlib.Path) -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT COUNT(*) FROM results WHERE framework=? AND benchmark=? AND validated=1",
                            (FRAMEWORK, KERNEL)).fetchone()[0]
    finally:
        conn.close()


@pytest.mark.integration
def test_one_rank_job_shards_by_rank_and_rolls_up(tmp_path):
    """The whole script: preflights, sharded run-framework argv, per-rank DB, merged rollup."""
    if shutil.which("hpcagent-bench") is None:
        pytest.skip("hpcagent-bench console script is not installed")

    env = _job_env(tmp_path)
    done = subprocess.run(["bash", str(SCRIPT)],
                          cwd=str(tmp_path),
                          env=env,
                          capture_output=True,
                          text=True,
                          timeout=900)
    assert done.returncode == 0, f"job failed ({done.returncode}):\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}"

    # The rank ran the framework over ITS shard of the selection, and the rollup closed the job.
    assert f"[rank 0/1] framework={FRAMEWORK}" in done.stdout
    assert "=== merged ===" in done.stdout

    rundir = tmp_path / "results" / "deterministic-local"
    csv = rundir / "shard-0.csv"
    assert csv.exists(), f"no per-shard CSV in {sorted(p.name for p in rundir.iterdir())}"
    assert KERNEL in csv.read_text()

    # SLURM_PROCID reached recording.db_shard(), so the rank wrote hpcagent_bench0.db and NOT the
    # base file -- the property that keeps concurrent ranks off one un-lockable network-FS SQLite.
    shard_db = tmp_path / "hpcagent_bench0.db"
    assert shard_db.exists(), "rank 0 did not write its own shard DB"
    assert _rows(shard_db) >= 1

    # aggregate-db merged the shard into the base file readers open.
    base_db = tmp_path / "hpcagent_bench.db"
    assert base_db.exists(), "the trailing aggregate-db did not build the base DB"
    assert _rows(base_db) == _rows(shard_db)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
