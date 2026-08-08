# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""validate_run.py: the post-run PASS/FAIL checks over a synthetic cluster run directory.

Builds the same tree run_cluster.sh + agent_driver.py + node_monitor.sh leave behind (judge shard
DBs, shared/agent-*/ write folders, per-worker claude.log files, monitor CSVs) and checks that an
intact run reports all-PASS while a run missing a log and a submission reports exactly those two
FAILs, gracefully, with no traceback.
"""
import importlib.util
import pathlib
import sys

import pytest

from hpcagent_bench.harness import recording

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "containers/cluster/example-script"

MONITOR_HEADER = "ts,cpu_pct,load1,mem_used_mib,mem_total_mib,gpu_pct,vram_used_mib,vram_total_mib"
MONITOR_ROW = "2026-01-01T00:00:00Z,10.0,0.1,100,1000,0.0,0,0"


def load_validate_run():
    """``sys.modules`` must carry the module BEFORE exec: ``CheckResult`` is a ``@dataclass`` under
    ``from __future__ import annotations``, and dataclass field resolution looks up
    ``sys.modules[cls.__module__]`` -- an unregistered module makes that lookup ``None`` and crashes."""
    spec = importlib.util.spec_from_file_location("validate_run", EXAMPLE / "validate_run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="validate_run")
def validate_run_fixture():
    return load_validate_run()


def seed_shard(path: pathlib.Path, *, run_id: str, kernel: str = "gemm", ts: int = 1) -> None:
    """One valid submissions row in a fresh shard DB, same shape as test_db_aggregate.py's _seed."""
    conn = recording.connect(str(path))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO benchmarks(name, track, kind, domain, dwarf, source) "
            "VALUES (?,?,?,?,?,?)", (kernel, "scientific_computing", "dense", "linalg", "dense_la", None))
        conn.execute(
            "INSERT INTO submissions(run_id, ts, benchmark, preset, datatype, language, "
            "source_mode, optimizer, baseline, speedup) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, ts, kernel, "S", "float64", "c", "restricted", "noop", "c", 1.5))
        conn.commit()
    finally:
        conn.close()


def build_run_dir(tmp_path: pathlib.Path,
                  *,
                  ranks: int = 2,
                  agents: int = 2,
                  drop_log: bool = False,
                  empty_agent: bool = False) -> pathlib.Path:
    """A run dir shaped like RUN_DIR after a real campaign: judge shards, shared write folders, agent
    logs, and one monitor CSV. ``drop_log`` and ``empty_agent`` punch the exact holes TASK 3 wants."""
    run_dir = tmp_path / "run"
    for rank in range(ranks):
        rank_dir = run_dir / "judge" / f"rank-{rank}"
        rank_dir.mkdir(parents=True)
        seed_shard(rank_dir / "hpcagent_bench.db", run_id=f"r{rank}", ts=rank + 1)

    for index in range(agents):
        agent_dir = run_dir / "shared" / f"agent-{index}"
        agent_dir.mkdir(parents=True)
        if not (empty_agent and index == agents - 1):
            (agent_dir / "gemm.c").write_text("void gemm(void){}\n")

    node_dir = run_dir / "agents" / "node-0"
    for worker in range(agents):
        worker_dir = node_dir / f"problem-{worker}-worker-{worker}"
        worker_dir.mkdir(parents=True)
        if not (drop_log and worker == agents - 1):
            (worker_dir / "claude.log").write_text("agent output\n")

    monitor_dir = run_dir / "monitor"
    monitor_dir.mkdir(parents=True)
    (monitor_dir / "agent-nid001.csv").write_text(f"{MONITOR_HEADER}\n{MONITOR_ROW}\n")
    return run_dir


# --- an intact run: everything PASSes ---------------------------------------------------------------
def test_intact_run_passes_every_check(tmp_path, validate_run):
    run_dir = build_run_dir(tmp_path)
    results = validate_run.run_checks(run_dir)
    assert all(r.ok for r in results), results
    assert validate_run.main([str(run_dir)]) == 0


def test_db_shards_check_reports_per_shard_and_merged_totals(tmp_path, validate_run):
    run_dir = build_run_dir(tmp_path, ranks=3)
    result = validate_run.check_db_shards(run_dir)
    assert result.ok, result.summary
    assert "3/3 shards" in result.summary
    assert "merged=3" in result.summary  # one submissions row per rank, none dedup


# --- the exact hole TASK 3 asks for: a missing claude.log + an empty agent dir ------------------------
def test_missing_log_and_empty_agent_dir_fail_only_those_checks(tmp_path, validate_run):
    run_dir = build_run_dir(tmp_path, drop_log=True, empty_agent=True)

    results = validate_run.run_checks(run_dir)
    by_name = {r.name: r for r in results}

    assert not by_name["submissions_disk"].ok
    assert "agent-1" in by_name["submissions_disk"].summary
    assert not by_name["agent_logs"].ok
    assert "problem-1-worker-1" in by_name["agent_logs"].summary

    # the holes are LOCAL: shard DBs and monitor CSVs were untouched and must still PASS
    assert by_name["db_shards"].ok
    assert by_name["monitor"].ok

    assert validate_run.main([str(run_dir)]) == 1


def test_report_prints_a_pass_fail_line_per_check(tmp_path, validate_run, capsys):
    run_dir = build_run_dir(tmp_path, drop_log=True)
    validate_run.main([str(run_dir)])
    out = capsys.readouterr().out
    assert "FAIL" in out and "PASS" in out
    assert "agent_logs" in out


# --- graceful degradation: a missing subtree is a FAIL, never a traceback -----------------------------
def test_missing_judge_dir_fails_cleanly(tmp_path, validate_run):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = validate_run.check_db_shards(run_dir)
    assert not result.ok
    assert "no judge/ dir" in result.summary


def test_missing_shared_dir_fails_cleanly(tmp_path, validate_run):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = validate_run.check_submissions_disk(run_dir)
    assert not result.ok
    assert "no shared/ dir" in result.summary


def test_missing_agents_dir_fails_cleanly(tmp_path, validate_run):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = validate_run.check_agent_logs(run_dir)
    assert not result.ok
    assert "no agents/ dir" in result.summary


def test_missing_monitor_dir_fails_cleanly(tmp_path, validate_run):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = validate_run.check_monitor(run_dir)
    assert not result.ok
    assert "no monitor/ dir" in result.summary


def test_a_run_dir_that_does_not_exist_at_all_still_reports_cleanly(tmp_path, validate_run):
    run_dir = tmp_path / "does-not-exist"
    results = validate_run.run_checks(run_dir)
    assert all(not r.ok for r in results)
    assert validate_run.main([str(run_dir)]) == 1


def test_monitor_csv_with_no_data_rows_is_flagged(tmp_path, validate_run):
    run_dir = build_run_dir(tmp_path)
    (run_dir / "monitor" / "judge-nid002.csv").write_text(MONITOR_HEADER + "\n")  # header only, 0 samples
    result = validate_run.check_monitor(run_dir)
    assert not result.ok
    assert "judge-nid002.csv" in result.summary
