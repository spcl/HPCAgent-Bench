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
import sqlite3
import subprocess
import sys
from types import ModuleType

import pytest

from hpcagent_bench.harness import recording

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "experiments"

MONITOR_HEADER = "ts,cpu_pct,load1,mem_used_mib,mem_total_mib,gpu_pct,vram_used_mib,vram_total_mib"
MONITOR_ROW = "2026-01-01T00:00:00Z,10.0,0.1,100,1000,0.0,0,0"

#: bytes that are neither a valid sqlite header nor valid UTF-8 -- a stand-in for a shard truncated
#: mid-write or a monitor CSV read off a corrupted disk. Fixed, not random, so a failure is reproducible.
GARBAGE_BYTES = b"\xff\xfe" * 100


def load_example_module(name: str) -> ModuleType:
    """``sys.modules`` must carry the module BEFORE exec: several of these modules declare a
    ``@dataclass`` under ``from __future__ import annotations``, and dataclass field resolution looks
    up ``sys.modules[cls.__module__]`` -- an unregistered module makes that lookup ``None`` and
    crashes."""
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="validate_run")
def validate_run_fixture():
    return load_example_module("validate_run")


@pytest.fixture(name="monitor_report")
def monitor_report_fixture():
    return load_example_module("monitor_report")


def seed_shard(path: pathlib.Path, *, run_id: str, kernel: str = "gemm", ts: int = 1) -> None:
    """One valid submissions row plus the run it belongs to, in a fresh shard DB -- the same shape
    as test_db_aggregate.py's ``_seed``. The measurement row carries no ``language``; the identity a
    figure groups by is the ``runs`` row joined by ``run_id``."""
    conn = recording.connect(str(path))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO benchmarks(name, track, dwarf, source) VALUES (?,?,?,?)",
            (kernel, "scientific_computing", "dense_la", None),
        )
        # The arm's language is one runs row per run, not a column on the measurement row.
        conn.execute(
            "INSERT OR IGNORE INTO runs(run_id, experiment, model, language, device, packet, rep, arm, first_seen) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, "validate", "stub-model", "c", "cpu", "", 1, run_id.split(".")[0], ts),
        )
        conn.execute(
            "INSERT INTO submissions(run_id, ts, benchmark, preset, datatype, "
            "source_mode, optimizer, baseline, speedup) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, ts, kernel, "S", "float64", "restricted", "noop", "c", 1.5),
        )
        conn.commit()
    finally:
        conn.close()


def build_run_dir(
    tmp_path: pathlib.Path, *, ranks: int = 2, agents: int = 2, drop_log: bool = False, empty_agent: bool = False
) -> pathlib.Path:
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
def test_intact_run_passes_every_check(tmp_path, validate_run) -> None:
    run_dir = build_run_dir(tmp_path)
    results = validate_run.run_checks(run_dir)
    assert all(r.ok for r in results), results
    assert validate_run.main([str(run_dir)]) == 0


def test_db_shards_check_reports_per_shard_and_merged_totals(tmp_path, validate_run) -> None:
    run_dir = build_run_dir(tmp_path, ranks=3)
    result = validate_run.check_db_shards(run_dir)
    assert result.ok, result.summary
    assert "3/3 shards" in result.summary
    assert "merged=3" in result.summary  # one submissions row per rank, none dedup


# --- the exact hole TASK 3 asks for: a missing claude.log + an empty agent dir ------------------------
def test_missing_log_and_empty_agent_dir_fail_only_those_checks(tmp_path, validate_run) -> None:
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


def test_report_prints_a_pass_fail_line_per_check(tmp_path, validate_run, capsys) -> None:
    run_dir = build_run_dir(tmp_path, drop_log=True)
    validate_run.main([str(run_dir)])
    out = capsys.readouterr().out
    assert "FAIL" in out and "PASS" in out
    assert "agent_logs" in out


# --- graceful degradation: a missing subtree is a FAIL, never a traceback -----------------------------
@pytest.mark.parametrize(
    "check_name,expected_summary",
    [
        ("check_db_shards", "no judge/ dir"),
        ("check_submissions_disk", "no shared/ dir"),
        ("check_agent_logs", "no agents/ dir"),
        ("check_monitor", "no monitor/ dir"),
    ],
    ids=["judge", "shared", "agents", "monitor"],
)
def test_a_missing_subtree_fails_cleanly(tmp_path, validate_run, check_name, expected_summary) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = getattr(validate_run, check_name)(run_dir)
    assert not result.ok
    assert expected_summary in result.summary


def test_a_run_dir_that_does_not_exist_at_all_still_reports_cleanly(tmp_path, validate_run) -> None:
    run_dir = tmp_path / "does-not-exist"
    results = validate_run.run_checks(run_dir)
    assert all(not r.ok for r in results)
    assert validate_run.main([str(run_dir)]) == 1


def test_monitor_csv_with_no_data_rows_is_flagged(tmp_path, validate_run) -> None:
    run_dir = build_run_dir(tmp_path)
    (run_dir / "monitor" / "judge-nid002.csv").write_text(MONITOR_HEADER + "\n")  # header only, 0 samples
    result = validate_run.check_monitor(run_dir)
    assert not result.ok
    assert "judge-nid002.csv" in result.summary


# --- a corrupt shard must fail loudly, not disappear into a partial merge -----------------------------
def test_merge_results_standalone_reports_corrupt_shard_and_fails_cleanly(tmp_path) -> None:
    run_dir = build_run_dir(tmp_path, ranks=2)
    bad_shard = run_dir / "judge" / "rank-1" / "hpcagent_bench.db"
    bad_shard.write_bytes(GARBAGE_BYTES)  # truncated/OOM-killed shard, not a valid sqlite file

    result = subprocess.run(
        [sys.executable, str(EXAMPLE / "merge_results.py"), str(run_dir)], capture_output=True, text=True, check=False
    )

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "rank-1" in result.stderr and "hpcagent_bench.db" in result.stderr


# --- the per-call trajectory must survive the merge, not just the leaderboard rows --------------------
def test_merge_results_carries_the_call_trajectory(tmp_path) -> None:
    """The judge writes a ``calls`` row for EVERY grade, so that table -- not ``submissions`` -- is
    where an arm's failures-before-success live. A merge that copied only the tables it was written
    against would drop the whole history when the run ends."""
    run_dir = build_run_dir(tmp_path, ranks=2)
    for rank in range(2):
        conn = recording.connect(str(run_dir / "judge" / f"rank-{rank}" / "hpcagent_bench.db"))
        try:
            conn.execute(
                "INSERT INTO calls(run_id, ts, benchmark, preset, datatype, source_mode, "
                "round, tokens, status, route) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (f"r{rank}", 1, "gemm", "S", "float64", "restricted", 1, 0, "build_error", "score"),
            )
            conn.commit()
        finally:
            conn.close()

    out = tmp_path / "merged.db"
    result = subprocess.run(
        [sys.executable, str(EXAMPLE / "merge_results.py"), str(run_dir), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    conn = sqlite3.connect(str(out))
    try:
        assert [row[0] for row in conn.execute("SELECT route FROM calls ORDER BY run_id")] == ["score", "score"]
        # A calls row holds no language of its own: it names a run, and the run names the arm's
        # language. Carrying `calls` without `runs` would merge a trajectory nothing can attribute.
        attributed = [
            row[0] for row in conn.execute("SELECT r.language FROM calls JOIN runs r USING (run_id) ORDER BY run_id")
        ]
    finally:
        conn.close()
    assert attributed == ["c", "c"]


# --- monitor_report must skip a garbage CSV, not lose the good ones with it ---------------------------
def test_monitor_report_skips_garbage_csv_and_still_reports_the_rest(
    tmp_path, monitor_report, capsys, monkeypatch
) -> None:
    monitor_dir = tmp_path / "monitor"
    monitor_dir.mkdir()
    (monitor_dir / "vllm-nid001.csv").write_text(f"{MONITOR_HEADER}\n{MONITOR_ROW}\n")
    (monitor_dir / "judge-nid002.csv").write_bytes(GARBAGE_BYTES)

    monkeypatch.setattr(sys, "argv", ["monitor_report.py", str(monitor_dir)])
    monitor_report.main()

    out, err = capsys.readouterr()
    assert "nid001" in out
    assert "skipped 1/2 CSV files" in out
    assert "skipped" in err and "nid002" in err


def test_monitor_report_exits_nonzero_when_every_csv_is_bad(tmp_path, monitor_report, capsys, monkeypatch) -> None:
    monitor_dir = tmp_path / "monitor"
    monitor_dir.mkdir()
    (monitor_dir / "judge-nid002.csv").write_bytes(GARBAGE_BYTES)

    monkeypatch.setattr(sys, "argv", ["monitor_report.py", str(monitor_dir)])
    with pytest.raises(SystemExit) as excinfo:
        monitor_report.main()

    err = capsys.readouterr().err
    assert "nid002" in err  # the one bad file is named on stderr as it is skipped
    assert "unreadable" in str(excinfo.value)  # and the all-skipped case says so plainly
