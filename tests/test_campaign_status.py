# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""experiments/campaign_status.py: the campaign dashboard's one JSON snapshot.

Fixture run dirs below are shaped like a real one (smoke-enroot-20260917/640066): an
``agents/node-0/problem-N-worker-N/{claude.log,tokens.json}`` tree and a
``judge/rank-0/hpcagent_bench0.db`` with the real ``calls``/``submissions`` schema, built small
and deterministic rather than copied byte-for-byte so each test's expected numbers are legible.
"""

import importlib.util
import json
import pathlib
import sqlite3
import sys

import pytest

from hpcagent_bench import paths

SPEC = importlib.util.spec_from_file_location("campaign_status", paths.ROOT / "experiments" / "campaign_status.py")
campaign_status = importlib.util.module_from_spec(SPEC)
# Registered before exec, same as test_collect_campaign.py: a module loaded by path alone has no
# entry in sys.modules for dataclasses (none used here) or for its own later re-import to find.
sys.modules[SPEC.name] = campaign_status
if str(paths.ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(paths.ROOT / "experiments"))
SPEC.loader.exec_module(campaign_status)


# --------------------------------------------------------------------------------------------
# health verdict classification
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "elapsed_seconds", "claude_log_age", "db_age", "has_ready", "dead_reason", "want"),
    [
        pytest.param("RUNNING", 3600, 10.0, 500.0, True, None, "ok", id="recent-claude-log-write-is-ok"),
        pytest.param("RUNNING", 3600, 890.0, 12.0, True, None, "ok", id="recent-db-write-alone-is-ok"),
        pytest.param(
            "RUNNING", 3600, 1300.0, 1400.0, True, None, "stalled", id="long-running-with-no-recent-write-is-stalled"
        ),
        pytest.param(
            "RUNNING", 300, 1300.0, 1400.0, True, None, "ok", id="stale-writes-under-20min-runtime-is-not-yet-stalled"
        ),
        pytest.param("RUNNING", 30, None, None, False, None, "starting", id="no-ready-line-yet-is-starting"),
        pytest.param("RUNNING", 30, 5.0, 5.0, False, None, "starting", id="unready-engine-outranks-a-fresh-write"),
        pytest.param(
            "RUNNING",
            5000,
            5.0,
            5.0,
            True,
            "'Stale file handle' in .out",
            "dead-engine",
            id="dead-engine-outranks-a-fresh-write",
        ),
    ],
)
def test_health_verdict_classification(
    state: str,
    elapsed_seconds: int | None,
    claude_log_age: float | None,
    db_age: float | None,
    has_ready: bool,
    dead_reason: str | None,
    want: str,
) -> None:
    """The four verdicts (docstring, "Liveness") must come out in the order that matters: a dead
    engine or an unready one is reported as such even while a file happens to be fresh."""
    verdict, _reason = campaign_status.health_verdict(
        state=state,
        elapsed_seconds=elapsed_seconds,
        claude_log_age=claude_log_age,
        db_age=db_age,
        has_ready=has_ready,
        dead_reason=dead_reason,
    )
    assert verdict == want, verdict


def test_a_finished_job_reports_no_health_verdict_not_a_frozen_one() -> None:
    """A COMPLETED/FAILED/CANCELLED arm is not "frozen" -- it is finished, and grading it against
    the running-only stall window would call every completed arm stalled."""
    verdict, reason = campaign_status.health_verdict(
        state="COMPLETED",
        elapsed_seconds=3600,
        claude_log_age=99999.0,
        db_age=99999.0,
        has_ready=True,
        dead_reason=None,
    )
    assert verdict is None, verdict
    assert "COMPLETED" in reason


def test_dead_engine_reason_finds_an_enginecore_crash_traceback() -> None:
    """job 640090's actual failure shape: an EngineCore worker traceback ending in a fatal
    exception, tagged by the process that died rather than by a request handler."""
    text = (
        "(EngineCore pid=60406) ERROR 09-17 13:10:07 [core.py:1195] EngineCore failed to start.\n"
        "(EngineCore pid=60406) ERROR 09-17 13:10:07 [core.py:1195] Traceback (most recent call last):\n"
        "(EngineCore pid=60406) ERROR 09-17 13:10:07 [core.py:1195]   File "
        '"/opt/vllm-src/vllm/v1/engine/core.py", line 1164, in run_engine_core\n'
    )
    assert campaign_status.dead_engine_reason(text, None) is not None


def test_dead_engine_reason_ignores_an_ordinary_apiserver_request_traceback() -> None:
    """job 640083 logs an APIServer-side "Traceback (most recent call last)" from serving.py while
    healthy and still generating tokens; only an EngineCore-tagged crash counts as dead-engine, or
    every busy arm with one failed request would be flagged dead."""
    text = "(APIServer pid=70067) ERROR 09-17 13:12:10 [serving.py:932] Traceback (most recent call last):\n"
    assert campaign_status.dead_engine_reason(text, None) is None


def test_dead_engine_reason_finds_a_bare_stale_file_handle() -> None:
    assert campaign_status.dead_engine_reason("OSError: [Errno 116] Stale file handle\n", None) is not None


def test_dead_engine_reason_finds_runtime_error_cancelled_in_the_err_file() -> None:
    assert campaign_status.dead_engine_reason(None, "RuntimeError: cancelled\n") is not None


# --------------------------------------------------------------------------------------------
# distinct kernel counting
# --------------------------------------------------------------------------------------------


def make_judge_db(path: pathlib.Path, calls: list[tuple[str, str, int, float]], submissions: list[str]) -> None:
    """A judge DB shaped like the real schema: only the columns campaign_status.py reads."""
    con = sqlite3.connect(path)
    con.execute(
        "create table calls (benchmark text, route text, correct integer, speedup real, id integer primary key)"
    )
    con.execute("create table submissions (benchmark text, id integer primary key)")
    con.executemany(
        "insert into calls (benchmark, route, correct, speedup) values (?, ?, ?, ?)",
        calls,
    )
    con.executemany("insert into submissions (benchmark) values (?)", [(b,) for b in submissions])
    con.commit()
    con.close()


def test_repeated_score_calls_on_one_kernel_count_it_once() -> None:
    """Three score calls against the same benchmark (retries within one episode) must not inflate
    kernels_scored to 3 -- the dashboard counts kernels covered, not calls made."""
    calls = [
        ("argmax_with_index", "score", 0, 0.0),
        ("argmax_with_index", "score", 0, 0.0),
        ("argmax_with_index", "score", 1, 21.1),
    ]
    metrics = campaign_status.kernel_metrics(calls, submitted=set())
    assert metrics["kernels_scored"] == 1, metrics


def test_repeated_correct_calls_on_one_kernel_contribute_one_value_to_the_speedup_median() -> None:
    """best_speedup_median is a median OVER KERNELS: a kernel scored correct five times at
    different speedups contributes its single BEST value once, not five points to the sample."""
    calls = [
        ("kernel_a", "score", 1, 2.0),
        ("kernel_a", "score", 1, 5.0),
        ("kernel_a", "submit", 1, 3.0),  # best for kernel_a is 5.0, not the submit call's 3.0
        ("kernel_b", "score", 1, 10.0),
    ]
    metrics = campaign_status.kernel_metrics(calls, submitted=set())
    # Median of {5.0, 10.0}, not of all four readings (which would be 3.5).
    assert metrics["best_speedup_median"] == 7.5, metrics


def test_kernels_correct_counts_distinct_benchmarks_not_calls() -> None:
    calls = [
        ("k1", "score", 1, 1.0),
        ("k1", "submit", 1, 1.0),
        ("k2", "score", 0, 0.0),
    ]
    metrics = campaign_status.kernel_metrics(calls, submitted=set())
    assert metrics["kernels_correct"] == 1, metrics


def test_kernels_submitted_unions_the_submit_route_with_the_submissions_table() -> None:
    """A kernel can reach 'submitted' through a submit-route call OR a submissions-table row (the
    task's own wording); a kernel present in only one of the two must still count once, not zero
    and not twice."""
    calls = [("only_call_submit", "submit", 1, 1.0)]
    metrics = campaign_status.kernel_metrics(calls, submitted={"only_table_submit", "only_call_submit"})
    assert metrics["kernels_submitted"] == 2, metrics


def test_a_score_only_kernel_that_never_reached_correct_has_no_speedup_contribution() -> None:
    """An incorrect kernel must not leak a speedup value into the median just because it was
    scored -- only benchmarks with a correct call belong in kernels_correct's speedup sample."""
    calls = [("wrong_kernel", "score", 0, 99.0)]
    metrics = campaign_status.kernel_metrics(calls, submitted=set())
    assert metrics["best_speedup_median"] is None, metrics


def test_read_calls_and_submissions_reads_the_real_schema_from_a_fixture_db(tmp_path: pathlib.Path) -> None:
    """End-to-end through sqlite: the DB reader must produce the same distinct-kernel counts as
    the pure kernel_metrics unit above, off an actual .db file shaped like judge/rank-0/*.db."""
    run_dir = tmp_path / "run"
    db_path = run_dir / "judge" / "rank-0" / "hpcagent_bench0.db"
    db_path.parent.mkdir(parents=True)
    make_judge_db(
        db_path,
        calls=[
            ("argmax_with_index", "score", 1, 21.0),
            ("argmax_with_index", "score", 1, 25.0),
            ("tsvc_2_s235", "score", 0, 0.0),
        ],
        submissions=["argmax_with_index"],
    )
    errors: list[str] = []
    calls, submitted = campaign_status.read_calls_and_submissions([db_path], errors)
    metrics = campaign_status.kernel_metrics(calls, submitted)
    assert errors == []
    assert metrics["kernels_scored"] == 2
    assert metrics["kernels_correct"] == 1
    assert metrics["kernels_submitted"] == 1
    assert metrics["best_speedup_median"] == 25.0


# --------------------------------------------------------------------------------------------
# missing-file robustness
# --------------------------------------------------------------------------------------------


def test_count_problems_on_a_missing_file_is_none_not_an_exception(tmp_path: pathlib.Path) -> None:
    assert campaign_status.count_problems(tmp_path / "does-not-exist.jsonl") is None


def test_read_env_on_a_missing_file_is_an_empty_mapping_not_an_exception(tmp_path: pathlib.Path) -> None:
    assert campaign_status.read_env(tmp_path / ".env.nope") == {}


def test_read_calls_and_submissions_on_a_missing_db_reports_an_error_and_returns_empty(tmp_path: pathlib.Path) -> None:
    errors: list[str] = []
    calls, submitted = campaign_status.read_calls_and_submissions([tmp_path / "no.db"], errors)
    assert calls == []
    assert submitted == set()
    assert errors, "a missing DB must be recorded, not silently dropped"


def test_read_calls_and_submissions_on_a_corrupt_db_file_does_not_raise(tmp_path: pathlib.Path) -> None:
    """A judge DB a rank is still writing to can be picked up mid-write; garbage bytes must read
    as zero calls plus a logged error, never propagate a sqlite3 exception up to the caller."""
    bad_db = tmp_path / "corrupt.db"
    bad_db.write_bytes(b"not a sqlite file at all")
    errors: list[str] = []
    calls, submitted = campaign_status.read_calls_and_submissions([bad_db], errors)
    assert calls == []
    assert submitted == set()
    assert errors


def test_token_totals_on_a_missing_claude_log_does_not_raise(tmp_path: pathlib.Path) -> None:
    errors: list[str] = []
    total = campaign_status.token_totals([tmp_path / "claude.log"], errors)
    assert total is None
    assert errors


def test_token_totals_with_no_worker_dirs_at_all_is_zero_not_none() -> None:
    """No claude.log anywhere (an arm whose agents have not started yet) is a real zero, distinct
    from every log being unreadable -- the dashboard should not equate "not started" with "error"."""
    errors: list[str] = []
    assert campaign_status.token_totals([], errors) == 0
    assert errors == []


def test_build_arm_never_raises_when_the_run_dir_is_entirely_absent(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inference-arm row whose .env file, .out file and run dir are all missing (a job that
    barely started) must still produce a dict with an errors list, not crash the whole report."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(campaign_status, "HERE", tmp_path)
    row = {
        "JobID": "999999",
        "JobName": "an-arm-with-nothing-on-disk-yet",
        "State": "RUNNING",
        "Elapsed": "00:00:05",
        "Start": "12:00:00",
        "End": "Unknown",
        "NNodes": "1",
        "Timelimit": "01:00:00",
        "ExitCode": "0:0",
    }
    result = campaign_status.build_arm(row, campaign_status.now_epoch(), tmp_path)
    assert result["kind"] == "other"  # no .env.<name> on disk, and no recognised helper prefix
    assert result["errors"] == []
    assert result["identity"]["jobid"] == "999999"


def test_engine_liveness_on_a_missing_out_file_is_all_none() -> None:
    result = campaign_status.engine_liveness(None)
    assert result == {"engine": None, "last_line_seconds_ago": None, "gen_throughput_tokens_s": None}


def test_framework_column_summary_with_no_matching_out_file_is_all_none(tmp_path: pathlib.Path) -> None:
    errors: list[str] = []
    summary = campaign_status.framework_column_summary("canon-llr-numba", "12345", tmp_path, errors)
    assert summary == {"csv_files": [], "ok": None, "unsupported": None, "crash": None}


# --------------------------------------------------------------------------------------------
# classification (the smoke-* / cpf-pre-* / canon-* ambiguity the task calls out)
# --------------------------------------------------------------------------------------------


def test_an_arm_whose_name_starts_with_smoke_but_has_its_own_env_file_is_an_inference_arm() -> None:
    """smoke-enroot-qwen38-c is a real arm (job 640066), not the "smoke-*" helper-job pattern --
    the .env.<name> test must win over the prefix heuristic."""
    assert campaign_status.classify("smoke-enroot-qwen38-c", env_exists=True) == "inference-arm"


def test_a_smoke_helper_job_with_no_env_file_is_other_not_inference_arm() -> None:
    assert campaign_status.classify("smoke-parallel", env_exists=False) == "other"


def test_a_cpf_pre_job_is_prerender() -> None:
    assert campaign_status.classify("cpf-pre-scicomp-focus40-gpu", env_exists=False) == "prerender"


def test_a_canon_job_is_framework_column() -> None:
    assert campaign_status.classify("canon-llr-numba", env_exists=False) == "framework-column"


def test_a_probe_job_is_other() -> None:
    assert campaign_status.classify("probe-ppcg", env_exists=False) == "other"


# --------------------------------------------------------------------------------------------
# framework-column CSV reading (canon_column.sh's own awk summary, ported to python)
# --------------------------------------------------------------------------------------------


def test_framework_column_summary_matches_canon_column_shs_own_ok_rule(tmp_path: pathlib.Path) -> None:
    """canon_column.sh's awk counts ok only when status=='ok' AND failure is empty -- a crash row
    has an empty failure field too, so status alone would double count it as ok."""
    out_root = tmp_path / "canon-llr-numba-42"
    out_root.mkdir(parents=True)
    (out_root / "canon-llr-numba-42.out").write_text("canon numba: done\n")
    csv_path = out_root / "numba.rank0.csv"
    csv_path.write_text(
        "framework,preset,datatype,kernel,impl,status,validated,median_ms,failure,error\n"
        "numba,fuzzed,float64,k1,default,ok,True,1.0,,\n"
        "numba,fuzzed,float64,k2,default,crash,False,,,segfault\n"
        "numba,fuzzed,float64,k3,default,ok,True,,unsupported,\n"
    )
    errors: list[str] = []
    summary = campaign_status.framework_column_summary("canon-llr-numba", "42", tmp_path, errors)
    assert summary["ok"] == 1, summary
    assert summary["crash"] == 1, summary
    assert summary["unsupported"] == 1, summary
    assert errors == []
