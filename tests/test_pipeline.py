# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The two-stage think -> grade pipeline: the judge re-grade folds authoritatively onto the
agent's think row, a re-verify failure downgrades it, and a remote judge slot dispatches the
grade over srun. Every test fakes the agent + the scorer -- no LLM, no compile, no GPU."""
import json

import optarena.agent_bench.pipeline as pipeline
from optarena.agent_bench.envelope import Submission
from optarena.agent_bench.judge_scheduler import DeviceSlot
from optarena.agent_bench.runner import RunRow, CallPoint
from optarena.agent_bench.scoring import Score, VerifyResult
from optarena.agent_bench.task import Task


def make_think_row(**over) -> RunRow:
    """A think stage's self-graded row -- the PROXY the judge re-grade overwrites. Carries the
    agent-side provenance (tokens / trajectory / prompt) that must survive the merge."""
    base = dict(
        task_id="gemm::c",
        kernel="gemm",
        language="c",
        source_mode="restricted",
        agent="stub",
        status="ok",
        correct=True,
        max_rel_error=0.0,
        native_ns=100,
        speedup=9.9,  # optimistic agent-side proxy speedup
        tokens=1234,
        trajectory=(CallPoint(1, 1234, 9.9, True, "ok"), ),
        prompt="the-prompt",
        rounds=1)
    base.update(over)
    return RunRow(**base)


def make_score(**over) -> Score:
    base = dict(
        correct=True,
        max_rel_error=1e-9,
        native_ns=200,
        build_ok=True,
        detail="",
        baseline_ns=400,
        speedup=2.0,  # authoritative judge speedup (differs from the proxy)
        baseline="numpy",
        public_correct=True,
        hidden_correct=True,
        hidden_passed=3,
        hidden_total=3,
        baselines={"numpy": 400},
        speedups={"numpy": 2.0},
        oracle="numpy")
    base.update(over)
    return Score(**base)


def a_submission() -> Submission:
    return Submission(language="c", source="void kernel_gemm(){}")


def local_grade(verify=True, **params):
    """A grade closure over a LOCAL cpu judge slot with the given grading params."""
    p = dict(preset="S", datatype="float64", repeat=1, oracle="numpy", baseline="numpy", verify=verify)
    p.update(params)
    return pipeline.make_grade(("srun", "--nodelist", "{node}", "-n", "1"), **p)


# ---- merge: authoritative measurement wins, provenance preserved -------------


def test_grade_overwrites_proxy_with_authoritative(monkeypatch):
    monkeypatch.setattr(pipeline, "score", lambda *a, **k: make_score(speedup=2.0, native_ns=200))
    grade = local_grade(verify=False)
    row = grade((make_think_row(speedup=9.9), a_submission()), Task("gemm", "restricted", "c"), DeviceSlot("cpu", 0))
    assert row.speedup == 2.0 and row.native_ns == 200 and row.baseline_ns == 400  # judge numbers, not the proxy
    assert row.tokens == 1234 and row.prompt == "the-prompt" and row.rounds == 1  # agent provenance survives
    assert row.trajectory and row.trajectory[0].speedup == 9.9  # the proxy trajectory is kept verbatim
    assert row.correct and row.status == "ok"


def test_grade_reverify_failure_downgrades_to_unverified(monkeypatch):
    monkeypatch.setattr(pipeline, "score", lambda *a, **k: make_score(correct=True, speedup=2.0))
    monkeypatch.setattr(pipeline, "independent_verify",
                        lambda *a, **k: VerifyResult(False, False, True, True, True, False, "fresh-seed-mismatch"))
    grade = local_grade(verify=True)
    row = grade((make_think_row(), a_submission()), Task("gemm", "restricted", "c"), DeviceSlot("gpu", 0))
    assert row.correct is False and row.status == "unverified"
    assert "judge re-verify failed" in row.detail and "fresh-seed-mismatch" in row.detail


def test_grade_reverify_pass_keeps_correct(monkeypatch):
    monkeypatch.setattr(pipeline, "score", lambda *a, **k: make_score(correct=True))
    monkeypatch.setattr(pipeline, "independent_verify",
                        lambda *a, **k: VerifyResult(True, True, True, True, True, False, ""))
    grade = local_grade(verify=True)
    row = grade((make_think_row(), a_submission()), Task("gemm", "restricted", "c"), DeviceSlot("cpu", 0))
    assert row.correct is True and row.status == "ok"


def test_grade_passthrough_when_agent_produced_nothing(monkeypatch):
    # No submission -> nothing for the judge to time: the think row (an agent_error/timeout)
    # is returned untouched and score() is never called.
    monkeypatch.setattr(pipeline, "score", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not grade")))
    grade = local_grade()
    think_row = make_think_row(status="agent_error", correct=False, speedup=0.0)
    row = grade((think_row, None), Task("gemm", "restricted", "c"), DeviceSlot("cpu", 0))
    assert row is think_row


# ---- remote judge slot: srun dispatch + JSON round-trip ----------------------


def test_grade_remote_srun_argv_and_parse(monkeypatch, tmp_path):
    from optarena import config

    captured = {}

    def fake_run(argv, check=False, **kw):
        captured["argv"] = argv
        infile = argv[argv.index("--input") + 1]
        outfile = argv[argv.index("--output") + 1]
        req = json.loads(open(infile).read())
        captured["req"] = req
        # Simulate the remote judge grading the shipped submission.
        payload = pipeline.grade_result_to_json((make_score(speedup=3.5), None))
        open(outfile, "w").write(json.dumps(payload))

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    config.set_override("pipeline.exchange_dir", str(tmp_path))
    try:
        grade = local_grade(verify=False)
        row = grade((make_think_row(), a_submission()), Task("gemm", "restricted", "c"), DeviceSlot("gpu", 2, "nid007"))
    finally:
        config.clear_override("pipeline.exchange_dir")
    argv = captured["argv"]
    assert argv[:3] == ["srun", "--nodelist", "nid007"]  # srun_wrap targeted the slot's node
    assert "grade-submission" in argv and "-m" in argv and "optarena.cli" in argv
    assert captured["req"]["kernel"] == "gemm" and captured["req"]["submission"]["source"]
    assert row.speedup == 3.5  # the remote judge's number folded onto the think row


def test_grade_result_json_roundtrip():
    vr = VerifyResult(True, True, True, True, True, False, "")
    for graded in ((make_score(speedup=2.0), vr), (make_score(), None)):
        back = pipeline.grade_result_from_json(pipeline.grade_result_to_json(graded))
        assert back[0].speedup == graded[0].speedup and back[0].baselines == graded[0].baselines
        assert (back[1] is None) == (graded[1] is None)


def test_grade_request_task_roundtrip():
    task = Task("gemm", "restricted", "c")
    req = pipeline.grade_request_to_json(a_submission(), task, {"preset": "S"})
    back = pipeline.task_from_request(req)
    assert (back.kernel, back.source_mode, back.language, back.residency) == ("gemm", "restricted", "c", "host")


# ---- run_pipeline end to end (fake agent + fake scorer, all-local slots) -----


def test_run_pipeline_orders_and_regrades(monkeypatch):
    # think just echoes a per-kernel proxy row; grade authoritatively re-times to 2.0.
    def fake_solve(agent, task, **k):
        return make_think_row(task_id=task.id, kernel=task.kernel, speedup=9.9), a_submission()

    monkeypatch.setattr(pipeline, "solve_task", fake_solve)
    monkeypatch.setattr(pipeline, "score", lambda *a, **k: make_score(speedup=2.0))
    monkeypatch.setattr(pipeline, "independent_verify",
                        lambda *a, **k: VerifyResult(True, True, True, True, True, False, ""))
    # Force an all-local single-slot pool (no env nodelist, no cupy) so nothing sruns.
    monkeypatch.delenv("OPTARENA_AGENT_NODES_EXPANDED", raising=False)
    monkeypatch.delenv("OPTARENA_JUDGE_NODES_EXPANDED", raising=False)
    monkeypatch.setattr("optarena.agent_bench.judge_scheduler.local_gpu_count", lambda: 0)

    tasks = [Task(k, "restricted", "c") for k in ("gemm", "gesummv", "atax")]
    rows = pipeline.run_pipeline(lambda: object(),
                                 tasks,
                                 preset="S",
                                 datatype="float64",
                                 repeat=1,
                                 oracle="numpy",
                                 baseline="numpy")
    assert [r.kernel for r in rows] == ["gemm", "gesummv", "atax"]  # input order preserved
    assert all(r.speedup == 2.0 and r.correct for r in rows)  # authoritative, not the 9.9 proxy


# ---- pipeline_enabled gating -------------------------------------------------


def test_pipeline_enabled_flag_and_env(monkeypatch):
    monkeypatch.delenv("OPTARENA_AGENT_NODES_EXPANDED", raising=False)
    monkeypatch.delenv("OPTARENA_JUDGE_NODES_EXPANDED", raising=False)
    monkeypatch.setattr("optarena.agent_bench.pipeline.AgentPoolConfig.from_config",
                        classmethod(lambda cls: pipeline.AgentPoolConfig(workers_per_node=1)))
    assert pipeline.pipeline_enabled("on") is True
    assert pipeline.pipeline_enabled("off") is False
    assert pipeline.pipeline_enabled("auto") is False  # single-box: no pool -> serial path
    monkeypatch.setenv("OPTARENA_JUDGE_NODES_EXPANDED", "nid001,nid002")
    assert pipeline.pipeline_enabled("auto") is True  # a judge nodelist marks the campaign
    monkeypatch.setattr("optarena.agent_bench.pipeline.AgentPoolConfig.from_config",
                        classmethod(lambda cls: pipeline.AgentPoolConfig(workers_per_node=4)))
    monkeypatch.delenv("OPTARENA_JUDGE_NODES_EXPANDED", raising=False)
    assert pipeline.pipeline_enabled("auto") is True  # >1 agent worker also marks it
