"""Unified auto-tuner interface + budget (optarena.autotune).

Pins the ONE-knob contract: every searching optimizer (TVM MetaSchedule, Triton
autotune, an Agent) draws its budget from :class:`TuningBudget`, and a framework
declares whether it is an auto-tuner.
"""

import pytest

from optarena.autotune import SCALES, AutoTuner, IdentityTuner, TuningBudget


def test_budget_scales():
    small = TuningBudget.from_env("small")
    full = TuningBudget.from_env("full")
    assert (small.trials, small.configs) == SCALES["small"]
    assert (full.trials, full.configs) == SCALES["full"]
    # a bare integer caps both backends explicitly.
    custom = TuningBudget.from_env("48")
    assert custom.scale == "custom" and custom.trials == 48 and custom.configs == 48
    # garbage falls back to the default scale.
    assert TuningBudget.from_env("nonsense").scale == "small"


def test_env_default(monkeypatch):
    monkeypatch.delenv("OPTARENA_TUNE_BUDGET", raising=False)
    assert TuningBudget.from_env().scale == "small"
    monkeypatch.setenv("OPTARENA_TUNE_BUDGET", "full")
    assert TuningBudget.from_env().scale == "full"


def test_legacy_env_overrides(monkeypatch):
    # The legacy per-framework knobs still win over the unified default.
    b = TuningBudget.from_env("small")
    monkeypatch.setenv("OPTARENA_TVM_METASCHEDULE_TRIALS", "200")
    assert b.tvm_trials() == 200
    monkeypatch.setenv("OPTARENA_TVM_METASCHEDULE_TRIALS", "full")
    assert b.tvm_trials() == SCALES["full"][0]
    monkeypatch.delenv("OPTARENA_TVM_METASCHEDULE_TRIALS", raising=False)
    assert b.tvm_trials() == SCALES["small"][0]

    monkeypatch.setenv("OPTARENA_TRITON_AUTOTUNE_SIZE", "full")
    assert b.triton_config_cap() == SCALES["full"][1]
    monkeypatch.delenv("OPTARENA_TRITON_AUTOTUNE_SIZE", raising=False)
    monkeypatch.setenv("OPTARENA_TRITON_AUTOTUNE_N", "7")
    assert b.triton_config_cap() == 7


def test_identity_tuner_returns_program_unchanged():
    obj = object()
    assert IdentityTuner().tune(obj, TuningBudget.from_env()) is obj


def test_autotuner_is_abstract():
    with pytest.raises(TypeError):
        AutoTuner()  # abstract: tune() unimplemented


def test_framework_declares_tuner_status():
    from optarena.infrastructure.framework import Framework, generate_framework
    np_fw = generate_framework("numpy")
    assert np_fw.is_autotuner is False
    assert np_fw.tuning_budget() is None

    class Tuner(Framework):
        is_autotuner = True

    t = Tuner("numpy")
    b = t.tuning_budget()
    assert isinstance(b, TuningBudget)


def test_tvm_and_triton_are_autotuners():
    # Class-level flag (no tvm/triton install needed to read it).
    from optarena.infrastructure.triton_framework import TritonFramework
    from optarena.infrastructure.tvm_cpu_framework import TVMCPUFramework
    assert TVMCPUFramework.is_autotuner is True
    assert TritonFramework.is_autotuner is True


def test_metaschedule_trials_delegates_to_budget(monkeypatch):
    from optarena.infrastructure.tvm_cpu_framework import metaschedule_trials
    monkeypatch.delenv("OPTARENA_TVM_METASCHEDULE_TRIALS", raising=False)
    monkeypatch.setenv("OPTARENA_TUNE_BUDGET", "full")
    assert metaschedule_trials() == SCALES["full"][0]
    monkeypatch.setenv("OPTARENA_TUNE_BUDGET", "small")
    assert metaschedule_trials() == SCALES["small"][0]


def test_agent_budget_tokens():
    from optarena.agent_bench.agent import budget_tokens
    assert budget_tokens(None, 512) == 512
    assert budget_tokens(256, 512) == 256
    assert budget_tokens(TuningBudget(scale="x", trials=1, configs=1, cost=1024), 512) == 1024
    assert budget_tokens(TuningBudget.from_env("small"), 512) == 512  # no cost -> default
