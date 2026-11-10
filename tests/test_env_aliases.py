# Copyright 2026 the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The deprecated-name alias layer (agent/hpcagent_agent/env_aliases.py): one-time warning, precedence, and
the config keys the old names still feed (hpcagent_bench/config.py LEGACY_ENV)."""

import pathlib

import pytest
from hpcagent_agent import env_aliases
from hpcagent_agent.driver import harnesses
from hpcagent_agent.harness import runner_common

from hpcagent_bench import config, flags


@pytest.fixture(autouse=True)
def fresh_warnings() -> None:
    """Clear the warn-once registry, so each test sees its own first warning."""
    env_aliases.WARNED.clear()


def deprecation_warnings(recwarn) -> int:
    """How many DeprecationWarnings ``recwarn`` holds."""
    return len([warning for warning in recwarn if warning.category is DeprecationWarning])


def test_a_deprecated_name_warns_once_and_maps(monkeypatch, recwarn) -> None:
    """The old short name still reads, warns exactly once, and the warning names the current variable."""
    monkeypatch.setenv("AGENT_EFFORT", "medium")
    assert env_aliases.resolve("AGENT_EFFORT", "xhigh") == "medium"
    assert env_aliases.value("AGENT_EFFORT") == "medium"
    assert deprecation_warnings(recwarn) == 1
    assert "HPCAGENT_BENCH_AGENT_EFFORT" in str(recwarn[0].message)


def test_the_current_name_wins_silently(monkeypatch, recwarn) -> None:
    """When both names are set, the configuration key's name decides and nothing warns."""
    monkeypatch.setenv("AGENT_EFFORT", "medium")
    monkeypatch.setenv("HPCAGENT_BENCH_AGENT_EFFORT", "low")
    assert env_aliases.resolve("AGENT_EFFORT", "xhigh") == "low"
    assert deprecation_warnings(recwarn) == 0


def test_a_current_name_alone_never_warns(monkeypatch, recwarn) -> None:
    monkeypatch.setenv("HPCAGENT_BENCH_AGENT_HARNESS", "tuner")
    assert env_aliases.value("HARNESS") == "tuner"
    assert deprecation_warnings(recwarn) == 0


def test_an_unset_knob_is_no_value(monkeypatch, recwarn) -> None:
    monkeypatch.delenv("AGENT_EFFORT", raising=False)
    monkeypatch.delenv("HPCAGENT_BENCH_AGENT_EFFORT", raising=False)
    assert env_aliases.value("AGENT_EFFORT") is None
    assert env_aliases.resolve("AGENT_EFFORT", "xhigh") == "xhigh"
    assert deprecation_warnings(recwarn) == 0


def test_an_empty_value_is_not_an_unset_one(monkeypatch) -> None:
    """A setup that pins the empty effort rung (send no effort field) must keep its empty value."""
    monkeypatch.setenv("AGENT_EFFORT", "")
    assert env_aliases.resolve("AGENT_EFFORT", "xhigh") == ""


def test_the_judge_wait_prefers_the_current_name() -> None:
    """A runner's judge call outlives the CURRENT judge wait, not a stale short-name value beside it."""
    both = {"HPCAGENT_BENCH_TIMEOUTS_JUDGE_S": "10", "JUDGE_TIMEOUT_SECONDS": "1800"}
    assert runner_common.judge_call_timeout(both) == 10 + runner_common.JUDGE_CALL_MARGIN_SECONDS
    assert runner_common.judge_call_timeout({"JUDGE_TIMEOUT_SECONDS": "1800"}) == 1800 + runner_common.JUDGE_CALL_MARGIN_SECONDS


def test_the_runner_env_feeds_both_judge_wait_names(tmp_path: pathlib.Path) -> None:
    """The driver's default reaches a child under both names, and a pre-cleanup parent's short name is carried
    to the current one instead of being overridden by the default."""
    context = harnesses.Context(
        harness="claude",
        workdir=tmp_path,
        prompt="p",
        prompt_file=tmp_path / "prompt.txt",
        mcp_config=tmp_path / "mcp.json",
        agent_dir=tmp_path / "agent",
        cwd=tmp_path / "shared",
        replica_root="http://localhost:8000",
        kernel="k",
        language="c",
        deadline=0.0,
        marker=tmp_path / "marker",
    )
    default = harnesses.runner_env(context, {"PATH": "/bin"})
    assert default["JUDGE_TIMEOUT_SECONDS"] == "300"
    assert default["HPCAGENT_BENCH_TIMEOUTS_JUDGE_S"] == "300"
    inherited = harnesses.runner_env(context, {"PATH": "/bin", "JUDGE_TIMEOUT_SECONDS": "900"})
    assert inherited["JUDGE_TIMEOUT_SECONDS"] == "900"
    assert inherited["HPCAGENT_BENCH_TIMEOUTS_JUDGE_S"] == "900"
    current = harnesses.runner_env(context, {"PATH": "/bin", "HPCAGENT_BENCH_TIMEOUTS_JUDGE_S": "77"})
    assert current["HPCAGENT_BENCH_TIMEOUTS_JUDGE_S"] == "77"


def test_ncores_reads_the_current_name_over_the_deprecated_one(monkeypatch) -> None:
    """``flags.ncores`` keeps reading the deprecated name, and the configuration key's name beats it."""
    monkeypatch.setattr(flags, "physical_cores", lambda cpus: 64)
    monkeypatch.setenv("HPCAGENT_BENCH_NCORES", "64")
    assert flags.ncores() == 64
    monkeypatch.setenv("HPCAGENT_BENCH_LIMITS_CORES", "8")
    assert flags.ncores() == 8


def test_every_deprecated_name_maps_a_live_config_key() -> None:
    """The config layer's table is the alias table folded to dotted keys: nothing stale, nothing missing."""
    assert config.LEGACY_ENV == {old: config.ENV_KEYS[current] for old, current in env_aliases.DEPRECATED.items()}
    assert set(config.LEGACY_ENV) == set(env_aliases.DEPRECATED)


def test_a_legacy_value_warns_once_and_reaches_its_key(monkeypatch, recwarn) -> None:
    """``config.legacy_value`` is the reader-side last layer: the old value, one warning."""
    monkeypatch.setenv("AGENT_EFFORT", "high")
    assert config.legacy_value("agent.effort") == "high"
    assert config.legacy_value("agent.effort") == "high"
    assert deprecation_warnings(recwarn) == 1


def test_the_config_key_beats_the_deprecated_name(monkeypatch) -> None:
    """An override of the key wins over a legacy value in the environment; the alias still reads for a reader
    that found no key value."""
    monkeypatch.setenv("HPCAGENT_BENCH_NCORES", "64")
    with config.overridden("limits.cores", 8):
        assert config.get("limits.cores") == 8
    assert config.legacy_value("limits.cores") == "64"


def test_a_key_without_an_alias_has_no_legacy_value() -> None:
    assert config.legacy_value("measurement.repeat") is None
