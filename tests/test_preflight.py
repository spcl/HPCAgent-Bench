# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The batch-job preflight: which columns run here, and the thread env it emits."""

import re

from hpcagent_bench.flags import Mode
from hpcagent_bench.harness import preflight


def test_an_agent_column_is_not_a_deterministic_one() -> None:
    """A deterministic sweep has no inference or judge role to place, so naming an agent column is
    a submission error -- caught before the allocation is spent, not during it."""
    assert preflight.check_deterministic(["numpy", "dace_cpu_parallel"]) == []
    assert preflight.check_deterministic(["numpy", "openai_agent"]) == ["openai_agent"]


def test_a_fatal_finding_exits_non_zero_and_emits_no_env() -> None:
    """The caller EVALS the env list. A fatal preflight must hand it nothing, so a refused job
    cannot half-configure itself from a partial result."""
    code, report, env = preflight.run(["openai_agent"], print_env=True)
    assert code == 1
    assert env == []
    assert any("FATAL" in line for line in report)


def test_report_and_env_are_separate_streams() -> None:
    """The hazard this split exists for: a diagnostic on the eval'd stream would be RUN as a
    shell command. Every env line must be an export and nothing else may be."""
    code, report, env = preflight.run(["numpy"], print_env=True)
    assert code == 0
    assert env, "no thread-count exports emitted"
    assert all(re.fullmatch(r"export [A-Z_]+=\S+", line) for line in env), env
    assert not any(line.startswith("export ") for line in report)


def test_env_is_the_documented_thread_source() -> None:
    """Not a second opinion about core counts: the same flags.cpu_env the harness documents."""
    _, _, env = preflight.run(["numpy"], print_env=True)
    assert dict(line[len("export ") :].split("=", 1) for line in env) == preflight.thread_env(Mode.MULTI_CORE)


def test_print_env_off_emits_nothing_to_eval() -> None:
    _, _, env = preflight.run(["numpy"], print_env=False)
    assert env == []
