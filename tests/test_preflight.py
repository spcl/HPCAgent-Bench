# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The batch-job preflight: which columns run here, and the thread env it emits."""

import re

from hpcagent_bench.flags import Mode
from hpcagent_bench.harness import preflight


def test_an_agent_column_is_not_a_deterministic_one() -> None:
    """A deterministic sweep has no inference or judge role to place, so naming an agent column is
    a submission error -- caught before the allocation is spent, not during it."""
    assert preflight.check_deterministic(["numpy", "dace_cpu"]) == []
    assert preflight.check_deterministic(["numpy", "openai_agent"]) == ["openai_agent"]


def test_deterministic_frameworks_matches_the_frozen_pre_derivation_list() -> None:
    """:data:`preflight.DETERMINISTIC_FRAMEWORKS` is now derived from
    ``FRAMEWORK_META[name]["sweep_deterministic"]`` rather than a second hand-kept list; this pins
    the derived set to the exact set the hand-kept list named, so moving the data does not silently
    add or drop a column a deterministic sweep may select."""
    frozen = {
        "numpy",
        "polly",
        "pluto",
        "cc",
        "cc_autopar",
        "llvm",
        "cpp",
        "fortran",
        "fortran_autopar",
        "flang",
        "dace_cpu",
        "dace_cpu_autoopt",
        "dace_cpu_canonicalize",
        "dace_cpu_parallel",
        "dace_gpu",
        "dace_gpu_autoopt",
        "dace_gpu_canonicalize",
        "dace_gpu_parallel",
    }
    assert set(preflight.DETERMINISTIC_FRAMEWORKS) == frozen


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
