# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The batch-job preflight, and the invariant that submission scripts do not re-implement it.

Three checks every submission script needs -- are these columns runnable here, does the installed
dace carry the fork's pipeline, does this node's compiler genuinely parallelize -- used to be
inlined in each script. They drifted: one grew a hand-rolled C probe that compiled ``-O3 <delta>``
and grepped for ``GOMP``, a weaker duplicate of ``flags.probe_autopar`` (which compiles the
column's REAL composed flags and accepts an outlined symbol as evidence too), and the Alps script
never grew the check at all.
"""
import pathlib
import re
from typing import List

import pytest

from hpcagent_bench.flags import Mode
from hpcagent_bench.harness import preflight

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The submission scripts that must delegate rather than re-implement.
SUBMIT_SCRIPTS = (
    REPO / "scripts" / "submit_deterministic.sbatch",
    REPO / "scripts" / "cscs" / "submit_foundation_alps.sbatch",
)


def script_texts() -> List[str]:
    return [path.read_text() for path in SUBMIT_SCRIPTS if path.is_file()]


def test_an_agent_column_is_not_a_deterministic_one() -> None:
    """A deterministic sweep has no inference or judge role to place, so naming an agent column is
    a submission error -- caught before the allocation is spent, not during it."""
    assert preflight.check_deterministic(["numpy", "dace_cpu"]) == []
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
    assert dict(line[len("export "):].split("=", 1) for line in env) == preflight.thread_env(Mode.MULTI_CORE)


def test_print_env_off_emits_nothing_to_eval() -> None:
    _, _, env = preflight.run(["numpy"], print_env=False)
    assert env == []


@pytest.mark.parametrize("path", [p for p in SUBMIT_SCRIPTS], ids=lambda p: p.name)
def test_the_submission_scripts_delegate_the_checks(path: pathlib.Path) -> None:
    """Both scripts must CALL the preflight, so a change to what a job checks lands in one place."""
    assert path.is_file(), path
    assert "hpcagent-bench preflight" in path.read_text(), f"{path.name} does not call the library preflight"


def test_no_script_hand_rolls_the_autopar_probe() -> None:
    """The regression this guards: a script compiling its own probe and grepping nm for GOMP is a
    second, weaker implementation of flags.probe_autopar, and it is how the two scripts diverged
    the first time -- one had a probe, the other silently had none."""
    for text in script_texts():
        assert "GOMP" not in text, "a submission script is grepping for GOMP; call flags.probe_autopar instead"
        assert "nm -u" not in text, "a submission script is inspecting objects itself"


def test_no_script_reimplements_cpu_env() -> None:
    """Same rule for thread counts: one source (flags.cpu_env), reached through the preflight."""
    for text in script_texts():
        assert "flags.cpu_env" not in text, "a submission script is importing cpu_env directly"
