# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``experiments/run_cluster.sh``'s service-death handling, right after it launches the three role
steps and waits on whichever dies first.

Two bugs this covers:

1. When a service step (vLLM or the judge) died while the agents were still running, the batch step
   used to ``exit 1`` immediately -- before the mandatory token-record extraction below it ever ran,
   and without writing ``EXTRACTION_FAILED``, so nothing on disk said the extraction was skipped.
2. The extraction's own success/failure marker was written only AT that point, which races a SIGTERM
   (scancel, or the job's time limit) against SIGKILL (KillWait): if the process is killed before it
   gets there, no marker is left at all. The fix writes the marker unconditionally, before either
   step can die, and only clears it once extraction actually succeeds -- so every exit from here on
   leaves the run either extracted or visibly marked for re-extraction, with nothing depending on
   catching the signal that ends it.

These tests lift the exact block -- the marker write, the ``wait -n``, and the branch that now stops
the agent step before falling through instead of exiting -- straight out of the file (never retyped)
and run it standalone with two background processes standing in for the service and agent role
steps, the same "read the real text, run only the real text" approach the other run_cluster.sh tests
in this directory take.
"""

import os
import pathlib
import re
import signal
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "experiments" / "run_cluster.sh").read_text()

START = "# The extraction below is MANDATORY,"
END = "# Post-run utilization verdicts"
BLOCK = TEXT[TEXT.index(START) : TEXT.index(END)]

assert "agent_status=1" in BLOCK, "the service-death branch no longer sets agent_status directly"
# A bare `exit 1` STATEMENT (not the phrase inside this file's own comment prose above) would skip
# the mandatory extraction that follows this block again.
assert not re.search(r"^\s*exit 1\s*$", BLOCK, flags=re.MULTILINE), (
    "the service-death branch exits before extraction again"
)


def run(setup: str, tail: str, run_dir: pathlib.Path) -> subprocess.CompletedProcess[str]:
    # setup runs BEFORE the lifted block: the block's first real statement is `wait -n`, so the
    # stand-in service/agent background jobs and agent_step_pid must already exist by then.
    script = f'#!/usr/bin/env bash\nset -euo pipefail\nRUN_DIR="{run_dir}"\n{setup}\n{BLOCK}\n{tail}\n'
    return subprocess.run(
        ["bash", "-c", script],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_a_dying_service_step_stops_the_agent_and_falls_through_instead_of_exiting(
    tmp_path: pathlib.Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tail = (
        'if kill -0 "${agent_step_pid}" 2>/dev/null; then echo AGENT_STILL_ALIVE; else echo AGENT_STOPPED; fi\n'
        'echo "agent_status=${agent_status}"\n'
        "echo REACHED_END"
    )
    result = run(
        "( sleep 0.2; exit 7 ) &\nsleep 30 & agent_step_pid=$!",
        tail,
        run_dir,
    )
    assert result.returncode == 0, result.stderr
    # The block itself never exits: falling through to REACHED_END is what lets the mandatory
    # extraction below it (not part of this lifted slice) still run.
    assert "REACHED_END" in result.stdout, result.stdout
    assert "AGENT_STOPPED" in result.stdout, result.stdout
    assert "agent_status=1" in result.stdout, result.stdout
    assert (run_dir / "EXTRACTION_FAILED").exists(), "the marker was not written before the steps ran"


def test_the_agent_finishing_first_leaves_the_surviving_service_step_alone(tmp_path: pathlib.Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tail = (
        'if kill -0 "${service_pid}" 2>/dev/null; then echo SERVICE_STILL_ALIVE; else echo SERVICE_STOPPED; fi\n'
        'echo "agent_status=${agent_status}"\n'
        'echo "service_pid=${service_pid}"\n'
        "echo REACHED_END"
    )
    result = run(
        # The service stand-in survives the block (that is what this test checks), so its stdout is
        # redirected away: inherited from a `&` job, it would otherwise keep subprocess.run's output
        # pipe open long after bash itself exits.
        "sleep 30 >/dev/null 2>&1 & service_pid=$!\n( sleep 0.2; exit 3 ) & agent_step_pid=$!",
        tail,
        run_dir,
    )
    try:
        assert result.returncode == 0, result.stderr
        assert "REACHED_END" in result.stdout, result.stdout
        # The else branch (agent finished, not a service) must not touch the still-running service step.
        assert "SERVICE_STILL_ALIVE" in result.stdout, result.stdout
        assert "agent_status=3" in result.stdout, result.stdout
        assert (run_dir / "EXTRACTION_FAILED").exists(), "the marker was not written before the steps ran"
    finally:
        # This test's own untouched stand-in for the surviving service step; nothing in the lifted
        # block manages it, so the test reaps it itself instead of leaking a 30s sleep per run.
        match = re.search(r"service_pid=(\d+)", result.stdout)
        if match:
            try:
                os.kill(int(match.group(1)), signal.SIGKILL)
            except ProcessLookupError:
                pass
