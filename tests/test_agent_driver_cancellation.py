# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""agent_driver.py: a task the JOB ended, as opposed to one the agent or the driver ended.

A scancel, or a step reaching the job's time limit, kills every agent where it stands. What each one
leaves behind then looks like a finished cheap task: a transcript that stops mid-episode, a token
count covering the part that ran, and whatever grade it last happened to get. Harvesting that
promotes a half-built answer, and reporting it prices part of an episode as a whole one -- so the
driver marks the task (T6) and the analysis drops it entire (X8).

The agent's OWN wall clock is not that. AGENT_TIMEOUT_SECONDS is an allowance the agent spent in
full, every submission it made along the way stands, and 604475/604476 ended 69 agents on it with
nothing else wrong; treating those as cancellations would delete the campaign.
"""

import importlib.util
import pathlib
import sys
import time
from collections.abc import Iterator
from types import ModuleType

import pytest

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "experiments"


def load_example_module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(name="driver")
def driver_fixture() -> Iterator[ModuleType]:
    module = load_example_module("agent_driver")
    yield module
    module.JOB_CANCELLED.clear()


def test_a_step_signalled_by_slurm_cancels_the_attempt_that_was_running(driver, monkeypatch) -> None:
    """scancel signals every process of the step, so the agent dies without a closing event and the
    handler has already recorded why. Without the handler the driver died with it, and the task read
    as an agent that crashed."""
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    driver.note_job_cancellation(15, None)
    assert driver.cancelled_by_the_job(returncode=-15, recorded=False) is True


def test_the_allocation_running_out_under_a_working_agent_cancels_it(driver, monkeypatch) -> None:
    """The other way a job ends: no signal reached this process yet, but the wall is here."""
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time()) + 60))
    assert driver.cancelled_by_the_job(returncode=1, recorded=False) is True


def test_an_agent_that_wrote_its_own_ending_is_not_cancelled(driver, monkeypatch) -> None:
    """A closing result event means the episode finished. Agents do finish in the last minutes of a
    job, and dropping those would delete the tasks a long arm ends on."""
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time())))
    driver.note_job_cancellation(15, None)
    assert driver.cancelled_by_the_job(returncode=0, recorded=True) is False


@pytest.mark.parametrize("returncode", [124, 125, 126, 123, 127])
def test_the_drivers_own_caps_are_not_cancellations(driver, monkeypatch, returncode: int) -> None:
    """RC_TIMEOUT, RC_TOKEN_BUDGET, RC_CONTEXT, RC_SUBMITTED, RC_API_TIMEOUT: each is an allowance
    the agent spent, and the task wall in particular must keep today's harvest."""
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time())))
    driver.note_job_cancellation(15, None)
    assert driver.cancelled_by_the_job(returncode, recorded=False) is False


def test_a_job_that_is_not_ending_cancels_nothing(driver, monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time()) + 4 * 3600))
    assert driver.cancelled_by_the_job(returncode=1, recorded=False) is False


def test_the_marker_names_the_worker_directory_the_task_was_run_in(driver, tmp_path: pathlib.Path) -> None:
    """Extraction reads the flag off this file, so it has to land beside the transcript."""
    driver.mark_cancelled(tmp_path, 137)
    assert (tmp_path / driver.CANCELLED_MARKER).read_text(encoding="utf-8").startswith("rc=137 ")
