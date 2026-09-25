# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``CLEAN=1`` and ``DEADLINE=`` in experiments/submit-scicomp-perf-playbook.sh, driven with SUBMIT=0.

Ported from experiments/submit-scicomp-dc.sh and submit-gpu-llr40.sh (same submit_common.sh wiring:
clean_suffix, deadline_setup, deadline_shrink_seconds): a clean re-run has to be recognisable from
the arm name alone, and a wave launched against a deadline has to END before it rather than be
killed mid-episode. Both are decided before a single node is allocated, so both are checked without
touching the queue.
"""

import datetime
import os
import pathlib
import re
import subprocess
import sys

import pytest

from tests.test_submit_scicomp_perf_playbook import KNOBS, LAUNCHER, REPO, env_dict, submit_tree

#: The CLEAN/DEADLINE knobs, on top of the launcher's own -- stripped from the inherited
#: environment so only a test's explicit values apply.
CLEAN_KNOBS = KNOBS | {"CLEAN", "DEADLINE", "DEADLINE_MARGIN_SECONDS", "MIN_AGENT_SECONDS", "STAGING_HOURS", "BEGIN"}

#: submit-scicomp-perf-playbook.sh's own default: the episode every arm of this campaign runs.
CONFIGURED_AGENT_SECONDS = 72000
#: arm_nodes.sh: image pull, engine start and the readiness probe, before any agent runs.
STAGING_SECONDS = 3 * 3600
#: submit-scicomp-perf-playbook.sh: the slack between the job's own end and the deadline.
MARGIN_SECONDS = 300
#: How far ahead the deadline is placed. Far enough that the job limit alone would allow a LONGER
#: episode than the campaign's, which is the case the cap exists for.
HOURS_AHEAD = 30
#: Close enough that the deadline, not the campaign, decides the episode.
HOURS_TIGHT = 15


def deadline(hours: float) -> str:
    """An ISO timestamp ``hours`` from now, spelled the way an operator types DEADLINE."""
    return (datetime.datetime.now() + datetime.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")


def seconds_of(walltime: str) -> int:
    hours, minutes, seconds = (int(part) for part in walltime.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def prepared(result: subprocess.CompletedProcess[str]) -> dict[str, str]:
    """``{arm: walltime}`` off the launcher's SUBMIT=0 report."""
    pattern = re.compile(r"^prepared (\S+) \(\d+ nodes, (\d\d:\d\d:\d\d), .*\) -- not submitted")
    return {match.group(1): match.group(2) for line in result.stdout.splitlines() if (match := pattern.match(line))}


def built_lines(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in result.stdout.splitlines() if line.startswith("prepared ")]


def arm_env(experiments: pathlib.Path, kind: str) -> pathlib.Path:
    return experiments / f".env.scicomp-perf-playbook-qwen38-{kind}"


def run_submit(root: pathlib.Path, **knobs: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in CLEAN_KNOBS and not k.startswith("SLURM_")}
    defaults = {
        "PATH": f"{root / 'bin'}:{env['PATH']}",
        "PY": sys.executable,
        "HPCAGENT_BENCH_REPO": str(REPO),
        "PYTHONPATH": f"{REPO}",
        "STAMP": "20260915",
        "STUB_MARKERS": str(root),
        "SUBMIT": "0",
        "MODELS": "qwen38",
        "REPEAT": "1",
        "JUDGE_NODES": "1",
    }
    env.update({**defaults, **knobs})
    return subprocess.run(
        ["bash", str(root / "experiments" / LAUNCHER)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


@pytest.fixture(name="clean", scope="module")
def clean_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[pathlib.Path, subprocess.CompletedProcess[str]]:
    """One CLEAN=1 wave (plain + perf-playbook-cpu) under a deadline far enough out that it never
    shrinks the episode."""
    tmp_path = tmp_path_factory.mktemp("clean")
    root = submit_tree(tmp_path)
    result = run_submit(root, CLEAN="1", DEADLINE=deadline(HOURS_AHEAD))
    return root / "experiments", result


def test_a_clean_wave_names_every_arm_and_every_job_with_the_suffix(
    clean: tuple[pathlib.Path, subprocess.CompletedProcess[str]],
) -> None:
    """The suffix is the whole mechanism: nothing else tells the analysis or the board that these
    tasks supersede the ones before them."""
    _, result = clean
    assert result.returncode == 0, result.stderr
    assert sorted(prepared(result)) == [
        "scicomp-perf-playbook-qwen38-perf-playbook-cpu-clean",
        "scicomp-perf-playbook-qwen38-plain-clean",
    ]


def test_a_clean_arm_keeps_the_identity_the_analysis_pairs_on(
    clean: tuple[pathlib.Path, subprocess.CompletedProcess[str]],
) -> None:
    """The suffix names no condition. An arm that also moved its packet or language would be a new
    condition with nothing to pair against."""
    experiments, _ = clean
    values = env_dict(arm_env(experiments, "perf-playbook-cpu-clean"))
    assert values["HPCAGENT_BENCH_RECORD_PACKET"] == "perf-playbook-cpu"
    assert values["HPCAGENT_BENCH_RECORD_LANGUAGE"] == "c"
    assert values["HPCAGENT_BENCH_RECORD_EXPERIMENT"] == "scicomp-focus40"
    assert values["HPCAGENT_BENCH_RECORD_ARM"] == "scicomp-perf-playbook-qwen38-perf-playbook-cpu-clean"


def test_a_deadline_shrinks_the_job_but_never_lengthens_the_episode(
    clean: tuple[pathlib.Path, subprocess.CompletedProcess[str]],
) -> None:
    """A deadline is a bound on the JOB, and a longer episode is a different condition: an arm whose
    agents get more time than the arms it is compared with is not comparable with them, nor with the
    same arm submitted an hour later under the same deadline."""
    experiments, result = clean
    reported = set(prepared(result).values())
    assert len(reported) == 1, reported
    limit = seconds_of(reported.pop())
    assert CONFIGURED_AGENT_SECONDS < limit - STAGING_SECONDS
    values = env_dict(arm_env(experiments, "perf-playbook-cpu-clean"))
    assert int(values["AGENT_TIMEOUT_SECONDS"]) == CONFIGURED_AGENT_SECONDS
    # The clock moves while the launcher runs, so the target is an upper bound, not an equality.
    target = HOURS_AHEAD * 3600 - MARGIN_SECONDS
    assert target - 120 <= limit <= target


def test_a_deadline_the_episode_does_not_fit_in_shortens_the_episode(tmp_path: pathlib.Path) -> None:
    """The other half of the same rule: what the job cannot cover, the agents do not get, or the job
    dies at its limit with the last batch ungraded and the arm is partly its own control."""
    root = submit_tree(tmp_path)
    result = run_submit(root, DEADLINE=deadline(HOURS_TIGHT))
    assert result.returncode == 0, result.stderr
    walltime = next(iter(prepared(result).values()))
    env = env_dict(root / "experiments" / ".env.scicomp-perf-playbook-qwen38-plain")
    agent = int(env["AGENT_TIMEOUT_SECONDS"])
    assert agent == seconds_of(walltime) - STAGING_SECONDS < CONFIGURED_AGENT_SECONDS


def test_a_deadline_wave_starts_now_instead_of_waiting(
    clean: tuple[pathlib.Path, subprocess.CompletedProcess[str]],
) -> None:
    """A DEADLINE wave must not be left waiting on a BEGIN nobody set."""
    _, result = clean
    for line in built_lines(result):
        assert " begin " not in line, line


def test_a_deadline_too_close_to_measure_anything_refuses_the_wave(tmp_path: pathlib.Path) -> None:
    """Under an hour of agent time buys a handful of turns per kernel and an arm of build failures;
    the nodes are better left in the queue."""
    root = submit_tree(tmp_path)
    result = run_submit(root, CLEAN="1", DEADLINE=deadline(0.9))
    assert result.returncode == 2, result.stdout
    assert "under the 3600s floor" in result.stderr
    assert list((root / "experiments").glob(".env.scicomp-perf-playbook-*")) == []


def test_a_wave_without_clean_or_a_deadline_is_unchanged(tmp_path: pathlib.Path) -> None:
    """Every wave so far ran without either, and neither default may move an arm name or a limit."""
    root = submit_tree(tmp_path)
    result = run_submit(root)
    assert result.returncode == 0, result.stderr
    assert sorted(prepared(result)) == [
        "scicomp-perf-playbook-qwen38-perf-playbook-cpu",
        "scicomp-perf-playbook-qwen38-plain",
    ]
    # arm_walltime: 2 kernels in 1 batch (AGENTS_PER_NODE=30) of AGENT_TIMEOUT_SECONDS (72000/3600=20h)
    # plus the staging allowance (3h).
    assert set(prepared(result).values()) == {"23:00:00"}
    assert " begin " not in built_lines(result)[0]
