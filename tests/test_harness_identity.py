# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The agent HARNESS is part of a run's identity, stamped by the launcher and named in the registry.

An arm env that names no harness must stamp exactly what it stamped before the harness existed:
every campaign submitted today goes through ``record_identity`` with seven arguments.
"""

import pathlib
import re
import shutil
import subprocess

import pytest

from hpcagent_bench import experiment_tags, paths

SCRIPT = paths.ROOT / "experiments" / "record_identity.sh"

#: What ``record_identity`` wrote before it took a harness, for the arguments in :func:`stamp`.
PRE_HARNESS_LINES = [
    "HPCAGENT_BENCH_RECORD_EXPERIMENT=harness-focus20",
    "HPCAGENT_BENCH_RECORD_MODEL=qwen38",
    "HPCAGENT_BENCH_RECORD_LANGUAGE=c",
    "HPCAGENT_BENCH_RECORD_DEVICE=cpu",
    "HPCAGENT_BENCH_RECORD_PACKET=",
    "HPCAGENT_BENCH_RECORD_ARM=harness-focus20-qwen38-claude",
]

#: The last line stamped from a script inside a git checkout, which this test tree is.
COMMIT_LINE = (
    "HPCAGENT_BENCH_RECORD_COMMIT="
    + subprocess.run(
        ["git", "-C", str(paths.ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
)


def stamp(env: pathlib.Path, *harness: str, script: pathlib.Path = SCRIPT) -> subprocess.CompletedProcess[str]:
    """Source the launcher helper and stamp one arm, passing ``harness`` only when given."""
    identity = ("harness-focus20", "qwen38", "c", "cpu", "", "harness-focus20-qwen38-claude")
    return subprocess.run(
        ["bash", "-c", '. "$0" && record_identity "$@"', str(script), str(env), *identity, *harness],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("harness", [(), ("",)], ids=["omitted", "empty"])
def test_an_arm_that_names_no_harness_stamps_what_it_did_before(tmp_path: pathlib.Path, harness: tuple[str, ...]):
    env = tmp_path / ".env.arm"
    done = stamp(env, *harness)
    assert done.returncode == 0, done.stderr
    assert env.read_text().splitlines() == [*PRE_HARNESS_LINES, COMMIT_LINE]


def test_a_named_harness_is_stamped(tmp_path: pathlib.Path):
    env = tmp_path / ".env.arm"
    done = stamp(env, "miniswe")
    assert done.returncode == 0, done.stderr
    assert env.read_text().splitlines() == [*PRE_HARNESS_LINES, "HPCAGENT_BENCH_RECORD_HARNESS=miniswe", COMMIT_LINE]


def test_the_submitting_checkout_commit_is_stamped(tmp_path: pathlib.Path) -> None:
    """containers/agent is mounted from the submitting tree, so its commit is the code the arm ran; the
    judge cannot resolve it because the container sees the tree without its repository."""
    env = tmp_path / ".env.arm"
    done = stamp(env)
    assert done.returncode == 0, done.stderr
    assert COMMIT_LINE.removeprefix("HPCAGENT_BENCH_RECORD_COMMIT=")
    assert env.read_text().splitlines()[-1] == COMMIT_LINE


def test_a_script_outside_a_git_checkout_stamps_no_commit_rather_than_a_guess(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "record_identity.sh"
    shutil.copy(SCRIPT, script)
    env = tmp_path / ".env.arm"
    done = stamp(env, script=script)
    assert done.returncode == 0, done.stderr
    assert env.read_text().splitlines() == PRE_HARNESS_LINES


def test_an_unknown_harness_is_refused_before_anything_is_stamped(tmp_path: pathlib.Path):
    """A typo must fail at submit time, not become a fifth harness no figure names."""
    env = tmp_path / ".env.arm"
    done = stamp(env, "mini-swe")
    assert (done.returncode, env.exists()) == (2, False), done.stderr
    assert "mini-swe" in done.stderr


def test_every_harness_the_launcher_accepts_has_a_display_name():
    """Two lists of one closed set: a harness added to one and not the other is either refused at
    submit or drawn under its raw tag."""
    accepted = re.search(r'^\s*""\|([a-z|]+)\)', SCRIPT.read_text(), re.MULTILINE)
    assert accepted is not None, f"no harness case in {SCRIPT}"
    assert sorted(accepted.group(1).split("|")) == sorted(experiment_tags.names("harnesses"))


@pytest.mark.parametrize(
    "harness, want",
    [
        ("claude", "Claude Code"),
        ("miniswe", "mini-SWE-agent"),
        ("openhands", "OpenHands"),
        ("optimas", "Optimas"),
        ("brand-new-harness", "brand-new-harness"),
    ],
)
def test_a_harness_is_spelled_by_the_registry(harness: str, want: str):
    assert experiment_tags.harness_name(harness) == want


def test_the_harness_experiment_has_a_display_name():
    assert experiment_tags.display_name("harness20") == "Agent Harness Comparison, Claude Native@20"
