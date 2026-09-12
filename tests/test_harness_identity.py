# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The agent HARNESS is part of a run's identity, stamped by the launcher and named in the registry.

An arm env that names no harness must stamp exactly what it stamped before the harness existed:
every campaign submitted today goes through ``record_identity`` with seven arguments.
"""

import importlib.util
import pathlib
import re
import subprocess
import sys

import pytest

from hpcagent_bench import experiment_tags, paths
from hpcagent_bench.harness import recording

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

MIGRATE_SPEC = importlib.util.spec_from_file_location("migrate_db", paths.ROOT / "scripts" / "migrate_db.py")
migrate = importlib.util.module_from_spec(MIGRATE_SPEC)
sys.modules[MIGRATE_SPEC.name] = migrate
MIGRATE_SPEC.loader.exec_module(migrate)


def stamp(env: pathlib.Path, *harness: str) -> subprocess.CompletedProcess[str]:
    """Source the launcher helper and stamp one arm, passing ``harness`` only when given."""
    identity = ("harness-focus20", "qwen38", "c", "cpu", "", "harness-focus20-qwen38-claude")
    return subprocess.run(
        ["bash", "-c", '. "$0" && record_identity "$@"', str(SCRIPT), str(env), *identity, *harness],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("harness", [(), ("",)], ids=["omitted", "empty"])
def test_an_arm_that_names_no_harness_stamps_what_it_did_before(tmp_path: pathlib.Path, harness: tuple[str, ...]):
    env = tmp_path / ".env.arm"
    done = stamp(env, *harness)
    assert done.returncode == 0, done.stderr
    assert env.read_text().splitlines() == PRE_HARNESS_LINES


def test_a_named_harness_is_stamped(tmp_path: pathlib.Path):
    env = tmp_path / ".env.arm"
    done = stamp(env, "miniswe")
    assert done.returncode == 0, done.stderr
    assert env.read_text().splitlines() == [*PRE_HARNESS_LINES, "HPCAGENT_BENCH_RECORD_HARNESS=miniswe"]


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
    assert experiment_tags.display_name("harness-focus20") == "Agent Harness Comparison@20"


def test_a_migrated_run_records_no_harness_rather_than_inventing_one(tmp_path: pathlib.Path):
    """No arm name carries a harness, and the live rows of the same campaigns record NULL, so a
    migrated run and a live run of one arm have to stay one group."""
    dest = recording.connect(str(tmp_path / "out.db"))
    try:
        run_id = "cpf-llr-focus40-qwen38-c.n0.p0.w0"
        migrate.write_runs(dest, [run_id], lambda r: migrate.parse_arm(migrate.arm_of(r)))
        dest.commit()
        assert list(dest.execute("SELECT run_id, harness FROM runs")) == [(run_id, None)]
    finally:
        dest.close()
