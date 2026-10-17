# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""MAX_NODES in hpcagent_bench/cluster/submit_common.sh: plan_waves splits the setups of one submit call into waves
whose summed node count is at most MAX_NODES, and submit_waves chains every wave afterany on all jobs of the previous
one. A stub ``sbatch`` numbers its jobs and records each agent job's name and dependency; nothing reaches Slurm."""

import pathlib
import re
import shutil
import subprocess
import sys

import pytest

from hpcagent_bench import paths
from tests.bash_stub import bash_stub
from tests.test_submit import submit, tree

BASH = shutil.which("bash")
assert BASH is not None

REPO = paths.ROOT
CLUSTER_DIR = REPO / "hpcagent_bench" / "cluster"

#: The first job id the chain stub prints; every later call prints the next integer.
FIRST_JOB_ID = 101

#: A stub sbatch: prints job ids FIRST_JOB_ID, FIRST_JOB_ID + 1, ... as ``<id>;cluster`` (sbatch --parsable on a
#: federation), and appends ``<id> <job-name> <dependency or ->`` per agent job to ``${STUB_MARKERS}/sbatch.chain``.
CHAIN_SBATCH = f"""last=$(cat "${{STUB_MARKERS}}/sbatch.last" 2>/dev/null || echo {FIRST_JOB_ID - 1})
id=$(( last + 1 ))
echo "${{id}}" > "${{STUB_MARKERS}}/sbatch.last"
name="" dependency="-"
for arg; do case "${{arg}}" in --job-name=*) name="${{arg#*=}}" ;; --dependency=*) dependency="${{arg#*=}}" ;; esac; done
case "$*" in *services.sbatch*) echo "${{id}} ${{name}} ${{dependency}}" >> "${{STUB_MARKERS}}/sbatch.chain" ;; esac
echo "${{id}};cluster"
"""


def run_bash(tmp_path: pathlib.Path, script: str, **env: str) -> subprocess.CompletedProcess[str]:
    """``script`` after sourcing setup_nodes.sh and submit_common.sh, run in ``tmp_path`` with ``tmp_path/bin``
    first on PATH."""
    probe = tmp_path / "probe.sh"
    probe.write_text(f"set -eu\n. {CLUSTER_DIR / 'setup_nodes.sh'}\n. {CLUSTER_DIR / 'submit_common.sh'}\n{script}\n")
    site = tmp_path / "site.env"
    site.write_text("")
    return subprocess.run(
        [BASH, str(probe)],
        env={
            "PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin",
            "USER": "tester",
            "STUB_MARKERS": str(tmp_path),
            "HPCAGENT_BENCH_HOST_PYTHON": sys.executable,
            "HPCAGENT_BENCH_SITE_ENV": str(site),
            "SBATCH_ACCOUNT": "project",
            "HPCAGENT_BENCH_JOB_GPUS_PER_NODE": "4",
            **env,
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def plan(tmp_path: pathlib.Path, max_nodes: str, *counts: int) -> subprocess.CompletedProcess[str]:
    """plan_waves over ``counts`` under ``max_nodes``; stdout is WAVE_NUMBERS, space-separated."""
    words = " ".join(str(count) for count in counts)
    return run_bash(tmp_path, f'plan_waves "{max_nodes}" {words}\necho "${{WAVE_NUMBERS[*]}}"')


def test_plan_waves_opens_a_wave_when_the_next_setup_would_pass_the_limit(tmp_path: pathlib.Path) -> None:
    done = plan(tmp_path, "32", 12, 12, 8, 20, 4, 32)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["1", "1", "1", "2", "2", "3"]


def test_plan_waves_without_a_limit_is_one_wave(tmp_path: pathlib.Path) -> None:
    done = plan(tmp_path, "", 12, 40, 8)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["1", "1", "1"]


def test_plan_waves_refuses_a_setup_larger_than_the_limit(tmp_path: pathlib.Path) -> None:
    done = plan(tmp_path, "8", 4, 9)
    assert done.returncode == 2
    assert "a setup of 9 nodes exceeds MAX_NODES=8" in done.stderr


@pytest.mark.parametrize("max_nodes", ["0", "-4", "eight", "08"])
def test_plan_waves_refuses_a_limit_that_is_not_a_positive_integer(tmp_path: pathlib.Path, max_nodes: str) -> None:
    done = plan(tmp_path, max_nodes, 4)
    assert done.returncode == 2
    assert f"MAX_NODES is a positive integer, got '{max_nodes}'" in done.stderr


def submit_three_setups(tmp_path: pathlib.Path, **env: str) -> subprocess.CompletedProcess[str]:
    """submit_waves over three 4-node setups ``s1 s2 s3`` under MAX_NODES=8 after job 77, with the chain stub."""
    bash_stub(tmp_path / "bin", "sbatch", CHAIN_SBATCH)
    for name in ("s1", "s2", "s3"):
        (tmp_path / f".env.{name}").write_text("INFERENCE_NODES=2\nAGENT_NODES=1\nJUDGE_NODES=1\n")
    return run_bash(
        tmp_path,
        "WAVE_ENVS=(.env.s1 .env.s2 .env.s3) WAVE_SETUPS=(s1 s2 s3) WAVE_WALLTIMES=(01:00:00 01:00:00 01:00:00)\n"
        'submit_waves 8 77 ""',
        **env,
    )


def test_each_wave_depends_on_every_job_of_the_previous_wave(tmp_path: pathlib.Path) -> None:
    done = submit_three_setups(tmp_path, SUBMIT="1")
    assert done.returncode == 0, done.stderr
    chain = (tmp_path / "sbatch.chain").read_text().splitlines()
    assert chain == ["101 s1 afterany:77", "102 s2 afterany:77", "103 s3 afterany:101:102"]


def test_a_dry_run_prints_the_waves_and_submits_nothing(tmp_path: pathlib.Path) -> None:
    done = submit_three_setups(tmp_path)
    assert done.returncode == 0, done.stderr
    headers = [line for line in done.stdout.splitlines() if line.startswith("wave ")]
    assert headers == ["wave 1/2: 8 nodes (MAX_NODES=8), after 77", "wave 2/2: 4 nodes (MAX_NODES=8), after wave 1"]
    assert not (tmp_path / "sbatch.chain").exists()


@pytest.mark.smoke
def test_submit_sh_splits_a_study_into_chained_waves(tmp_path: pathlib.Path) -> None:
    """submit.sh with MAX_NODES at two setups' nodes: four setups go out as two waves of two, the second wave
    afterany on both jobs of the first."""
    root = tree(tmp_path)
    bash_stub(root / "bin", "sbatch", CHAIN_SBATCH)
    knobs = {"KERNELS_FILE": "subset.txt", "LANGUAGES": "c hip", "PACKETS": "none lang-skills"}
    dry = submit(root, **knobs)
    assert dry.returncode == 0, dry.stderr
    nodes = {int(count) for count in re.findall(r"^prepared \S+ \((\d+) nodes", dry.stdout, re.MULTILINE)}
    assert len(nodes) == 1, dry.stdout
    max_nodes = str(2 * nodes.pop())

    done = submit(root, "--account", "project", **knobs, MAX_NODES=max_nodes, DEPEND_ON="77", SUBMIT="1")
    assert done.returncode == 0, done.stderr
    headers = [line for line in done.stdout.splitlines() if line.startswith("wave ")]
    assert headers == [
        f"wave 1/2: {max_nodes} nodes (MAX_NODES={max_nodes}), after 77",
        f"wave 2/2: {max_nodes} nodes (MAX_NODES={max_nodes}), after wave 1 (101:102)",
    ]
    chain = [line.split() for line in (root / "sbatch.chain").read_text().splitlines()]
    assert [(job, dependency) for job, _, dependency in chain] == [
        ("101", "afterany:77"),
        ("102", "afterany:77"),
        ("103", "afterany:101:102"),
        ("104", "afterany:101:102"),
    ]
