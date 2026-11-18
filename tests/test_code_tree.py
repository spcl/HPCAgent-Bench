# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""code_tree.sh: a job runs a frozen worktree of its staged commit, so a moved checkout neither stops a queued job nor
changes the code under a running one; trees no queued job runs are removed. Real git, a stub uv (its venv python runs
this interpreter), a stub squeue."""

import os
import pathlib
import subprocess
import sys

import pytest

from hpcagent_bench import paths
from tests.bash_stub import bash_stub

CLUSTER = paths.ROOT / "hpcagent_bench" / "cluster"
SEEDS_REL = "hidden"
GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def git(root: pathlib.Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), "-c", "protocol.file.allow=always", *args],
        env={**os.environ, **GIT_ENV},
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def checkout(tmp_path: pathlib.Path) -> pathlib.Path:
    """A live checkout with one submodule, whose origin is then deleted, and untracked secret seeds."""
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q")
    (origin / "module.txt").write_text("module\n")
    git(origin, "add", "-A")
    git(origin, "commit", "-q", "-m", "module")
    live = tmp_path / "live"
    (live / SEEDS_REL).mkdir(parents=True)
    git(live, "init", "-q")
    (live / "code.txt").write_text("v1\n")
    (live / SEEDS_REL / "README.md").write_text("seeds\n")
    (live / ".gitignore").write_text(f"{SEEDS_REL}/secret_seeds.json\n")
    git(live, "submodule", "-q", "add", str(origin), "third_party/module")
    git(live, "add", "-A")
    git(live, "commit", "-q", "-m", "v1")
    (live / SEEDS_REL / "secret_seeds.json").write_text('{"seed": 7}\n')
    subprocess.run(["rm", "-rf", str(origin)], check=True)
    return live


def run(tmp_path: pathlib.Path, script: str, **env: str) -> subprocess.CompletedProcess[str]:
    """``script`` after sourcing code_tree.sh, with a stub uv and squeue first on PATH."""
    stubs = tmp_path / "bin"
    bash_stub(
        stubs,
        "uv",
        f'echo "$PWD ${{VIRTUAL_ENV-unset}} ${{UV_LINK_MODE-hardlink}} $*" >> {tmp_path / "uv.calls"}\n'
        'mkdir -p "${UV_PROJECT_ENVIRONMENT}/bin"\n'
        f'printf \'#!/bin/sh\\nexec {sys.executable} "$@"\\n\' > "${{UV_PROJECT_ENVIRONMENT}}/bin/python"\n'
        'chmod +x "${UV_PROJECT_ENVIRONMENT}/bin/python"',
    )
    bash_stub(stubs, "squeue", 'printf "%s\\n" "${QUEUED_COMMANDS:-}"')
    return subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n. {CLUSTER / 'code_tree.sh'}\n{script}"],
        env={
            **os.environ,
            **GIT_ENV,
            "PATH": f"{stubs}:{os.environ['PATH']}",
            "HPCAGENT_BENCH_SCRATCH": str(tmp_path / "scratch"),
            "HPCAGENT_BENCH_HIDDEN_TESTS_REL": SEEDS_REL,
            "VIRTUAL_ENV": str(tmp_path / "live-venv"),
            "UV_PROJECT_ENVIRONMENT": str(tmp_path / "live-venv"),
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_job_runs_its_staged_commit_after_the_checkout_moved(tmp_path: pathlib.Path) -> None:
    """11-17: five queued jobs died at start because the shared checkout moved after staging. The tree of the staged
    commit keeps that commit's code, its submodule (from the checkout's own clone: the origin is gone) and the
    checkout's untracked seeds, and gets a venv of its own, never the caller's; a second job reuses it."""
    live = checkout(tmp_path)
    staged = git(live, "rev-parse", "HEAD")
    (live / "code.txt").write_text("v2\n")
    git(live, "commit", "-qam", "v2")
    done = run(tmp_path, f"freeze_code_tree {live} {staged[:9]}\nfreeze_code_tree {live} {staged}")
    assert done.returncode == 0, done.stderr
    first, second = done.stdout.split()
    tree = pathlib.Path(first)
    assert first == second == str(tmp_path / "scratch" / "code" / staged)
    assert git(tree, "rev-parse", "HEAD") == staged
    assert (tree / "code.txt").read_text() == "v1\n"
    assert (tree / "third_party" / "module" / "module.txt").read_text() == "module\n"
    assert (tree / SEEDS_REL / "secret_seeds.json").read_text() == '{"seed": 7}\n'
    assert (tree / ".venv" / "bin" / "python").exists()
    calls = (tmp_path / "uv.calls").read_text().splitlines()
    assert calls == [f"{tree} unset copy sync --frozen -q"], (
        "one venv per commit, in the tree, of copies, never the live one"
    )
    assert git(tree, "status", "--porcelain", "--untracked-files=no") == ""


def test_a_tree_whose_freeze_died_is_rebuilt(tmp_path: pathlib.Path) -> None:
    """A tree without its marker (a freeze killed half way) is not reused as if it were complete."""
    live = checkout(tmp_path)
    sha = git(live, "rev-parse", "HEAD")
    broken = tmp_path / "scratch" / "code" / sha
    broken.mkdir(parents=True)
    (broken / "half-written").write_text("")
    done = run(tmp_path, f"freeze_code_tree {live} HEAD")
    assert done.returncode == 0, done.stderr
    assert not (broken / "half-written").exists()
    assert (broken / "code.txt").read_text() == "v1\n"


@pytest.mark.parametrize(
    ("named", "grace", "kept"),
    [(True, "0", True), (False, "0", False), (False, "60", True)],
)
def test_gc_removes_only_trees_no_queued_job_runs(tmp_path: pathlib.Path, named: bool, grace: str, kept: bool) -> None:
    """The scratch inode quota holds a few trees: one no PENDING or RUNNING job runs (squeue's command) goes, once
    past its grace period (a submission that just froze it has not reached the queue yet)."""
    live = checkout(tmp_path)
    tree = pathlib.Path(run(tmp_path, f"freeze_code_tree {live} HEAD").stdout.strip())
    command = f"{tree}/hpcagent_bench/cluster/services.sbatch /x/.rendered/a.env" if named else "/other/job.sh"
    done = run(tmp_path, f"gc_code_trees {live}", QUEUED_COMMANDS=command, CODE_TREE_GRACE_MINUTES=grace)
    assert done.returncode == 0, done.stderr
    assert tree.exists() is kept
    assert (tree.name in git(live, "worktree", "list")) is kept


def test_gc_removes_nothing_when_the_queue_cannot_be_read(tmp_path: pathlib.Path) -> None:
    """Without squeue nothing says which trees are in use, so none is removed."""
    live = checkout(tmp_path)
    tree = pathlib.Path(run(tmp_path, f"freeze_code_tree {live} HEAD").stdout.strip())
    bash_stub(tmp_path / "bin", "squeue", "exit 1")
    done = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n. {CLUSTER / 'code_tree.sh'}\ngc_code_trees {live}"],
        env={
            **os.environ,
            "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
            "HPCAGENT_BENCH_SCRATCH": str(tmp_path / "scratch"),
            "CODE_TREE_GRACE_MINUTES": "0",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert tree.exists()


def test_submit_setup_job_submits_the_frozen_trees_services_sbatch(tmp_path: pathlib.Path) -> None:
    """The agent job runs the tree's services.sbatch with the tree as its checkout and the tree's venv as its host
    interpreter, while its env, problems file and data roots stay on the live checkout."""
    live = checkout(tmp_path)
    staged = git(live, "rev-parse", "--short=9", "HEAD")
    (live / "code.txt").write_text("v2\n")
    git(live, "commit", "-qam", "v2")
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (experiments / "problems.jsonl").write_text('{"kernel": "k"}\n')
    env = experiments / ".env.some-setup"
    env.write_text(
        "INFERENCE_NODES=2\nAGENT_NODES=1\nJUDGE_NODES=1\nPROBLEMS_FILE=problems.jsonl\n"
        f"HPCAGENT_BENCH_RECORD_COMMIT={staged}\n"
    )
    bash_stub(
        tmp_path / "bin",
        "sbatch",
        f'case "$*" in *services.sbatch*) printf "%s\\n" "$@" > {tmp_path / "sbatch.args"}; '
        f"env > {tmp_path / 'sbatch.env'} ;; esac\necho 4242",
    )
    (tmp_path / "site.env").write_text("")
    done = run(
        tmp_path,
        f". {CLUSTER / 'setup_nodes.sh'}\n. {CLUSTER / 'submit_common.sh'}\ncd {experiments}\n"
        f"submit_setup_job {env} some-setup 01:00:00",
        SUBMIT="1",
        PACK_AFTER="0",
        USER="tester",
        HPCAGENT_BENCH_REPO=str(live),
        HPCAGENT_BENCH_HOST_PYTHON=sys.executable,
        HPCAGENT_BENCH_SITE_ENV=str(tmp_path / "site.env"),
        SBATCH_ACCOUNT="project",
        HPCAGENT_BENCH_JOB_GPUS_PER_NODE="4",
    )
    assert done.returncode == 0, done.stdout + done.stderr
    tree = tmp_path / "scratch" / "code" / git(live, "rev-parse", f"{staged}^{{commit}}")
    args = (tmp_path / "sbatch.args").read_text().splitlines()
    assert args[-2] == f"{tree}/hpcagent_bench/cluster/services.sbatch"
    snapshot = pathlib.Path(args[-1])
    assert snapshot.parent == experiments / ".rendered"
    assert f"PROBLEMS_FILE={experiments / '.rendered'}/" in snapshot.read_text()
    exported = dict(line.split("=", 1) for line in (tmp_path / "sbatch.env").read_text().splitlines() if "=" in line)
    assert exported["HPCAGENT_BENCH_REPO"] == str(tree)
    assert exported["CLUSTER_SCRIPT_DIR"] == f"{tree}/hpcagent_bench/cluster"
    assert exported["HPCAGENT_BENCH_HOST_PYTHON"] == f"{tree}/.venv/bin/python"
    assert exported["HPCAGENT_BENCH_GENERATED_CACHE_HOST"] == f"{live}/.cache/generated"
    assert exported["HPCAGENT_BENCH_SCRATCH"] == str(tmp_path / "scratch")
    assert "VIRTUAL_ENV" not in exported
