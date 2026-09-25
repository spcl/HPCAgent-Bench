# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A launcher reads rosters, packets and CPF views from the tree it is run from, never another one.

A job runs the tree its launcher lives in (``beverin.sbatch`` derives ``HPCAGENT_BENCH_REPO`` from its
own location), so a launcher that reads the roster, the packet env or the CPF cache gate from a
different checkout builds an arm env against code the job never runs. Submitting from a pinned
worktree while the live checkout lagged behind it is exactly that case: the gate imported a module the
live tree did not have yet and refused every CPF arm.
"""

import pathlib
import re
import shutil
import subprocess

import pytest

EXPERIMENTS = pathlib.Path(__file__).resolve().parents[1] / "experiments"

#: The two lines every launcher resolves its tree with: the cd into its own directory, then OPT.
TREE_LINES = re.compile(r"^(cd -- .*|OPT=.*)$", re.MULTILINE)

LAUNCHERS = sorted(
    path.name for path in EXPERIMENTS.glob("*.sh") if re.search(r"^OPT=", path.read_text(), re.MULTILINE)
)


def resolved_opt(tmp_path: pathlib.Path, launcher: str, env: dict[str, str]) -> str:
    """OPT as ``launcher``'s own tree lines leave it, run from a copy of the launcher in a fresh tree."""
    lines = TREE_LINES.findall((EXPERIMENTS / launcher).read_text())
    probe = tmp_path / "repo" / "experiments" / launcher
    probe.parent.mkdir(parents=True)
    probe.write_text("\n".join(["set -eu", *lines, 'printf "%s" "${OPT}"']) + "\n")
    bash = shutil.which("bash")
    assert bash is not None
    done = subprocess.run([bash, str(probe)], env=env, capture_output=True, text=True, check=True, cwd=tmp_path)
    return done.stdout


def test_every_launcher_that_names_a_tree_is_checked() -> None:
    assert "submit-cpf-llr40.sh" in LAUNCHERS
    assert "submit-owed-wave.sh" in LAUNCHERS


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_a_launcher_reads_its_own_tree(tmp_path: pathlib.Path, launcher: str) -> None:
    """With OPT unset, OPT is the tree the launcher sits in, not a checkout under SCRATCH."""
    env = {"PATH": "/usr/bin:/bin", "SCRATCH": str(tmp_path / "scratch")}
    assert resolved_opt(tmp_path, launcher, env) == str(tmp_path / "repo")


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_an_exported_opt_still_wins(tmp_path: pathlib.Path, launcher: str) -> None:
    """A caller that exports OPT (a pinned worktree) keeps its explicit tree."""
    env = {"PATH": "/usr/bin:/bin", "SCRATCH": str(tmp_path / "scratch"), "OPT": "/pinned/tree"}
    assert resolved_opt(tmp_path, launcher, env) == "/pinned/tree"
