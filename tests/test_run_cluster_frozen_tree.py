# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``experiments/run_cluster.sh``'s FROZEN TREE block: a batch step copies the checkout once and
re-executes from the copy, so a commit landing mid-run never reaches the job (643369: every /score
died on "cannot import name 'decline_kind'"). The block is lifted from the real file and run in a
throwaway checkout whose run_cluster.sh stops right after it."""

import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "experiments" / "run_cluster.sh").read_text()
START = "# FROZEN TREE."
END = 'exec bash "${HPCAGENT_BENCH_FROZEN}/experiments/run_cluster.sh" "$@"\nfi\n'
BLOCK = TEXT[TEXT.index(START) : TEXT.index(END) + len(END)]


def launch(live: pathlib.Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    script = live / "experiments" / "run_cluster.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nSCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
        + BLOCK
        + 'echo "ran from ${SCRIPT_DIR} repo=${HPCAGENT_BENCH_REPO:-}"\n'
    )
    (live / "hpcagent_bench").mkdir(exist_ok=True)
    (live / "hpcagent_bench" / "module.py").write_text("OLD = 1\n")
    (live / ".git").mkdir(exist_ok=True)
    return subprocess.run(
        ["bash", str(script)], env={"PATH": "/usr/bin:/bin", **env}, capture_output=True, text=True, check=True
    )


def test_a_batch_step_runs_from_a_copy_later_edits_cannot_reach(tmp_path: pathlib.Path) -> None:
    live, runs = tmp_path / "live", tmp_path / "runs"
    live.mkdir()
    out = launch(live, {"SLURM_JOB_ID": "123", "RUN_ROOT": str(runs)}).stdout
    frozen = runs / ".src" / "123"
    assert out.count("frozen tree") == 1, out
    assert f"ran from {frozen}/experiments repo={frozen}" in out, out
    (live / "hpcagent_bench" / "module.py").write_text("NEW = 1\n")
    assert (frozen / "hpcagent_bench" / "module.py").read_text() == "OLD = 1\n"
    assert not (frozen / ".git").exists()


def test_a_step_that_inherits_the_frozen_tree_does_not_copy_again(tmp_path: pathlib.Path) -> None:
    live, runs = tmp_path / "live", tmp_path / "runs"
    live.mkdir()
    env = {"SLURM_JOB_ID": "123", "RUN_ROOT": str(runs), "HPCAGENT_BENCH_FROZEN": str(runs / ".src" / "123")}
    out = launch(live, env).stdout
    assert "frozen tree" not in out, out
    assert not (runs / ".src").exists()
