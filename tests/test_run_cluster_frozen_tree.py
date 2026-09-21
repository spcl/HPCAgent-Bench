# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``experiments/run_cluster.sh``'s FROZEN TREE block: a batch step copies the checkout once and
re-executes from the copy, so a commit landing mid-run never reaches the job (643369: every /score
died on "cannot import name 'decline_kind'"). The block is lifted from the real file and run in a
throwaway checkout whose run_cluster.sh stops right after it."""

import pathlib
import shutil
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
TEXT = (REPO / "experiments" / "run_cluster.sh").read_text()
START = "# FROZEN TREE."
END = "    export HPCAGENT_BENCH_FROZEN=live\nfi\n"
BLOCK = TEXT[TEXT.index(START) : TEXT.index(END) + len(END)]
REPORT = (
    'echo "ran from ${SCRIPT_DIR} repo=${HPCAGENT_BENCH_REPO:-} marker=${HPCAGENT_BENCH_FROZEN:-}'
    ' packs=${PACK_ROOT:-} matrices=${HPCAGENT_BENCH_CACHE_DIR:-} generated=${HPCAGENT_BENCH_GENERATED_CACHE_HOST:-}"\n'
)


def checkout(live: pathlib.Path) -> pathlib.Path:
    script = live / "experiments" / "run_cluster.sh"
    script.parent.mkdir(parents=True)
    script.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nSCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
        + BLOCK
        + REPORT
    )
    (live / "hpcagent_bench").mkdir()
    (live / "hpcagent_bench" / "module.py").write_text("OLD = 1\n")
    (live / ".git").mkdir()
    (live / "core_nid0001_1").write_text("dump")
    return script


def launch(script: pathlib.Path, env: dict[str, str], path: str = "/usr/bin:/bin") -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(script)], env={"PATH": path, **env}, capture_output=True, text=True, check=True)


def fake_rsync(bin_dir: pathlib.Path, rc: int) -> str:
    """A PATH whose rsync does the real copy, then exits ``rc``."""
    bin_dir.mkdir()
    (bin_dir / "rsync").write_text(f'#!/bin/bash\n{shutil.which("rsync")} "$@"\nexit {rc}\n')
    (bin_dir / "rsync").chmod(0o755)
    return f"{bin_dir}:/usr/bin:/bin"


def test_a_batch_step_runs_from_a_copy_beside_its_campaign_that_later_edits_cannot_reach(
    tmp_path: pathlib.Path,
) -> None:
    live, runs = tmp_path / "live", tmp_path / "runs" / "campaign"
    out = launch(checkout(live), {"SLURM_JOB_ID": "123", "RUN_ROOT": str(runs)}).stdout
    frozen = tmp_path / "runs" / ".frozen" / "job-123"
    assert out.count("frozen tree") == 1, out
    assert f"ran from {frozen}/experiments repo={frozen} marker={frozen}" in out, out
    assert f"packs={live}/.cache/packs matrices={live}/hpcagent_bench/.hpcagent_bench_cache" in out, out
    assert f"generated={live}/.cache/generated" in out, out
    (live / "hpcagent_bench" / "module.py").write_text("NEW = 1\n")
    assert (frozen / "hpcagent_bench" / "module.py").read_text() == "OLD = 1\n"
    assert not (frozen / ".git").exists() and not (frozen / "core_nid0001_1").exists()
    assert not runs.exists(), "nothing lands under RUN_ROOT, where extraction globs for databases"


def test_a_step_that_inherits_the_frozen_tree_does_not_copy_again(tmp_path: pathlib.Path) -> None:
    live, runs = tmp_path / "live", tmp_path / "runs" / "campaign"
    env = {"SLURM_JOB_ID": "123", "RUN_ROOT": str(runs), "HPCAGENT_BENCH_FROZEN": "/elsewhere"}
    out = launch(checkout(live), env).stdout
    assert "frozen tree" not in out and f"ran from {live}/experiments" in out, out
    assert not (tmp_path / "runs").exists()


def test_a_failed_copy_runs_the_job_on_the_live_tree_instead_of_killing_it(tmp_path: pathlib.Path) -> None:
    live, runs = tmp_path / "live", tmp_path / "runs" / "campaign"
    result = launch(checkout(live), {"SLURM_JOB_ID": "123", "RUN_ROOT": str(runs)}, fake_rsync(tmp_path / "bin", 23))
    assert "WARNING: could not freeze" in result.stderr, result.stderr
    assert f"ran from {live}/experiments repo= marker=live" in result.stdout, result.stdout


def test_a_file_vanishing_mid_copy_still_freezes(tmp_path: pathlib.Path) -> None:
    live, runs = tmp_path / "live", tmp_path / "runs" / "campaign"
    out = launch(checkout(live), {"SLURM_JOB_ID": "123", "RUN_ROOT": str(runs)}, fake_rsync(tmp_path / "bin", 24))
    assert f"marker={tmp_path}/runs/.frozen/job-123" in out.stdout, out.stdout


def test_a_run_root_inside_the_checkout_is_not_copied_into_itself(tmp_path: pathlib.Path) -> None:
    live = tmp_path / "live"
    result = launch(checkout(live), {"SLURM_JOB_ID": "123", "RUN_ROOT": str(live / "runs" / "campaign")})
    assert "(inside)" in result.stderr and "marker=live" in result.stdout, result
    assert not (live / "runs").exists()
