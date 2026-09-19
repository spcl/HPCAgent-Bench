# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""submit_common.sh's KERNELS_FILE isolation, shared by every submit-*.sh (2026-09-19 bug).

A subset/owed submission (KERNELS_FILE set) used to write straight into the arm's canonical
.env.<arm> and problems-<arm>.jsonl -- the exact files a PENDING job of the FULL-roster arm reads
when it starts. A dry run (SUBMIT=0) still writes those files up to the sbatch call, so even a
preview could silently rewrite a queued job's kernel list. Traced live to jobs 642644/642645
(experiments/owed/... reruns) and, earlier, a harness-focus20 problems file clobbered to 11 kernels.

Two things now guard this, both in submit_common.sh so every family submitter shares them:
  * arm_file_suffix/kernels_file_suffix -- a subset submission's env/problems names diverge from
    the canonical ones, so it can never overwrite them (test_submit_llrblind.py,
    test_submit_git_scicomp.py, test_submit_scicomp_perf_playbook.py exercise this per launcher).
  * refuse_if_queue_references -- exercised here directly with a stub squeue/sacct standing in for
    a real PENDING job, since a real cluster queue cannot be manufactured in a unit test.
"""

import pathlib
import shutil
import subprocess

BASH = shutil.which("bash")
assert BASH is not None

REPO = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"


def stub_account(directory: pathlib.Path) -> None:
    """A one-association sacctmgr: submit_common.sh sources account_env.sh even for a pure
    file-naming check, and this user's REAL associations are ambiguous (it refuses ambiguity,
    exit 1) -- unrelated to what these tests actually check, so a fixed single answer replaces it."""
    stub(directory, "sacctmgr", "printf 'a-g34\n'")


def stub(directory: pathlib.Path, name: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)


def run_probe(
    tmp_path: pathlib.Path, script: str, queue_body: str, sacct_body: str
) -> subprocess.CompletedProcess[str]:
    """Source submit_common.sh and run ``script`` against a stub squeue/sacct."""
    stub(tmp_path / "bin", "squeue", queue_body)
    stub(tmp_path / "bin", "sacct", sacct_body)
    stub_account(tmp_path / "bin")
    probe = tmp_path / "probe.sh"
    probe.write_text(f"set -eu\n. {EXPERIMENTS / 'submit_common.sh'}\n{script}\n")
    return subprocess.run(
        [BASH, str(probe)],
        env={"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "USER": "tester"},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


#: One PENDING job (id 999999) whose CLUSTER_ENV_FILE the tests below point at a candidate path.
QUEUE_ONE_JOB = "echo 999999"


def sacct_pointing_at(env_path: str) -> str:
    return f"printf '999999|sbatch --parsable --nodes=1 --export=ALL,CLUSTER_ENV_FILE={env_path} beverin.sbatch\\n'"


def test_a_kernels_file_subset_env_diverges_from_the_canonical_name(tmp_path: pathlib.Path) -> None:
    """arm_file_suffix at KERNELS_FILE unset is empty (canonical); set, it is never empty -- the
    whole mechanism a subset submission relies on to avoid the canonical filename."""
    stub_account(tmp_path / "bin")
    path = f"{tmp_path / 'bin'}:/usr/bin:/bin"
    result = subprocess.run(
        [BASH, "-c", f'. {EXPERIMENTS / "submit_common.sh"}; echo "[$(arm_file_suffix)]"'],
        env={"PATH": path},
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert result.stdout.strip() == "[]"

    result = subprocess.run(
        [BASH, "-c", f'. {EXPERIMENTS / "submit_common.sh"}; echo "[$(arm_file_suffix)]"'],
        env={"PATH": path, "KERNELS_FILE": "owed/arm-budget.txt"},
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert result.stdout.strip() == "[-arm-budget]"


def test_dry_run_refuses_to_overwrite_a_file_a_pending_job_reads(tmp_path: pathlib.Path) -> None:
    """The exact 642644/642645-class bug: a candidate env path matches a PENDING job's
    CLUSTER_ENV_FILE (from sacct's SubmitLine, mocked here) -- refused, nothing written, before
    SUBMIT is ever consulted (this check runs regardless of SUBMIT=0 or 1)."""
    target = tmp_path / ".env.cpf-llr-focus40-kimi27sglang-c-clean"
    target.write_text("PENDING-JOBS-OWN-CONTENT\n")
    result = run_probe(
        tmp_path,
        f'refuse_if_queue_references "{target}"; echo "refuse_if_queue_references returned $?"',
        QUEUE_ONE_JOB,
        sacct_pointing_at(str(target)),
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "job 999999 is PENDING/RUNNING and reads it as CLUSTER_ENV_FILE" in result.stderr
    # the function only REFUSES; it never touches the file itself
    assert target.read_text() == "PENDING-JOBS-OWN-CONTENT\n"


def test_a_problems_file_referenced_through_a_pending_jobs_own_env_is_also_refused(tmp_path: pathlib.Path) -> None:
    """PROBLEMS_FILE inside the referenced env is a bare name relative to ITS OWN directory --
    resolved against that, not matched by basename alone, so this only fires for the SAME file the
    queued job will actually read."""
    queued_env = tmp_path / ".env.other-arm"
    queued_env.write_text("PROBLEMS_FILE=problems-other-arm-owed.jsonl\n")
    problems = tmp_path / "problems-other-arm-owed.jsonl"
    result = run_probe(
        tmp_path,
        f'refuse_if_queue_references "{tmp_path}/.env.unrelated" "{problems}"',
        QUEUE_ONE_JOB,
        sacct_pointing_at(str(queued_env)),
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "reads it via" in result.stderr


def test_a_same_named_problems_file_in_a_different_directory_does_not_collide(tmp_path: pathlib.Path) -> None:
    """The false positive this fix must NOT introduce: a test's own tmp tree (or any other
    directory) can name a file identically to a real production problems file without the check
    firing -- only the resolved, same-directory path collides."""
    real_dir = tmp_path / "real-experiments"
    real_dir.mkdir()
    queued_env = real_dir / ".env.some-other-real-arm"
    queued_env.write_text("PROBLEMS_FILE=problems-git-scicomp-owed.jsonl\n")

    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    unrelated_problems = sandbox_dir / "problems-git-scicomp-owed.jsonl"
    result = run_probe(
        tmp_path,
        f'refuse_if_queue_references "{tmp_path}/.env.unrelated" "{unrelated_problems}"',
        QUEUE_ONE_JOB,
        sacct_pointing_at(str(queued_env)),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_an_empty_queue_passes_through(tmp_path: pathlib.Path) -> None:
    """No PENDING/RUNNING job at all (a stub squeue that prints nothing) must never block a
    submission -- the common case, exercised by every other submit-*.sh test in this suite."""
    target = tmp_path / ".env.some-arm"
    result = run_probe(tmp_path, f'refuse_if_queue_references "{target}"', "true", "true")
    assert result.returncode == 0, result.stdout + result.stderr
