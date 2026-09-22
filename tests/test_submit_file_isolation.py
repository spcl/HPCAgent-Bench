# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""submit_common.sh's KERNELS_FILE isolation, shared by every submit-*.sh (2026-09-19 bug).

A subset/owed submission (KERNELS_FILE set) used to write straight into the arm's canonical
.env.<arm> and problems-<arm>.jsonl -- the exact files a PENDING job of the FULL-roster arm reads
when it starts. A dry run (SUBMIT=0) still writes those files up to the sbatch call, so even a
preview could silently rewrite a queued job's kernel list. Traced live to jobs 642644/642645
(experiments/owed/... reruns) and, earlier, a harness-focus20 problems file clobbered to 11 kernels.

Three things now guard this, all in submit_common.sh so every family submitter shares them:
  * arm_file_suffix/kernels_file_suffix -- a subset submission's env/problems names diverge from
    the canonical ones, so it can never overwrite them (test_submit_llrblind.py,
    test_submit_git_scicomp.py, test_submit_scicomp_perf_playbook.py exercise this per launcher).
  * refuse_if_queue_references -- exercised here directly with a stub squeue/sacct standing in for
    a real PENDING job, since a real cluster queue cannot be manufactured in a unit test.
  * refuse_unfiltered_snapshot_problems (finalize_staged_env) -- a snapshot env's own PROBLEMS_FILE
    must diverge from the canonical "problems-<arm>.jsonl" too: naming the FILE apart (the bullet
    above) does not stop its CONTENTS from being the unfiltered full roster. Job 642734
    (2026-09-19) was hand-submitted as a "-cpfleak0919" snapshot that named itself apart but still
    pointed at the canonical file, silently re-running all 6 kernels a sibling job already owed.

Also covers TOKEN_SCALE/TIME_SCALE (2026-09-19): an owed rerun's tokens and wall clock used to
scale together under one BUDGET_SCALE, but a model whose base AGENT_TIMEOUT_SECONDS is already
large enough (Kimi's 8h) hits the partition's own MaxTime before a further 4x tokens does anything
useful -- these two knobs default to BUDGET_SCALE (every prior caller unchanged) and let a rerun
ask for more tokens without asking sbatch for a --time no partition will grant.
"""

import pathlib
import shutil
import subprocess

import pytest

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


def run_finalize_probe(tmp_path: pathlib.Path, staged_body: str, env_name: str) -> subprocess.CompletedProcess[str]:
    """Source submit_common.sh and call finalize_staged_env on a hand-written staged file."""
    staged = tmp_path / "staged.env.staging"
    staged.write_text(staged_body)
    env = tmp_path / env_name
    probe = tmp_path / "probe.sh"
    probe.write_text(
        "set -eu\n"
        f". {EXPERIMENTS / 'submit_common.sh'}\n"
        f'finalize_staged_env "{staged}" "{env}"\n'
        'echo "finalize_staged_env returned $?"\n'
    )
    result = subprocess.run(
        [BASH, str(probe)],
        env={"PATH": "/usr/bin:/bin", "USER": "tester"},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    result.staged_path = staged  # type: ignore[attr-defined]
    result.env_path = env  # type: ignore[attr-defined]
    return result


def test_a_suffixed_snapshot_pointed_at_the_canonical_problems_file_is_refused(tmp_path: pathlib.Path) -> None:
    """The exact 642734 bug (2026-09-19): a snapshot env named beyond its arm's own ".env.<arm>"
    (arm_file_suffix's mark of a subset/budget submission) whose PROBLEMS_FILE is nonetheless the
    bare canonical "problems-<arm>.jsonl" -- the full-roster file a PENDING job of the UNSUFFIXED
    arm reads at start. This silently re-ran all 6 kernels a sibling job already owed instead of
    the 3 it was submitted to exclude."""
    result = run_finalize_probe(
        tmp_path,
        "CAMPAIGN_ARM=cpf-llr-focus40-kimi27sglang-c-clean\n"
        "PROBLEMS_FILE=problems-cpf-llr-focus40-kimi27sglang-c-clean.jsonl\n",
        ".env.cpf-llr-focus40-kimi27sglang-c-clean-cpfleak0919",
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "is the canonical full-roster file" in result.stderr
    # bailed gate: neither the staged file nor the final env is left behind looking complete
    assert not result.staged_path.exists()  # type: ignore[attr-defined]
    assert not result.env_path.exists()  # type: ignore[attr-defined]


def test_a_suffixed_snapshot_pointed_at_its_own_filtered_problems_file_passes(tmp_path: pathlib.Path) -> None:
    """The non-bug: a subset submission's PROBLEMS_FILE carries the SAME suffix as its env (what
    submit-cpf-llr40.sh's arm_file_suffix actually produces) -- never refused."""
    result = run_finalize_probe(
        tmp_path,
        "CAMPAIGN_ARM=cpf-llr-focus40-kimi27sglang-c-clean\n"
        "PROBLEMS_FILE=problems-cpf-llr-focus40-kimi27sglang-c-clean-cpfleak0919.jsonl\n",
        ".env.cpf-llr-focus40-kimi27sglang-c-clean-cpfleak0919",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.env_path.read_text() == (  # type: ignore[attr-defined]
        "CAMPAIGN_ARM=cpf-llr-focus40-kimi27sglang-c-clean\n"
        "PROBLEMS_FILE=problems-cpf-llr-focus40-kimi27sglang-c-clean-cpfleak0919.jsonl\n"
    )


def test_the_canonical_unsuffixed_env_is_never_checked(tmp_path: pathlib.Path) -> None:
    """A full-roster arm's own ".env.<arm>" legitimately points at "problems-<arm>.jsonl" -- the
    check must only fire for a SNAPSHOT (a suffixed filename), never for the canonical arm itself."""
    result = run_finalize_probe(
        tmp_path,
        "CAMPAIGN_ARM=cpf-llr-focus40-kimi27sglang-c-clean\n"
        "PROBLEMS_FILE=problems-cpf-llr-focus40-kimi27sglang-c-clean.jsonl\n",
        ".env.cpf-llr-focus40-kimi27sglang-c-clean",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def run_common(tmp_path: pathlib.Path, script: str, extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Source submit_common.sh with the given knobs and run ``script`` against it."""
    stub_account(tmp_path / "bin")
    env = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "USER": "tester", **extra_env}
    return subprocess.run(
        [BASH, "-c", f". {EXPERIMENTS / 'submit_common.sh'}\n{script}"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_token_scale_and_time_scale_default_to_budget_scale(tmp_path: pathlib.Path) -> None:
    """Every owed rerun before 2026-09-19 set only BUDGET_SCALE and expected it to double both
    fields -- TOKEN_SCALE/TIME_SCALE must fall back to it so that caller is unchanged."""
    result = run_common(tmp_path, 'echo "[$TOKEN_SCALE][$TIME_SCALE]"', {"BUDGET_SCALE": "3"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[3][3]"


def test_scale_tokens_is_never_capped(tmp_path: pathlib.Path) -> None:
    """A token ceiling costs money, not a --time no partition will grant -- unlike scale_time,
    scale_tokens has no cap to hit."""
    result = run_common(tmp_path, "scale_tokens 12000000", {"TOKEN_SCALE": "4"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "48000000"


def test_scale_time_is_uncapped_under_the_partition_limit(tmp_path: pathlib.Path) -> None:
    """qwen38/oss120b's own base AGENT_TIMEOUT_SECONDS (14400, 4h) times 4 is 16h -- under the
    20h cap (23h partition limit minus 3h staging), so the plain scale applies untouched."""
    result = run_common(
        tmp_path, "scale_time 14400", {"TIME_SCALE": "4", "STAGING_HOURS": "3", "PARTITION_TIME_LIMIT_HOURS": "23"}
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "57600"  # 16h


def test_scale_time_clamps_to_the_partition_limit_minus_staging(tmp_path: pathlib.Path) -> None:
    """The 642640-class rerun this rule replaced: Kimi's own base AGENT_TIMEOUT_SECONDS (28800,
    8h) times 4 is 32h, which no sbatch --time the mi300 partition (MaxTime 24h) will ever grant --
    clamped to (23 - 3)h = 20h instead of asking for an allocation that leaves the job PENDING
    forever."""
    result = run_common(
        tmp_path, "scale_time 28800", {"TIME_SCALE": "4", "STAGING_HOURS": "3", "PARTITION_TIME_LIMIT_HOURS": "23"}
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "72000"  # 20h, not 115200 (32h)


def test_scaled_budget_from_reports_the_same_capped_value_it_writes(tmp_path: pathlib.Path) -> None:
    """The provenance RECORD_AGENT_TIMEOUT_SECONDS/RECORD_AGENT_MAX_TOKENS lines a submit-*.sh
    appends are exactly scaled_budget_from's own return value -- if that value were not already
    the capped one, a rerun's rows would record a budget the job never actually ran under."""
    base = tmp_path / ".env.base-kimi27sglang"
    base.write_text("AGENT_TIMEOUT_SECONDS=28800\nAGENT_MAX_TOKENS=12000000\n")
    result = run_common(
        tmp_path,
        f'echo "[$(scaled_budget_from "{base}" AGENT_TIMEOUT_SECONDS)]'
        f'[$(scaled_budget_from "{base}" AGENT_MAX_TOKENS)]"',
        {"TOKEN_SCALE": "4", "TIME_SCALE": "4", "STAGING_HOURS": "3", "PARTITION_TIME_LIMIT_HOURS": "23"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[72000][48000000]"  # 20h capped, 48M uncapped


@pytest.mark.parametrize(
    ("token_scale", "time_scale", "want"),
    [
        ("1", "1", ""),
        ("4", "4", "-budget4x"),
        ("2", "2", "-budget2x"),
        ("4", "2", "-tok4x-time2x"),
    ],
)
def test_budget_env_suffix_names_both_scales_when_they_diverge(
    tmp_path: pathlib.Path, token_scale: str, time_scale: str, want: str
) -> None:
    """Equal scales keep the pre-2026-09-19 "-budget<N>x" spelling byte-identical for every prior
    caller; TOKEN_SCALE and TIME_SCALE set independently (the Kimi 4x-tokens/capped-time rerun) get
    a suffix that names both numbers instead of collapsing them into one that is neither."""
    result = run_common(
        tmp_path, 'echo "[$(budget_env_suffix)]"', {"TOKEN_SCALE": token_scale, "TIME_SCALE": time_scale}
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"[{want}]"


def run_submit_arm_job_probe(tmp_path: pathlib.Path, extra_env: dict[str, str]) -> tuple[str, str]:
    """Source arm_nodes.sh + submit_common.sh, call submit_arm_job against a stub sbatch that
    records its own argv, and return (sbatch's captured argv, submit_arm_job's own stdout)."""
    env_file = tmp_path / ".env.some-arm"
    env_file.write_text("INFERENCE_NODES=2\nAGENT_NODES=1\nJUDGE_NODES=1\n")
    captured = tmp_path / "sbatch_argv.txt"
    stub(
        tmp_path / "bin",
        "sbatch",
        f'printf "%s\\n" "$@" > "{captured}"\nprintf "999000\\n"\n',
    )
    stub_account(tmp_path / "bin")
    probe = tmp_path / "probe.sh"
    probe.write_text(
        "set -eu\n"
        f". {EXPERIMENTS / 'arm_nodes.sh'}\n"
        f". {EXPERIMENTS / 'submit_common.sh'}\n"
        f'cd "{tmp_path}"\n'
        f'submit_arm_job "{env_file}" some-arm 01:00:00\n'
    )
    env = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "USER": "tester", "SUBMIT": "1", **extra_env}
    result = subprocess.run(
        [BASH, str(probe)], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return captured.read_text(), result.stdout


def test_hold_1_asks_sbatch_for_hold(tmp_path: pathlib.Path) -> None:
    """The 643115/643117 race (2026-09-19): a follow-up `scontrol hold` after sbatch returns can
    lose to the scheduler if a node is free that instant -- both started RUNNING before the hold
    call reached them, on a live checkout the anti-cheat merge gate had not yet cleared. HOLD=1
    passes --hold to the SAME sbatch call that submits the job, so there is no gap to race."""
    argv, stdout = run_submit_arm_job_probe(tmp_path, {"HOLD": "1"})
    assert "--hold" in argv.splitlines()
    assert "HELD" in stdout


def test_hold_unset_does_not_ask_sbatch_for_hold(tmp_path: pathlib.Path) -> None:
    """The default: every submit-*.sh call before 2026-09-19 never held, and must not start
    holding just because the knob now exists."""
    argv, stdout = run_submit_arm_job_probe(tmp_path, {})
    assert "--hold" not in argv.splitlines()
    assert "HELD" not in stdout
