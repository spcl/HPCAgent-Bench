# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""scripts/run_tests.sh --container: the one entry point for running the suite inside the judge
EDF instead of the login/compute-node toolchain (see that script's own header for why -- the
cluster's gcc has no -std=c23 and about 720 translator cases fail there as a false regression).

Runs the real script against a stub `sbatch` on PATH that records its argv and never touches
Slurm, mirroring tests/test_submit_llrblind.py's pattern for the submit-*.sh scripts.
"""

import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
RUN_TESTS = REPO / "scripts" / "run_tests.sh"
CONTAINER_SBATCH = REPO / "scripts" / "run_tests_container.sbatch"


def stub(directory: pathlib.Path, name: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / name
    script.write_text(f"#!/usr/bin/env bash\n{body}\n")
    script.chmod(0o755)


def stub_sbatch(bin_dir: pathlib.Path, marker: pathlib.Path) -> None:
    stub(bin_dir, "sbatch", f'printf \'%s\\n\' "$@" > "{marker}"\nexit 0')


def stub_account(bin_dir: pathlib.Path) -> None:
    """A one-association sacctmgr: account_env.sh runs for real (scripts/cscs/account_env.sh),
    just against a fixed placeholder answer instead of this user's actual (ambiguous)
    associations -- a made-up name on purpose, so this fixture is never mistaken for a real one."""
    stub(bin_dir, "sacctmgr", "printf 'placeholder-acct\n'")


def run_container(
    tmp_path: pathlib.Path, *args: str, scratch: str | None = "/nonexistent-scratch"
) -> tuple[subprocess.CompletedProcess[str], pathlib.Path]:
    bin_dir = tmp_path / "bin"
    marker = tmp_path / "sbatch-argv.txt"
    stub_sbatch(bin_dir, marker)
    stub_account(bin_dir)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path / "home"), "USER": "tester"}
    if scratch is not None:
        env["SCRATCH"] = scratch
    proc = subprocess.run(
        ["bash", str(RUN_TESTS), "--container", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return proc, marker


def test_container_flag_submits_with_the_partition_and_sbatch_script(tmp_path: pathlib.Path) -> None:
    """The account is NOT a literal here -- account_env.sh hands it to sbatch through
    SBATCH_ACCOUNT, exercised for real against the stub sacctmgr above."""
    proc, marker = run_container(tmp_path)
    assert proc.returncode == 0, proc.stderr
    argv = marker.read_text().splitlines()
    assert "--wait" in argv
    assert "-A" not in argv
    assert not any(word.startswith("--partition") for word in argv), "the site layer's SBATCH_PARTITION picks it"
    assert str(CONTAINER_SBATCH) in argv


def test_container_flag_passes_pytest_selection_through_to_the_sbatch_script(tmp_path: pathlib.Path) -> None:
    proc, marker = run_container(tmp_path, "-k", "test_something", "tests/test_foo.py")
    assert proc.returncode == 0, proc.stderr
    argv = marker.read_text().splitlines()
    tail = argv[argv.index(str(CONTAINER_SBATCH)) + 1 :]
    assert tail == ["-k", "test_something", "tests/test_foo.py"]


def test_container_flag_without_scratch_fails_before_touching_sbatch(tmp_path: pathlib.Path) -> None:
    proc, marker = run_container(tmp_path, scratch=None)
    assert proc.returncode != 0
    assert "SCRATCH" in proc.stderr
    assert not marker.exists(), "sbatch was invoked despite SCRATCH being unset"


def test_run_tests_container_sbatch_has_the_single_node_directives() -> None:
    """The directives baked into the companion .sbatch file itself, so a direct `sbatch
    scripts/run_tests_container.sbatch` (bypassing the wrapper) still lands on one whole node; the
    partition comes from the site layer (SBATCH_PARTITION), never a directive."""
    text = CONTAINER_SBATCH.read_text()
    assert "#SBATCH --partition" not in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --mem=0" in text
