# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""tools/run_tests.sh --container: the one entry point for running the suite inside the judge
EDF instead of the login/compute-node toolchain (see that script's own header for why -- the
cluster's gcc has no -std=c23 and about 720 translator cases fail there as a false regression).

Runs the real script against a stub `sbatch` on PATH that records its argv and never touches
Slurm, mirroring tests/test_submit_llrblind.py's pattern for the submit-*.sh scripts.
"""

import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
RUN_TESTS = REPO / "tools" / "run_tests.sh"
CONTAINER_SBATCH = REPO / "tools" / "run_tests_container.sbatch"


def stub_sbatch(bin_dir: pathlib.Path, marker: pathlib.Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "sbatch"
    script.write_text(f'#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "{marker}"\nexit 0\n')
    script.chmod(0o755)


def run_container(
    tmp_path: pathlib.Path, *args: str, scratch: str | None = "/nonexistent-scratch"
) -> tuple[subprocess.CompletedProcess[str], pathlib.Path]:
    bin_dir = tmp_path / "bin"
    marker = tmp_path / "sbatch-argv.txt"
    stub_sbatch(bin_dir, marker)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path / "home")}
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


def test_container_flag_submits_with_the_account_partition_and_sbatch_script(tmp_path: pathlib.Path) -> None:
    proc, marker = run_container(tmp_path)
    assert proc.returncode == 0, proc.stderr
    argv = marker.read_text().splitlines()
    assert "--wait" in argv
    assert argv[argv.index("-A") + 1] == "a-g34"
    assert "--partition=mi300" in argv
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


def test_run_tests_container_sbatch_has_the_mi300_single_node_directives() -> None:
    """The directives baked into the companion .sbatch file itself, so a direct `sbatch
    tools/run_tests_container.sbatch` (bypassing the wrapper) still lands on one mi300 node."""
    text = CONTAINER_SBATCH.read_text()
    assert "#SBATCH --partition=mi300" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --mem=0" in text
