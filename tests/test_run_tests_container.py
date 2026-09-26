# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""scripts/run_tests.sh --container submits scripts/ci_mi200.sbatch (the suite or the CI replay
inside the judge image). Runs the real script against a stub `sbatch` that records its argv."""

import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
RUN_TESTS = REPO / "scripts" / "run_tests.sh"
CONTAINER_SBATCH = REPO / "scripts" / "ci_mi200.sbatch"


def stub(directory: pathlib.Path, name: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / name
    script.write_text(f"#!/usr/bin/env bash\n{body}\n")
    script.chmod(0o755)


def stub_sbatch(bin_dir: pathlib.Path, marker: pathlib.Path) -> None:
    stub(bin_dir, "sbatch", f'printf \'%s\\n\' "$@" > "{marker}"\nexit 0')


def run_container(
    tmp_path: pathlib.Path, *args: str, scratch: str | None = "/nonexistent-scratch"
) -> tuple[subprocess.CompletedProcess[str], pathlib.Path]:
    bin_dir = tmp_path / "bin"
    marker = tmp_path / "sbatch-argv.txt"
    stub_sbatch(bin_dir, marker)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "USER": "tester",
        "HPCAGENT_BENCH_CI_PARTITION": "ci-part",
    }
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
    """The account is not an argument: sbatch reads SBATCH_ACCOUNT from the site layer."""
    proc, marker = run_container(tmp_path)
    assert proc.returncode == 0, proc.stderr
    argv = marker.read_text().splitlines()
    assert "--wait" in argv
    assert "-A" not in argv
    assert "--partition=ci-part" in argv, "the site layer's HPCAGENT_BENCH_CI_PARTITION picks it"
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


def test_ci_sbatch_has_the_single_node_directives() -> None:
    """A direct `sbatch scripts/ci_mi200.sbatch` (bypassing the wrapper) lands on one whole node and
    never requeues; the partition comes from the command line or SBATCH_PARTITION, never a
    directive (SBATCH_PARTITION would silently override one)."""
    text = CONTAINER_SBATCH.read_text()
    assert "#SBATCH --partition" not in text
    assert "#SBATCH --no-requeue" in text
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --mem=0" in text
