# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The gate of the CI steps that must not skip: an allowed reason passes, any other skip fails the run."""

import os
import pathlib
import subprocess
import sys

import pytest

from hpcagent_bench import paths

ALLOWED = 'import pytest\n\n\ndef test_gated() -> None:\n    pytest.skip("no CUDA device / cupy")\n'
PASSING = "def test_passing() -> None:\n    pass\n"


def gate(folder: pathlib.Path) -> subprocess.CompletedProcess[str]:
    """Run from the checkout, where ``-m pytest`` imports the plugin; the job's ``--cov`` stays out."""
    env = {**os.environ, "PYTEST_ADDOPTS": ""}
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "tests.no_silent_skips",
        str(folder),
    ]
    return subprocess.run(command, cwd=paths.ROOT, env=env, capture_output=True, text=True, timeout=120)


def test_a_skip_for_an_allowed_reason_passes(tmp_path: pathlib.Path) -> None:
    (tmp_path / "test_gated.py").write_text(ALLOWED)
    run = gate(tmp_path)
    assert run.returncode == 0, run.stdout


@pytest.mark.parametrize(
    "source",
    [
        'import pytest\n\n\ndef test_gated() -> None:\n    pytest.skip("mpirun is not on PATH")\n',
        'import pytest\n\npytest.importorskip("a_module_that_is_not_installed")\n',
    ],
    ids=["test", "collection"],
)
def test_any_other_skip_fails_a_run_that_otherwise_passes(source: str, tmp_path: pathlib.Path) -> None:
    (tmp_path / "test_gated.py").write_text(source)
    (tmp_path / "test_passing.py").write_text(PASSING)
    run = gate(tmp_path)
    assert run.returncode == pytest.ExitCode.TESTS_FAILED, run.stdout
    assert "skips this step does not allow" in run.stdout
