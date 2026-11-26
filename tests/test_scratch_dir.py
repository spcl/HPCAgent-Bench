# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``$HPCAGENT_BENCH_SCRATCH``: where logs, core dumps and native-mode submissions go, ``<repo>/.scratch`` by default.

``$SCRATCH`` is the site's bulk storage (run roots, caches) and a different thing; this one belongs to the checkout.
"""

import importlib
import pathlib
import subprocess

import pytest

from hpcagent_bench import paths
from hpcagent_bench.harness import native

REPO = paths.ROOT


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, "{root}/.scratch"),
        ({"HPCAGENT_BENCH_REPO": "{tmp}"}, "{tmp}/.scratch"),  # a repo the caller resolved
        ({paths.SCRATCH_ENV: "{tmp}/elsewhere"}, "{tmp}/elsewhere"),  # the variable names another root
        # ``$SCRATCH`` names run roots and caches; a native run's submissions stay with the checkout
        ({"SCRATCH": "{tmp}"}, "{root}/.scratch"),
    ],
    ids=["default", "caller_repo", "variable", "site_scratch"],
)
def test_the_default_is_the_checkouts_dot_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, env: dict[str, str], expected: str
) -> None:
    monkeypatch.delenv(paths.SCRATCH_ENV, raising=False)
    monkeypatch.delenv("HPCAGENT_BENCH_REPO", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value.format(tmp=tmp_path))
    assert paths.scratch_dir() == pathlib.Path(expected.format(root=paths.ROOT, tmp=tmp_path))


def test_native_submissions_go_under_the_scratch_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Read once, at import: a process picks its scratch directory when it starts."""
    monkeypatch.setenv(paths.SCRATCH_ENV, str(tmp_path))
    try:
        reloaded = importlib.reload(native)
        assert tmp_path / "native_runs" == reloaded.NATIVE_RUNS
        assert reloaded.run_dir("r1", "gemm") == tmp_path / "native_runs" / "r1" / "gemm"
    finally:
        monkeypatch.undo()
        importlib.reload(native)


def test_the_prompt_names_a_repo_relative_folder_or_the_variable_never_a_host_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    assert native.display_run_dir("gemm") == ".scratch/native_runs/<episode_id>/gemm"
    monkeypatch.setattr(native, "NATIVE_RUNS", tmp_path / "far" / "native_runs")
    shown = native.display_run_dir("gemm")
    assert shown == "$HPCAGENT_BENCH_SCRATCH/native_runs/<episode_id>/gemm"
    assert str(tmp_path) not in shown


def test_the_shell_environment_exports_the_same_default() -> None:
    done = subprocess.run(
        ["bash", "-c", f'. "{REPO}/hpcagent_bench/cluster/env.sh" && printf %s "${{HPCAGENT_BENCH_SCRATCH}}"'],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin", "SCRATCH": "/nonexistent-site-scratch"},
    )
    assert done.stdout == str(REPO / ".scratch")
