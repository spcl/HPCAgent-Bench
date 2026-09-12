# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``run_sparse_sweep`` used to swallow a failed validation: ``_run_sparse_one`` hardcoded
``ignore_errors=True`` into the per-kernel ``run_one`` call, so a validation failure never raised
inside the forked child, ``run_forked`` reported it as ok, and both the sweep and ``run-sparse``
exited 0 over a run in which a kernel failed validation."""

from collections.abc import Callable

import pytest

from hpcagent_bench.cli import main
from hpcagent_bench.frameworks.forked import RunResult
from hpcagent_bench.support.collect import sweep


def fake_run_forked_for(failing_benchname: str) -> Callable[..., RunResult[object]]:
    """A ``run_forked`` stand-in mirroring ``Test.run``'s real ``ignore_errors`` semantics for a
    validation failure on ``failing_benchname``: ok when the caller ignores errors (the swallowed
    bug this test pins), not ok when it does not (the fixed contract)."""

    def fake_run_forked(
        fn: object,
        benchname: str,
        framework_names: object,
        preset: object,
        validate: object,
        repeat: object,
        timeout: object,
        ignore_errors: bool,
        *rest: object,
        **kwargs: object,
    ) -> RunResult[object]:
        if benchname != failing_benchname:
            return RunResult(ok=True)
        return RunResult(ok=ignore_errors, error=None if ignore_errors else "ValueError: did not validate!")

    return fake_run_forked


@pytest.fixture
def two_sparse_benches(monkeypatch: pytest.MonkeyPatch) -> None:
    benches = [("spmv", {"csr": {}}), ("spgemm", {"csr": {}})]
    monkeypatch.setattr(sweep, "discover_sparse_benches", lambda filter_names=None: benches)


def test_run_sparse_sweep_returns_1_when_a_kernel_fails_validation(
    monkeypatch: pytest.MonkeyPatch, two_sparse_benches: None
) -> None:
    monkeypatch.setattr(sweep, "run_forked", fake_run_forked_for("spgemm"))
    rc = sweep.run_sparse_sweep("numpy", "S", True, 1, 1.0, None, None, None, True)
    assert rc == 1


def test_run_sparse_sweep_returns_0_when_nothing_fails(
    monkeypatch: pytest.MonkeyPatch, two_sparse_benches: None
) -> None:
    monkeypatch.setattr(sweep, "run_forked", fake_run_forked_for("no-such-kernel"))
    rc = sweep.run_sparse_sweep("numpy", "S", True, 1, 1.0, None, None, None, True)
    assert rc == 0


def test_cmd_run_sparse_exits_1_when_a_kernel_fails_validation(
    monkeypatch: pytest.MonkeyPatch, two_sparse_benches: None
) -> None:
    """The CLI propagates ``run_sparse_sweep``'s exit code directly (see test_cli_subcommands.py)."""
    monkeypatch.setattr(sweep, "run_forked", fake_run_forked_for("spgemm"))
    assert main(["run-sparse", "--ignore-errors"]) == 1
