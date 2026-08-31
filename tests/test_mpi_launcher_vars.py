# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The framework sweep must not let DaCe bootstrap MPI in a forked child.

DaCe calls ``ensure_mpi_initialized()`` when it is imported and a launcher variable is set. Slurm's
pmix plugin sets one (``PMIX_RANK``) for every step, MPI or not, so a per-kernel child of the sweep
imported DaCe, called ``MPI_Init``, and hung -- the whole GPU track produced zero rows for a day.
The variables are stripped before the first framework import; these tests pin that, and pin the two
things about it that are easy to break by tidying.
"""
import os

import pytest

from hpcagent_bench.support.collect.sweep import MPI_LAUNCHER_VARS, drop_mpi_launcher_vars


@pytest.fixture(name="launcher_env")
def launcher_env_fixture(monkeypatch):
    """A process that looks like a rank of a pmix-launched step, plus the shard variable."""
    for var in MPI_LAUNCHER_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PMIX_RANK", "2")
    monkeypatch.setenv("PMI_RANK", "2")
    monkeypatch.setenv("SLURM_PROCID", "2")
    return None


def test_the_launcher_variables_are_removed(launcher_env):
    removed = drop_mpi_launcher_vars()
    assert set(removed) == {"PMIX_RANK", "PMI_RANK"}
    assert not [var for var in MPI_LAUNCHER_VARS if var in os.environ]


def test_slurm_procid_survives(launcher_env):
    """The sweep shards on SLURM_PROCID and names per-rank build folders with it.

    DaCe excludes it from its own MPI trigger for the same reason -- it says a launcher started the
    process, not that the process is an MPI rank -- so stripping it would break sharding while
    fixing nothing.
    """
    drop_mpi_launcher_vars()
    assert os.environ["SLURM_PROCID"] == "2"


def test_it_is_idempotent_and_quiet_when_nothing_is_set(monkeypatch):
    for var in MPI_LAUNCHER_VARS:
        monkeypatch.delenv(var, raising=False)
    assert drop_mpi_launcher_vars() == []
    assert drop_mpi_launcher_vars() == []


def test_the_list_still_covers_what_dace_triggers_on():
    """DaCe owns the trigger list; ours is a hardcoded copy and must not fall behind it.

    The copy is deliberate -- reading it from DaCe would import DaCe, which is the import the strip
    exists to make safe -- so the only guard against drift is this comparison, made HERE where
    importing DaCe is already unavoidable.
    """
    from dace.sdfg.sdfg import MPI_RANK_VARS

    missing = sorted(set(MPI_RANK_VARS) - set(MPI_LAUNCHER_VARS))
    assert not missing, (f"DaCe initializes MPI on {missing}, which the sweep no longer strips; "
                         f"add them to MPI_LAUNCHER_VARS")


def test_the_distributed_residency_keeps_its_launcher_variables(launcher_env, monkeypatch):
    """An MPI rank must NOT be stripped: MPI is what the launcher already prepared for it.

    The two residencies fail in opposite directions -- a single-node child that lets MPI come up
    deadlocks, and an MPI rank that does not aborts on its first collective with no traceback -- so
    the switch is stated by the caller and defaults to the one that fails safe.
    """
    import hpcagent_bench.support.collect.sweep as sweep

    called = []
    monkeypatch.setattr(sweep, "drop_mpi_launcher_vars", lambda: called.append(True) or [])
    monkeypatch.setattr(sweep, "generate_framework", lambda *a, **k: None)
    monkeypatch.setattr(sweep, "Benchmark", lambda *a, **k: None)
    monkeypatch.setattr(sweep, "Test", lambda *a, **k: type("T", (), {"run": lambda *_, **__: {}})())

    sweep.run_one("k", [], "S", False, 1, 1.0, True, False, False, None, distributed=True)
    assert called == [], "an MPI rank had its launcher variables stripped"

    sweep.run_one("k", [], "S", False, 1, 1.0, True, False, False, None)
    assert called == [True], "the single-node default did not strip"
