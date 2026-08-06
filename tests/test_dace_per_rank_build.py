# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every rank of a multi-rank job builds into its OWN folder and its own PCH cache.

Ranks compile different SDFGs into the same ``.dacecache`` and the same precompiled-header cache,
and the build is not atomic. Racing there does not merely fail: a rank can load the ``.so`` another
rank is halfway through writing, so the run VALIDATES WRONG. That is why this is pinned by a test
rather than left to whoever notices the flake.
"""
import getpass
import pathlib

import pytest

from hpcagent_bench.frameworks import dace_framework

#: The launcher variables, each of which alone must be enough to detect a rank.
LAUNCHERS = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "MV2_COMM_WORLD_RANK")


@pytest.fixture(autouse=True)
def no_inherited_rank(monkeypatch):
    """The test process may itself have been launched by mpirun; start from a clean slate."""
    for name in LAUNCHERS + ("DACE_BUILD_CACHE_DIR", ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_every_launcher_s_rank_variable_is_read(monkeypatch, launcher) -> None:
    """Open MPI, MPICH/PMI, Slurm and MVAPICH each publish the rank under a different name; missing
    one silently returns the whole job to a single shared build folder."""
    monkeypatch.setenv(launcher, "3")
    assert dace_framework.mpi_rank() == "3"


def test_a_single_process_run_is_not_a_rank() -> None:
    """With no launcher variable set there is nothing to race, and DaCe's own defaults are kept --
    suffixing anyway would give every plain run a fresh cache and rebuild the world each time."""
    assert dace_framework.mpi_rank() is None


def test_a_rank_gets_its_own_build_folder(monkeypatch) -> None:
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
    captured = {}
    monkeypatch.setattr(dace_framework.dace.Config, "get", lambda *k: ".dacecache")
    monkeypatch.setattr(dace_framework.dace.Config, "set", lambda *k, value: captured.update({k: value}))
    dace_framework.pin_per_rank_build_dirs()
    assert captured[("default_build_folder", )] == str(pathlib.Path(".dacecache/rank2"))


def test_a_rank_gets_its_own_precompiled_header_cache(monkeypatch) -> None:
    """The PCH is ~110 MB per entry and keyed by compiler+flags, so two ranks share one entry unless
    the ROOT differs -- and that entry is written, not just read."""
    monkeypatch.setenv("SLURM_PROCID", "5")
    monkeypatch.setattr(dace_framework.dace.Config, "get", lambda *k: ".dacecache")
    monkeypatch.setattr(dace_framework.dace.Config, "set", lambda *k, value: None)
    dace_framework.pin_per_rank_build_dirs()
    root = pathlib.Path(dace_framework.os.environ["DACE_BUILD_CACHE_DIR"])
    assert root.name == "rank5"
    assert getpass.getuser() in str(root) or ".cache/dace" in str(root)


def test_an_explicit_cache_dir_is_left_alone(monkeypatch, tmp_path) -> None:
    """A job that already partitioned the cache itself (or points it at node-local scratch) must not
    have that decision silently re-taken."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    monkeypatch.setenv("DACE_BUILD_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(dace_framework.dace.Config, "get", lambda *k: ".dacecache")
    monkeypatch.setattr(dace_framework.dace.Config, "set", lambda *k, value: None)
    dace_framework.pin_per_rank_build_dirs()
    assert dace_framework.os.environ["DACE_BUILD_CACHE_DIR"] == str(tmp_path)


def test_pinning_twice_does_not_nest_the_rank_folder(monkeypatch) -> None:
    """optimize() runs per kernel; appending each time would give kernel 2 ``rank1/rank1``, a fresh
    empty cache that rebuilds everything and never hits."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    folder = {"value": ".dacecache"}
    monkeypatch.setattr(dace_framework.dace.Config, "get", lambda *k: folder["value"])
    monkeypatch.setattr(dace_framework.dace.Config, "set", lambda *k, value: folder.update({"value": value}))
    dace_framework.pin_per_rank_build_dirs()
    dace_framework.pin_per_rank_build_dirs()
    assert folder["value"] == str(pathlib.Path(".dacecache/rank1"))
