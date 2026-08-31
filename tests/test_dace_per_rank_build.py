# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every rank of a multi-rank job builds into its OWN folder and its own PCH cache.

Ranks compile different SDFGs into the same ``.dacecache`` and the same precompiled-header cache,
and the build is not atomic. Racing there does not merely fail: a rank can load the ``.so`` another
rank is halfway through writing, so the run VALIDATES WRONG. That is why this is pinned by a test
rather than left to whoever notices the flake.

DaCe grew the build-folder half of this itself (``cache_distaware``, spcl/dace#2466), so both
branches are covered here: with the knob present our suffix must NOT be added on top, without it
our suffix is the only thing there is. The PCH cache is not partitioned by #2466 and stays ours in
both. The two are told apart only by whether the fake config knows the key -- the one input the
probe reads.
"""
import getpass
import pathlib

import pytest

from hpcagent_bench.frameworks import dace_framework
from tests.optional_imports import import_or_skip

#: The launcher variables, each of which alone must be enough to detect a rank.
LAUNCHERS = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "MV2_COMM_WORLD_RANK")


def fake_config(monkeypatch, *, native: bool, folder: str = ".dacecache") -> dict[str, object]:
    """Install a fake ``dace.Config`` whose schema has ``cache_distaware`` only when ``native``.

    Returns the backing dict, so a test reads what was set rather than what was asked for."""
    values: dict[str, object] = {"default_build_folder": folder}
    if native:
        values["cache_distaware"] = True

    def get(*key: str) -> object:
        return values[".".join(key)]  # a missing key raises KeyError, exactly as DaCe's Config does

    def set_(*key: str, value: object) -> None:
        values[".".join(key)] = value

    monkeypatch.setattr(dace_framework.dace.Config, "get", get)
    monkeypatch.setattr(dace_framework.dace.Config, "set", set_)
    return values


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
    values = fake_config(monkeypatch, native=False)
    dace_framework.pin_per_rank_build_dirs()
    assert values["default_build_folder"] == str(pathlib.Path(".dacecache/rank2"))


def test_dace_s_own_per_rank_folders_are_not_suffixed_again(monkeypatch) -> None:
    """``cache_distaware`` already appends the rank inside DaCe. Appending ours on top nests a
    second level (``.dacecache/rank2_rank2``), which hits no cache and re-splits an already split
    one -- hiding, not fixing, a real collision."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
    values = fake_config(monkeypatch, native=True)
    dace_framework.pin_per_rank_build_dirs()
    assert values["default_build_folder"] == ".dacecache"
    assert values["cache_distaware"] is True


def test_dace_s_own_per_rank_folders_are_pinned_on(monkeypatch) -> None:
    """DaCe suffixes the folder only while the knob is on, so a ~/.dace.conf that turned it off
    would put every rank of a graded run back into one shared folder."""
    monkeypatch.setenv("SLURM_PROCID", "4")
    values = fake_config(monkeypatch, native=True)
    values["cache_distaware"] = False
    dace_framework.pin_per_rank_build_dirs()
    assert values["cache_distaware"] is True
    assert values["default_build_folder"] == ".dacecache"


@pytest.mark.parametrize("native", (False, True))
def test_a_rank_gets_its_own_precompiled_header_cache(monkeypatch, native) -> None:
    """The PCH is ~110 MB per entry and keyed by compiler+flags, so two ranks share one entry unless
    the ROOT differs -- and that entry is written, not just read. #2466 partitions the build folder
    only; ``build_cache.cache_root`` is still one root per node, so this stays ours either way."""
    monkeypatch.setenv("SLURM_PROCID", "5")
    fake_config(monkeypatch, native=native)
    dace_framework.pin_per_rank_build_dirs()
    root = pathlib.Path(dace_framework.os.environ["DACE_BUILD_CACHE_DIR"])
    assert root.name == "rank5"
    assert getpass.getuser() in str(root) or ".cache/dace" in str(root)


@pytest.mark.parametrize("native", (False, True))
def test_an_explicit_cache_dir_is_left_alone(monkeypatch, tmp_path, native) -> None:
    """A job that already partitioned the cache itself (or points it at node-local scratch) must not
    have that decision silently re-taken."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    monkeypatch.setenv("DACE_BUILD_CACHE_DIR", str(tmp_path))
    fake_config(monkeypatch, native=native)
    dace_framework.pin_per_rank_build_dirs()
    assert dace_framework.os.environ["DACE_BUILD_CACHE_DIR"] == str(tmp_path)


@pytest.mark.parametrize("native", (False, True))
def test_pinning_twice_does_not_nest_the_rank_folder(monkeypatch, native) -> None:
    """optimize() runs per kernel; appending each time would give kernel 2 ``rank1/rank1``, a fresh
    empty cache that rebuilds everything and never hits."""
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    values = fake_config(monkeypatch, native=native)
    dace_framework.pin_per_rank_build_dirs()
    dace_framework.pin_per_rank_build_dirs()
    expected = ".dacecache" if native else str(pathlib.Path(".dacecache/rank1"))
    assert values["default_build_folder"] == expected


def test_the_native_probe_reads_the_config_key_not_a_version(monkeypatch) -> None:
    """The knob landed on a branch, so no released version number tells the two DaCes apart. A
    DaCe whose schema lacks the key must fall back to our suffix, whatever it calls itself."""
    monkeypatch.setenv("PMI_RANK", "7")
    monkeypatch.setattr(dace_framework.importlib.metadata, "version", lambda name: "99.0.0")
    values = fake_config(monkeypatch, native=False)
    dace_framework.pin_per_rank_build_dirs()
    assert values["default_build_folder"] == str(pathlib.Path(".dacecache/rank7"))


def test_rank_env_covers_every_launcher_dace_knows() -> None:
    """DaCe splits the build folder on any of ITS names; a name only DaCe knows leaves mpi_rank()
    None, so the PCH cache stays shared across ranks while the build folder splits -- the exact
    half-partitioned state that produced the original library-load races."""
    dace_sdfg = import_or_skip("dace.sdfg.sdfg")
    missing = sorted(set(dace_sdfg.LAUNCHER_RANK_VARS) - set(dace_framework.RANK_ENV))
    assert not missing, f"DaCe learned launcher variables we do not probe: {missing}"
