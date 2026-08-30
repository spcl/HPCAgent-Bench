# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Two workers must not compile into one DaCe build folder.

CI runs both suites under ``pytest -n auto`` and DaCe's build folder is keyed by SDFG NAME, so two
workers compiling the same kernel share a directory that is not written atomically. The root
conftest splits it per worker; what is under test here is that the split happens and that DaCe
actually reads the channel it is written through.
"""
import os
import pathlib

import dace
import pytest

from conftest import pin_per_worker_dace_build_folder

BUILD_FOLDER_ENV = "DACE_default_build_folder"


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(BUILD_FOLDER_ENV, raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    return monkeypatch


def test_a_serial_run_is_left_alone(clean_env):
    """No worker, no race, and no directory the next serial run cannot reuse."""
    pin_per_worker_dace_build_folder()
    assert BUILD_FOLDER_ENV not in os.environ


def test_each_worker_gets_its_own_folder(clean_env):
    clean_env.setenv("PYTEST_XDIST_WORKER", "gw3")
    pin_per_worker_dace_build_folder()
    assert os.environ[BUILD_FOLDER_ENV] == str(pathlib.Path(".dacecache/gw3"))


def test_a_callers_own_pin_is_extended_not_replaced(clean_env):
    """Pointing the build at a fast disk has to keep working; it just splits underneath."""
    clean_env.setenv("PYTEST_XDIST_WORKER", "gw1")
    clean_env.setenv(BUILD_FOLDER_ENV, "/scratch/build")
    pin_per_worker_dace_build_folder()
    assert os.environ[BUILD_FOLDER_ENV] == str(pathlib.Path("/scratch/build/gw1"))


def test_the_split_does_not_nest_on_a_second_call(clean_env):
    """A second conftest load in the same worker would otherwise hand it a fresh empty cache."""
    clean_env.setenv("PYTEST_XDIST_WORKER", "gw2")
    pin_per_worker_dace_build_folder()
    pin_per_worker_dace_build_folder()
    assert os.environ[BUILD_FOLDER_ENV] == str(pathlib.Path(".dacecache/gw2"))


def test_dace_resolves_the_env_var_at_get_time(clean_env):
    """The claim the whole fix rests on: the pin binds however late dace was first imported.

    And its converse, which is why the pin is an env var and not ``Config.set``: an env override
    WINS over a later ``Config.set``, so nothing downstream can quietly put a worker back into the
    shared folder.
    """
    shipped = dace.Config.get("default_build_folder")
    clean_env.setenv(BUILD_FOLDER_ENV, "/scratch/pinned")
    assert dace.Config.get("default_build_folder") == "/scratch/pinned"
    try:
        dace.Config.set("default_build_folder", value="/scratch/elsewhere")
        assert dace.Config.get("default_build_folder") == "/scratch/pinned"
    finally:
        dace.Config.set("default_build_folder", value=shipped)
