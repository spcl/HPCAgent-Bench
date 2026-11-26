# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Two workers must not compile into one DaCe build folder.

CI runs both suites under ``pytest -n auto`` and DaCe's build folder is keyed by SDFG NAME, so two
workers compiling the same kernel share a directory that is not written atomically.
:mod:`tests.dace_build_isolation`, called from tests/conftest.py, splits it per worker; what is under
test here is that the split happens and that DaCe actually reads the channel it is written
through. Imported from that module and NOT from ``conftest``: both test trees have a ``conftest``
and CI collects them together, so the bare name resolves to whichever was imported first -- a
collection error that killed the whole integration job 21 minutes in, before any test ran.
"""

import os
import pathlib

import dace
import pytest

from tests.dace_build_isolation import pin_per_worker_dace_build_folder

BUILD_FOLDER_ENV = "DACE_default_build_folder"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.delenv(BUILD_FOLDER_ENV, raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    ("worker", "pin", "calls", "folder"),
    [
        (None, None, 1, None),
        ("gw3", None, 1, ".dacecache/gw3"),
        ("gw1", "/scratch/build", 1, "/scratch/build/gw1"),
        ("gw2", None, 2, ".dacecache/gw2"),
    ],
    ids=[
        "serial_run_left_alone",
        "own_folder_per_worker",
        "callers_pin_extended_not_replaced",
        "no_nesting_on_a_second_call",
    ],
)
def test_each_worker_gets_its_own_folder(
    clean_env: pytest.MonkeyPatch, worker: str | None, pin: str | None, calls: int, folder: str | None
) -> None:
    """No worker, no race, and no directory the next serial run cannot reuse. Pointing the build at a fast disk
    has to keep working; it just splits underneath. A second conftest load in the same worker would otherwise
    hand it a fresh empty cache."""
    if worker is not None:
        clean_env.setenv("PYTEST_XDIST_WORKER", worker)
    if pin is not None:
        clean_env.setenv(BUILD_FOLDER_ENV, pin)
    for unused in range(calls):
        pin_per_worker_dace_build_folder()
    if folder is None:
        assert BUILD_FOLDER_ENV not in os.environ
    else:
        assert os.environ[BUILD_FOLDER_ENV] == str(pathlib.Path(folder))


def test_an_already_imported_dace_gets_the_pin_too(clean_env: pytest.MonkeyPatch) -> None:
    """dace reads ``DACE_*`` once, when its configuration loads; this module imported it before the
    pin, so the pin must reach the loaded configuration as well as the environment."""
    shipped = dace.Config.get("default_build_folder")
    clean_env.setenv("PYTEST_XDIST_WORKER", "gw5")
    clean_env.setenv(BUILD_FOLDER_ENV, "/scratch/pinned")
    try:
        pin_per_worker_dace_build_folder()
        assert dace.Config.get("default_build_folder") == str(pathlib.Path("/scratch/pinned/gw5"))
    finally:
        dace.Config.set("default_build_folder", value=shipped)
