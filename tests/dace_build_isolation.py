# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-xdist-worker DaCe build folder, shared by both test trees.

A uniquely named module rather than ``conftest.py``: each tree has its own ``conftest``, and
collecting both in one run makes ``from conftest import ...`` resolve to whichever was imported
first.
"""

import os
import pathlib
import sys


def pin_per_worker_dace_build_folder() -> None:
    """Give every xdist worker its own DaCe build folder.

    DaCe's build folder is ``<default_build_folder>/<sdfg.name>``, and an SDFG's name comes from
    the program it was parsed from -- so two workers compiling the SAME kernel name land in ONE
    directory, and the build is not written atomically. That is the same race
    :func:`hpcagent_bench.frameworks.dace_framework.pin_per_rank_build_dirs` splits for MPI ranks,
    with pytest-xdist as the launcher instead of mpirun, and it fails the same three ways: a
    ``FileExistsError``, a library-load error, or -- worst -- a worker loading the ``.so`` another
    worker is halfway through writing, which VALIDATES WRONG. Its signature in CI is a
    ``Fatal Python error: Segfault`` / ``Aborted`` with no failing assertion, and a block of
    consecutive ``F``s from one worker while the others stay green.

    The env var binds a dace imported later (dace folds ``DACE_*`` into its configuration when it
    loads), and a dace already imported gets the same folder through ``Config.set``, so no suite
    has to import dace to be protected. A pin the caller already made is EXTENDED rather than replaced, so pointing the build at a
    fast disk keeps working and still splits per worker.

    ``sdfg.build_folder`` set explicitly on an SDFG still wins over this, which is what the sparse
    oracle relies on -- it isolates per BUILD, which is stricter.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker is None:
        return  # a serial run has nothing to race with
    base = pathlib.Path(os.environ.get("DACE_default_build_folder", ".dacecache"))
    if base.name == worker:
        return
    os.environ["DACE_default_build_folder"] = str(base / worker)
    loaded = sys.modules.get("dace.config")
    if loaded is not None:
        loaded.Config.set("default_build_folder", value=str(base / worker))
