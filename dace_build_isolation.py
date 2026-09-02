# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-xdist-worker DaCe build folder, in a module BOTH test trees can name.

Lives at the repo root rather than under ``hpcagent_bench`` because the translator suite
deliberately imports nothing from that package, and rather than inside ``conftest.py`` because
each tree has a ``conftest`` of its own: collecting ``tests/`` and
``hpcagent_bench/numpy_translators/tests/`` in one run makes a bare ``from conftest import ...``
resolve to whichever one was imported first, which is a COLLECTION error that kills the whole job
before a single test runs. A uniquely named module has no such ambiguity.
"""

import os
import pathlib


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

    An ENV VAR rather than ``dace.Config.set``: dace resolves ``DACE_*`` at every ``get``, so this
    binds however late dace is first imported, and no suite has to import dace to be protected.
    A pin the caller already made is EXTENDED rather than replaced, so pointing the build at a
    fast disk keeps working and still splits per worker.

    ``sdfg.build_folder`` set explicitly on an SDFG still wins over this, which is what the sparse
    oracle relies on -- it isolates per BUILD, which is stricter.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker is None:
        return  # a serial run has nothing to race with
    base = pathlib.Path(os.environ.get("DACE_default_build_folder", ".dacecache"))
    if base.name != worker:
        os.environ["DACE_default_build_folder"] = str(base / worker)
