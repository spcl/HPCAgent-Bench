# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Drop this process's core-dump limit -- the python-side half of the ``ulimit -c 0`` rule.

``scripts/check_core_dumps.py`` puts ``ulimit -c 0`` in every shell entry point this repo owns,
and anything those launch inherits it. Nothing reaches a script a human writes by hand outside the
checkout, and that is where 21.6 GB of ``core_beverin-ln001_*`` came from on 2026-09-20: a CPF
repro run started ``python3 diag.py`` from an ad-hoc login-node script, dace segfaulted twice, and
each dump was the interpreter's whole address space (17.5 GB and 4 GB).

Importing :mod:`hpcagent_bench` is the one thing every such process does, so the floor is set
there. It is one ``setrlimit`` call and it changes no behaviour a caller can observe. Unlike the
shell's ``ulimit -c 0``, which sets both limits, this touches the SOFT limit only and leaves the
hard one alone, so ``HPCAGENT_BENCH_CORE_DUMPS=1`` really does hand the dump back to someone
debugging. Measured 2026-09-20 from a shell at ``(unlimited, unlimited)``: the limit reads
``(0, unlimited)`` after the import, and the sampled canonicalize that segfaults leaves no file.
"""

import os
import resource

#: Set to ``1`` to keep core dumps: a debugger session that wants the dump, in a directory that
#: can hold it. Anything else (unset included) means the limit is dropped.
ALLOW = "HPCAGENT_BENCH_CORE_DUMPS"


def disable() -> None:
    """Set RLIMIT_CORE's soft limit to 0 unless ``HPCAGENT_BENCH_CORE_DUMPS=1`` says otherwise.

    Never raises: a platform without ``RLIMIT_CORE``, or a sandbox that refuses the call, must not
    turn ``import hpcagent_bench`` into an error over a hardening measure.
    """
    if os.environ.get(ALLOW) == "1":
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
        if soft != 0:
            resource.setrlimit(resource.RLIMIT_CORE, (0, hard))
    except (OSError, ValueError):
        pass
