# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Make a fork safe to take while an OpenMP thread pool is live.

The forking itself belongs to :func:`hpcagent_bench.frameworks.forked.run_forked`, which
already marshals results, timeouts and fatal signals; this module supplies only the one
thing it was missing.
"""

import ctypes
import os
import warnings

#: OpenMP runtimes whose thread pool must be torn down before a fork (see
#: :func:`pause_openmp_pools`). Probed by the sonames a linked node library actually records
#: in DT_NEEDED.
OMP_RUNTIME_SONAMES = ("libgomp.so.1", "libomp.so.5", "libomp.so", "libiomp5.so", "libnvomp.so")

#: ``omp_pause_resource_t`` (OpenMP 5.0). Both tear the pool down (what buys fork safety);
#: ``hard`` also frees threadprivate data, so ``soft`` is the default.
OMP_PAUSE_SOFT = 1
OMP_PAUSE_HARD = 2

#: name -> ``omp_pause_resource_t`` value, for a config/CLI knob.
OMP_PAUSE_MODES = {"soft": OMP_PAUSE_SOFT, "hard": OMP_PAUSE_HARD}

#: Exported by LLVM's libomp and Intel's libiomp5, never by libgomp. That family returns 1 from
#: omp_pause_resource_all when the runtime was never initialised or is already paused, and it
#: installs a pthread_atfork handler, so its refusal leaves no live pool to hang a child on.
KMP_PAUSE_SYMBOL = "__kmpc_pause_resource"


def exports(lib: ctypes.CDLL, symbol: str) -> bool:
    try:
        lib[symbol]
    except AttributeError:
        return False
    return True


def pause_openmp_pools(mode: int = OMP_PAUSE_SOFT) -> None:
    """Tear down the thread pool of every OpenMP runtime ALREADY loaded here, so the coming
    fork is safe.

    ``fork()`` duplicates only the calling thread, so a child entering a parallel region with
    the parent's pool live hangs forever; libgomp installs no ``pthread_atfork`` handler to
    recover (libomp does). ``RTLD_NOLOAD``: only pause a runtime already mapped -- plain
    ``CDLL`` would LOAD one this process never needed. Best effort but never silent: a missing
    symbol, or a refusal from a runtime outside the LLVM family, warns, since an unhardened fork
    really does deadlock.
    """
    for soname in OMP_RUNTIME_SONAMES:
        try:
            lib = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)
        except OSError:
            continue  # not loaded in this process: nothing to pause
        try:
            pause = lib.omp_pause_resource_all
        except AttributeError:
            warnings.warn(
                f"{soname}: no omp_pause_resource_all (pre-OpenMP-5.0 runtime); its thread pool "
                f"was NOT torn down before the fork -- fork safety for this runtime now rests on "
                f"its own pthread_atfork handler, if it installs one (libgomp installs none)."
            )
            continue  # best effort; the warning tells the caller the fork was left unhardened
        pause.argtypes = [ctypes.c_int]
        pause.restype = ctypes.c_int
        # libgomp refuses only inside a parallel region, where the pool is live and a child hangs.
        if pause(mode) != 0 and not exports(lib, KMP_PAUSE_SYMBOL):
            warnings.warn(
                f"{soname}: omp_pause_resource_all(mode={mode}) returned non-zero; its thread "
                f"pool was NOT torn down before the fork."
            )
