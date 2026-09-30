# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One process, every OpenMP user the image ships, ONE runtime file mapped.

    python3 one_openmp_gate.py [--optional [MODULE ...]] [--load LIBRARY ...]

Imports numpy and scipy, runs a numba prange whose threads call BLAS, loads a ``gcc -fopenmp``
library, imports each ``--optional`` module that is installed (default: :data:`OPTIONAL_USERS`; and
runs a torch op when torch is one), then asserts a single OpenMP runtime realpath is mapped. Run by
containers/lib/one_openmp.sh at image build and by containers/images/verify_image.py in the finished
image; tests/test_one_openmp_runtime.py runs it with no optional module. ``openmp_runtimes.py`` beside
it does the counting.
"""

import argparse
import ctypes
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import openmp_runtimes

#: Exit status for "two runtimes mapped", apart from every other failure (a crash, a missing module).
CONFLICT_EXIT = 3

#: The gate counts runtimes, it does not load-test BLAS (numpy_on_openblas.sh does, at 2 x nproc callers): a
#: few threads keep it safe beside a wheel's 64-thread pthreads OpenBLAS on a large host.
GATE_THREADS = 4

#: Wheels and frameworks that may bundle or load an OpenMP runtime of their own; imported when installed.
OPTIONAL_USERS = ("torch", "sklearn", "xgboost", "lightgbm", "jax", "cupy", "tvm", "pythran")

C_SOURCE = "int p(void) {\n  int n = 0;\n#pragma omp parallel reduction(max : n)\n  n = 1;\n  return n;\n}\n"


def numba_prange_calling_blas() -> None:
    """numba's OpenMP pool and OpenBLAS in one process: the pair whose second runtime cost nproc^2 threads."""
    import numba
    import numpy as np
    import scipy.linalg

    @numba.njit(parallel=True)
    def prange_blas(a: np.ndarray, out: np.ndarray) -> None:
        for i in numba.prange(out.shape[0]):
            out[i] = np.dot(a, a)[0, 0]

    numba.set_num_threads(min(GATE_THREADS, numba.get_num_threads()))
    a = np.random.default_rng(0).random((64, 64))
    prange_blas(a, np.empty(2 * GATE_THREADS))
    scipy.linalg.lu_factor(a)


def load_gcc_openmp_library() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src, lib = os.path.join(tmp, "p.c"), os.path.join(tmp, "libp.so")
        pathlib.Path(src).write_text(C_SOURCE)
        subprocess.run([os.environ.get("CC", "gcc"), "-fopenmp", "-fPIC", "-shared", src, "-o", lib], check=True)
        assert ctypes.CDLL(lib).p() == 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optional", nargs="*", default=list(OPTIONAL_USERS), metavar="MODULE")
    parser.add_argument("--load", nargs="*", default=[], metavar="LIBRARY", help="extra shared libraries to dlopen")
    args = parser.parse_args()
    os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
    numba_prange_calling_blas()
    load_gcc_openmp_library()
    for library in args.load:
        ctypes.CDLL(library)
    present = openmp_runtimes.import_all([], args.optional)
    if "torch" in present:
        import torch

        torch.ones(1 << 22).add_(1).sum()
    runtimes = openmp_runtimes.mapped_runtimes()
    try:
        openmp_runtimes.assert_single_runtime(runtimes, f"numpy, scipy, numba prange, gcc -fopenmp, {present}")
    except openmp_runtimes.OpenMPRuntimeConflict as conflict:
        print(conflict, file=sys.stderr)
        return CONFLICT_EXIT
    print("one OpenMP runtime mapped:", runtimes, "after", present)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
