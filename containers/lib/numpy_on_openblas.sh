#!/bin/sh
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Rebuild numpy and scipy, at the versions already installed, against the image's OpenBLAS.
#
#   numpy_on_openblas.sh <view prefix>      (e.g. /opt/view)
#
# The PyPI wheels bundle their own scipy-openblas: a pthreads build with MAX_THREADS=64 that crashed
# when numba's prange threads (192 on an mi300 node) called np.linalg at once, and a second BLAS
# runtime beside the image's OpenMP one. Built from source against <view>'s openblas (threads=openmp,
# blas_gate.sh), numpy and scipy share one BLAS and one OpenMP runtime with the C baselines, DaCe and
# numba (NUMBA_THREADING_LAYER=omp, set by the Dockerfile). The versions must not move: numpy
# computes every CPU reference. The gate at the end runs 2 x nproc numba prange iterations that each
# call np.dot and scipy.linalg.lu_factor concurrently.
set -eux
ulimit -c 0
view="$1"
py="$(command -v python3)"
numpy_v="$("${py}" -c 'import numpy; print(numpy.__version__)')"
scipy_v="$("${py}" -c 'import scipy; print(scipy.__version__)')"
PKG_CONFIG_PATH="${view}/lib/pkgconfig:${view}/lib64/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
export PKG_CONFIG_PATH
pkg-config --exists openblas
PIP_BREAK_SYSTEM_PACKAGES=1 "${py}" -m pip install --no-cache-dir --force-reinstall --no-deps \
    --no-binary numpy,scipy -Csetup-args=-Dblas=openblas -Csetup-args=-Dlapack=openblas \
    "numpy==${numpy_v}" "scipy==${scipy_v}"
PIP_BREAK_SYSTEM_PACKAGES=1 "${py}" -m pip uninstall -y scipy-openblas32 scipy-openblas64 || true

VIEW="${view}" NUMPY_V="${numpy_v}" SCIPY_V="${scipy_v}" "${py}" - <<'PY'
import os
import pathlib
import numpy
import scipy

view = os.environ["VIEW"]
assert numpy.__version__ == os.environ["NUMPY_V"], ("numpy moved", numpy.__version__)
assert scipy.__version__ == os.environ["SCIPY_V"], ("scipy moved", scipy.__version__)
for mod in (numpy, scipy):
    blas = mod.show_config(mode="dicts")["Build Dependencies"]["blas"]
    assert blas["name"] == "openblas", (mod.__name__, blas)
    assert str(blas.get("lib directory", "")).startswith(view), (mod.__name__, blas)
    bundled = [p for p in pathlib.Path(mod.__file__).parent.parent.rglob("*openblas*.so*") if view not in str(p)]
    assert not bundled, (mod.__name__, "bundles its own BLAS", bundled)
print("numpy", numpy.__version__, "scipy", scipy.__version__, "on", view)
PY

NUMBA_THREADING_LAYER=omp "${py}" - <<'PY'
import os

import numba
import numpy as np
from scipy.linalg import lu_factor

callers = 2 * (os.cpu_count() or 1)


@numba.njit(parallel=True)
def concurrent(a, out):
    for i in numba.prange(out.shape[0]):
        out[i] = np.dot(a, a)[0, 0]


a = np.random.default_rng(0).random((256, 256))
out = np.empty(callers)
concurrent(a, out)
assert np.allclose(out, (a @ a)[0, 0]), out
assert numba.threading_layer() == "omp", numba.threading_layer()
lu_factor(np.eye(512) + a[0, 0])
print("numba omp:", callers, "concurrent BLAS callers ok")
PY
