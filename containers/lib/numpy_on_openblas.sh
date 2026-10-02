#!/bin/sh
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Rebuild the numpy and scipy that uv.lock pins against the image's OpenBLAS.
#
#   numpy_on_openblas.sh <view prefix> <workspace>      (e.g. /opt/view /opt/hpcagent-bench)
#
# The PyPI wheels bundle their own scipy-openblas: a pthreads build with MAX_THREADS=64 that crashed
# when numba's prange threads (192 on an mi300 node) called np.linalg at once, and a second BLAS
# runtime beside the image's OpenMP one. Built from source against <view>'s openblas (threads=openmp,
# blas_gate.sh), numpy and scipy share one BLAS and one OpenMP runtime with the C baselines, DaCe and
# numba (NUMBA_THREADING_LAYER=omp, set by the Dockerfile). uv.lock decides the versions, so this is the
# same numpy that computes every CPU reference, only linked differently. <workspace> holds the COPY'd
# pyproject.toml, uv.lock and agent/pyproject.toml; the environment is the interpreter's prefix unless
# UV_PROJECT_ENVIRONMENT names one. The gate at the end runs 2 x nproc numba prange iterations that each
# call np.dot and scipy.linalg.lu_factor concurrently, in one process that maps a single OpenMP runtime
# (one_openmp.sh, which also links every wheel's bundled libgomp to the image's).
set -eux
ulimit -c 0
view="$1"
workspace="$2"
py="$(command -v python3)"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$("${py}" -c 'import sys; print(sys.prefix)')}"
export UV_PROJECT_ENVIRONMENT
PKG_CONFIG_PATH="${view}/lib/pkgconfig:${view}/lib64/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
export PKG_CONFIG_PATH
pkg-config --exists openblas
sync() {
    (cd "${workspace}" && uv sync --frozen --inexact --no-cache --python "${py}" --no-install-project \
        --no-install-package hpcagent-agent --group openblas-build "$@")
}
# numpy first: scipy's build imports the installed numpy, which one combined reinstall removes mid-build.
sync --reinstall-package numpy --no-binary-package numpy \
    --config-settings-package numpy:setup-args=-Dblas=openblas --config-settings-package numpy:setup-args=-Dlapack=openblas
# scipy without build isolation, so it compiles against the numpy just rebuilt: an isolated build env
# builds a numpy of its own from source (--no-binary), which scipy's meson then failed to import (AMD 655840).
# Its build tools are the locked openblas-build group, already in the environment.
sync --reinstall-package scipy --no-binary-package scipy --no-build-isolation-package scipy \
    --config-settings-package scipy:setup-args=-Dblas=openblas --config-settings-package scipy:setup-args=-Dlapack=openblas

# A spack-built gcc writes its runtime directory as DT_RPATH into everything it links (AMD 656542), and
# DT_RPATH is searched before LD_LIBRARY_PATH: the llvm context could not put its own libgomp.so.1 and
# libopenblas.so.0 first. patchelf rewrites the same path as DT_RUNPATH, which is searched after it.
site="$("${py}" -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')"
patchelf="$(command -v patchelf || echo "$(dirname "${py}")/patchelf")"
find "${site}/numpy" "${site}/scipy" -name '*.so' | while read -r so; do
    if readelf -d "${so}" | grep -q '(RPATH)'; then
        rpath="$("${patchelf}" --print-rpath "${so}")"
        "${patchelf}" --remove-rpath "${so}"
        "${patchelf}" --set-rpath "${rpath}" "${so}"
    fi
done

VIEW="${view}" "${py}" - <<'PY'
import os
import pathlib
import numpy
import scipy

view = os.environ["VIEW"]
# pkg-config reports the spack install prefix the view links to, so compare resolved directories.
view_lib = pathlib.Path(view, "lib", "libopenblas.so").resolve().parent
for mod in (numpy, scipy):
    blas = mod.show_config(mode="dicts")["Build Dependencies"]["blas"]
    assert blas["name"] == "openblas", (mod.__name__, blas)
    assert pathlib.Path(str(blas.get("lib directory", ""))).resolve() == view_lib, (mod.__name__, blas, view_lib)
    bundled = list(pathlib.Path(mod.__file__).parent.parent.glob(f"{mod.__name__}*libs/*openblas*"))
    assert not bundled, (mod.__name__, "bundles its own BLAS", bundled)
# numpy and scipy resolve libopenblas.so.0 BY SONAME and must stay movable between OpenMP contexts
# (hpcagent_bench/omp_context.py): a child of the llvm family puts its own libopenblas.so.0 first on
# LD_LIBRARY_PATH, which only works while nothing here carries an absolute DT_RPATH (searched BEFORE it).
# A DT_RUNPATH or $ORIGIN entry is fine.
import subprocess

for mod in (numpy, scipy):
    for so in sorted(pathlib.Path(mod.__file__).parent.rglob("*.so")):
        dynamic = subprocess.run(["readelf", "-d", str(so)], capture_output=True, text=True, check=True).stdout
        pinned = [ln.strip() for ln in dynamic.splitlines() if "(RPATH)" in ln and "[/" in ln]
        assert not pinned, (so, "carries an absolute RPATH, which beats LD_LIBRARY_PATH", pinned)
print("numpy", numpy.__version__, "scipy", scipy.__version__, "on", view)
PY

# One OpenMP runtime: spack links OpenBLAS against its gcc-runtime copy of libgomp, numba's pool
# against the system one, and two runtimes cannot see each other's parallel region (nproc^2 threads).
# one_openmp.sh links every libgomp copy to the compiler's and gates on a single mapped runtime.
here="$(cd "$(dirname "$0")" && pwd)"
sh "${here}/one_openmp.sh" "${view}"
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
