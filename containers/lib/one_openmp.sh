#!/bin/sh
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# One GNU OpenMP runtime file per image, then a gate that proves one is mapped.
#
#   one_openmp.sh [--link-only] <view prefix>      (e.g. /opt/view)
#
# ONE_OPENMP_ROOTS (default: /usr /lib /lib64 /opt /root /home and the view) lists the trees searched for copies.
#
# Two runtimes in one process cannot see each other's parallel region: OpenBLAS inside a numba
# prange thread opened a full team per caller (nproc^2 threads). Every libgomp copy the image holds
# -- the system one, spack's gcc-runtime copies, a wheel's bundled libgomp-<hash>.so.1* (torch,
# scikit-learn, xgboost, ...) -- becomes a link to the file the image compiler (CC, else gcc) ships.
# The loader keys objects by file, so a wheel that names its own copy maps the compiler's.
#
# Run it AFTER the last layer that can install an OpenMP runtime (a later uv pip install of a wheel
# that bundles one puts a copy back), and it is safe to run again. LLVM's libgomp shim (a link to
# libomp) and libiomp5/libomp copies are not GNU libgomp: they are left alone, and the gate fails
# when one is mapped. openmp_gate.py (COPYed beside this script) does the counting.
set -eu
ulimit -c 0
gate=1
if [ "$1" = --link-only ]; then
    gate=0
    shift
fi
view="$1"
here="$(cd "$(dirname "$0")" && pwd)"
py="$(command -v python3)"
gomp="$(readlink -f "$("${CC:-gcc}" -print-file-name=libgomp.so.1)")"
test -f "${gomp}"

# Only regular libgomp copies and links to one: never a 32-bit multilib, never LLVM's shim. A copy
# that defines a symbol version the compiler's libgomp lacks is not replaced but reported (a wheel
# built against a newer GCC than the image's would then fail at its first call into it).
roots=""
for dir in ${ONE_OPENMP_ROOTS:-/usr /lib /lib64 /opt /root /home} "${view}"; do
    [ -d "${dir}" ] && roots="${roots} ${dir}"
done
versions() { objdump -T "$1" | awk 'NF > 1 {print $(NF-1)}' | grep -E '^(GOMP|OMP)_' | sort -u; }
gomp_versions="$(mktemp)"
versions "${gomp}" > "${gomp_versions}"
# shellcheck disable=SC2086
find ${roots} -xdev \( -type f -o -type l \) \( -name 'libgomp.so.1*' -o -name 'libgomp-*.so*' \) \
        -not -path '*/lib32/*' -not -path '*/libx32/*' -not -path '*/32/*' -not -path '*/x32/*' \
        -print | sort -u | while read -r copy; do
    # A dangling link maps nothing (apt llvm's libgomp.so.1 shim without libomp: silent set -e exit, daint 4952006).
    real="$(readlink -f "${copy}")" || continue
    case "$(basename "${real}")" in libgomp*) ;; *) continue ;; esac
    [ "${real}" = "${gomp}" ] && continue
    missing="$(versions "${real}" | comm -23 - "${gomp_versions}" | tr '\n' ' ')"
    if [ -n "${missing}" ]; then
        echo "one_openmp: ${copy} needs ${missing}which ${gomp} lacks" >&2
        exit 1
    fi
    ln -sf "${gomp}" "${copy}"
    echo "one_openmp: ${copy} -> ${gomp}"
done
rm -f "${gomp_versions}"
[ "${gate}" = 1 ] || exit 0

# The gate: `openmp_gate.py one` runs numba prange calling BLAS, a gcc -fopenmp library and a torch op
# in one process, imports every other wheel that may bundle a runtime, and asserts one is mapped.
NUMBA_THREADING_LAYER=omp "${py}" "${here}/openmp_gate.py" one
