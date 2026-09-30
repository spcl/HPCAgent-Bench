#!/bin/sh
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Build-time gate on the image's OpenBLAS: the shapes and the concurrency that broke it.
#
#   blas_gate.sh <include dir> <lib dir>
#
# 1. Tall GEMMs, row- and column-major, one thread and every thread, under every x86 kernel family
#    OPENBLAS_CORETYPE can force: spack's NO_AVX512 with DYNAMIC_ARCH segfaulted row-major dgemm at
#    M >= 8192, K >= 512 on Zen 4 (spack-overlay/.../openblas/package.py).
# 2. Concurrent callers from plain pthreads (what numba's TBB pool and an agent's own threads do):
#    one per hardware thread with OpenBLAS threading on, then twice that with each call single-
#    threaded. A build whose MAX_THREADS is below the caller count overflows its buffer table
#    (Ubuntu's MAX_THREADS=64 OpenBLAS crashed at 192 callers).
# Any crash fails the build; the gate prints what it ran.
set -eu

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
inc="$1"
lib="$2"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
cat > "${work}/gemm.c" <<'C'
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  long m = atol(argv[1]), n = atol(argv[2]), k = atol(argv[3]);
  int row = argv[4][0] == 'r';
  double *a = calloc((size_t)m * k, sizeof(double)), *b = calloc((size_t)k * n, sizeof(double));
  double *c = calloc((size_t)m * n, sizeof(double));
  if (!a || !b || !c) return 2;
  for (long i = 0; i < m * k; ++i) a[i] = 1.0 / (double)(1 + i % 7);
  for (long i = 0; i < k * n; ++i) b[i] = 1.0 / (double)(1 + i % 5);
  if (row)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b, n, 0.0, c, n);
  else
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, m, b, k, 0.0, c, m);
  printf("dgemm %s %ldx%ldx%ld ok\n", row ? "row" : "col", m, n, k);
  return 0;
}
C
cat > "${work}/callers.c" <<'C'
#include <cblas.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
static void *work(void *unused) {
  (void)unused;
  int n = 256;
  double *a = calloc((size_t)n * n, sizeof(double)), *b = calloc((size_t)n * n, sizeof(double));
  double *c = calloc((size_t)n * n, sizeof(double));
  for (int r = 0; r < 10; ++r)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, a, n, b, n, 0.0, c, n);
  free(a); free(b); free(c);
  return 0;
}
int main(int argc, char **argv) {
  int callers = atoi(argv[1]);
  pthread_t *t = malloc(sizeof(pthread_t) * (size_t)callers);
  for (int i = 0; i < callers; ++i) pthread_create(&t[i], 0, work, 0);
  for (int i = 0; i < callers; ++i) pthread_join(t[i], 0);
  printf("%d concurrent dgemm callers ok\n", callers);
  return 0;
}
C
cc -O2 "${work}/gemm.c" -I"${inc}" -L"${lib}" -lopenblas -fopenmp -lm -Wl,-rpath,"${lib}" -o "${work}/gemm"
cc -O2 "${work}/callers.c" -I"${inc}" -L"${lib}" -lopenblas -fopenmp -lpthread -lm -Wl,-rpath,"${lib}" -o "${work}/callers"
coretypes=""
case "$(uname -m)" in x86_64) coretypes="Haswell SkylakeX Zen" ;; esac
for coretype in "" ${coretypes}; do
  for threads in 1 "$(nproc)"; do
    for shape in "100000 406 815" "8192 406 512" "406 100000 815"; do
      for order in row col; do
        # shellcheck disable=SC2086
        OPENBLAS_CORETYPE="${coretype}" OMP_NUM_THREADS="${threads}" "${work}/gemm" ${shape} "${order}" >/dev/null \
          || { echo "blas_gate: dgemm ${order} ${shape} crashed (coretype '${coretype}', ${threads} threads)" >&2; exit 1; }
      done
    done
  done
done
echo "blas_gate: tall GEMMs pass (coretypes: default ${coretypes})"
"${work}/callers" "$(nproc)" || { echo "blas_gate: $(nproc) concurrent callers crashed" >&2; exit 1; }
OMP_NUM_THREADS=1 "${work}/callers" "$(( 2 * $(nproc) ))" \
  || { echo "blas_gate: $(( 2 * $(nproc) )) single-threaded concurrent callers crashed" >&2; exit 1; }
