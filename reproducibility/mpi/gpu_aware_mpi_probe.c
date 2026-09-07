// Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Is this image's MPI GPU-aware -- can a DEVICE pointer be handed to MPI directly?
//
// Two answers, because on an MI300A the functional one alone is worthless. MI300A is an APU:
// host memory is device-addressable, so a hipMalloc'd buffer passed to a NON-GPU-aware MPI can
// still be read by the host-side transport and the exchange succeeds anyway. A green transfer
// therefore proves the call did not crash, NOT that the MPI knows what a device pointer is.
//
//   1. The MPI's OWN GPU-support query, and the one that is decisive. The two MPI families spell
//      it differently and the difference is not cosmetic:
//        * MPICH >= 4.1: MPIX_GPU_query_support(MPIX_GPU_SUPPORT_HIP, &flag), declared in mpi.h.
//          MPICH ships NO mpi-ext.h at all.
//        * Open MPI: MPIX_Query_rocm_support(), declared in mpi-ext.h behind MPIX_GPU_SUPPORT_ROCM.
//      Probing only the Open MPI spelling against MPICH compiles the query out and reports "NO"
//      for every MPICH, GPU-aware or not -- which is exactly what an earlier version of this file
//      did, voiding both its v5 and v6 verdicts. Hence: check for the query API, and if NEITHER
//      family's is present say so as "unknown", never as "no".
//   2. A real device-pointer MPI_Sendrecv between two ranks, verified elementwise. Reported for
//      what it is: necessary, not sufficient.
//
// A discrete GPU would fail (2) outright without (1); an APU may pass (2) regardless. Read them
// together, and treat (1) as the verdict.
//
// Build:  hipcc -x hip gpu_aware_mpi_probe.c $(mpicc -show | cut -d' ' -f2-) -o probe
// Run:    <launcher> -n 2 ./probe

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <hip/hip_runtime.h>

#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#define N 4096

#define HIP_CHECK(expr)                                                                                                \
  do {                                                                                                                 \
    hipError_t err_ = (expr);                                                                                          \
    if (err_ != hipSuccess) {                                                                                          \
      fprintf(stderr, "hip error %s at %s:%d\n", hipGetErrorString(err_), __FILE__, __LINE__);                         \
      MPI_Abort(MPI_COMM_WORLD, 10);                                                                                   \
    }                                                                                                                  \
  } while (0)

__global__ void fill(double *buf, int n, double base) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n)
    buf[i] = base + (double)i;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    if (rank == 0)
      fprintf(stderr, "gpu_aware_mpi_probe: needs exactly 2 ranks, got %d\n", size);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  // Question 1: what does the MPI itself say? -1 = it offers no such query in either spelling.
  int declared = -1;
  const char *api = "no GPU-support query in this MPI";
#if defined(MPIX_GPU_SUPPORT_HIP)
  api = "MPIX_GPU_query_support(MPIX_GPU_SUPPORT_HIP)";
  {
    int supported = 0;
    declared = (MPIX_GPU_query_support(MPIX_GPU_SUPPORT_HIP, &supported) == MPI_SUCCESS && supported) ? 1 : 0;
  }
#elif defined(MPIX_GPU_SUPPORT_ROCM)
  api = "MPIX_Query_rocm_support()";
  declared = MPIX_Query_rocm_support() ? 1 : 0;
#endif
  if (rank == 0) {
    const char *verdict = declared == 1   ? "YES"
                          : declared == 0 ? "NO (query present, MPI built without ROCm/HIP support)"
                                          : "UNKNOWN (this MPI exposes no GPU-support query)";
    printf("GPU-support query: %s -> %s\n", api, verdict);
  }

  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  if (ndev < 1) {
    if (rank == 0)
      fprintf(stderr, "gpu_aware_mpi_probe: no HIP device visible\n");
    MPI_Abort(MPI_COMM_WORLD, 3);
  }
  HIP_CHECK(hipSetDevice(rank % ndev));

  // Question 2: does a device pointer survive an actual exchange?
  double *send = NULL, *recv = NULL;
  HIP_CHECK(hipMalloc((void **)&send, N * sizeof(double)));
  HIP_CHECK(hipMalloc((void **)&recv, N * sizeof(double)));
  const double base = rank == 0 ? 1000.0 : 2000.0;
  fill<<<(N + 255) / 256, 256>>>(send, N, base);
  HIP_CHECK(hipDeviceSynchronize());

  const int peer = 1 - rank;
  // DEVICE pointers straight into MPI. This is the call that a non-GPU-aware MPI cannot serve
  // on a discrete GPU, and that an APU may serve by accident.
  MPI_Sendrecv(send, N, MPI_DOUBLE, peer, 0, recv, N, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  double *host = (double *)malloc(N * sizeof(double));
  if (host == NULL)
    MPI_Abort(MPI_COMM_WORLD, 4);
  HIP_CHECK(hipMemcpy(host, recv, N * sizeof(double), hipMemcpyDeviceToHost));

  const double expect = peer == 0 ? 1000.0 : 2000.0;
  int bad = 0;
  for (int i = 0; i < N; i++) {
    if (host[i] != expect + (double)i)
      bad++;
  }
  free(host);
  HIP_CHECK(hipFree(send));
  HIP_CHECK(hipFree(recv));

  int bad_total = 0;
  MPI_Reduce(&bad, &bad_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  int rc = 0;
  if (rank == 0) {
    printf("device-pointer MPI_Sendrecv: %s (%d/%d elements wrong)\n",
           bad_total == 0 ? "transferred correctly" : "WRONG DATA", bad_total, 2 * N);
    if (bad_total != 0) {
      printf("VERDICT: FAIL -- the exchange itself is broken\n");
      rc = 1;
    } else if (declared == 1) {
      printf("VERDICT: GPU-AWARE -- MPI declares ROCm/HIP support and device pointers work\n");
    } else if (declared == 0) {
      printf("VERDICT: NOT GPU-AWARE -- the transfer worked, but this MPI declares no ROCm/HIP\n");
      printf("         support. On an APU that is host memory being read behind your back,\n");
      printf("         not a device transport. A discrete GPU would fail outright.\n");
      rc = 1;
    } else {
      printf("VERDICT: INCONCLUSIVE -- this MPI exposes no GPU-support query, so the transfer\n");
      printf("         above is the only evidence and on an APU it proves nothing. Identify the\n");
      printf("         MPI and its query API before reading this as either answer.\n");
      rc = 2;
    }
  }
  MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return rc;
}
