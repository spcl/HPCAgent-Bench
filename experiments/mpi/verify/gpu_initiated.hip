// Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Stream-triggered ("GPU-initiated") MPI in MPICH: communication ENQUEUED on a hipStream, ordered
// after the kernels before it, without the host synchronizing in between.
//
//   MPIX_Stream_create(info{type=hipStream_t, value=<hex of the hipStream_t>})
//   MPIX_Stream_comm_create(MPI_COMM_WORLD, stream) -> stream communicator
//   MPIX_Send_enqueue / MPIX_Recv_enqueue ring, then MPIX_Allreduce_enqueue, all on that stream.
//
// MPICH needs a reserved VCI for a GPU stream (MPIR_CVAR_CH4_RESERVE_VCIS>=1; verify.sh sets it).
// Several `type` spellings are tried in order and the accepted one is reported: MPICH has used
// both the CUDA spelling for HIP builds and its own.
//
// Verdicts are kept apart on purpose:
//   UNSUPPORTED  the API is absent from mpi.h, or MPICH returned an error for a call (stage and
//                MPI error string reported) -- the stack does not offer it.
//   FAIL         every call returned MPI_SUCCESS but the data is wrong (or verify.sh's timeout
//                killed a hang) -- the stack claims it and gets it wrong. This is the dangerous one.
//   PASS         every call succeeded and the ring + allreduce results are elementwise right.
// Kernel-side initiation (a device function calling MPI) does not exist in MPICH; that would be
// rocSHMEM, which is out of scope here.
#include <hip/hip_runtime.h>

#include <string.h>

#include "verify_common.h"

__global__ void fill(float *p, float v, int n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n)
    p[i] = v + (float)(i % 5);
}

static int report(int rank, int v, const char *stage, int err, const char *extra) {
  char msg[MPI_MAX_ERROR_STRING] = "";
  int len = 0;
  if (err != MPI_SUCCESS)
    MPI_Error_string(err, msg, &len);
  if (rank == 0)
    printf("VERDICT gpu_initiated %s stage=%s %s%s%s\n", verdict_name(v), stage, extra, err ? " err=" : "", msg);
  return v == V_PASS ? 0 : 1;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  HC(hipSetDevice(local_rank(NULL)));
#if !defined(MPIX_STREAM_NULL)
  int rc = report(rank, V_UNSUPPORTED, "compile", MPI_SUCCESS, "no MPIX_Stream API in mpi.h");
  MPI_Finalize();
  return rc;
#else
  hipStream_t hs;
  HC(hipStreamCreate(&hs));
  const char *types[] = {"hipStream_t", "cudaStream_t", "hip_stream"};
  MPIX_Stream ms = MPIX_STREAM_NULL;
  MPI_Info info = MPI_INFO_NULL;
  int err = MPI_SUCCESS, t = 0;
  for (; t < 3; ++t) {
    MPI_Info_create(&info);
    MPI_Info_set(info, "type", types[t]);
    err = MPIX_Info_set_hex(info, "value", &hs, sizeof hs);
    if (err == MPI_SUCCESS)
      err = MPIX_Stream_create(info, &ms);
    MPI_Info_free(&info);
    if (err == MPI_SUCCESS)
      break;
  }
  char extra[128];
  snprintf(extra, sizeof extra, "type=%s", t < 3 ? types[t] : "none-accepted");
  if (world_worst(err != MPI_SUCCESS ? V_UNSUPPORTED : V_PASS) != V_PASS) {
    int rc = report(rank, V_UNSUPPORTED, "MPIX_Stream_create", err, extra);
    MPI_Finalize();
    return rc;
  }
  MPI_Comm sc = MPI_COMM_NULL;
  err = MPIX_Stream_comm_create(MPI_COMM_WORLD, ms, &sc);
  if (world_worst(err != MPI_SUCCESS ? V_UNSUPPORTED : V_PASS) != V_PASS) {
    int rc = report(rank, V_UNSUPPORTED, "MPIX_Stream_comm_create", err, extra);
    MPI_Finalize();
    return rc;
  }
  MPI_Comm_set_errhandler(sc, MPI_ERRORS_RETURN);

  const int n = 1 << 22;
  float *a = NULL, *b = NULL, *host = (float *)malloc(n * sizeof(float));
  HC(hipMalloc((void **)&a, n * sizeof(float)));
  HC(hipMalloc((void **)&b, n * sizeof(float)));
  fill<<<n / 256, 256, 0, hs>>>(a, (float)(rank + 1), n);
  HC(hipGetLastError());
  int to = (rank + 1) % size, from = (rank - 1 + size) % size;
  // Even ranks send first, odd ranks receive first: each stream serializes its own ops.
  if (rank % 2 == 0) {
    err = MPIX_Send_enqueue(a, n, MPI_FLOAT, to, 7, sc);
    if (err == MPI_SUCCESS)
      err = MPIX_Recv_enqueue(b, n, MPI_FLOAT, from, 7, sc, MPI_STATUS_IGNORE);
  } else {
    err = MPIX_Recv_enqueue(b, n, MPI_FLOAT, from, 7, sc, MPI_STATUS_IGNORE);
    if (err == MPI_SUCCESS)
      err = MPIX_Send_enqueue(a, n, MPI_FLOAT, to, 7, sc);
  }
  HC(hipStreamSynchronize(hs));
  if (world_worst(err != MPI_SUCCESS ? V_UNSUPPORTED : V_PASS) != V_PASS) {
    int rc = report(rank, V_UNSUPPORTED, "MPIX_Send/Recv_enqueue", err, extra);
    MPI_Finalize();
    return rc;
  }
  HC(hipMemcpy(host, b, n * sizeof(float), hipMemcpyDeviceToHost));
  long long bad_ring = 0, bad_ar = 0;
  for (int i = 0; i < n; ++i)
    bad_ring += host[i] != (float)(from + 1) + (float)(i % 5);

  // Refill on the stream right before the collective: the enqueue must wait for this kernel.
  fill<<<n / 256, 256, 0, hs>>>(a, (float)(rank + 1), n);
  double t0 = MPI_Wtime();
  err = MPIX_Allreduce_enqueue(a, b, n, MPI_FLOAT, MPI_SUM, sc);
  HC(hipStreamSynchronize(hs));
  double ms_ar = (MPI_Wtime() - t0) * 1e3;
  if (world_worst(err != MPI_SUCCESS ? V_UNSUPPORTED : V_PASS) != V_PASS) {
    snprintf(extra + strlen(extra), sizeof extra - strlen(extra), " ring_bad=%lld", world_sum(bad_ring));
    int rc = report(rank, V_UNSUPPORTED, "MPIX_Allreduce_enqueue", err, extra);
    MPI_Finalize();
    return rc;
  }
  HC(hipMemcpy(host, b, n * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i)
    bad_ar += host[i] != (float)(size * (size + 1) / 2) + (float)(size * (i % 5));
  bad_ring = world_sum(bad_ring);
  bad_ar = world_sum(bad_ar);
  int v = bad_ring == 0 && bad_ar == 0 ? V_PASS : V_FAIL;
  snprintf(extra + strlen(extra), sizeof extra - strlen(extra),
           " ring_bad=%lld allreduce_bad=%lld t_allreduce16M=%.2fms", bad_ring, bad_ar, ms_ar);
  int rc = report(rank, v, "all", MPI_SUCCESS, extra);
  MPI_Comm_free(&sc);
  MPIX_Stream_free(&ms);
  HC(hipFree(a));
  HC(hipFree(b));
  free(host);
  MPI_Finalize();
  return rc;
#endif
}
