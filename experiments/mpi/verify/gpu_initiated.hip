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
// A stream-enqueue call that MPICH accepts but never completes is the failure this probe is most
// exposed to (a progress thread that never runs leaves no output at all), so: stdout unbuffered,
// a `STEP` line before every call, ONE element per message, and a 30 s alarm around each call whose
// handler prints UNSUPPORTED_HANG naming the stage it stopped in. The reserved-VCI count MPICH
// needs for a GPU stream is read from MPI_T first and reported with the verdict.
//
// Verdicts are kept apart on purpose:
//   UNSUPPORTED  the API is absent from mpi.h, or MPICH returned an error for a call (stage and
//                MPI error string reported) -- the stack does not offer it.
//   UNSUPPORTED_HANG  a call was accepted and never returned: offered in name only.
//   FAIL         every call returned MPI_SUCCESS but the data is wrong -- the stack claims the
//                feature and gets it wrong. This is the dangerous one.
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

// MPICH puts a GPU stream on its own virtual communication interface; with none reserved,
// MPIX_Stream_create has nothing to attach to. Read the count MPICH is running with.
static int reserved_vcis(void) {
  int provided = 0, n = 0, value = -1;
  MPI_T_init_thread(MPI_THREAD_SINGLE, &provided);
  MPI_T_cvar_get_num(&n);
  for (int i = 0; i < n; ++i) {
    char name[256], desc[8];
    int nl = sizeof name, dl = sizeof desc, verb = 0, bind = 0, scope = 0, count = 0;
    MPI_Datatype dt;
    MPI_T_enum en;
    if (MPI_T_cvar_get_info(i, name, &nl, &verb, &dt, &en, desc, &dl, &bind, &scope) != MPI_SUCCESS)
      continue;
    if (strcmp(name, "MPIR_CVAR_CH4_RESERVE_VCIS") != 0 || bind != MPI_T_BIND_NO_OBJECT || dt != MPI_INT)
      continue;
    MPI_T_cvar_handle h;
    if (MPI_T_cvar_handle_alloc(i, NULL, &h, &count) != MPI_SUCCESS)
      continue;
    MPI_T_cvar_read(h, &value);
    MPI_T_cvar_handle_free(&h);
  }
  MPI_T_finalize();
  return value;
}

static int report(int rank, int v, const char *stage, int err, const char *extra) {
  watchdog_off();
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
  setvbuf(stdout, NULL, _IONBF, 0); // a killed step must still show which stage it stopped in
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  HC(hipSetDevice(local_rank(NULL)));
  int vcis = reserved_vcis();
  if (rank == 0)
    printf("STREAM reserved_vcis=%d (MPIR_CVAR_CH4_RESERVE_VCIS)\n", vcis);
#if !defined(MPIX_STREAM_NULL)
  int rc = report(rank, V_UNSUPPORTED, "compile", MPI_SUCCESS, "no MPIX_Stream API in mpi.h");
  MPI_Finalize();
  return rc;
#else
  hipStream_t hs;
  HC(hipStreamCreate(&hs));
  const char *types[] = {"hipStream_t", "cudaStream_t", "hip_stream"};
  const char *me = rank == 0 ? "gpu_initiated" : NULL;
  MPIX_Stream ms = MPIX_STREAM_NULL;
  MPI_Info info = MPI_INFO_NULL;
  int err = MPI_SUCCESS, t = 0;
  for (; t < 3; ++t) {
    if (rank == 0)
      printf("STEP MPIX_Stream_create type=%s\n", types[t]);
    watchdog(30, "UNSUPPORTED_HANG", me, "MPIX_Stream_create");
    MPI_Info_create(&info);
    MPI_Info_set(info, "type", types[t]);
    err = MPIX_Info_set_hex(info, "value", &hs, sizeof hs);
    if (err == MPI_SUCCESS)
      err = MPIX_Stream_create(info, &ms);
    MPI_Info_free(&info);
    watchdog(90, "UNSUPPORTED_HANG", me, "agreement");
    if (err == MPI_SUCCESS)
      break;
  }
  char extra[160];
  snprintf(extra, sizeof extra, "type=%s reserved_vcis=%d", t < 3 ? types[t] : "none-accepted", vcis);
  if (world_worst(err != MPI_SUCCESS ? V_UNSUPPORTED : V_PASS) != V_PASS) {
    int rc = report(rank, V_UNSUPPORTED, "MPIX_Stream_create", err, extra);
    MPI_Finalize();
    return rc;
  }
  MPI_Comm sc = MPI_COMM_NULL;
  if (rank == 0)
    printf("STEP MPIX_Stream_comm_create\n");
  watchdog(30, "UNSUPPORTED_HANG", me, "MPIX_Stream_comm_create");
  err = MPIX_Stream_comm_create(MPI_COMM_WORLD, ms, &sc);
  watchdog(90, "UNSUPPORTED_HANG", me, "agreement");
  if (world_worst(err != MPI_SUCCESS ? V_UNSUPPORTED : V_PASS) != V_PASS) {
    int rc = report(rank, V_UNSUPPORTED, "MPIX_Stream_comm_create", err, extra);
    MPI_Finalize();
    return rc;
  }
  MPI_Comm_set_errhandler(sc, MPI_ERRORS_RETURN);

  const int n = 1; // one element: this probe asks whether the path works at all, never how fast
  float *a = NULL, *b = NULL, *host = (float *)malloc(n * sizeof(float));
  HC(hipMalloc((void **)&a, n * sizeof(float)));
  HC(hipMalloc((void **)&b, n * sizeof(float)));
  fill<<<1, 64, 0, hs>>>(a, (float)(rank + 1), n);
  HC(hipGetLastError());
  if (rank == 0)
    printf("STEP MPIX_Send_enqueue / MPIX_Recv_enqueue\n");
  watchdog(30, "UNSUPPORTED_HANG", me, "MPIX_Send/Recv_enqueue");
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
  watchdog(90, "UNSUPPORTED_HANG", me, "agreement");
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
  fill<<<1, 64, 0, hs>>>(a, (float)(rank + 1), n);
  if (rank == 0)
    printf("STEP MPIX_Allreduce_enqueue\n");
  watchdog(30, "UNSUPPORTED_HANG", me, "MPIX_Allreduce_enqueue");
  double t0 = MPI_Wtime();
  err = MPIX_Allreduce_enqueue(a, b, n, MPI_FLOAT, MPI_SUM, sc);
  HC(hipStreamSynchronize(hs));
  watchdog(90, "UNSUPPORTED_HANG", me, "agreement");
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
  watchdog_off();
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
