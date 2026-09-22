// Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Is the image's MPICH GPU-aware on this MI300A, and what does a device pointer cost?
//
// On an APU a correct transfer from a hipMalloc buffer proves NOTHING by itself: host memory and
// device memory share one address space, so a non-GPU-aware MPI reads the buffer through the host
// path and the numbers still come out right. The verdict therefore rests on three things, and the
// transfer is only the last of them:
//   1. MPIX_GPU_query_support(MPIX_GPU_SUPPORT_HIP) -- MPICH's own answer (mpi.h, no mpi-ext.h).
//   2. The MPI_T control variables MPICH is running with (MPIR_CVAR_ENABLE_GPU, the OFI HMEM
//      switch, ...) and MPICH_GPU_SUPPORT_ENABLED / MPIR_CVAR_ENABLE_GPU from the environment.
//   3. hipPointerGetAttributes says the buffer really is device memory, and a Sendrecv ring plus
//      an Allreduce on it are elementwise right.
// PASS = query yes AND buffer is device memory AND data right. Query no + data right = FAIL
// ("works" only through the APU host path). The `control` mode runs under MPIR_CVAR_ENABLE_GPU=0
// and PASSes only if the query then says NO: that proves the query is live, not hard-wired.
//
// Also prints pingpong bandwidth host vs hipMalloc buffers, intra-node pair (0,1) and inter-node
// pair (0, first rank of the second node) -- the number that tells a real device path from a
// staged one.
//
// Usage: gpuaware [control]
#include <hip/hip_runtime.h>

#include <string.h>

#include "verify_common.h"

static void print_cvars(void) {
  int provided = 0, n = 0;
  MPI_T_init_thread(MPI_THREAD_SINGLE, &provided);
  MPI_T_cvar_get_num(&n);
  for (int i = 0; i < n; ++i) {
    char name[256], desc[8];
    int nl = sizeof name, dl = sizeof desc, verb = 0, bind = 0, scope = 0, count = 0;
    MPI_Datatype dt;
    MPI_T_enum en;
    if (MPI_T_cvar_get_info(i, name, &nl, &verb, &dt, &en, desc, &dl, &bind, &scope) != MPI_SUCCESS)
      continue;
    if (!(strstr(name, "GPU") || strstr(name, "HMEM") || strstr(name, "OFI_USE_PROVIDER")))
      continue;
    MPI_T_cvar_handle h;
    if (bind != MPI_T_BIND_NO_OBJECT || MPI_T_cvar_handle_alloc(i, NULL, &h, &count) != MPI_SUCCESS)
      continue;
    if (dt == MPI_INT && count == 1) {
      int v = 0;
      MPI_T_cvar_read(h, &v);
      printf("CVAR %s=%d\n", name, v);
    } else if (dt == MPI_CHAR && count < 512) {
      char v[512] = {0};
      MPI_T_cvar_read(h, v);
      printf("CVAR %s='%s'\n", name, v);
    }
    MPI_T_cvar_handle_free(&h);
  }
  MPI_T_finalize();
}

// One-direction MB/s between ranks a and b (bytes / half round trip); 0 on everyone else.
static double pingpong(int me, int a, int b, size_t bytes, int device, int iters) {
  char *buf = NULL;
  if (device) {
    HC(hipMalloc((void **)&buf, bytes));
    HC(hipMemset(buf, 1, bytes));
    HC(hipDeviceSynchronize());
  } else {
    buf = (char *)malloc(bytes);
    memset(buf, 1, bytes);
  }
  double t = 0.0;
  if (me == a || me == b) {
    int peer = me == a ? b : a;
    for (int it = -2; it < iters; ++it) {
      if (it == 0)
        t = MPI_Wtime();
      if (me == a) {
        MPI_Send(buf, (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
        MPI_Recv(buf, (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(buf, (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(buf, (int)bytes, MPI_BYTE, peer, 0, MPI_COMM_WORLD);
      }
    }
    t = MPI_Wtime() - t;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (device)
    HC(hipFree(buf));
  else
    free(buf);
  return t > 0 ? (double)bytes * iters * 2 / t / 1e6 : 0.0;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 0, ndev = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int control = argc > 1 && strcmp(argv[1], "control") == 0;
  int lrank = local_rank(NULL), nodes = node_count();
  HC(hipGetDeviceCount(&ndev));
  HC(hipSetDevice(lrank % (ndev > 0 ? ndev : 1)));

  int query = -1;
#if defined(MPIX_GPU_SUPPORT_HIP)
  if (MPIX_GPU_query_support(MPIX_GPU_SUPPORT_HIP, &query) != MPI_SUCCESS)
    query = -1;
#endif
  if (rank == 0) {
    const char *e1 = getenv("MPICH_GPU_SUPPORT_ENABLED"), *e2 = getenv("MPIR_CVAR_ENABLE_GPU");
    printf("GPUAWARE query_hip=%d env MPICH_GPU_SUPPORT_ENABLED=%s MPIR_CVAR_ENABLE_GPU=%s ndev=%d nodes=%d\n", query,
           e1 ? e1 : "unset", e2 ? e2 : "unset", ndev, nodes);
    print_cvars();
    fflush(stdout);
  }
  if (control) {
    int v = query == 0 ? V_PASS : V_FAIL;
    if (rank == 0)
      printf("VERDICT gpuaware_control %s query_hip=%d (MPIR_CVAR_ENABLE_GPU=0 must turn the query off)\n",
             verdict_name(v), query);
    MPI_Finalize();
    return v == V_PASS ? 0 : 1;
  }

  const int n = 1 << 22;
  float *dsend = NULL, *drecv = NULL, *host = (float *)malloc(n * sizeof(float));
  HC(hipMalloc((void **)&dsend, n * sizeof(float)));
  HC(hipMalloc((void **)&drecv, n * sizeof(float)));
  hipPointerAttribute_t attr;
  HC(hipPointerGetAttributes(&attr, dsend));
  int is_device = attr.type == hipMemoryTypeDevice;
  for (int i = 0; i < n; ++i)
    host[i] = (float)(rank + 1) + (float)(i % 7);
  HC(hipMemcpy(dsend, host, n * sizeof(float), hipMemcpyHostToDevice));
  int from = (rank - 1 + size) % size;
  long long bad = 0;
  MPI_Sendrecv(dsend, n, MPI_FLOAT, (rank + 1) % size, 1, drecv, n, MPI_FLOAT, from, 1, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  HC(hipMemcpy(host, drecv, n * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i)
    bad += host[i] != (float)(from + 1) + (float)(i % 7);
  MPI_Allreduce(dsend, drecv, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  HC(hipMemcpy(host, drecv, n * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i)
    bad += host[i] != (float)(size * (size + 1) / 2) + (float)(size * (i % 7));
  bad = world_sum(bad);
  int all_device = world_worst(!is_device) == 0;

  // Rank layout is block by node (srun default): the second node starts at the first rank whose
  // node-local rank is 0, other than rank 0.
  int first_other = size, cand = lrank == 0 && rank > 0 ? rank : size;
  MPI_Allreduce(&cand, &first_other, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  const size_t sizes[] = {1u << 20, 64u << 20};
  double bw[2][2][2] = {{{0}}}; // [size][host/device][intra/inter]
  for (int s = 0; s < 2; ++s)
    for (int d = 0; d < 2; ++d) {
      int iters = s == 0 ? 50 : 10;
      bw[s][d][0] = size > 1 ? pingpong(rank, 0, 1, sizes[s], d, iters) : 0;
      bw[s][d][1] = first_other < size ? pingpong(rank, 0, first_other, sizes[s], d, iters) : 0;
      if (rank == 0)
        printf("BW %-9s %6zu KiB intra(0-1)=%8.0f MB/s inter(0-%d)=%8.0f MB/s\n", d ? "hipMalloc" : "host",
               sizes[s] >> 10, bw[s][d][0], first_other < size ? first_other : -1, bw[s][d][1]);
    }
  int v = query == 1 && all_device && bad == 0 ? V_PASS : V_FAIL;
  if (rank == 0)
    printf("VERDICT gpuaware %s query_hip=%d device_ptr=%d bad=%lld bw64M_dev_intra=%.0f bw64M_dev_inter=%.0f "
           "bw64M_host_intra=%.0f bw64M_host_inter=%.0f MB/s\n",
           verdict_name(v), query, all_device, bad, bw[1][1][0], bw[1][1][1], bw[1][0][0], bw[1][0][1]);
  HC(hipFree(dsend));
  HC(hipFree(drecv));
  free(host);
  MPI_Finalize();
  return v == V_PASS ? 0 : 1;
}
