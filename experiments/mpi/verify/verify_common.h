// Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Shared by the MPI / GPU-aware MPI / stream-enqueue / RCCL verification programs.
//
// Every program ends with exactly one line per test on rank 0:
//   VERDICT <test> <PASS|FAIL|UNSUPPORTED> key=value ...
// verify.sh greps those lines into the results table. PASS and FAIL are about the numbers the
// program checked; UNSUPPORTED means the library refused the API (an error return, or the API
// is absent from mpi.h), which is a property of the stack, not a wrong answer.
#ifndef VERIFY_COMMON_H
#define VERIFY_COMMON_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define HC(expr)                                                                                                       \
  do {                                                                                                                 \
    hipError_t err_ = (expr);                                                                                          \
    if (err_ != hipSuccess) {                                                                                          \
      fprintf(stderr, "HIP %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, hipGetErrorString(err_));                    \
      MPI_Abort(MPI_COMM_WORLD, 2);                                                                                    \
    }                                                                                                                  \
  } while (0)

// The node-local rank, from MPI itself (not SLURM_LOCALID): the same answer under any launcher.
static inline int local_rank(MPI_Comm *shm_out) {
  int rank = 0, lrank = 0;
  MPI_Comm shm;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shm);
  MPI_Comm_rank(shm, &lrank);
  if (shm_out != NULL)
    *shm_out = shm;
  else
    MPI_Comm_free(&shm);
  return lrank;
}

// Number of distinct nodes in MPI_COMM_WORLD.
static inline int node_count(void) {
  MPI_Comm shm;
  int lrank = local_rank(&shm), lead = lrank == 0, nodes = 0;
  MPI_Allreduce(&lead, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Comm_free(&shm);
  return nodes;
}

// Sum of a per-rank failure count over the world: 0 on every rank means every rank checked clean.
static inline long long world_sum(long long mine) {
  long long all = 0;
  MPI_Allreduce(&mine, &all, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  return all;
}

// Worst per-rank status over the world, ordered PASS(0) < UNSUPPORTED(1) < FAIL(2).
enum { V_PASS = 0, V_UNSUPPORTED = 1, V_FAIL = 2 };
static inline int world_worst(int mine) {
  int all = 0;
  MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return all;
}
static inline const char *verdict_name(int v) { return v == V_PASS ? "PASS" : v == V_FAIL ? "FAIL" : "UNSUPPORTED"; }

#endif
