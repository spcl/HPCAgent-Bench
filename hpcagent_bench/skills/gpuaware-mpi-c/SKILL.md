---
name: gpuaware-mpi-c
description: "GPU-aware MPI in C. Use whenever you pass a device pointer to `MPI_Isend`/`MPI_Allreduce` across nodes, check `MPIX_GPU_query_support`, or hit a hang or segfault with no diagnostic."
when: "your MPI buffers are device pointers -- more than one GPU, on one node or across nodes: ALWAYS read this page before you stage anything through the host, and before you read a working call as proof of the fast path"
applies: {images: [amd, nvidia], multinode: true, languages: [c, cpp, hip]}
---

# gpuaware-mpi-c

`mpi-c` decides what you communicate; this page is what changes when the buffers are device memory.
The judge's MPI is MPICH 4.3.2 (library `mpi`, header `mpi.h`), built GPU-aware for HIP. Calls are
host-side, between kernel launches; nothing is callable from a kernel.

## Query it; a working call proves nothing

```c
int ok = 0;
MPIX_GPU_query_support(MPIX_GPU_SUPPORT_HIP, &ok);   /* declared in mpi.h, no mpi-ext.h */
```

Nothing needs exporting: `MPIR_CVAR_ENABLE_GPU` is 1 by default (=0 flips the query to 0).
`MPICH_GPU_SUPPORT_ENABLED` is Cray MPICH's variable, and this build ignores it.

MI300A is an APU: CPU and GPU share one HBM, so a device pointer is host-addressable and a
non-GPU-aware path also "works" (slowly, through the CPU). Success is not evidence of the RDMA
path. If `ok == 0`, stage through host buffers.

## What "GPU-aware" buys here, and when staging is still faster

It means MPICH ACCEPTS a device pointer, not that the NIC reads HBM. In this build
`MPIR_CVAR_CH4_OFI_ENABLE_HMEM=0` and `MPIR_CVAR_CH4_OFI_GPU_RDMA_THRESHOLD=0`, so OFF-NODE MPICH
stages the buffer through the host itself; ON-NODE it switches to IPC above
`MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD` (1 MiB). Under `MPIR_CVAR_GPU_FAST_COPY_MAX_SIZE` (4096 B)
it memcpys regardless. So hand-staging a small message costs nothing and hand-staging a large one
buys a second copy -- pass the device pointer, and spend the effort on overlap instead.

## Traps

- **Device binding is done for you.** The harness sets the device to the node-local rank before
  allocating your tiles. Never `hipSetDevice` (least of all to the global rank): later
  allocations, launches and RCCL land on another GPU than your tiles. `hipGetDevice` to read it.
- **MPI does not see streams.** `hipStreamSynchronize(stream)` after the kernel that fills a send
  buffer and before `MPI_Isend`/`MPI_Allreduce`; a missing sync is a silent race that often passes
  at small sizes. A receive buffer is valid only after `MPI_Wait`.
- **MPICH has no bf16 datatype.** `MPI_BYTE` moves bf16 -- count in BYTES, not elements -- and
  takes no `MPI_Op`, so it cannot reduce it; reduce in an fp32 buffer (`MPI_FLOAT`) or use RCCL
  (`ncclBfloat16`).
- **Overlap = the `mpi-c` split one level up**: fill + sync the send buffer, post `Irecv`/`Isend`,
  launch the kernel that needs no remote data, `Waitall`, launch the rest.
