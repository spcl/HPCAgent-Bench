---
name: gpuaware-mpi-c
description: "GPU-aware MPI in C: passing device pointers to MPI, the synchronization MPI cannot see,
and the check to run before trusting any of it."
when: "a multi-node task has to move data between GPUs without staging it through the host"
---

# gpuaware-mpi-c

Only when your tiles arrive as **device** pointers. `mpi-c` still decides what you communicate and
when; this page is what changes because the bytes live in GPU memory.

## What GPU-aware buys, and what it does not

A GPU-aware MPI accepts a device pointer where it would take a host pointer, and moves the bytes
over GPUDirect/RDMA without a staging copy through host memory. That is all it is. It is **not**
GPU-initiated: you still call MPI from the host, from ordinary host code, between kernel launches.
Nothing here is callable from inside a kernel.

The alternative you are replacing is the manual one, and it is always correct:

```c
hipMemcpy(h_buf, d_buf, n, hipMemcpyDeviceToHost);   /* stage down  */
MPI_Isend(h_buf, ...);                                /* send host   */
```

Manual staging is the fallback whenever the check below says the MPI is not GPU-aware. It costs an
extra copy each way; it never returns a wrong answer.

## Verify it, do not assume it

**Run the check. A non-GPU-aware MPI handed a device pointer does not report an error -- it hangs
or segfaults, with no diagnostic naming the cause.** Measured: an `MPI_Bcast` on a `hipMalloc`
pointer compiled cleanly and then deadlocked forever, because that MPI was built without ROCm
support. Nothing in the build or the run says so.

The check depends on which MPI you have:

| MPI | how to ask |
|---|---|
| MPICH 4.1+ | `MPIX_GPU_query_support(MPIX_GPU_SUPPORT_HIP, &ok)` |
| Open MPI | `MPIX_Query_rocm_support()` (ROCm), `MPIX_Query_cuda_support()` (CUDA) |
| Cray MPICH | **no query API exists.** It requires `MPICH_GPU_SUPPORT_ENABLED=1` in the environment and the GTL library linked; without the variable, device pointers segfault silently |

If the answer is no, or if there is no way to ask, stage through the host. Do not "try it and see":
the failure is a hang, and it consumes everything you had left.

## The synchronization MPI cannot do for you

**MPI has no concept of a GPU stream.** It cannot see that a kernel is still writing the buffer you
just handed it, and there is no clause that tells it. The dependence is yours to enforce:

```c
compute_kernel<<<...>>>(d_send, ...);
hipStreamSynchronize(stream);      /* MANDATORY -- the buffer is not ready without it */
MPI_Isend(d_send, ...);
```

Omit the synchronize and MPI reads the buffer mid-write: a wrong answer that usually still looks
right at small sizes and low rank counts, which is the worst way for it to fail.

The same on the receive side: the data is in `d_recv` only after the `MPI_Wait` returns, and a
kernel launched on `d_recv` before that wait reads garbage.

## The cost this imposes, and the only way around it

Every one of those synchronizes is a host-blocking stall that breaks kernel-launch pipelining --
the pipeline you built by launching ahead. This is the known structural weakness of GPU-aware MPI,
not a mistake in your code: interleaving MPI calls with kernels forces host syncs, and host syncs
remove the overlap you were trying to get.

What still works is the same split as `mpi-c`, moved up a level:

```
launch the kernel that fills the HALO faces
hipStreamSynchronize(stream)      /* only the halo buffers must be ready */
post Irecv / Isend on the device pointers
launch the INTERIOR kernel        /* runs on the GPU while MPI moves bytes */
MPI_Waitall(...)
launch the BOUNDARY kernel        /* consumes what arrived */
hipStreamSynchronize(stream)      /* before returning */
```

The interior kernel is the overlap. Without it the non-blocking calls buy nothing, exactly as on
the host.

## Errors that cost a turn

- **Device pointer into a non-GPU-aware MPI: hang or segfault, no message.** Check first.
- **A missing `hipStreamSynchronize` before a send is a silent race**, not a build error.
- **Do not return with a stream unsynchronized.** Your outputs are read the moment the call returns.
- **`MPI_Init` and the communicator still belong to the caller** -- everything in `mpi-c` about
  `MPI_Comm_f2c(comm)`, matched collectives and `MPI_PROC_NULL` applies unchanged.
- **One rank per GPU.** Set it once with `hipSetDevice(rank % devices_per_node)` before any
  allocation; ranks that all land on device 0 serialize and the numbers mean nothing.
