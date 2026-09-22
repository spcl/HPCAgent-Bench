---
name: gpuinit-mpi-c
description: "GPU-initiated (stream-triggered) MPI in C on MPICH. Use whenever a bf16 collective or a halo exchange has to run from a `hipStream_t` in stream order, or `MPIX_Stream_create` refuses."
when: "a multi-node GPU task alternates HIP kernels and collectives on one stream and the host sync between them is what you are paying for: ALWAYS read this page before you settle on RCCL or on host-called MPI"
applies: {images: [amd], multinode: true, languages: [c, cpp, hip]}
---

# gpuinit-mpi-c

Host-initiated GPU-aware MPI runs on the CPU: you sync the stream, THEN call MPI. MPICH's stream
communicators put the call itself on a `hipStream_t`, in stream order behind the kernels already
queued there. NOT device-side MPI: a kernel still cannot call it -- that is rocSHMEM, absent here.

## Setup: once, cached in a `static`

```c
hipStream_t s; hipStreamCreate(&s);
MPI_Info info; MPI_Info_create(&info);
MPI_Info_set(info, "type", "hipStream_t");       /* the handle's TYPE ... */
MPIX_Info_set_hex(info, "value", &s, sizeof s);  /* ... and the handle itself, as hex */
MPIX_Stream stream; MPIX_Stream_create(info, &stream);   /* an error code: check it, see below */
MPI_Info_free(&info);
MPI_Comm sc; MPIX_Stream_comm_create(MPI_Comm_f2c(comm), stream, &sc);
```

Per call, on that SAME stream: `kernel<<<g, b, 0, s>>>(...)`,
`MPIX_Allreduce_enqueue(send, recv, n, MPI_FLOAT, MPI_SUM, sc)`, the next kernel, then ONE
`hipStreamSynchronize(s)` before you return. `MPIX_Send_enqueue`/`MPIX_Recv_enqueue` are the
point-to-point pair; a stream serializes its own ops, so order a ring by rank parity or it
deadlocks in stream order, as a blocking `MPI_Send` ring does.

## Support, fallback, and when it wins

- `MPIR_CVAR_CH4_RESERVE_VCIS=1` must be in the environment: with no reserved interface,
  `MPIX_Stream_create` has nothing to attach the stream to.
- Absent at compile time -- `#if !defined(MPIX_STREAM_NULL)`; refused at run time -- an `MPIX_*`
  error code, and without `MPI_ERRORS_RETURN` on the communicator a refusal aborts the rank.
- The fallback is those same kernels and stream with `hipStreamSynchronize(s)` + host-side
  `MPI_Allreduce`; choose once, in the cached setup, never per call.
- Datatypes are MPI's: no bf16 datatype, so reduce in an fp32 buffer and downcast once.
- It removes one host round trip per communication STEP, so the gain grows with the number of
  steps, not the message size. One big allreduce per call has nothing to hide and RCCL wins it.

