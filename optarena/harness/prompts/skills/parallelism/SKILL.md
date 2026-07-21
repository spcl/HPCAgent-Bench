---
name: parallelism
description: Threading a kernel -- what is safe to parallelize, scheduling, false sharing.
---

- **Pick the outermost safe loop.** Parallelize high in the nest so each thread gets a
  large chunk; parallelizing an inner loop pays the fork/join cost per outer iteration.
- **Prove independence first.** A loop is parallel only if no iteration writes something
  another iteration reads. Loop-carried dependences must be removed (privatize a scalar,
  reorder, or split the loop) before the loop can be threaded.
- **Reductions** get `reduction(...)`, not a shared accumulator. Note that the summation
  order changes, so the result must still land inside the tolerance.
- **Schedule.** Use `static` for uniform iterations, `dynamic`/`guided` when the per-
  iteration cost varies (triangular loops, early exits). A wrong schedule leaves cores idle.
- **False sharing.** Two threads writing different elements of the same cache line
  serialize. Pad per-thread accumulators to a cache line, or accumulate in a local and
  write once at the end.
- **First touch.** On a multi-socket host, memory lands on the node that first writes it.
  Initialize arrays with the same loop decomposition the compute loop uses.

Check the scaling before keeping a change: if a thread count increase does not reduce the
time, the kernel is bandwidth-bound and threading is not the lever.
