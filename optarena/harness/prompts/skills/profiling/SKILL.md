---
name: profiling
description: Measuring before and after -- which tool answers which question.
---

Measure before you edit, and again after. A change you cannot measure is a change you
cannot defend.

| question | tool |
| --- | --- |
| where does the time go? | `perf record` + `perf report` |
| is it stalling on memory or on issue? | `perf stat` (cycles, instructions, cache-misses) |
| what is the cache behaviour of this nest? | `valgrind --tool=cachegrind` |
| which call path dominates? | `valgrind --tool=callgrind`, `pprof` (gperftools) |
| what bandwidth am I actually getting? | `likwid-perfctr`, PAPI counters |
| where are the allocations? | `heaptrack` |
| did it actually vectorize? | `objdump -d` on the symbol, or the compiler's vector report |

Two rules that save the most time:

1. **Compare like with like.** Same shapes, same thread count, same build flags. Run each
   configuration more than once -- a single timing on a shared host is noise.
2. **Attribute the win.** Change one thing at a time. If two transforms land together and
   the kernel gets slower, you cannot tell which one to revert.
