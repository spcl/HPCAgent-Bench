---
name: openmp-cpp
description: "OpenMP in C++: the four loop bins, the sharing clauses, and the build errors that
cost a turn."
---

# openmp-cpp

Grading is MULTI-CORE, baseline SERIAL, `-fopenmp` always on. Never hardcode a thread count --
the grading machine presets `OMP_NUM_THREADS`; read `omp_get_max_threads()` (`#include <omp.h>`).
(`<execution>` policies are the other threading spelling -- the lang-cpp page; one spelling per
loop.)

## A directive is an assertion, not a request

`parallel for` claims the iterations may run in any order. `simd` claims the lanes may. Nothing
checks either claim; both are believed. Assert wrongly and you get a RACE -- a wrong answer, not a
slow one, and one that looks plausible.

So derive the dependence vectors FIRST and name the axis each one crosses (lang-cpp has the test,
the rewrites that free an axis, and the stride question that decides whether threading pays at
all). That axis may not be threaded; any axis it does not cross may. Misfiling returns
`correct: false` and costs a round trip.

If the comment you are about to write names a dependence, the directive is wrong. The tell:

```cpp
// the recurrence relation requires sequential processing   <- correct diagnosis
#pragma omp simd                                            <- contradicts it
for (std::int64_t i = 2; i < n; i++)
    a[i] = a[i-2] + x[i];
```

## The four bins

**PARALLEL** -- every write lands at this iteration's own subscript, no scalar carries state.
`#pragma omp parallel for simd` on the OUTERMOST such loop: threads across cores, lanes within
each. A tiny trip count loses to the cost of forking the team.

**REDUCTION** -- the only carried state is an accumulator (sum, max, min, count). Same directive
plus `reduction(+:s)` (or `max:`, `min:`). Never a shared scalar, never hand-built per-thread
arrays. The clause also authorizes the FP reassociation the compiler refuses on its own, and the
graded tolerance covers it.

**Max or min WITH ITS INDEX** -- `reduction(max:m)` returns the value and LOSES the position, and
there is no built-in argmax operator. Declare one over a value-index pair; break ties toward the
SMALLER index or the answer disagrees with a serial sweep:

```cpp
struct vi_t { double v; std::int64_t i; };
#pragma omp declare reduction(argmax : vi_t : \
        omp_out = (omp_in.v > omp_out.v || \
                   (omp_in.v == omp_out.v && omp_in.i < omp_out.i)) ? omp_in : omp_out) \
        initializer(omp_priv = { -DBL_MAX, INT64_MAX })

vi_t best = { -DBL_MAX, INT64_MAX };
#pragma omp parallel for reduction(argmax:best)
for (std::int64_t i = 0; i < n; i++)
    if (v[i] > best.v) { best.v = v[i]; best.i = i; }
```

Fallback without new syntax, two passes, both parallel: `reduction(max:m)` for the value, then
`reduction(min:first)` over positions where `v[i] == m` (exact compare: same stored value).

**RECURRENCE** -- the written array is read at ANOTHER iteration's subscript. The loop carrying
the chain stays serial; threading it races. That does not make the NEST serial: thread an axis the
chain does not cross, or reshape the nest so there is one (lang-cpp). Prefix sum is the one
recurrence with a parallel spelling of its own -- `inclusive_scan` / `exclusive_scan` (lang-cpp),
or the directive form:

```cpp
double s = 0.0;
#pragma omp parallel for simd reduction(inscan,+:s)
for (std::int64_t i = 0; i < n; i++) {
    s += a[i];
    #pragma omp scan inclusive(s)
    out[i] = s;
}
```

`exclusive(s)` is the value-before-this-iteration variant. Scans reassociate; tolerance applies.

**SCATTER** -- writes through an index array, `a[idx[i]]`. If the task guarantees distinct
indices it is PARALLEL, no atomics. Only DUPLICATE indices collide: then per-thread copies merged
after the loop (usually fastest), or `#pragma omp atomic` on the update (often slower than
serial).

## Fork and barrier cost

A worksharing construct ends with an IMPLICIT BARRIER. So `omp parallel` wrapped around an outer
sequential loop with `omp for` inside pays one barrier per outer iteration: every thread waits for
the slowest share of the row before any of them starts the next one, and that wait is charged as
many times as the outer loop runs.

- Default spelling: ONE combined `parallel for` over the whole iteration space -- the team forks
  once and synchronizes once. `collapse(n)` when the outer loop alone is too short to fill the
  cores: exactly n PERFECTLY nested loops, nothing between the headers.
- `nowait` on the inner `for` when what follows does not read what it wrote. Reach for the
  `parallel` + `for` shape only when the region genuinely holds work BETWEEN the loops.

## Making a legal directive pay

- `schedule(static)` unless the per-iteration cost genuinely varies: it hands each thread one
  WIDE contiguous span, and a narrow strip breaks the memory stream and gives most of the win
  back. A dynamic schedule buys load balance with per-chunk bookkeeping.
- Keep per-thread partials a cache line apart, or the threads fight over one line (false sharing);
  combine them after the loop.
- `declare simd` on a helper called from the hot loop, else the call is a vectorization barrier.
- `#pragma omp unroll partial(4)` on the INNER loop of a nest you already thread -- never `full`:
  it deletes the loop the worksharing directive above needs.
- Split the construct when the shape demands it: `parallel for` on the outer loop, `simd`
  alone on the unit-stride inner one.

## Clauses

`shared(x)`: one object, all threads -- right for input/output arrays, a RACE when two threads
write one scalar. `private(x)`: own UNINITIALIZED copy -- every scratch scalar the body writes.
`firstprivate` / `lastprivate`: private, initialized from entry / sequentially-last value copied
back out. `reduction(op:x)`: what a sum/max/count wants. Getting sharing wrong is a race,
not a build error. Induction variables
are already private.

## Build errors that cost a turn

- **Skip `default(none)`.** The one variable you miss is always the accumulator -- which belongs
  in `reduction(...)` anyway.
- **Nothing between the directive and its loop, and the loop must be canonical.** One induction
  variable, initialized IN the header, bound known at entry; `for (; i >= 0; i -= 4)` is a build
  error.
- **`simd` is part of the directive NAME**: `parallel for simd schedule(static)`, never
  `parallel for schedule(static) simd`.
- **No `break` / `return` / `throw` out of a threaded loop.** A search loop keeps its trip count
  and reduces instead: `reduction(min:first)` over a per-iteration candidate.
- `nowait` does not exist on a combined `parallel for`; `schedule` is worksharing-only (on a bare
  `simd` it is a build error).
