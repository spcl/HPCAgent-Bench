---
name: openmp-fortran
description: "OpenMP in Fortran: the four loop bins, the sharing clauses, and the build errors
that cost a turn."
---

# openmp-fortran

Grading is MULTI-CORE, baseline SERIAL, `-fopenmp` always on. Never hardcode a thread count --
the grading machine presets `OMP_NUM_THREADS`; read `omp_get_max_threads()` (`use omp_lib`).
(`do concurrent` is the other threading spelling -- the lang-fortran page; one spelling per
loop, and `!$omp simd` cannot sit on a `do concurrent`.)

## A directive is an assertion, not a request

`parallel do` claims the iterations may run in any order. `simd` claims the lanes may. Nothing
checks either claim; both are believed. Assert wrongly and you get a RACE -- a wrong answer, not a
slow one, and one that looks plausible.

So derive the dependence vectors FIRST and name the axis each one crosses (lang-fortran has the
test, the rewrites that free an axis, and the stride question that decides whether threading pays
at all). That axis may not be threaded; any axis it does not cross may. Misfiling returns
`correct: false` and costs a round trip.

If the comment you are about to write names a dependence, the directive is wrong. The tell:

```fortran
! the recurrence relation requires sequential processing   <- correct diagnosis
!$omp simd                                                 <- contradicts it
do i = 3, n
  a(i) = a(i - 2) + x(i)
end do
```

## The four bins

**PARALLEL** -- every write lands at this iteration's own subscript, no scalar carries state.
`!$omp parallel do simd` on the OUTERMOST such loop: threads across cores, lanes within each. A
tiny trip count loses to the cost of forking the team.

**REDUCTION** -- the only carried state is an accumulator (sum, max, min, count). Same directive
plus `reduction(+:s)` (or `max:`, `min:`). Never a shared scalar, never hand-built per-thread
arrays. The clause also authorizes the FP reassociation the compiler refuses on its own, and the
graded tolerance covers it.

**Max or min WITH ITS INDEX** -- `reduction(max:m)` returns the value and LOSES the position, and
declaring a pair reduction needs a derived type. Two passes, both parallel, no new syntax; break
ties toward the SMALLER index or the answer disagrees with a serial sweep:

```fortran
!$omp parallel do reduction(max:m)
do i = 1, n
  m = max(m, v(i))
end do
first = n + 1
!$omp parallel do reduction(min:first)
do i = 1, n
  if (v(i) == m .and. i < first) first = i   ! exact compare: same stored value
end do
```

Serial or inside `simd`, `maxloc(v)` is one line -- score it against the two-pass form before
assuming threads win.

**RECURRENCE** -- the written array is read at ANOTHER iteration's subscript. The loop carrying
the chain stays serial; threading it races. That does not make the NEST serial: thread an axis the
chain does not cross, or reshape the nest so there is one (lang-fortran). Prefix sum is the one
recurrence with a directive of its own; statements before the `scan` feed the sum, statements
after read the scanned value:

```fortran
s = 0.0d0
!$omp parallel do simd reduction(inscan, +:s)
do i = 1, n
  s = s + c(i)
  !$omp scan inclusive(s)
  x(i) = s
end do
```

`exclusive(s)` is the value-before-this-iteration variant. Scans reassociate; tolerance applies.

**The clause is gfortran-only.** flang rejects it outright -- *not yet implemented: Unhandled
clause reduction with modifier* -- so if you select the LLVM compiler, write the two passes
yourself: every thread sums its own chunk, one thread prefix-sums the per-chunk totals, then each
thread re-walks its chunk starting from that offset. Same answer as the serial sweep, both
compilers:

```fortran
nt = omp_get_max_threads()               ! part(0:nt), zeroed
!$omp parallel private(t, lo, hi, i, run)
t = omp_get_thread_num()
lo = (n * t) / nt + 1
hi = (n * (t + 1)) / nt
run = 0.0d0
do i = lo, hi
  run = run + c(i)
end do
part(t + 1) = run
!$omp barrier
!$omp single
do i = 1, nt
  part(i) = part(i) + part(i - 1)
end do
!$omp end single
run = part(t)
do i = lo, hi
  run = run + c(i)
  x(i) = run
end do
!$omp end parallel
```

**SCATTER** -- writes through an index array, `a(idx(i))`. If the task guarantees distinct
indices it is PARALLEL, no atomics. Only DUPLICATE indices collide: then per-thread copies merged
after the loop (usually fastest), or `!$omp atomic` on the update (often slower than serial).

## Fork and barrier cost

A worksharing construct ends with an IMPLICIT BARRIER. So `!$omp parallel` wrapped around an outer
sequential loop with `!$omp do` inside pays one barrier per outer iteration: every thread waits for
the slowest share of the row before any of them starts the next one, and that wait is charged as
many times as the outer loop runs.

- Default spelling: ONE combined `parallel do` over the whole iteration space -- the team forks
  once and synchronizes once. `collapse(n)` when the outer loop alone is too short to fill the
  cores: exactly n PERFECTLY nested loops, nothing between the `do` statements.
- `nowait` on the inner `do` when what follows does not read what it wrote. Reach for the
  `parallel` + `do` shape only when the region genuinely holds work BETWEEN the loops.

## Making a legal directive pay

- `schedule(static)` unless the per-iteration cost genuinely varies: it hands each thread one
  WIDE contiguous span, and a narrow strip breaks the memory stream and gives most of the win
  back. A dynamic schedule buys load balance with per-chunk bookkeeping.
- Keep per-thread partials a cache line apart, or the threads fight over one line (false sharing);
  combine them after the loop.
- `declare simd` on a helper called from the hot loop, else the call is a vectorization barrier.
- `!$omp unroll partial(4)` on the INNER loop of a nest you already thread -- never `full`:
  it deletes the loop the worksharing directive above needs.
- Split the construct when the shape demands it: `parallel do` on the outer loop, `simd`
  alone on the unit-stride inner one.

## Clauses

`shared(x)`: one object, all threads -- right for input/output arrays, a RACE when two threads
write one scalar. `private(x)`: own UNINITIALIZED copy -- every scratch scalar the body writes.
`firstprivate` / `lastprivate`: private, initialized from entry / sequentially-last value copied
back out. `reduction(op:x)`: what a sum/max/count wants. Getting sharing wrong is a race,
not a build error. Loop indices are
private already.

## Build errors that cost a turn

- **End the loop with `end do` and write nothing after it.** The closing directive is optional
  and omitting it is always safe; if you write one it must name the SAME construct token for
  token: opening `!$omp parallel do simd` and closing `!$omp end parallel do` drops the `simd`
  and is a BUILD ERROR -- and gfortran blames the closing line.
- **`aligned(...)` is unavailable on ABI dummies** (*must be POINTER, ALLOCATABLE, Cray pointer
  or C_PTR*) -- rejected outright.
- **`!$omp workshare` does NOT thread on gcc** (gfortran lowers it to `single`). Rewrite array
  syntax as an explicit loop under `parallel do`.
- **Skip `default(none)`.** The one variable you miss is always the accumulator -- which belongs
  in `reduction(...)` anyway.
- **No `exit` / `cycle` to an outer loop, no `return`, out of a threaded loop.** A search loop
  keeps its trip count and reduces instead: `reduction(min:first)` over a per-iteration
  candidate.
- `nowait` does not exist on a combined `parallel do`; `schedule` is worksharing-only (on a bare
  `simd` it is a build error).
