# openmp-fortran

Grading is MULTI-CORE, baseline SERIAL, `-fopenmp` always on. Never hardcode a thread count --
the grading machine presets `OMP_NUM_THREADS`; read `omp_get_max_threads()` (`use omp_lib`).
Classify the loop first, then thread it: misfiling returns `correct: false` and costs a round
trip. (`do concurrent` is the other threading spelling -- the lang-fortran page; one spelling per
loop, and `!$omp simd` cannot sit on a `do concurrent`.)

## A dependence is a question, not a stop sign

For the array this loop WRITES, is it also read at another iteration's index -- `i-1`, `j+1`,
`idx(i)`? A directive asserts the answer is no; assert it wrongly and you get a race -- a wrong
answer, not a slow one. Having named the dependence, you are in one of four cases:

- **Another axis is free.** Thread that one. `a(i,j) = a(i,j-1)` carries on `j`, so the directive
  belongs on `i` -- not on whichever loop is outermost.
- **The statements can be separated.** A body mixing a recurrence with independent work fissions:
  the chain keeps a serial loop, the rest becomes a loop you can thread.
- **Every axis carries one -- still not serial.** Where `a(i,j)` reads `a(i-1,j)` and `a(i,j-1)`,
  no two points sharing `i+j` can depend on each other: skew, and the anti-diagonal runs in
  parallel. This is the case that gets abandoned as sequential;
  `loop-transformations-fortran` has the rewrite and its legality test.
- **The dependence is FALSE.** A scalar carried only to hand the next iteration a value it could
  recompute is not a dependence -- substitute it away and the loop is parallel. Reading a FUTURE
  element (`x(i+1)`) means the ORIGINAL value: keep a copy of the input (or write a fresh output)
  and it is gone. Rewrites on the `loop-transformations-fortran` page.

Legality settles whether you MAY thread; profit is separate. If the innermost loop is not the one
walking unit stride -- the FIRST subscript, since Fortran is column-major -- the nest is
bandwidth-bound and threading returns ~1.00x however many cores you use. Interchange FIRST -- the
right order run on ONE core beats the wrong order run on every core, so a directive is not by
itself progress -- then thread, and keep it only if it actually pays.

If the comment you are about to write names a dependence, the directive is wrong. The tell:

```fortran
! the recurrence relation requires sequential processing   <- correct diagnosis
!$omp simd                                                 <- contradicts it
do i = 3, n
  a(i) = a(i - 2) + x(i)
end do
```

That stride-2 chain is two independent chains (even/odd `i`): split along an axis the dependence
does not cross, or use the scan form below.

## The four bins

**PARALLEL** -- every write lands at this iteration's own subscript, no scalar carries state.
`!$omp parallel do simd` on the OUTERMOST such loop: threads across cores, lanes within each.

**REDUCTION** -- the only carried state is an accumulator (sum, max, min, count). Same directive
plus `reduction(+:s)` (or `max:`, `min:`). Never a shared scalar, never hand-built per-thread
arrays. The clause also authorizes the FP reassociation the compiler refuses on its own.

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

**RECURRENCE** -- the written array is read at ANOTHER iteration's subscript: `x(i-1)`, in-place
stencil, wavefront. Threading the chain is WRONG, not slow. Fission independent statements into
their own threaded loop; give a stencil a separate output; or work a dimension the chain does not
cross -- and that last one is where this bin gets left unfinished. Ask which subscript is
fastest-varying -- in Fortran the FIRST. If the chain is the INNER loop it strides across memory,
and the ORDER is the bug before the parallelism is: swap the loops so the chain runs outermost and the
unit-stride axis inner. That swap alone, on ONE core, beats threading the unswapped order several
times over. THEN thread the free axis, giving each thread a WIDE contiguous band of it -- a narrow
strip breaks the stream and gives most of the win back -- which is worth several times the swap
again. Threading the chain itself races: it prints a plausible answer and is still wrong.

```fortran
do k = 2, n                            ! the chain: outermost, SERIAL, never threaded
  do i = 1, n                          ! unit stride: vectorizes, and is the axis to thread
    u(i, k) = u(i, k - 1) + w(i, k)
  end do
end do
```

Prefix sum is the one recurrence with a directive; statements before the `scan` feed the
sum, statements after read the scanned value:

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

## Clauses

`shared(x)`: one object, all threads -- right for input/output arrays, a RACE when two threads
write one scalar. `private(x)`: own UNINITIALIZED copy -- every scratch scalar the body writes.
`firstprivate` / `lastprivate`: private, initialized from entry / sequentially-last value copied
back out. `reduction(op:x)`: what a sum/max/count wants. Getting sharing wrong is a RACE, not a
build error: it compiles, runs, and returns a different answer under load. Loop indices are
private already.

## Worth one line each

- `collapse(n)` when one loop is too short to fill cores -- exactly n PERFECTLY nested loops,
  nothing between the `do` statements.
- `declare simd` on a helper called from the hot loop, else the call is a vectorization barrier.
- `!$omp unroll partial(4)` on the INNER loop of a nest you already thread -- never `full`:
  it deletes the loop the worksharing directive above needs.
- Split the construct when the shape demands it: `parallel do` on the outer loop, `simd`
  alone on the unit-stride inner one.

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
- Split a combined construct and drop the closing barrier with `nowait` when the next loop does
  not read what this one wrote; keep the barrier when it does.
- `nowait` does not exist on a combined `parallel do`; `schedule` is worksharing-only (on a bare
  `simd` it is a build error).
