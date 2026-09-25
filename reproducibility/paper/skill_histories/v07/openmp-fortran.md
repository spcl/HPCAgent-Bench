# openmp-fortran

Grading is MULTI-CORE, baseline SERIAL, `-fopenmp` always on. Never hardcode a thread count --
the grading machine presets `OMP_NUM_THREADS`, so read `omp_get_max_threads()` (needs
`use omp_lib`). Classify the loop first, then thread it: misfiling returns
`correct: false` and costs a round trip.
(`do concurrent` is the other threading spelling -- the lang-fortran page; one spelling per loop,
`!$omp simd` cannot sit on a `do concurrent`.)

## A dependence is a question, not a stop sign

Before threading anything, answer one thing in words: for the array this loop WRITES, is it also
read at another iteration's index -- `i-1`, `j+1`, `idx(i)`? A directive asserts the answer is no.
Assert it wrongly and you get a race: a wrong answer, not a slow one, and the most expensive
mistake available here.

Having named the dependence, you are in one of three cases:

- **Another axis is free.** Thread that one instead. `a(i,j) = a(i,j-1)` carries its dependence
  on `j`, so the directive belongs on `i` -- not on whichever loop is outermost.
- **The statement can be separated.** A body mixing a recurrence with independent work fissions:
  the chain keeps a serial loop, everything else becomes a loop you can thread.
- **Every axis carries one -- which still does not mean serial.** Where `a(i,j)` reads `a(i-1,j)`
  and `a(i,j-1)`, both reads advance `i+j` by one, so no two points sharing `i+j` can depend on
  each other: skew, and the anti-diagonal runs in parallel. This is the case that gets abandoned
  as sequential. `loop-transformations-fortran` has the rewrite and its legality test.

Legality settles whether you MAY thread; profit is separate. If the innermost loop is not the one
walking unit stride -- the first subscript, since Fortran is column-major -- the nest is
bandwidth-bound and threading returns ~1.00x however many cores you use. Interchange first.

**A directive on a recurrence, with the comment saying so.** A directive is an
assertion, not a request: `simd` on a carried dependence claims lanes are independent while
the line above proves they are not:

```fortran
! the recurrence relation requires sequential processing   <- correct diagnosis
!$omp simd                                                 <- contradicts it
do i = 3, n
  a(i) = a(i - 2) + x(i)
end do
```

That stride-2 chain is two independent chains (even `i`, odd `i`): split the index space along an
axis the dependence does not cross, or use the scan form below. If the comment you are about to
write names a dependence, the directive is the wrong one.

## The four bins

**PARALLEL** -- every write lands at this iteration's own subscript, no scalar carries state.
`parallel do simd` on the OUTERMOST such loop: threads across cores, lanes within each.

```fortran
!$omp parallel do simd
do i = 1, n
  a(i) = c(i) * x(i) + d(i)
end do
```

**REDUCTION** -- the only carried state is an accumulator (sum, max, min, count). Same directive
plus `reduction(op:s)`. Never a shared scalar, never hand-built per-thread arrays. The clause also
authorizes the FP reassociation the compiler refuses on its own.

```fortran
s = 0.0d0
!$omp parallel do simd reduction(+:s)
do i = 1, n
  s = s + c(i) * d(i)
end do
```

**RECURRENCE** -- the written array is read at ANOTHER iteration's subscript: `x(i-1)`, in-place
stencil, wavefront. Threading it is WRONG, not slow. Fission the independent statements into their
own threaded loop and keep the chain serial; or thread a dimension the chain does not cross; or
give a stencil a separate output.

```fortran
do i = 2, n
  a(i) = a(i) + c(i) * d(i)   ! independent -- fission out, thread it
  b(i) = b(i-1) + a(i)        ! chain -- stays serial
end do
```

Prefix sum is the one recurrence with a directive; statements before the `scan` feed the sum,
statements after read the scanned value:

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

**SCATTER** -- writes through an index array, `a(idx(i))`. If the task guarantees distinct indices
it is PARALLEL, no atomics. Only DUPLICATE indices collide: then per-thread copies merged after the
loop (usually fastest), or `!$omp atomic` on the update (often slower than serial).

```fortran
subroutine hist(a, bin, n) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(inout) :: a(n)
  integer(c_int64_t), intent(in) :: bin(n)
  integer(c_int64_t) :: i
  !$omp parallel do
  do i = 1, n
    !$omp atomic
    a(bin(i)) = a(bin(i)) + 1.0d0
  end do
end subroutine hist
```

## Clauses

| clause | means |
|---|---|
| `shared(x)` | one object, all threads. Right for input/output arrays. Two threads writing one shared scalar is a RACE. |
| `private(x)` | own UNINITIALIZED copy per thread. Every scratch scalar the body writes. |
| `firstprivate(x)` | private, initialized from the value on entry. |
| `lastprivate(x)` | private, sequentially-last value copied back out. |
| `reduction(op:x)` | per-thread copy at `op`'s identity, combined at the end. What a sum/max/count wants. |

Getting sharing wrong is a RACE, not a build error: it compiles, runs, and returns a different
answer under load. Loop indices are private already.

## Worth one line each

- `collapse(n)` when one loop is too short to fill cores -- exactly n PERFECTLY nested loops,
  nothing between the `do` statements.
- `declare simd` on a helper called from the hot loop, else the call is a vectorization barrier.
- `!$omp unroll partial(4)` on the INNER loop of a nest you already thread -- never `full`:
  it deletes the loop the worksharing directive above needs.
- Split the construct when the shape demands it: `parallel do` on the outer loop, `simd`
  alone on the unit-stride inner one.

## Build errors that cost a turn

- **End the loop with `end do` and write nothing after it.** The closing directive is optional and
  omitting it is always safe; if you write one it must name the SAME construct token for token:
  opening `!$omp parallel do simd` and closing `!$omp end parallel do` drops the `simd` and is a
  BUILD ERROR -- and gfortran blames the closing line.
- **`aligned(...)` is unavailable on ABI dummies** (*must be POINTER, ALLOCATABLE, Cray pointer or
  C_PTR*) -- rejected outright.
- **`!$omp workshare` does NOT thread on gcc** (gfortran lowers it to `single`). Rewrite array
  syntax as an explicit loop under `parallel do`.
- **Skip `default(none)`.** The one variable you miss is always the accumulator -- which belongs
  in `reduction(...)` anyway. Leaving it off removes a whole class of build failure.
- **No `exit` / `cycle` to an outer loop, no `return`, out of a threaded loop.** A search loop
  keeps its trip count and reduces instead: `reduction(min:first)` over a per-iteration candidate.
- `nowait` does not exist on a combined `parallel do`; `schedule` is worksharing-only (on a bare
  `simd` it is a build error).
