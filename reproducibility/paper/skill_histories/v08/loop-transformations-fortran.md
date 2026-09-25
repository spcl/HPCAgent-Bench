# loop-transformations-fortran

A nest that resists `parallel do` usually needs its SHAPE changed first. A few rewrites cover
almost all of it, each with a mechanical legality test -- run it, do not guess. Fortran is
COLUMN-major: the FIRST subscript runs innermost.

## Dependence vectors

Two iterations touching the same element: write (later index - earlier index), one component per
loop, outermost first. `a(i,j) = a(i,j-1) + a(i-1,j)` carries `(0,1)` and `(1,0)`. POSITIVE =
first non-zero component is positive. The original nest always is; a rewrite is legal exactly
when every vector still is. A loop is PARALLEL when no vector has a non-zero at its position with
all outer components zero.

## Permutation -- swap two loops

**Legal** when permuting every vector the same way leaves them all positive. 2-deep nest: illegal
exactly when some dependence is `(+,-)`.

**Pays** when it puts the unit-stride axis innermost, or a parallel axis outward. One axis
carries the dependence, the other is free AND unit stride -- take both:

```fortran
do j = 2, n                            ! carries the dependence: serial
  !$omp parallel do simd               ! free and unit stride
  do i = 1, n
    aa(i, j) = aa(i, j - 1) + bb(i, j)
  end do
end do
```

## Distribution -- split one loop into several

**Legal** when statements on a dependence CYCLE stay together; the resulting loops run in
topological order of the statement graph. **Pays** on a body mixing a recurrence with independent
work: fused, the whole loop is serial; split, the chain keeps a serial loop and everything else
threads. Costs an extra pass over memory -- a bandwidth-bound body can come out slower.

## Wavefront -- every loop carries a dependence

Anti-diagonal iterations are independent: with `(0,1)` and `(1,0)` both advance `i+j` by one, so
no two points sharing `i+j` can depend on each other. Skewing renumbers without reordering, so it
is always legal; it exists to make the following interchange legal.

```fortran
do t = 3, n + m                        ! serial across diagonals
  !$omp parallel do                    ! parallel within one
  do i = max(2, t - m), min(n, t - 2)
    a(i, t - i) = a(i - 1, t - i) + a(i, t - i - 1)
  end do
end do
```

Bounds keep `j = t - i` inside `2..m` while `i` stays in `2..n`. Wins on long diagonals with real
work per point; a diagonal strides, and the team re-forks per diagonal. Skew over TILES, not
points, to restore unit stride inside a block and cut synchronisations to the number of block
diagonals.

## False dependences -- rewrite, no directive needed

Not every carried value is a dependence. Three shapes LOOK serial and are not; each dies to a
rewrite, after which the loop files under PARALLEL.

- **Rotated scalar.** A scalar saved only so the next iteration can read it is the previous
  iteration's expression by another name -- substitute it away (twice for a two-deep carry):

```fortran
do i = 1, n
  t = p(i) + q(i)
  r(i) = t - carry           ! carry == p(i-1) + q(i-1)
  carry = t
end do
! becomes (i == 1 peeled to use the entry value of carry)
!$omp parallel do simd
do i = 2, n
  r(i) = (p(i) + q(i)) - (p(i-1) + q(i-1))
end do
```

- **Read of a future element** (`x(i+1)` on the right while `x(i)` is written): the read means
  the ORIGINAL value, so keep a copy of the input, or write to a fresh output array -- either way
  the loop is parallel. Renaming costs one pass of memory traffic; score it.
- **Write-only clobber**: a scalar or element every iteration overwrites before reading is
  `private`/`lastprivate` scratch, not a dependence.

## Fusion and unswitching

**Fusion** merges two loops into one body. Legal when no dependence between the bodies reverses
direction; pays when the second loop re-reads what the first wrote: one pass over memory instead
of two, and a temporary array that existed only to connect them becomes a scalar in registers.
**Unswitching** hoists a LOOP-INVARIANT condition out: `if (scale > 0)` tested every iteration
becomes two loops behind one test, each a clean vectorizable body. Only for invariant guards -- a
data-dependent `if (c(i) > 0)` stays inside and becomes arithmetic (`merge`), not a branch. The
two combine: guarded loops often fuse only AFTER their invariant guards move out.

Cheaper exits first: a FALSE dependence needs only its rewrite; one real dimension needs
permutation; a body mixing a chain with independent statements needs distribution. What is left
may already be parallel.

`do concurrent` asserts independence too, so every legality test above applies to it unchanged.
