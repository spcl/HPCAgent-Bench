# loop-transformations-fortran

A nest that resists `parallel do` usually needs its SHAPE changed first. Three rewrites cover
almost all of it, each with a mechanical legality test -- run it, do not guess. Fortran is
COLUMN-major: the FIRST subscript runs innermost.

## Dependence vectors

Two iterations touching the same element: write (later index - earlier index), one component per
loop, outermost first. `a(i,j) = a(i,j-1) + a(i-1,j)` carries `(0,1)` and `(1,0)`. POSITIVE =
first non-zero component is positive. The original nest always is; a rewrite is legal exactly when
every vector still is. A loop is PARALLEL when no vector has a non-zero at its position with all
outer components zero.

## Permutation -- swap two loops

**Legal** when permuting every vector the same way leaves them all positive. 2-deep nest: illegal
exactly when some dependence is `(+,-)`.

**Pays** when it puts the unit-stride axis innermost, or a parallel axis outward. One axis carries
the dependence, the other is free AND unit stride -- take both:

```fortran
do j = 2, n                            ! carries the dependence: serial
  !$omp parallel do simd               ! free and unit stride
  do i = 1, n
    aa(i, j) = aa(i, j - 1) + bb(i, j)
  end do
end do
```

## Distribution -- split one loop into several

**Legal** when statements on a dependence CYCLE stay together; resulting loops run in topological
order of the statement graph.

**Pays** on a body mixing a recurrence with independent work -- fused, the whole loop is serial:

```fortran
do i = 2, n                            ! chain stays serial
  s(i) = s(i - 1) + x(i)
end do
!$omp parallel do simd
do i = 2, n
  y(i) = 2.0d0 * x(i)
end do
```

Costs an extra pass -- a memory-bound body can come out slower. Fusion is the inverse: legal when
no dependence between the bodies reverses, pays when the second re-reads the first.

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

Bounds keep `j = t - i` inside `2..m` while `i` stays in `2..n`. Buys parallelism with locality: a
diagonal strides, and the team re-forks per diagonal. Wins on long diagonals with real work per
point. Skew over TILES, not points, to restore unit stride inside a block and cut synchronisations
to the number of block diagonals.

Cheaper exits first: a dependence in one dimension only needs permutation; a body mixing a chain
with independent statements needs distribution. What is left may already be parallel.

`do concurrent` asserts independence too, so every legality test above applies to it unchanged.
