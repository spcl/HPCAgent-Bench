# loop-transformations-c

A nest that resists `parallel for` usually needs its SHAPE changed first. Three rewrites cover
almost all of it, each with a mechanical legality test -- run it, do not guess. C is row-major:
the LAST subscript runs innermost.

## Dependence vectors

Two iterations touching the same element: write (later index - earlier index), one component per
loop, outermost first. `a[i][j] = a[i-1][j] + a[i][j-1]` carries `(1,0)` and `(0,1)`. POSITIVE =
first non-zero component is positive. The original nest always is; a rewrite is legal exactly when
every vector still is. A loop is PARALLEL when no vector has a non-zero at its position with all
outer components zero.

## Permutation -- swap two loops

**Legal** when permuting every vector the same way leaves them all positive. 2-deep nest: illegal
exactly when some dependence is `(+,-)`.

**Pays** when it puts the unit-stride axis innermost, or a parallel axis outward. One axis carries
the dependence, the other is free AND unit stride -- take both:

```c
for (int64_t j = 1; j < n; j++)            /* carries the dependence: serial */
    #pragma omp parallel for simd          /* free and unit stride */
    for (int64_t i = 0; i < n; i++)
        aa[j*n + i] = aa[(j-1)*n + i] + bb[j*n + i];
```

## Distribution -- split one loop into several

**Legal** when statements on a dependence CYCLE stay together; resulting loops run in topological
order of the statement graph.

**Pays** on a body mixing a recurrence with independent work -- fused, the whole loop is serial:

```c
for (int64_t i = 1; i < n; i++) { s[i] = s[i-1] + x[i];  y[i] = 2.0 * x[i]; }
/* becomes */
for (int64_t i = 1; i < n; i++) s[i] = s[i-1] + x[i];    /* chain stays serial */
#pragma omp parallel for simd
for (int64_t i = 1; i < n; i++) y[i] = 2.0 * x[i];
```

Costs an extra pass -- a memory-bound body can come out slower. Fusion is the inverse: legal when
no dependence between the bodies reverses, pays when the second re-reads the first.

## Wavefront -- every loop carries a dependence

Anti-diagonal iterations are independent: with `(1,0)` and `(0,1)` both advance `i+j` by one, so
no two points sharing `i+j` can depend on each other. Skewing renumbers without reordering, so it
is always legal; it exists to make the following interchange legal.

```c
for (int64_t t = 2; t <= (n-1) + (m-1); t++) {          /* serial across diagonals */
    int64_t lo = t - (m-1) > 1 ? t - (m-1) : 1;
    int64_t hi = t - 1 < n-1 ? t - 1 : n-1;
    #pragma omp parallel for                            /* parallel within one */
    for (int64_t i = lo; i <= hi; i++)
        a[i*m + (t-i)] = a[(i-1)*m + (t-i)] + a[i*m + (t-i-1)];
}
```

Bounds keep `j = t - i` inside `1..m-1` while `i` stays in `1..n-1`. Buys parallelism with
locality: a diagonal strides, and the team re-forks per diagonal. Wins on long diagonals with real
work per point. Skew over TILES, not points, to restore unit stride inside a block and cut
synchronisations to the number of block diagonals.

Cheaper exits first: a dependence in one dimension only needs permutation; a body mixing a chain
with independent statements needs distribution. What is left may already be parallel.
