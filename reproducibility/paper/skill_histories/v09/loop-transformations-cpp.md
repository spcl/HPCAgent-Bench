# loop-transformations-cpp

A nest that resists `parallel for` usually needs its SHAPE changed first. A few rewrites cover
almost all of it, each with a mechanical legality test -- run it, do not guess. C++ is row-major:
the LAST subscript runs innermost.

## Dependence vectors

Two iterations touching the same element: write (later index - earlier index), one component per
loop, outermost first. `a[i][j] = a[i-1][j] + a[i][j-1]` carries `(1,0)` and `(0,1)`. POSITIVE =
first non-zero component is positive. The original nest always is; a rewrite is legal exactly
when every vector still is. A loop is PARALLEL when no vector has a non-zero at its position with
all outer components zero.

## Permutation -- swap two loops

**Legal** when permuting every vector the same way leaves them all positive. 2-deep nest: illegal
exactly when some dependence is `(+,-)`.

**Pays** when it puts the unit-stride axis innermost, or a parallel axis outward. One axis
carries the dependence, the other is free AND unit stride -- take both:

```cpp
for (std::int64_t j = 1; j < n; j++)       // carries the dependence: serial
    #pragma omp parallel for simd          // free and unit stride
    for (std::int64_t i = 0; i < n; i++)
        u[j*n + i] = u[(j-1)*n + i] + w[j*n + i];
```

## Distribution -- split one loop into several

**Legal** when statements on a dependence CYCLE stay together; the resulting loops run in
topological order of the statement graph. **Pays** on a body mixing a recurrence with independent
work: fused, the whole loop is serial; split, the chain keeps a serial loop and everything else
threads. Costs an extra pass over memory -- a bandwidth-bound body can come out slower.

## Wavefront -- every loop carries a dependence

Anti-diagonal iterations are independent: with `(1,0)` and `(0,1)` both advance `i+j` by one, so
no two points sharing `i+j` can depend on each other. Skewing renumbers without reordering, so it
is always legal; it exists to make the following interchange legal.

```cpp
for (std::int64_t t = 2; t <= (n-1) + (m-1); t++) {     // serial across diagonals
    std::int64_t lo = t - (m-1) > 1 ? t - (m-1) : 1;
    std::int64_t hi = t - 1 < n-1 ? t - 1 : n-1;
    #pragma omp parallel for                            // parallel within one
    for (std::int64_t i = lo; i <= hi; i++)
        a[i*m + (t-i)] = a[(i-1)*m + (t-i)] + a[i*m + (t-i-1)];
}
```

Bounds keep `j = t - i` inside `1..m-1` while `i` stays in `1..n-1`. Wins on long diagonals with
real work per point; a diagonal strides, and the team re-forks per diagonal. Skew over TILES, not
points, to restore unit stride inside a block and cut synchronisations to the number of block
diagonals.

## False dependences -- rewrite, no directive needed

Not every carried value is a dependence. Three shapes LOOK serial and are not; each dies to a
rewrite, after which the loop files under PARALLEL.

- **Rotated scalar.** A scalar saved only so the next iteration can read it is the previous
  iteration's expression by another name -- substitute it away (twice for a two-deep carry):

```cpp
for (std::int64_t i = 0; i < n; i++) {
    double t = p[i] + q[i];
    r[i] = t - carry;          // carry == p[i-1] + q[i-1]
    carry = t;
}
// becomes (i == 0 peeled to use the entry value of carry)
#pragma omp parallel for simd
for (std::int64_t i = 1; i < n; i++)
    r[i] = (p[i] + q[i]) - (p[i-1] + q[i-1]);
```

- **Read of a future element** (`x[i+1]` on the right while `x[i]` is written): the read means
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
data-dependent `if (c[i] > 0)` stays inside and becomes arithmetic (select/blend), not a branch.
The two combine: guarded loops often fuse only AFTER their invariant guards move out.

Cheaper exits first: a FALSE dependence needs only its rewrite; one real dimension needs
permutation; a body mixing a chain with independent statements needs distribution. What is left
may already be parallel.
