# loop-transformations-c

A nest that resists `parallel for` usually needs its SHAPE changed first. The main prompt's hints
say which rewrites exist and when each tends to pay. This page is the LEGALITY half: a mechanical
test per rewrite, so you check rather than guess. C is row-major -- the LAST subscript runs
innermost.

## Dependence vectors -- the test the rest of the page uses

Take two iterations that touch the same element, at least one of them writing. Write
(later index - earlier index), one component per loop, outermost first. In
`a[i][j] = a[i-1][j] + a[i][j-1]` the two reads give `(1,0)` and `(0,1)`.

A vector is POSITIVE when its first non-zero component is positive. Every vector of the ORIGINAL
nest is positive -- that is what "the original order is correct" means. A rewrite is LEGAL exactly
when every vector is still positive after it. A loop is PARALLEL when no vector has a non-zero at
that loop's position with all outer components zero.

Derive the vectors once, then read every question below off them.

## Permutation -- swap two loops

Permute every vector the same way and check they are all still positive. In a 2-deep nest that
reduces to: illegal exactly when some dependence is `(+,-)`.

## Distribution -- split one loop into several

Build the statement graph of the body (an edge where one statement depends on another). Statements
on a dependence CYCLE must stay in the same loop; everything else may split off, and the resulting
loops run in topological order of that graph.

Fusion is the inverse: legal when no dependence between the two bodies reverses direction, which
means fusing must not require a statement to read a value the merged loop has not written yet.

## Unswitching -- hoist a guard

Legal only when the condition is LOOP-INVARIANT: it reads nothing the loop writes and no induction
variable. A data-dependent guard cannot be hoisted; it stays inside and becomes arithmetic
(select), not a branch.

## Skewing -- when every axis carries a dependence

Every axis carrying a dependence does NOT mean the nest is serial. Renumber the iteration space by
a combination of the indices -- typically one new axis counting `i+j` -- so that iterations sharing
a value of the new outer index cannot depend on each other. Skewing renumbers without reordering,
so it is ALWAYS legal; it exists to make the interchange after it legal. Apply the permutation test
above to the skewed vectors: that is what decides whether the parallel axis really ends up inside.

Derive the new bounds from the old ones rather than guessing: each old index still has to stay
within its own original range after the substitution. Skew over TILES rather than points to keep
unit stride inside a block and to cut the number of synchronisations to the number of block
diagonals.

## Dependences that are not real

Three shapes look serial and are not. Each dies to a rewrite, not to a directive, and after the
rewrite the loop is simply parallel.

- A scalar written every iteration and read by the next carries no information the loop cannot
  recompute: it holds the previous iteration's expression under another name. Substitute the
  expression back in and the carry is gone. Peel the first iteration, which reads the value from
  before the loop. This works through two levels of carry as well -- substitute twice.
- A read of an element the loop writes LATER is an anti-dependence: the read means the ORIGINAL
  value. Read from a copy of the input, or write into a fresh output array, and the loop is
  parallel. Costs one pass of memory traffic, so score it.
- A scalar or element every iteration overwrites before reading is scratch, not state: it is
  `private` (or `lastprivate` if a later reader needs the final value).

## Order of attack

Cheapest exit first. A false dependence needs only its rewrite. A nest with one real dependence
direction needs a permutation. A body mixing a chain with independent statements needs a
distribution. Whatever is left may already be parallel. Reach for skewing only when the first three
leave every axis carrying something.
