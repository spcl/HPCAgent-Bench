# Canonical NumPy Form (CNF)

A specification for authoring OptArena kernels that are guaranteed to lower through
the `optarena/NumpyToC/` translator (and its C++/Fortran siblings).

---

## 1. Motivation

The NumpyToC translator turns `*_numpy.py` kernels into C/C++/Fortran. Historically
it chased *arbitrary* NumPy idioms — chained subscripts, rank-changing reshapes,
fancy indexing, whole-array reassignment — through roughly two dozen interacting
AST rewriter passes in `lowering.py`. Those passes share a fragile, mutable
`shape_table`, and when one pass rewrites a statement in a way another pass did not
anticipate, the shape table goes stale and emission produces wrong or
non-compiling code. **Canonical NumPy Form (CNF) inverts the contract.** Instead of
the translator bending to fit any kernel, we define a single, small NumPy subset
that is *provably* lowerable, and we rewrite kernels into it. The payoff is a
translator that can *delete* its riskiest passes (see §5), plus benchmark authors
who get a clear, mechanical rulebook (§4) and a CI gate (§6) instead of a guessing
game.

CNF rests on **three invariants**. They are decided. This document explains them,
shows canonical-vs-non-canonical pairs from real kernels, and gives a rewrite
cookbook.

---

## 2. The Three Invariants

### Invariant 1 — Static shape, known at declaration

Every array — input *or* temporary — has a shape fully determined by the kernel's
integer parameters at the point it first appears. No later statement changes an
array's rank or shape. If you need a different shape, declare a **new named
buffer**.

**Non-canonical** — `lenet_numpy.py` reassigns `x` with a different shape on nearly
every line:

```python
def lenet5(input, conv1, ..., N, C_before_fc1):
    x = relu(conv2d(input, conv1) + conv1bias)   # 4-D
    x = maxpool2d(x)                              # 4-D, smaller
    x = relu(conv2d(x, conv2) + conv2bias)        # 4-D
    x = maxpool2d(x)                              # 4-D
    x = np.reshape(x, (N, C_before_fc1))          # 2-D  <-- rank change
    x = relu(x @ fc1w + fc1b)                     # 2-D, new width
    ...
```

The shape table entry for `x` is rewritten five times; the rank-2 reshape on line 49
silently invalidates everything the slice/transpose passes recorded about the
4-D `x`.

**Canonical** — one named buffer per distinct shape:

```python
def lenet5(input, conv1, ..., N, C_before_fc1):
    c1 = np.empty((N, H1, W1, Cc1), dtype=input.dtype)   # conv1 output
    p1 = np.empty((N, H1 // 2, W1 // 2, Cc1), dtype=input.dtype)
    c2 = np.empty((N, H2, W2, Cc2), dtype=input.dtype)
    p2 = np.empty((N, H2 // 2, W2 // 2, Cc2), dtype=input.dtype)
    flat = np.empty((N, C_before_fc1), dtype=input.dtype)
    h1 = np.empty((N, fc1w.shape[1]), dtype=input.dtype)
    ...
    c1[:] = relu(conv2d(input, conv1) + conv1bias)
    p1[:] = maxpool2d(c1)
    ...
    flat[:] = np.reshape(p2, (N, C_before_fc1))   # reshape feeds a fresh buffer
```

Each buffer has exactly one shape for its whole lifetime; the table never goes
stale.

### Invariant 2 — Explicit indexing (no chained or fancy subscripts in compute)

Index arrays with **scalars or slices over their declared axes**. A row of a 3-D
array is taken with a *full* index, never a chained/partial one. No fancy indexing
(`a[index_array]`) in compute — that routes only through the sparse layout system.

**Non-canonical** — `contour_integral_numpy.py`, line 13. `Ham` is 3-D, but `Ham[n]`
takes a 2-D slab with a chained (rank-reducing) subscript that then participates in
an array add:

```python
Tz = np.zeros((NR, NR), dtype=np.complex128)
for n in range(slab_per_bc + 1):
    zz = np.power(z, slab_per_bc / 2 - n)
    Tz += zz * Ham[n]            # Ham[n] is a 2-D slab of a 3-D array
```

`_emit_subscript` has to *infer* that `Ham[n]` is a `[n, :, :]` slab and re-expand
it; the "chained `[][]` fallback" exists exactly for this and is a known source of
wrong indices.

**Canonical** — index every axis explicitly in a loop nest:

```python
Tz = np.zeros((NR, NR), dtype=np.complex128)
for n in range(slab_per_bc + 1):
    zz = np.power(z, slab_per_bc / 2 - n)
    for i in range(NR):
        for j in range(NR):
            Tz[i, j] += zz * Ham[n, i, j]   # every axis named
```

(Or, if you keep array-level ops: `Tz[:, :] += zz * Ham[n, :, :]` — the point is the
`n` axis is written, not chained.)

Contrast with the *already-canonical* `gemm_numpy.py`, which slices every axis it
touches:

```python
def kernel(alpha, beta, C, A, B):
    C[:] = alpha * A @ B + beta * C    # whole-array slice assign, no chaining
```

### Invariant 3 — Declare-then-fill, never grow

Temporaries are created at first use with a static shape via
`np.zeros / np.empty / np.ones((static_shape), dtype=)`, then written by index or
`[:]` slice assignment. No `append` / `concatenate` of varying length, no Python
`list` / `dict` / `set`, no dynamic growth.

**Non-canonical** (illustrative — the pattern CNF forbids):

```python
rows = []
for i in range(M):
    rows.append(compute_row(i))      # length grows at runtime
out = np.array(rows)                 # shape only known after the loop
```

**Canonical** — pre-declare the worst-case buffer, then fill by index:

```python
out = np.empty((M, K), dtype=np.float64)   # shape known up front
for i in range(M):
    out[i, :] = compute_row(i)
```

The already-canonical `jacobi_2d_numpy.py` is the model: every buffer is an input or
declared, and updates are pure slice assignments —

```python
for t in range(1, TSTEPS):
    B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                           A[2:, 1:-1] + A[:-2, 1:-1])
    A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + ...)
```

---

## 3. The Canonical Compute Vocabulary

Everything a CNF kernel may do. If it is not here, rewrite it (see §4) or it is out.

| Category | Allowed (IN) | Not allowed (OUT) |
| --- | --- | --- |
| Control flow | `for i in range(...)`, nested `for`, `while`, `if/else` | `for x in array:` (iterate values), comprehensions, generators |
| Array declaration | `np.zeros`, `np.empty`, `np.ones` with a static-shape tuple + `dtype=` | declaring with a runtime-computed/append-derived shape |
| Element access | scalar index `A[i, j]`, full-rank index `A[n, i, j]` | chained/partial index `A[n]` for rank>1 (Inv. 2) |
| Slicing | slices over declared axes `A[1:-1, :]`, `A[:, k]` | slices that drop into an undeclared rank |
| Assignment | `A[i, j] = e`, `A[:] = e`, `A[i, :] = e`, `+=`/`-=`/`*=`/`/=` aug-assign | chained assign `a = b = e`, whole-array *reassign with new shape* (Inv. 1) |
| Elementwise | `+ - * / **`, `np.exp`, `np.sqrt`, `np.power`, `np.abs`, `np.sin`, `np.cos`, `np.log`, `np.maximum`, `np.minimum`, `np.sign`, `np.tanh` | arbitrary ufuncs not on this list (add deliberately) |
| Reductions | `np.sum`, `np.max`, `np.min`, `np.mean`, `np.prod` with explicit `axis=` writing a declared buffer | reductions whose output shape feeds a *reassigned* variable |
| Linear algebra | `@` / `np.matmul`, `np.dot` | `np.linalg.inv` / `solve` etc. (out unless backed by a lib node) |
| Conditional select | `np.where(mask, a, b)`, scalar `if` | boolean-mask *indexing* `a[a > 0]` (use `np.where`) |
| Constants/scalars | `np.pi`, complex literals (`2.0j`), scalar math | — |
| Sparse layout | the CSR/COO gather forms recognised by `sparse_emit.py` / `validate_sparse.py` | ad-hoc fancy gather `vals @ x[cols]` outside that system |
| Transpose/reshape | only when feeding a **fresh declared buffer** of the target shape | in-place rank change of a live array (Inv. 1) |

`np.newaxis`, `np.mgrid`, `np.repeat`, `np.concatenate`, `np.append`, `.T` *inside an
expression*, list/dict/set literals, and `np.array([...])` of Python lists are all
**OUT** — each has a mechanical rewrite below.

---

## 4. Rewrite Cookbook

Mechanical Before/After transforms. Apply these to bring a failing kernel into CNF.

### 4.1 Chained subscript `A[n]` (rank>1) → full index

**Why:** a partial index leaves the translator to infer the dropped axes; spelling
them removes the guess.

```python
# Before  (contour_integral)
Tz += zz * Ham[n]                 # Ham is 3-D
```
```python
# After
for i in range(NR):
    for j in range(NR):
        Tz[i, j] += zz * Ham[n, i, j]
```

### 4.2 Reshape with rank change → declared copy with explicit index arithmetic

**Why:** rank-changing reshape invalidates the shape table; a fresh buffer plus
explicit flat-index math keeps shapes stable. (`stockham_fft` is the canonical
offender — `np.reshape(y, (R**i, R, ...))`, `np.reshape(D, (N,))`, etc.)

```python
# Before  (stockham_fft, twiddle build)
D = np.empty((R, R**i, R**(K - i - 1)), dtype=np.complex128)
D[:] = np.repeat(np.reshape(tmp, (R, R**i, 1)), R**(K - i - 1), axis=2)
tmp_twid = np.reshape(tmp_perm, (N,)) * np.reshape(D, (N,))
```
```python
# After  — keep D 3-D, write a separate flat buffer with index arithmetic
D = np.empty((R, R**i, R**(K - i - 1)), dtype=np.complex128)
for a in range(R):
    for b in range(R**i):
        for c in range(R**(K - i - 1)):
            D[a, b, c] = tmp[a, b]           # the "repeat" along axis 2

twid = np.empty((N,), dtype=np.complex128)   # declared flat buffer
for a in range(R):
    for b in range(R**i):
        for c in range(R**(K - i - 1)):
            flat = (a * R**i + b) * R**(K - i - 1) + c   # explicit C-order index
            twid[flat] = perm_flat[flat] * D[a, b, c]
```

The general rule: a rank-changing reshape becomes *(1)* a declared buffer of the
target shape and *(2)* a loop that copies with the explicit C-order flat-index
formula `((i0)*n1 + i1)*n2 + i2 ...`.

### 4.3 `np.mgrid` → explicit index fill

**Why:** `mgrid` materialises coordinate arrays implicitly; CNF wants the loop that
consumes them.

```python
# Before  (stockham_fft)
i_coord, j_coord = np.mgrid[0:R, 0:R]
dft_mat = np.exp(-2.0j * np.pi * i_coord * j_coord / R)
```
```python
# After
dft_mat = np.empty((R, R), dtype=np.complex128)
for i in range(R):
    for j in range(R):
        dft_mat[i, j] = np.exp(-2.0j * np.pi * i * j / R)
```

### 4.4 Whole-array reassign with shape change → named buffers

**Why:** see Invariant 1. One name = one shape.

```python
# Before  (lenet5)
x = maxpool2d(c1)
x = np.reshape(x, (N, C_before_fc1))
```
```python
# After
p1 = np.empty((N, H1 // 2, W1 // 2, Cc1), dtype=input.dtype)
flat = np.empty((N, C_before_fc1), dtype=input.dtype)
p1[:] = maxpool2d(c1)
flat[:] = np.reshape(p1, (N, C_before_fc1))   # reshape into a fresh buffer
```

### 4.5 Fancy index `a[idx]` → explicit gather loop (or sparse layout)

**Why:** fancy/gather indexing has no general lowering; spell the gather, or route
through the sparse system.

```python
# Before  (spmv)
for i in range(A_row.size - 1):
    cols = A_col[A_row[i]:A_row[i + 1]]
    vals = A_val[A_row[i]:A_row[i + 1]]
    y[i] = vals @ x[cols]            # x[cols] is a fancy gather
```
```python
# After — explicit gather loop (CSR walk)
for i in range(A_row.size - 1):
    acc = 0.0
    for k in range(A_row[i], A_row[i + 1]):
        acc += A_val[k] * x[A_col[k]]   # one scalar gather per nnz
    y[i] = acc
```

This CSR form is exactly what `sparse_emit.py` recognises; SpMV-class kernels should
be authored in this explicit-gather shape rather than `vals @ x[cols]`.

### 4.6 `x.T` in an expression → transposed access or named buffer

**Why:** an inline transpose forces the emitter to track a virtual axis swap; make
it concrete.

```python
# Before
y = A @ B.T
```
```python
# After (option A — index with axes swapped)
y = np.empty((A.shape[0], B.shape[0]), dtype=A.dtype)
for i in range(A.shape[0]):
    for j in range(B.shape[0]):
        s = 0.0
        for k in range(A.shape[1]):
            s += A[i, k] * B[j, k]      # B accessed as B^T
        y[i, j] = s
```
```python
# After (option B — materialise the transpose into a declared buffer first)
Bt = np.empty((B.shape[1], B.shape[0]), dtype=B.dtype)
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        Bt[j, i] = B[i, j]
y = A @ Bt
```

### 4.7 Tuple-of-arrays varying per iteration → N named tensors

**Why:** a Python tuple whose members change shape/identity per loop has no static
layout; give each its own named buffer.

```python
# Before
state = (np.zeros(n), np.zeros(n))
for t in range(T):
    state = step(state)        # tuple rebound each iter
```
```python
# After
s0 = np.zeros((n,), dtype=np.float64)
s1 = np.zeros((n,), dtype=np.float64)
for t in range(T):
    step_into(s0, s1)          # writes both buffers in place by index
```

### 4.8 `.append()` / dynamic growth → pre-declared worst-case buffer

**Why:** see Invariant 3 — declare the maximum extent, fill by index, track a count
if needed.

```python
# Before
out = []
for i in range(M):
    out.append(f(i))
```
```python
# After
out = np.empty((M,), dtype=np.float64)   # worst-case size known from params
for i in range(M):
    out[i] = f(i)
```

---

## 5. What CNF Lets the Translator Delete

With CNF guaranteed, these `lowering.py` mechanisms become unnecessary and can be
retired:

- **`_ssa_rename_reassigned`** (lowering.py:558) — invented fresh names for variables
  reassigned with a new shape. Invariant 1 means a name never changes shape, so there
  is nothing to rename.
- **`_LiftFreshArrayFromSlices`** (lowering.py:1774) — lifted a fresh array out of
  slice expressions when a buffer's shape didn't match its slice writes; its own
  comment notes it *"bails on the shape mismatch."* Declare-then-fill (Inv. 3)
  removes the mismatch.
- **Chained-subscript heroics in `_emit_subscript`** (emit.py:364, the
  *"Fall back to chained `[][]` if we have no shape info"* branch at emit.py:378) —
  full-rank indexing (Inv. 2) means the emitter always has shape info and never needs
  the fallback.
- **Rank-aware `expand_reshape` fallback** (lowering.py:2167+, the `x = np.reshape(x,
  ...)` rewrite and the `x.shape = expr` pre-pass at lowering.py:2958) — reshape only
  ever targets a fresh buffer of a declared shape (Inv. 1 / cookbook 4.2, 4.4), so the
  rank-changing in-place reshape path disappears.

The `shape_table`/`_harvest_local_shapes` machinery can then be a single up-front
declaration scan instead of a mutable structure threaded through ~22 passes.

---

## 6. The CNF contract (checked in review)

New kernels are reviewed against the following CNF invariants; a violation points at
the line and names the canonical fix (it is never auto-rewritten):

- **Inv. 1:** flag any `Name` target that is assigned a new shape after first
  declaration (track each name's declared shape; error if a later assign produces a
  different rank/shape).
- **Inv. 2:** flag `Subscript` nodes that partially index a rank>1 array (chained
  rank) in a compute context, and flag fancy indexing (`a[index_array]`) outside a
  whitelisted sparse-gather pattern.
- **Inv. 3:** flag `list`/`dict`/`set` literals, `.append`/`.extend`, `np.concatenate`,
  `np.append`, and any array declared from a runtime-sized source.
- **Vocabulary (§3):** flag `np.*` calls not in the allowed set (e.g. `np.mgrid`,
  `np.repeat`, inline `.T`).

Error messages should name the line, the violated invariant, and the cookbook entry:

```
contour_integral_numpy.py:13: CNF Invariant 2 (explicit indexing):
    chained subscript `Ham[n]` on 3-D array `Ham`.
    Fix (cookbook 4.1): index every axis, e.g. `Ham[n, i, j]` in a loop nest.

stockham_fft_numpy.py:21: CNF Invariant 1 (static shape):
    `np.reshape` changes the rank of live array `y`.
    Fix (cookbook 4.2/4.4): reshape into a freshly declared buffer.
```

A non-grandfathered kernel that violates these is rewritten before it lands (§7).

---

## 7. Migration Policy

- **Failing kernels are rewritten into CNF now.** A kernel that does not currently
  lower is the one we fix, using the cookbook in §4.
- **Passing kernels are grandfathered.** Kernels that already translate are left
  as-is and exempt from the validator for now; we do not churn working benchmarks.
  (Many of them — `gemm`, `jacobi_2d`, `atax`, `doitgen` — are already CNF or close to
  it and serve as positive examples.)
- **New kernels are authored in CNF from the start.** Any new benchmark — HPC,
  sparse, or foundation-model additions — must satisfy the CNF invariants (§6). Read
  this document and write to the vocabulary in §3 before submitting.

When in doubt: one name = one shape (Inv. 1), index every axis (Inv. 2), declare then
fill (Inv. 3).
