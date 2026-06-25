# NumpyToC — Kernel-author cheat sheet

> Audience: anyone writing numpy kernels (or PyTorch→numpy translators)
> targeting NumpyToC / NumpyToFortran. Lists what the pipeline can
> ingest today. Stick to this surface and the same kernel emits in C,
> C++, and Fortran from one numpy source.

---

## 1. Kernel signature

* **All inputs and outputs are passed as flat array buffers, C-style.**
  No return values. The benchmark harness allocates input + output
  arrays; the kernel mutates the output buffer in place.

  ```python
  # GOOD
  def kernel(A, B, C, alpha, beta):
      C[...] = alpha * A @ B + beta * C

  # BAD -- return value, would need tuple-unpack support
  def kernel(A, B):
      return A @ B
  ```

  If your reference numpy kernel returns the output, **rewrite it to
  write into a buffer parameter** (e.g. the canonical `_numpy.py`
  file is preserved for other backends, and a sibling
  `*_numpytoc_numpy.py` carries the buffer-form). See
  `banded_mmt_numpytoc_numpy.py` for the pattern.

* Scalars (int / float) pass by value. The dtype is inferred from the
  default in `bench_info/<short>.json` `init.scalars`. Use integer
  defaults for params that flow into subscripts; float defaults for
  numeric scalars.

* Symbols (`N`, `M`, `K`) come from `bench_info` `parameters` and
  appear in array shapes; do NOT pass them as args unless you also
  list them in `input_args`.

---

## 2. Data structures — AVOID

| Don't use | Reason |
|---|---|
| Tuples (multi-value return, tuple-unpack) | No tuple emit |
| Lists (Python list) | No list emit; use a flat numpy array |
| Dicts | No dict emit |
| `namedtuple`, `dataclass` | No struct emit |
| Helper functions returning tuples | Inline the helper instead |
| Dynamic-shape arrays (`Z = Z[mask]`) | Use static-shape + `length` cursor (see mandelbrot2_numpytoc) |
| Attribute shape mutation (`Xi.shape = N`) | Use `np.reshape(Xi, (N,))` -- this IS handled but the rewrite is explicit |
| In-place imports (`import scipy.sparse` inside body) | Top-level only |
| Tuple-return helpers (`return ret, lbound, ubound`) | Inline or buffer-form |

If you need a "tuple" of outputs, declare them as separate output
buffers in `bench_info` `output_args` and write into each.

---

## 3. Supported numpy ops (use these freely)

### Array creation / shape
* `np.zeros(shape, dtype=)`, `np.empty(...)`, `np.ones(...)`,
  `np.zeros_like(arr)`, `np.empty_like(arr)`, `np.ones_like(arr)`,
  `np.full(shape, val)`, `np.full_like(arr, val)`
* `np.ndarray((I, J, K), dtype=)` -- treated as `np.empty`
* `np.mgrid[0:R, 0:S]` -> two index grids
* `np.eye(N)`, `np.identity(N)`
* `np.linspace(start, stop, n)`, `np.arange(start, stop)` /
  `np.arange(stop)`
* `np.reshape(arr, new_shape)` -- shape-only, no data move
* **`x.shape = expr`** -- rewritten to `np.reshape` globally (the
  mandelbrot2 idiom)
* `arr.T` / `np.transpose(arr)` -- works on declared 2-D Names

### Elementwise math (use freely; map to the same intrinsic in all 3
emit targets)
* Arithmetic: `+`, `-`, `*`, `/`, `**`, `//`, `%`
* Math: `np.exp / log / sqrt / sin / cos / tan / tanh / abs / fabs`
* Compare: `<`, `<=`, `>`, `>=`, `==`, `!=`
* Boolean: `np.logical_and / logical_or / logical_not`,
  `&`, `|`, `^`, `~` (bitwise -- also work on bool arrays)
* Min/Max: `np.maximum / minimum / clip`
* Power: `np.power(a, b)`, `np.true_divide`, `np.copy`,
  `np.negative`

### Reductions (full and axis-aware)
* `np.sum / mean / prod / max / min / std / var` -- support
  `axis=None / int / tuple`, `keepdims=True/False`
* `np.argmax / argmin` -- axis None / int / tuple; tuple gives a
  flat-index across reduced axes
* `np.any / all / count_nonzero`
* `np.linalg.norm` (L2 only) -- axis-aware
* `np.linalg.cholesky` (Cholesky-Banachiewicz)
* `np.linalg.inv` (Gauss-Jordan with partial pivoting)
* `np.linalg.solve(A, b)` (Gauss-Jordan on augmented [A|b])
* `np.linalg.lstsq(A, b)[0]` (Gauss-Jordan solve form)
* `np.histogram(a, bins[, range][, weights])[0]` (per-element bucket)
* `np.dot`, `np.vdot`, `np.inner` (1-D and matrix forms)
* `np.matmul` / `@` -- bare 2-D Names; for higher rank use loops

### Indexing
* Integer indices: `arr[i, j, k]`
* Slices: `arr[1:N, :, k]`, `arr[:-1, ...]`, `arr[::-1]` (reverse),
  `arr[a:b]`, `arr[a:b:step]`
* Boolean masks: `arr[bool_mask]` -- works in fused form
  `mean(arr[mask])` / `sum(arr[mask])` / `max(arr[mask])` /
  `min(arr[mask])`. The materialised compacted array is NOT
  supported as a standalone value; only as the operand to a
  reduction in the same statement chain.
* `np.newaxis` / `None` as broadcast axis -- handled
* Fancy gather: `arr[idx_array]` where `idx_array` is 1-D int

### Conditional
* `if / else` with scalar conditions
* `while` loops with scalar conditions
* `np.where(cond, a, b)` (vector ternary)
* `for k in range(N):` and `for k in range(lo, hi):` and
  `for k in range(lo, hi, step):` (positive and negative step)

### Lifecycle / control
* Augmented assigns: `+= -= *= /= //= %= **= &= |= ^= <<= >>=`
* Boolean-mask augmented assign: `arr[mask] += value` etc.
* `for` loop iter dtype inherits from the iterated array

---

## 4. Tips for translators (PyTorch → numpy → NumpyToC)

* **Tensor reshape → `np.reshape`**. The `x.view(N, M)` PyTorch idiom
  maps directly. Avoid `.shape = ...` -- use `np.reshape` even though
  the pipeline rewrites it.
* **Tensor transpose → `np.transpose` or `arr.T`**. The 2-D form is
  fully supported.
* **Tensor permute (>2D) → write the loop**. NumpyToC supports
  `np.transpose` with a permutation argument but for clarity write
  the explicit triple loop.
* **PyTorch reduce ops → numpy equivalents** as listed above.
  `axis=` / `dim=` argument naming matches.
* **PyTorch in-place ops (`x.add_(y)`) → numpy `x += y`**.
* **No autograd, no requires_grad**, no `.detach()` etc. -- strip
  them in the converter.
* **No `torch.cat`** -- preallocate the target buffer and write
  element-wise into the offset region.
* **Dtype**: declare in `bench_info` `init.dtypes` for input arrays;
  use `np.zeros(..., dtype=np.float64)` for locals.

---

## 5. bench_info schema highlights

The `bench_info/<short>.json` file drives kernel-level type and
shape inference. Minimum:

```json
{
  "benchmark": {
    "name": "...",
    "short_name": "...",
    "relative_path": "<dir under benchmarks/>",
    "module_name": "<file stem without _numpy>",
    "func_name": "<callable to invoke>",
    "kind": "microbench",
    "domain": "...",
    "dwarf": "...",
    "parameters": {
      "S": { "N": 1000 },
      "M": { "N": 5000 },
      "L": { "N": 10000 },
      "paper": { "N": 8000 }
    },
    "init": {
      "func_name": "initialize",
      "input_args": ["N"],
      "output_args": ["A", "x", "b"],
      "shapes": { "A": "(N, N)", "x": "(N,)", "b": "(N,)" },
      "dtypes": { "A": "float64", "x": "float64", "b": "float64" },
      "scalars": { "max_iter": 100, "tol": 1.0e-6 }
    },
    "input_args": ["A", "x", "b", "max_iter", "tol"],
    "array_args": ["A", "x", "b"],
    "output_args": ["x"]
  }
}
```

Key fields:
* `init.shapes` -- per-array shape expressions over the symbol set;
  required for arrays not covered by `_shapes_from_initialize`.
* `init.dtypes` -- explicit per-array dtype override; wins over the
  initialize-source harvest. Use this when the initialize source
  uses helpers NumpyToC can't introspect (e.g. `rng_complex`).
* `init.scalars` -- default values for non-array scalar args.
  Integer defaults => integer C type (subscript-safe); float
  defaults => double.
* `array_args` -- subset of input_args that are arrays (drives the
  pointer-vs-value emit decision).
* `output_args` -- arrays whose values are written by the kernel
  (drives the `intent(inout)` Fortran decl).

---

## 6. Side-file variant for non-pure-numpy kernels

If the canonical `<short>_numpy.py` uses features the pipeline can't
ingest (dynamic shape, `.shape =`, tuple returns, scipy imports),
write a sibling **`<short>_numpytoc_numpy.py`** with the same
function name but a static-shape / buffer-form rewrite. The emit
script automatically shadows the canonical; other backends (numba /
pythran / cupy / jax) keep using the canonical untouched.

Examples in the tree:
* `mandelbrot2_numpytoc_numpy.py` -- static-shape `Z[:length]` form
  replacing the dynamic `Z = Z[mask]` shrink
* `banded_mmt_numpytoc_numpy.py` -- inline buffer-form replacing
  3-tuple returns through helper functions
* `gmres_numpytoc_numpy.py` -- pre-materialised lstsq `b` argument
* `vadv_numpytoc_numpy.py` -- explicit `[:-1, :, k]` writes
  replacing gt4py write-to-subset semantics

---

## 7. Quick reference -- features at a glance

| Category | Use freely | Avoid |
|---|---|---|
| Scalars | `int`, `float`, `bool` params + locals | Python `complex` (use `1j` literals if needed) |
| Arrays | `np.zeros / empty / ones / mgrid / linspace / arange`, `np.ndarray((shape,))` | Dynamic-resize: `np.append`, `np.concatenate` |
| Shape | `np.reshape`, `arr.shape[i]`, `arr.T` | `arr.shape = N` is auto-rewritten but discouraged |
| Math | All `np.<op>` elementwise + reductions listed in 3. | `np.fft`, `np.random`, `scipy.*` |
| Linalg | `cholesky / inv / solve / lstsq / norm / dot / @` (2-D) | Higher-rank `@` (write explicit loop) |
| Indexing | Int, slice (incl. step), boolean mask (when consumed by reduction), fancy gather | Multi-dim fancy index (`arr[ix, iy]`), advanced indexing combinators |
| Control | `if / else / while / for (+ negative step)`, `break`, `continue` | Generators, comprehensions (write explicit loops) |
| I/O | None | Any `print`, `open`, `os.*` |
| Calls | Inline helper functions (no tuple return) | Tuple/list/dict returns; recursion |

When unsure: start with the simplest explicit `for` loop and only
reach for numpy intrinsics where the gain is real. The same shape of
code emits in all three targets.
