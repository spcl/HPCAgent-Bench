# NumpyToJAX -- prototype numpy -> JAX kernel emitter

Auto-translates an HPCAgent-Bench `*_numpy.py` kernel into a JAX kernel. By default it
emits an **eager** (non-`jit`) kernel -- the most faithful 1:1 translation and
the widest coverage. With `jit=True` it instead runs a loop-lowering classifier
and the masking transforms to produce a compiled, hand-`*_jax.py`-style kernel.
It raises `EmitError` (rather than emitting something wrong) on what it cannot
lower.

```python
from numpyto_jax import emit_jax, EmitError
jax_src = emit_jax(open("gemm_numpy.py").read(), "kernel")            # eager (default)
jax_src = emit_jax(open("gemm_numpy.py").read(), "kernel", jit=True)  # jit/compiled
```

## Eager mode (default)

`jnp` mirrors `numpy`, so the bulk of the translation is mechanical: `np.` ->
`jnp.`, and in-place mutation -> functional updates (JAX arrays are immutable, so
`A[i] = v` becomes `A = A.at[i].set(v)`, `x += y` becomes `x = x + y`, and
`arr.shape = n` becomes `arr = arr.reshape(n)`). Eager JAX executes concrete
arrays op-by-op, so **Python control flow is kept verbatim** -- `for`/`while`/
`if`/`break`, *any* `range` step (`range(1, N, 2)`), data-dependent slices
(`A[i, :j]`), boolean indexing (`A[m]`), and shrinking-array compaction
(`Z = Z[I]`) all just run. No loop classification, no masking, no rejection --
this is what lets eager cover the strided / data-dependent loop_level_reasoning kernels
and shape-changing kernels (mandelbrot2) that the `jit` path must refuse.

Bare in-place ufuncs are rebound to their out arg (`np.multiply(Z, Z, Z)` ->
`Z = jnp.multiply(Z, Z)`); a bare `math` function is mapped to its `jnp` ufunc
(`sin(b)` -> `jnp.sin(b)`, `asin`->`arcsin`, `pow`->`power`) so it survives
vectorisation and tracing -- `math.sin` only accepts a host scalar and would
raise on a whole array or a traced `b[jg]` -- and the kernel's own imports are
still carried over for any unmapped name.

The cost is speed: a loop-heavy kernel (the `TSTEPS x N^2` polybench stencils --
seidel/adi/heat3d/jacobi) dispatches every scalar `.at[].set()` separately and
is **slow** eagerly where the `jit` path would vectorise it. Correctness is
preserved; `jit=True` is the escape hatch.

## The `jit=True` path -- loop lowering

The **load-bearing decision** is which JAX control-flow construct each Python
loop becomes:

| numpy loop | lowers to |
|---|---|
| `a[i] = f(b[i])`, no carry, index only inside subscripts | **vectorised** -- the loop disappears into a whole-array op |
| loop-carried state, no break | `jax.lax.fori_loop` carrying the state tuple |
| data-dependent `break` (or `while`) | `jax.lax.while_loop` carrying state + a `done` flag |

A `for i in range(...)` with an `if cond: ... break` guard carries the index plus
a `done` flag, in one of two shapes. The **convergence guard** (`if rsnew <
tol: break` -- the iterative solvers) freezes *carried* vars **after** the guard
with `jnp.where(_conv, old, new)` -- the converging iteration keeps its pre-break
value. The **search / capture** (`if a[i] > thr: index = i; value = a[i]; break`
-- find-first, s332/ext_break_capture) commits the capture **on** the converging
iteration with the opposite polarity `jnp.where(_conv, new, old)`. Post-break
*local temps* (minres's `beta`) are emitted plainly.

The carry set comes from a **read-before-write** analysis (recursing into nested
`if`/loops): a write-before-read temp (gramschmidt's `nrm`) stays loop-local,
while a **conditionally** written var (s258's `if a[i] > 0: s = ...` then a read of
`s`) is carried -- a one-branch write is *not* a definite write, so on the other
path the value persists across iterations and must thread through the loop. A
bare `while c: ...` also feeds the names read in **its own test** into the carry
analysis (channel_flow's `while udiff > .001`, where the body recomputes
`udiff`): the test is evaluated before each iteration, so a name it reads that
the body writes is a genuine cross-iteration carry -- otherwise the `_cond`
closure would capture the pre-loop value as a free var and the loop would never
terminate. This lets the data-dependent CFD convergence loop lower to a compiled
`lax.while_loop` instead of falling back to a slow eager Python loop.

## Transforms

* **Dynamic-slice reductions** -- `A[i, :j] @ A[:j, j]`, `np.dot(A[i,:k], ...)`:
  masked to full width (`np.where(arange<j, ..., 0)`), exact because the masked
  lanes are the reduction identity. Unlocks cholesky/lu/ludcmp/trisolv/trmm.
* **Dynamic-slice writes** -- `C[i, :i+1] += ...` -> masked `.at[i, :].set(where(...))`
  (syrk/syr2k/symm); two-sided bounds (`i:M`) and chained writes
  (`cov[i:M,i] = cov[i,i:M] = ...`) too (correlation/covariance).
* **Fixed-width windows** -- `input[:, i:i+K, j:j+K, :]` -> nested
  `lax.dynamic_slice_in_dim` (conv2d/lenet/resnet).
* **Boolean masks** -- `A[m] = v`, `A[m] **= ...`, `A[m].mean()`, and a masked
  subset bound to a temp (`v = data[m]; v.mean()`) -> `jnp.where`/masked sums
  (mandelbrot1, nbody, azimint_naive).
* **Data-dependent `if`** -- `if g: table[i,j]=max(...)` -> `jnp.where` selects
  (only live-out vars); `if c: return A else: return B` -> `return where(c,A,B)`;
  a *static* condition (jit args only, e.g. `if NR==NM`) stays a real branch.
  `max`/`min`->`jnp.maximum/minimum`, `and`/`or`->`&`/`|`. Powers nussinov,
  scattering.
* **Multi-function modules** -- helper functions (`relu`, `conv2d`, `build_up_b`)
  are emitted alongside the kernel; an in-place helper called for side effect
  (`build_up_b(b, ...)`) is rewritten to capture its result (`b = build_up_b(b,...)`).
* **Misc** -- reversed ranges (`range(a,b,-1)`), `for x in arr` -> indexed range,
  tuple-targets (`KE[i+1], PE[i+1] = ...`), `np_float`/`np_complex` globals,
  `np.ndarray`->`jnp.empty`, `np.histogram`->`jnp.histogram`, module constants,
  `static_argnames` for integer dims/bounds (by use, not spelling). An array
  fill populated with index arithmetic (`res[i] = rmax*i/npt`) stays a
  `fori_loop` + `.at[].set()` rather than being vectorised away.
* **Sparse** -- the iterative solvers (cg/bicg/bicgstab/minres) emit
  dense-looking `A @ x`, which runs unchanged when `A` arrives as a JAX `BCOO`.

More transforms: `np.flip(arr[:k])` -> a reversal gather (durbin/Levinson);
ragged CSR slices `A_col[A_row[i]:A_row[i+1]]` masked to full width
(spmv); a loop whose index feeds a shape (`reshape(y, (R**i, ...))`) is
**unrolled** to a real Python loop the tracer expands (stockham_fft).

## What it does *not* lower (clean `EmitError` / out of scope)

* **Shape-changing loops** -- mandelbrot2's boolean-mask compaction (`Z = Z[I]`
  shrinks the array each iteration): no static shape, cannot be `jit`-traced.
  The algorithm itself is jit-incompatible, not a translation gap.
* **Data-dependent Krylov dimension + lstsq** -- gmres (`m = k+1` reassigned at
  the break, then `H[:m]`/`lstsq`).
* **SpGEMM** -- spmm / banded_mmt (sparse @ sparse densifies -> OOM; the hand
  kernels manually densify one operand, which the emitter cannot infer).

## Verifying

Unit coverage lives in `hpcagent_bench/numpy_translators/tests/` --
`test_jax_inplace_helpers.py` and `test_jax_semantics_fixes.py` emit a kernel and
execute both the numpy source and the emitted JAX source, asserting they agree.

End to end, the emitter is exercised through the harness: `hpcagent_bench.autogen` calls
`emit_jax` on demand and `jax_framework` runs the result, so a `jax` sweep over the
corpus (`hpcagent-bench run-framework -f jax`) is the integration check. Sparse inputs are
converted to `BCOO` there, as in the real framework.

### AOT compilation (default)

By default the harness **AOT-compiles each kernel before running it** --
`jax.jit(kernel).lower(*args).compile()` -- and runs the compiled artifact (the
"prepare" then "run" split). This traces a loop-heavy kernel into one XLA
program, so the polybench stencils and TSVC loops run *fast* instead of
dispatching every eager op separately. Scalar/dim args are baked in as
constants (so `range(N)` traces); a kernel whose control flow is genuinely
data-dependent (mandelbrot2's compaction, gmres's break) cannot be traced and
**falls back to eager execution** -- the `(aot)`/`(eager)` classification of a
kernel says which happened.

Python's `float()` builtin would otherwise force the fallback: the TSVC argmax
kernels close with `result = maxv + float(index)`, and in the rolled body
`index` is a *traced* carry -- `float()` must return a host Python
float, which a tracer cannot provide, so the whole kernel would drop to eager.
The emitter rewrites `float(x)` to a traceable `jnp.asarray(x, jnp.float64)`
(safe because a float result only ever feeds arithmetic, never a `range`/shape
that needs a concrete int), so those kernels AOT-compile as `(aot-jit)`. `int()`
is deliberately left alone -- it *can* feed `range(int(x))`, which needs the
concrete value.

The flip side: AOT-compiling a *huge* unrolled loop (seidel/adi-style
`TSTEPS x N^2`) can take a long time and a lot of memory. The
rolled (`jit=True`) emission already lowers such loops to `lax.fori_loop` so
they do not unroll; the eager-AOT *fallback* (taken only when the rolled
emission is unavailable) is guarded by the time-step symbol heuristic:

* a `for _ in range(... TSTEPS ...)` loop is, by
  HPCAgent-Bench convention, a *time-march* loop: it must stay rolled, never unrolled,
  no matter how small the preset's value is. The check is a curated, extendable
  list of bound symbols (`TIMESTEP_SYMBOLS` = `TSTEPS`, `TSTEP`, `TMAX`,
  `NITER`, `NSTEPS`, ... in `numpyto_common.parallelism` -- matched
  case-insensitively as a substring of a
  loop-bound name; add a suite's own spelling there). Any kernel with such a
  loop skips eager-AOT and runs eagerly -- the size-independent signal.
