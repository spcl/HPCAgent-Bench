# NumpyToJAX — prototype numpy → JAX kernel emitter

Auto-translates an OptArena `*_numpy.py` kernel into a JAX kernel. By default it
emits an **eager** (non-`jit`) kernel — the most faithful 1:1 translation and
the widest coverage. With `jit=True` it instead runs a loop-lowering classifier
and the masking transforms to produce a compiled, hand-`*_jax.py`-style kernel.
It raises `EmitError` (rather than emitting something wrong) on what it cannot
lower.

```python
from optarena.NumpyToJAX import emit_jax, EmitError
jax_src = emit_jax(open("gemm_numpy.py").read(), "kernel")            # eager (default)
jax_src = emit_jax(open("gemm_numpy.py").read(), "kernel", jit=True)  # jit/compiled
```

## Eager mode (default)

`jnp` mirrors `numpy`, so the bulk of the translation is mechanical: `np.` →
`jnp.`, and in-place mutation → functional updates (JAX arrays are immutable, so
`A[i] = v` becomes `A = A.at[i].set(v)`, `x += y` becomes `x = x + y`, and
`arr.shape = n` becomes `arr = arr.reshape(n)`). Eager JAX executes concrete
arrays op-by-op, so **Python control flow is kept verbatim** — `for`/`while`/
`if`/`break`, *any* `range` step (`range(1, N, 2)`), data-dependent slices
(`A[i, :j]`), boolean indexing (`A[m]`), and shrinking-array compaction
(`Z = Z[I]`) all just run. No loop classification, no masking, no rejection —
this is what lets eager cover the strided / data-dependent foundation kernels
and shape-changing kernels (mandelbrot2) that the `jit` path must refuse.

Bare in-place ufuncs are rebound to their out arg (`np.multiply(Z, Z, Z)` →
`Z = jnp.multiply(Z, Z)`); a bare `math` function is mapped to its `jnp` ufunc
(`sin(b)` → `jnp.sin(b)`, `asin`→`arcsin`, `pow`→`power`) so it survives
vectorisation and tracing — `math.sin` only accepts a host scalar and would
raise on a whole array or a traced `b[jg]` — and the kernel's own imports are
still carried over for any unmapped name.

The cost is speed: a loop-heavy kernel (the `TSTEPS × N²` polybench stencils —
seidel/adi/heat3d/jacobi) dispatches every scalar `.at[].set()` separately and
is **slow** eagerly where the `jit` path would vectorise it. Correctness is
preserved; `jit=True` is the escape hatch.

## The `jit=True` path — loop lowering

The **load-bearing decision** is which JAX control-flow construct each Python
loop becomes:

| numpy loop | lowers to |
|---|---|
| `a[i] = f(b[i])`, no carry, index only inside subscripts | **vectorised** — the loop disappears into a whole-array op |
| loop-carried state, no break | `jax.lax.fori_loop` carrying the state tuple |
| data-dependent `break` (or `while`) | `jax.lax.while_loop` carrying state + a `done` flag |

A `for i in range(...)` with an `if cond: … break` guard carries the index plus
a `done` flag, in one of two shapes. The **convergence guard** (`if rsnew <
tol: break` — the iterative solvers) freezes *carried* vars **after** the guard
with `jnp.where(_conv, old, new)` — the converging iteration keeps its pre-break
value. The **search / capture** (`if a[i] > thr: index = i; value = a[i]; break`
— find-first, s332/ext_break_capture) commits the capture **on** the converging
iteration with the opposite polarity `jnp.where(_conv, new, old)`. Post-break
*local temps* (minres's `beta`) are emitted plainly.

The carry set comes from a **read-before-write** analysis (recursing into nested
`if`/loops): a write-before-read temp (gramschmidt's `nrm`) stays loop-local,
while a **conditionally** written var (s258's `if a[i] > 0: s = …` then a read of
`s`) is carried — a one-branch write is *not* a definite write, so on the other
path the value persists across iterations and must thread through the loop. A
bare `while c: …` also feeds the names read in **its own test** into the carry
analysis (channel_flow's `while udiff > .001`, where the body recomputes
`udiff`): the test is evaluated before each iteration, so a name it reads that
the body writes is a genuine cross-iteration carry — otherwise the `_cond`
closure would capture the pre-loop value as a free var and the loop would never
terminate. This lets the data-dependent CFD convergence loop lower to a compiled
`lax.while_loop` instead of falling back to a slow eager Python loop.

## Transforms

* **Dynamic-slice reductions** — `A[i, :j] @ A[:j, j]`, `np.dot(A[i,:k], …)`:
  masked to full width (`np.where(arange<j, …, 0)`), exact because the masked
  lanes are the reduction identity. Unlocks cholesky/lu/ludcmp/trisolv/trmm.
* **Dynamic-slice writes** — `C[i, :i+1] += …` → masked `.at[i, :].set(where(…))`
  (syrk/syr2k/symm); two-sided bounds (`i:M`) and chained writes
  (`cov[i:M,i] = cov[i,i:M] = …`) too (correlation/covariance).
* **Fixed-width windows** — `input[:, i:i+K, j:j+K, :]` → nested
  `lax.dynamic_slice_in_dim` (conv2d/lenet/resnet).
* **Boolean masks** — `A[m] = v`, `A[m] **= …`, `A[m].mean()`, and a masked
  subset bound to a temp (`v = data[m]; v.mean()`) → `jnp.where`/masked sums
  (mandelbrot1, nbody, azimint_naive).
* **Data-dependent `if`** — `if g: table[i,j]=max(…)` → `jnp.where` selects
  (only live-out vars); `if c: return A else: return B` → `return where(c,A,B)`;
  a *static* condition (jit args only, e.g. `if NR==NM`) stays a real branch.
  `max`/`min`→`jnp.maximum/minimum`, `and`/`or`→`&`/`|`. Powers nussinov,
  scattering.
* **Multi-function modules** — helper functions (`relu`, `conv2d`, `build_up_b`)
  are emitted alongside the kernel; an in-place helper called for side effect
  (`build_up_b(b, …)`) is rewritten to capture its result (`b = build_up_b(b,…)`).
* **Misc** — reversed ranges (`range(a,b,-1)`), `for x in arr` → indexed range,
  tuple-targets (`KE[i+1], PE[i+1] = …`), `np_float`/`np_complex` globals,
  `np.ndarray`→`jnp.empty`, `np.histogram`→`jnp.histogram`, module constants,
  `static_argnames` for integer dims/bounds (by use, not spelling). An array
  fill populated with index arithmetic (`res[i] = rmax*i/npt`) stays a
  `fori_loop` + `.at[].set()` rather than being vectorised away.
* **Sparse** — the iterative solvers (cg/bicg/bicgstab/minres) emit
  dense-looking `A @ x`, which runs unchanged when `A` arrives as a JAX `BCOO`.

More transforms: `np.flip(arr[:k])` → a reversal gather (durbin/Levinson);
ragged CSR slices `A_col[A_row[i]:A_row[i+1]]` masked to full width
(spmv); a loop whose index feeds a shape (`reshape(y, (R**i, …))`) is
**unrolled** to a real Python loop the tracer expands (stockham_fft).

## What it does *not* lower (clean `EmitError` / out of scope)

* **Shape-changing loops** — mandelbrot2's boolean-mask compaction (`Z = Z[I]`
  shrinks the array each iteration): no static shape, cannot be `jit`-traced.
  The algorithm itself is jit-incompatible, not a translation gap.
* **Data-dependent Krylov dimension + lstsq** — gmres (`m = k+1` reassigned at
  the break, then `H[:m]`/`lstsq`).
* **SpGEMM** — spmm / banded_mmt (sparse @ sparse densifies → OOM; the hand
  kernels manually densify one operand, which the emitter cannot infer).

## Verifying

`scripts/emit_jax_check.py` emits each kernel, runs it, and compares — *first*
against the hand-written `*_jax.py` on identical JAX inputs (`[vs-hand]`, the
authoritative "does it match the existing kernel?" signal), else the numpy
reference (`[vs-numpy]`, used for the whole foundation track, which has no hand
JAX). Sparse inputs are converted to `BCOO` like the real `jax_framework`.

```
python scripts/emit_jax_check.py                 # representative set
python scripts/emit_jax_check.py --all            # the 62 legacy benches
python scripts/emit_jax_check.py --foundation     # the 213 TSVC foundation puzzles
python scripts/emit_jax_check.py --everything     # both
SHOW=1 python scripts/emit_jax_check.py polybench/gemm   # print emitted source
```

### AOT compilation (default)

By default the harness **AOT-compiles each kernel before running it** —
`jax.jit(kernel).lower(*args).compile()` — and runs the compiled artifact (the
"prepare" then "run" split). This traces a loop-heavy kernel into one XLA
program, so the polybench stencils and TSVC loops run *fast* instead of
dispatching every eager op separately. Scalar/dim args are baked in as
constants (so `range(N)` traces); a kernel whose control flow is genuinely
data-dependent (mandelbrot2's compaction, gmres's break) can't be traced and
**falls back to eager execution** — the `(aot)`/`(eager)` tag on each result
line says which happened. `--no-aot` runs everything eagerly.

One thing that would otherwise force the fallback is Python's `float()` builtin:
the TSVC argmax kernels close with `result = maxv + float(index)`, and in the
rolled body `index` is a *traced* carry — `float()` must return a host Python
float, which a tracer can't provide, so the whole kernel would drop to eager.
The emitter rewrites `float(x)` to a traceable `jnp.asarray(x, jnp.float64)`
(safe because a float result only ever feeds arithmetic, never a `range`/shape
that needs a concrete int), so those kernels AOT-compile as `(aot-jit)`. `int()`
is deliberately left alone — it *can* feed `range(int(x))`, which needs the
concrete value.

The flip side is the user's own warning: AOT-compiling a *huge* unrolled loop
(seidel/adi-style `TSTEPS × N²`) can take a long time and a lot of memory. The
rolled (`jit=True`) emission already lowers such loops to `lax.fori_loop` so
they don't unroll; the eager-AOT *fallback* (taken only when the rolled
emission is unavailable) is guarded against the blow-up two ways:

* **Time-step symbol heuristic** — a `for _ in range(… TSTEPS …)` loop is, by
  OptArena convention, a *time-march* loop: it must stay rolled, never unrolled,
  no matter how small the preset's value is. The check is a curated, extendable
  list of bound symbols (`TIMESTEP_SYMBOLS` = `TSTEPS`, `TSTEP`, `TMAX`,
  `NITER`, `NSTEPS`, … — matched case-insensitively as a substring of a
  loop-bound name; add a suite's own spelling there). Any kernel with such a
  loop skips eager-AOT and runs eagerly. This is the size-independent signal,
  reliable enough in practice.
* **Trip-count estimate** — `_unroll_estimate` resolves each `range()` bound
  against the concrete preset sizes (and array shapes) and multiplies the
  loop nest; a product over the budget (`UNROLL_LIMIT`, 20 k) likewise skips
  eager-AOT. The RSS watchdog below is the hard backstop for whatever slips
  through.

### Robust sweeps (memory- and time-bounded)

So each bench runs in a **forked child watched by the parent**: the parent polls
the child's RSS and wall-clock and `SIGKILL`s it on breach, recording
`[OOM ]`/`[SLOW]` and moving on — a runaway compile takes down only that child,
never the sweep or the machine (via the kernel OOM-killer).

* `--mem G` — per-bench RSS cap in GB (**default 2.0**; a non-zero cap
  auto-enables isolation). The watchdog measures real RSS, so it doesn't
  false-trip on XLA's large virtual reservations the way `RLIMIT_AS` would.
* `--timeout S` — per-bench compile+run budget (inner `SIGALRM`, **default
  300**).

The harness also forces **single-threaded** XLA before `import jax`
(`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false …` + `OMP/MKL/OPENBLAS_NUM_THREADS=1`)
so a parallel compile/execution doesn't fan out across cores and inflate peak
memory.

### Coverage

`--everything` (62 legacy + 213 foundation, `--mem 2.0 --timeout 45`): **273
pass / 0 fail / 2 skip** — zero wrong answers. Legacy **60/62**: polybench
**32/32**, deep-learning, weather, sparse solvers (cg/bicg/bicgstab/minres),
spmv, stockham_fft, cavity/channel, nbody, scattering, crc16, contour_integral.
Foundation **213/213** (the whole TSVC track). The loop-heavy kernels that used
to land in SLOW — the polybench stencils (seidel/adi/heat_3d/jacobi),
durbin/syrk/correlation/covariance/trisolv, azimint, and channel_flow's
data-dependent CFD convergence `while` (now a compiled `lax.while_loop`) — now
AOT-compile and **pass fast** (the `(aot-jit)`/`(aot)` tag). Eager additionally
*emits and runs* the kernels the `jit` path refuses — mandelbrot2's
shrinking-array compaction (`Z = Z[I]`) executes faithfully and verifies within
budget.

The 2 non-passes are benign: **1 SLOW** (spmm, whose SpGEMM densifies past the
budget) and **1 EMIT** (banded_mmt — a packed-banded hand-algorithm with `map`/
`lambda`, chained-subscript stores and ragged data-dependent slices, which the
emitter cleanly refuses rather than mis-translate).
