# Failing DaCe SDFGs — root causes and desugaring fixes

Every `<module>_dace.py` is **auto-generated** by `numpyto_c.dace_emit` (a
near-verbatim copy of the numpy reference wrapped in `@dc.program`). The
failures below are all cases where the emitter passes a numpy idiom the DaCe
`@dc.program` frontend cannot lower. The fix belongs in the **emitter**
(`optarena/numpy_translators/src/numpyto_c/dace_emit.py`), i.e. desugar the
idiom to a DaCe-expressible form at emit time — not by hand-editing the
generated ports (they are regenerated).

## Status (2026-07-13)

**4 of 6 fixed** by emitter desugars in `dace_emit.py` (sweep: **23→27 ok /
6→2 fail, zero regressions**). Fixed: **gemver, covariance, covariance2,
correlation**. Still open (neither is a clean desugar): **durbin**
(dynamic-length reversed self-referential recurrence — hits dace's dynamic
transient limit) and **seissol_batched_gemm** (free-symbol `NQ` plumbing +
canonicalize CMake codegen error).

Desugars added (all verified, 259 `test_dace_emit` green):
- `_DesugarOuter` — `np.outer(a,b)` → `a[:,None]*b[None,:]`
- `_DesugarReverseSlice` — `x[::-1]` → `np.flip(x)`
- `_DesugarChainedAssign` — `a[s1]=a[s2]=rhs` → temp + per-target write
- `_DesugarBroadcastAugAssign` — `A op= b` (broadcast) → `A[:] = A op b`
- `_inline_symbol_aliases` — fold a pure-symbol reduction-dim alias
  (`__rd0_d1 = M`) into `M` instead of promoting a fresh, un-unifiable symbol

## Scope / provenance

- Enumerated with a **fork-isolated** dace_cpu S-preset validation sweep over
  29 dense-linear-algebra + structured-grid kernels (`scratchpad/dace_sweep.py`).
  Fork isolation matters: running a compiled DaCe kernel *non-forked* in a
  process that has already imported jax/scipy/mpi4py corrupts the heap
  (segfault / `free(): corrupted unsorted chunks`) — an artifact of mixed
  native libs, **not** a kernel bug. Under fork isolation **23/29 pass**
  (incl. gemm, k2mm, atax, mvt, cholesky, jacobi_2d, heat_3d, fdtd_2d, adi).
- **All 6 failures are pre-existing.** A/B against `HEAD` (before the
  `DaceFramework.optimize` integration) reproduces every one identically — the
  optimize/select-fastest change did not introduce or regress any of them.
- Each proposed desugaring below was **verified to parse** through
  `to_sdfg(simplify=False)` in isolation (`scratchpad/dace_desugar_probe.py`,
  `scratchpad/dace_realfix_probe.py`).

## Failures

### 1. gemver — `np.outer` is an untyped callback  ✅ FIXED (`_DesugarOuter`)

```python
A += np.outer(u1, v1) + np.outer(u2, v2)   # gemver_dace.py:13
```
```
DaceSyntaxError: Trying to operate on a callback return value with an
undefined type. Please add a type hint to "pyobject" ...
```
DaCe has no replacement for `np.outer`, so it treats it as an opaque Python
callback with an undefined return type.

**Desugar:** `np.outer(a, b)` → `a[:, None] * b[None, :]`
(equivalently `np.reshape(a,(N,1)) * np.reshape(b,(1,N))`). Both parse. ✅

### 2 & 3 & 4. covariance / covariance2 / correlation — reduction-dim symbol not unified  ✅ FIXED

> Three layers, peeled in order — covariance needed all three, covariance2 only
> the first: (1) `_inline_symbol_aliases` folds `__rd0_d1` → `M`; (2)
> `_DesugarBroadcastAugAssign` turns the in-place `data -= mean` (which parses
> but builds an SDFG with a `[0:M] -> data[0:N,0:M]` dimensionality mismatch)
> into `data[:] = data - mean`; (3) `_DesugarChainedAssign` splits covariance's
> `cov[i:M,i] = cov[i,i:M] = rhs` (`Write slicing not implemented`) into a temp
> plus two writes.

```python
mean = __rdo0            # shape (__rd0_d1,)
data -= mean             # data is [N, M]   -> covariance_dace.py:23 / correlation_dace.py:38
# covariance2: centered = data - mean       (binop form, same cause)
```
```
IndexError: could not broadcast input array from shape [__rd0_d1] into shape [N, M]
```
The emitter lowers the column-mean reduction into a fresh temp
`__rdo0 = np.empty((__rd0_d1,), ...)` and records the equivalence in
`__optarena_symbol_defs__ = [('__rd0_d1', 'M')]`, **but never applies it**. To
the DaCe frontend `__rd0_d1` is a free symbol, so it cannot prove
`__rd0_d1 == M` and the broadcast against `[N, M]` fails. The subtract itself
is fine — bare `data -= mean` parses when `mean` is `[M]`.

**Fix (emitter):** substitute the known symbol equivalences from
`__optarena_symbol_defs__` into the emitted reduction-output shape, so the temp
is declared `np.empty((M,), ...)` (concrete dim) rather than `(__rd0_d1,)`.
With `mean` shaped `(M,)`, `data -= mean` parses. ✅
(A weaker, more local desugar — writing `data -= mean[None, :]` — does **not**
help here, because the operand's free `__rd0_d1` dim is the actual blocker.)

### 5. durbin — negative-stride slice  ⚠️ STILL OPEN (desugar applied, deeper limit remains)

```python
alpha = -(r[k] + np.dot(r[:k][::-1], y[:k])) / beta   # durbin_dace.py:18
y[:k] += alpha * y[:k][::-1]                           # durbin_dace.py:19
```
```
DaceSyntaxError: Negative strides are not supported in subscripts.
Please use a Map scope to express this operation.
```
DaCe subscripts reject a negative step.

**Desugar applied:** `x[::-1]` → `np.flip(x)` (`_DesugarReverseSlice`). This is
the correct general transform (helps any static-length reverse) and the two
`[::-1]` are gone. But durbin still does **not** build: its reverses are over a
*dynamic* length `k` inside a sequential recurrence
(`np.dot(np.flip(r[:k]), y[:k])`), and dace can neither view-reverse a dynamic
slice into a reduction (`InvalidSDFGNodeError: Ambiguous or invalid edge
to/from a View access node`) nor materialise a dynamic-size temp
(`np.flip(r[:k]).copy()` → `_add_transient_data: NotImplementedError`).
Additionally `y[:k] += alpha * y[:k][::-1]` reads the *old* reversed self while
writing, so a naive scalarised copy loop would be a WAR miscompile. A real fix
needs a Map-scope reverse over a fixed-size (`[N]`) workspace with an explicit
snapshot — a kernel-shaped rewrite, not a general desugar. **Left open.**

### 6. seissol_batched_gemm — free symbol `NQ` not wired + canonicalize CMake error  ⚠️ STILL OPEN

```
KeyError: 'Missing program argument "NQ"'
DaCe optimize: failed to compile cpu canonicalize:
  CMake Error at CMakeLists.txt:281 (add_library)
```
Two distinct problems:
- **`NQ` not passed.** The compiled SDFG has a free symbol `NQ` that
  `DaceFramework.call_args` never supplies — `params(bench, impl)` does not
  include it. Either the manifest is missing `NQ` as a size symbol, or the
  dace port declares a symbol the numpy reference/manifest doesn't expose.
  Needs the symbol wired through the manifest + `__optarena_symbol_defs__` so
  `call_args` binds it.
- **canonicalize CMake `add_library` failure** — a codegen/build problem for
  that variant, separate from the missing argument. The `parallel`/`autoopt`
  variants compiled but then hit the same missing-`NQ` at call time. Worth
  investigating whether the batched-gemm nested structure emits an empty/
  duplicate library target.

This one is **not** a simple emitter desugar; it is a symbol-plumbing (and
possibly codegen) issue.

## Remaining work

1. **durbin** — Map-scope reverse over a fixed-`[N]` workspace + explicit
   old-value snapshot for the self-referential `y[:k] += alpha*y[:k][::-1]`.
   Kernel-shaped, not a general desugar.
2. **seissol_batched_gemm** — wire `NQ` through the manifest +
   `__optarena_symbol_defs__` so `call_args` binds it; investigate the
   canonicalize `add_library` CMake error (possible empty/duplicate lib target
   from the batched nest).

(gemver, covariance, covariance2, correlation are done — see Status above.)

## Reproduce

One kernel, full traceback — run in a fresh process (fork or subprocess) so a
native crash from an *unrelated* passing kernel can't confuse the result:

```python
# repro.py  ->  PYTHONPATH=. python repro.py covariance
import os, sys
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("MPI4PY_RC_INITIALIZE", "0")
from optarena.infrastructure import Benchmark, Test, generate_framework
stem = sys.argv[1]
test = Test(Benchmark(stem), generate_framework("dace_cpu"), generate_framework("numpy"))
test.run("S", validate=True, repeat=1, ignore_errors=False, datatype="float64")
```

Use the **module stem** (`covariance`, `k2mm`, `durbin`, `seissol_batched_gemm`),
not the yaml `short_name`. Desugar targets can be checked directly with a tiny
`@dc.program` calling `.to_sdfg(simplify=False)` (see the parse results quoted
above).

## Not a DaCe SDFG failure (harness naming mismatch — separate bug)

`tests/test_s_preset_integration.py` reports `2mm`, `3mm`, `covarian`,
`jacobi1d`, `seidel2d`, ... as "failed", but these fail at **benchmark lookup**,
before any DaCe work: the yaml `short_name` (`2mm`, `covarian`, `jacobi1d`) does
not match the `KERNELS` registry key / module stem (`k2mm`, `covariance`,
`jacobi_2d`). `Benchmark(short_name)` raises
`KeyError: unknown benchmark '2mm' (no co-located YAML manifest)`. This is an
integration-test naming bug, unrelated to the DaCe backend.
