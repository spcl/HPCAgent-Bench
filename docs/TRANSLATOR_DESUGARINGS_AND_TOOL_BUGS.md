# Translator Desugarings & Backend Tool Bugs

Living ledger for the numpy→{C, C++, Fortran, numba, pythran, jax, pluto} translators
(`optarena/numpy_translators/`). Two intertwined things are tracked here:

1. **Desugarings / emit-helpers we add** so a backend can express a kernel it otherwise
   rejects, or so an external tool (pluto, XLA) emits *correct / faster* code.
2. **Backend tool bugs** — defects in external tools (`polycc`/`pluto`/`pet`/CLooG, XLA)
   that we cannot fix in our lowering, with the representative kernels and the sanctioned
   disposition (auto reclassify-skip, or an emit-shape fix that makes the tool emit correct code).

The e2e gate (`tests/test_e2e_numerical.py`) translates each kernel to every backend and
compares against the kernel's own numpy. It is **strict-green**: `ok` passes, `skip:*` skips,
`FAIL:*` reds the build. There is **no** xfail-tolerance file any more — `tests/e2e_known_failures.txt`
and its loader were removed. A pair that legitimately can't pass must therefore be classified as a
`skip:*` (a backend/tool that genuinely can't express the kernel), never left as a `FAIL:*`. This
doc is the *why* behind each such disposition.

> Rule of thumb: **we never paper over a tool miscompile with a tuning flag** (see
> `tests/numerical_oracle.py` `_run_pluto`). If our emitted C/Fortran is bit-exact vs numpy and the
> tool still produces wrong output, that is a tool bug — reclassify it as a skip, do not mutate emit
> just to placate the tool unless the change is a legitimately better shape. For pluto this is now
> **automatic**: a post-transform `FAIL:*` whose sibling `c` backend is `ok` is recorded as
> `skip:unsupported:pluto-miscompile:*` (see §2).

---

## 1. Desugarings & emit-helpers we own

Status legend: **landed** = merged + unit-tested; **in-progress** = agent building;
**planned** = designed, not yet built.

### 1a. Op-support desugarings (make a kernel translatable at all)

| Op / pattern | Kernels needing it | Mechanism | Location | Status |
|---|---|---|---|---|
| module-level numeric tuple/list const inline (`_CW=(...)`) | laplacian_stencil_3d | fold to literal tuple (`seq_consts`) so `enumerate` unrolls to compile-time weights | `numpyto_common/frontend.py` `_inline_module_constants` | landed |
| `.ravel()`/`.flatten()` → `np.reshape(x,(-1,))` | poisson_cg_3d (`r.ravel()@r.ravel()`) | method rewriter | `numpyto_common/lowering.py` `_MethodCallRewriter` | landed |
| `enumerate(seq, start=s)` + literal-tuple unroll | laplacian_stencil_3d | `_EnumerateZipRewriter` start= handling + unroll | `numpyto_common/lowering.py` | landed |
| inline-hoist output shape for `roll`/`cholesky`/`tril`/`triu`/`reshape(-1)` | poisson, laplacian | `_CallHoister._derive_output_shape` branches | `numpyto_common/lib_nodes.py` | landed |
| `np.diag` (1-D→matrix w/ k offset; 2-D→delegate `expand_diagonal`) | ls3df_scf (Lanczos tridiagonal) | `expand_diag` zero-then-write; shape via `_iter_extent_of` | `numpyto_common/lib_nodes.py` | landed |
| `np.fft.fftfreq(N, d=)` | ls3df_scf | `expand_fftfreq`; even/odd neg-freq wrap; real output | `numpyto_common/lib_nodes.py` | landed |
| `np.einsum` with non-Name operand (`psi_frag[f]`) | fragment_patch_density, ls3df_scf | materialize operand to fresh scratch buffer, then expand | `numpyto_common/lib_nodes.py` `expand_einsum` | landed (caveat: Subscript operand nested *inside a BinOp* still won't hoist) |
| `np.linalg.eigvalsh(A)` (eigenvalues-only) | ls3df_scf (`_upper_bound`) | extend eigh cyclic-Jacobi, eigenvalues-only single-Name target | `numpyto_common/numpy_desugar.py` | landed |
| reduction method on a Call receiver (`np.abs(...).sum()`) | ls3df_scf | hoist Call receiver into temp before method rewrite | `numpyto_common/lowering.py` | landed |
| computed index in subscript (`U[np.argmax(...), j]`) | rayleigh_ritz_rotation | hoist non-trivial Call index into temp Name | `numpyto_common/lowering.py` | landed |
| arg-reduction over a computed operand (`idx = np.argmax(np.abs(v))`) | native capability (assignment-RHS sibling of the subscript-index form) | add `argmax`/`argmin` to the reduction-operand hoist set so the non-Name operand spills to a `__cb` temp before the arg-reduction scaffold (which requires a Name) runs | `numpyto_common/lib_nodes.py` `LibNodeRewriter.visit_Call` | landed |
| whole-array simultaneous rebind (`X,Y,sigma = Y,Ynew,sigma_new`) | chebyshev_filter_subspace, ls3df_scf | `_TupleAssignRewriter` copy-through temp buffers (not pointer-swap) | `numpyto_common/lowering.py` | landed |
| **index normalization** — chained / ellipsis / trailing subscript → canonical Name-base full-`Tuple` (`A[f][...,0]` → `A[f,...,0]`) | ls3df_scf, fragment_patch_density | normalize the index BEFORE libnode-expand so one code path handles every subscript form | `numpyto_common/lowering.py` `_lp_normalize_index_access` | landed |
| `np.meshgrid(..., indexing=)` multi-output | ls3df_scf | `expand_meshgrid` + multi-output tuple-unpack hoist | lib_nodes + lowering | planned |
| `np.ix_` open-mesh gather / scatter-add | fragment_patch_density, ls3df_scf | new advanced-index lowering to nested loops | lowering (+lib_nodes) | planned |

### 1b. Kernel-side faithful refactors (when the construct is genuinely un-static)

Some constructs are not a general translator capability worth adding; we instead rewrite the
kernel to a **bit-identical** translator-friendly form (verified `max|Δ|=0`).

| Kernel | Construct removed | Replacement | Status |
|---|---|---|---|
| ls3df_scf | Python-list Lanczos accumulators (`alphas=[]`/`.append`) | preallocated `np.zeros(_NLANC)` + integer counters | landed (bit-identical S+M) |
| ls3df_scf | `b_frag=[None]*n` None-cache | `np.zeros(n)` + boolean valid-mask (preserves freeze-on-first-iter; NOT eager recompute — the bound is intentionally frozen while V_tot drifts) | landed |

> Open follow-up: `np.diag(alphas[:na])` uses a runtime-counter slice (early break on
> `beta<1e-12`). If the runtime-length slice won't lower to static C/Fortran, switch to
> always running the full `_NLANC` steps → static `_NLANC×_NLANC` tridiagonal; the zero
> tail decouples so `eigvalsh(...).max()` is numerically identical.

### 1c. Emit-shape desugarings that help *pluto* emit correct code (planned)

These do **not** change our C's correctness (plain-C stays bit-exact); they change the *shape*
of the scop so `pet`/`pluto` stops miscompiling it. Gated on the pluto backend where noted.

| Fix | Clears | Mechanism | Location | Status |
|---|---|---|---|---|
| #1 `np.pad` edge-clamp → `max(0, min(d-1, s))` | stencil_3d, stencil_4d, stencil_4d_vc, vector_stencil_4d, vector_stencil_4d_vc | replace two guard-`if`s (pet: "data dependent conditions not supported" → 159 empty stmts → out_grid all-zeros) with a single min/max clamp keeping the subscript a bare name | `numpyto_common/lib_nodes.py` `_remap` edge branch | planned |
| #2 non-unit-stride loop → unit counter + affine induction | tsvc_2_s116 (+probe unrolled_dense, reroll_saxpy7, strided tsvc) | when `self.pluto` and `abs(step)!=1` constant, emit `int64 v=lo+step*__piv;` over a unit `__piv` (pet models `i+=4` as unit stride → wrong indices) | `numpyto_c/emit.py` `_emit_for` | planned |
| #3 scalar full-reduction → accumulate into destination element | lda_xc_potential (+likely ecrad_clamped_reduction, quasi_affine_reduce_*, atax-class) | retarget `float(np.sum(...))` temp to `out[0]` when it has a single downstream array-element store (pet drops the scalar `__cb=0` init+accum → uninit read) | lib_nodes / numpy_desugar | planned |

### 1d. JAX compile-time heuristics (help XLA emit faster) (planned)

Root cause: the oracle exercises the **eager** path (`numpyto_jax/core.py` `_emit_eager_body`),
which copies Python control flow *verbatim* — every static loop unrolls to trip-count distinct
XLA primitives (first-call compile cost) and trip-count sequential dispatches (per-call cost).
A mature loop classifier (`_classify_for` → VECTORIZE/FORI/WHILE) already exists but is only
reached on the dormant jit path. Route eager emission through it.

| Heuristic | Trigger | Emit | Win | Status |
|---|---|---|---|---|
| **H1** vectorize independent elementwise/stencil loops | `_classify_for==VECTORIZE` (write-once `a[i]=f(...)`) | whole-array op via existing `_devectorize_index` | removes recurring `.at[i].set` dispatch; kills large-preset `skip:too-long` | planned (first PR) |
| H2 re-roll large static carry loops | static `range`, trip≥8, FORI | `lax.fori_loop` (body compiled once) | O(trip)→O(1) first-call compiles | planned |
| H3 `lax.scan` for stacked carry-recurrence | FORI + monotone `out[i]=` slot | `lax.scan` | fewer scatters, better fusion | planned |
| H4 cap unroll to small (<8) static loops | complement of H2 | keep verbatim unroll | guard rail (small loops fuse cheaply) | policy |

`skip:too-long` = jax fork exceeds `JAX_FORK_TIMEOUT_S=180` (`numerical_oracle.py:50`); e2e retries
once at reduced `_JAX_E2E_MAX_SIZE`. It is a **perf** signal, not a correctness FAIL — jax is
verified correct at small size. (The concrete list of currently-skipped kernels is being swept.)

### 1e. LS3DF driver landed micro-fixes (dtype + shape folding)

Small shared-frontend fixes the LS3DF family forced out; each is bit-exact and helps every native
backend. All **landed**.

- **array-level `.real` / `.imag` / `creal` dtype narrowing** — an array result of `ifftn(...).real`
  narrows the *array* dtype to real (`double*`, not `double _Complex*`), not just scalars. C
  tolerated the implicit complex→real narrow (imag≈0); C++ `-std=c++20` refused it. Extends the
  `_fix_real_scalar_dtypes` / `_walk_complex` / `_REAL_FOR_COMPLEX` machinery to arrays
  (`numpyto_common/lowering.py`).
- **`.shape` / `.size` on a newaxis / subscript base** folded via `_iter_extent_of` (so
  `x[:, None].shape` / `A[f].size` resolve without a Name base).
- **`np.fft.fftfreq` / `fftn` two-level attribute shapes** resolved for the `fft.*` result temps.
- **tuple-local propagation** (`shp = Y.shape` then `shp[0]`) — the shape tuple flows through the
  local so later subscripts of it fold.
- **method-form `.reshape` / `.T` on a non-Name base** (`A[f].T`, `expr.reshape(...)`).
- **`np.eye` inline hoist** (identity materialized as a fresh local).
- **per-call-unique `linalg.inv` buffer** — each `inv(...)` gets its own scratch so two live
  inverses don't alias.
- **`expand_copy` allocation marker** — a whole-array copy target auto-declares its local.
- **mutated-`__inl*`-counter exclusion** — an inlined-call counter that is mutated is not treated
  as a foldable constant.
- **compound-token `.size` → BinOp** — a `.size` over a multi-axis base lowers to the product BinOp.

---

## 2. Backend tool bugs (external, not our lowering)

**Pluto verdict (root-caused live):** for every `::pluto` failure our emitted C is **bit-exact vs
numpy** (`run_kernel(..., only_backends={'c'})` → `ok`). `polycc` accepts the affine scop (RC=0)
then silently miscompiles. So *pluto is not failing because of our lowering.* Of 45 pairs: 3 are
correct non-affine skips, ~8–12 are sidesteppable by an emit-shape change (§1c), and the rest are
irreducible tool defects that now auto-classify as `skip:unsupported:pluto-miscompile` (c-ok guarded).

| Signature | Representative kernels | Root cause | Verdict | Disposition |
|---|---|---|---|---|
| non-affine indirection | edge_laplacian, unrolled_indirect, reroll_gather | data-dependent index; outside polyhedral model | correct-skip (not a fail) | `skip:unsupported` |
| pad edge guard-`if` rejected → out_grid all-zeros | stencil_3d/4d/4d_vc, vector_stencil_4d/_vc | `pet_to_pluto.cpp:565` "data dependent conditions not supported" | **ours (emit-shape)** | fix #1 → `ok` |
| non-unit stride mismodeled | tsvc_2_s116 (+probe) | pet models `i+=4` as unit stride | **ours (emit-shape)** | fix #2 → `ok` |
| scalar full-reduction init+accum dropped | lda_xc_potential (+likely ecrad, quasi_affine_reduce_*) | pet drops `__cb=0` init + accumulation → uninit read | **mixed (emit-sidesteppable)** | fix #3 → `ok` |
| reverse-loop double-negation OOB crash (SIG6/11) | adi, thomas_solve | pluto schedules on `−j`, emits subscript `- -t`=−j → heap OOB; textbook `for(j=N-2;j>=1;j--)` breaks identically | **pluto bug** | auto-skip (pluto-miscompile) |
| skew hyperplane int64 overflow | hotspot | 32-bit tile bound with ~2⁶² literal; "numerator too large" | **pluto bug** | auto-skip (pluto-miscompile) |
| smartfuse INT64_MAX-sentinel × symbolic bound | **kleinman_bylander_nonlocal** | CLooG emits `floord(nstate+9223372036854775807*ngrid-1,32)` → int64 overflow → band-0 loop `for(t3=0;t3<=-128)` never runs → output all-zeros; `--nofuse` bit-exact (2e-11) | **pluto bug** | auto-skip (pluto-miscompile; irreducible) |
| statement-drop / double-free (SIG6) | deriche, nussinov | pet/pluto codegen defect on valid affine input | **pluto bug** | auto-skip (pluto-miscompile) |
| transformed-C fails to compile | durbin | pluto emits invalid C | **pluto bug** | auto-skip (pluto-miscompile) |
| loop-carried tsvc (not individually root-caused) | ~14 tsvc_2_* | provisional pluto miscompile | **pluto (provisional)** | auto-skip (pluto-miscompile); probe for stride/reduction shape (§1c) to recover `ok` |

**pluto (Reading-B) irreducible set cannot be emptied by any lowering change** — and no longer
needs to be. With `e2e_known_failures.txt` gone, `_run_pluto` classifies every post-transform
`FAIL:*` whose sibling `c` backend is `ok` as `skip:unsupported:pluto-miscompile:*` (the guard:
our own C proves the affine scop bit-exact, so the fault is polycc's schedule). So the entire
"pluto bug" **Disposition** column above collapses to that one automatic skip — the table stays as
the root-cause record, no per-kernel list to maintain. The guard keeps it honest: a genuine emit
regression also reds `c`, so that pluto pair stays a real `FAIL:*`. The `*::pluto` rows still worth
an emit-shape fix (§1c) remain flagged there; landing one turns the skip back into an `ok`.

---

## 3. Gate semantics (strict-green) & the former LS3DF slice

There is no xfail file. Each `(kernel, backend)` pair resolves to exactly one of:

- **`ok`** — translated + bit-exact vs the kernel's numpy → passes.
- **`skip:*`** — the backend/tool legitimately can't express this kernel → skipped, not counted
  against the gate. Sub-reasons: `skip:not-installed` (tool absent), `skip:unsupported:*` (an op /
  scop the backend can't express), `skip:too-long` (jax compile-time perf, §1d), and
  `skip:unsupported:pluto-miscompile:*` (polycc miscompiled an affine scop our `c` proves correct —
  §2).
- **`FAIL:*`** — a real codegen/correctness gap → **reds the build**.

The old §3 table listed the LS3DF pairs that were then xfail-tolerated. Every native one is now
resolved — all 8 LS3DF stems are `ok` on c / cpp / fortran — and the pluto ones auto-skip. Verified
with `run_kernel(...)`:

| Former xfail entry | Was | Now |
|---|---|---|
| chebyshev_filter_subspace::{c,cpp,fortran} | whole-array buffer swap | **ok** — §1a rebind (landed) |
| fragment_patch_density::{c,cpp,fortran} | einsum non-Name + `np.ix_` scatter | **ok** — einsum + index path (landed) |
| rayleigh_ritz_rotation::{c,cpp,fortran} | argmax-in-subscript | **ok** — §1a computed-index hoist (landed) |
| ls3df_scf::{c,cpp,fortran} | eigvalsh + diag + fftfreq + list-refactor | **ok** — §1a/§1b/§1e combined (landed) |
| lda_xc_potential::pluto | pet drops scalar-reduction init+accum | `skip:unsupported:pluto-miscompile:exc:*` (emit-shape fix #3, §1c, would restore `ok`) |
| kleinman_bylander_nonlocal::pluto | pluto smartfuse int64 overflow | `skip:unsupported:pluto-miscompile:hpsi:*` (irreducible pluto bug) |

---

## 4. How to extend this doc

When you add a desugaring: add a §1 row (op, kernels, mechanism, file:line, status). When you
root-cause a backend miscompile: add a §2 row with the decisive evidence (the tool error string
or the diverging output) and the verdict (ours vs tool). When a pair's classification changes
(`FAIL:*` → `ok`, or `FAIL:*` → `skip:*`): update the §3 slice so the gate and this doc never
drift — there is no `e2e_known_failures.txt` to sync any more; the classification now lives in
`tests/numerical_oracle.py`.
