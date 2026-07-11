# Translator Desugarings & Backend Tool Bugs

Living ledger for the numpy→{C, C++, Fortran, numba, pythran, jax, pluto} translators
(`optarena/numpy_translators/`). Two intertwined things are tracked here:

1. **Desugarings / emit-helpers we add** so a backend can express a kernel it otherwise
   rejects, or so an external tool (pluto, XLA) emits *correct / faster* code.
2. **Backend tool bugs** — defects in external tools (`polycc`/`pluto`/`pet`/CLooG, XLA)
   that we cannot fix in our lowering, with the representative kernels and the sanctioned
   disposition (allowlist vs reclassify-skip).

The e2e gate (`tests/test_e2e_numerical.py`) translates each kernel to every backend and
compares against the kernel's own numpy. Pairs that legitimately can't pass are listed in
`tests/e2e_known_failures.txt` (xfail-tolerated). This doc is the *why* behind that file:
every entry should map to a row here.

> Rule of thumb: **we never paper over a tool miscompile with a tuning flag** (see
> `tests/numerical_oracle.py` ~L1092). If our emitted C/Fortran is bit-exact vs numpy and the
> tool still produces wrong output, that is a tool bug — allowlist or reclassify as a skip,
> do not mutate emit just to placate the tool unless the change is a legitimately better shape.

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
| `np.linalg.eigvalsh(A)` (eigenvalues-only) | ls3df_scf (`_upper_bound`) | extend eigh cyclic-Jacobi, eigenvalues-only single-Name target | `numpyto_common/numpy_desugar.py` | in-progress |
| reduction method on a Call receiver (`np.abs(...).sum()`) | ls3df_scf | hoist Call receiver into temp before method rewrite | `numpyto_common/lowering.py` | in-progress |
| computed index in subscript (`U[np.argmax(...), j]`) | rayleigh_ritz_rotation | hoist non-trivial Call index into temp Name | `numpyto_common/lowering.py` | in-progress |
| whole-array simultaneous rebind (`X,Y,sigma = Y,Ynew,sigma_new`) | chebyshev_filter_subspace, ls3df_scf | `_TupleAssignRewriter` copy-through temp buffers (not pointer-swap) | `numpyto_common/lowering.py` | in-progress |
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

---

## 2. Backend tool bugs (external, not our lowering)

**Pluto verdict (root-caused live):** for every `::pluto` failure our emitted C is **bit-exact vs
numpy** (`run_kernel(..., only_backends={'c'})` → `ok`). `polycc` accepts the affine scop (RC=0)
then silently miscompiles. So *pluto is not failing because of our lowering.* Of 45 pairs: 3 are
correct non-affine skips, ~8–12 are sidesteppable by an emit-shape change (§1c), and the rest are
irreducible tool defects that must stay allowlisted.

| Signature | Representative kernels | Root cause | Verdict | Disposition |
|---|---|---|---|---|
| non-affine indirection | edge_laplacian, unrolled_indirect, reroll_gather | data-dependent index; outside polyhedral model | correct-skip (not a fail) | `skip:unsupported` |
| pad edge guard-`if` rejected → out_grid all-zeros | stencil_3d/4d/4d_vc, vector_stencil_4d/_vc | `pet_to_pluto.cpp:565` "data dependent conditions not supported" | **ours (emit-shape)** | fix #1 → prune |
| non-unit stride mismodeled | tsvc_2_s116 (+probe) | pet models `i+=4` as unit stride | **ours (emit-shape)** | fix #2 → prune |
| scalar full-reduction init+accum dropped | lda_xc_potential (+likely ecrad, quasi_affine_reduce_*) | pet drops `__cb=0` init + accumulation → uninit read | **mixed (emit-sidesteppable)** | fix #3 → prune |
| reverse-loop double-negation OOB crash (SIG6/11) | adi, thomas_solve | pluto schedules on `−j`, emits subscript `- -t`=−j → heap OOB; textbook `for(j=N-2;j>=1;j--)` breaks identically | **pluto bug** | allowlist (opt: oracle `skip:reverse-loop`) |
| skew hyperplane int64 overflow | hotspot | 32-bit tile bound with ~2⁶² literal; "numerator too large" | **pluto bug** | allowlist |
| smartfuse INT64_MAX-sentinel × symbolic bound | **kleinman_bylander_nonlocal** | CLooG emits `floord(nstate+9223372036854775807*ngrid-1,32)` → int64 overflow → band-0 loop `for(t3=0;t3<=-128)` never runs → output all-zeros; `--nofuse` bit-exact (2e-11) | **pluto bug** | allowlist (cannot fix in emit) |
| statement-drop / double-free (SIG6) | deriche, nussinov | pet/pluto codegen defect on valid affine input | **pluto bug** | allowlist |
| transformed-C fails to compile | durbin | pluto emits invalid C | **pluto bug** | allowlist |
| loop-carried tsvc (not individually root-caused) | ~14 tsvc_2_* | provisional pluto miscompile | **pluto (provisional)** | allowlist unless a probe shows stride/reduction shape |

**pluto (Reading-B) irreducible set cannot be emptied by any lowering change.** Sanctioned
options: keep in `e2e_known_failures.txt`, or add oracle-side detectors that reclassify a
recognizable family (e.g. decreasing loop header → `skip:unsupported:reverse-loop`) so it leaves
the xfail list as an honest skip rather than a tracked FAIL.

---

## 3. `e2e_known_failures.txt` ledger (LS3DF-relevant slice)

| Entry | Cause | Path to remove |
|---|---|---|
| chebyshev_filter_subspace::{c,cpp,fortran} | whole-array buffer swap | §1a array-rebind desugar (in-progress) |
| fragment_patch_density::{c,cpp,fortran} | einsum non-Name (landed) + `np.ix_` scatter | §1a ix_ (planned) |
| rayleigh_ritz_rotation::{c,cpp,fortran} | argmax-in-subscript (eigh `w,U=` form already lowers) | §1a computed-index hoist (in-progress) |
| ls3df_scf::{c,cpp,fortran} | eigvalsh + diag + fftfreq + meshgrid + ix_ + list-refactor | §1a/§1b combined |
| lda_xc_potential::pluto | pet drops scalar-reduction init+accum | §1c fix #3 → removable |
| kleinman_bylander_nonlocal::pluto | pluto smartfuse int64 overflow | **not removable** — pluto bug; keep allowlisted or reclassify skip |

---

## 4. How to extend this doc

When you add a desugaring: add a §1 row (op, kernels, mechanism, file:line, status). When you
root-cause a backend miscompile: add a §2 row with the decisive evidence (the tool error string
or the diverging output) and the verdict (ours vs tool). When you prune/append
`e2e_known_failures.txt`: update the §3 ledger so the file and this doc never drift.
