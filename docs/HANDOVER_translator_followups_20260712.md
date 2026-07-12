# Handover — numpy→X translator + e2e gate follow-ups (2026-07-12)

Scope of the session that produced this doc: drive the **LS3DF kernel family** to native
green on the numpy→{C, C++, Fortran} translators, audit the translators, and queue the
JAX / pluto / dedup / docs follow-ups. This file is the "what still needs fixing" list for
whoever picks up next.

Everything the session finished is **committed** (working tree was clean at handoff, HEAD
`2d578fe7`). The items below are the open ones.

---

## 0. State at handoff (what is DONE)

- **LS3DF family, all 8 stems GREEN on `c` / `cpp` / `fortran`:**
  `laplacian_stencil_3d`, `poisson_cg_3d`, `lda_xc_potential`, `fragment_patch_density`,
  `kleinman_bylander_nonlocal`, `rayleigh_ritz_rotation`, `chebyshev_filter_subspace`,
  `ls3df_scf`. Verify:
  ```bash
  cd optarena && rm -rf .dacecache .pytest_cache
  PYTHONPATH=. python -c "from tests.numerical_oracle import run_kernel; \
    print(run_kernel('ls3df_scf','S',only_backends={'c','cpp','fortran'}))"
  # -> {'c': 'ok', 'cpp': 'ok', 'fortran': 'ok'}
  ```
- The last cpp gap (`ls3df_scf` `FAIL:compile`) was root-fixed: array-level `.real`/`.imag`
  (and `creal`-derived) results now narrow the **array** dtype to real, not just scalars, so
  `v = ifftn(...).real + V_ion + xc` declares `double*` instead of `double _Complex*`. C
  tolerated the implicit complex→real narrowing (imag≈0, so `c` was already `ok`); C++
  `-std=c++20` refused it. Fix extends the existing `_fix_real_scalar_dtypes` /
  `_walk_complex` / `_REAL_FOR_COMPLEX` machinery in `numpyto_common/lowering.py` to arrays.

- **Gate mechanism changed under us (by a concurrent session):** the xfail
  `tests/e2e_known_failures.txt` file + its loader were **removed**
  (commits `5ed9b20f`, `411612f9`). The e2e gate is now **strict-green**:
  `ok` → pass, `skip:*` → skip, `FAIL:*` → **red**. `tests/test_e2e_numerical.py`
  drives `_ALL_E2E_BACKENDS = (c, cpp, fortran, numba, pythran, jax, pluto)`, subsettable via
  `OPTARENA_E2E_BACKENDS`.

---

## 1. HIGH — Pluto miscompiles vs. the new strict-green gate

**Problem.** The strict-green gate has no xfail tolerance, but `pluto` is in the default
backend set. `polycc`/`pet` **miscompiles** several valid, affine C kernels we emit
bit-exact. Those pairs return `FAIL:*` (numeric divergence / crash / compile), which now
**reds the gate wherever `polycc` is installed**. It only looks green on a CI runner that
lacks `polycc` (there every pluto pair degrades to `skip:not-installed`).

Confirmed still-failing locally (polycc present):
```
kleinman_bylander_nonlocal :: pluto  -> FAIL:hpsi:d=2.05e+02
lda_xc_potential           :: pluto  -> FAIL:exc:d=2.68e+01
adi                        :: pluto  -> FAIL:u:d=1.00e+00
durbin                     :: pluto  -> FAIL:compile
nussinov                   :: pluto  -> FAIL:crash:SIG6
```
plus the ~44 `*::pluto` pairs the removed allowlist used to carry (tsvc_2_s*, stencils, etc.).

**Assessment (already done, stands):** our emitted C is bit-exact vs numpy for these; the
`c` backend is `ok` on the same kernels. The divergence is entirely in polycc's polyhedral
schedule (a tool bug), NOT our lowering. We do **not** paper over a tool miscompile with a
tuning flag (`numerical_oracle.py` ~L1092).

**Decision needed (owner = whoever redesigned the gate):** since pluto is a tool and its
miscompiles are not our correctness bugs, `_run_pluto` (`tests/numerical_oracle.py` ~L1065)
should classify a numeric-divergence / crash / miscompile as
`skip:unsupported:pluto-miscompile` (a tool-can't-express signal) rather than `FAIL:*`,
**guarded by** "the `c` backend on the same kernel is `ok`" so a genuine emit regression
still reds. That keeps the strict-green gate honest AND green where polycc is installed.
Until then, run the gate with `OPTARENA_E2E_BACKENDS` excluding `pluto`, or on a
polycc-absent runner.

**Do NOT** just delete pluto from the gate — a subset of the generic `*::pluto` fails are
fixable in emit (§4), and pluto coverage is worth keeping for those.

---

## 2. HIGH — JAX compile-time heuristics (H1 + H2)

User intent (refined over the session):
- **H1 — vectorization / whole-array recovery, applied FAIRLY to every framework.** When a
  kernel is written as a Python `for` loop that is a pure elementwise / reduction map, emit
  the idiomatic whole-array form on *every* backend that has one — Fortran array intrinsics
  (`sum`, `matmul`, array sections) and numpy full-array ops — not just JAX. The classifier
  must live in `numpyto_common` (shared), so JAX gets no lower-complexity algorithm than the
  others (fairness invariant). Not yet implemented.
- **H2 — JAX `lax.fori_loop` for high-trip loops.** A Python `for i in range(n)` with a
  large static trip count blows up XLA trace/compile time when unrolled. Lower it to
  `jax.lax.fori_loop` instead. Trip count comes from the kernel `.yaml` preset × size; make
  the threshold a config `jax.fori_trip_threshold`, **default 100**; if the trip count is
  unresolvable, prefer `fori`. Not yet implemented.
- **H3 (scan) was DROPPED** — a scan lowering would be a genuine algorithm change and can't
  be given to every framework fairly.

**Ownership caveat:** the Python-target emitters (`numpyto_jax`, `numpyto_numba`,
`numpyto_cupy`, `numpyto_pythran`) transform numpy source as **text/regex**, not AST, and are
another chat's. Coordinate before editing them. H1's classifier is safe to build in
`numpyto_common`; H2 needs a change in the jax emitter path — sync with its owner.

---

## 3. MED — Enumerate the JAX `skip:too-long` set (feeds H2)

The user asked "which tests are skipped due to taking too long to compile." A jax fork that
exceeds `JAX_FORK_TIMEOUT_S` (180s, env `OPTARENA_JAX_FORK_TIMEOUT_S`) records
`skip:too-long` (`numerical_oracle.py` ~L43). This list is **not yet produced**.

- A full jax sweep over 335 kernels is too slow (each too-long kernel burns the full 180s).
- 296/335 kernels contain a `for … in range(…)`, so loop-presence is not selective.
- Recommended: scope by **static high-trip-count** (parse each `<stem>_numpy.py` for
  `range(...)` bounds, resolve against the `.yaml` preset, keep trip > 100), then run
  `only_backends={'jax'}` on just those in the background and collect `skip:too-long`. That
  candidate set is exactly H2's target list.

---

## 4. MED — Pluto emit-fixes (turn some `FAIL:*::pluto` into `ok`, legitimately)

These are cases where a **better emit shape** (not a tuning flag) lets polycc succeed:
- **pad-clamp**: emit `min`/`max` boundary clamps in a form pet models affinely.
- **non-unit-stride**: stride-by-k loops polycc currently rejects.
- **reduction-into-element** (drives `lda_xc_potential::pluto`): a reduction that writes a
  single array element.
- **argmax/argmin materialize whitelist**: `numpyto_common/lib_nodes.py` — the non-Name
  operand materialization set excludes `argmax`/`argmin` (noted out-of-scope by an earlier
  agent). Add them so a reduction over a computed operand hoists. (This one also helps native
  backends, not just pluto.)

Keep the "never placate a tool with a flag" rule: only change emit when the new shape is
independently a legitimately-better/idiomatic lowering.

---

## 5. MED — Dedup decisions (need a human call)

Duplicate/near-duplicate kernels found; each needs a keep/fix/drop decision:
- **`minres`** — currently a mislabeled copy of `cg` (conjugate gradient), not MINRES.
  Fix (make it real MINRES) or drop.
- **Foundation ↔ TSVC exact-duplicate pairs** — collapse one of each:
  `wavefront_2d` / `s2111`, `s4121` / `vpvtv`, `s431` / `vpv`, `s3110` / `s13110`.
- **`daubechies_dwt2d`** (Halide-derived, spectral_methods) vs existing **`dwt2d`** — keep
  both (different transform) or drop one.

---

## 6. LOW — Docs

- `docs/TRANSLATOR_DESUGARINGS_AND_TOOL_BUGS.md` is the living ledger. Update:
  - the `np.linalg.eigvalsh` row still says **in-progress** — it **landed**.
  - add the LS3DF driver's fixes: array `.real`/`.imag`/`creal` dtype narrowing;
    `.shape`/`.size` on a newaxis/subscript base folded via `_iter_extent_of`;
    `np.fft.fftfreq`/`fftn` two-level attribute shapes; tuple-local propagation
    (`shp = Y.shape`); method-form `.reshape`/`.T` on non-Name bases; `np.eye` inline hoist;
    per-call-unique `linalg.inv` buffer; `expand_copy` allocation marker;
    mutated-`__inl*`-counter exclusion; compound-token `.size`→BinOp.
  - **§3 "known-failures ledger" is now obsolete** — `tests/e2e_known_failures.txt` was
    removed; rewrite it to describe the strict-green semantics + the pluto disposition (§1).
  - add the **index-normalization principle**: chained/ellipsis/trailing subscripts are
    desugared to a canonical `Name`-base full-`Tuple` subscript in `_lp_normalize_index_access`
    before libnode-expand (`A[f][...,0]` → `A[f,...,0]`).
- **README bottom** — add the framework-desugar section the user requested, AFTER H1/H2 and
  the pluto disposition land (so it documents the final behavior).

---

## 7. LOW — Housekeeping (was blocked / awaits go)

- **Memory prune**: a `MEMORY.md` edit failed mid-session (a concurrent session rewrote the
  file). Re-attempt the prune of completed tasks.
- **Stale design/handoff `.md` docs**: several are other-workstreams' property
  (dycore / canon / nest-forge / dace-fortran). Surface, do not delete another workstream's
  handoff. The only one flagged safe-dead was dace-fortran's `MERGE_HANDOFF_to_other_chat.md`
  — confirm with its owner before removing.

---

## 8. Backlog — new kernels (researched, patent/GPLv3-vetted, NOT built)

All cleared: no patents, GPLv3-compatible-with-attribution only, clean-room.
- **Image processing** (dwarf placement noted for each): `block_idct_jpeg`
  (spectral_methods), `sobel_saliency` (structured_grids) as start-here. Excluded on
  license/patent grounds: SLIC (AGPL/TTO), SIFT/SURF, seam-carving, H.264/MPEG-4, VisionWorks.
  Halide-derived already landed: `harris_corner`, `max_filter`, `histogram_equalization`,
  `daubechies_dwt2d`.
- **MPI** (from the "Using Advanced MPI" clean-room extraction): `cart_halo2d`,
  `dist_gemv_allreduce`. Start from a **single-node reference** and define **weak-scaling**
  properties before adding ranks.
- **Foundation / compiler-opt** (TSVC-like): LCALS + a few LLVM loop-opt starters (PolyBench
  is already in).

---

## Quick reference — files owned by the native path (safe to edit)

`optarena/numpy_translators/src/`:
`numpyto_common/{lowering.py, lib_nodes.py, frontend.py, numpy_desugar.py}`,
`numpyto_c/emit.py`, `numpyto_fortran/emit.py`, plus `tests/numerical_oracle.py` for gate
classification. Do **not** edit `numpyto_{numba,cupy,pythran,jax}` (regex emitters, other
chat). All temp/diag scripts go in the scratchpad as real `.py` files run with
`PYTHONPATH=. python <file>` — never `python -c`.
