# OptArena backlog

Tracked work that is scoped but not yet done. Newest on top.

## Translator capability gaps — empirically verified 2026-07-04

Each claimed gap was probed with a minimal kernel through c/cpp/fortran via the
op-oracle (`run_op`). Most catalogued "gaps" are DEAD `raise`s in secondary
paths shadowed by the real lowering — do NOT re-implement those. The ones that
ACTUALLY fail today:

- **`np.pad` reflect / wrap / symmetric** (c/cpp/fortran): only `constant` /
  `edge` lower to a native pad; other modes raise. (numba/pythran already inline
  an edge-pad copy nest.) Pure index-remap → bit-exact-testable.
- **matrix `np.linalg.norm(A, 1)` / `(A, inf)`** (max col / row abs-sum): the
  VECTOR ord=1/inf forms now lower (fixed 2026-07-04); the matrix forms raise
  cleanly (a different reduction).
- **einsum ellipsis `...`**: raises by design
  (`test_parse_einsum_ellipsis_unsupported` asserts it); needs a rank-aware
  ellipsis expansion of the subscript before the contraction lowering.
- **axis-aware L2 norm `np.linalg.norm(x, axis=1)`**: PRE-EXISTING compile bug
  (`__cb1 undeclared`, from the `_expand_axis_reduction` post-fn temp); the
  full-reduction `norm(r)` and Frobenius forms are fine.

Confirmed ALREADY DONE (catalog was stale — do not re-implement): batched ≥3-D
matmul, `tril`, N-D `argmax`, einsum / tensordot / inner / vdot,
cumsum / cumprod / diagonal / trace / median, method `.reshape`, ellipsis
INDEXING (`a[..., 0]`), Frobenius / L2 norm.

NOT here — the OTHER chat owns it: strided slice-assign `out[::2] = a[::2]`
(their in-progress step-handling in `lowering.py`; currently miscompiles — leave
it alone). The naive-lowering and contraction-family sections below are DONE.

## Translator BUG: batched-einsum Fortran emit non-determinism (flaky, real)

`np.einsum('...ij,...jk->...ik', ...)` — and the explicit `'Bij,Bjk->Bik'`
batched-GEMM form — emit CORRECT c/c++, but the FORTRAN emit
non-deterministically declares a size symbol `REAL` instead of `INTEGER`
(`out(N, M, B)` → "Expression must be of INTEGER type, found REAL"), so ~40% of
runs fail to compile. The einsum LOWERING (`expand_einsum`) is deterministic
(dict/list, insertion-ordered); the non-determinism is in the Fortran emit's
size-symbol type resolution — hash / iteration-order dependent (a fixed
`PYTHONHASHSEED` is stable per-seed; it only surfaces in the c→cpp→fortran
multi-emit sequence of one `run_op` — ~40% of `pytest` runs). Pre-existing —
reproduces on the EXPLICIT form, so it is NOT the ellipsis expansion. NOTE: the
`_const` numpy-scalar coercion (emit `0` not `np.int64(0)`) is ORTHOGONAL and
does NOT fix this — the numpy scalar reaches the size-symbol classification via a
different path (the `_is_int_scalar`/`is_int_expr` gate in numpyto_fortran/emit.py
~L393, `isinstance(_, int)`). FIX: make the emit's size-symbol integer
classification order-independent (sort the symbol set, or type size symbols
int64 unconditionally). Relevant to the SeisSol batched-GEMM / tensor-contraction
kernels. Repro: einsum matmul via the op-oracle on c/cpp/fortran, ~5 pytest runs.

## FV3 dycore — remaining after the gt==4 dry core (assembled + validated)

`hpc/structured_grids/fv3_dycore/` has a FULL gt==4 dry nonhydrostatic dycore
assembled top-to-bottom (`fv_dynamics_gt4` -> `dyn_core_gt4` -> c_sw/d_sw/vertical
solvers -> leaves), 104 tests + 1 xfail; every leaf/sub-solver per-stencil
GT4Py-bit-exact, the two driver loops orchestration-validated. Remaining:
- `remap_profile` not bit-exact at bottom-edge layers (k>=nk-2, iv=1/kord<9) — the
  one xfail; fixing it makes `map_single` + the remap step bit-exact. (+ iv∈{-2,-1,0,2}, kord∈{9,10}.)
- MOIST path (do_sat_adj=True): `saturation_adjustment` (~1099 lines) + moist remap
  energetics + consv_te. (dry path is done.)
- **gt<3 cubed-sphere (production) path**: all spherical edge machinery (c_sw/d_sw
  spherical edges, a2b_ord4 corners, fill_corners, divergence_corner metric edges),
  do_zero_order sponge, higher-nord d_sw. gt==4 is the doubly-periodic TEST config;
  gt<3 is the real-Earth path = large further effort.
- physical-E2E-vs-real-pyFV3: BLOCKED in-sandbox (pyfv3 import >400s; no savepoint
  data) — needs a precompiled pyFV3 env or serialized state.

## Translator BUG: cloudsc flux-accumulation emits literal zero (HIGH, real)

WHY this is here: surfaced by the netcdf-faithful cloudsc init. It was DORMANT
under the old uniform-random init (which left these fields ≈0, so zero==zero
passed trivially) — exactly the kind of miscompile correctness-faithful data is
meant to expose. With the real ECMWF reference atmosphere, `cloudsc` c/cpp/fortran
all emit **literal zero** for the flux/tendency outputs `tendency_loc_q`,
`pfsqrf`, `pfsqsf`, `pfsqltur`, `pfsqitur`, `pfsqif` (numpy/cupy/jax compute them
nonzero; all three NATIVE backends agree exactly with each other → ONE codegen
bug, not fp reassociation). Location: the final flux-accumulation loop,
`cloudsc_numpy.py` ~lines 1084-1107 (a vertical prefix accumulation
`pfpl[k+1] = pfpl[k] + ...` into an OUT array). FIX, do NOT allowlist — it is a
real miscompile, and `norm_error` is reserved for fp reassociation, not a
literal-zero. Signature documented in the cloudsc NOTICE.md. Must wait until the
in-flight translator-extension work lands (no concurrent edits to lib_nodes.py /
emit.py).

## SeisSol ADER-DG microkernels (HPC track, microkernel)

Add two HPC-track **microkernels** from SeisSol's ADER-DG method (the
"long batched matmul with irregular/weird shapes" workload; see CPE 2024
doi:10.1002/cpe.8037, the generators SeisSol/gemmforge + SeisSol/TensorForge,
the user's yateto github.com/ThrudPrimrose/yateto + a reference PDF the
user provided as info ("Improved GPU Kernel Generation for SeisSol …" in
/home/primrose/Downloads, NOT characterised as their thesis)):

1. **`seissol_batched_gemm`** — the ADER-DG element-local "star" update.
2. **`seissol_tensor_contraction`** — the ADER-DG volume contraction.

PRIMARY INSTANCE = **convergence order 7** (Nb=84) — the most interesting point:
big enough that the per-element matrices are non-trivial / cache-relevant, small
enough to stay the realistic "many tiny batched GEMMs" regime SeisSol actually
runs (order 9 / Nb=165 may be added later as a second fixed instance, but order 7
is the headline). The matrix shapes are **fixed by the method order** (physics);
only the **batch = number of elements** scales, so the S/M/L/XL presets vary
ONLY the batch:

| preset | batch (elements) | working set (fp64, order 7) | role |
|---|---|---|---|
| S  | 1024    | ~12 MB   | smoke / CI |
| M  | 16384   | ~190 MB  | medium |
| L  | 131072  | ~1.5 GB  | publication |
| XL | 524288  | ~6 GB    | GPU-scale (≥4 GB, DRAM/HBM-bound) |

(per element = I + Q = 2·Nb·9·8 B ≈ 12 KB at order 7; the shared `star`/`kDivM`
are one copy, negligible.) Order ∈ {7(,9)} is a discrete SET, NOT an interval;
batch is the size knob.

DIMENSIONS & WHY THE SHAPES ARE WHAT THEY ARE (order 7, Nb=84, nQ=9):
- **Nb = 84** = number of 3-D modal basis functions of a tetrahedral DG element at
  order O=7: `Nb = O·(O+1)·(O+2)/6 = 7·8·9/6 = 84`. ADER-DG stores each element's
  solution as Nb modal coefficients per quantity — so every per-element operand is
  `Nb × nQ` (84×9). (Leading dim padded to a multiple of 32 → 96, the vector-unit
  alignment gemmforge/TensorForge apply.)
- **nQ = 9** = the elastic quantities: 6 independent stress components + 3 velocity
  components. The PDE couples them through the flux Jacobian.
- **`star` 9×9, sparse (24/81 nnz)** = the elastic flux Jacobian A* (the
  "star matrix"): a constant-per-element (for constant material) 9×9 that couples
  stresses↔velocities; most of the 81 entries are structurally zero (24 nonzeros).
- **`seissol_batched_gemm` = `Q[b] += I[b] @ star`**, (M,N,K) = **(84, 9, 9)**,
  batched over elements b. This is the canonical "weird/tall-skinny" SeisSol shape:
  M=84 (huge, the basis dim) but N=K=9 (tiny, the quantity dim) — a tall-skinny
  batched GEMM whose RIGHT operand (`star`) is tiny and SHARED across the whole
  batch, while `I`/`Q` are per-element (strided). The thesis's canonical 56×9·9×9
  is the order-6 instance of exactly this. numpy: `Q + np.matmul(I, star)`.
  (A companion shape is the stiffness GEMM `kDivM[84×84] @ I` = (84,9,84), the
  shared sparse stiffness×inverse-mass matrix applied to the per-element DOFs.)
- **`seissol_tensor_contraction` = the volume update**, einsum
  **`'dkl,blq,dqp->bkp'`**: sum over the 3 spatial dims `d`, the basis index `l`,
  and the quantity index `q`. `kDivM (3,84,84)` (the 3 stiffness matrices, shared,
  sparse), `star (3,9,9)` (the 3 directional Jacobians, shared, sparse), `I,Q
  (b,84,9)` (per-element). It's the natural rank-3 form yateto decomposes into the
  loop-over-GEMM above; a simpler single-binary alternative is `'bikl,k->bil'`.

ATTRIBUTION: the kernels are the element-local operators of **SeisSol**'s ADER-DG
seismic-wave solver (github.com/SeisSol/SeisSol; basis/quantity/star definitions
in `codegen/kernels/aderdg.py`, `equations/elastic.py`, `matrices/star.xml`). The
batched-tiny-GEMM framing and the loop-over-GEMM / sparse-tensor decomposition are
from the user's **yateto** (github.com/ThrudPrimrose/yateto) and the SeisSol
code generators **gemmforge** + **TensorForge** (github.com/SeisSol/*), and the
fused-batched-GEMM method paper Dorozhinskii et al., *Concurrency and Computation:
P&E* 36(12), 2024, doi:10.1002/cpe.8037. Concrete order-6 example shapes
(56×9·9×9, etc.) and the loop-over-GEMM/sparse rationale are in the reference PDF
the user provided ("Improved GPU Kernel Generation for SeisSol …", in ~/Downloads).
Each benchmark header must credit SeisSol + yateto + the paper.

BLOCKED on the translator supporting **batched >=3-D matmul + einsum/tensordot**
(below).

## graupel (ICON aes cloud microphysics) — full Fortran->numpy port (HPC, proxy)

Source in dace-fortran: `tests/icon/graupel/aes_graupel/mo_aes_graupel.f90`
(`graupel_run`, lines 114-632 ≈ **518 lines** of kernel; module 1521 lines incl.
init/finalize/lookup tables) + caller `graupel_caller.f90` (random data gen +
run wrapper) + `test_aes_graupel_numerical_correctness.py` (gfortran-ref vs SDFG).

Signature: `graupel_run(nvec, ke, ivstart, ivend, kstart, dt, dz, t, p, rho,
qv, qc, qi, qr, qs, qg, qnc, prr_gsp, pri_gsp, prs_gsp, prg_gsp, pflx, pre_gsp)`.
Dims: gridcell fields `(ivec, ke)` = dz,p,rho (IN), t (INOUT), qv/qc/qi/qr/qs/qg
(INOUT), pflx (OUT); column fields `(ivec)` = qnc (IN), prr/pri/prs/prg/pre_gsp (OUT).
dt scalar; nvec/ke/ivstart/ivend/kstart int.

RANDOM DATA GENERATION (from `init_graupel_inputs_c`): an LCG / Mulberry32-style
scramble `s = s*1664525 + 1013904223; r = (s>>16 & 32767)/32768`, seeded off an
int `seed` (same routine seeds ref + SDFG so they match). The committed regime is
**warm + dry NO-OP**: only `dz = 100 + 400*r` (m) is random; `t=290 K`, `p=80000`,
`rho=1`, `qv=qc=qi=qr=qs=qg=0`, `qnc=1e8`. That keeps graupel on its no-op path
(no condensation / ice / terminal velocity) so the SDFG e2e gate is just
"reproduce the no-op outputs". Test sizes: `ivec=4, k_v=8, dt=30, seed=42`.

PORT PLAN (mirror vexx/lulesh full port):
1. Inline `graupel_run` via the dace-fortran fparser inliner → single TU; SDFG →
   C++ (FaCe DaCe), SoA layout. (graupel already has inline/build tests there.)
2. `graupel_numpy.py`: SoA numpy port of `graupel_run`, vectorised over (ivec,ke),
   faithful to the Fortran — `np.where` for the IF branches, `np.exp/np.log`,
   `np.maximum/minimum`, the PARAMETER tables (params_qr/qi/qs/qg) as constants.
   Watch for translator gaps: any **lookup-table interpolation** (`np.interp` is
   NOT yet supported → lower naive) and the `dmin_wetgrowth` spline table.
3. `graupel.py` `initialize(...)` replicating the LCG + regime (start with the
   committed NO-OP regime for a clean first gate; add an EXERCISING cold/moist
   regime as a 2nd preset so the microphysics paths actually run). `graupel.yaml`
   presets sizing (ivec,ke); dt + ivstart/ivend/kstart params (kind: microapp/proxy).
4. `test_reference.py`: compile the DaCe C++ (dace headers via `find_spec`, as
   vexx/lulesh) + ctypes; compare numpy vs C++ on identical seeded inputs.
COMPLEXITY: ~518-line microphysics with many branches + transcendentals + param
tables — bigger than velocity_tendencies, on the order of cloudsc; the no-op regime
makes first validation tractable (most branches skipped).

## Translator: contraction family (einsum / matmul / tensordot / inner / vdot)

Forward-looking (no current `*_numpy.py` uses them, but SeisSol will). All lower
to plain loops + a sum reduction => **no emitter changes**; pure `lib_nodes.py`
lift-as-node work reusing `_scalarize_at_iters` / `_iter_extent_of`. ~3-4 days;
einsum is the keystone, the rest are thin wrappers:

- **matmul call-form** (XS): register `("np","matmul")` -> existing `expand_matmul`.
- **batched >=3-D matmul** (S): extend `expand_matmul` (today 2-D only, rejects at
  lib_nodes.py:1324) to wrap the i/j/k nest in outer batch loops (`_iter_extent_of`
  already derives the batch extent). REQUIRED by the SeisSol microkernels.
- **einsum core** (M-L): subscript parser -> free indices=output loops, repeated=
  summed inner loops; scalarize each operand; special-case
  matmul/trace/transpose/diagonal/outer/sum to reuse existing expanders.
- **tensordot / inner / vdot** (S each): thin wrappers building an einsum subscript
  (vdot = flatten + conj-dot; conj already supported).

## Translator: naive-lowering ops (trace / cumsum / diagonal / median)

Lower to naive implementations (user-approved):
- **trace** -> `acc += A[i,i]`; **diagonal** -> `out[i]=A[i,i]` (build: `out[i,i]=v[i]`).
- **cumsum / cumprod** -> sequential prefix-scan loop `out[i]=out[i-1] ∘ a[i]`
  (axis-aware: kept axes outside, scan axis inside).
- **median** -> sort the (copied) data + pick middle / mean of the two middles
  (needs a small emitted sort routine; `percentile/quantile` follow once sort lands).

## Translator: lulesh advanced-indexing gaps (HIGH)

Blocking lulesh's NumpyToX auto-emit (currently allowlisted in
`tests/e2e_known_failures.txt`):
- **method `.reshape(a, b)`** (varargs/tuple) -> normalize to `np.reshape(x,(a,b))`
  (`_match_reshape` already parses it; only the FFT-grid rewriter consumes it today).
- **2-D fancy gather `xe[:, idx]`** (slice axis + N-D index array -> higher-rank
  result): extend `_WholeArrayAssignRewriter._expand` / `_SubscriptifyNames`.
- **ellipsis `...`** indexing: a `_ExpandEllipsis` pre-pass using the array rank.
  (`tril` is done: `expand_tril` at lib_nodes.py:3476, registered at :4764.)

## YAML style: stencil manifests missing final newline / header

`tests/test_yaml_style.py::test_all_owned_yaml_conforms` fails on a branch that
carries the structured-grid stencil benchmarks (not present in every checkout):

- `hpc/structured_grids/stencil_3d/stencil_3d.yaml` — no final newline
- `hpc/structured_grids/stencil_4d/stencil_4d.yaml` — no final newline
- `hpc/structured_grids/stencil_4d_vc/stencil_4d_vc.yaml` — no final newline
- `hpc/structured_grids/vector_stencil_4d/vector_stencil_4d.yaml` — no final newline
- `hpc/structured_grids/vector_stencil_4d_vc/vector_stencil_4d_vc.yaml` — missing `#` header on line 1 + no final newline

Fix: on the branch that has these files, run `python tests/check_yaml_style.py --fix`.

## Wire scalar/flag fuzzing into `initialize` (velocity_tendencies)

`optarena/fuzz.py` now samples discrete sets (`{set: [...]}`) as well as `[lo,hi]`
intervals, but a sampled value only reaches a legacy-`func_name` kernel if the
param is in `init.input_args`. To actually fuzz velocity_tendencies' mode
switches:

- Add a `fuzzed` preset to `velocity_tendencies.yaml` `parameters:` with the size
  ranges plus `istep: {set: [1, 2]}`, `lvn_only: {set: [0, 1]}`, `lextra_diffu: 0`
  (extra diffusion is always off for this kernel; `ldeepatmo`, `l_vert_nested`,
  `ddt_vn_cor_associated` stay fixed too).
- Make `initialize(...)` accept `istep`, `lvn_only`, `lextra_diffu` as optional
  args (defaulting to today's values) and list them in `init.input_args` so the
  sampled values flow through.

## vexx oracle: emission has uninitialized transients + regen unreliable

The macrokernel oracle harness (`tests/macrokernel_oracle.py` + `tests/test_velocity_oracle.py`)
is proven end-to-end on velocity (emitted C++ == numpy, bit-for-bit). For vexx it
drives the 173-arg kernel end-to-end (init/program/exit all run), but two issues
block validation:

1. **Uninitialized transients — FIXED.** The committed
   `vexx_bp_k_gpu_generated.cpp` was lowered from an SDFG that flags `Use of
   uninitialized transient "hpsi_d" / "psi_d"`, so the closing `hpsi = hpsi_d`
   read garbage and even the no-op path corrupted `hpsi`. Patched directly in the
   emitted C++ (a clearly-marked MANUAL FIX block right after the allocations)
   that captures the dropped `SOURCE=hpsi` / `SOURCE=psi` copies; the no-op drive
   now returns identity. The numpy port already had the right semantics (in-place
   `hpsi +=`), now documented. (A clean regeneration should fold this in so the
   patch isn't needed -- see below.)
2. **Regeneration stalls.** Re-emitting under py13/FaCe (`make_builder` +
   `merge_engine="regex"`; fparser inliner fails with `cannot find invfft_y`)
   parses + builds + validates the SDFG but the codegen step stalls/dies silently
   on this 2k-line kernel (tried `UCX_VFS_ENABLE=n`, 25-min budget).

Status: the collinear oracle drive now runs end-to-end (numpy SoA port +
emitted C++ both produce `hpsi` on consistent inputs). Fixes applied to the
emitted C++ (all clearly-marked MANUAL FIX blocks): `hpsi_d`/`psi_d`
`SOURCE=` init, and `dfftt__nl` (= `dfftt%nl_d`, the device FFT-grid map)
initialized from the input `dfftt_nl` -- all three were `new`'d-but-uninitialized
transients the SDFG lowering dropped.

**Remaining gap = the emitted FFT lowering looks defective.** The dace-fortran
FFT library node lowered every `fwfft`/`invfft` to the *same* explicit DFT with a
POSITIVE exponent and NO normalization (`out[k]=sum_n in[n] exp(+2pi i k n/N)`),
so `fwfft o invfft` scales by `N` and reverses instead of being the identity. The
numpy port (physically-correct QE convention: normalized `invfft`, unnormalized
`fwfft`, 3-D grid) differs from the emitted C++ by exactly `nrxxs^2` on the
collinear config. Rather than replicate a buggy lowering in numpy, the fix is a
clean emission with correct FFT normalization (ties into the regen item); then
the harness (`tests/macrokernel_oracle.py`, already proven on velocity) validates
the SoA port bit-for-bit. Repro: `scratchpad/vexx_collinear_drive.py`.

## Macrokernel oracle parity (generated-C++ vs numpy)

The dace-fortran-generated C++ for both kernels is staged, `dace::`-stripped via
`baseline/dace_shim.h`, and compiles standalone. Remaining:

- **velocity_tendencies**: add a unit test driving `baseline/velocity_tendencies_generated.cpp`
  (via its `__dace_init`/`__program`/`__dace_exit` entry points, all offsets 0)
  and the numpy port on identical flat-SoA inputs; assert every output equal.
- **vexx**: write the FULL numpy port of `vexx_bp_k_gpu` (all branches:
  noncolin / okvan / okpaw / tqr / negrp; helpers addusxx_g, newdxx_g,
  g2_convolution, qvan, add_nlxx_pot) from the canonical source
  `dace-fortran/tests/qe/exx_bp/ast_v1_vexx_bp_k_gpu.f90`, flat-SoA signature,
  clean physical names (not DaCe connector names), then the same generated-C++
  vs numpy comparison test.
- **cloudsc**: confirm `cloudsc_numpy.py` is a complete port of its Fortran source.

## Upstream: dace `ipow` exponent-0 bug

Fixed in this workspace's `Work/dace` (`runtime/include/dace/math.h`,
`ipow(a,0)` now returns 1) with a regression assert in `tests/runtime_test.cpp`.
File the fix upstream against the dace repo.
