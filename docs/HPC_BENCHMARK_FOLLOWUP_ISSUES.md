# HPC benchmark follow-ups — failing kernels, root causes, and fixes

Status of the four HPC kernels from the benchmark-follow-ups branch
(`origin/hpc-kernel-extractions`), diagnosed against **current `main`** (the
branch predates a lot of `main`, so several symptoms differ from the original
report — e.g. what was "compile failure" is now an earlier "emit failure").

Reproduced by materializing each benchmark into the working tree (untracked, not
committed) and running the real numerical oracle
(`tests/numerical_oracle.py :: run_kernel`) at the `S` preset across
`c / cpp / fortran / numba / pythran / jax`.

Legend: **emit** = translator can't produce native source; **compile** =
gcc/g++/gfortran reject the emitted source ("native compile"); **correctness** =
compiles + runs but the result disagrees with numpy.

---

## Summary

| Benchmark | Verdict | Blocking issue |
|-----------|---------|----------------|
| **DBCSR** | ✗ won't-fix | Python-OO: classes, dicts, dict-comprehensions — out of scope |
| **XSBench** | ✓ **fixed** | Manifest lacked `init.shapes` + `init.dtypes` → declare them (all backends pass) |
| **GROMACS NBNxM** | ✓ **c/cpp/numba bit-exact** | Numerical all-zero-force bug **fixed** (int-cast truncation barrier); Fortran needs one more emit fix (ternary/merge kind); data-dependent pair list still needs a size symbol |
| **LavaMD** | ◑ structural | Needs shapes/dtypes; `particles_per_box` pinned to a module constant + product-dimension shape; JAX can't trace the data-dependent loops |

Two general **translator** gaps surfaced and were fixed on `main` (helping every
kernel, not just these):

- **`len(array)`** → the array's symbolic first-dim size (was emitted literally →
  C/C++ "implicit declaration of `len`", Fortran "`len` intrinsic needs
  CHARACTER"). Fixed in `_ShapeMidExpressionRewriter`.
- **Module-constant bit-ops** — `CI_DO_COUL = 1 << 1`, `0x1 | 0x2`, `~mask` — are
  now constant-folded by `_inline_module_constants` (previously only +,-,*,/,//,%,**).

---

## DBCSR — won't-fix (Python object-orientation)

`dbcsr` builds the block-sparse product with Python **classes**
(`DBCSRKernel`, `HashTable`, `ProductWorkspace`), Python **dicts**, and a **dict
comprehension** (`{block_id: np.ascontiguousarray(...) for block_id in ...}`).
The translator does not support classes / dicts / dict-comprehensions, so the
kernel fails at emit:

```
NotImplementedError: expression DictComp: {block_id: np.ascontiguousarray(...) for ...}
```

**Decision:** dropped. To be an OptArena kernel it would need a full rewrite into
flat array + explicit-loop form (no classes, no dicts). The
`row_offsets[1:] = np.cumsum(row_sizes)` partial-slice cumulative write flagged
earlier is a real but *secondary* gap — the object-orientation blocks it long
before that line is reached.

---

## XSBench — FIXED (declare `init.shapes` + `init.dtypes`)

**Symptom (emit):** `NotImplementedError: expression Tuple: (n_samples,)` — the
lowered body contained `__inl1_n_gridpoints = (n_samples,)[1]`, i.e. indexing a
**1-element** tuple at `[1]`.

**Root cause:** the manifest declares **no** `init.shapes`, and the translator's
shape recovery (`_shapes_from_initialize`) fails here for two reasons:

1. it scans a *companion* file (`xsbench.py`), but this benchmark's `initialize`
   lives **inside** `xsbench_numpy.py`; and
2. `initialize` only **delegates** (`return generate_random_xsbench_inputs(...)`),
   so the array allocations are one level deeper than the recovery inspects.

With no shapes, every input array defaulted to the 1-D `(n_samples,)`, so
`nuclide_grid.shape[1]` (nuclide_grid is really `(n_isotopes, n_gridpoints, 6)`)
lowered to `(n_samples,)[1]` → out of bounds.

The Fortran-only follow-on (`init.dtypes` also empty): `num_nucs` (an int array)
defaulted to REAL, and `do ... = 0, num_nucs(mat+1) - 1` gave
`Error: End expression in DO loop must be integer` (C tolerates a real loop
bound; Fortran does not).

**Fix (manifest — a benchmark follow-up):** declare both blocks. Verified: with
these, `xsbench` is `ok` on **c / cpp / fortran / numba / jax** (pythran skips).

```yaml
init:
  shapes:
    p_energy_samples: (n_samples,)
    mat_samples: (n_samples,)
    num_nucs: (n_materials,)
    concs: (n_materials, max_num_nucs)
    egrid: (n_isotopes * n_gridpoints,)
    index_grid: (n_isotopes * n_gridpoints, n_isotopes)
    nuclide_grid: (n_isotopes, n_gridpoints, 6)
    mats: (n_materials, max_num_nucs)
  dtypes:
    mat_samples: int32
    num_nucs: int32
    index_grid: int32
    mats: int32
    # (the rest default to float64)
```

---

## GROMACS NBNxM — partial (multiple layers)

Diagnosed in order; each fix exposed the next:

1. **`skip:no-output`** → same missing `init.shapes`/`init.dtypes` as XSBench
   (the `f`/`fshift` outputs couldn't be shaped). Declaring them lets it emit.
2. **`len(coulomb_table_f)`** in the compute kernel → literal `len(...)` in C.
   **Fixed on `main`** (`len` → first-dim size).
3. **`FAIL:unresolved:CI_DO_COUL`** — module-level bit-flag constants
   (`CI_DO_COUL = 1 << 1`, `FULL_EXCLUSION_MASK = 0xFFFF`). **Fixed on `main`**
   (bit-op constant folding).
4. **`FAIL:__inl1_f: d≈4.0` — every force came out zero. FIXED.** Root cause: the
   lowering DROPPED `int(x)` casts (relying on an int-declared target to truncate
   implicitly). That erased the barrier the used-as-int analysis needs: from the
   Coulomb-table index `ri = int(rs)` it walked BACKWARD across the (now bare)
   `ri = rs` into `rs`, then `rs = rsq*rinv*tab_coul_scale`, mistyping the whole
   distance chain (`rsq` / `rinv` / `dx`/`dy`/`dz`) as integer. Each sub-1.0
   coordinate difference truncated to 0 → `rsq=0` → `fscal=0` → zero forces. Fix:
   **`int(x)` is now KEPT** (rendered `(int)(x)` in C/C++, `INT(x, kind)` in
   Fortran — every native emitter already handled a bare `int(x)`), and the
   backward int-ness closure is bounded by `pure_int_arith`. Now **c / cpp /
   numba are bit-exact** (verified `max|f - f_numpy| = 0.0`). Regression test:
   `optarena/numpy_translators/tests/test_int_cast_truncation.py`.

**Remaining (real work):**

- **Fortran** now clears the shared int-typing bug plus two more that this kernel
  exercised — a bitwise flag nested in a comparison (`(flags & CI_DO_LJ) != 0`,
  whose `flags` operand of `IAND` must be INTEGER) and boolean SCALAR locals
  (`do_lj` / `do_coul` / `half_lj`) that were declared `logical` but wrapped
  `/= 0` at use sites. **One issue remains**: the ternary `ci_sh = ci if ish == 0
  else -1` emits `merge(ci, -1, ...)` where `ci` is int64 and the `-1` literal is
  int32 — a `merge` kind mismatch (the literal-branch kind promotion misses the
  sanitized inlined name). GROMACS is the only corpus kernel that hits this, so
  it is tracked separately; c/cpp/numba do not need it.
- **`cj_cluster` / `cj_excl` have a data-dependent length** (the total number of
  cluster *pairs*, `23` at `S`) — not a clean function of the manifest params.
  It needs a dedicated size symbol (e.g. `n_cj`) whose value the harness resolves
  from the buffer at run time; the shapes I used to get this far were
  `cj_cluster: (n_cj,)`, `cj_excl: (n_cj,)`.

**Shapes/dtypes used** (for reference; `x`/`q`/`atom_type` are `4` atoms per
cluster, `coulomb_table_f` is `table_size + 1`):

```yaml
init:
  shapes:
    x: (4 * n_clusters, 3)
    q: (4 * n_clusters,)
    atom_type: (4 * n_clusters,)
    nbfp: (num_types, num_types, 2)
    ci_cluster: (n_clusters,)
    ci_shift: (n_clusters,)
    ci_cj_start: (n_clusters,)
    ci_cj_end: (n_clusters,)
    ci_flags: (n_clusters,)
    cj_cluster: (n_cj,)          # data-dependent — total cluster pairs
    cj_excl: (n_cj,)
    shift_vec: (n_shifts, 3)
    coulomb_table_f: (table_size + 1,)
  dtypes:
    atom_type: int32
    ci_cluster: int32
    ci_shift: int32
    ci_cj_start: int32
    ci_cj_end: int32
    ci_flags: int32
    cj_cluster: int32
    cj_excl: uint16
```

---

## LavaMD — structural (shapes) + JAX can't trace it

**Init error:** `ValueError: particles_per_box must match NUMBER_PAR_PER_BOX`.
`NUMBER_PAR_PER_BOX` is a **hard-coded module constant (100)**;
`generate_random_lavamd_inputs` *raises* unless `particles_per_box == 100`. So
`particles_per_box` is not a free size — it is pinned to a literal, and the
particle arrays' leading dimension is the **product** `n_boxes * particles_per_box`
(e.g. `rv: (n_boxes * particles_per_box, 4)`). The oracle resolves each size
symbol from a *single* array dimension, so a product dimension containing a
symbol (`particles_per_box`) that never appears standalone can't be back-solved.

**Fix options (benchmark side):** either (a) introduce one derived symbol
`n_particles = n_boxes * NUMBER_PAR_PER_BOX` and shape the particle arrays
`(n_particles, 4)` — `n_particles` then resolves from `rv`'s own dim; or
(b) make `particles_per_box` a genuine parameter (drop the `NUMBER_PAR_PER_BOX`
hard-pin) so the product's factors are both known.

### Why JAX hangs (explicitly investigated)

Two distinct facts:

- **On current `main` it does not hang — it fails to emit:**
  `EmitError: bare expression statement (possible in-place op)` — from the bare
  `_validate_inputs(...)` call statement (JAX's eager emitter rejects bare
  expression statements). So today it is a clean `skip`, not a timeout.

- **The timeout you saw (older version, once it *did* emit) is inherent to the
  kernel, not a fixable translator bug.** `lavamd_kernel` has a
  **data-dependent loop bound** — `for k in range(1 + int(neighbor_counts[l]))` —
  and **data-dependent gather indexing** (`pointer = int(neighbor_list[l, k-1])`),
  wrapped around a dense `100 × 100` particle double-loop that does in-place
  scatter-accumulation into `fv`. JAX is functional and traces Python `for`
  loops by **unrolling** them; a data-dependent bound forces concretization and
  the unrolled trace becomes an enormous chain of `fv.at[...].add(...)` scatter
  updates (order 10⁶ at the small preset), which XLA cannot compile in bounded
  time → the fork-oracle hits its deadline and reports `skip:too-long`.

  This is the "known JAX while-loop / data-dependent" class: the only ways to make
  JAX handle it are to rewrite the kernel with `jax.lax.fori_loop` /
  `dynamic_slice` (a JAX-specific kernel, not a faithful NumPy port) or to accept
  the `skip`. The oracle already bounds it (SIGKILL + `skip:too-long`), so it does
  not stall CI.

---

## What changed on `main` (translator) vs. what's a benchmark follow-up

**Translator fixes (landed on `main`, general):**

- `len(array)` → symbolic first-dim size (all native backends).
- Module-constant **bit-op** folding (`<<`, `>>`, `|`, `&`, `^`, `~`, and
  Name-composition of earlier flags).
- **`int(x)` truncation is preserved** (C/C++/Fortran) instead of dropped, so
  the used-as-int analysis does not propagate int-ness backward across a
  truncation into a float source. Fixes GROMACS' all-zero forces; also removes
  a latent bug (`y = int(x) + 0.5` with a double `y` used to keep the fraction).
- **Fortran int-typing** backward closure bounded by `pure_int_arith` (shared
  with C via `numpyto_common.frontend.pure_int_arith`); `_walk_bitwise_operands`
  descends into `Compare` / `BoolOp` (a bitwise flag can hide in `(f & M) != 0`);
  boolean SCALAR locals are registered as `logical` so a bare use is not wrapped
  `/= 0`.

**Benchmark follow-ups (manifest / kernel — for the PR):**

- XSBench: add `init.shapes` + `init.dtypes` (**confirmed fully green**).
- GROMACS: add shapes/dtypes (with a `n_cj` size symbol) — then a native
  correctness bug + Fortran + the data-dependent pair-list still need work.
- LavaMD: restructure the particle-array shape to a single resolvable symbol; the
  JAX timeout is inherent (accept `skip` or write a JAX-native variant).
- DBCSR: drop, or rewrite dict/class-free.

**Module constants — resolved:** they are **inlined** (folded to their literal
value in the body); this is the accepted design. The bit-op fold extension above
(`<<`, `>>`, `|`, `&`, `^`, `~`, flag-composition) is all that was needed — named
`#define` / `constexpr` / `parameter` emission is explicitly **not** pursued.
