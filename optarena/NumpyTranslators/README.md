# NumpyToC

A Python → C99 / C++ / Pluto-input emitter, deliberately narrow:

* **Input**: a Python function written in a numpy-numeric subset
  (plain loops with array indexing, optional `math.*` and `np.*`
  numeric primitives that already appear in the Foundation corpus,
  no slicing, no broadcasting, no objects).
* **Output**: three sibling source files plus an auto-generated
  ctypes binding. The C version uses ``clock_gettime``; the C++
  version uses ``<chrono>``; the Pluto input wraps the loop nest in
  ``#pragma scop`` / ``#pragma endscop`` and survives ``polycc --pet
  --tile`` for the affine subset.

The tool is one of a planned family. Each target lives in its own
top-level directory so the dependency surface stays per-target:

| Tool          | Folder         | Status        |
|---------------|----------------|---------------|
| **NumpyToC**  | `NumpyToC/`    | this package  |
| NumpyToDaCe   | `NumpyToDaCe/` | future        |
| NumpyToCuPy   | `NumpyToCuPy/` | future        |
| NumpyToPythran| `NumpyToPythran/` | future     |
| NumpyToNumba  | `NumpyToNumba/`| future        |

When the second target lands we lift the shared front-end / IR into
a small ``numpyto_common`` library; for the single-target version
they live inline.

## Why a new tool

OptArena's current Foundation pipeline translates from the *C++*
reference (`scripts/emit_c_variants.py`). Two failure modes that
the new path fixes:

1. **1D pointer math defeats polycc.** 165 of 183 kernels fail
   ``polycc --pet`` because the C++ uses ``A[i*N+j]`` instead of
   declared multi-dim arrays. The numpy source has the multi-dim
   shape natively; emitting from Python keeps that information.
2. **No ``@`` / ``np.dot`` lowering.** The C++ corpus is hand-coded
   with naive loops; numpy-native idioms get rewritten by us into
   the same loop bodies (triple-loop GEMM for ``A @ B``, accumulator
   loop for ``np.sum``), so the C output matches even when the
   Python uses an idiom.

## Roadmap (future targets)

* **NumpyToFortran** (next obvious target). Fortran already has
  the rich numeric standard library numpy emits: ``MATMUL`` for
  ``A @ B``, ``DOT_PRODUCT`` for ``np.dot``, ``SUM`` /
  ``MINVAL`` / ``MAXVAL`` for axis reductions, ``MAXLOC`` /
  ``MINLOC`` for argmax / argmin, ``MERGE`` for masked select,
  ``EXP`` / ``SQRT`` / ``LOG`` as ELEMENTAL. So Fortran emission
  is the smallest delta from NumpyToC's IR -- the body walker
  swaps subscript syntax, calls map to ALL CAPS, and we pick up
  optimised vendor kernels for free.
* **NumpyToDaCe**: same IR + auto-emit; allow user override.
* **NumpyToCuPy** / **NumpyToNumba**: same shape, ``cupy.`` /
  numba decorator prelude.
* **NumpyToPythran**: similar, with Pythran's ``#pythran export``
  comments.
* Jax / Triton / TVM stay hand-written -- the gap from numpy to
  those is too large to automate.

## Scope (NumpyToC v0.1)

Supports only what the Foundation corpus actually exercises:

* Arithmetic: ``+ - * / // %``, unary ``-``, comparisons.
* Control flow: ``for/while``, ``range``, ``if/elif/else``,
  ``break``/``continue``.
* Array indexing: 1D and N-D with affine OR non-affine indices.
  Indirect access (``A[idx[i]]``) is supported but the Pluto output
  drops to "no-scop" mode for that kernel.
* ``math.*`` numeric primitives observed in the corpus:
  ``exp``, ``sqrt`` (extensible — see ``lowering.MATH_BUILTINS``).
* ``np.*`` numeric primitives observed:
  ``np.zeros`` (lowered to a stack-allocated VLA temporary).
* Future: ``np.dot``, ``np.sum``, ``A @ B``, ``np.maximum`` (rewrite
  rules already declared but unused on the current corpus).

Explicitly **not** supported: slicing, broadcasting, ``dtype=``
arguments, ``np.array`` constructors, advanced indexing, ``axis=``
keyword reductions, object arrays, strings.

## Output shape

For one Python kernel ``s111(a, b, ITERATIONS, LEN_1D)`` with
``a: f64[LEN_1D]`` / ``b: f64[LEN_1D]`` annotations:

```
out/<short>/cpp_backend/
  <short>_d.c                  # C99 + clock_gettime
  <short>_f.c                  # C99, single precision
  <short>_d.cpp                # C++ wrapping the same body in <chrono>
  <short>_f.cpp                # C++, single precision
  <short>_pluto_input.c        # C99 + #pragma scop markers
  CMakeLists.txt
  <short>_binding.json         # ctypes signature for the wrapper
```

The wrapper at ``<short>_cpp.py`` is unchanged — it consumes the
binding JSON to build the ctypes argtypes list.

## CLI

```bash
numpyto_c emit \
    --kernel optarena/benchmarks/foundation/s111/s111_numpy.py \
    --bench-info bench_info/s111.json \
    --out optarena/benchmarks/foundation/s111/cpp_backend
```

Single command; runs through every step (parse → IR → lower → emit
× 3 targets → bindings). Idempotent.
