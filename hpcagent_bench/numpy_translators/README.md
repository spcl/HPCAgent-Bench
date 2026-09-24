# NumpyToX

Emits a numpy kernel (`<kernel>_numpy.py`) as C99, C++, Pluto input, Fortran, JAX, Numba, CuPy,
Pythran and DaCe source, plus a ctypes binding JSON for the native targets. Emitted kernels carry no
timing; the harness times each call. Kernel-author rules: [`CONTRIBUTOR_GUIDE.md`](CONTRIBUTOR_GUIDE.md).

| Target | Code | CLI `--target` |
|---|---|---|
| shared frontend, IR, lowering | `src/numpyto_common/` | |
| C / C++ / Pluto / Polly | `src/numpyto_c/` | `c`, `pluto`, `polly`, `c_omp`, `cpp_omp`, `cpp_isopar` |
| DaCe | `src/numpyto_c/dace_emit.py` (`emit_dace`) | library only |
| Fortran | `src/numpyto_fortran/` | `fortran`, `fortran_omp` |
| JAX | `src/numpyto_jax/` (`emit_jax`) | library only |
| Numba | `src/numpyto_numba/` | `numba` |
| CuPy | `src/numpyto_cupy/` | `cupy` |
| Pythran | `src/numpyto_pythran/` | `pythran` |

## Emit

```bash
numpyto --target c \
    --kernel hpcagent_bench/benchmarks/loop_level_reasoning/argmax_value/argmax_value_numpy.py \
    --bench-info argmax_value.json --out out/
```

`numpyto` forwards every flag after `--target` to the backend's `emit` subcommand (`numpyto_c emit`,
`numpyto_fortran emit`, ...). The bench-info JSON is the manifest in emitter form:
`hpcagent_bench.emit_bridge.legacy_bench_info_dict(load_spec("<kernel>"))`, as written by
`bench_info_tempfile`. [`CONTRIBUTOR_GUIDE.md`](CONTRIBUTOR_GUIDE.md) section 6 has a runnable
script.

C-family output, with `<base>` = `<kernel>_<precision>` (e.g. `argmax_value_fp64`):

```
out/
  <base>.c                     # C99
  <base>.cpp                   # C++, same body
  <base>_pluto_input.c         # C99 + #pragma scop / endscop
  <base>_binding.json          # ctypes signature
  <base>_pluto_binding.json    # Pluto signature (symbols first)
```

`--parallel` (targets `c_omp`, `cpp_omp`) writes `<base>_omp.{c,cpp}` + `<base>_omp_binding.json`
and refuses a kernel with no sound parallel loop. `--isopar` (target `cpp_isopar`) writes
`<base>_isopar.cpp` + `<base>_isopar_binding.json`. `--precision float32` remaps float arrays;
`--config <key>` picks a sparse configuration.
