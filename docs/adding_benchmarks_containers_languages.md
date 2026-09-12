# Contributing: add a benchmark, a container, or a language

Contributor guide for the three ways to extend HPCAgent-Bench: a new benchmark kernel, a new
container image variant, or a new compiled language. For contributor conventions (pip-first,
no literal compiler flags, YAML house style, no hand-editing generated siblings) see the
top-level [CONTRIBUTING.md](../CONTRIBUTING.md).

## Add a benchmark

Adding a benchmark kernel is covered in [docs/extending/benchmark.md](extending/benchmark.md):
the files you write (a NumPy reference and a manifest), the derived fields, validation, and
the checklist. Porting out of a real application rather than writing a kernel from scratch?
Profile it and choose the extraction boundary first: [`kernel_extraction.md`](kernel_extraction.md)
covers that end-to-end, including the manifest and the same validation steps.

## Add a container

Container images live in `containers/`. There is **one unified OCI recipe** --
`containers/hpcagent_bench.Dockerfile` -- selected per **hardware** by a build arg
`HW=cpu|nvidia|amd` (`cpu` is the default). Four backends run it, in preference order:
**Podman** (the default -- consumes the OCI tag directly, rootless and daemonless, so it is
the only one of the four that runs unprivileged on both a laptop and an HPC login node),
**Docker** (the same OCI tag under a daemon; needs dockerd and a root-equivalent group, so it
is the laptop / cloud-VM path, never the HPC one), **Apptainer** (builds a SIF FROM that same
OCI image -- a conversion, for sites that need one), and **`ce`**, CSCS Alps' Container Engine
(imports that same OCI image to SquashFS via `enroot`; selected by an `srun --environment=<edf>`
flag rather than run directly -- it has no wrapper command and no local launch form).

```
containers/hpcagent_bench.Dockerfile    the single OCI recipe          (build arg HW=cpu | nvidia | amd)
containers/cpu.def                Apptainer conversion recipe  (quickstart CPU .sif)
containers/judge.def              Apptainer conversion recipe  (the judge image)
```

The image is the full toolchain + HPC libraries + the Python deps in
`requirements/<hw>.txt`. Build the OCI image once (`podman build`, or `docker build` on a
machine that already runs a daemon) and run the tag directly; convert it to a SIF with
`apptainer build` (`docker-archive:...`) for a site that needs one -- the `cpu.def` quickstart
(`apptainer build hpcagent_bench-cpu.sif containers/cpu.def`) stays a valid Apptainer-native
shortcut. Compiler keys resolve from `hpcagent_bench/envs/compilers.yaml`. For the static
distributed (multi-endpoint) launch, see [docs/launch.md](launch.md).

## Add a language

Two edits, no NumpyToX change -- the binding/stub generator and the cffi loader
pick the language up automatically:

```
hpcagent_bench/envs/compilers.yaml   <- 1) a compiler block (install + compile/link templates)
hpcagent_bench/languages.py          <- 2) one LANG_EXT entry
```

Example -- adding **Rust** (`cdylib` -> a plain C-ABI `.so`):

```yaml
# hpcagent_bench/envs/compilers.yaml
rust:
  lang: rust                   # REQUIRED -- the per-language block lookup keys on it
  install: {apt: rustc}
  cc: rustc
  # baseline_ref names a constant in hpcagent_bench/flags.py -- never a literal -O3.
  compile: ["{cc}", "-O", "--crate-type=cdylib", "{baseline}", "{src}", "-o", "{lib}"]
  link: []                       # cdylib already links a C-ABI shared object
```
```python
# hpcagent_bench/languages.py
LANG_EXT = { ..., "rust": "rs" }     # no leading dot
```

The kernel then exports the canonical C symbol with `#[no_mangle] pub extern "C"`,
and the harness compiles + calls it like any other language.

## Add a framework backend

A **language** (above) is a compiled C-ABI target; a **framework** is a Python-side backend
(numba, dace, triton, tvm, cupy, pythran, jax, ...) whose implementation is auto-generated from
the NumPy reference. Most kernels never need this section -- it is for wiring up a *new*
framework, not a new kernel. Full mechanics, override points, and worked examples:
[frameworks.md](frameworks.md).
