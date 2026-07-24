# Contributing to HPCAgent-Bench

Contributor guide: **[README](README.md)** (the single doc). Jump to:

- [**Add a benchmark**](docs/CONTRIBUTING.md#add-a-benchmark) -- the two files you
  write; the C/C++/Fortran/... baselines are generated for you.
- [**Add a container**](docs/CONTRIBUTING.md#add-a-container) -- one Dockerfile +
  Apptainer `.def` per hardware (cpu/nvidia/amd).
- [**Add a language**](docs/CONTRIBUTING.md#add-a-language) -- two edits (incl. a
  Rust example).
- [**The optimizer loop & scoring**](README.md#the-optimizer-loop--scoring) and
  [**how the prompt is generated**](docs/PROMPTS.md).

Normative reference specs:

- [`hpcagent_bench/docs/abi_contract.md`](hpcagent_bench/docs/abi_contract.md) -- the canonical
  C-ABI every native kernel exposes.
- [`hpcagent_bench/docs/sparse_abi.md`](hpcagent_bench/docs/sparse_abi.md) -- how a sparse matrix is
  declared and unpacked.

Conventions: prefer `pip`; no literal compiler flags outside `hpcagent_bench/flags.py`;
classes and files are public-by-default (no leading-underscore names); reuse existing
harness utilities over new abstractions; edit the `*_numpy.py` reference (the
framework siblings regenerate from it) -- never hand-edit a generated sibling. A
manifest argument may not be named `workspace`, `workspace_size`, or `time_ns` --
those are reserved by the C-ABI (abi_contract.md Sec. 11) and rejected at load.

YAML house style (all HPCAgent-Bench-owned YAML -- the per-kernel manifests, the
config/env files): a one-line `#` header saying what the file is,
two-space structural indent, no tabs, no trailing whitespace, one final newline.
`python tests/check_yaml_style.py` is the gate (`--fix` for the mechanical
parts); GitHub Actions / docker-compose YAML follow their own schemas and are
exempt.

Dev tasks run through the `Makefile` (`make help` lists them): `make format`
(yapf + clang-format + fprettify, in place), `make test` (fast suite; the
`integration`-marked build/run tests are excluded locally but run in CI), and
`make run BENCH=gemm FW=dace_cpu,pluto PRESET=S`. They are thin wrappers over
`scripts/` and the `hpcagent-bench` CLI -- no logic lives in the Makefile.
