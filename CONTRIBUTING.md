# Contributing to HPCAgent-Bench

Set up, lint, test, and find the page for the thing you are adding. Normative specs:
[`abi_contract.md`](hpcagent_bench/docs/abi_contract.md) and
[`sparse_abi.md`](hpcagent_bench/docs/sparse_abi.md). How the loop and the score work:
[README](README.md#how-it-works); how a prompt is rendered: [prompts.md](docs/prompts.md).

## Development setup

Python 3.12 or newer. Every hardware extra includes the dev tools (the `dev` extra):

```sh
python -m venv .venv && . .venv/bin/activate
pip install --upgrade "pip>=25.1"
pip install -e ".[cpu]"                    # .[nvidia] / .[amd] on a GPU host
scripts/install_dace.sh                    # dace: the pinned spcl/dace@extended commit
pre-commit install
```

On a cluster, source `experiments/env.sh`: it loads the site layer, names the host interpreter
(`HPCAGENT_BENCH_HOST_PYTHON`) and sets `PYTHONHASHSEED=0` ([docs/configuration.md](docs/configuration.md)).

## Lint and format

| Gate | Command |
|---|---|
| format (ruff format, clang-format, fprettify; 120 columns) | `python scripts/checks/check_format.py --fix <files>` |
| lint | `ruff check <files>` |
| types | `pyright <files>`; files in `pyrightconfig.strict.json` also pass `pyright --project pyrightconfig.strict.json` |
| every hook (format, headers, naming, YAML style, manifest structure, ...) | `pre-commit run --files <files>` |

Conventions the hooks do not catch:

- No literal compiler flags outside `hpcagent_bench/flags.py`; compiler blocks live in
  `hpcagent_bench/envs/compilers.yaml`.
- Public names only (no leading underscore); ASCII source; comments state the present design.
- Edit the `<kernel>_numpy.py` reference, never a generated sibling (`*_dace.py`, `*_numba_np.py`,
  `cpp_backend/`, ...).
- A manifest argument may not be named `workspace`, `workspace_size` or `time_ns` (abi_contract.md
  Sec. 11).
- YAML the project owns: a one-line `#` header, two-space indent, no tabs or trailing whitespace
  (`python tests/check_yaml_style.py [--fix]`).

## Tests

```sh
python -m pytest -q -n 4 tests/test_framework_flavors.py          # a targeted selection, login node
scripts/run_tests.sh -q -n 4 tests/test_metrics_autovec.py        # same, with the derived env (BLAS, MPI, PATH)
```

The login node runs targeted selections only (at most `-n 4`). Anything that compiles many kernels,
and the full suite, runs on a compute node inside the judge image (its gcc 16 accepts `-std=c23`;
about 720 translator cases fail outside it for that reason alone):

```sh
. experiments/env.sh
P=--partition="$HPCAGENT_BENCH_CI_PARTITION"
sbatch $P scripts/ci_mi200.sbatch                                  # every CI job, about 3 hours
sbatch $P scripts/ci_mi200.sbatch --ci --jobs unit,mpi             # chosen CI jobs
sbatch $P scripts/ci_mi200.sbatch -q -n 16 tests/translators       # a pytest selection
```

With no arguments it replays `.github/workflows/tests.yml` through `scripts/ci_replay.py` and writes
per-step logs and `summary.txt` to `$SCRATCH/ci-replay/<jobid>`; `scripts/run_tests.sh --ci --list`
prints what it would run, and `scripts/run_tests.sh --container <args>` submits it and waits.

**`-m sealed`.** The judge grades agent code in a child that unshares a user, mount and pid
namespace (`hpcagent_bench/seal.py`). Tests of that child carry the `sealed` marker and skip on a
host that cannot enter a user namespace; CI runs them in the `mpi` job's sealed phase. Ask the
probe:

```sh
python -c "import tempfile; from hpcagent_bench import seal; d=tempfile.mkdtemp(); \
print(seal.probe(seal.SealPlan(hide=(), keep=(d,), readonly=(), workdir=d)) or 'can seal')"
```

`can seal` means `python -m pytest -m sealed tests/` runs them. Otherwise enable unprivileged user
namespaces (`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` on Ubuntu 24.04+) or
run them as CI does:

```sh
docker run --rm --privileged -v "$PWD:/repo" -w /repo ubuntu:24.04 bash -c '
  apt-get update -qq && apt-get install -y -qq python3 python3-venv build-essential gfortran \
    pkg-config libopenblas-dev libfftw3-dev liblapacke-dev &&
  python3 -m venv /venv && /venv/bin/pip install -q --upgrade pip &&
  /venv/bin/pip install -q -e ".[dev]" &&
  /venv/bin/python -m pytest -q -rfEs -m sealed tests/'
```

## Adding things

| Addition | Guide |
|---|---|
| a benchmark kernel | [docs/extending/benchmark.md](docs/extending/benchmark.md); from an application: [kernel_extraction.md](docs/kernel_extraction.md) |
| a framework column or a non-LLM optimizer | [docs/extending/optimizer.md](docs/extending/optimizer.md) |
| an agent harness | [docs/extending/agent-harness.md](docs/extending/agent-harness.md) |
| a skill page or an agent tool | [docs/extending/skills-and-tools.md](docs/extending/skills-and-tools.md) |
| a packet | [docs/extending/packets.md](docs/extending/packets.md) |
| a model or an inference engine | [docs/extending/inference.md](docs/extending/inference.md) |
| a prompt variant or hint | [docs/prompts.md](docs/prompts.md#variants) |
| a container image | [containers/README.md](containers/README.md) |
| an experiment arm | [experiments/README.md](experiments/README.md#configuration) |
| an agent | [docs/writing_an_agent.md](docs/writing_an_agent.md) |

Grading changes: every reported number is graded under one rule, `mw4x5`
([measurement_statistics.md](docs/measurement_statistics.md#the-final-grade-mw4x5)); changing it
changes it for every arm.

### Kernel provenance

Line 2 of a manifest states where the kernel comes from, as a YAML flow mapping in a comment:
`# provenance: {kind: derived, upstream: rodinia, detail: hotspot}` or `# provenance: {kind: original}`.
The vocabulary and every upstream are in `third_party/upstreams.yaml`. A new upstream gets an entry
there (and its license text in `third_party/licenses/` if missing); then
`python scripts/render_attribution.py --write` refreshes CONTRIBUTORS.md and NOTICE
(`tests/test_attribution.py` fails while they are stale).

### A sweep metric

A per-kernel quantity recorded beside the timings as long-format `kernel_metrics` rows
(`metric = "<name>.<count>"`). Add `hpcagent_bench/metrics/<name>.py` defining `enabled()`,
`measure_sweep(frmwrk, impl, bench, reports, datatype)` and `rows(measured, **stamp)`, and a
`metrics.<name>` switch (default `false`) in `hpcagent_bench/config.yaml`. `metrics.sweep_metrics()`
finds it; `autovec.py` and `parallelism.py` are the examples. Check:
`python -m pytest -q tests/test_metrics_registry.py tests/test_metrics_<name>.py`.

### A language

A compiled C-ABI target an agent can submit in:

| File | Change |
|---|---|
| `hpcagent_bench/envs/compilers.yaml` | a compiler block with `lang: <language>` and its compile/link templates |
| `hpcagent_bench/languages.py` | one `LANG_EXT` entry; a GPU language also a `GPU_HOST_LANG` entry |
| `hpcagent_bench/support/bindings/stubs.py` | a branch in `gen_call_stub` rendering the empty entry point |

`LANG_EXT` is the one list the `Language` enum, stub and binding languages and the delivery check
read. Optional: `languages.LANG_TARGET`, `contract.INDEX_BASE` for a 1-based language, a prompt
fragment `hpcagent_bench/harness/prompts/sections/lang/<language>.j2`, and a skill page
`hpcagent_bench/skills/lang-<language>/`. Check:
`python -m pytest -q tests/test_language_registry.py tests/test_bindings.py tests/test_compiler_family.py`.
