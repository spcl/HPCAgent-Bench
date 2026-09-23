# Contributing to HPCAgent-Bench

Contributor guide: **[README](README.md)** (the single doc). Jump to:

- [**Add a benchmark**](docs/adding_benchmarks_containers_languages.md#add-a-benchmark) -- the two files you
  write; the C/C++/Fortran/... baselines are generated for you.
- [**Add a container**](docs/adding_benchmarks_containers_languages.md#add-a-container) -- one Dockerfile (built with
  podman by default, docker a drop-in) + Apptainer `.def` per hardware (cpu/nvidia/amd).
- [**Add a language**](docs/adding_benchmarks_containers_languages.md#add-a-language) -- two edits (incl. a
  Rust example).
- [**The optimizer loop & scoring**](README.md#how-it-works) and
  [**how the prompt is generated**](docs/prompts.md).

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

## Lint gate

Four gates, in this order, all clang-tidy-strict on the code they cover (warnings are failures):

1. **`ruff check`** (lint + bugbear + pyupgrade modernization) -- select/ignore is curated in
   `pyproject.toml`'s `[tool.ruff.lint]`, not ruff's defaults: a bare `ruff check` reports 21k+
   findings, most of it one house-specific pattern (numpy kernels named after paper notation)
   repeated thousands of times. `--fix` applies the safe subset.
2. **`ruff format`** -- the repo formatter (120 cols); NOT yapf here (yapf is dace's choice, not
   this repo's -- see `scripts/check_format.py`).
3. **`pyright`** -- the type gate, preferred over mypy. `[tool.pyright]` runs STANDARD mode
   repo-wide (excluding the kernel corpus, same scope as the annotation ratchet below);
   `pyrightconfig.strict.json` is a separate, growing allowlist of files held to STRICT mode.
4. **`pylint`** -- the deep-analysis gate ruff's `PL*` rules only partially cover. Configured in
   `[tool.pylint]` in `pyproject.toml` so it does not re-flag what ruff and the naming hooks
   already own (see that section's comments for exactly which checks are off and why).

```console
# from the repo root, with the project's deps on the path
$ export PYTHONPATH="$PWD:$PWD/hpcagent_bench/numpy_translators/src"

$ ruff check hpcagent_bench experiments tests scripts tools     # gate 1 (curated select from pyproject.toml)
$ ruff check --fix hpcagent_bench experiments tests scripts tools   # apply the safe fixes
$ ruff format hpcagent_bench experiments tests scripts tools    # gate 2

$ pyright --pythonpath "$(command -v python)"                   # gate 3, standard mode, repo-wide
$ pyright --project pyrightconfig.strict.json                   # gate 3, strict mode, the allowlist

# pylint is not on this venv's PATH by design (never `pip install` into the shared venv) --
# install it privately once: pipx install pylint, or a throwaway venv:
#   python3 -m venv /path/to/lint-venv && /path/to/lint-venv/bin/pip install pylint
$ PYTHONPATH="$PWD:$PWD/hpcagent_bench/numpy_translators/src:$(python -c 'import site;print(site.getsitepackages()[0])')" \
    /path/to/lint-venv/bin/pylint --rcfile=pyproject.toml hpcagent_bench/spec.py   # gate 4, one file
```

**The ratchet.** Fixing 13k+ pre-existing ruff findings (or 5.3k pyright diagnostics) before the
gate can be enforced is not realistic in one pass, so `tests/test_ruff_ratchet.py` and
`tests/test_pyright_ratchet.py` measure the DIRECTION instead, exactly like the older
`tests/test_annotation_ratchet.py`: a per-file baseline count, checked both ways -- a file may not
GAIN findings, and the baseline may not OVERSTATE what is actually there (a stale entry is slack a
regression can hide in). Regenerate after a real cleanup:

```console
$ python tests/test_ruff_ratchet.py --write
$ python tests/test_pyright_ratchet.py --write      # slower (~3 min, whole-tree pyright)
```

`pre-commit` runs `scripts/check_lint_gate.py` (ruff check, curated, + the ruff ratchet) on the
files being committed -- fast, because it only lints what changed. It does NOT run pyright or
pylint (both are minutes-scale over the whole repo); those run via `make lint-gate` / CI / on
demand.

**The hand-off report.** `python scripts/lint_area_report.py` groups every current ruff finding by
top-level area and rule code, and tags each rule `safe-autofix` (ruff's own `--fix`, no behavior
change -- modernization, cosmetic rewrites) or `manual-review` (the fix, or the finding itself, can
change behavior: `B905` zip `strict=` raises on unequal-length inputs, `PLW1510` subprocess
`check=` was silently absent, `F841`/`RUF059` may be a real bug rather than dead code, `B006`/`B008`
is the mutable-default-argument bug the rule exists to catch). Never committed as a snapshot -- it
goes stale the moment a file changes; regenerate it, do not read an old copy.

Dev tasks run through the `Makefile` (`make help` lists them): `make format`
(ruff + clang-format + fprettify, in place), `make test` (fast suite; the
`integration`-marked build/run tests are excluded locally but run in CI), and
`make run BENCH=gemm FW=dace_cpu,pluto PRESET=S`. They are thin wrappers over
`scripts/` and the `hpcagent-bench` CLI -- no logic lives in the Makefile.

**Running the suite on the cluster**: `scripts/run_tests.sh [pytest args...]` derives the
environment a full run needs (PATH, PYTHONPATH, OpenBLAS/CPATH, the MPI knobs) and belongs on a
compute node for anything beyond a quick targeted selection -- see `scripts/suite.sbatch`.
`scripts/run_tests.sh --container [pytest args...]` submits and waits on one mi300 node, running the
same command INSIDE the judge image instead of the login/compute-node toolchain, whose gcc has no
`-std=c23` and fails roughly 720 translator cases as a false regression; the container ships the
gcc 16 toolchain graded runs already use. Never run the full suite on the login node.

**Tests that need a user namespace (`-m sealed`)**: the judge grades agent code in a child that
unshares a user, mount and pid namespace first (`hpcagent_bench/seal.py`), and the tests that cover
that child carry the `sealed` marker. They are collected everywhere; on a host that cannot enter a
user namespace they SKIP with the kernel's own refusal (`skip:no-userns: ...`), decided by the real
probe in `tests/seal_capability.py` rather than by a guess about the host.

That skip is honest, but a surface that only ever skips is a surface nothing covers -- a refused
seal once reached `main` reported as a scoring assertion (`ts.scaling is None`). So CI runs the
marked selection for real in the **`mpi-sealed`** job (`.github/workflows/tests.yml`), and that job
fails its setup rather than skipping if the capability is missing.

What such a runner needs:

- **Unprivileged user namespaces**, i.e. `unshare(CLONE_NEWUSER)` followed by writes to
  `/proc/self/setgroups`, `/proc/self/uid_map` and `/proc/self/gid_map`. A stock GitHub-hosted
  ubuntu runner refuses the id-map write with `EPERM`, which is why the job runs in a container
  started with `--privileged` -- that implies `--security-opt apparmor=unconfined` and
  `seccomp=unconfined`, so the host's AppArmor restriction on unprivileged user namespaces does not
  apply to the process tree. It is still a standard, free `ubuntu-latest` runner; nothing here
  needs self-hosted hardware or a billed runner.
- **MPI that launches real ranks**: OpenMPI (`openmpi-bin`, `libopenmpi-dev`) with `mpi4py` built
  from source against it, plus `OMPI_MCA_btl=self,vader` and oversubscription, because a CI runner
  has no fabric and few cores. MPICH's Hydra bootstraps as singleton worlds there, so `-n 4` would
  yield four world-size-1 jobs and the multi-rank tests could only self-skip.
- Running as root in the container additionally needs `OMPI_ALLOW_RUN_AS_ROOT{,_CONFIRM}`.

To reproduce locally, ask the probe first -- it answers by doing the thing, not by guessing:

```console
$ python -c "import tempfile; from hpcagent_bench import seal; d=tempfile.mkdtemp(); \
    print(seal.probe(seal.SealPlan(hide=(), keep=(d,), readonly=(), workdir=d)) or 'can seal')"
```

An empty answer (`can seal`) means `python -m pytest -m sealed tests/` runs the selection for real.
Otherwise it prints what the kernel refused. On a Linux box where unprivileged user namespaces are
switched off, either turn them on
(`sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`, Ubuntu 24.04+) or run the suite
in a privileged container, which is what CI does:

```console
$ docker run --rm --privileged -v "$PWD:/repo" -w /repo ubuntu:24.04 bash -c '
    apt-get update -qq &&
    apt-get install -y -qq python3 python3-venv build-essential gfortran pkg-config \
      libopenblas-dev libfftw3-dev liblapacke-dev &&
    python3 -m venv /venv && /venv/bin/pip install -q --upgrade pip &&
    /venv/bin/pip install -q --group testing -e ".[cpu]" &&
    /venv/bin/python -m pytest -q -rfEs -m sealed tests/'
```

(the same install `.github/actions/setup` performs, minus the extras only other jobs need; `--privileged` is the part that matters here.)

On the cluster the judge image already has the capability, so `scripts/run_tests.sh --container
-m sealed tests/` runs them without any of this.
