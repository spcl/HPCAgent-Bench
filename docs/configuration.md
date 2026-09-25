# Configuration: site and storage variables

Nothing in this repository names a cluster's filesystems, Slurm account, partition or nodes. Those
values come from environment variables, each with one default place:

| Where | What it resolves |
|---|---|
| `experiments/layers/site-<name>.env`, loaded by `scripts/site_env.sh` | this cluster's values: fast storage, partition, node exclusions, vendor paths |
| `scripts/cache_env.sh` | every cache and work directory, derived from `SCRATCH` and `FAST_SCRATCH` |
| `scripts/cscs/account_env.sh` | the Slurm account, read from your own Slurm associations |
| `experiments/env.sh` | the checkout and the venv; sources `scripts/repo_env.sh` and the two scripts above |
| `scripts/repo_env.sh` | the import path: the checkout (and `DACE_TREE` ahead of it), `PYTHONHASHSEED=0` |
| `pyproject.toml` (`[tool.hpcagent-bench] dace-pin`) | the dace commit a release installs and bakes into its images; jobs refresh to the latest extended ([below](#dace)) |
| `hpcagent_bench/paths.py` | the Python side of the same roots (`scratch_root`, `fast_scratch_root`) |

`cache_env.sh` and `account_env.sh` both load the site layer, so every submitter and every job sees
it. Campaign knobs (models, agents, judges, budgets) are not site values; they live in
`experiments/layers/common.env` and the model layers and are described in
[launch.md](launch.md) and [`experiments/LAUNCH.md`](../experiments/LAUNCH.md).

## Import path

An installed package (`pip install -e .`) needs nothing else. A checkout used without installing it
gets its import path from exactly one place:

| Who | How |
|---|---|
| a shell or job script | `. <checkout>/scripts/repo_env.sh`: the checkout on `PYTHONPATH`, `DACE_TREE` ahead of it when set, `PYTHONHASHSEED=0` |
| a command that starts inside a container | `<checkout>/scripts/repo_python script.py ...` (python3 after sourcing `repo_env.sh`; `REPO_PYTHON` picks another interpreter) |
| the test suite | `[tool.pytest.ini_options] pythonpath` in `pyproject.toml` |

Never set `PYTHONPATH` by hand and never edit `sys.path` in code. The one exception is a script that
runs inside the agent or judge image beside its sibling modules (the agent tools and harness
runners, `experiments/agent_driver.py` and its siblings, the `experiments/mpi` smokes): the images
set `PYTHONSAFEPATH=1`, which drops the script's own directory, so such a script puts that one
directory back. `tests/test_import_paths.py` fails on any other edit.

## Set up on your system

```bash
cd hpcagent-bench
cp experiments/layers/site-example.env experiments/layers/site.env   # gitignored
$EDITOR experiments/layers/site.env                                  # fill in your cluster's values
export SCRATCH=/path/to/your/scratch                                 # most HPC sites already set it
. experiments/env.sh                                                 # loads the layer, account, caches
echo "$FAST_SCRATCH $SBATCH_PARTITION $SBATCH_ACCOUNT $HF_HOME"
```

A minimal `site.env` for a cluster with a flash tier and a GPU partition named `gpu`:

```bash
FAST_SCRATCH="${FAST_SCRATCH:-/flash/${USER}}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu}"
SALLOC_PARTITION="${SALLOC_PARTITION:-${SBATCH_PARTITION}}"
```

Keep the `VAR="${VAR:-value}"` form: a value already exported in your shell then wins over the
layer, and sourcing the layer twice changes nothing. To keep the layer outside the checkout, point
`HPCAGENT_BENCH_SITE_ENV` at it instead of copying. With no layer at all, every variable below
takes its generic default, which runs everywhere `SCRATCH` is set.

`experiments/layers/site-cscs.env` is the layer for the CSCS Alps MI300A partition, the reference
setup the campaigns in this repository ran on. Use it there with
`export HPCAGENT_BENCH_SITE_ENV="$PWD/experiments/layers/site-cscs.env"`.

## Variables

### Site layer

| Variable | Default | Controls | CSCS value |
|---|---|---|---|
| `HPCAGENT_BENCH_SITE_ENV` | `experiments/layers/site.env` when it exists | which site layer `scripts/site_env.sh` loads; a named file that does not exist is an error | `experiments/layers/site-cscs.env` |
| `SBATCH_PARTITION` | unset (the cluster's default partition) | the partition of every `sbatch`; no script carries a `#SBATCH --partition` line | `mi300` |
| `SALLOC_PARTITION` | `SBATCH_PARTITION` | the same for `salloc` | `mi300` |
| `HPCAGENT_BENCH_EXCLUDE_NODES` | empty | a Slurm hostlist `experiments/regrade_rest.py` jobs avoid | a hostlist of five nodes |
| `HPCAGENT_BENCH_NETSTACK_BASE` | empty: the fabric check is skipped | host tree of the container network-stack artefacts `scripts/cscs/netstack_preflight.sh` checks before a campaign job | `/capstor/store/cscs/cscs/public/containers/netstack` |
| `HPCAGENT_BENCH_NETSTACK_VERSION`, `_NAME`, `_SOURCE` | the pinned bundle, `artifact` | which bundle under that base must exist | defaults |
| `HPCAGENT_BENCH_CI_PARTITION` | empty: `SBATCH_PARTITION` | partition of the CI replay, `scripts/run_tests.sh --container` (an MI250X node) | `mi200` |
| `HPCAGENT_BENCH_SITE_TESTS` | `0` | `1` runs the tests marked `site` (they need the cluster's Slurm and registered EDFs) | `1` |
| `HPCAGENT_BENCH_LOGIN_HOST`, `HPCAGENT_BENCH_SSH_JUMP` | empty: a placeholder | the login host and ssh jump chain in the laptop tunnel commands `containers/inference/serve-private.sbatch` prints | the Alps login and jump hosts |
| `HPCAGENT_BENCH_NICE` | `100` (`scripts/site_env.sh`) | the Slurm `--nice` every submitter passes when `NICE` is unset ([below](#submitting-nicely)) | default |

`SBATCH_PARTITION` is read by Slurm itself and overrides every `#SBATCH --partition` directive. A
job meant for other hardware passes `--partition=` on the command line, which wins over it;
`PARTITION=mi200` does that for campaign arms (`experiments/submit_common.sh`).

### Storage

| Variable | Default | Controls | CSCS value |
|---|---|---|---|
| `SCRATCH` | set by the site | bulk storage: runs, result DBs, logs, JIT caches, the venv | `/capstor/scratch/cscs/$USER` (set by the site) |
| `FAST_SCRATCH` | `SCRATCH`, else `<checkout>/.cache` | model weights and large read-mostly caches (`HF_HOME`, the judge's disk result store) | `/iopsstor/scratch/cscs/$USER` (flash tier) |
| `HPCAGENT_BENCH_CACHE` | `$FAST_SCRATCH/.hpcagentbench-cache` | root of the read-mostly caches | default |
| `HF_HOME` | `$HPCAGENT_BENCH_CACHE/hf` | Hugging Face hub (model weights) | default |
| `JIT_CACHE_ROOT` | `$SCRATCH/.hpcagentbench-cache`, else `<checkout>/.cache/jit` | compile/JIT caches (aiter, triton, inductor, vLLM), keyed by image below it | default |
| `HPCAGENT_BENCH_CPF_PRERENDER_DIR` | `$JIT_CACHE_ROOT/.cpf-prerender` | prerendered Canonical Parallel Form | default |
| `HPCAGENT_BENCH_TOOLS_DIR` | `$JIT_CACHE_ROOT/tools` | build tools not in the images (e.g. `ppcg`) | default |
| `HPCAGENT_BENCH_RUNS_ROOT` | `$JIT_CACHE_ROOT/runs` | per-job work directories of deterministic-framework jobs | default |
| `HPCAGENT_BENCH_RESULTS_DIR` | `$JIT_CACHE_ROOT/results` | persistent results those jobs merge into | default |
| `HPCAGENT_BENCH_DATA_ROOTS` | derived: top-level filesystems of `SCRATCH` and `FAST_SCRATCH` | what container EDFs bind-mount | derived |
| `HPCAGENT_BENCH_FROZEN_OBSERVATIONS` | a directory under `SCRATCH` | frozen observation CSVs the extractor merges | default |

### Checkout, Python and containers

| Variable | Default | Controls |
|---|---|---|
| `HPCAGENT_BENCH_REPO` | the checkout `experiments/env.sh` lives in | the tree scripts and jobs run from |
| `VENV` | `$SCRATCH/venv-hpcagent-bench-314` | the Python environment `env.sh` puts on `PATH` |
| `PY` | `$VENV/bin/python`, else the image's `python3` | the interpreter submitters call |
| `EDF_PATH` | `$HOME/.edf` | where registered container EDFs are looked up (Container Engine convention) |
| `CONTAINER_RUNTIME` | `enroot` | how `beverin.sbatch` starts containers: `enroot` or `ce` |
| `HPCAGENT_BENCH_HOST` | `SLURMD_NODENAME`, else the host name | the node name recorded with each result |

### dace

DaCe comes from the spcl/dace `extended` branch. It is not a PyPI dependency (PyPI rejects
direct-URL requirements) and `pyproject.toml` names no version of it.

| Variable | Default | Controls |
|---|---|---|
| `dace-pin` (`pyproject.toml`, `[tool.hpcagent-bench]`) | the one place it is written | the extended commit a release is tested with |
| `HPCAGENT_BENCH_DACE_REF` | installs and image builds: `pinned`; jobs: `extended` | which dace: `pinned` (the pin), a branch (its tip) or a full 40-character commit sha |
| `DACE_TREE` | unset | a dace checkout to run instead, exactly as it is (a fix branch under test); `scripts/repo_env.sh` puts it ahead of the installed dace and nothing refreshes it |
| `DACE_DIR` | `/opt/dace` | the image's editable dace checkout that `dace_refresh.sh` moves |

- **Install**: `scripts/install_dace.sh` installs the pin (`pip install "dace @
  git+https://github.com/spcl/dace.git@<pin>"`; `--editable DIR` for a checkout). README,
  CONTRIBUTING, CI, `scripts/rebuild_venv.sh` and the release smoke all use it, and the judge/agent
  image builds bake the pin, so a release install is reproducible.
- **Every job**: `containers/images/dace_refresh.sh` moves the image's `/opt/dace` to
  `HPCAGENT_BENCH_DACE_REF` before anything imports dace: by default the latest extended, with
  `HPCAGENT_BENCH_DACE_REF=pinned` to stay on the pin. A job that spans several containers
  resolves the ref to one sha on the batch host first (`dace_refresh.sh --resolve`), so every rank
  runs the same commit. A branch that cannot be fetched keeps the baked commit; a commit that
  cannot be reached fails the job. Bare metal (no `/opt/dace` checkout) runs the installed dace.
- **Provenance**: the refresh prints `dace-refresh: live commit <sha>` into the job log and writes
  `/opt/dace.commit`; canon columns stamp `dace <sha>` into `record.build` and `canon.db`'s `build`
  column; CPF prerender keys carry the dace commit.

To stay on the pin: `HPCAGENT_BENCH_DACE_REF=pinned sbatch ...`. To test a dace fix:
`DACE_TREE=<worktree> sbatch ...` (canon columns, CPF prerender).

### Submitting nicely

Every submitter passes `--nice`, so a batch of jobs yields to other users' work by default:
`NICE=<n>` for one submission, else `HPCAGENT_BENCH_NICE` (site layer; default `100`). The
campaign submitters also take `PRIORITY=<family>`, which picks the family's band
(`experiments/submit_common.sh`, `PRIORITY_NICE`) so waves start in the planned order. Slurm has no
environment variable for `--nice`, so a job script submitted by hand with a bare `sbatch` runs at
nice 0 unless the command line says `--nice="${HPCAGENT_BENCH_NICE}"`.

### Slurm account

| Variable | Default | Controls |
|---|---|---|
| `HPCAGENT_BENCH_ACCOUNT` | the single non-`root` association of `$USER` | the project account; required when you have several |
| `SBATCH_ACCOUNT`, `SLURM_ACCOUNT`, `SALLOC_ACCOUNT` | exported from `HPCAGENT_BENCH_ACCOUNT` | what `sbatch`, `srun` and `salloc` bill |

No script passes `-A`: a submitter naming its own account is how one campaign ends up billed to
two projects. With several associations and no `HPCAGENT_BENCH_ACCOUNT`, `account_env.sh` refuses
to pick one.

### Source repositories (`scripts/bootstrap_repos.sh`)

| Variable | Default |
|---|---|
| `BOOTSTRAP_ROOT` | `$SCRATCH` |
| `HPCAGENT_BENCH_GIT_URL` | `git@github.com:spcl/HPCAgent-Bench.git` |
| `DACE_GIT_URL`, `DACE_BRANCH` | `git@github.com:spcl/dace.git`, `extended` |
| `ARTIFACT_GIT_URL` | the paper artifact repository |

## Hardware profiles are not site values

`mi300` and `mi200` in image and EDF names (`hpcagent-bench-agent-mi300-latest`) and in
`PARTITION=mi200` / `experiments/layers/partition-mi200.env` name a GPU generation, not a Slurm
partition of one site: they select the image built for that architecture and its GPU count. The
Slurm partition that hardware sits in is the site layer's business.

## The guard

`tests/test_no_hardcoded_user_paths.py` scans every file `git ls-files` lists (tracked files only)
for storage mounts (`/capstor`, `/iopsstor`, ...), home directories, user names, site emails, Slurm
accounts, `#SBATCH` partition/account/node directives, node and login host names, literal
partitions (`--partition=`, `-p`, `*PARTITION=`), one campaign's run directories (dated or job-id
paths) and the site image registry, in live code (comments and docstrings may name a site to
explain it). Files, or single hits in a file, that legitimately carry such a value are allowlisted
in the test with one reason each: the CSCS site layer, the hardware-profile layer, this page, and
the MI300A/MI200 serving recipe's partition check.
