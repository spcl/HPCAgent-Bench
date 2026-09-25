# Sample job submissions

Concrete examples, not templates: edit the knobs at the top and submit. Each sample only sets
environment knobs and hands off to the real launcher, so a sample cannot drift from it. `-A
<account>` is required on Alps (or export `SBATCH_ACCOUNT`).

Three modes:

| mode | sample | a rank is a | launcher |
|---|---|---|---|
| agentic, judged | `samples/agentic_container.sbatch` | role (inference / judge / driver) | `scripts/submit_launch.sbatch` |
| deterministic, unjudged | `samples/deterministic_kernels_to_ranks.sbatch` | kernel shard | `scripts/submit_deterministic.sbatch` |
| campaign (login node) | `experiments/samples/llr40_*.sh` | one job per arm | `experiments/submit-*.sh` |

```bash
sbatch -A <account> samples/agentic_container.sbatch
sbatch -A <account> samples/deterministic_kernels_to_ranks.sbatch
DACE_MAIN=$SCRATCH/dace-main DACE_EXTENDED=$SCRATCH/dace-extended \
    sbatch -A <account> -N 8 samples/npbench_dace_flavors.sbatch
HPCAGENT_BENCH_ENV=$SCRATCH/hpcagent-env.sh DACE_MAIN=$SCRATCH/dace-main DACE_EXTENDED=$SCRATCH/dace-extended \
    PLAN=three-way sbatch -A <account> samples/cscs_alps_native.sbatch
```

## Agentic vs deterministic

| | `agentic_container.sbatch` | `deterministic_kernels_to_ranks.sbatch` |
|---|---|---|
| optimizer | LLM agent | numpy / polly / dace_cpu / ... |
| inference | vLLM endpoints on their own nodes | none |
| judge | dedicated judge node(s) | none |
| container | optional (`EDF=`) | optional (`EDF=`) |
| rerun | may differ (sampling) | identical artifact |

A deterministic optimizer that still wants judging uses the role launcher with no inference:

```bash
INFERENCE_ENDPOINTS=0 OPTIMIZER_NODES=2 JUDGE_NODES=1 sbatch -A <account> -N 3 scripts/submit_launch.sbatch
```

The deterministic launcher shards by kernel, not framework: kernel cost spans orders of magnitude
while the framework list is short. Rank `r` of `R` takes `kernels[r::R]` and runs every framework
over them; round-robin because neighbors in the sorted name list tend to have similar sizes.

## DaCe pipelines: `npbench_dace_flavors.sbatch`

Each DaCe pipeline is its own column, so every pipeline is measured on every kernel, including the
ones where it loses:

| flavor | pipeline | branch |
|---|---|---|
| `dace_cpu_parallel` | LoopToMap / MapCollapse / MapFusion | `main` and `extended` |
| `dace_cpu_autoopt` | upstream `auto_optimize` | `main` and `extended` |
| `dace_cpu_canonicalize` | `canonicalize` + `finalize_for_target` | `extended` only |

`GPU=1` swaps in `dace_gpu_*`. The four shared cells (parallel and autoopt on both trees) are the
control: same pipeline, kernel and preset, different tree, so any difference is the DaCe underneath.
That makes the canonicalize column readable as a claim about the optimizer.

Each stage checks its tree is on the claimed branch and refuses otherwise. `DACE_CHECKOUT=1` lets the
script `git checkout` instead (refused on a dirty tree), so `DACE_MAIN` and `DACE_EXTENDED` may be
one clone. Each stage gets its own `DACE_BUILD_ROOT` and its own `PYTHONPATH`.

`BENCH=all@npbench` (the default) is every kernel tagged `npbench` across tracks: 56, of which 51
are in `scientific_computing` and 5 (lenet, resnet, mlp, conv2d, softmax) in `machine_learning`.
`-N 8` with `--ntasks-per-node=4` gives 32 shards, each measuring on a quarter node; the thread
split comes from `hpcagent-bench preflight --ranks-per-node`.

Smoke-test one column locally first:

```bash
hpcagent-bench run-framework -b all@npbench -f dace_cpu_parallel -p S -r 3 --validate
```

For a single-tree comparison (e.g. against Pluto), edit the `STAGES` table in the sample to one
stage such as `"${DACE_MAIN} main numpy,dace_cpu_autoopt,pluto"`; the script sets `STAGES` itself,
so it cannot come from the environment.

### How a row records the flavor and tree

| column | example | set by |
|---|---|---|
| `framework` | `dace_cpu` | backend, flavor suffix stripped |
| `flavor` | `parallel` | the `--framework` name you ask for; NULL for a plain column |
| `build` | `extended` | `HPCAGENT_BENCH_RECORD_BUILD`; NULL for a single-tree run |

Readers fold both into one series name (`dace_cpu/parallel/extended`), as `variant` folds into the
benchmark name. The baseline column never folds, so a stamped `numpy` row stays `numpy`.

## Native on CSCS Alps: `cscs_alps_native.sbatch`

The same sweep over `scientific_computing` (171 kernels, `PRESET=L`) with no container, EDF or
`--environment`. `PLAN` picks the comparison:

- `PLAN=pipelines` (default): parallel + autoopt on `main`; parallel + autoopt + canonicalize on
  `extended`; numpy in an unstamped stage.
- `PLAN=three-way`: `dace_cpu_autoopt` on each tree, plus `numpy` and `pluto` in an unstamped stage.
  One optimizer on two trees isolates the tree.

Required, each checked at submission:

- `HPCAGENT_BENCH_ENV`: a script the job sources (site `module load`s plus the venv that has
  `hpcagent-bench`).
- `DACE_MAIN`, `DACE_EXTENDED`: DaCe checkouts on `main` / `extended`; the repo root, not `dace/`.

The helpers (`require_native_env`, `require_dace_tree`, `evict_base_sdfg_cache`) live in
`scripts/cscs/native_env.sh`; `ensure_branch` lives in `scripts/dace_branch.sh`. The build root
defaults to `$SCRATCH`, off node-local tmpfs; point `DACE_BUILD_ROOT` at a flash tier if one is
mounted (`scripts/cache_env.sh`, `scripts/cscs/env.toml.example`). No `--partition` is set: the
partition, account and scratch layout are the site-specific lines to check before a first run.

## Campaigns

Campaign samples run on the login node and submit one job per arm through `experiments/submit-*.sh`,
which holds the arm matrix, node budget and settle protocol.

| sample | measures |
|---|---|
| `experiments/samples/llr40_canon_baselines.sh` | seven compiler baselines (numba, cc, cc_autopar, dace_cpu(_canonicalize), dace_gpu(_canonicalize)) |
| `experiments/samples/llr40_cpf_ablation.sh` | no skill packet vs the canonical-parallel-form page alone, per model |
| `experiments/samples/llr40_gpu_models.sh` | skills on/off per model, one GPU language at a time |

```bash
experiments/samples/llr40_canon_baselines.sh
SUBMIT=0 experiments/samples/llr40_cpf_ablation.sh    # print the arms, submit nothing
MODELS=qwen38 LANGUAGES=hip experiments/samples/llr40_gpu_models.sh
```

The CPF ablation needs pre-rendered forms: the judge serves them from a directory and never renders
on demand. Render with `hpcagent-bench cpf` (or `experiments/prerender_cpf.sbatch`), then run
`experiments/preflight_gpu.sh`, which refuses when the form directory is short.

## Results

Every rank writes its own `hpcagent_bench<rank>.db` under `results/` in the repo: SQLite WAL needs a
`-shm` mapping that Lustre, NFS and GPFS do not provide. Shards are durable artifacts;
`record.db_path` refuses memory-backed storage. Merging is automatic and idempotent:

- DB readers (`hpcagent-bench plot`, `plot-dist`, `statistics/plot_speedup.py`) rebuild the
  aggregate when a shard is newer;
- `run-framework --summarize` merges when closing a run;
- `hpcagent-bench aggregate-db` forces it (add `--source <shard>` for shards in per-rank
  directories).

The deterministic job's exit status is the merged failure count across shards. Multi-stage samples
run every stage even after a failure and list the failed stages at the end.

## Plotting

| question | command |
|---|---|
| per-kernel speed-up, signed and banded | `python statistics/plot_speedup.py -b all@npbench -p XL --output results/plots/speedup.pdf` |
| speed-up vs token cost | `python statistics/plot_score_change.py data/observations.csv --experiment cpf-llr-focus40` |
| NPBench-style ratio table | `hpcagent-bench plot` |

`plot_speedup.py` plots signed relative change banded by order of magnitude, so a 0.5x regression
reads as large as a 2x win. It divides by `numba`; a sweep without a `numba` column writes no
figure. `--compact --bare` sizes it for a paper column, `--boxplot` shows run-to-run spread, `--demo`
renders synthetic data. Extract `data/observations.csv` with `python -m hpcagent_bench.experiments`
([`docs/plotting.md`](../docs/plotting.md)).
