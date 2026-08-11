# Sample job submissions

**Two** modes, six samples. They are concrete examples, not templates: edit the node counts and
kernel selection at the top and submit. Each one only sets env knobs and hands off to the real
script in `scripts/`, so a sample can never drift from the launcher it demonstrates.

There are two modes because there are two *deployments*, distinguished by what a rank number means.

| | `agentic_container.sbatch` | `deterministic_kernels_to_ranks.sbatch` |
|---|---|---|
| a rank is a | **role** (inference / judge / driver) | **kernel shard** |
| script | `scripts/submit_launch.sbatch` | `scripts/submit_deterministic.sbatch` |
| optimizer | an LLM agent | numpy / polly / dace_cpu / … |
| inference | vLLM endpoints on their own nodes | none |
| judge | dedicated judge node(s) | none |
| container | optional (`EDF=`) | no |
| result of a rerun | may differ (sampling) | identical artifact |

A **deterministic optimizer that still wants to be judged** is not a third mode — it is the
role-placed launcher with nothing to serve. `INFERENCE_ENDPOINTS=0 OPTIMIZER_NODES=O JUDGE_NODES=J
sbatch -N $((O+J)) scripts/submit_launch.sbatch` swaps the inference ranks for optimizer ranks and
keeps the judge, the settle protocol and the teardown identical. Use the sample here instead when you
only want timings and no scoring.

**Why the second mode shards by kernel and not by framework.** Kernel cost spans orders of magnitude
while the framework list is short and fixed, so a framework-per-rank split leaves most ranks idle
behind the slowest column. Each rank takes `kernels[rank::nranks]` and runs *every* framework over
its own kernels. Round-robin rather than contiguous blocks because neighbours in the sorted name list
tend to be similar sizes.

## The two mode exemplars

One sample per mode, kept minimal so the mode itself is what you read:

* **`agentic_container.sbatch`** — an LLM agent optimizing the corpus, scored by the judge. Set
  `EDF=` (optional) and the model / endpoint knobs `scripts/submit_launch.sbatch` takes. Produces
  judged scores plus timings.
* **`deterministic_kernels_to_ranks.sbatch`** — deterministic columns over a kernel selection. Set
  `FRAMEWORKS`, `BENCH`, `PRESET`. Produces the per-rank DB shards; the exit status is the merged
  failure count.

The four samples below are the second mode with a specific comparison already wired up.

## `npbench_dace_main_vs_pluto.sbatch` — one node, two optimizers, reports on

Mode 2 on a **single** node: `#SBATCH --nodes=1 --ntasks-per-node=4`, so four kernel shards each
measuring on a quarter of the node, over every kernel tagged `npbench` (`all@npbench` — a
`scientific_computing` subset plus a few ML kernels).

| column | what it is |
|---|---|
| `dace_cpu_autoopt` | upstream `auto_optimize`, on DaCe `main` |
| `pluto` | the polyhedral native column |
| `numpy` | the reference. **Not optional** — `plot` builds its speedup table against it and fails without it |

Both optimizer columns write **optimization reports**
(`$HPCAGENT_BENCH_PERF_REPORTS_OPT_REPORT=1`, `…_GENERATED_SOURCE=1`). DaCe replays the compile
command CMake recorded in its build folder's `compile_commands.json` with the repo's `-fopt-info`
flags; Pluto prepends `polycc`'s own transformation report to clang's remarks. Both come from a
separate compile-only run into a scratch directory, so the timed `.so` is untouched and a measured
number is identical with the reports on or off — but each costs one extra compile per kernel ×
column, which is why they are off by default.

    DACE_MAIN=~/src/dace-main sbatch -A <account> samples/npbench_dace_main_vs_pluto.sbatch

The branch check is the same `ensure_branch` the flavors job uses, now shared from
`scripts/dace_branch.sh` rather than copied — two copies is how the two would drift into disagreeing
about what "measured on `main`" means.

> **Read the Pluto column's report before comparing it.** The `pluto` column currently compiles the
> *untransformed* C++ — the same sources as `llvm`, with the same `clang++` — and never invokes
> `polycc`. The transformation report says so in its own header. Until that is fixed the column is a
> second `llvm`, not a polyhedral optimizer.

## `npbench_dace_flavors.sbatch` — one optimizer per column

Mode 2 again, with the columns split by SDFG pipeline instead of collapsed into one searching
`dace_cpu`. A search reports its winner, which answers *how fast is DaCe* and not *how fast is this
optimizer* — for the second you need every pipeline measured on every kernel, including the ones
where it loses. So each pipeline is its own flavor:

| flavor | pipeline | needs the fork? |
|---|---|---|
| `dace_cpu_parallel` | LoopToMap / MapCollapse / MapFusion | no — upstream transformations |
| `dace_cpu_autoopt` | upstream `auto_optimize` | no |
| `dace_cpu_canonicalize` | the fork's `canonicalize` + `finalize_for_target` | **yes** |
| `dace_gpu_parallel`, `dace_gpu_autoopt`, `dace_gpu_canonicalize` | same, offloaded | as above |

`parallel` and `autoopt` are upstream code, so they run on **both** branches — five stage-columns:

| | `main` | `extended` |
|---|---|---|
| `parallel` | ✓ | ✓ |
| `autoopt` | ✓ | ✓ |
| `canonicalize` | — | ✓ |

The four shared cells are the control that makes the fifth readable. Same pipeline, same kernel,
same preset, different tree: whatever differs there is the DaCe underneath rather than the
optimizer, and without it the canonicalize column cannot be read as a claim about the optimizer at
all.

A stage **verifies** its tree is on the branch it claims, and refuses to run otherwise — measuring
the wrong DaCe produces numbers that look entirely normal, which is the one failure this job design
exists to rule out. `DACE_CHECKOUT=1` lets the script `git checkout` instead (refused on a dirty
tree), which also means `DACE_MAIN` and `DACE_EXTENDED` can be the same clone.

### How a row records all this

One flat name on the command line (`--framework dace_cpu_parallel`), three columns in the DB:

| column | example | how it is set |
|---|---|---|
| `framework` | `dace_cpu` | the backend, flavor suffix stripped |
| `flavor` | `parallel` | which optimizer inside it — NULL for a plain column |
| `build` | `extended` | which DaCe tree ran — NULL for a single-build run |

The split is by whether you can **ask** for it. The pipeline is selectable, so it is part of the
framework name you type and lands in `flavor`. The tree is not — it is whatever `PYTHONPATH`
resolved to, a property of the deployment like `execution` — so it is stamped
(`HPCAGENT_BENCH_RECORD_BUILD`) rather than requested. Storing them apart keeps `GROUP BY
framework` gathering every DaCe row; `hpcagent-bench plot` folds them back into one series name
(`dace_cpu/parallel/extended`) exactly as it folds the sparse `variant` into the benchmark name.

    DACE_MAIN=~/src/dace-main DACE_EXTENDED=~/src/dace-extended \
        sbatch -A <account> -N 8 samples/npbench_dace_flavors.sbatch

`-N 8` with `--ntasks-per-node=4` is 32 kernel shards over 8 nodes, each rank measuring on a
quarter of a node. The rank count is read back from the allocation (`SLURM_NTASKS`), never from a
second knob, so the script cannot ask `srun` for a distribution the allocation does not have; the
thread split comes from `preflight --ranks-per-node`, because four ranks each claiming every core
measure contention while every rank's own log still looks correct. `GPU=1` swaps in the GPU
flavors.

`BENCH=all@npbench` is **every** kernel tagged `npbench`, across tracks — 54 of them. Not
`scientific_computing@npbench`: NPBench is not an HPC-only suite, and lenet, resnet, mlp, conv2d and softmax came
from it too and live under `machine_learning/` here. Selecting by track would quietly make "the NPBench
corpus" mean 49 of 54.

## Native on CSCS Daint/Alps — no container

Two samples run the same mode-2 sweep on Alps with **no container**: no EDF, no `--environment`, no
`ce.srun_flag`. They keep the site knowledge from
[`scripts/cscs/submit_loop_level_reasoning_alps.sbatch`](../scripts/cscs/submit_loop_level_reasoning_alps.sbatch) — `-A
<account>` mandatory, aarch64 GH200 nodes, the DaCe build folder off `/tmp` (tmpfs on these nodes),
results in the repo rather than node-local — and drop its container plumbing entirely.

* **`cscs_alps_native_three_way.sbatch`** — `pluto`, `dace_cpu_autoopt` on `main`, and
  `dace_cpu_autoopt` on `extended`. Three columns plus numpy, one speedup PDF.
* **`cscs_alps_native_pipelines.sbatch`** — `parallel` + `autoopt` on `main`, and `parallel` +
  `autoopt` + `canonicalize` on `extended`. Five stage-columns plus numpy, one speedup PDF.

The first isolates the **tree**: one optimizer, two DaCes, so whatever differs between the two
`autoopt` cells is the DaCe underneath. The second is the full pipeline grid, with the four shared
cells as the control for the fifth — same reasoning as `npbench_dace_flavors.sbatch`, over the `scientific_computing`
track instead of NPBench.

Native means **nothing comes from an image**, so all three must exist on the compute node, and each
is refused by name at submission rather than discovered on rank 3 of an allocation already charged:

* `HPCAGENT_BENCH_ENV` — a script the job `source`s: the site `module load`s plus the venv activate
  for the python that has `hpcagent-bench` installed.
* `DACE_MAIN` — a DaCe checkout on `main`. It goes on `PYTHONPATH`, so it is the **repo root**, not
  its `dace/`.
* `DACE_EXTENDED` — a DaCe checkout on `extended`, likewise.

    HPCAGENT_BENCH_ENV=$SCRATCH/hpcagent-env.sh DACE_MAIN=$SCRATCH/dace-main \
        DACE_EXTENDED=$SCRATCH/dace-extended \
        sbatch -A <account> samples/cscs_alps_native_three_way.sbatch

`require_native_env` / `require_dace_tree` / `evict_base_sdfg_cache` live in
[`scripts/cscs/native_env.sh`](../scripts/cscs/native_env.sh), shared by both, for the same reason
`ensure_branch` lives in `scripts/dace_branch.sh`: a second copy is how the two would drift.

`BENCH=scientific_computing` is **132** kernels. That is the number to size `--time` against — the defaults here are
12 h over 4 nodes (three-way) and 8 nodes (pipelines), at `PRESET=L`, and are a starting point, not a
measurement.

### Two traps these two avoid

**`numpy` must not carry a `build` stamp.** `HPCAGENT_BENCH_RECORD_BUILD` is read for *every*
framework, not just the DaCe ones, and `plot` folds a non-null `build` into the series name. A numpy
row stamped `main` therefore plots as `numpy/main`, and `heatmap_figure`'s `assert ('numpy' in
frmwrks)` fails — no speedup table, after the whole sweep. Both native samples run the
tree-independent columns (`numpy`, and `pluto` in the three-way job) in their own **unstamped**
stage, so the baseline stays `numpy` and is still measured exactly once.

> The two older DaCe samples above do **not** do this: `npbench_dace_main_vs_pluto.sbatch` exports
> `HPCAGENT_BENCH_RECORD_BUILD=main` for its whole run, and `npbench_dace_flavors.sbatch` puts
> `numpy` in its `main` stage. Both stamp the baseline and their final `plot` step trips that
> assert. Not fixed here — it is a change to files this section does not own.

**A base SDFG parsed by one tree must not be reused by the other.** `DACE_BUILD_ROOT` per stage
separates the compiled `.so`, but the *parsed* base SDFG is cached a level above it, in
`hpcagent_bench/benchmarks/<kernel>/.cache/<module>_cpu.sdfgz`, fingerprinted on the kernel sources
and the precision **only** — not on which DaCe parsed them (`DaceFramework._sdfg_fingerprint`). Two
trees in one job collide there: whichever stage runs first seeds the cache, and the second measures
its own pipelines over the *first* tree's parse. Ordering-dependent, and it produces numbers that
look entirely normal. Both samples call `evict_base_sdfg_cache` before each DaCe stage so every tree
parses with its own frontend; a miss is just a rebuild.

### Unverified

Carried over from `submit_loop_level_reasoning_alps.sbatch`, which says the same of itself: **none of this has
been checked against the site's own submission scripts.** The partition name, the account and the
scratch layout are the three things most likely to need a local edit — no `--partition` line is set
for that reason. `$SCRATCH` on Alps is still the parallel FS; `/iopsstor` (flash) suits thousands of
small compiler writes better than `/capstor`, so point `DACE_BUILD_ROOT` there if both are mounted,
but only the mount *names* are known here (from `scripts/cscs/env.toml.example`). There is no Slurm
on the development box, so both scripts are verified only by `bash -n`, by the column names and CLI
flags being checked against the code, and by their helpers being unit-exercised on a fake tree.

## Results and the DB

Every rank writes its **own** `hpcagent_bench<rank>.db` in the repo directory. That is not a
workaround for SQLite's locking: WAL needs a `-shm` mapping that Lustre/NFS/GPFS do not provide, so
one shared file across ranks is not an option. The shards are persistent artifacts, never scratch, and
never on memory-backed storage (`recording.base_db_path` refuses a tmpfs path outright).

Merging is automatic — no step to forget:

* a reader (`plot`, `plot-dist`) calls `recording.ensure_aggregated`, which builds the aggregate if it
  is missing *or older than a shard*;
* `run-framework --summarize` merges as part of closing the run;
* `hpcagent-bench aggregate-db` forces it now, for archiving or copying one file off the cluster.

The aggregate is always rebuilt from scratch, so merging twice cannot double the rows.

Both DaCe samples end by forcing the merge (`hpcagent-bench aggregate-db`, so the one file to copy
off the cluster exists whether or not anything reads it) and then rendering the **speed-up chart**
with `scripts/plot_speedup.py` — signed relative change, banded by order of magnitude. The old
NPBench-style **table** is opt-in (`hpcagent-bench plot`) and no job runs it for you: on its ratio
axis a 0.5x regression reads as a smaller event than a 1.5x win. Both go through the one loader, so
both fold `flavor` and `build` back into one series name (`dace_cpu/autoopt/main`) exactly as
`variant` folds into the benchmark name, and both re-run the merge if a shard moved, so the two
steps cannot disagree.

## Submitting

    sbatch -A <account> samples/agentic_container.sbatch
    sbatch -A <account> samples/deterministic_kernels_to_ranks.sbatch
    DACE_MAIN=... sbatch -A <account> samples/npbench_dace_main_vs_pluto.sbatch
    DACE_MAIN=... DACE_EXTENDED=... sbatch -A <account> -N 8 samples/npbench_dace_flavors.sbatch
    HPCAGENT_BENCH_ENV=... DACE_MAIN=... DACE_EXTENDED=... \
        sbatch -A <account> samples/cscs_alps_native_three_way.sbatch
    HPCAGENT_BENCH_ENV=... DACE_MAIN=... DACE_EXTENDED=... \
        sbatch -A <account> samples/cscs_alps_native_pipelines.sbatch

All of them write under `results/`. The deterministic job's exit status is the merged failure count across
shards, so a shard whose kernels stopped compiling (or silently miscompiled) fails the job instead of
disappearing into one rank's log. The multi-stage jobs run every stage even when an earlier one fails
and report the failed stage names at the end — losing half a comparison silently is the worst outcome.
