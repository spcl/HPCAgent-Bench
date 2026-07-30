# Sample job submissions

**Two** modes, three samples. They are concrete examples, not templates: edit the node counts and
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
`hpc@npbench`: NPBench is not an HPC-only suite, and lenet, resnet, mlp, conv2d and softmax came
from it too and live under `ml/` here. Selecting by track would quietly make "the NPBench
corpus" mean 49 of 54.

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

## Submitting

    sbatch -A <account> samples/agentic_container.sbatch
    sbatch -A <account> samples/deterministic_kernels_to_ranks.sbatch
    DACE_MAIN=... DACE_EXTENDED=... sbatch -A <account> -N 8 samples/npbench_dace_flavors.sbatch

All three write under `results/`. The deterministic job's exit status is the merged failure count across
shards, so a shard whose kernels stopped compiling (or silently miscompiled) fails the job instead of
disappearing into one rank's log.
