# Tutorial: Triton kernels for the scientific-computing track, against a numba baseline

This tutorial runs every kernel of the scientific-computing (SC) track in two places. A GPU node of the student
cluster runs your Triton kernels. A CPU node of Euler (AMD EPYC 7742) runs the numba baseline once, for the whole
class. Both runs record into one results file (`canon.db`), and one command plots each kernel's speedup of Triton
over numba. Neither run needs a container, a judge or a Slurm reservation of ours.

## What runs where, and why

| Run | Where | Column | Done by |
|---|---|---|---|
| numba baseline, timed once | Euler, one EPYC 7742 socket (64 cores) per rank | `numba` | the course staff, once |
| your kernels | the student cluster, one GPU | `triton` | each student, as often as needed |
| speedup plot | anywhere | none | each student |

The student cluster gives a job at most 3 CPU cores, too few to time a CPU baseline that means anything. So no
baseline is timed there: your Triton times are divided by the numba times measured on Euler. The speedup therefore
compares two machines, one GPU against one 64-core EPYC socket. Read it as "faster than a well-parallelized CPU
socket", not as a same-machine comparison. A sequential C baseline on 3 cores would take hours at the size used
here and would say little; this tutorial does not time one.

Both runs use the same size preset (`L`), the same per-kernel datatype and the same repetitions, so each kernel's
two rows describe the same computation.

- **Datatype, `-d fp32-if-allowed`.** Each kernel runs in fp32 when its manifest allows fp32, and in fp64
  otherwise. The SC track grades in fp64 by default; a kernel whose manifest lists `allowed_precisions: [fp64]`
  (a solver whose conditioning needs it, for example) stays in fp64. Both runs resolve the datatype per kernel the
  same way, so the rows pair up.
- **Repetitions, `-r 20`.** One untimed warm-up call (`measurement.warmup`), then 20 timed calls; the recorded time
  is their median. Triton autotunes on its first call, so the autotune search falls in the warm-up and never in a
  timed sample. This is the shape of the `mw1x20` grade protocol (one input, 20 timed runs), with the median in
  place of the paired Mann-Whitney test that a judge-side grade uses.
- **Validation.** Every call's output is checked against the compiled C reference before its time counts. A kernel
  that fails validation records `validated=0` and is left out of the plot.
- **Data on the GPU.** The Triton column receives every dense array as a `torch` tensor already on the GPU. The
  copy happens outside the timed bracket, so a Triton time is the kernel alone.

## 1. Install (both machines, no container)

```sh
git clone https://github.com/spcl/HPCAgent-Bench.git && cd HPCAgent-Bench
uv sync --extra nvgpu      # student cluster with NVIDIA GPUs (torch, triton, cupy for CUDA 13)
uv sync --extra amdgpu     # student cluster with AMD GPUs (ROCm 7.2 wheels)
uv sync --extra cpu        # Euler (numba, no GPU stack)
source .venv/bin/activate
```

Validation compiles each kernel's C reference, so the machine needs a C compiler and an OpenBLAS that
`pkg-config openblas` finds (on a module system, `module load gcc openblas`). The harness runs each validating
child in a private user namespace; on a host that does not allow user namespaces, turn that off:

```sh
export HPCAGENT_BENCH_GRADING_SEAL=false   # only where `unshare --user true` fails
```

## 2. Write a Triton kernel

A kernel's Triton implementation is one file, `<module>_triton.py`, beside its `<module>_numpy.py` under
`hpcagent_bench/benchmarks/scientific_computing/<dwarf>/<kernel>/`. 95 of the SC kernels already ship one (the
repository's baseline); replacing it in your checkout makes the `triton` column run yours. A kernel without the
file records `status=unsupported` and is skipped in the plot.

The contract (entry point, arguments, outputs, precision, CuPy for library calls) is in
[triton_authoring.md](triton_authoring.md). Start from `atax_triton.py` (`dense_linear_algebra/atax/`). Write the
kernel precision-generic (accumulate in `x.dtype.element_ty`), since `fp32-if-allowed` runs most kernels in fp32
and a few in fp64.

Check one kernel quickly while you work on it:

```sh
hpcagent-bench run-framework -f triton -b atax -p S -d fp32-if-allowed -r 3   # validation: SUCCESS is the wiring check
```

## 3. The numba baseline on Euler (course staff, once)

One job, `N` ranks, one EPYC socket per rank. The kernel list is split over the ranks (`kernels[r::N]`); each rank
runs its kernels one after another with numba's thread pool sized to its socket (64 threads), so ranks never share
cores. `N=2` fills one dual-socket node; `N=8` spreads the track over 4 nodes and finishes about 4 times sooner.
Each kernel's wall time is capped (`CANON_KERNEL_TIMEOUT_SEC`, default 7200 s); a kernel that hits the cap or
crashes still records a row (`status=timeout` or `crash`).

```sh
#!/bin/bash
#SBATCH --job-name=sc-numba-baseline
#SBATCH --nodes=4                 # N / 2 nodes
#SBATCH --ntasks-per-node=2       # one rank per socket
#SBATCH --cpus-per-task=64        # the socket's 64 cores
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --constraint=EPYC_7742    # Euler's feature name for the 7742 nodes; check `sinfo -o "%N %f"`
set -euo pipefail
cd "$HOME/HPCAgent-Bench" && source .venv/bin/activate
export HPCAGENT_BENCH_RESULTS_DIR=$SCRATCH/sc-baseline      # canon.db lands here
OUT=$SCRATCH/sc-baseline/euler-numba                         # its name is the run label of every row
mkdir -p "$OUT"
args=(--column numba --out-root "$OUT" --tag scientific_computing --preset L --datatype fp32-if-allowed)
srun --ntasks=1 --nodes=1 hpcagent-bench job baseline "${args[@]}" --phase begin    # drops an earlier run's rows
srun --cpus-per-task=64 --cpu-bind=cores hpcagent-bench job baseline "${args[@]}" --phase run
srun --ntasks=1 --nodes=1 hpcagent-bench job baseline "${args[@]}" --phase finish   # removes the build tree
hpcagent-bench run-framework --summarize "$HPCAGENT_BENCH_RESULTS_DIR/canon.db" --canon-run euler-numba
```

`job baseline` runs `run-framework -f numba -p L -d fp32-if-allowed` once per kernel, the same command as step 4
with another column. Publish the resulting `canon.db` to the class (read-only); it holds one row per SC kernel
with its median time, datatype, validation and the node it ran on.

## 4. Your kernels on the student cluster

Start from a copy of the published baseline, so both runs end up in one file:

```sh
cp /path/to/published/canon.db $SCRATCH/canon.db
hpcagent-bench run-framework -f triton -b scientific_computing -p L -d fp32-if-allowed -r 20 \
    --canon-db $SCRATCH/canon.db --canon-run triton-$USER
```

`-b scientific_computing` selects the whole track; a dwarf (`-b dense_linear_algebra`) or one kernel works too.
Each kernel runs in a child process of its own, so a crash in one kernel does not stop the sweep. Add `-e` to
rerun only the kernels without a validated row (after fixing a few). On a 3-core node with one GPU the track takes
a few hours at `L`; run it as a batch job:

```sh
sbatch --gpus=1 --cpus-per-task=3 --time=06:00:00 --wrap \
  "source .venv/bin/activate && hpcagent-bench run-framework -f triton -b scientific_computing -p L \
   -d fp32-if-allowed -r 20 --canon-db $SCRATCH/canon.db --canon-run triton-$USER"
```

## 5. Plot the speedup per kernel

```sh
hpcagent-bench plot kernel-speedups --canon-db $SCRATCH/canon.db \
    --canon-run euler-numba --canon-run triton-$USER \
    --canon-columns triton --baseline numba \
    --series-label triton="Triton (GPU)" --title "Triton over numba (one EPYC 7742 socket)" \
    --out figures/triton_vs_numba.pdf
```

One point per kernel: numba's median divided by Triton's, on a log axis, with the geomean over the kernels both
columns validated. It also writes the numbers as `triton_vs_numba-kernels.csv` and `-summary.csv`. `--tag-file`
restricts the axis to a list of kernels (one per line).

For the numbers behind the plot (wins, losses, paired geomean with its bootstrap interval, per-kernel table):

```sh
hpcagent-bench compare --canon-db $SCRATCH/canon.db --canon-run euler-numba --canon-run triton-$USER \
    --pair numba:triton --out results/triton_vs_numba   # writes results/triton_vs_numba.csv and -kernels.csv
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `cannot seal grading children` | the host forbids user namespaces | `export HPCAGENT_BENCH_GRADING_SEAL=false` |
| `calls cblas_*, but no BLAS links for c here` | the C reference cannot find OpenBLAS | install or load OpenBLAS so `pkg-config openblas` resolves |
| a kernel records `unsupported` | it has no `<module>_triton.py` | write one (step 2) |
| `validated=0` | the output differs from the C reference beyond the tolerance | rerun at `-p S` and compare; check fp32 accumulation |
| a kernel is missing from the plot | one of the two rows is not validated | `hpcagent-bench run-framework --summarize canon.db` lists each row's status |
