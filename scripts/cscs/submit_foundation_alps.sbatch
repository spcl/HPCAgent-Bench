#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# FOUNDATION TRACK on CSCS Alps (GH200, aarch64), under the Alps Container Engine.
#
# This is scripts/submit_deterministic.sbatch's Alps sibling: same rank -> kernel shard, same
# deterministic optimizers, but every step carries the container-engine flag (`--environment=$EDF`,
# read from container_backends.txt) because on Alps a step without it runs OUTSIDE the image.
# Deterministic means no vLLM and no judge: a deterministic optimizer emits the same artifact
# every run, so there is nothing to serve and nothing to score. A judged run is a different
# deployment (scripts/submit_launch.sbatch).
#
#   EDF=$SCRATCH/foundation.toml sbatch -A <account> -N 8 scripts/cscs/submit_foundation_alps.sbatch
#
# `-A <account>` is mandatory on Alps (or export SBATCH_ACCOUNT first).
#
# IMAGE SETUP. None is required: put a registry reference in the EDF's `image` field and the
# container engine pulls it, converts it to SquashFS and caches it. Pre-importing only skips
# that per-job pull:
#
#   enroot import -o $SCRATCH/ce-images/hpcagent_bench-aarch64.sqsh docker://<registry>#hpcagent_bench:cpu
#
# GH200 nodes are aarch64. One tag serves both architectures when it is a multi-arch manifest
# list -- build it once on a build box, never on the login node:
#
#   docker buildx build --platform linux/amd64,linux/arm64 --push \
#          -f containers/hpcagent_bench.Dockerfile --build-arg HW=cpu -t <registry>/hpcagent_bench:cpu .
#
# An amd64-only image will not run on a GH200 node, and the failure arrives as an exec-format
# error inside the first step rather than at submission.
#
# Overridable: FRAMEWORKS, BENCH, PRESET, REPEAT, TIMEOUT, DATATYPE.
#
#   BENCH=foundation@lvl2 PRESET=M FRAMEWORKS=dace_cpu,polly sbatch ... this script
#
# UNVERIFIED AGAINST THE SITE'S OWN SUBMISSION SCRIPTS. The Alps specifics here come from this
# repo's own notes (scripts/cscs/env.toml.example, docs/launch.md), which state that node
# allocation and srun submission are owned externally. The partition name, the account, the
# scratch layout and the reservation policy are the three things most likely to need a local
# edit before this runs as-is.

#SBATCH --job-name=hpcagent_bench-foundation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive                 # whole node -> full-core timing, no co-runners
#SBATCH --time=04:00:00
#SBATCH --output=results/foundation-alps-%j.out

set -euo pipefail

FRAMEWORKS="${FRAMEWORKS:-${FRAMEWORK:-dace_cpu}}"
BENCH="${BENCH:-foundation}"
PRESET="${PRESET:-L}"
REPEAT="${REPEAT:-10}"
TIMEOUT="${TIMEOUT:-200}"
DATATYPE="${DATATYPE:-}"
RANKS="${SLURM_JOB_NUM_NODES:-1}"

REPO="${HPCAGENT_BENCH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"

# The EDF is what puts a step inside the image. Without it the step silently runs on the bare
# node, where the toolchain and the pinned dace are absent -- so refuse now, by name, rather
# than let the whole allocation produce a log full of import errors.
if [[ -z "${EDF:-}" ]]; then
    echo "EDF is unset: point it at an Alps container-engine .toml" >&2
    echo "  cp scripts/cscs/foundation.toml.example \$SCRATCH/foundation.toml   # then edit 'image'" >&2
    echo "  EDF=\$SCRATCH/foundation.toml sbatch -A <account> $0" >&2
    exit 2
fi
if [[ ! -f "${EDF}" ]]; then
    echo "EDF=${EDF} does not exist" >&2
    exit 2
fi
export HPCAGENT_BENCH_REPO="${REPO}"   # the EDF interpolates this for its bind mount and workdir

# The flag that selects the container is read from the ONE spelling file the Python factory and
# the local bash launcher also read (hpcagent_bench/container_backends.txt, row `ce.srun_flag`),
# so Alps cannot drift from the rest of the harness on how CE is invoked.
CE_FLAG="$(sed -n 's/^ce\.srun_flag=//p' "${REPO}/hpcagent_bench/container_backends.txt")"
if [[ -z "${CE_FLAG}" ]]; then
    echo "container_backends.txt declares no ce.srun_flag; cannot select the Alps container" >&2
    exit 2
fi
CE=("${CE_FLAG}=${EDF}")

# DaCe writes one .so per SDFG variant per kernel. Keep that off the parallel FS, which turns
# thousands of small compiler writes into the job's bottleneck, and off /tmp, which is tmpfs on
# these nodes (the build would then compete with the run for RAM).
export DACE_default_build_folder="${DACE_BUILD_ROOT:-${SCRATCH:-${REPO}}}/dacecache-${SLURM_JOB_ID:-local}"
mkdir -p "${DACE_default_build_folder}"

# Results must outlive the allocation, so they live in the REPO, never node-local or memory-backed
# (recording.base_db_path refuses a tmpfs path outright). Each rank writes its OWN shard: WAL needs
# a -shm mapping no parallel FS provides, so one shared file across ranks is not an option.
RUNDIR="${REPO}/results/foundation-alps-${SLURM_JOB_ID:-local}"
mkdir -p "${RUNDIR}"

# Columns, dace pipeline and autopar capability, checked in the LIBRARY
# (hpcagent_bench/harness/preflight.py) -- the SAME call scripts/submit_deterministic.sbatch
# makes, so the two cannot drift. Run INSIDE the image: the login node's python is not the one
# that will run the sweep, and the core count that matters is the compute node's. Report goes to
# stderr, the thread-count exports to stdout, which is what is eval'd here. Sound only because
# every rank is --exclusive on its own node, so the srun steps below inherit that affinity.
eval "$(srun "${CE[@]}" --nodes=1 --ntasks=1 \
    hpcagent-bench preflight --frameworks "${FRAMEWORKS}" --print-env)"

echo "ranks=${RANKS} frameworks=${FRAMEWORKS} bench=${BENCH} preset=${PRESET} cores=${OMP_NUM_THREADS}"
echo "edf=${EDF} rundir=${RUNDIR}"

# Sharded by KERNEL, not by framework: kernel cost spans orders of magnitude while the framework
# list is short and fixed, so a framework-per-rank split parks most ranks behind the slowest
# column. Round-robin rather than contiguous blocks because neighbours in the sorted name list
# tend to be similar sizes. --cpu-bind=none so each task keeps its whole node's cores;
# run-framework forks EACH kernel, which is what contains a crashing or hanging kernel.
srun "${CE[@]}" --ntasks="${RANKS}" --ntasks-per-node=1 --cpu-bind=none bash -c '
set -euo pipefail
rank="${SLURM_PROCID:-0}"
IFS="," read -r -a fws <<< "'"${FRAMEWORKS}"'"
for fw in "${fws[@]}"; do
    echo "[rank ${rank}/'"${RANKS}"'] framework=${fw} cores=${OMP_NUM_THREADS}"
    hpcagent-bench run-framework \
        --benchmark "'"${BENCH}"'" --framework "${fw}" --preset "'"${PRESET}"'" \
        --repeat "'"${REPEAT}"'" --timeout "'"${TIMEOUT}"'" \
        --shard "${rank}/'"${RANKS}"'" \
        --csv "'"${RUNDIR}"'/shard-${rank}.csv" \
        --validate --ignore-errors '"${DATATYPE:+--datatype ${DATATYPE}}"'
done
'

# The merged failure count IS the job status, so a shard that stopped compiling fails the job
# instead of hiding in one rank's log.
echo "=== merged ==="
srun "${CE[@]}" --nodes=1 --ntasks=1 \
    hpcagent-bench run-framework --summarize "${RUNDIR}"/shard-*.csv
