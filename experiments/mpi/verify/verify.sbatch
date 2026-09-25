#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Re-runnable verification of the MPI / RCCL stack the ML-op scaling judge relies on, in the judge
# CE image (CE runtime only, no enroot), on 2 mi300 nodes:
#
#   0-build   1 rank    compile every probe with the HARNESS's resolved mpi + rccl flags (not
#                       mpicc), ldd: libmpi = image MPICH, librccl = image ROCm.  => "MPI links"
#   A-1n4r    1 node x 4 ranks, B-2n8r  2 nodes x 8 ranks, each:
#               mpi_hello          one world of the right size / node count, library = MPICH
#               gpuaware           MPIX_GPU_query_support(HIP) + MPI_T GPU cvars + device-pointer
#                                  Sendrecv/Allreduce + host vs hipMalloc bandwidth
#               gpuaware_control   same under MPIR_CVAR_ENABLE_GPU=0: the query MUST flip to 0
#               gpuaware_noipc     same with the GPU IPC / LMT route off: which route failed
#               gpu_initiated      MPIX_Stream on a hipStream + *_enqueue (UNSUPPORTED != FAIL)
#               rccl               RCCL all-reduce bf16 + fp32, busbw, NET/OFI vs NET/Socket
#   C-nested  from inside a CE step: `srun --overlap --mpi=pmi2 --environment=<EDF> -N2 -n8`,
#             then the judge's own launcher (hpcagent_bench.harness.mpi_gang) on 8 ranks.
#
# Step timeouts are a backstop only: each probe carries its own watchdogs, so a hung MPI call
# leaves a verdict in seconds. Their sum (420 + 2 x 960 + 300 s) fits the 45-minute limit even if
# every one of them fires.
#
# Result: ${SCRATCH}/hpcagent-bench-logs/mlscale-verify-<jobid>/results.txt (one row per check,
# PASS / FAIL / UNSUPPORTED + key numbers); per-step logs beside it. Exit 1 on any FAIL.
#
# Re-run from the checkout under test (its code is what gets resolved and compiled):
#   cd hpcagent-bench-wt/<worktree> && sbatch experiments/mpi/verify/verify.sbatch
#   EDF_NAME=hpcagent-bench-agent-mi300-latest sbatch experiments/mpi/verify/verify.sbatch  # other image
#
#SBATCH --job-name=mlscale-verify
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

set -uo pipefail
ulimit -c 0

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO="${HPCAGENT_BENCH_REPO:-$(cd -- "${SUBMIT_DIR}" && git rev-parse --show-toplevel)}"
EDF_NAME="${EDF_NAME:-hpcagent-bench-judge-mi300-latest}"
# An absolute path, not the name: the nested srun resolves it from inside the container.
EDF="${EDF:-${HOME}/.edf/${EDF_NAME}.toml}"
WORK="${SCRATCH:?}/hpcagent-bench-logs/mlscale-verify-${SLURM_JOB_ID:-local}"
V="${REPO}/experiments/mpi/verify"
mkdir -p "${WORK}/logs"
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ -f "${EDF}" ]] || { echo "EDF ${EDF} does not exist" >&2; exit 2; }
printf 'repo: %s (%s)\nedf:  %s\nnodes: %s\nwork: %s\n\n' "${REPO}" "$(git -C "${REPO}" rev-parse --short HEAD)" \
    "${EDF}" "${NODES[*]}" "${WORK}"

export HWLOC_COMPONENTS=-opencl,-levelzero,-gl
export HPCAGENT_BENCH_MPI_GANG_NODELIST="$(IFS=,; printf '%s' "${NODES[*]}")"

# step <name> <timeout_s> <srun args...> -- <cmd...>: one CE step, log + rc under ${WORK}/logs.
#
# --overlap and --time are both about the PREVIOUS step. Killing the srun CLIENT (the timeout
# below, or an aborted rank) does not reap the step: it keeps its nodes, and the next srun then
# waits on "Requested nodes are busy" until its own timeout -- which is how one hung probe turned
# every later step into a no-verdict failure. --overlap lets the next step share those nodes, and
# --time makes Slurm itself end a step whose client is gone. Both are needed: the first stops the
# wait, the second stops the leak.
step() {
    local name=$1 to=$2 args=() rc=0
    shift 2
    while [[ "$1" != "--" ]]; do args+=("$1"); shift; done
    shift
    echo "===== ${name} $(date +%T)"
    timeout --signal=KILL "${to}" srun --overlap --time="$(( to / 60 + 1 ))" "${args[@]}" --environment="${EDF}" "$@" \
        >"${WORK}/logs/${name}.log" 2>&1 || rc=$?
    echo "${rc}" >"${WORK}/logs/${name}.rc"
    grep '^VERDICT' "${WORK}/logs/${name}.log" || echo "(no verdict, rc=${rc})"
}

B="${WORK}/bin"
step 0-build__build 420 --nodes=1 --ntasks=1 --cpus-per-task=24 --nodelist="${NODES[0]}" -- \
    "${V}/verify.sh" build "${WORK}"

# <tag> <nodes> <ranks>
run_set() {
    local tag=$1 nodes=$2 ranks=$3
    local launch=(--nodes="${nodes}" --ntasks="${ranks}" --ntasks-per-node=4 --cpus-per-task=24 --hint=nomultithread
        --mpi=pmi2 --kill-on-bad-exit=1)
    step "${tag}__mpi_hello" 120 "${launch[@]}" -- "${B}/mpi_hello" "${ranks}" "${nodes}"
    step "${tag}__gpuaware" 200 "${launch[@]}" -- env MPIR_CVAR_DEBUG_SUMMARY=1 "${B}/gpuaware"
    step "${tag}__gpuaware_control" 90 "${launch[@]}" -- env MPIR_CVAR_ENABLE_GPU=0 "${B}/gpuaware" control
    step "${tag}__gpu_initiated" 150 "${launch[@]}" -- env MPIR_CVAR_CH4_RESERVE_VCIS=1 "${B}/gpu_initiated"
    step "${tag}__gpuaware_noipc" 200 "${launch[@]}" -- env MPIR_CVAR_CH4_IPC_GPU_P2P_THRESHOLD=1073741824 \
        "${B}/gpuaware"
    step "${tag}__rccl" 200 "${launch[@]}" -- env NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET "${B}/rccl_allreduce"
}
run_set A-1n4r 1 4
run_set B-2n8r 2 8

step C-nested__nested 300 --nodes=1 --ntasks=1 --cpus-per-task=24 --nodelist="${NODES[0]}" -- \
    "${V}/verify.sh" nested "${WORK}" "${EDF}"

echo
"${V}/verify.sh" report "${WORK}"
