#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Can this image's MPI take a DEVICE pointer? Two ranks, one node, one GPU each.
#
# Point it at any image; the answer is a property of the image, not of the harness:
#   sbatch reproducibility/mpi/gpu-aware-mpi.sbatch                             # v6, expect GPU-AWARE
#   EDF=<other-image> sbatch reproducibility/mpi/gpu-aware-mpi.sbatch   # any other image
#
# NO expected answer is recorded for v5 here. The earlier "v5 is NOT GPU-AWARE" reading came from a
# probe that asked Open MPI's question of an MPICH, which reports a false NO for every MPICH (see
# gpu_aware_mpi_probe.c); v5 has not been re-tested since that was fixed, so its state is unknown
# rather than negative. Writing a stale expectation into a comment is how a broken test keeps
# looking like a passing one.
#
# Read gpu_aware_mpi_probe.c before reading the result: on an MI300A the functional half of this
# test can pass on a NON-GPU-aware MPI, because host memory is device-addressable on an APU. The
# decisive line is the GPU-support query.
#
#SBATCH --job-name=gpu-aware-mpi
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/optarena-logs/gpu-aware-mpi-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/optarena-logs/gpu-aware-mpi-%j.out

set -Eeuo pipefail

# Beverin's core_pattern is the machine-global core_%h_%p and a dump lands in the crashing
# process's CWD; the HOME quota here is inodes.
ulimit -c 0
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO="${HPCAGENT_BENCH_REPO:-$(cd -- "${SUBMIT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || echo "${SUBMIT_DIR}")}"
EDF="${EDF:-optarena-amd-mi300-latest}"
WORK="${SCRATCH:?}/optarena-logs/gpu-aware-mpi-${SLURM_JOB_ID:-local}"
mkdir -p "${WORK}"

printf 'image: %s\nrepo:  %s\nwork:  %s\n\n' "${EDF}" "${REPO}" "${WORK}"

srun --ntasks=1 --cpus-per-task=24 --environment="${EDF}" bash -c '
set -Eeuo pipefail
ulimit -c 0
REPO='"${REPO}"'; WORK='"${WORK}"'
cd -- "${WORK}"
export HWLOC_COMPONENTS=-opencl,-levelzero,-gl

echo "################ which MPI"
for w in mpicc mpiexec mpirun.openmpi hipcc; do printf "%-14s %s\n" "$w" "$(command -v $w || echo MISSING)"; done
mpicc -show 2>&1 | head -1
echo

echo "################ build"
# hipcc compiles the HIP kernel; MPI comes in through mpicc -show, so the probe links whatever
# MPI this image would actually use rather than one named here.
MPIFLAGS="$(mpicc -show | cut -d" " -f2-)"
# -x hip compiles as C++, which drags in the MPI C++ BINDINGS -- and Open MPI 4.1 op_inln.h
# has function-pointer casts that clang rejects under -Werror. Those headers are deprecated and
# nothing here uses them, so skip them rather than lowering the warning level on our own code.
hipcc -x hip -O2 -Wall -Wextra -Werror --offload-arch=gfx942 \
    -DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1 -DMPI_NO_CPPBIND=1 \
    "${REPO}/reproducibility/mpi/gpu_aware_mpi_probe.c" ${MPIFLAGS} -o probe
echo "built"
echo

echo "################ launcher selection"
# A launcher here has two ways to be wrong and only one of them is loud:
#   * it starts ranks OUTSIDE the container (srun; and mpiexec/Hydra, which auto-detects Slurm
#     and launches hydra_pmi_proxy through srun) -- that fails visibly, as a missing binary;
#   * it starts them inside but without wiring PMI, so each rank is its own COMM_WORLD of size 1
#     -- P processes that each do the whole job, and NOTHING reports an error.
# So the launcher is CHOSEN by experiment against a trivial 2-rank world-size test, not assumed
# from the MPI family, and the probe itself aborts unless it gets exactly 2 ranks.
mpicc -O0 "${REPO}/reproducibility/mpi/mpi_worldsize.c" -o worldsize

LAUNCHER=""
try_launcher() {
    local out
    out="$(timeout 120 "$@" -n 2 ./worldsize 2>/dev/null | grep -o "WORLD=[0-9]*" | head -1 || true)"
    printf "  %-56s %s\n" "$*" "${out:-no output}"
    [ "${out}" = "WORLD=2" ]
}
for cand in \
    "mpirun.openmpi --allow-run-as-root --oversubscribe" \
    "mpiexec -launcher fork -rmk user" \
    "mpiexec" \
    "mpirun"; do
    # shellcheck disable=SC2086
    if command -v ${cand%% *} >/dev/null 2>&1 && try_launcher ${cand}; then
        LAUNCHER="${cand}"; break
    fi
done
if [ -z "${LAUNCHER}" ]; then
    echo "no launcher formed a 2-rank COMM_WORLD in this container; cannot test GPU-awareness" >&2
    exit 3
fi
echo "using: ${LAUNCHER}"
echo

echo "################ run (2 ranks)"
# shellcheck disable=SC2086
timeout 300 ${LAUNCHER} -n 2 ./probe
'
