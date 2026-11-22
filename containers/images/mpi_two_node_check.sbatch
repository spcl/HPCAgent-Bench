#!/usr/bin/env bash
#SBATCH --job-name=mpi-two-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --no-requeue
# The fabric check a one-node verify cannot make: MPI across two nodes of an image, under its production EDF.
#
#   EDF=hpcagent-bench-judge-amd-latest sbatch -A <account> -p mi300 containers/images/mpi_two_node_check.sbatch
#   EDF=... sbatch -A <account> -p normal containers/images/mpi_two_node_check.sbatch       # GH200 (Daint)
#
# Passes when one COMM_WORLD spans two distinct nodes, a host and a GPU-aware allreduce of device buffers return
# the expected sums, and rank 0's MPICH opened libfabric provider cxi. One-node verifies miss what this catches:
# the cross-node MPI_Init assertion of a ~pci hwloc (756270) and a silent tcp fallback.
set -uo pipefail

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
: "${EDF:?set EDF to the EDF of the image under test, a path or an EDF name on EDF_PATH}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/containers/images/lib/mpi_two_node.c" ]] \
    && here="${SLURM_SUBMIT_DIR}/containers/images"
work="$(mktemp -d "${SCRATCH:-/tmp}/mpi-two-node.XXXXXX")"
cp "${here}/lib/mpi_two_node.c" "${work}/"
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
cpus=$(( ${SLURM_CPUS_ON_NODE:-8} / ${SLURM_NTASKS_PER_NODE:-4} ))
srun -N1 -n1 -w "${nodes[0]}" -c 8 --environment="${EDF}" bash -c "
    cd '${work}' && if command -v hipcc >/dev/null; then
        hipcc -x hip -I/opt/view/include mpi_two_node.c -L/opt/view/lib -lmpi -o mpi_two_node.x
    else
        mpicc -DUSE_CUDA -I\"\${CUDA_HOME:-/usr/local/cuda}/include\" mpi_two_node.c -L\"\${CUDA_HOME:-/usr/local/cuda}/lib64\" -lcudart -o mpi_two_node.x
    fi" || { echo "MPI2N FAIL: the probe did not build"; exit 1; }
MPIR_CVAR_DEBUG_SUMMARY=1 srun -N2 --ntasks-per-node="${SLURM_NTASKS_PER_NODE:-4}" -c "${cpus}" --mpi=pmi2 \
    --kill-on-bad-exit=1 --time=10 --environment="${EDF}" "${work}/mpi_two_node.x" > "${work}/run.out" 2>&1
rc=$?
cat "${work}/run.out"
provider="$(sed -nE 's/^libfabric provider: *([A-Za-z0-9_;+-]+).*/\1/p' "${work}/run.out" | head -1)"
if (( rc == 0 )) && grep -q '^MPI2N PASS' "${work}/run.out" && [[ "${provider}" == cxi ]]; then
    echo "MPI2N OK: two nodes, provider ${provider}"
    exit 0
fi
echo "MPI2N FAIL: rc=${rc} provider=${provider:-none}"
exit 1
