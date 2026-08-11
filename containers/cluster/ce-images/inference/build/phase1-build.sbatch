#!/usr/bin/env bash
#SBATCH --job-name=phase1-hostdiag
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase1-build-%j.out
#SBATCH --error=logs/phase1-build-%j.err

# Steps 0+1 of the rebuild chain: pack host SDK, build the phase-1 diag image, import to
# sqsh, install its EDF. The gate runs in the NEXT job so build and gate failures stay apart.
set -euo pipefail

# Slurm spools this script, so no BASH_SOURCE paths: everything anchors on ROOT.
ROOT="${VLLM_BUILD_ROOT:?set VLLM_BUILD_ROOT (the rebuild root, NOT the production image dir)}"
PHASE1_DIR="${ROOT}/beverin-rocm723-host-ofi-phase1"
IMAGE_TAG="rocm723-ofi-host-diag:phase1"
CONTEXT="${ROOT}/phase1-context"

mkdir -p "${ROOT}/phase1-passed" "${ROOT}/logs"

# Scratch copy of the phase-1 dir as build context, so the SDK tarball never dirties the tree.
rm -rf "${CONTEXT}"
mkdir -p "${CONTEXT}"
cp -a "${PHASE1_DIR}/." "${CONTEXT}/"
bash "${CONTEXT}/pack-beverin-host-sdk.sh" "${CONTEXT}/beverin-host-sdk.tar.gz"

# Storage on node-local /dev/shm (CSCS guidance); never clobber an existing config.
if [[ ! -f "${HOME}/.config/containers/storage.conf" ]]; then
    mkdir -p "${HOME}/.config/containers"
    cat >"${HOME}/.config/containers/storage.conf" <<EOF
[storage]
driver = "overlay"
runroot = "/dev/shm/${USER}/runroot"
graphroot = "/dev/shm/${USER}/root"
EOF
fi

# Diskless nodes: stage blobs on /dev/shm not tiny /var/tmp (--mem=0 above pays the cgroup bill),
# wipe stale per-node podman state (only a cache), private runtime dir out of logind's reach.
unset DBUS_SESSION_BUS_ADDRESS
export TMPDIR="/dev/shm/${USER}/tmp"
export XDG_RUNTIME_DIR="/dev/shm/${USER}/xdg"
rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" "/dev/shm/${USER}/tmp" "/dev/shm/${USER}/xdg"
mkdir -p "${TMPDIR}"
mkdir -p -m 0700 "${XDG_RUNTIME_DIR}"

# cgroupfs: with the systemd manager a dying logind session reaps podman mid-pull (silent rc=1).
podman --cgroup-manager=cgroupfs build \
    --file "${CONTEXT}/Containerfile.rocm723-ofi-host-diag" \
    --tag "${IMAGE_TAG}" \
    "${CONTEXT}" || { rc=$?; echo "podman build failed rc=${rc}" >&2; df -h /dev/shm /var/tmp >&2; exit "${rc}"; }

# enroot's exit code lies when its cleanup fails after a good write; gate on the artifact
# (listing forces a read of the inode table at file END, which a truncated image fails).
PHASE1_SQSH="${ROOT}/phase1-passed/rocm723-ofi-host-diag-phase1.sqsh"
enroot import -x mount -o "${PHASE1_SQSH}" "podman://${IMAGE_TAG}" || true
unsquashfs -l "${PHASE1_SQSH}" opt >/dev/null

# EDF with the image path resolved, for the gate job's srun --environment.
mkdir -p "${HOME}/.edf"
sed "s|\${VLLM_BUILD_ROOT}|${ROOT}|g" "${PHASE1_DIR}/rocm723-ofi-host-diag.toml" \
    >"${HOME}/.edf/rocm723-ofi-host-diag.toml"

printf 'phase1 image: %s\n' "${PHASE1_SQSH}"
