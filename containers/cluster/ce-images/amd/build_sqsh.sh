#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-optarena-ce:amd-mi300-v4}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?SCRATCH must be set on CSCS}/ce-images/optarena-ce-amd-mi300-v4.sqsh}"
BASE_IMAGE="${BASE_IMAGE:-docker.io/rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
ROCM_ARCH="${ROCM_ARCH:-gfx942}"

mkdir -p "$(dirname "${OUTPUT_SQSH}")"

# Diskless nodes: temp + runtime dirs on /dev/shm, stale per-node podman state wiped (only a cache).
unset DBUS_SESSION_BUS_ADDRESS
export TMPDIR="/dev/shm/${USER}/tmp"
export XDG_RUNTIME_DIR="/dev/shm/${USER}/xdg"
rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" "/dev/shm/${USER}/tmp" "/dev/shm/${USER}/xdg"
mkdir -p "${TMPDIR}"
mkdir -p -m 0700 "${XDG_RUNTIME_DIR}"

# DaCe: resolve the TIP of extended HERE, and pass the sha in. The Dockerfile cannot do this --
# its layer cache keys on the command string, so a `--branch extended` clone is reused forever and
# the image ages into a pin. Resolving outside makes the sha part of the cache key, so the layer
# rebuilds exactly when extended moves and never otherwise.
DACE_COMMIT="${DACE_COMMIT:-$(git ls-remote https://github.com/spcl/dace.git refs/heads/extended | cut -f1)}"
[[ -n "${DACE_COMMIT}" ]] || { echo "could not resolve spcl/dace@extended" >&2; exit 2; }
printf 'dace @ %s\n' "${DACE_COMMIT}"

cd "${REPO_ROOT}"
# cgroupfs: with the systemd manager a dying logind session reaps podman mid-pull (silent rc=1).
podman --cgroup-manager=cgroupfs build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "DACE_COMMIT=${DACE_COMMIT}" \
  --build-arg "ROCM_ARCH=${ROCM_ARCH}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  .

# enroot's exit code lies when its cleanup fails after a good write; gate on the artifact
# (listing forces a read of the inode table at file END, which a truncated image fails).
enroot import -x mount -o "${OUTPUT_SQSH}" "podman://${IMAGE_TAG}" || true
unsquashfs -l "${OUTPUT_SQSH}" opt >/dev/null
printf 'Wrote %s\n' "${OUTPUT_SQSH}"
