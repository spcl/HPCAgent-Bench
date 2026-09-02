#!/usr/bin/env bash
# Build optarena-judge-agent-cuda and import it to SquashFS.
#
# Must run on an aarch64 GH200 node. Building it on x86_64 would mean qemu emulation of a
# multi-hour source build of gcc, llvm, MAGMA and PETSc, which is not a real option -- so the
# architecture is checked rather than emulated.
#
# Overrides: IMAGE_TAG, OUTPUT_SQSH, BASE_IMAGE, DACE_COMMIT.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-optarena-judge-agent-cuda:latest}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?SCRATCH must be set on CSCS}/ce-images/optarena-judge-agent-cuda.sqsh}"
BASE_IMAGE="${BASE_IMAGE:-jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.02-py3-alps6}"

arch="$(uname -m)"
if [[ "${arch}" != "aarch64" ]]; then
    echo "this image is aarch64/GH200; the build node is ${arch}. Emulating a source build of the" >&2
    echo "whole toolchain under qemu is not a workable substitute -- build it on a GH200 node." >&2
    exit 2
fi

mkdir -p "$(dirname "${OUTPUT_SQSH}")"

# Diskless nodes: temp + runtime dirs on /dev/shm, stale per-node podman state wiped (only a cache).
unset DBUS_SESSION_BUS_ADDRESS
export TMPDIR="/dev/shm/${USER}/tmp"
export XDG_RUNTIME_DIR="/dev/shm/${USER}/xdg"
rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" "/dev/shm/${USER}/tmp" "/dev/shm/${USER}/xdg"
mkdir -p "${TMPDIR}"
mkdir -p -m 0700 "${XDG_RUNTIME_DIR}"

# DaCe: resolve the TIP of extended HERE and pass the sha in. The Dockerfile cannot do this -- its
# layer cache keys on the command string, so a `--branch extended` clone is reused forever and the
# image ages into a pin nothing records. Resolving outside makes the sha part of the cache key, so
# the layer rebuilds exactly when extended moves and never otherwise.
DACE_COMMIT="${DACE_COMMIT:-$(git ls-remote https://github.com/spcl/dace.git refs/heads/extended | cut -f1)}"
[[ -n "${DACE_COMMIT}" ]] || { echo "could not resolve spcl/dace@extended" >&2; exit 2; }
printf 'dace @ %s\n' "${DACE_COMMIT}"

cd "${REPO_ROOT}"
# cgroupfs: with the systemd manager a dying logind session reaps podman mid-pull (silent rc=1).
podman --cgroup-manager=cgroupfs build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "DACE_COMMIT=${DACE_COMMIT}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  .

# enroot's exit code lies when its cleanup fails after a good write; gate on the artifact
# (listing forces a read of the inode table at file END, which a truncated image fails).
enroot import -x mount -o "${OUTPUT_SQSH}" "podman://${IMAGE_TAG}" || true
unsquashfs -l "${OUTPUT_SQSH}" opt >/dev/null

# The digest is the image's identity. Version suffixes are deliberately not in the NAME -- git plus
# this digest is what says which image a campaign ran, and a name cannot be kept honest by hand.
podman image inspect --format '{{.Digest}}' "${IMAGE_TAG}" | tee "${OUTPUT_SQSH}.digest"
printf 'Wrote %s\n' "${OUTPUT_SQSH}"
