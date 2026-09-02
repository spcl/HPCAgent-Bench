#!/usr/bin/env bash
# Build optarena-judge-agent-amd and export it as a squashfs enroot can mount.
#
# The tag carries NO version suffix. Version identity comes from git plus the image digest this
# script records, not from a name -- a "-v5" in the tag is what made two different images look
# like the same thing in a results table.
#
# Run it from anywhere; it derives the repository root itself and builds with the repo root as
# the context, because the Dockerfile COPYs requirements/, containers/agent and containers/judge.
#
#   containers/cluster/ce-images/judge-agent-amd/build.sh
#   OUTPUT_SQSH=$SCRATCH/ce-images/some-candidate.sqsh .../build.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-optarena-judge-agent-amd:latest}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?SCRATCH must be set on CSCS}/ce-images/optarena-judge-agent-amd.sqsh}"
BASE_IMAGE="${BASE_IMAGE:-docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
ROCM_ARCH="${ROCM_ARCH:-gfx942}"

mkdir -p "$(dirname "${OUTPUT_SQSH}")"

# Diskless nodes: temp and runtime dirs on /dev/shm, stale per-node podman state wiped (it is
# only a cache). DBUS_SESSION_BUS_ADDRESS is unset so podman does not try to talk to a session
# bus that is not there.
unset DBUS_SESSION_BUS_ADDRESS
export TMPDIR="/dev/shm/${USER}/tmp"
export XDG_RUNTIME_DIR="/dev/shm/${USER}/xdg"
rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" "/dev/shm/${USER}/tmp" "/dev/shm/${USER}/xdg"
mkdir -p "${TMPDIR}"
mkdir -p -m 0700 "${XDG_RUNTIME_DIR}"

# DaCe: resolve the TIP of extended HERE and pass the sha in. The Dockerfile cannot do this --
# its layer cache keys on the command string, so a `--branch extended` clone is reused forever
# and the image ages into a pin nothing records. Resolving outside makes the sha part of the
# cache key, so the layer rebuilds exactly when extended moves and never otherwise.
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

# The digest IS the version. Recorded next to the squashfs so a results table can name the exact
# image a campaign ran on without trusting a mutable tag.
podman image inspect --format '{{.Digest}}' "${IMAGE_TAG}" > "${OUTPUT_SQSH}.digest"
printf 'image digest %s\n' "$(cat "${OUTPUT_SQSH}.digest")"

# enroot's exit code lies when its cleanup fails after a good write, so gate on the ARTIFACT:
# listing forces a read of the inode table at file END, which a truncated image fails.
enroot import -x mount -o "${OUTPUT_SQSH}" "podman://${IMAGE_TAG}" || true
unsquashfs -l "${OUTPUT_SQSH}" opt >/dev/null
printf 'Wrote %s\n' "${OUTPUT_SQSH}"
