#!/usr/bin/env bash
set -euo pipefail

# Build the SGLang inference image and import it to a squashfs. Run it on a COMPUTE node via
# build.sbatch: the base alone is 52 GB and the Slingshot stack compiles from source.
#
# The build context is the REPOSITORY ROOT, because the Dockerfile bakes in the tuned fused_moe
# configs from there. Nothing is reached from outside the image at RUN time -- which is the point:
# flydsl used to arrive through PYTHONPATH=${SCRATCH}/pyprefix/sglang-rocm-mi30x, and an upgrade
# reached that way is invisible to the image digest.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-optarena-sglang:latest}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?SCRATCH must be set on CSCS}/ce-images/optarena-sglang.sqsh}"
# Pinned by DIGEST. The date stamp in the tag looks immutable and is not.
BASE_REPO="docker.io/lmsysorg/sglang-rocm:v0.5.18-rocm720-mi30x-20260822"
BASE_DIGEST="sha256:4af1a96f988c523fd2848138b035eb2bd7654454d4f371e8ab0a379d50ee3feb"
BASE_IMAGE="${BASE_IMAGE:-${BASE_REPO}@${BASE_DIGEST}}"
# The Slingshot Host Software release the host runs: the login node reports
# cray-libcxi-1.0.2-SHS13.1.0. Override it if the host driver moves.
SHS_REF="${SHS_REF:-release/shs-13.1.0}"

mkdir -p "$(dirname "${OUTPUT_SQSH}")"

# Diskless nodes: temp + runtime dirs on /dev/shm, stale per-node podman state wiped (only a cache).
unset DBUS_SESSION_BUS_ADDRESS
export TMPDIR="/dev/shm/${USER}/tmp"
export XDG_RUNTIME_DIR="/dev/shm/${USER}/xdg"
# The wipe goes through `podman unshare`: an image layer under root/overlay/*/diff is owned by
# a SUBUID, so a plain rm cannot touch it and leaves a half-deleted store the next build
# dies on. unshare enters the user namespace where those subuids map to this user.
podman unshare rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" 2>/dev/null || true
rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" "/dev/shm/${USER}/tmp" "/dev/shm/${USER}/xdg"
mkdir -p "${TMPDIR}"
mkdir -p -m 0700 "${XDG_RUNTIME_DIR}"

cd "${REPO_ROOT}"
# cgroupfs: with the systemd manager a dying logind session reaps podman mid-pull (silent rc=1).
podman --cgroup-manager=cgroupfs build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "SHS_REF=${SHS_REF}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  .

# Record what was actually built. The tag carries no version by design, so the digest and the
# in-image manifest are the only provenance a later run can be attributed to.
podman image inspect --format '{{.Digest}}' "${IMAGE_TAG}" | tee "${OUTPUT_SQSH}.image-digest"
podman run --rm "${IMAGE_TAG}" cat /opt/ofi/BUILD-MANIFEST.txt

# enroot's exit code lies when its cleanup fails after a good write; gate on the artifact
# (listing forces a read of the inode table at file END, which a truncated image fails).
enroot import -x mount -o "${OUTPUT_SQSH}" "podman://${IMAGE_TAG}" || true
unsquashfs -l "${OUTPUT_SQSH}" opt >/dev/null
printf 'Wrote %s\n' "${OUTPUT_SQSH}"
