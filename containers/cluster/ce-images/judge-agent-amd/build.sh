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
# shellcheck source=../build_common.sh
source "${SCRIPT_DIR}/../build_common.sh"

IMAGE_TAG="${IMAGE_TAG:-optarena-judge-agent-amd:latest}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?SCRATCH must be set on CSCS}/ce-images/optarena-judge-agent-amd.sqsh}"
# Pinned by DIGEST, matching the Dockerfile's ARG default. Passing the bare tag here would
# OVERRIDE that default and quietly unpin the build, and the base.name label would then
# record a mutable reference. Same shape as the three inference builders.
BASE_REPO="docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1"
BASE_DIGEST="sha256:a3b65813621095e3389269417e963725b59310184588c9d2490d44e6e83fa01c"
BASE_IMAGE="${BASE_IMAGE:-${BASE_REPO}@${BASE_DIGEST}}"
ROCM_ARCH="${ROCM_ARCH:-gfx942}"

mkdir -p "$(dirname "${OUTPUT_SQSH}")"

ce_podman_env

# DaCe: resolve the TIP of extended HERE and pass the sha in. The Dockerfile cannot do this --
# its layer cache keys on the command string, so a '--branch extended' clone is reused forever
# and the image ages into a pin nothing records. Resolving outside makes the sha part of the
# cache key, so the layer rebuilds exactly when extended moves and never otherwise.
DACE_COMMIT="${DACE_COMMIT:-$(git ls-remote https://github.com/spcl/dace.git refs/heads/extended | cut -f1)}"
[[ -n "${DACE_COMMIT}" ]] || { echo "could not resolve spcl/dace@extended" >&2; exit 2; }
printf 'dace @ %s\n' "${DACE_COMMIT}"

cd "${REPO_ROOT}"
ce_mirror_args
ce_require_mirror_commit "spcl/dace.git" "${DACE_COMMIT}"

# Spack binary buildcache on scratch: gcc 16 and llvm 22 are 60-80 minutes this image has paid
# repeatedly, every time to fail at something after them. The Dockerfile pushes here after each
# install and registers it as a mirror when non-empty; both halves no-op without the mount.
SPACK_BUILDCACHE="${SPACK_BUILDCACHE:-${SCRATCH:?}/spack-buildcache}"
mkdir -p "${SPACK_BUILDCACHE}"
CACHE_ARGS=(-v "${SPACK_BUILDCACHE}:/spack-buildcache:rw")
printf 'spack buildcache %s\n' "${SPACK_BUILDCACHE}"

# cgroupfs, not systemd: a dying logind session reaps podman mid-pull under the systemd manager,
# with a silent rc=1.
podman --cgroup-manager=cgroupfs build "${MIRROR_ARGS[@]}" "${CACHE_ARGS[@]}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "DACE_COMMIT=${DACE_COMMIT}" \
  --build-arg "ROCM_ARCH=${ROCM_ARCH}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  .

ce_export_image "${IMAGE_TAG}" "${OUTPUT_SQSH}"
