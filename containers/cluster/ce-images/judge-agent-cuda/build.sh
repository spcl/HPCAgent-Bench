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
# shellcheck source=../build_common.sh
source "${SCRIPT_DIR}/../build_common.sh"

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

ce_podman_env

# DaCe: resolve the TIP of extended HERE and pass the sha in. The Dockerfile cannot do this -- its
# layer cache keys on the command string, so a '--branch extended' clone is reused forever and the
# image ages into a pin nothing records. Resolving outside makes the sha part of the cache key, so
# the layer rebuilds exactly when extended moves and never otherwise.
DACE_COMMIT="${DACE_COMMIT:-$(git ls-remote https://github.com/spcl/dace.git refs/heads/extended | cut -f1)}"
[[ -n "${DACE_COMMIT}" ]] || { echo "could not resolve spcl/dace@extended" >&2; exit 2; }
printf 'dace @ %s\n' "${DACE_COMMIT}"

cd "${REPO_ROOT}"
# cgroupfs: with the systemd manager a dying logind session reaps podman mid-pull (silent rc=1).
ce_mirror_args

ce_cache_base_image

podman --cgroup-manager=cgroupfs build "${MIRROR_ARGS[@]}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "DACE_COMMIT=${DACE_COMMIT}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  .

ce_export_image "${IMAGE_TAG}" "${OUTPUT_SQSH}"
