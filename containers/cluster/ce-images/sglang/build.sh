#!/usr/bin/env bash
set -euo pipefail

# Build the SGLang inference image and import it to a squashfs. Run it on a COMPUTE node via
# build.sbatch: the base alone is 52 GB and cupy compiles from source.
#
# The build context is the REPOSITORY ROOT, because the Dockerfile bakes in the tuned fused_moe
# configs from there. Nothing is reached from outside the image at RUN time -- which is the point:
# flydsl used to arrive through PYTHONPATH=${SCRATCH}/pyprefix/sglang-rocm-mi30x, and an upgrade
# reached that way is invisible to the image digest.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
# shellcheck source=../build_common.sh
source "${SCRIPT_DIR}/../build_common.sh"

IMAGE_TAG="${IMAGE_TAG:-optarena-sglang:latest}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?SCRATCH must be set on CSCS}/ce-images/optarena-sglang.sqsh}"
# Pinned by DIGEST. The date stamp in the tag looks immutable and is not.
BASE_REPO="docker.io/lmsysorg/sglang-rocm:v0.5.19-rocm724-mi30x-20260908"
BASE_DIGEST="sha256:0405baaf36945fa8164c57d4f1b6b178bae5804fae606ff2db3816a6cab6dafc"
BASE_IMAGE="${BASE_IMAGE:-${BASE_REPO}@${BASE_DIGEST}}"
# This value MUST track the Dockerfile's ARG BASE_IMAGE default: passing it here OVERRIDES that
# default, so a stale line at this spot silently builds the wrong base while the Dockerfile reads
# correct. It is the reason a "rebuild" once produced ROCm 7.2.0 from a file that said 7.2.4.

mkdir -p "$(dirname "${OUTPUT_SQSH}")"

ce_podman_env

cd "${REPO_ROOT}"
# cgroupfs: with the systemd manager a dying logind session reaps podman mid-pull (silent rc=1).
ce_mirror_args

ce_gpu_args

ce_cache_base_image

podman --cgroup-manager=cgroupfs build "${MIRROR_ARGS[@]}" "${GPU_ARGS[@]}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "BASE_IMAGE_REF=${BASE_IMAGE_REF:-${BASE_IMAGE}}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  .

ce_export_image "${IMAGE_TAG}" "${OUTPUT_SQSH}"
