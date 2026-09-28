#!/usr/bin/env bash
set -euo pipefail

# Build the vLLM inference image and import it to a squashfs. Run it on a COMPUTE node via
# build.sbatch. The build context is the REPOSITORY ROOT, because the Dockerfile bakes in the tuned
# fused_moe configs that live in the repo.

# Slurm propagates the submitter's core limit and a dump lands in the CWD (an inode-quota checkout).
ulimit -c 0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=../build_common.sh
source "${SCRIPT_DIR}/../build_common.sh"

IMAGE_TAG="${IMAGE_TAG:-hpcagent-bench-vllm:latest}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${CE_IMAGES:?set SCRATCH or CE_IMAGES}/hpcagent-bench-vllm.sqsh}"
BASE_IMAGE="${BASE_IMAGE:-$(ce_dockerfile_base "${SCRIPT_DIR}/Dockerfile")}"

mkdir -p "$(dirname "${OUTPUT_SQSH}")"

# ROCM_ARCH: gpu_arch.env's AMD target list; the base carries device code for every one of them.
ce_amd_targets

ce_podman_env

# Pass-through for Dockerfile ARGs (a BASE_IMAGE of another release names its VLLM_VERSION too).
BUILD_ARGS=(--build-arg "ROCM_ARCH=${ROCM_ARCH}")
for kv in ${EXTRA_BUILD_ARGS:-}; do BUILD_ARGS+=(--build-arg "${kv}"); done
ce_pull_first "${SCRIPT_DIR}/Dockerfile" "vllm||${IMAGE_TAG}|${OUTPUT_SQSH}" -- "${BUILD_ARGS[@]}"
[[ "${CE_PULLED}" == 0 ]] || exit 0

ce_mirror_args
ce_gpu_args
ce_cache_base_image
ce_build "${SCRIPT_DIR}/Dockerfile" "" "${IMAGE_TAG}" "${OUTPUT_SQSH}" "${MIRROR_ARGS[@]}" "${GPU_ARGS[@]}" "${BUILD_ARGS[@]}"
