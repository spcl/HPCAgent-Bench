#!/usr/bin/env bash
# Build the GH200 vLLM image (arm64-only base) into its candidate squashfs, on a Daint node via
# build.sbatch. Build context is the repository root.
#
#   containers/images/vllm-cuda/build.sh
#   BASE_IMAGE=docker.io/vllm/vllm-openai:<tag>@sha256:<digest> EXTRA_BUILD_ARGS="VLLM_VERSION=<x.y.z> TORCH_CUDA=<12.9>" .../build.sh
set -euo pipefail

ulimit -c 0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=../build_common.sh
source "${SCRIPT_DIR}/../build_common.sh"
# shellcheck source=../images.env
source "${SCRIPT_DIR}/../images.env"

arch="$(uname -m)"
if [[ "${arch}" != "aarch64" ]]; then
    echo "this image is aarch64/GH200; the build node is ${arch}. Build it on a GH200 node." >&2
    exit 2
fi

IMAGE_TAG="${IMAGE_TAG:-hpcagent-bench-vllm-cuda:latest}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${CE_IMAGES:?set SCRATCH or CE_IMAGES}/${INFERENCE_VLLM_CUDA_CANDIDATE}}"
# Must equal the Dockerfile's ARG default.
BASE_IMAGE="${BASE_IMAGE:-docker.io/vllm/vllm-openai:v0.28.0-aarch64-cu129@sha256:60fa2715937e604931086a790fff2978c09995eff93439261ba09a79f02e9e68}"
IMAGE_VERSION="${IMAGE_VERSION:-dev}"
mkdir -p "$(dirname "${OUTPUT_SQSH}")"

ce_podman_env
cd "${REPO_ROOT}"

# Bare KEY=VALUE pairs for the ARGs a candidate varies (VLLM_VERSION, TORCH_CUDA with another base).
BUILD_ARGS=(--build-arg "IMAGE_VERSION=${IMAGE_VERSION}")
for kv in ${EXTRA_BUILD_ARGS:-}; do BUILD_ARGS+=(--build-arg "${kv}"); done
ce_pull_first "${SCRIPT_DIR}/Dockerfile" "vllm-cuda||${IMAGE_TAG}|${OUTPUT_SQSH}" -- "${BUILD_ARGS[@]}"
[[ "${CE_PULLED}" == 0 ]] || exit 0

ce_mirror_args
ce_cache_base_image
ce_build "${SCRIPT_DIR}/Dockerfile" "" "${IMAGE_TAG}" "${OUTPUT_SQSH}" "${MIRROR_ARGS[@]}" "${BUILD_ARGS[@]}"
