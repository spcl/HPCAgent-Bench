#!/usr/bin/env bash
# Build the GH200 vLLM inference image and import it to a squashfs. Run it on a Daint GH200 node via
# build.sbatch: the base is ~10 GB compressed and arm64-only, so an x86_64 host would emulate it.
#
# Build context is the repository root, like every image in ce-images, so a later COPY of a repo
# file needs no change here.
#
#   containers/cluster/ce-images/vllm-cuda/build.sh
#   EXTRA_BUILD_ARGS="BASE_IMAGE=docker.io/vllm/vllm-openai:v0.30.0-aarch64-cu129@sha256:<digest>" .../build.sh
set -euo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_<host>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
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
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?SCRATCH must be set on CSCS}/ce-images/${INFERENCE_VLLM_CUDA_SQSH%.sqsh}-candidate.sqsh}"
# MUST track the Dockerfile's ARG default: passing it here overrides that default.
BASE_IMAGE="${BASE_IMAGE:-docker.io/vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56}"
IMAGE_VERSION="${IMAGE_VERSION:-dev}"
mkdir -p "$(dirname "${OUTPUT_SQSH}")"

ce_podman_env
cd "${REPO_ROOT}"
ce_mirror_args
ce_cache_base_image

# Bare KEY=VALUE pairs for the ARGs a candidate varies (VLLM_VERSION with a different base).
EXTRA_ARGS=()
for kv in ${EXTRA_BUILD_ARGS:-}; do EXTRA_ARGS+=(--build-arg "${kv}"); done

# cgroupfs: with the systemd manager a dying logind session reaps podman mid-pull (silent rc=1).
podman --cgroup-manager=cgroupfs build "${MIRROR_ARGS[@]}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "BASE_IMAGE_REF=${BASE_IMAGE_REF:-${BASE_IMAGE}}" \
  --build-arg "IMAGE_VERSION=${IMAGE_VERSION}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  .

ce_export_image "${IMAGE_TAG}" "${OUTPUT_SQSH}"
