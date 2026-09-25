#!/usr/bin/env bash
set -euo pipefail

# Build the vLLM inference image and import it to a squashfs. Run it on a COMPUTE node via
# build.sbatch: the build pulls a ~30 GB ROCm base and compiles vLLM, flash-attn and aiter.
#
# The build context is the REPOSITORY ROOT, because the Dockerfile bakes in two files that live in
# the repo: the tuned fused_moe configs and the eager-PG sitecustomize patch. Nothing is reached
# from outside the image at RUN time, which is the property that matters -- a build input is under
# the image digest, an out-of-image PYTHONPATH is not.

# The cluster's core_pattern dumps into the crashing process's CWD; Slurm propagates the
# submitter's core limit, so the floor has to be set here to avoid littering the checkout.
ulimit -c 0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
# shellcheck source=../build_common.sh
source "${SCRIPT_DIR}/../build_common.sh"

IMAGE_TAG="${IMAGE_TAG:-hpcagent-bench-vllm:latest}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?SCRATCH must be set on CSCS}/ce-images/hpcagent-bench-vllm.sqsh}"
# Pinned by DIGEST, not by tag: rocm/pytorch publishes this release unsuffixed as rocm7.2_*, a
# mutable name.
BASE_REPO="docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1"
BASE_DIGEST="sha256:a3b65813621095e3389269417e963725b59310184588c9d2490d44e6e83fa01c"
BASE_IMAGE="${BASE_IMAGE:-${BASE_REPO}@${BASE_DIGEST}}"
# This MUST track the Dockerfile's ARG BASE_IMAGE default: passing it here OVERRIDES that default,
# so a stale line here builds a base the Dockerfile does not name.

mkdir -p "$(dirname "${OUTPUT_SQSH}")"

# ROCM_ARCH from gpu_arch.env for this job's partition; an unknown partition stops before any pull.
ce_gpu_arch

ce_podman_env

cd "${REPO_ROOT}"
# cgroupfs: with the systemd manager a dying logind session reaps podman mid-pull (silent rc=1).
ce_mirror_args

ce_gpu_args

ce_cache_base_image

# Pass-through for the ARGs a CANDIDATE image varies: VLLM_VERSION, AITER_REF and
# VLLM_ROCM_AITER_SWITCH. Empty by default, so the live hpcagent-bench-vllm.sqsh build is unchanged.
#   EXTRA_BUILD_ARGS="VLLM_VERSION=0.28.0 VLLM_ROCM_AITER_SWITCH=1" ./build.sh
EXTRA_ARGS=()
for kv in ${EXTRA_BUILD_ARGS:-}; do EXTRA_ARGS+=(--build-arg "${kv}"); done

podman --cgroup-manager=cgroupfs build "${MIRROR_ARGS[@]}" "${GPU_ARGS[@]}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "BASE_IMAGE_REF=${BASE_IMAGE_REF:-${BASE_IMAGE}}" \
  --build-arg "ROCM_ARCH=${ROCM_ARCH}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  .

ce_export_image "${IMAGE_TAG}" "${OUTPUT_SQSH}"
