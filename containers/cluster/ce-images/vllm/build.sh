#!/usr/bin/env bash
set -euo pipefail

# Build the vLLM inference image and import it to a squashfs. Run it on a COMPUTE node via
# build.sbatch: the build pulls a ~30 GB ROCm base and compiles vLLM, flash-attn and aiter.
#
# The build context is the REPOSITORY ROOT, because the Dockerfile bakes in two files that live in
# the repo: the tuned fused_moe configs and the eager-PG sitecustomize patch. Nothing is reached
# from outside the image at RUN time, which is the property that matters -- a build input is under
# the image digest, an out-of-image PYTHONPATH is not.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
# shellcheck source=../build_common.sh
source "${SCRIPT_DIR}/../build_common.sh"

IMAGE_TAG="${IMAGE_TAG:-optarena-vllm:latest}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?SCRATCH must be set on CSCS}/ce-images/optarena-vllm.sqsh}"
# Pinned by DIGEST, not by tag. rocm/pytorch has no 7.2.0-suffixed tag at all -- the 7.2.0 release
# is published unsuffixed as rocm7.2_* -- and an unsuffixed tag is exactly the mutable name the
# consolidation exists to stop trusting.
BASE_REPO="docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1"
BASE_DIGEST="sha256:a3b65813621095e3389269417e963725b59310184588c9d2490d44e6e83fa01c"
BASE_IMAGE="${BASE_IMAGE:-${BASE_REPO}@${BASE_DIGEST}}"

mkdir -p "$(dirname "${OUTPUT_SQSH}")"

ce_podman_env

cd "${REPO_ROOT}"
# cgroupfs: with the systemd manager a dying logind session reaps podman mid-pull (silent rc=1).
ce_mirror_args

ce_gpu_args

ce_cache_base_image

podman --cgroup-manager=cgroupfs build "${MIRROR_ARGS[@]}" "${GPU_ARGS[@]}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  .

ce_export_image "${IMAGE_TAG}" "${OUTPUT_SQSH}"
# Provenance, not a gate: sglang stages the OFI build at /opt/ofi and the vLLM images at
# /opt/aws-ofi-nccl. `|| true` because a missing manifest must not fail a build whose
# squashfs and archive are already written and verified -- which is exactly what it did.
podman run --rm "${IMAGE_TAG}" cat /opt/aws-ofi-nccl/BUILD-MANIFEST.txt || true
