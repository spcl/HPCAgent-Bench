#!/usr/bin/env bash
set -euo pipefail

# Build the SGLang MI250X image and import it to a squashfs. Run it on an mi200 node via
# build.sbatch. Build context is the REPOSITORY ROOT, as for ../sglang.

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_<host>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
# shellcheck source=../build_common.sh
source "${SCRIPT_DIR}/../build_common.sh"

IMAGE_TAG="${IMAGE_TAG:-hpcagent-bench-sglang-mi200:latest}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?SCRATCH must be set on CSCS}/ce-images/hpcagent-bench-sglang-mi200.sqsh}"
# Pinned by DIGEST, and it MUST track the Dockerfile's ARG BASE_IMAGE: passing it overrides that default.
BASE_REPO="docker.io/lmsysorg/sglang-rocm:v0.5.19-rocm724-mi30x-20260908"
BASE_DIGEST="sha256:0405baaf36945fa8164c57d4f1b6b178bae5804fae606ff2db3816a6cab6dafc"
BASE_IMAGE="${BASE_IMAGE:-${BASE_REPO}@${BASE_DIGEST}}"

mkdir -p "$(dirname "${OUTPUT_SQSH}")"

# ROCM_ARCH from gpu_arch.env for this job's partition; an unknown partition stops before any pull.
ce_gpu_arch

ce_podman_env

cd "${REPO_ROOT}"
ce_mirror_args
ce_gpu_args
ce_cache_base_image

# cgroupfs: with the systemd manager a dying logind session reaps podman mid-pull (silent rc=1).
podman --cgroup-manager=cgroupfs build "${MIRROR_ARGS[@]}" "${GPU_ARGS[@]}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "BASE_IMAGE_REF=${BASE_IMAGE_REF:-${BASE_IMAGE}}" \
  --build-arg "ROCM_ARCH=${ROCM_ARCH}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  .

ce_export_image "${IMAGE_TAG}" "${OUTPUT_SQSH}"
