#!/usr/bin/env bash
set -euo pipefail

# Build the vLLM 0.27.1 inference image and import it to a squashfs. Run it on a COMPUTE node via
# build.sbatch: the build pulls a ~30 GB ROCm base and compiles vLLM, flash-attn and aiter.
#
# The SIBLING of ce-images/vllm (0.23.0), which stays as the known-good fallback. This one is the
# candidate: it removes AMD's triton_kernels so vLLM's vendored pre-rename copy is the only one, and
# it must serve oss120b under an accuracy gate before anything is promoted onto it.
#
# The build context is the REPOSITORY ROOT, because the Dockerfile bakes in two files that live in
# the repo: the tuned fused_moe configs and the eager-PG sitecustomize patch. Nothing is reached
# from outside the image at RUN time, which is the property that matters.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-optarena-vllm-0271:latest}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?SCRATCH must be set on CSCS}/ce-images/optarena-vllm-0271.sqsh}"
# Pinned by DIGEST, not by tag. rocm/pytorch has no 7.2.0-suffixed tag at all -- the 7.2.0 release
# is published unsuffixed as rocm7.2_* -- and an unsuffixed tag is exactly the mutable name the
# consolidation exists to stop trusting. Same base as ce-images/vllm, so the two differ only in
# what they install.
BASE_REPO="docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1"
BASE_DIGEST="sha256:a3b65813621095e3389269417e963725b59310184588c9d2490d44e6e83fa01c"
BASE_IMAGE="${BASE_IMAGE:-${BASE_REPO}@${BASE_DIGEST}}"

mkdir -p "$(dirname "${OUTPUT_SQSH}")"

# Diskless nodes: temp + runtime dirs on /dev/shm, stale per-node podman state wiped (only a cache).
unset DBUS_SESSION_BUS_ADDRESS
export TMPDIR="/dev/shm/${USER}/tmp"
export XDG_RUNTIME_DIR="/dev/shm/${USER}/xdg"
rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" "/dev/shm/${USER}/tmp" "/dev/shm/${USER}/xdg"
mkdir -p "${TMPDIR}"
mkdir -p -m 0700 "${XDG_RUNTIME_DIR}"

cd "${REPO_ROOT}"
# cgroupfs: with the systemd manager a dying logind session reaps podman mid-pull (silent rc=1).
podman --cgroup-manager=cgroupfs build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  .

# Record what was actually built. The tag carries no digest by design, so this plus the in-image
# manifests is the only provenance a later run can be attributed to.
podman image inspect --format '{{.Digest}}' "${IMAGE_TAG}" | tee "${OUTPUT_SQSH}.image-digest"
podman run --rm "${IMAGE_TAG}" cat /opt/BUILD-PINS.txt
podman run --rm "${IMAGE_TAG}" cat /opt/aws-ofi-nccl/BUILD-MANIFEST.txt

# enroot's exit code lies when its cleanup fails after a good write; gate on the artifact
# (listing forces a read of the inode table at file END, which a truncated image fails).
enroot import -x mount -o "${OUTPUT_SQSH}" "podman://${IMAGE_TAG}" || true
unsquashfs -l "${OUTPUT_SQSH}" opt >/dev/null
printf 'Wrote %s\n' "${OUTPUT_SQSH}"
