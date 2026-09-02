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
# The wipe goes through `podman unshare`: an image layer under root/overlay/*/diff is owned by
# a SUBUID, so a plain rm cannot touch it and leaves a half-deleted store the next build
# dies on. unshare enters the user namespace where those subuids map to this user.
podman unshare rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" 2>/dev/null || true
rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" "/dev/shm/${USER}/tmp" "/dev/shm/${USER}/xdg"
mkdir -p "${TMPDIR}"
mkdir -p -m 0700 "${XDG_RUNTIME_DIR}"

cd "${REPO_ROOT}"
# cgroupfs: with the systemd manager a dying logind session reaps podman mid-pull (silent rc=1).
# The git mirror, when one exists. Every clone in the build is rewritten to it (see the Dockerfile),
# which is what finally took GitHub off the critical path: the rate limiter answers an
# unauthenticated clone with a 401 under load, and the callers that died on it -- spack's in-process
# package-repo clone, vLLM's CMake FetchContent of triton -- have no retry to give them. Refresh it
# from a login node with containers/cluster/ce-images/mirror-repos.sh. Absent, the build still works and still talks to GitHub.
MIRROR_ARGS=()
GIT_MIRRORS="${GIT_MIRRORS:-${SCRATCH:-}/git-mirrors}"
if [[ -d "${GIT_MIRRORS}" ]]; then
  MIRROR_ARGS=(-v "${GIT_MIRRORS}:/git-mirrors:ro")
  printf 'git mirror %s\n' "${GIT_MIRRORS}"
fi

# The GPU, handed to the build. aiter >= 0.1.19 reads the arch from `rocminfo` at IMPORT time and
# ignores GPU_ARCHS on purpose (get_gfx_runtime's docstring says so), and vLLM's rocm.py probes the
# device too -- so a device-less build cannot even import them, and the earlier note here that "a
# podman build cannot use GPUs" was wrong. Measured in job 619976: an mi300 job with NO --gres
# still exposes /dev/kfd (crw-rw-rw- root:render), and `podman build --device` reports gfx942
# inside a RUN step. Conditional, so a build on a node without the device fails in the image step
# that actually needs it rather than on an unusable --device flag.
GPU_ARGS=()
if [[ -e /dev/kfd ]]; then
  GPU_ARGS=(--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined
            --group-add keep-groups)
  printf 'gpu devices handed to the build\n'
fi

# Base image cache on scratch. The podman LAYER store cannot live there: capstor, iopsstor and the
# NFS home all reject user xattrs, so `overlay` and `fuse-overlayfs` fail on lsetxattr and `vfs`
# fails creating its pivot dir under a subuid (all three measured). The base image can, because a
# `dir:` tree is plain files. That is the part worth caching -- a 30-52 GB pull from a registry per
# job, on a store that is wiped every time because the nodes are diskless and it lives in RAM.
#
# Miss: pull over the network as before, then copy out for next time; the build still reads the
# copy already in the store, so this costs one write and never a second pull. Hit: read from
# scratch. Staged through a temp dir and renamed, so two builds racing cannot leave a half-written
# tree that later jobs would treat as a cache hit.
BASE_CACHE="${BASE_CACHE:-${SCRATCH:?}/base-images}"
base_dir="${BASE_CACHE}/$(printf '%s' "${BASE_IMAGE}" | tr '/:@' '___')"
if [[ -f "${base_dir}/manifest.json" ]]; then
  printf 'base image from cache %s\n' "${base_dir}"
  BASE_IMAGE="dir:${base_dir}"
elif podman pull -q "${BASE_IMAGE}" >/dev/null; then
  staging="${base_dir}.staging.$$"
  mkdir -p "${BASE_CACHE}"
  rm -rf "${staging}"
  if podman push -q "${BASE_IMAGE}" "dir:${staging}"; then
    rm -rf "${base_dir}" && mv "${staging}" "${base_dir}" \
      && printf 'base image cached to %s\n' "${base_dir}"
  else
    rm -rf "${staging}"
    printf 'base image could not be cached; this build is unaffected\n'
  fi
fi

podman --cgroup-manager=cgroupfs build "${MIRROR_ARGS[@]}" "${GPU_ARGS[@]}" \
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
