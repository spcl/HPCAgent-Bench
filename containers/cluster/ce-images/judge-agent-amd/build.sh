#!/usr/bin/env bash
# Build optarena-judge-agent-amd and export it as a squashfs enroot can mount.
#
# The tag carries NO version suffix. Version identity comes from git plus the image digest this
# script records, not from a name -- a "-v5" in the tag is what made two different images look
# like the same thing in a results table.
#
# Run it from anywhere; it derives the repository root itself and builds with the repo root as
# the context, because the Dockerfile COPYs requirements/, containers/agent and containers/judge.
#
#   containers/cluster/ce-images/judge-agent-amd/build.sh
#   OUTPUT_SQSH=$SCRATCH/ce-images/some-candidate.sqsh .../build.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-optarena-judge-agent-amd:latest}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?SCRATCH must be set on CSCS}/ce-images/optarena-judge-agent-amd.sqsh}"
BASE_IMAGE="${BASE_IMAGE:-docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
ROCM_ARCH="${ROCM_ARCH:-gfx942}"

mkdir -p "$(dirname "${OUTPUT_SQSH}")"

# Diskless nodes: temp and runtime dirs on /dev/shm, stale per-node podman state wiped (it is
# only a cache). DBUS_SESSION_BUS_ADDRESS is unset so podman does not try to talk to a session
# bus that is not there.
unset DBUS_SESSION_BUS_ADDRESS
export TMPDIR="/dev/shm/${USER}/tmp"
export XDG_RUNTIME_DIR="/dev/shm/${USER}/xdg"
# The wipe goes through 'podman unshare': an image layer under root/overlay/*/diff is owned by
# a subuid, not by this user, so a plain rm hits Permission denied and leaves a half-deleted
# store the next build dies on. unshare enters the user namespace where those subuids map to
# this user.
podman unshare rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" 2>/dev/null || true
rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" "/dev/shm/${USER}/tmp" "/dev/shm/${USER}/xdg"
mkdir -p "${TMPDIR}"
mkdir -p -m 0700 "${XDG_RUNTIME_DIR}"

# DaCe: resolve the TIP of extended HERE and pass the sha in. The Dockerfile cannot do this --
# its layer cache keys on the command string, so a '--branch extended' clone is reused forever
# and the image ages into a pin nothing records. Resolving outside makes the sha part of the
# cache key, so the layer rebuilds exactly when extended moves and never otherwise.
DACE_COMMIT="${DACE_COMMIT:-$(git ls-remote https://github.com/spcl/dace.git refs/heads/extended | cut -f1)}"
[[ -n "${DACE_COMMIT}" ]] || { echo "could not resolve spcl/dace@extended" >&2; exit 2; }
printf 'dace @ %s\n' "${DACE_COMMIT}"

cd "${REPO_ROOT}"
# The git mirror, when one exists. Every clone in the build is rewritten to it (see the Dockerfile),
# which took GitHub off the critical path: the rate limiter answers an unauthenticated clone with a
# 401 under load, and the callers that died on it -- spack's in-process package-repo clone, vLLM's
# CMake FetchContent of triton -- have no retry. Refresh from a login node with
# containers/cluster/ce-images/mirror-repos.sh. Absent, the build still works, via GitHub.
MIRROR_ARGS=()
GIT_MIRRORS="${GIT_MIRRORS:-${SCRATCH:-}/git-mirrors}"
if [[ -d "${GIT_MIRRORS}" ]]; then
  MIRROR_ARGS=(-v "${GIT_MIRRORS}:/git-mirrors:ro")
  printf 'git mirror %s\n' "${GIT_MIRRORS}"

  # The sha was resolved from GITHUB above, but the clone is rewritten to the MIRROR. A mirror
  # older than the branch tip therefore has a sha it cannot serve, and git says so as
  # `upload-pack: not our ref` -- two hours in, at the dace layer, with every expensive layer
  # already paid for. That is build 626608. Refresh here, where it costs seconds, and fail now
  # rather than then if the sha still is not there.
  dace_mirror="${GIT_MIRRORS}/spcl/dace.git"
  if [[ -d "${dace_mirror}" ]] && ! git -C "${dace_mirror}" cat-file -e "${DACE_COMMIT}" 2>/dev/null; then
    printf 'mirror lacks %s, refreshing ... ' "${DACE_COMMIT:0:12}"
    git -C "${dace_mirror}" remote update --prune >/dev/null 2>&1 && echo OK || echo FAILED
    git -C "${dace_mirror}" cat-file -e "${DACE_COMMIT}" 2>/dev/null || {
      echo "mirror ${dace_mirror} still cannot serve ${DACE_COMMIT}." >&2
      echo "Refresh it with containers/cluster/ce-images/mirror-repos.sh, or unset GIT_MIRRORS" >&2
      echo "to clone from GitHub directly." >&2
      exit 2
    }
  fi
fi

# Spack binary buildcache on scratch: gcc 16 and llvm 22 are 60-80 minutes this image has paid
# repeatedly, every time to fail at something after them. The Dockerfile pushes here after each
# install and registers it as a mirror when non-empty; both halves no-op without the mount.
SPACK_BUILDCACHE="${SPACK_BUILDCACHE:-${SCRATCH:?}/spack-buildcache}"
mkdir -p "${SPACK_BUILDCACHE}"
CACHE_ARGS=(-v "${SPACK_BUILDCACHE}:/spack-buildcache:rw")
printf 'spack buildcache %s\n' "${SPACK_BUILDCACHE}"

# cgroupfs, not systemd: a dying logind session reaps podman mid-pull under the systemd manager,
# with a silent rc=1.
podman --cgroup-manager=cgroupfs build "${MIRROR_ARGS[@]}" "${CACHE_ARGS[@]}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "DACE_COMMIT=${DACE_COMMIT}" \
  --build-arg "ROCM_ARCH=${ROCM_ARCH}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  .

# The digest IS the version. Recorded next to the squashfs so a results table can name the exact
# image a campaign ran on without trusting a mutable tag.
podman image inspect --format '{{.Digest}}' "${IMAGE_TAG}" > "${OUTPUT_SQSH}.digest"
printf 'image digest %s\n' "$(cat "${OUTPUT_SQSH}.digest")"

# enroot's exit code lies when cleanup fails after a good write, so gate on the ARTIFACT: listing
# reads the inode table at file END, which a truncated image fails. Remove the output first --
# enroot refuses to overwrite, `|| true` swallows that, and `unsquashfs -l` would then validate
# LAST run's file (620068 printed "IMAGE READY" over a stale image built with the wrong triton).
rm -f "${OUTPUT_SQSH}"
enroot import -x mount -o "${OUTPUT_SQSH}" "podman://${IMAGE_TAG}" || true
unsquashfs -l "${OUTPUT_SQSH}" opt >/dev/null
printf 'Wrote %s\n' "${OUTPUT_SQSH}"

# Optional registry push, so this image can be PULLED instead of rebuilt.
#
# It has to happen here and not in a later job: podman's graphroot is /dev/shm/$USER/root, which
# is node-local tmpfs wiped at the top of this script and gone when the job ends. Once that store
# disappears the only artifact left is the squashfs, and a squashfs is a flattened filesystem
# rather than an OCI image -- reimporting one loses the layers and the config. So an image that
# was not pushed while it was built has to be rebuilt to be pushed.
#
# AFTER the squashfs is written and verified, so a registry outage or a rejected layer costs the
# upload and never the local artifact this job exists to produce.
if [[ -n "${PUSH_REPO:-}" ]]; then
  "$(dirname -- "${SCRIPT_DIR}")/push_image.sh" "${IMAGE_TAG}" ${PUSH_TAGS:-} \
    || echo "push to ${PUSH_REPO} FAILED; the local image and squashfs are unaffected" >&2
fi
