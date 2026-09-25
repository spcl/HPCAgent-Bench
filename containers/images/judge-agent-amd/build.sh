#!/usr/bin/env bash
# Build hpcagent-bench-judge-agent-amd and export it as a squashfs enroot can mount.
#
# The tag carries NO version suffix. Version identity comes from git plus the image digest this
# script records, not from a name -- a "-v5" in the tag is what made two different images look
# like the same thing in a results table.
#
# Run it from anywhere; it derives the repository root itself and builds with the repo root as
# the context, because the Dockerfile COPYs requirements/, the harness build inputs from
# containers/agent/harness and containers/judge/requirements.txt. Tool scripts are bound at launch.
#
#   containers/images/judge-agent-amd/build.sh
#   OUTPUT_SQSH=$SCRATCH/ce-images/some-candidate.sqsh .../build.sh
set -euo pipefail

# The cluster's core_pattern dumps into the crashing process's CWD; Slurm propagates the
# submitter's core limit, so the floor has to be set here to avoid littering the checkout.
ulimit -c 0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=../build_common.sh
source "${SCRIPT_DIR}/../build_common.sh"

# TWO TARGETS, ONE BUILD. `judge` is `FROM agent` plus the hpcagent_bench install, so building
# both in this one invocation costs the agent build plus a pip layer -- the agent stage is already
# in the graphroot when the judge build starts. Building them in SEPARATE JOBS would cost two full
# builds: build_common.sh wipes the /dev/shm graphroot on entry (the nodes are diskless), so there
# is no layer cache between jobs and nothing of the agent survives to be reused.
BUILD_TARGETS="${BUILD_TARGETS:-agent judge}"
CE_DIR="${CE_DIR:-${SCRATCH:?SCRATCH must be set on CSCS}/ce-images}"

# Per-target output name, per partition (ce_amd_candidate): nothing points at a candidate until
# promote_image.sh renames it over a live name.
target_sqsh() {
    local name
    name="$(ce_amd_candidate "$1" "${CE_PARTITION}")" || return 2
    printf '%s/%s' "${CE_DIR}" "${name}"
}

IMAGE_TAG="${IMAGE_TAG:-hpcagent-bench-judge-agent-amd:latest}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH:?SCRATCH must be set on CSCS}/ce-images/hpcagent-bench-judge-agent-amd.sqsh}"
# Pinned by DIGEST, matching the Dockerfile's ARG default. Passing the bare tag here would
# OVERRIDE that default and quietly unpin the build, and the base.name label would then
# record a mutable reference. Same shape as the three inference builders.
BASE_REPO="docker.io/rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1"
BASE_DIGEST="sha256:a3b65813621095e3389269417e963725b59310184588c9d2490d44e6e83fa01c"
BASE_IMAGE="${BASE_IMAGE:-${BASE_REPO}@${BASE_DIGEST}}"
# ROCM_ARCH from gpu_arch.env for this job's partition; an unknown partition stops before any pull.
ce_gpu_arch
# The spack CPU target from cpu_target.env; passed only when the partition pins one, so an mi300
# build gets exactly the build args it always had.
ce_spack_target
SPACK_TARGET_ARGS=()
[[ -z "${SPACK_TARGET}" ]] || SPACK_TARGET_ARGS=(--build-arg "SPACK_TARGET=${SPACK_TARGET}")

# The version the LABEL records. Taken from the output name -- ...-v7.sqsh is v7 --
# so the label and the artifact cannot disagree.
IMAGE_VERSION="${IMAGE_VERSION:-$(basename "${OUTPUT_SQSH}" .sqsh | sed 's/.*-//')}"
mkdir -p "$(dirname "${OUTPUT_SQSH}")"

ce_podman_env

# The 30 GB rocm/pytorch base, read from scratch instead of re-pulled from Docker Hub every
# build. vllm/build.sh already caches this EXACT digest, so this build was fetching bytes
# that were already on disk. It rewrites BASE_IMAGE to a local dir: on a hit, which is why
# BASE_IMAGE_REF is passed below -- the base.name label must stay the registry reference.
ce_cache_base_image

# Wheels survive between jobs here, OUTSIDE the image, so a retry does not rebuild cupy
# from its sdist. The bind mount means nothing lands in an image layer either way. One cache per
# GPU arch: pip keys a built wheel by its sdist, not by HCC_AMDGPU_TARGET, so a shared cache hands
# one arch's cupy to the other's build.
PIP_CACHE="${PIP_CACHE:-${SCRATCH:?}/pip-cache/${ROCM_ARCH}}"
mkdir -p "${PIP_CACHE}"

# DaCe: resolve the commit HERE and pass the sha in. The Dockerfile cannot do this -- its layer
# cache keys on the command string, so a '--branch extended' clone is reused forever and the image
# ages into a pin nothing records. Resolving outside makes the sha part of the cache key.
# Default: the release's dace pin (scripts/dace_pin.env); HPCAGENT_BENCH_DACE_REF=extended bakes
# the tip. Jobs move the baked dace to the latest extended at start (dace_refresh.sh).
DACE_COMMIT="$(HPCAGENT_BENCH_DACE_REF="${HPCAGENT_BENCH_DACE_REF:-pinned}" \
    "${SCRIPT_DIR}/../dace_refresh.sh" --resolve)"
[[ "${DACE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "could not resolve spcl/dace@${DACE_COMMIT}" >&2; exit 2; }
printf 'dace @ %s\n' "${DACE_COMMIT}"

# libfabric, pinned the SAME way as dace: the tag is resolved to a SHA here and the sha is what the
# Dockerfile asserts. A `--branch v2.3.1` clone alone is not a pin -- a git tag is mutable, and
# podman's layer cache keys on the command string, so the layer would be reused forever and the
# image would age into a pin nothing recorded. Resolving outside makes the sha part of the cache
# key: the layer rebuilds exactly when the tag moves, and the build FAILS rather than silently
# taking different source.
#
# This is a COMPILE-TIME link target for spack's MPICH and nothing else -- it is deleted from the
# shipped image so MPI resolves at run time to the libfabric the netstack hook supplies (the pinned
# artifact bundle in the EDF templates, /opt/cray/libfabric/host in host mode). No RCCL net plugin
# is built here either; the hook supplies one, matched to the host driver.
LIBFABRIC_REF="${LIBFABRIC_REF:-v2.6.0}"
resolve_tag() {
    # ^{} dereferences an annotated tag to the commit it points at; without it a tag object's own
    # sha comes back and never matches `git rev-parse HEAD` in a checkout.
    local url="$1" ref="$2" sha
    sha="$(git ls-remote "${url}" "refs/tags/${ref}^{}" | cut -f1)"
    [[ -n "${sha}" ]] || sha="$(git ls-remote "${url}" "refs/tags/${ref}" | cut -f1)"
    [[ -n "${sha}" ]] || { echo "could not resolve ${ref} in ${url}" >&2; return 2; }
    printf '%s' "${sha}"
}
LIBFABRIC_COMMIT="${LIBFABRIC_COMMIT:-$(resolve_tag https://github.com/ofiwg/libfabric.git "${LIBFABRIC_REF}")}"
printf 'libfabric     %s @ %s\n' "${LIBFABRIC_REF}" "${LIBFABRIC_COMMIT}"

cd "${REPO_ROOT}"
ce_mirror_args
ce_require_mirror_commit "spcl/dace.git" "${DACE_COMMIT}"
ce_require_mirror_commit "ofiwg/libfabric.git" "${LIBFABRIC_COMMIT}"

# Spack binary buildcache on scratch: gcc 16 and llvm 22 are 60-80 minutes this image has paid
# repeatedly, every time to fail at something after them. The Dockerfile pushes here after each
# install and registers it as a mirror when non-empty; both halves no-op without the mount.
SPACK_BUILDCACHE="${SPACK_BUILDCACHE:-${SCRATCH:?}/spack-buildcache}"
mkdir -p "${SPACK_BUILDCACHE}"
CACHE_ARGS=(-v "${SPACK_BUILDCACHE}:/spack-buildcache:rw" -v "${PIP_CACHE}:/pip-cache:rw")
printf 'spack buildcache %s\n' "${SPACK_BUILDCACHE}"

# cgroupfs, not systemd: a dying logind session reaps podman mid-pull under the systemd manager,
# with a silent rc=1.
# Order matters: `agent` first so the judge build finds those layers already built. Each target
# is exported before the next is built, so a judge failure still leaves a usable agent image.
for target in ${BUILD_TARGETS}; do
    tag="hpcagent-bench-ce-${target}-amd:latest"
    # Honour an explicit OUTPUT_SQSH, but ONLY for a single-target build -- with two targets one
    # name cannot mean both, and silently writing the judge over the agent's path is exactly the
    # kind of thing that gets discovered a campaign later. build.sbatch sets OUTPUT_SQSH, so this
    # is the common case rather than a corner.
    if [[ -n "${OUTPUT_SQSH:-}" && "$(printf '%s\n' ${BUILD_TARGETS} | wc -w)" -eq 1 ]]; then
        out="${OUTPUT_SQSH}"
    else
        out="$(target_sqsh "${target}")"
    fi
    printf '\n===== building target %s -> %s =====\n' "${target}" "${out}"
    podman --cgroup-manager=cgroupfs build "${MIRROR_ARGS[@]}" "${CACHE_ARGS[@]}" \
      --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
      --build-arg "BASE_IMAGE_REF=${BASE_IMAGE_REF:-${BASE_IMAGE}}" \
      --build-arg "IMAGE_VERSION=${IMAGE_VERSION}" \
      --build-arg "DACE_COMMIT=${DACE_COMMIT}" \
      --build-arg "LIBFABRIC_REF=${LIBFABRIC_REF}" \
      --build-arg "LIBFABRIC_COMMIT=${LIBFABRIC_COMMIT}" \
      --build-arg "ROCM_ARCH=${ROCM_ARCH}" \
      "${SPACK_TARGET_ARGS[@]}" \
      --target "${target}" \
      -f "${SCRIPT_DIR}/Dockerfile" \
      -t "${tag}" \
      .
    ce_export_image "${tag}" "${out}"
done
