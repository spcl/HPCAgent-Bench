#!/usr/bin/env bash
# Build hpcagent-bench-judge-agent-amd and export it as a squashfs enroot can mount.
#
# The tag carries NO version suffix. Version identity comes from git plus the image digest this
# script records, not from a name -- a "-v5" in the tag is what made two different images look
# like the same thing in a results table.
#
# Run it from anywhere; it derives the repository root itself and builds with the repo root as
# the context, because the Dockerfile COPYs pyproject.toml (the dependency list) and the harness
# build inputs from containers/agent/harness. Tool scripts are bound at launch.
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
# in the graphroot when the judge build starts. The tmpfs layer store survives between jobs only on
# the node that built it (build_common.sh ce_podman_env), so separate jobs usually pay twice.
BUILD_TARGETS="${BUILD_TARGETS:-agent judge}"
: "${CE_IMAGES:?set SCRATCH or CE_IMAGES}"

# Per-target output name, per partition (ce_amd_candidate): nothing points at a candidate until
# promote_image.sh renames it over a live name.
target_sqsh() {
    local name
    name="$(ce_amd_candidate "$1" "${CE_PARTITION}")" || return 2
    printf '%s/%s' "${CE_IMAGES}" "${name}"
}

IMAGE_TAG="${IMAGE_TAG:-hpcagent-bench-judge-agent-amd:latest}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${CE_IMAGES}/hpcagent-bench-judge-agent-amd.sqsh}"
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

# DaCe: resolve the commit HERE and pass the sha in. The Dockerfile cannot do this -- its layer
# cache keys on the command string, so a '--branch extended' clone is reused forever and the image
# ages into a pin nothing records. Resolving outside makes the sha part of the cache key.
# Default: the release's dace pin (pyproject.toml dace-pin); HPCAGENT_BENCH_DACE_REF=extended bakes
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

# Everything but the base, which ce_cache_base_image may still rewrite to a local copy.
BUILD_ARGS=(
  --build-arg "IMAGE_VERSION=${IMAGE_VERSION}"
  --build-arg "DACE_COMMIT=${DACE_COMMIT}"
  --build-arg "LIBFABRIC_REF=${LIBFABRIC_REF}"
  --build-arg "LIBFABRIC_COMMIT=${LIBFABRIC_COMMIT}"
  --build-arg "ROCM_ARCH=${ROCM_ARCH}"
  "${SPACK_TARGET_ARGS[@]}"
)

# Per-target output path: OUTPUT_SQSH is honoured ONLY for a single-target build -- with two targets
# one name cannot mean both, and silently writing the judge over the agent's path is exactly the kind
# of thing that gets discovered a campaign later. build.sbatch sets OUTPUT_SQSH, so this is common.
target_out() {
    if [[ -n "${OUTPUT_SQSH:-}" && "$(printf '%s\n' ${BUILD_TARGETS} | wc -w)" -eq 1 ]]; then
        printf '%s' "${OUTPUT_SQSH}"
    else
        target_sqsh "$1"
    fi
}

# PULL FIRST: when the registry holds every target built from exactly these inputs, pull instead of
# spending the multi-hour build (build_common.sh ce_pull_first).
SPECS=()
for target in ${BUILD_TARGETS}; do
    SPECS+=("$(ce_amd_role "${target}" "${CE_PARTITION}")|${target}|hpcagent-bench-ce-${target}-amd:latest|$(target_out "${target}")")
done
ce_pull_first "${SCRIPT_DIR}/Dockerfile" "${SPECS[@]}" -- "${BUILD_ARGS[@]}"
[[ "${CE_PULLED}" == 0 ]] || exit 0

ce_mirror_args
ce_require_mirror_commit "spcl/dace.git" "${DACE_COMMIT}"
ce_require_mirror_commit "ofiwg/libfabric.git" "${LIBFABRIC_COMMIT}"

# The 30 GB rocm/pytorch base, read from scratch instead of re-pulled from Docker Hub every
# build. vllm/build.sh already caches this EXACT digest, so this build was fetching bytes
# that were already on disk. It rewrites BASE_IMAGE to a local dir: on a hit, which is why
# ce_build passes BASE_IMAGE_REF -- the base.name label must stay the registry reference.
ce_cache_base_image

# Spack binary buildcache on scratch: gcc 16 and llvm 22 are 60-80 minutes this image has paid
# repeatedly, every time to fail at something after them. The Dockerfile pushes here after each
# install and registers it as a mirror when non-empty; both halves no-op without the mount.
# Wheels survive between jobs in the pip cache, OUTSIDE the image, so a retry does not rebuild cupy
# from its sdist. One pip cache per GPU arch: pip keys a built wheel by its sdist, not by
# HCC_AMDGPU_TARGET, so a shared cache hands one arch's cupy to the other's build.
ce_cache_args spack-buildcache "pip-cache/${ROCM_ARCH}"

# Order matters: `agent` first so the judge build finds those layers already built. Each target
# is exported before the next is built, so a judge failure still leaves a usable agent image.
for target in ${BUILD_TARGETS}; do
    ce_build "${SCRIPT_DIR}/Dockerfile" "${target}" "hpcagent-bench-ce-${target}-amd:latest" "$(target_out "${target}")" \
        "${MIRROR_ARGS[@]}" "${CACHE_ARGS[@]}" "${BUILD_ARGS[@]}"
done
