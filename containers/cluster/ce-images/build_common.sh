#!/usr/bin/env bash
# Shared by every <image>/build.sh and build.sbatch. Source it:
#   source "$(dirname -- "${BASH_SOURCE[0]}")/../build_common.sh"

# No core dumps: they land in the CWD on an inode-quota filesystem.
ulimit -c 0
CE_IMAGES_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Diskless nodes: podman temp/runtime dirs on /dev/shm, stale per-node store wiped (only a cache).
ce_podman_env() {
    unset DBUS_SESSION_BUS_ADDRESS
    export TMPDIR="/dev/shm/${USER}/tmp"
    export XDG_RUNTIME_DIR="/dev/shm/${USER}/xdg"
    # Layers are owned by subuids, so the wipe runs inside `podman unshare`.
    podman unshare rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" 2>/dev/null || true
    rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" "/dev/shm/${USER}/tmp" \
           "/dev/shm/${USER}/xdg"
    mkdir -p "${TMPDIR}"
    mkdir -p -m 0700 "${XDG_RUNTIME_DIR}"
    # An account without its own podman storage config (a fresh Daint login) gets the same tmpfs
    # store the wipe above assumes.
    if [[ -z "${CONTAINERS_STORAGE_CONF:-}" && ! -f "${HOME}/.config/containers/storage.conf" ]]; then
        printf '[storage]\ndriver = "overlay"\nrunroot = "/dev/shm/%s/runroot"\ngraphroot = "/dev/shm/%s/root"\n' \
            "${USER}" "${USER}" > "/dev/shm/${USER}/storage.conf"
        export CONTAINERS_STORAGE_CONF="/dev/shm/${USER}/storage.conf"
    fi
}

# A private podman store for a node another podman may use (a login node), inherited by every
# podman call through CONTAINERS_STORAGE_CONF. $1 must be tmpfs: overlay and RUN-step mountpoints
# fail on Lustre. Use "$1/tmp" as TMPDIR.
ce_private_podman_store() {
    local store="$1"
    mkdir -p "${store}/root" "${store}/runroot" "${store}/tmp"
    printf '[storage]\ndriver = "overlay"\ngraphroot = "%s/root"\nrunroot = "%s/runroot"\n' \
        "${store}" "${store}" > "${store}/storage.conf"
    export CONTAINERS_STORAGE_CONF="${store}/storage.conf"
}

# Removes a ce_private_podman_store: containers and images first, then the tree (subuid-owned).
ce_remove_podman_store() {
    local store="$1"
    [[ -f "${store}/storage.conf" ]] || return 0
    CONTAINERS_STORAGE_CONF="${store}/storage.conf" podman rm -a -f >/dev/null 2>&1 || true
    CONTAINERS_STORAGE_CONF="${store}/storage.conf" podman rmi -a -f >/dev/null 2>&1 || true
    podman unshare rm -rf "${store}" 2>/dev/null || true
    rm -rf "${store}" 2>/dev/null || true
    [[ ! -e "${store}" ]] || echo "warning: ${store} is still there; remove it with podman unshare rm -rf" >&2
}

# Sets MIRROR_ARGS: mounts $GIT_MIRRORS (mirror-repos.sh) so every clone in the build avoids
# GitHub's rate limiter. Without a mirror the build clones from GitHub.
ce_mirror_args() {
    MIRROR_ARGS=()
    GIT_MIRRORS="${GIT_MIRRORS:-${SCRATCH:-}/git-mirrors}"
    if [[ -d "${GIT_MIRRORS}" ]]; then
        MIRROR_ARGS=(-v "${GIT_MIRRORS}:/git-mirrors:ro")
        printf 'git mirror %s\n' "${GIT_MIRRORS}"
    fi
}

# Fails fast when the mirror lacks a commit resolved from GitHub (the clone would fail hours in).
ce_require_mirror_commit() {
    local repo_path="$1" commit="$2"
    local mirror="${GIT_MIRRORS:-}/${repo_path}"
    [[ -n "${GIT_MIRRORS:-}" && -d "${mirror}" ]] || return 0
    git -C "${mirror}" cat-file -e "${commit}" 2>/dev/null && return 0
    printf 'mirror lacks %s, refreshing ... ' "${commit:0:12}"
    git -C "${mirror}" remote update --prune >/dev/null 2>&1 && echo OK || echo FAILED
    git -C "${mirror}" cat-file -e "${commit}" 2>/dev/null || {
        echo "mirror ${mirror} still cannot serve ${commit}." >&2
        echo "Refresh it with ce-images/mirror-repos.sh, or unset GIT_MIRRORS to use GitHub." >&2
        return 2
    }
}

# Sets GPU_ARGS: hands /dev/kfd and /dev/dri to the build when present, because aiter and vLLM
# probe the device at import time.
ce_gpu_args() {
    GPU_ARGS=()
    if [[ -e /dev/kfd ]]; then
        GPU_ARGS=(--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined
                  --group-add keep-groups)
        printf 'gpu devices handed to the build\n'
    fi
}

# Prints the AMD GPU arch gpu_arch.env names for partition $1; an unknown partition is refused.
ce_partition_arch() {
    local arch
    arch="$(sed -n "s/^GPU_ARCH_${1:-}=//p" "${CE_IMAGES_DIR}/gpu_arch.env")"
    if [[ ! "${arch}" =~ ^gfx[0-9a-f]+$ ]]; then
        echo "gpu_arch.env names no GPU arch for partition '${1:-}'" >&2
        return 2
    fi
    printf '%s\n' "${arch}"
}

# Exports ROCM_ARCH and CE_PARTITION for this job's partition (ROCM_PARTITION outside Slurm).
ce_gpu_arch() {
    local partition="${SLURM_JOB_PARTITION:-${ROCM_PARTITION:-}}" arch
    if [[ -z "${partition}" ]]; then
        echo "ce_gpu_arch: no SLURM_JOB_PARTITION; set ROCM_PARTITION for a dry run outside Slurm" >&2
        return 2
    fi
    if [[ -n "${ROCM_PARTITION:-}" && "${ROCM_PARTITION}" != "${partition}" ]]; then
        echo "ce_gpu_arch: ROCM_PARTITION=${ROCM_PARTITION} but this job runs on ${partition}" >&2
        return 2
    fi
    arch="$(ce_partition_arch "${partition}")" || return 2
    if [[ -n "${ROCM_ARCH:-}" && "${ROCM_ARCH}" != "${arch}" ]]; then
        echo "ce_gpu_arch: ROCM_ARCH=${ROCM_ARCH} disagrees with gpu_arch.env: ${partition} is ${arch}" >&2
        return 2
    fi
    export ROCM_ARCH="${arch}" CE_PARTITION="${partition}"
    printf 'gpu arch %s for partition %s\n' "${ROCM_ARCH}" "${partition}"
}

# Exports SPACK_TARGET for CE_PARTITION (set by ce_gpu_arch) from cpu_target.env; empty for a
# partition with no row, whose build keeps spack's host detection.
ce_spack_target() {
    SPACK_TARGET="$(sed -n "s/^SPACK_TARGET_${CE_PARTITION:?run ce_gpu_arch first}=//p" "${CE_IMAGES_DIR}/cpu_target.env")"
    if [[ -n "${SPACK_TARGET}" && ! "${SPACK_TARGET}" =~ ^[a-z0-9_]+$ ]]; then
        echo "cpu_target.env: '${SPACK_TARGET}' is not a spack target for partition ${CE_PARTITION}" >&2
        return 2
    fi
    export SPACK_TARGET
    printf 'spack target %s for partition %s\n' "${SPACK_TARGET:-<host>}" "${CE_PARTITION}"
}

# ce_amd_candidate <agent|judge> <partition>: the candidate a judge-agent-amd build target writes,
# read from the images.env row of that target and partition.
ce_amd_candidate() {
    local profile name
    case "$1" in
        agent) profile=judge-agent-amd ;;
        judge) profile=judge ;;
        *) echo "unknown build target '$1'" >&2; return 2 ;;
    esac
    [[ -n "${CE_IMAGE_TABLE:-}" ]] || source "${CE_IMAGES_DIR}/images.env"
    name="$(awk -v p="${profile}" -v part="${2:?partition}" \
        '$4 == "judge-agent-amd" && $5 == part && $6 == p {print $7}' <<<"${CE_IMAGE_TABLE}")"
    [[ -n "${name}" ]] || { echo "images.env has no judge-agent-amd ${1} row for partition ${2}" >&2; return 2; }
    printf '%s' "${name}"
}

# Caches the base image on scratch as a `dir:` tree (plain files; the podman layer store cannot live
# on scratch, which rejects user xattrs) and rewrites BASE_IMAGE to it on a hit. A miss pulls and
# copies out through a staging dir, so racing builds never leave a half-written cache entry.
ce_cache_base_image() {
    BASE_CACHE="${BASE_CACHE:-${SCRATCH:?}/base-images}"
    # The registry reference for the image label, kept before BASE_IMAGE becomes a local path.
    BASE_IMAGE_REF="${BASE_IMAGE_REF:-${BASE_IMAGE}}"
    local base_dir staging
    base_dir="${BASE_CACHE}/$(printf '%s' "${BASE_IMAGE}" | tr '/:@' '___')"
    if [[ -f "${base_dir}/manifest.json" ]]; then
        printf 'base image from cache %s\n' "${base_dir}"
        BASE_IMAGE="dir:${base_dir}"
    elif podman pull -q "${BASE_IMAGE}" >/dev/null; then
        staging="${base_dir}.staging.$$"
        mkdir -p "${BASE_CACHE}"
        rm -rf "${staging}"
        if podman push -q "${BASE_IMAGE}" "dir:${staging}"; then
            if [[ -f "${base_dir}/manifest.json" ]]; then
                rm -rf "${staging}"
                printf 'base image was cached by a concurrent build\n'
            elif mv -T "${staging}" "${base_dir}"; then
                printf 'base image cached to %s\n' "${base_dir}"
            else
                rm -rf "${staging}"
                printf 'base image could not be published; this build is unaffected\n'
            fi
        else
            rm -rf "${staging}"
            printf 'base image could not be cached; this build is unaffected\n'
        fi
    fi
}

# Verifies a candidate inside itself, under an EDF rendered from its production template, and
# writes the .verified marker promote_image.sh requires only on a clean verdict. Used by the GH200
# and CPU build.sbatch; AMD images use build_and_verify.sbatch. Probes run from `/` so a `dace`
# directory in the CWD cannot shadow the image's own.
#
#   ce_verify_candidate <template under ce-images/> <sqsh> <verify_image.py profile> [srun args...]
ce_verify_candidate() {
    local template="$1" sqsh="$2" profile="$3" repo edf modules rc=0
    shift 3
    repo="$(cd -- "${CE_IMAGES_DIR}/../../.." && pwd)"
    # shellcheck source=../../../scripts/cache_env.sh
    . "${repo}/scripts/cache_env.sh"
    edf="${SCRATCH:?}/.tmp/verify-${SLURM_JOB_ID:-$$}-${profile}.toml"
    mkdir -p "$(dirname "${edf}")"
    sed -e "s|\${SCRATCH}|${SCRATCH}|g" \
        -e "s|\"<hpcagent_bench_edf_mounts>\"|$(hpcagent_bench_edf_mounts), \"${repo}/containers/agent:/opt/hpcagent-bench-agent\"|" \
        -e "s|^image = .*|image = \"${sqsh}\"|" \
        -e "s|^workdir = .*|workdir = \"/\"|" \
        "${CE_IMAGES_DIR}/${template}" > "${edf}"
    rm -f "${sqsh}.verified"
    printf '\n===== verifying %s as profile=%s =====\n' "${sqsh}" "${profile}"
    srun "$@" --environment="${edf}" python3 "${CE_IMAGES_DIR}/verify_image.py" --profile "${profile}" \
        --verbose || rc=$((rc + 1))
    case "${profile}" in
        vllm-*) modules="numpy,torch,vllm,triton" ;;
        *)      modules="" ;;
    esac
    srun "$@" --environment="${edf}" python3 "${CE_IMAGES_DIR}/selfcontained_check.py" \
        ${modules:+--modules "${modules}"} || rc=$((rc + 1))
    case "${profile}" in
        judge*)
            srun "$@" --environment="${edf}" python3 "${CE_IMAGES_DIR}/tools_launch_check.py" \
                --agent-dir /opt/hpcagent-bench-agent --judge-tools "${repo}/containers/judge/tools" \
                || rc=$((rc + 1))
            ;;
    esac
    rm -f "${edf}"
    if [[ "${rc}" -eq 0 ]]; then
        printf 'verified profile=%s job=%s digest=%s\n' "${profile}" "${SLURM_JOB_ID:-none}" \
            "$(cat "${sqsh}.digest" 2>/dev/null || echo unknown)" > "${sqsh}.verified"
        printf 'VERIFIED: %s\n' "${sqsh}"
    else
        printf 'NOT VERIFIED (%s failed stage(s)): %s\n' "${rc}" "${sqsh}" >&2
    fi
    return "${rc}"
}

# After a successful `podman build`: digest sidecar, squashfs, OCI archive, optional push.
ce_export_image() {
    local image_tag="$1" output_sqsh="$2"

    # The digest is the build's identity; results tables cite it, never a tag.
    podman image inspect --format '{{.Digest}}' "${image_tag}" > "${output_sqsh}.digest"
    printf 'image digest %s\n' "$(cat "${output_sqsh}.digest")"

    # enroot's exit code is unreliable, so gate on the artifact: listing reads the inode table at
    # the end of the file, which a truncated image fails. Remove first: enroot will not overwrite.
    rm -f "${output_sqsh}"
    enroot import -x mount -o "${output_sqsh}" "podman://${image_tag}" || true
    unsquashfs -l "${output_sqsh}" opt >/dev/null
    printf 'Wrote %s\n' "${output_sqsh}"

    # The OCI archive keeps layers and config (a squashfs is flattened), so push_image.sh
    # --from-archive can publish later without a rebuild. SAVE_OCI_ARCHIVE=0 skips it.
    if [[ "${SAVE_OCI_ARCHIVE:-1}" != "0" ]]; then
        local archive="${output_sqsh%.sqsh}.oci.tar"
        rm -f "${archive}"
        if podman save --format oci-archive -o "${archive}" "${image_tag}"; then
            printf 'Wrote %s\n' "${archive}"
        else
            echo "OCI archive save FAILED: the squashfs is fine, but publishing this build later" >&2
            echo "would need a full rebuild. See ce-images/push_image.sh." >&2
        fi
    fi

    # Optional push, after the artifacts exist, so a registry failure never costs the build.
    if [[ -n "${PUSH_REPO:-}" ]]; then
        "${CE_IMAGES_DIR}/push_image.sh" "${image_tag}" ${PUSH_TAGS:-} \
          || echo "push to ${PUSH_REPO} FAILED; the local image and squashfs are unaffected" >&2
    fi
}
