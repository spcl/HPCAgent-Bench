#!/usr/bin/env bash
# Shared by every <image>/build.sh and build.sbatch. Source it:
#   source "$(dirname -- "${BASH_SOURCE[0]}")/../build_common.sh"
#
# Build defaults (IMAGE_REQUIREMENTS.md "Build defaults"), each an environment knob:
#   CE_BUILD_CACHE=1  keep the node's podman layer store and mount the spack buildcache and pip
#                     cache, so a failed build resumes from its last good layer; 0 builds cold.
#   CE_PULL=1         before building, pull the registry image whose build-inputs label matches
#                     this checkout (ce_pull_wanted); only = pull the tag or fail; 0 = always build.
#   PULL_REPO         where ce_pull_wanted looks: PUSH_REPO when set, else images.env REGISTRY_REPO.

# No core dumps: they land in the CWD on an inode-quota filesystem.
ulimit -c 0
CE_IMAGES_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=images.env
[[ -n "${CE_IMAGE_TABLE:-}" ]] || source "${CE_IMAGES_DIR}/images.env"
CE_BUILD_CACHE="${CE_BUILD_CACHE:-1}"
CE_PULL="${CE_PULL:-1}"
PULL_REPO="${PULL_REPO:-${PUSH_REPO:-${REGISTRY_REPO}}}"
# The image label that records ce_build_fingerprint; ce_pull_wanted compares against it.
CE_INPUTS_LABEL="org.hpcagent-bench.build-inputs"

# Diskless nodes: podman temp/runtime dirs on tmpfs (CE_TMPFS). The layer store under it survives
# between jobs on the same node (CE_BUILD_CACHE=1); CE_BUILD_CACHE=0, or a store podman cannot
# read, wipes it. Sets CE_BUILD_FLAGS for `podman build`.
ce_podman_env() {
    unset DBUS_SESSION_BUS_ADDRESS
    export TMPDIR="${CE_TMPFS}/tmp"
    export XDG_RUNTIME_DIR="${CE_TMPFS}/xdg"
    # Runtime state never outlives its job; layers are owned by subuids, so rm runs in `podman unshare`.
    podman unshare rm -rf "${CE_TMPFS}/runroot" 2>/dev/null || true
    rm -rf "${CE_TMPFS}/runroot" "${CE_TMPFS}/tmp" "${CE_TMPFS}/xdg"
    mkdir -p "${TMPDIR}"
    mkdir -p -m 0700 "${XDG_RUNTIME_DIR}"
    # An account without its own podman storage config (a fresh Daint login) gets the same tmpfs
    # store the wipe below assumes.
    if [[ -z "${CONTAINERS_STORAGE_CONF:-}" && ! -f "${HOME}/.config/containers/storage.conf" ]]; then
        printf '[storage]\ndriver = "overlay"\nrunroot = "%s/runroot"\ngraphroot = "%s/root"\n' \
            "${CE_TMPFS}" "${CE_TMPFS}" > "${CE_TMPFS}/storage.conf"
        export CONTAINERS_STORAGE_CONF="${CE_TMPFS}/storage.conf"
    fi
    CE_BUILD_FLAGS=(--layers=true)
    if [[ "${CE_BUILD_CACHE}" == 1 ]] && podman images >/dev/null 2>&1; then
        printf 'podman layer cache %s/root on %s (%s); resubmit with --nodelist=%s to resume\n' \
            "${CE_TMPFS}" "$(hostname)" "$(du -sh "${CE_TMPFS}/root" 2>/dev/null | cut -f1)" "$(hostname)"
        return 0
    fi
    podman unshare rm -rf "${CE_TMPFS}/root" 2>/dev/null || true
    rm -rf "${CE_TMPFS}/root"
    [[ "${CE_BUILD_CACHE}" == 1 ]] || CE_BUILD_FLAGS=(--no-cache)
}

# Sets CACHE_ARGS: the spack binary buildcache and pip wheel cache as build mounts. The Dockerfiles
# use each when mounted and skip it otherwise, so CE_BUILD_CACHE=0 leaves CACHE_ARGS empty.
#   ce_cache_args <spack cache name under SCRATCH> <pip cache name under SCRATCH>
ce_cache_args() {
    CACHE_ARGS=()
    if [[ "${CE_BUILD_CACHE}" != 1 ]]; then
        printf 'spack buildcache and pip cache OFF (CE_BUILD_CACHE=0)\n'
        return 0
    fi
    SPACK_BUILDCACHE="${SPACK_BUILDCACHE:-${SCRATCH:?}/$1}"
    PIP_CACHE="${PIP_CACHE:-${SCRATCH:?}/$2}"
    mkdir -p "${SPACK_BUILDCACHE}" "${PIP_CACHE}"
    CACHE_ARGS=(-v "${SPACK_BUILDCACHE}:/spack-buildcache:rw" -v "${PIP_CACHE}:/pip-cache:rw")
    printf 'spack buildcache %s\npip cache %s\n' "${SPACK_BUILDCACHE}" "${PIP_CACHE}"
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
        echo "Refresh it with containers/images/mirror-repos.sh, or unset GIT_MIRRORS to use GitHub." >&2
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

# ce_amd_role <agent|judge> <partition>: the images.env role a judge-agent-amd build target writes
# on that partition; ce_amd_candidate prints that role's candidate squashfs name.
ce_amd_role() {
    local profile role
    case "$1" in
        agent) profile=judge-agent-amd ;;
        judge) profile=judge ;;
        *) echo "unknown build target '$1'" >&2; return 2 ;;
    esac
    role="$(awk -v p="${profile}" -v part="${2:?partition}" \
        '$4 == "judge-agent-amd" && $5 == part && $6 == p {print $1}' <<<"${CE_IMAGE_TABLE}")"
    [[ -n "${role}" ]] || { echo "images.env has no judge-agent-amd ${1} row for partition ${2}" >&2; return 2; }
    printf '%s' "${role}"
}
ce_amd_candidate() {
    local role
    role="$(ce_amd_role "$@")" || return 2
    printf '%s' "$(ce_image "${role}" candidate)"
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

# The build's identity before it runs: sha256 over the Dockerfile through the target's stage, the
# build args, the base image reference and the git blobs of every path those stages COPY from the
# context. A path COPY reads that differs from git makes the inputs unknown (exit 1): no pull.
#   ce_build_fingerprint <dockerfile> <target|""> <podman build args>   (only --build-arg values count)
ce_build_fingerprint() {
    local dockerfile="$1" target="$2" context prefix
    shift 2
    context="$(cd -- "${CE_IMAGES_DIR}/../.." && pwd)"
    # Instructions are matched uppercase at the line start: heredoc Python has `from` lines too.
    prefix="$(awk -v t="${target}" '/^FROM[[:space:]]/ && done {exit}
        {print}
        /^FROM[[:space:]]/ && t != "" && toupper($(NF-1)) == "AS" && $NF == t {done = 1}' "${dockerfile}")"
    local -a srcs
    mapfile -t srcs < <(awk '/^COPY[[:space:]]/ && $0 !~ /--from=/ {
        for (i = 2; i < NF; i++) if ($i !~ /^--/) print $i }' <<<"${prefix}")
    if (( ${#srcs[@]} )) && ! git -C "${context}" diff --quiet HEAD -- "${srcs[@]}" 2>/dev/null; then
        echo "build inputs differ from git (uncommitted edits under: ${srcs[*]}); no pull" >&2
        return 1
    fi
    {
        printf '%s\n' "${prefix}"
        printf 'target=%s\nbase=%s\n' "${target}" "${BASE_IMAGE_REF:-${BASE_IMAGE:-}}"
        local prev="" arg
        for arg in "$@"; do
            [[ "${prev}" != --build-arg || "${arg}" == BASE_IMAGE=* || "${arg}" == BASE_IMAGE_REF=* ]] \
                || printf '%s\n' "${arg}"
            prev="${arg}"
        done | sort
        (( ${#srcs[@]} == 0 )) || git -C "${context}" ls-files -s -- "${srcs[@]}"
    } | sha256sum | cut -d' ' -f1
}

# Prints the value of image label $2 of registry image $1 (repo:tag) from its config blob, without
# pulling any layer. Anonymous; docker.io gets its pull token. Exit 1 when the tag cannot be read.
ce_registry_label() {
    local ref="$1" label="$2" repo tag host path token manifest arch digest accept
    local -a auth=()
    repo="${ref%:*}" tag="${ref##*:}"
    host="${repo%%/*}" path="${repo#*/}"
    command -v jq >/dev/null && command -v curl >/dev/null || return 1
    if [[ "${host}" == docker.io ]]; then
        host=registry-1.docker.io
        token="$(curl -fsS "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${path}:pull" \
            | jq -r .token)" || return 1
        auth=(-H "Authorization: Bearer ${token}")
    fi
    accept="application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json"
    accept+=",application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json"
    manifest="$(curl -fsSL "${auth[@]}" -H "Accept: ${accept}" "https://${host}/v2/${path}/manifests/${tag}")" \
        || return 1
    if jq -e .manifests >/dev/null <<<"${manifest}"; then
        case "$(uname -m)" in x86_64) arch=amd64 ;; aarch64) arch=arm64 ;; *) arch="$(uname -m)" ;; esac
        digest="$(jq -r --arg a "${arch}" \
            '[.manifests[] | select(.platform.architecture == $a)][0].digest // empty' <<<"${manifest}")"
        [[ -n "${digest}" ]] || return 1
        manifest="$(curl -fsSL "${auth[@]}" -H "Accept: ${accept}" \
            "https://${host}/v2/${path}/manifests/${digest}")" || return 1
    fi
    digest="$(jq -r '.config.digest // empty' <<<"${manifest}")"
    [[ -n "${digest}" ]] || return 1
    curl -fsSL "${auth[@]}" "https://${host}/v2/${path}/blobs/${digest}" \
        | jq -r --arg l "${label}" '.config.Labels[$l] // empty'
}

# ce_pull_wanted <role> <fingerprint>: exit 0 when the registry image of the role's images.env tag
# should replace the build (CE_PULL=only, or CE_PULL=1 and its build-inputs label equals the
# fingerprint), 1 to build, 2 when CE_PULL=only cannot name a tag.
ce_pull_wanted() {
    local role="$1" want="$2" tag have
    [[ "${CE_PULL}" != 0 ]] || return 1
    tag="$(ce_image "${role}" tag 2>/dev/null)" || tag=""
    if [[ -z "${tag}" ]]; then
        [[ "${CE_PULL}" != only ]] || { echo "CE_PULL=only: ${role} has no tag in images.env" >&2; return 2; }
        printf '%s: no registry tag in images.env; building\n' "${role}"
        return 1
    fi
    [[ "${CE_PULL}" != only ]] || return 0
    have="$(ce_registry_label "${PULL_REPO}:${tag}" "${CE_INPUTS_LABEL}" 2>/dev/null)" || have=""
    if [[ -n "${want}" && "${have}" == "${want}" ]]; then
        printf '%s: %s:%s was built from these inputs (%s); pulling\n' "${role}" "${PULL_REPO}" "${tag}" "${want:0:12}"
        return 0
    fi
    printf '%s: %s:%s inputs %s, this checkout %s; building\n' "${role}" "${PULL_REPO}" "${tag}" \
        "${have:-<unread or unlabelled>}" "${want:-<unknown>}"
    return 1
}

# Pull first. Each spec is "role|target|local tag|output sqsh" (target empty for a one-stage
# Dockerfile); the build args are what ce_build will pass. Sets CE_PULLED=1 when EVERY spec was
# pulled and exported, else 0 and the caller builds them all (a later stage builds FROM an earlier
# one, so a partial pull spares nothing); exit 2 when CE_PULL=only cannot be met. Call it plainly,
# not under || or if: the export relies on set -e. Sets CE_FINGERPRINT[target] for ce_build.
#   ce_pull_first <dockerfile> <spec>... -- <podman build args>
declare -A CE_FINGERPRINT=()
ce_pull_first() {
    local dockerfile="$1" spec role target tag out ref rc pull=1
    local -a specs=()
    CE_PULLED=0
    shift
    while [[ $# -gt 0 && "$1" != -- ]]; do specs+=("$1"); shift; done
    [[ $# -eq 0 ]] || shift
    for spec in "${specs[@]}"; do
        IFS='|' read -r role target tag out <<<"${spec}"
        CE_FINGERPRINT[${target:-_}]="$(ce_build_fingerprint "${dockerfile}" "${target}" "$@")" \
            || CE_FINGERPRINT[${target:-_}]=""
        rc=0
        ce_pull_wanted "${role}" "${CE_FINGERPRINT[${target:-_}]}" || rc=$?
        [[ "${rc}" -ne 2 ]] || return 2
        [[ "${rc}" -eq 0 ]] || pull=0
    done
    [[ "${pull}" == 1 ]] || return 0
    for spec in "${specs[@]}"; do
        IFS='|' read -r role target tag out <<<"${spec}"
        ref="${PULL_REPO}:$(ce_image "${role}" tag)"
        if ! podman pull "${ref}" || ! podman tag "${ref}" "${tag}"; then
            [[ "${CE_PULL}" != only ]] || return 2
            echo "pull of ${ref} FAILED; building instead" >&2
            return 0
        fi
    done
    # Exported as pulled, never pushed back (ce_export_image reads CE_PULLED).
    CE_PULLED=1
    for spec in "${specs[@]}"; do
        IFS='|' read -r role target tag out <<<"${spec}"
        ce_export_image "${tag}" "${out}"
    done
}

# Builds one target with the layer cache flags and the build-inputs label ce_pull_first computed,
# from the repository root as context, and exports it. BASE_IMAGE/BASE_IMAGE_REF come last, after
# ce_cache_base_image may have pointed BASE_IMAGE at the local copy.
#   ce_build <dockerfile> <target|""> <local tag> <output sqsh> <podman build args>
ce_build() {
    local dockerfile="$1" target="$2" tag="$3" out="$4" fp
    local -a extra=()
    shift 4
    fp="${CE_FINGERPRINT[${target:-_}]:-}"
    [[ -z "${fp}" ]] || extra+=(--label "${CE_INPUTS_LABEL}=${fp}")
    [[ -z "${target}" ]] || extra+=(--target "${target}")
    printf '\n===== building %s -> %s =====\n' "${target:-${tag}}" "${out}"
    # cgroupfs: with the systemd manager a dying logind session reaps podman mid-pull (silent rc=1).
    (cd -- "${CE_IMAGES_DIR}/../.." && podman --cgroup-manager=cgroupfs build "${CE_BUILD_FLAGS[@]}" "$@" \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --build-arg "BASE_IMAGE_REF=${BASE_IMAGE_REF:-${BASE_IMAGE}}" \
        "${extra[@]}" -f "${dockerfile}" -t "${tag}" .)
    ce_export_image "${tag}" "${out}"
}

# Verifies a candidate inside itself, under an EDF rendered from its production template, and
# writes the .verified marker promote_image.sh requires only on a clean verdict. Used by the GH200
# and CPU build.sbatch; AMD images use build_and_verify.sbatch. Probes run from `/` so a `dace`
# directory in the CWD cannot shadow the image's own.
#
#   ce_verify_candidate <template under containers/images/> <sqsh> <verify_image.py profile> [srun args...]
ce_verify_candidate() {
    local template="$1" sqsh="$2" profile="$3" repo edf modules rc=0
    shift 3
    repo="$(cd -- "${CE_IMAGES_DIR}/../.." && pwd)"
    # shellcheck source=../../scripts/cache_env.sh
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
                --agent-dir /opt/hpcagent-bench-agent --judge-web-search "${repo}/hpcagent_bench/harness/judge_web_search.py" \
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

# ce_refuse_mounted <sqsh>...: return 2 when an EDF in ~/.edf mounts one of the paths. Overwriting a
# mounted squashfs is how a running job starts reading a half-written inode table; the EDFs, not a
# hard-coded name, are what a job resolves. Build to a candidate name and promote by rename.
ce_refuse_mounted() {
    local mounted sqsh
    mounted="$(grep -hoE '^[[:space:]]*image[[:space:]]*=[[:space:]]*"[^"]+"' "${HOME}/.edf"/*.toml 2>/dev/null \
        | sed -E 's/.*"(.*)"/\1/' | sed -E "s|\\\$\\{SCRATCH\\}|${SCRATCH:-}|g; s|\\\$SCRATCH|${SCRATCH:-}|g")" || true
    for sqsh in "$@"; do
        [[ -n "${sqsh}" ]] || continue
        if grep -qxF -- "${sqsh}" <<<"${mounted}"; then
            echo "refusing to overwrite ${sqsh}: an EDF in ${HOME}/.edf mounts it, so a running job" >&2
            echo "would read a half-written squashfs. Build to a candidate name and promote by rename." >&2
            return 2
        fi
    done
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
            echo "would need a full rebuild. See containers/images/push_image.sh." >&2
        fi
    fi

    # Optional push, after the artifacts exist, so a registry failure never costs the build.
    if [[ -n "${PUSH_REPO:-}" && "${CE_PULLED:-0}" != 1 ]]; then
        "${CE_IMAGES_DIR}/push_image.sh" "${image_tag}" ${PUSH_TAGS:-} \
          || echo "push to ${PUSH_REPO} FAILED; the local image and squashfs are unaffected" >&2
    fi
}
