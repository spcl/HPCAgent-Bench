#!/usr/bin/env bash
# Shared by every <image>/build.sh. What differs between images is the base, the build-args and
# the Dockerfile; everything around that was copied five times, which is how the OCI archive
# reached four builders and not the fifth, and how one of them still called its build directory
# by a different name.
#
# Source it, do not execute it:
#   source "$(dirname -- "${BASH_SOURCE[0]}")/../build_common.sh"

CE_IMAGES_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Diskless nodes: temp and runtime dirs on /dev/shm, stale per-node podman state wiped (it is only
# a cache). DBUS_SESSION_BUS_ADDRESS is unset so podman does not try to talk to a session bus that
# is not there.
ce_podman_env() {
    unset DBUS_SESSION_BUS_ADDRESS
    export TMPDIR="/dev/shm/${USER}/tmp"
    export XDG_RUNTIME_DIR="/dev/shm/${USER}/xdg"
    # The wipe goes through `podman unshare`: an image layer under root/overlay/*/diff is owned by
    # a SUBUID, not by this user, so a plain rm hits Permission denied and leaves a half-deleted
    # store the next build dies on. unshare enters the user namespace where those subuids map here.
    podman unshare rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" 2>/dev/null || true
    rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" "/dev/shm/${USER}/tmp" \
           "/dev/shm/${USER}/xdg"
    mkdir -p "${TMPDIR}"
    mkdir -p -m 0700 "${XDG_RUNTIME_DIR}"
}

# Sets the global MIRROR_ARGS. Every clone in the build is rewritten to the mirror (see each
# Dockerfile), which took GitHub off the critical path: the rate limiter answers an
# unauthenticated clone with a 401 under load, and the callers that died on it -- spack's
# in-process package-repo clone, vLLM's CMake FetchContent of triton -- have no retry. Refresh it
# from a login node with mirror-repos.sh. Absent, the build still works and still uses GitHub.
ce_mirror_args() {
    MIRROR_ARGS=()
    GIT_MIRRORS="${GIT_MIRRORS:-${SCRATCH:-}/git-mirrors}"
    if [[ -d "${GIT_MIRRORS}" ]]; then
        MIRROR_ARGS=(-v "${GIT_MIRRORS}:/git-mirrors:ro")
        printf 'git mirror %s\n' "${GIT_MIRRORS}"
    fi
}

# A commit resolved from GITHUB but cloned from the MIRROR is a sha the mirror may not have, and
# git says so as `upload-pack: not our ref` -- two hours in, with every expensive layer already
# paid for. That is build 626608. Check it here, where it costs seconds.
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

# Sets the global GPU_ARGS. aiter >= 0.1.19 reads the arch from `rocminfo` at IMPORT time and
# ignores GPU_ARCHS on purpose, and vLLM's rocm.py probes the device too -- so a device-less build
# cannot even import them. Measured in job 619976: an mi300 job with NO --gres still exposes
# /dev/kfd, and `podman build --device` reports gfx942 inside a RUN step. Conditional, so a build
# on a node without the device fails in the step that needs it rather than on an unusable flag.
ce_gpu_args() {
    GPU_ARGS=()
    if [[ -e /dev/kfd ]]; then
        GPU_ARGS=(--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined
                  --group-add keep-groups)
        printf 'gpu devices handed to the build\n'
    fi
}

# Base image cache on scratch; rewrites the global BASE_IMAGE to a local `dir:` on a hit.
#
# The podman LAYER store cannot live on scratch: capstor, iopsstor and the NFS home all reject
# user xattrs, so `overlay` and `fuse-overlayfs` fail on lsetxattr and `vfs` fails creating its
# pivot dir under a subuid (all three measured). The base image can, because a `dir:` tree is
# plain files -- and it is the part worth caching, a 30-52 GB pull per job on a store that is
# wiped every time because the nodes are diskless and it lives in RAM.
#
# Miss: pull over the network as before, then copy out for next time; the build still reads the
# copy already in the store, so this costs one write and never a second pull. Staged through a
# temp dir and renamed, so two builds racing cannot leave a half-written tree that a later job
# would treat as a hit.
ce_cache_base_image() {
    BASE_CACHE="${BASE_CACHE:-${SCRATCH:?}/base-images}"
    # The REGISTRY reference, kept before BASE_IMAGE is rewritten to a local dir: on a cache hit
    # BASE_IMAGE becomes dir:${SCRATCH}/base-images/..., and labelling the image with that records
    # a path on somebody's scratch instead of the digest it came from -- a host path published
    # inside the artifact, and the reproducibility claim in base.name destroyed. Measured in the
    # vLLM 0.23.0 archive before this existed.
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

# Everything after a successful `podman build`: identity, artifact, archive, publish.
ce_export_image() {
    local image_tag="$1" output_sqsh="$2"

    # The digest IS the version. Recorded next to the squashfs so a results table can name the
    # exact image a campaign ran on without trusting a mutable tag.
    podman image inspect --format '{{.Digest}}' "${image_tag}" > "${output_sqsh}.digest"
    printf 'image digest %s\n' "$(cat "${output_sqsh}.digest")"

    # enroot's exit code lies when cleanup fails after a good write, so gate on the ARTIFACT:
    # listing reads the inode table at file END, which a truncated image fails. Remove the output
    # first -- enroot refuses to overwrite, `|| true` swallows that, and `unsquashfs -l` would
    # then validate LAST run's file (620068 printed "IMAGE READY" over a stale image).
    rm -f "${output_sqsh}"
    enroot import -x mount -o "${output_sqsh}" "podman://${image_tag}" || true
    unsquashfs -l "${output_sqsh}" opt >/dev/null
    printf 'Wrote %s\n' "${output_sqsh}"

    # An OCI archive beside the squashfs, so publishing does not have to happen during the build.
    # The squashfs cannot stand in: it is a flattened filesystem, so reimporting one collapses the
    # image into a single layer far past the registry's per-layer ceiling and drops the image
    # config. The archive keeps both, which turns `push_image.sh --from-archive` into a short job
    # rather than a multi-hour rebuild. SAVE_OCI_ARCHIVE=0 skips it and accepts that cost.
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

    # Optional registry push, AFTER the artifacts are written, so a registry outage or a rejected
    # layer costs the upload and never what this job exists to produce.
    if [[ -n "${PUSH_REPO:-}" ]]; then
        "${CE_IMAGES_DIR}/push_image.sh" "${image_tag}" ${PUSH_TAGS:-} \
          || echo "push to ${PUSH_REPO} FAILED; the local image and squashfs are unaffected" >&2
    fi
}
