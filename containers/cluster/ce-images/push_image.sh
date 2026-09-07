#!/usr/bin/env bash
# Push a built image to a registry, so it can be pulled instead of rebuilt.
#
# TWO WAYS IN, and the second exists because the first used to be the only one.
#
# podman's graphroot here is /dev/shm/$USER/root -- node-local tmpfs that every build.sh wipes on
# entry and that dies with the job. So a freshly built image exists only between `podman build`
# and the end of that job, and the .sqsh left on scratch is a flattened filesystem rather than an
# OCI image: reimporting one collapses it to a single layer, loses the image config, and lands
# well over the registry's per-layer ceiling anyway. That is why "push the images we already have"
# was impossible and an unpushed image had to be REBUILT to be published.
#
# build.sh now also writes an OCI archive beside the squashfs, which keeps the layer structure and
# the config, so publishing no longer has to happen during the build:
#
#   in-build, from build.sh, straight out of the build graphroot:
#     REGISTRY_USER=<user> REGISTRY_TOKEN=<token> \
#       PUSH_REPO=docker.io/<user>/optarena-judge-agent-amd ./push_image.sh <local-tag> [extra-tag...]
#
#   later, on any compute node, from the saved archive and with no rebuild:
#     REGISTRY_USER=<user> REGISTRY_TOKEN=<token> \
#       PUSH_REPO=docker.io/<user>/optarena-judge-agent-amd \
#       ./push_image.sh --from-archive $SCRATCH/ce-images/<name>.oci.tar [extra-tag...]
#
# The archive path loads into an ISOLATED graphroot so it cannot disturb a build sharing the node,
# and that graphroot must be tmpfs: rootless layer extraction onto Lustre fails when it cannot
# create its pivot dir. Which means the node needs free RAM for the decompressed image -- 60+ GB
# for the judge+agent one, the same as the build itself needs.
#
# Credentials come from the environment and are never written into the repo or echoed. A token
# (Docker Hub: Account Settings -> Personal access tokens) rather than a password, so it can be
# scoped and revoked; if a login already exists in the ambient auth file, set no credentials and
# this uses it.
set -Eeuo pipefail

PODMAN=(podman)
ARCHIVE=""
if [[ "${1:-}" == "--from-archive" ]]; then
    ARCHIVE="${2:?--from-archive needs the path to a .oci.tar written by build.sh}"
    shift 2
fi
PUSH_REPO="${PUSH_REPO:?set PUSH_REPO, e.g. docker.io/<user>/optarena-judge-agent-amd}"

if [[ -n "${ARCHIVE}" ]]; then
    [[ -f "${ARCHIVE}" ]] || { echo "no archive at ${ARCHIVE}" >&2; exit 2; }
    # Isolated and tmpfs, per the header. Not the default graphroot: a build on this node is using
    # that one, and unpacking 60 GB into it is how you break someone else's job.
    root="${PUSH_ROOT:-/dev/shm/${USER}/push-$$}"
    mkdir -p "${root}/root" "${root}/run"
    trap 'podman unshare rm -rf "${root}" 2>/dev/null || true' EXIT
    PODMAN=(podman --root "${root}/root" --runroot "${root}/run" --storage-driver overlay)
    echo "loading ${ARCHIVE} into ${root}"
    LOCAL_TAG="$("${PODMAN[@]}" pull "oci-archive:${ARCHIVE}" | tail -1)"
    [[ -n "${LOCAL_TAG}" ]] || { echo "loaded nothing from ${ARCHIVE}" >&2; exit 2; }
else
    LOCAL_TAG="${1:?usage: push_image.sh <local-tag> [extra-tag...]}"
    shift || true
fi

# Docker Hub rejects a layer over 10 GB and an image over 100 GB. Checked BEFORE the first byte
# goes out, because the alternative is discovering it partway through a multi-hour upload. Other
# registries differ (ECR allows 200 GB layers as of 2026-08), so the limits are overridable rather
# than baked in.
MAX_LAYER_GB="${MAX_LAYER_GB:-10}"
MAX_IMAGE_GB="${MAX_IMAGE_GB:-100}"

bytes_to_gb() { awk -v b="$1" 'BEGIN { printf "%.1f", b / 1073741824 }'; }

total="$("${PODMAN[@]}" image inspect --format '{{.Size}}' "${LOCAL_TAG}")"
printf 'image %s: %s GB total\n' "${LOCAL_TAG}" "$(bytes_to_gb "${total}")"
if awk -v b="${total}" -v m="${MAX_IMAGE_GB}" 'BEGIN { exit !(b > m * 1073741824) }'; then
    echo "refusing to push: $(bytes_to_gb "${total}") GB exceeds the ${MAX_IMAGE_GB} GB image limit" >&2
    exit 2
fi

# `podman history` reports per-layer sizes; the largest is the one that decides whether a push can
# succeed at all. Read them out of --format json, whose `size` is a plain byte count: the Go
# template's {{.Size}} humanises to strings like `2.05kB` and `0B`, and `numfmt --from=auto`
# rejects both of those two-letter suffixes. That combination made this gate exit 2 on EVERY
# image under `set -e`, with the reason swallowed -- a size check meant to prevent a rejected
# upload was instead refusing all of them.
biggest="$("${PODMAN[@]}" history --format json "${LOCAL_TAG}" \
           | grep -o '"size":[0-9]*' | cut -d: -f2 | sort -n | tail -1)"
if [[ -z "${biggest}" ]]; then
    echo "warning: could not read per-layer sizes; pushing without the layer-ceiling check" >&2
else
    printf 'largest layer: %s GB (registry limit %s GB)\n' "$(bytes_to_gb "${biggest}")" "${MAX_LAYER_GB}"
    if awk -v b="${biggest}" -v m="${MAX_LAYER_GB}" 'BEGIN { exit !(b > m * 1073741824) }'; then
        echo "refusing to push: a single layer exceeds ${MAX_LAYER_GB} GB, which the registry will" >&2
        echo "reject partway through the upload. Split that RUN into smaller layers first." >&2
        exit 2
    fi
fi

if [[ -n "${REGISTRY_USER:-}" && -n "${REGISTRY_TOKEN:-}" ]]; then
    registry="${PUSH_REPO%%/*}"
    printf '%s' "${REGISTRY_TOKEN}" | "${PODMAN[@]}" login --username "${REGISTRY_USER}" --password-stdin "${registry}"
fi

# The DIGEST is the version, exactly as build.sh records it locally -- a tag is mutable and two
# builds under one tag is the thing that made a results table unreadable before. Pushed as
# `sha-<12>` alongside whatever human-facing tags the caller names, so a campaign can always cite
# something immutable.
# What goes UP is an OCI image, stated rather than inherited. podman is only the tool used to
# move it: `podman push --format` defaults to "manifest type of source, with fallbacks", podman
# build's default can be flipped by BUILDAH_FORMAT in the environment, and a docker v2s2 manifest
# would otherwise be published without anything saying so. --format oci below is what makes the
# published artifact an OCI image regardless of how it was built or which path it took to here.
manifest="$("${PODMAN[@]}" image inspect --format '{{.ManifestType}}' "${LOCAL_TAG}")"
printf 'source manifest: %s\n' "${manifest}"
printf 'pushing as:      application/vnd.oci.image.manifest.v1+json (forced with --format oci)\n'

digest="$("${PODMAN[@]}" image inspect --format '{{.Digest}}' "${LOCAL_TAG}")"
short="sha-${digest#sha256:}"; short="${short:0:16}"

pushed=()
for tag in "${short}" "$@"; do
    target="${PUSH_REPO}:${tag}"
    echo "pushing ${target}"
    "${PODMAN[@]}" tag "${LOCAL_TAG}" "${target}"
    "${PODMAN[@]}" push --format oci "${target}"
    pushed+=("${target}")
done

printf '\npushed:\n'
printf '  %s\n' "${pushed[@]}"
printf 'pull with:\n  podman pull %s:%s\n' "${PUSH_REPO}" "${short}"
printf 'or import straight to squashfs:\n  enroot import -x mount -o <name>.sqsh docker://%s:%s\n' \
    "${PUSH_REPO}" "${short}"
