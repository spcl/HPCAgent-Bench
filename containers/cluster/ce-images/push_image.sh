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
CHECK_ONLY=0
# --check-only runs every gate that can reject an upload and then STOPS: no login, no bytes out.
# The registry limits below are the reason it exists. A single layer over the ceiling means the
# image cannot be published at all and the Dockerfile has to split that RUN -- discovering that
# partway through a 60 GB upload costs the upload and the hours around it. Order matters too:
# nothing here needs credentials, so readiness can be established before anyone hands them over.
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --from-archive)
            ARCHIVE="${2:?--from-archive needs the path to a .oci.tar written by build.sh}"
            shift 2 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        *) echo "unknown flag $1" >&2; exit 2 ;;
    esac
done
# Only a real push needs a destination.
if (( CHECK_ONLY )); then
    PUSH_REPO="${PUSH_REPO:-docker.io/local/check-only}"
else
    PUSH_REPO="${PUSH_REPO:?set PUSH_REPO, e.g. docker.io/<user>/optarena-judge-agent-amd}"
fi

if [[ -n "${ARCHIVE}" ]]; then
    [[ -f "${ARCHIVE}" ]] || { echo "no archive at ${ARCHIVE}" >&2; exit 2; }
    # Isolated and tmpfs, per the header. Not the default graphroot: a build on this node is using
    # that one, and unpacking 60 GB into it is how you break someone else's job.
    root="${PUSH_ROOT:-/dev/shm/${USER}/push-$$}"
    mkdir -p "${root}/root" "${root}/run"
    trap 'podman unshare rm -rf "${root}" 2>/dev/null || true' EXIT
    # ignore_chown_errors is REQUIRED here, not a tuning knob. This system has no /etc/subuid or
    # /etc/subgid entries, so `podman unshare` gets a uid_map of length 1 (0 -> $UID) and there is
    # no subordinate id to map a file owned by anything else to. Any layer carrying one -- the
    # base image's /etc/gshadow is gid 42 -- then fails to unpack with
    # "lchown /etc/gshadow: invalid argument", and the push dies before a byte goes out. Measured:
    # the default driver and plain overlay both fail on it, overlay + this option loads cleanly.
    # The chown is only lost inside this throwaway graphroot; the layer blobs pushed are the
    # archive's own bytes, so the published image is unaffected.
    PODMAN=(podman --root "${root}/root" --runroot "${root}/run" --storage-driver overlay
            --storage-opt ignore_chown_errors=true)
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
# COMPRESSED blob size, which is what the registry actually receives and what its ceiling is
# about. `podman history` reports the UNCOMPRESSED diff, and the two differ by ~3x here: it called
# the sglang image's biggest layer 23.3 GB where the blob that would be uploaded is 7.84 GB. Gating
# on the uncompressed number condemns images that would upload perfectly well, which is a worse
# failure than not checking -- it sends you into a rebuild you did not need.
#
# For an archive the manifest is authoritative and free to read, so prefer it. Only the in-build
# path (no archive) falls back to history, and there the number is an upper bound, not the limit.
biggest=""
if [[ -n "${ARCHIVE}" ]]; then
    biggest="$(python3 - "${ARCHIVE}" <<'PY'
import json, sys, tarfile
with tarfile.open(sys.argv[1]) as tar:
    index = json.load(tar.extractfile("index.json"))
    digest = index["manifests"][0]["digest"].split(":", 1)[1]
    manifest = json.load(tar.extractfile(f"blobs/sha256/{digest}"))
    print(max((layer["size"] for layer in manifest.get("layers", [])), default=0))
PY
)" || biggest=""
    [[ -n "${biggest}" ]] && printf 'largest layer measured from the archive manifest (compressed)\n'
fi
if [[ -z "${biggest}" ]]; then
    printf 'no archive manifest; falling back to uncompressed history size (an UPPER BOUND)\n'
    biggest="$("${PODMAN[@]}" history --format json "${LOCAL_TAG}" \
               | grep -o '"size":[0-9]*' | cut -d: -f2 | sort -n | tail -1)"
fi
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

if (( CHECK_ONLY )); then
    printf 'CHECK ONLY: within registry limits, nothing pushed.\n'
    printf '  image        %s GB (limit %s)\n' "$(bytes_to_gb "${total}")" "${MAX_IMAGE_GB}"
    printf '  largest layer %s GB (limit %s)\n' "$(bytes_to_gb "${biggest:-0}")" "${MAX_LAYER_GB}"
    printf '  manifest     %s\n' "$("${PODMAN[@]}" image inspect --format '{{.ManifestType}}' "${LOCAL_TAG}")"
    printf '  digest       %s\n' "$("${PODMAN[@]}" image inspect --format '{{.Digest}}' "${LOCAL_TAG}")"
    exit 0
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
