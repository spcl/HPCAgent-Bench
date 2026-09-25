#!/usr/bin/env bash
# Push one image to a registry, from the build's podman store or from its saved OCI archive.
#
#   REGISTRY_USER=<user> REGISTRY_TOKEN=<token> PUSH_REPO=docker.io/<user>/hpcagent-bench \
#     ./push_image.sh <local-tag> [extra-tag...]
#   PUSH_REPO=... ./push_image.sh --from-archive $SCRATCH/ce-images/<name>.oci.tar [extra-tag...]
#   ./push_image.sh --check-only --from-archive <archive>    # every gate, no login, no upload
#
# --from-archive unpacks into an isolated tmpfs graphroot (Lustre cannot hold one), so the node
# needs RAM for the decompressed image. Credentials come from the environment only; with none set
# an existing login in the ambient auth file is used. Every push also tags sha-<digest>.
set -Eeuo pipefail

ulimit -c 0
PODMAN=(podman)
ARCHIVE=""
CHECK_ONLY=0
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --from-archive)
            ARCHIVE="${2:?--from-archive needs the path to a .oci.tar written by build.sh}"
            shift 2 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        *) echo "unknown flag $1" >&2; exit 2 ;;
    esac
done
if (( CHECK_ONLY )); then
    PUSH_REPO="${PUSH_REPO:-docker.io/local/check-only}"
else
    PUSH_REPO="${PUSH_REPO:?set PUSH_REPO, e.g. docker.io/<user>/hpcagent-bench-judge-agent-amd}"
fi

if [[ -n "${ARCHIVE}" ]]; then
    [[ -f "${ARCHIVE}" ]] || { echo "no archive at ${ARCHIVE}" >&2; exit 2; }
    # Not the default graphroot: a build on this node may be using it.
    root="${PUSH_ROOT:-/dev/shm/${USER}/push-$$}"
    mkdir -p "${root}/root" "${root}/run"
    trap 'podman unshare rm -rf "${root}" 2>/dev/null || true' EXIT
    # No /etc/subuid entries here, so layers owned by other ids fail to unpack without
    # ignore_chown_errors. Only this throwaway store loses the chown; the pushed blobs are the
    # archive's own bytes.
    PODMAN=(podman --root "${root}/root" --runroot "${root}/run" --storage-driver overlay
            --storage-opt ignore_chown_errors=true)
    echo "loading ${ARCHIVE} into ${root}"
    LOCAL_TAG="$("${PODMAN[@]}" pull "oci-archive:${ARCHIVE}" | tail -1)"
    [[ -n "${LOCAL_TAG}" ]] || { echo "loaded nothing from ${ARCHIVE}" >&2; exit 2; }
else
    LOCAL_TAG="${1:?usage: push_image.sh <local-tag> [extra-tag...]}"
    shift || true
fi

# Docker Hub's limits (10 GB per layer, 100 GB per image), checked before any upload.
MAX_LAYER_GB="${MAX_LAYER_GB:-10}"
MAX_IMAGE_GB="${MAX_IMAGE_GB:-100}"

bytes_to_gb() { awk -v b="$1" 'BEGIN { printf "%.1f", b / 1073741824 }'; }

total="$("${PODMAN[@]}" image inspect --format '{{.Size}}' "${LOCAL_TAG}")"
printf 'image %s: %s GB total\n' "${LOCAL_TAG}" "$(bytes_to_gb "${total}")"
if awk -v b="${total}" -v m="${MAX_IMAGE_GB}" 'BEGIN { exit !(b > m * 1073741824) }'; then
    echo "refusing to push: $(bytes_to_gb "${total}") GB exceeds the ${MAX_IMAGE_GB} GB image limit" >&2
    exit 2
fi

# The ceiling applies to the compressed blob: read it from the archive manifest; without an archive
# `podman history` gives the uncompressed size, an upper bound.
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

# sha-<digest> is the immutable tag to cite. --format oci: podman otherwise follows the source
# manifest type, which BUILDAH_FORMAT can flip to docker v2s2.
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
