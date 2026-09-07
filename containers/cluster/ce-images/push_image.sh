#!/usr/bin/env bash
# Push a built image to a registry, so it can be pulled instead of rebuilt.
#
# WHEN THIS CAN RUN, which is the whole design: podman's graphroot here is /dev/shm/$USER/root --
# node-local tmpfs that every build.sh wipes on entry and that dies with the job. The image exists
# only between `podman build` and the end of that job. So this runs INSIDE the build job, from
# build.sh, and a standalone "push the images we already have" is not possible: the squashfs files
# on scratch are a flattened filesystem, not an OCI image, and reimporting one loses the layer
# structure and the image config. An image that was not pushed when it was built has to be rebuilt
# to be pushed.
#
#   REGISTRY_USER=<user> REGISTRY_TOKEN=<token> \
#     PUSH_REPO=docker.io/<user>/optarena-judge-agent-amd ./push_image.sh <local-tag> [extra-tag...]
#
# Credentials come from the environment and are never written into the repo or echoed. A token
# (Docker Hub: Account Settings -> Personal access tokens) rather than a password, so it can be
# scoped and revoked; if a login already exists in the ambient auth file, set no credentials and
# this uses it.
set -Eeuo pipefail

LOCAL_TAG="${1:?usage: push_image.sh <local-tag> [extra-tag...]}"
shift || true
PUSH_REPO="${PUSH_REPO:?set PUSH_REPO, e.g. docker.io/<user>/optarena-judge-agent-amd}"

# Docker Hub rejects a layer over 10 GB and an image over 100 GB. Checked BEFORE the first byte
# goes out, because the alternative is discovering it partway through a multi-hour upload. Other
# registries differ (ECR allows 200 GB layers as of 2026-08), so the limits are overridable rather
# than baked in.
MAX_LAYER_GB="${MAX_LAYER_GB:-10}"
MAX_IMAGE_GB="${MAX_IMAGE_GB:-100}"

bytes_to_gb() { awk -v b="$1" 'BEGIN { printf "%.1f", b / 1073741824 }'; }

total="$(podman image inspect --format '{{.Size}}' "${LOCAL_TAG}")"
printf 'image %s: %s GB total\n' "${LOCAL_TAG}" "$(bytes_to_gb "${total}")"
if awk -v b="${total}" -v m="${MAX_IMAGE_GB}" 'BEGIN { exit !(b > m * 1073741824) }'; then
    echo "refusing to push: $(bytes_to_gb "${total}") GB exceeds the ${MAX_IMAGE_GB} GB image limit" >&2
    exit 2
fi

# `podman history` reports per-layer sizes; the largest is the one that decides whether a push can
# succeed at all.
biggest="$(podman history --format '{{.Size}}' --no-trunc "${LOCAL_TAG}" \
           | numfmt --from=auto 2>/dev/null | sort -n | tail -1)"
if [[ -n "${biggest}" ]]; then
    printf 'largest layer: %s GB (registry limit %s GB)\n' "$(bytes_to_gb "${biggest}")" "${MAX_LAYER_GB}"
    if awk -v b="${biggest}" -v m="${MAX_LAYER_GB}" 'BEGIN { exit !(b > m * 1073741824) }'; then
        echo "refusing to push: a single layer exceeds ${MAX_LAYER_GB} GB, which the registry will" >&2
        echo "reject partway through the upload. Split that RUN into smaller layers first." >&2
        exit 2
    fi
fi

if [[ -n "${REGISTRY_USER:-}" && -n "${REGISTRY_TOKEN:-}" ]]; then
    registry="${PUSH_REPO%%/*}"
    printf '%s' "${REGISTRY_TOKEN}" | podman login --username "${REGISTRY_USER}" --password-stdin "${registry}"
fi

# The DIGEST is the version, exactly as build.sh records it locally -- a tag is mutable and two
# builds under one tag is the thing that made a results table unreadable before. Pushed as
# `sha-<12>` alongside whatever human-facing tags the caller names, so a campaign can always cite
# something immutable.
digest="$(podman image inspect --format '{{.Digest}}' "${LOCAL_TAG}")"
short="sha-${digest#sha256:}"; short="${short:0:16}"

pushed=()
for tag in "${short}" "$@"; do
    target="${PUSH_REPO}:${tag}"
    echo "pushing ${target}"
    podman tag "${LOCAL_TAG}" "${target}"
    podman push "${target}"
    pushed+=("${target}")
done

printf '\npushed:\n'
printf '  %s\n' "${pushed[@]}"
printf 'pull with:\n  podman pull %s:%s\n' "${PUSH_REPO}" "${short}"
printf 'or import straight to squashfs:\n  enroot import -x mount -o <name>.sqsh docker://%s:%s\n' \
    "${PUSH_REPO}" "${short}"
