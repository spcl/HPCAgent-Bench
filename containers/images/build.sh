#!/usr/bin/env bash
# Build one image and export each of its targets as a squashfs candidate the Container Engine can mount.
#
#   containers/images/build.sh <image>     <image> = a directory here: judge-agent-amd, judge-agent-cpu,
#                                           judge-agent-cuda, sglang, vllm, vllm-cuda
#   BUILD_TARGETS=judge containers/images/build.sh judge-agent-amd    one target (OUTPUT_SQSH overrides its path)
#   CE_IMAGE_FLAVOR=native containers/images/build.sh judge-agent-cpu this machine's CPU; never pushed
#   containers/images/build.sh <image> --roles                        print the images.env roles it writes
#
# <image>/image.sh holds what differs per image: TARGET_ROLE (Dockerfile target -> images.env role,
# "-" for a one-stage Dockerfile) with TARGET_ORDER, BASE_IMAGE pinned by digest, ce_image_args (the
# build args, which key the pull-first fingerprint) and ce_image_inputs (mirror and cache mounts, needed
# only when it really builds). Everything else is here and the same for every image: candidate names
# from images.env, the mounted-image guard, pull first (build_common.sh ce_pull_first), the build loop
# and the .sha256 sidecars. Later targets build FROM earlier ones (judge is FROM agent), so they build
# in TARGET_ORDER and each is exported before the next starts. The tag carries no version: identity is
# git plus the digest sidecar ce_export_image writes. Nothing points at a candidate until
# `registry.sh promote` renames it over the live name.
set -euo pipefail
# The cluster's core_pattern dumps into the CWD and Slurm propagates the submitter's core limit.
ulimit -c 0
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=build_common.sh
source "${HERE}/build_common.sh"

IMAGE="${1:?usage: build.sh <image directory under containers/images>}"
IMAGE_DIR="${HERE}/${IMAGE}"
[[ -f "${IMAGE_DIR}/image.sh" ]] || { echo "no ${IMAGE_DIR}/image.sh" >&2; exit 2; }
TARGET_ORDER=""
declare -A TARGET_ROLE=()
BUILD_ARGS=()
INPUT_ARGS=()
# shellcheck source=/dev/null
source "${IMAGE_DIR}/image.sh"
read -r -a targets <<<"${BUILD_TARGETS:-${TARGET_ORDER}}"
for target in "${targets[@]}"; do
    [[ -n "${TARGET_ROLE[${target}]:-}" ]] || { echo "${IMAGE} has no build target '${target}'" >&2; exit 2; }
done
if [[ "${2:-}" == --roles ]]; then
    for target in "${targets[@]}"; do printf '%s\n' "${TARGET_ROLE[${target}]}"; done
    exit 0
fi
: "${CE_IMAGES:?set SCRATCH or CE_IMAGES}"
mkdir -p "${CE_IMAGES}"
# OUTPUT_SQSH names the output of a one-target build only: two targets cannot share one name.
output() {
    if [[ -n "${OUTPUT_SQSH:-}" && ${#targets[@]} -eq 1 ]]; then
        printf '%s' "${OUTPUT_SQSH}"
    else
        printf '%s/%s' "${CE_IMAGES}" "$(ce_image "${TARGET_ROLE[$1]}" candidate)"
    fi
}
local_tag() {
    if [[ "$1" == - ]]; then printf 'hpcagent-bench-ce-%s:latest' "${IMAGE}"; else printf 'hpcagent-bench-ce-%s-%s:latest' "${IMAGE}" "$1"; fi
}
dockerfile_target() { [[ "$1" == - ]] || printf '%s' "$1"; }

outputs=()
for target in "${targets[@]}"; do outputs+=("$(output "${target}")"); done
ce_refuse_mounted "${outputs[@]}"

ce_podman_env
ce_image_args
for kv in ${EXTRA_BUILD_ARGS:-}; do BUILD_ARGS+=(--build-arg "${kv}"); done

specs=()
for target in "${targets[@]}"; do
    specs+=("${TARGET_ROLE[${target}]}|$(dockerfile_target "${target}")|$(local_tag "${target}")|$(output "${target}")")
done
ce_pull_first "${IMAGE_DIR}/Dockerfile" "${specs[@]}" -- "${BUILD_ARGS[@]}"
if [[ "${CE_PULLED}" == 0 ]]; then
    ce_image_inputs
    for target in "${targets[@]}"; do
        ce_build "${IMAGE_DIR}/Dockerfile" "$(dockerfile_target "${target}")" "$(local_tag "${target}")" \
            "$(output "${target}")" "${INPUT_ARGS[@]}" "${BUILD_ARGS[@]}"
    done
fi

missing=0
for sqsh in "${outputs[@]}"; do
    if [[ ! -f "${sqsh}" ]]; then
        echo "MISSING: ${sqsh}" >&2
        missing=$((missing + 1))
        continue
    fi
    sha256sum "${sqsh}" | tee "${sqsh}.sha256"
    echo "IMAGE READY: ${sqsh}"
done
[[ "${missing}" -eq 0 ]] || { echo "${missing} image(s) missing" >&2; exit 2; }
