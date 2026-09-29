#!/usr/bin/env bash
# Move images between this machine and the registry, and make a verified candidate live.
#
#   registry.sh promote <role>... | --all     rename each role's verified candidate over its live name
#   registry.sh push --check <role>... | --all  preflight only: sizes, flavor, tag (nothing published)
#   registry.sh push <role>... | --all        publish each role's live OCI archive as REGISTRY_REPO:<tag>
#   registry.sh pull <role> [sha256:<digest>] import the role's tag (or a pinned digest) as its squashfs
#
# Roles, tags and file names come from images.env; --all means every role of CE_PLATFORM (amd, gh200,
# cpu; default amd) that has what the action needs. A push reads the OCI archive the build saved beside
# the squashfs (ce_export_image): a squashfs is flattened and cannot stand in for one. Credentials come
# only from the environment (REGISTRY_USER, REGISTRY_TOKEN), never a file. Only the portable latest
# flavor is published: a native build or a tag not ending in -latest is refused. registry.sbatch runs
# this on a compute node.
set -Eeuo pipefail
ulimit -c 0
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=build_common.sh
source "${HERE}/build_common.sh"
MAX_LAYER_GB="${MAX_LAYER_GB:-10}"
MAX_IMAGE_GB="${MAX_IMAGE_GB:-100}"

usage() { sed -n '4,7p' "${BASH_SOURCE[0]}" >&2; exit 2; }

# roles_for <column> <args>: the roles named, or with --all every CE_PLATFORM role with that column.
roles_for() {
    local column="$1" role
    shift
    if [[ "${1:-}" == --all ]]; then
        for role in $(ce_roles "${CE_PLATFORM:-amd}"); do
            ce_image "${role}" "${column}" >/dev/null 2>&1 && printf '%s\n' "${role}"
        done
    else
        [[ $# -gt 0 ]] || usage
        printf '%s\n' "$@"
    fi
}

gb() { awk -v b="$1" 'BEGIN { printf "%.1f", b / 1073741824 }'; }

# push_one <role> <check 0|1>: load the role's archive into a private store, check it, push it.
push_one() {
    local role="$1" check="$2" tag archive root local_tag total biggest target
    : "${CE_IMAGES:?set SCRATCH or CE_IMAGES}"
    tag="$(ce_image "${role}" tag)" || { echo "${role}: images.env gives it no registry tag" >&2; return 1; }
    [[ "${tag}" == *-latest ]] || { echo "${role}: a published tag is <thing>-latest, got ${tag}" >&2; return 1; }
    archive="${CE_IMAGES}/$(ce_image "${role}" sqsh)"
    archive="${archive%.sqsh}.oci.tar"
    [[ -f "${archive}" ]] || { echo "${role}: no ${archive}; build and promote the role first" >&2; return 1; }
    root="${CE_TMPFS}/registry-$$-${role}"
    mkdir -p "${root}/root" "${root}/run"
    local -a pm=(podman --root "${root}/root" --runroot "${root}/run" --storage-driver overlay
                 --storage-opt ignore_chown_errors=true)
    local_tag="$("${pm[@]}" pull "oci-archive:${archive}" | tail -1)"
    total="$("${pm[@]}" image inspect --format '{{.Size}}' "${local_tag}")"
    # The largest layer as the registry sees it: compressed, from the archive's own manifest.
    biggest="$(python3 - "${archive}" <<'PY'
import json, sys, tarfile
with tarfile.open(sys.argv[1]) as tar:
    index = json.load(tar.extractfile("index.json"))
    digest = index["manifests"][0]["digest"].split(":", 1)[1]
    manifest = json.load(tar.extractfile(f"blobs/sha256/{digest}"))
    print(max((layer["size"] for layer in manifest.get("layers", [])), default=0))
PY
)"
    printf '%s -> %s:%s\n  image %s GB (limit %s), largest layer %s GB (limit %s), digest %s\n' "${role}" \
        "${REGISTRY_REPO}" "${tag}" "$(gb "${total}")" "${MAX_IMAGE_GB}" "$(gb "${biggest}")" "${MAX_LAYER_GB}" \
        "$("${pm[@]}" image inspect --format '{{.Digest}}' "${local_tag}")"
    local rc=0
    if awk -v b="${total}" -v m="${MAX_IMAGE_GB}" 'BEGIN { exit !(b > m * 1073741824) }'; then
        echo "  refusing: the image exceeds ${MAX_IMAGE_GB} GB" >&2; rc=1
    fi
    if awk -v b="${biggest}" -v m="${MAX_LAYER_GB}" 'BEGIN { exit !(b > m * 1073741824) }'; then
        echo "  refusing: a layer exceeds ${MAX_LAYER_GB} GB, which the registry rejects mid-upload" >&2; rc=1
    fi
    if [[ "$("${pm[@]}" image inspect --format '{{ index .Labels "org.hpcagent-bench.cpu-target" }}' "${local_tag}")" == native ]]; then
        echo "  refusing: a native build; publish only the portable baseline" >&2; rc=1
    fi
    if [[ "${rc}" -eq 0 && "${check}" == 0 ]]; then
        target="${REGISTRY_REPO}:${tag}"
        if [[ -n "${REGISTRY_USER:-}" && -n "${REGISTRY_TOKEN:-}" ]]; then
            printf '%s' "${REGISTRY_TOKEN}" | "${pm[@]}" login --username "${REGISTRY_USER}" --password-stdin \
                "${REGISTRY_REPO%%/*}"
        fi
        "${pm[@]}" tag "${local_tag}" "${target}"
        # OCI manifests, forced: podman would otherwise keep the source's docker v2s2 type.
        "${pm[@]}" push --format oci "${target}" || rc=1
        [[ "${rc}" -ne 0 ]] || echo "  PUSHED ${target}"
    fi
    podman unshare rm -rf "${root}" 2>/dev/null || rm -rf "${root}"
    return "${rc}"
}

# pull_one <role> [digest]: enroot-import the role's registry image as its live squashfs (or OUT).
pull_one() {
    local role="$1" ref out
    [[ "${CE_IMAGE_FLAVOR}" == latest ]] || { echo "the registry has latest images only; build a native one" >&2; return 2; }
    : "${CE_IMAGES:?set SCRATCH or CE_IMAGES}"
    if [[ -n "${2:-}" ]]; then
        [[ "$2" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "pin a digest as sha256:<64 hex>, got '$2'" >&2; return 2; }
        ref="${REGISTRY_REPO}@$2"
    else
        ref="${REGISTRY_REPO}:$(ce_image "${role}" tag)" || { echo "${role} has no published tag" >&2; return 2; }
    fi
    out="${OUT:-${CE_IMAGES}/$(ce_image "${role}" sqsh)}"
    mkdir -p "${CE_IMAGES}"
    ce_refuse_mounted "${out}"
    export ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH:-${CE_TMPFS}/enroot-tmp}"
    export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-${SCRATCH:?}/.enroot}"
    mkdir -p "${ENROOT_TEMP_PATH}" "${ENROOT_CACHE_PATH}"
    echo "pulling ${ref} -> ${out}"
    rm -f "${out}"
    enroot import -x mount -o "${out}" "docker://${ref}"
    sha256sum "${out}" | tee "${out}.sha256"
    echo "PULLED ${out}; verify it (verify_image.sbatch) before a campaign mounts it"
}

# promote_one <role>: rename a verified candidate (and its sidecars and archive) over the live name.
promote_one() {
    local role="$1" cand live marker ext
    : "${CE_IMAGES:?set SCRATCH or CE_IMAGES}"
    cand="${CE_IMAGES}/$(ce_image "${role}" candidate)"
    live="${CE_IMAGES}/$(ce_image "${role}" sqsh)"
    [[ -f "${cand}" ]] || { echo "${role}: no candidate at ${cand}"; return 0; }
    marker="${cand}.verified"
    if [[ ! -f "${marker}" ]]; then
        echo "${role}: REFUSING, ${cand##*/} carries no .verified marker (build_and_verify.sbatch)" >&2
        return 1
    fi
    if [[ -f "${cand}.digest" && "$(grep -oE 'digest=[^[:space:]]+' "${marker}" | cut -d= -f2)" != "$(cat "${cand}.digest")" ]]; then
        echo "${role}: REFUSING, ${cand##*/} was rebuilt after it was verified; re-verify it" >&2
        return 1
    fi
    ce_refuse_mounted "${live}" || return 1
    printf '%s: %s -> %s\n' "${role}" "${cand##*/}" "${live##*/}"
    for ext in "" .digest .sha256; do
        [[ ! -e "${cand}${ext}" ]] || mv -f -- "${cand}${ext}" "${live}${ext}"
    done
    if [[ -e "${cand%.sqsh}.oci.tar" ]]; then
        mv -f -- "${cand%.sqsh}.oci.tar" "${live%.sqsh}.oci.tar"
    else
        echo "  WARNING: no OCI archive; this role cannot be pushed without a rebuild" >&2
    fi
    rm -f -- "${marker}"
    PROMOTED+=("${live##*/}")
}

action="${1:-}"
[[ $# -gt 0 ]] && shift
failed=0
case "${action}" in
    promote)
        PROMOTED=()
        for role in $(roles_for candidate "$@"); do promote_one "${role}" || failed=$((failed + 1)); done
        # Re-render the EDFs of every row that mounts a promoted image (EDF-only views share its sqsh).
        edf_roles=()
        for role in $(ce_roles); do
            sqsh="$(ce_image "${role}" sqsh 2>/dev/null)" && ce_image "${role}" edf >/dev/null 2>&1 || continue
            [[ " ${PROMOTED[*]} " != *" ${sqsh} "* ]] || edf_roles+=("${role}")
        done
        if [[ ${#edf_roles[@]} -gt 0 ]]; then
            ALLOW_REPOINT=1 "${HERE}/install_edfs.sh" "${edf_roles[@]}" || failed=$((failed + 1))
        fi
        ;;
    push)
        check=0
        [[ "${1:-}" != --check ]] || { check=1; shift; }
        for role in $(roles_for tag "$@"); do push_one "${role}" "${check}" || failed=$((failed + 1)); done
        ;;
    pull)
        pull_one "${1:?usage: registry.sh pull <role> [sha256:<digest>]}" "${2:-}" || failed=1
        ;;
    *) usage ;;
esac
[[ "${failed}" -eq 0 ]] || { echo "${failed} failure(s)" >&2; exit 1; }
