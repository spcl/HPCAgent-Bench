#!/usr/bin/env bash
# Promote a verified candidate (images.env `candidate`) to its live name (`sqsh`), carry the
# .digest/.sha256/.oci.tar sidecars with it, repoint every EDF that named the candidate, and re-run
# install_edfs.sh. Safe while jobs run: a mounted squashfs is held by its inode.
#
#   ./promote_image.sh judge-agent-amd            # one role
#   ./promote_image.sh --all                      # every role of CE_PLATFORM (amd, gh200, cpu)
#   DRY_RUN=1 ./promote_image.sh --all            # say what would move, touch nothing
set -Eeuo pipefail

ulimit -c 0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=images.env
source "${SCRIPT_DIR}/images.env"
# shellcheck source=build_common.sh
source "${SCRIPT_DIR}/build_common.sh"
CE="${CE_IMAGES:-${SCRATCH:?set SCRATCH}/ce-images}"
DRY_RUN="${DRY_RUN:-0}"

role_candidate() {
    ce_image "$1" candidate
}
role_live() {
    ce_image "$1" sqsh
}

ALL_ROLES=""
for role in $(ce_roles "${CE_PLATFORM:-amd}"); do
    role_candidate "${role}" >/dev/null 2>&1 && ALL_ROLES="${ALL_ROLES:+${ALL_ROLES} }${role}"
done
case "${1:-}" in
    --all) roles="${ALL_ROLES}" ;;
    "")    echo "usage: $0 <role>... | --all   (roles: ${ALL_ROLES})" >&2; exit 2 ;;
    *)     roles="$*" ;;
esac

# Only a clean verify writes this marker, recording the digest it verified.
verified_marker() { printf '%s.verified' "$1"; }

EDF_DIR="${EDF_DIR:-${HOME}/.edf}"

# Repoint every EDF (hand-written ones included) whose image line names the renamed candidate.
repoint_edfs() {
    local from="$1" to="$2" edf current n=0
    for edf in "${EDF_DIR}"/*.toml; do
        [ -f "${edf}" ] || continue
        current="$(sed -nE 's/^[[:space:]]*image[[:space:]]*=[[:space:]]*"(.*)"/\1/p' "${edf}" | head -1)"
        [ "${current}" = "${from}" ] || continue
        [ "${DRY_RUN}" = "1" ] || sed -i -E "s|^([[:space:]]*image[[:space:]]*=[[:space:]]*)\".*\"|\1\"${to}\"|" "${edf}"
        printf '    repoint %s\n' "${edf##*/}"
        n=$((n + 1))
    done
    [ "${n}" -gt 0 ] || printf '    (no unmanaged EDF named %s)\n' "${from##*/}"
}

failed=0
moved=0
for role in ${roles}; do
    cand="${CE}/$(role_candidate "${role}")"
    live="${CE}/$(role_live "${role}")"
    if [ ! -f "${cand}" ]; then
        echo "${role}: no candidate at ${cand} -- nothing to promote"
        continue
    fi
    if [ ! -f "$(verified_marker "${cand}")" ]; then
        dir="$(ce_image "${role}" dir)"
        echo "${role}: REFUSING -- ${cand##*/} carries no .verified marker" >&2
        case "$(ce_image "${role}" platform)" in
            amd) echo "  run: IMAGE_DIR=${SCRIPT_DIR}/${dir} sbatch build_and_verify.sbatch" >&2 ;;
            *)   echo "  run: IMAGE_DIR=${SCRIPT_DIR}/${dir} sbatch ${SCRIPT_DIR}/${dir}/build.sbatch" >&2 ;;
        esac
        failed=$((failed + 1))
        continue
    fi
    # A rebuild in place leaves the old marker beside new bytes: compare the verified digest.
    marker_digest="$(grep -oE 'digest=[^[:space:]]+' "$(verified_marker "${cand}")" | cut -d= -f2)"
    image_digest="$(cat "${cand}.digest" 2>/dev/null || true)"
    if [ -n "${image_digest}" ] && [ "${marker_digest}" != "${image_digest}" ]; then
        echo "${role}: REFUSING -- ${cand##*/} was rebuilt after it was verified" >&2
        echo "  marker verified: ${marker_digest}" >&2
        echo "  image is now:    ${image_digest}" >&2
        echo "  re-verify it, do not promote on a stale pass" >&2
        failed=$((failed + 1))
        continue
    fi
    printf '%s\n  %s\n  -> %s\n' "${role}" "${cand##*/}" "${live##*/}"
    # The sidecars are the live name's provenance; .oci.tar is its only publishable form.
    for ext in "" .digest .sha256; do
        [ -e "${cand}${ext}" ] || continue
        [ "${DRY_RUN}" = "1" ] || mv -f -- "${cand}${ext}" "${live}${ext}"
        printf '    %s\n' "$(basename "${cand}${ext}")"
    done
    cand_tar="${cand%.sqsh}.oci.tar"
    live_tar="${live%.sqsh}.oci.tar"
    if [ -e "${cand_tar}" ]; then
        [ "${DRY_RUN}" = "1" ] || mv -f -- "${cand_tar}" "${live_tar}"
        printf '    %s\n' "$(basename "${cand_tar}")"
    else
        echo "    WARNING: no ${cand_tar##*/} -- this role cannot be PUBLISHED without a rebuild" >&2
    fi
    # The marker stays behind: it certifies a candidate, never a live name.
    [ "${DRY_RUN}" = "1" ] || rm -f -- "$(verified_marker "${cand}")"
    repoint_edfs "${cand}" "${live}"
    moved=$((moved + 1))
done

if [ "${DRY_RUN}" = "1" ]; then
    echo; echo "DRY RUN -- nothing moved"
    exit "$(( failed > 0 ))"
fi

if [ "${moved}" -gt 0 ]; then
    echo
    echo "repointing EDFs"
    ALLOW_REPOINT=1 "${SCRIPT_DIR}/install_edfs.sh"
fi
[ "${failed}" -eq 0 ] || { echo "${failed} role(s) NOT promoted" >&2; exit 1; }
