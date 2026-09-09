#!/usr/bin/env bash
# Promote a verified CANDIDATE image to the live name its EDFs resolve to.
#
#   ./promote_image.sh judge-agent-amd            # one role
#   ./promote_image.sh --all                      # every role that has a candidate
#   DRY_RUN=1 ./promote_image.sh --all            # say what would move, touch nothing
#
# Promotion was prose in images.env: "rename it over the live name, then re-run install_edfs.sh
# with ALLOW_REPOINT=1". Four roles times four files each is not a thing to do by hand, and the
# sidecars are the part that gets forgotten -- with one version per role the .digest is the ONLY
# record of which build a name currently holds, so a rename that leaves it behind makes the live
# image unattributable.
#
# The rename is safe while arms are running: a mounted squashfs is held by its inode, so a job
# that already started keeps reading the bytes it opened and only new jobs see the new image.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=images.env
source "${SCRIPT_DIR}/images.env"
CE="${CE_IMAGES:-${SCRATCH:?set SCRATCH}/ce-images}"
DRY_RUN="${DRY_RUN:-0}"

# The candidate each role builds to, matching build.sbatch's OUTPUT_SQSH defaults.
role_candidate() {
    case "$1" in
        judge-agent-amd) printf 'optarena-ce-amd-mi300-candidate.sqsh' ;;
        judge)           printf 'optarena-ce-judge-amd-mi300-candidate.sqsh' ;;
        sglang)          printf 'optarena-sglang-candidate.sqsh' ;;
        vllm)            printf 'optarena-vllm-candidate.sqsh' ;;
        *) return 2 ;;
    esac
}
role_live() {
    case "$1" in
        judge-agent-amd) printf '%s' "${JUDGE_AGENT_AMD_SQSH}" ;;
        judge)           printf '%s' "${JUDGE_AMD_SQSH}" ;;
        sglang)          printf '%s' "${INFERENCE_SGLANG_SQSH}" ;;
        vllm)            printf '%s' "${INFERENCE_VLLM_SQSH}" ;;
        *) return 2 ;;
    esac
}

ALL_ROLES="judge-agent-amd judge sglang vllm"
case "${1:-}" in
    --all) roles="${ALL_ROLES}" ;;
    "")    echo "usage: $0 <role>... | --all   (roles: ${ALL_ROLES})" >&2; exit 2 ;;
    *)     roles="$*" ;;
esac

# A candidate is promotable only if it was VERIFIED. build_and_verify.sbatch is the only path that
# runs the verifier, and it writes this marker next to the image on a clean verdict; a bare
# build.sbatch run does not. Without the marker this refuses, because "built" and "works" have
# been different things often enough here to cost whole campaigns.
verified_marker() { printf '%s.verified' "$1"; }

EDF_DIR="${EDF_DIR:-${HOME}/.edf}"

# Repoint every EDF that named the image we just renamed.
#
# install_edfs.sh (run at the end) re-renders the four MANAGED names from their templates, and for
# a long time that was assumed to be all of ~/.edf. It is not. An EDF written by hand is managed by
# nothing, so a promotion renames the image out from under it and leaves it pointing at a path that
# no longer exists -- the arm then dies at container start with nothing but "image does not exist",
# a long way from the rename that caused it. sglang-glm-halfconv is one such file and there is no
# reason to believe it is the last, so this keys off the RENAME rather than off a list of names:
# whatever pointed at the old path is what has to move.
repoint_edfs() {
    local from="$1" to="$2" edf current n=0
    for edf in "${EDF_DIR}"/*.toml; do
        [ -f "${edf}" ] || continue
        current="$(sed -nE 's/^[[:space:]]*image[[:space:]]*=[[:space:]]*"(.*)"/\1/p' "${edf}" | head -1)"
        [ "${current}" = "${from}" ] || continue
        # Anchored on the image line only; an EDF is mostly comments and env, and a looser
        # substitution would rewrite prose that happens to quote the path.
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
        # The judge is a second TARGET of the judge-agent-amd build, not a directory of its own.
        dir="${role}"; [ "${role}" = "judge" ] && dir="judge-agent-amd"
        echo "${role}: REFUSING -- ${cand##*/} carries no .verified marker" >&2
        echo "  run: IMAGE_DIR=${SCRIPT_DIR}/${dir} sbatch build_and_verify.sbatch" >&2
        failed=$((failed + 1))
        continue
    fi
    # A marker records the DIGEST it verified. Existence alone is not enough: a build that
    # overwrites an image in place leaves the OLD marker beside NEW bytes, and promoting on that
    # is promoting something nothing ever verified. build.sbatch (unlike build_and_verify.sbatch)
    # does exactly that. Compare, and refuse when they disagree.
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
    # The sidecars move WITH the image, or the live name loses its provenance. .oci.tar is the
    # only publishable form -- a squashfs reimports as one layer past the registry ceiling -- so
    # a promotion that drops it makes the role unpublishable without a rebuild.
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
    # The marker does not follow: it describes a candidate that was verified, and the live name
    # having one would make the next promotion's refusal check meaningless.
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
