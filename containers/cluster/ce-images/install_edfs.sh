#!/usr/bin/env bash
# Render the EDFs a job reaches an image through, from the templates in this repo.
#
# Two jobs, and the second is why this is a script rather than a paragraph in the README:
#
#   1. A default name. `optarena-amd-mi300-latest` resolves to whatever images.env calls latest,
#      so a campaign stops hardcoding a version and a promotion is one edit instead of 169.
#   2. A fresh clone. ~/.edf is not in the repo, so a checkout on another account has no way to
#      reach any image. Copying a teammate's EDF carries their absolute scratch path into your
#      jobs; this renders yours from ${SCRATCH}.
#
# The template is expanded HERE rather than left to the container engine: the installed v6 EDF
# holds absolute paths, so ${SCRATCH} is not something the CE can be relied on to substitute.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=images.env
source "${SCRIPT_DIR}/images.env"

: "${SCRATCH:?set SCRATCH -- an EDF is absolute paths and there is nothing sane to guess}"
CE_IMAGES="${CE_IMAGES:-${SCRATCH}/ce-images}"
EDF_DIR="${EDF_DIR:-${HOME}/.edf}"
mkdir -p "${EDF_DIR}"

# Repointing latest changes which image every unpinned job gets, including one that is queued now
# and starts in an hour. That is the one edit here worth being loud about, so it is opt-in.
render() {
    local name="$1" template="$2" sqsh="$3" target="${EDF_DIR}/$1.toml" image="${CE_IMAGES}/$3"

    if [[ ! -f "${image}" ]]; then
        echo "refusing to write ${name}: ${image} does not exist" >&2
        echo "  build it, or pull it with pull_image.sh, before naming it in an EDF" >&2
        return 1
    fi

    if [[ -f "${target}" ]]; then
        local current
        current="$(sed -nE 's/^[[:space:]]*image[[:space:]]*=[[:space:]]*"(.*)"/\1/p' "${target}" | head -1)"
        if [[ "${current}" != "${image}" && -z "${ALLOW_REPOINT:-}" ]]; then
            echo "refusing to repoint ${name}" >&2
            echo "  from: ${current}" >&2
            echo "  to:   ${image}" >&2
            echo "  every unpinned job, including queued ones, would move. Re-run with ALLOW_REPOINT=1" >&2
            return 1
        fi
    fi

    sed -e "s|\${SCRATCH}|${SCRATCH}|g" \
        -e "s|^image = .*|image = \"${image}\"|" \
        "${SCRIPT_DIR}/${template}" > "${target}"
    printf '  %-32s -> %s\n' "${name}" "${image}"
}

# One missing image used to abort the whole run under set -e, so a promotion that had three of
# four images installed nothing and left the fourth name unexplained. Each render is now reported
# and the script exits non-zero at the end, so a partial install is visible rather than silent.
failed=0
try_render() { render "$@" || failed=$((failed + 1)); }

echo "installing EDFs into ${EDF_DIR}"
try_render "${JUDGE_AGENT_AMD_EDF}"        "${JUDGE_AGENT_AMD_TEMPLATE}" "${JUDGE_AGENT_AMD_SQSH}"
try_render "${JUDGE_AGENT_AMD_EDF_LATEST}" "${JUDGE_AGENT_AMD_TEMPLATE}" "${JUDGE_AGENT_AMD_SQSH}"

# The inference pair. Their -latest aliases exist for the same reason the judge one does: a
# rebuild should be reachable by re-rendering, not by editing every campaign that names it. The
# version-named EDFs are left exactly as they are, so a run that must not move does not.
try_render "${INFERENCE_SGLANG_EDF_LATEST}" "${INFERENCE_SGLANG_TEMPLATE}" "${INFERENCE_SGLANG_SQSH}"
try_render "${INFERENCE_VLLM_EDF_LATEST}"   "${INFERENCE_VLLM_TEMPLATE}"   "${INFERENCE_VLLM_SQSH}"
try_render "${INFERENCE_VLLM_0271_EDF_LATEST}" "${INFERENCE_VLLM_0271_TEMPLATE}" "${INFERENCE_VLLM_0271_SQSH}"

echo
if [[ ${failed} -gt 0 ]]; then
    echo "${failed} EDF(s) NOT installed -- the names above still point wherever they did" >&2
fi
echo "follow images.env:  AMD_CE_ENV=${JUDGE_AGENT_AMD_EDF_LATEST}"
echo "                    INFERENCE_CE_ENV=${INFERENCE_SGLANG_EDF_LATEST} (or ${INFERENCE_VLLM_EDF_LATEST})"
echo "pin this run:       AMD_CE_ENV=${JUDGE_AGENT_AMD_EDF}"

exit $(( failed > 0 ))
