#!/usr/bin/env bash
# Render ~/.edf/<edf>.toml for every images.env row of CE_PLATFORM from its
# template, with this account's ${SCRATCH}, mount list, GPU arch and live squashfs filled in.
#
#   containers/images/install_edfs.sh                     # this host's platform (build_common.sh ce_platform)
#   CE_PLATFORM=gh200 containers/images/install_edfs.sh   # or name it: amd, gh200, nvidia-x86, cpu
#   containers/images/install_edfs.sh <role>...           # only these rows (registry.sh promote)
#   CE_IMAGE_RELEASE=v10 containers/images/install_edfs.sh   # freeze the live images as release v10 (images.env)
#
# Refuses to point an existing EDF at a different image unless ALLOW_REPOINT=1: every job that
# starts afterwards, queued ones included, would move. `registry.sh promote` sets it.
set -Eeuo pipefail

ulimit -c 0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=images.env
source "${SCRIPT_DIR}/images.env"

: "${SCRATCH:?set SCRATCH -- an EDF is absolute paths and there is nothing sane to guess}"
: "${CE_IMAGES:?set SCRATCH or CE_IMAGES}"
# shellcheck source=../../helpers/scripts/site_env.sh
. "${SCRIPT_DIR}/../../helpers/scripts/site_env.sh"
EDF_DIR="${EDF_DIR:-${EDF_PATH%%:*}}"
mkdir -p "${EDF_DIR}"

# shellcheck source=build_common.sh
source "${SCRIPT_DIR}/build_common.sh"

# render <edf name> <template> <sqsh> <partition|-> <preload|empty>
render() {
    local name="$1" template="$2" sqsh="$3" partition="$4" preload="$5"
    local target="${EDF_DIR}/$1.toml" image="${CE_IMAGES}/$3"
    if [[ ! -f "${image}" ]]; then
        echo "refusing to write ${name}: ${image} does not exist" >&2
        echo "  pull it:  sbatch ${SCRIPT_DIR}/registry.sbatch pull" >&2
        echo "  or build: containers/README.md" >&2
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
    ce_render_edf "${template}" "${image}" "${partition}" "${preload}" > "${target}.tmp"
    mv -f "${target}.tmp" "${target}"
    printf '  %-40s -> %s\n' "${name}" "${image}"
}

# freeze <live sqsh>: prints the release's <stem>-<release>.sqsh, a hard link of the live image made the first time
# (no copy, and a later promote or pull leaves it on the old inode), its sidecars copied: a pull rewrites the live
# .sha256 in place.
freeze() {
    local live="${CE_IMAGES}/$1" frozen="${CE_IMAGES}/${1%.sqsh}-${CE_IMAGE_RELEASE}.sqsh" ext
    if [[ ! -e "${frozen}" ]]; then
        [[ -f "${live}" ]] || { echo "refusing to freeze ${frozen##*/}: ${live} does not exist" >&2; return 1; }
        for ext in .digest .sha256; do
            [[ ! -f "${live}${ext}" ]] || cp -p -- "${live}${ext}" "${frozen}${ext}"
        done
        ln -- "${live}" "${frozen}"
    fi
    printf '%s\n' "${frozen##*/}"
}

CE_PLATFORM="$(ce_platform)"
case "${CE_PLATFORM}" in
    amd|gh200|nvidia-x86|cpu) ;;
    *) echo "CE_PLATFORM must be amd, gh200, nvidia-x86 or cpu, got '${CE_PLATFORM}'" >&2; exit 2 ;;
esac
echo "installing ${CE_PLATFORM} EDFs into ${EDF_DIR}"

failed=0
for role in ${*:-$(ce_roles "${CE_PLATFORM}")}; do
    edf="$(ce_image "${role}" edf)" || continue
    sqsh="$(ce_image "${role}" sqsh)"
    flags="$(ce_image "${role}" flags || true)"
    # An optional image (mi200) renders only when it is there.
    if [[ ",${flags}," == *,opt,* && ! -f "${CE_IMAGES}/${sqsh}" ]]; then
        continue
    fi
    if [[ -n "${CE_IMAGE_RELEASE}" ]]; then
        edf+="-${CE_IMAGE_RELEASE}"
        if [[ -f "${EDF_DIR}/${edf}.toml" ]]; then
            printf '  %-40s kept: a release EDF is never rewritten\n' "${edf}"
            continue
        fi
        sqsh="$(freeze "${sqsh}")" || { failed=$((failed + 1)); continue; }
    fi
    preload=""
    [[ ",${flags}," =~ ,preload=([^,]+), ]] && preload="${BASH_REMATCH[1]}"
    render "${edf}" "$(ce_image "${role}" template)" "${sqsh}" \
        "$(ce_image "${role}" partition || echo -)" "${preload}" || failed=$((failed + 1))
done

if [[ ${failed} -gt 0 ]]; then
    echo "${failed} EDF(s) NOT installed -- the names above still point wherever they did" >&2
fi
exit $(( failed > 0 ))
