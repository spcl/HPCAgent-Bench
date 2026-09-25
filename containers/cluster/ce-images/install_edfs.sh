#!/usr/bin/env bash
# Render ~/.edf/<edf>.toml for every images.env row of CE_PLATFORM (amd, gh200 or cpu) from its
# template, with this account's ${SCRATCH}, mount list, GPU arch and live squashfs filled in.
#
#   containers/cluster/ce-images/install_edfs.sh                     # beverin (amd)
#   CE_PLATFORM=gh200 containers/cluster/ce-images/install_edfs.sh   # daint
#   CE_PLATFORM=cpu   containers/cluster/ce-images/install_edfs.sh   # any host, CPU only
#
# Refuses to point an existing EDF at a different image unless ALLOW_REPOINT=1: every job that
# starts afterwards, queued ones included, would move. promote_image.sh sets it.
set -Eeuo pipefail

ulimit -c 0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=images.env
source "${SCRIPT_DIR}/images.env"
. "${SCRIPT_DIR}/../../../scripts/cache_env.sh"
EDF_MOUNTS="$(hpcagent_bench_edf_mounts)"

: "${SCRATCH:?set SCRATCH -- an EDF is absolute paths and there is nothing sane to guess}"
CE_IMAGES="${CE_IMAGES:-${SCRATCH}/ce-images}"
EDF_DIR="${EDF_DIR:-${HOME}/.edf}"
mkdir -p "${EDF_DIR}"

# shellcheck source=build_common.sh
source "${SCRIPT_DIR}/build_common.sh"

# render <edf name> <template> <sqsh> <partition|-> <preload|empty>
render() {
    local name="$1" template="$2" sqsh="$3" partition="$4" preload="$5"
    local target="${EDF_DIR}/$1.toml" image="${CE_IMAGES}/$3"
    if [[ ! -f "${image}" ]]; then
        echo "refusing to write ${name}: ${image} does not exist" >&2
        echo "  pull it:  sbatch ${SCRIPT_DIR}/pull_images.sbatch" >&2
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
    local arch=""
    if grep -qF '${GPU_ARCH}' "${SCRIPT_DIR}/${template}"; then
        arch="$(ce_partition_arch "${partition}")" || return 1
    fi
    local preload_edit=()
    if [[ -n "${preload}" ]]; then
        if [[ "$(grep -c '^LD_PRELOAD = "[^"]*"$' "${SCRIPT_DIR}/${template}")" != 1 ]]; then
            echo "refusing to write ${name}: ${template} has no single LD_PRELOAD line to extend" >&2
            return 1
        fi
        preload_edit=(-e "s|^LD_PRELOAD = \"\(.*\)\"$|LD_PRELOAD = \"\1:${preload}\"|")
    fi
    sed -e "s|\${SCRATCH}|${SCRATCH}|g" \
        -e "s|\"<hpcagent_bench_edf_mounts>\"|${EDF_MOUNTS}|" \
        -e "s|\${GPU_ARCH}|${arch}|g" \
        -e "s|^image = .*|image = \"${image}\"|" \
        "${preload_edit[@]}" \
        "${SCRIPT_DIR}/${template}" > "${target}"
    printf '  %-40s -> %s\n' "${name}" "${image}"
}

CE_PLATFORM="${CE_PLATFORM:-amd}"
case "${CE_PLATFORM}" in
    amd|gh200|cpu) ;;
    *) echo "CE_PLATFORM must be amd, gh200 or cpu, got '${CE_PLATFORM}'" >&2; exit 2 ;;
esac
echo "installing ${CE_PLATFORM} EDFs into ${EDF_DIR}"

failed=0
for role in $(ce_roles "${CE_PLATFORM}"); do
    edf="$(ce_image "${role}" edf)" || continue
    sqsh="$(ce_image "${role}" sqsh)"
    flags="$(ce_image "${role}" flags || true)"
    # An optional image (mi200) renders only when it is there.
    if [[ ",${flags}," == *,opt,* && ! -f "${CE_IMAGES}/${sqsh}" ]]; then
        continue
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
