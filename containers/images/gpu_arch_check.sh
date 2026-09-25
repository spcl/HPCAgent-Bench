#!/usr/bin/env bash
# Fails a job whose AMD GPU arch disagrees across its three sources: gpu_arch.env for this job's
# partition, the image's /opt/gpu-arch stamp, and the first GPU agent rocminfo reports in the image.
#
#   gpu_arch_check.sh <edf name or path>        (inside a Slurm allocation)
#
# Exit 2 on a mismatch, printing all three. An image built before the stamp existed has no
# /opt/gpu-arch: that WARNS and exits 0, so campaigns on those images keep launching.
set -euo pipefail
ulimit -c 0

# shellcheck source=build_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/build_common.sh"

in_image() {  # in_image <edf> <command...>: one task on one node of this allocation
    srun --overlap --nodes=1 --ntasks=1 --environment="$1" "${@:2}"
}

main() {
    local edf="${1:?usage: gpu_arch_check.sh <edf name or path>}" table stamp info probe
    table="$(ce_partition_arch "${SLURM_JOB_PARTITION:?gpu_arch_check.sh runs inside a Slurm allocation}")"
    stamp="$(in_image "${edf}" sh -c 'cat /opt/gpu-arch 2>/dev/null || true')"
    if [[ -z "${stamp}" ]]; then
        echo "gpu_arch_check: WARNING: ${edf} has no /opt/gpu-arch (image predates the stamp); arch not checked" >&2
        return 0
    fi
    info="$(in_image "${edf}" /opt/rocm/bin/rocminfo)" || { echo "gpu_arch_check: rocminfo failed in ${edf}" >&2; exit 2; }
    probe="$(sed -nE 's/^[[:space:]]*Name:[[:space:]]+(gfx[0-9a-f]+)[[:space:]]*$/\1/p' <<< "${info}" | sed -n 1p)"
    if [[ "${table}" == "${stamp}" && "${stamp}" == "${probe}" ]]; then
        printf 'gpu_arch_check: %s on %s: %s\n' "${edf}" "${SLURM_JOB_PARTITION}" "${table}"
        return 0
    fi
    printf 'gpu_arch_check: GPU arch mismatch for %s\n  gpu_arch.env[%s] = %s\n  /opt/gpu-arch = %s\n  rocminfo = %s\n' \
        "${edf}" "${SLURM_JOB_PARTITION}" "${table}" "${stamp}" "${probe:-none}" >&2
    exit 2
}

main "$@"
