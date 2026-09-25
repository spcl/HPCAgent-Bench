#!/usr/bin/env bash
# Resolve what the CSCS container hooks WILL use for the network stack, assert every piece is
# there and is the version this campaign is pinned to, and print the provenance. Run it on the
# HOST before srun; it inspects host paths, not the container.
#
#   . scripts/cscs/netstack_preflight.sh   # or run it directly
#
# The three hooks (10-netstack, 89-libfabric-cxi, 90-aws-ofi-nccl) default to
#     netstack_src=${OCI_ANNOTATION_com__hooks__netstack__source:-artifact}
# which resolves under ${HPCAGENT_BENCH_NETSTACK_BASE}/$(uname -m)/<ver>/<name> (the site layer
# names the base; no base means the site has no such hooks and the check is skipped). "host" mode's
# only plugin (variant=rocm6) needs libamdhip64.so.6, which the ROCm-7.2 images do not ship, so RCCL
# falls back to sockets. The artifact bundle carries its own libamdhip64.so.7, and is pinned by
# version+name so a site-side bump is a loud failure here instead of a campaign that silently
# straddles two fabric stacks.
#
# Override for a deliberate move: HPCAGENT_BENCH_NETSTACK_VERSION / _NAME, or "any" to accept
# whatever exists under the pinned version.
set -uo pipefail

. "$(dirname -- "${BASH_SOURCE[0]}")/../site_env.sh" || return 1 2>/dev/null || exit 1

# A dump lands in the crashing process's CWD (the checkout); Slurm propagates the submitter's limit.
ulimit -c 0
: "${HPCAGENT_BENCH_NETSTACK_SOURCE:=artifact}"  # must match the EDF annotation
: "${HPCAGENT_BENCH_NETSTACK_VERSION:=26.08.1}"
: "${HPCAGENT_BENCH_NETSTACK_NAME:=gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0}"
: "${HPCAGENT_BENCH_NETSTACK_BASE:=}"

netstack_preflight() {
    local rc=0 base lib plugin

    if [ "${HPCAGENT_BENCH_NETSTACK_SOURCE}" != "artifact" ]; then
        echo "netstack: source=${HPCAGENT_BENCH_NETSTACK_SOURCE} (not 'artifact')" >&2
        echo "  host mode's only installed plugin (variant=rocm6) needs libamdhip64.so.6, which" >&2
        echo "  this image does not ship (ROCm 7.2 -> .so.7 only). Set" >&2
        echo "  com.hooks.netstack.source=\"artifact\" unless the image is ROCm 6.x." >&2
        return 1
    fi
    if [ -z "${HPCAGENT_BENCH_NETSTACK_BASE}" ]; then
        echo "netstack: HPCAGENT_BENCH_NETSTACK_BASE unset (site layer); fabric check skipped" >&2
        return 0
    fi

    base="${HPCAGENT_BENCH_NETSTACK_BASE}/$(uname -m)/${HPCAGENT_BENCH_NETSTACK_VERSION}"
    if [ "${HPCAGENT_BENCH_NETSTACK_NAME}" = "any" ]; then
        HPCAGENT_BENCH_NETSTACK_NAME="$(cd -- "${base}" 2>/dev/null && ls -d */ 2>/dev/null | head -1 | tr -d /)"
    fi
    plugin="${base}/${HPCAGENT_BENCH_NETSTACK_NAME}/librccl-net.so"
    lib="${base}/${HPCAGENT_BENCH_NETSTACK_NAME}/libfabric.so.1"

    if [ ! -d "${base}" ]; then
        echo "netstack: MISSING version dir ${base}" >&2
        echo "  available: $(ls -d "${HPCAGENT_BENCH_NETSTACK_BASE}/$(uname -m)"/*/ 2>/dev/null | xargs -n1 basename | tr '\n' ' ')" >&2
        return 1
    fi
    [ -e "${lib}" ]    || { echo "netstack: MISSING libfabric ${lib}" >&2; rc=1; }
    [ -e "${plugin}" ] || {
        echo "netstack: MISSING RCCL plugin ${plugin}" >&2
        echo "  available names: $(ls "${base}" 2>/dev/null | tr '\n' ' ')" >&2
        rc=1; }

    # Provenance: what a results table should be able to cite.
    printf 'netstack: source=artifact version=%s name=%s cxi_nics=%s\n' \
        "${HPCAGENT_BENCH_NETSTACK_VERSION}" "${HPCAGENT_BENCH_NETSTACK_NAME}" \
        "$(ls -d /dev/cxi[0-9]* 2>/dev/null | wc -l)"
    return "${rc}"
}

# Executed directly -> act as a gate. Sourced -> just define the function.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    netstack_preflight || exit 1
fi
