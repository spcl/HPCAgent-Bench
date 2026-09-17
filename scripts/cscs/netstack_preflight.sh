#!/usr/bin/env bash
# Resolve what the CSCS container hooks WILL use for the network stack, assert every piece is
# there and is the version this campaign is pinned to, and print the provenance. Run it on the
# HOST before srun; it inspects host paths, not the container.
#
#   . scripts/cscs/netstack_preflight.sh   # or run it directly
#
# WHY THIS EXISTS, and why it aborts rather than warns.
#
# The three hooks (10-netstack, 89-libfabric-cxi, 90-aws-ofi-nccl) all begin with
#     netstack_src=${OCI_ANNOTATION_com__hooks__netstack__source:-artifact}
# and the DEFAULT, artifact, resolves under
#     /capstor/store/cscs/cscs/public/containers/netstack/$(uname -m)/<ver>/<name>
# /capstor does not exist here, so with site defaults that path never resolves. The
# hooks do not fail when it is missing: they set libfabric_host_path and plugin_host_path to
# files that are not there, the bind-mounts quietly do nothing, and RCCL falls back from
# Slingshot/CXI to TCP. The job still completes. It is merely far slower, and it looks like a
# slow model rather than a dead fabric -- which is the single most expensive way for this to fail.
#
# So: the EDFs set com.hooks.netstack.source = "host", and this script proves the host path is
# real before any allocation is spent on it.
#
# VERSION PINNING. The cxi hook hardcodes /opt/cray/libfabric/host/lib64/libfabric.so.1; there is
# no annotation naming a version directory. `host` is a root-owned symlink CSCS can repoint at
# any time (it currently points at 2.3.1, set 2026-09-16). Pinning therefore cannot mean choosing
# the path -- it means ASSERTING what the symlink resolved to, so a repoint is a loud failure
# naming the change instead of a fabric stack that silently differs between two arms of the same
# campaign. Measured 2026-09-16 on beverin, 1 MB fi_pingpong over cxi: 2.3.1 19.9 GB/s,
# 1.22.0 20.0 GB/s, host 19.3 GB/s -- within noise, so the pin is about REPRODUCIBILITY, not speed.
#
# Override for a deliberate move: HPCAGENT_BENCH_LIBFABRIC=<ver>, or "any" to accept whatever
# `host` resolves to.
set -uo pipefail

: "${HPCAGENT_BENCH_LIBFABRIC:=2.3.1}"          # pinned version, or "any"
: "${HPCAGENT_BENCH_NETSTACK_SOURCE:=host}"     # must match the EDF annotation
: "${HPCAGENT_BENCH_OFI_VARIANT:=rocm6}"        # MI300A; the only variant installed on beverin

netstack_preflight() {
    local rc=0 base resolved ver lib plugin

    if [ "${HPCAGENT_BENCH_NETSTACK_SOURCE}" != "host" ]; then
        echo "netstack: source=${HPCAGENT_BENCH_NETSTACK_SOURCE} (not 'host')" >&2
        echo "  The artifact tree lives under /capstor, which does not exist here. This WILL" >&2
        echo "  fall back to TCP. Set com.hooks.netstack.source=\"host\"." >&2
        return 1
    fi

    base=/opt/cray/libfabric/host
    resolved="$(readlink -f "${base}" 2>/dev/null || true)"
    ver="${resolved##*/}"
    lib="${base}/lib64/libfabric.so.1"
    plugin="/opt/cscs/aws-ofi-ccl-plugin/${HPCAGENT_BENCH_OFI_VARIANT}/librccl-net.so"

    [ -e "${lib}" ]    || { echo "netstack: MISSING libfabric ${lib}" >&2; rc=1; }
    [ -e "${plugin}" ] || {
        echo "netstack: MISSING RCCL plugin ${plugin}" >&2
        echo "  installed variants: $(ls /opt/cscs/aws-ofi-ccl-plugin 2>/dev/null | tr '\n' ' ')" >&2
        rc=1; }

    if [ "${HPCAGENT_BENCH_LIBFABRIC}" != "any" ] && [ "${ver}" != "${HPCAGENT_BENCH_LIBFABRIC}" ]; then
        echo "netstack: libfabric PIN BROKEN -- /opt/cray/libfabric/host -> ${ver:-<unresolved>}," >&2
        echo "  but this campaign is pinned to ${HPCAGENT_BENCH_LIBFABRIC}. CSCS repointed the" >&2
        echo "  symlink. Either accept it (HPCAGENT_BENCH_LIBFABRIC=${ver}) or pin deliberately," >&2
        echo "  but do NOT let one campaign straddle two fabric stacks." >&2
        echo "  available: $(ls -d /opt/cray/libfabric/*/ 2>/dev/null | xargs -n1 basename | tr '\n' ' ')" >&2
        rc=1
    fi

    # Provenance: what a results table should be able to cite.
    printf 'netstack: source=host libfabric=%s (%s) plugin=%s cxi_nics=%s\n' \
        "${ver:-?}" "$(basename "$(readlink -f "${lib}" 2>/dev/null)" 2>/dev/null || echo '?')" \
        "${HPCAGENT_BENCH_OFI_VARIANT}" "$(ls -d /dev/cxi[0-9]* 2>/dev/null | wc -l)"
    return "${rc}"
}

# Executed directly -> act as a gate. Sourced -> just define the function.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    netstack_preflight || exit 1
fi
