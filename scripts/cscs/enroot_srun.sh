#!/usr/bin/env bash
# Run a command inside an EDF's image through enroot directly, bypassing pyxis.
#
#   scripts/cscs/enroot_srun.sh <edf-name | /path/to/edf.toml> [srun args...] -- <command...>
#
# A bare name is looked up in ${EDF_DIR:-$HOME/.edf}; a path is used as given (run_cluster.sh passes
# the per-run, per-role EDF that derived_edf wrote, so each role keeps its own narrowed mounts).
#
# WHY. Since the Sep 2026 migration /etc/enroot/enroot.conf still sets ENROOT_CACHE_PATH under the
# decommissioned /capstor. pyxis starts containers from a SPANK plugin with a sanitised environment,
# so every `srun --environment=...` dies at task_init() and no user-side override reaches it
# (tried: exporting ENROOT_CACHE_PATH, sbatch --export, ENROOT_SYSCONF_PATH, --container-image).
# enroot called DIRECTLY honours its environment, so this does the pyxis job by hand.
#
# DESIGN, taken from a recipe that ran single-node sglang on beverin, and measured here 2026-09-17:
#   * `enroot start <image>.sqsh` MOUNTS the squashfs. Startup 8 s, zero tmpfs. (`enroot create`
#     would UNPACK it instead -- 53 GB of RAM and ~2 min per node -- which is why this never calls it.)
#   * enroot's runtime/data/temp dirs live in a per-task mktemp dir removed on exit, so nothing
#     accumulates on a node and two tasks on one node cannot collide.
#   * `enroot start` does NOT inherit the host environment -- measured: SLURM_PROCID, SLURM_NTASKS
#     and SLURM_LOCALID are all unset inside. Anything that derives a rank from them then believes
#     it is rank 0: the first smoke run of the canon columns had four ranks each run every kernel
#     and write one CSV concurrently (a second header line appeared mid-file). So the task's
#     identity is FORWARDED explicitly: HPCAGENT_BENCH_ENROOT_FORWARD=rank (default, an allowlist)
#     or =all (everything but the host's toolchain, what run_cluster.sh's role steps need). The rule
#     lives in enroot_forward.sh. Either way the host's PATH and LD_LIBRARY_PATH never replace the
#     image's, and where the EDF sets a variable the EDF wins.
#   * Forwarded VALUES never appear on a command line: the body exports each one and hands enroot
#     only its NAME (`--env NAME` copies the caller's value -- measured 2026-09-17, quotes and $
#     intact). run_cluster.sh forwards the inference key, and argv is readable by anyone on the node.
#   * --conf overrides the image entrypoint with `rc() { exec "$@"; }`, so the command given here
#     is what runs, not whatever the image would have started.
#   * Everything pyxis would have taken from the EDF is PORTED: image, mounts, workdir, every
#     [env] entry (passed with --env) and every [annotations] entry (as OCI_ANNOTATION_*, which is
#     where the hooks read them). A launcher that drops the [env] section silently runs a different
#     program from the one the EDF describes -- LD_LIBRARY_PATH alone decides whether RCCL finds
#     its network plugin.
#
# COMMUNICATION HOOKS -- HPCAGENT_BENCH_COMM_HOOKS=off|on. When unset, DERIVED from the arm:
#   on  when INFERENCE_NODES > 1, off otherwise.
# The need for hooks is a property of the TOPOLOGY, not of the model's name: they carry a GPU
# collective across a node boundary and nothing else. Across every arm in this repo that rule
# reproduces the hand-kept list exactly -- qwen38 (31 arms) and oss120b (26) run on one inference
# node and need none; kimi27sglang (22 arms) spans four and needs them. A list of model names would
# be right today and wrong the first time a model's node count changed.
# It errs in the SAFE direction: several independent single-node replicas also get hooks, which
# costs a host-library graft (proven safe, glibc 2.39 >= 2.38). The rule can never do the opposite
# -- leave a multi-node model to fall back silently from CXI to TCP -- and that is the failure that
# looks like a slow model rather than a misconfigured job.
#   off: cxi and aws_ofi_nccl disabled. No host libraries are grafted into the image. Right for
#        anything with no inter-node GPU collective: the deterministic framework columns, the
#        judge, and single-node inference.
#   on:  the EDF's own annotations (cxi + aws_ofi_nccl, variant rocm6). Needed for MULTI-NODE
#        collectives; without it RCCL falls back to TCP. Grafts ~29 host libraries built against
#        glibc 2.38, which the images satisfy (glibc 2.39, measured).
#   netstack.source is forced to "host" in BOTH modes: under the default "artifact",
#   10-netstack.sh looks under /capstor and exits 1, aborting container start outright.
#
# Delete this file once CSCS fixes enroot.conf and `srun --environment=` works again.
set -uo pipefail

EDF_NAME="${1:?usage: enroot_srun.sh <edf-name | edf.toml path> [srun args...] -- <command...>}"; shift
case "${EDF_NAME}" in
    */*.toml) EDF="${EDF_NAME}" ;;
    *)        EDF="${EDF_DIR:-${HOME}/.edf}/${EDF_NAME}.toml" ;;
esac
[ -f "${EDF}" ] || { echo "enroot_srun: no such EDF: ${EDF}" >&2; exit 2; }

srun_args=()
while [ $# -gt 0 ]; do
    [ "$1" = "--" ] && { shift; break; }
    srun_args+=("$1"); shift
done
[ $# -gt 0 ] || { echo "enroot_srun: no command after --" >&2; exit 2; }

# SCRATCH is RESOLVED, never trusted. A shell that predates the migration exports the dead
# /capstor value, and an earlier version of this script inherited it and failed with the very
# /capstor mkdir error it exists to route around.
if [[ -z "${SCRATCH:-}" || "${SCRATCH}" == /capstor/* || ! -d "${SCRATCH}" ]]; then
    SCRATCH=""
    for base in "/ritom/scratch/cscs/${USER}" "/capstor/scratch/cscs/${USER}"; do
        [ -d "${base}" ] && { SCRATCH="${base}/$(uname -m)"; break; }
    done
fi
[ -d "${SCRATCH:-}" ] || { echo "enroot_srun: cannot resolve a live scratch directory" >&2; exit 2; }
export SCRATCH

# Parse with tomllib, not a regex: an EDF is TOML and its values contain quotes and commas.
# The batch host's python3 is SLES 3.6, which has no tomllib; python3.11 is present on beverin.
py="$(command -v python3.11 || command -v python3)"
eval "$("${py}" - "${EDF}" <<'PY'
import shlex, sys, tomllib
with open(sys.argv[1], "rb") as fh:
    edf = tomllib.load(fh)
image = edf.get("image", "")
mounts = [m for m in edf.get("mounts", []) if m]
envs = [f"{k}={v}" for k, v in (edf.get("env") or {}).items()]
print(f"export HB_IMAGE={shlex.quote(image)}")
print(f"export HB_WORKDIR={shlex.quote(edf.get('workdir', ''))}")
print(f"export HB_MOUNTS={shlex.quote(chr(10).join(mounts))}")
print(f"export HB_ENVS={shlex.quote(chr(10).join(envs))}")
# `com.hooks.cxi.enabled = "true"` is a TOML DOTTED KEY, which tomllib parses as NESTED tables:
# {"com": {"hooks": {"cxi": {"enabled": "true"}}}}. Iterating the top level therefore yields one key,
# "com", and exported a single meaningless variable -- none of the hook annotations reached the
# hooks. The aws_ofi_nccl hook requires exactly "true" and exits silently otherwise, so RCCL found
# no network plugin and every multi-node collective ran over TCP sockets with a correct result.
# Flatten back to the dotted names the hooks read.
def flatten(table, prefix=""):
    for key, value in table.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from flatten(value, name)
        else:
            yield name, value
for key, value in flatten(edf.get("annotations") or {}):
    print(f"export OCI_ANNOTATION_{key.replace('.', '__')}={shlex.quote(str(value))}")
PY
)" || { echo "enroot_srun: could not parse ${EDF}" >&2; exit 2; }
[ -f "${HB_IMAGE}" ] || { echo "enroot_srun: image does not exist: ${HB_IMAGE}" >&2; exit 2; }

export OCI_ANNOTATION_com__hooks__netstack__source=host
if [[ -z "${HPCAGENT_BENCH_COMM_HOOKS:-}" ]]; then
    if [[ "${INFERENCE_NODES:-1}" =~ ^[0-9]+$ ]] && (( ${INFERENCE_NODES:-1} > 1 )); then
        HPCAGENT_BENCH_COMM_HOOKS=on
    else
        HPCAGENT_BENCH_COMM_HOOKS=off
    fi
fi
echo "enroot_srun: comm hooks ${HPCAGENT_BENCH_COMM_HOOKS} (INFERENCE_NODES=${INFERENCE_NODES:-unset})" >&2
if [[ "${HPCAGENT_BENCH_COMM_HOOKS}" != "on" ]]; then
    export OCI_ANNOTATION_com__hooks__cxi__enabled=false
    export OCI_ANNOTATION_com__hooks__aws_ofi_nccl__enabled=false
    # The inference EDFs FORCE the network plugin the aws_ofi_nccl hook mounts (NCCL_NET="AWS
    # Libfabric", NCCL_NET_PLUGIN=ofi -- the hook writes the same two itself when it runs). With the
    # hook off there is no such plugin, and a forced NCCL_NET makes RCCL refuse to initialize at all,
    # even for a collective that never leaves the node: sglang TP4 on one node died in 640062 with
    # "Failed to initialize any NET plugin" / "NCCL error: invalid usage". Dropped only here, so the
    # EDF keeps describing the multi-node run it was written for.
    HB_ENVS="$(grep -Ev '^(NCCL_NET|NCCL_NET_PLUGIN)=' <<<"${HB_ENVS}" || true)"
fi

export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-${SCRATCH}/.enroot/cache}"
# On the shared filesystem beside this script, so every compute node reads the same files.
HB_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HB_START_CONF="${HB_LIB_DIR}/enroot_start.conf"
HB_FORWARD_LIB="${HB_LIB_DIR}/enroot_forward.sh"
for f in "${HB_START_CONF}" "${HB_FORWARD_LIB}"; do
    [ -f "${f}" ] || { echo "enroot_srun: missing ${f}" >&2; exit 2; }
done
export HB_START_CONF HB_FORWARD_LIB
# Checked HERE, once: inside the body a bad mode would only make every variable look unforwardable.
case "${HPCAGENT_BENCH_ENROOT_FORWARD:=rank}" in
    rank|all) export HPCAGENT_BENCH_ENROOT_FORWARD ;;
    *) echo "enroot_srun: HPCAGENT_BENCH_ENROOT_FORWARD must be rank or all" >&2; exit 2 ;;
esac
echo "enroot_srun: forwarding ${HPCAGENT_BENCH_ENROOT_FORWARD} (EDF ${EDF})" >&2
HB_CMD="$(printf '%q ' "$@")"; export HB_CMD

# Runs once per task, on the compute node.
body='
set -uo pipefail
work=$(mktemp -d "/tmp/hb-enroot.${SLURM_JOB_ID:-local}.${SLURM_PROCID:-0}.XXXXXX")
trap '"'"'rm -rf -- "$work"'"'"' EXIT
export ENROOT_RUNTIME_PATH="$work/runtime" ENROOT_DATA_PATH="$work/data" ENROOT_TEMP_PATH="$work/tmp"
mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_RUNTIME_PATH" "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH"
args=(--rw --conf "$HB_START_CONF")
# An EDF mount is "src:dst[:ro|rw]". enroot --mount is NOT that: it turns every colon into a space
# and hands the result to enroot-mount as an fstab line, so "src:dst:ro" read "ro" as the
# filesystem TYPE (job 640058: the agent tools at /opt/hpcagent-bench-agent never mounted).
# Measured on a node, 2026-09-17: the two-field "src dst" form mounts read-write, creates a missing
# target and binds a mount point with submounts (/ritom/); a full "none x-create=...,bind" entry
# fails that last case with EINVAL. So read-write stays two fields, and only a read-only mount is
# spelled in full -- created as a file or a directory to match its source, as the site hooks do.
while IFS= read -r m; do
    [ -n "$m" ] || continue
    IFS=: read -r src dst mode <<< "$m"
    dst="${dst:-$src}"
    if [ "${mode:-rw}" = ro ]; then
        kind=dir; [ -f "$src" ] && kind=file
        args+=(--mount "$src:$dst:none:x-create=$kind,bind,ro,nosuid,nodev,private")
    else
        args+=(--mount "$src:$dst")
    fi
done <<< "$HB_MOUNTS"
# FORWARD: which variables, see enroot_forward.sh. Each is exported here as HBFWD_<name> and named
# to enroot WITHOUT a value, so no value is ever on the command line.
source "$HB_FORWARD_LIB"
declare -A edf_keys=()
while IFS= read -r e; do [ -n "$e" ] && edf_keys["${e%%=*}"]=1; done <<< "$HB_ENVS"
while IFS= read -r -d "" kv; do
    key="${kv%%=*}"
    [[ -n "${edf_keys[$key]:-}" ]] && continue
    hb_forwardable "$key" || continue
    export "HBFWD_${key}=${kv#*=}"
    args+=(--env "HBFWD_${key}")
done < <(env -0)
# EDF [env] last, so on any key both set, the EDF value is the one that arrives.
while IFS= read -r e; do [ -n "$e" ] && args+=(--env "$e"); done <<< "$HB_ENVS"
enroot start "${args[@]}" "$HB_IMAGE" /bin/bash -c "cd \"$HB_WORKDIR\" 2>/dev/null || true; $HB_CMD"
'
exec srun "${srun_args[@]}" bash -c "${body}"
