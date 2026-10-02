#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# submit.sh's plumbing: stage a setup's env from its setups.yaml base, then either report what would run
# (SUBMIT=0) or submit it as services.sbatch's CLUSTER_ENV_FILE. Sourced, not executed; the caller has
# sourced setup_nodes.sh and pin_env_kv.sh.

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
# render_env (a <base>:<model> base, flattened) and snapshot_env (the per-submission copy a job reads).
CLUSTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The setup configs (setups.yaml, layers/) and the operator's generated files (.env.<setup>, .rendered/).
EXPERIMENTS_DIR="$(cd -- "${CLUSTER_DIR}/../../experiments" && pwd)"
# Slurm output of the jobs this file submits: <scratch>/logs (hpcagent_bench.paths.scratch_dir).
HPCAGENT_BENCH_SCRATCH="${HPCAGENT_BENCH_SCRATCH:-$(cd -- "${CLUSTER_DIR}/../.." && pwd)/.scratch}"
. "${CLUSTER_DIR}/env_layers.sh"

# symbolic_path <root-var-name> <resolved-absolute-path> -- the path with the CURRENT value of
# ${<root-var-name>} rewritten back to a literal "${<root-var-name>}" prefix, for a value written into
# a setup's .env (tests/test_no_hardcoded_user_paths.py refuses a literal scratch path there). Every
# consumer sources the .env after cache_env.sh exported <root-var-name>. A path the root-var does not
# prefix is returned unchanged.
symbolic_path() {
    local root_name="$1" path="$2" root_value="${!1:-}"
    if [[ -n "${root_value}" && "${path}" == "${root_value}"/* ]]; then
        printf '%s\n' "\${${root_name}}${path#"${root_value}"}"
    else
        printf '%s\n' "${path}"
    fi
}

# hms <seconds> -> HH:MM:SS, for a sbatch --time computed off a deadline.
hms() { printf '%02d:%02d:%02d\n' "$(( $1 / 3600 ))" "$(( $1 % 3600 / 60 ))" "$(( $1 % 60 ))"; }

# clean_suffix <CLEAN> -> "-clean" when CLEAN=1, else "". A clean re-run keeps the setup's recorded
# identity (study, model, language, device, packet); only the setup, job, env and problems names
# carry the suffix, and the analysis prefers the -clean setup.
clean_suffix() {
    [[ "$1" == 1 ]] && printf -- '-clean' || printf ''
}

# BUDGET_SCALE=<N> scales a rerun's budget (the owed "budget" class). TOKEN_SCALE and TIME_SCALE
# default to it and scale tokens and wall clock separately (4x wall clock may not fit the partition).
BUDGET_SCALE=${BUDGET_SCALE:-1}
TOKEN_SCALE=${TOKEN_SCALE:-${BUDGET_SCALE}}
TIME_SCALE=${TIME_SCALE:-${BUDGET_SCALE}}

# JOB_FLAGS -- the job flags parse_job_flags took from the command line, as `hpcagent-bench job options` arguments.
JOB_FLAGS=()

# job_options [args...] -- `hpcagent-bench job options` (hpcagent_bench/cluster/systems.py), the ONE resolver of the
# job's sbatch fields, hardware and time limit: the flags submit.sh was given (parse_job_flags), else the
# environment, else the system's systems.yaml entry.
job_options() {
    "${HPCAGENT_BENCH_HOST_PYTHON:?source scripts/host_python.sh}" -m hpcagent_bench job options "${JOB_FLAGS[@]}" "$@"
}

# parse_job_flags "$@" -- the job flags of a submitter: --system, --account, --partition, --gpus-per-node, --hardware,
# --time (the job's time limit, else computed from the roster) and --nice, each as `--flag value` or `--flag=value`.
# The resolver's flags go to JOB_FLAGS, over the environment and systems.yaml; any other word is an error.
parse_job_flags() {
    local flag value
    while (( $# )); do
        flag="$1"
        if [[ "${flag}" == --*=* ]]; then
            value="${flag#*=}" flag="${flag%%=*}"
            shift
        else
            (( $# >= 2 )) || { echo "${flag} needs a value" >&2; return 2; }
            value="$2"
            shift 2
        fi
        case "${flag}" in
            --system | --account | --partition | --gpus-per-node | --hardware) JOB_FLAGS+=("${flag}" "${value}") ;;
            --time) TIME_LIMIT="${value}" ;;
            --nice) NICE="${value}" ;;
            *) echo "unknown argument ${flag} (job flags: --system --account --partition --gpus-per-node --hardware --time --nice)" >&2; return 2 ;;
        esac
    done
}

# scale_time <value> -> <value> * TIME_SCALE, clamped so one batch plus STAGING_HOURS still fits the partition's
# longest time limit (max_time_hours, from the system or HPCAGENT_BENCH_MAX_TIME_HOURS): a --time past it is one the
# partition never grants, and the job stays PENDING forever. Without a known limit nothing is clamped.
# scale_tokens is uncapped: a token ceiling costs money, not a job that never starts.
scale_time() {
    local scaled=$(( $1 * TIME_SCALE )) limit cap
    (( TIME_SCALE != 1 )) || { printf '%s\n' "${scaled}"; return 0; }
    limit=$(job_options --print max_time_hours) || return 2
    if [[ -n "${limit}" ]]; then
        cap=$(( (limit - STAGING_HOURS) * 3600 ))
        (( cap > 0 )) || {
            echo "scale_time: STAGING_HOURS=${STAGING_HOURS} leaves no room under the ${limit} h time limit" >&2
            return 2
        }
        (( scaled <= cap )) || scaled="${cap}"
    fi
    printf '%s\n' "${scaled}"
}
scale_tokens() { printf '%s\n' "$(( $1 * TOKEN_SCALE ))"; }

# scaled_budget_from <base> <KEY> -> <KEY>'s value in <base> (setups.yaml), scaled. Refuses a base that
# sets none: a scaled rerun would otherwise apply no scale at all.
scaled_budget_from() {
    local base="$1" key="$2" configured
    configured="$(render_env "${base}" | grep -oP "^${key}=\K[0-9]+" || true)"
    [[ -n "${configured}" ]] || { echo "scaled_budget_from: ${base} sets no ${key}" >&2; return 2; }
    case "${key}" in
        AGENT_TIMEOUT_SECONDS) scale_time "${configured}" ;;
        AGENT_MAX_TOKENS) scale_tokens "${configured}" ;;
        *) echo "scaled_budget_from: no scale defined for ${key}" >&2; return 2 ;;
    esac
}

# budget_env_suffix -> "" at the base budget; "-budget<N>x" when TOKEN_SCALE and TIME_SCALE agree;
# "-tok<N>x-time<M>x" otherwise. A scaled rerun never rewrites the canonical .env of its setup.
budget_env_suffix() {
    if [[ "${TOKEN_SCALE}" == "${TIME_SCALE}" ]]; then
        [[ "${TOKEN_SCALE}" == 1 ]] && return 0
        printf -- '-budget%sx' "${TOKEN_SCALE}"
    else
        printf -- '-tok%sx-time%sx' "${TOKEN_SCALE}" "${TIME_SCALE}"
    fi
}

# setup_file_suffix -> budget_env_suffix plus "-<KERNELS_FILE stem>" when a kernels file narrows the
# roster: the suffix of BOTH a setup's env and its problems file, so a subset or scaled submission can
# never overwrite the canonical files a PENDING job of the same setup still reads.
setup_file_suffix() {
    printf '%s' "$(budget_env_suffix)"
    [[ -z "${KERNELS_FILE:-}" ]] || printf -- '-%s' "$(basename -- "${KERNELS_FILE%.*}")"
}

# refuse_if_queue_references <env-path> [problems-path]
# Refuses when a PENDING or RUNNING job of this user reads <env-path> as its CLUSTER_ENV_FILE (sacct's
# SubmitLine), or <problems-path> as that job's PROBLEMS_FILE (resolved against the env's own
# directory). Both files are written before sbatch, even on a dry run, so without this a later
# submission could rewrite what a queued job has not read yet. Both paths must be absolute.
refuse_if_queue_references() {
    local env_path="$1" problems_path="${2:-}" jids jid cef pf
    command -v squeue >/dev/null 2>&1 || return 0
    jids=$(squeue -u "${USER:-$(id -un)}" -h -t PENDING,RUNNING -o '%i' 2>/dev/null | paste -sd, -)
    [[ -n "${jids}" ]] || return 0
    while IFS='|' read -r jid cef; do
        [[ -n "${jid}" && -n "${cef}" ]] || continue
        if [[ "${cef}" == "${env_path}" ]]; then
            echo "refusing to write ${env_path}: job ${jid} is PENDING/RUNNING and reads it as CLUSTER_ENV_FILE" >&2
            return 2
        fi
        [[ -n "${problems_path}" && -f "${cef}" ]] || continue
        pf=$(sed -n 's/^PROBLEMS_FILE=//p' "${cef}" | tail -n 1)
        [[ -z "${pf}" || "${pf}" == /* ]] || pf="$(dirname -- "${cef}")/${pf}"
        if [[ -n "${pf}" && "${pf}" == "${problems_path}" ]]; then
            echo "refusing to write ${problems_path}: job ${jid} is PENDING/RUNNING and reads it via ${cef}" >&2
            return 2
        fi
    done < <(sacct -j "${jids}" -X -P -o JobID,SubmitLine --noheader 2>/dev/null \
        | sed -n 's/^\([0-9]\+\)|.*CLUSTER_ENV_FILE=\([^[:space:]]*\).*/\1|\2/p')
}

# deadline_setup <deadline> <margin-seconds> -- a wave that must END before <deadline>: sets
# DEADLINE_LIMIT_SECONDS and DEADLINE_WALLTIME (the job's --time). Both stay 0/empty without one.
deadline_setup() {
    local deadline="$1" margin="$2" deadline_epoch
    DEADLINE_LIMIT_SECONDS=0
    DEADLINE_WALLTIME=""
    [[ -n "${deadline}" ]] || return 0
    deadline_epoch=$(date -d "${deadline}" +%s) \
        || { echo "DEADLINE=${deadline} is not a time date(1) understands" >&2; return 2; }
    DEADLINE_LIMIT_SECONDS=$(( deadline_epoch - $(date +%s) - margin ))
    DEADLINE_WALLTIME=$(hms "${DEADLINE_LIMIT_SECONDS}")
    echo "deadline ${deadline}: --time ${DEADLINE_WALLTIME}"
}

# agent_seconds <base> -- the wall clock ONE agent gets: the base's AGENT_TIMEOUT_SECONDS, scaled,
# then only ever SHORTENED by a deadline (after STAGING_HOURS of staging). Refuses under
# MIN_AGENT_SECONDS: too little to measure anything.
agent_seconds() {
    local configured left
    configured=$(scaled_budget_from "$1" AGENT_TIMEOUT_SECONDS) || return 2
    if (( ${DEADLINE_LIMIT_SECONDS:-0} > 0 )); then
        left=$(( DEADLINE_LIMIT_SECONDS - STAGING_HOURS * 3600 ))
        (( left < configured )) && configured="${left}"
    fi
    if (( configured < ${MIN_AGENT_SECONDS:-3600} )); then
        echo "DEADLINE ${DEADLINE:-} leaves ${configured}s per agent of $1, under ${MIN_AGENT_SECONDS:-3600}s" >&2
        return 2
    fi
    printf '%s\n' "${configured}"
}

# forms_missing <view> <language> <mode:form|dropin> <target> <kernel,...> -- one line per kernel the
# CPF cache view cannot serve (a missing form reads as HTTP 200 "unavailable", not an error), or one
# line for a view of the other target or a failed check, so the caller refuses on any output.
forms_missing() {
    local verified=(); [[ "$3" == dropin ]] && verified=(--verified)
    "${HPCAGENT_BENCH_HOST_PYTHON}" -m hpcagent_bench.cpf_cache check --view "$1" --language "$2" --mode "$3" --target "$4" \
        --kernels "$5" "${verified[@]}" || [[ $? == 1 ]] || echo "cpf_cache check failed for view $1"
}

# stage_base_env <base> <setup> <study> <stamp> <staged-out> [extra sed -e expr...]
# The setup's base (<base>:<model>, setups.yaml) rendered flat, with SETUP and RUN_ROOT
# rewritten. Written to <staged-out>, never the final setup env: a later gate that bails leaves no file
# that looks complete.
stage_base_env() {
    local base="$1" setup="$2" experiment="$3" stamp="$4" out="$5" flat
    shift 5
    flat="$(render_env "${base}")" || { echo "stage_base_env: cannot render base env ${base}" >&2; return 2; }
    sed -e "s|^SETUP=.*|SETUP=${setup}|" \
        -e "s|^RUN_ROOT=.*|RUN_ROOT=\${SCRATCH:?}/hpcagent-bench-runs/${experiment}-${stamp}|" \
        "$@" <<<"${flat}" >"${out}"
}

# The hardware (--hardware, HPCAGENT_BENCH_HARDWARE, the system's `hardware`) names the GPU generation whose
# images and serving layers a setup uses: `hpcagent-bench-agent-<hardware>-latest`, layers/hardware-<hardware>.env.
# It is not a Slurm partition (--partition is). layers/common.env names the hardware its EDF names and model layers
# carry (HPCAGENT_BENCH_BASE_HARDWARE); any other hardware renames them and pins its own layers.

# apply_hardware <env> <model> -- pins the job's GPUS_PER_NODE (the resolved --gpus-per-node, which every role's GPU
# split divides) and HPCAGENT_BENCH_HARDWARE into <env>. A hardware other than the base renames every *_CE_ENV from its
# -<base>- EDF to the -<hardware>- one, then pins layers/hardware-<hardware>.env and, for a model served on our nodes, layers/hardware-<hardware>-<model>.env over
# it. A hosted model (INFERENCE_SOURCE=service) runs no engine here, so it needs no serving layer. Refuses:
# no hardware under the Container Engine (the EDF names carry it), a hardware with no layer, a served model with no
# serving config on it, and a recorded study that does not name it, so its rows never pool with the base's.
apply_hardware() {
    local env="$1" model="$2" hardware base runtime layer kv study gpus
    gpus="$(job_options --require gpus_per_node --print gpus_per_node)" || return 2
    pin_env_kv "${env}" "GPUS_PER_NODE=${gpus}" || return 2
    hardware="$(job_options --print hardware)" || return 2
    base="$(sed -n 's/^HPCAGENT_BENCH_BASE_HARDWARE=//p' "${env}" | tail -1)"
    runtime="$(sed -n 's/^CONTAINER_RUNTIME=//p' "${env}" | tail -1)"
    if [[ -z "${hardware}" ]]; then
        [[ "${runtime:-ce}" != ce ]] || {
            echo "apply_hardware: no hardware; the EDF names carry it: pass --hardware <hardware> (or --system <s>), or set HPCAGENT_BENCH_HARDWARE" >&2
            return 2
        }
        return 0
    fi
    pin_env_kv "${env}" "HPCAGENT_BENCH_HARDWARE=${hardware}" || return 2
    [[ "${hardware}" != "${base}" ]] || return 0
    local dir="${EXPERIMENTS_DIR}/layers"
    local layers=("${dir}/hardware-${hardware}.env")
    grep -qx 'INFERENCE_SOURCE=service' "${env}" || layers+=("${dir}/hardware-${hardware}-${model}.env")
    sed -i -E "s/^([A-Z_]*CE_ENV=.*)-${base}-/\1-${hardware}-/" "${env}"
    for layer in "${layers[@]}"; do
        [[ -f "${layer}" ]] || { echo "apply_hardware: no ${layer##*/}: ${model} has no ${hardware} config" >&2; return 2; }
        while IFS= read -r kv; do
            pin_env_kv "${env}" "${kv}" || return 2
        done < <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "${layer}")
    done
    study="$(sed -n 's/^HPCAGENT_BENCH_RECORD_STUDY=//p' "${env}" | tail -1)"
    [[ "${study}" == *"${hardware}"* ]] || {
        echo "apply_hardware: study '${study}' does not name ${hardware}; ${hardware} rows need their own study" >&2
        return 2
    }
}

# apply_flavor <env> -- CE_IMAGE_FLAVOR=native (containers/images/images.env) renames every agent and
# judge *_CE_ENV of <env> from its -latest EDF to the -native one install_edfs.sh renders for a native
# build. The serving EDFs stay -latest: those images are prebuilt engines, the same either way.
apply_flavor() {
    local env="$1"
    case "${CE_IMAGE_FLAVOR:-latest}" in
        latest) return 0 ;;
        native) sed -i -E '/^INFERENCE_[A-Z_]*CE_ENV=/!s/^([A-Z_]*CE_ENV=.*)-latest$/\1-native/' "${env}" ;;
        *) echo "apply_flavor: CE_IMAGE_FLAVOR is latest or native, got '${CE_IMAGE_FLAVOR}'" >&2; return 2 ;;
    esac
}

# finalize_staged_env <staged> <env> -- the hardware, the image flavor, then the rename. A
# bailed gate leaves neither a staged nor a final file looking complete.
finalize_staged_env() {
    local staged="$1" env="$2"
    apply_hardware "${staged}" "$(sed -n 's/^HPCAGENT_BENCH_RECORD_MODEL=//p' "${staged}" | tail -1)" \
        || { rm -f "${staged}"; return 2; }
    apply_flavor "${staged}" || { rm -f "${staged}"; return 2; }
    mv -- "${staged}" "${env}"
}

# submit_setup_job <env> <setup> <walltime> [dep-ids] [begin] [detail]
# SUBMIT=1 submits a read-only snapshot of <env> and its problems file (snapshot_env) as
# services.sbatch's CLUSTER_ENV_FILE, chained afterany on <dep-ids>, held until <begin>, at --nice=NICE
# (default the site layer's HPCAGENT_BENCH_NICE), held with HOLD=1; anything else only reports.
# --no-requeue: a NODE_FAIL requeue reruns the job in the SAME run directory, stacking two runs' rows.
submit_setup_job() {
    local env="$1" setup="$2" walltime="$3" dep_ids="${4:-}" begin="${5:-}" detail="${6:-}" nodes snapshot jid resolved
    local -a options
    nodes=$(setup_nodes "${env}")
    if [[ "${SUBMIT:-0}" != 1 ]]; then
        echo "prepared ${setup} (${nodes} nodes${detail})${begin:+ begin ${begin}}${dep_ids:+ after ${dep_ids}} -- not submitted"
        return 0
    fi
    resolved=$(job_options) || return 2
    mapfile -t options <<<"${resolved}"
    [[ " ${options[*]} " != *" --account=root "* ]] \
        || { echo "submit_setup_job: account root is not a project account: pass --account or set SBATCH_ACCOUNT" >&2; return 2; }
    snapshot=$(snapshot_env "${env}" "${setup}") || return 2
    local logs="${HPCAGENT_BENCH_SCRATCH}/logs"
    mkdir -p -- "${logs}"
    local -a args=(--parsable --no-requeue --nodes="${nodes}" --time="${walltime}" --job-name="${setup}"
        --output="${logs}/services-%j.out" --error="${logs}/services-%j.err"
        --nice="${NICE:-${HPCAGENT_BENCH_NICE:-0}}" "${options[@]}")
    [[ -z "${dep_ids}" ]] || args+=(--dependency="afterany:${dep_ids}")
    [[ -z "${begin}" ]] || args+=(--begin="${begin}")
    [[ "${HOLD:-0}" != 1 ]] || args+=(--hold)
    # The env file pins any CPF view a setup's packet asks for; the caller's own must not leak into others.
    # CLUSTER_SCRIPT_DIR: under sbatch BASH_SOURCE is a spooled copy, so the job finds run_cluster.sh through this.
    jid=$(env -u CPF_DROPIN_DIR -u CPF_FORMS_DIR -u HPCAGENT_BENCH_SERVICE_CANONICAL_PARALLEL_FORM_DIR \
        CLUSTER_SCRIPT_DIR="${CLUSTER_DIR}" \
        sbatch "${args[@]}" --export=ALL,CLUSTER_ENV_FILE="${PWD}/${snapshot}" "${CLUSTER_DIR}/services.sbatch") || return 2
    echo "submitted ${setup} -> ${jid} (${nodes} nodes${detail}) env ${snapshot}"
}
