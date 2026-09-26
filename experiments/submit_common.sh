#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# submit.sh's plumbing: stage an arm's env from its arms.yaml base, then either report what would run
# (SUBMIT=0) or submit it as beverin.sbatch's CLUSTER_ENV_FILE. Sourced, not executed; the caller has
# sourced arm_nodes.sh and pin_env_kv.sh.

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
# render_env (a <campaign>:<model> base, flattened) and snapshot_env (the per-submission copy a job reads).
. "$(dirname -- "${BASH_SOURCE[0]}")/env_layers.sh"

# symbolic_path <root-var-name> <resolved-absolute-path> -- the path with the CURRENT value of
# ${<root-var-name>} rewritten back to a literal "${<root-var-name>}" prefix, for a value written into
# an arm's .env (tests/test_no_hardcoded_user_paths.py refuses a literal scratch path there). Every
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

# clean_suffix <CLEAN> -> "-clean" when CLEAN=1, else "". A clean re-run keeps the arm's recorded
# identity (experiment, model, language, device, packet); only the arm, job, env and problems names
# carry the suffix, and the analysis prefers the -clean arm.
clean_suffix() {
    [[ "$1" == 1 ]] && printf -- '-clean' || printf ''
}

# BUDGET_SCALE=<N> scales a rerun's budget (the owed "budget" class). TOKEN_SCALE and TIME_SCALE
# default to it and scale tokens and wall clock separately (4x wall clock may not fit the partition).
BUDGET_SCALE=${BUDGET_SCALE:-1}
TOKEN_SCALE=${TOKEN_SCALE:-${BUDGET_SCALE}}
TIME_SCALE=${TIME_SCALE:-${BUDGET_SCALE}}

# The partition's MaxTime with a safety margin (mi300: 24 h). A scaled AGENT_TIMEOUT_SECONDS past what
# fits asks sbatch for a --time the partition never grants, and the job stays PENDING forever.
PARTITION_TIME_LIMIT_HOURS=${PARTITION_TIME_LIMIT_HOURS:-23}

# scale_time <value> -> <value> * TIME_SCALE, clamped so one batch plus STAGING_HOURS still fits the
# partition. scale_tokens is uncapped: a token ceiling costs money, not a job that never starts.
scale_time() {
    local cap=$(( (PARTITION_TIME_LIMIT_HOURS - STAGING_HOURS) * 3600 ))
    (( cap > 0 )) || {
        echo "scale_time: STAGING_HOURS=${STAGING_HOURS} leaves no room under PARTITION_TIME_LIMIT_HOURS=${PARTITION_TIME_LIMIT_HOURS}" >&2
        return 2
    }
    local scaled=$(( $1 * TIME_SCALE ))
    (( scaled > cap )) && scaled="${cap}"
    printf '%s\n' "${scaled}"
}
scale_tokens() { printf '%s\n' "$(( $1 * TOKEN_SCALE ))"; }

# scaled_budget_from <base> <KEY> -> <KEY>'s value in <base> (arms.yaml), scaled. Refuses a base that
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
# "-tok<N>x-time<M>x" otherwise. A scaled rerun never rewrites the canonical .env of its arm.
budget_env_suffix() {
    if [[ "${TOKEN_SCALE}" == "${TIME_SCALE}" ]]; then
        [[ "${TOKEN_SCALE}" == 1 ]] && return 0
        printf -- '-budget%sx' "${TOKEN_SCALE}"
    else
        printf -- '-tok%sx-time%sx' "${TOKEN_SCALE}" "${TIME_SCALE}"
    fi
}

# arm_file_suffix -> budget_env_suffix plus "-<KERNELS_FILE stem>" when a kernels file narrows the
# roster: the suffix of BOTH an arm's env and its problems file, so a subset or scaled submission can
# never overwrite the canonical files a PENDING job of the same arm still reads.
arm_file_suffix() {
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
    "${PY}" -m hpcagent_bench.cpf_cache check --view "$1" --language "$2" --mode "$3" --target "$4" \
        --kernels "$5" "${verified[@]}" || [[ $? == 1 ]] || echo "cpf_cache check failed for view $1"
}

# stage_base_env <base> <arm> <experiment> <stamp> <staged-out> [extra sed -e expr...]
# The arm's base (<campaign>:<model>, arms.yaml) rendered flat, with CAMPAIGN_ARM and RUN_ROOT
# rewritten. Written to <staged-out>, never the final arm env: a later gate that bails leaves no file
# that looks complete.
stage_base_env() {
    local base="$1" arm="$2" experiment="$3" stamp="$4" out="$5" flat
    shift 5
    flat="$(render_env "${base}")" || { echo "stage_base_env: cannot render base env ${base}" >&2; return 2; }
    sed -e "s|^CAMPAIGN_ARM=.*|CAMPAIGN_ARM=${arm}|" \
        -e "s|^RUN_ROOT=.*|RUN_ROOT=\${SCRATCH:?}/hpcagent-bench-runs/${experiment}-${stamp}|" \
        "$@" <<<"${flat}" >"${out}"
}

# PARTITION -- the hardware profile the arms run on, named like its Slurm partition. Unset or mi300:
# the job lands on the site layer's SBATCH_PARTITION. Any other value needs layers/partition-<P>.env
# (and -<model>.env for the serving config); it is for smokes and overflow only.
partition_is_default() { [[ -z "${PARTITION:-}" || "${PARTITION}" == mi300 ]]; }

# apply_partition <env> <model> -- renames every *_CE_ENV of <env> from its -mi300- EDF to the -<P>-
# one, then pins layers/partition-<P>.env and layers/partition-<P>-<model>.env over it. Refuses a
# model with no serving config on <P> and a recorded experiment that does not name <P>, so its rows
# never pool with mi300 data.
apply_partition() {
    local env="$1" model="$2" layer kv experiment
    partition_is_default && return 0
    local dir; dir="$(dirname -- "${BASH_SOURCE[0]}")/layers"
    sed -i -E "s/^([A-Z_]*CE_ENV=.*)-mi300-/\1-${PARTITION}-/" "${env}"
    for layer in "${dir}/partition-${PARTITION}.env" "${dir}/partition-${PARTITION}-${model}.env"; do
        [[ -f "${layer}" ]] || { echo "apply_partition: no ${layer##*/}: ${model} has no ${PARTITION} config" >&2; return 2; }
        while IFS= read -r kv; do
            pin_env_kv "${env}" "${kv}" || return 2
        done < <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "${layer}")
    done
    experiment="$(sed -n 's/^HPCAGENT_BENCH_RECORD_EXPERIMENT=//p' "${env}" | tail -1)"
    [[ "${experiment}" == *"${PARTITION}"* ]] || {
        echo "apply_partition: experiment '${experiment}' does not name ${PARTITION}; ${PARTITION} rows need their own experiment" >&2
        return 2
    }
}

# partition_sbatch_args -- the sbatch words that move a job off the default partition, one per line.
partition_sbatch_args() {
    partition_is_default && return 0
    local layer; layer="$(dirname -- "${BASH_SOURCE[0]}")/layers/partition-${PARTITION}.env"
    printf '%s\n' "--partition=${PARTITION}" "--gpus-per-node=$(sed -n 's/^GPUS_PER_NODE=//p' "${layer}")"
}

# finalize_staged_env <staged> <env> -- the partition layers, then the rename. A bailed gate leaves
# neither a staged nor a final file looking complete.
finalize_staged_env() {
    local staged="$1" env="$2"
    apply_partition "${staged}" "$(sed -n 's/^HPCAGENT_BENCH_RECORD_MODEL=//p' "${staged}" | tail -1)" \
        || { rm -f "${staged}"; return 2; }
    mv -- "${staged}" "${env}"
}

# env_flag <env> <KEY> -- whether <KEY>'s last value in <env> is true/1/yes/on.
env_flag() {
    local value
    value=$(sed -n "s/^$2=//p" "$1" | tail -n 1 | tr -d "\"'" | tr '[:upper:]' '[:lower:]')
    [[ "${value}" == 1 || "${value}" == true || "${value}" == yes || "${value}" == on ]]
}

# submit_finalize_grade <agent-jid> <env> <label>
# The fast-submit mode's final grade (mw4x5): finalize_grade.sbatch chained on the agent job with
# afterany. None for an arm that grades in the job (HPCAGENT_BENCH_GRADING_FINAL_GRADE_ON_SUBMIT), an
# arm whose base says FINALIZE_GRADE=0 (the ML scaling track: mlscale-grade.sbatch), a smoke, and off
# the default partition. An empty <agent-jid> (a dry run) only reports it.
submit_finalize_grade() {
    local jid="$1" env="$2" label="$3" finalize_jid
    if grep -qx 'FINALIZE_GRADE=0' "${env}" || [[ "${label}" =~ (^|-)smoke[0-9]*(-|$) ]] || ! partition_is_default \
        || env_flag "${env}" HPCAGENT_BENCH_GRADING_FINAL_GRADE_ON_SUBMIT; then
        echo "  no finalize grade job for ${label}"
        return 0
    fi
    if [[ -z "${jid}" ]]; then
        echo "  finalize grade of ${label}: chained on it (afterany) when it is submitted"
        return 0
    fi
    finalize_jid=$(sbatch --parsable --dependency="afterany:${jid}" --nice=0 \
        ${HPCAGENT_BENCH_EXCLUDE_NODES:+--exclude="${HPCAGENT_BENCH_EXCLUDE_NODES}"} \
        --job-name="regrade-finalize-${jid}" finalize_grade.sbatch "${jid}") || return 2
    echo "  finalize grade of ${label} -> ${finalize_jid} (afterany:${jid})"
}

# submit_arm_job <env> <arm> <walltime> [dep-ids] [begin] [detail]
# SUBMIT=1 submits a read-only snapshot of <env> and its problems file (snapshot_env) as
# beverin.sbatch's CLUSTER_ENV_FILE, chained afterany on <dep-ids>, held until <begin>, at --nice=NICE
# (default the site layer's HPCAGENT_BENCH_NICE), held with HOLD=1; anything else only reports.
# --no-requeue: a NODE_FAIL requeue reruns the job in the SAME run directory, stacking two runs' rows.
submit_arm_job() {
    local env="$1" arm="$2" walltime="$3" dep_ids="${4:-}" begin="${5:-}" detail="${6:-}" nodes snapshot jid
    nodes=$(arm_nodes "${env}")
    if [[ "${SUBMIT:-0}" != 1 ]]; then
        echo "prepared ${arm} (${nodes} nodes${detail})${begin:+ begin ${begin}}${dep_ids:+ after ${dep_ids}} -- not submitted"
        submit_finalize_grade "" "${env}" "${arm}"
        return 0
    fi
    [[ -n "${SBATCH_ACCOUNT:-}" && "${SBATCH_ACCOUNT}" != root ]] \
        || { echo "set SBATCH_ACCOUNT (site layer or shell) to a project account" >&2; return 2; }
    snapshot=$(snapshot_env "${env}" "${arm}") || return 2
    local -a args=(--parsable --no-requeue --nodes="${nodes}" --time="${walltime}" --job-name="${arm}"
        --nice="${NICE:-${HPCAGENT_BENCH_NICE:-0}}")
    [[ -z "${dep_ids}" ]] || args+=(--dependency="afterany:${dep_ids}")
    [[ -z "${begin}" ]] || args+=(--begin="${begin}")
    [[ "${HOLD:-0}" != 1 ]] || args+=(--hold)
    if ! partition_is_default; then
        grep -qx "HPCAGENT_BENCH_PARTITION=${PARTITION}" "${env}" \
            || { echo "submit_arm_job: ${env} was not staged for PARTITION=${PARTITION}" >&2; return 2; }
        mapfile -t -O "${#args[@]}" args < <(partition_sbatch_args)
    fi
    # The env file pins any CPF view an arm's packet asks for; the caller's own must not leak into others.
    jid=$(env -u CPF_DROPIN_DIR -u CPF_FORMS_DIR -u HPCAGENT_BENCH_SERVICE_CANONICAL_PARALLEL_FORM_DIR \
        sbatch "${args[@]}" --export=ALL,CLUSTER_ENV_FILE="${PWD}/${snapshot}" beverin.sbatch) || return 2
    echo "submitted ${arm} -> ${jid} (${nodes} nodes${detail}) env ${snapshot}"
    submit_finalize_grade "${jid}" "${env}" "${arm}"
}
