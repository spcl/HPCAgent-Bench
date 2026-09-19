#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# A hosted-service model through the existing submitters, in LANES of chained jobs.
#
# A service is rate limited per ACCOUNT, so the arm's concurrency is set by the key, not by nodes:
# AGENTS_PER_NODE agents per job, LANES jobs at a time. At that width a 40-kernel roster cannot
# finish inside one job (scicomp gives each agent 20 h; the partition ends a job at 24 h), so every
# stage is split into shards of AGENTS_PER_NODE kernels, one job each, and each job waits for the
# one before it in its lane (afterany: a failed shard does not stall the rest). A shard's arm name
# carries -s<i>of<n>; its recorded identity (experiment, model, language, device, packet) does not,
# so the analysis pairs shards of one condition as one arm.
#
# One lane per language, stages run in STAGES order within a lane:
#   MODEL=unionalpha LANGS="c fortran" ./submit-service-chain.sh        SUBMIT=0 to prepare only
#   STAGES="scicomp:plain scicomp:lang-skills+perf-playbook-cpu llr:plain llr:skills"
set -euo pipefail
ulimit -c 0
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PY="${PY:-${SCRATCH:?set SCRATCH}/venv-hpcagent-bench-314/bin/python}"
export OPT="${OPT:-$(dirname "${PWD}")}"
export PYTHONPATH="${OPT}:${OPT}/hpcagent_bench/numpy_translators/src${PYTHONPATH:+:${PYTHONPATH}}"

MODEL=${MODEL:?MODEL names a .env.base-<model> service arm}
LANGS=${LANGS:-"c fortran"}
STAGES=${STAGES:-"scicomp:plain scicomp:lang-skills+perf-playbook-cpu llr:plain llr:skills"}
STAMP=${STAMP:-$(date +%Y%m%d)}
DEPEND_ON=${DEPEND_ON:-}
SHARD_DIR=${SHARD_DIR:-shards}

base=".env.base-${MODEL}"
# the base flattened through its layers (env_layers.sh): a key may live in layers/common.env
base_flat="$(./env_layers.sh render "${base}")" || { echo "${base} is missing or does not render" >&2; exit 2; }
grep -qx 'INFERENCE_SOURCE=service' <<<"${base_flat}" || { echo "${base} is not a service arm" >&2; exit 2; }
# concurrency is the base env's, sized for the key; a caller may narrow it, never silently widen it
BASE_WIDTH="$(grep -oP '^AGENTS_PER_NODE=\K[0-9]+' <<<"${base_flat}")"
export AGENTS_PER_NODE=${AGENTS_PER_NODE:-${BASE_WIDTH}}
(( AGENTS_PER_NODE <= BASE_WIDTH )) || { echo "AGENTS_PER_NODE=${AGENTS_PER_NODE} exceeds ${base}'s ${BASE_WIDTH}" >&2; exit 2; }
# no engine to pull or warm: only the judge starts before the agents
export STAGING_HOURS=${STAGING_HOURS:-1}
# the service key must be in THIS environment: sbatch --export=ALL is how it reaches the job
key_env="$(grep -oP '^INFERENCE_SERVICE_KEY_ENV=\K\S+' <<<"${base_flat}")"
[[ "${SUBMIT:-1}" != 1 || -n "${!key_env:-}" ]] || { echo "${key_env} is not exported" >&2; exit 2; }
if grep -qx 'INFERENCE_SERVICE_FREE_ONLY=1' <<<"${base_flat}"; then
    ( set -a; . <(printf '%s\n' "${base_flat}"); set +a; export "${key_env}=${!key_env:-unset}"; "${PY}" ./inference_service.py --check-free )
fi

. ./check_problems.sh
. ./submit_common.sh
. ./roster.sh

# roster <stage> -> kernel names, one per line
roster() {
    case "$1" in
        scicomp) kernels_file_list kernels-scicomp40.txt ;;
        llr) roster_for llr-focus40 | tr ',' '\n' ;;
        *) echo "unknown stage track $1" >&2; return 2 ;;
    esac
}

# shard_files <track> -> writes SHARD_DIR/<track>.s<i>of<n>.txt, prints their paths in order
shard_files() {
    local track="$1" i n
    local listed
    listed="$(roster "${track}")" || return 2
    mapfile -t names <<<"${listed}"
    (( ${#names[@]} > 0 )) || { echo "empty roster for ${track}" >&2; return 2; }
    n=$(( (${#names[@]} + AGENTS_PER_NODE - 1) / AGENTS_PER_NODE ))
    mkdir -p "${SHARD_DIR}"
    for (( i = 0; i < n; i++ )); do
        local file="${SHARD_DIR}/${track}-${MODEL}.s$((i + 1))of${n}.txt"
        printf '%s\n' "${names[@]:i*AGENTS_PER_NODE:AGENTS_PER_NODE}" >"${file}"
        printf '%s\n' "${file}"
    done
}

# submit_shard <lane-dep> <track> <kind> <lang> <shard-file> -> prints the job id (or "")
submit_shard() {
    local dep="$1" track="$2" kind="$3" lang="$4" file="$5" tag out jid
    tag="$(basename "${file}" .txt)"; tag="${tag##*.}"
    local -a cmd
    case "${track}" in
        scicomp)
            local exp=scicomp-dc; [[ "${lang}" == c ]] || exp="scicomp-dc-${lang}"
            if [[ "${kind}" == plain ]]; then
                cmd=(env EXPERIMENT="${exp}-${tag}" LANGUAGE="${lang}" ARMS=plain ./submit-scicomp-dc.sh)
            else
                exp=scicomp-perf-playbook; [[ "${lang}" == c ]] || exp="scicomp-perf-playbook-${lang}"
                cmd=(env EXPERIMENT="${exp}-${tag}" LANGUAGE="${lang}" PACKET="${kind}" ARMS="${kind}" ./submit-scicomp-perf-playbook.sh)
            fi ;;
        llr)
            cmd=(env EXPERIMENT="cpf-llr-focus40-${tag}" ARMS="${lang}:${kind}" ./submit-cpf-llr40.sh) ;;
    esac
    out="$(MODELS="${MODEL}" KERNELS_FILE="${file}" DEPEND_ON="${dep}" STAMP="${STAMP}" "${cmd[@]}")" || {
        printf '%s\n' "${out}" >&2; return 2; }
    printf '%s\n' "${out}" >&2
    jid="$(grep -oP 'submitted \S+ -> \K[0-9]+' <<<"${out}" | tail -1 || true)"
    printf '%s' "${jid}"
}

for lang in ${LANGS}; do
    dep="${DEPEND_ON}"
    for stage in ${STAGES}; do
        track="${stage%%:*}" kind="${stage#*:}"
        # not `mapfile < <(...)`: a process substitution's failure never reaches set -e, and an empty
        # roster would silently submit no stage at all
        shard_list="$(shard_files "${track}")" || exit 2
        mapfile -t files <<<"${shard_list}"
        for file in "${files[@]}"; do
            jid="$(submit_shard "${dep}" "${track}" "${kind}" "${lang}" "${file}")" || exit 2
            [[ -z "${jid}" ]] || dep="${jid}"
        done
    done
    echo "lane ${lang}: last job ${dep:-none}"
done
