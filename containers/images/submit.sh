#!/usr/bin/env bash
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# Submit a container or helper job with the sbatch options of the system it runs on.
#
#   containers/images/submit.sh <job.sbatch> [--system S] [--partition P] [--account A] [--time T]
#       [--gpus-per-node N] [--nice N] [--dry-run] [-- job args...]
#
# The partition, account and GPUs per node come from `hpcagent-bench job options`
# (hpcagent_bench/cluster/systems.py): a flag, else its environment variable or site-layer value, else the
# system's systems.yaml entry (the CSCS site layer names beverin). The flags parse as cluster/submit.sh's do
# (submit_common.sh parse_job_flags); --time defaults to the job's own #SBATCH line, --nice to
# $HPCAGENT_BENCH_NICE. For build_and_verify.sbatch <image> and ROLE=<role> verify_image.sbatch, the partition is
# the role's images.env row unless --partition or --system names one. --dry-run prints the sbatch command instead.
# Plain `sbatch -p P -A A <job.sbatch>` still works.
set -euo pipefail
ulimit -c 0
images="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "${images}/../.." && pwd)"
. "${repo}/scripts/site_env.sh"
. "${repo}/scripts/host_python.sh"
. "${repo}/hpcagent_bench/cluster/submit_common.sh"
. "${images}/images.env"

job="${1:?usage: containers/images/submit.sh <job.sbatch> [job flags] [--dry-run] [-- job args...]}"
shift
[[ -f "${job}" ]] || { echo "submit.sh: no job script ${job}" >&2; exit 2; }
flags=()
dry_run=0
while (( $# )); do
    case "$1" in
        --) shift; break ;;
        --dry-run) dry_run=1 ;;
        *) flags+=("$1") ;;
    esac
    shift
done
parse_job_flags "${flags[@]}" || exit 2

case "${job##*/}" in
    build_and_verify.sbatch) role="${1:-}" ;;
    verify_image.sbatch) role="${ROLE:-}" ;;
    *) role="" ;;
esac
if [[ -n "${role}" && " ${JOB_FLAGS[*]} " != *" --partition "* && " ${JOB_FLAGS[*]} " != *" --system "* ]] \
    && partition="$(ce_image "${role}" partition 2>/dev/null)"; then
    JOB_FLAGS+=(--partition "${partition}")
fi

resolved="$(job_options)" || exit 2
mapfile -t options <<<"${resolved}"
command=(sbatch "${options[@]}" ${TIME_LIMIT:+"--time=${TIME_LIMIT}"} "--nice=${NICE:-${HPCAGENT_BENCH_NICE}}"
    "${job}" "$@")
if (( dry_run )); then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}"
fi
