#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# ONE-SHOT: experiments/.env.<arm> -> experiments/campaigns/<experiment>/<group>/.env.<arm>
#
# experiments/ is 66 arm envs, ten launchers and the shared machinery in one flat directory, and
# nothing in the listing says which arms belong together. The database already knows: every arm
# stamps experiment / device / packet, so the folder can be DERIVED from what the arm records
# rather than from its name. One directory per experiment, one subdirectory per submission group.
#
# WHY A SCRIPT. beverin.sbatch is submitted with `--export=ALL,CLUSTER_ENV_FILE=<absolute path>`
# and sources that path at job start (beverin.sbatch:33), then re-exports it for the srun steps
# (:48). A live arm therefore holds an absolute path STRING to its env and re-resolves it for the
# rest of its life -- moving the file under a running campaign takes its configuration out from
# under it, hours into a multi-node allocation. The guards and the compat symlinks are the point of
# this file; the moves are the easy part.
set -euo pipefail
cd -- "$(git rev-parse --show-toplevel)"

SRC=experiments
DEST=experiments/campaigns
DRY=${DRY:-0}
FORCE=${FORCE:-0}

run() { if [[ "${DRY}" == 1 ]]; then printf 'would: %s\n' "$*"; else "$@"; fi; }

# The identity an arm records IS the folder. Anything that records none is a generator seed or a
# smoke: neither is a submission group, so neither gets a campaign directory.
group_of() {
    local env=$1 experiment device packet
    experiment=$(grep -m1 '^HPCAGENT_BENCH_RECORD_EXPERIMENT=' "${env}" | cut -d= -f2- || true)
    [[ -n "${experiment}" ]] || return 1
    device=$(grep -m1 '^HPCAGENT_BENCH_RECORD_DEVICE=' "${env}" | cut -d= -f2- || true)
    packet=$(grep -m1 '^HPCAGENT_BENCH_RECORD_PACKET=' "${env}" | cut -d= -f2- || true)
    # A CPF arm is its own submission group: it is prepared differently (prerender_cpf.sh) and
    # submitted on its own cadence, which is what "sub-submission" means here. Every other packet
    # rides along with the device group it was submitted in.
    case "${packet}" in
        cpf | cpfsrc) printf '%s/cpf-%s\n' "${experiment}" "${device:-cpu}" ;;
        *) printf '%s/%s\n' "${experiment}" "${device:-cpu}" ;;
    esac
}

# GUARD: a live arm re-resolves its CLUSTER_ENV_FILE. Image builds never read the campaign tree.
live=$(squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -vE '^(build-|suite$)' | wc -l)
if [[ "${live}" -gt 0 && "${DRY}" != 1 && "${FORCE}" != 1 ]]; then
    echo "REFUSING: ${live} campaign job(s) in flight; each re-resolves its own .env by absolute path" >&2
    squeue -u "${USER}" -o '%.10i %.36j %.9T %.10M %.5D' >&2
    echo "re-run with FORCE=1 to move anyway -- the compat symlinks below are what make that survivable" >&2
    exit 1
fi

moved=0
for env in "${SRC}"/.env.*; do
    [[ -f "${env}" ]] || continue
    group=$(group_of "${env}") || { printf 'skip (no identity): %s\n' "${env##*/}"; continue; }
    run mkdir -p "${DEST}/${group}"
    run git mv "${env}" "${DEST}/${group}/${env##*/}"
    # COMPAT SHIM. A job submitted before this ran holds the OLD absolute path and sources it again
    # at every step. git mv is a rename(), so an already-open fd follows the inode -- but a path
    # resolved fresh from the old string would 404 and the arm would run unconfigured. The symlink
    # keeps that string valid. Untracked and TEMPORARY: delete once the last pre-move job leaves
    # the queue.
    run ln -s "campaigns/${group}/${env##*/}" "${env}"
    moved=$((moved + 1))
done
echo "moved ${moved} arm env(s) into ${DEST}"

# The launchers stay in experiments/: they source arm_nodes.sh, record_identity.sh, skill_args.sh
# and pin_env_kv.sh as `. ./x.sh` after cd'ing to their own directory, and run_cluster.sh and
# beverin.sbatch are resolved relative to them. Moving them too would break all four sibling
# surfaces at once for a directory rename, so what moves is the DATA -- which is what a reader
# opening experiments/ was failing to make sense of.
echo
echo "verifying every launcher still finds the envs it writes"
left=$(git grep -nI '"\.env\.\|\.env\.\${' -- experiments/submit-*.sh | wc -l)
echo "  ${left} launcher reference(s) to .env.<arm> -- these now resolve through the symlinks"
echo
echo "next: ENV_DIR must point at the campaign folder in each launcher, then delete the symlinks"
echo "once squeue is clear of pre-move jobs."
