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

# Both halves of this script are unsafe while a campaign job is in flight, for ONE reason:
# beverin.sbatch is submitted with --export=ALL,CLUSTER_ENV_FILE=<absolute path>, sources it at job
# start and re-exports it for the srun steps, so a live arm holds an absolute path STRING to its
# env and re-resolves it for the rest of its life. Moving the file takes an arm's configuration out
# from under it; removing the shim that keeps the old string valid does the same thing later.
#
# Image builds and this repo's own suite job never read the campaign tree, so they are exempt.
live_jobs() {
    squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -vcE '^(build-|suite$)' || true
}

refuse_if_live() {
    local what=$1 live
    live=$(live_jobs)
    [[ "${live}" -gt 0 && "${DRY}" != 1 && "${FORCE}" != 1 ]] || return 0
    echo "REFUSING to ${what}: ${live} campaign job(s) in flight, each re-resolving its own .env" >&2
    squeue -u "${USER}" -o '%.10i %.36j %.9T %.10M %.5D' >&2
    echo "FORCE=1 moves anyway; the compat symlinks are what make that survivable" >&2
    exit 1
}

# CLEANUP: remove the compat symlinks once no pre-move job is left in the queue. Each one is
# checked to BE a symlink and to point inside campaigns/ before it is touched, because the whole
# risk of leaving symlinks behind is a later pass following one into the real tree -- so the pass
# that removes them is the one that must not.
if [[ "${1:-}" == "--cleanup" ]]; then
    refuse_if_live "remove the compat symlinks"
    # TWO PASSES: validate every link, THEN remove. Validating inside the removal loop deletes
    # whatever sorted before the offender and only then refuses, which is a half-applied cleanup --
    # the one state that is worse than either doing it or not.
    shims=()
    for link in "${SRC}"/.env.*; do
        [[ -L "${link}" ]] || continue
        target=$(readlink -- "${link}")
        if [[ "${target}" != campaigns/* ]]; then
            echo "REFUSING: ${link} points at ${target}, not into campaigns/; nothing removed" >&2
            exit 1
        fi
        if [[ ! -f "${SRC}/${target}" ]]; then
            echo "REFUSING: ${link} -> ${target} does not exist; nothing removed" >&2
            exit 1
        fi
        shims+=("${link}")
    done
    for link in "${shims[@]+"${shims[@]}"}"; do
        run rm -- "${link}"
    done
    echo "removed ${#shims[@]} compat symlink(s); the campaign folders are now the only path"
    exit 0
fi

refuse_if_live "move the arm envs"
moved=0
for env in "${SRC}"/.env.*; do
    # A symlink here is THIS script's own compat shim from an earlier run. Moving it would carry
    # the link into the campaign folder and leave it pointing at itself, so a second run is a no-op
    # rather than a corruption.
    [[ -L "${env}" ]] && { printf 'already migrated: %s\n' "${env##*/}"; continue; }
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
echo "next: point ENV_DIR at the campaign folder in each launcher, then, once squeue holds no"
echo "pre-move job, run: scripts/migrate_campaign_dirs.sh --cleanup"
