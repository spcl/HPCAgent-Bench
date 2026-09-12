#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# ONE-SHOT, three phases, dry by default:
#
#   envs      experiments/.env.<arm>            -> experiments/campaigns/<experiment>/<group>/
#   folders   paper_artifacts/experiments/      -> one folder per experiment, then per device
#   dirs      $SCRATCH/cpf-*-*, $SCRATCH/*-owed -> $SCRATCH/campaigns/<tag>/<target>/<kind>
#
#   scripts/migrate_campaign_dirs.sh                  every phase, printing what it would do
#   scripts/migrate_campaign_dirs.sh --execute dirs   one phase, for real
#   scripts/migrate_campaign_dirs.sh --cleanup        drop the envs compat symlinks
#   scripts/migrate_campaign_dirs.sh --cleanup-dirs   drop the scratch compat symlinks
#
# WHY A SCRIPT RATHER THAN A HANDFUL OF MOVES. Three surfaces re-resolve an absolute path string
# after the move would have happened:
#
#   - beverin.sbatch is submitted with --export=ALL,CLUSTER_ENV_FILE=<absolute path>, sources it at
#     job start (:33) and re-exports it for every srun step (:48). A live arm holds that string for
#     the rest of its life.
#   - a generated .env freezes its configuration, including the absolute form and drop-in
#     directories, and a queued job reads it by path at start rather than at submit.
#   - cpf_bridge.render_kernel forks per kernel and re-imports from disk, so a forms directory is
#     re-opened by name for as long as an arm is serving.
#
# git mv and mv are rename(), so an already-open fd follows the inode, but a path resolved fresh
# from the old string 404s and the arm runs unconfigured, silently: an unset or missing form
# directory answers `unavailable` with HTTP 200 and the treated arm collapses into its own control.
# The compat symlinks and the queue guard are the point of this file; the moves are the easy part.
set -euo pipefail
cd -- "$(git rev-parse --show-toplevel)"

SELF=scripts/migrate_campaign_dirs.sh   # names the old paths in its own tables; never sweep it
SRC=experiments
DEST=experiments/campaigns
SCRATCH=${SCRATCH:?}
ICLR26=${ICLR26:-${SCRATCH}/ICLR26Reproducibility}
PA=paper_artifacts/experiments          # relative to ICLR26
EXECUTE=0
FORCE=0
PHASES=()

while (($#)); do
    case $1 in
        --execute) EXECUTE=1 ;;
        --dry-run) EXECUTE=0 ;;
        --force) FORCE=1 ;;
        --cleanup | --cleanup-dirs) PHASES=("${1#--}") ;;
        envs | folders | dirs) PHASES+=("$1") ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
[[ ${#PHASES[@]} -gt 0 ]] || PHASES=(envs folders dirs)

run() { if [[ "${EXECUTE}" == 1 ]]; then "$@"; else printf 'would: %s\n' "$*"; fi; }

# Image builds and this repo's own suite job never read a campaign tree, so they are exempt.
# Everything else in the queue either holds a CLUSTER_ENV_FILE string or is serving forms.
live_jobs() { squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -vcE '^(build-|suite$)' || true; }

refuse_if_live() {
    local what=$1 live
    live=$(live_jobs)
    [[ "${live}" -gt 0 && "${EXECUTE}" == 1 && "${FORCE}" != 1 ]] || return 0
    echo "REFUSING to ${what}: ${live} campaign job(s) in flight, each re-resolving its own paths" >&2
    squeue -u "${USER}" -o '%.10i %.36j %.9T %.10M %.5D' >&2
    echo "--force moves anyway; the compat symlinks are what make that survivable" >&2
    exit 1
}

# A directory written in the last hour is being re-rendered right now. prerender_cpf.sh rewrites a
# forms directory in place while the arms reading it are mid-turn, so a move landing in that window
# swaps the tree under a fork that has already decided what file to open.
refuse_if_warm() {
    local src=$1 age
    [[ -e "${src}" && "${EXECUTE}" == 1 && "${FORCE}" != 1 ]] || return 0
    age=$(( ($(date +%s) - $(stat -c %Y "${src}")) / 60 ))
    [[ "${age}" -lt 60 ]] || return 0
    echo "REFUSING: ${src} changed ${age} minute(s) ago; a re-render may be in progress" >&2
    echo "--force moves anyway" >&2
    exit 1
}

# git mv when the path is tracked so history follows, plain mv when it is not. Asking git is the
# only reliable test: paper_artifacts holds tracked CSVs beside gitignored timings/ output.
move() {
    local repo=$1 src=$2 dst=$3
    [[ -e "${repo}/${src}" ]] || { printf 'absent, nothing to move: %s\n' "${src}"; return 0; }
    [[ -e "${repo}/${dst}" ]] && { echo "REFUSING: ${dst} already exists" >&2; exit 1; }
    run mkdir -p "${repo}/$(dirname "${dst}")"
    if git -C "${repo}" ls-files --error-unmatch "${src}" >/dev/null 2>&1 \
        || [[ -d "${repo}/${src}" && -n "$(git -C "${repo}" ls-files "${src}")" ]]; then
        run git -C "${repo}" mv "${src}" "${dst}"
    else
        run mv -- "${repo}/${src}" "${repo}/${dst}"
    fi
}

# edit <file> <sed expression> -- one anchored rewrite, refused if the anchor is not there. A sweep
# that matches nothing is how a call site is left pointing at a path that no longer exists.
edit() {
    local file=$1 expr=$2 before
    [[ -f "${file}" ]] || { echo "REFUSING: no ${file} to rewrite" >&2; exit 1; }
    before=$(sed "${expr}" "${file}")
    if [[ "${before}" == "$(cat "${file}")" ]]; then
        echo "REFUSING: ${expr} changes nothing in ${file}" >&2
        exit 1
    fi
    if [[ "${EXECUTE}" == 1 ]]; then
        printf '%s\n' "${before}" >"${file}"
    else
        printf 'would rewrite: %s  [%s]\n' "${file}" "${expr}"
    fi
}

phase_envs() {
    echo "== envs: ${SRC}/.env.<arm> -> ${DEST}/<experiment>/<group>/"
    refuse_if_live "move the arm envs"
    local moved=0 env group
    for env in "${SRC}"/.env.*; do
        # A symlink here is this script's own compat shim from an earlier run; a second run is a
        # no-op rather than a link pointing at itself.
        [[ -L "${env}" ]] && { printf 'already migrated: %s\n' "${env##*/}"; continue; }
        [[ -f "${env}" ]] || continue
        group=$(group_of "${env}") || { printf 'skip (no identity): %s\n' "${env##*/}"; continue; }
        run mkdir -p "${DEST}/${group}"
        run git mv "${env}" "${DEST}/${group}/${env##*/}"
        run ln -s "campaigns/${group}/${env##*/}" "${env}"
        moved=$((moved + 1))
    done
    echo "moved ${moved} arm env(s) into ${DEST}"
    echo "  ${SRC} keeps the launchers: they source arm_nodes.sh, record_identity.sh, skill_args.sh"
    echo "  and pin_env_kv.sh as \`. ./x.sh\` after cd'ing to their own directory."
}

# The identity an arm records IS the folder. Anything that records none is a generator seed or a
# smoke: neither is a submission group, so neither gets a campaign directory.
group_of() {
    local env=$1 experiment device packet
    experiment=$(grep -m1 '^HPCAGENT_BENCH_RECORD_EXPERIMENT=' "${env}" | cut -d= -f2- || true)
    [[ -n "${experiment}" ]] || return 1
    device=$(grep -m1 '^HPCAGENT_BENCH_RECORD_DEVICE=' "${env}" | cut -d= -f2- || true)
    packet=$(grep -m1 '^HPCAGENT_BENCH_RECORD_PACKET=' "${env}" | cut -d= -f2- || true)
    # The PACKET IS A TAG, not a folder. A cpf arm and its no-packet control are compared against
    # each other, and filing the treatment apart from its control is the one arrangement that makes
    # the pair hard to find. Device IS a folder: a CPU arm and a GPU arm share no control.
    # ${packet} is read above so a packet that records nothing still fails the identity check.
    : "${packet}"
    printf '%s/%s\n' "${experiment}" "${device:-cpu}"
}

# PHASE folders -- one folder per experiment, then per device, in the artifact repo.
#
# ORDER IS experiment / device, and the packet is a tag INSIDE the folder rather than a folder of
# its own: llr40/cpu holds the base, lang-skills, cpf and cpfsrc arms because they share one
# control and are read against each other, and a cpfsrc/ folder would file a treatment away from
# the control it is measured against. Device IS a folder because a CPU arm and a GPU arm share no
# control: they are different measurements of the same roster.
phase_folders() {
    echo "== folders: ${ICLR26}/${PA}"
    refuse_if_live "restructure the artifact folders"
    # The call sites are rewritten FIRST, against the paths as they stand, so a dry run reports the
    # same set of edits an execute would apply. The moves are what cannot be un-run, so they go
    # last: an interrupt then leaves edited scripts and an unmoved tree, which the next run's
    # "changes nothing" refusal names instead of half-applying.
    echo "-- call sites"
    edit "${ICLR26}/${PA}/collect_campaign.sh" 's|\$here/llr40/extract_llr40.py|$here/extract_llr40.py|'
    edit "${ICLR26}/${PA}/cpf-llr-focus40/collect.sh" 's|\.\./collect_campaign\.sh|../../collect_campaign.sh|'
    edit "${ICLR26}/${PA}/gpu-llr-focus40/collect.sh" 's|\.\./collect_campaign\.sh|../../collect_campaign.sh|'
    edit "${ICLR26}/paper_artifacts/reproduce.sh" \
        's|^for exp in cpf-llr-focus40 gpu-llr-focus40 git-scicomp; do|for exp in llr40/cpu llr40/gpu git-scicomp; do|'
    # timings/ is regenerated output and was ignored one level down; llr40's is now two.
    edit "${ICLR26}/.gitignore" 's|^paper_artifacts/experiments/\*/timings/$|paper_artifacts/experiments/**/timings/|'
    # collect_lowerings.py stays with the archived generation and imports the extractor as a
    # SIBLING. pytest and a bare `python collect_lowerings.py` both put the script's own directory
    # on the path and nothing else, so the import has to name where the extractor went.
    edit "${ICLR26}/${PA}/llr40/collect_lowerings.py" \
        's|^from extract_llr40 import|sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))\n\nfrom extract_llr40 import|'
    edit "${ICLR26}/${PA}/llr40/README.md" 's|/bin/python extract_llr40\.py|/bin/python ../../extract_llr40.py|g'

    echo "-- moves"
    # extract_llr40.py is the shared collector -- collect_campaign.sh calls it for every experiment,
    # not just llr40 -- so it moves up beside its caller before the directory it sits in is
    # archived. It imports nothing but the standard library.
    move "${ICLR26}" "${PA}/llr40/extract_llr40.py" "${PA}/extract_llr40.py"
    # The v9/v10/v11 llr40 artifact is a finished generation whose run roots were purged; its
    # README marks the rows that now exist nowhere else. It is archived under its own name so the
    # llr40 name is free for the experiment, and NOTHING in it is rewritten.
    move "${ICLR26}" "${PA}/llr40" "${PA}/archive/llr40"
    move "${ICLR26}" "${PA}/cpf-llr-focus40" "${PA}/llr40/cpu"
    move "${ICLR26}" "${PA}/gpu-llr-focus40" "${PA}/llr40/gpu"
    run mkdir -p "${ICLR26}/${PA}/scicomp-focus40/data" "${ICLR26}/${PA}/scicomp-focus40/figures"
}

# PHASE dirs -- the scratch campaign artefacts, W8.
#
# forms and dropin hold DIFFERENT artefacts under the same file names and both stay: a read form is
# what the judge serves, a drop-in is a head-start source in ABI argument order. What goes is the
# flat root that sorted a campaign's three kinds apart from each other.
#
# TABLE, not a loop over a glob: every row is a decision. cpf-forms-cpu-scicomp-focus40.contaminated
# and cpf-figs are deliberately absent -- they are proposed for deletion, and deleting is the one
# thing this script does not do.
dirs_table() {
    cat <<'EOT'
cpf-forms-cpu-llr-focus40       campaigns/llr-focus40/cpu/forms
cpf-dropin-cpu-llr-focus40      campaigns/llr-focus40/cpu/dropin
cpf-forms-gpu-llr-focus40       campaigns/llr-focus40/gpu/forms
cpf-forms-cpu-scicomp-focus40   campaigns/scicomp-focus40/cpu/forms
cpf-llr-focus40-owed            campaigns/llr-focus40/cpu/owed
gpu-llr-focus40-owed            campaigns/llr-focus40/gpu/owed
llr-focus40-owed                campaigns/llr-focus40/owed
cpf-snap-scicomp40              campaigns/scicomp-focus40/snapshot
EOT
}

phase_dirs() {
    echo "== dirs: ${SCRATCH}/<flat> -> ${SCRATCH}/campaigns/<tag>/<target>/<kind>"
    refuse_if_live "move the campaign directories"
    local src dst
    while read -r src dst; do
        [[ -n "${src}" ]] || continue
        refuse_if_warm "${SCRATCH}/${src}"
    done < <(dirs_table)
    echo "-- call sites"
    edit experiments/submit-cpf-llr40.sh \
        's|\${SCRATCH:?}/cpf-dropin-\${target}-\${TAG}|$(campaign_dir "${TAG}" dropin "${target}")|'
    edit experiments/submit-cpf-llr40.sh \
        's|\${SCRATCH:?}/cpf-forms-\${target}-\${TAG}|$(campaign_dir "${TAG}" forms "${target}")|'
    edit experiments/submit-scicomp-dc.sh \
        's|\${SCRATCH:?}/cpf-forms-cpu-\${RECORD_EXPERIMENT}|$(campaign_dir "${RECORD_EXPERIMENT}" forms cpu)|'
    edit experiments/submit-gpu-smoke5.sh \
        's|\${SCRATCH:?}/cpf-forms-gpu-llr-focus40|$(campaign_dir llr-focus40 forms gpu)|'
    edit experiments/submit-next-wave.sh \
        's|\${SCRATCH:?}/llr-focus40-owed|$(campaign_dir "${TAG}" owed)|'
    edit experiments/submit-next-wave.sh \
        's|\${SCRATCH:?}/cpf-dropin-cpu-llr-focus40|$(campaign_dir "${TAG}" dropin cpu)|'
    edit experiments/preflight_gpu.sh 's|\${SCRATCH}/cpf-forms-gpu-\${TAG}|$(campaign_dir "${TAG}" forms gpu)|'
    edit experiments/prerender_both.sbatch \
        's|"\${SCRATCH}/cpf-forms-\${target}-\${TAG}"|"$(campaign_dir "${TAG}" forms "${target}")"|g'
    # Each of those now calls campaign_dir, so each has to source the file that defines it. The
    # anchor differs per launcher because the set of siblings each one sources differs; naming the
    # line rather than guessing one is why this is a table.
    sourceline() {
        local file=$1 anchor=$2
        grep -q 'campaign_dirs\.sh' "${file}" && return 0
        edit "${file}" "0,\|^${anchor}\$|s||&\n${3}|"
    }
    sourceline experiments/submit-cpf-llr40.sh '\. \./arm_nodes\.sh' '. ./campaign_dirs.sh'
    sourceline experiments/submit-scicomp-dc.sh '\. \./arm_nodes\.sh' '. ./campaign_dirs.sh'
    sourceline experiments/submit-gpu-smoke5.sh '\. \./arm_nodes\.sh' '. ./campaign_dirs.sh'
    sourceline experiments/submit-next-wave.sh 'cd -- "\$(dirname -- "\${BASH_SOURCE\[0\]}")"' '. ./campaign_dirs.sh'
    sourceline experiments/preflight_gpu.sh 'source \./roster\.sh' '. ./campaign_dirs.sh'
    sourceline experiments/prerender_both.sbatch 'source "\${SD}/roster\.sh"' '. "${SD}/campaign_dirs.sh"'

    echo "-- moves"
    while read -r src dst; do
        [[ -n "${src}" ]] || continue
        if [[ ! -e "${SCRATCH}/${src}" ]]; then printf 'absent: %s\n' "${src}"; continue; fi
        if [[ -L "${SCRATCH}/${src}" ]]; then printf 'already migrated: %s\n' "${src}"; continue; fi
        [[ -e "${SCRATCH}/${dst}" ]] && { echo "REFUSING: ${dst} already exists" >&2; exit 1; }
        run mkdir -p "${SCRATCH}/$(dirname "${dst}")"
        run mv -- "${SCRATCH}/${src}" "${SCRATCH}/${dst}"
        # COMPAT SHIM. Every generated .env under experiments/ names the OLD absolute directory and
        # a queued job reads its env by path at start, so the old string has to keep resolving
        # until the last pre-move job leaves the queue. Untracked and TEMPORARY: --cleanup-dirs.
        run ln -s "${SCRATCH}/${dst}" "${SCRATCH}/${src}"
    done < <(dirs_table)

    if [[ "${EXECUTE}" == 1 ]]; then
        echo "-- verifying no launcher still spells a flat path"
        local left
        left=$(git grep -nI 'cpf-forms-\|cpf-dropin-\|llr-focus40-owed' -- experiments ":!${SELF}" \
            ":!experiments/.env.*" || true)
        if [[ -n "${left}" ]]; then
            echo "REFUSING TO CLAIM DONE: the sweep missed a spelling" >&2
            printf '%s\n' "${left}" >&2
            exit 1
        fi
    fi
    echo "  experiments/.env.<arm> are GENERATED and frozen: they keep the old absolute paths on"
    echo "  purpose and resolve through the compat symlinks. The generators above are the fix."
}

# CLEANUP: remove the compat symlinks once no pre-move job is left in the queue. Each link is
# checked to BE a link and to point at the new tree before it is touched, in a pass of its own:
# validating inside the removal loop deletes whatever sorted before the offender and only then
# refuses, which is the one state worse than either doing it or not.
phase_cleanup() {
    refuse_if_live "remove the envs compat symlinks"
    local link target shims=()
    for link in "${SRC}"/.env.*; do
        [[ -L "${link}" ]] || continue
        target=$(readlink -- "${link}")
        [[ "${target}" == campaigns/* ]] || { echo "REFUSING: ${link} -> ${target}" >&2; exit 1; }
        [[ -f "${SRC}/${target}" ]] || { echo "REFUSING: ${link} -> ${target} is absent" >&2; exit 1; }
        shims+=("${link}")
    done
    for link in "${shims[@]+"${shims[@]}"}"; do run rm -- "${link}"; done
    echo "removed ${#shims[@]} env compat symlink(s)"
}

phase_cleanup_dirs() {
    refuse_if_live "remove the scratch compat symlinks"
    local src dst links=()
    while read -r src dst; do
        [[ -n "${src}" && -L "${SCRATCH}/${src}" ]] || continue
        [[ "$(readlink -- "${SCRATCH}/${src}")" == "${SCRATCH}/${dst}" ]] \
            || { echo "REFUSING: ${src} does not point at ${dst}" >&2; exit 1; }
        [[ -d "${SCRATCH}/${dst}" ]] || { echo "REFUSING: ${dst} is absent" >&2; exit 1; }
        links+=("${src}")
    done < <(dirs_table)
    for src in "${links[@]+"${links[@]}"}"; do run rm -- "${SCRATCH}/${src}"; done
    echo "removed ${#links[@]} scratch compat symlink(s)"
}

[[ "${EXECUTE}" == 1 ]] || echo "DRY RUN -- nothing is moved. --execute to apply."
for phase in "${PHASES[@]}"; do
    case ${phase} in
        envs) phase_envs ;;
        folders) phase_folders ;;
        dirs) phase_dirs ;;
        cleanup) phase_cleanup ;;
        cleanup-dirs) phase_cleanup_dirs ;;
    esac
    echo
done
