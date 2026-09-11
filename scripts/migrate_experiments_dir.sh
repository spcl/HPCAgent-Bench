#!/usr/bin/env bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# ONE-SHOT: containers/cluster/example-script -> experiments, and the campaign samples in with it.
#
# The directory was named for what it demonstrated in 2024. It now holds 101 campaign arms, nine
# submit-*.sh launchers, beverin.sbatch, agent_driver.py, make_problems.py and the kernel lists --
# it is the experiments directory, and nothing about it is an example, a script, or a container.
# samples/llr40_*.sh are four-line wrappers that exec submit-*.sh INSIDE it, so they move next to
# what they call; samples/*.sbatch demo the launcher itself and stay where they are.
#
# WHY THIS IS A SCRIPT AND NOT FIVE COMMANDS. beverin.sbatch resolves SCRIPT_DIR at RUNTIME and
# execs run_cluster.sh out of it, so a live arm holds the path open -- renaming under a running
# campaign takes the job's own launcher out from under it, on a machine where an arm costs hours of
# 3-node allocation. The guards below are the point of the file; the moves are the easy part.
set -euo pipefail
cd -- "$(git rev-parse --show-toplevel)"

OLD=containers/cluster/example-script
NEW=experiments
DRY=${DRY:-0}
FORCE=${FORCE:-0}   # deliberate override; see COMPAT below -- it is what makes FORCE survivable
SELF=scripts/migrate_experiments_dir.sh   # names the old path in its own comments; never sweep it

run() { if [[ "${DRY}" == 1 ]]; then printf 'would: %s\n' "$*"; else "$@"; fi; }

# GUARD 1: a running job holds SCRIPT_DIR open. Image builds are exempt -- they never read the
# campaign tree; anything else might.
# DRY=1 changes nothing, so it skips the guards on purpose: a preview you can only take once it is
# already safe to run is not a preview, and reviewing the plan is most useful while the guards bite.
live=$(squeue -u "${USER}" -h -o '%j' 2>/dev/null | grep -v '^build-' | wc -l)
if [[ "${live}" -gt 0 && "${DRY}" != 1 && "${FORCE}" != 1 ]]; then
    echo "REFUSING: ${live} non-build job(s) in flight; a live arm execs ${OLD}/run_cluster.sh" >&2
    squeue -u "${USER}" -o '%.10i %.34j %.9T %.10M %.5D' >&2
    exit 1
fi

# GUARD 2: uncommitted work in the tree being moved belongs to whoever is mid-edit. git mv would
# carry it along silently and a rebase would land it under a path they never wrote.
dirty=$(git status --porcelain -- "${OLD}" | wc -l)
if [[ "${dirty}" -gt 0 && "${DRY}" != 1 && "${FORCE}" != 1 ]]; then
    echo "REFUSING: ${dirty} uncommitted path(s) under ${OLD}; commit or stash them first" >&2
    git status --short -- "${OLD}" >&2
    exit 1
fi

echo "moving ${OLD} -> ${NEW} ($(git ls-files "${OLD}" | wc -l) tracked files)"
run git mv "${OLD}" "${NEW}"

# COMPAT SHIM -- this is what makes FORCE=1 safe rather than merely permitted.
# A running job resolved SCRIPT_DIR to the OLD absolute path at submit time and keeps re-resolving
# against it for the rest of its life: srun steps (run_cluster.sh:943) and, at teardown,
# monitor_report.py / token_report.py / recoverable_report.py (:1002,:1011,:1017). git mv is a
# rename(), so fds already open follow the inode -- but a path resolved fresh from the old string
# would 404 and the job would finish ungraded. The symlink keeps that string valid. It is untracked
# and TEMPORARY: delete it once the last pre-migration job leaves the queue.
if [[ "${FORCE}" == 1 || "${DRY}" == 1 ]]; then
    echo "compat: ${OLD} -> symlink, so in-flight jobs keep resolving their SCRIPT_DIR"
    run ln -s "$(realpath -m --relative-to="$(dirname "${OLD}")" "${NEW}")" "${OLD}"
fi

echo "folding the campaign wrappers in beside the launchers they exec"
run mkdir -p "${NEW}/samples"
for wrapper in samples/llr40_*.sh; do
    [[ -e "${wrapper}" ]] || continue
    run git mv "${wrapper}" "${NEW}/samples/$(basename "${wrapper}")"
done

# THREE spellings reach this directory and a fixed-string sweep sees only the first. Eight files
# build the path from pathlib segments (paths.ROOT / "containers" / "cluster" / "example-script"),
# which shares not one substring with the literal; one more anchors off containers/ instead of the
# repo root, so its parents[] index has to shift as well. The rules are BRE for both git grep and
# sed, so metacharacters are escaped once and used twice. -I skips binaries; without it the figures
# would be rewritten as garbage.
sweep() {
    local pattern=$1 replacement=$2 hits
    mapfile -t hits < <(git grep -lI "${pattern}" -- . ":!${SELF}" || true)
    [[ "${#hits[@]}" -gt 0 ]] || return 0
    printf '  %s\n' "${hits[@]}"
    [[ "${DRY}" == 1 ]] || sed -i "s|${pattern}|${replacement}|g" "${hits[@]}"
}

echo "sweeping references"
sweep "${OLD}" "${NEW}"
sweep '"containers" / "cluster" / "example-script"' '"experiments"'
sweep 'TOOLS_DIR\.parents\[1\] / "cluster" / "example-script"' 'TOOLS_DIR.parents[2] / "experiments"'

# The assertion is what makes that list COMPLETE rather than merely long. Enumerating spellings is
# a judgement call and I already got it wrong once; this turns it into a check. A fourth spelling
# fails the script here, loudly, instead of leaving a test importing a path that no longer exists.
if [[ "${DRY}" != 1 ]]; then
    echo "verifying no spelling of the old name survives"
    left=$(git grep -nI 'example-script' -- . ":!${SELF}" || true)
    if [[ -n "${left}" ]]; then
        echo "REFUSING TO CLAIM DONE: the sweep missed a spelling" >&2
        printf '%s\n' "${left}" >&2
        exit 1
    fi
fi

echo
echo "done. now: pre-commit run --all-files && pytest -q --maxfail=20 tests/"
echo "most of the rewritten files are tests -- they are the check that the sweep was correct."
