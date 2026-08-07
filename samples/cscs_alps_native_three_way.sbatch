#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SAMPLE -- THREE-WAY optimizer comparison over the HPC track, on CSCS Daint/Alps, NATIVE (no
# container, no EDF, no --environment flag anywhere).
#
#   pluto                             the polyhedral native column, no DaCe involved
#   dace_cpu_autoopt on `main`        upstream auto_optimize, measured on upstream DaCe
#   dace_cpu_autoopt on `extended`    the SAME upstream optimizer, measured on the fork
#
# The two autoopt cells are the point: identical pipeline, identical kernel, identical preset, so
# whatever differs between them is the DaCe underneath rather than the optimizer. Pluto is the
# outside reference that says whether either DaCe is competitive at all.
#
#   HPCAGENT_BENCH_ENV=$SCRATCH/hpcagent-env.sh DACE_MAIN=$SCRATCH/dace-main \
#       DACE_EXTENDED=$SCRATCH/dace-extended \
#       sbatch -A <account> samples/cscs_alps_native_three_way.sbatch
#
# `-A <account>` is mandatory on Alps (or export SBATCH_ACCOUNT first).
#
# NATIVE means nothing is supplied by an image. You provide, and the job refuses to start without:
#   HPCAGENT_BENCH_ENV   a script this sources -- the site `module load`s plus the venv activate
#                        for the python that has hpcagent-bench and its toolchain installed
#   DACE_MAIN            a DaCe checkout on `main`, put on PYTHONPATH (repo root, not its dace/)
#   DACE_EXTENDED        a DaCe checkout on `extended`, likewise
# The trees may be the SAME clone: DACE_CHECKOUT=1 lets the script switch branches between stages
# (refused on a dirty tree). Two clones are better -- no switching, no rebuild churn.
#
# numpy is NOT optional: `hpcagent-bench plot` builds its speedup table against the numpy column.
# It rides in the reference stage below, ONCE, so it is not timed twice.
#
# Overridable: BENCH, PRESET, RANKS_PER_NODE, DACE_BUILD_ROOT, DACE_CHECKOUT, plus anything
# submit_deterministic takes. All the real work -- sharding, preflight, thread split, per-rank DB
# shards, the merged failure count -- is that script's; this file is a stage table and a loop.
#
# UNVERIFIED AGAINST THE SITE'S OWN SUBMISSION SCRIPTS, exactly as
# scripts/cscs/submit_loop_level_reasoning_alps.sbatch says of itself. The Alps specifics here are that
# script's, minus its container plumbing; they come from this repo's notes
# (scripts/cscs/env.toml.example, docs/launch.md), which state that node allocation and srun
# submission are owned externally. The partition name, the account and the scratch layout are the
# three things most likely to need a local edit before this runs as-is. No --partition line is set
# for that reason.

#SBATCH --job-name=hpcagent-alps-native-three-way
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4         # MUST equal RANKS_PER_NODE below: this is what allocates the
                                    # ranks, and srun cannot launch more tasks than were allocated
#SBATCH --exclusive                 # whole nodes: the ranks split them, no foreign co-runners
#SBATCH --time=12:00:00
#SBATCH --output=results/alps-native-three-way-%j.out

set -euo pipefail

REPO="${HPCAGENT_BENCH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO}"
source "${REPO}/scripts/dace_branch.sh"
source "${REPO}/scripts/cscs/native_env.sh"

export HPCAGENT_BENCH_REPO="${REPO}"
export BENCH="${BENCH:-scientific_computing}"        # every kernel in the HPC track -- 132 of them
export PRESET="${PRESET:-L}"
# Only a fallback: inside an allocation the launcher reads SLURM_NTASKS_PER_NODE, which is what the
# #SBATCH line above actually reserved. Kept equal to it so the two cannot disagree.
export RANKS_PER_NODE="${RANKS_PER_NODE:-4}"

require_native_env
require_dace_tree DACE_MAIN
require_dace_tree DACE_EXTENDED

# "<build> <dace tree> <columns>". `build` is the branch the stage measures AND the stamp put on
# every row it writes, so a row can never claim a tree it was not measured on.
#
# `none` is the reference stage: columns that do not import DaCe at all. It is a stage of its own
# because HPCAGENT_BENCH_RECORD_BUILD is read for EVERY framework, not just the DaCe ones, and
# plot folds a non-null build into the series name -- a numpy row stamped `main` plots as
# `numpy/main`, which is not the baseline plot divides by. Unstamped here, so it is.
STAGES=("none ${DACE_MAIN} numpy,pluto"
        "main ${DACE_MAIN} dace_cpu_autoopt"
        "extended ${DACE_EXTENDED} dace_cpu_autoopt")

BASE_PYTHONPATH="${PYTHONPATH:-}"
# DaCe writes one .so per SDFG variant per kernel. Off /tmp, which is tmpfs on these nodes (the
# build would compete with the run for RAM). $SCRATCH on Alps is still the parallel FS -- the flash
# tier (/iopsstor) handles thousands of small compiler writes better than /capstor, so point
# DACE_BUILD_ROOT there if both are mounted. UNVERIFIED: only the mount names are known here, from
# scripts/cscs/env.toml.example.
BUILD_ROOT="${DACE_BUILD_ROOT:-${SCRATCH:-${REPO}}}"
failed=()

for stage in "${STAGES[@]}"; do
    read -r build tree columns <<<"${stage}"
    if [[ "${build}" == "none" ]]; then
        stamp=""                    # tree-independent columns: unstamped, so `numpy` stays `numpy`
    else
        if ! ensure_branch "${tree}" "${build}"; then
            failed+=("${build}")
            echo "=== stage ${build} SKIPPED: wrong branch ===" >&2
            continue
        fi
        stamp="${build}"
        evict_base_sdfg_cache "${REPO}" cpu
    fi
    # PYTHONPATH from the SAVED base, never from the last iteration's: appending would leave both
    # trees on the path and the first one found would win every stage after the first.
    export PYTHONPATH="${tree}${BASE_PYTHONPATH:+:${BASE_PYTHONPATH}}"
    export FRAMEWORKS="${columns}"
    export RUN_TAG="${build}"
    export HPCAGENT_BENCH_RECORD_BUILD="${stamp}"
    # A build cache PER TREE. One shared cache would compile against whichever tree's headers got
    # there first, and the resulting binary is neither tree's -- a wrong measurement that builds.
    export DACE_BUILD_ROOT="${BUILD_ROOT}/${build}"

    # Every stage RUNS even if an earlier one failed, and the job still fails at the end. The
    # comparison this job exists for needs both autoopt halves, so losing one silently is the worst
    # outcome; a crash in one must not cost the others their allocation.
    echo "=== stage ${build}: ${columns} (dace=${tree}) ==="
    if ! bash "${REPO}/scripts/submit_deterministic.sbatch"; then
        failed+=("${build}")
        echo "=== stage ${build} FAILED; continuing with the remaining stages ===" >&2
    fi
done

# The per-rank DBs of every stage into one file. Rebuilt from scratch each time, so merging twice
# cannot double the rows; forced here so the artifact to copy off the cluster exists whether or not
# anything reads it.
echo "=== merging the rank DBs ==="
hpcagent-bench aggregate-db

# Signed relative change (0 = no change, +1 = 2x faster, -1 = 2x slower), banded by order of
# magnitude. NOT the old ratio-axis table, which reads a 0.5x regression as smaller than a 1.5x
# win; that one is opt-in now (`hpcagent-bench plot`). --no-usetex: a compute node rarely has LaTeX.
echo "=== speed-up chart ==="
python "${REPO}/scripts/plot_speedup.py" --benchmark "${BENCH}" --preset "${PRESET}" --no-usetex \
    --output "results/plots/alps-native-three-way-${SLURM_JOB_ID:-local}.pdf"

if (( ${#failed[@]} )); then
    echo "stages with failures: ${failed[*]}" >&2
    exit 1
fi
