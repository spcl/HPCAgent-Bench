#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Sourceable helper: the two things a NATIVE (no-container) multi-tree DaCe job has to get right.
#
#   source "${REPO}/scripts/cscs/native_env.sh"
#   require_native_env            # site env + hpcagent-bench on PATH, or exit
#   require_dace_tree DACE_MAIN   # the tree exists and actually holds a dace package
#   evict_base_sdfg_cache "${REPO}" cpu
#
# Its own file for the same reason as scripts/dace_branch.sh: two samples need all three, and a
# second copy is how they drift. Nothing here is CSCS-specific except where it is documented to be;
# it lives under scripts/cscs/ because the native Alps samples are its only callers so far.

# A container supplies the toolchain and the python env; native mode does not, and a compute node
# starts with neither. Fail HERE, by name, rather than as an import error on rank 3 of an 8-node
# allocation that has already been charged for.
require_native_env() {
    if [[ -n "${HPCAGENT_BENCH_ENV:-}" ]]; then
        if [[ ! -f "${HPCAGENT_BENCH_ENV}" ]]; then
            echo "HPCAGENT_BENCH_ENV=${HPCAGENT_BENCH_ENV} does not exist" >&2
            return 2
        fi
        # `set -u` off across the source: site `module` shell functions and venv activate scripts
        # routinely read unset variables, and the caller runs with -euo pipefail.
        set +u
        # shellcheck disable=SC1090  # the site env script is chosen at submit time, not known here
        source "${HPCAGENT_BENCH_ENV}"
        set -u
    fi
    if ! command -v hpcagent-bench >/dev/null 2>&1; then
        echo "hpcagent-bench is not on PATH." >&2
        echo "  native mode has no image to supply it: load the site modules and activate the venv" >&2
        echo "  that has it installed, from a script named by HPCAGENT_BENCH_ENV:" >&2
        echo "      HPCAGENT_BENCH_ENV=\$SCRATCH/hpcagent-env.sh sbatch -A <account> ..." >&2
        return 2
    fi
}

# Native mode puts a DaCe tree on PYTHONPATH instead of installing it, so "the directory exists" is
# not enough -- a path off by one level imports the SITE's dace and the run is attributed to a tree
# it never used. Checked for the package, not just the directory.
require_dace_tree() {
    local var="$1" tree="${!1:-}"
    if [[ ! -d "${tree}" ]]; then
        echo "${var} must point at a DaCe checkout (got '${tree}')" >&2
        echo "  DACE_MAIN=<upstream dace> DACE_EXTENDED=<spcl/dace@extended> sbatch ..." >&2
        return 2
    fi
    if [[ ! -f "${tree}/dace/__init__.py" ]]; then
        echo "${var}=${tree} has no dace/__init__.py; PYTHONPATH wants the REPO ROOT, not dace/" >&2
        return 2
    fi
}

# The per-kernel base-SDFG cache (<kernel>/.cache/<module>_<tag>.sdfgz) is fingerprinted on the
# kernel sources and the precision ONLY -- not on which DaCe parsed it (dace_framework
# _sdfg_fingerprint). Two trees in one job therefore collide: whichever stage runs first seeds the
# cache and the second silently measures its own pipelines over the FIRST tree's parse. A separate
# DACE_BUILD_ROOT does not cover this -- that separates the compiled .so, and this cache sits in the
# repo, above it. Evicted before each DaCe stage so every tree parses with its own frontend; a miss
# is a rebuild, so dropping it is only ever a cost.
evict_base_sdfg_cache() {
    local repo="$1" tag="${2:-cpu}"
    find "${repo}/hpcagent_bench/benchmarks" \
        \( -path "*/.cache/*_${tag}.sdfgz" -o -path "*/.cache/*_${tag}.sdfgz.fp" \) -delete
}
