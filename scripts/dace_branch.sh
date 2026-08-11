#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Sourceable helper: verify a DaCe checkout is on the branch a measurement stage CLAIMS to measure.
#
#   source "${REPO}/scripts/dace_branch.sh"
#   ensure_branch "${DACE_MAIN}" main || ...
#
# Its own file because two samples need it (samples/npbench_dace_flavors.sbatch measures main against
# extended; samples/npbench_dace_main_vs_pluto.sbatch measures main against a non-DaCe optimizer) and a
# second copy is exactly how the two would drift into disagreeing about what "measured on main" means.

# A stage measures the branch it CLAIMS to measure, or it does not run. Getting this wrong produces
# numbers that look perfectly normal and are attributed to the wrong DaCe -- the one failure this
# check exists to rule out, so it is checked rather than assumed.
#
# Checked, not checked-out, by default: a DaCe tree is usually someone's working tree, and switching
# its branch under them is not a job script's call. DACE_CHECKOUT=1 opts in, and even then a dirty
# tree is refused rather than carried onto another branch.
ensure_branch() {
    local tree="$1" want="$2" have
    have="$(git -C "${tree}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    if [[ "${have}" == "${want}" ]]; then
        return 0
    fi
    if [[ "${DACE_CHECKOUT:-0}" != "1" ]]; then
        echo "${tree} is on '${have:-<not a git checkout>}' but this stage measures '${want}'." >&2
        echo "  switch it yourself, or re-submit with DACE_CHECKOUT=1 to let this script do it:" >&2
        echo "      git -C ${tree} checkout ${want}" >&2
        return 1
    fi
    if [[ -n "$(git -C "${tree}" status --porcelain)" ]]; then
        echo "${tree} has uncommitted changes; refusing to check out '${want}' over them." >&2
        return 1
    fi
    git -C "${tree}" checkout "${want}"
}
