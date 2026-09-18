#!/usr/bin/env bash
# Moves the pre-unification cache locations (repo .cache/{generated,packs}, ad hoc ${SCRATCH}
# pip/spack dirs) onto the ONE root scripts/cache_env.sh now derives everything from
# (HPCAGENT_BENCH_CACHE, default ${SCRATCH}/.hpcagentbench-cache).
#
# DRY RUN BY DEFAULT. Prints the plan and touches nothing unless DRY_RUN=0.
#
#   ./scripts/migrate_cache_layout.sh              print the plan
#   DRY_RUN=0 ./scripts/migrate_cache_layout.sh     actually move things
#   FORCE=1 DRY_RUN=0 ./scripts/migrate_cache_layout.sh   ... even with jobs in squeue
#
# What this script does NOT touch, on purpose:
#   - HPCAGENT_BENCH_WEIGHTS_DIR / HF_HOME (weights). Its default did not change in this pass.
#   - The JIT engine tree (.triton/.vllm/.aiter/.inductor/.xdg/.home/.torch-ext) and
#     .cpf-prerender/tools/runs/results. HPCAGENT_BENCH_CACHE resolves to the exact same path
#     JIT_CACHE_ROOT already did, so there is nothing to move for any of these.
#   - base-images/ce-images. ce-images/ is where a PROMOTED image an EDF may currently point at
#     lives, not only build scratch -- see .cache/README.md's "Deferred" note. This script only
#     PRINTS that decision is still open; it never proposes a move for them.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${HPCAGENT_BENCH_REPO:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
DRY_RUN="${DRY_RUN:-1}"
FORCE="${FORCE:-0}"

# shellcheck source=./cache_env.sh
. "${REPO}/scripts/cache_env.sh"

if [[ "${DRY_RUN}" != "1" ]]; then
    if command -v squeue >/dev/null 2>&1; then
        running="$(squeue -h -u "${USER:-$(id -un)}" 2>/dev/null | wc -l)"
        if [[ "${running}" -gt 0 && "${FORCE}" != "1" ]]; then
            echo "REFUSING: ${running} job(s) in squeue for ${USER:-$(id -un)}. A running job may be" >&2
            echo "reading the pre-migration paths right now (see .cache/README.md's live-data warning)." >&2
            echo "Re-run with FORCE=1 once nothing is running, or wait for the queue to drain." >&2
            exit 1
        fi
    else
        echo "note: no squeue on this host; cannot check for running jobs" >&2
    fi
fi

# same_fs <a> <b>: true if the two paths (or their nearest existing ancestor) share a device id.
same_fs() {
    local a="$1" b="$2"
    while [[ ! -e "${a}" ]]; do a="$(dirname -- "${a}")"; done
    while [[ ! -e "${b}" ]]; do b="$(dirname -- "${b}")"; done
    [[ "$(stat -c %d -- "${a}")" == "$(stat -c %d -- "${b}")" ]]
}

# Destinations claimed by an earlier plan_move call in THIS run, so two pre-unification sources
# that collapse onto the same new path (pip-cache and .cache/pip both -> the one pip/ dir) are
# flagged instead of the second one silently merging (DRY_RUN=1) or racing a real mv (DRY_RUN=0).
declare -A PLANNED_DEST

# plan_move <old> <new> <label>: print what would happen (mv same-fs, rsync+verify cross-fs), and
# do it when DRY_RUN=0. Skips silently when <old> does not exist -- nothing to migrate.
plan_move() {
    local old="$1" new="$2" label="$3"
    if [[ ! -e "${old}" ]]; then
        printf '  [skip]  %-12s %s (does not exist)\n' "${label}" "${old}"
        return 0
    fi
    if [[ -n "${PLANNED_DEST[${new}]:-}" ]]; then
        printf '  [SKIP]  %-12s %s -> %s : COLLIDES with %s already planned for this destination -- merge by hand\n' \
            "${label}" "${old}" "${new}" "${PLANNED_DEST[${new}]}"
        return 0
    fi
    if [[ -e "${new}" ]]; then
        printf '  [SKIP]  %-12s %s -> %s : DESTINATION ALREADY EXISTS, resolve by hand\n' \
            "${label}" "${old}" "${new}"
        return 0
    fi
    PLANNED_DEST["${new}"]="${old}"
    local size
    size="$(du -sh -- "${old}" 2>/dev/null | cut -f1)"
    if same_fs "${old}" "${new%/*}"; then
        printf '  [mv]    %-12s %s (%s) -> %s\n' "${label}" "${old}" "${size:-?}" "${new}"
        if [[ "${DRY_RUN}" != "1" ]]; then
            mkdir -p -- "$(dirname -- "${new}")"
            mv -- "${old}" "${new}"
        fi
    else
        printf '  [rsync] %-12s %s (%s) -> %s (cross-filesystem, then verify + remove source)\n' \
            "${label}" "${old}" "${size:-?}" "${new}"
        if [[ "${DRY_RUN}" != "1" ]]; then
            mkdir -p -- "${new}"
            rsync -a -- "${old}/" "${new}/"
            if diff -rq -- "${old}" "${new}" >/dev/null 2>&1; then
                rm -rf -- "${old}"
            else
                echo "    VERIFY FAILED for ${label}: ${old} left in place, ${new} not removed" >&2
            fi
        fi
    fi
}

printf 'root:      %s\nweights:   %s (unchanged)\nmode:      %s\n\n' \
    "${HPCAGENT_BENCH_CACHE}" "${HF_HOME}" "$([[ "${DRY_RUN}" == "1" ]] && echo "DRY RUN" || echo "LIVE")"

echo "=== repo-relative caches -> the unified root ==="
plan_move "${REPO}/.cache/generated" "${HPCAGENT_BENCH_GENERATED_CACHE_HOST}" generated
plan_move "${REPO}/.cache/packs" "${HPCAGENT_BENCH_PACK_ROOT}" packs

echo
echo "=== ad hoc \${SCRATCH} caches -> the unified root ==="
plan_move "${SCRATCH:-}/pip-cache" "${HPCAGENT_BENCH_PIP_CACHE_DIR}" pip-cache
plan_move "${SCRATCH:-}/.cache/pip" "${HPCAGENT_BENCH_PIP_CACHE_DIR}" pip-cache-login
plan_move "${SCRATCH:-}/spack-buildcache" "${HPCAGENT_BENCH_SPACK_BUILDCACHE_DIR}" spack-buildcache
plan_move "${SCRATCH:-}/.tmp" "${HPCAGENT_BENCH_TMP_DIR}" tmp

echo
echo "=== NOT moved by this script (see the header) ==="
printf '  JIT tree (.triton/.vllm/.aiter/.inductor/.xdg/.home/.torch-ext), .cpf-prerender,\n'
printf '  tools/, runs/, results/: already under %s -- HPCAGENT_BENCH_CACHE is the same root\n' "${HPCAGENT_BENCH_CACHE}"
printf '  JIT_CACHE_ROOT already resolved to. Nothing to do.\n'
printf '  base-images (%s) / ce-images (%s): DEFERRED, open decision -- see .cache/README.md.\n' \
    "${SCRATCH:-}/base-images" "${SCRATCH:-}/ce-images"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo
    echo "DRY RUN. Nothing was moved. Re-run with DRY_RUN=0 to apply this plan."
fi
