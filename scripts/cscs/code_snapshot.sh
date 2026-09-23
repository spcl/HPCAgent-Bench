#!/usr/bin/env bash
# code_snapshot.sh <live checkout> <dest> -- copy the checkout a job runs from to <dest>, print the
# short commit sha it holds, exit non-zero (dest untouched) if the copy failed.
#
# The live checkout is fast-forwarded while jobs queue and run, so a job copies it once at start
# (run_cluster.sh, regrade.sbatch, mlscale-grade.sbatch) and every step runs from the copy.
#
# Tracked files come from ONE commit (`git archive` of HEAD), never from the working tree: a copy
# walked while a fast-forward rewrites the tree mixes files from both commits (643369: "cannot
# import name 'decline_kind'"), and a hand edit in the live tree would ride along unrecorded.
# Everything git does not track at that commit is copied from the working tree as it stands:
# the generated benchmark siblings (*_dace.py, cpp_backend/*.so, .cache/*.sdfgz -- without them
# every job would regenerate and recompile them), the submodule contents, and the untracked arm
# envs and problems files a job names relative to experiments/. Caches, run output, core dumps and
# job logs are left out. Built beside <dest> and renamed into place, so a half-built copy is never
# run and a requeued job id replaces its old copy whole.
set -euo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_<host>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0

live="${1:?live checkout}"
dest="${2:?destination}"
partial="${dest}.partial"

sha="$(git -C "${live}" rev-parse --verify 'HEAD^{commit}')"
rm -rf -- "${partial}"
trap 'rm -rf -- "${partial}"' EXIT
mkdir -p -- "${partial}"
git -C "${live}" archive --format=tar "${sha}" | tar -xf - -C "${partial}"

# What git does not track, as git lists it (one index-aware walk, ~1 s on the live tree; an rsync
# exclude of every tracked name took minutes): untracked and ignored files minus caches, run
# output, core dumps and job logs. rsync gets names that already passed this filter -- it refuses a
# --files-from name its own excludes match.
junk='(^|/)(\.git|__pycache__|\.hpcagent_bench_cache|\.ruff_cache|\.pytest_cache|\.mypy_cache)(/|$)|^(\.cache|results|\.perf_reports|paper)/|^core_|^[^/]*\.db$'
junk+='|^experiments/(core_|mwd-final-|logs/|results/)|^experiments/[^/]*-[0-9][^/]*\.(out|err)$'
# rsync 24: a file vanished between listing and copying (a cache entry replaced); the rest is copied.
copy() {
    local rc=0
    rsync -a "$@" || rc=$?
    [[ "${rc}" == 0 || "${rc}" == 24 ]] || { echo "code_snapshot: rsync $* failed (${rc})" >&2; return 1; }
}
git -C "${live}" ls-files -z --others | { grep -zvE "${junk}" || true; } |
    copy --from0 --files-from=- "${live}/" "${partial}/"
# Submodule contents: a submodule is one commit entry to the superproject, never listed above.
while IFS= read -r -d '' module; do
    copy --exclude=.git "${live}/${module}/" "${partial}/${module}/"
done < <(git -C "${live}" ls-tree -r -z --full-tree "${sha}" | sed -z -n 's|^160000 commit [0-9a-f]*\t||p')

rm -rf -- "${dest}"
mv -- "${partial}" "${dest}"
git -C "${live}" rev-parse --short "${sha}"
