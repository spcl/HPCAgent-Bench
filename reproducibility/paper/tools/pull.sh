#!/usr/bin/env bash
# Pull HPCAgent-Bench data from the cluster into ONE local mirror ($MIRROR), incrementally: re-run any
# time, only new or changed files move. Read-only on the cluster: nothing is created, moved or deleted
# there (the project sits at its file quota; staging on scratch stalls every stat/open). Nothing is
# deleted locally either (no --delete), which is how frozen jobs keep their data.
#   tools/pull.sh          update $MIRROR
#   ZIP=1 tools/pull.sh    also write SHA256SUMS and <mirror>-<date>.zip beside the mirror
# Env: tools/cluster.env (see cluster.env.example); TRIES, RETRY_SLEEP tune the retries.
# Then: MIRROR=... experiments/paper/reproduce.sh --extract
set -euo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_<host>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
. "$(dirname "${BASH_SOURCE[0]}")/cluster.sh"
case ${1:-} in
    -h | --help) usage; exit 0 ;;
    '') ;;
    *) echo "$(basename "$0"): unknown argument '$1' (see --help)" >&2; exit 2 ;;
esac
cluster_init
DEST=$MIRROR
mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)
failed=()

# The run directory keeps its cluster name: regrade DBs key a row by its path from `hpcagent-bench-runs/` on.
pull hpcagent-bench-runs hpcagent-bench-runs --prune-empty-dirs \
    --exclude='home/' --exclude='.cache/' --exclude='vllm/' --exclude='rocprof_out*/' --exclude='dacecache/' \
    --exclude='sandbox/' --include='*/' --include='.env' --include='*.resolved' --include='prompt.txt' \
    --include='observations/**' --include='*.json' --include='*.jsonl' --include='*.csv' \
    --include='*.db' --include='*.db-wal' --include='*.db-shm' --include='*.sqlite' --exclude='*' \
    || failed+=(hpcagent-bench-runs)
pull hpcagent-bench/experiments hb/experiments --exclude='__pycache__/' --exclude='*.err' --exclude='*.out' \
    || failed+=(hpcagent-bench/experiments)
pull audit-20260918/frozen-observations-0919 frozen-observations || failed+=(frozen-observations)
pull mlscale-grade mlscale-grade || failed+=(mlscale-grade)
pull ICLR26Reproducibility ICLR26Reproducibility --exclude='__pycache__/' --exclude='.git/' \
    || failed+=(ICLR26Reproducibility)
pull .hpcagentbench-cache/runs/canon canon-sweep --prune-empty-dirs --exclude='dacecache/' --exclude='dbg*/' \
    --include='*/' --include='*.csv' --include='*.db' --exclude='*' || failed+=(canon-sweep)

fetch "echo hpcagent-bench \$(git -C '$CLUSTER_SCRATCH/hpcagent-bench' rev-parse HEAD); echo pulled $(date -Is)" \
    "$DEST/COMMITS" || failed+=(COMMITS)
echo "mirror: $(du -sh "$DEST" | cut -f1) in $DEST"
# A partial mirror still updates in place, but is never zipped as a snapshot.
if ((${#failed[@]})); then
    echo "FAILED groups: ${failed[*]} (re-run to resume)" >&2
    exit 1
fi
if [[ ${ZIP:-0} == 1 ]]; then
    checksum_dir "$DEST"
    zipfile=$(dirname "$DEST")/$(basename "$DEST")-$(date +%Y%m%d-%H%M).zip
    (cd "$(dirname "$DEST")" && zip -qr "$zipfile" "$(basename "$DEST")")
    ls -la "$zipfile"
fi
echo DONE
