# Sourced by every script: paths, the Python that runs hpcagent_bench, and the checksum check.
set -euo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_<host>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Inside the benchmark's release this folder is reproducibility/paper/, and the checkout is two levels up.
[[ -z ${HPCAGENT_BENCH:-} && -d $ROOT/../../hpcagent_bench ]] && HPCAGENT_BENCH=$(cd "$ROOT/../.." && pwd)
: "${HPCAGENT_BENCH:?set HPCAGENT_BENCH to the hpcagent-bench checkout at the paper-experiments tag}"
export HPCAGENT_BENCH
PY=${PYTHON:-python3}
export PYTHONPATH="$HPCAGENT_BENCH:$HPCAGENT_BENCH/hpcagent_bench/numpy_translators/src"
# Deterministic plots: a fixed hash seed and a headless backend.
export MPLBACKEND=Agg PYTHONHASHSEED=0
D=$ROOT/data W=$ROOT/work T=$ROOT/tables F=$ROOT/figures L=$ROOT/lib
STATS=$HPCAGENT_BENCH/statistics
mkdir -p "$W" "$T" "$F"

# require <file>...: every input exists, or say which step makes it.
require() {
    local f
    for f; do [[ -e $f ]] || { echo "missing $f: run ./download.sh (data) or ./stats.sh (work, tables)" >&2; exit 2; }; done
}

# check [--record]: every figure and table matches SHA256SUMS, or rewrite it.
check() {
    local sums=$ROOT/SHA256SUMS
    if [[ ${1:-} == --record ]]; then
        (cd "$ROOT" && find figures tables -type f | LC_ALL=C sort | xargs sha256sum) >"$sums"
        echo "recorded $(wc -l <"$sums") checksums"
    else
        (cd "$ROOT" && sha256sum --quiet -c "$sums") && echo "OK: $(wc -l <"$sums") files match SHA256SUMS"
    fi
}
