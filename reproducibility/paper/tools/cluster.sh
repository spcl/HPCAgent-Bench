# Sourced by the tools/ scripts that read from the cluster: usage, environment (tools/cluster.env),
# one ssh transport definition, a retried read-only rsync pull and a retried remote fetch.
# Never writes on the cluster.
set -euo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_<host>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
here_tools=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# usage: the calling script's leading comment block, without the '#'.
usage() { sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0"; }

# cluster_init: load tools/cluster.env (variables already in the environment win) and build the transport.
cluster_init() {
    # shellcheck source=/dev/null
    [[ -f $here_tools/cluster.env ]] && . "$here_tools/cluster.env"
    local var
    for var in CLUSTER_HOST JUMP_HOST CLUSTER_SCRATCH MIRROR; do
        [[ -n ${!var:-} ]] || { echo "$(basename "$0"): $var is not set; copy tools/cluster.env.example to tools/cluster.env" >&2; exit 2; }
    done
    TRIES=${TRIES:-8}
    RETRY_SLEEP=${RETRY_SLEEP:-30}
    # Own TCP connection per call, jump included: a shared ControlMaster to the jump host wedges and
    # hangs every later ssh. Keepalives outlast the cluster's minute-long freezes.
    SSH_OPTS=(-o ControlPath=none -o "ProxyCommand=ssh -o ControlPath=none -W %h:%p $JUMP_HOST"
        -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o ConnectTimeout=120)
    # rsync -e splits on spaces and honors quotes, so the options with spaces are quoted.
    RSH=ssh
    local opt
    for opt in "${SSH_OPTS[@]}"; do [[ $opt == *' '* ]] && RSH+=" '$opt'" || RSH+=" $opt"; done
}

# pull <remote dir under CLUSTER_SCRATCH> <local dir under DEST> [rsync filter args...]
# The cluster freezes for about a minute at a time: every rsync has an I/O timeout and is retried,
# and a retry resumes (--partial). Never --delete: runs removed there keep their data here.
pull() {
    local src=$1 dst=$2 try rc
    shift 2
    mkdir -p "$DEST/$dst"
    for ((try = 1; try <= TRIES; try++)); do
        echo "$(date +%T) pull $src (try $try/$TRIES)"
        rc=0
        rsync -a --partial --timeout=180 -e "$RSH" "$@" "$CLUSTER_HOST:$CLUSTER_SCRATCH/$src/" "$DEST/$dst/" || rc=$?
        # 24 = files vanished during the transfer (live runs); everything else arrived.
        ((rc == 0 || rc == 24)) && return 0
        echo "rsync $src rc=$rc" >&2
        ((try == TRIES)) || sleep "$RETRY_SLEEP"
    done
    echo "FAILED $src after $TRIES tries" >&2
    return 1
}

# fetch <remote shell command> <local file>: the command's stdout, retried; the file is replaced
# only on success, so a failed fetch never leaves it truncated.
fetch() {
    local cmd=$1 out=$2 try
    for ((try = 1; try <= TRIES; try++)); do
        ssh "${SSH_OPTS[@]}" "$CLUSTER_HOST" "$cmd" > "$out.part" && { mv -f "$out.part" "$out"; return 0; }
        echo "ssh fetch of $(basename "$out") failed, try $try/$TRIES" >&2
        ((try == TRIES)) || sleep "$RETRY_SLEEP"
    done
    rm -f "$out.part"
    echo "FAILED fetch of $out" >&2
    return 1
}

# checksum_dir <dir>: SHA256SUMS over every file in <dir>, in a locale-independent order.
checksum_dir() {
    (cd "$1" && find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0r sha256sum > SHA256SUMS)
}
