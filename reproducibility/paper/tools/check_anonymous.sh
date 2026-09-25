#!/usr/bin/env bash
# Re-check a package (zip or directory) against .anonymize-terms.txt; prints "clean" or every leak.
#   tools/check_anonymous.sh <package.zip | dir>
set -euo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_<host>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
root=$(cd "$(dirname "$0")/.." && pwd)
target=${1:?usage: tools/check_anonymous.sh <package.zip | dir>}
if [[ -f $target ]]; then
    dir=$(mktemp -d)
    trap 'rm -rf "$dir"' EXIT
    unzip -q "$target" -d "$dir"
    target=$dir
fi
python3 "$root/tools/anonymize.py" --terms "$root/.anonymize-terms.txt" --check "$target"
