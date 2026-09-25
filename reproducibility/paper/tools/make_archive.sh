#!/usr/bin/env bash
# Pack data/ into the release archive download.sh fetches, anonymized, with DATA_SHA256SUMS.
#   tools/make_archive.sh [out.tar.zst]
# The databases are rewritten by tools/anonymize_dbs.py into a copy; data/ itself is never modified.
set -euo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_<host>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
root=$(cd "$(dirname "$0")/.." && pwd)
out=${1:-$root/work/data.tar.zst}
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cp -a "$root/data/." "$stage/"
find "$stage" -name '*.db' -delete
(cd "$root" && python3 tools/anonymize_dbs.py --out "$stage/.anon" data)
(cd "$stage/.anon/data" && find . -name '*.db' -exec cp --parents {} "$stage/" \;)
rm -rf "$stage/.anon"
(cd "$stage" && find . -type f | LC_ALL=C sort | xargs sha256sum) >"$root/DATA_SHA256SUMS"
tar --zstd -cf "$out" -C "$stage" .
echo "archive: $out ($(du -h "$out" | cut -f1)), $(wc -l <"$root/DATA_SHA256SUMS") files in DATA_SHA256SUMS"
