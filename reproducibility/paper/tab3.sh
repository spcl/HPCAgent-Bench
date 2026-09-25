#!/usr/bin/env bash
# Table 3: GH200 against MI300A, from the joined regrade of the final loop-level answers.

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_<host>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
. "$(dirname "$0")/lib.sh"
require "$D/transfer.csv"
"$PY" "$STATS/plot_transfer.py" --paired-csv "$D/transfer.csv" --out "$W/gh200-transfer" --table "$T/gh200-transfer.csv"
"$PY" "$L/gh200_table.py" "$T/gh200-transfer.csv" "$T/gh200-transfer-table.tex"
