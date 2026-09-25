#!/usr/bin/env bash
# Figure 8 (appendix): per-kernel speedup of the loop-level answers the pair tables use.

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_<host>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
. "$(dirname "$0")/lib.sh"
require "$W/llr-focus40.db"
"$PY" "$L/plot_cheating_per_kernel.py" --db "$W/llr-focus40.db" --out "$F/cheating_per_kernel.pdf"
