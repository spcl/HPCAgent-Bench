#!/usr/bin/env bash
# Figure 4: each treatment's cost ratio under every token weighting.

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_<host>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
. "$(dirname "$0")/lib.sh"
require "$W/harness20.db" "$W/llr-focus40.db"
"$PY" "$STATS/plot_cost_weighting.py" "$W/harness20.db" "$W/llr-focus40.db" \
    --pair "harness20-qwen38-openhands,harness20-qwen38-claude,OpenHands" \
    --pair "cpf-llr-focus40-qwen38-c-cpfsrc-v2,cpf-llr-focus40-qwen38-c,CPF" \
    --pair "gpu-llr-focus40-oss120b-hip-skills,gpu-llr-focus40-oss120b-hip,HIP Skills" \
    --width 1.75 --out "$F/cost_weighting" --table "$T/fig4.csv"
