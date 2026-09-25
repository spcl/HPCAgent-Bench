#!/usr/bin/env bash
# Figure 5: distributed ML scaling, speedup over PyTorch on one GPU.

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_<host>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
. "$(dirname "$0")/lib.sh"
require "$W/mlscale-torch.db"
"$PY" "$STATS/plot_scaling.py" "$W/mlscale-torch.db" --experiment mlscale- \
    --arm '^mlscale-(qwen38|oss120b)-hip(-dist-rccl-amd)?(-clean)?$' --figure mode-grid --quantity speedup \
    --kernels dist_layer_norm dist_cross_entropy dist_softmax --print-width 5.5 \
    --out "$F/ml_scaling" --table "$T/fig5.csv"
mv -f "$F/ml_scaling-mode-grid.pdf" "$F/ml_scaling.pdf"
mv -f "$F/ml_scaling-mode-grid.png" "$F/ml_scaling.png"
