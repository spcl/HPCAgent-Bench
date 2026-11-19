#!/bin/bash
# Section 4.5 numbers from the one-limit reruns. Usage: run.sh [RUN_DIR] (default out-ault-1130-abi).
X=${X:-$HOME/rv/ana1118}
A=$X/out/${1:-out-ault-1130-abi}
D=$X/ana/sec45/out-$(basename $A)
mkdir -p $D
OBS=$(ls $A/obs/obs-*.db | paste -sd,)
cd $X/code-den
singularity exec -B /scratch /scratch/ybudanaz/alloy-cv/judge-cpu.sif bash -c "unset PYTHONSAFEPATH; export PYTHONPATH=\$PWD:\$PWD/agent PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 HPCAGENT_BENCH_BUDGETS_MATCHED_LIMITS=flag; nice $HOME/rv/lead2/.venv-sif/bin/python $X/ana/sec45/sec45.py $OBS $A/tags $X/ana/sec45/comparators.csv $D" 2>&1 | tee $D/sec45.log
