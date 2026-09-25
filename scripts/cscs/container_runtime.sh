#!/usr/bin/env bash
# Print the container launcher a job on Beverin uses: `enroot` (scripts/cscs/enroot_srun.sh) unless
# the caller exports CONTAINER_RUNTIME (`ce` = srun --environment through pyxis).
#
#   CONTAINER_RUNTIME="$(scripts/cscs/container_runtime.sh)"
#
# WHY ENROOT. Under `ce` pyxis applies the EDF's comm-hook annotations (netstack, aws_ofi_nccl) and
# forced NCCL_NET/NCCL_NET_PLUGIN to EVERY step. A single-node inference step then fails at tensor
# parallel init with "NCCL error: invalid usage ... Failed to initialize any NET plugin" and the job
# ends in ~5 minutes: the 2026-09-17 17:00 wave, jobs 640160-640181, all 22 arms with
# INFERENCE_NODES=1. enroot_srun.sh turns the hooks on only for a multi-node inference step and strips
# the forced NCCL_NET otherwise; the same arms under enroot (640076-640083) completed.
# `ce` is therefore opt-in: use it only for a run whose every GPU collective crosses nodes, or once
# the EDFs gate the hooks per step.
set -uo pipefail

# A core dump lands in the crashing process's CWD (the checkout) and Slurm propagates the
# SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
printf '%s\n' "${CONTAINER_RUNTIME:-enroot}"
