#!/usr/bin/env bash
# Run registry.sh on a compute node: pushes and pulls move tens of GB through podman and enroot,
# which the login node has neither the tmpfs nor the patience for.
#
#   sbatch containers/images/registry.sbatch push --check --all
#   REGISTRY_USER=... REGISTRY_TOKEN=... sbatch containers/images/registry.sbatch push --all
#   sbatch containers/images/registry.sbatch pull judge-agent-amd
#
#SBATCH --job-name=registry
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --no-requeue
set -Eeuo pipefail
ulimit -c 0
REPO="${REPO:-${SLURM_SUBMIT_DIR:?submit from the checkout root or set REPO}}"
exec bash "${REPO}/containers/images/registry.sh" "$@"
