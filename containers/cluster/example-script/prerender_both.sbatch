#!/usr/bin/env bash
#SBATCH --job-name=cpf-prerender
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/cpf-prerender-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/cpf-prerender-%j.out
#
# Pre-render both canonical parallel forms for a roster, into the two directories the launcher
# gates on. Separate directories per target: the file names are identical on each side, and one
# mixed cache hands a CPU arm a device form.
#
# The roster comes from roster.sh, never from a list kept here. A copy of the 40 names lived in
# this file and in the launcher; either one drifting from the manifests produces a cache that
# covers most of a campaign, and a kernel the cache misses is answered `unavailable` with HTTP
# 200 -- a treated arm silently collapses into its own control rather than failing.
set -uo pipefail
ulimit -c 0
export SCRATCH=/capstor/scratch/cscs/$USER/x86_64
OPT="${SCRATCH}/optarena"
SD="${OPT}/containers/cluster/example-script"
source "${SD}/roster.sh"
TAG=${TAG:-llr-focus40}
KERNELS=${KERNELS:-$(roster_for "${TAG}")}
echo "prerender ${TAG}: $(tr ',' '\n' <<<"${KERNELS}" | grep -c .) kernels"
for target in cpu gpu; do
    bash "${SD}/prerender_cpf.sh" outer "${SCRATCH}/cpf-forms-${target}-${TAG}" "${KERNELS}" "${OPT}" "${target}"
    ext=cpp; [[ "${target}" == gpu ]] && ext=hip
    echo "${target} forms: $(find "${SCRATCH}/cpf-forms-${target}-${TAG}" -maxdepth 1 -name "*_cpf.${ext}" | wc -l)"
done
