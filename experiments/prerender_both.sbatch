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
SD="${OPT}/experiments"
source "${SD}/roster.sh"
TAG=${TAG:-llr-focus40}
KERNELS=${KERNELS:-$(roster_for "${TAG}")}

# The forms are only as current as the dace that rendered them, so move the tree to the tip of
# extended FIRST and print the commit the render is attributable to. Fast-forward only, and a
# network failure is not fatal: a slightly older tip still renders, and refusing to start would
# trade that for nothing. The tree is shared, so this never merges and never resets.
DACE_TREE=${DACE_TREE:-${SCRATCH}/dace}
if [[ -d "${DACE_TREE}/.git" ]]; then
    git -C "${DACE_TREE}" fetch --quiet origin extended \
        && git -C "${DACE_TREE}" merge --ff-only --quiet FETCH_HEAD \
        || echo "dace: could not fast-forward; rendering with what is checked out" >&2
    echo "dace: $(git -C "${DACE_TREE}" rev-parse --short HEAD) on $(git -C "${DACE_TREE}" rev-parse --abbrev-ref HEAD)"
else
    echo "dace: no checkout at ${DACE_TREE}" >&2
fi
export DACE_TREE
echo "prerender ${TAG}: $(tr ',' '\n' <<<"${KERNELS}" | grep -c .) kernels"
for target in cpu gpu; do
    bash "${SD}/prerender_cpf.sh" outer "${SCRATCH}/cpf-forms-${target}-${TAG}" "${KERNELS}" "${OPT}" "${target}"
    ext=cpp; [[ "${target}" == gpu ]] && ext=hip
    echo "${target} forms: $(find "${SCRATCH}/cpf-forms-${target}-${TAG}" -maxdepth 1 -name "*_cpf.${ext}" | wc -l)"
done
