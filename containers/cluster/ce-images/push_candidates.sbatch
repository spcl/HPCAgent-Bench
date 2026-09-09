#!/usr/bin/env bash
# Publish the verified candidate images to Docker Hub, from the saved OCI archives (no rebuild).
#
# Credentials come from the ENVIRONMENT and are never written here or echoed. Submit with:
#
#   REGISTRY_USER=thrudprimrose REGISTRY_TOKEN=<token> \
#     sbatch --export=ALL push_candidates.sbatch
#
# IMAGES= picks a subset; default is every image whose candidate archive passed the pre-check.
#SBATCH --job-name=push-candidates
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/push-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/push-%j.err
set -Eeuo pipefail
: "${REGISTRY_USER:?export REGISTRY_USER before submitting}"
: "${REGISTRY_TOKEN:?export REGISTRY_TOKEN before submitting}"
NS="${NS:-docker.io/${REGISTRY_USER}}"
TAG="${TAG:-$(date +%Y%m%d)}"
IMG=/capstor/scratch/cscs/${USER}/x86_64/ce-images
cd /capstor/scratch/cscs/${USER}/x86_64/optarena/containers/cluster/ce-images

# archive basename -> published repo name
declare -A REPO=(
  [optarena-ce-amd-mi300-candidate]="optarena-ce-amd-mi300"
  [optarena-ce-judge-amd-mi300-candidate]="optarena-ce-judge-amd-mi300"
  [optarena-sglang-candidate]="optarena-sglang"
  [optarena-vllm-candidate]="optarena-vllm"
)
IMAGES="${IMAGES:-optarena-ce-amd-mi300-candidate optarena-ce-judge-amd-mi300-candidate optarena-sglang-candidate optarena-vllm-candidate}"

rc=0
for a in ${IMAGES}; do
    dest="${NS}/${REPO[$a]:?no repo mapping for ${a}}"
    echo "===== pushing ${a}.oci.tar -> ${dest}:{${TAG},latest}"
    if PUSH_REPO="${dest}" ./push_image.sh --from-archive "${IMG}/${a}.oci.tar" "${TAG}" latest; then
        echo "  PUSHED ${dest}:${TAG}"
    else
        echo "  FAILED ${dest}" >&2; rc=1
    fi
    rm -rf "/dev/shm/${USER}/pushcheck" 2>/dev/null || true
done
echo "PUSH DONE rc=${rc}"
exit "${rc}"
