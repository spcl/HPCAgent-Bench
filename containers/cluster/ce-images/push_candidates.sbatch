#!/usr/bin/env bash
# Publish the verified candidate images to Docker Hub, from the saved OCI archives (no rebuild).
#
# Credentials come from the ENVIRONMENT and are never written here or echoed. Submit with:
#
#   REGISTRY_USER=<user> REGISTRY_TOKEN=<token> \
#     sbatch --export=ALL push_candidates.sbatch
#
# Publishes to docker.io/spcleth/hpcagent-bench by default -- see SINGLE_REPO below.
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

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_nid<node>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
: "${REGISTRY_USER:?export REGISTRY_USER before submitting}"
: "${REGISTRY_TOKEN:?export REGISTRY_TOKEN before submitting}"
NS="${NS:-docker.io/${REGISTRY_USER}}"
TAG="${TAG:-$(date +%Y%m%d)}"
IMG=/capstor/scratch/cscs/${USER}/x86_64/ce-images
cd /capstor/scratch/cscs/${USER}/x86_64/optarena/containers/cluster/ce-images

# archive basename -> published repo name
declare -A REPO=(
  [optarena-ce-amd-mi300-candidate]="optarena-ce-amd-mi300"
  # A rebuild that could not use the -candidate name (an EDF was mounting it) lands here.
  # PUSHING is independent of PROMOTION: the push reads the .oci.tar and touches nothing a
  # running job mounts, so it does not have to wait for a campaign to drain.
  [optarena-ce-amd-mi300-cxx20]="optarena-ce-amd-mi300"
  [optarena-ce-judge-amd-mi300-candidate]="optarena-ce-judge-amd-mi300"
  [optarena-sglang-candidate]="optarena-sglang"
  [optarena-vllm-candidate]="optarena-vllm"
)
# EVERY image goes to ONE repo, distinguished by a role prefix on the tag:
#   hpcagent-bench:{agent,judge,sglang,vllm}-<date>  and  ...-latest
# That is the project's published location, so it is the DEFAULT rather than a flag someone has
# to remember -- a push that silently lands in a personal namespace is worse than one that fails.
# Set SINGLE_REPO="" to fall back to one repo per image under ${NS}.
#
# The login identity and the destination are INDEPENDENT: REGISTRY_USER authenticates, SINGLE_REPO
# says where. But the authenticated account still needs WRITE access to the spcleth org, or the
# push fails with "denied" after loading a 23 GB archive.
declare -A ROLE=(
  [optarena-ce-amd-mi300-candidate]="agent"
  [optarena-ce-amd-mi300-cxx20]="agent"
  [optarena-ce-judge-amd-mi300-candidate]="judge"
  [optarena-sglang-candidate]="sglang"
  [optarena-vllm-candidate]="vllm"
)
SINGLE_REPO="${SINGLE_REPO-docker.io/spcleth/hpcagent-bench}"
IMAGES="${IMAGES:-optarena-ce-amd-mi300-candidate optarena-ce-judge-amd-mi300-candidate optarena-sglang-candidate optarena-vllm-candidate}"

rc=0
for a in ${IMAGES}; do
    if [ -n "${SINGLE_REPO}" ]; then
        role="${ROLE[$a]:?no role mapping for ${a}}"
        dest="${SINGLE_REPO}"; tags=("${role}-${TAG}" "${role}-latest")
    else
        dest="${NS}/${REPO[$a]:?no repo mapping for ${a}}"; tags=("${TAG}" latest)
    fi
    echo "===== pushing ${a}.oci.tar -> ${dest}:{$(IFS=,; echo "${tags[*]}")}"
    if PUSH_REPO="${dest}" ./push_image.sh --from-archive "${IMG}/${a}.oci.tar" "${tags[@]}"; then
        echo "  PUSHED ${dest}:${TAG}"
    else
        echo "  FAILED ${dest}" >&2; rc=1
    fi
    rm -rf "/dev/shm/${USER}/pushcheck" 2>/dev/null || true
done
echo "PUSH DONE rc=${rc}"
exit "${rc}"
