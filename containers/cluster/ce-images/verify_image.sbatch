#!/bin/bash
#SBATCH --job-name=verify-image
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=/capstor/scratch/cscs/ybudanaz/x86_64/ce-images/logs/verify-%x-%j.out
# Run verify_image.py INSIDE a built image and fail the job when a required library is missing.
#
#   sbatch --export=ALL,IMAGE=$SCRATCH/ce-images/optarena-judge-agent-amd.sqsh,PROFILE=judge-agent-amd \
#          containers/cluster/ce-images/verify_image.sbatch
#
# The EDF is generated here rather than kept in ~/.edf: this runs against a CANDIDATE image that no
# campaign mounts yet, and a checked-in EDF pointing at a candidate is how a candidate becomes live
# by accident.
set -euo pipefail
S=/capstor/scratch/cscs/ybudanaz/x86_64
IMAGE="${IMAGE:?set IMAGE to the .sqsh to verify}"
PROFILE="${PROFILE:-judge-agent-amd}"
test -f "${IMAGE}"

EDF="${S}/.tmp/verify-${SLURM_JOB_ID}.toml"
mkdir -p "$(dirname "${EDF}")"
cat > "${EDF}" <<TOML
image = "${IMAGE}"
mounts = ["/capstor/:/capstor/", "/iopsstor/:/iopsstor/"]
workdir = "/"
TOML

echo "=== verifying ${IMAGE} as profile=${PROFILE} ==="
srun --environment="${EDF}" python3 "${S}/optarena/containers/cluster/ce-images/verify_image.py" \
     --profile "${PROFILE}" --verbose
rc=$?
rm -f "${EDF}"
echo "=== required-failure count: ${rc} ==="
exit "${rc}"
