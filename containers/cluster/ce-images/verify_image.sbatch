#!/bin/bash
#SBATCH --job-name=verify-image
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/verify-%x-%j.out
# Run verify_image.py INSIDE a built image and fail the job when a required library is missing.
#
#   sbatch --export=ALL,IMAGE=$SCRATCH/ce-images/optarena-judge-agent-amd.sqsh,PROFILE=judge-agent-amd \
#          containers/cluster/ce-images/verify_image.sbatch
#
# The EDF is generated here rather than kept in ~/.edf: this runs against a CANDIDATE image that no
# campaign mounts yet, and a checked-in EDF pointing at a candidate is how a candidate becomes live
# by accident.
set -euo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and a dump lands in the crashing
# process's CWD, littering the checkout with core_nid<node>_<pid> files on a filesystem whose
# quota is inodes. Slurm propagates the SUBMITTER's core limit, so the floor has to be set here.
ulimit -c 0
# %u above is the only way to keep the log path portable in an SBATCH directive; the body can just
# use SCRATCH, which is what a clone on another account will have.
S="${SCRATCH:?set SCRATCH}"
IMAGE="${IMAGE:?set IMAGE to the .sqsh to verify}"
PROFILE="${PROFILE:-judge-agent-amd}"
test -f "${IMAGE}"

EDF="${S}/.tmp/verify-${SLURM_JOB_ID}.toml"
mkdir -p "$(dirname "${EDF}")"
# PATH is declared, not inherited, for the same reason the real EDF declares it: the CE does not
# preserve the image's own ENV reliably. Without this the CE handed the sglang image
# PATH=/usr/local/bin:/opt/rocm/bin:/usr/sbin:/usr/bin:/sbin:/bin -- no /opt/venv/bin -- so
# `python3` was /usr/bin/python3 and verify_image.py, which probes with sys.executable, reported
# numpy, torch, sglang, aiter, triton and flydsl all missing from an image that has every one of
# them. A verifier that fails a good image is as useless as one that passes a bad one.
#
# Both interpreter prefixes are named because only one exists per image: /opt/venv/bin in the
# sglang and judge images, /opt/pytorch211/bin in the vLLM pair.
cat > "${EDF}" <<TOML
image = "${IMAGE}"
mounts = ["/capstor/:/capstor/", "/iopsstor/:/iopsstor/"]
workdir = "/"

[env]
PATH = "/opt/venv/bin:/opt/pytorch211/bin:/opt/gcc/bin:/opt/view/bin:/opt/papi/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
TOML

echo "=== verifying ${IMAGE} as profile=${PROFILE} ==="
srun --environment="${EDF}" python3 "${S}/optarena/containers/cluster/ce-images/verify_image.py" \
     --profile "${PROFILE}" --verbose
rc=$?
rm -f "${EDF}"
echo "=== required-failure count: ${rc} ==="
exit "${rc}"
