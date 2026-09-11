#!/usr/bin/env bash
# Download the serving weights AND leave them with a Lustre layout that loads fast.
#
# WHY THIS EXISTS. The two halves cannot be done in one place: `hf download` runs inside the CE
# container (that is where huggingface_hub lives), and `lfs` exists only on the HOST. So a plain
# download leaves the layout unfixed and nothing notices -- iopsstor delivers 9.45 GB/s at 16
# readers against capstor's 0.83, but only if the blobs are actually striped.
#
# The PFL default on the hub dir is NOT sufficient, which is the part that surprises people:
# run_cluster.sh sets `lfs setstripe -E 64M -c 1 -E -1 -c 16 -S 4M` on ${HF_HOME}/hub, and the
# five models fetched in 2026-08 did inherit it -- but GLM-5.3, downloaded 2026-09-08 into that
# same directory, came down with all 141 blobs at stripe_count 1 / 1 MiB. Inheritance is not a
# property of hf download. RESTRIPE EVERY NEW MODEL AND VERIFY; never assume.
#
#   sbatch fetch_weights.sbatch                     # the four models the campaign serves
#   MODELS="zai-org/GLM-5.3" sbatch fetch_weights.sbatch
#   AUDIT_ONLY=1 sbatch fetch_weights.sbatch        # check layout, change nothing
#
#SBATCH --job-name=fetch-weights
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/fetch-weights-%j.out
#SBATCH --error=/capstor/scratch/cscs/%u/x86_64/ce-images/logs/fetch-weights-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${USER}/x86_64}"
FAST_SCRATCH="${FAST_SCRATCH:-/iopsstor/scratch/cscs/${USER}}"
[ -d "${FAST_SCRATCH}" ] || FAST_SCRATCH="${SCRATCH}"
export HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"
EDF="${EDF:-${HOME}/.edf/sglang-latest.toml}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"

# Every model the campaign serves. Weights are ~1.5 TB in total, so the default is the full set
# only because a partial one is the thing that bites six hours into a campaign.
MODELS="${MODELS:-moonshotai/Kimi-K2.7-Code zai-org/GLM-5.3 Qwen/Qwen3.8-27B-FP8 openai/gpt-oss-120b}"

STRIPE_COUNT="${STRIPE_COUNT:-16}"
STRIPE_SIZE="${STRIPE_SIZE:-4M}"
MIN_BLOB_BYTES="${MIN_BLOB_BYTES:-1073741824}"   # only wide-stripe what is worth striping

mkdir -p "${HF_HOME}/hub" "${SCRATCH}/ce-images/logs"
echo "job:     ${SLURM_JOB_ID}"
echo "HF_HOME: ${HF_HOME}"
echo "models:  ${MODELS}"
echo "mode:    $([ "${AUDIT_ONLY}" = 1 ] && echo 'AUDIT ONLY' || echo 'download + restripe')"
echo

# The PFL default. Harmless when it does not take (see the header) and correct for anything that
# does honour it, so it stays -- it is just not TRUSTED.
if command -v lfs >/dev/null 2>&1; then
    lfs setstripe -E 64M -c 1 -E -1 -c "${STRIPE_COUNT}" -S "${STRIPE_SIZE}" "${HF_HOME}/hub" 2>/dev/null \
        && echo "PFL default set on ${HF_HOME}/hub" \
        || echo "note: setstripe failed on ${HF_HOME}/hub (non-Lustre?) -- continuing"
else
    echo "note: no lfs on this host; layout cannot be checked or fixed here"
fi
echo

# ---- 1. download, inside the container, where huggingface_hub lives ---------------------------
if [ "${AUDIT_ONLY}" != 1 ]; then
    for repo in ${MODELS}; do
        echo "=== downloading ${repo}"
        MODEL_REPO="${repo}" srun --environment="${EDF}" --cpus-per-task="${SLURM_CPUS_ON_NODE:-96}" \
            /opt/venv/bin/python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

repo = os.environ["MODEL_REPO"]
# max_workers is the download parallelism, not a thread pool for one file; 8 is what the campaign
# path uses and what the mirror tolerates.
path = snapshot_download(repo_id=repo, max_workers=8)
print(f"resolved {repo} -> {path}", flush=True)
PY
    done
    echo
fi

# ---- 2. audit and fix the layout, on the HOST, where lfs exists -------------------------------
command -v lfs >/dev/null 2>&1 || { echo "no lfs: skipping layout audit"; exit 0; }

rc=0
for repo in ${MODELS}; do
    dir="${HF_HOME}/hub/models--${repo//\//--}"
    if [ ! -d "${dir}" ]; then
        echo "=== ${repo}: NOT PRESENT at ${dir}" >&2
        rc=1
        continue
    fi
    echo "=== ${repo}"
    total=0; narrow=0; fixed=0; still=0

    # `while read` fed by process substitution, NOT xargs: xargs takes its file list from stdin,
    # so the </dev/null that lfs needs would feed xargs an empty list -- it then migrates nothing,
    # prints done in the same second, and exits 0.
    while read -r blob; do
        total=$((total + 1))
        # head -1, never tail -1: lfs getstripe -c prints the count then a TRAILING BLANK LINE,
        # and tail -1 yields "" which every numeric guard reads as narrow. That single mistake
        # produced two entirely wrong audit tables in one session.
        sc="$(lfs getstripe -c "${blob}" </dev/null 2>/dev/null | head -1)"
        case "${sc}" in ''|*[!0-9]*) sc=0 ;; esac
        [ "${sc}" -ge "${STRIPE_COUNT}" ] && continue
        narrow=$((narrow + 1))
        if [ "${AUDIT_ONLY}" = 1 ]; then continue; fi
        # Safe ONLY because nothing is serving this model right now. A migrate under a running
        # server is what the "while nothing reads them" warning in run_cluster.sh is about.
        lfs migrate -c "${STRIPE_COUNT}" -S "${STRIPE_SIZE}" "${blob}" </dev/null >/dev/null 2>&1 || true
        sc2="$(lfs getstripe -c "${blob}" </dev/null 2>/dev/null | head -1)"
        case "${sc2}" in ''|*[!0-9]*) sc2=0 ;; esac
        if [ "${sc2}" -ge "${STRIPE_COUNT}" ]; then fixed=$((fixed + 1)); else still=$((still + 1)); fi
    done < <(find "${dir}" -type f -size +"${MIN_BLOB_BYTES}"c 2>/dev/null)

    printf '  blobs>1G=%s  narrow=%s  restriped=%s  STILL NARROW=%s\n' \
           "${total}" "${narrow}" "${fixed}" "${still}"
    # VERIFY rather than trust the migrate exit code, which is 0 even when it moved nothing.
    if [ "${still}" -gt 0 ]; then
        echo "  FAIL: ${still} blob(s) did not reach stripe_count ${STRIPE_COUNT}" >&2
        rc=1
    elif [ "${total}" -eq 0 ]; then
        echo "  FAIL: no blobs over 1G found -- the download did not land here" >&2
        rc=1
    else
        echo "  OK: every blob >1G is stripe_count >= ${STRIPE_COUNT}"
    fi
done

echo
if [ "${rc}" -eq 0 ]; then echo "WEIGHTS READY: downloaded and wide-striped under ${HF_HOME}"
else echo "WEIGHTS NOT READY -- see the failures above" >&2; fi
exit "${rc}"
