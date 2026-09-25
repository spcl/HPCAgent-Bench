#!/usr/bin/env bash
# Download the serving weights AND leave them with a Lustre layout that loads fast.
#
# The download (`hf download`, inside the CE container) and the restripe (`lfs`, host only)
# cannot happen in one place, so a plain download leaves the layout unfixed and nothing notices.
# The hub dir's PFL default is not sufficient in practice: restripe every new model and verify.
#
#   sbatch fetch_weights.sbatch                     # the four models the campaign serves
#   MODELS="zai-org/GLM-5.3" sbatch fetch_weights.sbatch
#   AUDIT_ONLY=1 sbatch fetch_weights.sbatch        # check layout, change nothing
#
# On Daint (GH200) the partition and the image differ; -p overrides the directive below, and the
# vLLM image has no venv:
#   EDF=hpcagent-bench-vllm-gh200-latest PYTHON=python3 MODELS="Qwen/Qwen3.8-27B-FP8" \
#     sbatch -A <project> -p normal fetch_weights.sbatch
#
#SBATCH --job-name=fetch-weights
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --no-requeue

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:?set SCRATCH}"
REPO="${REPO:-$(git -C "${SLURM_SUBMIT_DIR:-${PWD}}" rev-parse --show-toplevel 2>/dev/null || true)}"
[[ -n "${REPO}" && -f "${REPO}/scripts/cache_env.sh" ]] || { echo "FAIL: no checkout: set REPO or submit from inside one" >&2; exit 1; }
. "${REPO}/scripts/cache_env.sh"
EDF="${EDF:-${HOME}/.edf/hpcagent-bench-sglang-mi300-latest.toml}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"
# The interpreter inside EDF that has huggingface_hub: the SGLang image keeps it in its venv.
PYTHON="${PYTHON:-/opt/venv/bin/python3}"

# Every model the campaign serves; ~1.5 TB total, default to the full set to avoid a partial
# download biting hours into a campaign.
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

# The PFL default: harmless when it does not take, correct when it does; not trusted alone.
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
            "${PYTHON}" - <<'PY'
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

    # `while read` via process substitution, not xargs: the </dev/null lfs needs would feed
    # xargs an empty list, which migrates nothing and exits 0.
    while read -r blob; do
        total=$((total + 1))
        # head -1, never tail -1: getstripe -c prints a trailing blank line tail -1 would return.
        sc="$(lfs getstripe -c "${blob}" </dev/null 2>/dev/null | head -1)"
        case "${sc}" in ''|*[!0-9]*) sc=0 ;; esac
        [ "${sc}" -ge "${STRIPE_COUNT}" ] && continue
        narrow=$((narrow + 1))
        if [ "${AUDIT_ONLY}" = 1 ]; then continue; fi
        # Safe only because nothing is serving this model right now.
        lfs migrate -c "${STRIPE_COUNT}" -S "${STRIPE_SIZE}" "${blob}" </dev/null >/dev/null 2>&1 || true
        sc2="$(lfs getstripe -c "${blob}" </dev/null 2>/dev/null | head -1)"
        case "${sc2}" in ''|*[!0-9]*) sc2=0 ;; esac
        if [ "${sc2}" -ge "${STRIPE_COUNT}" ]; then fixed=$((fixed + 1)); else still=$((still + 1)); fi
    done < <(find "${dir}" -type f -size +"${MIN_BLOB_BYTES}"c 2>/dev/null)

    printf '  blobs>1G=%s  narrow=%s  restriped=%s  STILL NARROW=%s\n' \
           "${total}" "${narrow}" "${fixed}" "${still}"
    # Verify rather than trust the migrate exit code, which is 0 even when it moved nothing.
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
