#!/usr/bin/env bash
# Tune the aiter bf16 GEMM shapes our SGLang arms actually execute, once, on one node.
#
# Every aiter GEMM lookup in every kimi arm on record MISSES and falls back to torch: 2,084,892
# lookups, zero hits, each one logging "not found tuned config ... using torch solution:0". The
# table aiter ships is keyed on the exact (M, N, K) and only covers shapes AMD happened to tune.
#
# Only 17 (N, K) pairs exist and eight carry 99.6% of the traffic, so the shape list is small:
# scripts/emit_aiter_gemm_shapes.py writes those eight over a bounded M ladder, 672 shapes. M is
# the batch token count and has 7,591 distinct values in our logs, so full coverage is not the
# goal -- the ladder is.
#
# Writes the tuned CSV to shared scratch, NOT into the image. Serving then copies it over
# aiter/configs/bf16_tuned_gemm.csv at container start, so nothing tunes at request time: that is
# the failure mode that wedged six earlier aiter attempts (JIT behind a baton lock, outliving
# every deadline).
#SBATCH --job-name=tune-aiter-gemm
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/tune-aiter-gemm-%j.out
#SBATCH --error=logs/tune-aiter-gemm-%j.err
set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
INFERENCE_EDF="${INFERENCE_EDF:-sglang-rocm-mi30x}"
TUNE_DIR="${TUNE_DIR:-${SCRATCH}/aiter-tune}"
UNTUNED="${UNTUNED:-${TUNE_DIR}/bf16_untuned_gemm.csv}"
TUNED="${TUNED:-${TUNE_DIR}/bf16_tuned_gemm.csv}"
# aiter falls back to expanduser("~")/.aiter for template ops whatever AITER_JIT_DIR says, so HOME
# moves with the cache or the tuner writes into the real home and burns inodes there.
CACHE_ROOT="${JIT_CACHE_ROOT:-${SCRATCH}/.jit-cache}/${INFERENCE_EDF}"

[[ -s "${UNTUNED}" ]] || { echo "no shape list at ${UNTUNED}; run scripts/emit_aiter_gemm_shapes.py first" >&2; exit 2; }
mkdir -p "${TUNE_DIR}" "${CACHE_ROOT}/home" "${CACHE_ROOT}/xdg"
printf 'job:     %s\nshapes:  %s (%s rows)\nout:     %s\n\n' \
    "${SLURM_JOB_ID}" "${UNTUNED}" "$(($(wc -l <"${UNTUNED}") - 1))" "${TUNED}"

srun --nodes=1 --ntasks=1 --exclusive --export=ALL --environment="${INFERENCE_EDF}" \
     --cpus-per-task="${SLURM_CPUS_ON_NODE:-64}" bash -c "
    set -Eeuo pipefail
    export HOME='${CACHE_ROOT}/home' XDG_CACHE_HOME='${CACHE_ROOT}/xdg'
    export AITER_JIT_DIR='${CACHE_ROOT}/aiter' TOKENIZERS_PARALLELISM=false
    # The CE hands the step ROCR_VISIBLE_DEVICES; aiter's tuner reads HIP_VISIBLE_DEVICES, and
    # having both set makes the second an index INTO the first.
    if [[ -n \"\${ROCR_VISIBLE_DEVICES:-}\" ]]; then
        export HIP_VISIBLE_DEVICES=\"\${HIP_VISIBLE_DEVICES:-\${ROCR_VISIBLE_DEVICES}}\"
        unset ROCR_VISIBLE_DEVICES
    fi
    root=\$(python3 -c 'import aiter, pathlib; print(pathlib.Path(aiter.__file__).resolve().parents[1])')
    echo \"aiter checkout: \${root}\"
    tuner=\"\${root}/csrc/gemm_a16w16/gemm_a16w16_tune.py\"
    # The tuner has moved between aiter releases; find it rather than pin a path that silently
    # stops existing and turns this job into a 4 hour no-op.
    if [[ ! -f \"\${tuner}\" ]]; then
        tuner=\$(find \"\${root}\" -name 'gemm_a16w16_tune.py' -print -quit)
    fi
    [[ -n \"\${tuner}\" && -f \"\${tuner}\" ]] || { echo 'no gemm_a16w16_tune.py in this aiter build' >&2; exit 3; }
    echo \"tuner: \${tuner}\"
    cd \"\${root}\"
    python3 \"\${tuner}\" --input_file '${UNTUNED}' --tuned_file '${TUNED}' 2>&1 | tail -40
"
if [[ -s "${TUNED}" ]]; then
    printf '\ntuned rows: %s\n' "$(($(wc -l <"${TUNED}") - 1))"
    head -3 "${TUNED}"
else
    echo "tuner produced no ${TUNED}" >&2
    exit 4
fi
