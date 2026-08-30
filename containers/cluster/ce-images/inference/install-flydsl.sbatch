#!/usr/bin/env bash
# Install FlyDSL for the SGLang image, into shared scratch rather than the image.
#
# WHY IT MATTERS: AMD's own Kimi-on-MI300X profiling puts fused_moe at 87.8% of GPU time at low
# concurrency and 89.7% at high. Their fix is not to tune the Triton MoE but to replace it with a
# FlyDSL W4A16+BF16 kernel, measured at +162% throughput and -47% TTFT with no GSM8K regression.
# aiter dispatches to it when AITER_USE_FLYDSL_MOE=1 and the package imports.
#
# WHY PIP, NOT A SOURCE BUILD: FlyDSL ships its own LLVM/MLIR and embeds the MLIR python bindings.
# It does NOT use ROCm's MLIR or a system LLVM, and pointing MLIR_PATH at a foreign one is a
# documented build failure (`std::gcd not found`, redeclaration errors). So the spack LLVM's MLIR
# is the wrong MLIR here, and `scripts/build_llvm.sh` is a ~30 minute build we only need if the
# wheel does not cover gfx942 on this ROCm. Try the wheel first; the source path stays available.
#
# WHY --target ON SCRATCH: the SGLang CE image is not writable, so a pip install into it does not
# survive the step. A --target prefix on capstor persists and is added to PYTHONPATH at serve
# time. One prefix per image key, because a wheel built against another ROCm/torch fails late.
#SBATCH --job-name=install-flydsl
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=logs/install-flydsl-%j.out
#SBATCH --error=logs/install-flydsl-%j.err
set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
INFERENCE_EDF="${INFERENCE_EDF:-sglang-rocm-mi30x}"
PREFIX="${FLYDSL_PREFIX:-${SCRATCH}/pyprefix/${INFERENCE_EDF}}"
CACHE_ROOT="${JIT_CACHE_ROOT:-${SCRATCH}/.jit-cache}/${INFERENCE_EDF}"
mkdir -p "${PREFIX}" "${CACHE_ROOT}/home" "${CACHE_ROOT}/xdg"
printf 'job:    %s\nprefix: %s\n\n' "${SLURM_JOB_ID}" "${PREFIX}"

srun --nodes=1 --ntasks=1 --exclusive --export=ALL --environment="${INFERENCE_EDF}" \
     --cpus-per-task="${SLURM_CPUS_ON_NODE:-64}" bash -c "
    set -Eeuo pipefail
    export HOME='${CACHE_ROOT}/home' XDG_CACHE_HOME='${CACHE_ROOT}/xdg'
    export PIP_CACHE_DIR='${CACHE_ROOT}/pip'
    if [[ -n \"\${ROCR_VISIBLE_DEVICES:-}\" ]]; then
        export HIP_VISIBLE_DEVICES=\"\${HIP_VISIBLE_DEVICES:-\${ROCR_VISIBLE_DEVICES}}\"
        unset ROCR_VISIBLE_DEVICES
    fi
    echo '=== interpreter and torch the wheel must match'
    python3 -VV
    python3 -c 'import torch; print(\"torch\", torch.__version__, \"hip\", torch.version.hip)'

    echo '=== pip install flydsl'
    # No --upgrade of anything already in the image: a wheel that drags a different torch or
    # numpy into the prefix would shadow the image's and break serving in a way that looks like
    # a model bug. --no-deps first, then report what is missing rather than pulling it blind.
    python3 -m pip install --no-cache-dir --target '${PREFIX}' --no-deps flydsl 2>&1 | tail -20

    echo '=== import check'
    PYTHONPATH='${PREFIX}':\${PYTHONPATH:-} python3 - <<'PY'
import importlib, sys
try:
    m = importlib.import_module('flydsl')
except Exception as exc:
    print('flydsl import FAILED:', type(exc).__name__, exc)
    sys.exit(1)
print('flydsl', getattr(m, '__version__', '(no __version__)'), 'at', m.__file__)
PY

    echo '=== does aiter see a FlyDSL MoE path?'
    PYTHONPATH='${PREFIX}':\${PYTHONPATH:-} AITER_USE_FLYDSL_MOE=1 python3 - <<'PY'
import os
try:
    import aiter
except Exception as exc:
    print('aiter import failed:', exc); raise SystemExit(0)
print('aiter at', aiter.__file__)
hits = [n for n in dir(aiter) if 'flydsl' in n.lower() or 'fly' in n.lower()]
print('aiter flydsl-ish symbols:', hits or 'none')
PY
"
printf '\nprefix contents:\n'; ls -1 "${PREFIX}" | head -20
