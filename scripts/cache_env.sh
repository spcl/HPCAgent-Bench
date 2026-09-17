#!/usr/bin/env bash
# WHERE THE CACHES LIVE. Source it; never copy a path out of it.
#
#   . "${HPCAGENT_BENCH_REPO}/scripts/cache_env.sh"
#
# This file sets TWO roots and nothing else. That restraint is the point: run_cluster.sh already
# derives the seven individual knobs (HOME, XDG_CACHE_HOME, AITER_JIT_DIR, VLLM_CACHE_ROOT,
# TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR, TORCH_EXTENSIONS_DIR) from JIT_CACHE_ROOT, keyed by
# ${INFERENCE_CE_ENV}, and the inference EDFs deliberately override some of them. Exporting those
# knobs HERE would silently win over both.
#
# In particular AITER_JIT_DIR must NOT be set here. The sglang and vllm EDFs pin it to the
# IN-IMAGE prebuild at /opt/aiter-jit, and job 628077 measured what happens when a host directory
# wins instead: module_aiter_core loads from the host and the image's prebuilt copy goes unused.
#
# TWO ROOTS, because the two kinds of data have opposite shapes -- this split is measured, not
# stylistic:
#
#   WEIGHTS -> iopsstor (Lustre). Read once per rank at load, by many ranks at once; that
#              filesystem is 11x faster at 16 concurrent readers.
#   JIT     -> the general scratch. Small, many, written. Also keyed by inference EDF downstream,
#              because these artefacts are compiled against ONE ROCm/aiter build and a rank that
#              loads a mismatched .so fails late or silently.
#
# The JIT root is moved OUT of the repo. run_cluster.sh's fallback is
# ${HPCAGENT_BENCH_REPO}/.cache/jit, which grows build output inside a git checkout; the default
# below keeps the general-scratch placement that was chosen deliberately while leaving the tree
# clean. Set JIT_CACHE_ROOT yourself to override.

: "${FAST_SCRATCH:=/iopsstor/scratch/cscs/${USER:-$(id -un)}}"
: "${HPCAGENT_BENCH_CACHE:=${FAST_SCRATCH}/.hpcagentbench-cache}"
export FAST_SCRATCH HPCAGENT_BENCH_CACHE

# Weights. HF_HOME is HuggingFace's own contract and the hub is always $HF_HOME/hub, so this is
# the ONLY name for them -- a separate "weights dir" variable would be a second spelling of the
# same path, free to drift from the one the server loads from.
export HF_HOME="${HF_HOME:-${HPCAGENT_BENCH_CACHE}/hf}"

# JIT build artefacts. run_cluster.sh appends /${INFERENCE_CE_ENV} and derives the seven knobs.
export JIT_CACHE_ROOT="${JIT_CACHE_ROOT:-${SCRATCH:?set SCRATCH}/.hpcagentbench-cache}"

# Prerendered Canonical Parallel Form. Not a JIT artefact: it is device-independent text, reused
# across arms and engines, so it is neither keyed by EDF nor purged with the JIT tree.
export HPCAGENT_BENCH_CPF_PRERENDER_DIR="${HPCAGENT_BENCH_CPF_PRERENDER_DIR:-${JIT_CACHE_ROOT}/.cpf-prerender}"

# Build tools not (yet) baked into the agent/judge image -- e.g. ppcg, whose only runtime
# dependency the image ships (see hpcagent_bench/ppcg_transform.py) is the `ppcg` binary itself.
# Each tool keeps its own versioned subdirectory (`<tool>-<version>`) plus a `<tool>` symlink a
# build script repoints atomically, so a lookup that only knows the STABLE name never has to learn
# a version. Exported here, not hardcoded in the lookup, so a rebuild that moves the root (or a
# host with the tool somewhere else entirely) never needs a code change.
export HPCAGENT_BENCH_TOOLS_DIR="${HPCAGENT_BENCH_TOOLS_DIR:-${JIT_CACHE_ROOT}/tools}"

hpcagent_bench_cache_mkdirs() {
    mkdir -p "${HF_HOME}" "${JIT_CACHE_ROOT}" "${HPCAGENT_BENCH_CPF_PRERENDER_DIR}" "${HPCAGENT_BENCH_TOOLS_DIR}" \
        2>/dev/null || true
}
