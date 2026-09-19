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
# An unset SCRATCH falls back to HPCAGENT_BENCH_REPO -- the checkout's own root, which every caller
# of this file has ALREADY resolved (experiments/env.sh exports it before sourcing this script; see
# hpcagent_bench/paths.py's repo_root() for the Python side of the same default). That fallback only
# fires when HPCAGENT_BENCH_REPO is itself set: a bare `. cache_env.sh` with neither var configured
# still aborts here rather than landing caches under $HOME or /tmp where no later job would look --
# a container-run pytest suite (scripts/run_tests.sh --container) is the case this exists for. A
# caller with no scratch AND no repo (the pre-commit hooks, via scripts/run_hook.sh) passes
# JIT_CACHE_ROOT directly instead.
if [[ -z "${JIT_CACHE_ROOT:-}" ]]; then
    if [[ -n "${SCRATCH:-}" ]]; then
        JIT_CACHE_ROOT="${SCRATCH}/.hpcagentbench-cache"
    else
        : "${HPCAGENT_BENCH_REPO:?set SCRATCH, or HPCAGENT_BENCH_REPO, or pass JIT_CACHE_ROOT explicitly}"
        JIT_CACHE_ROOT="${HPCAGENT_BENCH_REPO}/.cache/jit"
    fi
fi
export JIT_CACHE_ROOT

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

# Deterministic-framework job work dirs (canon compiler-baseline columns and siblings: smoke sweeps,
# opt-report passes). Same shape as jit/ -- small-ish, many, WRITTEN by the job, one tree per job --
# so it sits beside jit/ under JIT_CACHE_ROOT rather than under HPCAGENT_BENCH_CACHE (the iopsstor
# weights root a job only READS from). Before this existed, submit-canon-llr40.sh defaulted
# out_root to ${SCRATCH}/canon-<tag>-<stamp> directly: a bare-scratch directory nothing ever swept,
# accumulating one DaCe build tree (dacecache-<column>[_rank<N>]) per column forever. A submitter
# derives its own job dir under this root as ${HPCAGENT_BENCH_RUNS_ROOT}/<job-kind>/<name>-<stamp>
# (mirroring JIT_CACHE_ROOT's own "root exported, suffix appended by the caller" pattern) and MUST
# NOT invent a path under ${SCRATCH} instead -- see .cache/README.md's "Job work dirs" section.
export HPCAGENT_BENCH_RUNS_ROOT="${HPCAGENT_BENCH_RUNS_ROOT:-${JIT_CACHE_ROOT}/runs}"

# The PERSISTENT, cross-job results store a job work dir's final per-kernel outcome is merged into
# before the work dir (build trees, per-rank shard DBs) is deleted -- never the destination a job
# writes its OWN per-rank shards to directly (those still need one file per rank per job; see the
# job-work-dir note above), or two jobs' rank 0 would race the same file. Consumers derive their own
# table/filename under this root (canon_column.sh's finalize step uses
# ${HPCAGENT_BENCH_RESULTS_DIR}/canon.db via scripts/merge_canon_results.py) rather than a single
# hardcoded name, so a second deterministic-framework family can add its own file here without
# renaming this one.
export HPCAGENT_BENCH_RESULTS_DIR="${HPCAGENT_BENCH_RESULTS_DIR:-${JIT_CACHE_ROOT}/results}"

# Container bind mounts, DERIVED from the roots above: the top-level filesystem of SCRATCH and of
# FAST_SCRATCH, deduplicated. An EDF writer mounts these instead of naming a filesystem, so moving
# scratch (/ritom <-> /capstor) changes $SCRATCH and nothing else.
hpcagent_bench_fs_root() {  # hpcagent_bench_fs_root <absolute path> -> /<first component>
    local rest=${1#/}
    printf '/%s\n' "${rest%%/*}"
}
HPCAGENT_BENCH_DATA_ROOTS=""
for root in "${SCRATCH:-}" "${FAST_SCRATCH}"; do
    [[ -n "${root}" ]] || continue
    fs=$(hpcagent_bench_fs_root "${root}")
    [[ " ${HPCAGENT_BENCH_DATA_ROOTS} " == *" ${fs} "* ]] || HPCAGENT_BENCH_DATA_ROOTS="${HPCAGENT_BENCH_DATA_ROOTS:+${HPCAGENT_BENCH_DATA_ROOTS} }${fs}"
done
unset root fs
export HPCAGENT_BENCH_DATA_ROOTS
hpcagent_bench_edf_mounts() {  # TOML array items for HPCAGENT_BENCH_DATA_ROOTS: "/a/:/a/", "/b/:/b/"
    local out="" fs
    for fs in ${HPCAGENT_BENCH_DATA_ROOTS}; do
        out="${out:+${out}, }\"${fs}/:${fs}/\""
    done
    printf '%s\n' "${out}"
}

hpcagent_bench_cache_mkdirs() {
    mkdir -p "${HF_HOME}" "${JIT_CACHE_ROOT}" "${HPCAGENT_BENCH_CPF_PRERENDER_DIR}" "${HPCAGENT_BENCH_TOOLS_DIR}" \
        "${HPCAGENT_BENCH_RUNS_ROOT}" "${HPCAGENT_BENCH_RESULTS_DIR}" \
        2>/dev/null || true
}
