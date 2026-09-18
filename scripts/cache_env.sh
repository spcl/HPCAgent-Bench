#!/usr/bin/env bash
# WHERE THE CACHES LIVE. Source it; never copy a path out of it.
#
#   . "${HPCAGENT_BENCH_REPO}/scripts/cache_env.sh"
#
# TWO ROOTS, because the two kinds of data have opposite shapes -- this split is measured, not
# stylistic:
#
#   WEIGHTS -> HPCAGENT_BENCH_WEIGHTS_DIR, iopsstor by default. Read once per rank at load, by many
#              ranks at once; that filesystem is 11x faster at 16 concurrent readers (job 593523).
#   CACHE   -> HPCAGENT_BENCH_CACHE, the general scratch. Everything else this repo builds once and
#              reuses: JIT build artefacts, prerendered CPF, tools, job work dirs, results,
#              generated lowerings, prepared-job packs, pip/spack build caches, tmp. Small, many,
#              written -- the opposite shape, and it must never contend with a weight load.
#
# ONE NAME PER ROOT. HF_HOME is HuggingFace's own contract (the hub is always $HF_HOME/hub) and
# derives from HPCAGENT_BENCH_WEIGHTS_DIR; nothing downstream should read the weights root
# directly, so two independent spellings of the same path never have a chance to drift apart.
# Every other named var below (HPCAGENT_BENCH_CPF_PRERENDER_DIR, _TOOLS_DIR, _RUNS_ROOT, ...) is a
# documented child of HPCAGENT_BENCH_CACHE -- see .cache/README.md for the full table. Setting any
# of them yourself is still honoured: this file only changes DEFAULTS, never a caller's own pin.
#
# JIT_CACHE_ROOT is the pre-unification name for HPCAGENT_BENCH_CACHE -- same root, kept as a full
# alias so a caller that already pins it (a rerun frozen to a cache, a test) keeps working. Setting
# either sets the other; setting both to different values is a caller error this file does not try
# to reconcile.
#
# run_cluster.sh derives the seven per-engine knobs (HOME, XDG_CACHE_HOME, AITER_JIT_DIR,
# VLLM_CACHE_ROOT, TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR, TORCH_EXTENSIONS_DIR) from
# JIT_CACHE_ROOT itself, keyed by ${INFERENCE_CE_ENV}, and the inference EDFs deliberately override
# some of them. Exporting those knobs HERE would silently win over both.
#
# In particular AITER_JIT_DIR must NOT be set here. The sglang and vllm EDFs pin it to the
# IN-IMAGE prebuild at /opt/aiter-jit, and job 628077 measured what happens when a host directory
# wins instead: module_aiter_core loads from the host and the image's prebuilt copy goes unused.

: "${FAST_SCRATCH:=/iopsstor/scratch/cscs/${USER:-$(id -un)}}"
export FAST_SCRATCH

# WEIGHTS. ONE variable (HPCAGENT_BENCH_WEIGHTS_DIR), iopsstor by default. HF_HOME derives from it
# and stays the thing everything downstream reads -- see the header.
: "${HPCAGENT_BENCH_WEIGHTS_DIR:=${FAST_SCRATCH}/.hpcagentbench-cache}"
export HPCAGENT_BENCH_WEIGHTS_DIR
export HF_HOME="${HF_HOME:-${HPCAGENT_BENCH_WEIGHTS_DIR}/hf}"

# THE GENERAL CACHE ROOT. No fallback: an unset SCRATCH aborts here instead of landing caches
# somewhere no later job looks. A caller with no scratch (the pre-commit hooks, via
# scripts/run_hook.sh) passes HPCAGENT_BENCH_CACHE or JIT_CACHE_ROOT explicitly.
if [[ -z "${HPCAGENT_BENCH_CACHE:-}" && -z "${JIT_CACHE_ROOT:-}" ]]; then
    : "${SCRATCH:?set SCRATCH (the general scratch root) or pass HPCAGENT_BENCH_CACHE/JIT_CACHE_ROOT explicitly}"
    HPCAGENT_BENCH_CACHE="${SCRATCH}/.hpcagentbench-cache"
fi
: "${HPCAGENT_BENCH_CACHE:=${JIT_CACHE_ROOT}}"
: "${JIT_CACHE_ROOT:=${HPCAGENT_BENCH_CACHE}}"
export HPCAGENT_BENCH_CACHE JIT_CACHE_ROOT

# Prerendered Canonical Parallel Form. Not a JIT artefact and NOT disposable the way the rest of
# this root is: it is an experiment INPUT reused across arms and engines, so it is neither keyed by
# EDF nor purged with the JIT tree, and a live view under it must never be renamed out from under a
# running arm. Name and default are UNCHANGED by this file's unification pass for exactly that
# reason -- see .cache/README.md.
export HPCAGENT_BENCH_CPF_PRERENDER_DIR="${HPCAGENT_BENCH_CPF_PRERENDER_DIR:-${HPCAGENT_BENCH_CACHE}/.cpf-prerender}"

# Build tools not (yet) baked into the agent/judge image -- e.g. ppcg, whose only runtime
# dependency the image ships (see hpcagent_bench/ppcg_transform.py) is the `ppcg` binary itself.
# Each tool keeps its own versioned subdirectory (`<tool>-<version>`) plus a `<tool>` symlink a
# build script repoints atomically, so a lookup that only knows the STABLE name never has to learn
# a version. Exported here, not hardcoded in the lookup, so a rebuild that moves the root (or a
# host with the tool somewhere else entirely) never needs a code change.
export HPCAGENT_BENCH_TOOLS_DIR="${HPCAGENT_BENCH_TOOLS_DIR:-${HPCAGENT_BENCH_CACHE}/tools}"

# Deterministic-framework job work dirs (canon compiler-baseline columns and siblings: smoke sweeps,
# opt-report passes). Same shape as the JIT tree -- small-ish, many, WRITTEN by the job, one tree
# per job. Before this existed, submit-canon-llr40.sh defaulted out_root to
# ${SCRATCH}/canon-<tag>-<stamp> directly: a bare-scratch directory nothing ever swept, accumulating
# one DaCe build tree (dacecache-<column>[_rank<N>]) per column forever. A submitter derives its own
# job dir under this root as ${HPCAGENT_BENCH_RUNS_ROOT}/<job-kind>/<name>-<stamp> (mirroring
# HPCAGENT_BENCH_CACHE's own "root exported, suffix appended by the caller" pattern) and MUST NOT
# invent a path under ${SCRATCH} instead -- see .cache/README.md's "Job work dirs" section.
export HPCAGENT_BENCH_RUNS_ROOT="${HPCAGENT_BENCH_RUNS_ROOT:-${HPCAGENT_BENCH_CACHE}/runs}"

# The PERSISTENT, cross-job results store a job work dir's final per-kernel outcome is merged into
# before the work dir (build trees, per-rank shard DBs) is deleted -- never the destination a job
# writes its OWN per-rank shards to directly (those still need one file per rank per job; see the
# job-work-dir note above), or two jobs' rank 0 would race the same file. Consumers derive their own
# table/filename under this root (canon_column.sh's finalize step uses
# ${HPCAGENT_BENCH_RESULTS_DIR}/canon.db via scripts/merge_canon_results.py) rather than a single
# hardcoded name, so a second deterministic-framework family can add its own file here without
# renaming this one.
export HPCAGENT_BENCH_RESULTS_DIR="${HPCAGENT_BENCH_RESULTS_DIR:-${HPCAGENT_BENCH_CACHE}/results}"

# Emitted reference lowerings (numpyto_* output), keyed by the CONTENT of each kernel's numpy
# source: an entry is valid for any image and an edited kernel misses rather than serving stale
# code, so this is regenerable and safe to place here. Used to default to
# ${HPCAGENT_BENCH_REPO}/.cache/generated, which grew tens of GB inside a git working tree.
export HPCAGENT_BENCH_GENERATED_CACHE_HOST="${HPCAGENT_BENCH_GENERATED_CACHE_HOST:-${HPCAGENT_BENCH_CACHE}/generated}"

# One manifest per prepared job (experiments/prepare_job.sh), rebuilt every time prepare_job.sh
# runs. Used to default to ${HPCAGENT_BENCH_REPO}/.cache/packs, same "grows inside the checkout"
# problem as generated/ above.
export HPCAGENT_BENCH_PACK_ROOT="${HPCAGENT_BENCH_PACK_ROOT:-${HPCAGENT_BENCH_CACHE}/packs}"

# pip wheel cache (login-side venv rebuilds) and the spack binary buildcache (image builds). Both
# used to default straight to ${SCRATCH}/pip-cache and ${SCRATCH}/spack-buildcache -- disposable,
# rebuildable, and outside every other named cache this repo has.
export HPCAGENT_BENCH_PIP_CACHE_DIR="${HPCAGENT_BENCH_PIP_CACHE_DIR:-${HPCAGENT_BENCH_CACHE}/pip}"
export HPCAGENT_BENCH_SPACK_BUILDCACHE_DIR="${HPCAGENT_BENCH_SPACK_BUILDCACHE_DIR:-${HPCAGENT_BENCH_CACHE}/spack-buildcache}"

# Scratch tmp, for callers that currently point TMPDIR at ${SCRATCH}/.tmp by hand (tools/suite.sbatch,
# tools/rebuild_venv.sh): /tmp is tmpfs here and a compile-heavy job fills it.
export HPCAGENT_BENCH_TMP_DIR="${HPCAGENT_BENCH_TMP_DIR:-${HPCAGENT_BENCH_CACHE}/tmp}"

# PROPOSED, NOT YET WIRED IN: container image caches. build_common.sh's ce_cache_base_image()
# (${SCRATCH}/base-images) and each ce-images/*/build.sh's CE_DIR (${SCRATCH}/ce-images) still use
# their own literal defaults -- ce-images/ is where a PROMOTED image an EDF is currently pointing at
# may live, not only build scratch, so repointing its default needs an explicit decision (which
# image-build scripts are safe to move, and a re-run of install_edfs.sh) rather than a silent
# default change here. See the cache-unification report for the open question.
: "${HPCAGENT_BENCH_BASE_IMAGES_DIR:=${HPCAGENT_BENCH_CACHE}/images/base}"
: "${HPCAGENT_BENCH_CE_IMAGES_DIR:=${HPCAGENT_BENCH_CACHE}/images/ce}"
export HPCAGENT_BENCH_BASE_IMAGES_DIR HPCAGENT_BENCH_CE_IMAGES_DIR

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
    mkdir -p "${HF_HOME}" "${HPCAGENT_BENCH_CACHE}" "${HPCAGENT_BENCH_CPF_PRERENDER_DIR}" \
        "${HPCAGENT_BENCH_TOOLS_DIR}" "${HPCAGENT_BENCH_RUNS_ROOT}" "${HPCAGENT_BENCH_RESULTS_DIR}" \
        "${HPCAGENT_BENCH_GENERATED_CACHE_HOST}" "${HPCAGENT_BENCH_PACK_ROOT}" \
        2>/dev/null || true
}
