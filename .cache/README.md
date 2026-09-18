# `.cache/` -- everything this repo builds once and reuses

Gitignored except this file: `.gitignore` ignores `**/.cache/` and everything under the repo-root
one, then re-includes this README so the directory explains itself.

**Nothing lives under the repo's own `.cache/` by default any more.** Every cache this repo writes
-- JIT build artefacts, prerendered CPF, generated lowerings, prepared-job packs, tools, job work
dirs, results, pip/spack build caches -- lives under ONE root on scratch,
`${HPCAGENT_BENCH_CACHE}` (default `${SCRATCH}/.hpcagentbench-cache`, `scripts/cache_env.sh`), so
that a git checkout never grows tens of GB of build output. This directory now holds only this
README; it exists so the doc has somewhere obvious to live.

## The two roots

| Variable | Default | What lives there | How to override |
|---|---|---|---|
| `HPCAGENT_BENCH_WEIGHTS_DIR` | `${FAST_SCRATCH}/.hpcagentbench-cache` | Model weights (iopsstor), via `HF_HOME=$HPCAGENT_BENCH_WEIGHTS_DIR/hf` | Set `HPCAGENT_BENCH_WEIGHTS_DIR`, or `HF_HOME` directly |
| `HPCAGENT_BENCH_CACHE` (alias: `JIT_CACHE_ROOT`) | `${SCRATCH}/.hpcagentbench-cache` | Everything below | Set `HPCAGENT_BENCH_CACHE` or `JIT_CACHE_ROOT` (same root; set one, not both to different values) |

Weights are split out on purpose: they are read once per rank at load, by many ranks at once, and
iopsstor measures 11x faster than the general scratch at 16 concurrent readers (job 593523).
Everything else here is small, many, written build output -- the opposite shape, and it must never
contend with a weight load. `HF_HOME` is HuggingFace's own contract (the hub is always
`$HF_HOME/hub`), so it is the ONLY name anything downstream reads for weights; nothing reads
`HPCAGENT_BENCH_WEIGHTS_DIR` directly except `scripts/cache_env.sh` itself.

## Named subdirectories of `${HPCAGENT_BENCH_CACHE}`

| Variable | Default | What lives there | How to override |
|---|---|---|---|
| `HPCAGENT_BENCH_CPF_PRERENDER_DIR` | `${HPCAGENT_BENCH_CACHE}/.cpf-prerender` | Pre-rendered Canonical Parallel Form: content-addressed cache + views. An experiment INPUT, not disposable -- see below. | `HPCAGENT_BENCH_CPF_PRERENDER_DIR` |
| `HPCAGENT_BENCH_TOOLS_DIR` | `${HPCAGENT_BENCH_CACHE}/tools` | Build tools not baked into the image (e.g. ppcg), one versioned subdir plus a `<tool>` symlink | `HPCAGENT_BENCH_TOOLS_DIR` |
| `HPCAGENT_BENCH_RUNS_ROOT` | `${HPCAGENT_BENCH_CACHE}/runs` | Deterministic-framework job work dirs (canon columns, smoke sweeps, opt-report passes) | `HPCAGENT_BENCH_RUNS_ROOT` |
| `HPCAGENT_BENCH_RESULTS_DIR` | `${HPCAGENT_BENCH_CACHE}/results` | Persistent, cross-job results store (`canon.db` and siblings) | `HPCAGENT_BENCH_RESULTS_DIR` |
| `HPCAGENT_BENCH_GENERATED_CACHE_HOST` | `${HPCAGENT_BENCH_CACHE}/generated` | Emitted reference lowerings (`numpyto_*` output), keyed by kernel source content | `HPCAGENT_BENCH_GENERATED_CACHE_HOST` |
| `HPCAGENT_BENCH_PACK_ROOT` | `${HPCAGENT_BENCH_CACHE}/packs` | One manifest per prepared job (`experiments/prepare_job.sh`) | `HPCAGENT_BENCH_PACK_ROOT` or `PACK_ROOT` |
| `HPCAGENT_BENCH_PIP_CACHE_DIR` | `${HPCAGENT_BENCH_CACHE}/pip` | pip wheel cache for the login-side venv rebuild | `HPCAGENT_BENCH_PIP_CACHE_DIR` or `PIP_CACHE_DIR` |
| `HPCAGENT_BENCH_SPACK_BUILDCACHE_DIR` | `${HPCAGENT_BENCH_CACHE}/spack-buildcache` | Spack binary buildcache for image builds | `HPCAGENT_BENCH_SPACK_BUILDCACHE_DIR` or `SPACK_BUILDCACHE` |
| `HPCAGENT_BENCH_TMP_DIR` | `${HPCAGENT_BENCH_CACHE}/tmp` | Scratch `TMPDIR` for compile-heavy jobs (`/tmp` is tmpfs on Beverin) | `HPCAGENT_BENCH_TMP_DIR` or `TMPDIR` |
| *(unnamed)* `jit/<image>/` | under `${HPCAGENT_BENCH_CACHE}` directly (`.triton`, `.vllm`, `.aiter`, `.inductor`, `.xdg`, `.home`, `.torch-ext`, each `/<engine-key>`) | aiter, triton, inductor, torch-extension and vLLM JIT artefacts, derived by `run_vllm_node` in `experiments/run_cluster.sh` | The seven engine env vars directly (`AITER_JIT_DIR`, `TRITON_CACHE_DIR`, ...) |

**Deferred, not yet wired to this root:** `HPCAGENT_BENCH_BASE_IMAGES_DIR` and
`HPCAGENT_BENCH_CE_IMAGES_DIR` are declared in `scripts/cache_env.sh` (defaulting to
`${HPCAGENT_BENCH_CACHE}/images/{base,ce}`) but the image-build scripts (`build_common.sh`'s
`ce_cache_base_image`, each `ce-images/*/build.sh`'s `CE_DIR`) still default to
`${SCRATCH}/base-images` and `${SCRATCH}/ce-images`. `ce-images/` is where a PROMOTED image an EDF
currently points at may live, not only build scratch -- repointing it needs an explicit decision
plus a re-run of `install_edfs.sh`, not a silent default change.

## Node-local JIT write layer

`${HPCAGENT_BENCH_CACHE}` is on `${SCRATCH}`, which is NFS on Beverin. Several engines compiling into
the same content-addressed files there at once turned "a file another client just replaced" into
`OSError: [Errno 116] Stale file handle` for whichever engine read mid-rewrite -- jobs 640074,
640075 and 640090 (three `oss120b` arms started within a minute of each other), each with one TP
worker dead and the engine hung. `run_vllm_node` in `experiments/run_cluster.sh` (both the vLLM and
SGLang paths) no longer lets an engine write the shared tree directly for the triton/inductor/vLLM
slice of the JIT cache: `experiments/jit_cache_layer.sh` seeds a node-local copy at
`${TMPDIR:-/tmp}/hpcagent-bench-jit-<job>-<rank>` from the shared tree, the engine compiles into
that copy, and it is published back to the shared tree -- add-only, one entry staged and renamed
into place at a time, so a reader never sees a partial one -- once `/health` answers, then again
every `HPCAGENT_BENCH_JIT_PUBLISH_INTERVAL_SECONDS` (default 1800). `HPCAGENT_BENCH_JIT_LOCAL=0`
disables the layer and writes the shared tree directly, as before. `AITER_JIT_DIR` is not part of
this layer; it is seeded once from the image's own prebuild and still writes the shared tree.

## The one rule that is not cosmetic

**The JIT slice is keyed by IMAGE and must stay that way.** Those artefacts are compiled against one
ROCm/aiter build, and a rank that loads a mismatched `.so` fails late or silently -- the same shape
as the shared-PCH contamination. Never flatten an engine's key into one directory.

`generated/` is the opposite and deliberately NOT image-keyed: a lowering is pure text derived from
`<module>_numpy.py`, its filename already carries a sha256 of that source, so an entry is valid for
any image and an edited kernel misses rather than serving stale code.

## What is NOT disposable

Pre-rendered canonical parallel forms (`${HPCAGENT_BENCH_CPF_PRERENDER_DIR}`). They are an
experiment INPUT, not something rebuilt on demand: an arm served a different form measures a
different treatment, so the "delete the cache, lose only time" rule below does not hold for them,
and a live view under `${HPCAGENT_BENCH_CPF_PRERENDER_DIR}/views/<name>` must never be renamed or
moved out from under a running arm. Filled by `experiments/prerender_cpf.sbatch` (see
`hpcagent_bench/cpf_cache.py`). Everything ELSE under `${HPCAGENT_BENCH_CACHE}` is reproducible from
the repo plus an image: deleting it costs time and never correctness.

## Filling it

`experiments/prepare_job.sh` writes `generated/` and `packs/`, and `jit/<image>/` derivation happens
in `run_cluster.sh`; `run_cluster.sh` calls `prepare_job.sh` first inside each arm.

## Job work dirs (deterministic-framework submitters)

`${HPCAGENT_BENCH_RUNS_ROOT}` (default `${HPCAGENT_BENCH_CACHE}/runs`, `scripts/cache_env.sh`) is
the root a deterministic-framework submitter -- `experiments/submit-canon-llr40.sh` today, any
canon/smoke/opt-report submitter going forward -- derives its OWN job work dir under, as
`${HPCAGENT_BENCH_RUNS_ROOT}/<job-kind>/<name>-<stamp>` (canon: `<job-kind>` is `canon`, `<name>` is
the roster tag, `<stamp>` is the submit date). Never spelled out as a bare `${SCRATCH}/<name>` path:
before this, `submit-canon-llr40.sh` defaulted `OUT_ROOT` straight to
`${SCRATCH}/canon-llr40-<stamp>`, a directory nothing ever swept, and a compiler-baseline sweep
leaves one DaCe build tree (`dacecache-<column>[_rank<N>]`) per column in it -- routinely the bulk
of the directory's size.

A submitter that sources `scripts/cache_env.sh` and derives its `OUT_ROOT` this way gets, for free,
`experiments/canon_column.sh`'s own two-part cleanup:

* **the per-rank `run-framework` shard DB moves INTO the job dir.** `record.db_path`
  (`hpcagent_bench/config.yaml`) is repo-relative by default, so an unmanaged out_root's shard DBs
  (`hpcagent_bench<rank>.db`) land beside the checkout itself -- four ranks x seven columns of one
  campaign is `hpcagent_bench{0..3}.db` sitting in the repo root. A managed out_root instead gets
  `HPCAGENT_BENCH_RECORD_DB_PATH` pointed at `<out_root>/db/<column>/hpcagent_bench.db` (one
  directory PER COLUMN, since several columns of one campaign share an out_root and must not race
  the same shard file).
* **the work dir is cleaned up at the END OF EACH COLUMN'S JOB, but only after a VERIFIED merge.**
  Once a column's srun step returns, `canon_column.sh` folds its CSV rows into the persistent,
  cross-run `${HPCAGENT_BENCH_RESULTS_DIR}/canon.db` (`scripts/merge_canon_results.py` -- append-
  only, unlike the whole-sweep-rebuild `scripts/collect_canon.py` the reproducibility repos call,
  because sibling columns are often still writing beside this one), checks the merged row count
  against an INDEPENDENT count of the same CSVs, and deletes that column's `dacecache-<column>*`
  build tree and `db/<column>/` shard DB only when the two counts agree. A mismatch (or any other
  merge failure) keeps every one of the column's files and prints why to the job's own `--output`
  log, which sits in `out_root` itself and is deliberately never deleted by this cleanup. The CSV
  (`<column>.rank<N>.csv`) is likewise never deleted: it is the documented hand-off the
  reproducibility repos' own `collect_canon.py` pass reads (`experiments/README.md`'s canon
  section, `reproducibility/canon/artifacts/README.md`), and only the persistent-DB copy is a bonus
  for this repo's own queries, not a replacement for it.

`${HPCAGENT_BENCH_RESULTS_DIR}` (default `${HPCAGENT_BENCH_CACHE}/results`) is the persistent
destination above: a job dir is reusable scratch a submitter may name however it likes, but the
RESULTS a job produced must survive its own job dir being cleared, so they are merged out to a fixed
root instead. A second deterministic-framework family adds its own file here (`<family>.db`) rather
than renaming `canon.db`.

## Migrating an older checkout

Before this unification, `generated/` and `packs/` defaulted into this directory
(`${HPCAGENT_BENCH_REPO}/.cache/generated`, `${HPCAGENT_BENCH_REPO}/.cache/packs`) and pip/spack
caches defaulted straight onto `${SCRATCH}` (`.cache/pip`, `pip-cache`, `spack-buildcache`) with no
named variable at all. `scripts/migrate_cache_layout.sh` prints the move from those locations (and
from the pre-unification JIT tree, unchanged in this pass) to the layout above; it is a dry run by
default and never runs automatically.
