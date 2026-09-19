# `.cache/` -- everything this repo builds once and reuses

Gitignored except this file: `.gitignore` ignores `**/.cache/` and everything under the repo-root
one, then re-includes this README so the directory explains itself. Nothing here is an input --
every file is reproducible from the repo plus an image, so deleting the whole directory costs time
and never correctness.

    .cache/
      generated/       emitted reference lowerings (numpyto_* output)
      packs/           one manifest per prepared job

`jit/<image>/` (aiter, triton, inductor, torch-extension and vLLM JIT artefacts) is NOT here: it
lives under `${JIT_CACHE_ROOT}` (default `${SCRATCH}/.hpcagentbench-cache`, `scripts/cache_env.sh`),
moved out of the checkout because it grows tens of GB of build output that a git working tree
should not carry -- see the "why the repo and not scratch" note below for `generated/`/`packs/`,
which is a different, smaller kind of artefact.

**No `$SCRATCH` at all** (a bare local invocation with no cluster session -- CI, a laptop clone, or
just a shell that never sourced `experiments/env.sh`): `JIT_CACHE_ROOT` falls back to
`${HPCAGENT_BENCH_REPO}/.cache/jit` -- the checkout's own root, resolved once by
`experiments/env.sh` and reused by every script instead of each guessing its own answer (`~/.cache`,
a bare `__file__` walk, the checkout's parent directory all used to disagree). The Python side of
the same default is `hpcagent_bench.paths.repo_root()`/`scratch_root()`. This only fires when
`HPCAGENT_BENCH_REPO` is set (every real caller has one) and neither `$SCRATCH` nor an explicit
`JIT_CACHE_ROOT` is; a bare `. cache_env.sh` with nothing configured at all still aborts loudly
rather than guessing. `scripts/run_tests.sh --container` (a real Slurm submission) always has
`$SCRATCH` and does not need this path -- it exists for the scripts that run with neither.

## Node-local JIT write layer

`${JIT_CACHE_ROOT}` is on `${SCRATCH}`, which is NFS on Beverin. Several engines compiling into the
same content-addressed files there at once turned "a file another client just replaced" into
`OSError: [Errno 116] Stale file handle` for whichever engine read mid-rewrite -- jobs 640074,
640075 and 640090 (three `oss120b` arms started within a minute of each other), each with one TP
worker dead and the engine hung. `run_vllm_node` in `experiments/run_cluster.sh` (both the vLLM and
SGLang paths) no longer lets an engine write the shared tree directly for the triton/inductor/vLLM
slice of `jit/`: `experiments/jit_cache_layer.sh` seeds a node-local copy at
`${TMPDIR:-/tmp}/hpcagent-bench-jit-<job>-<rank>` from the shared tree, the engine compiles into
that copy, and it is published back to the shared tree -- add-only, one entry staged and renamed
into place at a time, so a reader never sees a partial one -- once `/health` answers, then again
every `HPCAGENT_BENCH_JIT_PUBLISH_INTERVAL_SECONDS` (default 1800). `HPCAGENT_BENCH_JIT_LOCAL=0`
disables the layer and writes the shared tree directly, as before. `AITER_JIT_DIR` is not part of
this layer; it is seeded once from the image's own prebuild and still writes the shared tree.

## Node-local agent caches, hard-linked task material, dace_numeric (2026-09-19 inode fix)

The 2026-09-19 inode-quota incident (1.67M vs 1M on `$SCRATCH`) had three sources, all fixed the
same way -- move the many-small-files tree off the swept, shared root:

- **Agent JIT/pip caches.** `TRITON_CACHE_DIR`/`XDG_CACHE_HOME` were never set for an agent, so
  every episode's compiler defaulted to `$HOME/.triton` and `$HOME/.cache` under the persistent
  workdir (119k and 27k files/campaign, never swept). `agent_driver.worker_cache_root()` now points
  both at `${TMPDIR:-/tmp}/hpcagent-bench-agent-cache-<job>-<node>-<worker>`, removed when the
  worker exits.
- **Hard-linked task reference material.** `materialize_shared.sh` used to `cp` each kernel's
  reference files into every job's `shared/tasks/<kernel>/` (42k duplicates of the same repo
  files across a campaign's history). It now hard-links them (`ln -f`, falling back to `cp` only
  across a filesystem boundary, `EXDEV`) -- same inode, no extra file.
- **`dace_numeric` build tree.** The numerics harness's DaCe probe used to build under a bare
  `$SCRATCH/hpcagent_bench/dace_numeric` (35k inodes, outside the unified cache). `dace_build_root()`
  (`tests/numerical_oracle.py`) now builds under `${JIT_CACHE_ROOT}/dace_numeric` (else
  `HPCAGENT_BENCH_CACHE`), same root as `jit/`.

## Why the repo and not scratch (`generated/`, `packs/`)

Same filesystem either way -- the checkout and `$SCRATCH` are on the same scratch mount -- so this
is about finding it, not speed. It also outlives more: `FAST_SCRATCH` (iopsstor) purges at 14 days
against `SCRATCH`'s 30 (that retention number was measured on the retired Lustre scratch mount;
unconfirmed on the current `$SCRATCH` -- see `scripts/cache_env.sh`).

## The one rule that is not cosmetic

**`jit/` is keyed by IMAGE and must stay that way.** Those artefacts are compiled against one
ROCm/aiter build, and a rank that loads a mismatched `.so` fails late or silently -- the same shape
as the shared-PCH contamination. Never flatten `jit/<image>/` into one directory.

`generated/` is the opposite and deliberately NOT image-keyed: a lowering is pure text derived from
`<module>_numpy.py`, its filename already carries a sha256 of that source, so an entry is valid for
any image and an edited kernel misses rather than serving stale code.

## What is NOT here

Pre-rendered canonical parallel forms. They are an experiment INPUT, not something rebuilt on
demand: an arm served a different form measures a different treatment, and the rule above -- delete
the directory, lose only time -- does not hold for them. They live in the content-addressed cache
under `${HPCAGENT_BENCH_CPF_PRERENDER_DIR}/cache`, and an arm reads one through a view under
`${HPCAGENT_BENCH_CPF_PRERENDER_DIR}/views/<name>` (default `${SCRATCH}/.hpcagentbench-cache/.cpf-prerender`,
`scripts/cache_env.sh`), filled by `experiments/prerender_cpf.sbatch` (see `hpcagent_bench/cpf_cache.py`).

## Filling it

`experiments/prepare_job.sh` writes `generated/` and `packs/` here, and `jit/<image>/` under
`${JIT_CACHE_ROOT}`; `run_cluster.sh` calls it first inside each arm. Nothing else should write here.

## Job work dirs (deterministic-framework submitters)

`${HPCAGENT_BENCH_RUNS_ROOT}` (default `${JIT_CACHE_ROOT}/runs`, `scripts/cache_env.sh`) is the root
a deterministic-framework submitter -- `experiments/submit-canon-llr40.sh` today, any canon/smoke/
opt-report submitter going forward -- derives its OWN job work dir under, as
`${HPCAGENT_BENCH_RUNS_ROOT}/<job-kind>/<name>-<stamp>` (canon: `<job-kind>` is `canon`, `<name>` is
the roster tag, `<stamp>` is the submit date). Same shape as `jit/` -- small-ish, many, WRITTEN by
the job -- so it lives beside it under `${JIT_CACHE_ROOT}`, never spelled out as a bare
`${SCRATCH}/<name>` path: before this, `submit-canon-llr40.sh` defaulted `OUT_ROOT` straight to
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

`${HPCAGENT_BENCH_RESULTS_DIR}` (default `${JIT_CACHE_ROOT}/results`) is the persistent destination
above: a job dir is reusable scratch a submitter may name however it likes, but the RESULTS a job
produced must survive its own job dir being cleared, so they are merged out to a fixed root instead.
A second deterministic-framework family adds its own file here (`<family>.db`) rather than renaming
`canon.db`.
