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

## Why the repo and not scratch (`generated/`, `packs/`)

Same filesystem either way -- the checkout and `$SCRATCH` are on the same scratch mount -- so this
is about finding it, not speed. It also outlives more: iopsstor purges at 14 days against the
general scratch's 30 (that retention number was measured on the retired Lustre scratch mount; unconfirmed
on the current `$SCRATCH` at `/ritom`).

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
