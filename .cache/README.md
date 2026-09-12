# `.cache/` -- everything this repo builds once and reuses

Gitignored except this file: `.gitignore` ignores `**/.cache/` and everything under the repo-root
one, then re-includes this README so the directory explains itself. Nothing here is an input --
every file is reproducible from the repo plus an image, so deleting the whole directory costs time
and never correctness.

    .cache/
      jit/<image>/     aiter, triton, inductor, torch-extension and vLLM JIT artefacts
      generated/       emitted reference lowerings (numpyto_* output)
      packs/           one manifest per prepared job

## Why the repo and not scratch

Same filesystem either way -- the checkout and `$SCRATCH` are both on capstor -- so this is about
finding it, not speed. It also outlives more: iopsstor purges at 14 days against capstor's 30.

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
the directory, lose only time -- does not hold for them. They live under the campaign that renders
them, `$SCRATCH/campaigns/<tag>/<target>/{forms,dropin}`; `experiments/campaign_dirs.sh` names the
layout.

## Filling it

`experiments/prepare_job.sh` writes all three, and `run_cluster.sh` calls that first inside each
arm. Nothing else should write here.
