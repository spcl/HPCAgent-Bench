# CE images

Four AMD images, one directory each, one Dockerfile each, end to end. A fifth directory holds the
CUDA judge+agent recipe, written and parse-checked but not built on this cluster (no aarch64
partition here).

| directory | image | what it is |
|---|---|---|
| `judge-agent-amd/` | `optarena-judge-agent-amd.sqsh` | judge + agent: compilers, HPC libraries, solvers, profilers, frameworks |
| `vllm/` | `optarena-vllm-candidate.sqsh` | vLLM **0.23.0** inference, kept for oss120b's mxfp4 path |
| `vllm-0271/` | `optarena-vllm-0271-candidate.sqsh` | vLLM **0.27.1** inference |
| `sglang/` | `optarena-sglang-candidate.sqsh` | SGLang inference, the dominant serving engine |
| `judge-agent-cuda/` | not built here | the CUDA counterpart of `judge-agent-amd` |

`IMAGE_REQUIREMENTS.md` is the specification -- what each image must carry and why each entry is
load-bearing. This file is only how to build and check one.

## Build

Every image builds the same way: one node, its own `build.sbatch`, the directory passed in.

```bash
cd $SCRATCH/optarena
B=$PWD/containers/cluster/ce-images
sbatch --export=ALL,IMAGE_DIR=$B/judge-agent-amd $B/judge-agent-amd/build.sbatch
sbatch --export=ALL,VLLM_DIR=$B/vllm                $B/vllm/build.sbatch
sbatch --export=ALL,VLLM_0271_DIR=$B/vllm-0271      $B/vllm-0271/build.sbatch
sbatch --export=ALL,SGLANG_DIR=$B/sglang            $B/sglang/build.sbatch
```

The directory variable is REQUIRED and its name still differs per image; a plain `sbatch` with
none of them fails in a second rather than building the wrong thing.

Logs land in `$SCRATCH/ce-images/logs/`. The `.sqsh` and a `.digest` recording the image digest
land beside them in `$SCRATCH/ce-images/`. **The digest is the version**, not the tag: a `-v5` in
a name is what once made two different images look like the same thing in a results table.

Stagger the submissions. GitHub rate-limits the shared egress IP when several builds clone at
once; every network git call goes through `gitretry` (ten tries over ~29 minutes), but not
tripping the limiter is cheaper than surviving it.

## Verify

`verify_image.py` runs INSIDE an image and checks every library the benchmark can emit a call to,
resolved the way it will be resolved at grading time. Its exit status is the number of REQUIRED
things missing, so a gate can use it directly.

```bash
sbatch --export=ALL,IMAGE=$SCRATCH/ce-images/optarena-judge-agent-amd.sqsh,PROFILE=judge-agent-amd \
       containers/cluster/ce-images/verify_image.sbatch
```

`PROFILE` is `judge-agent-amd`, `vllm` or `sglang`. A serving image is held to the inference stack
and the fabric; the judge+agent image is held to the whole toolchain, solver set and framework
list.

`scripts/smoke_gpu_profilers.sh` is the other check worth running before an image goes live: it
reads ARTIFACTS rather than exit codes, reconciling `SQ_WAVES` against the launch geometry so a
profiler that runs and drops its rows fails instead of passing.

## Install an EDF

An image is reached from a job through an EDF in `~/.edf`. Point it at the `.sqsh`, mount the
filesystems the job needs, and nothing else:

```toml
image = "${SCRATCH}/ce-images/optarena-judge-agent-amd.sqsh"
mounts = ["/capstor/:/capstor/", "/iopsstor/:/iopsstor/", "${SCRATCH}:${SCRATCH}"]
workdir = "${SCRATCH}"
```

Then `srun --environment=<edf-name> ...`. Do not add an EDF for a CANDIDATE image: an EDF is how a
candidate becomes live by accident. `verify_image.sbatch` generates a throwaway one for exactly
this reason.
