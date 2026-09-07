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

## Publishing and pulling

An image can be pushed to a registry so it is PULLED rather than rebuilt -- a rebuild is one node
for hours, a pull is bandwidth.

Opt in at build time by naming the repository; every `build.sh` takes it:

```
REGISTRY_USER=<user> REGISTRY_TOKEN=<token> \
  PUSH_REPO=docker.io/<user>/optarena-judge-agent-amd PUSH_TAGS="v6 latest" \
  IMAGE_DIR=containers/cluster/ce-images/judge-agent-amd \
  sbatch containers/cluster/ce-images/judge-agent-amd/build.sbatch
```

The push runs after the squashfs is written and verified, so a registry failure costs the upload
and never the artifact the job exists to produce. Credentials come from the environment and are
never stored in the repo; use a scoped access token, not a password.

**It must happen in the build job, and that is not a preference.** podman's graphroot here is
`/dev/shm/$USER/root` -- node-local tmpfs, wiped at the top of every `build.sh` and gone when the
job ends. Between `podman build` and the end of that job is the only window in which an OCI image
exists at all. Afterwards the only artifact is the squashfs, and a squashfs is a flattened
filesystem rather than an OCI image: reimporting one loses the layer structure and the image
config. **An image that was not pushed while it was built has to be rebuilt to be pushed** -- there
is no script that can upload the `.sqsh` files already sitting on scratch.

Pulling, either as a container or straight to the squashfs the CE wants:

```
podman pull docker.io/<user>/optarena-judge-agent-amd:sha-<digest>
enroot import -x mount -o optarena-judge-agent-amd.sqsh \
    docker://docker.io/<user>/optarena-judge-agent-amd:sha-<digest>
```

Every push publishes a `sha-<digest>` tag beside the human-facing ones, because the digest is what
identifies a build -- a mutable tag over two different images is what made a results table
unreadable before.

**Size is the binding constraint.** These images run 37-54 GB. Docker Hub caps an image at 100 GB
and a single LAYER at 10 GB, so they fit but not with much room, and one oversized `RUN` would be
rejected partway through a multi-hour upload. `push_image.sh` therefore measures the image and its
largest layer and refuses BEFORE sending anything; both ceilings are overridable
(`MAX_LAYER_GB`, `MAX_IMAGE_GB`) since other registries differ -- ECR raised its layer limit to
200 GB in August 2026. Publishing the whole set is roughly 350 GB of upload and storage, which is
worth pricing against a registry account before starting.

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
