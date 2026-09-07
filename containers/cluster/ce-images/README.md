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

## Names and versions

`images.env` says which built image each NAME resolves to, and it is the only place a version is
written down. `install_edfs.sh` renders the EDFs from it:

```bash
containers/cluster/ce-images/install_edfs.sh
```

That writes `~/.edf/optarena-amd-mi300-latest` (follows `images.env`) and
`~/.edf/optarena-amd-mi300-<version>` (pinned). A campaign that names `latest` moves with a
promotion; one that must not move names the version. Promoting is one edit to `images.env` plus a
re-run with `ALLOW_REPOINT=1`, which is required because repointing `latest` moves every unpinned
job including ones already queued.

This exists because nothing pointed at an image by role: 169 campaign `.env` files still say
`optarena-amd-mi300-v4` and `canon_column.sh` says v5, which is not a decision anyone made -- it
is where each file stopped being edited. Those files are deliberately left alone while their
campaigns run; new ones should name `latest`.

`install_edfs.sh` is also what makes a fresh clone usable. `~/.edf` is not in the repo, so a
checkout on another account can reach no image at all, and copying someone else's EDF carries
their absolute scratch path into your jobs. Rendering from `${SCRATCH}` avoids both.

**The digest is still the identity of a build.** `latest` is for launching; a results table quotes
the `.digest` written beside the squashfs.

## Publishing and pulling

An image can be pushed to a registry so it is PULLED rather than rebuilt -- a rebuild is one node
for hours, a pull is bandwidth.

Every build writes an **OCI archive** (`<name>.oci.tar`) beside the squashfs. This is what makes
publishing independent of building. The squashfs cannot serve the purpose: it is a flattened
filesystem, so reimporting one collapses the image to a single layer far past Docker Hub's 10 GB
per-layer ceiling and drops the image config as well. The archive keeps the layers and the config
-- verified by round-tripping a multi-layer image through one into a fresh graphroot and reading
back the same layer count and the same `Env`.

Push during the build, when credentials are already to hand:

```bash
REGISTRY_USER=<user> REGISTRY_TOKEN=<token> \
  PUSH_REPO=docker.io/<user>/optarena-judge-agent-amd PUSH_TAGS="v6 latest" \
  IMAGE_DIR=containers/cluster/ce-images/judge-agent-amd \
  sbatch containers/cluster/ce-images/judge-agent-amd/build.sbatch
```

Or later, from the archive, with no rebuild:

```bash
REGISTRY_USER=<user> REGISTRY_TOKEN=<token> \
  PUSH_REPO=docker.io/<user>/optarena-judge-agent-amd \
  ./push_image.sh --from-archive $SCRATCH/ce-images/optarena-ce-amd-mi300-v6.oci.tar v6 latest
```

**Run that on a compute node.** The archive loads into an isolated graphroot, which must be tmpfs:
a rootless overlay cannot create its pivot dir on Lustre, so layer extraction there fails outright.
That means the node needs free RAM for the decompressed image, 60+ GB for the judge+agent one --
the same requirement the build has.

Pulling, straight to the squashfs the CE wants:

```bash
REGISTRY_NAMESPACE=<user> ./pull_image.sh judge-agent-amd sha-<digest>
```

Also a compute node, and for the same reason: enroot unpacks every layer before writing the
squashfs. `pull_image.sh` refuses to overwrite a squashfs any EDF in `~/.edf` mounts, so a running
arm never reads a half-written image; pull to `OUT=<candidate name>` and promote by rename.

Every push publishes a `sha-<digest>` tag beside the human-facing ones, because the digest is what
identifies a build -- a mutable tag over two different images is what made a results table
unreadable before. Prefer the `sha-` tag when reproducing.

**Size is the binding constraint.** These images run 37-62 GB. Docker Hub caps an image at 100 GB
and a single LAYER at 10 GB, so they fit but not with much room. `push_image.sh` measures the image
and its largest layer and refuses BEFORE sending anything; both ceilings are overridable
(`MAX_LAYER_GB`, `MAX_IMAGE_GB`) since other registries differ -- ECR raised its layer limit to
200 GB in August 2026. Publishing the whole set is roughly 350 GB of upload and storage, which is
worth pricing against a registry account before starting.

## Build

Every image builds the same way: one node, its own `build.sbatch`, the directory passed in.

```bash
cd $SCRATCH/optarena
B=$PWD/containers/cluster/ce-images
sbatch --export=ALL,IMAGE_DIR=$B/judge-agent-amd $B/judge-agent-amd/build.sbatch
sbatch --export=ALL,IMAGE_DIR=$B/vllm            $B/vllm/build.sbatch
sbatch --export=ALL,IMAGE_DIR=$B/vllm-0271       $B/vllm-0271/build.sbatch
sbatch --export=ALL,IMAGE_DIR=$B/sglang          $B/sglang/build.sbatch
```

`IMAGE_DIR` is REQUIRED and is now the same name for every image; the old per-image spellings
(`VLLM_DIR`, `VLLM_0271_DIR`, `SGLANG_DIR`) still work as fallbacks. A plain `sbatch` with none of
them fails in a second rather than building the wrong thing -- Slurm spools the batch script, so
`BASH_SOURCE` points into /var/spool/slurmd and the directory cannot be derived.

Logs land in `$SCRATCH/ce-images/logs/`. The `.sqsh` and a `.digest` recording the image digest
land beside them in `$SCRATCH/ce-images/`. **The digest is the version**, not the tag: a `-v5` in
a name is what once made two different images look like the same thing in a results table.

Stagger the submissions. GitHub rate-limits the shared egress IP when several builds clone at
once; every network git call goes through `gitretry` (ten tries over ~29 minutes), but not
tripping the limiter is cheaper than surviving it.

## dace, and nothing from outside the container

The image clones dace itself and bakes an exact commit (`DACE_COMMIT`, resolved by the builder
from the tip of `extended`, recorded in `/opt/dace.commit`). That is what makes the image
self-contained: the dace a run uses is fixed by the image digest, not by the state of anyone's
scratch directory.

Two things keep it that way.

`PYTHONSAFEPATH = "1"` in the EDF. dace is installed editable, so `import dace` resolves through
a finder -- and a plain DIRECTORY named `dace` on `sys.path` beats that finder and imports as an
empty namespace package. `sys.path` starts with the CWD, the EDF's `workdir` is `${SCRATCH}`, and
`${SCRATCH}/dace` is the live host checkout. Without this, `import dace` from the workdir
SUCCEEDS and returns a module with `__file__` None and no `SDFG`, surfacing later as a traceback
that reads like a packaging fault. Measured on v6: unusable from `${SCRATCH}` and from `/opt`,
usable from `/tmp`. `tests/test_edf_contract.py` gates it.

`dace_refresh.sh` to move forward. It advances `/opt/dace` -- the image's own tree, never
`${SCRATCH}/dace` -- to the tip of `extended`, so a run is not stuck behind the branch until the
next rebuild:

```bash
srun --environment=optarena-amd-mi300-latest containers/cluster/ce-images/dace_refresh.sh
```

Writes land in the container's ephemeral upper layer, so it is per-job and the image is
unchanged. A fetch failure is deliberately NOT fatal -- the baked commit is a working dace, and
refusing to start on a GitHub hiccup trades a slightly stale run for no run. It prints the live
commit either way, and that is the line a results table should quote, since the image digest no
longer determines the dace commit once this has run.

**Not yet wired into `run_cluster.sh`.** `beverin.sbatch` execs it by path, and campaigns are
running against it; editing a script bash is part-way through is how a live arm breaks.

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

An image is reached from a job through an EDF in `~/.edf`. Use `install_edfs.sh` rather than
hand-copying: it renders from the template in `judge-agent-amd/edf.toml.example`, which is the
file `tests/test_edf_contract.py` gates, so the PATH contract holds for every EDF it produces.
Hand-edited copies are how `/opt/venv/bin` went missing once already -- and without it `python3`
resolves to the system interpreter, which imports neither dace nor cupy.

```bash
containers/cluster/ce-images/install_edfs.sh          # latest + pinned
ALLOW_REPOINT=1 containers/cluster/ce-images/install_edfs.sh   # after bumping images.env
```

Then `srun --environment=optarena-amd-mi300-latest ...`.

Do not add an EDF for a CANDIDATE image: an EDF is how a candidate becomes live by accident.
`verify_image.sbatch` generates a throwaway one for exactly this reason.
