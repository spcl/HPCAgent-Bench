# Containers

Everything that builds, ships or runs inside an image. The judge grades inside an image, the agent
works inside an image, and the model is served from an image; this page builds all of them.

| path | what it is |
|---|---|
| `cluster/ce-images/<image>/` | one image per directory (`Dockerfile`, `build.sh`, `build.sbatch`, `edf.toml.example`) for the CSCS Container Engine: AMD MI300A/MI250X (beverin), NVIDIA GH200 (daint), CPU-only (any node) |
| `cluster/ce-images/images.env` | the image registry: one row per image; every script below reads it |
| `cluster/ce-images/inference/` | serving jobs, weight fetch, serving smokes and gates |
| `agent/` | agent-side prompt fragments, MCP tools, method packets and harness pins. Bound read-only into the agent container at launch, never copied into an image |
| `judge/` | the judge's web-search tool; the image installs only `judge/requirements.txt` |
| `hpcagent_bench.Dockerfile`, `cpu.def`, `judge.def`, `inference.def`, `agentbench.compose.yml` | the generic OCI / Apptainer recipes for a workstation or a non-CSCS cluster (`docs/launch.md`, `docs/hf_dataset_and_harbor.md`) |
| `pluto.Dockerfile` | standalone Pluto (`polycc`) for hosts without a judge image |
| `build-hptt.sh`, `build-tblis.sh`, `stdpar-gate.sh`, `parallelizer-gate.sh`, `install-extra-toolchains.sh` | shared build steps the Dockerfiles `COPY` |
| `LIBRARIES.md` | the numeric libraries an agent may link |

## Where skills come from

There is one copy of every skill page: `hpcagent_bench/skills/<name>/SKILL.md`. No image and no
file under `containers/` carries another.

* The judge image installs the package (`COPY hpcagent_bench` in each `judge` stage), so its
  skills are the package's.
* The agent image carries no `hpcagent_bench` (it holds the references agents are graded against).
  A campaign stages the pages a problems file names into `/shared/skills/` at launch
  (`experiments/make_problems.py --stage-skills`, called by `experiments/materialize_shared.sh`),
  from the submitting checkout.
* `containers/agent/` holds the prompt fragments and MCP tools (bound at launch as
  `/opt/hpcagent-bench-agent`) and the method packets (`agent/packets/<name>/`), which are served
  as tools, not skills.

Adding a skill therefore touches no container file; see [Adding a skill](#adding-a-skill).

## The image registry

`cluster/ce-images/images.env` holds one row per image:

```
role  prefix  platform  dir  partition  profile  candidate  sqsh  edf  template  tag  flags
```

A build writes `candidate`; `promote_image.sh` renames it over `sqsh` once it carries a `.verified`
marker; `install_edfs.sh` renders `~/.edf/<edf>.toml` from `template`; `pull_image.sh` and
`push_images.sbatch` move `sqsh` to and from `REGISTRY_REPO:<tag>`. Sourcing `images.env` also
defines `<PREFIX>_SQSH`, `_EDF_LATEST`, `_TEMPLATE`, `_CANDIDATE` and `_TAG`, which `experiments/`
reads. The `.digest` sidecar is the build's identity: cite it (or the `sha-<digest>` tag), never a
moving tag or EDF name.

| role | platform | image directory | EDF |
|---|---|---|---|
| `judge-agent-amd`, `judge` | amd | `judge-agent-amd` (targets `agent`, `judge`) | `hpcagent-bench-{agent,judge}-mi300-latest` |
| `judge-agent-amd-mi200`, `judge-mi200` | amd | `judge-agent-amd`, built on mi200 | `hpcagent-bench-{agent,judge}-mi200-latest` (+ `judge-mi200-mlscale`) |
| `sglang`, `vllm` | amd | `sglang`, `vllm` | `hpcagent-bench-{sglang,vllm}-mi300-latest` |
| `sglang-mi200` | amd | `sglang-mi200` | `hpcagent-bench-sglang-mi200-latest` |
| `judge-agent-cuda`, `judge-cuda` | gh200 | `judge-agent-cuda` (targets `agent`, `judge`) | `hpcagent-bench-{agent,judge}-gh200-latest` |
| `vllm-cuda` | gh200 | `vllm-cuda` | `hpcagent-bench-vllm-gh200-latest` |
| `judge-agent-cpu`, `judge-cpu` | cpu | `judge-agent-cpu` (targets `agent`, `judge`) | `hpcagent-bench-{agent,judge}-cpu-<arch>-latest` |

The `agent` target is the whole toolchain without `hpcagent_bench`; `judge` is `agent` plus the
installed package. Held-out tests are in neither: the judge reads them from the host checkout.

## Build, verify, promote

Every build writes a candidate squashfs (plus `.digest`, `.sha256` and an `.oci.tar` for publishing),
verifies it inside itself, and leaves promotion to a separate, explicit step. A mounted squashfs is
held by its inode, so promotion is safe while jobs run. There is no layer cache between build jobs.

Run everything below from `containers/cluster/ce-images/` of a checkout under `$SCRATCH` (the EDFs
mount `$SCRATCH` and the iopsstor scratch, not `$HOME`).

### AMD (beverin)

Pull the published images (the default: same bytes as published), then render the EDFs:

```bash
sbatch pull_images.sbatch                               # judge-agent-amd judge sglang vllm
./pull_image.sh judge-agent-amd sha-<digest>            # one role, pinned
./install_edfs.sh
```

Build when changing an image (partition `mi300` is in each `build.sbatch`):

```bash
IMAGE_DIR=$PWD/judge-agent-amd sbatch build_and_verify.sbatch   # both targets, ~2 h warm
IMAGE_DIR=$PWD/sglang          sbatch build_and_verify.sbatch   # ~1 h
IMAGE_DIR=$PWD/vllm            sbatch build_and_verify.sbatch   # ~4 h
# the mi200 pair and sglang-mi200 (gfx90a, spack target zen3)
REPO=$PWD/../../.. IMAGE_DIR=$PWD/judge-agent-amd \
  sbatch --partition=mi200 --cpus-per-task=64 --gpus-per-node=8 build_and_verify.sbatch
REPO=$PWD/../../.. IMAGE_DIR=$PWD/sglang-mi200 \
  sbatch --partition=mi200 --cpus-per-task=64 --gpus-per-node=8 build_and_verify.sbatch

DRY_RUN=1 ./promote_image.sh --all      # what would move
./promote_image.sh --all                # rename + sidecars + EDFs
```

Engine versions are build args: `EXTRA_BUILD_ARGS="VLLM_VERSION=0.28.0 AITER_REF=..." sbatch vllm/build.sbatch`.
Re-verify a candidate without rebuilding with `VERIFY_ONLY=1` on `build_and_verify.sbatch`, or:

```bash
IMAGE=$SCRATCH/ce-images/<candidate>.sqsh PROFILE=<profile> sbatch verify_image.sbatch
```

Fabric checks after a judge-agent build (name the candidate: the live names still hold the old image):

| job | nodes | answers |
|---|---|---|
| `mpi_check.sbatch` | 1 | which libfabric and provider MPI mapped, GPU-aware allreduce, PETSc HIP |
| `mpi_multinode_check.sbatch` | 2 | a cross-node MPI transfer rides cxi |
| `rccl_hook_check.sbatch` | 2 | RCCL selects the OFI plugin, not its TCP fallback (any role) |
| `inference/aiter_mla_check.sbatch` | 1 | aiter MLA kernels agree with an fp32 reference |

```bash
IMAGE=$SCRATCH/ce-images/hpcagent-bench-ce-amd-mi300-candidate.sqsh sbatch --dependency=afterok:<build job> mpi_check.sbatch
```

### NVIDIA GH200 (daint)

The scripts write no account or partition; sbatch reads both from the environment. A podman
storage config on `/dev/shm` is created on first use if the account has none.

```bash
export SBATCH_ACCOUNT=<project> SBATCH_PARTITION=normal
cd <checkout under $SCRATCH>/containers/cluster/ce-images

IMAGE_DIR=$PWD/vllm-cuda        sbatch vllm-cuda/build.sbatch          # < 1 h
IMAGE_DIR=$PWD/judge-agent-cuda sbatch judge-agent-cuda/build.sbatch   # up to 24 h cold, then a GPU probe

DRY_RUN=1 CE_PLATFORM=gh200 ./promote_image.sh --all
CE_PLATFORM=gh200 ./promote_image.sh --all                             # renders the gh200 EDFs
```

`judge-agent-cuda` carries the CUDA counterparts of `judge-agent-amd`: nvcc, cuBLAS/cuFFT/cuSOLVER/
cuSPARSE/cuRAND/cuTENSOR/NCCL, NVHPC (OpenACC), LLVM OpenMP offload to `sm_90`, CUDA-aware MPICH
(netmod=ofi, host Slurm PMI), PETSc/MAGMA/SuperLU_DIST/STRUMPACK/SUNDIALS `+cuda`, PAPI cuda/nvml,
Nsight, cupy, jax, Pluto, ppcg, dace and every agent harness. Both GPU images are CUDA 12.9 so the
CE's `aws_ofi_nccl` plugin (variant `cuda12`) matches; the serving smoke checks it loads.

Weights and serving (`inference/serve-daint.sbatch`; same served name, window and parsers as beverin):

```bash
cd inference
EDF=hpcagent-bench-vllm-gh200-latest PYTHON=python3 HF_TOKEN=<token> \
  MODELS="Qwen/Qwen3.8-27B-FP8 openai/gpt-oss-120b moonshotai/Kimi-K2.7-Code" \
  sbatch --time=08:00:00 fetch_weights.sbatch
cd ../../../..
umask 077; mkdir -p ~/.config/hpcagent-bench; openssl rand -hex 32 > ~/.config/hpcagent-bench/daint-endpoint.key
MODEL=qwen38  MODE=smoke sbatch -N 1 --time=01:00:00 containers/cluster/ce-images/inference/serve-daint.sbatch
MODEL=oss120b MODE=smoke sbatch -N 1 --time=01:00:00 containers/cluster/ce-images/inference/serve-daint.sbatch
MODEL=kimi    MODE=smoke sbatch -N 4 --time=02:00:00 containers/cluster/ce-images/inference/serve-daint.sbatch
MODEL=kimi             sbatch -N 4 --time=12:00:00 containers/cluster/ce-images/inference/serve-daint.sbatch
MODEL=kimi DRY_RUN=1 bash containers/cluster/ce-images/inference/serve-daint.sbatch   # print the command only
```

| `MODEL` | nodes | TP x PP | window | tool / reasoning parser |
|---|---|---|---|---|
| `qwen38` | 1 | 4 x 1 | 262144 | `qwen3_coder` / `qwen3` |
| `oss120b` | 1 | 4 x 1 | 131072 | `openai` / `openai_gptoss` |
| `kimi` | 4 (`SERVE_NODES=2` allowed) | 4 x 4 | 262144 | `kimi_k2` / `kimi_k2` |

From another Daint job, `source containers/cluster/ce-images/inference/alps-endpoint.sh <run dir>/endpoint.json`
checks the endpoint and exports `VLLM_BASE_URL`, `VLLM_API_KEY` and `VLLM_MODEL`. For a campaign the
endpoint is a service arm (`experiments/inference_service.py`) with
`AMD_CE_ENV=hpcagent-bench-agent-gh200-latest` and `JUDGE_CE_ENV=hpcagent-bench-judge-gh200-latest`.
`experiments/run_cluster.sh`, `experiments/beverin.sbatch` and the `experiments/layers/*.env` model
layers are still beverin-shaped.

### CPU only (any node, x86_64 or aarch64)

The light judge+agent pair: gcc 16, clang/flang 22 with Polly, OpenBLAS/BLIS/FFTW/ScaLAPACK/HDF5/
SuiteSparse/SuperLU/METIS/Scotch/ARPACK/MUMPS-seq, the distro MPICH, tblis, HPTT, Pluto, numba,
dace and the harnesses; no GPU stack, about an hour to build.

```bash
export SBATCH_ACCOUNT=<project> SBATCH_PARTITION=<partition of the target architecture>
IMAGE_DIR=$PWD/judge-agent-cpu sbatch judge-agent-cpu/build.sbatch
CE_PLATFORM=cpu ./promote_image.sh --all
srun -N1 --environment=hpcagent-bench-judge-cpu-$(uname -m)-latest python3 -c 'import hpcagent_bench, dace'
```

Without the Container Engine, the generic recipes build the same roles with docker/podman or Apptainer:

```bash
docker build -f containers/hpcagent_bench.Dockerfile --build-arg HW=cpu -t hpcagent_bench:cpu .
apptainer build hpcagent_bench-cpu.sif   containers/cpu.def      # agent
apptainer build hpcagent_bench-judge.sif containers/judge.def    # judge (harness baked in)
```

### Serving jobs and gates (`cluster/ce-images/inference/`)

The MI300A serving recipes (these jobs, the `sglang/` and `vllm/` Dockerfiles and EDF templates,
`moe-configs/` and the patches) carry the reasoning for each tuned value in their comments; keep it
when editing. vLLM stays at 0.23.0 for oss120b: 0.27.1 (branch `parked/vllm-0271`) served 2405
tok/s against 3013 on one pinned node with the same probe, dtype, quantization, MoE and attention
backends -- 25% slower, all of it in decode (steady state 2540 vs 3187; prefill within 0.3%).

| file | does |
|---|---|
| `fetch_weights.sbatch` | downloads into `$HF_HOME` inside an image, restripes on the host, fails unless every large blob is wide-striped |
| `serve-private.sbatch` | a private Qwen3.8 endpoint on one beverin node (`docs/serving/private-endpoint.md`) |
| `serve-daint.sbatch`, `alps-endpoint.sh` | GH200 serving and the client-side endpoint check |
| `smoke-kimi-sglang.sbatch`, `submit-glm53-sglang.sh`, `smoke-kimi-eager-pg.sbatch`, `smoke-kimi-replicas.sbatch` | multi-node serving smokes (SGLang; GLM-5.3 on SGLang; vLLM with the eager-PG patch; N vLLM replicas in one allocation) |
| `verify-tools-reasoning.py`, `accuracy-gate.py`, `agentlike-probe.py` | tool-call/reasoning, long-context accuracy and throughput gates against a live server |
| `prebuild-aiter-jit.sbatch` | warms the aiter JIT cache (each op must be called, not imported) |
| `tune-moe-int4-mi300a.sbatch`, `merge_moe_configs.py`, `moe-configs/` | MoE tuning; `moe-configs/` is build input for `sglang/` and `vllm/` |
| `external-eager-pg-patch/` | `sitecustomize.py` for the vLLM pipeline bootstrap on RCCL |
| `ue8m0-patch/` | `sitecustomize.py` giving `torch.Tensor` a `format_ue8m0` default, for an SGLang image without the build-time guard `sglang/Dockerfile` applies |
| `aiter_mla_check.*`, `sglang_kernel_launch_check.py` | kernel-level correctness checks without a server |

## Publishing

`push_images.sbatch` publishes `<sqsh>.oci.tar` under each role's tag in `REGISTRY_REPO`
(`docker.io/spcleth/hpcagent-bench`); `push_image.sh` adds an immutable `sha-<digest>` tag. The
default `DRY_RUN=1` runs every registry gate (10 GB per layer, 100 GB per image) and sends nothing.
Credentials come only from the environment.

```bash
DRY_RUN=1 sbatch push_images.sbatch
REGISTRY_USER=<user> REGISTRY_TOKEN=<token> DRY_RUN=0 ROLES="sglang vllm" sbatch push_images.sbatch
judge-agent-amd/build-judge-release.sh <git-ref>     # release judge: agent archive + package at <ref>
sqsh_to_oci.sh $SCRATCH/ce-images/<image>.sqsh      # an archive for a squashfs built without one
```

## Adding a container

1. Create `cluster/ce-images/<name>/` with `Dockerfile`, `build.sh`, `build.sbatch` and
   `edf.toml.example`. Copy the closest existing directory: `sglang/` or `vllm-cuda/` for a
   single-target image, `judge-agent-cuda/` for an agent+judge pair. `build.sh` sources
   `../build_common.sh` and `../images.env`, builds from the repository root and ends with
   `ce_export_image <tag> <candidate path>`; `build.sbatch` refuses to overwrite a squashfs an EDF
   mounts. The EDF template keeps the `"<hpcagent_bench_edf_mounts>"` item and an absolute `PATH`
   in `[env]` (the Container Engine drops the image's own `ENV`).
2. Add one row to `images.env` (one per build target) with the next role name, a new prefix, the
   platform, the directory, the partition (`-` outside beverin), the `verify_image.py` profile,
   `<live>-candidate.sqsh`, the live squashfs, the EDF name, the template and the tag (`-` until
   published).
3. A new kind of image also needs a `verify_image.py` profile (the checks it must pass).

`install_edfs.sh`, `promote_image.sh`, `pull_image.sh`, `pull_images.sbatch`, `push_images.sbatch`
and `build_and_verify.sbatch` pick the row up with no further edit. A new serving engine is also
launched by `experiments/run_cluster.sh` (`docs/extending/inference.md`).

Every package added to a Dockerfile gets a probe that uses it (compile, import, link) in the same
`RUN`, so a broken install fails the build rather than a campaign.

## Adding a skill

1. Create `hpcagent_bench/skills/<name>/SKILL.md` (frontmatter `name`, `description`, optional
   `when`; optional helper `*.py` beside it). Discovery (`load_skills`), packaging (`pyproject.toml`
   ships `skills/*/SKILL.md` and `skills/*/*.py`), the judge image and agent staging pick it up.
2. To hand it to an arm, name it in a packet: one entry under `packets:` in
   `hpcagent_bench/envs/registry.yaml` (`docs/extending/packets.md`).

Nothing under `containers/` changes, and no image is rebuilt. Details: `docs/extending/skills-and-tools.md`.

## Agent harness pins

Both judge-agent images install the harnesses from `agent/harness/`: `node/package.json` +
`package-lock.json` (claude-code 2.1.197, codex, qwen-code, opencode), `requirements-{miniswe,openhands,sweagent}.txt`
(one venv each under `/opt/harness/`), and `pins.env` (uv, node and their per-architecture sha256).
To bump one, edit the pin, run `agent/harness/freeze.sh`, run `tests/test_harness_pins.py`, rebuild.
Each image's final gate checks every version against these files.
