# CE images -- build, promote, extend

Four Dockerfiles, one directory each, each building one image end to end. `IMAGE_REQUIREMENTS.md`
is the *specification* (what an image must carry and why); this file is how to **build, verify,
promote and extend** one.

| directory | promoted name | role |
|---|---|---|
| `judge-agent-amd/` | `optarena-ce-amd-mi300.sqsh` (agent), `optarena-ce-judge-amd-mi300.sqsh` (judge) | **TWO targets, one Dockerfile.** `agent` is the whole toolchain and carries NO hpcagent_bench, so an agent cannot reach the references it is graded against. `judge` is `FROM agent` plus the library -- one extra layer. |
| `sglang/` | `optarena-sglang.sqsh` | SGLang inference -- the dominant serving engine (qwen38, kimi, GLM-5.3) |
| `vllm/` | `optarena-vllm.sqsh` | vLLM 0.23.0 inference, kept for oss120b's mxfp4 path |
| `judge-agent-cuda/` | not built here | CUDA counterpart of `judge-agent-amd`; parse-checked only, no NVIDIA partition on this cluster |

vLLM 0.27.1 was retired 2026-09-08 (25% slower than 0.23.0 on oss120b, entirely in decode) and
lives on the `parked/vllm-0271` branch. Do not re-derive that; restore the branch.

## The naming contract

**One image per role, no version in any name.** `images.env` is the only place a name maps to a
file, and `install_edfs.sh` renders the EDFs from it. Consumers name `-latest` and nothing else.

```
optarena-amd-mi300-latest -> optarena-ce-amd-mi300.sqsh
sglang-latest             -> optarena-sglang.sqsh
vllm-latest               -> optarena-vllm.sqsh
```

There is deliberately **no version to pin to**. A run that spells a version silently opts out of
every fix promoted after that file was written -- which is exactly how three offload arms ran on an
image built before the libomp fix and scored zero. What identifies a build is the `.digest`
sidecar, not a tag.

## Getting the images: PULL is the default

```bash
# every role, on a compute node (enroot unpacks 60+ GB before it writes the squashfs, and
# extraction onto Lustre fails outright -- a rootless overlay cannot create its pivot dir there)
sbatch containers/cluster/ce-images/pull_images.sbatch

# one role, pinned to a digest -- what a results table should cite
./pull_image.sh judge-agent-amd sha-<digest>

# then point the EDFs at what you fetched
./install_edfs.sh
```

A rebuild is one node for hours; the judge+agent image bootstraps gcc 16 and then llvm 22 before it
reaches PETSc and MAGMA. A pull is bandwidth. The reason that matters beyond time: a pull gets the
**same bytes we published**, so the digest in a results table is the digest that ran. A rebuild
from the same Dockerfile is a *different* image that merely resembles it -- apt and PyPI move
underneath it, and the digest will not match.

**Build instead when you are changing an image, or when a role has not been published yet.** Both
are real cases: `pull_images.sbatch` reports a per-role failure rather than aborting, and names the
build command for whatever it could not fetch.

Roles: `judge-agent-amd` (the agent image), `judge`, `sglang`, `vllm`.

## Build -> verify -> promote

```bash
# 1. build to a CANDIDATE name; never write over a name an EDF mounts
# IMAGE_DIR is the directory holding that image's build.sh -- NOT the output directory. It
# defaults to the SUBMIT dir, so submitting from ce-images/ without it fails in 2 s on
# `test -x <ce-images>/build.sh`. Output location is OUTPUT_SQSH, which already defaults to
# $SCRATCH/ce-images/optarena-<role>-candidate.sqsh.
# judge-agent-amd builds BOTH targets in ONE job (BUILD_TARGETS="agent judge", the default).
# That is not a convenience: build_common.sh wipes the /dev/shm graphroot on entry because the
# nodes are diskless, so there is no layer cache BETWEEN jobs. Two separate jobs would be two
# full 2 h builds; one job is the agent build plus a pip layer. Order matters -- agent first --
# and each target is exported before the next is built, so a judge failure still leaves a usable
# agent image.
sbatch --export=ALL,IMAGE_DIR=$PWD/judge-agent-amd judge-agent-amd/build.sbatch   # ~2h, both targets

# Or build and verify in ONE job, which is the entry point to PREFER for every role: an image
# that cannot pass verification does not report success, and the artifact keeps its CANDIDATE
# name until a human promotes it. It also writes the .verified marker promote_image.sh requires.
sbatch --export=ALL,IMAGE_DIR=$PWD/judge-agent-amd build_and_verify.sbatch   # ~2h, BOTH targets
sbatch --export=ALL,IMAGE_DIR=$PWD/sglang          build_and_verify.sbatch   # ~1h
sbatch --export=ALL,IMAGE_DIR=$PWD/vllm            build_and_verify.sbatch   # ~4h

# 2. verify a candidate on its own (build_and_verify already did this; this is the re-run path).
# IMAGE and PROFILE are ENV, not positional arguments.
IMAGE=$SCRATCH/ce-images/optarena-sglang-candidate.sqsh PROFILE=sglang \
  sbatch verify_image.sbatch

# 3. promote. One command per role, or --all.
DRY_RUN=1 ./promote_image.sh --all      # say what would move, touch nothing
./promote_image.sh --all                # rename + sidecars + ALLOW_REPOINT=1 install_edfs.sh
```

`promote_image.sh` **refuses a candidate that carries no `.verified` marker**, which only
`build_and_verify.sbatch` writes and only on a clean verdict. "Built" and "works" have been
different things often enough here to cost whole campaigns.

The rename is **safe while arms are running**: a mounted squashfs is held by its inode, so a job
that already started keeps reading the bytes it opened and only new jobs see the new image.
Overwriting a file in place is NOT safe -- that is why builds go to a candidate name first.

The script moves `.digest`, `.sha256` **and `.oci.tar`** with the image, which is the half that
used to get forgotten when this was four hand-typed `mv` lines. The first two are the only record
of *which* build a name currently holds -- with one version per role there is nothing else to tell
two builds apart. The `.oci.tar` matters for a different reason: it is what
`push_image.sh --from-archive` publishes. Leave it behind and the archive under the live name is
still the SUPERSEDED build, so the next push sends the old bytes under the promoted tag -- the
registry and the cluster then disagree while every checksum looks fine.

**There is no layer cache between build jobs.** `build_common.sh` wipes the `/dev/shm` podman
graphroot on entry because the nodes are diskless, so every build pays full cost. Budget 1-2h and
do not casually rebuild "to check something".

## Extending an image

### Adding a package or library
Edit the one Dockerfile. Then **add an assertion that it is actually usable**, not merely
installed -- every trap below is a case where something installed cleanly and failed at run time.
The existing Dockerfiles end with probe blocks; follow that shape:

```dockerfile
RUN set -eux; \
    <install>; \
    <compile or import something that USES it>; \
    <assert the result>            # fail the BUILD, not the campaign
```

### Bumping an engine version
`vllm/Dockerfile` and `sglang/Dockerfile` take `ARG`s (`VLLM_VERSION`, `AITER_REF`, ...).
`build.sh` passes `EXTRA_BUILD_ARGS` through, so a candidate build is:

```bash
# bare KEY=VALUE pairs -- build.sh adds the --build-arg itself
EXTRA_BUILD_ARGS="VLLM_VERSION=0.28.0 AITER_REF=v0.1.21.post1 VLLM_ROCM_AITER_SWITCH=1" \
  sbatch vllm/build.sbatch
```

Do not hardcode a version into an assert. `vllm/Dockerfile` asserts the aiter master switch
matches its **declared** `ARG`, so the check still means something when the value changes.

### Adding a model's kernels to the aiter prebuild
`inference/prebuild-aiter-jit.sbatch` warms `/opt/aiter-jit` so a server does not JIT-build on
first request. **`@compile_ops` is lazy: importing `aiter.ops.X` builds NOTHING.** Only an actual
call reaches `get_module()`. A prebuild that imports ten modules and calls one warms one -- which
cost GLM-5.3 ~15 minutes of JIT at every server start.

Add one representative **call** per module you want warmed. Two rules:
- the call may raise on shape/dtype -- the wrapper reaches `get_module()` before args hit C++, so
  the module is still built. Keep a per-call `try/except` and log "raised, module still built".
- watch for Python-side fast paths that never reach the kernel: `moe_sum` with `topk != 4` takes a
  `torch.sum` branch and prebuilds nothing while logging success.

Gate on **"some op module beyond `module_aiter_core` exists"**, never on a module *name* --
names differ across aiter versions (`module_rmsnorm` vs `module_rmsnorm_quant`).

### Adding a served model
Usually no image change at all -- check first:
```bash
srun --environment=sglang-latest python3 -c \
  "from sglang.srt.models import <mod>; print('present')"
```
GLM-5.3 needed **no rebuild**: `GlmMoeDsaForCausalLM` was already in the shipped image. The serving
config lives in `experiments/.env.*`, not in the image.

## Traps this directory has already paid for

**libomp: one symlink, never the directory.** LLVM 17+ puts `libomp.so` in a per-target libdir on
no default loader path, so a clang OpenMP build links clean and dies at `dlopen`. That same libdir
also holds LLVM's `libgomp.so.1` **shim**, so exposing the directory (`ld.so.conf.d` or
`LD_LIBRARY_PATH`) silently replaces GNU libgomp under every gcc arm. Symlink the one file and
probe both halves: clang's libomp dlopens, and gcc still resolves `libgomp.so.1` outside the LLVM
tree. Binaries with `DT_RUNPATH` are unaffected either way -- RUNPATH is searched before the cache.

**`PYTHONSAFEPATH=1` drops `sys.path[0]`.** It broke rocprof-compute's beside-itself import for
every judge job. If a tool relies on that, wrap it with an explicit `PYTHONPATH`.

**Probe in the environment the build uses.** The mimalloc link check ran in a different environment
than the compile it was predicting, and dropped `-lmimalloc` from 20% of offload builds.

**A "compiles" check is not a "loads" check is not a "runs on the device" check.** The offload arms
had passing flag tests, a linkable `.so`, and still scored 0/324 -- the reference could not load.
Where a device is involved, assert with `omp_is_initial_device()`, not with a successful compile.

**Verify the image, then verify a real job.** `test_compile_flags.py` inside the container catches
what the harness rpath cannot reach; a one-node gate job catches what neither does.

## Verifying

```bash
# IMAGE and PROFILE are ENV, not positional -- a path passed positionally is silently ignored
# and the default profile is verified instead.
IMAGE=$SCRATCH/ce-images/<image>.sqsh PROFILE=<judge-agent-amd|judge|sglang|vllm> \
  sbatch verify_image.sbatch
srun --environment=<edf> python -m pytest tests/test_compile_flags.py
```

`verify_image.py` also checks the solver gates (`HAVE_ISL`, `has_z3()`), which **fail closed and
silent** in dace: no islpy means `WavefrontSkew` is a no-op, no z3 means `LoopToMap` cannot prove.

`mpi_gpu_check.sh` runs as part of `build_and_verify.sbatch` for the judge-agent profiles, and
RUNS what the declarative table can only look for. `mpicc` on `PATH` and `libmpi.so` on disk were
both true of the distro MPICH whose wrapper and launcher came from different MPIs: four ranks each
came up as their own `COMM_WORLD` of size 1, every rank solved the whole problem, and the answer
verified. It proves instead:

* a size-4 communicator with a correct allreduce;
* a **device pointer** surviving a real allreduce, behind
  `MPIX_GPU_query_support(MPIX_GPU_SUPPORT_HIP)` -- the MPICH spelling, in `mpi.h`; the OpenMPI
  one reports a false NO on this stack;
* **which libfabric the live process mapped and which provider it chose**, read from
  `/proc/self/maps` rather than predicted with `ldd`;
* the RCCL net plugin being *selected* -- `NET/OFI`, not `NET/Socket`;
* `PETSC_HAVE_HIP` rather than just `libpetsc.so`.

The transport legs are the ones to read first, because **a correct allreduce proves correctness,
never transport** -- the same sum comes back over the `tcp` provider, several times slower, with
every other assertion still green. That mistake was made here once and reported as "MPI is already
reaching Slingshot".

All of libfabric, libcxi and `librccl-net.so` come from the pinned CSCS **netstack artifact**, so
the EDF must carry all five hook annotations. The images ship none of the three and a build gate
refuses any that survives; with a partial hook set `libmpi.so` does not resolve at all, which is a
loud failure rather than a silent fallback. Run it standalone against any image:

```bash
sbatch containers/cluster/ce-images/mpi_check.sbatch      # single node, generates its own EDF
sbatch containers/cluster/ce-images/mpi_multinode_check.sbatch   # 2 nodes: the cross-node claim
```

Single-node is the limit of what `mpi_gpu_check.sh` can prove about transport: it shows which
provider MPI *initialised*, not that a cross-node transfer rode it. It SKIPs the GPU-runtime checks
with no visible device and says so, so it never reports a pass for something it could not test --
except inside a batch job on `mi300`, where a missing GPU is a broken EDF and fails.

## Registry

One repository for every image: `docker.io/spcleth/hpcagent-bench`. The tag is the `.sqsh` basename
minus the `optarena-` prefix, so a tag names the ROLE and cannot drift from the file it was built
from. **Credentials come from the environment (`REGISTRY_USER`, `REGISTRY_TOKEN`) and are never
written into the repo.** Do not push without explicit instruction.

## What is on scratch

`/capstor/scratch/cscs/$USER/x86_64/ce-images/` -- roughly 45-60 GB per image. Delete an old image
only after confirming no running job mounts it:

```bash
for j in $(squeue -u $USER -h -o %i); do grep -h '^image' \
  $SCRATCH/hpcagent-bench-runs/*/$j/edf/*.toml 2>/dev/null; done | sort -u
```

The run directory's `edf/*.toml` records what a job **actually mounted**, which is the only
trustworthy answer -- the `.env` file on disk may have been rewritten since that job launched.
