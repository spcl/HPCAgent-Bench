# CE images -- build, promote, extend

Four Dockerfiles, one directory each, each building one image end to end. `IMAGE_REQUIREMENTS.md`
is the *specification* (what an image must carry and why); this file is how to **build, verify,
promote and extend** one.

| directory | promoted name | role |
|---|---|---|
| `judge-agent-amd/` | `optarena-ce-amd-mi300.sqsh` | judge + agent: compilers, HPC libraries, solvers, profilers, frameworks. One image serves BOTH the CPU and GPU arms. |
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

## Build -> verify -> promote

```bash
# 1. build to a CANDIDATE name; never write over a name an EDF mounts
IMAGE_DIR=/capstor/scratch/cscs/$USER/x86_64/ce-images \
  sbatch judge-agent-amd/build.sbatch          # ~2h; sglang/vllm ~1h

# 2. verify the candidate before it is anyone's problem
sbatch verify_image.sbatch <candidate>.sqsh    # required-tool + loader checks

# 3. promote: rename over the live name, keeping the sidecars
mv optarena-<role>-candidate.sqsh optarena-<role>.sqsh
mv optarena-<role>-candidate.digest optarena-<role>.digest    # and .sha256

# 4. repoint the EDFs
ALLOW_REPOINT=1 ./install_edfs.sh
```

The rename is **safe while arms are running**: a mounted squashfs is held by its inode, so a job
that already started keeps reading the bytes it opened and only new jobs see the new image.
Overwriting a file in place is NOT safe -- that is why builds go to a candidate name first.

Keep `.digest` and `.sha256` with the rename. With one version per role they are the only record
of *which* build a name currently holds.

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
config lives in `containers/cluster/example-script/.env.*`, not in the image.

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
sbatch verify_image.sbatch <image>.sqsh     # required tools resolve, nothing resolves outside
srun --environment=<edf> python -m pytest tests/test_compile_flags.py
```

`verify_image.py` also checks the solver gates (`HAVE_ISL`, `has_z3()`), which **fail closed and
silent** in dace: no islpy means `WavefrontSkew` is a no-op, no z3 means `LoopToMap` cannot prove.

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
