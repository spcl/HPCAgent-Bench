# What every consolidated image must carry

Four images, one Dockerfile each, everything baked IN -- no out-of-image `PYTHONPATH`, because
anything reached that way is invisible to the image digest and two runs of "the same image" can
then differ by it.

| image | base | notes |
|---|---|---|

## FOUR images. ROCm 7.2.0 everywhere. Nothing else in this directory.

| image | base | serves |
|---|---|---|
| judge + agent, AMD | ROCm 7.2.0, x86_64 | both roles -- `run_cluster.sh:814,817` already pass one `AMD_CE_ENV` to both `role_srun` calls |
| judge + agent, CUDA | CUDA, aarch64 / GH200 | same roles, other vendor |
| vLLM | ROCm 7.2.0 | oss120b |
| SGLang | ROCm 7.2.0 | qwen38, kimi -- 12 of 18 v9 arms |

**ROCm 7.2.0 is the global pin.** SGLang consumes a vendor prebuilt already tagged `rocm720`, so it
stays a straight pull; the vLLM images are ours and change one base line in a Dockerfile we are
writing anyway. The reverse (7.2.3 global) would mean building SGLang from source -- the expensive
direction, for no recorded benefit. NOTE before locking this in: the vLLM EDFs were deliberately
named `rocm723-*` and no rationale for 7.2.3 over 7.2.0 was ever recorded. Check the build chain's
history once; if 7.2.3 fixed something, that decision has to be revisited.

**The AMD/CUDA split stays.** "One image for judge and agent" means one per PLATFORM serving both
ROLES. It does not mean one across vendors: different base, architecture (x86_64 vs aarch64),
`hipcc` vs `nvcc`, cupy HIP source build vs wheel, HIP vs CUDA backends throughout. Merging them is
how a HIP build ends up seeing NVIDIA cub.

**Python 3.12 is the pin -- whatever the base already ships.** The AMD base
(`rocm/pytorch:rocm7.2_...py3.12_...`) ships 3.12, and installing 3.14 on top means rebuilding the
framework stack against a Python the vendor images do not target, for no measured benefit. So take
the base's interpreter and do NOT add another.

Two consequences to handle rather than discover:

* **The graded venv is 3.14.7 and `hpcagent-canon-ci.yml` runs 3.14**, so the container will grade
  on a different Python minor from CI and from local sweeps. dace declares
  `requires-python = ">=3.10, <3.15"`, so both are supported and this is a deliberate split, not a
  break -- but it must be WRITTEN DOWN, because a result that reproduces locally and not in the
  container will otherwise cost someone a day. (`ml-ci.yml` runs 3.13, so there are three.)
* **Both judge/agent bases must agree on the minor.** The AMD base is py3.12; confirm the CUDA base
  (`ngc-pytorch:26.02-py3-alps6`) is too. Two judge images grading on different Pythons is the one
  version split with no upside at all.

The final gate should assert the interpreter is the base's, pin numpy / scipy / pandas / astunparse
to the versions the judge grades with, and record the Python version in the provenance file so a
result can be attributed to it.

### The directory contains ONLY this

    <image>/Dockerfile        x4
    <image>/build.sh          x4
    <image>/build.sbatch      x4

No other folder, no stray `.py`, no probe scripts, no logs, no EDF tomls. **Anything currently kept
here because something references it must move INTO the image instead** -- that is what "everything
inside the image" means, and it is what makes these files deletable:

* `moe-configs/` + `merge_moe_configs.py` -- referenced by `run_cluster.sh:181`. Bake the tuned MoE
  configs into the inference images; losing them once voided a whole set of throughput numbers, so
  bake, do not drop.
* `external-eager-pg-patch/sitecustomize.py` -- referenced by `run_cluster.sh:168`. Belongs in the
  image's site-packages.
* `build/` and `build-chain.sh` -- the 6-job vLLM build chain is replaced by the single Dockerfile.
* `prebuild-aiter-jit.sbatch` -- aiter's JIT-on-first-request baton lock is an IMAGE problem; prebuild
  during the build, not as a separate job.
* `beverin-rocm723-host-ofi-phase1/`, `accuracy-gate.py`, the `smoke-kimi-*` and `gate-0271-*`
  scripts -- either fold into `build.sbatch` as a post-build gate, or delete.

`run_cluster.sh` must be updated in the same change, or deleting these breaks the launcher.

This document itself belongs in `docs/`, not here.

## Load-bearing, do not drop when rewriting a Dockerfile

**rocprof-compute needs its OWN interpreter.** ROCm installs the tool but not its Python deps.
Installing `/opt/rocm/libexec/rocprofiler-compute/requirements.txt` into the image environment is
WORSE than the breakage: it pins `astunparse==1.6.2`, and **dace declares astunparse as a
dependency** (`dace/pyproject.toml:71`), so it downgrades a package every CPU reference number is
computed with -- and it still produces nothing, because rocprof-compute 3.4.0's v3->v2 CSV
converter dies on pandas 3 (`merge on str and int64 for key 'Agent_Id'`): all 13 counter passes
run and all 13 rows are dropped, ending at "No profiling data found".

The fix that works: `python3 -m venv /opt/rocprof-compute-venv`, install `pandas==2.2.3`,
`numpy<2.3` and that requirements.txt into it, move `/opt/rocm/bin/rocprof-compute` aside and
replace it with a two-line `sh` wrapper exec'ing the venv's python on the libexec entry point.
A wrapper rather than a PATH entry, so it survives whatever PATH order a caller has. Guard the
build: fail if the venv cannot import pandas/tabulate/plotext/dash with pandas major == 2, if
`rocprof-compute --version` is non-zero, or if the image's own numpy/pandas/astunparse move.

**PAPI: rebuild it IN the Dockerfile, and do not expect AMD device counters from it.**
`rocm` and `rocm_smi` are not shipped components. Build them in a Dockerfile layer -- the exact
configure lines live in `harness/papi.py:1480 COMPONENT_BUILD`:

    ./configure --with-components=rocm       # PAPI_ROCM_ROOT -> the ROCm install
    ./configure --with-components=rocm_smi   # PAPI_ROCMSMI_ROOT -> the ROCm install

Two things to get right when verifying the result:

* **PAPI 7 initializes a component LAZILY** (`papi.py:151`). An untouched component reports itself
  disabled with "Not initialized. Access component events to initialize it.", so a build check that
  reads the status flag and stops will call a WORKING component broken. Enumerate its events first
  -- `component_reason()` already does exactly that, and separates "not built" from "built but
  would not come up". Use it rather than a flag read.
* Even rebuilt, `rocm_smi` was measured failing at "Error while initializing device tables" with
  `PAPI_ROCMSMI_ROOT` set -- the second category, a driver/device/permission problem the rebuild
  cannot fix.

So: AMD device counters come from `rocprofv3 --pmc`. Build the components anyway so
`component_reason()` reports the honest reason instead of "not built", but do not treat their
presence as a working path, and keep PAPI's CPU `perf_event` story intact -- that one works and is
what the profiling skill relies on.

**`rocprof-sys-sample`, never `rocprof-sys-run`.** `-run` executes the program, exits 0 and writes
nothing -- a silent no-op that reads as success.

**Also bake in:** flydsl (currently reached via an out-of-image `PYTHONPATH`), aiter (0.27.1 needs
it or it dies in `profile_run`; leave its master switch OFF -- it breaks MLA prefill on gfx942),
and aws-ofi-nccl / RCCL-OFI.

## Verifying an image

`scripts/smoke_gpu_profilers.sh` (+ `submit_gpu_profiler_smoke.sbatch`) runs on one node in ~3
minutes and reads ARTIFACTS rather than exit codes -- it reconciles `SQ_WAVES` against the launch
geometry (20 x 2^22 / 64 = 1,310,720) and fails rocprof-compute specifically when the passes run
and the rows are dropped. Point it at any image before it goes live.


## Toolchain both judge+agent images must carry

Same set on the AMD and the CUDA image; only the offload target differs.

| what | notes |
|---|---|
| `perf` | the CPU profiling path the skills teach; PAPI's `perf_event` component depends on it |
| tblis, OpenBLAS, LAPACK | see the OpenBLAS trap below |
| MPI | **mpich, GPU-aware for the platform** |
| GCC + Graphite | loop transforms; **OpenACC offload lives here**, not on LLVM |
| LLVM + MLIR + Polly | **OpenMP offload lives here**, not on GCC |
| vendor compiler | `amdclang` on AMD; **NVHPC** on CUDA -- and NVHPC is the ONLY OpenACC path |
| vendor profilers | AMD: rocprofv3 / rocprof-sys / rocprof-compute. CUDA: **ncu** + **Nsight Systems** |

### The offload matrix, corrected

|            | AMD (gfx942)                 | NVIDIA (GH200)          |
|------------|------------------------------|-------------------------|
| **OpenMP offload** | **LLVM** -- `amdgcn` | **LLVM** -- `nvptx`     |
| **OpenACC**        | **not supported**    | **NVHPC only**          |

OpenMP offload on LLVM is the portable path and is supported on both vendors -- hard-gate it on
both images. OpenACC is NVIDIA-only and comes from **NVHPC**, not from GCC: spack's `gcc` exposes
`+nvptx` and has no amdgcn variant at all, and a hand-rolled `--enable-offload-targets=amdgcn-amdhsa`
build (amdgcn newlib plus LLVM's assembler and linker) is not expressible as a spack spec. So an
AMD image simply has no OpenACC, and that is a property of the platform rather than a gap to close.

**Consequence worth acting on: GCC is not an offload compiler here.** It is wanted for **Graphite**
and as a host C/C++/Fortran compiler, nothing more. Do not spend build time or gate complexity on
GCC's nvptx offload on either image -- the `sm_90`/`sm_89` alias check matters only if something
actually routes offload through GCC, and nothing should.

### Traps this project has already paid for

* **Build GCC and LLVM through spack, and name `CC` / `CXX` / `FC` explicitly.** A stale configure
  cache beats `PATH` -- a toolchain that looks selected can still be ignored.
* **GCC's nvptx offload has NO `sm_90`.** It silently means `sm_89` against CUDA 13; the CSCS
  overlay patch is what fixes it. Do not assume a GH200 target is honoured because the build
  succeeded.
* **OpenBLAS must be the spack `threads=openmp` build, selected by GLOB.** The scipy wheel renames
  every symbol (`scipy_cblas_dgemm`), so linking against it NEVER resolves.
* **flang's `dc-to-openmp` needs LLVM >= 20.**
* **Do not let a HIP build see vendored NVIDIA cub** -- it dies. The backend is chosen only in
  `gpucub.cuh`.
* Known-good pairing already in service: **gcc 16 + llvm 22** (both stable releases); the
  unsuffixed image carries clang/flang 23.

## HPC libraries: ship them all

The list is not a wishlist -- it is what DaCe codegen can emit a call to, so a missing one turns a
valid lowering into a link error at grading time. Source: `dace/libraries/*/environments/`.

**Both images:** OpenBLAS, LAPACK, ScaLAPACK (the `pblas` nodes, plus its `thread_level` env),
tblis, HPTT (`tiled_transpose` / `tile_backends`), FFTW3, MPI (mpich GPU-aware; OpenMPI too, since
`ref_openmpi` and `intel_mkl_openmpi` environments exist), TBB, mimalloc, libmvec, ska_sort,
Eigen, HDF5, **PyTorch** (the `torch` and `onnx` library nodes need `pytorch_env`).

**AMD image (x86_64):** rocBLAS + hipBLAS, rocSOLVER, rocFFT + hipFFT, hipSPARSE, hipTENSOR,
hipCUB + rocPRIM (hipCUB does not build without rocPRIM), rocThrust, rocRAND.
Intel MKL is possible here (`intel_mkl`, `intel_mkl_mpich`, `intel_mkl_openmpi` environments) --
ship it, but OpenBLAS stays the selected BLAS.

**CUDA image (aarch64 / GH200):** cuBLAS, cuFFT, cuSOLVER (`cusolverdn`), cuSPARSE, cuTENSOR,
NVIDIA CUB, Thrust, cuRAND.
**Intel MKL is impossible on this image** -- it is x86_64-only and GH200 is aarch64. Any recipe
copied from the AMD image must drop it rather than fail the build.

### Solvers

Both images. The `scientific_computing` track is dense linear algebra, dynamic programming and
structured grids, so a solver an agent reaches for and does not find is a link error at grading
time exactly like a missing BLAS.

* **Dense / GPU-accelerated:** MAGMA -- it has both a CUDA and a HIP backend, so it belongs on both
  images, built against the matching vendor stack.
* **Sparse direct:** SuiteSparse (UMFPACK, CHOLMOD, SPQR), SuperLU and SuperLU_DIST, MUMPS,
  STRUMPACK.
* **Iterative / frameworks:** PETSc and Hypre. SLEPc and ARPACK-NG for eigenproblems.
* **Partitioners:** METIS, ParMETIS, Scotch -- not optional extras. PETSc, MUMPS and SuperLU_DIST
  all want them, and omitting them silently drops solver features rather than failing the build.

Vendor solvers are already listed above and are NOT a substitute: `rocSOLVER` / `cuSOLVER` cover
dense factorizations on device, nothing sparse-direct and nothing iterative.

Build-cost warning, since this is one Dockerfile end to end: **PETSc is the expensive node in this
graph.** It pulls hypre, METIS, ParMETIS and Scotch, and building it with GPU support against the
vendor stack dominates image build time. Build it in its own layer so a change elsewhere does not
invalidate it, and pin its version -- a PETSc that silently reconfigures its dependency set between
builds is the same attribution problem as a mutable image tag.

Two notes carried from measurement:

* **OpenBLAS must be the spack `threads=openmp` build, selected by glob** -- the scipy wheel renames
  every symbol (`scipy_cblas_dgemm`), so linking against the wheel NEVER resolves.
* **Keep vendored NVIDIA cub away from the HIP build.** The backend is chosen only in `gpucub.cuh`;
  a HIP build that sees NVIDIA cub dies.

## Build GCC and LLVM with spack, INSIDE the container

Use spack in the Dockerfile for the compilers, their GPU offload targets, and the polyhedral
optimizers -- GCC + **Graphite**, LLVM + **Polly** (plus MLIR). Spack is what makes the offload
variants and the polyhedral options selectable rather than hoping a distro package carries them.

Note this supersedes the earlier standing preference for running spack OUTSIDE containers: that
preference is about doing WORK on the host, and does not apply to constructing an image, where the
whole point is that the toolchain is baked in and reproducible from the Dockerfile alone.

Still applies when doing it: **name `CC` / `CXX` / `FC` explicitly**, because a stale configure
cache beats `PATH` and a toolchain that looks selected can still be ignored.

## One Dockerfile per image, end to end

Each image is built by exactly one Dockerfile that installs and builds everything it needs. No
out-of-image `PYTHONPATH`, no post-build install step, no "run this script first". If something is
reached from outside the image it is invisible to the image digest, and two runs of "the same
image" can then differ by it -- which is precisely the attribution problem the version suffixes
were compensating for.

## Baselines and frameworks (judge + agent images)

Every framework the harness can dispatch to must be importable, or an arm silently cannot run.
The set is defined by `hpcagent_bench/frameworks/*_framework.py`, not by taste:

**cupy, dace, jax, native, numba, pluto, pythran, triton, tvm.**

Plus the baseline layer: **numpy** (the default baseline -- `speedup = baseline_ns / native_ns`
resolves to it unless a track overrides), **scipy**, and **PyTorch**, which is the ML track's base
reference as well as what the `torch` / `onnx` DaCe library nodes need.

Platform notes that decide how each is installed:

* **cupy differs per image.** CUDA gets the wheel; AMD needs a **HIP source build** -- there is no
  ROCm wheel, and this is already the reason the two judge/agent images cannot be one Dockerfile.
* **JAX on ROCm is a separate build** from the CUDA one; do not assume the CUDA install recipe
  transfers.
* **triton**: the vendor build. On AMD note the inference images carry AMD's ROCm
  `triton_kernels 1.0.0+amd.rocm7.2.0`, which is NOT upstream PyPI `triton_kernels 0.1.0` and lacks
  `matmul_ogs` -- that difference is what pins vLLM to 0.23.0. Keep the judge/agent triton distinct
  from that and do not "fix" one with the other.
* **tvm** and **pythran** need building; pythran is a C++ transpiler so it needs the same compiler
  the rest of the image standardises on.
* **pluto** is a polyhedral source-to-source tool, not a Python package -- it needs isl, clan and
  candl. It pairs with the Graphite/Polly story above rather than with the Python stack.
* **dace** is installed from the extended branch, never pinned to a release -- see the standing
  rule that the venv tracks `origin/extended`.

Guard this the way the rocprof-compute layer is guarded: after installing, assert every adapter
imports, and assert numpy / scipy / pandas / astunparse are the versions the image intends. A
framework install that silently moves numpy changes every CPU reference number the judge computes.

## Open experiment: can the vLLM 0.23 pin be retired?

If it can, the target drops from five images to four. The pin exists for exactly one reason:
vLLM 0.27.1 routes gpt-oss through its mxfp4 path in `process_weights_after_loading` regardless of
`--dtype bfloat16`, and that path imports `triton_kernels.matmul_ogs`
(`fused_moe/oracle/mxfp4.py:1137`). AMD's `triton_kernels` has no `matmul_ogs` submodule; upstream
PyPI `triton_kernels 0.1.0` does. Jobs 601854-601857 died there in ~7 minutes; 600516 is the last
oss120b that served, on 0.23.

**Do not assume a ROCm-version bump fixes it.** Measured: BOTH vLLM images are already on ROCm
**7.2.3** (`rocm723-vllm-0.23.0-...sqsh`, `rocm723-vllm-0.27.1-...sqsh`), and the 0.27.1 image still
ships `triton_kernels 1.0.0+amd.rocm7.2.0`. The build tag does NOT track the image's ROCm, so
`+amd.rocm7.2.3` is a different build of the same AMD source tree, not upstream's. The question is
whether ANY AMD build exposes `matmul_ogs`, not which ROCm it was built against.

The experiment, in order, cheapest first:
1. In the existing 0.27.1 image: `python -c "import triton_kernels.matmul_ogs"`. If it imports, the
   premise is already stale and 0.27.1 is usable today.
2. If not, install `triton_kernels 1.0.0+amd.rocm7.2.3` (or newer) and re-check the same import.
3. Only if that also fails is the upstream-PyPI swap the remaining option -- and that is the
   unvalidated-on-gfx942 path the original note warns about, so it needs an accuracy gate, not just
   a successful serve.

**Global ROCm 7.2.3 pin.** Judge/agent and both vLLM images are already there. The only holdout is
SGLang at **7.2.0**, which is what every measured kimi/qwen38 throughput number ran on. Re-pinning
it to 7.2.3 is acceptable -- CPU results do not depend on the ROCm version -- but it invalidates
those serving-throughput figures, not the graded speedups. Re-measure tok/s after, do not carry the
7.2.0 numbers forward against a 7.2.3 image.

## Prefer apt for the standard numerical libraries, and make them REQUESTABLE

Two facts decide this together:

* `languages.library_tokens()` resolves a requestable library through **pkg-config**, not a path,
  "because prefixes are per-machine spack hashes". apt packages ship `.pc` files as a matter of
  course; a spack build lands in a per-hash prefix that pkg-config will not find unless the image
  also manages `PKG_CONFIG_PATH`. So an apt-installed library is requestable for free.
* `library_tokens` returning empty means "this host cannot build against it", and every caller
  treats that as NOT ON OFFER rather than an error -- because advertising a library the container
  lacks turns into a build failure recorded against the AGENT. That misattribution is the whole
  reason the path exists, so a library present in the image but absent from pkg-config is worse
  than useless: it is invisible.

**So: install LAPACK, ScaLAPACK, FFTW, HDF5, GSL, SuiteSparse, SuperLU, MUMPS, METIS/ParMETIS/Scotch
and friends from apt** where a distribution package exists, and reserve spack for what apt cannot
give: the compilers, and anything needing a variant apt does not build (GPU-enabled PETSc/MAGMA,
`threads=openmp` OpenBLAS).

**The netlib-LAPACK / OpenBLAS collision is a solved problem on Debian, not a reason to omit one.**
`libblas.so.3` and `liblapack.so.3` are `update-alternatives` slots: install both and SELECT which
implementation the soname resolves to. Set OpenBLAS as the alternative so everything still links one
implementation, and netlib remains present for anything that asks for it by name. Assert the
selection in the build gate -- an image where the alternative silently points at reference LAPACK
would make every BLAS-heavy reference number slower for no visible reason.

### libraries.yaml is the agent-facing surface and is far too short

Today it offers **nine**: `blas lapack fftw tbb hiptensor cutensor blis tblis hptt`. **ScaLAPACK is
absent**, as are MPI, SuiteSparse, SuperLU, MUMPS, STRUMPACK, PETSc, Hypre, SLEPc, ARPACK, MAGMA,
METIS/ParMETIS/Scotch, HDF5, Eigen, GSL, and every vendor BLAS/FFT/SPARSE/SOLVER
(cuBLAS/rocBLAS, cuFFT/rocFFT, cuSPARSE/hipSPARSE, cuSOLVER/rocSOLVER, CUB/hipCUB, Thrust/rocThrust).

Every library the image installs must have an entry, or the agent cannot reach it. Adding an entry
is cheap -- the resolution is pkg-config and the empty-means-unavailable rule already makes a
per-host gap safe -- but it must be done deliberately, because an entry whose `.pc` file is missing
degrades silently to "not on offer" and nobody notices the library was meant to be there.

**Gate it from the image side:** for every name in `libraries.yaml`, assert `pkg-config --exists`
succeeds in the built image. That turns "the agent could not link ScaLAPACK" from a silent
non-offer into a build failure, which is the correct place to find out.

### The agent needs a way to ask

`libraries.yaml` is requestable in principle, but the agent toolset is `score`, `submit`, `profile`,
`search`, `syntax_check` -- there is no surface for "what can I link, and give me the flags". Add
one, in the same shape as `profile`: a tool that lists what this host actually offers (the
pkg-config answer, not the yaml) and returns the compile and link tokens for a requested name. The
skills should then teach ASKING rather than listing library names, for the same reason the
profiling pages now teach the `/profile` route rather than command lines: a list in a page goes
stale against the image, while a tool answers for the image that is actually running.

## The serving configs the new images must stay compatible with

Read out of the live v9 env files, not from memory. A rebuilt image that cannot run these has
regressed, however clean its Dockerfile is.

### SGLang -- qwen38 and kimi, 72 env files

    INFERENCE_NODES=4   INFERENCE_MODE=pp
    SGLANG_PYTHON=/opt/venv/bin/python3
    SGLANG_USE_AITER=1
    SGLANG_ROCM_FUSED_DECODE_MLA=0
    SGLANG_SET_CPU_AFFINITY=0
    SGLANG_EXTRA_ARGS="--trust-remote-code --attention-backend triton --language-only
      --watchdog-timeout 1800 --kv-cache-dtype fp8_e4m3 --page-size 64 --context-length 262144
      --mem-fraction-static 0.42 --cuda-graph-max-bs-decode 64 --enable-metrics
      --enable-hierarchical-cache --pre-warm-nccl --reasoning-parser kimi_k2 --tool-call-parser kimi_k2"

What the image must therefore provide, each with the reason it is not negotiable:

* **`/opt/venv/bin/python3` must exist at that exact path** -- `SGLANG_PYTHON` names it literally.
  Relocating the venv silently breaks every SGLang arm.
* **aiter ON.** `SGLANG_USE_AITER=1` is set by every live kimi and qwen38 arm. The "master switch
  off" rule is a vLLM/gfx942 MLA-prefill fact and does NOT transfer here -- so the JIT prebuild
  matters more on this image, not less, since the baton lock is on the path every arm takes.
* **triton attention backend** must work (`--attention-backend triton`), which is the ROCm triton
  from the vendor base -- not PyPI's.
* `--page-size 64` and `--kv-cache-dtype fp8_e4m3` are the gfx942 recipe; `--cuda-graph-max-bs-decode 64`
  is the fix that recovered 4.7x KV (the KV shortfall was graph-capture residual, not mem-fraction).
* `--mem-fraction-static 0.42` is LOW on purpose: MI300A reports `is_integrated`, so SGLang sizes KV
  against node-wide RAM per rank and the fraction runs backwards. The VLM path also derates it by
  0.85 at parse time. Do not "fix" this upward.
* Both `--reasoning-parser kimi_k2` AND `--tool-call-parser kimi_k2`. With only one, turn 1 returns
  400 and is logged as SUCCESS.

### vLLM -- oss120b

    INFERENCE_NODES=1   INFERENCE_MODE=replicas
    VLLM_EXTRA_ARGS="--dtype bfloat16 --load-format safetensors --safetensors-load-strategy prefetch
      --generation-config auto --enable-auto-tool-choice --tool-call-parser openai
      --reasoning-parser openai_gptoss --max-model-len 131072 --gpu-memory-utilization 0.70
      --max-num-seqs 128"

`--generation-config auto`, never `vllm`: the `vllm` value DISCARDS the model's own
`generation_config.json`, which silently changed sampling in twenty env files once already.
`INFERENCE_MODE=replicas` is one standalone TP4 server per inference node behind LiteLLM
round-robin -- PP=4 lost 42% of engine time to 30 s stalls while PP=1 arms lost none.

**Smoke every rebuilt image against these arguments before promoting it**, not just against
"the server started". A tuned-MoE config that fails to load is the failure that voided a whole set
of throughput numbers, and it does not announce itself.
