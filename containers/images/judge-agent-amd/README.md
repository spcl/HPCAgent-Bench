# judge-agent-amd: problems found on Beverin and their fixes

This file records every defect we found in the AMD judge and agent images (`judge-agent-amd` and `judge` roles,
the `mi300` and `mi200` partitions of Beverin) on the way to v1.0. Each entry states the symptom, the cause, the fix and where
it lives, and the job that showed it. Read it before changing the Dockerfile, the EDF templates
(`agent.edf.toml.in`, `judge.edf.toml.in`), `mpi_gpu_check.sh` or `verify_image.py`: most of these defects pass a
build silently and fail only at run time, on another node, or under a profiler.

How to build, verify and promote the images is in [containers/README.md](../../README.md#amd-beverin); what every
image must carry is in [IMAGE_REQUIREMENTS.md](../IMAGE_REQUIREMENTS.md); the NCCL and RCCL settings are in
[docs/containers.md](../../../docs/containers.md#nccl-and-rccl-over-slingshot).

## MPI and the Slingshot fabric

### Cross-node `MPI_Init` asserted in `mpir_hwtopo.c`

- **Symptom.** Every MPI launch across two `mi300` nodes under the plain judge EDF aborted in `MPI_Init` with an
  assertion on the cxi NIC's `io_device` (`mpir_hwtopo.c`). One-node launches passed.
- **Cause.** The spack environment reused a cached `hwloc ~pci`. Without PCI support hwloc reports no PCI device,
  so MPICH cannot find the cxi NIC by its bus id.
- **Fix.** The image builds MPICH against `hwloc +pci`, and the MPI build gate refuses a hwloc without PCI support
  (6e961399c, in every image built since 2026-11-20). Until a two-node run without it passes, the judge and agent
  EDFs also preload the distro `libhwloc.so.15`, which sees PCI (9a81172fa).
- **Evidence.** Job 756270, with the preload: 2 and 8 ranks over two `mi300` nodes open cxi, at 14.4 µs for 8 B and
  20.5 GB/s for 16 MB, against 31 µs and 2.1 GB/s over tcp. Job 758676 on the v1.0 candidate: 8 ranks over two `mi300` nodes form one communicator, and a host allreduce and a GPU-aware allreduce of device buffers
  both return the expected sums.

### Hydra died of SIGFPE in a one-node step of a multi-node job

- **Symptom.** `mpi_gpu_check.sh` failed its multi-rank and GPU-aware checks with an empty message when the build
  ran in a two-node job (754888). The libfabric and provider probes then found nothing.
- **Cause.** `mpiexec -launcher fork` still reads its host list from the Slurm allocation. A one-node step of a
  two-node job sees the job's `SLURM_NODELIST` (two nodes) with the step's `SLURM_TASKS_PER_NODE` (1), and MPICH
  4.3.2's Hydra dies of SIGFPE on that pair before printing anything (rc 136).
- **Fix.** Ranks on this host start with `mpiexec -launcher fork -hosts localhost`, which asks Slurm nothing. A
  failed launch now reports its rc, the signal it decodes to and its whole stderr (b5aeb3de1). Reproduced and
  confirmed in 755662.

### A one-node provider probe chose sockets on a Slingshot node

- **Symptom.** The provider check of job 755924 opened sockets on a node with a cxi NIC.
- **Cause.** MPICH 4.3 skips cxi when all ranks share one host (`MPIR_CVAR_SINGLE_HOST_ENABLED=1`). The old check
  also read the `provider: cxi, score` lines, which list every candidate rather than the provider MPICH chose.
- **Fix.** The probe disables single-host mode, so it selects a provider the way a cross-node launch does, and reads
  only MPICH's `libfabric provider:` line (f36221b8d). `MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG` does not exist in
  4.3.2 and is not used.

### The netstack libfabric read as inconclusive

- **Symptom.** The libfabric check reported INCONCL on Slingshot nodes.
- **Cause.** The check knew the netstack artifact's libfabric only by its `/opt/cscs` path; the process maps the
  bind-mounted copy under the image's own path.
- **Fix.** The artifact is recognised by file identity (`-ef`). A Slingshot node is detected from
  `/sys/class/cxi` as well as `/dev/cxi*`, so a node whose cxi hook did not run fails instead of passing as a
  laptop. On Slingshot, a non-cxi provider (a silent tcp fallback), a missing provider line or no mapped libfabric
  each fail (b4072465a).

## Profilers

### rocprofv3 hung under the EDF's `LD_PRELOAD`

- **Symptom.** A one-kernel HIP probe under `rocprofv3` never returned and ignored `timeout`'s SIGTERM (757779).
  The same environment reached every `/profile` and `rocprof-compute` child of the judge.
- **Cause.** The AMD EDFs preload mimalloc and hwloc. Under that preload the tracer's ring buffer fails to mmap at
  program exit (`ring_buffer.cpp:106`, errno 22), rocprofv3 aborts, and its signal handler hangs.
- **Fix.** `gpu_profiling.rocprof_env` builds every traced child's environment without `LD_PRELOAD`;
  `rocprof_record` and the `rocprof-compute` profile use it, and `verify_image.py` builds the same environment for
  its probes (c3a30e3a6).

### rocprofv3 crashed OpenMP-offload builds as an OMPT tool

- **Symptom.** An offload build linking OpenBLAS crashed in `ompt_post_init` under `rocprofv3`.
- **Cause.** rocprofv3's preloaded library starts as an OMPT tool.
- **Fix.** Every AMD traced child runs with `OMP_TOOL=disabled` (`gpu_profiling.ROCPROF_CHILD_ENV`, 3861de4b1).

### likwid-perfctr did not know the MI300A CPU

- **Symptom.** In the `mi300` judge image, `likwid-perfctr` answered "Unsupported AMD Zen Processor" and listed no
  group (750671).
- **Cause.** Ubuntu's likwid predates the MI300A CPU (family 0x19, model 0x90); upstream added it after 5.5.1.
- **Fix.** likwid is built from a pinned master commit (653ba0a01) with `ACCESSMODE=perf_event`, since a container
  reaches no MSR (2674fa73c). In job 750681 it detects Zen4, lists its groups and counts `FLOPS_DP`.

### rocprofv3 hung when run from a read-only directory

- **Symptom.** In the full `verify_image.py` run of the v1.0 candidates (757798, 758027, 758676), the `rocprofv3`
  trace, `rocprofv3 --pmc` and `rocprof-compute` checks timed out (600 s, 600 s and 1500 s). The same probe passed
  in about one second in a separate job (758035).
- **Cause.** The verify EDF sets `workdir = "/"`, which is read-only in the container, and the three checks ran
  their tool from there. From an unwritable working directory, rocprofv3's ring buffer fails to mmap at exit
  (`ring_buffer.cpp:106`, errno 22); rocprofv3 aborts (signal 6) and its signal handler hangs, as under the
  preload above. The 758035 probe ran from a writable directory. In job 758676 the same probe fails from `/` and
  passes from `/tmp`, with and without `LD_PRELOAD` and with a minimal environment. The harness was not
  affected: `rocprof_record` and the `rocprof-compute` profile always run in their work directory.
- **Fix.** The checks run in their probe's temporary directory, as the harness does. All three pass from `/` in
  job 758676 (3 s, 2 s and 31 s).
- **Diagnosis.** A timed-out check used to report `rc=127` with only `Command [...] timed out`. It now reports rc
  124, as coreutils `timeout` does, and its note ends with the child's last line of output (b3f3fe421); that line
  named rocprofv3's signal handler.

## Image builds

### `spack location -i gcc` was ambiguous

A bare `gcc@16.2.0` also matched the external gcc spack had detected, so `spack location -i` refused the spec and
`/opt/gcc` pointed nowhere after a 25-minute gcc build. The Dockerfile names gcc's variants wherever it looks up the
prefix (2bc8f4895).

### spack links were OOM-killed on 512 GB nodes

`build_and_verify.sbatch` asked for a third of a node and capped spack at a fixed 64 jobs. The job now takes its
node exclusively, and `ce_spack_jobs` sets `SPACK_BUILD_JOBS` to the node's cores, at most one job per 8 GB: 64 on
a 512 GB `mi300` or `mi200` node, where more jobs OOM-killed the llvm and petsc links (8d145c8db).

### A failed role left its sibling promotable

A verify failure of one role used to leave the other role's `.verified` marker in place, and a rebuilt candidate
could be promoted on a stale marker. A failed role now clears every role's pass, and an export drops the old pass
(3ae323bc7).

## Operating the build, verify and push

These are procedures, not image defects. Each one cost a verify or suite run on the way to v1.0.

- **Never wrap `build_and_verify.sbatch` or `verify_image.sbatch` in an `srun` step.** Both start their own steps.
  Inside an outer step that holds a node's CPUs, those inner steps wait for resources until the job's time limit:
  job 758042 spent 6 h in "Requested nodes are busy". Call them from the batch shell.
- **Every inner step is one task on one node.** An `srun` without `--nodes` in a multi-node job asks for all the
  job's nodes, so in job 758676 both verifies waited while a suite step held the third node. The verify steps now
  pass `--nodes=1 --ntasks=1`.
- **Two nodes before promotion.** `containers/images/mpi_two_node_check.sbatch` (EDF= the candidate's EDF, two
  `mi300` nodes) checks one communicator across nodes, a GPU-aware allreduce and provider `cxi`. The `~pci` hwloc
  abort passed every one-node check and showed only in production runs (756270), as the GH200 images were checked
  across two nodes from the start.
- **The build gate reads libmpi's own NEEDED entries.** `ldd` is transitive; the gate also requires `libmpi.so` to
  NEED `libamdhip64`, which an MPL built without HIP does not (the mi300 v1.0 image NEEDs `libamdhip64.so.7`).
- **A push deletes the local `.oci.tar`.** `registry.sh push` removes the archive once the registry holds it. A
  verifier built after the push takes its judge from the registry tag (the default `BASE_IMAGE`), not from
  `oci-archive:`.
- **Promotion renames the candidate.** An EDF that names a `*-candidate.sqsh` stops working at `promote`. A run that
  must outlive the promotion names a hard link to the candidate instead.
- **The verifier records the checkout's HEAD.** Build it from a clean tree at the commit being released.
- **An in-image unit suite follows CI's split.** `tests/test_e2e_numerical.py` runs apart from the unit sweep, one
  backend set per run (`HPCAGENT_BENCH_E2E_BACKENDS=numba`, then `c,cpp,fortran`), with `--timeout=900`. Swept
  with the rest at `--timeout=600` and all five backends, including pluto, a stem's first test outlives the limit;
  pytest-timeout's thread method then calls `os._exit`, which xdist reports as "node down: Not properly
  terminated" and which can end the session in an xdist `INTERNALERROR` (757800, 758027, 758035). There was no OOM
  kill (`memory.events` `oom_kill 0`, 52 GB peak).

## The in-image unit suite

### Every GPU test failed once sixteen workers shared device 0

- **Symptom.** In the in-image sweep at `-n 16`, after a few hundred passing Triton tests every later GPU test of
  every worker failed: Triton references with `runtime_error`, torch, cupy and HIP tests with `hipErrorLaunchFailure`
  (unspecified launch failure), and some workers died in jax tests (758864, 758868, 758887; three of the four sweeps
  of 758887, and the GPU test files alone at `-n 16`). The Triton file alone at `-n 4` passed (758684).
- **Cause.** Nothing narrowed a worker's devices, so the in-process torch, Triton, jax and cupy calls of all sixteen
  workers ran on GPU 0. The failures start on every worker at once, which is a device-wide fault, not a context one
  test broke. The judge was not affected: each of its slots narrows its grading child to one GPU.
- **Fix.** `tests/worker_slot.py`, called from `tests/conftest.py` before any device runtime loads, narrows
  worker `gwN` to GPU `N mod count` (`ROCR_VISIBLE_DEVICES` on AMD), counting GPUs from sysfs, never through a
  runtime (f1ac69be7). A failed Triton reference test now shows the run's traceback, which the run printed and the
  `runtime_error` verdict dropped.

### An honest 100 ns kernel was timed at 1 ns

- **Symptom.** `test_cpu_refuses_gpu.py::test_an_honest_host_grade_keeps_its_measured_credit` failed in one sweep of
  758887: the honest C sum of 512 doubles read `native_ns=1.0`, was credited 4220x and marked suspect.
- **Cause.** The parent's clock subtracts the median handshake cost from each call and floored the result at 1 ns
  (`trusted_clock.summary`). Under load the handshake varies more than such a kernel lasts.
- **Fix.** A host call is floored at the handshake's spread (`trusted_clock.host_samples`, f1ac69be7): a call the
  noise hides is credited as the noise. Kernels this short are below what the harness can measure (see "What the
  clock can resolve" below).

### The memo guard's margin was tighter than the node's swing

`test_memo_guard.py::test_varied_inputs_suppress_the_memoized_speedup` failed about two runs in five. Every
repetition of five guarded and five honest grades (758886) alternates between runs of about 100 us and about
250 us for both sources, while a credited cache hit would skip the loop entirely. The margin is 5x (702b462a1).

### A worker outlived pytest-timeout while the clock held its child stopped

- **Symptom.** In six of seven sweeps of rounds 4 to 10 a grade, often a `test_numba_reference_overrides` case
  (`bdf_newton_krylov`, `channel_flow`), outlived pytest-timeout's 600 s. The kernel child was in state `T`, stopped
  inside `sendto` by the clock's `F_SETSIG(SIGSTOP)`, and the worker waited in `run_forked` (758887).
- **Cause.** No deadlock. The thread dump of 758889 shows the clock's DONE handler holding the candidate stopped
  while `check_outputs` computes the expected outputs: `run_expected` calls `cached_reference`, which ran a torch
  reference child sized to the whole node. With sixteen workers each such child ran about a hundred threads, and
  one did not finish within 600 s. The judge sizes each child to its slot, so this was a test-suite effect.
- **Fix.** `tests/worker_slot.py` also pins worker `gwN` to the N-th share of the cores, so every child it starts
  sizes itself to that share (dff818949). A pytest child a test starts in its own process inherits the share and is
  not pinned again: re-pinned, `bdf_newton_krylov`'s child ran on 18 of 288 cores and timed out (758901, aa06dfae3).
  No sweep of 758901 hung.

### jax processes sharing one GPU crashed and took the GPU with them

- **Symptom.** With one GPU per worker, three of six sweeps of 758901 still failed every later GPU test of some
  workers with `hipErrorLaunchFailure`, after a worker died in a jax test (mostly `lenet` and `conv2d`).
- **Cause.** At `-n 16` four jax processes share each MI300A GPU. One of them segfaults in XLA's executable call
  (`pxla` `__call__`), and the fault fails every process on that GPU (`hipCtxSetCurrent` 719). The jax tests alone
  at `-n 16` reproduce it (758901); the judge never runs two jax grades on one GPU.
- **Fix.** The jax reference tests share one `xdist_group`, so one worker runs them all (aa06dfae3). Their failure
  message carries the run's printed tail, as the Triton test's does.

### Forked ranks and pool workers of a multithreaded worker

- `test_cholesky_recording_race.py` failed in every sweep of 758901 and 758921: none of its four ranks, forked from
  the xdist worker, reported within 120 s. The file alone on one worker's 12 cores passes in 44 s. The ranks now
  start from a `forkserver`, as the dace emit pool's do.
- `test_dace_frontend_validity.py` lost a process-pool worker (758887, 758901): a child forked from a
  multithreaded xdist worker segfaulted importing dace. The pool now starts its workers with `forkserver`.
- `test_mpi_shard_driver_cpu_gloo_real_launch.py` failed four of five cases in the dedicated run of 758921 with
  "mpirun/mpiexec timed out": every rank imports torch, which took past the 90 s limit while seven sibling steps
  imported their own stacks. The file alone passes in 118 s serially and 28 s at `-n 8`. The limit is now 300 s.
- `test_packaging.py::test_wheel_is_pip_installable_and_complete` imported `hpcagent_bench` from the checkout
  because the probe script put it on `PYTHONPATH`; the probe no longer does. In 758921 it failed copying the
  package: a sibling worker's `write_atomic` staging file was renamed away mid-copy. The copy now skips `.*.tmp`.

## Open at the time of writing

- **GPU faults with four test processes per GPU.** With jax on one worker, three of five sweeps of 758921 still
  failed the later Triton reference tests of the workers on one or more GPUs, first with `hipErrorIllegalAddress`,
  then `hipErrorLaunchFailure`. The Triton file alone at `-n 16` passes (758901), so the fault needs a mix of GPU
  workloads on one device; which one faults first is not known. The judge runs one grade per GPU and no image
  build runs the suite in parallel. Until the cause is known, run the GPU test files at `-n 4`, one worker per GPU.
- **A stage that missed once.** `test_cpf_cache.py::test_check_and_stage_strip_the_kernel_names_the_render_and_verify_commands_strip`
  failed once in 758901: `check` found the gemm form and `stage` exited 1 right after. The test now shows what
  `stage` printed to stderr, which names the miss.
- **The plugin harness venv in the image.** `test_plugin_harness.py::test_a_plugin_harness_venv_is_built_from_its_install_with_the_agent_on_its_path`
  fails in every sweep of 758901 that reaches it: the harness venv can import `hpcagent_bench`. The file alone passes
  in the image (758921) and in CI, so an earlier test of the sweep leaves something behind; no test writes
  `PYTHONPATH`. The probe now names where `hpcagent_bench` was found.

- **What the clock can resolve.** On the mi300 judge the parent's handshake costs a median 33 us with a 45 us spread,
  so a host call under about 2.3 ms is recorded `timing_short`. No graded host cell of the November regrade is that
  short; the suite's preset S tests are (docs/measurement_statistics.md, "What the clock can resolve"). A device
  grade is no longer flagged short against the host handshake, which its GPU event time does not contain.
