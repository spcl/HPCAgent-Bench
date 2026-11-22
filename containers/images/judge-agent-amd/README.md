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

### A verify timeout left no evidence

- **Symptom.** In the full `verify_image.py` run of the v1.0 candidates (757798, 758027), the `rocprofv3` trace,
  `rocprofv3 --pmc` and `rocprof-compute` checks failed as `rc=127` with only `Command [...] timed out` as their
  note. The same probes pass in about one second when run alone in the same image, EDF, CPU count and core-file
  limit, also after every check that precedes them in the verify order (758035).
- **Fix.** A timed-out check now reports rc 124, as coreutils `timeout` does, and its note ends with the child's
  last line of output; `have_exe` still counts a hang as not runnable (b3f3fe421).

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

## Open at the time of writing

- **The rocprof checks of the full verify run** (see above): the rc 124 note of the next verify names where the
  child stopped.
- **`test_npbench_triton_references.py::test_triton_reference_matches_numpy[resnet-S]`.** Its worker died in the
  in-image suite of job 758042, and later Triton tests on that node failed with `hipGetDeviceCount()` error 719
  (unspecified launch failure) until the run hung. Job 758676 runs it alone on one GPU to separate a kernel fault
  from 16 workers sharing four GPUs.
