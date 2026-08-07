---
name: rocprof-compute-judge
description: Kernel-level analysis on AMD, off-judge -- Speed-of-Light first, then the memory chart, then the pipe. The ncu-shaped question, answered with CU-shaped numbers.
---

`rocprof` answers WHICH kernel owns device time. This page answers WHY THAT KERNEL IS SLOW: which
hardware block is at its limit, how far the kernel is from the roof, and which pipe was issuing.
It is the AMD counterpart of `ncu`, and the ladder below is the same ladder -- the numbers are not.

Run `rocprof` first anyway. A perfectly analysed kernel that owns 4% of the run is 4%.

## What was measured here, and what was not

**No PROFILE was collected and no number below was observed** -- not one metric, threshold, chart
or formula on this page came off a run. Every claim is quoted from upstream WITH ITS URL, in place:
the ROCm docs, `rocprofiler-compute`'s per-part metric definitions, or `rocprofiler-sdk`'s counter
definitions. Anything that could not be sourced was deleted rather than hedged, because a fenced
claim with no link is indistinguishable from an invented one. Check the first command against your
own `--help` before building a plan on it.

Exactly one thing WAS executed, and it is why nothing else was, on a Radeon 780M with ROCm 7.2.4:
`rocprof-compute` is INSTALLED by the
distro ROCm packages and still refuses to run, because it pins Python dependencies the system
Python does not satisfy. Every subcommand -- including `--help` -- exits after printing:

```
[ERROR] the 'astunparse==1.6.2' distribution does not meet version requirements to use rocprofiler-compute.
  --> version installed : 1.6.3
[ERROR] The 'plotext' package was not found in the current execution environment.
[ERROR] The 'dash>=3.0.0' package was not found in the current execution environment.
   ... 11 packages in total
```

Note it exits **0**, so a wrapper that checks the return code concludes the profile succeeded and
finds no output. The pin is exact (`==1.6.2`) and the installed version is NEWER, so this does not
resolve by upgrading. Build the venv with `--system-site-packages`:

```sh
python3 -m venv --system-site-packages ~/.venvs/rocprof-compute
~/.venvs/rocprof-compute/bin/pip install -r <rocm-root>/libexec/rocprofiler-compute/requirements.txt
```

The flag is not optional. `rocprof-compute` lives under the ROCm tree and imports the
distro-installed ROCm Python modules; an ISOLATED venv satisfies every pinned pip requirement and
then fails on those instead, which reads as the diagnosis having been wrong. Confirm
`rocprof-compute --help` actually prints its usage before assuming the tool is available on any
host.

What is NOT vendor folklore is the reading ORDER, and the reason to trust it here is a measured
one: on the NVIDIA twin of this page, following the ladder in order produced a **47.4x** kernel
speed-up, beating three of the vendor's own shipped recommendation blocks -- because the vendor's
blocks each argue for their own chapter and the ladder decides which chapter to be in. That part
ports. The thresholds do not.

## The name changed twice

| you may see | current name | what it is |
| --- | --- | --- |
| `omniperf` | `rocprof-compute` | THIS page: kernel-level counters, SOL, roofline |
| `omnitrace` | `rocprof-sys` | whole-application trace, CPU+GPU timeline |
| `rocprof` / `rocprofv2` | `rocprofv3` | the dispatch trace and raw `--pmc` collection |

The first row is in the successor's own title, "ROCm Compute Profiler (formerly Omniperf)"
(https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/), and the third has an
option-by-option comparison upstream
(https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/conceptual/comparing-with-legacy-tools.html).
Search results and older tuning guides are full of the left column. They describe the same tools.
If `rocprof-compute` is not found, try `omniperf` before concluding the tool is absent.

## How it runs

**The judge has no route to this tool.** `POST /profile` on a `hip` submission runs `rocprofv3` --
the dispatch trace, which kernel owns device time -- and that is the only instrument it attaches:
`linuxperf`, `papi` and `none` each come back 400 naming `rocprofv3`, because a device kernel has no
host-side bracket for them to run in. Nothing there replays a kernel for counters and nothing hands
back a workload directory. Take the kernel name off the trace, then run everything below on a box of
your own.

Three things about the tool that change how you use the ladder below:

- **You profile ORDINARY source.** There is no bracket to write and no counter to name -- unlike
  the PAPI bracket, the tool attributes per dispatch on its own. What you lose is the ability to ask
  about a REGION that is not a kernel.
- **Replay is your cost, and it is also your problem.** The tool runs your application repeatedly
  to collect all counters, so a program whose output depends on an unseeded RNG or on wall clock
  produces counter rows from runs that did different things. Nothing detects that for you. Fix the
  determinism before you profile, not after.
- **A multi-rank run needs a single-pass mode.** Default replay re-runs the workload, and the
  second `MPI_Init` is not legal -- so an MPI application profiled this way fails rather than
  answers. See the replay section below for the two documented ways out.

Run with `--no-roof` by default while iterating; collect the roofline once, at the end, when you
want the picture rather than a number.

## The two-command shape

Profiling writes a WORKLOAD DIRECTORY, and analysis reads it back. That split is the point: you
collect once and then ask many questions of the same data, so do not re-profile to change a
question.

```
workloads/<name>/<gpu_model>/
  log.txt                all profiling output
  perfmon/               one counter-set input file per collection run
  pmc_perf.csv           the merged counter results
  roofline.csv           absent if you passed --no-roof
  sysinfo.csv            the PART. read this first
```

That is upstream's own listing of the directory, and its own description of the extra files: "An
SoC parameters file, `sysinfo.csv`, is created to reflect the target device settings. All profiling
output is stored in `log.txt`. Roofline-specific benchmark results are stored in `roofline.csv`"
(https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html). The
`perfmon/` files are what the collector iterates: it globs `perfmon/*.txt` and runs the application
once per file
(https://github.com/ROCm/rocprofiler-compute/blob/develop/src/rocprof_compute_profile/profiler_base.py).

`sysinfo.csv` is the part's geometry, measured. It is what turns every occupancy sentence below
into arithmetic instead of folklore, and it is the file to open first.

## It REPLAYS your kernel, and that is the cost

`rocprof-compute` collects all available counters for the part, and no GPU has enough counter
hardware to do that in one pass. Upstream: "By default, ROCm Compute Profiler uses application
replay mode, which runs the workload multiple times to collect all performance counters" (profile
mode page, above) -- which is the `perfmon/*.txt` loop in the collector. Consequences, all of them
practical:

- **It is slow.** Expect many multiples of one run. Cut the work before you profile, not after.
- **Dispatches are SERIALIZED, and that is a SECOND distortion, independent of replay.** Upstream
  warns: "Kernel dispatches are serialized across HIP streams on the same GPU during profiling
  ... Kernels launched on separate HIP streams on the same GPU will not execute concurrently during
  profiling. Streams on different GPUs are not serialized", so "Kernel duration and throughput
  metrics reflect serialized execution, not the concurrent behavior that may occur during normal
  execution" (profile mode page, above). One pass is enough to get this; replay is not required.
- **Replay BREAKS MPI.** "This mode fails for MPI applications because running the application
  multiple times results in multiple `MPI_Init` and `MPI_Finalize` calls, which is not permitted by
  the MPI specification" (same page). The documented single-pass modes are
  `--iteration-multiplexing`, which "divides the total set of requested performance counters into
  smaller subsets that can be collected over multiple iterations of the kernel execution, thereby
  preventing the need for application replay", and `--set <name>`, "a predefined counter set that
  fits in a single pass". Multiplexing needs ROCprofiler-SDK from ROCm 7.0.0 or later, and it needs
  the workload to run enough iterations to cover every subset -- too few dispatches and some
  counters simply are not collected (same page).
- **The application must be deterministic and re-runnable.** Replay is repeated EXECUTION, so a run
  whose output depends on wall clock, RNG without a fixed seed, or a file it consumes-and-deletes
  produces counter rows from runs that did different things, and nothing in the merged CSV says so.
- **Roofline is a second collection stage** on top of the first: "The first stage collects all the
  counters needed for ROCm Compute Profiler analysis ... The second stage collects data for the
  roofline analysis (this stage can be disabled using `--no-roof`)" (same page). `--no-roof` is the
  first flag to reach for while you are iterating.

Narrow before you widen. Upstream's own filter list: "`-k`, `--kernel` Enables filtering kernels by
name. `-d`, `--dispatch` Enables filtering based on dispatch ID. `-b`, `--block` Enables collection
metrics for only the specified analysis report blocks"
(https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/use.html). Dispatch
"indices are 1-based, so the first dispatch of a kernel is `1`" and a range is `start:end` or
`start-end` (profile mode page, above) -- which is how you profile the steady-state iteration
rather than the cold first one.

```sh
rocprof-compute profile --name vcopy --no-roof -k vecCopy -d 3:8 -- ./vcopy -n 1048576
```

## Read it in this order

Stop at the first step that fires. The later numbers are consequences of the earlier ones, so a
number read out of order will send you to the wrong chapter with real evidence for it.

**1. System Speed-of-Light.** One panel, every major block as a percentage of its own peak. This
is the whole triage: the block nearest its roof is the one to work on, and every other panel in
the tool is an explanation of that one number. If nothing is near a roof, the kernel is
latency-bound and you are in step 2, not step 4.

**2. Wavefront launch and occupancy -- against the PART.** The wavefront width is the thing you
must not carry over. HIP: "The size of a warp is architecture dependent and always fixed: 64
threads for CDNA architectures [and] 32 threads for RDNA architectures"
(https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html), and
rocprof-compute repeats it where the counters are defined: "On AMD Instinct CDNA accelerators and
GCN GPUs, the wavefront size is always 64 work-items"
(https://github.com/ROCm/rocprofiler-compute/blob/develop/src/rocprof_compute_soc/analysis_configs/gfx942/0700_wavefront.yaml).

The slot count is per-architecture, and the two families give different answers. CDNA: a CU is
"Four 16-wide SIMD processors" with "An instruction buffer (per-SIMD) that contains execution slots
for up to 8 wavefronts (for 32 total wavefront slots on each CU)"
(https://github.com/ROCm/rocprofiler-compute/blob/develop/docs/conceptual/pipeline-descriptions.rst)
-- so a full CU is 32 x 64 = **2048 work-items**. RDNA: "RDNA 2 and RDNA 3 have 16 slots per SIMD"
with "4 SIMDs per WGP" (https://gpuopen.com/learn/occupancy-explained/), and a WGP is two CUs, so
64 wave32 slots per WGP is **1024 work-items per CU**. Read `sysinfo.csv` for the actual part
rather than either figure -- `rocprofv3`'s agent report spells the same number `Max_Waves_Per_Cu`
(https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html) --
because this is exactly the arithmetic that differs by generation.

Low occupancy has two causes this number cannot separate: too few workgroups for the CUs (fix the
decomposition), or a full grid capped by VGPRs or LDS per workgroup (fix the resource use). The
Wavefront Launch panel has the register and LDS figures that tell them apart, both with upstream's
own granularity warning attached -- `VGPRs` "may not exactly match the number of VGPRs requested by
the compiler due to allocation granularity", and `LDS Allocation` "may also be larger than what was
requested at compile time" (gfx942 wavefront panel, above).

Occupancy counts waves PARKED, not waves working -- upstream defines `Wavefront Occupancy` as "The
time-averaged number of wavefronts resident on the accelerator over the lifetime of the kernel"
(gfx942 SOL panel:
https://github.com/ROCm/rocprofiler-compute/blob/develop/src/rocprof_compute_soc/analysis_configs/gfx942/0200_system_speed_of_light.yaml).
It matters only once something else says the CUs stalled.

**3. The memory chart.** The one panel with no NVIDIA analogue worth borrowing: it lays out the
whole hierarchy -- vector L1D, scalar L1D, LDS, L2 (TCC), and the fabric out to HBM -- with the
traffic on each link, one YAML panel per level
(https://github.com/ROCm/rocprofiler-compute/tree/develop/src/rocprof_compute_soc/analysis_configs).
Read it as a flow. The level where the numbers stop shrinking is the level your working set does
not fit in, and that is the level to tile for.

The L2 panel prints `Hit Rate` as a percentage, `100 * TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)`
(https://github.com/ROCm/rocprofiler-compute/blob/develop/src/rocprof_compute_soc/analysis_configs/gfx942/1700_l2_cache.yaml);
the equivalent rocprofv3 metric counts `GL2C_HIT`/`GL2C_MISS` instead on gfx10 through gfx12
(https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/share/rocprofiler-sdk/counter_defs.yaml).
Read it as the EXPLANATION of the traffic, never on its own: a rising hit rate with unchanged HBM
bytes means you added accesses, not locality.

**4. Traffic against the algorithm's minimum.** Needs no peak and no roofline. Count the bytes the
kernel MUST move -- every input read once, every output written once -- and divide the measured
traffic by it.

**Check what KIND of number the panel in front of you holds; the two tools do not agree.**
rocprofv3's derived `FetchSize` is a VOLUME in kilobytes -- "The total kilobytes fetched from the
video memory", computed on gfx942 as `(TCC_BUBBLE_sum*128 + (TCC_EA0_RDREQ_sum - TCC_BUBBLE_sum -
TCC_EA0_RDREQ_32B_sum)*64 + TCC_EA0_RDREQ_32B_sum*32)/1024` (counter_defs.yaml, above), and
`WriteSize` is its twin. rocprof-compute's L2 panel builds the SAME numerator out of the SAME
counters and then divides by the kernel's duration instead of by 1024, declaring `unit: Gbps` --
"Read BW: The total number of bytes read by the L2 cache from Infinity Fabric divided by total
duration" (gfx942 L2 panel, above). Same counters, different quantity. Read that RATE as a volume
and this step's ratio is wrong by the kernel duration; carry the KILOBYTES habit into a tool that
reports bytes and it is wrong by 1024. The panel's `unit` field is the only thing that settles it.

- near 1 -- compulsory traffic. Tiling buys nothing; only a different algorithm does.
- well above 1 -- you are re-reading what should have stayed in cache. This is what tiling and
  fusion are for, and the ratio is how you check it worked.
- write bytes far above the output size -- uncoalesced stores, or a read-modify-write the source
  does not show.

**5. Which pipe.** Only once memory is excluded.

**The names differ between the two AMD tools, and one pair means opposite things.** Read the column
for the tool you are actually running:

| what you want to know | `rocprof-compute` prints | `rocprofv3 --pmc` name |
| --- | --- | --- |
| was the vector ALU busy | `VALU Utilization` | `VALUBusy` |
| how many LANES were active (DIVERGENCE) | `VALU Active Threads` | `VALUUtilization` |
| scalar pipe busy | `SALU Utilization` | `SALUBusy` |
| LDS bank conflicts | `LDS Bank Conflicts/Access`, `Bank Conflict Rate` | `LDSBankConflict` |

`VALUUtilization` and `VALU Utilization` are the trap: near-identical spellings, opposite
quantities. rocprof-compute's `VALU Utilization` "Indicates what percent of the kernel's duration
the VALU was busy executing instructions" (gfx942 SOL panel, above) -- a TIME fraction, `unit: pct`.
rocprofv3's `VALUUtilization` is "The percentage of active vector ALU threads in a wave"
(counter_defs.yaml, above) -- a LANE fraction. The divergence question on rocprof-compute is
**`VALU Active Threads`**: "the average level of divergence within a wavefront over the lifetime of
the kernel. The number of work-items that were active in a wavefront during execution of each VALU
instruction", `unit: Threads` with `peak: $wave_size` (gfx942 SOL panel, above) -- so read it
against 64 on CDNA rather than as a percentage.

The rocprofv3 expressions, read out of `counter_defs.yaml` (above):

| metric | expression on gfx942 |
| --- | --- |
| `VALUBusy` | `100*reduce(SQ_ACTIVE_INST_VALU,sum)/CU_NUM/reduce(GRBM_GUI_ACTIVE,max)` |
| `SALUBusy` | `100*reduce(SQ_INST_CYCLES_SALU,sum)/CU_NUM/reduce(GRBM_GUI_ACTIVE,max)` |
| `MemUnitStalled` | `100*TCP_TCP_TA_DATA_STALL_CYCLES_max/reduce(GRBM_GUI_ACTIVE,max)/SE_NUM` |
| `VALUUtilization` | `100*reduce(SQ_THREAD_CYCLES_VALU,sum)/(reduce(SQ_ACTIVE_INST_VALU,sum)*MAX_WAVE_SIZE)` |
| `LDSBankConflict` | `100*reduce(SQ_LDS_BANK_CONFLICT,sum)/reduce(GRBM_GUI_ACTIVE,max)/CU_NUM` |
| `L2CacheHit` | `100*reduce(TCC_HIT,sum)/(reduce(TCC_HIT,sum)+reduce(TCC_MISS,sum))` |
| `GPUBusy` | `100*reduce(GRBM_GUI_ACTIVE,max)/reduce(GRBM_COUNT,max)` |

The architecture list is PART OF the definition, and the lists are not the same: `VALUBusy`,
`SALUBusy`, `MemUnitStalled`, `VALUUtilization` and `LDSBankConflict` name `gfx942` explicitly,
while `L2CacheHit` and `GPUBusy` are registered for `gfx9`/`gfx90a` and switch to `GL2C_*` on
gfx10-gfx12, and `LDSBankConflict` becomes `SQC_LDS_BANK_CONFLICT / SQC_LDS_IDX_ACTIVE` there
(counter_defs.yaml, above). Ask the tool for the metric BY NAME and let it pick, rather than
hand-computing from a formula for the wrong part.

Note what those denominators are NOT: none of them is `SQ_BUSY_CU_CYCLES`. The normaliser is
`GRBM_GUI_ACTIVE` (GPU active cycles) scaled by a part constant (`CU_NUM`, `SE_NUM`), and
`VALUUtilization` alone divides by `MAX_WAVE_SIZE`, which is why it is the one that is a lane
fraction rather than a time fraction.

**`SQ_WAIT_INST_ANY` is not the memory stall**, though the name invites it and the two get
confused. The memory-unit stall is `TCP_TCP_TA_DATA_STALL_CYCLES`, above; `SQ_WAIT_INST_ANY` is
what rocprof-compute prints as `Issue Wait Cycles`, `AVG((4 * SQ_WAIT_INST_ANY) / $denom)` --
quad-cycles in which a wavefront "was unable to issue an instruction for any reason (e.g.,
execution pipe back-pressure, arbitration loss, etc.)", and upstream adds that it "is most useful
to get a sense of how waves were spending their time, rather than identification of a precise
limiter" (gfx942 wavefront panel, above).

Matrix work rides a separate pipe and a separate counter: `MFMA Utilization` is
`SQ_VALU_MFMA_BUSY_CYCLES` while `VALU Utilization` is `SQ_ACTIVE_INST_VALU` (gfx942 SOL panel,
above), so a GEMM-shaped kernel showing a low `VALU Utilization` is not idle, it is on the pipe you
did not look at.

**6. Roofline, last.** It tells you which side of the ridge point you are on and therefore which of
the steps above can pay at all -- it does not tell you what to change. Memory-bound kernels sit
left of the crossover, compute-bound right, and a kernel sitting far BELOW both curves is neither:
it is latency-bound, and the fix is occupancy or more work in flight, not traffic and not flops.

## What each finding costs the next

| pair | the conflict |
| --- | --- |
| occupancy -> registers | raising waves per SIMD means fewer VGPRs each; past a point the kernel spills to scratch and the extra waves are slower than the spill |
| tiling -> LDS | a bigger tile is more LDS per workgroup, which is itself an occupancy cap. The two settle together |
| LDS -> bank conflicts | the padding that fixes a conflict also changes the tile's LDS footprint, so re-read occupancy after |
| wave64 -> divergence | a 64-lane wave serialises a branch across twice the lanes of a 32-lane one, so the same source diverges harder on CDNA |
| replay -> trust | every counter row came from a DIFFERENT run of your app. Non-determinism does not show up as an error, it shows up as a number |

## Traps

- **`sysinfo.csv` before anything else.** Every occupancy and width sentence above depends on the
  part, and the part is in that file.
- **Do not port NVIDIA thresholds.** Wavefront width, LDS banking, the cache hierarchy and the
  matrix pipe all differ. A number meaning "bad" on an SM does not mean it on a CU.
- **A profiled run's wall clock belongs to no comparison.** Replay and dispatch serialization each
  make it meaningless on their own. Read the COUNTERS; take every speed-up from an uninstrumented
  build.
- **MPI needs a single-pass mode.** `--iteration-multiplexing` or `--set`; the default replay mode
  runs the workload again and the second `MPI_Init` is not legal.
- **Verify the answer.** A kernel that got faster and wrong measures nothing. This is not a
  formality on AMD: the fastest paths here often involve changing the wave width or the LDS
  layout, and both can change a reduction's summation order.
- **`--no-roof` while iterating.** Then one final run with the roofline when you want the picture.

## Documentation

- ROCm Compute Profiler (rocprof-compute), formerly Omniperf -- https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/
- Profile mode: replay, serialization, MPI, iteration multiplexing, every flag quoted above -- https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html
- Basic usage and the `-k` / `-d` / `-b` filters -- https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/use.html
- The performance model: SOL, memory chart, the per-block panels -- https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/performance-model.html
- Definitions: wavefront, work-item, divergence -- https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/definitions.html
- Pipeline descriptions, for the CU's SIMDs and wavefront slots -- https://github.com/ROCm/rocprofiler-compute/blob/develop/docs/conceptual/pipeline-descriptions.rst
- rocprof-compute's per-panel metric definitions and UNITS, per part (`gfx942/*.yaml`) -- https://github.com/ROCm/rocprofiler-compute/tree/develop/src/rocprof_compute_soc/analysis_configs
- The collector's replay loop, one application run per counter set -- https://github.com/ROCm/rocprofiler-compute/blob/develop/src/rocprof_compute_profile/profiler_base.py
- The derived-counter EXPRESSIONS and their architecture lists -- the authority for every formula above: https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/share/rocprofiler-sdk/counter_defs.yaml
- MI300/MI200 counter DEFINITIONS -- https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300-mi200-performance-counters.html
- Occupancy on AMD, wave-per-SIMD arithmetic (RDNA figures) -- https://gpuopen.com/learn/occupancy-explained/
- AMD Instinct MI300 (CDNA3) ISA reference, for the hardware numbers -- https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- AMD's own profiling walkthrough, roofline reading -- https://rocm.blogs.amd.com/software-tools-optimization/profiling-guide/novice/README.html
- HIP programming model: wavefront, CU, LDS, XCD -- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
