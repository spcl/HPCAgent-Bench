---
name: nsys
description: Trace a CUDA submission with Nsight Systems -- which kernel, which copy, which gap -- and know when only ncu can answer.
---

A GPU has no call stack to sample. The host thread launches asynchronously and then waits, so a
`perf` profile of a CUDA kernel shows one synchronization call and nothing about the device. What
the device did is RECORDED instead: CUPTI hands `nsys` one activity record per kernel launch and per
memory operation, and the profile is those records, not samples.

That changes what the tool can tell you. `nsys` answers **which kernel, how many times, and when**.
It cannot answer **why that kernel is slow** -- read "nsys or ncu" below before you spend a run on
the wrong instrument.

This is the device half of `profiling`, on NVIDIA. The host instruments (`perf`, PAPI CPU counters)
are that skill; a HIP submission is the `rocprof` skill's, because `nsys` traces CUDA and cannot see
an AMD queue.

## How it runs

`POST /profile` with `language: "cuda"` routes to the GPU path automatically -- the dispatch is the
LANGUAGE, so you ask the one route the same way whatever you submitted. Knobs that apply:
`reps`, `min_percent` (default 1.0), `residency` (`host` or `device`). `threads` does not apply, and
`counters: true` is a 503 `counters_unsupported` naming Nsight Compute as the tool that owns the
question and this route as not serving it -- PAPI counts HOST events, which say nothing about a
device kernel.

The judge attaches the tracer around the SAME measured child it times, on the SAME build, and
asks it for four reports. What it traces and what it deliberately does not are both decisions you
have to know to read the result:

- **`--trace=cuda,nvtx`** and nothing else. `osrt`, `cublas` and `cudnn` each add interception
  overhead to the run you are measuring. NVTX is traced because it is free, but do NOT bracket your
  kernel with `nvtxRangePush`/`nvtxRangePop` expecting to read the ranges back: no NVTX report is
  requested and no payload field carries one, so the ranges reach the recording and never reach
  you. What you would have added is instrumentation inside the code being graded, paid for and
  unread. Name your phases as separate KERNELS instead -- those the trace does rank, one row each
  (see `divide-and-conquer`).
- **`--sample=none --cpuctxsw=none`**. CPU sampling answers the host path's question, and these two
  are the parts of `nsys` that would drag `kernel.perf_event_paranoid` into a GPU profile -- IP
  samples need it at 2 or below and system-wide context switches need 0 or root, so a device
  measurement would fail for a host reason.
- **No `-g`, no `-G`.** Kernel names come from CUPTI, which reads them out of the fatbinary, so
  there is nothing for DWARF to add; and `-G` disables device optimization, which would profile a
  program nobody runs. The traced `.so` is byte-identical to the one the judge times.
- **A fresh export every time.** A recording summarised from a stale export is the one failure that
  breaks the profile -> change -> profile loop, because it feeds you the PREVIOUS run's numbers at
  exit 0 and a real speedup then reads as no change. The route re-exports; a hand-run does not by
  default, which is one more reason the number that counts comes back through `profile`.

Four reports the route does not ask for, so that you recognise the questions as out of reach rather
than spend turns hunting them: the host-side API summary (time inside `cudaMemcpy`,
`cudaLaunchKernel`, `cudaDeviceSynchronize` -- the gap's own accounting), the per-launch split into
API, queue and kernel time, the kernel summary WITH grid and block dims, and the NVTX summary --
which is why the paragraph above tells you not to bracket. What you have instead is `device_pct`
plus the launch geometry, and between them they answer the version of the question that changes
what you write.

## Pick the window before you divide

A span that contains compilation, allocation or first-touch context creation is not a measurement
window, and a busy percentage over it is a percentage of nothing. Any JIT framework (DaCe, Numba,
Triton, `torch.compile`) compiles INSIDE the traced span, after device activity has started. Field
test: a 17.55 s compile phase inside the device span put all-device-over-span at 0.04% against a
steady-state 6.01% -- 150x out, and 0.04% does not look broken, it looks like the verdict "the
device is idle, stop tuning kernels". Two checks, both before any division:

- **The untraced wall clock**, which the recording does not contain: the judge's `elapsed_ns` is
  it. A 28.75 ms device span inside a 0.40 s run means 93% of the wall is host-side setup that no
  kernel change reaches.
- **First and last `Start (ns)` in `cuda_gpu_trace`, and the gaps between rows.** A compile or an
  allocation phase is one gap orders of magnitude above the median. `cuda_api_sum` names it when
  there is one: a 104.7 ms `cudaMalloc` total over 3 calls whose MAXIMUM is 104.6 ms is one
  first-touch context creation, not three allocations.

If a phase is inside the span, the span is not the denominator: re-sum from the first activity after
it, or split the steady-state work into its own kernels and re-profile. The judge's `elapsed_ns`
is the measured rep rather than the process, which is why `device_pct` does not have this problem --
a division you do yourself does.

## `cuda_gpu_kern_sum` -- which kernel, and what to do about it

Per kernel: `instances` (launches), `total_ns`, `mean_ns`, `min_ns`, `max_ns`, `time_pct`.

**Rank by `total_ns`, then read `mean_ns` to decide what to do about the top row.** A 5 us kernel
launched 200,000 times owns more device time than a 50 ms kernel launched once, and `mean_ns` ranks
those the other way round; the payload's kernel list is sorted by `total_ns` for that reason. What
`mean_ns` tells you is HOW, not WHICH -- a big mean says the body, a small mean with a big
`instances` says the launch -- and it is the number to hold a body change against, because
`total_ns` moves with the rep count while the mean does not. Rank within one profile by the total;
compare two profiles by the mean.

`time_pct` is each kernel's share of the KERNELS LISTED, which is not device time: the copies sit
outside that denominator and are routinely larger than the kernels. Kernels below `min_percent` of
device time are dropped and COUNTED (`kernels_omitted`), so a short list is a short list, never a
truncated one.

The summary answers "fewer launches, a bigger grid, or a different algorithm" and they are three
different findings:

| what the rows show | the finding | the change |
| --- | --- | --- |
| one kernel at 80%+ of device time, `mean_ns` >= ~100 us | the kernel BODY is the cost | nothing here says why: `ncu` on that kernel |
| many `instances`, `mean_ns` under ~10 us | launch-bound: more time being told what to do than doing it | fuse, do more per launch, or capture a CUDA graph |
| `blocks` below the SM count (108 on A100, 132 on an H100 SXM5 but 114 on the PCIe card) | the grid does not fill the device | one element per thread, not one row; split the reduction |
| `mean_ns` flat as the input grows | fixed overhead, not the kernel | read the copies and the gap instead |
| `mean_ns` growing faster than the input | an algorithmic term | no geometry or launch change reaches it; change the algorithm |
| two or three kernels at ~30% each | no single hotspot | fusing them beats tuning any one of them |
| `max_ns` far above `mean_ns`, `min_ns` near it | one slow launch: JIT/module load, clock ramp, another tenant | check warmup covered it before believing the mean |

`device_pct` frames all of it: traced KERNEL time per rep against the measured host time per rep --
the copies are not inside it. Below ~50% the kernel is NOT what costs, and a faster kernel moves the
total by less than the number says.

**Quote the denominator with any ratio you compute yourself.** On one four-kernel trace, kernel time
over the device span read 16.5% ("the device idles between kernels"), kernel plus copy time over the
same span read 73.4% ("the device is saturated"), and kernel time over the untraced wall clock read
1.2% ("the kernel is a rounding error"). All three are correct arithmetic answering different
questions, 15x apart, straddling the threshold you were about to apply.

## The copies

`cuda_gpu_mem_time_sum` (how long) and `cuda_gpu_mem_size_sum` (how much) are separate reports,
joined here per operation into `memory[]`: `direction` (`h2d`, `d2h`, `d2d`, `memset`, `other`),
`count`, `total_ns`, `mean_ns`, `total` + `unit`.

The volume keeps nsys's OWN unit rather than being converted to bytes: releases disagree on whether
their `MB` is 10^6 or 2^20, and picking one invents a precision the recording does not have. Do the
bandwidth division yourself, carry the unit with it, and settle the ambiguity against a copy whose
size you know -- 2 MiB buffers reported as `2.097 MB` mean that build's `MB` is 10^6.

**`total_ns` in `memory[]` is summed over every rep the child ran -- warmup included.** Divide by
`reps + warmup` before you put it next to `elapsed_ns`, which is one rep. Comparing the two raw is
the commonest arithmetic error on this payload; `device_ns_per_rep` is already divided, the memory
rows are not.

**A bandwidth needs the link WIDTH, not just the generation**, and the width you want is the
board's maximum: an idle GPU downclocks its link, so a current-state reading says gen1 about a gen4
card. Where the link is not something you can read from here, treat the table below as the ceiling
you are checking a copy against rather than a measurement you have.

| link | per direction | a good copy lands near |
| --- | --- | --- |
| gen3 x8 | 7.88 GB/s | 6 |
| gen3 x16, gen4 x8 | 15.75 GB/s | 12-13 |
| gen4 x16, gen5 x8 | 31.5 GB/s | 25 |
| gen5 x16 | 63 GB/s | 50 |

GH200's NVLink-C2C is hundreds of GB/s and none of those rows apply. Reading a measured 13.1 GB/s
against an x16 row you assumed rather than queried turns 83% of wire into "42% of good" and earns a
source change worth nothing: on the box that measured it, pinning host memory moved H2D from 12.95
to 13.42 GB/s, about 4%.

What the numbers mean:

- **Transfer time near or above kernel time** -- the transfer IS the problem. No kernel change can
  reach it. Ask first whether the data changes between reps: if it does not, the copy is pure
  overhead and belongs outside the timed region entirely (that is what `residency: "device"` times).
- **Achieved bandwidth near the link's practical rate** (the table above) -- the copy is running as
  fast as the wire allows. The only remaining lever is moving LESS: keep buffers resident, transfer
  once and loop on the device, send fp32 where fp64 is not needed, or overlap with streams -- which
  HIDES the transfer behind compute but does not remove it.
- **Achieved bandwidth far below the link with large copies** -- pageable host memory, staged
  through the driver's bounce buffer. `cudaHostAlloc`/`cudaMallocHost` typically doubles it. That is
  the case pinning is for; near the ceiling it buys single-digit percent.
- **High `count`, tiny `mean_ns`** -- per-copy latency (a few microseconds each) dominates the
  volume. Batch them into one transfer of a packed buffer.
- **`memset` rows are work too.** A `cudaMemset` per rep is device time and device bandwidth; fold
  it into the kernel that was about to overwrite the buffer anyway.
- **`d2d` traffic you did not write** is usually a library staging a layout change.

## Gaps -- and idle is not the opposite of saturated

The gap is what the arithmetic leaves over:

```
gap_per_rep = elapsed_ns - device_ns_per_rep - (sum of memory total_ns) / (reps + warmup)
```

Where it goes:

- **Launch overhead** -- a few microseconds per launch, host and device side. Against
  `launch_count`, that is a bound you can check in one multiplication: 5000 launches at ~5 us is
  25 ms of nothing, and it will not shrink by making the kernel faster.
- **Synchronization stalls** -- a `cudaDeviceSynchronize` or a synchronous `cudaMemcpy` per rep
  turns an asynchronous pipeline into a round trip. `cuda_api_sum` names which call held the host.
- **Host-side work between launches** -- index math, allocation, a Python frame. The device is idle
  and no device-side change touches it.
- **Context creation** -- the first CUDA call costs 100 ms or more. It belongs in warmup; if it
  lands in a measured rep, the mean is fiction.

**Launch-bound is settled by the TOTALS, not by the gap size.** `cuda_api_sum`'s `cudaLaunchKernel`
total against the kernel total: measured on a 3351-launch trace, 7.20 ms of host time spent
launching 4.74 ms of device work, which is the textbook signature -- this skill's rule of thumb,
not a documented NVIDIA test, but it is what settles the verdict. Gap size does not show this and
usually points the other way -- on that same trace the median kernel-to-kernel gap was 640 ns
against a mean `cudaLaunchKernel` of 2149 ns, because the host runs far ahead and the queue hides
the launch cost from the device timeline. `cuda_kern_exec_sum` splits a launch into API, queue and
kernel time, but a deep queue is NOT the evidence: NVIDIA's help for that report says queue time
"is not inherently bad", it means "the GPU was busy running other tasks", and "if every kernel
launch is immediate, without any queue time, that _may_ indicate an idle GPU with poor
utilization". The column that signals is `QCount`, the launches that had any queue time at all: a
large `Count - QCount` alongside device-side gaps is the host starving the device. The fix is
fewer, bigger launches, or a CUDA graph -- nothing about the bodies matters until the count drops.

If you do capture a graph, `--cuda-graph-trace` defaults to `graph` on CUDA driver 11.7+: the graph
traces as ONE activity and its kernels leave `cuda_gpu_kern_sum` entirely. `--cuda-graph-trace=node`
shows them again, at the real per-node overhead.

**The distinction that matters most: `nsys` cannot tell you whether the GPU was SATURATED.** It
records that a kernel was RESIDENT. A kernel occupying 100% of the timeline while using 3% of the
SMs looks exactly like a kernel at peak -- same rows, same `device_pct`, same "the GPU is busy"
reading. Resident is not busy. The device-side utilization question is answered by counters
(`occupancy`, `device_utilization` below) or by `ncu`, never by the timeline.

Conversely a low `device_pct` IS conclusive, once the window is clean: the device really was idle
for that fraction, and the fix is on the host or in the copies.

## `cuda_gpu_trace` -- launch geometry bounds occupancy, it does not measure it

Distinct geometries, most-launched first: `grid` (blocks), `block`, `threads_per_block`,
`warps_per_block` (threads / 32), `blocks`, `registers_per_thread`, `shared_memory` + its unit.

Read it as a set of caps on how many blocks can be resident per SM:

- `blocks` below the SM count -- most of the device never gets work, whatever the kernel does.
- `registers_per_thread * threads_per_block` against the SM's 65536 registers -- 64 registers on a
  256-thread block is 16384, so at most 4 such blocks are resident, i.e. 32 of the 64 warp slots.
- `shared_memory` per block against the SM's shared-memory budget -- the same arithmetic, the other
  resource.
- `threads_per_block` not a multiple of 32 -- a 100-thread block is 4 warps with 28 lanes idle in
  the last one, on every block, on every SM.

**Achieved occupancy is not here and is not inferable from here.** It is a per-SM counter, read by
Nsight Compute and by nothing on this route. An occupancy number derived from geometry would be
indistinguishable from a measured one, so the payload ships the note instead of the number, and the
caps above are what you actually get to reason with.

## nsys or ncu -- what each one cannot answer

`nsys` answers **which kernel and when**: the ranked kernels, the launch count, the copies, the
gaps, the timeline. It CANNOT tell you why any of them is slow -- it records activity, not
counters, so there is no achieved occupancy, no stall reason, no memory throughput, no cache hit
rate anywhere in it. One traced run, small overhead.

`ncu` answers **why this kernel is slow**: stalls by reason, achieved occupancy, DRAM and L1/L2
throughput, divergence, register and spill pressure. It CANNOT tell you how often the kernel ran,
what ran around it, where the host waited, what the copies cost, or whether two kernels overlapped.
It REPLAYS each kernel many times and serialises them, so its wall clock is not your program's.

**`ncu` is not on this route**, and that is the more useful half of the comparison: the counter
questions -- stall reasons, achieved occupancy, cache hit rates -- are not available to you here, so
decide from `mean_ns`, the launch geometry and the copies, and say a counter question is unanswered
rather than guessing at it. If you ever do read an `ncu` number elsewhere, never quote its timing as
a speed: replay makes its numbers per-kernel counts, not durations comparable to anything.

## Device counters are not on this route either

`/profile` with `counters: true` on a `cuda` submission is refused (`counters_unsupported`): PAPI
counts HOST events, and host counts say nothing about a device kernel. PAPI does have device
components -- `cuda` (CUPTI) for kernel counters and `nvml` for device state -- and neither is
reachable through `/profile`, so neither is a measurement you can take here. Reaching around the
endpoint into the library is the same mistake as reaching around it into a profiler: the number
would describe a build and a driver process that nobody scored.

What that costs you, and what it does not. The kernel-counter half -- occupancy, DRAM traffic,
stall reasons, cache hit rates -- is genuinely unavailable, so name it as unmeasured rather than
inferring a figure from the geometry. The device-state half is a question about the MACHINE rather
than about your kernel (was the clock the same, did a later rep run hot), and the route answers the
version of it that matters: rerun and compare, because a result that does not reproduce is the
finding, whatever the clock was doing.

Three constraints are worth carrying anyway, because they invalidate device counts you read
elsewhere as surely as ones you took:

1. **Counter collection SERIALISES kernels and REPLAYS multi-pass metric sets.** A counted run's
   wall clock is not the plain run's. Read the counts, never the time -- and never put a counted
   run's milliseconds next to a timed run's. This is the same reason `ncu` timings are not speeds.
2. **CUPTI changed profiling APIs at Volta.** Pre-Volta parts answer through the event-group names
   (`achieved_occupancy`, `inst_executed`), Volta+ through PerfWorks (`sm__warps_active...`,
   `dram__bytes_read`). Different namespaces, so the event is resolved against what this install
   ENUMERATES rather than built from a template -- which is why a metric can be absent here and
   present on the next box.
3. **One event set counts ONE device through ONE context.** A second GPU needs a second event set,
   and work on another device or in another context is simply not counted -- which looks exactly
   like a kernel that did nothing.

And the rule behind all three: a quantity nobody could express comes back as a REASON, never as a
zero. On a GPU, a missing number and a zero counter are the two things a reader most reliably
confuses, and only one of them is a finding.

## The permission gate -- what an empty profile actually means

NVIDIA's driver can be configured to serve profiling to root only. When it is, CUPTI-based tools
refuse with **`ERR_NVGPUCTRPERM`** -- a message about administrators, from a library you never
named -- and PAPI's `cuda` component answers `PAPI_EMISC` at `PAPI_start`. What you SEE is an empty
profile, which is exactly what a fast kernel looks like.

The gate is on COUNTERS, so it does not fail everything equally: plain CUDA activity tracing (this
skill's four reports) usually survives it, while `ncu`, PAPI's `cuda` component and
`nsys --gpu-metrics-devices` (`--gpu-metrics-device` on older builds) do not. A run that gives you
kernel durations but refuses every counter is this gate, not a broken toolkit.

The harness classifies this as `insufficient_permissions` rather than as a failed trace. Recognise
it yourself by:

- `ERR_NVGPUCTRPERM` anywhere in stderr;
- `nsys`/`ncu` complaining about `CAP_SYS_ADMIN` or administrator privileges;
- a recording that exists but whose kernel summary is empty on a submission you know launches.

The gate is a driver module option and clearing it needs root, so it is the host's to fix and not
yours -- report it as "not measured" rather than spending turns around it. What is worth knowing is
how to recognise it, and that recognising it has a trap:

**The gate has two spellings.** The module option is `NVreg_RestrictProfilingToAdminUsers` and older
drivers echo it back, but the open kernel module publishes the INTERNAL name
`RmProfilingAdminOnly` instead. Matching only the documented one reports "no gate" on a gated box --
measured here on driver 595.84, which publishes `RmProfilingAdminOnly: 1` while every CUPTI tool on
it refuses.

## When the profiler says nothing

A profiler that reports nothing must never read as a fast kernel. The harness refuses with a named
cause instead of an empty profile, and each one has a different fix:

| cause | what it is | fix |
| --- | --- | --- |
| `nsys_missing` | Nsight Systems not on PATH | `nsight-systems-cli` from NVIDIA's CUDA repo; `nvidia-cuda-toolkit` lacks it |
| `no_gpu` | `/dev/nvidiactl` absent: no GPU here | `--gpus all` (docker), `--device nvidia.com/gpu=all` (podman), `--nv` |
| `insufficient_permissions` | the gate above | `NVreg_RestrictProfilingToAdminUsers=0`, or `--cap-add=CAP_SYS_ADMIN` |
| `nsys_failed` | no recording, another reason | read its stderr, which the error carries verbatim |
| `nsys_report_missing` | a recording, four empty reports | nsys older than 2022.1: upgrade, or use the old spellings |
| `no_kernels` | 0 GPU kernels traced | it ran on the host, the launch failed, or it forked: below |
| `counters_unsupported` | `counters: true` on a GPU submission | nothing to fix: counters are off-route, see above |
| `rocprof_unsupported` | a `hip` submission | nothing to fix: `nsys` cannot see an AMD queue, `rocprofv3` answers |

`no_kernels` has one cause that leaves no other trace: **`nsys` follows the whole process TREE but
NOT a bare `fork()` child.** Fork without exec is undefined behaviour for an injection-based tool,
so the child computes correctly and the timeline comes back EMPTY -- measured, same binary, same 20
launches: inline it reports 20 instances, fork first and the summary step answers `SKIPPED:
<name>.sqlite does not contain CUDA kernel data` while the child exits 0. There is a trace-the-fork
option and nsys's own help says it may crash or deadlock the app, so fix the FORK instead. `spawn`
and `exec` are both fine; for a Python workload in this repo,
`HPCAGENT_BENCH_RUNTIME_MP_CONTEXT=spawn`.

An empty kernel summary is a finding about how your submission starts its work, not about your
kernel.

## Traps

- **The trace covers warmup reps too.** `device_ns_per_rep` divides by `reps + warmup` for exactly
  that reason. Any number you divide yourself must use the same denominator.
- **Kernel names arrive demangled and long.** A C++ template kernel comes back as its full
  signature; the rendered text truncates at 44 characters, the JSON does not. Match on the JSON.
- **Tracing is not free**, only cheap. Compare a traced run against a traced run; take speedups from
  the graded measurement.
- **`nsys` traces the whole child process tree** -- except a fork without exec, above. A submission
  that spawns workers gets all of their device activity in one summary, which is what you want for
  totals and not what you want when attributing a kernel to a rank.
- **There is no `ncu` route, and no way to make one.** Everything past "which kernel" is
  unanswered here, so say so rather than substituting a number derived from geometry. What you have
  is the ranked kernels, the copies, the launch geometry and `device_pct`, and most kernels are
  decided by those before a counter would have added anything.

## Documentation

- Nsight Systems user guide, including the full CLI --
  https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Post-collection analysis: from 2025.5 the report definitions, the analysis rules and the SQLite
  schema live here rather than in the user guide --
  https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html
- The profiling permission gate --
  https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters
- Install, and the `perf_event_paranoid` levels --
  https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html
- Release notes: why fork-without-exec is not traceable --
  https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html
- SM counts per part (H100 SXM5 132, PCIe 114) --
  https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- A report's own `--help-reports` output is the authority on that report's columns
