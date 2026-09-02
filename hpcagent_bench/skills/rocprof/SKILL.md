---
name: rocprof
description: Read an AMD device profile as an MI300 finding rather than an NVIDIA one, and know which questions this route cannot answer.
---

The device half of `profiling`, on AMD. `perf` samples a host call stack; a HIP launch is
asynchronous, so a host profile of a HIP kernel shows the synchronization the host waited in and
nothing about the kernel. What the DEVICE did is recorded instead, one record per dispatch and per
copy.

## Ask through `profile`, and only through `profile`

`POST /profile` with `"language":"hip"`. The dispatch is the LANGUAGE, so it is the same route a C
or a CUDA submission asks; `tool` defaults to `rocprofv3` here, and naming `nsys` comes back as
`rocprof_unsupported` -- nsys traces CUDA and cannot see an AMD queue.

The judge attaches the tracer around the SAME measured child it times, on the SAME build. It traces
dispatches and memory copies and nothing else: no counters, no timeline. The build gets no extra
flags -- kernel names come out of the code object, and the device-debug switch would disable device
optimization, so the traced library is byte-identical to the one being graded.

That last property is the whole reason to use the route. A profiler you drive yourself measures a
binary you built, from a harness you wrote, on inputs you chose: three differences from the program
being scored, all in the direction that makes a profile agree with you. A finding that does not
come back through `profile` is a finding about a different program.

## What comes back, and what comes back `null`

Kernel rows: `name`, `instances`, `total_ns`, `mean_ns`, `min_ns`, `max_ns`, `time_pct`.
Memory rows: `operation`, `direction` (`h2d`/`d2h`/`d2d`/`memset`), `count`, `total_ns`, `mean_ns`,
`total`, `unit`.
Launch rows: `name`, `grid` (in BLOCKS), `block`, `threads_per_block`, `blocks`, `warps_per_block`,
`registers_per_thread` (per work-item), `shared_memory` (LDS bytes), `shared_memory_unit`,
`launches`.
Run totals: `device_ns`, `device_ns_per_rep`, `device_pct`, `launch_count`, `kernels_omitted`.

Those are built from what the tracer records. The dispatch trace carries `Workgroup_Size_{X,Y,Z}`
and `Grid_Size_{X,Y,Z}` (in WORK-ITEMS, which is why `grid` is converted to blocks for you),
`LDS_Block_Size`, `Scratch_Size`, `VGPR_Count`, `Accum_VGPR_Count` and `SGPR_Count`. The agent
report carries the PART: `Wave_Front_Size`, `Num_Xcc`, `Cu_Count`, `Simd_Count`,
`Max_Waves_Per_Simd`, `Lds_Size_In_Kb`, `Product_Name`. That second one is what makes every
occupancy sentence below arithmetic rather than folklore -- the geometry is MEASURED, not assumed.

`SGPR_Count` is not in the payload row: the scalar count is per wavefront and has no NVIDIA
counterpart, so it has no place in a vendor-independent schema. And the LDS column is
`LDS_Block_Size` on rocprofiler-sdk 1.1.0 and later, `Group_Segment_Size` before it.

Four fields come back `null` on AMD and never `0`, because a zero there would be a measurement:

- `shared_memory` -- the LDS column was renamed between rocprofiler-sdk generations, so it is
  absent when the trace carries neither spelling. When present it is rounded UP to the allocation
  granule, so it is an upper bound on what you spent.
- `total` / `unit` on a memory row -- the tracer TIMES copies and does not size them. A 2.4 ms
  transfer of 0 MB would be the lie; no volume is the truth.
- `min_ns` / `max_ns` -- absent when the host fell back to the deprecated v1 profiler, which has no
  per-kernel spread. The payload's `tool` field says which one ran.
- `warps_per_block` -- absent when the trace carried no agent report naming the wavefront width,
  because that width is read and never assumed. NVIDIA's warp is a constant 32; AMD's is not.

## Read the kernel summary

Rank by `total_ns` to find WHERE the time is; optimize against `mean_ns`. Total time is a launch-
count artifact -- change the rep count and it moves without anything getting faster.

`device_pct` is the first number to read, not the last. It is device time per rep over the host's
measured time per rep. Low, with a healthy-looking kernel table, means the device was IDLE and the
host is the bottleneck: launch gaps, a synchronous copy per rep, a `hipDeviceSynchronize` inside the
loop. No device-side change fixes that, and no amount of kernel tuning will move the score until it
is gone.

`kernels_omitted` counts what fell below `min_percent` (default 1%). A short table is a filtered
table, not a simple kernel.

## The three questions this route does not answer

Knowing these are out of reach is worth a turn; spending turns discovering it is not.

**A timeline.** There is none here -- no host sampling, no gap view, no system-wide correlation.
`device_pct` is the proxy and it is enough to act on: low means the gap is real, and the fixes are
host-side and few (fewer launches, no per-rep synchronize, no per-rep copy, one allocation outside
the loop). You do not need to see the gap to close it.

**Device counters** -- occupancy, cache hit rates, VALU utilization. `counters:true` is HOST PAPI
and is refused for a HIP submission (`counters_unsupported`), because CPU counts say nothing about
a device kernel; and the PAPI device components are not the way around that. The `rocm` component
is written against ROCProfiler V1, which AMD is retiring, and its SDK-based replacement `rocp_sdk`
exists only from PAPI 7.2.0, newer than a distribution build. So plan the work without them. Two
substitutes carry most of the weight: the launch geometry arrives WITH the trace, which makes the
occupancy arithmetic below available for free, and register/LDS pressure is a COMPILE-time answer
rather than a measured one -- see `opt-reports`.

Two properties of counting are worth carrying anyway, because they invalidate counter numbers you
meet in a paper or a report as surely as ones you measured. Collection SERIALISES dispatches and
replays multi-pass metric sets, so a counted run's wall clock belongs to no comparison -- read the
counts, never the time. And AMD's units are not NVIDIA's: fetch and write volumes come back in KB
against NVIDIA's bytes, power in uW against mW, temperature in millidegC against degC. The unit
ships with the count; never move a number across vendors without it.

**A roofline, or a per-kernel counter drill-down.** No route. Decide from `mean_ns`, the launch
geometry, and the memory rows against the bytes your kernel must move.

## The names changed twice, and the documentation did not

| tool | what it is | NVIDIA analogue |
| --- | --- | --- |
| `rocprofv3` | the counter and trace CLI behind this route, ROCm >= 6.2 | CUPTI, plus ncu's counter side |
| `rocprof` (v1), `rocprofv2` | the earlier CLIs -- different flags AND different output schemas | -- |
| `rocprof-sys` (was Omnitrace) | timeline and host sampling | Nsight Systems (`nsys`) |
| `rocprof-compute` (was Omniperf) | per-kernel counters, roofline, occupancy | Nsight Compute (`ncu`) |
| `rocm-smi` / `amd-smi` | board state: clocks, power, partition mode | `nvidia-smi` |

Anything written before ROCm 6.2 will tell you to trace with the superseded CLIs, and papers before
the 2024 rename call the other two Omnitrace and Omniperf. Read a recipe's version banner before
you believe its flags. Neither of the last two is on this route, and neither belongs inside a timed
run in any case -- both serialise the work they measure.

## MI300 is a chiplet part, and one of them is an APU

**Multiple XCDs.** MI300X (`gfx942`) is 8 accelerator complex dies, 304 CUs, one HBM3 pool, and
each XCD has its OWN L2. Workgroups are handed to XCDs in turn, so two workgroups that share a tile
very likely sit on different dies and each pull their own copy through the shared Infinity Cache
behind L2.

What that changes: inter-workgroup L2 reuse -- a real tactic on a monolithic GPU, where a second
block finds the first block's lines still in L2 -- is much weaker here. Keep reuse INSIDE the
workgroup, in LDS, and treat an L2 hit rate as a per-die number rather than a device-wide one. Tail
effects are chunkier too: with 8 XCDs a grid of 9 workgroups runs in two rounds and 7/8 of the
second round is idle silicon. Size grids in multiples of the XCD count times the workgroups you fit
per CU.

**Partition mode changes what "the GPU" means.** Compute partitioning (SPX, one agent per package,
versus CPX, one agent per XCD) and memory partitioning (NPS1/NPS4) are set out of band by the
operator. In CPX the card enumerates as EIGHT agents, so a run pinned to device 0 gets one eighth of
the part and its kernel times are not comparable with an SPX run's. You cannot set this and you
cannot see it from the payload -- which is the point: two AMD numbers from two machines are not
comparable unless somebody recorded the mode beside them.

**MI300A is an APU.** Zen4 cores and CUs share one package and ONE physical HBM pool, so "H2D
transfer" there does not mean what it means on a discrete card:

- a fast or absent H2D row on MI300A is not a fast link. There is no link. The copy is a copy within
  one memory, and it can be elided entirely.
- so the NVIDIA finding for a transfer-bound kernel -- pin the host memory, overlap the copy with a
  stream -- is the wrong fix. The APU fix is to NOT COPY: allocate once with a fine-grained or
  managed allocation, hand the same pointer to host and device, and check the copy left the report.
- the trap in the other direction: under unified memory, data movement becomes PAGE MIGRATION, and
  the memory-copy trace does not record migrations at all. Zero copy rows plus kernel time you
  cannot account for is migration, not a free lunch. On MI300A the migration may genuinely be a
  no-op; on a discrete MI300X it certainly is not.

## The same symptom, the other vendor

Half of what is written about GPU optimization is written about NVIDIA parts (the `nsys` skill is
this repo's NVIDIA half). Four quantities decide whether a finding ports:

| quantity | NVIDIA | AMD CDNA (MI300) | consequence |
| --- | --- | --- | --- |
| lane group | warp, 32, fixed | wavefront, 64 -- and RDNA parts are 32 | block sizes and divergence granularity both double |
| occupancy | warps/SM as % of peak | waves per CU: `Max_Waves_Per_Simd * Simd_Count / Cu_Count` (32 on MI300X) | different unit, not comparable as a number |
| on-chip scratch | shared memory, carved out of a unified L1/shared budget | LDS, 64 KB per CU, SEPARATE from the vector L1 | a bigger tile does not cost you L1 here |
| register count | one number in the launch record | a vector count per work-item, plus a separate scalar file | the vector count ports; a kernel can be scalar-register-bound here in a way NVIDIA has no analogue for |

Read as tiling decisions:

- a 32-thread block is a whole warp and HALF a wavefront: half the lanes idle on every issue. Block
  sizes start at 64 and go up in 64s. A 256-thread tile is 8 warps on NVIDIA, 4 wavefronts here.
- divergence granularity is 64. A branch that splits the data every 32 elements is free on NVIDIA
  (each warp takes one side) and costs BOTH sides on AMD. Regroup so a branch's granularity is at
  least a wavefront.
- a warp-level reduction written as five shuffle steps over 32 lanes is a six-step reduction over 64
  here. A hard-coded 32 in a shuffle sequence silently drops half the data.
- the occupancy question is "how many workgroups fit in 64 KB of LDS and in the register budget",
  not "how much did I take away from L1". A tile sized to fit NVIDIA's 48 KB default has room here
  -- and a tile that fits by 64 lanes may not.

## The refusals, and what each one means for you

Every one of these is "not measured". None is a fast kernel, and none is fixed by changing the
kernel.

| cause | what happened |
| --- | --- |
| `rocprof_unsupported` | `nsys` was named for a HIP submission -- ask the default instead |
| `not_linux` | ROCm is Linux-only; there is no AMD GPU to trace here |
| `rocprof_missing` | no profiler on the host at all |
| `no_amd_gpu` | no device node, or only a CPU agent was enumerated |
| `kfd_permission_denied` | `/dev/kfd` exists and this process may not open it |
| `rocminfo_missing` | the profiler is installed and the ROCm runtime is not -- `rocminfo` is what proves the runtime, and a profiler is not one |
| `rocprof_failed` | the tool exited non-zero; its own message is quoted back. If the quoted error names YOUR program and a missing `libhsa-amd-aqlprofile64.so.1`, the code is fine: that library is injected into the child, so a host missing it fails in your program's name |
| `rocprof_report_missing` | it exited 0 and wrote no kernel report |
| `no_kernels` | the trace contains zero dispatches -- the submission ran on the host, or a launch failed silently. Check the launch's error code, because this one IS yours |
| `counters_unsupported` | host counters were asked for on a device kernel; see the three questions above |

## The environment decides what got measured

The traced run inherits the environment unchanged, deliberately: pinning it would profile a
differently-configured run than the one being graded. You do not set these, and you cannot read
them back out of the payload -- which is exactly why each one is worth recognising by its effect,
because every one of them produces a report that looks like a finding.

- `ROCR_VISIBLE_DEVICES` filters at the ROCr runtime; `HIP_VISIBLE_DEVICES` (and
  `CUDA_VISIBLE_DEVICES`, which HIP also honours) filters at the HIP layer ON TOP of that result.
  They COMPOSE and re-index, so "device 1" in one layer is not device 1 in the other, and two
  profiles taken under different filters are two different parts.
- `HSA_ENABLE_SDMA=0` -- copies stop using the DMA engines and become blit KERNELS. They then
  appear in the kernel report and VANISH from the memory-copy report. An empty memory report next
  to a kernel nobody wrote is this, not a kernel that stopped copying.
- `HSA_XNACK=1` -- page-fault-driven unified memory. Copies become migrations, which nothing in
  this trace records.
- `HSA_OVERRIDE_GFX_VERSION` -- tells the runtime the part is a different ISA. The code that ran
  was compiled for something else; the numbers are real and they are not this part's.

**`/dev/kfd` is the permission gate as well as the presence check.** ROCm reaches the device
through the GROUP that owns that node, so a user outside `render`/`video` sees a device that
appears ABSENT rather than forbidden. This is AMD's analogue of NVIDIA's `ERR_NVGPUCTRPERM` and it
is NOT the same KIND of gate: dispatch tracing here needs device ACCESS, not `CAP_SYS_ADMIN`.
Carrying the NVIDIA fix across adds a capability, changes nothing, and ends with a conclusion that
the GPU is broken.

## Two rules that survive both vendors

1. **Absent is not zero.** A `null` transfer volume, a missing min/max and a `null` LDS size mean
   the tool never looked. A `0` means it looked and counted nothing. Only the second is a finding.
   The trap is specific here: because the LDS column was renamed, a reader pinned to one spelling
   reports the other's 16 KB workgroup as `0 B` -- a budget it says is free and you have already
   spent.
2. **A profiler that reported nothing is not a fast kernel.** Every refusal above has a named cause;
   treat it as "not measured" and say so, rather than optimizing against an empty profile. A blank
   report that reads as "0.00 ms on the device" is the one failure this whole route exists to
   prevent.
