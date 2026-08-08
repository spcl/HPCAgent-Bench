---
name: rocprofv3
description: Trace an AMD GPU run with rocprofv3 -- which kernel, which copy, which gap -- rank by total_ns not mean_ns, and know when only rocprof-compute can answer.
---

The device half of a profile, on AMD. `perf` samples a host call stack; a HIP launch is
ASYNCHRONOUS, so a host profile of a HIP kernel shows the synchronisation the host waited in and
nothing about the kernel. What the DEVICE did is recorded instead, one record per dispatch and per
copy.

This is the AMD counterpart of `nsys`. It answers WHICH kernel and WHICH copy. It does not answer
why a kernel is slow -- that is `rocprof-compute`.

## What was measured here, and what was not

The trace below WAS executed here: Radeon 780M (**gfx1103**, RDNA3 integrated), ROCm 7.2.4,
rocprofiler-sdk 1.1.0, against a real HIP fixture. Every CSV column named below was read back off
that run. What was NOT verified here is anything CDNA-specific -- an iGPU has no HBM and no
Infinity Fabric, and the MI300 counter expressions are a different architecture's -- so treat the
tool MECHANICS as measured and the MI300 numbers as documentation.

Everything NOT measured here is quoted from upstream with its URL, in place. A claim on this page
with no run behind it and no link behind it would be indistinguishable from an invented one, so
there are none: what could not be sourced was deleted rather than hedged.

WARNING: `rocprofv3` needs `hsa-amd-aqlprofile` and does not pull it in. Without it the run dies
with `error while loading shared libraries: libhsa-amd-aqlprofile64.so.1` -- prefixed with **YOUR
program's name**, not the profiler's, because the library is injected into the child. The binary
links and runs fine standalone, so this reads as a bug in your code and is not one. `apt install
hsa-amd-aqlprofile`.

The COUNTER half could not be exercised here at all -- see "Counters" below for why, which is a
measured result rather than a gap. Everything this page says about `--pmc` SEMANTICS (pass
splitting, the budget, cross-pass ratios) is read from the rocprofv3 source and docs, not run.

The READING RULE in "rank by the right column" is not vendor folklore -- it was measured on the
NVIDIA twin of this page, where the fixture's launch-bound kernel owns **67.3%** of device time by
total and ranks **DEAD LAST** by mean. That arithmetic is vendor-independent.

## The name changed twice

| you may see | current name | what it is |
| --- | --- | --- |
| `rocprof`, `rocprofv2` | `rocprofv3` | THIS page: dispatch trace, `--pmc` counters |
| `omniperf` | `rocprof-compute` | kernel-level analysis, SOL, roofline |
| `omnitrace` | `rocprof-sys` | whole-application CPU+GPU timeline |

Upstream keeps the first row itself -- rocprofiler-sdk publishes a rocprof/rocprofv2/rocprofv3
option-by-option comparison
(https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/conceptual/comparing-with-legacy-tools.html)
-- and the second is in the successor's own title, "ROCm Compute Profiler (formerly Omniperf)"
(https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/). Older tuning guides use the
left column throughout. A host with only the deprecated v1 takes a DIFFERENT command line and
produces a different schema -- see the bottom of this page.

## How it runs

```sh
rocprofv3 --kernel-trace --memory-copy-trace --stats --output-format csv \
          --output-directory prof --output-file run -- ./your_app <args>
```

`--` separates the tool's flags from the application's -- every upstream invocation is written
`rocprofv3 <options> -- <application_path>`
(https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html). It
matters: without it the first application argument is parsed as a tool flag.

## The four reports

They answer different questions:

| report | file | what it answers |
| --- | --- | --- |
| kernel stats | `*_kernel_stats.csv` | per kernel: `Name`, `Calls`, `TotalDurationNs`, `AverageNs`, `Percentage`, `MinNs`, `MaxNs`, `StdDev` |
| memory copy stats | `*_memory_copy_stats.csv` | per operation: how long H2D / D2H took. **No byte volume in CSV** -- see below |
| kernel trace | `*_kernel_trace.csv` | per dispatch: `Workgroup_Size_{X,Y,Z}`, `Grid_Size_{X,Y,Z}` (in WORK-ITEMS), `LDS_Block_Size`, `Scratch_Size`, **`VGPR_Count`**, `Accum_VGPR_Count`, **`SGPR_Count`**, `Start_Timestamp`, `End_Timestamp` |
| agent info | `*_agent_info.csv` | the PART: `Wave_Front_Size`, `Num_Xcc`, `Cu_Count`, `Simd_Count`, `Max_Waves_Per_Simd`, `Max_Waves_Per_Cu`, `Lds_Size_In_Kb` |
| domain stats | `*_domain_stats.csv` | per API/dispatch DOMAIN totals -- the top-level split before you rank within one |

Those column lists are the writer's own: the eight-name stats header and the kernel-trace header
are the literal arguments to the CSV files' constructors, and the `_kernel_stats` / `_memory_copy_stats`
/ `_domain_stats` suffixes come from the domain table next to them
(https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/lib/output/generateCSV.cpp,
https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/lib/output/domain_type.cpp). The
agent columns are in the tool docs, which print the header verbatim
(https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html).

Find them RECURSIVELY. Measured on rocprofiler-sdk 1.1.0 the layout is FLAT --
`<dir>/<prefix>_kernel_stats.csv` and friends, no subdirectories -- but the default output path is
`%hostname%/%pid%` when `--output-directory` is not given
(https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/bin/rocprofv3.py), and a glob that
assumes one layout silently finds nothing on the other.

**`LDS_Block_Size` is an UPPER BOUND, not the request.** The writer rounds the dispatch's
`group_segment_size` up to the LDS allocation granule --
`(group_segment_size + (lds_block_size - 1)) & ~(lds_block_size - 1)` (generateCSV.cpp, above) --
and rocprof-compute says the same of its own `LDS Allocation`: "This may also be larger than what
was requested at compile time due to both allocation granularity and dynamic per-dispatch LDS
allocations"
(https://github.com/ROCm/rocprofiler-compute/blob/develop/src/rocprof_compute_soc/analysis_configs/gfx942/0700_wavefront.yaml).
The column was named `Group_Segment_Size` on ROCm 6.2, and carried the RAW value with no rounding
(https://github.com/ROCm/rocprofiler-sdk/blob/docs/6.2.0/source/lib/rocprofiler-sdk-tool/generateCSV.cpp),
so a reader that matches only one of the two names reports a 16 KB workgroup as 0 B on whichever
generation it was not pinned to -- a budget reported free and then spent twice.

The REGISTER COUNTS are the reason to read the kernel trace even when you already have the stats:
`VGPR_Count` and `SGPR_Count` are what turn "occupancy is low" into a cause, and they are per
dispatch rather than per kernel.

**Read `*_agent_info.csv` first.** It is the part's geometry, measured, and it is what makes every
occupancy sentence arithmetic instead of folklore -- `Max_Waves_Per_Cu` is the denominator, so you
never have to guess it. `Grid_Size_*` is in WORK-ITEMS -- "The total number of work-items (or,
threads) launched as a part of the kernel dispatch. In HIP, this is equivalent to the total grid
size multiplied by the total workgroup (or, block) size"
(https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html) -- so
divide by `Workgroup_Size_*` to get the block count, or every occupancy number you derive is wrong
by the block size.

## Rank by the right column

`TotalDurationNs`, not `AverageNs`. The kernel worth working on is the one that owns the most
device time in aggregate, and the two columns disagree exactly when it matters: a trivial kernel
launched thousands of times can own most of the run while ranking last by mean. On the NVIDIA
fixture built for this, the 64-launches-per-rep kernel owns 67.3% of device time and has the
smallest mean of the four. Sorting that table by mean picks the wrong kernel with a real number.

`Percentage` is that ranking already done for you -- but it is each row's duration over the
DOMAIN's total, not over the wall clock (generateCSV.cpp, above), so it answers "which kernel" and
never "was the GPU busy". Use it, then check `Calls`: a high percentage with a high call count is a
LAUNCH problem (batch, fuse, or use a graph), and a high percentage with a low call count is a
KERNEL problem (go to `rocprof-compute`).

## Was the device busy at all?

Sum `TotalDurationNs` across kernels and divide by the wall clock of the same run. This is the
first number to compute and the one that decides whether any of the rest matters.

- **Device percentage low** -- the GPU is idle most of the run. The finding is on the HOST: launch
  gaps, synchronous copies, a `hipDeviceSynchronize` in the timestep loop, or work that never got
  offloaded. No kernel-level tool will help; fix the gaps first.
- **Device percentage high, one kernel dominant** -- go to `rocprof-compute` for that kernel.
- **Device percentage high, time spread evenly** -- an algorithmic or fusion question, not a
  per-kernel one.

**Exclude one-time setup from the wall clock before you divide.** On the NVIDIA twin this exact
recipe read **0.04% against a truth of 6.01%** -- a 150x error -- because a JIT compile sat inside
the span being divided by. AMD has the same hazard in a different place: the first dispatch of a
code object pays a load, and `hipMalloc` of a large buffer is not free. Time the STEADY-STATE
reps, not the process.

## Copies carry no byte volume in the CSV

Measured on rocprofiler-sdk 1.1.0: `*_memory_copy_trace.csv` has exactly these columns --

```
Kind, Direction, Stream_Id, Source_Agent_Id, Destination_Agent_Id,
Correlation_Id, Start_Timestamp, End_Timestamp
```

-- and that is the whole header the CSV writer is constructed with, eight names with no size among
them (https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/lib/output/generateCSV.cpp).

**The RECORD has the number; only the CSV emitter drops it.** The buffer-tracing record declares
`uint64_t bytes; ///< bytes copied`
(https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/include/rocprofiler-sdk/buffer_tracing.h),
and the other emitters write it: JSON serialises the field by name
(https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/include/rocprofiler-sdk/cxx/serialization/save.hpp),
Perfetto attaches it to the copy slice as `copy_bytes`
(https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/lib/output/generatePerfetto.cpp),
and the rocpd database stores it as the copy's `size`
(https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/lib/output/generateRocpd.cpp).

So ask for both. `--output-format` takes a LIST (`csv`, `json`, `pftrace`, `otf2`, `rocpd`)
(https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/bin/rocprofv3.py), and one run
writes both files:

```sh
rocprofv3 --kernel-trace --memory-copy-trace --stats --output-format csv json \
          --output-directory prof --output-file run -- ./your_app <args>
```

Read the ranking off the CSV and `bytes` off the JSON, joined on `correlation_id`. The achieved
rate is then `bytes` over that record's own `end_timestamp - start_timestamp` -- no guessing from
the source, and no unit to convert, because the field is bytes.

Then compare against the link: a PCIe-attached part and an Infinity-Fabric-attached one differ by
an order of magnitude, and an integrated GPU has neither -- it shares the host memory controller,
so a "copy" there is not the same operation at all.

The actionable findings are almost always structural rather than rate-related: a copy inside the
timestep loop that could be hoisted, a H2D of data the device already had, or pageable host memory
where pinned would let the copy overlap.

## Counters, when the trace has done its job

`--pmc` collects hardware counters per dispatch. It is the raw form of what `rocprof-compute`
packages, and it is the right tool when you want ONE number rather than a whole analysis.

**FIRST check that your part HAS counters, because the failure mode is a crash, not a refusal.**
Measured on gfx1103 (RDNA3 integrated, ROCm 7.2.4, rocprofiler-sdk 1.1.0), every `--pmc` run ends:

```
rocprofiler_iterate_agent_supported_counters failed for agent 1 (gfx1103)
  :: Agent HW architecture is not supported, no counter metrics found.
terminate called after throwing an instance of 'std::out_of_range'
  what():  unordered_map::at
[rocprofv3_error_signal_handler] rocprofv3 caught signal 6
```

The unsupported-agent line is a WARNING and the run continues, so the tool aborts on the empty
counter map several seconds later. It then hangs in `queue.cpp` ("Timeout while waiting for queue
sync: 1 kernels still active") for a further 10s+ before finalizing. So the observable is a SIGABRT
and a hang in a program that runs clean without the profiler -- the same trap as the missing
aqlprofile library above, and it will read as your kernel faulting.

**`HSA_OVERRIDE_GFX_VERSION` does not rescue this**, and it is the first thing to reach for
because it is the standard escape hatch for an unsupported target. Measured: with
`HSA_OVERRIDE_GFX_VERSION=11.0.0` exported and the kernel compiled `--offload-arch=gfx1100`, the
application itself runs clean, and `--pmc` produces the SAME abort -- the warning still names
**gfx1103**. The override is a ROCr/HIP-layer lie about the ISA; rocprofiler reads the real hardware
ID when it enumerates counters, so the two never meet. A missing counter set on your part is not a
configuration you can talk your way out of.

Three consequences. Run every `--pmc` invocation under `timeout -k`, not a bare `timeout`: measured,
the SIGTERM at the deadline is caught by rocprofv3's own signal handler, logged as "caught signal
15", and the process keeps running -- it needs a SIGKILL to die. Never leave one unattended in a
wrapper that assumes `timeout` terminates things. And treat counter support as a per-ARCHITECTURE
question: the trace side of this page works on the same part where the counter side aborts, so
"rocprofv3 works here" says nothing about whether `--pmc` does. Consumer and integrated RDNA parts
are the ones to check first; the CDNA datacenter parts these counter names are documented for are
where the support is.

```sh
rocprofv3 --pmc SQ_WAVES GRBM_GUI_ACTIVE TCC_HIT_sum TCC_MISS_sum -- ./your_app
```

Results land in one directory per pass, and the file inside is PID-prefixed: counter collection
"generates a `./pmc_n/counter_collection.csv` file prefixed with the process ID. For each `pmc`
row, a directory `pmc_n` containing a `counter_collection.csv` file is generated, where n = 1 for
the first row and so on"
(https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html). So glob
`pmc_*/*_counter_collection.csv` rather than naming it.

**The counter budget is hardware, and exceeding it FAILS THE JOB -- it does not replay.** From
rocprofv3's own `--pmc` help: *"job will fail if entire set of counters cannot be collected in
single pass"*
(https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/bin/rocprofv3.py), repeated in the
docs as "Job fails if the entire set of counters can't be collected in a single pass" (same tool
page as above). This is the opposite of `ncu`, which quietly replays the kernel until it has every
metric, and the opposite of rocprof v1. So an over-long counter list costs you the run rather than
the wall-clock, and the remedy is yours to apply: split the list across passes yourself, using the
input file below. Never grow a `--pmc` list hoping the tool will cope.

**Repeating `--pmc` does NOT give you two passes -- it silently DISCARDS the first.** The option is
declared `nargs="*"` with no `append` action, and the launcher then joins the ONE survivor into a
single row, `"pmc: {}".format(" ".join(args.pmc))` (rocprofv3.py, above). Nothing warns. Multi-pass
comes from an INPUT FILE: "For multi-pass execution, include multiple `pmc` rows in the input file.
Counters in each `pmc` row can be collected in each application run" (tool page, above).

```
pmc: SQ_WAVES SQ_BUSY_CU_CYCLES
pmc: TCC_HIT_sum TCC_MISS_sum
```

```sh
rocprofv3 -i counters.txt -- ./your_app
```

Which means the same rule as every other counter instrument: **two counters from two different
passes came from two different executions of your kernel.** A ratio across passes is only
legitimate through a denominator both passes measured (`GRBM_GUI_ACTIVE` is the usual one), and it
is only meaningful at all if the application is deterministic.

Name a counter without a dimension specifier and rocprofv3 aggregates for you: "Specify the counter
name without dimension specifiers (e.g., `pmc: TCC_MISS`). The `rocprofv3` tool will automatically
collect accumulated values across all instances", and per-instance values need "JSON output format,
which includes detailed dimension information for individual counter instances" (tool page, above).
That is the AMD equivalent of the `:stat=sum` problem on NVIDIA, resolved in the opposite
direction: here the aggregate is the default and the breakdown is the thing you ask for.

## The deprecated v1, if that is all the host has

```sh
rocprof --stats --timestamp on -o prof/run.csv ./your_app
```

No `--`: the documented synopsis is `rocprof [-h] ... [-o <output CSV file>] <app command line>`,
with the workload following the options directly, `--stats` writes one `<output name>.stats.csv`,
and `--timestamp <on|off>` is what puts `dispatch/begin/end/complete` on each row
(https://github.com/ROCm/rocprofiler/blob/amd-master/doc/rocprof_tool.md).

It DOES print launch geometry, so that column survives the fallback even though the schema does
not. The dispatch line is `grd(%u), wgr(%u), lds(%u), scr(%u), arch_vgpr(%u), accum_vgpr(%u),
sgpr(%u), wave_size(%u)`
(https://github.com/ROCm/rocprofiler/blob/amd-master/test/tool/tool.cpp), and its `lds` is rounded
up to the same LDS granule v3's `LDS_Block_Size` is.

Two more v1 facts before you compare anything to a v3 run, both from the tool doc above: its text
input file is read "automatically rerun application for every pmc line", so a pass there is a whole
extra execution of your program, and "profiling has limitation of serializing submitted kernels".
Check which binary you actually ran before concluding the data is broken.

## Traps

- **`--` before the application.** Missing it turns your app's first argument into a tool flag.
- **Find the CSVs recursively.** Flat, or under the default `%hostname%/%pid%`.
- **`Grid_Size_*` is WORK-ITEMS.** Divide by workgroup size for blocks.
- **A traced run's wall clock is not a timed run's.** Take every speed-up from an uninstrumented
  build.
- **`--pmc` SERIALIZES dispatches; the trace does not.** "Counter collection in *dispatch counting*
  mode requires serialized execution of kernels on a target device", and for co-dependent kernels
  that must run simultaneously "kernel serialization leads to deadlock"
  (https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/counter_collection_services.html).
  A hang under `--pmc` on an application that runs clean is this, not your kernel.
- **The build gets no extra flags.** Kernel names come from the code object, and the device-debug
  switch would disable device optimisation -- so the traced binary is the one you timed.
- **Which device is measured.** `ROCR_VISIBLE_DEVICES` is "a list of device indices or UUIDs that
  will be exposed to applications"
  (https://rocm.docs.amd.com/en/docs-7.2.4/conceptual/gpu-isolation.html), so `device 0` in the
  report is the first EXPOSED one and not necessarily the one you think. Check `*_agent_info.csv`
  against the part you meant.
- **Verify the answer.** A kernel that got faster and wrong measures nothing.

## Documentation

- Application tracing and profiling with rocprofv3 -- https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- ROCprofiler-SDK -- https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/
- Counter collection services, for the serialization rule -- https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/counter_collection_services.html
- rocprof / rocprofv2 / rocprofv3, option by option -- https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/conceptual/comparing-with-legacy-tools.html
- The CSV writer -- every column name and the LDS rounding -- https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/lib/output/generateCSV.cpp
- The report file names, per domain -- https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/lib/output/domain_type.cpp
- The memory-copy record, where `bytes` lives -- https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/include/rocprofiler-sdk/buffer_tracing.h
- The JSON serialiser that emits it -- https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/include/rocprofiler-sdk/cxx/serialization/save.hpp
- The rocprofv3 launcher, for `--pmc` and `--output-format` -- https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/bin/rocprofv3.py
- The deprecated v1's own doc and dispatch line -- https://github.com/ROCm/rocprofiler/blob/amd-master/doc/rocprof_tool.md and https://github.com/ROCm/rocprofiler/blob/amd-master/test/tool/tool.cpp
- MI300/MI200 counters, for the `--pmc` names -- https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300-mi200-performance-counters.html
- AMD's profiling walkthrough -- https://rocm.blogs.amd.com/software-tools-optimization/profiling-guide/novice/README.html
- ROCm Compute Profiler, where a slow kernel goes next -- https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/
- HIP programming model: wavefront, CU, LDS -- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
