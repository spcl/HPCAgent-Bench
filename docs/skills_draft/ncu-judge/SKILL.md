---
name: ncu-judge
description: What the SMs did inside ONE CUDA kernel -- the judge has NO ncu route and refuses by name; what it gives you instead, and how to target the launch yourself.
---

`ncu` REPLAYS. To collect a large metric set it runs the SAME launch many times, saving and
restoring the memory the kernel writes between passes, with the GPU clocks pinned and the caches
flushed. The `Duration` it reports is a device measurement of a replayed, clock-pinned, cold-cache
launch, and NVIDIA documents that host timers and CUDA events cannot give you a workload duration
under `ncu` at all. **Never quote an `ncu` duration as a time and never put one next to a timed
run.** What it gives you is COUNTS -- what the SMs did inside one launch, which is the one question
a tracer cannot answer.

Trace first (`nsys`): it names
the kernel and the launch count, and `ncu` on the wrong kernel is a perfectly analysed 4% of the run.

## How it runs

**There is no judge route to SM counters, and this is the page that says so plainly.** `ncu`
replays one launch many times with the clocks pinned and the caches flushed; nothing in the judge's
measurement path does that, and asking for counters on a device submission is refused BY NAME:

```sh
curl -s -X POST "$JUDGE_URL/profile" -H 'Content-Type: application/json' \
  -d '{"kernel":"<kernel>","language":"cuda","rank":<judge rank>,"counters":true,
       "source":"<your full source>"}'
# -> HTTP 503  {"cause": "counters_unsupported", ...}   -- the refusal names this tool
```

The judge URL, the kernel name, your language and your rank are the ones your task statement
gave you -- substitute them; this page cannot know them.

The one thing the judge WILL do feeds an `ncu` run you do yourself.

**Trace it.** The same `/profile` call WITHOUT `counters` runs Nsight Systems and returns which
kernel owns device time and how many times it launched. That name is the `-k` for the command
below, and it is the step that stops you analysing a perfectly measured 4% of the run.

That trace is ALSO the only instrument the judge will attach to a `cuda` submission. `linuxperf`,
`papi` and `none` each come back 400 naming `nsys`, because a device kernel has no host-side
bracket for them to run in -- so there is no judge route that builds your instrumented source and
hands back its stdout. One rule governs the source you DO send:

- **Only `-I`, `-D`, `-l` and `-L` survive from `build`.** `-O3`, `-march=`, `-fopenmp` and
  `-ffast-math` are dropped -- the judge's own matrix supplies those. Single-token forms only, so
  `-I /path` as two tokens loses the path, and `-l:libfoo.so` or any `-l` containing `/` is
  rejected as an injection form.

Everything finer than the trace you take on your own box. Bracket each launch with CUDA events and
you learn WHICH launch is the odd one -- the cold first, the one whose convergence differs -- so
`-s` lands on a steady-state launch instead of the one that happened to be first.

Those milliseconds are a TIME and the counters below are not: the per-launch rows come from an
ordinary run, while every number the rest of this page teaches comes from a replayed, clock-pinned,
cold-cache launch. Use the timings to choose the launch and to check that a change moved the clock;
use `ncu` to find out why. Never put the two in one table.

Nothing on `/profile` is scored -- no `speedup`, no `native_ns`, and the scorer is never called.
Submit the CLEAN source to `/submit`: events and syncs are work inside the timed region, so a
scored run of instrumented code is a slower run of the wrong program.

## Is it installed

Three documented locations, and `which` alone under-reports -- the NVIDIA HPC SDK ships its own
copy, and a CUDA Toolkit `.run` install (the usual cluster case) puts it under `/usr/local/cuda-*`:

```sh
which ncu nv-nsight-cu-cli                                  # PATH
ls -d /usr/local/cuda*/nsight-compute*/ncu                  # CUDA Toolkit .run install
ls -d /opt/nvidia/nsight-compute/*/ncu                      # .deb / .rpm install
find /opt/nvidia/hpc_sdk -maxdepth 6 -name ncu              # SDK-bundled
ncu --version
```

Defaults change between releases, so read `ncu --help` on the binary you will actually invoke.

Measured on this dev box (RTX 4050 Laptop, AD107, 20 SMs, driver 595.84): `ncu` IS on PATH at
`/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/compilers/bin/ncu`, version 2025.4.1.0, and a newer
standalone sits at `/opt/nvidia/nsight-compute/2026.2.1/ncu`, version 2026.2.1.0.

Counter collection is driver-gated: `grep -E 'RmProfilingAdminOnly|RestrictProfilingToAdminUsers'
/proc/driver/nvidia/params` -- `0` is open, `1` needs root plus a driver reload, and both spellings
name the same setting. This box reads `0`, and every number below was collected through it.

Command shapes come from `ncu --help` on these binaries, and every metric name, report row LABEL
and **numeric threshold** below comes from this install's own `<install>/sections/*.section` and
`*.py` -- NVIDIA's shipped rules, grep-able at the paths named below. The THRESHOLDS are identical
across the two installed versions; the rules are not. `HighPipeUtilization.py` and `Tile.py` differ
substantively, the other 22 by copyright year alone. The entire manual set also ships offline as
`<install>/docs/{ProfilingGuide,NsightComputeCli,NsightCompute,ReleaseNotes,CustomizationGuide}/index.html`
-- the same text served at the URLs at the end of this page, so no claim here needs a network fetch
to re-check. What a real kernel READS against those thresholds was collected here.

## Target ONE kernel

A 1052-launch run profiled whole is hours of replay for one answer. Narrow first, always:

```sh
ncu -k regex:jacobi -c 1 -s 20 --set basic -o prof -f -- ./app input
```

- **`-k` / `--kernel-name`** takes a bare name for an exact match or `regex:<expr>`. It matches on
  the `function` basis by default -- "function name without parameters, templates etc.", so BOTH the
  parameter list and the template arguments are stripped, and `regex:mykernel<float>` matches
  nothing. Anchor on the bare name. `--kernel-name-base demangled|mangled` switches.
  **`regex:` is UNANCHORED -- it is a substring match, and a sibling whose name merely CONTAINS
  yours is matched too.** Measured: `-k regex:k_bank -c 3` profiled `k_bankpad` as its second
  result, and only the report header said so; `-c`/`-s` count across the union of the matches, so a
  skip meant for one kernel steps through another. `-k k_bank` (bare name = exact) and
  `-k 'regex:^k_bank$'` both hit only the intended kernel.
- **`-c` / `--launch-count`** caps how many matching launches are profiled. Almost always `1`.
  `--filter-mode` (default `global`, else `per-gpu` / `per-launch-config`) decides whether `-c`/`-s`
  count collectively or per device / per shape.
- **`-s` / `--launch-skip`** skips matching launches first -- use it to step past warmup and JIT, so
  you profile a steady-state launch instead of the cold one. (`--launch-skip-before-match` counts
  ALL launches, not just matching ones; that is the other flag and it is rarely what you want.)
- **`--kernel-id ctx:stream:[name-operator:]name:invocation`** when one kernel name is launched on
  several streams with different shapes. The optional operator field takes `regex:`, so
  `--kernel-id :7:regex:^foo:` is "any kernel in stream 7 starting with foo".
- **`-o` / `--export`** writes a `.ncu-rep` you can re-read offline without re-running. `-f` to
  overwrite. A run that fails to collect writes NO file at all, whatever `-o` said.

## Sets and sections -- the cost knob

`--set` picks a bundle, `--section` picks one. Cost is REPLAY PASSES, and passes are NOT the metric
count: `ncu` groups all metrics requested for a launch into as few passes as the hardware counters
allow, so a set listing thousands of metrics is tens of passes, not thousands. The `--list-sets`
column is headed "Estimated Metrics" -- read it as relative cost only. Its numbers vary per
architecture AND per `ncu` version, so run `--list-sets` on the binary you will use rather than
porting a number. Measured here on 2026.2.1 / AD107:

| set | Estimated Metrics | sections you get | when |
| --- | --- | --- | --- |
| `basic` (default) | 213 | LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution | first look, always |
| `detailed` | 1071 | + ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, MWA_Chart, SourceCounters, SpeedOfLight_RooflineChart, Tile | after `basic` names a direction |
| `roofline` | 5919 | SpeedOfLight, WorkloadDistribution + five roofline charts | rarely; see below |
| `full` | 7381 | everything above + SchedulerStats, WarpStateStats, MWA_Tables, InstructionStats, NumaAffinity, PmSampling, Nvlink_Tables, Nvlink_Topology | last resort, one launch only |
| `nvlink` | 122 | Nvlink, Nvlink_Tables, Nvlink_Topology | multi-GPU link traffic, nothing else |
| `pmsampling` | 554 | PmSampling, PmSampling_WarpStates | a time series, not per-launch counts |

`full` reads 8051 on 2025.4.1 against 7381 on 2026.2.1 for the same chip: a version artefact, not a
workload fact. `C2CLink` is in NO set and must be named explicitly. `Tile` (2025.4+) analyses
CUDA-tile / cuTile workloads and is dead weight on an ordinary kernel -- it rides along in
`detailed` regardless. All the roofline sections moved INTO `full` at 2025.1, so `--set full` also
pays for five roofline charts you did not ask for.

**Four sections the decision path needs are `full`-only, and two of them cost more than their own
rows.** `MemoryWorkloadAnalysis_Tables` carries every bytes-per-sector row, every per-operation
L1/L2 hit rate, and the `MemoryCacheAccessPattern` / `SharedMemoryConflicts` / `LocalMemoryUsage`
rules -- all absent from `--set detailed`. `SchedulerStats` carries `IssueSlotUtilization`, which is
the PARENT of both occupancy rules, so without it they still run but with empty parent weights and
their estimates degrade from GLOBAL to LOCAL. Ask by name rather than paying for `full`:

```sh
ncu -k regex:jacobi -c 1 \
    --section SpeedOfLight --section LaunchStats --section Occupancy \
    --section SchedulerStats --section WarpStateStats \
    --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables \
    -- ./app input
```

**A `--section` selection collects that section's METRICS but not its RULES: a rule runs only if its
PARENTS were collected too, and the whole tree roots in SpeedOfLight.** The shipped chain is
`SOLBottleneck -> Memory -> {SharedMemoryConflicts, UncoalescedAccess}`, so both coalescing rules
need SpeedOfLight AND MemoryWorkloadAnalysis above them. Measured on a kernel with 96.87% of its
shared-load wavefronts in conflicts: `--section MemoryWorkloadAnalysis --section
MemoryWorkloadAnalysis_Tables --section SourceCounters` fired **ZERO** rules -- every table, no
verdict. Adding `--section SpeedOfLight` fired `Est. Speedup: 95.35% -- ... on average a 32.0 - way
bank conflict across all 1048576 shared load requests ... 96.88% of the overall 33559407 wavefronts
for shared loads`. **Put SpeedOfLight in every named-section run.** A childless section prints
numbers and no diagnosis, which is the failure that looks most like a clean kernel.

`ncu --list-sections` prints the identifiers `--section` takes. Asking for a few sections beats
`--set full` every time, as long as you ask for the parents too. `--metrics a,b,c` is cheapest of
all, and if the selection fits in ONE pass `ncu` skips the save-and-restore entirely.
`ncu --list-metrics` lists the metric NAMES the current section selection would collect -- names
only, not a cost or a pass count.

## Reading the text output

**`--page details` prints section HEADERS ONLY.** `--print-details` defaults to `header`, and every
chart and table in a section BODY -- the Memory Throughput Breakdown above all, plus the Memory
Chart, the Memory Tables, the occupancy curves and the Warps Per Scheduler chart -- is a body item
and is simply not printed. Measured here on `--set basic`: the default output is 81 lines with ZERO
breakdown rows; `--print-details all` is 376 lines and the breakdown appears. Every procedure below
that names a TABLE needs the flag:

```sh
ncu -i prof.ncu-rep --page details --print-details all
```

`--page raw` does not rescue it -- "No unresolved `regex:`, `group:`, or `breakdown:` metrics are
included". The constituents are still there under their full names, so a raw dump lets you rebuild
the breakdown by hand: scan `*__throughput.avg.pct_of_peak_sustained_*` and apply the prefix map in
the next section.

Four more facts, all from `<install>/docs/NsightComputeCli/index.html`:

- **Details keys on LABELS, raw keys on metric IDs.** Details: "If the metric was given a label in
  the section, it is used instead." The row you read as `Memory Throughput` is
  `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` on the raw page. **Diff on raw,
  read on details.** `--print-metric-name {label|name|label-name}` overrides it.
- **Values auto-scale, which is poison for a diff**: "Both metric unit and value are automatically
  adjusted to the most fitting order of magnitude", so 1.66 ms and 1660 us are the same row on two
  runs. `--print-units base` pins them and `--csv` implies it; `--print-fp` forces floating point.
- **Instanced metrics print semicolon-joined inside one cell** (`240; 240; 240; 240`).
  `--print-metric-instances` (`none` default / `values` / `details`) changes the row SHAPE, so two
  dumps taken with different settings do not line up.
- **`--print-rule-details` prints the rule's focus-metric table and "currently has no effect in CSV
  mode"** -- rule KPIs are unreachable from any CSV pipeline.

## Which number is relative to what

Half the wrong conclusions come from treating an absolute count as a percentage or a
peak-relative percentage as an absolute. Sort them before reading anything:

**Already normalised -- a percentage OF A HARDWARE PEAK, no ceiling needed.** Everything ending
`.pct_of_peak_sustained_elapsed` or `.pct_of_peak_sustained_active`; `Achieved Occupancy` and
`Theoretical Occupancy`, which NVIDIA defines as "the ratio of the number of active warps per
multiprocessor to the maximum number of possible active warps". A **Throughput** metric is
additionally a maximum, not an average: NVIDIA states "throughput metrics return the maximum
percentage value of their constituent counters".

**Absolute -- meaningless until you quote its ceiling, and the ceiling is always a different row.**
`Issued Warp Per Scheduler` is warps per active cycle against 1.0. `Warp Cycles Per Issued
Instruction` is cycles and has no ceiling -- it IS the denominator for every stall reason, and every
`..._per_issue_active.ratio` stall is cycles measured against it. `Avg. Active Threads Per Warp` is
against 32. The five `Block Limit *` rows are BLOCKS per SM measured against each other, smallest
binding. `Waves Per SM` is waves, with 1.0 the floor below which the grid cannot fill the device.
`Average Bytes Per Sector For Global Loads` is bytes against its own `Maximum Bytes Per Sector` row.

**The one that catches people: `Memory Throughput` is not DRAM throughput, and the label is used
TWICE.** In SpeedOfLight it is `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`, a
PERCENT and a maximum over the hierarchy, sitting beside three separate unit rows: `DRAM
Throughput`, `L1/TEX Cache Throughput`, `L2 Cache Throughput`. In MemoryWorkloadAnalysis the row
with the SAME label is `dram__bytes.sum.per_second` -- an absolute Gbyte/s. Grep that label across a
details dump and you mix a percentage with a bandwidth; key on the unit column or on the section
header above the row. Measured here on a bank-conflicting shared-memory kernel: `Memory Throughput`
96.49%, `DRAM Throughput` 1.05%, `L2 Cache Throughput` 0.43%, `L1/TEX Cache Throughput` 98.40%. On
that kernel every change that cuts DRAM bytes buys exactly nothing.

**The three unit rows are on DIFFERENT bases, so ranking them against each other is invalid.**
`L1/TEX Cache Throughput` is `l1tex__throughput.avg.pct_of_peak_sustained_ACTIVE`; `L2 Cache
Throughput` and `DRAM Throughput` are `pct_of_peak_sustained_ELAPSED`. An `_active` percentage
counts only cycles that unit was busy, so it inflates exactly when L1TEX idles for part of the
launch. Measured here on the uncoalesced kernel: the header reads `L1/TEX Cache Throughput` 64.63%
while the same unit's largest ELAPSED constituent in the breakdown is 33.30% -- and the headline
`Memory Throughput` is 56.57%, LOWER than the L1 row above it, which cannot happen if a maximum and
its constituent share a basis. Take L1 on the elapsed basis from `MemoryWorkloadAnalysis_Tables`
("L1TEX Throughput") or ask for `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` by name.

**So use NVIDIA's own discriminator instead of ranking the rows.** `SpeedOfLight.py` takes the
highest-valued constituent of the Memory Throughput Breakdown and maps its metric-name prefix:
`{dram, fbp, fbpa} -> DRAM`, `l1tex -> L1`, `{lts, ltc} -> L2`. It does this only when memory SOL
exceeds compute SOL; when compute wins, no unit is named at all. Rows whose prefix is not in that
map (the `GPU: ... Internal Activity` pseudo-rows, which are often on top) are skipped, so read down
until a prefix matches. That is the entire mechanism behind the rule sentence "Start by analyzing L1
in the Memory Workload Analysis section" -- and the breakdown it reads is a BODY table, so you need
`--print-details all` to see what the rule saw.

**The three MemoryWorkloadAnalysis header percentages answer three different questions.** Its
shipped `Description:` states them: memory limits a kernel "when fully utilizing the involved
hardware units (Mem Busy), exhausting the available communication bandwidth between those units (Max
Bandwidth), or by reaching the maximum throughput of issuing memory instructions (Mem Pipes Busy)".

| row | metric | saturating means | you cut |
| --- | --- | --- | --- |
| Mem Busy | `gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed` | a memory UNIT (an L1TEX/L2/DRAM data path) is busy | accesses |
| Max Bandwidth | `gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed` | an INTERCONNECT between units is saturated | bytes moved between levels |
| Mem Pipes Busy | `sm__memory_throughput.avg.pct_of_peak_sustained_elapsed` | the SM cannot ISSUE memory instructions faster | instruction COUNT: fewer, wider accesses |

`Mem Pipes Busy` is an `sm__` metric -- an SM-side issue-rate limit, not a memory-system one -- so
more bandwidth never fixes it.

## ncu FLUSHES the caches, so a cache-resident kernel reads as DRAM-bound

`--cache-control` defaults to `all`, which invalidates L1 and L2 before EVERY replay pass. The
point is reproducibility -- pass 3 must see what pass 1 saw -- and the cost is that the kernel is
measured cold, which is not how it runs.

That is invisible until the working set fits in cache, and then it dominates the headline number.
Same kernel, same binary, 6 MB of buffers against this part's 24 MB of L2:

| `--cache-control` | `dram__bytes_read` | `DRAM Throughput` |
| --- | --- | --- |
| `all` (the DEFAULT) | 4.20 MB | **90.04%** |
| `none` | 2.05 MB | **0.14%** |

A 640x swing on THAT kernel in the number that decides whether you are memory-bound, from a flag
nobody sets. Scale the same kernel to a 96 MB working set and the two agree (94.53% against
94.33%), because then the data genuinely does not fit and the flush changes nothing.

**The artefact is guaranteed in the BYTES and only sometimes visible in the percentage.**
`DRAM Throughput` is bytes over TIME, and dropping the flush shortens the launch as well as feeding
it, so the two can move together and cancel. A second 6 MiB kernel measured here: `all` reads
`dram__bytes_read` 4.20 MB / DRAM Throughput 89.18% / Duration 24.64 us, `none` reads 2.56 KB /
76.91% / 9.38 us -- the byte count moves **1640x** while the percentage moves 1.16x, because the
duration collapsed 2.6x with it. On the kernel in the table the percentage did collapse. Read the
byte counters; treat the percentage as the reading that may or may not survive the flag.

So: **a high `DRAM Throughput` on a kernel whose working set fits in L2 is not evidence of DRAM
traffic** -- check `dram__bytes_read` under `none` before you believe it. It is the common shape in
a timestep loop, where the same arrays are revisited every step and are hot by the second
iteration. Re-run with `--cache-control none` before you spend a day cutting DRAM traffic that the
real run never moves.

**`--cache-control none` is only valid on a SINGLE-PASS collection.** NVIDIA: valid "if only a
single kernel replay pass is necessary", otherwise it "can lead to inconsistent and out-of-bounds
metric values" -- because passes 2..N then see whatever pass 1 left in cache. `--set basic` is 8
passes on this box, so do NOT pair it with `none`. Source the uncached reading from an explicit
one-pass `--metrics` run (the table above was collected that way) and print `Duration` beside it so
a replay count that grew is visible.

This also reconciles ncu against an in-situ counter. PAPI's cuda component does not touch the
caches, so on that same 6 MB kernel it reported near-zero DRAM traffic while ncu reported 90% of
peak. Neither is broken. They answer different questions -- cold-start cost against steady-state
cost -- and which one you want depends on whether your kernel is called once or a thousand times.

## Read it in this order

NVIDIA ships its own ordering and it is not in prose: each rule in `<install>/sections/*.py`
declares `get_parent_rules_identifiers()`, and that parent chain is a tree rooted at the Speed Of
Light bottleneck rule. `grep -A1 get_parent_rules_identifiers <install>/sections/*.py` prints it.
Each step RULES OUT the ones it does not branch into:

1. **`Compute (SM) Throughput` and `Memory Throughput`** (SpeedOfLight). Either >= 80: you are
   resource-bound and steps 2-5 cannot help. Both < 60: latency, and 2-5 are the whole job.
2. **`Waves Per SM`** (LaunchStats), only if step 1 said latency. Below 1.0 the grid cannot fill the
   device at ANY occupancy. This reading **kills step 4 outright**: it is a grid-size finding, and
   occupancy work on a kernel without one full wave of blocks cannot pay.
3. **`Issued Warp Per Scheduler`** (SchedulerStats), ceiling 1.0, idle below 0.6. Then
   `Active Warps Per Scheduler` / `Theoretical Warps Per Scheduler` decides the rest -- and **the
   denominator is a trap**. The rule divides by `smsp__maximum_warps_avg_per_active_cycle`, which is
   LAUNCH-dependent and is labelled `Theoretical Warps Per Scheduler` in the **Occupancy** section
   (and in SchedulerStats' chart BODY). The similarly named SchedulerStats row `Theoretical Active
   Warps Per Scheduler` is `smsp__warps_active.avg.peak_sustained` -- the HARDWARE maximum, constant
   for the chip -- and the rule's own prose quotes THAT while branching on the other. Measured here
   on a 1024-thread kernel: active 7.86, hardware max 12, theoretical 8; the printed message says
   "Out of the maximum of 12 warps per scheduler". 7.86/12 = 0.65 sends you to the stall table;
   7.86/8 = 0.98 is what the rule actually used, and it named load imbalance. Fetch
   `smsp__maximum_warps_avg_per_active_cycle` explicitly. Three branches, from
   `IssueSlotUtilization.py`:
   - **`Active Warps Per Scheduler` < 1.0 -- the rule STOPS.** The scheduler is "limited to less
     than a warp per instruction" and no eligible-warp analysis runs at all. Fix the launch.
   - **ratio < 0.8** -- warps are allocated but not ELIGIBLE: they are stalled, go to step 5.
   - **ratio >= 0.8** -- the launch already has nearly every warp it is entitled to, so occupancy is
     not the gap either and NVIDIA's rule names load imbalance first, stalls only after.

   Occupancy (step 4) is what you reach for when the ISSUE rate is fine and the warp count is not.
4. **Occupancy**, only if step 3 sent you. `Theoretical` (a static property of the launch) before
   the gap to `Achieved` (a measured one); they fail for different reasons and take different fixes.
5. **WarpStateStats**, last. A stall reason means nothing until step 3 has shown issue slots are
   actually being lost.

## The reading -> action table

Thresholds are NVIDIA's own, read out of the shipped rules on this box: `SpeedOfLight.py`
(80 / 60 / 10), `TheoreticalOccupancy.py` (80), `AchievedOccupancy.py` (10),
`IssueSlotUtilization.py` (0.6 / 0.8, plus the 1.0-warp stop), `CPIStall.py` (0.8 / 0.3),
`ThreadDivergence.py` (24, against EITHER `Avg. Active Threads Per Warp` or `Avg. Not Predicated Off
Threads Per Warp`), `SharedMemoryConflicts.py` (10, evaluated SEPARATELY for loads and stores),
`LocalMemoryUsage.py` (10), `SlowPipeLimiter.py` (80 / 20 / 25), `HighPipeUtilization.py`
(20 / 60 / 80 on the busiest pipe's active-cycles percent, then `inst_util / active_util` < 0.3 =
"high-latency instructions", > 0.7 = "frequent, low-latency instructions"), `WorkloadImbalance.py`
(5% minimum speedup, per unit: SM, SMSP, L1 slice, L2 slice, DRAM slice), `LaunchStatistics.py`
(20). They are where NVIDIA's rule text fires, not laws.

Three rules do not work that way. **`UncoalescedGlobalAccess`, `UncoalescedSharedAccess` and
`MemoryCacheAccessPattern` have NO threshold** -- the first two return early only when the excess is
zero and the third sets `threshold_speedup_percent = 0`, so they fire on ANY excess and the message
appearing means nothing. Read the percentage they report. **`LocalMemoryUsage` declares no parent
rule**, so its advice appears even on a kernel the SOL rule called compute-bound. And the top-level
SOL rule **never emits an `Est. Speedup` of its own**; its verdict is one of six NAMED titles --
`Small Grid`, `Latency Issue`, `High Compute Throughput`, `High Memory Throughput`, `Balanced
Throughput`, `High Throughput` -- so parse the name, not the prose. A seventh, `NVLink-Centric
Scheduling`, warns that on such a launch a LOW SOL number is an artefact of SMs being unavailable.

| you read | it means | you change |
| --- | --- | --- |
| both throughputs < 60% AND `Waves Per SM` < 1 | the grid cannot fill the device; nothing in the kernel is the limit | more blocks: widen the grid. Do NOT tune occupancy |
| both throughputs < 60%, `Waves Per SM` >= 1 | latency-bound: no resource is near peak | steps 3-5; the body and the traffic are both fine |
| the two within 10 points of each other, neither < 60% | balanced -- cutting one side alone moves nothing | cut BOTH work and traffic; fusion is the single change that does both |
| `Memory Throughput` >= 80% | bandwidth-bound at whichever unit the Breakdown names | cut TRAFFIC, not the loop body: fuse, tile for reuse, recompute, narrower dtype |
| `Compute (SM) Throughput` >= 80% | compute-bound | the only levers left are less work and narrower types -- fp32 over fp64, intrinsics, tensor path |
| `Compute (SM)` >= 80%, average pipe utilisation < 20%, max-minus-avg > 25 points | one slow pipe holds the SM busy while the rest idle | move math off it: fp64 -> fp32 or int |
| `Issued Warp Per Scheduler` < 0.6 AND `Active Warps Per Scheduler` < 1.0 | fewer than one warp per scheduler is resident: NVIDIA's rule stops here and analyses nothing further | the launch shape -- block size, grid size, the occupancy limiters. Reading stalls is premature |
| `Issued Warp Per Scheduler` < 0.6 AND active/theoretical < 0.8 | warps are allocated but not eligible -- they are stalled | the stall table below |
| `Issued Warp Per Scheduler` < 0.6 AND active/theoretical >= 0.8 | nearly every warp occupancy allows is resident, so occupancy is not the gap | load imbalance first; stalls only after |
| `Achieved Occupancy` HIGH and both throughputs LOW | occupancy was never the problem | go to the stall reasons. Block size is not banned -- NVIDIA names it first when a LIMITER binds (the rows below) -- but changing it to raise an already-high occupancy is motion, not progress |
| `Theoretical Occupancy` < 80%, smallest limiter `Block Limit Registers` | register count caps resident blocks | `__launch_bounds__`, `-maxrregcount`, fewer live values |
| ... smallest is `Block Limit Shared Mem` | shared memory caps resident blocks | smaller tile, or `cudaFuncAttributePreferredSharedMemoryCarveout` |
| ... smallest is `Block Limit Warps` | BLOCK SIZE caps it, and it binds from both ends: too large strands warps, too small wastes block slots | resize, then re-read the limiter |
| ... smallest is `Block Limit SM` | the hardware blocks-per-SM ceiling, nothing you allocated | only MORE warps per block moves it |
| ... smallest is `Block Limit Barriers` | too many barriers per block (CC 90+) | fewer `__syncthreads()`. The rule CANNOT name this limiter -- it is absent from its enum, so the text will blame something else. Read the row |
| `Theoretical - Achieved` > 10 points | the launch could fill the SM and did not: scheduling overhead, tail, imbalance | even work per block, hunt an early `return`. **Re-read `Waves Per SM` first**: inside `1 <= Waves Per SM < 5` NVIDIA attributes the gap to the TAIL (its rule prices a partial wave at `1/(1 + whole waves)` and fires at 20%, which is where the 5 comes from), and the fix is more, smaller waves -- not load balancing |
| `Avg. Active Threads Per Warp` < 24 (of 32) | divergence or early thread completion | fix the BRANCH, not the occupancy. Source Counters names the lines |
| `Average Bytes Per Sector For Global Loads` far below its `Maximum` row (MWA_Tables, `full`-only) | uncoalesced: consecutive threads touch scattered addresses | transpose the layout, or stage via shared. Metrics under "Coalescing" below |
| shared bank conflicts >= 10% of shared wavefronts | shared-memory bank conflicts, reported separately for loads and stores | the rule also prints an **N-way** figure (`wavefronts/requests`) -- that is what picks the padding, the 10% is only the firing gate |
| `L1TEX Hit Rate` / `L2 Hit Rate` low where you expected reuse | the working set exceeds that level | smaller tile, different loop order, block the loop |
| local-memory instructions > 10% of instructions executed | register spill, or a dynamically indexed array in local scope | fewer live values, or index that array statically. `derived__local_spilling_requests` (2025.1+) reads spills directly instead of inferring them |

## Est. Speedup is closed-form arithmetic, and its TYPE is load-bearing

Two different quantities share the prefix. NVIDIA: "Global estimates (`Est. Speedup`) are an
approximation of the decrease in workload runtime, whereas local estimates (`Est. Local Speedup`)
are an approximation of the increase in efficiency of the hardware utilization of the particular
performance problem the rule addresses." **An `Est. Local Speedup: 43.83%` is not 43.83% off your
runtime** and must never be reported as one.

None of these is a measurement. All 13 formulas are plain arithmetic on metrics already in the
report, readable in `<install>/sections/*.py`: `Small Grid` is `(sm_count - grid_size)/sm_count`,
`Tail Effect` is `1/(1 + whole_waves)`, `ThreadDivergence` is
`(1 - min(active_thr, pred_on_thr)/32) * compute_sol`, and `IssueSlotUtilization` is
`min(1 - issue_active, 1 - max(sm_sol, mem_sol)/100)` and is **always LOCAL**. Three consequences:

- **They are not additive.** Several rules multiply by the same SOL weight, so summing across rules
  double-counts.
- **A GLOBAL child of `IssueSlotUtilization` is `min(parent_local, own_local)` relabelled** -- an
  upper bound with a soft derivation. `CPIStall`, `TheoreticalOccupancy` and `AchievedOccupancy` all
  sit there.
- **On `--set basic` the occupancy rules have no parent at all** (their parent lives in `full`-only
  SchedulerStats), so they run with empty parent weights and their estimates degrade to LOCAL. Same
  numbers, weaker claim.

To rank KERNELS instead of rules, `property__estimated_speedup` ("Maximal relative speedup
achievable for this profile result as estimated by the guided analysis rules") is a Summary-page
column: `--print-summary per-kernel`.

## What occupancy is worth, in NVIDIA's own words

The shipped `Occupancy.section` description is the whole of the in-tool guidance: "Higher occupancy
does not always result in higher performance, however, low occupancy always reduces the ability to
hide latencies". The Best Practices Guide is blunter -- **"improving occupancy from 66 percent to
100 percent generally does not translate to a similar increase in performance"**, because "a lower
occupancy kernel will have more registers available per thread ... with a high degree of exposed
instruction-level parallelism (ILP) it is, in some cases, possible to fully cover latency with a low
occupancy" (BPG 11.3). The Programming Guide quantifies the floor: on CC 7.x most arithmetic
instructions take 4 cycles, so **16 active warps per SM** cover arithmetic latency, and "fewer warps
are needed" if the warps carry ILP (PG 5.2.3 -- the restructured live guide has dropped this text,
so cite the 12.4.1 archive).

**Measure it rather than argue about it, without touching the kernel** (BPG 11.4): raise the
dynamic-shared-memory argument at launch. That lowers occupancy and changes nothing else. If the
runtime does not move, occupancy is not your lever and step 4 can be skipped for this kernel.

Two corrections to the limiter reading:

- **The `TheoreticalOccupancy` rule cannot name a barrier limiter.** It maps four limiters to
  English phrases ("the number of required registers", "the required amount of shared memory", "the
  number of blocks that can fit on the SM", "the number of warps within each block") and
  `launch__occupancy_limit_barriers` is absent from its enum. On Hopper+ a barrier-bound launch
  shows a low `Block Limit Barriers` in the header while the rule text blames something else. Read
  the five rows yourself -- smallest binds, ties are named jointly.
- **`Waves Per SM` is not a pure grid property.** "The size of a Wave scales with the number of
  available SMs of a GPU, but also with the occupancy of the kernel", so raising occupancy makes a
  wave BIGGER and lowers Waves Per SM. Step 2 still outranks step 4 as a priority call, but the
  causal claim is softer than it looks: a register-starved kernel can read below one wave partly
  BECAUSE its occupancy is low.

## Coalescing: the metric everyone quotes is in no section, and still answers

`l1tex__average_t_sectors_per_request*` appears in ZERO sections, rules or guides on either install
-- but it is a valid metric NAME, and `--metrics` resolves it in one pass. Measured:
`l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` read **32.00** on the
uncoalesced kernel and **4.00** on its coalesced twin. So it is the one-metric spot check, and it is
not what any rule computed. What the live rules actually read:

- **Bytes per sector** (MWA_Tables, `MemoryCacheAccessPattern`):
  `smsp__sass_average_data_bytes_per_sector_mem_{global,local}_op_{ld,st}.ratio` against the
  `.max_rate` submetric of the SAME metric -- that submetric is the "Maximum Bytes Per Sector For
  ..." row. Up to FOUR messages, one per (global|local) x (load|store), each titled with the cache
  level where the inefficiency costs most (`L1TEX Global Load Access Pattern`, `DRAM Global Store
  Access Pattern`, ...). The L2 and DRAM variants add "This applies to the {X}% of sectors missed in
  L1TEX/L2" -- a low L1 hit rate is what promotes a coalescing problem from an L1 nuisance to a DRAM
  cost. This row is a GLOBAL-access row and answers nothing else: measured on a 32-way
  shared-memory bank conflict and its padded twin, both read Average = Maximum = 32 byte/sector.
- **Excessive sectors and wavefronts** (SourceCounters, `UncoalescedGlobalAccess` /
  `UncoalescedSharedAccess`): `memory_l2_theoretical_sectors_global` against its `..._ideal`,
  exposed as `derived__memory_l2_theoretical_sectors_global_excessive`; the shared analogue is
  `derived__memory_l1_wavefronts_shared_excessive`. Per-line markers read "{P}% of this line's
  global accesses are excessive" and exist on the SOURCE page only.

Both fire on any excess whatsoever, so the number to report is the percentage, never the message.

**One-pass cross-check with a hard floor, no section needed** -- sectors per request, either form:

```sh
ncu -k jacobi -c 1 --metrics \
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
    -- ./app input                                       # the ratio, already divided

ncu -k jacobi -c 1 --metrics \
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    -- ./app input                                       # its two constituents, absolute
```

The pair is the same number with the counts visible -- 16,777,216 sectors over 524,288 requests on
the uncoalesced kernel here, 2,097,152 over the same 524,288 on the coalesced one -- and sectors are
the unit the shipped rules work in, so quote the pair when you report a finding and the ratio when
you are only checking.

A warp-wide 32-bit load needs 128 bytes, and a sector is 32 B, so the floor is **4**; 8 for
64-bit, 16 for 128-bit. **32 means fully scattered** -- one sector per lane. NVIDIA's own worked
example went 32 -> 4 by making `threadIdx.x` the fastest-varying subscript, for an 87.5% cut in
transactions and 68% in duration. Two metrics, one pass, no save-and-restore: cheap enough to run
before committing to `--set full`.

## The stall table

`--section WarpStateStats`. Every reason is spelled
`smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` and is **in cycles, not
percent**. Its share is that value divided by `Warp Cycles Per Issued Instruction` =
`smsp__average_warp_latency_per_inst_issued.ratio` -- renamed in 2025.4+, older material and most
blog posts say `smsp__average_warps_active_per_issue_active.ratio`. The section carries a second,
nearly identically labelled row, `Warp Cycles Per Executed Instruction` =
`smsp__average_warps_active_per_inst_executed.ratio`; dividing by that one is wrong. NVIDIA's own
rule acts when `Issued Warp Per Scheduler` < 0.8 AND that share exceeds 0.3, so a reason with a
large absolute cycle count and a small share is not your finding. Rule titles are mechanical --
`reason.replace("_", " ").title() + " Stalls"`, hence `Mio Throttle Stalls` and `Lg Throttle
Stalls` -- so match them case-insensitively.

**There are 19 rows on this chip, not 11**, and one of them is not a stall at all.

**A shared-memory bank conflict does not surface on the row that names it.** Measured on a 32-way
conflict: `mio_throttle` TOP at 71.0%, whose action below is "wider shared loads" -- the wrong
change -- and `short_scoreboard` second at 24.4%. The conflict rule named it exactly (`32.0 - way
bank conflict`, 96.88% of shared-load wavefronts, `Est. Speedup: 95.35%`) and costs MWA_Tables plus
its SpeedOfLight parent. When either row fires with shared memory in play, run that rule before you
act on the stall row.

| high share of | it means | you change |
| --- | --- | --- |
| `long_scoreboard` | waiting on an L1TEX dependency: global, local, surface, texture | coalescing, then more bytes in flight (wider loads, unroll), then shared-memory staging |
| `short_scoreboard` | an MIO dependency, not L1TEX: usually shared memory, sometimes MUFU or dynamic branches | kill bank conflicts; keep hot values in registers |
| `mio_throttle` | the MIO instruction queue is FULL: shared ops, special math and dynamic branches share it | fewer but WIDER shared loads; cheaper transcendentals |
| `lg_throttle` | the L1 queue for local/global ops is full: LG instructions issued extremely often | fewer, wider global accesses; check for local-memory spills |
| `barrier` | warps waiting at `__syncthreads()` for siblings | balance work BEFORE the barrier, or use fewer. At >= 512 threads NVIDIA suggests splitting the block |
| `math_pipe_throttle` | one math pipeline is oversubscribed; genuinely compute-bound | rebalance the instruction mix across pipes, or more active warps to hide it |
| `wait` | a fixed-latency dependency chain | ILP: independent work between dependent instructions; fast-math. Tops the list only in already-optimised kernels |
| `no_instruction` | i-cache miss, or a grid with less than one full wave | unroll LESS, shrink the loop body -- and re-read `Waves Per SM`, which is the other cause |
| `drain` | after EXIT, waiting for stores to land | the kernel writes a lot at the very end; coalesce those stores or reduce in parallel |
| `branch_resolving` | waiting for a branch TARGET to be computed and the warp PC updated: divergence, indirect branches, jump tables | fewer jump/branch operations, coalesce conditionals. Read it next to `Avg. Active Threads Per Warp`; NVIDIA explicitly pairs it with `no_instruction` |
| `imc_miss` | constant-cache miss; lanes reading DIFFERENT constant addresses serialise | make the warp read one constant address, or move the data out of constant memory. **CC <= 9.0 only** (`Filter { MaxArch: CC_90 }`) -- on Blackwell the same event surfaces as `short_scoreboard`, so a high `short_scoreboard` there is no longer automatically a shared-memory finding |
| `tex_throttle` | the L1 queue for TEXTURE/surface ops is full | fewer texture fetches and surface accesses; "texture can accept four threads' requests per cycle, whereas global accepts 32 threads", so converting a texture lookup to a global load can be the fix |
| `membar` | waiting on a memory barrier | drop unnecessary fences; make the outstanding memory operations that the barrier waits on optimal first |
| `sleeping` | every thread in the warp is blocked, yielded or asleep | fewer `NANOSLEEP`s, shorter delays, and group threads so a whole warp sleeps together |
| `dispatch_stall` | the warp HAS an instruction ready and the dispatcher held it back on a conflict | not a latency finding; look at the instruction mix and the pipes |
| `misc` | "a miscellaneous hardware reason" -- NVIDIA's whole description | nothing actionable; if it tops the list, your finding is elsewhere |
| `not_selected` | eligible warps queued behind another | nothing is wrong: you have MORE occupancy than you need. NVIDIA suggests REDUCING active warps for locality |
| `selected` (NOT a stall) | the warp WAS picked and issued that cycle -- the label is `Selected`, not `Stall Selected`, and `CPIStall` has no entry for it | issue is working; the inversion of a stall. If this dominates, stop reading this table |

`warpgroup_arrive` is Hopper-only and spelled three ways: metric
`smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio` (`MinArch/MaxArch: CC_90`), report
label `Stall GMMA`, rule title `Warpgroup Arrive Stalls`.

Raising occupancy fixes `long_scoreboard`, `wait` or `math_pipe_throttle` only when another warp
could then issue -- which is what step 3 established before you got here.

To get from any of these to a LINE of code: build with `-lineinfo`, add
`--section SourceCounters --import-source yes`, then read the report with
`--page source --print-source cuda,sass` (values `sass` default, `ptx`, `cuda`, `cuda,sass`;
correlation exists only in `sass` and `cuda,sass`). Two things about that page:

- **2025.4 flipped warp sampling to the `_not_issued` variants** by default, "to avoid pointing to
  source locations where warp stalls are mitigated by having sufficient numbers of warps during an
  issue cycle to hide latency". Only the `smsp__pcsamp_*` sampling metrics have a `_not_issued`
  form -- the per-issue-active aggregates in the table above do not. So **a line that is hot in the
  plain sampling and cold in `_not_issued` is latency you are ALREADY hiding**, not a finding, and a
  hotspot list from 2025.4 or later is not comparable with one from an older `ncu`.
- **`CPIStall`'s source markers have TWO gates**: a line is marked when it holds > 30% of that
  line's own samples for one reason AND accounts for > 10% of all sampled stalls in the kernel.
  Marker text: "This line is responsible for {X}% of all warp stalls. {Y}% of the stalls for this
  line are of type {reason}."

## Roofline

`--set roofline` costs 5919 estimated metrics against `basic`'s 213 to restate what the two
SpeedOfLight percentages already said: left of the ridge point is memory-bound, right is
compute-bound, distance below the roof is the headroom, on the roof means done. Its one addition is
arithmetic INTENSITY -- FLOP per byte of DRAM traffic -- which you move rightward by increasing
reuse (tiling, fusion) and never by adding arithmetic. Run it when you intend to change the
intensity; otherwise step 1 has already decided. Check the version before you believe it: the FP32
SOL roofline formula was wrong until 2026.2 and the non-tensor FP16 one until 2025.1.

## The replay trap

Replay is what makes the full metric set possible and it is what makes the numbers not your run's:

- **Every pass reads the SAME inputs.** Pass one saves ALL GPU memory the kernel can reach (which
  can spill to host memory and dominate runtime on a big working set); after that `ncu` restores
  only the subset the kernel writes. So a kernel whose behaviour depends on its data is
  characterised on ONE launch's data. If launch 900 has different convergence, sparsity or branch
  mix from launch 1, profile launch 900 (`-s`) -- do not average, you cannot. A one-pass `--metrics`
  selection skips save-and-restore and does not have this problem.
- **Caches are flushed between passes by default** (`--cache-control=all`), so the hit rates you
  read are cold-start rates. A kernel that in the real run inherits a warm L2 will look WORSE here.
  `--cache-control=none` gives the opposite bias. Worse, when the hit and the query counters land in
  different passes the RATE itself can carry significant error, so treat a hit rate as a direction,
  not a figure. Neither setting is your program; state which one you used.
- **Clocks are pinned, but to WHAT changed.** `--clock-control` defaults to `boost` from Nsight
  Compute 2026.1 onward and to `base` (rated TDP) before that -- on this box 2026.2.1 reports
  `(=boost)` and 2025.4.1 reports `(=base)`. Either way passes are comparable to each other and
  neither matches a real run's clock behaviour. Check `ncu --help | grep clock-control` and say
  which one you got. There are four values, not two: `base`, `boost`, `force-boost` (Ampere+ on
  driver 560, Turing+ on 580) and `none`. It FAILS on a MIG partition -- lock the whole GPU with
  `nvidia-smi` there -- and is a no-op on Linux aarch64 Orin (GA10b) and Thor (GB10b).
- **`ncu` serialises kernel launches by default,** so overlap, concurrent kernels, launch gaps and
  copy/compute overlap do not survive into the report. That is a property of KERNEL replay, not of
  `ncu`: `--replay-mode range` and `app-range` replay whole ranges of launches and API calls and are
  documented to execute kernels WITHOUT serialization -- use them when concurrency is required for
  correctness or is the thing you are measuring. Everything else about concurrency needs a tracer.
- **`--replay-mode application`** re-runs the whole program per pass instead of the kernel, for when
  the kernel's state cannot be snapshot-restored -- but it demands a deterministic program, and one
  with a random seed or an adaptive loop will silently profile different work in each pass.

## Reading a report offline

Profile once, read many times -- no re-run, no second gate. This only exists if collection SUCCEEDED:
a run that hit `ERR_NVGPUCTRPERM` wrote no `.ncu-rep` at all, whatever `-o` said.

```sh
ncu -i prof.ncu-rep --page details --print-details all   # sections, rules AND the body tables
ncu -i prof.ncu-rep --page raw --csv                     # every collected metric, parseable
ncu -i prof.ncu-rep --page details --print-summary per-kernel
```

The `details` page carries NVIDIA's own rule text ("this kernel is bound by ..."): the table above
executed for you, same thresholds, same metrics. A starting hypothesis, not a finding -- it does not
know what your kernel is allowed to change.

**The install ships six reference reports in BEFORE/AFTER PAIRS** under `<install>/extras/samples/`.
They need no GPU and no counter permission, so they are how you learn the text format, validate a
parser and rehearse the diff below:

| directory | before -> after | rules that fire |
| --- | --- | --- |
| `uncoalescedGlobalAccesses/` | `addConstDouble3` -> `addConstDouble` | `UncoalescedGlobalAccess`, `MemoryCacheAccessPattern` |
| `sharedBankConflicts/` | `transposeCoalesced` -> `transposeNoBankConflicts` | `SharedMemoryConflicts`, `UncoalescedSharedAccess` |
| `instructionMix/` | `sobelDouble` -> `sobelFloat` | `HighPipeUtilization`, `SlowPipeLimiter`, `FPInstructions` |

Every one of their READMEs collects with `ncu --set full --import-source on -o X.ncu-rep ./app` off
a `-lineinfo` build. A fourth sample, `interKernelCommunication/`, ships no report and exists to
document why **`ncu ./app` HANGS** on mutually dependent concurrent kernels ("this will hang because
the kernel launches are serialized and cannot run concurrently") and that `--replay-mode range` with
`--nvtx-include` is the fix -- at the price the same README states: "the set of available metrics
for the 'range' workload type is a subset of those available for the 'kernel' workload type".

## Comparing two profiles

**There is no `--baseline` in the CLI.** Baselines are a `ncu-ui` feature (Add Baseline, saved to a
`.ncu-bln`, drawn as difference bars in the section headers). From the CLI you join two dumps:

```sh
ncu -i before.ncu-rep --page raw --csv --print-units base --print-metric-instances none > before.csv
ncu -i after.ncu-rep  --page raw --csv --print-units base --print-metric-instances none > after.csv
# join on the metric-name column
```

`raw` because its key is the metric ID, which does not get re-worded between releases the way a
label does; `base` units because auto-scaling silently changes the unit under a value; instances
pinned because that setting changes the row shape. **Both dumps must come from the SAME `ncu`
version** -- record it beside the numbers. From 2026.1 there is a better path than CSV:
`pip install ncu-report` reads a `.ncu-rep` directly in Python.

**Collect for comparability, not just for numbers.** NVIDIA: "When comparing results, we recommend
to lock clocks with `nvidia-smi` externally before profiling and use `--clock-control none` for
ncu", and for cache-sensitive kernels "use `--replay-mode application --cache-control none` for ncu
to let the application handle priming the caches implicitly". That second sentence is the documented
way out of the single-pass restriction on `--cache-control none` above.

**Diff counts, not percentages.** Reproducibility, best to worst:

1. Launch and device attributes, occupancy limiters, `Waves Per SM` -- static properties of the
   launch, exact.
2. Counters: `.sum` of instructions, sectors, bytes, requests -- exact under kernel replay.
3. `pct_of_peak_*`, durations, frequencies -- move with clocks, and the `--clock-control` DEFAULT
   itself changed between 2025.4 and 2026.1.
4. Ratios and hit rates -- worst, because their numerator and denominator can land in different
   passes. "A metric such as hit rate (hits / queries) can have significant error if hits and
   queries are collected on different passes and the kernel does not saturate the GPU to reach a
   steady state (generally > 20 us)." NVIDIA's mitigation is worth reading twice: "**Reducing the
   number of metrics collected at the same time can also improve precision**" -- a targeted
   `--metrics` run is more ACCURATE, not merely cheaper.

One more contaminant, and it is the common laptop and workstation case: "GPU engines other than the
one measured by a metric (display, copy engine, video encoder, video decoder, etc.) potentially
access shared resources during profiling ... If the kernel launch is small, the other engine(s) can
cause significant confusion in e.g. the DRAM results". Profile short kernels on a GPU with no active
display. Two `ncu` processes on one device also serialise against each other through
`TMPDIR/nvidia/nsight_compute/lock.<UUID>`, so a shared box gives you queueing, not corruption.

## Version deltas that change the numbers

Only the ones that change an action, from `<install>/docs/ReleaseNotes/index.html`:

- **2026.1: `--clock-control` default `base` -> `boost`.** Every percent-of-peak collected from
  2026.1 on is against a different clock basis than a 2025.x default run.
- **The roofline formulas were WRONG**: non-tensor FP16 until 2025.1 ("incorrect multiplier"), FP32
  until 2026.2 ("Fixed the `achieved_fp32` formula in the SOL roofline rules"). Do not trust an
  older roofline verdict and never compare one across that boundary.
- **2025.3 rescaled `launch__waves_per_multiprocessor` to the SMs of a green context** -- step 2's
  metric changed meaning on a partitioned GPU, where SOL percentages are relative to the partition.
- **2025.4 flipped warp sampling to `_not_issued`** (source page, above).
- **2026.2.1's `HighPipeUtilization` knows Blackwell pipes** -- `aluheavy`, `fmaheavy`, `fmalite`
  and the `fmaheavy` subpipes -- that 2025.4.1 cannot see at all, plus a "Most frequently executed
  instructions for pipeline {X}" table. The which-pipe answer genuinely differs between the two
  installs on this box.
- **L2 compression metrics are arch-split under IDENTICAL labels**: `lts__gcomp_*` on CC 80-89 /
  110 / 120-121, `lrc__ilc_*` on CC 90-100 / 103. A metric-ID-keyed diff across architectures breaks
  exactly here.
- **Register spilling became first-class at 2025.1**: `sass__inst_executed_register_spilling`, plus
  MemoryWorkloadAnalysis's `derived__local_spilling_requests` / `derived__shared_spilling_requests`
  and their `_pct` overhead rows.

## Traps

- **An `ncu` report with no kernels is an environment finding**, not a fast kernel. Check the exit
  code and stderr before you conclude anything about the code.
- **A tracer and `ncu` do not print the same kernel name.** A trace generally carries a fuller
  demangled name; `ncu` matches the stripped `function` form, and `--rename-kernels` defaults to on
  (`=1`) so it simplifies demangled names further, driven by a `ncu-kernel-renames.yaml` looked up
  in the CWD and `$HOME/.config/NVIDIA Corporation`. The RENAMED name is what `-k` and `--kernel-id`
  match against ("you can use the renamed names to filter the kernels while profiling as well"), so
  that is a second reason a pattern copied out of a trace misses; `--rename-kernels off` restores
  the demangled names. Anchor the regex on the short unique part, not on a copied signature.
- **Profiling overhead is not confined to the profiled launch.** `-c 1` on a kernel that runs 1052
  times collects one launch, but `ncu` serialises ALL launches in the process and there is a large
  one-time cost for the first profiled kernel in each context. The other 1051 are not untouched and
  the surrounding wall time is not a baseline -- take the baseline from a run without `ncu`.
- **A metric absent here can be present on the next box**, and the query defaults hide metrics. Ask
  `ncu --query-metrics --chips <chip>` -- it needs no GPU and no counter permission -- and note that
  the default `--query-metrics-collection profiling` does NOT list the occupancy limiters or
  `Waves Per SM`. The collection NAMES are stable, the row counts are a version fact: measured on
  2026.2.1 here, `--query-metrics-collection launch` returns **91** rows and `occupancy` returns
  **8** lines / 3 metrics, against 61 and 5 on the other install for the same chip -- re-run it
  rather than porting a count. `launch` is the only place `launch__occupancy_limit_*` and
  `launch__waves_per_multiprocessor` appear; `occupancy` holds
  `sm__maximum_warps_per_active_cycle_pct` and `smsp__maximum_warps_avg_per_active_cycle` -- the
  step-3 denominator. That `occupancy` collection is **new in 2026.1** and absent from the 2025.4.1
  HPC-SDK copy. `--list-chips` names the chips you can ask about.

## Documentation

- **All of the NVIDIA pages below ship offline inside the install** as `<install>/docs/<Guide>/index.html` -- same text, no network, and the version you actually run
- Nsight Compute CLI reference: `--set`, `--section`, kernel filtering, replay modes, the `--print-*` output flags, `--clock-control` defaults -- https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- Profiling guide: replay, serialization, overhead, and what each metric means -- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- Metric structure: which suffixes are already percent-of-peak, and why a throughput is a MAX -- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
- Stall reason semantics, cited by NVIDIA's own shipped `CPIStall.py` -- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference
- Which workloads each pipeline handles, cited by `SlowPipeLimiter.py` -- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder
- Reducing uncoalesced device memory accesses, cited by `UncoalescedAccess.py` -- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses
- Optimizing occupancy, cited by `AchievedOccupancy.py`; 11.3 carries the "66 to 100 percent" sentence and 11.4 the shared-memory occupancy experiment -- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy
- The 16-warps latency figure, in the ARCHIVE only (the live guide dropped it) -- https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html
- Analysis-Driven Optimization, three parts: NVIDIA's own read-a-report loop in "pareto order", with the stopping rule made explicit -- cross-check the kernel against an external bandwidth measurement and stop when it beats `bandwidthTest` -- https://developer.nvidia.com/blog/analysis-driven-optimization-preparing-for-analysis-with-nvidia-nsight-compute-part-1/
- Transactions per request, and the floor of 4: the replacement for the dead `gld_efficiency` -- https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/
- Install locations and general usage -- https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
- 2026.1 release notes, where the `--clock-control` default became `boost` -- https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/updates-2026-1.html
- The metric naming scheme, which is not guessable -- https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html
- The profiling permission gate -- https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters
