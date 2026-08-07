---
name: ncu
description: Profile ONE CUDA kernel yourself with Nsight Compute -- read the numbers in NVIDIA's own order, against NVIDIA's own thresholds, and turn each reading into a change.
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

You run this yourself, on your own build -- there is no judge route. `ncu` has to be on the box the
kernel runs on and the driver's profiling gate has to be open; the next section checks both.

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
`*.py` -- NVIDIA's shipped rules, grep-able at the paths named below, and identical across both
installed versions. What a real kernel READS against those thresholds was collected here.

## Target ONE kernel

A 1052-launch run profiled whole is hours of replay for one answer. Narrow first, always:

```sh
ncu -k regex:jacobi -c 1 -s 20 --set basic -o prof -f -- ./app input
```

- **`-k` / `--kernel-name`** takes a bare name for an exact match or `regex:<expr>`. It matches on
  the `function` basis by default -- "function name without parameters, templates etc.", so BOTH the
  parameter list and the template arguments are stripped, and `regex:mykernel<float>` matches
  nothing. Anchor on the bare name. `--kernel-name-base demangled|mangled` switches.
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
| `detailed` | 1071 | + Compute/MemoryWorkloadAnalysis, MWA_Chart, SourceCounters, Tile, roofline chart | after `basic` names a direction |
| `roofline` | 5919 | SpeedOfLight + five roofline charts + WorkloadDistribution | rarely; see below |
| `full` | 7381 | everything, and the ONLY set carrying SchedulerStats, WarpStateStats, MWA_Tables, InstructionStats | last resort, one launch only |

`full` reads 8051 on 2025.4.1 against 7381 on 2026.2.1 for the same chip: a version artefact, not a
workload fact. **The decision path below needs three sections no bundle short of `full` carries.**
Ask for them by name rather than paying for `full`:

```sh
ncu -k regex:jacobi -c 1 \
    --section SpeedOfLight --section LaunchStats --section Occupancy \
    --section SchedulerStats --section WarpStateStats \
    -- ./app input
```

`ncu --list-sections` prints the identifiers `--section` takes. Asking for two sections beats
`--set full` every time. `--metrics a,b,c` is cheapest of all, and if the selection fits in ONE pass
`ncu` skips the save-and-restore entirely. `ncu --list-metrics` lists the metric NAMES the current
section selection would collect -- names only, not a cost or a pass count.

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

**The one that catches people: `Memory Throughput` is not DRAM throughput.** It is
`gpu__compute_memory_throughput...`, the maximum over the memory hierarchy, and `DRAM Throughput`,
`L1/TEX Cache Throughput` and `L2 Cache Throughput` are three SEPARATE rows in the same header.
`Memory Throughput` at 85% with `DRAM Throughput` at 30% means L1 or L2 is the saturated unit, and
every change that cuts DRAM bytes buys nothing. Read the Memory Throughput Breakdown, which exists
to name the contributor, before you touch a single access.

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

A 640x swing in the one number that decides whether you are memory-bound, from a flag nobody sets.
Scale the same kernel to a 96 MB working set and the two agree (94.53% against 94.33%), because
then the data genuinely does not fit and the flush changes nothing.

So: **a high `DRAM Throughput` on a kernel whose working set fits in L2 is an artefact of the
default.** It is the common shape in a timestep loop, where the same arrays are revisited every
step and are hot by the second iteration. Re-run with `--cache-control none` before you spend a day
cutting DRAM traffic that the real run never moves.

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
3. **`Issued Warp Per Scheduler`** (SchedulerStats), ceiling 1.0, idle below 0.6. Then one branch
   decides the rest: `Active Warps Per Scheduler` / `Theoretical Warps Per Scheduler`. Below 0.8,
   warps are allocated but not ELIGIBLE -- they are stalled, so go to step 5's stall table. At or
   above 0.8 the launch already has nearly every warp it is entitled to, so occupancy is not the gap
   either and NVIDIA's rule names load imbalance first, stalls only after. Occupancy (step 4) is
   what you reach for when the ISSUE rate is fine and the warp count is not.
4. **Occupancy**, only if step 3 sent you. `Theoretical` (a static property of the launch) before
   the gap to `Achieved` (a measured one); they fail for different reasons and take different fixes.
5. **WarpStateStats**, last. A stall reason means nothing until step 3 has shown issue slots are
   actually being lost.

## The reading -> action table

Thresholds are NVIDIA's own, read out of the shipped rules on this box: `SpeedOfLight.py`
(80 / 60 / 10), `TheoreticalOccupancy.py` (80), `AchievedOccupancy.py` (10),
`IssueSlotUtilization.py` (0.6 / 0.8), `CPIStall.py` (0.8 / 0.3), `ThreadDivergence.py` (24),
`SharedMemoryConflicts.py` (10), `LocalMemoryUsage.py` (10), `SlowPipeLimiter.py` (80 / 20 / 25),
`LaunchStatistics.py` (20). They are where NVIDIA's rule text fires, not laws.

| you read | it means | you change |
| --- | --- | --- |
| both throughputs < 60% AND `Waves Per SM` < 1 | the grid cannot fill the device; nothing in the kernel is the limit | more blocks: widen the grid. Do NOT tune occupancy |
| both throughputs < 60%, `Waves Per SM` >= 1 | latency-bound: no resource is near peak | steps 3-5; the body and the traffic are both fine |
| the two within 10 points of each other, neither < 60% | balanced -- cutting one side alone moves nothing | cut BOTH work and traffic; fusion is the single change that does both |
| `Memory Throughput` >= 80% | bandwidth-bound at whichever unit the Breakdown names | cut TRAFFIC, not the loop body: fuse, tile for reuse, recompute, narrower dtype |
| `Compute (SM) Throughput` >= 80% | compute-bound | the only levers left are less work and narrower types -- fp32 over fp64, intrinsics, tensor path |
| `Compute (SM)` >= 80%, average pipe utilisation < 20%, max-minus-avg > 25 points | one slow pipe holds the SM busy while the rest idle | move math off it: fp64 -> fp32 or int |
| `Issued Warp Per Scheduler` < 0.6 AND active/theoretical < 0.8 | warps are allocated but not eligible -- they are stalled | the stall table below |
| `Issued Warp Per Scheduler` < 0.6 AND active/theoretical >= 0.8 | nearly every warp occupancy allows is resident, so occupancy is not the gap | load imbalance first; stalls only after |
| `Achieved Occupancy` HIGH and both throughputs LOW | occupancy was never the problem | go to the stall reasons. Block size is not banned -- NVIDIA names it first when a LIMITER binds (the rows below) -- but changing it to raise an already-high occupancy is motion, not progress |
| `Theoretical Occupancy` < 80%, smallest limiter `Block Limit Registers` | register count caps resident blocks | `__launch_bounds__`, `-maxrregcount`, fewer live values |
| ... smallest is `Block Limit Shared Mem` | shared memory caps resident blocks | smaller tile, or `cudaFuncAttributePreferredSharedMemoryCarveout` |
| ... smallest is `Block Limit Warps` | BLOCK SIZE caps it, and it binds from both ends: too large strands warps, too small wastes block slots | resize, then re-read the limiter |
| ... smallest is `Block Limit SM` | the hardware blocks-per-SM ceiling, nothing you allocated | only MORE warps per block moves it |
| ... smallest is `Block Limit Barriers` | too many barriers per block | fewer `__syncthreads()` |
| `Theoretical - Achieved` > 10 points | the launch could fill the SM and did not: scheduling overhead, tail, imbalance | even work per block, hunt an early `return`. **Re-read `Waves Per SM` first**: inside `1 <= Waves Per SM < 5` NVIDIA attributes the gap to the TAIL (its rule prices a partial wave at `1/(1 + whole waves)` and fires at 20%, which is where the 5 comes from), and the fix is more, smaller waves -- not load balancing |
| `Avg. Active Threads Per Warp` < 24 (of 32) | divergence or early thread completion | fix the BRANCH, not the occupancy. Source Counters names the lines |
| `Average Bytes Per Sector For Global Loads` far below its `Maximum` | uncoalesced: consecutive threads touch scattered addresses | transpose the layout, or stage via shared |
| shared bank conflicts >= 10% of shared wavefronts | shared-memory bank conflicts | pad the leading dimension, or change the access stride |
| `L1TEX Hit Rate` / `L2 Hit Rate` low where you expected reuse | the working set exceeds that level | smaller tile, different loop order, block the loop |
| local-memory instructions > 10% of instructions executed | register spill, or a dynamically indexed array in local scope | fewer live values, or index that array statically |

## The stall table

`--section WarpStateStats`. Every reason is spelled
`smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` and is **in cycles, not
percent**. Its share is that value divided by `Warp Cycles Per Issued Instruction`. NVIDIA's own
rule acts when `Issued Warp Per Scheduler` < 0.8 AND that share exceeds 0.3, so a reason with a
large absolute cycle count and a small share is not your finding.

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
| `imc_miss` | constant-cache miss; lanes reading DIFFERENT constant addresses serialise | make the warp read one constant address, or move the data out of constant memory |
| `not_selected` | eligible warps queued behind another | nothing is wrong: you have MORE occupancy than you need. NVIDIA suggests REDUCING active warps for locality |

Raising occupancy fixes `long_scoreboard`, `wait` or `math_pipe_throttle` only when another warp
could then issue -- which is what step 3 established before you got here.

To get from any of these to a LINE of code: build with `-lineinfo`, add
`--section SourceCounters --import-source yes`, then read the report with
`--page source --print-source cuda,sass`.

## Roofline

`--set roofline` costs 5919 estimated metrics against `basic`'s 213 to restate what the two
SpeedOfLight percentages already said: left of the ridge point is memory-bound, right is
compute-bound, distance below the roof is the headroom, on the roof means done. Its one addition is
arithmetic INTENSITY -- FLOP per byte of DRAM traffic -- which you move rightward by increasing
reuse (tiling, fusion) and never by adding arithmetic. Run it when you intend to change the
intensity; otherwise step 1 has already decided.

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
  which one you got.
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
ncu -i prof.ncu-rep --page details                 # sections plus the built-in rules
ncu -i prof.ncu-rep --page raw --csv               # every collected metric, parseable
ncu -i prof.ncu-rep --page details --print-summary per-kernel
```

The `details` page carries NVIDIA's own rule text ("this kernel is bound by ..."): the table above
executed for you, same thresholds, same metrics. A starting hypothesis, not a finding -- it does not
know what your kernel is allowed to change.

## Traps

- **An `ncu` report with no kernels is an environment finding**, not a fast kernel. Check the exit
  code and stderr before you conclude anything about the code.
- **A tracer and `ncu` do not print the same kernel name.** A trace generally carries a fuller
  demangled name; `ncu` matches the stripped `function` form, and `--rename-kernels` defaults to on
  (`=1`) so it simplifies demangled names further, driven by a `ncu-kernel-renames.yaml` looked up
  in the CWD and `$HOME/.config/NVIDIA Corporation`. Anchor the regex on the short unique part, not
  on a signature you copied out of a trace.
- **Profiling overhead is not confined to the profiled launch.** `-c 1` on a kernel that runs 1052
  times collects one launch, but `ncu` serialises ALL launches in the process and there is a large
  one-time cost for the first profiled kernel in each context. The other 1051 are not untouched and
  the surrounding wall time is not a baseline -- take the baseline from a run without `ncu`.
- **A metric absent here can be present on the next box**, and the query defaults hide metrics. Ask
  `ncu --query-metrics --chips <chip>` -- it needs no GPU and no counter permission -- and note that
  the default `--query-metrics-collection profiling` does NOT list the occupancy limiters or
  `Waves Per SM`. Measured here: `--query-metrics-collection launch` returns 61 rows and is the only
  place `launch__occupancy_limit_*` and `launch__waves_per_multiprocessor` appear;
  `--query-metrics-collection occupancy` returns 5, holding `sm__maximum_warps_per_active_cycle_pct`
  and `smsp__maximum_warps_avg_per_active_cycle`. `--list-chips` names the chips you can ask about.

## Documentation

- Nsight Compute CLI reference: `--set`, `--section`, kernel filtering, replay modes, `--clock-control` defaults -- https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- Profiling guide: replay, serialization, overhead, and what each metric means -- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- Metric structure: which suffixes are already percent-of-peak, and why a throughput is a MAX -- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure
- Stall reason semantics, cited by NVIDIA's own shipped `CPIStall.py` -- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference
- Which workloads each pipeline handles, cited by `SlowPipeLimiter.py` -- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder
- Reducing uncoalesced device memory accesses, cited by `UncoalescedAccess.py` -- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses
- Optimizing occupancy, cited by `AchievedOccupancy.py` -- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy
- Install locations and general usage -- https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
- 2026.1 release notes, where the `--clock-control` default became `boost` -- https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/updates-2026-1.html
- The metric naming scheme, which is not guessable -- https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html
- The profiling permission gate -- https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters
