---
name: linuxperf
description: Finds where CPU time went with linux perf -- record, self vs children, flat kernels, unwind modes, off-CPU, perf stat and diff, traps. Not counter brackets, not GPU.
---

|  | `linuxperf` | `papi-cpu` |
|---|---|---|
| answers | WHERE the time goes | WHY it is slow there |
| mechanism | statistical sampling of the call stack | exact hardware counts over a bracket |
| needs a code change | no | yes -- a start/stop bracket |
| granularity | whatever is a symbol | whatever you bracket |
| main failure | too few samples (a flat or noisy profile) | too short a region (measuring the instrument) |
| perturbs the run | barely | yes -- never compare a counted run's wall clock |

Start here: `perf` is free, needs no edit, and tells you which region is worth counting. Counting
a region that owns 5% of the time is a wasted run whatever the counters say. The order INVERTS
when the kernel is one flat function with no internal symbols -- then this page has nothing to
attribute to, so bracket phases with PAPI counters first to find which one owns the cycles, and come
back once you have promoted that phase to a function.

A kernel that runs on a device goes to `nsys` or `ncu` instead: a host call graph of a device
kernel shows the launch and the wait, not the work.

## The procedure

Run it in order and stop at the first branch that fires.

0. **Check the profile can SEE the run, before recording anything.**

   ```sh
   perf stat -e cycles:u,cycles:k,page-faults -- ./run
   perf stat -e task-clock,context-switches,cpu-migrations,page-faults,major-faults -- ./run
   ```

   Everything below samples `cycles:u`, so it is blind to every cycle spent anywhere else. Three
   outcomes, three different tools -- and the first two get mistaken for each other constantly:

   | reading | what it is | where to go |
   | --- | --- | --- |
   | `cycles:k` a large share of the total | ON-CPU KERNEL work: fault handler, syscalls, `copy_to_user` | still sampleable -- `perf record -e cycles:k -g` names the kernel symbol |
   | `CPUs utilized` (task-clock/elapsed) well under the thread count | genuinely OFF-CPU: blocked on I/O, futex, sleep | invisible to `cycles:u` AND `cycles:k`; next section |
   | `cycles:u` dominant | on-CPU user work | the rest of this page |

   Measured on `harris_corner` at preset S: `cycles:u` 3.51 G (22.4%), `cycles:k` 12.14 G (77.6%),
   5,049 page faults per rep -- the FIRST row, because a minor fault is on-CPU kernel work. A
   user-mode profile of that run puts 72% self on the kernel symbol and points at the loops; the
   actual win was 6.3x from hoisting ten per-call `malloc`/`free` temporaries out of the hot path,
   which `cycles:u` cannot see at all. `perf record -e cycles:k -g` would have named the fault
   handler outright instead of leaving it to inference -- its symbols need `kptr_restrict` = 0.
   Fix that first, then record.

   The pass looks like this: measured on a three-phase compute kernel, `cycles:u` 4.44 G,
   `cycles:k` 85.0 M = **1.9%**, 208 page faults per rep. Under ~5% kernel time the user-mode
   profile describes the run -- continue. The second command needs no privilege and costs no PMU
   counter (all software events): `major-faults` above 0 is real disk I/O, many
   `context-switches` with few `cpu-migrations` is blocking, many migrations is placement.
1. **Record enough of the kernel.** The kernel must own most of the recording, or you profiled
   startup. At 999 Hz, ~0.3 s of kernel work is the usual floor -- raise the rep count until the
   kernel's total% is the biggest number on the page.
2. **Rank by SELF time.** The first frame in that list you own is the candidate.
3. **Check its share.** Below ~30% of the run, usually stop: at 30% the whole-run ceiling is
   1/(1-0.30) = 1.43x even if you make the frame free. Go find the frame that owns the rest.

   **If no frame owns the rest -- if the profile is FLAT across many phases -- the flatness IS the
   finding.** A chain of passes each at 5-11% has no per-frame edit worth making; the top phase's
   ceiling is 1.12x. Compare total bytes moved per rep against the last-level cache and fuse the
   passes instead, which cuts traffic no single-loop transform can touch.

   **The middle case is the common one: a few frames each at 25-40%.** Measured -- 40.32 / 30.48 /
   28.39, summing to 99.19% -- which is neither one frame nor flat, and step 3 as stated covers
   neither. Compute each frame's own ceiling and take the largest: 1/(1-0.4032) = 1.68x here, and
   no edit to that frame can beat it. Making that phase 6.63x faster returned 1.55x on the whole
   run, inside the prediction. Say the ceiling out loud before spending the day.
4. **Read children to find who is responsible.** Walk down from a high-children/low-self caller to
   the first frame whose body IS the algorithm rather than dispatching, packing or copying.
5. **Only then ask what the machine was doing there.** That is a hardware counter bracket, not a
   sampler. `perf stat -M PipelineL1` gives the coarse whole-process version with no code change;
   the per-phase version is a PAPI bracket.

**Unresolved `[k]` hex addresses are kernel frames**, not a broken unwind -- `kptr_restrict`
withheld the symbols. They are a different failure from `[unknown]` (a truncated DWARF stack, whose
fix and its price are in Traps) and the fix is not the same. Their share is a LOWER BOUND on
kernel-mode time: more than a few percent means go back to step 0 and count `cycles:k`. Measured:
`kptr_restrict=1`, `[k] 0xffffffffb891ba28` at 0.11% in the report against `cycles:k` = 1.9%.

That separation holds in `perf report` only. **`perf script` prints kernel frames as
`ffffffffb7200b90 [unknown] ([unknown])`** -- character for character what a cut-off DWARF stack
produces. Only the `0xffffffff...` address range tells the two apart there.

## Off-CPU time

Row two of step 0: the process is blocked, so there are no cycles to sample and no cycles event ever
finds it. `perf record --off-cpu` collects it with BPF; the sample period is nanoseconds slept, so
`perf report`'s Overhead column ranks by BLOCKED TIME and is directly comparable to a wall-clock
budget (perf-record(1)).

Three preconditions, and the first one fails in the worst possible way:

- **`perf version --build-options | grep bpf_skeletons` must read `[ on ]`.** It is `[ OFF ]` on
  this box, and a perf built without skeletons does not refuse: it prints a warning and then
  RECORDS A NORMAL ON-CPU PROFILE ANYWAY. You get a plausible file that answers the other question.
- **root.** The BPF program needs it.
- **FRAME POINTERS.** perf-record(1) is explicit that BPF can collect stack traces from frame
  pointers only, so `--call-graph=dwarf` does nothing for the off-CPU half and a binary built
  without them "might see bogus addresses". This one case inverts the build advice below: an
  off-CPU plan wants `-fno-omit-frame-pointer`.

`--off-cpu-thresh` defaults to 500 ms. Below it, waits are not emitted as samples at all -- they are
accumulated in a BPF map and appear as an aggregate after every regular sample, with no per-event
context. A kernel whose off-CPU time is microsecond futex waits needs `--off-cpu-thresh 1`.

If the wait is a kernel lock, `perf lock contention -ab -- ./run` gives
`contended / total wait / max wait / avg wait / type / caller`; the `-b` (BPF) path is gated by the
same `bpf_skeletons` build. The non-BPF fallback (`perf record -e sched:sched_stat_sleep,...` then
`perf inject -s`) needs `/proc/sys/kernel/sched_schedstats` = 1 (it reads 0 here) and a readable
tracefs -- `/sys/kernel/tracing` is `drwx------ root` on this box, so `perf list 'sched:*'` returns
nothing as a normal user even at `perf_event_paranoid=0`. That is a tracefs permission, not a
paranoid setting, and no `perf_event_paranoid` change fixes it.

## Reading perf stat

The `#` column is not free text. perf computes those lines from the JSON metric group `Default`, and
several carry a threshold it ships and colors red when exceeded (perf-stat(1)).

| line | formula | shipped threshold |
| --- | --- | --- |
| `insn per cycle` | `instructions / cpu-cycles` | `< 1`. The ceiling is dispatch width -- 6 ops/cycle on Zen4, so IPC 1.0 is ~17% of peak, not "fine" |
| `GHz` | `cpu-cycles / task-clock` | none. Frequency WHILE ON CPU; blind to blocked time, so a low value is downclocking, never idleness |
| `CPUs utilized` | `task-clock / duration_time` | none. The only default line that sees off-CPU time |
| `frontend cycles idle` | `stalled-cycles-frontend / cpu-cycles` | `> 0.1` |
| `backend cycles idle` | `stalled-cycles-backend / cpu-cycles` | `> 0.2` -- but read the next paragraph first |
| `% of all branches` | `branch-misses / branches` | `> 0.05` |

**On Zen4, `backend cycles idle 0.00%` means the EVENT DOES NOT EXIST.** `perf list hardware` on
this CPU has no `stalled-cycles-backend` alias, and the metric is written
`stalled-cycles-backend / cpu-cycles if has_event(stalled-cycles-backend) else 0` -- so it prints a
confident zero on a workload that is entirely backend bound. That is the highest silent-wrong-answer
risk in perf's own default output. **The red/green also dies the moment you pipe** -- perf's color
default is tty-only and `perf stat` has no `--stdio-color` (that flag belongs to `perf report`) --
so apply the table by hand when reading a log.

**`--topdown` errors out on AMD**: `Topdown requested but the topdown metric groups aren't present.`
perf builds the group name `TopdownL<n>` by string surgery and AMD ships the same buckets under
other names. The command here is `perf stat -M PipelineL1 -- ./run`: five level-1 buckets
(`retiring`, `bad_speculation`, `frontend_bound`, `backend_bound`, plus `smt_contention`, which
Intel's TMA does not have). `-M PipelineL2` splits `backend_bound` into `backend_bound_memory` vs
`backend_bound_cpu` by whether retirement was blocked on an incomplete LOAD -- the fastest
is-it-memory-bound answer perf can give, and the cue to reach for the data side below. AMD's buckets
carry no thresholds; Intel's shipped `tma_*` numbers are the working rule (`backend_bound > 0.1`,
`frontend_bound > 0.15`, `bad_speculation > 0.15`, `retiring > 0.75` is good) and they NEST: do not
read a level-2 bucket whose parent is under threshold.

### Multiplexing

**A trailing `(NN.NN%)` on a counter line appears if and only if that event was multiplexed** --
perf prints it only when `run != ena`. No parenthesis means the counter ran the whole time.

- The number in front of it is `raw * enabled/running`, which the perf wiki calls an ESTIMATE, not a
  count. `--no-scale` shows what was actually observed.
- **Never form a ratio from two events with different running percentages.** They were measured in
  different windows. The fix is an event group -- `perf stat -e '{cycles,instructions}'` is
  scheduled all-or-nothing, one window, real ratio. A group cannot exceed the counter budget or mix
  PMUs.
- `<not counted>` and `<not supported>` are different bugs. Not supported = no such event on this
  PMU (a typo, or the other vendor's event name). Not counted = supported, but it never got a slot
  or the process exited before the first read.
- The budget here is 6 general-purpose PMCs, minus one held by the NMI watchdog while
  `/proc/sys/kernel/nmi_watchdog` is 1. Software events (`task-clock`, `page-faults`,
  `context-switches`, `duration_time`) take no PMC, so mixing those in is free.
- `-d`/`-dd`/`-ddd` ask a dozen hardware events of those slots, so every line comes back
  multiplexed -- hypothesis generator only, re-measure in a group anything you will quote. On Zen4
  the LLC half of `-d` comes back `<not supported>` outright: there are no generic `LLC-loads` /
  `LLC-load-misses` aliases. Use `-M l3_cache`, `-M l2_cache`, `-M l1_dcache`.

## Build for profiling

Keep the release flags and add `-g`. Nothing else. `-g` emits DWARF beside the code and changes no
instruction, so the profiled build times like the submitted one.

`-fno-omit-frame-pointer` is not needed for `--call-graph=dwarf`, which unwinds from `.eh_frame` --
gcc and clang emit it on x86-64 whether or not you pass `-g`. Measured here: a `-O3` build with no
`-g` at all still unwound to `main` and `_start`. Leave it off in the build whose wall clock you
report; it costs a general-purpose register in every function. It is not worthless, though: frame
pointers are the unwind that survives a perf not linked against libunwind/libdw, a stack deeper
than the DWARF dump, and eBPF profilers, which cannot DWARF-unwind at all.

Profiling a `-O0` build tells you about a program nobody runs.

## How it runs

```sh
perf record -q -e cycles:u --call-graph=dwarf -F 999 -o perf.data -- ./run
perf report -i perf.data --stdio --no-children -g none   # SELF time, ranked flat -- what to edit
perf report -i perf.data --stdio --no-children           # same, plus a call tree under each entry
perf report -i perf.data --stdio                         # children (cumulative) -- who is responsible
perf script -i perf.data -F comm,ip,sym,dso --no-inline  # one line per frame, leaf first
```

`-g none` is what makes the list flat: once callchains are recorded, `report` defaults to `-g graph`
and prints a call tree under every entry, `--no-children` or not. **Pass `--call-graph` explicitly
too** -- its default is `fp`, the mode this page proves wrong on libc and CPython, and it fails
silently either way: measured here it TRUNCATED (lost `main`, `__libc_start_call_main`, `_start`),
and it can equally invent a plausible wrong chain or shatter one hot symbol into a dozen lines.
None of the three errors. `record -- cmd` samples the
command AND its descendants,
so a runner that forks the measured child is still profiled. `cycles:u` is user-space only; kernel
samples need a lower `perf_event_paranoid` and answer a different question. **Do not wait for perf
to warn you about `kptr_restrict`**: measured here, no such warning appeared on any of ~10
recordings, with or without `-q`, while kernel symbols WERE being withheld. Read
`/proc/sys/kernel/kptr_restrict` yourself (1 here, and `/proc/kallsyms` reads all-zero addresses);
the tell in the report is `[k] 0xffffffff...` where a symbol should be. See step 0. `-F 999` rather
than 1000 so the sampler
cannot phase-lock onto a kernel whose own period is a round number of milliseconds; `-F` is a
REQUEST, throttled down to `kernel.perf_event_max_sample_rate`, so add `--strict-freq` to turn a
silent tenth of the samples into an error. `-q` silences perf's own chatter -- `perf script` has no
`-q`, it errors with ``unknown switch `q'``.

## Self and children

Two columns, two different findings:

| column | means | ranks |
| --- | --- | --- |
| self (exclusive) | time in this frame's own instructions | WHAT to optimize |
| children (inclusive) | this frame plus everything it called | WHO is responsible |

High children with near-zero self is a caller: walk down, do not edit here. A high-self leaf inside
`libopenblas` or `libc` is not your loop; your decision is about the call, not its body.

Self percentages are shares of the WHOLE recording -- process start, input construction, then the
reps -- and they sum to 100%. Children percentages DO NOT: a caller and its callee both count the
same samples, so the column routinely sums past 100%. Never add two children numbers.

## Your kernel is one function

The corpus reference kernels are generated by FLATTENING the whole computation into a single
`extern "C"` function. `cavity_flow`'s numpy source has three (`build_up_b`, `pressure_poisson`,
`cavity_flow`); the generated C++ has exactly one user function, `cavtflow_fp64` -- the entry
symbol is `<short_name>_fp64`, not the python name -- and the other two phases have no symbol at
all, not even a `static` one. The translator flattened them;
the compiler did not inline them away, so no compiler flag brings them back. A ranked self-time
list therefore has exactly ONE entry for your kernel. That is the shape of the profile, not a
broken tool.

The way out is to give a phase a symbol of its own, in a DIAGNOSTIC build:

```c
__attribute__((noinline)) static void phase_pressure(double *__restrict__ p,
                                                     const double *__restrict__ b, ...) { ... }
```

Split the flat body into `noinline` phase functions, rebuild, profile, and each phase gets its own
line in the ranked list. The cost is one call per invocation, which is nothing next to a phase big
enough to measure. **Mark every pointer parameter `__restrict__`, and check the split build's wall clock still
matches the flat one.** Lifting a nest out of a function where the compiler knew the buffers
could not alias, into one taking plain pointers, can lose the vectorization -- and then you have
profiled a de-vectorized program and attributed its time to the wrong phase.

**gcc renames the phases you just made.** IPA clones a `noinline static` function, so the symbol in
the report is `phase_pressure.constprop.0`, not `phase_pressure` -- and `nm` shows only the suffixed
name, so a reader grepping the source name finds nothing. `.isra.N`, `.part.N` and `.cold` are the
same family (measured: `phase_colsum.constprop.0`, `__printf_fp_buffer_1.isra.0`), and `.cold`
additionally splits one function across two entries in the ranked list. Anchor every grep on the
prefix.

Submit the version without the split -- or keep it only if you measured the cost as
zero.

**When phases already ARE separate functions, `-g` is enough and `noinline` is not needed.**
Measured: a `static` helper inlined at `-O3` disappears from `nm`, and perf still recovers it from
DWARF -- `perf report --stdio` prints `---inner_phase (inlined)` in the call tree with no extra
flag (`--inline` is ON by default; `--no-inline` is the flag that hides it), and `perf script`
without `--no-inline` emits it as a frame. What inline expansion does NOT do is split
the ranked self-time list: the enclosing symbol still holds 99.38% and the phase appears only
inside the call graph.

## What perf still tells you about a flat kernel

Three findings survive having one symbol. From a real run -- `cavity_flow`, C++, preset S, one
thread, 300 reps, 440 samples of `cycles:u`:

| symbol | dso | self% | total% |
| --- | --- | --- | --- |
| `cavtflow_fp64` | `libcavtflow.so` | 22.50 | 36.14 |
| `__memmove_avx512_unaligned_erms` | `libc.so.6` | 13.64 | 13.64 |
| `_PyEval_EvalFrameDefault` | `libpython3.12.so.1.0` | 10.91 | 88.86 |

1. **The kernel's share of the process.** 36.14% total. Everything else is the driver, and a
   transform that halves the kernel moves the wall clock by 18%.
2. **What the compiler turned your code into.** The kernel's own instructions are 22.50%; the
   remaining 36.14 - 22.50 = 13.64 points are `__memmove_avx512_unaligned_erms` UNDER it in the call
   graph -- the `un`/`vn`/`b` array copies became memmove calls. Nothing in the source says memmove.
   Its flat 13.64 equals its share under the kernel, so every memmove sample came through your code;
   a flat number BIGGER than the child number is the same symbol reached by another call path, since
   the flat list sums over all paths and the tree shows only the part under your frame.
3. **Thread attribution, in ONE run.** `perf record -s` then `perf report -T`, or
   `perf report --stdio --sort tid,sym`, splits the profile per thread. Comparing separate runs at
   different thread counts confounds the serial fraction with every other thread-count effect.

**The rep count decides whether any of this is trustworthy.** The same kernel at the default 50
reps put 8.48% of the recording on `cavtflow_fp64` -- fewer than 30 samples out of 330, with the
interpreter owning the rest. At 300 reps it is 36.14% and 159 samples. One rep is 0.489 ms here, so
50 reps is 24 ms of kernel work inside a ~0.3 s process. Raise reps until the kernel's total% is
the biggest number on the page, then read it.

**Profile the kernel's real inputs.** 137 corpus kernels define their own `initialize`. A uniform
random fill written for the profiling driver measures a different workload for any kernel whose
branches, iteration count or sparsity are data dependent.

## The flame graph, in text

`flamegraph.pl` and `perf script report flamegraph` are often not installed. perf prints the same
thing without them:

```sh
perf report -i perf.data --stdio --no-children -g folded,1,caller | grep -E '^[0-9]+\.[0-9]+%'
```

```
99.38% _start;__libc_start_main_impl (inlined);__libc_start_call_main;main;kernel_fp64;inner_phase (inlined)
```

Each surviving line is one folded stack, root first, with its share of the recording. The `grep` is
not optional: unfiltered, perf interleaves the folded lines with the ordinary ranked histogram and
its `#` headers, which any stackcollapse consumer chokes on. The `1` is the callchain threshold in
percent (perf's default is 0.5), so chains under it are dropped and the lines do not sum to 100%.

**Zero lines out means you recorded without callchains, not that the program has no stacks.**
`perf record` collects NONE unless `-g` or `--call-graph` was passed -- the "its default is `fp`"
note above is the default of the OPTION, not of `record`. Measured on a recording made without it:
the flat self ranking is still perfect (`phase_colsum.constprop.0` 39.32%) and this pipeline prints
zero lines, exit 1, no error at all. Confirm with `perf report --header-only | grep -c callchain`,
then re-record with `--call-graph=dwarf`.

The reading rules are the flame graph's rules:

- **Width is cumulative on-CPU time.** The widest box at a level is the biggest consumer. Width can
  come from one slow call or a million fast ones; the graph cannot tell you which.
- **The y axis is stack depth and the TOP box is what was running.** Everything below it is
  ancestry, not cost of its own.
- **The x axis carries no time ordering at all.** Frames are sorted alphabetically so identical
  boxes merge. Left-to-right is not a sequence; do not read one into it.
- **A wide plateau is the target.** A tall narrow tower is a deep call stack that costs nothing.
- **Broken or truncated stacks** are an unwind failure, not a shallow program. Fix the unwind
  before you read anything else.

## fp vs dwarf vs lbr

Same samples, three ways to get the stack under them. This is a decision, not a menu.

| mode | overhead | correct when | fails by |
| --- | --- | --- | --- |
| `--call-graph=fp` | near free | every frame kept its frame pointer | truncating, inventing a plausible wrong chain, or fragmenting one symbol into a dozen lines |
| `--call-graph=dwarf` | the expensive one | AOT build with `.eh_frame`, to the dump size | `[unknown]` past the copied stack |
| `--call-graph=lbr` | cheap, most accurate | Intel, and only to LBR depth | silently truncating past LBR depth |

The overhead column is qualitative; no upstream doc puts numbers on it. `dwarf` also needs a perf
linked against libunwind or libdw, and has nothing to unwind for a JIT frame (numba, JVM, V8).

**Reach for `dwarf`.** It is the only one that is right on a build you did not compile yourself,
which includes libc, CPython and every BLAS. `fp` is wrong there and does not say so: measured on
this box, an `fp` unwind of a two-phase C program produced
`phase_axpy <- call_init (inlined) <- __libc_start_main_impl <- _start`, with `main` missing and a
frame that never ran in its place. The dwarf unwind of the same binary gave the real chain.

**`fp` does not merely truncate -- it SHATTERS the ranking into exactly the shape step 3 tells you
to act on.** Measured on a three-phase `-g` binary whose dwarf profile is ONE line at 40.32%: the
same workload under `--call-graph=fp` folded into ~10 lines of 2.6-4.0% each, top line 4.02%,
rooted at garbage hex (`0xe3f123`, `0x37451996`, `0x389496b9`) and carrying a fabricated parent --
`main;now_s (inlined);phase_colsum.constprop.0`, where the timer function `now_s` never called the
phase. No correct stack survived. A reader applying step 3 to that reads "flat profile, ceiling
1.04x, no edit worth making", the exact opposite of the truth. So: before invoking the flatness
rule, check that the small entries are DISTINCT SYMBOLS. Repeats of one symbol under differing or
hex roots are an unwind artifact, and a plausible flat profile taken under `fp` is an artifact until
a dwarf recording confirms it.

The price of `dwarf` is that every sample copies the full `stack-size`, however shallow the stack
really was: measured, 8.5 KB of `perf.data` per sample at the default 8192 and 66 KB at
`dwarf,65528`. Multiply by samples ACTUALLY taken, not by wall clock -- a fully user-bound run at
999 Hz costs ~8 MB/s at the default, a half-user-space run half that. Same 0.44 s workload, three
`perf.data` files: `fp` 35 KB, `dwarf` 1.9 MB, `dwarf,65528` 13 MB.

**LBR is not available on this box.** `--call-graph=lbr` asks the PMU for branch-stack call-stack
mode, which is an Intel LBR feature; on Zen4 (`amd_lbr_v2`) perf refuses with `cycles:uH: PMU
Hardware or event type doesn't support branch stack sampling`. Plain branch records still work
(`perf record -e cycles:u -b`), but they are branch history, not a call graph. Where LBR does work,
read the depth off the box rather than off a per-microarchitecture table:
`/sys/bus/event_source/devices/cpu/caps/branches` (16 here, matching `amd_lbr_v2`; 16 on Nehalem
through Broadwell, 32 on Skylake and later, 8 on Atom/Silvermont). Past that depth children time is
meaningless rather than approximate.

## Which array stalled, not which loop

`cycles:u` sampling names the instruction and never the data. Two commands name the data. Both ride
IBS on AMD, so both need root or `CAP_PERFMON` -- IBS has no user/kernel filtering and
`perf_event_paranoid=0` does not substitute.

- **`perf mem record -- ./run`** then `perf mem report --stdio`. Its **Overhead column is
  latency-weighted, not a sample count**: the weight is the access latency in cycles, so it ranks by
  total stall time, which is a different rule from every other command on this page. The
  `symbol_daddr` sort key names the hot ARRAY. Reach for it when a flat profile's bytes per rep
  exceed the LLC and you need to know which buffer to fuse around, or when `backend_bound_memory`
  cleared its threshold. `--ldlat` is Zen5-onward -- `ibs_op/format/` here holds only
  `cnt_ctl, l3missonly, swfilt` -- so on Zen4 bias the sample population with
  `perf record -e ibs_op/l3missonly=1/ -c 100000` instead.
- **`perf c2c record -a -- ./run`** then `perf c2c report -NN -d lcl --stdio`, for false sharing.
  Reach for it when a threaded kernel scales sub-linearly and no lock appears in the profile. Zen4
  is supported; Zen3 explicitly is not. On a single socket there is no remote node, so `Rmt Hitm`
  stays 0 and the table reads clean on a genuinely contended line unless you sort `-d lcl`. The
  Pareto table is the point: different OFFSETS in one line touched by different threads is FALSE
  sharing -- pad or split the struct; the SAME offset is TRUE sharing -- an algorithmic dependency
  that padding will not fix.

## Comparing two recordings

```sh
perf diff before.data after.data      # no args: perf.data.old against perf.data
```

**`Delta` is a difference of SHARES, not of time**: `d = A_percent - B_percent`. A uniform 2x
speedup therefore prints ~0 everywhere, and a win on one function pushes every other function's
share UP, which reads as a regression on all of them. `perf diff` answers "how did the mix shift",
never "what got faster". For absolute movement pass `--period` (raw period values for both sides)
and `-c ratio` (`A_period / B_period`); `--formula` prints what it computed.

Two more traps. A row with a BLANK Baseline column exists only in the new file, and
`--baseline-only` suppresses exactly those -- which is every newly-hot function, so it is the wrong
default for an A/B. And **pairing is BY SYMBOL NAME**: a rename, an inlining change, or the
`noinline` phase split this page recommends silently unpairs entries and manufactures large deltas
on both sides. Diff two builds that differ in one transform, not in structure.

The timing verdict never comes from `perf diff`:

```sh
perf stat --null -r 5 --table -- ./run
```

`--null` counts nothing, so the wall clock is uncounted -- this page's opening table says a counted
run's wall clock belongs to no comparison, and A/B is where the temptation is. `-r N` prints mean
+/- stddev; `--table` prints every run, the only way to tell a bimodal five from a merely noisy
five. `-D msecs` starts measuring after startup, a second answer to step 1 that does not need more
reps. Both arms take the identical event list -- one extra event in one arm can push that arm into
multiplexing and change its overhead.

**Reject the pair if the `GHz` line moved.** `cpu-cycles / task-clock` is free, is in the default
output, and is the cheapest validity guard there is: a 2% win at a 4% higher clock is not a win.
Governor `performance` is not enough while `cpufreq/boost` is 1 and `amd_pstate` runs
`active` + `prefcore` -- the clock still moves between runs, and the scheduler can land the thread
on a core with a different preferred frequency.

## Traps

**`[unknown]` frames mean the unwind stopped, not that nothing ran.** DWARF copies at most
`stack-size` bytes per sample, 8192 by default; a deeper stack is silently cut off. 65528 is the
maximum and perf rejects more with `callchain: Incorrect stack dump size (max 65528)`. **But raising
`stack-size` at all needs matching `-m`/`--mmap-pages` headroom, and both ways of getting it wrong
are silent.** Measured on a default box (`perf_event_mlock_kb` = 516): EVERY `dwarf,N` with N > 8192
dies with

```
Permission error mapping pages.
Consider increasing /proc/sys/kernel/perf_event_mlock_kb,
or try again with a smaller value of -m/--mmap_pages.
```

and under the `-q` this page itself prescribes that message never prints -- you get a **0-byte**
`perf.data` at exit 0. Adding `-m 8` clears the mapping error and loses samples instead:
`dwarf,16384 -m 8` -> `Processed 84 samples and lost 16.67%`; `dwarf,32768 -m 8` and
`dwarf,65528 -m 8` -> `lost 100.00%`, still exit 0, a 21 KB file with no samples in it. So raise the
dump size only together with `-m`, and check `perf report --stats` for lost events before reading
the profile. Raising `perf_event_mlock_kb` is the real fix and it is a root write: ASK THE USER,
do not assume it.

A SECOND cut-off is independent of all of that and no `stack-size` reaches it: `perf report
--max-stack` and `kernel.perf_event_max_stack` both default to 127 frames. Keep the `[unknown]`
entries in whatever you fold: dropping a frame silently re-parents its callees and invents a call
path that never happened.

**Check for lost events before trusting any dwarf recording.** `perf record` prints
`WARNING: LOST n chunks, Check IO/CPU overload` when it cannot keep up; `perf report --stats`
reports `INFO: x.xxx% lost events (n out of m, in k chunks)`; `perf script --show-lost-events` puts
them inline. dwarf at 999 Hz is the standard cause -- 8.5 KB per sample has to reach the disk. Loss
is not uniform over the run, so a lossy profile is BIASED, not merely small, and a 100%-lost one
still exits 0.

**A stripped `.so` still profiles -- as long as you only need the exported symbol.** The kernel
entry point lives in `.dynsym`, which `strip --strip-all` does not remove: measured, a fully
stripped library still reported `kernel_fp64` at 99.45%. Its `static` helpers live in `.symtab` and
are gone, and perf prints raw addresses like `0x0000000000001196` for them, one entry per address
rather than one per function.

**A separate debug file must sit where the debuglink points.** perf does follow `.gnu_debuglink`.
Measured on the same stripped library: with `libk.so.debug` beside `libk.so` the static symbol
resolved (98.22%), with it in `.debug/` beside the library it resolved (99.52%), and with the file
moved elsewhere perf fell back to raw addresses. Copying the debug file next to the library is the
whole fix. Recording on a cluster node and reading on your box is a different fix: `perf record
--buildid-all` then `perf archive`, which resolves through the `~/.debug` build-id cache instead.

**C++ names.** perf demangles by default in both `report` and `script` -- you get
`void kern::axpy<double>(double*, double const*, unsigned long)`, not `_ZN4kern4axpyIdEEvPT_PKS1_m`.
If you see `_ZN`, something passed `--no-demangle` or the text came from a tool that does not
demangle; pipe it through `c++filt`.

**Inlining moves the blame.** At `-O3` a hot leaf is credited to whatever inlined it, so a
suspiciously large function is usually several. Inline frames are shown by DEFAULT in both `report`
and `script`; `--no-inline` is what suppresses them, and it is the fast reading because it keeps
one sample on one symbol.

**A sampled IP is skidded.** `cycles:u` is not a precise event: the recorded instruction pointer can
sit some way past the instruction that cost the cycles. Symbol ranking survives that, per-line
attribution does not, so do not read `perf annotate` as truth off an imprecise event. Precise mode
(PEBS on Intel, IBS on AMD) bounds the skid.

**On AMD, `cpu/caps/max_precise = 0` does NOT mean precise sampling is unavailable.** That cap
describes the CORE PMU, which is inherently non-precise; the kernel FORWARDS `cycles:p` to the IBS
Op PMU (`ibs_op//`), whose rip has skid ZERO -- "The rip of IBS samples has skid 0"
(`arch/x86/events/amd/ibs.c`) -- so on Zen4 `perf annotate` IS readable if you paid for a root
`cycles:p` run. The availability check is `ls /sys/bus/event_source/devices/ | grep ibs` and
`cat /sys/bus/event_source/devices/ibs_op/caps/zen4_ibs_extensions` (present, and 1, on this box),
not `max_precise`. Two prices. IBS has no user/kernel filtering, so it needs root or `CAP_PERFMON`
and `perf_event_paranoid=0` is not enough. And `cycles:up` -- the user-only spelling -- is REJECTED
before Zen6; the documented form is `perf record -e ibs_op/swfilt=1/u`. IBS is also count-driven:
give it `-c <count>`, not `-F`.

**A sample count is a sample count.** The relative standard error of a frame holding k samples is
about 1/sqrt(k) OF ITS OWN COUNT: 100 samples is +/-10% of 100, not +/-10 points of the profile;
10 samples is +/-32% of 10. In the 440-sample profile above a 1% entry is four samples, which is
noise wearing a percentage. Do not rank two frames that are a few samples apart -- record longer
instead, and use `--percent-limit 1` to stop printing the noise floor.

**`# (Cannot load tips.txt file, please install perf!)` at the foot of every `--stdio` report is
cosmetic.** Ubuntu's perf package ships without the tips file. Nothing about the profile is wrong;
do not spend a step debugging the install.

**A profile says where the time WENT.** It never says what would be faster. That is a hypothesis
you form from it and then measure, one change at a time, on the same box at the same thread count.

## Documentation

- perf wiki, tutorial and man pages -- https://perf.wiki.kernel.org/index.php/Main_Page
- `perf record` flags, including every `--call-graph` mode, `--strict-freq`, `--buildid-all` -- https://man7.org/linux/man-pages/man1/perf-record.1.html
- `perf report` -- `-g` print types, `--inline`, `--max-stack`, `--percent-limit`, children over 100% -- https://man7.org/linux/man-pages/man1/perf-report.1.html
- `perf stat` -- the `Default` metric group and its thresholds, `--null`, `-r`, `--table`, `-D`, `--topdown` -- https://man7.org/linux/man-pages/man1/perf-stat.1.html
- `perf diff` -- `Delta` as a share difference, `--period`, `-c ratio`, `--baseline-only` -- https://man7.org/linux/man-pages/man1/perf-diff.1.html
- `perf mem` -- latency-weighted overhead, `symbol_daddr`, per-arch events -- https://man7.org/linux/man-pages/man1/perf-mem.1.html
- `perf c2c` -- Shared Data Cache Line and Pareto tables, arch support -- https://man7.org/linux/man-pages/man1/perf-c2c.1.html
- `perf amd-ibs` -- `cycles:p` forwarding, `swfilt`, `l3missonly`, root requirement -- https://man7.org/linux/man-pages/man1/perf-amd-ibs.1.html
- perf wiki tutorial -- multiplexing, scaling as an ESTIMATE, `--no-scale` -- https://perfwiki.github.io/main/tutorial/
- Brendan Gregg, off-CPU analysis -- what a cycles profile cannot see, and the tracepoint fallback -- https://www.brendangregg.com/offcpuanalysis.html
- LBR depth per microarchitecture -- `lbr_nr` in the kernel's `intel_pmu_lbr_init_*` -- https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/arch/x86/events/intel/lbr.c
- Brendan Gregg, perf examples (the most practical reference for this tool) -- https://www.brendangregg.com/perf.html
- Brendan Gregg, CPU flame graphs -- how to read one, and what the axes do NOT mean -- https://www.brendangregg.com/FlameGraphs/cpuflamegraphs.html
