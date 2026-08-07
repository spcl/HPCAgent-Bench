---
name: linuxperf
description: Finds where CPU time went with linux perf -- record, self vs children, flat kernels, unwind modes, traps. Not counters, not GPU.
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
   ```

   Everything below samples `cycles:u`, so it is blind to every cycle spent in the kernel. If
   `cycles:k` is a large share of the total, the report you are about to take describes a MINORITY
   of the wall clock and the target is off-CPU -- the allocator, page faults, syscalls -- not any
   frame that will appear in it. Measured on `harris_corner` at preset S: `cycles:u` 3.51 G
   (22.4%), `cycles:k` 12.14 G (77.6%), 5,049 page faults per rep. A user-mode profile of that run
   puts 72% self on the kernel symbol and points at the loops; the actual win was 6.3x from
   hoisting ten per-call `malloc`/`free` temporaries out of the hot path, which `cycles:u` cannot
   see at all. Fix that first, then record.
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
4. **Read children to find who is responsible.** Walk down from a high-children/low-self caller to
   the first frame whose body IS the algorithm rather than dispatching, packing or copying.
5. **Only then ask what the machine was doing there.** That is a hardware counter bracket, not a
   sampler.

**Unresolved `[k]` hex addresses are kernel frames**, not a broken unwind -- `kptr_restrict`
withheld the symbols. They are a different failure from `[unknown]` (a truncated DWARF stack, fixed
with a bigger `--call-graph=dwarf,N`) and the fix is not the same. Their share is a LOWER BOUND on
kernel-mode time: more than a few percent means go back to step 0 and count `cycles:k`.

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
and it can equally invent a plausible wrong chain. Neither errors. `record -- cmd` samples the
command AND its descendants,
so a runner that forks the measured child is still profiled. `cycles:u` is user-space only; kernel
samples need a lower `perf_event_paranoid` and answer a different question, so perf's
`kptr_restrict` warning at record time is NOT harmless -- it is why kernel frames come back as
bare `[k]` hex, and on a kernel whose cost is off-CPU it is the only thing pointing at that. See
step 0. `-F 999` rather than 1000 so the sampler
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

Submit the version without it -- or keep it only if you measured the cost as
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
| `--call-graph=fp` | near free | every frame kept its frame pointer | truncating, or inventing a plausible wrong chain |
| `--call-graph=dwarf` | the expensive one | AOT build with `.eh_frame`, to the dump size | `[unknown]` past the copied stack |
| `--call-graph=lbr` | cheap, most accurate | Intel, and only to LBR depth | silently truncating past LBR depth |

The overhead column is qualitative; no upstream doc puts numbers on it. `dwarf` also needs a perf
linked against libunwind or libdw, and has nothing to unwind for a JIT frame (numba, JVM, V8).

**Reach for `dwarf`.** It is the only one that is right on a build you did not compile yourself,
which includes libc, CPython and every BLAS. `fp` is wrong there and does not say so: measured on
this box, an `fp` unwind of a two-phase C program produced
`phase_axpy <- call_init (inlined) <- __libc_start_main_impl <- _start`, with `main` missing and a
frame that never ran in its place. The dwarf unwind of the same binary gave the real chain.

The price of `dwarf` is that every sample copies the full `stack-size`, however shallow the stack
really was: measured, 8.5 KB of `perf.data` per sample at the default 8192 and 66 KB at
`dwarf,65528`. Multiply by samples ACTUALLY taken, not by wall clock -- a fully user-bound run at
999 Hz costs ~8 MB/s at the default, a half-user-space run half that. Same 0.44 s workload, three
`perf.data` files: `fp` 35 KB, `dwarf` 1.9 MB, `dwarf,65528` 13 MB.

**LBR is not available on this box.** `--call-graph=lbr` asks the PMU for branch-stack call-stack
mode, which is an Intel LBR feature; on Zen4 (`amd_lbr_v2`) perf refuses with `cycles:uH: PMU
Hardware or event type doesn't support branch stack sampling`. Plain branch records still work
(`perf record -e cycles:u -b`), but they are branch history, not a call graph. Where LBR does work,
the hardware buffer holds 16 entries (Nehalem through Broadwell) or 32 (Skylake and later); 8 is
Atom/Silvermont. Past that depth children time is meaningless rather than approximate.

## Traps

**`[unknown]` frames mean the unwind stopped, not that nothing ran.** DWARF copies at most
`stack-size` bytes per sample, 8192 by default; a deeper stack is silently cut off. Raise it:
`--call-graph=dwarf,65528` -- 65528 is the maximum, and perf rejects more with `callchain:
Incorrect stack dump size (max 65528)`. That is ~66 KB of `perf.data` per sample, so size the file
before you record long. A SECOND cut-off is independent of it and no `stack-size` reaches it:
`perf report --max-stack` and `kernel.perf_event_max_stack` both default to 127 frames. Keep the
`[unknown]` entries in whatever you fold: dropping a frame silently re-parents its callees and
invents a call path that never happened.

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
attribution does not, so never read `perf annotate` as truth. Precise mode (`cycles:up`, PEBS on
Intel, IBS on AMD) bounds the skid, and is not always there -- `max_precise` under
`/sys/bus/event_source/devices/cpu/caps/` reads 0 on this box.

**A sample count is a sample count.** The relative standard error of a frame holding k samples is
about 1/sqrt(k) OF ITS OWN COUNT: 100 samples is +/-10% of 100, not +/-10 points of the profile;
10 samples is +/-32% of 10. In the 440-sample profile above a 1% entry is four samples, which is
noise wearing a percentage. Do not rank two frames that are a few samples apart -- record longer
instead, and use `--percent-limit 1` to stop printing the noise floor.

**A profile says where the time WENT.** It never says what would be faster. That is a hypothesis
you form from it and then measure, one change at a time, on the same box at the same thread count.

## Documentation

- perf wiki, tutorial and man pages -- https://perf.wiki.kernel.org/index.php/Main_Page
- `perf record` flags, including every `--call-graph` mode, `--strict-freq`, `--buildid-all` -- https://man7.org/linux/man-pages/man1/perf-record.1.html
- `perf report` -- `-g` print types, `--inline`, `--max-stack`, `--percent-limit`, children over 100% -- https://man7.org/linux/man-pages/man1/perf-report.1.html
- LBR depth per microarchitecture -- `lbr_nr` in the kernel's `intel_pmu_lbr_init_*` -- https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/arch/x86/events/intel/lbr.c
- Brendan Gregg, perf examples (the most practical reference for this tool) -- https://www.brendangregg.com/perf.html
- Brendan Gregg, CPU flame graphs -- how to read one, and what the axes do NOT mean -- https://www.brendangregg.com/FlameGraphs/cpuflamegraphs.html
