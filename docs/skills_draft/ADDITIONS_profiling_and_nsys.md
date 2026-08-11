# Additions for the two EXISTING skills

`hpcagent_bench/skills/profiling/SKILL.md` (357 lines) and `hpcagent_bench/skills/nsys/SKILL.md`
(298 lines) already exist and are heavily pinned by `tests/test_skill_content.py`. These are the
blocks to MERGE INTO them, not replacements. Sources at the bottom.

---

## For `profiling` (linux perf): lead with the hottest function

The current page teaches the instrument. It does not say plainly what to do with the output
first. Add this near the top, before the metric tables:

> ## Find the one function that owns the time
>
> A profile has one job before any other: name the function you should be editing. Everything
> else on this page is for after you know that.
>
> ```sh
> perf record -F 99 -g -- ./your_run          # -g is not optional: no -g, no call graph
> perf script -q -F comm,ip,sym,dso --no-inline
> perf report --stdio --sort=symbol           # ranked, self time first
> ```
>
> Read the ranked list top-down and stop at the first function that is yours. Two columns and
> they answer different questions:
>
> - **Self (exclusive)** -- time in that function's own instructions. This is the one that tells
>   you where to edit.
> - **Children (inclusive)** -- that function plus everything it called. High children with low
>   self means the work is deeper; follow it down rather than editing here.
>
> The decision rule: if the top self-time function is below ~30% of the run, optimizing it cannot
> give you more than a 1.4x speedup no matter how well you do it. Look for a flatter problem --
> or accept that the win is structural (fewer calls, a different algorithm) rather than local.
>
> **C++ names come out mangled.** `perf report` demangles by default; `perf script` may not.
> Pipe through `c++filt` if you see `_ZN...`. A profile you cannot read the names of is a profile
> you will misattribute.
>
> **Inlined functions do not appear.** `--no-inline` is fast but attributes inlined work to the
> caller; at `-O3` that means the hot leaf may be reported as the function that inlined it. If a
> hot function looks implausibly large, that is why.

Then the flame-graph block, which is what the text output is a projection of:

> ## Reading a flame graph, in text
>
> If you generate one (`perf script | stackcollapse-perf.pl | flamegraph.pl > out.svg`), read it
> by these rules -- they are the same rules that make the text report meaningful:
>
> - **Width = cumulative time on CPU.** Widest box at any level is the biggest consumer. Width can
>   come from one slow call or many fast ones; the graph does not distinguish them.
> - **Y-axis = stack depth.** The TOP box is what was actually on-CPU. Everything under it is
>   ancestry, not cost of its own.
> - **X-axis means nothing.** It is not time. Frames are sorted alphabetically to merge boxes.
>   Left-to-right ordering carries no information at all -- do not read it as a sequence.
> - **A wide plateau** is sustained time in one function or chain: the thing to optimize.
>   **A tall narrow tower** is a deep call stack that costs almost nothing: ignore it.
> - **Broken stacks** come from frame-pointer omission. The harness's profiled build is `-g` only,
>   which keeps line info; if stacks look truncated, that is the cause, and the fix is a build flag
>   you should not be adding to a scored submission.

## For `nsys`: what to do with the timeline first

> ## The first three numbers, in order
>
> 1. **Was the GPU busy at all?** `device_pct` -- device time over wall clock. Below ~50% the
>    kernel is not your problem: the host is. Fix the launch pattern or the transfers first,
>    because making a kernel faster cannot fill a gap where the GPU was idle.
> 2. **Which kernel owns device time?** The kernel summary, sorted by `total_ns`. Use `total_ns`,
>    not `mean_ns`: a 5 us kernel launched 200,000 times beats a 50 ms kernel launched once.
>    `launch_count` next to it is what tells you which of those you have.
> 3. **What is between the kernels?** Gaps on the timeline are the finding, not the background.
>    A gap is one of: the host was computing, the host was blocked on a sync, a transfer was in
>    flight, or launch overhead dominated because the kernels are too small. The API trace tells
>    you which. Kernels that are short AND gappy mean fuse them or raise the work per launch --
>    not micro-optimize the body.
>
> Occupancy is the one number nsys does not have. It hands that question to `ncu`.

---

## The core prompt must describe the profiling skills, not inline them

MEASURED, today, `hpcagent_bench/harness/prompts/sections/skills.j2`:

```
skill            body lines
loopnest                 17
memory                   17
nsys                    293
opt-reports             173
parallelism              17
profiling               352
rocprof                 263
vectorization            19
TOTAL                  1169 lines into EVERY prompt
```

`sections/skills.j2` is included unconditionally from `task.j2` and inlines **every skill's full
body**. The four profiling pages are **1081 of those 1169 lines -- 92%** -- and they are in the
prompt of every agent whether or not it ever profiles. Adding `papi-standalone` (142) and
`papi-counters` (~165) takes it to ~1476, of which ~1388 is profiling.

The index already exists and is the right shape:

```jinja
## Skills
Focused guides for the transforms below. Each is a self-contained note; use the one that
matches what the profile says is slow.
{% for skill in other_skills %}
- **{{ skill.name }}** -- {{ skill.description }}
{% endfor %}
```

So the fix is to stop inlining the profiling bodies unconditionally:

1. Name the set in `prompts.py`, next to `GENERAL_SKILL`:
   ```python
   #: Skills whose BODY is inlined only when profiling is enabled. Each is a long instrument
   #: manual, and an agent that never profiles pays for all of them in every prompt.
   PROFILING_SKILLS = frozenset({"profiling", "nsys", "rocprof", "opt-reports",
                                 "papi-counters", "papi-standalone"})
   ```
2. `build_context` passes `profiling: bool` -- true when the strategy is `profile_first`, or when
   a `prompt.profiling` config knob asks for it.
3. `skills.j2` keeps the index line for EVERY skill (that is what makes a skill discoverable), and
   inlines the body only for `skill.name not in PROFILING_SKILLS or profiling`.

The index line then has to carry its own weight, because with profiling off it is all the agent
gets. Each must say the INSTRUMENT and the QUESTION, in one line:

- **profiling** -- where the time went on the CPU (`perf` call graph) and what the machine did to
  spend it (PAPI counters). Start here; it routes to the others.
- **papi-counters** -- hardware counters through the judge: one call, one run per counter, ratios back.
- **papi-standalone** -- counters for ONE region of your own source, via the header-only helper.
- **opt-reports** -- what the compiler did and did not do, and whether a refusal was legality or cost.
- **nsys** -- which CUDA kernel and which copy owns device time, and whether the GPU was busy at all.
- **rocprof** -- the same question on AMD, where the tool names and the lane width are different.

Cheaper variant if the template change is unwanted: keep inlining, but ship `nsys` and `rocprof`
only when the target actually has that vendor's GPU. That saves 556 lines on a CPU run and needs no
gating flag -- but it makes the prompt depend on the judge's hardware, which is a property the
prompt does not otherwise have. The gated version above is the better structure.

## Sources

- Brendan Gregg, *CPU Flame Graphs* -- https://www.brendangregg.com/FlameGraphs/cpuflamegraphs.html
  (width = cumulative on-CPU time; y = stack depth, top box is on-CPU; x-axis is alphabetical and
  carries no time ordering; plateaus vs towers; broken stacks from frame-pointer omission; inlining
  removes frames)
- Brendan Gregg, *Flame Graphs* index -- https://www.brendangregg.com/flamegraphs.html
- NVIDIA, *Nsight Systems Post-Collection Analysis Guide* --
  https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html
  (using CPU sampling and OS-runtime blocked-state backtraces to explain gaps between kernels;
  NVTX annotation to attribute them)
- Modular, *GPU profiling with Nsight Systems* -- https://docs.modular.com/gpu-system-profiling/
  (start with nsys for orientation: where the GPU is busy, where it stalls, which kernels dominate)
- TU Dresden ZIH, *Read CPU Performance Counters with PAPI* --
  https://compendium.hpc.tu-dresden.de/software/papi/
- PAPI preset event reference (PAPI_L1_DCM / L1_ICM / L2_DCM / L3_TCM, MFLOPS, IPC) --
  https://en.wikipedia.org/wiki/Performance_Application_Programming_Interface
