# Profiling skills -- delivery order

Drafts. NOT shipped: these live here rather than in `hpcagent_bench/skills/` because a skill that
tells every agent to include a header which does not exist yet is worse than no skill. One `mv` per
directory once the thing it documents is real.

## The order is a requirement, not a preference

**ALL STANDALONE SKILLS SHIP FIRST. No judge-based skill ships until the standalone set is DONE.**

Standalone means: the agent runs the instrument itself, in its own container, and interprets the
output itself. Judge-based means: the agent asks the judge and reads a report the judge produced.

Why the order matters and is not arbitrary:

- The standalone path is the one that works with no judge, no network and no harness. It is the
  floor. If it is not solid, the judge path is a convenience layered on a gap.
- The judge path's report is derived from the SAME tables (`papi.RATIOS`, `papi.METRICS`) and
  teaches the SAME interpretation. Writing the standalone page first forces the interpretation to
  be written once, in the place that cannot delegate it; the judge page then routes to it instead
  of restating it.
- Building the judge path first would let the interpretation live only inside the judge's rendered
  output, and the standalone page would end up a thin command reference with the reasoning missing.

## Phase 1 -- STANDALONE (all of these before anything in phase 2)

| skill | what the agent runs itself | status |
|---|---|---|
| `papi-standalone` | the header-only helper: `papi_init` / `start` / `stop` / `finalize` | DRAFT WRITTEN -- blocked on the header existing |
| `perf-standalone` | `perf record -g`, `perf report`, call graph and flame graph, by hand | TO WRITE -- extract from the existing `profiling` skill |
| `nsys-standalone` | `nsys profile` / `nsys stats` on its own build | TO WRITE -- existing `nsys` skill is already mostly this |
| `rocprof-standalone` | `rocprofv3` on its own build | TO WRITE -- existing `rocprof` skill is already mostly this |
| `opt-reports` | compiles with the report flags and reads the report | ALREADY STANDALONE -- audit only |

The existing `profiling` skill is MIXED: it teaches the `perf` commands (standalone) and the judge
counter endpoint (judge-based) on one page. Splitting it is part of phase 1 -- the standalone half
becomes `perf-standalone`, and the judge half waits for phase 2.

Every phase-1 page must carry, in full, and not by reference to a judge report:
- always run the kernel, and how to tell a partial execution from a fast one,
- how to compare two metrics (same run vs different runs),
- which direction is better for each quantity, and the conditions under which that is meaningful,
- how to read the tool's own output format (ranked self time, flame graph, timeline).

## Phase 2 -- JUDGE-BASED (BLOCKED until phase 1 is done)

| skill | what the judge does | status |
|---|---|---|
| `papi-counters` | judge runs the kernel once per counter, returns ratios | DRAFT WRITTEN -- **HELD**, do not ship before phase 1 |
| judge call graph | judge returns the `perf` profile | folded into `papi-counters` for now |

A phase-2 page is short by construction: the request, the response shape, and a pointer to the
phase-1 page for what the numbers mean. If a phase-2 page needs to explain a ratio, that
explanation belongs in phase 1 and is missing there.

## FINAL TARGET SHAPE -- 2026-08-02, supersedes the "collapse to 5" table below

One skill per INSTRUMENT. The earlier plan merged the three profilers into one page; that is
withdrawn -- an agent has one machine and one vendor, and a merged page makes it read two vendor
manuals it cannot use.

| skill | instrument | question it answers | state |
|---|---|---|---|
| `general` | -- | what is LEGAL (the contract) | exists, 23 lines, shrink to the contract paragraph |
| `optimization-hints` | -- | what transform to try, in what order | IN PROGRESS: merging the four stubs |
| `opt-reports` | the compiler | what the compiler did and refused, and whether that was legality or cost | exists, 173 lines, audit only |
| `linuxperf` | `perf` | which function owns the CPU time | IN PROGRESS (drafted as `perf-standalone`) |
| `papi-cpu` | PAPI | what the CPU did while it ran | IN PROGRESS (drafted as `papi-standalone`) |
| `papi-gpu` | PAPI | what the GPU did while a kernel ran | TO WRITE: one region per kernel, sync both sides |
| `nsys` | Nsight Systems | which CUDA kernel and copy owns device time, and whether the GPU was busy | exists, 293 lines, audit + resplit |
| `ncu` | Nsight Compute | what the SMs did inside ONE kernel | TO WRITE |

### Every instrument skill has TWO VARIANTS

Not two instruments -- two ways to reach the same instrument, and they differ ONLY in who runs it
and how the output comes back.

**Variant 1 -- self-service.** The agent runs the tool itself, in its own container. The repo
provides what it needs to do that (the PAPI header, the event list, the command lines). Nothing
leaves the agent's machine. This is the floor: it works with no judge and no network.

**Variant 2 -- agent instruments, judge executes.** The agent instruments its source however it
sees fit and submits the instrumented artifact. The judge runs it and returns the output. Two
requirements that make this work, and both belong ON THE PAGE:

- **The page must print the EXACT command the judge will run.** Not a description of it. The agent
  has to be able to predict what comes back, or it is instrumenting blind and reading a format it
  did not expect.
- **The instrumented artifact writes its profile to STDOUT**, so the judge redirects stdout
  straight into the response. That makes the contract one line and needs no side-channel file, no
  agreed path, and no cleanup.

The stdout contract has two consequences the pages must state plainly, or a submission produces a
response the judge cannot parse:
- The kernel itself must print NOTHING. Any stray printf lands in the middle of the profile.
- The profile must be self-delimiting, so a partial or truncated run is detectable rather than
  silently parsed as a complete one.

**Variant 2 is a SPECIALIZATION of variant 1: the same page, with the EXECUTION section swapped for
"delegate to the judge". Nothing else differs.**

Which part of the kernel to bracket, how to read an IPC, which direction is better, why two counts
from different runs need a shared denominator -- none of it depends on who pressed the button, so
none of it is rewritten. Variant 2 is not a shorter page that points at variant 1; it is the SAME
page, complete and readable on its own, with one section replaced:

| section | variant 1 | variant 2 |
|---|---|---|
| what the instrument answers | identical | identical |
| where to put the region | identical | identical |
| how to read the numbers | identical | identical |
| comparing two metrics | identical | identical |
| direction-of-goodness | identical | identical |
| traps | identical | identical |
| **HOW IT RUNS** | you compile and run it yourself | you instrument, the judge runs it, output comes back on stdout -- with the curl form, the `JudgeClient` form, and the exact judge command |

**Therefore the shared sections must be BYTE-IDENTICAL and a test must pin that.** Two hand-written
twins drift silently, and a drifted pair is worse than either page alone -- one of them is then
teaching something the other contradicts. Either generate variant 2 from variant 1 with the
execution section substituted, or write both and add an assertion to
`tests/test_skill_content.py` that every shared section matches exactly between the pair. The
generated route is better: it makes drift impossible rather than merely detectable.

This also settles the variant-1 purity rule above. Because the shared text is literally the same
bytes, it CANNOT mention the judge, `JudgeClient`, `/app/...` or `hpcagent_bench` -- anything
repo-specific has to live in the execution section, which is the only part that differs. The rule
stops being a style guideline and becomes a mechanical consequence of the structure.

### Every skill links to its tool's official documentation

MEASURED: zero of the 17 skill files, draft or shipped, contains a single URL. That is the gap this
rule closes.

The distinction that matters, since it looks like it contradicts the no-cross-reference rule:

- **Never link to another SKILL PAGE.** With body gating on, that page may not be in the prompt at
  all, so the pointer resolves to nothing. Inline the fact instead.
- **Always link to UPSTREAM DOCUMENTATION.** A vendor doc URL is stable, always reachable, and is
  the only honest way to say "this page summarises; the authority is there". It also gives a reader
  somewhere to go when the page is wrong -- which it eventually will be, because tools change and
  a skill file does not.

Each page ends with a short `## Documentation` block: the tool's own reference, plus any single
page that is genuinely worth reading in full. Not a bibliography -- three or four links, each one a
reader would actually open.

A link earns its place by answering a question the page deliberately does NOT: the full flag
reference, the complete metric list, the vendor's own troubleshooting page. A link to a blog post
that says what the page already says is padding.

The review pass currently verifying every claim against upstream docs is collecting exactly these
URLs. Fold in whatever it returns, since those are the pages that actually settled a question.

### The line between the two variants -- a DEFECT in the current drafts

The drafts have drifted across this line and must be corrected.

**Variant 1 assumes an ARBITRARY AGENT that can compile and run its own code. Nothing else.**
It is the general case, not the HPCAgent-Bench case. A variant-1 page may assume: a compiler, a
shell, and the source it is optimizing. It may NOT mention `/app/<kernel>/reference.py`,
`signature.json`, `JudgeClient`, `hpcagent_bench`, `grading._data_seeded`, a judge URL, a rank, or
any container layout. If the page names a path only this repo has, it has failed -- someone outside
this repo must be able to follow it start to finish.

That has a consequence for the input rule. Variant 1 cannot say "measure with the inputs the judge
grades you on", because a general reader has no judge. It says the generic version: build the
buffers ONCE and use the same ones for the counted run and the correctness check, and understand
that counts taken on data you invented describe the workload you invented.

CURRENT DEFECTS in `papi-standalone`: the opening paragraph routes the reader to
`JudgeClient.profile(sub, kernel, counters=True, counter_group="overview")`, and the input section
cites `/app/<kernel>/reference.py`, `signature.json` and `grading._data_seeded`. All of it moves to
the variant-2 page. Same audit needed on `perf-standalone` once it lands.

**Variant 2 is the HPCAgent-Bench page, and it is where the judge lives.** It must show BOTH call
forms, the way `hpcagent_bench/tools/counters.md` already does for the existing endpoint:
- the raw HTTP call -- a `curl -X POST {{ judge_url }}/profile` line with the real JSON body
- the Python call -- `JudgeClient("{{ judge_url }}", rank={{ judge_rank }}).…` with the real
  arguments
plus the exact command the judge will run on the submitted artifact, and the stdout contract. Read
`counters.md` for the house style and match it; do not invent a third way of documenting a judge
call.

### Which skills need TWO variants, and which need one

The rule is not per-skill taste. **A tool that RUNS the kernel needs a judge variant. A tool that
only reads or compiles the SOURCE does not.**

A runtime instrument produces a different answer on a different machine, so who executes it is a
real question: the agent's container and the judge's node have different CPUs, different GPUs,
different counter availability and different permission gates. A compile-time tool produces the
same answer wherever it runs, because its input is the source and its output is the compiler's
opinion. Shipping it to a judge buys nothing and costs a round trip.

| skill | variants | why |
|---|---|---|
| `linuxperf` | 2 | runs the kernel; sampling is machine-specific |
| `papi-cpu` | 2 | runs the kernel; counter availability is per-CPU |
| `papi-gpu` | 2 | runs the kernel; counter availability AND the driver gate are per-box |
| `nsys` | 2 | runs the kernel on a device the agent may not have |
| `ncu` | 2 | same, and the profiling permission gate is the usual blocker |
| `opt-reports` | **1** | COMPILE-time. The compiler's verdict on the agent's own source is the same verdict anywhere. |
| `static-analysis` | **1** | COMPILE-time, same reason. clang-tidy and cppcheck read source, they do not run it. |
| `optimization-hints` | 1 | not an instrument; nothing executes |
| `general` | 1 | the contract |
| `pytorch-to-numpy` | 1 | a porting task, verified against torch locally |

So five instruments x 2 = 10 pages, plus 5 single pages = 15 skill files total.

Every one of them, both variants, carries the `## Documentation` block. The links are identical
between a v1/v2 pair -- same tool, same upstream -- which is consistent with the byte-identical
rule: the doc block is shared text, not execution text.

**Both variants exist as their OWN FILES** -- ten instrument pages, not five with two sections.
The interpretation-lives-once rule above is a structural instruction for HOW to write the pair, not
permission to merge them: the variant-2 page states its execution contract in full and then points
at its variant-1 sibling by name for the reading, rather than restating it.

### perf and PAPI are COMPLEMENTARY -- both ship, and both pages say how they compose

They answer different questions with different mechanisms, and neither substitutes for the other.

**This table goes at the TOP of both `linuxperf` and `papi-cpu`, immediately after the frontmatter,
before anything else.** It is the summary a reader needs before they can decide whether they are on
the right page at all, and a reader who has one instrument never reaches for the other unless the
first thing they see says so. Verbatim on both pages, so the two cannot drift.

|  | `linuxperf` | `papi-cpu` |
|---|---|---|
| answers | WHERE the time goes | WHY it is slow there |
| mechanism | statistical sampling of the call stack | exact hardware counts over a bracket |
| needs a code change | no | yes -- a start/stop bracket |
| granularity | whatever is a symbol | whatever you bracket |
| main failure | too few samples (a flat or noisy profile) | too short a region (measuring the instrument) |
| perturbs the run | barely | yes -- never compare a counted run's wall clock |

**Normal order: perf first, PAPI second.** perf is free and needs no edit, and it tells you which
region is worth counting. Counting a region that owns 5% of the time is a wasted run whatever the
counters say.

**The inversion, which is the common case on this corpus.** The generated kernels are ONE flat
function, so perf has a single symbol and cannot localize inside it. There the order flips: bracket
the phases with PAPI to find which one owns the cycles, THEN promote that phase to a
`__attribute__((noinline))` function so perf can show you its call graph and its libc children.
PAPI localizes, perf explains -- the opposite of the usual direction, and a page that only teaches
the usual direction leaves the reader stuck on every kernel in the corpus.

**Where each is the only answer.** perf alone finds work outside your kernel (the cavity_flow run
was 64% interpreter and import) and names a libc callee you never wrote (the memmove that was a
third of kernel time). PAPI alone gives per-thread imbalance, cache and branch behaviour, and the
roofline position -- none of which a sampled call graph can express.

### Region selection -- REQUIRED content on both CPU pages

The hardest part of either instrument is not the invocation, it is deciding WHERE to measure. On a
flattened kernel (one function, no internal symbols) a reader with no guidance brackets the whole
thing and learns nothing. Both pages must name the candidates outright.

**`papi-cpu` -- bracket TOP-LEVEL LOOPS and PARALLEL REGIONS.**
- The outermost loop of each phase. It is the unit a transform actually changes, and because
  start/stop accumulate, a phase costing 20 us per iteration over 500 iterations clears the ~10 ms
  floor that a single visit never would.
- Every `#pragma omp parallel` / `parallel for`. Two reasons, and both are specific to counters
  rather than to timing: thread imbalance and false sharing only exist inside a parallel region and
  are invisible outside it, and the counters are PER-THREAD, so a region boundary that matches the
  team boundary is the only one whose per-thread numbers mean anything. A bracket that spans a
  team's creation counts threads that did not exist for all of it.

**`linuxperf` -- promote the suspected region to a FUNCTION.**
perf attributes to symbols, so a region only becomes visible by becoming a symbol:
`__attribute__((noinline)) static void phase_x(...)`. Prime candidates, in order:
- **Top-level loops** -- same phases as above, so the two instruments answer about the same units
  and their findings compose.
- **Branch arms.** Split the arms of a data-dependent branch into their own functions and perf
  tells you which arm is hot -- something counters cannot: a misprediction rate says the branch is
  unpredictable, not which side dominates. This is the one case where perf beats PAPI on a flat
  kernel.

Both pages point at `optimization-hints` for WHAT the phases are and which transform applies once a
phase is named. Neither restates it.

### Field test -- required before any of these ship

Once written, each skill is tested by a FRESH opus agent that has never seen this conversation,
given only the skill and a real kernel, on:
- a **GPU kernel**, and
- a **CPU kernel that genuinely has several functions** -- which is harder than it sounds, because
  the generated corpus references are flattened into one function. Either find a kernel whose
  source really does keep helpers, or the test is precisely whether the skill's
  `__attribute__((noinline))` guidance is enough to recover per-phase symbols.

The test is not "did the agent like the page". It is: following ONLY this page, did the agent reach
a correct finding about the kernel, and where did it get stuck or invent something the page did not
give it. A page that needs the reader to already know the answer has failed.

PARKED -- do not develop, do not rewrite:
- `amdprof` (rocprofv3, the AMD counterpart of `nsys`). The existing `rocprof` skill stays shipped
  as-is, 263 lines, untouched.
- the AMD counterpart of `ncu` (`rocprof-compute`). Not started.

The two drafts written under the old names get renamed on the way in: `perf-standalone` ->
`linuxperf`, `papi-standalone` -> `papi-cpu`. The frontmatter `name:` MUST equal the directory
name (pinned by `tests/test_skill_content.py`), so the rename is two changes, not one.

**This makes the prompt gating mandatory, not optional.** Five instrument pages inline into every
prompt. Today's four already cost 1081 lines; adding `papi-cpu`, `papi-gpu` and `ncu` while keeping
`rocprof` puts it well past 1600 -- in the prompt of every agent, on a box that has at most one of
the three vendors. Ship the gate WITH these pages.

## TARGET SHAPE -- superseded, kept for the reasoning

DECIDED 2026-08-02. The index an agent reads becomes five lines, and "which page do I open"
stops being a question it has to answer.

| skill | absorbs | today | target |
|---|---|---|---|
| `general` | the CONTRACT only -- what is legal, what you must not do | 23 | ~10 |
| `optimization-hints` | `loopnest` + `memory` + `parallelism` + `vectorization`, plus the generic transform bullets currently sitting in `general` | 70 + ~12 | ~60 |
| `opt-reports` | unchanged | 173 | 173 |
| `profiling` | WHERE THE TIME WENT, all three vendors: CPU `perf`, NVIDIA `nsys`, AMD `rocprofv3` | 352 + 293 + 263 | ~600 |
| `perfcounters` | WHAT THE MACHINE DID: PAPI on CPU and on GPU | (the counter half of `profiling`) | ~300 |

LATER, explicitly NOT the next task: `ncu` (NVIDIA per-kernel SM counters) and the AMD counter
equivalent. Do not start these.

The split between the last two is the one that matters and it is not by vendor -- it is by
QUESTION. `profiling` answers "which function or which kernel owns the time". `perfcounters`
answers "and what was the machine doing while it ran". An agent reaches for the first one first,
always; the second only after the first has named something.

`general` shrinks because its bulleted list of example transforms (dead-code elimination, LICM,
tiling, AoS/SoA, reassociation) is the same generic content as `loopnest` and `memory` and belongs
in `optimization-hints` with them. What must STAY in `general` is the contract paragraph: do not
change the signature, do not time inside the kernel, do not read or special-case the hidden inputs,
do not trade correctness for speed. That paragraph is what a submission is graded against.
`general` is structurally special -- `load_skills` returns it apart from the list, and
`optimization_guidance=False` drops every other skill while keeping it (pinned by
`tests/test_prompt_skills.py`). Do not merge it into `optimization-hints`.

Two tensions this creates, both solvable, both worth naming:

- **A merged `profiling` is ~600 lines and two thirds of it is a GPU manual the reader does not
  have the hardware for.** An agent on a CPU-only box would carry 556 lines about `nsys` and
  MI300 chiplets. This is why the merge only works TOGETHER with the gating change below: one
  page, three clearly-marked vendor sections, body inlined only when profiling is enabled.
- **`rocprof`'s 263 lines are mostly MI300-specific** (XCD chiplets, wavefront 64 against warp 32,
  KFD group permissions, the Omnitrace/Omniperf renames). That detail is load-bearing on AMD and
  noise everywhere else. Keep it as its own section with its own heading rather than blending it
  into a vendor-neutral narrative -- a reader on MI300 must be able to find it, and a reader on
  anything else must be able to skip it.

`tests/test_skill_content.py` currently pins about 25 assertions across `profiling`, `nsys` and
`rocprof` by SKILL NAME. Every one of those has to be repointed at the merged page. That is the
mechanical cost of this consolidation and it is the part most likely to be skipped.

## Scope decisions, 2026-08-02

- **`papi-counters` (judge) counts the WHOLE program, not regions.** No region API on that path at
  all. Regions are the standalone page's job, where the agent owns the source. That is why the two
  pages are not two spellings of one thing: whole-program from the outside, per-region from the
  inside. It also settles the section-0 fork in the design doc -- the judge path needs only the
  "whole intersection, library-driven" form.
- **`ncu` (CUDA hardware counters) is a TODO, not phase 1 or phase 2.** `nsys` answers which kernel
  and which copy owns device time; `ncu` answers what the SM did inside one kernel, and it is a
  separate instrument with a separate cost model (it replays a kernel many times). Write it after
  both phases, or not at all until something needs it. The `nsys` page already hands the occupancy
  question to `ncu --set full`, which is the right amount of coupling for now.
- **GPU PAPI: one region per kernel, with a device sync on both sides.** Unlike the CPU case there
  is no meaningful "whole program" device count -- launches are asynchronous, so a bracket that
  does not synchronise measures the launch, not the kernel. So: `cudaDeviceSynchronize()` before
  `start` and again before `stop`, one bracket per kernel launch. State plainly that the syncs are
  part of the measurement and that a synchronised run is not a timed run -- forcing the syncs
  removes exactly the overlap a real run depends on.

## Prompt cost

`sections/skills.j2` inlines every skill's full body into every prompt -- 1169 lines today, 92% of
it profiling. See `ADDITIONS_profiling_and_nsys.md` for the measurement and the gating fix. That fix
should land WITH phase 1, not after it: phase 1 roughly doubles the profiling text, and shipping
that unconditionally would put ~1400 lines of instrument manuals in the prompt of every agent that
never profiles.

## Files here

- five variant-1 instrument pages: `linuxperf/`, `papi-cpu/`, `papi-gpu/`, `nsys/`, `ncu/`
- five variant-2 twins: the same directory plus `-judge`, GENERATED from the variant-1 page with
  the `## How it runs` section substituted. That heading is the swap point on all ten pages, and
  `tests/test_skill_content.py` pins every OTHER section as byte-identical between a pair.
- `VARIANT2_judge_contract.md` -- the shared judge contract the five `-judge` pages implement, and
  the list of what the repo still has to build before any of them ships
- `papi-counters/` -- DELETED. It was an earlier hand-written draft of what is now
  `papi-cpu-judge`, under a name that does not fit the scheme.
- `ADDITIONS_profiling_and_nsys.md` -- merge blocks for the existing `profiling` and `nsys` skills,
  plus the prompt-gating measurement and design
