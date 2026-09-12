# DESIGN: `hpc_papi.h` -- region-level hardware counters for optimizing agents

This page describes a header-only C library that lets a program bracket one region of
its own source and read back hardware performance-counter values for it, and records
why the design looks the way it does.

**Status.** The four-call API is built and tested: `hpcagent_bench/helpers/papi/hpc_papi.h`
(generated), `hpcagent_bench/helpers/papi/header.py` (the generator and reader), and
`tests/test_papi_header.py` (pins the header to the tables it was generated from). It is
NOT wired into the harness: no build path adds its include directory, no submission
check refuses or strips it, and no agent-facing skill teaches it. It is a standalone,
tested tool today, not something a graded run can reach yet.

Today an agent's only counter surface is the judge's `POST /profile` with
`tool:"papi"`, which counts the WHOLE run from outside (see the `profiling` skill).
That cannot answer "which of my three loop nests is missing L2" or "did my tiling
actually raise arithmetic intensity in the hot region". `hpc_papi.h` exists to let a
program bracket a REGION inside its own source; wiring it into a graded run is future
work (see "Not built" below).

Grounded in `hpcagent_bench/harness/papi.py` (the metric/ratio/cause tables the header
is generated from and the reader re-uses) and the DaCe reference at
`dace/runtime/include/dace/perf/papi.h` (`spcl/dace`, branch `papi-fix-2`), which this
design borrows its OpenMP thread-registration and memory-fence handling from.

---

## 0. The API (the whole surface, as shipped)

Four calls, no named regions -- start/stop bracket THE region (the whole program
between init and finalize, or repeated pairs around one phase inside a loop, since
pairs accumulate):

```c
int  hpc_papi_init(void);      /* 0 = counting; <0 = degraded (the report says why).
                                 * Opens its own parallel region to register every
                                 * OpenMP thread -- call ONCE from serial code, and do
                                 * not wrap this call in a parallel region of your own. */
void hpc_papi_start(void);     /* begin counting on every thread */
void hpc_papi_stop(void);      /* end it; pairs accumulate across repeated calls */
int  hpc_papi_finalize(void);  /* write $HPC_PAPI_OUT (default ./hpc_papi.json);
                                 * 0 = a counted report, <0 = a degraded one */
```

Usage:

```c
#define HPC_PAPI_IMPLEMENTATION   // in EXACTLY one translation unit
#include <papi/hpc_papi.h>        // -I<repo>/hpcagent_bench/helpers

hpc_papi_init();
hpc_papi_start();
/* ... the region ... */
hpc_papi_stop();
hpc_papi_finalize();
```

`libpapi` is `dlopen`'d, so nothing goes on the link line: a host without PAPI still
compiles and still runs, degraded, with a named cause in the report. Never aborts,
never exits, never allocates inside a bracketed region, never touches a
floating-point value.

Read a report back with `python -m hpcagent_bench.helpers.papi --read hpc_papi.json`.
The header emits raw counts and no ratios; every division lives in
`hpcagent_bench.harness.papi.RATIOS` (Sec. 6).

Earlier drafts of this design proposed named regions (`hpc_papi_region` /
`hpc_papi_begin` / `hpc_papi_end`), a caller-supplied sweep driver
(`hpc_papi_sweep` + fill callbacks), and per-language source fragments. None of that
shipped; see "Not built" below for what a tagged-region extension would need.

---

## 1. Counter selection -- one armed set, budget-limited

At `hpc_papi_init`, in order (cheap checks first):

1. `__APPLE__` -> `not_linux`, stop.
2. Read `/proc/sys/kernel/perf_event_paranoid`: absent -> `no_perf_events`; too
   restrictive -> `perf_event_paranoid`. Checked before `dlopen`, because PAPI's own
   failure there is `PAPI_ESYS`, which reads like a broken install rather than a
   permission gate.
3. `dlopen("libpapi.so")`, falling back to `libpapi.so.<N>` -> `papi_missing`.
4. `PAPI_library_init` and `PAPI_thread_init` -- `papi_init_failed` on failure.
5. `budget = PAPI_num_cmp_hwctrs(0)` (overridable with `$HPC_PAPI_BUDGET`, for testing
   the packing).
6. For every metric in the generated table (15 metrics, from
   `hpcagent_bench.harness.papi.METRICS`), try each candidate expression in order and
   arm the first whose every event fits in the ONE remaining armed set. `cycles` and
   `instructions` claim their registers first, ahead of every other metric -- they are
   the denominators of nearly every ratio in `papi.RATIOS`, and a ratio whose
   denominator was never armed cannot be computed.
7. A metric that does not fit the budget is reported `count: null` with a `missing`
   reason naming it, plus a hint to re-run with `$HPC_PAPI_METRICS` narrowed to the
   metrics that matter for that region -- never substituted, never multiplexed (a
   multiplexed number is an estimate wearing a count's clothes).

There is exactly one armed set and one measured pass: `start`/`stop` bracket a region
of a program this header does not drive, so it cannot re-run the kernel to cover a
second batch of metrics automatically. Getting the rest of the intersection means
running the program again with a different `$HPC_PAPI_METRICS` selection.

`$HPC_PAPI_UNAVAILABLE` (comma-separated event names) forces specific events absent,
so the degradation ladder can be exercised on a machine that actually has them.

---

## 2. OpenMP thread registration

`hpc_papi_init` calls `PAPI_thread_init(omp_get_thread_num)` once, then opens
`#pragma omp parallel num_threads(...)` itself and registers each thread
(`PAPI_register_thread`, `PAPI_create_eventset`, `PAPI_add_named_event`) inside
`#pragma omp critical` -- the caller never opens a parallel region for this. A memory
fence (`_mm_mfence` on x86-64 with GCC, `__atomic_thread_fence(__ATOMIC_SEQ_CST)` on
aarch64) runs before every `start`/`stop`/read, so the helper's own stores cannot drift
across the region boundary and land inside the counts. aarch64 needs this as much as
x86-64: its weaker memory ordering permits MORE reordering across the boundary, not
less.

A failing per-thread `PAPI_add_named_event` degrades that thread's counts rather than
aborting the process; a thread that appears after `init` (`threads_moved`) or a run
under `OMP_WAIT_POLICY=active` (barrier spin inflating cycles on an imbalanced kernel)
are both named in the report's caveats rather than silently trusted.

---

## 3. Header-only, and how PAPI is reached

**One definition across translation units.** stb-style: without
`HPC_PAPI_IMPLEMENTATION` the header is declarations only; with it, definitions.

**No `-lpapi`, ever.** `libpapi` is `dlopen`'d at first use and its ~14 entry points
(`PAPI_library_init`, `PAPI_thread_init`, `PAPI_register_thread`,
`PAPI_create_eventset`, `PAPI_add_named_event`, `PAPI_query_named_event`,
`PAPI_num_cmp_hwctrs`, `PAPI_start`, `PAPI_stop`, `PAPI_read`, `PAPI_strerror`, ...) are
`dlsym`'d, mirroring `harness/papi.py`'s own ctypes path. Requiring `-lpapi` on the
link line would make the build fail on a host with no `libpapi.so` dev symlink, which
would contradict the "a host without PAPI still compiles" guarantee. The header
declares `PAPI_OK` / `PAPI_NULL` and the function-pointer types itself rather than
`#include <papi.h>`, which is not guaranteed installed.

`-fopenmp` is assumed already on the build (`omp.h` / `omp_get_thread_num` need it,
falling back to a serial stub -- one thread, reported as such -- when `_OPENMP` is
undefined).

---

## 4. Degradation, named and never silent

Every cause is a value of one enum (`hpcagent_bench.harness.papi.CAUSES`, 18 entries
today), and `test_causes_enum_matches_papi_causes` pins the header's copy to it.
CPU-relevant causes:

| Condition | Cause |
|---|---|
| macOS | `not_linux` |
| `perf_event_paranoid` absent | `no_perf_events` |
| `perf_event_paranoid` too restrictive | `perf_event_paranoid` |
| `dlopen` fails | `papi_missing` |
| library/thread init fails | `papi_init_failed` |
| no metric fits the armed set | `events_unsupported` |
| a thread appears after `init` | `threads_moved` |
| fewer measured reps than expected | `no_measured_rep` |
| one thread burned all the cycles | `not_openmp` |

Invariants, in every case: no `exit`, no `abort`, no allocation inside a bracketed
region, no change to any floating-point value, no new link dependency, no nonzero
process exit code from a degraded count. `count: null` is never `0` -- a metric this
CPU could not arm is absent, not a measured zero (a genuinely counted zero, like
`PAPI_FMA_INS` reading 0 for gemm on Zen4, has to stay readable as the measurement it
is). A failed report is all-zeros PLUS a non-empty `error`; all-zeros with an empty
`error` cannot happen.

---

## 5. The report

Flat -- one bracketed region per process, so there is no region tree. Schema
`hpc_papi/1`:

```json
{"schema": "hpc_papi/1", "error": "", "cause": "", "host": "nid001234",
 "fence": "mfence", "threads": 8, "threads_counted": 8, "reps": 1,
 "elapsed_ns": 12345678, "hardware_counters": 5, "smt": false,
 "caveats": ["a counted build is a diagnostic build: never ship it, ..."],
 "metrics": [
   {"metric": "cycles", "expression": "PAPI_TOT_CYC", "count": 94100322,
    "elapsed_ns": 12345678, "reps_counted": 1, "hardware_counters": 5,
    "threads_counted": 8, "scope": "all_threads", "per_thread": [...]},
   {"metric": "l3_cache_misses", "expression": "", "count": null,
    "missing": "needs 2 more of this CPU's 5 counter register(s) than one armed set has left; run again with HPC_PAPI_METRICS=l3_cache_misses"}
 ]}
```

`derive()` reads only `metric`, `count`, `expression`, `elapsed_ns` -- the rest rides
along untouched.

---

## 6. The measurement trap (a usage rule today, not yet an enforced one)

**A counted build is a diagnostic build. It should never be the scored submission** --
the header's own top comment says so, and a counted run's wall clock must never be
compared to anything, not even its own. This is the same prohibition
`skills/general/SKILL.md` already states for "time inside the kernel", with a counter
in place of a clock.

Nothing in the harness enforces it yet: no build path adds `-I` for the header's
directory, no `submission.build` token filter drops `-DHPC_PAPI*` or an `-I` naming
it, and no source scan refuses a scored build containing `hpc_papi_`. A submission
that includes the header today would need to supply its own include path and would
build; only the missing agent-facing skill and the lack of any prompt mention keep an
agent from stumbling into it. Wiring in that enforcement -- structural (no `-I` on the
scored path), token deny (drop `-I`/`-D` naming the header), and a source scan
refusing `hpc_papi_` in a scored build -- is part of "Not built" below.

**And the physics.** `PAPI_read` costs on the order of 1-3 us per thread, and the fence
drains the store buffer. Bracketing an inner loop body perturbs it by more than most
transforms gain. The rule for whoever uses this: bracket a region that runs
>= ~10 ms, never a loop body, and never compare a counted run's time to anything.

---

## 7. Tooling: one formula table, in Python

**The header computes no ratios.** It emits raw counts; `harness/papi.py`'s
`RATIOS`, `derive`, `imbalance`, `PER_THREAD_FORMULAS`, `IMBALANCE_FORMULA` stay the
only place a division happens, enforced by schema (the per-metric row the header
writes is exactly `derive()`'s input), not by discipline.

```sh
python -m hpcagent_bench.helpers.papi --emit-header        # print the generated header
python -m hpcagent_bench.helpers.papi --write               # regenerate hpc_papi.h in place
python -m hpcagent_bench.helpers.papi --read hpc_papi.json  # counts -> ratios
```

`--read` runs `papi.derive(metrics)` then the same rendering the `/profile` endpoint
already uses, with the same `formula` and `reading` strings per value.
`measurement_caveats` / `governor()` read the CURRENT host, so `--read` refuses to
render caveats when the report's `host` differs from the reading machine.

Environment the header reads: `HPC_PAPI_OUT` (report path, default `./hpc_papi.json`),
`HPC_PAPI_METRICS` (comma-separated metric names to arm; default is as many as fit the
budget), `HPC_PAPI_BUDGET` (override the counter-register budget), `HPC_PAPI_UNAVAILABLE`
(force named PAPI events absent, for testing the degradation ladder), `HPC_PAPI_VERBOSE`
(echo the degradation cause to stderr).

---

## 8. Tests that pin it to `harness/papi.py`

`tests/test_papi_header.py`: header freshness against the generator
(`test_header_is_up_to_date`), the event table matching `papi.METRICS` exactly
(`test_event_table_matches_papi_metrics`, `test_metric_index_pins_order_and_candidate_count`),
the cause enum matching `papi.CAUSES` (`test_causes_enum_matches_papi_causes`,
`test_every_cause_named_in_the_body_is_a_real_one`), `cycles`/`instructions` always
forced (`test_forced_metrics_are_the_denominators`), no ratio math in the header
(`test_the_header_computes_no_ratios`), no `-lpapi` anywhere
(`test_header_never_reaches_the_link_line`), never `exit`/`abort`
(`test_no_exit_and_no_abort`), every report row is valid `derive()` input
(`test_report_rows_are_derive_input`), the header compiling warning-free as C and as
C++ (`test_header_compiles_warning_free`, `test_header_compiles_as_cxx`), declarations
without the implementation macro (`test_declarations_only_without_the_implementation_macro`),
an actually-counted region producing counts and ratios end to end
(`test_a_counted_region_reports_counts_and_ratios`), a report that gives up before
selection still naming the wanted events
(`test_a_report_that_gives_up_before_selection_still_names_the_events`), absence-vs-zero
(`test_absence_is_null_and_failure_is_zero_with_an_error`), the budget bounding one
armed set (`test_the_budget_bounds_one_armed_set`), `$HPC_PAPI_METRICS` selection
keeping the forced denominators (`test_selecting_a_metric_keeps_the_denominators`), and
`--read` printing the error before the counts (`test_read_prints_the_error_before_the_counts`).

---

## Not built (proposed, no rationale here should be read as already true of the code)

**Tagged regions.** `hpc_papi_start(tag)` / `hpc_papi_stop(tag)` so a report can name
more than one region per run. Without a tag a CPU report has nothing to attribute a
bracket to -- `perf` names symbols because the linker did, and a counter bracket has no
symbol at all, so the caller would declare the scope and the tag would be what the
report is keyed by. A second `start` on an already-open tag should be an ERROR, not a
nested region or a silent re-arm: one event set is live at a time, and overlapping
regions would attribute the same counts twice. Sequential tags in one run are fine.
This would extend the ALREADY-SHIPPED single-pass, budget-limited selection (Sec. 1)
per tag -- each tag its own bracket and its own armed set -- rather than reintroduce a
multi-pass-per-metric replay loop; an earlier draft of this section assumed the
latter and is why this note calls it out explicitly.

**Per-language source fragments.** A `papi-init` / `papi-start` / `papi-stop` /
`papi-finalize` fragment generator, gated by a `-DHPC_PAPI` switch, so a restricted
(source-only) submission could turn instrumentation on and off without editing its own
build line, including a `bind(C)` Fortran interface against the header's C entry
points. Nothing in `hpcagent_bench/helpers/papi/header.py` generates fragments today;
the only generated artifact is the header itself.

**Two agent-facing skills.** `papi-standalone` (instrument your own source, drive it
yourself) and `papi-counters` (call it through the judge's `/profile` API, which runs
the kernel and owns the inputs). Neither exists yet under `hpcagent_bench/skills/`; the
`profiling` skill covers only the existing whole-run `/profile` counters.

**Harness enforcement of the measurement trap** (Sec. 6): the structural `-I` gate, the
`submission.build` token deny, and the scored-build source scan refusing `hpc_papi_`.

**GPU counters per kernel.** Collected per kernel from the existing device trace (the
trace already ranks kernels), reusing the vendor tools' own rendered text
(`rocprofv3`, `rocprof-compute`, `ncu`) rather than re-rendering into a schema of this
header's own.

**`.so` (prebuilt-binary) delivery.** A prebuilt `.so` is never recompiled, so the
enforcement above would not reach it; whether an uninstrumented `.so` can still be
counted from the outside (PAPI attach by TID rather than by OpenMP thread number) is
an open question, not a shipped path.
