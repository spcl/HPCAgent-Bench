# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Emit ``hpc_papi.h`` from the harness tables, and read back the report it writes.

Two directions, one table. :func:`header_text` prints
:data:`hpcagent_bench.harness.papi.METRICS`, :data:`~hpcagent_bench.harness.papi.CAUSES`,
:data:`~hpcagent_bench.harness.papi.PER_THREAD_METRICS` and the version-probe range as C, so the
header cannot hold a metric this repo does not know about and cannot miss one it does.
:func:`read_report` goes the other way: the header emits RAW COUNTS in exactly
:func:`~hpcagent_bench.harness.papi.counting_worker`'s row shape, and every division happens here
through :func:`~hpcagent_bench.harness.papi.derive` and the same renderers the ``/profile``
endpoint prints. The header does no arithmetic beyond a signed sum, which is what keeps
:data:`~hpcagent_bench.harness.papi.RATIOS` the only place a formula exists.

The generated file is TRACKED, not built on demand: an agent's compile line must find a header
that is already there, and ``tests/test_papi_header.py`` regenerates it and diffs.
"""
import argparse
import json
import pathlib
import socket
import sys
from typing import Dict, List, Sequence, Tuple

from hpcagent_bench.harness import papi, profiling

#: The generated header. Beside this module so it ships with the package and so the include path
#: is the helpers directory (``-I<repo>/hpcagent_bench/helpers`` -> ``#include <papi/hpc_papi.h>``).
HEADER: pathlib.Path = pathlib.Path(__file__).with_name("hpc_papi.h")

#: What ``--read`` prints above the counter table when the report names a different machine. The
#: metric rows are the counted host's; anything this process would read from sysfs is not.
FOREIGN_HOST = ("this report was written on {there!r} and is being read on {here!r}: the counts "
                "are that machine's, and any cache-line or SMT fact below is this one's")


def event_names() -> Tuple[str, ...]:
    """Every distinct PAPI event name :data:`~hpcagent_bench.harness.papi.METRICS` can ask for.

    The upper bound on one armed event set, so the header sizes its per-thread slot from the table
    rather than from a guessed constant.
    """
    seen: Dict[str, None] = {}
    for candidates in papi.METRICS.values():
        for candidate in candidates:
            for term in candidate:
                seen.setdefault(papi.event_name(term), None)
    return tuple(seen)


def c_terms(candidate: Sequence[str], width: int) -> str:
    """One candidate as a C initializer, NULL-terminated. The leading ``-`` is kept: it is the
    SIGN, and dropping it here would turn a derived metric into a sum of its parts."""
    terms = [f'"{term}"' for term in candidate] + ["NULL"] * (width - len(candidate))
    return "{" + ", ".join(terms) + "}"


def tables() -> str:
    """Every generated table, in one block: metrics, causes, the denominators, the version range.

    Emitted already CLANG-FORMAT CLEAN (LLVM base, 120 cols), because ``scripts/check_format.py``
    formats every tracked ``.h`` and a generator that disagreed with it would fight the pre-commit
    hook forever -- with ``test_header_is_up_to_date`` failing after every commit as the symptom.
    """
    width = max(len(c) for cands in papi.METRICS.values() for c in cands) + 1  # + the NULL terminator
    majors, minors = papi.VERSION_MAJORS, papi.VERSION_MINORS
    lines = [
        "/* ---- GENERATED TABLES. There is no second copy: hpcagent_bench.helpers.papi prints",
        " * these from hpcagent_bench.harness.papi, and tests/test_papi_header.py parses them back",
        " * and asserts equality including candidate order and the leading '-' sign. ------------ */",
        "",
        f"#define HPC_PAPI_OK {papi.PAPI_OK}",
        f"#define HPC_PAPI_NULLSET {papi.PAPI_NULL}",
        f"#define HPC_PAPI_NMETRIC {len(papi.METRICS)}",
        f"#define HPC_PAPI_NTERM {width}",
        f"#define HPC_PAPI_MAXEV {len(event_names())}",
        f"#define HPC_PAPI_LINE {papi.DEFAULT_LINE_BYTES}",
        "",
        "/* PAPI_VER_CURRENT is a header constant and libpapi exports no version symbol, so the",
        " * version is PROBED, newest first, exactly as hpcagent_bench.harness.papi.initialised does. */",
        f"#define HPC_PAPI_MAJOR_FIRST {majors[0]}",
        f"#define HPC_PAPI_MAJOR_LAST {majors[-1]}",
        f"#define HPC_PAPI_MINOR_FIRST {minors[0]}",
        f"#define HPC_PAPI_MINOR_LAST {minors[-1]}",
        "",
        "/* Machine-readable degradation reasons, in order. */",
        "enum {",
    ]
    lines += [f"  HPC_C_{cause}," for cause in papi.CAUSES]
    lines += ["};", "", "static const char *const HPC_PAPI_CAUSES[] = {"]
    lines += [f'    "{cause}",' for cause in papi.CAUSES]
    lines += [
        "};",
        "",
        "/* Forced into the armed set before anything else: they are the denominators of nearly",
        " * every ratio, and a ratio whose numerator and denominator came from two different armed",
        " * sets is a ratio over two different schedules. */",
        "static const char *const HPC_PAPI_FORCED[] = {" + ", ".join(f'"{m}"' for m in papi.PER_THREAD_METRICS) + "};",
        "",
        "/* A candidate is a term list, best candidate first; a leading '-' subtracts; NULL ends it. */",
    ]
    for metric, candidates in papi.METRICS.items():
        lines.append(f"static const char *const HPC_PAPI_CAND_{metric}[][HPC_PAPI_NTERM] = {{")
        lines += [f"    {c_terms(candidate, width)}," for candidate in candidates]
        lines.append("};")
    lines += [
        "", "static const struct {", "  const char *name;", "  const char *const (*cand)[HPC_PAPI_NTERM];",
        "  int ncand;", "} HPC_PAPI_METRIC[HPC_PAPI_NMETRIC] = {"
    ]
    for metric, candidates in papi.METRICS.items():
        lines.append(f'    {{"{metric}", HPC_PAPI_CAND_{metric}, {len(candidates)}}},')
    lines += ["};", ""]
    return "\n".join(lines)


BANNER = r'''/* hpc_papi.h -- region hardware counters for a kernel you are optimizing. HEADER-ONLY.
 *
 * GENERATED by hpcagent_bench.helpers.papi -- DO NOT EDIT. Regenerate with
 *     python -m hpcagent_bench.helpers.papi --write
 *
 *     #define HPC_PAPI_IMPLEMENTATION     // in EXACTLY one translation unit
 *     #include <papi/hpc_papi.h>          // -I<repo>/hpcagent_bench/helpers
 *
 *     hpc_papi_init();                    // ONCE, from serial code. It opens its own parallel
 *                                         // region to register every OpenMP thread -- do not
 *                                         // wrap this call in one of yours.
 *     hpc_papi_start();  ...  hpc_papi_stop();   // brackets THE region. Pairs ACCUMULATE, so a
 *                                                // phase inside a loop can be bracketed.
 *     hpc_papi_finalize();                // writes $HPC_PAPI_OUT (default ./hpc_papi.json)
 *
 * Read the report back with:  python -m hpcagent_bench.helpers.papi --read hpc_papi.json
 * It emits RAW COUNTS and no ratios; every division lives in hpcagent_bench.harness.papi.RATIOS.
 *
 * libpapi is dlopen'd, so nothing goes on the link line: a host without PAPI still COMPILES and
 * still RUNS, degraded, with a named cause in the report. This never aborts, never exits, never
 * allocates inside a bracketed region and never touches a floating-point value.
 *
 * A counted build is a DIAGNOSTIC build. Bracket a region of >= ~10 ms, never a loop body, and
 * never compare a counted run's wall clock against anything -- not even its own.
 *
 * Failure is loud by construction: every count reads 0 AND the report's "error" is non-empty.
 * All zeros with an empty "error" cannot happen, which is what keeps a GENUINELY counted zero
 * (PAPI_FMA_INS reads exactly 0 for gemm on Zen4) readable as the measurement it is. A metric
 * this CPU cannot express is "count": null with a reason -- absent, never zero.
 *
 * Environment:
 *   HPC_PAPI_OUT      report path (default ./hpc_papi.json)
 *   HPC_PAPI_METRICS  comma-separated metric names to arm; default is as many as fit the budget
 *   HPC_PAPI_BUDGET   override the counter-register budget (testing the packing)
 *   HPC_PAPI_VERBOSE  echo the degradation cause to stderr
 */
#ifndef HPC_PAPI_H
#define HPC_PAPI_H

#ifdef __cplusplus
extern "C" {
#endif

int hpc_papi_init(void); /* 0 = counting, <0 = degraded (the report says why) */
void hpc_papi_start(void);
void hpc_papi_stop(void);
int hpc_papi_finalize(void); /* 0 = a counted report, <0 = a degraded one. NOT an exit code. */

#ifdef __cplusplus
}
#endif

#endif /* HPC_PAPI_H */

#ifdef HPC_PAPI_IMPLEMENTATION
#ifndef HPC_PAPI_IMPLEMENTED
#define HPC_PAPI_IMPLEMENTED

/* Nothing here needs a feature-test macro. The harness compiles C at -std=c17, which hides every
 * POSIX declaration, so the hostname is READ FROM /proc and the alignment is done by hand rather
 * than reaching for gethostname or posix_memalign. */
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#else /* a serial TU still counts, on one thread, and the report says so */
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#define omp_in_parallel() 0
#endif

/* The fence's job is to stop the helper's OWN buffered stores from drifting across the region
 * boundary and landing inside the counts. aarch64 reorders MORE than x86-64, so "no fence there"
 * -- what the DaCe reference this borrows from does -- is exactly backwards. */
#if defined(__x86_64__) && defined(__GNUC__)
#include <x86intrin.h>
#define HPC_PAPI_FENCE _mm_mfence()
#define HPC_PAPI_FENCE_NAME "mfence"
#elif defined(__aarch64__)
#define HPC_PAPI_FENCE __atomic_thread_fence(__ATOMIC_SEQ_CST)
#define HPC_PAPI_FENCE_NAME "atomic_seq_cst"
#else
#define HPC_PAPI_FENCE ((void)0)
#define HPC_PAPI_FENCE_NAME "none"
#endif

#ifdef __cplusplus
#define HPC_PAPI_ALIGN alignas(HPC_PAPI_LINE)
#else
#define HPC_PAPI_ALIGN _Alignas(HPC_PAPI_LINE)
#endif

'''

BODY = r'''
/* ---- state ---------------------------------------------------------------------------------- */

/* One per OpenMP thread, cache-line aligned and >= 2 lines wide: false sharing between two
 * threads' counter slots corrupts the very measurement this is taking. */
typedef struct {
  HPC_PAPI_ALIGN long long acc[HPC_PAPI_MAXEV]; /* accumulated across every start/stop pair */
  long long now[HPC_PAPI_MAXEV];                /* PAPI_stop's landing buffer, so stop allocates nothing */
  int eventset;
  int rc;
} hpc_papi_slot;

static struct {
  void *dl;
  int (*library_init)(int);
  int (*thread_init)(unsigned long (*)(void));
  int (*register_thread)(void);
  int (*unregister_thread)(void);
  int (*create_eventset)(int *);
  int (*destroy_eventset)(int *);
  int (*cleanup_eventset)(int);
  int (*add_named_event)(int, const char *);
  int (*query_named_event)(const char *);
  int (*num_cmp_hwctrs)(int);
  int (*start)(int);
  int (*stop)(int, long long *);
  char *(*strerror)(int);
} hpc_papi;

static hpc_papi_slot *hpc_papi_slots;
static void *hpc_papi_block; /* what malloc returned; hpc_papi_slots is the line-aligned view */
static int hpc_papi_nthread;
static int hpc_papi_budget;
static int hpc_papi_nev;                                  /* distinct events in the armed set */
static const char *hpc_papi_ev[HPC_PAPI_MAXEV];           /* their names, in slot order */
static int hpc_papi_pick[HPC_PAPI_NMETRIC];               /* chosen candidate, -1 = not armed */
static int hpc_papi_at[HPC_PAPI_NMETRIC][HPC_PAPI_NTERM]; /* term -> slot index */
static char hpc_papi_why[HPC_PAPI_NMETRIC][160];          /* why a metric is absent; empty = armed */
static char hpc_papi_err[512];
static const char *hpc_papi_cause = "";
static int hpc_papi_live;
static int hpc_papi_open;
static int hpc_papi_reps;
static int hpc_papi_done;
static long long hpc_papi_ns;
static struct timespec hpc_papi_t0;

/* ---- plumbing ------------------------------------------------------------------------------- */

static void hpc_papi_fail(int cause, const char *fmt, ...) {
  va_list ap;
  if (hpc_papi_err[0]) /* the FIRST cause is the one that explains the rest */
    return;
  va_start(ap, fmt);
  vsnprintf(hpc_papi_err, sizeof hpc_papi_err, fmt, ap);
  va_end(ap);
  hpc_papi_cause = HPC_PAPI_CAUSES[cause];
  hpc_papi_live = 0;
  if (getenv("HPC_PAPI_VERBOSE"))
    fprintf(stderr, "hpc_papi: %s: %s\n", hpc_papi_cause, hpc_papi_err);
}

/* PAPI's own text, so it stays right across PAPI versions; the code rides along because PAPI's
 * table does not cover everything its components return. */
static const char *hpc_papi_text(int rc) {
  const char *text = hpc_papi.strerror ? hpc_papi.strerror(rc) : NULL;
  return text ? text : "unknown PAPI error";
}

static int hpc_papi_sysfs_int(const char *path, int *out) {
  FILE *f = fopen(path, "r");
  int ok;
  if (!f)
    return 0;
  ok = fscanf(f, "%d", out) == 1;
  fclose(f);
  return ok;
}

static void hpc_papi_sysfs_str(const char *path, char *out, int n) {
  FILE *f = fopen(path, "r");
  int i = 0;
  out[0] = '\0';
  if (!f)
    return;
  for (; i < n - 1; i++) {
    int c = fgetc(f);
    if (c == EOF || c == '\n')
      break;
    out[i] = (char)c;
  }
  out[i] = '\0';
  fclose(f);
}

/* PAPI_thread_init wants an unsigned-long id function. A wrapper rather than a cast of
 * omp_get_thread_num: a function-pointer cast that lies about the return type is undefined. */
static unsigned long hpc_papi_thread_id(void) { return (unsigned long)omp_get_thread_num(); }

static const char *hpc_papi_bare(const char *term) { return term[0] == '-' ? term + 1 : term; }

static int hpc_papi_listed(const char *list, const char *name) {
  size_t want = strlen(name);
  const char *p = list;
  while (*p) {
    const char *end;
    size_t len;
    while (*p == ' ' || *p == ',')
      p++;
    end = p;
    while (*end && *end != ',')
      end++;
    len = (size_t)(end - p);
    while (len && p[len - 1] == ' ')
      len--;
    if (len == want && !strncmp(p, name, want))
      return 1;
    p = end;
  }
  return 0;
}

static int hpc_papi_forced(const char *name) {
  size_t i;
  for (i = 0; i < sizeof HPC_PAPI_FORCED / sizeof HPC_PAPI_FORCED[0]; i++)
    if (!strcmp(HPC_PAPI_FORCED[i], name))
      return 1;
  return 0;
}

static int hpc_papi_slot_of(const char *event) {
  int i;
  for (i = 0; i < hpc_papi_nev; i++)
    if (!strcmp(hpc_papi_ev[i], event))
      return i;
  return -1;
}

/* ---- bring-up ------------------------------------------------------------------------------- */

#define HPC_PAPI_SYM(field, name)                                                                                      \
  do {                                                                                                                 \
    *(void **)(&hpc_papi.field) = dlsym(hpc_papi.dl, name);                                                            \
    if (!hpc_papi.field) {                                                                                             \
      hpc_papi_fail(HPC_C_papi_missing, "the loaded libpapi has no %s", name);                                         \
      return -1;                                                                                                       \
    }                                                                                                                  \
  } while (0)

/* macOS and the perf_event gate, in that order and BEFORE dlopen. A closed gate makes PAPI's own
 * error PAPI_ESYS at PAPI_start, which reads like a broken install. One function rather than a
 * branch in init, so nothing below it is compiled-but-unreferenced off Linux. */
static int hpc_papi_gate(void) {
#if !defined(__linux__)
  hpc_papi_fail(HPC_C_not_linux, "PAPI counting is wired for Linux only; on macOS the hardware "
                                 "counters are behind Instruments' 'CPU Counters' template, which cannot be driven "
                                 "from a process");
  return -1;
#else
  int paranoid = 0;
  if (!hpc_papi_sysfs_int("/proc/sys/kernel/perf_event_paranoid", &paranoid)) {
    hpc_papi_fail(HPC_C_no_perf_events,
                  "/proc/sys/kernel/perf_event_paranoid is absent: this kernel "
                  "exposes no perf_event subsystem, so PAPI's cpu component has nothing to count with");
    return -1;
  }
  if (paranoid > 2) {
    hpc_papi_fail(HPC_C_perf_event_paranoid,
                  "kernel.perf_event_paranoid=%d blocks unprivileged "
                  "perf_event_open; need <= 2 ('sudo sysctl -w kernel.perf_event_paranoid=2', or run "
                  "the container with --cap-add=CAP_PERFMON)",
                  paranoid);
    return -1;
  }
  return 0;
#endif
}

static int hpc_papi_load(void) {
  char soname[32];
  int major;
  /* PAPI never reaches the link line: requiring the dev symlink would make the BUILD fail on a
   * host without PAPI, and a diagnostic must never be able to break a build. */
  hpc_papi.dl = dlopen("libpapi.so", RTLD_NOW | RTLD_GLOBAL);
  for (major = HPC_PAPI_MAJOR_FIRST; !hpc_papi.dl && major >= HPC_PAPI_MAJOR_LAST; major--) {
    snprintf(soname, sizeof soname, "libpapi.so.%d", major);
    hpc_papi.dl = dlopen(soname, RTLD_NOW | RTLD_GLOBAL);
  }
  if (!hpc_papi.dl) {
    hpc_papi_fail(HPC_C_papi_missing,
                  "libpapi could not be dlopen'd (%s); install PAPI "
                  "(Debian/Ubuntu: 'apt install libpapi-dev') or put it on the loader path",
                  dlerror() ? dlerror() : "no reason given");
    return -1;
  }
  HPC_PAPI_SYM(library_init, "PAPI_library_init");
  HPC_PAPI_SYM(thread_init, "PAPI_thread_init");
  HPC_PAPI_SYM(register_thread, "PAPI_register_thread");
  HPC_PAPI_SYM(unregister_thread, "PAPI_unregister_thread");
  HPC_PAPI_SYM(create_eventset, "PAPI_create_eventset");
  HPC_PAPI_SYM(destroy_eventset, "PAPI_destroy_eventset");
  HPC_PAPI_SYM(cleanup_eventset, "PAPI_cleanup_eventset");
  HPC_PAPI_SYM(add_named_event, "PAPI_add_named_event");
  HPC_PAPI_SYM(query_named_event, "PAPI_query_named_event");
  HPC_PAPI_SYM(num_cmp_hwctrs, "PAPI_num_cmp_hwctrs");
  HPC_PAPI_SYM(start, "PAPI_start");
  HPC_PAPI_SYM(stop, "PAPI_stop");
  HPC_PAPI_SYM(strerror, "PAPI_strerror");
  return 0;
}

static int hpc_papi_bring_up(void) {
  int major, minor;
  for (major = HPC_PAPI_MAJOR_FIRST; major >= HPC_PAPI_MAJOR_LAST; major--)
    for (minor = HPC_PAPI_MINOR_FIRST; minor >= HPC_PAPI_MINOR_LAST; minor--) {
      int want = (major << 24) | (minor << 16);
      if (hpc_papi.library_init(want) == want)
        return want;
    }
  return 0;
}

/* The first candidate every one of whose events this CPU reports, or -1. Names resolve HERE and
 * nowhere else: start and stop touch no strings. */
static int hpc_papi_resolve(int m) {
  int c, t;
  for (c = 0; c < HPC_PAPI_METRIC[m].ncand; c++) {
    int ok = 1;
    for (t = 0; HPC_PAPI_METRIC[m].cand[c][t] && ok; t++)
      ok = hpc_papi.query_named_event(hpc_papi_bare(HPC_PAPI_METRIC[m].cand[c][t])) == HPC_PAPI_OK;
    if (ok)
      return c;
  }
  return -1;
}

/* Pack metrics into ONE armed set. There is one pass because there is one API: start/stop bracket
 * a region of a program this header does not drive, so it cannot re-run the kernel for a second
 * pass. A metric that does not fit is ABSENT with a reason and the name of the knob that gets it
 * -- never multiplexed, because a multiplexed number is an estimate wearing a count's clothes. */
static void hpc_papi_arm(int m) {
  const char *const *terms;
  const char *add[HPC_PAPI_NTERM];
  int nadd = 0, c, t, i;

  c = hpc_papi_resolve(m);
  if (c < 0) {
    snprintf(hpc_papi_why[m], sizeof hpc_papi_why[m], "no candidate expression is available on this CPU");
    return;
  }
  terms = HPC_PAPI_METRIC[m].cand[c];
  for (t = 0; terms[t]; t++) {
    const char *event = hpc_papi_bare(terms[t]);
    int seen = hpc_papi_slot_of(event) >= 0;
    for (i = 0; i < nadd && !seen; i++)
      seen = !strcmp(add[i], event);
    if (!seen)
      add[nadd++] = event;
  }
  if (hpc_papi_nev + nadd > hpc_papi_budget) {
    snprintf(hpc_papi_why[m], sizeof hpc_papi_why[m],
             "needs %d more of this CPU's %d counter register(s) than one armed set has left; "
             "run again with HPC_PAPI_METRICS=%s",
             nadd, hpc_papi_budget, HPC_PAPI_METRIC[m].name);
    return;
  }
  for (i = 0; i < nadd; i++)
    hpc_papi_ev[hpc_papi_nev++] = add[i];
  for (t = 0; terms[t]; t++)
    hpc_papi_at[m][t] = hpc_papi_slot_of(hpc_papi_bare(terms[t]));
  hpc_papi_pick[m] = c;
}

static void hpc_papi_select(void) {
  const char *want = getenv("HPC_PAPI_METRICS");
  int round, m;
  for (round = 0; round < 2; round++)
    for (m = 0; m < HPC_PAPI_NMETRIC; m++) {
      if (hpc_papi_pick[m] >= 0 || hpc_papi_why[m][0])
        continue;
      if ((round == 0) != (hpc_papi_forced(HPC_PAPI_METRIC[m].name) != 0))
        continue; /* the denominators claim their registers first */
      /* HPC_PAPI_METRICS cannot deselect a denominator. Two metrics that did not fit one
       * armed set come from two different RUNS, and the only honest way to compare them is
       * per-instruction or per-cycle -- so both runs have to have counted those. */
      if (want && !hpc_papi_forced(HPC_PAPI_METRIC[m].name) && !hpc_papi_listed(want, HPC_PAPI_METRIC[m].name)) {
        snprintf(hpc_papi_why[m], sizeof hpc_papi_why[m], "not named by HPC_PAPI_METRICS");
        continue;
      }
      hpc_papi_arm(m);
    }
}

int hpc_papi_init(void) {
  size_t bytes;
  const char *want_budget;
  int m, th, failed = -1;

  if (hpc_papi_live)
    return 0;
  if (hpc_papi_err[0])
    return -1;
  for (m = 0; m < HPC_PAPI_NMETRIC; m++)
    hpc_papi_pick[m] = -1;

  if (hpc_papi_gate() < 0)
    return -1;
  if (hpc_papi_load() < 0)
    return -1;
  if (!hpc_papi_bring_up()) {
    hpc_papi_fail(HPC_C_papi_init_failed,
                  "PAPI_library_init rejected every version from %d.x down to "
                  "%d.x: the loaded libpapi is newer than this range or broken ('papi_avail' will print "
                  "the same failure)",
                  HPC_PAPI_MAJOR_FIRST, HPC_PAPI_MAJOR_LAST);
    return -1;
  }
  /* Without this every thread shares one PAPI thread context and the per-thread sets below are
   * all the same set. It must come after library_init and before any register_thread. */
  if (hpc_papi.thread_init(hpc_papi_thread_id) != HPC_PAPI_OK) {
    hpc_papi_fail(HPC_C_papi_init_failed, "PAPI_thread_init failed, so per-thread counting is unavailable");
    return -1;
  }

  hpc_papi_budget = hpc_papi.num_cmp_hwctrs(0);
  want_budget = getenv("HPC_PAPI_BUDGET");
  if (want_budget && *want_budget) {
    /* strtol with the end pointer checked, not atoi: atoi turns a typo into 0, which then falls
     * into the branch below and reports "PAPI reports 0 counter register(s) on this CPU" -- so a
     * mistyped variable is diagnosed as a property of the hardware. */
    char *end;
    long asked = strtol(want_budget, &end, 10);
    if (*end || asked <= 0) {
      hpc_papi_fail(HPC_C_events_unsupported, "HPC_PAPI_BUDGET=%s is not a positive integer", want_budget);
      return -1;
    }
    hpc_papi_budget = (int)asked;
  }
  if (hpc_papi_budget > HPC_PAPI_MAXEV)
    hpc_papi_budget = HPC_PAPI_MAXEV;
  if (hpc_papi_budget <= 0) {
    hpc_papi_fail(HPC_C_events_unsupported,
                  "PAPI reports %d counter register(s) on this CPU, so nothing "
                  "can be armed without multiplexing -- which is an estimate, not a count",
                  hpc_papi_budget);
    return -1;
  }
  hpc_papi_select();
  if (!hpc_papi_nev) {
    hpc_papi_fail(HPC_C_events_unsupported, "not one metric resolved to events this CPU reports "
                                            "('papi_avail' lists what it has)");
    return -1;
  }

  hpc_papi_nthread = omp_get_max_threads();
  if (omp_in_parallel() || hpc_papi_nthread < 1) {
    hpc_papi_fail(HPC_C_threads_moved,
                  "hpc_papi_init must be called from SERIAL code: it opens its own "
                  "parallel region to register every thread, and a nested one registers a different team");
    return -1;
  }
  bytes = (size_t)hpc_papi_nthread * sizeof(hpc_papi_slot);
  hpc_papi_block = malloc(bytes + HPC_PAPI_LINE);
  if (!hpc_papi_block) {
    hpc_papi_fail(HPC_C_run_failed, "could not allocate %d cache-line-aligned counter slot(s)", hpc_papi_nthread);
    return -1;
  }
  /* Aligned by hand: the slot is a whole number of lines wide, so aligning the base is what
   * keeps two threads' counters off one line. */
  hpc_papi_slots =
      (hpc_papi_slot *)(void *)(((uintptr_t)hpc_papi_block + HPC_PAPI_LINE - 1) & ~(uintptr_t)(HPC_PAPI_LINE - 1));
  memset(hpc_papi_slots, 0, bytes);

  /* The WHOLE per-thread setup is serialized. PAPI's event-set creation is not thread-safe and
   * racing it produces intermittent WRONG COUNTS rather than a clean failure -- which is why
   * this is structural and not something a test could be trusted to catch. */
#pragma omp parallel num_threads(hpc_papi_nthread)
  {
    int t = omp_get_thread_num();
    hpc_papi_slot *slot = &hpc_papi_slots[t];
    slot->eventset = HPC_PAPI_NULLSET;
#pragma omp critical(hpc_papi_setup)
    {
      int i;
      slot->rc = hpc_papi.register_thread();
      if (slot->rc == HPC_PAPI_OK)
        slot->rc = hpc_papi.create_eventset(&slot->eventset);
      for (i = 0; i < hpc_papi_nev && slot->rc == HPC_PAPI_OK; i++)
        slot->rc = hpc_papi.add_named_event(slot->eventset, hpc_papi_ev[i]);
    }
  }
  for (th = 0; th < hpc_papi_nthread; th++)
    if (hpc_papi_slots[th].rc != HPC_PAPI_OK)
      failed = th;
  if (failed >= 0) {
    /* Events resolved once, before any thread existed, so every set is identical by
     * construction. If one still fails, the whole armed set degrades rather than reporting a
     * shorter vector than it declared. */
    hpc_papi_fail(HPC_C_events_unsupported, "thread %d could not arm the %d resolved event(s): %s", failed,
                  hpc_papi_nev, hpc_papi_text(hpc_papi_slots[failed].rc));
    return -1;
  }
  hpc_papi_live = 1;
  return 0;
}

/* ---- the region ----------------------------------------------------------------------------- */

void hpc_papi_start(void) {
  int t;
  if (!hpc_papi_live || hpc_papi_open)
    return;
  if (omp_in_parallel() || omp_get_max_threads() != hpc_papi_nthread) {
    hpc_papi_fail(HPC_C_threads_moved,
                  "hpc_papi_start ran with %d thread(s) available where init "
                  "registered %d (or inside a parallel region): the counts would be missing whatever "
                  "ran on the threads nothing was armed on",
                  omp_get_max_threads(), hpc_papi_nthread);
    return;
  }
  hpc_papi_open = 1;
#pragma omp parallel num_threads(hpc_papi_nthread)
  {
    hpc_papi_slot *slot = &hpc_papi_slots[omp_get_thread_num()];
    HPC_PAPI_FENCE; /* drain this thread's own stores BEFORE the counters arm */
    slot->rc = hpc_papi.start(slot->eventset);
  }
  /* AFTER the arming region, to match hpc_papi_stop, which stamps BEFORE its own. The bracket has
   * to be symmetric or it is not a bracket: taking t0 first charged every rep one thread-team fork
   * plus one PAPI_start to the wall clock while the counters saw none of it, so every derived rate
   * (instructions/ns, bytes/ns) came out low by a fixed per-rep constant -- worst on exactly the
   * short regions where a rate matters most, and an empty bracket would report time against no
   * work. */
  clock_gettime(CLOCK_MONOTONIC, &hpc_papi_t0);
  for (t = 0; t < hpc_papi_nthread; t++)
    if (hpc_papi_slots[t].rc != HPC_PAPI_OK)
      hpc_papi_fail(HPC_C_events_unsupported, "PAPI_start failed on thread %d: %s", t,
                    hpc_papi_text(hpc_papi_slots[t].rc));
}

void hpc_papi_stop(void) {
  struct timespec t1;
  int t;
  if (!hpc_papi_live || !hpc_papi_open)
    return;
  clock_gettime(CLOCK_MONOTONIC, &t1);
#pragma omp parallel num_threads(hpc_papi_nthread)
  {
    hpc_papi_slot *slot = &hpc_papi_slots[omp_get_thread_num()];
    int i;
    HPC_PAPI_FENCE; /* everything the region wrote must land before the counters are read */
    slot->rc = hpc_papi.stop(slot->eventset, slot->now);
    if (slot->rc == HPC_PAPI_OK)
      for (i = 0; i < hpc_papi_nev; i++)
        slot->acc[i] += slot->now[i]; /* pairs ACCUMULATE: a phase in a loop is one region */
  }
  hpc_papi_open = 0;
  hpc_papi_ns += (long long)(t1.tv_sec - hpc_papi_t0.tv_sec) * 1000000000LL + (t1.tv_nsec - hpc_papi_t0.tv_nsec);
  hpc_papi_reps++;
  for (t = 0; t < hpc_papi_nthread; t++)
    if (hpc_papi_slots[t].rc != HPC_PAPI_OK)
      hpc_papi_fail(HPC_C_events_unsupported, "PAPI_stop failed on thread %d: %s", t,
                    hpc_papi_text(hpc_papi_slots[t].rc));
}

/* ---- the report ----------------------------------------------------------------------------- */

static void hpc_papi_json_str(FILE *out, const char *s) {
  fputc('"', out);
  for (; s && *s; s++) {
    unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\')
      fprintf(out, "\\%c", c);
    else if (c < 0x20)
      fprintf(out, "\\u%04x", c);
    else
      fputc((int)c, out);
  }
  fputc('"', out);
}

/* One thread's value for metric m: the signed sum of its terms, so a derived metric is one
 * number like a direct one. */
static long long hpc_papi_value(int m, int thread) {
  const char *const *terms = HPC_PAPI_METRIC[m].cand[hpc_papi_pick[m]];
  long long v = 0;
  int t;
  for (t = 0; terms[t]; t++) {
    long long raw = hpc_papi_slots[thread].acc[hpc_papi_at[m][t]];
    v += terms[t][0] == '-' ? -raw : raw;
  }
  return v;
}

static void hpc_papi_write_metric(FILE *out, int m) {
  const char *const *terms = hpc_papi_pick[m] >= 0 ? HPC_PAPI_METRIC[m].cand[hpc_papi_pick[m]] : NULL;
  int counted = terms && !hpc_papi_err[0];
  long long total = 0;
  int t, i;

  fputs("  {\"metric\": ", out);
  hpc_papi_json_str(out, HPC_PAPI_METRIC[m].name);
  if (!terms && !hpc_papi_err[0]) {
    /* ABSENT, not zero: the distinction hpcagent_bench.harness.papi.missing() enforces one
     * level down. The whole-report failure below is the other rule -- zeros, beside an error. */
    fputs(", \"expression\": \"\", \"count\": null, \"missing\": ", out);
    hpc_papi_json_str(out, hpc_papi_why[m][0] ? hpc_papi_why[m] : "not armed");
    fputs("}", out);
    return;
  }
  fputs(", \"expression\": \"", out);
  for (t = 0; terms && terms[t]; t++)
    fprintf(out, "%s%s", t ? (terms[t][0] == '-' ? " - " : " + ") : "", hpc_papi_bare(terms[t]));
  fputs("\", \"events\": [", out);
  for (t = 0; terms && terms[t]; t++) {
    if (t)
      fputs(", ", out);
    hpc_papi_json_str(out, hpc_papi_bare(terms[t]));
  }
  if (counted)
    for (i = 0; i < hpc_papi_nthread; i++)
      total += hpc_papi_value(m, i);
  fprintf(out,
          "], \"derived\": %s, \"count\": %lld, \"elapsed_ns\": %lld, \"reps_counted\": %d, "
          "\"hardware_counters\": %d, \"threads_counted\": %d, \"scope\": \"all_threads\", \"per_thread\": [",
          (terms && terms[1]) ? "true" : "false", total, hpc_papi_ns, hpc_papi_reps, hpc_papi_budget,
          counted ? hpc_papi_nthread : 0);
  for (i = 0; counted && i < hpc_papi_nthread; i++)
    fprintf(out, "%s%lld", i ? ", " : "", hpc_papi_value(m, i));
  fputs("]}", out);
}

static void hpc_papi_write(const char *path) {
  FILE *out = fopen(path, "w");
  int m, first = 1, smt = 0;
  int smt_known = hpc_papi_sysfs_int("/sys/devices/system/cpu/smt/active", &smt);
  char host[256];
  if (!out) {
    if (getenv("HPC_PAPI_VERBOSE"))
      fprintf(stderr, "hpc_papi: cannot write %s\n", path);
    return;
  }
  hpc_papi_sysfs_str("/proc/sys/kernel/hostname", host, (int)sizeof host);
  fputs("{\"schema\": \"hpc_papi/1\", \"error\": ", out);
  hpc_papi_json_str(out, hpc_papi_err);
  fputs(", \"cause\": ", out);
  hpc_papi_json_str(out, hpc_papi_cause);
  fputs(", \"host\": ", out);
  hpc_papi_json_str(out, host);
  fprintf(out,
          ", \"fence\": \"%s\", \"threads\": %d, \"threads_counted\": %d, \"reps\": %d, "
          "\"elapsed_ns\": %lld, \"hardware_counters\": %d, \"smt\": %s, \"caveats\": [",
          HPC_PAPI_FENCE_NAME, hpc_papi_nthread, hpc_papi_nthread, hpc_papi_reps, hpc_papi_ns, hpc_papi_budget,
          smt_known ? (smt ? "true" : "false") : "null");
  hpc_papi_json_str(out, "a counted build is a diagnostic build: never ship it, and never compare its "
                         "wall clock against anything");
  if (!strcmp(HPC_PAPI_FENCE_NAME, "none")) {
    fputs(", ", out);
    hpc_papi_json_str(out, "no memory fence is emitted on this architecture, so buffered stores may "
                           "drift across the region boundary and land inside these counts");
  }
  if (getenv("OMP_WAIT_POLICY") && !strcmp(getenv("OMP_WAIT_POLICY"), "active")) {
    fputs(", ", out);
    hpc_papi_json_str(out, "OMP_WAIT_POLICY=active: idle workers SPIN at barriers and that spin is "
                           "counted as region cycles (measured 4.01x inflation on an imbalanced kernel)");
  }
  if (hpc_papi_nthread == 1) {
    fputs(", ", out);
    hpc_papi_json_str(out, "one OpenMP thread was registered, so these counts are one thread's share "
                           "-- check OMP_NUM_THREADS and whether the TU was built with -fopenmp");
  }
  fputs("], \"metrics\": [\n", out);
  for (m = 0; m < HPC_PAPI_NMETRIC; m++) {
    if (!first)
      fputs(",\n", out);
    first = 0;
    hpc_papi_write_metric(out, m);
  }
  fputs("\n]}\n", out);
  fclose(out);
}

int hpc_papi_finalize(void) {
  const char *path = getenv("HPC_PAPI_OUT");
  long long seen = 0;
  int m, i;

  if (hpc_papi_done)
    return hpc_papi_err[0] ? -1 : 0;
  hpc_papi_done = 1;
  if (hpc_papi_open)
    hpc_papi_stop();
  if (hpc_papi_live && !hpc_papi_reps)
    hpc_papi_fail(HPC_C_no_measured_rep, "no region was bracketed: hpc_papi_start and hpc_papi_stop "
                                         "were never paired, so nothing was counted");
  if (!hpc_papi_live && !hpc_papi_err[0])
    hpc_papi_fail(HPC_C_run_failed, "hpc_papi_init was never called, so no counter was ever armed");
  /* All zeros with an empty error is the one report a reader could misread as a fast kernel, so
   * it is made impossible here. A single counted zero stays exactly what it is. */
  for (m = 0; m < HPC_PAPI_NMETRIC && !hpc_papi_err[0]; m++)
    for (i = 0; i < hpc_papi_nthread; i++)
      if (hpc_papi_pick[m] >= 0 && hpc_papi_value(m, i))
        seen = 1;
  if (!hpc_papi_err[0] && !seen)
    hpc_papi_fail(HPC_C_no_measured_rep, "every armed metric read 0 on every thread: the counters "
                                         "armed but the bracketed region did not reach them");

  hpc_papi_write(path && path[0] ? path : "hpc_papi.json");

  if (hpc_papi_slots) {
#pragma omp parallel num_threads(hpc_papi_nthread)
    {
      hpc_papi_slot *slot = &hpc_papi_slots[omp_get_thread_num()];
#pragma omp critical(hpc_papi_setup)
      {
        if (slot->eventset != HPC_PAPI_NULLSET) {
          hpc_papi.cleanup_eventset(slot->eventset);
          hpc_papi.destroy_eventset(&slot->eventset);
        }
        hpc_papi.unregister_thread();
      }
    }
    free(hpc_papi_block);
    hpc_papi_block = NULL;
    hpc_papi_slots = NULL;
  }
  hpc_papi_live = 0;
  return hpc_papi_err[0] ? -1 : 0;
}

#endif /* HPC_PAPI_IMPLEMENTED */
#endif /* HPC_PAPI_IMPLEMENTATION */
'''


def header_text() -> str:
    """The whole generated header, byte for byte as the tracked file must be."""
    return BANNER + tables() + BODY


def counters(report: dict) -> dict:
    """The report as :func:`~hpcagent_bench.harness.profiling.render_counters` input.

    A rename and nothing else: the header writes
    :func:`~hpcagent_bench.harness.papi.counting_worker`'s row shape on purpose, so the renderers
    and :func:`~hpcagent_bench.harness.papi.derive` take it unchanged.
    """
    rows = report["metrics"]
    return {
        "group": "hpc_papi region",
        "threads": report["threads"],
        "threads_counted": report["threads_counted"],
        "smt": report["smt"],
        # ONE armed set, so one run -- not one run per metric. That is the whole difference from
        # the /profile path, and it is why every metric below is same-pass comparable.
        "runs": 1,
        "metrics": rows,
        "derived": papi.derive(rows),
    }


def read_report(path: pathlib.Path) -> List[str]:
    """A report as text: the error FIRST, then the counts, the ratios and the thread spread.

    The error comes first because a failed collection reports every count as 0, and a reader that
    reaches the table before the error reads a fast kernel out of a broken one.
    """
    report = json.loads(path.read_text())
    lines = [
        f"{path} -- schema {report['schema']}, host {report['host'] or '?'}, {report['threads']} thread(s), "
        f"{report['reps']} region pair(s), {report['elapsed_ns'] / 1e6:.3f} ms bracketed, "
        f"{report['fence']} fence"
    ]
    if report["error"]:
        return lines + [
            "", f"  ERROR ({report['cause']}): {report['error']}", "",
            "  every count in this report is 0 BECAUSE of that error, not because the kernel "
            "did nothing."
        ]
    if report["host"] and report["host"] != socket.gethostname():
        lines.append("  NOTE: " + FOREIGN_HOST.format(there=report["host"], here=socket.gethostname()))
    if report["smt"] is None:
        lines.append("  NOTE: /sys/devices/system/cpu/smt/active was unreadable on the counted host, so "
                     "whether two counted threads shared one core's caches is unknown")
    lines += [f"  caveat: {note}" for note in report["caveats"]]
    lines.append("  every count below came from ONE armed set in ONE run, so these metrics are directly "
                 "comparable; a metric listed as not fitting the registers needs a second run and is "
                 "comparable only through instructions or cycles")
    lines += profiling.render_counters(counters(report))
    cycles = next((row for row in report["metrics"] if row["metric"] == "cycles" and row.get("per_thread")), None)
    spread = papi.imbalance([v for v in cycles["per_thread"] if v > 0]) if cycles else None
    if spread:
        lines += [
            "", f"  thread imbalance {spread['max_over_mean']:.2f}x over {spread['threads']} working "
            f"thread(s) = {spread['formula']}", f"    {spread['reading']}"
        ]
    return lines


def main(argv: Sequence[str] = ()) -> int:
    """``--emit-header`` / ``--write`` / ``--read <report>``."""
    parser = argparse.ArgumentParser(prog="python -m hpcagent_bench.helpers.papi", description=__doc__)
    parser.add_argument("--emit-header", action="store_true", help="print the header to stdout")
    parser.add_argument("--write", action="store_true", help=f"regenerate {HEADER}")
    parser.add_argument("--read", type=pathlib.Path, metavar="REPORT", help="print a report's counts and ratios")
    args = parser.parse_args(list(argv) or None)
    if args.emit_header:
        sys.stdout.write(header_text())
    if args.write:
        HEADER.write_text(header_text())
        print(f"wrote {HEADER}")
    if args.read:
        print("\n".join(read_report(args.read)))
    if not (args.emit_header or args.write or args.read):
        parser.print_help()
        return 2
    return 0
