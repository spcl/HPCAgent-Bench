/* Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * hpcagent_papi.h -- hardware counters for a REGION you name, inside your own source.
 *
 * The whole surface is two calls:
 *
 *     hpc_papi_start("advect");
 *     ... the code you want counted ...
 *     hpc_papi_stop("advect");
 *
 * Nothing else: the library initialises itself on first use and prints its report when the process
 * exits, so there is no init to forget and no finalize to misplace.
 *
 * CALL THEM WHERE THE WORK HAPPENS. In a parallel kernel that means INSIDE the parallel region,
 * so every thread brackets its own work:
 *
 *     #pragma omp parallel
 *     {
 *         hpc_papi_start("advect");
 *         #pragma omp for
 *         for (int i = 0; i < n; ++i) { ... }
 *         hpc_papi_stop("advect");
 *     }
 *
 * This is not a style preference. A PAPI event set is bound to ONE thread's counter registers --
 * there is no "the process" to count -- so a set must be created, started and stopped by the same
 * OS thread that did the work. Bracketing a parallel region from OUTSIDE looks right and is not:
 * the runtime is free to hand the next region different OS threads, and the stop then reads a
 * thread that ran nothing. Measured while writing this header, that shape returned a clean
 * `"value": 0` with no error on most runs -- a zero that reads as free code.
 *
 * A serial region needs no parallel block; the team is then just one thread and the same two calls
 * do the right thing.
 *
 * WHY A TAG. `perf` can name what it found because the linker named it; a counter bracket has no
 * symbol at all. The harness cannot see a "region" inside your kernel, so you declare the scope and
 * the tag is what the report is keyed by. It is the counter half of what the `divide-and-conquer`
 * skill does with noinline: there the compiler carries the name, here you do.
 *
 * ONE METRIC PER RUN, chosen by the harness through HPCAGENT_PAPI_METRIC. A CPU has a handful of
 * counter registers, and asking for more than fits makes PAPI multiplex and hand back estimates
 * that read exactly like counts. So the harness replays the whole kernel once per metric and this
 * header counts exactly one -- which is also what keeps the regions in one report comparable: they
 * were counted in the same pass, on the same schedule.
 *
 * NO NESTING. A second start while this thread has a region open is an ERROR, whatever the tag:
 * one event set is live per thread, and overlapping regions would count the inner one twice.
 * Sequential regions are the supported shape, so a report is a list of tags and never a stack.
 *
 * WHAT COMES BACK. Per tag: the summed count over the threads that ran it, how many times it was
 * entered, how many threads counted, and the busiest thread's share -- because a sum hides the one
 * thing a parallel region is usually limited by. Absence is never a zero: anything that goes wrong
 * arrives as `fault`, and the report prints even when nothing could be counted at all.
 */

#ifndef HPCAGENT_PAPI_H
#define HPCAGENT_PAPI_H

#include <papi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef HPC_PAPI_MAX_REGIONS
#define HPC_PAPI_MAX_REGIONS 32
#endif

#ifndef HPC_PAPI_MAX_THREADS
#define HPC_PAPI_MAX_THREADS 256
#endif

/* The marker the harness greps for. Deliberately NOT the measurement child's own result prefix:
 * two parsers keying on one marker is how a workload's output gets read as a measurement. */
#define HPC_PAPI_PREFIX "HPCAGENT_PAPI "

#define HPC_PAPI_TAG_LEN 64

typedef struct {
  char tag[HPC_PAPI_TAG_LEN];
  long long value[HPC_PAPI_MAX_THREADS]; /* written only by the thread it belongs to */
  long long entries[HPC_PAPI_MAX_THREADS];
} hpc_papi_region;

typedef struct {
  int started;
  int usable;
  int event;
  int n;
  int sets[HPC_PAPI_MAX_THREADS];
  int open[HPC_PAPI_MAX_THREADS]; /* per THREAD: one region live at a time */
  int open_index[HPC_PAPI_MAX_THREADS];
  hpc_papi_region regions[HPC_PAPI_MAX_REGIONS];
  char metric[HPC_PAPI_TAG_LEN];
  char fault[256];
} hpc_papi_state;

static hpc_papi_state hpc_papi__s;

static void hpc_papi__report(void);

/* Record the FIRST thing that went wrong; the rest are its consequences. */
static void hpc_papi__fault_text(const char *text) {
#ifdef _OPENMP
#pragma omp critical(hpc_papi_fault)
#endif
  {
    if (!hpc_papi__s.fault[0]) {
      snprintf(hpc_papi__s.fault, sizeof(hpc_papi__s.fault), "%s", text);
      hpc_papi__s.usable = 0;
    }
  }
}

static void hpc_papi__fault(const char *what, int code) {
  char buf[256];
  const char *why = PAPI_strerror(code);
  snprintf(buf, sizeof(buf), "%s (PAPI %d: %s)", what, code, why ? why : "no message");
  hpc_papi__fault_text(buf);
}

static int hpc_papi__me(void) {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

/* Resolve the one metric and start the library. Serial, once, before any thread counts. */
static void hpc_papi__init(void) {
  const char *want = getenv("HPCAGENT_PAPI_METRIC");
  int code = 0;
  int rc = 0;
  int t = 0;

  hpc_papi__s.started = 1;
  hpc_papi__s.usable = 1;
  snprintf(hpc_papi__s.metric, sizeof(hpc_papi__s.metric), "%s", (want && *want) ? want : "PAPI_TOT_CYC");
  for (t = 0; t < HPC_PAPI_MAX_THREADS; ++t) {
    hpc_papi__s.sets[t] = PAPI_NULL;
  }
  /* FIRST, before anything below can fail: a fault that prints nothing is indistinguishable from
   * a run nobody instrumented, which is the one failure this header exists to prevent. */
  atexit(hpc_papi__report);

  rc = PAPI_library_init(PAPI_VER_CURRENT);
  if (rc != PAPI_VER_CURRENT) {
    hpc_papi__fault("PAPI_library_init", rc);
    return;
  }
#ifdef _OPENMP
  rc = PAPI_thread_init((unsigned long (*)(void))omp_get_thread_num);
  if (rc != PAPI_OK) {
    hpc_papi__fault("PAPI_thread_init", rc);
    return;
  }
#endif
  rc = PAPI_event_name_to_code(hpc_papi__s.metric, &code);
  if (rc != PAPI_OK) {
    hpc_papi__fault("this CPU cannot count the requested metric", rc);
    return;
  }
  hpc_papi__s.event = code;
}

/* This thread's event set, created on first use BY THIS THREAD -- which is the whole point. */
static int hpc_papi__set(int me) {
  int set = PAPI_NULL;
  int rc = 0;

  if (hpc_papi__s.sets[me] != PAPI_NULL) {
    return hpc_papi__s.sets[me];
  }
  rc = PAPI_register_thread();
  if (rc != PAPI_OK) {
    hpc_papi__fault("PAPI_register_thread", rc);
    return PAPI_NULL;
  }
  rc = PAPI_create_eventset(&set);
  if (rc != PAPI_OK) {
    hpc_papi__fault("PAPI_create_eventset", rc);
    return PAPI_NULL;
  }
  rc = PAPI_add_event(set, hpc_papi__s.event);
  if (rc != PAPI_OK) {
    hpc_papi__fault("PAPI_add_event", rc);
    return PAPI_NULL;
  }
  hpc_papi__s.sets[me] = set;
  return set;
}

static int hpc_papi__find(const char *tag) {
  int i = 0;
  int found = -1;

#ifdef _OPENMP
#pragma omp critical(hpc_papi_regions)
#endif
  {
    for (i = 0; i < hpc_papi__s.n; ++i) {
      if (strncmp(hpc_papi__s.regions[i].tag, tag, HPC_PAPI_TAG_LEN - 1) == 0) {
        found = i;
        break;
      }
    }
    if (found < 0 && hpc_papi__s.n < HPC_PAPI_MAX_REGIONS) {
      found = hpc_papi__s.n++;
      snprintf(hpc_papi__s.regions[found].tag, HPC_PAPI_TAG_LEN, "%s", tag);
    }
  }
  return found;
}

/* Begin counting the region called ``tag`` ON THIS THREAD. */
static inline void hpc_papi_start(const char *tag) {
  int me = 0;
  int idx = 0;
  int set = PAPI_NULL;
  int rc = 0;

#ifdef _OPENMP
#pragma omp critical(hpc_papi_init)
#endif
  {
    if (!hpc_papi__s.started) {
      hpc_papi__init();
    }
  }
  if (!hpc_papi__s.usable) {
    return;
  }
  me = hpc_papi__me();
  if (me >= HPC_PAPI_MAX_THREADS) {
    hpc_papi__fault_text("more threads than this header can count");
    return;
  }
  if (hpc_papi__s.open[me]) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "hpc_papi_start('%s') while '%s' is still open on this thread: regions do not "
             "nest, one event set is live per thread",
             tag ? tag : "", hpc_papi__s.regions[hpc_papi__s.open_index[me]].tag);
    hpc_papi__fault_text(buf);
    return;
  }
  idx = hpc_papi__find(tag ? tag : "");
  if (idx < 0) {
    hpc_papi__fault_text("more distinct tags than this header can hold");
    return;
  }
  set = hpc_papi__set(me);
  if (set == PAPI_NULL) {
    return;
  }
  rc = PAPI_start(set);
  if (rc != PAPI_OK) {
    /* An unchecked start is how a region reports 0 and reads as free code. */
    hpc_papi__fault("PAPI_start", rc);
    return;
  }
  hpc_papi__s.open[me] = 1;
  hpc_papi__s.open_index[me] = idx;
}

/* End the region called ``tag`` on this thread and add what it counted. */
static inline void hpc_papi_stop(const char *tag) {
  int me = 0;
  int idx = 0;
  long long got = 0;
  int rc = 0;

  if (!hpc_papi__s.started) {
    /* A stop with no start still has to REPORT: silence reads as uninstrumented. */
#ifdef _OPENMP
#pragma omp critical(hpc_papi_init)
#endif
    {
      if (!hpc_papi__s.started) {
        hpc_papi__init();
      }
    }
  }
  if (!hpc_papi__s.usable) {
    return;
  }
  me = hpc_papi__me();
  if (me >= HPC_PAPI_MAX_THREADS) {
    return;
  }
  if (!hpc_papi__s.open[me]) {
    char buf[256];
    snprintf(buf, sizeof(buf), "hpc_papi_stop('%s') with no region open on this thread", tag ? tag : "");
    hpc_papi__fault_text(buf);
    return;
  }
  idx = hpc_papi__s.open_index[me];
  if (tag && strncmp(hpc_papi__s.regions[idx].tag, tag, HPC_PAPI_TAG_LEN - 1) != 0) {
    char buf[256];
    snprintf(buf, sizeof(buf), "hpc_papi_stop('%s') closes '%s'", tag, hpc_papi__s.regions[idx].tag);
    hpc_papi__fault_text(buf);
    return;
  }
  rc = PAPI_stop(hpc_papi__s.sets[me], &got);
  hpc_papi__s.open[me] = 0;
  if (rc != PAPI_OK) {
    hpc_papi__fault("PAPI_stop", rc);
    return;
  }
  hpc_papi__s.regions[idx].value[me] += got; /* only this thread writes this slot */
  hpc_papi__s.regions[idx].entries[me] += 1;
}

/* One line the harness parses, printed at exit. JSON, because the harness owns the rendering --
 * every other measured child in this repo hands back a marker line and lets Python format it. */
static void hpc_papi__json(const char *text) {
  const unsigned char *p = (const unsigned char *)text;
  for (; *p; ++p) {
    switch (*p) {
    case '"':
      fputs("\\\"", stdout);
      break;
    case '\\':
      fputs("\\\\", stdout);
      break;
    case '\n':
      fputs("\\n", stdout);
      break;
    case '\r':
      fputs("\\r", stdout);
      break;
    case '\t':
      fputs("\\t", stdout);
      break;
    default:
      if (*p < 0x20) {
        printf("\\u%04x", *p);
      } else {
        putchar((int)*p);
      }
    }
  }
}

static void hpc_papi__report(void) {
  int i = 0;
  int t = 0;
  int counted_any = 0;
  int nonzero_any = 0;

  for (t = 0; t < HPC_PAPI_MAX_THREADS; ++t) {
    if (hpc_papi__s.open[t] && !hpc_papi__s.fault[0]) {
      char buf[256];
      snprintf(buf, sizeof(buf), "'%s' was never stopped on thread %d",
               hpc_papi__s.regions[hpc_papi__s.open_index[t]].tag, t);
      snprintf(hpc_papi__s.fault, sizeof(hpc_papi__s.fault), "%s", buf);
    }
  }
  printf("%s{\"metric\": \"", HPC_PAPI_PREFIX);
  hpc_papi__json(hpc_papi__s.metric);
  printf("\", \"regions\": [");
  for (i = 0; i < hpc_papi__s.n; ++i) {
    long long total = 0;
    long long entries = 0;
    long long busiest = 0;
    int counted = 0;
    for (t = 0; t < HPC_PAPI_MAX_THREADS; ++t) {
      long long v = hpc_papi__s.regions[i].value[t];
      if (hpc_papi__s.regions[i].entries[t] > 0) {
        counted += 1;
        counted_any = 1;
        entries += hpc_papi__s.regions[i].entries[t];
        total += v;
        if (v != 0) {
          nonzero_any = 1;
        }
        if (v > busiest) {
          busiest = v;
        }
      }
    }
    printf("%s{\"tag\": \"", i ? ", " : "");
    hpc_papi__json(hpc_papi__s.regions[i].tag);
    /* ``busiest`` is here because a SUM hides the finding a parallel region is usually limited
     * by: threads that split the work evenly and threads where one does most of it add up to
     * the same total. busiest/(total/counted) is that region's imbalance. */
    printf("\", \"value\": %lld, \"entries\": %lld, \"threads\": %d, \"busiest\": %lld}", total, entries, counted,
           busiest);
  }
  printf("]");
  /* Every region counted, every count zero. PAPI answered PAPI_OK throughout, so nothing above
   * is a fault -- and the numbers are still worthless: measured on a shared login node whose PMU
   * hands userspace nothing, PAPI's own papi_command_line reports 0 for PAPI_TOT_INS just as this
   * does. A report that stays silent here is a report that says a kernel executed no
   * instructions, which is the "absent is not zero" trap in its most convincing form. */
  if (hpc_papi__s.n > 0 && !hpc_papi__s.fault[0] && counted_any && !nonzero_any) {
    snprintf(hpc_papi__s.fault, sizeof(hpc_papi__s.fault),
             "every region counted 0 on every thread: this host's PMU delivered no counts for "
             "%s, so these are not measurements",
             hpc_papi__s.metric);
  }
  printf(", \"fault\": ");
  if (hpc_papi__s.fault[0]) {
    putchar('"');
    hpc_papi__json(hpc_papi__s.fault);
    printf("\"}\n");
  } else {
    printf("null}\n");
  }
  fflush(stdout);
}

#endif /* HPCAGENT_PAPI_H */
