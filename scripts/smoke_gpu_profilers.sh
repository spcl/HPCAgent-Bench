#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Does every instrument the profiling skills send an agent to actually PRODUCE something here?
#
# Each check below exists because the tool has a way of failing that looks like a fast kernel: a
# profiler that exits 0 and writes nothing, a counter pass whose rows are all dropped downstream,
# a systems-profiler front end that is not the one that writes output. So nothing here asserts
# "the command ran" -- every check reads the artifact and, where the answer is derivable, checks
# the number against the launch geometry.
#
#   srun --partition=mi300 -N1 --environment=optarena-amd-mi300-v5 scripts/smoke_gpu_profilers.sh
#   sbatch scripts/submit_gpu_profiler_smoke.sbatch
#
# Exits non-zero naming the first instrument that could not answer. WORK defaults to a scratch
# directory; ARCH defaults to gfx942.
set -uo pipefail

WORK="${WORK:-${SCRATCH:-/tmp}/gpu-profiler-smoke}"
ARCH="${ARCH:-gfx942}"
REPS=20
BLOCK=256
LOG2N=22

failures=0
pass() { printf 'PASS  %-24s %s\n' "$1" "${2:-}"; }
fail() { printf 'FAIL  %-24s %s\n' "$1" "${2:-}"; failures=$((failures + 1)); }
skip() { printf 'SKIP  %-24s %s\n' "$1" "${2:-}"; }

mkdir -p "${WORK}" && cd "${WORK}" || { echo "cannot use WORK=${WORK}" >&2; exit 2; }

cat > smoke.hip <<'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void smoke_axpy(float a, const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}
int main() {
  const int n = 1 << 22;
  float *x = nullptr, *y = nullptr;
  if (hipMalloc(&x, n * sizeof(float)) != hipSuccess) return 2;
  if (hipMalloc(&y, n * sizeof(float)) != hipSuccess) return 2;
  (void)hipMemset(x, 0, n * sizeof(float));
  (void)hipMemset(y, 0, n * sizeof(float));
  for (int r = 0; r < 20; ++r)
    hipLaunchKernelGGL(smoke_axpy, dim3(n / 256), dim3(256), 0, 0, 2.0f, x, y, n);
  (void)hipDeviceSynchronize();
  printf("ok\n");
  (void)hipFree(x);
  (void)hipFree(y);
  return 0;
}
EOF

# Three stages of very different cost, each kept as its own symbol. This is the divide-and-conquer
# skill's whole mechanism: if perf cannot tell these apart, that page is wrong.
cat > stages.c <<'EOF'
#include <stdio.h>
#define N 2000000
static double a[N], b[N], c[N];
__attribute__((noinline)) static void stage_cheap(void) { for (int i = 0; i < N; ++i) c[i] = a[i] + b[i]; }
__attribute__((noinline)) static void stage_mid(void) {
  for (int r = 0; r < 8; ++r) for (int i = 0; i < N; ++i) c[i] += a[i] * b[i];
}
__attribute__((noinline)) static void stage_hot(void) {
  for (int r = 0; r < 60; ++r) for (int i = 0; i < N; ++i) c[i] = c[i] * 1.0000001 + a[i];
}
int main(void) {
  for (int i = 0; i < N; ++i) { a[i] = i * 1e-6; b[i] = i * 2e-6; }
  stage_cheap(); stage_mid(); stage_hot();
  printf("%f\n", c[N / 2]);
  return 0;
}
EOF

if ! hipcc -O3 --offload-arch="${ARCH}" -o smoke smoke.hip > build-hip.log 2>&1; then
  fail hipcc "see ${WORK}/build-hip.log"
else
  pass hipcc "${ARCH}"
fi
if ! "${CC:-gcc}" -O3 -g -o stages stages.c > build-c.log 2>&1; then
  fail cc "see ${WORK}/build-c.log"
else
  pass cc "-O3 -g"
fi

# --- rocprofv3, the trace the harness itself runs -------------------------------------------
rm -rf trace
if rocprofv3 --kernel-trace --memory-copy-trace --stats --output-format csv \
     --output-directory trace --output-file gpu -- ./smoke > trace.log 2>&1; then
  stats="$(find trace -name '*kernel_stats.csv' | head -1)"
  if [ -n "${stats}" ] && grep -q smoke_axpy "${stats}"; then
    calls="$(python3 -c "
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if 'smoke_axpy' in r['Name']]
print(sum(int(r['Calls']) for r in rows))
" "${stats}")"
    if [ "${calls}" = "${REPS}" ]; then
      pass rocprofv3-trace "${calls} dispatches"
    else
      fail rocprofv3-trace "kernel_stats says ${calls} calls, the program launched ${REPS}"
    fi
  else
    fail rocprofv3-trace "no kernel_stats row for the kernel that ran"
  fi
else
  fail rocprofv3-trace "exited non-zero; see ${WORK}/trace.log"
fi

# --- rocprofv3 --pmc: device counters, and the one number we can derive ----------------------
rm -rf counters
if rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU --kernel-include-regex smoke_axpy \
     --output-format csv --output-directory counters --output-file cnt -- ./smoke > counters.log 2>&1; then
  cc_csv="$(find counters -name '*counter_collection.csv' | head -1)"
  if [ -n "${cc_csv}" ]; then
    # Wavefronts are 64 lanes wide on CDNA, so REPS launches of 2^LOG2N work-items is a number
    # we know before measuring. A counter that does not reconcile with it has been misread.
    expected=$(( REPS * (1 << LOG2N) / 64 ))
    got="$(python3 -c "
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['Counter_Name'] == 'SQ_WAVES']
print(int(sum(float(r['Counter_Value']) for r in rows)))
" "${cc_csv}")"
    if [ "${got}" = "${expected}" ]; then
      pass rocprofv3-pmc "SQ_WAVES ${got} = ${REPS} x 2^${LOG2N} / 64"
    else
      fail rocprofv3-pmc "SQ_WAVES ${got}, geometry says ${expected}"
    fi
  else
    fail rocprofv3-pmc "no counter_collection.csv"
  fi
else
  fail rocprofv3-pmc "exited non-zero; see ${WORK}/counters.log"
fi

# --- rocprof-sys-sample: the timeline. -run is NOT this and writes nothing -------------------
rm -rf timeline
if rocprof-sys-sample --output "${WORK}/timeline" -- ./smoke > timeline.log 2>&1; then
  proto="$(find timeline -name 'perfetto-trace-*.proto' -size +1k | head -1)"
  if [ -n "${proto}" ]; then
    pass rocprof-sys-sample "$(basename "${proto}")"
  else
    fail rocprof-sys-sample "ran and wrote no perfetto trace (is this rocprof-sys-run?)"
  fi
else
  fail rocprof-sys-sample "exited non-zero; see ${WORK}/timeline.log"
fi

# --- rocprof-compute: thirteen passes are worth nothing if the rows are dropped --------------
rm -rf workloads
if rocprof-compute profile -n smoke --no-roof -- ./smoke > compute.log 2>&1; then
  if grep -q "Error converting" compute.log; then
    fail rocprof-compute "counter passes ran and were DROPPED converting v3 CSV (pandas mismatch)"
  else
    perf_csv="$(find workloads -name 'pmc_perf.csv' | head -1)"
    if [ -n "${perf_csv}" ] && [ "$(wc -l < "${perf_csv}")" -gt 1 ]; then
      wl="$(dirname "${perf_csv}")"
      if rocprof-compute analyze -p "${wl}" --block 6.2 > analyze.log 2>&1 \
         && ! grep -q "No profiling data found" analyze.log; then
        pass rocprof-compute "$(wc -l < "${perf_csv}") counter rows, analyze populated"
      else
        fail rocprof-compute "profiled but analyze found no data; see ${WORK}/analyze.log"
      fi
    else
      fail rocprof-compute "no pmc_perf.csv with rows; see ${WORK}/compute.log"
    fi
  fi
else
  fail rocprof-compute "exited non-zero; see ${WORK}/compute.log"
fi

# --- perf: does a named stage rank apart from its neighbours? --------------------------------
paranoid="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unknown)"
if [ "${paranoid}" != "unknown" ] && [ "${paranoid}" -le 2 ] 2>/dev/null; then
  if perf record -q -e cycles:u --call-graph=dwarf -F 999 -o stages.data -- ./stages > perf.log 2>&1; then
    hot="$(perf script -i stages.data -F comm,ip,sym --no-inline 2>/dev/null \
           | grep -o 'stage_[a-z]*' | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')"
    seen="$(perf script -i stages.data -F comm,ip,sym --no-inline 2>/dev/null \
            | grep -o 'stage_[a-z]*' | sort -u | wc -l)"
    if [ "${hot}" = "stage_hot" ] && [ "${seen}" -ge 2 ]; then
      pass perf-stage-split "${seen} stages resolved, hottest is stage_hot"
    else
      fail perf-stage-split "hottest symbol ${hot:-none}, ${seen} stages resolved"
    fi
  else
    fail perf-stage-split "perf record failed; see ${WORK}/perf.log"
  fi
else
  skip perf-stage-split "perf_event_paranoid=${paranoid}"
fi

# --- PAPI: CPU counters are the half that is meant to work here ------------------------------
if command -v papi_component_avail > /dev/null 2>&1; then
  if papi_component_avail 2>/dev/null | grep -A1 "^Active components:" | grep -q perf_event; then
    pass papi-cpu "perf_event component active"
  else
    fail papi-cpu "no active perf_event component"
  fi
  # Device counters through PAPI are NOT expected here; report the state without failing on it.
  if papi_component_avail 2>/dev/null | grep -qE "Name:[[:space:]]+(rocm|rocp_sdk)\b"; then
    pass papi-gpu "a rocm component is compiled in"
  else
    skip papi-gpu "no rocm/rocp_sdk component -- rocprofv3 --pmc is the device-counter route"
  fi
else
  fail papi-cpu "papi_component_avail not on PATH"
fi

echo
if [ "${failures}" -eq 0 ]; then
  echo "gpu profiler smoke: every instrument answered"
else
  echo "gpu profiler smoke: ${failures} instrument(s) could not answer"
fi
exit "$(( failures > 0 ))"
