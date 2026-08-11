---
name: papi-gpu-amd
description: Count what an AMD GPU did inside ONE of your kernels with PAPI's rocp_sdk component -- start/stop per region, the two-sided empty/live self-test, and the environment traps that return silent zeros.
---

`rocprof` answers WHICH kernel owns device time. This page answers WHAT THE DEVICE DID while one
kernel ran: HBM bytes moved, L2 hits, waves launched, VALU busy. You bracket your own code, so the
answer is attributed to a region you chose rather than to a symbol.

This is the AMD twin of `papi-gpu`. The discipline is identical because the failure mode is
identical; the component, the event names and the environment traps are not.

## What was measured here, and what was not

**No AMD counter was ever read on the box this was written on.** There IS an AMD GPU on it -- a
Radeon 780M under ROCm 7.2.4 -- and that part exposes no performance counters at all, which is the
one AMD fact measured here and the whole of the next section. Every other AMD claim below is a quote
from a named upstream file with that file's URL beside it -- PAPI's `rocp_sdk` sources and README,
ROCm's `counter_defs.yaml`, rocprof-compute's `gfx942` panels. What could not be quoted was DELETED
rather than fenced: a warning label at the top does not tell you which fenced line was right.

What IS carried over from measurement is the METHOD: the start/stop-versus-read-delta result below
was measured on NVIDIA hardware here, against known ground truth. The MECHANISM behind it is a
different one on AMD -- see the end of that section -- which is why the self-test in
`gpu_papi_init` is what makes the method portable: it fails loudly on a box this page could not be
tested on. **Run it before you believe a number.**

## First, ask whether this part has counters AT ALL

```sh
rocprofv3-avail info --pmc          # ROCm >= 6.x; older: rocprofv3 --list-avail
```

Run it BEFORE you build anything. It asks the driver the same question PAPI does, and on a part
without counters it does not fail -- it returns an EMPTY LIST. Measured here, Radeon 780M (gfx1103,
RDNA3 integrated), ROCm 7.2.4:

```
W... metadata.cpp:254] rocprofiler_iterate_agent_supported_counters failed for agent 1 (gfx1103)
    :: Agent HW architecture is not supported, no counter metrics found.
GPU:0
Name:gfx1103
```

`rocminfo` reports the agent as `amdgcn-amd-amdhsa--gfx1103`, HIP runs kernels, everything works
except the thing this page is about. PAPI's component initialises, enumerates ZERO events, and every
name you try fails to resolve -- a build that took twenty minutes to produce nothing. `rocp_sdk`
supports CDNA2 and CDNA3 only, so consumer RDNA parts and APUs are out of scope by design and not by
bug. If that warning prints, stop here.

## Start and stop the event set per region -- a read-delta does NOT attribute

`PAPI_read` leaves the set counting and looks like it brackets a region. On a GPU component it does
not: on either vendor the read returns an ACCUMULATED total cut at a host-side boundary with no
fixed relationship to kernel completion, so a "delta" is the difference of two accumulations sliced
somewhere other than where you think. The two components get there differently -- the NVIDIA twin
closes and reopens the CUPTI range inside the read, `rocp_sdk` subtracts two snapshots of a buffer
an asynchronous callback fills -- and the AMD half is spelled out at the end of this section.

Measured on the NVIDIA twin of this component (RTX 4050, PAPI 7.2.0.0), four kernels of
deliberately different shape, 25 regions each, against each kernel's compulsory traffic:

| region | truth / rep | `PAPI_start`/`PAPI_stop` | read-delta |
| --- | --- | --- | --- |
| streams b and c into a | 128 MiB | **134.26 MB** | 128.4 MB |
| touches 64 KB, 64 launches | 64 KB | **77.9 KB** | 93.5 MB |
| reads a, 64 FMAs, writes a | 64 MiB | **67.08 MB** | 111.1 MB |
| reads a and c, divergent | 128 MiB | **134.27 MB** | 126.0 MB |

Start/stop lands on the compulsory traffic to within 0.1% on every row. The read-delta is wrong on
every row and wrong by **1300x** on the 64 KB one. Note what that does to a comparison: the true
spread across those four kernels is 2100x and the read-delta reports 1.2x. It does not add noise,
it FLATTENS the ranking you are profiling to find.

**PAPI's `rocp_sdk` README documents an asynchrony on AMD too**, in its own words: in dispatch mode
"PAPI may read zeros if reading takes place immediately after the return of a GPU kernel", because
"calls such as hipDeviceSynchronize() do not guarantee that ROCprofiler has been called and all
counter buffers have been flushed", so "it is recommended that the user code adds a delay between
the return of a kernel and calls to PAPI_read(), PAPI_stop(), etc"
(https://github.com/icl-utk-edu/papi/blob/master/src/components/rocp_sdk/README.md). A delay is a
race you cannot see losing -- too short and you read zero, slightly longer and you read a number
that looks fine and is not yours. Do not tune a sleep.

**`PAPI_stop` does NOT force that flush**, so do not port the NVIDIA sentence. `PAPI_stop` calls
`_papi_hwi_read()` and only then the component's `stop`
(https://github.com/icl-utk-edu/papi/blob/master/src/papi.c), and `rocp_sdk_stop` calls
`rocprofiler_sdk_stop` and drops the vendor context without reading anything
(https://github.com/icl-utk-edu/papi/blob/master/src/components/rocp_sdk/rocp_sdk.c) -- the value
you get at stop was read exactly the way `PAPI_read` reads it.

What start/stop buys on this component is a WINDOW WITH AN ORIGIN. `rocp_sdk_stop` sets
`vendor_ctx = NULL`, so the next `rocp_sdk_start` re-opens the vendor context, and
`rocprofiler_sdk_start` then zeroes `ctx->counters[i]` for every event before counting resumes
(https://github.com/icl-utk-edu/papi/blob/master/src/components/rocp_sdk/sdk_class.cpp). A
read-delta has no origin: it subtracts two snapshots of a buffer that `record_callback()` fills
ASYNCHRONOUSLY, so a record that lands late is charged to whichever read it beat. Close the range
with `PAPI_stop`, treat a zero as unproven rather than as a measurement, and check the region count.

Which failure you are exposed to depends on the MODE, and the default is not the kernel-attributed
one: `rocp_sdk` defaults to device sampling and dispatch mode is opt-in through
`PAPI_ROCP_SDK_DISPATCH_MODE=1` (README, above). ROCprofiler-SDK defines them as different
questions -- dispatch counting collects "on a per-kernel launch basis", device counting collects
"on a device level ... not tied to a specific kernel execution, which encompasses collecting
counter values for a specific time range"
(https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/counter_collection_services.html).
In the default mode your bracket is a TIME RANGE over the whole device, so anything else running on
that device lands inside it. That is what the empty half of the self-test below is checking for.

## Two components, and the old one is deprecated

```sh
papi_component_avail | grep -A2 -E 'Name:[[:space:]]+(rocm|rocp_sdk)'
```

| component | build | what it is, and where it works |
| --- | --- | --- |
| `rocp_sdk` | `--with-components="rocp_sdk"` | **default.** ROCprofiler-SDK. CDNA2 and CDNA3 ONLY -- MI210, MI250X, MI300A -- on ROCm 6.3.2 to 7.2.0 |
| `rocm` | `--with-components="rocm"` | DEPRECATED. rocprofiler v1, pre-MI300 parts, and only when `rocp_sdk` is unavailable |

Upstream: "The `rocm` component is deprecated starting at the AMD Instinct MI300A and will continue
to be for any future AMD device releases. Please instead use the `rocp_sdk` component", and "For AMD
devices older than the AMD Instinct MI300A, PAPI should not be configured with both `rocm` and
`rocp_sdk`" (https://github.com/icl-utk-edu/papi/blob/master/src/components/rocm/README.md). Neither
is built by default: like the `cuda` component, a distribution PAPI on a box with a perfectly good
GPU usually has neither, and rebuilding is the only fix -- next section. `rocm_smi` is a third
component and not a third choice: power, temperature and clocks, no performance counters, its own
root `PAPI_ROCMSMI_ROOT`.

## Building a PAPI that has an AMD component

`configure` lives in `src/`, not at the tarball root, and `--with-components` takes ONE quoted
space-separated list. The build itself needs no root and writes nothing outside `--prefix`.

**A runtime ROCm is not a buildable ROCm.** Both components compile against headers the runtime
packages do not ship: measured here, a stock `/opt/rocm-7.2.4` runtime install has neither
`include/hsa` nor `include/rocprofiler`, so a `rocm` build stops at `#include <rocprofiler.h>`.
`rocp_sdk` needs `hsa-rocr-dev` too even though it never touches rocprofiler v1, because
`include/rocprofiler-sdk/hsa.h` includes `<hsa/hsa.h>`. Installing those needs root, so ASK THE USER
to run it:

```sh
sudo apt install hsa-rocr-dev rocprofiler-sdk       # for rocp_sdk
sudo apt install hsa-rocr-dev rocprofiler-dev       # for the deprecated rocm component
```

```sh
curl -LO https://github.com/icl-utk-edu/papi/releases/download/papi-7-2-0-t/papi-7.2.0.tar.gz
tar xf papi-7.2.0.tar.gz && cd papi-7.2.0/src       # configure is HERE, not one level up
export PAPI_ROCP_SDK_ROOT=/opt/rocm                 # PAPI_ROCM_ROOT for the old component
./configure --prefix=$HOME/papi --with-components="rocp_sdk"
make -j8 && make install
```

`Rules.rocp_sdk` compiles `sdk_class.cpp` with `$(CXX)`, so a C++ compiler is a hard build
requirement for this component -- neither README says so. If ROCprofiler-SDK lives outside ROCm, set
`PAPI_ROCP_SDK_ROOT` and `PAPI_ROCM_ROOT` both.

The roots are read at RUN time too, and the two components differ in how forgiving they are:

- `rocp_sdk` dlopens `$PAPI_ROCP_SDK_ROOT/lib/librocprofiler-sdk.so`; `PAPI_ROCP_SDK_LIB` (a FULL
  path) takes precedence, and with neither set it falls back to a bare `dlopen`, i.e. ldconfig and
  `LD_LIBRARY_PATH`.
- `PAPI_ROCM_ROOT` is MANDATORY for `rocm` with NO ld.so fallback -- unset, the component disables
  itself with `Can't load libhsa-runtime64.so, PAPI_ROCM_ROOT not set.` It sets `HSA_TOOLS_LIB` and
  `ROCP_METRICS` for you; let it. Measured here on ROCm 7.2.4 the unversioned
  `lib/librocprofiler64.so` exists only with `rocprofiler-dev` installed, while PAPI's own search
  tries `.so.1` first and succeeds -- so an `HSA_TOOLS_LIB` you export by hand points at nothing.

`PAPI_ROCP_SDK_DISPATCH_MODE=1` selects per-kernel dispatch counting; the DEFAULT is device sampling
over a time range, which is a different question (see the first section). Export it, and call
`PAPI_library_init()` before any HIP call -- see the environment traps below, that ordering is the
one every other rule here depends on.

**The PAPI utilities are STATICALLY linked, so `papi_component_avail` reports its OWN build.**
`LD_LIBRARY_PATH` cannot move it, and a distro copy earlier on `PATH` will keep saying there is no
AMD component whatever you export. Call yours by absolute path; link the probe with an rpath or it
loads whichever `libpapi.so` `ld.so` finds first and then reports the same non-existent problem.

```sh
$HOME/papi/bin/papi_component_avail | grep -A3 -E 'Name:[[:space:]]+(rocm|rocp_sdk)'
$HOME/papi/bin/papi_native_avail -i rocp_sdk:::
hipcc -O2 -o probe probe.cpp -I$HOME/papi/include -L$HOME/papi/lib -lpapi -Wl,-rpath,$HOME/papi/lib
ldd ./probe | grep papi        # must be YOUR prefix
```

| symptom, exact string | fix |
| --- | --- |
| build: `hsa/hsa.h: No such file or directory` | ASK THE USER: `sudo apt install hsa-rocr-dev` |
| build: `rocprofiler.h: No such file or directory` | ASK THE USER: `sudo apt install rocprofiler-dev hsa-rocr-dev` |
| `\-> Disabled: Can't load libhsa-runtime64.so, PAPI_ROCM_ROOT not set.` | export `PAPI_ROCM_ROOT` at RUN time, not only at build time |
| `\-> Disabled: Could not dlopen() librocprofiler-sdk.so. Set either PAPI_ROCP_SDK_ROOT, or PAPI_ROCP_SDK_LIB.` | `export PAPI_ROCP_SDK_LIB=/opt/rocm/lib/librocprofiler-sdk.so` -- full path, wins over the root |
| `\-> Disabled: Invalid path in PAPI_ROCP_SDK_LIB: <path>` | an explicit override that does not open has no fallback. Fix it or unset it |
| `\-> Disabled: Rocprofiler metrics.xml file not found.` | `export ROCP_METRICS=$PAPI_ROCM_ROOT/lib/rocprofiler/metrics.xml` (ROCm >= 5.2.0) |
| `Could not obtain all functions from librocprofiler-sdk.so. Possible library version mismatch.` | ROCm older than 6.3.2, upstream's floor. Upgrade ROCprofiler-SDK |
| component Active, `Native: 0` | the part has no counters -- run the `rocprofv3-avail` check above |
| a probe prints "no component" while your own `papi_component_avail` shows it | it linked a different `libpapi.so`. Rebuild with `-Wl,-rpath` |

## The two environment traps that return silent zeros

Both produce a counter of 0 with no error anywhere, which reads exactly like a kernel that did no
work. This is the failure this whole page exists to prevent.

- **`AQLPROFILE_READ_API=0` is CONDITIONAL -- do not export it blind.** Upstream: "For ROCm >=
  6.2.0, the environment variable `AQLPROFILE_READ_API` should be set to 0 for intercept mode and 1
  (or unset) for sampling mode. Otherwise, counter values in intercept mode will return 0"
  (https://github.com/icl-utk-edu/papi/blob/master/src/components/rocm/README.md). Intercept mode is
  opt-in: the `rocm` component reads `ROCP_HSA_INTERCEPT` and falls back to sampling mode when it is
  unset (`roc_profiler.c`,
  https://github.com/icl-utk-edu/papi/blob/master/src/components/rocm/roc_profiler.c), and
  rocprofiler documents that variable as "if set then HSA dispatches intercepting is enabled"
  (https://rocm.docs.amd.com/projects/rocprofiler/en/latest/reference/rocprofiler_spec.html). The
  string does not appear anywhere in the `rocp_sdk` sources. Set it only if you deliberately chose
  intercept mode on the old component and are reading zeros.
- **`PAPI_library_init()` must run BEFORE any HIP call.** Upstream: "If an application is linked
  against the static PAPI library libpapi.a, then the application must call PAPI_library_init()
  through PAPI_add_named_event()/PAPI_add_event()/PAPI_enum_cmp_event() before calling any hip
  routines ... If the application is linked against the dynamic library libpapi.so, then the order
  of operations does not matter"
  (https://github.com/icl-utk-edu/papi/blob/master/src/components/rocp_sdk/README.md). The `rocm`
  component states the WHY: its environment exports "are read once by AMD with the first HIP
  function call, and if HIP sets up without them, PAPI may not read counters correctly." Static or
  not, the ordering costs nothing, so keep it.

That last one fights the CUDA rule, so do not port the ordering across: on NVIDIA you arm AFTER a
warmup launch because the component profiles through a live context. On AMD you initialise PAPI
FIRST. Same library, opposite order, and each is silent when you get it wrong.

## Event names

```sh
papi_component_avail                        # which of the two you actually have
papi_native_avail -i rocp_sdk:::            # every event THAT component enumerates
papi_native_avail -e rocp_sdk:::SQ_CYCLES   # ONE event, resolved, defaults filled in
```

**The prefix is the component name, and the two components do not share one.** `rocp_sdk.c`
declares `.name = "rocp_sdk"`
(https://github.com/icl-utk-edu/papi/blob/master/src/components/rocp_sdk/rocp_sdk.c), so events are
`rocp_sdk:::EVENT_NAME:device=N` -- upstream's own test runner spells them
`rocp_sdk:::SQ_CYCLES:device=0`
(https://github.com/icl-utk-edu/papi/blob/master/src/components/rocp_sdk/tests/run_rocp_sdk_tests.sh).
The deprecated component answers to `rocm:::`, so copying a `rocm:::` example onto a `rocp_sdk`
build resolves nothing. Enumerate first and use whatever prefix comes back.

Device indices run `[0, N-1]` over VISIBLE devices, so `ROCR_VISIBLE_DEVICES` renumbers them and a
resource manager that hands you a subset changes what `device=0` means. The `rocm` README says to
map it "Preferably the UUID of the device ... (see hipDeviceGetUuid and HSA_AMD_AGENT_INFO_UUID)"
rather than trusting the index; the same isolation applies here.

**`DIMENSION_*=` picks ONE instance of a multi-instance counter, and omitting it SUMS.** That is
the qualifier most examples leave off, and it silently changes the quantity. Upstream states the
rule in the component, of the records whose dimensions match your qualifiers: "This means that if a
qualifier is missing, we will get the sum"
(`sdk_class.cpp`,
https://github.com/icl-utk-edu/papi/blob/master/src/components/rocp_sdk/sdk_class.cpp). Upstream's
test runner shows the spelling, in either order:

```sh
rocp_sdk:::SQ_BUSY_CYCLES:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=0:device=0
rocp_sdk:::TCC_CYCLE:device=0:DIMENSION_INSTANCE=2
rocp_sdk:::SQ_BUSY_CYCLES:DIMENSION_INSTANCE=0   # no device= -- the component appends :device=0
```

Which dimensions an event HAS is per event, so enumerate rather than guess. The counter definitions
say the same thing from the hardware side: `SQ_WAVES` "Returns one value per-SE (aggregates of SIMD
values)"
(https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/share/rocprofiler-sdk/counter_defs.yaml),
so an unqualified `SQ_WAVES` is the sum over shader engines -- right for a total, wrong for a
per-SE comparison.

**A fractional value does not survive this component.** `record_callback()` sums the matching
records into a `double` and then accumulates that into `long long int *_counter_values` with `+=`
(`sdk_class.cpp`, above), and the comment there explains why it accumulates at all: "Rocprofiler-SDK
default behavior in dispatch mode is to only report the value of the counters since the dispatch of
the kernel. However, PAPI semantics dictate that counter values are only reset by PAPI_reset(), etc,
not by kernel invocations." Two consequences, both for derived metrics: a percentage is SUMMED over
the dispatches inside your bracket rather than averaged, and its fraction is truncated. Never
bracket a derived metric under `rocp_sdk`, and never bit-reinterpret the return as a `double` --
there is no `double` in there to recover.

Ask a QUESTION, then find the event that answers it on THIS device. A hard-coded event list is a
list that stops working: the names differ by generation, and CDNA and RDNA do not even agree on
what a wavefront is.

## The code

```c
#include <papi.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>

static int gpu_es = PAPI_NULL;
static long long gpu_total = 0;
static const char *gpu_event = NULL;
static int gpu_ok = 0, gpu_regions = 0;

/* Call FIRST, before ANY hip call -- see the environment traps above. */
static int gpu_papi_init(const char *event_name)
{
    gpu_ok = 0; gpu_total = 0; gpu_regions = 0; gpu_event = event_name;
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "papi-gpu-amd: library_init failed\n"); return -1;
    }
    int cid = -1;
    for (int i = 0; i < PAPI_num_components(); ++i) {
        const PAPI_component_info_t *ci = PAPI_get_component_info(i);
        if (ci && (!strcmp(ci->name, "rocp_sdk") || !strcmp(ci->name, "rocm"))) { cid = i; break; }
    }
    if (cid < 0) { fprintf(stderr, "papi-gpu-amd: no rocp_sdk/rocm component\n"); return -1; }
    int rc; long long probe = 0;
    /* A GPU event set must be bound to the GPU component; the default (0) is the CPU. */
    if ((rc = PAPI_create_eventset(&gpu_es)) != PAPI_OK) goto fail;
    if ((rc = PAPI_assign_eventset_component(gpu_es, cid)) != PAPI_OK) goto fail;
    if ((rc = PAPI_add_named_event(gpu_es, event_name)) != PAPI_OK) goto fail;
    /* HALF ONE of the self-test: arm and disarm around NOTHING. It surfaces a refusal HERE rather
       than at the first region, and the value must come back ~0. In device-sampling mode the
       bracket is a time range over the whole device, so an empty bracket reporting real work means
       you are counting the device, not your region -- STOP. Half two is in the caller: 0 over
       an empty bracket is the RIGHT answer, so this half cannot catch a dead counter. */
    if ((rc = PAPI_start(gpu_es)) != PAPI_OK) goto fail;
    if ((rc = PAPI_stop(gpu_es, &probe)) != PAPI_OK) goto fail;
    if (probe > 4096) {
        fprintf(stderr, "papi-gpu-amd: EMPTY BRACKET READ %lld, not ~0 -- not attributing\n", probe);
        return -1;
    }
    gpu_ok = 1;
    return 0;
fail:
    fprintf(stderr, "papi-gpu-amd: %s: %s (code %d)\n", event_name, PAPI_strerror(rc), rc);
    return -1;
}

/* START and STOP per region. PAPI_start opens the window -- it re-opens the vendor context and
   zeroes the counters -- and PAPI_stop reads it and closes it. A PAPI_read delta across the same
   span has no such origin and is not a measurement of that span. */
static void gpu_region_begin(void)
{
    if (gpu_ok && PAPI_start(gpu_es) != PAPI_OK) gpu_ok = 0;
}

/* Returns THIS region's value. Read a percentage from here, per region; never from gpu_total. */
static long long gpu_region_end(void)
{
    if (!gpu_ok) return 0;
    long long v = 0;
    if (PAPI_stop(gpu_es, &v) != PAPI_OK) { gpu_ok = 0; return 0; }
    gpu_total += v;                                /* ACCUMULATES -- only meaningful for a COUNT */
    ++gpu_regions;
    return v;
}

static void gpu_papi_forget(void)                  /* drop the self-test region from the totals */
{
    gpu_total = 0; gpu_regions = 0;
}

static void gpu_papi_report(void)
{
    if (!gpu_ok) { printf("%s = ERROR (not counted)\n", gpu_event ? gpu_event : "?"); return; }
    if (gpu_regions > 0 && gpu_total == 0)         /* known work, nothing counted: not a quiet kernel */
        printf("%s = SILENT ZERO over %d regions (not a measurement)\n", gpu_event, gpu_regions);
    else
        printf("%s = %lld   (regions: %d)\n", gpu_event, gpu_total, gpu_regions);
    PAPI_cleanup_eventset(gpu_es); PAPI_destroy_eventset(&gpu_es);
}
```

`gpu_total` accumulates across visits, so a 20 us kernel called 500 times is measurable without
changing what you measured. A `PAPI_start` after a `PAPI_stop` is a supported re-arm, not a leak:
the event set is created once and destroyed once.

**Accumulate EXTENSIVE counters only** -- `SQ_WAVES`, `FetchSize`, `WriteSize`, and cycle counts
like `GRBM_GUI_ACTIVE`. A sum of percentages is not a percentage, and the metrics you most want to
ask for are percentages upstream: `GPUBusy` is "The percentage of time GPU was busy", `L2CacheHit`
"The percentage of fetch, write, atomic, and other instructions that hit the data in L2 cache",
`VALUBusy` "The percentage of GPUTime vector ALU instructions are processed", `MemUnitStalled` "The
percentage of GPUTime the memory unit is stalled", `VALUUtilization` "The percentage of active
vector ALU threads in a wave"
(https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/share/rocprofiler-sdk/counter_defs.yaml).
Under this component they are worse than a bad average -- the component sums them across dispatches
and truncates (see "Event names") -- so do not bracket them at all. Take derived metrics from
`rocprofv3 --pmc <name>` and read raw ratios per region from what `gpu_region_end` returns.

**What makes the bracket work on AMD was NOT verified here.** The start/stop result above was
measured on NVIDIA; on `rocp_sdk` the mechanism is the re-opened, re-zeroed window described in the
first section, not a flush at `PAPI_stop`. Treat a zero as unproven rather than as a measurement,
and check the region count.

## How it runs

Use it:

```c
if (gpu_papi_init(argv[1]) != 0) return 2;  /* BEFORE any hip call -- see the traps */
gpu_region_begin();                         /* HALF TWO of the self-test: bracket KNOWN work */
your_kernel<<<grid, block>>>(...);          /* warmup */
hipDeviceSynchronize();
if (gpu_region_end() == 0) {                /* 0 over work that ran = silent zero, so refuse */
    fprintf(stderr, "papi-gpu-amd: LIVE BRACKET READ 0 -- not counting\n"); return 2;
}
gpu_papi_forget();                          /* the warmup is a probe, not a measurement */
for (int step = 0; step < nt; ++step) {
    gpu_region_begin();
    your_kernel<<<grid, block>>>(...);      /* ONE kernel per region */
    hipDeviceSynchronize();                 /* the kernel must have RUN -- and upstream says even
                                               this does not guarantee the buffers are flushed */
    gpu_region_end();
}
gpu_papi_report();
check_results();                            /* ALWAYS verify -- a wrong answer measures nothing */
```

Both halves matter and each catches what the other cannot: the empty bracket catches a counter
running device-wide, the live bracket catches a dead one. A guard that only fires when the number
is too BIG passes the silent zero this page exists to prevent -- pick an event your warmup kernel
must move, or the live half proves nothing.

```sh
hipcc -O2 -o probe probe.cpp -lpapi
```

One counter per run. Loop outside the program:

```sh
# EXTENSIVE counters only -- gpu_total accumulates across regions, and a sum of percentages is not
# a percentage. Derived metrics come from rocprofv3, not from this bracket.
for ev in rocp_sdk:::SQ_WAVES \
          rocp_sdk:::FetchSize \
          rocp_sdk:::WriteSize \
          rocp_sdk:::GRBM_GUI_ACTIVE; do
  ./probe "$ev:device=0"
done
```

## One region per kernel

A kernel launch returns immediately, so under a read-delta you would need a device synchronise to
have any hope of bracketing the kernel -- and, as the table above shows, it still would not work.
Under `PAPI_start`/`PAPI_stop` the WINDOW does not need the synchronise; the kernel HAVING RUN does,
which is why one sits inside the bracket. Do NOT port the NVIDIA page's "no sync of your own": there
it was measured to change nothing, here upstream says the sync is not even sufficient.

**A counted run's wall clock belongs to no comparison.** Upstream is explicit that the tooling
changes the schedule: "Counter collection in dispatch counting mode requires serialized execution
of kernels on a target device"
(https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/counter_collection_services.html),
which removes exactly the kernel/copy and kernel/kernel overlap a real run depends on -- about 2x on
the NVIDIA twin. Read the COUNTS; take every speedup from the uninstrumented build.

One kernel per region: two kernels in one bracket give you their sum, and a sum cannot be
attributed. Move the bracket and run again. Bracket INSIDE the timestep loop, not around it.

## Reading the numbers

The counts are yours; the THRESHOLDS below are vendor-doc reasoning, so calibrate on your own
kernel. Counters do not name a bottleneck. They eliminate candidates, in this order -- stop at the
first step that fires, because the later numbers are consequences of the earlier ones.

**1. Was the device even the problem?** If `rocprof` already showed device time well under the
wall clock, stop. Launch gaps and copies are host findings and no counter below moves them.

**2. Occupancy -- against the part, not against a number you remember.** The wavefront width is
the thing you must not assume. HIP: "The size of a warp is architecture dependent and always fixed:
64 threads for CDNA architectures [and] 32 threads for RDNA architectures"
(https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html), and
rocprof-compute repeats it where the counters are defined: "On AMD Instinct CDNA accelerators and
GCN GPUs, the wavefront size is always 64 work-items. Thus, the total number of wavefronts should be
equivalent to the ceiling of grid size divided by 64"
(https://github.com/ROCm/rocprofiler-compute/blob/develop/src/rocprof_compute_soc/analysis_configs/gfx942/0700_wavefront.yaml).
Every "threads per block for full occupancy" number you know from NVIDIA is off by that factor.

The wave-slot ceiling is per-architecture and this page will not guess it: gpuopen publishes "In
RDNA1, each SIMD has 20 slots available for assigned wavefronts" and "RDNA 2 and RDNA 3 have 16
slots per SIMD" (https://gpuopen.com/learn/occupancy-explained/) and no CDNA figure. Read your
part's from the agent listing, `rocprofv3 --list-avail`
(https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html), and
read residency as rocprof-compute defines it -- "The time-averaged number of wavefronts resident on
the accelerator over the lifetime of the kernel" (gfx942 wavefront panel, above).

High occupancy is not a goal. Occupancy counts waves PARKED, not waves working -- a kernel with
enough memory work in flight per wave runs at peak with half the slots empty.

**3. Memory stall, read WITH the traffic.** `MemUnitStalled` is "The percentage of GPUTime the
memory unit is stalled"; read it against `FetchSize` + `WriteSize`, which upstream defines as "The
total kilobytes fetched from the video memory" and "The total kilobytes written to the video
memory" -- KILOBYTES, not bytes, and that is the one unit trap on this vendor (`counter_defs.yaml`,
above).

| stall | traffic | what it is | what to change |
| --- | --- | --- | --- |
| high | low | LATENCY-bound: too few loads in flight | more occupancy, unroll, wider loads |
| high | high | BANDWIDTH-bound: the wire is the limit | move less -- tile for reuse, fuse, shrink the dtype |
| low | high | streaming at rate, nothing wasted | only an algorithmic change moves it |
| low | low | not memory at all | go to 5 |

**4. Traffic against the algorithm's minimum.** The most actionable number here, and it needs no
peak: work out how many bytes the kernel MUST move -- every input read once, every output written
once -- and divide the measured `FetchSize + WriteSize` by it.

- ratio near 1 -- compulsory. Tiling buys nothing; only a different algorithm does.
- ratio well above 1 -- you are re-reading data that should have stayed in cache. Check the L2 hit
  rate next (step 5). This is what a tiling or fusion change is for, and the ratio checks it worked.
- write bytes far above the output size -- uncoalesced stores, or a read-modify-write the source
  does not show.

**5. L2 hit rate -- and check your PART is in the definition.** `counter_defs.yaml` defines
`L2CacheHit` as `100*reduce(TCC_HIT,sum)/(reduce(TCC_HIT,sum)+reduce(TCC_MISS,sum))` for
gfx9/gfx900/gfx906/gfx908/gfx90a, and as the same shape over `GL2C_HIT`/`GL2C_MISS` for gfx10,
gfx11 and gfx12 -- a different cache block with different counter names, so a `TCC_*` request
returns nothing on RDNA rather than a wrong number. WARNING: that file lists NO `L2CacheHit`
definition for gfx940/gfx941/gfx942/gfx950, so on MI300 the derived metric does not exist; the raw
`TCC_HIT` and `TCC_MISS` do have gfx942 definitions there, so collect those two and divide, per
region. Read the ratio as the EXPLANATION of step 4, never on its own: a rising hit rate with
unchanged fetch bytes means you added accesses, not locality.

**6. Which pipe, last -- ask by NAME, never transcribe a formula.** `rocprofv3 --pmc VALUBusy --
./your_app` gives you the number the vendor stands behind: "The derived metrics are the counters
derived from the basic counters using mathematical expressions" and the tool evaluates them for the
part it is running on
(https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html).

This page prints no expressions, and neither should your notes: a copied formula is wrong in two
directions at once, and the counter NAMES resolve either way, so a wrong denominator returns a
plausible wrong number in silence. The expressions are per-ARCHITECTURE -- `VALUBusy` has one
definition for gfx9 through gfx950 and a different one for gfx12; `LDSBankConflict` is
`SQ_LDS_BANK_CONFLICT` over `GRBM_GUI_ACTIVE` on gfx9/gfx942 and `SQC_LDS_BANK_CONFLICT` over
`SQC_LDS_IDX_ACTIVE` on gfx10+ (`counter_defs.yaml`, above). And they are per-SOURCE: the legacy
`metrics.xml` still shipping with the old rocprofiler carries its own variants of the same names,
including one block that normalises `MemUnitStalled` by `GRBM_GUI_ACTIVE` and another that
normalises it by `ACTIVE_CYCLES`
(https://github.com/ROCm/rocprofiler/blob/amd-master/test/tool/metrics.xml). Pick the tool, not the
formula.

Two readings to be careful with, both of which invite an NVIDIA habit that does not transfer:

- `VALUUtilization` in `counter_defs.yaml` is lane occupancy within a wave -- "The percentage of
  active vector ALU threads in a wave. A lower number can mean either more thread divergence in a
  wave or that the work-group size is not a multiple of 64" -- the DIVERGENCE number.
  `rocprof-compute` prints something spelled almost identically, `VALU Utilization`, and defines it
  as the opposite quantity: "Indicates what percent of the kernel's duration the VALU was busy
  executing instructions." Its divergence metric is `VALU Active Threads`, "the average level of
  divergence within a wavefront over the lifetime of the kernel", in units of threads with peak
  `$wave_size`
  (https://github.com/ROCm/rocprofiler-compute/blob/develop/src/rocprof_compute_soc/analysis_configs/gfx942/1100_compute_units_compute_pipeline.yaml).
  Two tools, near-identical spellings, different quantities.
- Whichever you read, it is scaled by the wavefront width, so the SAME source branch reads
  differently on CDNA (64 lanes) and RDNA (32). Never compare it across parts.

## Comparing two counters -- they always came from different runs

One counter per run means every ratio spans two executions. That is only legitimate through **a
denominator BOTH runs measured**. Collect `rocp_sdk:::GRBM_GUI_ACTIVE` -- GPU active CYCLES -- in
every run, and divide each raw count by its OWN run's value before comparing. It is a duration, so
it is a normaliser and not evidence the two runs did the same work: a run that got slower has more
of them.

WARNING: Not `GPUBusy`. Upstream describes it as "The percentage of time GPU was busy" and defines
it as `100*reduce(GRBM_GUI_ACTIVE,max)/reduce(GRBM_COUNT,max)` -- a PERCENTAGE of time, not a cycle
count -- so dividing by it inverts the normalisation instead of applying it. Note also that its
architecture list stops at gfx90a and gfx12: on gfx940/gfx941/gfx942/gfx950 the identical expression
is published under the name `GPU_UTIL` instead (`counter_defs.yaml`, above). `GRBM_GUI_ACTIVE`
itself is a raw counter with a gfx942 definition, which is why it is the one to collect.

Same binary, same input, same grid is what makes two runs comparable. With all three held, an
active-cycle count that still moves by more than a few percent means something outside the code
moved, and no ratio built from those runs is trustworthy.

Two rules override all of it:

- **The kernel's work is the invariant.** If the fetch byte count moved between two versions meant
  to compute the same thing, recheck correctness before reading any other number.
- **A counter improving while the uninstrumented run gets slower is not an improvement.**

## Traps

- **A count of 0 is a measurement; ERROR is not.** The code prints `ERROR (not counted)` when setup
  failed. Read that line before the numbers. On this vendor a silent 0 is also what the environment
  traps produce, which is why the self-test refuses to continue.
- **The self-test has TWO halves and needs both.** `gpu_papi_init`'s empty bracket catches a counter
  running device-wide instead of attributing -- it reads back large. It cannot catch a dead counter,
  because 0 IS the right answer for an empty bracket, so the live bracket around the warmup catches
  that one by refusing a 0 over work that provably ran. `gpu_papi_report` repeats the check over the
  whole run: an all-zero total with the expected region count is the silent-zero failure, not a
  kernel that moved nothing.
- **A cache-resident working set reports near-zero HBM traffic, and that is CORRECT.** Before
  calling a traffic counter broken, scale the working set past the last-level cache and check the
  number tracks. On a part with a large MALL/Infinity Cache this bites at sizes that feel big.
- **`regions:` must be the launch count you expect.** Fewer means brackets were skipped.
- **The counted binary is not your submission.** Build the probe separately; submit the clean
  source.
- **Never run the probe under `rocprofv3` or `rocprof-compute`.** They are the same profiling client
  the component needs, and upstream does not share it: "There may only be one counting service
  configured per agent in a context and can be only one active context that is profiling a single
  agent at a time"
  (https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/_doxygen/rocprofiler-sdk/html/group__device__counting__service.html).
- **Do not port NVIDIA thresholds.** Wavefront width, LDS banking and the cache hierarchy all
  differ. A number that means "bad" on an SM does not mean it on a CU.

## Documentation

- PAPI project home -- https://icl.utk.edu/papi/
- PAPI `rocp_sdk` component: build flags, env vars, dispatch mode, the flushing-delay limitation --
  https://github.com/icl-utk-edu/papi/blob/master/src/components/rocp_sdk/README.md
- `rocp_sdk.c`: the component name, and what `start`/`stop` actually call --
  https://github.com/icl-utk-edu/papi/blob/master/src/components/rocp_sdk/rocp_sdk.c
- `sdk_class.cpp`: counter zeroing at start, the `+=` into `long long`, dimension-qualifier semantics --
  https://github.com/icl-utk-edu/papi/blob/master/src/components/rocp_sdk/sdk_class.cpp
- The component's own test runner -- the authority for event-name and `DIMENSION_*=` spelling:
  https://github.com/icl-utk-edu/papi/blob/master/src/components/rocp_sdk/tests/run_rocp_sdk_tests.sh
- `papi.c`: `PAPI_stop` reads before it stops -- https://github.com/icl-utk-edu/papi/blob/master/src/papi.c
- `rocp_sdk` support matrix: CDNA2/CDNA3, MI210/MI250X/MI300A, ROCm 6.3.2 to 7.2.0 -- https://github.com/icl-utk-edu/papi/wiki/Hardware-and-Software-Support-%E2%80%90-ROCP_SDK-Component
- PAPI `rocm` component (deprecated from MI300A), `AQLPROFILE_READ_API` -- https://github.com/icl-utk-edu/papi/blob/master/src/components/rocm/README.md
- PAPI `rocm_smi` component -- power, clocks, temperature, not counters -- https://github.com/icl-utk-edu/papi/blob/master/src/components/rocm_smi/README.md
- ROCprofiler-SDK, the library `rocp_sdk` sits on -- https://github.com/ROCm/rocprofiler-sdk
- ROCm installation, which packages ship headers versus runtime -- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
- `roc_profiler.c`: intercept mode is selected by `ROCP_HSA_INTERCEPT`, sampling is the fallback --
  https://github.com/icl-utk-edu/papi/blob/master/src/components/rocm/roc_profiler.c
- ROCprofiler-SDK counter collection services: dispatch versus device counting, kernel serialisation --
  https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/counter_collection_services.html
- `rocprofv3`: `--list-avail`, `--pmc`, and what a derived metric is --
  https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- Counter and derived-metric DEFINITIONS with their per-architecture expressions -- the authority
  for every counter description quoted above:
  https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/share/rocprofiler-sdk/counter_defs.yaml
- The legacy `metrics.xml`, which spells some of the same names differently --
  https://github.com/ROCm/rocprofiler/blob/amd-master/test/tool/metrics.xml
- rocprof-compute's per-panel metric definitions and UNITS, per part (`gfx942/*.yaml`) --
  https://github.com/ROCm/rocprofiler-compute/tree/develop/src/rocprof_compute_soc/analysis_configs
- MI300/MI200 counter definitions and units (note: this page gives no expressions) -- https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300-mi200-performance-counters.html
- Occupancy on AMD: wave slots per SIMD, RDNA1 and RDNA2/3 only -- https://gpuopen.com/learn/occupancy-explained/
- AMD Instinct MI300 (CDNA3) ISA reference, for the hardware numbers -- https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- HIP programming model: wavefront size per architecture -- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
