# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Central matrix of build / runtime flags.

The values live here; the assembly lives in each
:class:`hpcagent_bench.framework.Framework` subclass. Frameworks compose by
referencing the constants below; they must NOT string-literal ``-O3``
or ``-march=native`` themselves (tests/test_no_literal_flags.py enforces this).

The matrix splits along three axes:

* :class:`Mode` -- the four evaluation modes a kernel can run in.
  Drives both the autopar selection on the CPU side and the choice of
  GPU backend.
* CPU compiler -- baseline flags per ``clang``, ``gcc``, ``icpx``.
* Autopar delta -- additional flag bundle to append for
  :attr:`Mode.MULTI_CORE` (Polly / GCC autopar / Pluto / NVHPC Mconcur).

GPU flags are kept tight (``CUDA_BASELINE`` / ``HIP_BASELINE``);
adding a new autopar / autovec knob is one constant + one referrer in
the framework's :meth:`compile_args`.
"""

from __future__ import annotations
import enum
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import tempfile
from functools import lru_cache
from typing import NamedTuple

from hpcagent_bench import config, osinfo, paths


class Mode(enum.Enum):
    """The four evaluation modes per kernel."""

    SINGLE_CORE = "single_core"
    MULTI_CORE = "multi_core"
    GPU_CUDA = "gpu_cuda"
    GPU_HIP = "gpu_hip"


# ---------------------------------------------------------------------------
# CPU compiler baselines (single source of truth for ``-O3``, ``-march=...``,
# math flags, PIC). Append-only -- changing a constant here propagates to
# every framework that references it.
# ---------------------------------------------------------------------------

# Two deliberate defaults live here. (1) -ffast-math is OFF: finite-math, reciprocal and approx-func
# rewrites change what a kernel computes. The FP-relax knobs below (no errno, no FP traps, no
# signed-zero preservation) are kept, and reassociation is a separate licence (see _FP_ASSOC).
# (2) -fopenmp is ON: single-core timing stays fair because flags.cpu_env pins OMP_NUM_THREADS=1
# outside MULTI_CORE. clang pins LLVM's own runtime (-fopenmp=libomp), the one whose calls it
# generates; the other drivers keep their plain OpenMP flag.
_FP_RELAX = "-fno-math-errno -fno-trapping-math -fno-signed-zeros"

#: Reassociation licence: permits reordering a floating-point reduction (``np.sum`` is pairwise, so
#: the oracle already does) and nothing else of -ffast-math. Pinned because gfortran reassociates at
#: _FP_RELAX alone while gcc (C), clang and flang do not; GCC ignores it without _FP_RELAX beside it.
#: OFF by default (``flags.fp_associative``): it moves the baseline every speedup is a ratio against,
#: so a campaign sets it for all of its waves or none. Read at import through :func:`config.get` or
#: ``$HPCAGENT_BENCH_FLAGS_FP_ASSOCIATIVE=1``; ``config.set_override`` comes too late to change it.
_FP_ASSOC = "-fassociative-math" if config.get("flags.fp_associative", False) else ""

#: FP contraction pinned to ``fast``: gcc and icx default to it, clang to ``on`` (one expression only),
#: so unpinned compiler columns, and a DaCe kernel that splits an expression across statements, would
#: differ in fma fusion. IEEE sanctions it (fma is correctly rounded); it does not imply -ffast-math.
_FP_CONTRACT = "-ffp-contract=fast"

#: nvc's spelling of the same thing: ``-Mfma``, on by default at ``-O2`` and above, so this states the
#: default. Unverified without the NVIDIA HPC SDK (INSTALL_NVHPC); ``containers/parallelizer-gate.sh``
#: checks it at image build.
_FP_CONTRACT_NVHPC = "-Mfma"

# OS/arch-aware pieces of the CPU baselines (Linux, macOS, WSL2 == Linux). (1) ``-march=native``
# everywhere except Apple-Silicon macOS, whose clang wants ``-mcpu=native``. (2) clang's ``libomp``
# pin is Linux-only; on macOS plain ``-fopenmp`` resolves to whatever runtime the compiler carries.
# (3) libmvec is glibc-only, reached by a different knob per compiler family (see below).
ARCH_NATIVE = "-mcpu=native" if (osinfo.IS_MACOS and osinfo.is_arm()) else "-march=native"
#: clang links LLVM's OWN runtime, not GNU's: libomp is what an LLVM toolchain ships and what
#: Polly's parallel backend is exercised against.
_OPENMP_CLANG = "-fopenmp=libomp" if osinfo.IS_LINUX else "-fopenmp"

#: The libmvec decl header handed to GCC (see the file for the full rationale).
VECMATH_H: pathlib.Path = paths.ROOT / "hpcagent_bench" / "envs" / "vecmath.h"

# glibc's vector libm, per compiler family. Both baselines carry it or neither, or the cc-vs-llvm
# column compares libmvec against scalar libm rather than gcc against clang.
#
# clang has a built-in flag; -Xarch_host confines it to the HOST pass, since an offload build
# otherwise dies with "unsupported option 'libmvec' for target 'amdgcn'". GCC has none (-mveclibabi=
# knows only acml/aocl/svml) and glibc's <bits/math-vector.h> gates the decls behind __FAST_MATH__,
# which cannot be faked (it sets _GLIBCXX_FAST_MATH=1 and flips math_errhandling), so GCC gets an
# equivalent decl header via -include. shlex.quote because {baseline} is expanded with shlex.split.
_VECLIB_CLANG = " -Xarch_host -fveclib=libmvec" if osinfo.IS_LINUX else ""
_VECLIB_GCC = f" -include {shlex.quote(str(VECMATH_H))}" if osinfo.IS_LINUX else ""

#: The optimization level every CPU baseline compiles at, named so that a DIAGNOSTIC tool -- e.g.
#: clang-tidy parsing generated source -- can request the same level without string-literalling it.
OPT_LEVEL = "-O3"

#: Clang baseline: -O3 + native arch + OpenMP + vectorized libm (no fast-math). The ``libomp`` pin
#: and ``libmvec`` apply on Linux only (see the OS-aware pieces above).
CPU_BASELINE_CLANG = (
    f"-O3 {ARCH_NATIVE} {_OPENMP_CLANG} {_FP_RELAX} {_FP_ASSOC} {_FP_CONTRACT} -fstrict-aliasing -fPIC{_VECLIB_CLANG}"
)

#: GCC baseline for C / C++: -O3 + native arch + OpenMP + vectorized libm (no fast-math).
#: The libmvec half arrives as a decl header (``_VECLIB_GCC``), not a flag -- gcc has no -fveclib.
CPU_BASELINE_GCC = (
    f"-O3 {ARCH_NATIVE} -fopenmp {_FP_RELAX} {_FP_ASSOC} {_FP_CONTRACT} -fstrict-aliasing -fPIC{_VECLIB_GCC}"
)

#: GCC baseline for Fortran: CPU_BASELINE_GCC minus the C decl header, which gfortran rejects (fatal
#: under -Werror). gfortran reaches libmvec through the distro driver spec pre-including glibc's
#: math-vector-fortran.h, a host property tests/test_vecmath.py checks.
CPU_BASELINE_GFORTRAN = f"-O3 {ARCH_NATIVE} -fopenmp {_FP_RELAX} {_FP_ASSOC} {_FP_CONTRACT} -fstrict-aliasing -fPIC"

#: NVHPC baseline for C / C++ / Fortran. ``_FP_RELAX`` and ``_FP_ASSOC`` need no nvc spelling: nvc
#: relaxes errno, trapping and signed zeros AND reassociates by default (``-Kieee`` turns that off).
#: ``-tp=native`` is its ``-march=native``, ``-mp`` its host ``-fopenmp``.
CPU_BASELINE_NVHPC = f"-O3 -tp=native -mp {_FP_CONTRACT_NVHPC} -fPIC"

#: nvc++ implements ``<execution>`` itself -- ``-stdpar=multicore`` is what makes ``par`` parallel,
#: and it is needed at COMPILE as well as at link. Without it ``par`` silently takes the sequential
#: overloads, the same failure ``STDPAR_LINK_TBB`` guards against on libstdc++.
CPU_BASELINE_NVCXX = f"{CPU_BASELINE_NVHPC} -stdpar=multicore"
STDPAR_LINK_NVHPC = "-stdpar=multicore"

#: nvhpc's optimization report. `-Minfo=all` covers vectorization, inlining and, on an offload
#: build, the `accel` channel that says which loops became kernels and which were refused.
NVHPC_OPT_REPORT = "-Minfo=all"

#: icx defaults to fp-model=fast; precise must come first (last spelling wins over _FP_RELAX).
#: ``-qopenmp`` is Intel's spelling of ``-fopenmp``, which it accepts with ``-Wrecommended-option``.
#: ``-Wno-overriding-option`` silences the per-TU notice that ``-ffp-contract=fast`` overrides the
#: contraction half of ``-fp-model=precise``; that override is intended (see ``_FP_CONTRACT``).
CPU_BASELINE_ICPX = (
    f"-O3 -xHost -fp-model=precise -qopenmp {_FP_RELAX} {_FP_ASSOC} {_FP_CONTRACT} "
    f"-Wno-overriding-option -fPIC -qopt-zmm-usage=high"
)

#: Appended to a PROFILED build (``Sandbox.build(debug=True)``, the /profile endpoint) so perf can
#: name the symbols it samples. Only ``-g``: it emits DWARF beside the code without changing it, so
#: a profiled build times identically to the scored one. No ``-fno-omit-frame-pointer`` -- perf
#: unwinds with DWARF here (perf_reports.PERF_CALL_GRAPH), and a frame pointer WOULD cost a register.
DEBUG_SYMBOLS: list[str] = ["-g"]

#: Pythran transpiles Python to C++ and forwards these flags to its backend compiler; kept in the
#: matrix so no framework string-literals them. ``-DUSE_XSIMD`` selects pythran's xsimd vector
#: backend; the rest match the CPU baseline, with NO ``-ffast-math``. ``_VECLIB_GCC`` rather than
#: ``_VECLIB_CLANG``: the decl header is accepted by gcc AND clang, ``-fveclib`` by clang only.
PYTHRAN_BASELINE = f"-DUSE_XSIMD -fopenmp {ARCH_NATIVE} {_FP_RELAX} {_FP_ASSOC} {_FP_CONTRACT} -fPIC{_VECLIB_GCC}"

#: LLVM Fortran (``flang`` / ``flang-new``) baseline, the Fortran companion to ``CPU_BASELINE_CLANG``
#: (no fast-math). flang rejects ``-fno-math-errno`` (a no-op for Fortran intrinsics) and has no
#: ``-fno-trapping-math`` spelling. ``-fno-signed-zeros`` rides WITH the licence: LLVM vectorizes a
#: reduction only with reassoc AND nsz, so ``-fassociative-math`` alone is silently ignored.
_FP_ASSOC_FLANG = f"{_FP_ASSOC} -fno-signed-zeros" if _FP_ASSOC else ""
FLANG_BASELINE = f"-O3 {ARCH_NATIVE} -fopenmp {_FP_ASSOC_FLANG} {_FP_CONTRACT} -fPIC"

#: flang's route to glibc's vector libm (no distro driver spec pre-includes it, unlike gfortran's).
#: PROBE-GATED at use (languages._veclib_accepted), since an older flang rejects it. Empty off Linux.
VECLIB_FLANG = "-fveclib=libmvec" if osinfo.IS_LINUX else ""

# ---------------------------------------------------------------------------
# Warnings -- a diagnostic axis, not an optimization one, so it is a separate
# constant appended after the baseline (``warnings_ref`` in compilers.yaml,
# resolved by languages._resolve_baseline the same way autopar is) rather than
# folded into CPU_BASELINE_*.
# ---------------------------------------------------------------------------

#: -Wall -Wextra, one spelling for gcc, g++, clang, clang++ and gfortran. Deliberately NOT
#: ``-Werror``: tests/test_warnings_ratchet.py tracks the warning count and only allows it down.
WARNINGS_BASIC = "-Wall -Wextra"

# ---------------------------------------------------------------------------
# C++ parallel algorithms (<execution>). LINK-side only, and only for the source that
# uses them -- see languages.stdpar_link_flags for when it is appended.
# ---------------------------------------------------------------------------

#: The runtime libstdc++ implements ``std::execution::par`` / ``par_unseq`` over; link-side only.
#: libstdc++ picks the backend per TU (``_GLIBCXX_USE_TBB_PAR_BACKEND __has_include(<tbb/tbb.h>)``):
#: with TBB headers the policies need libtbb, without them they run SERIAL and ``-ltbb`` is a link
#: error, so :func:`languages.stdpar_link_flags` asks the compiler the same ``__has_include`` question.
STDPAR_LINK_TBB = "-ltbb"

#: The allocator every graded C/C++ submission links, appended only when the toolchain resolves it:
#: on a host without it `-lmimalloc` fails EVERY build (the STDPAR_LINK_TBB trap), so
#: :func:`languages.mimalloc_link_flags` asks by linking.
LINK_MIMALLOC = "-lmimalloc"

# ---------------------------------------------------------------------------
# Multi-core autopar deltas. Each is appended on top of the CPU baseline.
# ``GCC_AUTOPAR`` and similar carry a ``{n}`` placeholder that
# :func:`compose_autopar` substitutes with the resolved core count.
# ---------------------------------------------------------------------------

#: LLVM Polly + OpenMP. ``-fopenmp=libomp`` pins clang to LLVM's own OpenMP runtime, the one its
#: codegen emits calls into.
#:
#: clang accepts these options whether or not Polly outlines anything, and a column that outlines
#: nothing is serial ``-O3`` under an autopar label. :func:`polly_capability` checks the compiled
#: object with ``nm``, and ``cpp_runtime.assert_autopar_capable`` declines a VACUOUS column
#: (``NotSupportedByFramework``). ``-polly-process-unprofitable`` and ``-polly-parallel-force`` are
#: BOTH required: the first passes the SCoP through the profitability heuristic, the second emits
#: parallel code for it; either alone outlines nothing.
POLLY_PAR = (
    f"-mllvm -polly -mllvm -polly-parallel -mllvm -polly-parallel-force "
    f"-mllvm -polly-process-unprofitable {_OPENMP_CLANG}"
)

#: GCC autopar + Graphite, the gcc counterpart of POLLY_PAR.
#:
#: ``-ftree-parallelize-loops={n}`` bakes N into ``GOMP_parallel`` and overrides ``OMP_NUM_THREADS``,
#: so :func:`ncores` fixes the RUN-time thread count at BUILD time.
#:
#: ``-floop-parallelize-all`` uses Graphite's dependence analysis; ``-fgraphite-identity`` and
#: ``-floop-nest-optimize`` enable SCoP detection and the polyhedral transforms. Graphite often
#: rejects the SCoP at its dependence stage (``cannot handle dependences``), leaving the object
#: unchanged; ``--param graphite-allow-codegen-errors=1`` would force it through with INCORRECT
#: codegen, so it is not set. Tests assert only that gcc ACCEPTS the flags
#: (tests/test_compile_flags.py), never that they change codegen.
GCC_AUTOPAR = "-ftree-parallelize-loops={n} -floop-parallelize-all -fgraphite-identity -floop-nest-optimize -fopenmp"

#: Native-construct threading: honor Fortran ``do concurrent``'s independence promise with real
#: threads, on every family. Appended to EVERY build of a block that declares ``doconcurrent_ref``
#: in compilers.yaml, regardless of build mode -- the run environment is always multi-core
#: (``native_call.grading_cpus``).
#:
#: - flang: lowers ``do concurrent`` ONLY (``__kmpc_fork_call``, honors OMP_NUM_THREADS; the
#:   "experimental" warning is normal). Needs LLVM >= 20.
#: - gfortran: parloops. Also threads any other loop it proves independent, identically on every
#:   arm. Thread count is FIXED at compile time from ``{n}``, sized like GCC_AUTOPAR.
#: - ifx: no extra flag; it threads ``do concurrent`` under the OpenMP flag in CPU_BASELINE_ICPX
#:   (Intel-documented, unverified here).
#: - nvfortran: ``-stdpar=multicore``. No compilers.yaml block references it until the opt-in
#:   NVIDIA HPC SDK layer is baked into the images.
DO_CONCURRENT_FLANG = "-fdo-concurrent-to-openmp=host"
DO_CONCURRENT_GFORTRAN = "-ftree-parallelize-loops={n}"
DO_CONCURRENT_NVFORTRAN = "-stdpar=multicore"

#: Pluto pre-processes the source; only OpenMP is added at compile time.
#:
#: ``polycc --parallel`` emits ``#pragma omp parallel for``, and clang accepts ``-fopenmp=libgomp``
#: while generating no OpenMP for that pragma (the ``=<lib>`` form wins over plain ``-fopenmp`` in
#: either order). The Pluto leg therefore spells plain ``-fopenmp``, and :func:`pluto_capability`
#: gates the column on the object actually referencing an OpenMP runtime.
PLUTO_PAR = "-fopenmp"

#: The Pluto column's clang baseline: :data:`CPU_BASELINE_CLANG` with the OpenMP spelling
#: swapped for the one that works (see :data:`PLUTO_PAR`). Written as a substitution rather than
#: a second literal so the two baselines cannot drift in any flag EXCEPT the one that must differ.
CPU_BASELINE_CLANG_PLUTO = CPU_BASELINE_CLANG.replace(_OPENMP_CLANG, PLUTO_PAR)

#: NVHPC pure-source CPU auto-parallelization (analogue of GCC ``-ftree-parallelize-loops``).
NVHPC_CONCUR = "-Mconcur"

# ---------------------------------------------------------------------------
# Autopar capability probe. An autopar flag set being ACCEPTED (compiles, links, runs) is not
# evidence it parallelizes anything, so the only evidence trusted here is ``nm`` on a compiled
# object: an undefined parallel-runtime reference, or a defined symbol matching the compiler's
# outline-body naming (Polly's ``*_polly_subfn``, GCC Graphite's ``*_loopfn``/``*._omp_fn``).
# ---------------------------------------------------------------------------


class AutoparVerdict(enum.Enum):
    """Three states, not a bool -- "accepted but useless" needs its own name, since that is
    exactly the failure mode this probe exists to catch (a bool cannot say it)."""

    REJECTED = "rejected"  #: the compiler/toolchain does not accept these flags at all.
    VACUOUS = "vacuous"  #: flags accepted, object built, but nothing was outlined.
    OK = "ok"  #: a parallel loop body was genuinely outlined.


class AutoparProbe(NamedTuple):
    """One probe result: the verdict plus the ``nm`` evidence (or compiler error) behind it,
    so a caller reporting "unavailable" can name the cause instead of just the verdict."""

    verdict: AutoparVerdict
    detail: str


#: THREE self-contained SCoPs (Static Control Parts) -- elementwise, a stencil nest, and a matmul
#: with an inner reduction -- because backends decline different shapes (clang + Polly declines the
#: matmul, gcc autopar the stencil). Outlining ANY of them is the verdict; a backend that outlines
#: none of the three is not parallelizing. Always C: Polly and Graphite work on middle-end IR, and
#: ``restrict`` is a C keyword but only a C++ extension.
_AUTOPAR_PROBE_SOURCE = """\
void axpy(double *restrict a, const double *restrict b, int n) {
  for (int i = 0; i < n; i++) a[i] = b[i] * 2.0;
}

void jac(double *restrict out, const double *restrict in, int n) {
  for (int i = 1; i < n - 1; i++)
    for (int j = 1; j < n - 1; j++)
      out[i * n + j] = 0.25 * (in[(i - 1) * n + j] + in[(i + 1) * n + j] + in[i * n + j - 1] + in[i * n + j + 1]);
}

void mm(double *restrict C, const double *restrict A, const double *restrict B, int n) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      double s = 0.0;
      for (int k = 0; k < n; k++) s += A[i * n + k] * B[k * n + j];
      C[i * n + j] = s;
    }
}
"""

#: A loop the source ALREADY marks parallel, for probing whether a compiler honours an explicit
#: ``#pragma omp parallel for`` -- what a source-to-source column (``polycc --parallel``) needs. Same
#: ``nm`` evidence: an object with no runtime call runs the loop serially (see :data:`PLUTO_PAR`).
_OPENMP_PROBE_SOURCE = """\
#include <omp.h>
void ax(double *restrict y, const double *restrict x, double a, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) y[i] += a * x[i];
}
"""

#: Undefined references that ARE a call into an OpenMP runtime: GNU ``libgomp`` spells them
#: ``GOMP_*``, LLVM ``libomp`` spells them ``__kmpc_*``. Both count -- the probe asks whether a
#: runtime is entered, not which vendor's.
OMP_RUNTIME_CALL_PATTERN = r"GOMP_|__kmpc_"

#: The same question for C++ ``<execution>`` policies, whose runtime is TBB rather than OpenMP.
#: libstdc++'s parallel algorithms dispatch into ``tbb::detail::r1::*`` (mangled ``_ZN3tbb...``);
#: the ``__TBB_`` alternative covers the C-linkage entry points other builds emit.
STDPAR_RUNTIME_CALL_PATTERN = r"_ZN3tbb|__TBB_"

#: Polly's outlined parallel body, e.g. ``mm_polly_subfn.0``.
POLLY_OUTLINE_PATTERN = r"polly_subfn"

#: One ``std::execution::par_unseq`` call -- what a ``cpp_isopar`` kernel IS. No flag to probe: what
#: varies is the backend libstdc++ picked (see :data:`STDPAR_LINK_TBB`), and a serial pick shows only
#: in whether the object calls a parallel runtime.
STDPAR_PROBE_SOURCE = """\
#include <algorithm>
#include <execution>
void ax(double *y, const double *x, int n) {
  std::transform(std::execution::par_unseq, x, x + n, y, y, [](double a, double b) { return a + b; });
}
"""

#: Matches no symbol at all -- for a probe whose only evidence is the OpenMP runtime call, because
#: the parallelism came from the SOURCE (a pragma) rather than from the compiler inventing an
#: outlined body it would then have to be recognised by name.
NO_OUTLINE_PATTERN = r"(?!)"

#: GCC Graphite / ``-ftree-parallelize-loops``'s outlined body, e.g. ``mm._loopfn.0`` or
#: ``mm._omp_fn.0`` (naming has varied across gcc versions; both are matched).
GCC_AUTOPAR_OUTLINE_PATTERN = r"_loopfn|\._omp_fn"


def _nm(nm_exe: str, args: list[str], obj: pathlib.Path) -> str | None:
    """``nm``'s stdout, or ``None`` if the invocation itself failed (unsupported flag, exotic
    object format, ...) -- distinguished from "ran and found nothing" so the caller can fail
    closed rather than misread a broken invocation as a clean zero count."""
    try:
        proc = subprocess.run([nm_exe, *args, str(obj)], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


@lru_cache(typed=True)
def probe_autopar(
    compiler: str,
    flags: str,
    outline_pattern: str,
    source: str = _AUTOPAR_PROBE_SOURCE,
    runtime_pattern: str = OMP_RUNTIME_CALL_PATTERN,
    suffix: str = ".c",
) -> AutoparProbe:
    """Does ``compiler flags`` genuinely outline a parallel loop, or merely accept the flags?

    Compiles ``source`` to an object in a fresh temp dir with ``compiler`` and ``flags`` (the
    column's REAL flags -- baseline + autopar delta, e.g. from :func:`compose_autopar`), then
    inspects the object with ``nm``. Nothing else counts as evidence: not the compiler's exit
    code beyond compiling, not whether a benchmark kernel later validates. ``outline_pattern``
    is a regex matched against ``nm``'s defined-symbol output (:data:`POLLY_OUTLINE_PATTERN` /
    :data:`GCC_AUTOPAR_OUTLINE_PATTERN`); an undefined ``runtime_pattern`` reference (a call into
    the parallel runtime -- :data:`OMP_RUNTIME_CALL_PATTERN` by default, either vendor's OpenMP)
    is independently sufficient, since a compiler could name its outlined body anything.

    ``source`` defaults to :data:`_AUTOPAR_PROBE_SOURCE` -- a plain nest the compiler must find
    parallelism in by itself. A source-to-source column passes :data:`_OPENMP_PROBE_SOURCE`
    instead, which already carries the pragma, so the question becomes whether the compiler
    honours it (see :func:`pluto_capability`).

    ``runtime_pattern`` and ``suffix`` exist because "parallel" is not always spelled OpenMP in
    C: a ``<execution>`` column enters TBB from C++ (:data:`STDPAR_RUNTIME_CALL_PATTERN`,
    ``.cpp``, see :func:`languages.isopar_capability`). Both stay parameters of THIS function
    rather than becoming a second probe, since the evidence -- compile, then ``nm`` -- is the
    same and only what counts as a runtime call differs.

    Parameterised by ``(compiler, flags, outline_pattern)`` rather than hardcoded per column,
    so a future autopar backend (Pluto, NVHPC ``-Mconcur``, ...) reuses this function instead
    of a bespoke check. ``@lru_cache(typed=True)`` -- this shells out to a compiler and must
    run once per process, not once per kernel.

    Degrades honestly where ``nm`` differs or is absent (macOS ships a BSD ``nm`` with a
    different flag surface; a stripped-down PATH may have none at all): with no ``nm`` to
    produce positive evidence, the verdict is :attr:`AutoparVerdict.VACUOUS` (fail CLOSED --
    "cannot confirm parallelism happened" must never read as "it did").
    """
    exe = shutil.which(compiler)
    if exe is None:
        return AutoparProbe(AutoparVerdict.REJECTED, f"{compiler!r} not found on PATH")
    with tempfile.TemporaryDirectory(prefix="hpcagent_bench_autopar_probe_") as tmp:
        src = pathlib.Path(tmp) / f"probe{suffix}"
        obj = pathlib.Path(tmp) / "probe.o"
        src.write_text(source)
        argv = [exe, *shlex.split(flags), "-c", str(src), "-o", str(obj)]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as e:
            return AutoparProbe(AutoparVerdict.REJECTED, f"failed to run {compiler}: {e}")
        if proc.returncode != 0 or not obj.is_file():
            return AutoparProbe(AutoparVerdict.REJECTED, f"compile rejected: {proc.stderr.strip()[-500:]}")

        nm = shutil.which("nm")
        if nm is None:
            return AutoparProbe(AutoparVerdict.VACUOUS, "nm unavailable on this host -- cannot confirm outlining")
        undefined = _nm(nm, ["-u"], obj)
        defined = _nm(nm, [], obj)
        if undefined is None or defined is None:
            return AutoparProbe(AutoparVerdict.VACUOUS, "nm invocation failed on this host -- cannot confirm outlining")

        runtime_calls = sum(1 for line in undefined.splitlines() if re.search(runtime_pattern, line))
        outlined = sum(1 for line in defined.splitlines() if re.search(outline_pattern, line))
        detail = f"runtime_calls={runtime_calls} outlined={outlined}"
        if runtime_calls > 0 or outlined > 0:
            return AutoparProbe(AutoparVerdict.OK, detail)
        return AutoparProbe(AutoparVerdict.VACUOUS, f"flags accepted, nothing outlined ({detail})")


def polly_capability() -> AutoparProbe:
    """The measured :class:`AutoparProbe` for THIS host's clang + :data:`POLLY_PAR`, at the
    real column's baseline+autopar flags (:func:`compose_autopar`)."""
    composed = compose_autopar(CPU_BASELINE_CLANG, POLLY_PAR, Mode.MULTI_CORE)
    return probe_autopar("clang", composed, POLLY_OUTLINE_PATTERN)


def gcc_autopar_capability() -> AutoparProbe:
    """The measured :class:`AutoparProbe` for THIS host's gcc + :data:`GCC_AUTOPAR`, at the
    real column's baseline+autopar flags (:func:`compose_autopar`)."""
    composed = compose_autopar(CPU_BASELINE_GCC, GCC_AUTOPAR, Mode.MULTI_CORE)
    return probe_autopar("gcc", composed, GCC_AUTOPAR_OUTLINE_PATTERN)


#: NVHPC's parallel runtime, for :func:`nvhpc_autopar_capability`: wider than
#: :data:`OMP_RUNTIME_CALL_PATTERN` because ``-Mconcur`` may enter NVIDIA's own (``__nv_*`` / ``_mp_*``).
NVHPC_RUNTIME_CALL_PATTERN = r"GOMP_|__kmpc_|__nv_|_mp_"


def nvhpc_autopar_capability() -> AutoparProbe:
    """The measured :class:`AutoparProbe` for THIS host's nvc + :data:`NVHPC_CONCUR`.

    Gates the ``cc_nvhpc_autopar`` column the same way :func:`polly_capability` gates Polly's, and
    for the same reason: ``-Mconcur`` is a request, not a guarantee, and an nvc that declines every
    loop hands back a serial object under a parallel label. Returns ``REJECTED`` when nvc is simply
    absent, which is the normal state of an image built without ``INSTALL_NVHPC=1``.

    UNVERIFIED against a real nvc -- the SDK is not in either CE image at the time of writing.
    That is precisely why this is a probe and not an assumption.
    """
    composed = compose_autopar(CPU_BASELINE_NVHPC, NVHPC_CONCUR, Mode.MULTI_CORE)
    return probe_autopar("nvc", composed, NO_OUTLINE_PATTERN, runtime_pattern=NVHPC_RUNTIME_CALL_PATTERN)


# Intel oneAPI has NO auto-parallelizer column: the LLVM-based icx accepts icc-classic's
# ``-parallel`` with warning #10430 and exit code 0, and emits no OpenMP runtime reference. An
# ``ICX_AUTOPAR`` constant would publish serial numbers under an auto-parallelizer's name, so the
# oneAPI arm is baseline-only (``cc_oneapi``).


def pluto_capability() -> AutoparProbe:
    """The measured :class:`AutoparProbe` for THIS host's clang at the Pluto column's REAL build
    flags (:data:`CPU_BASELINE_CLANG_PLUTO` + :data:`PLUTO_PAR`).

    Asks a different question than :func:`polly_capability`, because the Pluto column is
    source-to-source: polycc has ALREADY written ``#pragma omp parallel for`` into the code that
    gets compiled, so nothing needs to be auto-discovered. What must be true is that clang turns
    that pragma into a runtime call -- and the measured answer is not automatic (see
    :data:`PLUTO_PAR`: the shared clang baseline's OpenMP spelling drops the pragma in silence).
    Hence :data:`_OPENMP_PROBE_SOURCE` and no outline pattern to match: the OpenMP runtime call
    IS the evidence, and a host that produces none must not run this column at all rather than
    time Pluto's parallel output single-threaded under a parallel label."""
    composed = f"{CPU_BASELINE_CLANG_PLUTO} {PLUTO_PAR}"
    return probe_autopar("clang", composed, NO_OUTLINE_PATTERN, _OPENMP_PROBE_SOURCE)


# ---------------------------------------------------------------------------
# Optimization-report flags -- what the vectorizer DID and did NOT do, to stderr.
# Referenced by a compiler block's ``report_ref`` in ``compilers.yaml``. OFF by default: added only
# when a report is requested, and then only to the SEPARATE compile-only run that
# :func:`hpcagent_bench.benchmarks.cpp_runtime.opt_report_text` makes -- never to the timed build.
#
# Both compilers report to STDERR: GCC's ``=<file>`` form APPENDS across compiles and clang's
# ``-foptimization-record-file=`` CLOBBERS, while stderr gives both one capture path.
# ---------------------------------------------------------------------------

#: GCC / gfortran vectorization report. ``optimized`` carries the vector WIDTH, ``missed`` the
#: refusal REASON. Not ``-fopt-info-all`` (mostly non-vectorizer noise) and not
#: ``-fsave-optimization-record`` (gzip-JSON at several times the compile time, with no consumer).
GCC_OPT_REPORT = "-fopt-info-vec-optimized -fopt-info-vec-missed"

#: Clang / clang++ vectorization report. ``-Rpass*`` regexes match against PASS
#: names, so the vectorizer passes are named explicitly (``-Rpass=.*`` floods with
#: asm-printer noise). ``-Rpass-analysis`` is clang's counterpart of gcc's ``missed:`` reason line.
#: No ``-g`` is needed: the stderr diagnostics carry the frontend's own source location.
CLANG_OPT_REPORT = (
    "-Rpass=loop-vectorize|slp-vectorizer -Rpass-missed=loop-vectorize|slp-vectorizer -Rpass-analysis=loop-vectorize"
)

#: Intel oneAPI (icx / icpx / ifx) vectorization + parallelization report. Both phases are named:
#: ``vec`` is the counterpart of the two above, and ``par`` says what the OpenMP layer did, which
#: is the only route to threads this vendor has (see the note on the absent ``ICX_AUTOPAR``).
ICX_OPT_REPORT = "-qopt-report=3 -qopt-report-phase=par,vec"

# ---------------------------------------------------------------------------
# GPU baselines. The arch suffix (``-arch=sm_<SM>`` / ``--offload-arch=<gfx>``)
# is appended by the framework after :func:`detect_sm` / :func:`detect_gfx`.
# ---------------------------------------------------------------------------

#: NVCC baseline -- the host pass receives the CPU relax set via ``-Xcompiler`` and the device
#: pass keeps nvcc's IEEE defaults (``-prec-div``/``-prec-sqrt`` true, no flush-to-zero).
#: ``-arch=sm_<SM>`` is appended per-host by :func:`compose_cuda` after :func:`detect_sm`.
#: NO ``--use_fast_math`` and no host ``-ffast-math``: rule (1) at the top of this module holds on
#: the GPU too, since a GPU submission is graded against the same NumPy oracle and compared against
#: the same CPU baselines.
CUDA_BASELINE = f"-O3 -Xcompiler='-O3 -march=native {_FP_RELAX} {_FP_ASSOC} {_FP_CONTRACT} -fPIC'"

#: HIP (AMD) baseline -- hipcc is clang-based and takes the relax flags natively (no
#: ``-Xcompiler``), so one spelling covers its host and device passes. ``--offload-arch=<gfx>``
#: is appended per-host by :func:`compose_hip` after :func:`detect_gfx`. No ``-ffast-math``, for
#: the reason on :data:`CUDA_BASELINE`. ``-fopenmp`` because hipcc also builds the host entry
#: translation unit, whose ``#pragma omp`` lines are otherwise IGNORED and run serial.
HIP_BASELINE = f"-O3 -march=native -fopenmp {_FP_RELAX} {_FP_ASSOC} {_FP_CONTRACT} -fPIC"

# Directive-offload flag sets; ``{arch}`` filled by :func:`languages.offload_flags` from the arch
# :func:`languages.offload_arch` PROBED, never from a constant. One toolchain owns each model:
# LLVM offloads OpenMP on both vendor legs, NVHPC offloads OpenACC. No gcc legs: a gcc built
# ``--enable-offload-defaulted`` runs a target region on the HOST with no diagnostic.

#: CUDA compute capabilities, newest first. A VOCABULARY, not a per-compiler ceiling:
#: :func:`languages.offload_arch` walks DOWN it from the device's own capability until the compiler
#: accepts one, so a stale entry costs nothing and a new toolchain needs no edit.
#: Only NVIDIA gets a ladder. PTX is forward-compatible, so a lower ``sm_`` still runs on a higher
#: device; AMD has no such property (gfx1103 code does not run on gfx942), so an AMD offload arch is
#: matched EXACTLY or the leg is unsupported.
SM_LADDER: tuple[str, ...] = (
    "sm_121",
    "sm_120",
    "sm_110",
    "sm_103",
    "sm_100",
    "sm_90",
    "sm_89",
    "sm_87",
    "sm_86",
    "sm_80",
    "sm_75",
    "sm_70",
    "sm_62",
    "sm_60",
    "sm_53",
)

OMP_TARGET_LLVM_NVIDIA = "-fopenmp --offload-arch={arch}"
OMP_TARGET_LLVM_AMD = "-fopenmp --offload-arch={arch}"

OPENACC_NVHPC_NVIDIA = "-acc -gpu={arch}"

# ---------------------------------------------------------------------------
# Probes -- minimal, environment-overridable, fail-soft. Frameworks rely on
# these to fill the host-specific bits without each having to spawn its own
# ``nvidia-smi`` subprocess.
# ---------------------------------------------------------------------------

#: sysfs node listing the SMT siblings of a logical CPU, e.g. ``"0,8"`` for both halves of
#: one physical core. Two logical CPUs on the same core report the SAME string, which is
#: what makes it a physical-core key.
SIBLINGS = "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"


def physical_cores(cpus: set[int]) -> int:
    """The number of distinct PHYSICAL cores among the logical ``cpus``.

    Counts distinct SMT sibling groups, so a hyperthreaded pair collapses to the one core it
    really is. A cpu whose topology is unreadable (non-Linux, or a container that does not
    mount sysfs) counts as its own core -- the conservative reading, since the alternative is
    to merge cores that are actually distinct.
    """
    groups: set[str] = set()
    for cpu in cpus:
        try:
            with open(SIBLINGS.format(cpu=cpu)) as fh:
                groups.add(fh.read().strip())
        except OSError:
            groups.add(str(cpu))
    return len(groups)


def smt_enabled() -> bool:
    """Whether this machine runs more than one hardware thread per physical core.

    Sits beside :func:`physical_cores` because it is the same topology question asked the other
    way: logical cpus > physical cores means SMT is on. It matters for a HARDWARE COUNTER rather
    than for sizing -- two SMT siblings share the physical core's L1/L2, so a cache-miss count
    taken while a sibling is busy measures the pair, not the thread
    (:mod:`hpcagent_bench.harness.papi` pins around this and reports it either way).

    Machine-wide on purpose, not this process's share: a sibling belonging to somebody ELSE's
    process perturbs our counts exactly as much as one of ours would.
    """
    total = os.cpu_count() or 1
    return total > physical_cores(set(range(total)))


def ncores() -> int:
    """The number of physical cores available to THIS process, for OMP / autopar sizing.

    Three things this must get right, each of which it previously got wrong:

    1. PHYSICAL, not logical. ``os.cpu_count()`` counts hyperthreads, so on a 16-thread /
       8-core box it returned 16 and autopar was sized at 2x the real cores.
    2. THIS PROCESS's share, not the machine's. ``os.cpu_count()`` is affinity-blind: under
       ``taskset -c 0-3`` it still says 16. That matters most where it costs most -- one node
       with 288 cores running 4 ranks gives each rank 72, and a rank that reads 288 oversubscribes
       its cores 4x. ``sched_getaffinity`` sees the binding that SLURM/taskset/cgroups applied.
    3. The SLURM allocation when there is no binding to read. If the rank IS bound, affinity is
       exact and authoritative and SLURM is not consulted -- ``SLURM_CPUS_PER_TASK`` counts
       LOGICAL cpus, so dividing it by the SMT factor undercounts an allocation made with
       ``--hint=nomultithread``. It is used only when affinity still spans the whole machine,
       i.e. we were allocated a share but not confined to it.

    ``OMP_NUM_THREADS`` is deliberately NOT a source. It is a request rather than an
    allocation, and it is the very variable :func:`cpu_env` sets: reading it would let a
    parent's ``OMP_NUM_THREADS=1`` bake ``-ftree-parallelize-loops=1`` into a cached .so that
    every later multi-core run would then reuse. Never raises.
    """
    total = os.cpu_count() or 1
    try:
        allowed = os.sched_getaffinity(0)
    except AttributeError:  # macOS / Windows expose no affinity API
        allowed = set(range(total))
    env = os.environ.get("HPCAGENT_BENCH_NCORES")
    if env and env.isdigit():
        n = int(env)
        if n > 0:  # HPCAGENT_BENCH_NCORES=0 must NOT set OMP/autopar thread counts to 0
            # The submit script exports the NODE's SMT-free topology, which is what an unbound
            # process wants. A CONFINED one must still see only its share, or the override
            # re-creates the very oversubscription affinity exists to prevent.
            if len(allowed) < total:
                return max(1, min(n, physical_cores(allowed)))
            return n
    n = physical_cores(allowed)
    # Unbound: affinity tells us nothing about our share, so fall back to what SLURM says it
    # gave us (converted to cores at the machine's SMT width).
    if len(allowed) >= total:
        slurm = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm and slurm.isdigit() and int(slurm) > 0:
            smt = max(1, total // max(1, physical_cores(set(range(total)))))
            n = min(n, max(1, int(slurm) // smt))
    return max(1, n)


def detect_sm() -> str:
    """Return the CUDA compute capability of the local GPU as ``"sm_XX"``.

    Honours ``HPCAGENT_BENCH_SM`` override. When ``nvidia-smi`` is unavailable
    or fails, returns ``"sm_80"`` (Ampere) as a conservative default.
    """
    env = os.environ.get("HPCAGENT_BENCH_SM")
    if env:
        return env if env.startswith("sm_") else f"sm_{env}"
    try:
        out = (
            subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], timeout=5)
            .decode()
            .strip()
            .splitlines()
        )
        if out:
            cap = out[0].strip().replace(".", "")
            return f"sm_{cap}"
    except Exception:
        pass
    return "sm_80"


def detect_gfx() -> str:
    """Return the AMD GPU GFX target (e.g. ``"gfx90a"``).

    Honours ``HPCAGENT_BENCH_GFX`` override. Falls back to ``"gfx90a"``
    (MI210) when ``rocminfo`` is unavailable.
    """
    env = os.environ.get("HPCAGENT_BENCH_GFX")
    if env:
        return env
    try:
        out = subprocess.check_output(["rocminfo"], timeout=5).decode()
        m = re.search(r"Name:\s+(gfx\w+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "gfx90a"


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def cpu_env(mode: Mode, threads: int | None = None) -> dict[str, str]:
    """Return the env vars that pin thread counts for ``mode``.

    For :attr:`Mode.SINGLE_CORE` every well-known threading knob is
    forced to 1 (numpy + MKL + OpenBLAS + OpenMP) so a single-core
    measurement does not silently spill into BLAS-side parallelism.
    For :attr:`Mode.MULTI_CORE` they are set to :func:`ncores`.

    ``threads`` pins an EXPLICIT count instead of the mode's default -- the thread sweep
    :mod:`hpcagent_bench.harness.profiling` runs, which needs 1/2/4/... from one source of
    threading knobs rather than a second list of env var names.
    """
    n = str(threads) if threads else ("1" if mode is Mode.SINGLE_CORE else str(ncores()))
    return {
        "OMP_NUM_THREADS": n,
        "MKL_NUM_THREADS": n,
        "OPENBLAS_NUM_THREADS": n,
        "BLIS_NUM_THREADS": n,
    }


# ---------------------------------------------------------------------------
# Composition helpers -- frameworks call these instead of string-literal'ing.
# ---------------------------------------------------------------------------


def compose_autopar(baseline: str, autopar: str | None, mode: Mode, cores: int | None = None) -> str:
    """Append ``autopar`` to ``baseline`` when ``mode`` is :attr:`Mode.MULTI_CORE`.

    ``{n}`` becomes ``cores``, defaulting to :func:`ncores` for host probes. Grading callers pass
    :func:`languages.grading_ncores`: the compile is unpinned, the timed child is not.
    """
    if mode is not Mode.MULTI_CORE or autopar is None:
        return baseline
    return f"{baseline} {autopar.format(n=cores or ncores())}"


def compose_cuda(arch: str | None = None) -> str:
    """Build the NVCC / clang-CUDA flag string for the resolved SM."""
    return f"{CUDA_BASELINE} -arch={arch or detect_sm()}"


def compose_hip(arch: str | None = None) -> str:
    """Build the HIP flag string for the resolved GFX target."""
    return f"{HIP_BASELINE} --offload-arch={arch or detect_gfx()}"
