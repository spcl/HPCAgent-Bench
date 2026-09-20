# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Running ``ppcg``: the ONE place the PPCG column's source-to-source step is spelled.

PPCG (Verdoolaege et al., TACO 2013, doi 10.1145/2400682.2400713) is the GPU sibling of Pluto:
same polyhedral front end (pet + isl), same ``#pragma scop`` input, CUDA out instead of tiled
OpenMP C. The two columns are the polyhedral pair -- Pluto on CPU, PPCG on GPU -- so the scop
selection and the affine guard are IMPORTED from :mod:`hpcagent_bench.pluto_transform` rather than
restated: a scop one column refuses is a scop the other cannot legally transform either.

PPCG emits TWO files per input (``<stem>_host.cu`` carrying the driver and ``<stem>_kernel.cu``
carrying the kernels) plus a ``<stem>_kernel.hu`` header they share. Both .cu files compile; the
header does not, so it is not returned.

PPCG HAS NO AMD TARGET -- ``--target`` takes ``c``, ``cuda`` or ``opencl`` and nothing else -- so a
HIP build translates what it wrote with ROCm's own ``hipify-perl`` and compiles that with hipcc. Same
polyhedral transform, same measurement; only the GPU it lands on differs.

Three columns share this one transform, differing only in which vendor they hand its output to:
``ppcg_cuda`` and ``ppcg_hip`` NAME theirs, so a result row says which GPU produced it without the
reader having to know what was installed on the node; bare ``ppcg`` names none and follows the local
toolchain (:func:`hpcagent_bench.languages.gpu_backend`). A named column that the host cannot build
declines through :class:`NotSupportedByFramework` rather than quietly measuring the other vendor.
"""

import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

from hpcagent_bench.frameworks.errors import NotSupportedByFramework, ToolMissing
from hpcagent_bench.languages import LANG_EXT, gpu_backend
from hpcagent_bench.pluto_transform import assert_affine, scop_inputs

#: The framework name this module transforms for -- used in every decline message.
FRAMEWORK = "ppcg"

#: How ``ppcg`` is invoked. ``--target=cuda`` is the whole point of the column; the two tile-size
#: knobs are ppcg's own defaults spelled out, so a host that changes them cannot silently change
#: what this column measures.
PPCG_ARGS: Tuple[str, ...] = ("--target=cuda", "--tile", "--tile-size=32")

#: ROCm's CUDA-to-HIP source translator, run on ppcg's output on an AMD host (see the module
#: docstring). Named once so the decline message and the call cannot disagree about the tool.
HIPIFY = "hipify-perl"


#: Direct override: a specific ``ppcg`` prefix (the directory holding ``bin/ppcg``), for a host
#: that built it somewhere :func:`ppcg_exe` would not otherwise look.
PPCG_HOME_ENV = "HPCAGENT_BENCH_PPCG_HOME"

#: Where a build script installs tools this image does not ship (see ``scripts/cache_env.sh``).
#: ``ppcg`` is not in the agent/judge image yet -- it needs building from source -- so this is the
#: env-var indirection the module docstring's "honour a prefix rather than a hardcoded path" means:
#: a cache path lives in ONE place (a shell resolver, keyed by ``$SCRATCH``, never a literal in this
#: file, which the no-hardcoded-user-paths lint would refuse anyway) and this file only reads it.
TOOLS_DIR_ENV = "HPCAGENT_BENCH_TOOLS_DIR"


def _exe_under_prefix(prefix: Optional[str], name: str) -> Optional[str]:
    """``<prefix>/bin/<name>`` if that file exists and is executable, else ``None``."""
    if not prefix:
        return None
    candidate = pathlib.Path(prefix) / "bin" / name
    return str(candidate) if os.access(candidate, os.X_OK) else None


def _ppcg_run_env(exe: str) -> Optional[Dict[str, str]]:
    """The environment ``ppcg`` must run under: its OWN ``lib`` dir prepended to
    ``LD_LIBRARY_PATH``, or ``None`` when that directory does not exist (an ordinary system
    install, where the default environment is already correct and this would be a no-op anyway).

    ppcg links pet and isl by an absolute RPATH into its own install prefix -- correct, and
    verified with ``readelf -d`` on the binaries this built -- but RPATH/RUNPATH is the THIRD
    place the dynamic linker looks, after ``LD_LIBRARY_PATH``. This image's own EDF sets
    ``LD_LIBRARY_PATH`` to include ``/usr/lib/x86_64-linux-gnu`` (needed for the CXI hook's
    libcurl -- see the EDF's own comment), which is exactly where Ubuntu packages ``libisl23`` as
    a build dependency of gcc's Graphite pass. That copy shares ppcg's isl's SONAME
    (``libisl.so.23``) but predates ``isl_id_set_alloc``, so with LD_LIBRARY_PATH left alone the
    loader finds the OLDER system isl first and ppcg dies at startup: ``ppcg: symbol lookup
    error: .../libpet.so.10: undefined symbol: isl_id_set_alloc`` (measured, job 640113). The
    same clash exists whether ppcg lives in the shared tools cache or is baked into the image
    (containers/cluster/ce-images/judge-agent-amd/Dockerfile's PPCG stage) -- it is a property of
    the SEARCH ORDER, not of where ppcg was installed -- so this runs unconditionally rather than
    only for a cache build.
    """
    lib = pathlib.Path(exe).resolve().parent.parent / "lib"
    if not lib.is_dir():
        return None
    env = dict(os.environ)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib}:{existing}" if existing else str(lib)
    return env


def ppcg_lookup() -> Tuple[Optional[str], str]:
    """The ``ppcg`` this host will actually run, as ``(exe, "")`` -- or ``(None, why not)``.

    Candidates in order: :data:`PPCG_HOME_ENV`, ``<tools dir>/ppcg/bin/ppcg``, PATH. The first two
    never fire on a host that has not built ppcg, so a host that installs ``ppcg`` onto PATH the
    ordinary way is unaffected; what they add is that a ppcg built into
    ``$HPCAGENT_BENCH_TOOLS_DIR/ppcg-<version>`` (a ``<tool>`` symlink pointed at the current build,
    see ``scripts/cache_env.sh``) is found WITHOUT the image's EDF ``PATH`` -- re-declared
    absolutely at run time, and not including that cache -- ever naming the directory.

    A candidate has to RUN, not merely exist, and that is what makes the ORDER safe. The cache
    outranks the image on purpose (a host pinning a build to test it must win), so an executable
    there that dies at startup would otherwise shadow the pinned ppcg the image carries and take
    the column down with it -- measured: the cache build left over from the /ritom scratch
    migration still has its executable bit and still fails with ``libLLVM-17.so.1: cannot open
    shared object file``. Probed with the tool's own ``--version``, under exactly the environment
    :func:`run_ppcg` gives it (see :func:`_ppcg_run_env`), so what this accepts is the call the
    column actually makes. Every refusal is kept and reported together: "installed but broken" and
    "not here at all" are different faults and only one of them is fixed by installing anything.
    """
    tools_dir = os.environ.get(TOOLS_DIR_ENV)
    candidates = (
        _exe_under_prefix(os.environ.get(PPCG_HOME_ENV), "ppcg"),
        _exe_under_prefix(f"{tools_dir}/ppcg" if tools_dir else None, "ppcg"),
        shutil.which("ppcg"),
    )
    refused: List[str] = []
    for exe in candidates:
        if exe is None:
            continue
        try:
            proc = subprocess.run([exe, "--version"], capture_output=True, text=True, env=_ppcg_run_env(exe))
        except OSError as exc:
            refused.append(f"{exe} cannot be executed: {exc}")
            continue
        if proc.returncode == 0:
            return exe, ""
        refused.append(f"{exe} is installed but does not run: {(proc.stderr or proc.stdout).strip()[-300:]}")
    if refused:
        return None, "; ".join(refused)
    return None, (
        f"ppcg is not installed on this host: nothing under ${PPCG_HOME_ENV}, ${TOOLS_DIR_ENV}/ppcg/bin, or PATH"
    )


def ppcg_exe() -> Optional[str]:
    """The ``ppcg`` binary this host runs, or ``None``. See :func:`ppcg_lookup` for the order."""
    return ppcg_lookup()[0]


def ppcg_problem() -> str:
    """``""`` when this host has a ``ppcg`` that runs, else what is wrong with every candidate."""
    return ppcg_lookup()[1]


def hipify_exe() -> Optional[str]:
    """``hipify-perl`` on PATH, or under ``$ROCM_PATH/bin`` when PATH omits it, or ``None`` when
    this host has neither. The image already ships ``hipify-perl`` under ``$ROCM_PATH/bin`` and
    puts that directory on PATH, so the fallback exists for a host or launch mode where PATH does
    not carry the image's own declaration -- never a hardcoded ROCm path, since ``ROCM_PATH`` is
    the same env var the rest of this image's tooling (``compilers.yaml``'s hipcc block) reads."""
    exe = shutil.which(HIPIFY)
    if exe is not None:
        return exe
    return _exe_under_prefix(os.environ.get("ROCM_PATH"), HIPIFY)


def missing_tool(backend: Optional[str] = None) -> str:
    """``""`` when this host can build the ppcg column for ``backend``, else why not.

    The ONE answer to "is this column's compiler here", asked per kernel by
    :func:`transformed_sources` and once per job by ``hpcagent_bench.harness.preflight`` -- so a
    preflight cannot pass on a toolchain the build would then decline, and the job log and the
    per-kernel row cannot give two different reasons for one absence.
    """
    problem = ppcg_problem()
    if problem:
        return problem
    if resolve_backend(backend) == "hip" and hipify_exe() is None:
        return f"ppcg emits CUDA and this column builds HIP, but {HIPIFY} is not installed"
    return ""


def resolve_backend(backend: Optional[str]) -> str:
    """The GPU vendor a ppcg column builds for: the one it NAMES, or the local toolchain's.

    The ``ppcg_cuda`` / ``ppcg_hip`` columns name theirs, so they measure the vendor they are
    labelled with on every host. Bare ``ppcg`` names none and keeps following whatever is installed
    (:func:`hpcagent_bench.languages.gpu_backend`), which is what it has always done.
    """
    if backend is None:
        return gpu_backend()
    if backend not in LANG_EXT:
        raise KeyError(f"unknown GPU backend {backend!r}; expected one of {sorted(LANG_EXT)}")
    return backend


def transformed_paths(scop: pathlib.Path, backend: Optional[str] = None) -> List[pathlib.Path]:
    """The two GPU sources ppcg's transform of ``scop`` compiles to, in compile order.

    ``.hip`` for a ROCm build rather than the ``.cu`` ppcg wrote: the extension is what makes hipcc
    build them for an AMD device instead of reading them as CUDA for an NVIDIA one.
    """
    stem, ext = scop.stem, LANG_EXT[resolve_backend(backend)]
    return [scop.with_name(f"{stem}_host.{ext}"), scop.with_name(f"{stem}_kernel.{ext}")]


def hipify(scratch: pathlib.Path, stem: str) -> None:
    """Rewrite ppcg's CUDA output in ``scratch`` as HIP, before anything is moved next to the scop.

    The shared ``<stem>_kernel.hu`` is translated IN PLACE and keeps its name: ppcg has already
    written the ``#include`` that names it into both halves and offers no way to rename it.
    """
    exe = hipify_exe()
    if exe is None:  # transformed_sources already declines for this; here it keeps the argv typed
        raise ToolMissing(FRAMEWORK, stem, f"{HIPIFY} is not installed on this host")
    for half in ("host", "kernel"):
        cu = scratch / f"{stem}_{half}.cu"
        translated = subprocess.run([exe, str(cu)], capture_output=True, text=True, check=True)
        (scratch / f"{stem}_{half}.{LANG_EXT['hip']}").write_text(translated.stdout)
    subprocess.run([exe, "-inplace", str(scratch / f"{stem}_kernel.hu")], capture_output=True, check=True)


#: Prepended to ppcg's host output. ppcg copies everything OUTSIDE the scop through verbatim, so
#: what reaches nvcc/hipcc is the translator's C11 prelude -- and both drivers compile a .cu/.hip as
#: C++, where ``restrict`` is not a keyword at all. Spelled as a define in the SOURCE rather than as
#: a ``-Drestrict=`` in compilers.yaml because that file is shared with every other GPU column, and
#: none of the others compiles C.
CXX_COMPAT_PROLOGUE: str = (
    "/* hpcagent_bench: ppcg emits C and the GPU drivers compile it as C++. */\n#define restrict __restrict__\n"
)

#: The prelude's ``__npb_conj`` calls C99 ``conj`` on a ``double _Complex``. In C++ that name is
#: ``std::conj``, a template over ``std::complex<T>`` which no ``_Complex`` argument matches, so the
#: helper fails to compile even for the many kernels that never call it. Rewritten at its ONE call
#: site rather than by defining ``conj`` away: a define at the top of the file would also rewrite the
#: declarations ``<complex.h>`` itself pulls in.
CONJ_CALL_RE = re.compile(r"(__npb_conj\(double _Complex z\)\s*\{\s*return\s+)conj(\s*\(z\);)")


def cxx_compat(host: str, entry: str) -> str:
    """ppcg's host output, rewritten into the C++ subset nvcc and hipcc actually accept.

    Three edits, all forced by the same fact -- ppcg's input is C and its output is compiled as C++:
    the ``restrict`` qualifier, the ``conj`` call in the prelude, and the entry point's LINKAGE. That
    last one is not cosmetic: the harness resolves this symbol by name through ctypes, so a C++
    -mangled definition builds and links fine and is then simply not found at call time.
    ``extern "C"`` is also what the translator's own C++ emitter wraps the entry in, so this makes
    ppcg's output agree with the sibling column rather than inventing a convention.
    """
    host = CXX_COMPAT_PROLOGUE + CONJ_CALL_RE.sub(r"\1__builtin_conj\2", host)
    return re.sub(rf"^(void\s+{re.escape(entry)}\s*\()", r'extern "C" \1', host, count=1, flags=re.MULTILINE)


def entry_symbol(scop: pathlib.Path) -> str:
    """The exported kernel symbol in ``scop``: its stem without the translator's ``_pluto_input`` tag."""
    return scop.stem.removesuffix("_pluto_input")


def drop_const_params(scop_src: str, entry: str) -> str:
    """The scop with ``const`` dropped from the entry point's parameters, for ppcg to read instead.

    ppcg gives each device pointer the qualifiers its host counterpart had, then passes that pointer
    to ``cudaMalloc``/``cudaMemcpy``/``cudaFree``, whose parameters are ``void *``. C allows the
    implicit cast away from ``const``; the C++ that nvcc and hipcc actually compile does not, so a
    read-only input array is enough to make the generated host half unbuildable ("no matching
    function for call to 'hipFree'").

    Applied to a COPY handed to ppcg, never to the file on disk: that file is the Pluto column's
    input too, and it is also what :func:`scop_inputs` freshness-checks against. ``const`` is a
    compile-time qualifier over a positional ctypes call, so dropping it changes no semantics --
    only which overload the GPU driver's headers will accept.
    """
    match = re.search(rf"^(void\s+{re.escape(entry)}\s*\()([^)]*)(\))", scop_src, flags=re.MULTILINE)
    if match is None:
        return scop_src
    return scop_src[: match.start(2)] + match.group(2).replace("const ", "") + scop_src[match.end(2) :]


def offloaded(kernel_src: str) -> bool:
    """Whether ppcg actually put a kernel on the GPU, i.e. its device half declares a ``__global__``.

    ppcg does NOT fail on a scop it cannot handle. It writes the input back out with the loop nest
    untouched, an empty ``_kernel`` file, exit status 0, and nothing on stderr -- which the old
    "returncode == 0 and both files exist" check accepted. That builds, runs the ORIGINAL serial loop
    on the host, and records the result as a polyhedral GPU number: the exact silent mislabelling the
    Pluto column was rebuilt to stop, in the column that copied its structure.
    """
    return "__global__" in kernel_src


def run_ppcg(
    scop: pathlib.Path, backend: Optional[str] = None, args: Sequence[str] = PPCG_ARGS, timeout: Optional[float] = None
) -> Tuple[List[str], subprocess.CompletedProcess]:
    """Transform one scop with ``ppcg``. Returns ``(argv, result)``.

    ppcg names its outputs after the INPUT and writes them into the current directory, with no
    ``-o`` for the pair, so it runs in a throwaway cwd and the results are moved next to ``scop``
    only on success -- a failed run leaves no half-written .cu for the build to pick up.

    That cwd is made BESIDE THE SCOP, not in ``$TMPDIR``. Publication is one ``os.replace`` per
    file, which is atomic and which therefore cannot cross a filesystem boundary: with the default
    temporary directory this raised ``OSError: [Errno 18] Invalid cross-device link`` for every
    kernel on a cluster node, where ``$TMPDIR`` is node-local and the checkout is on Lustre
    (measured, job 644285 -- eight affine kernels, eight ``runtime_error`` rows, the failure that
    had left this column with no measurement of any kind). ``shutil.move`` would paper over it by
    copying, and copying is not atomic: a concurrent build would be free to pick up a half-written
    ``.hip``. Same rule as the Pluto column, which puts its temporary OUTPUT beside the destination
    for exactly this reason (:func:`pluto_transform.run_polycc`).

    What it reads is a copy in that same cwd (:func:`drop_const_params`), not ``scop`` itself. The
    returned argv names ``scop``, which is the file a reader can re-run this on and the only one that
    outlives the call; the copy differs from it by qualifiers that change nothing about the transform.
    """
    exe = ppcg_exe()
    vendor = resolve_backend(backend)
    entry = entry_symbol(scop)
    argv = [str(exe), *args, str(scop)]
    with tempfile.TemporaryDirectory(dir=scop.parent, prefix=f".{FRAMEWORK}_transform_") as scratch:
        # ppcg names its outputs after the input's STEM, so the copy has to keep it.
        readable = pathlib.Path(scratch) / scop.name
        readable.write_text(drop_const_params(scop.read_text(), entry))
        proc = subprocess.run(
            [str(exe), *args, str(readable)],
            cwd=scratch,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_ppcg_run_env(str(exe)),
        )
        if proc.returncode == 0:
            host_cu = pathlib.Path(scratch) / f"{scop.stem}_host.cu"
            if host_cu.is_file():
                host_cu.write_text(cxx_compat(host_cu.read_text(), entry))
            if vendor == "hip":
                hipify(pathlib.Path(scratch), scop.stem)
            for produced in transformed_paths(scop, vendor):
                src = pathlib.Path(scratch) / produced.name
                if src.is_file():
                    os.replace(src, produced)
            header = pathlib.Path(scratch) / f"{scop.stem}_kernel.hu"
            if header.is_file():
                os.replace(header, scop.with_name(header.name))
    return argv, proc


def transformed_sources(cpp_backend: pathlib.Path, base: str, backend: Optional[str] = None) -> List[pathlib.Path]:
    """The ppcg-transformed CUDA the ``ppcg`` column compiles, generated on demand.

    Mirrors :func:`pluto_transform.transformed_sources`: regenerate when stale, reuse when fresh,
    and DECLINE rather than fall back to untransformed source -- a PPCG column built from the
    emitted C would be an nvcc column wearing PPCG's label.
    """
    vendor = resolve_backend(backend)
    # THE TOOL BEFORE THE KERNEL. A host with no ppcg has nothing to say about any kernel, and
    # asking the scop question first says it anyway: job 640520 declined 55 of its 248 rows as "the
    # translator emitted no #pragma scop" -- a fact about those kernels -- on a node where the real
    # and only answer was that the image shipped no ppcg at all, which the other 193 rows did say.
    problem = missing_tool(vendor)
    if problem:
        raise ToolMissing(FRAMEWORK, base, problem)
    scops = scop_inputs(cpp_backend, base)
    if not scops:
        raise NotSupportedByFramework(FRAMEWORK, base, "the translator emitted no #pragma scop for this kernel")
    out: List[pathlib.Path] = []
    for scop in scops:
        assert_affine(scop, base)
        produced = transformed_paths(scop, vendor)
        if any(not p.exists() or p.stat().st_mtime < scop.stat().st_mtime for p in produced):
            argv, proc = run_ppcg(scop, vendor)
            if proc.returncode != 0 or any(not p.is_file() for p in produced):
                raise NotSupportedByFramework(
                    FRAMEWORK, base, f"ppcg rejected {scop.name}: {proc.stderr.strip()[-500:]}"
                )
        # Re-read rather than trusting the run above: a passthrough that was published by an EARLIER
        # call is fresh against its scop, so a gate that only fired on a fresh transform would let
        # the second run of the same kernel time the serial host loop.
        if not offloaded(produced[1].read_text()):
            raise NotSupportedByFramework(
                FRAMEWORK,
                base,
                f"ppcg offloaded nothing from {scop.name}: it copied the scop through "
                "unchanged and emitted no __global__, so there is no GPU kernel to time",
            )
        out.extend(produced)
    return out
