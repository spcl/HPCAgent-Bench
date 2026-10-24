# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Running ``ppcg`` (the GPU sibling of Pluto): scop selection and the affine guard are imported from
:mod:`hpcagent_bench.polyhedral.pluto`. ppcg has no AMD target, so a HIP build hipifies its CUDA output; ``ppcg_cuda``
/ ``ppcg_hip`` name their vendor (declining when it is absent), bare ``ppcg`` follows the local toolchain."""

import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence

from hpcagent_bench.categories import Language
from hpcagent_bench.frameworks.errors import NotSupportedByFramework, ToolMissing
from hpcagent_bench.languages import GPU_HOST_LANG, LANG_EXT, gpu_backend
from hpcagent_bench.polyhedral.pluto_normalize import normalize_ppcg_input
from hpcagent_bench.polyhedral.pluto import OVERRIDE_INPUT_SUFFIX, assert_affine, scop_inputs
from hpcagent_bench.translators.numpyto_c.emit import NPB_HD_GUARD

__all__ = [
    "CONJ_CALL_RE",
    "CXX_COMPAT_PROLOGUE",
    "DEVICE_SYNC",
    "DEV_DECL_RE",
    "FRAMEWORK",
    "HIPIFY",
    "HIP_MIRROR_CALL",
    "PPCG_ARGS",
    "PPCG_HOME_ENV",
    "PRELUDE_HELPER_RE",
    "TOOLS_DIR_ENV",
    "VLA_CAST_RE",
    "cxx_compat",
    "device_helpers",
    "device_resident_host",
    "drop_const_params",
    "entry_body_end",
    "entry_params",
    "entry_symbol",
    "exe_under_prefix",
    "hipify",
    "hipify_exe",
    "missing_tool",
    "offloaded",
    "ppcg_exe",
    "ppcg_input",
    "ppcg_lookup",
    "ppcg_problem",
    "resolve_backend",
    "run_ppcg",
    "transformed_paths",
    "transformed_sources",
    "with_device_helpers",
]

#: The framework name this module transforms for -- used in every decline message.
FRAMEWORK = "ppcg"

#: How ``ppcg`` is invoked; the tile knobs are ppcg's defaults spelled out so a host cannot change what is measured.
PPCG_ARGS: tuple[str, ...] = ("--target=cuda", "--tile", "--tile-size=32")

#: ROCm's CUDA-to-HIP source translator, run on ppcg's output on an AMD host.
HIPIFY = "hipify-perl"

#: Direct override: a ``ppcg`` prefix (the directory holding ``bin/ppcg``).
PPCG_HOME_ENV = "HPCAGENT_BENCH_PPCG_HOME"

#: Where a build script installs tools the image does not ship (``helpers/scripts/cache_env.sh``); never a literal path.
TOOLS_DIR_ENV = "HPCAGENT_BENCH_TOOLS_DIR"


def exe_under_prefix(prefix: str | None, name: str) -> str | None:
    """``<prefix>/bin/<name>`` if that file exists and is executable, else ``None``."""
    if not prefix:
        return None
    candidate = pathlib.Path(prefix) / "bin" / name
    return str(candidate) if os.access(candidate, os.X_OK) else None


def ppcg_run_env(exe: str) -> dict[str, str] | None:
    """``ppcg``'s run environment: its own ``lib`` dir prepended to ``LD_LIBRARY_PATH`` (``None`` when absent).

    RPATH ranks below ``LD_LIBRARY_PATH``, which the image points at an older system ``libisl.so.23`` lacking
    ``isl_id_set_alloc``; this holds wherever ppcg is installed, so it applies unconditionally."""
    lib = pathlib.Path(exe).resolve().parent.parent / "lib"
    if not lib.is_dir():
        return None
    env = dict(os.environ)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib}:{existing}" if existing else str(lib)
    return env


def ppcg_lookup() -> tuple[str | None, str]:
    """The ``ppcg`` this host will run, as ``(exe, "")``, or ``(None, why not)``.

    Candidates in order: :data:`PPCG_HOME_ENV`, ``<tools dir>/ppcg/bin/ppcg``, PATH. Each must RUN ``--version``
    under :func:`ppcg_run_env` (a broken cache build must not shadow the image's); every refusal is reported."""
    tools_dir = os.environ.get(TOOLS_DIR_ENV)
    # THUNKS: a later source is never consulted once an earlier one answers (an override must stop the PATH search).
    sources: tuple[Callable[[], str | None], ...] = (
        lambda: exe_under_prefix(os.environ.get(PPCG_HOME_ENV), "ppcg"),
        lambda: exe_under_prefix(f"{tools_dir}/ppcg" if tools_dir else None, "ppcg"),
        lambda: shutil.which("ppcg"),
    )
    refused: list[str] = []
    for source in sources:
        exe = source()
        if exe is None:
            continue
        try:
            proc = subprocess.run(
                [exe, "--version"], capture_output=True, text=True, env=ppcg_run_env(exe), check=False
            )
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


def ppcg_exe() -> str | None:
    """The ``ppcg`` binary this host runs, or ``None``. See :func:`ppcg_lookup` for the order."""
    return ppcg_lookup()[0]


def ppcg_problem() -> str:
    """``""`` when this host has a ``ppcg`` that runs, else what is wrong with every candidate."""
    return ppcg_lookup()[1]


def hipify_exe() -> str | None:
    """``hipify-perl`` on PATH, else under ``$ROCM_PATH/bin``, else ``None``."""
    exe = shutil.which(HIPIFY)
    if exe is not None:
        return exe
    return exe_under_prefix(os.environ.get("ROCM_PATH"), HIPIFY)


def missing_tool(backend: Language | None = None) -> str:
    """``""`` when this host can build the ppcg column for ``backend``, else why not (shared by the build and
    preflight, so they cannot disagree)."""
    problem = ppcg_problem()
    if problem:
        return problem
    if resolve_backend(backend) is Language.HIP and hipify_exe() is None:
        return f"ppcg emits CUDA and this column builds HIP, but {HIPIFY} is not installed"
    return ""


def resolve_backend(backend: Language | None) -> Language:
    """The GPU vendor a ppcg column builds for: the one it NAMES, or the local toolchain's (bare ``ppcg``)."""
    if backend is None:
        return gpu_backend()
    if backend not in GPU_HOST_LANG:
        raise KeyError(f"unknown GPU backend {backend.text!r}; expected one of {[lang.text for lang in GPU_HOST_LANG]}")
    return backend


def transformed_paths(scop: pathlib.Path, backend: Language | None = None) -> list[pathlib.Path]:
    """The two GPU sources ppcg's transform of ``scop`` compiles to, in compile order (``.hip`` on ROCm, so hipcc
    builds them for AMD)."""
    stem, ext = scop.stem, LANG_EXT[resolve_backend(backend)]
    return [scop.with_name(f"{stem}_host.{ext}"), scop.with_name(f"{stem}_kernel.{ext}")]


def hipify(scratch: pathlib.Path, stem: str) -> None:
    """Rewrite ppcg's CUDA output in ``scratch`` as HIP; the shared ``<stem>_kernel.hu`` keeps its name (both halves
    include it)."""
    exe = hipify_exe()
    if exe is None:  # declined in transformed_sources; keeps the argv typed
        raise ToolMissing(FRAMEWORK, stem, f"{HIPIFY} is not installed on this host")
    for half in ("host", "kernel"):
        cu = scratch / f"{stem}_{half}.cu"
        translated = subprocess.run([exe, str(cu)], capture_output=True, text=True, check=True)
        (scratch / f"{stem}_{half}.{LANG_EXT[Language.HIP]}").write_text(translated.stdout)
    subprocess.run([exe, "-inplace", str(scratch / f"{stem}_kernel.hu")], capture_output=True, check=True)


#: One ``hipMalloc``/``hipMemcpy``/``hipFree`` statement naming the device mirror ``{mirror}``, possibly
#: inside ppcg's ``cudaCheckReturn`` macro; group 1 is its indent. ``[^;]*`` is safe: none of the
#: three calls' arguments (a size expression, ``hipMemcpyHostToDevice``) contains a semicolon.
HIP_MIRROR_CALL = r"^([ \t]*)(?:cudaCheckReturn\()?hip(?:Malloc|Memcpy|Free)\([^;]*\b{mirror}\b[^;]*\);[ \t]*$"

#: A mirror's declaration, ``T *dev_<arg>;`` -- group 1 the indent, group 2 the element type, group 3
#: the array name. ``--target=cuda`` always spells a mirror ``dev_<array>``.
DEV_DECL_RE = re.compile(r"^([ \t]*)([A-Za-z_][\w \t]*?)\s*\*\s*dev_(\w+)\s*;[ \t]*$", re.MULTILINE)


def entry_params(host: str, entry: str) -> list[str]:
    """The parameter NAMES of ``entry``'s definition in ``host``, in order; ``[]`` if absent.

    A name is the last identifier before any ``[`` (VLA brackets hold no comma); the list runs to its BALANCED ``)``."""
    for match in re.finditer(rf"\b{re.escape(entry)}\s*\(", host):
        depth, end = 1, match.end()
        while depth and end < len(host):
            depth += {"(": 1, ")": -1}.get(host[end], 0)
            end += 1
        if depth or not re.match(r"\s*\{", host[end:]):
            continue  # a call or a declaration, not the definition
        params = host[match.end() : end - 1].split(",")
        names = [re.findall(r"[A-Za-z_]\w*", param.split("[", 1)[0]) for param in params]
        return [found[-1] for found in names if found]
    return []


def entry_body_end(host: str, entry: str) -> int:
    """The offset of the ``}`` closing ``entry``'s definition in ``host``; ``-1`` if absent."""
    for match in re.finditer(rf"\b{re.escape(entry)}\s*\(", host):
        depth, end = 1, match.end()
        while depth and end < len(host):
            depth += {"(": 1, ")": -1}.get(host[end], 0)
            end += 1
        opening = re.match(r"\s*\{", host[end:])
        if depth or not opening:
            continue
        depth, end = 1, end + opening.end()
        while depth and end < len(host):
            depth += {"{": 1, "}": -1}.get(host[end], 0)
            end += 1
        return end - 1 if not depth else -1
    return -1


#: Ends a device-resident entry: its stripped copy back to the host was the only wait on its kernels.
DEVICE_SYNC = "  (void)hipDeviceSynchronize();\n"


def device_resident_host(host: str, entry: str) -> str:
    """ppcg's hipified host code using ``entry``'s array PARAMETERS as device pointers instead of device mirrors.

    The harness puts array arguments on the device before the timed call, so each parameter mirror's declaration
    becomes an alias ``T *dev_X = (T *) X;`` (well-typed for a VLA parameter) and its malloc/copy/free calls are
    dropped. Mirrors of LOCAL arrays are host buffers and stay. The entry then ends with a device synchronize: the
    dropped device-to-host copy was its only wait, and without it the caller read and freed the outputs while kernels
    still ran (fdtd_2d: wrong values, hipErrorIllegalAddress). Raises when no parameter mirror is found; call it only
    on an :func:`offloaded` transform."""
    params = set(entry_params(host, entry))
    aliased: list[str] = []

    def alias(match: "re.Match[str]") -> str:
        indent, ctype, name = match.group(1), match.group(2).strip(), match.group(3)
        if name not in params:
            return match.group(0)
        aliased.append(name)
        return f"{indent}{ctype} *dev_{name} = ({ctype} *) {name};"

    rewritten = DEV_DECL_RE.sub(alias, host)
    if not aliased:
        raise ValueError(f"no ppcg-style 'dev_<arg>' device mirror of a parameter of {entry!r} found to strip")
    for name in aliased:
        # An EMPTY STATEMENT, not a deletion: ppcg guards a copy with a brace-less ``if``.
        rewritten = re.sub(HIP_MIRROR_CALL.format(mirror=f"dev_{name}"), r"\1;", rewritten, flags=re.MULTILINE)
    end = entry_body_end(rewritten, entry)
    if end < 0:
        raise ValueError(f"no definition of {entry!r} found to end with a device synchronize")
    return rewritten[:end] + DEVICE_SYNC + rewritten[end:]


#: Prepended to ppcg's host output: nvcc/hipcc compile the C prelude as C++, where ``restrict`` is no keyword (not
#: a ``-D`` in compilers.yaml, which every other GPU column shares).
CXX_COMPAT_PROLOGUE: str = (
    "/* hpcagent_bench: ppcg emits C and the GPU drivers compile it as C++. */\n#define restrict __restrict__\n"
)

#: The prelude's ``__npb_conj`` calls C99 ``conj``, which no C++ overload matches for ``_Complex``; rewritten at its
#: one call site (defining ``conj`` away would also hit ``<complex.h>``).
CONJ_CALL_RE = re.compile(r"(__npb_conj\(double _Complex z\)\s*\{\s*return\s+)conj(\s*\(z\);)")

#: A transient's ``T (*X)[E] = (T (*)[E])malloc(...)``: C++ refuses the VLA cast, so it is respelled as ``X``'s own
#: type. Group 1 is the declaration through ``=``, group 2 the variable.
VLA_CAST_RE = re.compile(
    r"^(\s*[\w ]+\(\*\s*(\w+)\)(?:\[[^\]\n]*\])+\s*=\s*)\([\w ]+\(\*\)(?:\[[^\]\n]*\])+\)", re.MULTILINE
)


def cxx_compat(host: str, entry: str) -> str:
    """ppcg's host output rewritten into C++ nvcc/hipcc accept: ``restrict``, the prelude's ``conj``, a transient's
    VLA cast, and ``extern "C"`` on the entry (ctypes resolves it by its unmangled name)."""
    host = CXX_COMPAT_PROLOGUE + VLA_CAST_RE.sub(r"\1(__typeof__(\2))", CONJ_CALL_RE.sub(r"\1__builtin_conj\2", host))
    return re.sub(rf"^(void\s+{re.escape(entry)}\s*\()", r'extern "C" \1', host, count=1, flags=re.MULTILINE)


#: A prelude helper's definition head, ``static inline NPB_HD <ret> <name>(``; group 1 is the name.
PRELUDE_HELPER_RE = re.compile(r"^static inline NPB_HD [^(\n]*?\b(\w+)\(", re.MULTILINE)


def device_helpers(prelude: str, kernel_src: str) -> str:
    """The translator's prelude helpers ``kernel_src`` calls, copied verbatim (callees first, behind the ``NPB_HD``
    guard) for ppcg's device half, which otherwise lacks their definitions; ``""`` when none."""
    bodies: dict[str, str] = {}
    for match in PRELUDE_HELPER_RE.finditer(prelude):
        open_brace = prelude.index("{", match.end())
        depth, end = 0, open_brace
        for end in range(open_brace, len(prelude)):
            depth += {"{": 1, "}": -1}.get(prelude[end], 0)
            if depth == 0:
                break
        bodies[match.group(1)] = prelude[match.start() : end + 1]
    wanted: list[str] = []
    pending = [n for n in re.findall(r"\b(\w+)\s*\(", kernel_src) if n in bodies]
    while pending:
        name = pending.pop()
        if name in wanted:
            continue
        wanted.append(name)
        pending.extend(n for n in re.findall(r"\b(\w+)\s*\(", bodies[name]) if n in bodies and n != name)
    if not wanted:
        return ""
    # Callees first: a helper must be declared before a helper that calls it.
    return NPB_HD_GUARD + "".join(f"{bodies[n]}\n" for n in reversed(wanted))


def with_device_helpers(prelude: str, kernel_src: str) -> str:
    """ppcg's device half with the :func:`device_helpers` it calls, inserted after its ``_kernel.hu`` include."""
    helpers = device_helpers(prelude, kernel_src)
    if not helpers:
        return kernel_src
    head, rest = kernel_src.split("\n", 1)
    return f"{head}\n#include <stdint.h>\n#include <math.h>\n{helpers}{rest}"


def entry_symbol(scop: pathlib.Path) -> str:
    """The exported kernel symbol in ``scop``: its stem without the ``_pluto_input``/``_ppcg_input`` (or override)
    tag."""
    stem = scop.stem.removesuffix(pathlib.Path(OVERRIDE_INPUT_SUFFIX).stem)
    return stem.removesuffix("_pluto_input").removesuffix("_ppcg_input")


def ppcg_input(scop: pathlib.Path) -> pathlib.Path:
    """``scop``'s helper-inlined twin ``<base>_fpNN_ppcg_input.c`` when the translator wrote one, else ``scop``."""
    twin = scop.with_name(scop.name.replace("_pluto_input.c", "_ppcg_input.c"))
    return twin if twin != scop and twin.is_file() else scop


def drop_const_params(scop_src: str, entry: str) -> str:
    """The scop with ``const`` dropped from the entry's parameters (C++ will not pass a const device pointer to
    ``cudaFree``); applied to ppcg's COPY, never the shared file on disk."""
    match = re.search(rf"^(void\s+{re.escape(entry)}\s*\()([^)]*)(\))", scop_src, flags=re.MULTILINE)
    if match is None:
        return scop_src
    return scop_src[: match.start(2)] + match.group(2).replace("const ", "") + scop_src[match.end(2) :]


def offloaded(kernel_src: str) -> bool:
    """Whether ppcg put a kernel on the GPU (its device half declares a ``__global__``): on a scop it cannot handle,
    ppcg exits 0 and writes the serial loop back out."""
    return "__global__" in kernel_src


def run_ppcg(
    scop: pathlib.Path, backend: Language | None = None, args: Sequence[str] = PPCG_ARGS, timeout: float | None = None
) -> tuple[list[str], subprocess.CompletedProcess]:
    """Transform one scop with ``ppcg``; returns ``(argv, result)``.

    Runs in a throwaway cwd BESIDE the scop (``os.replace`` publication is atomic only within a filesystem) on a
    :func:`drop_const_params` copy; outputs are moved next to ``scop`` only on success."""
    exe = ppcg_exe()
    vendor = resolve_backend(backend)
    entry = entry_symbol(scop)
    argv = [str(exe), *args, str(scop)]
    with tempfile.TemporaryDirectory(dir=scop.parent, prefix=f".{FRAMEWORK}_transform_") as scratch:
        # ppcg names its outputs after the input's STEM, so the copy has to keep it.
        readable = pathlib.Path(scratch) / scop.name
        # pet's normalizations apply to ppcg as to polycc; the Pluto-only rewrites are left out (normalize_ppcg_input).
        readable.write_text(drop_const_params(normalize_ppcg_input(scop.read_text()), entry))
        proc = subprocess.run(
            [str(exe), *args, str(readable)],
            cwd=scratch,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=ppcg_run_env(str(exe)),
            check=False,
        )
        if proc.returncode == 0:
            host_cu = pathlib.Path(scratch) / f"{scop.stem}_host.cu"
            if host_cu.is_file():
                host_cu.write_text(cxx_compat(host_cu.read_text(), entry))
            kernel_cu = pathlib.Path(scratch) / f"{scop.stem}_kernel.cu"
            if kernel_cu.is_file():
                kernel_cu.write_text(with_device_helpers(readable.read_text(), kernel_cu.read_text()))
            if vendor is Language.HIP:
                hipify(pathlib.Path(scratch), scop.stem)
                host_hip = pathlib.Path(scratch) / f"{scop.stem}_host.hip"
                kernel_hip = pathlib.Path(scratch) / f"{scop.stem}_kernel.hip"
                # A passthrough has no mirror: published as is, declined by the offload gate.
                if host_hip.is_file() and kernel_hip.is_file() and offloaded(kernel_hip.read_text()):
                    host_hip.write_text(device_resident_host(host_hip.read_text(), entry))
            for produced in transformed_paths(scop, vendor):
                src = pathlib.Path(scratch) / produced.name
                if src.is_file():
                    pathlib.Path(src).replace(produced)
            header = pathlib.Path(scratch) / f"{scop.stem}_kernel.hu"
            if header.is_file():
                header.replace(scop.with_name(header.name))
    return argv, proc


def transformed_sources(cpp_backend: pathlib.Path, base: str, backend: Language | None = None) -> list[pathlib.Path]:
    """The ppcg-transformed GPU sources the ``ppcg`` column compiles, regenerated when stale; DECLINES rather than
    falling back to untransformed source."""
    vendor = resolve_backend(backend)
    # The tool before the kernel: without ppcg, no row should decline as "no #pragma scop".
    problem = missing_tool(vendor)
    if problem:
        raise ToolMissing(FRAMEWORK, base, problem)
    scops = [ppcg_input(scop) for scop in scop_inputs(cpp_backend, base)]
    if not scops:
        raise NotSupportedByFramework(FRAMEWORK, base, "the translator emitted no #pragma scop for this kernel")
    out: list[pathlib.Path] = []
    for scop in scops:
        assert_affine(scop, base)
        produced = transformed_paths(scop, vendor)
        if any(not p.exists() or p.stat().st_mtime < scop.stat().st_mtime for p in produced):
            argv, proc = run_ppcg(scop, vendor)
            if proc.returncode != 0 or any(not p.is_file() for p in produced):
                raise NotSupportedByFramework(
                    FRAMEWORK, base, f"ppcg rejected {scop.name}: {proc.stderr.strip()[-500:]}"
                )
        # Re-read: a fresh passthrough published by an EARLIER call must still be declined.
        if not offloaded(produced[1].read_text()):
            raise NotSupportedByFramework(
                FRAMEWORK,
                base,
                f"ppcg offloaded nothing from {scop.name}: it copied the scop through "
                "unchanged and emitted no __global__, so there is no GPU kernel to time",
            )
        out.extend(produced)
    return out
