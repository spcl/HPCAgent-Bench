# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""OpenMP contexts: one OpenMP runtime per process, chosen by the toolchain family that built the code.

Two runtimes in one process cannot see each other's parallel region (OpenBLAS inside a numba prange
thread opens a team per caller), and no single runtime serves every toolchain: clang cannot target
libgomp, gcc cannot target libomp's native ABI, NVHPC has its own. So an image carries one CONTEXT
per family under :func:`context_root` (default ``/opt/omp``), each a directory ``<root>/<context>/``:

``gnu``    (gcc, g++, gfortran; also the image default)  libgomp of the image's gcc, and the
           OpenMP-linking libraries built by it.
``llvm``   (clang, clang++, flang, hipcc, amdclang, Polly, OpenMP offload, numba)  LLVM's
           libomp, and the same libraries rebuilt with clang. ``libgomp.so.1`` is a link to libomp INSIDE this
           directory only: numba's OpenMP pool and any other GOMP-ABI code resolve to libomp there, and
           nowhere else.
``nvhpc``  (nvc, nvc++, nvfortran ``-mp``; CUDA image only)  libnvomp and NVHPC's own BLAS/LAPACK.

``lib/`` of a context holds everything a process of that family must load first; ``view/`` is the
prefix its OpenMP-linking catalog libraries are built into (pkg-config, headers, libraries).
:func:`context_env` is the environment of a grading child of that family: the loader finds the
context's ``lib/`` ahead of every other directory, so every soname the family's code needs
(``libopenblas.so.0`` for numpy and scipy, ``libgomp.so.1`` for numba) resolves inside the context.
The child is a fresh interpreter (the loader reads ``LD_LIBRARY_PATH`` at exec), which is why
:func:`hpcagent_bench.frameworks.forked.run_forked` spawns instead of forking when it is handed one.

Nothing here reads the image: a host without ``<root>/<context>/`` (a login node, CI) has no contexts,
:func:`context_env` is empty there and children run as before.
"""

import ast
import functools
import json
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from typing import Protocol

from hpcagent_bench import config, openmp_runtimes

__all__ = [
    "CONTEXTS",
    "DEFAULT_CONTEXT",
    "FAMILY_CONTEXT",
    "GNU",
    "LLVM",
    "NVHPC",
    "context_build_env",
    "context_dir",
    "context_env",
    "context_for_family",
    "context_for_library",
    "context_for_toolchain",
    "context_for_python_source",
    "context_root",
    "context_runtime",
    "context_view",
    "library_refusal",
    "numba_omp_pool_launched",
    "spawn_needed",
]

GNU = "gnu"
LLVM = "llvm"
NVHPC = "nvhpc"

#: Every context, the default first.
CONTEXTS: tuple[str, ...] = (GNU, LLVM, NVHPC)

#: What a process is when nothing names a context: the image's own environment IS the gnu context.
DEFAULT_CONTEXT = GNU

#: Toolchain family (:data:`hpcagent_bench.languages.COMPILER_FAMILIES`) -> context.
FAMILY_CONTEXT: Mapping[str, str] = {"gcc": GNU, "llvm": LLVM, "nvhpc": NVHPC}

#: numba's threading layer per context: ``omp`` binds ``libgomp.so.1`` (libomp in llvm), and NVHPC has no
#: GOMP interface to bind, so numba there runs its own ``workqueue`` pool.
NUMBA_LAYER: Mapping[str, str] = {GNU: "omp", LLVM: "omp", NVHPC: "workqueue"}

#: ``config.yaml`` key naming the directory that holds the context directories.
ROOT_KEY = "runtime.omp_context_root"

#: Environment variable a child carries so its own gate and log lines name the context it runs in.
CONTEXT_ENV = "HPCAGENT_BENCH_OMP_CONTEXT"

#: Python modules whose presence puts a python delivery in the llvm context: numba compiles through
#: LLVM and its OpenMP pool is a GOMP-ABI client, served by libomp there.
LLVM_MODULES: frozenset[str] = frozenset({"numba"})


def context_for_family(family: str) -> str:
    """The context of toolchain ``family``; :class:`KeyError` for a family outside
    :data:`FAMILY_CONTEXT`, so a new family cannot silently run in the gnu context."""
    try:
        return FAMILY_CONTEXT[family]
    except KeyError:
        raise KeyError(
            f"toolchain family {family!r} has no OpenMP context; expected one of {sorted(FAMILY_CONTEXT)}"
        ) from None


class HasFamily(Protocol):
    """What :func:`context_for_toolchain` reads off :class:`hpcagent_bench.languages.Toolchain`."""

    @property
    def family(self) -> str: ...


def context_for_toolchain(toolchain: HasFamily) -> str:
    """The context of the toolchain a submission builds with (:func:`hpcagent_bench.languages.submission_toolchain`,
    which already resolves the setup's pin, a request, and an offload leg's own driver): its family, or the
    gnu context for a driver outside every family (``nvcc`` compiles host code with gcc)."""
    return context_for_family(toolchain.family) if toolchain.family else DEFAULT_CONTEXT


def context_for_python_source(source: str) -> str:
    """The context of a python delivery: llvm when it imports numba, else gnu (numpy on OpenBLAS, torch,
    dace, cupy and triton run on the image default)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return DEFAULT_CONTEXT
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    return LLVM if imported & LLVM_MODULES else DEFAULT_CONTEXT


#: The runtime a shared library records in ``DT_NEEDED``, by soname stem, and the context that runtime is.
NEEDED_RUNTIME_CONTEXT: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"libgomp[-\w]*\.so"), GNU),
    (re.compile(r"lib(?:omp|iomp5)[-\w]*\.so"), LLVM),
    (re.compile(r"libnvomp[-\w]*\.so"), NVHPC),
)

#: Seconds ``readelf`` gets to read one library's dynamic section.
READELF_TIMEOUT_S = 30


def context_for_library(path: pathlib.Path) -> str:
    """The context of a PREBUILT library: that of the OpenMP runtime its ``DT_NEEDED`` names (libgomp gnu,
    libomp or libiomp5 llvm, libnvomp nvhpc), the default when it names none or cannot be read."""
    try:
        dynamic = subprocess.run(
            ["readelf", "-d", str(path)], capture_output=True, text=True, check=False, timeout=READELF_TIMEOUT_S
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return DEFAULT_CONTEXT
    needed = [line.rpartition("[")[2].rstrip("]") for line in dynamic.splitlines() if "(NEEDED)" in line]
    for soname in needed:
        for pattern, context in NEEDED_RUNTIME_CONTEXT:
            if pattern.match(soname):
                return context
    return DEFAULT_CONTEXT


def context_root() -> pathlib.Path:
    """The directory holding the context directories (``runtime.omp_context_root``)."""
    return pathlib.Path(config.get_str(ROOT_KEY, "/opt/omp"))


def context_dir(context: str) -> pathlib.Path | None:
    """``<root>/<context>`` when this host has it, else ``None`` (no such context here). ``""`` is the
    default context, the image's own environment."""
    context = context or DEFAULT_CONTEXT
    if context not in CONTEXTS:
        raise KeyError(f"unknown OpenMP context {context!r}; expected one of {CONTEXTS}")
    path = context_root() / context
    return path if path.is_dir() else None


def context_view(context: str) -> pathlib.Path | None:
    """The prefix ``context``'s OpenMP-linking catalog libraries are installed under, or ``None``."""
    base = context_dir(context)
    return base / "view" if base is not None and (base / "view").is_dir() else None


def spawn_needed(context: str) -> bool:
    """Whether a child of ``context`` needs a fresh interpreter: the loader fixed the parent's runtime
    at exec, so only a context other than the parent's own (the default) and present here changes
    anything."""
    return bool(context) and context != DEFAULT_CONTEXT and context_dir(context) is not None


def numba_omp_pool_launched() -> bool:
    """Whether this process has launched numba's ``omp`` threading layer.

    A child forked from such a process terminates with SIGTERM ("fork() called from a process already
    using GNU OpenMP, this is unsafe") the moment it enters a numba parallel region, so a caller about to
    fork a numba child checks this first. numba is only looked up in ``sys.modules``: a process that never
    imported it has launched nothing."""
    numba = sys.modules.get("numba")
    if numba is None:
        return False
    try:
        return bool(numba.threading_layer() == "omp")
    except ValueError:  # numba raises until a parallel region has launched a layer
        return False


def prepend_path(entries: Iterable[str], current: str) -> str:
    """``entries`` (absent directories dropped) ahead of the ``:``-separated ``current``, deduplicated."""
    seen: list[str] = []
    for item in [*entries, *current.split(os.pathsep)]:
        if item and item not in seen:
            seen.append(item)
    return os.pathsep.join(seen)


def context_env(context: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment ENTRIES that put a child in ``context``: ``LD_LIBRARY_PATH`` with the context's
    ``lib/`` first, numba's threading layer, and the context name. Empty when the host has no such
    context, so the caller's child is exactly what it was."""
    root = context_dir(context)
    if root is None:
        return {}
    env = base if base is not None else os.environ
    lib = root / "lib"
    context = context or DEFAULT_CONTEXT
    return {
        "LD_LIBRARY_PATH": prepend_path([str(lib)] if lib.is_dir() else [], env.get("LD_LIBRARY_PATH", "")),
        "NUMBA_THREADING_LAYER": NUMBA_LAYER[context],
        CONTEXT_ENV: context,
    }


def context_build_env(context: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The search paths a BUILD in ``context`` resolves libraries by: the context's ``view/`` ahead of the
    image's own, so pkg-config, the compiler and CMake find the family's OpenMP-linking variants first
    (an ``-L`` into the llvm view is what makes the llvm ``libopenblas.so.0`` the one NEEDED). Empty
    for a context without a view."""
    view = context_view(context)
    if view is None:
        return {}
    env = base if base is not None else os.environ
    root = str(view)
    return {
        "PKG_CONFIG_PATH": prepend_path(
            [f"{root}/lib/pkgconfig", f"{root}/lib64/pkgconfig", f"{root}/share/pkgconfig"],
            env.get("PKG_CONFIG_PATH", ""),
        ),
        "LIBRARY_PATH": prepend_path([f"{root}/lib", f"{root}/lib64"], env.get("LIBRARY_PATH", "")),
        "CPATH": prepend_path([f"{root}/include"], env.get("CPATH", "")),
        "CMAKE_PREFIX_PATH": prepend_path([root], env.get("CMAKE_PREFIX_PATH", "")),
    }


#: The per-image record of which OpenMP runtimes each catalog library's build maps in each context, written
#: by :mod:`hpcagent_bench.omp_catalog` at image build: ``{context: {library: [runtime realpath, ...] | null}}``
#: (``null``: the context cannot link it at all).
CATALOG_FILE = "catalog.json"


def context_runtime(context: str) -> str | None:
    """The realpath of the runtime ``context``'s children map (``lib/libgomp.so.1`` there: libgomp in gnu,
    libomp in llvm), ``None`` for nvhpc (any one libnvomp) and for a context this host lacks."""
    root = context_dir(context)
    if root is None or (context or DEFAULT_CONTEXT) == NVHPC:
        return None
    link = root / "lib" / "libgomp.so.1"
    return os.path.realpath(link) if link.exists() else None


@functools.lru_cache(maxsize=8, typed=True)
def read_catalog(path: str, mtime_ns: int) -> dict[str, dict[str, list[str] | None]]:
    """The parsed catalog record at ``path`` (re-read when the file changes)."""
    del mtime_ns  # the cache key only
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def catalog_record() -> dict[str, dict[str, list[str] | None]] | None:
    """The image's catalog record, or ``None`` on a host that has none (a login node, CI)."""
    path = context_root() / CATALOG_FILE
    try:
        return read_catalog(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError):
        return None


def library_refusal(name: str, context: str) -> str:
    """Why catalog library ``name`` may not be linked by a ``context`` build, or ``""``, by the image's own
    record (:func:`refusal_from`); a host with no record refuses nothing."""
    return refusal_from(catalog_record(), name, context)


def refusal_from(record: dict[str, dict[str, list[str] | None]] | None, name: str, context: str) -> str:
    """Why ``name`` may not be linked by a ``context`` build according to ``record``, or ``""``.

    The record says which OpenMP runtimes the library's build maps when it is linked in that context.
    Anything but that context's own runtime would put a second one in every process that loads it, so it
    is refused up front, as a request fault, and never fails inside the grading child. No record, a
    context the record lacks, and a library it does not list refuse nothing."""
    context = context or DEFAULT_CONTEXT
    if record is None or context not in record or name not in record[context]:
        return ""
    runtimes = record[context][name]
    if runtimes is None:
        return f"{name} has no build the {context} OpenMP context can link"
    expected = context_runtime(context)
    if context == NVHPC:  # one libnvomp and nothing else
        nvhpc = [r for r in runtimes if openmp_runtimes.NVHPC_RUNTIME.fullmatch(os.path.basename(r))]
        foreign = [r for r in runtimes if r not in nvhpc] + nvhpc[1:]
    else:
        foreign = [r for r in runtimes if r != expected]
    if not foreign:
        return ""
    names = ", ".join(sorted(os.path.basename(r) for r in runtimes))
    return (
        f"{name} maps {names} when linked in the {context} OpenMP context (the toolchain family this "
        f"submission builds with), which runs on {os.path.basename(expected) if expected else 'libnvomp'} alone; "
        "pick another family or another library"
    )
