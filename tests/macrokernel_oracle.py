# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared helper for the macrokernel oracle tests: compiles the dace-fortran-emitted C++ fixture and
compares it end to end, on identical inputs, against the numpy port. dace's include dir is discovered
from the installed python package, never hard-coded, so it works in CI and locally alike."""
import ctypes
import functools
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional

from optarena import languages


@functools.lru_cache(maxsize=1)
def dace_include_dir() -> Optional[str]:
    """``<dace package>/runtime/include``, or ``None`` when dace is not importable."""
    try:
        import dace
    except ImportError:
        return None
    inc = os.path.join(os.path.dirname(dace.__file__), "runtime", "include")
    return inc if os.path.isfile(os.path.join(inc, "dace", "dace.h")) else None


def have_oracle_toolchain() -> bool:
    """True when the emitted C++ can be built: a C++ compiler + dace headers."""
    return bool((shutil.which("c++") or shutil.which("g++")) and dace_include_dir())


def compile_emitted_so(cpp_path: str, out_so: str, *, extra_flags: List[str] = ()) -> str:
    """Compile a dace-emitted ``.cpp`` into a ctypes-loadable ``.so``, built serially (no -fopenmp):
    ``dace::wcr_fixed::reduce_atomic`` is only race-free without an OpenMP parallel for."""
    inc = dace_include_dir()
    if inc is None:
        raise RuntimeError("dace headers not found; install dace (pip install dace)")
    cc = shutil.which("c++") or shutil.which("g++")
    cmd = [cc, languages.std_flag("cpp"), "-O2", "-fPIC", "-shared", f"-I{inc}", cpp_path, "-o", out_so, *extra_flags]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("emitted-C++ build failed:\n" + " ".join(cmd) + "\n" + proc.stderr[:4000])
    return out_so


# --- driving the emitted kernel via ctypes ----------------------------------
# A DaCe-emitted kernel exposes three C entry points: __dace_init_<name>, __program_<name>, and
# __dace_exit_<name>. The flat-SoA arg list is huge but mechanical, so it's parsed from the .cpp.

_VALUE_CTYPE = {
    "int": ctypes.c_int,
    "int64_t": ctypes.c_int64,
    "double": ctypes.c_double,
    "bool": ctypes.c_bool,
    "float": ctypes.c_float
}


def _parse_args(cpp_text: str, fn: str):
    """``[(name, ctype_token, is_pointer), ...]`` for entry ``fn`` in the .cpp."""
    m = re.search(re.escape(fn) + r"\s*\(([^;{]*?)\)\s*\{", cpp_text, re.S)
    if not m:
        raise RuntimeError(f"signature for {fn} not found in emitted C++")
    out = []
    for raw in m.group(1).split(","):
        a = raw.replace("__restrict__", "").strip()
        if not a:
            continue
        is_ptr = "*" in a
        tok = a.split()[0]
        name = re.sub(r"[*\s].*$", "", a.split()[-1]) if not is_ptr else a.replace("*", " ").split()[-1]
        out.append((name, tok, is_ptr))
    return out


def call_emitted(cpp_path: str, so_path: str, kernel: str, *, buffers: Dict, scalars: Dict) -> None:
    """Run a DaCe-emitted kernel on caller-provided flat-SoA inputs, in place. Array shape symbols are
    taken from ``buffers[arr].shape[k]``, so the caller only supplies genuine inputs, not the derived
    dimension args."""
    text = open(cpp_path).read()
    lib = ctypes.CDLL(so_path)

    def resolve(name, tok, is_ptr):
        if is_ptr:
            return ctypes.c_void_p(buffers[name].ctypes.data)
        if name.startswith("offset_"):
            # Fortran arrays are 1-based, so the per-dim offset is the lower bound 1.
            return ctypes.c_int64(1)
        if name in scalars:
            return _VALUE_CTYPE[tok](scalars[name])
        md = re.match(r"(.+)_d(\d+)$", name)
        if md:
            arr, dim = md.group(1), int(md.group(2))
            # The dim symbol may use the flattened struct-member spelling vs. the pointer arg's.
            buf = buffers.get(arr)
            if buf is None:
                buf = buffers.get(arr.replace("__", "_"))
            return _VALUE_CTYPE[tok](int(buf.shape[dim]) if buf is not None and dim < buf.ndim else 1)
        raise KeyError(f"no value for emitted-kernel arg {name!r} ({tok})")

    def invoke(fn, state=None, ret=None):
        args = _parse_args(text, fn)
        if state is not None:
            args = args[1:]  # the first arg is __state (the handle)
        f = lib[fn]
        f.restype = ret
        vals = [resolve(*a) for a in args]
        if state is not None:
            f.argtypes = [ctypes.c_void_p] + [type(v) for v in vals]
            return f(state, *vals)
        f.argtypes = [type(v) for v in vals]
        return f(*vals)

    handle = invoke(f"__dace_init_{kernel}", ret=ctypes.c_void_p)
    invoke(f"__program_{kernel}", state=handle, ret=None)
    ex = lib[f"__dace_exit_{kernel}"]
    ex.argtypes = [ctypes.c_void_p]
    ex.restype = ctypes.c_int
    ex(handle)
