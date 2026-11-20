# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every supported elementwise math ufunc computes numpy's answer on C, C++ and Fortran (tiny kernels emitted,
built and called via ctypes; a backend without its compiler skips)."""

import ctypes
import json
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from hpcagent_bench import languages
from hpcagent_bench.categories import Language
from hpcagent_bench.translators.numpyto_c.bindings import emit_binding
from hpcagent_bench.translators.numpyto_c.emit import emit_c, emit_cpp
from hpcagent_bench.translators.numpyto_common.frontend import parse_kernel
from hpcagent_bench.translators.numpyto_common.ir import KernelIR
from hpcagent_bench.translators.numpyto_common.lowering import lower
from hpcagent_bench.translators.numpyto_fortran.emit import emit_fortran

_CT = {"int": ctypes.c_int, "double": ctypes.c_double, "int64": ctypes.c_int64, "int32": ctypes.c_int32}

#: unary ufuncs validated on an array operand in (0.1, 0.9) -- a domain
#: valid for every one (arcsin/arctanh/log included).
UNARY = [
    "tan",
    "sinh",
    "cosh",
    "arcsin",
    "arccos",
    "arctan",
    "arcsinh",
    "arctanh",
    "exp2",
    "expm1",
    "log2",
    "log10",
    "log1p",
    "cbrt",
    "floor",
    "ceil",
    "trunc",
    "rint",
    "around",
    "square",
    "reciprocal",
    "sign",
    "degrees",
    "radians",
]
BINARY = ["arctan2", "hypot", "copysign", "fmod", "fmax", "fmin"]

# symbol == file stem == binding symbol (the canonical scheme: one name, no
# _auto / per-compiler suffix). emit_binding(base_name="k") records "k" for every
# language, so each backend emits its source with fn_name "k".
_BACKENDS = {
    "c": (emit_c, "c", "k", ["gcc", "-O2", languages.std_flag(Language.C), "-shared", "-fPIC"], "gcc"),
    "cpp": (emit_cpp, "cpp", "k", ["g++", "-O2", languages.std_flag(Language.CPP), "-shared", "-fPIC"], "g++"),
    "fortran": (
        emit_fortran,
        "fortran",
        "k",
        [
            "gfortran",
            "-O2",
            "-ffree-form",
            "-ffree-line-length-none",
            languages.std_flag(Language.FORTRAN),
            "-shared",
            "-fPIC",
        ],
        "gfortran",
    ),
}


#: Every ufunc, with its arity, as one output each of ONE kernel: a backend compiles and calls it once.
FUNCS: list[tuple[str, int]] = [(fn, 1) for fn in UNARY] + [(fn, 2) for fn in BINARY]
OUTS = [f"out_{fn}" for fn, _ in FUNCS]


def _kernel_ir(d: pathlib.Path) -> KernelIR:
    arr = ["a", "b", *OUTS]
    body = "".join(
        f"    {out}[:] = np.{fn}(a)\n" if nargs == 1 else f"    {out}[:] = np.{fn}(a, b)\n"
        for out, (fn, nargs) in zip(OUTS, FUNCS, strict=True)
    )
    (d / "k_numpy.py").write_text(f"import numpy as np\ndef k({', '.join(arr)}):\n{body}")
    (d / "k.json").write_text(
        json.dumps(
            {
                "benchmark": {
                    "name": "k",
                    "short_name": "k",
                    "relative_path": ".",
                    "module_name": "k",
                    "func_name": "k",
                    "domain": "d",
                    "dwarf": "d",
                    "parameters": {"S": {"N": 32}},
                    "init": {
                        "func_name": "",
                        "input_args": [],
                        "output_args": [],
                        "arrays": dict.fromkeys(arr, "(N,)"),
                    },
                    "input_args": arr,
                    "array_args": arr,
                    "output_args": OUTS,
                }
            }
        )
    )
    return lower(parse_kernel(d / "k_numpy.py", d / "k.json"))


@pytest.mark.parametrize("backend", list(_BACKENDS))
def test_every_ufunc_matches_numpy(backend: str) -> None:
    emit, sym_key, fname, compile_cmd, exe = _BACKENDS[backend]
    if shutil.which(exe) is None:
        pytest.skip(f"{exe} not available")
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        kir = _kernel_ir(d)
        ext = {"c": "c", "cpp": "cpp", "fortran": "f90"}[backend]
        src = d / f"k.{ext}"
        src.write_text(emit(kir, fn_name=fname))
        emit_binding(kir, d / "kb.json", base_name="k")
        binding = json.loads((d / "kb.json").read_text())
        so = d / "k.so"
        r = subprocess.run([*compile_cmd, str(src), "-o", str(so)], capture_output=True, text=True, check=False)
        assert r.returncode == 0, f"{backend} compile failed:\n{r.stderr}"
        rng = np.random.default_rng(0)
        a = rng.uniform(0.1, 0.9, 32)
        b = rng.uniform(0.1, 0.9, 32)
        data: dict[str, object] = {"a": a.copy(), "b": b.copy(), "N": 32, **{out: np.zeros(32) for out in OUTS}}
        lib = ctypes.CDLL(str(so))
        cfn = lib[binding["symbols"][sym_key]]
        cargs, keep = [], []
        for arg in binding["args"]:
            nm, kind = arg["name"], arg["kind"]
            if kind in _CT:
                # Scalars are passed BY VALUE: the emitted Fortran is C-bound
                # (``value`` attribute), same as C/C++ -- matching native_runtime.
                v = int(data[nm]) if kind.startswith("int") else float(data[nm])
                cargs.append(_CT[kind](v))
            else:
                buf = np.ascontiguousarray(data[nm])
                keep.append(buf)
                cargs.append(buf.ctypes.data_as(ctypes.c_void_p))
        cfn(*cargs)
        wrong = []
        for out, (fn, nargs) in zip(OUTS, FUNCS, strict=True):
            expected = getattr(np, fn)(a) if nargs == 1 else getattr(np, fn)(a, b)
            if not np.allclose(data[out], expected, rtol=1e-9, atol=1e-9):
                wrong.append(f"{fn}: max diff {np.abs(data[out] - expected).max():.2e}")
        assert not wrong, f"{backend}: {wrong}"
