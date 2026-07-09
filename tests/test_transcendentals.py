"""Numerical validation of the elementwise transcendental / math ufuncs.

The NumpyToX emitters lower ``out = np.<fn>(a)`` (and the 2-arg forms)
to a per-element loop. This proves every supported ufunc computes the
same answer as numpy across all three compiled backends (C / C++ /
Fortran) -- catching name-mapping gaps (``arctan2``->``atan2``,
``rint``->``ANINT``, ``copysign``->``SIGN``), promotion collisions, and
the inline-expression forms (``square``/``reciprocal``/``sign``/
``degrees``/``radians``) that must read identically in C and Fortran.

Each case builds a tiny kernel, emits it, compiles a shared library and
calls it via ctypes (the binding JSON drives the argument order, exactly
like the numerical oracle). Skips a backend whose compiler is absent.
"""
import ctypes
import json
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from numpyto_c.frontend import parse_kernel
from numpyto_c.lowering import lower
from numpyto_c.emit import emit_c, emit_cpp  # noqa: E402
from numpyto_c.bindings import emit_binding  # noqa: E402
from numpyto_fortran.emit import emit_fortran  # noqa: E402

_CT = {"int": ctypes.c_int, "double": ctypes.c_double, "int64": ctypes.c_int64, "int32": ctypes.c_int32}

#: unary ufuncs validated on an array operand in (0.1, 0.9) -- a domain
#: valid for every one (arcsin/arctanh/log included).
UNARY = [
    "tan", "sinh", "cosh", "arcsin", "arccos", "arctan", "arcsinh", "arctanh", "exp2", "expm1", "log2", "log10",
    "log1p", "cbrt", "floor", "ceil", "trunc", "rint", "around", "square", "reciprocal", "sign", "degrees", "radians"
]
BINARY = ["arctan2", "hypot", "copysign", "fmod", "fmax", "fmin"]

# symbol == file stem == binding symbol (the canonical scheme: one name, no
# _auto / per-compiler suffix). emit_binding(base_name="k") records "k" for every
# language, so each backend emits its source with fn_name "k".
_BACKENDS = {
    "c": (emit_c, "c", "k", ["gcc", "-O2", "-std=c17", "-shared", "-fPIC"], "gcc"),
    "cpp": (emit_cpp, "cpp", "k", ["g++", "-O2", "-std=c++20", "-shared", "-fPIC"], "g++"),
    "fortran": (emit_fortran, "fortran", "k",
                ["gfortran", "-O2", "-ffree-form", "-ffree-line-length-none", "-std=f2018", "-shared",
                 "-fPIC"], "gfortran"),
}


def _kernel_ir(d, fn, nargs):
    arr = (["a", "out"] if nargs == 1 else ["a", "b", "out"])
    body = (f"out[:] = np.{fn}(a)" if nargs == 1 else f"out[:] = np.{fn}(a, b)")
    (d / "k_numpy.py").write_text(f"import numpy as np\ndef k({', '.join(arr[:-1])}, out):\n    {body}\n")
    (d / "k.json").write_text(
        json.dumps({
            "benchmark": {
                "name": fn,
                "short_name": "k",
                "relative_path": ".",
                "module_name": "k",
                "func_name": "k",
                "kind": "m",
                "domain": "d",
                "dwarf": "d",
                "parameters": {
                    "S": {
                        "N": 32
                    }
                },
                "init": {
                    "func_name": "",
                    "input_args": [],
                    "output_args": [],
                    "shapes": {
                        x: "(N,)"
                        for x in arr
                    }
                },
                "input_args": arr,
                "array_args": arr,
                "output_args": ["out"]
            }
        }))
    return lower(parse_kernel(d / "k_numpy.py", d / "k.json"))


def _numpy_ref(fn, nargs, a, b):
    out = np.empty_like(a)
    g = {"np": np}
    exec(
        f"def k({'a, out' if nargs == 1 else 'a, b, out'}):\n"
        f"    out[:] = np.{fn}({'a' if nargs == 1 else 'a, b'})", g)
    (g["k"](a.copy(), out) if nargs == 1 else g["k"](a.copy(), b.copy(), out))
    return out


def _run_backend(backend, fn, nargs):
    emit, sym_key, fname, compile_cmd, exe = _BACKENDS[backend]
    if shutil.which(exe) is None:
        pytest.skip(f"{exe} not available")
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        kir = _kernel_ir(d, fn, nargs)
        ext = {"c": "c", "cpp": "cpp", "fortran": "f90"}[backend]
        src = d / f"k.{ext}"
        src.write_text(emit(kir, fn_name=fname))
        emit_binding(kir, d / "kb.json", base_name="k")
        binding = json.loads((d / "kb.json").read_text())
        so = d / "k.so"
        r = subprocess.run(compile_cmd + [str(src), "-o", str(so)], capture_output=True, text=True)
        assert r.returncode == 0, f"{backend} compile failed:\n{r.stderr}"
        rng = np.random.default_rng(0)
        a = rng.uniform(0.1, 0.9, 32)
        b = rng.uniform(0.1, 0.9, 32)
        expected = _numpy_ref(fn, nargs, a, b)
        got = np.zeros(32)
        data = {"a": a.copy(), "b": b.copy(), "out": got, "N": 32}
        lib = ctypes.CDLL(str(so))
        cfn = lib[binding["symbols"][sym_key]]
        cargs, keep = [], []
        for arg in binding["args"]:
            nm, kind = arg["name"], arg["kind"]
            if kind in _CT:
                # Scalars are passed BY VALUE: the emitted Fortran is C-bound
                # (``value`` attribute), same as C/C++ -- matching cpp_runtime.
                v = int(data[nm]) if kind.startswith("int") else float(data[nm])
                cargs.append(_CT[kind](v))
            else:
                buf = np.ascontiguousarray(data[nm])
                keep.append(buf)
                cargs.append(buf.ctypes.data_as(ctypes.c_void_p))
        cfn(*cargs)
        assert np.allclose(got, expected, rtol=1e-9,
                           atol=1e-9), (f"{backend}/{fn}: max diff {np.abs(got - expected).max():.2e}")


@pytest.mark.parametrize("backend", list(_BACKENDS))
@pytest.mark.parametrize("fn", UNARY)
def test_unary_transcendental(backend, fn):
    _run_backend(backend, fn, 1)


@pytest.mark.parametrize("backend", list(_BACKENDS))
@pytest.mark.parametrize("fn", BINARY)
def test_binary_transcendental(backend, fn):
    _run_backend(backend, fn, 2)
