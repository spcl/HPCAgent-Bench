"""A by-value scalar declared a FLOAT in ``init.scalars`` is typed ``double`` -- never truncated to int.

A nest extractor (nest-forge) that outlines a loop nest carries every loop-local scalar into the kernel
boundary. A staged ``a_index = a[i]`` read of a float array is such a scalar: it holds a DOUBLE value, but
it is a by-value parameter, not an array. If the translator typed it ``int64_t`` (the default for a name in
the ``parameters`` sizing block), ``int64_t a_index = a[i]`` would TRUNCATE every value -- for data in
``[0, 1)`` to exactly ``0`` -- so a find-first / conditional-reduce nest reads the wrong value and the
whole kernel diverges from the numpy oracle.

The contract these tests pin: a name declared in ``init.scalars`` with a FLOAT default is a value scalar,
so the C / C++ signature declares it ``double`` and Fortran ``real(c_double)`` -- both a read-only
threshold and a body-assigned staged read. This is the numpyto side of the nest-forge boundary-dtype fix
(nest-forge emits such scalars under ``init.scalars`` from the SDFG symbol dtype, and types the matching
ctypes arg ``c_double``); locking it here keeps the two in step.
"""
import ctypes
import json
import pathlib
import tempfile

import numpy as np

# read-only float threshold: if `thr` truncates to int (0.5 -> 0) every positive `a[i]` passes.
_THRESH_SRC = ("def f(a, out, thr):\n"
               "    out[0] = 0.0\n"
               "    for i in range(len(a)):\n"
               "        if a[i] > thr:\n"
               "            out[0] = out[0] + a[i]\n")

# body-assigned staged read (the `a_index = a[i]` shape a nest extractor produces): `v` is written before
# read, holds a[i] (double). If typed int it truncates the comparison and the captured value.
_STAGED_SRC = ("def g(a, out, v):\n"
               "    out[0] = 0.0\n"
               "    for i in range(len(a)):\n"
               "        v = a[i]\n"
               "        if v > 0.5:\n"
               "            out[0] = out[0] + v\n")


def _bench_info(func, scalar):
    """bench_info for a kernel ``func(a, out, <scalar>)`` with ``scalar`` declared a FLOAT in
    ``init.scalars`` (NOT an integer sizing symbol in ``parameters``)."""
    return {
        "benchmark": {
            "name": "k",
            "short_name": "k",
            "relative_path": "",
            "module_name": "k",
            "func_name": func,
            "parameters": {
                "S": {
                    "N": 16
                }
            },
            "input_args": ["a", "out", scalar],
            "array_args": ["a", "out"],
            "output_args": ["out"],
            "init": {
                "shapes": {
                    "a": "(N,)",
                    "out": "(1,)"
                },
                "scalars": {
                    scalar: 0.0
                }
            },
        }
    }


def _emit(src, func, scalar):
    from numpyto_common.frontend import parse_kernel
    from numpyto_common.lowering import lower
    from numpyto_c.emit import emit_c, emit_cpp
    from numpyto_fortran.emit import emit_fortran
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(src)
    (d / "bi.json").write_text(json.dumps(_bench_info(func, scalar)))
    c = emit_c(lower(parse_kernel(d / "k_numpy.py", d / "bi.json")), fn_name=func)
    cpp = emit_cpp(lower(parse_kernel(d / "k_numpy.py", d / "bi.json")), fn_name=func)
    f90 = emit_fortran(lower(parse_kernel(d / "k_numpy.py", d / "bi.json")), fn_name=func)
    return d, c, cpp, f90


def _scalar_desc(src, func, scalar):
    from numpyto_common.frontend import parse_kernel
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(src)
    (d / "bi.json").write_text(json.dumps(_bench_info(func, scalar)))
    kir = parse_kernel(d / "k_numpy.py", d / "bi.json")
    return {s.name: s.dtype for s in kir.scalars}, {sym.name for sym in kir.symbols}


def test_init_scalar_float_is_value_scalar_not_size_symbol():
    # The frontend classifies a float-declared name as a float SCALAR, not an integer sizing SYMBOL.
    scalars, symbols = _scalar_desc(_THRESH_SRC, "f", "thr")
    assert scalars.get("thr") in ("float64", "double"), scalars
    assert "thr" not in symbols, f"thr wrongly classified as an int size symbol: {symbols}"


def test_threshold_scalar_declared_double_all_langs():
    _d, c, cpp, f90 = _emit(_THRESH_SRC, "f", "thr")
    assert "double thr" in c, c
    assert "double thr" in cpp, cpp
    # Fortran: a real(c_double) value dummy, never an integer (which would truncate 0.5 -> 0).
    assert "real(c_double)" in f90 and "thr" in f90, f90
    assert "integer(c_int64_t), value :: thr" not in f90, f90


def test_staged_body_assigned_scalar_declared_double_all_langs():
    _d, c, cpp, f90 = _emit(_STAGED_SRC, "g", "v")
    assert "double v" in c, c
    assert "double v" in cpp, cpp
    assert "integer(c_int64_t), value :: v" not in f90, f90


def test_staged_scalar_fortran_drops_intent_in_when_reassigned():
    """A value scalar the body REASSIGNS (``v = a[i]``, the staged read a nest
    extractor leaks) must NOT be ``intent(in)`` in Fortran: an intent(in) dummy on
    the LHS is a hard compile error. The ``value`` attribute keeps it a local copy
    so the per-element restage is legal and the ABI is unchanged."""
    _d, _c, _cpp, f90 = _emit(_STAGED_SRC, "g", "v")
    assert "real(c_double), value :: v" in f90, f90
    assert "intent(in) :: v" not in f90, f90


def test_staged_scalar_fortran_compiles_and_runs():
    """The regression that the dtype checks above MISS: ``real(c_double)`` v was
    right, but it was emitted ``value, intent(in)`` and gfortran rejected the
    ``v = a[i]`` assignment (``Dummy argument 'v' with INTENT(IN) in variable
    definition context``). Compile + run the emitted Fortran and check the masked
    sum matches numpy end to end."""
    import shutil
    import subprocess

    if shutil.which("gfortran") is None:
        import pytest
        pytest.skip("no gfortran")
    d, _c, _cpp, f90 = _emit(_STAGED_SRC, "g", "v")
    (d / "g.f90").write_text(f90)
    so = d / "libg.so"
    cmd = ["gfortran", "-O2", "-fPIC", "-shared", "-ffp-contract=off", str(d / "g.f90"), "-o", str(so)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr  # the intent(in) relaxation makes this compile

    rng = np.random.default_rng(0)
    a = rng.random(4096)  # in [0, 1)
    expected = float(a[a > 0.5].sum())
    out = np.zeros(1)
    # Bind by the emitted bind(C) signature order (arrays/syms/scalars per the
    # canonical emit); v is the reassigned value double, N the int64 size.
    sig = [ln for ln in f90.splitlines() if 'bind(C, name="g")' in ln]
    assert sig, f90  # subroutine is exported extern-C as `g`
    lib = ctypes.CDLL(str(so))
    fn = lib["g"]
    # Parameter order from the subroutine header (may span a continuation line).
    hdr = f90.split("subroutine g(", 1)[1].split(") bind", 1)[0].replace("&", " ").replace("\n", " ")
    names = [p.strip() for p in hdr.split(",") if p.strip()]
    argmap = {
        "a": a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        "out": out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        "v": ctypes.c_double(0.0),
        "N": ctypes.c_int64(len(a)),
    }
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_double) if n in ("a", "out") else (ctypes.c_double if n == "v" else ctypes.c_int64)
        for n in names
    ]
    fn.restype = None
    fn(*[argmap[n] for n in names])
    assert abs(out[0] - expected) < 1e-9, f"got {out[0]}, expected masked sum {expected}"


def _shutil_which(name):
    import shutil
    return shutil.which(name)


def test_threshold_scalar_not_truncated_end_to_end_c():
    # Discriminating run: thr = 0.5, a in [0, 1). A double `thr` sums only a[i] > 0.5; a truncated int
    # `thr` (== 0) would sum EVERY element. The two differ, so a regression to int is loud.
    cc = _shutil_which("gcc") or _shutil_which("clang")
    if cc is None:
        import pytest
        pytest.skip("no C compiler")
    import subprocess
    d, c, _cpp, _f90 = _emit(_THRESH_SRC, "f", "thr")
    (d / "f.c").write_text(c)
    so = d / "libf.so"
    r = subprocess.run(
        [cc, "-O2", "-fPIC", "-shared", "-ffp-contract=off",
         str(d / "f.c"), "-o", str(so)],
        capture_output=True,
        text=True)
    assert r.returncode == 0, r.stderr

    rng = np.random.default_rng(0)
    a = rng.random(4096)  # in [0, 1)
    thr = 0.5
    expected = float(a[a > thr].sum())
    truncated = float(a.sum())  # what an int(thr)==0 kernel would produce
    assert abs(expected - truncated) > 1.0  # the two semantics are well separated

    lib = ctypes.CDLL(str(so))
    fn = lib["f"]
    # Signature order is (a, out, thr, N) per the emitted C; bind by reading it would be sturdier, but the
    # canonical emit for this kernel is arrays-then-scalars-then-sizes -- assert the double lands correctly.
    sig = [ln for ln in c.splitlines() if "f(" in ln and "void" in ln][0]
    assert "double thr" in sig, sig
    out = np.zeros(1)
    # Build args in the emitted signature order.
    order = sig.split("f(")[1].split(")")[0].split(",")
    names = [p.strip().split()[-1].lstrip("*") for p in order]
    argmap = {
        "a": a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        "out": out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        "thr": ctypes.c_double(thr),
        "N": ctypes.c_int64(len(a))
    }
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_double) if n in ("a", "out") else (ctypes.c_double if n == "thr" else ctypes.c_int64)
        for n in names
    ]
    fn.restype = None
    fn(*[argmap[n] for n in names])
    assert abs(out[0] - expected) < 1e-9, f"got {out[0]}, expected float-threshold sum {expected}"
