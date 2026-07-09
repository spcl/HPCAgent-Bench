"""Numpy-faithfulness regression tests for the numba / pythran / cupy emitters.

Three semantic bugs -- one per backend -- are pinned here:

1. numba: the ``njit_parallel`` flavor used to rewrite the FIRST ``range``
   for-loop to ``nb.prange`` unconditionally, racing a loop-carried scan
   (``a[i] = a[i-1] + x[i]``). The rewrite now only fires on a loop a
   conservative dependency check PROVES independent; a scan / reduction /
   scatter stays serial (correct).

2. pythran: an ``#pythran export`` type used to default an unknown param /
   dtype to ``float64``, type-punning a bool / int argument in the oracle's
   positional call. Unknown dtypes now fail loudly. Separately, pythran's
   ``np.maximum`` / ``np.minimum`` / ``np.sign`` SUPPRESS NaN (unlike numpy);
   they are rewritten to NaN-propagating forms.

3. cupy: ``import numpy`` (no alias) was rebound to ``import cupy as cp`` while
   bare ``numpy.`` refs became ``cupy.`` -- only ``cp`` was bound, so every
   ``cupy.foo`` raised ``NameError``. All numpy refs now bind to ``cp``.

The source-level asserts check the emitted text directly; the numerical asserts
round-trip each idiom through the ``run_op`` oracle (or, for cupy, a guarded GPU
run) against numpy.
"""
import importlib.util
import pathlib
import tempfile

import numpy as np
import pytest

from numpyto_cupy.emit import emit_cupy
from numpyto_numba.emit import emit_numba
from numpyto_pythran.emit import _pythran_scalar_type


# --------------------------------------------------------------------------- #
# Shared oracle loader (mirrors test_jax_semantics_fixes).                     #
# --------------------------------------------------------------------------- #
def _oracle():
    import shutil
    if not (shutil.which("gcc") and shutil.which("gfortran") and shutil.which("g++")):
        pytest.skip("gcc/g++/gfortran needed for the native oracle emit step")
    try:
        import _op_oracle
    except ImportError:
        spec = importlib.util.spec_from_file_location("_op_oracle",
                                                      pathlib.Path(__file__).resolve().parent / "_op_oracle.py")
        _op_oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_op_oracle)
    return _op_oracle


def _assert_ok(status, backend, label):
    s = status[backend]
    if s.startswith("skip"):
        pytest.skip(f"{label}: {backend} {s}")
    assert not s.startswith("FAIL"), f"{label}: {s}"


# --------------------------------------------------------------------------- #
# 1. numba: prange only on a provably-independent loop.                        #
# --------------------------------------------------------------------------- #
_SCAN = ("import numpy as np\n"
         "def scan(x, a):\n"
         "    a[0] = x[0]\n"
         "    for i in range(1, x.shape[0]):\n"
         "        a[i] = a[i - 1] + x[i]\n")

_ELEMENTWISE = ("import numpy as np\n"
                "def f(x, out):\n"
                "    for i in range(x.shape[0]):\n"
                "        out[i] = x[i] * 2.0\n")


def test_numba_parallel_does_not_prange_scan():
    # The scan reads a[i-1] (a previously-written cell): prange would reorder
    # iterations and read a not-yet-written value. Must stay serial ``range``.
    out = emit_numba(_SCAN, flavor="njit_parallel")
    assert "nb.prange" not in out
    assert "in range(1, x.shape[0])" in out


def test_numba_parallel_pranges_independent_loop():
    # A pure elementwise map (each iteration touches only its own cell) IS safe.
    out = emit_numba(_ELEMENTWISE, flavor="njit_parallel")
    assert "for i in nb.prange(" in out


@pytest.mark.parametrize("body,label", [
    ("        out[0] += x[i]\n", "scalar/same-cell reduction"),
    ("        out[int(perm[i])] = x[i]\n", "data-dependent scatter"),
    ("        out[i] = out[i - 1] + x[i]\n", "index-shifted stencil"),
])
def test_numba_parallel_refuses_dependent_loops(body, label):
    src = ("import numpy as np\n"
           "def f(x, out, perm):\n"
           "    for i in range(x.shape[0]):\n" + body)
    assert "nb.prange" not in emit_numba(src, flavor="njit_parallel"), label


def test_numba_scan_via_oracle():
    # End-to-end through run_op (default njit flavor): the prefix sum must match
    # numpy exactly on the numba backend.
    no = _oracle()
    st = no.run_op(_SCAN,
                   "scan", {"x": np.arange(1.0, 6.0)}, {"a": (5, )}, {"N": 5},
                   shapes={
                       "x": "(N,)",
                       "a": "(N,)"
                   },
                   backends=("numba", ))
    _assert_ok(st, "numba", "numba-scan")


def test_numba_parallel_flavor_scan_stays_correct():
    # Prove the njit_parallel-emitted scan is numerically correct (serial fallback
    # produces the true prefix sum; a blind prange would race and diverge).
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba not installed")
    x = np.arange(1.0, 8.0)
    nb_src = emit_numba(_SCAN, flavor="njit_parallel")
    assert "nb.prange" not in nb_src
    with tempfile.TemporaryDirectory() as td:
        mod = pathlib.Path(td) / "scan_p.py"
        mod.write_text(nb_src)
        spec = importlib.util.spec_from_file_location("scan_p", mod)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        a = np.zeros_like(x)
        m.scan(x.copy(), a)
    np.testing.assert_allclose(a, np.cumsum(x))


# --------------------------------------------------------------------------- #
# 2. pythran: dtype fail-loud + NaN-propagating max/min/sign.                  #
# --------------------------------------------------------------------------- #
def test_pythran_scalar_type_resolves_int_bool():
    # Known non-float dtypes map to their pythran spelling (not float64).
    assert _pythran_scalar_type("int", "x") == "int"
    assert _pythran_scalar_type("int32", "x") == "int32"
    assert _pythran_scalar_type("bool", "x") == "bool"


def test_pythran_scalar_type_unknown_fails_loud():
    # An unmappable dtype must raise, NOT silently become float64 (a wrong
    # element type type-puns the oracle's positional call).
    with pytest.raises(ValueError, match="cannot map dtype"):
        _pythran_scalar_type("float128", "array 'q'")


def test_pythran_int_param_roundtrips():
    # ``k`` (used only as a ``range`` bound) is declared ``int`` and drives the
    # loop count; the result must match numpy on the pythran backend.
    no = _oracle()
    src = ("import numpy as np\n"
           "def f(x, k, out):\n"
           "    for i in range(k):\n"
           "        out[i] = x[i] * 2.0\n")
    st = no.run_op(src,
                   "f", {
                       "x": np.arange(5.0),
                       "k": 3
                   }, {"out": (5, )}, {"N": 5},
                   shapes={
                       "x": "(N,)",
                       "out": "(N,)"
                   },
                   backends=("pythran", ))
    _assert_ok(st, "pythran", "pythran-int-param")


def test_pythran_maximum_propagates_nan():
    # numpy's np.maximum propagates NaN; pythran's suppresses it. The rewrite
    # restores propagation -- checked with equal_nan comparison in the oracle.
    no = _oracle()
    src = "import numpy as np\ndef f(a, b, out):\n    out[:] = np.maximum(a, b)\n"
    st = no.run_op(src,
                   "f", {
                       "a": np.array([np.nan, 1.0, 3.0]),
                       "b": np.array([2.0, np.nan, 1.0])
                   }, {"out": (3, )}, {"N": 3},
                   shapes={
                       "a": "(N,)",
                       "b": "(N,)",
                       "out": "(N,)"
                   },
                   backends=("pythran", ))
    _assert_ok(st, "pythran", "pythran-maximum-nan")


def test_pythran_sign_propagates_nan():
    no = _oracle()
    src = "import numpy as np\ndef f(a, out):\n    out[:] = np.sign(a)\n"
    st = no.run_op(src,
                   "f", {"a": np.array([np.nan, -2.0, 0.0, 5.0])}, {"out": (4, )}, {"N": 4},
                   shapes={
                       "a": "(N,)",
                       "out": "(N,)"
                   },
                   backends=("pythran", ))
    _assert_ok(st, "pythran", "pythran-sign-nan")


# --------------------------------------------------------------------------- #
# 3. cupy: consistent ``cp`` binding + import-form handling.                   #
# --------------------------------------------------------------------------- #
_CUPY_SRC = ("import numpy\n"
             "def f(a, out):\n"
             "    out[:] = numpy.sqrt(a) + numpy.pi\n")


def test_cupy_import_form_binds_cp_consistently():
    out = emit_cupy(_CUPY_SRC)
    assert "import cupy as cp" in out
    assert "import numpy" not in out
    # Every numpy ref rebound to ``cp`` -- no undefined ``cupy.`` / ``numpy.`` left.
    assert "cupy." not in out
    assert "numpy." not in out
    assert "cp.sqrt(a)" in out and "cp.pi" in out


def test_cupy_import_form_runs_on_gpu():
    cp = pytest.importorskip("cupy")
    try:
        _ = int((cp.arange(3) + 1).sum())  # probe a real device.
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no cupy runtime: {type(exc).__name__}: {exc}")
    ns: dict = {}
    exec(compile(emit_cupy(_CUPY_SRC), "<cupy>", "exec"), ns)
    a = np.arange(1.0, 6.0)
    a_dev = cp.asarray(a)
    out_dev = cp.zeros(5)
    ns["f"](a_dev, out_dev)
    np.testing.assert_allclose(cp.asnumpy(out_dev), np.sqrt(a) + np.pi)
