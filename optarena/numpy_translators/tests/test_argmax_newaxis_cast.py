"""Unit tests for the emitter feature-gaps surfaced by the viterbi kernel
(graphical-models dwarf) plus the ``np.<dtype>()`` scalar cast:

* ``np.<dtype>(x)`` scalar constructor -> a C cast / Fortran conversion
  intrinsic (kind from the dtype registry, never hardcoded).
* ``np.argmax(a, axis=k)`` / ``np.argmin`` return an INDEX ARRAY (int64),
  not a scalar -- so the hoisted ``__cb`` temp is allocated + typed.
* ``A[i] = B`` with a partial-subscript LHS (a row) lowers to a per-element
  copy loop instead of a pointer store.
* ``V[:, None]`` newaxis broadcast binds the slice to the LEADING iter
  (``V[w0]``), not the trailing one -- the ``_SubscriptifyNames`` rewriter
  that drives the whole-array per-element expansion.

Each test pins one rule so a regression points straight at it.
"""

import ast
import json
import pathlib
import subprocess
import sys
import tempfile

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from numpyto_common.lib_nodes import _CallHoister
from numpyto_common.lowering import _SubscriptifyNames, _WholeArrayAssignRewriter


def _expr(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


def _ivars(*names):
    return [ast.Name(id=n, ctx=ast.Load()) for n in names]


# --------------------------------------------------------------------------- #
# A. ``_SubscriptifyNames`` -- newaxis binds the slice to the right iter       #
# --------------------------------------------------------------------------- #


def _subscriptify(src, iters, shapes):
    return ast.unparse(_SubscriptifyNames(shapes, iters).visit(_expr(src)))


def test_newaxis_trailing_binds_leading_iter():
    # ``V[:, None]`` at iters (w0, w1): the slice is axis 0 -> V[w0]; the
    # None axis consumes w1 but adds no source index. (Was wrongly V[w1].)
    assert _subscriptify("V[:, None]", ["__w0", "__w1"], {"V": ("K", )}) == "V[__w0]"


def test_newaxis_leading_binds_trailing_iter():
    # ``V[None, :]`` at iters (w0, w1): None consumes w0, slice -> V[w1].
    assert _subscriptify("V[None, :]", ["__w0", "__w1"], {"V": ("K", )}) == "V[__w1]"


def test_newaxis_between_axes_on_2d():
    # ``A[:, None, :]`` on (N, M) at iters (w0, w1, w2) -> A[w0, w2].
    out = _subscriptify("A[:, None, :]", ["__w0", "__w1", "__w2"], {"A": ("N", "M")})
    assert out == "A[__w0, __w2]"


def test_plain_slice_pair_unchanged_by_newaxis_fix():
    # Regression guard: no newaxis -> the right-alignment is unchanged.
    out = _subscriptify("A[:, j]", ["__w0"], {"A": ("N", "M")})
    assert out == "A[__w0, j]"


# --------------------------------------------------------------------------- #
# B. ``np.argmax(a, axis=k)`` hoists to an int64 INDEX ARRAY                    #
# --------------------------------------------------------------------------- #


def _hoist(src, shapes):
    """Run the call-hoister over ``target = <src>`` and return
    (array_temps, scalar_temps, local_dtypes)."""
    array_temps, scalar_temps, local_dtypes = {}, {}, {}
    h = _CallHoister(shapes, scalar_temps, array_temps, [0], local_dtypes)
    h.visit(_expr(src))
    return array_temps, scalar_temps, local_dtypes


def test_argmax_axis_hoists_to_int64_array_temp():
    arr, scal, dts = _hoist("np.argmax(scores, axis=0)", {"scores": ("K", "K")})
    assert arr and not scal  # array-returning, not scalar
    (name, shape), = arr.items()
    assert shape == ("K", )  # kept axis (axis 1)
    assert dts[name] == "int64"  # index dtype, not double


def test_argmin_axis_hoists_to_int64_array_temp():
    arr, scal, dts = _hoist("np.argmin(scores, axis=1)", {"scores": ("K", "M")})
    (name, shape), = arr.items()
    assert shape == ("K", ) and dts[name] == "int64"


def test_argmax_no_axis_stays_scalar():
    # Regression guard: full argmax (axis=None) is still a scalar temp.
    arr, scal, dts = _hoist("np.argmax(V)", {"V": ("K", )})
    assert scal and not arr


# --------------------------------------------------------------------------- #
# C. Partial-subscript LHS row copy: ``back[t] = cb``                          #
# --------------------------------------------------------------------------- #


def test_partial_subscript_assign_expands_to_copy_loop():
    shapes = {"back": ("T", "K"), "cb": ("K", )}
    rw = _WholeArrayAssignRewriter(shapes, real_arrays=set(shapes))
    node = ast.parse("back[t] = cb").body[0]
    out = rw.visit(node)
    out = out if isinstance(out, list) else [out]
    mod = ast.fix_missing_locations(ast.Module(body=out, type_ignores=[]))
    fors = [s for s in ast.walk(mod) if isinstance(s, ast.For)]
    assert len(fors) == 1  # one loop over the K row
    src = ast.unparse(mod)
    assert "back[t, __w0] = cb[__w0]" in src  # element-wise row copy


def test_full_subscript_assign_not_expanded():
    # Regression guard: a fully-indexed scalar store is left alone.
    shapes = {"back": ("T", "K"), "cb": ("K", )}
    rw = _WholeArrayAssignRewriter(shapes, real_arrays=set(shapes))
    node = ast.parse("back[t, k] = cb[k]").body[0]
    out = rw.visit(node)
    assert not isinstance(out, list)  # untouched single Assign


# --------------------------------------------------------------------------- #
# D. ``np.<dtype>(x)`` scalar cast (full emit pipeline, C + Fortran)           #
# --------------------------------------------------------------------------- #

_CAST_KERNEL = """import numpy as np


def cast_demo(out_i, out_f, xf, xi, N):
    out_i[0] = np.int64(xf[0])
    out_f[0] = np.float64(xi[0])
"""

_CAST_BENCH = {
    "benchmark": {
        "func_name": "cast_demo",
        "array_args": ["out_i", "out_f", "xf", "xi"],
        "input_args": ["out_i", "out_f", "xf", "xi"],
        "output_args": ["out_i", "out_f"],
        "init": {
            "shapes": {
                "out_i": "(1,)",
                "out_f": "(1,)",
                "xf": "(N,)",
                "xi": "(N,)"
            },
            "dtypes": {
                "out_i": "int64",
                "out_f": "float64",
                "xf": "float64",
                "xi": "int64"
            },
        },
        "parameters": {
            "S": {
                "N": 4
            }
        },
        "short_name": "cast_demo",
    },
    "track": "foundation",
    "precisions": ["fp64"],
}


def _emit(target):
    from numpyto_common.frontend import parse_kernel
    from numpyto_common.lowering import lower
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        kp = d / "cast_demo_numpy.py"
        kp.write_text(_CAST_KERNEL)
        bi = d / "bi.json"
        bi.write_text(json.dumps(_CAST_BENCH))
        kir = lower(parse_kernel(kp, bi))
        if target == "c":
            from numpyto_c.emit import emit_c
            return emit_c(kir, fn_name="cast_demo")
        from numpyto_fortran.emit import emit_fortran
        return emit_fortran(kir, fn_name="cast_demo")


def test_np_dtype_cast_c():
    src = _emit("c")
    assert "(int64_t)(xf[0])" in src  # np.int64 -> C int cast
    assert "(double)(xi[0])" in src  # np.float64 -> C double cast


def test_np_dtype_cast_fortran():
    src = _emit("fortran")
    # Conversion intrinsic + KIND token from the registry (never hardcoded);
    # index 0 lowers to the 1-based ``(0) + 1`` subscript.
    assert "INT(xf((0) + 1), c_int64_t)" in src
    assert "REAL(xi((0) + 1), c_double)" in src
