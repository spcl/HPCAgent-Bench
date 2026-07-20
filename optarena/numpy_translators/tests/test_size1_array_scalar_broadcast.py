"""A scalar-used local assigned an all-size-1 broadcast is a SCALAR, not a ``(1,)`` array.

The TSVC s315 argmax nest (a max/argmax over ``a`` seeded from a ``(1,)`` buffer ``x``) lowered to::

    for i in range(N):
        c = (a[i] > x)      # x is shape (1,): numpy broadcasts -> c is a (1,) bool array
        if c:
            ...

numpyto typed ``c`` from the broadcast extent as a size-1 array (``bool c[1]`` + a ``for __w0`` write),
yet ``c`` is USED as a scalar (``c = 0`` init, ``if (c)``, ``out[0] = c``). The declaration (scalar) and
the array-style writes (``memset(c, ...)`` / ``c[__w0] = ...``) disagreed, so the emitted C / Fortran did
not compile. numpyto already reads a size-1 array element-wise as ``x[0]`` / ``x(1)``, so an all-size-1
broadcast is a SCALAR: the fix stops registering such a local as an array (``extent_is_scalar``) and
scalarises a bare size-1-array READ to ``x[0]`` / ``x(1)`` (without double-indexing an explicit ``x[0]``).
"""
import ast
import json
import pathlib
import tempfile

import numpy as np

from _op_oracle import run_op

_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")

# max + argmax over ``a`` seeded from the size-1 buffer x[0]; ``c`` is the s315 scalar-from-size-1-broadcast.
_SRC = ("def f(a, x, out):\n"
        "    m = x[0]\n"
        "    idx = 0\n"
        "    for i in range(len(a)):\n"
        "        c = (a[i] > x)\n"
        "        if c:\n"
        "            m = a[i]\n"
        "            idx = i\n"
        "    out[0] = m + float(idx)\n")


def _all_ok(res):
    return all(v == "ok" or v.startswith("skip") for v in res.values()), res


def test_extent_is_scalar_helper():
    from numpyto_common.lib_nodes import extent_is_scalar
    one = tuple(ast.parse("1").body[0].value for _ in range(1))
    assert extent_is_scalar(one)  # (1,) -> scalar
    assert extent_is_scalar((ast.parse("1").body[0].value, ast.parse("1").body[0].value))  # (1, 1) -> scalar
    assert extent_is_scalar(())  # rank-0 is already scalar
    assert not extent_is_scalar(None)  # unknown extent is NOT asserted scalar
    assert not extent_is_scalar((ast.parse("N").body[0].value, ))  # (N,) is a real array


def test_scalar_local_from_size1_broadcast_all_backends():
    a = np.array([0.2, 0.9, 0.5, 0.7, 0.1, 0.95, 0.3], dtype=np.float64)
    ok, res = _all_ok(
        run_op(_SRC,
               "f", {
                   "a": a,
                   "x": np.array([0.0])
               }, {"out": (1, )}, {"N": len(a)},
               shapes={
                   "a": "(N,)",
                   "x": "(1,)",
                   "out": "(1,)"
               },
               backends=_ALL))
    assert ok, res


def _kir(src):
    from numpyto_common.frontend import parse_kernel
    from numpyto_common.lowering import lower
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(src)
    bi = {
        "benchmark": {
            "name": "k",
            "short_name": "k",
            "relative_path": "",
            "module_name": "k",
            "func_name": "f",
            "parameters": {
                "S": {
                    "N": 8
                }
            },
            "input_args": ["a", "x", "out"],
            "array_args": ["a", "x", "out"],
            "output_args": ["out"],
            "init": {
                "shapes": {
                    "a": "(N,)",
                    "x": "(1,)",
                    "out": "(1,)"
                },
                "dtypes": {}
            },
        }
    }
    (d / "bi.json").write_text(json.dumps(bi))
    return lower(parse_kernel(d / "k_numpy.py", d / "bi.json"))


def test_c_declares_scalar_and_scalarises_size1_read():
    from numpyto_c.emit import emit_c
    c = emit_c(_kir(_SRC), fn_name="f")
    assert "c[1]" not in c and "bool c[" not in c  # ``c`` is a scalar, not a size-1 array
    assert "a[i] > x[0]" in c  # the bare size-1 array ``x`` is read as its element
    assert "x[0][0]" not in c and "x[0][" not in c  # ...but an explicit x[0] is NOT double-indexed


def test_fortran_declares_scalar_and_scalarises_size1_read():
    from numpyto_fortran.emit import emit_fortran
    f = emit_fortran(_kir(_SRC), fn_name="f")
    assert "> x)" not in f  # ``x`` is scalarised, so no rank-0-vs-rank-1 ``a(i+1) > x``
    assert "x(1)" in f  # the size-1 array is read as element ``x(1)``
    assert "x(1)(" not in f  # explicit subscripts are not double-indexed
