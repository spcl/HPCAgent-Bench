"""Negative CONSTANT index ``a[-k]`` -> explicit ``a[N-k]`` for the C / C++ ABI.

numpy (and fortran / numba / pythran / jax) wrap a negative index from the end
(``a[-1]`` is the last element), but C has no negative indexing -- ``a[-1]``
underflows the pointer and reads garbage. The C emitter therefore normalizes a
negative constant index against the array's known dimension size, at both read
and write positions. A negative SLICE bound (``a[:-1]``) is a different construct
and is left to the slice lowering (it already works).
"""
import ast

import numpy as np

from _op_oracle import run_op
from numpyto_c.emit import _negative_const_k

_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")


def _all_ok(res):
    return all(v == "ok" or v.startswith("skip") for v in res.values()), res


# --------------------------------------------------------------------------- #
# _negative_const_k: -1 is a UnaryOp, not a Constant                          #
# --------------------------------------------------------------------------- #


def test_negative_const_k_recognizes_forms():
    assert _negative_const_k(ast.parse("-1", mode="eval").body) == 1
    assert _negative_const_k(ast.parse("-3", mode="eval").body) == 3
    assert _negative_const_k(ast.parse("2", mode="eval").body) is None  # non-negative
    assert _negative_const_k(ast.parse("i", mode="eval").body) is None  # not constant
    assert _negative_const_k(ast.parse("True", mode="eval").body) is None  # bool is not an index


# --------------------------------------------------------------------------- #
# the C emit turns ``a[-1]`` into ``a[N - 1]`` (no literal ``[-1]``)          #
# --------------------------------------------------------------------------- #


def _emit_c(src, inputs, shapes, syms):
    import json
    import pathlib
    import tempfile
    from numpyto_common.frontend import parse_kernel
    from numpyto_common.lowering import lower
    from numpyto_c.emit import emit_c
    d = pathlib.Path(tempfile.mkdtemp())
    npy = d / "k_numpy.py"
    npy.write_text(src)
    bi = {
        "benchmark": {
            "name": "k",
            "short_name": "k",
            "relative_path": "",
            "module_name": "k",
            "func_name": "f",
            "parameters": {
                "S": dict(syms)
            },
            "input_args": inputs,
            "array_args": [a for a in inputs if a in shapes],
            "output_args": [],
            "init": {
                "shapes": shapes
            }
        }
    }
    (d / "bi.json").write_text(json.dumps(bi))
    return emit_c(lower(parse_kernel(npy, d / "bi.json")), fn_name="f")


def test_c_emit_normalizes_bare_negative_index():
    c = _emit_c("import numpy as np\ndef f(a, out):\n out[0] = a[-1]\n", ["a", "out"], {
        "a": "(N,)",
        "out": "(2,)"
    }, {"N": 6})
    assert "a[-1]" not in c and "N - 1" in c


def test_c_emit_leaves_positive_index_alone():
    c = _emit_c("import numpy as np\ndef f(a, out):\n out[0] = a[2]\n", ["a", "out"], {
        "a": "(N,)",
        "out": "(2,)"
    }, {"N": 6})
    assert "a[2]" in c


# --------------------------------------------------------------------------- #
# numerical: bit-exact vs numpy across every backend                          #
# --------------------------------------------------------------------------- #


def test_bare_negative_index_read():
    a = np.arange(6, dtype=np.float64)
    ok, res = _all_ok(
        run_op("import numpy as np\ndef f(a, out):\n out[0] = a[-1]\n out[1] = a[-2]\n",
               "f", {"a": a}, {"out": (2, )}, {"N": 6},
               shapes={
                   "a": "(N,)",
                   "out": "(2,)"
               },
               backends=_ALL))
    assert ok, res


def test_negative_index_write():
    a = np.arange(6, dtype=np.float64)
    ok, res = _all_ok(
        run_op("import numpy as np\ndef f(a, out):\n out[:] = a\n out[-1] = 99.0\n",
               "f", {"a": a}, {"out": (6, )}, {"N": 6},
               shapes={
                   "a": "(N,)",
                   "out": "(N,)"
               },
               backends=_ALL))
    assert ok, res


def test_negative_index_2d_mixed_axes():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    ok, res = _all_ok(
        run_op("import numpy as np\ndef f(a, out):\n out[0] = a[-1, -1]\n out[1] = a[1, -1]\n out[2] = a[-1, 2]\n",
               "f", {"a": a}, {"out": (3, )}, {
                   "M": 3,
                   "N": 4
               },
               shapes={
                   "a": "(M, N)",
                   "out": "(3,)"
               },
               backends=_ALL))
    assert ok, res


def test_negative_slice_bound_still_works():
    # ``a[:-1]`` is a slice bound, NOT an index -- left to the slice lowering.
    a = np.arange(6, dtype=np.float64)
    ok, res = _all_ok(
        run_op("import numpy as np\ndef f(a, out):\n out[:] = a[1:] - a[:-1]\n",
               "f", {"a": a}, {"out": (5, )}, {"N": 6},
               shapes={
                   "a": "(N,)",
                   "out": "(5,)"
               },
               backends=_ALL))
    assert ok, res
