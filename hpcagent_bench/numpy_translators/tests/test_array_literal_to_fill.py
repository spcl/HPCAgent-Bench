# Copyright 2025 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``np.array([...])`` lowered to an allocation plus one store per element.

fv3 names the handful of rows it must touch out of order as a small literal array --
``ia = np.array([i_start - 1, i_end])`` -- and then indexes through it. The backends have no
array CONSTRUCTOR, only allocations and stores, so the call was refused where it stood.

The element type is the whole risk. numpy types an int list as an INTEGER buffer; left implicit
the backends default to double, and an index vector of doubles is either a build error or a
meaningless subscript. So the type is taken from an explicit ``dtype=``, from all-literal
elements, or -- for the symbolic elements above -- from the name being read ONLY as a subscript
index, which makes it an index vector. Anything else keeps the refusal, which is why the two
negative cases below matter as much as the positive ones.
"""
import json
import pathlib
import re
import tempfile

import numpy as np
import pytest

from _op_oracle import _bench_info, run_op
from numpyto_c.emit import emit_c
from numpyto_common.frontend import parse_kernel
from numpyto_common.lowering import lower


def emit_c_for(src: str, func: str, shapes=None, syms=None, inputs=("src", ), outputs=("out", )) -> str:
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(src)
    shapes = shapes or {"src": "(N, M)", "out": "(N, M)"}
    syms = syms or {"N": 8, "M": 4}
    (d / "bi.json").write_text(json.dumps(_bench_info(func, list(inputs), list(outputs), shapes, syms, None)))
    return emit_c(lower(parse_kernel(d / "k_numpy.py", d / "bi.json")), fn_name=func)


def _decl_of(text: str, name: str) -> str:
    """The declaration line for local ``name`` in the emitted C."""
    hits = re.findall(rf"^\s*(\w[\w ]*?)\s+{name}\[", text, re.M)
    assert hits, f"no declaration of {name!r} in:\n{text}"
    return hits[0]


def test_symbolic_index_vector_runs():
    """The fv3 form end to end: elements are arithmetic over scalars, and the array is only
    ever a subscript index."""
    src = ("import numpy as np\n"
           "def pick(src, out):\n"
           "    N = src.shape[0]\n"
           "    ia = np.array([1, N - 2])\n"
           "    out[ia, :] = src[ia - 1, :] * 2.0\n")
    rng = np.random.default_rng(0)
    res = run_op(src,
                 "pick", {"src": rng.standard_normal((8, 4))}, {"out": (8, 4)}, {
                     "N": 8,
                     "M": 4
                 },
                 shapes={
                     "src": "(N, M)",
                     "out": "(N, M)"
                 },
                 backends=("c", "fortran"))
    assert set(res) == {"c", "fortran"}, res
    assert all(v == "ok" for v in res.values()), res


def test_int_literal_list_is_an_integer_buffer():
    """An int list is an integer buffer, as it is in numpy. Emitted as double it would be a
    subscript of the wrong type, which is the failure this rule exists to prevent."""
    text = emit_c_for(("import numpy as np\n"
                       "def pick(src, out):\n"
                       "    ia = np.array([0, 2, 3])\n"
                       "    out[ia, :] = src[ia, :] * 2.0\n"), "pick")
    assert "int" in _decl_of(text, "ia"), _decl_of(text, "ia")


def test_float_literal_list_is_a_float_buffer():
    """The same rule the other way: a float list is NOT narrowed to an index type just because
    the mechanism was built for index vectors."""
    text = emit_c_for(("import numpy as np\n"
                       "def pick(src, out):\n"
                       "    coef = np.array([1.5, 2.5])\n"
                       "    out[0, :] = src[0, :] * coef[0] + coef[1]\n"), "pick")
    decl = _decl_of(text, "coef")
    assert "double" in decl or "float" in decl, decl


def test_elements_stored_in_order():
    """The stores are what carry the values; a missing or reordered one is silent."""
    text = emit_c_for(("import numpy as np\n"
                       "def pick(src, out):\n"
                       "    ia = np.array([0, 2, 3])\n"
                       "    out[ia, :] = src[ia, :] * 2.0\n"), "pick")
    assert re.search(r"ia\[0\] = 0;", text), text
    assert re.search(r"ia\[2\] = 3;", text), text


def test_symbolic_elements_not_read_as_an_index_still_refuse():
    """Nothing in the AST types ``lo``/``hi``, and the array is read as a VALUE here, not as a
    subscript. Guessing double would be a miscompile; the refusal is the correct answer."""
    src = ("import numpy as np\n"
           "def pick(src, out):\n"
           "    lo = src[0, 0]\n"
           "    hi = src[0, 1]\n"
           "    v = np.array([lo, hi])\n"
           "    out[0, :] = src[0, :] * v[0] + v[1]\n")
    with pytest.raises(NotImplementedError):
        emit_c_for(src, "pick")


def test_nested_literal_still_refuses():
    """``np.array([[...], [...]])`` builds a 2-D array; this rewriter claims the flat literal
    only, and a partial claim on the nested one would size the buffer wrong."""
    src = ("import numpy as np\n"
           "def pick(src, out):\n"
           "    m = np.array([[1.0, 2.0], [3.0, 4.0]])\n"
           "    out[0, :] = src[0, :] * m[0, 0]\n")
    with pytest.raises(NotImplementedError):
        emit_c_for(src, "pick")
