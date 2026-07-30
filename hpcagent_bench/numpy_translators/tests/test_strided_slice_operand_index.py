"""A STRIDED slice used as an operand must be READ with its stride.

``_iter_extent_of`` has always divided a strided slice's trip count by ``|step|``
(``a[0::2]`` over 12 elements -> 6 iterations), but its partner
``_scalarize_at_iters`` -- which renders the operand at each of those iterations --
read only ``slice.lower`` and ignored ``step`` entirely. Extent and index therefore
disagreed: the loop ran the right NUMBER of times over the WRONG elements, walking
the first ``n // k`` entries contiguously instead of every ``k``-th one, and a
``a[::-1]`` operand was read FORWARD.

Silent in every case -- the shapes match, so nothing OOB, nothing to warn about:

* ``np.maximum(a[0::2], b)`` over ``a = 1..12`` gave ``[1 2 3 4 5 6]``, not ``[1 3 5 7 9 11]``;
* ``np.dot(a[0::2], ones(6))`` gave 21 instead of 36;
* ``np.maximum(a[::-1], 0)`` gave the un-reversed array.

The bare strided slice-ASSIGN (``lo[:] = b[0::2]``, dwt2d's Haar) goes through
lowering's own slice rewriters and was always correct; it is pinned here too so the
shared-index fix cannot regress it.
"""
import ast

import numpy as np
from _op_oracle import run_op

from numpyto_common.lib_nodes import _iter_extent_of, _scalarize_at_iters

_NATIVE = ("c", "cpp", "fortran")
_A12 = np.arange(1.0, 13.0)
_ZERO6 = np.zeros(6)


def _assert_ok(res):
    for backend, status in res.items():
        assert status == "ok" or status.startswith("skip"), f"{backend}: {status}"
    assert any(status == "ok" for status in res.values()), f"all skipped (vacuous): {res}"


def _index_of(expr: str, shape) -> str:
    """Render ``expr`` at a single iter ``i`` and return the unparsed index text."""
    node = ast.parse(expr, mode="eval").body
    out = _scalarize_at_iters(node, [ast.Name(id="i", ctx=ast.Load())], {"a": shape})
    return ast.unparse(ast.fix_missing_locations(out))


# ---- structural: the rendered index carries the stride ---- #


def test_positive_stride_scales_the_iter():
    assert _index_of("a[0::2]", ("12", )) == "a[i * 2]"
    assert _index_of("a[3::2]", ("12", )) == "a[i * 2 + 3]"


def test_full_axis_reverse_counts_down_from_the_last_element():
    assert _index_of("a[::-1]", ("N", )) == "a[i * -1 + (N - 1)]"
    assert _index_of("a[::-2]", ("12", )) == "a[i * -2 + (12 - 1)]"


def test_unit_stride_index_is_unchanged():
    # The step-free forms must render exactly as before -- no stray ``* 1``.
    assert _index_of("a[:]", ("N", )) == "a[i]"
    assert _index_of("a[2:]", ("N", )) == "a[i + 2]"


def test_extent_and_index_agree_on_the_last_element():
    # The last iteration must land inside the axis: extent 6, index 2*5 = 10 < 12.
    ext = _iter_extent_of(ast.parse("a[0::2]", mode="eval").body, {"a": ("12", )})
    assert ast.unparse(ast.fix_missing_locations(ext[0])) == "6"


# ---- numerical: every backend matches numpy ---- #


def test_strided_operand_of_elementwise_matches_numpy():
    src = "import numpy as np\ndef f(a, b, out):\n    out[:] = np.maximum(a[0::2], b)\n"
    _assert_ok(
        run_op(src,
               "f", {
                   "a": _A12,
                   "b": _ZERO6
               }, {"out": (6, )}, {"N": 12},
               shapes={
                   "a": "(12,)",
                   "b": "(6,)",
                   "out": "(6,)"
               },
               backends=_NATIVE))


def test_reverse_operand_of_elementwise_matches_numpy():
    src = "import numpy as np\ndef f(a, b, out):\n    out[:] = np.maximum(a[::-1], b)\n"
    _assert_ok(
        run_op(src,
               "f", {
                   "a": np.arange(1.0, 7.0),
                   "b": _ZERO6
               }, {"out": (6, )}, {"N": 6},
               shapes={
                   "a": "(6,)",
                   "b": "(6,)",
                   "out": "(6,)"
               },
               backends=_NATIVE))


def test_strided_operand_of_dot_matches_numpy():
    src = "import numpy as np\ndef f(a, b, out):\n    out[0] = np.dot(a[0::2], b)\n"
    _assert_ok(
        run_op(src,
               "f", {
                   "a": _A12,
                   "b": np.ones(6)
               }, {"out": (1, )}, {"N": 12},
               shapes={
                   "a": "(12,)",
                   "b": "(6,)",
                   "out": "(1,)"
               },
               backends=_NATIVE))


def test_strided_slice_assign_still_matches_numpy():
    # dwt2d's Haar shape: a bare strided slice ASSIGN, which lowering (not the
    # operand scalarizer) rewrites. Was always correct -- pinned against regression.
    src = ("import numpy as np\n"
           "def f(a, lo, hi):\n"
           "    lo[:] = a[0::2]\n"
           "    hi[:] = a[1::2]\n")
    _assert_ok(
        run_op(src,
               "f", {"a": _A12}, {
                   "lo": (6, ),
                   "hi": (6, )
               }, {"N": 12},
               shapes={
                   "a": "(12,)",
                   "lo": "(6,)",
                   "hi": "(6,)"
               },
               backends=_NATIVE))
