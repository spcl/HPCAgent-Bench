"""Unit tests for axis-aware reductions.

Cover:
* ``axis=None`` -- full reduction (scalar result).
* ``axis=k`` -- single integer axis (keepdims True / False, negative ``k``).
* ``axis=(k1, k2, ...)`` -- tuple form, reducing multiple axes.
* ``axis=[k1, k2]`` -- list form, same semantics as the tuple.

Each test parses the source AST, drives ``_expand_axis_reduction``
through ``expand_sum`` (a thin wrapper that supplies the addition
op_fn and 0.0 init), and inspects the resulting statement list for the
expected loop structure -- iteration count and inner ``+=`` form.

Section D covers the OPERAND side of the same reductions: an instance norm reduces over
``np.expand_dims(np.expand_dims(z, 1), 1)``, whose newaxis rewrite used to leave a chained
subscript no shape resolver could size.
"""

import ast
from typing import Dict

import numpy as np
import pytest
from _op_oracle import run_op

from numpyto_common.frontend import _AxisReshapeToIndexing
from numpyto_common.lib_nodes import _read_axis_keepdims, expand_sum

_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")


def _call_args(src: str):
    call = ast.parse(src, mode="eval").body
    return call.args, call.keywords


def _target(name: str) -> ast.Name:
    return ast.Name(id=name, ctx=ast.Store())


def _count_for_loops(stmts) -> int:
    n = 0
    for stmt in stmts:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.For):
                n += 1
    return n


# --------------------------------------------------------------------------- #
# A. ``_read_axis_keepdims`` parsing                                          #
# --------------------------------------------------------------------------- #


def test_read_axis_none_no_keepdims():
    args, kws = _call_args("np.sum(arr)")
    assert _read_axis_keepdims(args, kws) == (None, False)


def test_read_axis_int_positive():
    args, kws = _call_args("np.sum(arr, axis=2)")
    assert _read_axis_keepdims(args, kws) == ([2], False)


def test_read_axis_int_negative_unary():
    args, kws = _call_args("np.sum(arr, axis=-1)")
    assert _read_axis_keepdims(args, kws) == ([-1], False)


def test_read_axis_tuple_form():
    args, kws = _call_args("np.sum(arr, axis=(1, 2, 3))")
    assert _read_axis_keepdims(args, kws) == ([1, 2, 3], False)


def test_read_axis_list_form_with_keepdims():
    args, kws = _call_args("np.sum(arr, axis=[0, 1], keepdims=True)")
    assert _read_axis_keepdims(args, kws) == ([0, 1], True)


def test_read_axis_positional_int():
    args, kws = _call_args("np.sum(arr, 1)")
    assert _read_axis_keepdims(args, kws) == ([1], False)


def test_read_axis_positional_tuple():
    args, kws = _call_args("np.sum(arr, (0, 2))")
    assert _read_axis_keepdims(args, kws) == ([0, 2], False)


# --------------------------------------------------------------------------- #
# B. Loop structure for axis=int                                              #
# --------------------------------------------------------------------------- #


def test_sum_axis_0_emits_two_loops_for_2d():
    """``np.sum(arr, axis=0)`` with arr:(N, M) -> outer over M
    (kept axis), inner over N (reduction axis)."""
    args, kws = _call_args("np.sum(arr, axis=0)")
    stmts = expand_sum(_target("out"), args, {"arr": ("N", "M")}, kws)
    assert _count_for_loops(stmts) == 2


def test_sum_axis_1_emits_two_loops_for_3d():
    args, kws = _call_args("np.sum(arr, axis=1)")
    stmts = expand_sum(_target("out"), args, {"arr": ("N", "M", "K")}, kws)
    # 3-D - 1 reduction axis = 2 outer + 1 inner = 3 loops.
    assert _count_for_loops(stmts) == 3


# --------------------------------------------------------------------------- #
# C. Axis-tuple reductions                                                    #
# --------------------------------------------------------------------------- #


def test_sum_axis_tuple_2_of_4_emits_correct_loop_count():
    """``np.sum(arr, axis=(1, 2))`` on a 4-D array -> 2 outer kept
    axes + 2 inner reduction axes = 4 loops total."""
    args, kws = _call_args("np.sum(arr, axis=(1, 2))")
    stmts = expand_sum(_target("out"), args, {"arr": ("N", "H", "W", "C")}, kws)
    assert _count_for_loops(stmts) == 4


def test_sum_axis_tuple_3_of_4_collapses_to_one_kept_axis():
    """conv2d-style: ``np.sum(arr, axis=(1, 2, 3))`` on a 4-D array
    keeps only axis 0 -> 1 outer loop + 3 inner reduction loops."""
    args, kws = _call_args("np.sum(arr, axis=(1, 2, 3))")
    stmts = expand_sum(_target("out"), args, {"arr": ("N", "H", "W", "C")}, kws)
    assert _count_for_loops(stmts) == 4


def test_sum_axis_tuple_all_axes_reduces_to_scalar():
    """``np.sum(arr, axis=(0, 1))`` on a 2-D array reduces every axis
    and emits the same code as ``axis=None`` (just two for-loops, no
    Subscripts on the target since out is scalar)."""
    args, kws = _call_args("np.sum(arr, axis=(0, 1))")
    stmts = expand_sum(_target("out"), args, {"arr": ("N", "M")}, kws)
    assert _count_for_loops(stmts) == 2


def test_sum_axis_tuple_with_keepdims_writes_to_const_zero():
    """With keepdims=True the target subscript fills the reduced
    axes with constant 0. Structural check: a Subscript whose slice
    contains ``Constant(0)`` shows up on the LHS."""
    args, kws = _call_args("np.sum(arr, axis=(1, 2), keepdims=True)")
    stmts = expand_sum(_target("out"), args, {"arr": ("N", "H", "W", "C")}, kws)
    # Walk for any Subscript on Store side that has Constant(0) in
    # its slice -- the keepdims-zero positions.
    has_const_zero = False
    for stmt in stmts:
        for sub in ast.walk(stmt):
            if (isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Tuple)):
                for elt in sub.slice.elts:
                    if isinstance(elt, ast.Constant) and elt.value == 0:
                        has_const_zero = True
    assert has_const_zero


def test_sum_axis_list_equivalent_to_tuple():
    """``axis=[1, 2]`` parses the same as ``axis=(1, 2)``."""
    args, kws = _call_args("np.sum(arr, axis=[1, 2])")
    stmts = expand_sum(_target("out"), args, {"arr": ("N", "H", "W", "C")}, kws)
    assert _count_for_loops(stmts) == 4


def test_sum_axis_tuple_with_negative_axis():
    """``axis=(-1,)`` resolves against the operand rank."""
    args, kws = _call_args("np.sum(arr, axis=(-1,))")
    stmts = expand_sum(_target("out"), args, {"arr": ("N", "M")}, kws)
    # 2-D - 1 reduction axis = 1 outer + 1 inner = 2 loops.
    assert _count_for_loops(stmts) == 2


def test_sum_axis_tuple_rejects_duplicates():
    """``np.sum(arr, axis=(1, 1))`` is a user error -- numpy
    rejects this with ValueError; the expander raises
    NotImplementedError so the outer fallback path can take over."""
    args, kws = _call_args("np.sum(arr, axis=(1, 1))")
    with pytest.raises(NotImplementedError, match="duplicate"):
        expand_sum(_target("out"), args, {"arr": ("N", "M", "K")}, kws)


# --------------------------------------------------------------------------- #
# D. Reducing over an expand_dims / squeeze operand                           #
# --------------------------------------------------------------------------- #


def _reshape_to_index(src: str, ranks: Dict[str, int]) -> str:
    tree = _AxisReshapeToIndexing(ranks).visit(ast.parse(src, mode="eval").body)
    return ast.unparse(ast.fix_missing_locations(tree))


def test_nested_expand_dims_is_one_subscript():
    """Two ``expand_dims`` merge into ONE newaxis subscript, not ``z[:, None, :][:, None, :, :]``.

    The chain is what broke the reduction over it: ``_iter_extent_of`` sizes a subscript of a
    NAME, so a subscript of a subscript came back unsized, the reduction operand was never
    hoisted to a temp, and ``np.mean`` reached the emitter unlowered.
    """
    assert _reshape_to_index("np.expand_dims(np.expand_dims(z, axis=1), axis=1)", {"z": 2}) == "z[:, None, None, :]"


def test_nested_squeeze_is_one_subscript():
    """The undo side merges the same way: two ``squeeze`` calls index one subscript."""
    assert _reshape_to_index("np.squeeze(np.squeeze(t, axis=1), axis=1)", {"t": 4}) == "t[:, 0, 0, :]"


def test_expand_dims_of_a_partial_slice_is_left_chained():
    """A partial slice keeps an offset an outer index would drop, so it is NOT merged."""
    assert _reshape_to_index("np.expand_dims(a[1:3], axis=0)", {"a": 1}) == "a[1:3][None, :]"


def test_mean_over_nested_expand_dims():
    """``np.mean(np.expand_dims(np.expand_dims(z, 1), 1), axis=(2, 3), keepdims=True)`` --
    the instance-norm operand shape, reduced over a tuple axis."""
    z = np.linspace(-3.0, 5.0, 12).reshape(3, 4)
    src = ("import numpy as np\n"
           "def f(z, out):\n"
           "    t = np.expand_dims(np.expand_dims(z, axis=1), axis=1)\n"
           "    m = np.mean(t, axis=(2, 3), keepdims=True)\n"
           "    out[:] = np.squeeze(np.squeeze(m, axis=1), axis=1)\n")
    res = run_op(src,
                 "f", {"z": z}, {"out": (3, 1)}, {
                     "NB": 3,
                     "NC": 4
                 },
                 shapes={
                     "z": "(NB, NC)",
                     "out": "(NB, 1)"
                 },
                 backends=_ALL)
    assert all(v == "ok" or v.startswith("skip") for v in res.values()), res


def test_instance_norm_over_expanded_operand():
    """The whole idiom the ML corpus writes: mean + var over the expanded axes, then squeeze back.

    ``np.var`` shares the reduction operand path with ``np.mean``, and the division by the
    reduction count is what makes a wrong count show up as a wrong value rather than a wrong shape.
    """
    z = np.linspace(-2.0, 6.0, 12).reshape(3, 4)
    src = ("import numpy as np\n"
           "def f(z, out):\n"
           "    t = np.expand_dims(np.expand_dims(z, axis=1), axis=1)\n"
           "    m = np.mean(t, axis=(2, 3), keepdims=True)\n"
           "    v = np.var(t, axis=(2, 3), keepdims=True)\n"
           "    n = (t - m) / np.sqrt(v + 1e-05)\n"
           "    out[:] = np.squeeze(np.squeeze(n, axis=1), axis=1)\n")
    res = run_op(src,
                 "f", {"z": z}, {"out": (3, 4)}, {
                     "NB": 3,
                     "NC": 4
                 },
                 shapes={
                     "z": "(NB, NC)",
                     "out": "(NB, NC)"
                 },
                 backends=_ALL)
    assert all(v == "ok" or v.startswith("skip") for v in res.values()), res


# --------------------------------------------------------------------------- #
# E. Blocked accumulation -- a full float sum is partial sums, not one chain.   #
# --------------------------------------------------------------------------- #


def _full_sum_stmts(shape):
    args, kws = _call_args("np.sum(a)")
    return expand_sum(_target("s"), args, {"a": shape}, kwargs=kws, local_dtypes={})


def test_full_float_sum_accumulates_in_blocks():
    txt = ast.unparse(ast.fix_missing_locations(ast.Module(body=_full_sum_stmts(("N", )), type_ignores=[])))
    # A block loop over N // 128 with its own accumulator, then the leftover elements.
    assert "range(N // 128)" in txt, txt
    assert "range(128)" in txt, txt
    assert "range(N // 128 * 128, N)" in txt, txt


def test_blocked_sum_keeps_the_outer_axes_as_plain_loops():
    # Only the innermost axis is blocked -- an outer axis stays a plain nest, which is what the
    # parallelism and isopar recognisers walk.
    txt = ast.unparse(ast.fix_missing_locations(ast.Module(body=_full_sum_stmts(("M", "N")), type_ignores=[])))
    assert "range(M)" in txt, txt
    assert "range(N // 128)" in txt, txt


def test_integer_sum_is_not_blocked():
    # Integer addition is exact and associative, so blocking buys nothing and only adds code.
    args, kws = _call_args("np.sum(a)")
    stmts = expand_sum(_target("s"), args, {"a": ("N", )}, kwargs=kws, local_dtypes={"a": "int64"})
    txt = ast.unparse(ast.fix_missing_locations(ast.Module(body=stmts, type_ignores=[])))
    assert "128" not in txt, txt
    assert "range(N)" in txt, txt


def test_axis_sum_is_not_blocked():
    # Blocking is the full-reduction chain only; an axis reduction keeps its per-element loop.
    args, kws = _call_args("np.sum(a, axis=1)")
    stmts = expand_sum(_target("s"), args, {"a": ("M", "N")}, kwargs=kws, local_dtypes={})
    txt = ast.unparse(ast.fix_missing_locations(ast.Module(body=stmts, type_ignores=[])))
    assert "128" not in txt, txt


def test_blocked_sum_adds_initial_exactly_once():
    """``initial=`` seeds the WHOLE sum, not each block.

    The first version of the blocked path initialised every block accumulator to the reduction's
    ``init``, which with ``initial=`` is the caller's seed -- so the answer gained one seed per
    block, silently and only for arrays long enough to have more than one. The block accumulator
    starts at zero; the seed stays on the outer accumulator.
    """
    n = 1000
    a = np.random.default_rng(0).random(n)
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    out[0] = np.sum(a, initial=7.0)\n")
    res = run_op(src,
                 "f", {"a": a}, {"out": (1, )}, {"N": n},
                 shapes={
                     "a": "(N,)",
                     "out": "(1,)"
                 },
                 backends=("c", "cpp", "fortran"))
    assert all(v == "ok" or v.startswith("skip") for v in res.values()), res
    assert any(v == "ok" for v in res.values()), f"no backend ran it: {res}"


def test_large_fp32_sum_agrees_with_numpy_pairwise():
    """The reason blocking exists, at a tolerance a serial chain does NOT meet.

    numpy sums pairwise, so its error grows with log(n) while a naive chain's grows with n. A/B
    measured through this very harness on this seeded data, n = 2**22, emitted float32, gcc -O2
    (which does not reassociate): the emitted sum differs from numpy's by **5.2e-05** relative with
    one accumulator and **1.9e-06** blocked. ``rtol`` below sits at 1e-05, a factor ~5 on each
    side, so a regression to a single accumulator FAILS here rather than drifting silently until
    some future large-N kernel disagrees.

    ``dtypes=`` is not optional: without it the harness emits float64, whose naive error is ~1e-11
    and which therefore passes whatever the accumulation does -- a green test proving nothing.
    """
    n = 1 << 22
    a = np.random.default_rng(0).random(n, dtype=np.float32)
    src = ("import numpy as np\n"
           "def f(a, out):\n"
           "    out[0] = np.sum(a)\n")
    res = run_op(src,
                 "f", {"a": a}, {"out": (1, )}, {"N": n},
                 shapes={
                     "a": "(N,)",
                     "out": "(1,)"
                 },
                 rtol=1e-5,
                 atol=0.0,
                 dtypes={
                     "a": "float32",
                     "out": "float32"
                 },
                 backends=("c", "cpp", "fortran"))
    assert all(v == "ok" or v.startswith("skip") for v in res.values()), res
    assert any(v == "ok" for v in res.values()), f"no backend ran it: {res}"
