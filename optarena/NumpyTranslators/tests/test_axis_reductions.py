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
"""

import ast

import pytest

from numpyto_c.lib_nodes import _read_axis_keepdims, expand_sum


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
    stmts = expand_sum(_target("out"),
                       args, {"arr": ("N", "H", "W", "C")}, kws)
    assert _count_for_loops(stmts) == 4


def test_sum_axis_tuple_3_of_4_collapses_to_one_kept_axis():
    """conv2d-style: ``np.sum(arr, axis=(1, 2, 3))`` on a 4-D array
    keeps only axis 0 -> 1 outer loop + 3 inner reduction loops."""
    args, kws = _call_args("np.sum(arr, axis=(1, 2, 3))")
    stmts = expand_sum(_target("out"),
                       args, {"arr": ("N", "H", "W", "C")}, kws)
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
    stmts = expand_sum(_target("out"),
                       args, {"arr": ("N", "H", "W", "C")}, kws)
    # Walk for any Subscript on Store side that has Constant(0) in
    # its slice -- the keepdims-zero positions.
    has_const_zero = False
    for stmt in stmts:
        for sub in ast.walk(stmt):
            if (isinstance(sub, ast.Subscript)
                    and isinstance(sub.slice, ast.Tuple)):
                for elt in sub.slice.elts:
                    if isinstance(elt, ast.Constant) and elt.value == 0:
                        has_const_zero = True
    assert has_const_zero


def test_sum_axis_list_equivalent_to_tuple():
    """``axis=[1, 2]`` parses the same as ``axis=(1, 2)``."""
    args, kws = _call_args("np.sum(arr, axis=[1, 2])")
    stmts = expand_sum(_target("out"),
                       args, {"arr": ("N", "H", "W", "C")}, kws)
    assert _count_for_loops(stmts) == 4


def test_sum_axis_tuple_with_negative_axis():
    """``axis=(-1,)`` resolves against the operand rank."""
    args, kws = _call_args("np.sum(arr, axis=(-1,))")
    stmts = expand_sum(_target("out"),
                       args, {"arr": ("N", "M")}, kws)
    # 2-D - 1 reduction axis = 1 outer + 1 inner = 2 loops.
    assert _count_for_loops(stmts) == 2


def test_sum_axis_tuple_rejects_duplicates():
    """``np.sum(arr, axis=(1, 1))`` is a user error -- numpy
    rejects this with ValueError; the expander raises
    NotImplementedError so the outer fallback path can take over."""
    args, kws = _call_args("np.sum(arr, axis=(1, 1))")
    with pytest.raises(NotImplementedError, match="duplicate"):
        expand_sum(_target("out"),
                   args, {"arr": ("N", "M", "K")}, kws)
