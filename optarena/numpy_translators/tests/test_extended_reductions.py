"""Unit tests for ``np.var / std / argmax / argmin / any / all /
count_nonzero / linalg.norm`` expanders with axis = None / int /
tuple semantics + keepdims.

The contract under test is the LOOP STRUCTURE produced by each
expander -- iteration count and per-axis subscript form. Numerical
correctness is checked separately by the compile-and-run sweep.
"""

import ast

import pytest

from numpyto_c.lib_nodes import (
    expand_var, expand_any, expand_all, expand_count_nonzero,
    expand_argmax, expand_argmin, expand_linalg_norm,
)


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


# ---------------------------------------------------------------------------
# A. np.var                                                                   #
# ---------------------------------------------------------------------------

def test_var_axis_none_two_dim():
    """``np.var(A)`` on A:(N, M) -> walk every axis, scalar result."""
    args, kws = _call_args("np.var(A)")
    stmts = expand_var(_target("out"), args, {"A": ("N", "M")}, kws)
    # 2 (mean) + 2 (sum-of-squared-dev) = 4 loops total.
    assert _count_for_loops(stmts) == 4


def test_var_axis_tuple_of_three_on_four_dim():
    """conv2d-style: ``np.var(arr, axis=(1, 2, 3))`` on (N, H, W, C) ->
    keeps axis 0, reduces inner three. 1 outer + 3 inner = 4 loops
    for mean, 1 outer + 3 inner = 4 loops for the squared-dev pass.
    Total = 8."""
    args, kws = _call_args("np.var(A, axis=(1, 2, 3))")
    stmts = expand_var(_target("out"),
                       args, {"A": ("N", "H", "W", "C")}, kws)
    assert _count_for_loops(stmts) == 8


def test_var_keepdims_writes_const_zero_on_reduced_axis():
    args, kws = _call_args("np.var(A, axis=1, keepdims=True)")
    stmts = expand_var(_target("out"),
                       args, {"A": ("N", "M", "K")}, kws)
    seen_const_zero = False
    for stmt in stmts:
        for sub in ast.walk(stmt):
            if (isinstance(sub, ast.Subscript)
                    and isinstance(sub.slice, ast.Tuple)):
                for elt in sub.slice.elts:
                    if isinstance(elt, ast.Constant) and elt.value == 0:
                        seen_const_zero = True
    assert seen_const_zero


# ---------------------------------------------------------------------------
# B. np.any / np.all                                                          #
# ---------------------------------------------------------------------------

def test_any_axis_int_one_dim_result():
    args, kws = _call_args("np.any(A, axis=0)")
    stmts = expand_any(_target("out"), args, {"A": ("N", "M")}, kws)
    # 1 outer (kept axis M) + 1 inner (reduction axis N) = 2 loops.
    assert _count_for_loops(stmts) == 2


def test_all_axis_tuple_collapses_to_scalar():
    args, kws = _call_args("np.all(A, axis=(0, 1))")
    stmts = expand_all(_target("out"), args, {"A": ("N", "M")}, kws)
    # All axes reduced, no kept axes -> 2 inner loops.
    assert _count_for_loops(stmts) == 2


def test_count_nonzero_axis_negative():
    """``np.count_nonzero(A, axis=-1)`` on (N, M, K) -> kept axes 0/1.
    1 + 1 = 2 outer + 1 inner = 3 loops."""
    args, kws = _call_args("np.count_nonzero(A, axis=-1)")
    stmts = expand_count_nonzero(_target("out"),
                                  args, {"A": ("N", "M", "K")}, kws)
    assert _count_for_loops(stmts) == 3


# ---------------------------------------------------------------------------
# C. np.argmax / np.argmin                                                    #
# ---------------------------------------------------------------------------

def test_argmax_axis_none_full_reduction():
    """``np.argmax(A)`` -> flat argmax. Walks every axis."""
    args, kws = _call_args("np.argmax(A)")
    stmts = expand_argmax(_target("out"), args, {"A": ("N", "M")}, kws)
    assert _count_for_loops(stmts) == 2


def test_argmax_axis_int_two_dim():
    """``np.argmax(A, axis=0)`` -> 1 outer (kept) + 1 inner (reduction)."""
    args, kws = _call_args("np.argmax(A, axis=0)")
    stmts = expand_argmax(_target("out"), args, {"A": ("N", "M")}, kws)
    assert _count_for_loops(stmts) == 2


def test_argmin_axis_int_three_dim():
    """``np.argmin(A, axis=1)`` on (N, M, K) -> 2 outer + 1 inner."""
    args, kws = _call_args("np.argmin(A, axis=1)")
    stmts = expand_argmin(_target("out"),
                          args, {"A": ("N", "M", "K")}, kws)
    assert _count_for_loops(stmts) == 3


def test_argmax_axis_tuple_two_axes_on_three_dim():
    """``np.argmax(A, axis=(0, 1))`` on (N, M, K) -> keeps axis 2;
    per-K position writes the FLAT index across (N, M). Loop count:
    1 outer (K) + 2 inner = 3."""
    args, kws = _call_args("np.argmax(A, axis=(0, 1))")
    stmts = expand_argmax(_target("out"),
                          args, {"A": ("N", "M", "K")}, kws)
    assert _count_for_loops(stmts) == 3


def test_argmin_axis_tuple_three_of_four():
    """``np.argmin(A, axis=(1, 2, 3))`` on (N, H, W, C) -> keeps axis 0;
    1 outer + 3 inner = 4 loops."""
    args, kws = _call_args("np.argmin(A, axis=(1, 2, 3))")
    stmts = expand_argmin(_target("out"),
                          args, {"A": ("N", "H", "W", "C")}, kws)
    assert _count_for_loops(stmts) == 4


def test_argmax_axis_tuple_duplicate_rejected():
    """``np.argmax(A, axis=(0, 0))`` is a user error -- numpy itself
    raises ValueError on duplicate axes."""
    args, kws = _call_args("np.argmax(A, axis=(0, 0))")
    with pytest.raises(NotImplementedError, match="duplicate"):
        expand_argmax(_target("out"), args, {"A": ("N", "M")}, kws)


# ---------------------------------------------------------------------------
# D. np.linalg.norm (full axis + keepdims)                                    #
# ---------------------------------------------------------------------------

def test_linalg_norm_full_reduction():
    """Default form -- L2 norm over every axis -> scalar."""
    args, kws = _call_args("np.linalg.norm(A)")
    stmts = expand_linalg_norm(_target("out"),
                                args, {"A": ("N", "M")}, kws)
    # 2 loops for the squared-sum + a finalisation sqrt assignment.
    assert _count_for_loops(stmts) == 2


def test_linalg_norm_axis_int_two_dim():
    """``np.linalg.norm(A, axis=0)`` -> per-column L2."""
    args, kws = _call_args("np.linalg.norm(A, axis=0)")
    stmts = expand_linalg_norm(_target("out"),
                                args, {"A": ("N", "M")}, kws)
    # 1 outer (kept axis M) + 1 inner (reduction axis N) = 2.
    assert _count_for_loops(stmts) == 2


def test_linalg_norm_axis_tuple_3_of_4():
    """``np.linalg.norm(A, axis=(1, 2, 3))`` on (N, H, W, C) -> 1
    outer + 3 inner = 4 loops."""
    args, kws = _call_args("np.linalg.norm(A, axis=(1, 2, 3))")
    stmts = expand_linalg_norm(_target("out"),
                                args, {"A": ("N", "H", "W", "C")}, kws)
    assert _count_for_loops(stmts) == 4


def test_linalg_norm_keepdims_true():
    args, kws = _call_args("np.linalg.norm(A, axis=0, keepdims=True)")
    stmts = expand_linalg_norm(_target("out"),
                                args, {"A": ("N", "M")}, kws)
    # Same loop count -- keepdims only affects the target subscript.
    assert _count_for_loops(stmts) == 2
    seen_const_zero = False
    for stmt in stmts:
        for sub in ast.walk(stmt):
            if (isinstance(sub, ast.Subscript)
                    and isinstance(sub.slice, ast.Tuple)):
                for elt in sub.slice.elts:
                    if isinstance(elt, ast.Constant) and elt.value == 0:
                        seen_const_zero = True
    assert seen_const_zero


def test_linalg_norm_rejects_unsupported_ord():
    """Supported now: the default 2-norm, ``ord=1`` (sum|v|) and ``ord=inf`` (max|v|).
    An arbitrary p-norm (``ord=3``) has no closed-form elementwise lowering, so it must
    still raise (callers do it by hand)."""
    args, kws = _call_args("np.linalg.norm(A, ord=3)")
    with pytest.raises(NotImplementedError):
        expand_linalg_norm(_target("out"), args, {"A": ("N",)}, kws)
