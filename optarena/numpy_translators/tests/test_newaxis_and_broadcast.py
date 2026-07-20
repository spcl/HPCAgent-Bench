"""Unit tests for newaxis insertion + broadcasting + reductions in
NumpyToC's lowering machinery.

These cover the *structural* behaviour of ``_iter_extent_of`` and
``_scalarize_at_iters`` -- the two helpers everything in
``lib_nodes`` builds on. The goal is the contract: given a source AST
and a shape table, the extent / scalarisation come out matching the
numpy semantics the user would write by hand.

Each test exercises one axis-shape combination so a regression
points straight at the failing rule.
"""

import ast

from numpyto_common.lib_nodes import _iter_extent_of, _scalarize_at_iters


def _expr(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


def _unparse_ext(ext) -> tuple:
    return tuple(ast.unparse(e) for e in ext)


def _ivar(name: str) -> ast.Name:
    return ast.Name(id=name, ctx=ast.Load())


# --------------------------------------------------------------------------- #
# A. Plain shape lookup                                                        #
# --------------------------------------------------------------------------- #


def test_bare_name_returns_full_shape():
    """``A`` with shape (N, M) -> extent (N, M)."""
    ext = _iter_extent_of(_expr("A"), {"A": ("N", "M")})
    assert _unparse_ext(ext) == ("N", "M")


def test_unknown_name_returns_none():
    assert _iter_extent_of(_expr("A"), {}) is None


# --------------------------------------------------------------------------- #
# B. Slicing                                                                   #
# --------------------------------------------------------------------------- #


def test_full_slice_yields_full_extent():
    # Zero-lower-bound is folded -- ``A[:]`` yields the raw upper bound.
    ext = _iter_extent_of(_expr("A[:]"), {"A": ("N", )})
    assert _unparse_ext(ext) == ("N", )


def test_subscript_partial_indexing_preserves_trailing_extent():
    """``A[i, :]`` on (N, M) -> extent (M,) (scalar axis 0, slice 1)."""
    ext = _iter_extent_of(_expr("A[i, :]"), {"A": ("N", "M")})
    assert _unparse_ext(ext) == ("M", )


def test_subscript_negative_upper_resolves_against_axis_len():
    """``A[1:-1]`` on (N,) -> extent (N - 1 - 1,)."""
    ext = _iter_extent_of(_expr("A[1:-1]"), {"A": ("N", )})
    # hi = N - 1, lo = 1, extent = N - 1 - 1.
    s, = ext
    assert ast.unparse(s) == "N - 1 - 1"


def test_subscript_omitted_trailing_axes_filled_in():
    """``A[1:N-1]`` on (N, M) -> extent (N-1-1, M)."""
    ext = _iter_extent_of(_expr("A[1:N - 1]"), {"A": ("N", "M")})
    assert len(ext) == 2
    assert ast.unparse(ext[0]) == "N - 1 - 1"
    assert ast.unparse(ext[1]) == "M"


# --------------------------------------------------------------------------- #
# C. ``np.newaxis`` (== ``None``) extent                                       #
# --------------------------------------------------------------------------- #


def test_newaxis_trailing_inserts_length_1():
    """``Y[:, None]`` on (N,) -> extent (N, 1). The classic
    column-vector pattern used by mandelbrot's
    ``X + Y[:, None] * 1j``."""
    ext = _iter_extent_of(_expr("Y[:, None]"), {"Y": ("N", )})
    assert _unparse_ext(ext) == ("N", "1")


def test_newaxis_leading_inserts_length_1_at_front():
    """``Y[None, :]`` on (N,) -> extent (1, N) (row-vector form)."""
    ext = _iter_extent_of(_expr("Y[None, :]"), {"Y": ("N", )})
    assert _unparse_ext(ext) == ("1", "N")


def test_newaxis_both_sides_keeps_middle_axis():
    """``A[None, :, None]`` on (N,) -> extent (1, N, 1)."""
    ext = _iter_extent_of(_expr("A[None, :, None]"), {"A": ("N", )})
    assert _unparse_ext(ext) == ("1", "N", "1")


def test_newaxis_between_existing_axes_on_2d():
    """``A[:, None, :]`` on (N, M) -> extent (N, 1, M) (the
    conv2d ``input[:, i:i+K, j:j+K, :, np.newaxis]`` shape after
    indexing the slice axes scalar)."""
    ext = _iter_extent_of(_expr("A[:, None, :]"), {"A": ("N", "M")})
    assert _unparse_ext(ext) == ("N", "1", "M")


# --------------------------------------------------------------------------- #
# D. Broadcast extents through BinOp                                           #
# --------------------------------------------------------------------------- #


def test_binop_extent_picks_wider_operand():
    """``A + b`` where A:(N, M) and b:(M,) reports A's 2-D extent."""
    ext = _iter_extent_of(_expr("A + b"), {"A": ("N", "M"), "b": ("M", )})
    assert _unparse_ext(ext) == ("N", "M")


def test_binop_extent_broadcasts_newaxis_against_vector():
    """``X + Y[:, None]`` with X:(M,) and Y[:, None]:(N, 1) ->
    extent (N, M). The trailing-1 axis on Y[:, None] stretches
    to M; the leading axis on X is implicit-1 and stretches to
    N. This is the mandelbrot per-pixel grid."""
    ext = _iter_extent_of(_expr("X + Y[:, None]"), {"X": ("M", ), "Y": ("N", )})
    assert _unparse_ext(ext) == ("N", "M")


def test_binop_broadcast_equal_rank():
    """``A + B`` with both (N, M) -> (N, M); no stretching needed."""
    ext = _iter_extent_of(_expr("A + B"), {"A": ("N", "M"), "B": ("N", "M")})
    assert _unparse_ext(ext) == ("N", "M")


def test_binop_broadcast_short_against_2d():
    """``A + b`` with A:(N, M) and b:(M,) -> (N, M)
    (rank-aligned broadcast pads b's leading axis)."""
    ext = _iter_extent_of(_expr("A + b"), {"A": ("N", "M"), "b": ("M", )})
    assert _unparse_ext(ext) == ("N", "M")


# --------------------------------------------------------------------------- #
# E. Scalarisation -- iter consumption                                         #
# --------------------------------------------------------------------------- #


def test_scalarize_bare_name_uses_single_iter():
    """``A`` with shape (N,) scalarised at iter ``i`` -> ``A[i]``."""
    out = _scalarize_at_iters(_expr("A"), [_ivar("i")], {"A": ("N", )})
    assert ast.unparse(out) == "A[i]"


def test_scalarize_2d_name_uses_tuple():
    """``A`` shape (N, M) at iters (i, j) -> ``A[i, j]``."""
    out = _scalarize_at_iters(_expr("A"), [_ivar("i"), _ivar("j")], {"A": ("N", "M")})
    assert ast.unparse(out) == "A[i, j]"


def test_scalarize_broadcasted_1d_subscripts_trailing_iter():
    """``b`` shape (M,) at iters (i, j) -> ``b[j]`` (numpy
    broadcasts a vector against the trailing axis of the iter
    nest)."""
    out = _scalarize_at_iters(_expr("b"), [_ivar("i"), _ivar("j")], {"b": ("M", )})
    assert ast.unparse(out) == "b[j]"


# --------------------------------------------------------------------------- #
# F. Scalarisation -- newaxis drops its iter                                   #
# --------------------------------------------------------------------------- #


def test_scalarize_y_newaxis_drops_inner_iter():
    """``Y[:, None]`` at iters (i, j) -> ``Y[i]`` -- the
    inserted None axis does not contribute to Y's subscript;
    the j iter is consumed but discarded."""
    out = _scalarize_at_iters(_expr("Y[:, None]"), [_ivar("i"), _ivar("j")], {"Y": ("N", )})
    assert ast.unparse(out) == "Y[i]"


def test_scalarize_y_newaxis_leading_drops_outer_iter():
    """``Y[None, :]`` at iters (i, j) -> ``Y[j]`` (leading
    None consumes i; the slice consumes j)."""
    out = _scalarize_at_iters(_expr("Y[None, :]"), [_ivar("i"), _ivar("j")], {"Y": ("N", )})
    assert ast.unparse(out) == "Y[j]"


def test_scalarize_mandelbrot_pattern():
    """``X + Y[:, None] * 1j`` at iters (i, j) ->
    ``X[j] + Y[i] * 1j`` -- the full mandelbrot per-element form."""
    out = _scalarize_at_iters(_expr("X + Y[:, None] * 1j"), [_ivar("i"), _ivar("j")], {"X": ("M", ), "Y": ("N", )})
    assert ast.unparse(out) == "X[j] + Y[i] * 1j"


def test_scalarize_conv2d_pattern_simplified():
    """Simplified conv2d-style: ``input[:, :, :, np.newaxis] *
    weights[np.newaxis, :, :, :]`` at iters (n, h, w, c_out)
    sees input as ``input[n, h, w]`` (newaxis drops c_out from
    input subscript) and weights as ``weights[h, w, c_out]``
    (leading newaxis drops n)."""
    out = _scalarize_at_iters(_expr("input[:, :, :, None] * weights[None, :, :, :]"),
                              [_ivar("n"), _ivar("h"), _ivar("w"), _ivar("c")], {
                                  "input": ("N", "H", "W"),
                                  "weights": ("H", "W", "C")
                              })
    assert ast.unparse(out) == "input[n, h, w] * weights[h, w, c]"


# --------------------------------------------------------------------------- #
# G. Reduction extents -- existing ``np.sum`` etc.                             #
# --------------------------------------------------------------------------- #
# Full reduction-call extent is handled in ``_expand_axis_reduction``;
# the extent helper itself only sees the array argument.


def test_extent_of_reduction_argument_drops_axis():
    """``np.sum(A, axis=0)`` is handled by the reduction expander;
    here we just verify that ``A`` is still pickable as a 2-D
    extent so the expander can compute the result rank."""
    ext = _iter_extent_of(_expr("A"), {"A": ("N", "M")})
    assert _unparse_ext(ext) == ("N", "M")
