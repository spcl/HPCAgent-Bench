# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scalarising an operand that carries its own broadcast reshape.

Three defects, all in the same place: how a rewriter binds an operand to the iters of the loop nest
it is being read under. Each one compiled cleanly and produced a WRONG element (or a C literal
``None``), so the assertions here are on the lowered form, not on a status code -- a numeric check
alone would have said "wrong answer" without saying which operand.

* A subscript on a COMPUTED base (``(mask != 0)[:, None]``) had the base left whole-array under a
  scalar subscript: ``(ptr != 0)[i]``, which C++ rejects outright and C compiles as a pointer read.
* A subscript operand was LEFT-aligned against the nest where numpy right-aligns, so
  ``np.where(cond4d, cxyz3d, 0)`` read ``cxyz`` at the OUTER three loops.
* An advanced index carrying newaxis reshapes (``gather_z[:, None, None]``) was aligned by its
  slice axes alone, so it read the INNERMOST iter and kept the ``None``s in the emitted subscript.
"""

import ast

import pytest

from numpyto_common.lib_nodes import _scalarize_at_iters
from numpyto_common.lowering import _SliceToScalarRewriter, _const


def _iters(n):
    return [ast.Name(id=f"__w{i}", ctx=ast.Load()) for i in range(n)]


def _scalarised(src, shapes, n):
    """``src`` rendered at an ``n``-deep nest by the np.* expanders' scalariser."""
    return ast.unparse(_scalarize_at_iters(ast.parse(src, mode="eval").body, _iters(n), shapes))


def _fused(src, shapes, n):
    """``src`` rendered at an ``n``-deep nest by the slice-fusion rewriter (the whole-array path)."""
    full = [ast.Slice(lower=None, upper=None, step=None) for _ in range(n)]
    zero = [(_const(0), _const(0)) for _ in range(n)]
    rewriter = _SliceToScalarRewriter(shapes, _iters(n), zero, "out", full)
    return ast.unparse(rewriter.visit(ast.parse(src, mode="eval").body))


def test_computed_base_is_scalarised_not_subscripted():
    # The base IS the array; the ``[:, None]`` only says which nest axis it varies along.
    got = _scalarised("(mask != 0)[:, None]", {"mask": ("np",)}, 2)
    assert got == "mask[__w0] != 0", got
    assert "None" not in got, "a literal newaxis reached the emitter"


def test_a_lower_rank_subscript_operand_right_aligns():
    # numpy broadcasts right-aligned: under a 4-deep nest a rank-3 read takes the LAST three iters.
    got = _scalarised("cxyz[:a, :b, :c]", {"cxyz": ("A", "B", "C")}, 4)
    assert got == "cxyz[__w1, __w2, __w3]", got


def test_an_equal_rank_subscript_operand_is_unchanged():
    # The offset is zero when the ranks already agree -- the arithmetic that was there before.
    got = _scalarised("cxyz[:a, :b, :c]", {"cxyz": ("A", "B", "C")}, 3)
    assert got == "cxyz[__w0, __w1, __w2]", got


@pytest.mark.parametrize(
    "src,nest,want",
    [
        ("grid[gz[:, None, None], gy[None, :, None], gx[None, None, :]]", 3, "grid[gz[__w0], gy[__w1], gx[__w2]]"),
        # Two vectors and a scalar axis: a rank-2 result, so under a 3-deep nest it right-aligns.
        ("grid[gz[:, None], gy[None, :], 0]", 2, "grid[gz[__w0], gy[__w1], 0]"),
        ("grid[gz[:, None], gy[None, :], 0]", 3, "grid[gz[__w1], gy[__w2], 0]"),
    ],
)
def test_open_mesh_gather_binds_each_vector_to_its_own_axis(src, nest, want):
    # ``A[a[:, None, None], b[None, :, None], c[None, None, :]]`` is the open mesh np.ix_ spells:
    # each vector varies along ITS OWN result axis, so each takes its own iter.
    shapes = {"grid": ("N", "N", "N"), "gz": ("nz",), "gy": ("ny",), "gx": ("nx",)}
    got = _fused(src, shapes, nest)
    assert got == want, got
    assert "None" not in got, "a literal newaxis reached the emitter"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
