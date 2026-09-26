# Copyright 2025 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Two ways a compile-time rank was lost, and the group-norm axis fold that needed it back.

``np.mean(y, axis=tuple(range(2, y.ndim)))`` is how every KernelBench port spells "every axis from
here on". It only folds if ``y``'s rank is known at that point, and two separate gaps meant it was
not:

* ``tuple_of`` recognised a tuple CONCAT and ``x.shape``, but not a SLICE of one, so
  ``y = x.reshape((n, g, c // g) + x.shape[2:])`` never collapsed and ``y`` had no rank.
* ``track_rank`` dropped a name's rank outright for any rebinding inside a branch or a loop.
  Max-pool allocates ``out`` rank 4 and then rebinds ``out = np.maximum(out, ...)`` inside the tap
  loops -- a rebinding that provably preserves the rank -- and every statement after it lost the
  rank, including two inlined helpers later.

Both surface far from the cause, as ``axis must be a compile-time integer``. The conservative
direction is asserted too: a non-linear rebinding that CHANGES the rank must still drop it, because
after the loop the name holds one or the other and nothing here knows which.
"""

import ast

import pytest

from hpcagent_bench.translators.numpyto_common.numpy_desugar import rank_table
from hpcagent_bench.translators.numpyto_common.tuple_desugar import Env, TupleDesugar, desugar_tuples


def fold(
    src: str, ranks: dict, int_scalars: frozenset[str] = frozenset(), arrays: frozenset[str] = frozenset()
) -> TupleDesugar:
    """Run the interpreter over ``src``'s single function and return it, for its rank table."""
    fn = ast.parse(src).body[0]
    interp = TupleDesugar(int_scalars, frozenset(), arrays, ranks)
    fn.body = interp.run(fn.body, Env(bound=set(int_scalars) | set(arrays)), linear=True)
    interp.folded = fn
    return interp


def source_of(interp: TupleDesugar) -> str:
    return ast.unparse(interp.folded)


def test_an_allocation_sized_by_a_scalar_expression_is_one_dimensional() -> None:
    """cp2k_density_matrix_trs4 sizes its index arrays ``np.zeros(n_block_rows * block_size)``; an unranked
    index array kept the desugar from spelling ``x[rows, cols, cols] -= s`` as a loop dace accepts."""
    fn = ast.parse("def f(n, b):\n    d = np.zeros(n * b, dtype=np.int64)\n    e = np.zeros(n, dtype=np.int64)\n")
    ranks = rank_table(fn.body[0], {"n": 0, "b": 0})
    assert (ranks.get("d"), ranks.get("e")) == (1, 1), ranks


def test_a_slice_of_the_shape_tuple_is_still_compile_time() -> None:
    """``x.shape[2:]`` -- the "trailing axes" idiom. The elements stay symbolic; only the LENGTH
    has to be compile-time, which is what the concat and the following ``ndim`` need."""
    interp = fold(
        "def f(x, g):\n    n, c = x.shape[0], x.shape[1]\n    y = x.reshape((n, g, c // g) + x.shape[2:])\n",
        ranks={"x": 4},
        int_scalars=frozenset({"g", "n", "c"}),
        arrays=frozenset({"x"}),
    )
    assert "x.reshape((n, g, c // g, x.shape[2], x.shape[3]))" in source_of(interp)
    assert interp.ranks["y"] == 5


def test_the_group_norm_axis_folds_once_the_reshape_collapses() -> None:
    interp = fold(
        "def f(x, g):\n"
        "    n, c = x.shape[0], x.shape[1]\n"
        "    y = x.reshape((n, g, c // g) + x.shape[2:])\n"
        "    m = np.mean(y, axis=tuple(range(2, y.ndim)), keepdims=True)\n",
        ranks={"x": 4},
        int_scalars=frozenset({"g", "n", "c"}),
        arrays=frozenset({"x"}),
    )
    assert "axis=(2, 3, 4)" in source_of(interp)


def test_a_negative_shape_slice_folds_too() -> None:
    """``x.shape[:-1]`` is the same idiom from the other end; the bound must fold, not the extents."""
    interp = fold("def f(x):\n    y = x.reshape(x.shape[:-1] + (1,))\n", ranks={"x": 3}, arrays=frozenset({"x"}))
    assert "x.reshape((x.shape[0], x.shape[1], 1))" in source_of(interp)
    assert interp.ranks["y"] == 3


def test_a_symbolic_slice_bound_declines() -> None:
    """The LENGTH is what must be compile-time. A bound that is a runtime scalar leaves the tuple
    alone rather than guessing a length."""
    interp = fold(
        "def f(x, k):\n    y = x.reshape(x.shape[k:] + (1,))\n",
        ranks={"x": 3},
        int_scalars=frozenset({"k"}),
        arrays=frozenset({"x"}),
    )
    # ``x.shape`` still expands to its per-axis tuple -- that part is rank-3 knowledge, not a guess.
    # The SLICE is what must survive: a ``[k:]`` left standing is the decline, and any fixed-length
    # tuple in its place would be a length invented from a runtime scalar.
    src = source_of(interp)
    assert "(x.shape[0], x.shape[1], x.shape[2])[k:]" in src, src
    assert "y" not in interp.ranks


def test_a_rank_preserving_rebind_inside_a_loop_keeps_the_rank() -> None:
    """The max-pool accumulator. ``out`` is rank 4 before the loop and rank 4 after every iteration,
    so it is rank 4 afterwards whether the loop ran or not."""
    interp = fold(
        "def f(x, ks):\n"
        "    out = np.full((1, 2, 3, 4), 0.0)\n"
        "    for k in range(ks):\n"
        "        out = np.maximum(out, x)\n"
        "    y = out.reshape(out.shape[1:])\n",
        ranks={"x": 4},
        int_scalars=frozenset({"ks"}),
        arrays=frozenset({"x"}),
    )
    assert interp.ranks["out"] == 4
    assert "out.reshape((out.shape[1], out.shape[2], out.shape[3]))" in source_of(interp)


def test_a_rank_changing_rebind_inside_a_loop_drops_the_rank() -> None:
    """The conservative half: after the loop ``out`` is rank 4 or rank 1 depending on a trip count
    nothing here can evaluate, so it has no compile-time rank. Reporting either would be a guess."""
    interp = fold(
        "def f(x, ks):\n"
        "    out = np.full((1, 2, 3, 4), 0.0)\n"
        "    for k in range(ks):\n"
        "        out = np.full((5,), 0.0)\n",
        ranks={"x": 4},
        int_scalars=frozenset({"ks"}),
        arrays=frozenset({"x"}),
    )
    assert "out" not in interp.ranks


def test_desugar_tuples_entry_point_agrees() -> None:
    """The module entry point, not just the interpreter -- it is what the frontend calls."""
    fn = ast.parse(
        "def f(x, g):\n"
        "    n, c = x.shape[0], x.shape[1]\n"
        "    y = x.reshape((n, g, c // g) + x.shape[2:])\n"
        "    m = np.mean(y, axis=tuple(range(2, y.ndim)), keepdims=True)\n"
    ).body[0]
    desugar_tuples(fn, int_scalars=frozenset({"g", "n", "c"}), arrays=frozenset({"x"}), ranks={"x": 4})
    assert "axis=(2, 3, 4)" in ast.unparse(fn)


@pytest.mark.parametrize(
    ("src", "name", "want"),
    [
        ("def k(p, f):\n    shp = p[f].shape\n    X = (p[f] * 2.0).reshape(shp)\n", "X", 3),
        (
            "def k(p, m):\n    Y = p * 1.0\n    for i in range(m):\n        shp = Y.shape\n"
            "        Y = (Y * 2.0).reshape(shp)\n",
            "Y",
            4,
        ),
        ("def k(p, n):\n    X = p.reshape(n)\n", "X", 1),
    ],
    ids=["shape of a subscript", "loop-carried shape", "scalar length"],
)
def test_a_reshape_to_a_shape_local_takes_the_shapes_rank(src: str, name: str, want: int) -> None:
    """ls3df_scf reshapes to ``shp = Y.shape`` while ``Y``'s rank still depends on that reshape.
    Counting the tuple-valued name as one dimension closed the cycle at rank 1; a scalar length
    really is one dimension."""
    ranks = rank_table(ast.parse(src), {"p": 4})
    assert ranks.get(name) == want, ranks


def test_a_last_axis_read_after_a_reshape_to_a_subscripts_shape_is_the_last_axis() -> None:
    """The extent the rank feeds: ls3df_scf's inlined ``X.reshape(-1, X.shape[-1])`` sits in the fragment
    loop, where the tuple pass reads ``X``'s rank off the table; rank 1 spelled it ``X.shape[0]``, and the
    flattened block had ``Lb`` columns instead of ``nstate``."""
    fn = ast.parse(
        "def k(p, m):\n"
        "    for f in range(m):\n"
        "        shp = p[f].shape\n"
        "        X = (p[f] * 2.0).reshape(shp)\n"
        "        flat = X.reshape(-1, X.shape[-1])\n"
    ).body[0]
    desugar_tuples(
        fn,
        int_scalars=frozenset({"m"}),
        float_scalars=frozenset(),
        arrays=frozenset({"p"}),
        ranks=rank_table(fn, {"p": 4}),
    )
    got = ast.unparse(fn)
    assert "X.shape[0]" not in got and ("X.shape[-1]" in got or "X.shape[2]" in got), got
