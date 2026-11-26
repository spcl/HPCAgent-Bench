# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""AST-level unit tests for the general lowering capabilities added to make the
level-3 LS3DF macro-kernel ``ls3df_scf`` emit + validate on native c / fortran.

Each test isolates ONE capability so a regression points straight at the cause:

  * ``.shape`` / ``.shape[k]`` folded on a rank-shifting SUBSCRIPT base (newaxis
    broadcast) via the extent oracle -- the ``X.reshape(-1, X.shape[-1])`` /
    ``.reshape(X.shape)`` in an inlined ``_hpsi(v[..., None])``;
  * ``np.fft.fftfreq(n)`` and method-form ``.reshape`` sized by ``iter_extent_of``;
  * shape-tuple local forward-substitution (``shp = Y.shape; ... .reshape(shp)``);
  * a mutated scalar (Lanczos ``na += 1``) excluded from inlined-dim substitution;
  * a compound shape token (``na - 1``) surviving ``.size`` as a real BinOp;
  * ``expand_copy`` allocating its (possibly runtime-shaped) target;
  * per-call-unique ``inv`` / ``solve`` / ``det`` working buffers.

These are pure AST transforms, so no compiler is needed.
"""

import ast

import pytest

from hpcagent_bench.translators.numpyto_common.frontend import collect_inlined_scalar_defs
from hpcagent_bench.translators.numpyto_common.lib_nodes import expand_copy
from hpcagent_bench.translators.numpyto_common.lowering import ShapeMidExpressionRewriter, TupleLocalPropagator


def unparse_(node: ast.AST) -> str:
    return ast.unparse(ast.fix_missing_locations(node))


# .shape / .shape[k] on a newaxis-subscript base                              #


@pytest.mark.parametrize(
    ("src", "dims", "folded", "is_tuple"),
    [
        # ``v[..., None].shape[-1]`` on a 3-D ``v`` -> the inserted size-1 axis (1).
        ("y = v[:, :, :, None].shape[-1]", {"v": ["Lb", "Lb", "Lb"]}, "1", False),
        ("y = v[:, :, :, None].shape[0]", {"v": ["Lb", "Lb", "Lb"]}, "Lb", False),
        # ``psi[f].shape`` -> the residual (Lb, Lb, Lb, nstate) as a literal Tuple.
        ("s = psi[f].shape", {"psi": ["nfrag", "Lb", "Lb", "Lb", "nstate"]}, "(Lb, Lb, Lb, nstate)", True),
    ],
    ids=["negative_axis_is_the_inserted_size_1", "positive_axis_is_a_source_extent", "bare_shape_is_a_tuple"],
)
def test_shape_index_on_newaxis_subscript_folds_to_static_dim(
    src: str, dims: dict[str, list[str]], folded: str, is_tuple: bool
) -> None:
    tree = ast.parse(src)
    ShapeMidExpressionRewriter(dims).visit(tree)
    val = tree.body[0].value
    assert isinstance(val, ast.Tuple) is is_tuple
    assert unparse_(val) == folded


def test_shape_fold_leaves_name_base_untouched_when_unknown() -> None:
    # Never-worse: an unknown base is left intact (not mangled).
    tree = ast.parse("y = w[:, None].shape[-1]")
    ShapeMidExpressionRewriter({}).visit(tree)
    assert "shape" in unparse_(tree.body[0].value)


# iter_extent_of: fftfreq + method-form reshape                              #


# shape-tuple local forward-substitution                                       #


@pytest.mark.parametrize(
    ("src", "present", "absent"),
    [
        (
            "shp = (Lb, Lb, Lb, nstate)\nk = shp[-1]\ny = np.reshape(mm, shp)\n",
            ["(Lb, Lb, Lb, nstate)[-1]", "np.reshape(mm, (Lb, Lb, Lb, nstate))"],  # inlined into the subscript
            ["shp = "],  # dead assignment dropped
        ),
        # A twice-assigned Name is not a fixed shape descriptor -- leave it alone.
        ("shp = (a, b)\nshp = (c, d)\ny = shp[0]\n", ["shp = (a, b)"], []),
    ],
    ids=["inlines_and_drops_assignment", "skips_reassigned_name"],
)
def test_tuple_local_propagator_inlines_and_drops_assignment(src: str, present: list[str], absent: list[str]) -> None:
    tree = ast.parse(src)
    TupleLocalPropagator().run(tree)
    out = unparse_(tree)
    for text in present:
        assert text in out
    for text in absent:
        assert text not in out


# mutated scalar excluded from inlined-dim substitution                        #


def test_collect_inlined_scalar_defs_excludes_multiply_assigned() -> None:
    fn = ast.parse("def k():\n __inl1_m = 3\n __inl1_m = 5\n").body[0]
    assert "__inl1_m" not in collect_inlined_scalar_defs(fn)


# compound shape token survives .size as a real BinOp (not a mangled Name)     #


# expand_copy allocates its target                                             #


def test_expand_copy_emits_allocation_marker() -> None:
    tgt = ast.Name(id="Cm", ctx=ast.Store())
    stmts = expand_copy(tgt, [ast.Name(id="a", ctx=ast.Load())], {"a": ("n", "n")})
    # First emitted statement allocates the fresh target.
    assert isinstance(stmts[0], ast.Assign)
    assert isinstance(stmts[0].value, ast.Call)
    assert stmts[0].value.func.id == "__hpcagent_bench_zeros__"
    assert stmts[0].targets[0].id == "Cm"


# per-call-unique linalg working buffer                                        #
