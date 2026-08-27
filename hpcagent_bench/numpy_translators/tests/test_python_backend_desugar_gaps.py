# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Three desugar gaps that kept azimint_naive off the python backends, each with its own cause.

They only look like one bug because one kernel hit all three. In order: pythran refuses a ``with``
statement outright; a mask-filtered index array was ranked 0, so the scatter desugar saw no driver
axis and left ``np.add.at`` standing for dace; and the scatter's own materialisation wrapped a
SCALAR value in ``np.ascontiguousarray``, which is a 0-d array pythran cannot ``+=`` into a double.

The end-to-end gates are ``test_cholesky2_contour_pythran_e2e[azimint_naive]`` and
``test_dace_feature_kernels_desugared[azimint_naive]``; these pin the rewrites themselves, since a
compile failure several template layers deep in pythran names none of them.
"""
import ast

from numpyto_common.numpy_desugar import _AddAtInline, _SpliceErrstate, expr_rank, rank_table


def _apply(pass_obj, src: str) -> str:
    """Run one desugar pass over ``src``'s statements, the way the pipeline drives them."""
    body: list = []
    for stmt in ast.parse(src).body:
        res = pass_obj.visit(stmt)
        if res is None:
            continue
        body.extend(res if isinstance(res, list) else [res])
    mod = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    return ast.unparse(mod)


def test_errstate_body_is_spliced_out_of_the_with():
    """``errstate`` changes what numpy REPORTS, never what it computes, so the body stands alone."""
    out = _apply(_SpliceErrstate(), "with np.errstate(invalid='ignore'):\n    res[:] = a / b\n")
    assert out == "res[:] = a / b"


def test_a_non_errstate_with_is_left_alone():
    """A ``with`` that owns something is not a no-op; dropping it would be a different program."""
    src = "with open(path) as fh:\n    data = fh.read()"
    assert _apply(_SpliceErrstate(), src) == src


def test_mask_filtered_index_keeps_rank_one():
    """``bin_id = bin_id[valid]`` is a mask filter: rank 1, not the rank 0 a scalar index gives.

    Rank 0 made ``_AddAtInline`` find no driver axis, so ``np.add.at`` survived into the dace
    program, which cannot trace it.
    """
    tree = ast.parse("bin_id2 = bin_id[valid]")
    ranks = rank_table(tree, {"bin_id": 1, "valid": 1})
    assert ranks["bin_id2"] == 1, ranks


def test_a_scalar_name_index_still_drops_an_axis():
    """The common case must not regress: a loop iterator indexing a row drops one axis."""
    tree = ast.parse("row = A[i]")
    ranks = rank_table(tree, {"A": 2, "i": 0})
    assert ranks["row"] == 1, ranks


def test_an_ambiguous_index_rank_reports_nothing():
    """A rank-2 index on a rank-2 base reads as 1 (boolean mask) or 3 (gather).

    The two disagree and nothing here can tell them apart, so the table must stay silent rather
    than pick one -- a wrong rank is what the mask-filter bug above was made of.
    """
    tree = ast.parse("z = A[idx]")
    ranks = rank_table(tree, {"A": 2, "idx": 2})
    assert "z" not in ranks, ranks


def test_scalar_scatter_value_is_not_wrapped_in_ascontiguousarray():
    """``np.add.at(counts, idx, 1)``: the 1 stays a scalar.

    ``np.ascontiguousarray(1)`` is a 0-d ARRAY, and pythran has no ``double += 0-d array``.
    """
    out = _apply(_AddAtInline({"counts": 1, "idx": 1}), "np.add.at(counts, idx, 1)")
    assert "np.ascontiguousarray(1)" not in out, out
    assert "counts[__sc0_x0[__sc0_i0]] += 1" in out, out


def test_an_array_scatter_value_is_still_materialized():
    """The materialisation exists for a lazy numpy_expr operand -- an array value keeps it."""
    out = _apply(_AddAtInline({"sums": 1, "idx": 1, "vals": 1}), "np.add.at(sums, idx, vals)")
    assert "__sc0_v = np.ascontiguousarray(vals)" in out, out
