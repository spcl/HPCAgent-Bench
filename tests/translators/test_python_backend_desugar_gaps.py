# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Three desugar gaps that kept azimint_naive off the python backends, each with its own cause.

They only look like one bug because one kernel hit all three. In order: pythran refuses a ``with``
statement outright; a mask-filtered index array was ranked 0, so the scatter desugar saw no driver
axis and left ``np.add.at`` standing for dace; and the scatter's own materialisation wrapped a
SCALAR value in ``np.ascontiguousarray``, which is a 0-d array pythran cannot ``+=`` into a double.

The end-to-end gate is ``test_dace_feature_kernels_desugared[azimint_naive]`` (the pythran gate it
used to sit beside was gone by v0.3.0); these pin the rewrites themselves, since a compile
failure several template layers deep in pythran names none of them.
"""

import ast

from hpcagent_bench.translators.numpyto_common.numpy_desugar import AddAtInline, SpliceErrstate, rank_table


def apply(pass_obj: ast.NodeTransformer, src: str) -> str:
    """Run one desugar pass over ``src``'s statements, the way the pipeline drives them."""
    body: list = []
    for stmt in ast.parse(src).body:
        res = pass_obj.visit(stmt)
        if res is None:
            continue
        body.extend(res if isinstance(res, list) else [res])
    mod = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    return ast.unparse(mod)


def test_a_non_errstate_with_is_left_alone() -> None:
    """A ``with`` that owns something is not a no-op; dropping it would be a different program."""
    src = "with open(path) as fh:\n    data = fh.read()"
    assert apply(SpliceErrstate(), src) == src


def test_an_ambiguous_index_rank_reports_nothing() -> None:
    """A rank-2 index on a rank-2 base reads as 1 (boolean mask) or 3 (gather).

    The two disagree and nothing here can tell them apart, so the table must stay silent rather
    than pick one -- a wrong rank is what the mask-filter bug above was made of.
    """
    tree = ast.parse("z = A[idx]")
    ranks = rank_table(tree, {"A": 2, "idx": 2})
    assert "z" not in ranks, ranks


def test_scalar_scatter_value_is_not_wrapped_in_ascontiguousarray() -> None:
    """``np.add.at(counts, idx, 1)``: the 1 stays a scalar.

    ``np.ascontiguousarray(1)`` is a 0-d ARRAY, and pythran has no ``double += 0-d array``.
    """
    out = apply(AddAtInline({"counts": 1, "idx": 1}), "np.add.at(counts, idx, 1)")
    assert "np.ascontiguousarray(1)" not in out, out
    assert "counts[__sc0_x0[__sc0_i0]] += 1" in out, out
