# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""A tuple assignment is lowered to one statement per name before the DaCe emitter's shape passes.

``n, c, h, w = x.shape`` is what the helper inliner emits, and leaving it whole was the single
biggest reason a generated program was refused: each unpacked name reached the frontend as an
ordinary local, so it minted a fresh opaque symbol per use and the buffer sized from them could not
be written from ``x`` -- ``[batch_size, 3, 224, 224]`` against ``[__sym___inl6_n_0, ...]``.

The SWAP case is the one that must not be lowered naively: statements in source order would
overwrite a name before the other read it, which is a wrong answer rather than a refusal.
"""

import ast

from hpcagent_bench.translators.numpyto_c.dace_emit import SplitTupleAssign


def split(source: str) -> str:
    tree = SplitTupleAssign().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def test_a_shape_unpack_becomes_one_indexed_read_per_name() -> None:
    """The subscript spelling is what ``ShapeToSymbol`` resolves to declared extents."""
    out = split("n, c, h, w = x.shape")
    assert out.splitlines() == [
        "n = x.shape[0]",
        "c = x.shape[1]",
        "h = x.shape[2]",
        "w = x.shape[3]",
    ]


def test_a_rotation_through_a_shared_name_also_latches() -> None:
    """Any read of a bound name is enough -- ``c`` is untouched but ``a`` and ``b`` still rotate."""
    out = split("a, b, c = b, c, a").splitlines()
    assert len(out) == 6, out
    assert all(" = " in line for line in out)


def test_an_expression_reading_a_bound_name_latches_too() -> None:
    """The read need not be the whole element: ``b + 1`` reads ``b``, so ``b`` must be latched
    before ``a`` is overwritten."""
    out = split("a, b = b + 1, a * 2").splitlines()
    assert len(out) == 4, out


def test_a_nested_tuple_assignment_is_split_too() -> None:
    """The inliner emits shape unpacks inside loop bodies as readily as at the top level."""
    out = split("for i in range(4):\n    n, c = x.shape\n")
    assert "n = x.shape[0]" in out
    assert "c = x.shape[1]" in out
