# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
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

from numpyto_c.dace_emit import SplitTupleAssign


def unchanged(source: str) -> str:
    """``ast.unparse``'s own rendering of ``source`` -- what "left alone" looks like after a
    round-trip (it parenthesises a tuple right-hand side that the input spelled bare)."""
    return ast.unparse(ast.parse(source))


def split(source: str) -> str:
    tree = SplitTupleAssign().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def test_a_shape_unpack_becomes_one_indexed_read_per_name():
    """The subscript spelling is what ``_ShapeToSymbol`` resolves to declared extents."""
    out = split("n, c, h, w = x.shape")
    assert out.splitlines() == [
        "n = x.shape[0]",
        "c = x.shape[1]",
        "h = x.shape[2]",
        "w = x.shape[3]",
    ]


def test_a_plain_tuple_assignment_becomes_one_statement_per_name():
    assert split("a, b = p, q").splitlines() == ["a = p", "b = q"]


def test_a_swap_goes_through_temporaries():
    """``a, b = b, a`` in source order would assign ``a = b`` and then read the NEW a."""
    out = split("a, b = b, a").splitlines()
    assert len(out) == 4, out
    assert out[0].endswith("= b") and out[1].endswith("= a")
    first, second = out[0].split(" = ")[0], out[1].split(" = ")[0]
    assert out[2] == f"a = {first}" and out[3] == f"b = {second}"


def test_a_rotation_through_a_shared_name_also_latches():
    """Any read of a bound name is enough -- ``c`` is untouched but ``a`` and ``b`` still rotate."""
    out = split("a, b, c = b, c, a").splitlines()
    assert len(out) == 6, out
    assert all(" = " in line for line in out)


def test_an_expression_reading_a_bound_name_latches_too():
    """The read need not be the whole element: ``b + 1`` reads ``b``, so ``b`` must be latched
    before ``a`` is overwritten."""
    out = split("a, b = b + 1, a * 2").splitlines()
    assert len(out) == 4, out


def test_independent_sources_are_not_latched():
    """No name on the left is read on the right, so temporaries would be pure noise in the output."""
    assert split("a, b = c + 1, d * 2").splitlines() == ["a = c + 1", "b = d * 2"]


def test_a_mismatched_arity_is_left_alone():
    """``a, b = f()`` has no per-name spelling to produce here; a later pass must see it intact."""
    assert split("a, b = f()") == unchanged("a, b = f()")


def test_a_subscript_target_is_not_a_plain_unpack():
    """``out[0], out[1] = p, q`` writes into an array; the shape passes below must not treat those
    as names they can alias."""
    assert split("out[0], out[1] = p, q") == unchanged("out[0], out[1] = p, q")


def test_a_nested_tuple_assignment_is_split_too():
    """The inliner emits shape unpacks inside loop bodies as readily as at the top level."""
    out = split("for i in range(4):\n    n, c = x.shape\n")
    assert "n = x.shape[0]" in out and "c = x.shape[1]" in out
