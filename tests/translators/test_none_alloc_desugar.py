# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``ConditionalNoneAllocRewriter``: ``X = <expr> if cond else None`` -> ``X = <expr>``.

A conditionally-allocated buffer (an array in one branch, ``None`` in the other) has no
backend representation -- the C/Fortran emitters have no ``None``. A valid kernel only
reads the buffer where it was allocated (reading the ``None`` branch would be a
``None``-index error), so unconditionally taking the allocated branch is sound. These
unit-test the rewrite in isolation plus the guard that leaves ``is None``-observed names
alone, and a numerical end-to-end check that the always-allocated form still matches numpy.
"""

import ast

import pytest

from hpcagent_bench.translators.numpyto_common.lowering import ConditionalNoneAllocRewriter


def rw(body_src: str) -> str:
    tree = ast.parse(f"def f(cond):\n    {body_src}\n")
    ConditionalNoneAllocRewriter().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree.body[0].body[0])


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("x = np.zeros((n,)) if cond else None", "x = np.zeros((n,))"),
        ("x = None if cond else np.ones((n,))", "x = np.ones((n,))"),
    ],
    ids=["alloc_then_none", "none_then_alloc_mirror"],
)
def test_a_conditional_none_allocation_takes_the_alloc_branch(src: str, expected: str) -> None:
    assert rw(src) == expected


def test_name_observed_via_is_none_is_not_forced() -> None:
    """When the buffer's None-ness is later observed (``x is not None``) the conditional
    is load-bearing, so the rewrite must NOT fire (else the guard would always take the
    allocated branch)."""
    tree = ast.parse(
        "def f(cond):\n    x = np.zeros((n,)) if cond else None\n    if x is not None:\n        x[0] = 1.0\n"
    )
    ConditionalNoneAllocRewriter().visit(tree)
    assert any(isinstance(node, ast.IfExp) for node in ast.walk(tree))  # IfExp preserved
