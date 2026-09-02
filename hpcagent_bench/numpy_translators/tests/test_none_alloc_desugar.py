"""``_ConditionalNoneAllocRewriter``: ``X = <expr> if cond else None`` -> ``X = <expr>``.

A conditionally-allocated buffer (an array in one branch, ``None`` in the other) has no
backend representation -- the C/Fortran emitters have no ``None``. A valid kernel only
reads the buffer where it was allocated (reading the ``None`` branch would be a
``None``-index error), so unconditionally taking the allocated branch is sound. These
unit-test the rewrite in isolation plus the guard that leaves ``is None``-observed names
alone, and a numerical end-to-end check that the always-allocated form still matches numpy.
"""

import ast

import numpy as np
from _op_oracle import run_op

from numpyto_common.lowering import _ConditionalNoneAllocRewriter


def _rw(body_src: str) -> str:
    tree = ast.parse(f"def f(cond):\n    {body_src}\n")
    _ConditionalNoneAllocRewriter().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree.body[0].body[0])


def test_alloc_then_none_takes_the_alloc_branch():
    assert _rw("x = np.zeros((n,)) if cond else None") == "x = np.zeros((n,))"


def test_none_then_alloc_mirror_takes_the_alloc_branch():
    assert _rw("x = None if cond else np.ones((n,))") == "x = np.ones((n,))"


def test_non_none_ternary_is_left_alone():
    assert _rw("y = a if cond else b") == "y = a if cond else b"


def test_name_observed_via_is_none_is_not_forced():
    """When the buffer's None-ness is later observed (``x is not None``) the conditional
    is load-bearing, so the rewrite must NOT fire (else the guard would always take the
    allocated branch)."""
    tree = ast.parse(
        "def f(cond):\n    x = np.zeros((n,)) if cond else None\n    if x is not None:\n        x[0] = 1.0\n"
    )
    _ConditionalNoneAllocRewriter().visit(tree)
    assert any(isinstance(node, ast.IfExp) for node in ast.walk(tree))  # IfExp preserved


_SRC = """
import numpy as np

def condnone(a, cond, out):
    tmp = np.zeros(a.shape) if cond > 0 else None
    for i in range(a.shape[0]):
        if cond > 0:
            tmp[i] = a[i] * 2.0
    for i in range(a.shape[0]):
        if cond > 0:
            out[i] = tmp[i]
        else:
            out[i] = a[i]
"""


def test_conditional_none_alloc_matches_numpy_end_to_end():
    """The always-allocated lowering runs + matches numpy on C/C++/Fortran (the backends
    that have no ``None``); the verbatim backends run the numpy form directly."""
    a = np.arange(6, dtype=np.float64)
    res = run_op(
        _SRC,
        "condnone",
        {"a": a, "cond": 1},
        {"out": (6,)},
        {"N": 6},
        shapes={"a": "(N,)", "out": "(N,)"},
        backends=("c", "cpp", "fortran"),
    )
    for backend, status in res.items():
        assert status in ("ok",) or status.startswith("skip"), f"{backend}: {status}"
