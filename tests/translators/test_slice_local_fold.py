# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""A local bound to a ``slice(...)`` object is inlined into the subscripts that use it.

ICON's velocity_tendencies names its level windows -- ``top = slice(0, nlev)``, ``rest =
slice(1, nlev)`` -- and indexes with them. No backend has a slice OBJECT, and worse, a Name sitting
in an index slot reads as a SCALAR index: the sizer dropped that axis, so a rank-3 gather was
recorded rank 2 and the shape derived through it disagreed with the buffer allocated for the same
variable. That surfaced as a re-binding refusal many statements later, naming a variable that was
never the problem -- which is why these assert the rewrite itself, not just that the kernel lowers.
"""

import ast

import pytest

from hpcagent_bench.translators.numpyto_common.frontend import FoldSliceLocals


def fold_(src: str) -> str:
    fn = ast.parse(src).body[0]
    FoldSliceLocals().apply(fn)
    ast.fix_missing_locations(fn)
    return ast.unparse(fn)


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        # slice(None) is the helper's default window: every level, i.e. a plain ``:``.
        ("def f(A, i, b):\n    lvl = slice(None)\n    return A[i, lvl, b]\n", "A[i, :, b]"),
        # slice(n) is ``:n`` -- the one argument is the STOP, not the start.
        ("def f(x, n):\n    w = slice(n)\n    return x[w]\n", "x[:n]"),
    ],
    ids=["slice_none_is_a_bare_colon", "single_argument_is_a_stop_bound"],
)
def test_short_slice_spellings_fold_to_their_subscript(src: str, expected: str) -> None:
    out = fold_(src)
    assert expected in out, out


def test_two_different_windows_on_one_name_are_left_alone() -> None:
    """Which window a use sees depends on the binding live at that point, which this pass cannot
    see -- so it declines rather than pick one."""
    src = "def f(x, nlev, flag):\n    w = slice(0, nlev)\n    if flag:\n        w = slice(1, nlev)\n    return x[w]\n"
    out = fold_(src)
    assert "slice(0, nlev)" in out, out
    assert "slice(1, nlev)" in out, out


@pytest.mark.parametrize("expr", ["A[top]", "A[top, b]", "A[i, top, b]"])
def test_folds_in_every_index_position(expr: str) -> None:
    out = fold_(f"def f(A, i, b, nlev):\n    top = slice(0, nlev)\n    return {expr}\n")
    assert "0:nlev" in out, out
