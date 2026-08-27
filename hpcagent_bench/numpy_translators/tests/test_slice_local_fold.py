# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
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

from numpyto_common.frontend import _FoldSliceLocals


def _fold(src: str) -> str:
    fn = ast.parse(src).body[0]
    _FoldSliceLocals().apply(fn)
    ast.fix_missing_locations(fn)
    return ast.unparse(fn)


def test_bounded_slice_local_is_inlined_and_its_binding_dropped():
    out = _fold("def f(A, i, b, nlev):\n    top = slice(0, nlev)\n    return A[i, top, b]\n")
    assert "A[i, 0:nlev, b]" in out, out
    assert "slice(" not in out, out


def test_slice_none_becomes_a_bare_colon():
    """``slice(None)`` is the helper's default window: every level, i.e. a plain ``:``."""
    out = _fold("def f(A, i, b):\n    lvl = slice(None)\n    return A[i, lvl, b]\n")
    assert "A[i, :, b]" in out, out


def test_single_argument_slice_is_a_stop_bound():
    """``slice(n)`` is ``:n`` -- the one argument is the STOP, not the start."""
    out = _fold("def f(x, n):\n    w = slice(n)\n    return x[w]\n")
    assert "x[:n]" in out, out


def test_step_is_carried():
    out = _fold("def f(x, a, b, s):\n    w = slice(a, b, s)\n    return x[w]\n")
    assert "x[a:b:s]" in out, out


def test_the_same_window_bound_twice_still_folds():
    """velocity_tendencies rebinds ``rest = slice(1, nlev)`` in two scopes, identically."""
    src = ("def f(x, nlev, flag):\n"
           "    if flag:\n"
           "        rest = slice(1, nlev)\n"
           "        a = x[rest]\n"
           "    else:\n"
           "        rest = slice(1, nlev)\n"
           "        a = x[rest]\n"
           "    return a\n")
    out = _fold(src)
    assert out.count("x[1:nlev]") == 2, out
    assert "slice(" not in out, out


def test_two_different_windows_on_one_name_are_left_alone():
    """Which window a use sees depends on the binding live at that point, which this pass cannot
    see -- so it declines rather than pick one."""
    src = ("def f(x, nlev, flag):\n"
           "    w = slice(0, nlev)\n"
           "    if flag:\n"
           "        w = slice(1, nlev)\n"
           "    return x[w]\n")
    out = _fold(src)
    assert "slice(0, nlev)" in out and "slice(1, nlev)" in out, out


def test_a_name_used_outside_an_index_keeps_its_binding():
    """Only index slots are rewritten; a slice passed on as a value still needs the object."""
    out = _fold("def f(x, nlev, g):\n    w = slice(0, nlev)\n    y = g(w)\n    return x[w] + y\n")
    assert "w = slice(0, nlev)" in out, out
    assert "x[0:nlev]" in out, out


@pytest.mark.parametrize("expr", ["A[top]", "A[top, b]", "A[i, top, b]"])
def test_folds_in_every_index_position(expr):
    out = _fold(f"def f(A, i, b, nlev):\n    top = slice(0, nlev)\n    return {expr}\n")
    assert "0:nlev" in out, out
