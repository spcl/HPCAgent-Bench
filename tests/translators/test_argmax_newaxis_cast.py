# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the emitter feature-gaps surfaced by the viterbi kernel
(graphical-models dwarf):

* ``np.argmax(a, axis=k)`` / ``np.argmin`` return an INDEX ARRAY (int64),
  not a scalar -- so the hoisted ``__cb`` temp is allocated + typed.
* ``A[i] = B`` with a partial-subscript LHS (a row) lowers to a per-element
  copy loop instead of a pointer store.
* ``V[:, None]`` newaxis broadcast binds the slice to the LEADING iter
  (``V[w0]``), not the trailing one -- the ``SubscriptifyNames`` rewriter
  that drives the whole-array per-element expansion.

Each test pins one rule so a regression points straight at it.
"""

import ast

import pytest

from hpcagent_bench.translators.numpyto_common.lib_nodes import CallHoister
from hpcagent_bench.translators.numpyto_common.lowering import SubscriptifyNames, WholeArrayAssignRewriter


def expr_(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


# A. ``SubscriptifyNames`` -- newaxis binds the slice to the right iter       #


def subscriptify(src: str, iters: list[str], shapes: dict[str, tuple[str, ...]]) -> str:
    return ast.unparse(SubscriptifyNames(shapes, iters).visit(expr_(src)))


@pytest.mark.parametrize(
    ("src", "iters", "shapes", "want"),
    [
        # The slice is axis 0 -> V[w0]; the None axis consumes w1 but adds no source index. (Was wrongly V[w1].)
        ("V[:, None]", ["__w0", "__w1"], {"V": ("K",)}, "V[__w0]"),
        # None consumes w0, slice -> V[w1].
        ("V[None, :]", ["__w0", "__w1"], {"V": ("K",)}, "V[__w1]"),
        ("A[:, None, :]", ["__w0", "__w1", "__w2"], {"A": ("N", "M")}, "A[__w0, __w2]"),
        # Regression guard: no newaxis -> the right-alignment is unchanged.
        ("A[:, j]", ["__w0"], {"A": ("N", "M")}, "A[__w0, j]"),
    ],
    ids=["trailing_newaxis", "leading_newaxis", "newaxis_between_axes", "plain_slice_pair"],
)
def test_a_newaxis_binds_the_slice_to_the_iterator_of_its_own_axis(
    src: str, iters: list[str], shapes: dict[str, tuple[str, ...]], want: str
) -> None:
    assert subscriptify(src, iters, shapes) == want


# B. ``np.argmax(a, axis=k)`` hoists to an int64 INDEX ARRAY                    #


def hoist(
    src: str, shapes: dict[str, tuple[str, ...]]
) -> tuple[dict[str, tuple[str, ...]], dict[str, str], dict[str, str]]:
    """Run the call-hoister over ``target = <src>`` and return
    (array_temps, scalar_temps, local_dtypes)."""
    array_temps, scalar_temps, local_dtypes = {}, {}, {}
    h = CallHoister(shapes, scalar_temps, array_temps, [0], local_dtypes)
    h.visit(expr_(src))
    return array_temps, scalar_temps, local_dtypes


@pytest.mark.parametrize(
    ("src", "shapes"),
    [("np.argmax(scores, axis=0)", {"scores": ("K", "K")}), ("np.argmin(scores, axis=1)", {"scores": ("K", "M")})],
    ids=["argmax", "argmin"],
)
def test_an_axis_argmax_or_argmin_hoists_to_an_int64_array_temp(src: str, shapes: dict[str, tuple[str, ...]]) -> None:
    arr, scal, dts = hoist(src, shapes)
    assert arr
    assert not scal
    ((name, shape),) = arr.items()
    assert shape == ("K",)  # the kept axis
    assert dts[name] == "int64"  # index dtype, not double


def test_argmax_no_axis_stays_scalar() -> None:
    # Regression guard: full argmax (axis=None) is still a scalar temp.
    arr, scal, dts = hoist("np.argmax(V)", {"V": ("K",)})
    assert scal
    assert not arr


# C. Partial-subscript LHS row copy: ``back[t] = cb``                          #


def test_partial_subscript_assign_expands_to_copy_loop() -> None:
    shapes = {"back": ("T", "K"), "cb": ("K",)}
    rw = WholeArrayAssignRewriter(shapes, real_arrays=set(shapes))
    node = ast.parse("back[t] = cb").body[0]
    out = rw.visit(node)
    out = out if isinstance(out, list) else [out]
    mod = ast.fix_missing_locations(ast.Module(body=out, type_ignores=[]))
    fors = [s for s in ast.walk(mod) if isinstance(s, ast.For)]
    assert len(fors) == 1  # one loop over the K row
    src = ast.unparse(mod)
    assert "back[t, __w0] = cb[__w0]" in src  # element-wise row copy


def test_full_subscript_assign_not_expanded() -> None:
    # Regression guard: a fully-indexed scalar store is left alone.
    shapes = {"back": ("T", "K"), "cb": ("K",)}
    rw = WholeArrayAssignRewriter(shapes, real_arrays=set(shapes))
    node = ast.parse("back[t, k] = cb[k]").body[0]
    out = rw.visit(node)
    assert not isinstance(out, list)  # untouched single Assign
