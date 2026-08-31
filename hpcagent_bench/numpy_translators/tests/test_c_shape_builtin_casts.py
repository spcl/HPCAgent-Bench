# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Python builtins inside a shape token have to become C, not be copied through.

A kernel defends a manifest scalar with ``int(group_norm_num_groups)`` before passing it as an
extent, and that call rides into the allocation's shape. ``int`` is a TYPE in C, not a function, so
copying the token through emitted ``int(group_norm_num_groups)`` and the compiler stopped at
``expected ')' before 'group_norm_num_groups'`` -- a whole kernel lost to one token.

The cast is not a workaround for the syntax: it is the same operation. Python's ``int()`` and a C
integer cast both truncate toward zero.
"""
import ast

import pytest
from numpyto_c.emit import _c_shape_token, _render_c_shape


def render(expr: str) -> str:
    return _render_c_shape(ast.parse(expr, mode="eval"))


def test_int_call_becomes_a_cast_not_a_call() -> None:
    out = render("int(num_groups)")
    assert out == "(int64_t)(num_groups)"
    # The failure this pins is textual: an emitted `int(` is the syntax error itself.
    assert "int(num_groups)" not in out


def test_int_cast_survives_inside_a_product() -> None:
    out = render("batch_size * int(num_groups)")
    assert out == "(batch_size * (int64_t)(num_groups))"


def test_int_cast_composes_with_floor_division() -> None:
    """`//` still has to become int_floor underneath the cast, not C's `/`."""
    assert render("int(c // num_groups)") == "(int64_t)(int_floor(c, num_groups))"


def test_shape_token_entry_point_agrees() -> None:
    """The token entry point is what the allocation sites call, so pin it too."""
    assert _c_shape_token("int(num_groups)") == "(int64_t)(num_groups)"


@pytest.mark.parametrize("expr", ["int_floor(a, b)", "__npb_int_pow(a, b)"])
def test_real_c_helpers_are_still_emitted_as_calls(expr: str) -> None:
    """Only `int` is reinterpreted; a genuine C helper call must stay a call."""
    assert render(expr) == expr
