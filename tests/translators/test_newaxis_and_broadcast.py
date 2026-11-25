# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for newaxis insertion + broadcasting + reductions in
NumpyToC's lowering machinery.

These cover the *structural* behaviour of ``iter_extent_of`` and
``scalarize_at_iters`` -- the two helpers everything in
``lib_nodes`` builds on. The goal is the contract: given a source AST
and a shape table, the extent / scalarisation come out matching the
numpy semantics the user would write by hand.

Each test exercises one axis-shape combination so a regression
points straight at the failing rule.
"""

import ast

import pytest

from hpcagent_bench.translators.numpyto_common.lib_nodes import iter_extent_of, scalarize_at_iters


def expr_(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


def unparse_ext(ext: tuple[ast.Constant | ast.Name, ...]) -> tuple:
    return tuple(ast.unparse(e) for e in ext)


def ivar(name: str) -> ast.Name:
    return ast.Name(id=name, ctx=ast.Load())


# A. Plain shape lookup                                                        #


# B-C. Extents: slicing, ``np.newaxis`` (== ``None``) and chained indexing                #

GATHER_SHAPES = {
    "x": ("n_atoms", "3"),
    "aj": ("pairs", "UNROLLJ"),
    "nbfp": ("nt", "nt", "2"),
    "type_i": ("UNROLLI",),
    "type_j": ("pairs", "UNROLLJ"),
    "psi": ("F", "X", "Y", "K"),
    "table": ("n",),
    "ri": ("pairs", "UNROLLI", "UNROLLJ"),
}


@pytest.mark.parametrize(
    ("src", "shapes", "extent"),
    [
        # Zero-lower-bound is folded: ``A[:]`` yields the raw upper bound.
        ("A[:]", {"A": ("N",)}, ("N",)),
        # Scalar axis 0, slice axis 1.
        ("A[i, :]", {"A": ("N", "M")}, ("M",)),
        # hi = N - 1, lo = 1, extent = N - 1 - 1.
        ("A[1:-1]", {"A": ("N",)}, ("N - 1 - 1",)),
        # Row-vector form.
        ("Y[None, :]", {"Y": ("N",)}, ("1", "N")),
        ("A[None, :, None]", {"A": ("N",)}, ("1", "N", "1")),
        # The conv2d ``input[:, i:i+K, j:j+K, :, np.newaxis]`` shape after indexing the slice axes scalar.
        ("A[:, None, :]", {"A": ("N", "M")}, ("N", "1", "M")),
        # The scalar-chain case the gather rule must not break: ``psi[f]`` drops axis 0.
        ("psi[f][..., 0]", GATHER_SHAPES, ("X", "Y")),
    ],
    ids=[
        "full_slice_yields_full_extent",
        "partial_indexing_preserves_trailing_extent",
        "negative_upper_resolves_against_axis_len",
        "newaxis_leading_inserts_length_1_at_front",
        "newaxis_both_sides_keeps_middle_axis",
        "newaxis_between_existing_axes_on_2d",
        "chained_scalar_index_still_drops_its_axis",
    ],
)
def test_iteration_extent_follows_numpy_slicing_and_newaxis(
    src: str, shapes: dict[str, tuple[str, ...]], extent: tuple[str, ...]
) -> None:
    assert unparse_ext(iter_extent_of(expr_(src), shapes)) == extent


# E-F. Scalarisation: iter consumption, and newaxis dropping its iter                       #


@pytest.mark.parametrize(
    ("src", "iters", "shapes", "scalarized"),
    [
        # A vector broadcasts against the trailing axis of the iter nest.
        ("b", ["i", "j"], {"b": ("M",)}, "b[j]"),
        # The leading None consumes i; the slice consumes j.
        ("Y[None, :]", ["i", "j"], {"Y": ("N",)}, "Y[j]"),
        # Simplified conv2d: newaxis drops c_out from input's subscript, the leading newaxis drops n from weights.
        (
            "input[:, :, :, None] * weights[None, :, :, :]",
            ["n", "h", "w", "c"],
            {"input": ("N", "H", "W"), "weights": ("H", "W", "C")},
            "input[n, h, w] * weights[h, w, c]",
        ),
    ],
    ids=["broadcast_1d_takes_trailing_iter", "leading_newaxis_drops_outer_iter", "conv2d_pattern_simplified"],
)
def test_scalarization_consumes_one_iter_per_axis(
    src: str, iters: list[str], shapes: dict[str, tuple[str, ...]], scalarized: str
) -> None:
    out = scalarize_at_iters(expr_(src), [ivar(i) for i in iters], shapes)
    assert ast.unparse(out) == scalarized
