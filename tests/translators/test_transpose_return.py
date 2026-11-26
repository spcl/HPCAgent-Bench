# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Returning a transposed VIEW must materialize into a fresh output buffer.

Two halves: (1) every transpose spelling desugars to the single ``np.transpose``
function form (``TransposeRewriter``); (2) a returned transpose promotes to an
output buffer with the reversed / permuted shape (``shape_from_transpose`` +
the frontend return-promotion).
"""

import ast

import pytest

from hpcagent_bench.translators.numpyto_common.ir import KernelIR
from hpcagent_bench.translators.numpyto_common.lib_nodes import expand_transpose
from hpcagent_bench.translators.numpyto_common.lowering import TransposeRewriter
from tests.translators.op_oracle import parse_source


def rewrite_(src: str, sparse: list[str] | tuple[()] = ()) -> str:
    tree = ast.parse(src)
    TransposeRewriter(set(sparse)).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# desugar: every transpose spelling -> np.transpose(...)                       #


@pytest.mark.parametrize(
    ("src", "expected"),
    [("y = x.T", "y = np.transpose(x)"), ("y = x.transpose((0, 2, 1))", "y = np.transpose(x, (0, 2, 1))")],
    ids=["dot_T", "method_with_tuple_arg"],
)
def test_transpose_spelling_desugars_to_np_transpose(src: str, expected: str) -> None:
    assert rewrite_(src) == expected


def test_sparse_transpose_not_densified() -> None:
    # a sparse matrix's ``A.T`` / ``A.transpose()`` must stay a method so the
    # SpMV hoister can flip CSR<->CSC on its own buffers.
    assert rewrite_("y = A.T @ x", sparse=["A"]) == "y = A.T @ x"
    assert rewrite_("y = A.transpose()", sparse=["A"]) == "y = A.transpose()"


# frontend: a returned transpose promotes to an output buffer                  #


def parse_(src: str, input_args: list[str], shapes: dict[str, str], syms: dict[str, int]) -> KernelIR:
    from hpcagent_bench.translators.numpyto_common.lowering import lower

    return lower(parse_source(src, "f", input_args, [], shapes, syms))


@pytest.mark.parametrize(
    ("src", "shapes", "syms", "expected"),
    [
        ("import numpy as np\ndef f(x):\n return x.T\n", {"x": "(M, N)"}, {"M": 3, "N": 4}, ("N", "M")),
        (
            "import numpy as np\ndef f(x):\n return np.transpose(x, (0, 2, 1))\n",
            {"x": "(A, B, C)"},
            {"A": 2, "B": 3, "C": 4},
            ("A", "C", "B"),
        ),
    ],
    ids=["reversed_shape", "permuted_shape"],
)
def test_return_transpose_promotes_the_permuted_shape(
    src: str, shapes: dict[str, str], syms: dict[str, int], expected: tuple[str, ...]
) -> None:
    kir = parse_(src, ["x"], shapes, syms)
    outs = [a for a in kir.arrays if a.is_output]
    assert len(outs) == 1
    assert tuple(outs[0].shape) == expected


def test_tuple_return_with_transpose_promotes_both_into_outputs() -> None:
    kir = parse_(
        "import numpy as np\ndef f(x, y):\n return x.T, y * 2\n",
        ["x", "y"],
        {"x": "(M, N)", "y": "(M, N)"},
        {"M": 3, "N": 4},
    )
    outs = {a.name: tuple(a.shape) for a in kir.arrays if a.is_output}
    assert len(outs) == 2
    assert ("N", "M") in outs.values()
    assert ("M", "N") in outs.values()


# a permutation may not write into the buffer it reads                         #


def test_transpose_into_its_own_source_is_refused() -> None:
    """``tap = np.moveaxis(tap, -1, 1)`` reaches the expander as a transpose whose target IS its
    source. The expansion is an element-by-element copy through a permuted index map, so writing it
    back into the same buffer overwrites source cells before they are read: the result is neither
    the input nor its permutation. The SSA pass normally renames the target first -- this is the
    guard for when it does not, because the failure mode is a kernel that compiles and returns
    wrong numbers rather than one that refuses.
    """
    target = ast.Name(id="tap", ctx=ast.Store())
    args = [
        ast.Name(id="tap", ctx=ast.Load()),
        ast.Tuple(elts=[ast.Constant(value=i) for i in (0, 2, 1)], ctx=ast.Load()),
    ]
    with pytest.raises(NotImplementedError, match="into itself"):
        expand_transpose(target, args, {"tap": ("A", "B", "C")})
