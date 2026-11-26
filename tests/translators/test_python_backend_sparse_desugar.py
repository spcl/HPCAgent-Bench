# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""The CSR row-index idiom, lowered for the python backends (numba / dace).

``row_index = np.repeat(np.arange(M), np.diff(A_indptr))`` then a weighted ``np.bincount`` is how
the sparse references spell a matvec. None of the three ops exists in those backends: dace routes
each to a Python callback, which drags the whole program back into the interpreter and then refuses
the nested call outright. Each is replaced by the loop that defines it.

The end-to-end gate is ``test_sparse_oracle.py::test_sparse_kernel_dace_matches_scipy[spmv]``;
these pin the rewrites, which a callback warning several frames deep does not name.
"""

import ast

import pytest

from hpcagent_bench.translators.numpyto_common.numpy_desugar import (
    BincountInline,
    DiffToSliceDifference,
    RepeatCountsInline,
    StripAstypeCopyKwarg,
)


def apply(pass_obj: ast.NodeTransformer, src: str) -> str:
    body: list = []
    for stmt in ast.parse(src).body:
        res = pass_obj.visit(stmt)
        if res is None:
            continue
        body.extend(res if isinstance(res, list) else [res])
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])))


@pytest.mark.parametrize(
    ("make_pass", "src", "expected"),
    [
        # The identity numpy documents, and the only form these backends trace natively.
        (DiffToSliceDifference, "d = np.diff(p)", "d = p[1:] - p[:-1]"),
        # ``copy`` decides whether numpy MAY alias, never what the values are; dace takes no such arg.
        (StripAstypeCopyKwarg, "y = x.astype(np.float32, copy=False)", "y = x.astype(np.float32)"),
    ],
    ids=["diff_becomes_the_slice_difference", "astype_copy_kwarg_is_dropped"],
)
def test_diff_becomes_the_slice_difference(make_pass: type[ast.NodeTransformer], src: str, expected: str) -> None:
    assert apply(make_pass(), src) == expected


def test_per_element_repeat_walks_a_running_offset() -> None:
    """The destination offset is the prefix sum of the counts, never ``i * K``."""
    out = apply(RepeatCountsInline({"p": 1}), "r = np.repeat(np.arange(M), np.diff(p))")
    assert "__rp0_pos = 0" in out, out
    assert "for __rp0_r in range(p[__rp0_i + 1] - p[__rp0_i]):" in out, out
    assert "__rp0[__rp0_pos] = __rp0_src[__rp0_i]" in out, out
    # The extent telescopes: sum(diff(p)) == p[-1] - p[0], with no data read.
    assert "np.zeros(p[-1] - p[0]" in out, out


@pytest.mark.parametrize(
    ("pass_obj", "src"),
    [
        # The result length would be ``idx.max() + 1`` -- data, not a shape we may invent.
        (BincountInline({"idx": 1}), "h = np.bincount(idx)"),
        # Only a first difference telescopes; any other count needs a sum we cannot derive here.
        (RepeatCountsInline({"v": 1, "counts": 1}), "r = np.repeat(v, counts)"),
    ],
    ids=["bincount_without_minlength", "per_element_count_that_is_not_a_difference"],
)
def test_a_form_whose_extent_is_data_is_left_standing(pass_obj: ast.NodeTransformer, src: str) -> None:
    assert apply(pass_obj, src) == src
