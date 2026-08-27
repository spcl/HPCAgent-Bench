# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``np.add.at`` whose values carry the axes the index leaves untouched.

``np.add.at(c_blocks, flat_c_pos, alpha * flat_prod)`` (cp2k_density_matrix_trs4) scatters rank-2
BLOCKS through a rank-1 index into a rank-3 target: ``A[idx]`` keeps ``A``'s trailing axes, so a
rank-3 values array is the shape numpy itself demands. The scatter desugar read any rank mismatch
as unmodelled broadcasting and refused, which took the kernel off every python backend.
"""
import ast

import numpy as np
import pytest

from numpyto_common.numpy_desugar import DesugarError, _AddAtInline


def _apply(pass_obj, src: str) -> str:
    body: list = []
    for stmt in ast.parse(src).body:
        res = pass_obj.visit(stmt)
        if res is None:
            continue
        body.extend(res if isinstance(res, list) else [res])
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])))


def test_block_values_get_a_loop_per_untouched_axis():
    # The two trailing axes are iterated, not left as a subarray ``+=``: scalar accumulation is
    # what every backend supports, and it keeps the unbuffered duplicate-index order.
    out = _apply(_AddAtInline({"c": 3, "pos": 1, "prod": 3, "alpha": 0}), "np.add.at(c, pos, alpha * prod)")
    assert "for __sc0_t0 in range(__sc0_v.shape[1]):" in out, out
    assert "for __sc0_t1 in range(__sc0_v.shape[2]):" in out, out
    assert "c[__sc0_x0[__sc0_i0], __sc0_t0, __sc0_t1] += __sc0_v[__sc0_i0, __sc0_t0, __sc0_t1]" in out, out


def test_matching_shape_scatter_is_untouched():
    # edge_laplacian's flat scatter: nothing trailing, so no extra loop may appear.
    out = _apply(_AddAtInline({"Lx": 1, "src": 1, "flux": 1}), "np.add.at(Lx, src, flux)")
    assert out == ("__sc0_x0 = np.ascontiguousarray(src)\n"
                   "__sc0_v = np.ascontiguousarray(flux)\n"
                   "for __sc0_i0 in range(__sc0_x0.shape[0]):\n"
                   "    Lx[__sc0_x0[__sc0_i0]] += __sc0_v[__sc0_i0]"), out


def test_a_genuine_broadcast_still_refuses():
    # rank-1 index into a rank-1 target leaves NO axis untouched, so rank-2 values really are an
    # unmodelled broadcast -- the refusal this generalisation must not swallow.
    with pytest.raises(DesugarError, match="axes the index leaves untouched"):
        _apply(_AddAtInline({"Lx": 1, "src": 1, "flux": 2}), "np.add.at(Lx, src, flux)")


def test_lowered_loop_reproduces_numpy_with_duplicate_indices():
    # The property the lowering exists for: duplicate targets accumulate, they do not overwrite.
    rng = np.random.default_rng(0)
    nb, bs, nv = 5, 3, 11
    ref = rng.standard_normal((nb, bs, bs))
    got = ref.copy()
    pos = rng.integers(0, nb, nv)
    prod = rng.standard_normal((nv, bs, bs))
    alpha = 0.75
    assert nv - len(set(pos.tolist())) > 0, "the fixture must actually contain duplicate indices"
    np.add.at(ref, pos, alpha * prod)
    src = _apply(_AddAtInline({"c": 3, "pos": 1, "prod": 3, "alpha": 0}), "np.add.at(c, pos, alpha * prod)")
    exec(compile(src, "<lowered>", "exec"), {"np": np}, {"c": got, "pos": pos, "prod": prod, "alpha": alpha})
    assert np.array_equal(ref, got)
