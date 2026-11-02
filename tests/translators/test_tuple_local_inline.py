# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""A tuple-valued local is inlined into its uses only while its elements keep their values.

QE vexx builds an FFT grid shape as ``grid = (n1, n2, n3)`` and reshapes with
``cg.reshape(grid + (-1,))``.
:class:`~hpcagent_bench.translators.numpyto_common.frontend.body_rewrites.FoldTupleLocals`
substitutes the tuple into its uses, which re-reads every element at the USE site -- sound only
while no element changes value between the binding and a use. A kernel that rebinds a scalar
extent afterwards (``n = n * 2`` for a second phase) made the fold read the NEW value where numpy
kept the snapshot the tuple captured.
"""

import ast

from hpcagent_bench.translators.numpyto_common.frontend.body_rewrites import FoldTupleLocals


def fold_(src: str, params: set[str]) -> str:
    fn = ast.parse(src).body[0]
    folder = FoldTupleLocals(params)
    folder.collect(fn)
    folder.visit(fn)
    ast.fix_missing_locations(fn)
    return ast.unparse(fn)


def test_an_element_rebound_after_the_binding_blocks_the_inline() -> None:
    """``shp`` snapshots ``(n, m)`` at its binding; inlining it past ``n = n * 2`` re-reads ``n``."""
    src = "def f(x, out, n):\n    m = n + 1\n    shp = (n, m)\n    n = n * 2\n    out[:] = x.reshape(shp + (-1,))\n"
    out = fold_(src, {"x", "out", "n"})
    assert "shp = (n, m)" in out, out
    assert "x.reshape(shp + (-1,))" in out, out
