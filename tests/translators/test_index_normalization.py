# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Consolidated index-access normalisation: Ellipsis / newaxis / scalar-chained.

The single ``normalize-index-access`` lowering phase (chained-flatten -> ellipsis-
expand -> trailing-slice pad) runs AFTER the shape harvest, so an ``...`` / newaxis
on a POST-INLINE local resolves against a now-known rank. These are the exact
subscript shapes the LS3DF fragment solver (``_hpsi``) exercises -- ``vloc[..., None]
* X`` and ``psi_frag[f][..., 0]`` -- checked numerically against numpy on the C and
Fortran backends (the ABI backends that flatten to a raw pointer, where a surviving
``...`` would otherwise reach the emitter as an unlowerable literal Ellipsis)."""

import numpy as np
import pytest

from tests.translators.op_oracle import all_ok, run_op

BACKENDS = ("c", "fortran")
TOL = 1e-6


def normal(seed: int, *shape: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape)


@pytest.mark.parametrize(
    ("src", "inputs", "out_shape", "syms", "shapes"),
    [
        # ``a[..., 0]`` on a 3-D array -> ``a[:, :, 0]`` (the trailing scalar keeps the
        # last axis; the Ellipsis fills the two leading source axes).
        (
            "import numpy as np\ndef f(a, out):\n    out[:, :] = a[..., 0]\n",
            {"a": normal(0, 3, 4, 5)},
            (3, 4),
            {"M": 3, "N": 4, "K": 5},
            {"a": "(M,N,K)", "out": "(M,N)"},
        ),
        # ``a[..., None] * x`` -- the LS3DF ``vloc[..., None] * X`` shape: the Ellipsis
        # fills a's two axes, the newaxis inserts a trailing size-1 axis that broadcasts
        # against x's last axis.
        (
            "import numpy as np\ndef f(a, x, out):\n    out[:, :, :] = a[..., None] * x\n",
            {"a": normal(1, 3, 4), "x": normal(11, 3, 4, 5)},
            (3, 4, 5),
            {"M": 3, "N": 4, "K": 5},
            {"a": "(M,N)", "x": "(M,N,K)", "out": "(M,N,K)"},
        ),
        # ``A[i][..., 0]`` -- a scalar-chained subscript flattened to ``A[i, ..., 0]``
        # then the Ellipsis expanded to ``A[i, :, :, 0]`` (LS3DF ``psi_frag[f][..., 0]``).
        (
            "import numpy as np\ndef f(A, out):\n    for i in range(A.shape[0]):\n        out[i, :, :] = A[i][..., 0]\n",
            {"A": normal(2, 2, 3, 4, 5)},
            (2, 3, 4),
            {"NF": 2, "M": 3, "N": 4, "K": 5},
            {"A": "(NF,M,N,K)", "out": "(NF,M,N)"},
        ),
        # ``A[i, ..., j]`` -- an Ellipsis between two scalar indices: axis 0 and the last
        # axis are consumed, the Ellipsis fills the middle two -> ``A[i, :, :, j]``.
        (
            "import numpy as np\n"
            "def f(A, out):\n"
            "    for i in range(A.shape[0]):\n"
            "        for j in range(A.shape[3]):\n"
            "            out[i, j, :, :] = A[i, ..., j]\n",
            {"A": normal(3, 2, 3, 4, 2)},
            (2, 2, 3, 4),
            {"M": 2, "P": 3, "Q": 4, "N": 2},
            {"A": "(M,P,Q,N)", "out": "(M,N,P,Q)"},
        ),
        # ``np.transpose(a, perm)[..., None]`` -- the base is a CALL, so the normalisation phase
        # (which only fires on a Name) leaves the Ellipsis standing. The call hoister materialises
        # the transpose into a sized temp one phase later, and the expansion has to happen there or
        # the literal Ellipsis reaches the emitter (mamba2_return_final_state's ``b_decayed``).
        (
            "import numpy as np\ndef f(a, x, out):\n    out[:, :, :, :] = np.transpose(a, (0, 2, 1))[..., None] * x\n",
            {"a": normal(4, 2, 3, 4), "x": normal(14, 2, 4, 3, 5)},
            (2, 4, 3, 5),
            {"M": 2, "N": 3, "K": 4, "P": 5},
            {"a": "(M,N,K)", "x": "(M,K,N,P)", "out": "(M,K,N,P)"},
        ),
        # ``(np.sum(m * n, axis=-1) / d)[..., None]`` -- cfd's shape. The base is an ARITHMETIC
        # expression, so there is no name to look a rank up under; its operands are sized, which is
        # all the expansion needs. Left unexpanded, the Ellipsis reaches the emitter as a literal.
        (
            "import numpy as np\ndef f(m, n, d, x, out):\n"
            "    out[:, :, :] = (np.sum(m * n, axis=-1) / d)[..., None] * x\n",
            {"m": normal(5, 3, 4, 5), "n": normal(15, 3, 4, 5), "d": normal(25, 3, 4) + 3.0, "x": normal(35, 3, 4, 5)},
            (3, 4, 5),
            {"M": 3, "N": 4, "K": 5},
            {"m": "(M,N,K)", "n": "(M,N,K)", "d": "(M,N)", "x": "(M,N,K)", "out": "(M,N,K)"},
        ),
    ],
    ids=[
        "trailing_scalar",
        "then_newaxis_broadcast",
        "after_a_scalar_chain",
        "between_two_scalars",
        "over_a_hoisted_call_base",
        "over_an_arithmetic_base",
    ],
)
def test_ellipsis_expands_to_the_axes_it_stands_for(
    src: str,
    inputs: dict[str, np.ndarray],
    out_shape: tuple[int, ...],
    syms: dict[str, int],
    shapes: dict[str, str],
) -> None:
    res = run_op(src, "f", inputs, {"out": out_shape}, syms, shapes=shapes, backends=BACKENDS, rtol=TOL, atol=TOL)
    ok, r = all_ok(res)
    assert ok, r
