# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A value hoist computes a call's temp in FRONT of the statement that uses it (np.einsum, axis reductions,
np.repeat(axis=), ...). A call inside a comprehension that reads the comprehension's own variable cannot move there:
the hoisted ``__rsrc0 = a[i]`` ran before ``i`` existed, so the backend program died on a NameError. A call that
reads none of those names still hoists.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from hpcagent_bench.translators.numpyto_common.numpy_desugar import desugar_for_python_backend
from tests.translators.source_module import run_source


def kernel_ir(**arrays: tuple[str, ...]) -> SimpleNamespace:
    """The fields ``desugar_for_python_backend`` reads off a KernelIR, for a kernel named ``k``."""
    return SimpleNamespace(
        kernel_name="k", arrays=[SimpleNamespace(name=n, shape=s, dtype=None) for n, s in arrays.items()], scalars=[]
    )


def run_kernel(src: str, *args: object) -> None:
    namespace: dict[str, object] = {"np": np}
    run_source(src, namespace, "<desugared>")
    namespace["k"](*args)


@pytest.mark.parametrize(
    ("src", "arrays", "hoists"),
    [
        (
            "def k(a, out, n):\n    out[:] = np.array([np.max(a[i], axis=0) for i in range(n)])\n",
            {"a": ("K", "M", "N"), "out": ("K", "N")},
            False,
        ),
        (
            'def k(a, out, n):\n    out[:] = np.array([np.einsum("ij,jk->ik", q, q)[0] for q in a])\n',
            {"a": ("K", "M", "M"), "out": ("K", "M")},
            False,
        ),
        # A call that reads no comprehension variable still hoists.
        (
            "def k(a, out, n):\n    out[:] = np.array([np.max(a, axis=0)[i] for i in range(n)])\n",
            {"a": ("K", "M", "N"), "out": ("M", "N")},
            True,
        ),
    ],
    ids=["reduction_over_the_comprehension_variable", "einsum_over_the_comprehension_target", "reduction_reading_none"],
)
def test_a_call_reading_the_comprehension_variable_stays_in_the_comprehension(
    src: str, arrays: dict[str, tuple[str, ...]], hoists: bool
) -> None:
    got = desugar_for_python_backend(src, kernel_ir(**arrays), backend="numba")

    if not hoists:
        assert got == src
        return
    a = np.random.default_rng(0).random((4, 3, 5))
    out, expected = np.empty((3, 5)), np.array([np.max(a, axis=0)[i] for i in range(3)])
    run_kernel(got, a, out, 3)

    assert "np.max(" not in got
    assert got.index("__rdo0 = np.empty(") < got.index("out[:] = np.array([__rdo0[i] for i in range(n)])")
    assert np.array_equal(out, expected)
