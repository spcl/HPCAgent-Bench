# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compiler row labels and the shared point and value marks of the per-kernel compiler figure."""

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from hpcagent_bench.stats import style
from hpcagent_bench.stats.figures import kernel_speedups

TAG_KERNELS: tuple[str, ...] = ("k1", "k2", "k3")


def canon_table(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"run": "r1", "column": column, "kernel": kernel, "median_ms": ms, "validated": 1}
         for column, kernel, ms in rows]
    )  # fmt: skip


@pytest.fixture(name="canon")
def canon_fixture() -> pd.DataFrame:
    """numba times every kernel; the CPU DaCe column all three, the GPU one two, Pluto one."""
    return canon_table(
        [
            ("numba", "k1", 100.0), ("numba", "k2", 200.0), ("numba", "k3", 100.0),
            ("dace_cpu_canon", "k1", 10.0), ("dace_cpu_canon", "k2", 20.0),
            ("dace_cpu_canon", "k3", 50.0),
            ("dace_gpu_canon", "k1", 1.0), ("dace_gpu_canon", "k2", 2.0),
            ("pluto", "k1", 50.0),
        ]
    )  # fmt: skip


@pytest.mark.parametrize(
    ("variants", "want_labels"),
    [
        # the registry aliases both DaCe device variants to one optimizer: drawn together they get distinct labels
        pytest.param(("dace_cpu_canon", "dace_gpu_canon"), None, id="two-variants"),
        # the fallback fires only on a collision; a figure drawing one variant is unchanged
        pytest.param(("dace_cpu_canon",), ["DaCe+CPF"], id="one-variant"),
    ],
)
def test_two_device_variants_of_one_optimizer_get_distinct_row_labels(
    canon: pd.DataFrame, variants: tuple[str, ...], want_labels: list[str] | None
) -> None:
    """The compiler per-kernel figure labels a canon row by its optimizer, and the registry aliases
    both DaCe device variants to one. Drawn together they printed one name twice in the legend."""
    rows = kernel_speedups.kernel_rows(canon, None, TAG_KERNELS, variants)
    labels = [row.label for row in rows]
    if want_labels is None:
        assert len(set(labels)) == 2, labels
    else:
        assert labels == want_labels


def test_an_undelivered_placeholder_is_drawn_hollow() -> None:
    """A cross in the series' colour on a FILLED mark of that colour is invisible: 34 of 40 PPCG
    placeholders read as measured 1x results. The placeholder's face must be empty."""
    fig, ax = plt.subplots()
    style.point_mark(ax, 0.0, 1.0, "#cc79a7", "v", True, delivered=False)
    mark = ax.collections[1]  # the white halo is [0], the mark itself [1]
    assert len(mark.get_facecolors()) == 0 or mark.get_facecolors()[0][3] == 0.0
