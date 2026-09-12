# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The one save idiom every figure goes through."""

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from hpcagent_bench.stats import style


def test_a_figure_is_written_in_exactly_the_formats_asked_for(tmp_path: pathlib.Path) -> None:
    """A stale file in a format nobody asked for is read as current by whoever opens the directory."""
    fig, _ = plt.subplots()
    stem = style.save(fig, tmp_path / "nested" / "figure", formats=("pdf", "svg"))
    assert sorted(p.name for p in stem.parent.iterdir()) == ["figure.pdf", "figure.svg"]
