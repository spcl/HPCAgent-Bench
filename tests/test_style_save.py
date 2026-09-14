# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The one save idiom every figure goes through."""

import pathlib

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from hpcagent_bench.stats import style


def test_a_figure_is_written_in_exactly_the_formats_asked_for(tmp_path: pathlib.Path) -> None:
    """A stale file in a format nobody asked for is read as current by whoever opens the directory."""
    fig, _ = plt.subplots()
    stem = style.save(fig, tmp_path / "nested" / "figure", formats=("pdf", "svg"))
    assert sorted(p.name for p in stem.parent.iterdir()) == ["figure.pdf", "figure.svg"]


def test_a_figure_saved_at_two_times_is_byte_identical(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A published figure is regenerated and diffed against the committed one. A file that stamps the
    time of its write differs on every rerun, so a changed number cannot be told from a rerun.
    ``SOURCE_DATE_EPOCH`` is matplotlib's clock for that stamp; the two saves run a day apart."""
    for epoch, folder in (("0", "first"), ("86400", "second")):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", epoch)
        fig, axes = plt.subplots()
        axes.plot([1.0, 2.0, 3.0])
        style.save(fig, tmp_path / folder / "figure", formats=("pdf", "svg", "png"))
    for name in ("figure.pdf", "figure.svg", "figure.png"):
        first, second = (tmp_path / folder / name for folder in ("first", "second"))
        assert first.read_bytes() == second.read_bytes(), f"{name} depends on when it was saved"
