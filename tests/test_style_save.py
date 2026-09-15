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


def test_the_value_axis_helper_draws_a_major_grid_and_never_a_minor_one() -> None:
    """One helper, one grid. Every figure in the repo goes through ``value_axis``, so switching the
    minor lines off here is what switches them off everywhere: a minor line is a second grid at a
    second weight, and once a figure is reduced for print the panel reads as a texture the marks sit
    on rather than a reference they sit against."""
    for scale in ("linear", "log"):
        fig, ax = plt.subplots()
        try:
            ax.set_yscale(scale)
            style.value_axis(ax, "y")
            assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
            assert not [tick for tick in ax.yaxis.get_minor_ticks() if tick.gridline.get_visible()]
            assert not [tick for tick in ax.xaxis.get_major_ticks() if tick.gridline.get_visible()]
        finally:
            plt.close(fig)


def test_a_log_value_axis_labels_no_minor_tick_in_scientific_notation() -> None:
    """Matplotlib's own log minor formatter labels a 3x10^n tick whenever few majors are visible,
    which puts a scientific-notation number on an axis whose majors are plain ones."""
    fig, ax = plt.subplots()
    try:
        ax.set_yscale("log")
        ax.set_ylim(400.0, 3000.0)
        style.value_axis(ax, "y", log_base=10.0)
        fig.canvas.draw()
        assert [tick.get_text() for tick in ax.yaxis.get_minorticklabels()] == []
    finally:
        plt.close(fig)


def test_a_point_mark_that_never_delivered_an_answer_carries_a_cross_on_the_model_shape() -> None:
    """A kernel the arm was served and never solved enters every aggregate at 1x and keeps its
    tokens, so the mark is a PLACEHOLDER, not a measurement. The cross says so while the shape still
    names the model and the colour still names the intervention."""
    fig, ax = plt.subplots()
    try:
        style.point_mark(ax, 0.0, 1.0, "#0072b2", "o", True, delivered=True)
        delivered_marks = len(ax.collections)
        style.point_mark(ax, 1.0, 1.0, "#0072b2", "o", True, delivered=False)
        assert len(ax.collections) == delivered_marks * 2 + 1
        assert ax.collections[-1].get_zorder() > style.MARK_Z
    finally:
        plt.close(fig)
