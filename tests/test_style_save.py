# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The one save idiom every figure goes through."""

import pathlib
import re

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


@pytest.mark.parametrize(("scale", "minor_grid"), [
    ("linear", False),
    ("log", True),
])  # fmt: skip
def test_the_value_axis_helper_rules_minors_on_a_log_axis_only(scale: str, minor_grid: bool) -> None:
    """One helper, one grid. Every figure goes through ``value_axis``, so the minor ruling it draws
    on a log axis (user, 2026-09-22) is what every log value axis gets; a linear axis does not know
    whether it holds log2 units or a count, so its caller names that to ``minor_ticks``."""
    fig, ax = plt.subplots()
    try:
        ax.set_yscale(scale)
        style.value_axis(ax, "y")
        assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
        drawn = [tick for tick in ax.yaxis.get_minor_ticks() if tick.gridline.get_visible()]
        assert bool(drawn) == minor_grid
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
        texts = [tick.get_text() for tick in ax.yaxis.get_minorticklabels()]
        assert texts and not any(texts)
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


def small_figure(width_in: float, fontsize: float = style.PRINT_TICK_PT) -> plt.Figure:
    """A one-axes figure at ``width_in`` whose text is all ``fontsize`` and whose ink is narrower."""
    fig, ax = plt.subplots(figsize=(width_in, 1.5))
    ax.plot([1.0, 2.0])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel("x", fontsize=fontsize)
    fig.tight_layout(pad=1.5)
    return fig


def test_a_paper_figure_is_saved_exactly_as_wide_as_it_is_placed(tmp_path: pathlib.Path) -> None:
    """A tight crop sets the width from the ink, and the page then rescales the type by placed/drawn:
    the cost figure was drawn 2.586in wide, placed at 2.475in, and printed its 7pt ticks at 6.7pt."""
    style.save(
        small_figure(style.ICLR_WRAP_WIDTH_IN), tmp_path / "wrap", formats=("pdf",), width_in=style.ICLR_WRAP_WIDTH_IN
    )

    box = re.search(rb"/MediaBox\s*\[\s*0 0 ([0-9.]+)", (tmp_path / "wrap.pdf").read_bytes())
    assert box is not None
    assert float(box.group(1)) / 72.0 == pytest.approx(style.ICLR_WRAP_WIDTH_IN, abs=1e-3)


@pytest.mark.parametrize("fontsize", [style.PRINT_MIN_PT - 1.0, style.PRINT_LABEL_PT + 2.0])
def test_a_paper_figure_with_type_off_the_print_scale_is_refused(tmp_path: pathlib.Path, fontsize: float) -> None:
    """Text that one figure shrinks to fit (or draws at authoring size) prints at a size no other
    figure on the page uses; the save refuses it instead of writing it."""
    with pytest.raises(ValueError, match="text outside"):
        style.save(small_figure(3.0, fontsize), tmp_path / "f", formats=("pdf",), width_in=3.0)


def test_a_paper_figure_drawn_at_another_width_is_refused(tmp_path: pathlib.Path) -> None:
    """A canvas of the wrong width would be rescaled by the page, which is the defect the check exists for."""
    with pytest.raises(ValueError, match="placed at"):
        style.save(small_figure(3.0), tmp_path / "f", formats=("pdf",), width_in=style.ICLR_WRAP_WIDTH_IN)


def test_a_self_sized_figure_on_the_print_scale_still_has_its_type_checked(tmp_path: pathlib.Path) -> None:
    """``fixed`` figures (efficacy, per-kernel) size their own canvas; ``print_size`` gives them the same check."""
    with pytest.raises(ValueError, match="text outside"):
        style.save(small_figure(3.0, 12.0), tmp_path / "f", formats=("pdf",), fixed=True, print_size=True)
