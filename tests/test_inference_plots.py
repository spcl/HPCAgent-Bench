# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the inference-aware figures in hpcagent_bench.plotting.

The load-bearing assertion is negative: a fitted normal curve must NOT be drawn over a
non-normal sample. That is the misleading figure the whole exercise exists to prevent, so it is
asserted against the rendered Axes rather than trusted to a code path reading correctly.
"""
import pathlib
import sqlite3
from typing import Dict, List, Tuple

import numpy as np
import pytest

from hpcagent_bench import inference, plotting

#: The synthetic DB uses REAL short_names so the shared report ordering resolves them.
KERNELS: Tuple[Tuple[str, str], ...] = (("heat3d", "Physics"), ("jacobi2d", "Physics"))


def build_results_db(db: pathlib.Path, shift: float = 0.0) -> None:
    """A small results DB with the real schema. ``shift`` scales dace_cpu's runtime so a corpus
    sweep can be pointed at either an all-null corpus or one with a planted win."""
    from sqlmodel import Session

    from hpcagent_bench.frameworks.schema import Result, results_engine

    rng = np.random.default_rng(0)
    engine = results_engine(str(db))
    with Session(engine) as session:
        for kernel, domain in KERNELS:
            for framework, base in (("numpy", 10.0), ("dace_cpu", 10.0 * (1.0 - shift))):
                for value in base * rng.lognormal(0.0, 0.05, 40):
                    session.add(
                        Result(timestamp=1_700_000_000,
                               benchmark=kernel,
                               domain=domain,
                               preset="S",
                               framework=framework,
                               agent=None,
                               validated=True,
                               time=float(value),
                               native_time=None,
                               datatype="float64",
                               variant=None,
                               prompt_hash=None,
                               execution="native"))
        session.commit()


def test_diagnostics_figure_renders_for_a_normal_sample(tmp_path: pathlib.Path) -> None:
    samples = np.random.default_rng(0).normal(100.0, 5.0, 200)
    assert inference.check_normality(samples).normal, "premise: this branch needs a normal sample"
    out = plotting.plot_sample_diagnostics(samples,
                                           title="normal-cell",
                                           output=str(tmp_path / "normal.pdf"),
                                           drop=False,
                                           usetex=False)
    assert pathlib.Path(out).exists() and pathlib.Path(out).stat().st_size > 0


def test_diagnostics_figure_renders_for_a_skewed_sample(tmp_path: pathlib.Path) -> None:
    samples = 100.0 * np.random.default_rng(0).lognormal(0.0, 0.5, 200)
    assert not inference.check_normality(samples).normal
    out = plotting.plot_sample_diagnostics(samples,
                                           title="skewed-cell",
                                           output=str(tmp_path / "skewed.pdf"),
                                           drop=False,
                                           usetex=False)
    assert pathlib.Path(out).exists() and pathlib.Path(out).stat().st_size > 0


def rendered_labels(monkeypatch: pytest.MonkeyPatch, samples: np.ndarray, tmp_path: pathlib.Path) -> Dict[str, object]:
    """Render the diagnostics figure and capture the Axes before save_figure closes them."""
    captured: List[object] = []
    original = plotting.save_figure

    def spy(output: str, fig) -> str:
        captured.append([(ax.get_title(), [line.get_label() for line in ax.get_lines()]) for ax in fig.axes])
        captured.append(" ".join(text.get_text() for text in fig.texts))  # the suptitle lives here
        return original(output, fig)

    monkeypatch.setattr(plotting, "save_figure", spy)
    plotting.plot_sample_diagnostics(samples, title="cell", output=str(tmp_path / "f.pdf"), drop=False, usetex=False)
    return {"axes": captured[0], "suptitle": captured[1]}


def test_no_normal_curve_is_drawn_over_a_non_normal_sample(monkeypatch: pytest.MonkeyPatch,
                                                           tmp_path: pathlib.Path) -> None:
    """⛔ THE point of the whole exercise. A Gaussian over right-skewed wall-clock is the
    misleading plot; assert against the rendered artists, not against the code path."""
    skewed = 100.0 * np.random.default_rng(0).lognormal(0.0, 0.5, 200)
    normal = np.random.default_rng(0).normal(100.0, 5.0, 200)

    skewed_labels = [
        label for _title, labels in rendered_labels(monkeypatch, skewed, tmp_path)["axes"] for label in labels
    ]
    normal_labels = [
        label for _title, labels in rendered_labels(monkeypatch, normal, tmp_path)["axes"] for label in labels
    ]
    assert "fitted normal" not in skewed_labels
    assert "fitted normal" in normal_labels  # and it IS drawn where it is honest to draw it


def test_figure_labels_the_interval_type(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """A reader must be able to tell a parametric interval from a bootstrap one FROM THE FIGURE."""
    skewed = 100.0 * np.random.default_rng(0).lognormal(0.0, 0.5, 200)
    normal = np.random.default_rng(0).normal(100.0, 5.0, 200)

    skewed_titles = [title for title, _labels in rendered_labels(monkeypatch, skewed, tmp_path)["axes"]]
    normal_titles = [title for title, _labels in rendered_labels(monkeypatch, normal, tmp_path)["axes"]]
    assert any("bootstrap" in t and "median" in t for t in skewed_titles), skewed_titles
    assert any(t.endswith("t CI for mean") for t in normal_titles), normal_titles


def test_figure_reports_the_verdict_reason(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    skewed = 100.0 * np.random.default_rng(0).lognormal(0.0, 0.5, 200)
    suptitle = rendered_labels(monkeypatch, skewed, tmp_path)["suptitle"]
    assert "n=200" in str(suptitle) and "shapiro" in str(suptitle)


def test_diagnostics_refuses_an_empty_sample(tmp_path: pathlib.Path) -> None:
    with pytest.raises(RuntimeError, match="no usable samples"):
        plotting.plot_sample_diagnostics([], title="empty", output=str(tmp_path / "e.pdf"), usetex=False)


def test_corpus_comparisons_applies_fdr_and_finds_a_planted_win(tmp_path: pathlib.Path) -> None:
    """The corpus-level caller must return ADJUSTED significance, and must still see a real 40%
    win through the correction."""
    db = tmp_path / "planted.db"
    build_results_db(db, shift=0.4)
    rows = plotting.corpus_comparisons("dace_cpu", "numpy", db=str(db), preset="S")
    assert len(rows) == len(KERNELS)
    assert all(row.significant_adjusted for row in rows)
    assert all(row.comparison.test == "mann-whitney" for row in rows)  # independent samples
    assert all(row.comparison.ratio > 1.4 for row in rows)
    assert all(row.pvalue_adjusted >= row.comparison.pvalue for row in rows)


def test_corpus_comparisons_stays_quiet_on_an_all_null_corpus(tmp_path: pathlib.Path) -> None:
    """Same distribution on both sides: nothing may be declared significant."""
    db = tmp_path / "null.db"
    build_results_db(db, shift=0.0)
    rows = plotting.corpus_comparisons("dace_cpu", "numpy", db=str(db), preset="S")
    assert rows and not any(row.significant_adjusted for row in rows)


def test_corpus_comparisons_order_is_deterministic(tmp_path: pathlib.Path) -> None:
    """Row order reaches the report table, so two reads of one DB must agree."""
    db = tmp_path / "order.db"
    build_results_db(db, shift=0.4)
    first = [row.key for row in plotting.corpus_comparisons("dace_cpu", "numpy", db=str(db), preset="S")]
    second = [row.key for row in plotting.corpus_comparisons("dace_cpu", "numpy", db=str(db), preset="S")]
    assert first == second
    assert sqlite3.connect(str(db)).execute("SELECT COUNT(*) FROM results").fetchone()[0] > 0
