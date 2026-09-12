# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``scripts/plot_score_change.py`` -- the score-vs-cost figure and the stars on it.

The load-bearing assertions are about MULTIPLICITY. One figure carries three models x two
languages x two axes, so twelve signed-rank tests decide its marks, and twelve uncorrected 5%
thresholds paint at least one star on 46% of figures where nothing happened. What is asserted here
is therefore that a raw threshold which would have starred a point does not survive the correction,
that the mark is gated on the corrected verdict, and that the figure says so where a reader looks
-- a corrected star and an uncorrected star are the same pixels.
"""

import importlib.util
import pathlib
import sys

import pandas as pd
import pytest

from hpcagent_bench.harness import efficacy

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Models that exist in the tag registry, so the palette resolves them instead of warning.
MODELS: tuple[str, ...] = ("qwen38", "oss120b", "kimi27sglang")
LANGUAGES: tuple[str, ...] = ("c", "fortran")

#: Kernels per (model, language) cell. Above ``summary.MIN_PAIRS_FOR_INTERVAL``, so every cell in
#: the fixture is a test that actually ran and the family is the full twelve.
KERNELS: int = 8


def load_script():
    """Import ``scripts/plot_score_change.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("plot_score_change", REPO / "scripts" / "plot_score_change.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: dataclasses resolves a string annotation through
    # sys.modules[cls.__module__], which is None for a module loaded by path alone.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plot = load_script()


#: Per-kernel factors whose two smallest magnitudes go the wrong way, so the exact signed-rank
#: statistic is W- = 3 at n = 8 and the raw two-sided p is 0.0391 -- just inside a per-row 5%
#: threshold, which is the case the correction has to catch.
MARGINAL: tuple[float, ...] = (1.02, 1.04, 1.0 / 1.005, 1.0 / 1.01, 1.08, 1.10, 1.12, 1.14)

#: Every kernel moved the same way and a long way: a real effect, which the correction must NOT eat.
DECISIVE: tuple[float, ...] = tuple(1.25 + 0.01 * k for k in range(KERNELS))

#: A win and a loss of equal size, alternating: nothing happened, on either axis.
FLAT: tuple[float, ...] = tuple(1.05 if k % 2 == 0 else 1.0 / 1.05 for k in range(KERNELS))


def episode(arm: str, model: str, language: str, kernel: int, run: str, speedup: float, tokens: float) -> list[dict]:
    """One episode as the judge records it: a GRADED row carrying the speed-up and no token count,
    and a ``call`` row carrying the cumulative token count and no timings.

    Two record types because the figure's two axes come off two different ones. A fixture with one
    row carrying both is the shape that let the loader filter on ``speedup > 0 and tokens > 0`` and
    silently keep the call rows alone.
    """
    common = {
        "arm": arm,
        "model": model,
        "language": language,
        "benchmark": f"k{kernel}",
        "run_root": run,
        "job": run,
        "run_id": run,
        "baseline": "numba",
        "attempt_index": 1,
        "ts_ms": kernel,
    }
    return [
        {**common, "record": "submission", "speedup": speedup, "tokens": None},
        {**common, "record": "call", "speedup": speedup, "tokens": tokens},
    ]


def observations(gains: tuple[float, ...], winner: tuple[str, str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Control and treated frames for all six cells, ``gains`` applied to ``winner`` alone.

    With ``winner`` set, the five other cells get :data:`FLAT` and the only raw thresholds the
    family can cross are the winner's two. With ``winner`` None every cell gets ``gains``.
    """
    control, treated = [], []
    for model in MODELS:
        for language in LANGUAGES:
            for kernel in range(KERNELS):
                treated_cell = winner is None or (model, language) == winner
                gain = gains[kernel] if treated_cell else FLAT[kernel]
                arm = f"{model}-{language}"
                run = f"{arm}-{kernel}"
                control += episode(arm, model, language, kernel, run, 2.0, 1000.0)
                treated += episode(f"{arm}-skills", model, language, kernel, f"{run}-t", 2.0 * gain, 1000.0 / gain)
    return pd.DataFrame(control), pd.DataFrame(treated)


def test_a_raw_threshold_that_would_have_starred_a_point_does_not_survive_the_correction() -> None:
    """One cell reaching p = 0.039 on its own is what a per-row ``p < 0.05`` reads as a finding. It
    is one of twelve tests on the figure, and corrected across them the value is 0.23 -- so the star
    it would have drawn is not supported by the figure it would have been drawn on."""
    before, after = observations(MARGINAL, winner=("qwen38", "c"))
    frame = plot.points(before, after)
    winner = frame[(frame.model == "qwen38") & (frame.language == "c")].iloc[0]
    assert winner.score_p == pytest.approx(0.0390625), "the fixture has to cross a raw 5% threshold"
    assert winner.score_p_adjusted > 0.05
    assert winner.score_verdict == efficacy.NOT_SIGNIFICANT
    assert not (frame.score_verdict == efficacy.SIGNIFICANT).any()


def test_an_effect_every_arm_shows_still_survives_the_correction() -> None:
    """The correction has to cost power, not all of it: BH keeps a finding that the whole family
    agrees on, which is what separates it from simply refusing to mark anything."""
    before, after = observations(DECISIVE, winner=None)
    frame = plot.points(before, after)
    assert (frame.score_verdict == efficacy.SIGNIFICANT).all()
    assert (frame.score_p_adjusted < 0.05).all()


def test_the_family_the_marks_are_corrected_over_is_every_test_the_figure_could_mark() -> None:
    """Six (model, language) cells on two axes is twelve, and the count is reported rather than
    left to a reader to reconstruct from how the table happened to be built."""
    before, after = observations(MARGINAL, winner=("qwen38", "c"))
    frame = plot.points(before, after)
    assert len(frame) == 6
    assert plot.family_size(frame) == 12


@pytest.mark.parametrize(
    "score_verdict, cost_verdict, starred",
    [
        pytest.param(efficacy.SIGNIFICANT, efficacy.NOT_SIGNIFICANT, True, id="score only"),
        pytest.param(efficacy.NOT_SIGNIFICANT, efficacy.SIGNIFICANT, True, id="cost only"),
        pytest.param(efficacy.NOT_SIGNIFICANT, efficacy.NOT_SIGNIFICANT, False, id="neither"),
        pytest.param(efficacy.UNDERPOWERED, efficacy.UNDERPOWERED, False, id="never tested"),
        pytest.param(efficacy.UNCORRECTED, efficacy.UNCORRECTED, False, id="not in a family"),
    ],
)
def test_the_figure_stars_a_point_only_on_a_corrected_verdict(
    score_verdict: str, cost_verdict: str, starred: bool
) -> None:
    """The gate is the VERDICT, so a raw p that was never corrected, and a pairing too small for
    any test, both draw a plain label -- the two states a boolean flag had nowhere to put."""
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        [
            {"model": "qwen38", "language": "c", "skills": False, "log2_speedup": 1.0, "tokens": 1000.0},
            {"model": "qwen38", "language": "c", "skills": True, "log2_speedup": 1.4, "tokens": 900.0},
        ]
    )
    stats = pd.DataFrame(
        [
            {
                "model": "qwen38",
                "language": "c",
                "score_verdict": score_verdict,
                "cost_verdict": cost_verdict,
                "family_size": 12,
            }
        ]
    )
    fig, ax = plt.subplots()
    try:
        plot.draw_absolute(ax, frame, stats)
        labels = [text.get_text() for text in ax.texts]
    finally:
        plt.close(fig)
    assert ("C *" in labels) == starred, labels


def test_the_figure_names_the_correction_and_the_size_of_the_family() -> None:
    """An uncorrected star and a corrected one are identical pixels, so the only place the reader
    can learn which they are looking at is the key beside the mark."""
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        [
            {"model": "qwen38", "language": "c", "skills": False, "log2_speedup": 1.0, "tokens": 1000.0},
            {"model": "qwen38", "language": "c", "skills": True, "log2_speedup": 1.4, "tokens": 900.0},
        ]
    )
    stats = pd.DataFrame(
        [
            {
                "model": "qwen38",
                "language": "c",
                "score_verdict": efficacy.SIGNIFICANT,
                "cost_verdict": "",
                "family_size": 12,
            }
        ]
    )
    fig, ax = plt.subplots()
    try:
        labels = [handle.get_label() for handle in plot.draw_absolute(ax, frame, stats)]
    finally:
        plt.close(fig)
    assert "BH q < 0.05 of 12" in labels, labels
