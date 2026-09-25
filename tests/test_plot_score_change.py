# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``statistics/plot_score_change.py`` -- the efficacy figure and the stars on it.

The load-bearing assertions are about MULTIPLICITY (a raw threshold does not survive the BH
correction, a real effect does) and about the DOT-ROW CONTRACT: one column per (LLM, delivery), the
control hollow beside the packet arm, each row carrying only its own axis's star, and the drawing
itself lives in :mod:`hpcagent_bench.stats.figures.efficacy` -- this script only wires the data.
"""

import argparse
import dataclasses
import importlib.util
import math
import pathlib
import sys
import tempfile
import warnings

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure

from hpcagent_bench.harness import efficacy
from hpcagent_bench.stats import cost, palette, score_rule
from hpcagent_bench.stats import style as plotstyle
from hpcagent_bench.stats.figures import efficacy as efficacy_figures

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Models that exist in the tag registry, so the palette resolves them instead of warning.
MODELS: tuple[str, ...] = ("qwen38", "oss120b", "kimi27sglang")
LANGUAGES: tuple[str, ...] = ("c", "fortran")

#: Kernels per (model, language) cell. Above ``summary.MIN_PAIRS_FOR_INTERVAL``, so every cell in
#: the fixture is a test that actually ran and the family is the full twelve.
KERNELS: int = 8


def load_script():
    """Import ``statistics/plot_score_change.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("plot_score_change", REPO / "statistics" / "plot_score_change.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plot = load_script()

#: Per-kernel factors whose two smallest magnitudes go the wrong way, so the paired t test on their
#: logs gives a raw two-sided p of 0.0197 at n = 8 -- inside a per-row 5% threshold, which is the case
#: the correction has to catch.
MARGINAL: tuple[float, ...] = (1.02, 1.04, 1.0 / 1.005, 1.0 / 1.01, 1.08, 1.10, 1.12, 1.14)

#: Every kernel moved the same way and a long way: a real effect, which the correction must NOT eat.
DECISIVE: tuple[float, ...] = tuple(1.25 + 0.01 * k for k in range(KERNELS))

#: A win and a loss of equal size, alternating: nothing happened, on either axis.
FLAT: tuple[float, ...] = tuple(1.05 if k % 2 == 0 else 1.0 / 1.05 for k in range(KERNELS))


def spent(tokens: float) -> dict[str, float]:
    """A task row's token columns: the total, stated as fresh input alone so every cost card prices
    the task at ``tokens``."""
    return {"tokens": tokens, "tokens_fresh_input": tokens, "tokens_cached_input": 0.0, "tokens_output": 0.0}


def episode(arm: str, model: str, language: str, kernel: int, run: str, speedup: float, tokens: float) -> list[dict]:
    """One episode as the judge records it: a GRADED row carrying the speed-up and no token count,
    and a ``task`` row carrying the token total and no timings."""
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
        "timing_reduction": "mwd-v2",
    }
    return [
        {
            **common,
            "record": "submission",
            "speedup": speedup,
            "tokens": None,
            "suspect": 0,
            # SC15 Rule 4: a summarized ratio travels with the times it was taken over, and
            # stats.rules refuses a table whose cost columns are entirely absent.
            "baseline_ns": 1.0e6,
            "native_ns": 1.0e6 / speedup,
        },
        {**common, "record": "task", "speedup": None, **spent(tokens), "suspect": None},
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


def one_arm_raw(
    model: str = "qwen38",
    language: str = "c",
    off_speedup: float = 1.0,
    on_speedup: float = 1.4,
    off_tokens: float = 1000.0,
    on_tokens: float = 900.0,
    kernels: int = KERNELS,
) -> pd.DataFrame:  # fmt: skip
    """One arm's RAW tagged rows (``skills`` True/False), the shape
    :func:`~hpcagent_bench.stats.figures.efficacy.reduce_pair` pairs per kernel.

    A small alternating per-kernel jitter, never zero: every kernel at exactly the same ratio has no
    spread for :func:`~hpcagent_bench.stats.summary.geomean_ci` to draw an interval around, which a
    real campaign never is.
    """
    control, treated = [], []
    for kernel in range(kernels):
        jitter = 1.0 + (0.03 if kernel % 2 == 0 else -0.03)
        control += episode(f"{model}-{language}", model, language, kernel, f"c{kernel}", off_speedup, off_tokens)
        treated += episode(f"{model}-{language}-skills", model, language,
                           kernel, f"t{kernel}", on_speedup * jitter,
                           on_tokens / jitter)  # fmt: skip
    frame = pd.concat([pd.DataFrame(control).assign(skills=False), pd.DataFrame(treated).assign(skills=True)])
    return frame.reset_index(drop=True)


def one_arm_stats(
    model: str = "qwen38",
    language: str = "c",
    score_verdict: str = efficacy.NOT_SIGNIFICANT,
    cost_verdict: str = efficacy.NOT_SIGNIFICANT,
    family_size: int = 2,
) -> pd.DataFrame:  # fmt: skip
    return pd.DataFrame([{
        "model": model,
        "language": language,
        "score_verdict": score_verdict,
        "cost_verdict": cost_verdict,
        "family_size": family_size
    }])  # fmt: skip


def test_a_raw_threshold_that_would_have_starred_a_point_does_not_survive_the_correction() -> None:
    """One cell reaching p = 0.020 on its own is what a per-row ``p < 0.05`` reads as a finding. It
    is one of twelve tests on the figure, and corrected across them the value is 0.12 -- so the star
    it would have drawn is not supported by the figure it would have been drawn on."""
    before, after = observations(MARGINAL, winner=("qwen38", "c"))
    frame = plot.points(before, after)
    winner = frame[(frame.model == "qwen38") & (frame.language == "c")].iloc[0]
    assert winner.score_p == pytest.approx(0.019747, abs=1e-5), "the fixture has to cross a raw 5% threshold"
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
    assert plot.efficacy_figures.family_size(frame) == 12


#: The card for a frame that carries only ``arm`` and ``packet``: ``effective`` leaves the frame
#: unpriced, so a test about packets needs no token columns.
PACKET_ONLY = cost.resolve("effective")


def test_load_reads_skills_off_the_recorded_packet_before_the_arm_name(tmp_path: pathlib.Path) -> None:
    """An arm renamed away from the ``-skills`` suffix but recording ``lang-skills`` loads as skilled, and a recorded
    packet beats a ``-skills`` name."""
    path = tmp_path / "observations.csv"
    pd.DataFrame(
        [
            {"arm": "renamed-qwen38-c", "packet": "skills"},
            {"arm": "qwen38-fortran-skills", "packet": "cpf"},
            {"arm": "qwen38-c-skills", "packet": ""},
        ]
    ).to_csv(path, index=False)

    frame = plot.load(path, prefix="", card=PACKET_ONLY)

    by_arm = frame.set_index("arm").skills
    assert bool(by_arm["renamed-qwen38-c"]) is True
    assert bool(by_arm["qwen38-fortran-skills"]) is False
    assert bool(by_arm["qwen38-c-skills"]) is True


def test_load_counts_a_composite_packet_as_skilled(tmp_path: pathlib.Path) -> None:
    """``llrsingle`` records ``lang-skills+no-score-tool``; comparing the whole packet for
    equality against the bare ``lang-skills`` key read every one of its skilled arms as
    unskilled."""
    path = tmp_path / "observations.csv"
    pd.DataFrame(
        [
            {"arm": "llrsingle-qwen38-c-skills", "packet": "lang-skills+no-score-tool"},
            {"arm": "llrsingle-qwen38-c", "packet": "no-score-tool"},
        ]
    ).to_csv(path, index=False)

    frame = plot.load(path, prefix="", card=PACKET_ONLY)

    by_arm = frame.set_index("arm").skills
    assert bool(by_arm["llrsingle-qwen38-c-skills"]) is True
    assert bool(by_arm["llrsingle-qwen38-c"]) is False


def test_control_rows_is_exactly_the_no_packet_arm(tmp_path: pathlib.Path) -> None:
    """``control_rows`` needs no treatment list at all: the control is exactly the canonical empty
    packet, whatever treatments a campaign happens to run."""
    path = tmp_path / "observations.csv"
    pd.DataFrame(
        [
            {"arm": "cpf-llr-focus40-qwen38-c-perf-playbook-cpu", "packet": "perf-playbook-cpu"},
            {"arm": "cpf-llr-focus40-qwen38-c-cpfsrc", "packet": "cpfsrc"},
            {"arm": "cpf-llr-focus40-qwen38-c-skills", "packet": "skills"},
            {"arm": "cpf-llr-focus40-qwen38-c", "packet": ""},
        ]
    ).to_csv(path, index=False)

    frame_all = plot.load(path, prefix="", card=PACKET_ONLY)
    control = plot.control_rows(frame_all)

    assert set(control.arm) == {"cpf-llr-focus40-qwen38-c"}


def test_a_perf_playbook_arm_never_enters_the_control_side(tmp_path: pathlib.Path) -> None:
    """The bug this guards: an arm recording a treatment ``control_rows`` does not name by string
    must never be silently counted as part of the no-packet control."""
    path = tmp_path / "observations.csv"
    pd.DataFrame(
        [
            {"arm": "cpf-llr-focus40-qwen38-c-perf-playbook-cpu", "packet": "perf-playbook-cpu"},
            {"arm": "cpf-llr-focus40-qwen38-c", "packet": ""},
        ]
    ).to_csv(path, index=False)

    frame_all = plot.load(path, prefix="", card=PACKET_ONLY)
    control = plot.control_rows(frame_all)

    assert "cpf-llr-focus40-qwen38-c-perf-playbook-cpu" not in set(control.arm)
    assert set(control.arm) == {"cpf-llr-focus40-qwen38-c"}


def test_treatment_frame_tags_the_control_false_and_the_treatment_true() -> None:
    """The figure reads an on/off ``skills`` flag; ``treatment_frame`` builds it from the packet
    split rather than the historical column name."""
    frame_all = pd.DataFrame(
        [
            {"arm": "a-control", "packet": "", "model": "qwen38", "language": "c"},
            {"arm": "a-cpfsrc", "packet": "cpfsrc", "model": "qwen38", "language": "c"},
            {"arm": "a-skills", "packet": "skills", "model": "qwen38", "language": "c"},
        ]
    )
    tagged = plot.treatment_frame(frame_all, "cpfsrc")
    by_arm = tagged.set_index("arm").skills
    assert bool(by_arm["a-control"]) is False
    assert bool(by_arm["a-cpfsrc"]) is True
    assert "a-skills" not in by_arm.index


def test_points_never_raises_a_bare_keyerror_when_the_two_sides_share_no_model_language() -> None:
    """The bug this guards: ``pd.DataFrame([])`` (an empty ``rows`` list) has NO columns at all, so
    ``.dropna(subset=["score", "cost"])`` on it raised a bare ``KeyError`` where the caller expected
    "these two sides pair on nothing"."""
    before = pd.DataFrame([{"model": "oss120b", "language": "c", "record": "submission"}])
    after = pd.DataFrame([{"model": "oss120b", "language": "", "record": "submission"}])

    frame = plot.points(before, after)

    assert frame.empty
    assert list(frame.columns) == list(plot.POINT_COLUMNS)


def test_a_treatment_arm_that_never_recorded_its_language_still_pairs_against_control(
    tmp_path: pathlib.Path,
) -> None:
    """``load`` (through ``experiments.fill_arm_identity``) must recover ``c`` from the arm's own
    name, and ``one_treatment_panel`` must then find the (model, language) key it shares with its
    control instead of finding nothing."""
    path = tmp_path / "observations.csv"
    rows = []
    for kernel in range(KERNELS):
        common = {
            "benchmark": f"k{kernel}",
            "suspect": 0,
            "baseline": "numba",
            "run_root": "j1",
            "job": "j1",
            "attempt_index": 1,
            "ts_ms": kernel,
            "timing_reduction": "mwd-v2",
        }  # fmt: skip
        for arm, packet, language, speedup in (
            ("cpf-llr-focus40-oss120b-c", "", "c", 2.0),
            ("cpf-llr-focus40-oss120b-c-cpf", "cpf", "", 2.4),
        ):
            run = f"{arm}-{kernel}"
            base = {**common, "arm": arm, "packet": packet, "language": language, "run_id": run}
            rows.append({
                **base, "record": "submission",
                "speedup": speedup,
                "baseline_ns": 1000.0,
                "native_ns": 1000.0 / speedup
            })  # fmt: skip
            rows.append({**base, "record": "task", "speedup": None, **spent(1000.0)})
    pd.DataFrame(rows).to_csv(path, index=False)

    frame_all = plot.load(path, prefix="")
    treated = frame_all[frame_all.arm == "cpf-llr-focus40-oss120b-c-cpf"]
    assert set(treated.language) == {"c"}, "the arm name is the last resort when no row ever recorded it"

    control = plot.control_rows(frame_all)
    roster = sorted(frame_all.benchmark.dropna().unique())
    built = plot.one_treatment_panel(frame_all, control, "cpf", roster)

    assert built is not None
    stats, frame = built
    del frame  # this test is about the stats table, not the raw rows behind it
    assert len(stats) == 1
    assert (stats.iloc[0].model, stats.iloc[0].language) == ("oss120b", "c")


def test_complete_side_arms_drops_an_arm_short_of_the_roster_and_names_it_on_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An arm missing a roster kernel is dropped, not entered at any stand-in value, and named so
    the drop is auditable."""
    roster = ["k0", "k1", "k2"]
    control = pd.DataFrame({"arm": ["ctrl"] * 3, "benchmark": roster})
    treated = pd.DataFrame(
        {"arm": ["good-cpf", "good-cpf", "good-cpf", "short-cpf"], "benchmark": ["k0", "k1", "k2", "k0"]}
    )

    kept = plot.complete_side_arms(control, treated, roster, "cpf", include_incomplete=False)

    assert kept == {"ctrl", "good-cpf"}
    err = capsys.readouterr().err
    assert "cpf: dropping short-cpf (1/3 roster kernels)" in err


def test_include_incomplete_keeps_a_short_arm_and_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    roster = ["k0", "k1", "k2"]
    control = pd.DataFrame({"arm": ["ctrl"] * 3, "benchmark": roster})
    treated = pd.DataFrame({"arm": ["short-cpf"], "benchmark": ["k0"]})

    kept = plot.complete_side_arms(control, treated, roster, "cpf", include_incomplete=True)

    assert kept == {"ctrl", "short-cpf"}
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# The paired interval, the stars, the tick budget, the left margin and the key.


def test_error_bars_are_drawn_on_both_axes_from_the_paired_geomean() -> None:
    """Every arm's mark carries a crossed 95% interval: SC15 Rule 5/7, computed from
    ``summary.geomean_ci`` on the paired per-kernel ratios."""
    frame = one_arm_raw(on_speedup=2.0, on_tokens=400.0)
    series = efficacy_figures.reduce_pair(frame[~frame.skills], frame[frame.skills])
    assert series is not None
    assert series.x_low < series.x < series.x_high
    assert series.y_low < series.y < series.y_high


def test_a_marks_significance_superscript_reads_off_the_corrected_verdict_per_axis() -> None:
    """The gate is the VERDICT, per axis: a raw p that was never corrected, and a pairing too small
    for any test, both draw NO superscript. Each row carries only its own axis's symbol: ``*`` on
    the speed-up row, ``+`` on the cost row."""
    import matplotlib.pyplot as plt

    def symbols_of(stats: pd.DataFrame) -> dict[str, list[str]]:
        significance = efficacy_figures.axis_significance(stats)
        row = dataclasses.replace(arrow_row(), leg="C")
        drawn: dict[str, list[str]] = {}
        for measure in ("speedup", "cost"):
            fig, ax = plt.subplots()
            try:
                efficacy_figures.draw_measure_row(ax, [row], measure, "^", significance)
                drawn[measure] = [text.get_text() for text in ax.texts]
            finally:
                plt.close(fig)
        return drawn

    neither = symbols_of(one_arm_stats())
    both = symbols_of(one_arm_stats(score_verdict=efficacy.SIGNIFICANT, cost_verdict=efficacy.SIGNIFICANT))
    assert neither == {"speedup": [], "cost": []}, neither
    assert both == {
        "speedup": [efficacy_figures.SCORE_SIG_MARK],
        "cost": [efficacy_figures.COST_SIG_MARK],
    }, both


@pytest.mark.parametrize(
    ("span", "expected_step"),
    [(2.2, 1), (8.0, 1), (9.0, 2), (16.0, 2), (17.0, 4), (64.0, 8)],
)
def test_x_tick_step_widens_before_the_tick_budget_is_crossed(span: float, expected_step: int) -> None:
    """A step of 1 (every power of 2) crams a label onto every few pixels once the window is wide
    enough -- two crashed-to-1/512x and ran-away-to-256x kernels in the same panel widened it past
    twenty octaves in production. The step doubles (every power of 4, then 16, ...) instead of
    landing on an arbitrary 'nice number' spacing, so every tick still names a clean ratio."""
    assert efficacy_figures.x_tick_step(span) == expected_step
    assert span / efficacy_figures.x_tick_step(span) <= efficacy_figures.MAX_X_TICKS - 1


def y_axis_left_margin(low: float, high: float) -> float:
    """:func:`~hpcagent_bench.stats.figures.efficacy.required_left_margin` for a bare axes carrying
    only the token-cost Y axis, over ``[low, high]``: the fixture
    :func:`test_required_left_margin_grows_with_the_widest_tick_the_range_draws` measures."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    fig.set_dpi(plotstyle.SAVE_DPI)
    ax.set_yscale("log", base=2.0)
    ax.set_ylim(low, high)
    plotstyle.value_axis(ax, "y", log_base=2.0)
    ax.yaxis.set_major_formatter(FuncFormatter(plotstyle.ratio_tick))
    ax.set_ylabel("Token-Cost Ratio, Treated / Control", fontsize=plotstyle.LABEL_PT * 0.68)
    ax.tick_params(axis="both", labelsize=plotstyle.TICK_PT * 0.6)
    try:
        return efficacy_figures.required_left_margin(fig, ax)
    finally:
        plt.close(fig)


def test_required_left_margin_grows_with_the_widest_tick_the_range_draws() -> None:
    """A direct measurement: a wide-ranging axis's own tick labels (``1/128x`` .. ``128x``) beside
    the panel's Y label need more room than a narrow range's short ones (``1x``) -- a flat fraction
    of the row's width, reserved regardless of content, cannot tell
    the two apart and ran a real outlier kernel's label off the canvas before this measured it."""
    wide = y_axis_left_margin(2.0**-7, 2.0**7)  # 1/128x .. 128x, the dynamic range a real outlier kernel gave
    narrow = y_axis_left_margin(0.5, 2.0)  # 1/2x .. 2x
    assert wide > narrow, (wide, narrow)


def test_the_interval_note_is_fixed_text_and_does_not_name_the_estimator() -> None:
    """Fixed text, so the key carries it once
    (:func:`~hpcagent_bench.stats.figures.efficacy.legend_tail`). The estimator is NOT named: "95%
    log-t CI" on two of five rows was the densest text in the figure, and which interval it is
    belongs in the caption beside the test it came from (user, 2026-09-20)."""
    note = efficacy_figures.interval_note("Speed-up")
    assert "log-t" not in note, note
    assert note == efficacy_figures.interval_note("Speed-up"), "fixed text, not sample-size-chosen"


def test_the_figure_key_carries_one_interval_note_per_axis() -> None:
    """Both intervals reach the reader in the figure's ONE key."""
    handles = efficacy_figures.legend_tail()
    labels = [h.get_label() for h in handles]
    assert efficacy_figures.interval_note("Speed-Up") in labels, labels
    assert efficacy_figures.interval_note("Token Cost") in labels, labels


@pytest.mark.parametrize(
    ("symbols", "wanted"),
    [
        ((False, False), []),
        ((True, False), [efficacy_figures.SCORE_SIG_LABEL]),
        ((False, True), [efficacy_figures.COST_SIG_LABEL]),
        ((True, True), [efficacy_figures.SCORE_SIG_LABEL, efficacy_figures.COST_SIG_LABEL]),
    ],
)
def test_the_key_explains_a_superscript_exactly_when_the_panel_drew_one(
    symbols: tuple[bool, bool], wanted: list[str]
) -> None:
    """A reader meeting ``*`` beside a mark has to be able to look it up (user, 2026-09-20). A row
    for a symbol NO mark wears is the opposite problem: a lookup made for nothing."""
    labels = [h.get_label() for h in efficacy_figures.legend_tail(symbols)]
    explained = [line for line in labels if "Significant" in line]
    assert explained == [
        f"{mark}  {text}"
        for mark, text in zip(
            (efficacy_figures.SCORE_SIG_MARK, efficacy_figures.COST_SIG_MARK),
            (efficacy_figures.SCORE_SIG_LABEL, efficacy_figures.COST_SIG_LABEL),
            strict=True,
        )
        if text in wanted
    ], explained


def test_an_undelivered_kernel_still_counts_in_the_served_geomean() -> None:
    """A kernel the arm was served and never verified is a placeholder, not a measurement
    (``population.DELIVERED_COLUMN``) -- under ``served`` it still scores 1x and still counts its
    tokens in the geomean, and the pair records it as not delivered."""
    control_rows_list = [
        ep for kernel in range(KERNELS) for ep in episode("a-c", "qwen38", "c", kernel, f"c{kernel}", 2.0, 1000.0)
    ]
    treated_rows_list = [
        ep for kernel in range(KERNELS) for ep in episode("a-c-skills", "qwen38", "c", kernel, f"t{kernel}", 2.5, 900.0)
    ]
    # One extra kernel BOTH sides reference -- the control verified it normally, the treated side
    # only ever wrote a token-bearing task row for it, never a verified submission: served, and
    # never delivered.
    control_rows_list += episode("a-c", "qwen38", "c", "missing", "c-missing", 2.0, 1000.0)
    treated_rows_list.append({
        "arm": "a-c-skills",
        "model": "qwen38",
        "language": "c",
        "benchmark": "kmissing",
        "run_root": "tm",
        "job": "tm",
        "run_id": "tm",
        "baseline": "numba",
        "attempt_index": 1,
        "ts_ms": 0,
        "timing_reduction": "mwd-v2",
        "record": "task",
        "speedup": None,
        "tokens": 950.0,
        "suspect": None,
    })  # fmt: skip
    control = pd.DataFrame(control_rows_list)
    treated = pd.DataFrame(treated_rows_list)

    series = efficacy_figures.reduce_pair(control, treated, over="served")

    assert series is not None
    assert series.delivered < series.kernels
    assert not efficacy_figures.paired_kernels(control, treated).delivered.all()


# ---------------------------------------------------------------------------
# The EXPLICIT-PAIR entry point: a comparison whose two sides are two campaigns, or whose condition
# is not a packet suffix at all, drawn through the same figure.

BLIND_PAIR: tuple[str, str] = ("llrblind-qwen38-c-skills", "cpf-llr-focus40-qwen38-c-skills")
SCICOMP_PAIR: tuple[str, str] = ("git-scicomp-qwen38-c-repo", "git-scicomp-qwen38-c-kernel")


def family_csv(pairs: list[tuple[str, str]], score_verdict: str, cost_verdict: str, n: int = 40) -> pd.DataFrame:
    """A family table in the shape ``statistics/paired_arms.py`` writes, one row per pair per leg."""
    return pd.DataFrame(
        [
            {
                "family": "demo",
                "score_rule": score_rule.SCORE_RULE,
                "cost_model": cost.DEFAULT_COST_MODEL,
                "kernel_policy": efficacy_figures.SPEEDUP_OVER,
                "arm_a": treated,
                "arm_b": control,
                "leg": leg,
                "n_pairs": n,
                "n_tested": n,
                "estimate_a_over_b": 0.8,
                "ci_low": 0.7,
                "ci_high": 0.9,
                "p_adjusted": 0.01,
                "verdict": score_verdict if leg == plot.SPEEDUP_LEG else cost_verdict,
            }  # fmt: skip
            for treated, control in pairs
            for leg in (plot.SPEEDUP_LEG, plot.TOKENS_LEG)
        ]
    )


def test_a_pairs_leg_names_the_language_and_every_packet_both_arms_carried() -> None:
    """llrblind runs C and C with the skill pages against their own scored arms, so a leg label of
    the language alone would draw two different arms as one."""
    assert plot.pair_leg_label(BLIND_PAIR, "no-score") == "C +skills"
    plain = ("llrblind-qwen38-fortran", "cpf-llr-focus40-qwen38-fortran")
    assert plot.pair_leg_label(plain, "no-score") == "Fortran"


def test_a_pairs_leg_never_names_the_intervention_the_two_sides_differ_in() -> None:
    """git-scicomp's two arms differ in `repo` against `kernel`; naming either on the label would
    say on every row what the figure's own title says once."""
    label = plot.pair_leg_label(SCICOMP_PAIR, "repo")
    assert label == "C"
    assert "repo" not in label and "kernel" not in label


def test_the_stars_come_off_the_family_csv_and_are_never_recomputed_here() -> None:
    """``statistics/paired_arms.py`` already ran the paired test and the Benjamini-Hochberg
    correction over exactly this family, and the paper's table is printed from the same CSV."""
    table = family_csv([BLIND_PAIR], efficacy.SIGNIFICANT, efficacy.NOT_SIGNIFICANT)
    stats = plot.family_stats(table, "no-score")

    assert list(stats.leg) == ["C +skills"]
    assert list(stats.model) == ["qwen38"]
    assert list(stats.score_verdict) == [efficacy.SIGNIFICANT]
    assert list(stats.cost_verdict) == [efficacy.NOT_SIGNIFICANT]
    assert list(stats.kernels) == [40]
    assert plot.efficacy_figures.family_size(stats) == 2


def test_the_family_csv_declares_the_pairs_in_the_order_it_wrote_them() -> None:
    """The family's own declared order, not a re-sort."""
    table = family_csv([BLIND_PAIR, SCICOMP_PAIR], efficacy.NOT_SIGNIFICANT, efficacy.NOT_SIGNIFICANT)
    assert plot.family_pairs(table) == [BLIND_PAIR, SCICOMP_PAIR]


def observation_rows(arm: str, speedup: float, tokens: float, kernels: int = KERNELS) -> list[dict[str, object]]:
    """One arm's rows in the shape an extraction writes: a graded submission and a task total per
    kernel."""
    rows: list[dict[str, object]] = []
    for kernel in range(kernels):
        common = {
            "arm": arm,
            "benchmark": f"k{kernel}",
            "suspect": 0,
            "baseline": "numba",
            "run_root": "j1",
            "job": "j1",
            "run_id": f"{arm}-{kernel}",
            "attempt_index": 1,
            "ts_ms": kernel,
            "timing_reduction": "mwd-v2",
        }  # fmt: skip
        rows.append({
            **common, "record": "submission",
            "speedup": speedup,
            "baseline_ns": 1000.0,
            "native_ns": 1000.0 / speedup
        })  # fmt: skip
        rows.append({**common, "record": "task", "speedup": None, **spent(tokens)})
    return rows


def test_pair_frame_tags_each_arm_by_name_and_which_side_of_the_pair_it_is() -> None:
    """The two sides live in two campaigns with different arm prefixes, so there is no packet
    suffix to split on -- the pair names the arms directly."""
    frame_all = pd.DataFrame(observation_rows(BLIND_PAIR[0], 2.0, 150e3) + observation_rows(BLIND_PAIR[1], 4.0, 200e3))

    tagged = plot.pair_frame(frame_all, [BLIND_PAIR], "no-score")

    assert set(tagged.leg) == {"C +skills"}
    assert set(tagged.model) == {"qwen38"}
    assert set(tagged[tagged.skills].arm) == {BLIND_PAIR[0]}
    assert set(tagged[~tagged.skills].arm) == {BLIND_PAIR[1]}

    series = efficacy_figures.reduce_pair(tagged[~tagged.skills], tagged[tagged.skills])
    assert series is not None
    assert series.y == pytest.approx(150e3 / 200e3)


def test_resolve_row_repeats_broadcasts_a_bare_policy_and_checks_a_sequences_length() -> None:
    """A joined row's comparisons need not share ONE repeat policy: git-scicomp's designed-3x-repeats
    median and llr-focus40's reruns-take-latest sit in the same row."""
    assert efficacy_figures.resolve_row_repeats("median", 3) == ["median", "median", "median"]
    assert efficacy_figures.resolve_row_repeats(["latest", "median"], 2) == ["latest", "median"]
    with pytest.raises(ValueError, match="panels 3"):
        efficacy_figures.resolve_row_repeats(["latest"], 3)


def test_a_comparison_specs_own_repeats_overrides_the_row_default(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``repeats=`` on one ``--comparison`` spec reaches ONLY that panel; a spec without it keeps
    ``--repeats``. A policy dropped on the way to the row builder is the same bug the
    ``--pairs-csv`` route already shipped once."""
    obs = tmp_path / "obs.csv"
    pd.DataFrame(observation_rows(SCICOMP_PAIR[0], 1.1, 1.4e6) + observation_rows(SCICOMP_PAIR[1], 0.6, 900e3)).to_csv(
        obs, index=False
    )
    table = tmp_path / "pairs.csv"
    family_csv([SCICOMP_PAIR], efficacy.NOT_SIGNIFICANT, efficacy.NOT_SIGNIFICANT).to_csv(table, index=False)

    seen_repeats: list[object] = []
    real = efficacy_figures.figure_dot_row

    def record(*args: object, **kwargs: object) -> object:
        seen_repeats.append(kwargs["repeats"])
        return real(*args, **kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]

    monkeypatch.setattr(plot.efficacy_figures, "figure_dot_row", record)
    old_argv = sys.argv
    sys.argv = [
        "plot_score_change.py", str(obs), "--comparison",
        f"title=Kernel Formulation;intervention=repo;pairs={table};repeats=median",
        "--out", str(tmp_path / "fig.pdf"), "--table", str(tmp_path / "table.csv"),
    ]  # fmt: skip
    try:
        plot.main()
    finally:
        sys.argv = old_argv
    assert seen_repeats == [["median"]]


# ---------------------------------------------------------------------------
# The control's own shape, and the arrow a named comparison carries.


@pytest.mark.parametrize("treatment", ["lang-skills", "cpf", "cpfsrc", "repo", "no-score-tool"])
def test_no_packet_is_ever_given_the_shape_the_control_wears(treatment: str) -> None:
    """The control is HOLLOW, and hollow-versus-filled alone does not separate two marks once a
    figure is reduced to a column (user, 2026-09-20): it is a different OUTLINE. A packet handed
    that outline would put a filled circle beside a hollow one and undo the separation."""
    assert efficacy_figures.treatment_marker(treatment) != efficacy_figures.CONTROL_MARKER


def test_a_difference_spec_reads_delivery_colon_model_pairs() -> None:
    """``--difference`` is the one place the arrow grammar is decided."""
    assert efficacy_figures.parse_differences("HIP:qwen38, Triton:kimi27sglang ") == frozenset(
        {("qwen38", "HIP"), ("kimi27sglang", "Triton")}
    )


@pytest.mark.parametrize(
    ("value", "want"), [(6.34919, "6.3x"), (0.92137, "0.9x"), (1.0, "1.0x"), (4.0, "4.0x"), (0.5, "0.5x")]
)
def test_an_arrows_factor_is_printed_to_one_decimal(value: float, want: str) -> None:
    """The label beside a mark is what a reader quotes, one decimal (user, 2026-09-21); the tick
    spelling keeps full precision, which beside a mark reads as ``6.34919x``."""
    assert plotstyle.ratio_label(value) == want


def arrow_row() -> efficacy_figures.ArmRow:
    """One category whose packet arm is 4x its control on both measures."""
    control = efficacy_figures.ArmPoint(1.0, 0.9, 1.1, 1e6, 0.9e6, 1.1e6, 40, 40)
    treated = efficacy_figures.ArmPoint(3.0, 2.9, 3.1, 4e6, 3.9e6, 4.1e6, 40, 40)
    return efficacy_figures.ArmRow("qwen38", "HIP", "#1f77b4", control, treated)


@pytest.mark.parametrize("measure", ["speedup", "cost"])
def test_only_a_named_comparison_gets_an_arrow_and_it_carries_the_factor(measure: str) -> None:
    """An arrow on every column is a second grid, so they are asked for by name. Its label is the
    factor BETWEEN the two marks -- on the speed-up row the axis holds log2, so the factor is a
    power of two and not the difference the axis shows."""
    import matplotlib.pyplot as plt

    def texts(differences: frozenset[tuple[str, str]]) -> list[str]:
        fig, ax = plt.subplots()
        try:
            efficacy_figures.draw_measure_row(ax, [arrow_row()], measure, "^", {}, differences=differences)
            return [text.get_text() for text in ax.texts]
        finally:
            plt.close(fig)

    # Printed as every value beside a mark is, to one decimal (``style.ratio_label``, 2026-09-21).
    assert plotstyle.ratio_label(4.0) in texts(frozenset({("qwen38", "HIP")}))
    assert plotstyle.ratio_label(4.0) not in texts(frozenset())


@pytest.mark.parametrize("measure", ["speedup", "cost"])
def test_a_measure_row_border_never_opens_an_empty_tick_step(measure: str) -> None:
    """A border snaps to the next tick only when the data reaches within half a step of it; a 0.94x
    interval end opened the speed-up axis down to 0.5x, a whole empty step (user, 2026-09-25).
    Earlier (2026-09-20) every border snapped to a tick."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        efficacy_figures.draw_measure_row(ax, [arrow_row()], measure, "^", {}, efficacy_figures.PAPER_CONFIG)
        low, high = ax.get_ylim()
        data_low, data_high = ax.dataLim.y0, ax.dataLim.y1
        ticks = sorted(ax.get_yticks())
        log = ax.get_yscale() == "log"
        span = (lambda a, b: math.log(b / a)) if log else (lambda a, b: b - a)
        step = min(span(a, b) for a, b in zip(ticks, ticks[1:]))
        reach = efficacy_figures.SNAP_REACH * step + 1e-9
        assert span(low, data_low) <= reach, (low, data_low, step)
        assert span(data_high, high) <= reach, (high, data_high, step)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "name", ["i) Loop Reasoning (LLR), CPU", "iii) Repository Context", "iv) Repo vs. Kernel", "ii) X"]
)
def test_a_panel_name_never_folds_onto_a_third_line(name: str) -> None:
    """A third line comes out of the panel. The width is SEARCHED because the first line carries
    the tag as well as its first word ("iii) Repository"), which no per-word rule predicts -- the
    estimate put "Repository Context" on three lines in an 11-character column."""
    lines = efficacy_figures.PAPER_CONFIG.max_name_lines
    folded = efficacy_figures.wrapped_label(name, efficacy_figures.name_line_width(name, 11, lines))
    assert folded.count("\n") + 1 <= lines, folded


def test_parse_spec_reads_semicolon_separated_key_value_pairs() -> None:
    """``--comparison`` takes one string per panel; the parser is the only place its grammar is
    decided."""
    spec = plot.parse_spec("title=Kernel Formulation;intervention=repo;pairs=x.csv;control-label=Bare Kernel")
    assert spec == {
        "title": "Kernel Formulation",
        "intervention": "repo",
        "pairs": "x.csv",
        "control-label": "Bare Kernel",
    }  # fmt: skip


# ---------------------------------------------------------------------------
# Ratio tick labels: a ratio below 1 is spelled in full, never rounded to 0x.


def test_the_token_cost_axis_formatter_spells_a_ratio_below_one_as_a_fraction() -> None:
    assert plotstyle.ratio_tick(0.125) == "0.125x"
    assert plotstyle.ratio_tick(1.0) == "1x"
    assert plotstyle.ratio_tick(8.0) == "8x"


# ---------------------------------------------------------------------------
# Several packets sharing ONE panel (``treatments=``): read and recorded per packet.


def one_arm_observations_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    """One tiny campaign as an extracted-observations CSV: a control, a ``skills`` arm and a
    ``cpf`` arm, two models -- what :func:`plot.build_multi_comparison` reads off disk. Reuses
    :func:`observation_rows`'s own shape so the emitted rows carry the ``baseline_ns``/``native_ns``
    columns :func:`~hpcagent_bench.stats.figures.efficacy.pairs_table` requires (SC15 Rule 4)."""
    rows: list[dict[str, object]] = []
    for model, base in (("qwen38", 2.0), ("oss120b", 3.0)):
        for packet, suffix, factor in (("", "", 1.0), ("skills", "-skills", 1.2), ("cpf", "-cpf", 0.8)):
            arm = f"exp-{model}-c{suffix}"
            rows += [
                {**row, "language": "c", "packet": packet} for row in observation_rows(arm, base * factor, 1000.0 / factor)
            ]  # fmt: skip
    path = tmp_path / "observations.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_build_multi_comparison_reads_treatments_as_a_comma_list() -> None:
    obs = one_arm_observations_csv(pathlib.Path(tempfile.mkdtemp()))
    built = plot.build_multi_comparison(
        {"treatments": "skills,cpf", "title": "CPU"}, [obs], "exp", "latest", False, plot.cost.resolve()
    )
    assert built is not None
    title, treatments, stats, frame = built
    assert title == "CPU"
    assert list(treatments) == ["skills", "cpf"]
    assert set(frame) == {"skills", "cpf"}
    assert set(stats) == {"skills", "cpf"}
    assert set(frame["skills"].model.unique()) == {"qwen38", "oss120b"}


def test_several_treatments_of_one_campaign_draw_one_dot_row(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two ``--treatment`` flags join as columns of ONE row, each against the same control."""
    drawn: list[list[str]] = []
    real = efficacy_figures.figure_dot_row

    def record(panels, *args: object, **kwargs: object) -> object:
        drawn.append([treatment for _, treatment, _, _ in panels])
        return real(panels, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(plot.efficacy_figures, "figure_dot_row", record)
    obs = one_arm_observations_csv(tmp_path)
    out = tmp_path / "fig.pdf"
    old_argv = sys.argv
    sys.argv = [
        "plot_score_change.py", str(obs), "--experiment", "exp", "--treatment", "skills", "--treatment", "cpf",
        "--out", str(out), "--table", str(tmp_path / "table.csv"),
    ]  # fmt: skip
    try:
        plot.main()
    finally:
        sys.argv = old_argv
    assert drawn == [["skills", "cpf"]]
    assert out.exists()


def test_write_panel_tables_merges_a_multi_treatment_panel_into_one_packet_tagged_csv(
    tmp_path: pathlib.Path,
) -> None:
    obs = one_arm_observations_csv(tmp_path)
    built = plot.build_multi_comparison(
        {"treatments": "skills,cpf"}, [obs], "exp", "latest", False, plot.cost.resolve()
    )
    assert built is not None
    _, _, stats, frame = built
    table = tmp_path / "table.csv"
    plot.write_panel_tables(table, "-cpu", stats, frame, "latest")
    written = pd.read_csv(table.with_name("table-cpu.csv"))
    absolute = pd.read_csv(table.with_name("table-cpu-absolute.csv"))
    assert set(written.packet) == {"skills", "cpf"}
    assert set(absolute.packet) == {"skills", "cpf"}


def test_a_kernel_without_a_token_total_keeps_its_speed_up_and_the_table_says_n() -> None:
    """A treated kernel whose task row was lost still has a verified answer: the speed-up
    coordinate is the geomean over EVERY paired kernel (the family CSV's own score leg), the token
    coordinate over the kernels priced on both sides, and the table records that n. Intersecting
    the two moved Kimi's C skill-pages point from 0.83x (38 kernels) to 1.01x (19)."""
    frame = one_arm_raw(on_speedup=2.0, on_tokens=500.0)
    unpriced = {f"k{kernel}" for kernel in range(0, KERNELS, 2)}
    frame = frame[~(frame.skills & (frame.record == "task") & frame.benchmark.isin(unpriced))]

    series = efficacy_figures.reduce_pair(frame[~frame.skills], frame[frame.skills])

    assert series is not None
    assert (series.kernels, series.token_kernels) == (KERNELS, KERNELS - len(unpriced))
    ratios = [2.0 * (1.03 if kernel % 2 == 0 else 0.97) for kernel in range(KERNELS)]
    assert 2.0**series.x == pytest.approx(math.exp(np.mean(np.log(ratios))))
    priced = [0.5 / (1.03 if kernel % 2 == 0 else 0.97) for kernel in range(KERNELS) if f"k{kernel}" not in unpriced]
    assert series.y == pytest.approx(math.exp(np.mean(np.log(priced))))
    table = efficacy_figures.pairs_table(frame.assign(baseline_ns=2.0e6, native_ns=1.0e6))
    assert table.token_kernels.tolist() == [series.token_kernels]
    assert table.kernels.tolist() == [KERNELS]


def test_the_pairs_csv_route_draws_its_marks_under_the_repeat_policy_it_was_asked_for(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``figure_from_pairs`` once passed neither ``--repeats`` nor the figure config on to the
    figure, so its marks were drawn under the default ``latest`` while the table written
    one line above used the policy the caller asked for. On git-scicomp, whose campaign rule is the
    median of three repeats, that put Kimi-K2.7-Code at 3.57x where its own CSV said 0.67x -- the
    difference between the repository helping and hurting."""
    rows: list[dict] = []
    for index in range(KERNELS):
        # Three runs per kernel that disagree: the LATEST is fast, their median is slow, so a
        # figure drawn under the wrong policy lands on the far side of 1x.
        for run, speedup in (("r1", 0.5), ("r2", 0.5), ("r3", 8.0)):
            rows += episode("git-repo", "qwen38", "c", index, f"k{index}-{run}", speedup, 100.0)
        rows += episode("git-kernel", "qwen38", "c", index, f"k{index}-c", 1.0, 100.0)
    observations_csv = tmp_path / "obs.csv"
    pd.DataFrame(rows).to_csv(observations_csv, index=False)

    pairs_csv = tmp_path / "pairs.csv"
    pd.DataFrame(
        [
            {
                "family": "f",
                "cost_model": "effective",
                "score_rule": score_rule.SCORE_RULE,
                "kernel_policy": efficacy_figures.SPEEDUP_OVER,
                "arm_a": "git-repo",
                "arm_b": "git-kernel",
                "leg": leg,
                "verdict": efficacy.NOT_SIGNIFICANT,
                "n_pairs": KERNELS,
                "p_adjusted": 0.9,
            }
            for leg in (plot.SPEEDUP_LEG, plot.TOKENS_LEG)
        ]
    ).to_csv(pairs_csv, index=False)

    frame = plot.pair_frame(plot.load_all([observations_csv]), [("git-repo", "git-kernel")], "repo")

    def treated_x(policy: str) -> float:
        points = efficacy_figures.arm_points(frame[~frame.skills], frame[frame.skills], policy)
        assert points is not None
        return points[1].x

    by_policy = {policy: treated_x(policy) for policy in ("median", "latest")}
    assert by_policy["median"] != pytest.approx(by_policy["latest"]), by_policy

    drawn: list[float] = []
    real = efficacy_figures.arm_points

    def spy(*args, **kwargs):
        points = real(*args, **kwargs)
        drawn.append(points[1].x)
        return points

    monkeypatch.setattr(efficacy_figures, "arm_points", spy)
    args = argparse.Namespace(
        speedup_over=efficacy_figures.SPEEDUP_OVER,
        pairs_csv=pairs_csv,
        observations=[observations_csv],
        intervention="repo",
        control_label="Kernel Formulation",
        cost_model="effective",
        cost_models=None,
        repeats="median",
        out=tmp_path / "f.pdf",
        table=tmp_path / "f.csv",
        channels="model-packet",
        dots_panel_labels="outside",
        difference="",
        success_row=True,
        dots_row_height=2.3,
    )
    plot.figure_from_pairs(args, efficacy_figures.DEFAULT_CONFIG)
    # The figure and the -absolute table beside it: both under the requested policy.
    assert drawn and drawn == pytest.approx([by_policy["median"]] * len(drawn)), (drawn, by_policy)


def solved_and_failed_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Five kernels: the control answers k0-k3 at 2x and gets k4 wrong; the treated arm answers all
    five, k0-k3 at 4x and k4 at 8x."""
    control = [row for row in observation_rows(BLIND_PAIR[1], 2.0, 100.0, kernels=5)
               if not (row["benchmark"] == "k4" and row["record"] == "submission")]  # fmt: skip
    treated = observation_rows(BLIND_PAIR[0], 4.0, 150.0, kernels=5)
    treated = [{**row, "speedup": 8.0, "native_ns": 125.0} if row["benchmark"] == "k4" and row["record"] == "submission"
               else row for row in treated]  # fmt: skip
    tagged = plot.pair_frame(pd.DataFrame(control + treated), [BLIND_PAIR], "no-score")
    return tagged[~tagged.skills], tagged[tagged.skills]


def test_by_default_a_wrong_answer_is_no_speedup_and_counts_against_the_success_rate() -> None:
    """2026-09-21: the speed-up of both arms is over the kernels BOTH solved, so the control's wrong
    k4 is not scored as its baseline and the treated arm's k4 win does not lift it either; the
    failure is the success rate's to show."""
    points = efficacy_figures.arm_points(*solved_and_failed_pair())
    assert points is not None
    control, treated = points
    assert 2.0**control.x == pytest.approx(2.0) and 2.0**treated.x == pytest.approx(4.0)
    assert control.kernels == treated.kernels == 4
    assert (control.solved, control.served, treated.solved, treated.served) == (4, 5, 5, 5)


def test_the_fallback_reading_scores_the_wrong_answer_at_one() -> None:
    """``served`` keeps the old reading: every kernel, a failure at 1x."""
    points = efficacy_figures.arm_points(*solved_and_failed_pair(), over="served")
    assert points is not None
    control, treated = points
    assert 2.0**control.x == pytest.approx(2.0 ** (4.0 / 5.0))
    assert 2.0**treated.x == pytest.approx((4.0**4 * 8.0) ** (1.0 / 5.0))
    assert control.kernels == 5


def test_the_dot_row_stacks_the_success_row_between_speedup_and_cost(tmp_path: pathlib.Path) -> None:
    assert efficacy_figures.MEASURES == ("speedup", "success", "cost")
    control, treated = solved_and_failed_pair()
    frame = pd.concat([control, treated])
    stats = plot.points(control, treated)
    efficacy_figures.figure_dot_row([("Blind", "no-score", stats, frame)], tmp_path / "dots.pdf")
    assert (tmp_path / "dots.pdf").exists()


def drawn_dot_row(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, measures: tuple[str, ...]) -> Figure:
    """The figure :func:`figure_dot_row` hands to ``style.save``, kept open to be measured."""
    kept: list[Figure] = []
    monkeypatch.setattr(plotstyle, "save", lambda fig, stem, fixed=False, **options: kept.append(fig) or stem)
    control, treated = solved_and_failed_pair()
    panel = ("Blind", "no-score", plot.points(control, treated), pd.concat([control, treated]))
    efficacy_figures.figure_dot_row([panel], tmp_path / "dots.pdf", measures=measures)
    return kept[0]


def box_inches(fig: Figure, ax: Axes) -> tuple[float, float]:
    width, height = fig.get_size_inches()
    box = ax.get_position()
    return box.width * width, box.height * height


def test_dropping_the_success_row_keeps_the_width_and_every_other_box(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The success row is optional: turning it off may only shorten the canvas. A figure with and
    one without it sit in one paper, so the speed-up and cost boxes must be the same size in both.
    The success row is 0.45 of a row and the speed-up and cost rows 0.7 of one (user, 2026-09-22),
    so the success row is its MEASURE_HEIGHT share of a speed-up row (0.45/0.875 since the speed-up
    row grew 25%, user 2026-09-25)."""
    full = drawn_dot_row(tmp_path, monkeypatch, efficacy_figures.MEASURES)
    short = drawn_dot_row(tmp_path, monkeypatch, ("speedup", "cost"))
    assert full.get_size_inches()[0] == pytest.approx(short.get_size_inches()[0])
    assert full.get_size_inches()[1] > short.get_size_inches()[1]
    speedup, success, cost = full.axes[:3]
    assert box_inches(full, speedup) == pytest.approx(box_inches(short, short.axes[0]), abs=1e-3)
    assert box_inches(full, cost) == pytest.approx(box_inches(short, short.axes[1]), abs=1e-3)
    width, height = box_inches(full, speedup)
    share = efficacy_figures.MEASURE_HEIGHT["success"] / efficacy_figures.MEASURE_HEIGHT["speedup"]
    assert box_inches(full, success) == pytest.approx((width, height * share), abs=1e-3), "the success row's share"


def test_a_full_roster_mark_on_the_ceiling_is_drawn_whole() -> None:
    """A mark at N sits on the axis limit's 5% headroom, thinner than a mark in a half-height row, so
    a clipped mark printed as half a circle."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    row = efficacy_figures.ArmRow("qwen38", "HIP", "#1f77b4", arm(1.0, 2.0, 8, 8), arm(1.0, 2.0, 8, 8))
    efficacy_figures.draw_success_row(ax, [row], "^", efficacy_figures.PAPER_CONFIG, "Solved (%)")
    marks = [collection for collection in ax.collections if isinstance(collection, PathCollection)]
    assert marks and not any(mark.get_clip_on() for mark in marks)
    plt.close(fig)


def test_the_success_row_draws_its_marks_and_no_interval() -> None:
    """The roster is fixed, so the count solved is a census, not a sample (user, 2026-09-22): a Wilson
    bar under a 10/10 mark reaching down to 7 read as seven solved."""
    fig, ax = plt.subplots()
    row = efficacy_figures.ArmRow("qwen38", "HIP", "#1f77b4", arm(1.0, 2.0, 7, 10), arm(1.0, 2.0, 10, 10))
    efficacy_figures.draw_success_row(ax, [row], "^", efficacy_figures.PAPER_CONFIG, "Solved (%)")
    assert [collection for collection in ax.collections if isinstance(collection, PathCollection)]
    assert not [collection for collection in ax.collections if isinstance(collection, LineCollection)]
    plt.close(fig)


@pytest.mark.parametrize(("solved", "served", "want"), [
    ((2, 8), 8, [(-0.16, 0.25), (0.16, 1.0)]),
    ((10, 10), 10, [(-0.16, 1.0), (0.16, 1.0)]),
    ((0, 31), 40, [(-0.16, 0.0), (0.16, 0.775)]),
])  # fmt: skip
def test_a_success_mark_sits_at_the_solved_rate(
    solved: tuple[int, int], served: int, want: list[tuple[float, float]]
) -> None:
    """USER 2026-09-25: the row is the solved RATE, solved over served, so pairs with different
    rosters share one 0-100% scale; the control's hollow mark left of the column, the treated one
    right, and a 10/10 arm exactly on the 100% ceiling."""
    fig, ax = plt.subplots()
    control, treated = (arm(1.0, 2.0, count, served) for count in solved)
    row = efficacy_figures.ArmRow("qwen38", "HIP", "#1f77b4", control, treated)
    efficacy_figures.draw_success_row(ax, [row], "^", efficacy_figures.PAPER_CONFIG, "Solved (%)")
    marks = [collection for collection in ax.collections if isinstance(collection, PathCollection)]
    drawn = sorted({(float(x), float(y)) for mark in marks for x, y in mark.get_offsets()})
    assert drawn == pytest.approx(want)
    plt.close(fig)


def test_every_value_row_of_the_dot_row_carries_a_minor_grid(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """User, 2026-09-22: more minor ticks on the paper plots -- the speed-up, tasks-completed and
    token rows alike, each ruled by its own axis kind (log2 units, a count, a log10 count)."""
    kept: list[Figure] = []
    monkeypatch.setattr(plotstyle, "save", lambda fig, stem, fixed=False, **options: kept.append(fig) or stem)
    rows = observation_rows(BLIND_PAIR[1], 2.0, 100.0) + observation_rows(BLIND_PAIR[0], 4.0, 1000.0)
    frame = plot.pair_frame(pd.DataFrame(rows), [BLIND_PAIR], "no-score")
    stats = plot.points(frame[~frame.skills], frame[frame.skills])
    efficacy_figures.figure_dot_row([("Blind", "no-score", stats, frame)], tmp_path / "dots.pdf")
    for ax in kept[0].axes[:3]:
        assert [tick for tick in ax.yaxis.get_minor_ticks() if tick.gridline.get_visible()], ax.get_ylabel()


def test_a_key_too_tall_for_its_band_grows_the_canvas_instead_of_covering_the_names(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fixed band made the key the thing that gave: at text width it shrank to its floor, still did
    not fit, and was drawn over the category names."""
    kept: list[Figure] = []
    monkeypatch.setattr(plotstyle, "save", lambda fig, stem, fixed=False, **options: kept.append(fig) or stem)
    config = dataclasses.replace(efficacy_figures.PAPER_CONFIG, legend_chrome_in=0.05, legend_min_scale=1.0)
    control, treated = solved_and_failed_pair()
    panel = ("Blind", "no-score", plot.points(control, treated), pd.concat([control, treated]))
    efficacy_figures.figure_dot_row([panel], tmp_path / "dots.pdf", config=config)
    fig = kept[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    (legend,) = fig.legends
    names = [label.get_window_extent(renderer) for label in fig.axes[-1].get_xticklabels() if label.get_text()]
    assert legend.get_window_extent(renderer).y1 <= min(box.y0 for box in names)
    assert legend.get_window_extent(renderer).y0 >= 0.0


@pytest.mark.parametrize("legs", [("C", "Fortran") * 3])
def test_category_names_still_touching_on_two_lines_step_down_until_clear(legs: tuple[str, ...]) -> None:
    """Three "Fortran" placeholders two columns apart share the staggered second line and touched.
    The column is as narrow as that gets while the step-down floor (``category_min_scale``, the
    shared 6pt print floor) can still clear them; any narrower is the floor's warning, not a smaller type."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.3, 1.0))
    config = efficacy_figures.PAPER_CONFIG
    rows = [efficacy_figures.ArmRow("qwen38", leg, "#1f77b4", arm(1.0, 2.0), arm(1.0, 2.0)) for leg in legs]
    ax.set_xlim(-0.6, len(rows) - 0.4)
    efficacy_figures.draw_category_axis(ax, rows, config)
    efficacy_figures.stagger_crowded_ticks(fig, [ax], config)
    renderer = fig.canvas.get_renderer()
    size = ax.get_xticklabels()[0].get_fontsize()
    assert size < config.type_.tick_pt
    assert not plotstyle.crowded_ticks(ax, renderer, size / 3.0 * fig.dpi / 72.0)
    plt.close(fig)


def test_a_difference_label_under_the_top_tick_settles_inside_the_frame_and_off_the_marks() -> None:
    """The label starts above both intervals; where one ends at the top tick it printed across the
    frame, and holding it under the frame must not land it back on a mark."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.2, 1.0))
    row = efficacy_figures.ArmRow("qwen38", "HIP", "#1f77b4", arm(1.0, 2.0), arm(3.5, 4.0))
    efficacy_figures.draw_measure_row(ax, [row], "speedup", "^", {}, differences=frozenset({("qwen38", "HIP")}))
    plotstyle.settle_clear_labels(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    (label,) = [text for text in ax.texts if text.get_gid() == plotstyle.CLEAR_GID]
    box, frame = label.get_window_extent(renderer), ax.get_window_extent(renderer)
    assert frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
    assert not any(box.overlaps(mark) for mark in plotstyle.mark_boxes(ax))
    plt.close(fig)


@pytest.mark.parametrize(("success_row", "want"), [
    (True, ("speedup", "success", "cost")),
    (False, ("speedup", "cost")),
])  # fmt: skip
def test_without_the_success_row_speedup_and_cost_keep_their_order(success_row: bool, want: tuple[str, ...]) -> None:
    assert plot.dot_measures(argparse.Namespace(success_row=success_row)) == want


def arm(x: float, high: float, solved: int = 4, served: int = 6) -> efficacy_figures.ArmPoint:
    """An arm at ``log2`` speed-up ``x`` whose interval tops out at ``high``."""
    return efficacy_figures.ArmPoint(x, x - 1.0, high, 1e5, 5e4, 2e5, served, served, solved, served)


def test_a_success_row_nobody_was_served_is_not_a_singular_axis() -> None:
    """Every arm of a column pending leaves N = 0, and a headroom taken as a fraction of N set the
    limits to (0, 0): matplotlib warns and expands them on its own."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    unserved = arm(1.0, 2.0, solved=0, served=0)
    row = efficacy_figures.ArmRow("qwen38", "HIP", "#1f77b4", unserved, unserved)
    try:
        efficacy_figures.draw_success_row(ax, [row], "^", efficacy_figures.PAPER_CONFIG, "")
        low, high = ax.get_ylim()
    finally:
        plt.close(fig)
    assert low < 0.0 < high


def test_a_difference_label_sits_above_both_intervals_not_on_the_treated_mark() -> None:
    """Beside the bracket the label landed on the treated mark, which is only ``dodge`` away."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    row = efficacy_figures.ArmRow("qwen38", "HIP", "#1f77b4", arm(4.0, 5.0), arm(4.5, 6.0))
    efficacy_figures.draw_measure_row(ax, [row], "speedup", "^", {}, differences=frozenset({("qwen38", "HIP")}))
    (label,) = [text for text in ax.texts if text.get_text().endswith("x")]
    assert label.xy == (0, 6.0), label.xy
    assert (label.get_ha(), label.get_va()) == ("center", "bottom")
    plt.close(fig)


def test_the_success_row_runs_to_100_percent_and_carries_no_x_ticks() -> None:
    """The top tick is 100% (every served kernel solved), marked by a dashed rule; the axis runs 5%
    past it so the rule is not the frame."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    row = efficacy_figures.ArmRow("qwen38", "HIP", "#1f77b4", arm(1.0, 2.0, 2, 8), arm(1.0, 2.0, 8, 8))
    efficacy_figures.draw_success_row(ax, [row], "^", efficacy_figures.PAPER_CONFIG, "Solved (%)")
    fig.canvas.draw()
    assert list(ax.get_yticks()) == [0.0, 0.5, 1.0]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["0", "50", "100"]
    assert ax.get_ylim() == pytest.approx((-0.05, 1.05))
    (ceiling,) = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert list(ceiling.get_ydata()) == [1.0, 1.0]
    assert not ax.texts
    assert all(tick.tick1line.get_markersize() == 0.0 for tick in ax.xaxis.get_major_ticks())
    plt.close(fig)


@pytest.mark.parametrize(("solved", "served", "want"), [(8, 8, 1.0), (0, 40, 0.0), (31, 40, 0.775), (0, 0, 0.0)])
def test_the_success_rate_is_solved_over_served_and_zero_for_an_unserved_arm(
    solved: int, served: int, want: float
) -> None:
    """An arm nobody served yet has no rate to show; it reads 0, and the row skips drawing it."""
    assert efficacy_figures.success_rate(solved, served) == want


def test_a_short_cost_row_labels_one_two_and_five_of_every_decade() -> None:
    config = efficacy_figures.measure_row_config(efficacy_figures.PAPER_CONFIG, 0.98)
    assert config.token_subs == (1.0, 2.0, 5.0)


@pytest.mark.parametrize(("legs", "staggered"), [
    (("HIP", "OpenMP Offloading", "Triton") * 3, True),
    (("C",), False),
])  # fmt: skip
def test_crowded_category_ticks_alternate_two_lines_and_sparse_ones_do_not(
    legs: tuple[str, ...], staggered: bool
) -> None:
    """Nine categories on a 1.6in axis: "Triton" beside "OMP" printed as one word."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.6, 1.0))
    config = efficacy_figures.PAPER_CONFIG
    rows = [efficacy_figures.ArmRow("qwen38", leg, "#1f77b4", arm(1.0, 2.0), arm(1.0, 2.0)) for leg in legs]
    ax.set_xlim(-0.6, len(rows) - 0.4)
    efficacy_figures.draw_category_axis(ax, rows, config)
    before = [tick.get_pad() for tick in ax.xaxis.get_major_ticks()]
    efficacy_figures.stagger_crowded_ticks(fig, [ax], config)
    after = [tick.get_pad() for tick in ax.xaxis.get_major_ticks()]
    assert after[0::2] == before[0::2]
    assert (after[1::2] != before[1::2]) is staggered, (before, after)
    plt.close(fig)


@pytest.mark.parametrize(
    ("intervention", "leg", "shape"),
    [
        pytest.param("packets", "C-Terse", palette.packet_marker("caveman"), id="packets"),
        pytest.param("harness", "OpenHands", palette.harness_marker("openhands"), id="harness"),
    ],
)
def test_a_per_column_panel_never_asks_the_registry_for_its_pseudo_intervention(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    intervention: str,
    leg: str,
    shape: str,
) -> None:
    """``intervention=packets``/``harness`` names no registered treatment: each column wears its own
    packet's or harness's shape, and drawing the panel logs no palette fallback."""
    kept: list[Figure] = []
    monkeypatch.setattr(plotstyle, "save", lambda fig, stem, fixed=False, **options: kept.append(fig) or stem)
    stub = ("Per Column", intervention, pd.DataFrame(), pd.DataFrame())
    config = dataclasses.replace(efficacy_figures.PAPER_CONFIG, mark_pending=True)
    with caplog.at_level("WARNING", logger=palette.__name__):
        (column,) = efficacy_figures.dot_columns([stub], ["latest"], "model-packet", [], placeholders=[leg],
                                                 pending=["qwen38"])  # fmt: skip
        efficacy_figures.figure_dot_row([stub], tmp_path / "dots.pdf", config=config, placeholders=[leg],
                                        pending=["qwen38"])  # fmt: skip
    plt.close(kept[0])
    assert [record.getMessage() for record in caplog.records] == []
    assert column.shape == ""
    assert [(row.leg, row.shape) for row in column.rows] == [(leg, shape)]


def test_a_pending_model_gets_an_empty_category_in_registry_order() -> None:
    drawn = efficacy_figures.ArmRow(
        "qwen38", "C", "#000000", efficacy_figures.EMPTY_POINT, efficacy_figures.EMPTY_POINT
    )  # fmt: skip
    rows = efficacy_figures.pending_rows([drawn], ["kimi27sglang", "qwen38", "oss120b"], "model-packet")
    assert [row.model for row in rows] == palette.in_order(["qwen38", "kimi27sglang", "oss120b"])
    assert {row.leg for row in rows} == {"C"}
    assert efficacy_figures.pending_rows([], [], "model-packet") == []


def pending_dot_row(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, mark: bool) -> Figure:
    """A stub panel with two pending models, drawn and kept open."""
    kept: list[Figure] = []
    monkeypatch.setattr(plotstyle, "save", lambda fig, stem, fixed=False, **options: kept.append(fig) or stem)
    config = dataclasses.replace(efficacy_figures.PAPER_CONFIG, mark_pending=mark)
    stub = ("Scientific", "cpfsrc", pd.DataFrame(), pd.DataFrame())
    efficacy_figures.figure_dot_row([stub], tmp_path / "dots.pdf", config=config, pending=["oss120b,kimi27sglang"])
    return kept[0]


def test_mark_pending_draws_a_question_mark_per_pending_category_on_every_row(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fig = pending_dot_row(tmp_path, monkeypatch, mark=True)
    for ax in fig.axes:
        assert [t.get_text() for t in ax.texts if t.get_gid() == plotstyle.PENDING_GID] == ["?", "?"]
    assert plotstyle.PENDING_LABEL in [t.get_text() for legend in fig.legends for t in legend.get_texts()]


def test_without_mark_pending_a_pending_category_stays_empty(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fig = pending_dot_row(tmp_path, monkeypatch, mark=False)
    assert not any(t.get_gid() == plotstyle.PENDING_GID for ax in fig.axes for t in ax.texts)
    # The slots are kept either way: two models of one language share one tick (user, 2026-09-25)
    # and the pending one still holds its column's width.
    assert len(fig.axes[0].get_xticks()) == 1
    assert fig.axes[0].get_xlim() == pytest.approx((-0.6, efficacy_figures.GROUP_STEP + 0.6))


def test_the_key_is_centred_on_the_canvas_and_never_wider(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key is centred on the whole canvas, Y-label strip included, so it sits on the page's
    centre (user, 2026-09-25; before, it was centred on the panels alone), and it drops columns
    before it runs past the canvas."""
    fig = pending_dot_row(tmp_path, monkeypatch, mark=True)
    renderer = fig.canvas.get_renderer()
    (legend,) = fig.legends
    box = legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    assert box.width <= 1.0 + 1e-6
    assert (box.x0 + box.x1) / 2.0 == pytest.approx(0.5, abs=0.01)


def test_a_single_value_axis_is_left_unsnapped() -> None:
    """A stub row's only value is its 1x line; snapping it would set equal limits (a singular axis)."""
    fig, ax = plt.subplots()
    try:
        ax.set_ylim(-1.0, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            efficacy_figures.snap_axis_to_ticks(ax, 0.0, 0.0)
        assert ax.get_ylim() == (-1.0, 1.0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("marks", "cost", "want"),
    [
        # 2026-09-25: marks at or above 1x floor the row at 1x (log2 0), the interval cut there.
        pytest.param([0.0, 3.0], False, (0.0, 5.0), id="speedup-floored-at-1x"),
        pytest.param([-1.0, 3.0], False, (-3.0, 5.0), id="speedup-mark-below-1x-keeps-two-octaves"),
        pytest.param([1e5, 4e5], True, (2.5e4, 1.6e6), id="cost-a-factor-four-past-the-marks"),
        pytest.param([math.nan], False, (-math.inf, math.inf), id="no-mark-cuts-nothing"),
    ],
)
def test_intervals_reach_a_factor_four_past_the_outermost_marks(
    marks: list[float], cost: bool, want: tuple[float, float]
) -> None:
    """User, 2026-09-22: a few-kernel interval down to 0.004x stretched the GPU panel over twenty
    octaves and its ticks read 0.00391x; the panel now spans its marks and a bounded reach. User,
    2026-09-25: only an interval that runs that far past the marks is cut, with an arrowhead, and a
    speed-up row whose marks are all at or above 1x is cut at 1x."""
    assert efficacy_figures.interval_bounds(marks, cost, efficacy_figures.PAPER_CONFIG) == pytest.approx(want)


def test_an_interval_past_the_reach_is_cut_at_it_with_an_arrowhead() -> None:
    """The axis must not follow the interval out, and the cut end has to say the interval goes on."""
    fig, ax = plt.subplots()
    wide = efficacy_figures.ArmPoint(0.0, -9.0, 12.0, 1e5, 5e4, 2e5, 6, 6, 6, 6)
    row = efficacy_figures.ArmRow("qwen38", "HIP", "#1f77b4", arm(1.0, 2.0), wide)
    efficacy_figures.draw_measure_row(ax, [row], "speedup", "^", {})
    low, high = ax.get_ylim()
    assert high < 12.0 and low > -9.0, (low, high)
    heads = {line.get_marker() for line in ax.lines if line.get_marker() in ("^", "v")}
    assert heads == {"^", "v"}, heads
    plt.close(fig)


def few_kernel_arm(x: float, kernels: int) -> efficacy_figures.ArmPoint:
    """An arm at ``log2`` speed-up ``x`` over ``kernels`` kernels whose interval runs 20 octaves wide."""
    return efficacy_figures.ArmPoint(x, x - 10.0, x + 10.0, 1e5, 5e4, 2e5, kernels, 40, kernels, 40)


def test_a_mark_from_fewer_than_six_kernels_draws_no_interval_and_does_not_stretch_the_axis() -> None:
    """User, 2026-09-22: Qwen's GPU OpenMP control solved three kernels, its interval ran 0.19x to
    302x and was the only one in the figure cut at both ends; below six kernels (the paper's rule,
    summary.MIN_PAIRS_FOR_INTERVAL) the mark stands alone."""
    lines = {}
    limits = {}
    for kernels in (5, 6):
        fig, ax = plt.subplots()
        point = few_kernel_arm(2.0, kernels)
        row = efficacy_figures.ArmRow("qwen38", "OpenMP Offload", "#1f77b4", point, point)
        efficacy_figures.draw_measure_row(ax, [row], "speedup", "^", {}, config=efficacy_figures.PAPER_CONFIG)
        lines[kernels] = len(ax.lines)
        limits[kernels] = ax.get_ylim()
        plt.close(fig)
    assert lines[5] < lines[6], lines
    assert limits[5][1] - limits[5][0] < limits[6][1] - limits[6][0], limits


@pytest.mark.parametrize(
    ("kernels", "want"),
    [
        pytest.param(3, True, id="three-kernels-needs-the-note"),
        pytest.param(5, True, id="five-kernels-needs-the-note"),
        pytest.param(6, False, id="six-kernels-draws-its-interval"),
        pytest.param(0, False, id="an-unmeasured-slot-is-not-a-few-kernel-mark"),
    ],
)
def test_the_key_notes_a_missing_interval_only_when_a_few_kernel_mark_is_drawn(kernels: int, want: bool) -> None:
    point = few_kernel_arm(1.0, kernels)
    row = efficacy_figures.ArmRow("qwen38", "HIP", "#1f77b4", point, point)
    assert efficacy_figures.few_kernel_marks([row], efficacy_figures.PAPER_CONFIG) is want


def comparator_csv(speedups: dict[str, list[float | None]]) -> pd.DataFrame:
    """A ``comparators.py`` table: one row per (kernel, comparator), ``speedup`` None where the
    comparator has no valid run on that kernel."""
    return pd.DataFrame([
        {"kernel": f"k{index}", "comparator": name, "device": "cpu", "numba_ms": 1.0,
         "ms": None if value is None else 1.0 / value, "speedup": value}
        for name, values in speedups.items() for index, value in enumerate(values)
    ])  # fmt: skip


def test_a_comparators_mark_is_the_geomean_over_its_valid_kernels_and_its_solved_share() -> None:
    table = comparator_csv({"pluto": [2.0, 8.0, None, None]})
    (pluto,) = efficacy_figures.comparators_from_table(table, [("pluto", "C")])
    point = efficacy_figures.comparator_point(pluto)
    assert 2.0**point.x == pytest.approx(4.0)
    assert (point.solved, point.served) == (2, 4)
    assert math.isnan(point.y)  # a comparator spends no tokens


@pytest.mark.parametrize(("kernels", "interval"), [(5, False), (6, True)])
def test_a_comparator_gets_no_interval_below_min_pairs_for_interval(kernels: int, interval: bool) -> None:
    assert efficacy_figures.summary.MIN_PAIRS_FOR_INTERVAL == 6
    values = [1.5, 2.0, 3.0, 2.5, 4.0, 1.2][:kernels]
    point = efficacy_figures.comparator_point(efficacy_figures.Comparator("pluto", "C", tuple(values), 40))
    assert math.isfinite(point.x_low) is interval and math.isfinite(point.x_high) is interval
    if interval:
        want = efficacy_figures.summary.geomean_ci(values)
        assert (2.0**point.x_low, 2.0**point.x_high) == pytest.approx((want.low, want.high))


def comparator_dot_row(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Figure:
    """The blind pair plus Pluto under its C tick, drawn at paper size and kept open."""
    kept: list[Figure] = []
    monkeypatch.setattr(plotstyle, "save", lambda fig, stem, fixed=False, **options: kept.append(fig) or stem)
    control, treated = solved_and_failed_pair()
    panel = ("Blind", "no-score", plot.points(control, treated), pd.concat([control, treated]))
    pluto = efficacy_figures.Comparator("pluto", "", (2.0, 3.0, 4.0, 1.5, 2.5, 3.5), 10)
    efficacy_figures.figure_dot_row([panel], tmp_path / "dots.pdf", row_width_in=plotstyle.ICLR_TEXT_WIDTH_IN,
                                    comparators=[[pluto]])  # fmt: skip
    return kept[0]


def marks_in(ax: Axes, colour: str) -> int:
    """Filled marks of ``colour`` on ``ax``."""
    return sum(
        1
        for collection in ax.collections
        if isinstance(collection, PathCollection)
        and len(collection.get_facecolors())
        and matplotlib.colors.to_hex(collection.get_facecolors()[0]) == colour
    )


def test_a_comparator_is_drawn_on_the_speedup_and_solved_rows_and_never_on_cost(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fig = comparator_dot_row(tmp_path, monkeypatch)
    colour = palette.framework_color("pluto")
    speedup, solved, spend = fig.axes[:3]
    assert (marks_in(speedup, colour), marks_in(solved, colour), marks_in(spend, colour)) == (1, 1, 0)
    # It sits beside the models under their one delivery tick, and the key names it.
    assert len(speedup.get_xticks()) == 1
    assert "Pluto" in [text.get_text() for legend in fig.legends for text in legend.get_texts()]


def test_a_comparator_never_wears_a_packets_shape_or_shares_one() -> None:
    shapes = efficacy_figures.comparator_shapes(["ppcg_hip", "jax_gpu", "pluto", "jax_cpu"])
    packets_worn = {palette.packet_marker(key) for key in palette.hue_order("packets")}
    assert set(shapes) == {"pluto", "ppcg_hip", "jax"}  # jax_cpu and jax_gpu are one JAX
    assert len(set(shapes.values())) == 3
    assert not set(shapes.values()) & (packets_worn | {efficacy_figures.CONTROL_MARKER})


def test_the_paper_key_sets_five_columns(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """USER 2026-09-25: five columns, so the paper key takes two lines instead of three."""
    assert efficacy_figures.PAPER_CONFIG.legend_ncol == 5
    fig = comparator_dot_row(tmp_path, monkeypatch)
    renderer = fig.canvas.get_renderer()
    (legend,) = fig.legends
    columns = {round(text.get_window_extent(renderer).x0) for text in legend.get_texts()}
    assert len(columns) == min(5, len(legend.get_texts())), len(columns)


def test_the_solved_row_is_never_starred_because_the_solved_rate_is_not_tested() -> None:
    """Only the speed-up and cost legs are in the family: a speed-up verdict must not leak onto the
    solved row as if the solved rate had been tested."""
    stats = pd.DataFrame([{"model": "qwen38", "leg": "C", "score_verdict": efficacy.SIGNIFICANT,
                           "cost_verdict": efficacy.SIGNIFICANT}])  # fmt: skip
    significance = efficacy_figures.axis_significance(stats)
    row = efficacy_figures.ArmRow("qwen38", "C", "#000000", arm(1.0, 1.5), arm(2.0, 2.5))
    texts = {}
    for measure in ("speedup", "success"):
        fig, ax = plt.subplots()
        efficacy_figures.draw_measure_row(ax, [row], measure, "D", significance)
        texts[measure] = [text.get_text() for text in ax.texts]
        plt.close(fig)
    assert efficacy_figures.SCORE_SIG_MARK in texts["speedup"]
    assert texts["success"] == [], texts["success"]


def test_the_cost_row_is_priced_with_the_billed_card_unless_told_otherwise() -> None:
    control, treated = solved_and_failed_pair()
    cached = {"tokens_cached_input": 1000.0}
    control = control.assign(**{k: np.where(control.record == "task", v, np.nan) for k, v in cached.items()})
    treated = treated.assign(**{k: np.where(treated.record == "task", v, np.nan) for k, v in cached.items()})
    billed = efficacy_figures.paired_kernels(control, treated)
    effective = efficacy_figures.paired_kernels(control, treated, card=cost.resolve("effective"))
    # billed charges the 1000 cached tokens at a tenth; effective charges them nothing.
    assert (billed.control_tokens - effective.control_tokens).to_numpy() == pytest.approx(100.0)
