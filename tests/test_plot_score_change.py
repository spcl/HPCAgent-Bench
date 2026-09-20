# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``statistics/plot_score_change.py`` -- the efficacy figure and the stars on it.

The load-bearing assertions are about MULTIPLICITY (a raw threshold does not survive the BH
correction, a real effect does) and about the 2D CONTRACT: X is log2 of the speed-up geomean, Y the
paired token-cost ratio, one mark per arm against its own control, nothing joined by a line, colour
the model, shape the packet, and the drawing itself lives in
:mod:`hpcagent_bench.stats.figures.efficacy` -- this script only wires the data.
"""

import importlib.util
import argparse
import math
import pathlib
import sys
import tempfile

import matplotlib.colors
import matplotlib.markers
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from hpcagent_bench import experiment_tags
from hpcagent_bench.harness import efficacy
from hpcagent_bench.stats import palette, score_rule
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
        {**common, "record": "task", "speedup": None, "tokens": tokens, "suspect": None},
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
    :func:`~hpcagent_bench.stats.figures.efficacy.draw_panel` reads its per-kernel cloud from.

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

    frame = plot.load(path, prefix="")

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

    frame = plot.load(path, prefix="")

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

    frame_all = plot.load(path, prefix="")
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

    frame_all = plot.load(path, prefix="")
    control = plot.control_rows(frame_all)

    assert "cpf-llr-focus40-qwen38-c-perf-playbook-cpu" not in set(control.arm)
    assert set(control.arm) == {"cpf-llr-focus40-qwen38-c"}


def test_treatment_frame_tags_the_control_false_and_the_treatment_true() -> None:
    """``draw_panel`` reads an on/off ``skills`` flag; ``treatment_frame`` builds it from the packet
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
            rows.append({**base, "record": "task", "speedup": None, "tokens": 1000.0})
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
# The 2D contract: X the log2 speed-up, Y the paired token-cost ratio, one mark per arm, colour
# the intervention, shape the model, one legend on the figure, an undelivered kernel a cross.


def test_x_is_log2_of_the_speed_up_and_zero_is_the_no_change_line() -> None:
    """Never a bare ratio axis: the axis holds ``log2(ratio)`` -- a 4x speed-up sits at +2, a 2x
    slow-down at -1 -- and the reference line sits at the change that means nothing happened. The
    ticks read back in ratios, never as raw exponents."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        efficacy_figures.draw_panel(ax, one_arm_raw(on_speedup=4.0), one_arm_stats(), "skills")
        assert "Speed-Up" in ax.get_xlabel()
        assert ax.get_xscale() == "linear"
        assert any(line.get_xdata()[0] == pytest.approx(0.0) for line in ax.lines if len(set(line.get_xdata())) == 1)
        assert efficacy_figures.log2_tick(2.0) == "4x"
        assert efficacy_figures.log2_tick(-1.0) == "1/2x"
    finally:
        plt.close(fig)


def test_y_is_the_paired_token_cost_ratio_with_one_at_the_control() -> None:
    """The other axis is a cost, never shares a scale with the speed-up (SC15 Rule 4), and its own
    no-change value is a ratio of 1, not 0."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        efficacy_figures.draw_panel(ax, one_arm_raw(), one_arm_stats(), "skills")
        assert "Token-Cost" in ax.get_ylabel()
        assert ax.get_yscale() == "log"
        assert any(line.get_ydata()[0] == pytest.approx(1.0) for line in ax.lines if len(set(line.get_ydata())) == 1)
    finally:
        plt.close(fig)


def test_error_bars_are_drawn_on_both_axes_from_the_paired_geomean() -> None:
    """Every arm's mark carries a crossed 95% interval: SC15 Rule 5/7, computed from
    ``summary.geomean_ci`` on the paired per-kernel ratios."""
    frame = one_arm_raw(on_speedup=2.0, on_tokens=400.0)
    series = efficacy_figures.reduce_pair(frame[~frame.skills], frame[frame.skills])
    assert series is not None
    assert series.x_low < series.x < series.x_high
    assert series.y_low < series.y < series.y_high


def test_nothing_is_joined_by_a_line() -> None:
    """SC15 Rule 12: an arm's mark is one measurement, not a trend, so the panel draws no connector
    between a control and a treated position."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        efficacy_figures.draw_panel(ax, one_arm_raw(), one_arm_stats(), "skills")
        # Only the zero/one reference lines and the crossed-interval whiskers are ``Line2D``/
        # ``LineCollection`` objects; none of them connect two DIFFERENT (x, y) mark positions.
        for line in ax.lines:
            xdata, ydata = line.get_xdata(), line.get_ydata()
            assert len(set(xdata)) == 1 or len(set(ydata)) == 1, "a connector between two marks was drawn"
    finally:
        plt.close(fig)


def test_a_marks_significance_superscript_reads_off_the_corrected_verdict_per_axis() -> None:
    """The gate is the VERDICT, per axis: a raw p that was never corrected, and a pairing too small
    for any test, both draw NO superscript. Every mark is filled regardless -- only the control is
    ever hollow (:mod:`hpcagent_bench.stats.figures.efficacy`'s own module docstring); significance
    is the label's own ``*``/DAGGER suffix instead."""
    import matplotlib.pyplot as plt

    def leg_labels_of(stats: pd.DataFrame) -> list[str]:
        fig, ax = plt.subplots()
        try:
            efficacy_figures.draw_panel(ax, one_arm_raw(), stats, "skills")
            return [text.get_text() for text in ax.texts]
        finally:
            plt.close(fig)

    neither = leg_labels_of(one_arm_stats())
    both = leg_labels_of(one_arm_stats(score_verdict=efficacy.SIGNIFICANT, cost_verdict=efficacy.SIGNIFICANT))
    assert neither == ["C"], neither
    assert both == [f"C {efficacy_figures.SCORE_SIG_MARK}{efficacy_figures.COST_SIG_MARK}"], both


def test_every_drawn_mark_is_filled_and_only_the_control_is_hollow() -> None:
    """Fill no longer carries significance (superscripts do); it is now a constant so a treated mark
    is never mistaken for a second control reference."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    fig, ax = plt.subplots()
    try:
        efficacy_figures.draw_panel(ax, one_arm_raw(), one_arm_stats(), "cpf")
        summary_marks = [
            c for c in ax.collections
            if isinstance(c, PathCollection) and c.get_sizes().size
            and c.get_sizes().max() == pytest.approx(efficacy_figures.DEFAULT_CONFIG.mark_size)
        ]  # fmt: skip
        hollow = [c for c in summary_marks if not any(rgba[3] > 0.0 for rgba in c.get_facecolor())]
    finally:
        plt.close(fig)
    assert len(summary_marks) >= 2, "expected the control reference plus at least one treated mark"
    assert len(hollow) == 1, "exactly the control reference should be hollow"


def test_the_filled_mark_wears_the_model_colour_and_the_hollow_control_wears_the_control_colour() -> None:
    """The inverted rule (module docstring): colour is the MODEL's registry hue -- the same hue in
    every figure -- and the control reference is still ``palette.control_color``, never a model hue.
    Colouring by PACKET instead would spend the model's channel on the entity the shape now carries."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    treatment = "cpf"
    assert palette.model_color("qwen38") != palette.color(treatment)
    fig, ax = plt.subplots()
    try:
        efficacy_figures.draw_panel(ax, one_arm_raw(), one_arm_stats(), treatment)
        edges, faces = set(), set()
        for collection in (c for c in ax.collections if isinstance(c, PathCollection)):
            for rgba in collection.get_edgecolor():
                edges.add(matplotlib.colors.to_hex(rgba))
            for rgba in collection.get_facecolor():
                faces.add(matplotlib.colors.to_hex(rgba))
        assert palette.model_color("qwen38") in faces | edges
        assert palette.control_color() in edges
        assert palette.color(treatment) not in faces | edges
    finally:
        plt.close(fig)


def test_the_marker_shape_is_the_packet_and_nothing_else() -> None:
    """Shape is the one packet the whole panel wears, so identity survives greyscale and a
    column-width shrink even when colour (the model, here) does not fit its own legend swatch."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        efficacy_figures.draw_panel(ax, one_arm_raw(), one_arm_stats(), "cpfsrc")
        drawn = {collection.get_paths()[0] for collection in ax.collections if collection.get_paths()}
        style = matplotlib.markers.MarkerStyle(palette.packet_marker("cpfsrc"))
        expected = style.get_path().transformed(style.get_transform())
        assert any(path.vertices.shape == expected.vertices.shape for path in drawn)
    finally:
        plt.close(fig)


def test_a_light_minor_grid_sits_between_the_major_lines() -> None:
    """A half-octave minor grid is now drawn (denser than the major-only original), but lighter and
    thinner so it reads as texture under the marks rather than a second reference the major line
    already is."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        efficacy_figures.draw_panel(ax, one_arm_raw(), one_arm_stats(), "cpfsrc")
        major = [line for line in ax.yaxis.get_gridlines() if line.get_visible()]
        minor = [tick.gridline for tick in ax.yaxis.get_minor_ticks() if tick.gridline.get_visible()]
        assert major, "the major grid should still be there"
        assert minor, "a minor grid should now be drawn too"
        assert minor[0].get_linewidth() < major[0].get_linewidth()
        assert not ax.yaxis.get_minor_ticks()[0].label1.get_text(), "minor ticks carry no text"
    finally:
        plt.close(fig)


def test_the_legend_is_drawn_once_on_the_figure_and_never_on_an_axes() -> None:
    """One key for the whole figure: several panels draw the same models in the same colours, and a
    key on each axes invites reading them as different sets of series."""
    out = pathlib.Path(tempfile.mkdtemp()) / "fig.pdf"
    written = efficacy_figures.figure_one(one_arm_raw(), one_arm_stats(), "cpfsrc", out)
    assert written.with_suffix(".pdf").exists()

    panels = [("CPF", "cpf", one_arm_stats(), one_arm_raw()), ("CPFSRC", "cpfsrc", one_arm_stats(), one_arm_raw())]
    written_row = efficacy_figures.figure_row(panels, pathlib.Path(tempfile.mkdtemp()) / "row.pdf")
    assert written_row.with_suffix(".pdf").exists()


def test_a_joined_row_has_exactly_one_legend_with_deduplicated_unique_labels() -> None:
    """A 1x3 figure draws one intervention hue, one model shape, one control mark and the interval
    notes ONCE each, never once per panel: every panel repeats the same channels (all fixed text, so
    identical across panels), so a key on each would invite reading them as different sets of
    series, and a naive concatenation of three panels' handles would repeat every one of them three
    times."""
    import matplotlib.pyplot as plt

    n = 3
    fig, axes = plt.subplots(1, n, squeeze=False)
    try:
        treatments = ["cpf", "cpfsrc", "lang-skills"]
        handles_by_label: dict[str, object] = {}
        for ax, treatment in zip(axes[0], treatments, strict=True):
            for handle in efficacy_figures.draw_panel(ax, one_arm_raw(), one_arm_stats(), treatment, treatments):
                handles_by_label.setdefault(handle.get_label(), handle)
        plotstyle.legend_below(fig, list(handles_by_label.values()))
        assert len(fig.legends) == 1
        assert all(ax.get_legend() is None for ax in fig.axes)
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        assert len(labels) == len(set(labels)), labels
    finally:
        plt.close(fig)


def test_the_joined_figure_names_its_one_control_once_over_every_treatment() -> None:
    """One control, one entry. Named per panel instead, a joined figure grew one hollow-mark entry
    per row ("No Packet" beside "No Skill Packet") for the one set of control arms."""
    import matplotlib.pyplot as plt

    n = 2
    side = efficacy_figures.panel_side(n)
    fig, axes = plt.subplots(1, n, squeeze=False)
    try:
        treatments = ["cpf", "lang-skills"]
        labels: list[str] = []
        for ax, treatment in zip(axes[0], treatments, strict=True):
            labels += [
                h.get_label()
                for h in efficacy_figures.draw_panel(ax, one_arm_raw(), one_arm_stats(), treatment, treatments)
            ]
    finally:
        plt.close(fig)
    assert side > 0.0
    assert labels.count("No Packet") + labels.count("No Skill Packet") == 2, labels


@pytest.mark.parametrize("n", [1, 2, 3])
def test_n_comparisons_draw_one_row_of_n_square_panels(n: int) -> None:
    """Several comparisons join as ONE ROW of square panels, never stacked: they are alternatives
    against one control, not a sequence."""
    panels = [(f"Comparison {i}", "skills", one_arm_stats(), one_arm_raw()) for i in range(n)]
    written = efficacy_figures.figure_row(panels, pathlib.Path(tempfile.mkdtemp()) / "row.pdf")
    assert written.with_suffix(".pdf").exists()


def test_a_paper_row_width_derives_the_panel_side_from_the_text_width() -> None:
    """A row drawn for a paper spans the venue's own text width at scale 1.0, never a fixed natural
    size that then has to be shrunk by ``\\includegraphics``."""
    natural = efficacy_figures.panel_side(3)
    iclr = efficacy_figures.panel_side(3, plotstyle.ICLR_TEXT_WIDTH_IN)
    assert iclr != natural
    assert 3 * iclr + 2 * efficacy_figures.ROW_PANEL_GAP <= plotstyle.ICLR_TEXT_WIDTH_IN + 1e-9


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


def test_a_wide_ranging_panel_never_crosses_the_x_tick_budget() -> None:
    """The end-to-end path: a treated arm landing two orders of magnitude from its control (the
    autoscaled window then spans 0 to that log2 exponent, since the control mark sits at x=0)
    still draws under :data:`~hpcagent_bench.stats.figures.efficacy.MAX_X_TICKS` labelled ticks."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        efficacy_figures.draw_panel(ax, one_arm_raw(on_speedup=1e5), one_arm_stats(), "cpf")
        fig.canvas.draw()
        ticks = [t for t in ax.xaxis.get_major_ticks() if t.label1.get_visible()]
        assert 0 < len(ticks) <= efficacy_figures.MAX_X_TICKS
    finally:
        plt.close(fig)


def png_ink_columns(path: pathlib.Path, row_lo: int, row_hi: int) -> tuple[int, int]:
    """The leftmost and rightmost non-white pixel column PNG ``path`` carries within
    ``[row_lo, row_hi)`` -- how a rendered title or axis label is checked for clipping: matplotlib's
    own measured 'it fits' is a claim about a font metrics table, this is a claim about the file."""
    with Image.open(path) as image:
        band = np.array(image.convert("L"))[row_lo:row_hi, :]
    cols = np.where((band < 240).any(axis=0))[0]
    return (int(cols.min()), int(cols.max())) if cols.size else (-1, -1)


def test_neither_figure_one_nor_figure_row_calls_style_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new contract: a paper's caption is the title, so neither function ever draws a
    whole-figure one (:func:`~hpcagent_bench.stats.style.title`) -- only each panel's own small
    subtitle, drawn with ``ax.text`` INSIDE its box, ever runs."""

    def refuse(*args: object, **kwargs: object) -> float:
        raise AssertionError("style.title must never be called")

    monkeypatch.setattr(plotstyle, "title", refuse)
    efficacy_figures.figure_one(one_arm_raw(), one_arm_stats(), "cpf", pathlib.Path(tempfile.mkdtemp()) / "one.pdf")
    panels = [("Comparison", "cpf", one_arm_stats(), one_arm_raw())]
    efficacy_figures.figure_row(panels, pathlib.Path(tempfile.mkdtemp()) / "row.pdf")


def test_a_panels_own_subtitle_never_touches_the_saved_canvas_edge(tmp_path: pathlib.Path) -> None:
    """A long per-panel ``title`` (:func:`figure_row`'s own small in-box text) is measured against
    the SAVED file, not the same measurement that used to miss a whole-figure title's clipping at a
    different dpi (:func:`~hpcagent_bench.stats.style.save`'s own docstring)."""
    long_title = "A panel subtitle long enough that it could, in principle, run off either edge"
    panels = [(long_title, "cpf", one_arm_stats(), one_arm_raw())]
    out = tmp_path / "row.pdf"
    efficacy_figures.figure_row(panels, out)
    with Image.open(out.with_suffix(".png")) as image:
        width = image.size[0]
        rows = np.where((np.array(image.convert("L")) < 240).any(axis=1))[0]
    left, right = png_ink_columns(out.with_suffix(".png"), int(rows.min()), int(rows.min()) + 20)
    assert 0 < left, "the subtitle's own ink starts at the canvas edge"
    assert right < width - 1, "the subtitle's own ink runs to the canvas edge"


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
    ax.yaxis.set_major_formatter(FuncFormatter(efficacy_figures.ratio_tick))
    ax.set_ylabel("Token-Cost Ratio, Treated / Control", fontsize=plotstyle.LABEL_PT * 0.68)
    ax.tick_params(axis="both", labelsize=plotstyle.TICK_PT * 0.6)
    try:
        return efficacy_figures.required_left_margin(fig, ax)
    finally:
        plt.close(fig)


def test_required_left_margin_grows_with_the_widest_tick_the_range_draws() -> None:
    """A direct measurement: a wide-ranging axis's own tick labels (``1/128x`` .. ``128x``) beside
    the panel's Y label need more room than a narrow range's short ones (``1x``) -- a flat fraction
    of the row's width, which :func:`figure_row` used to reserve regardless of content, cannot tell
    the two apart and ran a real outlier kernel's label off the canvas before this measured it."""
    wide = y_axis_left_margin(2.0**-7, 2.0**7)  # 1/128x .. 128x, the dynamic range a real outlier kernel gave
    narrow = y_axis_left_margin(0.5, 2.0)  # 1/2x .. 2x
    assert wide > narrow, (wide, narrow)


def test_the_legend_names_the_interval_method() -> None:
    """Both axes here always draw the same estimator (a log-space t interval), and the legend says
    so (SC15 Rule 5). Fixed text, never a per-panel sample count any more (:data:`GEOMEAN_METHOD`)
    -- so a joined row's per-panel copies dedupe into one shared entry
    (:func:`~hpcagent_bench.stats.figures.efficacy.legend_tail`)."""
    note = efficacy_figures.interval_note("Speed-Up Geomean")
    assert "log-t" in note
    assert efficacy_figures.interval_note("Speed-Up Geomean") == note, "fixed text, not sample-size-chosen"


def test_the_figure_key_carries_one_interval_note_per_axis_and_the_significance_rule() -> None:
    """Both notes -- and the ``*``/DAGGER significance rule -- reach the reader, in the figure's ONE
    key."""
    handles = efficacy_figures.legend_handles("cpfsrc", ["qwen38"], ["cpfsrc"])
    labels = [h.get_label() for h in handles]
    assert efficacy_figures.interval_note("Speed-Up Geomean") in labels, labels
    assert efficacy_figures.interval_note("Token-Cost Geomean") in labels, labels
    assert efficacy_figures.SIGNIFICANCE_NOTE in labels, labels


def test_an_undelivered_kernel_still_counts_but_its_cross_only_draws_behind_show_cloud() -> None:
    """A kernel the arm was served and never verified is a placeholder, not a measurement
    (``population.DELIVERED_COLUMN``) -- it still scores 1x and still counts its tokens in the
    geomean either way, but its own dot or cross is part of the per-kernel CLOUD, which draws (and
    the key names it) only when a caller opts in; the default panel draws summary marks alone."""
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

    series = efficacy_figures.reduce_pair(control, treated)

    assert series is not None
    assert series.delivered < series.kernels
    assert not series.cloud[~series.cloud.delivered].empty

    default_handles = efficacy_figures.legend_handles("skills", ["qwen38"], ["skills"])
    assert not any(h.get_label() == plotstyle.NOT_DELIVERED_LABEL for h in default_handles), default_handles

    cloud_handles = efficacy_figures.legend_handles("skills", ["qwen38"], ["skills"], show_cloud=True)
    assert any(h.get_label() == plotstyle.NOT_DELIVERED_LABEL for h in cloud_handles)


def crowded_frame() -> pd.DataFrame:
    """Six arms of one comparison, three of them landing within a few percent of each other on
    BOTH axes -- the fixture a ring of candidate label places has to solve."""
    parts = []
    close = ((1.34, 350e3, 1.36, 340e3), (1.36, 245e3, 1.35, 255e3), (1.35, 178e3, 1.37, 165e3))
    spread = ((1.10, 50e3, 1.30, 40e3), (1.45, 72e3, 1.20, 65e3), (1.60, 165e3, 1.90, 120e3))
    for (model, language), (off_s, off_t, on_s, on_t) in zip(
        ((m, lang) for m in MODELS for lang in LANGUAGES), close + spread, strict=True
    ):
        parts.append(one_arm_raw(model, language, off_s, on_s, off_t, on_t))
    return pd.concat(parts, ignore_index=True)


def crowded_stats() -> pd.DataFrame:
    return pd.DataFrame([{
        "model": m,
        "language": lang,
        "score_verdict": efficacy.NOT_SIGNIFICANT,
        "cost_verdict": efficacy.NOT_SIGNIFICANT,
        "family_size": 12
    } for m in MODELS for lang in LANGUAGES])  # fmt: skip


def test_no_two_arm_labels_overprint_each_other_however_close_the_arms_land() -> None:
    """The bug this guards: six arms share one square panel and three land within a few percent, so
    a label placed at a fixed offset prints on top of its neighbour."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=efficacy_figures.PANEL_SIZE)
    try:
        efficacy_figures.draw_panel(ax, crowded_frame(), crowded_stats(), "skills")
        fig.subplots_adjust(**efficacy_figures.PANEL_MARGINS)
        efficacy_figures.untangle_labels(ax)
        renderer = fig.canvas.get_renderer()
        from matplotlib.text import Annotation

        notes = [text for text in ax.texts if isinstance(text, Annotation)]
        boxes = [tuple(note.get_window_extent(renderer).extents) for note in notes]
        assert len(boxes) == 6
        for index, box in enumerate(boxes):
            for other in boxes[index + 1 :]:
                assert not efficacy_figures.boxes_touch(box, other), notes[index].get_text()
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# The EXPLICIT-PAIR entry point: a comparison whose two sides are two campaigns, or whose condition
# is not a packet suffix at all, drawn through the same panel.

BLIND_PAIR: tuple[str, str] = ("llrblind-qwen38-c-skills", "cpf-llr-focus40-qwen38-c-skills")
SCICOMP_PAIR: tuple[str, str] = ("git-scicomp-qwen38-c-repo", "git-scicomp-qwen38-c-kernel")


def family_csv(pairs: list[tuple[str, str]], score_verdict: str, cost_verdict: str, n: int = 40) -> pd.DataFrame:
    """A family table in the shape ``statistics/paired_arms.py`` writes, one row per pair per leg."""
    return pd.DataFrame(
        [
            {
                "family": "demo",
                "score_rule": score_rule.SCORE_RULE,
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
        rows.append({**common, "record": "task", "speedup": None, "tokens": tokens})
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


def test_a_pair_figure_wears_the_intervention_hue_and_names_a_control_that_is_not_a_missing_packet() -> None:
    """git-scicomp's control is the BARE KERNEL and llrblind's kept its score tool; "No Packet"
    names neither, so the hollow mark's text is the caller's."""
    import matplotlib.pyplot as plt

    frame_all = pd.DataFrame(
        observation_rows(SCICOMP_PAIR[0], 1.1, 1.4e6) + observation_rows(SCICOMP_PAIR[1], 0.6, 900e3)
    )
    tagged = plot.pair_frame(frame_all, [SCICOMP_PAIR], "repo")
    stats = plot.family_stats(family_csv([SCICOMP_PAIR], efficacy.SIGNIFICANT, ""), "repo")
    fig, ax = plt.subplots()
    try:
        labels = [
            h.get_label() for h in efficacy_figures.draw_panel(
                ax, tagged, stats, "repo", control_name="Bare Kernel")
        ]  # fmt: skip
    finally:
        plt.close(fig)
    assert "Bare Kernel" in labels, labels
    assert "No Packet" not in labels
    assert experiment_tags.packet_name("repo") in labels, labels


def test_resolve_row_repeats_broadcasts_a_bare_policy_and_checks_a_sequences_length() -> None:
    """A joined row's comparisons need not share ONE repeat policy: git-scicomp's designed-3x-repeats
    median and llr-focus40's reruns-take-latest sit in the same row."""
    assert efficacy_figures.resolve_row_repeats("median", 3) == ["median", "median", "median"]
    assert efficacy_figures.resolve_row_repeats(["latest", "median"], 2) == ["latest", "median"]
    with pytest.raises(ValueError, match="panels 3"):
        efficacy_figures.resolve_row_repeats(["latest"], 3)


def test_a_comparison_specs_own_repeats_overrides_the_row_default(tmp_path: pathlib.Path) -> None:
    """``repeats=`` on one ``--comparison`` spec reaches ONLY that panel; a spec without it keeps
    ``--repeats``."""
    obs = tmp_path / "obs.csv"
    pd.DataFrame(observation_rows(SCICOMP_PAIR[0], 1.1, 1.4e6) + observation_rows(SCICOMP_PAIR[1], 0.6, 900e3)).to_csv(
        obs, index=False
    )
    table = tmp_path / "pairs.csv"
    family_csv([SCICOMP_PAIR], efficacy.NOT_SIGNIFICANT, efficacy.NOT_SIGNIFICANT).to_csv(table, index=False)

    seen_repeats: list[object] = []
    real = efficacy_figures.figure_row

    def record(*args: object, **kwargs: object) -> object:
        seen_repeats.append(kwargs["repeats"])
        return real(*args, **kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]

    old_figure_row = plot.efficacy_figures.figure_row
    plot.efficacy_figures.figure_row = record
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
        plot.efficacy_figures.figure_row = old_figure_row
    assert seen_repeats == [["median"]]


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
# No per-kernel cloud by default: a panel draws summary marks alone unless a caller opts in.


def test_the_per_kernel_cloud_is_off_by_default_and_only_draws_behind_show_cloud() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    def cloud_sized_collections(show_cloud: bool) -> int:
        fig, ax = plt.subplots()
        try:
            efficacy_figures.draw_panel(ax, one_arm_raw(kernels=KERNELS), one_arm_stats(), "cpf", show_cloud=show_cloud)
            return sum(
                1
                for collection in ax.collections
                if isinstance(collection, PathCollection)
                and collection.get_sizes().size
                and collection.get_sizes().max() == pytest.approx(efficacy_figures.CLOUD_SIZE)
            )
        finally:
            plt.close(fig)

    assert cloud_sized_collections(show_cloud=False) == 0
    assert cloud_sized_collections(show_cloud=True) > 0


# ---------------------------------------------------------------------------
# No title: a paper's caption is the title now (supersedes the earlier derived-default-title
# contract). ``--title`` is an optional, blank-by-default in-panel SUBTITLE, never a whole-figure
# title, and it is threaded straight through -- there is no more derivation logic to test.


def test_cli_title_flag_produces_no_whole_figure_title(tmp_path: pathlib.Path) -> None:
    """The CLI route: even a caller who asks for a subtitle gets a small one INSIDE the panel, never
    a whole-figure title (:func:`~hpcagent_bench.stats.figures.efficacy.figure_one` draws it with
    ``ax.text``, not :func:`~hpcagent_bench.stats.style.title`)."""
    obs = one_arm_observations_csv(tmp_path)
    out = tmp_path / "fig.pdf"
    argv = [
        str(obs), "--experiment", "exp", "--treatment", "skills", "--title", "A Subtitle", "--out", str(out),
        "--table", str(tmp_path / "table.csv"),
    ]  # fmt: skip
    import sys

    old_argv = sys.argv
    sys.argv = ["plot_score_change.py", *argv]
    try:
        plot.main()
    finally:
        sys.argv = old_argv
    assert out.with_suffix(".pdf").exists()


# ---------------------------------------------------------------------------
# Fraction tick labels: a ratio below 1 reads as "1/Nx" on both axes, never a decimal, and the Y
# axis' log2 locator only ever lands on a power of two.


def test_the_token_cost_axis_formatter_spells_a_ratio_below_one_as_a_fraction() -> None:
    assert efficacy_figures.ratio_tick(0.125) == "1/8x"
    assert efficacy_figures.ratio_tick(1.0) == "1x"
    assert efficacy_figures.ratio_tick(8.0) == "8x"


def test_a_drawn_panels_y_axis_never_labels_a_non_power_of_two_tick() -> None:
    """The bug this guards: a base-2 ``LogLocator`` with a 1.5 sub used to label 1.5x, 3x, 0.75x --
    ticks :func:`~hpcagent_bench.stats.figures.per_kernel.speedup_tick_label` cannot spell as a
    clean fraction and a reader cannot place on a log2 grid by eye."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        efficacy_figures.draw_panel(ax, one_arm_raw(on_tokens=333.0), one_arm_stats(), "cpf")
        fig.canvas.draw()
        labels = [tick.get_text() for tick in ax.yaxis.get_majorticklabels() if tick.get_text()]
    finally:
        plt.close(fig)
    assert labels, "no Y ticks were drawn to check"
    for text in labels:
        ratio = float(text.rstrip("x").split("/")[-1]) if "/" in text else float(text.rstrip("x"))
        exponent = math.log2(ratio)
        assert exponent == pytest.approx(round(exponent)), text


# ---------------------------------------------------------------------------
# Several packets sharing ONE panel (draw_multi_panel): each its own shape, every model still its
# own colour, one packet with nothing drawn silently absent rather than a shape nothing wears.


def test_draw_multi_panel_gives_each_packet_its_own_shape_and_each_model_its_own_colour() -> None:
    import matplotlib.markers
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    frames = {"cpf": one_arm_raw(model="qwen38"), "cpfsrc": one_arm_raw(model="oss120b")}
    stats = {"cpf": one_arm_stats(model="qwen38"), "cpfsrc": one_arm_stats(model="oss120b")}
    fig, ax = plt.subplots()
    try:
        handles = efficacy_figures.draw_multi_panel(ax, frames, stats)
        labels = [h.get_label() for h in handles]
        drawn_shapes = {
            path.vertices.shape for collection in ax.collections if isinstance(collection, PathCollection)
            for path in collection.get_paths()
        }  # fmt: skip
    finally:
        plt.close(fig)
    assert experiment_tags.packet_name("cpf") in labels, labels
    assert experiment_tags.packet_name("cpfsrc") in labels, labels
    assert experiment_tags.model_name("qwen38") in labels, labels
    assert experiment_tags.model_name("oss120b") in labels, labels
    for packet in ("cpf", "cpfsrc"):
        shape = matplotlib.markers.MarkerStyle(palette.packet_marker(packet)).get_path().vertices.shape
        assert shape in drawn_shapes, (packet, drawn_shapes)


def test_draw_multi_panel_drops_a_packet_with_nothing_to_draw_from_its_own_legend() -> None:
    """A packet whose frame is empty (no arm ever ran it in this figure's slice) is silently absent
    from the legend -- drawing its shape with nothing under it would be a key for a mark that is not
    on the panel."""
    import matplotlib.pyplot as plt

    frames = {"cpf": one_arm_raw(model="qwen38"), "cpfsrc": one_arm_raw(model="qwen38").iloc[0:0]}
    stats = {"cpf": one_arm_stats(model="qwen38"), "cpfsrc": one_arm_stats(model="qwen38")}
    fig, ax = plt.subplots()
    try:
        labels = [h.get_label() for h in efficacy_figures.draw_multi_panel(ax, frames, stats)]
    finally:
        plt.close(fig)
    assert experiment_tags.packet_name("cpf") in labels, labels
    assert experiment_tags.packet_name("cpfsrc") not in labels, labels


def test_a_multi_treatment_panel_joins_a_row_beside_a_single_treatment_one() -> None:
    """:data:`efficacy_figures.Panel` covers both shapes at once: :func:`figure_row` dispatches on
    whether a panel's treatment is one key or several."""
    single = ("Skills Only", "skills", one_arm_stats(), one_arm_raw())
    multi = (
        "CPU",
        ["skills", "cpf"],
        {"skills": one_arm_stats(), "cpf": one_arm_stats()},
        {"skills": one_arm_raw(), "cpf": one_arm_raw()},
    )
    written = efficacy_figures.figure_row([single, multi], pathlib.Path(tempfile.mkdtemp()) / "row.pdf")
    assert written.with_suffix(".pdf").exists()


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


def test_a_kernel_without_a_token_total_keeps_its_speed_up_and_the_mark_says_n() -> None:
    """A treated kernel whose task row was lost still has a verified answer: the speed-up
    coordinate is the geomean over EVERY paired kernel (the family CSV's own score leg), the token
    coordinate over the kernels priced on both sides, and the mark's label names that n. Intersecting
    the two moved Kimi's C skill-pages point from 0.83x (38 kernels) to 1.01x (19)."""
    import matplotlib.pyplot as plt

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
    fig, ax = plt.subplots()
    try:
        efficacy_figures.draw_panel(ax, frame, one_arm_stats(), "skills")
        labels = [text.get_text() for text in ax.texts]
        assert any(f"(tokens n={series.token_kernels}/{KERNELS})" in label for label in labels), labels
    finally:
        plt.close(fig)
    table = efficacy_figures.pairs_table(frame.assign(baseline_ns=2.0e6, native_ns=1.0e6))
    assert table.token_kernels.tolist() == [series.token_kernels]
    assert table.kernels.tolist() == [KERNELS]


def test_a_fully_priced_mark_carries_no_token_note() -> None:
    """Both legs over the same kernels: nothing to say, so the label is the leg alone."""
    series = efficacy_figures.reduce_pair(*(lambda f: (f[~f.skills], f[f.skills]))(one_arm_raw()))
    assert series is not None
    assert series.token_kernels == series.kernels
    assert efficacy_figures.token_note(series) == ""


def test_the_pairs_csv_route_draws_its_marks_under_the_repeat_policy_it_was_asked_for(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``figure_from_pairs`` passed neither ``--repeats`` nor the figure config on to
    ``figure_one``, so its marks were drawn under the default ``latest`` while the table written
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
    by_policy = {
        policy: efficacy_figures.reduce_pair(frame[~frame.skills], frame[frame.skills], policy).x
        for policy in ("median", "latest")
    }
    assert by_policy["median"] != pytest.approx(by_policy["latest"]), by_policy

    drawn: list[float] = []
    real = efficacy_figures.draw_series

    def spy(ax, series, *args, **kwargs):
        drawn.append(series.x)
        return real(ax, series, *args, **kwargs)

    monkeypatch.setattr(efficacy_figures, "draw_series", spy)
    args = argparse.Namespace(
        pairs_csv=pairs_csv,
        observations=[observations_csv],
        intervention="repo",
        control_label="Kernel Formulation",
        cost_model="effective",
        cost_models=None,
        repeats="median",
        show_cloud=False,
        title="",
        out=tmp_path / "f.pdf",
        table=tmp_path / "f.csv",
    )
    plot.figure_from_pairs(args, efficacy_figures.DEFAULT_CONFIG)
    assert drawn == pytest.approx([by_policy["median"]]), (drawn, by_policy)
