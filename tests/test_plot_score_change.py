# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``scripts/plot_score_change.py`` -- the efficacy figure and the stars on it.

The load-bearing assertions are about MULTIPLICITY (a raw threshold does not survive the BH
correction, a real effect does) and about the 2D CONTRACT: X is log2 of the speed-up geomean, Y the
paired token-cost ratio, one mark per arm against its own control, nothing joined by a line, colour
the intervention, shape the model, and the drawing itself lives in
:mod:`hpcagent_bench.stats.figures.efficacy` -- this script only wires the data.
"""

import importlib.util
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
from hpcagent_bench.stats import palette
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
    """Import ``scripts/plot_score_change.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("plot_score_change", REPO / "scripts" / "plot_score_change.py")
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
        {**common, "record": "submission", "speedup": speedup, "tokens": None, "suspect": 0},
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
    model: str = "qwen38", language: str = "c", off_speedup: float = 1.0, on_speedup: float = 1.4,
    off_tokens: float = 1000.0, on_tokens: float = 900.0, kernels: int = KERNELS,
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
        treated += episode(
            f"{model}-{language}-skills", model, language, kernel, f"t{kernel}", on_speedup * jitter, on_tokens / jitter
        )  # fmt: skip
    frame = pd.concat([pd.DataFrame(control).assign(skills=False), pd.DataFrame(treated).assign(skills=True)])
    return frame.reset_index(drop=True)


def one_arm_stats(
    model: str = "qwen38", language: str = "c", score_verdict: str = efficacy.NOT_SIGNIFICANT,
    cost_verdict: str = efficacy.NOT_SIGNIFICANT, family_size: int = 2,
) -> pd.DataFrame:  # fmt: skip
    return pd.DataFrame(
        [{"model": model, "language": language, "score_verdict": score_verdict, "cost_verdict": cost_verdict,
          "family_size": family_size}]
    )  # fmt: skip


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
            "benchmark": f"k{kernel}", "suspect": 0, "baseline": "numba", "run_root": "j1", "job": "j1",
            "attempt_index": 1, "ts_ms": kernel, "timing_reduction": "mwd-v2",
        }  # fmt: skip
        for arm, packet, language, speedup in (
            ("cpf-llr-focus40-oss120b-c", "", "c", 2.0),
            ("cpf-llr-focus40-oss120b-c-cpf", "cpf", "", 2.4),
        ):
            run = f"{arm}-{kernel}"
            base = {**common, "arm": arm, "packet": packet, "language": language, "run_id": run}
            rows.append(
                {**base, "record": "submission", "speedup": speedup, "baseline_ns": 1000.0, "native_ns": 1000.0 / speedup}
            )  # fmt: skip
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


def test_the_figure_stars_a_point_only_on_a_corrected_verdict() -> None:
    """The gate is the VERDICT: a raw p that was never corrected, and a pairing too small for any
    test, both draw a plain label."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        handles = efficacy_figures.draw_panel(
            ax, one_arm_raw(), one_arm_stats(score_verdict=efficacy.SIGNIFICANT), "skills"
        )
        labels = [text.get_text() for text in ax.texts]
    finally:
        plt.close(fig)
    assert "C *" in labels, labels
    assert any(h.get_label() == "BH q < 0.05 of 2" for h in handles), [h.get_label() for h in handles]


def test_the_filled_mark_wears_the_packet_colour_and_the_hollow_control_wears_the_control_colour() -> None:
    """The one colour rule: a packet's hue is the entity's, so it is the same hue in every figure --
    and the control reference is ``palette.control_color``, never one more packet hue. Colouring by
    MODEL instead spent the packet's channel on the shape's entity."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    treatment = "cpf"
    assert palette.color(treatment) != palette.model_color("qwen38")
    fig, ax = plt.subplots()
    try:
        efficacy_figures.draw_panel(ax, one_arm_raw(), one_arm_stats(), treatment)
        edges, faces = set(), set()
        for collection in (c for c in ax.collections if isinstance(c, PathCollection)):
            for rgba in collection.get_edgecolor():
                edges.add(matplotlib.colors.to_hex(rgba))
            for rgba in collection.get_facecolor():
                faces.add(matplotlib.colors.to_hex(rgba))
        assert palette.color(treatment) in faces | edges
        assert palette.control_color() in edges
        assert palette.model_color("qwen38") not in faces | edges
    finally:
        plt.close(fig)


def test_the_marker_shape_is_the_model_and_nothing_else() -> None:
    """Shape is always the model, so identity survives greyscale and a column-width shrink, where
    colour alone does not."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        efficacy_figures.draw_panel(ax, one_arm_raw(), one_arm_stats(), "cpfsrc")
        drawn = {collection.get_paths()[0] for collection in ax.collections if collection.get_paths()}
        style = matplotlib.markers.MarkerStyle(palette.marker("qwen38"))
        expected = style.get_path().transformed(style.get_transform())
        assert any(path.vertices.shape == expected.vertices.shape for path in drawn)
    finally:
        plt.close(fig)


def test_neither_axis_enables_a_minor_grid() -> None:
    """Major grid only. A minor line is a second grid at a second weight, and once the figure is
    reduced for print the panel reads as a texture instead of a reference."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        efficacy_figures.draw_panel(ax, one_arm_raw(), one_arm_stats(), "cpfsrc")
        assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
        assert not [tick for tick in ax.yaxis.get_minor_ticks() if tick.gridline.get_visible()]
        assert not [tick for tick in ax.xaxis.get_minor_ticks() if tick.gridline.get_visible()]
    finally:
        plt.close(fig)


def test_the_legend_is_drawn_once_on_the_figure_and_never_on_an_axes() -> None:
    """One key for the whole figure: several panels draw the same models in the same colours, and a
    key on each axes invites reading them as different sets of series."""
    out = pathlib.Path(tempfile.mkdtemp()) / "fig.pdf"
    written = efficacy_figures.figure_one(one_arm_raw(), one_arm_stats(), "cpfsrc", "label", out)
    assert written.with_suffix(".pdf").exists()

    panels = [("CPF", "cpf", one_arm_stats(), one_arm_raw()), ("CPFSRC", "cpfsrc", one_arm_stats(), one_arm_raw())]
    written_row = efficacy_figures.figure_row(panels, "label", pathlib.Path(tempfile.mkdtemp()) / "row.pdf")
    assert written_row.with_suffix(".pdf").exists()


def test_a_joined_row_has_exactly_one_legend_with_deduplicated_unique_labels() -> None:
    """A 1x3 figure draws one intervention hue, one model shape, one control mark and one
    undelivered cross ONCE each, never once per panel: every panel repeats the same channels, so a
    key on each would invite reading them as different sets of series, and a naive concatenation of
    three panels' handles would repeat every one of them three times."""
    import matplotlib.pyplot as plt

    n = 3
    fig, axes = plt.subplots(1, n, squeeze=False)
    try:
        treatments = ["cpf", "cpfsrc", "lang-skills"]
        handles_by_label: dict[str, object] = {}
        for ax, treatment in zip(axes[0], treatments, strict=True):
            for handle in efficacy_figures.draw_panel(ax, one_arm_raw(), one_arm_stats(), treatment, True, treatments):
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
                for h in efficacy_figures.draw_panel(ax, one_arm_raw(), one_arm_stats(), treatment, True, treatments)
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
    written = efficacy_figures.figure_row(panels, "label", pathlib.Path(tempfile.mkdtemp()) / "row.pdf")
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


def test_a_joined_rows_title_never_touches_the_saved_canvas_edge(tmp_path: pathlib.Path) -> None:
    """``style.title`` shrinks its font to fit the WIDTH IT MEASURES -- against a figure still at
    matplotlib's default dpi, before :func:`~hpcagent_bench.stats.style.save` writes the PNG at
    :data:`~hpcagent_bench.stats.style.SAVE_DPI`. FreeType hints a glyph run tighter at a lower dpi,
    so a title that 'fit' at measurement time came out overflowing both edges of the saved file --
    this is caught on the file itself, not on the same measurement that missed it the first time."""
    panels = [("Comparison", "cpf", one_arm_stats(), one_arm_raw())]
    long_label = "A row title long enough to need the shrink-to-fit loop to do real work here"
    out = tmp_path / "row.pdf"
    efficacy_figures.figure_row(panels, long_label, out)
    with Image.open(out.with_suffix(".png")) as image:
        width = image.size[0]
        rows = np.where((np.array(image.convert("L")) < 240).any(axis=1))[0]
    left, right = png_ink_columns(out.with_suffix(".png"), int(rows.min()), int(rows.min()) + 40)
    assert 0 < left, "the title's own ink starts at the canvas edge"
    assert right < width - 1, "the title's own ink runs to the canvas edge"


def test_required_left_margin_reserves_more_than_the_rows_old_flat_fraction() -> None:
    """A direct measurement: a wide-ranging axis's own long tick labels (``0.0078125x``) beside the
    panel's Y label need more room than :func:`figure_row` used to reserve, a FLAT 0.13 of the row's
    width regardless of content -- on a two-panel NATURAL row (3.6in side each) that fraction gave
    the label 0.13 * 7.45 =~ 0.97in, and the label's own ink ran past it off the canvas."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    fig.set_dpi(plotstyle.SAVE_DPI)
    ax.set_yscale("log", base=2.0)
    ax.set_ylim(2.0**-7, 2.0**7)  # 0.0078125x .. 128x, the dynamic range a real outlier kernel gave
    plotstyle.value_axis(ax, "y", log_base=2.0)
    ax.yaxis.set_major_formatter(FuncFormatter(efficacy_figures.ratio_tick))
    ax.set_ylabel("Token-Cost Ratio, Treated / Control", fontsize=plotstyle.LABEL_PT * 0.68)
    ax.tick_params(axis="both", labelsize=plotstyle.TICK_PT * 0.6)
    try:
        left_in = efficacy_figures.required_left_margin(fig, ax)
        old_fixed_in = 0.13 * 7.45  # the row's own former constant, at a real two-panel row's width
        assert left_in > old_fixed_in, (left_in, old_fixed_in)
    finally:
        plt.close(fig)


def test_the_legend_names_the_interval_method_and_the_kernels_it_is_over() -> None:
    """Both axes here always draw the same estimator (a log-space t interval), and the legend says
    so and over how many kernels (SC15 Rule 5)."""
    note = efficacy_figures.interval_note([8, 8], "Speed-Up Geomean")
    assert "log-t" in note
    assert "n=8" in note


def test_interval_note_names_are_fixed_never_chosen_by_sample_size() -> None:
    """Unlike the older per-arm reduction, the paired geomean drawn here
    (``summary.geomean_ci``) is ALWAYS the log-space t interval, at any n -- there is no adaptive
    bootstrap fallback to name."""
    small = efficacy_figures.interval_note([2], "Speed-Up Geomean")
    large = efficacy_figures.interval_note([200], "Speed-Up Geomean")
    assert "log-t" in small and "log-t" in large


def test_the_figure_key_carries_one_interval_note_per_axis() -> None:
    """Both notes reach the reader, in the figure's ONE key."""
    handles = efficacy_figures.legend_handles("cpfsrc", ["qwen38"], ["note-a", "note-b"], ["cpfsrc"], 12)
    labels = [h.get_label() for h in handles]
    assert "note-a" in labels and "note-b" in labels, labels


def test_an_undelivered_kernel_draws_a_cross_and_the_legend_names_it() -> None:
    """A kernel the arm was served and never verified is a placeholder, not a measurement
    (``population.DELIVERED_COLUMN``); the cloud draws it as a cross and the key explains it."""
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
    treated_rows_list.append(
        {
            "arm": "a-c-skills", "model": "qwen38", "language": "c", "benchmark": "kmissing", "run_root": "tm",
            "job": "tm", "run_id": "tm", "baseline": "numba", "attempt_index": 1, "ts_ms": 0,
            "timing_reduction": "mwd-v2", "record": "task", "speedup": None, "tokens": 950.0, "suspect": None,
        }
    )  # fmt: skip
    control = pd.DataFrame(control_rows_list)
    treated = pd.DataFrame(treated_rows_list)

    series = efficacy_figures.reduce_pair(control, treated)

    assert series is not None
    assert series.delivered < series.kernels
    assert not series.cloud[~series.cloud.delivered].empty

    handles = efficacy_figures.legend_handles("skills", ["qwen38"], ["a", "b"], ["skills"], 1)
    assert any(h.get_label() == plotstyle.NOT_DELIVERED_LABEL for h in handles)


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
    return pd.DataFrame(
        [
            {"model": m, "language": lang, "score_verdict": efficacy.NOT_SIGNIFICANT,
             "cost_verdict": efficacy.NOT_SIGNIFICANT, "family_size": 12}
            for m in MODELS
            for lang in LANGUAGES
        ]
    )  # fmt: skip


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
    """A family table in the shape ``experiments/paired_arms.py`` writes, one row per pair per leg."""
    return pd.DataFrame(
        [
            {
                "family": "demo",
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
    """``experiments/paired_arms.py`` already ran the paired test and the Benjamini-Hochberg
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
            "arm": arm, "benchmark": f"k{kernel}", "suspect": 0, "baseline": "numba", "run_root": "j1", "job": "j1",
            "run_id": f"{arm}-{kernel}", "attempt_index": 1, "ts_ms": kernel, "timing_reduction": "mwd-v2",
        }  # fmt: skip
        rows.append(
            {**common, "record": "submission", "speedup": speedup, "baseline_ns": 1000.0, "native_ns": 1000.0 / speedup}
        )  # fmt: skip
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
            h.get_label()
            for h in efficacy_figures.draw_panel(ax, tagged, stats, "repo", control_name="Bare Kernel")
        ]  # fmt: skip
    finally:
        plt.close(fig)
    assert "Bare Kernel" in labels, labels
    assert "No Packet" not in labels
    assert experiment_tags.packet_name("repo") in labels, labels


def test_parse_spec_reads_semicolon_separated_key_value_pairs() -> None:
    """``--comparison`` takes one string per panel; the parser is the only place its grammar is
    decided."""
    spec = plot.parse_spec("title=Kernel Formulation;intervention=repo;pairs=x.csv;control-label=Bare Kernel")
    assert spec == {
        "title": "Kernel Formulation", "intervention": "repo", "pairs": "x.csv", "control-label": "Bare Kernel",
    }  # fmt: skip
