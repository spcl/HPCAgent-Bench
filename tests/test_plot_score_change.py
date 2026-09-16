# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``scripts/plot_score_change.py`` -- the score-vs-cost figure and the stars on it.

The load-bearing assertions are about MULTIPLICITY. One figure carries three models x two
languages x two axes, so twelve paired tests decide its marks, and twelve uncorrected 5%
thresholds paint at least one star on 46% of figures where nothing happened. What is asserted here
is therefore that a raw threshold which would have starred a point does not survive the correction,
that the mark is gated on the corrected verdict, and that the figure says so where a reader looks
-- a corrected star and an uncorrected star are the same pixels.
"""

import importlib.util
import pathlib
import sys
import tempfile

import matplotlib.colors
import matplotlib.markers
import pandas as pd
import pytest

from hpcagent_bench import experiment_tags, packets
from hpcagent_bench.harness import efficacy
from hpcagent_bench.stats import palette, summary
from hpcagent_bench.stats import style as plotstyle

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
    assert plot.family_size(frame) == 12


def test_load_reads_skills_off_the_recorded_packet_before_the_arm_name(tmp_path: pathlib.Path) -> None:
    """An arm renamed away from the ``-skills`` suffix but recording ``lang-skills`` loads as skilled, and a recorded
    packet beats a ``-skills`` name. An arm that recorded NO packet takes the name's token: the llr-focus40 kimi
    ``-skills`` arms never stamped one, and reading them as the control dropped five of six skills pairs (this case
    used to assert the opposite)."""
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
    """A hard-coded ``(skills, cpf, cpfsrc)`` "known treatments" triple let a FOURTH treatment
    (``perf-playbook-cpu`` on cpf-llr-focus40) read as part of the control, scoring the campaign's
    real control against a mixture. ``control_rows`` needs no treatment list at all: the control is
    exactly the canonical empty packet, whatever treatments a campaign happens to run."""
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
    (here ``perf-playbook-cpu``, absent from the old hard-coded triple) must never be silently
    counted as part of the no-packet control."""
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


def thin(log2_speedup: float, tokens: float) -> dict[str, float]:
    """An ``absolute_points`` row over too few kernels for an interval, as the table writes one."""
    nan = float("nan")
    return {
        "log2_speedup": log2_speedup,
        "log2_speedup_low": nan,
        "log2_speedup_high": nan,
        "tokens": tokens,
        "tokens_low": nan,
        "tokens_high": nan,
        "kernels": float(KERNELS),
    }


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
            {"model": "qwen38", "language": "c", "skills": False, **thin(1.0, 1000.0)},
            {"model": "qwen38", "language": "c", "skills": True, **thin(1.4, 900.0)},
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
    fig, axes = plt.subplots(1, len(plot.PANELS))
    try:
        plot.draw_absolute(list(axes), frame, stats, "skills")
        labels = [text.get_text() for ax in axes for text in ax.texts]
    finally:
        plt.close(fig)
    assert ("C *" in labels) == starred, labels


def crowded_legs() -> dict[tuple[str, str], tuple[tuple[float, float], tuple[float, float]]]:
    """Six arms of one CPU skills figure, three of them landing within a few percent of each
    other -- the fixture the old ring of candidate label places could not solve."""
    return {
        ("qwen38", "c"): ((2.75, 350e3), (2.82, 310e3)),
        ("qwen38", "fortran"): ((2.77, 245e3), (2.80, 255e3)),
        ("oss120b", "c"): ((2.13, 50e3), (2.24, 52e3)),
        ("oss120b", "fortran"): ((2.38, 72e3), (2.29, 65e3)),
        ("kimi27sglang", "c"): ((2.75, 178e3), (2.79, 165e3)),
        ("kimi27sglang", "fortran"): ((2.83, 165e3), (2.81, 160e3)),
    }


def crowded_figure() -> tuple[object, list[object]]:
    """The crowded fixture drawn at the script's own size, with the layout already settled."""
    import matplotlib.pyplot as plt

    legs = crowded_legs()
    frame = pd.DataFrame(
        [
            {"model": model, "language": language, "skills": on, **thin(*point)}
            for (model, language), sides in legs.items()
            for on, point in zip((False, True), sides, strict=True)
        ]
    )
    verdict = {"score_verdict": efficacy.NOT_SIGNIFICANT, "cost_verdict": efficacy.NOT_SIGNIFICANT, "family_size": 12}
    stats = pd.DataFrame([{"model": model, "language": language, **verdict} for model, language in legs])
    fig, axes = plt.subplots(1, len(plot.PANELS), figsize=plot.PANEL_SIZE)
    plot.draw_absolute(list(axes), frame, stats, "skills")
    fig.subplots_adjust(**plot.PANEL_MARGINS)
    for ax in axes:
        plot.stack_labels(ax)
    return fig, list(axes)


def test_no_two_arm_labels_overprint_each_other_however_close_the_arms_land() -> None:
    """The bug this guards: six arms share one square panel and three land within a few percent, so
    there is no free place AROUND a mark and every label after the first printed on top of its
    neighbour -- "Fortran" over "C", three times in one panel. A column solves the vertical order
    instead of searching for a hole."""
    import matplotlib.pyplot as plt
    from matplotlib.text import Annotation

    fig, axes = crowded_figure()
    try:
        renderer = fig.canvas.get_renderer()
        for ax in axes:
            notes = [text for text in ax.texts if isinstance(text, Annotation)]
            boxes = [tuple(note.get_window_extent(renderer).extents) for note in notes]
            assert len(boxes) == len(crowded_legs())
            for index, box in enumerate(boxes):
                for other in boxes[index + 1 :]:
                    assert not (box[0] < other[2] and other[0] < box[2] and box[1] < other[3] and other[1] < box[3]), (
                        notes[index].get_text()
                    )
    finally:
        plt.close(fig)


def test_every_arm_label_sits_at_one_x_in_a_single_right_hand_column() -> None:
    """A column, not a ring: every label shares one x just right of the treated marks, so the only
    thing left to settle is the vertical order and a reader scans one list rather than hunting."""
    import matplotlib.pyplot as plt
    from matplotlib.text import Annotation

    fig, axes = crowded_figure()
    try:
        renderer = fig.canvas.get_renderer()
        for ax in axes:
            notes = [text for text in ax.texts if isinstance(text, Annotation)]
            lefts = {round(note.get_window_extent(renderer).x0, 3) for note in notes}
            assert len(lefts) == 1, lefts
            assert all(tuple(note.xyann)[0] == plot.LABEL_GAP_PT for note in notes)
    finally:
        plt.close(fig)


def test_a_label_the_column_moved_off_its_mark_gets_a_leader_line() -> None:
    """A label pushed away from its own height needs to say which mark it belongs to; one that
    barely moved does not, and a leader there would be a line for nothing."""
    import matplotlib.pyplot as plt

    fig, axes = crowded_figure()
    try:
        leaders = len(axes[0].lines) - len(crowded_legs())  # one pair link per arm is drawn first
    finally:
        plt.close(fig)
    assert leaders > 0


def test_spread_pushes_labels_apart_and_keeps_them_inside_the_column() -> None:
    """The sweep itself: ascending input, a minimum step between neighbours, nothing past either
    end. A column too short comes out evenly packed rather than short of a label, since a missing
    label reads as a missing arm."""
    assert plot.spread([10.0, 10.5, 11.0], 5.0, 0.0, 100.0) == [10.0, 15.0, 20.0]

    settled = plot.spread([90.0, 95.0, 99.0], 5.0, 0.0, 100.0)
    assert settled[-1] <= 100.0
    assert all(b - a >= 5.0 - 1e-9 for a, b in zip(settled, settled[1:], strict=False))

    floored = plot.spread([1.0, 1.0, 1.0], 5.0, 0.0, 6.0)
    assert min(floored) >= 0.0 and max(floored) <= 6.0


def treatment_panel(treatment: str, model: str = "qwen38") -> "plot.PanelRow":
    """One synthetic row, the shape :func:`plot.figure_treatments` takes: a row titled by its own
    treatment, which is what a figure joining several treatments against one control draws."""
    stats = pd.DataFrame(
        [
            {
                "model": model,
                "language": "c",
                "score_verdict": efficacy.NOT_SIGNIFICANT,
                "cost_verdict": efficacy.NOT_SIGNIFICANT,
                "family_size": 2,
            }
        ]
    )
    absolute = pd.DataFrame(
        [
            {"model": model, "language": "c", "skills": False, **thin(1.0, 1000.0)},
            {"model": model, "language": "c", "skills": True, **thin(1.4, 900.0)},
        ]
    )
    return plot.PanelRow(packets.label(treatment), treatment, stats, absolute)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_n_treatments_draw_one_row_of_two_square_panels_each(n: int) -> None:
    """A comparison is TWO panels -- speed-up and tokens -- and each one is square: both carry a
    measured value on Y, and a reader compares them by panel shape as well as by content."""
    panels = [treatment_panel(f"treatment{i}") for i in range(n)]
    fig = plot.build_treatments_figure(panels, "label")
    try:
        assert len(fig.axes) == n * len(plot.PANELS)
        assert all(ax.get_box_aspect() == pytest.approx(1.0) for ax in fig.axes)
        width, _height = fig.get_size_inches()
        expected = len(plot.PANELS) * plot.SQUARE_PANEL_SIDE + plot.INNER_GAP_IN + plot.LEFT_IN + plot.RIGHT_IN
        assert width == pytest.approx(expected)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_double_column_caps_a_joined_figures_row_width() -> None:
    """``--double-column`` is the figure's budget on a paper page: a comparison's two panels plus
    their chrome must fit inside ``style.DOUBLE_COLUMN_WIDTH`` inches, not take a fixed natural
    panel size each."""
    panels = [treatment_panel(f"treatment{i}") for i in range(3)]
    fig = plot.build_treatments_figure(panels, "label", double_column=True)
    try:
        width, _ = fig.get_size_inches()
        assert width <= plotstyle.DOUBLE_COLUMN_WIDTH + 1e-6
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_treatment_frame_tags_the_control_false_and_the_treatment_true() -> None:
    """``absolute_points``/``draw_absolute`` read an on/off ``skills`` flag; ``treatment_frame``
    builds it from the packet split rather than the historical column name."""
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


def test_the_figure_names_the_correction_and_the_size_of_the_family() -> None:
    """An uncorrected star and a corrected one are identical pixels, so the only place the reader
    can learn which they are looking at is the key beside the mark."""
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        [
            {"model": "qwen38", "language": "c", "skills": False, **thin(1.0, 1000.0)},
            {"model": "qwen38", "language": "c", "skills": True, **thin(1.4, 900.0)},
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
    fig, axes = plt.subplots(1, len(plot.PANELS))
    try:
        labels = [handle.get_label() for handle in plot.draw_absolute(list(axes), frame, stats, "skills")]
    finally:
        plt.close(fig)
    assert "BH q < 0.05 of 12" in labels, labels


def test_points_never_raises_a_bare_keyerror_when_the_two_sides_share_no_model_language() -> None:
    """The bug this guards: ``pd.DataFrame([])`` (an empty ``rows`` list) has NO columns at all, so
    ``.dropna(subset=["score", "cost"])`` on it raised a bare ``KeyError(['score', 'cost'])`` where
    the caller expected "these two sides pair on nothing" -- exactly what an arm whose ``language``
    was never recorded produced against its control (disjoint (model, language) sets)."""
    before = pd.DataFrame([{"model": "oss120b", "language": "c", "record": "submission"}])
    after = pd.DataFrame([{"model": "oss120b", "language": "", "record": "submission"}])

    frame = plot.points(before, after)

    assert frame.empty
    assert list(frame.columns) == list(plot.POINT_COLUMNS)


def test_a_treatment_arm_that_never_recorded_its_language_still_pairs_against_control(
    tmp_path: pathlib.Path,
) -> None:
    """The real bug, end to end: ``cpf-llr-focus40-oss120b-c-cpf`` recorded ``packet=cpf`` on every
    row but ``language`` on none of them. ``load`` (through ``experiments.fill_arm_identity``) must
    recover ``c`` from the arm's own name, and ``one_treatment_panel`` must then find the (model,
    language) key it shares with its control instead of finding nothing."""
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
        }
        for arm, packet, language, speedup in (
            ("cpf-llr-focus40-oss120b-c", "", "c", 2.0),
            ("cpf-llr-focus40-oss120b-c-cpf", "cpf", "", 2.4),
        ):
            run = f"{arm}-{kernel}"
            base = {**common, "arm": arm, "packet": packet, "language": language, "run_id": run}
            rows.append(
                {
                    **base,
                    "record": "submission",
                    "speedup": speedup,
                    "baseline_ns": 1000.0,
                    "native_ns": 1000.0 / speedup,
                }
            )
            rows.append({**base, "record": "call", "speedup": speedup, "tokens": 1000.0})
            rows.append({**base, "record": "task", "speedup": None, "tokens": 1000.0})
    pd.DataFrame(rows).to_csv(path, index=False)

    frame_all = plot.load(path, prefix="")
    treated = frame_all[frame_all.arm == "cpf-llr-focus40-oss120b-c-cpf"]
    assert set(treated.language) == {"c"}, "the arm name is the last resort when no row ever recorded it"

    control = plot.control_rows(frame_all)
    roster = sorted(frame_all.benchmark.dropna().unique())
    built = plot.one_treatment_panel(frame_all, control, "cpf", roster)

    assert built is not None
    stats, _absolute = built
    assert len(stats) == 1
    assert (stats.iloc[0].model, stats.iloc[0].language) == ("oss120b", "c")


def test_complete_side_arms_drops_an_arm_short_of_the_roster_and_names_it_on_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Same completeness rule as ``scripts/plot_kernel_comparison.py`` (``population.complete_arms``),
    so the two figures never disagree about which arms exist: an arm missing a roster kernel is
    dropped, not entered at any stand-in value, and named so the drop is auditable."""
    roster = ["k0", "k1", "k2"]
    control = pd.DataFrame({"arm": ["ctrl"] * 3, "benchmark": roster})
    treated = pd.DataFrame(
        {"arm": ["good-cpf", "good-cpf", "good-cpf", "short-cpf"], "benchmark": ["k0", "k1", "k2", "k0"]}
    )

    kept = plot.complete_side_arms(control, treated, roster, "cpf", include_incomplete=False)

    assert kept == {"ctrl", "good-cpf"}
    err = capsys.readouterr().err
    assert "cpf: dropping short-cpf (1/3 roster kernels)" in err


@pytest.mark.parametrize(
    "treatment, hollow, filled",
    [
        pytest.param("skills", "No Skill Packet", "All Skill Pages", id="skills"),
        pytest.param("cpf", "No Packet", "Canonical Parallel Form Page", id="cpf"),
        pytest.param("cpfsrc", "No Packet", "Canonical Parallel Form as Source", id="cpfsrc"),
    ],
)
def test_the_hollow_and_filled_legend_marks_name_the_treatment(treatment: str, hollow: str, filled: str) -> None:
    """The hollow/filled pair used to say "No Skills"/"Skills" on every treatment, which misnamed a
    CPF panel as if the treatment under test were a skill. The hollow mark reads
    ``packets.control_label``, the filled mark the treatment's own registry name -- a CPF page
    panel says "No Packet" / "Canonical Parallel Form Page", never "No Skills" / "Skills"."""
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        [
            {"model": "qwen38", "language": "c", "skills": False, **thin(1.0, 1000.0)},
            {"model": "qwen38", "language": "c", "skills": True, **thin(1.4, 900.0)},
        ]
    )
    stats = pd.DataFrame(
        [
            {
                "model": "qwen38",
                "language": "c",
                "score_verdict": efficacy.NOT_SIGNIFICANT,
                "cost_verdict": efficacy.NOT_SIGNIFICANT,
                "family_size": 2,
            }
        ]
    )
    fig, axes = plt.subplots(1, len(plot.PANELS))
    try:
        labels = [handle.get_label() for handle in plot.draw_absolute(list(axes), frame, stats, treatment)]
    finally:
        plt.close(fig)
    assert hollow in labels, labels
    assert filled in labels, labels
    assert "No Skills" not in labels
    assert "Skills" not in labels


def test_the_model_legend_lists_only_a_model_with_a_drawn_point() -> None:
    """A model the control side ran under SOME OTHER treatment (Kimi, on the CPF-page campaign's
    ``cpfsrc`` arm but never its ``cpf`` one) still shows up in ``frame`` as a control-only row --
    ``hues``/``shapes`` reserved it a colour, and the old legend built straight off those kept
    naming a model this panel never draws a point for."""
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        [
            {"model": "qwen38", "language": "c", "skills": False, **thin(1.0, 1000.0)},
            {"model": "qwen38", "language": "c", "skills": True, **thin(1.4, 900.0)},
            # kimi27sglang: control side only -- no treated row, so no point is ever drawn for it.
            {"model": "kimi27sglang", "language": "c", "skills": False, **thin(1.0, 1000.0)},
        ]
    )
    stats = pd.DataFrame(
        [
            {
                "model": "qwen38",
                "language": "c",
                "score_verdict": efficacy.NOT_SIGNIFICANT,
                "cost_verdict": efficacy.NOT_SIGNIFICANT,
                "family_size": 2,
            }
        ]
    )
    fig, axes = plt.subplots(1, len(plot.PANELS))
    try:
        labels = [handle.get_label() for handle in plot.draw_absolute(list(axes), frame, stats, "cpf")]
    finally:
        plt.close(fig)
    assert experiment_tags.model_name("qwen38") in labels, labels
    assert experiment_tags.model_name("kimi27sglang") not in labels, labels


def test_include_incomplete_keeps_a_short_arm_and_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    roster = ["k0", "k1", "k2"]
    control = pd.DataFrame({"arm": ["ctrl"] * 3, "benchmark": roster})
    treated = pd.DataFrame({"arm": ["short-cpf"], "benchmark": ["k0"]})

    kept = plot.complete_side_arms(control, treated, roster, "cpf", include_incomplete=True)

    assert kept == {"ctrl", "short-cpf"}
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# The drawing conventions, pinned. Colour is the PACKET, shape is the MODEL, the measured value is
# on Y, the legend belongs to the FIGURE, and the grid is major only.


def one_arm_figure(treatment: str) -> tuple[object, list[object], pd.DataFrame]:
    """A one-arm figure in the layout the script draws: two panels, one (model, language) pair."""
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        [
            {"model": "qwen38", "language": "c", "skills": False, **thin(1.0, 1000.0)},
            {"model": "qwen38", "language": "c", "skills": True, **thin(1.4, 900.0)},
        ]
    )
    stats = pd.DataFrame(
        [
            {
                "model": "qwen38",
                "language": "c",
                "score_verdict": efficacy.NOT_SIGNIFICANT,
                "cost_verdict": efficacy.NOT_SIGNIFICANT,
                "family_size": 2,
            }
        ]
    )
    fig, axes = plt.subplots(1, len(plot.PANELS))
    plot.draw_absolute(list(axes), frame, stats, treatment)
    return fig, list(axes), stats


def test_the_measured_value_is_on_the_y_axis_of_both_panels() -> None:
    """Rule one: a speed-up and a token count are measured quantities and never sit on X. X carries
    the two CONDITIONS, which are categories, so it is the axis that must stay linear and unscaled."""
    import matplotlib.pyplot as plt

    fig, axes, _stats = one_arm_figure("cpfsrc")
    try:
        assert [ax.get_yscale() for ax in axes] == ["log", "log"]
        assert [ax.get_xscale() for ax in axes] == ["linear", "linear"]
        assert [ax.get_ylabel() for ax in axes] == [panel.label for panel in plot.PANELS]
        assert [tick.get_text() for tick in axes[0].get_xticklabels()] == list(plot.CONDITION_LABELS)
    finally:
        plt.close(fig)


def test_the_filled_mark_wears_the_packet_colour_and_the_hollow_one_the_control_colour() -> None:
    """The one colour rule: a packet's hue is the entity's, from ``palette.color``, so it is the
    same hue in every figure -- and the control is ``palette.control_color``, never one more packet
    hue. Colouring by MODEL (the bug this pins shut) spent the packet's channel on the shape's
    entity, so one arm read as a different treatment in each figure it appeared in."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    # `cpf`, not `cpfsrc`: each ramp starts at the same hue, so the FIRST packet and the FIRST
    # model draw the same colour and a model-coloured mark would pass unnoticed.
    treatment = "cpf"
    assert palette.color(treatment) != palette.model_color("qwen38")
    fig, axes, _stats = one_arm_figure(treatment)
    try:
        edges = set()
        faces = set()
        for collection in (c for c in axes[0].collections if isinstance(c, PathCollection)):
            for rgba in collection.get_edgecolor():
                edges.add(matplotlib.colors.to_hex(rgba))
            for rgba in collection.get_facecolor():
                faces.add(matplotlib.colors.to_hex(rgba))
        assert palette.color(treatment) in faces
        assert palette.control_color() in edges
        assert palette.model_color("qwen38") not in faces | edges
    finally:
        plt.close(fig)


def test_the_marker_shape_is_the_model_and_nothing_else() -> None:
    """Shape is always the model (``palette.marker``), so identity survives greyscale and a
    column-width shrink, where colour alone does not."""
    import matplotlib.pyplot as plt

    fig, axes, _stats = one_arm_figure("cpfsrc")
    try:
        drawn = {collection.get_paths()[0] for collection in axes[0].collections if collection.get_paths()}
        expected = (
            matplotlib.markers.MarkerStyle(palette.marker("qwen38"))
            .get_path()
            .transformed(matplotlib.markers.MarkerStyle(palette.marker("qwen38")).get_transform())
        )
        assert any(path.vertices.shape == expected.vertices.shape for path in drawn)
    finally:
        plt.close(fig)


def test_neither_panel_enables_a_minor_grid() -> None:
    """Major grid only. A minor line is a second grid at a second weight, and once the figure is
    reduced for print the panel reads as a texture instead of a reference."""
    import matplotlib.pyplot as plt

    fig, axes, _stats = one_arm_figure("cpfsrc")
    try:
        for ax in axes:
            assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
            assert not [tick for tick in ax.yaxis.get_minor_ticks() if tick.gridline.get_visible()]
            assert not [tick for tick in ax.xaxis.get_minor_ticks() if tick.gridline.get_visible()]
    finally:
        plt.close(fig)


def test_the_legend_is_drawn_once_on_the_figure_and_never_on_an_axes() -> None:
    """One key for the whole figure (``style.legend_below``): both panels draw the same models in
    the same colours, and a key on each axes invites reading them as two different sets of series."""
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        [
            {"model": "qwen38", "language": "c", "skills": False, **thin(1.0, 1000.0)},
            {"model": "qwen38", "language": "c", "skills": True, **thin(1.4, 900.0)},
        ]
    )
    stats = pd.DataFrame(
        [
            {
                "model": "qwen38",
                "language": "c",
                "score_verdict": efficacy.NOT_SIGNIFICANT,
                "cost_verdict": efficacy.NOT_SIGNIFICANT,
                "family_size": 2,
            }
        ]
    )
    out = pathlib.Path(tempfile.mkdtemp()) / "fig.pdf"
    plot.figure_absolute(frame, stats, "cpfsrc", "label", out)
    assert out.exists()

    fig = plot.build_treatments_figure([treatment_panel("cpf"), treatment_panel("cpfsrc")], "label")
    try:
        assert len(fig.legends) == 1
        assert all(ax.get_legend() is None for ax in fig.axes)
    finally:
        plt.close(fig)


def test_the_joined_figure_names_its_one_control_once_over_every_treatment() -> None:
    """One control, one entry. Named per panel instead, a joined figure grew one hollow-mark entry
    per row ("No Packet" beside "No Skill Packet") for the one set of control arms."""
    import matplotlib.pyplot as plt

    fig = plot.build_treatments_figure([treatment_panel("cpf"), treatment_panel("lang-skills")], "label")
    try:
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
    finally:
        plt.close(fig)
    assert labels.count("No Packet") + labels.count("No Skill Packet") == 1


def test_the_legend_names_the_interval_method_and_the_kernels_it_is_over() -> None:
    """A log-t interval and a bootstrap interval are drawn the same way, so the figure has to say
    which it is showing and over how many kernels (SC15 Rule 5)."""
    import matplotlib.pyplot as plt

    fig, axes, _stats = one_arm_figure("cpfsrc")
    try:
        labels = [
            handle.get_label()
            for handle in plot.legend_handles("cpfsrc", ["qwen38"], _stats, ["note-a", "note-b"], ["cpfsrc"])
        ]
    finally:
        plt.close(fig)
    assert "Pair Link" in labels, labels
    assert "note-a" in labels and "note-b" in labels, labels


def test_the_interval_note_names_the_method_the_kernel_count_selects() -> None:
    """``summary.geomean_interval`` picks log-t at or above ``LOG_T_MIN_SAMPLES`` kernels and a
    log-space bootstrap below it; the note a reader sees must track that choice, not a constant."""
    wide = pd.DataFrame([{"kernels": float(summary.LOG_T_MIN_SAMPLES)}])
    narrow = pd.DataFrame([{"kernels": float(summary.LOG_T_MIN_SAMPLES - 1)}])

    assert "log-t" in plot.interval_note(wide, plot.SPEEDUP_PANEL)
    assert "bootstrap" in plot.interval_note(narrow, plot.SPEEDUP_PANEL)
    assert f"n={summary.LOG_T_MIN_SAMPLES}" in plot.interval_note(wide, plot.SPEEDUP_PANEL)


def test_each_panel_names_the_estimator_its_own_whiskers_draw() -> None:
    """The two panels do not draw the same interval. The speed-up whisker is
    ``summary.geomean_interval``; the token whisker is ``summary.median_ci``, a percentile bootstrap
    of the MEDIAN (``population.kernel_medians``). One note naming only the geomean's interval
    labelled every token whisker as a claim it does not make."""
    frame = pd.DataFrame([{"kernels": float(summary.LOG_T_MIN_SAMPLES)}])

    speedup = plot.interval_note(frame, plot.SPEEDUP_PANEL)
    tokens = plot.interval_note(frame, plot.TOKENS_PANEL)

    assert speedup.startswith("Geomean") and "log-t" in speedup
    assert tokens.startswith("Median") and plot.MEDIAN_INTERVAL_METHOD in tokens
    assert "Geomean" not in tokens
    assert f"n={summary.LOG_T_MIN_SAMPLES}" in speedup and f"n={summary.LOG_T_MIN_SAMPLES}" in tokens


def test_a_note_over_legs_of_different_n_names_every_method_those_legs_use() -> None:
    """``geomean_interval`` chooses per LEG, so a figure whose legs straddle ``LOG_T_MIN_SAMPLES``
    draws two different intervals the same way. Naming the method of the SMALLEST leg called every
    wider leg's whisker something it is not."""
    small, large = summary.MIN_INTERVAL_SAMPLES, summary.LOG_T_MIN_SAMPLES + 5
    mixed = pd.DataFrame([{"kernels": float(small)}, {"kernels": float(large)}])

    note = plot.interval_note(mixed, plot.SPEEDUP_PANEL)

    assert summary.interval_method(small) in note, note
    assert summary.interval_method(large) in note, note
    assert f"n={small}" in note and f"n={large}" in note, note


def test_the_figure_key_carries_one_interval_note_per_panel() -> None:
    """Both notes reach the reader, and they reach it in the figure's ONE key."""
    import matplotlib.pyplot as plt

    fig = plot.build_treatments_figure([treatment_panel("cpfsrc")], "label")
    try:
        assert len(fig.legends) == 1
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
    finally:
        plt.close(fig)
    assert sum(1 for label in labels if label.startswith("Geomean, 95%")) == 1, labels
    assert sum(1 for label in labels if label.startswith("Median, 95%")) == 1, labels


# ---------------------------------------------------------------------------
# The EXPLICIT-PAIR entry point: a comparison whose two sides are two campaigns, or whose condition
# is not a packet suffix at all, drawn as the same two square panels.

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
            }
            for treated, control in pairs
            for leg in (plot.SPEEDUP_LEG, plot.TOKENS_LEG)
        ]
    )


def test_a_pairs_leg_names_the_language_and_every_packet_both_arms_carried() -> None:
    """llrblind runs C and C with the skill pages against their own scored arms, so a leg label of
    the language alone would draw two different arms as one. The INTERVENTION is never named: the
    title and the legend already say which side is which, and repeating it on every label states
    once more what the figure states once."""
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
    correction over exactly this family, and the paper's table is printed from the same CSV. A
    figure that re-derived the statistic could star a pair the table calls not significant, and a
    reader would have no way to tell which of the two is the finding."""
    table = family_csv([BLIND_PAIR], efficacy.SIGNIFICANT, efficacy.NOT_SIGNIFICANT)
    stats = plot.family_stats(table, "no-score")

    assert list(stats.leg) == ["C +skills"]
    assert list(stats.model) == ["qwen38"]
    assert list(stats.score_verdict) == [efficacy.SIGNIFICANT]
    assert list(stats.cost_verdict) == [efficacy.NOT_SIGNIFICANT]
    assert list(stats.kernels) == [40]
    assert plot.family_size(stats) == 2


def test_the_family_csv_declares_the_pairs_in_the_order_it_wrote_them() -> None:
    """The family's own declared order, not a re-sort: a caller's model/language loop is the order
    a reader of the table already has in front of them."""
    table = family_csv([BLIND_PAIR, SCICOMP_PAIR], efficacy.NOT_SIGNIFICANT, efficacy.NOT_SIGNIFICANT)
    assert plot.family_pairs(table) == [BLIND_PAIR, SCICOMP_PAIR]


def test_the_label_column_widens_for_a_longer_leg_so_no_label_is_written_off_the_canvas() -> None:
    """A fixed-size save writes anything past the edge into nothing, and "Fortran +skills" is half
    as wide again as "Fortran"."""
    short = pd.DataFrame([{"leg": "C"}])
    long = pd.DataFrame([{"leg": "Fortran +skills"}])
    assert plot.label_column_in(long) > plot.label_column_in(short)
    assert plot.label_column_in(short) >= plot.LABEL_COLUMN_IN


def test_a_pair_figure_wears_the_intervention_hue_and_names_a_control_that_is_not_a_missing_packet(
    tmp_path: pathlib.Path,
) -> None:
    """git-scicomp's control is the BARE KERNEL and llrblind's kept its score tool; "No Packet"
    names neither, so the hollow mark's text is the caller's. The treated side still takes its hue
    and its display name from the registry, like every other intervention."""
    import matplotlib.pyplot as plt

    absolute = pd.DataFrame(
        [
            {"model": "qwen38", "language": "c", "leg": "C", "skills": False, **thin(0.6, 900e3)},
            {"model": "qwen38", "language": "c", "leg": "C", "skills": True, **thin(1.1, 1.4e6)},
        ]
    )
    stats = plot.family_stats(family_csv([SCICOMP_PAIR], efficacy.SIGNIFICANT, ""), "repo")
    fig, axes = plt.subplots(1, len(plot.PANELS))
    try:
        labels = [
            handle.get_label()
            for handle in plot.draw_absolute(list(axes), absolute, stats, "repo", control_name="Bare Kernel")
        ]
    finally:
        plt.close(fig)
    assert "Bare Kernel" in labels, labels
    assert "No Packet" not in labels
    assert experiment_tags.packet_name("repo") in labels, labels


def observation_rows(arm: str, speedup: float, tokens: float, kernels: int = KERNELS) -> list[dict[str, object]]:
    """One arm's rows in the shape an extraction writes: a graded submission and a task total per
    kernel, which is what ``population.kernel_medians`` reduces."""
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
        }
        rows.append(
            {
                **common,
                "record": "submission",
                "speedup": speedup,
                "baseline_ns": 1000.0,
                "native_ns": 1000.0 / speedup,
            }
        )
        rows.append({**common, "record": "task", "speedup": None, "tokens": tokens})
    return rows


def test_pair_points_reduces_each_arm_by_name_and_tags_which_side_of_the_pair_it_is() -> None:
    """The two sides live in two campaigns with different arm prefixes, so there is no packet
    suffix to split on -- the pair names the arms, and the reduction is the one every other panel
    runs (``population.kernel_medians``). TREATMENT first, matching ``--pair TREATMENT,CONTROL``."""
    frame = pd.DataFrame(observation_rows(BLIND_PAIR[0], 2.0, 150e3) + observation_rows(BLIND_PAIR[1], 4.0, 200e3))
    points = plot.pair_points(frame, [BLIND_PAIR], "no-score")

    by_side = points.set_index("skills")
    assert set(points.leg) == {"C +skills"}
    assert set(points.model) == {"qwen38"}
    assert by_side.loc[True].log2_speedup == pytest.approx(1.0)
    assert by_side.loc[False].log2_speedup == pytest.approx(2.0)
    assert by_side.loc[True].tokens == pytest.approx(150e3)


def model_split_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One comparison over three models, the shape :func:`plot.model_rows` splits."""
    stats = pd.DataFrame(
        [
            {
                "model": model,
                "language": "c",
                "score_verdict": efficacy.NOT_SIGNIFICANT,
                "cost_verdict": efficacy.NOT_SIGNIFICANT,
                "family_size": 6,
            }
            for model in ("qwen38", "oss120b", "kimi27sglang")
        ]
    )
    absolute = pd.DataFrame(
        [
            {"model": model, "language": "c", "skills": skills, **thin(1.0 + index, 1000.0)}
            for index, model in enumerate(("qwen38", "oss120b", "kimi27sglang"))
            for skills in (False, True)
        ]
    )
    return stats, absolute


def test_one_comparison_splits_into_one_row_per_model_in_registry_order() -> None:
    """A single label column holding every arm of a twelve-arm comparison is unreadable; a row per
    model gives each its own column, and the shape still says which model a mark is."""
    stats, absolute = model_split_frames()
    rows = plot.model_rows(absolute, stats, "no-score")
    assert [row.title for row in rows] == [
        experiment_tags.model_name(m) for m in palette.in_order(["qwen38", "oss120b", "kimi27sglang"])
    ]
    assert all(row.treatment == "no-score" for row in rows)
    assert all(len(row.absolute) == 2 and len(row.stats) == 1 for row in rows)


def test_a_model_row_figure_draws_two_square_panels_per_model_under_one_legend() -> None:
    """Every row is the same two panels, and the legend belongs to the figure: the rows draw the
    same intervention in the same colour and differ only in which model's shapes they carry."""
    import matplotlib.pyplot as plt

    stats, absolute = model_split_frames()
    fig = plot.build_treatments_figure(plot.model_rows(absolute, stats, "no-score"), "No Score Tool")
    try:
        assert len(fig.axes) == 3 * len(plot.PANELS)
        assert all(ax.get_box_aspect() == pytest.approx(1.0) for ax in fig.axes)
        assert all(ax.get_legend() is None for ax in fig.axes)
        assert len(fig.legends) == 1
    finally:
        plt.close(fig)


def test_a_model_row_is_titled_by_its_model_and_not_by_the_intervention() -> None:
    """The intervention is one for the whole figure and the title says it once; each row has to say
    which model it is, or three identical rows carry no way to tell them apart."""
    import matplotlib.pyplot as plt

    stats, absolute = model_split_frames()
    fig = plot.build_treatments_figure(plot.model_rows(absolute, stats, "no-score"), "No Score Tool")
    try:
        texts = {text.get_text() for text in fig.texts}
        assert {experiment_tags.model_name("qwen38"), experiment_tags.model_name("oss120b")} <= texts
        assert packets.label("no-score") not in texts - {"No Score Tool"}
    finally:
        plt.close(fig)
