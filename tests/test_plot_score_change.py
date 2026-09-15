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

import pandas as pd
import pytest

from hpcagent_bench import experiment_tags
from hpcagent_bench.harness import efficacy
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
    fig, ax = plt.subplots()
    try:
        plot.draw_absolute(ax, frame, stats, "skills")
        labels = [text.get_text() for text in ax.texts]
    finally:
        plt.close(fig)
    assert ("C *" in labels) == starred, labels


def treatment_panel(treatment: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """One synthetic (treatment, stats, absolute) triple, the shape :func:`plot.figure_treatments` takes."""
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
    absolute = pd.DataFrame(
        [
            {"model": "qwen38", "language": "c", "skills": False, **thin(1.0, 1000.0)},
            {"model": "qwen38", "language": "c", "skills": True, **thin(1.4, 900.0)},
        ]
    )
    return treatment, stats, absolute


@pytest.mark.parametrize("n", [1, 2, 3])
def test_n_treatments_draw_n_square_panels(n: int) -> None:
    """One axes per treatment, and every one SQUARE -- the point of joining them side by side is
    that a reader compares panel shape as well as content."""
    panels = [treatment_panel(f"treatment{i}") for i in range(n)]
    fig = plot.build_treatments_figure(panels, "label")
    try:
        assert len(fig.axes) == n
        width, height = fig.get_size_inches()
        assert width == pytest.approx(n * plot.SQUARE_PANEL_SIDE + plot.SQUARE_PANEL_GAP * (n - 1))
        assert height == pytest.approx(plot.SQUARE_PANEL_SIDE)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_double_column_caps_a_joined_figures_row_width() -> None:
    """``--double-column`` is the figure's budget on a paper page: N panels must fit inside
    ``style.DOUBLE_COLUMN_WIDTH`` inches total, not N times a fixed natural panel size."""
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
    fig, ax = plt.subplots()
    try:
        labels = [handle.get_label() for handle in plot.draw_absolute(ax, frame, stats, "skills")]
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
    fig, ax = plt.subplots()
    try:
        labels = [handle.get_label() for handle in plot.draw_absolute(ax, frame, stats, treatment)]
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
    fig, ax = plt.subplots()
    try:
        labels = [handle.get_label() for handle in plot.draw_absolute(ax, frame, stats, "cpf")]
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
