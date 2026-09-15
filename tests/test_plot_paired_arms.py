# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``scripts/plot_paired_arms.py``: the forest plot over any ``experiments/paired_arms.py`` family
CSV -- one row per pair, drawn from the table's own columns, never a recomputed statistic.
"""

import importlib.util
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.transforms
import pandas as pd
import pytest

from hpcagent_bench import experiment_tags
from hpcagent_bench.harness import efficacy
from hpcagent_bench.stats import palette
from hpcagent_bench.stats import style as plotstyle

REPO = pathlib.Path(__file__).resolve().parents[1]

#: A tiny two-pair, two-leg family shaped exactly like ``experiments/paired_arms.py --out`` writes:
#: one row per (pair, leg), the same columns, the tokens leg absent for a pair with no call rows.
PAIR_ROWS = [
    {
        "family": "blind-vs-scored",
        "arm_a": "cpf-llr-focus40-oss120b-c",
        "arm_b": "llrblind-oss120b-c",
        "baseline": "numba",
        "n_a": 38,
        "n_b": 35,
        "n_both": 34,
        "n_only_a": 4,
        "n_only_b": 1,
        "coverage_p": 0.37,
        "leg": "speedup",
        "n_pairs": 34,
        "n_tested": 34,
        "estimate_a_over_b": 1.42,
        "ci_low": 1.10,
        "ci_high": 1.83,
        "wins_a": 22,
        "wins_b": 12,
        "ties": 0,
        "method": "hodges-lehmann",
        "p_value": 0.01,
        "p_adjusted": 0.02,
        "verdict": efficacy.SIGNIFICANT,
    },
    {
        "family": "blind-vs-scored",
        "arm_a": "cpf-llr-focus40-oss120b-c",
        "arm_b": "llrblind-oss120b-c",
        "baseline": "numba",
        "n_a": 38,
        "n_b": 35,
        "n_both": 34,
        "n_only_a": 4,
        "n_only_b": 1,
        "coverage_p": 0.37,
        "leg": "tokens",
        "n_pairs": 30,
        "n_tested": 30,
        "estimate_a_over_b": 2.10,
        "ci_low": 1.60,
        "ci_high": 2.75,
        "wins_a": 25,
        "wins_b": 5,
        "ties": 0,
        "method": "hodges-lehmann",
        "p_value": 0.001,
        "p_adjusted": 0.004,
        "verdict": efficacy.SIGNIFICANT,
    },
    {
        "family": "blind-vs-scored",
        "arm_a": "cpf-llr-focus40-qwen38-fortran",
        "arm_b": "llrblind-qwen38-fortran",
        "baseline": "numba",
        "n_a": 26,
        "n_b": 24,
        "n_both": 20,
        "n_only_a": 6,
        "n_only_b": 4,
        "coverage_p": 0.75,
        "leg": "speedup",
        "n_pairs": 3,
        "n_tested": 3,
        "estimate_a_over_b": 1.05,
        "ci_low": math.nan,
        "ci_high": math.nan,
        "wins_a": 2,
        "wins_b": 1,
        "ties": 0,
        "method": "hodges-lehmann",
        "p_value": math.nan,
        "p_adjusted": math.nan,
        "verdict": efficacy.UNDERPOWERED,
    },
    # No "tokens" leg for this pair: paired_arms.py found no call rows both arms share. The plot
    # must draw an empty right-panel row for it rather than raising or renumbering the rows below.
]


def load_script():
    """Import ``scripts/plot_paired_arms.py`` as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("plot_paired_arms", REPO / "scripts" / "plot_paired_arms.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plot = load_script()


def table() -> pd.DataFrame:
    return pd.DataFrame(PAIR_ROWS)


# ---------------------------------------------------------------------------
# Reading the table: never a recomputed statistic, always the table's own row order and columns.


def test_pair_order_is_the_tables_own_first_appearance_order() -> None:
    pairs = plot.pair_order(table())
    assert pairs == [
        ("cpf-llr-focus40-oss120b-c", "llrblind-oss120b-c"),
        ("cpf-llr-focus40-qwen38-fortran", "llrblind-qwen38-fortran"),
    ]


def test_rows_for_reads_the_tables_own_estimate_and_interval_not_a_recomputation() -> None:
    pairs = plot.pair_order(table())
    rows = plot.rows_for(table(), "speedup", pairs)
    assert rows[0].estimate == pytest.approx(1.42)
    assert (rows[0].low, rows[0].high) == pytest.approx((1.10, 1.83))
    assert rows[0].significant is True


def test_rows_for_marks_a_pair_missing_the_leg_as_a_blank_row() -> None:
    """The oss120b/qwen38-fortran pair has no tokens leg row at all. Both panels must still share
    one y axis, so the missing leg comes back as a finite-free row rather than shrinking the list."""
    pairs = plot.pair_order(table())
    rows = plot.rows_for(table(), "tokens", pairs)
    assert len(rows) == len(pairs)
    assert math.isnan(rows[1].estimate)


def test_rows_for_reads_the_underpowered_verdict_as_not_significant() -> None:
    pairs = plot.pair_order(table())
    rows = plot.rows_for(table(), "speedup", pairs)
    assert rows[1].significant is False


def test_pair_label_names_both_arms_from_the_table_not_a_parsed_convention() -> None:
    assert plot.pair_label(("cpf-llr-focus40-oss120b-c", "llrblind-oss120b-c")) == (
        "cpf-llr-focus40-oss120b-c\nvs llrblind-oss120b-c"
    )


# ---------------------------------------------------------------------------
# identity_label: model/language/shared-packet, read from the arm names -- real names of both
# experiments this script draws (llrblind-vs-scored, git-scicomp).


def test_identity_label_names_model_language_and_a_packet_both_arms_share() -> None:
    """llrblind-vs-scored's skilled pair: the scored campaign against its blind control, same
    model, language and packet on both sides -- exactly the identity a reader wants, not the two
    campaign-prefixed arm names."""
    pair = ("cpf-llr-focus40-qwen38-c-skills", "llrblind-qwen38-c-skills")
    assert plot.identity_label(pair) == "Qwen3.8-27B, C, All Skill Pages"


def test_identity_label_omits_a_packet_neither_arm_carries() -> None:
    """The unskilled llrblind-vs-scored pair: no packet on either side, so nothing is said about
    one -- ``packets.label("")`` reads "No Skill Packet", which would misname a pair that never ran
    a skill packet in the first place."""
    pair = ("cpf-llr-focus40-oss120b-fortran", "llrblind-oss120b-fortran")
    assert plot.identity_label(pair) == "GPT-OSS-120B, Fortran"


def test_identity_label_drops_a_packet_that_differs_between_the_two_arms() -> None:
    """git-scicomp's repo-vs-kernel pair: the two arms name the SAME model, no language token, and
    a packet-like condition (``repo``/``kernel``) that differs -- exactly the comparison the pair
    exists to make. The title ("Repository vs Kernel...") and the axis's own "(a / b)" already say
    which side is which, so the row names only what both arms share: the model."""
    pair = ("git-scicomp-qwen38-repo", "git-scicomp-qwen38-kernel")
    assert plot.identity_label(pair) == "Qwen3.8-27B"


def test_identity_label_falls_back_to_raw_when_nothing_resolves() -> None:
    pair = ("control-arm", "treated-arm")
    assert plot.identity_label(pair) == plot.pair_label(pair)


def test_row_labels_disambiguates_two_pairs_that_resolve_to_one_identity() -> None:
    """Two DIFFERENT pairs (different raw arm names) that happen to share every part
    ``identity_label`` names collapse to one string on their own; ``row_labels`` must still hand
    the figure two distinct row labels, not one repeated twice."""
    pairs = [
        ("cpf-llr-focus40-qwen38-c-skills", "llrblind-qwen38-c-skills"),
        ("cpf-llr-focus40-qwen38-c-skills-v2", "llrblind-qwen38-c-skills-v2"),
    ]
    labels = plot.row_labels(pairs, "identity")
    assert labels[pairs[0]] == "Qwen3.8-27B, C, All Skill Pages"
    assert labels[pairs[1]] == "Qwen3.8-27B, C, All Skill Pages (2)"
    assert len(set(labels.values())) == 2


def test_row_labels_raw_mode_is_the_old_two_line_arm_name_label() -> None:
    pairs = [("cpf-llr-focus40-oss120b-c", "llrblind-oss120b-c")]
    assert plot.row_labels(pairs, "raw")[pairs[0]] == plot.pair_label(pairs[0])


# ---------------------------------------------------------------------------
# Colour and marker: the model registry, never a hardcoded prefix.


def test_named_model_reads_the_token_either_arm_shares_with_the_registry() -> None:
    pair = ("cpf-llr-focus40-oss120b-c", "llrblind-oss120b-c")
    assert plot.named_model(pair, ["oss120b", "qwen38"]) == "oss120b"


def test_named_model_is_blank_for_a_pair_naming_no_registered_model() -> None:
    pair = ("control-arm", "treated-arm")
    assert plot.named_model(pair, ["oss120b", "qwen38"]) == ""


def test_pair_style_falls_back_to_a_neutral_mark_for_an_unregistered_pair() -> None:
    """No registered model is a neutral circle; no packet difference is the CONTROL colour, which
    is the palette's own word for "the thing a treatment is read against", not an ink constant."""
    pairs = [("control-arm", "treated-arm")]
    colors, shapes = plot.pair_style(pairs, ["oss120b", "qwen38"])
    assert colors[pairs[0]] == palette.control_color()
    assert shapes[pairs[0]] == "o"


def test_a_pairs_colour_is_the_intervention_it_tests_and_its_shape_is_the_model() -> None:
    """Colour is the INTERVENTION, shape is the MODEL. Colouring by model spent the intervention's
    channel on the entity the shape already carries, so a family comparing three packets drew all
    three in one hue per model and the same colour meant a different treatment in every row."""
    pair = ("cpf-llr-focus40-oss120b-c-cpfsrc", "cpf-llr-focus40-oss120b-c")
    colors, shapes = plot.pair_style([pair], ["oss120b", "qwen38"])

    assert colors[pair] == palette.color("cpfsrc")
    assert colors[pair] != palette.model_color("oss120b")
    assert shapes[pair] == palette.marker("oss120b")


def test_two_arms_naming_one_packet_are_not_a_packet_comparison_and_keep_the_control_colour() -> None:
    """A family whose variable is something else -- an episode count, a repeat policy -- must not
    borrow the hue of a packet BOTH of its arms carried, which would claim a treatment it is not
    testing."""
    pair = ("git-scicomp-oss120b-c-repo", "git-scicomp-qwen38-c-repo")
    colors, _shapes = plot.pair_style([pair], ["oss120b", "qwen38"])
    assert colors[pair] == palette.control_color()


def test_the_legend_names_every_model_by_shape_and_every_intervention_by_colour() -> None:
    """One channel per entity, each drawn the way the rows draw it: a model handle in its own hue
    would claim a channel the rows spend on the intervention."""
    pairs = [
        ("cpf-llr-focus40-oss120b-c-cpfsrc", "cpf-llr-focus40-oss120b-c"),
        ("cpf-llr-focus40-qwen38-c-cpf", "cpf-llr-focus40-qwen38-c"),
    ]
    colors, shapes = plot.pair_style(pairs, list(experiment_tags.order("models")))
    handles = plot.model_legend(pairs, colors, shapes)
    labels = [handle.get_label() for handle in handles]

    assert experiment_tags.model_name("oss120b") in labels
    assert experiment_tags.packet_name("cpfsrc") in labels
    assert experiment_tags.packet_name("cpf") in labels
    for handle in handles:
        if handle.get_label() in {experiment_tags.model_name(m) for m in ("oss120b", "qwen38")}:
            assert handle.get_color() == plotstyle.MUTED


def test_the_ratio_axis_carries_a_major_grid_and_no_minor_one() -> None:
    """Major grid only, on the measured axis. The row axis carries names, where a guide line per
    category measures nothing."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        plot.style_ratio_axis(ax, [plot.Row(("a", "b"), 1.1, 1.0, 1.2, False, 5)])
        assert any(line.get_visible() for line in ax.xaxis.get_gridlines())
        assert not [tick for tick in ax.xaxis.get_minor_ticks() if tick.gridline.get_visible()]
        assert not [tick for tick in ax.yaxis.get_major_ticks() if tick.gridline.get_visible()]
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# The log2 ratio axis: per_kernel.py's own tick and 1x-line rule, reused rather than reimplemented.


def test_ratio_ticks_always_spans_at_least_a_quarter_to_four_x() -> None:
    rows = [plot.Row(("a", "b"), 1.1, 1.0, 1.2, False, 5)]
    ticks = plot.ratio_ticks(rows)
    assert 0.25 in ticks and 1.0 in ticks and 4.0 in ticks


def test_ratio_ticks_grows_to_cover_a_wide_estimate() -> None:
    rows = [plot.Row(("a", "b"), 20.0, 15.0, 25.0, False, 5)]
    ticks = plot.ratio_ticks(rows)
    assert max(ticks) >= 25.0


def test_ratio_ticks_skips_a_blank_row() -> None:
    """A pair with no leg (:func:`rows_for`'s NaN row) must not push the axis to NaN."""
    rows = [plot.Row(("a", "b"), math.nan, math.nan, math.nan, False, 0)]
    assert plot.ratio_ticks(rows) == plot.ratio_ticks([])


# ---------------------------------------------------------------------------
# The gap between the two panels: wide enough that the boundary tick labels never touch.


def boundary_label_boxes(
    fig: matplotlib.figure.Figure,
) -> tuple[matplotlib.transforms.Bbox, matplotlib.transforms.Bbox]:
    """The left panel's rightmost tick label and the right panel's leftmost one, as rendered."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_speed, ax_tokens = fig.axes[0], fig.axes[1]
    left = ax_speed.get_xticklabels()[-1].get_window_extent(renderer)
    right = ax_tokens.get_xticklabels()[0].get_window_extent(renderer)
    return left, right


def test_the_two_panels_boundary_tick_labels_do_not_overlap_after_a_draw() -> None:
    fig = plot.build_figure(table(), "gap check", False)
    try:
        left, right = boundary_label_boxes(fig)
        assert left.x1 <= right.x0
    finally:
        plt.close(fig)


def test_a_wide_ratio_range_still_clears_the_gap_after_it_widens() -> None:
    """A family whose estimates span more decades gets WIDER edge labels ("1/16x", "16x") than the
    default gap was sized for; the gap must grow to clear those too, not just the narrow case."""
    wide = pd.DataFrame(
        [
            {**PAIR_ROWS[0], "estimate_a_over_b": 18.0, "ci_low": 12.0, "ci_high": 22.0},
            {**PAIR_ROWS[1], "estimate_a_over_b": 0.07, "ci_low": 0.05, "ci_high": 0.09},
        ]
    )
    fig = plot.build_figure(wide, "wide range", False)
    try:
        left, right = boundary_label_boxes(fig)
        assert left.x1 <= right.x0
    finally:
        plt.close(fig)


def test_a_long_ratio_label_widens_the_gap_until_the_axis_labels_clear() -> None:
    """Each axis label is centred under its own panel, so a long contrast runs under the
    neighbouring panel's label unless the gap grows for the labels too, not only the ticks."""
    fig = plot.build_figure(table(), "gap check", False, ratio="Repository / Kernel")
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        left = fig.axes[0].xaxis.label.get_window_extent(renderer)
        right = fig.axes[1].xaxis.label.get_window_extent(renderer)
        assert left.x1 <= right.x0, (left.x1, right.x0)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# End to end: a tiny synthetic CSV, exactly the shape paired_arms.py --out writes.


def test_the_script_draws_one_pdf_and_one_png_over_a_synthetic_csv(tmp_path: pathlib.Path) -> None:
    csv_path = tmp_path / "paired.csv"
    table().to_csv(csv_path, index=False)
    out = tmp_path / "figures" / "blind_vs_scored"

    result = plot.main([str(csv_path), "--label", "Blind vs Scored", "--out", str(out.with_suffix(".pdf"))])

    assert result == 0
    assert out.with_suffix(".pdf").exists()
    assert out.with_suffix(".png").exists()


def test_the_ratio_axes_name_which_arm_is_the_numerator() -> None:
    """A row label names only what both arms share, so the axis is the one place a reader learns
    which side of the contrast is on top of the ratio."""
    fig = plot.build_figure(table(), "Blind vs Scored", False, ratio="Scored / Blind")
    speed, tokens = fig.axes[:2]
    assert speed.get_xlabel() == "Speedup Ratio (Scored / Blind)", speed.get_xlabel()
    assert tokens.get_xlabel() == "Token Ratio (Scored / Blind)", tokens.get_xlabel()
    plt.close(fig)


def test_build_figure_raises_on_an_empty_table() -> None:
    with pytest.raises(SystemExit, match="no rows"):
        plot.build_figure(pd.DataFrame(columns=list(PAIR_ROWS[0])), "empty", False)
