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

import pandas as pd
import pytest

from hpcagent_bench.harness import efficacy
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
# Colour and marker: the model registry, never a hardcoded prefix.


def test_named_model_reads_the_token_either_arm_shares_with_the_registry() -> None:
    pair = ("cpf-llr-focus40-oss120b-c", "llrblind-oss120b-c")
    assert plot.named_model(pair, ["oss120b", "qwen38"]) == "oss120b"


def test_named_model_is_blank_for_a_pair_naming_no_registered_model() -> None:
    pair = ("control-arm", "treated-arm")
    assert plot.named_model(pair, ["oss120b", "qwen38"]) == ""


def test_pair_style_falls_back_to_a_neutral_mark_for_an_unregistered_pair() -> None:
    pairs = [("control-arm", "treated-arm")]
    colors, shapes = plot.pair_style(pairs, ["oss120b", "qwen38"])
    assert colors[pairs[0]] == plotstyle.MUTED
    assert shapes[pairs[0]] == "o"


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
# End to end: a tiny synthetic CSV, exactly the shape paired_arms.py --out writes.


def test_the_script_draws_one_pdf_and_one_png_over_a_synthetic_csv(tmp_path: pathlib.Path) -> None:
    csv_path = tmp_path / "paired.csv"
    table().to_csv(csv_path, index=False)
    out = tmp_path / "figures" / "blind_vs_scored"

    result = plot.main([str(csv_path), "--label", "Blind vs Scored", "--out", str(out.with_suffix(".pdf"))])

    assert result == 0
    assert out.with_suffix(".pdf").exists()
    assert out.with_suffix(".png").exists()


def test_build_figure_raises_on_an_empty_table() -> None:
    with pytest.raises(SystemExit, match="no rows"):
        plot.build_figure(pd.DataFrame(columns=list(PAIR_ROWS[0])), "empty", False)
