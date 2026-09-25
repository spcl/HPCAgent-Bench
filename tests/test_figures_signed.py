# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Consumers of the signed-change figures: the exclusion rules, the pairing rule, and the tables.

The sweep these figures read takes eight hours, so the shapes that break a plotting script -- an
arm whose CSV never appeared, an arm with one usable kernel and therefore no interval, a miscompile
that must not be drawn as "no change" -- are exercised against a synthetic sweep directory instead
of against whichever of them the next real run happens to contain.

The property the paired figure lives or dies on is PAIRING: a kernel only one of the two tools
compiled must leave the comparison entirely, in both directions. A run where that silently stops
holding still draws a plausible-looking figure, so it is asserted here rather than eyeballed.
"""

import csv
import math
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from PIL import Image

from hpcagent_bench.stats import canon, palette, rules, style, summary
from hpcagent_bench.stats.figures import per_kernel, signed

#: The sweep's column order; the fixture writes the real schema, not a convenient subset.
FIELDS = ("framework", "preset", "datatype", "kernel", "impl", "status", "validated", "median_ms", "failure", "error")


def row(framework: str, kernel: str, ms: str, status: str = "ok", validated: str = "True") -> dict[str, str]:
    return {
        "framework": framework,
        "preset": "XL",
        "datatype": "float64",
        "kernel": kernel,
        "impl": "dace",
        "status": status,
        "validated": validated,
        "median_ms": ms,
        "failure": "",
        "error": "",
    }


def write(path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(name="sweep")
def sweep_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    """A sweep directory covering every case the real run can produce.

    Both file shapes appear on purpose: the baseline is sharded the way a four-rank job writes it
    and the arms are unsharded, so a reader of this fixture can see that the two are read the same
    way.
    """
    baseline = [row(signed.BASELINE, f"tsvc_2_s{i}", "100.0") for i in range(1, 7)]
    baseline.append(row(signed.BASELINE, "tsvc_2_slow", "100.0"))
    baseline.append(row(signed.BASELINE, "jacobi_2d", "100.0"))
    write(tmp_path / f"{signed.BASELINE}.rank0.csv", baseline[:4])
    write(tmp_path / f"{signed.BASELINE}.rank1.csv", baseline[4:])

    # A normal spread (2x, 4x, 1.25x, 10x), one kernel SLOWER than the reference, one crash, one
    # unvalidated answer, and one non-TSVC kernel that is out of scope rather than excluded.
    write(
        tmp_path / "dace_cpu_canonicalize.csv",
        [
            row("dace_cpu_canonicalize", "tsvc_2_s1", "50.0"),
            row("dace_cpu_canonicalize", "tsvc_2_s2", "25.0"),
            row("dace_cpu_canonicalize", "tsvc_2_s3", "80.0"),
            row("dace_cpu_canonicalize", "tsvc_2_s4", "10.0"),
            row("dace_cpu_canonicalize", "tsvc_2_slow", "400.0"),
            row("dace_cpu_canonicalize", "tsvc_2_s5", "", status="crash", validated=""),
            row("dace_cpu_canonicalize", "tsvc_2_s6", "1.0", validated="False"),
            row("dace_cpu_canonicalize", "jacobi_2d", "5.0"),
        ],
    )
    # One usable kernel: the t-interval is undefined at n=1 and must not raise.
    write(tmp_path / "dace_cpu.csv", [row("dace_cpu", "tsvc_2_s1", "20.0")])
    # cc_llvm_autopar gets NO file at all -- the arm that never ran.
    return tmp_path


@pytest.fixture(name="paired_sweep")
def paired_sweep_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    """Canon against two comparison arms, carrying every shape the real sweep can produce.

    Canon times s1..s4 plus s_only. dace main times s1..s4 (s3 CRASHED, so it drops out of the
    pair) and s_main_only, which canon never timed. llvm+polly times s1 and s2 only.
    """
    canon = [
        row(signed.REFERENCE, "tsvc_2_s1", "50.0"),
        row(signed.REFERENCE, "tsvc_2_s2", "100.0"),
        row(signed.REFERENCE, "tsvc_2_s3", "100.0"),
        row(signed.REFERENCE, "tsvc_2_s4", "200.0"),
        row(signed.REFERENCE, "tsvc_2_s_only", "10.0"),
    ]
    write(tmp_path / f"{signed.REFERENCE}.rank0.csv", canon[:3])
    write(tmp_path / f"{signed.REFERENCE}.rank1.csv", canon[3:])
    write(
        tmp_path / "dace_cpu.csv",
        [
            row("dace_cpu", "tsvc_2_s1", "100.0"),
            row("dace_cpu", "tsvc_2_s2", "100.0"),
            row("dace_cpu", "tsvc_2_s3", "", status="failed"),
            row("dace_cpu", "tsvc_2_s4", "100.0"),
            row("dace_cpu", "tsvc_2_s_main_only", "10.0"),
        ],
    )
    write(
        tmp_path / "cc_llvm_autopar.csv",
        [row("cc_llvm_autopar", "tsvc_2_s1", "200.0"), row("cc_llvm_autopar", "tsvc_2_s2", "50.0")],
    )
    return tmp_path


def test_rejects_are_counted_not_plotted(sweep: pathlib.Path) -> None:
    arm = signed.read_arm(sweep, "dace_cpu_canonicalize")
    assert set(arm.times) == {"tsvc_2_s1", "tsvc_2_s2", "tsvc_2_s3", "tsvc_2_s4", "tsvc_2_slow"}
    assert arm.rejected["status=crash"] == 1
    assert arm.rejected["not validated"] == 1
    # The non-TSVC kernel is out of scope, so it is neither timed nor counted as an exclusion.
    assert "jacobi_2d" not in arm.times
    assert sum(arm.rejected.values()) == 2


def test_slower_than_baseline_lands_below_zero(sweep: pathlib.Path) -> None:
    arm = signed.read_arm(sweep, "dace_cpu_canonicalize")
    reference = signed.read_arm(sweep, signed.BASELINE)
    values = signed.against_baseline(arm, reference.times)
    assert signed.signed_change(values["tsvc_2_slow"]) == pytest.approx(-3.0)
    assert signed.signed_change(values["tsvc_2_s1"]) == pytest.approx(1.0)


def test_missing_arm_reads_as_empty(sweep: pathlib.Path) -> None:
    arm = signed.read_arm(sweep, "cc_llvm_autopar")
    assert not arm.times and not arm.rejected


def test_render_survives_every_degenerate_arm(sweep: pathlib.Path, tmp_path: pathlib.Path) -> None:
    out = signed.arms_figure(sweep, tmp_path / "figure")
    assert out.with_suffix(".pdf").is_file() and out.with_suffix(".svg").is_file()


def test_missing_baseline_is_fatal(tmp_path: pathlib.Path) -> None:
    write(tmp_path / "dace_cpu.csv", [row("dace_cpu", "tsvc_2_s1", "20.0")])
    with pytest.raises(SystemExit, match=signed.BASELINE):
        signed.arms_figure(tmp_path, tmp_path / "figure")


def test_pairs_only_kernels_both_tools_timed(paired_sweep: pathlib.Path) -> None:
    canon = signed.read_arm(paired_sweep, signed.REFERENCE)
    other = signed.read_arm(paired_sweep, "dace_cpu")
    ratios = signed.paired(canon.times, other.times)
    # s3 crashed in the comparison arm and s_only/s_main_only are one-sided: all three drop out.
    assert sorted(ratios) == ["tsvc_2_s1", "tsvc_2_s2", "tsvc_2_s4"]


def test_ratio_is_other_over_canon(paired_sweep: pathlib.Path) -> None:
    canon = signed.read_arm(paired_sweep, signed.REFERENCE)
    other = signed.read_arm(paired_sweep, "dace_cpu")
    ratios = signed.paired(canon.times, other.times)
    assert ratios["tsvc_2_s1"] == pytest.approx(2.0)  # canon 50ms vs main 100ms -- canon wins
    assert ratios["tsvc_2_s4"] == pytest.approx(0.5)  # canon 200ms vs main 100ms -- canon loses


def test_sign_test_counts_both_directions_and_ignores_ties(paired_sweep: pathlib.Path) -> None:
    canon = signed.read_arm(paired_sweep, signed.REFERENCE)
    other = signed.read_arm(paired_sweep, "dace_cpu")
    wins, losses = signed.sign_test(signed.paired(canon.times, other.times))
    assert (wins, losses) == (1, 1)  # s2 is an exact tie and counts for neither side


def test_narrow_comparison_arm_keeps_its_own_n(paired_sweep: pathlib.Path) -> None:
    canon = signed.read_arm(paired_sweep, signed.REFERENCE)
    polly = signed.read_arm(paired_sweep, "cc_llvm_autopar")
    # polly compiled two kernels, so its row has n=2 -- NOT canon's five.
    assert len(signed.paired(canon.times, polly.times)) == 2


def test_render_survives_a_missing_comparison_arm(paired_sweep: pathlib.Path, tmp_path: pathlib.Path) -> None:
    (paired_sweep / "cc_llvm_autopar.csv").unlink()
    out = signed.paired_figure(paired_sweep, tmp_path / "canon")
    assert out.with_suffix(".pdf").is_file() and out.with_suffix(".svg").is_file()


def test_missing_reference_is_fatal(tmp_path: pathlib.Path) -> None:
    write(tmp_path / "dace_cpu.csv", [row("dace_cpu", "tsvc_2_s1", "1.0")])
    with pytest.raises(SystemExit):
        signed.paired_figure(tmp_path, tmp_path / "canon")


def test_the_table_carries_the_costs_behind_every_ratio(sweep: pathlib.Path) -> None:
    """SC15 Rule 4: a speed-up alone is uninterpretable, so the milliseconds travel with it."""
    frame = signed.table(signed.arm_rows(sweep))
    assert list(frame.columns) == list(signed.TABLE_COLUMNS)
    canon = frame[(frame.framework == "dace_cpu_canonicalize") & (frame.kernel == "tsvc_2_s1")].iloc[0]
    assert canon.numerator_ms == pytest.approx(100.0) and canon.denominator_ms == pytest.approx(50.0)
    assert canon.speedup == pytest.approx(2.0) and canon.signed_change == pytest.approx(1.0)


def test_a_ratio_table_with_no_costs_is_refused() -> None:
    """The rule is a check, not a comment: a ratio with no costs behind it has to fail loudly."""
    with pytest.raises(rules.RuleViolation, match="Rule 4"):
        rules.require_costs(pd.DataFrame({"speedup": [1.4]}), "speedup", ("numerator_ms",))


def test_every_summarized_row_carries_an_interval(sweep: pathlib.Path) -> None:
    """SC15 Rules 5 and 7: a nondeterministic measurement is never a bare point estimate."""
    frame = signed.summary_table(signed.arm_rows(sweep))
    assert list(frame.columns) == list(signed.SUMMARY_COLUMNS)
    multi = frame[frame.row == "dace canon"].iloc[0]
    assert multi.geomean_low < multi.geomean < multi.geomean_high
    # One usable kernel has no spread to estimate, so its interval collapses onto the point rather
    # than pretending to bound something.
    single = frame[frame.row == "dace main"].iloc[0]
    assert single.n == 1 and single.geomean_low == pytest.approx(single.geomean_high)


def test_an_arm_that_never_ran_is_reported_not_dropped(sweep: pathlib.Path) -> None:
    frame = signed.summary_table(signed.arm_rows(sweep))
    absent = frame[frame.row == "llvm + polly"].iloc[0]
    assert absent.n == 0 and absent.excluded == "none"


def test_the_tables_are_written_beside_the_figure(sweep: pathlib.Path, tmp_path: pathlib.Path) -> None:
    signed.arms_figure(sweep, tmp_path / "figure")
    assert (tmp_path / "figure-kernels.csv").is_file() and (tmp_path / "figure-summary.csv").is_file()


def test_a_sweep_with_no_overlap_draws_nothing_rather_than_failing_a_rule(tmp_path: pathlib.Path) -> None:
    """A baseline that shares no kernel with any arm has nothing to summarize, which is an empty
    figure and not a broken rule -- the checks guard a ratio that IS reported, never the absence."""
    write(tmp_path / f"{signed.BASELINE}.csv", [row(signed.BASELINE, "tsvc_2_s1", "100.0")])
    write(tmp_path / "dace_cpu_canonicalize.csv", [row("dace_cpu_canonicalize", "tsvc_2_s9", "50.0")])
    rows = signed.arm_rows(tmp_path)
    assert signed.table(rows).empty
    assert (signed.summary_table(rows)["n"] == 0).all()
    out = signed.arms_figure(tmp_path, tmp_path / "empty")
    assert out.with_suffix(".pdf").is_file()


# --------------------------------------------------------------------------------------------
# llr-focus40 compiler figure: DaCe's own canon-sweep columns beside every model's CPF arm, all
# against numba, drawn by per_kernel on its log2 ratio axis. The row-building
# helpers below stand in for canon.read_times/population.kernel_answers/graded_episode_rows
# without a real sweep or a real campaign DB.
# --------------------------------------------------------------------------------------------

ROSTER40: tuple[str, ...] = ("k1", "k2", "k3")

#: A roster wide enough for the token geomean's interval (summary.MIN_PAIRS_FOR_INTERVAL kernels): a
#: tokens table over fewer has no interval on any row and is refused under Rule 5.
ROSTER_TOKENS: tuple[str, ...] = ("k1", "k2", "k3", "k4", "k5", "k6")


def canon_table(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """A ``canon`` table frame: (column, kernel, median_ms), always validated."""
    return pd.DataFrame(
        [{"run": "r1", "column": column, "kernel": kernel, "median_ms": ms, "validated": "True"}
         for column, kernel, ms in rows]
    )  # fmt: skip


def episode_row(arm: str, benchmark: str, speedup: float, run_suffix: str = "1") -> dict[str, object]:
    """One ``record=submission`` episode row: what ``population.kernel_answers`` and
    ``population.graded_episode_rows`` both need."""
    return {
        "run_root": f"j{run_suffix}",
        "job": f"j{run_suffix}",
        "run_id": f"{arm}-{benchmark}-{run_suffix}",
        "arm": arm,
        "record": "submission",
        "benchmark": benchmark,
        "speedup": speedup,
        "baseline_ns": 1.0e6,  # 1 ms, in nanoseconds -- ANSWER_COLUMNS' own unit
        "native_ns": 1.0e6 / speedup,
        "baseline": "numba",
        "suspect": 0,
        "ts_ms": int(run_suffix),
        "attempt_index": 1,
        "timing_reduction": "mwd-v2",
    }


def token_row(arm: str, benchmark: str, tokens: float, run_suffix: str = "1") -> dict[str, object]:
    """One ``record=task`` row: what ``population.kernel_tokens`` reads a spend off (spec T4)."""
    return {
        "run_root": f"j{run_suffix}", "job": f"j{run_suffix}", "run_id": f"{arm}-{benchmark}-{run_suffix}",
        "arm": arm, "record": "task", "benchmark": benchmark, "tokens": tokens, "ts_ms": int(run_suffix),
    }  # fmt: skip


@pytest.fixture(name="llr40_canon")
def llr40_canon_fixture() -> pd.DataFrame:
    """dace_cpu and dace_cpu_canonicalize against numba: k1 and k2 validated on every column, k3
    validated on numba alone (a compiler that never timed it) -- ROSTER40 names all three."""
    return canon_table(
        [
            ("numba", "k1", 100.0), ("numba", "k2", 200.0), ("numba", "k3", 100.0),
            ("dace_cpu", "k1", 50.0), ("dace_cpu", "k2", 100.0),
            ("dace_cpu_canonicalize", "k1", 10.0), ("dace_cpu_canonicalize", "k2", 20.0),
        ]
    )  # fmt: skip


@pytest.fixture(name="llr40_observations")
def llr40_observations_fixture() -> pd.DataFrame:
    """Two models, each with a ``cpf`` and a ``cpfsrc`` arm, complete over ROSTER_TOKENS (and so
    over ROSTER40); qwen38's cpfsrc arm runs k1 twice (a repeat, for the per-kernel interval),
    everything else once."""
    rows: list[dict[str, object]] = []
    for model in ("qwen38", "oss120b"):
        for condition, suffix, speedups in (
            ("cpf", "c-cpf", {"k1": 2.0, "k2": 3.0, "k3": 1.5, "k4": 1.2, "k5": 4.0, "k6": 1.4}),
            ("cpfsrc", "c-cpfsrc", {"k1": 2.5, "k2": 3.5, "k3": 1.8, "k4": 1.1, "k5": 5.0, "k6": 1.6}),
        ):
            arm = f"cpf-llr-focus40-{model}-{suffix}"
            for index, (benchmark, speedup) in enumerate(speedups.items()):
                rows.append(episode_row(arm, benchmark, speedup))
                rows.append(token_row(arm, benchmark, 1000.0 + 100.0 * index))
            if model == "qwen38" and condition == "cpfsrc":
                # A second, slightly different episode of k1: the per-kernel interval this row's
                # ratios_low/ratios_high bound is over THESE repeats, not over the kernel axis.
                # Its own task row, or "latest" would supersede k1's only token measurement with a
                # run that spent none (population.latest_runs: a rerun with no persisted task still
                # supersedes the earlier one).
                rows.append(episode_row(arm, "k1", 2.7, run_suffix="2"))
                rows.append(token_row(arm, "k1", 1200.0, run_suffix="2"))
    return pd.DataFrame(rows)


def test_canon_row_matches_the_ratio_and_scopes_to_the_roster(llr40_canon: pd.DataFrame) -> None:
    row = signed.canon_kernel_row(llr40_canon, "dace_cpu_canonicalize", ROSTER40)
    assert row.ratios == {"k1": pytest.approx(10.0), "k2": pytest.approx(10.0), "k3": 1.0}
    assert row.numerator_ms["k1"] == pytest.approx(100.0) and row.numerator_ms["k2"] == pytest.approx(200.0)
    assert row.denominator_ms["k1"] == pytest.approx(10.0) and row.denominator_ms["k2"] == pytest.approx(20.0)
    # k3 is FILLED at 1x, not dropped (2026-09-20 rule): numba timed it, dace_cpu_canonicalize
    # never did, and that "no result" is flagged rather than made to look like a real measurement.
    assert row.ratios["k3"] == 1.0 and math.isnan(row.denominator_ms["k3"])
    assert row.delivered == {"k1": True, "k2": True, "k3": False}
    assert row.color == palette.framework_color("dace_cpu_canonicalize")
    assert row.marker == palette.marker("cpf")
    assert row.label == "Canonical Parallel Form"
    # No repetition and no agent: nothing to bound, nothing spent.
    assert row.ratios_low == {} and row.ratios_high == {} and row.tokens == {}


def test_canon_row_ignores_kernels_outside_the_roster(llr40_canon: pd.DataFrame) -> None:
    """Regression: a canon sweep commonly spans more kernels than one figure's roster. A row that
    is not scoped to the roster would let the summary geomean a population the panel never drew."""
    row = signed.canon_kernel_row(llr40_canon, "dace_cpu", ("k1",))
    assert set(row.ratios) == {"k1"}


def test_two_arms_of_one_model_share_shape_and_differ_in_hue(llr40_observations: pd.DataFrame) -> None:
    """Channel rule: SHAPE is the LLM, COLOUR is the packet/condition."""
    cpf = signed.agent_kernel_row(llr40_observations, "cpf-llr-focus40-qwen38-c-cpf", "qwen38", "cpf", ROSTER40)
    cpfsrc = signed.agent_kernel_row(
        llr40_observations, "cpf-llr-focus40-qwen38-c-cpfsrc", "qwen38", "cpfsrc", ROSTER40
    )
    assert cpf.marker == cpfsrc.marker == palette.marker("qwen38")
    assert cpf.color != cpfsrc.color
    assert cpf.color == palette.color("cpf") and cpfsrc.color == palette.color("cpfsrc")


def test_two_models_same_condition_share_hue_and_differ_in_shape(llr40_observations: pd.DataFrame) -> None:
    qwen = signed.agent_kernel_row(llr40_observations, "cpf-llr-focus40-qwen38-c-cpfsrc", "qwen38", "cpfsrc", ROSTER40)
    oss = signed.agent_kernel_row(llr40_observations, "cpf-llr-focus40-oss120b-c-cpfsrc", "oss120b", "cpfsrc", ROSTER40)
    assert qwen.color == oss.color == palette.color("cpfsrc")
    assert qwen.marker != oss.marker
    assert qwen.marker == palette.marker("qwen38") and oss.marker == palette.marker("oss120b")


def test_agent_row_carries_rule4_costs_and_a_repeat_interval(llr40_observations: pd.DataFrame) -> None:
    row = signed.agent_kernel_row(llr40_observations, "cpf-llr-focus40-qwen38-c-cpfsrc", "qwen38", "cpfsrc", ROSTER40)
    assert row.ratios["k1"] > 0.0 and row.numerator_ms["k1"] == pytest.approx(1.0)
    # k1 ran twice (2.5x and 2.7x): its interval is a real band, not a degenerate point.
    assert row.ratios_low["k1"] < row.ratios_high["k1"]
    # k2 ran once: one sample has no spread to estimate, so its interval collapses onto the point
    # (geomean_ci's own contract) rather than being omitted or fabricated.
    assert row.ratios_low["k2"] == pytest.approx(row.ratios_high["k2"])
    assert row.tokens["k1"] == pytest.approx(1200.0)  # "latest" run's own task total, not k1's first


def test_the_summary_slot_leaves_out_a_compilers_1x_placeholders(llr40_canon: pd.DataFrame) -> None:
    """dace_cpu_canonicalize never timed k3, which is drawn crossed at 1x; entering that 1x would
    make the summary partly a statement about coverage (2026-09-21: solved kernels only). Its two
    solved kernels are both 10x, so the printed geomean is 10x -- not 4.6x with the placeholder."""
    rows = signed.llr40_rows(llr40_canon, None, ROSTER40)
    fig = signed.llr40_figure(rows, ROSTER40)
    try:
        printed = [text.get_text() for text in fig.axes[0].texts if text.get_gid() == style.CLEAR_GID]
    finally:
        plt.close(fig)
    assert style.ratio_label(10.0) in printed, printed
    assert style.ratio_label(100.0 ** (1.0 / 3.0)) not in printed, printed


def test_rows_without_observations_draws_canon_only(llr40_canon: pd.DataFrame) -> None:
    rows = signed.llr40_rows(llr40_canon, None, ROSTER40)
    assert [row.framework for row in rows] == list(signed.LLR40_CANON_COLUMNS)


def test_rows_keep_only_roster_complete_conditions(llr40_canon: pd.DataFrame, llr40_observations: pd.DataFrame) -> None:
    rows = signed.llr40_rows(llr40_canon, llr40_observations, ROSTER40)
    arms = {row.framework for row in rows}
    assert arms == {
        "dace_cpu", "dace_cpu_canonicalize",
        "cpf-llr-focus40-qwen38-c-cpf", "cpf-llr-focus40-qwen38-c-cpfsrc",
        "cpf-llr-focus40-oss120b-c-cpf", "cpf-llr-focus40-oss120b-c-cpfsrc",
    }  # fmt: skip


def test_adding_compiler_columns_does_not_change_any_agent_rows_ratios(
    llr40_canon: pd.DataFrame, llr40_observations: pd.DataFrame
) -> None:
    """Wiring Pluto/ppcg_hip into the figure (``statistics/plot_llr40_compilers.py``'s own
    ``--canon-columns`` default, 2026-09-20) only ADDS rows -- it must never change an agent arm's
    own per-kernel speed-up (its S_i). ``pluto``/``ppcg_hip`` are absent from ``llr40_canon`` here
    (never a validated row, exactly the historical ppcg canon sweep, job 640520), so every roster
    kernel on those two rows fills at 1x -- and every agent row's ratios must be BIT-IDENTICAL to
    the two-column baseline."""
    columns_2 = signed.LLR40_CANON_COLUMNS
    columns_4 = (*columns_2, "pluto", "ppcg_hip")
    before = {
        row.framework: row.ratios
        for row in signed.llr40_rows(llr40_canon, llr40_observations, ROSTER40, canon_columns=columns_2)
    }
    after = {
        row.framework: row.ratios
        for row in signed.llr40_rows(llr40_canon, llr40_observations, ROSTER40, canon_columns=columns_4)
    }
    agent_arms = [key for key in before if key not in columns_2]
    assert agent_arms  # the fixture must actually carry agent rows, or this test proves nothing
    for arm in agent_arms:
        assert after[arm] == before[arm], arm
    assert "pluto" not in before and "ppcg_hip" not in before
    assert "pluto" in after and "ppcg_hip" in after
    rows_4 = signed.llr40_rows(llr40_canon, llr40_observations, ROSTER40, canon_columns=columns_4)
    pluto_row = next(row for row in rows_4 if row.framework == "pluto")
    assert pluto_row.ratios == {k: 1.0 for k in ROSTER40}  # no validated pluto row anywhere -> every kernel 1x
    assert pluto_row.delivered == {k: False for k in ROSTER40}


def test_llr40_figure_renders_with_missing_marks_and_rule_checked_tables(
    llr40_canon: pd.DataFrame, llr40_observations: pd.DataFrame, tmp_path: pathlib.Path
) -> None:
    out = tmp_path / "llr40"
    stem = signed.llr40_two_row_figure(llr40_canon, llr40_observations, ROSTER_TOKENS, out, dpi=72.0)
    assert stem.with_suffix(".pdf").is_file() and stem.with_suffix(".png").is_file()
    kernels = pd.read_csv(tmp_path / "llr40-kernels.csv")
    # dace_cpu never timed k3 (test_canon_row_matches_the_ratio_and_scopes_to_the_roster): the
    # emitted table carries it at 1x rather than dropping the row (2026-09-20 rule).
    dace_k3 = kernels[(kernels.framework == "dace_cpu") & (kernels.kernel == "k3")]
    assert len(dace_k3) == 1 and dace_k3["speedup"].iloc[0] == pytest.approx(1.0)
    summary = pd.read_csv(tmp_path / "llr40-summary.csv")
    assert (summary["n"] > 0).all()
    assert (summary["geomean_low"] <= summary["geomean"]).all()
    assert (summary["geomean"] <= summary["geomean_high"]).all()
    token_summary = pd.read_csv(tmp_path / "llr40-tokens-summary.csv")
    # The two canon columns spend no tokens and must be ABSENT, never a zero row.
    assert set(token_summary["framework"]).isdisjoint(signed.LLR40_CANON_COLUMNS)


def test_token_summary_table_excludes_rows_with_no_tokens(
    llr40_canon: pd.DataFrame, llr40_observations: pd.DataFrame
) -> None:
    rows = signed.llr40_rows(llr40_canon, llr40_observations, ROSTER_TOKENS)
    frame = signed.token_summary_table(rows)
    assert set(frame.columns) == set(signed.TOKEN_SUMMARY_COLUMNS)
    assert set(frame["framework"]).isdisjoint(signed.LLR40_CANON_COLUMNS)


def test_the_tokens_table_is_the_token_slots_geomean(
    llr40_canon: pd.DataFrame, llr40_observations: pd.DataFrame
) -> None:
    """Paper rule: the token summary is the geomean over the served kernels -- the value and interval
    the figure's token slot draws."""
    rows = signed.llr40_rows(llr40_canon, llr40_observations, ROSTER_TOKENS)
    frame = signed.token_summary_table(rows).set_index("framework")
    tokens = signed.llr40_metrics(rows, ROSTER_TOKENS)[1]
    for row, one in zip(rows, tokens.series, strict=True):
        if not row.tokens:
            continue
        point, low, high = per_kernel.summary_geomean(one.cells)
        got = frame.loc[row.framework]
        assert (got.gm_tokens, got.gm_tokens_low, got.gm_tokens_high) == pytest.approx((point, low, high))
        assert got.gm_tokens == pytest.approx(summary.geomean(list(row.tokens.values())))


def test_a_tokens_table_too_thin_for_any_interval_fails_rule_5(
    llr40_canon: pd.DataFrame, llr40_observations: pd.DataFrame
) -> None:
    """Three kernels support no median interval, and a table whose every row is a bare point is
    exactly what Rule 5 refuses -- a thin roster is named, never drawn as if it were precise."""
    rows = signed.llr40_rows(llr40_canon, llr40_observations, ROSTER40)
    with pytest.raises(rules.RuleViolation, match="Rule 5"):
        signed.token_summary_table(rows)


def test_the_summary_table_leaves_out_a_compilers_placeholders(llr40_canon: pd.DataFrame) -> None:
    """dace_cpu_canonicalize solved k1 and k2 (10x each) and never timed k3 (1x, crossed): the
    summary row is the slot's number, 10x over n=2, not 4.6x over three."""
    (row,) = signed.llr40_rows(llr40_canon, None, ROSTER40, canon_columns=("dace_cpu_canonicalize",))
    summary_row = signed.summary_table([row]).iloc[0]
    assert summary_row["n"] == 2
    assert summary_row["geomean"] == pytest.approx(10.0)
    assert set(signed.table([row])["kernel"]) == set(ROSTER40)  # the placeholder is still a kernel row


# --------------------------------------------------------------------------------------------
# The user's four corrections to the rendered figure: log2 geometry read back as ratios, a
# visible summary whisker, an omitted tokens panel with nothing to draw, and a legend that
# clears the rotated kernel labels and says what it is showing.
# --------------------------------------------------------------------------------------------


def test_log2_change_matches_signed_change_sign_and_zero() -> None:
    from hpcagent_bench.stats.summary import log2_change

    assert log2_change(1.0) == pytest.approx(0.0)
    assert log2_change(2.0) == pytest.approx(1.0)
    assert log2_change(0.5) == pytest.approx(-1.0)
    # A 75x outlier sits at a SMALL log2 distance, not at 74 the way signed_change would place it.
    assert log2_change(75.0) < 7.0


def test_log2_change_is_nan_off_a_placeholder() -> None:
    from hpcagent_bench.stats.summary import log2_change

    assert math.isnan(log2_change(0.0))
    assert math.isnan(log2_change(-1.0))


def test_the_speedup_panel_is_log2_geometry_read_back_in_ratios(llr40_canon: pd.DataFrame) -> None:
    """Every doubling the same distance apart (a linear signed change put one 75x outlier 74 units
    from zero and swamped every other mark), and the ticks still read as speed-ups: the "2x" tick
    sits at the ratio 2 on a base-2 log axis."""
    fig = signed.llr40_figure(signed.llr40_rows(llr40_canon, None, ROSTER40), ROSTER40)
    try:
        ax = fig.axes[0]
        labels = [tick.get_text() for tick in ax.get_yticklabels()]
        positions = list(ax.get_yticks())
        scale, base = ax.get_yscale(), ax.yaxis.get_transform().base
    finally:
        plt.close(fig)
    assert (scale, base) == ("log", 2.0)
    assert "1x" in labels and "2x" in labels and "4x" in labels
    assert positions[labels.index("2x")] == pytest.approx(2.0)


def test_the_1x_line_wears_the_baselines_own_colour(llr40_canon: pd.DataFrame) -> None:
    """The 1x line IS the baseline (numba), so it is drawn in numba's colour, not a neutral rule."""
    fig = signed.llr40_figure(signed.llr40_rows(llr40_canon, None, ROSTER40), ROSTER40)
    try:
        colours = [line.get_color() for line in fig.axes[0].get_lines() if list(line.get_ydata()) == [1.0, 1.0]]
    finally:
        plt.close(fig)
    assert colours == [palette.framework_color(signed.LLR40_BASELINE)], colours


def test_figure_omits_the_tokens_panel_when_no_row_spends_tokens(llr40_canon: pd.DataFrame) -> None:
    rows = signed.llr40_rows(llr40_canon, None, ROSTER40)
    fig = signed.llr40_figure(rows, ROSTER40, "title")
    try:
        assert len(fig.axes) == 1
        assert fig.axes[0].get_ylabel() == "Speed-up over Numba"
    finally:
        plt.close(fig)


def test_figure_keeps_the_tokens_panel_when_a_row_spends_tokens(
    llr40_canon: pd.DataFrame, llr40_observations: pd.DataFrame
) -> None:
    rows = signed.llr40_rows(llr40_canon, llr40_observations, ROSTER40)
    fig = signed.llr40_figure(rows, ROSTER40, "title")
    try:
        assert len(fig.axes) == 2
        assert fig.axes[1].get_ylabel() == "Tokens spent"
    finally:
        plt.close(fig)


def test_legend_names_each_optimizer_and_the_cross_only_when_one_is_drawn(llr40_canon: pd.DataFrame) -> None:
    """The legend keys what a reader cannot read off the axes: which shape is which optimizer, and
    the cross when a kernel carries one. The interval method is the caption's."""
    rows = signed.llr40_rows(llr40_canon, None, ROSTER40)
    labels = [handle.get_label() for handle in signed.legend_handles(rows, signed.llr40_metrics(rows, ROSTER40))]
    assert labels == ["DaCe", "Canonical Parallel Form", style.NOT_DELIVERED_LABEL]
    complete = signed.llr40_rows(llr40_canon, None, ("k1",))
    handles = signed.legend_handles(complete, signed.llr40_metrics(complete, ("k1",)))
    assert [handle.get_label() for handle in handles] == ["DaCe", "Canonical Parallel Form"]


def test_standalone_optimizers_wear_their_own_shapes(llr40_canon: pd.DataFrame) -> None:
    rows = signed.llr40_rows(llr40_canon, None, ROSTER40)
    assert [row.marker for row in rows] == [palette.marker("dace"), palette.marker("cpf")]
    assert rows[0].marker != rows[1].marker


def test_figure_prints_at_text_width_and_names_its_summary_statistic(llr40_canon: pd.DataFrame) -> None:
    """Drawn at the size the page prints it: a figure* is text width, and a single speed-up panel
    is a short strip, not a page. The axis label names the baseline and the 1x tick stays a ratio;
    every kernel has a visible tick, and the one summary statistic is named by an x tick under its
    slots (user, 2026-09-22)."""
    rows = signed.llr40_rows(llr40_canon, None, ROSTER40)
    fig = signed.llr40_figure(rows, ROSTER40)
    try:
        width, height = (float(v) for v in fig.get_size_inches())
        labels = [label.get_text() for label in fig.axes[0].get_yticklabels()]
        kernel_ticks = [label.get_text() for label in fig.axes[0].get_xticklabels()]
        tick_length = fig.axes[0].xaxis.get_major_ticks()[0].tick1line.get_markersize()
        annotations = [text.get_text() for text in fig.axes[0].texts]
    finally:
        plt.close(fig)
    assert width == pytest.approx(style.DOUBLE_COLUMN_WIDTH)
    assert height < 2.6
    assert "1x" in labels and fig.axes[0].get_ylabel() == "Speed-up over Numba"
    assert len(kernel_ticks) == len(ROSTER40) + 1 and kernel_ticks[-1] == "Geomean"
    assert "Geomean" not in annotations
    assert tick_length > 0.0
    assert fig.texts == []  # no title unless one is asked for


def test_summary_column_prints_each_geomean_value(llr40_canon: pd.DataFrame) -> None:
    rows = signed.llr40_rows(llr40_canon, None, ROSTER40)
    fig = signed.llr40_figure(rows, ROSTER40)
    try:
        texts = [text.get_text() for text in fig.axes[0].texts]
    finally:
        plt.close(fig)
    # The spelling is the shared speller's, not restated here: the property is that every row's
    # geomean is printed.
    (speed,) = signed.llr40_metrics(rows, ROSTER40)
    expected = [style.ratio_label(per_kernel.summary_geomean(one.cells)[0]) for one in speed.series]
    assert all(value in texts for value in expected), (expected, texts)


def test_render_at_150dpi_matches_the_requested_dpi(
    llr40_canon: pd.DataFrame, llr40_observations: pd.DataFrame, tmp_path: pathlib.Path
) -> None:
    out = tmp_path / "llr40_dpi"
    signed.llr40_two_row_figure(llr40_canon, llr40_observations, ROSTER_TOKENS, out, dpi=150.0)
    with Image.open(out.with_suffix(".png")) as image:
        assert image.info["dpi"][0] == pytest.approx(150.0, abs=1.0)


@pytest.fixture(name="pending_canon")
def pending_canon_fixture() -> pd.DataFrame:
    """numba times k1-k3; the CPF column validates k1, runs k2 and fails it, and has no row for k3."""
    frame = canon_table([("numba", "k1", 100.0), ("numba", "k2", 100.0), ("numba", "k3", 100.0),
                         ("dace_cpu_canonicalize", "k1", 10.0), ("dace_cpu_canonicalize", "k2", 50.0)])  # fmt: skip
    frame.loc[(frame["column"] == "dace_cpu_canonicalize") & (frame["kernel"] == "k2"), "validated"] = "False"
    return frame


def test_by_default_a_kernel_never_attempted_is_a_failure_at_one(pending_canon: pd.DataFrame) -> None:
    row = signed.canon_kernel_row(pending_canon, "dace_cpu_canonicalize", ROSTER40)
    assert row.ratios == {"k1": pytest.approx(10.0), "k2": 1.0, "k3": 1.0}
    assert row.pending == frozenset() and row.excluded == "none"


def test_mark_pending_splits_a_kernel_never_attempted_from_one_that_failed(pending_canon: pd.DataFrame) -> None:
    """k2 ran and failed: it keeps its cross at 1x, a kernel row, and -- solved kernels only -- no
    place in the summary. k3 has no row yet: it leaves the ratios and the geomean and is named
    pending."""
    row = signed.canon_kernel_row(pending_canon, "dace_cpu_canonicalize", ROSTER40, mark_pending=True)
    assert row.ratios == {"k1": pytest.approx(10.0), "k2": 1.0}
    assert row.delivered == {"k1": True, "k2": False}
    assert row.pending == frozenset({"k3"}) and row.excluded == "1 pending"
    summary_row = signed.summary_table([row]).iloc[0]
    assert summary_row["n"] == 1 and summary_row["excluded"] == "1 pending"
    assert summary_row["geomean"] == pytest.approx(10.0)
    assert set(signed.table([row])["kernel"]) == {"k1", "k2"}


def test_a_failed_kernel_draws_at_one_and_enters_no_summary(pending_canon: pd.DataFrame) -> None:
    """The 1x placeholder is a drawing convention, not a measurement: k2 failed, so it draws in its
    kernel slot and stays out of the geomean, which would otherwise credit the failure with parity
    and pull a 10x row down to 3.2x. The success rate is the separate number (``excluded``)."""
    row = signed.canon_kernel_row(pending_canon, "dace_cpu_canonicalize", ROSTER40)
    assert row.ratios == {"k1": pytest.approx(10.0), "k2": 1.0, "k3": 1.0}
    summary_row = signed.summary_table([row]).iloc[0]
    assert summary_row["n"] == 1
    assert summary_row["geomean"] == pytest.approx(10.0)
    assert summary_row["wins"] == 1 and summary_row["losses"] == 0


def test_a_kernel_the_baseline_never_ran_is_pending_too(pending_canon: pd.DataFrame) -> None:
    frame = pending_canon[~((pending_canon["column"] == "numba") & (pending_canon["kernel"] == "k1"))]
    row = signed.canon_kernel_row(frame, "dace_cpu_canonicalize", ROSTER40, mark_pending=True)
    assert row.pending == frozenset({"k1", "k3"})


def test_mark_pending_keeps_an_arm_not_yet_served_every_kernel(
    llr40_canon: pd.DataFrame, llr40_observations: pd.DataFrame
) -> None:
    partial = llr40_observations[
        ~((llr40_observations["arm"] == "cpf-llr-focus40-oss120b-c-cpf") & (llr40_observations["benchmark"] == "k3"))
    ]
    arm = "cpf-llr-focus40-oss120b-c-cpf"
    assert arm not in {row.framework for row in signed.llr40_rows(llr40_canon, partial, ROSTER40)}
    rows = {row.framework: row for row in signed.llr40_rows(llr40_canon, partial, ROSTER40, mark_pending=True)}
    assert rows[arm].pending == frozenset({"k3"}) and set(rows[arm].ratios) == {"k1", "k2"}
    assert rows["cpf-llr-focus40-qwen38-c-cpf"].pending == frozenset()


def test_the_legend_keys_pending_apart_from_the_cross(pending_canon: pd.DataFrame) -> None:
    """A pending-only kernel adds the pending entry and not the cross; a failure adds the cross."""
    only_pending = signed.canon_kernel_row(pending_canon, "dace_cpu_canonicalize", ("k1", "k3"), mark_pending=True)
    metrics = signed.llr40_metrics([only_pending], ("k1", "k3"))
    labels = [handle.get_label() for handle in signed.legend_handles([only_pending], metrics)]
    assert labels == ["Canonical Parallel Form", style.PENDING_LABEL]
    both = signed.canon_kernel_row(pending_canon, "dace_cpu_canonicalize", ROSTER40, mark_pending=True)
    metrics = signed.llr40_metrics([both], ROSTER40)
    labels = [handle.get_label() for handle in signed.legend_handles([both], metrics)]
    assert labels == ["Canonical Parallel Form", style.NOT_DELIVERED_LABEL, style.PENDING_LABEL]


def test_the_figure_draws_one_pending_mark_per_pending_kernel(pending_canon: pd.DataFrame) -> None:
    row = signed.canon_kernel_row(pending_canon, "dace_cpu_canonicalize", ROSTER40, mark_pending=True)
    fig = signed.llr40_figure([row], ROSTER40)
    try:
        assert sum(c.get_gid() == style.PENDING_GID for c in fig.axes[0].collections) == 1
    finally:
        plt.close(fig)


def test_a_kernel_numba_did_not_verify_is_timed_against_the_fallback() -> None:
    """2026-09-21: where Numba fails, C autopar is the baseline; the row says how many kernels took it."""
    frame = canon_table([("numba", "k1", 100.0), ("cc_autopar", "k1", 80.0), ("cc_autopar", "k2", 40.0),
                         ("dace_cpu_canonicalize", "k1", 10.0), ("dace_cpu_canonicalize", "k2", 10.0)])  # fmt: skip
    row = signed.canon_kernel_row(frame, "dace_cpu_canonicalize", ("k1", "k2"), baseline_fallback="cc_autopar")
    assert row.ratios == {"k1": pytest.approx(10.0), "k2": pytest.approx(4.0)}
    assert row.numerator_ms["k2"] == pytest.approx(40.0) and row.excluded == "1 over cc_autopar"
    unfilled = signed.canon_kernel_row(frame, "dace_cpu_canonicalize", ("k1", "k2"))
    assert unfilled.ratios["k2"] == 1.0 and unfilled.delivered["k2"] is False


def test_the_fallback_never_replaces_a_numba_time() -> None:
    times = {"numba": {"k1": 100.0}, "cc_autopar": {"k1": 5.0, "k2": 7.0}}
    merged, filled = canon.with_fallback(times, "numba", "cc_autopar")
    assert merged["numba"] == {"k1": 100.0, "k2": 7.0} and filled == frozenset({"k2"})
    assert canon.with_fallback(times, "numba", "") == (times, frozenset())


def test_a_shorter_panel_makes_a_shorter_figure_at_the_same_width(llr40_canon: pd.DataFrame) -> None:
    rows = signed.llr40_rows(llr40_canon, None, ROSTER40)
    tall, short = signed.llr40_figure(rows, ROSTER40), signed.llr40_figure(rows, ROSTER40, panel_height_in=1.0)
    try:
        assert short.get_size_inches()[0] == pytest.approx(tall.get_size_inches()[0])
        assert short.get_size_inches()[1] == pytest.approx(tall.get_size_inches()[1] - 0.5, abs=0.01)
    finally:
        plt.close(tall)
        plt.close(short)
