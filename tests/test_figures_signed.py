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
import pathlib

import pandas as pd
import pytest

from hpcagent_bench.stats import rules
from hpcagent_bench.stats.figures import signed

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
