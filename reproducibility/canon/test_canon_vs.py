"""Consumers of plot_canon_vs: the pairing rule, the sign test, and the degenerate comparisons.

The property this figure lives or dies on is PAIRING -- a kernel only one of the two tools compiled
must leave the comparison entirely, in both directions. A run where that silently stops holding
still draws a plausible-looking figure, so it is asserted here rather than eyeballed.

Usage:  python3 -m pytest test_canon_vs.py
"""

from __future__ import annotations

import pathlib

import pytest

import plot_canon_vs
import test_tsvc_speedup as fixtures


@pytest.fixture(name="sweep")
def sweep_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    """Canon against two comparison arms, carrying every shape the real sweep can produce.

    Canon times s1..s4 plus s_only. dace main times s1..s4 (s3 CRASHED, so it drops out of the pair)
    and s_main_only, which canon never timed. llvm+polly times s1 and s2 only.
    """
    canon = [
        fixtures.row(plot_canon_vs.REFERENCE, "tsvc_2_s1", "50.0"),
        fixtures.row(plot_canon_vs.REFERENCE, "tsvc_2_s2", "100.0"),
        fixtures.row(plot_canon_vs.REFERENCE, "tsvc_2_s3", "100.0"),
        fixtures.row(plot_canon_vs.REFERENCE, "tsvc_2_s4", "200.0"),
        fixtures.row(plot_canon_vs.REFERENCE, "tsvc_2_s_only", "10.0"),
    ]
    fixtures.write(tmp_path / f"{plot_canon_vs.REFERENCE}.rank0.csv", canon[:3])
    fixtures.write(tmp_path / f"{plot_canon_vs.REFERENCE}.rank1.csv", canon[3:])
    fixtures.write(
        tmp_path / "dace_cpu.csv",
        [
            fixtures.row("dace_cpu", "tsvc_2_s1", "100.0"),
            fixtures.row("dace_cpu", "tsvc_2_s2", "100.0"),
            fixtures.row("dace_cpu", "tsvc_2_s3", "", status="failed"),
            fixtures.row("dace_cpu", "tsvc_2_s4", "100.0"),
            fixtures.row("dace_cpu", "tsvc_2_s_main_only", "10.0"),
        ],
    )
    fixtures.write(
        tmp_path / "cc_llvm_autopar.csv",
        [
            fixtures.row("cc_llvm_autopar", "tsvc_2_s1", "200.0"),
            fixtures.row("cc_llvm_autopar", "tsvc_2_s2", "50.0"),
        ],
    )
    return tmp_path


def test_pairs_only_kernels_both_tools_timed(sweep: pathlib.Path) -> None:
    canon = plot_canon_vs.plot_tsvc_speedup.read_arm(sweep, plot_canon_vs.REFERENCE)
    other = plot_canon_vs.plot_tsvc_speedup.read_arm(sweep, "dace_cpu")
    ratios = plot_canon_vs.paired(canon.times, other.times)
    # s3 crashed in the comparison arm and s_only/s_main_only are one-sided: all three drop out.
    assert sorted(ratios) == ["tsvc_2_s1", "tsvc_2_s2", "tsvc_2_s4"]


def test_ratio_is_other_over_canon(sweep: pathlib.Path) -> None:
    canon = plot_canon_vs.plot_tsvc_speedup.read_arm(sweep, plot_canon_vs.REFERENCE)
    other = plot_canon_vs.plot_tsvc_speedup.read_arm(sweep, "dace_cpu")
    ratios = plot_canon_vs.paired(canon.times, other.times)
    assert ratios["tsvc_2_s1"] == pytest.approx(2.0)  # canon 50ms vs main 100ms -- canon wins
    assert ratios["tsvc_2_s4"] == pytest.approx(0.5)  # canon 200ms vs main 100ms -- canon loses


def test_sign_test_counts_both_directions_and_ignores_ties(sweep: pathlib.Path) -> None:
    canon = plot_canon_vs.plot_tsvc_speedup.read_arm(sweep, plot_canon_vs.REFERENCE)
    other = plot_canon_vs.plot_tsvc_speedup.read_arm(sweep, "dace_cpu")
    wins, losses = plot_canon_vs.sign_test(plot_canon_vs.paired(canon.times, other.times))
    assert (wins, losses) == (1, 1)  # s2 is an exact tie and counts for neither side


def test_narrow_comparison_arm_keeps_its_own_n(sweep: pathlib.Path) -> None:
    canon = plot_canon_vs.plot_tsvc_speedup.read_arm(sweep, plot_canon_vs.REFERENCE)
    polly = plot_canon_vs.plot_tsvc_speedup.read_arm(sweep, "cc_llvm_autopar")
    # polly compiled two kernels, so its row has n=2 -- NOT canon's five.
    assert len(plot_canon_vs.paired(canon.times, polly.times)) == 2


def test_render_survives_a_missing_comparison_arm(sweep: pathlib.Path, tmp_path: pathlib.Path) -> None:
    (sweep / "cc_llvm_autopar.csv").unlink()
    plot_canon_vs.plot_tsvc_speedup.style()
    out = plot_canon_vs.render(sweep, tmp_path / "canon")
    assert out.with_suffix(".pdf").is_file() and out.with_suffix(".png").is_file()


def test_missing_reference_is_fatal(tmp_path: pathlib.Path) -> None:
    fixtures.write(tmp_path / "dace_cpu.csv", [fixtures.row("dace_cpu", "tsvc_2_s1", "1.0")])
    with pytest.raises(SystemExit):
        plot_canon_vs.render(tmp_path, tmp_path / "canon")
