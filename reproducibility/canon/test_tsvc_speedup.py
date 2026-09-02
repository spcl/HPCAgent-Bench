"""Consumers of plot_tsvc_speedup: the signed axis, the exclusion rules, and the degenerate arms.

The sweep this figure reads takes eight hours, so the shapes that break a plotting script -- an arm
whose CSV never appeared, an arm with one usable kernel and therefore no interval, a miscompile that
must not be drawn as "no change" -- are exercised here against a synthetic sweep directory instead of
against whichever of them the next real run happens to contain.

Usage:  python3 -m pytest test_tsvc_speedup.py
"""

from __future__ import annotations

import csv
import pathlib

import pytest

import plot_tsvc_speedup

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

    Both file shapes appear on purpose: the baseline is sharded the way a four-rank job writes it and
    the arms are unsharded, so a reader of this fixture can see that the two are read the same way.
    """
    baseline = [row(plot_tsvc_speedup.BASELINE, f"tsvc_2_s{i}", "100.0") for i in range(1, 7)]
    baseline.append(row(plot_tsvc_speedup.BASELINE, "tsvc_2_slow", "100.0"))
    baseline.append(row(plot_tsvc_speedup.BASELINE, "jacobi_2d", "100.0"))
    write(tmp_path / f"{plot_tsvc_speedup.BASELINE}.rank0.csv", baseline[:4])
    write(tmp_path / f"{plot_tsvc_speedup.BASELINE}.rank1.csv", baseline[4:])

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


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(2.0, 1.0), (3.0, 2.0), (1.0, 0.0), (0.5, -1.0), (0.25, -3.0), (4.0, 3.0), (1.0 / 3.0, -2.0)],
)
def test_signed_axis_mapping(ratio: float, expected: float) -> None:
    assert plot_tsvc_speedup.signed(ratio) == pytest.approx(expected)


@pytest.mark.parametrize("ratio", [1.25, 2.0, 3.0, 7.5, 100.0])
def test_signed_axis_is_odd_about_no_change(ratio: float) -> None:
    """A win and the identical loss must be the same distance from zero, or the axis flatters one."""
    assert plot_tsvc_speedup.signed(1.0 / ratio) == pytest.approx(-plot_tsvc_speedup.signed(ratio))


def test_rejects_are_counted_not_plotted(sweep: pathlib.Path) -> None:
    arm = plot_tsvc_speedup.read_arm(sweep, "dace_cpu_canonicalize")
    assert set(arm.times) == {"tsvc_2_s1", "tsvc_2_s2", "tsvc_2_s3", "tsvc_2_s4", "tsvc_2_slow"}
    assert arm.rejected["status=crash"] == 1
    assert arm.rejected["not validated"] == 1
    # The non-TSVC kernel is out of scope, so it is neither timed nor counted as an exclusion.
    assert "jacobi_2d" not in arm.times
    assert sum(arm.rejected.values()) == 2


def test_slower_than_baseline_lands_below_zero(sweep: pathlib.Path) -> None:
    arm = plot_tsvc_speedup.read_arm(sweep, "dace_cpu_canonicalize")
    reference = plot_tsvc_speedup.read_arm(sweep, plot_tsvc_speedup.BASELINE)
    values = plot_tsvc_speedup.speedups(arm, reference.times)
    assert plot_tsvc_speedup.signed(values["tsvc_2_slow"]) == pytest.approx(-3.0)
    assert plot_tsvc_speedup.signed(values["tsvc_2_s1"]) == pytest.approx(1.0)


def test_missing_arm_reads_as_empty(sweep: pathlib.Path) -> None:
    arm = plot_tsvc_speedup.read_arm(sweep, "cc_llvm_autopar")
    assert not arm.times and not arm.rejected


def test_single_point_interval_collapses() -> None:
    """n=1 has no spread to estimate, so the interval is the point -- and must not raise."""
    centre, low, high = plot_tsvc_speedup.geomean_interval([5.0])
    assert centre == pytest.approx(5.0) and low == pytest.approx(5.0) and high == pytest.approx(5.0)


def test_geomean_interval_brackets_the_centre() -> None:
    centre, low, high = plot_tsvc_speedup.geomean_interval([2.0, 4.0, 8.0])
    assert centre == pytest.approx(4.0)
    assert low < centre < high


def test_render_survives_every_degenerate_arm(sweep: pathlib.Path, tmp_path: pathlib.Path) -> None:
    plot_tsvc_speedup.style()
    out = plot_tsvc_speedup.render(sweep, tmp_path / "figure")
    assert out.with_suffix(".pdf").is_file() and out.with_suffix(".png").is_file()


def test_missing_baseline_is_fatal(tmp_path: pathlib.Path) -> None:
    write(tmp_path / "dace_cpu.csv", [row("dace_cpu", "tsvc_2_s1", "20.0")])
    with pytest.raises(SystemExit, match=plot_tsvc_speedup.BASELINE):
        plot_tsvc_speedup.render(tmp_path, tmp_path / "figure")
