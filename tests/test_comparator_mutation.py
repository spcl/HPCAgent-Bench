# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mutation test of the output comparator as the grade calls it (:mod:`helpers.scripts.comparator_mutation`): which
corrupted answers the band accepts at fp64 and fp32, and what a shorter accumulation length changes.

The accepted sets below are a record of the comparator's behaviour, not a wish: every accepted wrong answer differs
from the reference only by less than the absolute floor ``max(atol, eps_acc * sqrt(l) * ||ref||_inf)``. A change to
the comparator, the bands or the floor that moves any verdict fails here and has to restate the set."""

import pytest

from helpers.scripts import comparator_mutation as mutation
from hpcagent_bench.precision import Precision, accumulation_eps, tolerance_band, ungradeable

WITHIN_BAND = frozenset({"noise_0.1x_rtol", "noise_0.5x_rtol", "noise_1x_rtol"})
NOISE_ABOVE_BAND = frozenset({"noise_2x_rtol", "noise_10x_rtol", "noise_100x_rtol", "noise_1000x_rtol"})
SLICES = frozenset(f"{change}_{which}_slice" for change in ("zeroed", "flipped") for which in ("lo", "mid", "hi"))

#: Wrong answers accepted at some gradeable length, beyond noise inside the relative band.
ACCEPTED_WRONG: dict[tuple[str, str], frozenset[str]] = {
    ("normal", "fp64"): frozenset(),
    ("normal", "fp32"): frozenset(),
    ("six_decades", "fp64"): frozenset(),
    # Row magnitude 1e-3 under a peak of 2e3: below eps32 * sqrt(l) * peak from l = 100 (zeroed) or 316 (flipped).
    ("six_decades", "fp32"): frozenset({"zeroed_lo_slice", "flipped_lo_slice"}),
    # The peak (8e-3) is under atol / rtol = 1e-2, so atol, not rtol, bounds the error.
    ("cancellation", "fp64"): frozenset({"noise_2x_rtol", "largest_1.5x_rtol"}),
    ("cancellation", "fp32"): frozenset({"noise_2x_rtol", "largest_1.5x_rtol"}),
    ("near_zero", "fp64"): frozenset(),
    # Rows of magnitude 1e-7 are under fp32's atol of 1e-5 at any length.
    ("near_zero", "fp32"): frozenset({"zeroed_lo_slice", "zeroed_mid_slice", "flipped_lo_slice", "flipped_mid_slice"}),
    # Every magnitude is 1e-7: fp64's atol (1e-11) still resolves 1e-4 of it, fp32's (1e-5) resolves nothing.
    ("tiny", "fp64"): NOISE_ABOVE_BAND | {"largest_1.5x_rtol", "largest_10x_rtol", "offset_10x_rtol_rms"},
    ("tiny", "fp32"): frozenset(mutation.CORRUPTIONS) - WITHIN_BAND,
}


@pytest.fixture(scope="module")
def rows() -> list[mutation.Row]:
    return list(mutation.sweep())


@pytest.fixture(scope="module")
def checks() -> list[mutation.Control]:
    # The 1e8-term sums take the script tens of seconds; the test stops at 1e5.
    return list(mutation.controls(long_terms=(100, 100_000)))


def test_the_sweep_covers_every_case(rows: list[mutation.Row]) -> None:
    cases = len(mutation.REFERENCES) * len(mutation.PRECISIONS) * len(mutation.CORRUPTIONS) * len(mutation.LENGTHS)
    assert len(rows) == cases == 600
    assert [mutation.probed_length(length) for length in mutation.LENGTHS] == [10, 316, 10_000]


def test_every_corruption_changes_the_answer() -> None:
    for kind in mutation.REFERENCES:
        for precision in mutation.PRECISIONS:
            ref = mutation.reference(kind, precision)
            for name in mutation.CORRUPTIONS:
                candidate = mutation.corrupted(name, ref, tolerance_band(precision).rtol)
                assert (candidate != ref).any(), (kind, precision, name)


def test_noise_inside_the_relative_band_is_accepted_wherever_the_length_is_gradeable(rows: list[mutation.Row]) -> None:
    for row in rows:
        if row.corruption in WITHIN_BAND:
            assert row.probed == mutation.ACCEPT, row
            assert row.symbolic in (mutation.ACCEPT, mutation.REFUSED), row


def test_the_accepted_wrong_answers_are_exactly_the_recorded_ones(rows: list[mutation.Row]) -> None:
    accepted: dict[tuple[str, str], set[str]] = {key: set() for key in ACCEPTED_WRONG}
    for row in rows:
        if row.corruption not in WITHIN_BAND and mutation.ACCEPT in (row.symbolic, row.probed):
            accepted[row.reference, row.precision].add(row.corruption)
    assert accepted == ACCEPTED_WRONG


def test_unit_scale_and_fp64_structural_corruptions_are_never_accepted(rows: list[mutation.Row]) -> None:
    structural = SLICES | {"index_shift_by_one", "constant_zero", "constant_mean", "constant_peak"}
    for row in rows:
        beyond = row.corruption not in WITHIN_BAND
        if (row.reference == "normal" and beyond) or (row.precision == "fp64" and row.corruption in structural):
            assert mutation.ACCEPT not in (row.symbolic, row.probed), row


def test_a_shorter_length_never_accepts_what_the_longer_one_rejected(rows: list[mutation.Row]) -> None:
    assert not [row for row in rows if row.symbolic == mutation.REJECT and row.probed == mutation.ACCEPT]


def test_the_verdict_changes_of_the_shorter_length(rows: list[mutation.Row]) -> None:
    counts = mutation.transitions(rows)
    assert counts == {
        (mutation.ACCEPT, mutation.ACCEPT): 150,
        (mutation.REJECT, mutation.REJECT): 349,
        # six_decades fp32 zeroed_lo_slice, l = 100 -> 10: the only verdict a shorter gradeable length changes.
        (mutation.ACCEPT, mutation.REJECT): 1,
        # fp32 at l = 1e8 is refused; at the probed 1e4 it is graded.
        (mutation.REFUSED, mutation.ACCEPT): 40,
        (mutation.REFUSED, mutation.REJECT): 60,
    }
    tightened = [row for row in rows if (row.symbolic, row.probed) == (mutation.ACCEPT, mutation.REJECT)]
    assert [(row.reference, row.precision, row.corruption, row.symbolic_length) for row in tightened] == [
        ("six_decades", "fp32", "zeroed_lo_slice", 100)
    ]


def test_what_a_newly_gradeable_length_accepts_was_already_accepted_at_a_gradeable_length(
    rows: list[mutation.Row],
) -> None:
    opened = [row for row in rows if (row.symbolic, row.probed) == (mutation.REFUSED, mutation.ACCEPT)]
    assert opened
    assert {(row.precision, row.symbolic_length) for row in opened} == {("fp32", 100_000_000)}
    for row in opened:
        # The same case at the next symbolic length down (1e5, longer than the probed 1e4) is accepted too.
        same = [
            other
            for other in rows
            if (other.reference, other.precision, other.corruption) == (row.reference, row.precision, row.corruption)
            and other.symbolic_length == 100_000
        ]
        assert [other.symbolic for other in same] == [mutation.ACCEPT], row


def test_refused_is_the_ungradeable_guard(rows: list[mutation.Row]) -> None:
    for row in rows:
        precision = Precision(row.precision)
        guard = ungradeable(accumulation_eps(precision), row.symbolic_length, tolerance_band(precision).rtol)
        assert (row.symbolic == mutation.REFUSED) == guard, row
        assert row.probed != mutation.REFUSED, row


def test_unchanged_references_pass_wherever_the_length_is_gradeable(checks: list[mutation.Control]) -> None:
    identity = [one for one in checks if one.name.startswith("identity:")]
    assert len(identity) == 60
    for one in identity:
        refused = (one.precision, one.length) == ("fp32", 100_000_000)
        assert one.verdict == (mutation.REFUSED if refused else mutation.ACCEPT), one


def test_a_reordered_sum_of_the_true_length_passes_at_that_length_and_at_the_longer_symbolic_one(
    checks: list[mutation.Control],
) -> None:
    sums = [one for one in checks if one.name == "reordered_sum"]
    assert {(one.precision, one.terms, one.length) for one in sums} == {
        (precision, terms, length)
        for precision in ("fp64", "fp32")
        for terms, length in ((10, 10), (10, 100), (100, 100), (316, 316), (316, 100_000), (100_000, 100_000))
    }
    for one in sums:
        assert one.verdict == mutation.ACCEPT, one
        assert one.need < 0.01, one


def test_a_reordered_cancelling_sum_needs_the_cancellation_factor_at_fp32(checks: list[mutation.Control]) -> None:
    verdicts = {(one.name.partition(":")[0], one.name.partition("=")[2], one.precision): one for one in checks}
    for kappa in ("4.02e+03", "40.3", "1"):
        for name in ("cancelling_sum", "cancelling_sum_kappa"):
            assert verdicts[name, kappa, "fp64"].verdict == mutation.ACCEPT
    # No cancellation: the true length is enough.
    assert verdicts["cancelling_sum", "1", "fp32"].verdict == mutation.ACCEPT
    # Terms 40x the output: a correct fp32 build fails at the true length and passes at l * kappa**2.
    assert verdicts["cancelling_sum", "40.3", "fp32"].verdict == mutation.REJECT
    assert verdicts["cancelling_sum", "40.3", "fp32"].need > 1.0
    assert verdicts["cancelling_sum_kappa", "40.3", "fp32"].verdict == mutation.ACCEPT
    # Terms 4000x the output: wrong at the true length, and l * kappa**2 is past what fp32 can grade at all.
    assert verdicts["cancelling_sum", "4.02e+03", "fp32"].verdict == mutation.REJECT
    assert verdicts["cancelling_sum_kappa", "4.02e+03", "fp32"].verdict == mutation.REFUSED


def test_the_report_lists_the_accepted_cases_and_the_changes(rows: list[mutation.Row]) -> None:
    report = mutation.render(rows, [])
    assert "| six_decades | fp32 | zeroed_lo_slice | accept / reject (changed) | accept / accept |" in report
    assert "| refused | accept | 40 |" in report
    assert "| normal | fp32 | constant_zero |" not in report
