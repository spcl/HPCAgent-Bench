# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The stdlib signed-rank test and the scipy one must be the SAME test, not two similar ones.

This repo computes the Wilcoxon signed-rank p twice on purpose: the figures take it from scipy, and
``experiments/ablation_stats.py`` is stdlib-only because it runs on a login node from a shell that
never activated the benchmark environment. Two implementations are fine. Two cutoffs are not.

THE FAILURE THIS PREVENTS. One module switched to the normal approximation above n = 25 while the
other inherited scipy's ``auto`` heuristic and stayed exact further, so the same test on the same 40
kernels was published as p = 0.18329 in one table and p = 0.18762 in another -- and the
approximation is the anti-conservative side, so a p reported under 0.05 from it might not be. These
pin that both paths read one threshold, that neither claims exactness where the exact null does not
apply, and that they return the same number wherever both are exact.
"""

import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import wilcoxon

from hpcagent_bench.stats import signed_rank, summary

#: Sizes spanning the range these tables reach and a little past it. 40 is the llr focus roster,
#: which is where the two implementations actually disagreed.
EXACT_RANGE = (6, 8, 12, 19, 25, 26, 32, 40, 55, 80, 120)

#: Effect sizes chosen to land the p across the whole interval, including either side of 0.05 --
#: agreement only in the flat middle would miss exactly the disagreements that change a claim.
SHIFTS = (0.0, 0.1, 0.25, 0.45, 0.8, 1.4)


def clean(n: int, shift: float, seed: int) -> list[float]:
    """A tie-free, zero-free sample: the case where BOTH implementations must be exact."""
    values = np.random.default_rng(seed).normal(shift, 1.0, n)
    # Nudge any accidental duplicate |d| apart; a tie makes the exact null inapplicable, which is a
    # different property (below) and would hide a real disagreement here.
    while len(set(np.abs(values).tolist())) != n:
        values = values + np.random.default_rng(seed + 1000).normal(0.0, 1e-9, n)
    return values.tolist()


@pytest.mark.parametrize("n", EXACT_RANGE)
@pytest.mark.parametrize("shift", SHIFTS)
def test_the_stdlib_and_scipy_paths_return_one_p_wherever_both_are_exact(n: int, shift: float) -> None:
    """The property the two implementations exist under: same test, same number, every size."""
    values = clean(n, shift, seed=n * 31 + int(shift * 100))
    used, stdlib_p, method = signed_rank.signed_rank_p(values)
    assert used == n and method == "signed-rank-exact"
    scipy_p = float(wilcoxon(values, method="exact", zero_method="wilcox").pvalue)
    assert stdlib_p == pytest.approx(scipy_p, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("n", EXACT_RANGE)
def test_the_two_paths_agree_on_a_zero_bearing_sample(n: int) -> None:
    """A zero supports neither direction, so both drop it -- and must drop the SAME ones, or the
    two report different n for one comparison."""
    values = clean(n, 0.3, seed=n * 7)[: n - 2] + [0.0, 0.0]
    used, stdlib_p, method = signed_rank.signed_rank_p(values)
    assert used == n - 2
    nonzero = [v for v in values if v != 0.0]
    expected = float(
        wilcoxon(nonzero, method="exact" if method.endswith("exact") else "approx", correction=True).pvalue
    )
    assert stdlib_p == pytest.approx(expected, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("n", (8, 15, 26, 40, 80))
def test_a_tied_sample_takes_the_approximation_in_both_paths(n: int) -> None:
    """The exact null counts subsets of the DISTINCT ranks 1..n. With a tie the ranks are midranks
    and that lattice no longer holds, so an "exact" p there is wrong rather than imprecise. Neither
    path may claim it, and both must fall to the SAME tie- and continuity-corrected approximation.

    ``correction=True`` is asked for EXPLICITLY on the scipy side because scipy defaults it off
    while the stdlib path always applies the half-step; the default would make this a comparison of
    two different tests that happen to be close."""
    values = np.round(np.random.default_rng(n).normal(0.3, 1.0, n), 1).tolist()
    absolute = [abs(v) for v in values if v != 0.0]
    assert len(set(absolute)) < len(absolute), "fixture is not tied; the property is untested"
    assert not signed_rank.use_exact(absolute)
    used, stdlib_p, method = signed_rank.signed_rank_p(values)
    assert method == "signed-rank-approx"
    nonzero = [v for v in values if v != 0.0]
    expected = float(wilcoxon(nonzero, method="approx", zero_method="wilcox", correction=True).pvalue)
    assert used == len(nonzero)
    assert stdlib_p == pytest.approx(expected, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("n", EXACT_RANGE)
@pytest.mark.parametrize("shift", (0.0, 0.3, 0.9))
def test_paired_change_reports_the_same_p_as_the_stdlib_rule(n: int, shift: float) -> None:
    """The figure path is the scipy one; this is the end-to-end statement that a number on a figure
    and the same number in the ablation table cannot differ."""
    values = clean(n, shift, seed=n * 13 + int(shift * 10))
    change = summary.paired_change(values)
    _used, stdlib_p, method = signed_rank.signed_rank_p(values)
    assert change.method == method
    assert change.pvalue == pytest.approx(stdlib_p, rel=1e-12, abs=1e-15)


def test_both_paths_read_one_threshold() -> None:
    """The cutoff is a shared CONSTANT, never a number each module picked. If this stops holding,
    the two agree today and drift the next time either is touched."""
    ablation = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "ablation_stats.py"
    spec = importlib.util.spec_from_file_location("ablation_stats_threshold", ablation)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.EXACT_MAX_N is signed_rank.EXACT_MAX_N
    assert module.signed_rank.__file__ == signed_rank.__file__


def test_the_threshold_sits_where_the_exact_null_is_still_affordable() -> None:
    """The cutoff is measured, not inherited from scipy. Guards the two ways it goes wrong: raised
    past what the login-node DP can pay, or lowered back under the sizes these tables reach."""
    assert 40 <= signed_rank.EXACT_MAX_N, "the llr focus roster is 40 kernels and must stay exact"
    assert signed_rank.EXACT_MAX_N <= 250, "past this the stdlib DP costs seconds per distinct n"
    assert signed_rank.use_exact([float(i) for i in range(signed_rank.EXACT_MAX_N)])
    assert not signed_rank.use_exact([float(i) for i in range(signed_rank.EXACT_MAX_N + 1)])


def test_the_exact_null_counts_every_sign_assignment() -> None:
    """The DP is a subset-sum count, so its totals must come to 2**n and be symmetric about the
    mean rank sum -- the two ways a wrong recurrence shows up without changing any single p much."""
    for n in (1, 2, 5, 9, 14):
        counts = signed_rank.null_counts(n)
        assert sum(counts) == 2**n
        assert list(counts) == list(reversed(counts))
        assert len(counts) == n * (n + 1) // 2 + 1


def test_the_shared_rule_imports_without_the_benchmark_environment() -> None:
    """ablation_stats runs from a shell that never activated the venv, so the file it loads by path
    must pull in nothing third-party. An `import numpy` added here would break that script on the
    machine it is written for, and only there."""
    source = pathlib.Path(signed_rank.__file__)
    probe = (
        "import importlib.util,sys;"
        "before=set(sys.modules);"
        f"spec=importlib.util.spec_from_file_location('sr', {str(source)!r});"
        "m=importlib.util.module_from_spec(spec);sys.modules['sr']=m;spec.loader.exec_module(m);"
        "print(sorted(n for n in set(sys.modules)-before "
        "if n.split('.')[0] in ('numpy','scipy','pandas','matplotlib','yaml')))"
    )
    done = subprocess.run([sys.executable, "-I", "-c", probe], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]", f"the shared rule pulled in {done.stdout.strip()}"
