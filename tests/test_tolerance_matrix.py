# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A validation band must be satisfiable by two correct implementations.

``numpy.isclose`` passes when ``|a - b| <= atol + rtol * |b|``. The ``rtol`` term scales with
the value, so it vanishes exactly where a reference is 0.0 -- and a reference of 0.0 is not an
edge case in a low-precision format, it is what a small result quantizes to. ``atol`` is the
only term that reaches those elements, and set below one ULP it asks two implementations to
agree more finely than the format can represent a difference at all.

That is not hypothetical. The fp8 rows carried ``atol=1e-2`` against an eps of ``0.125`` and
``atol=1e-1`` against ``0.25``, so ``arc_distance`` at fp8 failed CI on ~9% of its elements
with numpy and jax both correct -- they agreed to 0 ULP at the median and differed by at most
4 ULP, while the band admitted 1.4. fp16 and bf16 were never affected because their hand-tuned
atols already sat at 1.02x and 1.28x their eps. Only the fp8 rows were round decimals.

So the rule was already there, followed by every format that worked and broken by the two that
did not. These tests pin it.
"""
import numpy as np
import pytest

from hpcagent_bench.precision import (DTYPES, Precision, TOLERANCE_MATRIX, atol_below_one_ulp, derived_band,
                                      machine_eps, numpy_dtype)


def test_every_band_is_satisfiable():
    """No format's ``atol`` is finer than one ULP of that format.

    The check is generated from :class:`Precision` itself, so a format added tomorrow is
    covered without anyone extending a list here.
    """
    unsatisfiable = atol_below_one_ulp()
    assert not unsatisfiable, (
        "these bands demand agreement finer than the format can represent, so no pair of "
        "correct implementations can pass them: " +
        ", ".join(f"{p.value}: atol={atol:g} < eps={eps:g}"
                  for p, (atol, eps) in sorted(unsatisfiable.items(), key=lambda kv: kv[0].value)))


@pytest.mark.parametrize("precision", list(Precision), ids=lambda p: p.value)
def test_a_one_ulp_disagreement_passes_the_band(precision):
    """The end the invariant exists for: perturb a value by one ULP of its own format and the
    band still accepts it. Asserted on the WORST case for a relative tolerance -- a reference of
    exactly zero, where only ``atol`` can reach."""
    rtol, atol = TOLERANCE_MATRIX[precision].as_tuple()
    dtype = numpy_dtype(precision)
    one_ulp = float(np.asarray([machine_eps(precision)], dtype=dtype)[0])
    assert np.isclose(
        one_ulp, 0.0, rtol=rtol,
        atol=atol), (f"{precision.value}: a reference of 0.0 against a candidate one ULP away ({one_ulp:g}) "
                     f"fails its own band (rtol={rtol:g}, atol={atol:g}) -- unsatisfiable")


def test_derived_band_never_falls_below_one_ulp():
    """The invariant holds for an UNLISTED format too. ``derived_band`` is what a new precision
    gets before anyone tunes it, and its ``rtol * 1e-2`` rule alone goes below one ULP as soon as
    the mantissa is short -- so the floor lives in the derivation, not only in the pinned table.
    A guard that fires and is then silenced by hand-editing the override table would restore the
    exact drift this file removes."""
    for precision in Precision:
        band = derived_band(precision)
        eps = machine_eps(precision)
        assert band.atol >= eps, (f"{precision.value}: derived atol {band.atol:g} is below one ULP ({eps:g})")


def test_the_fp8_formats_are_the_coarse_case_this_guards():
    """Premise check. The invariant is only interesting while some format is coarse enough for
    one ULP to be a large number; if fp8 were ever dropped, this file would still pass while
    guarding nothing, so the premise is asserted rather than assumed."""
    for precision in (Precision.FP8_E4M3, Precision.FP8_E5M2):
        assert precision in DTYPES, f"{precision.value} left the corpus; re-check what this file still covers"
        assert machine_eps(precision) >= 0.1, (f"{precision.value} eps is {machine_eps(precision):g} -- no longer the "
                                               "coarse case, so the guard's premise changed")
