# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
""":mod:`hpcagent_bench.support.exact_math` on the host: its ``frexp`` is numpy's bit for bit, a scalar stays a scalar,
and the functions stay within a few ulp of numpy's. Device agreement is :mod:`test_device_exactness`."""

import numpy as np
import pytest

from hpcagent_bench.support import exact_math

#: Ulps the series-based functions may differ from numpy's libm-backed ones.
ULP_BAND = 4


def spread() -> np.ndarray:
    """Normal and subnormal magnitudes of both signs, with zero, the infinities and nan."""
    rng = np.random.default_rng(0)
    normal = rng.standard_normal(100_000) * 10.0 ** rng.integers(-300, 300, 100_000)
    special = [0.0, -0.0, np.inf, -np.inf, np.nan, 5e-324, -5e-324, 1e-310, 2.2250738585072014e-308, 1.0, 0.5]
    return np.concatenate([normal, special])


def test_frexp_is_numpys_bit_for_bit() -> None:
    values = spread()
    mantissa, exponent = exact_math.frexp(values)
    expected_mantissa, expected_exponent = np.frexp(values)
    same = (mantissa.view(np.int64) == expected_mantissa.view(np.int64)) | (np.isnan(mantissa) & np.isnan(values))
    assert same.all()
    assert np.array_equal(exponent, expected_exponent)


@pytest.mark.parametrize("name", ["frexp", "log", "exp", "sin", "cos", "tanh", "expm1"])
def test_a_scalar_stays_a_scalar(name: str) -> None:
    """A 0-d input gives 0-d outputs: ``round()`` and scalar assignment downstream need them."""
    result = vars(exact_math)[name](np.float64(0.75))
    for part in result if isinstance(result, tuple) else (result,):
        assert np.ndim(part) == 0, f"{name} turned a scalar into shape {np.shape(part)}"
    assert np.ndim(exact_math.pow(np.float64(27.0), 1.0 / 3.0)) == 0


@pytest.mark.parametrize(("name", "reference"), [("log", np.log), ("exp", np.exp), ("sin", np.sin), ("cos", np.cos)])
def test_a_function_is_within_a_few_ulp_of_numpy(name: str, reference: np.ufunc) -> None:
    values = np.random.default_rng(1).uniform(0.001, 50.0, 100_000)
    got, expected = vars(exact_math)[name](values), reference(values)
    # Measured against the larger of the result and 1: near a zero of sin, cos or log the error is absolute.
    gap = np.abs(got - expected) / np.spacing(np.maximum(np.abs(expected), 1.0))
    assert float(np.max(gap)) <= ULP_BAND


def test_interp_is_numpys() -> None:
    rng = np.random.default_rng(3)
    points = np.sort(rng.uniform(-5.0, 5.0, 64))
    values = rng.standard_normal(64)
    at = np.concatenate([rng.uniform(-7.0, 7.0, 10_000), points, [points[0], points[-1]]])
    got, expected = exact_math.interp(at, points, values), np.interp(at, points, values)
    assert float(np.max(np.abs(got - expected) / np.spacing(np.maximum(np.abs(expected), 1.0)))) <= ULP_BAND


def test_divide_by_real_divides_each_part() -> None:
    rng = np.random.default_rng(2)
    z = rng.standard_normal(1000) + 1j * rng.standard_normal(1000)
    d = rng.uniform(0.5, 2.0, 1000)
    got = exact_math.divide_by_real(z, d)
    assert np.array_equal(got.real, z.real / d)
    assert np.array_equal(got.imag, z.imag / d)


if __name__ == "__main__":
    test_frexp_is_numpys_bit_for_bit()
    for each in ["frexp", "log", "exp", "sin", "cos", "tanh", "expm1"]:
        test_a_scalar_stays_a_scalar(each)
    test_a_function_is_within_a_few_ulp_of_numpy("log", np.log)
    test_a_function_is_within_a_few_ulp_of_numpy("exp", np.exp)
    test_a_function_is_within_a_few_ulp_of_numpy("sin", np.sin)
    test_a_function_is_within_a_few_ulp_of_numpy("cos", np.cos)
    test_interp_is_numpys()
    test_divide_by_real_divides_each_part()
