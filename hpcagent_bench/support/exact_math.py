# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Elementary functions and reductions that give the same bits on numpy and on cupy, for kernel initializers.

glibc and a GPU's math library round ``exp``, ``log``, ``sin`` and ``pow`` differently in the last place, so an
initializer that calls them builds different inputs on the host and on the device. These versions use only
operations IEEE 754 rounds exactly the same everywhere -- ``+ - * /``, ``sqrt``, ``rint``, ``ldexp``,
comparisons and ``where`` -- each as its own array operation, so no compiler fuses two of them into an FMA.
They are accurate to a few ulp of float64, which is all an input distribution needs. Arguments are converted to
float64 first; a float32 result is the float64 one rounded once. On the host a large argument takes a numba loop
doing the same operations in the same order (no fast-math, so no FMA): the same bits without one temporary array
per operation.

A floating reduction is summed in an order the array library chooses (numpy's pairwise blocks, a GPU's tree), so
:func:`sum`, :func:`cumsum`, :func:`matvec` and :func:`norm` fix the order themselves: a binary tree of whole-array
additions (halves added elementwise until one row is left), and a Hillis-Steele scan. An integer or boolean
reduction is exact in any order and runs as the library's own.
"""

import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numba as nb
import numpy as np
from numpy.typing import ArrayLike, NDArray

from hpcagent_bench.support.array_module import module_of

__all__ = [
    "EXPM1_SERIES_LIMIT",
    "EXP_LIMIT",
    "cos",
    "cumsum",
    "divide_by_real",
    "exp",
    "expm1",
    "frexp",
    "interp",
    "ipow",
    "linspace",
    "log",
    "logspace",
    "matmul",
    "matvec",
    "mean",
    "norm",
    "pow",
    "sin",
    "sum",
    "tanh",
]

type Array = NDArray[Any]

#: ``expm1`` takes its own Taylor series below this magnitude, where ``exp(x) - 1`` would cancel.
EXPM1_SERIES_LIMIT = 0.25
#: ``exp`` saturates beyond this: float64 overflows above about 709.78 and underflows below about -745.
EXP_LIMIT = 708.0
#: ``ln 2`` split so that ``k * LN2_HI`` is exact for every ``|k| < 2**11``.
LN2_HI = 6.93147180369123816490e-01
LN2_LO = 1.90821492927058770002e-10
INV_LN2 = 1.0 / math.log(2.0)
#: The float64 layout :func:`frexp` reads: 52 mantissa bits, an 11-bit exponent biased so that ``[1/2, 1)`` is 1022.
MANTISSA_BITS = 52
EXPONENT_MASK = 0x7FF
EXPONENT_HALF = 1022
SMALLEST_NORMAL = 2.0**-1022
SUBNORMAL_SHIFT = 54
SUBNORMAL_SCALE = 2.0**SUBNORMAL_SHIFT
#: ``pi / 2`` in three parts (Cody-Waite), exact products for ``|k| < 2**20``.
HALF_PI_1 = 1.5707963267341256e00
HALF_PI_2 = 6.077100506506192e-11
HALF_PI_3 = 2.0222662487959506e-21
INV_HALF_PI = 2.0 / math.pi
#: Taylor terms of ``exp`` on ``|r| <= ln2 / 2``, ``sin``/``cos`` on ``|r| <= pi / 4``, and ``atanh`` on
#: ``|s| <= 0.172`` (``log`` of a mantissa in ``[sqrt(1/2), sqrt(2))``): each truncation is below 2**-56.
EXP_TERMS = 14
TRIG_TERMS = 11
ATANH_TERMS = 12


#: Host arguments of at least this many elements take the numba loop (:func:`on_host_loop`).
HOST_LOOP_MIN = 1 << 16
EXP_COEFFICIENTS = tuple(1.0 / math.factorial(n) for n in range(EXP_TERMS))
SIN_COEFFICIENTS = tuple((-1.0) ** n / math.factorial(2 * n + 1) for n in range(TRIG_TERMS))
COS_COEFFICIENTS = tuple((-1.0) ** n / math.factorial(2 * n) for n in range(TRIG_TERMS))
ATANH_COEFFICIENTS = tuple(1.0 / (2 * n + 1) for n in range(ATANH_TERMS))


@nb.njit(cache=True)
def horner_scalar(r: float, coefficients: tuple[float, ...]) -> float:
    total = coefficients[-1]
    for i in range(len(coefficients) - 2, -1, -1):
        total = total * r
        total = total + coefficients[i]
    return total


@nb.njit(cache=True)
def trig_loop(x: Array, cosine: bool) -> Array:
    """:func:`sin` (``cosine`` false) or :func:`cos` of every element, operation for operation."""
    out = np.empty_like(x)
    for i in range(x.size):
        k = np.rint(x[i] * INV_HALF_PI)
        r = x[i] - k * HALF_PI_1
        r = r - k * HALF_PI_2
        r = r - k * HALF_PI_3
        quadrant = k - 4.0 * np.floor(k * 0.25)
        r2 = r * r
        s = r * horner_scalar(r2, SIN_COEFFICIENTS)
        c = horner_scalar(r2, COS_COEFFICIENTS)
        if cosine:
            out[i] = c if quadrant == 0.0 else (-s if quadrant == 1.0 else (-c if quadrant == 2.0 else s))
        else:
            out[i] = s if quadrant == 0.0 else (c if quadrant == 1.0 else (-s if quadrant == 2.0 else -c))
    return out


@nb.njit(cache=True)
def exp_loop(x: Array) -> Array:
    """:func:`exp` of every real element, operation for operation."""
    out = np.empty_like(x)
    for i in range(x.size):
        value = min(max(x[i], -EXP_LIMIT), EXP_LIMIT)
        k = np.rint(value * INV_LN2)
        r = value - k * LN2_HI
        r = r - k * LN2_LO
        out[i] = math.ldexp(horner_scalar(r, EXP_COEFFICIENTS), int(k))
    return out


@nb.njit(cache=True)
def log_loop(x: Array) -> Array:
    """:func:`log` of every element, operation for operation."""
    out = np.empty_like(x)
    for i in range(x.size):
        value = x[i]
        mantissa, exponent = math.frexp(value if value > 0.0 else 1.0)
        if mantissa < math.sqrt(0.5):
            mantissa = mantissa * 2.0
            exponent = exponent - 1
        e = np.float64(exponent)
        s = (mantissa - 1.0) / (mantissa + 1.0)
        result = 2.0 * s * horner_scalar(s * s, ATANH_COEFFICIENTS)
        result = result + e * LN2_LO
        result = result + e * LN2_HI
        out[i] = -np.inf if value == 0.0 else (np.nan if value < 0.0 else result)
    return out


@nb.njit(cache=True)
def tree_sum_loop(rows: Array) -> Array:
    """:func:`sum` over the first axis of a 2-D ``rows``, per column, in its halving order."""
    length, width = rows.shape
    out = np.empty(width, dtype=rows.dtype)
    buf = np.empty(length, dtype=rows.dtype)
    for j in range(width):
        for k in range(length):
            buf[k] = rows[k, j]
        n = length
        while n > 1:
            half = n // 2
            for k in range(half):
                buf[k] = buf[k] + buf[half + k]
            if n % 2:
                buf[half] = buf[2 * half]
            n = half + n % 2
        out[j] = buf[0]
    return out


def on_host_loop(value: Array, xp: ModuleType) -> bool:
    """Whether a float64 argument is a host array large enough for the numba loop to beat the array operations."""
    return xp is np and value.size >= HOST_LOOP_MIN


def host_loop(loop: Callable[..., Array], value: Array, *args: object) -> Array:
    """``loop`` over ``value``'s elements, in ``value``'s shape."""
    flat = np.ascontiguousarray(value).reshape(-1)
    return loop(flat, *args).reshape(value.shape)


def array_module(*values: object, xp: ModuleType | None = None) -> ModuleType:
    """``xp`` when given, else the array module of the first array among ``values`` (numpy for scalars), so a call on
    a device array stays on the device without being told."""
    if xp is not None:
        return xp
    for value in values:
        module = module_of(value)
        if module is not None:
            return module
    return np


def as_f64(x: ArrayLike, xp: ModuleType) -> Array:
    """``x`` as a float64 array of ``xp``; a complex ``x`` is refused rather than silently made real."""
    value = xp.asarray(x)
    if value.dtype.kind == "c":
        raise TypeError("exact_math: a complex argument (only exp takes one)")
    return value.astype(xp.float64, copy=False)


def horner(r: Array, coefficients: list[float], xp: ModuleType) -> Array:
    """``sum(c_i r**i)`` by Horner's rule, one exact multiply and one exact add per term."""
    total = xp.full(r.shape, coefficients[-1], dtype=xp.float64)
    for coefficient in reversed(coefficients[:-1]):
        total = total * r
        total = total + coefficient
    return total


def exp(x: ArrayLike, xp: ModuleType | None = None) -> Array:
    """``e**x``: ``x = k ln2 + r``, ``exp(r)`` by its Taylor series, then ``ldexp`` by ``k``. A complex ``x = a + ib``
    gives ``e**a (cos b + i sin b)`` as complex128."""
    xp = array_module(x, xp=xp)
    value = xp.asarray(x)
    if value.dtype.kind == "c":
        magnitude = exp(value.real, xp)
        return magnitude * cos(value.imag, xp) + 1j * (magnitude * sin(value.imag, xp))
    value = as_f64(value, xp)
    if on_host_loop(value, xp):
        return host_loop(exp_loop, value)
    value = xp.clip(value, -EXP_LIMIT, EXP_LIMIT)
    k = xp.rint(value * INV_LN2)
    r = value - k * LN2_HI
    r = r - k * LN2_LO
    series = horner(r, [1.0 / math.factorial(n) for n in range(EXP_TERMS)], xp)
    return xp.ldexp(series, k.astype(xp.int32))


def expm1(x: ArrayLike, xp: ModuleType | None = None) -> Array:
    """``e**x - 1``, without the cancellation of ``exp(x) - 1`` near 0 (the Taylor series ``x + x**2/2 + ...``
    there)."""
    xp = array_module(x, xp=xp)
    value = as_f64(x, xp)
    series = value * horner(value, [1.0 / math.factorial(n + 1) for n in range(EXP_TERMS)], xp)
    return xp.where(xp.abs(value) < EXPM1_SERIES_LIMIT, series, exp(value, xp) - 1.0)


def frexp(x: ArrayLike, xp: ModuleType | None = None) -> tuple[Array, Array]:
    """``x = m 2**e`` with ``|m|`` in ``[1/2, 1)``, read off the float64 bit pattern; ``0``, ``inf`` and ``nan`` give
    themselves and ``e = 0``. ``cupy.frexp`` on ROCm rounds ``m`` to float32 precision, so the library's own is not
    used."""
    xp = array_module(x, xp=xp)
    # A C-order copy, so the bit view below is defined; ``ascontiguousarray`` would make a 0-d value 1-d.
    value = as_f64(x, xp).copy(order="C")
    special = (value == 0.0) | ~xp.isfinite(value)
    subnormal = ~special & (xp.abs(value) < SMALLEST_NORMAL)
    # A subnormal is scaled into the normal range first (exact: a power of two), and its exponent shifted back.
    scaled = xp.where(subnormal, xp.where(subnormal, value, 0.0) * SUBNORMAL_SCALE, value).copy(order="C")
    bits = scaled.view(xp.int64)
    exponent = ((bits >> MANTISSA_BITS) & EXPONENT_MASK) - EXPONENT_HALF
    exponent = xp.where(subnormal, exponent - SUBNORMAL_SHIFT, exponent)
    mantissa = ((bits & ~(EXPONENT_MASK << MANTISSA_BITS)) | (EXPONENT_HALF << MANTISSA_BITS)).view(xp.float64)
    return xp.where(special, value, mantissa), xp.where(special, 0, exponent).astype(xp.int32)


def divide_by_real(z: ArrayLike, d: ArrayLike, xp: ModuleType | None = None) -> Array:
    """``z / d`` for a complex ``z`` and a real ``d``, one real division per part: a complex division by
    ``d + 0j`` rounds differently on numpy and on cupy."""
    xp = array_module(z, d, xp=xp)
    numerator, denominator = xp.asarray(z), as_f64(d, xp)
    return numerator.real / denominator + 1j * (numerator.imag / denominator)


def interp(x: ArrayLike, points: ArrayLike, values: ArrayLike, xp: ModuleType | None = None) -> Array:
    """Piecewise-linear interpolation of ``values`` at increasing ``points``, as numpy's ``interp`` (clamped to the
    end values outside ``[points[0], points[-1]]``): ``slope * (x - points[j]) + values[j]`` with
    ``slope = (values[j+1] - values[j]) / (points[j+1] - points[j])``, one exact operation at a time (cupy orders
    the arithmetic differently)."""
    xp = array_module(x, points, values, xp=xp)
    at, knots, heights = as_f64(x, xp), as_f64(points, xp), as_f64(values, xp)
    last = knots.shape[0] - 1
    j = xp.clip(xp.searchsorted(knots, at, side="right") - 1, 0, max(last - 1, 0))
    if last == 0:
        return xp.full(at.shape, heights[0], dtype=xp.float64)
    slope = (heights[j + 1] - heights[j]) / (knots[j + 1] - knots[j])
    inside = slope * (at - knots[j]) + heights[j]
    inside = xp.where(at < knots[0], heights[0], inside)
    return xp.where(at >= knots[last], heights[last], inside)


def log(x: ArrayLike, xp: ModuleType | None = None) -> Array:
    """Natural logarithm of a positive ``x``: ``x = m 2**e`` with ``m`` in ``[sqrt(1/2), sqrt(2))``, then
    ``log m = 2 atanh((m - 1) / (m + 1))``. ``0`` gives ``-inf``; a negative ``x`` gives ``nan``."""
    xp = array_module(x, xp=xp)
    value = as_f64(x, xp)
    if on_host_loop(value, xp):
        return host_loop(log_loop, value)
    # A non-positive x takes the series at 1.0, whose result the last two lines replace.
    mantissa, exponent = frexp(xp.where(value > 0.0, value, 1.0), xp)
    low = mantissa < math.sqrt(0.5)
    mantissa = xp.where(low, mantissa * 2.0, mantissa)
    exponent = xp.where(low, exponent - 1, exponent).astype(xp.float64)
    s = (mantissa - 1.0) / (mantissa + 1.0)
    s2 = s * s
    series = horner(s2, [1.0 / (2 * n + 1) for n in range(ATANH_TERMS)], xp)
    result = 2.0 * s * series
    result = result + exponent * LN2_LO
    result = result + exponent * LN2_HI
    result = xp.where(value == 0.0, -xp.inf, result)
    return xp.where(value < 0.0, xp.nan, result)


def reduce_half_pi(x: ArrayLike, xp: ModuleType) -> tuple[Array, Array]:
    """``x = k pi/2 + r`` with ``|r| <= pi/4``: ``r`` and ``k mod 4`` (as float64)."""
    value = as_f64(x, xp)
    k = xp.rint(value * INV_HALF_PI)
    r = value - k * HALF_PI_1
    r = r - k * HALF_PI_2
    r = r - k * HALF_PI_3
    return r, k - 4.0 * xp.floor(k * 0.25)


def sin_cos_series(r: Array, xp: ModuleType) -> tuple[Array, Array]:
    """``sin r`` and ``cos r`` for ``|r| <= pi/4`` by their Taylor series in ``r**2``."""
    r2 = r * r
    sin_terms = [(-1.0) ** n / math.factorial(2 * n + 1) for n in range(TRIG_TERMS)]
    cos_terms = [(-1.0) ** n / math.factorial(2 * n) for n in range(TRIG_TERMS)]
    return r * horner(r2, sin_terms, xp), horner(r2, cos_terms, xp)


def sin(x: ArrayLike, xp: ModuleType | None = None) -> Array:
    """``sin x``, for ``|x| < 2**20`` (larger arguments lose the reduction's exactness)."""
    xp = array_module(x, xp=xp)
    if on_host_loop(value := as_f64(x, xp), xp):
        return host_loop(trig_loop, value, False)
    r, quadrant = reduce_half_pi(x, xp)
    s, c = sin_cos_series(r, xp)
    return xp.where(quadrant == 0.0, s, xp.where(quadrant == 1.0, c, xp.where(quadrant == 2.0, -s, -c)))


def cos(x: ArrayLike, xp: ModuleType | None = None) -> Array:
    """``cos x``, for ``|x| < 2**20``."""
    xp = array_module(x, xp=xp)
    if on_host_loop(value := as_f64(x, xp), xp):
        return host_loop(trig_loop, value, True)
    r, quadrant = reduce_half_pi(x, xp)
    s, c = sin_cos_series(r, xp)
    return xp.where(quadrant == 0.0, c, xp.where(quadrant == 1.0, -s, xp.where(quadrant == 2.0, -c, s)))


def tanh(x: ArrayLike, xp: ModuleType | None = None) -> Array:
    """``tanh x = (e**2x - 1) / (e**2x + 1)``, with the sign folded so the exponential never overflows."""
    xp = array_module(x, xp=xp)
    value = as_f64(x, xp)
    e = exp(-2.0 * xp.abs(value), xp)
    magnitude = (1.0 - e) / (1.0 + e)
    return xp.where(value < 0.0, -magnitude, magnitude)


def ipow(x: ArrayLike, n: int, xp: ModuleType | None = None) -> Array:
    """``x**n`` for a non-negative integer ``n`` by repeated squaring (``x**2`` is one multiply)."""
    xp = array_module(x, xp=xp)
    if n < 0:
        raise ValueError(f"ipow: negative exponent {n}; divide by ipow(x, {-n})")
    base = xp.asarray(x)
    result = xp.ones_like(base)
    while n:
        if n & 1:
            result = result * base
        n >>= 1
        if n:
            base = base * base
    return result


def pow(x: ArrayLike, y: ArrayLike, xp: ModuleType | None = None) -> Array:  # noqa: A001 -- the function's own name in every math library
    """``x**y`` for a positive ``x``: ``exp(y log x)``."""
    xp = array_module(x, y, xp=xp)
    return exp(as_f64(y, xp) * log(x, xp), xp)


def linspace(start: float, stop: float, num: int, endpoint: bool = True, xp: ModuleType = np) -> Array:
    """``num`` evenly spaced float64 values from ``start``: ``start + (stop - start) * (i / d)`` with ``d`` the
    number of intervals, one exact operation at a time (numpy and cupy compute ``linspace`` differently). The last
    value is ``stop`` exactly when ``endpoint``."""
    count = int(num)
    intervals = max(count - 1, 1) if endpoint else max(count, 1)
    fraction = xp.arange(count, dtype=xp.float64) / intervals
    values = start + (stop - start) * fraction
    if endpoint and count > 1:
        values[-1] = stop
    return values


def logspace(start: float, stop: float, num: int, base: float = 10.0, xp: ModuleType = np) -> Array:
    """``base ** linspace(start, stop, num)``."""
    return pow(xp.full(int(num), base, dtype=xp.float64), linspace(start, stop, num, xp=xp), xp)


def exact_in_any_order(values: Array) -> bool:
    """Whether a reduction of ``values`` is exact whatever its order: integers and booleans."""
    return values.dtype.kind in "biu"


def sum(x: ArrayLike, axis: int | None = None, keepdims: bool = False, xp: ModuleType | None = None) -> Array:  # noqa: A001 -- numpy's name
    """``x`` summed over ``axis`` (every element for ``None``) in a fixed binary-tree order."""
    xp = array_module(x, xp=xp)
    values = xp.asarray(x)
    if exact_in_any_order(values):
        return values.sum(axis=axis, keepdims=keepdims)
    if axis is None:
        flat = sum(values.reshape(-1), 0, False, xp)
        return flat.reshape((1,) * values.ndim) if keepdims else flat
    axis = axis % values.ndim
    rows = xp.moveaxis(values, axis, 0)
    if rows.shape[0] > 0 and values.dtype in (np.float32, np.float64) and on_host_loop(values, xp):
        total = tree_sum_loop(np.ascontiguousarray(rows.reshape(rows.shape[0], -1))).reshape(rows.shape[1:])
        return xp.expand_dims(total, axis) if keepdims else total
    if rows.shape[0] == 0:
        total = xp.zeros(rows.shape[1:], dtype=values.dtype)
    else:
        while rows.shape[0] > 1:
            half = rows.shape[0] // 2
            paired = rows[:half] + rows[half : 2 * half]
            rows = xp.concatenate((paired, rows[2 * half :]), axis=0) if rows.shape[0] % 2 else paired
        total = rows[0]
    return xp.expand_dims(total, axis) if keepdims else total


def mean(x: ArrayLike, axis: int | None = None, keepdims: bool = False, xp: ModuleType | None = None) -> Array:
    """:func:`sum` divided by the count, as float64 for an integer ``x``."""
    xp = array_module(x, xp=xp)
    values = xp.asarray(x)
    count = values.size if axis is None else values.shape[axis]
    total = sum(values.astype(xp.float64) if exact_in_any_order(values) else values, axis, keepdims, xp)
    return total / max(count, 1)


def cumsum(x: ArrayLike, axis: int | None = None, xp: ModuleType | None = None) -> Array:
    """Running sum over ``axis`` (the flattened ``x`` for ``None``) by a Hillis-Steele scan: step ``d`` adds the
    element ``d`` back to every element, for ``d = 1, 2, 4, ...``."""
    xp = array_module(x, xp=xp)
    values = xp.asarray(x)
    if exact_in_any_order(values):
        return xp.cumsum(values, axis=axis)
    if axis is None:
        values, axis = values.reshape(-1), 0
    rows = xp.moveaxis(values, axis % values.ndim, 0).copy()
    step = 1
    while step < rows.shape[0]:
        shifted = rows[:-step].copy()
        rows[step:] = rows[step:] + shifted
        step *= 2
    return xp.ascontiguousarray(xp.moveaxis(rows, 0, axis % values.ndim))


def matvec(a: ArrayLike, x: ArrayLike, xp: ModuleType | None = None) -> Array:
    """``a @ x`` for a matrix ``a`` and a vector ``x``: each row's products summed by :func:`sum`."""
    xp = array_module(a, x, xp=xp)
    return sum(xp.asarray(a) * xp.asarray(x)[None, :], 1, False, xp)


def norm(x: ArrayLike, axis: int | None = None, keepdims: bool = False, xp: ModuleType | None = None) -> Array:
    """The Euclidean norm over ``axis``: ``sqrt`` of the :func:`sum` of squares (``|x|**2`` for complex)."""
    xp = array_module(x, xp=xp)
    values = xp.asarray(x)
    squares = (values * values.conj()).real if values.dtype.kind == "c" else values * values
    return xp.sqrt(sum(squares, axis, keepdims, xp))


def matmul(a: ArrayLike, b: ArrayLike, xp: ModuleType | None = None) -> Array:
    """``a @ b`` for matrices: every product formed, then summed over the inner axis by :func:`sum`. Memory is
    ``m k n``, so this is for the small products of an initializer (a few projectors, three coordinates)."""
    xp = array_module(a, b, xp=xp)
    left, right = xp.asarray(a), xp.asarray(b)
    return sum(left[:, :, None] * right[None, :, :], 1, False, xp)
