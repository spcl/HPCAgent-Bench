# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""numpy's ``Generator`` draw methods on the counter-based generator, for kernel initializers.

An initializer receives ``rng``, a :class:`CounterGenerator`, and ``xp``, the array module its arrays live in
(numpy, or cupy when the grade's data is device-resident). Each draw call takes the next stream of the
generator's seed and fills its whole array from :mod:`hpcagent_bench.support.counter_rng`, so

* the same calls give the same bits on numpy and cupy: a GPU grade builds its inputs on the device, and they
  equal the host inputs the cached reference ran on;
* every call is one parallel fill (numba threads on the host, an elementwise kernel on the device);
* call ``k``'s values depend on the seed and ``k`` only, never on how large the earlier draws were.

The continuous distributions beyond uniform and normal transform those draws with
:mod:`hpcagent_bench.support.exact_math`, never with the array library's own ``exp``/``log``, which round
differently in glibc and in a GPU's math library. ``gamma`` is the Wilson-Hilferty cube of a normal draw: no
rejection loop (whose draw count would depend on the values), and close to the true law for a shape of 1 or more.
"""

import dataclasses
import math
import secrets
from collections.abc import Sequence
from types import ModuleType
from typing import Any, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray

from hpcagent_bench.support import counter_rng, exact_math
from hpcagent_bench.support.array_module import module_of

__all__ = ["OPEN_FLOOR", "SEED_BITS", "CounterGenerator", "Draw", "Shape", "Size", "shape_of"]

#: Bits of a seed drawn from fresh entropy (a non-negative int64).
SEED_BITS = 63

#: The least uniform draw a logarithm takes: ``u`` is in [0, 1), and ``log(0)`` is not a value.
OPEN_FLOOR = 0.5**54

#: A draw's ``size``: ``None`` for one value, an extent, or a shape -- numpy's convention.
type Size = int | Sequence[int] | None
#: What a draw returns: the array, or one Python number for ``size=None`` (each method's overloads say which).
type Draw = NDArray[Any] | int | float
#: A location or scale argument: a number, or an array broadcast against the draw.
type Operand = float | NDArray[Any]
#: A sized draw's ``size``.
type Shape = int | Sequence[int]


def shape_of(size: Size) -> tuple[int, ...]:
    """``size`` as a shape; ``None`` is one element (the caller unwraps it)."""
    if size is None:
        return (1,)
    if isinstance(size, (int, np.integer)):
        return (int(size),)
    return tuple(int(extent) for extent in size)


@dataclasses.dataclass(slots=True)
class CounterGenerator:
    """The subset of ``numpy.random.Generator`` initializers use, drawn from ``counter_rng`` on ``xp``.

    ``stream`` is the stream the next call takes. A draw with ``size=None`` returns a Python scalar."""

    seed: int
    xp: ModuleType = np
    stream: int = 0

    @classmethod
    def fresh(cls, xp: ModuleType = np) -> "CounterGenerator":
        """A generator of fresh entropy: different on every call, as an unseeded numpy generator."""
        return cls(secrets.randbits(SEED_BITS), xp)

    def next_stream(self) -> int:
        """The stream of the next draw; advances the generator."""
        stream = self.stream
        self.stream += 1
        return stream

    def scalar_or_array(self, values: NDArray[Any], size: Size) -> Draw:
        """``values`` as numpy returns them: one Python number for ``size=None``, else the array."""
        if size is None:
            return values.reshape(-1)[0].item()
        return self.xp.asarray(values)  # 0-d arithmetic gives a scalar; a sized draw is always an array

    @overload
    def random(self, size: None = None, dtype: DTypeLike = np.float64) -> float: ...
    @overload
    def random(self, size: Shape, dtype: DTypeLike = np.float64) -> NDArray[Any]: ...

    def random(self, size: Size = None, dtype: DTypeLike = np.float64) -> Draw:
        """Uniform in [0, 1)."""
        values = counter_rng.uniform_field(shape_of(size), self.seed, self.next_stream(), self.xp, dtype)
        return self.scalar_or_array(values, size)

    @overload
    def uniform(self, low: float = 0.0, high: float = 1.0, size: None = None) -> float: ...
    @overload
    def uniform(self, low: Operand, high: Operand, size: Shape) -> NDArray[Any]: ...
    @overload
    def uniform(self, low: Operand = 0.0, high: Operand = 1.0, *, size: Shape) -> NDArray[Any]: ...
    @overload
    def uniform(self, low: NDArray[Any], high: Operand = 1.0, size: None = None) -> NDArray[Any]: ...
    @overload
    def uniform(self, low: float, high: NDArray[Any], size: None = None) -> NDArray[Any]: ...

    def uniform(self, low: Operand = 0.0, high: Operand = 1.0, size: Size = None) -> Draw:
        """Uniform in [low, high): ``low + (high - low) * u``, as numpy computes it. ``low``/``high`` may be arrays
        (then ``size`` defaults to their broadcast shape)."""
        if size is None and (np.ndim(low) or np.ndim(high)):
            size = np.broadcast_shapes(np.shape(low), np.shape(high))
        unit = counter_rng.uniform_field(shape_of(size), self.seed, self.next_stream(), self.xp)
        return self.scalar_or_array(low + (high - low) * unit, size)

    @overload
    def standard_normal(self, size: None = None, dtype: DTypeLike = np.float64) -> float: ...
    @overload
    def standard_normal(self, size: Shape, dtype: DTypeLike = np.float64) -> NDArray[Any]: ...

    def standard_normal(self, size: Size = None, dtype: DTypeLike = np.float64) -> Draw:
        """Mean 0, variance 1 (Irwin-Hall, tails at +-6)."""
        values = counter_rng.normal_field(shape_of(size), self.seed, self.next_stream(), self.xp, dtype)
        return self.scalar_or_array(values, size)

    @overload
    def normal(self, loc: float = 0.0, scale: float = 1.0, size: None = None) -> float: ...
    @overload
    def normal(self, loc: Operand, scale: Operand, size: Shape) -> NDArray[Any]: ...
    @overload
    def normal(self, loc: Operand = 0.0, scale: Operand = 1.0, *, size: Shape) -> NDArray[Any]: ...
    @overload
    def normal(self, loc: NDArray[Any], scale: Operand = 1.0, size: None = None) -> NDArray[Any]: ...
    @overload
    def normal(self, loc: float, scale: NDArray[Any], size: None = None) -> NDArray[Any]: ...

    def normal(self, loc: Operand = 0.0, scale: Operand = 1.0, size: Size = None) -> Draw:
        """``loc + scale * z``; ``loc``/``scale`` may be arrays."""
        if size is None and (np.ndim(loc) or np.ndim(scale)):
            size = np.broadcast_shapes(np.shape(loc), np.shape(scale))
        unit = counter_rng.normal_field(shape_of(size), self.seed, self.next_stream(), self.xp)
        return self.scalar_or_array(loc + scale * unit, size)

    @overload
    def integers(
        self, low: int, high: int | None = None, size: None = None, dtype: DTypeLike = np.int64, endpoint: bool = False
    ) -> int: ...
    @overload
    def integers(
        self, low: int, high: int | None, size: Shape, dtype: DTypeLike = np.int64, endpoint: bool = False
    ) -> NDArray[Any]: ...

    def integers(
        self, low: int, high: int | None = None, size: Size = None, dtype: DTypeLike = np.int64, endpoint: bool = False
    ) -> Draw:
        """Integers in [low, high) (``[low, high]`` with ``endpoint``); ``integers(n)`` is [0, n)."""
        if high is None:
            low, high = 0, low
        bound = int(high) - int(low) + int(endpoint)
        if bound <= 0:
            raise ValueError(f"integers: empty range [{low}, {high}{']' if endpoint else ')'}")
        drawn = counter_rng.integers_field(shape_of(size), self.seed, bound, self.next_stream(), self.xp)
        values = (drawn + int(low)).astype(dtype, copy=False)
        return self.scalar_or_array(values, size)

    def permutation(self, x: int | NDArray[Any]) -> NDArray[Any]:
        """A permutation of ``range(x)``, or of ``x``'s first axis: the stable order of one 64-bit hash per
        position, which every array library sorts the same way."""
        count = int(x) if isinstance(x, (int, np.integer)) else int(x.shape[0])
        keys = counter_rng.bits(self.xp.arange(count, dtype=self.xp.uint64), self.seed, self.next_stream(), self.xp)
        order = self.xp.argsort(keys, kind="stable")
        if isinstance(x, (int, np.integer)):
            return order
        return x[order]

    def shuffle(self, x: NDArray[Any]) -> None:
        """Permute ``x`` along its first axis, in place."""
        x[...] = self.permutation(x)

    @overload
    def choice(
        self,
        a: int | NDArray[Any] | Sequence[Any],
        size: None = None,
        replace: bool = True,
        p: Sequence[float] | NDArray[Any] | None = None,
    ) -> Any: ...  # noqa: ANN401 -- one element of ``a``, whatever ``a`` holds
    @overload
    def choice(
        self,
        a: int | NDArray[Any] | Sequence[Any],
        size: Shape,
        replace: bool = True,
        p: Sequence[float] | NDArray[Any] | None = None,
    ) -> NDArray[Any]: ...

    def choice(
        self,
        a: int | NDArray[Any] | Sequence[Any],
        size: Size = None,
        replace: bool = True,
        p: Sequence[float] | NDArray[Any] | None = None,
    ) -> Draw:
        """Draw from ``a`` (``range(a)`` for an int). ``p`` weights are accumulated on the host, so every array
        library compares the same thresholds; without replacement, the first ``size`` of a permutation."""
        population = int(a) if isinstance(a, (int, np.integer)) else len(a)
        if not replace:
            count = math.prod(shape_of(size))
            if count > population:
                raise ValueError(f"choice: {count} draws without replacement from {population}")
            picked = self.permutation(population)[:count].reshape(shape_of(size))
        elif p is None:
            picked = self.integers(0, population, shape_of(size))
        else:
            device = module_of(p)
            weights = device.asnumpy(p) if device not in (None, np) else p  # thresholds are summed on the host
            cumulative = np.cumsum(np.asarray(weights, dtype=np.float64))
            cumulative /= cumulative[-1]
            unit = counter_rng.uniform_field(shape_of(size), self.seed, self.next_stream(), self.xp)
            picked = self.xp.minimum(
                self.xp.searchsorted(self.xp.asarray(cumulative), unit, side="right"), population - 1
            )
        values = picked if isinstance(a, (int, np.integer)) else self.xp.asarray(a)[picked]
        return self.scalar_or_array(values, size)

    def open_unit(self, size: Size) -> NDArray[np.float64]:
        """Uniform in (0, 1): the next stream's draw, lifted off zero for a logarithm."""
        unit = counter_rng.uniform_field(shape_of(size), self.seed, self.next_stream(), self.xp)
        return self.xp.maximum(unit, OPEN_FLOOR)

    @overload
    def standard_exponential(self, size: None = None) -> float: ...
    @overload
    def standard_exponential(self, size: Shape) -> NDArray[Any]: ...

    def standard_exponential(self, size: Size = None) -> Draw:
        """``-log u``: exponential of rate 1."""
        return self.exponential(1.0, size)

    @overload
    def exponential(self, scale: float = 1.0, size: None = None) -> float: ...
    @overload
    def exponential(self, scale: float, size: Shape) -> NDArray[Any]: ...

    def exponential(self, scale: float = 1.0, size: Size = None) -> Draw:
        """``-scale log u``."""
        return self.scalar_or_array(-scale * exact_math.log(self.open_unit(size), self.xp), size)

    @overload
    def lognormal(self, mean: float = 0.0, sigma: float = 1.0, size: None = None) -> float: ...
    @overload
    def lognormal(self, mean: float, sigma: float, size: Shape) -> NDArray[Any]: ...

    def lognormal(self, mean: float = 0.0, sigma: float = 1.0, size: Size = None) -> Draw:
        """``exp(mean + sigma z)``."""
        return self.scalar_or_array(exact_math.exp(self.normal(mean, sigma, shape_of(size)), self.xp), size)

    @overload
    def gamma(self, shape: float, scale: float = 1.0, size: None = None) -> float: ...
    @overload
    def gamma(self, shape: float, scale: float, size: Shape) -> NDArray[Any]: ...

    def gamma(self, shape: float, scale: float = 1.0, size: Size = None) -> Draw:
        """Wilson-Hilferty: ``shape (1 - 1/(9 shape) + z / (3 sqrt(shape)))**3``, floored at 0, times ``scale``."""
        if shape <= 0.0:
            raise ValueError(f"gamma: shape {shape} is not positive")
        z = self.standard_normal(shape_of(size))
        root = self.xp.maximum(1.0 - 1.0 / (9.0 * shape) + z / (3.0 * math.sqrt(shape)), 0.0)
        return self.scalar_or_array(shape * scale * exact_math.ipow(root, 3, self.xp), size)

    @overload
    def beta(self, a: float, b: float, size: None = None) -> float: ...
    @overload
    def beta(self, a: float, b: float, size: Shape) -> NDArray[Any]: ...

    def beta(self, a: float, b: float, size: Size = None) -> Draw:
        """``X / (X + Y)`` of two gamma draws of shapes ``a`` and ``b``."""
        x = self.gamma(a, 1.0, shape_of(size))
        y = self.gamma(b, 1.0, shape_of(size))
        total = x + y
        return self.scalar_or_array(self.xp.where(total > 0.0, x / self.xp.where(total > 0.0, total, 1.0), 0.5), size)

    @overload
    def laplace(self, loc: float = 0.0, scale: float = 1.0, size: None = None) -> float: ...
    @overload
    def laplace(self, loc: float, scale: float, size: Shape) -> NDArray[Any]: ...

    def laplace(self, loc: float = 0.0, scale: float = 1.0, size: Size = None) -> Draw:
        """``loc + scale s log u``: a random sign ``s`` times an exponential draw."""
        sign = self.xp.where(self.random(shape_of(size)) < 0.5, -1.0, 1.0)
        return self.scalar_or_array(loc + scale * sign * self.exponential(1.0, shape_of(size)), size)
