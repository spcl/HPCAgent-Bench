# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""``noise``: an input distribution that multiplies float inputs by ``1 + eps * u``, ``u`` in ``[-1, 1)``.

It is opt-in and never default. Three ways to select it, none of which a manifest or run gets by accident:

* ``dist: noise`` on one array of a manifest's ``init.arrays``: the array is drawn from ``uniform`` and perturbed;
* ``distribution="noise"`` for a run (``variant_spec={"base": "normal", "eps": 1e-5}`` names another base and a step),
  which draws every array without a distribution of its own that way;
* ``inputs.noise: true`` in the configuration (``HPCAGENT_BENCH_INPUTS_NOISE=1``), which perturbs EVERY float input
  array of every initializer, declarative or custom, after it has been built (:func:`apply_to_inputs`).

``u`` is the counter generator's uniform draw (:mod:`hpcagent_bench.support.counter_rng`) at the flat index of each
element, keyed by the draw's seed and the array's position: the same bits on every machine and library, distinct per
array, and different for every seed. The error is relative, so zeros stay zero, signs stay, and a positive array stays
positive. An array the noise would push out of its declared interval is clipped to it, and any other array to its own
largest magnitude, so the perturbation never widens a range. Integer, index, boolean and sparse payloads, and arrays
of a structural distribution (``well_conditioned``, ``near_singular``, ...), are left as they are: a multiplicative
error would break the structure those exist to provide.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from hpcagent_bench.dtypes import compute_view, is_float_dtype
from hpcagent_bench.precision import Precision
from hpcagent_bench.support import counter_rng
from hpcagent_bench.support.distributions import domain as domain_mod
from hpcagent_bench.support.distributions import register_distribution

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from hpcagent_bench.spec import BenchSpec

__all__ = ["DEFAULT_EPS", "NOISE_SPEC_SEED", "apply_to_inputs", "default_eps", "perturb"]

#: The relative step per format: about four units in the last place of the format for the narrow ones, where a
#: smaller step would round away, and 1e-6 (4.5e9 ulps) for float64, where the step is visible only to a kernel
#: that reads its input exactly. A format without an entry (float8) has no step that is both seen and small.
DEFAULT_EPS = {"float64": 1e-6, "float32": 1e-5, "float16": 4e-3, "bfloat16": 3e-2}

#: The stream a registered ``noise`` draw takes its seed from in the array's own generator.
NOISE_SPEC_SEED = "noise_seed"


def default_eps(dtype: np.typing.DTypeLike) -> float | None:
    """The step ``noise`` uses for ``dtype`` when none is given; ``None`` for a format with no useful one."""
    return DEFAULT_EPS.get(np.dtype(dtype).name)


def perturb(
    values: np.ndarray,
    seed: int,
    stream: int,
    eps: float | None = None,
    interval: tuple[float, float] | None = None,
) -> np.ndarray:
    """``values * (1 + eps * u)`` for the counter draw ``u`` in ``[-1, 1)`` of ``(seed, stream)``, in ``values``' dtype.

    ``eps`` defaults per format (:data:`DEFAULT_EPS`); an array of a format with none is returned unchanged. The
    result stays inside ``interval`` when one is given, else inside ``+-max|values|``."""
    step = eps if eps else default_eps(values.dtype)
    if step is None or values.size == 0:
        return values
    wide = compute_view(values)
    draw = counter_rng.uniform_field(
        wide.shape, seed, stream, dtype=wide.dtype if wide.dtype == np.float32 else np.float64
    )
    factor = 1.0 + wide.dtype.type(step) * (draw * 2.0 - 1.0).astype(wide.dtype, copy=False)
    result = wide * factor
    low, high = interval if interval is not None else (-float(np.max(np.abs(wide))), float(np.max(np.abs(wide))))
    return np.clip(result, low, high).astype(values.dtype, copy=False)


@register_distribution("noise")
def noise(shape: tuple[int, ...], precision: Precision, spec: dict[str, Any] | None) -> np.ndarray | dict[str, Any]:
    """The ``base`` distribution (default ``uniform``) perturbed by :func:`perturb`; ``eps`` overrides the step. The
    noise seed is drawn from the array's own generator, so it is fixed by the run's seed and the array's position."""
    from hpcagent_bench.support import distributions  # the registry imports this module, so not at the top

    spec = dict(spec or {})
    base = str(spec.pop("base", "uniform"))
    eps = float(spec.pop("eps", 0.0))
    values = distributions.get(base)(shape, precision, spec)
    rng = spec.get("rng")
    seed = int(rng.integers(0, 2**63)) if rng is not None else int(np.random.default_rng().integers(0, 2**63))
    if not isinstance(values, np.ndarray) or not is_float_dtype(values.dtype):
        return values
    interval = domain_mod.of(spec)
    return perturb(values, seed, 0, eps or None, interval if isinstance(interval, tuple) else None)


def apply_to_inputs(
    spec: "BenchSpec", data: "MutableMapping[str, Any]", seed: int, eps: float | None = None
) -> list[str]:
    """Perturb, in ``data``, every float array of ``spec.init.output_args`` that is not of a structural distribution,
    by :func:`perturb` with the array's position in ``output_args`` as its stream. Returns the names perturbed."""
    init = spec.init
    if init is None:
        return []
    touched: list[str] = []
    for stream, name in enumerate(init.output_args):
        value = data.get(name)
        if not isinstance(value, np.ndarray) or not is_float_dtype(value.dtype):
            continue
        if init.dists.get(name) in domain_mod.STRUCTURAL:
            continue
        interval = domain_mod.of({"domain": init.domains[name]}) if name in init.domains else None
        data[name] = perturb(value, seed, stream, eps, interval if isinstance(interval, tuple) else None)
        touched.append(name)
    return touched
