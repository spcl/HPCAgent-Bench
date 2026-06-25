# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dimension fuzzing for benchmark inputs.

A kernel may declare a ``fuzzed`` preset whose size params are RANGES
(``N: [lo, hi]``) instead of scalars::

    parameters:
      S:      {N: 400000, npt: 1000}
      L:      {N: 1000000, npt: 1000}
      fuzzed: {N: [1000000, 4000000], npt: 1000}   # N fuzzed, npt fixed

Absent an explicit ``fuzzed`` preset, the range defaults to
``[L * size_lo_mult, L * size_hi_mult]`` from ``config.yaml`` (so every
kernel is fuzzable without a manifest edit). For fuzz iteration ``i``, each
range is sampled (log-uniform by default) from a seeded RNG (``seeds.fuzz + i``)
so a run is reproducible yet varied across iterations. Scalar params pass
through unchanged.
"""
import numpy as np

from optarena import config
from typing import Any, Dict

FUZZED_PRESET = "fuzzed"


def is_range(value: Any) -> bool:
    """``True`` when a parameter value is a ``[lo, hi]`` fuzz range."""
    return (isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value))


def _sample_one(lo: float, hi: float, rng, distribution: str) -> int:
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        return lo
    if distribution == "log_uniform":
        val = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    else:  # uniform
        val = float(rng.uniform(lo, hi))
    return int(round(val))


def resolve_ranges(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Per-param fuzz spec: each value is a ``[lo, hi]`` range or a fixed scalar.

    Prefers an explicit ``fuzzed`` preset; otherwise the default range brackets
    the ``L`` (publication) size: ``[L, L + XL]`` per integer size param
    (lo = the ``L`` value, hi = ``L + XL`` -- always >= L, "big enough"). Falls
    back to ``L * fuzz.size_hi_mult`` for the high bound when there is no ``XL``
    preset, and to the largest preset when there is no ``L``. Non-integer /
    size-1 params are kept fixed.
    """
    if FUZZED_PRESET in parameters:
        return dict(parameters[FUZZED_PRESET])
    base = (parameters.get("L") or next(iter(parameters.values())))
    step = parameters.get("XL") or {}  # additive width: from L toward the XL (GPU) size
    hi_m = float(config.get("fuzz.size_hi_mult", 4.0))
    out: Dict[str, Any] = {}
    for name, value in base.items():
        if isinstance(value, int) and value > 1:
            hi = value + int(step[name]) if isinstance(step.get(name), int) else int(value * hi_m)
            out[name] = [value, max(hi, value)]
        else:
            out[name] = value
    return out


def pick_data_distribution(fuzz_spec: Dict[str, Any], iteration: int = 0) -> str:
    """The input-value distribution for fuzz ``iteration``.

    A kernel's manifest ``fuzz.data_distributions`` lists one or more registered
    distributions (scipy-backed or numpy); iterations CYCLE through them so a
    sweep probes each. Falls back to the singular ``fuzz.data_distribution``
    (manifest or config) when no list is given. Returns ``""`` if nothing is set
    (the caller keeps its own default).
    """
    dists = (fuzz_spec or {}).get("data_distributions")
    if isinstance(dists, (list, tuple)) and dists:
        return str(dists[int(iteration) % len(dists)])
    return str((fuzz_spec or {}).get("data_distribution", "") or "")


def sample_params(parameters: Dict[str, Any], iteration: int = 0) -> Dict[str, int]:
    """Concrete size params for fuzz ``iteration`` -- seeded by
    ``seeds.fuzz + iteration`` so the draw is reproducible yet varies per
    iteration. Ranges are sampled per ``fuzz.size_distribution``; scalars pass
    through."""
    ranges = resolve_ranges(parameters)
    seed = int(config.get("seeds.fuzz", 42)) + int(iteration)
    rng = np.random.default_rng(seed)
    distribution = config.get("fuzz.size_distribution", "log_uniform")
    out: Dict[str, int] = {}
    for name, value in ranges.items():
        out[name] = _sample_one(value[0], value[1], rng, distribution) if is_range(value) else value
    return out


def iterations() -> int:
    """Configured number of fuzz iterations (``fuzz.iterations``)."""
    return int(config.get("fuzz.iterations", 20))
