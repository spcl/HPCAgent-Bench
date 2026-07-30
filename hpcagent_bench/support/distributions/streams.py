# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One RNG stream per array, instead of one stream per kernel.

A single ``default_rng(seed)`` threaded through every array makes array *N*'s values depend on how
many draws arrays *0..N-1* happened to make. That couples the arrays three ways: materialising one
array alone gives different data than materialising all of them, adding an array shifts every array
after it, and the fill cannot be parallelised at all.

``SeedSequence.spawn`` breaks the coupling: child *i* is a function of the seed and *i* only.
Measured on 8 arrays of 1e7 doubles -- 708ms sequential, 181ms on 8 threads (3.91x), values
bit-identical either way, because NumPy releases the GIL inside the fill.

The bit generators are round-robined over the spawned children. MT19937 is deliberately absent:
1.9x slower than PCG64 for the same fill and 2.5KB of state per stream, in exchange for nothing.
"""
import concurrent.futures
from typing import Any, Callable, List, Sequence

import numpy as np
from numpy.random import PCG64, PCG64DXSM, SFC64, Generator, Philox, SeedSequence

#: Cycled over the spawned children, so consecutive arrays draw from different algorithms.
#: Philox is the slow one (+38%) but it is the counter-based member of the set, and it is the only
#: one of the four with a GPU counterpart at all -- keep it in the rotation.
ROUND_ROBIN = (PCG64, PCG64DXSM, SFC64, Philox)

#: Below this many elements the fill is not worth a thread; pool startup dominates.
THREAD_MIN_ELEMENTS = 1 << 22


def spawn_streams(seed: Any, count: int) -> List[Generator]:
    """``count`` independent generators, round-robined over :data:`ROUND_ROBIN`.

    ``seed`` of ``None`` still spawns -- the children are independent of each other, just not
    reproducible across runs.
    """
    return [
        Generator(ROUND_ROBIN[i % len(ROUND_ROBIN)](child)) for i, child in enumerate(SeedSequence(seed).spawn(count))
    ]


def fill(tasks: Sequence[Callable[[], Any]], elements: int, workers: int = 8) -> List[Any]:
    """Run the per-array fills, on threads once there is enough work to pay for them.

    Order-preserving, and identical to calling each task in turn: the tasks draw from streams that
    are independent by construction, so concurrency cannot perturb the values.
    """
    if elements < THREAD_MIN_ELEMENTS or len(tasks) < 2:
        return [t() for t in tasks]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        return list(pool.map(lambda t: t(), tasks))


def clip_to_precision(raw: np.ndarray, cap: float) -> np.ndarray:
    """Clamp SAMPLED values into the precision's safe range, in place.

    Clamping the distribution's BOUNDS instead can invert a range lying entirely above the cap into
    ``low > high``; clamping the samples cannot.
    """
    np.clip(raw, -cap, cap, out=raw)
    return raw
