# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""The GPU implementation of :mod:`hpcagent_bench.support.counter_rng`: one elementwise cupy kernel per draw,
every element (a block of threads per block of elements) hashing its own flat index.

The kernels run the reference's integer arithmetic on 64-bit unsigned words and its exact float conversions, so the
bits equal numpy's. ``first`` is the flat index of the first element, as in :mod:`counter_rng_numba`. The kernels
compile on first use (NVRTC or hipRTC) and cupy caches them.
"""

import cupy as cp  # pyright: ignore[reportMissingImports]  # optional dep, not in the dev env
import numpy as np

from hpcagent_bench.support import counter_rng as reference

__all__ = [
    "CENTRE",
    "INTEGERS",
    "NORMAL",
    "PREAMBLE",
    "SCALE_21",
    "UNIFORM",
    "integers",
    "normal",
    "uniform",
]

PREAMBLE = f"""
__device__ inline unsigned long long counter_hash(unsigned long long index, unsigned long long key) {{
    unsigned long long state = (index + 1ULL) * {reference.GAMMA}ULL + key;
    state ^= state >> {reference.MIX_SHIFT_1};
    state *= {reference.MULTIPLIER_1}ULL;
    state ^= state >> {reference.MIX_SHIFT_2};
    state *= {reference.MULTIPLIER_2}ULL;
    return state ^ (state >> {reference.MIX_SHIFT_3});
}}
"""

SCALE_21 = 0.5**reference.PIECE_BITS
CENTRE = reference.PIECES * (1.0 - SCALE_21) / 2.0

UNIFORM = cp.ElementwiseKernel(
    "uint64 first, uint64 key, uint32 shift, float64 scale",
    "T out",
    "out = (T)((double)(counter_hash(first + (unsigned long long)i, key) >> shift) * scale);",
    "counter_rng_uniform",
    preamble=PREAMBLE,
)
NORMAL = cp.ElementwiseKernel(
    "uint64 first, uint64 key",
    "T out",
    f"""
    unsigned long long base = (first + (unsigned long long)i) * {reference.NORMAL_WORDS}ULL;
    unsigned long long total = 0ULL;
    for (int word = 0; word < {reference.NORMAL_WORDS}; ++word) {{
        unsigned long long value = counter_hash(base + (unsigned long long)word, key);
        total += value & {reference.PIECE_MASK}ULL;
        total += (value >> {reference.PIECE_BITS}) & {reference.PIECE_MASK}ULL;
        total += (value >> {2 * reference.PIECE_BITS}) & {reference.PIECE_MASK}ULL;
    }}
    out = (T)((double)total * {SCALE_21!r} - {CENTRE!r});
    """,
    "counter_rng_normal",
    preamble=PREAMBLE,
)
INTEGERS = cp.ElementwiseKernel(
    "uint64 first, uint64 key, uint64 bound",
    "T out",
    "out = (T)(counter_hash(first + (unsigned long long)i, key) % bound);",
    "counter_rng_integers",
    preamble=PREAMBLE,
)


def uniform(out: cp.ndarray, first: int, seed: int, stream: int) -> cp.ndarray:
    """Fill the float32 or float64 ``out`` with the uniform draw of the flat indices ``first`` onward."""
    mantissa = reference.F64_BITS if out.dtype == np.float64 else reference.F32_BITS
    UNIFORM(
        np.uint64(first), np.uint64(reference.key(seed, stream)), np.uint32(reference.WORD_BITS - mantissa),
        0.5**mantissa, out,
    )  # fmt: skip
    return out


def normal(out: cp.ndarray, first: int, seed: int, stream: int) -> cp.ndarray:
    """Fill the float32 or float64 ``out`` with the normal draw of the flat indices ``first`` onward."""
    NORMAL(np.uint64(first), np.uint64(reference.key(seed, stream)), out)
    return out


def integers(out: cp.ndarray, first: int, seed: int, bound: int, stream: int) -> cp.ndarray:
    """Fill the int64 ``out`` with the integer draw below ``bound`` of the flat indices ``first`` onward."""
    INTEGERS(np.uint64(first), np.uint64(reference.key(seed, stream)), np.uint64(bound), out)
    return out
