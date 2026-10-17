# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dense matrices with a guaranteed structure, built from exact operations only (no matrix product).

A product ``L L^T`` or ``P P^H`` is a floating reduction whose order numpy's BLAS and a GPU's choose differently,
so an initializer that forms one builds different bits on the host and on the device. These builders reach the
same structure -- symmetric (Hermitian) positive definite, or a prescribed condition number -- with elementwise
operations and :mod:`hpcagent_bench.support.exact_math` alone.
"""

from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

from hpcagent_bench.support import exact_math
from hpcagent_bench.support.distributions.perturbation import Perturbation

__all__ = ["DOMINANCE", "dominant_hermitian", "polybench_spd", "polybench_lower", "reflected_spectrum"]

#: The diagonal of a dominant matrix, per row of off-diagonal entries of magnitude at most 2 (with jitter): ``3 N``
#: exceeds every row's off-diagonal sum, so the matrix is strictly diagonally dominant, hence positive definite when
#: symmetric, and LU-factorizable without pivoting.
DOMINANCE = 3.0


def polybench_lower(n: int, datatype: DTypeLike, xp: ModuleType = np) -> NDArray[Any]:
    """PolyBench's lower-triangular factor: ``(-j % n) / n + 1`` below the diagonal, 1 on it, 0 above."""
    row = xp.arange(n)[:, None]
    column = xp.arange(n)[None, :]
    values = xp.broadcast_to(((-xp.arange(n) % n) / n + 1).astype(datatype), (n, n))
    lower = xp.where(column < row, values, xp.zeros((), dtype=datatype)).astype(datatype)
    diagonal = xp.arange(n)
    lower[diagonal, diagonal] = 1.0
    return lower


def polybench_spd(n: int, datatype: DTypeLike, draw: Perturbation, xp: ModuleType = np) -> NDArray[Any]:
    """A symmetric positive-definite matrix from PolyBench's factor ``L``: ``L + L^T`` of the jittered factor, with
    the diagonal ``DOMINANCE * n`` times the factor's (jittered) unit diagonal. Strictly diagonally dominant, so
    positive definite and factorizable without pivoting -- what PolyBench's ``L L^T`` gives, without the product."""
    lower = polybench_lower(n, datatype, xp)
    draw.jitter(lower)
    spd = lower + lower.T
    diagonal = xp.arange(n)
    spd[diagonal, diagonal] = (DOMINANCE * n) * lower[diagonal, diagonal]
    return spd.astype(datatype, copy=False)


def dominant_hermitian(entries: NDArray[Any], xp: ModuleType = np) -> NDArray[Any]:
    """``(P + P^H) / 2`` plus ``(sqrt(2) n + 1) I`` for a square ``P`` of entries of magnitude at most ``sqrt(2)``
    (real and imaginary parts in [0, 1)): Hermitian (symmetric when real) and strictly diagonally dominant, hence
    positive definite."""
    n = entries.shape[0]
    hermitian = (entries + entries.conj().T) * 0.5
    diagonal = xp.arange(n)
    hermitian[diagonal, diagonal] = hermitian[diagonal, diagonal] + (np.sqrt(2.0) * n + 1.0)
    return hermitian


def reflected_spectrum(u: NDArray[Any], singular: NDArray[Any], v: NDArray[Any], xp: ModuleType = np) -> NDArray[Any]:
    """``H_u S H_v`` for ``S = [diag(singular); 0]`` (``m x n``, ``m = len(u) >= n = len(singular) = len(v)``) and the
    Householder reflections ``H = I - 2 w w^T / (w^T w)``: an ``m x n`` matrix whose singular values are exactly
    ``singular`` (reflections are orthogonal), expanded into outer products so no matrix product is formed:
    ``S - 2 u (u^T S) - 2 (S v) v^T + 4 (u^T S v) u v^T``."""
    rows, columns = u.shape[0], singular.shape[0]
    u = u / exact_math.norm(u, xp=xp)
    v = v / exact_math.norm(v, xp=xp)
    u_s = u[:columns] * singular  # u^T S
    s_v = xp.zeros(rows, dtype=xp.float64)
    s_v[:columns] = singular * v  # S v
    cross = exact_math.sum(u_s * v, xp=xp)  # u^T S v
    matrix = xp.zeros((rows, columns), dtype=xp.float64)
    diagonal = xp.arange(columns)
    matrix[diagonal, diagonal] = singular
    matrix = matrix - 2.0 * (u[:, None] * u_s[None, :])
    matrix = matrix - 2.0 * (s_v[:, None] * v[None, :])
    return matrix + (4.0 * cross) * (u[:, None] * v[None, :])
