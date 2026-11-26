# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Soundness holes in the parallel-loop analysis (must err toward serial).

Three cross-iteration dependences that the predicates wrongly cleared for
``#pragma omp parallel for`` / ``nb.prange``, each a silent race -> wrong numbers:

A. a WRITTEN array indexed by the loop var on TWO different axes (``A[i,j] = A[j,i]``
   -- an in-place transpose): every ``A[i,j]`` looked idx-safe in isolation, but the
   per-iteration regions overlap.
B. a loop-carried scalar READ before it is WRITTEN in the body (``b[i] = s; s = a[i]``
   -- a lag): not self-referential and not an aug-assign, so those guards miss it.
C. a reduction accumulator whose LIVE value is captured each iteration (``s = s + a[i];
   out[i] = s`` -- a prefix scan): accepted as ``reduction(+:s)``, which hands ``out`` racy
   partial sums.

The regression half pins the legitimate patterns that must STAY parallel (an in-place
transpose of a READ-ONLY source, a write-before-read private temp, a plain reduction).
"""

import ast

import pytest

from hpcagent_bench.translators.numpyto_common.parallelism import loop_is_parallel_safe, loop_reduction


def stmt_(src: str) -> ast.stmt:
    return ast.parse(src).body[0]


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        # A: cross-axis write to a written array is a race.
        ("for i in range(N):\n    for j in range(N):\n        A[i, j] = A[j, i] + 1.0\n", False),
        # out is written only at [i, j]; a is READ-ONLY, so its transposed read cannot race.
        ("for i in range(N):\n    for j in range(N):\n        out[i, j] = a[j, i] + 1.0\n", True),
        # B: b[i] reads the PREVIOUS iteration's s (a lag) -> serial-only.
        ("for i in range(N):\n    b[i] = s\n    s = a[i]\n", False),
        # s is (re)written before it is read every iteration -> a private temp, safe to parallelize.
        ("for i in range(N):\n    s = a[i]\n    b[i] = s + 1.0\n", True),
        ("for i in range(N):\n    for j in range(M):\n        c[i] = c[i] + t\n    t = a[i]\n", False),
    ],
    ids=[
        "inplace_transpose_is_unsafe",
        "transpose_of_readonly_source_is_safe",
        "carried_scalar_read_before_write_is_unsafe",
        "write_before_read_private_temp_is_safe",
        "carried_scalar_inside_inner_loop_is_unsafe",
    ],
)
def test_parallel_safety_follows_cross_iteration_dependences(src: str, expected: bool) -> None:
    assert bool(loop_is_parallel_safe(stmt_(src))) is expected


# C: a captured accumulator is a scan, not a reduction
@pytest.mark.parametrize(
    "src",
    [
        "for i in range(N):\n    s = s + a[i]\n    out[i] = s\n",
        "for i in range(N):\n    s = s + a[i]\n    c[i] = s * 2.0\n",
    ],
    ids=["prefix_scan_capture", "accumulator_used_in_other_expr"],
)
def test_a_live_accumulator_captured_each_iteration_is_not_a_reduction(src: str) -> None:
    assert loop_reduction(stmt_(src)) is None


# D: a store through a tuple target is a store
@pytest.mark.parametrize(
    ("src", "safe"),
    [
        # dwt2d's level loop: every level rewrites the corner the next reads, but the tuple's stores were invisible and
        # the loop became ``nb.prange``.
        (
            "for lvl in range(m):\n    h = n >> lvl\n    out[:h, :h], out[h:2 * h, :h] = f(out[: 2 * h, : 2 * h])\n",
            False,
        ),
        ("for i in range(n):\n    a[i], b[i] = c[i], d[i]\n", True),
    ],
    ids=["overlapping_stores", "independent_stores"],
)
def test_overlapping_stores_through_a_tuple_target_are_not_parallel_safe(src: str, safe: bool) -> None:
    assert bool(loop_is_parallel_safe(stmt_(src))) is safe
