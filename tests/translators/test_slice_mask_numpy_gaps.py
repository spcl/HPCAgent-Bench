# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""NumPy-op gaps surfaced auditing the NumpyToC surface (PR#5 follow-up).

Four families, each validated numerically vs numpy across C / C++ / Fortran:

* ``a[::-1]`` reverse (and general strided) slices -- dropping the step turns a
  reverse copy into a plain forward copy;
* boolean-mask REDUCTIONS ``np.sum/mean/max/min(a[m])`` inline, not only the
  two-statement ``t = a[m]; np.sum(t)`` form (otherwise the masked select
  materialises as a wrong integer gather ``a[m[i]]``);
* ``np.any`` / ``np.all`` / ``np.count_nonzero`` -- the bool-as-int accumulator
  (``acc + (x != 0)``) is invalid Fortran arithmetic, so it is an if-guarded
  integer accumulator, and their non-Name arg hoists like ``sum``;
* ``~mask`` (bitwise-not on a boolean array) -- numpy logical negation, not integer
  bitwise NOT (``~1`` -> the truthy ``-2`` in C, an ``i`` argument error in Fortran).
"""

import numpy as np
import pytest

from tests.translators.op_oracle import all_ok, run_op

# Full backend matrix: native c/c++/fortran + numba. A backend
# that does not lower a pattern reports ``skip`` (pythran declines the fancy
# slice/mask forms) and is accepted; only a real FAIL fails the test.
ALL = ("c", "cpp", "fortran", "numba")


def run_(
    body: str,
    ins: dict[str, np.ndarray | float],
    outs: dict[str, tuple[int, ...]],
    syms: dict[str, int],
    shapes: dict[str, str],
) -> tuple[bool, dict[str, str]]:
    src = "import numpy as np\ndef f(" + ", ".join(list(ins) + list(outs)) + "):\n" + body + "\n"
    return all_ok(run_op(src, "f", ins, outs, syms, shapes=shapes, backends=ALL))


A = np.linspace(0.5, 3.0, 6)

S1 = {"N": 6}
VEC6 = {"a": "(N,)", "out": "(N,)"}
SCALAR_OUT = {"a": "(N,)", "out": "(1,)"}
MASKED_MAX_A = np.array([10.0, 1.0, 2.0, 3.0])
ROWS_2X3 = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
S = np.linspace(-1.0, 3.0, 6)  # ~ (bitwise-not on a boolean mask = logical negation) inputs
NOT_BODY = " m1 = a > 0.0\n m2 = a < 2.0\n out[:] = np.where(~({}), a, 0.0)"


@pytest.mark.parametrize(
    ("body", "ins", "outs", "syms", "shapes"),
    [
        # reverse / strided slices
        (" out[:] = a[::-1]", {"a": A}, {"out": (6,)}, S1, VEC6),
        (
            " b = a[::-2]\n for i in range(3):\n  out[i] = b[i]",
            {"a": A},
            {"out": (3,)},
            S1,
            {"a": "(N,)", "out": "(3,)"},
        ),
        # boolean-mask reductions
        (" m = a > 1.5\n out[0] = np.sum(a[m])", {"a": A}, {"out": (1,)}, S1, SCALAR_OUT),
        # Index 0 is masked OUT and holds the global max; the accumulator must seed
        # from the first masked HIT, not ``arr[0]`` (correct masked-max = 3.0).
        (" m = a < 5.0\n out[0] = np.max(a[m])", {"a": MASKED_MAX_A}, {"out": (1,)}, {"N": 4}, SCALAR_OUT),
        # ``np.sum(a[idx])`` with an INTEGER index array is a gather-sum, NOT a masked reduction --
        # it must stay a gather (the mask peephole is gated on a known-boolean index).
        (
            " out[0] = np.sum(a[idx])",
            {"a": A, "idx": np.array([0, 2, 4])},
            {"out": (1,)},
            S1,
            {"a": "(N,)", "idx": "(3,)", "out": "(1,)"},
        ),
        # any / all / count_nonzero
        (" m = a > 1.5\n out[0] = np.any(m)", {"a": A}, {"out": (1,)}, S1, SCALAR_OUT),
        (" m = a > 0.0\n out[0] = np.all(m)", {"a": A}, {"out": (1,)}, S1, SCALAR_OUT),
        (" m = a > 1.5\n out[0] = np.count_nonzero(m)", {"a": A}, {"out": (1,)}, S1, SCALAR_OUT),
        (
            " out[0] = np.count_nonzero(a)",
            {"a": np.array([0.0, 1.5, 0.0, 2.0, 3.0, 0.0])},
            {"out": (1,)},
            S1,
            SCALAR_OUT,
        ),
        (
            " m = a > 1.0\n out[:] = np.any(m, axis=0)",
            {"a": ROWS_2X3},
            {"out": (3,)},
            {"M": 2, "N": 3},
            {"a": "(M, N)", "out": "(N,)"},
        ),
        (
            " m = a >= 0.0\n out[:] = np.all(m, axis=1)",
            {"a": ROWS_2X3},
            {"out": (2,)},
            {"M": 2, "N": 3},
            {"a": "(M, N)", "out": "(M,)"},
        ),
        (" m = a > 1.0\n out[:] = np.where(~m, a, 0.0)", {"a": S}, {"out": (6,)}, S1, VEC6),
        # ``~`` applied to a ``& | ^`` COMBINE (De Morgan), not just a bare Name -- the operand is a
        # BinOp, so the bool-detection must recurse. C bitwise ``~`` on the 0/1 combine would give the
        # truthy -2 (always-true where); Fortran NOT() on a logical is a compile error.
        (NOT_BODY.format("m1 & m2"), {"a": S}, {"out": (6,)}, S1, VEC6),
        (NOT_BODY.format("m1 | m2"), {"a": S}, {"out": (6,)}, S1, VEC6),
        (NOT_BODY.format("m1 ^ m2"), {"a": S}, {"out": (6,)}, S1, VEC6),
    ],
    ids=[
        "reverse_whole_array_assign",
        "strided_reverse_step2",
        "masked_sum_inline",
        "masked_max_seed_excludes_index0",
        "integer_gather_reduction_is_not_masked",
        "any_over_named_mask",
        "all_over_named_mask",
        "count_nonzero_of_mask",
        "count_nonzero_of_float",
        "any_along_axis",
        "all_along_axis",
        "not_mask_in_where",
        "not_over_and_combine",
        "not_over_or_combine",
        "not_over_xor_combine",
    ],
)
def test_slice_and_mask_forms_match_numpy_on_every_backend(
    body: str,
    ins: dict[str, np.ndarray | float],
    outs: dict[str, tuple[int, ...]],
    syms: dict[str, int],
    shapes: dict[str, str],
) -> None:
    ok, res = run_(body, ins, outs, syms, shapes)
    assert ok, res


# masked reductions as an EXPLICIT LOOP
#
# The vectorised ``np.sum(a[m])`` above is the source form; a DaCe loop-nest
# extractor lowers the SAME masked reduction to an explicit
# ``for i: v = a[i]; if v > K: acc += v`` -- a per-element STAGED READ into a
# value scalar ``v`` that is REASSIGNED every iteration, plus a guarded
# accumulate. That scalar is BOTH a by-value dummy argument AND written in the
# body, which Fortran forbids on an ``intent(in)`` dummy -- so before the
# intent(in) relaxation this whole family failed to COMPILE for Fortran (C / C++
# have no intent, so they were unaffected). These lock the loop form across
# every backend. ``K``/``v`` are float value scalars (never int-truncated).

M = np.linspace(-1.0, 2.0, 8)


@pytest.mark.parametrize(
    "body",
    [
        # The nest-extracted masked SUM: ``v`` is a reassigned value scalar (Fortran must drop its ``intent(in)``).
        pytest.param(
            " acc = 0.0\n for i in range(len(a)):\n  v = a[i]\n  if v > K:\n   acc = acc + v\n out[0] = acc", id="sum"
        ),
        # Masked MAX: seed from a sentinel, keep the largest ``v > K``.
        pytest.param(
            " m = -1.0e30\n for i in range(len(a)):\n  v = a[i]\n  if v > K:\n   if v > m:\n    m = v\n out[0] = m",
            id="max",
        ),
        # Masked COUNT: integer accumulator over ``v > K``.
        pytest.param(
            " c = 0\n for i in range(len(a)):\n  v = a[i]\n  if v > K:\n   c = c + 1\n out[0] = c", id="count"
        ),
    ],
)
def test_masked_explicit_loop_staged_scalar(body: str) -> None:
    ok, res = run_(body, {"a": M, "K": 0.5, "v": 0.0}, {"out": (1,)}, {"N": 8}, {"a": "(N,)", "out": "(1,)"})
    assert ok, res
