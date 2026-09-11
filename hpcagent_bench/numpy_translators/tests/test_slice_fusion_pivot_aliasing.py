"""A fused slice assignment reads its own array's invariant element BEFORE the nest.

``SliceFusion`` stores one element per iteration, so ``A[k, k:] = A[k, k:] / A[k, k]``
used to scalarise into a loop whose ``si1 == k`` iteration overwrites the pivot and whose
every later iteration then divides by the ``1.0`` it just stored. numpy evaluates the whole
RHS against the pre-assignment array, so all three native backends disagreed with it by
1.86e+00 on a 6x6 -- silently, since nothing in the emitted code looks wrong. This is the
Gauss-elimination shape (``row = row - factor * pivot_row`` with the pivot IN the row being
written), and the pivot is what has to be read once, ahead of the nest.

Staging is decided structurally -- a full-rank, Slice-free, index-array-free read of the
array being written is loop-invariant by construction -- so a NON-aliasing kernel keeps its
numbers exactly: cholesky / lu read ``A[k, k]`` while writing rows ``k + 1:``, and the
staged temp holds the same value every iteration would have loaded. What changes there is
one load instead of a trip count's worth.

The numeric half alone would pass for the wrong reason (any hoist, correct or not, fixes
the numbers on a 6x6), so the structural half pins WHERE the read lands.
"""

import ast

import numpy as np
from _op_oracle import run_op

from numpyto_common.lowering import INVARIANT_SELF_READ_PREFIX, SliceFusion

_NATIVE = ("c", "cpp", "fortran")


def _fuse(src: str, shapes: dict) -> str:
    tree = SliceFusion(shapes).visit(ast.parse(src))
    return ast.unparse(ast.fix_missing_locations(tree))


def _assert_ok(res: dict) -> None:
    for backend, status in res.items():
        assert status == "ok" or status.startswith("skip"), f"{backend}: {status}"
    assert any(status == "ok" for status in res.values()), f"all skipped (vacuous): {res}"


# ---- structural: the invariant read is staged ahead of the loop ---- #


def test_pivot_read_is_staged_before_the_fused_nest():
    assert _fuse("A[k, k:] = A[k, k:] / A[k, k]", {"A": ["N", "N"]}) == (
        f"{INVARIANT_SELF_READ_PREFIX}1 = A[k, k]\n"
        "for si1 in range(k, N):\n"
        f"    A[k, si1] = A[k, si1 + (k - k)] / {INVARIANT_SELF_READ_PREFIX}1"
    )


def test_staged_pivot_leaves_no_read_of_the_written_array_in_the_body():
    fused = _fuse("x[:] = x[:] - x[0]", {"x": ["N"]})
    body = fused.split("\n", 1)[1]
    assert "x[0]" not in body, f"the invariant read survived inside the nest:\n{fused}"


def test_the_same_invariant_read_is_staged_once():
    fused = _fuse("A[i, :] = (A[i, j] + A[i, :]) * A[i, j]", {"A": ["N", "N"]})
    assert fused.count(f"{INVARIANT_SELF_READ_PREFIX}1 = A[i, j]") == 1
    assert fused.count(f"{INVARIANT_SELF_READ_PREFIX}2") == 0, f"a repeated read got its own temp:\n{fused}"


def test_an_iterated_read_of_the_written_array_is_not_staged():
    # ``A[k, k:]`` moves with the iter var, so it is not invariant and must stay in the body.
    fused = _fuse("A[k, k:] = A[k, k:] * 2.0", {"A": ["N", "N"]})
    assert INVARIANT_SELF_READ_PREFIX not in fused, f"an iterated read was staged:\n{fused}"


def test_a_read_of_another_array_is_left_alone():
    # gaussian's elimination step: the pivot row belongs to the SAME array but is read at a
    # moving column, and ``mult`` is a different array. Byte for byte the old lowering.
    assert _fuse("A[k + 1:, k:] -= mult[:, None] * A[k, k:]", {"A": ["N", "N"], "mult": ["N"]}) == (
        "for si0 in range(k + 1, N):\n"
        "    for si1 in range(k, N):\n"
        "        A[si0, si1] -= mult[si0 + (0 - (k + 1))] * A[k, si1 + (k - k)]"
    )


def test_a_gather_index_is_not_mistaken_for_an_invariant_element():
    # ``A[k, idx]`` is a gather whose result is a whole row, not one element.
    fused = _fuse("A[k, :] = A[k, idx]", {"A": ["N", "N"], "idx": ["N"]})
    assert INVARIANT_SELF_READ_PREFIX not in fused, f"an advanced index was staged as a scalar:\n{fused}"


def test_a_guarded_read_keeps_its_guard():
    # The test exists to keep the element from being addressed; hoisting past it would load
    # where numpy never does.
    fused = _fuse("A[k, :] = A[k, k] if k < N else 0.0", {"A": ["N", "N"]})
    assert INVARIANT_SELF_READ_PREFIX not in fused, f"a guarded read was hoisted out of its guard:\n{fused}"


# ---- numerical: the aliasing kernel agrees with numpy on every native backend ---- #


def test_pivot_scaling_matches_numpy():
    src = (
        "import numpy as np\n"
        "def f(S, N, A):\n"
        "    A[:, :] = S[:, :]\n"
        "    for k in range(N):\n"
        "        A[k, k:] = A[k, k:] / A[k, k]\n"
    )
    n = 6
    s = np.random.default_rng(0).random((n, n)) + 2.0
    _assert_ok(
        run_op(
            src,
            "f",
            {"S": s, "N": n},
            {"A": (n, n)},
            {"N": n},
            shapes={"S": "(N, N)", "A": "(N, N)"},
            backends=_NATIVE,
        )
    )


def test_gauss_jordan_elimination_matches_numpy():
    # The full shape the defect is named for: normalise the pivot row in place, then
    # eliminate every other row against it.
    src = (
        "import numpy as np\n"
        "def f(S, N, A):\n"
        "    A[:, :] = S[:, :]\n"
        "    for k in range(N):\n"
        "        A[k, :] = A[k, :] / A[k, k]\n"
        "        for i in range(N):\n"
        "            if i != k:\n"
        "                A[i, :] = A[i, :] - A[i, k] * A[k, :]\n"
    )
    n = 5
    s = np.random.default_rng(3).random((n, n)) + np.eye(n) * float(n)
    _assert_ok(
        run_op(
            src,
            "f",
            {"S": s, "N": n},
            {"A": (n, n)},
            {"N": n},
            shapes={"S": "(N, N)", "A": "(N, N)"},
            backends=_NATIVE,
            rtol=1e-11,
            atol=1e-11,
        )
    )
