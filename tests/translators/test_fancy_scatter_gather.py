# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fancy index-array scatter / gather with a runtime-materialised index row.

The QE ultrasoft real-space augmentation (``vexx_k`` -> ``_addusxx_r`` /
``_newdxx_r``) reads a per-atom box of grid points out of an integer table and
then scatters / gathers a length-``box`` vector through it:

    box = tabxx_box[ia]           # (nat, K) int table -> (K,) index array
    if box.size == 0: continue
    rhoc[box] += qr[:, ijh] * s   # SCATTER-ADD at the box points
    aux = np.dot(qr, vc[box])     # GATHER the box points, reduce

The C/Fortran emitter has no array-valued-subscript notion, so ``box`` must be
materialised into a rank-1 local (int / float dtype preserved) and the
scatter/gather lowered to per-element loops. numpy accumulates duplicate scatter
indices, so ``A[idx] += rhs`` is the faithful loop form (not a vector store).

The ``np.roll`` case mirrors the ``negrp > 1`` band-group circular shift.
"""

import numpy as np

from tests.translators.op_oracle import all_ok, run_op

ALL = ("c", "cpp", "fortran", "numba")


def test_scatter_add_duplicate_indices_is_buffered() -> None:
    # numpy fancy ``A[idx] += rhs`` is BUFFERED, not accumulating: a repeated index
    # is written once (last-write-wins), NOT summed. The lowering snapshots the old
    # gathered values then stores, so duplicate indices match numpy bit-exact.
    # (pythran's fancy augmented-assignment ACCUMULATES duplicates -- a pythran
    # divergence from numpy, unrelated to the native lowering -- so it is skipped.)
    src = "import numpy as np\ndef scatter(idx, rhs, A):\n A[idx] += rhs\n"
    idx = np.array([0, 0, 1, 3, 3, 3], dtype=np.int64)
    rhs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ok, res = all_ok(
        run_op(
            src,
            "scatter",
            {"idx": idx, "rhs": rhs},
            {"A": (5,)},
            {"nk": 6, "nn": 5},
            shapes={"idx": "(nk,)", "rhs": "(nk,)", "A": "(nn,)"},
            backends=ALL,
        )
    )
    assert ok, res


def test_roll_sliced_self_assign() -> None:
    # ``X[:, :, k] = np.roll(X[:, :, k], -1, axis=1)`` -- the QE vexx negrp>1
    # band-group shift: sliced operand AND target, in place. The decompose reads a
    # snapshot so the in-place write is safe.
    src = "import numpy as np\ndef roll_self(buf):\n buf[:, :, 0] = np.roll(buf[:, :, 0], -1, axis=1)\n"
    R, C, D = 3, 5, 1
    ok, res = all_ok(
        run_op(
            src, "roll_self", {}, {"buf": (R, C, D)}, {"R": R, "C": C, "D": D}, shapes={"buf": "(R,C,D)"}, backends=ALL
        )
    )
    assert ok, res


# Chained gather: associativity does NOT hold for an advanced index            #


def test_chained_gather_composes_the_outer_index_into_the_index_array() -> None:
    """``A[idx][j] == A[idx[j]]``, NOT ``A[idx, j]``.

    Flattening positionally produced a subscript with more indices than the base has axes, and the
    scalarizer then handed the outer iterators to the wrong axes. An outer entry that lands on a
    gathered axis indexes the index array instead, a newaxis between two of them included.
    """
    import ast as ast_

    from hpcagent_bench.translators.numpyto_common.lowering import ChainedSubscriptFlattener

    x = np.arange(12.0).reshape(4, 3)
    aj = np.array([[2, 0], [1, 3], [3, 3]])
    assert np.array_equal(x[aj][:, None, :, :], x[aj[:, None], :])

    tree = ast_.parse("y = x[aj][:, None, :, :]")
    ChainedSubscriptFlattener({"x": ("n", "3"), "aj": ("p", "j")}).visit(tree)
    assert ast_.unparse(tree).strip() == "y = x[aj[:, None], :]"


# Broadcast gather: several advanced indices in ONE subscript numpy-broadcast  #
# instead of summing their ranks (icon_gather's A[idx, lev, blk] regression)   #


def test_gather_two_broadcast_arrays_plus_scalar_axis() -> None:
    """``t = A[r, c, 0]``: r (nrow,1) and c (1,ncol) BROADCAST to (nrow,ncol);
    the trailing literal ``0`` is a scalar axis that numpy still counts as
    "advanced" (so it stays adjacent to r/c) but adds neither its own axis
    nor rank -- the icon_gather ``A[idx, lev, 0]`` shape in miniature."""
    src = (
        "import numpy as np\n"
        "def gather3(A, r, c, out):\n"
        " nrow, ncol, nblk = A.shape\n"
        " t = np.zeros((nrow, ncol), A.dtype)\n"
        " t = A[r, c, 0]\n"
        " out[:, :] = t\n"
    )
    nrow, ncol, nblk = 5, 4, 3
    rng = np.random.default_rng(0)
    A = rng.standard_normal((nrow, ncol, nblk))
    r = rng.integers(0, nrow, size=(nrow, 1)).astype(np.int64)
    c = rng.integers(0, ncol, size=(1, ncol)).astype(np.int64)
    ok, res = all_ok(
        run_op(
            src,
            "gather3",
            {"A": A, "r": r, "c": c},
            {"out": (nrow, ncol)},
            {"nrow": nrow, "ncol": ncol, "nblk": nblk},
            shapes={"A": "(nrow,ncol,nblk)", "r": "(nrow,1)", "c": "(1,ncol)", "out": "(nrow,ncol)"},
            backends=("c", "cpp", "fortran"),
        )
    )
    assert ok, res


def test_gather_broadcast_axis_extent_one_reads_index_zero() -> None:
    """``t = A[r, c]``: r (nrow,1) and c (1,ncol) broadcast to (nrow,ncol) --
    each operand's OWN size-1 axis must read index 0, not the (larger) shared
    iter, or it runs off its length-1 allocation."""
    src = (
        "import numpy as np\n"
        "def gather_bcast(A, r, c, out):\n"
        " nrow, ncol = A.shape\n"
        " t = np.zeros((nrow, ncol), A.dtype)\n"
        " t = A[r, c]\n"
        " out[:, :] = t\n"
    )
    nrow, ncol = 5, 4
    rng = np.random.default_rng(2)
    A = rng.standard_normal((nrow, ncol))
    r = rng.integers(0, nrow, size=(nrow, 1)).astype(np.int64)
    c = rng.integers(0, ncol, size=(1, ncol)).astype(np.int64)
    ok, res = all_ok(
        run_op(
            src,
            "gather_bcast",
            {"A": A, "r": r, "c": c},
            {"out": (nrow, ncol)},
            {"nrow": nrow, "ncol": ncol},
            shapes={"A": "(nrow,ncol)", "r": "(nrow,1)", "c": "(1,ncol)", "out": "(nrow,ncol)"},
            backends=("c", "cpp", "fortran"),
        )
    )
    assert ok, res
