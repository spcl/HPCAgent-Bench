"""Fancy index-array scatter / gather with a runtime-materialised index row.

The QE ultrasoft real-space augmentation (``vexx_all_paths`` -> ``_addusxx_r`` /
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

from _op_oracle import run_op

_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")


def _all_ok(res):
    return all(v == "ok" or v.startswith("skip") for v in res.values()), res


def _boxes(nat, K, N, seed):
    rng = np.random.default_rng(seed)
    return np.stack([np.sort(rng.choice(N, size=K, replace=False)).astype(np.int64) for _ in range(nat)])


def test_scatter_add_materialised_box():
    # ``box = mbox[ia]`` (row -> int index array), then ``out[box] += col * s``:
    # a fancy scatter-ACCUMULATE at the box points (mirrors _addusxx_r on rhoc).
    src = ("import numpy as np\n"
           "def scatter_add(mbox, qr, coef, out):\n"
           " nat = mbox.shape[0]\n"
           " nh = qr.shape[2]\n"
           " for ia in range(nat):\n"
           "  box = mbox[ia]\n"
           "  if box.size == 0:\n"
           "   continue\n"
           "  for ih in range(nh):\n"
           "   col = qr[ia, :, ih]\n"
           "   out[box] += col * coef[ia, ih]\n")
    nat, K, N, nh = 2, 4, 16, 3
    rng = np.random.default_rng(1)
    mbox = _boxes(nat, K, N, 1)
    qr = rng.standard_normal((nat, K, nh))
    coef = rng.standard_normal((nat, nh))
    ok, res = _all_ok(
        run_op(src,
               "scatter_add", {
                   "mbox": mbox,
                   "qr": qr,
                   "coef": coef
               }, {"out": (N, )}, {
                   "nat": nat,
                   "K": K,
                   "N": N,
                   "nh": nh
               },
               shapes={
                   "mbox": "(nat,K)",
                   "qr": "(nat,K,nh)",
                   "coef": "(nat,nh)",
                   "out": "(N,)"
               },
               backends=_ALL))
    assert ok, res


def test_scatter_add_duplicate_indices_is_buffered():
    # numpy fancy ``A[idx] += rhs`` is BUFFERED, not accumulating: a repeated index
    # is written once (last-write-wins), NOT summed. The lowering snapshots the old
    # gathered values then stores, so duplicate indices match numpy bit-exact.
    # (pythran's fancy augmented-assignment ACCUMULATES duplicates -- a pythran
    # divergence from numpy, unrelated to the native lowering -- so it is skipped.)
    src = ("import numpy as np\n"
           "def scatter(idx, rhs, A):\n"
           " A[idx] += rhs\n")
    idx = np.array([0, 0, 1, 3, 3, 3], dtype=np.int64)
    rhs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ok, res = _all_ok(
        run_op(src,
               "scatter", {
                   "idx": idx,
                   "rhs": rhs
               }, {"A": (5, )}, {
                   "nk": 6,
                   "nn": 5
               },
               shapes={
                   "idx": "(nk,)",
                   "rhs": "(nk,)",
                   "A": "(nn,)"
               },
               backends=_ALL,
               skip_backends={"pythran": "fancy-augassign-accumulates"}))
    assert ok, res


def test_gather_dot_materialised_box():
    # ``vc[box]`` gather (index array) reduced against a column via ``np.dot``
    # (mirrors _newdxx_r: aux = np.dot(qr, vc[box])).
    src = ("import numpy as np\n"
           "def gather_dot(mbox, qr, vc, out):\n"
           " nat = mbox.shape[0]\n"
           " nh = qr.shape[2]\n"
           " for ia in range(nat):\n"
           "  box = mbox[ia]\n"
           "  for ih in range(nh):\n"
           "   col = qr[ia, :, ih]\n"
           "   out[ia, ih] += np.dot(col, vc[box])\n")
    nat, K, N, nh = 2, 4, 16, 3
    rng = np.random.default_rng(2)
    mbox = _boxes(nat, K, N, 2)
    qr = rng.standard_normal((nat, K, nh))
    vc = rng.standard_normal((N, ))
    ok, res = _all_ok(
        run_op(src,
               "gather_dot", {
                   "mbox": mbox,
                   "qr": qr,
                   "vc": vc
               }, {"out": (nat, nh)}, {
                   "nat": nat,
                   "K": K,
                   "N": N,
                   "nh": nh
               },
               shapes={
                   "mbox": "(nat,K)",
                   "qr": "(nat,K,nh)",
                   "vc": "(N,)",
                   "out": "(nat,nh)"
               },
               backends=_ALL))
    assert ok, res


def test_roll_axis1():
    # ``np.roll(A, -1, axis=1)`` circular shift on a bare array.
    src = ("import numpy as np\n"
           "def roll_axis1(A, out):\n"
           " out[:, :] = np.roll(A, -1, axis=1)\n")
    R, C = 3, 5
    A = np.arange(R * C, dtype=np.float64).reshape(R, C)
    ok, res = _all_ok(
        run_op(src,
               "roll_axis1", {"A": A}, {"out": (R, C)}, {
                   "R": R,
                   "C": C
               },
               shapes={
                   "A": "(R,C)",
                   "out": "(R,C)"
               },
               backends=_ALL))
    assert ok, res


def test_roll_sliced_operand():
    # ``np.roll(A[:, :, k], ...)`` -- a SLICED operand (2-D view of a 3-D array).
    src = ("import numpy as np\n"
           "def roll_slice(buf, out):\n"
           " out[:, :] = np.roll(buf[:, :, 0], -1, axis=1)\n")
    R, C, D = 3, 5, 2
    buf = np.arange(R * C * D, dtype=np.float64).reshape(R, C, D)
    ok, res = _all_ok(
        run_op(src,
               "roll_slice", {"buf": buf}, {"out": (R, C)}, {
                   "R": R,
                   "C": C,
                   "D": D
               },
               shapes={
                   "buf": "(R,C,D)",
                   "out": "(R,C)"
               },
               backends=_ALL))
    assert ok, res


def test_roll_sliced_self_assign():
    # ``X[:, :, k] = np.roll(X[:, :, k], -1, axis=1)`` -- the QE vexx negrp>1
    # band-group shift: sliced operand AND target, in place. The decompose reads a
    # snapshot so the in-place write is safe.
    src = ("import numpy as np\n"
           "def roll_self(buf):\n"
           " buf[:, :, 0] = np.roll(buf[:, :, 0], -1, axis=1)\n")
    R, C, D = 3, 5, 1
    buf = np.arange(R * C * D, dtype=np.float64).reshape(R, C, D).copy()
    ok, res = _all_ok(
        run_op(src,
               "roll_self", {}, {"buf": (R, C, D)}, {
                   "R": R,
                   "C": C,
                   "D": D
               },
               shapes={"buf": "(R,C,D)"},
               backends=_ALL))
    assert ok, res


# --------------------------------------------------------------------------- #
# Chained gather: associativity does NOT hold for an advanced index            #
# --------------------------------------------------------------------------- #


def test_chained_gather_is_not_flattened_into_one_subscript():
    """``A[idx][j] == A[idx[j]]``, NOT ``A[idx, j]``.

    Flattening it produced a subscript with more indices than the base has axes, and the
    scalarizer then handed the outer iterators to the wrong axes.
    """
    import ast as _ast

    from numpyto_common.lowering import _ChainedSubscriptFlattener

    tree = _ast.parse("y = x[aj][:, None, :, :]")
    _ChainedSubscriptFlattener({"x": ("n", "3"), "aj": ("p", "j")}).visit(tree)
    assert _ast.unparse(tree).strip() == "y = x[aj][:, None, :, :]"


def test_chained_scalar_index_is_still_flattened():
    """A genuinely scalar inner index keeps the existing collapse: ``psi[f][..., 0]``."""
    import ast as _ast

    from numpyto_common.lowering import _ChainedSubscriptFlattener

    tree = _ast.parse("y = psi[f][..., 0]")
    _ChainedSubscriptFlattener({"psi": ("F", "X", "Y", "K")}).visit(tree)
    assert _ast.unparse(tree).strip() == "y = psi[f, ..., 0]"


# --------------------------------------------------------------------------- #
# Broadcast gather: several advanced indices in ONE subscript numpy-broadcast  #
# instead of summing their ranks (icon_gather's A[idx, lev, blk] regression)   #
# --------------------------------------------------------------------------- #


def test_gather_two_broadcast_arrays_plus_scalar_axis():
    """``t = A[r, c, 0]``: r (nrow,1) and c (1,ncol) BROADCAST to (nrow,ncol);
    the trailing literal ``0`` is a scalar axis that numpy still counts as
    "advanced" (so it stays adjacent to r/c) but adds neither its own axis
    nor rank -- the icon_gather ``A[idx, lev, 0]`` shape in miniature."""
    src = ("import numpy as np\n"
           "def gather3(A, r, c, out):\n"
           " nrow, ncol, nblk = A.shape\n"
           " t = np.zeros((nrow, ncol), A.dtype)\n"
           " t = A[r, c, 0]\n"
           " out[:, :] = t\n")
    nrow, ncol, nblk = 5, 4, 3
    rng = np.random.default_rng(0)
    A = rng.standard_normal((nrow, ncol, nblk))
    r = rng.integers(0, nrow, size=(nrow, 1)).astype(np.int64)
    c = rng.integers(0, ncol, size=(1, ncol)).astype(np.int64)
    ok, res = _all_ok(
        run_op(src,
               "gather3", {
                   "A": A,
                   "r": r,
                   "c": c
               }, {"out": (nrow, ncol)}, {
                   "nrow": nrow,
                   "ncol": ncol,
                   "nblk": nblk
               },
               shapes={
                   "A": "(nrow,ncol,nblk)",
                   "r": "(nrow,1)",
                   "c": "(1,ncol)",
                   "out": "(nrow,ncol)"
               },
               backends=("c", "cpp", "fortran")))
    assert ok, res


def test_gather_index_array_mixed_with_real_slice():
    """``t = A[r, :]``: a rank-1 index array on axis 0 next to a real ``:``
    slice on axis 1 -- the slice keeps its own axis, the array replaces its
    own axis with its own rank (a single-array adjacent broadcast group)."""
    src = ("import numpy as np\n"
           "def gather_slice(A, r, out):\n"
           " nrow, ncol = A.shape\n"
           " t = np.zeros((nrow, ncol), A.dtype)\n"
           " t = A[r, :]\n"
           " out[:, :] = t\n")
    nrow, ncol = 5, 4
    rng = np.random.default_rng(1)
    A = rng.standard_normal((nrow, ncol))
    r = rng.integers(0, nrow, size=(nrow, )).astype(np.int64)
    ok, res = _all_ok(
        run_op(src,
               "gather_slice", {
                   "A": A,
                   "r": r
               }, {"out": (nrow, ncol)}, {
                   "nrow": nrow,
                   "ncol": ncol
               },
               shapes={
                   "A": "(nrow,ncol)",
                   "r": "(nrow,)",
                   "out": "(nrow,ncol)"
               },
               backends=("c", "cpp", "fortran")))
    assert ok, res


def test_gather_broadcast_axis_extent_one_reads_index_zero():
    """``t = A[r, c]``: r (nrow,1) and c (1,ncol) broadcast to (nrow,ncol) --
    each operand's OWN size-1 axis must read index 0, not the (larger) shared
    iter, or it runs off its length-1 allocation."""
    src = ("import numpy as np\n"
           "def gather_bcast(A, r, c, out):\n"
           " nrow, ncol = A.shape\n"
           " t = np.zeros((nrow, ncol), A.dtype)\n"
           " t = A[r, c]\n"
           " out[:, :] = t\n")
    nrow, ncol = 5, 4
    rng = np.random.default_rng(2)
    A = rng.standard_normal((nrow, ncol))
    r = rng.integers(0, nrow, size=(nrow, 1)).astype(np.int64)
    c = rng.integers(0, ncol, size=(1, ncol)).astype(np.int64)
    ok, res = _all_ok(
        run_op(src,
               "gather_bcast", {
                   "A": A,
                   "r": r,
                   "c": c
               }, {"out": (nrow, ncol)}, {
                   "nrow": nrow,
                   "ncol": ncol
               },
               shapes={
                   "A": "(nrow,ncol)",
                   "r": "(nrow,1)",
                   "c": "(1,ncol)",
                   "out": "(nrow,ncol)"
               },
               backends=("c", "cpp", "fortran")))
    assert ok, res


def test_gather_front_placed_broadcast_separated_by_real_slice():
    """``t = A[blk[:, :, 0], :, idx[:, :, 0]]``: two COMPOUND array-valued
    advanced indices with a real slice sitting BETWEEN them in the source
    text. numpy moves their broadcast result to the FRONT -- result shape
    (NB, NPROMA, NLEV) -- instead of the in-place layout the C emit used to
    read the index arrays at (a segfaulting off-by-one-axis misalignment).
    zekin_gather's ``z_kin_hor_e[edge_blk[:, :, e], :, edge_idx[:, :, e]]``
    in miniature."""
    src = ("import numpy as np\n"
           "def gather_front(A, blk, idx, out):\n"
           " NPROMA, NLEV, NB = A.shape\n"
           " t = np.zeros((NB, NPROMA, NLEV), A.dtype)\n"
           " t = A[blk[:, :, 0], :, idx[:, :, 0]]\n"
           " out[:, :, :] = t\n")
    NPROMA, NLEV, NB = 4, 3, 5
    rng = np.random.default_rng(3)
    A = rng.standard_normal((NPROMA, NLEV, NB))
    blk = rng.integers(0, NPROMA, size=(NB, NPROMA, 1)).astype(np.int64)
    idx = rng.integers(0, NB, size=(NB, NPROMA, 1)).astype(np.int64)
    ok, res = _all_ok(
        run_op(src,
               "gather_front", {
                   "A": A,
                   "blk": blk,
                   "idx": idx
               }, {"out": (NB, NPROMA, NLEV)}, {
                   "NPROMA": NPROMA,
                   "NLEV": NLEV,
                   "NB": NB
               },
               shapes={
                   "A": "(NPROMA,NLEV,NB)",
                   "blk": "(NB,NPROMA,1)",
                   "idx": "(NB,NPROMA,1)",
                   "out": "(NB,NPROMA,NLEV)"
               },
               backends=("c", "cpp", "fortran")))
    assert ok, res
