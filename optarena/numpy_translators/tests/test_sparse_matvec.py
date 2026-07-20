"""Numerical validation of every sparse matvec / matmul dispatcher.

Each dispatcher in ``numpyto_common.sparse_emit`` emits a numpy-style loop
nest (``for`` loops + subscripts). We exec that AST directly against
numpy/scipy buffers and compare to scipy's reference product. This
validates the dispatcher ALGORITHM independent of the C/Fortran emit
(the loop nest is the single source of truth both backends render).

Covers all 9 layouts for matvec (csr/csc/coo/dia/bcsr/bcoo/ell/jds/
sell_c_sigma) plus csr@csr->dense and csr@dense-matrix.
"""
import ast

import numpy as np
import pytest

sp = pytest.importorskip("scipy.sparse")

from numpyto_common import sparse_emit as se


def _run(stmts, scope):
    """Wrap ``stmts`` in a module, exec it against ``scope`` (a dict of
    numpy arrays + size ints), return the mutated scope."""
    mod = ast.Module(body=list(stmts), type_ignores=[])
    ast.fix_missing_locations(mod)
    code = compile(mod, "<dispatcher>", "exec")
    exec(code, {"range": range}, scope)
    return scope


def _dense(A):
    return np.asarray(A.todense()) if sp.issparse(A) else np.asarray(A)


# ---------------------------------------------------------------------------
# matvec: <format> @ dense vector -> dense vector
# ---------------------------------------------------------------------------


def _make_csr(M, N, density=0.3, seed=0):
    A = sp.random(M, N, density=density, format="csr", random_state=seed, dtype=np.float64)
    A.sort_indices()
    return A


def test_csr_matvec():
    M, N = 12, 9
    A = _make_csr(M, N, seed=1)
    x = np.random.default_rng(1).random(N)
    y = np.zeros(M)
    stmts = se.expand_matmul_csr_dense_vec(ast.Name(id="y", ctx=ast.Store()), {
        "indptr": "ip",
        "indices": "ix",
        "data": "da"
    }, "x", "M")
    _run(stmts, {"y": y, "ip": A.indptr, "ix": A.indices, "da": A.data, "x": x, "M": M})
    assert np.allclose(y, A @ x)


def test_csc_matvec():
    M, N = 10, 11
    Acsr = _make_csr(M, N, seed=2)
    A = Acsr.tocsc()
    A.sort_indices()
    x = np.random.default_rng(2).random(N)
    y = np.zeros(M)
    stmts = se.expand_matmul_csc_dense_vec(ast.Name(id="y", ctx=ast.Store()), {
        "indptr": "ip",
        "indices": "ix",
        "data": "da"
    }, "x", "M", "N")
    _run(stmts, {"y": y, "ip": A.indptr, "ix": A.indices, "da": A.data, "x": x, "M": M, "N": N})
    assert np.allclose(y, Acsr @ x)


def test_coo_matvec():
    M, N = 8, 8
    Acsr = _make_csr(M, N, seed=3)
    A = Acsr.tocoo()
    x = np.random.default_rng(3).random(N)
    y = np.zeros(M)
    stmts = se.expand_matmul_coo_dense_vec(ast.Name(id="y", ctx=ast.Store()), {
        "row": "r",
        "col": "c",
        "data": "da"
    }, "x", "M", "NNZ")
    _run(stmts, {"y": y, "r": A.row, "c": A.col, "da": A.data, "x": x, "M": M, "NNZ": A.nnz})
    assert np.allclose(y, Acsr @ x)


def test_dia_matvec():
    M = N = 10
    Acsr = _make_csr(M, N, density=0.4, seed=4)
    A = Acsr.todia()
    x = np.random.default_rng(4).random(N)
    y = np.zeros(M)
    ndiag = A.data.shape[0]
    stmts = se.expand_matmul_dia_dense_vec(ast.Name(id="y", ctx=ast.Store()), {
        "data": "da",
        "offsets": "off"
    }, "x", "M", "N", "ND")
    _run(stmts, {"y": y, "da": A.data, "off": A.offsets, "x": x, "M": M, "N": N, "ND": ndiag})
    assert np.allclose(y, Acsr @ x)


def test_bcsr_matvec():
    R = C = 2
    Mb, Nb = 4, 3  # block rows / cols
    M, N = Mb * R, Nb * C
    Acsr = _make_csr(M, N, density=0.5, seed=5)
    A = Acsr.tobsr(blocksize=(R, C))
    x = np.random.default_rng(5).random(N)
    y = np.zeros(M)
    nbr = A.indptr.shape[0] - 1
    stmts = se.expand_matmul_bcsr_dense_vec(ast.Name(id="y", ctx=ast.Store()), {
        "indptr": "ip",
        "indices": "ix",
        "data": "da"
    }, "x", "NBR", "R", "C", "M")
    _run(stmts, {"y": y, "ip": A.indptr, "ix": A.indices, "da": A.data, "x": x, "NBR": nbr, "R": R, "C": C, "M": M})
    assert np.allclose(y, Acsr @ x)


def test_bcoo_matvec():
    R = C = 2
    Mb, Nb = 4, 3  # block rows / cols
    M, N = Mb * R, Nb * C
    Acsr = _make_csr(M, N, density=0.5, seed=9)
    A = Acsr.tobsr(blocksize=(R, C))
    # block-COO: expand bsr block-row pointers into per-block row coords;
    # scipy has no bcoo, so this is the canonical bsr -> bcoo conversion.
    nbrows = A.indptr.shape[0] - 1
    brow = np.repeat(np.arange(nbrows), np.diff(A.indptr)).astype(np.int64)
    bcol = A.indices.astype(np.int64)
    data = A.data  # [n_blocks, R, C]
    nblk = data.shape[0]
    x = np.random.default_rng(9).random(N)
    y = np.zeros(M)
    stmts = se.expand_matmul_bcoo_dense_vec(ast.Name(id="y", ctx=ast.Store()), {
        "row": "r",
        "col": "c",
        "data": "da"
    }, "x", "M", "NBLK", "R", "C")
    _run(stmts, {"y": y, "r": brow, "c": bcol, "da": data, "x": x, "M": M, "NBLK": nblk, "R": R, "C": C})
    assert np.allclose(y, Acsr @ x)


def _to_ell(A):
    """Build ELLPACK (indices[M, maxnz], data[M, maxnz]) from a CSR A.
    Padding: index -1, data 0."""
    A = A.tocsr()
    A.sort_indices()
    M = A.shape[0]
    row_len = np.diff(A.indptr)
    maxnz = int(row_len.max()) if M else 0
    indices = np.full((M, maxnz), -1, dtype=np.int64)
    data = np.zeros((M, maxnz), dtype=np.float64)
    for i in range(M):
        lo, hi = A.indptr[i], A.indptr[i + 1]
        n = hi - lo
        indices[i, :n] = A.indices[lo:hi]
        data[i, :n] = A.data[lo:hi]
    return indices, data, maxnz


def test_ell_matvec():
    M, N = 9, 7
    Acsr = _make_csr(M, N, seed=6)
    indices, data, maxnz = _to_ell(Acsr)
    x = np.random.default_rng(6).random(N)
    y = np.zeros(M)
    stmts = se.expand_matmul_ell_dense_vec(ast.Name(id="y", ctx=ast.Store()), {
        "indices": "ix",
        "data": "da"
    }, "x", "M", "MAXNZ")
    _run(stmts, {"y": y, "ix": indices, "da": data, "x": x, "M": M, "MAXNZ": maxnz})
    assert np.allclose(y, Acsr @ x)


def _to_jds(A):
    """Build JDS (perm, jd_ptr, col_ind, jdiag) from a CSR A.

    Rows sorted by DESCENDING length; jagged diagonals stored
    column-major (the d-th nonzero of every row that has one).
    """
    A = A.tocsr()
    A.sort_indices()
    M = A.shape[0]
    row_len = np.diff(A.indptr)
    perm = np.argsort(-row_len, kind="stable").astype(np.int64)
    maxlen = int(row_len.max()) if M else 0
    col_ind = []
    jdiag = []
    jd_ptr = [0]
    for d in range(maxlen):
        for r in perm:
            lo = A.indptr[r]
            if d < row_len[r]:
                col_ind.append(A.indices[lo + d])
                jdiag.append(A.data[lo + d])
        jd_ptr.append(len(col_ind))
    return (perm, np.array(jd_ptr,
                           dtype=np.int64), np.array(col_ind,
                                                     dtype=np.int64), np.array(jdiag,
                                                                               dtype=np.float64), len(jd_ptr) - 1)


def test_jds_matvec():
    M, N = 10, 8
    Acsr = _make_csr(M, N, density=0.4, seed=7)
    perm, jd_ptr, col_ind, jdiag, njd = _to_jds(Acsr)
    x = np.random.default_rng(7).random(N)
    y = np.zeros(M)
    stmts = se.expand_matmul_jds_dense_vec(ast.Name(id="y", ctx=ast.Store()), {
        "perm": "perm",
        "jd_ptr": "jdp",
        "col_ind": "ci",
        "jdiag": "jd"
    }, "x", "M", "NJD")
    # jds dispatcher uses a scratch accumulator ``__jds_y_perm`` (size M).
    _run(
        stmts, {
            "y": y,
            "perm": perm,
            "jdp": jd_ptr,
            "ci": col_ind,
            "jd": jdiag,
            "x": x,
            "M": M,
            "NJD": njd,
            "__jds_y_perm": np.zeros(M)
        })
    assert np.allclose(y, Acsr @ x)


def _to_sell(A, C):
    """Build SELL-C-sigma (slice_ptr, col_idx, val, row_len, perm) with
    sigma = C (sort within each C-row slice). Column-major within slice."""
    A = A.tocsr()
    A.sort_indices()
    M = A.shape[0]
    row_len_full = np.diff(A.indptr)
    # sort rows by descending length within each slice window of size C
    perm = np.arange(M, dtype=np.int64)
    for s in range(0, M, C):
        blk = perm[s:s + C]
        order = np.argsort(-row_len_full[blk], kind="stable")
        perm[s:s + C] = blk[order]
    nslices = (M + C - 1) // C
    slice_ptr = [0]
    col_idx = []
    val = []
    row_len = np.zeros(M, dtype=np.int64)
    for gr in range(M):
        row_len[gr] = row_len_full[perm[gr]]
    for s in range(nslices):
        rows = perm[s * C:(s + 1) * C]
        w = int(row_len_full[rows].max()) if len(rows) else 0
        # column-major: for col in range(w): for r in range(C): emit
        for col in range(w):
            for r in range(C):
                gidx = s * C + r
                if gidx < M and col < row_len_full[perm[gidx]]:
                    rr = perm[gidx]
                    lo = A.indptr[rr]
                    col_idx.append(A.indices[lo + col])
                    val.append(A.data[lo + col])
                else:
                    col_idx.append(0)
                    val.append(0.0)
        slice_ptr.append(len(val))
    return (np.array(slice_ptr,
                     dtype=np.int64), np.array(col_idx,
                                               dtype=np.int64), np.array(val, dtype=np.float64), row_len, perm, nslices)


def test_sell_c_sigma_matvec():
    M, N, C = 10, 8, 4
    Acsr = _make_csr(M, N, density=0.4, seed=8)
    slice_ptr, col_idx, val, row_len, perm, nslices = _to_sell(Acsr, C)
    x = np.random.default_rng(8).random(N)
    y = np.zeros(M)
    stmts = se.expand_matmul_sell_c_sigma_dense_vec(ast.Name(id="y", ctx=ast.Store()), {
        "slice_ptr": "sp",
        "col_idx": "ci",
        "val": "v",
        "row_len": "rl",
        "perm": "perm"
    }, "x", "M", "NSL", "C")
    _run(
        stmts, {
            "y": y,
            "sp": slice_ptr,
            "ci": col_idx,
            "v": val,
            "rl": row_len,
            "perm": perm,
            "x": x,
            "M": M,
            "NSL": nslices,
            "C": C,
            "__sell_acc": 0.0
        })
    assert np.allclose(y, Acsr @ x)


# ---------------------------------------------------------------------------
# matmul: csr @ csr -> dense, csr @ dense-matrix -> dense
# ---------------------------------------------------------------------------


def test_csr_csr_dense():
    NI, NK, NJ = 7, 6, 5
    A = _make_csr(NI, NK, seed=10)
    B = _make_csr(NK, NJ, seed=11)
    M = np.zeros((NI, NJ))
    stmts = se.expand_matmul_csr_csr_dense("M", {
        "indptr": "aip",
        "indices": "aix",
        "data": "ad"
    }, {
        "indptr": "bip",
        "indices": "bix",
        "data": "bd"
    }, "NI", "NJ")
    _run(
        stmts, {
            "M": M,
            "aip": A.indptr,
            "aix": A.indices,
            "ad": A.data,
            "bip": B.indptr,
            "bix": B.indices,
            "bd": B.data,
            "NI": NI,
            "NJ": NJ
        })
    assert np.allclose(M, _dense(A @ B))


def test_csr_dense_mat():
    NI, NK, NC = 6, 5, 4
    A = _make_csr(NI, NK, seed=12)
    B = np.random.default_rng(12).random((NK, NC))
    M = np.zeros((NI, NC))
    stmts = se.expand_matmul_csr_dense_mat("M", {"indptr": "ip", "indices": "ix", "data": "da"}, "B", "NI", "NC")
    _run(stmts, {"M": M, "ip": A.indptr, "ix": A.indices, "da": A.data, "B": B, "NI": NI, "NC": NC})
    assert np.allclose(M, _dense(A) @ B)
