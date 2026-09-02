"""The CSR x CSR -> CSR Gustavson expander, EXECUTED against scipy.

``expand_matmul_csr_csr`` is ~200 lines of two-pass symbolic/numeric SpGEMM with a dense mark array
doubling as an intrusive linked list. Nothing referenced it but its own dispatch-table entry: no test
built its AST, let alone ran it, so a wrong sentinel or an off-by-one in the drain would have shipped
silently. These build the emitted AST, execute it as Python, and compare against ``scipy``'s own
``csr_matmul``.

Running the AST rather than a compiled kernel is deliberate: it is the SAME statements the C and
Fortran walkers consume, so an algorithmic bug is caught here, at the layer that owns it, without a
toolchain in the loop. What this does NOT cover is the walker's lowering of those statements -- that
belongs to the native oracle.
"""

import ast

import numpy as np
import pytest

from numpyto_common.sparse_emit import expand_matmul_csr_csr

sp = pytest.importorskip("scipy.sparse")

_BUFS = ("indptr", "indices", "data")


def _build_fn(n_rows_sym: str = "NR", n_cols_sym: str = "NK"):
    """Compile the expander's statements into a callable over the named buffers."""
    lhs = {b: f"A_{b}" for b in _BUFS}
    rhs = {b: f"B_{b}" for b in _BUFS}
    out = {b: f"C_{b}" for b in _BUFS}
    body = expand_matmul_csr_csr(ast.Name(id="C", ctx=ast.Store()), lhs, rhs, out, n_rows_sym, n_cols_sym)
    params = [*lhs.values(), *rhs.values(), *out.values(), "__csr_mark", "__csr_acc", n_rows_sym, n_cols_sym]
    fn = ast.FunctionDef(
        name="spgemm",
        args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg=p) for p in params], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=body,
        decorator_list=[],
        returns=None,
    )
    mod = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    ns = {}
    exec(compile(mod, "<spgemm>", "exec"), ns)  # noqa: S102 -- executing our own emitted AST is the point
    return ns["spgemm"]


def _run(A, B):
    """``A @ B`` through the emitted SpGEMM; returns an unsorted-column CSR triple."""
    nr, nk = A.shape[0], B.shape[1]
    # Worst-case output nnz, the bound the caller is documented to size C's buffers to.
    cap = int(sum(B.indptr[j + 1] - B.indptr[j] for i in range(nr) for j in A.indices[A.indptr[i] : A.indptr[i + 1]]))
    c_indptr = np.zeros(nr + 1, dtype=np.int64)
    c_indices = np.zeros(max(cap, 1), dtype=np.int64)
    c_data = np.zeros(max(cap, 1), dtype=np.float64)
    _build_fn()(
        A.indptr.astype(np.int64),
        A.indices.astype(np.int64),
        A.data.astype(np.float64),
        B.indptr.astype(np.int64),
        B.indices.astype(np.int64),
        B.data.astype(np.float64),
        c_indptr,
        c_indices,
        c_data,
        np.zeros(nk, dtype=np.int64),
        np.zeros(nk, dtype=np.float64),
        nr,
        nk,
    )
    return c_indptr, c_indices, c_data


def _dense(A, B):
    """The emitted result densified, so the comparison does not depend on column order.

    The expander drains columns in linked-list pop order, so C's columns are UNSORTED within a row
    by design; comparing CSR triples directly would fail on a correct result.
    """
    c_indptr, c_indices, c_data = _run(A, B)
    out = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(A.shape[0]):
        for p in range(c_indptr[i], c_indptr[i + 1]):
            out[i, c_indices[p]] += c_data[p]
    return out, c_indptr


def _rand(m, n, density, seed):
    return sp.random(m, n, density=density, format="csr", random_state=seed, dtype=np.float64)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_scipy_on_random_matrices(seed):
    A, B = _rand(9, 7, 0.3, seed), _rand(7, 11, 0.3, seed + 100)
    got, _ = _dense(A, B)
    np.testing.assert_allclose(got, (A @ B).toarray(), rtol=1e-12, atol=0.0)


def test_indptr_row_counts_match_the_distinct_columns_touched():
    """PASS 1 counts nnz per row; PASS 2 drains that many. A disagreement corrupts every later row,
    since PASS 2 writes from C_indptr[i] -- so pin the counts, not just the values."""
    A, B = _rand(12, 8, 0.4, 7), _rand(8, 9, 0.4, 8)
    _, c_indptr = _dense(A, B)
    ref = (A @ B).tocsr()
    assert list(np.diff(c_indptr)) == list(np.diff(ref.indptr)), "per-row nnz disagrees with scipy"


def test_empty_rows_produce_empty_output_rows():
    # A row of A with no entries must leave C's row empty and NOT advance the write cursor;
    # a mis-set __pos would spill the next row's values into it.
    A = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]]))
    B = sp.csr_matrix(np.array([[3.0, 0.0], [0.0, 4.0]]))
    got, c_indptr = _dense(A, B)
    np.testing.assert_allclose(got, (A @ B).toarray())
    assert c_indptr[2] == c_indptr[1], "empty row consumed output slots"


def test_all_zero_operand_yields_empty_result():
    A = sp.csr_matrix((4, 4), dtype=np.float64)
    B = _rand(4, 4, 0.5, 3)
    got, c_indptr = _dense(A, B)
    assert c_indptr[-1] == 0
    np.testing.assert_allclose(got, np.zeros((4, 4)))


def test_repeated_column_contributions_accumulate():
    """The dense accumulator must SUM every contribution to a column, and the linked list must push
    that column exactly once. A push-per-contribution would emit duplicate entries whose row count
    then exceeds what PASS 1 reserved."""
    # Every A row hits both B rows, and both B rows hit column 0 -- so column 0 of C accumulates two
    # products through one list node.
    A = sp.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    B = sp.csr_matrix(np.array([[5.0, 6.0], [7.0, 0.0]]))
    c_indptr, c_indices, _ = _run(A, B)
    for i in range(A.shape[0]):
        cols = list(c_indices[c_indptr[i] : c_indptr[i + 1]])
        assert len(cols) == len(set(cols)), f"row {i} emitted a duplicate column: {cols}"
    got, _ = _dense(A, B)
    np.testing.assert_allclose(got, (A @ B).toarray())


def test_single_element_matrices():
    A = sp.csr_matrix(np.array([[2.0]]))
    B = sp.csr_matrix(np.array([[3.0]]))
    got, _ = _dense(A, B)
    np.testing.assert_allclose(got, np.array([[6.0]]))


def test_wide_inner_dimension_does_not_overflow_the_mark_array():
    """``__csr_mark`` / ``__csr_acc`` are sized by cols(B), not by the inner dimension -- a scratch
    array sized off the wrong symbol would index out of bounds exactly here (inner >> cols)."""
    A, B = _rand(3, 40, 0.5, 11), _rand(40, 4, 0.5, 12)
    got, _ = _dense(A, B)
    np.testing.assert_allclose(got, (A @ B).toarray(), rtol=1e-12, atol=0.0)


def test_dense_operands_are_still_correct():
    # Fully dense CSR: every column is touched every row, the worst case for the mark/list reset.
    A = sp.csr_matrix(np.arange(1.0, 13.0).reshape(3, 4))
    B = sp.csr_matrix(np.arange(1.0, 21.0).reshape(4, 5))
    got, _ = _dense(A, B)
    np.testing.assert_allclose(got, (A @ B).toarray(), rtol=1e-12, atol=0.0)
