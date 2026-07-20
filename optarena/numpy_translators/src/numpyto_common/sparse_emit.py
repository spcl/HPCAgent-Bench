"""Per-format sparse-matmul dispatchers (Workstream 0).

Each function takes an AST snippet for one sparse-matmul op and returns the
lowered loop nest (incl. output zero-init). Routed from
:mod:`numpyto_common.lib_nodes`'s matmul hoister via
:data:`SPARSE_MATMUL_DISPATCH`. Common shape::

    def expand_matmul_<lhs_fmt>_<rhs_fmt>(target, lhs, rhs, shape_table) -> List[ast.stmt]

* ``target`` -- ``ast.Name`` for the output array.
* ``lhs``, ``rhs`` -- ``ast.Name`` for the operand arrays.
* ``shape_table`` -- hoister's symbolic-dimension dict, passed through.

Raises :class:`NotImplementedError` for unimplemented combinations so the
hoister falls back to the dense path or reports an actionable error.
"""

import ast
from typing import Callable, Dict, List, Optional, Tuple

# Sparse-op result-layout rule (directive #3).
#: Result-layout sentinel.
DENSE = "dense"

#: Per-target: can a sparse-x-sparse op (SpGEMM) produce a *sparse* result?
#: False densifies. Keyed by target name.
#:
#: * C/Fortran CAN emit CSR-output Gustavson SpGEMM, but the matmul hoister
#:   always densifies sparse-x-sparse inside ``alpha*(A@B)+beta*C`` (feeds a
#:   dense buffer) -- so here the answer is "dense". CSR-output is reached
#:   only via a dedicated pure-SpGEMM path, not the hoister.
#: * JAX has no sparse-x-sparse -> sparse (``BCOO @ BCOO`` densifies) -- why
#:   ``spmm`` (CSR @ CSR) is a documented skip, not an ad-hoc refusal.
FRAMEWORK_SPARSE_CAPS: Dict[str, bool] = {
    "c": False,
    "fortran": False,
    "jax": False,
    "numba": True,  # follows the numpy/scipy source (scipy CSR@CSR -> CSR)
    "pythran": True,
    "cupy": True,
}


def result_layout(lhs_layout: Optional[str], rhs_layout: Optional[str], target: str = "c") -> str:
    """Layout of ``lhs @ rhs``'s result given the operand layouts.

    ``lhs_layout``/``rhs_layout``: sparse format name or ``None`` (dense).
    Returns the result's sparse format, or :data:`DENSE`.

    Rule (directive #3): ``sparse @ dense``/``dense @ sparse`` -> always
    **dense** (matches scipy). ``sparse @ sparse`` -> lhs's layout iff the
    target can produce a sparse result (:data:`FRAMEWORK_SPARSE_CAPS`), else
    **dense**.

    Single source of truth for the layout algebra; backends may still apply a
    context-specific override (e.g. C/Fortran always densifies inside
    ``alpha*(A@B)+beta*C`` regardless of caps).
    """
    if lhs_layout is None or rhs_layout is None:
        return DENSE  # sparse x dense / dense x sparse -> dense, always
    # both sparse
    if FRAMEWORK_SPARSE_CAPS.get(target, False):
        return lhs_layout  # SpGEMM into the lhs layout
    return DENSE


def _name(s: str) -> ast.Name:
    return ast.Name(id=s, ctx=ast.Load())


def _store(s: str) -> ast.Name:
    return ast.Name(id=s, ctx=ast.Store())


def _const(v) -> ast.Constant:
    return ast.Constant(value=v)


def _range_call(start, stop) -> ast.Call:
    args = [start, stop] if start is not None else [stop]
    return ast.Call(func=_name("range"), args=args, keywords=[])


def _subscript(base: str, *axes: ast.expr, ctx=None) -> ast.Subscript:
    if not axes:
        raise ValueError(f"_subscript: empty axes for {base}")
    sl = (axes[0] if len(axes) == 1 else ast.Tuple(elts=list(axes), ctx=ast.Load()))
    return ast.Subscript(value=_name(base), slice=sl, ctx=(ctx or ast.Load()))


def _add(a: ast.expr, b: ast.expr) -> ast.BinOp:
    return ast.BinOp(left=a, op=ast.Add(), right=b)


def _sub(a: ast.expr, b: ast.expr) -> ast.BinOp:
    return ast.BinOp(left=a, op=ast.Sub(), right=b)


def _mul(a: ast.expr, b: ast.expr) -> ast.BinOp:
    return ast.BinOp(left=a, op=ast.Mult(), right=b)


def _zero_init_loop(target_id: str, iter_var: str, n_sym: str) -> ast.For:
    return ast.For(
        target=_store(iter_var),
        iter=_range_call(None, _name(n_sym)),
        body=[ast.Assign(targets=[_subscript(target_id, _name(iter_var), ctx=ast.Store())], value=_const(0.0))],
        orelse=[])


def expand_matmul_csr_csr_dense(
        target_id: str,  # dense temp to fill, shape (NI, NJ)
        lhs_buffers: Dict[str, str],  # A: indptr / indices / data  (NI x NK)
        rhs_buffers: Dict[str, str],  # B: indptr / indices / data  (NK x NJ)
        n_rows_sym: str,  # NI
        n_cols_sym: str,  # NJ (output columns)
) -> List[ast.stmt]:
    """``M = A @ B`` for CSR-A, CSR-B accumulated into a DENSE ``M``::

        for i in range(NI):
            for k in range(NJ):
                M[i, k] = 0.0
        for i in range(NI):
            for jj in range(A_indptr[i], A_indptr[i + 1]):
                j = A_indices[jj]
                v = A_data[jj]
                for kk in range(B_indptr[j], B_indptr[j + 1]):
                    k = B_indices[kk]
                    M[i, k] += v * B_data[kk]

    spmm form: scipy's ``alpha*A@B + beta*C`` is dense (sparse+dense->dense),
    so the temp is dense ``(NI, NJ)`` and the surrounding expression lowers
    through the existing dense path. Gustavson row-by-row, writing straight
    into the dense temp -- no symbolic pass, no CSR output buffers.

    Reference: Gustavson, ACM TOMS 4(3) 1978.
    """
    a_indptr = lhs_buffers["indptr"]
    a_indices = lhs_buffers["indices"]
    a_data = lhs_buffers["data"]
    b_indptr = rhs_buffers["indptr"]
    b_indices = rhs_buffers["indices"]
    b_data = rhs_buffers["data"]

    # Zero-init the dense temp: for i: for k: M[i, k] = 0.0
    zero_inner = ast.For(target=_store("__zk"),
                         iter=_range_call(None, _name(n_cols_sym)),
                         body=[
                             ast.Assign(targets=[_subscript(target_id, _name("__zi"), _name("__zk"), ctx=ast.Store())],
                                        value=_const(0.0))
                         ],
                         orelse=[])
    zero_loop = ast.For(target=_store("__zi"), iter=_range_call(None, _name(n_rows_sym)), body=[zero_inner], orelse=[])

    # M[i, k] += v * B_data[kk]
    accum = ast.AugAssign(target=_subscript(target_id, _name("__i"), _name("__k"), ctx=ast.Store()),
                          op=ast.Add(),
                          value=_mul(_name("__v"), _subscript(b_data, _name("__kk"))))
    kk_loop = ast.For(target=_store("__kk"),
                      iter=_range_call(_subscript(b_indptr, _name("__j")),
                                       _subscript(b_indptr, _add(_name("__j"), _const(1)))),
                      body=[
                          ast.Assign(targets=[_store("__k")], value=_subscript(b_indices, _name("__kk"))),
                          accum,
                      ],
                      orelse=[])
    jj_loop = ast.For(target=_store("__jj"),
                      iter=_range_call(_subscript(a_indptr, _name("__i")),
                                       _subscript(a_indptr, _add(_name("__i"), _const(1)))),
                      body=[
                          ast.Assign(targets=[_store("__j")], value=_subscript(a_indices, _name("__jj"))),
                          ast.Assign(targets=[_store("__v")], value=_subscript(a_data, _name("__jj"))),
                          kk_loop,
                      ],
                      orelse=[])
    i_loop = ast.For(target=_store("__i"), iter=_range_call(None, _name(n_rows_sym)), body=[jj_loop], orelse=[])
    return [zero_loop, i_loop]


def expand_matmul_csr_dense_mat(
        target_id: str,  # dense temp (NR, NC)
        lhs_buffers: Dict[str, str],  # A: indptr / indices / data  (NR x NK)
        rhs_name: str,  # dense B (NK x NC)
        n_rows_sym: str,  # NR
        n_cols_sym: str,  # NC (columns of dense B / result)
) -> List[ast.stmt]:
    """``M = A @ B`` for CSR-A and DENSE-B -> dense ``M`` (NR x NC)::

        for i in range(NR):
            for c in range(NC):
                M[i, c] = 0.0
        for i in range(NR):
            for jj in range(A_indptr[i], A_indptr[i + 1]):
                j = A_indices[jj]
                v = A_data[jj]
                for c in range(NC):
                    M[i, c] += v * B[j, c]

    Row-of-A times dense-B: one nonzero of A scales a whole row of B into
    the result row. Output zero-initialised first.
    """
    a_indptr = lhs_buffers["indptr"]
    a_indices = lhs_buffers["indices"]
    a_data = lhs_buffers["data"]
    # zero-init
    zero_inner = ast.For(target=_store("__zc"),
                         iter=_range_call(None, _name(n_cols_sym)),
                         body=[
                             ast.Assign(targets=[_subscript(target_id, _name("__zi"), _name("__zc"), ctx=ast.Store())],
                                        value=_const(0.0))
                         ],
                         orelse=[])
    zero_loop = ast.For(target=_store("__zi"), iter=_range_call(None, _name(n_rows_sym)), body=[zero_inner], orelse=[])
    accum = ast.AugAssign(target=_subscript(target_id, _name("__i"), _name("__c"), ctx=ast.Store()),
                          op=ast.Add(),
                          value=_mul(_name("__v"), _subscript(rhs_name, _name("__j"), _name("__c"))))
    c_loop = ast.For(target=_store("__c"), iter=_range_call(None, _name(n_cols_sym)), body=[accum], orelse=[])
    jj_loop = ast.For(target=_store("__jj"),
                      iter=_range_call(_subscript(a_indptr, _name("__i")),
                                       _subscript(a_indptr, _add(_name("__i"), _const(1)))),
                      body=[
                          ast.Assign(targets=[_store("__j")], value=_subscript(a_indices, _name("__jj"))),
                          ast.Assign(targets=[_store("__v")], value=_subscript(a_data, _name("__jj"))),
                          c_loop,
                      ],
                      orelse=[])
    i_loop = ast.For(target=_store("__i"), iter=_range_call(None, _name(n_rows_sym)), body=[jj_loop], orelse=[])
    return [zero_loop, i_loop]


def expand_matmul_csr_dense_vec(
    target: ast.Name,
    lhs_buffers: Dict[str, str],  # {"indptr": "A_indptr", "indices": "A_indices", "data": "A_data"}
    rhs_name: str,
    n_rows_sym: str,
) -> List[ast.stmt]:
    """``y = A @ x`` for CSR-A (NR x NK) and dense-x (NK,) -> dense-y (NR,)::

        for i in range(NR):
            y[i] = 0
            for k in range(A_indptr[i], A_indptr[i + 1]):
                y[i] += A_data[k] * x[A_indices[k]]

    Replaces the old fancy-gather hack, which only worked because the
    canonical spmv kernel happened to use ``x[cols]`` indexing.
    """
    yi = _subscript(target.id, _name("__i"), ctx=ast.Store())
    indptr = lhs_buffers["indptr"]
    indices = lhs_buffers["indices"]
    data = lhs_buffers["data"]
    # ``range(A_indptr[i], A_indptr[i + 1])``
    inner_start = _subscript(indptr, _name("__i"))
    inner_stop = _subscript(indptr, ast.BinOp(left=_name("__i"), op=ast.Add(), right=_const(1)))
    inner_body = [
        ast.AugAssign(target=yi,
                      op=ast.Add(),
                      value=ast.BinOp(left=_subscript(data, _name("__k")),
                                      op=ast.Mult(),
                                      right=_subscript(rhs_name, _subscript(indices, _name("__k")))))
    ]
    inner_loop = ast.For(target=_store("__k"), iter=_range_call(inner_start, inner_stop), body=inner_body, orelse=[])
    outer = ast.For(target=_store("__i"),
                    iter=_range_call(None, _name(n_rows_sym)),
                    body=[
                        ast.Assign(targets=[yi], value=_const(0.0)),
                        inner_loop,
                    ],
                    orelse=[])
    return [outer]


def expand_matmul_jds_dense_vec(
        target: ast.Name,
        lhs_buffers: Dict[str, str],  # {"perm":..., "jd_ptr":..., "col_ind":..., "jdiag":...}
        rhs_name: str,
        n_rows_sym: str,
        n_jds_sym: str,  # number of jagged diagonals
) -> List[ast.stmt]:
    """``y = A @ x`` for JDS-A and dense-x -> dense-y::

        for i in range(NR):
            y_perm[i] = 0
        for jd in range(njd):
            jd_start = A_jd_ptr[jd]
            jd_len   = A_jd_ptr[jd + 1] - jd_start
            for r in range(jd_len):
                y_perm[r] += A_jdiag[jd_start + r] * x[A_col_ind[jd_start + r]]
        for i in range(NR):
            y[A_perm[i]] = y_perm[i]

    JDS sorts rows by descending length, then stores each row's first
    nonzero (the "1st jagged diagonal"), then each row's second if it has
    one, etc. The permutation unscatters sorted-y back to row order.

    Reference: Saad's SPARSKIT; `Netlib Templates
    <https://netlib.org/linalg/html_templates/node95.html>`_.
    """
    perm = lhs_buffers["perm"]
    jd_ptr = lhs_buffers["jd_ptr"]
    col_ind = lhs_buffers["col_ind"]
    jdiag = lhs_buffers["jdiag"]
    # Scratch ``y_perm`` (sorted-order accumulator); caller lifts it to a
    # fresh local via lowering.py's zeros_locals machinery.
    y_perm = "__jds_y_perm"

    # y_perm[i] = 0 for all i.
    init_loop = ast.For(
        target=_store("__i"),
        iter=_range_call(None, _name(n_rows_sym)),
        body=[ast.Assign(targets=[_subscript(y_perm, _name("__i"), ctx=ast.Store())], value=_const(0.0))],
        orelse=[])

    # jd_len = A_jd_ptr[jd + 1] - A_jd_ptr[jd]
    jd_next = _subscript(jd_ptr, ast.BinOp(left=_name("__jd"), op=ast.Add(), right=_const(1)))
    jd_len = ast.BinOp(left=jd_next, op=ast.Sub(), right=_subscript(jd_ptr, _name("__jd")))
    inner_idx = ast.BinOp(left=_subscript(jd_ptr, _name("__jd")), op=ast.Add(), right=_name("__r"))
    inner_body = [
        ast.AugAssign(target=_subscript(y_perm, _name("__r"), ctx=ast.Store()),
                      op=ast.Add(),
                      value=ast.BinOp(left=_subscript(jdiag, inner_idx),
                                      op=ast.Mult(),
                                      right=_subscript(rhs_name, _subscript(col_ind, inner_idx))))
    ]
    inner_loop = ast.For(target=_store("__r"), iter=_range_call(None, jd_len), body=inner_body, orelse=[])
    jd_loop = ast.For(target=_store("__jd"), iter=_range_call(None, _name(n_jds_sym)), body=[inner_loop], orelse=[])

    # Unscatter: y[A_perm[i]] = y_perm[i]
    unscatter = ast.For(target=_store("__i"),
                        iter=_range_call(None, _name(n_rows_sym)),
                        body=[
                            ast.Assign(targets=[_subscript(target.id, _subscript(perm, _name("__i")), ctx=ast.Store())],
                                       value=_subscript(y_perm, _name("__i")))
                        ],
                        orelse=[])

    return [init_loop, jd_loop, unscatter]


def expand_matmul_sell_c_sigma_dense_vec(
        target: ast.Name,
        lhs_buffers: Dict[str, str],
        rhs_name: str,
        n_rows_sym: str,
        n_slices_sym: str,
        slice_height_sym: str,  # C parameter
) -> List[ast.stmt]:
    """``y = A @ x`` for SELL-C-sigma-A and dense-x -> dense-y::

        for s in range(nslices):
            sl_start = slice_ptr[s]
            sl_end   = slice_ptr[s + 1]
            sl_width = (sl_end - sl_start) / C    # padded row length
            for r in range(C):
                global_r = s * C + r
                if global_r >= NR: break
                acc = 0
                for col in range(sl_width):
                    e = sl_start + col * C + r    # column-major slice
                    if col < row_len[global_r]:
                        acc += val[e] * x[col_idx[e]]
                y[perm[global_r]] = acc

    SELL-C-sigma stores each slice column-major so SIMD lanes (width C) load
    contiguously. ``row_len`` skips padded zeros; ``perm`` unscatters the
    sorted-y back to original row order.

    Emitted form avoids ``break`` (not expressible in the per-element C/
    Fortran emit) by guarding trailing rows of the final slice with
    ``if global_r < NR`` instead.

    Reference: Kreutzer et al. SIAM SISC 36(5) 2014; `arXiv:1307.6209
    <https://arxiv.org/abs/1307.6209>`_.
    """
    slice_ptr = lhs_buffers["slice_ptr"]
    col_idx = lhs_buffers["col_idx"]
    val = lhs_buffers["val"]
    row_len = lhs_buffers["row_len"]
    perm = lhs_buffers["perm"]
    C = slice_height_sym
    acc = "__sell_acc"

    # global_r = s * C + r
    global_r = _add(_mul(_name("__s"), _name(C)), _name("__r"))
    # sl_start = slice_ptr[s]; sl_width = (slice_ptr[s+1] - sl_start) / C
    sl_start = _subscript(slice_ptr, _name("__s"))
    sl_next = _subscript(slice_ptr, _add(_name("__s"), _const(1)))
    sl_width = ast.BinOp(left=_sub(sl_next, sl_start), op=ast.FloorDiv(), right=_name(C))
    # e = sl_start + col * C + r
    e_expr = _add(_add(_subscript(slice_ptr, _name("__s")), _mul(_name("__col"), _name(C))), _name("__r"))
    # acc += val[e] * x[col_idx[e]]   guarded by col < row_len[global_r]
    accum = ast.AugAssign(target=_store(acc),
                          op=ast.Add(),
                          value=_mul(_subscript(val, _name("__e")),
                                     _subscript(rhs_name, _subscript(col_idx, _name("__e")))))
    col_guard = ast.If(test=ast.Compare(left=_name("__col"),
                                        ops=[ast.Lt()],
                                        comparators=[_subscript(row_len, _name("__gr"))]),
                       body=[
                           ast.Assign(targets=[_store("__e")], value=e_expr),
                           accum,
                       ],
                       orelse=[])
    col_loop = ast.For(target=_store("__col"), iter=_range_call(None, sl_width), body=[col_guard], orelse=[])
    # if global_r < NR: <compute + write y[perm[gr]]>
    row_guard = ast.If(test=ast.Compare(left=_name("__gr"), ops=[ast.Lt()], comparators=[_name(n_rows_sym)]),
                       body=[
                           ast.Assign(targets=[_store(acc)], value=_const(0.0)),
                           col_loop,
                           ast.Assign(targets=[_subscript(target.id, _subscript(perm, _name("__gr")), ctx=ast.Store())],
                                      value=_name(acc)),
                       ],
                       orelse=[])
    r_loop = ast.For(target=_store("__r"),
                     iter=_range_call(None, _name(C)),
                     body=[
                         ast.Assign(targets=[_store("__gr")], value=global_r),
                         row_guard,
                     ],
                     orelse=[])
    slice_loop = ast.For(target=_store("__s"), iter=_range_call(None, _name(n_slices_sym)), body=[r_loop], orelse=[])
    return [slice_loop]


def expand_matmul_csc_dense_vec(
    target: ast.Name,
    lhs_buffers: Dict[str, str],  # {"indptr": ..., "indices": ..., "data": ...}
    rhs_name: str,
    n_rows_sym: str,
    n_cols_sym: str,
) -> List[ast.stmt]:
    """``y = A @ x`` for CSC-A (NR x NK) and dense-x (NK,) -> dense-y::

        for i in range(NR): y[i] = 0
        for j in range(NK):
            for k in range(A_indptr[j], A_indptr[j + 1]):
                y[A_indices[k]] += A_data[k] * x[j]

    CSC stores by column: column ``j`` contributes ``A[:, j] * x[j]``, a
    scatter-add into ``y`` (data-dependent row index) needing zero-init
    first. Equivalent to a CSR-transpose spmv.

    Reference: `scipy.sparse.csc_matrix
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html>`_.
    """
    indptr = lhs_buffers["indptr"]
    indices = lhs_buffers["indices"]
    data = lhs_buffers["data"]
    init_loop = _zero_init_loop(target.id, "__i", n_rows_sym)
    scatter_target = _subscript(target.id, _subscript(indices, _name("__k")), ctx=ast.Store())
    inner_body = [
        ast.AugAssign(target=scatter_target,
                      op=ast.Add(),
                      value=_mul(_subscript(data, _name("__k")), _subscript(rhs_name, _name("__j"))))
    ]
    inner_loop = ast.For(target=_store("__k"),
                         iter=_range_call(_subscript(indptr, _name("__j")),
                                          _subscript(indptr, _add(_name("__j"), _const(1)))),
                         body=inner_body,
                         orelse=[])
    col_loop = ast.For(target=_store("__j"), iter=_range_call(None, _name(n_cols_sym)), body=[inner_loop], orelse=[])
    return [init_loop, col_loop]


def expand_matmul_coo_dense_vec(
    target: ast.Name,
    lhs_buffers: Dict[str, str],  # {"row": ..., "col": ..., "data": ...}
    rhs_name: str,
    n_rows_sym: str,
    nnz_sym: str,
) -> List[ast.stmt]:
    """``y = A @ x`` for COO-A and dense-x -> dense-y::

        for i in range(NR): y[i] = 0
        for k in range(nnz):
            y[A_row[k]] += A_data[k] * x[A_col[k]]

    Flat ``(row, col, val)`` triples; each scatter-adds one product into
    ``y[row]`` (needs zero-init first). Order-independent; duplicate
    coordinates accumulate (scipy summed-duplicates semantics).

    Reference: `scipy.sparse.coo_matrix
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html>`_.
    """
    row = lhs_buffers["row"]
    col = lhs_buffers["col"]
    data = lhs_buffers["data"]
    init_loop = _zero_init_loop(target.id, "__i", n_rows_sym)
    scatter_target = _subscript(target.id, _subscript(row, _name("__k")), ctx=ast.Store())
    nnz_body = [
        ast.AugAssign(target=scatter_target,
                      op=ast.Add(),
                      value=_mul(_subscript(data, _name("__k")), _subscript(rhs_name, _subscript(col, _name("__k")))))
    ]
    nnz_loop = ast.For(target=_store("__k"), iter=_range_call(None, _name(nnz_sym)), body=nnz_body, orelse=[])
    return [init_loop, nnz_loop]


def expand_matmul_dia_dense_vec(
    target: ast.Name,
    lhs_buffers: Dict[str, str],  # {"data": ..., "offsets": ...}
    rhs_name: str,
    n_rows_sym: str,
    n_cols_sym: str,
    n_diags_sym: str,
) -> List[ast.stmt]:
    """``y = A @ x`` for DIA-A (NR x NK) and dense-x -> dense-y::

        for i in range(NR): y[i] = 0
        for d in range(ndiag):
            o = A_offsets[d]
            for i in range(NR):
                j = i + o
                if 0 <= j < NK:
                    y[i] += A_data[d, j] * x[j]

    ``A_offsets[d]`` is the diagonal's offset from the main (scipy/LAPACK:
    ``o > 0`` super-diagonal, ``o < 0`` sub-diagonal). scipy keys the data
    column by the *destination* column ``j`` -- read is ``A_data[d, j]``,
    NOT ``A_data[d, i]``. ``0 <= j < NK`` guards off-matrix padding.

    Reference: `scipy.sparse.dia_matrix
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.html>`_.
    """
    data = lhs_buffers["data"]
    offsets = lhs_buffers["offsets"]
    init_loop = _zero_init_loop(target.id, "__i", n_rows_sym)
    j_expr = _add(_name("__i"), _name("__o"))
    accum = ast.AugAssign(target=_subscript(target.id, _name("__i"), ctx=ast.Store()),
                          op=ast.Add(),
                          value=_mul(_subscript(data, _name("__d"), _name("__j")), _subscript(rhs_name, _name("__j"))))
    guard = ast.If(test=ast.Compare(left=_const(0),
                                    ops=[ast.LtE(), ast.Lt()],
                                    comparators=[_name("__j"), _name(n_cols_sym)]),
                   body=[accum],
                   orelse=[])
    row_loop = ast.For(target=_store("__i"),
                       iter=_range_call(None, _name(n_rows_sym)),
                       body=[ast.Assign(targets=[_store("__j")], value=j_expr), guard],
                       orelse=[])
    diag_loop = ast.For(target=_store("__d"),
                        iter=_range_call(None, _name(n_diags_sym)),
                        body=[
                            ast.Assign(targets=[_store("__o")], value=_subscript(offsets, _name("__d"))),
                            row_loop,
                        ],
                        orelse=[])
    return [init_loop, diag_loop]


def expand_matmul_bcsr_dense_vec(
        target: ast.Name,
        lhs_buffers: Dict[str, str],  # {"indptr": ..., "indices": ..., "data": ...}
        rhs_name: str,
        n_block_rows_sym: str,
        block_r_sym: str,  # R: rows per block
        block_c_sym: str,  # C: cols per block
        n_rows_sym: str,  # total scalar rows = n_block_rows * R
) -> List[ast.stmt]:
    """``y = A @ x`` for BCSR-A (block R x C) and dense-x -> dense-y::

        for i in range(NR): y[i] = 0
        for bi in range(nbrows):
            for k in range(A_indptr[bi], A_indptr[bi + 1]):
                bj = A_indices[k]
                for r in range(R):
                    for c in range(C):
                        y[bi*R + r] += A_data[k, r, c] * x[bj*C + c]

    BCSR is CSR over a grid of dense ``R x C`` blocks: ``indptr``/``indices``
    index *block* rows/columns, ``data`` is 3-D ``[nnz_blocks, R, C]``. ``y``
    (length ``NR = n_block_rows * R``) is zero-initialized first. ``NR`` is
    its own symbol rather than ``n_block_rows_sym * R`` because the latter
    may be a compound expression (``len(indptr) - 1``) unsafe to multiply.

    Reference: `scipy.sparse.bsr_matrix
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html>`_.
    """
    indptr = lhs_buffers["indptr"]
    indices = lhs_buffers["indices"]
    data = lhs_buffers["data"]
    init_loop = ast.For(
        target=_store("__i"),
        iter=_range_call(None, _name(n_rows_sym)),
        body=[ast.Assign(targets=[_subscript(target.id, _name("__i"), ctx=ast.Store())], value=_const(0.0))],
        orelse=[])
    out_row = _add(_mul(_name("__bi"), _name(block_r_sym)), _name("__r"))
    x_col = _add(_mul(_name("__bj"), _name(block_c_sym)), _name("__c"))
    accum = ast.AugAssign(target=_subscript(target.id, out_row, ctx=ast.Store()),
                          op=ast.Add(),
                          value=_mul(_subscript(data, _name("__k"), _name("__r"), _name("__c")),
                                     _subscript(rhs_name, x_col)))
    c_loop = ast.For(target=_store("__c"), iter=_range_call(None, _name(block_c_sym)), body=[accum], orelse=[])
    r_loop = ast.For(target=_store("__r"), iter=_range_call(None, _name(block_r_sym)), body=[c_loop], orelse=[])
    k_loop = ast.For(target=_store("__k"),
                     iter=_range_call(_subscript(indptr, _name("__bi")),
                                      _subscript(indptr, _add(_name("__bi"), _const(1)))),
                     body=[
                         ast.Assign(targets=[_store("__bj")], value=_subscript(indices, _name("__k"))),
                         r_loop,
                     ],
                     orelse=[])
    brow_loop = ast.For(target=_store("__bi"),
                        iter=_range_call(None, _name(n_block_rows_sym)),
                        body=[k_loop],
                        orelse=[])
    return [init_loop, brow_loop]


def expand_matmul_bcoo_dense_vec(
        target: ast.Name,
        lhs_buffers: Dict[str, str],  # {"row": ..., "col": ..., "data": ...}
        rhs_name: str,
        n_rows_sym: str,  # total scalar rows = n_block_rows * R
        n_blocks_sym: str,  # number of stored R x C blocks
        block_r_sym: str,  # R: rows per block
        block_c_sym: str,  # C: cols per block
) -> List[ast.stmt]:
    """``y = A @ x`` for BCOO-A (block R x C) and dense-x -> dense-y::

        for i in range(NR): y[i] = 0
        for k in range(n_blocks):
            bi = A_row[k]
            bj = A_col[k]
            for r in range(R):
                for c in range(C):
                    y[bi*R + r] += A_data[k, r, c] * x[bj*C + c]

    BCOO is COO over a grid of dense ``R x C`` blocks: ``row``/``col`` hold
    each stored block's *block* coordinates, ``data`` is 3-D ``[n_blocks, R,
    C]``. One scatter-add pass; ``y`` (length ``NR = n_block_rows * R``) is
    zero-initialized first. Order-independent; duplicates accumulate (COO
    summed-duplicates semantics). Block analogue of
    :func:`expand_matmul_coo_dense_vec`, sharing the block layout of
    :func:`expand_matmul_bcsr_dense_vec`.

    Reference: block generalization of `scipy.sparse.coo_matrix
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html>`_
    over `scipy.sparse.bsr_matrix
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html>`_
    block structure.
    """
    row = lhs_buffers["row"]
    col = lhs_buffers["col"]
    data = lhs_buffers["data"]
    init_loop = _zero_init_loop(target.id, "__i", n_rows_sym)
    out_row = _add(_mul(_name("__bi"), _name(block_r_sym)), _name("__r"))
    x_col = _add(_mul(_name("__bj"), _name(block_c_sym)), _name("__c"))
    accum = ast.AugAssign(target=_subscript(target.id, out_row, ctx=ast.Store()),
                          op=ast.Add(),
                          value=_mul(_subscript(data, _name("__k"), _name("__r"), _name("__c")),
                                     _subscript(rhs_name, x_col)))
    c_loop = ast.For(target=_store("__c"), iter=_range_call(None, _name(block_c_sym)), body=[accum], orelse=[])
    r_loop = ast.For(target=_store("__r"), iter=_range_call(None, _name(block_r_sym)), body=[c_loop], orelse=[])
    k_loop = ast.For(target=_store("__k"),
                     iter=_range_call(None, _name(n_blocks_sym)),
                     body=[
                         ast.Assign(targets=[_store("__bi")], value=_subscript(row, _name("__k"))),
                         ast.Assign(targets=[_store("__bj")], value=_subscript(col, _name("__k"))),
                         r_loop,
                     ],
                     orelse=[])
    return [init_loop, k_loop]


def expand_matmul_ell_dense_vec(
        target: ast.Name,
        lhs_buffers: Dict[str, str],  # {"indices": ..., "data": ...}
        rhs_name: str,
        n_rows_sym: str,
        max_nnz_sym: str,  # max nonzeros per row (slot count)
) -> List[ast.stmt]:
    """``y = A @ x`` for ELL-A (NR x maxnz padded) and dense-x -> dense-y::

        for i in range(NR):
            y[i] = 0
            for s in range(maxnz):
                col = A_indices[i, s]
                if col >= 0:
                    y[i] += A_data[i, s] * x[col]

    ``A_indices``/``A_data`` are rectangular ``[NR, maxnz]``; padding slots
    carry sentinel column ``-1`` (data ``0``), masked by ``col >= 0``.
    Per-row reduction, so ``y[i]`` resets inside the outer loop.

    Reference: `cusparse ELL / ELLPACK
    <https://docs.nvidia.com/cuda/cusparse/index.html#ellpack-ell>`_.
    """
    indices = lhs_buffers["indices"]
    data = lhs_buffers["data"]
    col_assign = ast.Assign(targets=[_store("__col")], value=_subscript(indices, _name("__i"), _name("__s")))
    accum = ast.AugAssign(target=_subscript(target.id, _name("__i"), ctx=ast.Store()),
                          op=ast.Add(),
                          value=_mul(_subscript(data, _name("__i"), _name("__s")), _subscript(rhs_name,
                                                                                              _name("__col"))))
    guard = ast.If(test=ast.Compare(left=_name("__col"), ops=[ast.GtE()], comparators=[_const(0)]),
                   body=[accum],
                   orelse=[])
    slot_loop = ast.For(target=_store("__s"),
                        iter=_range_call(None, _name(max_nnz_sym)),
                        body=[col_assign, guard],
                        orelse=[])
    row_loop = ast.For(target=_store("__i"),
                       iter=_range_call(None, _name(n_rows_sym)),
                       body=[
                           ast.Assign(targets=[_subscript(target.id, _name("__i"), ctx=ast.Store())],
                                      value=_const(0.0)),
                           slot_loop,
                       ],
                       orelse=[])
    return [row_loop]


def expand_matmul_csr_csr(
        target: ast.Name,
        lhs_buffers: Dict[str, str],  # A: indptr / indices / data
        rhs_buffers: Dict[str, str],  # B: indptr / indices / data
        out_buffers: Dict[str, str],  # C: indptr / indices / data
        n_rows_sym: str,  # NR  = rows(A) = rows(C)
        n_cols_sym: str,  # NK  = cols(B) = cols(C)
) -> List[ast.stmt]:
    """``C = A @ B`` for CSR-A (NR x NM), CSR-B (NM x NK) -> CSR-C (NR x NK).

    Gustavson's row-by-row SpGEMM, scipy's two-pass ``csr_matmul``.

    PASS 1 (symbolic, fills ``C_indptr``)::

        for i in range(NR):
            for k in range(NK): __mark[k] = -1
            __nnz = 0
            for jj in range(A_indptr[i], A_indptr[i+1]):
                j = A_indices[jj]
                for kk in range(B_indptr[j], B_indptr[j+1]):
                    k = B_indices[kk]
                    if __mark[k] != i:
                        __mark[k] = i; __nnz += 1
            C_indptr[i+1] = C_indptr[i] + __nnz

    PASS 2 (numeric, fills ``C_indices``/``C_data``) uses a dense
    accumulator ``__acc`` (size NK) and an intrusive linked list in
    ``__mark`` over the touched columns, so output columns drain without a
    per-row sort.

    LHS-format-wins: CSR @ CSR -> CSR. Worst-case output nnz is
    ``sum_i sum_{j in A[i]} nnz(B[j])``; the caller sizes ``C_indices``/
    ``C_data`` to that bound. ``__csr_mark`` (int, NK) and ``__csr_acc``
    (float, NK) are scratch the caller lifts to fresh locals via
    zeros_locals. Output columns are unsorted within each row (linked-list
    pop order) -- downstream code needing sorted CSR must sort per row.

    Reference: Gustavson, ACM TOMS 4(3) 1978; scipy ``csr_matmul``.
    """
    a_indptr = lhs_buffers["indptr"]
    a_indices = lhs_buffers["indices"]
    a_data = lhs_buffers["data"]
    b_indptr = rhs_buffers["indptr"]
    b_indices = rhs_buffers["indices"]
    b_data = rhs_buffers["data"]
    c_indptr = out_buffers["indptr"]
    c_indices = out_buffers["indices"]
    c_data = out_buffers["data"]
    mark = "__csr_mark"
    acc = "__csr_acc"

    # PASS 1 (symbolic).
    reset_mark = ast.For(target=_store("__k"),
                         iter=_range_call(None, _name(n_cols_sym)),
                         body=[ast.Assign(targets=[_subscript(mark, _name("__k"), ctx=ast.Store())], value=_const(-1))],
                         orelse=[])
    p1_if = ast.If(test=ast.Compare(left=_subscript(mark, _name("__k")), ops=[ast.NotEq()], comparators=[_name("__i")]),
                   body=[
                       ast.Assign(targets=[_subscript(mark, _name("__k"), ctx=ast.Store())], value=_name("__i")),
                       ast.AugAssign(target=_store("__nnz"), op=ast.Add(), value=_const(1)),
                   ],
                   orelse=[])
    p1_kk = ast.For(target=_store("__kk"),
                    iter=_range_call(_subscript(b_indptr, _name("__j")),
                                     _subscript(b_indptr, _add(_name("__j"), _const(1)))),
                    body=[
                        ast.Assign(targets=[_store("__k")], value=_subscript(b_indices, _name("__kk"))),
                        p1_if,
                    ],
                    orelse=[])
    p1_jj = ast.For(target=_store("__jj"),
                    iter=_range_call(_subscript(a_indptr, _name("__i")),
                                     _subscript(a_indptr, _add(_name("__i"), _const(1)))),
                    body=[
                        ast.Assign(targets=[_store("__j")], value=_subscript(a_indices, _name("__jj"))),
                        p1_kk,
                    ],
                    orelse=[])
    set_indptr = ast.Assign(targets=[_subscript(c_indptr, _add(_name("__i"), _const(1)), ctx=ast.Store())],
                            value=_add(_subscript(c_indptr, _name("__i")), _name("__nnz")))
    pass1 = ast.For(target=_store("__i"),
                    iter=_range_call(None, _name(n_rows_sym)),
                    body=[reset_mark,
                          ast.Assign(targets=[_store("__nnz")], value=_const(0)), p1_jj, set_indptr],
                    orelse=[])

    # PASS 2 (numeric).
    reset_both = ast.For(target=_store("__k"),
                         iter=_range_call(None, _name(n_cols_sym)),
                         body=[
                             ast.Assign(targets=[_subscript(acc, _name("__k"), ctx=ast.Store())], value=_const(0.0)),
                             ast.Assign(targets=[_subscript(mark, _name("__k"), ctx=ast.Store())], value=_const(-1)),
                         ],
                         orelse=[])
    accum = ast.AugAssign(target=_subscript(acc, _name("__k"), ctx=ast.Store()),
                          op=ast.Add(),
                          value=_mul(_name("__v"), _subscript(b_data, _name("__kk"))))
    push_if = ast.If(test=ast.Compare(left=_subscript(mark, _name("__k")), ops=[ast.Eq()], comparators=[_const(-1)]),
                     body=[
                         ast.Assign(targets=[_subscript(mark, _name("__k"), ctx=ast.Store())], value=_name("__head")),
                         ast.Assign(targets=[_store("__head")], value=_name("__k")),
                         ast.AugAssign(target=_store("__len"), op=ast.Add(), value=_const(1)),
                     ],
                     orelse=[])
    p2_kk = ast.For(target=_store("__kk"),
                    iter=_range_call(_subscript(b_indptr, _name("__j")),
                                     _subscript(b_indptr, _add(_name("__j"), _const(1)))),
                    body=[
                        ast.Assign(targets=[_store("__k")], value=_subscript(b_indices, _name("__kk"))),
                        accum,
                        push_if,
                    ],
                    orelse=[])
    p2_jj = ast.For(target=_store("__jj"),
                    iter=_range_call(_subscript(a_indptr, _name("__i")),
                                     _subscript(a_indptr, _add(_name("__i"), _const(1)))),
                    body=[
                        ast.Assign(targets=[_store("__j")], value=_subscript(a_indices, _name("__jj"))),
                        ast.Assign(targets=[_store("__v")], value=_subscript(a_data, _name("__jj"))),
                        p2_kk,
                    ],
                    orelse=[])
    drain_body = [
        ast.Assign(targets=[_subscript(c_indices, _name("__pos"), ctx=ast.Store())], value=_name("__head")),
        ast.Assign(targets=[_subscript(c_data, _name("__pos"), ctx=ast.Store())],
                   value=_subscript(acc, _name("__head"))),
        ast.AugAssign(target=_store("__pos"), op=ast.Add(), value=_const(1)),
        ast.Assign(targets=[_store("__next")], value=_subscript(mark, _name("__head"))),
        ast.Assign(targets=[_subscript(mark, _name("__head"), ctx=ast.Store())], value=_const(-1)),
        ast.Assign(targets=[_subscript(acc, _name("__head"), ctx=ast.Store())], value=_const(0.0)),
        ast.Assign(targets=[_store("__head")], value=_name("__next")),
    ]
    drain_loop = ast.For(target=_store("__d"), iter=_range_call(None, _name("__len")), body=drain_body, orelse=[])
    pass2 = ast.For(target=_store("__i"),
                    iter=_range_call(None, _name(n_rows_sym)),
                    body=[
                        reset_both,
                        ast.Assign(targets=[_store("__head")], value=_const(-2)),
                        ast.Assign(targets=[_store("__len")], value=_const(0)),
                        p2_jj,
                        ast.Assign(targets=[_store("__pos")], value=_subscript(c_indptr, _name("__i"))),
                        drain_loop,
                    ],
                    orelse=[])

    init_indptr0 = ast.Assign(targets=[_subscript(c_indptr, _const(0), ctx=ast.Store())], value=_const(0))
    return [init_indptr0, pass1, pass2]


# Dispatch table: hoister calls in here when an operand has a non-dense
# layout. NotImplementedError means the combo isn't supported -> the caller
# reports a clear error at parse time, before reaching the C/Fortran walker.

DispatchKey = Tuple[str, str, str]  # (lhs_format, rhs_format, op)

SPARSE_MATMUL_DISPATCH: Dict[DispatchKey, Callable] = {
    # <format> x dense vector -> dense vector  (spmv)
    ("csr", "dense", "matmul_vec"):
    expand_matmul_csr_dense_vec,
    ("csc", "dense", "matmul_vec"):
    expand_matmul_csc_dense_vec,
    ("coo", "dense", "matmul_vec"):
    expand_matmul_coo_dense_vec,
    ("dia", "dense", "matmul_vec"):
    expand_matmul_dia_dense_vec,
    ("bcsr", "dense", "matmul_vec"):
    expand_matmul_bcsr_dense_vec,
    ("bcoo", "dense", "matmul_vec"):
    expand_matmul_bcoo_dense_vec,
    ("ell", "dense", "matmul_vec"):
    expand_matmul_ell_dense_vec,
    ("jds", "dense", "matmul_vec"):
    expand_matmul_jds_dense_vec,
    ("sell_c_sigma", "dense", "matmul_vec"):
    expand_matmul_sell_c_sigma_dense_vec,
    # <format> x <format> -> <format>  (spmm; CSR-output Gustavson)
    ("csr", "csr", "matmul_mat"):
    expand_matmul_csr_csr,
    # <format> x <format> -> dense  (spmm into a dense temp; the common
    # case when the surrounding expression scales/adds a dense matrix)
    ("csr", "csr", "matmul_dense"):
    expand_matmul_csr_csr_dense,
}
