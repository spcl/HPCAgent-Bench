"""Chained-subscript collapse, config-flag typing, and slice-fusion gather offset.

These are the general translator features the QE exact-exchange kernel
(``vexx_all_paths``) needs to emit natively for EVERY configuration (the
translation is orthogonal to the config flags -- one binary handles all of them):

  * ``tabxx_qr[ia][:, ijtoh[ih, jh]]`` -- a CHAINED subscript (a scalar row index,
    then a full slice + a scalar column). numpy basic indexing associates, so it
    collapses to a single ``tabxx_qr[ia, :, col]`` that the shape harvest /
    scalarizer / dot-product operand path handle uniformly.
  * ``okvan`` / ``tqr`` / ... -- boolean CONFIG FLAGS. A bool-valued preset scalar
    is a ``bool`` parameter (C ``bool`` / Fortran ``logical``), not an integer
    size symbol, so the ``if okvan and tqr:`` conditionals type-check.
  * ``big_result[ip*n:ip*n+n] -= rg[nlg]`` (the noncolin ``npol=2`` finalise) --
    a slice-assign into a NON-zero-start destination gathering through a
    length-matched index array: the gather index must be read at the LOCAL offset
    ``si0 - ip*n``, not the absolute ``si0`` (which runs off ``nlg``).
"""
import ast

import numpy as np
from _op_oracle import run_op

from numpyto_common.frontend import _collect_bool_preset_names
from numpyto_common.lib_nodes import _reads_complex
from numpyto_common.lowering import (_CollapseChainedSubscripts, _ShapeMidExpressionRewriter, _SliceToScalarRewriter)

_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")


def _ok(res):
    return all(v == "ok" or v.startswith("skip") for v in res.values()), res


def _expr(s: str) -> ast.AST:
    return ast.parse(s, mode="eval").body


def _unparse(node: ast.AST) -> str:
    return ast.unparse(ast.fix_missing_locations(node))


# ---- structural: chained subscript collapses to a single subscript ----


def _collapse(expr: str, shapes) -> str:
    node = ast.parse(expr, mode="eval").body
    new = _CollapseChainedSubscripts({k: tuple(v) for k, v in shapes.items()}).visit(node)
    return ast.unparse(ast.fix_missing_locations(new))


def test_collapse_scalar_row_then_slice_column():
    # ``tabxx_qr[ia][:, c]`` -- scalar row consumes axis 0; the slice + column
    # apply to the remaining axes -> ``tabxx_qr[ia, :, c]``.
    assert _collapse("A[ia][:, c]", {"A": ("nat", "K", "nij")}) == "A[ia, :, c]"


def test_collapse_slice_then_trailing_scalar():
    # ``becxx[:, j, k][m]`` -- the trailing index selects the surviving full-slice
    # axis 0 -> ``becxx[m, j, k]``.
    assert _collapse("A[:, j, k][m]", {"A": ("nkb", "nb", "nks")}) == "A[m, j, k]"


def test_collapse_bails_on_unknown_base_shape():
    # No shape for the base array -> left chained (cannot resolve trailing axes).
    assert _collapse("A[i][:, c]", {}) == "A[i][:, c]"


def test_collapse_bails_on_partial_inner_slice():
    # A bounded inner slice is not a plain associate -> left untouched.
    assert _collapse("A[1:3][j]", {"A": ("n", "m")}) == "A[1:3][j]"


def test_collapse_bails_on_fancy_inner_index():
    # ``A[idx][j]`` with ``idx`` an ARRAY is a fancy GATHER (== ``A[idx[j]]``), not
    # a scalar associate -- collapsing to ``A[idx, j]`` would change the access, so
    # it must be left untouched (``idx`` known-array via its shape-table entry).
    assert _collapse("A[idx][j]", {"A": ("n", "m"), "idx": ("k", )}) == "A[idx][j]"


def test_collapse_keeps_scalar_inner_when_a_sibling_name_is_an_array():
    # ``A[i][j]`` -- ``i`` is a plain scalar (absent from the shape table) even
    # though some OTHER name ``idx`` is an array: the scalar associate still fires.
    assert _collapse("A[i][j]", {"A": ("n", "m"), "idx": ("k", )}) == "A[i, j]"


def test_collapse_bails_on_ellipsis_inner():
    # An ellipsis stands for an unknown number of axes -> cannot align -> bail.
    assert _collapse("A[..., j][k]", {"A": ("n", "m", "p")}) == "A[..., j][k]"


# ---- pure: a boolean preset value is a config-flag name (typed bool) ----


def test_bool_preset_names_picks_boolean_flags_not_int_symbols():
    params = {
        "S": {
            "N": 6,
            "okvan": False,
            "tqr": False,
            "negrp": 1
        },
        "fuzzed": {
            "N": [6, 16],
            "okvan": {
                "set": [False, True]
            },
            "negrp": {
                "set": [1, 2]
            }
        },
    }
    # ``okvan`` is boolean everywhere it is pinned; ``N`` / ``negrp`` are integers.
    assert _collect_bool_preset_names(params) == {"okvan", "tqr"}


# ---- numeric: bit-close to numpy across every backend ----


def test_chained_column_dot_matches_numpy():
    # ``np.dot(A[ia][:, 1], v[box[ia]])`` -- the collapsed chained column dotted
    # with a materialised-box gather (vexx_k ``_newdxx_r``).
    src = ("import numpy as np\n"
           "def f(A, box, v, out):\n"
           "    nat = A.shape[0]\n"
           "    for ia in range(nat):\n"
           "        bx = box[ia]\n"
           "        col = A[ia][:, 1]\n"
           "        out[ia] = np.dot(col, v[bx])\n")
    nat, K, ncol, N = 3, 4, 2, 10
    rng = np.random.default_rng(0)
    A = rng.standard_normal((nat, K, ncol))
    box = np.stack([np.sort(rng.choice(N, K, replace=False)) for _ in range(nat)]).astype(np.int64)
    v = rng.standard_normal(N)
    res = run_op(src,
                 "f", {
                     "A": A,
                     "box": box,
                     "v": v
                 }, {"out": (nat, )}, {
                     "nat": nat,
                     "K": K,
                     "ncol": ncol,
                     "N": N
                 },
                 shapes={
                     "A": "(nat,K,ncol)",
                     "box": "(nat,K)",
                     "v": "(N,)",
                     "out": "(nat,)"
                 },
                 backends=_ALL)
    ok, r = _ok(res)
    assert ok, r


def test_slice_assign_gather_offset_matches_numpy():
    # ``out[k*n:k*n+n] -= r[idx]`` for k in 0,1 -- the length-``n`` gather index
    # ``idx`` must be read at the LOCAL slice offset, so k=1 does not run off it
    # (the vexx_k noncolin npol=2 finalise OOB).
    src = ("import numpy as np\n"
           "def f(r, idx, out):\n"
           "    n = idx.shape[0]\n"
           "    for k in range(2):\n"
           "        out[k * n:k * n + n] -= r[idx]\n")
    n, M, P = 5, 12, 10
    rng = np.random.default_rng(1)
    r = rng.standard_normal(M)
    idx = rng.integers(0, M, size=n).astype(np.int64)
    res = run_op(src,
                 "f", {
                     "r": r,
                     "idx": idx
                 }, {"out": (P, )}, {
                     "n": n,
                     "M": M,
                     "P": P
                 },
                 shapes={
                     "r": "(M,)",
                     "idx": "(n,)",
                     "out": "(P,)"
                 },
                 backends=_ALL)
    ok, rr = _ok(res)
    assert ok, rr


def test_shape_of_complex_array_is_integer_bound():
    # ``n = z.shape[0]`` reads a DIMENSION (int) even though ``z`` is complex --
    # a complex-typed loop bound would make ``for i in range(n)`` a type error
    # (vexx_k ``ngm = qgm.shape[0]``).
    src = ("import numpy as np\n"
           "def f(z, out):\n"
           "    n = z.shape[0]\n"
           "    for i in range(n):\n"
           "        out[i] = z[i].real + z[i].imag\n")
    N = 6
    rng = np.random.default_rng(2)
    z = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex128)
    res = run_op(src, "f", {"z": z}, {"out": (N, )}, {"N": N}, shapes={"z": "(N,)", "out": "(N,)"}, backends=_ALL)
    ok, r = _ok(res)
    assert ok, r


# ---- .shape / len / .size resolve to the SYMBOLIC dims that were provided ----
# ``_ShapeMidExpressionRewriter`` rewrites an inline extent read against the array's
# declared shape tuple, so ``A.shape[k]`` emits the k-th shape SYMBOL (not a native
# ``.shape`` the C / Fortran backends cannot lower).


def _rewrite_shape(expr: str, shapes) -> str:
    node = _expr(expr)
    new = _ShapeMidExpressionRewriter({k: tuple(v) for k, v in shapes.items()}).visit(node)
    return _unparse(new)


def test_shape_index_maps_to_declared_symbol():
    # ``A.shape[k]`` -> the k-th token of the provided shape tuple.
    assert _rewrite_shape("A.shape[0]", {"A": ("nat", "K", "nij")}) == "nat"
    assert _rewrite_shape("A.shape[1]", {"A": ("nat", "K", "nij")}) == "K"
    assert _rewrite_shape("A.shape[2]", {"A": ("nat", "K", "nij")}) == "nij"


def test_shape_index_in_range_bound_maps_to_symbol():
    # The common ``for i in range(A.shape[0])`` -> ``range(nat)``.
    assert _rewrite_shape("range(A.shape[0])", {"A": ("nat", "K")}) == "range(nat)"


def test_shape_index_numeric_dim_is_a_literal():
    # A concrete (pinned) dimension resolves to an integer literal, not a symbol.
    assert _rewrite_shape("A.shape[1]", {"A": ("N", "3")}) == "3"


def test_bare_shape_maps_to_symbol_tuple():
    # ``A.shape`` (e.g. ``np.zeros(A.shape)``) -> the full symbol tuple.
    assert _rewrite_shape("A.shape", {"A": ("N", "M")}) == "(N, M)"


def test_len_and_size_map_to_symbols():
    # ``len(A)`` == ``A.shape[0]``; ``A.size`` == product of the shape symbols.
    assert _rewrite_shape("len(A)", {"A": ("N", "M")}) == "N"
    assert _rewrite_shape("A.size", {"A": ("N", "M")}) == "N * M"


def test_shape_of_unknown_array_is_left_untouched():
    # No declared shape -> cannot resolve -> the read is left as-is.
    assert _rewrite_shape("A.shape[0]", {}) == "A.shape[0]"


# ---- .shape read is INTEGER even off a complex array (skips the complex walk) ----


def test_reads_complex_skips_shape_subtree_bare_and_compound():
    # ``qgm`` is complex, but a ``.shape`` read yields integer DIMENSIONS. The
    # complex-dtype predicate must skip the ``.shape`` subtree for the bare form
    # AND the compound arithmetic form, else the integer bound is tagged complex
    # (vexx_k ``ngm = qgm.shape[0]`` / ``qgm.shape[0] - 1``).
    dt = {"qgm": "complex128"}
    assert _reads_complex(_expr("qgm.shape[0]"), dt) is False
    assert _reads_complex(_expr("qgm.shape[0] - 1"), dt) is False
    assert _reads_complex(_expr("2 * qgm.shape[0] + 1"), dt) is False


def test_reads_complex_still_detects_a_genuine_complex_value_read():
    # A real VALUE read of the complex array (not its shape) is still complex.
    dt = {"qgm": "complex128"}
    assert _reads_complex(_expr("qgm[i] + 1.0"), dt) is True
    assert _reads_complex(_expr("qgm.shape[0] * qgm[0]"), dt) is True
    assert _reads_complex(_expr("(1 + 2j)"), dt) is True


# ---- trailing implicit-axis pad reads at the LOCAL slice offset (iter - start) ----


def _pad_trailing(rhs_expr: str, start, source_shape):
    # Drive _SliceToScalarRewriter for a single-slice LHS assignment whose slice
    # starts at ``start``, on a partial-scalar RHS of a higher-rank source.
    iv = ast.Name(id="si", ctx=ast.Load())
    lhs_slice = ast.Slice(lower=(None if start == 0 else ast.Constant(start)), upper=None, step=None)
    rw = _SliceToScalarRewriter(
        array_shapes={"dH": tuple(source_shape)},
        iter_vars=[iv],
        lhs_ranges=[(ast.Constant(start), ast.Constant(start + 3))],
        lhs_name="out",
        lhs_dims=[lhs_slice],
    )
    return _unparse(rw.visit(_expr(rhs_expr)))


def test_trailing_pad_reads_local_offset_for_nonzero_start():
    # ``out[2:2+M] = dH[a, b]`` -- dH rank 3, RHS names 2 axes; the implicit
    # trailing axis is padded with the LHS slice iter at its LOCAL position
    # ``si - 2``, so a non-zero-start destination spans dH's length-M axis from 0
    # (the absolute ``si`` would run off the axis).
    assert _pad_trailing("dH[a, b]", 2, ("A", "B", "M")) == "dH[a, b, si - 2]"


def test_trailing_pad_no_offset_for_zero_start():
    # A zero-start slice needs no correction: local offset == absolute index.
    assert _pad_trailing("dH[a, b]", 0, ("A", "B", "M")) == "dH[a, b, si]"


# ---- iter-start offset copies the shared start node (no AST aliasing) ----


def test_iter_minus_start_copies_shared_start_node():
    # ``start`` is the SAME node object as the loop-header ``range`` lower bound, so
    # ``_iter_minus_start`` must embed a COPY -- else one mutable subtree lives in two
    # tree positions and a later in-place rewrite of the loop bound corrupts the
    # gather offset (and vice versa).
    start = _expr("ip * n")  # a shared BinOp: the slice lower bound / loop start
    iv = ast.Name(id="si", ctx=ast.Load())
    out = _SliceToScalarRewriter._iter_minus_start(iv, start)
    assert _unparse(out) == "si - ip * n"
    embedded = [b for b in ast.walk(out) if isinstance(b, ast.BinOp) and isinstance(b.op, ast.Mult)]
    assert embedded and embedded[0] is not start  # a copy, not the aliased node


def test_iter_minus_start_zero_start_is_bare_fresh_iter():
    # start == 0 -> bare iter (no offset), and a FRESH Name (not the passed object).
    iv = ast.Name(id="si", ctx=ast.Load())
    out = _SliceToScalarRewriter._iter_minus_start(iv, ast.Constant(value=0))
    assert _unparse(out) == "si"
    assert out is not iv
