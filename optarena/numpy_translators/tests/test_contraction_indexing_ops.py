"""AST-level unit tests for the contraction / scan / indexing op batch.

Each test asserts the expander (or lowering rewriter) emits the expected loop
structure or intrinsic, so a regression points straight at the cause. The
end-to-end numerical correctness across c / cpp / fortran / numba / jax is
covered by ``test_contraction_indexing_ops_e2e`` via the standalone oracle.
"""
import ast

import pytest

from numpyto_common.lib_nodes import (NP_CALL_EXPANDERS, _matmul_result_shape, _parse_einsum_subscripts, expand_cumprod,
                                      expand_cumsum, expand_diagonal, expand_einsum, expand_inner, expand_linalg_norm,
                                      expand_median, expand_reshape, expand_roll, expand_tensordot, expand_trace,
                                      expand_tril, expand_triu, expand_vdot)
from numpyto_common.lowering import (_EllipsisExpander, _MatmulCallRewriter, _ReshapeMethodRewriter)


def _name(n):
    return ast.Name(id=n, ctx=ast.Load())


def _unparse(stmts):
    mod = ast.fix_missing_locations(ast.Module(body=list(stmts), type_ignores=[]))
    return ast.unparse(mod)


# --------------------------------------------------------------------------- #
# A.1  np.matmul call-form normalizes to the ``@`` BinOp                       #
# --------------------------------------------------------------------------- #


def test_matmul_call_normalized_to_binop():
    tree = ast.parse("c = np.matmul(a, b)")
    _MatmulCallRewriter().visit(tree)
    rhs = tree.body[0].value
    assert isinstance(rhs, ast.BinOp) and isinstance(rhs.op, ast.MatMult)
    assert rhs.left.id == "a" and rhs.right.id == "b"


def test_matmul_call_three_args_left_alone():
    # ``np.matmul(a, b, out)`` (out= kwarg form) is not the 2-arg shape we map.
    tree = ast.parse("c = np.matmul(a, b, out)")
    _MatmulCallRewriter().visit(tree)
    assert isinstance(tree.body[0].value, ast.Call)


# --------------------------------------------------------------------------- #
# A.2  batched matmul result-shape (both-batched is the new case)              #
# --------------------------------------------------------------------------- #


def test_matmul_result_shape_both_batched():
    assert _matmul_result_shape(("B", "M", "K"), ("B", "K", "N")) == ("B", "M", "N")


def test_matmul_result_shape_one_sided_batched():
    assert _matmul_result_shape(("B", "M", "K"), ("K", "N")) == ("B", "M", "N")
    assert _matmul_result_shape(("M", "K"), ("B", "K", "N")) == ("B", "M", "N")


def test_matmul_result_shape_batch_mismatch_is_none():
    # Different leading batch dims do not contract.
    assert _matmul_result_shape(("B", "M", "K"), ("C", "K", "N")) is None


# --------------------------------------------------------------------------- #
# A.3  einsum subscript parse + loop structure                                 #
# --------------------------------------------------------------------------- #


def test_parse_einsum_explicit():
    assert _parse_einsum_subscripts("ij,jk->ik") == (["ij", "jk"], "ik")


def test_parse_einsum_implicit_output():
    # numpy implicit output = singly-occurring indices, alphabetical.
    assert _parse_einsum_subscripts("ij,jk") == (["ij", "jk"], "ik")


def test_parse_einsum_ellipsis_unsupported():
    with pytest.raises(NotImplementedError):
        _parse_einsum_subscripts("...ij->...i")


# --------------------------------------------------------------------------- #
# A.4  batched (>=3-D) matmul desugaring for the verbatim Python backends      #
#      (numba / pythran cannot type stacked ``@``; lower to a loop of GEMMs)   #
# --------------------------------------------------------------------------- #
from types import SimpleNamespace  # noqa: E402

from numpyto_common.numpy_desugar import desugar_for_python_backend  # noqa: E402


def _kir(kernel_name, **arrays):
    """Minimal KernelIR stand-in: name + (name -> shape-tuple) arrays."""
    arrs = [SimpleNamespace(name=n, shape=s) for n, s in arrays.items()]
    return SimpleNamespace(kernel_name=kernel_name, arrays=arrs)


def test_batched_matmul_desugars_to_gemm_loop():
    # The canonical SeisSol batched GEMM: 3-D ``I`` @ shared 2-D ``star``.
    src = ("def kernel(Q, I, star):\n"
           "    Q[:] = Q + I @ star\n")
    kir = _kir("kernel", Q=("b", "n", "q"), I=("b", "n", "q"), star=("q", "q"))
    out = desugar_for_python_backend(src, kir)
    tree = ast.parse(out)
    fn = tree.body[0]
    loop = fn.body[0]
    assert isinstance(loop, ast.For), "batched @ must become a for-loop of GEMMs"
    assert loop.iter.args[0].value.attr == "shape"  # range(I.shape[0])
    # Inside: a 2-D ``@`` (both operands dropped to rank 2 by [bv]).
    body_src = _unparse(loop.body)
    assert "@" in body_src and "[__bm0]" in body_src
    # The shared 2-D ``star`` is NOT batch-indexed (broadcast).
    assert "star[__bm0]" not in body_src and "star" in body_src


def test_2d_matmul_left_verbatim():
    # An ordinary 2-D GEMM is supported by numba/pythran -> emit unchanged.
    src = ("def kernel(C, A, B):\n"
           "    C[:] = A @ B\n")
    kir = _kir("kernel", C=("m", "n"), A=("m", "k"), B=("k", "n"))
    assert desugar_for_python_backend(src, kir) == src


def test_reshape_wrapped_matmul_lowers_to_contraction():
    # doitgen: ``np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))``.
    # The batched-matmul pass still refuses reshape-wrapped operands (indexing the
    # leading axis would miscompile), but the dedicated reshape-matmul pass
    # recognises the unit-dim-insertion form and lowers it to a contraction loop
    # into a fresh temp (so the ``A = f(A)`` WAR is safe).
    src = ("def kernel(NR, NQ, NP, A, C4):\n"
           "    A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))\n")
    kir = _kir("kernel", A=("NR", "NQ", "NP"), C4=("NP", "NP"))
    out = desugar_for_python_backend(src, kir)
    assert "@" not in out and "reshape" not in out and "for " in out


def test_no_matmul_returned_bytewise_unchanged():
    # No trigger token at all -> byte-for-byte identity (no reparse churn).
    src = "def kernel(a, b):\n    a[:] = a + b  # comment kept\n"
    assert desugar_for_python_backend(src, _kir("kernel", a=("n", ), b=("n", ))) is src


def test_np_pad_edge_inlined_to_loop_nest():
    # numba / pythran cannot type np.pad -> inline an edge-pad copy loop nest.
    src = ("def kernel(in_grid, out_grid, N, R):\n"
           "    padded = np.pad(in_grid, pad_width=R, mode='edge')\n"
           "    out_grid[:] = padded[:N, :N, :N]\n")
    kir = _kir("kernel", in_grid=("N", "N", "N"), out_grid=("N", "N", "N"))
    out = desugar_for_python_backend(src, kir)
    assert "np.pad" not in out, "np.pad must be expanded"
    assert "np.empty" in out and out.count("for ") >= 3  # rank-3 copy nest
    assert "min(max(" in out  # edge clamp


def test_np_pad_per_axis_tuple_widths():
    # vector stencils use per-axis ((R,R),...,(0,0)) widths -> last axis unpadded.
    src = ("def kernel(in_grid, out_grid, N, R):\n"
           "    padded = np.pad(in_grid, pad_width=((R, R), (R, R), (R, R), (0, 0)), mode='edge')\n")
    kir = _kir("kernel", in_grid=("N", "N", "N", "C"))
    out = desugar_for_python_backend(src, kir)
    assert "np.pad" not in out and out.count("for ") >= 4  # rank-4 nest


def test_einsum_inlined_to_contraction_loops():
    # The SeisSol tensor contraction: 3-operand einsum nested in an add.
    src = ("def kernel(Q, I, kDivM, star):\n"
           "    Q[:] = Q + np.einsum('dkl,blq,dqp->bkp', kDivM, I, star)\n")
    kir = _kir("kernel", Q=("b", "k", "p"), I=("b", "l", "q"), kDivM=("d", "k", "l"), star=("d", "q", "p"))
    out = desugar_for_python_backend(src, kir)
    assert "einsum" not in out, "einsum must be expanded"
    assert "+=" in out and out.count("for ") == 6  # 3 output + 3 contracted axes
    assert "Q[:] = Q + __es0" in out  # einsum hoisted to a temp, add preserved


def test_einsum_matmul_has_output_loops_and_summed_inner():
    st = {"a": ("M", "K"), "b": ("K", "N")}
    out = _unparse(expand_einsum(_name("out"), [ast.Constant("ij,jk->ik"), _name("a"), _name("b")], st))
    # i, k are output loops; j is the summed accumulation loop.
    assert "for __es_i in range(M):" in out
    assert "for __es_k in range(N):" in out
    assert "for __es_j in range(K):" in out
    assert "out[__es_i, __es_k] = 0.0" in out
    assert "out[__es_i, __es_k] +=" in out


def test_einsum_trace_is_scalar_accumulation():
    st = {"a": ("M", "M")}
    out = _unparse(expand_einsum(_name("s"), [ast.Constant("ii->"), _name("a")], st))
    # No output letters -> scalar target, single summed loop over the diagonal.
    assert "s = 0.0" in out and "s += a[__es_i, __es_i]" in out


def test_einsum_transpose_no_summation():
    st = {"a": ("M", "N")}
    out = _unparse(expand_einsum(_name("out"), [ast.Constant("ij->ji"), _name("a")], st))
    assert "out[__es_j, __es_i] = a[__es_i, __es_j]" in out


def test_einsum_seissol_three_operand():
    st = {"g": ("D", "KK", "L"), "h": ("NB", "L", "Q"), "c": ("D", "Q", "P")}
    out = _unparse(
        expand_einsum(_name("out"),
                      [ast.Constant("dkl,blq,dqp->bkp"),
                       _name("g"), _name("h"), _name("c")], st))
    # b, k, p are output loops; d, l, q are summed inner loops.
    for v in ("__es_b", "__es_k", "__es_p", "__es_d", "__es_l", "__es_q"):
        assert f"for {v} in range" in out
    assert "out[__es_b, __es_k, __es_p] +=" in out


# --------------------------------------------------------------------------- #
# A.4  tensordot / inner / vdot                                                #
# --------------------------------------------------------------------------- #


def test_tensordot_axes1_is_matmul_contraction():
    st = {"a": ("M", "K"), "b": ("K", "N")}
    out = _unparse(
        expand_tensordot(_name("out"), [_name("a"), _name("b")],
                         st,
                         kwargs=[ast.keyword(arg="axes", value=ast.Constant(1))]))
    assert "out[__es_a, __es_c] +=" in out  # contracts the shared K axis


def test_inner_rank1_is_dot():
    st = {"u": ("K", ), "v": ("K", )}
    out = _unparse(expand_inner(_name("s"), [_name("u"), _name("v")], st))
    assert "s = 0.0" in out and "s += u[__r0] * v[__r0]" in out


def test_vdot_real_no_conjugate():
    # Real operands: no conj() call (CONJG/__npb_conj is invalid on a real scalar).
    st = {"u": ("K", ), "v": ("K", )}
    out = _unparse(expand_vdot(_name("s"), [_name("u"), _name("v")], st, local_dtypes={}))
    assert "conj" not in out and "s += u[__vd] * v[__vd]" in out


def test_vdot_complex_conjugates_first_operand():
    st = {"u": ("K", ), "v": ("K", )}
    out = _unparse(expand_vdot(_name("s"), [_name("u"), _name("v")], st, local_dtypes={"u": "complex128"}))
    assert "np.conj(u[__vd])" in out


# --------------------------------------------------------------------------- #
# B.5  trace / diagonal direct                                                 #
# --------------------------------------------------------------------------- #


def test_trace_sums_diagonal():
    out = _unparse(expand_trace(_name("s"), [_name("a")], {"a": ("M", "M")}))
    assert "s = 0.0" in out and "s += a[__tr, __tr]" in out


def test_diagonal_copies_diagonal():
    out = _unparse(expand_diagonal(_name("out"), [_name("a")], {"a": ("M", "M")}))
    assert "out[__dg] = a[__dg, __dg]" in out


# --------------------------------------------------------------------------- #
# B.6  cumsum / cumprod prefix scan                                            #
# --------------------------------------------------------------------------- #


def test_cumsum_1d_prefix_recurrence():
    out = _unparse(expand_cumsum(_name("out"), [_name("a")], {"a": ("N", )}))
    assert "out[0] = a[0]" in out
    assert "out[__cs0] = out[__cs0 - 1] + a[__cs0]" in out
    assert "range(1, N)" in out


def test_cumprod_uses_mult():
    out = _unparse(expand_cumprod(_name("out"), [_name("a")], {"a": ("N", )}))
    assert "out[__cs0] = out[__cs0 - 1] * a[__cs0]" in out


def test_cumsum_axis1_scans_inner_axis():
    out = _unparse(
        expand_cumsum(_name("out"), [_name("a")], {"a": ("M", "N")},
                      kwargs=[ast.keyword(arg="axis", value=ast.Constant(1))]))
    # axis 0 is the outer loop; the scan recurrence runs along axis 1.
    assert "for __cs0 in range(M):" in out
    assert "out[__cs0, __cs1] = out[__cs0, __cs1 - 1] + a[__cs0, __cs1]" in out


def _slice_target(name, lower):
    """``name[lower:]`` Store subscript for a partial-slice scan target."""
    return ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()),
                         slice=ast.Slice(lower=(None if lower is None else ast.Constant(lower)),
                                         upper=None, step=None),
                         ctx=ast.Store())


def test_cumsum_partial_slice_target_shifts_write_index():
    """``row_offsets[1:] = np.cumsum(m_sizes)`` (DBCSR): the scan runs over the
    operand's length but every write lands at ``lower + i``. Before the fix a
    partial-slice target never reached the cumulative expander at all, so the
    ``np.cumsum`` call survived to emit and raised 'not supported'."""
    out = _unparse(expand_cumsum(_slice_target("row_offsets", 1), [_name("m_sizes")], {"m_sizes": ("M", )}))
    assert "row_offsets[0 + 1] = m_sizes[0]" in out
    assert "row_offsets[__cs0 + 1] = row_offsets[__cs0 - 1 + 1] + m_sizes[__cs0]" in out
    assert "range(1, M)" in out


def test_cumsum_zero_lower_slice_target_has_no_offset():
    """A ``[0:]`` (or bare ``[:]``) lower bound adds no shift -- the emitted
    recurrence is identical to a bare-Name target, so the fix never perturbs
    the existing full-target path."""
    shifted = _unparse(expand_cumsum(_slice_target("out", 0), [_name("a")], {"a": ("N", )}))
    bare = _unparse(expand_cumsum(_name("out"), [_name("a")], {"a": ("N", )}))
    assert shifted == bare
    assert "+ 0" not in shifted


# --------------------------------------------------------------------------- #
# B.7  median: copy + sort + pick middle                                       #
# --------------------------------------------------------------------------- #


def test_median_sorts_and_picks_middle():
    allocs = {}
    out = _unparse(expand_median(_name("s"), [_name("a")], {"a": ("N", )}, fresh_local_allocs=allocs))
    assert "__md_buf" in allocs  # scratch buffer registered
    assert "while" in out  # the in-place sort routine
    assert "N // 2" in out  # middle index
    assert "% 2" in out  # even/odd parity test


# --------------------------------------------------------------------------- #
# C.8  np.roll modular index                                                   #
# --------------------------------------------------------------------------- #


def test_roll_uses_modular_source_index():
    out = _unparse(expand_roll(_name("out"), [_name("a"), ast.Constant(3)], {"a": ("N", )}))
    # ((i - shift) % N + N) % N keeps the source index non-negative.
    assert "out[__rl0] = a[((__rl0 - 3) % N + N) % N]" in out


# --------------------------------------------------------------------------- #
# C.9  reshape method-form + ellipsis expansion                                #
# --------------------------------------------------------------------------- #


def test_reshape_method_varargs_to_func():
    tree = ast.parse("y = a.reshape(3, 4)")
    _ReshapeMethodRewriter().visit(tree)
    call = tree.body[0].value
    assert isinstance(call.func, ast.Attribute) and call.func.attr == "reshape"
    assert call.func.value.id == "np" and call.args[0].id == "a"
    assert isinstance(call.args[1], ast.Tuple) and len(call.args[1].elts) == 2


def test_reshape_method_tuple_to_func():
    tree = ast.parse("y = a.reshape((3, 4))")
    _ReshapeMethodRewriter().visit(tree)
    call = tree.body[0].value
    assert call.func.value.id == "np" and len(call.args[1].elts) == 2


def test_ellipsis_trailing_expands_to_full_slices():
    tree = ast.parse("y = a[..., 0]")
    _EllipsisExpander({"a": ["M", "N", "P"]}).visit(tree)
    sub = tree.body[0].value
    elts = sub.slice.elts
    assert isinstance(elts[0], ast.Slice) and isinstance(elts[1], ast.Slice)
    assert isinstance(elts[2], ast.Constant) and elts[2].value == 0


def test_ellipsis_leading_expands_to_full_slices():
    tree = ast.parse("y = a[0, ...]")
    _EllipsisExpander({"a": ["M", "N", "P"]}).visit(tree)
    elts = tree.body[0].value.slice.elts
    assert isinstance(elts[0], ast.Constant) and elts[0].value == 0
    assert isinstance(elts[1], ast.Slice) and isinstance(elts[2], ast.Slice)


# --------------------------------------------------------------------------- #
# C.10 np.tril mask (mirror of triu)                                          #
# --------------------------------------------------------------------------- #


def test_tril_registered():
    assert ("np", "tril") in NP_CALL_EXPANDERS


def test_tril_keeps_lower_triangle():
    out = _unparse(expand_tril(_name("out"), [_name("a")], {"a": ("M", "M")}))
    # lower triangle keeps ``j <= i`` (the complement of triu's ``j >= i``).
    assert "__j <= __i" in out
    assert "a[__i, __j]" in out and "else 0.0" in out


def test_triu_keeps_upper_triangle():
    out = _unparse(expand_triu(_name("out"), [_name("a")], {"a": ("M", "M")}))
    assert "__j >= __i" in out


def test_linalg_norm_ord1_inf_vector_and_matrix():
    """np.linalg.norm ord=1 -> sum|v| (vector) / max column abs-sum (matrix),
    ord=inf -> max|v| / max row abs-sum, all without sqrt. A POSITIONAL ord must
    not be misread as ``axis`` (the pre-fix bug returned the L2 norm); an
    unsupported ord or a >2-D operand raises rather than miscompute."""
    vec = {"a": ("N", )}
    l1 = _unparse(expand_linalg_norm(_name("s"), [_name("a"), ast.Constant(value=1)], vec))
    assert "abs(" in l1 and "sqrt" not in l1  # sum of |v|, not sqrt(sum v^2)
    # ``np.inf`` reaches the expander as the lowered Name("INFINITY").
    linf = _unparse(expand_linalg_norm(_name("s"), [_name("a"), _name("INFINITY")], vec))
    assert "abs(" in linf and "sqrt" not in linf and " if " in linf  # running max via IfExp
    mat = _unparse(expand_linalg_norm(_name("s"), [_name("a"), ast.Constant(value=1)], {"a": ("M", "N")}))
    assert "abs(" in mat and " if " in mat and "sqrt" not in mat  # max over per-column abs-sums
    with pytest.raises(NotImplementedError):  # unsupported ord
        expand_linalg_norm(_name("s"), [_name("a"), ast.Constant(value=3)], vec)
    with pytest.raises(NotImplementedError):  # >2-D operand not supported
        expand_linalg_norm(_name("s"), [_name("a"), ast.Constant(value=1)], {"a": ("M", "N", "P")})


# --------------------------------------------------------------------------- #
# Numerical oracle: emit + compile + run each op, compare vs numpy.            #
# --------------------------------------------------------------------------- #


def _oracle():
    import shutil
    if not (shutil.which("gcc") and shutil.which("gfortran") and shutil.which("g++")):
        pytest.skip("gcc/g++/gfortran needed for the native numerical check")
    import numpy as np  # noqa: F401
    try:
        import _op_oracle  # tests/ is on sys.path under pytest's rootdir
    except ImportError:
        import importlib.util
        import pathlib
        spec = importlib.util.spec_from_file_location("_op_oracle",
                                                      pathlib.Path(__file__).resolve().parent / "_op_oracle.py")
        _op_oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_op_oracle)
    return _op_oracle


# Full backend matrix: native c/c++/fortran + numba + pythran + jax. A backend that cannot
# lower a given op reports ``skip`` (accepted, per the reference test_slice_mask); only a real
# FAIL fails the test. Every kernel below is a single numpy-call expression (no python loop),
# so jax lowers to a ``jnp.*`` library call, never a forked ``lax.while_loop`` -- no hang risk.
_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")


def _assert_ok(status, label):
    fails = {b: s for b, s in status.items() if s.startswith("FAIL")}
    assert not fails, f"{label}: {fails}"


#: (id, numpy source, func, input shapes/arrays, output shape, syms, sym-shapes).
#: Output is read from the kernel's last param (an OUT buffer written in place).
@pytest.mark.parametrize("label,src,func,ins,out_shape,syms,shapes", [
    ("matmul_call", "import numpy as np\ndef f(a,b,out):\n    out[:] = np.matmul(a, b)\n", "f", [("a", (4, 6)),
                                                                                                 ("b", (6, 5))],
     (4, 5), {
         "M": 4,
         "K": 6,
         "N": 5
     }, {
         "a": "(M, K)",
         "b": "(K, N)",
         "out": "(M, N)"
     }),
    ("batched_both", "import numpy as np\ndef f(a,b,out):\n    out[:] = a @ b\n", "f", [("a", (3, 4, 6)),
                                                                                        ("b", (3, 6, 5))], (3, 4, 5), {
                                                                                            "NB": 3,
                                                                                            "M": 4,
                                                                                            "K": 6,
                                                                                            "N": 5
                                                                                        }, {
                                                                                            "a": "(NB, M, K)",
                                                                                            "b": "(NB, K, N)",
                                                                                            "out": "(NB, M, N)"
                                                                                        }),
    ("einsum_seissol", "import numpy as np\ndef f(g,h,c,out):\n    out[:] = np.einsum('dkl,blq,dqp->bkp', g, h, c)\n",
     "f", [("g", (2, 3, 4)), ("h", (2, 4, 3)), ("c", (2, 3, 2))], (2, 3, 2), {
         "DD": 2,
         "KK": 3,
         "LL": 4,
         "NB": 2,
         "QQ": 3,
         "PP": 2
     }, {
         "g": "(DD, KK, LL)",
         "h": "(NB, LL, QQ)",
         "c": "(DD, QQ, PP)",
         "out": "(NB, KK, PP)"
     }),
    ("trace", "import numpy as np\ndef f(a,out):\n    s = np.trace(a)\n    out[0] = s\n", "f", [("a", (5, 5))], (1, ), {
        "M": 5
    }, {
        "a": "(M, M)",
        "out": "(1,)"
    }),
    ("cumsum_axis1", "import numpy as np\ndef f(a,out):\n    out[:] = np.cumsum(a, axis=1)\n", "f", [("a", (4, 5))],
     (4, 5), {
         "M": 4,
         "N": 5
     }, {
         "a": "(M, N)",
         "out": "(M, N)"
     }),
    ("median_even", "import numpy as np\ndef f(a,out):\n    s = np.median(a)\n    out[0] = s\n", "f", [("a", (8, ))],
     (1, ), {
         "N": 8
     }, {
         "a": "(N,)",
         "out": "(1,)"
     }),
    # Partial-slice cumsum target (DBCSR row/col offsets):
    # ``out[1:] = np.cumsum(a)`` writes the prefix sums shifted by one, with
    # ``out[0]`` left at its initialised value. Exercises the offset-aware
    # cumulative expander end to end.
    ("cumsum_slice_target",
     "import numpy as np\ndef f(a,out):\n    out[0] = 0.0\n    out[1:] = np.cumsum(a)\n", "f", [("a", (6, ))], (7, ), {
         "N": 6,
         "M": 7
     }, {
         "a": "(N,)",
         "out": "(M,)"
     }),
    ("roll", "import numpy as np\ndef f(a,out):\n    out[:] = np.roll(a, 3)\n", "f", [("a", (8, ))], (8, ), {
        "N": 8
    }, {
        "a": "(N,)",
        "out": "(N,)"
    }),
    ("reshape_method", "import numpy as np\ndef f(a,out):\n    out[:] = a.reshape(3, 4)\n", "f", [("a", (12, ))],
     (3, 4), {
         "N": 12
     }, {
         "a": "(N,)",
         "out": "(3, 4)"
     }),
    ("fancy_gather_2d", "import numpy as np\ndef f(xe,idx,out):\n    out[:] = xe[:, idx]\n", "f", [("xe", (4, 6)),
                                                                                                   ("idx", "IDX")],
     (4, 4), {
         "M": 4,
         "N": 6,
         "K": 4
     }, {
         "xe": "(M, N)",
         "idx": "(K,)",
         "out": "(M, K)"
     }),
    ("ellipsis", "import numpy as np\ndef f(a,out):\n    out[:] = a[..., 0]\n", "f", [("a", (3, 4, 5))], (3, 4), {
        "M": 3,
        "N": 4,
        "P": 5
    }, {
        "a": "(M, N, P)",
        "out": "(M, N)"
    }),
    ("tril", "import numpy as np\ndef f(a,out):\n    out[:] = np.tril(a)\n", "f", [("a", (5, 5))], (5, 5), {
        "M": 5
    }, {
        "a": "(M, M)",
        "out": "(M, M)"
    }),
    ("norm_l1", "import numpy as np\ndef f(a,out):\n    out[0] = np.linalg.norm(a, 1)\n", "f", [("a", (6, ))], (1, ), {
        "N": 6
    }, {
        "a": "(N,)",
        "out": "(1,)"
    }),
    ("norm_linf", "import numpy as np\ndef f(a,out):\n    out[0] = np.linalg.norm(a, np.inf)\n", "f", [("a", (6, ))],
     (1, ), {
         "N": 6
     }, {
         "a": "(N,)",
         "out": "(1,)"
     }),
],
                         ids=lambda v: v if isinstance(v, str) and v.isidentifier() else "")
def test_contraction_indexing_ops_e2e(label, src, func, ins, out_shape, syms, shapes):
    import numpy as np
    no = _oracle()
    rng = np.random.default_rng(0)
    inputs = {}
    for nm, sh in ins:
        if sh == "IDX":
            inputs[nm] = np.array([0, 2, 4, 1], dtype=np.int64)
        else:
            inputs[nm] = rng.random(sh)
    status = no.run_op(src, func, inputs, {"out": out_shape}, syms, shapes=shapes, backends=_ALL)
    _assert_ok(status, label)


# --------------------------------------------------------------------------- #
# Reshape order= (C vs F): expand_reshape must honour column-major reshape so   #
# QE vexx_k's order="F" FFT band-pair reshapes lower correctly.                #
# --------------------------------------------------------------------------- #


def _reshape_src_index(order):
    """Lower ``out = np.reshape(A, (P, Q), order=order)`` for A:(N,), out:(P,Q)
    and return the unparsed source subscript expression A[...]."""
    target = ast.Name(id="out", ctx=ast.Store())
    args = [_name("A")]
    shape_table = {"A": ("N", ), "out": ("P", "Q")}
    kwargs = ([ast.keyword(arg="order", value=ast.Constant(value=order))] if order else None)
    stmts = expand_reshape(target, args, shape_table, kwargs=kwargs)
    txt = _unparse(stmts)
    # The source read is ``A[<expr>]``; grab <expr>.
    return txt.split("A[", 1)[1].split("]", 1)[0]


def test_reshape_c_order_row_major_index():
    # C order: flat = r0 * Q + r1 -> A[(__r0) * Q + __r1]; Q scales the row.
    idx = _reshape_src_index("C")
    assert "* Q" in idx or ") * (Q)" in idx
    assert "* P" not in idx


def test_reshape_f_order_column_major_index():
    # F order: flat = r0 + r1 * P -> A[__r0 + (__r1) * P]; P scales the column.
    idx = _reshape_src_index("F")
    assert "* P" in idx or ") * (P)" in idx
    assert "* Q" not in idx


def test_reshape_default_is_c_order():
    # No order= kwarg defaults to C (matches numpy + prior behaviour).
    assert _reshape_src_index(None) == _reshape_src_index("C")
