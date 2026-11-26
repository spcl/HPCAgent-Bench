# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""AST-level unit tests for the contraction / scan / indexing op batch.

Each test asserts the expander (or lowering rewriter) emits the expected loop
structure or intrinsic, so a regression points straight at the cause. The
end-to-end numerical correctness across c / cpp / fortran / numba is
covered by ``test_contraction_indexing_ops_e2e`` via the standalone oracle.
"""

import ast
import types
from types import SimpleNamespace

import pytest

from hpcagent_bench.translators.numpyto_common.lib_nodes import (
    dims_agree,
    expand_einsum,
    expand_inner,
    expand_linalg_norm,
    expand_reshape,
    expand_tensordot,
    expand_vdot,
    matmul_result_shape,
    parse_einsum_subscripts,
    shape_exprs_differ_numerically,
    shape_exprs_equal,
    sympify_shape,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar import desugar_for_python_backend
from tests.translators import op_oracle


def name_(n: str) -> ast.Name:
    return ast.Name(id=n, ctx=ast.Load())


def unparse_(stmts: list[ast.stmt]) -> str:
    mod = ast.fix_missing_locations(ast.Module(body=list(stmts), type_ignores=[]))
    return ast.unparse(mod)


# A.1  np.matmul call-form normalizes to the ``@`` BinOp                       #


# A.2  batched matmul result-shape (both-batched is the new case)              #


@pytest.mark.parametrize(
    ("a", "b", "aliases", "expected"),
    [
        pytest.param(("B", "M", "K"), ("K", "N"), None, ("B", "M", "N"), id="left-batched"),
        pytest.param(("M", "K"), ("B", "K", "N"), None, ("B", "M", "N"), id="right-batched"),
        # Different leading batch dims do not contract.
        pytest.param(("B", "M", "K"), ("C", "K", "N"), None, None, id="batch-mismatch"),
        pytest.param(
            ("seq", "batch", "channels"), ("embed_dim", "3 * embed_dim"), None, None, id="unaliased-contraction-dim"
        ),
        pytest.param(
            ("seq", "batch", "channels"),
            ("embed_dim", "3 * embed_dim"),
            {"channels": "embed_dim"},
            ("seq", "batch", "3 * embed_dim"),
            id="aliased-contraction-dim",
        ),
        # An alias must not let a rank-3 operand contract with a rank-4 one.
        pytest.param(
            ("batch", "M", "K"),
            ("batch_size", "H", "K", "N"),
            {"batch": "batch_size"},
            None,
            id="aliased-batch-dim-still-checks-rank",
        ),
    ],
)
def test_matmul_result_shape_one_sided_batched(
    a: tuple[str, ...], b: tuple[str, ...], aliases: dict[str, str] | None, expected: tuple[str, ...] | None
) -> None:
    assert matmul_result_shape(a, b, aliases) == expected


# A.2b shape tokens from two vocabularies still name one extent                #


@pytest.mark.parametrize(
    ("a", "b", "aliases", "table", "expected"),
    [
        # ``batch, channels, h, w = x.shape`` binds locals the init.shapes side spells with symbols.
        pytest.param(
            "channels", "embed_dim", {"channels": "embed_dim", "batch": "batch_size"}, None, True, id="alias-1"
        ),
        pytest.param("batch", "batch_size", {"channels": "embed_dim", "batch": "batch_size"}, None, True, id="alias-2"),
        pytest.param(
            "channels", "batch_size", {"channels": "embed_dim", "batch": "batch_size"}, None, False, id="alias-differs"
        ),
        # swin's patch-merge doubles a stage's channels: the two sides stay textually different
        # after substitution, and only the symbolic rung settles them.
        pytest.param("4 * c", "16 * embed_dim", {"c": "4 * embed_dim"}, None, True, id="symbolic-agrees"),
        pytest.param("4 * c", "15 * embed_dim", {"c": "4 * embed_dim"}, None, False, id="symbolic-differs"),
        # An inlined helper spells its dims as a read off a LOCAL array, which no alias resolves.
        pytest.param(
            "__inl91_c",
            "embed_dim",
            {"__inl91_c": "__inl8_y.shape[3]"},
            {"__inl8_y": ("b", "h", "w", "embed_dim")},
            True,
            id="shape-read-agrees",
        ),
        pytest.param(
            "__inl91_c",
            "h",
            {"__inl91_c": "__inl8_y.shape[3]"},
            {"__inl8_y": ("b", "h", "w", "embed_dim")},
            False,
            id="shape-read-differs",
        ),
        # Unresolvable must decline, never claim agreement: a wrong True contracts over two extents.
        pytest.param("a b c", "embed_dim", {"zz": "1"}, None, False, id="unparseable-token"),
    ],
)
def test_dims_agree_through_a_dimension_alias(
    a: str, b: str, aliases: dict[str, str], table: dict[str, tuple[str, ...]] | None, expected: bool
) -> None:
    assert bool(dims_agree(a, b, aliases, table)) is expected


def no_simplify(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``sympy.simplify`` fatal, so a test can pin which rung settled a pair."""
    import sympy

    def explode(*args_: object, **kwargs_: object) -> None:
        raise AssertionError("sympy.simplify was reached")

    monkeypatch.setattr(sympy, "simplify", explode)
    shape_exprs_equal.cache_clear()


@pytest.mark.parametrize(
    ("a", "b", "aliases", "expected"),
    [
        # An expanded polynomial is canonical, so a non-zero expansion is already the answer. simplify
        # was measured at 785 ms on one of these pairs.
        pytest.param("(c + 6 * g) * h * w", "c * h * w + 6 * g * h * w", {"zz": "1"}, True, id="equal-polynomials"),
        pytest.param("4 * c", "15 * embed_dim", {"c": "4 * embed_dim"}, False, id="unequal-polynomials"),
        # The dominant shape in the ML track: a floor makes the difference non-polynomial, so only the
        # numeric point settles it short of simplify.
        pytest.param(
            "n * ((h - 2) // 1 + 1) * ((w - 2) // 1 + 1)", "n * oh * ow", {"zz": "1"}, False, id="conv-extent"
        ),
    ],
)
def test_a_polynomial_difference_never_reaches_simplify(
    monkeypatch: pytest.MonkeyPatch, a: str, b: str, aliases: dict[str, str], expected: bool
) -> None:
    no_simplify(monkeypatch)
    assert bool(dims_agree(a, b, aliases)) is expected


@pytest.mark.parametrize(
    ("a", "b", "refuted"),
    [
        # ``2 * a`` and ``a + 7`` both give 14 at a = 7, which is why one point is not enough.
        pytest.param("2 * a", "a + 7", True, id="agrees-at-the-first-point"),
        # The refutation may only ever answer False faster -- never claim a difference that is not there.
        pytest.param("4 * (4 * e)", "16 * e", False, id="equal-pair"),
        # A floor on both sides is where a careless probe would round its way to a false refutation.
        pytest.param("2 * (c // g)", "(c // g) + (c // g)", False, id="equal-floors"),
    ],
)
def test_the_numeric_probe_refutes_a_pair_that_agrees_at_the_first_point(a: str, b: str, refuted: bool) -> None:
    import sympy

    assert bool(shape_exprs_differ_numerically(sympy.sympify(a), sympy.sympify(b))) is refuted


@pytest.mark.parametrize(
    ("calls", "misses", "hits"),
    [
        # The symbolic compare is 61% of a densenet lowering uncached, and reshape_axis_groups asks
        # about the same pair once per axis of every reshape.
        pytest.param([("4 * c", "16 * embed_dim")] * 3, 1, 2, id="repeated-pair"),
        # A zero difference is symmetric, so the mirrored pair must not take a second entry.
        pytest.param([("4 * c", "16 * embed_dim"), ("16 * embed_dim", "4 * c")], 1, 1, id="mirrored-pair"),
    ],
)
def test_the_symbolic_rung_is_asked_about_each_pair_once(calls: list[tuple[str, str]], misses: int, hits: int) -> None:
    shape_exprs_equal.cache_clear()
    aliases = {"c": "4 * embed_dim"}
    for a, b in calls:
        assert dims_agree(a, b, aliases)
    info = shape_exprs_equal.cache_info()
    assert (info.misses, info.hits) == (misses, hits)


def test_sympify_shape_declines_a_token_that_does_not_parse() -> None:
    # None is what makes dims_agree answer False instead of raising out of the expander.
    assert sympify_shape("a b c") is None
    assert str(sympify_shape("4 * embed_dim")) == "4*embed_dim"


# A.3  einsum subscript parse + loop structure                                 #


def test_parse_einsum_implicit_output() -> None:
    # numpy implicit output = singly-occurring indices, alphabetical.
    assert parse_einsum_subscripts("ij,jk") == (["ij", "jk"], "ik")


def test_parse_einsum_ellipsis_unsupported() -> None:
    with pytest.raises(NotImplementedError):
        parse_einsum_subscripts("...ij->...i")


# A.4  batched (>=3-D) matmul desugaring for the verbatim Python backends      #
#      (numba cannot type stacked ``@``; lower to a loop of GEMMs)   #


def kir_(kernel_name: str, **arrays: tuple[str, ...]) -> SimpleNamespace:
    """Minimal KernelIR stand-in: name + (name -> shape-tuple) arrays."""
    arrs = [SimpleNamespace(name=n, shape=s, dtype=None) for n, s in arrays.items()]
    return SimpleNamespace(kernel_name=kernel_name, arrays=arrs, scalars=[])


def test_2d_matmul_left_verbatim() -> None:
    # An ordinary 2-D GEMM is supported by numba -> emit unchanged.
    src = "def kernel(C, A, B):\n    C[:] = A @ B\n"
    kir = kir_("kernel", C=("m", "n"), A=("m", "k"), B=("k", "n"))
    assert desugar_for_python_backend(src, kir) == src


def test_reshape_wrapped_matmul_lowers_to_contraction() -> None:
    # doitgen: ``np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))``.
    # The batched-matmul pass still refuses reshape-wrapped operands (indexing the
    # leading axis would miscompile), but the dedicated reshape-matmul pass
    # recognises the unit-dim-insertion form and lowers it to a contraction loop
    # into a fresh temp (so the ``A = f(A)`` WAR is safe).
    src = "def kernel(NR, NQ, NP, A, C4):\n    A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))\n"
    kir = kir_("kernel", A=("NR", "NQ", "NP"), C4=("NP", "NP"))
    out = desugar_for_python_backend(src, kir)
    assert "@" not in out
    assert "reshape" not in out
    assert "for " in out


def test_np_pad_edge_inlined_to_loop_nest() -> None:
    # numba cannot type np.pad -> inline an edge-pad copy loop nest.
    src = (
        "def kernel(in_grid, out_grid, N, R):\n"
        "    padded = np.pad(in_grid, pad_width=R, mode='edge')\n"
        "    out_grid[:] = padded[:N, :N, :N]\n"
    )
    kir = kir_("kernel", in_grid=("N", "N", "N"), out_grid=("N", "N", "N"))
    out = desugar_for_python_backend(src, kir)
    assert "np.pad" not in out, "np.pad must be expanded"
    assert "np.empty" in out
    assert out.count("for ") >= 3
    assert "min(max(" in out  # edge clamp


def test_einsum_trace_is_scalar_accumulation() -> None:
    st = {"a": ("M", "M")}
    out = unparse_(expand_einsum(name_("s"), [ast.Constant("ii->"), name_("a")], st))
    # No output letters -> scalar target, single summed loop over the diagonal.
    assert "s = 0.0" in out
    assert "s += a[__es_i, __es_i]" in out


# A.4  tensordot / inner / vdot                                                #


def test_tensordot_axes1_is_matmul_contraction() -> None:
    st = {"a": ("M", "K"), "b": ("K", "N")}
    out = unparse_(
        expand_tensordot(
            name_("out"), [name_("a"), name_("b")], st, kwargs=[ast.keyword(arg="axes", value=ast.Constant(1))]
        )
    )
    assert "out[__es_a, __es_c] +=" in out  # contracts the shared K axis


def test_inner_rank1_is_dot() -> None:
    st = {"u": ("K",), "v": ("K",)}
    out = unparse_(expand_inner(name_("s"), [name_("u"), name_("v")], st))
    assert "s = 0.0" in out
    assert "s += u[__r0] * v[__r0]" in out


@pytest.mark.parametrize(
    ("local_dtypes", "present", "absent"),
    [
        # Real operands: no conj() call (CONJG/__npb_conj is invalid on a real scalar).
        pytest.param({}, "s += u[__vd] * v[__vd]", "conj", id="real"),
        pytest.param({"u": "complex128"}, "np.conj(u[__vd])", None, id="complex-conjugates-first-operand"),
    ],
)
def test_vdot_real_no_conjugate(local_dtypes: dict[str, str], present: str, absent: str | None) -> None:
    st = {"u": ("K",), "v": ("K",)}
    out = unparse_(expand_vdot(name_("s"), [name_("u"), name_("v")], st, local_dtypes=local_dtypes))
    assert present in out
    if absent is not None:
        assert absent not in out


# B.5  trace / diagonal direct                                                 #


# B.6  cumsum / cumprod prefix scan                                            #


# B.7  median: copy + sort + pick middle                                       #


# C.8  np.roll modular index                                                   #


# C.9  reshape method-form + ellipsis expansion                                #


# C.10 np.tril mask (mirror of triu)                                          #


def test_linalg_norm_ord1_inf_vector_and_matrix() -> None:
    """np.linalg.norm ord=1 -> sum|v| (vector) / max column abs-sum (matrix),
    ord=inf -> max|v| / max row abs-sum, all without sqrt. A POSITIONAL ord must
    not be misread as ``axis`` (which returns the L2 norm); an
    unsupported ord or a >2-D operand raises rather than miscompute."""
    vec = {"a": ("N",)}
    l1 = unparse_(expand_linalg_norm(name_("s"), [name_("a"), ast.Constant(value=1)], vec))
    assert "abs(" in l1
    assert "sqrt" not in l1
    # ``np.inf`` reaches the expander as the lowered Name("INFINITY").
    linf = unparse_(expand_linalg_norm(name_("s"), [name_("a"), name_("INFINITY")], vec))
    assert "abs(" in linf
    assert "sqrt" not in linf
    assert " if " in linf
    mat = unparse_(expand_linalg_norm(name_("s"), [name_("a"), ast.Constant(value=1)], {"a": ("M", "N")}))
    assert "abs(" in mat
    assert " if " in mat
    assert "sqrt" not in mat
    with pytest.raises(NotImplementedError):  # unsupported ord
        expand_linalg_norm(name_("s"), [name_("a"), ast.Constant(value=3)], vec)
    with pytest.raises(NotImplementedError):  # >2-D operand not supported
        expand_linalg_norm(name_("s"), [name_("a"), ast.Constant(value=1)], {"a": ("M", "N", "P")})


# Numerical oracle: emit + compile + run each op, compare vs numpy.            #


def oracle() -> types.ModuleType:
    return op_oracle.native()


# Full backend matrix: native c/c++/fortran + numba. A backend that cannot
# lower a given op reports ``skip`` (accepted, per the reference test_slice_mask); only a real
# FAIL fails the test.
ALL = ("c", "cpp", "fortran", "numba")

#: Backends that must actually RUN an op, not skip it. A skip stays accepted -- a backend that
#: cannot lower an op should not fail the op's test -- but accepting skips is also how a green
#: test can mean NO backend ran the case: `np.triu` looked green here until the raw status dict
#: was printed. These three are the reference lowering, so if none of them ran, nothing was graded.
MUST_NOT_ALL_SKIP = ("c", "cpp", "fortran")


def assert_ok(status: dict[str, str], label: str) -> None:
    fails = {b: s for b, s in status.items() if s.startswith("FAIL")}
    assert not fails, f"{label}: {fails}"
    native = {b: status.get(b) for b in MUST_NOT_ALL_SKIP}
    ran = [b for b, s in native.items() if s == "ok"]
    assert ran, f"{label}: no native backend ran it, so this case graded NOTHING -- {native}"


#: (id, numpy source, func, input shapes/arrays, output shape, syms, sym-shapes).
#: Output is read from the kernel's last param (an OUT buffer written in place).
@pytest.mark.parametrize(
    ("label", "src", "func", "ins", "out_shape", "syms", "shapes"),
    [
        (
            "matmul_call",
            "import numpy as np\ndef f(a,b,out):\n    out[:] = np.matmul(a, b)\n",
            "f",
            [("a", (4, 6)), ("b", (6, 5))],
            (4, 5),
            {"M": 4, "K": 6, "N": 5},
            {"a": "(M, K)", "b": "(K, N)", "out": "(M, N)"},
        ),
        (
            "batched_both",
            "import numpy as np\ndef f(a,b,out):\n    out[:] = a @ b\n",
            "f",
            [("a", (3, 4, 6)), ("b", (3, 6, 5))],
            (3, 4, 5),
            {"NB": 3, "M": 4, "K": 6, "N": 5},
            {"a": "(NB, M, K)", "b": "(NB, K, N)", "out": "(NB, M, N)"},
        ),
        (
            "einsum_seissol",
            "import numpy as np\ndef f(g,h,c,out):\n    out[:] = np.einsum('dkl,blq,dqp->bkp', g, h, c)\n",
            "f",
            [("g", (2, 3, 4)), ("h", (2, 4, 3)), ("c", (2, 3, 2))],
            (2, 3, 2),
            {"DD": 2, "KK": 3, "LL": 4, "NB": 2, "QQ": 3, "PP": 2},
            {"g": "(DD, KK, LL)", "h": "(NB, LL, QQ)", "c": "(DD, QQ, PP)", "out": "(NB, KK, PP)"},
        ),
        (
            "trace",
            "import numpy as np\ndef f(a,out):\n    s = np.trace(a)\n    out[0] = s\n",
            "f",
            [("a", (5, 5))],
            (1,),
            {"M": 5},
            {"a": "(M, M)", "out": "(1,)"},
        ),
        (
            "cumsum_axis1",
            "import numpy as np\ndef f(a,out):\n    out[:] = np.cumsum(a, axis=1)\n",
            "f",
            [("a", (4, 5))],
            (4, 5),
            {"M": 4, "N": 5},
            {"a": "(M, N)", "out": "(M, N)"},
        ),
        (
            "median_even",
            "import numpy as np\ndef f(a,out):\n    s = np.median(a)\n    out[0] = s\n",
            "f",
            [("a", (8,))],
            (1,),
            {"N": 8},
            {"a": "(N,)", "out": "(1,)"},
        ),
        (
            "roll",
            "import numpy as np\ndef f(a,out):\n    out[:] = np.roll(a, 3)\n",
            "f",
            [("a", (8,))],
            (8,),
            {"N": 8},
            {"a": "(N,)", "out": "(N,)"},
        ),
        (
            "reshape_method",
            "import numpy as np\ndef f(a,out):\n    out[:] = a.reshape(3, 4)\n",
            "f",
            [("a", (12,))],
            (3, 4),
            {"N": 12},
            {"a": "(N,)", "out": "(3, 4)"},
        ),
        (
            "fancy_gather_2d",
            "import numpy as np\ndef f(xe,idx,out):\n    out[:] = xe[:, idx]\n",
            "f",
            [("xe", (4, 6)), ("idx", "IDX")],
            (4, 4),
            {"M": 4, "N": 6, "K": 4},
            {"xe": "(M, N)", "idx": "(K,)", "out": "(M, K)"},
        ),
        (
            "ellipsis",
            "import numpy as np\ndef f(a,out):\n    out[:] = a[..., 0]\n",
            "f",
            [("a", (3, 4, 5))],
            (3, 4),
            {"M": 3, "N": 4, "P": 5},
            {"a": "(M, N, P)", "out": "(M, N)"},
        ),
        (
            "tril",
            "import numpy as np\ndef f(a,out):\n    out[:] = np.tril(a)\n",
            "f",
            [("a", (5, 5))],
            (5, 5),
            {"M": 5},
            {"a": "(M, M)", "out": "(M, M)"},
        ),
        (
            "norm_l1",
            "import numpy as np\ndef f(a,out):\n    out[0] = np.linalg.norm(a, 1)\n",
            "f",
            [("a", (6,))],
            (1,),
            {"N": 6},
            {"a": "(N,)", "out": "(1,)"},
        ),
        (
            "norm_linf",
            "import numpy as np\ndef f(a,out):\n    out[0] = np.linalg.norm(a, np.inf)\n",
            "f",
            [("a", (6,))],
            (1,),
            {"N": 6},
            {"a": "(N,)", "out": "(1,)"},
        ),
    ],
    ids=lambda v: v if isinstance(v, str) and v.isidentifier() else "",
)
def test_contraction_indexing_ops_e2e(
    label: str,
    src: str,
    func: str,
    ins: list[tuple[str, tuple[int, ...] | str]],
    out_shape: tuple[int, ...],
    syms: dict[str, int],
    shapes: dict[str, str],
) -> None:
    import numpy as np

    no = oracle()
    rng = np.random.default_rng(0)
    inputs = {}
    for nm, sh in ins:
        if sh == "IDX":
            inputs[nm] = np.array([0, 2, 4, 1], dtype=np.int64)
        else:
            inputs[nm] = rng.random(sh)
    status = no.run_op(src, func, inputs, {"out": out_shape}, syms, shapes=shapes, backends=ALL)
    assert_ok(status, label)


def test_triu_of_inline_full_mask_e2e() -> None:
    """The transformer causal mask verbatim: an inline ``np.full`` under an inline
    ``np.triu``, inside a BinOp. Its own test rather than a row in the table above,
    because adding a row reflows the whole parametrize literal under the formatter.

    ``exp`` turns the -inf band into an exact 0, so masking the WRONG triangle (or
    none at all) is a whole-magnitude disagreement with numpy, not drift."""
    import numpy as np

    src = (
        "import numpy as np\n"
        "def f(a,out):\n"
        "    n = a.shape[0]\n"
        "    out[:] = np.exp(a + np.triu(np.full((n, n), -np.inf), 1))\n"
    )
    inputs = {"a": np.random.default_rng(0).random((5, 5))}
    status = oracle().run_op(
        src, "f", inputs, {"out": (5, 5)}, {"M": 5}, shapes={"a": "(M, M)", "out": "(M, M)"}, backends=ALL
    )
    assert_ok(status, "triu_of_inline_full_mask")


# Reshape order= (C vs F): expand_reshape must honour column-major reshape so   #
# QE vexx_k's order="F" FFT band-pair reshapes lower correctly.                #


def reshape_src_index(order: str | None) -> str:
    """Lower ``out = np.reshape(A, (P, Q), order=order)`` for A:(N,), out:(P,Q)
    and return the unparsed source subscript expression A[...]."""
    target = ast.Name(id="out", ctx=ast.Store())
    args = [name_("A")]
    shape_table = {"A": ("N",), "out": ("P", "Q")}
    kwargs = [ast.keyword(arg="order", value=ast.Constant(value=order))] if order else None
    stmts = expand_reshape(target, args, shape_table, kwargs=kwargs)
    txt = unparse_(stmts)
    # The source read is ``A[<expr>]``; grab <expr>.
    return txt.split("A[", 1)[1].split("]", 1)[0]


@pytest.mark.parametrize(
    ("order", "scaled", "unscaled"),
    [
        # C order: flat = r0 * Q + r1 -> A[(__r0) * Q + __r1]; Q scales the row.
        pytest.param("C", "Q", "P", id="C"),
        # F order: flat = r0 + r1 * P -> A[__r0 + (__r1) * P]; P scales the column.
        pytest.param("F", "P", "Q", id="F"),
        # No order= kwarg defaults to C (matches numpy + prior behaviour).
        pytest.param(None, "Q", "P", id="default-is-C"),
    ],
)
def test_reshape_c_order_row_major_index(order: str | None, scaled: str, unscaled: str) -> None:
    idx = reshape_src_index(order)
    assert f"* {scaled}" in idx or f") * ({scaled})" in idx
    assert f"* {unscaled}" not in idx
    assert idx == reshape_src_index(order or "C")


def test_sympify_shape_declines_a_read_past_the_end_of_a_tuple_token() -> None:
    # sympify EVALUATES the token, so an out-of-range read arrives as a bare IndexError from
    # inside sympy's parser -- uncaught, it aborted the whole reshape expander.
    assert sympify_shape("(out_channels, 1, 1)[3]") is None
    assert str(sympify_shape("(out_channels, 1, 1)[0]")) == "out_channels"


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        pytest.param("int_floor(a, b)", "a // b", True, id="int-floor-spelling"),
        pytest.param(
            "(h + 2 * p - k) // s + 1", "int_floor(h + 2 * p - k, s) + 1", True, id="conv-extent-int-floor-spelling"
        ),
        pytest.param("int_floor(2 * oh, 2)", "oh", True, id="int-floor-cancels-against-a-bare-extent"),
        pytest.param("(2 * oh) // 2", "oh", True, id="floordiv-cancels-against-a-bare-extent"),
        pytest.param("int_floor(a, b)", "a // c", False, id="different-divisor"),
        pytest.param("int_floor(a, b)", "a // (b + 1)", False, id="different-divisor-expression"),
    ],
)
def test_a_floordiv_extent_equals_the_int_floor_spelling_of_itself(a: str, b: str, expected: bool) -> None:
    """``a // b`` and ``int_floor(a, b)`` are one quantity in two vocabularies -- ``//`` is what a
    manifest and a numpy reference write, ``int_floor`` is what the C and dace sides name it. Left
    unnormalised, sympify makes the first a ``floor`` and the second an opaque Function, and a
    compare across the two declines for extents that are the same number.

    The normalisation goes TOWARD sympy's ``floor`` and not toward ``int_floor``: an integer symbol lets sympy
    cancel ``floor(2 * oh / 2)`` to ``oh``, and an opaque ``int_floor`` never cancels against anything.
    The negative controls: collapsing both spellings onto one head must not collapse the operands with
    them -- a wrong True there contracts over two different extents, which is a miscompile."""
    shape_exprs_equal.cache_clear()
    assert bool(shape_exprs_equal(a, b)) is expected
