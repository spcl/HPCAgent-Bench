"""Axis-reduction desugar for the numba / pythran (and dace) Python backends.

These backends run the numpy body (near-)verbatim, so a reduction they cannot
type -- ``keepdims=True`` (neither supports it), a tuple ``axis=(1, 2, 3)`` over
a >4-D array (numba cannot lower it), a negative ``axis=-1`` literal, or a
``np.sum`` axis form -- must be lowered to an explicit loop nest first. This
surfaced auditing the KernelBench ML kernels: softmax's stable form is
``x - max(x, axis=-1, keepdims=True)`` and conv/pool reduce a broadcast product
over a tuple axis. The AST-level tests pin the rewrite; the two numerical tests
confirm bit-exact agreement with numpy on numba + pythran.
"""
import ast
from types import SimpleNamespace

import numpy as np

from numpyto_common.numpy_desugar import (_axis_list, _const_int, _param_body_rank_evidence, _reduce_axis_stmts,
                                          desugar_for_python_backend)


def _kir(kernel_name, **arrays):
    """Minimal KernelIR stand-in: name + (name -> shape-tuple) arrays."""
    arrs = [SimpleNamespace(name=n, shape=s) for n, s in arrays.items()]
    return SimpleNamespace(kernel_name=kernel_name, arrays=arrs)


# --------------------------------------------------------------------------- #
# axis parsing: negative literals + tuple forms                               #
# --------------------------------------------------------------------------- #


def test_const_int_handles_negation():
    # ``-1`` parses as UnaryOp(USub, Constant(1)), not Constant(-1).
    assert _const_int(ast.parse("-1", mode="eval").body) == -1
    assert _const_int(ast.parse("2", mode="eval").body) == 2
    assert _const_int(ast.parse("True", mode="eval").body) is None  # bool is not an axis
    assert _const_int(ast.parse("x", mode="eval").body) is None


def test_axis_list_normalizes_negative_and_tuple():

    def parse(s):
        return ast.parse(s, mode="eval").body

    assert _axis_list(parse("-1"), 4) == [3]
    assert _axis_list(parse("(1, 2, 3)"), 5) == [1, 2, 3]
    assert _axis_list(parse("(-2, -1)"), 4) == [2, 3]
    # out of [-rank, rank): our rank estimate is wrong -> bail, do not wrap.
    assert _axis_list(parse("3"), 2) is None
    assert _axis_list(parse("(1, 2, 3)"), 2) is None


# --------------------------------------------------------------------------- #
# the rewrite: keepdims, sum, tuple axis                                       #
# --------------------------------------------------------------------------- #


def test_negative_axis_now_reduces():
    # Before the fix ``axis=-1`` was left verbatim (parsed as a UnaryOp, not a
    # constant), so numba/pythran saw the unsupported axis form.
    src = "def k(x, out):\n out[:] = np.max(x, axis=-1)\n"
    kir = _kir("k", x=("M", "N"), out=("M", ))
    got = desugar_for_python_backend(src, kir, backend="numba")
    assert "np.max(" not in got and "for " in got


def test_keepdims_keeps_a_size_one_dim():
    src = "def k(x, out):\n m = np.max(x, axis=-1, keepdims=True)\n out[:] = x - m\n"
    kir = _kir("k", x=("M", "N"), out=("M", "N"))
    got = desugar_for_python_backend(src, kir, backend="numba")
    # result allocated with a trailing size-1 axis, written at index [..., 0].
    assert ", 1)" in got and ", 0]" in got
    assert "np.max(" not in got


def test_sum_axis_desugars_with_accumulator():
    src = "def k(x, out):\n out[:] = np.sum(x, axis=1)\n"
    kir = _kir("k", x=("M", "N"), out=("M", ))
    got = desugar_for_python_backend(src, kir, backend="numba")
    assert "np.sum(" not in got and "+=" in got


def test_tuple_axis_reduces_over_all_named_axes():
    # conv/pool inner reduction: reduce a rank-4 slice over axes (1, 2).
    src = "def k(x, out):\n out[:] = np.max(x, axis=(1, 2))\n"
    kir = _kir("k", x=("A", "B", "C", "D"), out=("A", "D"))
    got = desugar_for_python_backend(src, kir, backend="numba")
    assert "np.max(" not in got
    # two nested reduction loops (one per reduced axis).
    assert got.count("_j1 in range") == 1 and got.count("_j2 in range") == 1


def test_full_reduction_without_keepdims_left_verbatim():
    # every axis reduced -> a scalar; the backend's own full ``np.sum(x)`` handles
    # it, and emitting ``tmp[] = ...`` would be a syntax error.
    src = "def k(x, out):\n out[0] = np.sum(x, axis=(0, 1))\n"
    kir = _kir("k", x=("M", "N"), out=("one", ))
    assert desugar_for_python_backend(src, kir, backend="numba") == src


def test_tuple_axis_argmax_refused():
    # numpy itself rejects a tuple axis for argmin/argmax -> leave verbatim.
    src = "def k(x, out):\n out[:] = np.argmax(x, axis=(1, 2))\n"
    kir = _kir("k", x=("A", "B", "C"), out=("A", ))
    assert desugar_for_python_backend(src, kir, backend="numba") == src


def test_reduce_axis_stmts_mean_divides_by_element_count():
    stmts = _reduce_axis_stmts("t", "s", "mean", [1, 2], rank=3, ctr=0)
    body = ast.unparse(ast.fix_missing_locations(ast.Module(body=stmts, type_ignores=[])))
    assert "/ (__rd0_d1 * __rd0_d2)" in body  # divisor is the product of reduced dims


# --------------------------------------------------------------------------- #
# body-usage rank evidence (fixes the reshaped-local poisoning)               #
# --------------------------------------------------------------------------- #


def test_param_body_rank_evidence_from_shape_and_subscript():
    fn = ast.parse("def h(x, w):\n a = x.shape[3]\n b = w[:, 0:1, 0:1, :]\n").body[0]
    ev = _param_body_rank_evidence(fn)
    assert ev["x"] == 4  # x.shape[3] -> rank >= 4
    assert ev["w"] == 4  # 4 non-newaxis index positions -> rank >= 4


def test_body_evidence_overrides_poisoned_callsite_rank():
    # A helper is called with a local that is reshaped to a smaller rank later,
    # poisoning the flow-insensitive call-site inference. The helper's own
    # ``x.shape[3]`` / tuple axis must still pin rank 4 so the tuple-axis reduce
    # lowers instead of wrapping the axes into a scalar over-reduction.
    src = ("def pool(x):\n"
           "    out = np.empty((x.shape[0], x.shape[3]), x.dtype)\n"
           "    out[:] = np.max(x[:, 0:2, 0:2, :], axis=(1, 2))\n"
           "    return out\n"
           "def kernel(img, out):\n"
           "    y = pool(img)\n"
           "    z = np.reshape(y, (y.shape[0] * y.shape[1],))\n"
           "    out[:] = z\n")
    kir = _kir("kernel", img=("N", "H", "W", "C"), out=("P", ))
    got = desugar_for_python_backend(src, kir, backend="numba")
    assert "np.max(" not in got  # the pooling reduce lowered (rank not poisoned to 2)
    assert "[]" not in got  # no scalar-index over-reduction was emitted


# --------------------------------------------------------------------------- #
# numerical: bit-exact vs numpy on numba (the reduction the raw njit rejects)  #
# --------------------------------------------------------------------------- #
# numba is the backend these reductions block (keepdims / tuple axis raise a
# TypingError when njit'd verbatim); the op-oracle now emits through the same
# desugar the real oracle uses, so ``ok`` here means genuinely-lowered-and-exact.


def _numba(src, ins, outs, syms, shapes):
    from _op_oracle import run_op
    res = run_op(src, "f", ins, outs, syms, shapes=shapes, backends=("numba", ))
    return res["numba"]


def test_softmax_keepdims_matches_numpy_on_numba():
    src = ("import numpy as np\n"
           "def f(x, out):\n"
           " m = np.max(x, axis=-1, keepdims=True)\n"
           " e = np.exp(x - m)\n"
           " out[:] = e / np.sum(e, axis=-1, keepdims=True)\n")
    x = np.linspace(-3.0, 3.0, 24).reshape(4, 6)
    assert _numba(src, {"x": x}, {"out": (4, 6)}, {"M": 4, "N": 6}, {"x": "(M, N)", "out": "(M, N)"}) == "ok"


def test_tuple_axis_pool_matches_numpy_on_numba():
    src = ("import numpy as np\n"
           "def f(x, out):\n"
           " out[:] = np.sum(x, axis=(1, 2))\n")
    x = np.arange(2 * 3 * 3 * 5, dtype=np.float64).reshape(2, 3, 3, 5)
    assert _numba(src, {"x": x}, {"out": (2, 5)}, {
        "A": 2,
        "B": 3,
        "C": 3,
        "D": 5
    }, {
        "x": "(A, B, C, D)",
        "out": "(A, D)"
    }) == "ok"
