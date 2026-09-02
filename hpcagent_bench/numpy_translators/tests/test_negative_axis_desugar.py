"""Negative ``axis=`` -> positive index desugar for the python backends.

numpy counts a negative axis back from the last (``axis=-1`` == ``rank - 1``),
but some backend runtimes mishandle it -- pythran's ``np.flip`` / ``np.stack``
with ``axis=-1`` silently return the WRONG result. The native C/Fortran lowering
and the reduction desugar already normalize the axis, so only the verbatim-body
python backends are affected; ``_NormalizeNegativeAxis`` rewrites the literal to
the positive index it denotes (``rank + axis``) so every backend agrees. The
axis-space rank is the operand's rank for an axis-preserving op (flip/roll/
cumsum/concatenate/...) and operand rank + 1 for an axis-ADDING op (stack/
expand_dims).
"""

import ast
from types import SimpleNamespace

import numpy as np

from _op_oracle import run_op
from numpyto_common.numpy_desugar import _NormalizeNegativeAxis, desugar_for_python_backend


def _kir(kernel_name, **arrays):
    arrs = [SimpleNamespace(name=n, shape=s, dtype="float64") for n, s in arrays.items()]
    return SimpleNamespace(kernel_name=kernel_name, arrays=arrs)


def _desugar(body, **arrays):
    src = "import numpy as np\ndef f(a, b, out):\n" + body
    return desugar_for_python_backend(src, _kir("f", **arrays), backend="pythran")


def _norm_axis(expr, ranks):
    """Run ``_NormalizeNegativeAxis`` over one expression; return the axis kwarg
    value (an int) after normalization, or the raw node when left verbatim."""
    call = ast.parse(expr, mode="eval").body
    _NormalizeNegativeAxis(ranks).visit(call)
    kw = next(k for k in call.keywords if k.arg == "axis")
    return kw.value.value if isinstance(kw.value, ast.Constant) else kw.value


# --------------------------------------------------------------------------- #
# the rewrite, in isolation: axis-preserving vs axis-adding, rank sourcing     #
# --------------------------------------------------------------------------- #


def test_axis_preserving_uses_operand_rank():
    # rank-2 ``a`` -> flip axis -1 addresses axis 1, -2 addresses axis 0.
    assert _norm_axis("np.flip(a, axis=-1)", {"a": 2}) == 1
    assert _norm_axis("np.flip(a, axis=-2)", {"a": 2}) == 0
    assert _norm_axis("np.concatenate((a, b), axis=-1)", {"a": 2, "b": 2}) == 1


def test_axis_adding_op_adds_one_to_rank():
    # stack / expand_dims produce one MORE axis than the operand.
    assert _norm_axis("np.stack((a, b), axis=-1)", {"a": 2, "b": 2}) == 2  # axis-space 3
    assert _norm_axis("np.expand_dims(a, axis=-1)", {"a": 2}) == 2


def test_positive_axis_untouched():
    assert _norm_axis("np.flip(a, axis=1)", {"a": 2}) == 1  # already positive


def test_unknown_operand_rank_left_verbatim():
    # no rank for ``a`` -> the negative axis is NOT guessed (stays a UnaryOp).
    node = _norm_axis("np.flip(a, axis=-1)", {})
    assert isinstance(node, ast.UnaryOp)


def test_out_of_range_negative_left_verbatim():
    # -3 on a rank-2 operand would wrap negative -> our rank estimate is off, so
    # leave it verbatim rather than emit a wrong axis.
    node = _norm_axis("np.flip(a, axis=-3)", {"a": 2})
    assert isinstance(node, ast.UnaryOp)


# --------------------------------------------------------------------------- #
# end to end through the desugar (flip further lowers to a reverse-step slice) #
# --------------------------------------------------------------------------- #


def test_flip_negative_axis_lowers_to_reverse_slice():
    # normalized to axis 1, the existing flip pass turns it into ``a[:, ::-1]``
    # (it previously BAILED on ``axis=-1`` -- a UnaryOp, not a Constant -- leaving
    # ``np.flip(a, axis=-1)`` for pythran to mishandle).
    out = _desugar(" out[:] = np.flip(a, axis=-1)\n", a=("M", "N"), b=("M", "N"), out=("M", "N"))
    assert "[:, ::-1]" in out and "flip" not in out


def test_stack_negative_axis_normalized_in_place():
    # stack has no lowering pass -> it stays ``np.stack`` but with the axis fixed.
    out = _desugar(" out[:] = np.stack((a, b), axis=-1)\n", a=("M", "N"), b=("M", "N"), out=("M", "N", "2"))
    assert "axis=2)" in out and "axis=-1" not in out


def test_positive_stack_axis_returned_verbatim():
    # nothing lowers a positive-axis stack -> source is returned byte-for-byte.
    src = "import numpy as np\ndef f(a, b, out):\n out[:] = np.stack((a, b), axis=1)\n"
    assert (
        desugar_for_python_backend(src, _kir("f", a=("M", "N"), b=("M", "N"), out=("M", "N", "2")), backend="pythran")
        == src
    )


# --------------------------------------------------------------------------- #
# numerical: pythran now matches numpy for negative-axis flip / stack          #
# --------------------------------------------------------------------------- #


def _pythran_ok(res):
    return res["pythran"] in ("ok",) or res["pythran"].startswith("skip"), res


def test_flip_negative_axis_pythran_bit_exact():
    a = np.arange(24, dtype=np.float64).reshape(4, 6)
    res = run_op(
        "import numpy as np\ndef f(a, out):\n out[:] = np.flip(a, axis=-1)\n",
        "f",
        {"a": a},
        {"out": (4, 6)},
        {"M": 4, "N": 6},
        shapes={"a": "(M, N)", "out": "(M, N)"},
        backends=("pythran",),
    )
    assert res["pythran"] == "ok", res


def test_stack_negative_axis_pythran_bit_exact():
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(6, 12, dtype=np.float64).reshape(2, 3)
    res = run_op(
        "import numpy as np\n"
        "def k(a, b, out):\n"
        "    c = np.stack((a, b), axis=-1)\n"
        "    for i in range(out.shape[0]):\n"
        "        for j in range(out.shape[1]):\n"
        "            for l in range(out.shape[2]):\n"
        "                out[i, j, l] = c[i, j, l]\n",
        "k",
        {"a": a, "b": b},
        {"out": (2, 3, 2)},
        {"M": 2, "N": 3},
        shapes={"a": "(M, N)", "b": "(M, N)", "out": "(M, N, 2)"},
        backends=("pythran",),
    )
    assert res["pythran"] == "ok", res
