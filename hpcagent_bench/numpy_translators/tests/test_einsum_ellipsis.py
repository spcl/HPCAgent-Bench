"""``np.einsum`` with an ``...`` ellipsis: the ellipsis is expanded to explicit
index letters from each operand's rank (in ``expand_einsum``, which has the
shape table), then lowered by the existing contraction machinery.

The plain-subscript parser stays ellipsis-free -- its
``test_parse_einsum_ellipsis_unsupported`` guard is intact; only the expander
does the rank-aware expansion.
"""

import numpy as np
import pytest
from _op_oracle import run_op

from numpyto_common.lib_nodes import _expand_einsum_ellipsis

_NATIVE = ("c", "cpp", "fortran")


def _assert_ok(res, label):
    fails = {b: s for b, s in res.items() if not (s == "ok" or s.startswith("skip"))}
    assert not fails, f"{label}: {fails}"


def test_expand_einsum_ellipsis_to_explicit_letters():
    # One broadcast axis -> a fresh shared index letter; the output ellipsis
    # expands to the same letters (ellipsis-first, as numpy orders it).
    assert _expand_einsum_ellipsis("...ij->...ji", [3]) == "Aij->Aji"
    assert _expand_einsum_ellipsis("...ij,...jk->...ik", [3, 3]) == "Aij,Ajk->Aik"
    # Two broadcast axes (rank-4 operand, two explicit indices).
    assert _expand_einsum_ellipsis("...ij->...ji", [4]) == "ABij->ABji"


def test_expand_einsum_ellipsis_rejects_bad_forms():
    with pytest.raises(NotImplementedError):  # implicit output not supported with ellipsis
        _expand_einsum_ellipsis("...ij", [3])
    with pytest.raises(NotImplementedError):  # differing ellipsis ranks (broadcast) unsupported
        _expand_einsum_ellipsis("...ij,...jk->...ik", [4, 3])


def test_einsum_ellipsis_batched_transpose_e2e():
    rng = np.random.default_rng(0)
    src = "import numpy as np\ndef f(a, out):\n    out[:] = np.einsum('...ij->...ji', a)\n"
    a = rng.random((2, 3, 4))
    _assert_ok(
        run_op(
            src,
            "f",
            {"a": a},
            {"out": (2, 4, 3)},
            {"B": 2, "M": 3, "N": 4},
            shapes={"a": "(B, M, N)", "out": "(B, N, M)"},
            backends=_NATIVE,
        ),
        "einsum-ellipsis-transpose",
    )


def test_einsum_ellipsis_batched_matmul_e2e():
    # Ellipsis WITH a contraction: '...ij,...jk->...ik' expands to 'Aij,Ajk->Aik'
    # (a batched GEMM). c/cpp only: the batched-einsum FORTRAN emit
    # non-deterministically types a size symbol REAL (~40% flaky, hash/order
    # dependent, reproduces on the EXPLICIT 'Bij,Bjk->Bik' too -- NOT the ellipsis
    # expansion). Root is the emit's size-symbol integer
    # classification, separate from the np.int64-in-constants hardening.
    rng = np.random.default_rng(0)
    src = "import numpy as np\ndef f(a, b, out):\n    out[:] = np.einsum('...ij,...jk->...ik', a, b)\n"
    a, b = rng.random((2, 3, 4)), rng.random((2, 4, 5))
    _assert_ok(
        run_op(
            src,
            "f",
            {"a": a, "b": b},
            {"out": (2, 3, 5)},
            {"B": 2, "M": 3, "K": 4, "N": 5},
            shapes={"a": "(B, M, K)", "b": "(B, K, N)", "out": "(B, M, N)"},
            backends=("c", "cpp"),
        ),
        "einsum-ellipsis-matmul",
    )


def test_const_coerces_numpy_scalar():
    """A numpy scalar must never reach an ast.Constant: under numpy 2.0 it
    unparse to ``np.int64(0)`` (breaking dace's sympy loop-range parse) and it
    fails the Fortran emit's ``isinstance(_, int)`` integer test. ``_const``
    coerces it to the plain Python value so every backend emits a bare ``0``."""
    from numpyto_common.lib_nodes import _const

    ci = _const(np.int64(0))
    assert type(ci.value) is int and ci.value == 0
    cf = _const(np.float64(1.5))
    assert type(cf.value) is float and cf.value == 1.5
    # plain Python values pass through untouched.
    assert type(_const(3).value) is int and type(_const(2.0).value) is float
