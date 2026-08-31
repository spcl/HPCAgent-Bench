"""A helper's MODE guard is a compare, not a bare flag, and it has to fuse just the same.

``_fuse_guarded_returns`` turns ``if FLAG: return A`` before a trailing ``return B`` into one
``return A if FLAG else B``, which is what makes an early-returning helper inlinable at all. It
only recognised ``flag`` and ``not flag``, so a two-state selector fused and a multi-state one did
not -- ``_logsumexp(keepdims=False)`` inlined while kl_div_loss' ``_kl_div``, whose three return
paths are guarded by ``reduction == 'batchmean'`` and ``reduction == 'sum'``, fused nothing,
matched no inlinable form, and emitted no DaCe program at all.

The compare is decidable for the same reason the flag is: the call site binds the parameter to a
literal at every site (:func:`_static_flag_params`), so substitution leaves two constants. It also
HAS to be decided here -- a string has no type in the C or Fortran parameter tables, so a mode that
survived to the backend could not be passed at all.

The load-bearing assertion is not that it emits. It is WHICH arm it emits: a fold that picks the
wrong branch still produces a program, and every value it computes is wrong.
"""
import ast

import numpy as np
import pytest

from _op_oracle import run_op
from numpyto_common.frontend import _is_static_flag_test

BACKENDS = ("c", "cpp", "fortran", "numba", "pythran", "jax")

#: Three return paths on a string selector, the shape kl_div_loss ships. The arms are deliberately
#: far apart numerically so a wrong pick cannot pass as round-off.
MODE_SRC = ("import numpy as np\n"
            "def _reduce(v, mode='mean'):\n"
            " if mode == 'total':\n"
            "  return np.sum(v)\n"
            " if mode == 'first':\n"
            "  return v[0]\n"
            " return np.sum(v) / v.shape[0]\n"
            "def f(x, out):\n"
            " out[0] = _reduce(x * 2.0, mode='total')\n")


def flags(*names):
    return frozenset(names)


def test_a_mode_compare_is_a_static_flag_test():
    """``mode == 'total'`` on a call-site literal is as decidable as a bare ``flag``."""
    test = ast.parse("mode == 'total'", mode="eval").body
    assert _is_static_flag_test(test, flags("mode"))
    assert _is_static_flag_test(ast.parse("mode != 'total'", mode="eval").body, flags("mode"))
    assert _is_static_flag_test(ast.parse("'total' == mode", mode="eval").body, flags("mode"))


@pytest.mark.parametrize(
    "expr,reason",
    [("mode == other", "neither side is a literal, so nothing is decided"),
     ("thing == 'total'", "the name is not a parameter every call site pins to a literal"),
     ("mode < 'total'", "an ordering compare is not a selector"),
     ("mode == 'a' == 'b'", "a chained compare has no single decidable pair")],
)
def test_an_undecidable_compare_is_not_a_static_flag_test(expr, reason):
    """Fusing an undecidable guard would leave an ``IfExp`` over ARRAY branches standing, which C's
    ``?:`` rejects outright and Fortran's ``merge`` evaluates on BOTH arms -- so a guarded division
    or subscript would run on exactly the values the guard exists to exclude."""
    assert not _is_static_flag_test(ast.parse(expr, mode="eval").body, flags("mode")), reason


def test_the_selected_arm_is_the_one_the_reference_takes():
    """Every backend, against the numpy reference's own answer. ``mode='total'`` selects the SUM,
    and the mean arm it must not select differs by a factor of eight here."""
    x = np.arange(1.0, 9.0)
    verdicts = run_op(MODE_SRC,
                      "f", {"x": x}, {"out": (1, )}, {"N": 8},
                      shapes={
                          "x": "(N,)",
                          "out": "(N,)"
                      },
                      backends=BACKENDS)
    assert all(v == "ok" or v.startswith("skip") for v in verdicts.values()), verdicts
