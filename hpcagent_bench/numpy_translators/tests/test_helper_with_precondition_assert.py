"""A precondition ``assert`` must not cost a helper its inlining.

Both the emitter and ``numpy_desugar`` already DROP an ``assert``: a kernel runs on
oracle-validated inputs, so the guard never fires and the message is a Python string the native
backends cannot express. The inlinable-helper classifier did not know that, and listed the
statements a spliceable body may hold without ``Assert`` among them.

Excluding it did not make the helper safer. It made it uninlinable, and an uninlined helper
survives as a CALL -- which a ``@dc.program`` cannot make, since it binds no helper. One
``assert groups == 1`` line was the whole reason conv_pointwise_2d emitted no DaCe program at all.
"""
import ast

import numpy as np
import pytest

from _op_oracle import run_op
from numpyto_common.frontend import INLINABLE_STMTS

BACKENDS = ("c", "cpp", "fortran", "numba", "pythran", "jax")

#: Form 3: statements, then a trailing ``return``, with the precondition first.
RETURNING_SRC = ("import numpy as np\n"
                 "def _scaled(x, factor):\n"
                 " assert factor > 0\n"
                 " doubled = x * 2.0\n"
                 " return doubled * factor\n"
                 "def f(x, out):\n"
                 " out[:] = _scaled(x, 3.0)\n")

#: Form 4: a void helper writing through its argument, with no return at all.
VOID_SRC = ("import numpy as np\n"
            "def _fill(x, out):\n"
            " assert out.shape[0] > 0\n"
            " out[:] = x + 1.0\n"
            "def f(x, out):\n"
            " _fill(x, out)\n")


def test_what_the_backends_drop_the_classifier_accepts():
    """Pins the rule, not one kernel: dropping a statement downstream and refusing to splice it
    upstream cannot both be right, and it was the refusal that cost whole kernels."""
    assert ast.Assert in INLINABLE_STMTS
    assert ast.Pass in INLINABLE_STMTS


@pytest.mark.parametrize("src", [RETURNING_SRC, VOID_SRC], ids=["returning", "void"])
def test_a_helper_guarded_by_an_assert_still_reaches_every_backend(src):
    """End to end, because the inlining is only half the claim: the spliced assert then has to
    LEAVE, or each native backend fails to compile a Python statement it cannot express."""
    x = np.arange(8.0)
    verdicts = run_op(src,
                      "f", {"x": x}, {"out": (8, )}, {"N": 8},
                      shapes={
                          "x": "(N,)",
                          "out": "(N,)"
                      },
                      backends=BACKENDS)
    assert all(v == "ok" or v.startswith("skip") for v in verdicts.values()), verdicts
