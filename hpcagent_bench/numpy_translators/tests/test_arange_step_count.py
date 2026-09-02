"""``np.arange`` element count under a step -- including a NEGATIVE one.

The count was ``(stop - start + step - 1) // step``, which is a positive-step identity, and the
array holding the result was sized ``stop - start``, which ignores the step entirely. For
``np.arange(10, 0, -1)`` that is a count of 12 in an array of size -10:

* C declared ``int64_t t[-10]`` and did not compile;
* Fortran took the negative bound as an empty array, ran the 12-iteration loop off the end of it,
  and returned garbage -- which the oracle graded as a pass, because the values it compared were
  whatever was on the stack.

Both now come from one :func:`arange_count`, so the extent an array is declared with and the trip
count the loop runs cannot disagree. A step > 1 is here too: it was over-allocating (count
``stop - start`` for ``stop - start`` / step elements), which is wasteful rather than wrong, but
the same expression fixes it.
"""

import numpy as np
import pytest

from _op_oracle import run_op

_NATIVE = ("c", "cpp", "fortran")


def _run(expr, n):
    """``t = <expr>`` copied element-wise into an ``n``-long int64 output, on every native backend."""
    src = f"import numpy as np\ndef f(out):\n    t = {expr}\n    for i in range(out.shape[0]):\n        out[i] = t[i]\n"
    res = run_op(
        src, "f", {}, {"out": (n,)}, {"N": n}, shapes={"out": "(N,)"}, dtypes={"out": "int64"}, backends=_NATIVE
    )
    assert all(v == "ok" or v.startswith("skip") for v in res.values()), (expr, res)
    assert any(v == "ok" for v in res.values()), (expr, res)


@pytest.mark.parametrize(
    "expr,n",
    [
        ("np.arange(10, 0, -1)", 10),
        ("np.arange(0, -10, -2)", 5),
        ("np.arange(-3, -9, -3)", 2),
    ],
)
def test_negative_step_arange_matches_numpy(expr, n):
    assert len(eval(expr)) == n, "test's own expectation disagrees with numpy"  # noqa: S307
    _run(expr, n)


@pytest.mark.parametrize(
    "expr,n",
    [
        ("np.arange(0, 10, 2)", 5),
        ("np.arange(1, 10, 3)", 3),
        ("np.arange(0, 10)", 10),
        ("np.arange(7)", 7),
    ],
)
def test_positive_step_arange_still_matches_numpy(expr, n):
    assert len(eval(expr)) == n, "test's own expectation disagrees with numpy"  # noqa: S307
    _run(expr, n)


def test_count_is_folded_for_literal_bounds():
    """A literal arange must size its array with a plain integer: an expression there has to be
    evaluable in a Fortran declaration, where ``/`` truncates instead of flooring."""
    import ast

    from numpyto_common.lib_nodes import arange_count

    def count(text):
        return arange_count(ast.parse(text, mode="eval").body.args)

    for text, want in [
        ("np.arange(10, 0, -1)", 10),
        ("np.arange(0, 10, 2)", 5),
        ("np.arange(0, 10, -1)", 0),
        ("np.arange(5)", 5),
        ("np.arange(2, 9)", 7),
    ]:
        node = count(text)
        assert isinstance(node, ast.Constant), (text, ast.unparse(node))
        assert node.value == want, (text, node.value, want)
        assert node.value == len(eval(text))  # noqa: S307 -- numpy is the definition


def test_symbolic_bounds_keep_a_sign_correct_expression():
    """With a runtime bound there is nothing to fold, so the emitted form must be the ceil that
    holds for either sign -- not the positive-step-only identity."""
    import ast

    from numpyto_common.lib_nodes import arange_count

    node = arange_count(ast.parse("np.arange(a, b, s)", mode="eval").body.args)
    text = ast.unparse(node)
    assert "//" in text and text.startswith("-"), text
    for a, b, s in [(10, 0, -1), (0, 10, 2), (0, 10, 1), (3, 3, 1), (0, 10, -1)]:
        got = eval(text, {"a": a, "b": b, "s": s})  # noqa: S307
        assert got == len(np.arange(a, b, s)) or (got < 0 and len(np.arange(a, b, s)) == 0), (a, b, s, got)
