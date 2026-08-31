"""A tuple-returning helper is spliced even when it computes locals first, or is called by a sibling.

A helper returning several values has no ABI to be called across -- C has one return slot -- so it is
spliced into each call site as one expression instead. Two things stopped that happening, and each
cost whole kernels their DaCe program rather than degrading anything visible:

* the splice needed a body that ONLY returns, and the useful helpers compute index locals first
  (``_tap_span`` binds ``offset``, ``rhs`` and its bounds, interleaved with the guards that bail);
* call sites were searched in the kernel body alone, and ``_tap_span`` is called only from
  ``_conv_transpose3d`` -- a sibling helper -- so the splice declined and left it a real function
  that nothing could reach.

The load-bearing test is the last one. Folding a chain of locals into three guarded returns is a
rewrite that is easy to get subtly wrong -- an off-by-one in a bound reads as a plausible program --
so the folded expression is evaluated against the original function over the whole parameter space.
"""
import ast
import itertools

import pytest

from numpyto_common.frontend import _folded_straight_line, _return_expression, _tuple_leaves

#: ``_tap_span``'s exact shape: locals INTERLEAVED with guarded returns, and a docstring first.
TAP_SPAN = ('def _tap_span(in_size, out_size, stride, padding, k):\n'
            '    """Valid input/output slice bounds for one kernel tap."""\n'
            '    offset = k - padding\n'
            '    iz_lo = 0 if offset >= 0 else (-offset + stride - 1) // stride\n'
            '    rhs = out_size - 1 - offset\n'
            '    if rhs < 0 or iz_lo >= in_size:\n'
            '        return iz_lo, iz_lo, 0, 0\n'
            '    iz_hi = min(in_size, rhs // stride + 1)\n'
            '    if iz_hi <= iz_lo:\n'
            '        return iz_lo, iz_lo, 0, 0\n'
            '    oz_lo = iz_lo * stride + offset\n'
            '    oz_hi = oz_lo + (iz_hi - iz_lo - 1) * stride + 1\n'
            '    return iz_lo, iz_hi, oz_lo, oz_hi\n')


def parse(src: str) -> ast.FunctionDef:
    return next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef))


def test_a_docstring_does_not_stop_the_fold():
    """The docstring is an ``Expr`` and it is the FIRST statement, so a fold that stopped at the
    first non-assignment folded nothing at all and every helper below looked unfoldable."""
    folded = _folded_straight_line(parse(TAP_SPAN).body)
    assert [type(s).__name__ for s in folded] == ["If", "If", "Return"]


def test_locals_interleaved_with_guards_are_all_folded():
    """``_tap_span`` bails, computes ``iz_hi``, bails again, then computes the bounds it returns.
    Folding only a LEADING run leaves the later locals bound to nothing the expression can see."""
    expr = _return_expression(_folded_straight_line(parse(TAP_SPAN).body))
    assert expr is not None
    assert len(_tuple_leaves(expr)) == 3  # one per return path
    assert all(isinstance(leaf, ast.Tuple) for leaf in _tuple_leaves(expr))


def test_a_rebound_local_declines():
    """One substitution cannot stand for two values, so a name bound twice refuses outright rather
    than folding whichever binding it saw last."""
    rebound = ("def f(a):\n"
               "    t = a + 1\n"
               "    t = t * 2\n"
               "    return t, t\n")
    assert _folded_straight_line(parse(rebound).body) is None


def test_a_guard_that_rebinds_a_folded_local_declines():
    """A local reassigned inside a branch is live differently on each path; substituting its first
    value into the reads after the branch would silently compute the wrong bound."""
    rebinding_guard = ("def f(a, c):\n"
                       "    t = a + 1\n"
                       "    if c:\n"
                       "        t = 0\n"
                       "    return t, t\n")
    assert _folded_straight_line(parse(rebinding_guard).body) is None


def test_a_statement_the_fold_cannot_express_declines():
    """Only assignments, guards and returns. A loop has no single-expression form, and guessing one
    would splice a body that does not run."""
    looping = ("def f(a, n):\n"
               "    t = a\n"
               "    for i in range(n):\n"
               "        t = t + i\n"
               "    return t, t\n")
    assert _folded_straight_line(parse(looping).body) is None


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_the_folded_expression_computes_what_the_helper_computes(stride):
    """The whole point. Every path through ``_tap_span`` -- both bail-outs and the live one -- over
    the parameter space a transposed convolution actually reaches. A fold that is merely shorter,
    or that gets one bound off by one, emits a plausible kernel that quietly writes the wrong
    slice, so agreement is checked rather than assumed."""
    fn = parse(TAP_SPAN)
    scope: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<tap>", "exec"), {"min": min}, scope)  # noqa: S102
    original = scope["_tap_span"]
    folded = ast.unparse(_return_expression(_folded_straight_line(fn.body)))
    for in_size, out_size, padding, k in itertools.product(range(1, 7), range(1, 9), range(0, 3), range(0, 4)):
        env = {"in_size": in_size, "out_size": out_size, "stride": stride, "padding": padding, "k": k}
        assert tuple(original(in_size, out_size, stride, padding, k)) == tuple(eval(folded, {"min": min}, env)), env
