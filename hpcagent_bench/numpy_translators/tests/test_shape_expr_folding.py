"""Algebraic folding of shape-token expressions.

Inlining a helper's size locals wraps one more parenthesised layer per level, so a network whose
helpers nest five deep emits an extent hundreds of characters long at every loop bound and every
allocation -- densenet121's Fortran reached 10k lines and stopped compiling within the timeout.

Two things are checked, and the second matters more than the first. That the folder SHRINKS the
usual conv/pool output-size chains, and that it never changes what they EVALUATE to: every case is
re-evaluated against the unfolded form over a range of inputs, so a rewrite that happens to be
shorter but wrong fails here rather than as silent numerical noise three layers down.
"""
import ast
import itertools

import pytest
from numpyto_common.frontend import fold_shape_expr

#: (expression, expected folded form). Each is a real extent shape produced by the inliner.
CASES = [
    ("h + 0", "h"),
    ("h - 0", "h"),
    ("h * 1", "h"),
    ("1 * h", "h"),
    ("0 + h", "h"),
    ("h // 1", "h"),
    ("2 + 3", "5"),
    ("7 // 2", "3"),
    # The identity never fires until the chain's literals are gathered: this is the shape a
    # stride-1, pad-0, kernel-1 convolution layer produces, once per nesting level.
    ("(h + 0 - 1) // 1 + 1", "h"),
    ("(((h + 0 - 1) // 1 + 1) + 0 - 1) // 1 + 1", "h"),
    # Padding and kernel do not vanish, they combine: 2 * 3 - 7 is one literal, so the whole
    # pad/kernel adjustment of a stride-2 layer reduces to a single term.
    ("(h + 2 * 3 - 7) // 2 + 1", "(h - 1) // 2 + 1"),
    # A real four-deep densenet extent. Each ``//`` is opaque to the chain walk, so the divisions
    # stay exactly where they were and only the bookkeeping between them collapses.
    ("((((width + 6 - 7) // 2 + 1) + 2 - 3) // 2 + 1 + 0 - 1) // 1 + 1", "(width - 1) // 2 // 2 + 1"),
    # A numerator whose every non-constant term is a LITERAL multiple of the divisor: the division
    # is exact on all of them, so the leftover constant is all that is left to round.
    # raman_fitting allocates its jacobian ``3 * ((3 * K + 2) // 3) + 1`` and writes its columns
    # through slices sized ``K``; unfolded, the frontend had ``int_floor(3*K + 2, 3)`` on one side
    # and ``K`` on the other and could not prove them equal.
    ("(3 * K + 2) // 3", "K"),
    ("3 * ((3 * K + 2) // 3) + 1", "3 * K + 1"),
    ("(6 * K + 4) // 3", "2 * K + 1"),
    ("(4 * a + 2 * b + 3) // 2", "2 * a + b + 1"),
    # A unary minus belongs to the +/- chain, and terms spelled the same combine. cp2k spells one
    # length twice -- ``nrel = 2 * span + 1`` and the extent of ``np.arange(-span, span + 1)`` --
    # and apart they became two symbols nothing could prove equal.
    ("(span + 1) - (-span)", "2 * span + 1"),
    ("x + x", "2 * x"),
    ("a + b - a", "b"),
]

#: Numerators the rule above must NOT touch, with why. Sharper than the general
#: :func:`test_division_is_not_distributed` cases: each is one edit away from a case that DOES fold,
#: so they pin where the exact-multiple requirement actually sits.
NOT_EXACT = [
    ("(K + 2) // 3", "K carries no factor of 3"),
    ("(3 * K + 2) // 2", "3 * K is not a multiple of 2"),
    ("(3 * K + 2 * b) // 3", "2 * b is not a multiple of 3"),
    ("(K * K + 3) // 3", "a non-literal factor is not a multiple this rule can see"),
]


@pytest.mark.parametrize("expr,expected", CASES)
def test_folds_to_expected(expr, expected):
    assert fold_shape_expr(expr) == expected


@pytest.mark.parametrize("expr,_expected", CASES)
def test_folding_preserves_value(expr, _expected):
    """The folded form must agree with the original on every input, not just on a lucky one."""
    names = sorted({n.id for n in ast.walk(ast.parse(expr, mode="eval")) if isinstance(n, ast.Name)})
    folded = fold_shape_expr(expr)
    for combo in itertools.product(range(1, 12), repeat=len(names)):
        env = dict(zip(names, combo))
        assert eval(folded, {}, env) == eval(expr, {}, env), (expr, folded, env)


@pytest.mark.parametrize("expr,reason", NOT_EXACT)
def test_an_inexact_numerator_keeps_its_division(expr, reason):
    assert fold_shape_expr(expr) == expr, reason


@pytest.mark.parametrize("expr,_expected", CASES)
def test_folding_preserves_value_for_negative_operands_too(expr, _expected):
    """``//`` rounds toward -inf, so a rewrite can agree on every positive input and still be wrong.

    The extents this folder sees are sizes, but it is asked about them mid-expression, where a
    subtraction has already been folded in -- and the exact-multiple rule is stated over ALL
    integers, so this is the range it has to be checked on. A rule that only held for positives
    would pass the test above and fail here.
    """
    names = sorted({n.id for n in ast.walk(ast.parse(expr, mode="eval")) if isinstance(n, ast.Name)})
    folded = fold_shape_expr(expr)
    for combo in itertools.product(range(-9, 3), repeat=len(names)):
        env = dict(zip(names, combo))
        assert eval(folded, {}, env) == eval(expr, {}, env), (expr, folded, env)


def test_a_chain_that_cancels_completely_is_left_alone():
    """``a - a`` is 0, but rebuilding it as one needs a term to lead with and there is none. The
    folder returns the node untouched rather than inventing a literal -- an extent of 0 from a
    rewrite would be far worse than an unfolded one."""
    assert fold_shape_expr("a - a") == "a - a"


def test_shrinks_the_nested_form():
    deep = "((((width + 6 - 7) // 2 + 1) + 2 - 3) // 2 + 1 + 0 - 1) // 1 + 1"
    assert len(fold_shape_expr(deep)) < len(deep)


@pytest.mark.parametrize("expr", ["h", "arr.shape[0]", "n * m", "(h - 1) // 2 + 1"])
def test_already_minimal_is_left_alone(expr):
    """A token with nothing to gather must come back byte-identical -- the fold is not a reformat."""
    assert fold_shape_expr(expr) == expr


def test_unparseable_token_passes_through():
    """Shape tokens are strings from several producers; one that is not a Python expression is
    returned as-is rather than raising, since folding is an optimisation and not a validation."""
    assert fold_shape_expr("n +") == "n +"


@pytest.mark.parametrize("expr", ["(h + 2) // 2", "(h - 1) // 2 + 1", "h // 2 * 2", "(h + 3) % 4"])
def test_division_is_not_distributed(expr):
    """``//`` rounds toward -inf, so pushing a division through an add is wrong for any operand that
    is not an exact multiple. These must survive untouched however tempting they look."""
    assert fold_shape_expr(expr) == expr
