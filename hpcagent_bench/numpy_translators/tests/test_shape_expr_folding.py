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
