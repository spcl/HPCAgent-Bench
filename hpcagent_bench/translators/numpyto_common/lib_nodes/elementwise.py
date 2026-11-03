"""Elementwise ufuncs (arithmetic, comparisons, logic, libm calls), clip and where."""

import ast
from collections.abc import Callable, Sequence

from hpcagent_bench.translators.numpyto_common.ast_build import name_
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import (
    all_integer_operands,
    Extent,
    as_float64,
    broadcast_extents,
    extent_is_one,
    iter_extent_of,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import cmp_, first_name, wrap_for_loops
from hpcagent_bench.translators.numpyto_common.lib_nodes.scalarize import scalarize_at_iters

__all__ = [
    "UNARY_C_MATH",
    "binary_call_expander",
    "decimals_of",
    "expand_add",
    "expand_clip",
    "expand_cos_arr",
    "expand_divide",
    "expand_elementwise",
    "expand_equal",
    "expand_exp_arr",
    "expand_greater",
    "expand_greater_equal",
    "expand_less",
    "expand_less_equal",
    "expand_log_arr",
    "expand_logical_and",
    "expand_logical_not",
    "expand_logical_or",
    "expand_maximum",
    "expand_minimum",
    "expand_multiply",
    "expand_negative",
    "expand_not_equal",
    "expand_power",
    "expand_round",
    "expand_sin_arr",
    "round_to_decimals",
    "expand_sqrt_arr",
    "expand_subtract",
    "expand_tanh",
    "expand_where",
    "unary_call_expander",
    "unary_elementwise",
    "unary_expr_expander",
]


def has_bounded_slice(node: ast.Subscript) -> bool:
    axes = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    return any(isinstance(e, ast.Slice) and (e.lower or e.upper) for e in axes)


def broadcast_extent(a: ast.expr, b: ast.expr, shape_table: dict[str, tuple[str, ...]]) -> Extent | None:
    """The iteration extent of a binary op: the numpy BROADCAST of both operands, not just the first.
    ``np.maximum(a(M,), B(N, M))`` iterates the full (N, M) output; ``scalarize_at_iters`` then indexes
    each operand against the full iter nest, reading a size-1/missing leading axis with a constant 0."""
    ea = iter_extent_of(a, shape_table)
    eb = iter_extent_of(b, shape_table)
    if ea is None or eb is None:
        return ea or eb
    for operand, own, other in ((a, ea, eb), (b, eb, ea)):
        stretched = any(
            extent_is_one(x) and not extent_is_one(y) for x, y in zip(reversed(own), reversed(other), strict=False)
        )
        if stretched and isinstance(operand, ast.Subscript) and has_bounded_slice(operand):
            raise NotImplementedError(f"{ast.unparse(operand)} broadcasts a sliced unit axis")
    return broadcast_extents(ea, eb)


def expand_elementwise(
    target: ast.Name,
    args: Sequence[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    op_fn: Callable[[ast.expr, ast.expr], ast.expr],
) -> list[ast.stmt]:
    """``out = op(a, b)`` -> per-element loop nest.

    Iteration extent comes from the first array-valued operand
    (Name or Subscript-with-Slice). Operands are scalarized via
    ``scalarize_at_iters`` so slice arguments
    (``np.maximum(A[i, :], B[i, :])``) lower as cleanly as bare
    Names. Scalars on either operand broadcast -- ``np.maximum(0, x)``
    or ``np.minimum(x, 0)`` are equally fine.
    """
    if len(args) != 2:
        raise NotImplementedError("elementwise needs 2 args")
    a, b = args
    extent = broadcast_extent(a, b, shape_table)
    if extent is None:
        raise NotImplementedError("elementwise: extent unknown for both args")
    iters = [name_(f"__r{i}") for i in range(len(extent))]

    # Constants / scalar Names broadcast; arrays scalarize.
    def maybe_scalar(node: ast.expr) -> ast.expr:
        if isinstance(node, ast.Constant):
            return node
        if isinstance(node, ast.Name) and not shape_table.get(node.id):
            return node
        return scalarize_at_iters(node, iters, shape_table)

    sa = maybe_scalar(a)
    sb = maybe_scalar(b)
    idx = iters[0] if len(iters) == 1 else ast.Tuple(elts=list(iters), ctx=ast.Load())
    body: list[ast.stmt] = [
        ast.Assign(targets=[ast.Subscript(value=name_(target.id), slice=idx, ctx=ast.Store())], value=op_fn(sa, sb))
    ]
    return wrap_for_loops([i.id for i in iters], extent, body)


def expand_minimum(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return expand_elementwise(t, a, s, lambda x, y: ast.Call(func=name_("min"), args=[x, y], keywords=[]))


def expand_maximum(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return expand_elementwise(t, a, s, lambda x, y: ast.Call(func=name_("max"), args=[x, y], keywords=[]))


def expand_add(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return expand_elementwise(t, a, s, lambda x, y: ast.BinOp(left=x, op=ast.Add(), right=y))


def expand_multiply(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return expand_elementwise(t, a, s, lambda x, y: ast.BinOp(left=x, op=ast.Mult(), right=y))


def expand_power(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    """``out = np.power(a, b)`` -> per-element ``a[i] ** b[i]``.

    A ``**`` BinOp, NOT a bare ``pow(...)`` call: each backend's ``**`` routing already
    dispatches on the operand type (C picks ``__npb_int_pow`` over libm's double ``pow``,
    Fortran's ``**`` is exact on integers). A ``pow`` Name call bypassed that and rounded
    every int64 result above 2**53 through a double."""
    return expand_elementwise(t, a, s, lambda x, y: ast.BinOp(left=x, op=ast.Pow(), right=y))


def expand_subtract(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return expand_elementwise(t, a, s, lambda x, y: ast.BinOp(left=x, op=ast.Sub(), right=y))


def expand_divide(
    target: ast.Name,
    args: Sequence[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    local_dtypes: dict[str, str] | None = None,
) -> list[ast.stmt]:
    """``out = np.divide(a, b)`` -> per-element TRUE division.

    numpy ``divide`` / ``true_divide`` is always true division (int64 / int64 -> float64),
    but this expander runs at libnode-expand -- AFTER lowering's ``TrueDivisionPromoter``
    phase, which never sees the ``Div`` synthesized here. So apply the promoter's own cast
    directly: an all-integer pair gets its left operand wrapped in ``np.float64(...)``,
    which both native emitters render as a floating divide. Anything not PROVABLY integer
    is left alone (an unwarranted cast would force fp64 into an fp32 kernel)."""
    int_div = all_integer_operands(args, local_dtypes)

    def op_fn(x: ast.expr, y: ast.expr) -> ast.expr:
        return ast.BinOp(left=as_float64(x) if int_div else x, op=ast.Div(), right=y)

    return expand_elementwise(target, args, shape_table, op_fn)


def expand_less(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return expand_elementwise(t, a, s, cmp_(ast.Lt))


def expand_less_equal(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return expand_elementwise(t, a, s, cmp_(ast.LtE))


def expand_greater(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return expand_elementwise(t, a, s, cmp_(ast.Gt))


def expand_greater_equal(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return expand_elementwise(t, a, s, cmp_(ast.GtE))


def expand_equal(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return expand_elementwise(t, a, s, cmp_(ast.Eq))


def expand_not_equal(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return expand_elementwise(t, a, s, cmp_(ast.NotEq))


def expand_logical_and(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return expand_elementwise(t, a, s, lambda x, y: ast.BoolOp(op=ast.And(), values=[x, y]))


def expand_logical_or(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return expand_elementwise(t, a, s, lambda x, y: ast.BoolOp(op=ast.Or(), values=[x, y]))


def expand_logical_not(
    target: ast.Name, args: Sequence[ast.expr], shape_table: dict[str, tuple[str, ...]]
) -> list[ast.stmt]:
    """``out = np.logical_not(a)`` -> per-element ``out[i] = not a[i]``."""
    return unary_elementwise(target, args, shape_table, lambda x: ast.UnaryOp(op=ast.Not(), operand=x))


def expand_negative(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    """``out = np.negative(a)`` -> ``out[i] = -a[i]``."""
    if first_name(a) is None:
        raise NotImplementedError("np.negative needs a Name arg")
    return unary_elementwise(t, a, s, lambda x: ast.UnaryOp(op=ast.USub(), operand=x))


def expand_tanh(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return unary_elementwise(t, a, s, lambda x: ast.Call(func=name_("tanh"), args=[x], keywords=[]))


def expand_sin_arr(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return unary_elementwise(t, a, s, lambda x: ast.Call(func=name_("sin"), args=[x], keywords=[]))


def expand_cos_arr(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return unary_elementwise(t, a, s, lambda x: ast.Call(func=name_("cos"), args=[x], keywords=[]))


def expand_exp_arr(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return unary_elementwise(t, a, s, lambda x: ast.Call(func=name_("exp"), args=[x], keywords=[]))


def expand_log_arr(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return unary_elementwise(t, a, s, lambda x: ast.Call(func=name_("log"), args=[x], keywords=[]))


def expand_sqrt_arr(t: ast.Name, a: Sequence[ast.expr], s: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    return unary_elementwise(t, a, s, lambda x: ast.Call(func=name_("sqrt"), args=[x], keywords=[]))


def unary_elementwise(
    target: ast.Name,
    args: Sequence[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    op_fn: Callable[[ast.expr], ast.expr],
) -> list[ast.stmt]:
    """Common scaffold for ``out = np.<unary>(expr)`` -> per-element op.

    Accepts any array-valued expression: bare Name, slice subscript,
    or BinOp / UnaryOp / Call whose iteration extent is derivable.
    """
    if not args:
        raise NotImplementedError("unary elementwise needs an arg")
    a = args[0]
    extent = iter_extent_of(a, shape_table)
    if extent is None:
        raise NotImplementedError("unary elementwise: extent unknown")
    iters = [name_(f"__r{i}") for i in range(len(extent))]
    sa = scalarize_at_iters(a, iters, shape_table)
    idx = iters[0] if len(iters) == 1 else ast.Tuple(elts=list(iters), ctx=ast.Load())
    body: list[ast.stmt] = [
        ast.Assign(targets=[ast.Subscript(value=name_(target.id), slice=idx, ctx=ast.Store())], value=op_fn(sa))
    ]
    return wrap_for_loops([i.id for i in iters], extent, body)


def expand_clip(target: ast.Name, args: Sequence[ast.expr], shape_table: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    """``out = np.clip(a, lo, hi)`` -> ``out[i] = min(hi, max(lo, a[i]))``. numpy
    defines clip as ``minimum(a_max, maximum(a, a_min))``, so a degenerate
    ``lo > hi`` resolves to hi -- matched by the outer ``min`` (the reversed
    order would return ``lo`` instead).
    """
    a = first_name(args)
    if len(args) != 3 or a is None:
        raise NotImplementedError("np.clip needs Name + 2 scalar args")
    shape = shape_table.get(a.id)
    if not shape:
        raise NotImplementedError("np.clip: shape unknown")
    iters = [f"__r{i}" for i in range(len(shape))]
    idx = name_(iters[0]) if len(iters) == 1 else ast.Tuple(elts=[name_(i) for i in iters], ctx=ast.Load())
    a_sub = ast.Subscript(value=name_(a.id), slice=idx, ctx=ast.Load())
    clamped = ast.Call(
        func=name_("min"), args=[args[2], ast.Call(func=name_("max"), args=[args[1], a_sub], keywords=[])], keywords=[]
    )
    body: list[ast.stmt] = [
        ast.Assign(targets=[ast.Subscript(value=name_(target.id), slice=idx, ctx=ast.Store())], value=clamped)
    ]
    return wrap_for_loops(iters, shape, body)


def expand_where(target: ast.Name, args: Sequence[ast.expr], shape_table: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    """``out = np.where(cond, a, b)`` -> elementwise ternary. ``cond`` may be a
    bare mask Name or a whole-array comparison (lulesh's BC selection ``sel ==
    XI_M_SYMM``); ``a``/``b`` may be a Name or any whole-array expression (incl.
    a fancy gather ``delv[ielem]``). Every operand is scalarised recursively.
    """
    if len(args) != 3:
        raise NotImplementedError("np.where needs cond + 2 args")
    # Result shape: the hoister registered the target temp's shape; fall back to
    # the mask Name's own shape when ``where`` is assigned directly.
    shape = shape_table.get(target.id) if isinstance(target, ast.Name) else None
    if not shape and isinstance(args[0], ast.Name):
        shape = shape_table.get(args[0].id)
    if not shape:
        raise NotImplementedError("np.where: shape unknown")
    iters = [f"__r{i}" for i in range(len(shape))]
    idx = name_(iters[0]) if len(iters) == 1 else ast.Tuple(elts=[name_(i) for i in iters], ctx=ast.Load())
    iter_nodes = [name_(i) for i in iters]

    def maybe_sub(arg: ast.expr) -> ast.expr:
        return scalarize_at_iters(arg, iter_nodes, shape_table)

    ternary = ast.IfExp(test=maybe_sub(args[0]), body=maybe_sub(args[1]), orelse=maybe_sub(args[2]))
    body: list[ast.stmt] = [
        ast.Assign(targets=[ast.Subscript(value=name_(target.id), slice=idx, ctx=ast.Store())], value=ternary)
    ]
    return wrap_for_loops(iters, shape, body)


# Elementwise transcendental/math ufuncs (ARRAY form). Mirrors the scalar
# TRIG/ALG_TRANS lists in lowering/mathfuncs.py's MATH_BUILTINS so a function usable
# scalar-side is usable array-side too.
#
# Functions with a libm name (sin, atan2, rint, ...) emit a plain call --
# resolved through <math.h>/<cmath> -- via unary_call_expander/
# binary_call_expander. Functions without one (square, reciprocal, sign,
# degrees, radians) emit an inline expr (x*x, 1.0/x, ...) via
# unary_expr_expander: language-agnostic, so no helper functions/macros
# needed beyond the prelude's min/max/int_floor.


def unary_call_expander(c_name: str) -> Callable:
    """Elementwise expander for a unary numpy ufunc that maps directly to
    a libm function (``np.tan(arr)`` -> ``out[i] = tan(arr[i])``).

    Any argument past the operand (``out=``, ``where=``, ``np.round``'s ``decimals``) asks for
    something the libm call does not do, so it is refused rather than dropped."""

    def expand_(
        target: ast.Name,
        args: Sequence[ast.expr],
        shape_table: dict[str, tuple[str, ...]],
        kwargs: list[ast.keyword] | None = None,
    ) -> list[ast.stmt]:
        if len(args) != 1 or kwargs:
            raise NotImplementedError(f"np ufunc lowered to {c_name}: arguments past the operand are not supported")
        return unary_elementwise(
            target, args, shape_table, lambda x: ast.Call(func=name_(c_name), args=[x], keywords=[])
        )

    return expand_


def decimals_of(extra_args: Sequence[ast.expr], kwargs: Sequence[ast.keyword]) -> int:
    """The ``decimals`` of ``np.round(x, decimals)`` / ``np.round(x, decimals=d)``: 0 when absent.

    It picks the algorithm, so it must be a compile-time integer; ``out=`` or anything else is refused."""
    given = list(extra_args) + [kw.value for kw in kwargs if kw.arg == "decimals"]
    if len(extra_args) > 1 or any(kw.arg != "decimals" for kw in kwargs) or len(given) > 1:
        raise NotImplementedError("np.round: only the operand and ``decimals`` are supported")
    if not given:
        return 0
    node = given[0]
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        sign, node = -1, node.operand
    if not (isinstance(node, ast.Constant) and type(node.value) is int):
        raise NotImplementedError(f"np.round: decimals {ast.unparse(given[0])!r} must be an integer literal")
    return sign * node.value


def round_to_decimals(x: ast.expr, decimals: int) -> ast.expr:
    """numpy's ``round(x, decimals)``: scale by ``10**decimals``, round half to even, scale back.

    That is numpy's own algorithm (multiply, ``rint``, divide; divide first for a negative
    ``decimals``), so the result matches it bit for bit, including where the scaled value is not
    exact. ``rint`` rounds ties to even in the default rounding mode, as numpy does; C's ``round``
    would round them away from zero."""
    if decimals == 0:
        return ast.Call(func=name_("rint"), args=[x], keywords=[])
    factor = ast.Constant(value=float(10 ** abs(decimals)))
    inward, outward = (ast.Mult(), ast.Div()) if decimals > 0 else (ast.Div(), ast.Mult())
    scaled = ast.Call(func=name_("rint"), args=[ast.BinOp(left=x, op=inward, right=factor)], keywords=[])
    return ast.BinOp(left=scaled, op=outward, right=ast.Constant(value=factor.value))


def expand_round(
    target: ast.Name,
    args: Sequence[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    """``out = np.round(a, decimals)`` -> ``out[i] = round_to_decimals(a[i], decimals)``."""
    if not args:
        raise NotImplementedError("np.round needs an operand")
    decimals = decimals_of(args[1:], kwargs or [])
    return unary_elementwise(target, args[:1], shape_table, lambda x: round_to_decimals(x, decimals))


def unary_expr_expander(make: Callable[[ast.expr], ast.expr]) -> Callable:
    """Elementwise expander for a unary ufunc with no direct libm name --
    the result is an expression of the (scalarised) operand. ``make`` may
    use the operand twice; it is deep-copied per use to avoid sharing a
    single AST node across the tree."""
    return lambda t, a, s: unary_elementwise(t, a, s, make)


#: numpy unary ufuncs that map 1:1 to a libm call. The scalar form is already
#: handled by ``MATH_BUILTINS``; this registers the ARRAY form so ``out =
#: np.tan(arr)`` lowers to a loop. ``round``/``around`` map to ``rint``
#: (round-half-to-even, matching numpy; C ``round`` is half-away-from-zero).
UNARY_C_MATH: dict[str, str] = {
    "tan": "tan",
    "sinh": "sinh",
    "cosh": "cosh",
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "arcsinh": "asinh",
    "arccosh": "acosh",
    "arctanh": "atanh",
    "exp2": "exp2",
    "expm1": "expm1",
    "log2": "log2",
    "log10": "log10",
    "log1p": "log1p",
    "cbrt": "cbrt",
    "floor": "floor",
    "ceil": "ceil",
    "trunc": "trunc",
    "rint": "rint",
    "round": "rint",
    "around": "rint",
    "fabs": "fabs",
    "erf": "erf",
    "erfc": "erfc",
    "tgamma": "tgamma",
    "lgamma": "lgamma",
}


def binary_call_expander(c_name: str) -> Callable:
    """Elementwise expander for a binary numpy ufunc that maps to a libm
    call (``np.arctan2(a, b)`` -> ``out[i] = atan2(a[i], b[i])``).
    Broadcasts a scalar second operand. Mirrors :func:`expand_power`."""

    def expand_(target: ast.Name, args: Sequence[ast.expr], shape_table: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
        if len(args) != 2:
            raise NotImplementedError(f"np.{c_name} needs 2 args")
        a, b = args
        extent = broadcast_extent(a, b, shape_table)
        if extent is None:
            raise NotImplementedError(f"np.{c_name}: extent unknown")
        iters = [name_(f"__r{i}") for i in range(len(extent))]
        sa = scalarize_at_iters(a, iters, shape_table)
        sb = scalarize_at_iters(b, iters, shape_table)
        idx = iters[0] if len(iters) == 1 else ast.Tuple(elts=list(iters), ctx=ast.Load())
        body: list[ast.stmt] = [
            ast.Assign(
                targets=[ast.Subscript(value=name_(target.id), slice=idx, ctx=ast.Store())],
                value=ast.Call(func=name_(c_name), args=[sa, sb], keywords=[]),
            )
        ]
        return wrap_for_loops([i.id for i in iters], list(extent), body)

    return expand_
