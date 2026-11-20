"""Operator and intrinsic tables for the imperative backends, keyed by target.

Authored once here so the AST-operator -> target-syntax mapping is not
duplicated across the C and Fortran emitters (and, in Phase 4, so the shared
``BaseEmitter`` can look up its column by target).
"""

import ast
import math

__all__ = ["BINOP", "BOOLOP", "CMPOP", "FORTRAN_FN_EXPR", "FORTRAN_INTRINSICS", "folded_float_constant"]

#: AST binary-op type -> target operator string.
BINOP: dict[str, dict[type[ast.operator], str]] = {
    "c": {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.Mod: "%",
        # FloorDiv is intercepted earlier and emitted via the ``int_floor``
        # macro -- never reaches this table.
        ast.BitOr: "|",
        ast.BitAnd: "&",
        ast.BitXor: "^",
        ast.LShift: "<<",
        ast.RShift: ">>",
    },
    "fortran": {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.FloorDiv: "/",
        ast.Mod: "MOD",
        ast.Pow: "**",
    },
}

#: AST compare-op type -> target operator string.
CMPOP: dict[str, dict[type[ast.cmpop], str]] = {
    "c": {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
    },
    "fortran": {
        ast.Eq: "==",
        ast.NotEq: "/=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
    },
}

#: AST bool-op type -> target operator string.
BOOLOP: dict[str, dict[type[ast.boolop], str]] = {
    "c": {ast.And: "&&", ast.Or: "||"},
    "fortran": {ast.And: ".AND.", ast.Or: ".OR."},
}

#: Fortran-only: libm function name -> Fortran intrinsic.
FORTRAN_INTRINSICS: dict[str, str] = {
    "exp": "EXP",
    "sqrt": "SQRT",
    "log": "LOG",
    "sin": "SIN",
    "cos": "COS",
    "pow": "**",
    "fabs": "ABS",
    "abs": "ABS",
    # different spelling vs the C name:
    "ceil": "CEILING",
    "trunc": "AINT",
    "rint": "ANINT",
    "round": "ANINT",
    "fmod": "MOD",
    "fmax": "MAX",
    "fmin": "MIN",
    "tgamma": "GAMMA",
    "lgamma": "LOG_GAMMA",
    "copysign": "SIGN",
}

#: Fortran-only: libm unary funcs with no Fortran intrinsic -> expression of the arg.
FORTRAN_FN_EXPR: dict[str, str] = {
    "cbrt": "({a}) ** (1.0_{rk} / 3.0_{rk})",
    "log2": "LOG({a}) / LOG(2.0_{rk})",
    "exp2": "2.0_{rk} ** ({a})",
    "expm1": "EXP({a}) - 1.0_{rk}",
    "log1p": "LOG(1.0_{rk} + ({a}))",
}


def folded_float_constant(node: ast.expr) -> float | None:
    """The value of ``node`` when it is a float constant expression (numeric literals under unary +/- and binary
    + - * /, at least one of them a float), folded in double as Python evaluates it; else None.

    A narrow kernel compares against it in double, as the references it is graded against do (the numba reference
    compares a float32 value with a float64 constant in double, the C reference computes in fp64): folded in float,
    ``1.0 - 1e-8`` is 1.0, so ``za > 1.0 - 1e-8`` never holds for a full cloud (cloudsc's snow flux off at fp32).
    """
    value, is_float = fold(node)
    if value is None or not is_float or not math.isfinite(value):
        return None
    return value


def fold(node: ast.expr) -> tuple[float | None, bool]:
    """``(value, any float leaf)`` of a numeric constant expression, ``(None, False)`` for anything else."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return None, False
        return float(node.value), isinstance(node.value, float)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value, is_float = fold(node.operand)
        if value is None:
            return None, False
        return (-value if isinstance(node.op, ast.USub) else value), is_float
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, left_float = fold(node.left)
        right, right_float = fold(node.right)
        if left is None or right is None:
            return None, False
        if isinstance(node.op, ast.Add):
            return left + right, left_float or right_float
        if isinstance(node.op, ast.Sub):
            return left - right, left_float or right_float
        if isinstance(node.op, ast.Mult):
            return left * right, left_float or right_float
        if right == 0.0:
            return None, False
        return left / right, True
    return None, False
