"""Operator and intrinsic tables for the imperative backends, keyed by target.

Authored once here so the AST-operator -> target-syntax mapping is not
duplicated across the C and Fortran emitters (and, in Phase 4, so the shared
``BaseEmitter`` can look up its column by target). JAX does not use these -- it
unparses Python operators directly.
"""
import ast
from typing import Dict, Type

#: AST binary-op type -> target operator string.
BINOP: Dict[str, Dict[Type[ast.AST], str]] = {
    "c": {
        ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
        ast.Mod: "%",
        # FloorDiv is intercepted earlier and emitted via the ``int_floor``
        # macro -- never reaches this table.
        ast.BitOr: "|", ast.BitAnd: "&", ast.BitXor: "^",
        ast.LShift: "<<", ast.RShift: ">>",
    },
    "fortran": {
        ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
        ast.FloorDiv: "/", ast.Mod: "MOD",
        ast.Pow: "**",
    },
}

#: AST compare-op type -> target operator string.
CMPOP: Dict[str, Dict[Type[ast.AST], str]] = {
    "c": {
        ast.Eq: "==", ast.NotEq: "!=",
        ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    },
    "fortran": {
        ast.Eq: "==", ast.NotEq: "/=",
        ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    },
}

#: AST bool-op type -> target operator string.
BOOLOP: Dict[str, Dict[Type[ast.AST], str]] = {
    "c": {ast.And: "&&", ast.Or: "||"},
    "fortran": {ast.And: ".AND.", ast.Or: ".OR."},
}

#: Fortran-only: libm function name -> Fortran intrinsic.
FORTRAN_INTRINSICS: Dict[str, str] = {
    "exp": "EXP", "sqrt": "SQRT", "log": "LOG",
    "sin": "SIN", "cos": "COS", "pow": "**",
    "fabs": "ABS", "abs": "ABS",
    # different spelling vs the C name:
    "ceil": "CEILING", "trunc": "AINT", "rint": "ANINT", "round": "ANINT",
    "fmod": "MOD", "fmax": "MAX", "fmin": "MIN",
    "tgamma": "GAMMA", "lgamma": "LOG_GAMMA", "copysign": "SIGN",
}

#: Fortran-only: libm unary funcs with no Fortran intrinsic -> expression of the arg.
FORTRAN_FN_EXPR: Dict[str, str] = {
    "cbrt": "({a}) ** (1.0_{rk} / 3.0_{rk})",
    "log2": "LOG({a}) / LOG(2.0_{rk})",
    "exp2": "2.0_{rk} ** ({a})",
    "expm1": "EXP({a}) - 1.0_{rk}",
    "log1p": "LOG(1.0_{rk} + ({a}))",
}
