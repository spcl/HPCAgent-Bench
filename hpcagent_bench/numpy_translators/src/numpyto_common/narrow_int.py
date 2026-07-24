"""Shared narrow-int wrap oracle for the imperative backends (C / C++ / Fortran).

numpy evaluates an elementwise integer op at the OPERAND dtype and wraps at that
width. The native backends promote a narrow read (int8/16/32, uint8/16/32) to the
int64 ABI integer and compute wide, so an INTERMEDIATE that overflows the element
width does not wrap. That gap is observable only when a narrow ``+``/``-``/``*``
result feeds a NON-ring op: ``+``/``-``/``*`` compose as a ring homomorphism mod
2**w, so wrapping every step and wrapping only at the store give congruent bytes --
but ``//`` (floor-division) is not a homomorphism and reads the un-wrapped
intermediate, so ``(a + b) // 2`` at int8 diverges (numpy -28, wide 100).

The fix is to re-wrap the wide result of every narrow-int ``+``/``-``/``*`` (and
unary ``-``) back to its element width. To decide WHEN, an emitter needs the numpy
result dtype of a subtree; this module is the ONE differentially-tested definition
of that inference. The first attempt used two divergent hand-rolled oracles, which
truncated integer true division and int*float -- both FLOAT results this inference
reports as non-integer, so no wrap fires on them.

The inference is deliberately CONSERVATIVE: an operand it cannot resolve to a
concrete dtype (a call result, an unknown name) makes the whole subtree UNKNOWN and
no wrap fires -- matching today's behaviour for that subtree rather than risking a
wrong wrap. Only ``+``/``-``/``*`` and unary ``-`` are wrapped: ``//``/``%``/``**``
are bounded by their in-range operands, so they never overflow their own width --
only their sub-expressions do, which the recursion already covers.
"""
import ast
from typing import Callable, Optional

from numpyto_common import dtypes

#: Result-dtype categories the inference returns. A concrete integer carries its
#: canonical numpy dtype name; the rest are sentinels distinct from any dtype name.
_UNKNOWN = None  # cannot resolve (call, unknown name, logical) -> never wrap
_FLOAT = "\0float"  # float/complex result -> never a narrow-int wrap
_WEAK_INT = "\0weakint"  # a Python int literal: does not widen a concrete int (NEP 50)

#: BinOps under which numpy keeps an integer result (so the inference recurses into
#: them). ``Div`` is handled separately -- true division is always float.
_INT_PRESERVING = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr, ast.BitXor,
                   ast.LShift, ast.RShift)

#: The ops whose wide result is re-wrapped: a narrow intermediate overflows here.
_WRAP_BINOPS = (ast.Add, ast.Sub, ast.Mult)

NameDtype = Callable[[str], Optional[str]]


def _is_float_or_complex(dtype: str) -> bool:
    try:
        c = dtypes.canonical(dtype)
    except KeyError:
        return False
    return c.startswith("float") or c.startswith("complex")


def _narrow_width(dtype: str) -> Optional[int]:
    """Itemsize (bytes) of a NARROW integer dtype (< the 8-byte ABI int), else None."""
    if not dtypes.is_integer(dtype):
        return None
    w = dtypes.itemsize(dtype)
    return w if w < 8 else None


def _leaf_dtype(dtype: Optional[str]):
    """Category of a resolved leaf dtype: its canonical integer name, ``_FLOAT``, or
    ``_UNKNOWN`` (bool / fp8-storage / unresolved -- none is a narrow-int arithmetic
    operand)."""
    if dtype is None:
        return _UNKNOWN
    if dtypes.is_integer(dtype):
        return dtypes.canonical(dtype)
    if _is_float_or_complex(dtype):
        return _FLOAT
    return _UNKNOWN


def _combine(a, b):
    """Numpy promotion of two inferred categories."""
    if a is _UNKNOWN or b is _UNKNOWN:
        return _UNKNOWN
    if a == _FLOAT or b == _FLOAT:
        return _FLOAT
    if a == _WEAK_INT and b == _WEAK_INT:
        return _WEAK_INT
    if a == _WEAK_INT:  # a Python int literal does not widen the concrete operand
        return b
    if b == _WEAK_INT:
        return a
    return dtypes.promote_integers(a, b)  # both concrete integers


def _infer(node: ast.AST, name_dtype: NameDtype):
    """The numpy result-dtype category of an expression."""
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return _UNKNOWN
        if isinstance(v, int):
            return _WEAK_INT
        if isinstance(v, (float, complex)):
            return _FLOAT
        return _UNKNOWN
    if isinstance(node, ast.Name):
        return _leaf_dtype(name_dtype(node.id))
    if isinstance(node, ast.Subscript):
        base = node.value
        while isinstance(base, ast.Subscript):  # chained a[i][j] -> Name a
            base = base.value
        return _leaf_dtype(name_dtype(base.id)) if isinstance(base, ast.Name) else _UNKNOWN
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, (ast.USub, ast.UAdd)):
            return _infer(node.operand, name_dtype)
        return _UNKNOWN  # not / ~ -> logical, not arithmetic
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Div):
            return _FLOAT  # true division is always float
        if isinstance(node.op, _INT_PRESERVING):
            return _combine(_infer(node.left, name_dtype), _infer(node.right, name_dtype))
        return _UNKNOWN
    return _UNKNOWN  # Call / Compare / BoolOp / IfExp -> not a narrow-int wrap site


def wrap_dtype(node: ast.AST, name_dtype: NameDtype) -> Optional[str]:
    """The canonical narrow integer dtype a node's numpy result must be wrapped to
    (e.g. ``"int8"``), or None when no wrap is needed.

    ``name_dtype(name)`` resolves a bare name to its numpy dtype tag (or None when
    unknown -- a shape symbol / loop index should resolve to ``"int64"`` so it is
    the wide, no-wrap operand).
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        pass
    elif isinstance(node, ast.BinOp) and isinstance(node.op, _WRAP_BINOPS):
        pass
    else:
        return None
    cat = _infer(node, name_dtype)
    if cat is _UNKNOWN or cat == _FLOAT or cat == _WEAK_INT:
        return None
    return cat if _narrow_width(cat) is not None else None
