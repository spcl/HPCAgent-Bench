"""Shared narrow-int wrap oracle for the imperative backends (C / C++ / Fortran).

numpy evaluates an elementwise integer op at the OPERAND dtype and wraps at that
width. The native backends promote a narrow read (int8/16/32, uint8/16/32) to the
int64 ABI integer and compute wide, so an INTERMEDIATE that overflows the element
width does not wrap. That gap is observable only when a narrow ``+``/``-``/``*``
result feeds a NON-ring op: ``+``/``-``/``*`` compose as a ring homomorphism mod
2**w, so wrapping every step and wrapping only at the store give congruent bytes --
but ``//`` (floor-division) is not a homomorphism and reads the un-wrapped
intermediate, so ``(a + b) // 2`` at int8 diverges (numpy -28, wide 100).

The fix is to re-wrap the wide result of every narrow-int ``+``/``-``/``*``/``**``/
``<<`` (and unary ``-``) back to its element width. To decide WHEN, an emitter
needs the numpy result dtype of a subtree; this module is the ONE
differentially-tested definition of that inference. The first attempt used two
divergent hand-rolled oracles, which truncated integer true division and int*float
-- both FLOAT results this inference reports as non-integer, so no wrap fires on
them.

The inference is deliberately CONSERVATIVE: an operand it cannot resolve to a
concrete dtype (a call result, an unknown name) makes the whole subtree UNKNOWN and
no wrap fires -- matching today's behaviour for that subtree rather than risking a
wrong wrap. ``+``/``-``/``*``/``**``/``<<`` and unary ``-`` are wrapped because each
can produce a magnitude exceeding its narrow operands (``**`` is exponential and
``<<`` shifts set bits past the top of the width, exactly like ``*`` by a power of
two -- int8 ``16 ** 2`` wraps 256 -> 0 and ``50 << 2`` wraps 200 -> -56 the same way
``100 + 100`` wraps 200 -> -56). ``//``/``%`` are bounded by their in-range operands
(a floor-divide or remainder cannot exceed the dividend's magnitude) so they never
overflow their own width; ``&``/``|``/``^``/``>>`` only combine or drop bits already
inside the width, so they cannot either. Only their sub-expressions can overflow,
which the recursion already covers.
"""

import ast
from collections.abc import Callable

from hpcagent_bench.translators.numpyto_common import dtypes

#: A resolved dtype category: a canonical integer dtype name, ``FLOAT``, ``WEAK_INT``, or
#: ``UNKNOWN`` (``None``).
Category = str | None

#: Result-dtype categories the inference returns. A concrete integer carries its
#: canonical numpy dtype name; the rest are sentinels distinct from any dtype name.
UNKNOWN = None  # cannot resolve (call, unknown name, logical) -> never wrap
FLOAT = "\0float"  # float/complex result -> never a narrow-int wrap
WEAK_INT = "\0weakint"  # a Python int literal: does not widen a concrete int (NEP 50)

#: BinOps under which numpy keeps an integer result (so the inference recurses into
#: them). ``Div`` is handled separately -- true division is always float.
INT_PRESERVING = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
)

#: The ops whose wide result is re-wrapped: a narrow intermediate overflows here.
#: ``**`` is exponential and ``<<`` shifts bits past the top of the width, so both
#: can exceed a narrow operand's range exactly like ``*`` can -- unlike ``//``/``%``
#: (bounded by the dividend) or ``&``/``|``/``^``/``>>`` (never grow past the width).
WRAP_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Pow, ast.LShift)

NameDtype = Callable[[str], str | None]


def is_float_or_complex(dtype: str) -> bool:
    try:
        c = dtypes.canonical(dtype)
    except KeyError:
        return False
    return c.startswith("float") or c.startswith("complex")


def narrow_width(dtype: str) -> int | None:
    """Itemsize (bytes) of a NARROW integer dtype (< the 8-byte ABI int), else None."""
    if not dtypes.is_integer(dtype):
        return None
    w = dtypes.itemsize(dtype)
    return w if w < 8 else None


def leaf_dtype(dtype: str | None) -> Category:
    """Category of a resolved leaf dtype: its canonical integer name, ``FLOAT``, or
    ``UNKNOWN`` (bool / fp8-storage / unresolved -- none is a narrow-int arithmetic
    operand)."""
    if dtype is None:
        return UNKNOWN
    if dtypes.is_integer(dtype):
        return dtypes.canonical(dtype)
    if is_float_or_complex(dtype):
        return FLOAT
    return UNKNOWN


def combine(a: Category, b: Category) -> Category:
    """Numpy promotion of two inferred categories."""
    if a is UNKNOWN or b is UNKNOWN:
        return UNKNOWN
    if a == FLOAT or b == FLOAT:
        return FLOAT
    if a == WEAK_INT and b == WEAK_INT:
        return WEAK_INT
    if a == WEAK_INT:  # a Python int literal does not widen the concrete operand
        return b
    if b == WEAK_INT:
        return a
    return dtypes.promote_integers(a, b)  # both concrete integers


def infer(node: ast.AST, name_dtype: NameDtype) -> Category:
    """The numpy result-dtype category of an expression."""
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return UNKNOWN
        if isinstance(v, int):
            return WEAK_INT
        if isinstance(v, (float, complex)):
            return FLOAT
        return UNKNOWN
    if isinstance(node, ast.Name):
        return leaf_dtype(name_dtype(node.id))
    if isinstance(node, ast.Subscript):
        base = node.value
        while isinstance(base, ast.Subscript):  # chained a[i][j] -> Name a
            base = base.value
        return leaf_dtype(name_dtype(base.id)) if isinstance(base, ast.Name) else UNKNOWN
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, (ast.USub, ast.UAdd)):
            return infer(node.operand, name_dtype)
        return UNKNOWN  # not / ~ -> logical, not arithmetic
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Div):
            return FLOAT  # true division is always float
        if isinstance(node.op, INT_PRESERVING):
            return combine(infer(node.left, name_dtype), infer(node.right, name_dtype))
        return UNKNOWN
    return UNKNOWN  # Call / Compare / BoolOp / IfExp -> not a narrow-int wrap site


def wrap_dtype(node: ast.AST, name_dtype: NameDtype) -> str | None:
    """The canonical narrow integer dtype a node's numpy result must be wrapped to
    (e.g. ``"int8"``), or None when no wrap is needed.

    ``name_dtype(name)`` resolves a bare name to its numpy dtype tag (or None when
    unknown -- a shape symbol / loop index should resolve to ``"int64"`` so it is
    the wide, no-wrap operand).
    """
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        pass
    elif isinstance(node, ast.BinOp) and isinstance(node.op, WRAP_BINOPS):
        pass
    else:
        return None
    cat = infer(node, name_dtype)
    if cat is UNKNOWN or cat == FLOAT or cat == WEAK_INT:
        return None
    return cat if narrow_width(cat) is not None else None
