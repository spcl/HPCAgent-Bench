"""Symbolic shape tokens: alias substitution and extent equality."""

import ast
import re
from functools import lru_cache
from typing import TYPE_CHECKING

__all__ = [
    "DIM_EXPAND_ROUNDS",
    "DIM_IDENT_RE",
    "DIM_PROBE_POINTS",
    "NP_ZEROS_ALIASES",
    "SHAPE_READ_RE",
    "dims_agree",
    "shape_exprs_differ_numerically",
    "shape_exprs_equal",
    "static_shape_of",
    "substitute_dim_aliases",
    "sympify_shape",
]

if TYPE_CHECKING:
    import sympy


#: ``np.zeros_like`` etc. share a shape with another array. The
#: rewriter at the lower() level translates these into a local-array
#: declaration the existing ``ZerosRewriter`` already understands.
NP_ZEROS_ALIASES: tuple[str, ...] = (
    "zeros",
    "empty",
    "zeros_like",
    "empty_like",
    "ones",
    "ones_like",
    "ndarray",  # ``np.ndarray((I, J, K), dtype=...)`` -- raw uninitialised
    # allocator used by gt4py-derived weather kernels (vadv).
    # Same shape harvest as ``np.empty``.
)


def static_shape_of(expr: ast.expr, axis: int, shape_table: dict[str, tuple[str, ...]]) -> str | None:
    """Static (loop-var-free) shape token for the given axis of an expression,
    or None if not derivable. ``Subscript(Name, ...)`` returns the source
    array's full axis size from its declared shape, regardless of slice
    bounds, so a temp can be declared at function scope without depending on a
    loop variable."""
    if isinstance(expr, ast.Name):
        shape = shape_table.get(expr.id)
        if shape and axis < len(shape):
            return shape[axis]
    if isinstance(expr, ast.Subscript):
        name = expr.value.id if isinstance(expr.value, ast.Name) else None
        shape = shape_table.get(name) if name else None
        if shape:
            # Skip non-Slice axes to align with the array's full rank.
            sl = expr.slice
            axes = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            slice_count = 0
            for i, ax in enumerate(axes):
                if isinstance(ax, ast.Slice):
                    if slice_count == axis and i < len(shape):
                        return shape[i]
                    slice_count += 1
    return None


#: Single identifier inside a shape-token string, matched on word boundaries so substituting ``c``
#: never hits ``channels`` or ``__inl6_c``.
DIM_IDENT_RE = re.compile(r"[A-Za-z_]\w*")

#: ``arr.shape[i]`` inside a shape token -- a DIMENSION read, resolvable against the shape table.
SHAPE_READ_RE = re.compile(r"(\w+)\.shape\[(\d+)\]")

#: Cap on alternating alias/shape-read expansion rounds. Each round can expose new names, so the
#: two rewrites do not reach a joint fixpoint in one pass; a chain deeper than this is pathological
#: and stopping early only costs a declined matmul.
DIM_EXPAND_ROUNDS: int = 8


def substitute_dim_aliases(
    token: str, aliases: dict[str, str], shape_table: dict[str, tuple[str, ...]] | None = None
) -> str:
    """Rewrite one shape token into the kernel's PARAMETER vocabulary, to a fixpoint.

    A kernel names its own dimensions (``batch, channels, h, w = x.shape``), so the SAME extent
    reaches a comparison spelled two ways: ``channels`` from a body local, ``embed_dim`` from
    ``init.shapes``. ``aliases`` maps each dimension local to its definition; expanding both sides
    puts them in one vocabulary. Cycle-guarded on the active substitution chain, so a
    self-referential def stops expanding instead of recursing forever.

    An inlined helper spells its dims as a read off a LOCAL array (``__inl91_c =
    __inl8_y.shape[3]``), which no alias can resolve on its own -- hence ``shape_table``, and hence
    the rounds: resolving a shape read exposes fresh names to alias-expand, and vice versa.
    """

    def expand(text: str, active: tuple[str, ...]) -> str:

        def repl(m: "re.Match") -> str:
            ident = m.group(0)
            if ident not in aliases or ident in active:
                return ident
            return "(" + expand(aliases[ident], active + (ident,)) + ")"

        return DIM_IDENT_RE.sub(repl, text)

    def resolve_shape_reads(text: str) -> str:

        def repl(m: "re.Match") -> str:
            shape = shape_table.get(m.group(1))
            idx = int(m.group(2))
            if shape is None or idx >= len(shape):
                return m.group(0)
            return "(" + str(shape[idx]) + ")"

        return SHAPE_READ_RE.sub(repl, text)

    # Each round is "aliases to a fixpoint, then resolve shape reads", and a round only REPEATS
    # when a shape read actually resolved -- that is the only thing that can expose a name the
    # alias pass has not seen. Looping on any change instead would restart the per-chain cycle
    # guard, and a self-referential def would grow one level per round instead of stopping.
    text = str(token)
    for unused in range(DIM_EXPAND_ROUNDS):
        grown = expand(text, ())
        if not shape_table:
            return grown
        resolved = resolve_shape_reads(grown)
        if resolved == grown:
            return grown
        text = resolved
    return text


@lru_cache(maxsize=None, typed=True)
def sympify_shape(text: str) -> "sympy.Expr | None":
    """``text`` as a sympy expression, or ``None`` when it does not parse.

    One shape token is compared against many others, so without this the same string is re-parsed
    once per PAIR. Keyed on the string and never on a sympy object: sympy equality is structural
    and would collapse distinct tokens onto one entry.

    Every non-call identifier is bound to a Symbol first. Bare ``sympify`` resolves a name against
    sympy's own namespace, where ``N`` is the numeric-evaluation FUNCTION, ``S`` the singleton
    registry, and ``E``/``I``/``O``/``Q``/``pi``/``beta``/``gamma``/``zeta`` are constants or
    functions -- so ``N`` came back uncomparable and every extent naming it answered "not equal"
    however plainly equal it was. Those are ordinary dimension names in this corpus.

    The symbols are integral and non-negative because an extent is: without that, a pooled
    ``(2 * oh) // 2`` stays a ``floor`` sympy cannot cancel against ``oh``, and the two agreeing
    extents read as two different shapes. The assumption only lets sympy PROVE equalities that
    already hold for the integer extents these symbols stand for.
    """
    import sympy  # Deferred: sympy costs ~100s of ms to import and most kernels never reach here.

    # An unresolved ``A.shape[i]`` is not sympy syntax but a fixed extent: fold it to an atom. Both
    # sides mangle the same way, so the surrounding arithmetic still cancels.
    text = SHAPE_READ_RE.sub(lambda m: f"__shp_{m.group(1)}_{m.group(2)}", text)
    # ``a // b`` and ``int_floor(a, b)`` are ONE quantity in two spellings -- ``//`` is what a
    # manifest and a numpy source write, ``int_floor`` is what the C/dace side names it. sympify
    # turns the first into ``floor(a/b)`` and leaves the second an opaque Function, so a compare
    # across the two answered "not equal" for extents that are the same number. Normalise onto
    # sympy's own head, never the other way: ``floor`` over integer symbols CANCELS
    # (``floor(2*oh/2)`` is ``oh``), and an opaque ``int_floor`` does not -- measured both ways.
    names = {
        "int_floor": lambda a, b: sympy.floor(a / b),
        "int_ceil": lambda a, b: sympy.ceiling(a / b),
    }
    for m in DIM_IDENT_RE.finditer(text):
        if text[m.end() : m.end() + 1] != "(":  # a call target is a function, not a dimension
            names[m.group()] = sympy.Symbol(m.group(), integer=True, nonnegative=True)
    try:
        return sympy.sympify(text, locals=names)
    except (SyntaxError, TypeError, AttributeError, ValueError, IndexError, sympy.SympifyError):
        # ``sympify`` EVALUATES the token, so any exception the expression can raise is on this
        # path: a token that reads past the end of a tuple literal arrives as a bare ``IndexError``
        # from inside sympy's parser. Unresolvable is None, which the callers read as "not equal"
        # and decline on -- the safe direction, per :func:`dims_agree`.
        return None


#: Integer points the numeric refutation evaluates at. Two of them, because a single point lets a
#: pair coincide by accident (``2 * a`` and ``a + 7`` both give 14 at ``a = 7``), and fixed rather
#: than random so a verdict never depends on the run.
DIM_PROBE_POINTS: tuple[tuple[int, ...], ...] = ((7, 11, 13, 17, 19, 23, 29, 31), (3, 41, 5, 37, 2, 43, 11, 47))


def shape_exprs_differ_numerically(ea: "sympy.Expr", eb: "sympy.Expr") -> bool:
    """``True`` when the two expressions disagree at one integer point, which REFUTES equality.

    Shape tokens agree when they agree as FUNCTIONS of their symbols, so one disagreeing
    assignment settles the question -- in microseconds, against the hundreds of milliseconds
    ``simplify`` spends reaching the same verdict on a conv extent like ``(h - k) // s + 1``.
    Agreement at a point proves nothing and falls through to the symbolic rung, so this can only
    ever answer False faster, never answer True.
    """
    symbols = sorted(ea.free_symbols | eb.free_symbols, key=str)
    if not symbols or len(symbols) > len(DIM_PROBE_POINTS[0]):
        return False
    for values in DIM_PROBE_POINTS:
        point = dict(zip(symbols, values))
        try:
            va, vb = ea.subs(point), eb.subs(point)
        except (TypeError, ValueError, ZeroDivisionError):
            return False
        if not (va.is_number and vb.is_number):
            return False
        if va != vb:
            return True
    return False


@lru_cache(maxsize=None, typed=True)
def shape_exprs_equal(sa: str, sb: str) -> bool:
    """``True`` when two already alias-substituted shape expressions denote the same extent.

    Split out of :func:`dims_agree` so the symbolic work is memoized. :func:`reshape_axis_groups`
    asks about the same pair once per axis of every reshape, and a densenet-sized model reaches
    this rung thousands of times over a vocabulary of a few dozen tokens -- profiled at 61% of the
    whole lowering before the cache. Pure in its arguments: the alias table that produced the two
    strings is already folded into them.

    Four rungs, and ``simplify`` -- measured at 784 ms on one conv-extent pair -- is the last.
    ``expand`` PROVES the linear cases (swin's ``4 * (4 * embed_dim)`` against ``16 * embed_dim``);
    a non-zero expansion of a POLYNOMIAL difference is already conclusive, so those need nothing
    further; and :func:`shape_exprs_differ_numerically` refutes the rest in microseconds. Every
    rung answers exactly what ``simplify`` alone answered -- this is a cost cut, not a semantic
    change.
    """
    import sympy  # Deferred, as in sympify_shape.

    ea, eb = sympify_shape(sa), sympify_shape(sb)
    if ea is None or eb is None:
        return False
    try:
        diff = ea - eb
        if sympy.expand(diff) == 0:
            return True
        if diff.is_polynomial():
            return False  # An expanded polynomial is canonical: non-zero here means non-zero.
        if shape_exprs_differ_numerically(ea, eb):
            return False
        return bool(sympy.simplify(diff) == 0)
    except (TypeError, AttributeError, ValueError):
        return False


def dims_agree(
    a: str, b: str, aliases: dict[str, str] | None = None, shape_table: dict[str, tuple[str, ...]] | None = None
) -> bool:
    """``True`` when two shape tokens denote the same extent.

    Three rungs, cheapest first, because the first answers nearly every call: literal string
    equality; equality after :func:`substitute_dim_aliases` puts both in the parameter vocabulary;
    and only then a symbolic compare, for the case where substitution leaves arithmetically-equal
    but textually different expressions (swin's ``4 * (4 * embed_dim)`` against ``16 * embed_dim``).

    Unresolvable is FALSE, never True: a wrong ``True`` here contracts over two different extents,
    which is a miscompile, while a wrong ``False`` only declines a matmul the hoister then refuses.
    """
    if a == b:
        return True
    if not aliases and not shape_table:
        return False
    sa = substitute_dim_aliases(a, aliases or {}, shape_table)
    sb = substitute_dim_aliases(b, aliases or {}, shape_table)
    if sa == sb:
        return True
    # Order-insensitive key: both rungs above are symmetric and so is a zero difference, so
    # ``(a, b)`` and ``(b, a)`` must not occupy two cache entries.
    return shape_exprs_equal(sa, sb) if sa <= sb else shape_exprs_equal(sb, sa)
