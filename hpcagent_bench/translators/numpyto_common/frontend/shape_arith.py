"""Shape-expression arithmetic: exact integer simplification and inlined scalar-dimension locals."""

import ast
import re
from functools import lru_cache
from collections.abc import Callable

__all__ = [
    "FOLD_OPS",
    "IDENT_RE",
    "ShapeArithFolder",
    "binding_counts",
    "collect_inlined_scalar_defs",
    "combine_like_terms",
    "const_int",
    "divide_multiple_term",
    "exact_multiple_factor",
    "exact_quotient_with_remainder",
    "fold_shape_expr",
    "gather_add_chain",
    "is_scalar_dim_rhs",
    "literal_axis",
    "resolve_shape_attr_tokens",
    "scaled_term",
    "substitute_inlined_scalar_defs",
]


def resolve_shape_attr_tokens(tokens: tuple[str, ...], parsed_seed: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Replace ``arr.shape[i]`` occurrences in each shape token with the
    ``i``-th element of ``arr``'s seed shape (``A.shape[1]`` -> ``N``)."""

    def repl_(m: "re.Match[str]") -> str:
        arr, idx = m.group(1), int(m.group(2))
        ts = parsed_seed.get(arr)
        if ts is not None and idx < len(ts):
            return str(ts[idx])
        return m.group(0)

    return tuple(re.sub(r"(\w+)\.shape\[(\d+)\]", repl_, str(tok)) for tok in tokens)


#: Word-boundary matcher for a single identifier token inside a shape
#: string (so substituting ``K`` does not also hit ``C_out`` / ``__inl1_K``).
IDENT_RE = re.compile(r"[A-Za-z_]\w*")


def binding_counts(fn: ast.FunctionDef) -> dict[str, int]:
    """How often each name is bound in ``fn`` (tuple targets unpacked); an augmented assignment
    counts twice, so an updated name is never single-assignment."""
    counts: dict[str, int] = {}

    def count(tgt: ast.AST, inc: int) -> None:
        if isinstance(tgt, ast.Name):
            counts[tgt.id] = counts.get(tgt.id, 0) + inc
        elif isinstance(tgt, (ast.Tuple, ast.List)):
            for e in tgt.elts:
                count(e, inc)

    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.AugAssign):
            count(stmt.target, 2)
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                count(t, 1)
    return counts


def collect_inlined_scalar_defs(fn: ast.FunctionDef, prefix: str | None = "__inl") -> dict[str, str]:
    """Map each SCALAR-dimension local under ``fn`` to its RHS.

    Helper inlining (:class:`InlineHelpers`) lifts a helper's body locals
    into the kernel under an ``__inl<k>_`` prefix. The scalar ones are
    dimension definitions (``__inl1_N = input.shape[0]``) that end up inside
    the inlined output array's shape (``np.empty((__inl1_N, ...))``); left
    unresolved they're un-bindable shape symbols. Substituting them away
    (:func:`substitute_inlined_scalar_defs`) makes the shape a pure function
    of real kernel parameters again.

    ``prefix`` restricts collection to names starting with it (the default,
    the inliner's own ``__inl`` prefix); pass ``None`` to collect every
    single-assignment scalar-dim local regardless of name -- used to harvest
    a legacy ``initialize()`` companion module's own derived locals (conv2d's
    ``H_out = H - K + 1``, lulesh's ``NE = numElem``).

    Only scalar-expression RHS (Name/Constant/BinOp/``arr.shape[i]``/etc.) is
    collected -- an array-valued RHS is the inlined local array itself, not
    a dimension. Returns ``{name: ast.unparse(rhs)}`` for first assignments.
    """
    # A rebound name is a runtime value (a step counter), not a fixed dimension.
    rebind_counts = binding_counts(fn)
    defs: dict[str, str] = {}
    for stmt in ast.walk(fn):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            continue
        name = stmt.targets[0].id
        if name in defs:
            continue
        if prefix is not None and not name.startswith(prefix):
            continue
        if rebind_counts.get(name, 0) > 1:
            continue
        if not is_scalar_dim_rhs(stmt.value):
            continue
        defs[name] = ast.unparse(stmt.value)
    return defs


def is_scalar_dim_rhs(node: ast.AST) -> bool:
    """``True`` when ``node`` is a scalar-dimension expression (the RHS of
    an inlined ``__inl<k>_`` size local) rather than an array value.

    Accepts Names, integer Constants, ``arr.shape[i]`` subscripts and
    BinOps thereof. Rejects array constructors / generic calls / slices
    (those are the inlined local *array*, not one of its dimensions).
    """
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int)
    if isinstance(node, ast.UnaryOp):
        return is_scalar_dim_rhs(node.operand)
    if isinstance(node, ast.BinOp):
        return is_scalar_dim_rhs(node.left) and is_scalar_dim_rhs(node.right)
    # ``arr.shape[i]`` -- Subscript of a ``.shape`` Attribute on a Name.
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "shape"
        and isinstance(node.value.value, ast.Name)
    ):
        return True
    return False


def substitute_inlined_scalar_defs(tokens: tuple[str, ...], defs: dict[str, str]) -> tuple[str, ...]:
    """Rewrite shape ``tokens`` by inlining the ``__inl<k>_`` scalar-dim
    definitions from ``defs`` to a fixpoint (defs may reference one
    another, e.g. ``__inl1_H_out`` uses ``__inl1_K``).

    Substitution is identifier-boundary safe (``IDENT_RE``) so it never
    partial-matches a longer name. After the fixpoint every ``__inl``
    token is gone, leaving real params and ``arr.shape[i]`` references the
    existing resolvers concretise. Cycle-guarded: bounded by the number of
    defs (a self/mutually-referential def stops expanding once it would
    re-introduce a name already on the active substitution chain)."""
    if not defs:
        return tokens

    def expand_(text: str, active: tuple[str, ...]) -> str:

        def repl_(m: "re.Match[str]") -> str:
            ident = m.group(0)
            if ident not in defs or ident in active:
                return ident
            return "(" + expand_(defs[ident], active + (ident,)) + ")"

        return IDENT_RE.sub(repl_, text)

    return tuple(fold_shape_expr(expand_(str(tok), ())) for tok in tokens)


#: Binary ops foldable on two integer literals. ``/`` is absent on purpose: a shape token divides
#: exactly, but ``a / b`` on ints is a FLOAT in Python and folding it would emit ``3.0`` as an extent.
FOLD_OPS: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
}


def const_int(node: ast.expr) -> int | None:
    """``node`` as a Python int, or None. Accepts a negated literal (``-1`` parses as a UnaryOp)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = const_int(node.operand)
        if inner is not None:
            return -inner if isinstance(node.op, ast.USub) else inner
    return None


def exact_multiple_factor(numerator: ast.expr, denominator: ast.expr) -> int | None:
    """``k`` iff ``numerator`` is literally ``k * denominator`` or ``denominator * k``.

    ``(k * x) // x`` is ``k`` for every nonzero integer ``x`` -- no divisibility assumption is
    needed, ``k * x`` IS a multiple of ``x`` by construction. This is what a shape alias's own
    definition (``nnz = 3 * n``) folds back into once ``nnz`` is substituted into ``nnz // n``:
    without it the emitted extent is ``n * int_floor(3 * n, n)``, which the destination shape
    ``3 * n`` textually is but the DaCe frontend cannot prove equal to.
    """
    if not isinstance(numerator, ast.BinOp) or not isinstance(numerator.op, ast.Mult):
        return None
    denom_dump = ast.dump(denominator)
    left_k, right_k = const_int(numerator.left), const_int(numerator.right)
    if left_k is not None and ast.dump(numerator.right) == denom_dump:
        return left_k
    if right_k is not None and ast.dump(numerator.left) == denom_dump:
        return right_k
    return None


def divide_multiple_term(term: ast.expr, divisor: int) -> ast.expr | None:
    """``term / divisor`` iff ``term`` is literally a constant multiple of it, else ``None``."""
    if not isinstance(term, ast.BinOp) or not isinstance(term.op, ast.Mult):
        return None
    for const_side, other in ((term.left, term.right), (term.right, term.left)):
        factor = const_int(const_side)
        if factor is None or factor % divisor:
            continue
        quotient = factor // divisor
        return other if quotient == 1 else ast.BinOp(left=ast.Constant(value=quotient), op=ast.Mult(), right=other)
    return None


def exact_quotient_with_remainder(numerator: ast.expr, divisor: int) -> ast.expr | None:
    """``(d*A + c) // d`` -> ``A + c//d`` -- true for EVERY integer ``A`` and ``c``.

    Not the distribution the folder refuses below: that one splits a numerator whose terms are not
    multiples of the divisor, and is wrong exactly because the division is inexact. Here every
    non-constant term is a LITERAL multiple, so ``floor((d*A + c)/d) == A + floor(c/d)`` regardless
    of either sign -- the leftover constant carries whatever it contributes and nothing is rounded
    away. raman_fitting's jacobian is allocated ``3 * ((3 * K + 2) // 3) + 1`` and its columns
    written through slices of the same shape; unfolded, the frontend saw ``K`` on one side and
    ``int_floor(3*K + 2, 3)`` on the other and could not prove the two equal.
    """
    if divisor <= 0:
        return None
    terms: list[tuple[int, ast.expr]] = []
    constant = 0

    def walk(expr: ast.expr, sign: int) -> None:
        nonlocal constant
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub)):
            walk(expr.left, sign)
            walk(expr.right, sign if isinstance(expr.op, ast.Add) else -sign)
            return
        value = const_int(expr)
        if value is None:
            terms.append((sign, expr))
        else:
            constant += sign * value

    walk(numerator, 1)
    if not terms:
        return None
    quotients = [(sign, divide_multiple_term(term, divisor)) for sign, term in terms]
    if any(q is None for sign_, q in quotients):
        return None
    lead = next((i for i, (sign, q_) in enumerate(quotients) if sign > 0), None)
    if lead is None:
        return None  # the identity still holds; there is just no leading term to rebuild the sum from
    out = quotients[lead][1]
    for i, (sign, quotient) in enumerate(quotients):
        if i != lead:
            out = ast.BinOp(left=out, op=ast.Add() if sign > 0 else ast.Sub(), right=quotient)
    remainder = constant // divisor  # floor division, so a negative constant carries its own -1
    if remainder:
        out = ast.BinOp(
            left=out, op=ast.Add() if remainder > 0 else ast.Sub(), right=ast.Constant(value=abs(remainder))
        )
    return out


class ShapeArithFolder(ast.NodeTransformer):
    """Simplify a shape expression using integer identities that hold for EVERY value.

    Four rewrites, each unconditionally true over the integers (for a nonzero divisor, which a
    shape denominator always is): literal-op-literal folds to its value; ``x + 0`` / ``x - 0`` /
    ``x * 1`` / ``x // 1`` collapse to ``x``; ``(k * x) // x`` collapses to ``k``; and a chain of
    ``+``/``-`` gathers its literals into one trailing term.

    Deliberately absent beyond that: anything else about ``//``'s operands. ``(x + 2) // 2`` is NOT
    ``x // 2 + 1`` when x is not a multiple of 2, and floor division rounds toward -inf, so
    distributing it is wrong in general -- those divisions stay exactly where they were.
    """

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        self.generic_visit(node)
        left, right = const_int(node.left), const_int(node.right)
        op = FOLD_OPS.get(type(node.op))
        if op is not None and left is not None and right is not None:
            return ast.copy_location(ast.Constant(value=op(left, right)), node)
        if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and left is not None and right not in (None, 0):
            value = left // right if isinstance(node.op, ast.FloorDiv) else left % right
            return ast.copy_location(ast.Constant(value=value), node)
        if isinstance(node.op, ast.FloorDiv) and right is None:
            factor = exact_multiple_factor(node.left, node.right)
            if factor is not None:
                return ast.copy_location(ast.Constant(value=factor), node)
        if isinstance(node.op, ast.FloorDiv) and right is not None:
            quotient = exact_quotient_with_remainder(node.left, right)
            if quotient is not None:
                return ast.copy_location(ast.fix_missing_locations(quotient), node)
        # Identities. Commutative ones match either side; ``x - 0`` and ``x // 1`` only the right,
        # since ``0 - x`` negates and ``1 // x`` does not simplify.
        if isinstance(node.op, (ast.Add, ast.Mult)):
            unit = 0 if isinstance(node.op, ast.Add) else 1
            if right == unit:
                return node.left
            if left == unit:
                return node.right
        if isinstance(node.op, ast.Sub) and right == 0:
            return node.left
        if isinstance(node.op, ast.FloorDiv) and right == 1:
            return node.left
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return gather_add_chain(node)
        return node


def scaled_term(coefficient: int, term: ast.expr) -> ast.expr:
    """``term`` for a coefficient of 1, else ``coefficient * term``."""
    if coefficient == 1:
        return term
    return ast.BinOp(left=ast.Constant(value=coefficient), op=ast.Mult(), right=term)


def combine_like_terms(terms: list[tuple[int, ast.expr]]) -> list[tuple[int, ast.expr]]:
    """Sum the signs of structurally identical terms, dropping any that cancel to zero.

    ``span - (-span)`` is ``2 * span`` and ``a - a`` is nothing at all. Keyed on ``ast.dump``, so
    only terms spelled the same combine -- this decides no equality the text does not already make
    obvious. First-appearance order is kept, since the emitted extent is read by people.
    """
    order: list[str] = []
    coefficients: dict[str, int] = {}
    nodes: dict[str, ast.expr] = {}
    for sign, term in terms:
        key = ast.dump(term)
        if key not in coefficients:
            order.append(key)
            nodes[key] = term
        coefficients[key] = coefficients.get(key, 0) + sign
    return [(coefficients[key], nodes[key]) for key in order if coefficients[key]]


def gather_add_chain(node: ast.BinOp) -> ast.expr:
    """``((h + 6) - 7) + 1`` -> ``h + 0`` -> ``h``: sum the literals in one ``+``/``-`` chain.

    Without this the identities above never fire. Each inlined helper layer appends its own ``+ pad``
    / ``- kernel`` / ``+ 1``, so the literals arrive interleaved with the symbol and no single
    rewrite sees ``x + 0``; folding the chain is what makes a five-deep conv output-size expression
    collapse instead of growing one parenthesised layer per helper.

    A unary minus is part of the chain, and repeated terms COMBINE: ``(span + 1) - (-span)`` is
    ``2 * span + 1``, which is how cp2k_grid_integrate spells one length twice -- once as
    ``nrel = 2 * span + 1`` and once as the extent of ``np.arange(-span, span + 1)``. Left apart,
    the two became separate minted symbols the frontend could not prove equal. Both rewrites are
    ordinary integer identities, like everything else here.
    """
    terms: list[tuple[int, ast.expr]] = []
    total = 0

    def walk(expr: ast.expr, sign: int) -> None:
        nonlocal total
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub)):
            walk(expr.left, sign)
            walk(expr.right, sign if isinstance(expr.op, ast.Add) else -sign)
            return
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
            walk(expr.operand, sign if isinstance(expr.op, ast.UAdd) else -sign)
            return
        value = const_int(expr)
        if value is None:
            terms.append((sign, expr))
        else:
            total += sign * value

    walk(node, 1)
    terms = combine_like_terms(terms)
    if not terms or all(sign < 0 for sign, unused in terms):
        return node  # a bare literal, or a fully-negated chain -- rebuilding it gains nothing
    lead = next(i for i, (sign, unused) in enumerate(terms) if sign > 0)
    out = scaled_term(*terms[lead])
    for i, (sign, term) in enumerate(terms):
        if i == lead:
            continue
        out = ast.BinOp(left=out, op=ast.Add() if sign > 0 else ast.Sub(), right=scaled_term(abs(sign), term))
    if total:
        out = ast.BinOp(left=out, op=ast.Add() if total > 0 else ast.Sub(), right=ast.Constant(value=abs(total)))
    return ast.copy_location(ast.fix_missing_locations(out), node)


@lru_cache(maxsize=None, typed=True)
def fold_shape_expr(text: str) -> str:
    """Simplify a shape-token expression; returns ``text`` unchanged if it does not parse.

    Inlining a helper's size locals wraps one more layer of parentheses per level
    (:func:`substitute_inlined_scalar_defs`), so a network whose helpers nest five deep emits a
    single extent hundreds of characters long -- repeated at every loop bound and every allocation.
    densenet121's Fortran came out at 10k lines and did not finish compiling. The arithmetic is
    almost entirely ``+ 0`` / ``- 1 + 1`` / ``// 1`` that the identities above erase.

    Cached: a parse and an unparse per call, asked once per extent per pass over the same handful
    of distinct tokens -- 33% of a mobilenet lowering once the symbolic compare stopped dominating.
    Pure in ``text`` (the folder rebuilds its tree from the string every call), so the entry can
    never go stale.
    """
    if not isinstance(text, str) or not any(c in text for c in "+-*/"):
        return text
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return text
    return ast.unparse(ShapeArithFolder().visit(tree).body)


def literal_axis(sl: ast.expr) -> int | None:
    """The integer axis of a ``.shape[k]`` read, or ``None`` when it is not a literal one."""
    if isinstance(sl, ast.Constant) and isinstance(sl.value, int) and not isinstance(sl.value, bool):
        return sl.value
    if (
        isinstance(sl, ast.UnaryOp)
        and isinstance(sl.op, ast.USub)
        and isinstance(sl.operand, ast.Constant)
        and isinstance(sl.operand.value, int)
    ):
        return -sl.operand.value
    return None
