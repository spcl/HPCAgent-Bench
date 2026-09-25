"""AST builders and small shape/slice predicates shared by every expander."""

import ast
from collections.abc import Iterable, Sequence
from typing import Any
from collections.abc import Callable

from hpcagent_bench.translators.numpyto_common.subscripts import is_full_slice


def name_(n: str) -> ast.Name:
    return ast.Name(id=n, ctx=ast.Load())


def const_(v: Any) -> ast.Constant:
    # numpy scalars (np.int64(0)) aren't Python int, so Fortran emit misclassifies
    # a size symbol built from one as REAL, and numpy 2.0's repr unparses it as
    # ``np.int64(0)``, breaking dace's sympy range parse. Coerce so every backend
    # sees a bare ``0`` / ``0.0``.
    if type(v).__module__.startswith("numpy"):
        v = v.item()
    return ast.Constant(value=v)


def store_(n: str) -> ast.Name:
    return ast.Name(id=n, ctx=ast.Store())


def attr_call(mod: str, attr: str, args: list[ast.expr]) -> ast.Call:
    return ast.Call(func=ast.Attribute(value=name_(mod), attr=attr, ctx=ast.Load()), args=args, keywords=[])


# Expanders -- each returns replacement statements for the original assignment.


def make_iter_name(prefix: str, depth: int) -> str:
    return f"{prefix}{depth}"


def wrap_for_loops(iters: list[str], bounds: Sequence[str | ast.expr], body: list[ast.stmt]) -> list[ast.stmt]:
    """Wrap ``body`` in nested ``for v in range(bound):`` loops, outermost first.

    ``bounds`` entries are either a string (rendered via :func:`const_or_name`)
    or an already-built AST expression (passed through unchanged).
    """
    out = body
    for var, bound in zip(reversed(iters), reversed(bounds)):
        bound_node = const_or_name(bound) if isinstance(bound, str) else bound
        out = [
            ast.For(
                target=store_(var),
                iter=ast.Call(func=name_("range"), args=[bound_node], keywords=[]),
                body=out,
                orelse=[],
            )
        ]
    return out


def ast_eq(a: ast.AST, b: ast.AST) -> bool:
    """Structural equality for two AST expressions -- only ``Name``/``Constant``
    ids/values and matching ``BinOp`` ops; anything else is False, so the
    algebraic simplifier just falls back to the unsimplified form."""
    if type(a) is not type(b):
        return False
    if isinstance(a, ast.Name):
        return a.id == b.id
    if isinstance(a, ast.Constant):
        return a.value == b.value
    if isinstance(a, ast.BinOp):
        return type(a.op) is type(b.op) and ast_eq(a.left, b.left) and ast_eq(a.right, b.right)
    return False


def simplify_sub(hi: ast.AST, lo: ast.AST) -> ast.AST | None:
    """Algebraic simplification for ``hi - lo``. Returns ``None``
    when the form doesn't match a known simplifying pattern."""
    if ast_eq(hi, lo):
        return ast.Constant(value=0)
    if isinstance(hi, ast.BinOp) and isinstance(hi.op, ast.Add):
        # ``(lo + K) - lo`` -> K, at ANY depth of the ``+`` chain. A convolution tap slice spells
        # its upper bound ``ky + (oh - 1) * stride + 1``, which puts ``lo`` one level down where the
        # top-level match missed it. The extent is the same number either way, but the unsimplified
        # form NAMES the tap loop's variable, and the buffer it sizes is declared outside that loop
        # -- so the emitted C does not compile (alexnet, lenet5).
        for near, far in ((hi.left, hi.right), (hi.right, hi.left)):
            if ast_eq(near, lo):
                return far
            inner = simplify_sub(near, lo)
            if inner is not None:
                # ``(near - lo) + far`` is ``(near + far) - lo``, which is ``hi - lo``.
                return ast.BinOp(left=inner, op=ast.Add(), right=far)
    # ``(K - lo) - lo`` and other forms don't simplify in general.
    return None


def is_full_slice_subscript(node: ast.Subscript) -> bool:
    """Return True when ``node`` is a Subscript whose slice is a
    full slice ``:`` (or a tuple of full slices ``:, :``)."""
    sl = node.slice
    if isinstance(sl, ast.Slice):
        return is_full_slice(sl)
    if isinstance(sl, ast.Tuple):
        return all(is_full_slice(e) for e in sl.elts)
    return False


def const_or_name(token: str) -> ast.expr:
    """Render a shape token as the matching AST node: int literal -> ``Constant``,
    bare identifier -> ``Name``, compound expression (``"N * 2"``, ``"x.shape[3]"``)
    -> re-parsed via ``ast.parse(mode="eval")`` into real Subscript/BinOp nodes.

    Compound support matters because ``resolve_shape_token`` stringifies BinOps
    and ``arr.shape[i]`` refs into the shape table; without re-parsing, downstream
    AST walkers (e.g. the source-order shape resolver) would only see an opaque
    ``Name(id=full-text)``.
    """
    try:
        return const_(int(token))
    except (TypeError, ValueError):
        pass
    if isinstance(token, str) and token.isidentifier():
        return name_(token)
    # Compound expression -- re-parse.
    try:
        return ast.parse(str(token), mode="eval").body
    except (SyntaxError, ValueError):
        return name_(str(token))


def shape_total_product(shape: tuple[str, ...]) -> ast.expr:
    """Return an AST for ``shape[0] * shape[1] * ...`` -- used by mean."""
    parts = [const_or_name(s) for s in shape]
    expr = parts[0]
    for p in parts[1:]:
        expr = ast.BinOp(left=expr, op=ast.Mult(), right=p)
    return expr


def slice_step_const(sl: ast.Slice) -> int | None:
    """Return a Slice's constant integer step (``a[lo:hi:k]`` -> ``k``), or ``None`` when
    there is no step or it is not a nonzero integer constant. A NEGATIVE step (``a[::-1]``
    reverse, ``a[::-2]``) is returned as-is; callers handle the reverse index mapping and
    take ``abs`` for the element count. A symbolic step is unsupported (``None``)."""
    step = sl.step
    if step is None:
        return None
    v = const_int(step)
    return v if v not in (None, 0) else None


def slice_step_expr(sl: ast.Slice) -> ast.expr | None:
    """A slice's step when it is SYMBOLIC -- an expression rather than a literal.

    ``None`` for no step and for a literal one; :func:`slice_step_const` answers that question.
    The value is a runtime stride the kernel takes across the ABI (a conv/pool ``stride``), so it
    cannot be folded to a literal and the loop nest has to carry it.

    Only a BOUNDED slice reaches here (see ``reject_unsupported_slices``), and that is what makes
    the positive-stride lowering ``start + pos * step`` sound without knowing the sign: under a
    negative step numpy flips the bound defaults, so ``lo:hi:k`` with ``lo < hi`` is EMPTY, and the
    assignment consuming it already fails in numpy. There is no correct run to preserve.
    """
    step = sl.step
    if step is None or const_int(step) is not None:
        return None
    return step


def slice_step_any(sl: ast.Slice) -> int | ast.expr | None:
    """A slice's step as a literal ``int``, as an ``ast.expr`` when it is symbolic, or ``None``.

    The one accessor the index and extent builders share, so a symbolic stride can never reach one
    of them while the other still reads the slice as contiguous -- the two must walk the same run.
    """
    const = slice_step_const(sl)
    return const if const is not None else slice_step_expr(sl)


def step_is_negative(step: int | ast.expr | None) -> bool:
    """``step`` is a literal negative stride -- the numpy reverse. A symbolic step is never this:
    it is emitted as a positive stride, which is the only sign a bounded slice can carry."""
    return isinstance(step, int) and step < 0


def step_node(step: ast.expr | int) -> ast.expr:
    """``step`` as an expression, whether it arrived as a literal int or already as one."""
    return const_(step) if isinstance(step, int) else step


def is_shape_scalar(node: ast.AST) -> bool:
    """``True`` for a ``.shape`` read -- ``A.shape`` (a tuple of dimensions) or
    ``A.shape[i]`` (one dimension). Both are INTEGER-valued regardless of ``A``'s
    element dtype, so a value/dtype walk must not descend into them."""
    if isinstance(node, ast.Attribute) and node.attr == "shape":
        return True
    return isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "shape"


def reads_complex(expr: ast.AST, local_dtypes: dict[str, str]) -> bool:
    """True iff evaluating ``expr`` reads a complex value: a ``Constant(complex)``
    or a ``Name`` tagged complex in ``local_dtypes``. ``.shape`` subtrees are
    skipped -- always integer dimensions even when the array is complex.
    Shared predicate for the dtype-propagation passes (``CallHoister.infer_complex``,
    ``LibNodeRewriter.visit_Assign``)."""
    if is_shape_scalar(expr):
        return False
    if isinstance(expr, ast.Constant):
        return isinstance(expr.value, complex)
    if isinstance(expr, ast.Name):
        dt = local_dtypes.get(expr.id)
        return bool(dt and dt.startswith("complex"))
    return any(reads_complex(c, local_dtypes) for c in ast.iter_child_nodes(expr))


def slice_axes(node: ast.AST) -> list[ast.AST]:
    """Flat list of per-axis index nodes for any Subscript: ``A[i]`` -> ``[i]``,
    ``A[i, j]`` -> ``[i, j]``. A Slice axis is returned as the Slice node itself
    so callers can decide whether to scalarize."""
    if not isinstance(node, ast.Subscript):
        return []
    sl = node.slice
    if isinstance(sl, ast.Tuple):
        return list(sl.elts)
    return [sl]


def is_special_axis(elt: ast.expr) -> bool:
    """A rank-shifting subscript entry: numpy newaxis (``None``) or ``...``
    (Ellipsis). Neither consumes exactly one source axis the ordinary way -- a
    newaxis inserts a size-1 result axis; an Ellipsis fills the un-consumed
    source axes."""
    return isinstance(elt, ast.Constant) and (elt.value is None or elt.value is Ellipsis)


def is_scalar_axis(elt: ast.expr) -> bool:
    """A subscript entry that consumes ONE source axis as a scalar index -- an
    int Constant or a bare Name (loop iter / symbol), never a Slice, Ellipsis or
    newaxis."""
    if isinstance(elt, ast.Constant):
        return elt.value is not Ellipsis and elt.value is not None
    return isinstance(elt, ast.Name)


REDUCTION_NAMES: set[str] = {
    "sum",
    "mean",
    "prod",
    "std",
    "var",
    "min",
    "max",
    "argmin",
    "argmax",
    "any",
    "all",
    "count_nonzero",
    "median",
    # dot/vdot/matmul-like calls also collapse the operand to a scalar or
    # lower-rank array -- not preserved at the operand's iter extent. Same for
    # np.linalg.* (norm collapses to scalar; lstsq returns a tuple).
    "dot",
    "vdot",
    "inner",
    "norm",
    "det",
    "lstsq",
}


def is_reduction_call(call: ast.Call) -> bool:
    """True for ``np.sum(...)``/``arr.sum(...)``/similar reduction calls --
    only the registry of names above counts."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr in REDUCTION_NAMES
    if isinstance(call.func, ast.Name):
        return call.func.id in REDUCTION_NAMES
    return False


def is_const_one(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value == 1


def name_id(node: ast.AST) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def truthy(x: ast.expr) -> ast.Compare:
    """``x != 0`` -- element truthiness. On a boolean mask the Fortran emitter
    folds ``<logical> /= 0`` back to the bare logical; C reads it as 0/1."""
    return ast.Compare(left=x, ops=[ast.NotEq()], comparators=[const_(0)])


def falsy(x: ast.expr) -> ast.Compare:
    return ast.Compare(left=x, ops=[ast.Eq()], comparators=[const_(0)])


def if_set(
    test_fn: Callable[[ast.expr], ast.expr], value_fn: Callable[[ast.expr], ast.expr]
) -> Callable[[ast.expr, ast.expr, ast.expr], ast.stmt]:
    """Build an ``update_fn`` that, per element, tests ``test_fn(src)`` and on
    hit assigns ``value_fn(load)`` to the accumulator. Keeps the accumulator
    INTEGER (0/1 or a count) so no backend needs bool-as-int arithmetic (which
    Fortran rejects)."""

    def f_(store: ast.expr, load: ast.expr, src: ast.expr) -> ast.If:
        return ast.If(test=test_fn(src), body=[ast.Assign(targets=[store], value=value_fn(load))], orelse=[])

    return f_


def resolve_shape(arr_node: ast.expr, shape_table: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    if not isinstance(arr_node, ast.Name):
        raise NotImplementedError("reduction operand is not a bare Name")
    shape = shape_table.get(arr_node.id)
    if shape is None:
        raise NotImplementedError(f"shape of {arr_node.id!r} not in IR's shape table")
    return shape


def alloc_marker(name: str) -> ast.Assign:
    """``<name> = __hpcagent_bench_zeros__()`` -- the allocation-SITE marker. A local
    whose extent depends on a body-computed scalar can't be malloc'd at fn-top
    (the scalar is garbage there), so the emitter declares it NULL and defers the
    malloc to this marker, placed after the scalar's assignment. An expander that
    consumes the original Assign must re-emit the marker or the buffer stays
    NULL; for a fn-top-malloc'd static local the marker is a no-op, so emitting
    it unconditionally is always safe."""
    return ast.Assign(
        targets=[ast.Name(id=name, ctx=ast.Store())],
        value=ast.Call(func=ast.Name(id="__hpcagent_bench_zeros__", ctx=ast.Load()), args=[], keywords=[]),
    )


def cmp_(op: type[ast.cmpop]) -> Callable[[ast.expr, ast.expr], ast.expr]:
    """Return an op_fn that builds ``ast.Compare(left=x, ops=[op], comparators=[y])``."""
    return lambda x, y: ast.Compare(left=x, ops=[op()], comparators=[y])


def const_int(node: ast.expr | None) -> int | None:
    """A plain (possibly negative) int constant, or ``None``. A negative literal parses as
    ``UnaryOp(USub, Constant(n))`` -- not ``Constant(-n)`` -- so handle both."""
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -node.operand.value
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def mul_exts(exprs: Iterable[ast.expr]) -> ast.expr:
    """Left-folded product of the given extent expressions (``1`` when empty) -- used to
    size a ``reshape(-1)`` dimension from the source extent and the other target dims."""
    exprs = list(exprs)
    if not exprs:
        return const_(1)
    prod = exprs[0]
    for e in exprs[1:]:
        prod = ast.BinOp(left=prod, op=ast.Mult(), right=e)
    return prod


def flat_index_(iters: list[str], shape: tuple[str, ...]) -> ast.expr:
    """Row-major flat index ``((i0*d1 + i1)*d2 + i2)...`` for ``iters`` over
    ``shape``."""
    idx: ast.expr = name_(iters[0])
    for k in range(1, len(iters)):
        idx = ast.BinOp(
            left=ast.BinOp(left=idx, op=ast.Mult(), right=const_or_name(shape[k])), op=ast.Add(), right=name_(iters[k])
        )
    return idx
