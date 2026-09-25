"""Subscript entry helpers shared by the view and chain rewrites."""

import ast
import copy
from collections.abc import Sequence

from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import const_int, slice_step_any, step_is_negative
from hpcagent_bench.translators.numpyto_common.lowering.mathfuncs import MATH_BUILTINS
from hpcagent_bench.translators.numpyto_common.lowering.shape_reads import const_int_index, is_newaxis


def slice_dims(node: ast.Subscript) -> list[ast.AST]:
    """Return per-axis slice entries (either ``Slice`` or non-slice index)."""
    sl = node.slice
    if isinstance(sl, ast.Tuple):
        return list(sl.elts)
    return [sl]


def has_any_slice(node: ast.AST) -> bool:
    """``True`` iff ``node`` is a subscript whose dim list contains a ``Slice``."""
    if not isinstance(node, ast.Subscript):
        return False
    return any(isinstance(d, ast.Slice) for d in slice_dims(node))


def advanced_runs(dims: list[ast.AST]) -> list[list[int]]:
    """Group subscript ``dims`` positions into maximal runs of ADVANCED entries.

    numpy counts a plain scalar index as "advanced" for this purpose, same as an
    index array -- only a real ``Slice`` or a newaxis breaks a run (numpy docs:
    "not x[arr1, :, 1] since 1 is an advanced index in this regard"). Two or more
    runs means the advanced indices are SEPARATED, and numpy moves their broadcast
    result to the FRONT instead of leaving it in place; callers that only implement
    the in-place (single-run) placement use this to detect and refuse that case."""
    runs: list[list[int]] = []
    cur: list[int] = []
    for i, d in enumerate(dims):
        if isinstance(d, ast.Slice) or is_newaxis(d):
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.append(i)
    if cur:
        runs.append(cur)
    return runs


def slice_free_gather_layout(
    dims: list[ast.AST], run_rank: int, source_rank: int
) -> tuple[list[ast.AST], int, int, int]:
    """``(entries that read a source axis, first result axis of the broadcast block, result rank, implicit
    trailing axes)`` of a gather with no slice. A newaxis inserts a unit result axis and reads nothing. The
    block stays behind the newaxes before it, and moves to the FRONT once a newaxis separates advanced entries."""
    kept = [d for d in dims if not is_newaxis(d)]
    trailing = max(0, source_rank - len(kept))
    runs = advanced_runs(dims)
    before = 0 if len(runs) > 1 else len(dims[: runs[0][0]])
    return kept, before, len(dims) - len(kept) + run_rank + trailing, trailing


def basic_axis_count(dims: Sequence[ast.AST]) -> int:
    """Result axes the slices and newaxes of a subscript add, one each."""
    return sum(1 for d in dims if isinstance(d, ast.Slice) or is_newaxis(d))


def name_of_subscript(node: ast.Subscript) -> str | None:
    return node.value.id if isinstance(node.value, ast.Name) else None


def iter_var_name(axis: int) -> str:
    """Stable iter-var generator: ``si0``, ``si1``, ``si2``, ..."""
    return f"si{axis}"


#: Spellings an elementwise ufunc has ALREADY been lowered to by the time the slice-to-scalar
#: rewriter runs -- ``np.maximum`` becomes a bare ``fmax`` well before this pass.
#: The POST-rename spellings of the elementwise intrinsics: ``MathRewriter`` turns ``np.atan2``
#: into a bare ``atan2``, and the per-element rewriter has to recognise the renamed form to
#: subscriptify its arguments. Restating four of them left ``atan2``/``hypot``/``asin``/``pow`` with
#: whole-array pointers inside a per-element store, which does not compile. Derived, not restated.
LOWERED_ELEMENTWISE: set[str] = set(MATH_BUILTINS.values()) | {"max", "min"}


def np_func_name(func: ast.AST) -> str | None:
    """The elementwise function a call names, either spelling: ``np.maximum`` before lowering,
    a bare ``fmax`` after it. ``None`` when the callee is neither."""
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in ("np", "numpy"):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def const_(value: int) -> ast.Constant:
    return ast.Constant(value=value)


def binop(left: ast.AST, op, right: ast.AST) -> ast.BinOp:
    return ast.BinOp(left=left, op=op, right=right)


def gather_slice_offset(e: ast.Slice) -> ast.AST | None:
    """Source offset of result element 0 of a step-free slice, or ``None`` when the axis is not one
    a gather can bind at ``offset + iter``: a strided slice reads every ``step``-th element and a
    negative start counts from the end, neither of which a bare ``offset + iter`` expresses."""
    if e.step is not None:
        return None
    if e.lower is None:
        return const_(0)
    lo = const_int(e.lower)
    if lo is not None:
        return None if lo < 0 else const_(lo)
    return None if isinstance(e.lower, ast.UnaryOp) and isinstance(e.lower.op, ast.USub) else e.lower


def shift_index(idx: ast.AST, offset: ast.AST) -> ast.AST:
    """``idx + offset``, or just ``idx`` when the offset is a literal zero."""
    if isinstance(offset, ast.Constant) and offset.value == 0:
        return idx
    return binop(idx, ast.Add(), copy.deepcopy(offset))


def is_scalar_index(elt: ast.expr) -> bool:
    """A subscript element that selects (consumes) a single source axis: an int
    Constant, a bare Name (loop iter / symbol), or integer ARITHMETIC over those
    (``hn[2 * l][:]``) -- NOT a Slice, Ellipsis, or newaxis (``None``)."""
    if isinstance(elt, ast.Constant):
        return elt.value is not Ellipsis and elt.value is not None
    if isinstance(elt, ast.BinOp):
        return is_scalar_index(elt.left) and is_scalar_index(elt.right)
    if isinstance(elt, ast.UnaryOp):
        return is_scalar_index(elt.operand)
    return isinstance(elt, ast.Name)


def rebases_onto_view_axis(view_slice: ast.Slice, use_slice: ast.Slice) -> bool:
    """Can ``use_slice`` be rebased onto view axis ``view_slice`` by :func:`compose_kept_axis`?

    That algebra is ``start + step*use``, which is only the numpy answer when both slices run
    forward and no bound counts from an end: a negative literal bound is measured from the view's
    LAST element, and a use stop under an existing view stop is the ``min`` numpy clamps to, which
    the composition would silently drop.
    """
    if has_negative_step([view_slice, use_slice]):
        return False
    for bound in (use_slice.lower, use_slice.upper, use_slice.step):
        k = None if bound is None else const_int_index(bound)
        if k is not None and k < 0:
            return False
    return view_slice.upper is None or use_slice.upper is None


def view_scale(step: ast.expr | None, factor: ast.expr) -> ast.expr:
    """``step * factor``, or a bare copy of ``factor`` when ``step`` is the implicit 1."""
    if step is None or (isinstance(step, ast.Constant) and step.value == 1):
        return copy.deepcopy(factor)
    return ast.BinOp(left=copy.deepcopy(step), op=ast.Mult(), right=copy.deepcopy(factor))


def view_offset(start: ast.expr | None, step: ast.expr | None, index: ast.expr) -> ast.expr:
    """``start + step * index``, dropping the ``start`` term when it is the implicit 0."""
    scaled = view_scale(step, index)
    if start is None or (isinstance(start, ast.Constant) and start.value == 0):
        return scaled
    return ast.BinOp(left=copy.deepcopy(start), op=ast.Add(), right=scaled)


def compose_kept_axis(view_slice: ast.Slice, use_dim: ast.expr) -> ast.expr:
    """Compose one KEPT (Slice) view axis with the use-site index/slice landing on it.

    ``view_slice`` is the view's own ``start:stop:step`` on the underlying base axis
    (any part possibly ``None``, meaning the numpy default). A further slice
    ``a:b:c`` on the view axis composes to ``(start+step*a):(start+step*b):(step*c)``
    on the base axis; a bare use ``:`` reuses the view's own bound on that side
    unchanged. A scalar use index ``j`` composes to the single point
    ``start + step*j`` -- numpy squeeze then drops the axis, same as it would for a
    direct integer index into the base array.
    """
    vstart, vstop, vstep = view_slice.lower, view_slice.upper, view_slice.step
    if isinstance(use_dim, ast.Slice):
        ustart, ustop, ustep = use_dim.lower, use_dim.upper, use_dim.step
        new_lower = copy.deepcopy(vstart) if ustart is None else view_offset(vstart, vstep, ustart)
        new_upper = copy.deepcopy(vstop) if ustop is None else view_offset(vstart, vstep, ustop)
        new_step = copy.deepcopy(vstep) if ustep is None else view_scale(vstep, ustep)
        return ast.Slice(lower=new_lower, upper=new_upper, step=new_step)
    return view_offset(vstart, vstep, use_dim)


def has_negative_step(elts: list[ast.expr]) -> bool:
    """``True`` iff any ``Slice`` among ``elts`` carries a literal NEGATIVE step.

    A negative step flips numpy's default bounds (``a[::-2]`` starts at the LAST
    element, not 0), which :func:`compose_kept_axis`'s ``start + step*index``
    algebra assumes is never the case (same "positive stride" convention
    :func:`slice_step_expr` documents for the rest of this file). A SYMBOLIC step
    is never flagged -- it is always emitted as a positive stride, per that same
    convention -- only a literal negative one is provably wrong to compose.
    """
    return any(isinstance(e, ast.Slice) and step_is_negative(slice_step_any(e)) for e in elts)


def is_fancy_dim(e: ast.expr, array_shapes: dict[str, list[str]]) -> bool:
    """``True`` for a subscript element that is an ADVANCED (gather) index rather than
    a basic scalar/slice one: an index-array Name, or a newaxis/Ellipsis."""
    if isinstance(e, ast.Slice):
        return False
    if is_newaxis(e):
        return True
    return isinstance(e, ast.Name) and e.id in array_shapes
