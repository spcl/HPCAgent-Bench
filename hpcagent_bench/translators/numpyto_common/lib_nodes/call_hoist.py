"""Hoist registered numpy calls out of expressions into temporaries."""

import ast
from collections.abc import Callable
from types import NotImplementedType

from hpcagent_bench.translators.numpyto_common import dtypes
from hpcagent_bench.translators.numpyto_common.lib_nodes.call_args import const_axis, kwarg_or_pos, read_axis_keepdims
from hpcagent_bench.translators.numpyto_common.lib_nodes.constructors import arange_count
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import (
    INT_PRESERVING_ELEMENTWISE,
    all_integer_operands,
    broadcast_extents,
    concat_operands_axis,
    iter_extent_of,
    sum_width_tokens,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import attr_call, const_int, reads_complex
from hpcagent_bench.translators.numpyto_common.lib_nodes.matmul_hoist import MatmulHoister
from hpcagent_bench.translators.numpyto_common.lib_nodes.registry import ELEMENTWISE_SHAPE_OPS, NP_CALL_EXPANDERS
from hpcagent_bench.translators.numpyto_common.lib_nodes.repeat import diff_operand
from hpcagent_bench.translators.numpyto_common.subscripts import has_slice_subscript

__all__ = [
    "AXIS_REDUCTIONS",
    "FFT_TRANSFORMS",
    "OUTPUT_SHAPE_RULES",
    "SCALAR_RESULT_OPS",
    "SHAPE_PRESERVING_OPS",
    "SPILL_FIRST_OPERAND",
    "UNHANDLED",
    "CallHoister",
    "allocator_shape",
    "bincount_shape",
    "concatenate_shape",
    "diagonal_shape",
    "diff_shape",
    "elementwise_shape",
    "fromfunction_shape",
    "hstack_shape",
    "leading_count_shape",
    "named_operand_shape",
    "numpy_call_key",
    "outer_shape",
    "permuted_shape",
    "reduction_shape",
    "reshape_name_shape",
    "reshape_tuple_shape",
    "routed_extent",
    "values_operand_shape",
]


def numpy_call_key(call: ast.Call) -> tuple[str, str] | None:
    """The expander registry key for a call: ``np.<name>`` -> ``("np", name)``, ``np.linalg.<name>`` ->
    ``("np", "linalg.<name>")``, ``<module>.<name>`` -> ``(module, name)``; None for anything else."""
    func = call.func
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return ("np" if func.value.id == "np" else func.value.id, func.attr)
        if (
            isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "np"
        ):
            return ("np", f"{func.value.attr}.{func.attr}")
    return None


#: What an output-shape rule returns for an argument form it does not cover: the next rule for the
#: op is tried, and ``None`` (decline to hoist) when none answers.
UNHANDLED = NotImplemented

type ShapeTokens = tuple[str, ...]
type RuleResult = ShapeTokens | None | NotImplementedType


def routed_extent(
    hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None
) -> RuleResult:
    """Size ``np.<op>(*args, **keywords)`` with :func:`iter_extent_of`, so the shape logic lives in
    one place: contractions (``tensordot``'s ``axes`` is often a KEYWORD; dropping it would default to
    ``axes=2`` and mis-size the temp), ``np.pad``, ``np.diag`` and the axis movers ``swapaxes`` /
    ``expand_dims`` / ``squeeze`` / ``moveaxis`` (NON_ELEMENTWISE, so without this they would
    silently decline to hoist, leaving ``q @ np.swapaxes(k, -1, -2)`` for the emitter)."""
    if len(args) < (2 if op in {"einsum", "tensordot", "inner"} else 1):
        return UNHANDLED
    call = attr_call("np", op, list(args))
    call.keywords = list(keywords or [])
    ext = iter_extent_of(call, hoister.shape_table)
    return UNHANDLED if ext is None else tuple(ast.unparse(e) for e in ext)


def bincount_shape(
    hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None
) -> RuleResult:
    """``np.bincount(idx, weights=w, minlength=M)`` -> exactly M slots (see expand_bincount)."""
    minlength = kwarg_or_pos(args, keywords or [], 2, "minlength") if args else None
    return UNHANDLED if minlength is None else (ast.unparse(minlength),)


def values_operand_shape(
    hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None
) -> RuleResult:
    """``np.searchsorted(a, v)``: one index per element of the VALUES operand ``v``."""
    ext = iter_extent_of(args[1], hoister.shape_table) if len(args) >= 2 else None
    return UNHANDLED if ext is None else tuple(ast.unparse(e) for e in ext)


def allocator_shape(
    hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None
) -> RuleResult:
    """linspace(start, stop, n) -> (n,); arange(stop) -> (stop,); arange(start, stop[, step]) -> its
    element count. The 3-arg form goes through arange_count: ``stop - start`` ignores the step, which
    over-allocates for step > 1 and is NEGATIVE for a step < 0."""
    if op == "linspace":
        return (ast.unparse(args[2]),) if len(args) >= 3 else UNHANDLED
    if len(args) == 1:
        return (ast.unparse(args[0]),)
    if len(args) == 2:
        return (ast.unparse(ast.BinOp(left=args[1], op=ast.Sub(), right=args[0])),)
    if len(args) >= 3:
        return (ast.unparse(arange_count(list(args[:3]))),)
    return UNHANDLED


def fromfunction_shape(
    hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None
) -> RuleResult:
    """``np.fromfunction(lambda..., (N, M))``: the SECOND arg is the shape."""
    if len(args) < 2:
        return UNHANDLED
    sh = args[1]
    elts = sh.elts if isinstance(sh, (ast.Tuple, ast.List)) else [sh]
    return tuple(ast.unparse(e) for e in elts)


def leading_count_shape(
    hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None
) -> RuleResult:
    """A 1-D result whose length is an argument: ``np.histogram(a, bins)`` -> ``hist`` of ``bins``
    (the ``[0]`` unwrap selects it); ``np.fft.fftfreq(n, d=...)`` -> ``n`` frequencies."""
    position = 1 if op == "histogram" else 0
    return (ast.unparse(args[position]),) if len(args) > position else UNHANDLED


def named_operand_shape(position: int) -> Callable[..., RuleResult]:
    """A rule for an op whose result has the declared shape of its Name operand at ``position``:
    ``np.linalg.inv``, ``cholesky``, the ``np.fft`` transforms, ``roll``, ``tril``, ``triu`` (first
    operand) and ``np.linalg.solve`` (x has b's shape)."""

    def rule(hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None) -> RuleResult:
        if len(args) <= position or not isinstance(args[position], ast.Name):
            return UNHANDLED
        shape = hoister.shape_table.get(args[position].id)
        return tuple(shape) if shape else UNHANDLED

    return rule


def reshape_name_shape(
    hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None
) -> RuleResult:
    """``np.reshape(a, shape)`` of a known Name: the shape arg, a single ``-1`` resolved to prod(source)
    / prod(other dims) -- lets ``a.ravel() @ a.ravel()`` (lowered to reshape) hoist out of the matmul."""
    if len(args) < 2 or not isinstance(args[0], ast.Name):
        return UNHANDLED
    src = hoister.shape_table.get(args[0].id)
    sh = args[1]
    elts = sh.elts if isinstance(sh, (ast.Tuple, ast.List)) else [sh]
    toks = [ast.unparse(e) for e in elts]
    if src is None:
        return UNHANDLED
    prod_src = "(" + ") * (".join(str(s) for s in src) + ")"
    if any(str(t).strip() == "-1" for t in toks):
        others = [t for t in toks if str(t).strip() != "-1"]
        if others:
            denom = "(" + ") * (".join(str(t) for t in others) + ")"
            neg = f"({prod_src}) / ({denom})"
        else:
            neg = f"({prod_src})"
        toks = [neg if str(t).strip() == "-1" else str(t) for t in toks]
    return tuple(str(t) for t in toks)


def concatenate_shape(
    hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None
) -> RuleResult:
    """``np.concatenate((a, b, ...), axis=k)`` -> the common shape, axis k summed."""
    if not args:
        return UNHANDLED
    try:
        unused, shapes, axis = concat_operands_axis(args, keywords, hoister.shape_table)
    except NotImplementedError:
        shapes = None
    if not shapes:
        return UNHANDLED
    base = list(shapes[0])
    base[axis] = "(" + ") + (".join(s[axis] for s in shapes) + ")"
    return tuple(base)


def elementwise_shape(
    hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None
) -> RuleResult:
    """Elementwise ops: the broadcast of ALL operand extents, not the first operand's -- the temp for
    ``np.maximum(a(M,), B(N, M))`` is ``(N, M)``, matching the expander's own broadcast iteration."""
    acc: tuple[ast.expr, ...] | None = None
    for arg in args:
        ext = iter_extent_of(arg, hoister.shape_table)
        if ext is None:
            continue
        acc = ext if acc is None else broadcast_extents(acc, ext)
    return UNHANDLED if acc is None else tuple(ast.unparse(e) for e in acc)


def hstack_shape(
    hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None
) -> RuleResult:
    """``np.hstack((a, b, c))``: axis 1 for 2-D Name operands, axis 0 for 1-D; the widths sum and the
    other axis is shared."""
    if not args:
        return UNHANDLED
    ops = list(args[0].elts) if (len(args) == 1 and isinstance(args[0], ast.Tuple)) else list(args)
    shapes = []
    for op_arg in ops:
        if not isinstance(op_arg, ast.Name):
            return None
        s = hoister.shape_table.get(op_arg.id)
        if not s:
            return None
        shapes.append(s)
    if not shapes:
        return None
    rank = len(shapes[0])
    if rank == 1:
        return (sum_width_tokens([s[0] for s in shapes]),)
    if rank == 2:
        return (shapes[0][0], sum_width_tokens([s[1] for s in shapes]))
    return None


def diff_shape(hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None) -> RuleResult:
    """``np.diff(a[, n=1][, axis])``: one fewer element along the axis (the last by default)."""
    if not args or not isinstance(args[0], ast.Name):
        return UNHANDLED
    shape = hoister.shape_table.get(args[0].id)
    if not shape:
        return None
    rank = len(shape)
    n_node = kwarg_or_pos(args, keywords, 1, "n")
    if n_node is not None and const_int(n_node) != 1:
        return None
    ax_node = kwarg_or_pos(args, keywords, 2, "axis")
    ax = rank - 1 if ax_node is None else const_axis(ax_node, rank)
    if ax is None:
        return None
    out = list(shape)
    ext = out[ax]
    out[ax] = str(int(ext) - 1) if ext.strip().isdigit() else f"({ext}) - 1"
    return tuple(out)


def permuted_shape(
    hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None
) -> RuleResult:
    """``triu`` / ``flip`` keep the Name operand's shape; ``np.transpose(A[, axes])`` honours the perm
    (positional or ``axes=``) and otherwise reverses the axes."""
    if not args or not isinstance(args[0], ast.Name):
        return UNHANDLED
    shape = hoister.shape_table.get(args[0].id)
    if not shape:
        return None
    if op != "transpose":
        return tuple(shape)
    perm_arg = kwarg_or_pos(args, keywords, 1, "axes")
    if isinstance(perm_arg, (ast.Tuple, ast.List)):
        perm = [e.value for e in perm_arg.elts if isinstance(e, ast.Constant) and isinstance(e.value, int)]
        if len(perm) == len(shape):
            return tuple(shape[p] for p in perm)
    return tuple(reversed(shape))


def reduction_shape(
    hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None
) -> RuleResult:
    """Axis-aware reductions: the reduced axes removed (size 1 under keepdims). ``argmax`` / ``argmin``
    return the index array over the kept axes; axis-aware ``linalg.norm`` is a per-line L2 reduction;
    ``var`` sizes as ``std`` does. The axis / keepdims come from the stash visit_Call set (``args``
    does not carry the parent call's keywords)."""
    if not (args and isinstance(args[0], ast.Name)):
        return UNHANDLED
    src_shape = hoister.shape_table.get(args[0].id)
    if not src_shape:
        return UNHANDLED
    kw_axes, kw_keep = hoister._cur_axis, hoister._cur_keepdims
    if kw_axes is None:
        return None  # scalar -- not array-shape
    if isinstance(kw_axes, int):
        kw_axes = [kw_axes]
    resolved = []
    for a in kw_axes:
        na = a + len(src_shape) if a < 0 else a
        if 0 <= na < len(src_shape):
            resolved.append(na)
    axes_set = set(resolved)
    if kw_keep:
        return tuple("1" if i in axes_set else s for i, s in enumerate(src_shape))
    return tuple(s for i, s in enumerate(src_shape) if i not in axes_set)


def reshape_tuple_shape(
    hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None
) -> RuleResult:
    """``np.reshape(x, (...))`` of any source: the tuple's entries, a ``-1`` resolved against a Name
    source's element count over the product of the other target dims (``/`` renders as integer
    division in C/Fortran)."""
    if len(args) < 2 or not isinstance(args[1], ast.Tuple):
        return UNHANDLED
    parts = []
    for e in args[1].elts:
        if const_int(e) is not None:
            parts.append(str(const_int(e)))
        elif isinstance(e, ast.Name):
            parts.append(e.id)
        else:
            parts.append(ast.unparse(e))
    neg1 = [i for i, p in enumerate(parts) if p.strip() == "-1"]
    src = args[0]
    src_shape = hoister.shape_table.get(src.id) if isinstance(src, ast.Name) else None
    if len(neg1) == 1 and src_shape:
        total = " * ".join(f"({t})" for t in src_shape)
        others = [p for j, p in enumerate(parts) if j != neg1[0]]
        denom = " * ".join(f"({p})" for p in others) if others else "1"
        parts[neg1[0]] = f"({total}) / ({denom})"
    return tuple(parts)


def outer_shape(
    hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None
) -> RuleResult:
    """``np.outer(a, b)`` / ``np.add.outer(a, b)`` of two 1-D operands: ``(len(a), len(b))``."""
    if len(args) != 2:
        return UNHANDLED
    a_ext = iter_extent_of(args[0], hoister.shape_table)
    b_ext = iter_extent_of(args[1], hoister.shape_table)
    if a_ext is not None and b_ext is not None and len(a_ext) == 1 and len(b_ext) == 1:
        return (ast.unparse(a_ext[0]), ast.unparse(b_ext[0]))
    return UNHANDLED


def diagonal_shape(
    hoister: "CallHoister", op: str, args: list[ast.expr], keywords: list[ast.keyword] | None
) -> RuleResult:
    """``np.diagonal(a)`` of a SQUARE rank-2 operand: one element per row. Unsized, the diagonal would
    stay inline inside e.g. ``np.tanh(...)``, where the elementwise scalariser has no cell to read."""
    if len(args) != 1:
        return UNHANDLED
    d_ext = iter_extent_of(args[0], hoister.shape_table)
    if d_ext is not None and len(d_ext) == 2 and ast.unparse(d_ext[0]) == ast.unparse(d_ext[1]):
        return (ast.unparse(d_ext[0]),)
    return UNHANDLED


#: ``(ops, rule)`` in the order :meth:`CallHoister.derive_output_shape` tries them; the first rule
#: covering the op that does not answer :data:`UNHANDLED` sizes the hoisted temp.
OUTPUT_SHAPE_RULES: tuple[tuple[frozenset[str] | set[str], Callable[..., RuleResult]], ...] = (
    (frozenset({"einsum", "tensordot", "inner"}), routed_extent),
    (frozenset({"bincount"}), bincount_shape),
    (frozenset({"searchsorted"}), values_operand_shape),
    (frozenset({"pad"}), routed_extent),
    (frozenset({"linspace", "arange"}), allocator_shape),
    (frozenset({"fromfunction"}), fromfunction_shape),
    (frozenset({"histogram"}), leading_count_shape),
    (frozenset({"linalg.inv"}), named_operand_shape(0)),
    (frozenset({"linalg.solve"}), named_operand_shape(1)),
    (frozenset({"fft.fftn", "fft.ifftn", "fft.fft", "fft.ifft"}), named_operand_shape(0)),
    (frozenset({"fft.fftfreq"}), leading_count_shape),
    (frozenset({"diag"}), routed_extent),
    (frozenset({"roll", "linalg.cholesky", "tril", "triu"}), named_operand_shape(0)),
    (frozenset({"swapaxes", "expand_dims", "squeeze", "moveaxis"}), routed_extent),
    (frozenset({"reshape"}), reshape_name_shape),
    (frozenset({"concatenate"}), concatenate_shape),
    (ELEMENTWISE_SHAPE_OPS, elementwise_shape),
    (frozenset({"hstack"}), hstack_shape),
    (frozenset({"diff"}), diff_shape),
    (frozenset({"transpose", "triu", "flip"}), permuted_shape),
    (
        frozenset({"sum", "max", "min", "mean", "prod", "std", "var", "argmax", "argmin", "linalg.norm"}),
        reduction_shape,
    ),
    (frozenset({"reshape"}), reshape_tuple_shape),
    (frozenset({"outer", "add.outer"}), outer_shape),
    (frozenset({"diagonal"}), diagonal_shape),
    (frozenset({"linalg.cholesky", "linalg.inv"}), named_operand_shape(0)),
)


#: ``np.<name>`` calls whose non-Name first operand spills to a temp before the call hoists, so the
#: expander sees a Name: the reductions, the arg-reductions (whose scaffold requires a Name), and the
#: shape-preserving index ops -- ``np.roll(psi_frag[f], m, axis)`` spills ``psi_frag[f]`` instead of
#: leaving the whole-array roll buried in a broadcast BinOp for the scalariser to mangle.
SPILL_FIRST_OPERAND: frozenset[tuple[str, str]] = frozenset(
    {
        ("np", k)
        for k in {
            "sum",
            "max",
            "min",
            "mean",
            "prod",
            "std",
            "var",
            "median",
            "any",
            "all",
            "count_nonzero",
            "argmax",
            "argmin",
            "repeat",
            "transpose",
            "reshape",
            "triu",
            "tril",
            "flip",
            "roll",
            "copy",
            "array",
            "bincount",
            "cumsum",
            "cumprod",
            "swapaxes",
            "expand_dims",
            "squeeze",
            "moveaxis",
        }
    }
    | {("np", "fft.fftn"), ("np", "fft.ifftn"), ("np", "fft.fft"), ("np", "fft.ifft")}
)

#: Calls whose hoisted result is a scalar (see :meth:`CallHoister.returns_scalar` for the exceptions).
SCALAR_RESULT_OPS = frozenset(
    {
        "sum",
        "max",
        "min",
        "mean",
        "prod",
        "std",
        "var",
        "dot",
        "vdot",
        "inner",
        "linalg.norm",
        "linalg.det",
        "argmax",
        "argmin",
        "any",
        "all",
        "count_nonzero",
        "median",
        "trace",
    }
)

#: Reductions that return an ARRAY when given an axis. ``var`` belongs here for the same reason
#: ``std`` does -- they are one op (``expand_var_or_std``, std is var plus a sqrt).
AXIS_REDUCTIONS = frozenset(
    {
        "sum",
        "max",
        "min",
        "mean",
        "prod",
        "std",
        "var",
        "argmax",
        "argmin",
        "any",
        "all",
        "count_nonzero",
        "linalg.norm",
    }
)

#: The ``np.fft`` transforms: shape-preserving, and complex-valued even from a real input.
FFT_TRANSFORMS = frozenset({"fft.fftn", "fft.ifftn", "fft.fft", "fft.ifft"})

#: Shape-preserving ops whose result inherits the source array's dtype: ``Xiv = np.reshape(Xi,
#: (xn * yn,))`` with ``Xi`` int64 keeps Xiv int64, not the default double.
SHAPE_PRESERVING_OPS = frozenset(
    {"reshape", "repeat", "copy", "array", "asarray", "ascontiguousarray", "transpose", "flip"}
)


class CallHoister(ast.NodeTransformer):
    """Hoist any registered ``np.*`` call buried in an expression to a fresh
    temp ``__cb<n>``; the expander then lowers ``__cb<n> = call(...)``. A
    scalar-returning call (reduction/dot/std) hoists to a scalar local; an
    array-returning call (copy/outer/transpose) hoists to an array temp whose
    shape is inferred from its arguments.
    """

    def __init__(
        self,
        shape_table: dict[str, tuple[str, ...]],
        scalar_temps: dict[str, bool],
        array_temps: dict[str, tuple[str, ...]],
        counter: list[int],
        local_dtypes: dict[str, str] | None = None,
        dim_aliases: dict[str, str] | None = None,
        blas: bool = False,
    ) -> None:
        self.shape_table = shape_table
        self.scalar_temps = scalar_temps
        self.array_temps = array_temps
        self.counter = counter
        #: Both forwarded to the nested ``MatmulHoister`` (see its docstring).
        self.dim_aliases: dict[str, str] = dim_aliases or {}
        self.blas = blas
        # Side-effect dtype table (shared with the lowering pipeline)
        # so a ``__cb<n>`` whose RHS contains complex literals or
        # complex-typed Name references is tagged ``complex128``.
        self.local_dtypes: dict[str, str] = local_dtypes if local_dtypes is not None else {}
        self.pre_stmts: list[ast.stmt] = []
        #: Never populated on this class; forwarded to the nested ``MatmulHoister``,
        #: which treats ``None`` the same as an empty sparse-array table.
        self.sparse: dict[str, object] | None = None
        #: Axis/keepdims of the reduction call ``visit_Call`` is currently hoisting;
        #: read back by ``derive_output_shape`` within that same call.
        self._cur_axis: list[int] | None = None
        self._cur_keepdims: bool = False

    def infer_complex(self, expr: ast.AST) -> bool:
        """``True`` iff ``expr`` reads a complex value (skipping ``.shape`` reads)."""
        return reads_complex(expr, self.local_dtypes)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.visit_call_children(node)
        self.hoist_argument_matmuls(node)
        key = numpy_call_key(node)
        if key in SPILL_FIRST_OPERAND and node.args and not isinstance(node.args[0], ast.Name):
            self.spill_first_operand(node)
        if key is None or key not in NP_CALL_EXPANDERS:
            return node
        self.stash_axis_keepdims(key, node)
        self.counter[0] += 1
        temp = f"__cb{self.counter[0]}"
        is_scalar = self.returns_scalar(key, node)
        if not is_scalar:
            shape = self.derive_output_shape(key, node.args, node.keywords)
            if shape is None:
                return node
            self.array_temps[temp] = shape
            self.shape_table[temp] = shape
            self.type_array_temp(temp, key[1], node)
        else:
            self.type_scalar_temp(temp, key[1], node)
        # Emit a ``__cb<n> = __hpcagent_bench_zeros__()`` marker first so the emit
        # walker can inline-declare the temp at the marker site -- required
        # when the temp's shape depends on an enclosing for-loop iter
        # (stockham_fft's ``R ** i``). The subsequent ``__cb<n> = call(...)``
        # then lowers into a per-element copy via the existing call-expansion path.
        if not is_scalar:
            self.pre_stmts.append(
                ast.Assign(
                    targets=[ast.Name(id=temp, ctx=ast.Store())],
                    value=ast.Call(func=ast.Name(id="__hpcagent_bench_zeros__", ctx=ast.Load()), args=[], keywords=[]),
                )
            )
        # Synthesise an Assign that the LibNodeRewriter will lower.
        self.pre_stmts.append(ast.Assign(targets=[ast.Name(id=temp, ctx=ast.Store())], value=node))
        return ast.Name(id=temp, ctx=ast.Load())

    def visit_call_children(self, node: ast.Call) -> None:
        """Visit the call's children -- except the ``np.diff`` count of ``np.repeat(src,
        np.diff(p))``: its telescoping sum (see expand_repeat / diff_operand) needs the ORIGINAL
        ``np.diff`` call form, and ``np.diff`` is itself a registered call that a plain visit would
        hoist into an opaque ``__cb<n>`` temp before the repeat expander ever ran."""
        if numpy_call_key(node) == ("np", "repeat") and len(node.args) >= 2 and diff_operand(node.args[1]) is not None:
            node.func = self.visit(node.func)
            node.args = [(a if i == 1 else self.visit(a)) for i, a in enumerate(node.args)]
            node.keywords = [self.visit(kw) for kw in node.keywords]
        else:
            self.generic_visit(node)

    def hoist_argument_matmuls(self, node: ast.Call) -> None:
        """Hoist the matmuls inside the call args first: ``np.maximum(input @ w1 + b1, 0)`` ->
        ``__mm1 = input @ w1; ...; np.maximum(__mm1 + b1, 0)``, so the elementwise expander sees a
        bare BinOp on Names, not a MatMult."""
        mm = MatmulHoister(
            self.shape_table,
            self.array_temps,
            self.counter,
            local_dtypes=self.local_dtypes,
            sparse=self.sparse,
            dim_aliases=self.dim_aliases,
            blas=self.blas,
        )
        node.args = [mm.visit(a) for a in node.args]
        self.pre_stmts.extend(mm.pre_stmts)

    def spill_first_operand(self, node: ast.Call) -> None:
        """Spill a sized non-Name first operand into a fresh ``__cb<n>`` temp, so the expander sees a
        Name: ``np.mean(a * b)`` -> ``__cb<n> = a * b; np.mean(__cb<n>)``.

        When the operand carries slice-bearing Subscripts (``np.max(x[:, 2i:2i+2, :], axis=(1, 2))``)
        the post-LibNodeRewriter lift can no longer recover x's per-statement shape, so the spill is
        the slice-LHS form instead -- marker + ``__cb[:, ...] = first`` -- which slice-fusion lowers
        into a per-element copy; otherwise ``__cb<n> = first``, which lower_prelude_calls turns into a
        per-element copy via WholeArrayAssignRewriter."""
        first = node.args[0]
        ext = iter_extent_of(first, self.shape_table)
        if ext is None:
            return
        self.counter[0] += 1
        temp = f"__cb{self.counter[0]}"
        shape = tuple(ast.unparse(e) for e in ext)
        self.array_temps[temp] = shape
        self.shape_table[temp] = shape
        if self.infer_complex(first):
            self.local_dtypes[temp] = "complex128"
        if has_slice_subscript(first):
            rank = len(shape)
            slice_form = (
                ast.Slice(lower=None, upper=None, step=None)
                if rank == 1
                else ast.Tuple(
                    elts=[ast.Slice(lower=None, upper=None, step=None) for unused in range(rank)],
                    ctx=ast.Load(),
                )
            )
            marker = ast.Assign(
                targets=[ast.Name(id=temp, ctx=ast.Store())],
                value=ast.Call(func=ast.Name(id="__hpcagent_bench_zeros__", ctx=ast.Load()), args=[], keywords=[]),
            )
            slice_lhs = ast.Subscript(value=ast.Name(id=temp, ctx=ast.Load()), slice=slice_form, ctx=ast.Store())
            self.pre_stmts.append(marker)
            self.pre_stmts.append(ast.Assign(targets=[slice_lhs], value=first))
        else:
            self.pre_stmts.append(ast.Assign(targets=[ast.Name(id=temp, ctx=ast.Store())], value=first))
        node.args[0] = ast.Name(id=temp, ctx=ast.Load())

    def stash_axis_keepdims(self, key: tuple[str, str], node: ast.Call) -> None:
        """Stash the call's axis / keepdims for :meth:`derive_output_shape`. ``linalg.norm``'s
        positional layout is ``(v, ord, axis, keepdims)``, unlike a reduction's 2nd-positional
        ``axis``: its ``ord`` is stripped first (mirroring ``expand_linalg_norm``), else a positional
        ord (``norm(a, 1)``) would read as ``axis=1``."""
        if key == ("np", "linalg.norm"):
            norm_args = [node.args[0]] + list(node.args[2:]) if node.args else []
            norm_kwargs = [kw for kw in node.keywords if kw.arg != "ord"]
            self._cur_axis, self._cur_keepdims = read_axis_keepdims(norm_args, norm_kwargs)
        else:
            self._cur_axis, self._cur_keepdims = read_axis_keepdims(node.args, node.keywords)

    def returns_scalar(self, key: tuple[str, str], node: ast.Call) -> bool:
        """Whether the hoisted call's result is a scalar. ``np.inner`` is scalar ONLY for rank-1 x
        rank-1; an axis-aware reduction given an axis returns an array."""
        if key[1] not in SCALAR_RESULT_OPS:
            return False
        if key[1] == "inner":
            ranks = [len(self.shape_table.get(a.id, ())) for a in node.args if isinstance(a, ast.Name)]
            if any(r > 1 for r in ranks):
                return False
        return not (key[1] in AXIS_REDUCTIONS and self._cur_axis is not None)

    def type_array_temp(self, temp: str, op: str, node: ast.Call) -> None:
        """The element dtype of an array temp, where it is not the double default: ``argmax`` /
        ``argmin`` produce int64 indices; complex operands (and every ``np.fft`` transform, even of
        a real input) give complex128; a shape-preserving op inherits its source's dtype; an
        all-integer elementwise ufunc, or an ``np.where`` over integer VALUES (the condition's dtype
        is ignored), stays int64 rather than round-tripping through a double."""
        if op in {"argmax", "argmin"}:
            self.local_dtypes[temp] = "int64"
        if self.infer_complex(node) or op in FFT_TRANSFORMS:
            self.local_dtypes[temp] = "complex128"
        if op in SHAPE_PRESERVING_OPS and node.args and temp not in self.local_dtypes:
            first = node.args[0]
            if isinstance(first, ast.Name):
                src_dt = self.local_dtypes.get(first.id)
                if src_dt:
                    self.local_dtypes[temp] = src_dt
        if (
            op in INT_PRESERVING_ELEMENTWISE
            and temp not in self.local_dtypes
            and all_integer_operands(node.args, self.local_dtypes)
        ):
            self.local_dtypes[temp] = "int64"
        if (
            op == "where"
            and len(node.args) == 3
            and temp not in self.local_dtypes
            and all_integer_operands(node.args[1:], self.local_dtypes)
        ):
            self.local_dtypes[temp] = "int64"

    def type_scalar_temp(self, temp: str, op: str, node: ast.Call) -> None:
        """Record a scalar temp and its dtype: complex operands give complex128; a value-preserving
        reduction (max / min / sum / prod) of an INTEGER-tagged Name keeps that integer dtype, so
        Fortran's running-max ``merge(int_elem, acc, ...)`` is not a kind mismatch (mean / std / var /
        median produce a float even from an int input)."""
        self.scalar_temps[temp] = True
        if self.infer_complex(node):
            self.local_dtypes[temp] = "complex128"
        elif op in {"max", "min", "sum", "prod"} and node.args and isinstance(node.args[0], ast.Name):
            src_dt = self.local_dtypes.get(node.args[0].id)
            if src_dt and dtypes.is_integer(src_dt):
                self.local_dtypes[temp] = src_dt

    def derive_output_shape(
        self, key: tuple[str, str], args: list[ast.expr], keywords: list[ast.keyword] | None = None
    ) -> tuple[str, ...] | None:
        """Shape tokens of the temp a hoisted ``np.<op>(*args)`` fills, from the first rule in
        :data:`OUTPUT_SHAPE_RULES` that answers for the op; None declines the hoist."""
        op = key[1]
        for ops, rule in OUTPUT_SHAPE_RULES:
            if op in ops:
                shape = rule(self, op, args, keywords)
                if shape is not UNHANDLED:
                    return shape
        return None
