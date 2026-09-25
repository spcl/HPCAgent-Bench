"""Axis reductions, keepdims, masked reductions, ``ufunc.reduce`` and negative-axis normalisation."""

import ast

from hpcagent_bench.translators.numpyto_common.subscripts import is_newaxis
from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import (
    REDUCE_FNS,
    RankedRewritePass,
    RewritePass,
    const_int,
    np_attr,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.hoist import HoistForm, HoistTables, ValueHoist
from hpcagent_bench.translators.numpyto_common.numpy_desugar.kinds import dtype_kind
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import expr_rank


def axis_list(ax: ast.AST | None, rank: int) -> list[int] | None:
    """An ``axis=k`` / ``axis=(1, 2)`` node -> sorted non-negative axis indices, or None when it is not all
    constant ints in range."""
    if ax is None:
        return None

    def in_range(v: int | None) -> bool:
        # An axis outside [-rank, rank) means the rank estimate is wrong: bail rather than wrap it.
        return v is not None and -rank <= v < rank

    if isinstance(ax, (ast.Tuple, ast.List)):
        vals = [const_int(e) for e in ax.elts]
        if not vals or any(not in_range(v) for v in vals):
            return None
        return sorted({v % rank for v in vals})
    v = const_int(ax)
    return [v % rank] if in_range(v) else None


def reduce_axis_stmts(
    tname: str,
    sname: str,
    op: str,
    axes: list[int],
    rank: int,
    ctr: int,
    keepdims: bool = False,
    elem_is_float: bool = False,
    ddof: int = 0,
    elem_kind: str | None = None,
) -> list[ast.stmt]:
    """Statements reducing ``sname`` over the sorted ``axes`` into a new ``tname`` with an explicit loop nest.

    The numba/pythran form of ``np.<op>(x, axis=..., keepdims=...)``: numba rejects ``keepdims`` and a tuple
    axis over a >4-D array, pythran rejects ``keepdims``. ``keepdims`` keeps each reduced axis as a size-1
    output dim."""
    p = f"__rd{ctr}"
    axset = set(axes)
    out_axes = [i for i in range(rank) if i not in axset]
    d = [f"{p}_d{i}" for i in range(rank)]
    lines: list[str] = [f"{d[i]} = {sname}.shape[{i}]" for i in range(rank)]
    is_arg = op in ("argmin", "argmax")
    # ``mean``/``std``/``var`` keep a FLOAT input's dtype; an integer / bool / unknown input gives float64.
    float_res = f"{sname}.dtype" if elem_is_float else "np.float64"
    # ``sum``/``prod`` over bool or integer input accumulate in int64, as numpy upcasts to the platform
    # int; the input width would wrap. min/max pick an ELEMENT, so they keep the input dtype.
    acc_res = "np.int64" if (op in ("sum", "prod") and elem_kind in ("int", "bool")) else f"{sname}.dtype"
    dtype = (
        "np.int64"
        if is_arg
        else "np.bool_"
        if op in ("any", "all")
        else float_res
        if op in ("mean", "std", "var")
        else acc_res
    )
    shape_dims = [(d[i] if i in out_axes else "1") for i in range(rank)] if keepdims else [d[i] for i in out_axes]
    lines.append(f"{tname} = np.empty(({''.join(s + ', ' for s in shape_dims)}), {dtype})")
    o = {i: f"{p}_k{i}" for i in out_axes}
    ind = ""
    for i in out_axes:
        lines.append(f"{ind}for {o[i]} in range({d[i]}):")
        ind += "    "
    tgt_idx = [(o[i] if i in out_axes else "0") for i in range(rank)] if keepdims else [o[i] for i in out_axes]
    tgt = f"{tname}[{', '.join(tgt_idx)}]"
    jv = {ax: f"{p}_j{ax}" for ax in axes}
    count = " * ".join(d[ax] for ax in axes)

    def elem(seed: bool = False):
        # Reduced axes take their loop var, or 0 for the comparison seed.
        return f"{sname}[{', '.join(('0' if seed else jv[i]) if i in axset else o[i] for i in range(rank))}]"

    def reduce_loops(base_ind: str, body: list[str]) -> None:
        cur = base_ind
        for ax in axes:
            lines.append(f"{cur}for {jv[ax]} in range({d[ax]}):")
            cur += "    "
        for b in body:
            lines.append(f"{cur}{b}")

    if op in ("any", "all"):
        lines.append(f"{ind}{tgt} = {'True' if op == 'all' else 'False'}")
        if op == "all":
            reduce_loops(ind, [f"if not {elem()}:", f"    {tgt} = False"])
        else:
            reduce_loops(ind, [f"if {elem()}:", f"    {tgt} = True"])
    elif op == "sum":
        lines.append(f"{ind}{tgt} = 0")
        reduce_loops(ind, [f"{tgt} += {elem()}"])
    elif op == "prod":
        lines.append(f"{ind}{tgt} = 1")
        reduce_loops(ind, [f"{tgt} *= {elem()}"])
    elif op == "mean":
        lines.append(f"{ind}{tgt} = 0.0")
        reduce_loops(ind, [f"{tgt} += {elem()}"])
        lines.append(f"{ind}{tgt} = {tgt} / ({count})")
    elif op in ("var", "std"):
        # Two passes: the mean (/ N), then the squared deviations / (N - ddof), as numpy.
        var_denom = f"({count}) - {ddof}" if ddof else f"({count})"
        m, dv = f"{p}_m", f"{p}_dv"
        lines.append(f"{ind}{m} = 0.0")
        reduce_loops(ind, [f"{m} += {elem()}"])
        lines.append(f"{ind}{m} = {m} / ({count})")
        lines.append(f"{ind}{tgt} = 0.0")
        reduce_loops(ind, [f"{dv} = {elem()} - {m}", f"{tgt} += {dv} * {dv}"])
        lines.append(f"{ind}{tgt} = {tgt} / ({var_denom})")
        if op == "std":
            lines.append(f"{ind}{tgt} = np.sqrt({tgt})")
    elif op in ("min", "amin", "max", "amax"):
        cmp = "<" if op in ("min", "amin") else ">"
        # numpy min/max PROPAGATE NaN. `elem != elem` captures a NaN element into tgt, and once tgt is NaN
        # every comparison is False, so no finite element displaces it.
        e = elem()
        lines.append(f"{ind}{tgt} = {elem(seed=True)}")
        reduce_loops(ind, [f"if {e} != {e} or {e} {cmp} {tgt}:", f"    {tgt} = {e}"])
    else:  # argmin / argmax -- single axis (the hoister rejects tuple-axis arg*)
        ax = axes[0]
        cmp = "<" if op == "argmin" else ">"
        best = f"{p}_best"
        e = elem()
        lines.append(f"{ind}{best} = {elem(seed=True)}")
        lines.append(f"{ind}{tgt} = 0")
        lines.append(f"{ind}for {jv[ax]} in range(1, {d[ax]}):")
        # numpy argmin/argmax return the index of the FIRST NaN. `best == best` is False once best is NaN,
        # locking that index in; `elem != elem` lets a NaN win over a finite running best.
        lines.append(f"{ind}    if {best} == {best} and ({e} != {e} or {e} {cmp} {best}):")
        lines.append(f"{ind}        {best} = {e}")
        lines.append(f"{ind}        {tgt} = {jv[ax]}")
    return ast.parse("\n".join(lines)).body


def reduce_call_parts(node: ast.Call, kw: dict[str | None, ast.expr]) -> tuple[str, ast.expr, ast.expr | None] | None:
    """``(op, operand, axis)`` of ``np.<op>(x, axis=k)`` or of the method form ``x.<op>(axis=k)``, else None."""
    npop = np_attr(node)
    if npop is not None and npop in REDUCE_FNS and node.args:
        return npop, node.args[0], kw.get("axis") or (node.args[1] if len(node.args) > 1 else None)
    func = node.func
    if (
        isinstance(func, ast.Attribute)
        and func.attr in REDUCE_FNS
        and not (isinstance(func.value, ast.Name) and func.value.id in ("np", "numpy"))
    ):
        return func.attr, func.value, kw.get("axis") or (node.args[0] if node.args else None)
    return None


def reduce_ddof(op: str, ddof: ast.expr | None) -> int | None:
    """The ``ddof`` a var/std reduction divides by (0 for every other op), or None when it is not a literal int."""
    if op not in ("var", "std") or ddof is None:
        return 0
    if isinstance(ddof, ast.Constant) and isinstance(ddof.value, int) and not isinstance(ddof.value, bool):
        return ddof.value
    return None


def hoist_reduce_axis(node: ast.AST, hoist: ValueHoist) -> ast.expr | None:
    """``np.<reduce>(x, axis=k)`` or ``x.<reduce>(axis=k)`` -> the temp its reduction loop fills.

    A non-Name ``x`` is hoisted to a temp first. Left verbatim: a non-constant axis or ddof, a rank<2 operand,
    and a reduction over every axis without keepdims (the backend's scalar form)."""
    if not isinstance(node, ast.Call):
        return None
    kw = {k.arg: k.value for k in node.keywords}
    parts = reduce_call_parts(node, kw)
    if parts is None:
        return None
    op, arg, ax = parts
    rank = expr_rank(arg, hoist.tables.ranks)
    if rank is None or rank < 2:
        return None
    axes = axis_list(ax, rank)
    if not axes:
        return None
    kd = kw.get("keepdims")
    keepdims = isinstance(kd, ast.Constant) and kd.value is True
    if len(axes) == rank and not keepdims:
        return None  # every axis reduced -> a scalar; leave the backend's full reduction
    if op in ("argmin", "argmax") and len(axes) > 1:
        return None  # numpy itself rejects a tuple axis for argmin/argmax
    ddof = reduce_ddof(op, kw.get("ddof"))
    if ddof is None:
        # Refused before the operand is hoisted, whose temp would otherwise stay unread and collide.
        return None
    if isinstance(arg, ast.Name):
        sname = arg.id
    else:
        sname = f"__rsrc{hoist.ctr}"
        hoist.queue([f"{sname} = {ast.unparse(arg)}"])
    temp = f"__rdo{hoist.ctr}"
    elem_kind = dtype_kind(arg, hoist.tables.dtypes)
    hoist.pre.extend(
        reduce_axis_stmts(temp, sname, op, axes, rank, hoist.ctr, keepdims, elem_kind == "float", ddof, elem_kind)
    )
    hoist.ctr += 1
    return ast.Name(id=temp, ctx=ast.Load())


REDUCE_AXIS_HOIST = HoistForm(frozenset(REDUCE_FNS), (), hoist_reduce_axis)


#: Axis reductions the DaCe frontend lowers to its own ``Reduce`` node, which canon then lowers to a
#: library reduction; the loop nest would hide them. Float only: numpy widens an integer ``sum``'s
#: accumulator, the ``Reduce`` does not. No ``keepdims`` either, which the DaCe frontend lacks.
DACE_NATIVE_REDUCE_FNS = frozenset({"sum", "prod", "mean", "min", "max", "amin", "amax"})


def hoist_reduce_axis_unless_native(node: ast.AST, hoist: ValueHoist) -> ast.expr | None:
    """:func:`hoist_reduce_axis`, declining a float reduction the DaCe frontend lowers itself."""
    if isinstance(node, ast.Call) and not any(k.arg == "keepdims" for k in node.keywords):
        parts = reduce_call_parts(node, {k.arg: k.value for k in node.keywords})
        if (
            parts is not None
            and parts[0] in DACE_NATIVE_REDUCE_FNS
            and dtype_kind(parts[1], hoist.tables.dtypes) == "float"
        ):
            return None
    return hoist_reduce_axis(node, hoist)


DACE_REDUCE_AXIS_HOIST = HoistForm(frozenset(REDUCE_FNS), (), hoist_reduce_axis_unless_native)


def keepdims_index(axes: list[int]) -> list[ast.expr] | None:
    """Subscript entries putting a length-1 axis back at each of ``axes`` WITHOUT knowing the operand's rank.

    An ``...`` absorbs the untouched axes: non-negative axes anchor at the FRONT (``axis=1`` ->
    ``[:, None, ...]``), negative ones at the BACK (``axis=-2`` -> ``[..., None, :]``). Mixed signs need the
    rank to interleave, so they get ``None``.
    """
    if all(a >= 0 for a in axes):
        entries: list[ast.expr] = [ast.Constant(value=None) if i in axes else ast.Slice() for i in range(max(axes) + 1)]
        return entries + [ast.Constant(value=Ellipsis)]
    if all(a < 0 for a in axes):
        return [ast.Constant(value=Ellipsis)] + [
            ast.Constant(value=None) if i in axes else ast.Slice() for i in range(min(axes), 0)
        ]
    return None


class KeepdimsToNewaxis(RewritePass):
    """``np.sum(x, axis=1, keepdims=True)`` -> ``np.sum(x, axis=1)[:, None, ...]``.

    dace's reductions take no ``keepdims`` argument, so the kwarg rejects the whole program. Runs after
    :func:`hoist_reduce_axis`, which loop-lowers every call whose operand rank it knows; this takes the
    rest, hence the rank-free :func:`keepdims_index`.

    Left alone: no axis (every axis kept needs the rank), a non-constant or mixed-sign axis, the
    ``x.sum(...)`` method form, and a ``keepdims`` that is not a literal ``True``."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        kw = next((k for k in node.keywords if k.arg == "keepdims"), None)
        if kw is None or not node.args or np_attr(node) not in REDUCE_FNS:
            return node
        if not (isinstance(kw.value, ast.Constant) and kw.value.value is True):
            return node
        ax = next((k.value for k in node.keywords if k.arg == "axis"), None) or (
            node.args[1] if len(node.args) > 1 else None
        )
        given = ax.elts if isinstance(ax, (ast.Tuple, ast.List)) else ([ax] if ax is not None else [])
        axes = [const_int(e) for e in given]
        if not axes or None in axes:
            return node
        entries = keepdims_index(sorted(axes))
        if entries is None:
            return node
        node.keywords = [k for k in node.keywords if k.arg != "keepdims"]
        self.changed = True
        index = ast.Tuple(elts=entries, ctx=ast.Load())
        return ast.copy_location(ast.Subscript(value=node, slice=index, ctx=ast.Load()), node)


#: Full reductions of a boolean-mask select this lowers.
MASKED_REDUCE_OPS = {"mean"}


def masked_reduce_of(node: ast.AST, gathers: dict[str, tuple]):
    """``v.mean()`` / ``np.mean(v)`` (a full reduction of a name in ``gathers``) -> ``(op, name)``, else None."""
    if not isinstance(node, ast.Call) or node.keywords:
        return None
    f = node.func
    if (
        isinstance(f, ast.Attribute)
        and f.attr in MASKED_REDUCE_OPS
        and not node.args
        and isinstance(f.value, ast.Name)
        and f.value.id in gathers
    ):
        return f.attr, f.value.id
    if (
        np_attr(node) in MASKED_REDUCE_OPS
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in gathers
    ):
        return np_attr(node), node.args[0].id
    return None


def masked_reduce_lines(temp: str, a: str, mask: str, rank: int, op: str, p: str) -> list[str]:
    """Source lines reducing ``a[mask]`` into scalar ``temp`` with an accumulate loop; an empty selection
    gives ``np.nan``, as numpy's mean-of-empty."""
    idx = ", ".join(f"{p}_i{d}" for d in range(rank))
    lines = [f"{p}_s = 0.0", f"{p}_n = 0"]
    deep = ""
    for d in range(rank):
        lines.append(f"{deep}for {p}_i{d} in range({a}.shape[{d}]):")
        deep += "    "
    lines += [f"{deep}if {mask}[{idx}]:", f"{deep}    {p}_s += {a}[{idx}]", f"{deep}    {p}_n += 1"]
    # ``mean`` is the only supported op.
    lines += [f"if {p}_n > 0:", f"    {temp} = {p}_s / {p}_n", "else:", f"    {temp} = np.nan"]
    return lines


def is_bool_mask(mask: ast.AST, a: ast.AST, ranks: dict[str, int], dtypes: dict[str, str]) -> bool:
    """True iff ``mask`` is a bool-kind array of ``a``'s rank, i.e. ``a[mask]`` is a boolean select."""
    if isinstance(mask, (ast.Tuple, ast.Slice)) or is_newaxis(mask):
        return False
    ar = expr_rank(a, ranks)
    if ar is None or expr_rank(mask, ranks) != ar:
        return False
    return dtype_kind(mask, dtypes) == "bool"


def masked_reduce_map(fn: ast.AST, ranks: dict[str, int], dtypes: dict[str, str]) -> dict[str, tuple]:
    """``{name: (a_Name, mask_ast)}`` for every ``name = a[mask]`` boolean select whose EVERY load is a
    supported masked reduction, so the select can be dropped and each reduction inlined."""
    gathers: dict[str, tuple] = {}
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and is_bool_mask(node.value.slice, node.value.value, ranks, dtypes)
        ):
            gathers[node.targets[0].id] = (node.value.value, node.value.slice)
    ok: dict[str, tuple] = {}
    for v, am in gathers.items():
        loads = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == v and isinstance(n.ctx, ast.Load))
        reds = sum(1 for n in ast.walk(fn) if masked_reduce_of(n, {v: am}) is not None)
        if reds and reds == loads:
            ok[v] = am
    return ok


def hoist_masked_reduce(node: ast.AST, hoist: ValueHoist) -> ast.expr | None:
    """``v.mean()`` / ``np.mean(v)`` over a vetted ``v = a[mask]`` -> the temp its accumulate loop fills.

    The select is a dynamic-length array pythran cannot type and dace cannot shape."""
    gathers = hoist.tables.gathers
    hit = masked_reduce_of(node, gathers)
    if hit is None:
        return None
    op, v = hit
    a, mask = gathers[v]
    p = f"__mr{hoist.ctr}"
    hoist.ctr += 1
    temp = f"{p}_o"
    hoist.queue(masked_reduce_lines(temp, a.id, ast.unparse(mask), expr_rank(a, hoist.tables.ranks), op, p))
    return ast.Name(id=temp, ctx=ast.Load())


def has_masked_selects(tables: HoistTables) -> bool:
    """Whether the scope holds a vetted masked select to drop and reduce."""
    return bool(tables.gathers)


def drops_masked_select(stmt: ast.stmt, tables: HoistTables) -> bool:
    """A vetted ``v = a[mask]`` select; every use of ``v`` inlines its own loop, so the drop is safe."""
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id in tables.gathers
        and isinstance(stmt.value, ast.Subscript)
    )


MASKED_REDUCE_HOIST = HoistForm(
    frozenset(MASKED_REDUCE_OPS), (), hoist_masked_reduce, has_masked_selects, drops_masked_select
)


#: numpy ops whose result keeps the operand's rank, so a negative ``axis=`` counts back from it.
AXIS_PRESERVING_OPS = {
    "flip",
    "roll",
    "cumsum",
    "cumprod",
    "nancumsum",
    "nancumprod",
    "sort",
    "argsort",
    "concatenate",
    "diff",
    "gradient",
}


#: ops that ADD one axis, so the axis-space is the operand rank PLUS one
#: (``np.stack((a, b), axis=-1)`` on rank-2 operands addresses axis 2).
AXIS_ADDING_OPS = {"stack", "expand_dims"}


class NormalizeNegativeAxis(RankedRewritePass):
    """``np.flip(a, axis=-1)`` -> ``np.flip(a, axis=1)`` for a rank-2 ``a``.

    pythran's ``np.flip``/``np.stack`` with ``axis=-1`` silently return the wrong result. The axis space is
    the first operand's rank, plus one for an axis-ADDING op; an unknown rank is left verbatim. Only the
    ``axis=`` keyword is normalized: the positional slot differs per op (``np.roll``'s second is the shift)."""

    def operand_rank(self, node: ast.Call) -> int | None:
        # A sequence operand (``np.stack((a, b))``) takes its first element's rank.
        if not node.args:
            return None
        a0 = node.args[0]
        if isinstance(a0, (ast.Tuple, ast.List)):
            return expr_rank(a0.elts[0], self.ranks) if a0.elts else None
        return expr_rank(a0, self.ranks)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        op = np_attr(node)
        adding = op in AXIS_ADDING_OPS
        if not adding and op not in AXIS_PRESERVING_OPS:
            return node
        kw = next((k for k in node.keywords if k.arg == "axis"), None)
        if kw is None:
            return node
        val = const_int(kw.value)
        if val is None or val >= 0:
            return node
        base = self.operand_rank(node)
        if base is None:
            return node
        pos = base + (1 if adding else 0) + val  # val < 0
        if pos < 0:
            return node  # rank estimate off -> leave verbatim rather than wrap wrong
        kw.value = ast.copy_location(ast.Constant(value=pos), kw.value)
        self.changed = True
        return node


#: ``np.<ufunc>.reduce`` -> the plain reducer it equals. A DaCe ``Reduce`` library node re-emits
#: reductions in ufunc form (``np.add.reduce(x, axis=k)``); the native backends have no ufunc
#: dispatch, so canonicalise to the reducer the loop-lowering already handles.
UFUNC_REDUCE_TO_CALL = {
    "add": "sum",
    "multiply": "prod",
    "maximum": "max",
    "minimum": "min",
    "logical_and": "all",
    "logical_or": "any",
}


class UfuncReduceToReducer(RewritePass):
    """``np.add.reduce(x, axis=k)`` -> ``np.sum(x, axis=k)`` (and prod/max/min/all/any).

    ``ufunc.reduce`` defaults to ``axis=0``, the reducer to a full reduction, so a missing axis becomes an
    explicit ``axis=0``. Runs before the elementwise-ufunc desugars, which would read ``np.add`` as an add."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        f = node.func
        if (
            isinstance(f, ast.Attribute)
            and f.attr == "reduce"
            and isinstance(f.value, ast.Attribute)
            and isinstance(f.value.value, ast.Name)
            and f.value.value.id in ("np", "numpy")
            and f.value.attr in UFUNC_REDUCE_TO_CALL
        ):
            self.changed = True
            has_axis = len(node.args) > 1 or any(k.arg == "axis" for k in node.keywords)
            keywords = list(node.keywords)
            if not has_axis:
                keywords.append(ast.keyword(arg="axis", value=ast.Constant(value=0)))
            return ast.copy_location(
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="np", ctx=ast.Load()),
                        attr=UFUNC_REDUCE_TO_CALL[f.value.attr],
                        ctx=ast.Load(),
                    ),
                    args=node.args,
                    keywords=keywords,
                ),
                node,
            )
        return node
