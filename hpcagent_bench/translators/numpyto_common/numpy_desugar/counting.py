"""Counting and scatter forms: ``np.diff``, ``repeat``, ``bincount``, ``add.at``, ``searchsorted``, ``histogram``."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.ast_build import const_int, name_, store_, walk_nodes
from hpcagent_bench.translators.numpyto_common.emit_helpers.numpy_names import numpy_call_attr
from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import (
    DesugarError,
    RankedRewritePass,
    RewritePass,
    replace_call_with_name,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.hoist import HoistForm, ValueHoist
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import expr_rank
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ufuncs import ufunc_method_op

__all__ = [
    "AT_OPS",
    "HISTOGRAM_HOIST",
    "REPEAT_AXIS_HOIST",
    "AddAtInline",
    "BincountInline",
    "DiffToSliceDifference",
    "RepeatCountsInline",
    "SearchsortedMaterialize",
    "SegmentSumInline",
    "StripAstypeCopyKwarg",
    "hoist_histogram",
    "hoist_repeat_axis",
]


AT_OPS = {"add": "+=", "subtract": "-=", "multiply": "*="}


class DiffToSliceDifference(RewritePass):
    """``np.diff(p)`` -> ``p[1:] - p[:-1]``.

    Every python backend traces the slice form natively; dace otherwise falls back to a Python
    callback for the whole program. Only the single-argument form: ``n``/``axis``/``prepend`` compute
    something else and are left standing.
    """

    __slots__ = ()

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if numpy_call_attr(node) != "diff" or len(node.args) != 1 or node.keywords:
            return node
        base = node.args[0]
        self.changed = True
        tail = ast.Subscript(
            value=copy.deepcopy(base),
            slice=ast.Slice(lower=ast.Constant(value=1), upper=None, step=None),
            ctx=ast.Load(),
        )
        head = ast.Subscript(
            value=copy.deepcopy(base),
            slice=ast.Slice(lower=None, upper=ast.Constant(value=-1), step=None),
            ctx=ast.Load(),
        )
        return ast.copy_location(ast.BinOp(left=tail, op=ast.Sub(), right=head), node)


class StripAstypeCopyKwarg(RewritePass):
    """``x.astype(dt, copy=False)`` -> ``x.astype(dt)``.

    ``copy`` only decides whether numpy may return the input itself when the dtype already matches;
    the value is the same. dace's astype replacement rejects the keyword.
    """

    __slots__ = ()

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "astype"
            and any(k.arg == "copy" for k in node.keywords)
        ):
            node.keywords = [k for k in node.keywords if k.arg != "copy"]
            self.changed = True
        return node


def repeated_rows(node: ast.stmt) -> tuple[str, ast.expr, ast.expr] | None:
    """``R = np.repeat(np.arange(N[, dtype=...]), np.diff(P))`` -> ``(R, N, P)``: the row of every CSR entry."""
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
        return None
    call = node.value
    if not (isinstance(call, ast.Call) and numpy_call_attr(call) == "repeat" and len(call.args) == 2) or call.keywords:
        return None
    rows, counts = call.args
    if not (isinstance(rows, ast.Call) and numpy_call_attr(rows) == "arange" and len(rows.args) == 1):
        return None
    if not (isinstance(counts, ast.Call) and numpy_call_attr(counts) == "diff" and len(counts.args) == 1):
        return None
    return node.targets[0].id, rows.args[0], counts.args[0]


def segment_sum_call(node: ast.AST, rows: str, extent: ast.expr) -> ast.Call | None:
    """The ``np.bincount(rows, weights=W, minlength=N)`` in ``node`` that sums the CSR entries of each row."""
    for call in walk_nodes(node):
        if not (isinstance(call, ast.Call) and numpy_call_attr(call) == "bincount" and len(call.args) == 1):
            continue
        kw = {k.arg: k.value for k in call.keywords}
        if not (isinstance(call.args[0], ast.Name) and call.args[0].id == rows) or set(kw) != {"weights", "minlength"}:
            continue
        if ast.unparse(kw["minlength"]) == ast.unparse(extent):
            return call
    return None


class SegmentSumInline(RewritePass):
    """``np.bincount(np.repeat(np.arange(N), np.diff(P)), weights=W, minlength=N)`` -> per-row sums of ``W``.

    A CSR SpMV spells its row sums that way (minife). Lowered one call at a time it is a serial prefix-sum fill of
    one row index per entry, then a scatter-add through it; as one loop over the rows each sums its own entries
    ``W[P[i] - P[0] : P[i + 1] - P[0]]`` in index order -- bincount's order -- with no index array, no carried
    position, and a write ``out[i]`` the rows share nothing through.
    """

    __slots__ = ("fn", "ranks")

    def __init__(self, fn: ast.FunctionDef | ast.Module, ranks: dict[str, int]) -> None:
        super().__init__()
        #: The function the visited statements belong to: a row index read anywhere else in it stays.
        self.fn = fn
        self.ranks = ranks
        # A pair at the top level spans two of the statements the driver visits one at a time: rewritten here, before
        # it walks the body.
        while self.rewrite_block(fn, fn.body):
            self.changed = True

    def generic_visit(self, node: ast.AST) -> ast.AST:
        # Any statement list under the statement: inlined helpers put the pair inside the solver's loops.
        while any(self.rewrite_block(self.fn, body) for body in statement_lists(node)):
            self.changed = True
        return node

    def rewrite_block(self, fn: ast.FunctionDef | ast.Module, body: list[ast.stmt]) -> bool:
        """Rewrite the first pair in the statement list ``body``, in place; whether one was."""
        for k, stmt in enumerate(body):
            found = repeated_rows(stmt)
            if found is None:
                continue
            rows, extent, ptr = found
            if sum(1 for n in walk_nodes(fn) if isinstance(n, ast.Name) and n.id == rows) != 2:
                continue  # the binding and one bincount read, or not ours
            users = [j for j in range(k + 1, len(body)) if segment_sum_call(body[j], rows, extent) is not None]
            if len(users) != 1:
                continue
            j = users[0]
            call = segment_sum_call(body[j], rows, extent)
            if call is None:
                continue
            weights = {kw.arg: kw.value for kw in call.keywords}["weights"]
            name = f"__ss{self._ctr}"
            self._ctr += 1
            p, n, w = ast.unparse(ptr), ast.unparse(extent), ast.unparse(weights)
            # The entries summed in place when their array is an elementwise expression read nowhere else: minife's
            # ``values[:nnz] * x[cols[:nnz]]`` was a 27-per-row temporary written and read back every CG iteration.
            binding = single_use_binding(fn, body[:j], weights)
            entry = element_at(binding.value if binding else weights, ast.Name(f"{name}_j"), self.ranks)
            summand = f"{name}_w[{name}_j]" if entry is None else ast.unparse(entry)
            lines = [
                *([f"{name}_w = {w}"] if entry is None else []),
                f"{name} = np.zeros(({n},), dtype=np.float64)",
                f"for {name}_i in range({n}):",
                f"    {name}_s = 0.0",
                f"    for {name}_j in range({p}[{name}_i] - {p}[0], {p}[{name}_i + 1] - {p}[0]):",
                f"        {name}_s = {name}_s + {summand}",
                f"    {name}[{name}_i] = {name}_s",
            ]
            replace_call_with_name(body[j], call, name)
            pre = ast.parse("\n".join(lines)).body
            for new in pre:
                ast.copy_location(new, body[j])
                ast.fix_missing_locations(new)
            kept = [stmt for stmt in body[k + 1 : j] if entry is None or stmt is not binding]
            body[:] = body[:k] + kept + pre + body[j:]
            return True
        return False


def single_use_binding(fn: ast.AST, before: list[ast.stmt], weights: ast.expr) -> ast.Assign | None:
    """The ``W = expr`` among ``before`` that ``weights`` names, when that read is the only one of ``W`` in ``fn``."""
    if not isinstance(weights, ast.Name):
        return None
    bindings = [
        stmt
        for stmt in before
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == weights.id
    ]
    uses = sum(1 for n in walk_nodes(fn) if isinstance(n, ast.Name) and n.id == weights.id)
    if len(bindings) != 1 or uses != 2:
        return None
    binding = bindings[0]
    # Read at the bincount instead of the binding: no statement in between may change what it reads.
    reads = {n.id for n in walk_nodes(binding.value) if isinstance(n, ast.Name)}
    between = before[before.index(binding) + 1 :]
    return None if any(may_write(stmt, reads) for stmt in between) else binding


def may_write(stmt: ast.stmt, names: set[str]) -> bool:
    """Whether ``stmt`` may change one of ``names``: binds it, stores into it, or hands it to a call."""
    for node in walk_nodes(stmt):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in names:
            return True
        if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            base = node.value
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if isinstance(base, ast.Name) and base.id in names:
                return True
        if isinstance(node, ast.Call):
            operands = [node.func, *node.args, *(k.value for k in node.keywords)]
            if any(isinstance(n, ast.Name) and n.id in names for o in operands for n in walk_nodes(o)):
                return True
    return False


def element_at(expr: ast.expr, j: ast.Name, ranks: dict[str, int]) -> ast.expr | None:
    """Element ``j`` of the 1-D elementwise expression ``expr`` (``a[:n] * x[c[:n]]`` -> ``a[j] * x[c[j]]``), or
    ``None`` when ``expr`` is anything else: a name of unknown rank, a call, a 2-D operand."""

    def element(node: ast.expr) -> tuple[ast.expr, bool] | None:
        """``(element j of node, whether it varies with j)``."""
        if isinstance(node, ast.Constant):
            return node, False
        if isinstance(node, ast.Name):
            rank = ranks.get(node.id)
            if rank == 0:
                return node, False
            return (ast.Subscript(node, copy.deepcopy(j), ast.Load()), True) if rank == 1 else None
        if isinstance(node, ast.UnaryOp):
            inner = element(node.operand)
            return None if inner is None else (ast.UnaryOp(node.op, inner[0]), inner[1])
        if isinstance(node, ast.BinOp):
            left, right = element(node.left), element(node.right)
            if left is None or right is None:
                return None
            return ast.BinOp(left[0], node.op, right[0]), left[1] or right[1]
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and ranks.get(node.value.id) == 1:
            if isinstance(node.slice, ast.Slice):
                if node.slice.step is not None:
                    return None
                at = (
                    copy.deepcopy(j)
                    if node.slice.lower is None
                    else ast.BinOp(node.slice.lower, ast.Add(), copy.deepcopy(j))
                )
                return ast.Subscript(node.value, at, ast.Load()), True
            index = element(node.slice)  # a gather through an index array
            if index is None or not index[1]:
                return None
            return ast.Subscript(node.value, index[0], ast.Load()), True
        return None

    found = element(expr)
    return found[0] if found is not None and found[1] else None


def statement_lists(node: ast.AST) -> list[list[ast.stmt]]:
    """Every non-empty statement list in ``node``: bodies, else branches and finally blocks."""
    lists: list[list[ast.stmt]] = []
    for holder in walk_nodes(node):
        if isinstance(holder, (ast.FunctionDef, ast.For, ast.While, ast.If, ast.With, ast.Try, ast.ExceptHandler)):
            lists.append(holder.body)
        if isinstance(holder, (ast.For, ast.While, ast.If, ast.Try)):
            lists.append(holder.orelse)
        if isinstance(holder, ast.Try):
            lists.append(holder.finalbody)
    return [stmts for stmts in lists if stmts]


class RepeatCountsInline(RankedRewritePass):
    """``np.repeat(src, np.diff(p))`` -> a prefix-sum fill loop.

    numba and dace have no array-count repeat. The destination offset is the running sum of the
    counts, and the output length ``sum(counts)`` is data -- except for a first difference, where it
    telescopes to ``p[-1] - p[0]``. That is the only form claimed; any other per-element count is
    left standing.
    """

    __slots__ = ()

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        calls = [
            n
            for n in walk_nodes(node.value)
            if isinstance(n, ast.Call) and numpy_call_attr(n) == "repeat" and len(n.args) == 2 and not n.keywords
        ]
        pre: list[ast.stmt] = []
        for call in calls:
            counts = call.args[1]
            if not isinstance(counts, ast.Call) or numpy_call_attr(counts) != "diff" or len(counts.args) != 1:
                continue  # scalar count, or a sum we cannot derive -- not ours
            ptr = ast.unparse(counts.args[0])
            name = f"__rp{self._ctr}"
            self._ctr += 1
            src, i, r, pos = f"{name}_src", f"{name}_i", f"{name}_r", f"{name}_pos"
            lines = [
                f"{src} = {ast.unparse(call.args[0])}",
                f"{name} = np.zeros({ptr}[-1] - {ptr}[0], dtype={src}.dtype)",
                f"{pos} = 0",
                f"for {i} in range({src}.shape[0]):",
                f"    for {r} in range({ptr}[{i} + 1] - {ptr}[{i}]):",
                f"        {name}[{pos}] = {src}[{i}]",
                f"        {pos} = {pos} + 1",
            ]
            pre.extend(ast.parse("\n".join(lines)).body)
            replace_call_with_name(node, call, name)
        if not pre:
            return node
        self.changed = True
        out = [*pre, node]
        for stmt in out:
            ast.copy_location(stmt, node)
            ast.fix_missing_locations(stmt)
        return out


class BincountInline(RankedRewritePass):
    """``np.bincount(idx, weights=w, minlength=M)`` -> zero M slots, then a scatter-add loop.

    numba has no bincount and dace routes it to a Python callback. Duplicate indices accumulate,
    hence ``+=`` and not a store.

    ``minlength`` is required: numpy sizes the result ``max(minlength, idx.max() + 1)``, and the second
    term is data. Sizing it M agrees with numpy whenever every index is below M, which a caller storing
    the result into an M-sized buffer already requires.
    """

    __slots__ = ()

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        calls = [
            n for n in walk_nodes(node.value) if isinstance(n, ast.Call) and numpy_call_attr(n) == "bincount" and n.args
        ]
        if not calls:
            return node
        pre: list[ast.stmt] = []
        for call in calls:
            kw = {k.arg: k.value for k in call.keywords}
            minlength = kw.get("minlength") or (call.args[2] if len(call.args) > 2 else None)
            weights = kw.get("weights") or (call.args[1] if len(call.args) > 1 else None)
            if minlength is None:
                return node  # data-dependent extent -- leave it standing rather than guess one
            name = f"__bc{self._ctr}"
            self._ctr += 1
            idx, it = ast.unparse(call.args[0]), f"{name}_i"
            dtype = f"{ast.unparse(weights)}.dtype" if weights is not None else "np.int64"
            rhs = f"{ast.unparse(weights)}[{it}]" if weights is not None else "1"
            lines = [
                f"{name} = np.zeros({ast.unparse(minlength)}, dtype={dtype})",
                f"for {it} in range({idx}.shape[0]):",
                f"    {name}[{idx}[{it}]] += {rhs}",
            ]
            pre.extend(ast.parse("\n".join(lines)).body)
            call.func = name_("__bincount_result__")
            replace_call_with_name(node, call, name)
        self.changed = True
        out = [*pre, node]
        for stmt in out:
            ast.copy_location(stmt, node)
            ast.fix_missing_locations(stmt)
        return out


class AddAtInline(RankedRewritePass):
    """``np.add.at(A, idx, vals)`` (also subtract/multiply) -> an explicit scatter loop. numba has
    no ufunc.at; a sequential ``+=`` loop keeps its defining property -- duplicate indices accumulate
    (unlike ``A[idx] += vals``). ``idx`` is one index array or a tuple of index arrays and scalar
    indices; the first index array drives the loop, scalars ride each iteration."""

    __slots__ = ()

    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        call = node.value
        if not isinstance(call, ast.Call):
            return node
        op = ufunc_method_op(call, "at")
        first, second = (call.args[0], call.args[1]) if len(call.args) >= 2 else (None, None)
        if op not in AT_OPS or not isinstance(first, ast.Name) or second is None:
            return node
        A = first.id
        elts = second.elts if isinstance(second, ast.Tuple) else [second]
        vals = call.args[2] if len(call.args) > 2 else None
        driver_rank = next((r for e in elts if (r := expr_rank(e, self.ranks)) and r >= 1), None)
        if driver_rank is None:
            return node
        p = f"__sc{self._ctr}"
        self._ctr += 1
        iters = [f"{p}_i{k}" for k in range(driver_rank)]
        it = ", ".join(iters)
        # ``np.ascontiguousarray`` materialises each hoisted index / value array: pythran cannot
        # tuple-index a lazy numpy_expr. A no-op on an already-contiguous array.
        pre, idx_exprs, first_arr = [], [], None
        for j, e in enumerate(elts):
            if (expr_rank(e, self.ranks) or 0) >= 1:
                t = f"{p}_x{j}"
                pre.append(f"{t} = np.ascontiguousarray({ast.unparse(e)})")
                idx_exprs.append(f"{t}[{it}]")
                first_arr = first_arr or t
            else:
                idx_exprs.append(ast.unparse(e))
        trail_iters: list[str] = []
        tv = ""
        if vals is None:
            rhs = "1"
        else:
            tv = f"{p}_v"
            vr = expr_rank(vals, self.ranks)
            if vr == 0:
                # A scalar stays a scalar: ``np.ascontiguousarray(1)`` is a 0-d array, and pythran has
                # no ``double += 0-d array``.
                tv = ast.unparse(vals)
            else:
                pre.append(f"{tv} = np.ascontiguousarray({ast.unparse(vals)})")
            # ``A[idx]`` keeps the axes the index tuple does not address, so vals may carry trailing
            # block axes. Those get their own loops rather than a subarray ``+=``: scalar accumulation
            # is what every backend supports, and it keeps the unbuffered duplicate-index order.
            trailing = vr - driver_rank if vr else 0
            a_rank = self.ranks.get(A)
            if vr and trailing and (trailing < 0 or a_rank is None or a_rank - len(elts) != trailing):
                # Anything else needs numpy broadcast alignment, which is not modelled.
                raise DesugarError(
                    f"np.add.at values ndim {vr} != index ndim {driver_rank} "
                    f"plus the {'unknown' if a_rank is None else a_rank - len(elts)} "
                    "axes the index leaves untouched (only scalar, matching-shape or "
                    "block-shaped values are lowered)"
                )
            trail_iters = [f"{p}_t{k}" for k in range(trailing)]
            idx_exprs.extend(trail_iters)
            rhs = f"{tv}[{', '.join(iters + trail_iters)}]" if vr and vr >= 1 else tv
        lines, deepen = list(pre), ""
        for k in range(driver_rank):
            lines.append(f"{deepen}for {iters[k]} in range({first_arr}.shape[{k}]):")
            deepen += "    "
        for k, t in enumerate(trail_iters):
            lines.append(f"{deepen}for {t} in range({tv}.shape[{driver_rank + k}]):")
            deepen += "    "
        lines.append(f"{deepen}{A}[{', '.join(idx_exprs)}] {AT_OPS[op]} {rhs}")
        self.changed = True
        return [ast.copy_location(s, node) for s in ast.parse("\n".join(lines)).body]


class SearchsortedMaterialize(RewritePass):
    """``np.searchsorted(<expr>, v)`` -> ``np.searchsorted(np.ascontiguousarray(<expr>), v)``.

    pythran keeps an array expression as a lazy ``numpy_expr`` whose iterator is forward-only, and
    searchsorted walks it backwards (g++: no ``operator--`` on a numpy_expr_iterator). Materialising
    is a no-op for an already-contiguous array.
    """

    __slots__ = ()

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        calls = [
            n
            for n in walk_nodes(node.value)
            if isinstance(n, ast.Call) and numpy_call_attr(n) == "searchsorted" and n.args
        ]
        if not calls:
            return node
        pre: list[ast.stmt] = []
        for call in calls:
            # Even a name is materialised: pythran binds a name to the lazy expression itself.
            tmp = f"__ss{self._ctr}"
            self._ctr += 1
            pre.append(
                ast.Assign(
                    targets=[store_(tmp)],
                    value=ast.Call(
                        func=ast.Attribute(value=name_("np"), attr="ascontiguousarray", ctx=ast.Load()),
                        args=[call.args[0]],
                        keywords=[],
                    ),
                )
            )
            call.args[0] = name_(tmp)
        self.changed = True
        out = [*pre, node]
        for stmt in out:
            ast.copy_location(stmt, node)
            ast.fix_missing_locations(stmt)
        return out


def hoist_histogram(node: ast.expr, hoist: ValueHoist) -> ast.expr | None:
    """``np.histogram(a, bins[, lo, hi][, weights=w])[0]`` -> the temp its binning loop fills.

    A min/max scan for the default range, the bin edges, then per-element binning
    ``b = int((a-lo)*bins/(hi-lo))`` clamped to ``[0, bins-1]`` and walked one step against the edges
    (numpy's own correction), accumulating ``1`` (or ``w[i]``). numba has no np.histogram."""
    if not (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == 0):
        return None
    call = node.value
    if not (isinstance(call, ast.Call) and numpy_call_attr(call) == "histogram" and len(call.args) >= 2):
        return None
    a, bins = ast.unparse(call.args[0]), ast.unparse(call.args[1])
    kw = {k.arg: k.value for k in call.keywords}
    lo: ast.expr | None = None
    hi: ast.expr | None = None
    if len(call.args) >= 4:
        lo, hi = call.args[2], call.args[3]
    rng = kw.get("range")
    if isinstance(rng, ast.Tuple) and len(rng.elts) == 2:
        lo, hi = rng.elts
    weights = kw.get("weights")
    p = f"__hist{hoist.ctr}"
    hoist.ctr += 1
    lines: list[str] = []
    if lo is None or hi is None:
        lo_s, hi_s = f"{p}_lo", f"{p}_hi"
        lines += [
            f"{lo_s} = {a}[0]",
            f"{hi_s} = {a}[0]",
            f"for {p}_s in range({a}.shape[0]):",
            f"    if {a}[{p}_s] < {lo_s}: {lo_s} = {a}[{p}_s]",
            f"    if {a}[{p}_s] > {hi_s}: {hi_s} = {a}[{p}_s]",
        ]
    else:
        lo_s, hi_s = f"({ast.unparse(lo)})", f"({ast.unparse(hi)})"
    temp = f"{p}_o"
    # int64 counts when unweighted, the weights' own dtype when weighted. A non-Name weights
    # expression is hoisted so its dtype is read off a name, not re-evaluated.
    if weights is None:
        wdtype, add = "np.int64", "1"
    else:
        wname = weights.id if isinstance(weights, ast.Name) else f"{p}_w"
        if not isinstance(weights, ast.Name):
            lines.append(f"{wname} = {ast.unparse(weights)}")
        wdtype, add = f"{wname}.dtype", f"{wname}[{p}_i]"
    # numpy's bin is defined by its edge array, not the closed form: it truncates the same
    # index, then walks it one step against linspace's edges, and the two round apart. The
    # edges are rebuilt with linspace's arithmetic in the sample dtype, one rounding per
    # statement, since half an ulp of edge already misbins samples. Only edges 0..bins-1 are
    # read (the walk up stops at bins-1), hence ``bins`` entries and no top edge.
    edges = f"{p}_e"
    lines += [
        f"{edges} = np.zeros({bins}, {a}.dtype)",
        f"{edges}[0] = ({hi_s} - {lo_s}) / {bins}",
        f"{p}_st = {edges}[0]",
        f"for {p}_j in range({bins}):",
        f"    {edges}[{p}_j] = {p}_j * {p}_st",
        f"    {edges}[{p}_j] = {edges}[{p}_j] + {lo_s}",
    ]
    # numpy drops samples outside [lo, hi] (only the last bin is closed); the clamp alone would
    # fold them into the end bins. The guard is a no-op for the auto range.
    lines += [
        f"{temp} = np.zeros({bins}, {wdtype})",
        f"for {p}_i in range({a}.shape[0]):",
        f"    if {lo_s} <= {a}[{p}_i] and {a}[{p}_i] <= {hi_s}:",
        f"        {p}_b = int(({a}[{p}_i] - {lo_s}) * {bins} / ({hi_s} - {lo_s}))",
        f"        if {p}_b < 0: {p}_b = 0",
        f"        if {p}_b > {bins} - 1: {p}_b = {bins} - 1",
        f"        if {a}[{p}_i] < {edges}[{p}_b]: {p}_b = {p}_b - 1",
        # Nested, not ``and``: the DaCe frontend evaluates both operands of a condition, so ``b < bins - 1
        # and a[i] >= e[b + 1]`` read e[bins] past the edges for a sample in the last bin (azimint_hist, ASan).
        f"        if {p}_b < {bins} - 1:",
        f"            if {a}[{p}_i] >= {edges}[{p}_b + 1]: {p}_b = {p}_b + 1",
        f"        {temp}[{p}_b] += {add}",
    ]
    hoist.queue(lines)
    return name_(temp)


HISTOGRAM_HOIST = HoistForm(frozenset({"histogram"}), (), hoist_histogram)


def hoist_repeat_axis(node: ast.expr, hoist: ValueHoist) -> ast.expr | None:
    """``np.repeat(x, m, axis=k)`` (literal axis, scalar count) -> the temp its gather loop ``out[..., j, ...] =
    x[..., j // m, ...]`` fills. numba rejects ``axis=`` on np.repeat."""
    if not isinstance(node, ast.Call) or numpy_call_attr(node) != "repeat" or len(node.args) < 2:
        return None
    kw = {k.arg: k.value for k in node.keywords}
    ax = kw.get("axis") or (node.args[2] if len(node.args) > 2 else None)
    axis = None if ax is None else const_int(ax)
    if axis is None:
        return None  # no axis or a non-literal one: leave verbatim
    x, m = node.args[0], node.args[1]
    rank = expr_rank(x, hoist.tables.ranks)
    if rank is None or not -rank <= axis < rank:
        # An axis outside the rank means the rank estimate is wrong; wrapping it would repeat
        # along a different axis.
        return None
    k = axis % rank
    p = f"__rp{hoist.ctr}"
    hoist.ctr += 1
    pre = []
    if isinstance(x, ast.Name):
        xid = x.id
    else:
        xid = f"{p}_x"
        pre.append(f"{xid} = {ast.unparse(x)}")
    ms = f"({ast.unparse(m)})"
    dims = [(f"{xid}.shape[{d}] * {ms}" if d == k else f"{xid}.shape[{d}]") for d in range(rank)]
    iters = [f"{p}_i{d}" for d in range(rank)]
    out = f"{p}_o"
    lines = [*pre, f"{out} = np.empty(({', '.join(dims)},), {xid}.dtype)"]
    deep = ""
    for d in range(rank):
        lines.append(f"{deep}for {iters[d]} in range({dims[d]}):")
        deep += "    "
    src_idx = ", ".join((f"{iters[k]} // {ms}" if d == k else iters[d]) for d in range(rank))
    lines.append(f"{deep}{out}[{', '.join(iters)}] = {xid}[{src_idx}]")
    hoist.queue(lines)
    return name_(out)


REPEAT_AXIS_HOIST = HoistForm(frozenset({"repeat"}), (), hoist_repeat_axis)
