"""numba-only rewrites: outer-broadcast peel, Fortran-order reshape, dtype fixups, slice objects, ``ndim``, None
placeholders, axis scans."""

import ast
import copy
from collections.abc import Sequence
from typing import TypeGuard

from hpcagent_bench.translators.numpyto_common.ast_build import (
    const_int,
    expr_of,
    name_,
    range_for,
    set_field,
    store_,
    walk_nodes,
)
from hpcagent_bench.translators.numpyto_common.emit_helpers.numpy_names import is_numpy_module, numpy_call_attr
from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import (
    RankedRewritePass,
    RewritePass,
    name_store_counts,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.hoist import BINDING_EXPRESSIONS, HoistForm, ValueHoist
from hpcagent_bench.translators.numpyto_common.numpy_desugar.kinds import dtype_kind
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import expr_rank
from hpcagent_bench.translators.numpyto_common.subscripts import is_full_slice, is_newaxis

__all__ = [
    "ARITHMETIC_NODES",
    "BINARY_UFUNCS",
    "BroadcastAugAssignSpellOut",
    "CUMULATIVE_AXIS_HOIST",
    "ONE",
    "NdimFold",
    "NonePlaceholderDrop",
    "NumbaDtypeFixups",
    "OuterBroadcastPeel",
    "ReshapeFortranOrderInline",
    "RollInline",
    "SliceObjectInline",
    "astype_receiver",
    "bcast_tokens",
    "cast_like",
    "one_slice_index",
    "scalar_index",
    "slice_extent_token",
    "token_reads",
    "without_leading_newaxis",
]


#: Extent token for an axis pinned to one element (a newaxis or a 1-long slice).
ONE = "1"

#: Two-operand numpy ufuncs that broadcast like an arithmetic operator.
BINARY_UFUNCS = frozenset(
    {"maximum", "minimum", "fmax", "fmin", "add", "subtract", "multiply", "divide", "true_divide"}
)

#: Nodes a slice bound may be built of to be re-evaluated per peeled row: names, constants, arithmetic.
ARITHMETIC_NODES = (ast.Name, ast.Constant, ast.BinOp, ast.UnaryOp, ast.operator, ast.unaryop, ast.expr_context)


def bcast_tokens(a: list[str], b: list[str]) -> list[str]:
    """numpy's right-aligned broadcast over two extent-token vectors.

    Two differently spelled non-``1`` tokens name the same extent (the source would not run
    otherwise), so either one answers."""
    n = max(len(a), len(b))
    a = [ONE] * (n - len(a)) + list(a)
    b = [ONE] * (n - len(b)) + list(b)
    return [y if x == ONE else x for x, y in zip(a, b, strict=False)]


def token_reads(token: str, names: Sequence[str]) -> bool:
    """Whether the extent token ``token`` reads a name of ``names``."""
    return any(isinstance(n, ast.Name) and n.id in names for n in walk_nodes(ast.parse(token, mode="eval")))


def slice_extent_token(e: ast.Slice, base: str | None) -> str | None:
    """A ``Slice`` entry's extent as a source token, or None when it cannot be read off the text."""
    if e.step is not None:
        return None
    if e.lower is None and e.upper is None:
        return base
    lo = 0 if e.lower is None else const_int(e.lower)
    hi = None if e.upper is None else const_int(e.upper)
    if lo is not None and hi is not None:
        return str(hi - lo) if hi >= lo else None
    if lo == 0 and e.upper is not None:
        return ast.unparse(e.upper)
    return None


def one_slice_index(e: ast.Slice) -> ast.expr | None:
    """A constant 1-long slice's single index (``a[..., 0:1, ...]`` -> ``0``). numba's parfor
    analysis equates a length-1 axis with a length-N one instead of broadcasting it, so a leading
    singleton must disappear, not shrink."""
    if e.step is not None or slice_extent_token(e, None) != ONE:
        return None
    return copy.deepcopy(e.lower) if e.lower is not None else ast.Constant(value=0)


def without_leading_newaxis(e: ast.expr) -> ast.expr | None:
    """``x[None, :]`` -> ``x``, or None without a leading newaxis. Broadcasting prepends the
    singleton back, so both spell one value."""
    if not isinstance(e, ast.Subscript):
        return None
    if is_newaxis(e.slice):
        return e.value
    if not isinstance(e.slice, ast.Tuple):
        return None
    entries = list(e.slice.elts)
    n = 0
    while n < len(entries) and is_newaxis(entries[n]):
        n += 1
    if n == 0:
        return None
    rest = entries[n:]
    while rest and is_full_slice(rest[-1]):
        rest.pop()
    if not rest:
        return e.value
    sl = rest[0] if len(rest) == 1 else ast.Tuple(elts=rest, ctx=ast.Load())
    return ast.Subscript(value=e.value, slice=sl, ctx=ast.Load())


def scalar_index(e: ast.expr, ranks: dict[str, int]) -> bool:
    """A subscript entry that consumes its base axis and contributes none. An unknown-rank Name
    reads as a scalar, as in :func:`expr_rank`."""
    if isinstance(e, ast.Constant):
        return isinstance(e.value, int) and not isinstance(e.value, bool)
    return isinstance(e, ast.Name) and ranks.get(e.id, 0) == 0


def astype_receiver(expr: ast.expr) -> ast.expr | None:
    """``x`` of ``x.astype(dtype)``: an elementwise cast that keeps ``x``'s shape; None for anything else."""
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr == "astype"
        and len(expr.args) == 1
        and not expr.keywords
        and not is_numpy_module(expr.func.value)
    ):
        return expr.func.value
    return None


class OuterBroadcastPeel(RankedRewritePass):
    """Peel an OUTER-PRODUCT broadcast's outermost axis into an explicit loop (numba only).

    Under ``parallel=True`` (the build the benchmark measures) numba's parfor analysis asserts
    ``Sizes of $a, $b do not match`` when two operands put non-1 extents in different axes, as in
    ``a[:, None] + b[None, :]``. Each peeled row is an ordinary broadcast the analysis accepts and
    still a parallelisable array expression; reshapes and ``np.add.outer`` hit the same assert. A row
    still an outer product is peeled again (:meth:`peel_rows`); a view of one goes through a temp
    (:meth:`hoist_view_base`).

    Extents are ``<name>.shape[k]`` tokens from the rank table, since manifest shape symbols are
    not bound in the kernel. An operand whose axes cannot be placed declines the whole statement."""

    __slots__ = ()

    def operands_(self, node: ast.AST) -> list[ast.expr] | None:
        """The operands an elementwise node broadcasts together, or None when it is not one."""
        if isinstance(node, ast.BinOp):
            return None if isinstance(node.op, ast.MatMult) else [node.left, node.right]
        if isinstance(node, ast.Compare):
            return [node.left, *node.comparators]
        if isinstance(node, ast.BoolOp):
            return list(node.values)
        if (
            isinstance(node, ast.Call)
            and numpy_call_attr(node) == "where"
            and len(node.args) == 3
            and not node.keywords
        ):
            return list(node.args)
        if (
            isinstance(node, ast.Call)
            and numpy_call_attr(node) in BINARY_UFUNCS
            and len(node.args) == 2
            and not node.keywords
            and is_numpy_module(node.func.value if isinstance(node.func, ast.Attribute) else node.func)
        ):
            return list(node.args)
        return None

    def combine(self, nodes: Sequence[ast.expr]) -> list[str] | None:
        out: list[str] = []
        for n in nodes:
            ext = self.extents_(n)
            if ext is None:
                return None
            out = bcast_tokens(out, ext)
        return out

    def extents_(self, expr: ast.expr) -> list[str] | None:
        """Per-axis extent tokens of an expression, or None when this pass cannot place its axes."""
        if isinstance(expr, ast.Constant):
            return [] if isinstance(expr.value, (bool, int, float, complex)) else None
        if isinstance(expr, ast.Name):
            return [f"{expr.id}.shape[{k}]" for k in range(self.ranks.get(expr.id, 0))]
        if isinstance(expr, ast.UnaryOp):
            return self.extents_(expr.operand)
        ops = self.operands_(expr)
        if ops is not None:
            return self.combine(ops)
        if isinstance(expr, ast.Subscript):
            axes = self.axes_(expr)
            return None if axes is None else [axis[2] for axis in axes]
        if numpy_call_attr(expr) == "arange":
            return [f"({ast.unparse(expr)}).shape[0]"]  # always 1-D, so it only ever broadcasts
        receiver = astype_receiver(expr)
        if receiver is not None:
            return self.extents_(receiver)
        return None

    def axes_(self, sub: ast.Subscript) -> list[tuple[int | None, str, str]] | None:
        """``(entry index or None, kind, extent token)`` per OUTPUT axis of a subscript."""
        if not isinstance(sub.value, (ast.Name, ast.Subscript)):
            return None  # a computed base would need peeling through
        base = self.extents_(sub.value)
        if base is None:
            return None
        entries = list(sub.slice.elts) if isinstance(sub.slice, ast.Tuple) else [sub.slice]
        axes: list[tuple[int | None, str, str]] = []
        bi = 0
        for i, e in enumerate(entries):
            if is_newaxis(e):
                axes.append((i, "new", ONE))
            elif isinstance(e, ast.Slice):
                if bi >= len(base):
                    return None
                ext = slice_extent_token(e, base[bi])
                if ext is None and e.step is None:  # runtime bounds: the view's own extent, clipping included
                    ext = f"({ast.unparse(sub)}).shape[{len(axes)}]"
                if ext is None:
                    return None
                axes.append((i, "slice", ext))
                bi += 1
            elif scalar_index(e, self.ranks):
                bi += 1
            else:
                return None
        if bi > len(base):
            return None
        axes.extend((None, "tail", base[k]) for k in range(bi, len(base)))
        return axes

    def outer_product(self, value: ast.expr, rank: int) -> bool:
        """True when some elementwise node of this rank has a shape numba's analysis asserts on and
        peeling axis 0 dissolves: non-1 extents in different axes with one on axis 0 (an outer
        product), or two full-rank operands sharing a non-1 axis 0 that split on a later axis
        (``g[:, None] * X``)."""
        for node in walk_nodes(value):
            ops = self.operands_(node)
            if ops is None:
                continue
            sized = [self.extents_(o) for o in ops]
            exts = [e for e in sized if e is not None and len(e) <= rank]
            if len(exts) != len(sized):
                continue
            if max((len(e) for e in exts), default=0) != rank:
                continue
            padded = [[ONE] * (rank - len(e)) + e for e in exts]
            for ia, a in enumerate(padded):
                for ib, b in enumerate(padded):
                    if a[0] == ONE or not any(a[j] == ONE and b[j] != ONE for j in range(1, rank)):
                        continue
                    if b[0] == ONE or (len(exts[ia]) == rank and len(exts[ib]) == rank):
                        return True
        return False

    def peel(self, expr: ast.expr, idx: str, rank: int) -> ast.expr | None:
        ext = self.extents_(expr)
        if ext is None or len(ext) > rank:
            return None
        if len(ext) < rank:
            return expr  # lower rank: broadcasts along the peeled axis
        if isinstance(expr, ast.UnaryOp):
            operand = self.peel(expr.operand, idx, rank)
            return None if operand is None else ast.UnaryOp(op=expr.op, operand=operand)
        ops = self.operands_(expr)
        if ops is not None:
            candidates = [self.peel(o, idx, rank) for o in ops]
            peeled = [p for p in candidates if p is not None]
            return None if len(peeled) != len(candidates) else self.rebuild(expr, peeled)
        if isinstance(expr, ast.Name):
            return ast.Subscript(value=expr, slice=expr_of(idx), ctx=ast.Load())
        if isinstance(expr, ast.Subscript):
            return self.peel_subscript(expr, idx)
        receiver = astype_receiver(expr)
        if receiver is not None and isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            inner = self.peel(receiver, idx, rank)
            if inner is None:
                return None
            method = ast.Attribute(value=inner, attr="astype", ctx=ast.Load())
            return ast.Call(func=method, args=list(expr.args), keywords=list(expr.keywords))
        return None

    def leaf_extents(self, expr: ast.expr) -> list[list[str]] | None:
        """The extent tokens of every operand an elementwise tree broadcasts together, or None."""
        if isinstance(expr, ast.UnaryOp):
            return self.leaf_extents(expr.operand)
        ops = self.operands_(expr)
        if ops is None:
            ext = self.extents_(expr)
            return None if ext is None else [ext]
        out: list[list[str]] = []
        for o in ops:
            sub = self.leaf_extents(o)
            if sub is None:
                return None
            out.extend(sub)
        return out

    def axis_tokens(self, value: ast.expr, rank: int) -> set[str] | None:
        """The non-1 extent tokens the operands of the rank-``rank`` ``value`` give its axis 0, or None.

        :func:`bcast_tokens` takes two spellings as one extent, which a size-1 axis breaks: ``p[i, :, None] * n[i]``
        with ``p`` of shape ``(M, 1)`` would loop ``range(p.shape[1])`` and fill one row of the result."""
        leaves = self.leaf_extents(value)
        if leaves is None:
            return None
        return {([ONE] * (rank - len(e)) + e)[0] for e in leaves if len(e) <= rank} - {ONE}

    def peel_rows(
        self, value: ast.expr, ivar: str, rank: int
    ) -> tuple[list[str], ast.expr, ast.expr, list[list[str]]] | None:
        """``(loop variables, row, probe, checks)`` peeling one axis per loop until the row is no outer product
        (warpx_field_gather's ``sx[i] * sy[:, None] * sz`` still is), or None to decline. ``probe`` is the row
        at index 0 of every peeled axis: only its dtype is read, but it must compile too.

        An inner axis two operands size (force_lj's ``fpair[i, :, None] * dpos[i]``) is peeled only when its
        tokens read no loop variable, and ``checks`` lists them, axis 0 included: the caller raises unless they
        agree, so a size-1 axis numpy would broadcast is an error, never a wrong answer (:meth:`axis_tokens`)."""
        row = self.peel(copy.deepcopy(value), ivar, rank)
        probe = self.peel(copy.deepcopy(value), "0", rank)
        if row is None or probe is None:
            return None
        ivars = [ivar]
        checks: list[list[str]] = []
        while (depth := rank - len(ivars)) >= 2:
            ext = self.extents_(row)
            if ext is None or len(ext) != depth or ext[0] == ONE or not self.outer_product(row, depth):
                break
            tokens = self.axis_tokens(row, depth)
            if not tokens or (len(tokens) > 1 and any(token_reads(t, ivars) for t in tokens)):
                break
            inner = f"{ivar}{len(ivars)}"
            next_row = self.peel(copy.deepcopy(row), inner, depth)
            next_probe = self.peel(copy.deepcopy(probe), "0", depth)
            if next_row is None or next_probe is None:
                break
            if len(tokens) > 1:
                checks.append(sorted(tokens))
            row, probe = next_row, next_probe
            ivars.append(inner)
        if len(ivars) > 1 and len(outer := self.axis_tokens(value, rank) or set()) > 1:
            checks.insert(0, sorted(outer))
        return ivars, row, probe, checks

    def rebuild(self, expr: ast.AST, peeled: list[ast.expr]) -> ast.expr:
        if isinstance(expr, ast.BinOp):
            return ast.BinOp(left=peeled[0], op=expr.op, right=peeled[1])
        if isinstance(expr, ast.Compare):
            return ast.Compare(left=peeled[0], ops=expr.ops, comparators=peeled[1:])
        if isinstance(expr, ast.BoolOp):
            return ast.BoolOp(op=expr.op, values=peeled)
        if isinstance(expr, ast.Call):
            return ast.Call(func=expr.func, args=peeled, keywords=[])
        raise TypeError(f"cannot rebuild {type(expr).__name__}")

    def peel_subscript(self, sub: ast.Subscript, idx: str) -> ast.expr | None:
        axes = self.axes_(sub)
        if not axes:
            return None
        entries = list(sub.slice.elts) if isinstance(sub.slice, ast.Tuple) else [sub.slice]
        pos, kind, ext0 = axes[0]
        if pos is None:
            if ext0 == ONE:
                return None
            entries = [*entries, expr_of(idx)]  # axis 0 is an unindexed trailing base axis
        elif kind == "new":
            entries = entries[:pos] + entries[pos + 1 :]
        else:
            e = entries[pos]
            if not isinstance(e, ast.Slice):
                return None
            if ext0 == ONE:
                repl = e.lower if e.lower is not None else ast.Constant(value=0)
            elif e.lower is None or const_int(e.lower) == 0:
                repl = expr_of(idx)
            elif all(isinstance(n, ARITHMETIC_NODES) for n in walk_nodes(e.lower)):
                repl = ast.BinOp(left=copy.deepcopy(e.lower), op=ast.Add(), right=expr_of(idx))
            else:
                return None
            entries = [*entries[:pos], repl, *entries[pos + 1 :]]
        return self.tidy(sub.value, entries)

    def tidy(self, base: ast.expr, entries: Sequence[ast.expr]) -> ast.expr:
        """Drop entries the peel made redundant: leading singleton out-axes (see
        :func:`one_slice_index`) and trailing full slices (``a[i, :]`` is ``a[i]``)."""
        kept: list[ast.expr] = []
        for i, e in enumerate(entries):
            if is_newaxis(e):
                continue
            if isinstance(e, ast.Slice):
                lone = one_slice_index(e)
                if lone is None:
                    kept.extend(entries[i:])
                    break
                kept.append(lone)
                continue
            kept.append(e)
        while kept and is_full_slice(kept[-1]):
            kept.pop()
        if not kept:
            return base
        sl = kept[0] if len(kept) == 1 else ast.Tuple(elts=kept, ctx=ast.Load())
        return ast.Subscript(value=base, slice=sl, ctx=ast.Load())

    def store_base(self, target: ast.expr, rank: int) -> str | None:
        """``T[:]`` / ``T[:, :]`` over a rank-``rank`` ``T`` -> ``T``; anything else declines."""
        if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)):
            return None
        entries = list(target.slice.elts) if isinstance(target.slice, ast.Tuple) else [target.slice]
        if not all(is_full_slice(e) for e in entries) or len(entries) > rank:
            return None
        return target.value.id if self.ranks.get(target.value.id) == rank else None

    def drop_newaxes(self, value: ast.expr) -> None:
        """Drop every leading newaxis the enclosing broadcast makes redundant.

        numba's analysis equates a (1, n) operand with an (m, n) one instead of broadcasting it;
        ``x[None, :]`` -> ``x`` is exact only while the node's rank does not change."""
        for node in walk_nodes(value):
            ops = self.operands_(node)
            if ops is None:
                continue
            rank = expr_rank(node, self.ranks)
            if rank is None:
                continue
            for i, operand in enumerate(ops):
                bare = without_leading_newaxis(operand)
                if bare is None and isinstance(operand, ast.UnaryOp):
                    # ``-ew[lo:hi][None, :]``: a sign around the newaxis broadcasts the same way.
                    inner = without_leading_newaxis(operand.operand)
                    bare = None if inner is None else ast.UnaryOp(op=operand.op, operand=inner)
                if bare is None:
                    continue
                trial = list(ops)
                trial[i] = bare
                if expr_rank(self.rebuild(node, trial), self.ranks) != rank:
                    continue
                ops[i] = bare
                self.replace(node, i, bare)
                self.changed = True

    def replace(self, node: ast.AST, i: int, operand: ast.expr) -> None:
        if isinstance(node, ast.BinOp):
            node.left, node.right = (operand, node.right) if i == 0 else (node.left, operand)
        elif isinstance(node, ast.Compare):
            if i == 0:
                node.left = operand
            else:
                node.comparators[i - 1] = operand
        elif isinstance(node, ast.BoolOp):
            node.values[i] = operand
        elif isinstance(node, ast.Call):
            node.args[i] = operand

    def hoist_view_base(self, node: ast.Assign) -> list[ast.stmt] | None:
        """``x = (a + b[:, None])[:, None, :]`` -> the product bound to a temp and peeled, then
        ``x = temp[:, None, :]``; None when the value is no view of an outer product (:meth:`axes_` cannot place a
        computed base's axes), or when two operands size its axis 0, which a single peel does not check."""
        value = node.value
        if not isinstance(value, ast.Subscript) or self.operands_(value.value) is None:
            return None
        ext = self.extents_(value.value)
        if ext is None or len(self.axis_tokens(value.value, len(ext)) or ()) != 1:
            return None
        temp = f"__ob{self._ctr}_v"
        self.ranks[temp] = len(ext)
        bind = ast.Assign(targets=[store_(temp)], value=value.value)
        ast.copy_location(bind, node)
        peeled = self.visit_Assign(bind)
        if not isinstance(peeled, list):
            del self.ranks[temp]
            return None
        view = ast.Assign(
            targets=node.targets, value=ast.Subscript(value=name_(temp), slice=value.slice, ctx=ast.Load())
        )
        ast.copy_location(view, node)
        ast.fix_missing_locations(view)
        return [*peeled, view]

    def hoist_matmul_operands(self, node: ast.Assign) -> list[ast.stmt] | None:
        """``c = (a @ b) / n[:, None]`` -> ``t = a @ b`` then ``c = t / n[:, None]`` peeled: a product has no
        extents this pass can place, a bound temp of known rank has. None when no product operand of the
        elementwise value has a known rank, or the rewritten statement still does not peel."""
        ops = self.operands_(node.value)
        if ops is None:
            return None
        products = [
            o for o in ops if isinstance(o, ast.BinOp) and isinstance(o.op, ast.MatMult) and expr_rank(o, self.ranks)
        ]
        if not products:
            return None
        value = copy.deepcopy(node.value)
        binds: list[ast.stmt] = []
        temps: list[str] = []
        for product in products:
            temp = f"__ob{self._ctr}_m{len(temps)}"
            rank = expr_rank(product, self.ranks)
            if rank is None:
                return None
            self.ranks[temp] = rank
            temps.append(temp)
            binds.append(ast.copy_location(ast.Assign(targets=[store_(temp)], value=product), node))
            for parent in walk_nodes(value):
                for field, child in ast.iter_fields(parent):
                    if isinstance(child, ast.AST) and ast.dump(child) == ast.dump(product):
                        set_field(parent, field, name_(temp))
        rest = ast.copy_location(ast.Assign(targets=node.targets, value=value), node)
        peeled = self.visit_Assign(rest)
        if not isinstance(peeled, list):
            for temp in temps:
                del self.ranks[temp]
            return None
        for b in binds:
            ast.fix_missing_locations(b)
        return [*binds, *peeled]

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        if len(node.targets) != 1:
            return node
        self.drop_newaxes(node.value)
        hoisted = self.hoist_view_base(node)
        if hoisted is not None:
            return hoisted
        if self.extents_(node.value) is None:
            lifted = self.hoist_matmul_operands(node)
            if lifted is not None:
                return lifted
        ext = self.extents_(node.value)
        if ext is None or len(ext) < 2 or ext[0] == ONE or not self.outer_product(node.value, len(ext)):
            return node
        rank = len(ext)
        target = node.targets[0]
        base = target.id if isinstance(target, ast.Name) else self.store_base(target, rank)
        # A target the value also reads would see rows the loop already rewrote; it, and a store that
        # is not a whole-array slice (``H[lo:hi, :]``), fills a fresh temp assigned afterwards.
        direct = base is not None and not any(isinstance(n, ast.Name) and n.id == base for n in walk_nodes(node.value))
        if not direct and not isinstance(target, (ast.Name, ast.Subscript)):
            return node
        temp, ivar = f"__ob{self._ctr}", f"__ob{self._ctr}_i"
        peeled = self.peel_rows(node.value, ivar, rank)
        if peeled is None:
            return node
        ivars, row, probe, checks = peeled
        self._ctr += 1
        self.changed = True
        out: list[ast.stmt] = [
            ast.parse(
                f"if not ({' == '.join(c)}):\n    raise ValueError('operands could not be broadcast together')"
            ).body[0]
            for c in checks
        ]
        dest = base if direct and base is not None else f"{temp}_o"
        shape = ", ".join(ext)
        if isinstance(target, ast.Name):
            # A bare Name is a binding, so allocate it. The probe row carries the promoted dtype
            # (``X + Y[:, None] * 1j`` is complex though neither operand is); only its dtype is read.
            out.append(ast.Assign(targets=[store_(temp)], value=probe))
            out.append(ast.parse(f"{dest} = np.empty(({shape},), {temp}.dtype)").body[0])
        elif not direct and isinstance(target, ast.Subscript):
            # The store casts into the target's dtype, so the temp takes it.
            out.append(ast.parse(f"{dest} = np.empty(({shape},), ({ast.unparse(target.value)}).dtype)").body[0])
        index = name_(ivar) if len(ivars) == 1 else ast.Tuple(elts=[name_(v) for v in ivars], ctx=ast.Load())
        loop: ast.stmt = ast.Assign(targets=[ast.Subscript(value=name_(dest), slice=index, ctx=ast.Store())], value=row)
        for k in reversed(range(len(ivars))):
            loop = range_for(ivars[k], [expr_of(ext[k])], [loop])
        out.append(loop)
        if not direct:
            out.append(ast.Assign(targets=[target], value=name_(dest)))
        for s in out:
            ast.copy_location(s, node)
            ast.fix_missing_locations(s)
        return out


class SliceObjectInline(ast.NodeTransformer):
    """``b = slice(lo, hi)`` read back as ``X[b, :]`` -> ``X[lo:hi, :]`` (numba only).

    :func:`expr_rank` reads a Name index as a scalar, so ``X[b, :]`` would rank 1, not 2. The
    substitution is exact when the binding is the name's only store and every name ``lo``/``hi``
    read is bound at most once. Runs before the rank table is built."""

    __slots__ = ("changed", "slices")

    def __init__(self, fn: ast.FunctionDef) -> None:
        self.changed = False
        self.slices: dict[str, ast.Slice] = {}
        if any(isinstance(n, (ast.Lambda, ast.FunctionDef)) and n is not fn for n in walk_nodes(fn)):
            return
        stores = name_store_counts(fn)
        for node in walk_nodes(fn):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "slice"
                and 1 <= len(node.value.args) <= 3
                and not node.value.keywords
            ):
                continue
            args = node.value.args
            if any(isinstance(a, ast.Starred) for a in args) or stores.get(node.targets[0].id, 0) != 1:
                continue
            read = {n.id for a in args for n in walk_nodes(a) if isinstance(n, ast.Name)}
            if any(stores.get(r, 0) > 1 for r in read):
                continue
            bounds = [None, args[0], None] if len(args) == 1 else [*args, None][:3]
            lower, upper, step = (
                None if b is None or (isinstance(b, ast.Constant) and b.value is None) else b for b in bounds
            )
            self.slices[node.targets[0].id] = ast.Slice(lower=lower, upper=upper, step=step)

    def entry(self, e: ast.expr) -> ast.expr:
        if isinstance(e, ast.Name) and isinstance(e.ctx, ast.Load) and e.id in self.slices:
            self.changed = True
            return copy.deepcopy(self.slices[e.id])
        return e

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.slice, ast.Tuple):
            node.slice.elts = [self.entry(e) for e in node.slice.elts]
        else:
            node.slice = self.entry(node.slice)
        return node


def is_none_assign(node: ast.AST) -> TypeGuard[ast.Assign]:
    """``a = b = None``."""
    return isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value is None


def none_targets(node: ast.AST) -> set[str]:
    """Every name some ``a = None`` under ``node`` binds."""
    return {t.id for n in walk_nodes(node) if is_none_assign(n) for t in n.targets if isinstance(t, ast.Name)}


def bare_reads(node: ast.AST) -> set[str]:
    """The names ``node`` reads other than as the base of a subscript or an attribute."""
    return {
        child.id
        for parent in walk_nodes(node)
        for child in ast.iter_child_nodes(parent)
        if isinstance(child, ast.Name)
        and isinstance(child.ctx, ast.Load)
        and not (isinstance(parent, (ast.Subscript, ast.Attribute)) and parent.value is child)
    }


def name_stores(node: ast.AST) -> list[ast.Name]:
    """The ``Name`` stores under ``node`` that bind in its scope: a comprehension's or lambda's are its own."""
    if isinstance(node, BINDING_EXPRESSIONS):
        return []
    own = [node] if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) else []
    return own + [n for child in ast.iter_child_nodes(node) for n in name_stores(child)]


def stored_names(node: ast.AST) -> set[str]:
    return {n.id for n in name_stores(node)}


def uncovered_reads(stmts: Sequence[ast.stmt], bound: set[str], names: set[str], unsafe: set[str]) -> None:
    """Add to ``unsafe`` every name of ``names`` some statement reads bare where it may still hold a ``None``.

    ``bound`` holds the names whose last binding on every path is not ``None``. A compound statement's own
    bindings do not count after it, and every name it may set to ``None`` is unbound after it, in a loop body
    and test, and in what follows a ``try`` body."""
    bound = set(bound)
    for stmt in stmts:
        if isinstance(stmt, (ast.For, ast.While, ast.If, ast.With, ast.Try, ast.TryStar)):
            after = bound - none_targets(stmt)
            if isinstance(stmt, ast.For):
                unsafe |= bare_reads(stmt.iter) & (names - bound)
                uncovered_reads(stmt.body, after | stored_names(stmt.target), names, unsafe)
                uncovered_reads(stmt.orelse, after, names, unsafe)
            elif isinstance(stmt, ast.While):
                unsafe |= bare_reads(stmt.test) & (names - after)
                uncovered_reads(stmt.body, after, names, unsafe)
                uncovered_reads(stmt.orelse, after, names, unsafe)
            elif isinstance(stmt, ast.If):
                unsafe |= bare_reads(stmt.test) & (names - bound)
                uncovered_reads(stmt.body, bound, names, unsafe)
                uncovered_reads(stmt.orelse, bound, names, unsafe)
            elif isinstance(stmt, ast.With):
                for item in stmt.items:
                    unsafe |= bare_reads(item.context_expr) & (names - bound)
                uncovered_reads(stmt.body, bound, names, unsafe)
            else:
                uncovered_reads(stmt.body, bound, names, unsafe)
                for block in (*(h.body for h in stmt.handlers), stmt.orelse, stmt.finalbody):
                    uncovered_reads(block, after, names, unsafe)
            bound = after
        elif isinstance(
            stmt, (ast.AsyncFor, ast.AsyncWith, ast.Match, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            unsafe |= names  # not followed: keep every placeholder
        else:
            unsafe |= bare_reads(stmt) & (names - bound)
            if is_none_assign(stmt):
                bound -= none_targets(stmt)
            else:
                bound |= stored_names(stmt)


class NonePlaceholderDrop(ast.NodeTransformer):
    """``a = b = None`` ahead of the array bindings that replace it -> dropped (numba only).

    numba types the name Optional(array) at the join and its parfor analysis asserts ``Dimension mismatch``
    (warpx_esirkepov_deposition's ``sx_new = sx_old = ... = None``); the rank table drops a name of two ranks.
    Exact when the name is bound again and read bare (``f(a)``, ``a is None``) only where no ``None`` can reach
    (:func:`uncovered_reads`): what is left reads the placeholder only as ``a[...]`` / ``a.attr``, which numpy
    raises on. A placeholder that was a whole statement leaves a ``pass``. Runs before the rank table is built."""

    __slots__ = ("changed", "names")

    def __init__(self, fn: ast.FunctionDef) -> None:
        self.changed = False
        self.names: set[str] = set()
        if any(isinstance(n, (ast.Lambda, ast.Global, ast.Nonlocal)) for n in walk_nodes(fn)):
            return
        placeholders = none_targets(fn)
        if not placeholders:
            return
        args = fn.args
        params = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg] if a}
        placeholder_stores = {id(t) for n in walk_nodes(fn) if is_none_assign(n) for t in n.targets}
        rebound = {n.id for n in name_stores(fn) if id(n) not in placeholder_stores}
        unsafe: set[str] = set()
        uncovered_reads(fn.body, set(), placeholders, unsafe)
        self.names = (placeholders & rebound) - unsafe - params

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if not is_none_assign(node):
            return node
        kept = [t for t in node.targets if not (isinstance(t, ast.Name) and t.id in self.names)]
        if len(kept) == len(node.targets):
            return node
        self.changed = True
        if not kept:
            return ast.copy_location(ast.Pass(), node)
        node.targets = kept
        return node


class ReshapeFortranOrderInline(RewritePass):
    """``x.reshape(d0, ..., dk, order="F")`` -> ``np.ascontiguousarray(x.T).reshape((dk, ..., d0)).T``
    (numba only).

    numba's ``reshape`` takes no keyword. Fortran order is C order on the axis-reversed array, so
    the transposed spelling is exact. A shape passed as one name cannot be reversed and stays verbatim."""

    __slots__ = ()

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "reshape" and node.args and len(node.keywords) == 1):
            return node
        if is_numpy_module(f.value):
            return node  # function form ``np.reshape(x, shape, order=...)``
        kw = node.keywords[0]
        if kw.arg != "order" or not (isinstance(kw.value, ast.Constant) and kw.value.value == "F"):
            return node
        if len(node.args) == 1 and isinstance(node.args[0], ast.Tuple):
            dims = list(node.args[0].elts)
        elif len(node.args) > 1 or isinstance(node.args[0], ast.Constant):
            dims = list(node.args)
        else:
            return node
        if any(isinstance(d, ast.Starred) for d in dims):
            return node
        shape = ", ".join(ast.unparse(d) for d in reversed(dims))
        self.changed = True
        return ast.copy_location(
            expr_of(f"np.ascontiguousarray(({ast.unparse(f.value)}).T).reshape(({shape},)).T"), node
        )


class BroadcastAugAssignSpellOut(RankedRewritePass):
    """``t op= v`` where ``v`` broadcasts (lower rank, or a newaxis) -> ``t[...] = t op v`` (numba only).

    numba's parfor lowering refuses an in-place broadcast; the plain store it lowers, and an outer-shaped one
    is then peeled by :class:`OuterBroadcastPeel`. A whole-array target is stored through full slices."""

    __slots__ = ()

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        target = node.target
        if not isinstance(target, (ast.Name, ast.Subscript)) or isinstance(node.op, ast.MatMult):
            return node
        rank = expr_rank(target, self.ranks)
        value_rank = expr_rank(node.value, self.ranks)
        if rank is None or rank < 2 or value_rank is None or value_rank < 1:
            return node
        newaxis = any(
            isinstance(n, ast.Subscript) and any(is_newaxis(e) for e in walk_nodes(n.slice))
            for n in walk_nodes(node.value)
        )
        if value_rank >= rank and not newaxis:
            return node
        if isinstance(target, ast.Name):
            full = ast.Tuple(elts=[ast.Slice() for _ in range(rank)], ctx=ast.Load())
            store: ast.expr = ast.Subscript(value=name_(target.id), slice=full, ctx=ast.Store())
            load: ast.expr = name_(target.id)
        else:
            store = copy.deepcopy(target)
            load = copy.deepcopy(target)
            load.ctx = ast.Load()
        value = ast.BinOp(left=load, op=node.op, right=node.value)
        self.changed = True
        return ast.fix_missing_locations(ast.copy_location(ast.Assign(targets=[store], value=value), node))


class RollInline(RankedRewritePass):
    """``np.roll(x, s, axis=k)`` with a literal axis -> the two wrapped pieces concatenated (numba only).

    numba's ``np.roll`` takes no axis. ``x`` must be a name or a sliced view of one, re-read once per piece."""

    __slots__ = ()

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if numpy_call_attr(node) != "roll" or not is_numpy_module(
            node.func.value if isinstance(node.func, ast.Attribute) else node.func
        ):
            return node
        kw = {k.arg: k.value for k in node.keywords}
        if len(node.args) < 2 or set(kw) - {"axis"} or (len(node.args) > 2 and "axis" in kw):
            return node
        axis_node = node.args[2] if len(node.args) > 2 else kw.get("axis")
        axis = const_int(axis_node)
        x, shift = node.args[0], node.args[1]
        rank = expr_rank(x, self.ranks)
        pure = isinstance(x, ast.Name) or (
            isinstance(x, ast.Subscript)
            and isinstance(x.value, ast.Name)
            and all(isinstance(n, ARITHMETIC_NODES + (ast.Slice, ast.Tuple)) for n in walk_nodes(x.slice))
        )
        if axis is None or rank is None or not pure or not 0 <= (axis % rank if rank else -1) < rank:
            return node
        axis %= rank
        text, extent = ast.unparse(x), f"({ast.unparse(x)}).shape[{axis}]"
        cut = f"{extent} - ({ast.unparse(shift)}) % {extent}"
        lead = ", ".join([":"] * axis)
        lead = f"{lead}, " if lead else ""
        tail, head = f"({text})[{lead}{cut}:]", f"({text})[{lead}:{cut}]"
        self.changed = True
        return ast.copy_location(expr_of(f"np.concatenate(({tail}, {head}), axis={axis})"), node)


class NumbaDtypeFixups(ast.NodeTransformer):
    """Two spellings numba's dtype-strict typing refuses where numpy accepts them (numba only).

    * ``dtype=bool``: numba does not accept the builtin ``bool`` as a dtype; ``np.bool_`` is the same.
    * real ``@`` complex: numba requires equal dtypes. Casting the real side to the complex name's
      ``.dtype`` is numpy's promotion. Only fires when both kinds are known and the complex side is a
      Name, so reading its dtype evaluates nothing twice."""

    __slots__ = ("changed", "kinds")

    def __init__(self, kinds: dict[str, str]) -> None:
        self.kinds = kinds
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        attr = numpy_call_attr(node)
        if attr is None:
            return node
        for kw in node.keywords:
            if kw.arg == "dtype" and isinstance(kw.value, ast.Name) and kw.value.id == "bool":
                kw.value = ast.copy_location(expr_of("np.bool_"), kw.value)
                self.changed = True
        if attr in ("zeros", "ones", "empty") and len(node.args) > 1:
            dt = node.args[1]
            if isinstance(dt, ast.Name) and dt.id == "bool":
                node.args[1] = ast.copy_location(expr_of("np.bool_"), dt)
                self.changed = True
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.op, ast.MatMult):
            return node
        left, right = dtype_kind(node.left, self.kinds), dtype_kind(node.right, self.kinds)
        if left == "float" and right == "complex" and isinstance(node.right, ast.Name):
            node.left = cast_like(node.left, node.right.id)
        elif left == "complex" and right == "float" and isinstance(node.left, ast.Name):
            node.right = cast_like(node.right, node.left.id)
        else:
            return node
        self.changed = True
        return node


def cast_like(operand: ast.expr, like: str) -> ast.expr:
    """``operand.astype(like.dtype)``."""
    method = ast.Attribute(value=operand, attr="astype", ctx=ast.Load())
    return ast.Call(func=method, args=[expr_of(f"{like}.dtype")], keywords=[])


class NdimFold(ast.NodeTransformer):
    """``x.ndim`` for a parameter bound once with an agreed rank -> that rank, then fold ``K == K'`` and
    ``a if <bool constant> else b`` (numba only). ``y = x if x.ndim == 2 else x[:, None]`` has two
    branch ranks the rank table cannot join; the fold keeps the branch that runs."""

    __slots__ = ("changed", "known")

    def __init__(self, fn: ast.FunctionDef, agreed: dict[str, int]) -> None:
        stores = name_store_counts(fn)
        self.known = {name: rank for name, rank in agreed.items() if stores.get(name, 0) == 1}
        self.changed = False

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if node.attr == "ndim" and isinstance(node.value, ast.Name) and node.value.id in self.known:
            self.changed = True
            return ast.copy_location(ast.Constant(value=self.known[node.value.id]), node)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
            return node
        left, right = node.left, node.comparators[0]
        ints = [c for c in (left, right) if isinstance(c, ast.Constant) and type(c.value) is int]
        if len(ints) != 2:
            return node
        equal = ints[0].value == ints[1].value
        self.changed = True
        return ast.copy_location(ast.Constant(value=equal if isinstance(node.ops[0], ast.Eq) else not equal), node)

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
            self.changed = True
            return node.body if node.test.value else node.orelse
        return node


#: numpy scan -> the operator one step of it applies.
CUMULATIVE_OPS = {"cumsum": "+", "cumprod": "*"}


def hoist_cumulative_axis(node: ast.expr, hoist: ValueHoist) -> ast.expr | None:
    """``np.cumsum(x, axis=k)`` / ``np.cumprod``, ``k`` a constant -> a temp filled one slice of axis ``k`` at a time
    (numba only: its cumprod takes no ``axis``, and its cumsum scans serially).

    Each slice is the previous one combined with the next input slice, numpy's left-to-right order, so the result
    is bit for bit numpy's; the dtype is numba's cumsum's, numpy's but for a narrow unsigned integer (``int64``, not
    ``uint64``). A negative ``k`` indexes from the back (``x[..., i, :]``), so no rank is read and a helper called at
    several (mamba2's ``_segsum``) is right at each; it runs before :class:`NormalizeNegativeAxis` pins the axis."""
    op = numpy_call_attr(node)
    if op not in CUMULATIVE_OPS or not isinstance(node, ast.Call) or not 1 <= len(node.args) <= 2:
        return None
    if [kw.arg for kw in node.keywords] != (["axis"] if len(node.args) == 1 else []):
        return None  # no axis, or dtype= / out=
    k = const_int(node.keywords[0].value if node.keywords else node.args[1])
    if k is None:
        return None
    t = f"__cs{hoist.ctr}"
    hoist.ctr += 1
    x, i, sym = f"{t}_x", f"{t}_i", CUMULATIVE_OPS[op]

    def at(e: str) -> str:
        return f"{':, ' * k}{e}" if k >= 0 else f"..., {e}{', :' * (-k - 1)}"

    hoist.queue(
        [
            f"{x} = {ast.unparse(node.args[0])}",
            f"{t} = np.empty({x}.shape, np.cumsum(np.zeros(1, {x}.dtype)).dtype)",
            f"if {x}.shape[{k}] > 0:",
            f"    {t}[{at('0')}] = {x}[{at('0')}]",
            f"    for {i} in range(1, {x}.shape[{k}]):",
            f"        {t}[{at(i)}] = {t}[{at(f'{i} - 1')}] {sym} {x}[{at(i)}]",
        ]
    )
    return name_(t)


CUMULATIVE_AXIS_HOIST = HoistForm(frozenset(CUMULATIVE_OPS), (), hoist_cumulative_axis)
