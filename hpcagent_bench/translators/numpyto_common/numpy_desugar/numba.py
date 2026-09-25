"""numba-only rewrites: outer-broadcast peel, Fortran-order reshape, dtype fixups, slice objects, ``ndim``."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.subscripts import is_full_slice, is_newaxis
from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import (
    RankedRewritePass,
    RewritePass,
    const_int,
    expr_of,
    np_attr,
    name_store_counts,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.kinds import dtype_kind
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import expr_rank


#: Extent token for an axis pinned to one element (a newaxis or a 1-long slice).
ONE = "1"


def bcast_tokens(a: list[str], b: list[str]) -> list[str]:
    """numpy's right-aligned broadcast over two extent-token vectors.

    Two differently spelled non-``1`` tokens name the same extent (the source would not run
    otherwise), so either one answers."""
    n = max(len(a), len(b))
    a = [ONE] * (n - len(a)) + list(a)
    b = [ONE] * (n - len(b)) + list(b)
    return [y if x == ONE else x for x, y in zip(a, b)]


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


def scalar_index(e: ast.AST, ranks: dict[str, int]) -> bool:
    """A subscript entry that consumes its base axis and contributes none. An unknown-rank Name
    reads as a scalar, as in :func:`expr_rank`."""
    if isinstance(e, ast.Constant):
        return isinstance(e.value, int) and not isinstance(e.value, bool)
    return isinstance(e, ast.Name) and ranks.get(e.id, 0) == 0


class OuterBroadcastPeel(RankedRewritePass):
    """Peel an OUTER-PRODUCT broadcast's outermost axis into an explicit loop (numba only).

    Under ``parallel=True`` (the build the benchmark measures) numba's parfor analysis asserts
    ``Sizes of $a, $b do not match`` when two operands put non-1 extents in different axes, as in
    ``a[:, None] + b[None, :]``. Each peeled row is an ordinary broadcast the analysis accepts and
    still a parallelisable array expression; reshapes and ``np.add.outer`` hit the same assert.

    Extents are ``<name>.shape[k]`` tokens from the rank table, since manifest shape symbols are
    not bound in the kernel. An operand whose axes cannot be placed declines the whole statement."""

    def operands_(self, node: ast.AST) -> list[ast.expr] | None:
        """The operands an elementwise node broadcasts together, or None when it is not one."""
        if isinstance(node, ast.BinOp):
            return None if isinstance(node.op, ast.MatMult) else [node.left, node.right]
        if isinstance(node, ast.Compare):
            return [node.left, *node.comparators]
        if isinstance(node, ast.BoolOp):
            return list(node.values)
        if np_attr(node) == "where" and len(node.args) == 3 and not node.keywords:
            return list(node.args)
        return None

    def combine(self, nodes: list[ast.expr]) -> list[str] | None:
        out: list[str] = []
        for n in nodes:
            ext = self.extents_(n)
            if ext is None:
                return None
            out = bcast_tokens(out, ext)
        return out

    def extents_(self, expr: ast.AST) -> list[str] | None:
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
            return None if axes is None else [ext for unused, unused, ext in axes]
        return None

    def axes_(self, sub: ast.Subscript):
        """``(entry index or None, kind, extent token)`` per OUTPUT axis of a subscript."""
        if not isinstance(sub.value, (ast.Name, ast.Subscript)):
            return None  # a computed base would need peeling through
        base = self.extents_(sub.value)
        if base is None:
            return None
        entries = list(sub.slice.elts) if isinstance(sub.slice, ast.Tuple) else [sub.slice]
        axes, bi = [], 0
        for i, e in enumerate(entries):
            if is_newaxis(e):
                axes.append((i, "new", ONE))
            elif isinstance(e, ast.Slice):
                if bi >= len(base):
                    return None
                ext = slice_extent_token(e, base[bi])
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

    def outer_product(self, value: ast.AST, rank: int) -> bool:
        """True when some elementwise node of this rank has a shape numba's analysis asserts on and
        peeling axis 0 dissolves: non-1 extents in different axes with one on axis 0 (an outer
        product), or two full-rank operands sharing a non-1 axis 0 that split on a later axis
        (``g[:, None] * X``)."""
        for node in ast.walk(value):
            ops = self.operands_(node)
            if ops is None:
                continue
            exts = [self.extents_(o) for o in ops]
            if any(e is None or len(e) > rank for e in exts):
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
            peeled = [self.peel(o, idx, rank) for o in ops]
            return None if any(p is None for p in peeled) else self.rebuild(expr, peeled)
        if isinstance(expr, ast.Name):
            return ast.Subscript(value=expr, slice=expr_of(idx), ctx=ast.Load())
        if isinstance(expr, ast.Subscript):
            return self.peel_subscript(expr, idx)
        return None

    def rebuild(self, expr: ast.expr, peeled: list[ast.expr]) -> ast.expr:
        if isinstance(expr, ast.BinOp):
            return ast.BinOp(left=peeled[0], op=expr.op, right=peeled[1])
        if isinstance(expr, ast.Compare):
            return ast.Compare(left=peeled[0], ops=expr.ops, comparators=peeled[1:])
        if isinstance(expr, ast.BoolOp):
            return ast.BoolOp(op=expr.op, values=peeled)
        return ast.Call(func=expr.func, args=peeled, keywords=[])

    def peel_subscript(self, sub: ast.Subscript, idx: str) -> ast.expr | None:
        axes = self.axes_(sub)
        if not axes:
            return None
        entries = list(sub.slice.elts) if isinstance(sub.slice, ast.Tuple) else [sub.slice]
        pos, kind, ext0 = axes[0]
        if pos is None:
            if ext0 == ONE:
                return None
            entries = entries + [expr_of(idx)]  # axis 0 is an unindexed trailing base axis
        elif kind == "new":
            entries = entries[:pos] + entries[pos + 1 :]
        else:
            e = entries[pos]
            if ext0 == ONE:
                repl = e.lower if e.lower is not None else ast.Constant(value=0)
            elif e.lower is None or const_int(e.lower) == 0:
                repl = expr_of(idx)
            elif isinstance(e.lower, (ast.Name, ast.Constant)):
                repl = ast.BinOp(left=copy.deepcopy(e.lower), op=ast.Add(), right=expr_of(idx))
            else:
                return None
            entries = entries[:pos] + [repl] + entries[pos + 1 :]
        return self.tidy(sub.value, entries)

    def tidy(self, base: ast.expr, entries: list[ast.expr]) -> ast.expr:
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

    def store_base(self, target: ast.AST, rank: int) -> str | None:
        """``T[:]`` / ``T[:, :]`` over a rank-``rank`` ``T`` -> ``T``; anything else declines."""
        if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)):
            return None
        entries = list(target.slice.elts) if isinstance(target.slice, ast.Tuple) else [target.slice]
        if not all(is_full_slice(e) for e in entries) or len(entries) > rank:
            return None
        return target.value.id if self.ranks.get(target.value.id) == rank else None

    def drop_newaxes(self, value: ast.AST) -> None:
        """Drop every leading newaxis the enclosing broadcast makes redundant.

        numba's analysis equates a (1, n) operand with an (m, n) one instead of broadcasting it;
        ``x[None, :]`` -> ``x`` is exact only while the node's rank does not change."""
        for node in ast.walk(value):
            ops = self.operands_(node)
            rank = None if ops is None else expr_rank(node, self.ranks)
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
        else:
            node.args[i] = operand

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if len(node.targets) != 1:
            return node
        self.drop_newaxes(node.value)
        ext = self.extents_(node.value)
        if ext is None or len(ext) < 2 or ext[0] == ONE or not self.outer_product(node.value, len(ext)):
            return node
        rank = len(ext)
        target = node.targets[0]
        base = target.id if isinstance(target, ast.Name) else self.store_base(target, rank)
        # A target the value also reads would see rows the loop already rewrote; it, and a store that
        # is not a whole-array slice (``H[lo:hi, :]``), fills a fresh temp assigned afterwards.
        direct = base is not None and not any(isinstance(n, ast.Name) and n.id == base for n in ast.walk(node.value))
        if not direct and not isinstance(target, (ast.Name, ast.Subscript)):
            return node
        temp, ivar = f"__ob{self._ctr}", f"__ob{self._ctr}_i"
        row = self.peel(copy.deepcopy(node.value), ivar, rank)
        probe = self.peel(copy.deepcopy(node.value), "0", rank)
        if row is None or probe is None:
            return node
        self._ctr += 1
        self.changed = True
        out: list[ast.stmt] = []
        dest = base if direct and base is not None else f"{temp}_o"
        shape = ", ".join(ext)
        if isinstance(target, ast.Name):
            # A bare Name is a binding, so allocate it. The probe row carries the promoted dtype
            # (``X + Y[:, None] * 1j`` is complex though neither operand is); only its dtype is read.
            out.append(ast.Assign(targets=[ast.Name(id=temp, ctx=ast.Store())], value=probe))
            out.append(ast.parse(f"{dest} = np.empty(({shape},), {temp}.dtype)").body[0])
        elif not direct and isinstance(target, ast.Subscript):
            # The store casts into the target's dtype, so the temp takes it.
            out.append(ast.parse(f"{dest} = np.empty(({shape},), ({ast.unparse(target.value)}).dtype)").body[0])
        store = ast.parse(f"{dest}[{ivar}] = 0").body[0]
        store.value = row
        loop = ast.parse(f"for {ivar} in range({ext[0]}): pass").body[0]
        loop.body = [store]
        out.append(loop)
        if not direct:
            out.append(ast.Assign(targets=[target], value=ast.Name(id=dest, ctx=ast.Load())))
        for s in out:
            ast.copy_location(s, node)
            ast.fix_missing_locations(s)
        return out


class SliceObjectInline(ast.NodeTransformer):
    """``b = slice(lo, hi)`` read back as ``X[b, :]`` -> ``X[lo:hi, :]`` (numba only).

    :func:`expr_rank` reads a Name index as a scalar, so ``X[b, :]`` would rank 1, not 2. The
    substitution is exact when the binding is the name's only store and every name ``lo``/``hi``
    read is bound at most once. Runs before the rank table is built."""

    def __init__(self, fn: ast.FunctionDef) -> None:
        self.changed = False
        self.slices: dict[str, ast.Slice] = {}
        if any(isinstance(n, (ast.Lambda, ast.FunctionDef)) and n is not fn for n in ast.walk(fn)):
            return
        stores = name_store_counts(fn)
        for node in ast.walk(fn):
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
            read = {n.id for a in args for n in ast.walk(a) if isinstance(n, ast.Name)}
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


class ReshapeFortranOrderInline(RewritePass):
    """``x.reshape(d0, ..., dk, order="F")`` -> ``np.ascontiguousarray(x.T).reshape((dk, ..., d0)).T``
    (numba only).

    numba's ``reshape`` takes no keyword. Fortran order is C order on the axis-reversed array, so
    the transposed spelling is exact. A shape passed as one name cannot be reversed and stays verbatim."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "reshape" and node.args and len(node.keywords) == 1):
            return node
        if isinstance(f.value, ast.Name) and f.value.id in ("np", "numpy"):
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


class NumbaDtypeFixups(ast.NodeTransformer):
    """Two spellings numba's dtype-strict typing refuses where numpy accepts them (numba only).

    * ``dtype=bool``: numba does not accept the builtin ``bool`` as a dtype; ``np.bool_`` is the same.
    * real ``@`` complex: numba requires equal dtypes. Casting the real side to the complex name's
      ``.dtype`` is numpy's promotion. Only fires when both kinds are known and the complex side is a
      Name, so reading its dtype evaluates nothing twice."""

    def __init__(self, kinds: dict[str, str]) -> None:
        self.kinds = kinds
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        attr = np_attr(node)
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
