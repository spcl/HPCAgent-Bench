"""Constructor rewrites: full, eye, zeros, copies and ``np.mgrid``."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.ast_build import name_, numpy_attribute, range_for, store_
from hpcagent_bench.translators.numpyto_common.emit_helpers.numpy_names import is_numpy_module
from hpcagent_bench.translators.numpyto_common.lib_nodes.dims import NP_ZEROS_ALIASES
from hpcagent_bench.translators.numpyto_common.lowering.hoisting import StmtHoister
from hpcagent_bench.translators.numpyto_common.lowering.shape_harvest import ctor_shape_arg
from hpcagent_bench.translators.numpyto_common.lowering.shape_reads import resolve_shape_token

__all__ = [
    "CopyToAllocAndFill",
    "EyeCallHoister",
    "EyeToZerosDiagonal",
    "FullCallHoister",
    "FullLikeRewriter",
    "MgridLowering",
    "ZerosRewriter",
    "copied_name",
    "ctor_dtype_literal",
    "ctor_dtype_src",
    "shape_from_ast",
]


class FullCallHoister(StmtHoister):
    """Materialise a nested ``np.full(...)`` / ``np.full_like(...)`` call into its own
    ``__full<k> = <call>`` statement, so the direct-assign :class:`FullLikeRewriter`
    can split it into an allocation plus a broadcast fill.

    The causal-mask idiom builds its ``-inf`` band inline --
    ``scores + np.triu(np.full((n, n), -np.inf, dtype=x.dtype), 1)`` -- where ``np.full``
    is buried two calls deep. ``CallHoister``'s triu first-arg spill is gated on a
    resolvable extent, and an inline constructor is never sized by the harvest, so
    without this the whole ``np.triu`` reaches the emitter unlowered. Mirrors
    :class:`EyeCallHoister`; a call already the direct RHS of an assignment is left
    for :class:`FullLikeRewriter` to consume."""

    __slots__ = ()

    @staticmethod
    def is_full_call(v: ast.expr) -> bool:
        return (
            isinstance(v, ast.Call)
            and isinstance(v.func, ast.Attribute)
            and v.func.attr in ("full", "full_like")
            and is_numpy_module(v.func.value)
            and len(v.args) >= 2
        )

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if self.is_full_call(node):
            return self.spill(node, "__full")
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and self.is_full_call(node.value):
            return node
        return self.flush(node)


class FullLikeRewriter(ast.NodeTransformer):
    """``X = np.full_like(src, val)`` -> ``X = np.empty_like(src); X[:] = val`` and
    ``X = np.full(shape, val)`` -> ``X = np.empty(shape); X[:] = val``.

    The existing empty-alias shape harvest declares X (shape from src / the shape
    arg) and the whole-array scalar-broadcast assign fills it -- so no dedicated
    full/full_like emitter path is needed (lulesh ``pbvc = np.full_like(bvc, c1s)``)."""

    __slots__ = ()

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        v = node.value
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(v, ast.Call)
            and isinstance(v.func, ast.Attribute)
            and v.func.attr in ("full_like", "full")
            and is_numpy_module(v.func.value)
            and len(v.args) >= 2
        ):
            return node
        tgt = node.targets[0]
        alloc_attr = "empty_like" if v.func.attr == "full_like" else "empty"
        dtype_kw = [kw for kw in v.keywords if kw.arg == "dtype"]
        alloc = ast.Assign(
            targets=[store_(tgt.id)],
            value=ast.Call(
                func=numpy_attribute(alloc_attr),
                args=[v.args[0]],
                keywords=dtype_kw,
            ),
        )
        fill = ast.Assign(
            targets=[
                ast.Subscript(
                    value=name_(tgt.id),
                    slice=ast.Slice(lower=None, upper=None, step=None),
                    ctx=ast.Store(),
                )
            ],
            value=v.args[1],
        )
        for s in (alloc, fill):
            ast.copy_location(s, node)
        ast.fix_missing_locations(alloc)
        ast.fix_missing_locations(fill)
        return [alloc, fill]


class EyeCallHoister(StmtHoister):
    """Materialise a nested ``np.eye(...)`` / ``np.identity(...)`` call into its own
    ``__eye<k> = <call>`` statement, so the direct-assign :class:`EyeToZerosDiagonal`
    can lower it to a zeros + diagonal fill.

    LS3DF's generalized Rayleigh-Ritz adds an identity jitter inline --
    ``s_sub = 0.5 * (...) + 1.0e-12 * np.eye(k)`` -- where ``np.eye`` is buried in an
    expression, not a standalone assignment. A call that is already the direct RHS of
    an assignment is left in place for the diagonal rewriter to consume."""

    __slots__ = ()

    @staticmethod
    def is_eye_call(v: ast.expr) -> bool:
        return (
            isinstance(v, ast.Call)
            and isinstance(v.func, ast.Attribute)
            and v.func.attr in ("eye", "identity")
            and is_numpy_module(v.func.value)
            and bool(v.args)
        )

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if self.is_eye_call(node):
            return self.spill(node, "__eye")
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and self.is_eye_call(node.value):
            return node
        return self.flush(node)


class CopyToAllocAndFill(ast.NodeTransformer):
    """``X = a.copy()`` / ``X = np.copy(a)`` -> ``X = np.empty_like(a)`` plus ``X[:] = a``.

    The shape harvest already SHARES ``a``'s shape for a copy, but sharing a shape is not
    declaring a buffer: ``X`` never became an array descriptor, so a later whole-array read of it
    (``X[0, :, :]``) had no rank to scalarize against and the bare slice reached the emitter. Both
    halves here are primitives every backend lowers, so the existing zeros harvest declares ``X``
    and the whole-array assign lowers as any other copy would -- the same shape as the ``np.eye``
    desugar beside it.

    Name source only: a copy of a temporary (``grid[0].copy()``) has no name for ``empty_like`` to
    size against, and the shape-sharing path already declines it for the same reason. The method
    form is matched here as well as the function form, since ``MethodCallRewriter`` runs later.
    """

    __slots__ = ()

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        src = copied_name(node)
        if src is None or not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            return node
        alloc = ast.Assign(
            targets=[copy.deepcopy(node.targets[0])],
            value=ast.Call(
                func=numpy_attribute("empty_like"),
                args=[copy.deepcopy(src)],
                keywords=[],
            ),
        )
        fill = ast.Assign(
            targets=[ast.Subscript(value=copy.deepcopy(node.targets[0]), slice=ast.Slice(), ctx=ast.Store())],
            value=copy.deepcopy(src),
        )
        return [ast.copy_location(alloc, node), ast.copy_location(fill, node)]


def copied_name(node: ast.Assign) -> ast.Name | None:
    """The array a whole-array copy assignment reads, for either spelling, else ``None``."""
    value = node.value
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "copy"
        and not value.keywords
    ):
        return None
    receiver = value.func.value
    if is_numpy_module(receiver):
        return value.args[0] if len(value.args) == 1 and isinstance(value.args[0], ast.Name) else None
    return receiver if isinstance(receiver, ast.Name) and not value.args else None


class EyeToZerosDiagonal(ast.NodeTransformer):
    """``X = np.eye(n)`` / ``np.eye(m, n)`` / ``np.identity(n)`` -> a zeros
    allocation plus an explicit diagonal fill::

        X = np.zeros((n, n))            # or (m, n)
        for __eye<k> in range(n):       # range(min(m, n)) when rectangular
            X[__eye<k>, __eye<k>] = 1.0

    Built from primitives every backend already lowers (``np.zeros`` + a loop +
    a scalar store), so no per-emitter identity path is needed. The zeros
    harvest then declares X and picks up the ``(n, n)`` shape as usual. Native
    lowering only -- the python backends keep the builtin ``np.eye``.
    """

    __slots__ = ("_n",)

    def __init__(self) -> None:
        self._n = 0

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        v = node.value
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(v, ast.Call)
            and isinstance(v.func, ast.Attribute)
            and v.func.attr in ("eye", "identity")
            and is_numpy_module(v.func.value)
            and v.args
        ):
            return node
        tgt = node.targets[0].id
        rows = v.args[0]
        # ``eye(m, n)`` with a real second extent is rectangular (diagonal =
        # min(m, n)); ``eye(n)`` / ``identity(n)`` (or ``eye(n, None)``) is square.
        rectangular = (
            v.func.attr == "eye"
            and len(v.args) >= 2
            and not (isinstance(v.args[1], ast.Constant) and v.args[1].value is None)
        )
        cols = v.args[1] if rectangular else copy.deepcopy(rows)
        # ``k=`` (or eye's 3rd positional) shifts the unit diagonal: numpy writes 1.0 at
        # (i, i+k). Fill X[t + off_r, t + off_c] for t in range(min(M - off_r, N - off_c))
        # with off_r = max(0, -k), off_c = max(0, k); k == 0 is the plain main diagonal.
        # ``identity`` has no k.
        k_node = None
        for kw in v.keywords:
            if kw.arg == "k":
                k_node = kw.value
        if k_node is None and v.func.attr == "eye" and len(v.args) >= 3:
            k_node = v.args[2]
        it = f"__diag{self._n}"
        self._n += 1

        def shift(expr, off, op):
            if isinstance(off, int):
                return (
                    copy.deepcopy(expr)
                    if off == 0
                    else ast.BinOp(left=copy.deepcopy(expr), op=op, right=ast.Constant(value=off))
                )
            return ast.BinOp(left=copy.deepcopy(expr), op=op, right=copy.deepcopy(off))

        if k_node is None or (isinstance(k_node, ast.Constant) and k_node.value == 0):
            count = (
                ast.Call(
                    func=name_("min"),
                    args=[copy.deepcopy(rows), copy.deepcopy(cols)],
                    keywords=[],
                )
                if rectangular
                else copy.deepcopy(rows)
            )
            row_idx, col_idx = name_(it), name_(it)
        else:
            off_r: int | ast.expr
            off_c: int | ast.expr
            if isinstance(k_node, ast.Constant) and isinstance(k_node.value, int):
                off_r, off_c = max(0, -k_node.value), max(0, k_node.value)
            else:
                off_r = ast.Call(
                    func=name_("max"),
                    args=[ast.Constant(value=0), ast.UnaryOp(op=ast.USub(), operand=copy.deepcopy(k_node))],
                    keywords=[],
                )
                off_c = ast.Call(
                    func=name_("max"),
                    args=[ast.Constant(value=0), copy.deepcopy(k_node)],
                    keywords=[],
                )
            count = ast.Call(
                func=name_("min"),
                args=[shift(rows, off_r, ast.Sub()), shift(cols, off_c, ast.Sub())],
                keywords=[],
            )
            row_idx = shift(name_(it), off_r, ast.Add())
            col_idx = shift(name_(it), off_c, ast.Add())

        dtype_kw = [kw for kw in v.keywords if kw.arg == "dtype"]
        zeros = ast.Assign(
            targets=[store_(tgt)],
            value=ast.Call(
                func=numpy_attribute("zeros"),
                args=[ast.Tuple(elts=[copy.deepcopy(rows), cols], ctx=ast.Load())],
                keywords=dtype_kw,
            ),
        )
        loop = range_for(
            it,
            [count],
            [
                ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=name_(tgt),
                            slice=ast.Tuple(elts=[row_idx, col_idx], ctx=ast.Load()),
                            ctx=ast.Store(),
                        )
                    ],
                    value=ast.Constant(value=1.0),
                )
            ],
        )
        for s in (zeros, loop):
            ast.copy_location(s, node)
        ast.fix_missing_locations(zeros)
        ast.fix_missing_locations(loop)
        return [zeros, loop]


class ZerosRewriter(ast.NodeTransformer):
    """Turn ``x = np.zeros((N, K))`` (and family) into a side-table entry.

    Recognises every member of :data:`numpyto_common.lib_nodes.NP_ZEROS_ALIASES`
    (``np.zeros`` / ``np.empty`` / ``np.ones`` / ``np.zeros_like`` /
    ``np.empty_like``). For the ``_like`` forms the LHS shape comes
    from the named array (looked up in :attr:`shape_table`) instead
    of from the call's explicit shape argument.
    """

    __slots__ = ("aliases", "dtype_literal", "dtype_src", "fills", "shape_table", "zeros")

    def __init__(self, shape_table: dict[str, tuple[str, ...]] | None = None) -> None:
        self.zeros: dict[str, tuple[str, ...]] = {}
        # Fill kind per harvested local, keyed by name: the constructor
        # attr (``zeros`` / ``ones`` / ``empty`` / ``zeros_like`` / ...).
        # Lets the emitter pick the right initialiser when a constructor
        # aliases an OUTPUT parameter (zeros -> memset 0, ones -> fill 1,
        # empty -> nothing) instead of declaring a shadowing local.
        self.fills: dict[str, str] = {}
        #: Harvested local -> the array whose dtype it was built to match, for the two constructors
        #: that say so: ``np.zeros_like(a)`` and ``np.zeros(shape, a.dtype)``. Without it every such
        #: local falls back to the kernel's default float, which is not a slower answer for a
        #: COMPLEX source -- it is a buffer half the width, and the imaginary part is dropped on the
        #: way in, with nothing on the path saying so.
        self.dtype_src: dict[str, str] = {}
        #: Harvested local -> the dtype its constructor named OUTRIGHT (``np.zeros(n, np.float64)``).
        #: A stated dtype is a decision, not a hint: the eigenVALUE array of a complex Hermitian
        #: problem is declared real on purpose, and inference that reaches it through the complex
        #: matrix it was computed from would widen it and then compare two complex with ``<``.
        self.dtype_literal: dict[str, str] = {}
        self.aliases = set(NP_ZEROS_ALIASES)
        self.shape_table = shape_table or {}

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and is_numpy_module(node.value.func.value)
            and node.value.func.attr in self.aliases
        ):
            name = node.targets[0].id
            attr = node.value.func.attr
            shape: tuple[str, ...] | None = None
            if attr.endswith("_like"):
                # ``np.zeros_like(a)`` -> share ``a``'s shape, and its dtype.
                if node.value.args and isinstance(node.value.args[0], ast.Name):
                    other = node.value.args[0].id
                    shape = self.shape_table.get(other)
                    self.dtype_src[name] = other
            else:
                shape_arg = ctor_shape_arg(node.value)
                shape = shape_from_ast(shape_arg, self.shape_table)
                src = ctor_dtype_src(node.value)
                if src is not None:
                    self.dtype_src[name] = src
                lit = ctor_dtype_literal(node.value)
                if lit is not None:
                    self.dtype_literal[name] = lit
            if shape is not None:
                self.zeros[name] = shape
                self.fills[name] = attr
                # Replace the call with a marker the emitter recognises.
                node.value = ast.Call(
                    func=name_("__hpcagent_bench_zeros__"),
                    args=[],
                    keywords=[],
                )
        return node


def ctor_dtype_src(call: ast.Call) -> str | None:
    """The array a constructor's ``dtype`` argument points at -- ``np.zeros(shape, a.dtype)`` or
    ``np.zeros(shape, dtype=a.dtype)`` -> ``"a"``. ``None`` for a literal dtype or none at all."""
    kw = {k.arg: k.value for k in call.keywords}
    node = kw.get("dtype") or (call.args[1] if len(call.args) > 1 else None)
    if isinstance(node, ast.Attribute) and node.attr == "dtype" and isinstance(node.value, ast.Name):
        return node.value.id
    return None


def ctor_dtype_literal(call: ast.Call) -> str | None:
    """The dtype a constructor names outright -- ``np.zeros(n, np.float64)`` -> ``"float64"``.
    ``None`` when the dtype is absent or comes from another array."""
    kw = {k.arg: k.value for k in call.keywords}
    node = kw.get("dtype") or (call.args[1] if len(call.args) > 1 else None)
    if isinstance(node, ast.Attribute) and node.attr != "dtype":
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def shape_from_ast(node, shape_table=None) -> tuple[str, ...]:
    """Return the source-level shape ``(N, K)`` from an AST tuple / list / int.

    ``np.empty([x.shape[0], x.shape[1] // 2, ...])`` resolves the per-axis
    ``arr.shape[i]`` references against the optional ``shape_table``.
    ``np.zeros(C.shape, ...)`` mirrors C's shape from the table.
    """
    if node is None:
        return ()
    if isinstance(node, (ast.Tuple, ast.List)):
        if shape_table is not None:
            return tuple(resolve_shape_token(e, shape_table) for e in node.elts)
        return tuple(ast.unparse(e) for e in node.elts)
    # ``np.zeros(C.shape, ...)`` -- single-arg whole-shape mirror.
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "shape"
        and isinstance(node.value, ast.Name)
        and shape_table is not None
    ):
        src = shape_table.get(node.value.id)
        if src is not None:
            return tuple(src)
    # A single scalar shape arg (``np.zeros(M.shape[0], np.float64)`` -- the eigh
    # eigenvalue vector's 1-D extent over a LOCAL operand): resolve an
    # ``arr.shape[i]`` token the same way a shape-TUPLE element already is, so the
    # local's ``.shape[0]`` folds to its dimension symbol instead of surviving as an
    # unlowerable ``M.shape[0]`` malloc / allocate extent.
    if shape_table is not None:
        return (resolve_shape_token(node, shape_table),)
    return (ast.unparse(node),)


class MgridLowering(ast.NodeTransformer):
    """Lower ``X0, X1, ... = np.mgrid[a0:b0, a1:b1, ...]`` to a
    sequence of ``Xk = np.zeros(shape, dtype=np.int64)`` markers
    plus one per-element init loop per axis.

    ``np.mgrid[0:R, 0:S]`` returns two 2-D arrays of shape (R, S)::

        I[i, j] = i
        J[i, j] = j

    NumpyToC has no shape-mutating ``mgrid`` object; we expand it
    eagerly into the per-element initialisers and emit `np.empty`
    declarations whose shape harvest then picks them up like any
    other local array.
    """

    __slots__ = ()

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)):
            return node
        rhs = node.value
        if not (
            isinstance(rhs, ast.Subscript)
            and isinstance(rhs.value, ast.Attribute)
            and is_numpy_module(rhs.value.value)
            and rhs.value.attr == "mgrid"
        ):
            return node
        targets = [t for t in node.targets[0].elts if isinstance(t, ast.Name)]
        if len(targets) != len(node.targets[0].elts):
            return node
        sl = rhs.slice
        entries = sl.elts if isinstance(sl, ast.Tuple) else [sl]
        axes = [a for a in entries if isinstance(a, ast.Slice)]
        if len(axes) != len(targets) or len(axes) != len(entries):
            return node
        highs = [ax.upper for ax in axes if ax.upper is not None]
        if len(highs) != len(axes):
            return node
        shape_elts: list[ast.expr] = []
        for ax, hi in zip(axes, highs):
            lo = ax.lower if ax.lower is not None else ast.Constant(value=0)
            shape_elts.append(ast.BinOp(left=hi, op=ast.Sub(), right=lo))
        shape_tuple = ast.Tuple(elts=shape_elts, ctx=ast.Load())
        out: list[ast.stmt] = []
        iters = [name_(f"__mg{k}") for k in range(len(targets))]
        for k, tgt in enumerate(targets):
            out.append(
                ast.Assign(
                    targets=[store_(tgt.id)],
                    value=ast.Call(
                        func=numpy_attribute("empty"),
                        args=[shape_tuple],
                        keywords=[
                            ast.keyword(
                                arg="dtype",
                                value=ast.Attribute(value=name_("np"), attr="int64", ctx=ast.Load()),
                            )
                        ],
                    ),
                )
            )
            lower_k = axes[k].lower
            lo_k = lower_k if lower_k is not None else ast.Constant(value=0)
            idx_expr: ast.expr = name_(iters[k].id)
            if not (isinstance(lo_k, ast.Constant) and lo_k.value == 0):
                idx_expr = ast.BinOp(left=idx_expr, op=ast.Add(), right=lo_k)
            slice_form = iters[0] if len(iters) == 1 else ast.Tuple(elts=[name_(it.id) for it in iters], ctx=ast.Load())
            body = [
                ast.Assign(
                    targets=[ast.Subscript(value=name_(tgt.id), slice=slice_form, ctx=ast.Store())],
                    value=idx_expr,
                )
            ]
            # Wrap the body in nested loops, deepest first.
            stmt: list[ast.stmt] = [*body]
            for it, ax, ax_hi in zip(reversed(iters), reversed(axes), reversed(highs)):
                ax_lo = ax.lower if ax.lower is not None else ast.Constant(value=0)
                bound = (
                    ax_hi
                    if isinstance(ax_lo, ast.Constant) and ax_lo.value == 0
                    else ast.BinOp(left=ax_hi, op=ast.Sub(), right=ax_lo)
                )
                stmt = [range_for(it.id, [bound], stmt)]
            out.extend(stmt)
        return out
