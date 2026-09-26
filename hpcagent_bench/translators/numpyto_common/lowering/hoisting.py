"""Statement hoisting base plus method-call and computed-index hoisters."""

import ast

from hpcagent_bench.translators.numpyto_common.lowering.mathfuncs import METHOD_TO_NP

__all__ = ["ComputedIndexCallHoister", "MethodCallRewriter", "StmtHoister"]


class StmtHoister(ast.NodeTransformer):
    """Base for rewriters that must lift a sub-expression into a fresh temp
    assignment emitted immediately before the statement that contains it.

    A subclass calls :meth:`spill` from its expression visitor to swap an
    inline sub-expression for a fresh Name and stage ``<temp> = <expr>`` in
    :attr:`pre_stmts`; this base flushes the staged assignments into the
    enclosing block right before the current statement. Mirrors the
    ``pre_stmts`` lift ``ScalarTimesMatmulRewriter`` uses -- generalised so the
    splice happens inline (``NodeTransformer`` flattens a returned statement
    list into the parent body) rather than at the top-level driver, so a spill
    inside a loop / branch body lands in that same body at any nesting depth.

    The per-statement save/restore of :attr:`pre_stmts` keeps a spill from a
    compound statement's header (``if`` test / ``for`` iter) separate from
    spills produced by its body statements: the header's temps flush before the
    compound statement, each body statement's temps flush before that body
    statement.
    """

    def __init__(self) -> None:
        #: Temp assignments staged for the statement currently being flushed.
        self.pre_stmts: list[ast.stmt] = []
        #: Monotonic id for unique hoist-temp names across the whole body.
        self._hoist_ctr: list[int] = [0]

    def spill(self, expr: ast.expr, prefix: str) -> ast.Name:
        """Stage ``<prefix><n> = <expr>`` and return a Load Name for the temp."""
        self._hoist_ctr[0] += 1
        name = f"{prefix}{self._hoist_ctr[0]}"
        self.pre_stmts.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=expr))
        return ast.Name(id=name, ctx=ast.Load())

    def flush(self, node: ast.stmt):
        saved = self.pre_stmts
        self.pre_stmts = []
        self.generic_visit(node)
        pre = self.pre_stmts
        self.pre_stmts = saved
        if not pre:
            return node
        for s in pre:
            ast.copy_location(s, node)
            ast.fix_missing_locations(s)
        return pre + [node]

    visit_Assign = flush
    visit_AugAssign = flush
    visit_Expr = flush
    visit_Return = flush
    visit_If = flush
    visit_While = flush
    visit_For = flush


class MethodCallRewriter(StmtHoister):
    """Translate ``a.copy()``, ``A.max()`` etc. into their ``np.``
    counterparts so the LibNodeRewriter picks them up uniformly.

    Fires when the receiver is a bare Name (a parameter or declared local) or a
    Subscript of one -- neither a module-like identifier (``np`` / ``numpy`` /
    ``math`` / ``scipy``), else ``np.max(x)`` would wrongly become
    ``np.max(np, x)``. A Call receiver (``np.abs(rho_in - rho_out).sum()``) is
    hoisted to a fresh temp first (:class:`StmtHoister`), so the method operates
    on a bare Name -- the reduction expanders and backends never accept an
    inline sub-expression receiver.
    """

    __slots__ = ()

    MODULE_NAMES = frozenset({"np", "numpy", "math", "scipy"})

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        # ``a.ravel()`` / ``a.flatten()`` -> ``np.reshape(a, (-1,))``: a 1-D view /
        # copy the reshape expander already lowers (``.ravel() @ .ravel()`` is the
        # flattened-dot idiom in the CG kernels).
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("ravel", "flatten")
            and not node.args
            and isinstance(func.value, ast.Name)
            and func.value.id not in self.MODULE_NAMES
        ):
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="reshape", ctx=ast.Load()),
                args=[func.value, ast.Tuple(elts=[ast.Constant(value=-1)], ctx=ast.Load())],
                keywords=[],
            )
        if not (isinstance(func, ast.Attribute) and func.attr in METHOD_TO_NP):
            return node
        recv = func.value
        # Receiver may be a bare Name (a parameter / declared local) or a
        # Subscript of one (``grid[0].copy()`` -- a row materialised into a
        # fresh local). Both lower through the same ``np.<fn>`` expanders, which
        # scalarize Subscript operands. A module identifier (``np.max(x)``) is
        # never a receiver here -- that ``x`` is the argument, not the receiver.
        if isinstance(recv, ast.Call):
            # Call receiver (``np.abs(rho_in - rho_out).sum()``): materialise the
            # inner Call into a fresh temp emitted before this statement, then
            # reduce over the bare Name. The reduction expanders / backends only
            # accept a Name or Subscript receiver, never an inline Call.
            recv = self.spill(recv, "__mc")
        elif not (
            (isinstance(recv, ast.Name) and recv.id not in self.MODULE_NAMES)
            or (
                isinstance(recv, ast.Subscript)
                and isinstance(recv.value, ast.Name)
                and recv.value.id not in self.MODULE_NAMES
            )
        ):
            return node
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=METHOD_TO_NP[func.attr], ctx=ast.Load()),
            args=[recv] + list(node.args),
            keywords=node.keywords,
        )


class ComputedIndexCallHoister(StmtHoister):
    """Hoist a Call used as a subscript index into a fresh temp Name assignment
    emitted before the statement, so the index is a bare Name the backends emit.

    ``U[np.argmax(absU[:, j]), j]`` / ``v[np.argmax(np.abs(v))]`` -- a library
    Call sitting in index position -- becomes ``__ix = np.argmax(...)`` staged
    before the statement and the subscript indexes with ``__ix``. Structural, not
    argmax-specific: any Call index (single-index or ANY position of a tuple
    subscript) is spilled EXCEPT the scalar builtins the emitters already render
    inline as an index expression (``int`` / ``abs`` / ``min`` / ``max`` /
    ``len`` / ``round``), which need no pre-statement.

    ``argmax`` / ``argmin`` index calls need one extra step: their expander
    requires a bare-Name operand, but the LibNode call-hoister only materialises a
    non-Name first arg for the VALUE reductions (max / min / sum / ...), never for
    argmax / argmin. So when the hoisted index call is an argmax / argmin over a
    non-Name operand -- a slice (``absU[:, j]``) or a nested Call
    (``np.abs(w)``) -- that operand is materialised into its own temp Name first
    (a whole-array copy the later lift lowers), so the reduction reaches its
    expander with a Name operand.
    """

    __slots__ = ()

    #: Scalar builtins each backend renders inline in index position -- left in
    #: place so a plain ``hist[int(x)]`` does not gain a needless spill temp.
    INLINE_INDEX_BUILTINS = frozenset({"int", "abs", "min", "max", "len", "round"})

    #: ``np`` arg-reductions whose expander needs a bare-Name operand (the
    #: LibNode call-hoister leaves their non-Name first arg unmaterialised).
    ARG_REDUCTIONS = frozenset({"argmax", "argmin"})

    def should_hoist(self, e: ast.expr) -> bool:
        if not isinstance(e, ast.Call):
            return False
        f = e.func
        if isinstance(f, ast.Name) and f.id in self.INLINE_INDEX_BUILTINS:
            return False
        return True

    def is_arg_reduction(self, call: ast.Call) -> bool:
        f = call.func
        return (
            isinstance(f, ast.Attribute)
            and f.attr in self.ARG_REDUCTIONS
            and isinstance(f.value, ast.Name)
            and f.value.id in ("np", "numpy")
        )

    def hoist_index(self, e: ast.Call) -> ast.Name:
        """Spill index Call ``e`` to a fresh Name. For an argmax / argmin over a
        non-Name operand, materialise that operand into its own temp Name first so
        the reduction expander (which needs a Name operand) can lower it."""
        if self.is_arg_reduction(e) and e.args and not isinstance(e.args[0], ast.Name):
            e.args[0] = self.spill(e.args[0], "__ixa")
        return self.spill(e, "__ix")

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        sl = node.slice
        is_tuple = isinstance(sl, ast.Tuple)
        elts = list(sl.elts) if is_tuple else [sl]
        new_elts = [self.hoist_index(e) if self.should_hoist(e) else e for e in elts]
        if new_elts != elts:
            node.slice = ast.Tuple(elts=new_elts, ctx=ast.Load()) if is_tuple else new_elts[0]
        return node
