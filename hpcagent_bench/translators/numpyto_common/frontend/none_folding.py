"""``None`` handling: static ``is None`` folds and first-iteration ``None``-seeded accumulators."""

import ast

from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet


class FoldStaticNoneBranches(ast.NodeTransformer):
    """Constant-fold a decidable ``is [not] None`` compare and eliminate the now-dead
    ``IfExp``/``if`` branches.

    Inlining a helper with an OPTIONAL parameter (``def f(a, mask=None): ...
    if mask is not None: ...``) substitutes the call site's argument for that
    parameter, so the guard becomes decidable either way:

    * the argument was OMITTED -> the literal ``None`` is substituted, leaving
      ``if None is not None:`` (fv3_dycore's FiniteVolumeTransport);
    * the argument was SUPPLIED -> the expression passed is substituted, leaving
      ``x.shape[0] if x.shape[0] is None else int(x.shape[0])`` (examinimd passes
      ``n_local=x.shape[0]``), and an indexing expression is never ``None``.

    Both fold, because a backend can emit neither: there is no ``is`` operator in
    the C/Fortran comparison tables and no ``None`` literal to compare against.

    What is NOT folded is a compare whose non-``None`` side is a bare NAME: a local
    genuinely bound to ``None`` (``out = None`` ... ``if out is None:``) is a real
    runtime question. A kernel PARAMETER name is decidable -- always supplied across
    the C ABI -- and :class:`FoldParamNoneGuard` folds that case, where the
    parameter list is known. ``None`` as a subscript index (``np.newaxis``) is never
    an ``is`` operand.
    """

    #: Expression forms that cannot evaluate to ``None`` whatever their operands are bound to:
    #: indexing/attribute access yields an element, arithmetic yields a number, a comparison
    #: yields a bool, a display yields a container. Deliberately excludes ``Name`` (may be bound
    #: to ``None``), ``Call`` (a helper may return it) and ``BoolOp`` (``a or None``).
    NEVER_NONE = (ast.Subscript, ast.Attribute, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Tuple, ast.List)

    @staticmethod
    def is_static_none(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and node.value is None

    @classmethod
    def never_none(cls, node: ast.AST) -> bool:
        return isinstance(node, cls.NEVER_NONE) or (isinstance(node, ast.Constant) and node.value is not None)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if not (len(node.ops) == 1 and isinstance(node.ops[0], (ast.Is, ast.IsNot))):
            return node
        left, right = node.left, node.comparators[0]
        none_left, none_right = self.is_static_none(left), self.is_static_none(right)
        if none_left == none_right:  # neither or both ``None`` -> nothing to decide against
            if none_left:
                return ast.copy_location(ast.Constant(value=isinstance(node.ops[0], ast.Is)), node)
            return node
        if not self.never_none(right if none_left else left):
            return node
        # ``<never None> is None`` -> False, ``is not None`` -> True.
        return ast.copy_location(ast.Constant(value=isinstance(node.ops[0], ast.IsNot)), node)

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
            return node.body if node.test.value else node.orelse
        return node

    def visit_If(self, node: ast.If) -> ast.stmt | list[ast.stmt]:
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
            # Splice in the live branch (a stmt list); an empty branch -> drop.
            return node.body if node.test.value else node.orelse
        return node


def bare_none_assign_target(stmt: ast.stmt) -> str | None:
    """The name ``X`` when ``stmt`` is exactly ``X = None``, else ``None``."""
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is None
    ):
        return stmt.targets[0].id
    return None


def none_compare(test: ast.expr) -> tuple[str, bool] | None:
    """``(name, is_op)`` for a decidable ``<name> is[ not] None`` compare with the Name on either
    side (``is_op`` True for ``is``, False for ``is not``), else ``None``."""
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], (ast.Is, ast.IsNot))):
        return None
    left, right = test.left, test.comparators[0]
    none_left = isinstance(left, ast.Constant) and left.value is None
    none_right = isinstance(right, ast.Constant) and right.value is None
    if none_left == none_right:  # neither or both -> undecidable
        return None
    target = right if none_left else left
    if not isinstance(target, ast.Name):
        return None
    return target.id, isinstance(test.ops[0], ast.Is)


def none_toggle_op(test: ast.expr, name: str) -> bool | None:
    """``True``/``False`` for a decidable ``<name> is[ not] None`` compare naming ``name``, else
    ``None``. ``True`` for ``is`` (the branch taken while ``name`` is still ``None``), ``False`` for
    ``is not``."""
    decoded = none_compare(test)
    if decoded is None or decoded[0] != name:
        return None
    return decoded[1]


def assigns_name(stmts: list[ast.stmt], name: str) -> bool:
    """Whether some statement in ``stmts`` writes ``name`` directly -- ``name = <expr>`` (the seed
    branch, ``out = patch.copy()``) or ``name += <expr>`` (the combiner branch: avgpool_core's
    running sum keeps its own ``+=`` rather than an ``np.add(acc, patch, out=acc)`` roundtrip)."""
    return any(
        (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
            and s.targets[0].id == name
        )
        or (isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name) and s.target.id == name)
        for s in stmts
    )


def flag_guard(flag: str, seed_when_none: bool) -> ast.Compare:
    """``flag == 0`` (mirrors an original ``is``) or ``flag != 0`` (mirrors ``is not``) -- ``flag``
    is 0 exactly while the accumulator would still have read as ``None``, so either comparison keeps
    the ORIGINAL branch taken on the very first pass and its mirror on every later one."""
    op: ast.cmpop = ast.Eq() if seed_when_none else ast.NotEq()
    return ast.Compare(left=ast.Name(id=flag, ctx=ast.Load()), ops=[op], comparators=[ast.Constant(value=0)])


def flag_set_stmt(flag: str) -> ast.Assign:
    return ast.Assign(targets=[ast.Name(id=flag, ctx=ast.Store())], value=ast.Constant(value=1))


def rewrite_none_toggle(stmts: list[ast.stmt], start: int, name: str, flag: str, in_loop: bool) -> bool:
    """Find the (possibly nested) first-ITERATION toggle on ``name`` at or after index ``start`` of
    ``stmts`` and rewrite it in place to test ``flag`` instead of ``name``'s ``None``-ness, then mark
    ``flag`` seen right after it. Recurses into every nested block; returns ``True`` once one toggle
    is found and rewritten (a second accumulator sharing the same seed name would need its own flag,
    one call each).

    ``in_loop`` (True once the search has descended into a ``for``/``while``) is what tells a genuine
    per-ITERATION toggle (max_pooling_2d's ``acc = tap if acc is None else np.maximum(acc, tap)``,
    re-decided every pass through the loop) apart from a plain default-argument fold resolved once,
    straight-line (conv2d_avg_pool_sigmoid_sum's inlined ``stride = kernel_size if stride is None
    else _as_tuple(stride, 2)`` -- ALSO self-referential in its non-None branch, but never
    re-executed, so :mod:`tuple_desugar`'s own ``x is None`` kind-tracking already folds it; peeling
    it here first would just hide the ``None`` from that fold behind an equally unresolved flag).
    Outside a loop this declines and leaves the ``None`` for that pass to handle.

    Takes the REAL statement list plus a start index rather than a pre-sliced sublist: the toggle
    site gets ``flag = 1`` spliced in with ``list.insert``, which only lands in the tree ``stmts``
    itself is -- a slice copy's insert is invisible to the caller.
    """
    for idx in range(start, len(stmts)):
        stmt = stmts[idx]
        if in_loop and isinstance(stmt, ast.If):
            seed_when_none = none_toggle_op(stmt.test, name)
            if seed_when_none is not None and assigns_name(stmt.body, name) and assigns_name(stmt.orelse, name):
                stmt.test = flag_guard(flag, seed_when_none)
                stmts.insert(idx + 1, flag_set_stmt(flag))
                return True
        elif (
            in_loop
            and isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name
            and isinstance(stmt.value, ast.IfExp)
        ):
            seed_when_none = none_toggle_op(stmt.value.test, name)
            if seed_when_none is not None:
                # A ternary REPLACES the whole Assign with an if/else statement rather than just
                # swapping its test in place: max_pooling_2d's ``acc`` is an ARRAY, and a C/C++
                # ternary on two array operands does not compile (confirmed: gcc rejects
                # ``acc = flag ? tap : fmax(acc, tap)`` outright, "invalid operands ... double and
                # double *") -- unlike a per-branch ARRAY ASSIGN, which the emitters already lower
                # as a whole-array copy either way (this is exactly the shape densenet's own
                # if/else spelling already produces and compiles clean).
                new_if = ast.If(
                    test=flag_guard(flag, seed_when_none),
                    body=[ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=stmt.value.body)],
                    orelse=[ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=stmt.value.orelse)],
                )
                ast.copy_location(new_if, stmt)
                ast.fix_missing_locations(new_if)
                stmts[idx] = new_if
                stmts.insert(idx + 1, flag_set_stmt(flag))
                return True
        for field in ("body", "orelse"):
            nested = vars(stmt).get(field)
            if isinstance(nested, list):
                nested_in_loop = in_loop or isinstance(stmt, (ast.For, ast.While))
                if rewrite_none_toggle(nested, 0, name, flag, nested_in_loop):
                    return True
    return False


class PeelNoneSeededAccumulators(ast.NodeTransformer):
    """``X = None`` followed (anywhere below, typically inside a loop nest) by a first-iteration
    toggle on ``X`` -- either ``X = seed if X is None else combiner(X, ...)`` (max_pooling_2d's
    tap-loop) or ``if X is None: X = seed`` / ``else: X = combiner(X, ...)`` (densenet's
    ``out``/``acc`` pooling cores) -- rewritten to an explicit ``__x_seen`` flag: ``X = None``
    becomes ``__x_seen = 0``, the toggle's ``X is[not] None`` becomes ``__x_seen ==[!]= 0``, and
    ``__x_seen = 1`` is inserted right after the toggle.

    Neither backend has a ``None`` value, so a local that reads as ``None`` on its first use and a
    real array afterward has no direct C/Fortran translation. This is NOT the same case
    :class:`numpyto_common.lowering.calls.ConditionalNoneAllocRewriter` handles (a buffer that is
    genuinely allocated under one runtime condition and never read otherwise, where forcing the
    allocated branch is sound) -- here ``X``'s ``None``-ness IS observed, every single time the loop
    runs, which is exactly the case that rewriter declines. The flag replays the SAME state machine
    the ``None`` check already was (0 = "not seen yet") without assuming anything about the
    combiner -- no reduction identity (``-inf`` for ``np.maximum``, ``+inf`` for ``np.minimum``, ...)
    needs to be known or guessed, so this is sound for any first-iteration seed, not only max/min.

    Declines (leaves the ``None`` standing, for :func:`drop_dead_none_bindings` or an eventual
    refusal to sort out) when no matching toggle is found below the bind -- a local genuinely
    returned or read as ``None`` is a different, unhandled shape, not this one.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        taken = OrderedSet(n.id for n in ast.walk(node) if isinstance(n, ast.Name))
        self.rewrite_block(node.body, taken)
        return node

    def rewrite_block(self, stmts: list[ast.stmt], taken: OrderedSet[str]) -> None:
        i = 0
        while i < len(stmts):
            stmt = stmts[i]
            name = bare_none_assign_target(stmt)
            if name is not None:
                flag = unique_name(f"__{name}_seen", taken)
                if rewrite_none_toggle(stmts, i + 1, name, flag, in_loop=False):
                    taken.add(flag)
                    stmt.targets[0].id = flag
                    stmt.value = ast.Constant(value=0)
                    ast.fix_missing_locations(stmt)
            else:
                # The bind itself may sit inside a branch/loop rather than at this exact level
                # (a guarded accumulator init); keep looking one level down for more starts.
                for field in ("body", "orelse"):
                    nested = vars(stmt).get(field)
                    if isinstance(nested, list):
                        self.rewrite_block(nested, taken)
            i += 1


def unique_name(base: str, taken: OrderedSet[str]) -> str:
    """``base``, or ``base`` suffixed with a counter, that is not already in ``taken``."""
    if base not in taken:
        return base
    k = 1
    while f"{base}{k}" in taken:
        k += 1
    return f"{base}{k}"
