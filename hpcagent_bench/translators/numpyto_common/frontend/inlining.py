"""Helper inlining: inlinable forms, the inliner, static-flag guards, constant-list loop unrolling."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet


def has_loop_control(body: list[ast.stmt]) -> bool:
    """True when ``body`` has a ``break``/``continue`` bound to its own loop
    (not one nested inside a further For/While, which would capture it)."""

    def walk_(stmts: list[ast.stmt]) -> bool:
        for s in stmts:
            if isinstance(s, (ast.Break, ast.Continue)):
                return True
            if isinstance(s, (ast.For, ast.While, ast.FunctionDef)):
                continue  # a nested loop captures its own break/continue
            for f in ("body", "orelse", "finalbody"):
                sub = vars(s).get(f)
                if isinstance(sub, list) and walk_(sub):
                    return True
            for h in vars(s).get("handlers") or []:
                if walk_(h.body):
                    return True
        return False

    return walk_(body)


def resolve_call_args(call: ast.Call, helper: ast.FunctionDef) -> list[ast.expr] | None:
    """Pair call-site arguments with the helper's positional
    parameters, filling unsupplied trailing parameters with their
    default value when ``helper.args.defaults`` provides one.

    ``def batchnorm2d(x, eps=1e-5)`` called as ``batchnorm2d(arr)``
    yields ``[arr, Constant(1e-5)]``.

    KEYWORD arguments bind by name (``_logsumexp(x, axis=1)``), never to the parameter's default.

    Returns ``None`` when the call cannot be reconciled (too many positional args, an unknown or
    doubly-bound keyword, or a missing param without a default) -- the inliner then leaves the Call
    untouched.
    """
    param_names = [a.arg for a in helper.args.args]
    defaults = dict(zip(param_names[len(param_names) - len(helper.args.defaults) :], helper.args.defaults))
    call_args = list(call.args)
    if len(call_args) > len(param_names) or any(kw.arg is None for kw in call.keywords):
        return None  # too many positionals, or a **kwargs splat we cannot resolve
    bound = dict(zip(param_names, call_args))
    for kw in call.keywords:
        if kw.arg not in param_names or kw.arg in bound:
            return None
        bound[kw.arg] = kw.value
    resolved = [bound.get(name, defaults.get(name)) for name in param_names]
    present = [a for a in resolved if a is not None]
    return present if len(present) == len(resolved) else None


def strip_docstrings_(stmts: list[ast.stmt]) -> list[ast.stmt]:
    """Return ``stmts`` with leading / standalone string-literal Expr
    statements removed.

    Helper-body docstrings show up as ``Expr(Constant(str))`` and would
    otherwise be treated as statements by the inliner / classifier.
    """
    return [
        s
        for s in stmts
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))
    ]


#: Statements a helper body may hold and still be spliceable into its caller.
#: ``Assert``/``Pass`` earn their place the same way they do downstream: a kernel runs on
#: oracle-validated inputs, so an ``assert groups == 1`` never fires, and both the emitter and
#: ``numpy_desugar`` already drop one. Excluding them here did not make the helper safer -- it made
#: it UNINLINABLE, and a helper that is not inlined survives as a call into a ``@dc.program``,
#: which binds no helper at all: conv_pointwise_2d and kl_div_loss emitted no DaCe program because
#: of one precondition line apiece.
INLINABLE_STMTS = (ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Expr, ast.While, ast.Assert, ast.Pass)


def constant_truth(test: ast.expr) -> bool | None:
    """``test``'s value when every leaf is a literal, else ``None``.

    Narrow on purpose: literals, ``and``/``or`` over them, ``not``, and a comparison of two
    literals. That is the shape SPECIALISATION leaves behind -- a guard on pinned scalars becomes
    ``True and True and True and True`` -- and nothing wider is needed to recognize it.
    """
    if isinstance(test, ast.Constant):
        return bool(test.value)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = constant_truth(test.operand)
        return None if inner is None else not inner
    if isinstance(test, ast.BoolOp):
        values = [constant_truth(v) for v in test.values]
        if any(v is None for v in values):
            return None
        return all(values) if isinstance(test.op, ast.And) else any(values)
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        try:
            left, right = ast.literal_eval(test.left), ast.literal_eval(test.comparators[0])
        except (ValueError, TypeError, SyntaxError):
            return None
        op = test.ops[0]
        for kind, answer in (
            (ast.Eq, left == right),
            (ast.NotEq, left != right),
            (ast.Lt, left < right),
            (ast.LtE, left <= right),
            (ast.Gt, left > right),
            (ast.GtE, left >= right),
        ):
            if isinstance(op, kind):
                return bool(answer)
    return None


def fold_constant_branches(fn: ast.FunctionDef) -> bool:
    """Replace every ``if`` in ``fn`` whose test is a compile-time constant with the taken branch.

    Specialisation is what makes this pay: cloning a helper per call signature pins its scalar
    arguments, so ``_conv2d``'s ``if kh == 1 and kw == 1 and stride == 1 and padding == 0`` reads
    ``if False and False and False and True`` in the stride-2 clone. Both arms return, and they
    return DIFFERENT extents -- which retires the array-return classification for a helper where
    only one arm is reachable.
    """
    changed = False

    def walk(body: list[ast.stmt]) -> list[ast.stmt]:
        nonlocal changed
        out: list[ast.stmt] = []
        for stmt in body:
            if isinstance(stmt, (ast.If, ast.For, ast.While)):
                stmt.body, stmt.orelse = walk(stmt.body), walk(stmt.orelse)
            elif isinstance(stmt, ast.Try):
                stmt.body, stmt.orelse = walk(stmt.body), walk(stmt.orelse)
                stmt.finalbody = walk(stmt.finalbody)
            if isinstance(stmt, ast.If):
                taken = constant_truth(stmt.test)
                if taken is not None:
                    changed = True
                    out.extend(stmt.body if taken else stmt.orelse)
                    continue
            out.append(stmt)
        return out

    fn.body = walk(fn.body)
    if changed:
        ast.fix_missing_locations(fn)
    return changed


def collect_inlinable_helpers(tree: ast.Module, kernel_fn: ast.FunctionDef) -> dict[str, ast.FunctionDef]:
    """Return a name -> FunctionDef map for every top-level helper
    eligible for inlining.

    Forms recognised:

    * Single ``return expr``.
    * ``if cond: return a; else: return b`` -> IfExp.
    * Multi-statement body ending with ``return expr``: a sequence of
      simple Assign / AugAssign / For / If statements followed by a
      ``return``. Inlined as a statement block whose final value is
      assigned to the call's target.
    """
    out: dict[str, ast.FunctionDef] = {}

    def classify(node: ast.FunctionDef) -> bool:
        body = strip_docstrings_(node.body)
        if not body:
            return False
        # Form 1: single ``return expr``.
        if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
            return True
        # Form 2: ``if cond: return a; else: return b``.
        if (
            len(body) == 1
            and isinstance(body[0], ast.If)
            and len(body[0].body) == 1
            and isinstance(body[0].body[0], ast.Return)
            and len(body[0].orelse) == 1
            and isinstance(body[0].orelse[0], ast.Return)
        ):
            return True
        # Form 3: multi-statement body ending with ``return expr``. No
        # early returns / yields / nested defs allowed. ``Expr`` statements are
        # allowed (side-effect void calls -- lulesh ``_integrate_stress`` runs
        # ``np.add.at(fx, nodelist, sfx)`` scatters then ``return determ``).
        if isinstance(body[-1], ast.Return) and body[-1].value is not None:
            mid = body[:-1]
            if all(isinstance(s, INLINABLE_STMTS) for s in mid):
                if not any(isinstance(sub, ast.Return) for s in mid for sub in ast.walk(s)):
                    return True
        # Form 4: void helper -- simple Assign / AugAssign / For / While / If / Expr
        # statements with NO Return (in-place writes to argument arrays).
        if all(isinstance(s, INLINABLE_STMTS) for s in body):
            if not any(isinstance(sub, ast.Return) for s in body for sub in ast.walk(s)):
                return True
        return False

    # Top-level helpers defined ABOVE the kernel...
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node is not kernel_fn and classify(node):
            out[node.name] = node
    # ...AND helpers defined NESTED inside the kernel body (ICON
    # velocity_tendencies' ``def gat(A, idx, blk, n, jk): return A[...]`` gather
    # shorthand). These are stripped from the body after their calls are inlined
    # (see _InlineHelpers.visit_FunctionDef) -- a backend can't emit a Python
    # ``def``, so the only correct lowering is full inlining.
    for node in ast.walk(kernel_fn):
        if isinstance(node, ast.FunctionDef) and node is not kernel_fn and classify(node):
            out[node.name] = node
    return out


def static_flag_params(tree: ast.Module) -> dict[str, frozenset[str]]:
    """Per helper, the parameters bound to a compile-time literal at EVERY call site.

    Such a parameter is a configuration flag, not data: after inlining substitutes the literal, a
    branch on it is statically decidable and folds away. A parameter that any call site binds to an
    expression is excluded -- there its value is only known at run time.
    """
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    literal: dict[str, dict[str, bool]] = {name: {} for name in defs}
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        if not (isinstance(call.func, ast.Name) and call.func.id in defs):
            continue
        hdef = defs[call.func.id]
        pnames = [a.arg for a in hdef.args.args]
        # An unsupplied trailing parameter takes its default, which is itself a compile-time value
        # when the default is a literal -- ``_logsumexp(x, axis=1)`` leaves keepdims=False.
        bound: dict[str, ast.expr | None] = {}
        offset = len(pnames) - len(hdef.args.defaults)
        for i, default in enumerate(hdef.args.defaults):
            bound[pnames[offset + i]] = default
        for i, arg in enumerate(call.args):
            if i < len(pnames):
                bound[pnames[i]] = arg
        for kw in call.keywords:
            if kw.arg is not None:
                bound[kw.arg] = kw.value
        for pname in pnames:
            value = bound.get(pname)
            is_literal = isinstance(value, ast.Constant)
            literal[call.func.id][pname] = literal[call.func.id].get(pname, True) and is_literal
    return {name: frozenset(p for p, ok in flags.items() if ok) for name, flags in literal.items()}


def fuse_guarded_returns(tree: ast.Module) -> None:
    """``if FLAG: return A`` immediately before a trailing ``return B`` -> ``return A if FLAG else B``.

    What it buys is inlinability: an early return anywhere but the last statement disqualifies a
    helper, so the KernelBench ``_logsumexp(x, axis, keepdims)`` was emitted as its OWN kernel --
    whose ``axis`` and ``keepdims`` are not declared parameters. Once fused it inlines, and the call
    site's literal ``keepdims=False`` folds the branch away entirely.

    That fold is a PRECONDITION, not a bonus, so the guard must be a parameter every call site binds
    to a literal (:func:`static_flag_params`). A runtime guard would leave the IfExp standing, and
    an IfExp over ARRAY branches has no target form: C's ``?:`` rejects the operand types outright,
    and Fortran's ``merge`` is rank-strict (and evaluates BOTH branches, so a guarded division or
    subscript would run on the values the guard exists to exclude).
    """
    flags_by_helper = static_flag_params(tree)
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        flags = flags_by_helper.get(fn.name, frozenset())
        body = fn.body
        while True:
            lift_pure_assignment_over_guard(body, flags)
            if not (
                len(body) >= 2
                and isinstance(body[-1], ast.Return)
                and body[-1].value is not None
                and isinstance(body[-2], ast.If)
                and not body[-2].orelse
                and len(body[-2].body) == 1
                and isinstance(body[-2].body[0], ast.Return)
                and body[-2].body[0].value is not None
                and is_static_flag_test(body[-2].test, flags)
            ):
                break
            guard = body[-2]
            fused = ast.Return(value=ast.IfExp(test=guard.test, body=guard.body[0].value, orelse=body[-1].value))
            body[-2:] = [ast.copy_location(fused, guard)]
        ast.fix_missing_locations(fn)


def is_pure_expression(node: ast.expr) -> bool:
    """``True`` when evaluating ``node`` cannot do anything but produce a value.

    A call is the whole exclusion: it may write an argument array in place, and the corpus' helpers
    do exactly that. A walrus binds a second name, which a move would rebind on a path that never
    bound it. What is left is arithmetic over names, constants, shape attributes and subscripts,
    which is what a shape or index expression is made of.
    """
    impure = (ast.Call, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)
    return not any(isinstance(sub, impure) for sub in ast.walk(node))


def lift_pure_assignment_over_guard(body: list[ast.stmt], flags: frozenset[str]) -> None:
    """Move a pure ``name = expr`` that sits BETWEEN a static-flag guard and the trailing return up
    above the guard, in place, so the two returns become adjacent and :func:`fuse_guarded_returns`
    can see them.

    ``_instance_norm``'s affine branch reads ``shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)``, bound
    after the ``if weight is None: return y`` guard. The fuse only ever looked at the last two
    statements, so the guard stayed an early return, the helper inlined under no form, and
    conv2d_instance_norm_divide emitted no DaCe program at all.

    Lifting is only sound because the moved statement is pure and independent: it runs on a path
    where it did not run before, so a call (which could write an argument in place) is refused, and
    a binding the guard itself READS is refused because moving it would change which value the
    guard sees.
    """
    while len(body) >= 3:
        assign, guard = body[-2], body[-3]
        if not (
            isinstance(body[-1], ast.Return)
            and isinstance(assign, ast.Assign)
            and len(assign.targets) == 1
            and isinstance(assign.targets[0], ast.Name)
            and is_pure_expression(assign.value)
        ):
            return
        if not (
            isinstance(guard, ast.If)
            and not guard.orelse
            and len(guard.body) == 1
            and isinstance(guard.body[0], ast.Return)
            and is_static_flag_test(guard.test, flags)
        ):
            return
        target = assign.targets[0].id
        if any(isinstance(sub, ast.Name) and sub.id == target for sub in ast.walk(guard)):
            return  # the guard reads it, so the value it sees would change
        body[-3:] = [assign, guard, body[-1]]


def is_static_flag_test(test: ast.expr, flags: frozenset[str]) -> bool:
    """``flag`` / ``not flag`` / ``flag == <literal>`` / ``flag is <literal>`` on a parameter that is
    a literal at every call site.

    The compare form is how a multi-way MODE selects, and it is decidable on exactly the same
    grounds as the bare flag: the call site's literal is substituted into the body, leaving two
    constants the inline fixpoint's own folding decides. kl_div_loss' ``_kl_div`` guards its three
    return paths with ``reduction == 'batchmean'`` / ``== 'sum'``, and without this it fused
    nothing, inlined under no form, and emitted no program at all.

    ``is``/``is not`` is the same test spelled the way an OPTIONAL argument is checked -- torch's
    affine-less norms pass ``weight=None``, and ``if weight is None: return y`` is how the helper
    says so. Identity and equality agree here because the operands are a parameter and a literal.
    """
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return is_static_flag_test(test.operand, flags)
    ops = (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ops):
        left, right = test.left, test.comparators[0]
        named = [side for side in (left, right) if isinstance(side, ast.Name) and side.id in flags]
        return bool(named) and any(isinstance(side, ast.Constant) for side in (left, right))
    return isinstance(test, ast.Name) and test.id in flags


def flatten_nested_helpers(tree: ast.Module) -> None:
    """Inline helpers NESTED inside other top-level helpers, in place.

    lulesh's compute helpers carry a one-line column shorthand ``def c(a, i):
    return a[:, i]`` and call it. That nested ``def`` makes the OUTER helper
    un-inlinable (a FunctionDef isn't an allowed mid statement in
    :func:`collect_inlinable_helpers`) and it's never exposed to the
    kernel-level fixpoint, since the parent never inlines -- a deadlock.
    Inlining the nested defs into their parent (then dropping them) leaves
    each outer helper nested-def-free. Iterated for helpers nested more than
    one level deep."""
    for unused in range(16):
        changed = False
        for h in list(tree.body):
            if not isinstance(h, ast.FunctionDef):
                continue
            if not any(isinstance(n, ast.FunctionDef) for n in h.body):
                continue
            inl = collect_inlinable_helpers(tree, h)  # top-level + nested-in-h
            if not inl:
                continue
            HoistMultiStmtHelpers(inl).visit(h)
            InlineHelpers(inl).visit(h)
            ast.fix_missing_locations(h)
            changed = True
        if not changed:
            break


def is_const_list_literal(node: ast.AST) -> bool:
    """A non-empty list/tuple literal usable as a compile-time-unrollable loop
    iterable: lulesh's ``faces = [(0,1,2,3), (0,4,5,1), ...]`` AND the inlined
    ``for nk in (n0, n1, n2, n3)``. Elements may be constants, names, or nested
    sequences -- the loop body is cloned once per element with the loop variable
    substituted, so any element expression is fine."""
    return isinstance(node, (ast.List, ast.Tuple)) and bool(node.elts)


class LoopVarSubst(ast.NodeTransformer):
    """Substitute a (now compile-time-known) loop variable with one list element.

    Handles a Tuple target (``for (a, b, d, e) in faces`` -> a/b/d/e bound to the
    element's components) and a single Name target (``for f in faces`` -> ``*f`` in
    a call expanded to the element's components, and bare ``f`` replaced by it)."""

    def __init__(self, target: ast.AST, elt: ast.AST) -> None:
        self.elt = elt
        self.map: dict[str, ast.AST] = {}
        if (
            isinstance(target, ast.Tuple)
            and isinstance(elt, (ast.Tuple, ast.List))
            and len(target.elts) == len(elt.elts)
        ):
            for t, v in zip(target.elts, elt.elts):
                if isinstance(t, ast.Name):
                    self.map[t.id] = v
        self.single = target.id if isinstance(target, ast.Name) else None

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        # After substitution ``*f`` has become ``*(c0, c1, ...)`` (a Starred over a
        # literal tuple/list) -- splat it into the call's positional args.
        if any(isinstance(a, ast.Starred) and isinstance(a.value, (ast.Tuple, ast.List)) for a in node.args):
            new_args: list[ast.expr] = []
            for a in node.args:
                if isinstance(a, ast.Starred) and isinstance(a.value, (ast.Tuple, ast.List)):
                    new_args.extend(copy.deepcopy(e) for e in a.value.elts)
                else:
                    new_args.append(a)
            node.args = new_args
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load):
            if node.id in self.map:
                return copy.deepcopy(self.map[node.id])
            if self.single is not None and node.id == self.single:
                return copy.deepcopy(self.elt)
        return node


def single_list_bindings(fn: ast.FunctionDef) -> dict[str, list[ast.expr]]:
    """Locals bound exactly once, to a list or tuple literal: ``name -> elements``."""
    binds_count: dict[str, int] = {}
    for s in ast.walk(fn):
        if isinstance(s, ast.Assign):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    binds_count[t.id] = binds_count.get(t.id, 0) + 1
    list_binds: dict[str, list[ast.expr]] = {}
    for s in ast.walk(fn):
        if (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
            and is_const_list_literal(s.value)
            and binds_count.get(s.targets[0].id) == 1
        ):
            list_binds[s.targets[0].id] = s.value.elts
    return list_binds


class ConstListLoopUnroller(ast.NodeTransformer):
    """Clone a ``for`` body once per element of its literal (or once-bound literal) iterable."""

    def __init__(self, list_binds: dict[str, list[ast.expr]]) -> None:
        self.list_binds = list_binds
        #: Bound list names whose loops were unrolled; their bindings are dead afterwards.
        self.consumed: set[str] = set()

    def visit_For(self, node: ast.For) -> ast.stmt | list[ast.stmt]:
        self.generic_visit(node)
        if node.orelse or has_loop_control(node.body):
            return node
        seq: list[ast.expr] | None = None
        src: str | None = None
        if is_const_list_literal(node.iter) and isinstance(node.iter, (ast.List, ast.Tuple)):
            seq = list(node.iter.elts)
        elif isinstance(node.iter, ast.Name) and node.iter.id in self.list_binds:
            seq = self.list_binds[node.iter.id]
            src = node.iter.id
        if seq is None:
            return node
        out: list[ast.stmt] = []
        for elt in seq:
            for st in node.body:
                cloned = ast.parse(ast.unparse(st)).body[0]
                cloned = LoopVarSubst(node.target, elt).visit(cloned)
                ast.fix_missing_locations(cloned)
                out.append(cloned)
        if src is not None:
            self.consumed.add(src)
        return out


class DropListBindings(ast.NodeTransformer):
    def __init__(self, names: set[str]) -> None:
        self.names = names

    def visit_Assign(self, node: ast.Assign) -> ast.stmt | None:
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in self.names
            and is_const_list_literal(node.value)
        ):
            return None
        return node


def unroll_const_list_loops(fn: ast.FunctionDef) -> None:
    """Unroll ``for x in <literal list>`` (or a local bound once to one): no backend iterates a
    Python list. The consumed binding is dropped. A body with its own ``break``/``continue`` is left
    alone -- cloning would rebind those to the enclosing loop -- and emit rejects its literal."""
    unroller = ConstListLoopUnroller(single_list_bindings(fn))
    unroller.visit(fn)
    if unroller.consumed:
        DropListBindings(unroller.consumed).visit(fn)
    ast.fix_missing_locations(fn)


class HoistMultiStmtHelpers(ast.NodeTransformer):
    """Lift Form-3 helper Calls out of expression contexts so the
    multi-statement inliner can consume them via Assign-level visits.

    A Form-3 helper is a multi-statement body ending in ``return expr``.
    Single-return/void helpers aren't hoisted -- those are already
    substituted at expression/statement level by ``InlineHelpers``.

    Operates per-statement: each top-level statement is rewritten in place;
    helper Calls inside non-Assign-of-Call expressions are replaced by fresh
    ``__hcall<n>`` temps, with their Assigns prepended.
    """

    def __init__(self, helpers: dict[str, ast.FunctionDef], counter: list[int] | None = None) -> None:
        self.helpers = helpers
        self.multi_stmt = {name: fn for name, fn in helpers.items() if is_multi_stmt_return_form(fn)}
        # Shared across the inline fixpoint -- see _InlineHelpers re: prefix reuse.
        self._counter = counter if counter is not None else [0]
        self._pending: list[ast.stmt] = []
        #: Names the tree ALREADY binds, so a fresh temp never lands on one. Three call sites
        #: build this transformer and two of them start their own counter, so both series minted
        #: ``__hcall1``, ``__hcall2``, ... over the same body: resnet101 ended up with
        #: ``__hcall4`` bound once to a 256-channel convolution and once to a 64-channel
        #: batch-norm. The shape table holds one entry per name, so every extent read off either
        #: was the other's -- and, downstream, the whole layer went unresolved.
        self._taken: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._taken |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        node.body = self.rewrite_stmt_list(node.body)
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        node.body = self.rewrite_stmt_list(node.body)
        node.orelse = self.rewrite_stmt_list(node.orelse)
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        node.body = self.rewrite_stmt_list(node.body)
        node.orelse = self.rewrite_stmt_list(node.orelse)
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        node.test = self.rewrite_expr(node.test)
        # The test's hoisted ``__hcall<n> = helper(..)`` Assigns are queued in
        # ``self._pending`` for the CALLER's _rewrite_stmt_list to place BEFORE this
        # If. Rewriting the branches would otherwise drain that queue into the
        # if-BODY (_rewrite_stmt_list unconditionally flushes _pending per
        # statement) -- the temp would then be assigned inside the branch its own
        # test reads, a use-before-def (distribution_search's line-search
        # ``if max(abs(residual(trial))) < cur:``). Park it across the branches.
        pending, self._pending = self._pending, []
        node.body = self.rewrite_stmt_list(node.body)
        node.orelse = self.rewrite_stmt_list(node.orelse)
        self._pending = pending
        return node

    def rewrite_stmt_list(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in stmts:
            # Skip the "Assign of a direct helper Call" form -- the
            # multi-statement inliner already handles those. We only
            # want to hoist NESTED helper Calls.
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id in self.multi_stmt
            ):
                out.append(stmt)
                continue
            # Same skip for a helper called as a bare STATEMENT: its value is discarded, so
            # hoisting it to ``__hcall<n> = helper(..)`` invents a consumer that does not exist.
            # The temp then dead-stores away and its remaining ``Expr(__hcall<n>)`` folds back to
            # the helper's return expression -- a stranded ``(ux, uy, uz)`` no backend can render.
            # ``InlineHelpers.visit_Expr`` splices this form and drops the dead return instead.
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id in self.multi_stmt
            ):
                out.append(stmt)
                continue
            # Recurse into nested control flow first.
            stmt = self.visit(stmt)
            if isinstance(stmt, ast.Assign):
                stmt.value = self.rewrite_expr(stmt.value)
            elif isinstance(stmt, ast.AugAssign):
                stmt.value = self.rewrite_expr(stmt.value)
            elif isinstance(stmt, ast.Expr):
                stmt.value = self.rewrite_expr(stmt.value)
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                stmt.value = self.rewrite_expr(stmt.value)
            out.extend(self._pending)
            self._pending = []
            out.append(stmt)
        return out

    def rewrite_expr(self, expr: ast.expr) -> ast.expr:
        """Walk ``expr``; replace every multi-stmt helper Call with a
        fresh ``__hcall<n>`` Name and queue an Assign in
        ``self._pending``."""

        class Replacer(ast.NodeTransformer):
            outer = self

            def visit_Call(self_inner, call: ast.Call) -> ast.AST:
                # Recurse into args / kwargs first.
                self_inner.generic_visit(call)
                if isinstance(call.func, ast.Name) and call.func.id in self.multi_stmt:
                    self._counter[0] += 1
                    temp = f"__hcall{self._counter[0]}"
                    while temp in self._taken:
                        self._counter[0] += 1
                        temp = f"__hcall{self._counter[0]}"
                    self._taken.add(temp)
                    self._pending.append(ast.Assign(targets=[ast.Name(id=temp, ctx=ast.Store())], value=call))
                    return ast.Name(id=temp, ctx=ast.Load())
                return call

        return Replacer().visit(expr)


def is_multi_stmt_return_form(fn: ast.FunctionDef) -> bool:
    """``True`` for Form-3 helpers (multi-statement body ending with
    ``return expr``)."""
    body = strip_docstrings_(fn.body)
    if len(body) <= 1:
        return False
    last = body[-1]
    return isinstance(last, ast.Return) and last.value is not None


class InlineHelpers(ast.NodeTransformer):
    """Substitute calls to recognised helpers with their inline body.

    Three forms:

    * Single ``return expr`` -> replace the call expression by ``expr``
      with parameter Names substituted.
    * ``if cond: return a; else: return b`` -> IfExp.
    * Multi-statement body ending with ``return expr`` -> replace the
      enclosing ``Assign / Return`` statement with the helper body
      (parameters renamed, locals prefixed to avoid collisions) plus
      one ``Assign`` of the call-site target to the helper's return
      expression. Statement-level inlining is handled at the
      Assign-level visit; expression-level inlining for the single-
      return forms remains in visit_Call.
    """

    def __init__(self, helpers: dict[str, ast.FunctionDef], counter: list[int] | None = None) -> None:
        self.helpers = helpers
        # The ``__inl<N>_`` prefix counter MUST persist across the parse_kernel
        # inline fixpoint: a nested helper exposed in a later iteration would
        # otherwise reuse a prefix an outer helper already took in an earlier
        # one (lulesh ``_integrate_stress``'s local ``b`` colliding with the
        # nested ``_calc_shape_fn_derivatives``'s ``b`` -- both becoming
        # ``__inl1_b``, crossing their shapes).
        self._counter = counter if counter is not None else [0]

    def visit_Assign(self, node: ast.Assign) -> ast.stmt | list[ast.stmt]:
        self.generic_visit(node)
        if (
            len(node.targets) == 1
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in self.helpers
        ):
            helper = self.helpers[node.value.func.id]
            body = strip_docstrings_(helper.body)
            # Multi-statement form -- mid statements followed by Return.
            if len(body) > 1 and isinstance(body[-1], ast.Return) and body[-1].value is not None:
                param_names = [a.arg for a in helper.args.args]
                call_args = resolve_call_args(node.value, helper)
                if call_args is None:
                    return node
                node.value.args = call_args
                node.value.keywords = []
                self._counter[0] += 1
                prefix = f"__inl{self._counter[0]}_"
                # Map params to call args; locals (assigned in body) get
                # the prefix so multiple inlines don't collide.
                local_names = collect_assigned_names(body[:-1])
                arg_map = dict(zip(param_names, node.value.args))
                rename: dict[str, ast.AST] = dict(arg_map)
                # A parameter REASSIGNED in the body (lulesh _phi's ``delvm =
                # delvm * normd``) becomes a fresh prefixed local, initialised
                # from the call argument first -- otherwise its first read is
                # uninitialised heap garbage (native backends only; numba/cupy
                # use a real Python var). Value semantics: a fresh copy, so the
                # caller's argument array is never mutated by the rebind.
                reassigned_params: list[str] = []
                for ln in local_names:
                    rename[ln] = ast.Name(id=f"{prefix}{ln}", ctx=ast.Load())
                    if ln in arg_map:
                        reassigned_params.append(ln)
                # Substitute throughout the helper body and the return
                # expression.
                renamer = SubstNames(rename)
                new_body: list[ast.stmt] = []
                for pn_ in reassigned_params:
                    init_ = ast.Assign(
                        targets=[ast.Name(id=f"{prefix}{pn_}", ctx=ast.Store())],
                        value=ast.parse(ast.unparse(arg_map[pn_]), mode="eval").body,
                    )
                    ast.fix_missing_locations(init_)
                    new_body.append(init_)
                for stmt in body[:-1]:
                    cloned = ast.parse(ast.unparse(stmt)).body[0]
                    cloned = renamer.visit(cloned)
                    ast.fix_missing_locations(cloned)
                    new_body.append(cloned)
                ret_expr = ast.parse(ast.unparse(body[-1].value), mode="eval").body
                ret_expr = renamer.visit(ret_expr)
                ast.fix_missing_locations(ret_expr)
                tgt = node.targets[0]
                # A tuple-target multi-output helper (lulesh ``b, detJ =
                # _calc_shape_fn_derivatives(..)`` whose body ends ``return b,
                # volume``) must be DESTRUCTURED into per-element assigns -- a
                # backend has no runtime tuple, so ``(b, detJ) = (x, y)`` would
                # reach emit as an unlowerable Tuple. ``_`` elements are discarded.
                if (
                    isinstance(tgt, ast.Tuple)
                    and isinstance(ret_expr, ast.Tuple)
                    and len(tgt.elts) == len(ret_expr.elts)
                ):
                    for t_elt, v_elt in zip(tgt.elts, ret_expr.elts):
                        if isinstance(t_elt, ast.Name) and t_elt.id == "_":
                            continue
                        a = ast.Assign(targets=[t_elt], value=v_elt)
                        ast.fix_missing_locations(a)
                        new_body.append(a)
                else:
                    new_body.append(ast.Assign(targets=[tgt], value=ret_expr))
                return new_body
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.stmt | list[ast.stmt] | None:
        # Void helper call as a statement -- ``helper(arr, ...)`` with
        # no return value. Inline the helper body (parameters renamed)
        # in place of the call statement.
        self.generic_visit(node)
        if not (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in self.helpers
        ):
            return node
        helper = self.helpers[node.value.func.id]
        body = strip_docstrings_(helper.body)
        # A helper called as a bare STATEMENT has its value discarded: the helper mutates its array
        # parameters and the caller ignores what it hands back (WarpX's Boris pusher returns the
        # three momentum arrays it just updated in place). So a trailing ``return`` is dead HERE,
        # whatever it means at an Assign call site. Declining left the tail expression stranded as
        # a bare ``(ux, uy, uz)`` statement, which no backend can render. An early return elsewhere
        # in the body IS control flow, so that form still declines.
        if body and isinstance(body[-1], ast.Return):
            if any(isinstance(n, ast.Return) for s in body[:-1] for n in ast.walk(s)):
                return node
            body = body[:-1]
            if not body:
                return node
        param_names = [a.arg for a in helper.args.args]
        call_args = resolve_call_args(node.value, helper)
        if call_args is None:
            return node
        node.value.args = call_args
        node.value.keywords = []
        self._counter[0] += 1
        prefix = f"__inl{self._counter[0]}_"
        local_names = collect_assigned_names(body)
        rename: dict[str, ast.AST] = dict(zip(param_names, node.value.args))
        for ln in local_names:
            if ln in param_names:
                # The helper rebinds a parameter (e.g. ``pn = p.copy()``
                # then later uses of ``pn``). Don't rename it, but
                # tracking it in ``rename`` would shadow the call-site
                # arg -- which is what we want for ``pn`` to remain a
                # distinct local through the inlined body.
                continue
            rename[ln] = ast.Name(id=f"{prefix}{ln}", ctx=ast.Load())
        renamer = SubstNames(rename)
        new_body: list[ast.stmt] = []
        for stmt in body:
            cloned = ast.parse(ast.unparse(stmt)).body[0]
            cloned = renamer.visit(cloned)
            ast.fix_missing_locations(cloned)
            new_body.append(cloned)
        return new_body

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Recurse so calls inside the def (and the kernel body) are inlined,
        # then DROP any nested helper def whose calls we just inlined -- a
        # backend cannot emit a Python ``def``. The kernel itself is never in
        # ``helpers`` so it is preserved.
        self.generic_visit(node)
        if node.name in self.helpers:
            return None
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not (isinstance(node.func, ast.Name) and node.func.id in self.helpers):
            return node
        helper = self.helpers[node.func.id]
        param_names = [a.arg for a in helper.args.args]
        call_args = resolve_call_args(node, helper)
        if call_args is None:
            return node
        node.args = call_args
        node.keywords = []
        subst = dict(zip(param_names, node.args))
        body_stmts = strip_docstrings_(helper.body)
        if len(body_stmts) == 1 and isinstance(body_stmts[0], ast.Return):
            return SubstNames(subst).visit(
                ast.fix_missing_locations(ast.parse(ast.unparse(body_stmts[0].value), mode="eval").body)
            )
        if (
            len(body_stmts) == 1
            and isinstance(body_stmts[0], ast.If)
            and len(body_stmts[0].body) == 1
            and len(body_stmts[0].orelse) == 1
        ):
            cond = ast.parse(ast.unparse(body_stmts[0].test), mode="eval").body
            then = ast.parse(ast.unparse(body_stmts[0].body[0].value), mode="eval").body
            else_ = ast.parse(ast.unparse(body_stmts[0].orelse[0].value), mode="eval").body
            ifexp = ast.IfExp(test=cond, body=then, orelse=else_)
            return SubstNames(subst).visit(ast.fix_missing_locations(ifexp))
        return node


def collect_assigned_names(stmts: list[ast.stmt]) -> OrderedSet[str]:
    """Return the set of Name targets assigned in any of ``stmts``,
    recursing into For / If bodies.

    A TUPLE/LIST target contributes every Name it binds (a for-loop over
    ``enumerate``/``zip``, or an unpacking assign). Missing these lets an
    inlined helper's loop index escape the ``__inl<k>_`` rename and clobber a
    caller symbol of the same name -- chebyshev_filter_subspace's ``_hpsi``
    stencil loop var ``m`` vs the kernel's Chebyshev-degree ``m`` (the
    inlined loop overwrote ``m`` to len(_CW), truncating the degree loop)."""
    # Ordered: a helper parameter that the body REASSIGNS is initialised from its call
    # argument in the order this walk found it (_InlineHelpers below), so hash order here
    # would shuffle the emitted prologue -- conv2d_relu_bias_add's stride/padding/dilation.
    out: OrderedSet[str] = OrderedSet()

    def bind(target: ast.expr | None) -> None:
        if isinstance(target, ast.Name):
            out.add(target.id)
        elif isinstance(target, ast.Starred):
            bind(target.value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                bind(elt)

    for s in stmts:
        for sub in ast.walk(s):
            if isinstance(sub, ast.Assign):
                for t in sub.targets:
                    bind(t)
            elif isinstance(sub, ast.AugAssign):
                bind(sub.target)
            elif isinstance(sub, ast.For):
                bind(sub.target)
    return out


class SubstNames(ast.NodeTransformer):
    """Replace ``Name(p)`` references with the call-site expression /
    renamed local. Load-context substitution deep-copies the AST so
    multiple substitutions don't share state; Store-context only
    renames when the substitution target is itself a Name (so local
    renames work but a param-arg replacement on a Store context is
    silently rejected to keep AST validity)."""

    def __init__(self, subst: dict[str, ast.AST]) -> None:
        self.subst = subst

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id not in self.subst:
            return node
        repl = self.subst[node.id]
        if isinstance(node.ctx, ast.Load):
            return ast.fix_missing_locations(ast.parse(ast.unparse(repl), mode="eval").body)
        # Store / Del context: only rename if the replacement is a
        # bare Name -- that's the per-helper local-rename case.
        if isinstance(repl, ast.Name):
            return ast.Name(id=repl.id, ctx=node.ctx)
        return node
