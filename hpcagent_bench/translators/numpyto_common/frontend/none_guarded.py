"""Helpers returning ``None`` or a tuple, spliced into their callers with the caller's guard."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.frontend.inlining import (
    SubstNames,
    collect_assigned_names,
    resolve_call_args,
    strip_docstrings_,
)
from hpcagent_bench.translators.numpyto_common.frontend.none_folding import none_toggle_op


def is_none_sentinel(value: ast.expr | None) -> bool:
    """``True`` for a literal ``None``, or a non-empty tuple/list whose elements are all ``None``
    (``_transpose_taps``'s ``return None, None`` spelling of the same "no valid range" signal)."""
    if isinstance(value, ast.Constant):
        return value.value is None
    if isinstance(value, (ast.Tuple, ast.List)):
        return bool(value.elts) and all(is_none_sentinel(e) for e in value.elts)
    return False


def find_none_guard(mid: list[ast.stmt]) -> int | None:
    """Index of the ONE ``if <cond>: return <None sentinel>`` (no ``elif``/``else``) in ``mid``, or
    ``None`` when there is not exactly one such guard."""
    hits = [
        i
        for i, s in enumerate(mid)
        if (
            isinstance(s, ast.If)
            and not s.orelse
            and len(s.body) == 1
            and isinstance(s.body[0], ast.Return)
            and is_none_sentinel(s.body[0].value)
        )
    ]
    return hits[0] if len(hits) == 1 else None


def collect_none_guarded_helpers(tree: ast.Module, kernel_fn: ast.FunctionDef) -> dict[str, ast.FunctionDef]:
    """Top-level helpers shaped like ``_tap_range``/``_transpose_taps``: ordinary computation, ONE
    ``if <empty range>: return None`` (or ``return None, None, ...``) early exit, more computation,
    then a final ``return <tuple>``.

    :func:`collect_inlinable_helpers`'s Form 3 refuses ANY early return outright -- a tuple-or-None
    result has no C/Fortran ABI to inline INTO as a value. This is the one early-return shape
    :class:`SpliceNoneGuardedCalls` can still splice: every caller in the corpus responds to "no
    valid range" with a plain control statement (``continue``), never a further use of the sentinel
    as a value, so the ``None`` never needs an ABI of its own.
    """
    out: dict[str, ast.FunctionDef] = {}

    def classify(node: ast.FunctionDef) -> bool:
        body = strip_docstrings_(node.body)
        if len(body) < 2 or not (isinstance(body[-1], ast.Return) and body[-1].value is not None):
            return False
        mid = body[:-1]
        guard_idx = find_none_guard(mid)
        if guard_idx is None:
            return False
        rest = mid[:guard_idx] + mid[guard_idx + 1 :]
        if not all(isinstance(s, (ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Expr, ast.While)) for s in rest):
            return False
        return not any(isinstance(sub, ast.Return) for s in rest for sub in ast.walk(s))

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node is not kernel_fn and classify(node):
            out[node.name] = node
    return out


class SpliceNoneGuardedCalls:
    """Inline one call to a :func:`collect_none_guarded_helpers` helper TOGETHER with the caller's
    own "is None" guard and tuple unpack, so no intermediate ``None``-or-tuple value ever exists for
    an emitter to choke on. Handles both caller spellings seen in the corpus:

    * ``X = H(...)``; ``if X is None: continue``; ``a, b, ... = X`` (``_tap_range``: the helper's
      result is bound to a plain name first, unpacked in a later statement).
    * ``a, b = H(...)``; ``if a is None: continue`` (``_transpose_taps``: Python destructures the
      return tuple directly at the call, so the ``is None`` check names one of the unpacked
      elements instead of the call's own target).

    Reuses the SAME renaming primitives :class:`InlineHelpers` uses (:func:`resolve_call_args`,
    :class:`SubstNames`, :func:`collect_assigned_names`, the ``__inl<k>_`` prefix for a
    reassigned param) rather than a second renaming scheme; only the STATEMENT SHAPE spliced in
    differs (an early-return guard becomes ``if <cond>: continue`` and the tail ``return`` becomes a
    direct assignment to the caller's own unpack targets, both inline in the caller's block, with no
    helper function and no intermediate name surviving at all).
    """

    def __init__(self, helpers: dict[str, ast.FunctionDef], counter: list[int]) -> None:
        self.helpers = helpers
        self._counter = counter
        #: The function :meth:`apply` is walking -- the scope a deferred unpack is searched in.
        self.fn_: ast.FunctionDef | None = None

    def apply(self, fn: ast.FunctionDef) -> bool:
        self.fn_ = fn
        return self.rewrite_block(fn.body)

    def ordered_stmts(self) -> list[ast.stmt]:
        """Every statement of the enclosing function in SOURCE order (a plain ``ast.walk`` is
        breadth-first, which cannot answer "does the unpack run after the call")."""
        out: list[ast.stmt] = []

        def walk(block: list[ast.stmt]) -> None:
            for st in block:
                out.append(st)
                for field in ("body", "orelse", "finalbody"):
                    nested = vars(st).get(field)
                    if isinstance(nested, list):
                        walk(nested)

        walk(self.fn_.body)
        return out

    def deferred_unpack(self, call_stmt: ast.Assign, name: str) -> ast.Assign | None:
        """The one ``a, b, c = <name>`` that consumes this call's result from DEEPER in the nest,
        or ``None``.

        ``_conv_transpose3d`` binds ``rz`` in the ``kz`` loop and unpacks it two loops down, once
        every tap is known to be in range, so the adjacent-unpack spelling above never matches and
        the helper stayed a tuple-returning function with no ABI. Splicing is sound here for the
        same reason it is there -- the unpack is rewritten IN PLACE (nothing moves across the
        loops), and the requirements below make ``name`` a single-assignment value whose only
        readers are the guard and that unpack."""
        stmts = self.ordered_stmts()
        if call_stmt not in stmts:
            return None
        writes = [
            st
            for st in stmts
            if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in st.targets)
        ]
        if writes != [call_stmt]:
            return None
        unpacks = [
            st
            for st in stmts
            if (
                isinstance(st, ast.Assign)
                and len(st.targets) == 1
                and isinstance(st.targets[0], (ast.Tuple, ast.List))
                and all(isinstance(e, ast.Name) for e in st.targets[0].elts)
                and isinstance(st.value, ast.Name)
                and st.value.id == name
            )
        ]
        if len(unpacks) != 1 or stmts.index(unpacks[0]) <= stmts.index(call_stmt):
            return None
        # Every other read of ``name`` would survive the splice with nothing to bind it to.
        readers = sum(
            1
            for sub in ast.walk(self.fn_)
            if isinstance(sub, ast.Name) and sub.id == name and isinstance(sub.ctx, ast.Load)
        )
        guard = stmts[stmts.index(call_stmt) + 1] if stmts.index(call_stmt) + 1 < len(stmts) else None
        guard_reads = (
            sum(1 for sub in ast.walk(guard.test) if isinstance(sub, ast.Name) and sub.id == name)
            if isinstance(guard, ast.If)
            else 0
        )
        if readers != guard_reads + 1:
            return None
        return unpacks[0]

    def rewrite_block(self, stmts: list[ast.stmt]) -> bool:
        changed = False
        i = 0
        while i < len(stmts):
            spliced = self.try_splice(stmts, i)
            if spliced is not None:
                new_stmts, consumed, deferred, unpack = spliced
                if deferred is not None:
                    # Rewrite the far-away unpack where it stands; the splice site keeps only the
                    # computation and the guard.
                    deferred.value = unpack.value
                    deferred.targets = unpack.targets
                    ast.fix_missing_locations(deferred)
                else:
                    new_stmts = new_stmts + [unpack]
                stmts[i : i + consumed] = new_stmts
                changed = True
                i += len(new_stmts)
                continue
            for field in ("body", "orelse"):
                nested = vars(stmts[i]).get(field)
                if isinstance(nested, list) and self.rewrite_block(nested):
                    changed = True
            i += 1
        return changed

    def call_shape(
        self, stmts: list[ast.stmt], i: int
    ) -> tuple[ast.Assign, ast.If, list[ast.expr], int, ast.Assign | None] | None:
        """``(call_stmt, guard_stmt, final_targets, consumed, deferred)`` for a recognised call at
        ``stmts[i]``, or ``None``. ``consumed`` is 2 for the direct-destructure spelling, 3 when a
        separate unpack statement follows a bare-name call target. ``deferred`` is the unpack
        statement when it sits deeper in the nest instead (see :meth:`deferred_unpack`)."""
        call_stmt = stmts[i]
        if not (
            isinstance(call_stmt, ast.Assign)
            and len(call_stmt.targets) == 1
            and isinstance(call_stmt.value, ast.Call)
            and isinstance(call_stmt.value.func, ast.Name)
            and call_stmt.value.func.id in self.helpers
        ):
            return None
        target = call_stmt.targets[0]
        guard_stmt = stmts[i + 1] if i + 1 < len(stmts) else None
        if not isinstance(guard_stmt, ast.If):
            return None
        if isinstance(target, ast.Name):
            if none_toggle_op(guard_stmt.test, target.id) is not True:
                return None
            unpack = stmts[i + 2] if i + 2 < len(stmts) else None
            if (
                isinstance(unpack, ast.Assign)
                and len(unpack.targets) == 1
                and isinstance(unpack.targets[0], (ast.Tuple, ast.List))
                and isinstance(unpack.value, ast.Name)
                and unpack.value.id == target.id
            ):
                return call_stmt, guard_stmt, list(unpack.targets[0].elts), 3, None
            deferred = self.deferred_unpack(call_stmt, target.id)
            deferred_target = deferred.targets[0] if deferred is not None else None
            if deferred is not None and isinstance(deferred_target, (ast.Tuple, ast.List)):
                return call_stmt, guard_stmt, list(deferred_target.elts), 2, deferred
            return None
        if isinstance(target, (ast.Tuple, ast.List)):
            names = [e for e in target.elts if isinstance(e, ast.Name)]
            if len(names) != len(target.elts):
                return None
            guard_name = next((e.id for e in names if none_toggle_op(guard_stmt.test, e.id) is True), None)
            if guard_name is not None:
                return call_stmt, guard_stmt, list(target.elts), 2, None
        return None

    def try_splice(
        self, stmts: list[ast.stmt], i: int
    ) -> tuple[list[ast.stmt], int, ast.Assign | None, ast.Assign] | None:
        shape = self.call_shape(stmts, i)
        if shape is None:
            return None
        call_stmt, guard_stmt, final_targets, consumed, deferred = shape
        if not (
            len(guard_stmt.body) == 1
            and not guard_stmt.orelse
            and isinstance(guard_stmt.body[0], (ast.Continue, ast.Break, ast.Pass, ast.Return))
        ):
            return None
        helper = self.helpers[call_stmt.value.func.id]
        call_args = resolve_call_args(call_stmt.value, helper)
        if call_args is None:
            return None
        body = strip_docstrings_(helper.body)
        mid = body[:-1]
        guard_idx = find_none_guard(mid)
        if guard_idx is None:
            return None  # re-validated defensively; _collect_none_guarded_helpers already checked
        ret_value = body[-1].value
        ret_elts = ret_value.elts if isinstance(ret_value, (ast.Tuple, ast.List)) else [ret_value]
        if len(final_targets) != len(ret_elts):
            return None

        param_names = [a.arg for a in helper.args.args]
        local_names = collect_assigned_names(mid[:guard_idx] + mid[guard_idx + 1 :])
        arg_map = dict(zip(param_names, call_args))
        rename: dict[str, ast.AST] = dict(arg_map)
        self._counter[0] += 1
        prefix = f"__inl{self._counter[0]}_"
        reassigned_params = []
        for ln in local_names:
            rename[ln] = ast.Name(id=f"{prefix}{ln}", ctx=ast.Load())
            if ln in arg_map:
                reassigned_params.append(ln)
        renamer = SubstNames(rename)

        def clone_rename(stmt: ast.stmt) -> ast.stmt:
            cloned = ast.parse(ast.unparse(stmt)).body[0]
            cloned = renamer.visit(cloned)
            ast.fix_missing_locations(cloned)
            return cloned

        def clone_rename_expr(expr: ast.expr) -> ast.expr:
            cloned = ast.parse(ast.unparse(expr), mode="eval").body
            cloned = renamer.visit(cloned)
            ast.fix_missing_locations(cloned)
            return cloned

        new_stmts: list[ast.stmt] = []
        for pn in reassigned_params:
            init = ast.Assign(
                targets=[ast.Name(id=f"{prefix}{pn}", ctx=ast.Store())],
                value=ast.parse(ast.unparse(arg_map[pn]), mode="eval").body,
            )
            ast.fix_missing_locations(init)
            new_stmts.append(init)
        for stmt in mid[:guard_idx]:
            new_stmts.append(clone_rename(stmt))
        cond = clone_rename_expr(mid[guard_idx].test)
        handler = ast.parse(ast.unparse(guard_stmt.body[0])).body[0]
        new_stmts.append(ast.copy_location(ast.If(test=cond, body=[handler], orelse=[]), call_stmt))
        for stmt in mid[guard_idx + 1 :]:
            new_stmts.append(clone_rename(stmt))
        ret_expr = clone_rename_expr(ret_value)
        ret_elts_renamed = ret_expr.elts if isinstance(ret_expr, (ast.Tuple, ast.List)) else [ret_expr]
        targets_copy = [copy.deepcopy(t) for t in final_targets]
        if len(targets_copy) == 1:
            unpack = ast.Assign(targets=targets_copy, value=ret_elts_renamed[0])
        else:
            unpack = ast.Assign(
                targets=[ast.Tuple(elts=targets_copy, ctx=ast.Store())],
                value=ast.Tuple(elts=ret_elts_renamed, ctx=ast.Load()),
            )
        ast.fix_missing_locations(unpack)
        return new_stmts, consumed, deferred, unpack
