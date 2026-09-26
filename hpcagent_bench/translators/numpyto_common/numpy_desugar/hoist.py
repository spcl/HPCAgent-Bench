"""Value hoisting: lift a matched call out of its expression into statements that compute a temp."""

import ast
from dataclasses import dataclass
from collections.abc import Callable

from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet

__all__ = [
    "BINDING_EXPRESSIONS",
    "VALUE_STATEMENTS",
    "FormRewriter",
    "HoistForm",
    "HoistTables",
    "ValueHoist",
    "always_live",
    "drops_nothing",
    "has_cue",
    "reads_any",
    "scope_bound_names",
]


@dataclass(frozen=True, slots=True)
class HoistTables:
    """One function scope's facts the value-hoist forms read, built once before that scope's passes run."""

    ranks: dict[str, int]
    dtypes: dict[str, str]
    #: Vetted ``v = a[mask]`` boolean selects (:func:`masked_reduce_map`).
    gathers: dict[str, tuple]
    #: The ``np.linalg`` ops the backend lacks, and the ``solve`` rhs ranks it lowers anyway.
    lower_ops: set
    solve_rhs_ranks: frozenset


def always_live(tables: HoistTables) -> bool:
    """The liveness hook of a form that can match in any scope."""
    return True


def drops_nothing(stmt: ast.stmt, tables: HoistTables) -> bool:
    """The statement hook of a form that only hoists."""
    return False


@dataclass(frozen=True, slots=True)
class HoistForm:
    """One expression form :class:`ValueHoist` lowers into loops.

    ``rewrite`` swaps a matched node for a fresh temp (numbered from the hoist's ``ctr``) and queues the statements
    computing it on the hoist, or returns None. It can only match in a statement holding a cue: a node whose type is
    in ``cue_kinds`` or an attribute named in ``cue_attrs``. ``live`` says whether a scope's tables let the form match
    at all; ``drop`` removes a whole statement.
    """

    cue_attrs: frozenset[str]
    cue_kinds: tuple[type[ast.AST], ...]
    rewrite: Callable[[ast.AST, "ValueHoist"], ast.expr | None]
    live: Callable[[HoistTables], bool] = always_live
    drop: Callable[[ast.stmt, HoistTables], bool] = drops_nothing


VALUE_STATEMENTS = (ast.Assign, ast.AugAssign, ast.Return, ast.Expr)


#: Expressions binding names for their own body: a temp hoisted out of one cannot read those names.
BINDING_EXPRESSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda)


def has_cue(value: ast.AST, form: HoistForm) -> bool:
    """Whether ``value`` holds a cue of ``form``: one scan, so a value the form cannot match is never rewritten."""
    attrs, kinds = form.cue_attrs, form.cue_kinds
    stack: list[object] = [value]
    while stack:
        node = stack.pop()
        if not isinstance(node, ast.AST):
            continue  # a None dict key
        if type(node) in kinds or (type(node) is ast.Attribute and node.attr in attrs):
            return True
        for child in vars(node).values():
            if type(child) is list:
                stack.extend(child)
            elif isinstance(child, ast.AST):
                stack.append(child)
    return False


def scope_bound_names(node: ast.AST) -> OrderedSet[str]:
    """Every parameter and stored name under a comprehension or lambda: a superset of what it binds for its body."""
    names: OrderedSet[str] = OrderedSet()
    for sub in ast.walk(node):
        if isinstance(sub, ast.arg):
            names.add(sub.arg)
        elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
            names.add(sub.id)
    return names


def reads_any(node: ast.AST, names: OrderedSet[str]) -> bool:
    """Whether ``node`` reads or writes a name in ``names``."""
    return any(isinstance(sub, ast.Name) and sub.id in names for sub in ast.walk(node))


class FormRewriter(ast.NodeTransformer):
    """Rewrite one statement's value with a hoist's form, bottom-up so inner matches go first. A node reading a name
    an enclosing comprehension or lambda binds stays put: its temp would be computed before that name exists."""

    def __init__(self, hoist: "ValueHoist") -> None:
        self.hoist = hoist
        self.bound: OrderedSet[str] = OrderedSet()

    def visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, BINDING_EXPRESSIONS):
            enclosing = self.bound
            self.bound = enclosing | scope_bound_names(node)
            self.generic_visit(node)
            self.bound = enclosing
            return node
        self.generic_visit(node)
        if self.bound and reads_any(node, self.bound):
            return node
        replacement = self.hoist.form.rewrite(node, self.hoist)
        return node if replacement is None else ast.copy_location(replacement, node)


class ValueHoist:
    """Hoist one :class:`HoistForm` out of every value-bearing statement (Assign / AugAssign / Return / Expr), splicing
    the statements computing each temp in front of it. Walks statements only and rewrites just the values holding the
    form's cue. ``ctr`` carries across statements so every temp name is fresh."""

    __slots__ = ("form", "tables", "live", "ctr", "pre", "changed")

    def __init__(self, form: HoistForm, tables: HoistTables) -> None:
        self.form = form
        self.tables = tables
        self.live = form.live(tables)
        self.ctr = 0
        self.pre: list[ast.stmt] = []
        self.changed = False

    def visit(self, stmt: ast.stmt) -> list[ast.stmt]:
        return self.block([stmt]) if self.live else [stmt]

    def queue(self, lines: list[str]) -> None:
        """Queue source lines computing a temp, spliced in front of the current statement."""
        self.pre.extend(ast.parse("\n".join(lines)).body)

    def block(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in stmts:
            if isinstance(stmt, VALUE_STATEMENTS):
                out.extend(self.statement(stmt))
            else:
                self.descend(stmt)
                out.append(stmt)
        return out

    def descend(self, node: ast.AST) -> None:
        """Hoist in every statement list under a compound statement, handler and match-case bodies included."""
        fields = vars(node)
        for name in type(node)._fields:
            children = fields.get(name)
            if type(children) is not list or not children:
                continue
            if isinstance(children[0], ast.stmt):
                children[:] = self.block(children)
            elif isinstance(children[0], (ast.excepthandler, ast.match_case)):
                for owner in children:
                    self.descend(owner)

    def statement(self, stmt: ast.Assign | ast.AugAssign | ast.Return | ast.Expr) -> list[ast.stmt]:
        if self.form.drop(stmt, self.tables):
            self.changed = True
            return []
        value = stmt.value
        if value is None or not has_cue(value, self.form):
            return [stmt]
        stmt.value = FormRewriter(self).visit(value)
        if not self.pre:
            return [stmt]
        self.changed = True
        hoisted, self.pre = self.pre, []
        return hoisted + [stmt]
