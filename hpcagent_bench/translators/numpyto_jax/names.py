"""Name-flow queries and small AST builders over a kernel's statements."""

import ast
import copy

__all__ = [
    "as_store",
    "base_name",
    "deep_copy",
    "definite_writes",
    "has_break",
    "is_assignable",
    "is_identity_test",
    "is_np_attr",
    "load",
    "names_loaded",
    "names_stored",
    "reads_before_write",
    "stmt_rhs_loads",
    "store_target_names",
    "tuple_expr",
    "upward_exposed",
]


def names_loaded(node: ast.AST) -> set[str]:
    """Names read (Load context) anywhere under ``node``."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add(n.id)
    return out


def store_target_names(t: ast.AST, out: set[str]) -> None:
    """Names one assignment target binds: a ``Name``, a ``Subscript``'s base
    array (``a[i] = ..`` mutates ``a``), each element of a tuple/list unpack
    (lulesh's ``x, y, z, .. = _lagrange_nodal(..)``), or a starred target."""
    if isinstance(t, ast.Subscript):
        base = t
        while isinstance(base, ast.Subscript):
            base = base.value
        if isinstance(base, ast.Name):
            out.add(base.id)
    elif isinstance(t, ast.Name):
        out.add(t.id)
    elif isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            store_target_names(e, out)
    elif isinstance(t, ast.Starred):
        store_target_names(t.value, out)


def names_stored(node: ast.AST) -> set[str]:
    """Names assigned (a plain ``Name`` target, incl. ``a[i] = ...`` whose base
    array name is mutated, tuple-unpack targets, and augmented assigns)."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                store_target_names(t, out)
    return out


def has_break(body: list[ast.stmt]) -> bool:
    for s in body:
        for n in ast.walk(s):
            if isinstance(n, ast.Break):
                return True
    return False


def definite_writes(stmts: list[ast.stmt]) -> set[str]:
    """Names DEFINITELY written by straight-line execution of ``stmts``. A write
    inside a loop (may run zero times) is not definite; a write inside an ``if``
    counts only when it happens on BOTH arms."""
    out: set[str] = set()
    for s in stmts:
        if isinstance(s, ast.If):
            out |= definite_writes(s.body) & definite_writes(s.orelse)
        elif isinstance(s, (ast.For, ast.While)):
            continue  # a zero-trip loop writes nothing
        else:
            out |= names_stored(ast.Module(body=[s], type_ignores=[]))
    return out


def upward_exposed(stmts: list[ast.stmt]) -> set[str]:
    """Live-in of a straight-line block: names read before being definitely
    written. Decides which vars a preceding loop must thread OUT -- a plain
    ``names_loaded`` over the rest-of-body over-approximates (vadv/cloudsc's
    scratch ``bcol``/``zqadj``, re-derived before use, would wrongly look live
    and pull an undefined name into the carry tuple).

    A loop's writes are never definite (may run zero times), so both a
    read-before-write and a write-only var stay exposed -- conservative:
    keep a carry rather than drop a needed one."""
    read: set[str] = set()
    written: set[str] = set()

    def add_reads(names: set[str]) -> None:
        for nm in names:
            if nm not in written:
                read.add(nm)

    for s in stmts:
        if isinstance(s, ast.For):
            add_reads(names_loaded(s.iter))
            add_reads(upward_exposed(s.body))
        elif isinstance(s, ast.While):
            add_reads(names_loaded(s.test))
            add_reads(upward_exposed(s.body))
        elif isinstance(s, ast.If):
            add_reads(names_loaded(s.test))
            add_reads(upward_exposed(s.body))
            add_reads(upward_exposed(s.orelse))
            written |= definite_writes(s.body) & definite_writes(s.orelse)
        else:
            add_reads(stmt_rhs_loads(s))
            written |= names_stored(ast.Module(body=[s], type_ignores=[]))
    return read


def stmt_rhs_loads(s: ast.stmt) -> set[str]:
    """Names read by a statement, *excluding* a bare ``Name`` assignment
    target (which is a pure write). Subscript-target container reads and all
    RHS reads count. An ``AugAssign`` target is read (augmented update)."""
    if isinstance(s, ast.AugAssign):
        # ``x <op>= v`` reads x and v; ``A[i] <op>= v`` reads A, i and v.
        reads = names_loaded(s.value) | names_loaded(s.target)
        reads.add(base_name(s.target))
        return reads
    if isinstance(s, ast.Assign):
        loads = set().union(*[names_loaded(t) for t in s.targets]) if s.targets else set()
        # a plain ``x = ...`` target Name is a write, not a read
        for t in s.targets:
            if isinstance(t, ast.Name):
                loads.discard(t.id)
        return loads | names_loaded(s.value)
    return names_loaded(s)


def base_name(t: ast.AST) -> str:
    while isinstance(t, ast.Subscript):
        t = t.value
    return t.id if isinstance(t, ast.Name) else "<expr>"


def load(t: ast.AST) -> ast.AST:
    t2 = ast.fix_missing_locations(ast.parse(ast.unparse(t), mode="eval").body)
    return t2


def reads_before_write(stmts: list[ast.stmt], v: str) -> bool:
    """Does ``v`` appear as an input (RHS / container / index read) in ``stmts``
    before any statement writes it? A nested compound statement is treated
    conservatively -- any load of ``v`` inside it counts as a read."""
    written: set[str] = set()
    for s in stmts:
        if v in stmt_rhs_loads(s) and v not in written:
            return True
        written |= names_stored(ast.Module(body=[s], type_ignores=[]))
    return False


def is_identity_test(test: ast.AST) -> bool:
    """A pure identity check (``x is None``/``is not None``, fv3_dycore's
    optional ``if del6_v is not None:``). Trace-time-concrete (never a value on
    a traced array), so the ``if`` stays a REAL Python branch -- ``jnp.where``
    can't test/select on ``None`` and would execute both arms."""
    return (
        isinstance(test, ast.Compare) and bool(test.ops) and all(isinstance(op, (ast.Is, ast.IsNot)) for op in test.ops)
    )


def tuple_expr(names: list[str]) -> str:
    """A Python tuple literal that is valid for 0/1/n elements."""
    return "()" if not names else "(" + ", ".join(names) + ",)"


def is_assignable(node: ast.AST) -> bool:
    """An expression that can be an assignment target (so a mutating helper's
    in-place arg can be rebound from its return)."""
    return isinstance(node, (ast.Name, ast.Subscript, ast.Attribute))


def as_store(node: ast.AST) -> ast.AST:
    """A copy of an assignable expression with Store context -- the original stays
    a Load arg inside the call, so it must not be mutated in place."""
    n = copy.deepcopy(node)
    n.ctx = ast.Store()
    return n


def is_np_attr(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Name)
        and node.value.id in ("np", "jnp")
    )


def deep_copy(node: ast.AST) -> ast.AST:
    return ast.parse(ast.unparse(node), mode="eval").body
