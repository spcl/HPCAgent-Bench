"""SSA renaming of reassigned locals (dace refuses a rebinding that changes shape)."""

import ast


#: The lowering's allocation-site marker. Its dace expansion looks targets up by name
#: (``zeros_locals``) and drops unknown ones, so a marker-bound name must not be renamed.
ZEROS_MARKER = "__hpcagent_bench_zeros__"


#: Version suffix, distinct from the C/Fortran lowering's ``__v<n>``.
SSA_SUFFIX = "__ssa"


def store_root(target: ast.expr) -> str | None:
    """The name a store target ultimately writes into (``x``, ``x[i]``, ``x.f[i]``)."""
    while isinstance(target, (ast.Subscript, ast.Attribute)):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def top_level_plain_bindings(fn: ast.FunctionDef, blocked: set[str]) -> tuple[dict[str, int], set[int]]:
    """How often each name is bound by a top-level ``name = ...`` of ``fn``, and the ids of those target
    nodes. A name bound to the zeros marker is added to ``blocked``."""
    counts: dict[str, int] = {}
    plain: set[int] = set()
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            counts[stmt.targets[0].id] = counts.get(stmt.targets[0].id, 0) + 1
            plain.add(id(stmt.targets[0]))
            if (
                isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == ZEROS_MARKER
            ):
                blocked.add(stmt.targets[0].id)
    return counts, plain


def ssa_versionable(fn: ast.AST, pinned: set[str]) -> set[str]:
    """Names bound more than once by a plain top-level ``name = ...`` of ``fn`` and touched no other way.

    Any other binding blocks the name: a branch/loop binding would need a phi; an element write
    mutates the current version's buffer; a parameter, ``return``, ``global``/``nonlocal``, closure
    read or ``pinned`` kir name is read from outside under the original spelling."""
    if not isinstance(fn, ast.FunctionDef):
        return set()  # module scope: a rename would escape
    args = fn.args
    blocked = {p.arg for p in args.posonlyargs + args.args + args.kwonlyargs} | pinned
    blocked.update(v.arg for v in (args.vararg, args.kwarg) if v is not None)
    counts, plain = top_level_plain_bindings(fn, blocked)
    used: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            used.add(node.id)
            if not isinstance(node.ctx, ast.Load) and id(node) not in plain:
                blocked.add(node.id)  # a nested-block, loop-target, unpack or del binding
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            blocked.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            blocked.add(node.name)  # ``except E as e`` binds through a str field, not a Name node
        elif isinstance(node, (ast.Return, ast.Lambda)) or (isinstance(node, ast.FunctionDef) and node is not fn):
            blocked.update(n.id for n in ast.walk(node) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                for sub in ast.walk(target):
                    root = store_root(sub) if isinstance(sub, (ast.Subscript, ast.Attribute)) else None
                    if root is not None:
                        blocked.add(root)
    return {
        n
        for n, c in counts.items()
        if c > 1 and n not in blocked and not any(u.startswith(n + SSA_SUFFIX) for u in used)
    }


class SsaRename(ast.NodeTransformer):
    """``x = a`` ... ``x = b`` -> ``x = a`` ... ``x__ssa1 = b``, later reads following the new name.

    DaCe refuses rebinding a name to a different shape/dtype. Only names :func:`ssa_versionable`
    clears are touched, so no version is merged across a branch."""

    def __init__(self, fn: ast.AST, pinned: set[str]) -> None:
        self.fn = fn
        self.pinned = pinned
        self.versionable: set[str] | None = None
        self.seen: dict[str, int] = {}
        self.version: dict[str, str] = {}
        self.changed = False

    def visit_Name(self, node: ast.Name) -> ast.AST:
        current = self.version.get(node.id)
        if current is None or not isinstance(node.ctx, ast.Load):
            return node
        return ast.copy_location(ast.Name(id=current, ctx=node.ctx), node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if self.versionable is None:
            # Scanned lazily: the driver replaces ``fn.body`` after every earlier pass.
            self.versionable = ssa_versionable(self.fn, self.pinned)
        self.generic_visit(node)  # the rhs reads the current version, before the rebinding
        target = node.targets[0] if len(node.targets) == 1 else None
        if not isinstance(target, ast.Name) or target.id not in self.versionable:
            return node
        self.seen[target.id] = nth = self.seen.get(target.id, 0) + 1
        if nth > 1:
            fresh = f"{target.id}{SSA_SUFFIX}{nth - 1}"
            self.version[target.id] = fresh
            target.id = fresh
            self.changed = True
        return node
