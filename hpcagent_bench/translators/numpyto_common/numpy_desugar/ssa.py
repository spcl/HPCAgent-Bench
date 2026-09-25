"""SSA renaming of reassigned locals (dace refuses a rebinding that changes shape)."""

import ast


#: The lowering's allocation-site marker. Its dace expansion is keyed BY NAME
#: (``zeros_locals``) and DROPS an allocation whose target it cannot find there, so a
#: marker-bound name has to keep the name the lowering recorded.
ZEROS_MARKER = "__hpcagent_bench_zeros__"


#: Version suffix. Distinct from the lowering pass's ``__v<n>`` (that one renames the
#: C/Fortran IR tree), so a name that went through both carries two readable versions.
SSA_SUFFIX = "__ssa"


def store_root(target: ast.expr) -> str | None:
    """The name a store target ultimately writes into (``x``, ``x[i]``, ``x.f[i]``)."""
    while isinstance(target, (ast.Subscript, ast.Attribute)):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def ssa_versionable(fn: ast.AST, pinned: set[str]) -> set[str]:
    """Names a straight-line SSA rename may re-version: bound MORE THAN ONCE by a plain
    ``name = ...`` at the TOP level of ``fn``, and touched no other way.

    Every other binding or escape blocks the name outright. Merging two versions across
    an ``if`` / loop body needs a phi this pass does not build; an element write
    (``x[i] = ...``) mutates the buffer the CURRENT version is bound to; and a parameter,
    a ``return``, a ``global``/``nonlocal``, a closure read or a ``pinned`` kir name is
    read by something outside this body, which would still spell the original name."""
    if not isinstance(fn, ast.FunctionDef):
        return set()  # module scope: these are globals, and a rename escapes the scope
    args = fn.args
    blocked = {p.arg for p in args.posonlyargs + args.args + args.kwonlyargs} | pinned
    blocked.update(v.arg for v in (args.vararg, args.kwarg) if v is not None)
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
    """``x = a`` ... ``x = b`` -> ``x = a`` ... ``x__ssa1 = b``, with every read up to the
    next rebinding following the new name.

    DaCe refuses a second ``x = <array>`` whose shape/dtype differs from the first
    (``Cannot reassign value to variable "x"``); one name per value removes the refusal
    without changing what the program computes. Only a name :func:`ssa_versionable`
    cleared is touched, so a version never has to be merged across a branch, and a name
    bound once keeps its spelling (no churn in the generated corpus)."""

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
            # The driver rebinds ``fn.body`` after EVERY pass, so the scan has to see the body
            # this pass is walking -- not the one that existed when the pass list was built.
            self.versionable = ssa_versionable(self.fn, self.pinned)
        self.generic_visit(node)  # the rhs (and any x[i] base) reads the CURRENT version
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
