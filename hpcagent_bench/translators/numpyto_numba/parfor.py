"""Constructs numba's parfor pass (``parallel=True``) answers differently from numpy, and the body rewrites
that keep ``parallel=True`` despite them."""

import ast

from hpcagent_bench.translators.numpyto_common.ast_build import const_int, name_, walk_nodes
from hpcagent_bench.translators.numpyto_common.parallelism import index_exprs
from hpcagent_bench.translators.numpyto_common.subscripts import base_name, is_full_slice

__all__ = [
    "PARFOR_UNSAFE_CALLS",
    "RouteUnsafeCalls",
    "REORDERING_OPS",
    "MATERIALIZE_BASE",
    "abs_offset",
    "has_unmaterializable_overlap",
    "has_unroutable_unsafe_call",
    "is_reshape_call",
    "materializable_store",
    "materialize_overlapping_stores",
    "numpy_aliases",
    "provably_disjoint",
    "reordered",
    "reshape_bound_names",
    "reshape_operand",
    "routable_call",
    "route_parfor_unsafe_calls",
    "overlapping_reads",
    "overlapping_slice_store",
    "parfor_unsafe_calls",
    "spell_out_overlapping_augassigns",
    "spell_out_reshape_augassigns",
    "spelled_out_augassign",
]

#: Whole-array calls the parfor pass answers differently from numpy (numba 0.65+): ``max`` / ``min`` drop
#: NaN, and a rectangular ``eye(m, n)`` fused with a following parallel loop is read back as square.
PARFOR_UNSAFE_CALLS = frozenset({"max", "min", "amax", "amin", "eye", "identity"})


def parfor_unsafe_calls(tree: ast.AST) -> list[ast.Call]:
    """The :data:`PARFOR_UNSAFE_CALLS` calls under ``tree``, spelled ``np.max(a)`` or ``a.max()``."""
    return [
        n
        for n in walk_nodes(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in PARFOR_UNSAFE_CALLS
    ]


def numpy_aliases(tree: ast.Module) -> set[str]:
    """The names ``numpy`` is imported as at module level (``import numpy as np`` -> ``np``)."""
    return {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "numpy"
    }


def routable_call(call: ast.Call) -> bool:
    """A call :func:`route_parfor_unsafe_calls` can hand to a function: no ``*args`` and no ``**kwargs``."""
    return not any(isinstance(a, ast.Starred) for a in call.args) and all(k.arg is not None for k in call.keywords)


def has_unroutable_unsafe_call(src: str) -> bool:
    """True if ``src`` calls a :data:`PARFOR_UNSAFE_CALLS` name in a form :func:`routable_call` refuses."""
    return any(not routable_call(c) for c in parfor_unsafe_calls(ast.parse(src)))


class RouteUnsafeCalls(ast.NodeTransformer):
    """Each :data:`PARFOR_UNSAFE_CALLS` call as a call of a module function doing the same (``x.max()`` ->
    ``serial_max(x)``), one function per spelling (call form, positional count, keyword names)."""

    def __init__(self, numpy_names: set[str], taken: set[str]) -> None:
        self.numpy_names = numpy_names
        self.taken = taken
        #: spelling -> (function name, its source)
        self.helpers: dict[tuple[str, str, int, tuple[str, ...]], tuple[str, str]] = {}

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in PARFOR_UNSAFE_CALLS and routable_call(node)):
            return node
        module = func.value.id if isinstance(func.value, ast.Name) and func.value.id in self.numpy_names else ""
        keywords = tuple(k.arg for k in node.keywords if k.arg is not None)
        spelling = (module, func.attr, len(node.args), keywords)
        if spelling not in self.helpers:
            name = f"serial_{func.attr}"
            k = 1
            while name in self.taken:
                name = f"serial_{func.attr}_{k}"
                k += 1
            self.taken.add(name)
            positional = [f"a{i}" for i in range(len(node.args))]
            params = ([] if module else ["obj"]) + positional + list(keywords)
            passed = ", ".join(positional + [f"{kw}={kw}" for kw in keywords])
            callee = f"{module}.{func.attr}" if module else f"obj.{func.attr}"
            self.helpers[spelling] = (
                name,
                f"\n\ndef {name}({', '.join(params)}):\n    return {callee}({passed})\n",
            )
        args = ([] if module else [func.value]) + list(node.args) + [k.value for k in node.keywords]
        return ast.copy_location(
            ast.Call(func=ast.Name(id=self.helpers[spelling][0], ctx=ast.Load()), args=args, keywords=[]), node
        )


def route_parfor_unsafe_calls(src: str) -> tuple[str, frozenset[str]]:
    """``src`` with every routable unsafe call made through a module function appended to it, and those
    functions' names. The caller must compile them without ``parallel=True``: a call is opaque to the parfor
    pass, so the helper runs numba's serial implementation."""
    tree = ast.parse(src)
    outermost: list[ast.Call] = []
    inner: set[int] = set()
    for call in sorted(parfor_unsafe_calls(tree), key=lambda c: (c.lineno, c.col_offset)):
        if id(call) in inner or not routable_call(call):
            continue
        outermost.append(call)
        inner |= {id(n) for n in walk_nodes(call) if n is not call}
    if not outermost:
        return src, frozenset()
    taken = {n.id for n in walk_nodes(tree) if isinstance(n, ast.Name)} | {
        n.name for n in walk_nodes(tree) if isinstance(n, ast.FunctionDef)
    }
    router = RouteUnsafeCalls(numpy_aliases(tree), taken)
    edits: list[tuple[int, int, str]] = []
    for call in outermost:
        if call.end_lineno is None or call.end_col_offset is None:
            continue
        routed = router.visit(ast.parse(ast.unparse(call), mode="eval")).body
        start = abs_offset(src, call.lineno, call.col_offset)
        end = abs_offset(src, call.end_lineno, call.end_col_offset)
        edits.append((start, end, ast.unparse(routed)))
    for start, end, text in sorted(set(edits), reverse=True):
        src = src[:start] + text + src[end:]
    helpers = "".join(helper[1] for helper in router.helpers.values())
    return src.rstrip("\n") + "\n" + helpers, frozenset(helper[0] for helper in router.helpers.values())


#: Ops that REORDER the elements they are handed. Under one of these, an RHS subscript that is
#: textually identical to the LHS is still a race: ``y[:k] += a * np.flip(y[:k])`` has element i
#: reading element k-1-i, so a parfor over i reads cells other iterations are writing. Without this
#: list the identical-subscript rule below would call durbin safe, which it is not.
REORDERING_OPS = frozenset({"flip", "fliplr", "flipud", "roll", "rot90", "transpose", "sort", "argsort"})


def provably_disjoint(lhs: ast.Subscript, rhs: ast.Subscript) -> bool:
    """True when the two subscripts cannot name a common element.

    The only case decided here is the one the corpus actually needs: some axis where BOTH sides
    are integer constants that differ. ``p[-1, :]`` against ``p[-2, :]`` is the boundary-condition
    copy every stencil ends with, and it touches disjoint rows. Signs must match -- ``a[0]`` and
    ``a[-1]`` are the SAME element on a length-1 axis, so mixing them decides nothing."""
    left, right = index_exprs(lhs), index_exprs(rhs)
    for a, b in zip(left, right, strict=False):
        ca, cb = const_int(a), const_int(b)
        if ca is None or cb is None:
            continue
        if (ca < 0) != (cb < 0):
            continue
        if ca != cb:
            return True
    return False


def reordered(stmt: ast.AST, target: ast.Subscript) -> bool:
    """True if ``target`` appears anywhere under a :data:`REORDERING_OPS` call in ``stmt``, or
    under a negative-step slice -- both make an element-for-element read a permuted one."""
    for node in walk_nodes(stmt):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else ""
            if name in REORDERING_OPS and any(child is target for child in walk_nodes(node)):
                return True
    for node in walk_nodes(target):
        if isinstance(node, ast.Slice) and node.step is not None and (const_int(node.step) or 0) < 0:
            return True
    return False


def overlapping_slice_store(stmt: ast.AST) -> bool:
    """True if ``stmt`` stores through a slice of an array its right-hand side may read at other cells."""
    return bool(overlapping_reads(stmt))


def overlapping_reads(stmt: ast.AST) -> list[ast.expr]:
    """The right-hand-side reads that may name other cells of the array ``stmt`` stores a slice of. A bare
    read of the whole array counts unless every target axis is a full ``:`` (the elementwise ``a[:] += a``)."""
    if not isinstance(stmt, (ast.AugAssign, ast.Assign)):
        return []
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
    reads: list[ast.expr] = []
    for target in targets:
        if not isinstance(target, ast.Subscript) or not any(isinstance(c, ast.Slice) for c in walk_nodes(target)):
            continue
        lhs_name = base_name(target)
        if lhs_name is None:
            continue
        whole_target = all(is_full_slice(d) for d in index_exprs(target))
        # A Name that is a Subscript's base is that subscript's read, which the subscript rules below decide.
        subscript_bases = {id(node.value) for node in walk_nodes(stmt.value) if isinstance(node, ast.Subscript)}
        for rhs in walk_nodes(stmt.value):
            if isinstance(rhs, ast.Name) and rhs.id == lhs_name and not whole_target and id(rhs) not in subscript_bases:
                reads.append(rhs)
                continue
            if not isinstance(rhs, ast.Subscript) or base_name(rhs) != lhs_name:
                continue
            if provably_disjoint(target, rhs):
                continue
            if ast.dump(target.slice) == ast.dump(rhs.slice) and not reordered(stmt, rhs):
                continue
            reads.append(rhs)
    return reads


def materializable_store(stmt: ast.AST) -> bool:
    """True if ``stmt`` is an overlapping slice store with one target and a plain ``=``."""
    return isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and overlapping_slice_store(stmt)


def has_unmaterializable_overlap(src: str) -> bool:
    """True if some overlapping slice store of ``src`` is not a :func:`materializable_store`."""
    return any(overlapping_slice_store(s) and not materializable_store(s) for s in walk_nodes(ast.parse(src)))


#: Base name of the module function returning a copy of an overlapping store's right-hand side.
MATERIALIZE_BASE = "materialized"


def materialize_overlapping_stores(src: str) -> tuple[str, str | None]:
    """``src`` with the right-hand side of every :func:`materializable_store` passed through an appended
    function returning a copy of it, and that function's name (``None`` when no store needs it). The parfor
    pass may store ``a[1:] = a[:-1]`` straight from the array it writes; the opaque copy restores numpy's
    evaluate-then-store order."""
    tree = ast.parse(src)
    stores = [s for s in walk_nodes(tree) if isinstance(s, ast.Assign) and materializable_store(s)]
    if not stores:
        return src, None
    taken = {n.id for n in walk_nodes(tree) if isinstance(n, ast.Name)} | {
        n.name for n in walk_nodes(tree) if isinstance(n, ast.FunctionDef)
    }
    name = MATERIALIZE_BASE
    k = 1
    while name in taken:
        name = f"{MATERIALIZE_BASE}_{k}"
        k += 1
    edits: list[tuple[int, int, str]] = []
    for stmt in stores:
        value = stmt.value
        if value.end_lineno is None or value.end_col_offset is None:
            continue
        start = abs_offset(src, value.lineno, value.col_offset)
        end = abs_offset(src, value.end_lineno, value.end_col_offset)
        edits.append((start, end, f"{name}({src[start:end]})"))
    for start, end, text in sorted(set(edits), reverse=True):
        src = src[:start] + text + src[end:]
    helper = f"\n\ndef {name}(x):\n    return np.copy(x)\n"
    return src.rstrip("\n") + "\n" + helper, name


def spell_out_overlapping_augassigns(src: str) -> str:
    """Rewrite each :func:`overlapping_slice_store` augmented assignment into its plain form
    (:func:`spelled_out_augassign`).

    numba evaluates ``out[1:] += out[:-1]`` in place, element by element, even in a serial build:
    each cell adds the neighbour it has just updated, a running sum where numpy adds the old
    values. ``out[1:] = out[1:] + out[:-1]`` builds the right-hand side first, as numpy does, and
    numba's slice store copies an overlapping source. Statements are spliced in place, so the rest
    of the body stays verbatim."""
    tree = ast.parse(src)
    edits: list[tuple[int, int, str]] = []
    for stmt in walk_nodes(tree):
        if (
            isinstance(stmt, ast.AugAssign)
            and not isinstance(stmt.op, ast.MatMult)
            and overlapping_slice_store(stmt)
            and stmt.end_lineno is not None
            and stmt.end_col_offset is not None
        ):
            start = abs_offset(src, stmt.lineno, stmt.col_offset)
            end = abs_offset(src, stmt.end_lineno, stmt.end_col_offset)
            edits.append((start, end, ast.unparse(spelled_out_augassign(stmt))))
    for start, end, text in sorted(set(edits), reverse=True):
        src = src[:start] + text + src[end:]
    return src


def abs_offset(src: str, lineno: int, col: int) -> int:
    """Absolute character offset of (1-based ``lineno``, 0-based ``col``). The
    source is ASCII, so ``col_offset`` (a UTF-8 byte offset) equals the char offset."""
    lines = src.splitlines(keepends=True)
    return sum(len(line) for line in lines[: lineno - 1]) + col


def is_reshape_call(node: ast.expr) -> bool:
    """``np.reshape(b, s)`` or ``b.reshape(s)``."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "reshape"


def reshape_operand(node: ast.expr, reshaped: set[str]) -> bool:
    """True if the elementwise expression ``node`` has a reshape as one of its OPERANDS: the call
    itself or a name in ``reshaped``, reached through arithmetic only. A reshape inside a call's
    arguments (``np.sum(b.reshape(...))``) or an index is not an operand of the store."""
    if isinstance(node, ast.BinOp):
        return reshape_operand(node.left, reshaped) or reshape_operand(node.right, reshaped)
    if isinstance(node, ast.UnaryOp):
        return reshape_operand(node.operand, reshaped)
    return is_reshape_call(node) or (isinstance(node, ast.Name) and node.id in reshaped)


def reshape_bound_names(fn: ast.FunctionDef) -> set[str]:
    """Names ``fn`` binds, anywhere, directly to a reshape (``t = np.reshape(b, s)``)."""
    return {
        t.id
        for n in walk_nodes(fn)
        if isinstance(n, ast.Assign) and is_reshape_call(n.value)
        for t in n.targets
        if isinstance(t, ast.Name)
    }


def spelled_out_augassign(stmt: ast.AugAssign) -> ast.Assign:
    """``t op= v`` as the plain store numba lowers correctly: ``t[...] = t op v`` for a whole array
    (the store must land in the caller's buffer, not rebind the name), ``s = s op v`` for a
    subscript target -- numpy's own meaning of an augmented store through an index."""
    target = stmt.target
    load = name_(target.id) if isinstance(target, ast.Name) else target
    value = ast.BinOp(left=load, op=stmt.op, right=stmt.value)
    store: ast.expr
    if isinstance(target, ast.Name):
        store = ast.Subscript(value=name_(target.id), slice=ast.Constant(Ellipsis), ctx=ast.Store())
    else:
        store = target
    return ast.Assign(targets=[store], value=value, lineno=stmt.lineno)


def spell_out_reshape_augassigns(src: str) -> str:
    """Rewrite each augmented assignment with a reshape as a right-hand operand into its plain form
    (:func:`spelled_out_augassign`).

    Under ``parallel=True`` numba 0.65-0.67 lowers ``out += np.reshape(bias, (1, c, 1))`` -- and
    ``out[0] += b.reshape(c, 1)``, and the same through a name bound to the reshape -- as if the
    right-hand side had ``out``'s full shape: it reads the broadcast operand's flat buffer past its
    end, and the result is garbage or NaN with no error (every conv kernel's closing bias add). The
    plain store ``out[...] = out + np.reshape(...)`` broadcasts correctly and still runs as a parfor.
    A broadcast whose size-1 axes are known only at run time does not reach this: numba raises
    its own AssertionError there, a clean decline. Statements are spliced in place, so the rest of
    the body stays verbatim."""
    tree = ast.parse(src)
    edits: list[tuple[int, int, str]] = []
    for fn in (n for n in walk_nodes(tree) if isinstance(n, ast.FunctionDef)):
        reshaped = reshape_bound_names(fn)
        for stmt in walk_nodes(fn):
            if (
                isinstance(stmt, ast.AugAssign)
                and isinstance(stmt.target, (ast.Name, ast.Subscript))
                and not isinstance(stmt.op, ast.MatMult)
                and reshape_operand(stmt.value, reshaped)
            ):
                start = abs_offset(src, stmt.lineno, stmt.col_offset)
                if stmt.end_lineno is None or stmt.end_col_offset is None:
                    continue
                end = abs_offset(src, stmt.end_lineno, stmt.end_col_offset)
                edits.append((start, end, ast.unparse(spelled_out_augassign(stmt))))
    for start, end, text in sorted(set(edits), reverse=True):
        src = src[:start] + text + src[end:]
    return src
