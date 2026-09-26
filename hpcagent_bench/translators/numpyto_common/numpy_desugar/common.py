"""Shared pieces of the desugar passes: the refusal type, numpy call recognition, small AST helpers."""

import ast

__all__ = [
    "AUG_OP_SRC",
    "LIKE_CTORS",
    "REDUCE_FNS",
    "SHAPE_CTORS",
    "DesugarError",
    "RankedRewritePass",
    "RewritePass",
    "as_stmts",
    "const_int",
    "eigh_alias_names",
    "eigh_call_ab",
    "eigh_call_kind",
    "expr_of",
    "is_eigh_assign_target",
    "name_store_counts",
    "np_attr",
    "np_submodule_attr",
    "reachable_functions",
    "replace_call_with_name",
    "tuple_len",
]


class DesugarError(NotImplementedError):
    """A pass matched a construct it owns but cannot lower this variant of it.

    Never swallowed, so the emit fails instead of producing a wrong kernel. A construct no pass owns is left
    verbatim (a clean backend skip)."""


class RewritePass(ast.NodeTransformer):
    """A desugar pass: ``changed`` turns True once it rewrites anything, ``_ctr`` numbers the temps it mints."""

    def __init__(self) -> None:
        self.changed = False
        self._ctr = 0


class RankedRewritePass(RewritePass):
    """A :class:`RewritePass` that reads the scope's name -> rank table."""

    __slots__ = ("ranks",)

    def __init__(self, ranks: dict[str, int]) -> None:
        super().__init__()
        self.ranks = ranks


# Constructors whose first arg is a shape tuple -> result rank = len(shape).
SHAPE_CTORS = {"empty", "zeros", "ones", "full", "ndarray"}


LIKE_CTORS = {"empty_like", "zeros_like", "ones_like"}


def np_attr(node: ast.AST) -> str | None:
    """``np.<attr>`` / ``numpy.<attr>`` call -> ``attr`` else None."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in ("np", "numpy")
    ):
        return node.func.attr
    return None


def np_submodule_attr(node: ast.AST, submodule: str) -> str | None:
    """``np.<submodule>.<attr>(...)`` call (``np.fft.fft``, ``np.linalg.solve``) -> ``attr``, else None."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == submodule
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id in ("np", "numpy")
    ):
        return node.func.attr
    return None


def tuple_len(node: ast.AST) -> int | None:
    if isinstance(node, (ast.Tuple, ast.List)):
        return len(node.elts)
    return None


#: numpy reductions that take an ``axis`` (drops the reduced axes; no axis ->
#: scalar). Used only for ndim propagation, not rewriting.
REDUCE_FNS = {"sum", "prod", "mean", "std", "var", "min", "max", "amin", "amax", "argmin", "argmax", "any", "all"}


def const_int(node: ast.AST) -> int | None:
    """A constant integer literal, including a negated one (``axis=-1`` parses as
    ``UnaryOp(USub, Constant(1))``, NOT ``Constant(-1)``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = const_int(node.operand)
        return None if v is None else -v
    return None


def replace_call_with_name(root: ast.AST, target: ast.Call, name: str) -> None:
    """Swap one already-lowered Call node for a Name reference, wherever it sits in ``root``."""

    class Swap(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node) if node is target else node

    Swap().visit(root)


def as_stmts(res) -> list[ast.stmt]:
    """A NodeTransformer result (one node, a list, or a dropped ``None``) as a list."""
    if res is None:
        return []
    return res if isinstance(res, list) else [res]


def eigh_alias_names(tree: ast.AST) -> set:
    """Names bound to ``scipy.linalg.eigh`` by ``from scipy.linalg import eigh [as X]``; attribute calls
    (``np.linalg.eigh``, ``scipy.linalg.eigh``) are recognised separately."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("scipy.linalg", "scipy"):
            for al in node.names:
                base = al.name if node.module == "scipy.linalg" else None
                if base == "eigh" or al.name == "linalg.eigh":
                    out.add(al.asname or al.name)
    return out


def eigh_call_kind(node: ast.AST, alias_names: set):
    """``eigh(a[, b], ...)`` / ``eigvalsh(a, ...)`` -> ``(kind, a_node, b_node_or_None, kwargs)``, else None.

    Matches ``np.linalg``, ``scipy.linalg`` and an imported ``eigh`` alias. ``kind`` is ``"eigh"`` (an
    eigenpair ``(w, U)``) or ``"eigvalsh"`` (eigenvalues only). numpy has no generalized ``eigvalsh``, so
    its second positional argument is ``UPLO``, never a metric ``b``."""
    if not isinstance(node, ast.Call) or not node.args:
        return None
    f = node.func
    linalg_attr = np_submodule_attr(node, "linalg")
    scipy_attr = (
        f.attr
        if isinstance(f, ast.Attribute)
        and isinstance(f.value, ast.Attribute)
        and f.value.attr == "linalg"
        and isinstance(f.value.value, ast.Name)
        and f.value.value.id == "scipy"
        else None
    )
    if linalg_attr == "eigvalsh" or scipy_attr == "eigvalsh":
        kind = "eigvalsh"
    elif linalg_attr == "eigh" or scipy_attr == "eigh" or (isinstance(f, ast.Name) and f.id in alias_names):
        kind = "eigh"
    else:
        return None
    kw = {k.arg: k.value for k in node.keywords}
    a = node.args[0]
    b = None if kind == "eigvalsh" else (node.args[1] if len(node.args) > 1 else kw.get("b"))
    return kind, a, b, kw


def eigh_call_ab(node: ast.AST, alias_names: set):
    """:func:`eigh_call_kind` without the kind: ``(a_node, b_node_or_None, kwargs)``, else None."""
    hit = eigh_call_kind(node, alias_names)
    return None if hit is None else hit[1:]


def is_eigh_assign_target(node: ast.AST, alias_names: set) -> bool:
    """True for an assignment :class:`EighLoopRewriter` lowers directly: ``w, v = eigh(...)`` or
    ``w = eigvalsh(...)``. Its RHS must not be hoisted first, or the rewriter no longer sees the call."""
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
        return False
    hit = eigh_call_kind(node.value, alias_names)
    if hit is None:
        return False
    tgt = node.targets[0]
    if isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2 and all(isinstance(e, ast.Name) for e in tgt.elts):
        return True
    return hit[0] == "eigvalsh" and isinstance(tgt, ast.Name)


#: augmented-assignment op -> its source spelling (the scatter loop is emitted as text).
AUG_OP_SRC = {
    ast.Add: "+=",
    ast.Sub: "-=",
    ast.Mult: "*=",
    ast.Div: "/=",
    ast.FloorDiv: "//=",
    ast.Mod: "%=",
    ast.Pow: "**=",
    ast.BitAnd: "&=",
    ast.BitOr: "|=",
    ast.BitXor: "^=",
    ast.LShift: "<<=",
    ast.RShift: ">>=",
}


def expr_of(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


def name_store_counts(fn: ast.FunctionDef) -> dict[str, int]:
    """How many times each name is bound in ``fn``: every Store/Del plus one per parameter."""
    counts: dict[str, int] = {}
    for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        counts[a.arg] = counts.get(a.arg, 0) + 1
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            counts[node.id] = counts.get(node.id, 0) + 1
    return counts


def reachable_functions(funcs: list[ast.FunctionDef], entry: str) -> set[str]:
    """Names of the functions ``entry`` reaches through any Name reference, itself included; every name
    when ``entry`` is not among ``funcs``."""
    by_name = {fn.name: fn for fn in funcs}
    if entry not in by_name:
        return set(by_name)
    seen: set[str] = set()
    stack = [entry]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        stack.extend(n.id for n in ast.walk(by_name[name]) if isinstance(n, ast.Name) and n.id in by_name)
    return seen
