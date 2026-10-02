"""Kept helpers: per-call-signature specialisation and compile-time argument binding."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.ast_build import SubstituteLoads, map_blocks, name_, store_
from hpcagent_bench.translators.numpyto_common.frontend.helper_params import ConstArg
from hpcagent_bench.translators.numpyto_common.frontend.inlining import collect_assigned_names, fold_constant_branches
from hpcagent_bench.translators.numpyto_common.frontend.none_folding import FoldStaticNoneBranches
from hpcagent_bench.translators.numpyto_common.frontend.shapes import resolve_array_ref
from hpcagent_bench.translators.numpyto_common.ir import ArrayDesc

__all__ = [
    "bind_call_constants",
    "call_arg_key",
    "drop_unreachable_after_return",
    "literal_call_arg",
    "literal_key",
    "rewrite_returns_to_outparam",
    "specialise_helper_by_call_signature",
    "substitute_names",
]


def call_arg_key(arg: ast.expr, kernel_fn: ast.FunctionDef, arr_by: dict[str, ArrayDesc]) -> ConstArg | tuple[str, ...]:
    """What a call argument contributes to a helper's specialisation key.

    A constant is folded into the body, and an array's extents are emitted as constants, so two
    sites disagreeing on either need two helpers. Keying on the argument's NAME instead would
    split sites that agree, and keying on nothing (the argument's shape being unresolvable) merges
    sites that do not -- mamba2 calls ``_segsum`` on two differently shaped locals, and one body
    cannot serve both.
    """
    if isinstance(arg, ast.Constant):
        return arg.value
    resolved = resolve_array_ref(kernel_fn, arg, arr_by)
    return resolved[0] if resolved is not None else ast.unparse(arg)


def specialise_helper_by_call_signature(
    tree: ast.Module,
    hdef: ast.FunctionDef,
    owners: list[tuple[ast.FunctionDef, dict[str, ArrayDesc]]],
) -> list[ast.FunctionDef]:
    """Give each distinct set of constant call arguments to ``hdef`` its own copy of the helper.

    A kept helper folds its call site's literal arguments into its body, so one emitted function
    serves exactly one set of them. resnet101 calls ``_conv2d(x, w, 1, 0)`` and
    ``_conv2d(h, w, 2, 3)``; specialising on the first and calling it from the second would run a
    stride-1 body for a stride-2 call. So the second signature gets ``_conv2d__s2``, a verbatim copy
    whose own call sites point at it, and every later pass sees two ordinary helpers each with one
    consistent signature.

    ``owners`` are every function that may call ``hdef`` -- the kernel and each helper already built,
    with the descriptors of the arrays it names. The literals a built helper passes on exist only
    after its own call arguments were folded into it, so a helper is split just before it is built,
    once all its callers are.

    Keyed on constant arguments and on the DECLARED shape of any array argument the owner names,
    since the body is specialised on both. A local rebound to several shapes across the body still
    refuses: which shape reaches which call site is not decidable before lowering, so there is
    nothing to key on. Returns the clones, in order.
    """
    pnames = [a.arg for a in hdef.args.args]
    by_key: dict[tuple[tuple[str, ConstArg | tuple[str, ...]], ...], list[ast.Call]] = {}
    for owner_fn, arr_by in owners:
        for node in ast.walk(owner_fn):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == hdef.name):
                continue
            if len(node.args) != len(pnames) or node.keywords:
                return []  # an arity/keyword mismatch is a different failure; leave it be
            key = tuple((pn, call_arg_key(a, owner_fn, arr_by)) for pn, a in zip(pnames, node.args))
            by_key.setdefault(key, []).append(node)
    existing = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    clones: list[ast.FunctionDef] = []
    # The first key keeps the original name, so a helper called one way is untouched.
    for index, key in enumerate(list(by_key)[1:], start=2):
        name = f"{hdef.name}__s{index}"
        while name in existing:
            index += 1
            name = f"{hdef.name}__s{index}"
        existing.add(name)
        clone = copy.deepcopy(hdef)
        clone.name = name
        tree.body.append(clone)
        clones.append(clone)
        for site in by_key[key]:
            site.func.id = name
    if clones:
        ast.fix_missing_locations(tree)
    return clones


def substitute_names(node: ast.Expression, consts: dict[str, ast.expr]) -> ast.Expression:
    """Replace each ``Load`` use of a name in ``consts`` with its constant expr.

    Returns the (possibly replaced) root so a bare-Name ``node`` is not lost."""

    return SubstituteLoads(consts).visit(node)


def drop_unreachable_after_return(stmts: list[ast.stmt]) -> list[ast.stmt]:
    """Truncate ``stmts`` right after its first unconditional ``return``, recursing into every
    nested block (``If``/``For``/``While`` body + orelse) so the same trim applies there too.

    ``FoldStaticNoneBranches`` replaces a statically-true ``if None is None: return y`` with
    just its body (``[return y]``) -- but that only swaps the ``If`` NODE for its branch; the
    ORIGINAL SIBLINGS after it (``conv2d_instance_norm_divide``'s ``shape = ...; return
    y * None.reshape(shape) + None.reshape(shape)``, dead now that the guard is gone) are
    untouched and still reach the emitter, which has no lowering for a call on a substituted
    ``None``. A `return` appearing directly in a statement list is reached unconditionally
    whenever that list runs, so anything after it there can never execute -- safe to drop
    regardless of what runs before it.
    """
    out: list[ast.stmt] = []
    for stmt in stmts:
        map_blocks(stmt, drop_unreachable_after_return)
        out.append(stmt)
        if isinstance(stmt, ast.Return):
            break
    return out


def rewrite_returns_to_outparam(hfn: ast.FunctionDef, hret: str) -> None:
    """Rewrite every ``return <expr>`` into ``<hret>[:] = <expr>`` + a bare
    ``return`` -- so the whole-array return lowers like any slice assignment and
    the helper emits as a ``void`` out-param function."""

    class Ret(ast.NodeTransformer):
        def visit_Return(self, n: ast.Return) -> ast.stmt | list[ast.stmt]:
            if n.value is None:
                return n
            store = ast.Assign(
                targets=[
                    ast.Subscript(
                        value=name_(hret),
                        slice=ast.Slice(lower=None, upper=None, step=None),
                        ctx=ast.Store(),
                    )
                ],
                value=n.value,
            )
            bare = ast.Return(value=None)
            ast.copy_location(store, n)
            ast.copy_location(bare, n)
            return [store, bare]

    Ret().visit(hfn)
    ast.fix_missing_locations(hfn)


def literal_call_arg(arg: ast.expr) -> bool:
    """Whether a call argument is compile-time in the callee's body.

    A literal TUPLE is as compile-time as a scalar literal and has to be treated the same way:
    ``_adaptive_avg_pool3d(h3, (1, 1, 1), ...)`` guards its whole body on ``output_size == (1, 1, 1)``
    and indexes ``output_size[0]``. Left a plain parameter, the tuple branch never folds -- the helper
    then indexes what the call site typed as a scalar (refused outright), and its two surviving
    returns disagree on extent, which sizes the pool's out-param at its INPUT's shape.
    """
    if isinstance(arg, ast.Constant):
        return True
    return (
        isinstance(arg, (ast.Tuple, ast.List)) and bool(arg.elts) and all(isinstance(e, ast.Constant) for e in arg.elts)
    )


def literal_key(arg: ast.expr) -> ConstArg:
    """What two call sites have to AGREE on for one specialised body to serve both.

    A scalar literal keys on its VALUE, as it always did -- ``1`` and ``True`` are the same
    argument to a body that only branches on it. A literal tuple has no such equivalence to
    preserve, so it keys on its text.
    """
    return arg.value if isinstance(arg, ast.Constant) else ast.unparse(arg)


def bind_call_constants(hfn: ast.FunctionDef, consts: dict[str, ast.expr]) -> None:
    """Give the helper body each compile-time call argument, then prune what that makes dead.

    A parameter the body REASSIGNS is SEEDED with a leading ``param = <const>`` instead of being
    substituted. Substituting one drops its reassignment on the floor: ``_conv2d``'s
    ``stride = _as_tuple(stride, 2)`` rebinds ``stride``, so with ``1`` already pasted over every
    read, ``stride[0]`` had become ``1[0]`` and the tuple the rebinding built was never seen --
    and ``_adaptive_avg_pool2d``'s ``oh, ow = output_size`` read the raw ``1`` rather than the
    ``(1, 1)`` its own ``isinstance`` guard had just built from it.
    """
    reassigned = collect_assigned_names(hfn.body)
    direct = {name: value for name, value in consts.items() if name not in reassigned}
    seeded = [name for name in consts if name in reassigned]
    if direct:
        substitute_names(hfn, direct)
    for name in reversed(seeded):
        hfn.body.insert(0, ast.Assign(targets=[store_(name)], value=copy.deepcopy(consts[name])))
    if consts:
        FoldStaticNoneBranches().visit(hfn)
        # Same pruning one step wider: the substitution decides ordinary guards too, and
        # ``_conv2d``'s ``if kh == 1 and kw == 1 and stride == 1 and padding == 0`` is the shape
        # it leaves behind. An undecided guard is left exactly as it is.
        fold_constant_branches(hfn)
        hfn.body = drop_unreachable_after_return(hfn.body)
        ast.fix_missing_locations(hfn)
