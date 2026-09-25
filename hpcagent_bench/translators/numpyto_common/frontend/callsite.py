"""Kept helpers: caller-side extents, call-site statements and ABI argument order."""

import ast
import copy
from collections.abc import Iterator, Sequence

from hpcagent_bench.translators.numpyto_common.ir import ArrayDesc, KernelIR
from hpcagent_bench.translators.numpyto_common.frontend.helper_specialize import substitute_names
from hpcagent_bench.translators.numpyto_common.frontend.inlining import collect_assigned_names
from hpcagent_bench.translators.numpyto_common.frontend.shape_arith import literal_axis, fold_shape_expr
from hpcagent_bench.translators.numpyto_common.emit_helpers.tokens import IDENT_RE


def value_names(node: ast.AST) -> set[str]:
    """Every name an expression reads as a VALUE -- a call's callee is not one of them.

    ``int`` in ``(height - 1) // int(conv_stride) + 1`` is the builtin, not an extent the caller
    has to hold; counting it made an otherwise fully-resolved shape look unresolvable.
    """
    names: set[str] = set()
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Call):
            for arg in list(child.args) + [kw.value for kw in child.keywords]:
                names |= value_names(ast.Expression(body=arg)) | ({arg.id} if isinstance(arg, ast.Name) else set())
        else:
            names |= value_names(child)
    return names


def held_before_table(owner_fn: ast.FunctionDef) -> dict[int | None, set[str]]:
    """``{id(node): names bound before the statement holding it}``; ``None`` -> all of them."""
    table: dict[int | None, set[str]] = {}
    held: set[str] = set()
    for stmt in owner_fn.body:
        for node in ast.walk(stmt):
            table.setdefault(id(node), held)
        held = held | set(collect_assigned_names([stmt]))
    table[None] = held
    return table


def shape_symbols(arrays: list[ArrayDesc]) -> set[str]:
    """Free identifiers appearing in array-param shape expressions (``ngm`` in a
    ``(3, ngm)`` shape) -- the symbols a helper must receive to size its loops."""
    syms: set[str] = set()
    for a in arrays:
        for tok in a.shape:
            try:
                for node in ast.walk(ast.parse(str(tok), mode="eval")):
                    if isinstance(node, ast.Name):
                        syms.add(node.id)
            except SyntaxError:
                pass
    return syms


#: How far :func:`caller_side_symbol` chases a shape symbol through the helper's own locals before
#: giving up. A shape scalar built from one or two extents is the whole population; a deeper chain is
#: a helper doing real work in its dimensions, which wants shape-generic parameters, not a longer
#: substitution.
EXTRA_SYM_DEPTH = 4


def sole_local_binding(hfn: ast.FunctionDef, name: str) -> ast.expr | None:
    """The one expression ``name`` is bound to in ``hfn``, or ``None`` if it is not bound exactly once.

    By a plain ``name = <expr>``, and to ONE value: a name a loop rebinds, or two branches give
    genuinely different values, has nothing the caller could compute ahead of the call. Several
    bindings that FOLD to the same expression are one value, though -- resnet101's ``_conv2d`` binds
    ``oh`` as ``(h + 6 - 7) // 2 + 1`` on one path and ``(h - 1) // 2 + 1`` on the other, which
    :func:`fold_shape_expr` shows are the same extent written twice.

    ``hfn``'s OWN scope only. A nested def is a separate scope that may reuse the name -- resnet101
    binds ``oh`` in ``_conv2d`` and again in the 1x1 helper nested inside it -- and an ``ast.walk``
    counts both, calls the name ambiguous, and refuses a call the caller could make.
    """
    found: list[ast.expr] = []
    for node in scope_nodes(hfn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                found.append(node.value)
        elif isinstance(node, (ast.AugAssign, ast.For)):
            bound = node.target
            if isinstance(bound, ast.Name) and bound.id == name:
                return None
    if not found:
        return None
    if len({fold_shape_expr(ast.unparse(value)) for value in found}) != 1:
        return None
    return found[0]


def scope_nodes(fn: ast.FunctionDef) -> Iterator[ast.AST]:
    """Every node in ``fn``'s own scope: like :func:`ast.walk`, but never descending into a nested
    ``def``/``lambda``, whose names belong to a different scope."""
    stack: list[ast.AST] = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def fold_caller_shape_reads(text: str, arr_by: dict[str, "ArrayDesc"] | None) -> str:
    """``a.shape[k]`` in a caller-side extent -> the extent the caller already declares for ``a``.

    Respelling a helper's local through its call-site argument puts a shape READ back into the
    caller's vocabulary: eigh_test's ``n = m.shape[0]`` with ``m`` bound to ``a`` came back as
    ``a.shape[0]``. A read is not a value any ABI carries (see :func:`resolve_shape_reads`), and
    left standing it is not one bad token but two bad SLOTS -- ``shape_symbols`` reads ``a`` out
    of it and the emitter reads ``shape``, so the signature grew two parameters no call passes.

    Only a literal axis off a Name the caller declares is folded; anything else keeps its text, so
    an unresolved read still reaches the pass that owns its refusal.
    """
    if not arr_by or ".shape" not in text:
        return text

    class Fold(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            base = node.value
            if isinstance(base, ast.Attribute) and base.attr == "shape" and isinstance(base.value, ast.Name):
                desc = arr_by.get(base.value.id)
                axis = literal_axis(node.slice)
                if desc is not None and axis is not None and -len(desc.shape) <= axis < len(desc.shape):
                    return ast.copy_location(ast.parse(str(desc.shape[axis]), mode="eval").body, node)
            self.generic_visit(node)
            return node

    try:
        return ast.unparse(Fold().visit(ast.parse(text, mode="eval")).body)
    except SyntaxError:
        return text


def caller_side_symbol(
    sym: str,
    held: set[str],
    decl_pnames: list[str],
    site_args: list[ast.expr],
    hfn: ast.FunctionDef,
    hname: str,
    depth: int = 0,
    arr_by: dict[str, "ArrayDesc"] | None = None,
) -> str:
    """The source ONE call site passes for a shape symbol the helper receives but the caller never named.

    ``extra_syms`` is the helper's own vocabulary: a parameter the constant-fold pruned
    (``_maxpool1d``'s ``c``, passed ``1`` at its only site) or a local its body binds (``out_len``).
    Emitting one by NAME would put an identifier into the CALLER that nothing there declares, and
    ``promote_free_names_to_params`` would rescue it exactly as it rescues a genuine free parameter
    -- as a scalar int in the ABI: an emitted signature carrying ``n`` / ``c`` / ``out_len`` slots
    the harness binding never passes, which a positional call cannot notice (every argument after
    the first extra slot is read from the wrong register).

    So the symbol is translated into caller vocabulary here instead. Most need no translation:
    ``infer_helper_params`` reads a helper's extents off its CALL SITE, so the majority of them are
    already the caller's own names (``batch_size``) and pass straight through -- ``held`` is what
    distinguishes those from the helper-only ones, and getting that backwards refuses a call the
    caller could make perfectly well.
    """
    if sym in held:
        return sym
    if depth > EXTRA_SYM_DEPTH:
        raise NotImplementedError(
            f"helper {hname!r} defines shape symbol {sym!r} through more than "
            f"{EXTRA_SYM_DEPTH} of its own locals; it needs shape-generic parameters"
        )
    by_param = dict(zip(decl_pnames, site_args))
    if sym in by_param:
        return ast.unparse(by_param[sym])

    binding = sole_local_binding(hfn, sym)
    if binding is None:
        raise NotImplementedError(
            f"helper {hname!r} needs shape symbol {sym!r}, which is neither one of its "
            f"parameters nor bound exactly once in its body; the caller has no expression "
            f"to pass for it"
        )
    expr = substitute_names(ast.Expression(body=copy.deepcopy(binding)), by_param).body
    # Whatever is still a helper-local after that substitution is another link in the same chain.
    locals_bound = collect_assigned_names(hfn.body)
    leftover = sorted(
        {n.id for n in ast.walk(expr) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} & set(locals_bound)
    )
    if leftover:
        nested = {
            nm: ast.parse(
                caller_side_symbol(nm, held, decl_pnames, site_args, hfn, hname, depth + 1, arr_by), mode="eval"
            ).body
            for nm in leftover
        }
        expr = substitute_names(ast.Expression(body=expr), nested).body
    if isinstance(expr, (ast.Tuple, ast.List, ast.Dict, ast.Set)):
        raise NotImplementedError(
            f"helper {hname!r} needs shape symbol {sym!r}, but the caller would have to pass "
            f"{ast.unparse(expr)} -- a shape slot takes one integer, not a sequence"
        )
    return fold_caller_shape_reads(ast.unparse(expr), arr_by)


def held_before(owner_fn: ast.FunctionDef, site: ast.stmt) -> set[str]:
    """Names the caller has BOUND by the time ``site`` runs.

    A local the caller computes is a perfectly good thing to pass -- squeezenet's kernel binds
    ``oh0`` itself and its ``_conv2d__s5`` descriptor is sized by that name -- so a ``held`` set of
    only parameters and descriptors refuses a call the caller could make.

    Strictly BEFORE, though. A name bound later in the body is not available at the call, and
    passing it reads it before it is written: the extent silently becomes whatever the buffer held,
    which is a wrong answer rather than an error.
    """
    table = held_before_table(owner_fn)
    return set(table.get(id(site), table[None]))


def caller_side_shape(
    tokens: Sequence[str],
    held: set[str],
    decl_pnames: list[str],
    site_args: list[ast.expr],
    hfn: ast.FunctionDef,
    hname: str,
    arr_by: dict[str, "ArrayDesc"] | None = None,
) -> list[str]:
    """``tokens`` -- one helper descriptor's shape -- respelled in the CALLER's vocabulary.

    The call is not the only place a helper's names reach the caller. :func:`build_callsite_stmts`
    also ALLOCATES the argument temps and the return buffer there, and it sized them straight off the
    helper's own descriptors: ``__hcall1 = np.empty((n, c, out_len))`` put three helper-only names
    into the caller's body, where ``promote_free_names_to_params`` rescued each as a scalar int
    parameter. Fixing only the argument list left this route open and the ABI still wrong -- the
    emitted signature grew the same slots, one statement later.
    """
    out: list[str] = []
    for token in tokens:
        expr = ast.parse(str(token), mode="eval").body
        # Only names the token reads as a VALUE. A call's CALLEE is not one: ``int`` in
        # ``(height - 1) // int(conv_stride) + 1`` is the builtin, and asking
        # :func:`caller_side_symbol` to respell it refused the whole call ("needs shape symbol
        # 'int'"). :func:`value_names` already draws that line for the free-name check below.
        operands = value_names(ast.Expression(body=expr))
        subst = {
            n.id: ast.parse(
                caller_side_symbol(n.id, held, decl_pnames, site_args, hfn, hname, 0, arr_by), mode="eval"
            ).body
            for n in ast.walk(expr)
            if isinstance(n, ast.Name) and n.id not in held and n.id in operands
        }
        respelled = ast.unparse(substitute_names(ast.Expression(body=expr), subst).body) if subst else str(token)
        # A token that needed no substitution can still BE a shape read -- one respelled at an
        # earlier site and carried here through the caller's own descriptor table.
        out.append(fold_caller_shape_reads(respelled, arr_by))
    return out


def build_callsite_stmts(
    lhs: ast.expr,
    name: str,
    pnames: list[str],
    kept_args: list[ast.expr],
    extra_srcs: list[str],
    param_info: dict[str, tuple[list[str], str]],
    hret_shape: list[str],
    hret_dtype: str,
    hidx: str,
    inout: bool = False,
    live_buffers: frozenset[str] = frozenset(),
) -> list[ast.stmt]:
    """Replacement statements for an array-returning helper call.

    Slice / non-bare array args are first materialised into contiguous temps (a
    strided column ``xk[:, k]`` cannot be passed as a flat pointer, and a slice in
    the call would otherwise trip the per-element slice lowering). ``extra_srcs`` are the
    helper's extra shape symbols already rendered in CALLER vocabulary by
    :func:`caller_side_symbol` -- appending the helper's own names here is what leaked them
    into the emitted ABI. A bare-array target is then filled in place (the emitter
    appends it as the out-param); a slice target fills a temp, then copies it in.

    ``inout`` says the target buffer is ALREADY one of ``kept_args``: the helper reads and
    writes one parameter, so it holds ONE ABI slot and the call passes the pointer once.
    Appending it a second time is what emitted ``_maxpool2d(h, h, n)`` -- two ``restrict``
    pointers to the same buffer, which is undefined behaviour, not a redundant argument.

    ``live_buffers`` names the parent's declared arrays. A bare target that is NOT one of them is a
    fresh local -- mlp's ``x = relu(input @ w1 + b1)``, squeezenet's ``__hcall1`` -- and passing it
    as the out-param without allocating it left the name bound to nothing: no Store anywhere, so
    ``promote_free_names_to_params`` rescued it as a scalar int PARAMETER. It then entered the
    emitted ABI the binding never passes, and the C call handed an integer to a pointer dummy
    (``passing argument 1 of 'build_up_b' makes pointer from integer without a cast``). Allocate it
    here instead, AFTER ``pre`` -- the argument temps read the target's previous value on a rebind
    (``x = relu(x @ w2 + b2)``), so an allocation ahead of them would clobber what they read.
    """
    pre: list[str] = []
    call_srcs: list[str] = []
    for k, (pn, arg) in enumerate(zip(pnames, kept_args)):
        info = param_info.get(pn)
        if info is not None and not isinstance(arg, ast.Name):
            shp, dt = info
            atmp = f"__harg_{hidx}_{k}"
            pre.append(f"{atmp} = np.empty(({', '.join(shp)},), dtype=np.{dt})")
            pre.append(f"{atmp}[:] = {ast.unparse(arg)}")
            call_srcs.append(atmp)
        else:
            call_srcs.append(ast.unparse(arg))
    call_srcs.extend(extra_srcs)
    # Built in ``input_args`` order; :func:`reorder_helper_call_args` permutes the whole call into
    # ABI order once every helper KernelIR exists. A BARE call statement (not ``tmp = h(...)``,
    # which would be seen as a whole-array reassignment and lowered element-wise). A bare-array
    # target is written in place; a slice target fills a fresh temp, then a normal slice copy
    # stores it.
    if isinstance(lhs, ast.Name):
        if not inout:
            # A target the call still READS is a rebinding of a buffer that already exists
            # (``x = relu(x @ w2 + b2)``); allocating it here would clear what the call is about to
            # read. Only a target nothing reads is a first binding that needs the buffer.
            reads = {ident for src in call_srcs for ident in IDENT_RE.findall(src)}
            call_srcs.append(lhs.id)
            if lhs.id not in live_buffers and lhs.id not in reads:
                pre.append(f"{lhs.id} = np.empty(({', '.join(hret_shape)},), dtype=np.{hret_dtype})")
        return ast.parse("\n".join(pre + [f"{name}({', '.join(call_srcs)})"])).body
    tmp = f"__hret_tmp_{hidx}"
    call_srcs.append(tmp)
    lines = pre + [
        f"{tmp} = np.empty(({', '.join(hret_shape)},), dtype=np.{hret_dtype})",
        f"{name}({', '.join(call_srcs)})",
        f"{ast.unparse(lhs)} = {tmp}",
    ]
    return ast.parse("\n".join(lines)).body


def reorder_helper_call_args(trees: list[ast.AST], helpers: list[KernelIR]) -> None:
    """Permute every surviving-helper call from source order into ``KernelIR.param_order()`` order.

    This is the only place a helper's parameter NAMES and its call-site argument EXPRESSIONS are
    both in hand -- downstream every emitter sees positional AST nodes with the names gone. Doing
    it here makes the definition (which reads ``param_order()`` too) and the call read one
    ordering function, and reaches C, C++, Fortran, Pluto and DaCe at once since all five render
    this same tree. Two transposed same-typed pointers compile clean, so a second implementation
    of the order would not be caught by any compiler.
    """
    perms: dict[str, list[int]] = {}
    for h in helpers:
        # abi_param_order, not param_order: a helper carrying a parameter the descriptor lists do not
        # cover (kl_div's `reduction` config flag) falls back to declaration order rather than losing
        # it. The emitters read the same method, so definition and call stay in step.
        order = h.abi_param_order()
        if order == h.input_args:
            continue
        slot = {name: i for i, name in enumerate(h.input_args)}
        if set(order) != set(slot):
            raise ValueError(f"helper {h.kernel_name}: ABI order {order} is not a permutation of {h.input_args}")
        perms[h.kernel_name] = [slot[name] for name in order]
    if not perms:
        return
    for tree in trees:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            perm = perms.get(node.func.id)
            if perm is None:
                continue
            # Arity is the only check available here -- the names are gone by now -- so it must be
            # a hard error, not a skip. A call still spelling the helper's ORIGINAL signature can
            # match this length by coincidence, and permuting it then produces a call whose
            # arguments are unrelated to the parameters they land on. Every site is rewritten to
            # `input_args` order in `build_helper_kirs`, so a mismatch here is a defect upstream.
            if len(node.args) != len(perm):
                raise NotImplementedError(
                    f"helper {node.func.id!r}: call site has {len(node.args)} args but the "
                    f"ABI order has {len(perm)}; the call was not rewritten to the helper ABI"
                )
            node.args = [node.args[i] for i in perm]


class ReplaceStmts(ast.NodeTransformer):
    """Replace specific ``Assign`` nodes (keyed by ``id``) with a stmt list."""

    def __init__(self, mapping: dict[int, list[ast.stmt]]) -> None:
        self.mapping = mapping

    def visit_Assign(self, node: ast.Assign) -> ast.stmt | list[ast.stmt]:
        repl = self.mapping.get(id(node))
        if repl is None:
            return node
        for s in repl:
            ast.copy_location(s, node)
            ast.fix_missing_locations(s)
        return repl
