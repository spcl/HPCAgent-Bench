"""One :class:`KernelIR` per helper that survives inlining."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.ir import ArrayDesc, KernelIR, ScalarDesc, SymbolDesc
from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet
from hpcagent_bench.translators.numpyto_common.frontend.axes import reject_symbolic_axis, reject_unsupported_slices
from hpcagent_bench.translators.numpyto_common.frontend.body_rewrites import native_desugar
from hpcagent_bench.translators.numpyto_common.frontend.callsite import (
    ReplaceStmts,
    build_callsite_stmts,
    caller_side_shape,
    caller_side_symbol,
    held_before,
    reorder_helper_call_args,
    shape_symbols,
)
from hpcagent_bench.translators.numpyto_common.frontend.helper_params import (
    DescKey,
    infer_helper_params,
    mark_written_outputs,
    reject_subscripted_scalar_params,
    widen_counting_scalar_params,
)
from hpcagent_bench.translators.numpyto_common.frontend.helper_shapes import (
    desc_key,
    helper_return_array_shape,
    helper_return_shape_from_body,
    structure_key,
    call_specialized_body,
    helper_call_local_arrays,
    helper_returns_rank0,
    target_shape_is_the_call_itself,
)
from hpcagent_bench.translators.numpyto_common.frontend.helper_specialize import (
    bind_call_constants,
    literal_call_arg,
    literal_key,
    rewrite_returns_to_outparam,
    specialise_helpers_by_call_signature,
)
from hpcagent_bench.translators.numpyto_common.frontend.inlining import HoistMultiStmtHelpers, unroll_const_list_loops
from hpcagent_bench.translators.numpyto_common.frontend.module_constants import inline_module_constants
from hpcagent_bench.translators.numpyto_common.frontend.shapes import local_array_def, conflicting_rebind_shapes
from hpcagent_bench.translators.numpyto_common.frontend.tuple_helpers import (
    InlineTupleHelperCalls,
    desugar_helper_tuples,
    fold_call_arg_constant,
    rewrite_helper_axes,
    tuple_template_for_call,
)


def collect_called_helper_defs(tree: ast.Module, kernel_fn: ast.FunctionDef) -> list[ast.FunctionDef]:
    """Top-level helper ``FunctionDef``s still CALLED after inlining -- the ones
    inlining could not absorb (an early ``return`` / recursion). Collected
    transitively: a captured helper may call another non-inlinable helper, which
    must be emitted too. Returned in definition order (a callee defined above its
    caller emits first, so no forward declaration is needed)."""
    defs_by_name: dict[str, ast.FunctionDef] = {
        n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n is not kernel_fn
    }
    captured: dict[str, ast.FunctionDef] = {}
    frontier: list[ast.AST] = [kernel_fn]
    while frontier:
        node = frontier.pop()
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id in defs_by_name
                and sub.func.id not in captured
            ):
                d = defs_by_name[sub.func.id]
                captured[sub.func.id] = d
                frontier.append(d)
    # Definition order (as they appear in the module), so a helper that calls
    # another emits after its callee.
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in captured]


def helper_call_sites(
    fn: ast.FunctionDef,
) -> tuple[dict[str, ast.Call], dict[str, ast.Assign], dict[str, list[ast.Assign]]]:
    """Helper call sites inside ONE scope: the first call of each name, its enclosing assignment
    (``X = h(...)`` / ``X[:, j] = h(...)`` -- the LHS classifies the return, array out-param vs
    by-value scalar, and sizes it), and EVERY ``X = h(...)`` site.

    Rewriting only the first left the others spelling the helper's ORIGINAL signature while the
    definition had moved to the out-param ABI; the arity happened to still match, so the reorder
    permuted unrelated slots and emitted a call that does not compile (vgg16's
    ``_maxpool2d(2, h[...], 2)``)."""
    call_of: dict[str, ast.Call] = {}
    assign_of: dict[str, ast.Assign] = {}
    assigns_of: dict[str, list[ast.Assign]] = {}
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ):
            assigns_of.setdefault(node.value.func.id, []).append(node)
            if node.value.func.id not in call_of:
                call_of[node.value.func.id] = node.value
                assign_of[node.value.func.id] = node
    for node in ast.walk(fn):  # plain-call fallback (scalar helper in an expression)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in call_of:
            call_of[node.func.id] = node
    return call_of, assign_of, assigns_of


def helpers_callers_first(helper_defs: list[ast.FunctionDef], kernel_fn: ast.FunctionDef) -> list[ast.FunctionDef]:
    """``helper_defs`` reordered so a helper is visited before every helper it calls.

    A helper's parameters are inferred from a call site, and when the only site sits in a SIBLING
    that sibling's descriptor tables are the resolution scope -- which exist only once the sibling
    itself has been built. Independent helpers keep definition order. The emission order the C
    backend needs (callee defined above its caller, so no forward declaration) is a property of
    ``out``, restored by sorting it back at the end of :func:`build_helper_kirs`."""
    names = {h.name: h for h in helper_defs}
    calls = {
        h.name: sorted(
            {
                n.func.id
                for n in ast.walk(h)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in names
                and n.func.id != h.name
            }
        )
        for h in helper_defs
    }
    ordered: list[ast.FunctionDef] = []
    placed: OrderedSet[str] = OrderedSet()

    def visit(h: ast.FunctionDef) -> None:
        if h.name in placed:
            return
        placed.add(h.name)
        ordered.append(h)
        for callee in calls[h.name]:
            visit(names[callee])

    # Seeded from what the KERNEL calls, not from definition order: helpers are conventionally
    # written callee-above-caller, so walking definition order visits a sibling-only callee before
    # the caller whose tables resolve it, and it is skipped for having no reachable call site --
    # which is the very refusal this ordering exists to remove.
    for node in ast.walk(kernel_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names:
            visit(names[node.func.id])
    for h in helper_defs:
        visit(h)
    return ordered


def abi_hostile_arguments(tree: ast.Module, hname: str) -> list[str]:
    """Constants a call site binds to ``hname`` that no C or Fortran parameter table has a type for.

    ``None`` is not a value in either language and neither is a string selector, so a helper called
    as ``_reduce(v, mode='total')`` or ``_default_stride(None, k)`` has no standalone ABI however
    well its body lowers -- the argument only exists to be resolved at the call site. Both forms are
    decidable exactly BECAUSE the site pins them to a literal, which is what the inline path does
    with them; refusing here hands the kernel to that path rather than emitting a signature naming a
    parameter the caller cannot pass.
    """
    seen: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == hname):
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Constant) and (arg.value is None or isinstance(arg.value, str)):
                spelling = repr(arg.value)
                if spelling not in seen:
                    seen.append(spelling)
    return seen


def build_helper_kirs(
    tree: ast.Module, kernel_fn: ast.FunctionDef, parent: KernelIR, keep_helpers: bool
) -> list[KernelIR]:
    """One :class:`KernelIR` per non-inlinable called helper (see
    :func:`collect_called_helper_defs`). Each helper param's type/shape is read
    off the FIRST call site's argument via :func:`infer_param_desc`; module
    constants (``_THRESH = 5.0``) are inlined into the helper body. The return is
    classified scalar (by-value) or array (out-param, added as a leading param).

    Only DIRECT kernel-body call sites are resolved here (args refer to the
    kernel's own params); a helper called only from another helper is skipped
    (left for a later pass) so we never infer against the wrong scope.
    """
    helper_defs = collect_called_helper_defs(tree, kernel_fn)
    if not helper_defs:
        return []
    # Every rewrite below keys off a call site that is the direct RHS of an assignment: that is
    # where the result's target lives, and the target is what says whether the return is an array.
    # A helper called only as another call's ARGUMENT (resnet101's ``_batch_norm(_conv2d(..), ..)``)
    # has no such site, so it was classified by-value and its shape-changing calls were left inside
    # a bare ``return`` where no expander reaches them. Lift each nested call into its own
    # assignment first -- the same lift the INLINE path already performs, for the same reason.
    arr_by = {a.name: a for a in parent.arrays}
    HoistMultiStmtHelpers({h.name: h for h in helper_defs}).visit(kernel_fn)
    ast.fix_missing_locations(kernel_fn)
    if specialise_helpers_by_call_signature(tree, kernel_fn, helper_defs, arr_by):
        helper_defs = collect_called_helper_defs(tree, kernel_fn)
    out: list[KernelIR] = []
    #: {id(owner tree): {id(Assign): replacement stmts}} -- applied per owner, because a helper's
    #: call sites are not all in the kernel body.
    callsite_rewrites: dict[int, dict[int, list[ast.stmt]]] = {}
    #: Every scope a helper call can live in, paired with the descriptors its arguments resolve
    #: against: the kernel body, then each helper as it is built. A helper reached only from a
    #: SIBLING (lulesh's ``_calc_force_for_nodes``, called from ``_lagrange_nodal``) has no
    #: kernel-body call at all, and resolving its arguments against the kernel's tables would read
    #: the wrong scope. Helpers are visited callers-first, so the owner is already registered.
    scopes: list[tuple[ast.FunctionDef, list[ArrayDesc], list[ScalarDesc], list[SymbolDesc]]] = [
        (kernel_fn, parent.arrays, parent.scalars, parent.symbols)
    ]
    #: Memo for the chase below; ``generation`` retires entries whose HELPER trees have moved on.
    local_arrays: dict[tuple[str, int, DescKey, DescKey, DescKey], dict[str, ArrayDesc]] = {}
    generation = 0

    def rewrote(owner: ast.FunctionDef) -> None:
        """``owner``'s tree changed: renumber it and retire the memo."""
        nonlocal generation
        generation += 1
        ast.fix_missing_locations(owner)

    hidx_of = {id(h): i for i, h in enumerate(helper_defs)}
    for hdef in helpers_callers_first(helper_defs, kernel_fn):
        hidx = hidx_of[id(hdef)]
        for owner_fn, oarrays, oscalars, osymbols in scopes:
            call_of, assign_of, assigns_of = helper_call_sites(owner_fn)
            if hdef.name in call_of:
                break
        else:
            # Not called from the kernel or from any helper built so far -- nothing reaches it.
            continue
        # Only while BUILDING the kept form: this runs again on the inlined pass, over whatever
        # resisted inlining, and there a refusal has no fallback left: it would be a hard failure.
        hostile = abi_hostile_arguments(tree, hdef.name) if keep_helpers else []
        if hostile:
            raise NotImplementedError(
                f"helper {hdef.name!r} is called with {hostile}, which no ABI carries; "
                f"it must be inlined into its caller"
            )
        oarr_by = {a.name: a for a in oarrays}
        osca_by = {s.name: s for s in oscalars}
        osym_by = {s.name: s for s in osymbols}
        # A local this scope binds from ANOTHER helper's call carries a shape too, and only the
        # callee's return says what it is -- see :func:`helper_call_local_arrays`. Resolution only:
        # these never join ``oarrays``, which is the emitted ABI, and never ``live_buffers``, which
        # is what says a target already HAS a buffer -- one of these does not yet, and suppressing
        # its allocation is what put an unbound name into the ABI as a scalar int.
        local_key = (
            structure_key(owner_fn),
            generation,
            desc_key(oarr_by),
            desc_key(osca_by),
            desc_key(osym_by),
        )
        chased = local_arrays.get(local_key)
        if chased is None:
            chased = helper_call_local_arrays(owner_fn, helper_defs, oarr_by, osca_by, osym_by)
            local_arrays[local_key] = chased
        # Fresh copy: the descriptors are mutable and consumers mark ``is_output`` on them.
        oarr_by.update(copy.deepcopy(chased))
        call = call_of[hdef.name]
        assign = assign_of.get(hdef.name)
        lhs = assign.targets[0] if assign is not None else None
        hret_shape, hret_dtype = helper_return_array_shape(lhs, oarr_by, owner_fn)
        # Every extent this helper is built from comes off the first call site, so a call-site
        # array whose name is rebound to a different shape elsewhere in the body makes the whole
        # inference unsound -- see :func:`conflicting_rebind_shapes`.
        for node in ([lhs] if lhs is not None else []) + list(call.args):
            clash = conflicting_rebind_shapes(owner_fn, node, oarr_by, ignore=assign)
            if clash is not None:
                raise NotImplementedError(
                    f"helper {hdef.name!r} is called on {node.id!r}, which is rebound to both {clash[0]} and "
                    f"{clash[1]}; the helper's extents are emitted as constants and cannot serve both"
                )
        hfn = copy.deepcopy(hdef)
        pnames = [a.arg for a in hfn.args.args]
        # The parent's folded names carry over: this helper's array params reuse the
        # parent's (already folded) shapes, so neither set may be re-promoted.
        hconsts = set(parent.inlined_consts) | set(inline_module_constants(tree, hfn, pnames))
        # Same native-backend desugars the kernel body already ran (BUG-3: a helper
        # that survives inlining kept its ``np.newaxis`` / ufunc-``out=`` / roll-on-
        # slice / ``.real`` / ``.ndim``-guard forms). Runs before ``_mark_written_
        # outputs`` so a ufunc-out / roll rewrite is seen as a write to its target.
        native_desugar(hfn)
        # Same const-list unroll the kernel body gets: ``for k in [0, 1, 2, 3]`` has no native
        # form, and a helper that is NOT inlined never passed through the caller-side pass that
        # consumes it (lulesh's face-node loops, which only surface once its helpers survive).
        unroll_const_list_loops(hfn)

        if hret_shape is None or target_shape_is_the_call_itself(owner_fn, lhs, oarr_by, hdef.name):
            # Either no call site stores the result into an array -- ``_conv2d(...)`` is only ever
            # an ARGUMENT to another helper (resnet101's ``_batch_norm(_conv2d(x, w, 1, 0), ..)``)
            # -- or the target told us nothing the call did not. The helper's own body says what it
            # returns, and reading that wrong classifies an array return as by-value: no out-param
            # is added, the returns stay as ``return <expr>``, and every shape-changing call inside
            # one reaches the emitter unlowered, because the expanders only ever see assignments.
            probe = call_specialized_body(hfn, pnames, call.args)
            body_shape, body_dtype = helper_return_shape_from_body(
                probe, pnames, call.args, oarr_by, osca_by, osym_by, owner_fn
            )
            # ``None`` from the body means two different things and they want opposite decisions:
            # "this returns a scalar" and "this could not be sized". Only the first may overrule a
            # target-side guess. A helper whose body PROVABLY returns rank 0 is a reduction (array
            # in, scalar out) -- bdf_newton_krylov's WRMS norms, jfnk_bratu's 2-norms and dot
            # products -- and keeping the guess there (the broadcast join of the call's own
            # arguments) classified it as array-returning: the caller allocated an operand-shaped
            # buffer and the call was broadcast over it, one invocation per element of the very
            # array it reduces, which is a double handed to a pointer parameter.
            if (
                body_shape is not None
                or hret_shape is None
                or helper_returns_rank0(probe, pnames, call.args, oarr_by, osca_by, osym_by, owner_fn)
            ):
                hret_shape, hret_dtype = body_shape, body_dtype

        if hret_shape is None:
            # SCALAR (by-value) return -- params inferred straight from the call. A compile-time
            # call-site arg is substituted into the body first, same as the array-return branch
            # below: ``_as_tuple(value, dims)``'s ``dims`` is a compile-time constant at every
            # call site (a literal, or an expression like ``x.ndim - 2`` that folds to one against
            # the kernel's own rank table), but is left a plain parameter Name unless substituted
            # here, and ``tuple(value for _ in range(dims))`` cannot resolve its trip count off a
            # Name. Params are NOT dropped afterwards (unlike the array branch): this call site is
            # a real ``ast.Call`` sitting wherever the kernel body already put it, not one this
            # function rewrites, so the signature must keep every argument slot the call passes.
            call_consts = {}
            for pn, a in zip(pnames, call.args):
                if literal_call_arg(a):
                    call_consts[pn] = a
                    continue
                folded = fold_call_arg_constant(a, oarrays, oscalars, osymbols)
                if folded is not None:
                    call_consts[pn] = folded
            bind_call_constants(hfn, call_consts)
            arrays, scalars, symbols = infer_helper_params(pnames, call.args, oarr_by, osca_by, osym_by, owner_fn)
            # Fold this helper's OWN compile-time tuples (``tuple(range(2, x.ndim))`` and the
            # rest of tuple_desugar.py) against ITS param ranks, same as the kernel body got at
            # ``parse_kernel``'s own ``desugar_tuples`` call -- a surviving helper is its own
            # KernelIR and never went through that call. Must run BEFORE the structural-axis
            # guards below: an unfolded ``axes = tuple(range(2, x.ndim))`` is still a runtime
            # Call at that point, which is exactly the "symbolic axis" the guard exists to catch.
            rewrite_helper_axes(hfn, arrays, scalars)
            desugar_helper_tuples(hfn, arrays, scalars, symbols)
            # A helper whose folded body is nothing but ``return (a, b, ...)`` has no C/Fortran
            # ABI -- there is no tuple return value -- so it is not emitted as a function at all.
            # Splice its (per-call-substituted) result into every call site instead, then re-run
            # the kernel's own tuple fold so a use like ``stride[0]`` resolves against the spliced
            # elements exactly as it would against a source-level ``stride = (s, s)``. Declines
            # (keeps the function) when some call site does not match this helper's arity /
            # keyword shape -- that call still needs a real function to reach, so nothing here
            # may delete it.
            # Every tree that calls this helper, paired with the scope its arguments resolve
            # against. A SIBLING helper's call is spliced like any other: a tuple return has no ABI
            # ANYWHERE, so declining those left the helper a real function that nothing could reach
            # and no program was emitted at all. ``_tap_span`` is called only from
            # ``_conv_transpose3d``, never from the kernel body, which is the whole of why three
            # conv_transpose kernels refused. The owner is the resolution scope and not
            # ``kernel_fn``: at a sibling's call site the arguments are that sibling's own locals.
            # A sibling ALREADY built is represented by its DEEPCOPY, not by the original node in
            # ``helper_defs``: splicing into the original lands in a tree nothing emits.
            built = {ir.kernel_name: ir.tree for ir in out}
            owners = [kernel_fn] + [built.get(h.name, h) for h in helper_defs if h is not hdef]
            scope_of = {id(fn): (a, sc, sy) for fn, a, sc, sy in scopes}
            default_scope = (parent.arrays, parent.scalars, parent.symbols)
            calls = [
                (owner, n)
                for owner in owners
                for n in ast.walk(owner)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == hdef.name
            ]
            templates: dict[int, ast.expr] = {}
            if calls and all(not c.keywords and len(c.args) == len(pnames) for unused, c in calls):
                for owner, site in calls:
                    oa, osc, osy = scope_of.get(id(owner), default_scope)
                    template = tuple_template_for_call(
                        hdef,
                        site,
                        tree,
                        parent,
                        {a.name: a for a in oa},
                        {s.name: s for s in osc},
                        {s.name: s for s in osy},
                        owner,
                    )
                    if template is None:
                        templates.clear()
                        break
                    templates[id(site)] = template
            if templates:
                for owner in {id(o): o for o, unused in calls}.values():
                    InlineTupleHelperCalls(pnames, templates).visit(owner)
                    # An owner already BUILT never passes through this loop again, so the tuples
                    # just spliced into it are folded here or not at all; one not yet built folds
                    # them against its own tables on its own pass, as it always did.
                    oa, osc, osy = scope_of.get(id(owner), default_scope)
                    desugar_helper_tuples(owner, oa, osc, osy)
                    reject_symbolic_axis(owner)
                    reject_unsupported_slices(owner)
                    rewrote(owner)
                continue
            # The splice above declined, so this helper has to become a real function -- and a
            # helper that returns SEVERAL values cannot: C has one return slot and nothing here
            # classifies which member is an out-param. Refuse, so a level-3 kernel retries with
            # inlining on (:func:`parse_kernel`) and everything else reports the reason here
            # instead of at emit, where nothing retries.
            if isinstance(lhs, ast.Tuple) or any(
                isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple) for n in ast.walk(hfn)
            ):
                raise NotImplementedError(
                    f"helper {hdef.name!r} returns a tuple; it has no standalone ABI and "
                    f"must be inlined into its caller"
                )
            # Reaching the by-value branch means the call site's target did not resolve to an array.
            # If the helper nonetheless returns one of its OWN allocations, the classification is
            # wrong: the value belongs in an out-param, and returning it by value hands back a
            # pointer where a double is declared (and, once the helper frees its locals on exit, a
            # dangling one). The C emitter catches this at the return; refuse here instead, so
            # Fortran -- which has no such check and emitted a subroutine gfortran rejects with
            # "VALUE attribute conflicts with FUNCTION attribute" -- is covered by the same rule.
            # The helper's OWN inferred params are the scope a local allocation resolves against:
            # ``np.zeros(..., dtype=x.dtype)`` names a parameter, and an empty table made that a
            # hard refusal ("the dtype expression does not resolve") from inside a guard whose
            # only question is whether the return is a local allocation at all.
            harr_by = {a.name: a for a in arrays}
            for n in ast.walk(hfn):
                if (
                    isinstance(n, ast.Return)
                    and isinstance(n.value, ast.Name)
                    and local_array_def(hfn, n.value.id, harr_by) is not None
                ):
                    raise NotImplementedError(
                        f"helper {hdef.name!r} returns its own array {n.value.id!r} by value; the call site's "
                        f"target did not resolve to an array, so there is no out-param to write it into"
                    )
            # A helper that survives inlining is emitted as its OWN kernel and lowered through the
            # same expanders, so it needs the same structural guards the kernel body gets. Without
            # them a symbolic axis inside a helper reached lowering and was read as "no axis" --
            # the instance-norm helpers reduced over EVERY axis instead of the spatial ones.
            reject_symbolic_axis(hfn)
            reject_unsupported_slices(hfn)
            reject_subscripted_scalar_params(hfn, scalars, hdef.name)
            widen_counting_scalar_params(hfn, scalars)
            mark_written_outputs(hfn, arrays)
            # Shape symbols this helper's array params name (``ny``/``nx`` in cavity_flow's
            # ``(ny, nx)``) but the call does not pass. The emitters size the dummy's dimensions
            # from them, so they ARE parameters of the emitted function -- leaving them out of the
            # descriptor lists put them in the definition and not in the call, which does not
            # compile ("too few arguments to function 'build_up_b'"). Declare them and append them
            # to every call site, in one fixed order; :func:`reorder_helper_call_args` then
            # permutes definition and call into the same ABI order as for any other parameter.
            extra_syms = sorted(s for s in shape_symbols(arrays) if s not in set(pnames))
            if extra_syms:
                # ``calls`` pairs every call with the scope its arguments resolve against. A caller
                # can only pass a shape symbol it holds itself, so a sibling that does not declare
                # one cannot reach this helper -- refuse rather than emit a call naming an
                # identifier that is not in scope there. A caller holds a symbol its OWN array
                # shapes name even when its signature never declared it: those become parameters
                # of the emitted function by this very rule (conv2d_bias holds ``N``/``C_in``/
                # ``C_out`` only through ``input``'s ``(N, H, W, C_in)``).
                for owner, unused in calls:
                    oa, osc, osy = scope_of.get(id(owner), default_scope)
                    held = (
                        {sy.name for sy in osy}
                        | {a.name for a in oa}
                        | {sc.name for sc in osc}
                        | set(shape_symbols(oa))
                    )
                    absent = [sy for sy in extra_syms if sy not in held]
                    if absent:
                        raise NotImplementedError(
                            f"helper {hdef.name!r} needs shape symbols {absent}, which its caller does not hold; "
                            f"it must be inlined into its caller"
                        )
                symbols.extend(SymbolDesc(name=s) for s in extra_syms)
                for owner, site in calls:
                    site.args.extend(ast.Name(id=s, ctx=ast.Load()) for s in extra_syms)
                    rewrote(owner)
            out.append(
                KernelIR(
                    tree=hfn,
                    kernel_name=hdef.name,
                    short_name=hdef.name,
                    input_args=list(pnames) + extra_syms,
                    symbols=symbols,
                    arrays=arrays,
                    scalars=scalars,
                    source_path=parent.source_path,
                    inlined_consts=hconsts,
                    # A helper reached here because its result is not stored into an array.
                    # That is a by-value scalar return only when it actually RETURNS a value;
                    # a helper that writes through its array params and returns nothing is
                    # void. Fortran synthesizes a result dummy for "scalar" and C types the
                    # function by its return, so calling a void helper "scalar" put a
                    # parameter in the definition that no call site passes.
                    return_kind="scalar"
                    if any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(hfn))
                    else None,
                )
            )
            scopes.append((hfn, arrays, scalars, symbols))
            continue

        # ARRAY return: specialize the helper at its call site by folding every
        # literal arg into the body (``x_gamma_extrapolation`` -> ``False``) and
        # pruning the now-dead branches this exposes (config-only vcut/gamma
        # paths whose tuples & sibling-helper calls don't lower). Params left
        # unused are then dropped along with their call-site args, keeping
        # signature and call site aligned.
        call_consts = {pn: a for pn, a in zip(pnames, call.args) if literal_call_arg(a)}
        # ``bind_call_constants`` also prunes what the substitution makes dead: a statically-true
        # ``if None is None: return y`` leaves ORIGINAL siblings behind (conv2d_instance_norm_divide's
        # dead ``shape = ...; return y * None.reshape(...) + ...``) that still reference the
        # substituted-away ``None``, and ``used`` right below would otherwise still count
        # ``weight``/``bias`` as read there, keeping dead params alive.
        bind_call_constants(hfn, call_consts)
        used = {n.id for n in ast.walk(hfn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        keep = [(pn, a) for pn, a in zip(pnames, call.args) if pn in used]
        decl_pnames = list(pnames)  # before the unused-param prune, for binding later call sites
        pnames = [pn for pn, unused in keep]
        kept_args = [a for unused, a in keep]
        hfn.args.args = [a for a in hfn.args.args if a.arg in used]
        hfn.args.defaults = []
        arrays, scalars, symbols = infer_helper_params(pnames, kept_args, oarr_by, osca_by, osym_by, owner_fn)
        # See the scalar-return branch above: fold this helper's own compile-time tuples against
        # its param ranks BEFORE the structural-axis guards, and before ``hret`` (not yet a real
        # body reference) is appended to ``arrays`` below.
        rewrite_helper_axes(hfn, arrays, scalars)
        desugar_helper_tuples(hfn, arrays, scalars, symbols)
        reject_symbolic_axis(hfn)
        reject_unsupported_slices(hfn)
        reject_subscripted_scalar_params(hfn, scalars, hdef.name)
        widen_counting_scalar_params(hfn, scalars)
        mark_written_outputs(hfn, arrays)
        # ``X = h(X, ...)`` returns into a buffer the call ALREADY passes in. That parameter is
        # in-out and takes ONE ABI slot: appending a separate out-param would put the same pointer
        # in two ``restrict`` slots. Only sound when the two agree on shape -- a helper whose
        # dimensions are emitted as constants cannot read one extent and write another through a
        # single descriptor, so that case is refused rather than silently aliased.
        inout_param = None
        if isinstance(lhs, ast.Name):
            inout_param = next(
                (pn for pn, a in zip(pnames, kept_args) if isinstance(a, ast.Name) and a.id == lhs.id), None
            )
        if inout_param is not None:
            desc = next((a for a in arrays if a.name == inout_param), None)
            if desc is None or tuple(str(s) for s in desc.shape) != tuple(str(s) for s in hret_shape):
                got = tuple(desc.shape) if desc is not None else None
                raise NotImplementedError(
                    f"helper {hdef.name!r} returns into its own argument {inout_param!r}, but the argument is "
                    f"{got} and the result is {tuple(hret_shape)}; one in-out pointer cannot carry both extents "
                    f"while a helper's dimensions are emitted as constants"
                )
            desc.is_output = True
            hret = inout_param
        else:
            # The returned array becomes a trailing out-param the body writes into.
            hret = f"__hret_{hidx}"
            arrays.append(ArrayDesc(name=hret, dtype=hret_dtype, shape=tuple(hret_shape), is_output=True))
        # Shape symbols the helper's array params reference (``ngm`` in ``g``'s
        # ``(3, ngm)``) that are not already passed as args must be received too;
        # declare them here (so they are not re-promoted) and thread them into the
        # call in a fixed order.
        extra_syms = sorted(s for s in shape_symbols(arrays) if s not in set(pnames))
        # A caller can only pass a shape symbol it holds itself. The kernel body holds every
        # declared symbol; a SIBLING helper holds only what its own signature received, so one
        # missing name would emit a call naming an identifier that is not in scope there.
        if extra_syms and owner_fn is not kernel_fn:
            held = (
                {sy.name for sy in osymbols}
                | {a.name for a in oarrays}
                | {sc.name for sc in oscalars}
                | set(shape_symbols(oarrays))
            )
            absent = [sy for sy in extra_syms if sy not in held]
            if absent:
                raise NotImplementedError(
                    f"helper {hdef.name!r} needs shape symbols {absent}, which its calling "
                    f"helper does not hold; it must be inlined into its caller"
                )
        symbols.extend(SymbolDesc(name=s) for s in extra_syms)
        rewrite_returns_to_outparam(hfn, hret)
        out.append(
            KernelIR(
                tree=hfn,
                kernel_name=hdef.name,
                short_name=hdef.name,
                input_args=list(pnames) + extra_syms + ([] if inout_param is not None else [hret]),
                symbols=symbols,
                arrays=arrays,
                scalars=scalars,
                source_path=parent.source_path,
                inlined_consts=hconsts,
                return_kind=hret,
            )
        )
        scopes.append((hfn, arrays, scalars, symbols))
        if assign is not None:
            param_info = {a.name: (a.shape, a.dtype) for a in arrays if a.name != hret}
            for sidx, site in enumerate(assigns_of.get(hdef.name, [assign])):
                if not isinstance(site.value, ast.Call):
                    continue
                site_args = site.value.args
                if len(site_args) != len(decl_pnames):
                    raise NotImplementedError(
                        f"helper {hdef.name!r} is called with {len(site_args)} args at one "
                        f"site and declares {len(decl_pnames)}; the call sites disagree"
                    )
                # The body was SPECIALIZED against the first site's literal args, so a site passing a
                # different constant cannot call it. Refuse rather than emit a call to a body
                # specialized for someone else.
                site_consts = {pn: literal_key(a) for pn, a in zip(decl_pnames, site_args) if literal_call_arg(a)}
                first_consts = {pn: literal_key(a) for pn, a in call_consts.items() if literal_call_arg(a)}
                if site_consts != first_consts:
                    raise NotImplementedError(
                        f"helper {hdef.name!r} is specialized on {first_consts} but another "
                        f"call site passes {site_consts}; give the two calls their own helper"
                    )
                # The body is also specialized on the first site's SHAPES -- `infer_helper_params`
                # reads them off that site's arguments and the emitter bakes them in as literals
                # (vgg16's `_maxpool2d` hardcodes c=3, h=224, w=224). A site passing a differently
                # shaped array would run those literal strides over its own buffer and read out of
                # bounds, which is a segfault, not a wrong number. Refuse until a helper can take
                # its shapes as parameters instead of constants.
                site_kept = [a for pn, a in zip(decl_pnames, site_args) if pn in pnames]
                for pn, first_a, site_a in zip(pnames, kept_args, site_kept):
                    if not (isinstance(first_a, ast.Name) and isinstance(site_a, ast.Name)):
                        continue
                    first_d, site_d = oarr_by.get(first_a.id), oarr_by.get(site_a.id)
                    if first_d is not None and site_d is not None and tuple(first_d.shape) != tuple(site_d.shape):
                        raise NotImplementedError(
                            f"helper {hdef.name!r} is specialized on {first_a.id}{tuple(first_d.shape)} but another "
                            f"call site passes {site_a.id}{tuple(site_d.shape)}; a helper cannot serve two shapes "
                            f"while its dimensions are emitted as constants"
                        )
                # What the CALLER can name -- see :func:`caller_side_symbol`. Resolved per SITE: two
                # sites can pass different extents for the same helper symbol.
                owner_held = (
                    {sy.name for sy in osymbols}
                    | {a.name for a in oarrays}
                    | {sc.name for sc in oscalars}
                    | set(shape_symbols(oarrays))
                    | {a.arg for a in owner_fn.args.args}
                    | held_before(owner_fn, site)
                )
                extra_srcs = [
                    caller_side_symbol(sy, owner_held, decl_pnames, site_args, hfn, hdef.name, 0, oarr_by)
                    for sy in extra_syms
                ]
                # The temps and the return buffer are allocated in the caller too, and were sized off
                # the helper's descriptors: same leak, one statement later. See _caller_side_shape.
                site_param_info = {
                    pn: (caller_side_shape(shp, owner_held, decl_pnames, site_args, hfn, hdef.name, oarr_by), dt)
                    for pn, (shp, dt) in param_info.items()
                }
                site_hret_shape = caller_side_shape(
                    hret_shape, owner_held, decl_pnames, site_args, hfn, hdef.name, oarr_by
                )
                callsite_rewrites.setdefault(id(owner_fn), {})[id(site)] = build_callsite_stmts(
                    site.targets[0],
                    hdef.name,
                    pnames,
                    site_kept,
                    extra_srcs,
                    site_param_info,
                    site_hret_shape,
                    hret_dtype,
                    f"{hidx}_{sidx}" if sidx else hidx,
                    inout=inout_param is not None,
                    live_buffers=frozenset(a.name for a in oarrays),
                )
    for owner_fn, unused, unused, unused in scopes:
        rewrites = callsite_rewrites.get(id(owner_fn))
        if rewrites:
            ReplaceStmts(rewrites).visit(owner_fn)
            ast.fix_missing_locations(owner_fn)
    # A surviving helper may CALL a sibling helper, and a helper reached only that way got no
    # KernelIR above ("called only from another helper -- resolve in a later pass"). Nothing then
    # emits it, and the call reaches a function that does not exist: lulesh's `_lagrange_nodal`
    # calling `_calc_force_for_nodes` compiled to an implicit declaration and linked to nothing.
    # Refuse, so a level-3 kernel retries with inlining on and the whole chain is flattened.
    emitted = {h.kernel_name for h in out}
    known = {h.name for h in helper_defs}
    for h in out:
        missing = sorted(
            {
                n.func.id
                for n in ast.walk(h.tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in known
                and n.func.id not in emitted
            }
        )
        if missing:
            raise NotImplementedError(
                f"helper {h.kernel_name!r} calls {missing}, which are reached only from "
                f"another helper and are not emitted as functions of their own"
            )
    # Last, so every helper KernelIR (hence every param_order()) is final and the rewritten
    # call sites above are in the tree. Helper bodies too: a helper may call a sibling helper.
    reorder_helper_call_args([kernel_fn] + [h.tree for h in out], out)
    # Back to DEFINITION order: helpers are BUILT callers-first (see :func:`helpers_callers_first`)
    # but must be EMITTED callee-first, so a C caller sees a definition and not an implicit
    # declaration.
    defn_order = {h.name: i for i, h in enumerate(helper_defs)}
    out.sort(key=lambda ir: defn_order[ir.kernel_name])
    return out
