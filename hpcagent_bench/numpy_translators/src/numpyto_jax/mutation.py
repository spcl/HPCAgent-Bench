"""In-place parameter mutation made functional: mutated params become extra return values and
every call site rebinds them."""

import ast

from numpyto_jax.names import as_store, base_name, is_assignable


def helper_mutation_map(helpers: list[ast.FunctionDef], mut_map: dict | None = None) -> dict:
    """Map ``helper_name -> emitted-return rebind slots`` for helpers that
    mutate a param in place (jax arrays are immutable, so the mutation only
    survives if the call site captures it).

    ``augment_returns`` turns such a return into ``own_elts + [mutated params
    not already returned by name]`` (a no-return helper returns just the
    mutated params). Each slot is ``("mut", pos)`` -- new value of the arg at
    param ``pos`` -- or ``("val",)`` -- a genuine return the LHS captures.
    ``rewrite_inplace_helper_calls`` uses these to rebind every call site:

    * ``build_up_b(b, ..)`` (mutates ``b``, returns None) -> ``b = build_up_b(b, ..)``;
    * ``_addusxx_r(rhoc, ..)`` (mutates AND returns it) -> ``rhoc = _addusxx_r(rhoc, ..)``;
    * ``fac = _g2_convolution_all(cf, cd, ..)`` (returns + mutates caches) ->
      ``fac, cf, cd = _g2_convolution_all(cf, cd, ..)``.
    """
    out = {}
    for h in helpers:
        params = [a.arg for a in h.args.args]
        muts = mut_map[h.name] if mut_map is not None else mutated_params(h, params)
        if not muts:
            continue
        slots = emitted_return_slots(h, params, muts)
        if slots is not None:
            out[h.name] = slots
    return out


def emitted_return_slots(h: ast.FunctionDef, params: list[str], muts: list[str]) -> list[tuple] | None:
    """Ordered rebind slots for a mutating helper's EMITTED return, or None when
    it can't be rewritten safely (a return the augmentation can't reach, or
    inconsistent return points)."""
    mpos = {m: params.index(m) for m in muts}
    rets = own_returns(h)
    top = [s for s in h.body if isinstance(s, ast.Return) and s.value is not None]
    if not rets:
        # No own return: the emitter appends ``return <mutated, signature order>``.
        return [("mut", mpos[m]) for m in muts]
    if len(rets) != len(top):
        return None  # a return nested in a branch/loop -- augmentation only reaches fn.body
    slot_sets = set()
    for r in top:
        elts = r.value.elts if isinstance(r.value, ast.Tuple) else [r.value]
        present = {e.id for e in elts if isinstance(e, ast.Name)}
        slots = [("mut", mpos[e.id]) if (isinstance(e, ast.Name) and e.id in mpos) else ("val",) for e in elts]
        slots += [("mut", mpos[m]) for m in muts if m not in present]
        slot_sets.add(tuple(slots))
    # Every return point must rebind identically for a single call-site rewrite.
    return list(next(iter(slot_sets))) if len(slot_sets) == 1 else None


def rewrite_inplace_helper_calls(fn: ast.FunctionDef, helper_mut: dict) -> None:
    """Capture a mutating helper's in-place effect at every call site, per its
    emitted-return slots (see ``helper_mutation_map``). A ``("mut", pos)``
    slot rebinds the arg at ``pos`` (a Name, or a view like ``deexx[:, ii]``
    that flows through ``functionalize_stmt`` into ``.at[:, ii].set(..)``);
    a ``("val",)`` slot comes from the call's LHS."""

    def rebind_targets(call: ast.Call, lhs_elts: list[ast.AST]) -> list[ast.AST] | None:
        slots = helper_mut[call.func.id]
        if sum(1 for s in slots if s[0] == "val") != len(lhs_elts):
            return None  # LHS arity must match the genuine return values
        for kind, *rest in slots:
            if kind == "mut" and (rest[0] >= len(call.args) or not is_assignable(call.args[rest[0]])):
                return None  # mutated param not passed positionally / not assignable
        vi = iter(lhs_elts)
        return [as_store(next(vi) if s[0] == "val" else call.args[s[1]]) for s in slots]

    def rebind(call: ast.Call, lhs_elts: list[ast.AST], node: ast.stmt) -> ast.stmt:
        tgts = rebind_targets(call, lhs_elts)
        if tgts is None:
            return node
        tgt = tgts[0] if len(tgts) == 1 else ast.Tuple(elts=tgts, ctx=ast.Store())
        return ast.copy_location(ast.Assign(targets=[tgt], value=call), node)

    def is_mut_call(c: ast.AST) -> bool:
        return isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in helper_mut

    class Rewriter(ast.NodeTransformer):
        def visit_Expr(self, node: ast.Expr) -> ast.stmt:  # bare call: no captured return values
            return rebind(node.value, [], node) if is_mut_call(node.value) else node

        def visit_Assign(self, node: ast.Assign) -> ast.stmt:  # value-captured call: LHS supplies the ``val`` slots
            if is_mut_call(node.value) and len(node.targets) == 1:
                lhs = node.targets[0]
                return rebind(node.value, lhs.elts if isinstance(lhs, ast.Tuple) else [lhs], node)
            return node

    Rewriter().visit(fn)
    ast.fix_missing_locations(fn)


def augment_returns(fn: ast.FunctionDef, mutated: list[str]) -> None:
    """Append in-place-mutated params to each TOP-LEVEL return, so a kernel
    returning a derived value while mutating outputs (channel_flow returns
    the step COUNT but mutates ``u``/``v``) still hands back the functional
    results. Only direct ``fn.body`` returns are touched; already-returned
    names aren't duplicated. A no-return kernel uses the caller's append path."""
    for stmt in fn.body:
        if not (isinstance(stmt, ast.Return) and stmt.value is not None):
            continue
        cur = list(stmt.value.elts) if isinstance(stmt.value, ast.Tuple) else [stmt.value]
        present = {e.id for e in cur if isinstance(e, ast.Name)}
        extra = [m for m in mutated if m not in present]
        if extra:
            stmt.value = ast.Tuple(elts=cur + [ast.Name(id=m, ctx=ast.Load()) for m in extra], ctx=ast.Load())


def own_returns(fn: ast.FunctionDef) -> list[ast.Return]:
    """``Return`` statements belonging to ``fn`` itself, not a nested helper
    def -- a plain ``ast.walk`` would descend into a nested ``def gat(..):
    return ..`` (velocity_tendencies), making the kernel look like it already
    returns and skipping the mutated-output augmentation."""
    out: list[ast.Return] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue  # a nested scope -- its returns are its own
            if isinstance(child, ast.Return) and child.value:
                out.append(child)
            visit(child)

    visit(fn)
    return out


def mutated_params(fn: ast.FunctionDef, params: list[str]) -> list[str]:
    """Params written in place, returned in **signature order** (HPCAgent-Bench's
    ``output_args`` convention) rather than mutation-encounter order."""
    mutated: set[str] = set()
    for s in ast.walk(fn):
        if isinstance(s, ast.AugAssign):
            # ``p += x`` (whole array) and ``p[i] += x`` are both numpy in-place
            # updates that DO propagate to the caller's buffer.
            base = s.target
            while isinstance(base, ast.Subscript):
                base = base.value
            if isinstance(base, ast.Name) and base.id in params:
                mutated.add(base.id)
        elif isinstance(s, ast.Assign):
            for t in s.targets:
                # Only a SUBSCRIPT target (``p[i]=``, ``p[:]=``) writes the
                # caller's array in place; a plain ``p = <expr>`` just REBINDS
                # the local (not propagated, not an output). Treating it as one
                # wrongly augments the return (``_upper_bound``'s ``v = v /
                # norm`` would return ``(theta, v)``, breaking a ``1.2 *
                # _upper_bound(..)`` call site with ``tuple * float``).
                if not isinstance(t, ast.Subscript):
                    continue
                base = t.value
                while isinstance(base, ast.Subscript):
                    base = base.value
                if isinstance(base, ast.Name) and base.id in params:
                    mutated.add(base.id)
        # np.add.at(target, idx, val) mutates its first arg through a Call,
        # not an assign target -- later rewritten to ``.at[idx].add(val)``,
        # so the param IS mutated and must be returned.
        elif isinstance(s, ast.Call):
            f = s.func
            if isinstance(f, ast.Attribute) and f.attr == "at" and isinstance(f.value, ast.Attribute) and s.args:
                base = s.args[0]
                while isinstance(base, ast.Subscript):
                    base = base.value
                if isinstance(base, ast.Name) and base.id in params:
                    mutated.add(base.id)
    return [p for p in params if p in mutated]


def mutation_maps(funcs: dict) -> dict:
    """``{func_name: [mutated params, signature order]}`` computed transitively.

    A param is an output if the function writes through it directly
    (:func:`mutated_params`) or passes it to a callee that mutates that
    position (fv3_dycore's ``xppm`` mutates ``al``/``xflux`` only via its
    ``compute_al_x``/``xppm_flux`` calls). The fixpoint propagates callee
    outputs back so an orchestrating helper is still recognised as mutating,
    and its call site functionalises (``al, xflux = xppm(..)``) instead of
    staying a bare-expression statement.

    A plain rebind ``p = <expr>`` is NOT a mutation, so a helper normalising
    a param locally (``v = v / norm``) isn't wrongly turned into a
    tuple-returning output."""
    params_of = {name: [a.arg for a in f.args.args] for name, f in funcs.items()}
    muts = {name: set(mutated_params(f, params_of[name])) for name, f in funcs.items()}
    changed = True
    while changed:
        changed = False
        for name, f in funcs.items():
            pset = set(params_of[name])
            for call in ast.walk(f):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in funcs):
                    continue
                gparams = params_of[call.func.id]
                for mp in muts[call.func.id]:
                    pos = gparams.index(mp)
                    if pos < len(call.args):
                        base = base_name(call.args[pos])
                        if base in pset and base not in muts[name]:
                            muts[name].add(base)
                            changed = True
    return {name: [p for p in params_of[name] if p in muts[name]] for name in funcs}
