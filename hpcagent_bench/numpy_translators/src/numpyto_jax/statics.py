"""Which parameters must be static (concrete at trace time) for the kernel and each helper."""

import ast

from numpyto_jax.errors import EmitError
from numpyto_jax.names import names_loaded
from numpyto_jax.state import STATE
from numpyto_jax.vocab import ARRAY_ATTRS, LEADING_DATA_FUNCS, SHAPE_FUNCS, STATIC_BUILTINS


def static_want(fn: ast.FunctionDef, params: list[str]) -> set[str]:
    """Params ``fn`` needs CONCRETE at trace time: those feeding a ``range()``
    bound, an array-shape dimension, or an ``if``/``while``/ternary CONDITION.
    (Array-likeness is NOT excluded here -- see :func:`static_params`.)"""
    pset = set(params)
    want: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            is_range = isinstance(n.func, ast.Name) and n.func.id == "range"
            attr = n.func.attr if isinstance(n.func, ast.Attribute) else None
            if is_range or attr in SHAPE_FUNCS:
                # Funcs with a leading data-array arg (reshape(a, shape),
                # histogram(a, bins), ...) -- skip arg 0; its dims live after.
                scan = n.args[1:] if attr in LEADING_DATA_FUNCS else n.args
                for a in scan:
                    want |= names_loaded(a) & pset
        # A scalar controlling an if/while/ternary must be concrete: a
        # data-dependent branch can't yield a bool from a traced scalar, and
        # (contour_integral's ``if NR == NM`` picking ``inv`` vs ``solve``)
        # the arms may have INCOMPATIBLE shapes no ``jnp.where`` can merge.
        # Such a param is an implicit dimension even without feeding a shape
        # (fv3_dycore's ``hord`` via ``8 if hord == 10 else hord``); excluded
        # in ``static_params`` if also used as data.
        elif isinstance(n, (ast.If, ast.While, ast.IfExp)):
            want |= names_loaded(n.test) & pset
    return want


def array_like_params(fn: ast.FunctionDef, params: list[str]) -> set[str]:
    """Params ever used as an ARRAY VALUE -- subscripted, accessed via an array
    attribute/reduction (``x.shape``, ``x.max()``), or a whole-array ``@``
    operand -- so they are data, never a static scalar dimension."""
    array_like: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
            array_like.add(n.value.id)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.attr in ARRAY_ATTRS:
            array_like.add(n.value.id)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult):
            for side in (n.left, n.right):
                if isinstance(side, ast.Name) and side.id in params:
                    array_like.add(side.id)
    return array_like


def static_params(fn: ast.FunctionDef, params: list[str]) -> list[str]:
    """Params requiring concreteness during tracing (``range`` bound, shape dim,
    branch predicate), minus those ever used as array data."""
    want = static_want(fn, params)
    array_like = array_like_params(fn, params)
    return [p for p in params if p in want and p not in array_like]


def value_names(node: ast.AST, pset: set[str]) -> set[str]:
    """Param names appearing in ``node`` as a VALUE -- EXCLUDING those that occur
    only as the base of a ``.shape`` / ``.size`` / ``.ndim`` access (a statically
    known dimension, not the array's data). So ``int(egrid.shape[0])`` yields
    nothing, while ``int(num_nucs[mat])`` yields ``num_nucs`` (and ``mat``)."""
    skip = {
        id(n.value)
        for n in ast.walk(node)
        if isinstance(n, ast.Attribute) and n.attr in ("shape", "size", "ndim") and isinstance(n.value, ast.Name)
    }
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in pset and id(n) not in skip}


def concrete_bound_want(fn: ast.FunctionDef, params: list[str]) -> set[str]:
    """Params whose value (or an element) feeds a ``range()`` bound or shape
    dimension -- needs a concrete int at trace time. Unlike
    :func:`static_want` this excludes branch predicates (lowers to
    ``jnp.where``, cloudsc's ``if ldcum[jl-1]``) and ``.shape`` accesses, so
    only a genuinely un-AoT-able bound (xsbench's ``num_nucs[mat]``) lands here."""
    pset = set(params)
    want: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            is_range = isinstance(n.func, ast.Name) and n.func.id == "range"
            attr = n.func.attr if isinstance(n.func, ast.Attribute) else None
            if is_range or attr in SHAPE_FUNCS:
                scan = n.args[1:] if attr in LEADING_DATA_FUNCS else n.args
                for a in scan:
                    want |= value_names(a, pset)
    return want


def propagate_param_flow(funcs: dict, seed: dict) -> dict:
    """Fix-point closure of a per-function param set over call edges: a
    caller param flowing into a callee param already in the set joins it.
    ``seed`` maps each function to its own params; shared by the static-want,
    concrete-bound-want and array-like passes (fv3_dycore's ``hord`` reaches
    ``xppm_flux``'s ``if mord == 5`` only via the call chain)."""
    params_of = {name: [a.arg for a in f.args.args] for name, f in funcs.items()}
    out = {name: set(seed[name]) for name in funcs}
    changed = True
    while changed:
        changed = False
        for name, f in funcs.items():
            pset = set(params_of[name])
            for call in ast.walk(f):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in funcs):
                    continue
                gparams = params_of[call.func.id]
                for wp in out[call.func.id]:
                    pos = gparams.index(wp)
                    if pos < len(call.args):
                        for nm in names_loaded(call.args[pos]) & pset:
                            if nm not in out[name]:
                                out[name].add(nm)
                                changed = True
    return out


def transitive_array_like(funcs: dict) -> dict:
    """``{func_name: array-like params}`` propagated forward across calls: a
    param passed directly into a callee position that's array-like is itself
    array-like. gromacs/xsbench pass index arrays straight through to helpers
    that subscript them -- a body-only check misses that these are DATA."""
    params_of = {name: [a.arg for a in f.args.args] for name, f in funcs.items()}
    # Seed with array-like PARAMS only (``array_like_params`` also reports local
    # arrays and ``np`` from ``np.max`` -- neither is a callee param to trace).
    al = {name: (set(array_like_params(f, params_of[name])) & set(params_of[name])) for name, f in funcs.items()}
    changed = True
    while changed:
        changed = False
        for name, f in funcs.items():
            pset = set(params_of[name])
            for call in ast.walk(f):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in funcs):
                    continue
                gparams = params_of[call.func.id]
                for gp in al[call.func.id]:
                    pos = gparams.index(gp)  # gp is a callee param -> always present
                    if (
                        pos < len(call.args)
                        and isinstance(call.args[pos], ast.Name)
                        and call.args[pos].id in pset
                        and call.args[pos].id not in al[name]
                    ):
                        al[name].add(call.args[pos].id)
                        changed = True
    return al


def is_static_expr(node: ast.AST, ctx: set[str]) -> bool:
    """Is ``node`` evaluable to a concrete value given the static names in
    ``ctx``? Literals, ``ctx`` names, arithmetic/compare/ternary over such,
    ``.shape``/``.size``/``.ndim``, and a pure builtin/``np.``/``math.`` call
    over static args qualify -- ``seq[i]`` or a user-helper call do not."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id in ctx
    if isinstance(node, ast.BinOp):
        return is_static_expr(node.left, ctx) and is_static_expr(node.right, ctx)
    if isinstance(node, ast.UnaryOp):
        return is_static_expr(node.operand, ctx)
    if isinstance(node, ast.BoolOp):
        return all(is_static_expr(v, ctx) for v in node.values)
    if isinstance(node, ast.Compare):
        return is_static_expr(node.left, ctx) and all(is_static_expr(c, ctx) for c in node.comparators)
    if isinstance(node, ast.IfExp):
        return all(is_static_expr(x, ctx) for x in (node.test, node.body, node.orelse))
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(is_static_expr(e, ctx) for e in node.elts)
    if isinstance(node, ast.Attribute):
        return node.attr in ("shape", "size", "ndim")  # a statically-known dimension of any array
    if isinstance(node, ast.Subscript):  # e.g. ``x.shape[0]``
        return is_static_expr(node.value, ctx) and is_static_expr(node.slice, ctx)
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name) and f.id in STATIC_BUILTINS:
            return all(is_static_expr(a, ctx) for a in node.args)
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("np", "jnp", "math"):
            return all(is_static_expr(a, ctx) for a in node.args)
        return False
    return False


def static_ctx(fn: ast.FunctionDef, base: set[str]) -> set[str]:
    """``base`` grown with static LOCALS -- single-assignment ``name =
    <static expr>`` (fv3_dycore's ``ord_inner = 8 if hord == 10 else hord``).
    A name assigned more than once is skipped (conservative)."""
    counts: dict = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    counts[t.id] = counts.get(t.id, 0) + 1
    ctx = set(base)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                t = node.targets[0].id
                if counts.get(t) == 1 and t not in ctx and is_static_expr(node.value, ctx):
                    ctx.add(t)
                    changed = True
    return ctx


def concrete_params(funcs: dict, kernel_name: str, kernel_static: list[str]) -> dict:
    """``{func_name: params CONCRETE at trace time}`` computed forward from
    the kernel's static args: a callee param is concrete iff at EVERY call
    site it's built only from the caller's concrete params/constants/
    literals. This is what a helper's :data:`STATE.emit_static` must be --
    fv3_dycore's ``hord`` reaches ``xppm_flux``'s ``if mord == 5`` as static,
    whereas nussinov's ``match(seq[i], seq[j])`` passes TRACED elements, so
    ``match``'s ``if b1 + b2 == 3`` must lower to ``jnp.where``. Requiring
    ALL call sites to agree keeps a helper shared between static and traced
    callers correctly non-concrete."""
    params_of = {name: [a.arg for a in f.args.args] for name, f in funcs.items()}
    defaults_of: dict = {}
    for name, f in funcs.items():
        ps = params_of[name]
        dflts = f.args.defaults
        off = len(ps) - len(dflts)
        defaults_of[name] = {ps[off + k]: d for k, d in enumerate(dflts)}
    mc = set(STATE.module_consts)
    concrete = {name: set() for name in funcs}
    concrete[kernel_name] = set(kernel_static)
    changed = True
    while changed:
        changed = False
        votes: dict = {name: {} for name in funcs}
        for caller, f in funcs.items():
            ctx = static_ctx(f, set(concrete[caller]) | mc)
            for call in ast.walk(f):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in funcs):
                    continue
                callee = call.func.id
                passed = static_arguments(call, params_of[callee], defaults_of[callee], ctx, mc)
                for p, st in passed.items():
                    votes[callee][p] = votes[callee].get(p, True) and st
        for callee in funcs:
            if callee == kernel_name:
                continue
            newset = {p for p, st in votes[callee].items() if st}
            if newset != concrete[callee]:
                concrete[callee] = newset
                changed = True
    return concrete


def static_arguments(
    call: ast.Call, params: list[str], defaults: dict[str, ast.expr], ctx: set[str], module_consts: set[str]
) -> dict[str, bool]:
    """Per callee parameter: is the value this call site binds to it static? An omitted argument
    falls back to its default, which is evaluated in module scope."""
    passed: dict = {}
    for pos, arg in enumerate(call.args):
        if pos < len(params):
            passed[params[pos]] = is_static_expr(arg, ctx)
    for kw in call.keywords:
        if kw.arg in params:
            passed[kw.arg] = is_static_expr(kw.value, ctx)
    for p in params:
        if p not in passed:
            d = defaults.get(p)
            passed[p] = d is not None and is_static_expr(d, module_consts)
    return passed


def transitive_static(kernel_name: str, funcs: dict) -> list[str]:
    """The kernel's ``static_argnames``, computed transitively: a param is
    static if it (or a value flowing from it) feeds a range/shape/branch
    anywhere in a reachable helper. fv3_dycore's ``hord``/``grid_type`` never
    feed a branch in the kernel itself -- only deep inside ``xppm_flux``'s
    ``if mord == 5`` -- so without propagation they'd stay traced and raise
    ``TracerBoolConversionError``."""
    params_of = {name: [a.arg for a in f.args.args] for name, f in funcs.items()}
    want = propagate_param_flow(funcs, {n: static_want(f, params_of[n]) for n, f in funcs.items()})
    cwant = propagate_param_flow(funcs, {n: concrete_bound_want(f, params_of[n]) for n, f in funcs.items()})
    kparams = params_of[kernel_name]
    array_like = transitive_array_like(funcs)[kernel_name]
    # A param needing a CONCRETE int (range bound/shape dim) that's also array
    # DATA is a data-dependent bound (xsbench's ``num_nucs[mat]``) -- can't be
    # ``static_argnames`` (unhashable array) nor traced into a ``range``, so
    # refuse and let eager (running the loop directly) take over. A mere
    # branch on an array element is NOT a conflict (lowers to ``jnp.where``),
    # so ``cwant`` -- not the full ``want`` -- gates this.
    conflict = [p for p in kparams if p in cwant[kernel_name] and p in array_like]
    if conflict:
        raise EmitError(f"data-dependent shape/bound from array data: {', '.join(conflict)}")
    return [p for p in kparams if p in want[kernel_name] and p not in array_like]
