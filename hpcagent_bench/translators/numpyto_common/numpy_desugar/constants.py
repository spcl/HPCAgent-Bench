"""Compile-time folds: ``finfo`` eps, constant comprehensions, list-comprehension unrolls, defaults."""

import ast
import copy
import math
from collections.abc import Sequence

from hpcagent_bench.translators.numpyto_common import dtypes
from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import const_int, name_store_counts


def fd_step(precision: str | None = None) -> str:
    """``sqrt(machine epsilon)`` of the working float type, as a source literal.

    MINPACK's ``fdjac2`` forward-difference step (``h = sqrt(eps) * |p_j|``); :func:`curve_fit_lm_lines`
    shares it so the emitted fit has scipy's Jacobian truncation error and stationary point.

    Must track ``precision``: the literal is emitted into the body, which ``apply_precision`` never
    rewrites, and an fp64 step added to an fp32 parameter rounds away, zeroing the Jacobian. A float
    with no numpy finfo (fp8 storage) gets the fp64 step.
    """
    return repr(math.sqrt(dtypes.float_eps(working_float_dtype(precision))))


class FinfoEpsFold(ast.NodeTransformer):
    """``np.finfo(<anything>).eps`` -> the machine epsilon of the working float dtype, as a literal.

    A round-off bound (MINPACK's ftol/xtol, a finite-difference step) must follow the precision the
    kernel is lowered to; an accuracy requirement (a solver's ``tol=1e-6``) is fixed at every width
    and must not go through here. Folded because the emitters write source text: there is no
    ``finfo`` at native run time.
    """

    def __init__(self, precision: str | None = None) -> None:
        self.eps = dtypes.float_eps(working_float_dtype(precision))

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        call = node.value
        if (
            node.attr == "eps"
            and isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "finfo"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id in ("np", "numpy")
        ):
            return ast.copy_location(ast.Constant(value=self.eps), node)
        return node


def fold_finfo_eps(tree: ast.Module, precision: str | None = None) -> None:
    """Fold every ``np.finfo(...).eps`` in ``tree`` to the working precision's epsilon."""
    FinfoEpsFold(precision).visit(tree)


def working_float_dtype(precision: str | None = None) -> str:
    """The float dtype a lowering allocates its own scratch in: ``precision``, else ``float64``.

    A hardcoded ``float64`` would run an fp32 kernel's scratch at double width and narrow on the
    store back into its fp32 target.
    """
    return dtypes.canonical(precision) if precision else "float64"


#: builtins a constant comprehension may call: pure, side-effect free, and identical
#: at desugar time and at runtime.
CONST_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "divmod": divmod,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def const_name_values(fn: ast.AST) -> dict[str, object]:
    """Names bound EXACTLY once inside ``fn``, to a literal -> that literal's value.
    A name stored anywhere else (a second assignment, a loop target, a parameter)
    is dropped: this table is flow-insensitive, so it may only hold values that are
    the same at every program point."""
    stores: dict[str, int] = {}
    values: dict[str, object] = {}
    if isinstance(fn, ast.FunctionDef):
        for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
            stores[a.arg] = stores.get(a.arg, 0) + 1
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            stores[node.id] = stores.get(node.id, 0) + 1
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                continue
    return {n: v for n, v in values.items() if stores.get(n) == 1}


def const_literal_ast(value: object) -> ast.expr | None:
    """A folded python value -> its literal AST, or None when it has no literal
    spelling (an empty set, a non-scalar leaf such as a range/generator)."""
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return ast.Constant(value=value)
    if isinstance(value, (list, tuple, set, frozenset)):
        if isinstance(value, (set, frozenset)):
            try:
                value = sorted(value)  # set iteration is hash order -- the emitted source must not vary
            except TypeError:
                return None
            elts = [const_literal_ast(v) for v in value]
            return ast.Set(elts=elts) if elts and all(e is not None for e in elts) else None
        elts = [const_literal_ast(v) for v in value]
        if any(e is None for e in elts):
            return None
        if isinstance(value, list):
            return ast.List(elts=elts, ctx=ast.Load())
        return ast.Tuple(elts=elts, ctx=ast.Load())
    if isinstance(value, dict):
        keys = [const_literal_ast(k) for k in value]
        vals = [const_literal_ast(v) for v in value.values()]
        if any(e is None for e in keys + vals):
            return None
        return ast.Dict(keys=keys, values=vals)
    return None


class ConstComprehensionFold(ast.NodeTransformer):
    """``[int(round(fr * 4)) for fr in (0.5, 1.0)]`` -> the literal ``[2, 4]``. The
    DaCe frontend refuses every comprehension.

    A comprehension touching any runtime value (a parameter, an array element, a
    symbol) is left alone: unrolling it would pin a trip count only the runtime
    knows. Attribute and subscript reads, lambdas, and calls to anything but a
    whitelisted pure builtin count as runtime. Inner comprehensions fold first."""

    def __init__(self, consts: dict[str, object]) -> None:
        self.consts = consts
        self.changed = False

    def foldable(self, node: ast.expr) -> bool:
        bound = {n.id for g in node.generators for n in ast.walk(g.target) if isinstance(n, ast.Name)}
        for n in ast.walk(node):
            if isinstance(
                n,
                (
                    ast.Attribute,
                    ast.Subscript,
                    ast.Lambda,
                    ast.Starred,
                    ast.NamedExpr,
                    ast.Await,
                    ast.Yield,
                    ast.YieldFrom,
                    ast.JoinedStr,
                ),
            ):
                return False
            if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in CONST_BUILTINS):
                return False
            if isinstance(n, ast.Name) and not (n.id in bound or n.id in self.consts or n.id in CONST_BUILTINS):
                return False
            if isinstance(n, ast.comprehension) and n.is_async:
                return False
        return True

    def fold_(self, node: ast.expr) -> ast.AST:
        if not self.foldable(node):
            return node
        expr = ast.Expression(body=copy.deepcopy(node))
        ast.fix_missing_locations(expr)
        # A comprehension body runs in its own scope and resolves free names through
        # GLOBALS, so the constant table goes in as globals, not as locals.
        env = {"__builtins__": {}, **CONST_BUILTINS, **self.consts}
        try:
            value = eval(compile(expr, "<desugar>", "eval"), env)  # noqa: S307 -- every name is a proven constant
        except Exception:  # noqa: BLE001 -- any failure to evaluate just means "not foldable"
            return node
        if isinstance(node, ast.GeneratorExp):
            value = tuple(value)  # a genexp yields once; a tuple literal is the constant form of that
        lit = const_literal_ast(value)
        if lit is None:
            return node
        self.changed = True
        return ast.copy_location(lit, node)

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        self.generic_visit(node)
        return self.fold_(node)

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        self.generic_visit(node)
        return self.fold_(node)

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        self.generic_visit(node)
        return self.fold_(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        self.generic_visit(node)
        return self.fold_(node)


#: An unrolled comprehension copies its body once per element; this caps the source (and SDFG) blow-up.
UNROLL_MAX = 64


def const_iterable(node: ast.expr, consts: dict[str, object]) -> Sequence[object] | None:
    """A comprehension iterable already fixed at desugar time -> its values, else None.
    Covers a literal list/tuple, a ``range`` of literal bounds, and a name the const
    table resolved."""
    if isinstance(node, ast.Name):
        value = consts.get(node.id)
        return value if isinstance(value, (list, tuple)) else None
    if isinstance(node, (ast.List, ast.Tuple)):
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and node.args
        and len(node.args) < 4
        and not node.keywords
    ):
        bounds = [const_int(a) for a in node.args]
        if None in bounds or bounds[2:] == [0]:
            return None  # a zero step is a ValueError, not an iterable
        return range(*bounds)
    return None


class SubstConstName(ast.NodeTransformer):
    """Every LOAD of ``name`` -> ``value`` (one unrolled comprehension iteration)."""

    def __init__(self, name: str, value: ast.expr) -> None:
        self.name = name
        self.value = value

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id != self.name or not isinstance(node.ctx, ast.Load):
            return node
        return ast.copy_location(copy.deepcopy(self.value), node)


class ListCompUnroll(ast.NodeTransformer):
    """``[f(x, i) for i in range(3)]`` -> ``[f(x, 0), f(x, 1), f(x, 2)]``.

    A constant iterable driving a runtime body (a comprehension constant end to end is
    :class:`ConstComprehensionFold`'s). Only the loop goes away; the body stays verbatim.

    Left alone: a non-constant iterable, any ``if`` guard, more than one ``for`` clause,
    a non-Name target, an element with no literal spelling, and a body holding a lambda
    or rebinding the target -- either would capture the substituted literal instead of
    shadowing it."""

    def __init__(self, consts: dict[str, object]) -> None:
        self.consts = consts
        self.changed = False

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        self.generic_visit(node)  # an inner comprehension folds/unrolls first
        if len(node.generators) != 1:
            return node
        gen = node.generators[0]
        if gen.ifs or gen.is_async or not isinstance(gen.target, ast.Name):
            return node
        values = const_iterable(gen.iter, self.consts)
        if values is None or len(values) > UNROLL_MAX:
            return node
        name = gen.target.id
        shadowed = any(
            isinstance(n, ast.Lambda) or (isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Load))
            for n in ast.walk(node.elt)
        )
        if shadowed:
            return node
        elts: list[ast.expr] = []
        for v in values:
            lit = const_literal_ast(v)
            if lit is None:
                return node
            elts.append(SubstConstName(name, lit).visit(copy.deepcopy(node.elt)))
        self.changed = True
        return ast.copy_location(ast.List(elts=elts, ctx=ast.Load()), node)


#: Backends whose kernel is called positionally through ``kir.input_args`` and nothing else, so a
#: defaulted parameter outside that list is a constant -- the fold the native frontend already does.
DEFAULT_FOLDING_BACKENDS = frozenset({"numba", "pythran"})


def has_defaulted_parameters(fn: ast.FunctionDef) -> bool:
    """True when ``fn`` declares a positional or keyword-only parameter with a default."""
    return bool(fn.args.defaults) or any(d is not None for d in fn.args.kw_defaults)


def fold_kernel_defaults(fn: ast.FunctionDef, input_args: Sequence[str]) -> bool:
    """Fold the kernel's defaulted parameters the harness never passes into body constants, through the
    native frontend's own :func:`numpyto_common.frontend.module_constants.fold_default_args`. True when one folded.

    numba counts a keyword-only parameter as required, so an unfolded one breaks the entry's arity."""
    # Imported here: frontend imports this module at its top.
    from hpcagent_bench.translators.numpyto_common.frontend import fold_default_args

    before = ast.dump(fn.args)
    fold_default_args(fn, list(input_args))
    return ast.dump(fn.args) != before


def fold_constant_helper_arguments(tree: ast.Module, kernel_name: str) -> bool:
    """Substitute a helper parameter that EVERY call site passes the same ``True``/``False``/``None``
    literal into that helper's body (numba only). True when one was substituted.

    numba types both arms of ``if flag:`` even when every caller passes ``False``, so a ``None`` buffer
    read under the dead arm fails typing; the substituted literal lets :class:`DeadBranchElim` drop that
    arm first. A helper whose name escapes as a value, a call with keywords or a starred argument, or a
    parameter the body rebinds is left alone."""
    helpers = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != kernel_name}
    sites, escaped = helper_call_sites(tree, helpers)
    changed = False
    for name, fn in helpers.items():
        calls = sites.get(name, [])
        if not calls or name in escaped or not substitutable_helper(fn, calls):
            continue
        changed = substitute_loads(fn, constant_parameters(fn, calls)) or changed
    return changed


def helper_call_sites(
    tree: ast.Module, helpers: dict[str, ast.FunctionDef]
) -> tuple[dict[str, list[ast.Call]], set[str]]:
    """Every call of a helper by name, and the helpers whose name is read any other way."""
    sites: dict[str, list[ast.Call]] = {}
    callee_names: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in helpers:
            sites.setdefault(node.func.id, []).append(node)
            callee_names.add(id(node.func))
    escaped = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id in helpers and id(n) not in callee_names
    }
    return sites, escaped


def substitutable_helper(fn: ast.FunctionDef, calls: list[ast.Call]) -> bool:
    """True when every parameter of ``fn`` binds positionally at every call and ``fn`` nests no scope."""
    if fn.args.vararg or fn.args.kwarg or fn.args.posonlyargs:
        return False
    if any(c.keywords or any(isinstance(a, ast.Starred) for a in c.args) for c in calls):
        return False
    return not any(isinstance(n, (ast.Lambda, ast.FunctionDef)) and n is not fn for n in ast.walk(fn))


def constant_parameters(fn: ast.FunctionDef, calls: list[ast.Call]) -> dict[str, object]:
    """Parameters of ``fn`` every call passes the same ``True``/``False``/``None`` -> that value."""
    stores = name_store_counts(fn)
    # A literal spelled as a subscript base, attribute owner or callee (``None[:, 0]``) is a
    # SyntaxWarning at compile time even under a dead arm, so such a parameter stays a name.
    structural: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, (ast.Subscript, ast.Attribute)) and isinstance(n.value, ast.Name):
            structural.add(n.value.id)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            structural.add(n.func.id)
    subst: dict[str, object] = {}
    for i, param in enumerate(fn.args.args):
        if stores.get(param.arg, 0) != 1 or param.arg in structural:
            continue
        passed = [c.args[i] if i < len(c.args) else None for c in calls]
        if not all(isinstance(p, ast.Constant) and (p.value is None or isinstance(p.value, bool)) for p in passed):
            continue
        values = {repr(p.value) for p in passed if isinstance(p, ast.Constant)}
        if len(values) == 1 and isinstance(passed[0], ast.Constant):
            subst[param.arg] = passed[0].value
    return subst


def substitute_loads(fn: ast.FunctionDef, subst: dict[str, object]) -> bool:
    """Replace every load of a name in ``subst`` inside ``fn`` by its literal. True when one was replaced."""
    changed = False
    for node in ast.walk(fn):
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.Name) and isinstance(value.ctx, ast.Load) and value.id in subst:
                setattr(node, field, ast.copy_location(ast.Constant(value=subst[value.id]), value))
                changed = True
            elif isinstance(value, list):
                for k, item in enumerate(value):
                    if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id in subst:
                        value[k] = ast.copy_location(ast.Constant(value=subst[item.id]), item)
                        changed = True
    return changed
