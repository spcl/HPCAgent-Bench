"""dtype KIND inference (bool < int < float < complex) over a module, helper calls included."""

import ast
from collections.abc import Mapping
from typing import NamedTuple

from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import (
    LIKE_CTORS,
    SHAPE_CTORS,
    eigh_call_kind,
    np_attr,
    np_submodule_attr,
    reachable_functions,
)


# dtype "kind" ordered by promotion rank (numpy-style: bool < int < float < complex).
KIND_RANK = {"bool": 0, "int": 1, "float": 2, "complex": 3}


#: ``np.fft.<name>`` -> the dtype kind it returns; a name absent here has no known kind.
FFT_RESULT_KINDS = {
    **dict.fromkeys(("fft", "ifft", "fft2", "ifft2", "fftn", "ifftn", "rfft", "rfft2", "rfftn", "ihfft"), "complex"),
    **dict.fromkeys(("irfft", "irfft2", "irfftn", "hfft", "fftfreq", "rfftfreq"), "float"),
}


#: Array methods whose result keeps the receiver's dtype kind.
KIND_KEEPING_METHODS = frozenset({"reshape", "ravel", "flatten", "copy", "conj", "conjugate", "squeeze", "transpose"})


#: ``np.<name>`` dtype spellings -> kind.
DTYPE_NAME_KIND = {
    "bool": "bool",
    "bool_": "bool",
    "int8": "int",
    "int16": "int",
    "int32": "int",
    "int64": "int",
    "intp": "int",
    "intc": "int",
    "uint8": "int",
    "uint16": "int",
    "uint32": "int",
    "uint64": "int",
    "float16": "float",
    "float32": "float",
    "float64": "float",
    "float": "float",
    "double": "float",
    "complex64": "complex",
    "complex128": "complex",
    "complex": "complex",
}


def kind_of_dtype_str(dt: str | None) -> str | None:
    """A numpy dtype tag string (``"int64"``, ``"float32"``) -> its kind."""
    if not dt:
        return None
    return DTYPE_NAME_KIND.get(dt) or (
        "int"
        if dt.startswith(("int", "uint"))
        else "float"
        if dt.startswith("float")
        else "complex"
        if dt.startswith("complex")
        else "bool"
        if dt.startswith("bool")
        else None
    )


def dtype_arg_kind(node: ast.AST) -> str | None:
    """A dtype ARGUMENT (``np.int64`` / ``np.dtype('f8')`` / a bare name) -> kind."""
    if isinstance(node, ast.Attribute):
        return DTYPE_NAME_KIND.get(node.attr)
    if isinstance(node, ast.Name):
        return DTYPE_NAME_KIND.get(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return kind_of_dtype_str(node.value)
    return None


def promote_kind(a: str | None, b: str | None) -> str | None:
    """numpy-style promotion; ``None`` (unknown) is contagious, so an unknown operand never passes as known."""
    if a is None or b is None:
        return None
    return a if KIND_RANK[a] >= KIND_RANK[b] else b


#: ufuncs whose result is always a boolean array (regardless of operand dtype).
BOOL_UFUNCS = {
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    "isnan",
    "isinf",
    "isfinite",
    "isclose",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
}


class CallKinds(NamedTuple):
    """Kinds calls to module-level helpers evaluate to (:func:`module_kind_tables`): ``values`` for a helper
    returning one value, ``tuples`` per element for one returning a tuple (``None`` = element unproven)."""

    values: Mapping[str, str]
    tuples: Mapping[str, tuple[str | None, ...]]


NO_CALLS = CallKinds({}, {})


def dtype_position_kind(dt: ast.AST, dtypes: dict[str, str], calls: CallKinds = NO_CALLS) -> str | None:
    """Kind a DTYPE argument names: a spelling (``np.int64``, ``"f8"``), ``x.dtype``, or a name holding one
    (a helper's ``dtype`` parameter fed its caller's ``X.dtype``)."""
    if isinstance(dt, ast.Attribute) and dt.attr == "dtype":
        return dtype_kind(dt.value, dtypes, calls)
    spelled = dtype_arg_kind(dt)
    if spelled is None and isinstance(dt, ast.Name):
        return dtypes.get(dt.id)
    return spelled


def call_kind(value: ast.Call, dtypes: dict[str, str], calls: CallKinds = NO_CALLS) -> str | None:
    """Kind of a call the numpy-attribute rules of :func:`dtype_kind` do not cover.

    A module helper returns its proven kind. Builtin ``max``/``min`` return one operand UNPROMOTED, so only
    operands of one kind give a kind. ``np.linalg.norm``/``eigvalsh`` are real for any input,
    ``np.tensordot`` promotes its operands, ``np.eye`` defaults to float."""
    f = value.func
    if isinstance(f, ast.Name):
        if f.id in calls.values:
            return calls.values[f.id]
        if f.id in ("float", "int"):
            return f.id
        if f.id in ("max", "min") and value.args and not value.keywords:
            kinds = {dtype_kind(arg, dtypes, calls) for arg in value.args}
            return kinds.pop() if len(kinds) == 1 else None
        return None
    if np_submodule_attr(value, "linalg") in ("norm", "eigvalsh"):
        return "float"
    attr = np_attr(value)
    if attr == "tensordot" and len(value.args) >= 2:
        return promote_kind(dtype_kind(value.args[0], dtypes, calls), dtype_kind(value.args[1], dtypes, calls))
    if attr == "eye":
        dt = next((k.value for k in value.keywords if k.arg == "dtype"), value.args[3] if len(value.args) > 3 else None)
        return "float" if dt is None else dtype_position_kind(dt, dtypes, calls)
    return None


def constant_kind(value: object) -> str | None:
    """Kind of a Python literal; ``bool`` is tested before ``int``, its base class."""
    for typ, kind in ((bool, "bool"), (int, "int"), (float, "float"), (complex, "complex")):
        if isinstance(value, typ):
            return kind
    return None


def real_part_kind(kind: str | None) -> str | None:
    """Kind of a value measured off ``kind`` (``.real``, ``np.abs``): complex becomes float."""
    return "float" if kind == "complex" else kind


def binop_kind(value: ast.BinOp, dtypes: dict[str, str], calls: CallKinds) -> str | None:
    lk, rk = dtype_kind(value.left, dtypes, calls), dtype_kind(value.right, dtypes, calls)
    if isinstance(value.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
        return "bool" if (lk == "bool" or rk == "bool") else promote_kind(lk, rk)
    if isinstance(value.op, (ast.Div, ast.MatMult)):
        p = promote_kind(lk, rk)
        return "float" if p in ("int", "bool") else p  # true division promotes to float
    return promote_kind(lk, rk)


def dtype_kind(value: ast.AST, dtypes: dict[str, str], calls: CallKinds = NO_CALLS) -> str | None:
    """Best-effort dtype KIND of an expression under the name -> kind table and the helpers' return kinds.

    ``None`` = unknown; callers treat it conservatively (e.g. never desugar a matmul as integer)."""
    if isinstance(value, ast.Name):
        return dtypes.get(value.id)
    if isinstance(value, ast.Constant):
        return constant_kind(value.value)
    if isinstance(value, (ast.Compare, ast.BoolOp)):
        return "bool"
    if isinstance(value, ast.UnaryOp):
        return "bool" if isinstance(value.op, ast.Not) else dtype_kind(value.operand, dtypes, calls)
    if isinstance(value, ast.BinOp):
        return binop_kind(value, dtypes, calls)
    if isinstance(value, ast.Subscript):
        return dtype_kind(value.value, dtypes, calls)  # indexing preserves dtype
    if isinstance(value, ast.Attribute) and value.attr in ("T", "dtype"):
        # A transpose is a permuted view; ``x.dtype`` names x's own kind (read as a dtype argument).
        return dtype_kind(value.value, dtypes, calls)
    if isinstance(value, ast.Attribute) and value.attr in ("real", "imag"):
        return real_part_kind(dtype_kind(value.value, dtypes, calls))
    if isinstance(value, ast.Call):
        return call_expr_kind(value, dtypes, calls)
    return None


def call_expr_kind(value: ast.Call, dtypes: dict[str, str], calls: CallKinds) -> str | None:
    """Kind of a call: ``np.fft`` / ``np.linalg`` first, then ``.astype`` and kind-keeping methods, then
    :func:`np_function_kind`."""
    fft = np_submodule_attr(value, "fft")
    if fft is not None:
        return FFT_RESULT_KINDS.get(fft)
    linalg = np_submodule_attr(value, "linalg")
    if linalg in ("cholesky", "inv") and value.args:
        return dtype_kind(value.args[0], dtypes, calls)  # a factor/inverse keeps the operand's kind
    if linalg == "solve" and len(value.args) >= 2:
        return promote_kind(dtype_kind(value.args[0], dtypes, calls), dtype_kind(value.args[1], dtypes, calls))
    f = value.func
    if isinstance(f, ast.Attribute) and f.attr == "astype" and value.args:
        return dtype_arg_kind(value.args[0])
    attr = np_attr(value)
    if attr is None and isinstance(f, ast.Attribute) and f.attr in KIND_KEEPING_METHODS:
        return dtype_kind(f.value, dtypes, calls)
    return np_function_kind(attr, value, dtypes, calls)


#: ``np.<name>(x, ...)`` whose result has ``x``'s kind.
FIRST_ARG_KIND_FUNCS = frozenset(LIKE_CTORS) | {
    "astype",
    "copy",
    "ascontiguousarray",
    "asarray",
    "array",
    "reshape",
    "sqrt",
    "exp",
    "sin",
    "cos",
    "conj",
    "conjugate",
    "sum",
    "prod",
    "transpose",
    "sign",
    "moveaxis",
    "diag",
}


#: ``np.<name>(x)`` that MEASURES ``x``: a complex ``x`` gives a real result. Typing ``np.abs(z)`` complex
#: would put a magnitude in a complex buffer the backend then cannot compare with ``>``.
MEASURING_FUNCS = frozenset({"real", "imag", "abs", "absolute", "angle"})


#: ``np.<name>(x)`` that lands on a float; only ``mean`` carries a complex ``x`` through. Left unknown,
#: :func:`reduce_axis_stmts` would fall back to a float64 accumulator inside an fp32 kernel.
MOMENT_FUNCS = frozenset({"mean", "std", "var"})


#: ``np.<name>(..., a, b)`` whose result promotes its last two arguments.
PAIR_PROMOTING_FUNCS = frozenset({"where", "minimum", "maximum", "clip"})


def np_function_kind(attr: str | None, value: ast.Call, dtypes: dict[str, str], calls: CallKinds) -> str | None:
    """Kind of ``np.<attr>(...)``; a call no rule here covers goes to :func:`call_kind`."""
    if attr in BOOL_UFUNCS:
        return "bool"
    if attr in DTYPE_NAME_KIND:
        return DTYPE_NAME_KIND[attr]  # np.int64(x) scalar cast
    if attr in SHAPE_CTORS:
        kw = {k.arg: k.value for k in value.keywords}
        dt = kw.get("dtype") or (value.args[1] if len(value.args) > 1 else None)
        # ``np.zeros(shape, x.dtype)`` reads through to x: an unknown kind would make the temp float64
        # and drop the imaginary part of the complex operand it mirrors.
        return dtype_position_kind(dt, dtypes, calls) if dt is not None else "float"  # default float64
    if attr in PAIR_PROMOTING_FUNCS and len(value.args) >= 2:
        return promote_kind(dtype_kind(value.args[-2], dtypes, calls), dtype_kind(value.args[-1], dtypes, calls))
    if value.args and (attr in FIRST_ARG_KIND_FUNCS or attr in MEASURING_FUNCS or attr in MOMENT_FUNCS):
        return derived_kind(attr, dtype_kind(value.args[0], dtypes, calls))
    return call_kind(value, dtypes, calls)


def derived_kind(attr: str | None, inner: str | None) -> str | None:
    """Kind of ``np.<attr>(x)`` for ``x`` of kind ``inner``."""
    if attr in MEASURING_FUNCS:
        return real_part_kind(inner)
    if attr in MOMENT_FUNCS and inner is not None:
        return "complex" if (attr == "mean" and inner == "complex") else "float"
    return inner


def dtype_table_(tree: ast.AST, seed: dict[str, str], calls: CallKinds = NO_CALLS) -> dict[str, str]:
    """Propagate dtype kinds across straight-line assignments to a fixpoint (:func:`assigned_kinds`)."""
    dtypes = dict(seed)
    for unused in range(8):
        changed = False
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            for name, k in assigned_kinds(node.targets[0], node.value, dtypes, calls):
                if k is not None and dtypes.get(name) != k:
                    dtypes[name] = k
                    changed = True
        if not changed:
            break
    return dtypes


def assigned_kinds(
    target: ast.expr, value: ast.expr, dtypes: dict[str, str], calls: CallKinds
) -> list[tuple[str, str | None]]:
    """``(name, kind)`` for each name ``target = value`` binds; a tuple target reads :func:`tuple_element_kinds`."""
    if isinstance(target, ast.Name):
        return [(target.id, dtype_kind(value, dtypes, calls))]
    if not isinstance(target, (ast.Tuple, ast.List)):
        return []
    kinds = tuple_element_kinds(value, dtypes, calls)
    if kinds is None or len(kinds) != len(target.elts):
        return []
    return [(elt.id, kind) for elt, kind in zip(target.elts, kinds) if isinstance(elt, ast.Name)]


def tuple_element_kinds(value: ast.expr, dtypes: dict[str, str], calls: CallKinds) -> list[str | None] | None:
    """Per-element kinds of a value unpacked into a tuple target; ``None`` for a value of no known arity.

    A tuple literal gives each element's kind, a helper its proven element kinds, and ``eigh`` real
    eigenvalues with eigenvectors of its operands' kind promoted to at least float."""
    if isinstance(value, (ast.Tuple, ast.List)):
        if any(isinstance(elt, ast.Starred) for elt in value.elts):
            return None
        return [dtype_kind(elt, dtypes, calls) for elt in value.elts]
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in calls.tuples:
        return list(calls.tuples[value.func.id])
    hit = eigh_call_kind(value, set())
    if hit is None or hit[0] != "eigh":
        return None
    unused, a_node, b_node, kw = hit
    only = kw.get("eigvals_only")
    if only is not None and not (isinstance(only, ast.Constant) and only.value is False):
        return None
    vectors = promote_kind(dtype_kind(a_node, dtypes, calls), "float")
    if b_node is not None:
        vectors = promote_kind(vectors, dtype_kind(b_node, dtypes, calls))
    return ["float", vectors]


#: A helper parameter ``(function, name)`` or what it returns ``(function, index)``: -1 the whole value, 0.. one
#: element of a returned tuple.
KindSlot = tuple[str, str | int]


NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def scope_nodes(fn: ast.FunctionDef) -> list[ast.AST]:
    """Every node of ``fn``'s body outside nested functions, lambdas and classes."""
    out: list[ast.AST] = []
    stack: list[ast.AST] = [stmt for stmt in fn.body if not isinstance(stmt, NESTED_SCOPES)]
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(child for child in ast.iter_child_nodes(node) if not isinstance(child, NESTED_SCOPES))
    return out


def returned_values(fn: ast.FunctionDef) -> list[ast.expr]:
    """The value of every ``return`` in ``fn``; empty when ``fn`` is a generator or may return ``None`` (a bare
    ``return``, or a body that does not end in ``return``/``raise``)."""
    if not fn.body or not isinstance(fn.body[-1], (ast.Return, ast.Raise)):
        return []
    values: list[ast.expr] = []
    for node in scope_nodes(fn):
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return []
        if isinstance(node, ast.Return):
            if node.value is None:
                return []
            values.append(node.value)
    return values


def return_arity(values: list[ast.expr]) -> int | None:
    """-1 when no returned value is a tuple literal, n when every one is an n-tuple, ``None`` otherwise."""
    tuples = [value for value in values if isinstance(value, ast.Tuple)]
    if not values or any(isinstance(elt, ast.Starred) for tup in tuples for elt in tup.elts):
        return None
    if not tuples:
        return -1
    arities = {len(tup.elts) for tup in tuples}
    return arities.pop() if len(tuples) == len(values) and len(arities) == 1 else None


class HelperSites(NamedTuple):
    """Every call of a module-level function by name with the function it sits in, and the functions whose
    name is read any other way (a value, a rebinding, a call from module scope): their sites are not all known."""

    calls: dict[str, list[tuple[str, ast.Call]]]
    escaped: set[str]


def helper_sites(tree: ast.Module, funcs: dict[str, ast.FunctionDef]) -> HelperSites:
    """The :class:`HelperSites` of ``funcs`` in ``tree``."""
    calls: dict[str, list[tuple[str, ast.Call]]] = {name: [] for name in funcs}
    callee_nodes: set[int] = set()
    for owner, fn in funcs.items():
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
                calls[node.func.id].append((owner, node))
                callee_nodes.add(id(node.func))
    names = (node for node in ast.walk(tree) if isinstance(node, ast.Name))
    return HelperSites(calls, {node.id for node in names if node.id in funcs and id(node) not in callee_nodes})


def parameter_names(fn: ast.FunctionDef) -> list[str]:
    """``fn``'s named parameters: positional-only, positional, keyword-only."""
    return [p.arg for p in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]


def argument_kinds(
    call: ast.Call, fn: ast.FunctionDef, dtypes: dict[str, str], calls: CallKinds
) -> dict[str, str | None]:
    """Kind each parameter of ``fn`` binds at ``call`` (positional, keyword or default); all unknown when a
    starred or ``**`` argument hides which parameter gets what."""
    names = parameter_names(fn)
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(k.arg is None for k in call.keywords):
        return {name: None for name in names}
    positional = [p.arg for p in fn.args.posonlyargs + fn.args.args]
    defaults = list(zip(positional[len(positional) - len(fn.args.defaults) :], fn.args.defaults))
    defaults += [(p.arg, d) for p, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if d is not None]
    kinds: dict[str, str | None] = {name: dtype_kind(default, {}) for name, default in defaults}
    kinds.update((name, dtype_kind(arg, dtypes, calls)) for name, arg in zip(positional, call.args))
    kinds.update((k.arg, dtype_kind(k.value, dtypes, calls)) for k in call.keywords if k.arg is not None)
    return {name: kinds.get(name) for name in names}


class KindModule(NamedTuple):
    """What :func:`module_kind_tables` reads once: the top-level functions, the kernel and its declared kinds,
    every call site, and each function's :func:`return_arity`."""

    funcs: dict[str, ast.FunctionDef]
    kernel_name: str
    kernel_kinds: dict[str, str]
    sites: HelperSites
    arities: dict[str, int]


def assumed_call_kinds(module: KindModule, assumed: dict[KindSlot, str]) -> CallKinds:
    """The helper return kinds ``assumed`` holds, in the form :func:`dtype_kind` reads."""
    values = {name: assumed[(name, -1)] for name in module.arities if (name, -1) in assumed}
    tuples = {
        name: tuple(assumed.get((name, i)) for i in range(arity))
        for name, arity in module.arities.items()
        if arity >= 0
    }
    return CallKinds(values, tuples)


def parameter_seed(module: KindModule, name: str, assumed: dict[KindSlot, str]) -> dict[str, str]:
    """The kernel's declared kinds, or a helper's ``assumed`` parameter kinds."""
    if name == module.kernel_name:
        return dict(module.kernel_kinds)
    return {p: assumed[(name, p)] for p in parameter_names(module.funcs[name]) if (name, p) in assumed}


def slot_observations(
    module: KindModule, assumed: dict[KindSlot, str]
) -> tuple[dict[str, dict[str, str]], dict[KindSlot, set[str | None]]]:
    """Every function's kind table under ``assumed``, and each slot's kinds under those tables: one per
    ``return`` and per call site, ``None`` at a site that cannot be read."""
    calls = assumed_call_kinds(module, assumed)
    tables = {name: dtype_table_(fn, parameter_seed(module, name, assumed), calls) for name, fn in module.funcs.items()}
    seen: dict[KindSlot, set[str | None]] = {}
    for name, fn in module.funcs.items():
        arity = module.arities.get(name)
        if arity is None:
            continue
        for value in returned_values(fn):
            elements = value.elts if arity >= 0 and isinstance(value, ast.Tuple) else [value]
            for index, elt in enumerate(elements):
                seen.setdefault((name, index if arity >= 0 else -1), set()).add(dtype_kind(elt, tables[name], calls))
    for callee, at in module.sites.calls.items():
        for owner, call in at if callee != module.kernel_name else []:
            for param, kind in argument_kinds(call, module.funcs[callee], tables[owner], calls).items():
                seen.setdefault((callee, param), set()).add(kind)
    for callee in module.sites.escaped - {module.kernel_name}:
        for param in parameter_names(module.funcs[callee]):
            seen.setdefault((callee, param), set()).add(None)
    return tables, seen


def grow_assumptions(
    seen: dict[KindSlot, set[str | None]], assumed: dict[KindSlot, str], refuted: set[KindSlot]
) -> bool:
    """Assume each slot's one KNOWN kind, unknown sites ignored; a slot seen with two known kinds is refuted
    for good. True when anything changed; each slot is assumed and refuted at most once, so this ends."""
    changed = False
    for slot, kinds in seen.items():
        known = {kind for kind in kinds if kind is not None}
        if slot in refuted or not known:
            continue
        if len(known) > 1 or (slot in assumed and known != {assumed[slot]}):
            refuted.add(slot)
            assumed.pop(slot, None)
            changed = True
        elif slot not in assumed:
            assumed[slot] = known.pop()
            changed = True
    return changed


def retract_assumptions(seen: dict[KindSlot, set[str | None]], assumed: dict[KindSlot, str]) -> bool:
    """Drop every assumption some return or call site did not reproduce exactly. True when any was dropped."""
    refuted = [slot for slot, kind in assumed.items() if seen.get(slot) != {kind}]
    for slot in refuted:
        del assumed[slot]
    return bool(refuted)


def module_kind_tables(tree: ast.Module, kernel_name: str, kernel_kinds: dict[str, str]) -> dict[str, dict[str, str]]:
    """Dtype-KIND tables of the kernel and every top-level helper, kinds carried across calls.

    A helper parameter takes the kind its argument has at EVERY call site; a helper's return (per element of
    a returned tuple) the kind of every ``return``. Cycles resolve optimistically: each slot is first assumed
    from its known sites, then every assumption a round does not reproduce exactly is retracted until none
    is. What survives reproduces itself from the kernel's declared kinds; everything else stays unknown."""
    funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    arities = {name: arity for name, fn in funcs.items() if (arity := return_arity(returned_values(fn))) is not None}
    module = KindModule(funcs, kernel_name, kernel_kinds, helper_sites(tree, funcs), arities)
    assumed: dict[KindSlot, str] = {}
    refuted: set[KindSlot] = set()
    tables, seen = slot_observations(module, assumed)
    while grow_assumptions(seen, assumed, refuted):
        tables, seen = slot_observations(module, assumed)
    while retract_assumptions(seen, assumed):
        tables, seen = slot_observations(module, assumed)
    return tables


#: ``(function, name)`` -> every kind observed flowing into that parameter or local.
KindObservations = dict[tuple[str, str], set[str | None]]


def agreed_return_kinds(funcs: list[ast.FunctionDef], tables: dict[str, dict[str, str]]) -> dict[str, str]:
    """Each function whose ``return`` values all have one known kind -> that kind."""
    returns: dict[str, str] = {}
    for fn in funcs:
        kinds = {dtype_kind(r.value, tables[fn.name]) for r in ast.walk(fn) if isinstance(r, ast.Return) and r.value}
        kind = next(iter(kinds)) if len(kinds) == 1 else None
        if kind is not None:
            returns[fn.name] = kind
    return returns


def observe_helper_calls(
    fn: ast.FunctionDef,
    table: dict[str, str],
    by_name: dict[str, ast.FunctionDef],
    returns: dict[str, str],
    observed: KindObservations,
) -> None:
    """Add to ``observed`` the kind of each positional argument ``fn`` passes a helper, and the return kind
    of each name ``fn`` binds to a helper call."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in by_name:
                observed.setdefault((fn.name, node.targets[0].id), set()).add(returns.get(value.func.id))
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in by_name):
            continue
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            continue
        params = [a.arg for a in by_name[node.func.id].args.args]
        for pname, arg in zip(params, node.args):
            kind = dtype_kind(arg, table)
            if kind is None and isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                kind = returns.get(arg.func.id)
            observed.setdefault((node.func.id, pname), set()).add(kind)


def infer_param_kinds(
    funcs: list[ast.FunctionDef], kernel_name: str, kernel_kinds: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Per-function dtype-KIND seeds across helper boundaries, iterated to a fixpoint.

    A helper parameter takes the kind every call site passes (sites that disagree or cannot be typed leave it
    unknown); a name bound to a helper call takes that helper's return kind. Only call sites the kernel
    reaches are read: numba compiles nothing else, and an unreached helper's untyped argument would veto."""
    by_name = {fn.name: fn for fn in funcs}
    reachable = reachable_functions(funcs, kernel_name)
    seeds: dict[str, dict[str, str]] = {fn.name: {} for fn in funcs}
    if kernel_name in seeds:
        seeds[kernel_name] = dict(kernel_kinds)
    for unused in range(6):
        tables = {fn.name: dtype_table_(fn, seeds[fn.name]) for fn in funcs}
        returns = agreed_return_kinds(funcs, tables)
        observed: KindObservations = {}
        for fn in funcs:
            if fn.name in reachable:
                observe_helper_calls(fn, tables[fn.name], by_name, returns, observed)
        new: dict[str, dict[str, str]] = {name: dict(seed) for name, seed in seeds.items()}
        for (owner, name), kinds in observed.items():
            kind = next(iter(kinds)) if len(kinds) == 1 else None
            if kind is None or (owner == kernel_name and name in kernel_kinds):
                continue
            new[owner][name] = kind
        if new == seeds:
            break
        seeds = new
    return seeds
