"""Which ``range`` loops of the generated numba become ``nb.prange``: in every loop nest, the outermost loop PROVEN
independent -- by :func:`~hpcagent_bench.translators.numpyto_common.parallelism.loop_is_parallel_safe`, the predicate
the C / Fortran OpenMP emitters use too, or as a reduction numba's prange carries -- whose written arrays alias
nothing, that hands no array to a helper writing it, and that runs inside no other prange."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.ast_build import ALL_BLOCK_FIELDS, walk_nodes
from hpcagent_bench.translators.numpyto_common.parallelism import (
    bare_store_names,
    index_exprs,
    loop_index,
    loop_is_parallel_safe,
    loop_reduction,
    reads_before_write,
    reads_name,
    scalars_live_after,
    statements_after,
    written_arrays,
)
from hpcagent_bench.translators.numpyto_common.subscripts import base_name
from hpcagent_bench.translators.numpyto_numba.parfor import abs_offset

__all__ = [
    "ALLOCATING_CALLS",
    "MUTATING_METHODS",
    "PRANGE_REDUCTION_OPS",
    "WRITING_FUNCTIONS",
    "binds_name",
    "callers_first",
    "hands_a_written_array_twice",
    "calls_a_mutating_helper",
    "handed_to_written_params",
    "index_live_after",
    "is_allocation",
    "leaves_the_loop",
    "may_view",
    "module_calls",
    "names_written_by_call",
    "names_written_in",
    "numba_reduction",
    "outermost_safe_loops",
    "parallelize_range_loops",
    "prange_safe",
    "range_loop",
    "scalar_literal",
    "statement_blocks",
    "unaliased",
    "unit_step",
    "private_scalars",
    "without_updates",
    "writes_first_argument",
    "written_in_body",
    "written_parameters",
]


def unit_step(call: ast.Call) -> bool:
    """True when ``range(...)``'s step is the literal ``1`` (or omitted): numba's parfor pass lowers
    only that prange, and raises ``UnsupportedRewriteError`` on any other step -- a negative
    literal (``range(n - 1, -1, -1)``) or a runtime one (``range(k, n, m)``) alike."""
    if len(call.args) < 3:
        return True
    step = call.args[2]
    return isinstance(step, ast.Constant) and type(step.value) is int and step.value == 1


#: Array methods that write into the array they are called on (``a.fill(0.0)``).
MUTATING_METHODS = frozenset({"fill", "sort", "put", "partition", "itemset"})


#: ``np.<name>(a, ...)`` functions that write into their FIRST argument, and ufunc methods (``np.add.at(a, ...)``).
WRITING_FUNCTIONS = frozenset({"copyto", "put", "putmask", "place", "fill_diagonal", "put_along_axis"})


def writes_first_argument(call: ast.Call) -> bool:
    """``np.copyto(a, b)``-style functions and ``np.add.at(a, i, v)``: ``a`` is written in place."""
    fn = call.func
    if not isinstance(fn, ast.Attribute):
        return False
    if fn.attr in WRITING_FUNCTIONS:
        return True
    return fn.attr == "at" and isinstance(fn.value, ast.Attribute)


def handed_to_written_params(call: ast.Call, params: list[str], written: frozenset[str]) -> set[str]:
    """Names of the arrays ``call`` hands to a helper parameter in ``written``, positionally or by
    keyword."""
    handed = [(params[i], a) for i, a in enumerate(call.args) if i < len(params)]
    handed += [(k.arg, k.value) for k in call.keywords if k.arg is not None]
    return {name for p, a in handed if p in written and (name := base_name(a)) is not None}


def names_written_by_call(call: ast.Call, mutates: dict[str, frozenset[str]], params: dict[str, list[str]]) -> set[str]:
    """Names of the arrays ``call`` writes into: through a helper in ``mutates``, an ``out=``
    argument, or a :data:`MUTATING_METHODS` method."""
    fn = call.func
    if isinstance(fn, ast.Name) and fn.id in mutates:
        return handed_to_written_params(call, params[fn.id], mutates[fn.id])
    names = {name for k in call.keywords if k.arg == "out" and (name := base_name(k.value)) is not None}
    if call.args and writes_first_argument(call) and (name := base_name(call.args[0])) is not None:
        names.add(name)
    if isinstance(fn, ast.Attribute) and fn.attr in MUTATING_METHODS and (name := base_name(fn.value)) is not None:
        names.add(name)
    return names


def names_written_in(fn: ast.FunctionDef, mutates: dict[str, frozenset[str]], params: dict[str, list[str]]) -> set[str]:
    """Names of the arrays ``fn`` writes into, closed over views: ``row = p[i]; row[j] = 0.0``
    writes ``p``."""
    stored = {
        name
        for n in walk_nodes(fn)
        if isinstance(n, (ast.Assign, ast.AugAssign))
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        if isinstance(t, ast.Subscript) and (name := base_name(t)) is not None
    }
    # ``p += expr`` on a whole array writes into the caller's buffer, like a subscript store does.
    stored |= {n.target.id for n in walk_nodes(fn) if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)}
    for n in walk_nodes(fn):
        if isinstance(n, ast.Call):
            stored |= names_written_by_call(n, mutates, params)
    views = [
        (t.id, base)
        for n in walk_nodes(fn)
        if isinstance(n, ast.Assign) and (base := base_name(n.value)) is not None
        for t in n.targets
        if isinstance(t, ast.Name)
    ]
    while grown := {base for view, base in views if view in stored and base not in stored}:
        stored |= grown
    return stored


def written_parameters(tree: ast.Module) -> tuple[dict[str, frozenset[str]], dict[str, list[str]]]:
    """Module-level function name -> the names of the parameters a call to it writes into, and
    name -> its parameter list.

    A write counts through a subscript store (``p[...] = ...``, ``p[...] += ...``), an in-place ``p += ...``, a view of the
    parameter (``row = p[i]; row[j] = ...``), an ``out=`` argument or a mutating method, and a call
    to another helper that writes the parameter it is handed -- iterated to a fixed point, so a
    wrapper around a writing helper writes too."""
    fns = [fn for fn in tree.body if isinstance(fn, ast.FunctionDef)]
    params = {fn.name: [a.arg for a in fn.args.args] for fn in fns}
    mutates: dict[str, frozenset[str]] = {fn.name: frozenset() for fn in fns}
    while True:
        grown = {fn.name: frozenset(params[fn.name]) & names_written_in(fn, mutates, params) for fn in fns}
        if grown == mutates:
            return mutates, params
        mutates = grown


def calls_a_mutating_helper(loop: ast.For, mutates: dict[str, frozenset[str]], params: dict[str, list[str]]) -> bool:
    """True if ``loop``'s body hands an array to a module helper that writes into it: the write
    happens in the callee, where the loop's own dependence check cannot see it (rb_sor's time loop
    calls its half-sweep, which updates ``u`` in place for the next step to read)."""
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in mutates
        and handed_to_written_params(n, params[n.func.id], mutates[n.func.id])
        for n in walk_nodes(ast.Module(body=list(loop.body), type_ignores=[]))
    )


#: Calls that allocate a fresh array: a name bound only by one of these aliases nothing.
ALLOCATING_CALLS = frozenset(
    {"zeros", "empty", "ones", "full", "zeros_like", "empty_like", "ones_like", "full_like", "copy", "array"}
)

#: Reductions numba's prange recognises (``+=``, ``*=``, ``acc = acc + x``, ``acc = max(acc, x)`` ...) by
#: the operator :func:`~hpcagent_bench.translators.numpyto_common.parallelism.loop_reduction` reports.
PRANGE_REDUCTION_OPS = frozenset({"+", "*", "max", "min"})


def range_loop(node: ast.AST) -> bool:
    """``for <name> in range(...)`` with a unit step (:func:`unit_step`)."""
    return (
        isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
        and not node.iter.keywords
        and unit_step(node.iter)
    )


def statement_blocks(stmt: ast.stmt) -> list[list[ast.stmt]]:
    """The statement lists ``stmt`` holds: its bodies, branches and ``finally`` (:data:`ALL_BLOCK_FIELDS`), each
    ``except`` handler's body and each ``match`` case's."""
    blocks = [value for field, value in ast.iter_fields(stmt) if field in ALL_BLOCK_FIELDS and isinstance(value, list)]
    if isinstance(stmt, (ast.Try, ast.TryStar)):
        blocks += [handler.body for handler in stmt.handlers]
    if isinstance(stmt, ast.Match):
        blocks += [case.body for case in stmt.cases]
    return blocks


def leaves_the_loop(loop: ast.For) -> bool:
    """A ``break`` / ``continue`` of ``loop`` itself or a ``return`` anywhere in its body: a prange body is one
    iteration with no way out."""

    def walk(stmts: list[ast.stmt], own: bool) -> bool:
        for stmt in stmts:
            if isinstance(stmt, ast.Return) or (own and isinstance(stmt, (ast.Break, ast.Continue))):
                return True
            inner = own and not isinstance(stmt, (ast.For, ast.While))
            if any(walk(block, inner) for block in statement_blocks(stmt)):
                return True
        return False

    return walk(loop.body, True) or walk(loop.orelse, False)


def is_allocation(value: ast.expr) -> bool:
    """``np.zeros(...)`` and the other :data:`ALLOCATING_CALLS`: a value no other name shares."""
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr in ALLOCATING_CALLS
        and isinstance(value.func.value, ast.Name)
    )


def unaliased(fn: ast.FunctionDef, name: str, ranks: dict[str, int]) -> bool:
    """Whether the array ``name`` provably shares its buffer with no other name of ``fn``: a parameter never
    rebound, or a local bound only by allocations (:func:`is_allocation`), and no name bound to an expression
    that may be a view of it (:func:`may_view`). ``ranks`` are the declared ranks of ``fn``'s parameters."""
    params = {a.arg for a in fn.args.args}
    allocated = False
    for n in walk_nodes(fn):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    if name in params or not is_allocation(n.value):
                        return False
                    allocated = True
                elif isinstance(t, (ast.Name, ast.Tuple, ast.List, ast.Starred)) and (
                    name in bare_store_names(t) or may_view(n.value, name, ranks)
                ):
                    return False
        elif isinstance(n, (ast.AnnAssign, ast.NamedExpr, ast.For, ast.withitem, ast.comprehension)):
            target = n.optional_vars if isinstance(n, ast.withitem) else n.target
            if target is not None and name in bare_store_names(target):
                return False
            if isinstance(n, ast.NamedExpr) and may_view(n.value, name, ranks):
                return False
    return name in params or allocated


def may_view(value: ast.expr, name: str, ranks: dict[str, int]) -> bool:
    """Whether ``value`` may evaluate to a view of ``name``: the name itself, a subscript or attribute of it, or
    a call it is handed to (``x.reshape(...)``, ``np.transpose(x)``). Arithmetic on it allocates, and an element
    read (every declared axis indexed by an integer) is a scalar."""
    if isinstance(value, ast.Name):
        return value.id == name
    if isinstance(value, ast.Subscript):
        index = index_exprs(value)
        if (
            isinstance(value.value, ast.Name)
            and value.value.id == name
            and len(index) == ranks.get(name)
            and not any(isinstance(e, ast.Slice) or isinstance(e, ast.Constant) and e.value is None for e in index)
        ):
            return False
        return may_view(value.value, name, ranks)
    if isinstance(value, (ast.Attribute, ast.Starred)):
        return may_view(value.value, name, ranks)
    if isinstance(value, ast.Call):
        return may_view(value.func, name, ranks) or any(may_view(a, name, ranks) for a in value.args)
    if isinstance(value, (ast.Tuple, ast.List)):
        return any(may_view(e, name, ranks) for e in value.elts)
    if isinstance(value, ast.IfExp):
        return may_view(value.body, name, ranks) or may_view(value.orelse, name, ranks)
    return False


def written_in_body(loop: ast.For, mutates: dict[str, frozenset[str]], params: dict[str, list[str]]) -> set[str]:
    """Arrays ``loop`` writes: by a subscript or in-place store, and by any call that writes the array it is
    handed (a writing helper, ``out=``, a mutating method)."""
    body = ast.Module(body=list(loop.body), type_ignores=[])
    written = written_arrays(body)
    written |= {
        n.target.id for n in walk_nodes(body) if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)
    }
    for n in walk_nodes(body):
        if isinstance(n, ast.Call):
            written |= names_written_by_call(n, mutates, params)
    return written


def numba_reduction(loop: ast.For, fn: ast.FunctionDef) -> str | None:
    """The accumulator of a reduction numba's prange carries (:data:`PRANGE_REDUCTION_OPS`), proven by
    :func:`loop_reduction`, or ``None``. The accumulator must be a scalar: every binding of it outside the loop
    is a numeric literal (``s = 0.0``, ``m = -np.inf``), and it is not a parameter. The body updates it once:
    numba refuses a second update (tsvc_2_s319's two ``sum_val = sum_val + ...``)."""
    found = loop_reduction(loop)
    if found is None or found[0] not in PRANGE_REDUCTION_OPS:
        return None
    op, acc = found
    if acc in {a.arg for a in fn.args.args}:
        return None
    combines = [n for n in walk_nodes(ast.Module(body=list(loop.body), type_ignores=[])) if binds_name(n, acc)]
    if len(combines) != 1:
        return None  # numba's parfor reduction takes ONE update of the accumulator per iteration.
    if op in ("max", "min") and not all(
        isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == op
        and len(n.value.args) == 2
        and not n.value.keywords
        for n in combines
    ):
        return None  # numba reduces through the builtin ``max`` / ``min`` only.
    inside = {id(n) for n in combines}
    seeds = [n for n in walk_nodes(fn) if binds_name(n, acc) and id(n) not in inside]
    if not seeds or not all(isinstance(n, ast.Assign) and scalar_literal(n.value) for n in seeds):
        return None
    return acc


def binds_name(node: ast.AST, name: str) -> bool:
    """``name = ...`` / ``name op= ...`` with a bare-name target."""
    if isinstance(node, ast.AugAssign):
        return isinstance(node.target, ast.Name) and node.target.id == name
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


def scalar_literal(value: ast.expr) -> bool:
    """A numeric literal, signed or not, or ``np.inf`` / ``np.nan`` and kin."""
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.USub, ast.UAdd)):
        value = value.operand
    if isinstance(value, ast.Constant):
        return type(value.value) in (int, float)
    return isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.attr in ("inf", "nan")


def private_scalars(loop: ast.For) -> set[str]:
    """Scalars every iteration binds afresh before reading them: the first top-level statement of the body that
    mentions one is a plain ``s = <expr not reading s>``. Their updates later in the iteration (``s += ...``,
    the row accumulator of a matvec) are private to it."""
    seen: set[str] = set()
    private: set[str] = set()
    for stmt in loop.body:
        mentioned = {n.id for n in walk_nodes(stmt) if isinstance(n, ast.Name)} - seen
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id in mentioned
            and not reads_name(stmt.value, stmt.targets[0].id)
        ):
            private.add(stmt.targets[0].id)
        seen |= mentioned
    return private


def without_updates(loop: ast.For, names: set[str]) -> ast.For:
    """A copy of ``loop`` whose updates of ``names`` (``s op= e``, ``s = f(s, e)``) are bare reads of their value
    (``e``, ``f(s, e)``): what they read still counts for the independence check, the dependence they carry on
    ``s`` does not."""
    probe = copy.deepcopy(loop)

    class Neutralise(ast.NodeTransformer):
        def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
            if isinstance(node.target, ast.Name) and node.target.id in names:
                return ast.copy_location(ast.Expr(value=node.value), node)
            return node

        def visit_Assign(self, node: ast.Assign) -> ast.stmt:
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in names and reads_name(node.value, name):
                    return ast.copy_location(ast.Expr(value=node.value), node)
            return node

    probe.body = [Neutralise().visit(stmt) for stmt in probe.body]
    return probe


def index_live_after(fn: ast.FunctionDef, loop: ast.For) -> bool:
    """Whether the loop's own index is read after the loop: Python leaves it at the last iteration's value."""
    after = statements_after(fn, loop)
    return after is None or reads_before_write(after, {loop_index(loop)}, set())


def prange_safe(
    loop: ast.For,
    fn: ast.FunctionDef,
    ranks: dict[str, int],
    mutates: dict[str, frozenset[str]],
    params: dict[str, list[str]],
) -> bool:
    """Whether ``loop`` may run as ``nb.prange``: PROVEN independent (:func:`loop_is_parallel_safe`) once the
    updates of its :func:`private_scalars` are discounted, or a reduction numba carries (:func:`numba_reduction`)
    with everything else proven so; no body scalar but the accumulator read after the loop; every array it
    writes provably unaliased (:func:`unaliased`); no helper writing an array it hands over
    (:func:`calls_a_mutating_helper`); no way out of an iteration (:func:`leaves_the_loop`); and its index not
    read after it."""
    if not range_loop(loop) or leaves_the_loop(loop) or calls_a_mutating_helper(loop, mutates, params):
        return False
    private = private_scalars(loop)
    live = scalars_live_after(fn, loop)
    probe = without_updates(loop, private)
    if live or not loop_is_parallel_safe(probe):
        acc = numba_reduction(loop, fn)
        if acc is None or acc in private or live - {acc}:
            return False
        probe = without_updates(loop, private | {acc})
        if not loop_is_parallel_safe(probe):
            return False
    if index_live_after(fn, loop):
        return False
    # The probe's writes: a private scalar's or the accumulator's update is not an array store.
    return all(unaliased(fn, name, ranks) for name in written_in_body(probe, mutates, params))


def outermost_safe_loops(
    stmts: list[ast.stmt],
    fn: ast.FunctionDef,
    ranks: dict[str, int],
    mutates: dict[str, frozenset[str]],
    params: dict[str, list[str]],
) -> list[ast.For]:
    """In every loop nest under ``stmts``, the outermost loop :func:`prange_safe` proves; its own body is not
    searched again (one prange per nest, never nested)."""
    found: list[ast.For] = []
    for stmt in stmts:
        if isinstance(stmt, ast.For) and prange_safe(stmt, fn, ranks, mutates, params):
            found.append(stmt)
            continue
        for block in statement_blocks(stmt):
            found += outermost_safe_loops(block, fn, ranks, mutates, params)
    return found


def module_calls(node: ast.AST, functions: set[str]) -> set[str]:
    """The module functions called anywhere under ``node``."""
    return {
        n.func.id
        for n in walk_nodes(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in functions
    }


def hands_a_written_array_twice(
    call: ast.Call, mutates: dict[str, frozenset[str]], params: dict[str, list[str]]
) -> bool:
    """Whether ``call`` hands one array to a parameter the callee writes AND to another parameter: inside the
    callee the two names alias, which its own independence check cannot see."""
    if not (isinstance(call.func, ast.Name) and call.func.id in mutates):
        return False
    names = params[call.func.id]
    handed = [(names[i], a) for i, a in enumerate(call.args) if i < len(names)]
    handed += [(k.arg, k.value) for k in call.keywords if k.arg is not None]
    bases = [(p, base_name(a)) for p, a in handed]
    written = {b for p, b in bases if p in mutates[call.func.id] and b is not None}
    return any(sum(1 for unused, b in bases if b == w) > 1 for w in written)


def callers_first(calls: dict[str, set[str]]) -> list[str]:
    """The functions of the call graph ``calls`` (caller -> callees), every caller before its callees; a
    function on a cycle is left out."""
    pending = {name: 0 for name in calls}
    for callees in calls.values():
        for callee in callees:
            pending[callee] += 1
    ready = sorted(name for name, count in pending.items() if count == 0)
    order: list[str] = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        for callee in sorted(calls[name]):
            pending[callee] -= 1
            if pending[callee] == 0:
                ready.append(callee)
    return order


def parallelize_range_loops(
    src: str, serial: frozenset[str] = frozenset(), ranks: dict[str, dict[str, int]] | None = None
) -> tuple[str, frozenset[str]]:
    """Rewrite the outermost proven loop of every loop nest (:func:`outermost_safe_loops`) to ``nb.prange``.

    Returns the source and the module functions that run INSIDE a prange body (called from one, directly or
    not): those must be serial builds and keep no prange of their own -- a parallel region launched from a
    worker thread is nested parallelism, which numba's workqueue layer aborts on. The functions in ``serial``
    (the parfor fixes' serial helpers) are never searched. ``ranks`` maps a function to its parameters' declared
    ranks (the kernel's, from the manifest)."""
    tree = ast.parse(src)
    fns = {fn.name: fn for fn in tree.body if isinstance(fn, ast.FunctionDef)}
    mutates, params = written_parameters(tree)
    aliased = {
        n.func.id
        for n in walk_nodes(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and hands_a_written_array_twice(n, mutates, params)
    }
    chosen = {
        name: outermost_safe_loops(fn.body, fn, (ranks or {}).get(name, {}), mutates, params)
        for name, fn in fns.items()
        if name not in serial and name not in aliased
    }
    calls = {name: module_calls(fn, set(fns)) for name, fn in fns.items()}
    nested: set[str] = set()
    # Callers before callees, so a function's place inside a prange body is settled before its own loops are
    # considered. A function on a call cycle never comes up and keeps no prange.
    for name in callers_first(calls):
        if name in nested:
            continue
        frontier = [callee for loop in chosen.get(name, []) for callee in module_calls(loop, set(fns))]
        while frontier:
            callee = frontier.pop()
            if callee not in nested:
                nested.add(callee)
                frontier += calls[callee]
    ordered = set(callers_first(calls))
    chosen = {name: found for name, found in chosen.items() if name in ordered}
    heads = [
        loop.iter.func
        for name, found in chosen.items()
        if name not in nested
        for loop in found
        if isinstance(loop.iter, ast.Call)
    ]
    for func in sorted(heads, key=lambda f: (f.lineno, f.col_offset), reverse=True):
        off = abs_offset(src, func.lineno, func.col_offset)
        if src[off : off + 5] == "range":
            src = src[:off] + "nb.prange" + src[off + 5 :]
    return src, frozenset(nested)
