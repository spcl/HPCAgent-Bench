"""Timestep-loop detection carried on the IR.

A *source-form* judgement (not a dependence analysis): a ``for t in
range(TSTEPS)`` loop steps time -- each iteration depends on the last -- so it
must stay rolled (never unrolled or vectorized). The imperative backends
(C / Fortran) emit loops regardless; JAX consults this so a timestep loop lowers
to ``lax.fori_loop`` / ``while_loop`` and never unrolls.
"""
import ast
from typing import Tuple

#: Symbol-name fragments that mark a time-stepping loop bound (OptArena /
#: polybench convention). Matched case-insensitively as a substring so
#: ``TSTEPS`` / ``t_steps`` / ``NITER`` all count.
TIMESTEP_SYMBOLS: Tuple[str, ...] = (
    "TSTEPS", "TSTEP", "TIMESTEPS", "TMAX", "NITER", "NSTEPS", "NTIMESTEPS", "NTIME",
)


def _range_bound_names(node: ast.For):
    """Names referenced in the ``range(...)`` bounds of a ``for`` loop, or
    ``None`` if the loop is not a ``for x in range(...)``."""
    it = node.iter
    if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
            and it.func.id == "range"):
        return None
    names = set()
    for arg in it.args:
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Name):
                names.add(sub.id)
    return names


def is_timestep_loop(node: ast.AST, timestep_symbols: Tuple[str, ...] = TIMESTEP_SYMBOLS) -> bool:
    """True when ``node`` is a ``for`` loop whose range bound references a
    time-stepping symbol (so it must stay rolled, never unrolled)."""
    if not isinstance(node, ast.For):
        return False
    names = _range_bound_names(node)
    if not names:
        return False
    syms = tuple(s.lower() for s in timestep_symbols)
    return any(s in nm.lower() for nm in names for s in syms)


# --------------------------------------------------------------------------- #
# Parallel-loop analysis (OpenMP parallel-scope emission).
#
# A *source-form* dependence check (not a full polyhedral analysis): decide,
# from a loop's AST alone, whether ``for idx in range(...)`` can be emitted as a
# ``#pragma omp parallel for`` -- possibly with a ``reduction(op:acc)`` clause --
# without changing results. Errs toward serial: anything not proven independent
# stays a plain loop. The imperative backends (C / Fortran) consume these; the
# same predicates gate numba's ``prange`` (single source of truth).
# --------------------------------------------------------------------------- #


class UnsupportedParallelError(NotImplementedError):
    """The kernel cannot be soundly emitted as an OpenMP-parallel region without
    an atomic (a data-dependent / colliding scatter) or has no parallelizable
    loop at all. The parallel emit variant raises this; the caller falls back to
    the sequential emitter (which is always valid)."""


def index_exprs(sub: ast.Subscript) -> list:
    """The per-axis index expressions of a subscript (``A[i, j]`` -> ``[i, j]``)."""
    sl = sub.slice
    return list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]


def reads_name(node: ast.AST, name: str) -> bool:
    """True when ``name`` is referenced anywhere in ``node``."""
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def is_range_for(node: ast.AST) -> bool:
    """True for a plain ``for <name> in range(...)`` loop (single-name target)."""
    return (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range")


def subscript_idx_safe(sub: ast.Subscript, idx: str) -> bool:
    """A subscript of a WRITTEN array is cross-iteration independent under a
    parallel loop on ``idx`` iff ``idx`` appears as a BARE index in >=1 axis and
    never in a derived form. Iteration ``i`` then only ever touches the ``i``-th
    slice, so reordering iterations cannot make one read a cell another wrote.

    Refused (returns False):
    * ``idx`` shifted / scaled (``A[i-1]``, ``A[2*i]``) -- a stencil straddles iterations;
    * ``idx`` inside a data-dependent index (``A[perm[i]]``) -- unknown collision pattern;
    * ``idx`` absent (``out[0]``, ``A[j]``) -- a same-cell write every iteration (reduction).
    """
    bare = False
    for e in index_exprs(sub):
        if isinstance(e, ast.Name) and e.id == idx:
            bare = True
            continue
        if reads_name(e, idx):
            return False  # idx appears, but not as a bare axis index.
    return bare


def written_arrays(body: ast.AST) -> set:
    """Names of arrays written (via a subscript store or aug-store) anywhere in ``body``."""
    out: set = set()
    for n in ast.walk(body):
        targets = n.targets if isinstance(n, ast.Assign) else ([n.target] if isinstance(n, ast.AugAssign) else [])
        for t in targets:
            if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                out.add(t.value.id)
    return out


def loop_is_parallel_safe(node: ast.AST) -> bool:
    """Conservatively decide whether ``for idx in range(...)`` can run in parallel
    (independent iterations) without changing results. Errs toward serial: any
    pattern not proven independent returns False. Scalar reductions / carried
    scalars are rejected here -- they are handled by :func:`loop_reduction`."""
    if not is_range_for(node):
        return False
    idx = node.target.id
    body = ast.Module(body=list(node.body), type_ignores=[])
    for n in ast.walk(body):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and reads_name(n.value, t.id):
                    return False  # self-referential carried scalar (``s = s + ...``).
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            return False  # scalar reduction / accumulator (``s += ...``).
    written = written_arrays(body)
    for n in ast.walk(body):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in written:
            if not subscript_idx_safe(n, idx):
                return False
    return True


#: Call-leaf names that are a max / min combiner (bare ``max`` or ``np.maximum`` / ``fmax`` ...).
_MAX_NAMES = frozenset({"max", "maximum", "amax", "fmax"})
_MIN_NAMES = frozenset({"min", "minimum", "amin", "fmin"})
#: Python aug-assign op -> OpenMP reduction operator (only the associative ones we can express).
_AUG_REDUCTION_OP = {ast.Add: "+", ast.Mult: "*"}


def _call_leaf(func: ast.AST):
    """The bare / attribute-leaf name of a call target (``max`` or ``np.maximum`` -> the last name)."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def reduction_op(value: ast.AST, acc: str):
    """If ``value`` combines the accumulator ``acc`` with one other term under an
    OpenMP-expressible associative operator, return the clause operator
    ('+', '*', 'max', 'min'); else None. Recognizes ``acc + x`` / ``acc * x``
    (either operand order) and ``max(acc, x)`` / ``np.maximum(acc, x)`` / min."""
    if isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Mult)):
        if _reads_acc(value.left, acc) or _reads_acc(value.right, acc):
            return "+" if isinstance(value.op, ast.Add) else "*"
        return None
    if isinstance(value, ast.Call):
        leaf = _call_leaf(value.func)
        if leaf in _MAX_NAMES and any(_reads_acc(a, acc) for a in value.args):
            return "max"
        if leaf in _MIN_NAMES and any(_reads_acc(a, acc) for a in value.args):
            return "min"
    return None


def _reads_acc(node: ast.AST, acc: str) -> bool:
    return isinstance(node, ast.Name) and node.id == acc


def loop_reduction(node: ast.AST):
    """If ``node`` is a ``for`` loop whose ONLY cross-iteration dependence is a
    single scalar accumulator combined under an OpenMP-expressible operator
    (+, *, max, min), return ``(op, acc_name)`` for a ``reduction(op:acc)``
    clause; else None. Every array write must be iteration-independent, so the
    accumulator is the sole race a reduction clause must cover."""
    if not is_range_for(node):
        return None
    idx = node.target.id
    body = ast.Module(body=list(node.body), type_ignores=[])
    accs: dict = {}
    for n in ast.walk(body):
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            op = _AUG_REDUCTION_OP.get(type(n.op))
            if op is None:
                return None  # a scalar aug-op we cannot express as a reduction
            if accs.get(n.target.id, op) != op:
                return None  # same name accumulated under two different ops
            accs[n.target.id] = op
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if not isinstance(t, ast.Name):
                    continue
                op = reduction_op(n.value, t.id)
                if op is not None:
                    if accs.get(t.id, op) != op:
                        return None
                    accs[t.id] = op
                elif reads_name(n.value, t.id):
                    return None  # self-referential scalar that is not a known reduction
    if len(accs) != 1:
        return None  # 0 -> not a reduction; >1 -> too complex to clause soundly
    acc, op = next(iter(accs.items()))
    written = written_arrays(body)
    for n in ast.walk(body):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in written:
            if not subscript_idx_safe(n, idx):
                return None
    return op, acc


def _index_is_indirect(sub: ast.Subscript) -> bool:
    """True if any axis index of ``sub`` is data-dependent -- contains a nested
    subscript (``A[perm[i]]``), i.e. a gathered index that may collide on write."""
    return any(any(isinstance(x, ast.Subscript) for x in ast.walk(e)) for e in index_exprs(sub))


def has_indirect_scatter(tree: ast.AST) -> bool:
    """True if ``tree`` contains a store whose target index is data-dependent (a
    colliding scatter, ``A[perm[i]] = ...`` / ``+=``). The parallel emitter
    refuses such a kernel rather than emit an atomic."""
    for n in ast.walk(tree):
        targets = n.targets if isinstance(n, ast.Assign) else ([n.target] if isinstance(n, ast.AugAssign) else [])
        for t in targets:
            if isinstance(t, ast.Subscript) and _index_is_indirect(t):
                return True
    return False


def any_parallelizable_loop(tree: ast.AST) -> bool:
    """True if ``tree`` has at least one non-timestep ``for`` loop that is either
    iteration-independent or a recognized scalar reduction -- i.e. the parallel
    variant would emit at least one ``#pragma omp parallel for``."""
    return any(
        not is_timestep_loop(n) and (loop_is_parallel_safe(n) or loop_reduction(n) is not None)
        for n in ast.walk(tree) if isinstance(n, ast.For))
