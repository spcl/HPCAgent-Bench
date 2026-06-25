"""Loop / op parallelism classification carried on the IR.

A *source-form* judgement (not a dependence analysis): the rule honors how the
author wrote the kernel rather than proving independence.

* a **timestep loop** (``for t in range(TSTEPS)``) -> :data:`SEQ` -- each step
  depends on the last; never unroll or vectorize it.
* an **independent element-wise range for-loop** (``for i in range(N): a[i] =
  f(b[i], ...)`` -- target is exactly ``a[i]``, the RHS neither reads the array
  it writes nor uses ``i`` outside a subscript) -> :data:`PARALLEL` -- it is a
  pure map that lowers to a whole-array op.
* any **other range for-loop** (loop-carried state, a ``break``, a non-``i``
  subscript, or a bare-index use) -> :data:`SEQ` -- it must stay an iterative
  loop.
* a **slice / whole-array op** (``a[1:-1] = b[:-2] + b[2:]``) -> :data:`PARALLEL`
  -- the author wrote the vectorized form, so it is data-parallel by
  construction.

The imperative backends (C / Fortran) **ignore** this -- they emit loops
regardless. JAX **uses** it as the loop-lowering decision: ``PARALLEL`` lowers
to a whole-array op, ``SEQ`` lowers to ``lax.fori_loop`` / ``while_loop`` (never
auto-vectorized), and a timestep ``SEQ`` additionally never unrolls. The
element-wise test mirrors the JAX emitter's own ``VECTORIZE`` criterion, so
``PARALLEL`` is exactly the set the emitter can devectorise -- consuming the
annotation cannot regress its vectorization.
"""
import ast
from typing import Optional, Tuple

SEQ = "seq"
PARALLEL = "parallel"

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


def _is_slice_op(node: ast.AST) -> bool:
    """True when ``node`` is a whole-array / slice assignment (a target
    subscript whose index contains a ``Slice``)."""
    if not isinstance(node, (ast.Assign, ast.AugAssign)):
        return False
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for t in targets:
        if isinstance(t, ast.Subscript):
            sl = t.slice
            elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            if any(isinstance(e, ast.Slice) for e in elts):
                return True
    return False


def _names_loaded(node: ast.AST) -> set:
    """Set of ``Name`` ids read (``Load`` context) anywhere under ``node``."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _index_is_exactly(slc: ast.AST, name: str) -> bool:
    """True when a subscript index is exactly the bare name ``name`` (``a[i]``),
    not ``a[i - 1]`` / ``a[i, j]`` / ``a[2]``."""
    return isinstance(slc, ast.Name) and slc.id == name


def _exact_index_names(tree: ast.AST, name: str) -> set:
    """The set of ``Name`` *nodes* (by identity) that are the **whole** index of
    a subscript indexed by exactly ``name`` -- the ``i`` in ``a[i]``, but not
    the ``i`` inside ``a[i - 1]`` (a compound slice)."""
    out = set()
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Subscript) and _index_is_exactly(sub.slice, name):
            out.add(id(sub.slice))
    return out


def _is_independent_elementwise(node: ast.For) -> bool:
    """True when ``for i in range(...)`` is a pure element-wise map the emitter
    can devectorise: no ``break``, the index target is a bare ``Name`` ``i``,
    and **every** body statement is ``arr[i] = expr`` where the target subscript
    is exactly ``i``, the RHS does not read ``arr`` (the array it writes -> no
    loop-carried state), and every use of ``i`` in the RHS is a *whole* ``x[i]``
    subscript. A bare-index use (``a[i] = i``) or an offset read (``a[i] =
    b[i - 1]``) keeps ``i`` after the subscripts are dropped, so the loop is not
    a clean whole-array op -> :data:`SEQ`.

    This is the JAX emitter's ``VECTORIZE`` criterion, lifted to the IR so the
    annotation and the emitter agree on exactly which loops devectorise."""
    target = node.target
    if not isinstance(target, ast.Name):
        return False
    i = target.id
    if any(isinstance(n, ast.Break) for n in ast.walk(node)):
        return False
    if not node.body:
        return False
    for s in node.body:
        if not (isinstance(s, ast.Assign) and len(s.targets) == 1):
            return False
        tgt = s.targets[0]
        if not (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name)):
            return False
        if not _index_is_exactly(tgt.slice, i):
            return False
        if tgt.value.id in _names_loaded(s.value):  # reads the array it writes
            return False
        # Every ``i`` in the RHS must be the whole index of a subscript (``x[i]``);
        # a bare ``i`` or an offset ``x[i - 1]`` leaves ``i`` behind when the
        # ``[i]`` subscripts are dropped, so the map is not devectorisable.
        exact = _exact_index_names(s.value, i)
        for n in ast.walk(s.value):
            if isinstance(n, ast.Name) and n.id == i and id(n) not in exact:
                return False
    return True


def classify(node: ast.AST, timestep_symbols: Tuple[str, ...] = TIMESTEP_SYMBOLS) -> Optional[str]:
    """Parallelism class of a statement node: :data:`PARALLEL` for a slice /
    whole-array op or an independent element-wise ``range`` for-loop,
    :data:`SEQ` for any other ``range`` for-loop (carried state, ``break``,
    timestep, bare-index), or ``None`` when the node is neither a loop nor a
    slice op. A timestep loop is always :data:`SEQ` regardless of its body."""
    if isinstance(node, ast.For) and _range_bound_names(node) is not None:
        if is_timestep_loop(node, timestep_symbols):
            return SEQ
        return PARALLEL if _is_independent_elementwise(node) else SEQ
    if _is_slice_op(node):
        return PARALLEL
    return None
