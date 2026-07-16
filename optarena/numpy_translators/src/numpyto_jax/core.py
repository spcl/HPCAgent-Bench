# Copyright 2025 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prototype numpy -> JAX emitter.

JAX's ``jnp`` mirrors ``numpy``, so most of the translation is source-level:
rewrite ``np.`` -> ``jnp.``, turn in-place mutation into functional updates
(jax arrays are immutable), and -- the load-bearing decision -- **lower each
Python loop to the right JAX control-flow construct**:

* ``for i in range(N): ...`` with a data-dependent ``break`` (or a ``while``)
  -> :func:`jax.lax.while_loop` carrying state + a ``done`` flag. Statements
  after the break-guard are frozen with ``jnp.where`` on the break condition
  (so the iteration that converges still commits the updates it made before
  the break, and nothing after). This is the iterative-solver shape.
* ``for i in range(N): ...`` with loop-carried state and no break
  -> :func:`jax.lax.fori_loop` carrying the state tuple.
* ``for i in range(N): ...`` whose body only reads/writes element ``i``
  (no carry) -> **vectorised** away (the loop becomes a whole-array op).

The numpy reference often mutates an output in place and returns ``None``;
the emitted kernel instead returns the (functional) output(s) -- the harness
treats a full set of returned values as the outputs.

Scope: a prototype. It covers the elementwise / reduction / matmul / solver
shapes; unsupported constructs raise ``EmitError`` so the driver can fall
back rather than emit something wrong.
"""
from __future__ import annotations

import ast
import copy
from typing import List, Optional, Set


class EmitError(Exception):
    """A numpy construct the prototype does not (yet) lower."""


#: True while emitting the classifier (``jit=True``) form. Read by
#: :func:`_np_to_jnp` to lower a data-dependent ternary ``a if cond else b`` to
#: ``jnp.where`` (a traced Python ternary cannot yield a concrete bool); left
#: verbatim in the eager form, whose concrete condition needs no rewrite.
_JIT_MODE = False


# ---------------------------------------------------------------------------
# Small AST helpers
# ---------------------------------------------------------------------------
def _names_loaded(node: ast.AST) -> Set[str]:
    """Names read (Load context) anywhere under ``node``."""
    out: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add(n.id)
    return out


def _store_target_names(t: ast.AST, out: Set[str]) -> None:
    """Collect the names a single assignment target binds: a plain ``Name``, the
    base array of a ``Subscript`` (``a[i] = ..`` mutates ``a``), each element of
    a tuple/list unpack (``x, y, z = f(...)`` binds all three -- lulesh's per-step
    ``x, y, z, .. = _lagrange_nodal(..)``), and a starred target."""
    if isinstance(t, ast.Subscript):
        base = t
        while isinstance(base, ast.Subscript):
            base = base.value
        if isinstance(base, ast.Name):
            out.add(base.id)
    elif isinstance(t, ast.Name):
        out.add(t.id)
    elif isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            _store_target_names(e, out)
    elif isinstance(t, ast.Starred):
        _store_target_names(t.value, out)


def _names_stored(node: ast.AST) -> Set[str]:
    """Names assigned (a plain ``Name`` target, incl. ``a[i] = ...`` whose base
    array name is mutated, tuple-unpack targets, and augmented assigns)."""
    out: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                _store_target_names(t, out)
    return out


def _has_break(body: List[ast.stmt]) -> bool:
    for s in body:
        for n in ast.walk(s):
            if isinstance(n, ast.Break):
                return True
    return False


# Bare ``from math import sin, sqrt, …`` functions: fine eagerly on a scalar
# ``b[i]``, but once a loop is vectorised (``sin(b)`` over the whole array) or
# traced in a ``fori_loop`` (``sqrt`` on a traced ``b[jg]``), ``math.f`` raises
# -- it only accepts a host Python float. Map them to the elementwise ``jnp``
# ufunc. Most names match; the inverse-trig and power names differ.
_MATH_TO_JNP = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "exp": "exp",
    "log": "log",
    "log2": "log2",
    "log10": "log10",
    "sqrt": "sqrt",
    "pow": "power",
    "floor": "floor",
    "ceil": "ceil",
    "fabs": "fabs",
}


def _np_to_jnp(tree: ast.AST) -> ast.AST:
    """Rewrite ``np.<x>`` -> ``jnp.<x>`` (and bare ``np`` -> ``jnp``)."""

    class _R(ast.NodeTransformer):

        def visit_Name(self, node):
            if node.id == "np":
                return ast.copy_location(ast.Name(id="jnp", ctx=node.ctx), node)
            # optarena injects ``np_float``/``np_complex`` as framework globals;
            # under the x64 config they resolve to the 64-bit dtypes.
            if node.id == "np_float":
                return ast.copy_location(
                    ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="float64", ctx=node.ctx), node)
            if node.id == "np_complex":
                return ast.copy_location(
                    ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="complex128", ctx=node.ctx), node)
            return node

        def visit_Attribute(self, node):
            self.generic_visit(node)
            # ``np.ndarray(shape, dtype=..)`` is a bare uninitialized array
            # constructor; ``jnp`` has no such call -- use ``jnp.empty``.
            if node.attr == "ndarray":
                return ast.copy_location(ast.Attribute(value=node.value, attr="empty", ctx=node.ctx), node)
            # jax arrays have no C/F memory-layout distinction and are always
            # contiguous, so ``np.ascontiguousarray`` / ``np.asfortranarray`` (the
            # latter inserted by ``_rewrite_eigh`` around a non-Name eigh operand,
            # and used directly by gromacs_nbnxm) map to plain ``jnp.asarray``;
            # ``jnp`` has no ``ascontiguousarray``.
            if node.attr in ("ascontiguousarray", "asfortranarray"):
                return ast.copy_location(ast.Attribute(value=node.value, attr="asarray", ctx=node.ctx), node)
            # ``np.intp`` / ``np.uintp`` (the platform-word integer dtypes lulesh
            # uses for gather-index arrays) have no ``jnp`` spelling; under x64
            # they are the 64-bit integer types.
            if node.attr in ("intp", "uintp"):
                return ast.copy_location(
                    ast.Attribute(value=node.value, attr="int64" if node.attr == "intp" else "uint64", ctx=node.ctx),
                    node)
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            # jnp array constructors don't accept numpy's ``order=`` (C/F memory
            # layout) -- jax arrays have no user-facing layout distinction. Drop
            # it from constructors only (jnp.reshape DOES honour ``order=``, so it
            # is left intact). ``np.zeros(s, dtype=.., order='F')`` -> ``jnp.zeros(s, dtype=..)``.
            if (isinstance(node.func, ast.Attribute) and node.func.attr
                    in ("zeros", "ones", "empty", "full", "zeros_like", "ones_like", "empty_like", "full_like")):
                node.keywords = [k for k in node.keywords if k.arg != "order"]
            # Python builtins ``max``/``min`` over two OR MORE (traced) scalar
            # args fold into a left-nested chain of the elementwise jnp ufunc:
            # ``max(0, a, b, c)`` -> ``jnp.maximum(jnp.maximum(jnp.maximum(0, a),
            # b), c)`` (the DP recurrences' 3-/4-way ``max`` -- needleman_wunsch /
            # smith_waterman -- whose traced scalars a Python ``max`` cannot
            # compare). A single-iterable ``max(seq)`` (one arg) is left alone.
            if isinstance(node.func, ast.Name) and node.func.id in ("max", "min") \
                    and len(node.args) >= 2 and not node.keywords:
                attr = "maximum" if node.func.id == "max" else "minimum"
                expr = node.args[0]
                for rhs in node.args[1:]:
                    expr = ast.Call(
                        func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                        args=[expr, rhs],
                        keywords=[])
                return ast.copy_location(expr, node)
            # Bare ``math`` functions (``sin(b)``, ``sqrt(b[jg])``) -> ``jnp``
            # ufuncs so a vectorised / traced argument works (see _MATH_TO_JNP).
            if isinstance(node.func, ast.Name) and node.func.id in _MATH_TO_JNP:
                return ast.copy_location(
                    ast.Call(func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()),
                                                attr=_MATH_TO_JNP[node.func.id],
                                                ctx=ast.Load()),
                             args=node.args,
                             keywords=node.keywords), node)
            # Python's ``float(x)`` builtin forces host concretisation (it must
            # return a Python ``float``), which a traced value cannot provide --
            # so a rolled ``jit`` body ending in ``maxv + float(index)`` (the
            # TSVC argmax checksum) fails to AOT-trace and falls back to slow
            # eager. Route it through a traceable JAX cast instead. ``float`` is
            # always safe to rewrite: its result only ever feeds arithmetic /
            # fill / comparison, never a ``range`` or shape (which would need a
            # concrete int). ``int`` is intentionally *not* rewritten -- an
            # ``int(x)`` can feed ``range(int(x))`` / a shape that requires a
            # concrete Python int, which a traced cast would break in eager; the
            # corpus has no value-position ``int()`` (add context-aware handling
            # if one appears).
            if isinstance(node.func, ast.Name) and node.func.id == "float" and len(node.args) == 1:
                return ast.copy_location(
                    ast.Call(func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()),
                                                attr="asarray",
                                                ctx=ast.Load()),
                             args=[
                                 node.args[0],
                                 ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="float64", ctx=ast.Load())
                             ],
                             keywords=[]), node)
            return node

        def visit_IfExp(self, node):
            self.generic_visit(node)
            # A ternary ``a if cond else b`` whose condition is data-dependent
            # (not a static jit arg / module constant) cannot yield a concrete
            # Python bool under trace -- lower it to the value-select
            # ``jnp.where(cond, a, b)`` (lulesh's Courant limit ``cand if cand <
            # dtcourant else dtcourant``). A static-condition ternary
            # (fv3_dycore's ``8 if hord == 10 else hord`` on a static ``hord``)
            # stays a real Python ternary, so its arms may differ in shape. Only
            # rewritten in the jit form; the eager form runs the concrete ternary.
            # An identity test (``x if del6_v is not None else None``, fv3_dycore's
            # optional damping) is trace-time concrete (a traced array is never
            # ``None``) and may select ``None`` -- keep it a Python ternary, as
            # ``_emit_if`` does for the statement form.
            none_branch = any(isinstance(b, ast.Constant) and b.value is None for b in (node.body, node.orelse))
            if not _JIT_MODE or _is_identity_test(node.test) or none_branch \
                    or _names_loaded(node.test) <= (_EMIT_STATIC | _MODULE_CONSTS):
                return node
            where = ast.Call(func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="where", ctx=ast.Load()),
                             args=[_bool_cond_ast(node.test), node.body, node.orelse],
                             keywords=[])
            return ast.copy_location(where, node)

        def visit_BoolOp(self, node):
            self.generic_visit(node)
            # ``a and b`` / ``a or b`` mean two different things depending on the
            # operands. When EVERY operand is a provably-boolean array expression
            # (a comparison or an ``np.logical_*`` / predicate call), this is a
            # whole-array mask combine, and the faithful lowering is the
            # elementwise ``&`` / ``|`` (a Python bool from a traced array is
            # impossible, and bitwise on booleans is exact).
            #
            # For non-boolean operands Python/numpy ``and``/``or`` return one
            # OPERAND by truthiness (``n = n or N`` yields ``N`` when ``n`` is
            # falsy) -- ``&`` / ``|`` would instead bit-combine the VALUES, which
            # is a miscompile. There is no single traceable rewrite of a general
            # value-select ``and``/``or`` (a full lowering would need
            # ``jnp.where(bool(first), ...)``, itself untraceable on a non-scalar,
            # and short-circuit side effects). In eager mode -- the default and
            # the widest-coverage path -- the operands are concrete host
            # scalars/0-d arrays, so leaving the Python ``and``/``or`` verbatim
            # preserves exact numpy semantics (a jit trace of such a construct
            # would raise honestly rather than silently miscompile).
            if all(_is_bool_expr(v) for v in node.values):
                bitop = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
                expr = node.values[0]
                for rhs in node.values[1:]:
                    expr = ast.BinOp(left=expr, op=bitop, right=rhs)
                return ast.copy_location(expr, node)
            return node

    return _R().visit(tree)


# ---------------------------------------------------------------------------
# Loop lowering -- the core of the emitter
# ---------------------------------------------------------------------------

# A time-stepping loop (``for t in range(TSTEPS)``) must stay rolled -- never
# unrolled into a Python loop -- because each step depends on the previous one
# and unrolling a long trip count blows up the trace. The classification is the
# shared source-form rule in :mod:`numpyto_common.parallelism`, now that JAX
# lives under the common ``numpy_translators`` src and can import it directly.
from numpyto_common.parallelism import is_timestep_loop as _is_timestep_loop  # noqa: E402


class LoopKind:
    VECTORIZE = "vectorize"  # independent elementwise -> whole-array op
    FORI = "fori_loop"  # fixed trip count, loop-carried state
    WHILE = "while_loop"  # data-dependent termination (break / while)


def _index_in_shape(node: ast.For, i: str) -> bool:
    """Does the loop index appear in a shape/count argument (``reshape(_, (R**i,
    …))``, ``zeros((i, …))``) inside the body? Such uses need a concrete index."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in _SHAPE_FUNCS:
            scan = n.args[1:] if n.func.attr in _LEADING_DATA_FUNCS else n.args
            for a in scan:
                if i in _names_loaded(a):
                    return True
    return False


def _classify_for(node: ast.For) -> str:
    """Decide which JAX construct a ``for i in range(...)`` lowers to."""
    if _has_break(node.body):
        return LoopKind.WHILE
    target = node.target
    if not isinstance(target, ast.Name):
        return LoopKind.FORI
    i = target.id
    # Carried iff some variable is both written and (read in the body OR an
    # array written by index that is also read) -- i.e. state threads across
    # iterations. A body that only does ``a[i] = f(<things indexed by i>)``
    # with no other read of ``a`` is independent -> vectorisable.
    stored = _names_stored(ast.Module(body=node.body, type_ignores=[]))
    for s in node.body:
        if not (isinstance(s, ast.Assign) and len(s.targets) == 1):
            return LoopKind.FORI  # anything non-trivial -> safe carried form
        tgt = s.targets[0]
        if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
            # a[<i>] = expr ; independent only if subscript is exactly i, the
            # RHS doesn't read `a` (the array written), and the RHS uses `i`
            # *only* inside ``arr[i]`` subscripts. If `i` appears as a bare
            # scalar (e.g. a fill ``res[i] = rmax * i / npt``), dropping the
            # loop would leave `i` dangling -> keep a fori_loop + .at[].set().
            arr = tgt.value.id
            if arr in _names_loaded(s.value):
                return LoopKind.FORI
            if not _is_index_i(tgt.slice, i):
                return LoopKind.FORI
            if i in _names_loaded(_devectorize_index(s.value, i)):
                return LoopKind.FORI
        else:
            return LoopKind.FORI
    # also: a var written as a plain Name and read => carried
    for s in node.body:
        if isinstance(s.targets[0], ast.Name) and s.targets[0].id in stored:
            return LoopKind.FORI
    return LoopKind.VECTORIZE


def _is_index_i(sl: ast.AST, i: str) -> bool:
    return isinstance(sl, ast.Name) and sl.id == i


def _carried_vars(body: List[ast.stmt], extra_live: Set[str], cond_names: Set[str] = frozenset()) -> List[str]:
    """Variables that genuinely thread across iterations.

    A var is loop-carried iff either it is **read before it is written** in
    the body (a true cross-iteration dependency -- e.g. ``trace += ...`` or
    ``A[i] = f(A[...])``), or it is written and **live after the loop**
    (``extra_live``). A var that is written before any read and is not
    live-out is a loop-*local* temp (e.g. gramschmidt's ``nrm``): it must NOT
    be threaded, else the initial carry tuple references it before it exists.

    ``cond_names`` are the names read in the loop's *own* condition (a
    ``while`` test). That test is evaluated before the body each iteration, so
    any name it reads that the body writes is a genuine cross-iteration carry
    (channel_flow's ``udiff``: ``while udiff > .001`` reads it, the body
    recomputes it). Without this the ``_cond`` closure would capture the
    pre-loop value as a free var and the loop would never terminate.
    """
    stored = _names_stored(ast.Module(body=body, type_ignores=[]))
    carried: Set[str] = set()
    written: Set[str] = set()

    def cond_reads(names):
        # A name read in a loop/if *condition* before being written is a
        # genuine cross-iteration read (s318's ``if v > maxv`` reads the carried
        # ``maxv`` before the branch updates it); the guard means it can never
        # have been written earlier in the same iteration, so it must be carried.
        for nm in names:
            if nm in stored and nm not in written:
                carried.add(nm)

    cond_reads(cond_names)  # the loop's own test is evaluated before the body

    def walk(stmts):
        # Recurse into compound statements so a temp written-then-read *inside*
        # an if/loop (e.g. scattering's dHG/dHD) is seen as local, not carried.
        for s in stmts:
            if isinstance(s, (ast.For, ast.While)):
                cond_reads(_names_loaded(s.iter if isinstance(s, ast.For) else s.test))
                walk(s.body)
            elif isinstance(s, ast.If):
                # A write inside ONE branch of an ``if`` is *conditional*, not a
                # definite write: on the other path the variable keeps its prior
                # (cross-iteration) value, so a later read at the outer level
                # must treat it as carried. Only a var written on BOTH branches
                # is a definite write that kills a subsequent read. (s258:
                # ``if a[i] > 0: s = d[i]*d[i]`` then ``b[i] = s*c[i] + d[i]`` --
                # ``s`` persists across iterations when the guard is false, so it
                # must be carried, not dropped to a ``_body``-local.)
                cond_reads(_names_loaded(s.test))
                saved = set(written)
                walk(s.body)
                wbody = set(written)
                written.clear()
                written.update(saved)
                walk(s.orelse)
                wose = set(written)
                written.clear()
                written.update(wbody & wose)  # definite = written on both paths
            else:
                for nm in _stmt_rhs_loads(s):
                    if nm in stored and nm not in written:
                        carried.add(nm)  # read before write -> cross-iteration
                written.update(_names_stored(s))

    walk(body)
    for nm in stored:
        if nm in extra_live:
            carried.add(nm)  # value escapes the loop
    return sorted(carried)


def _definite_writes(stmts: List[ast.stmt]) -> Set[str]:
    """Names DEFINITELY written by straight-line execution of ``stmts``. A write
    inside a loop (may run zero times) is not definite; a write inside an ``if``
    counts only when it happens on BOTH arms."""
    out: Set[str] = set()
    for s in stmts:
        if isinstance(s, ast.If):
            out |= (_definite_writes(s.body) & _definite_writes(s.orelse))
        elif isinstance(s, (ast.For, ast.While)):
            continue  # a zero-trip loop writes nothing
        else:
            out |= _names_stored(ast.Module(body=[s], type_ignores=[]))
    return out


def _upward_exposed(stmts: List[ast.stmt]) -> Set[str]:
    """Names whose value ENTERING ``stmts`` is observable -- read before being
    (definitely) written. This is the precise live-in of a straight-line block,
    used to decide which vars a preceding loop must thread OUT: a plain
    ``_names_loaded`` over the rest-of-body over-approximates, treating a var the
    rest RE-derives before use (vadv/cloudsc's per-iteration scratch ``bcol`` /
    ``zqadj``, read only inside a *later* loop that recomputes it first) as live,
    which then pulls an undefined name into the earlier loop's carry tuple.

    A loop body's writes are NOT definite (it may run zero times), so a var it
    reads-before-writes is upward-exposed and a var it only writes stays exposed
    for later reads -- both conservative (keeps a carry rather than dropping a
    needed one)."""
    read: Set[str] = set()
    written: Set[str] = set()

    def add_reads(names) -> None:
        for nm in names:
            if nm not in written:
                read.add(nm)

    for s in stmts:
        if isinstance(s, ast.For):
            add_reads(_names_loaded(s.iter))
            add_reads(_upward_exposed(s.body))
        elif isinstance(s, ast.While):
            add_reads(_names_loaded(s.test))
            add_reads(_upward_exposed(s.body))
        elif isinstance(s, ast.If):
            add_reads(_names_loaded(s.test))
            add_reads(_upward_exposed(s.body))
            add_reads(_upward_exposed(s.orelse))
            written |= (_definite_writes(s.body) & _definite_writes(s.orelse))
        else:
            add_reads(_stmt_rhs_loads(s))
            written |= _names_stored(ast.Module(body=[s], type_ignores=[]))
    return read


def _stmt_rhs_loads(s: ast.stmt) -> Set[str]:
    """Names read by a statement, *excluding* a bare ``Name`` assignment
    target (which is a pure write). Subscript-target container reads and all
    RHS reads count. An ``AugAssign`` target is read (augmented update)."""
    if isinstance(s, ast.AugAssign):
        # ``x <op>= v`` reads x and v; ``A[i] <op>= v`` reads A, i and v.
        reads = _names_loaded(s.value) | _names_loaded(s.target)
        reads.add(_base_name(s.target))
        return reads
    if isinstance(s, ast.Assign):
        loads = set().union(*[_names_loaded(t) for t in s.targets]) \
            if s.targets else set()
        # a plain ``x = ...`` target Name is a write, not a read
        for t in s.targets:
            if isinstance(t, ast.Name):
                loads.discard(t.id)
        return loads | _names_loaded(s.value)
    return _names_loaded(s)


# ---------------------------------------------------------------------------
# In-place -> functional rewrite
# ---------------------------------------------------------------------------
def _functionalize_stmt(s: ast.stmt) -> List[ast.stmt]:
    """Rewrite an in-place statement into a functional rebind:

    * ``x <op>= v`` -> ``x = x <op> v`` (then re-process the result, so a
      subscript target like ``A[i] += v`` flows into the ``.at`` form);
    * ``a[idx] = v`` -> ``a = a.at[idx].set(v)``;
    * ``a[:] = v`` -> ``a = v``.

    Other statements pass through unchanged.
    """
    if isinstance(s, ast.AugAssign):
        assign = ast.Assign(targets=[s.target], value=ast.BinOp(left=_load(s.target), op=s.op, right=s.value))
        return _functionalize_stmt(ast.copy_location(assign, s))
    # ``arr.shape = newshape`` is numpy's in-place reshape; jax arrays are
    # immutable -> ``arr = arr.reshape(newshape)``.
    if isinstance(s, ast.Assign) and len(s.targets) == 1 and \
            isinstance(s.targets[0], ast.Attribute) and s.targets[0].attr == "shape" and \
            isinstance(s.targets[0].value, ast.Name):
        name = s.targets[0].value
        call = ast.Call(func=ast.Attribute(value=ast.Name(id=name.id, ctx=ast.Load()), attr="reshape", ctx=ast.Load()),
                        args=[s.value],
                        keywords=[])
        new = ast.Assign(targets=[ast.Name(id=name.id, ctx=ast.Store())], value=call)
        return [ast.copy_location(new, s)]
    if isinstance(s, ast.Assign) and len(s.targets) == 1 and \
            isinstance(s.targets[0], ast.Subscript):
        tgt = s.targets[0]
        # Flatten a chained subscript target ``a[i][j]`` into a single multi-axis
        # index ``a[i, j]`` (it indexes consecutive leading axes, which is
        # numpy-equivalent to tuple indexing for basic indices). A naive
        # ``a[i].at[j].set(v)`` would rebind ``a`` to the single row ``a[i]``,
        # dropping the rest of the array.
        indices: List[ast.expr] = []
        base = tgt
        while isinstance(base, ast.Subscript):
            indices.append(base.slice)
            base = base.value
        indices.reverse()
        arr = base
        arr_name = _base_name(arr)
        sl = indices[0] if len(indices) == 1 else ast.Tuple(elts=indices, ctx=ast.Load())
        name = ast.Name(id=arr_name, ctx=ast.Store())
        if _is_full_slice(sl):
            # ``a[:] = <scalar>`` fills every element -- a plain ``a = <scalar>``
            # would rebind ``a`` to a SCALAR (then ``a.at[...]`` / array uses
            # break). Broadcast via ``jnp.full_like`` so ``a`` stays an array
            # of its original shape/dtype (edge_laplacian's ``Lx[:] = 0.0``).
            if isinstance(s.value, ast.Constant) and not isinstance(s.value.value, str):
                fill = ast.Call(func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()),
                                                   attr="full_like",
                                                   ctx=ast.Load()),
                                args=[ast.Name(id=arr_name, ctx=ast.Load()), s.value],
                                keywords=[])
                new = ast.Assign(targets=[name], value=fill)
            else:
                # ``a[:] = <array-expr>`` is an in-place store: numpy broadcasts
                # the RHS to ``a``'s shape and casts it to ``a``'s dtype. A plain
                # rebind ``a = <expr>`` would instead let ``a`` inherit the RHS's
                # shape/dtype, silently changing the output buffer -- keep the
                # declared shape+dtype via ``broadcast_to(...).astype(a.dtype)``.
                new = ast.Assign(targets=[name], value=_broadcast_astype(arr, s.value))
        else:
            at = ast.Subscript(value=ast.Attribute(value=arr, attr="at", ctx=ast.Load()), slice=sl, ctx=ast.Load())
            call = ast.Call(func=ast.Attribute(value=at, attr="set", ctx=ast.Load()), args=[s.value], keywords=[])
            new = ast.Assign(targets=[name], value=call)
        return [ast.copy_location(new, s)]
    return [s]


#: numpy unbuffered-scatter ufunc -> jax ``.at[idx].<method>`` name.
_SCATTER_AT_METHOD = {"add": "add", "subtract": "add", "multiply": "multiply", "maximum": "max", "minimum": "min"}


def _scatter_at_assign(call: ast.Call) -> Optional[ast.Assign]:
    """``np.<ufunc>.at(target, idx[, vals])`` (numpy's unbuffered scatter) ->
    the jax functional rebind ``target = target.at[idx].<method>(vals)``.

    Returns ``None`` when ``call`` is not a ``np.<ufunc>.at`` form. ``add.at``
    is the common scatter-accumulate (edge_laplacian's ``np.add.at(Lx, src,
    flux)``); ``subtract.at`` maps to ``.add(-vals)`` since jax has no
    ``.subtract``."""
    f = call.func
    if not (isinstance(f, ast.Attribute) and f.attr == "at" and isinstance(f.value, ast.Attribute)
            and isinstance(f.value.value, ast.Name) and f.value.value.id in ("np", "numpy")
            and f.value.attr in _SCATTER_AT_METHOD and len(call.args) >= 2):
        return None
    op = f.value.attr
    target, idx = call.args[0], call.args[1]
    vals: ast.expr = call.args[2] if len(call.args) > 2 else ast.Constant(value=1)
    if op == "subtract":
        vals = ast.UnaryOp(op=ast.USub(), operand=vals)
    at = ast.Subscript(value=ast.Attribute(value=target, attr="at", ctx=ast.Load()), slice=idx, ctx=ast.Load())
    rebind = ast.Call(func=ast.Attribute(value=at, attr=_SCATTER_AT_METHOD[op], ctx=ast.Load()),
                      args=[vals],
                      keywords=[])
    return ast.copy_location(ast.Assign(targets=[ast.Name(id=_base_name(target), ctx=ast.Store())], value=rebind), call)


def _base_name(t: ast.AST) -> str:
    while isinstance(t, ast.Subscript):
        t = t.value
    return t.id if isinstance(t, ast.Name) else "<expr>"


def _load(t: ast.AST) -> ast.AST:
    t2 = ast.fix_missing_locations(ast.parse(ast.unparse(t), mode="eval").body)
    return t2


def _is_full_slice(sl: ast.AST) -> bool:
    return isinstance(sl, ast.Slice) and sl.lower is None and sl.upper is None \
        and sl.step is None


def _broadcast_astype(arr: ast.AST, value: ast.expr) -> ast.Call:
    """``jnp.broadcast_to(value, arr.shape).astype(arr.dtype)`` -- the faithful
    lowering of numpy's in-place ``arr[:] = value`` store, which broadcasts the
    RHS to ``arr``'s shape and casts it to ``arr``'s dtype (dtype/shape inferred
    from the live array, never hardcoded)."""
    name = _base_name(arr)
    shape = ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr="shape", ctx=ast.Load())
    bcast = ast.Call(func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="broadcast_to", ctx=ast.Load()),
                     args=[value, shape],
                     keywords=[])
    dtype = ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr="dtype", ctx=ast.Load())
    return ast.Call(func=ast.Attribute(value=bcast, attr="astype", ctx=ast.Load()), args=[dtype], keywords=[])


# ---------------------------------------------------------------------------
# Statement / body emission
# ---------------------------------------------------------------------------
def _u(node: ast.AST) -> str:
    """Unparse with np->jnp already applied."""
    return ast.unparse(_np_to_jnp(ast.fix_missing_locations(node)))


def _emit_body(body: List[ast.stmt], live_out: Set[str], indent: str, defined: Set[str] = frozenset()) -> List[str]:
    """Emit a straight-line / looped statement list to JAX source lines.

    ``defined`` is the set of names already bound on entry to this body (params
    for the function body, the carry tuple for a loop body). It grows over the
    statements and is handed to each loop so the loop can verify its carry-tuple
    init references only bound names -- see :func:`_emit_for`."""
    lines: List[str] = []
    cur = set(defined)
    for k, s in enumerate(body):
        if k:
            # Every name STORED by a preceding statement (anywhere, incl. inside
            # a branch/loop) is treated as bound here -- an over-approximation
            # that never falsely flags a defined carry, while still catching a
            # loop that threads a temp NO preceding statement writes.
            cur |= _names_stored(ast.Module(body=[body[k - 1]], type_ignores=[]))
        if isinstance(s, (ast.For, ast.While, ast.If)):
            # A var assigned inside the construct and read by a *later*
            # statement here is live past it, so the loop/if must thread it
            # out: contour_integral's ``if ..: X = -X`` then ``P0 += X``, and
            # s318's argmax whose ``index``/``maxv`` are read only *after* the
            # loop. Fold the rest-of-body reads into live_out for all three
            # (previously done for ``if`` only -- a ``for``/``while`` that
            # produced an after-loop value silently dropped it from the carry).
            rest = _upward_exposed(body[k + 1:])
            if isinstance(s, ast.For):
                lines += _emit_for(s, live_out | rest, indent, cur)
            elif isinstance(s, ast.While):
                lines += _emit_while(s, live_out | rest, indent, cur)
            else:
                lines += _emit_if(s, live_out | rest, indent, cur)
        elif isinstance(s, ast.Return):
            lines.append(indent + _u(s))
        elif isinstance(s, (ast.Assign, ast.AugAssign)):
            for fs in _functionalize_stmt(s):
                lines.append(indent + _u(fs))
        elif isinstance(s, (ast.Import, ast.ImportFrom, ast.Pass, ast.Raise, ast.Assert)):
            continue  # input-validation guards never fire on oracle-valid inputs
        elif isinstance(s, ast.FunctionDef):
            # Nested helper def (velocity_tendencies' ``gat``). JAX is Python,
            # so emit it as a real nested function (np->jnp applied in the body);
            # it stays in scope for the calls that follow in this body.
            arglist = ", ".join(a.arg for a in s.args.args)
            lines.append(f"{indent}def {s.name}({arglist}):")
            inner = _emit_body(s.body, set(), indent + "    ", {a.arg for a in s.args.args})
            lines += inner if inner else [indent + "    pass"]
        elif isinstance(s, ast.Expr):
            # A docstring/constant is a no-op; a bare call like
            # ``np.multiply(Z, Z, Z)`` is an in-place op with effects we cannot
            # safely drop -> fall back rather than silently miscompile.
            if isinstance(s.value, ast.Constant):
                continue
            sc = (_scatter_at_assign(s.value) if isinstance(s.value, ast.Call) else None)
            if sc is not None:  # np.add.at(...) -> a = a.at[idx].add(...)
                lines.append(indent + _u(sc))
                continue
            raise EmitError("bare expression statement (possible in-place op)")
        else:
            raise EmitError(f"unsupported statement: {type(s).__name__}")
    return lines


_EMIT_STATIC: Set[str] = set()

#: module-level constant names carried into the emitted module (weather
#: stencils' ``BET_M``, ls3df's ``_CW``/``_C0``); populated per ``emit_jax``
#: call and read by :func:`_is_static_iterable` to decide whether a non-``range``
#: ``for`` iterates a compile-time-constant sequence and can be unrolled.
_MODULE_CONSTS: Set[str] = set()

#: module-level SCALAR constants mapped to their concrete Python value (a subset
#: of :data:`_MODULE_CONSTS` -- only the int/float/bool/complex ones). Populated
#: per ``emit_jax`` call and read by :func:`_fold_const_branches` to prune a
#: branch whose test is fully determined by such constants (cloudsc's
#: ``if yrecldp_nssopt == 0: .. elif == 1: ..`` with ``yrecldp_nssopt = 1``).
_MODULE_CONST_VALUES: dict = {}

#: FUNCTION-local names bound once to a constant-literal sequence (lulesh's
#: ``faces = [(0, 1, 2, 3), ...]``). Reset per :func:`_emit_function` call and
#: read by :func:`_is_static_iterable` so a ``for (a, b, d, e) in faces:`` over
#: such a name unrolls as a literal Python loop (the ints index array columns)
#: instead of being rejected as a non-``range`` iterable.
_LOCAL_CONSTS: Set[str] = set()

#: names that alias ``scipy.linalg.eigh`` in the current module (cegterg's
#: ``_sci_eigh``); populated per ``emit_jax`` call and read by ``_rewrite_eigh``.
_EIGH_ALIASES: Set[str] = set()


def _is_static_iterable(node: ast.AST) -> bool:
    """Is ``node`` a compile-time-constant iterable a jit trace can UNROLL --
    a literal tuple/list of constants (``for axis in (0, 1, 2)``), a module
    constant sequence (``_CW``), or an ``enumerate``/``zip``/``reversed`` over
    such (ls3df's ``for m, w in enumerate(_CW, start=1)``)? Every free name it
    reads must be a static jit arg or a module constant, so the emitted Python
    ``for`` unrolls at trace time rather than needing a traced iterator."""
    static = _EMIT_STATIC | _MODULE_CONSTS | _LOCAL_CONSTS
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_names_loaded(e) <= static for e in node.elts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
            node.func.id in ("enumerate", "zip", "reversed"):
        return all(_is_static_iterable(a) or _names_loaded(a) <= static for a in node.args)
    if isinstance(node, ast.Name):
        return node.id in _MODULE_CONSTS or node.id in _LOCAL_CONSTS
    return False


def _is_const_literal(node: ast.AST) -> bool:
    """A compile-time constant literal: a ``Constant``, a signed constant, or a
    ``list``/``tuple`` nesting of such (lulesh's ``[(0, 1, 2, 3), (4, 5, 6, 7),
    ...]`` face-index table)."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple)):
        return all(_is_const_literal(e) for e in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_const_literal(node.operand)
    return False


def _local_const_seq_names(fn: ast.FunctionDef) -> Set[str]:
    """Function-local names bound EXACTLY ONCE to a constant-literal list/tuple
    (lulesh's ``faces = [(0, 1, 2, 3), ...]``). Such a name is a concrete Python
    sequence in the emitted function, so a ``for ... in <name>:`` over it unrolls
    at trace time (see :func:`_is_static_iterable`)."""
    counts: dict = {}
    literal: Set[str] = set()
    for s in ast.walk(fn):
        if isinstance(s, ast.Assign):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    counts[t.id] = counts.get(t.id, 0) + 1
                    if len(s.targets) == 1 and isinstance(s.value,
                                                          (ast.List, ast.Tuple)) and _is_const_literal(s.value):
                        literal.add(t.id)
    return {n for n in literal if counts.get(n) == 1}


def _rewrite_eigh(fn: ast.FunctionDef) -> None:
    """Rewrite ``w, v = eigh(a[, b], subset_by_index=[lo, hi])`` (np.linalg /
    scipy.linalg / an imported alias) to the Cholesky-reduced form whose standard
    step is a native ``np.linalg.eigh(C)`` -- ``jnp.linalg.eigh`` handles the
    complex-Hermitian standard case, but jax has no generalized eigh, so the
    ``a x = w b x`` reduction runs on jnp.linalg.cholesky / inv / matmul (np->jnp
    happens downstream). In place."""
    from numpyto_common.numpy_desugar import _eigh_call_ab, _eigh_stmts

    class _R(ast.NodeTransformer):

        def __init__(self):
            self.ctr = 0

        def visit_Assign(self, node: ast.Assign):
            if len(node.targets) != 1:
                return node
            hit = _eigh_call_ab(node.value, _EIGH_ALIASES)
            if hit is None:
                return node
            a_node, b_node, kw = hit
            tgt = node.targets[0]
            if not (isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2
                    and all(isinstance(e, ast.Name) for e in tgt.elts)):
                return node
            w, v = tgt.elts[0].id, tgt.elts[1].id
            p = f"__eigh{self.ctr}"
            self.ctr += 1
            pre: List[str] = []

            def name_of(nd, tag):
                if isinstance(nd, ast.Name):
                    return nd.id
                pre.append(f"{p}_{tag} = np.ascontiguousarray({ast.unparse(nd)})")
                return f"{p}_{tag}"

            aname = name_of(a_node, "a")
            bname = name_of(b_node, "b") if b_node is not None else None
            s = kw.get("subset_by_index")
            if isinstance(s, (ast.List, ast.Tuple)) and len(s.elts) == 2:
                lo, hi = ast.unparse(s.elts[0]), f"({ast.unparse(s.elts[1])}) + 1"
            else:
                lo, hi = "None", "None"
            lines = pre + _eigh_stmts(w, v, aname, bname, lo, hi, p, native_std=True)
            return [ast.copy_location(st, node) for st in ast.parse("\n".join(lines)).body]

    _R().visit(fn)
    ast.fix_missing_locations(fn)


def _emit_if(node: ast.If, live_out: Set[str], indent: str, defined: Set[str] = frozenset()) -> List[str]:
    """Lower an ``if`` to ``jnp.where`` selects, or keep it as a real Python
    branch when the condition is static (concrete jit args only -- e.g.
    contour_integral's ``if NR == NM`` choosing ``inv`` vs ``solve``, whose
    branches have incompatible shapes and so cannot be ``where``-merged).

    * static condition           -> emitted ``if``/``else`` verbatim.
    * ``if c: return A`` ...      -> ``return jnp.where(c, A, B)``.
    * otherwise                  -> snapshot/restore/``where``-select per var.
    """
    if _names_loaded(node.test) <= _EMIT_STATIC or _is_identity_test(node.test):
        cond = _u(node.test)
        lines = [f"{indent}if {cond}:"]
        lines += _emit_body(node.body, live_out, indent + "    ", defined) or [f"{indent}    pass"]
        if node.orelse:
            lines.append(f"{indent}else:")
            lines += _emit_body(node.orelse, live_out, indent + "    ", defined) or [f"{indent}    pass"]
        return lines
    cond = _cond_str(node.test)
    # if/else that simply returns -> a single selected return
    if _is_return_only(node.body) and _is_return_only(node.orelse):
        a = _u(node.body[0].value)
        b = _u(node.orelse[0].value)
        return [f"{indent}return jnp.where({cond}, {a}, {b})"]

    if any(isinstance(s, ast.Return) for s in node.body + node.orelse):
        raise EmitError("if-branch mixes return with assignments")
    assigned = _names_stored(ast.Module(body=node.body + node.orelse, type_ignores=[]))
    # Only variables that escape the ``if`` (live after it) are snapshotted +
    # selected; branch-local temporaries (e.g. scattering's ``dHG``/``dHD``,
    # which feed the in-branch accumulation only) are emitted plainly.
    select = sorted(v for v in assigned if v in live_out)
    if not select:
        # No escaping effect to gate: emit the then-branch as-is. (An else with
        # no live-out effect is a no-op.)
        return _emit_body(node.body, live_out, indent, defined)
    # Snapshot/restore temps must be unique *per if node*, not per depth: an
    # ``if``/``elif`` chain flattens to two ``If`` nodes emitted at the SAME
    # indent (the elif lands in the outer node's ``orelse``, recursed at the
    # same level), so a depth-based tag collides -- the inner branch overwrites
    # ``_cond``/``_then`` and the outer ``jnp.where`` then selects the inner
    # branch twice, dropping the outer write (ext_peel_multi_back). Source
    # position is unique per node and stateless.
    tag = f"{node.lineno}_{node.col_offset}"
    pre = {v: f"_pre{tag}_{v}" for v in select}
    then = {v: f"_then{tag}_{v}" for v in select}
    # A snapshot ``_pre = v`` (and its restore before the else) is only needed
    # when the INCOMING value of ``v`` is observable: either a branch reads ``v``
    # before it writes it (so the incoming value feeds that read), or a branch
    # does NOT assign ``v`` (so its side of the ``where`` keeps the incoming
    # value). When ``v`` is assigned on BOTH branches and read by neither before
    # it is written, the merge is fully determined by the two branch values and
    # the incoming value is never used -- snapshotting it would be dead, and
    # (worse) when ``v`` is FIRST bound inside the branches, ``_pre = v`` reads
    # an unbound local and raises ``UnboundLocalError`` at trace time
    # (contour_integral's ``X = inv(Tz)`` / ``X = solve(Tz, Y)`` in an
    # ``if NR == NM`` with no prior ``X``). Emit the snapshot/restore only where
    # the incoming value is genuinely live.
    body_stored = _names_stored(ast.Module(body=node.body, type_ignores=[]))
    orelse_stored = _names_stored(ast.Module(body=node.orelse, type_ignores=[]))

    def _needs_pre(v: str) -> bool:
        if not (v in body_stored and v in orelse_stored):
            return True
        return _reads_before_write(node.body, v) or _reads_before_write(node.orelse, v)

    needs = {v: _needs_pre(v) for v in select}
    # Capture the condition *before* the branches run -- they may overwrite the
    # very variables it tests (crc16's ``if crc&1 ^ ...: crc = crc>>1 ^ poly``).
    lines = [f"{indent}_cond{tag} = ({cond})"]
    lines += [f"{indent}{pre[v]} = {v}" for v in select if needs[v]]
    lines += _emit_body(node.body, live_out, indent, defined)
    lines += [f"{indent}{then[v]} = {v}" for v in select]
    lines += [f"{indent}{v} = {pre[v]}" for v in select if needs[v]]
    if node.orelse:
        lines += _emit_body(node.orelse, live_out, indent, defined | body_stored)
    lines += [f"{indent}{v} = jnp.where(_cond{tag}, {then[v]}, {v})" for v in select]
    return lines


def _reads_before_write(stmts: List[ast.stmt], v: str) -> bool:
    """Does ``v`` appear as an input (RHS / container / index read) in ``stmts``
    before any statement writes it? A nested compound statement is treated
    conservatively -- any load of ``v`` inside it counts as a read."""
    written: Set[str] = set()
    for s in stmts:
        if v in _stmt_rhs_loads(s) and v not in written:
            return True
        written |= _names_stored(ast.Module(body=[s], type_ignores=[]))
    return False


def _is_return_only(body: List[ast.stmt]) -> bool:
    return len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None


def _is_identity_test(test: ast.AST) -> bool:
    """A pure identity presence check (``x is None`` / ``x is not None``, e.g.
    fv3_dycore's optional ``if del6_v is not None:`` damping). Its result is a
    trace-time-concrete fact (an argument is None or it is not), never a value
    on a traced array, so the ``if`` is a REAL Python branch -- ``jnp.where`` can
    neither test nor select on ``None`` and would (worse) execute both arms."""
    return isinstance(test, ast.Compare) and bool(test.ops) \
        and all(isinstance(op, (ast.Is, ast.IsNot)) for op in test.ops)


def _split_on_break(body: List[ast.stmt]):
    """Split the loop body around the ``if`` whose branch ENDS in ``break``.

    Returns ``(before, cond, on_break, after)``:

    * ``before``   -- stmts ahead of the guard; run every iteration.
    * ``cond``     -- the break test (``None`` if no such guard is found).
    * ``on_break`` -- stmts inside the guard *before* the break -- the *capture*
      that runs on the converging iteration (s332's ``index = i; value = a[i]``;
      empty for the bare convergence guard ``if rsnew < tol: break``).
    * ``after``    -- stmts past the guard; run only when NOT converged.
    """
    for k, s in enumerate(body):
        if isinstance(s, ast.If) and s.body and isinstance(s.body[-1], ast.Break) and not s.orelse:
            return body[:k], s.test, s.body[:-1], body[k + 1:]
    return body, None, [], []


def _tup(names: List[str]) -> str:
    """A Python tuple literal that is valid for 0/1/n elements."""
    return "()" if not names else "(" + ", ".join(names) + ",)"


def _parse_range(rng: ast.Call):
    """``range`` -> ``(lo_expr, hi_expr, backward, stride)``.

    ``backward`` is True for a ``-1`` step (``range(a, b, -1)`` iterates
    a, a-1, ..., b+1). ``stride`` is the *positive* step as a source string
    (``"1"`` for the common unit step, ``"W"`` / ``"7"`` for a strided tile
    loop ``range(1, N - 1, W)``); a strided forward range drives a forward
    counter and recovers ``i = lo + _k * stride`` (see :func:`_emit_for`)."""
    args = rng.args
    if len(args) == 1:
        return "0", _u(args[0]), False, "1"
    if len(args) == 2:
        return _u(args[0]), _u(args[1]), False, "1"
    if len(args) == 3:
        step = args[2]
        if isinstance(step, ast.Constant) and step.value == 1:
            return _u(args[0]), _u(args[1]), False, "1"
        if isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub) \
                and isinstance(step.operand, ast.Constant) \
                and step.operand.value == 1:
            return _u(args[0]), _u(args[1]), True, "1"
        # A negative *constant* step other than -1 is an unsupported backward
        # stride; anything else (a positive constant or a symbol like ``W``) is
        # taken as a forward stride.
        if isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub):
            raise EmitError("only -1 backward range() is supported")
        if isinstance(step, ast.Constant) and not (isinstance(step.value, int) and step.value > 0):
            raise EmitError("non-positive range() step is not supported")
        return _u(args[0]), _u(args[1]), False, _u(step)
    raise EmitError("malformed range()")


def _emit_for(node: ast.For, live_out: Set[str], indent: str, defined: Set[str] = frozenset()) -> List[str]:
    kind = _classify_for(node)
    i = node.target.id if isinstance(node.target, ast.Name) else "_i"
    rng = node.iter
    if not (isinstance(rng, ast.Call) and isinstance(rng.func, ast.Name) and rng.func.id == "range"):
        # A non-``range`` iterable that is a compile-time-constant sequence
        # (ls3df's ``for axis in (0, 1, 2)`` / ``for m, w in enumerate(_CW,
        # start=1)``) is emitted as a literal Python ``for`` the tracer unrolls
        # -- a tiny static trip count, so any loop-carried rebind (``acc``) just
        # threads as a normal Python value across the unrolled iterations. A
        # non-constant iterable can't be traced and honestly raises.
        # A literal tuple/list of Names (lulesh's ``for nk in (n0, n1, n2, n3):``
        # over face-corner indices) ALWAYS has a static trip count and unrolls at
        # trace time -- the element values need not be compile-time constants,
        # only the LENGTH does. Emit it literally too, UNLESS a loop-target name
        # feeds a shape (where the value must be concrete): then keep refusing so
        # the eager path (which runs the sequence directly) takes over.
        literal_seq = isinstance(rng, (ast.Tuple, ast.List)) and \
            all(isinstance(e, (ast.Name, ast.Constant)) or _is_const_literal(e) for e in rng.elts)
        if _is_static_iterable(rng) or literal_seq:
            bound = {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
            if literal_seq and not _is_static_iterable(rng) and any(_index_in_shape(node, b) for b in bound):
                raise EmitError("literal-sequence loop index feeds a shape (needs concrete unroll)")
            lines = [f"{indent}for {_u(node.target)} in {_u(rng)}:"]
            lines += _emit_body(node.body, live_out, indent + "    ", defined | bound) or [f"{indent}    pass"]
            return lines
        raise EmitError("only `for i in range(...)` is supported")
    lo, hi, backward, stride = _parse_range(rng)

    # If the index feeds a *shape* (stockham_fft's ``reshape(y, (R**i, …))``),
    # it must be concrete -- emit a real Python loop that the tracer unrolls.
    # Sound only when the trip count is static (jit args / constants) AND the
    # loop is not a time-stepping loop (those must stay rolled -- unrolling a
    # long timestep trip count blows up the trace; the parallelism policy makes
    # this a guarantee, not a heuristic).
    if (_index_in_shape(node, i) and _range_args_static(rng) and not backward and stride == "1"
            and not _is_timestep_loop(node)):
        lines = [f"{indent}for {i} in range({lo}, {hi}):"]
        lines += _emit_body(node.body, live_out, indent + "    ", defined | {i}) or [f"{indent}    pass"]
        return lines

    if kind == LoopKind.VECTORIZE:
        # a[i] = f(b[i], ...)  ->  a = f(b, ...)  (drop the [i] indexing)
        out = []
        for s in node.body:
            t = s.targets[0]
            arr = t.value.id
            rhs = _devectorize_index(s.value, i)
            # A RHS that reads `i` (only ever inside ``x[i]`` subscripts here)
            # devectorises to a full-array expression of the target's shape, so
            # ``a = rhs`` is shape-correct. A *loop-invariant* RHS (``a[i] = a0``,
            # ``a[i] = 0.0``) has no `i`, so ``a = rhs`` would collapse `a` to
            # that scalar/row -- broadcast-fill it back to `a`'s shape (and dtype)
            # instead. (s293's whole-array fill.)
            if i in _names_loaded(s.value):
                out.append(f"{indent}{arr} = {_u(rhs)}")
            else:
                out.append(f"{indent}{arr} = jnp.full_like({arr}, {_u(rhs)})")
        return out

    carried = _carried_vars(node.body, live_out)
    if not carried:
        raise EmitError("loop carries no observable state")
    # The carry-tuple init references each carried var by name, so every one must
    # be bound BEFORE the loop. A var pulled into the carry that no preceding
    # statement defines (cloudsc's ``zqe``: a per-iteration block-local temp only
    # LOOKS carried because an ``if nssopt == 0: .. elif ..`` chain with no
    # ``else`` reads as a conditional write) would make the init reference an
    # unbound local -- an honest ``UnboundLocalError`` at trace time. Fall back
    # to the eager emit rather than emit a module that cannot run.
    missing = [v for v in carried if v not in defined]
    if missing:
        raise EmitError(f"loop carry not defined before the loop: {', '.join(missing)}")
    inner = indent + "    "
    st = _tup(carried)
    if kind == LoopKind.FORI:
        # A unit forward step drives the index directly. A backward ``-1`` step
        # or a forward stride ``s > 1`` (tiled ``range(1, N-1, W)``) instead
        # drives a forward counter ``_k`` over ``[0, trip)`` and recovers the
        # real index: backward ``i = lo - _k`` (trip ``lo - hi``); strided
        # ``i = lo + _k*s`` (trip ``ceil((hi - lo) / s)``).
        if backward:
            ctr, lo2, hi2 = "_k", "0", f"({lo}) - ({hi})"
            recover = f"{inner}{i} = ({lo}) - _k"
        elif stride != "1":
            ctr, lo2, hi2 = "_k", "0", f"(({hi}) - ({lo}) + ({stride}) - 1) // ({stride})"
            recover = f"{inner}{i} = ({lo}) + _k * ({stride})"
        else:
            ctr, lo2, hi2, recover = i, lo, hi, None
        body_inner = _emit_body(node.body, set(carried), inner, set(carried) | {i})
        lines = [f"{indent}def _body({ctr}, _c):", f"{inner}{st} = _c"]
        if recover:
            lines.append(recover)
        lines += body_inner
        lines += [f"{inner}return {st}", f"{indent}{st} = lax.fori_loop({lo2}, {hi2}, _body, {st})"]
        return lines
    # WHILE: range + break -> while_loop carrying the index + a done flag.
    if backward or stride != "1":
        raise EmitError("backward/strided range with break is not supported")
    return _emit_while_break(node, carried, lo, hi, i, indent)


def _expand_parallel_assigns(stmts: List[ast.stmt]) -> List[ast.stmt]:
    """Split a parallel Name-tuple assign ``a, b = e0, e1`` into ``__pa = (e0,
    e1); a = __pa[0]; b = __pa[1]`` so each target becomes a single Name rebind
    (ls3df's Lanczos ``v_prev, v = v, w / beta``). The temp snapshots the whole
    RHS first, preserving numpy's simultaneous-assignment semantics under the
    break-guard's per-var ``jnp.where`` freeze."""
    out: List[ast.stmt] = []
    for s in stmts:
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], (ast.Tuple, ast.List)) \
                and all(isinstance(e, ast.Name) for e in s.targets[0].elts):
            _TUPLE_CTR[0] += 1
            tmp = f"__pa{_TUPLE_CTR[0]}"
            out.append(ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=s.value))
            for k, e in enumerate(s.targets[0].elts):
                item = ast.Subscript(value=ast.Name(id=tmp, ctx=ast.Load()),
                                     slice=ast.Constant(value=k),
                                     ctx=ast.Load())
                out.append(ast.Assign(targets=[ast.Name(id=e.id, ctx=ast.Store())], value=item))
        else:
            out.append(s)
    return [ast.fix_missing_locations(x) for x in out]


def _emit_while_break(node, carried, lo, hi, i, indent):
    before, cond, on_break, after = _split_on_break(node.body)
    if cond is None:
        raise EmitError("break loop without an `if cond: ... break` guard")
    on_break = _expand_parallel_assigns(on_break)
    after = _expand_parallel_assigns(after)
    full = [i] + carried + ["_done"]
    inner = indent + "    "
    st = _tup(full)
    lines = [
        f"{indent}def _cond(_c):", f"{inner}{st} = _c", f"{inner}return ({i} < {hi}) & jnp.logical_not(_done)",
        f"{indent}def _body(_c):", f"{inner}{st} = _c"
    ]
    lines += _emit_body(before, set(carried), inner, set(carried) | {i})
    lines.append(f"{inner}_conv = ({_cond_str(cond)})")
    cset = set(carried)

    def _frozen(stmts, when_conv):
        # Emit each rebind, freezing a *carried* var with ``jnp.where`` so it is
        # only updated on the intended branch; a *local temp* (minres's ``beta``)
        # is emitted plainly. ``when_conv`` picks the polarity: the *capture*
        # inside the guard takes the new value WHEN converged
        # ``where(_conv, new, old)``; statements *after* the guard keep the old
        # value when converged ``where(_conv, old, new)``.
        for s in stmts:
            for fs in _functionalize_stmt(s):
                if not (isinstance(fs, ast.Assign) and len(fs.targets) == 1 and isinstance(fs.targets[0], ast.Name)):
                    raise EmitError("break-guard statement is not a simple rebind")
                tgt = fs.targets[0].id
                if tgt not in cset:
                    lines.append(f"{inner}{_u(fs)}")
                elif when_conv:
                    lines.append(f"{inner}{tgt} = jnp.where(_conv, {_u(fs.value)}, {tgt})")
                else:
                    lines.append(f"{inner}{tgt} = jnp.where(_conv, {tgt}, {_u(fs.value)})")

    # The capture (``index = i``) commits on the converging iteration; the
    # post-guard update (cg/minres's next-iterate maths) is skipped on it.
    _frozen(on_break, when_conv=True)
    _frozen(after, when_conv=False)
    ret = "(" + ", ".join([f"{i} + 1"] + carried + ["_conv | _done"]) + ",)"
    init = "(" + ", ".join([lo] + carried + ["jnp.bool_(False)"]) + ",)"
    lines += [f"{inner}return {ret}", f"{indent}{st} = lax.while_loop(_cond, _body, {init})"]
    return lines


def _emit_while(node: ast.While, live_out: Set[str], indent: str, defined: Set[str] = frozenset()) -> List[str]:
    carried = _carried_vars(node.body, live_out, _names_loaded(node.test))
    if not carried:
        raise EmitError("while-loop carries no observable state")
    missing = [v for v in carried if v not in defined]
    if missing:
        raise EmitError(f"while carry not defined before the loop: {', '.join(missing)}")
    inner = indent + "    "
    st = _tup(carried)
    lines = [
        f"{indent}def _cond(_c):", f"{inner}{st} = _c", f"{inner}return ({_cond_str(node.test)})",
        f"{indent}def _body(_c):", f"{inner}{st} = _c"
    ]
    lines += _emit_body(node.body, set(carried), inner, set(carried))
    lines += [f"{inner}return {st}", f"{indent}{st} = lax.while_loop(_cond, _body, {st})"]
    return lines


def _devectorize_index(node: ast.AST, i: str) -> ast.AST:
    """Drop ``[i]`` subscripts so an independent elementwise loop body becomes
    a whole-array expression."""

    class _R(ast.NodeTransformer):

        def visit_Subscript(self, n):
            self.generic_visit(n)
            if _is_index_i(n.slice, i):
                return n.value
            return n

    return _R().visit(ast.fix_missing_locations(ast.parse(ast.unparse(node), mode="eval"))).body


# ---------------------------------------------------------------------------
# Top-level: emit a full kernel module
# ---------------------------------------------------------------------------
def emit_jax(numpy_src: str, func_name: str, jit: bool = False) -> str:
    """Translate the ``func_name`` function in ``numpy_src`` to JAX source.

    By default the kernel is emitted in **eager** mode: ``np.`` -> ``jnp.`` with
    in-place mutation made functional, but Python control flow (``for``/``while``
    /``if``/``break``, arbitrary ``range`` steps, data-dependent slices) is kept
    verbatim and the function is *not* ``jax.jit``-decorated. Eager JAX executes
    concrete arrays op-by-op, so it supports dynamic shapes / boolean indexing /
    breaks that a traced ``jit`` kernel cannot -- this is the most faithful
    1:1 translation and covers the widest set of kernels (notably the strided /
    data-dependent foundation loops).

    With ``jit=True`` the loop-lowering classifier kicks in instead (vectorise /
    ``fori_loop`` / ``while_loop`` + the masking transforms) and the kernel is
    ``@jax.jit``-decorated -- the compiled, hand-``*_jax.py``-style form.

    Helper functions the kernel calls (e.g. ``relu``/``softmax`` for ``mlp``)
    are emitted as plain module-level functions ahead of the kernel."""
    tree = ast.parse(numpy_src)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name), None)
    if fn is None:
        raise EmitError(f"function {func_name!r} not found")
    from numpyto_common.numpy_desugar import _eigh_alias_names
    _EIGH_ALIASES.clear()
    _EIGH_ALIASES.update(_eigh_alias_names(tree))
    _MODULE_CONSTS.clear()
    _MODULE_CONSTS.update(_module_constant_names(tree, func_name))
    _MODULE_CONST_VALUES.clear()
    _MODULE_CONST_VALUES.update(_module_const_values(tree, func_name))
    # Substitute whole-array ``local = param`` aliases with the param itself
    # (ICON velocity_tendencies aliases ~40 params: ``vt = p_diag_vt``). In
    # functional jax an in-place write through the alias rebinds the LOCAL, so
    # the param's output would never be returned; folding the alias makes the
    # mutation land on the param so it is recognised as an in-place output --
    # mirroring the C/Fortran frontend's _SubstituteParamAliases.
    from numpyto_common.frontend import _SubstituteParamAliases
    _alias = _SubstituteParamAliases([a.arg for a in fn.args.args])
    _alias.collect(fn)
    _alias.visit(fn)
    ast.fix_missing_locations(fn)
    # Emit ONLY the helpers REACHABLE (transitively called) from the target --
    # not every module-level function. A module may co-locate sibling kernels the
    # target never calls (vexx's full-config ``vexx_all_paths`` + its in-place US/
    # PAW helpers); emitting those would choke on constructs jax can't express
    # even though they are irrelevant to ``func_name``.
    _defined = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def _called_names(node: ast.AST) -> set:
        return {
            c.func.id
            for c in ast.walk(node)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in _defined
        }

    _reachable: set = set()
    _frontier = _called_names(fn)
    while _frontier:
        nm = _frontier.pop()
        if nm in _reachable or nm in (func_name, "initialize"):
            continue
        _reachable.add(nm)
        _frontier |= _called_names(_defined[nm])
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _reachable]
    # Fold each helper's OWN ``local = param`` aliases too (fv3_dycore's stencil
    # helpers open with ``f = field`` then mutate ``f[i, j] = ..``): without this
    # the in-place writes land on the local alias, the param is not seen as
    # mutated, and its bare call site never functionalises to ``field = helper(
    # field, ..)`` -- leaving a bare-expression statement the emitter rejects.
    for h in helpers:
        h_alias = _SubstituteParamAliases([a.arg for a in h.args.args])
        h_alias.collect(h)
        h_alias.visit(h)
        ast.fix_missing_locations(h)
    # Transitive in-place-mutation map over the kernel + every reachable helper,
    # computed on the alias-folded source so a param mutated only through a bare
    # sub-helper call is still recognised as an output at both the call site
    # (functionalised) and the function's own return.
    funcs = {func_name: fn}
    funcs.update({h.name: h for h in helpers})
    mut_map = _mutation_maps(funcs)
    helper_mut = _helper_mutation_map(helpers, mut_map)
    # Kernel static_argnames, computed transitively so a control param used only
    # inside a called helper's branch (fv3_dycore's ``hord``) is still concrete.
    kernel_static = _transitive_static(func_name, funcs) if jit else []
    # Each HELPER's concrete (trace-time-known) params, flowed FORWARD from the
    # kernel's static args -- so a helper branch on a genuinely static value
    # (fv3_dycore's ``if mord == 5``) stays a Python branch, while one on traced
    # data (nussinov's ``match(seq[i], seq[j])``) lowers to ``jnp.where``.
    concrete = _concrete_params(funcs, func_name, kernel_static) if jit else {}

    head = [
        "import jax",
        # numpy's default float/int are 64-bit; jax silently narrows to 32-bit
        # unless x64 is enabled. Turn it on at the TOP of the module (before any
        # jnp array is built) so ``jnp.float64``/``int64`` are honoured and the
        # kernel matches the numpy reference's precision.
        "jax.config.update('jax_enable_x64', True)",
        "import jax.numpy as jnp",
        "from jax import lax",
        "from functools import partial",
    ]
    # Carry over the kernel module's own imports (minus ``numpy`` -- ``jnp``
    # replaces it) so e.g. a TSVC kernel's ``from math import sin, sqrt`` and
    # its bare ``sin(b[i])`` calls resolve in the emitted module.
    head += _extra_imports(tree)
    head += ["", ""]
    consts = _module_constants(tree, func_name)
    if consts:
        head += consts + [""]
    eager = not jit
    for h in helpers:
        head += _emit_function(h,
                               decorate=None,
                               helper_mut=helper_mut,
                               eager=eager,
                               mutated=mut_map.get(h.name),
                               static=(sorted(concrete.get(h.name, ())) if jit else None)) + ["", ""]
    deco = _kernel_decorator(fn, kernel_static) if jit else None
    return "\n".join(head + _emit_function(fn, decorate=deco, helper_mut=helper_mut, eager=eager,
                                           mutated=mut_map.get(func_name), static=kernel_static)) + "\n"


def _helper_mutation_map(helpers: List[ast.FunctionDef], mut_map: Optional[dict] = None) -> dict:
    """Map ``helper_name -> emitted-return rebind slots`` for every helper that
    mutates a param in place (jax arrays are immutable, so the mutation only
    survives if the call site captures it back).

    The emitter turns such a helper's return into ``own_return_elts + [mutated
    params not already returned by name]`` (``_augment_returns``; a no-return
    helper returns just the mutated params). Each returned slot is either
    ``("mut", pos)`` -- that value is the new value of the arg passed at param
    ``pos`` -- or ``("val",)`` -- a genuine return value the call's LHS captures.
    ``_rewrite_inplace_helper_calls`` uses the slots to rebind every call site:

    * ``build_up_b(b, ...)`` (mutates ``b``, returns None) -> ``b = build_up_b(b, ...)``;
    * ``_addusxx_r(rhoc, ...)`` (mutates AND ``return rhoc``) -> ``rhoc = _addusxx_r(rhoc, ...)``;
    * ``fac = _g2_convolution_all(cf, cd, ...)`` (returns a column, mutates the
      ``cf``/``cd`` caches) -> ``fac, cf, cd = _g2_convolution_all(cf, cd, ...)``.
    """
    out = {}
    for h in helpers:
        params = [a.arg for a in h.args.args]
        muts = mut_map[h.name] if mut_map is not None else _mutated_params(h, params)
        if not muts:
            continue
        slots = _emitted_return_slots(h, params, muts)
        if slots is not None:
            out[h.name] = slots
    return out


def _emitted_return_slots(h: ast.FunctionDef, params: List[str], muts: List[str]):
    """Ordered rebind slots for a mutating helper's EMITTED return, or None when
    it can't be rewritten safely (a return the augmentation can't reach, or
    inconsistent return points)."""
    mpos = {m: params.index(m) for m in muts}
    rets = _own_returns(h)
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
        slots = [("mut", mpos[e.id]) if (isinstance(e, ast.Name) and e.id in mpos) else ("val", ) for e in elts]
        slots += [("mut", mpos[m]) for m in muts if m not in present]
        slot_sets.add(tuple(slots))
    # Every return point must rebind identically for a single call-site rewrite.
    return list(next(iter(slot_sets))) if len(slot_sets) == 1 else None


def _rewrite_inplace_helper_calls(fn: ast.FunctionDef, helper_mut: dict) -> None:
    """Capture a mutating helper's in-place effect at every call site, per its
    emitted-return slots (see ``_helper_mutation_map``). A ``("mut", pos)`` slot
    rebinds the arg passed at ``pos`` (a plain Name, or a subscript/attribute view
    like ``deexx[:, ii]`` that then flows through ``_functionalize_stmt`` into
    ``deexx = deexx.at[:, ii].set(...)``); a ``("val",)`` slot is taken from the
    call's LHS."""

    def _targets(call: ast.Call, lhs_elts: List[ast.AST]):
        slots = helper_mut[call.func.id]
        if sum(1 for s in slots if s[0] == "val") != len(lhs_elts):
            return None  # LHS arity must match the genuine return values
        for kind, *rest in slots:
            if kind == "mut" and (rest[0] >= len(call.args) or not _is_assignable(call.args[rest[0]])):
                return None  # mutated param not passed positionally / not assignable
        vi = iter(lhs_elts)
        return [_as_store(next(vi) if s[0] == "val" else call.args[s[1]]) for s in slots]

    def _rebind(call, lhs_elts, node):
        tgts = _targets(call, lhs_elts)
        if tgts is None:
            return node
        tgt = tgts[0] if len(tgts) == 1 else ast.Tuple(elts=tgts, ctx=ast.Store())
        return ast.copy_location(ast.Assign(targets=[tgt], value=call), node)

    def _is_mut_call(c):
        return isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in helper_mut

    class _T(ast.NodeTransformer):

        def visit_Expr(self, node):  # bare call: no captured return values
            return _rebind(node.value, [], node) if _is_mut_call(node.value) else node

        def visit_Assign(self, node):  # value-captured call: LHS supplies the ``val`` slots
            if _is_mut_call(node.value) and len(node.targets) == 1:
                lhs = node.targets[0]
                return _rebind(node.value, lhs.elts if isinstance(lhs, ast.Tuple) else [lhs], node)
            return node

    _T().visit(fn)
    ast.fix_missing_locations(fn)


def _is_assignable(node: ast.AST) -> bool:
    """An expression that can be an assignment target (so a mutating helper's
    in-place arg can be rebound from its return)."""
    return isinstance(node, (ast.Name, ast.Subscript, ast.Attribute))


def _as_store(node: ast.AST) -> ast.AST:
    """A copy of an assignable expression with Store context -- the original stays
    a Load arg inside the call, so it must not be mutated in place."""
    n = copy.deepcopy(node)
    n.ctx = ast.Store()
    return n


def _kernel_decorator(fn: ast.FunctionDef, static: Optional[List[str]] = None) -> str:
    params = [a.arg for a in fn.args.args]
    if static is None:
        static = _static_params(fn, params)
    if static:
        return "@partial(jax.jit, static_argnames=(" + ", ".join(f"{s!r}" for s in static) + ",))"
    return "@jax.jit"


def _emit_function(fn: ast.FunctionDef,
                   decorate: Optional[str],
                   helper_mut: Optional[dict] = None,
                   eager: bool = False,
                   mutated: Optional[List[str]] = None,
                   static: Optional[List[str]] = None) -> List[str]:
    """Translate one function (kernel or helper) to JAX source lines."""
    global _JIT_MODE
    if helper_mut:
        _rewrite_inplace_helper_calls(fn, helper_mut)
    if eager:
        return _emit_function_eager(fn, decorate, mutated)
    _JIT_MODE = True
    _rewrite_eigh(fn)
    _LOCAL_CONSTS.clear()
    _LOCAL_CONSTS.update(_local_const_seq_names(fn))
    _desugar_foreach(fn)
    _fold_const_branches(fn)
    params = [a.arg for a in fn.args.args]
    # Static args are concrete at trace time -- an ``if``/ternary testing only
    # them stays a real branch, and a loop whose index feeds a shape is unrolled.
    # Set before the slice transforms so they treat an unrolled index as
    # concrete (its ``:R**i`` slices are static, not data-dependent). The kernel
    # uses the TRANSITIVE static set (matching its decorator's static_argnames);
    # a HELPER falls back to its own per-function static params -- it too gets
    # concrete values for those args from the (jitted) caller, so its
    # ``ord_inner = 8 if hord == 10 else hord`` (fv3_dycore) must stay a Python
    # ternary, not degrade to ``jnp.where`` on a 0-d array, and its ``if mord ==
    # 5`` PPM-order branch must pick ONE arm rather than blend both.
    _EMIT_STATIC.clear()
    _EMIT_STATIC.update(static if static is not None else _static_params(fn, params))

    _expand_tuple_targets(fn)
    _expand_chained_assigns(fn)
    _boolean_mask_transform(fn)
    _rewrite_flip_prefix(fn)
    _mask_reduction_slices(fn)
    _mask_slice_reads(fn)
    _mask_dynamic_writes(fn)
    _dynamic_window_slices(fn)
    _reject_dynamic_slices(fn)

    returns = _own_returns(fn)
    if mutated is None:
        mutated = _mutated_params(fn, params)
    live_out: Set[str] = set(params)

    if returns and mutated:
        _augment_returns(fn, mutated)
    body_lines = _emit_body(fn.body, live_out, "    ", set(params) | _MODULE_CONSTS)
    if not returns:
        # numpy mutated in place + returned None -> return the mutated outputs.
        body_lines.append("    return " + ", ".join(mutated))

    head = [decorate] if decorate else []
    head.append(f"def {fn.name}({_signature(fn)}):")
    return head + body_lines


def _emit_function_eager(fn: ast.FunctionDef, decorate: Optional[str],
                         mutated: Optional[List[str]] = None) -> List[str]:
    """Emit one function in **eager** mode: Python control flow verbatim, only
    in-place mutation made functional (jax arrays are immutable even eagerly).
    No loop classification, no masking -- eager JAX runs dynamic slices, boolean
    indexing and data-dependent breaks directly on concrete arrays."""
    global _JIT_MODE
    _JIT_MODE = False
    # Multi-target / tuple-of-subscript assigns still need splitting so each
    # subscript target functionalises to its own ``.at[..].set(..)``.
    _rewrite_eigh(fn)
    _expand_tuple_targets(fn)
    _expand_chained_assigns(fn)
    params = [a.arg for a in fn.args.args]
    returns = _own_returns(fn)
    if mutated is None:
        mutated = _mutated_params(fn, params)
    if returns and mutated:
        _augment_returns(fn, mutated)
    body_lines = _emit_eager_body(fn.body, "    ")
    if not returns:
        body_lines.append("    return " + ", ".join(mutated))
    head = [decorate] if decorate else []
    head.append(f"def {fn.name}({_signature(fn)}):")
    return head + body_lines


def _emit_eager_body(body: List[ast.stmt], indent: str) -> List[str]:
    """Recursively emit a statement list with control flow kept literal."""
    inner = indent + "    "
    lines: List[str] = []
    for s in body:
        if isinstance(s, ast.For):
            if s.orelse:
                raise EmitError("for-else not supported")
            lines.append(f"{indent}for {_u(s.target)} in {_u(s.iter)}:")
            lines += _emit_eager_body(s.body, inner) or [inner + "pass"]
        elif isinstance(s, ast.While):
            if s.orelse:
                raise EmitError("while-else not supported")
            lines.append(f"{indent}while {_u(s.test)}:")
            lines += _emit_eager_body(s.body, inner) or [inner + "pass"]
        elif isinstance(s, ast.If):
            lines.append(f"{indent}if {_u(s.test)}:")
            lines += _emit_eager_body(s.body, inner) or [inner + "pass"]
            if s.orelse:
                lines.append(f"{indent}else:")
                lines += _emit_eager_body(s.orelse, inner) or [inner + "pass"]
        elif isinstance(s, (ast.Return, ast.Break, ast.Continue, ast.Pass)):
            lines.append(indent + _u(s))
        elif isinstance(s, (ast.Assign, ast.AugAssign)):
            for fs in _functionalize_stmt(s):
                lines.append(indent + _u(fs))
        elif isinstance(s, ast.Expr):
            if isinstance(s.value, ast.Constant):  # docstring / bare constant
                continue
            fs = _functionalize_bare_expr(s.value)
            if fs is None:
                raise EmitError("bare expression statement (possible in-place op)")
            lines.append(indent + _u(fs))
        elif isinstance(s, (ast.Import, ast.ImportFrom, ast.Raise, ast.Assert)):
            continue  # input-validation guards never fire on oracle-valid inputs
        elif isinstance(s, ast.FunctionDef):
            # Nested helper def (velocity_tendencies' ``gat``) -- emit as a
            # nested Python function (np->jnp applied), in scope for later calls.
            arglist = ", ".join(a.arg for a in s.args.args)
            lines.append(f"{indent}def {s.name}({arglist}):")
            lines += _emit_eager_body(s.body, inner) or [inner + "pass"]
        else:
            raise EmitError(f"unsupported statement: {type(s).__name__}")
    return lines


def _functionalize_bare_expr(call: ast.AST) -> Optional[ast.Assign]:
    """A bare ``np.<ufunc>(..., out)`` statement has effect only through its out
    array -- rebind it: ``np.multiply(Z, Z, Z)`` -> ``Z = np.multiply(Z, Z)``,
    ``np.add(Z, C, out=Z)`` -> ``Z = np.add(Z, C)``. Returns None if there is no
    capturable out target (so the caller can fall back rather than drop effects).
    """
    if not isinstance(call, ast.Call):
        return None
    sc = _scatter_at_assign(call)  # np.add.at(a, idx, v) -> a = a.at[idx].add(v)
    if sc is not None:
        return sc
    for kw in call.keywords:  # explicit out= keyword wins
        if kw.arg == "out" and isinstance(kw.value, ast.Name):
            new = ast.Call(func=call.func, args=call.args, keywords=[k for k in call.keywords if k.arg != "out"])
            return ast.Assign(targets=[ast.Name(id=kw.value.id, ctx=ast.Store())], value=new)
    # positional out: an ``np``/``jnp`` ufunc whose last positional arg is a Name
    # (a bare ufunc statement has no other observable effect).
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and \
            call.func.value.id in ("np", "jnp") and call.args and isinstance(call.args[-1], ast.Name):
        out = call.args[-1]
        new = ast.Call(func=call.func, args=call.args[:-1], keywords=call.keywords)
        return ast.Assign(targets=[ast.Name(id=out.id, ctx=ast.Store())], value=new)
    return None


def _extra_imports(tree: ast.Module) -> List[str]:
    """The module's own import statements, minus ``numpy`` (``jnp`` stands in)."""
    out: List[str] = []
    for s in tree.body:
        if isinstance(s, ast.Import):
            names = [a for a in s.names if a.name.split(".")[0] != "numpy"]
            if names:
                out.append(ast.unparse(ast.Import(names=names)))
        elif isinstance(s, ast.ImportFrom):
            if (s.module or "").split(".")[0] == "numpy":
                continue
            out.append(ast.unparse(s))
    return out


def _module_constants(tree: ast.Module, func_name: str) -> List[str]:
    """Top-level ``NAME = <literal expr>`` assignments the kernel closes over
    (e.g. weather-stencil ``BET_M``/``BET_P``). Carried verbatim (np->jnp) so
    the emitted module is self-contained. A tuple-unpack constant
    (``_GAMMA, _B1, _B2 = -0.1423, 1.0529, 0.3334`` -- lda_xc_potential's
    Perdew-Zunger coefficients) is carried too, so every name it binds
    resolves in the emitted module."""
    out: List[str] = []
    for s in tree.body:
        if not (isinstance(s, ast.Assign) and len(s.targets) == 1):
            continue
        tgt = s.targets[0]
        if isinstance(tgt, ast.Name):
            names = [tgt.id]
        elif isinstance(tgt, ast.Tuple) and all(isinstance(e, ast.Name) for e in tgt.elts):
            names = [e.id for e in tgt.elts]
        else:
            continue
        if func_name not in names:
            out.append(_u(s))
    return out


def _module_constant_names(tree: ast.Module, func_name: str) -> Set[str]:
    """The names bound by the module-level constant assignments carried by
    :func:`_module_constants` (single-Name or tuple-of-Names targets)."""
    out: Set[str] = set()
    for s in tree.body:
        if not (isinstance(s, ast.Assign) and len(s.targets) == 1):
            continue
        tgt = s.targets[0]
        if isinstance(tgt, ast.Name):
            names = [tgt.id]
        elif isinstance(tgt, ast.Tuple) and all(isinstance(e, ast.Name) for e in tgt.elts):
            names = [e.id for e in tgt.elts]
        else:
            continue
        if func_name not in names:
            out.update(names)
    return out


def _module_const_values(tree: ast.Module, func_name: str) -> dict:
    """Map each module-level ``NAME = <scalar literal expr>`` (and scalar
    tuple-unpack) to its concrete Python value, evaluated top-to-bottom so a
    constant defined from earlier ones (``LCG_M = 1 << 63``) resolves. Only
    int/float/bool/complex results are kept; a constant that references numpy or
    builds an array (``MATERIAL_PROBABILITIES = np.array([...])``) fails the
    restricted eval and is skipped. Read by :func:`_fold_const_branches`."""
    env: dict = {}
    for s in tree.body:
        if not (isinstance(s, ast.Assign) and len(s.targets) == 1):
            continue
        tgt = s.targets[0]
        try:
            val = eval(compile(ast.Expression(body=s.value), "<const>", "eval"), {"__builtins__": {}}, dict(env))
        except Exception:  # noqa: BLE001 -- references np / an array / an unknown name
            continue
        if isinstance(tgt, ast.Name):
            if isinstance(val, (int, float, complex)):  # bool is an int subclass
                env[tgt.id] = val
        elif isinstance(tgt, ast.Tuple) and all(isinstance(e, ast.Name) for e in tgt.elts) \
                and isinstance(val, tuple) and len(val) == len(tgt.elts):
            for e, v in zip(tgt.elts, val):
                if isinstance(v, (int, float, complex)):
                    env[e.id] = v
    env.pop(func_name, None)
    return env


def _fold_const_branches(fn: ast.FunctionDef) -> None:
    """Prune an ``if``/``elif`` whose test is fully determined by module-level
    scalar constants (:data:`_MODULE_CONST_VALUES`) down to the taken branch --
    exactly what Python would do, so it is semantics-preserving. This keeps a
    constant-configured branch chain (cloudsc's ``if yrecldp_nssopt == 0: ..
    elif == 1: zqe = .. elif == 2: ..`` with ``yrecldp_nssopt = 1``) from
    reading as a *conditional* write of ``zqe`` -- which the carry analysis would
    otherwise flag as loop-carried-but-undefined and refuse."""
    # Drop any module constant SHADOWED by a param or local assignment here --
    # inside this function that name is the local (traced) value, not the module
    # constant, so folding on the module value would be a miscompile.
    shadowed = {a.arg for a in fn.args.args} | _names_stored(fn)
    usable = {k: v for k, v in _MODULE_CONST_VALUES.items() if k not in shadowed}
    if not usable:
        return

    class _T(ast.NodeTransformer):

        def visit_If(self, node):
            self.generic_visit(node)  # fold inner elif chain first
            if _names_loaded(node.test) <= set(usable):
                try:
                    truth = eval(compile(ast.Expression(body=node.test), "<fold>", "eval"), {"__builtins__": {}},
                                 dict(usable))
                except Exception:  # noqa: BLE001
                    return node
                return node.body if truth else node.orelse  # [] removes the dead branch entirely
            return node

    _T().visit(fn)
    ast.fix_missing_locations(fn)


def _signature(fn: ast.FunctionDef) -> str:
    return ast.unparse(_np_to_jnp(ast.fix_missing_locations(
        ast.parse(f"def _({ast.unparse(fn.args)}): pass").body[0]))).split("(", 1)[1].rsplit(")", 1)[0]


def _loop_vars(fn: ast.FunctionDef) -> Set[str]:
    # Exclude indices of loops that will be unrolled: those become concrete
    # Python ints, so their ``:R**i``-style slices are static, not dynamic.
    return {n.target.id
            for n in ast.walk(fn) if isinstance(n, ast.For) and isinstance(n.target, ast.Name)} - _unroll_loop_vars(fn)


def _unroll_loop_vars(fn: ast.FunctionDef) -> Set[str]:
    """Indices of ``for i in range(STATIC)`` loops whose body uses ``i`` in a
    shape -- emitted as real Python loops the tracer unrolls (stockham_fft)."""
    out: Set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and \
                isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name) and n.iter.func.id == "range":
            if _index_in_shape(n, n.target.id) and _range_args_static(n.iter):
                out.add(n.target.id)
    return out


def _range_args_static(rng: ast.Call) -> bool:
    names: Set[str] = set()
    for a in rng.args:
        names |= _names_loaded(a)
    return names <= _EMIT_STATIC


def _desugar_foreach(fn: ast.FunctionDef) -> None:
    """``for x in arr:`` -> ``for _fe in range(arr.shape[0]): x = arr[_fe]`` so
    array-element iteration reuses the ``range`` loop machinery (crc16's
    ``for b in data``, contour_integral's ``for z in int_pts``). Only a plain
    Name iterable is handled."""

    class _T(ast.NodeTransformer):

        def visit_For(self, node):
            self.generic_visit(node)
            it = node.iter
            if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
                return node
            # A constant-literal sequence (lulesh's ``faces``) is a concrete Python
            # list in the emitted function -> leave the ``for`` literal so it
            # unrolls at trace time, rather than indexing it like an array.
            if isinstance(it, ast.Name) and (it.id in _MODULE_CONSTS or it.id in _LOCAL_CONSTS):
                return node
            if not (isinstance(node.target, ast.Name) and isinstance(it, ast.Name)):
                return node
            idx = "_fe_" + node.target.id
            bind = ast.Assign(targets=[ast.Name(id=node.target.id, ctx=ast.Store())],
                              value=ast.Subscript(value=ast.Name(id=it.id, ctx=ast.Load()),
                                                  slice=ast.Name(id=idx, ctx=ast.Load()),
                                                  ctx=ast.Load()))
            shape0 = ast.Subscript(value=ast.Attribute(value=ast.Name(id=it.id, ctx=ast.Load()),
                                                       attr="shape",
                                                       ctx=ast.Load()),
                                   slice=ast.Constant(value=0),
                                   ctx=ast.Load())
            node.target = ast.Name(id=idx, ctx=ast.Store())
            node.iter = ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[shape0], keywords=[])
            node.body = [bind] + node.body
            return ast.fix_missing_locations(node)

    _T().visit(fn)
    ast.fix_missing_locations(fn)


_TUPLE_CTR = [0]


def _expand_tuple_targets(fn: ast.FunctionDef) -> None:
    """``a[x], b[y] = expr`` -> ``__tup = expr; a[x] = __tup[0]; b[y] = __tup[1]``
    so each subscript target functionalises independently (nbody's
    ``KE[i+1], PE[i+1] = getEnergy(...)``). Plain Name-only unpacks are left
    alone -- JAX unpacks tuples directly."""

    class _T(ast.NodeTransformer):

        def visit_Assign(self, node):
            self.generic_visit(node)
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Tuple):
                return node
            elts = node.targets[0].elts
            if not any(isinstance(e, ast.Subscript) for e in elts):
                return node
            _TUPLE_CTR[0] += 1
            tup = f"__tup{_TUPLE_CTR[0]}"
            out = [ast.Assign(targets=[ast.Name(id=tup, ctx=ast.Store())], value=node.value)]
            for k, e in enumerate(elts):
                item = ast.Subscript(value=ast.Name(id=tup, ctx=ast.Load()),
                                     slice=ast.Constant(value=k),
                                     ctx=ast.Load())
                out.append(ast.Assign(targets=[e], value=item))
            return [ast.copy_location(s, node) for s in out]

    _T().visit(fn)
    ast.fix_missing_locations(fn)


_CHAIN_CTR = [0]


def _expand_chained_assigns(fn: ast.FunctionDef) -> None:
    """``a = b = rhs`` -> ``__chain = rhs; a = __chain; b = __chain`` so each
    target is a single assignment the later passes can rewrite (covariance /
    correlation write the same row+column from one dot)."""

    class _T(ast.NodeTransformer):

        def visit_Assign(self, node):
            self.generic_visit(node)
            if len(node.targets) <= 1:
                return node
            _CHAIN_CTR[0] += 1
            tmp = f"__chain{_CHAIN_CTR[0]}"
            out = [ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=node.value)]
            out += [ast.Assign(targets=[t], value=ast.Name(id=tmp, ctx=ast.Load())) for t in node.targets]
            return [ast.copy_location(s, node) for s in out]

    _T().visit(fn)
    ast.fix_missing_locations(fn)


_BOOL_FUNCS = ("logical_and", "logical_or", "logical_not", "logical_xor", "less", "greater", "less_equal",
               "greater_equal", "equal", "not_equal", "isfinite", "isnan", "isinf")


def _is_bool_expr(node: ast.AST) -> bool:
    """A provably-boolean array expression: a comparison, an ``np.logical_*`` /
    predicate call, or a bitwise combination (``&``/``|``/``^``/``~``) of such
    (force_lj's ``in_range = (rsq < cutoffsq) & (rsq > 0.0)`` -- a boolean mask
    that ``_boolean_mask_transform`` must recognise to lower ``A[mask] = ..`` to
    ``where`` rather than an untraceable dynamic-size boolean index)."""
    if isinstance(node, ast.Compare):
        return True
    if isinstance(node, ast.Call):
        return any(_is_np_attr(node.func, f) for f in _BOOL_FUNCS)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
        return _is_bool_expr(node.left) and _is_bool_expr(node.right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return _is_bool_expr(node.operand)
    return False


def _bool_cond_ast(node: ast.AST) -> ast.AST:
    """Coerce a Python-truthiness condition into an explicit boolean-array
    expression a jit trace can evaluate directly: ``a and b`` -> ``mask(a) &
    mask(b)``, ``a or b`` -> ``mask(a) | mask(b)``, ``not a`` -> ``~mask(a)``.
    A non-boolean operand ``x`` (cloudsc's integer cumulus flag ``ldcum[jl-1]``,
    mixed with comparisons in ``if ldcum[jl-1] and plude > rlmin: ...``) becomes
    ``x != 0`` -- its numpy truthiness -- so the whole condition lowers to a
    traceable mask instead of raising ``TracerBoolConversionError`` on the bare
    Python ``and``. Already-boolean operands (comparisons / ``np.logical_*`` /
    bitwise-of-bool) are left untouched, so an all-comparison condition emits the
    same ``&``/``|`` mask as before."""
    if isinstance(node, ast.BoolOp):
        op = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        expr = _bool_cond_ast(node.values[0])
        for v in node.values[1:]:
            expr = ast.BinOp(left=expr, op=op, right=_bool_cond_ast(v))
        return expr
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return ast.UnaryOp(op=ast.Invert(), operand=_bool_cond_ast(node.operand))
    if _is_bool_expr(node):
        return node
    return ast.Compare(left=node, ops=[ast.NotEq()], comparators=[ast.Constant(value=0)])


def _cond_str(test: ast.AST) -> str:
    """Unparse a condition (np->jnp) as a traceable boolean-array expression."""
    return _u(_bool_cond_ast(test))


def _boolean_mask_transform(fn: ast.FunctionDef) -> None:
    """Lower boolean-mask indexing (no static shape under jit) to ``where``:

    * ``A[m] = rhs``         -> ``A = np.where(m, rhs|A[m]->A, A)``
    * ``A[m] <op>= rhs``     -> ``A = np.where(m, A <op> rhs|..., A)``
    * ``A[m].mean()``        -> ``np.sum(np.where(m, A, 0)) / np.sum(m)``
    * ``A[m].sum()``         -> ``np.sum(np.where(m, A, 0))``

    where ``m`` is a comparison/``np.logical_*`` result (or a name bound to
    one). Masked-out lanes are the identity, so this is exact. Powers
    mandelbrot1's escape update and nbody's ``inv_r3[inv_r3>0]**-1.5``."""
    bool_names = set()
    for s in ast.walk(fn):
        if isinstance(s, ast.Assign) and _is_bool_expr(s.value):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    bool_names.add(t.id)

    def is_mask(idx):
        return _is_bool_expr(idx) or (isinstance(idx, ast.Name) and idx.id in bool_names)

    # Inline a masked-subset temp: ``v = data[mask]; ... v.mean()`` (azimint_
    # naive) -> drop the def and substitute ``data[mask]`` so the reduction
    # rewrite below applies. Only single-assignment Names qualify.
    store_counts: dict = {}
    for s in ast.walk(fn):
        for t in (s.targets if isinstance(s, ast.Assign) else []):
            if isinstance(t, ast.Name):
                store_counts[t.id] = store_counts.get(t.id, 0) + 1
    subset_map = {}
    for s in ast.walk(fn):
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) and \
                isinstance(s.value, ast.Subscript) and is_mask(s.value.slice) and store_counts.get(s.targets[0].id) == 1:
            subset_map[s.targets[0].id] = s.value
    if subset_map:

        class _Sub(ast.NodeTransformer):

            def visit_Assign(self, node):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and \
                        node.targets[0].id in subset_map:
                    return None  # drop the now-inlined definition
                self.generic_visit(node)
                return node

            def visit_Name(self, n):
                if isinstance(n.ctx, ast.Load) and n.id in subset_map:
                    return _copy(subset_map[n.id])
                return n

        _Sub().visit(fn)
        ast.fix_missing_locations(fn)

    def widen(node):

        class _W(ast.NodeTransformer):

            def visit_Subscript(self, n):
                self.generic_visit(n)
                return n.value if is_mask(n.slice) else n

        return _W().visit(_copy(node))

    class _T(ast.NodeTransformer):

        def visit_Call(self, node):
            self.generic_visit(node)
            # ``X[m].mean()`` / ``.sum()``  and ``np.sum(X[m])`` / ``np.mean``
            red = None
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("mean", "sum") and \
                    isinstance(node.func.value, ast.Subscript) and is_mask(node.func.value.slice):
                red, sub = node.func.attr, node.func.value
            elif (_is_np_attr(node.func, "sum") or _is_np_attr(node.func, "mean")) and len(node.args) == 1 and \
                    isinstance(node.args[0], ast.Subscript) and is_mask(node.args[0].slice):
                red, sub = node.func.attr, node.args[0]
            if red is None:
                return node
            m, arr = sub.slice, sub.value
            masked = _np_call("where", [_copy(m), arr, ast.Constant(value=0)])
            total = _np_call("sum", [masked])
            if red == "sum":
                return ast.copy_location(total, node)
            return ast.copy_location(ast.BinOp(left=total, op=ast.Div(), right=_np_call("sum", [_copy(m)])), node)

        def visit_Assign(self, node):
            self.generic_visit(node)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript) and \
                    is_mask(node.targets[0].slice):
                tgt = node.targets[0]
                m, arr = tgt.slice, tgt.value
                new = _np_call("where", [_copy(m), widen(node.value), _copy(arr)])
                return ast.copy_location(ast.Assign(targets=[_copy(arr)], value=new), node)
            return node

        def visit_AugAssign(self, node):
            self.generic_visit(node)
            if isinstance(node.target, ast.Subscript) and is_mask(node.target.slice):
                tgt = node.target
                m, arr = tgt.slice, tgt.value
                rhs = ast.BinOp(left=_copy(arr), op=node.op, right=widen(node.value))
                new = _np_call("where", [_copy(m), rhs, _copy(arr)])
                return ast.copy_location(ast.Assign(targets=[_copy(arr)], value=new), node)
            return node

    _T().visit(fn)
    ast.fix_missing_locations(fn)


def _np_call(name: str, args: List[ast.AST]) -> ast.Call:
    return ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=name, ctx=ast.Load()),
                    args=args,
                    keywords=[])


def _mask_reduction_slices(fn: ast.FunctionDef) -> None:
    """Rewrite a variable-width slice that feeds a reduction into a masked
    full-width operand, so the reduction no longer needs a dynamic shape.

    The triangular linear-algebra kernels all reduce over a prefix/suffix:
    ``A[i, :j] @ A[:j, j]``, ``np.dot(A[i, :k], A[j, :k])``,
    ``np.dot(A[i+1:, i], B[i+1:, j])``. Because the masked-out entries are 0 --
    the identity for ``@``/``sum`` -- replacing ``X[.., :j, ..]`` with
    ``np.where(np.arange(n) < j, X[.., :, ..], 0)`` is exact. Operands that are
    not a clean one-sided dynamic slice are left alone (and may be rejected
    later)."""
    lv = _loop_vars(fn)

    class _T(ast.NodeTransformer):

        def visit_BinOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, ast.MatMult):
                node.left = _maybe_mask(node.left, lv)
                node.right = _maybe_mask(node.right, lv)
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if _is_np_attr(node.func, "dot") and len(node.args) == 2:
                node.args = [_maybe_mask(a, lv) for a in node.args]
            return node

    _T().visit(fn)
    ast.fix_missing_locations(fn)


def _is_np_attr(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == name and \
        isinstance(node.value, ast.Name) and node.value.id in ("np", "jnp")


def _dyn_slice_info(node: ast.AST, lv: Set[str]):
    """For ``Arr[.., dynamic-slice, ..]`` return ``(arr, axis, lower, upper)``
    (each bound an AST or None) when at least one bound depends on a loop var;
    else None. Covers one-sided (``:j``, ``i:``) and two-sided-but-one-dynamic
    (``i:M``, ``i+1:M``) slices. ``arr`` must be a plain Name and the slice the
    only ``ast.Slice`` in the subscript."""
    if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)):
        return None
    sl = node.slice
    elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]

    def is_dyn(e):
        return isinstance(e, ast.Slice) and e.step is None and \
            ((e.lower is not None and bool(_names_loaded(e.lower) & lv)) or
             (e.upper is not None and bool(_names_loaded(e.upper) & lv)))

    dyn_positions = [k for k, e in enumerate(elts) if is_dyn(e)]
    if len(dyn_positions) != 1:
        return None
    # Other axes must be plain int indices or full ``:`` slices (the only ones
    # the 1-D axis mask broadcasts cleanly against).
    p = dyn_positions[0]
    for k, e in enumerate(elts):
        if k != p and isinstance(e, ast.Slice) and not _is_full_slice(e):
            return None
    s = elts[p]
    return node.value, p, s.lower, s.upper


def _axis_mask(arr: ast.AST, p: int, lower: Optional[ast.AST], upper: Optional[ast.AST]) -> ast.AST:
    """``np.arange(arr.shape[p])`` constrained by the present bounds:
    ``(arange >= lower) & (arange < upper)``."""
    shape_p = ast.Subscript(value=ast.Attribute(value=arr, attr="shape", ctx=ast.Load()),
                            slice=ast.Constant(value=p),
                            ctx=ast.Load())
    arange = ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="arange", ctx=ast.Load()),
                      args=[shape_p],
                      keywords=[])
    terms = []
    if lower is not None:
        terms.append(ast.Compare(left=arange, ops=[ast.GtE()], comparators=[_copy(lower)]))
    if upper is not None:
        terms.append(ast.Compare(left=arange, ops=[ast.Lt()], comparators=[_copy(upper)]))
    mask = terms[0]
    for t in terms[1:]:
        mask = ast.BinOp(left=mask, op=ast.BitAnd(), right=t)
    return mask


def _widen_to_full(node: ast.Subscript, p: int) -> ast.AST:
    """Replace the dynamic slice axis ``p`` of a subscript with full ``:``."""
    if isinstance(node.slice, ast.Tuple):
        new_elts = list(node.slice.elts)
        new_elts[p] = ast.Slice(lower=None, upper=None, step=None)
        return ast.Subscript(value=node.value, slice=ast.Tuple(elts=new_elts, ctx=ast.Load()), ctx=ast.Load())
    return node.value  # ``v[:k]`` -> whole vector ``v``


def _maybe_mask(node: ast.AST, lv: Set[str]) -> ast.AST:
    info = _dyn_slice_info(node, lv)
    if info is None:
        return node
    arr, p, lower, upper = info
    full = _widen_to_full(node, p)
    mask = _axis_mask(arr, p, lower, upper)
    where = ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="where", ctx=ast.Load()),
                     args=[mask, full, ast.Constant(value=0)],
                     keywords=[])
    return ast.copy_location(where, node)


def _widen_dynamic_slices(node: ast.AST, lv: Set[str]) -> ast.AST:
    """Drop one-sided dynamic-slice bounds to full ``:`` (no zeroing -- a write
    mask does the truncation). Used on the RHS of a masked dynamic write."""

    class _W(ast.NodeTransformer):

        def visit_Subscript(self, n):
            self.generic_visit(n)
            info = _dyn_slice_info(n, lv)
            if info is None:
                return n
            _, p, _, _ = info
            if isinstance(n.slice, ast.Tuple):
                elts = list(n.slice.elts)
                elts[p] = ast.Slice(lower=None, upper=None, step=None)
                return ast.copy_location(
                    ast.Subscript(value=n.value, slice=ast.Tuple(elts=elts, ctx=ast.Load()), ctx=ast.Load()), n)
            return n.value  # ``v[:k]`` -> ``v``

    return _W().visit(node)


def _mask_dynamic_writes(fn: ast.FunctionDef) -> None:
    """Rewrite a write to a variable-width prefix/suffix into a masked write
    over the full axis: ``C[i, :i+1] += rhs`` becomes
    ``C[i, :] = np.where(np.arange(n) < i+1, C[i, :] + rhs_widened, C[i, :])``
    (then functionalised to ``.at[i, :].set(...)`` downstream). Covers the
    syrk/syr2k/symm rank-update prefixes."""
    lv = _loop_vars(fn)

    class _T(ast.NodeTransformer):

        def _rewrite(self, target, value):
            info = _dyn_slice_info(target, lv)
            if info is None:
                return None
            arr, p, lower, upper = info
            full = _widen_dynamic_slices(target, lv)
            mask = _axis_mask(arr, p, lower, upper)
            keep = _widen_dynamic_slices(target, lv)
            where = ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="where", ctx=ast.Load()),
                             args=[mask, value, keep],
                             keywords=[])
            return ast.Assign(targets=[full], value=where)

        def visit_AugAssign(self, node):
            self.generic_visit(node)
            if _dyn_slice_info(node.target, lv) is None:
                return node
            old = _widen_dynamic_slices(_copy(node.target), lv)
            rhs = ast.BinOp(left=old, op=node.op, right=_widen_dynamic_slices(node.value, lv))
            out = self._rewrite(node.target, rhs)
            return ast.copy_location(out, node) if out else node

        def visit_Assign(self, node):
            self.generic_visit(node)
            if len(node.targets) != 1 or _dyn_slice_info(node.targets[0], lv) is None:
                return node
            out = self._rewrite(node.targets[0], _widen_dynamic_slices(node.value, lv))
            return ast.copy_location(out, node) if out else node

    _T().visit(fn)
    ast.fix_missing_locations(fn)


def _copy(node: ast.AST) -> ast.AST:
    return ast.parse(ast.unparse(node), mode="eval").body


def _rewrite_flip_prefix(fn: ast.FunctionDef) -> None:
    """``np.flip(arr[:k])`` (reverse of a dynamic prefix) -> a reversal gather
    ``arr[np.clip(k-1 - np.arange(n), 0, n-1)]``. The surrounding reduction /
    write masking then truncates to the first ``k`` lanes, so durbin's
    ``np.dot(np.flip(r[:k]), y[:k])`` and ``y[:k] += a*np.flip(y[:k])`` lower
    without a data-dependent shape."""
    lv = _loop_vars(fn)

    class _T(ast.NodeTransformer):

        def visit_Call(self, node):
            self.generic_visit(node)
            if not (_is_np_attr(node.func, "flip") and len(node.args) == 1):
                return node
            arg = node.args[0]
            if not (isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name)
                    and isinstance(arg.slice, ast.Slice) and arg.slice.lower is None and arg.slice.upper is not None and
                    (_names_loaded(arg.slice.upper) & lv)):
                return node
            arr, k = arg.value, arg.slice.upper
            n = ast.Subscript(value=ast.Attribute(value=arr, attr="shape", ctx=ast.Load()),
                              slice=ast.Constant(value=0),
                              ctx=ast.Load())
            arange = _np_call("arange", [n])
            km1 = ast.BinOp(left=_copy(k), op=ast.Sub(), right=ast.Constant(value=1))
            idx = ast.BinOp(left=km1, op=ast.Sub(), right=arange)
            hi = ast.BinOp(left=ast.Subscript(value=ast.Attribute(value=arr, attr="shape", ctx=ast.Load()),
                                              slice=ast.Constant(value=0),
                                              ctx=ast.Load()),
                           op=ast.Sub(),
                           right=ast.Constant(value=1))
            clipped = _np_call("clip", [idx, ast.Constant(value=0), hi])
            return ast.copy_location(ast.Subscript(value=arr, slice=clipped, ctx=ast.Load()), node)

    _T().visit(fn)
    ast.fix_missing_locations(fn)


def _mask_slice_reads(fn: ast.FunctionDef) -> None:
    """Mask a dynamic-slice read bound to an intermediate: ``cols = A_col[
    A_row[i]:A_row[i+1]]`` -> ``cols = np.where(mask, A_col, 0)`` (full width,
    0-filled). The CSR SpMV pattern then computes ``vals @ x[cols]`` correctly --
    masked ``vals`` lanes are 0 (the ``@`` identity) so the gather at the
    0-filled ``cols`` lanes contributes nothing. Mirrors the hand spmv_jax."""
    lv = _loop_vars(fn)

    class _T(ast.NodeTransformer):

        def visit_Assign(self, node):
            self.generic_visit(node)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and \
                    isinstance(node.value, ast.Subscript) and _dyn_slice_info(node.value, lv) is not None:
                node.value = _maybe_mask(node.value, lv)
            return node

    _T().visit(fn)
    ast.fix_missing_locations(fn)


def _dynamic_window_slices(fn: ast.FunctionDef) -> None:
    """Rewrite a fixed-width sliding window ``arr[.., i:i+K, ..]`` (K static,
    i a loop var) to ``lax.dynamic_slice_in_dim(arr, i, K, axis)`` -- the conv
    kernels' ``input[:, i:i+K, j:j+K, :, None]``. Multiple windowed axes nest;
    the residual int/full/newaxis indices apply afterwards."""
    lv = _loop_vars(fn)

    def window(s):
        # ``lo:lo+W`` with lo dynamic and W static -> (lo, W); else None.
        if not (isinstance(s, ast.Slice) and s.lower is not None and s.upper is not None and s.step is None):
            return None
        if not (_names_loaded(s.lower) & lv):
            return None
        u = s.upper
        if isinstance(u, ast.BinOp) and isinstance(u.op, ast.Add):
            if ast.unparse(u.left) == ast.unparse(s.lower) and not (_names_loaded(u.right) & lv):
                return s.lower, u.right
            if ast.unparse(u.right) == ast.unparse(s.lower) and not (_names_loaded(u.left) & lv):
                return s.lower, u.left
        return None

    class _T(ast.NodeTransformer):

        def visit_Subscript(self, node):
            self.generic_visit(node)
            if not isinstance(node.value, ast.Name):
                return node
            elts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            wins = [(k, window(e)) for k, e in enumerate(elts)]
            wins = [(k, w) for k, w in wins if w is not None]
            if not wins:
                return node
            arr = node.value
            for k, (start, width) in wins:
                arr = ast.Call(func=ast.Attribute(value=ast.Name(id="lax", ctx=ast.Load()),
                                                  attr="dynamic_slice_in_dim",
                                                  ctx=ast.Load()),
                               args=[arr, start, width, ast.Constant(value=k)],
                               keywords=[])
            resid = list(elts)
            for k, _ in wins:
                resid[k] = ast.Slice(lower=None, upper=None, step=None)
            new_slice = ast.Tuple(elts=resid, ctx=ast.Load()) if isinstance(node.slice, ast.Tuple) else resid[0]
            return ast.copy_location(ast.Subscript(value=arr, slice=new_slice, ctx=ast.Load()), node)

    _T().visit(fn)
    ast.fix_missing_locations(fn)


def _for_is_unrolled(node: ast.For) -> bool:
    """Will ``_emit_for`` UNROLL this ``for`` at trace time (its index/carry stay
    concrete Python values), rather than lower it to a rolled ``fori_loop`` /
    ``while_loop`` (its carry becomes a tracer)? True for a static-iterable /
    literal-sequence loop and for a static-range loop whose index feeds a shape
    (and is not a time-stepping loop). Used to decide which loop-written scalars
    become tracers post-loop -- see :func:`_loop_index_tainted`."""
    rng = node.iter
    if not (isinstance(rng, ast.Call) and isinstance(rng.func, ast.Name) and rng.func.id == "range"):
        return _is_static_iterable(rng) or isinstance(rng, (ast.Tuple, ast.List))
    i = node.target.id if isinstance(node.target, ast.Name) else "_i"
    return _index_in_shape(node, i) and _range_args_static(rng) and not _is_timestep_loop(node)


def _rolled_loop_writes(fn: ast.FunctionDef) -> Set[str]:
    """Names written inside a ROLLED loop body (a ``while`` or a non-unrolled
    ``for``). Such a name is a loop CARRY -- a tracer once the loop is a
    ``fori_loop`` / ``while_loop`` -- so slicing an array by it after the loop
    (ls3df's Lanczos ``alphas[:na]`` with the step counter ``na += 1``) is just
    as data-dependent as slicing by the index itself."""
    out: Set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.While):
            out |= _names_stored(ast.Module(body=n.body, type_ignores=[]))
        elif isinstance(n, ast.For) and not _for_is_unrolled(n):
            out |= _names_stored(ast.Module(body=n.body, type_ignores=[]))
    return out


def _loop_index_tainted(fn: ast.FunctionDef) -> Set[str]:
    """Loop-index vars, scalars carried through a rolled loop (:func:`_rolled_
    loop_writes`), plus every scalar TRANSITIVELY derived from one by a plain
    ``name = <expr(tainted)>`` assignment (dwt2d's ``s = n >> lvl`` with ``lvl``
    the ``fori`` index). A slice bound built from such a name is just as
    data-dependent as one naming the index directly."""
    tainted = set(_loop_vars(fn)) | _rolled_loop_writes(fn)
    changed = True
    while changed:
        changed = False
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                    and n.targets[0].id not in tainted and (_names_loaded(n.value) & tainted):
                tainted.add(n.targets[0].id)
                changed = True
    return tainted


def _reject_dynamic_slices(fn: ast.FunctionDef) -> None:
    """Raise if any ``ast.Slice`` bound depends on a loop-index variable
    (e.g. cholesky's ``A[i, :j]`` with ``j`` a loop var). Such variable-length
    slices have no static shape, so they cannot be traced -- a hand-written
    JAX kernel would have to mask/pad. Honest fallback beats broken output.
    Unrolled-loop indices are excluded (concrete -> their slices are static).
    The taint is transitive so a slice bound built from a loop-index-derived
    scalar (dwt2d's ``s = n >> lvl`` then ``out[:s, :s]``) is caught too, rather
    than emitting a module that crashes with a traced slice at run time."""
    tainted = _loop_index_tainted(fn)
    for n in ast.walk(fn):
        if isinstance(n, ast.Slice):
            for part in (n.lower, n.upper, n.step):
                if part is not None and (_names_loaded(part) & tainted):
                    raise EmitError("data-dependent slice bound (needs masking/padding)")


_SHAPE_FUNCS = ("zeros", "ones", "empty", "full", "reshape", "arange", "zeros_like", "ones_like", "broadcast_to",
                "tile", "repeat", "linspace", "histogram")

_ARRAY_ATTRS = ("shape", "size", "ndim", "T", "dtype", "real", "imag", "max", "min", "mean", "sum", "std", "var", "dot",
                "copy", "flatten", "ravel", "reshape", "transpose", "conj", "prod", "argmax", "argmin")

# Shape/count funcs whose first positional arg is the data array, not a dim.
_LEADING_DATA_FUNCS = ("reshape", "histogram", "tile", "repeat", "broadcast_to", "zeros_like", "ones_like",
                       "empty_like", "full_like")


def _static_want(fn: ast.FunctionDef, params) -> Set[str]:
    """Params ``fn`` needs CONCRETE at trace time: those feeding a ``range()``
    bound, an array-shape dimension, or an ``if``/``while``/ternary CONDITION.
    (Array-likeness is NOT excluded here -- see :func:`_static_params`.)"""
    pset = set(params)
    want: Set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            is_range = isinstance(n.func, ast.Name) and n.func.id == "range"
            attr = n.func.attr if isinstance(n.func, ast.Attribute) else None
            if is_range or attr in _SHAPE_FUNCS:
                # Funcs with a leading data-array arg (reshape(a, shape),
                # histogram(a, bins), ...) -- skip arg 0; its dims live after.
                scan = n.args[1:] if attr in _LEADING_DATA_FUNCS else n.args
                for a in scan:
                    want |= (_names_loaded(a) & pset)
        # A scalar param that drives an ``if``/``while``/ternary CONDITION is a
        # control parameter that must be concrete at trace time: a data-dependent
        # Python branch on a traced scalar cannot yield a bool, and
        # (contour_integral's ``if NR == NM`` choosing ``inv`` vs ``solve``) the
        # two arms may have INCOMPATIBLE shapes that no ``jnp.where`` can merge --
        # only a static predicate picks a single arm. Such a param is an implicit
        # dimension (``NM`` = ``Y.shape[1]``) even when it never feeds a shape.
        # fv3_dycore's ``ord_inner = 8 if hord == 10 else hord`` makes ``hord``
        # control-critical. (Excluded in _static_params if also used as data.)
        elif isinstance(n, (ast.If, ast.While, ast.IfExp)):
            want |= (_names_loaded(n.test) & pset)
    return want


def _array_like_params(fn: ast.FunctionDef, params) -> Set[str]:
    """Params ever used as an ARRAY VALUE -- subscripted, accessed via an array
    attribute/reduction (``x.shape``, ``x.max()``), or a whole-array ``@``
    operand -- so they are data, never a static scalar dimension."""
    array_like: Set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
            array_like.add(n.value.id)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.attr in _ARRAY_ATTRS:
            array_like.add(n.value.id)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult):
            for side in (n.left, n.right):
                if isinstance(side, ast.Name) and side.id in params:
                    array_like.add(side.id)
    return array_like


def _static_params(fn: ast.FunctionDef, params) -> List[str]:
    """Params requiring concreteness during tracing (``range`` bound, shape dim,
    branch predicate), minus those ever used as array data."""
    want = _static_want(fn, params)
    array_like = _array_like_params(fn, params)
    return [p for p in params if p in want and p not in array_like]


def _value_names(node: ast.AST, pset: Set[str]) -> Set[str]:
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


def _concrete_bound_want(fn: ast.FunctionDef, params) -> Set[str]:
    """Params whose VALUE (or an element of it) feeds a ``range()`` bound or an
    array SHAPE dimension -- positions that need a concrete Python int at trace
    time. Unlike :func:`_static_want` this EXCLUDES branch predicates (a data-
    dependent ``if`` / ternary lowers to ``jnp.where``, cloudsc's ``if
    ldcum[jl-1]``) and ``.shape`` accesses (statically known), so only a
    genuinely un-AoT-able array-as-a-bound (xsbench's ``num_nucs[mat]`` sizing
    the nuclide loop) lands here."""
    pset = set(params)
    want: Set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            is_range = isinstance(n.func, ast.Name) and n.func.id == "range"
            attr = n.func.attr if isinstance(n.func, ast.Attribute) else None
            if is_range or attr in _SHAPE_FUNCS:
                scan = n.args[1:] if attr in _LEADING_DATA_FUNCS else n.args
                for a in scan:
                    want |= _value_names(a, pset)
    return want


def _propagate_param_flow(funcs: dict, seed: dict) -> dict:
    """Fix-point closure of a per-function param set over call edges: a caller
    param that flows (via a call argument) into a callee param already in the set
    joins the set. ``seed`` maps each function to a set of ITS OWN params; the
    result maps each to that set grown transitively. Shared by the static-want,
    concrete-bound-want and array-like passes (fv3_dycore's ``hord`` reaches
    ``xppm_flux``'s ``if mord == 5`` only through the call chain)."""
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
                        for nm in _names_loaded(call.args[pos]) & pset:
                            if nm not in out[name]:
                                out[name].add(nm)
                                changed = True
    return out


def _transitive_array_like(funcs: dict) -> dict:
    """``{func_name: array-like params}`` propagated FORWARD across calls: a param
    passed DIRECTLY (as a plain Name) into a callee position that is array-like
    there is itself array-like. gromacs/xsbench pass their integer index arrays
    (``ci_cluster``, ``num_nucs``) straight through to helpers that subscript
    them -- the kernel body never touches them, so a body-only check misses that
    they are DATA, not static scalar dimensions."""
    params_of = {name: [a.arg for a in f.args.args] for name, f in funcs.items()}
    # Seed with array-like PARAMS only (``_array_like_params`` also reports local
    # arrays and ``np`` from ``np.max`` -- neither is a callee param to trace).
    al = {name: (set(_array_like_params(f, params_of[name])) & set(params_of[name])) for name, f in funcs.items()}
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
                    if pos < len(call.args) and isinstance(call.args[pos], ast.Name) and call.args[pos].id in pset \
                            and call.args[pos].id not in al[name]:
                        al[name].add(call.args[pos].id)
                        changed = True
    return al


#: builtins that return a concrete scalar from concrete args (so ``abs(iord)`` /
#: ``int(x)`` stay static) -- read by :func:`_is_static_expr`.
_STATIC_BUILTINS = {"abs", "int", "float", "min", "max", "round", "len", "bool", "sum"}


def _is_static_expr(node: ast.AST, ctx: Set[str]) -> bool:
    """Is ``node`` evaluable to a concrete Python value at trace time, given the
    static names in ``ctx``? Literals, ``ctx`` names, arithmetic/compare/ternary
    over such, a ``.shape``/``.size``/``.ndim`` access, and a pure builtin /
    ``np.``/``math.`` call over static args all qualify -- so ``abs(iord)`` and
    ``int(egrid.shape[0]) - 1`` are static while ``seq[i]`` (a traced array
    element) and a user-helper call are not."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id in ctx
    if isinstance(node, ast.BinOp):
        return _is_static_expr(node.left, ctx) and _is_static_expr(node.right, ctx)
    if isinstance(node, ast.UnaryOp):
        return _is_static_expr(node.operand, ctx)
    if isinstance(node, ast.BoolOp):
        return all(_is_static_expr(v, ctx) for v in node.values)
    if isinstance(node, ast.Compare):
        return _is_static_expr(node.left, ctx) and all(_is_static_expr(c, ctx) for c in node.comparators)
    if isinstance(node, ast.IfExp):
        return all(_is_static_expr(x, ctx) for x in (node.test, node.body, node.orelse))
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_static_expr(e, ctx) for e in node.elts)
    if isinstance(node, ast.Attribute):
        return node.attr in ("shape", "size", "ndim")  # a statically-known dimension of any array
    if isinstance(node, ast.Subscript):  # e.g. ``x.shape[0]``
        return _is_static_expr(node.value, ctx) and _is_static_expr(node.slice, ctx)
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name) and f.id in _STATIC_BUILTINS:
            return all(_is_static_expr(a, ctx) for a in node.args)
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("np", "jnp", "math"):
            return all(_is_static_expr(a, ctx) for a in node.args)
        return False
    return False


def _static_ctx(fn: ast.FunctionDef, base: Set[str]) -> Set[str]:
    """``base`` (a function's concrete params + module constants) grown with its
    static LOCALS -- a single-assignment ``name = <static expr>`` (fv3_dycore's
    ``ord_inner = 8 if hord == 10 else hord``, ``nx = nhalo + ni + nhalo``). A
    name assigned more than once is skipped (conservative)."""
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
                if counts.get(t) == 1 and t not in ctx and _is_static_expr(node.value, ctx):
                    ctx.add(t)
                    changed = True
    return ctx


def _concrete_params(funcs: dict, kernel_name: str, kernel_static: List[str]) -> dict:
    """``{func_name: params CONCRETE at trace time}`` computed FORWARD from the
    kernel's static args: a callee param is concrete iff at EVERY call site it is
    passed a value built only from the caller's concrete params, module constants
    and literals (plus static locals derived from those). This is what a helper's
    :data:`_EMIT_STATIC` must be -- fv3_dycore's ``hord`` reaches ``xppm_flux``'s
    ``if mord == 5`` as a static value (real Python branch), whereas nussinov's
    ``match(seq[i], seq[j])`` passes TRACED array elements, so ``match``'s
    ``b1``/``b2`` are NOT concrete and its ``if b1 + b2 == 3`` must lower to
    ``jnp.where`` rather than a Python branch on a tracer. Requiring ALL call
    sites to agree keeps a helper shared between a static and a traced caller
    correctly non-concrete."""
    params_of = {name: [a.arg for a in f.args.args] for name, f in funcs.items()}
    defaults_of: dict = {}
    for name, f in funcs.items():
        ps = params_of[name]
        dflts = f.args.defaults
        off = len(ps) - len(dflts)
        defaults_of[name] = {ps[off + k]: d for k, d in enumerate(dflts)}
    mc = set(_MODULE_CONSTS)
    concrete = {name: set() for name in funcs}
    concrete[kernel_name] = set(kernel_static)
    changed = True
    while changed:
        changed = False
        votes: dict = {name: {} for name in funcs}
        for caller, f in funcs.items():
            ctx = _static_ctx(f, set(concrete[caller]) | mc)
            for call in ast.walk(f):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in funcs):
                    continue
                callee = call.func.id
                cps = params_of[callee]
                passed: dict = {}
                for pos, arg in enumerate(call.args):
                    if pos < len(cps):
                        passed[cps[pos]] = _is_static_expr(arg, ctx)
                for kw in call.keywords:
                    if kw.arg in cps:
                        passed[kw.arg] = _is_static_expr(kw.value, ctx)
                for p in cps:  # an omitted arg falls back to its default (evaluated in module scope)
                    if p not in passed:
                        d = defaults_of[callee].get(p)
                        passed[p] = d is not None and _is_static_expr(d, mc)
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


def _transitive_static(kernel_name: str, funcs: dict) -> List[str]:
    """The kernel's ``static_argnames``, computed TRANSITIVELY across calls: a
    kernel param is static if it (or a value it flows into via a call argument)
    feeds a ``range``/shape/branch anywhere in a reachable helper. fv3_dycore's
    ``hord``/``grid_type`` never feed a branch in the KERNEL -- only deep inside
    ``xppm_flux``'s ``if mord == 5`` and ``_fv_tp_2d``'s ``8 if hord == 10 else
    hord`` -- so without propagation they stay traced and the Python branch on
    them raises ``TracerBoolConversionError``."""
    params_of = {name: [a.arg for a in f.args.args] for name, f in funcs.items()}
    want = _propagate_param_flow(funcs, {n: _static_want(f, params_of[n]) for n, f in funcs.items()})
    cwant = _propagate_param_flow(funcs, {n: _concrete_bound_want(f, params_of[n]) for n, f in funcs.items()})
    kparams = params_of[kernel_name]
    array_like = _transitive_array_like(funcs)[kernel_name]
    # A param whose VALUE must be a CONCRETE int (a range bound / shape dim) yet
    # is array DATA is a data-dependent bound (xsbench's ``num_nucs[mat]`` sizing
    # the nuclide loop, gromacs' cluster arrays). It can be neither a
    # ``static_argnames`` (an array is unhashable) nor traced into a ``range`` --
    # the classifier can't express it, so refuse and let the eager form (which
    # runs the concrete loop directly) take over. A mere branch on an array
    # element is NOT a conflict (it lowers to ``jnp.where``), so ``cwant`` -- not
    # the full ``want`` -- gates this.
    conflict = [p for p in kparams if p in cwant[kernel_name] and p in array_like]
    if conflict:
        raise EmitError(f"data-dependent shape/bound from array data: {', '.join(conflict)}")
    return [p for p in kparams if p in want[kernel_name] and p not in array_like]


def _augment_returns(fn: ast.FunctionDef, mutated: List[str]) -> None:
    """Append in-place-mutated params to each TOP-LEVEL ``return`` so a kernel
    that returns a scalar/derived value while mutating output arrays in place
    (channel_flow returns the step COUNT but mutates ``u``/``v``) still hands
    the functional results back. Only direct ``fn.body`` returns are touched
    (the function's real exit points); names already returned are not
    duplicated. A no-return kernel is handled by the caller's append path."""
    for stmt in fn.body:
        if not (isinstance(stmt, ast.Return) and stmt.value is not None):
            continue
        cur = list(stmt.value.elts) if isinstance(stmt.value, ast.Tuple) else [stmt.value]
        present = {e.id for e in cur if isinstance(e, ast.Name)}
        extra = [m for m in mutated if m not in present]
        if extra:
            stmt.value = ast.Tuple(elts=cur + [ast.Name(id=m, ctx=ast.Load()) for m in extra], ctx=ast.Load())


def _own_returns(fn: ast.FunctionDef) -> List[ast.Return]:
    """``Return`` statements belonging to ``fn`` itself -- NOT to a nested helper
    def (a plain ``ast.walk`` descends into nested ``def gat(...): return ...``
    helpers, e.g. velocity_tendencies' gather shorthand, so the kernel would look
    like it already returns and the in-place output augmentation would be skipped,
    leaving the mutated outputs unreturned)."""
    out: List[ast.Return] = []

    def _visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue  # a nested scope -- its returns are its own
            if isinstance(child, ast.Return) and child.value:
                out.append(child)
            _visit(child)

    _visit(fn)
    return out


def _mutated_params(fn: ast.FunctionDef, params) -> List[str]:
    """Params written in place, returned in **signature order** (OptArena's
    ``output_args`` convention) rather than mutation-encounter order."""
    mutated: Set[str] = set()
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
                # Only a SUBSCRIPT target (``p[i] = ``, ``p[:] = ``) writes the
                # caller's array in place. A plain ``p = <expr>`` merely REBINDS the
                # local name -- numpy does not propagate that to the caller, so it
                # is NOT an output. Treating it as one wrongly augments the emitted
                # return (``_upper_bound``'s ``v = v / norm`` normalisation makes it
                # return ``(theta, v)``, breaking the ``1.2 * _upper_bound(...)``
                # call site with a ``tuple * float``).
                if not isinstance(t, ast.Subscript):
                    continue
                base = t.value
                while isinstance(base, ast.Subscript):
                    base = base.value
                if isinstance(base, ast.Name) and base.id in params:
                    mutated.add(base.id)
        # numpy ufunc in-place scatter ``np.add.at(target, idx, val)`` mutates its
        # first arg through a Call, not an assignment target -- the jax emitter
        # later rewrites it to ``target = target.at[idx].add(val)``, so the param
        # IS mutated and must be returned.
        elif isinstance(s, ast.Call):
            f = s.func
            if isinstance(f, ast.Attribute) and f.attr == "at" and isinstance(f.value, ast.Attribute) and s.args:
                base = s.args[0]
                while isinstance(base, ast.Subscript):
                    base = base.value
                if isinstance(base, ast.Name) and base.id in params:
                    mutated.add(base.id)
    return [p for p in params if p in mutated]


def _mutation_maps(funcs: dict) -> dict:
    """``{func_name: [mutated params, signature order]}`` computed TRANSITIVELY.

    A param is an output if the function writes through it in place -- directly
    (``p[i] = ..``, ``p += ..``, ``np.add.at(p, ..)``; :func:`_mutated_params`)
    OR by passing it to a callee that mutates its argument at that position
    (fv3_dycore's ``xppm`` mutates ``al``/``xflux`` only through its bare
    ``compute_al_x(.., al, ..)`` / ``xppm_flux(.., al, xflux, ..)`` calls). The
    fixpoint propagates callee outputs back to the caller's args so a helper that
    merely orchestrates in-place sub-stencils is still recognised as mutating --
    and its bare call site therefore functionalises to ``al, xflux = xppm(..)``
    rather than surviving as a bare-expression statement.

    A plain rebind ``p = <expr>`` is NOT a mutation (numpy does not propagate it),
    so a helper that only normalises a param locally (``v = v / norm``) is not
    wrongly turned into a tuple-returning output."""
    params_of = {name: [a.arg for a in f.args.args] for name, f in funcs.items()}
    muts = {name: set(_mutated_params(f, params_of[name])) for name, f in funcs.items()}
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
                        base = _base_name(call.args[pos])
                        if base in pset and base not in muts[name]:
                            muts[name].add(base)
                            changed = True
    return {name: [p for p in params_of[name] if p in muts[name]] for name in funcs}
