"""Jit-mode statement emission: each Python loop lowered to a vectorised op, ``lax.fori_loop`` or
``lax.while_loop``, and in-place updates made functional."""

import ast
import copy
import enum

from hpcagent_bench.translators.numpyto_common.parallelism import is_timestep_loop
from hpcagent_bench.translators.numpyto_common.subscripts import is_full_slice

from hpcagent_bench.translators.numpyto_jax.errors import EmitError
from hpcagent_bench.translators.numpyto_jax.jnp import cond_str, unparse_jnp
from hpcagent_bench.translators.numpyto_jax.names import (
    base_name,
    has_break,
    is_identity_test,
    load,
    names_loaded,
    names_stored,
    reads_before_write,
    stmt_rhs_loads,
    tuple_expr,
    upward_exposed,
)
from hpcagent_bench.translators.numpyto_jax.state import STATE
from hpcagent_bench.translators.numpyto_jax.vocab import LEADING_DATA_FUNCS, SHAPE_FUNCS

__all__ = [
    "ROW_REDUCE_FUNCS",
    "SCATTER_AT_METHOD",
    "LoopKind",
    "broadcast_astype",
    "carried_vars",
    "classify_for",
    "devectorize_index",
    "emit_body",
    "emit_for",
    "emit_fori",
    "emit_if",
    "emit_iterable_for",
    "emit_vectorized",
    "emit_while",
    "emit_while_break",
    "expand_parallel_assigns",
    "functionalize_stmt",
    "index_in_shape",
    "is_const_literal",
    "is_index_i",
    "is_return_only",
    "is_static_iterable",
    "local_const_seq_names",
    "loop_vars",
    "parse_range",
    "range_args_static",
    "range_covers_full_extent",
    "row_reduce_rewrite",
    "row_reduce_target",
    "scatter_at_assign",
    "split_on_break",
    "unroll_loop_vars",
]


class LoopKind(enum.Enum):
    """How a jit-mode ``for`` loop lowers."""

    VECTORIZE = "vectorize"  # independent elementwise -> whole-array op
    FORI = "fori_loop"  # fixed trip count, loop-carried state
    WHILE = "while_loop"  # data-dependent termination (break / while)


def index_in_shape(node: ast.For, i: str) -> bool:
    """Does the loop index appear in a shape/count argument (``reshape(_, (R**i,
    ...))``, ``zeros((i, ...))``) inside the body? Such uses need a concrete index."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in SHAPE_FUNCS:
            scan = n.args[1:] if n.func.attr in LEADING_DATA_FUNCS else n.args
            for a in scan:
                if i in names_loaded(a):
                    return True
    return False


def classify_for(node: ast.For) -> LoopKind:
    """Decide which JAX construct a ``for i in range(...)`` lowers to."""
    if has_break(node.body):
        return LoopKind.WHILE
    target = node.target
    if not isinstance(target, ast.Name):
        return LoopKind.FORI
    i = target.id
    # Carried iff written and (read in body, or an index-written array also
    # read) -- state threading. Pure ``a[i] = f(<i-indexed>)`` with no other
    # read of ``a`` is independent -> vectorisable.
    stored = names_stored(ast.Module(body=node.body, type_ignores=[]))
    for s in node.body:
        if not (isinstance(s, ast.Assign) and len(s.targets) == 1):
            return LoopKind.FORI  # anything non-trivial -> safe carried form
        tgt = s.targets[0]
        if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
            # a[<i>] = expr is independent only if the subscript is exactly i,
            # RHS doesn't read `a`, and RHS uses `i` only inside subscripts. A
            # bare-scalar use of `i` (e.g. ``res[i] = rmax * i / npt``) would
            # dangle if the loop were dropped -> keep fori_loop + .at[].set().
            arr = tgt.value.id
            if arr in names_loaded(s.value):
                return LoopKind.FORI
            if not is_index_i(tgt.slice, i):
                return LoopKind.FORI
            # A rank-reducing reduction over the indexed row (``np.sum(a[i])``)
            # can't devectorise by dropping ``[i]`` alone -- see
            # ``row_reduce_rewrite``. It rewrites the plain, no-extra-argument
            # case into the equivalent axis reduction; anything it can't
            # safely rewrite (an explicit ``axis=``, say) keeps this loop
            # carried instead of risking a shape/value miscompile.
            if any(row_reduce_target(n, i) is not None and row_reduce_rewrite(n, i) is None for n in ast.walk(s.value)):
                return LoopKind.FORI
            if i in names_loaded(devectorize_index(s.value, i)):
                return LoopKind.FORI
        else:
            return LoopKind.FORI
    # also: a var written as a plain Name and read => carried
    for s in node.body:
        if isinstance(s.targets[0], ast.Name) and s.targets[0].id in stored:
            return LoopKind.FORI
    # A whole-array rebind is correct only if the loop writes EVERY index; a partial range
    # (explicit start, explicit step, or an offset stop like N-1) leaves some elements
    # untouched, so vectorising would clobber them -- keep the index-preserving fori form.
    if not range_covers_full_extent(node.iter):
        return LoopKind.FORI
    return LoopKind.VECTORIZE


def range_covers_full_extent(it: ast.AST) -> bool:
    """True for a ``range(stop)`` that spans an array's whole first axis: a single argument
    (start 0, step 1) that is a bare size symbol, ``len(x)``, ``x.shape[k]``, or a literal.
    Two+ args (explicit start/step) or an arithmetic stop (``N-1``, ``N//2``) are partial."""
    if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"):
        return False
    if len(it.args) != 1 or it.keywords:
        return False
    arg = it.args[0]
    if isinstance(arg, ast.Name):
        return True
    if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
        return True
    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "len":
        return True
    if isinstance(arg, ast.Attribute) and arg.attr == "shape":
        return True
    return isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Attribute) and arg.value.attr == "shape"


def is_index_i(sl: ast.AST, i: str) -> bool:
    return isinstance(sl, ast.Name) and sl.id == i


def carried_vars(body: list[ast.stmt], extra_live: set[str], cond_names: set[str] = frozenset()) -> list[str]:
    """Variables that genuinely thread across iterations.

    Carried iff read-before-written in the body (a cross-iteration dependency,
    e.g. ``trace += ...``) or written and live after the loop (``extra_live``).
    Written-before-read and not live-out is a loop-local temp (gramschmidt's
    ``nrm``) and must NOT be threaded, else the carry-tuple init references it
    before it exists.

    ``cond_names``: names read in the loop's own condition, evaluated before
    the body each iteration -- any the body writes is a genuine carry
    (channel_flow's ``while udiff > .001``); otherwise ``_cond`` would close
    over the pre-loop value and the loop would never terminate.
    """
    stored = names_stored(ast.Module(body=body, type_ignores=[]))
    carried: set[str] = set()
    written: set[str] = set()

    def cond_reads(names) -> None:
        # A condition read before write is a genuine cross-iteration carry
        # (s318's ``if v > maxv`` reads ``maxv`` before the branch updates it).
        for nm in names:
            if nm in stored and nm not in written:
                carried.add(nm)

    cond_reads(cond_names)  # the loop's own test is evaluated before the body

    def walk(stmts) -> None:
        # Recurse into compound stmts so a temp written-then-read inside an
        # if/loop (scattering's dHG/dHD) reads as local, not carried.
        for s in stmts:
            if isinstance(s, (ast.For, ast.While)):
                cond_reads(names_loaded(s.iter if isinstance(s, ast.For) else s.test))
                walk(s.body)
            elif isinstance(s, ast.If):
                # A write in ONE branch only is conditional, not definite -- the
                # other path keeps the prior (cross-iteration) value, so an
                # outer read must treat it as carried. Only a write on BOTH
                # branches is definite (s258: ``if a[i]>0: s=d[i]*d[i]`` then
                # ``b[i]=s*c[i]+d[i]`` -- ``s`` persists when the guard is false).
                cond_reads(names_loaded(s.test))
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
                for nm in stmt_rhs_loads(s):
                    if nm in stored and nm not in written:
                        carried.add(nm)  # read before write -> cross-iteration
                written.update(names_stored(s))

    walk(body)
    for nm in stored:
        if nm in extra_live:
            carried.add(nm)  # value escapes the loop
    return sorted(carried)


# In-place -> functional rewrite
def functionalize_stmt(s: ast.stmt) -> list[ast.stmt]:
    """Rewrite an in-place statement into a functional rebind: ``x <op>= v``
    -> ``x = x <op> v`` (re-processed, so ``A[i] += v`` flows into ``.at``);
    ``a[idx] = v`` -> ``a = a.at[idx].set(v)``; ``a[:] = v`` -> ``a = v``.
    Other statements pass through unchanged."""
    if isinstance(s, ast.AugAssign):
        assign = ast.Assign(targets=[s.target], value=ast.BinOp(left=load(s.target), op=s.op, right=s.value))
        return functionalize_stmt(ast.copy_location(assign, s))
    # arr.shape = newshape is numpy's in-place reshape; jax is immutable ->
    # arr = arr.reshape(newshape).
    if (
        isinstance(s, ast.Assign)
        and len(s.targets) == 1
        and isinstance(s.targets[0], ast.Attribute)
        and s.targets[0].attr == "shape"
        and isinstance(s.targets[0].value, ast.Name)
    ):
        name = s.targets[0].value
        call = ast.Call(
            func=ast.Attribute(value=ast.Name(id=name.id, ctx=ast.Load()), attr="reshape", ctx=ast.Load()),
            args=[s.value],
            keywords=[],
        )
        new = ast.Assign(targets=[ast.Name(id=name.id, ctx=ast.Store())], value=call)
        return [ast.copy_location(new, s)]
    if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Subscript):
        tgt = s.targets[0]
        # Flatten a chained target ``a[i][j]`` into one multi-axis index
        # ``a[i, j]`` (numpy-equivalent for basic indices): a naive
        # ``a[i].at[j].set(v)`` would rebind ``a`` to just the row ``a[i]``.
        indices: list[ast.expr] = []
        base = tgt
        while isinstance(base, ast.Subscript):
            indices.append(base.slice)
            base = base.value
        indices.reverse()
        arr = base
        arr_name = base_name(arr)
        sl = indices[0] if len(indices) == 1 else ast.Tuple(elts=indices, ctx=ast.Load())
        name = ast.Name(id=arr_name, ctx=ast.Store())
        if is_full_slice(sl):
            # a[:] = <scalar> fills every element; a plain a = <scalar> would
            # rebind a to a SCALAR. jnp.full_like keeps a's shape/dtype
            # (edge_laplacian's ``Lx[:] = 0.0``).
            if isinstance(s.value, ast.Constant) and not isinstance(s.value.value, str):
                fill = ast.Call(
                    func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="full_like", ctx=ast.Load()),
                    args=[ast.Name(id=arr_name, ctx=ast.Load()), s.value],
                    keywords=[],
                )
                new = ast.Assign(targets=[name], value=fill)
            else:
                # a[:] = <expr> broadcasts RHS to a's shape and casts to a's
                # dtype; a plain rebind would instead inherit the RHS's
                # shape/dtype, silently changing the output buffer.
                new = ast.Assign(targets=[name], value=broadcast_astype(arr, s.value))
        else:
            at = ast.Subscript(value=ast.Attribute(value=arr, attr="at", ctx=ast.Load()), slice=sl, ctx=ast.Load())
            call = ast.Call(func=ast.Attribute(value=at, attr="set", ctx=ast.Load()), args=[s.value], keywords=[])
            new = ast.Assign(targets=[name], value=call)
        return [ast.copy_location(new, s)]
    return [s]


#: numpy unbuffered-scatter ufunc -> jax ``.at[idx].<method>`` name.
SCATTER_AT_METHOD = {"add": "add", "subtract": "add", "multiply": "multiply", "maximum": "max", "minimum": "min"}


def scatter_at_assign(call: ast.Call) -> ast.Assign | None:
    """``np.<ufunc>.at(target, idx[, vals])`` -> the jax rebind
    ``target = target.at[idx].<method>(vals)``. None when not that form.
    ``add.at`` is scatter-accumulate (edge_laplacian's ``np.add.at(Lx, src,
    flux)``); ``subtract.at`` maps to ``.add(-vals)`` (jax has no ``.subtract``)."""
    f = call.func
    if not (
        isinstance(f, ast.Attribute)
        and f.attr == "at"
        and isinstance(f.value, ast.Attribute)
        and isinstance(f.value.value, ast.Name)
        and f.value.value.id in ("np", "numpy")
        and f.value.attr in SCATTER_AT_METHOD
        and len(call.args) >= 2
    ):
        return None
    op = f.value.attr
    target, idx = call.args[0], call.args[1]
    vals: ast.expr = call.args[2] if len(call.args) > 2 else ast.Constant(value=1)
    if op == "subtract":
        vals = ast.UnaryOp(op=ast.USub(), operand=vals)
    at = ast.Subscript(value=ast.Attribute(value=target, attr="at", ctx=ast.Load()), slice=idx, ctx=ast.Load())
    rebind = ast.Call(
        func=ast.Attribute(value=at, attr=SCATTER_AT_METHOD[op], ctx=ast.Load()), args=[vals], keywords=[]
    )
    return ast.copy_location(ast.Assign(targets=[ast.Name(id=base_name(target), ctx=ast.Store())], value=rebind), call)


def broadcast_astype(arr: ast.AST, value: ast.expr) -> ast.Call:
    """``jnp.broadcast_to(value, arr.shape).astype(arr.dtype)`` -- faithful
    lowering of ``arr[:] = value`` (broadcasts + casts to ``arr``'s shape/dtype,
    inferred from the live array, never hardcoded)."""
    name = base_name(arr)
    shape = ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr="shape", ctx=ast.Load())
    bcast = ast.Call(
        func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="broadcast_to", ctx=ast.Load()),
        args=[value, shape],
        keywords=[],
    )
    dtype = ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr="dtype", ctx=ast.Load())
    return ast.Call(func=ast.Attribute(value=bcast, attr="astype", ctx=ast.Load()), args=[dtype], keywords=[])


def emit_body(body: list[ast.stmt], live_out: set[str], indent: str, defined: set[str] = frozenset()) -> list[str]:
    """Emit a straight-line/looped statement list to JAX source lines.

    ``defined``: names already bound entering this body (params, or the loop
    carry tuple); grows as statements are emitted and is passed to each loop
    so it can verify its carry-tuple init references only bound names (see
    :func:`emit_for`)."""
    lines: list[str] = []
    cur = set(defined)
    for k, s in enumerate(body):
        if k:
            # Any name STORED by the preceding statement (incl. inside a
            # branch/loop) counts as bound -- over-approximates safely, but
            # still catches a loop threading a temp nothing prior writes.
            cur |= names_stored(ast.Module(body=[body[k - 1]], type_ignores=[]))
        if isinstance(s, (ast.For, ast.While, ast.If)):
            # A var assigned here and read by a later statement is live past
            # it, so the construct must thread it out (contour_integral's
            # ``if ..: X = -X`` then ``P0 += X``; s318's argmax read after the
            # loop) -- fold rest-of-body reads into live_out for all three.
            rest = upward_exposed(body[k + 1 :])
            if isinstance(s, ast.For):
                lines += emit_for(s, live_out | rest, indent, cur)
            elif isinstance(s, ast.While):
                lines += emit_while(s, live_out | rest, indent, cur)
            else:
                lines += emit_if(s, live_out | rest, indent, cur)
        elif isinstance(s, ast.Return):
            lines.append(indent + unparse_jnp(s))
        elif isinstance(s, (ast.Assign, ast.AugAssign)):
            for fs in functionalize_stmt(s):
                lines.append(indent + unparse_jnp(fs))
        elif isinstance(s, (ast.Import, ast.ImportFrom, ast.Pass, ast.Raise, ast.Assert)):
            continue  # input-validation guards never fire on oracle-valid inputs
        elif isinstance(s, ast.FunctionDef):
            # Nested helper def (velocity_tendencies' ``gat``) -- emit as a
            # real nested function (np->jnp applied); stays in scope after.
            #
            # ast.unparse over the whole ``arguments`` node, not a join of parameter NAMES: the
            # names alone drop every default, and vexx_k's ``def fwfft(col, batch=None)`` then
            # refuses its own one-argument call site with "missing 1 required positional argument".
            # It also carries keyword-only and positional-only markers, which a name join cannot
            # spell at all. Defaults are unparsed from the already np->jnp-rewritten tree, so an
            # expression default is rewritten like any other.
            lines.append(f"{indent}def {s.name}({ast.unparse(s.args)}):")
            inner = emit_body(s.body, set(), indent + "    ", {a.arg for a in s.args.args})
            lines += inner if inner else [indent + "    pass"]
        elif isinstance(s, ast.Expr):
            # A docstring/constant is a no-op; a bare call like
            # np.multiply(Z, Z, Z) has effects we can't safely drop.
            if isinstance(s.value, ast.Constant):
                continue
            sc = scatter_at_assign(s.value) if isinstance(s.value, ast.Call) else None
            if sc is not None:  # np.add.at(...) -> a = a.at[idx].add(...)
                lines.append(indent + unparse_jnp(sc))
                continue
            raise EmitError("bare expression statement (possible in-place op)")
        else:
            raise EmitError(f"unsupported statement: {type(s).__name__}")
    return lines


def is_static_iterable(node: ast.AST) -> bool:
    """Is ``node`` a compile-time-constant iterable a jit trace can unroll: a
    literal tuple/list, a module constant sequence, or ``enumerate``/``zip``/
    ``reversed`` over such (ls3df's ``for m, w in enumerate(_CW, start=1)``)?
    Every free name must be a static jit arg or module constant."""
    static = STATE.emit_static | STATE.module_consts | STATE.local_consts
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(names_loaded(e) <= static for e in node.elts)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("enumerate", "zip", "reversed")
    ):
        return all(is_static_iterable(a) or names_loaded(a) <= static for a in node.args)
    if isinstance(node, ast.Name):
        return node.id in STATE.module_consts or node.id in STATE.local_consts
    return False


def is_const_literal(node: ast.AST) -> bool:
    """A compile-time constant literal: a ``Constant``, a signed constant, or a
    ``list``/``tuple`` nesting of such (lulesh's ``[(0, 1, 2, 3), (4, 5, 6, 7),
    ...]`` face-index table)."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple)):
        return all(is_const_literal(e) for e in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return is_const_literal(node.operand)
    return False


def local_const_seq_names(fn: ast.FunctionDef) -> set[str]:
    """Function-local names bound EXACTLY ONCE to a constant-literal list/tuple
    (lulesh's ``faces = [(0, 1, 2, 3), ...]``). Such a name is a concrete Python
    sequence in the emitted function, so a ``for ... in <name>:`` over it unrolls
    at trace time (see :func:`is_static_iterable`)."""
    counts: dict = {}
    literal: set[str] = set()
    for s in ast.walk(fn):
        if isinstance(s, ast.Assign):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    counts[t.id] = counts.get(t.id, 0) + 1
                    if len(s.targets) == 1 and isinstance(s.value, (ast.List, ast.Tuple)) and is_const_literal(s.value):
                        literal.add(t.id)
    return {n for n in literal if counts.get(n) == 1}


def emit_if(node: ast.If, live_out: set[str], indent: str, defined: set[str] = frozenset()) -> list[str]:
    """Lower an ``if`` to ``jnp.where`` selects, or keep a real Python branch
    when the condition is static (contour_integral's ``if NR == NM`` picks
    ``inv`` vs ``solve``, whose incompatible-shape branches can't ``where``-merge).

    * static condition      -> emitted ``if``/``else`` verbatim.
    * ``if c: return A`` .. -> ``return jnp.where(c, A, B)``.
    * otherwise             -> snapshot/restore/``where``-select per var.
    """
    if names_loaded(node.test) <= STATE.emit_static or is_identity_test(node.test):
        cond = unparse_jnp(node.test)
        lines = [f"{indent}if {cond}:"]
        lines += emit_body(node.body, live_out, indent + "    ", defined) or [f"{indent}    pass"]
        if node.orelse:
            lines.append(f"{indent}else:")
            lines += emit_body(node.orelse, live_out, indent + "    ", defined) or [f"{indent}    pass"]
        return lines
    cond = cond_str(node.test)
    # if/else that simply returns -> a single selected return
    if is_return_only(node.body) and is_return_only(node.orelse):
        a = unparse_jnp(node.body[0].value)
        b = unparse_jnp(node.orelse[0].value)
        return [f"{indent}return jnp.where({cond}, {a}, {b})"]

    if any(isinstance(s, ast.Return) for s in node.body + node.orelse):
        raise EmitError("if-branch mixes return with assignments")
    assigned = names_stored(ast.Module(body=node.body + node.orelse, type_ignores=[]))
    # Only vars that escape the ``if`` (live after it) are snapshotted+selected;
    # branch-local temps (scattering's ``dHG``/``dHD``) are emitted plainly.
    select = sorted(v for v in assigned if v in live_out)
    if not select:
        # No escaping effect to gate -- emit the then-branch as-is (a no-effect else is a no-op).
        return emit_body(node.body, live_out, indent, defined)
    # Snapshot/restore temps are tagged per if-NODE, not depth: an if/elif
    # chain flattens to two If nodes at the SAME indent (elif is the outer
    # node's orelse), so a depth tag would collide and the outer write gets
    # dropped (ext_peel_multi_back). Source position is unique and stateless.
    tag = f"{node.lineno}_{node.col_offset}"
    pre = {v: f"_pre{tag}_{v}" for v in select}
    then = {v: f"_then{tag}_{v}" for v in select}
    # A snapshot ``_pre = v`` is needed only when v's INCOMING value is
    # observable: a branch reads v before writing it, or a branch doesn't
    # assign v at all. When both branches assign v and neither reads it
    # first, the incoming value is unused -- snapshotting would be dead and,
    # worse, if v is FIRST bound inside the branches, ``_pre = v`` raises
    # UnboundLocalError at trace time (contour_integral's ``if NR == NM``
    # choosing ``X = inv(Tz)`` / ``X = solve(Tz, Y)`` with no prior ``X``).
    body_stored = names_stored(ast.Module(body=node.body, type_ignores=[]))
    orelse_stored = names_stored(ast.Module(body=node.orelse, type_ignores=[]))

    def needs_pre(v: str) -> bool:
        if not (v in body_stored and v in orelse_stored):
            return True
        return reads_before_write(node.body, v) or reads_before_write(node.orelse, v)

    needs = {v: needs_pre(v) for v in select}
    # Capture the condition *before* the branches run -- they may overwrite the
    # very variables it tests (crc16's ``if crc&1 ^ ...: crc = crc>>1 ^ poly``).
    lines = [f"{indent}_cond{tag} = ({cond})"]
    lines += [f"{indent}{pre[v]} = {v}" for v in select if needs[v]]
    lines += emit_body(node.body, live_out, indent, defined)
    lines += [f"{indent}{then[v]} = {v}" for v in select]
    lines += [f"{indent}{v} = {pre[v]}" for v in select if needs[v]]
    if node.orelse:
        lines += emit_body(node.orelse, live_out, indent, defined | body_stored)
    lines += [f"{indent}{v} = jnp.where(_cond{tag}, {then[v]}, {v})" for v in select]
    return lines


def is_return_only(body: list[ast.stmt]) -> bool:
    return len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None


def split_on_break(body: list[ast.stmt]):
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
            return body[:k], s.test, s.body[:-1], body[k + 1 :]
    return body, None, [], []


def parse_range(rng: ast.Call):
    """``range`` -> ``(lo_expr, hi_expr, backward, stride)``.

    ``backward`` is True for a ``-1`` step (``range(a, b, -1)`` iterates a,
    a-1, .., b+1). ``stride`` is the positive step as source text (``"1"`` for
    unit step, ``"W"``/``"7"`` for a tiled ``range(1, N-1, W)``); a strided
    forward range recovers ``i = lo + _k * stride`` (see :func:`emit_for`)."""
    args = rng.args
    if len(args) == 1:
        return "0", unparse_jnp(args[0]), False, "1"
    if len(args) == 2:
        return unparse_jnp(args[0]), unparse_jnp(args[1]), False, "1"
    if len(args) == 3:
        step = args[2]
        if isinstance(step, ast.Constant) and step.value == 1:
            return unparse_jnp(args[0]), unparse_jnp(args[1]), False, "1"
        if (
            isinstance(step, ast.UnaryOp)
            and isinstance(step.op, ast.USub)
            and isinstance(step.operand, ast.Constant)
            and step.operand.value == 1
        ):
            return unparse_jnp(args[0]), unparse_jnp(args[1]), True, "1"
        # A negative constant step other than -1 is unsupported; anything
        # else (a positive constant or symbol like ``W``) is a forward stride.
        if isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub):
            raise EmitError("only -1 backward range() is supported")
        if isinstance(step, ast.Constant) and not (isinstance(step.value, int) and step.value > 0):
            raise EmitError("non-positive range() step is not supported")
        return unparse_jnp(args[0]), unparse_jnp(args[1]), False, unparse_jnp(step)
    raise EmitError("malformed range()")


def emit_for(node: ast.For, live_out: set[str], indent: str, defined: set[str] = frozenset()) -> list[str]:
    kind = classify_for(node)
    i = node.target.id if isinstance(node.target, ast.Name) else "_i"
    rng = node.iter
    if not (isinstance(rng, ast.Call) and isinstance(rng.func, ast.Name) and rng.func.id == "range"):
        return emit_iterable_for(node, live_out, indent, defined)
    lo, hi, backward, stride = parse_range(rng)

    # If the index feeds a shape (stockham_fft's ``reshape(y, (R**i, ..))``),
    # it must be concrete -- emit a real Python loop the tracer unrolls. Sound
    # only when the trip count is static AND the loop isn't a time-stepping
    # loop (those must stay rolled; the parallelism policy guarantees this).
    if (
        index_in_shape(node, i)
        and range_args_static(rng)
        and not backward
        and stride == "1"
        and not is_timestep_loop(node)
    ):
        lines = [f"{indent}for {i} in range({lo}, {hi}):"]
        lines += emit_body(node.body, live_out, indent + "    ", defined | {i}) or [f"{indent}    pass"]
        return lines

    if kind == LoopKind.VECTORIZE:
        return emit_vectorized(node, i, indent)

    carried = carried_vars(node.body, live_out)
    if not carried:
        raise EmitError("loop carries no observable state")
    # The carry-tuple init references each carried var by name, so every one
    # must be bound BEFORE the loop. A var that only LOOKS carried (cloudsc's
    # ``zqe``: an ``if nssopt==0: .. elif ..`` chain with no ``else`` reads as
    # a conditional write) would reference an unbound local -- raise so the
    # caller falls back to eager rather than emit a module that can't run.
    missing = [v for v in carried if v not in defined]
    if missing:
        raise EmitError(f"loop carry not defined before the loop: {', '.join(missing)}")
    if kind == LoopKind.FORI:
        return emit_fori(node, carried, (lo, hi, backward, stride), i, indent)
    # WHILE: range + break -> while_loop carrying the index + a done flag.
    if backward or stride != "1":
        raise EmitError("backward/strided range with break is not supported")
    return emit_while_break(node, carried, lo, hi, i, indent)


def emit_iterable_for(node: ast.For, live_out: set[str], indent: str, defined: set[str]) -> list[str]:
    """A loop over a non-``range`` iterable. A compile-time-constant iterable (ls3df's
    ``enumerate(_CW, start=1)``) emits as a literal Python for the tracer unrolls; a carried rebind
    just threads as a normal value. A literal tuple/list of Names (lulesh's face-corner
    ``for nk in (n0, n1, n2, n3):``) also unrolls -- only the LENGTH need be static -- unless a
    loop target feeds a shape, which needs a concrete value (eager runs the sequence directly)."""
    rng = node.iter
    literal_seq = isinstance(rng, (ast.Tuple, ast.List)) and all(
        isinstance(e, (ast.Name, ast.Constant)) or is_const_literal(e) for e in rng.elts
    )
    if not (is_static_iterable(rng) or literal_seq):
        raise EmitError("only `for i in range(...)` is supported")
    bound = {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
    if literal_seq and not is_static_iterable(rng) and any(index_in_shape(node, b) for b in bound):
        raise EmitError("literal-sequence loop index feeds a shape (needs concrete unroll)")
    lines = [f"{indent}for {unparse_jnp(node.target)} in {unparse_jnp(rng)}:"]
    lines += emit_body(node.body, live_out, indent + "    ", defined | bound) or [f"{indent}    pass"]
    return lines


def emit_vectorized(node: ast.For, i: str, indent: str) -> list[str]:
    """``a[i] = f(b[i], ...)`` -> ``a = f(b, ...)`` (drop the ``[i]`` indexing)."""
    out = []
    for s in node.body:
        t = s.targets[0]
        arr = t.value.id
        rhs = devectorize_index(s.value, i)
        # A RHS reading `i` (only inside ``x[i]`` subscripts) devectorises
        # to a full-array expr, so ``a = rhs`` is shape-correct. A
        # loop-invariant RHS (``a[i] = 0.0``) has no `i`, so ``a = rhs``
        # would collapse `a` to that scalar -- broadcast-fill instead (s293).
        if i in names_loaded(s.value):
            out.append(f"{indent}{arr} = {unparse_jnp(rhs)}")
        else:
            out.append(f"{indent}{arr} = jnp.full_like({arr}, {unparse_jnp(rhs)})")
    return out


def emit_fori(node: ast.For, carried: list[str], rng: tuple[str, str, bool, str], i: str, indent: str) -> list[str]:
    """``lax.fori_loop`` carrying ``carried``. A unit forward step drives the index directly; a
    backward (-1) or forward-strided (tiled ``range(1, N-1, W)``) range drives a counter ``_k`` over
    ``[0, trip)`` and recovers the real index (``lo - _k`` / ``lo + _k*s``)."""
    lo, hi, backward, stride = rng
    inner = indent + "    "
    st = tuple_expr(carried)
    if backward:
        ctr, lo2, hi2 = "_k", "0", f"({lo}) - ({hi})"
        recover = f"{inner}{i} = ({lo}) - _k"
    elif stride != "1":
        ctr, lo2, hi2 = "_k", "0", f"(({hi}) - ({lo}) + ({stride}) - 1) // ({stride})"
        recover = f"{inner}{i} = ({lo}) + _k * ({stride})"
    else:
        ctr, lo2, hi2, recover = i, lo, hi, None
    body_inner = emit_body(node.body, set(carried), inner, set(carried) | {i})
    lines = [f"{indent}def _body({ctr}, _c):", f"{inner}{st} = _c"]
    if recover:
        lines.append(recover)
    lines += body_inner
    lines += [f"{inner}return {st}", f"{indent}{st} = lax.fori_loop({lo2}, {hi2}, _body, {st})"]
    return lines


def expand_parallel_assigns(stmts: list[ast.stmt]) -> list[ast.stmt]:
    """Split a parallel Name-tuple assign ``a, b = e0, e1`` into ``__pa = (e0,
    e1); a = __pa[0]; b = __pa[1]`` so each target becomes a single Name rebind
    (ls3df's Lanczos ``v_prev, v = v, w / beta``). The temp snapshots the whole
    RHS first, preserving numpy's simultaneous-assignment semantics under the
    break-guard's per-var ``jnp.where`` freeze."""
    out: list[ast.stmt] = []
    for s in stmts:
        if (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], (ast.Tuple, ast.List))
            and all(isinstance(e, ast.Name) for e in s.targets[0].elts)
        ):
            STATE.tuple_ctr += 1
            tmp = f"__pa{STATE.tuple_ctr}"
            out.append(ast.Assign(targets=[ast.Name(id=tmp, ctx=ast.Store())], value=s.value))
            for k, e in enumerate(s.targets[0].elts):
                item = ast.Subscript(
                    value=ast.Name(id=tmp, ctx=ast.Load()), slice=ast.Constant(value=k), ctx=ast.Load()
                )
                out.append(ast.Assign(targets=[ast.Name(id=e.id, ctx=ast.Store())], value=item))
        else:
            out.append(s)
    return [ast.fix_missing_locations(x) for x in out]


def emit_while_break(node, carried, lo, hi, i, indent):
    before, cond, on_break, after = split_on_break(node.body)
    if cond is None:
        raise EmitError("break loop without an `if cond: ... break` guard")
    on_break = expand_parallel_assigns(on_break)
    after = expand_parallel_assigns(after)
    full = [i] + carried + ["_done"]
    inner = indent + "    "
    st = tuple_expr(full)
    lines = [
        f"{indent}def _cond(_c):",
        f"{inner}{st} = _c",
        f"{inner}return ({i} < {hi}) & jnp.logical_not(_done)",
        f"{indent}def _body(_c):",
        f"{inner}{st} = _c",
    ]
    lines += emit_body(before, set(carried), inner, set(carried) | {i})
    lines.append(f"{inner}_conv = ({cond_str(cond)})")
    cset = set(carried)

    def frozen(stmts, when_conv) -> None:
        # Freeze a carried var with jnp.where so it updates only on the
        # intended branch; a local temp (minres's ``beta``) emits plainly.
        # ``when_conv``: the in-guard capture takes the new value WHEN
        # converged; post-guard statements keep the OLD value when converged.
        for s in stmts:
            for fs in functionalize_stmt(s):
                if not (isinstance(fs, ast.Assign) and len(fs.targets) == 1 and isinstance(fs.targets[0], ast.Name)):
                    raise EmitError("break-guard statement is not a simple rebind")
                tgt = fs.targets[0].id
                if tgt not in cset:
                    lines.append(f"{inner}{unparse_jnp(fs)}")
                elif when_conv:
                    lines.append(f"{inner}{tgt} = jnp.where(_conv, {unparse_jnp(fs.value)}, {tgt})")
                else:
                    lines.append(f"{inner}{tgt} = jnp.where(_conv, {tgt}, {unparse_jnp(fs.value)})")

    # The capture (``index = i``) commits on the converging iteration; the
    # post-guard update (cg/minres's next-iterate maths) is skipped on it.
    frozen(on_break, when_conv=True)
    frozen(after, when_conv=False)
    ret = "(" + ", ".join([f"{i} + 1"] + carried + ["_conv | _done"]) + ",)"
    init = "(" + ", ".join([lo] + carried + ["jnp.bool_(False)"]) + ",)"
    lines += [f"{inner}return {ret}", f"{indent}{st} = lax.while_loop(_cond, _body, {init})"]
    return lines


def emit_while(node: ast.While, live_out: set[str], indent: str, defined: set[str] = frozenset()) -> list[str]:
    carried = carried_vars(node.body, live_out, names_loaded(node.test))
    if not carried:
        raise EmitError("while-loop carries no observable state")
    missing = [v for v in carried if v not in defined]
    if missing:
        raise EmitError(f"while carry not defined before the loop: {', '.join(missing)}")
    inner = indent + "    "
    st = tuple_expr(carried)
    lines = [
        f"{indent}def _cond(_c):",
        f"{inner}{st} = _c",
        f"{inner}return ({cond_str(node.test)})",
        f"{indent}def _body(_c):",
        f"{inner}{st} = _c",
    ]
    lines += emit_body(node.body, set(carried), inner, set(carried))
    lines += [f"{inner}return {st}", f"{indent}{st} = lax.while_loop(_cond, _body, {st})"]
    return lines


#: Whole-array reductions confirmed rank-reducing over an indexed row: ``f(a[i])``
#: collapses ALL of the row's axes, not just the batch axis ``i`` -- naively
#: dropping ``[i]`` (as for an elementwise read) turns a per-row VALUE into a
#: full-array SCALAR. See ``row_reduce_rewrite``.
ROW_REDUCE_FUNCS = frozenset({"sum", "max", "min", "mean", "prod"})


def row_reduce_target(node: ast.AST, i: str) -> ast.expr | None:
    """If ``node`` is a call to one of ``ROW_REDUCE_FUNCS`` -- module form
    ``np.f(a[i])``/``jnp.f(a[i])`` or method form ``a[i].f()`` -- whose reduced
    row is exactly the bare-index subscript ``a[i]``, return the base array
    Name (``a``). None otherwise (including a non-matching function/shape)."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if (
        isinstance(node.func.value, ast.Name)
        and node.func.value.id in ("np", "jnp")
        and node.func.attr in ROW_REDUCE_FUNCS
        and len(node.args) >= 1
    ):
        arg = node.args[0]
    elif node.func.attr in ROW_REDUCE_FUNCS:
        arg = node.func.value
    else:
        return None
    if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name) and is_index_i(arg.slice, i):
        return arg.value
    return None


def row_reduce_rewrite(node: ast.AST, i: str) -> ast.expr | None:
    """Rewrite a plain row-reduction found by ``row_reduce_target`` into the
    equivalent whole-array axis reduction ``f(a, axis=tuple(range(1, a.ndim)))``
    -- exactly the per-row scalar for ANY rank of ``a`` (an empty axis tuple at
    rank 1 is a no-op, matching ``f(scalar) == scalar``). None when an extra
    positional/keyword argument is present (an explicit ``axis=``, say) --
    guessing how that interacts with the added batch axis would be guesswork,
    so the caller refuses to vectorise instead of risking another miscompile."""
    base = row_reduce_target(node, i)
    if base is None:
        return None
    is_method = not (isinstance(node.func.value, ast.Name) and node.func.value.id in ("np", "jnp"))
    extra_args = node.args if is_method else node.args[1:]
    if extra_args or node.keywords:
        return None
    axis = ast.Call(
        func=ast.Name(id="tuple", ctx=ast.Load()),
        args=[
            ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[ast.Constant(value=1), ast.Attribute(value=copy.deepcopy(base), attr="ndim", ctx=ast.Load())],
                keywords=[],
            )
        ],
        keywords=[],
    )
    call = ast.Call(
        func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=node.func.attr, ctx=ast.Load()),
        args=[copy.deepcopy(base)],
        keywords=[ast.keyword(arg="axis", value=axis)],
    )
    return ast.copy_location(call, node)


def devectorize_index(node: ast.AST, i: str) -> ast.AST:
    """Drop ``[i]`` subscripts so an independent elementwise loop body becomes
    a whole-array expression. A row-reduction (``np.sum(a[i])``) is rewritten
    to the equivalent axis reduction first (``row_reduce_rewrite``) -- a bare
    subscript-strip alone would collapse it to a full-array scalar."""

    class Rewriter(ast.NodeTransformer):
        def visit_Call(self, n: ast.Call) -> ast.expr:
            rewritten = row_reduce_rewrite(n, i)
            if rewritten is not None:
                return rewritten
            self.generic_visit(n)
            return n

        def visit_Subscript(self, n: ast.Subscript) -> ast.expr:
            self.generic_visit(n)
            if is_index_i(n.slice, i):
                return n.value
            return n

    return Rewriter().visit(ast.fix_missing_locations(ast.parse(ast.unparse(node), mode="eval"))).body


def loop_vars(fn: ast.FunctionDef) -> set[str]:
    # Exclude indices of loops that will be unrolled: those become concrete
    # Python ints, so their ``:R**i``-style slices are static, not dynamic.
    return {
        n.target.id for n in ast.walk(fn) if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
    } - unroll_loop_vars(fn)


def unroll_loop_vars(fn: ast.FunctionDef) -> set[str]:
    """Indices of ``for i in range(STATIC)`` loops whose body uses ``i`` in a
    shape -- emitted as real Python loops the tracer unrolls (stockham_fft)."""
    out: set[str] = set()
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.For)
            and isinstance(n.target, ast.Name)
            and isinstance(n.iter, ast.Call)
            and isinstance(n.iter.func, ast.Name)
            and n.iter.func.id == "range"
            and index_in_shape(n, n.target.id)
            and range_args_static(n.iter)
        ):
            out.add(n.target.id)
    return out


def range_args_static(rng: ast.Call) -> bool:
    names: set[str] = set()
    for a in rng.args:
        names |= names_loaded(a)
    return names <= STATE.emit_static
