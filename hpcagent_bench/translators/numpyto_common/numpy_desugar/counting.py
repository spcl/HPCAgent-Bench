"""Counting and scatter forms: ``np.diff``, ``repeat``, ``bincount``, ``add.at``, ``searchsorted``, ``histogram``."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import (
    DesugarError,
    const_int,
    np_attr,
    replace_call_with_name,
)
from hpcagent_bench.translators.numpyto_common.numpy_desugar.hoist import HoistForm, ValueHoist
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import expr_rank
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ufuncs import ufunc_method_op


AT_OPS = {"add": "+=", "subtract": "-=", "multiply": "*="}


class DiffToSliceDifference(ast.NodeTransformer):
    """``np.diff(p)`` -> ``p[1:] - p[:-1]``.

    The identity numpy documents, and every python backend traces the slice form natively -- dace
    otherwise falls back to a Python callback for the whole enclosing program, which is not a
    lowering at all. Only the single-argument form: an ``n``/``axis``/``prepend`` argument is a
    different computation, left standing for the caller to see.
    """

    def __init__(self) -> None:
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if np_attr(node) != "diff" or len(node.args) != 1 or node.keywords:
            return node
        base = node.args[0]
        self.changed = True
        tail = ast.Subscript(
            value=copy.deepcopy(base),
            slice=ast.Slice(lower=ast.Constant(value=1), upper=None, step=None),
            ctx=ast.Load(),
        )
        head = ast.Subscript(
            value=copy.deepcopy(base),
            slice=ast.Slice(lower=None, upper=ast.Constant(value=-1), step=None),
            ctx=ast.Load(),
        )
        return ast.copy_location(ast.BinOp(left=tail, op=ast.Sub(), right=head), node)


class StripAstypeCopyKwarg(ast.NodeTransformer):
    """``x.astype(dt, copy=False)`` -> ``x.astype(dt)``.

    ``copy`` is a hint about whether numpy may return the input unchanged when the dtype already
    matches; the VALUE is the same either way. dace's astype replacement takes no such argument and
    fails the build outright (``_ndarray_astype() got an unexpected keyword argument 'copy'``).
    """

    def __init__(self) -> None:
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "astype"
            and any(k.arg == "copy" for k in node.keywords)
        ):
            node.keywords = [k for k in node.keywords if k.arg != "copy"]
            self.changed = True
        return node


class RepeatCountsInline(ast.NodeTransformer):
    """``np.repeat(src, np.diff(p))`` -> a prefix-sum fill loop.

    A PER-ELEMENT repeat count is a different computation from the scalar one: the destination
    offset is the running sum of the counts, not ``i * K``. numba and dace have no array-count
    repeat, so the CSR row-index idiom fell back to a callback.

    The output length is ``sum(counts)``, which is data -- except for a first difference, where it
    TELESCOPES to ``p[-1] - p[0]``. That is the only form claimed here; any other per-element count
    is left standing rather than sized by a guess.
    """

    def __init__(self, ranks: dict[str, int]) -> None:
        self.ranks = ranks
        self.changed = False
        self._ctr = 0

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        calls = [n for n in ast.walk(node.value) if np_attr(n) == "repeat" and len(n.args) == 2 and not n.keywords]
        pre: list[ast.stmt] = []
        for call in calls:
            counts = call.args[1]
            if np_attr(counts) != "diff" or len(counts.args) != 1:
                continue  # scalar count, or a sum we cannot derive -- not ours
            if (expr_rank(call.args[1], self.ranks) or 0) < 1 and not isinstance(counts, ast.Call):
                continue
            ptr = ast.unparse(counts.args[0])
            name = f"__rp{self._ctr}"
            self._ctr += 1
            src, i, r, pos = f"{name}_src", f"{name}_i", f"{name}_r", f"{name}_pos"
            lines = [
                f"{src} = {ast.unparse(call.args[0])}",
                f"{name} = np.zeros({ptr}[-1] - {ptr}[0], dtype={src}.dtype)",
                f"{pos} = 0",
                f"for {i} in range({src}.shape[0]):",
                f"    for {r} in range({ptr}[{i} + 1] - {ptr}[{i}]):",
                f"        {name}[{pos}] = {src}[{i}]",
                f"        {pos} = {pos} + 1",
            ]
            pre.extend(ast.parse("\n".join(lines)).body)
            replace_call_with_name(node, call, name)
        if not pre:
            return node
        self.changed = True
        out = pre + [node]
        for stmt in out:
            ast.copy_location(stmt, node)
            ast.fix_missing_locations(stmt)
        return out


class BincountInline(ast.NodeTransformer):
    """``np.bincount(idx, weights=w, minlength=M)`` -> zero M slots, then a scatter-add loop.

    No python backend implements it: numba has no bincount at all, and dace routes it to a Python
    callback that drags the whole program back into the interpreter. The loop is the definition --
    duplicate indices ACCUMULATE, which is why it is ``+=`` and not a store.

    ``minlength`` is required. numpy sizes the result ``max(minlength, idx.max() + 1)`` and the
    second term is data; every corpus caller assigns the result into a buffer of exactly M, so an
    index at or past M would make numpy return a LONGER array and the assignment itself would raise.
    """

    def __init__(self, ranks: dict[str, int]) -> None:
        self.ranks = ranks
        self.changed = False
        self._ctr = 0

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        calls = [n for n in ast.walk(node.value) if np_attr(n) == "bincount" and n.args]
        if not calls:
            return node
        pre: list[ast.stmt] = []
        for call in calls:
            kw = {k.arg: k.value for k in call.keywords}
            minlength = kw.get("minlength") or (call.args[2] if len(call.args) > 2 else None)
            weights = kw.get("weights") or (call.args[1] if len(call.args) > 1 else None)
            if minlength is None:
                return node  # data-dependent extent -- leave it standing rather than guess one
            name = f"__bc{self._ctr}"
            self._ctr += 1
            idx, it = ast.unparse(call.args[0]), f"{name}_i"
            dtype = f"{ast.unparse(weights)}.dtype" if weights is not None else "np.int64"
            rhs = f"{ast.unparse(weights)}[{it}]" if weights is not None else "1"
            lines = [
                f"{name} = np.zeros({ast.unparse(minlength)}, dtype={dtype})",
                f"for {it} in range({idx}.shape[0]):",
                f"    {name}[{idx}[{it}]] += {rhs}",
            ]
            pre.extend(ast.parse("\n".join(lines)).body)
            call.func = ast.Name(id="__bincount_result__", ctx=ast.Load())
            replace_call_with_name(node, call, name)
        self.changed = True
        out = pre + [node]
        for stmt in out:
            ast.copy_location(stmt, node)
            ast.fix_missing_locations(stmt)
        return out


class AddAtInline(ast.NodeTransformer):
    """``np.add.at(A, idx, vals)`` -> an explicit scatter loop. numba has no
    ufunc.at; a sequential ``+=`` loop reproduces its defining property --
    duplicate indices accumulate (unlike ``A[idx] += vals``). Handles a single
    1-D index (edge_laplacian's ``np.add.at(Lx, src, flux)``) and a tuple of
    index arrays + scalar axes (icon_scatter's ``np.add.at(out, (i2d, jk, j2d),
    val)``); the driver is the first index array, scalars ride each iteration."""

    def __init__(self, ranks: dict[str, int]) -> None:
        self.ranks = ranks
        self.changed = False
        self._ctr = 0

    def visit_Expr(self, node: ast.Expr):
        self.generic_visit(node)
        call = node.value
        op = ufunc_method_op(call, "at") if isinstance(call, ast.Call) else None
        if op not in AT_OPS or len(call.args) < 2 or not isinstance(call.args[0], ast.Name):
            return node
        A = call.args[0].id
        elts = call.args[1].elts if isinstance(call.args[1], ast.Tuple) else [call.args[1]]
        vals = call.args[2] if len(call.args) > 2 else None
        driver_rank = next((r for e in elts if (r := expr_rank(e, self.ranks)) and r >= 1), None)
        if driver_rank is None:
            return node
        p = f"__sc{self._ctr}"
        self._ctr += 1
        iters = [f"{p}_i{k}" for k in range(driver_rank)]
        it = ", ".join(iters)
        # ``np.ascontiguousarray`` materialises each hoisted index / value array:
        # pythran keeps ``nbr_idx[:, :, n] - 1`` as a lazy numpy_expr that cannot
        # be indexed by a tuple in the scatter loop (icon_scatter); it is a no-op
        # for an already-contiguous array under numba.
        pre, idx_exprs, first_arr = [], [], None
        for j, e in enumerate(elts):
            if (expr_rank(e, self.ranks) or 0) >= 1:
                t = f"{p}_x{j}"
                pre.append(f"{t} = np.ascontiguousarray({ast.unparse(e)})")
                idx_exprs.append(f"{t}[{it}]")
                first_arr = first_arr or t
            else:
                idx_exprs.append(ast.unparse(e))
        trail_iters: list[str] = []
        if vals is None:
            rhs = "1"
        else:
            tv = f"{p}_v"
            vr = expr_rank(vals, self.ranks)
            if vr == 0:
                # A SCALAR value stays a scalar: ``np.ascontiguousarray(1)`` is a 0-d ARRAY, and
                # pythran then has no ``double += 0-d array``. The materialisation exists for a lazy
                # numpy_expr operand, which a scalar is not.
                tv = ast.unparse(vals)
            else:
                pre.append(f"{tv} = np.ascontiguousarray({ast.unparse(vals)})")
            # ``A[idx]`` keeps the axes the index tuple does NOT address, so a rank-1 index into a
            # rank-3 A selects rank-2 blocks and vals is legitimately rank 3 (cp2k_density_matrix_trs4's
            # ``np.add.at(c_blocks, flat_c_pos, alpha * flat_prod)``). Those trailing axes get their
            # own loops rather than a subarray ``+=``: scalar accumulation is what every backend
            # supports, and it keeps the unbuffered duplicate-index order this lowering exists for.
            trailing = vr - driver_rank if vr else 0
            a_rank = self.ranks.get(A)
            if vr and trailing and (trailing < 0 or a_rank is None or a_rank - len(elts) != trailing):
                # Anything else would need numpy broadcast alignment we do not model -> fail loudly.
                raise DesugarError(
                    f"np.add.at values ndim {vr} != index ndim {driver_rank} "
                    f"plus the {'unknown' if a_rank is None else a_rank - len(elts)} "
                    "axes the index leaves untouched (only scalar, matching-shape or "
                    "block-shaped values are lowered)"
                )
            trail_iters = [f"{p}_t{k}" for k in range(trailing)]
            idx_exprs.extend(trail_iters)
            rhs = f"{tv}[{', '.join(iters + trail_iters)}]" if vr and vr >= 1 else tv
        lines, deepen = list(pre), ""
        for k in range(driver_rank):
            lines.append(f"{deepen}for {iters[k]} in range({first_arr}.shape[{k}]):")
            deepen += "    "
        for k, t in enumerate(trail_iters):
            lines.append(f"{deepen}for {t} in range({tv}.shape[{driver_rank + k}]):")
            deepen += "    "
        lines.append(f"{deepen}{A}[{', '.join(idx_exprs)}] {AT_OPS[op]} {rhs}")
        self.changed = True
        return [ast.copy_location(s, node) for s in ast.parse("\n".join(lines)).body]


class SearchsortedMaterialize(ast.NodeTransformer):
    """``np.searchsorted(<expr>, v)`` -> ``np.searchsorted(np.ascontiguousarray(<expr>), v)``.

    pythran keeps ``rmax * np.arange(npt + 1) / npt`` as a lazy ``numpy_expr`` whose iterator is
    forward-only, and searchsorted needs to walk it backwards -- the g++ error is a missing
    ``operator--`` on a numpy_expr_iterator, several template layers deep and nowhere near the line
    that caused it. Materialising the sorted operand is a no-op for an array that is already
    contiguous, which is what the other backends see.
    """

    def __init__(self) -> None:
        self.changed = False
        self._ctr = 0

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        calls = [n for n in ast.walk(node.value) if isinstance(n, ast.Call) and np_attr(n) == "searchsorted" and n.args]
        if not calls:
            return node
        pre: list[ast.stmt] = []
        for call in calls:
            # A NAME is not enough: pythran binds a name to the lazy expression itself, so
            # ``edges = rmax * np.arange(n) / npt`` stays an unmaterialised numpy_expr at the use.
            tmp = f"__ss{self._ctr}"
            self._ctr += 1
            pre.append(
                ast.Assign(
                    targets=[ast.Name(id=tmp, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="np", ctx=ast.Load()), attr="ascontiguousarray", ctx=ast.Load()
                        ),
                        args=[call.args[0]],
                        keywords=[],
                    ),
                )
            )
            call.args[0] = ast.Name(id=tmp, ctx=ast.Load())
        self.changed = True
        out = pre + [node]
        for stmt in out:
            ast.copy_location(stmt, node)
            ast.fix_missing_locations(stmt)
        return out


def hoist_histogram(node: ast.AST, hoist: ValueHoist) -> ast.expr | None:
    """``np.histogram(a, bins[, lo, hi][, weights=w])[0]`` -> the temp its binning loop fills (azimint's ``histw =
    np.histogram(r, n, weights=d)[0]``): a min/max scan
    for the default range, the ``np.linspace(lo, hi, bins + 1)`` edge array, then per-element
    binning ``b = int((a-lo)*bins/(hi-lo))`` clamped to ``[0, bins-1]``, walked one step against
    those edges (numpy's own correction) and accumulating ``1`` (or ``w[i]``). numba has no
    np.histogram; this is the same loop the C/Fortran backends lower (azimint_hist)."""
    if not (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == 0):
        return None
    call = node.value
    if not (isinstance(call, ast.Call) and np_attr(call) == "histogram" and len(call.args) >= 2):
        return None
    a, bins = ast.unparse(call.args[0]), ast.unparse(call.args[1])
    kw = {k.arg: k.value for k in call.keywords}
    lo: ast.expr | None = None
    hi: ast.expr | None = None
    if len(call.args) >= 4:
        lo, hi = call.args[2], call.args[3]
    rng = kw.get("range")
    if isinstance(rng, ast.Tuple) and len(rng.elts) == 2:
        lo, hi = rng.elts
    weights = kw.get("weights")
    p = f"__hist{hoist.ctr}"
    hoist.ctr += 1
    lines: list[str] = []
    if lo is None or hi is None:
        lo_s, hi_s = f"{p}_lo", f"{p}_hi"
        lines += [
            f"{lo_s} = {a}[0]",
            f"{hi_s} = {a}[0]",
            f"for {p}_s in range({a}.shape[0]):",
            f"    if {a}[{p}_s] < {lo_s}: {lo_s} = {a}[{p}_s]",
            f"    if {a}[{p}_s] > {hi_s}: {hi_s} = {a}[{p}_s]",
        ]
    else:
        lo_s, hi_s = f"({ast.unparse(lo)})", f"({ast.unparse(hi)})"
    temp = f"{p}_o"
    # numpy's histogram is int64 COUNTS when unweighted and the weights' own dtype when
    # weighted -- never an unconditional float64, which both lies about the unweighted
    # result and narrows a float32 weighted one. A non-Name weights expression is hoisted
    # so the dtype can be read off a name rather than re-evaluating the expression.
    if weights is None:
        wdtype, add = "np.int64", "1"
    else:
        wname = weights.id if isinstance(weights, ast.Name) else f"{p}_w"
        if not isinstance(weights, ast.Name):
            lines.append(f"{wname} = {ast.unparse(weights)}")
        wdtype, add = f"{wname}.dtype", f"{wname}[{p}_i]"
    # numpy's bin is defined by its EDGE ARRAY, not by the closed form below: it truncates
    # the same index and then walks it one step against linspace's edges. The two round
    # apart, and without the walk 6 of azimint_hist's 400000 fp32 samples land one bin over
    # -- 0.2% on a bin ratio, past the fp32 band. So the edges are rebuilt with linspace's
    # own arithmetic and, crucially, in the SAMPLE dtype, one rounding per statement: half
    # an ulp of edge is worth ~20 misbinned samples at this count. Only edges 0..bins-1 are
    # read (the walk up stops at bins-1), hence ``bins`` entries and no top edge.
    edges = f"{p}_e"
    lines += [
        f"{edges} = np.zeros({bins}, {a}.dtype)",
        f"{edges}[0] = ({hi_s} - {lo_s}) / {bins}",
        f"{p}_st = {edges}[0]",
        f"for {p}_j in range({bins}):",
        f"    {edges}[{p}_j] = {p}_j * {p}_st",
        f"    {edges}[{p}_j] = {edges}[{p}_j] + {lo_s}",
    ]
    # numpy drops samples outside [lo, hi] (only the last bin is closed); the clamp alone
    # would fold them into bin 0 / bin-1 instead. Guard the increment. For an auto lo/hi
    # (a.min()/a.max()) every element is in range, so the guard is a no-op there.
    lines += [
        f"{temp} = np.zeros({bins}, {wdtype})",
        f"for {p}_i in range({a}.shape[0]):",
        f"    if {lo_s} <= {a}[{p}_i] and {a}[{p}_i] <= {hi_s}:",
        f"        {p}_b = int(({a}[{p}_i] - {lo_s}) * {bins} / ({hi_s} - {lo_s}))",
        f"        if {p}_b < 0: {p}_b = 0",
        f"        if {p}_b > {bins} - 1: {p}_b = {bins} - 1",
        f"        if {a}[{p}_i] < {edges}[{p}_b]: {p}_b = {p}_b - 1",
        f"        if {p}_b < {bins} - 1 and {a}[{p}_i] >= {edges}[{p}_b + 1]: {p}_b = {p}_b + 1",
        f"        {temp}[{p}_b] += {add}",
    ]
    hoist.queue(lines)
    return ast.Name(id=temp, ctx=ast.Load())


HISTOGRAM_HOIST = HoistForm(frozenset({"histogram"}), (), hoist_histogram)


def hoist_repeat_axis(node: ast.AST, hoist: ValueHoist) -> ast.expr | None:
    """``np.repeat(x, m, axis=k)`` (literal axis, scalar count) -> the temp its gather loop ``out[..., j, ...] =
    x[..., j // m, ...]`` fills (numpy repeats each slice ``m`` times consecutively along ``axis``). numba rejects the
    ``axis=`` kwarg on np.repeat (stockham's ``np.repeat(reshape(tmp, (R, R**i, 1)), R**(K-i-1), axis=2)``)."""
    if not isinstance(node, ast.Call) or np_attr(node) != "repeat" or len(node.args) < 2:
        return None
    kw = {k.arg: k.value for k in node.keywords}
    ax = kw.get("axis") or (node.args[2] if len(node.args) > 2 else None)
    axis = None if ax is None else const_int(ax)
    if axis is None:
        return None  # no axis / non-literal -> leave verbatim (a clean skip)
    x, m = node.args[0], node.args[1]
    rank = expr_rank(x, hoist.tables.ranks)
    if rank is None or not -rank <= axis < rank:
        # An axis outside the rank means the rank estimate is wrong; wrapping it into range
        # would repeat along a different axis, so leave the call verbatim (as :func:`axis_list`).
        return None
    k = axis % rank
    p = f"__rp{hoist.ctr}"
    hoist.ctr += 1
    pre = []
    if isinstance(x, ast.Name):
        xid = x.id
    else:
        xid = f"{p}_x"
        pre.append(f"{xid} = {ast.unparse(x)}")
    ms = f"({ast.unparse(m)})"
    dims = [(f"{xid}.shape[{d}] * {ms}" if d == k else f"{xid}.shape[{d}]") for d in range(rank)]
    iters = [f"{p}_i{d}" for d in range(rank)]
    out = f"{p}_o"
    lines = pre + [f"{out} = np.empty(({', '.join(dims)},), {xid}.dtype)"]
    deep = ""
    for d in range(rank):
        lines.append(f"{deep}for {iters[d]} in range({dims[d]}):")
        deep += "    "
    src_idx = ", ".join((f"{iters[k]} // {ms}" if d == k else iters[d]) for d in range(rank))
    lines.append(f"{deep}{out}[{', '.join(iters)}] = {xid}[{src_idx}]")
    hoist.queue(lines)
    return ast.Name(id=out, ctx=ast.Load())


REPEAT_AXIS_HOIST = HoistForm(frozenset({"repeat"}), (), hoist_repeat_axis)
