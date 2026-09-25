"""``np.linalg``: norm, lstsq, cholesky, solve, inv, det as self-contained loop algorithms."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.lib_nodes.call_args import read_axis_keepdims
from hpcagent_bench.translators.numpyto_common.lib_nodes.contractions import OP_SPILL_TEMP
from hpcagent_bench.translators.numpyto_common.lib_nodes.elementwise import args_one_name
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of_
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import (
    attr_call,
    const_,
    const_or_name,
    make_iter_name,
    name_,
    reads_complex,
    store_,
    wrap_for_loops,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.reductions import expand_axis_reduction
from hpcagent_bench.translators.numpyto_common.lib_nodes.scalarize import scalarize_at_iters


def classify_norm_ord(node: ast.expr | None) -> str | None:
    """Classify a ``np.linalg.norm`` ``ord`` argument.

    ``"l2"`` for the default (``None`` / 2 -> Euclidean vector norm or
    Frobenius matrix norm), ``"l1"`` for ``ord=1``, ``"inf"`` for ``np.inf``
    / ``math.inf`` / ``float("inf")``. Anything else (3, ``'nuc'``, ``'fro'``
    spelled out, ``-inf``, the matrix spectral 2-norm) -> ``None``, so the
    caller raises rather than emit a silently-wrong norm.
    """
    if node is None:
        return "l2"
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return None
        if v is None or v == 2:
            return "l2"
        if v == 1:
            return "l1"
        if isinstance(v, float) and v == float("inf"):
            return "inf"
        return None
    if isinstance(node, ast.Attribute) and node.attr in ("inf", "Inf", "PINF"):
        return "inf"
    # ``np.inf`` is rewritten to the bare Name ``INFINITY`` (the lowered
    # numeric-constant token, see lowering/mathfuncs.py) before this expander runs.
    if isinstance(node, ast.Name) and node.id in ("INFINITY", "inf", "Inf"):
        return "inf"
    return None


def expand_linalg_norm(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    """``s = np.linalg.norm(v[, ord, axis=None, keepdims=False])``. numpy puts
    ``ord`` SECOND (positional) -- unlike a reduction, whose second positional
    is ``axis`` -- so it's parsed explicitly before delegating axis/keepdims
    handling (else a positional ``ord`` is misread as ``axis``: ``ord=1``
    spuriously raises, ``ord=np.inf`` silently returns the L2 norm). Supported:
    ``ord`` in {None, 2} is L2/Frobenius via squared-sum + sqrt, full or
    axis-aware; ``ord`` in {1, inf} for a 1-D operand is ``sum(|v|)``/``max(|v|)``.
    A matrix 1/inf norm, an ord+axis combination, or any other ``ord`` raises
    ``NotImplementedError`` -- never a silent wrong norm.
    """
    if not args:
        raise NotImplementedError("np.linalg.norm needs an operand")
    kwargs = list(kwargs or [])
    a = args[0]
    ord_node: ast.expr | None = args[1] if len(args) >= 2 else None
    for kw in kwargs:
        if kw.arg == "ord":
            ord_node = kw.value
    kind = classify_norm_ord(ord_node)
    if kind is None:
        raise NotImplementedError("np.linalg.norm: unsupported ord (only None/2, 1, inf)")
    # Strip ``ord`` (positional arg[1] or keyword) so the shared axis reader
    # sees the reduction layout (operand, axis, keepdims).
    reduction_args = [a] + list(args[2:])
    reduction_kwargs = [kw for kw in kwargs if kw.arg != "ord"]
    axes, keepdims = read_axis_keepdims(reduction_args, reduction_kwargs)

    if kind == "l2":
        if axes is None:
            # Full reduction -- scalar accumulator + sqrt.
            extent = iter_extent_of_(a, shape_table)
            if extent is None:
                raise NotImplementedError("np.linalg.norm: cannot derive iteration extent")
            iters = [make_iter_name("__nr", i) for i in range(len(extent))]
            sa = scalarize_at_iters(a, [name_(it) for it in iters], shape_table)
            acc_init = ast.Assign(targets=[store_(target.id)], value=const_(0.0))
            inner = [
                ast.AugAssign(target=store_(target.id), op=ast.Add(), value=ast.BinOp(left=sa, op=ast.Mult(), right=sa))
            ]
            loops = wrap_for_loops(iters, extent, inner)
            finish = ast.Assign(
                targets=[store_(target.id)], value=ast.Call(func=name_("sqrt"), args=[name_(target.id)], keywords=[])
            )
            return [acc_init, *loops, finish]
        # Axis-aware -- reduce per axis kept, sum-of-squares, then sqrt.
        sq_op = lambda acc, x: ast.BinOp(left=acc, op=ast.Add(), right=ast.BinOp(left=x, op=ast.Mult(), right=x))
        sqrt_post = lambda lvalue, divisor: ast.Assign(
            targets=[
                lvalue
                if isinstance(lvalue, ast.Name)
                else ast.Subscript(value=lvalue.value, slice=lvalue.slice, ctx=ast.Store())
            ],
            value=ast.Call(
                func=name_("sqrt"),
                args=[
                    lvalue
                    if isinstance(lvalue, ast.Name)
                    else ast.Subscript(value=lvalue.value, slice=lvalue.slice, ctx=ast.Load())
                ],
                keywords=[],
            ),
        )
        return expand_axis_reduction(
            target, reduction_args, reduction_kwargs, shape_table, init=const_(0.0), op_fn=sq_op, post_fn=sqrt_post
        )

    # ``ord`` in {1, inf}. A 1-D operand is a vector norm (sum|v| / max|v|); a
    # 2-D operand is a matrix norm (ord=1 = max column abs-sum, ord=inf = max row
    # abs-sum) -- the max over per-line abs-sums.
    if axes is not None:
        raise NotImplementedError("np.linalg.norm: ord=1/inf with axis= not supported")
    extent = iter_extent_of_(a, shape_table)
    if extent is None:
        raise NotImplementedError("np.linalg.norm: cannot derive iteration extent")
    if len(extent) == 1:
        it = make_iter_name("__nr", 0)
        sa = scalarize_at_iters(a, [name_(it)], shape_table)
        abs_sa = ast.Call(func=name_("abs"), args=[sa], keywords=[])
        acc_init = ast.Assign(targets=[store_(target.id)], value=const_(0.0))
        if kind == "l1":
            inner = [ast.AugAssign(target=store_(target.id), op=ast.Add(), value=abs_sa)]
        else:  # inf: running max of |v| (|v| >= 0, so 0 is a safe max identity)
            inner = [
                ast.Assign(
                    targets=[store_(target.id)],
                    value=ast.IfExp(
                        test=ast.Compare(left=copy.deepcopy(abs_sa), ops=[ast.Gt()], comparators=[name_(target.id)]),
                        body=abs_sa,
                        orelse=name_(target.id),
                    ),
                )
            ]
        loops = wrap_for_loops([it], extent, inner)
        return [acc_init, *loops]
    if len(extent) == 2 and isinstance(a, ast.Name):
        # Matrix ord=1 / ord=inf: accumulate each line's abs-sum into a scalar
        # ``__nmc`` then keep the running max. ord=1 sums down columns (outer j),
        # ord=inf sums across rows (outer i); the element a[i, j] is the same.
        m_ext, n_ext = extent
        i_it, j_it, csum = "__nmi", "__nmj", "__nmc"
        elem = ast.Call(
            func=name_("abs"),
            args=[
                ast.Subscript(
                    value=name_(a.id), slice=ast.Tuple(elts=[name_(i_it), name_(j_it)], ctx=ast.Load()), ctx=ast.Load()
                )
            ],
            keywords=[],
        )
        if kind == "l1":  # outer over columns j, inner over rows i
            outer_it, outer_bound, inner_it, inner_bound = j_it, n_ext, i_it, m_ext
        else:  # inf: outer over rows i, inner over columns j
            outer_it, outer_bound, inner_it, inner_bound = i_it, m_ext, j_it, n_ext
        inner_loop = ast.For(
            target=store_(inner_it),
            iter=ast.Call(func=name_("range"), args=[copy.deepcopy(inner_bound)], keywords=[]),
            body=[ast.AugAssign(target=store_(csum), op=ast.Add(), value=elem)],
            orelse=[],
        )
        keep_max = ast.Assign(
            targets=[store_(target.id)],
            value=ast.IfExp(
                test=ast.Compare(left=name_(csum), ops=[ast.Gt()], comparators=[name_(target.id)]),
                body=name_(csum),
                orelse=name_(target.id),
            ),
        )
        outer_loop = ast.For(
            target=store_(outer_it),
            iter=ast.Call(func=name_("range"), args=[copy.deepcopy(outer_bound)], keywords=[]),
            body=[ast.Assign(targets=[store_(csum)], value=const_(0.0)), inner_loop, keep_max],
            orelse=[],
        )
        return [ast.Assign(targets=[store_(target.id)], value=const_(0.0)), outer_loop]
    raise NotImplementedError("np.linalg.norm: ord=1/inf supported for a 1-D or 2-D operand")


def guarded_div(num: ast.expr, denom: ast.expr) -> ast.expr:
    """``denom != 0 ? num / denom : 0`` -- guards the naive Gaussian-elimination
    solve against a zero pivot. A rank-deficient system (e.g. GMRES after an
    early break leaves a singular ``H``) makes a diagonal pivot exactly 0;
    numpy's SVD-based ``lstsq`` still returns a finite minimum-norm solution,
    but the unguarded division would emit NaN/inf. Inert for a full-rank system."""
    return ast.IfExp(
        test=ast.Compare(left=copy.deepcopy(denom), ops=[ast.NotEq()], comparators=[const_(0.0)]),
        body=ast.BinOp(left=num, op=ast.Div(), right=denom),
        orelse=const_(0.0),
    )


def expand_lstsq(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    fresh_local_allocs: dict[str, tuple[str, ...]] | None = None,
) -> list[ast.stmt]:
    """``y = np.linalg.lstsq(A, b, rcond=...)[0]`` -> in-place Gaussian
    elimination with partial pivoting, writing the solution into ``target``.

    Conservative scope: A must be a SQUARE M x M region, either a bare Name
    (shape (M, M)) or a Subscript like ``H[:m, :]`` whose first slice has a
    ``stop`` (m); b matches as a Name of shape (M,) or a Subscript ``e1[:m]``.
    M at codegen time is the slice's ``stop`` symbol or the array's first dim.
    A and b are mutated in place -- unlike numpy's side-effect-free ``lstsq``,
    but acceptable since the only caller in scope (gmres) uses the result once
    and never re-reads H or b after.

    Heuristic: detects the ``np.linalg.lstsq(...)[0]`` form via the caller
    (LibNodeRewriter handles the Subscript unwrap); takes only the positional
    args, ``rcond`` in ``kwargs`` is consumed and ignored.
    """
    if len(args) < 2:
        raise NotImplementedError("np.linalg.lstsq needs A and b args")
    a_node, b_node = args[0], args[1]
    a_size = lstsq_first_axis_size(a_node, shape_table)
    if a_size is None:
        raise NotImplementedError("np.linalg.lstsq: cannot infer A's leading dimension")
    a_name, a_base = lstsq_array_base(a_node)
    if a_name is None:
        raise NotImplementedError("np.linalg.lstsq: A must be a Name or simple slice")
    # ``b`` may be an expression (gmres passes ``beta * e1[:m]``); Gaussian
    # elimination needs an indexable, MUTABLE b, so materialize it into a fresh
    # length-M temp vector via a fill loop first. A bare Name/simple slice is
    # used in place.
    pre: list[ast.stmt] = []
    b_name, b_base = lstsq_array_base(b_node)
    if b_name is None:
        b_name, b_base = "__lq_b", None
        bi_ = "__lq_bi"
        # Allocation marker first (mirrors expand_solve/expand_inv): length-M
        # ``__lq_b``, whose M depends on a body-computed scalar (gmres ``m =
        # min(max_iter, n)``), is deferred-malloc'd here. Without it the NULL
        # pointer is dereferenced in the fill loop below.
        pre.append(
            ast.Assign(
                targets=[store_(b_name)], value=ast.Call(func=name_("__hpcagent_bench_zeros__"), args=[], keywords=[])
            )
        )
        elem_ = scalarize_at_iters(b_node, [name_(bi_)], shape_table)
        pre.append(
            ast.For(
                target=store_(bi_),
                iter=ast.Call(func=name_("range"), args=[a_size], keywords=[]),
                body=[
                    ast.Assign(
                        targets=[ast.Subscript(value=name_(b_name), slice=name_(bi_), ctx=ast.Store())], value=elem_
                    )
                ],
                orelse=[],
            )
        )
        if fresh_local_allocs is not None:
            fresh_local_allocs[b_name] = (ast.unparse(a_size),)
    # The solution vector ``target`` is written element-wise by back
    # substitution; register its shape so the caller allocates it.
    if isinstance(target, ast.Name):
        shape_table[target.id] = (ast.unparse(a_size),)
    # M = A.shape[0]
    p_iter = "__lq_p"
    r_iter = "__lq_r"
    c_iter = "__lq_c"
    factor = "__lq_factor"
    sum_v = "__lq_sum"
    p_name = name_(p_iter)
    r_name = name_(r_iter)
    c_name = name_(c_iter)
    a_pp = lstsq_index2d(a_name, p_name, p_name, a_base)
    a_rp = lstsq_index2d(a_name, r_name, p_name, a_base)
    a_pc = lstsq_index2d(a_name, p_name, c_name, a_base)
    a_rcol = lstsq_index2d(a_name, r_name, c_name, a_base)
    a_rr = lstsq_index2d(a_name, r_name, r_name, a_base)
    b_p = lstsq_index1d(b_name, p_name, b_base)
    b_r = lstsq_index1d(b_name, r_name, b_base)
    # Forward elimination over pivot p:
    #   for p in 0..M:
    #     for r in p+1..M:
    #       factor = A[r,p] / A[p,p]
    #       for c in p+1..M: A[r,c] -= factor * A[p,c]
    #       b[r] -= factor * b[p]
    inner_c = [
        ast.AugAssign(
            target=ast.Subscript(
                value=name_(a_name), slice=ast.Tuple(elts=[r_name, c_name], ctx=ast.Load()), ctx=ast.Store()
            ),
            op=ast.Sub(),
            value=ast.BinOp(left=name_(factor), op=ast.Mult(), right=a_pc),
        )
    ]
    inner_c_for = ast.For(
        target=store_(c_iter),
        iter=ast.Call(
            func=name_("range"), args=[ast.BinOp(left=p_name, op=ast.Add(), right=const_(1)), a_size], keywords=[]
        ),
        body=inner_c,
        orelse=[],
    )
    factor_assign = ast.Assign(targets=[store_(factor)], value=guarded_div(a_rp, a_pp))
    b_aug = ast.AugAssign(
        target=ast.Subscript(value=name_(b_name), slice=r_name, ctx=ast.Store()),
        op=ast.Sub(),
        value=ast.BinOp(left=name_(factor), op=ast.Mult(), right=b_p),
    )
    inner_r = ast.For(
        target=store_(r_iter),
        iter=ast.Call(
            func=name_("range"), args=[ast.BinOp(left=p_name, op=ast.Add(), right=const_(1)), a_size], keywords=[]
        ),
        body=[factor_assign, inner_c_for, b_aug],
        orelse=[],
    )
    fwd = ast.For(
        target=store_(p_iter), iter=ast.Call(func=name_("range"), args=[a_size], keywords=[]), body=[inner_r], orelse=[]
    )
    # Back substitution:
    #   for r in M-1..0 (reverse):
    #     sum = b[r]
    #     for c in r+1..M: sum -= A[r,c] * y[c]
    #     y[r] = sum / A[r,r]
    y_c = ast.Subscript(value=name_(target.id), slice=c_name, ctx=ast.Load())
    y_r = ast.Subscript(value=name_(target.id), slice=r_name, ctx=ast.Store())
    bs_inner = [
        ast.AugAssign(target=store_(sum_v), op=ast.Sub(), value=ast.BinOp(left=a_rcol, op=ast.Mult(), right=y_c))
    ]
    bs_inner_for = ast.For(
        target=store_(c_iter),
        iter=ast.Call(
            func=name_("range"), args=[ast.BinOp(left=r_name, op=ast.Add(), right=const_(1)), a_size], keywords=[]
        ),
        body=bs_inner,
        orelse=[],
    )
    bs_sum_init = ast.Assign(targets=[store_(sum_v)], value=b_r)
    bs_y_assign = ast.Assign(targets=[y_r], value=guarded_div(name_(sum_v), a_rr))
    # Reverse iteration via ``range(M-1, -1, -1)``.
    bs = ast.For(
        target=store_(r_iter),
        iter=ast.Call(
            func=name_("range"),
            args=[ast.BinOp(left=a_size, op=ast.Sub(), right=const_(1)), const_(-1), const_(-1)],
            keywords=[],
        ),
        body=[bs_sum_init, bs_inner_for, bs_y_assign],
        orelse=[],
    )
    return pre + [fwd, bs]


def lstsq_first_axis_size(node: ast.expr, shape_table: dict[str, tuple[str, ...]]) -> ast.expr | None:
    """First-axis size of ``node`` as an AST expression: bare Name -> shape_table
    lookup; Subscript ``H[:m, ...]`` -> the explicit ``m`` stop."""
    if isinstance(node, ast.Name):
        shape = shape_table.get(node.id)
        if shape:
            return const_or_name(shape[0])
        return None
    if isinstance(node, ast.Subscript):
        sl = node.slice
        first = sl.elts[0] if isinstance(sl, ast.Tuple) else sl
        if isinstance(first, ast.Slice) and first.upper is not None:
            return first.upper
        if isinstance(first, ast.Slice) and first.upper is None:
            # Whole axis -- fall back to shape_table of the base.
            if isinstance(node.value, ast.Name):
                shape = shape_table.get(node.value.id)
                if shape:
                    return const_or_name(shape[0])
        if not isinstance(first, ast.Slice) and isinstance(node.value, ast.Name):
            shape = shape_table.get(node.value.id)
            if shape:
                return const_or_name(shape[0])
    return None


def lstsq_array_base(node: ast.expr) -> tuple[str | None, list[ast.expr | None] | None]:
    """Return ``(name, base_offsets)`` for a Name or simple slice subscript.
    ``base_offsets`` is the lower-bound shift per axis (list of ast.expr) so the
    expander can rewrite ``A[i, j]`` as ``Name[i + base0, j + base1]``. Zero for
    a Name or ``H[:m, :]`` (lower=None defaults to 0); captured for ``H[2:m,
    :]`` (non-zero lower)."""
    if isinstance(node, ast.Name):
        return node.id, None
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        sl = node.slice
        slots = sl.elts if isinstance(sl, ast.Tuple) else [sl]
        bases = []
        for s in slots:
            if isinstance(s, ast.Slice):
                bases.append(s.lower if s.lower is not None else const_(0))
            else:
                bases.append(None)  # concrete index -- not a slice axis
        return node.value.id, bases
    return None, None


def lstsq_index2d(name: str, i: ast.expr, j: ast.expr, base: list[ast.expr | None] | None) -> ast.Subscript:
    """Build ``name[i, j]`` (or ``name[i + base0, j + base1]`` when
    a non-zero base is present)."""
    if base is not None:
        slot_i = (
            i
            if (isinstance(base[0], ast.Constant) and base[0].value == 0)
            else ast.BinOp(left=i, op=ast.Add(), right=base[0])
        )
        slot_j = (
            j
            if (isinstance(base[1], ast.Constant) and base[1].value == 0)
            else ast.BinOp(left=j, op=ast.Add(), right=base[1])
        )
    else:
        slot_i, slot_j = i, j
    return ast.Subscript(value=name_(name), slice=ast.Tuple(elts=[slot_i, slot_j], ctx=ast.Load()), ctx=ast.Load())


def lstsq_index1d(name: str, i: ast.expr, base: list[ast.expr | None] | None) -> ast.Subscript:
    if base is not None and not (isinstance(base[0], ast.Constant) and base[0].value == 0):
        slot = ast.BinOp(left=i, op=ast.Add(), right=base[0])
    else:
        slot = i
    return ast.Subscript(value=name_(name), slice=slot, ctx=ast.Load())


def expand_cholesky(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    local_dtypes: dict[str, str] | None = None,
) -> list[ast.stmt]:
    """``L = np.linalg.cholesky(A)`` -> Cholesky-Banachiewicz triple loop.
    Computes ``L`` such that ``L @ L.conj().T == A`` for Hermitian positive-
    definite ``A`` (the conjugate is a no-op for real matrices). Naive O(n^3),
    no blocking::

        for j in range(n):
            s = A[j, j]
            for k in range(j):
                s -= L[j, k] * conj(L[j, k])
            L[j, j] = sqrt(s)
            for i in range(j + 1, n):
                s = A[i, j]
                for k in range(j):
                    s -= L[i, k] * conj(L[j, k])
                L[i, j] = s / L[j, j]
    """
    if not args_one_name(args):
        raise NotImplementedError("np.linalg.cholesky needs Name arg")
    a = args[0]
    a_shape = shape_table.get(a.id)
    if not a_shape or len(a_shape) != 2:
        raise NotImplementedError("cholesky: only 2-D arg")
    if local_dtypes is not None:
        a_dt = local_dtypes.get(a.id)
        if a_dt:
            local_dtypes[target.id] = a_dt
            local_dtypes.setdefault("__s", a_dt)
    n = a_shape[0]
    n_ast = const_or_name(n)
    inner_k = [
        ast.AugAssign(
            target=store_("__s"),
            op=ast.Sub(),
            value=ast.BinOp(
                left=ast.Subscript(
                    value=name_(target.id),
                    slice=ast.Tuple(elts=[name_("__j"), name_("__k")], ctx=ast.Load()),
                    ctx=ast.Load(),
                ),
                op=ast.Mult(),
                right=attr_call(
                    "np",
                    "conj",
                    [
                        ast.Subscript(
                            value=name_(target.id),
                            slice=ast.Tuple(elts=[name_("__j"), name_("__k")], ctx=ast.Load()),
                            ctx=ast.Load(),
                        )
                    ],
                ),
            ),
        ),
    ]
    inner_i = [
        ast.Assign(
            targets=[store_("__s")],
            value=ast.Subscript(
                value=name_(a.id), slice=ast.Tuple(elts=[name_("__i"), name_("__j")], ctx=ast.Load()), ctx=ast.Load()
            ),
        ),
        ast.For(
            target=store_("__k"),
            iter=ast.Call(func=name_("range"), args=[name_("__j")], keywords=[]),
            body=[
                ast.AugAssign(
                    target=store_("__s"),
                    op=ast.Sub(),
                    value=ast.BinOp(
                        left=ast.Subscript(
                            value=name_(target.id),
                            slice=ast.Tuple(elts=[name_("__i"), name_("__k")], ctx=ast.Load()),
                            ctx=ast.Load(),
                        ),
                        op=ast.Mult(),
                        right=attr_call(
                            "np",
                            "conj",
                            [
                                ast.Subscript(
                                    value=name_(target.id),
                                    slice=ast.Tuple(elts=[name_("__j"), name_("__k")], ctx=ast.Load()),
                                    ctx=ast.Load(),
                                )
                            ],
                        ),
                    ),
                )
            ],
            orelse=[],
        ),
        ast.Assign(
            targets=[
                ast.Subscript(
                    value=name_(target.id),
                    slice=ast.Tuple(elts=[name_("__i"), name_("__j")], ctx=ast.Load()),
                    ctx=ast.Store(),
                )
            ],
            value=ast.BinOp(
                left=name_("__s"),
                op=ast.Div(),
                right=ast.Subscript(
                    value=name_(target.id),
                    slice=ast.Tuple(elts=[name_("__j"), name_("__j")], ctx=ast.Load()),
                    ctx=ast.Load(),
                ),
            ),
        ),
    ]
    j_body = [
        ast.Assign(
            targets=[store_("__s")],
            value=ast.Subscript(
                value=name_(a.id), slice=ast.Tuple(elts=[name_("__j"), name_("__j")], ctx=ast.Load()), ctx=ast.Load()
            ),
        ),
        ast.For(
            target=store_("__k"),
            iter=ast.Call(func=name_("range"), args=[name_("__j")], keywords=[]),
            body=inner_k,
            orelse=[],
        ),
        ast.Assign(
            targets=[
                ast.Subscript(
                    value=name_(target.id),
                    slice=ast.Tuple(elts=[name_("__j"), name_("__j")], ctx=ast.Load()),
                    ctx=ast.Store(),
                )
            ],
            value=ast.Call(func=name_("sqrt"), args=[name_("__s")], keywords=[]),
        ),
        ast.For(
            target=store_("__i"),
            iter=ast.Call(
                func=name_("range"),
                args=[ast.BinOp(left=name_("__j"), op=ast.Add(), right=const_(1)), n_ast],
                keywords=[],
            ),
            body=inner_i,
            orelse=[],
        ),
    ]
    # numpy's cholesky returns 0 in the strict upper triangle, but the
    # Banachiewicz loop below only writes the lower triangle + diagonal.
    # Pre-zero the upper triangle so unwritten cells aren't malloc garbage;
    # ``target`` is a fresh temp (!= ``a``), so this can't corrupt the source.
    zero_upper = ast.For(
        target=store_("__zi"),
        iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
        body=[
            ast.For(
                target=store_("__zj"),
                iter=ast.Call(
                    func=name_("range"),
                    args=[ast.BinOp(left=name_("__zi"), op=ast.Add(), right=const_(1)), n_ast],
                    keywords=[],
                ),
                body=[
                    ast.Assign(
                        targets=[
                            ast.Subscript(
                                value=name_(target.id),
                                slice=ast.Tuple(elts=[name_("__zi"), name_("__zj")], ctx=ast.Load()),
                                ctx=ast.Store(),
                            )
                        ],
                        value=const_(0.0),
                    )
                ],
                orelse=[],
            )
        ],
        orelse=[],
    )
    return [
        zero_upper,
        ast.For(
            target=store_("__j"), iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]), body=j_body, orelse=[]
        ),
    ]


def expand_linalg_solve(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    local_dtypes: dict[str, str] | None = None,
    fresh_local_allocs: dict[str, tuple[str, ...]] | None = None,
) -> list[ast.stmt]:
    """``x = np.linalg.solve(A, b)`` solves ``Ax = b`` for square A. Implemented
    as Gauss-Jordan elimination on the augmented [A | b] matrix. Non-Name
    arguments are materialised into fresh locals first, so callers can write
    ``np.linalg.solve(normal + damping * np.diag(scale), -gradient)``. A must
    be 2-D and b 1-D or 2-D; x is written into target with the same shape as b.
    """

    def temp_name() -> str:
        name = f"__sol_arg{LINALG_AW[0]}"
        LINALG_AW[0] += 1
        return name

    def infer_expr_dtype(expr: ast.expr) -> str | None:
        if local_dtypes is None:
            return None
        if reads_complex(expr, local_dtypes):
            return "complex128"
        for node in ast.walk(expr):
            if isinstance(node, ast.Name):
                dt = local_dtypes.get(node.id)
                if dt:
                    return dt
        return None

    out: list[ast.stmt] = []
    materialized = list(args)
    for i, arg in enumerate(materialized):
        if isinstance(arg, ast.Name):
            continue
        ext = iter_extent_of_(arg, shape_table)
        if ext is None:
            raise NotImplementedError("np.linalg.solve: argument shape not inferable")
        tmp = temp_name()
        shape_tokens = tuple(ast.unparse(e) for e in ext)
        shape_table[tmp] = shape_tokens
        if fresh_local_allocs is not None:
            fresh_local_allocs[tmp] = shape_tokens
        arg_dt = infer_expr_dtype(arg)
        if arg_dt is not None and local_dtypes is not None:
            local_dtypes[tmp] = arg_dt
        out.append(
            ast.Assign(
                targets=[store_(tmp)], value=ast.Call(func=name_("__hpcagent_bench_zeros__"), args=[], keywords=[])
            )
        )
        out.append(ast.Assign(targets=[store_(tmp)], value=arg))
        materialized[i] = name_(tmp)
    args = materialized

    if len(args) < 2 or not isinstance(args[0], ast.Name) or not isinstance(args[1], ast.Name):
        raise NotImplementedError("np.linalg.solve needs Name args")
    a = args[0]
    b = args[1]
    a_shape = shape_table.get(a.id)
    b_shape = shape_table.get(b.id)
    if not a_shape or len(a_shape) != 2:
        raise NotImplementedError("np.linalg.solve: A must be 2-D")
    if not b_shape or len(b_shape) not in (1, 2):
        raise NotImplementedError("np.linalg.solve: b must be 1-D or 2-D")
    n = a_shape[0]
    n_ast = const_or_name(n)
    # Same Gauss-Jordan body as expand_linalg_inv, but the row ops apply to
    # ``b`` (not the identity), giving x = A^-1 @ b. ``__sol_aw`` is the
    # working copy of A. A fixed name would alias across call sites with
    # different sizes, so number it like ``expand_linalg_inv`` does.
    aw_name = f"__sol_aw{LINALG_AW[0]}"
    LINALG_AW[0] += 1
    aw = lambda r, c: ast.Subscript(value=name_(aw_name), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Load())
    aw_store = lambda r, c: ast.Subscript(
        value=name_(aw_name), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Store()
    )
    # ``b`` indexing depends on rank.
    is_2d = len(b_shape) == 2

    def b_load(r: ast.expr, c: ast.expr | None = None) -> ast.Subscript:
        if is_2d:
            return ast.Subscript(value=name_(target.id), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Load())
        return ast.Subscript(value=name_(target.id), slice=r, ctx=ast.Load())

    def b_store(r: ast.expr, c: ast.expr | None = None) -> ast.Subscript:
        if is_2d:
            return ast.Subscript(value=name_(target.id), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Store())
        return ast.Subscript(value=name_(target.id), slice=r, ctx=ast.Store())

    # Publish working-buffer shape + dtype + fresh-local alloc so the emit
    # declares the buffer as a flat 2-D buffer of A's element dtype (same
    # logic as ``expand_linalg_inv``).
    shape_table[aw_name] = (n, n)
    a_dt = None
    if local_dtypes is not None:
        a_dt = local_dtypes.get(a.id) or local_dtypes.get(b.id)
        if a_dt is not None:
            local_dtypes[aw_name] = a_dt
            local_dtypes[target.id] = a_dt
            for nm in ("__sol_tmp", "__sol_factor"):
                local_dtypes.setdefault(nm, a_dt)
    if fresh_local_allocs is not None:
        fresh_local_allocs[aw_name] = (n, n)
    out.append(
        ast.Assign(
            targets=[store_(aw_name)], value=ast.Call(func=name_("__hpcagent_bench_zeros__"), args=[], keywords=[])
        )
    )
    # Init: copy A into __sol_aw and b into target.
    if is_2d:
        m_ast = const_or_name(b_shape[1])
        copy_inner = ast.For(
            target=store_("__sol_j"),
            iter=ast.Call(func=name_("range"), args=[m_ast], keywords=[]),
            body=[
                ast.Assign(
                    targets=[b_store(name_("__sol_i"), name_("__sol_j"))],
                    value=ast.Subscript(
                        value=name_(b.id),
                        slice=ast.Tuple(elts=[name_("__sol_i"), name_("__sol_j")], ctx=ast.Load()),
                        ctx=ast.Load(),
                    ),
                )
            ],
            orelse=[],
        )
    else:
        copy_inner = ast.Assign(
            targets=[b_store(name_("__sol_i"))],
            value=ast.Subscript(value=name_(b.id), slice=name_("__sol_i"), ctx=ast.Load()),
        )
    out.append(
        ast.For(
            target=store_("__sol_i"),
            iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
            body=[
                ast.For(
                    target=store_("__sol_j"),
                    iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
                    body=[
                        ast.Assign(
                            targets=[aw_store(name_("__sol_i"), name_("__sol_j"))],
                            value=ast.Subscript(
                                value=name_(a.id),
                                slice=ast.Tuple(elts=[name_("__sol_i"), name_("__sol_j")], ctx=ast.Load()),
                                ctx=ast.Load(),
                            ),
                        )
                    ],
                    orelse=[],
                ),
                copy_inner if is_2d else copy_inner,
            ],
            orelse=[],
        )
    )
    # Gauss-Jordan on (__sol_aw | target).
    K = name_("__sol_k")
    P = name_("__sol_p")
    R = name_("__sol_r")
    C = name_("__sol_c")
    F = name_("__sol_factor")
    T = name_("__sol_tmp")
    # Pivot search.
    pivot_init = ast.Assign(targets=[store_("__sol_p")], value=K)
    pivot_scan = ast.For(
        target=store_("__sol_r"),
        iter=ast.Call(func=name_("range"), args=[ast.BinOp(left=K, op=ast.Add(), right=const_(1)), n_ast], keywords=[]),
        body=[
            ast.If(
                test=ast.Compare(
                    left=ast.Call(func=name_("abs"), args=[aw(R, K)], keywords=[]),
                    ops=[ast.Gt()],
                    comparators=[ast.Call(func=name_("abs"), args=[aw(P, K)], keywords=[])],
                ),
                body=[ast.Assign(targets=[store_("__sol_p")], value=R)],
                orelse=[],
            )
        ],
        orelse=[],
    )
    # Swap row p and row k in __sol_aw.
    swap_aw = [
        ast.Assign(targets=[store_("__sol_tmp")], value=aw(K, C)),
        ast.Assign(targets=[aw_store(K, C)], value=aw(P, C)),
        ast.Assign(targets=[aw_store(P, C)], value=T),
    ]
    swap_aw_loop = ast.For(
        target=store_("__sol_c"), iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]), body=swap_aw, orelse=[]
    )
    # Swap row p and row k in target (the b-side).
    if is_2d:
        m_ast = const_or_name(b_shape[1])
        swap_b = [
            ast.Assign(targets=[store_("__sol_tmp")], value=b_load(K, C)),
            ast.Assign(targets=[b_store(K, C)], value=b_load(P, C)),
            ast.Assign(targets=[b_store(P, C)], value=T),
        ]
        swap_b_loop = ast.For(
            target=store_("__sol_c"),
            iter=ast.Call(func=name_("range"), args=[m_ast], keywords=[]),
            body=swap_b,
            orelse=[],
        )
    else:
        swap_b_loop = ast.If(
            test=ast.Compare(left=P, ops=[ast.NotEq()], comparators=[K]),
            body=[
                ast.Assign(targets=[store_("__sol_tmp")], value=b_load(K)),
                ast.Assign(targets=[b_store(K)], value=b_load(P)),
                ast.Assign(targets=[b_store(P)], value=T),
            ],
            orelse=[],
        )
    # Divide pivot row by aw[k, k]. Stash divisor.
    pivot_div_stash = ast.Assign(targets=[store_("__sol_factor")], value=aw(K, K))
    pivot_div_aw_body = [
        ast.Assign(targets=[aw_store(K, C)], value=ast.BinOp(left=aw(K, C), op=ast.Div(), right=F)),
    ]
    pivot_div_aw = ast.For(
        target=store_("__sol_c"),
        iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
        body=pivot_div_aw_body,
        orelse=[],
    )
    if is_2d:
        pivot_div_b_body = [
            ast.Assign(targets=[b_store(K, C)], value=ast.BinOp(left=b_load(K, C), op=ast.Div(), right=F)),
        ]
        pivot_div_b = ast.For(
            target=store_("__sol_c"),
            iter=ast.Call(func=name_("range"), args=[const_or_name(b_shape[1])], keywords=[]),
            body=pivot_div_b_body,
            orelse=[],
        )
    else:
        pivot_div_b = ast.Assign(targets=[b_store(K)], value=ast.BinOp(left=b_load(K), op=ast.Div(), right=F))
    # Eliminate other rows.
    elim_factor = ast.Assign(targets=[store_("__sol_factor")], value=aw(R, K))
    elim_aw_inner = ast.For(
        target=store_("__sol_c"),
        iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
        body=[
            ast.Assign(
                targets=[aw_store(R, C)],
                value=ast.BinOp(left=aw(R, C), op=ast.Sub(), right=ast.BinOp(left=F, op=ast.Mult(), right=aw(K, C))),
            )
        ],
        orelse=[],
    )
    if is_2d:
        elim_b_inner = ast.For(
            target=store_("__sol_c"),
            iter=ast.Call(func=name_("range"), args=[const_or_name(b_shape[1])], keywords=[]),
            body=[
                ast.Assign(
                    targets=[b_store(R, C)],
                    value=ast.BinOp(
                        left=b_load(R, C), op=ast.Sub(), right=ast.BinOp(left=F, op=ast.Mult(), right=b_load(K, C))
                    ),
                )
            ],
            orelse=[],
        )
    else:
        elim_b_inner = ast.Assign(
            targets=[b_store(R)],
            value=ast.BinOp(left=b_load(R), op=ast.Sub(), right=ast.BinOp(left=F, op=ast.Mult(), right=b_load(K))),
        )
    elim_outer = ast.For(
        target=store_("__sol_r"),
        iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
        body=[
            ast.If(
                test=ast.Compare(left=R, ops=[ast.NotEq()], comparators=[K]),
                body=[elim_factor, elim_aw_inner, elim_b_inner],
                orelse=[],
            )
        ],
        orelse=[],
    )
    k_body = [pivot_init, pivot_scan, swap_aw_loop, swap_b_loop, pivot_div_stash, pivot_div_aw, pivot_div_b, elim_outer]
    out.append(
        ast.For(
            target=store_("__sol_k"),
            iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
            body=k_body,
            orelse=[],
        )
    )
    return out


#: Monotone counter for the ``inv`` scratch working-copy buffer. A fixed name
#: (``__inv_aw``) would alias across call sites: LS3DF's Rayleigh-Ritz runs
#: ``np.linalg.inv`` once per SCF sub-solve, each a different size, so the
#: second call's shape would overwrite the first's declaration and the first
#: inversion would malloc from an uninitialised dimension. A per-call suffix
#: keeps each buffer distinct. Reset per translation unit by :func:`reset_temp_counters`.
LINALG_AW = [0]


def reset_temp_counters() -> None:
    """Zero the scratch-buffer name counters at the start of a translation unit.

    They only have to be unique WITHIN one kernel, but they are module state, so left
    running they number the second kernel emitted in a process from wherever the first
    stopped -- the emitted text then depends on what else the process translated, and
    re-emitting the same kernel twice yields two different sources. Called by
    :func:`numpyto_common.lowering.lower`, the single entry point of a translation unit."""
    OP_SPILL_TEMP[0] = 0
    LINALG_AW[0] = 0


def expand_linalg_inv(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    local_dtypes: dict[str, str] | None = None,
    fresh_local_allocs: dict[str, tuple[str, ...]] | None = None,
) -> list[ast.stmt]:
    """``X = np.linalg.inv(A)`` -- in-place Gauss-Jordan elimination with
    partial pivoting on the augmented [A | I] matrix (textbook, Golub & Van
    Loan Algorithm 3.4.1):

    1. Form A_work = copy of A, X = I.
    2. For each pivot column k: find row with max |A_work[k:, k]|, swap rows k
       and pivot; divide pivot row by A_work[k, k]; for each row i != k,
       subtract A_work[i, k] * pivot row.
    3. Result: A_work becomes I, X becomes A^-1.

    Conservative: ``A`` must be a Name with a known square 2-D shape. A is
    preserved -- the elimination runs on a copy, ``__inv_aw``.
    """
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.linalg.inv needs Name first arg")
    a = args[0]
    shape = shape_table.get(a.id)
    if not shape or len(shape) != 2:
        raise NotImplementedError("np.linalg.inv: only 2-D square input supported")
    n = shape[0]
    n_ast = const_or_name(n)
    aw_name = f"__inv_aw{LINALG_AW[0]}"
    LINALG_AW[0] += 1
    out: list[ast.stmt] = []
    # Publish the working buffer's shape so the emit flattens ``__inv_aw[i, j]``
    # as ``i*n + j`` instead of chained ``[i][j]``, and its dtype so the emit
    # declares the right element type (e.g. ``double _Complex __inv_aw[n*n]``).
    shape_table[aw_name] = (n, n)
    a_dt = None
    if local_dtypes is not None:
        a_dt = local_dtypes.get(a.id)
        if a_dt is not None:
            local_dtypes[aw_name] = a_dt
            local_dtypes[target.id] = a_dt
            # Scalar swap / pivot temps carry A's dtype too.
            for nm in ("__inv_tmp", "__inv_factor"):
                local_dtypes.setdefault(nm, a_dt)
    if fresh_local_allocs is not None:
        fresh_local_allocs[aw_name] = (n, n)
    out.append(
        ast.Assign(
            targets=[store_(aw_name)], value=ast.Call(func=name_("__hpcagent_bench_zeros__"), args=[], keywords=[])
        )
    )
    # Copy A to a working buffer ``__inv_aw[i, j]``; initialise target
    # as the identity I[i, j].
    out.append(
        ast.For(
            target=store_("__inv_i"),
            iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
            body=[
                ast.For(
                    target=store_("__inv_j"),
                    iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
                    body=[
                        ast.Assign(
                            targets=[
                                ast.Subscript(
                                    value=name_(aw_name),
                                    slice=ast.Tuple(elts=[name_("__inv_i"), name_("__inv_j")], ctx=ast.Load()),
                                    ctx=ast.Store(),
                                )
                            ],
                            value=ast.Subscript(
                                value=name_(a.id),
                                slice=ast.Tuple(elts=[name_("__inv_i"), name_("__inv_j")], ctx=ast.Load()),
                                ctx=ast.Load(),
                            ),
                        ),
                        ast.Assign(
                            targets=[
                                ast.Subscript(
                                    value=name_(target.id),
                                    slice=ast.Tuple(elts=[name_("__inv_i"), name_("__inv_j")], ctx=ast.Load()),
                                    ctx=ast.Store(),
                                )
                            ],
                            value=ast.IfExp(
                                test=ast.Compare(left=name_("__inv_i"), ops=[ast.Eq()], comparators=[name_("__inv_j")]),
                                body=const_(1.0),
                                orelse=const_(0.0),
                            ),
                        ),
                    ],
                    orelse=[],
                )
            ],
            orelse=[],
        )
    )
    # Outer loop over pivot column k = 0..n:
    # 1) find pivot row p = k; for r in k+1..n: if |aw[r,k]| > |aw[p,k]|: p = r
    # 2) swap rows p and k in aw and target
    # 3) divide aw[k, :] and target[k, :] by aw[k, k]
    # 4) for r != k: factor = aw[r, k]; aw[r, :] -= factor * aw[k, :];
    #                target[r, :] -= factor * target[k, :]
    aw = lambda r, c: ast.Subscript(value=name_(aw_name), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Load())
    aw_store = lambda r, c: ast.Subscript(
        value=name_(aw_name), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Store()
    )
    tgt = lambda r, c: ast.Subscript(
        value=name_(target.id), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Load()
    )
    tgt_store = lambda r, c: ast.Subscript(
        value=name_(target.id), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Store()
    )
    # Every use below is a FRESH node from this factory, never a shared one: the Fortran
    # emitter's loop-variable uniquifier mutates Name nodes in place, and __inv_r / __inv_c each
    # bind THREE separate sibling/nested loops here (pivot_scan+elim_outer; swap_loop+pivot_div+
    # elim_inner_loop). A shared node's rename from the first loop stuck on every later occurrence,
    # so elim_outer's body kept reading pivot_scan's row and swap_loop's column -- silently wrong
    # data (not a crash) that only showed up as a numeric mismatch, and one build compiled from
    # freed-then-realloc'd memory besides. See FortranRenameTemps.visit_Name.
    nm = lambda tag: name_(f"__inv_{tag}")
    # Pivot search.
    pivot_init = ast.Assign(targets=[store_("__inv_p")], value=nm("k"))
    pivot_scan = ast.For(
        target=store_("__inv_r"),
        iter=ast.Call(
            func=name_("range"), args=[ast.BinOp(left=nm("k"), op=ast.Add(), right=const_(1)), n_ast], keywords=[]
        ),
        body=[
            ast.If(
                test=ast.Compare(
                    left=ast.Call(func=name_("abs"), args=[aw(nm("r"), nm("k"))], keywords=[]),
                    ops=[ast.Gt()],
                    comparators=[ast.Call(func=name_("abs"), args=[aw(nm("p"), nm("k"))], keywords=[])],
                ),
                body=[ast.Assign(targets=[store_("__inv_p")], value=nm("r"))],
                orelse=[],
            )
        ],
        orelse=[],
    )
    # Swap row p and row k in both aw and target.
    swap_body = [
        ast.Assign(targets=[store_("__inv_tmp")], value=aw(nm("k"), nm("c"))),
        ast.Assign(targets=[aw_store(nm("k"), nm("c"))], value=aw(nm("p"), nm("c"))),
        ast.Assign(targets=[aw_store(nm("p"), nm("c"))], value=nm("tmp")),
        ast.Assign(targets=[store_("__inv_tmp")], value=tgt(nm("k"), nm("c"))),
        ast.Assign(targets=[tgt_store(nm("k"), nm("c"))], value=tgt(nm("p"), nm("c"))),
        ast.Assign(targets=[tgt_store(nm("p"), nm("c"))], value=nm("tmp")),
    ]
    swap_loop = ast.For(
        target=store_("__inv_c"),
        iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
        body=swap_body,
        orelse=[],
    )
    # Divide pivot row by aw[k, k] -- stash it first since the loop overwrites
    # aw[k, k] itself before every use.
    pivot_div_stash = ast.Assign(targets=[store_("__inv_factor")], value=aw(nm("k"), nm("k")))
    pivot_div_body_safe = [
        ast.Assign(
            targets=[tgt_store(nm("k"), nm("c"))],
            value=ast.BinOp(left=tgt(nm("k"), nm("c")), op=ast.Div(), right=nm("factor")),
        ),
        ast.Assign(
            targets=[aw_store(nm("k"), nm("c"))],
            value=ast.BinOp(left=aw(nm("k"), nm("c")), op=ast.Div(), right=nm("factor")),
        ),
    ]
    pivot_div = [
        pivot_div_stash,
        ast.For(
            target=store_("__inv_c"),
            iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
            body=pivot_div_body_safe,
            orelse=[],
        ),
    ]
    # Eliminate other rows.
    elim_factor = ast.Assign(targets=[store_("__inv_factor")], value=aw(nm("r"), nm("k")))
    elim_body_inner = [
        ast.Assign(
            targets=[tgt_store(nm("r"), nm("c"))],
            value=ast.BinOp(
                left=tgt(nm("r"), nm("c")),
                op=ast.Sub(),
                right=ast.BinOp(left=nm("factor"), op=ast.Mult(), right=tgt(nm("k"), nm("c"))),
            ),
        ),
        ast.Assign(
            targets=[aw_store(nm("r"), nm("c"))],
            value=ast.BinOp(
                left=aw(nm("r"), nm("c")),
                op=ast.Sub(),
                right=ast.BinOp(left=nm("factor"), op=ast.Mult(), right=aw(nm("k"), nm("c"))),
            ),
        ),
    ]
    elim_inner_loop = ast.For(
        target=store_("__inv_c"),
        iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
        body=elim_body_inner,
        orelse=[],
    )
    elim_outer = ast.For(
        target=store_("__inv_r"),
        iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
        body=[
            ast.If(
                test=ast.Compare(left=nm("r"), ops=[ast.NotEq()], comparators=[nm("k")]),
                body=[elim_factor, elim_inner_loop],
                orelse=[],
            )
        ],
        orelse=[],
    )
    # K-loop body.
    k_body = [pivot_init, pivot_scan, swap_loop] + pivot_div + [elim_outer]
    out.append(
        ast.For(
            target=store_("__inv_k"),
            iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
            body=k_body,
            orelse=[],
        )
    )
    return out


def expand_linalg_det(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    local_dtypes: dict[str, str] | None = None,
    fresh_local_allocs: dict[str, tuple[str, ...]] | None = None,
) -> list[ast.stmt]:
    """``s = np.linalg.det(A)`` -- LU factorisation (Gaussian elimination
    with partial pivoting) on a scratch copy ``__det_aw`` of A. The
    determinant is the product of the pivots (U's diagonal) times
    ``(-1) ** (# row swaps)``.

    ``target`` is a scalar (the hoister lifts a nested ``np.linalg.det``
    to a fresh scalar temp before this fires). Conservative: ``A`` must be
    a Name with a known square 2-D shape; ``A`` is preserved -- the
    elimination runs on the copy ``__det_aw``.
    """
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.linalg.det needs Name first arg")
    if not isinstance(target, ast.Name):
        raise NotImplementedError("np.linalg.det: scalar Name target expected")
    a = args[0]
    shape = shape_table.get(a.id)
    if not shape or len(shape) != 2:
        raise NotImplementedError("np.linalg.det: only 2-D square input supported")
    n = shape[0]
    n_ast = const_or_name(n)
    out: list[ast.stmt] = []
    # Publish the working buffer's shape + dtype + fresh-local alloc so the
    # emit declares ``__det_aw`` as a flat 2-D buffer of A's element dtype
    # (same registration logic as ``expand_linalg_inv``).
    shape_table["__det_aw"] = (n, n)
    a_dt = None
    if local_dtypes is not None:
        a_dt = local_dtypes.get(a.id)
        if a_dt is not None:
            local_dtypes["__det_aw"] = a_dt
            for nm in ("__det_tmp", "__det_factor"):
                local_dtypes.setdefault(nm, a_dt)
            # The determinant of a complex matrix is complex; keep the
            # accumulator's dtype aligned with A's element dtype.
            local_dtypes.setdefault(target.id, a_dt)
    if fresh_local_allocs is not None:
        fresh_local_allocs["__det_aw"] = (n, n)
    aw = lambda r, c: ast.Subscript(
        value=name_("__det_aw"), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Load()
    )
    aw_store = lambda r, c: ast.Subscript(
        value=name_("__det_aw"), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Store()
    )
    out.append(
        ast.Assign(
            targets=[store_("__det_aw")], value=ast.Call(func=name_("__hpcagent_bench_zeros__"), args=[], keywords=[])
        )
    )
    # Copy A into the working buffer.
    out.append(
        ast.For(
            target=store_("__det_i"),
            iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
            body=[
                ast.For(
                    target=store_("__det_j"),
                    iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
                    body=[
                        ast.Assign(
                            targets=[aw_store(name_("__det_i"), name_("__det_j"))],
                            value=ast.Subscript(
                                value=name_(a.id),
                                slice=ast.Tuple(elts=[name_("__det_i"), name_("__det_j")], ctx=ast.Load()),
                                ctx=ast.Load(),
                            ),
                        )
                    ],
                    orelse=[],
                )
            ],
            orelse=[],
        )
    )
    # Accumulator starts at 1 (product of pivots * swap sign).
    out.append(ast.Assign(targets=[store_(target.id)], value=const_(1.0)))
    K = name_("__det_k")
    P = name_("__det_p")
    R = name_("__det_r")
    C = name_("__det_c")
    F = name_("__det_factor")
    T = name_("__det_tmp")
    # Pivot search: p = argmax_{r >= k} |aw[r, k]|.
    pivot_init = ast.Assign(targets=[store_("__det_p")], value=K)
    pivot_scan = ast.For(
        target=store_("__det_r"),
        iter=ast.Call(func=name_("range"), args=[ast.BinOp(left=K, op=ast.Add(), right=const_(1)), n_ast], keywords=[]),
        body=[
            ast.If(
                test=ast.Compare(
                    left=ast.Call(func=name_("abs"), args=[aw(R, K)], keywords=[]),
                    ops=[ast.Gt()],
                    comparators=[ast.Call(func=name_("abs"), args=[aw(P, K)], keywords=[])],
                ),
                body=[ast.Assign(targets=[store_("__det_p")], value=R)],
                orelse=[],
            )
        ],
        orelse=[],
    )
    # Swap rows p and k (when distinct) and flip the running sign.
    swap_body = [
        ast.Assign(targets=[store_("__det_tmp")], value=aw(K, C)),
        ast.Assign(targets=[aw_store(K, C)], value=aw(P, C)),
        ast.Assign(targets=[aw_store(P, C)], value=T),
    ]
    swap_loop = ast.For(
        target=store_("__det_c"),
        iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
        body=swap_body,
        orelse=[],
    )
    sign_flip = ast.Assign(targets=[store_(target.id)], value=ast.UnaryOp(op=ast.USub(), operand=name_(target.id)))
    swap_if = ast.If(
        test=ast.Compare(left=P, ops=[ast.NotEq()], comparators=[K]), body=[swap_loop, sign_flip], orelse=[]
    )
    # Multiply the determinant by this pivot.
    pivot_mul = ast.Assign(
        targets=[store_(target.id)], value=ast.BinOp(left=name_(target.id), op=ast.Mult(), right=aw(K, K))
    )
    # Eliminate rows below k (guard a zero pivot so a singular matrix
    # yields det == 0 rather than a NaN from divide-by-zero).
    elim_factor = ast.Assign(
        targets=[store_("__det_factor")], value=ast.BinOp(left=aw(R, K), op=ast.Div(), right=aw(K, K))
    )
    elim_inner = ast.For(
        target=store_("__det_c"),
        iter=ast.Call(func=name_("range"), args=[ast.BinOp(left=K, op=ast.Add(), right=const_(1)), n_ast], keywords=[]),
        body=[
            ast.Assign(
                targets=[aw_store(R, C)],
                value=ast.BinOp(left=aw(R, C), op=ast.Sub(), right=ast.BinOp(left=F, op=ast.Mult(), right=aw(K, C))),
            )
        ],
        orelse=[],
    )
    elim_outer = ast.For(
        target=store_("__det_r"),
        iter=ast.Call(func=name_("range"), args=[ast.BinOp(left=K, op=ast.Add(), right=const_(1)), n_ast], keywords=[]),
        body=[elim_factor, elim_inner],
        orelse=[],
    )
    elim_guard = ast.If(
        test=ast.Compare(
            left=ast.Call(func=name_("abs"), args=[aw(K, K)], keywords=[]), ops=[ast.Gt()], comparators=[const_(0.0)]
        ),
        body=[elim_outer],
        orelse=[],
    )
    k_body = [pivot_init, pivot_scan, swap_if, pivot_mul, elim_guard]
    out.append(
        ast.For(
            target=store_("__det_k"),
            iter=ast.Call(func=name_("range"), args=[n_ast], keywords=[]),
            body=k_body,
            orelse=[],
        )
    )
    return out
