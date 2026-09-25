"""Diagonal and triangular views: triu, tril, trace, diagonal, diag."""

import ast

from hpcagent_bench.translators.numpyto_common.lib_nodes.call_args import kwarg_or_pos
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of_
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import const_, const_int, name_, store_, wrap_for_loops
from hpcagent_bench.translators.numpyto_common.lib_nodes.scalarize import scalarize_at_iters


def expand_triangular(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None,
    lower: bool,
) -> list[ast.stmt]:
    """Shared triu/tril lowering: copy ``A[i, j]`` where it's on the kept side of
    the ``i + k`` diagonal, else 0. ``lower=False`` keeps ``j >= i + k`` (upper);
    ``lower=True`` keeps ``j <= i + k``. Optional ``k`` offset (positional or
    ``k=``) defaults to 0."""
    name = "np.tril" if lower else "np.triu"
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError(f"{name} needs Name first arg")
    a = args[0]
    shape = shape_table.get(a.id)
    if not shape or len(shape) != 2:
        raise NotImplementedError(f"{name}: only 2-D supported")
    m, n = shape
    k_arg: ast.expr = const_(0)
    k_ = kwarg_or_pos(args, kwargs, 1, "k")
    if k_ is not None:
        k_arg = k_
    a_sub = ast.Subscript(
        value=name_(a.id), slice=ast.Tuple(elts=[name_("__i"), name_("__j")], ctx=ast.Load()), ctx=ast.Load()
    )
    if isinstance(k_arg, ast.Constant) and k_arg.value == 0:
        threshold: ast.expr = name_("__i")
    else:
        threshold = ast.BinOp(left=name_("__i"), op=ast.Add(), right=k_arg)
    cmp_op = ast.LtE() if lower else ast.GtE()
    body = [
        ast.Assign(
            targets=[
                ast.Subscript(
                    value=name_(target.id),
                    slice=ast.Tuple(elts=[name_("__i"), name_("__j")], ctx=ast.Load()),
                    ctx=ast.Store(),
                )
            ],
            value=ast.IfExp(
                test=ast.Compare(left=name_("__j"), ops=[cmp_op], comparators=[threshold]),
                body=a_sub,
                orelse=const_(0.0),
            ),
        )
    ]
    return wrap_for_loops(["__i", "__j"], (m, n), body)


def expand_triu(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    """``out = np.triu(A [, k])`` -> ``out[i, j] = A[i, j] if j >= i+k else 0``.
    Optional ``k`` offset (default 0) selects the diagonal; ``k=1`` skips the
    main diagonal (strict upper-triangular).
    """
    return expand_triangular(target, args, shape_table, kwargs, lower=False)


def expand_trace(target: ast.expr, args: list[ast.expr], shape_table: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    """``np.trace(A)`` -> ``sum_i A[i, i]`` (the diagonal sum)."""
    if len(args) != 1 or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.trace needs one bare-Name 2-D arg")
    shape = shape_table.get(args[0].id)
    if shape is None or len(shape) != 2:
        raise NotImplementedError("np.trace needs a 2-D array")
    it = "__tr"
    diag = ast.Subscript(
        value=name_(args[0].id), slice=ast.Tuple(elts=[name_(it), name_(it)], ctx=ast.Load()), ctx=ast.Load()
    )
    body = [ast.AugAssign(target=store_(target.id), op=ast.Add(), value=diag)]
    return [ast.Assign(targets=[store_(target.id)], value=const_(0.0)), *wrap_for_loops([it], [shape[0]], body)]


def expand_diagonal(target: ast.expr, args: list[ast.expr], shape_table: dict[str, tuple[str, ...]]) -> list[ast.stmt]:
    """``np.diagonal(A)`` -> ``out[i] = A[i, i]``."""
    if len(args) != 1 or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.diagonal needs one bare-Name 2-D arg")
    shape = shape_table.get(args[0].id)
    if shape is None or len(shape) != 2:
        raise NotImplementedError("np.diagonal needs a 2-D array")
    it = "__dg"
    diag = ast.Subscript(
        value=name_(args[0].id), slice=ast.Tuple(elts=[name_(it), name_(it)], ctx=ast.Load()), ctx=ast.Load()
    )
    body = [ast.Assign(targets=[ast.Subscript(value=name_(target.id), slice=name_(it), ctx=ast.Store())], value=diag)]
    return wrap_for_loops([it], [shape[0]], body)


def expand_diag(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    """``np.diag(v [, k])`` -- construct a diagonal matrix, or extract one. 1-D
    ``v`` of shape ``(n,)`` -> an ``(n+|k|, n+|k|)`` matrix, all zeros except
    ``out[i, i+k] = v[i]`` for ``k >= 0`` (``out[i-k, i] = v[i]`` for ``k < 0``),
    matching ``numpy.diag``. 2-D operand -> extract the main diagonal (delegates
    to :func:`expand_diagonal`). ``k`` must be a constant int (sizes the
    result). The matrix is zeroed first, then the diagonal written, so no
    out-of-range read occurs."""
    if not args:
        raise NotImplementedError("np.diag needs an operand")
    v = args[0]
    ext = iter_extent_of_(v, shape_table)
    if ext is None:
        raise NotImplementedError("np.diag: operand shape unknown")
    if len(ext) == 2:
        return expand_diagonal(target, args, shape_table)  # extract-diagonal
    if len(ext) != 1:
        raise NotImplementedError("np.diag: only 1-D / 2-D operands supported")
    k_node = kwarg_or_pos(args, kwargs, 1, "k")
    if k_node is None:
        k = 0
    else:
        k = const_int(k_node)
        if k is None:
            raise NotImplementedError("np.diag: offset k must be a constant int")
    n_tok = ast.unparse(ext[0])
    side_tok = n_tok if k == 0 else f"({n_tok}) + {abs(k)}"
    # Zero the whole (side x side) matrix.
    zero_body = [
        ast.Assign(
            targets=[
                ast.Subscript(
                    value=name_(target.id),
                    slice=ast.Tuple(elts=[name_("__dg_zi"), name_("__dg_zj")], ctx=ast.Load()),
                    ctx=ast.Store(),
                )
            ],
            value=const_(0.0),
        )
    ]
    zero_loops = wrap_for_loops(["__dg_zi", "__dg_zj"], [side_tok, side_tok], zero_body)
    # Write v along the k-th diagonal.
    it = "__dg_i"
    v_elem = scalarize_at_iters(v, [name_(it)], shape_table)
    if k >= 0:
        row: ast.expr = name_(it)
        col: ast.expr = name_(it) if k == 0 else ast.BinOp(left=name_(it), op=ast.Add(), right=const_(k))
    else:
        row = ast.BinOp(left=name_(it), op=ast.Add(), right=const_(-k))
        col = name_(it)
    set_body = [
        ast.Assign(
            targets=[
                ast.Subscript(value=name_(target.id), slice=ast.Tuple(elts=[row, col], ctx=ast.Load()), ctx=ast.Store())
            ],
            value=v_elem,
        )
    ]
    return zero_loops + wrap_for_loops([it], [n_tok], set_body)


def expand_tril(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    """``np.tril(A, k=0)`` -> lower-triangular copy (zero where ``j > i + k``).

    Mirrors :func:`expand_triu` with the complementary mask."""
    return expand_triangular(target, args, shape_table, kwargs, lower=True)
