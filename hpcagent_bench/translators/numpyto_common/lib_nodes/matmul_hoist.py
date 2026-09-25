"""Hoist ``A @ B`` subexpressions into temporaries the matmul expander can lower."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.lib_nodes.blas import BLAS_GEMM_MARKER, BLAS_INELIGIBLE_DTYPES
from hpcagent_bench.translators.numpyto_common.lib_nodes.dims import static_shape_of, dims_agree
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of_
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import (
    alloc_marker,
    const_,
    const_or_name,
    name_,
    store_,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.scalarize import scalarize_at_iters


def matmul_result_shape(
    a_shape: tuple[str, ...],
    b_shape: tuple[str, ...],
    dim_aliases: dict[str, str] | None = None,
    shape_table: dict[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...] | None:
    """``A @ B``'s result shape under numpy broadcasting rules. Supports 1-D x
    2-D / 2-D x 1-D / 2-D x 2-D; batched ``(*batch, m, k) @ (k, n) -> (*batch,
    m, n)`` (rank(a) >= 3, rank(b) == 2) and its mirror (rank(a) == 2, rank(b)
    >= 3); and both-batched ``(*batch, m, k) @ (*batch, k, n) -> (*batch, m,
    n)`` when both ranks are >= 3 and share the same leading batch dims.

    Dimension agreement goes through :func:`dims_agree` rather than ``==``: the two operands'
    tokens come from different vocabularies (a body local vs an ``init.shapes`` symbol), so string
    identity declines contractions whose extents match. ``dim_aliases`` is what reconciles them.
    """
    # Normalise first: a PARAMETER's shape arrives as a list and a hoisted temp's as a tuple, so the
    # ``a_shape[:-2] == b_shape[:-2]`` batch test below was comparing a list against a tuple and
    # always answering False -- every batched matmul mixing the two was silently declined.
    a_shape, b_shape = tuple(a_shape), tuple(b_shape)

    def agree(x: str, y: str) -> bool:
        return dims_agree(str(x), str(y), dim_aliases, shape_table)

    if len(a_shape) == 2 and len(b_shape) == 2:
        return (a_shape[0], b_shape[1])
    if len(a_shape) == 2 and len(b_shape) == 1:
        return (a_shape[0],)
    if len(a_shape) == 1 and len(b_shape) == 2:
        return (b_shape[1],)
    if len(a_shape) >= 3 and len(b_shape) >= 3:
        # (*batch, m, k) @ (*batch, k, n) -> (*batch, m, n): identical batch.
        batch_ok = len(a_shape) == len(b_shape) and all(agree(x, y) for x, y in zip(a_shape[:-2], b_shape[:-2]))
        if batch_ok and agree(a_shape[-1], b_shape[-2]):
            return tuple(a_shape[:-2]) + (a_shape[-2], b_shape[-1])
        return None
    if len(a_shape) >= 3 and len(b_shape) == 2:
        # (*batch, m, k) @ (k, n) -> (*batch, m, n)
        if agree(a_shape[-1], b_shape[0]):
            return tuple(a_shape[:-1]) + (b_shape[1],)
    if len(a_shape) == 2 and len(b_shape) >= 3:
        # (m, k) @ (*batch, k, n) -> (*batch, m, n)
        if agree(a_shape[1], b_shape[-2]):
            return tuple(b_shape[:-2]) + (a_shape[0], b_shape[-1])
    return None


def hoist_matmul(
    matmul: ast.BinOp,
    shape_table: dict[str, tuple[str, ...]],
    temp_arrays: dict[str, tuple[str, ...]],
    temp_counter: list[int],
    dim_aliases: dict[str, str] | None = None,
    blas: bool = False,
) -> tuple[str | None, list[ast.stmt]]:
    """Hoist a ``lhs @ rhs`` subexpression to a fresh temp array. Returns
    ``(temp_name, pre_stmts)``: caller substitutes ``temp_name`` for the matmul
    expression and prepends ``pre_stmts`` before the enclosing assignment.
    ``None`` signals an unsupported form (caller falls through to
    ``NotImplementedError``).

    Handles slice operands ``A[i, :j] @ A[:j, j]`` by lowering to a scalar
    accumulator loop (dot-product form) when both operands have 1-D iteration
    extent.
    """
    # Slice-aware matmuls via iteration extent, three forms:
    #   1-D x 1-D -> scalar dot (e.g. ``A[i, :j] @ A[:j, j]``).
    #   1-D x 2-D -> 1-D vector ``out[j] = sum_l a[l] * b[l, j]``.
    #   2-D x 1-D -> 1-D vector ``out[i] = sum_l a[i, l] * b[l]``.
    l_ext = iter_extent_of_(matmul.left, shape_table)
    r_ext = iter_extent_of_(matmul.right, shape_table)
    if l_ext is not None and r_ext is not None and len(l_ext) == 1 and len(r_ext) == 1:
        temp_counter[0] += 1
        temp = f"__mm{temp_counter[0]}"
        # Scalar temp -- caller declares it as ``double``.
        iter_var = f"__mml{temp_counter[0]}"
        sa = scalarize_at_iters(matmul.left, [name_(iter_var)], shape_table)
        sb = scalarize_at_iters(matmul.right, [name_(iter_var)], shape_table)
        stmts = [
            ast.Assign(targets=[store_(temp)], value=const_(0.0)),
            ast.For(
                target=store_(iter_var),
                iter=ast.Call(func=name_("range"), args=[l_ext[0]], keywords=[]),
                body=[
                    ast.AugAssign(target=store_(temp), op=ast.Add(), value=ast.BinOp(left=sa, op=ast.Mult(), right=sb))
                ],
                orelse=[],
            ),
        ]
        return temp, stmts
    # 1-D x 2-D / 2-D x 1-D slice-form matmul (matrix-vector).
    if l_ext is not None and r_ext is not None and {len(l_ext), len(r_ext)} == {1, 2}:
        temp_counter[0] += 1
        temp = f"__mm{temp_counter[0]}"
        # Output is 1-D; the shared K axis is the matching extent.
        if len(l_ext) == 1:  # 1-D x 2-D: out[j] = sum_l a[l] * b[l, j]
            k_extent, n_extent = l_ext[0], r_ext[1]
            # Use the FULL extent of the RHS array as the temp shape so
            # the function-scope declaration doesn't depend on a loop
            # variable. The actual iteration uses the dynamic extent.
            shape = (static_shape_of(matmul.right, 1, shape_table) or ast.unparse(n_extent),)
            temp_arrays[temp] = shape
            shape_table[temp] = shape
            l_iter = name_(f"__mml{temp_counter[0]}")  # k
            out_iter = name_(f"__mmj{temp_counter[0]}")  # j
            sa = scalarize_at_iters(matmul.left, [l_iter], shape_table)
            sb = scalarize_at_iters(matmul.right, [l_iter, out_iter], shape_table)
            stmts = [
                ast.For(
                    target=store_(out_iter.id),
                    iter=ast.Call(func=name_("range"), args=[n_extent], keywords=[]),
                    body=[
                        ast.Assign(
                            targets=[ast.Subscript(value=name_(temp), slice=out_iter, ctx=ast.Store())],
                            value=const_(0.0),
                        ),
                        ast.For(
                            target=store_(l_iter.id),
                            iter=ast.Call(func=name_("range"), args=[k_extent], keywords=[]),
                            body=[
                                ast.AugAssign(
                                    target=ast.Subscript(value=name_(temp), slice=out_iter, ctx=ast.Store()),
                                    op=ast.Add(),
                                    value=ast.BinOp(left=sa, op=ast.Mult(), right=sb),
                                )
                            ],
                            orelse=[],
                        ),
                    ],
                    orelse=[],
                )
            ]
        else:  # 2-D x 1-D
            m_extent, k_extent = l_ext[0], l_ext[1]
            shape = (static_shape_of(matmul.left, 0, shape_table) or ast.unparse(m_extent),)
            temp_arrays[temp] = shape
            shape_table[temp] = shape
            out_iter = name_(f"__mmi{temp_counter[0]}")
            l_iter = name_(f"__mml{temp_counter[0]}")
            sa = scalarize_at_iters(matmul.left, [out_iter, l_iter], shape_table)
            sb = scalarize_at_iters(matmul.right, [l_iter], shape_table)
            stmts = [
                ast.For(
                    target=store_(out_iter.id),
                    iter=ast.Call(func=name_("range"), args=[m_extent], keywords=[]),
                    body=[
                        ast.Assign(
                            targets=[ast.Subscript(value=name_(temp), slice=out_iter, ctx=ast.Store())],
                            value=const_(0.0),
                        ),
                        ast.For(
                            target=store_(l_iter.id),
                            iter=ast.Call(func=name_("range"), args=[k_extent], keywords=[]),
                            body=[
                                ast.AugAssign(
                                    target=ast.Subscript(value=name_(temp), slice=out_iter, ctx=ast.Store()),
                                    op=ast.Add(),
                                    value=ast.BinOp(left=sa, op=ast.Mult(), right=sb),
                                )
                            ],
                            orelse=[],
                        ),
                    ],
                    orelse=[],
                )
            ]
        return temp, stmts
    # 2-D x 2-D scalarised form: either operand may be a BinOp /
    # Subscript expression instead of a bare Name. Recover their iter
    # extents and scalarise at the matmul loop indices (i, l) / (l, j).
    if (
        l_ext is not None
        and r_ext is not None
        and len(l_ext) == 2
        and len(r_ext) == 2
        and not (isinstance(matmul.left, ast.Name) and isinstance(matmul.right, ast.Name))
    ):
        temp_counter[0] += 1
        temp = f"__mm{temp_counter[0]}"
        m_extent, k_extent = l_ext[0], l_ext[1]
        unused, n_extent = r_ext
        shape = (
            static_shape_of(matmul.left, 0, shape_table) or ast.unparse(m_extent),
            static_shape_of(matmul.right, 1, shape_table) or ast.unparse(n_extent),
        )
        temp_arrays[temp] = shape
        shape_table[temp] = shape
        i_iter = name_(f"__mmi{temp_counter[0]}")
        j_iter = name_(f"__mmj{temp_counter[0]}")
        l_iter = name_(f"__mml{temp_counter[0]}")
        sa = scalarize_at_iters(matmul.left, [i_iter, l_iter], shape_table)
        sb = scalarize_at_iters(matmul.right, [l_iter, j_iter], shape_table)
        out_sub = ast.Tuple(elts=[i_iter, j_iter], ctx=ast.Load())
        stmts = [
            ast.For(
                target=store_(i_iter.id),
                iter=ast.Call(func=name_("range"), args=[m_extent], keywords=[]),
                body=[
                    ast.For(
                        target=store_(j_iter.id),
                        iter=ast.Call(func=name_("range"), args=[n_extent], keywords=[]),
                        body=[
                            ast.Assign(
                                targets=[ast.Subscript(value=name_(temp), slice=out_sub, ctx=ast.Store())],
                                value=const_(0.0),
                            ),
                            ast.For(
                                target=store_(l_iter.id),
                                iter=ast.Call(func=name_("range"), args=[k_extent], keywords=[]),
                                body=[
                                    ast.AugAssign(
                                        target=ast.Subscript(value=name_(temp), slice=out_sub, ctx=ast.Store()),
                                        op=ast.Add(),
                                        value=ast.BinOp(left=sa, op=ast.Mult(), right=sb),
                                    )
                                ],
                                orelse=[],
                            ),
                        ],
                        orelse=[],
                    )
                ],
                orelse=[],
            )
        ]
        return temp, stmts
    # BATCHED scalarised form -- the batched counterpart of the 2-D x 2-D branch above. The
    # bare-Name batched path further down reads both operands' declared shapes, so it declines the
    # moment one is an expression: conv_transpose2d's ``xg_flat @ wg[:, :, ky, kx]`` is
    # (n, h*w, in_per_group) @ (in_per_group, out_per_group), and declining it left the contraction
    # to slice fusion, which refuses. Extents come from ``iter_extent_of_`` here, which reads
    # through the slice, so the operand's spelling stops mattering.
    if (
        l_ext is not None
        and r_ext is not None
        and max(len(l_ext), len(r_ext)) >= 3
        and min(len(l_ext), len(r_ext)) >= 2
        and not (isinstance(matmul.left, ast.Name) and isinstance(matmul.right, ast.Name))
    ):
        # Both operands batched must agree on the batch RANK: numpy would broadcast a mismatch,
        # and the bare-Name path below does not model that either. Refuse rather than guess.
        if len(l_ext) >= 3 and len(r_ext) >= 3 and len(l_ext) != len(r_ext):
            return None, []
        temp_counter[0] += 1
        temp = f"__mm{temp_counter[0]}"
        ctr = temp_counter[0]
        batch_ext = (l_ext if len(l_ext) >= len(r_ext) else r_ext)[:-2]
        batched = matmul.left if len(l_ext) >= len(r_ext) else matmul.right
        m_extent, k_extent = l_ext[-2], l_ext[-1]
        n_extent = r_ext[-1]
        # Declare the temp from STATIC axis tokens where they exist, so the function-scope
        # declaration never names a loop variable (same rule as the branches above).
        shape = tuple(
            static_shape_of(batched, axis, shape_table) or ast.unparse(ext) for axis, ext in enumerate(batch_ext)
        ) + (
            static_shape_of(matmul.left, len(l_ext) - 2, shape_table) or ast.unparse(m_extent),
            static_shape_of(matmul.right, len(r_ext) - 1, shape_table) or ast.unparse(n_extent),
        )
        temp_arrays[temp] = shape
        shape_table[temp] = shape
        batch_iters = [name_(f"__mmb{ctr}_{i}") for i in range(len(batch_ext))]
        i_iter = name_(f"__mmi{ctr}")
        j_iter = name_(f"__mmj{ctr}")
        l_iter = name_(f"__mml{ctr}")
        left_iters = ([*batch_iters] if len(l_ext) >= 3 else []) + [i_iter, l_iter]
        right_iters = ([*batch_iters] if len(r_ext) >= 3 else []) + [l_iter, j_iter]
        sa = scalarize_at_iters(matmul.left, left_iters, shape_table)
        sb = scalarize_at_iters(matmul.right, right_iters, shape_table)
        out_sub = ast.Tuple(elts=[*batch_iters, i_iter, j_iter], ctx=ast.Load())
        out_ref = lambda ctx: ast.Subscript(value=name_(temp), slice=copy.deepcopy(out_sub), ctx=ctx)
        body: list[ast.stmt] = [
            ast.For(
                target=store_(j_iter.id),
                iter=ast.Call(func=name_("range"), args=[n_extent], keywords=[]),
                body=[
                    ast.Assign(targets=[out_ref(ast.Store())], value=const_(0.0)),
                    ast.For(
                        target=store_(l_iter.id),
                        iter=ast.Call(func=name_("range"), args=[k_extent], keywords=[]),
                        body=[
                            ast.AugAssign(
                                target=out_ref(ast.Store()),
                                op=ast.Add(),
                                value=ast.BinOp(left=sa, op=ast.Mult(), right=sb),
                            )
                        ],
                        orelse=[],
                    ),
                ],
                orelse=[],
            )
        ]
        body = [
            ast.For(
                target=store_(i_iter.id),
                iter=ast.Call(func=name_("range"), args=[m_extent], keywords=[]),
                body=body,
                orelse=[],
            )
        ]
        for iter_node, ext in zip(reversed(batch_iters), reversed(batch_ext)):
            body = [
                ast.For(
                    target=store_(iter_node.id),
                    iter=ast.Call(func=name_("range"), args=[ext], keywords=[]),
                    body=body,
                    orelse=[],
                )
            ]
        return temp, body
    if not (isinstance(matmul.left, ast.Name) and isinstance(matmul.right, ast.Name)):
        return None, []
    a_name, b_name = matmul.left.id, matmul.right.id
    a_shape = shape_table.get(a_name)
    b_shape = shape_table.get(b_name)
    if not a_shape or not b_shape:
        return None, []
    result_shape = matmul_result_shape(a_shape, b_shape, dim_aliases, shape_table)
    if result_shape is None:
        return None, []

    temp_counter[0] += 1
    temp = f"__mm{temp_counter[0]}"
    temp_arrays[temp] = result_shape
    shape_table[temp] = result_shape

    # Batched matmul ``(*batch, m, k) @ (k, n) -> (*batch, m, n)``: wrap a plain
    # 2-D matmul body in a loop nest over the batch dims, indexing the LHS by
    # ``[*batch, m, k]`` and writing the temp by ``[*batch, m, n]``. Same shape
    # for ``(m, k) @ (*batch, k, n)``.
    if (
        (len(a_shape) >= 3 and len(b_shape) == 2)
        or (len(a_shape) == 2 and len(b_shape) >= 3)
        or (len(a_shape) >= 3 and len(b_shape) >= 3)
    ):
        if not (isinstance(matmul.left, ast.Name) and isinstance(matmul.right, ast.Name)):
            return None, []
        a_name_b, b_name_b = matmul.left.id, matmul.right.id
        ctr = temp_counter[0]
        # Which side(s) carry the batch dims. Both-batched broadcasts the SAME
        # batch index into both operands; one-sided indexes only that operand.
        a_batch = len(a_shape) >= 3
        b_batch = len(b_shape) >= 3
        if a_batch:
            batch_shape = a_shape[:-2]
            m, k = a_shape[-2], a_shape[-1]
            n = b_shape[-1]
        else:
            batch_shape = b_shape[:-2]
            m, k = a_shape
            n = b_shape[-1]
        batch_iters = [f"__mmb{ctr}_{i}" for i in range(len(batch_shape))]
        i_iter, j_iter, l_iter = f"__mmi{ctr}", f"__mmj{ctr}", f"__mml{ctr}"
        batch_names = [name_(b) for b in batch_iters]
        # Each operand's subscript is prefixed with the batch iters iff that
        # operand is batched; the output is always batched.
        a_sub_elts = (batch_names if a_batch else []) + [name_(i_iter), name_(l_iter)]
        b_sub_elts = (batch_names if b_batch else []) + [name_(l_iter), name_(j_iter)]
        out_sub_elts = batch_names + [name_(i_iter), name_(j_iter)]
        out_sub = ast.Tuple(elts=out_sub_elts, ctx=ast.Load())
        a_sub = ast.Tuple(elts=a_sub_elts, ctx=ast.Load()) if len(a_sub_elts) > 1 else a_sub_elts[0]
        b_sub = ast.Tuple(elts=b_sub_elts, ctx=ast.Load()) if len(b_sub_elts) > 1 else b_sub_elts[0]
        # Innermost: out[*batch, i, j] = 0; for l: out += a[*] * b[*].
        zero_assign = ast.Assign(
            targets=[ast.Subscript(value=name_(temp), slice=out_sub, ctx=ast.Store())], value=const_(0.0)
        )
        accum = ast.AugAssign(
            target=ast.Subscript(value=name_(temp), slice=out_sub, ctx=ast.Store()),
            op=ast.Add(),
            value=ast.BinOp(
                left=ast.Subscript(value=name_(a_name_b), slice=a_sub, ctx=ast.Load()),
                op=ast.Mult(),
                right=ast.Subscript(value=name_(b_name_b), slice=b_sub, ctx=ast.Load()),
            ),
        )
        l_loop = ast.For(
            target=store_(l_iter),
            iter=ast.Call(func=name_("range"), args=[const_or_name(k)], keywords=[]),
            body=[accum],
            orelse=[],
        )
        j_loop = ast.For(
            target=store_(j_iter),
            iter=ast.Call(func=name_("range"), args=[const_or_name(n)], keywords=[]),
            body=[zero_assign, l_loop],
            orelse=[],
        )
        i_loop = ast.For(
            target=store_(i_iter),
            iter=ast.Call(func=name_("range"), args=[const_or_name(m)], keywords=[]),
            body=[j_loop],
            orelse=[],
        )
        # Wrap with the batch loops, outermost first.
        current: ast.stmt = i_loop
        for bi, bdim in zip(reversed(batch_iters), reversed(list(batch_shape))):
            current = ast.For(
                target=store_(bi),
                iter=ast.Call(func=name_("range"), args=[const_or_name(bdim)], keywords=[]),
                body=[current],
                orelse=[],
            )
        return temp, [current]

    # Emit the matmul loop nest that fills ``temp``.
    stmts: list[ast.stmt] = []
    if len(a_shape) == 2 and len(b_shape) == 2:
        m, k = a_shape
        unused, n = b_shape
        if blas:
            # A bare Expr, NOT an assignment to the temp: an ``Assign`` whose target is an array is
            # what the later elementwise/slice passes wrap in an iteration nest, which turned the
            # gemm into a per-element call indexing its own operands. The allocation marker ahead of
            # it carries what the assignment was needed for -- it stores to ``temp``, so the buffer
            # is malloc'd and the name is never mistaken for a free one to promote to a parameter.
            gemm = ast.Call(
                func=name_(BLAS_GEMM_MARKER),
                args=[
                    name_(a_name),
                    name_(b_name),
                    name_(temp),
                    const_or_name(m),
                    const_or_name(n),
                    const_or_name(k),
                ],
                keywords=[],
            )
            return temp, [alloc_marker(temp), ast.Expr(value=gemm)]
        stmts.append(
            ast.For(
                target=store_("__i"),
                iter=ast.Call(func=name_("range"), args=[const_or_name(m)], keywords=[]),
                body=[
                    ast.For(
                        target=store_("__j"),
                        iter=ast.Call(func=name_("range"), args=[const_or_name(n)], keywords=[]),
                        body=[
                            ast.Assign(
                                targets=[
                                    ast.Subscript(
                                        value=name_(temp),
                                        slice=ast.Tuple(elts=[name_("__i"), name_("__j")], ctx=ast.Load()),
                                        ctx=ast.Store(),
                                    )
                                ],
                                value=const_(0.0),
                            ),
                            ast.For(
                                target=store_("__l"),
                                iter=ast.Call(func=name_("range"), args=[const_or_name(k)], keywords=[]),
                                body=[
                                    ast.AugAssign(
                                        target=ast.Subscript(
                                            value=name_(temp),
                                            slice=ast.Tuple(elts=[name_("__i"), name_("__j")], ctx=ast.Load()),
                                            ctx=ast.Store(),
                                        ),
                                        op=ast.Add(),
                                        value=ast.BinOp(
                                            left=ast.Subscript(
                                                value=name_(a_name),
                                                slice=ast.Tuple(elts=[name_("__i"), name_("__l")], ctx=ast.Load()),
                                                ctx=ast.Load(),
                                            ),
                                            op=ast.Mult(),
                                            right=ast.Subscript(
                                                value=name_(b_name),
                                                slice=ast.Tuple(elts=[name_("__l"), name_("__j")], ctx=ast.Load()),
                                                ctx=ast.Load(),
                                            ),
                                        ),
                                    )
                                ],
                                orelse=[],
                            ),
                        ],
                        orelse=[],
                    )
                ],
                orelse=[],
            )
        )
    elif len(a_shape) == 2 and len(b_shape) == 1:
        m, k = a_shape
        stmts.append(
            ast.For(
                target=store_("__i"),
                iter=ast.Call(func=name_("range"), args=[const_or_name(m)], keywords=[]),
                body=[
                    ast.Assign(
                        targets=[ast.Subscript(value=name_(temp), slice=name_("__i"), ctx=ast.Store())],
                        value=const_(0.0),
                    ),
                    ast.For(
                        target=store_("__l"),
                        iter=ast.Call(func=name_("range"), args=[const_or_name(k)], keywords=[]),
                        body=[
                            ast.AugAssign(
                                target=ast.Subscript(value=name_(temp), slice=name_("__i"), ctx=ast.Store()),
                                op=ast.Add(),
                                value=ast.BinOp(
                                    left=ast.Subscript(
                                        value=name_(a_name),
                                        slice=ast.Tuple(elts=[name_("__i"), name_("__l")], ctx=ast.Load()),
                                        ctx=ast.Load(),
                                    ),
                                    op=ast.Mult(),
                                    right=ast.Subscript(value=name_(b_name), slice=name_("__l"), ctx=ast.Load()),
                                ),
                            )
                        ],
                        orelse=[],
                    ),
                ],
                orelse=[],
            )
        )
    else:  # len(a)==1, len(b)==2
        k, n = b_shape
        stmts.append(
            ast.For(
                target=store_("__j"),
                iter=ast.Call(func=name_("range"), args=[const_or_name(n)], keywords=[]),
                body=[
                    ast.Assign(
                        targets=[ast.Subscript(value=name_(temp), slice=name_("__j"), ctx=ast.Store())],
                        value=const_(0.0),
                    ),
                    ast.For(
                        target=store_("__l"),
                        iter=ast.Call(func=name_("range"), args=[const_or_name(k)], keywords=[]),
                        body=[
                            ast.AugAssign(
                                target=ast.Subscript(value=name_(temp), slice=name_("__j"), ctx=ast.Store()),
                                op=ast.Add(),
                                value=ast.BinOp(
                                    left=ast.Subscript(value=name_(a_name), slice=name_("__l"), ctx=ast.Load()),
                                    op=ast.Mult(),
                                    right=ast.Subscript(
                                        value=name_(b_name),
                                        slice=ast.Tuple(elts=[name_("__l"), name_("__j")], ctx=ast.Load()),
                                        ctx=ast.Load(),
                                    ),
                                ),
                            )
                        ],
                        orelse=[],
                    ),
                ],
                orelse=[],
            )
        )
    return temp, stmts


class MatmulHoister(ast.NodeTransformer):
    """Replace ``A @ B`` subexpressions with a fresh temp Name and record the
    matmul loop nest that fills the temp. Multiple matmuls in one expression
    each get their own temp (chained ``A @ B @ C`` lifts to two temps fused
    left-to-right)."""

    def __init__(
        self,
        shape_table: dict[str, tuple[str, ...]],
        temp_arrays: dict[str, tuple[str, ...]],
        temp_counter: list[int],
        local_dtypes: dict[str, str] | None = None,
        sparse: dict[str, object] | None = None,
        dim_aliases: dict[str, str] | None = None,
        blas: bool = False,
    ) -> None:
        self.shape_table = shape_table
        #: Target renders a dense 2-D float GEMM as a BLAS call rather than a loop nest.
        self.blas = blas
        self.temp_arrays = temp_arrays
        self.temp_counter = temp_counter
        self.local_dtypes: dict[str, str] = local_dtypes if local_dtypes is not None else {}
        #: Dimension local -> its definition, so a contraction whose operands spell the same extent
        #: two ways (``channels`` vs ``embed_dim``) is recognised instead of declined.
        self.dim_aliases: dict[str, str] = dim_aliases or {}
        #: Logical-name -> SparseArrayDesc (from KernelIR.sparse). When
        #: a matmul's operands are sparse, route to the sparse emitter.
        self.sparse: dict[str, object] = sparse or {}
        self.pre_stmts: list[ast.stmt] = []

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.MatMult):
            # Sparse path: both operands are logical sparse arrays.
            sp = self.try_hoist_sparse_matmul(node)
            if sp is not None:
                temp, stmts = sp
                self.pre_stmts.extend(self.prepend_alloc_markers(stmts))
                return ast.Name(id=temp, ctx=ast.Load())
            node = self.materialise_call_operands(node)
            temp, stmts = hoist_matmul(
                node,
                self.shape_table,
                self.temp_arrays,
                self.temp_counter,
                self.dim_aliases,
                blas=self.blas and self.blas_eligible(node),
            )
            if temp is not None:
                self.pre_stmts.extend(self.prepend_alloc_markers(stmts))
                # Propagate complex dtype across the matmul: if
                # either operand carries a complex tag (or a complex
                # Constant somewhere in the subtree), tag the matmul
                # temp ``__mm<n>`` so its decl is the right C type.
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, complex):
                        self.local_dtypes[temp] = "complex128"
                        break
                    if isinstance(sub, ast.Name):
                        dt = self.local_dtypes.get(sub.id)
                        if dt and dt.startswith("complex"):
                            self.local_dtypes[temp] = "complex128"
                            break
                return ast.Name(id=temp, ctx=ast.Load())
        return node

    def blas_eligible(self, node: ast.BinOp) -> bool:
        """True when both operands are real floats, so a BLAS gemm computes the same contraction.

        Integer, boolean and complex matmuls have no real-BLAS equivalent and keep the loop nest.
        A complex literal anywhere in the subtree counts, matching how the caller tags the temp.
        """
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Constant) and isinstance(sub_node.value, complex):
                return False
            if isinstance(sub_node, ast.Name):
                dt = self.local_dtypes.get(sub_node.id, "")
                if dt.startswith(BLAS_INELIGIBLE_DTYPES):
                    return False
        return True

    def materialise_call_operands(self, node: ast.BinOp) -> ast.BinOp:
        """Spill a CALL-valued matmul operand to a temp array, so the hoister sees a bare Name.

        ``relu_self_attention`` writes ``np.maximum(scores, 0.0) @ v``. The elementwise call has a
        perfectly well-defined extent, but the loop nest below indexes its operands by name, so the
        matmul was declined -- and a declined matmul reaches slice fusion, where scalarising it
        would drop the contraction. Materialising is what numpy does anyway; the guard downstream
        stays exactly as strict.

        Two things are deliberately left alone, on the same principle -- do not reroute what
        already lowers. A ``Subscript`` operand has its own slice-aware path, and a RANK-1 operand
        reaches the scalar dot-product form, which reads a call operand happily via
        ``iter_extent_of_``; spilling either would trade a working lowering for an extra temp
        array and a copy loop.
        """
        left, right = node.left, node.right
        for side in ("left", "right"):
            operand = left if side == "left" else right
            if not isinstance(operand, ast.Call):
                continue
            ext = iter_extent_of_(operand, self.shape_table)
            if ext is None or len(ext) < 2:
                continue
            nm, stmts = self.materialise_dense_operand(operand)
            if nm is None:
                continue
            self.pre_stmts.extend(self.prepend_alloc_markers(stmts))
            if side == "left":
                left = name_(nm)
            else:
                right = name_(nm)
        if left is node.left and right is node.right:
            return node
        return ast.BinOp(left=left, op=ast.MatMult(), right=right)

    def prepend_alloc_markers(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        """Prepend a ``__hpcagent_bench_zeros__()`` allocation marker for each array
        temp written in ``stmts`` (first-write order). A matmul/column-slice
        temp whose shape depends on a body-computed scalar (gmres ``n``/``m``)
        can't be malloc'd at fn-top -- the marker defers its malloc to this
        site, which always follows the scalar's assignment. For a
        param-shaped temp already malloc'd at fn-top the marker is a no-op, so
        prepending one unconditionally is safe.
        """
        seen: list[str] = []
        for s in stmts:
            for sub in ast.walk(s):
                tgt = None
                if isinstance(sub, ast.Assign) and sub.targets:
                    tgt = sub.targets[0]
                elif isinstance(sub, ast.AugAssign):
                    tgt = sub.target
                if tgt is None:
                    continue
                while isinstance(tgt, ast.Subscript):
                    tgt = tgt.value
                if isinstance(tgt, ast.Name) and tgt.id in self.temp_arrays and tgt.id not in seen:
                    seen.append(tgt.id)
        return [alloc_marker(n) for n in seen] + stmts

    def try_hoist_sparse_matmul(self, node: ast.BinOp) -> tuple[str, list[ast.stmt]] | None:
        """Route ``A @ B`` through the sparse emitter when an operand carries a
        sparse layout. Returns ``(temp_name, stmts)`` for the fresh result
        temp, or ``None`` when neither operand is sparse (dense path handles
        it).

        Type algebra (raises ``NotImplementedError`` on unsupported combos, so
        a clear failure surfaces at lowering rather than silent wrong
        numerics): ``sparse @ dense``/``dense @ sparse`` -> **dense** (matvec
        if the dense operand is 1-D, matmat -- CSR only for now -- if 2-D);
        ``csr @ csr`` -> **dense** result temp (the surrounding ``alpha *
        (A@B) + beta * C`` densifies it anyway, matching scipy's ``sparse @
        sparse + dense``; a CSR-output Gustavson form exists separately for
        pure-SpGEMM kernels); every other ``sparse @ sparse`` combo errors.

        This is the C/Fortran realisation of
        :func:`numpyto_common.sparse_emit.result_layout`: every supported case
        here densifies, matching ``result_layout(..., target="c") == DENSE`` --
        the hoister runs in the dense-accumulation context, so it always
        densifies rather than emitting a CSR-output SpGEMM.
        """
        if not self.sparse:
            return None
        # A sparse operand is always a bare logical Name (slicing a CSR buffer
        # set is unsupported). When exactly one operand is sparse and the other
        # is a non-Name dense expression -- e.g. GMRES's column slice ``A @
        # Q[:, k]`` -- materialise the dense operand into a fresh temp array so
        # the SpMV/SpMM expanders (which require a declared dense array) can
        # consume it.
        pre: list[ast.stmt] = []
        # Sparse TRANSPOSE matvec ``A.T @ x`` (bicg's ``A.T @ p_tilde``): A's
        # CSR buffers are exactly A.T's CSC buffers (and vice versa), and COO
        # transposes by swapping row/col roles -- so a transpose reuses the
        # same physical buffers under the dual format, no extra data.
        tr = self.transpose_sparse_desc(node.left)
        if tr is not None and isinstance(node.right, ast.Name) and node.right.id not in self.sparse:
            td, transposed = tr
            dense_shape = self.shape_table.get(node.right.id)
            if dense_shape and len(dense_shape) == 1:
                self.temp_counter[0] += 1
                temp = f"__mm{self.temp_counter[0]}"
                n_rows = td.logical_shape[0] if td.logical_shape else "0"
                self.temp_arrays[temp] = (n_rows,)
                self.shape_table[temp] = (n_rows,)
                return temp, pre + self.sparse_matvec(td, node.right.id, temp, transposed=transposed)
        l_sparse = isinstance(node.left, ast.Name) and node.left.id in self.sparse
        r_sparse = isinstance(node.right, ast.Name) and node.right.id in self.sparse
        if not (l_sparse or r_sparse):
            return None  # neither operand is a sparse Name -- dense path
        if l_sparse and not isinstance(node.right, ast.Name):
            nm, stmts = self.materialise_dense_operand(node.right, max_rank=1)
            if nm is None:
                return None
            pre.extend(stmts)
            node = ast.BinOp(left=node.left, op=ast.MatMult(), right=name_(nm))
        elif r_sparse and not isinstance(node.left, ast.Name):
            nm, stmts = self.materialise_dense_operand(node.left, max_rank=1)
            if nm is None:
                return None
            pre.extend(stmts)
            node = ast.BinOp(left=name_(nm), op=ast.MatMult(), right=node.right)
        if not (isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name)):
            return None
        la = self.sparse.get(node.left.id)
        ra = self.sparse.get(node.right.id)
        if la is None and ra is None:
            return None  # purely dense -- not our path
        from hpcagent_bench.translators.numpyto_common import sparse_emit as se

        # sparse @ sparse
        if la is not None and ra is not None:
            lfmt, rfmt = la.format, ra.format
            if lfmt == "csr" and rfmt == "csr":
                self.temp_counter[0] += 1
                temp = f"__mm{self.temp_counter[0]}"
                ni = la.logical_shape[0] if la.logical_shape else "0"
                nj = (
                    ra.logical_shape[1]
                    if len(ra.logical_shape) > 1
                    else (ra.logical_shape[0] if ra.logical_shape else "0")
                )
                self.temp_arrays[temp] = (ni, nj)
                self.shape_table[temp] = (ni, nj)
                stmts = se.expand_matmul_csr_csr_dense(temp, la.buffers, ra.buffers, ni, nj)
                return temp, pre + stmts
            raise NotImplementedError(
                f"sparse @ sparse only supports csr @ csr; got "
                f"{lfmt} @ {rfmt} ({node.left.id} @ {node.right.id}). "
                "Convert operands to CSR or split the kernel."
            )

        # sparse @ dense / dense @ sparse -- exactly one operand is sparse.
        if la is not None:
            sp_desc, dense_name, sp_on_left = la, node.right.id, True
        else:
            sp_desc, dense_name, sp_on_left = ra, node.left.id, False
        dense_shape = self.shape_table.get(dense_name)
        rank = len(dense_shape) if dense_shape else None
        if rank == 1:
            # matvec: sparse (M x N) @ dense (N,) -> dense (M,).
            if not sp_on_left:
                raise NotImplementedError(
                    "dense (1-D) @ sparse is a row-vector times matrix; not supported -- write it as sparse.T @ x."
                )
            self.temp_counter[0] += 1
            temp = f"__mm{self.temp_counter[0]}"
            n_rows = sp_desc.logical_shape[0] if sp_desc.logical_shape else "0"
            self.temp_arrays[temp] = (n_rows,)
            self.shape_table[temp] = (n_rows,)
            stmts = self.sparse_matvec(sp_desc, dense_name, temp)
            return temp, pre + stmts
        # matmat sparse @ dense (2-D) -> dense -- CSR only for now.
        if rank == 2 and sp_on_left and sp_desc.format == "csr":
            self.temp_counter[0] += 1
            temp = f"__mm{self.temp_counter[0]}"
            n_rows = sp_desc.logical_shape[0] if sp_desc.logical_shape else "0"
            n_cols = dense_shape[1]
            self.temp_arrays[temp] = (n_rows, n_cols)
            self.shape_table[temp] = (n_rows, n_cols)
            stmts = se.expand_matmul_csr_dense_mat(temp, sp_desc.buffers, dense_name, n_rows, n_cols)
            return temp, pre + stmts
        raise NotImplementedError(
            f"sparse @ dense for format {sp_desc.format} with dense rank "
            f"{rank} not supported ({node.left.id} @ {node.right.id})."
        )

    def materialise_dense_operand(
        self, expr: ast.expr, max_rank: int | None = None
    ) -> tuple[str | None, list[ast.stmt]]:
        """Copy a non-Name dense matmul operand into a fresh temp array, so the consumer -- which
        requires a *declared* array -- sees a bare Name. Two callers want this: the SpMV/SpMM
        expanders, for a column slice like ``Q[:, k]`` in ``A @ Q[:, k]``, and the dense hoister,
        for a call-valued operand like ``np.maximum(scores, 0.0) @ v``. Returns ``(temp_name,
        stmts)`` filling the temp, or ``(None, [])`` when the extent doesn't resolve.

        ``max_rank`` bounds what is accepted. The sparse path passes 1 on purpose: a 2-D dense
        slice on the sparse side (SpMM with a sliced RHS) must fall through and fail loudly rather
        than emit wrong shapes.
        """
        ext = iter_extent_of_(expr, self.shape_table)
        if not ext:
            return None, []
        if max_rank is not None and len(ext) > max_rank:
            return None, []
        self.temp_counter[0] += 1
        n = self.temp_counter[0]
        temp = f"__spv{n}"
        # Carry a complex dtype tag from any complex base array so the
        # temp's C decl matches (real default otherwise).
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Name):
                dt = self.local_dtypes.get(sub.id)
                if dt and dt.startswith("complex"):
                    self.local_dtypes[temp] = "complex128"
                    break
        shape = tuple((static_shape_of(expr, ax, self.shape_table) or ast.unparse(e)) for ax, e in enumerate(ext))
        self.temp_arrays[temp] = shape
        self.shape_table[temp] = shape
        iters = [name_(f"__spvi{n}_{ax}") for ax in range(len(ext))]
        elem = scalarize_at_iters(expr, iters, self.shape_table)
        sub_slice = ast.Tuple(elts=list(iters), ctx=ast.Load()) if len(iters) > 1 else iters[0]
        body: ast.stmt = ast.Assign(
            targets=[ast.Subscript(value=name_(temp), slice=sub_slice, ctx=ast.Store())], value=elem
        )
        for it, extent in zip(reversed(iters), reversed(list(ext))):
            body = ast.For(
                target=store_(it.id),
                iter=ast.Call(func=name_("range"), args=[extent], keywords=[]),
                body=[body],
                orelse=[],
            )
        return temp, [body]

    def transpose_sparse_desc(self, operand: ast.expr) -> tuple[object, bool] | None:
        """If ``operand`` is ``A.T`` for a sparse ``A``, return ``(desc, transposed)`` describing
        ``A.T`` so the matvec dispatcher emits ``A.T @ x`` directly; ``None`` otherwise.

        Two kinds of transpose, and the flag says which:

        * **relabelled** (``transposed=False``) -- CSR's buffers ARE its transpose's CSC buffers
          (and vice versa), and COO transposes by swapping the row/col roles. The descriptor alone
          carries the transpose, so the forward per-format matvec is correct as-is.
        * **access-transposed** (``transposed=True``) -- DIA and BCSR have no dual descriptor over
          the same buffers (negating DIA's offsets re-keys its data columns; BCSR would need
          indptr/indices rebuilt over block columns AND every block transposed). The format stays
          put and the ``*_t`` matvec transposes the ACCESS instead.

        ``logical_shape`` is reversed either way, so the caller sizes the result temp off the
        transpose's own row count without knowing which kind it got.
        """
        if not (
            isinstance(operand, ast.Attribute)
            and operand.attr == "T"
            and isinstance(operand.value, ast.Name)
            and operand.value.id in self.sparse
        ):
            return None
        from hpcagent_bench.translators.numpyto_common.ir import SparseArrayDesc

        d = self.sparse[operand.value.id]
        ls = list(d.logical_shape) if d.logical_shape else []
        swapped = tuple(reversed(ls)) if len(ls) >= 2 else tuple(ls)
        dual = {"csr": "csc", "csc": "csr"}.get(d.format)
        if dual is not None:
            return SparseArrayDesc(name=d.name, format=dual, logical_shape=swapped, buffers=dict(d.buffers)), False
        if d.format == "coo":
            b = dict(d.buffers)
            if "row" in b and "col" in b:
                b["row"], b["col"] = d.buffers["col"], d.buffers["row"]
            return SparseArrayDesc(name=d.name, format="coo", logical_shape=swapped, buffers=b), False
        if d.format in ("dia", "bcsr"):
            return SparseArrayDesc(name=d.name, format=d.format, logical_shape=swapped, buffers=dict(d.buffers)), True
        return None

    def sparse_matvec(self, sp_desc: object, dense_name: str, temp: str, transposed: bool = False) -> list[ast.stmt]:
        """Build the per-format matvec loop nest filling 1-D ``temp``.

        Derives each format's extra size symbols from the sparse
        descriptor's logical shape + physical buffer shapes, then calls
        the matching dispatcher in ``sparse_emit``.

        ``transposed`` marks the DIA/BCSR ``A.T @ x`` case, where the descriptor is A's own
        (only its logical shape is reversed) and the ``*_t`` dispatcher transposes the access --
        see :meth:`transpose_sparse_desc`. Every other format arrives already relabelled, so a
        forward matvec on the dual descriptor is the transpose.
        """
        from hpcagent_bench.translators.numpyto_common import sparse_emit as se

        fmt = sp_desc.format
        bufs = sp_desc.buffers
        tgt = name_(temp)
        n_rows = sp_desc.logical_shape[0] if sp_desc.logical_shape else "0"
        n_cols = sp_desc.logical_shape[1] if len(sp_desc.logical_shape) > 1 else "0"

        def buf_shape(role: str, axis: int) -> str | None:
            """Shape token of the physical buffer for ``role`` at ``axis``,
            looked up from the shape table (physical buffers are declared
            arrays)."""
            phys = bufs.get(role)
            sh = self.shape_table.get(phys) if phys else None
            if sh and axis < len(sh):
                return sh[axis]
            return None

        if fmt == "csr":
            return se.expand_matmul_csr_dense_vec(tgt, bufs, dense_name, n_rows)
        if fmt == "csc":
            return se.expand_matmul_csc_dense_vec(tgt, bufs, dense_name, n_rows, n_cols)
        if fmt == "coo":
            nnz = buf_shape("data", 0) or "0"
            return se.expand_matmul_coo_dense_vec(tgt, bufs, dense_name, n_rows, nnz)
        if fmt == "dia":
            ndiag = buf_shape("data", 0) or "0"
            # Transposed: the descriptor's shape is reversed, so A's own (rows, cols) are
            # (n_cols, n_rows) here -- the expanders take A's dims, not A.T's, to stay
            # readable side by side.
            if transposed:
                return se.expand_matmul_dia_t_dense_vec(tgt, bufs, dense_name, n_cols, n_rows, ndiag)
            return se.expand_matmul_dia_dense_vec(tgt, bufs, dense_name, n_rows, n_cols, ndiag)
        if fmt == "ell":
            maxnz = buf_shape("data", 1) or "0"
            return se.expand_matmul_ell_dense_vec(tgt, bufs, dense_name, n_rows, maxnz)
        if fmt == "jds":
            # njd = len(jd_ptr) - 1
            jdlen = buf_shape("jd_ptr", 0)
            njd = f"({jdlen}) - 1" if jdlen else "0"
            # The dispatcher uses a sorted-order scratch accumulator but relies
            # on the caller to declare it (see expand_matmul_jds_* docstring):
            # register it as a fresh (n_rows,) local so the emitter allocates
            # it -- the dispatcher zeroes it itself.
            self.temp_arrays["__jds_y_perm"] = (n_rows,)
            self.shape_table["__jds_y_perm"] = (n_rows,)
            return se.expand_matmul_jds_dense_vec(tgt, bufs, dense_name, n_rows, njd)
        if fmt == "bcsr":
            # block dims live on the descriptor's logical shape vs buffer
            # data shape [nnz_blk, R, C]; n_block_rows = len(indptr) - 1.
            iplen = buf_shape("indptr", 0)
            nbr = f"({iplen}) - 1" if iplen else "0"
            R = buf_shape("data", 1) or "1"
            C = buf_shape("data", 2) or "1"
            if transposed:
                # ``n_rows`` is A.T's row count, i.e. A's COLUMN count -- exactly the length
                # the transposed matvec writes.
                return se.expand_matmul_bcsr_t_dense_vec(tgt, bufs, dense_name, nbr, R, C, n_rows)
            return se.expand_matmul_bcsr_dense_vec(tgt, bufs, dense_name, nbr, R, C, n_rows)
        if fmt == "bcoo":
            # block-COO: row[k]/col[k] hold block coords, data is
            # [n_blocks, R, C]; n_blocks = len(row); the total scalar
            # row count is n_rows (the descriptor's logical row dim).
            nblk = buf_shape("row", 0) or buf_shape("data", 0) or "0"
            R = buf_shape("data", 1) or "1"
            C = buf_shape("data", 2) or "1"
            return se.expand_matmul_bcoo_dense_vec(tgt, bufs, dense_name, n_rows, nblk, R, C)
        if fmt == "sell_c_sigma":
            nsl = buf_shape("slice_ptr", 0)
            nslices = f"({nsl}) - 1" if nsl else "0"
            # slice height C is a kernel parameter; default symbol "C".
            return se.expand_matmul_sell_c_sigma_dense_vec(tgt, bufs, dense_name, n_rows, nslices, "C")
        raise NotImplementedError(f"sparse matvec for format {fmt!r} not supported.")
