"""Iteration extent of array-valued expressions and numpy broadcasting over them."""

import ast
import copy
from collections.abc import Sequence

from hpcagent_bench.translators.numpyto_common import dtypes
from hpcagent_bench.translators.numpyto_common.lib_nodes.array_methods import ARRAY_METHOD_SHAPE_OPS
from hpcagent_bench.translators.numpyto_common.lib_nodes.call_args import (
    axes_kwarg,
    axis_literal_or_refuse,
    const_axis,
    kwarg_or_pos,
    np_call_attr,
    np_fft_attr,
    pad_widths,
    stack_axis,
    tensordot_axes,
    parse_einsum_subscripts,
    read_axis_keepdims,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.dims import NP_ZEROS_ALIASES
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import (
    const_,
    const_int,
    const_or_name,
    is_const_one,
    is_reduction_call,
    is_scalar_axis,
    is_special_axis,
    mul_exts,
    name_,
    name_id,
    simplify_sub,
    slice_step_any,
    step_is_negative,
    slice_axes,
)


def operand_token_shape(node: ast.expr, shape_table: dict[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    """Residual shape TOKENS (not AST nodes -- stays consistent with the shape
    table) of an einsum/contraction operand. Bare ``Name(A)`` -> A's declared
    shape. Anything else (a Subscript slice/index chain, a matmul, a
    shape-preserving call...) routes through the general ``iter_extent_of_``
    sizer, which already knows a partial slice's actual bound (``a[k:k+H]``
    -> ``H``, not the base axis's full extent) alongside every other form it
    resolves. ``None`` if unresolvable."""
    nm = name_id(node)
    if nm:
        return shape_table.get(nm)
    ext = iter_extent_of_(node, shape_table)
    return tuple(ast.unparse(e) for e in ext) if ext is not None else None


def chained_base_shape(node: ast.expr, shape_table: dict[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    """Residual token-shape of a SCALAR-chained subscript base ``A[i, j][...]``,
    when every inner index is a single-axis scalar (int Constant / bare Name):
    numpy combined-basic-indexing drops one leading axis per scalar, e.g.
    ``psi_frag[f]`` on ``(F, X, Y, Z, K)`` -> ``(X, Y, Z, K)``. Lets
    :func:`iter_extent_of_` size a chained access like ``psi_frag[f][..., 0]``
    whose base isn't yet flattened to a single Name subscript. ``None`` for a
    non-Name base or a Slice/Ellipsis/newaxis inner index."""
    if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)):
        return None
    # A bare Name axis is scalar ONLY when it is not a known array: ``x[aj]`` with aj an index
    # ARRAY is a gather, which KEEPS the axis (broadcast to aj's shape) instead of dropping it.
    # Treating it as a scalar shortened the residual shape, and an outer subscript
    # (``x[aj][:, None, :, :]``) then ran off the end and gave up on the whole extent.
    for axis in slice_axes(node):
        if not is_scalar_axis(axis):
            return None
        if isinstance(axis, ast.Name) and shape_table.get(axis.id):
            return None
    return operand_token_shape(node, shape_table)


def contraction_result_extent(expr: ast.Call, shape_table: dict[str, tuple[str, ...]]) -> tuple[ast.expr, ...] | None:
    """Output iter-extent of an ``np.einsum``/``tensordot``/``inner`` call.
    einsum uses its subscript string directly; tensordot/inner are mapped to
    an equivalent einsum spec first. ``None`` if operand shapes don't resolve."""
    attr = expr.func.attr
    if attr == "einsum":
        if not (isinstance(expr.args[0], ast.Constant) and isinstance(expr.args[0].value, str)):
            return None
        try:
            inputs, output = parse_einsum_subscripts(expr.args[0].value)
        except NotImplementedError:
            return None
        operand_nodes = expr.args[1:]
    else:
        # tensordot / inner: build the equivalent spec from operand ranks. Operands need not be
        # bare Names -- ``operand_token_shape`` resolves a slice/index-chain's residual rank too
        # (conv2d's tensordot contracts a sliced 4-D input against a partially-indexed
        # ``weights[ki, kj]``).
        a, b = expr.args[0], expr.args[1]
        shape_a, shape_b = operand_token_shape(a, shape_table), operand_token_shape(b, shape_table)
        if not shape_a or not shape_b:
            return None
        ra, rb = len(shape_a), len(shape_b)
        letters = "abcdefghijklmnopqrstuvwxyz"
        if attr == "inner":
            a_spec, b_spec = list(letters[:ra]), list(letters[ra : ra + rb])
            b_spec[-1] = a_spec[-1]
            inputs = ["".join(a_spec), "".join(b_spec)]
            output = "".join(a_spec[:-1] + b_spec[:-1])
        else:  # tensordot default axes=2
            kwargs = expr.keywords
            axes_node = expr.args[2] if len(expr.args) > 2 else axes_kwarg(kwargs)
            try:
                a_ax, b_ax = tensordot_axes(axes_node, ra, rb)
            except NotImplementedError:
                return None  # sizer contract: an unresolved extent is None, never an exception
            a_spec = list(letters[:ra])
            b_spec = [None] * rb
            nxt = ra
            for ca, cb in zip(a_ax, b_ax):
                b_spec[cb] = a_spec[ca]
            for i in range(rb):
                if b_spec[i] is None:
                    b_spec[i] = letters[nxt]
                    nxt += 1
            inputs = ["".join(a_spec), "".join(b_spec)]
            output = "".join(
                [c for i, c in enumerate(a_spec) if i not in a_ax] + [c for i, c in enumerate(b_spec) if i not in b_ax]
            )
        operand_nodes = [a, b]
    letter_extent: dict[str, str] = {}
    for spec, node in zip(inputs, operand_nodes):
        shape = operand_token_shape(node, shape_table)
        if shape is None or len(shape) != len(spec):
            return None
        for letter, dim in zip(spec, shape):
            letter_extent.setdefault(letter, dim)
    if not output:
        return None  # scalar
    return tuple(const_or_name(letter_extent[c]) for c in output)


def concat_extent(attr: str, expr: ast.Call, shape_table: dict[str, tuple[str, ...]]) -> tuple[ast.expr, ...] | None:
    """Extent of a concatenation call, or ``None`` when the operands do not agree on one.

    Every operand must size, share a rank, and agree on every axis but the joined one, whose
    extents are summed. Anything else -- a mixed rank, an unresolved operand, an axis numpy would
    broadcast -- returns ``None`` rather than a shape that is merely plausible.
    """
    operands = (
        list(expr.args[0].elts)
        if (len(expr.args) == 1 and isinstance(expr.args[0], (ast.Tuple, ast.List)))
        else list(expr.args)
    )
    extents = [iter_extent_of_(operand, shape_table) for operand in operands]
    if not extents or any(e is None for e in extents):
        return None
    rank = len(extents[0])
    if any(len(e) != rank for e in extents):
        return None
    if attr == "vstack" and rank == 1:
        return None  # vstack STACKS 1-D operands into a new leading axis; that is a rank change
    axis = 0 if rank == 1 or attr == "vstack" else 1
    if attr == "concatenate":
        node = kwarg_or_pos(expr.args[1:], expr.keywords, 0, "axis")
        axis = 0 if node is None else const_axis(node, rank)
        if axis is None:
            return None
    if axis >= rank:
        return None
    kept = [ast.unparse(e) for k, e in enumerate(extents[0]) if k != axis]
    if any([ast.unparse(e) for k, e in enumerate(extent) if k != axis] != kept for extent in extents[1:]):
        return None  # the untouched axes are spelled differently; nothing here can prove them equal
    widths = [extent[axis] for extent in extents]
    literals = [const_int(w) for w in widths]
    if all(v is not None for v in literals):
        joined: ast.expr = const_(sum(literals))  # ``3``, not ``1 + 1 + 1``: this becomes a stride
    else:
        joined = widths[0]
        for width in widths[1:]:
            joined = ast.BinOp(left=joined, op=ast.Add(), right=width)
    return tuple(joined if k == axis else extents[0][k] for k in range(rank))


def sum_width_tokens(tokens: Sequence[str]) -> str:
    """The concatenated width of ``tokens``, folded when every one of them is a literal.

    ``"3"``, not ``"1+1+1"``: this token becomes a stride in the emitted index arithmetic, and the
    same buffer is sized through a second path that folds. One buffer spelled two ways reads as two.
    """
    values = [int(tok) for tok in tokens if str(tok).lstrip("-").isdigit()]
    return str(sum(values)) if len(values) == len(tokens) else "+".join(str(tok) for tok in tokens)


def iter_extent_of_(expr: ast.expr, shape_table: dict[str, tuple[str, ...]]) -> tuple[ast.expr, ...] | None:
    """Iteration extent of an array-valued expression. ``Name(A)`` -> A's full
    shape; ``Subscript(A, axes)`` -> upper-minus-lower per Slice axis in order
    (non-Slice axes are scalar, contribute nothing); negative slice bounds
    resolve against the operand's declared shape. ``None`` for unsupported
    forms -- caller falls through to ``NotImplementedError``.
    """
    if isinstance(expr, ast.Name):
        shape = shape_table.get(expr.id)
        return None if shape is None else tuple(const_or_name(s) for s in shape)
    if isinstance(expr, ast.BinOp):
        # numpy ``@`` (MatMult) treats the last two axes as the matrix; leading
        # axes are batched and broadcast per the numpy spec:
        #
        #   1-D @ 1-D                  -> scalar (None)
        #   2-D @ 1-D                  -> (M,)
        #   1-D @ 2-D                  -> (N,)
        #   2-D @ 2-D                  -> (M, N)
        #   (..., M, K) @ (..., K, N)  -> (..., M, N)  (batched)
        #   (..., M, K) @ (K, N)       -> (..., M, N)  (broadcast 2-D rhs)
        #   (M, K) @ (..., K, N)       -> (..., M, N)  (broadcast 2-D lhs)
        if isinstance(expr.op, ast.MatMult):
            l_ext = iter_extent_of_(expr.left, shape_table)
            r_ext = iter_extent_of_(expr.right, shape_table)
            if l_ext is None or r_ext is None:
                return None
            ll, rl = len(l_ext), len(r_ext)
            if ll == 1 and rl == 1:
                return None  # scalar
            if ll == 2 and rl == 1:
                return (l_ext[0],)
            if ll == 1 and rl == 2:
                return (r_ext[1],)
            if ll == 2 and rl == 2:
                return (l_ext[0], r_ext[1])
            # Batched matmul: last two dims are M,K and K,N; broadcast leading axes.
            if ll >= 2 and rl >= 2:
                l_batch = l_ext[:-2]
                r_batch = r_ext[:-2]
                l_mn = (l_ext[-2], l_ext[-1])
                r_mn = (r_ext[-2], r_ext[-1])
                batch = broadcast_extents(l_batch, r_batch)
                return tuple(batch) + (l_mn[0], r_mn[1])
            # 1-D paired with batched 2-D (collapse leading axis):
            #   ``a (K,) @ B (..., K, N) -> (..., N)``
            #   ``A (..., M, K) @ b (K,) -> (..., M)``
            if ll == 1 and rl > 2:
                return tuple(r_ext[:-2]) + (r_ext[-1],)
            if rl == 1 and ll > 2:
                return tuple(l_ext[:-2]) + (l_ext[-2],)
            return None
        # Broadcast the two extents axis-by-axis, aligned from the right,
        # picking the non-1 axis at each position (numpy rules).
        # ``A + b`` with A:(N, M), b:(M,) -> (N, M).
        # ``X + Y[:, None]`` with X:(M,), Y[:, None]:(N, 1) -> (N, M).
        l_ext = iter_extent_of_(expr.left, shape_table)
        r_ext = iter_extent_of_(expr.right, shape_table)
        if l_ext is None:
            return r_ext
        if r_ext is None:
            return l_ext
        return broadcast_extents(l_ext, r_ext)
    if isinstance(expr, ast.UnaryOp):
        return iter_extent_of_(expr.operand, shape_table)
    if isinstance(expr, ast.Compare):
        ext = iter_extent_of_(expr.left, shape_table)
        for operand in expr.comparators:
            other = iter_extent_of_(operand, shape_table)
            if ext is None:
                ext = other
            elif other is not None:
                ext = broadcast_extents(ext, other)
        return ext
    if isinstance(expr, ast.BoolOp):
        ext = None
        for operand in expr.values:
            other = iter_extent_of_(operand, shape_table)
            if ext is None:
                ext = other
            elif other is not None:
                ext = broadcast_extents(ext, other)
        return ext
    if isinstance(expr, ast.IfExp):
        body_ext = iter_extent_of_(expr.body, shape_table)
        orelse_ext = iter_extent_of_(expr.orelse, shape_table)
        if body_ext is None:
            return orelse_ext
        if orelse_ext is None:
            return body_ext
        return broadcast_extents(body_ext, orelse_ext)
    if isinstance(expr, ast.Call):
        # ``np.fft.*`` is a two-level attribute (``func.value`` is ``np.fft``),
        # missed by the single-level ``np.<attr>`` matcher below. ``fftfreq(n)``
        # builds a length-n 1-D array; complex transforms/shifts are shape-
        # preserving -- sizing them here lets the harvest resolve LS3DF's
        # ``kx = 2*pi*np.fft.fftfreq(N)`` -> ``np.meshgrid`` -> ``gsq`` chain before
        # scalarization. REAL-FFT variants change the transformed axis length
        # (``rfftfreq(n)`` -> ``n//2+1``; ``rfftn``/``irfftn`` resize the last
        # axis) and are NOT sized here -- bail to None rather than emit a wrong extent.
        fft = np_fft_attr(expr)
        if fft is not None:
            if fft == "fftfreq" and expr.args:
                return (copy.deepcopy(expr.args[0]),)
            if fft in ("fftn", "ifftn", "fft", "ifft", "fft2", "ifft2", "fftshift", "ifftshift") and expr.args:
                return iter_extent_of_(expr.args[0], shape_table)
            return None
        # Method-form ``<expr>.reshape(...)``: receiver is the operand, not ``np``,
        # so the ``np.reshape`` branch below (and the harvest, which runs before
        # the method->function normaliser) misses it. Left unsized, a reshape
        # target like ``X = (Yf @ C).reshape(shp)`` poisons every derived shape
        # (LS3DF's Rayleigh-Ritz).
        if (
            isinstance(expr.func, ast.Attribute)
            and expr.func.attr == "reshape"
            and not (isinstance(expr.func.value, ast.Name) and expr.func.value.id in ("np", "numpy"))
            and expr.args
        ):
            if len(expr.args) == 1 and isinstance(expr.args[0], (ast.Tuple, ast.List)):
                elts = list(expr.args[0].elts)
            elif len(expr.args) == 1 and isinstance(expr.args[0], (ast.Name, ast.Constant, ast.BinOp, ast.UnaryOp)):
                elts = [expr.args[0]]
            else:
                elts = list(expr.args)  # varargs ``.reshape(a, b, c)``
            neg1 = [i for i, e in enumerate(elts) if const_int(e) == -1]
            if len(neg1) == 1:
                base = iter_extent_of_(expr.func.value, shape_table)
                if base is None:
                    return None
                others = [e for j, e in enumerate(elts) if j != neg1[0]]
                denom = mul_exts(others) if others else const_(1)
                elts[neg1[0]] = ast.BinOp(left=mul_exts(base), op=ast.Div(), right=denom)
            elif neg1:
                return None
            return tuple(elts)
        # Axis-aware reduction: ``np.sum(operand, axis=k)`` -> operand's extent
        # with axis k removed (size 1 if keepdims); axis=None collapses to a
        # scalar (None). Makes the reduction's result shape available to every
        # shape-propagation caller (gem's ``r = np.sqrt(np.sum(d * d, axis=2))``,
        # force_lj, kmeans).
        if is_reduction_call(expr):
            # The METHOD spelling carries its operand in the RECEIVER, not in ``args[0]``:
            # ``m.any(axis=-1)`` resolved to None where ``np.any(m, axis=-1)`` resolved fine, so
            # every shape built on one silently fell back to a broadcast partner's extent --
            # cp2k_density_matrix_trs4's ``c_pos`` came out (nnz, 1) instead of (nnz, fanout), and
            # the scatter that reads it then ran over a third of the contributions.
            method_form = isinstance(expr.func, ast.Attribute) and not (
                isinstance(expr.func.value, ast.Name) and expr.func.value.id in ("np", "numpy")
            )
            # ``read_axis_keepdims`` reads the axis from positional slot 1, so the method's args are
            # shifted by one to put them in the vocabulary it expects.
            red_args = ([expr.func.value] + list(expr.args)) if method_form else list(expr.args)
            if not red_args:
                return None
            axes, keepdims = read_axis_keepdims(red_args, expr.keywords)
            if axes is None:
                return None
            base = iter_extent_of_(red_args[0], shape_table)
            if base is None:
                return None
            n = len(base)
            norm = {a % n for a in axes}
            if keepdims:
                return tuple(const_(1) if i in norm else base[i] for i in range(n))
            return tuple(base[i] for i in range(n) if i not in norm) or None
        # Shape-CHANGING ops: result extent is NOT the operand's. ``np.reshape(A,
        # newshape)`` -> newshape; treating it as elementwise would propagate the
        # wrong rank to an enclosing BinOp (stockham_fft's
        # ``tmp_twid = np.reshape(tmp_perm, (N,)) * np.reshape(D, (N,))`` must be
        # rank-1, not tmp_perm's rank-3). ``repeat``/``transpose`` aren't statically
        # resolvable here -- bail to None rather than report the wrong extent.
        # ``np.<x>`` and the two-level ufunc-method spellings ``np.<ufunc>.<method>`` alike. The
        # dotted form never reached here, so the ``maximum.accumulate`` branch below was
        # unreachable and every ``np.<op>.outer`` came back with its first operand's extent.
        # Method spelling of a shape op: the receiver IS the operand, so ``rho.copy()`` says what
        # ``np.copy(rho)`` says. Routed once, here, rather than as a method branch per op -- the
        # two spellings had already diverged for reshape and for the reductions above, each fixed
        # separately after a kernel came back sized off a broadcast partner instead. ``astype`` and
        # ``flatten`` have no numpy function twin to route to and answer directly.
        if isinstance(expr.func, ast.Attribute) and not (
            isinstance(expr.func.value, ast.Name) and expr.func.value.id in ("np", "numpy")
        ):
            method = expr.func.attr
            if method == "astype":
                return iter_extent_of_(expr.func.value, shape_table)  # dtype only, never the shape
            if method in ("ravel", "flatten"):
                base = iter_extent_of_(expr.func.value, shape_table)
                return None if base is None else (mul_exts(base),)
            if method in ARRAY_METHOD_SHAPE_OPS:
                routed = ast.Call(
                    func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=method, ctx=ast.Load()),
                    args=[expr.func.value] + list(expr.args),
                    keywords=list(expr.keywords),
                )
                return iter_extent_of_(ast.copy_location(routed, expr), shape_table)
        attr = np_call_attr(expr.func)
        if attr is not None:
            # An array CONSTRUCTOR states its extent in its shape argument -- read it
            # directly, since an inline constructor (never assigned to a Name) is
            # never sized by the harvest. Otherwise the triu/tril first-arg hoist in
            # ``CallHoister`` (gated on a resolvable extent) leaves it buried and the
            # bare-Name expander rejects it -- gpt2_block's causal-mask
            # ``np.triu(np.ones((seq, seq), np.float32), 1)``. ``*_like`` aliases take
            # an array, not a shape -- mirror that operand's extent instead.
            # ``np.full`` states its extent exactly as the zeros aliases do, but it is not one of
            # them: it carries a fill VALUE that the alias rewriter would drop, turning a -inf pad
            # into a zero pad. Sized here, aliased nowhere.
            # ``np.linalg`` is a TWO-level attribute, so every single-level branch below reads it
            # as nothing. ``inv``/``cholesky`` are shape-preserving factors; ``solve`` returns x
            # with b's shape. Sized here and not only in lowering's harvest, because a shape read
            # that resolves against a factorisation is asked at parse time -- ls3df_scf reaches its
            # whole Rayleigh-Ritz block through ``Linv``, and an unsized factor breaks that chain.
            # Square identities and the 1-D generators state their extent in an argument, same as
            # the zeros aliases do (ls3df_scf's ``np.eye(k)`` jitter term).
            if attr in ("eye", "identity") and expr.args:
                n = copy.deepcopy(expr.args[0])
                return (
                    (n, copy.deepcopy(expr.args[1]))
                    if attr == "eye" and len(expr.args) >= 2 and not const_int(expr.args[1]) is None
                    else (n, copy.deepcopy(n))
                )
            if attr == "arange" and len(expr.args) == 1:
                return (copy.deepcopy(expr.args[0]),)
            if attr in ("hstack", "vstack", "concatenate") and expr.args:
                # Concatenation is the one shape-changing form a helper's RETURN commonly takes
                # (nbody's ``return np.hstack((ax, ay, az))``), and an unsized one leaves the
                # caller's own broadcast join over the ARGUMENTS as the only answer -- ``(N, N)``
                # for a result that is ``(N, 3)``. Axis 1 for 2-D under hstack, axis 0
                # for 1-D and for vstack; ``concatenate`` reads its own ``axis``.
                return concat_extent(attr, expr, shape_table)
            if attr in ("linalg.inv", "linalg.cholesky") and expr.args:
                return iter_extent_of_(expr.args[0], shape_table)
            if attr == "linalg.solve" and len(expr.args) >= 2:
                return iter_extent_of_(expr.args[1], shape_table)
            if (attr in NP_ZEROS_ALIASES or attr in ("full", "full_like")) and expr.args:
                if attr.endswith("_like"):
                    return iter_extent_of_(expr.args[0], shape_table)
                shape_arg = expr.args[0]
                if isinstance(shape_arg, (ast.Tuple, ast.List)):
                    return tuple(copy.deepcopy(e) for e in shape_arg.elts)
                return (copy.deepcopy(shape_arg),)
            if attr == "reshape" and len(expr.args) >= 2:
                newshape = expr.args[1]
                elts: list[ast.expr] | None = None
                if isinstance(newshape, (ast.Tuple, ast.List)):
                    elts = list(newshape.elts)
                elif isinstance(newshape, (ast.Name, ast.Constant, ast.BinOp, ast.UnaryOp)):
                    elts = [newshape]
                if elts is not None:
                    # Resolve a ``-1`` placeholder (``x.reshape(batch, -1)``) to
                    # ``total_source_size // product(other target dims)``.
                    neg1 = [i for i, e in enumerate(elts) if const_int(e) == -1]
                    if len(neg1) == 1:
                        base = iter_extent_of_(expr.args[0], shape_table)
                        if base is None:
                            return None
                        others = [e for j, e in enumerate(elts) if j != neg1[0]]
                        denom = mul_exts(others) if others else const_(1)
                        # ``/`` (renders as integer division in C/Fortran for int dims),
                        # not ``//`` which is not valid C when the token is emitted.
                        elts[neg1[0]] = ast.BinOp(left=mul_exts(base), op=ast.Div(), right=denom)
                    elif neg1:
                        return None  # more than one -1 is ambiguous
                    return tuple(elts)
            if attr == "transpose" and expr.args:
                # ``x.T``/``np.transpose(x)`` -> operand's extent with axes reversed
                # (or permuted by an explicit axes tuple). nbody's ``dx = x.T - x``
                # (x is (N, 1)) must broadcast to (N, N); ``None`` here would collapse
                # it to (N, 1) -- ``dx`` then allocates (N, 1) but is written (N, N)
                # (heap overflow), and the ``(dx*inv_r3) @ mass`` matmul mis-contracts.
                base = iter_extent_of_(expr.args[0], shape_table)
                if base is None:
                    return None
                if len(expr.args) >= 2 and isinstance(expr.args[1], (ast.Tuple, ast.List)):
                    perm = [
                        e.value for e in expr.args[1].elts if isinstance(e, ast.Constant) and isinstance(e.value, int)
                    ]
                    if len(perm) == len(base):
                        return tuple(base[p] for p in perm)
                return tuple(reversed(base))
            # Shape aliases -> operand's extent with axes swapped / a unit axis
            # inserted or dropped, so an enclosing BinOp broadcasts and the fresh
            # target allocates at the right rank.
            if attr == "swapaxes" and len(expr.args) >= 3:
                base = iter_extent_of_(expr.args[0], shape_table)
                if base is None:
                    return None
                i, j = const_axis(expr.args[1], len(base)), const_axis(expr.args[2], len(base))
                if i is None or j is None:
                    return None
                out = list(base)
                out[i], out[j] = out[j], out[i]
                return tuple(out)
            if attr == "moveaxis" and len(expr.args) >= 3:
                # numpy's own algorithm: drop the source axis, reinsert it at the destination among
                # what is left. Without this the sizer returns None, the hoister declines, and the
                # call survives to the emitter as an unsupported ``np.moveaxis``.
                base = iter_extent_of_(expr.args[0], shape_table)
                if base is None:
                    return None
                src = const_axis(expr.args[1], len(base))
                dst = const_axis(expr.args[2], len(base))
                if src is None or dst is None:
                    return None
                rest = [e for n, e in enumerate(base) if n != src]
                return tuple(rest[:dst] + [base[src]] + rest[dst:])
            if attr == "expand_dims" and expr.args:
                base = iter_extent_of_(expr.args[0], shape_table)
                if base is None:
                    return None
                axis = const_axis(kwarg_or_pos(expr.args, expr.keywords, 1, "axis"), len(base) + 1)
                if axis is None:
                    return None
                out = list(base)
                out.insert(axis, const_(1))
                return tuple(out)
            if attr == "squeeze" and expr.args:
                base = iter_extent_of_(expr.args[0], shape_table)
                if base is None:
                    return None
                axis_node = kwarg_or_pos(expr.args, expr.keywords, 1, "axis")
                if axis_node is not None:
                    axis = const_axis(axis_node, len(base))
                    if axis is None or not (isinstance(base[axis], ast.Constant) and base[axis].value == 1):
                        return None
                    out = [e for k, e in enumerate(base) if k != axis]
                else:
                    out = [e for e in base if not (isinstance(e, ast.Constant) and e.value == 1)]
                return tuple(out) if out else (const_(1),)
            if attr == "take" and len(expr.args) >= 2:
                base = iter_extent_of_(expr.args[0], shape_table)
                # A LITERAL index takes one element off the axis, so numpy drops that axis
                # entirely -- distinct from an unresolvable index, which is what ``None`` from
                # ``iter_extent_of_`` otherwise means. Conflating the two left the enclosing
                # ``np.expand_dims`` / ``np.concatenate`` unsized and refused.
                lit_index = const_int(expr.args[1])
                idx_ext = None if lit_index is not None else iter_extent_of_(expr.args[1], shape_table)
                if base is None or (lit_index is None and (idx_ext is None or len(idx_ext) != 1)):
                    return None
                if lit_index is not None:
                    axis_node = kwarg_or_pos(expr.args, expr.keywords, 2, "axis")
                    if axis_node is None:
                        return None  # flat take on an N-D source: numpy ravels first
                    axis = const_axis(axis_node, len(base))
                    if axis is None:
                        return None
                    out = [e for k, e in enumerate(base) if k != axis]
                    return tuple(out) or None
                axis_node = kwarg_or_pos(expr.args, expr.keywords, 2, "axis")
                if axis_node is None:
                    return idx_ext if len(base) == 1 else None  # flat take on a 1-D source
                axis = const_axis(axis_node, len(base))
                if axis is None:
                    return None
                out = list(base)
                out[axis] = idx_ext[0]
                return tuple(out)
            if attr == "repeat":
                return None
            # ``np.<op>.outer(a, b)`` pairs every element of a with every element of b: rank 2,
            # sized from the two operands' own extents.
            if attr.endswith(".outer") and len(expr.args) == 2:
                l_out = iter_extent_of_(expr.args[0], shape_table)
                r_out = iter_extent_of_(expr.args[1], shape_table)
                if l_out is None or r_out is None or len(l_out) != 1 or len(r_out) != 1:
                    return None
                return (l_out[0], r_out[0])
            # ``np.einsum(subscripts, *operands)`` -> the output extent: one axis per
            # output index letter, sized from the operand that introduces it. The
            # elementwise fallthrough below would wrongly take the first operand's
            # full rank.
            if attr in ("einsum", "tensordot", "inner") and len(expr.args) >= 2:
                ext = contraction_result_extent(expr, shape_table)
                if ext is not None:
                    return ext
                return None
            if attr in ("trace", "vdot", "median"):
                return None  # scalar result
            if attr == "diagonal" and expr.args:
                base = iter_extent_of_(expr.args[0], shape_table)
                return (base[0],) if base else None
            # ``np.diag(v [, k])`` -- a 1-D operand builds an ``(n+|k|, n+|k|)``
            # matrix (single source of truth for the constructed shape); a 2-D
            # operand extracts the main diagonal (length of its first axis).
            if attr == "diag" and expr.args:
                base = iter_extent_of_(expr.args[0], shape_table)
                if base is None:
                    return None
                if len(base) == 2:
                    return (base[0],)
                if len(base) != 1:
                    return None
                k_node = kwarg_or_pos(expr.args, expr.keywords, 1, "k")
                if k_node is None:
                    off = 0
                else:
                    kc = const_int(k_node)
                    if kc is None:
                        return None  # non-const offset can't size the result
                    off = abs(kc)
                side = base[0] if off == 0 else ast.BinOp(left=base[0], op=ast.Add(), right=const_(off))
                return (side, copy.deepcopy(side))
            # ``np.cumsum``/``np.cumprod`` and the ``maximum``/``minimum.accumulate`` running
            # extremes: a prefix scan is shape-preserving along its
            # axis, so the result takes the operand's extent. Skipping this leaves a
            # fresh LHS (histogram_equalization's ``cdf = np.cumsum(hist)``) unsized,
            # so ``ELEMENT_WRITE_EXPANDERS`` (gated on the target already having a
            # shape) silently skips it and ``cdf[0] = ..`` stores through a NULL
            # pointer (SIGSEGV). numpy flattens an axis-less scan over an N-D operand,
            # which the cumulative expander rejects -- leave unresolved rather than
            # claim a wrong shape.
            # ``np.searchsorted(a, v)`` returns one index per element of ``v``, so the result takes
            # the VALUES operand's extent, not the sorted array's.
            if attr == "searchsorted" and len(expr.args) >= 2:
                return iter_extent_of_(expr.args[1], shape_table)
            if attr in ("cumsum", "cumprod", "maximum.accumulate", "minimum.accumulate") and expr.args:
                base = iter_extent_of_(expr.args[0], shape_table)
                if base is None:
                    return None
                if kwarg_or_pos(expr.args, expr.keywords, 1, "axis") is not None or len(base) == 1:
                    return base
                return None
            # ``np.pad(src, pad_width, ...)`` -> each source axis grown by its
            # ``before + after`` width (scalar R or per-axis tuple). The stencil
            # ghost cells / the vector variants' unpadded component axis.
            if attr == "pad" and expr.args:
                base = iter_extent_of_(expr.args[0], shape_table)
                if base is None:
                    return None
                pad_arg = kwarg_or_pos(expr.args, expr.keywords, 1, "pad_width")
                return pad_output_extent(base, pad_arg)
            # ``np.concatenate((a, b, ...), axis=k)`` -> the operands' common
            # shape with axis ``k`` summed across operands (dwt2d Haar
            # recompose). Other axes are taken from the first operand.
            if attr == "concatenate" and expr.args:
                try:
                    names, shapes, axis = concat_operands_axis(expr.args, expr.keywords, shape_table)
                except NotImplementedError:
                    return None
                base = list(shapes[0])
                summed = "(" + ") + (".join(s[axis] for s in shapes) + ")"
                base[axis] = summed
                return tuple(const_or_name(t) for t in base)
            # ``np.stack((a, b, ...), axis=k)`` -> the operands' common shape with a NEW
            # size-N axis inserted at k (N = number of operands); out's rank = rank + 1.
            if attr == "stack" and expr.args:
                try:
                    names, shapes, unused = concat_operands_axis(expr.args, expr.keywords, shape_table)
                    axis = stack_axis(expr.args, expr.keywords, len(shapes[0]))
                except NotImplementedError:
                    return None
                out = [const_or_name(t) for t in shapes[0]]
                out.insert(axis, const_(len(names)))
                return tuple(out)
        # A BINARY ufunc BROADCASTS its operands; the first-arg rule below is only right when one
        # operand carries the whole extent. ``np.equal(a[:, None, :], b[:, :, None])`` -- what the
        # frontend rewrites ``a[:, None, :] == b[:, :, None]`` into -- came back with the LEFT
        # side's (N, 1, F) rather than (N, F, F), so cp2k_density_matrix_trs4's match plane was
        # declared a third of its size and every reduction over it ran short. The Compare spelling
        # already broadcast; the function spelling now agrees with it.
        if isinstance(expr.func, ast.Attribute) and expr.func.attr in BROADCASTING_UFUNCS and len(expr.args) >= 2:
            ext = broadcast_children(list(expr.args), shape_table)
            if ext is not None:
                return ext
        # Elementwise/unary math functions (abs, sqrt, exp, sin, cos, log, ...)
        # preserve the operand's iter extent -- pick the first arg that resolves.
        for arg in expr.args:
            ext = iter_extent_of_(arg, shape_table)
            if ext is not None:
                return ext
        return None
    if isinstance(expr, ast.Compare):
        # ``a == b`` etc: broadcast the operand extents (numpy rules) so an outer
        # comparison ``a[:, None] == b[None, :]`` with (N, 1) and (1, N) yields
        # (N, N), not just the left side's (N, 1) (smith_waterman/needleman_wunsch
        # substitution matrix).
        return broadcast_children([expr.left, *expr.comparators], shape_table)
    if isinstance(expr, ast.BoolOp):
        return broadcast_children(expr.values, shape_table)
    if isinstance(expr, ast.Subscript):
        name = name_id(expr.value)
        if name:
            shape = shape_table.get(name)
        else:
            # Chained scalar-indexed base ``A[i, j][outer]``: resolve the base's
            # residual shape so the outer axes below index it -- not yet flattened
            # to a single Name subscript at harvest time.
            shape = chained_base_shape(expr.value, shape_table)
            if shape is None:
                # Any other sized base: a CALL result indexed directly, which is what
                # ``np.expand_dims(np.take(x, 0, axis=k), axis=k)`` becomes once the frontend
                # rewrites expand_dims to a newaxis index. Its own extent is the shape the
                # axes below index against.
                base_ext = iter_extent_of_(expr.value, shape_table)
                shape = tuple(ast.unparse(e) for e in base_ext) if base_ext is not None else None
        axes = slice_axes(expr)
        ext: list[ast.expr] = []
        src_axis = 0  # source-axis pointer -- advances on Slice / scalar
        # axes, NOT on ``None`` (newaxis -- pure result-axis insertion).
        # Explicit (non-special) axes consumed; Ellipsis fills the rest.
        n_src_consumers = sum(1 for ax in axes if not is_special_axis(ax))
        # numpy ADVANCED indexing: several integer-ARRAY indices together broadcast
        # into a SINGLE group of result axes, not one per array -- ``u2[q, r, s]``
        # with q/r/s all (J,) -> (J,), not (J,J,J) (fft_3d checksum gather). Collect
        # the index-array shapes and emit the broadcast once, at the first
        # index-array's result position.
        idx_array_extents: list[tuple[ast.expr, ...]] = []
        idx_group_pos: int | None = None
        for ax in axes:
            if isinstance(ax, ast.Constant) and ax.value is None:
                # numpy newaxis -- inserts a length-1 result axis without
                # consuming a source axis.
                ext.append(const_(1))
                continue
            if isinstance(ax, ast.Constant) and ax.value is Ellipsis:
                # numpy Ellipsis -- expands to the full slices of every source
                # axis the other (explicit) entries do not consume, each
                # contributing that axis's full extent. Needs the source rank.
                if not shape:
                    return None
                for unused in range(max(len(shape) - n_src_consumers, 0)):
                    if src_axis >= len(shape):
                        return None
                    ext.append(const_or_name(shape[src_axis]))
                    src_axis += 1
                continue
            if isinstance(ax, ast.Slice):
                axis_len = const_or_name(shape[src_axis]) if shape and src_axis < len(shape) else None
                lo = resolve_negative(ax.lower, axis_len) if ax.lower is not None else const_(0)
                hi = resolve_negative(ax.upper, axis_len) if ax.upper is not None else axis_len
                if hi is None or lo is None:
                    return None
                if isinstance(hi, ast.Constant) and isinstance(lo, ast.Constant):
                    raw: ast.expr = const_(hi.value - lo.value)
                elif isinstance(lo, ast.Constant) and lo.value == 0:
                    # ``hi - 0`` simplifies to ``hi``.
                    raw = hi
                else:
                    # ``hi - lo`` algebraic simplification:
                    #  ``(lo + K) - lo`` -> K  (slice ``[i:i+K]``)
                    #  ``(K + lo) - lo`` -> K
                    #  ``lo - lo``       -> 0
                    simplified = simplify_sub(hi, lo)
                    raw = simplified if simplified is not None else ast.BinOp(left=hi, op=ast.Sub(), right=lo)
                # Strided slice ``a[lo:hi:k]`` has ``ceil((hi - lo) / k)``
                # elements (== ``len(range(lo, hi, k))``). dwt2d's Haar
                # ``b[:, 0::2]`` over an even axis ``s`` -> ``s // 2``.
                step = slice_step_any(ax)
                if step_is_negative(step) and (ax.lower is not None or ax.upper is not None):
                    # A BOUNDED reverse slice (``a[lo::-1]``/``a[:hi:-1]``): numpy flips
                    # the bound defaults under a negative step, so ``raw = hi - lo`` is
                    # NOT the element count and the ceil below would over-count, reading
                    # OOB. Only full-axis reverse (``a[::-k]``) is reliably counted --
                    # bail on the bounded form.
                    return None
                if isinstance(step, ast.expr):
                    # Symbolic stride: ceil(raw / step) with the divisor carried as an expression.
                    # No abs() -- a bounded slice's step is positive or the numpy source is empty.
                    exact = span_multiple_of(raw, step)
                    if exact is not None:
                        ext.append(exact)
                    else:
                        ext.append(
                            ast.BinOp(
                                left=ast.BinOp(
                                    left=ast.BinOp(left=raw, op=ast.Add(), right=step), op=ast.Sub(), right=const_(1)
                                ),
                                op=ast.FloorDiv(),
                                right=step,
                            )
                        )
                elif step is not None and step != 1:
                    # Element count is ceil(raw / |step|) -- a full-axis negative step
                    # (reverse) spans the same number of elements as its positive
                    # magnitude.
                    astep = abs(step)
                    if isinstance(raw, ast.Constant):
                        ext.append(const_((raw.value + astep - 1) // astep))
                    else:
                        ext.append(
                            ast.BinOp(
                                left=ast.BinOp(left=raw, op=ast.Add(), right=const_(astep - 1)),
                                op=ast.FloorDiv(),
                                right=const_(astep),
                            )
                        )
                else:
                    ext.append(raw)
            elif isinstance(ax, ast.Name) and shape_table.get(ax.id):
                # Fancy-index gather: ``arr[idx]``, ``idx`` a known-shape int array
                # (a scalar Name -- loop var/symbol -- has no shape, contributes
                # nothing). Multiple index arrays broadcast into ONE result-axis
                # group -- record, emit later.
                if idx_group_pos is None:
                    idx_group_pos = len(ext)
                idx_array_extents.append(tuple(const_or_name(s) for s in shape_table[ax.id]))
            elif advanced_index_rank(ax, shape_table):
                # Advanced-index EXPRESSION axis (``edge_idx[:, :, 0] - 1``): a
                # sliced/offset index array used as a gather index. Its result
                # extent is its own extent; joins the same broadcast group as any
                # bare-Name index.
                ie = iter_extent_of_(ax, shape_table)
                if ie is not None:
                    if idx_group_pos is None:
                        idx_group_pos = len(ext)
                    idx_array_extents.append(tuple(ie))
            # scalar axis: contributes nothing to result extent but advances
            # the source-axis pointer.
            src_axis += 1
        if idx_array_extents:
            # Broadcast the index extents together (numpy advanced index). Picking the
            # LONGEST is not the numpy rule and is wrong whenever the ranks tie:
            # ``nbfp[ti[None, :, None], tj[:, None, :], 0]`` with ti (1, I, 1) and tj (P, 1, J)
            # is (P, I, J), not the first operand's (1, I, 1).
            group = idx_array_extents[0]
            for other in idx_array_extents[1:]:
                group = broadcast_extents(group, other)
            ext[idx_group_pos:idx_group_pos] = list(group)
        # Append any trailing axes that the Subscript didn't index --
        # numpy implicitly takes the full extent for omitted trailing
        # axes. ``path[:]`` on a 2-D ``path`` returns a 2-D extent.
        if shape and src_axis < len(shape):
            for i in range(src_axis, len(shape)):
                ext.append(const_or_name(shape[i]))
        return tuple(ext) if ext else None
    if isinstance(expr, ast.Attribute):
        # ``A.T`` reverses the axes; ``.real`` / ``.imag`` pick a component and keep them. The
        # ``np.transpose`` branch above already claims to answer for ``x.T``, but an attribute is
        # never a Call and never reached it -- so a transpose spelled the short way resolved in the
        # RANK table and to nothing here, and the two disagreed about the same expression.
        if expr.attr == "T":
            base = iter_extent_of_(expr.value, shape_table)
            return None if base is None else tuple(reversed(base))
        if expr.attr in ("real", "imag"):
            return iter_extent_of_(expr.value, shape_table)
    return None


def broadcast_extents(l_ext: tuple[ast.expr, ...], r_ext: tuple[ast.expr, ...]) -> tuple[ast.expr, ...]:
    """Numpy broadcasting on two extent tuples: align from the right, pad the
    shorter on the left with implicit 1. Per aligned axis, a literal ``1``
    stretches to the other side; otherwise the left side wins (shape mismatches
    surface later, at scalarise time)."""
    rank = max(len(l_ext), len(r_ext))
    l_pad = (const_(1),) * (rank - len(l_ext)) + l_ext
    r_pad = (const_(1),) * (rank - len(r_ext)) + r_ext
    out: list[ast.expr] = []
    for l, r in zip(l_pad, r_pad):
        # A size-1 axis on either side stretches to the other's extent -- a size-1
        # RIGHT axis must yield the LEFT extent, not silently keep the (already
        # equal) left, so ``B(N, M) * a(N, 1)`` broadcasts to ``M`` rather than
        # dropping it.
        if extent_is_one(l):
            out.append(r)
        elif extent_is_one(r):
            out.append(l)
        else:
            # Equal extents keep either; a genuine runtime-1 mismatch can't resolve
            # statically, so take the left (the scalarizer indexes each operand by
            # its own shape, so a per-operand size-1 axis still reads with a 0).
            out.append(l)
    return tuple(out)


def extent_is_one(node: ast.expr) -> bool:
    """True when an extent is the literal ``1`` -- a bare ``Constant(1)`` or a
    node that unparses to ``"1"`` (a shape token ``"1"`` re-parsed via
    ``const_or_name``). A symbolic extent that is only 1 at runtime cannot be
    detected here."""
    if is_const_one(node):
        return True
    try:
        return ast.unparse(node).strip() == "1"
    except (AttributeError, ValueError):
        return False


def extent_is_scalar(ext: tuple[ast.expr, ...] | None) -> bool:
    """True when a broadcast extent is entirely size-1: such a value is a SCALAR
    in numpyto's model, not a ``T t[1]`` array (e.g. ``t = (a[i] > x)`` with ``x``
    shape ``(1,)``). Misregistering it as an array desyncs scalar uses (``t = 0``,
    ``if t``) from the array-style ``memset``/``t[__w0] = ...`` writes the extent
    would drive -- doesn't compile. Rank-0 (empty tuple) is already scalar."""
    return ext is not None and all(extent_is_one(e) for e in ext)


def is_integer_expr(node: ast.AST, local_dtypes: dict[str, str], array_names: set[str] = frozenset()) -> bool:
    """Best-effort: does ``node`` evaluate to an integer? Recognises int Constants,
    Names tagged integer in ``local_dtypes``, and ``+ - * % //`` over integer
    operands.

    An ARRAY Name (in ``array_names``) counts as integer only when
    ``local_dtypes`` explicitly tags it int -- untagged arrays default to float.
    A non-array Name (loop iter/shape symbol) is integer by default, so
    ``j % nx`` (j int-tagged, nx a symbol) stays integer."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        dt = local_dtypes.get(node.id)
        if dt is not None:
            return dtypes.is_integer(dt)
        return node.id not in array_names  # untagged array -> float default
    if isinstance(node, ast.Subscript):
        # An element/gather of an integer-typed array is itself integer
        # (QE index tables ``dfftt_nl[gki]``, ``igk_exx[:n, k]``); an untagged
        # base defaults to float.
        base = node.value
        if isinstance(base, ast.Name):
            dt = local_dtypes.get(base.id)
            return dt is not None and dtypes.is_integer(dt)
        return is_integer_expr(base, local_dtypes, array_names)
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Mod, ast.FloorDiv)):
            return False
        return is_integer_expr(node.left, local_dtypes, array_names) and is_integer_expr(
            node.right, local_dtypes, array_names
        )
    if isinstance(node, ast.UnaryOp):
        return is_integer_expr(node.operand, local_dtypes, array_names)
    return False


def as_float64(node: ast.expr) -> ast.expr:
    """Wrap ``node`` in ``np.float64(...)`` -- the cast both native emitters render
    (``(double)(x)`` / ``REAL(x, kind=c_double)``). Same spelling lowering's
    ``TrueDivisionPromoter`` uses, so the two paths stay one convention."""
    return ast.Call(func=ast.Attribute(value=name_("np"), attr="float64", ctx=ast.Load()), args=[node], keywords=[])


def provably_integer(node: ast.expr, local_dtypes: dict[str, str]) -> bool:
    """True when ``node``'s VALUE is certainly integer: an int Constant, or a Name /
    element-of-Name tagged with an integer dtype.

    Deliberately stricter than :func:`is_integer_expr`, which reads an UNTAGGED
    non-array Name as integer -- right when classifying size symbols, wrong here:
    a ufunc operand that is merely untagged (an undeclared float scalar, a float
    array) must not be taken for an integer, because that decides whether the
    result dtype is integral."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        dt = local_dtypes.get(node.id)
        return dt is not None and dtypes.is_integer(dt)
    if isinstance(node, ast.Subscript):
        return provably_integer(node.value, local_dtypes)
    return False


def all_integer_operands(args: list[ast.expr], local_dtypes: dict[str, str] | None) -> bool:
    """True when EVERY operand of a ufunc call is provably integer -- i.e. numpy would
    promote the result to an integer dtype. An absent dtype table answers False."""
    if not local_dtypes or not args:
        return False
    return all(provably_integer(a, local_dtypes) for a in args)


#: Elementwise ufuncs whose numpy result dtype is the PROMOTED OPERAND dtype, so an
#: all-integer call returns an integer array. Their hoisted temp must be declared
#: integer, not the double default: a double temp rounds every value above 2**53
#: (``np.power(3, 39)`` came back 11 short). ``divide`` is NOT here -- it always
#: returns float, and its cast is applied in :func:`expand_divide`.
#: numpy ufuncs whose result extent is the BROADCAST of their operands, not the first one's. Only
#: functions whose every argument is an OPERAND belong here: a call whose second positional slot is
#: an axis, a shape or a tolerance must keep the first-arg rule, or that slot's extent would be
#: folded into the result.
BROADCASTING_UFUNCS: set[str] = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "true_divide",
    "floor_divide",
    "power",
    "float_power",
    "mod",
    "remainder",
    "fmod",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "hypot",
    "arctan2",
    "logaddexp",
    "logaddexp2",
    "copysign",
    "nextafter",
    "heaviside",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "equal",
    "not_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "left_shift",
    "right_shift",
}

INT_PRESERVING_ELEMENTWISE: set[str] = {"add", "subtract", "multiply", "power", "maximum", "minimum"}


def broadcast_children(
    children: list[ast.expr], shape_table: dict[str, tuple[str, ...]]
) -> tuple[ast.expr, ...] | None:
    """Fold every child's iter extent through numpy broadcasting, skipping
    scalar (None-extent) children. Returns the broadcast extent, or None when
    no child has an extent. Shared by the Compare / BoolOp extent branches."""
    acc: tuple[ast.expr, ...] | None = None
    for child in children:
        ext = iter_extent_of_(child, shape_table)
        if ext is None:
            continue
        acc = ext if acc is None else broadcast_extents(acc, ext)
    return acc


def resolve_negative(node: ast.AST, axis_len: ast.expr | None) -> ast.expr | None:
    """Resolve a slice bound: negative int -> ``axis_len - K``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value < 0:
        if axis_len is None:
            return None
        return ast.BinOp(left=axis_len, op=ast.Sub(), right=const_(-node.value))
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and axis_len is not None
    ):
        return ast.BinOp(left=axis_len, op=ast.Sub(), right=const_(node.operand.value))
    return node


def sliced_index_rank(axes: list[ast.expr]) -> int | None:
    """Rank of an index array read through ``axes``: one per slice and one per newaxis, since ``mat[:, None]``
    is rank 2. ``None`` when no slice keeps an axis of the array itself."""
    slices = sum(1 for a in axes if isinstance(a, ast.Slice))
    newaxes = sum(1 for a in axes if isinstance(a, ast.Constant) and a.value is None)
    return slices + newaxes if slices else None


def advanced_index_rank(expr: ast.expr, shape_table: dict[str, tuple[str, ...]]) -> int | None:
    """Broadcast rank of an advanced-index EXPRESSION used as one axis of an outer
    gather, or ``None`` if ``expr`` isn't one: a Subscript on a known array with
    >=1 Slice axis, possibly wrapped in arithmetic (ICON's ``edge_idx[:, :, 0] -
    1``). Rank = number of Slice axes. Lets the SliceFusion RHS scalarizer
    recognise ``w[idx[:, :, 0] - 1, jk, blk[:, :, 0] - 1]`` as an advanced-index
    group, not a plain scalar axis."""
    if isinstance(expr, ast.Subscript):
        name = name_id(expr.value)
        if name and shape_table.get(name):
            return sliced_index_rank(slice_axes(expr))
        return None
    if isinstance(expr, ast.Name):
        # A bare index ARRAY is an advanced index of its own rank. Only the sliced spelling was
        # recognised, so an OFFSET gather (``coulomb_table_f[ri + 1]``) read as a scalar axis.
        shape = shape_table.get(expr.id)
        return len(shape) if shape else None
    if isinstance(expr, ast.BinOp):
        return advanced_index_rank(expr.left, shape_table) or advanced_index_rank(expr.right, shape_table)
    if isinstance(expr, ast.UnaryOp):
        return advanced_index_rank(expr.operand, shape_table)
    return None


#: Element value of an array constructor, for the constructors whose fill is DEFINED. ``empty`` /
#: ``empty_like`` / ``ndarray`` are absent on purpose: their contents are whatever the allocation
#: held, and naming a value for them would put an invented number into the emitted kernel.
CTOR_FILL: dict[str, float] = {"zeros": 0.0, "zeros_like": 0.0, "ones": 1.0, "ones_like": 1.0}


def ctor_fill_element(expr: ast.Call) -> ast.expr | None:
    """The scalar every element of ``np.zeros(...)`` / ``np.ones_like(...)`` / ``np.full(...)``
    holds, or ``None`` when the call is not such a constructor."""
    if not (
        isinstance(expr.func, ast.Attribute)
        and isinstance(expr.func.value, ast.Name)
        and expr.func.value.id in ("np", "numpy")
    ):
        return None
    attr = expr.func.attr
    if attr in ("full", "full_like") and len(expr.args) >= 2:
        return copy.deepcopy(expr.args[1])
    value = CTOR_FILL.get(attr)
    return None if value is None else const_(value)


def span_multiple_of(span: ast.expr, step: ast.expr) -> ast.expr | None:
    """The other factor when ``span`` is syntactically ``<expr> * step``, else ``None``.

    ``ceil(A * s / s) == A`` exactly, for every positive ``s``, so a strided slice whose span is a
    multiple of its stride has a plain extent. The pooling kernels slice
    ``padded[kz:kz + out * stride:stride]``; unfolded that extent reads
    ``(out * stride + stride - 1) // stride`` -- the same number as ``out``, spelled so that no
    token comparison can see it.
    """
    if not (isinstance(span, ast.BinOp) and isinstance(span.op, ast.Mult)):
        return None
    step_txt = ast.unparse(step)
    for factor, other in ((span.left, span.right), (span.right, span.left)):
        if ast.unparse(factor) == step_txt:
            return copy.deepcopy(other)
    return None


def concat_operands_axis(
    args: list[ast.expr], kwargs: list[ast.keyword] | None, shape_table: dict[str, tuple[str, ...]]
) -> tuple[list[str | None], list[tuple[str, ...]], int]:
    """Shared parse for ``np.concatenate`` / ``np.stack``-style calls: return
    ``(names, shapes, axis)``. The sequence is the first positional arg (a
    tuple/list of array Names); ``axis`` is a keyword or the 2nd positional
    (default 0, normalised against the operand rank)."""
    kwargs = kwargs or []
    if not args:
        raise NotImplementedError("np.concatenate needs a sequence arg")
    seq = args[0]
    if not isinstance(seq, (ast.Tuple, ast.List)):
        raise NotImplementedError("np.concatenate: sequence must be a tuple/list")
    # ``axis`` may be a plain or negated literal (``axis=-1`` parses as
    # ``UnaryOp(USub, Constant(1))``, not ``Constant(-1)``); ``const_int``
    # accepts both, and the ``axis < 0`` fixup below resolves it mod rank.
    # ``axis=None`` is not "no axis": numpy FLATTENS every operand and concatenates the results,
    # which is a different output rank. Refuse rather than fall through to the axis-0 default.
    axis_node = kwarg_or_pos(args, kwargs, 1, "axis")
    if isinstance(axis_node, ast.Constant) and axis_node.value is None:
        raise NotImplementedError("np.concatenate(axis=None) flattens every operand first; not lowered")
    axis = axis_literal_or_refuse(axis_node, "np.concatenate", 0)
    names: list[str | None] = []
    shapes: list[tuple[str, ...]] = []
    for op in seq.elts:
        if isinstance(op, ast.Name):
            s = shape_table.get(op.id)
            if s is None:
                raise NotImplementedError(f"np.concatenate: shape of {op.id} unknown")
            names.append(op.id)
            shapes.append(tuple(s))
            continue
        # A non-Name operand -- dwt2d's rotate ``np.concatenate((e[:, 1:], e[:,
        # 0:1]), axis=1)`` -- still has a resolvable EXTENT, all the shape rule
        # needs. The expander materialises such operands into Names before
        # calling this, so ``names`` stays complete; reject a leftover None
        # rather than emit a nameless read.
        ext = iter_extent_of_(op, shape_table)
        if ext is None:
            raise NotImplementedError("np.concatenate: operand must be a Name")
        names.append(None)
        shapes.append(tuple(ast.unparse(e) for e in ext))
    rank = len(shapes[0])
    if any(len(s) != rank for s in shapes):
        raise NotImplementedError("np.concatenate: mixed ranks unsupported")
    if axis < 0:
        axis += rank
    return names, shapes, axis


def pad_output_extent(src_extent: tuple[ast.expr, ...], pad_arg: ast.expr | None) -> tuple[ast.expr, ...] | None:
    """Output extent of ``np.pad``: each source axis grown by ``before+after``.

    ``src_extent`` is the tuple of source-axis extent AST nodes; returns the
    per-axis output extent nodes, or ``None`` if ``pad_arg`` is unsupported."""
    widths = pad_widths(pad_arg, len(src_extent))
    if widths is None:
        return None
    out = []
    for d, (before, after) in zip(src_extent, widths):
        total = ast.BinOp(left=copy.deepcopy(before), op=ast.Add(), right=copy.deepcopy(after))
        out.append(ast.BinOp(left=d, op=ast.Add(), right=total))
    return tuple(out)
