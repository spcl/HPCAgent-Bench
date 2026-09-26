"""Iteration extent of array-valued expressions and numpy broadcasting over them."""

import ast
import copy
from collections.abc import Callable, Sequence
from types import NotImplementedType

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

__all__ = [
    "BROADCASTING_UFUNCS",
    "CTOR_FILL",
    "EXPR_EXTENT",
    "INT_PRESERVING_ELEMENTWISE",
    "NP_CALL_EXTENT",
    "UNHANDLED",
    "advanced_index_rank",
    "all_integer_operands",
    "arange_extent",
    "as_float64",
    "attribute_extent",
    "binop_extent",
    "boolop_extent",
    "broadcast_children",
    "broadcast_extents",
    "call_extent",
    "chained_base_shape",
    "compare_extent",
    "concat_call_extent",
    "concat_extent",
    "concat_operands_axis",
    "constructor_extent",
    "contraction_call_extent",
    "contraction_result_extent",
    "ctor_fill_element",
    "diag_extent",
    "diagonal_extent",
    "ellipsis_extent",
    "expand_dims_extent",
    "extent_is_one",
    "extent_is_scalar",
    "eye_extent",
    "fft_extent",
    "first_operand_extent",
    "ifexp_extent",
    "index_array_extent",
    "is_integer_expr",
    "is_numpy_receiver",
    "iter_extent_of",
    "matmul_extent",
    "method_extent",
    "method_reshape_extent",
    "moveaxis_extent",
    "name_extent",
    "np_call_extent",
    "operand_token_shape",
    "outer_extent",
    "pad_extent",
    "pad_output_extent",
    "provably_integer",
    "reduction_extent",
    "reshape_call_extent",
    "reshape_target",
    "resolve_negative",
    "scan_extent",
    "second_operand_extent",
    "slice_count",
    "sliced_index_rank",
    "span_multiple_of",
    "squeeze_extent",
    "stack_extent",
    "subscript_base_shape",
    "subscript_extent",
    "sum_width_tokens",
    "swapaxes_extent",
    "take_extent",
    "transpose_extent",
    "unaryop_extent",
    "unsized_extent",
]


def operand_token_shape(node: ast.expr, shape_table: dict[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    """Residual shape TOKENS (not AST nodes -- stays consistent with the shape
    table) of an einsum/contraction operand. Bare ``Name(A)`` -> A's declared
    shape. Anything else (a Subscript slice/index chain, a matmul, a
    shape-preserving call...) routes through the general ``iter_extent_of``
    sizer, which already knows a partial slice's actual bound (``a[k:k+H]``
    -> ``H``, not the base axis's full extent) alongside every other form it
    resolves. ``None`` if unresolvable."""
    nm = name_id(node)
    if nm:
        return shape_table.get(nm)
    ext = iter_extent_of(node, shape_table)
    return tuple(ast.unparse(e) for e in ext) if ext is not None else None


def chained_base_shape(node: ast.expr, shape_table: dict[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    """Residual token-shape of a SCALAR-chained subscript base ``A[i, j][...]``,
    when every inner index is a single-axis scalar (int Constant / bare Name):
    numpy combined-basic-indexing drops one leading axis per scalar, e.g.
    ``psi_frag[f]`` on ``(F, X, Y, Z, K)`` -> ``(X, Y, Z, K)``. Lets
    :func:`iter_extent_of` size a chained access like ``psi_frag[f][..., 0]``
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
    extents = [iter_extent_of(operand, shape_table) for operand in operands]
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


type Extent = tuple[ast.expr, ...]
type ShapeTable = dict[str, tuple[str, ...]]
#: A call sizer's answer: an extent, None (known to be unsized), or :data:`UNHANDLED`.
type CallSize = Extent | None | NotImplementedType

#: What an ``np.<attr>`` sizer returns for an argument form it does not cover: the call then falls
#: through to the broadcasting / first-operand rules in :func:`call_extent`.
UNHANDLED = NotImplemented


def iter_extent_of(expr: ast.expr, shape_table: ShapeTable) -> Extent | None:
    """Iteration extent of an array-valued expression. ``Name(A)`` -> A's full
    shape; ``Subscript(A, axes)`` -> upper-minus-lower per Slice axis in order
    (non-Slice axes are scalar, contribute nothing); negative slice bounds
    resolve against the operand's declared shape. ``None`` for unsupported
    forms -- caller falls through to ``NotImplementedError``.
    """
    sizer = EXPR_EXTENT.get(type(expr))
    return None if sizer is None else sizer(expr, shape_table)


def name_extent(expr: ast.Name, shape_table: ShapeTable) -> Extent | None:
    shape = shape_table.get(expr.id)
    return None if shape is None else tuple(const_or_name(s) for s in shape)


def binop_extent(expr: ast.BinOp, shape_table: ShapeTable) -> Extent | None:
    """``@`` contracts (:func:`matmul_extent`); every other operator broadcasts its operands axis by
    axis, aligned from the right, picking the non-1 axis at each position (numpy rules): ``A + b``
    with A:(N, M), b:(M,) -> (N, M); ``X + Y[:, None]`` with X:(M,), Y[:, None]:(N, 1) -> (N, M)."""
    if isinstance(expr.op, ast.MatMult):
        return matmul_extent(expr, shape_table)
    return broadcast_children([expr.left, expr.right], shape_table)


def matmul_extent(expr: ast.BinOp, shape_table: ShapeTable) -> Extent | None:
    """numpy ``@`` treats the last two axes as the matrix; leading axes are batched and broadcast::

    1-D @ 1-D                  -> scalar (None)
    2-D @ 1-D                  -> (M,)
    1-D @ 2-D                  -> (N,)
    2-D @ 2-D                  -> (M, N)
    (..., M, K) @ (..., K, N)  -> (..., M, N)  (batched)
    (..., M, K) @ (K, N)       -> (..., M, N)  (broadcast 2-D rhs)
    (M, K) @ (..., K, N)       -> (..., M, N)  (broadcast 2-D lhs)
    (K,) @ (..., K, N)         -> (..., N)
    (..., M, K) @ (K,)         -> (..., M)
    """
    l_ext = iter_extent_of(expr.left, shape_table)
    r_ext = iter_extent_of(expr.right, shape_table)
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
    if ll >= 2 and rl >= 2:
        batch = broadcast_extents(l_ext[:-2], r_ext[:-2])
        return tuple(batch) + (l_ext[-2], r_ext[-1])
    if ll == 1 and rl > 2:
        return tuple(r_ext[:-2]) + (r_ext[-1],)
    if rl == 1 and ll > 2:
        return tuple(l_ext[:-2]) + (l_ext[-2],)
    return None


def unaryop_extent(expr: ast.UnaryOp, shape_table: ShapeTable) -> Extent | None:
    return iter_extent_of(expr.operand, shape_table)


def compare_extent(expr: ast.Compare, shape_table: ShapeTable) -> Extent | None:
    """``a == b`` etc: the operand extents broadcast (numpy rules), so an outer comparison
    ``a[:, None] == b[None, :]`` with (N, 1) and (1, N) yields (N, N), not the left side's (N, 1)."""
    return broadcast_children([expr.left, *expr.comparators], shape_table)


def boolop_extent(expr: ast.BoolOp, shape_table: ShapeTable) -> Extent | None:
    return broadcast_children(expr.values, shape_table)


def ifexp_extent(expr: ast.IfExp, shape_table: ShapeTable) -> Extent | None:
    return broadcast_children([expr.body, expr.orelse], shape_table)


def attribute_extent(expr: ast.Attribute, shape_table: ShapeTable) -> Extent | None:
    """``A.T`` reverses the axes; ``.real`` / ``.imag`` pick a component and keep them. An attribute
    is never a Call, so the ``np.transpose`` sizer does not answer for ``x.T``; this does."""
    if expr.attr == "T":
        base = iter_extent_of(expr.value, shape_table)
        return None if base is None else tuple(reversed(base))
    if expr.attr in ("real", "imag"):
        return iter_extent_of(expr.value, shape_table)
    return None


def is_numpy_receiver(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id in ("np", "numpy")


def call_extent(expr: ast.Call, shape_table: ShapeTable) -> Extent | None:
    """A call's result extent: the ``np.fft`` family, the method and function spellings of the
    shape-changing ops, reductions over an axis, then a broadcasting binary ufunc, then any
    elementwise function (the first operand whose extent resolves)."""
    fft = np_fft_attr(expr)
    if fft is not None:
        return fft_extent(fft, expr, shape_table)
    method = expr.func.attr if isinstance(expr.func, ast.Attribute) and not is_numpy_receiver(expr.func.value) else None
    if method == "reshape" and expr.args:
        return method_reshape_extent(expr, shape_table)
    if is_reduction_call(expr):
        return reduction_extent(expr, shape_table, method_form=method is not None)
    if method is not None:
        sized = method_extent(method, expr, shape_table)
        if sized is not UNHANDLED:
            return sized
    attr = np_call_attr(expr.func)
    if attr is not None:
        sized = np_call_extent(attr, expr, shape_table)
        if sized is not UNHANDLED:
            return sized
    # A BINARY ufunc BROADCASTS its operands; the first-arg rule below is only right when one
    # operand carries the whole extent. ``np.equal(a[:, None, :], b[:, :, None])`` -- what the
    # frontend rewrites ``a[:, None, :] == b[:, :, None]`` into -- is (N, F, F), not the left
    # side's (N, 1, F), the same answer the Compare spelling gives.
    if isinstance(expr.func, ast.Attribute) and expr.func.attr in BROADCASTING_UFUNCS and len(expr.args) >= 2:
        ext = broadcast_children(list(expr.args), shape_table)
        if ext is not None:
            return ext
    # Elementwise/unary math functions (abs, sqrt, exp, sin, cos, log, ...)
    # preserve the operand's iter extent -- pick the first arg that resolves.
    for arg in expr.args:
        ext = iter_extent_of(arg, shape_table)
        if ext is not None:
            return ext
    return None


def fft_extent(fft: str, expr: ast.Call, shape_table: ShapeTable) -> Extent | None:
    """``np.fft.*`` is a two-level attribute (``func.value`` is ``np.fft``). ``fftfreq(n)`` builds a
    length-n 1-D array; complex transforms/shifts are shape-preserving. REAL-FFT variants change the
    transformed axis length (``rfftfreq(n)`` -> ``n//2+1``; ``rfftn``/``irfftn`` resize the last
    axis) and are NOT sized here -- None rather than a wrong extent."""
    if fft == "fftfreq" and expr.args:
        return (copy.deepcopy(expr.args[0]),)
    if fft in ("fftn", "ifftn", "fft", "ifft", "fft2", "ifft2", "fftshift", "ifftshift") and expr.args:
        return iter_extent_of(expr.args[0], shape_table)
    return None


def reshape_target(elts: list[ast.expr], source: ast.expr, shape_table: ShapeTable) -> Extent | None:
    """``elts`` with a single ``-1`` placeholder resolved to ``total_source_size / product(other
    target dims)`` (``/`` renders as integer division in C/Fortran for int dims; ``//`` is not valid
    C when the token is emitted). More than one ``-1`` is ambiguous: None."""
    neg1 = [i for i, e in enumerate(elts) if const_int(e) == -1]
    if len(neg1) == 1:
        base = iter_extent_of(source, shape_table)
        if base is None:
            return None
        others = [e for j, e in enumerate(elts) if j != neg1[0]]
        denom = mul_exts(others) if others else const_(1)
        elts[neg1[0]] = ast.BinOp(left=mul_exts(base), op=ast.Div(), right=denom)
    elif neg1:
        return None
    return tuple(elts)


def method_reshape_extent(expr: ast.Call, shape_table: ShapeTable) -> Extent | None:
    """Method-form ``<expr>.reshape(...)``: the receiver is the operand, not ``np``. Left unsized, a
    reshape target like ``X = (Yf @ C).reshape(shp)`` poisons every derived shape."""
    if len(expr.args) == 1 and isinstance(expr.args[0], (ast.Tuple, ast.List)):
        elts = list(expr.args[0].elts)
    elif len(expr.args) == 1 and isinstance(expr.args[0], (ast.Name, ast.Constant, ast.BinOp, ast.UnaryOp)):
        elts = [expr.args[0]]
    else:
        elts = list(expr.args)  # varargs ``.reshape(a, b, c)``
    return reshape_target(elts, expr.func.value, shape_table)


def reduction_extent(expr: ast.Call, shape_table: ShapeTable, *, method_form: bool) -> Extent | None:
    """Axis-aware reduction: ``np.sum(operand, axis=k)`` -> operand's extent with axis k removed
    (size 1 if keepdims); axis=None collapses to a scalar (None).

    The METHOD spelling carries its operand in the RECEIVER, not in ``args[0]``; ``m.any(axis=-1)``
    must size exactly as ``np.any(m, axis=-1)`` does. ``read_axis_keepdims`` reads the axis from
    positional slot 1, so the method's args are shifted by one into the vocabulary it expects."""
    red_args = ([expr.func.value] + list(expr.args)) if method_form else list(expr.args)
    if not red_args:
        return None
    axes, keepdims = read_axis_keepdims(red_args, expr.keywords)
    if axes is None:
        return None
    base = iter_extent_of(red_args[0], shape_table)
    if base is None:
        return None
    n = len(base)
    norm = {a % n for a in axes}
    if keepdims:
        return tuple(const_(1) if i in norm else base[i] for i in range(n))
    return tuple(base[i] for i in range(n) if i not in norm) or None


def method_extent(method: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """Method spelling of a shape op: the receiver IS the operand, so ``rho.copy()`` says what
    ``np.copy(rho)`` says and is routed to that sizer once, here. ``astype`` and ``flatten`` have no
    numpy function twin and answer directly. :data:`UNHANDLED` for any other method."""
    if method == "astype":
        return iter_extent_of(expr.func.value, shape_table)  # dtype only, never the shape
    if method in ("ravel", "flatten"):
        base = iter_extent_of(expr.func.value, shape_table)
        return None if base is None else (mul_exts(base),)
    if method in ARRAY_METHOD_SHAPE_OPS:
        routed = ast.Call(
            func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=method, ctx=ast.Load()),
            args=[expr.func.value] + list(expr.args),
            keywords=list(expr.keywords),
        )
        return iter_extent_of(ast.copy_location(routed, expr), shape_table)
    return UNHANDLED


def np_call_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """The extent of ``np.<attr>(...)`` (``attr`` may be dotted: ``linalg.solve``, ``add.outer``)
    from its sizer in :data:`NP_CALL_EXTENT`, or :data:`UNHANDLED` when none covers the call."""
    sizer = NP_CALL_EXTENT.get(attr)
    if sizer is None and attr.endswith(".outer"):
        sizer = outer_extent
    if sizer is None:
        return UNHANDLED
    return sizer(attr, expr, shape_table)


def eye_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """Square identities state their extent in an argument, as the zeros aliases do."""
    if not expr.args:
        return UNHANDLED
    n = copy.deepcopy(expr.args[0])
    return (
        (n, copy.deepcopy(expr.args[1]))
        if attr == "eye" and len(expr.args) >= 2 and not const_int(expr.args[1]) is None
        else (n, copy.deepcopy(n))
    )


def arange_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    return (copy.deepcopy(expr.args[0]),) if len(expr.args) == 1 else UNHANDLED


def concat_call_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """Concatenation is the one shape-changing form a helper's RETURN commonly takes (``return
    np.hstack((ax, ay, az))``); unsized, the caller's broadcast join over the ARGUMENTS would answer
    ``(N, N)`` for a result that is ``(N, 3)``."""
    return concat_extent(attr, expr, shape_table) if expr.args else UNHANDLED


def first_operand_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """``np.linalg.inv`` / ``cholesky``: shape-preserving factors of their operand. Sized here and
    not only in lowering's harvest, because a shape read that resolves against a factorisation is
    asked at parse time."""
    return iter_extent_of(expr.args[0], shape_table) if expr.args else UNHANDLED


def second_operand_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """``np.linalg.solve(A, b)`` returns x with b's shape; ``np.searchsorted(a, v)`` returns one index
    per element of the VALUES operand, not the sorted array."""
    return iter_extent_of(expr.args[1], shape_table) if len(expr.args) >= 2 else UNHANDLED


def constructor_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """An array CONSTRUCTOR states its extent in its shape argument -- read directly, since an inline
    constructor (never assigned to a Name) is never sized by the harvest. ``*_like`` takes an array,
    not a shape: it mirrors that operand's extent. ``np.full`` is sized here but is not a zeros
    alias: it carries a fill VALUE the alias rewriter would drop."""
    if not expr.args:
        return UNHANDLED
    if attr.endswith("_like"):
        return iter_extent_of(expr.args[0], shape_table)
    shape_arg = expr.args[0]
    if isinstance(shape_arg, (ast.Tuple, ast.List)):
        return tuple(copy.deepcopy(e) for e in shape_arg.elts)
    return (copy.deepcopy(shape_arg),)


def reshape_call_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """``np.reshape(A, newshape)`` -> newshape (never the operand's rank: an enclosing BinOp would
    broadcast the wrong rank), with a ``-1`` placeholder resolved."""
    if len(expr.args) < 2:
        return UNHANDLED
    newshape = expr.args[1]
    if isinstance(newshape, (ast.Tuple, ast.List)):
        elts = list(newshape.elts)
    elif isinstance(newshape, (ast.Name, ast.Constant, ast.BinOp, ast.UnaryOp)):
        elts = [newshape]
    else:
        return UNHANDLED
    return reshape_target(elts, expr.args[0], shape_table)


def transpose_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """The operand's extent with axes reversed, or permuted by an explicit axes tuple. ``dx = x.T -
    x`` with x (N, 1) must broadcast to (N, N)."""
    if not expr.args:
        return UNHANDLED
    base = iter_extent_of(expr.args[0], shape_table)
    if base is None:
        return None
    if len(expr.args) >= 2 and isinstance(expr.args[1], (ast.Tuple, ast.List)):
        perm = [e.value for e in expr.args[1].elts if isinstance(e, ast.Constant) and isinstance(e.value, int)]
        if len(perm) == len(base):
            return tuple(base[p] for p in perm)
    return tuple(reversed(base))


def swapaxes_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    if len(expr.args) < 3:
        return UNHANDLED
    base = iter_extent_of(expr.args[0], shape_table)
    if base is None:
        return None
    i, j = const_axis(expr.args[1], len(base)), const_axis(expr.args[2], len(base))
    if i is None or j is None:
        return None
    out = list(base)
    out[i], out[j] = out[j], out[i]
    return tuple(out)


def moveaxis_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """numpy's own algorithm: drop the source axis, reinsert it at the destination among what is
    left. Unsized, the call would survive to the emitter as an unsupported ``np.moveaxis``."""
    if len(expr.args) < 3:
        return UNHANDLED
    base = iter_extent_of(expr.args[0], shape_table)
    if base is None:
        return None
    src = const_axis(expr.args[1], len(base))
    dst = const_axis(expr.args[2], len(base))
    if src is None or dst is None:
        return None
    rest = [e for n, e in enumerate(base) if n != src]
    return tuple(rest[:dst] + [base[src]] + rest[dst:])


def expand_dims_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    if not expr.args:
        return UNHANDLED
    base = iter_extent_of(expr.args[0], shape_table)
    if base is None:
        return None
    axis = const_axis(kwarg_or_pos(expr.args, expr.keywords, 1, "axis"), len(base) + 1)
    if axis is None:
        return None
    out = list(base)
    out.insert(axis, const_(1))
    return tuple(out)


def squeeze_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    if not expr.args:
        return UNHANDLED
    base = iter_extent_of(expr.args[0], shape_table)
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


def take_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """A LITERAL index takes one element off the axis, so numpy drops that axis entirely --
    distinct from an unresolvable index, which is what None otherwise means."""
    if len(expr.args) < 2:
        return UNHANDLED
    base = iter_extent_of(expr.args[0], shape_table)
    lit_index = const_int(expr.args[1])
    idx_ext = None if lit_index is not None else iter_extent_of(expr.args[1], shape_table)
    if base is None or (lit_index is None and (idx_ext is None or len(idx_ext) != 1)):
        return None
    axis_node = kwarg_or_pos(expr.args, expr.keywords, 2, "axis")
    if lit_index is not None:
        if axis_node is None:
            return None  # flat take on an N-D source: numpy ravels first
        axis = const_axis(axis_node, len(base))
        if axis is None:
            return None
        out = [e for k, e in enumerate(base) if k != axis]
        return tuple(out) or None
    if axis_node is None:
        return idx_ext if len(base) == 1 else None  # flat take on a 1-D source
    axis = const_axis(axis_node, len(base))
    if axis is None:
        return None
    out = list(base)
    out[axis] = idx_ext[0]
    return tuple(out)


def unsized_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """``repeat`` is not statically resolvable here; ``trace`` / ``vdot`` / ``median`` are scalars."""
    return None


def outer_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """``np.<op>.outer(a, b)`` pairs every element of a with every element of b: rank 2."""
    if len(expr.args) != 2:
        return UNHANDLED
    l_out = iter_extent_of(expr.args[0], shape_table)
    r_out = iter_extent_of(expr.args[1], shape_table)
    if l_out is None or r_out is None or len(l_out) != 1 or len(r_out) != 1:
        return None
    return (l_out[0], r_out[0])


def contraction_call_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """``np.einsum`` / ``tensordot`` / ``inner`` -> one axis per output index, sized from the operand
    that introduces it (the first-operand fallthrough would take the first operand's full rank)."""
    return contraction_result_extent(expr, shape_table) if len(expr.args) >= 2 else UNHANDLED


def diagonal_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    if not expr.args:
        return UNHANDLED
    base = iter_extent_of(expr.args[0], shape_table)
    return (base[0],) if base else None


def diag_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """``np.diag(v [, k])``: a 1-D operand builds an ``(n+|k|, n+|k|)`` matrix; a 2-D operand
    extracts the main diagonal (length of its first axis)."""
    if not expr.args:
        return UNHANDLED
    base = iter_extent_of(expr.args[0], shape_table)
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


def scan_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """A prefix scan (``cumsum`` / ``cumprod`` / ``maximum|minimum.accumulate``) is shape-preserving
    along its axis. numpy flattens an axis-less scan over an N-D operand, which the cumulative
    expander rejects -- left unresolved rather than a wrong shape."""
    if not expr.args:
        return UNHANDLED
    base = iter_extent_of(expr.args[0], shape_table)
    if base is None:
        return None
    if kwarg_or_pos(expr.args, expr.keywords, 1, "axis") is not None or len(base) == 1:
        return base
    return None


def pad_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """``np.pad(src, pad_width, ...)`` -> each source axis grown by its ``before + after`` width."""
    if not expr.args:
        return UNHANDLED
    base = iter_extent_of(expr.args[0], shape_table)
    if base is None:
        return None
    return pad_output_extent(base, kwarg_or_pos(expr.args, expr.keywords, 1, "pad_width"))


def stack_extent(attr: str, expr: ast.Call, shape_table: ShapeTable) -> CallSize:
    """``np.stack((a, b, ...), axis=k)`` -> the operands' common shape with a NEW size-N axis
    inserted at k (N = number of operands)."""
    if not expr.args:
        return UNHANDLED
    try:
        names, shapes, unused = concat_operands_axis(expr.args, expr.keywords, shape_table)
        axis = stack_axis(expr.args, expr.keywords, len(shapes[0]))
    except NotImplementedError:
        return None
    out = [const_or_name(t) for t in shapes[0]]
    out.insert(axis, const_(len(names)))
    return tuple(out)


def subscript_extent(expr: ast.Subscript, shape_table: ShapeTable) -> Extent | None:
    """One result axis per Slice (its element count), per newaxis (1) and per Ellipsis-covered source
    axis; scalar axes consume a source axis and contribute nothing. Several integer-ARRAY indices
    broadcast into ONE group of result axes at the first one's position (numpy advanced indexing:
    ``u2[q, r, s]`` with q/r/s all (J,) is (J,), not (J, J, J)). Omitted trailing axes keep their full
    extent."""
    shape = subscript_base_shape(expr, shape_table)
    axes = slice_axes(expr)
    ext: list[ast.expr] = []
    src_axis = 0  # advances on Slice / scalar axes, NOT on newaxis (a pure result-axis insertion)
    n_src_consumers = sum(1 for ax in axes if not is_special_axis(ax))
    idx_array_extents: list[Extent] = []
    idx_group_pos: int | None = None
    for ax in axes:
        if isinstance(ax, ast.Constant) and ax.value is None:
            ext.append(const_(1))
            continue
        if isinstance(ax, ast.Constant) and ax.value is Ellipsis:
            filled = ellipsis_extent(shape, src_axis, n_src_consumers)
            if filled is None:
                return None
            ext.extend(filled)
            src_axis += len(filled)
            continue
        if isinstance(ax, ast.Slice):
            count = slice_count(ax, const_or_name(shape[src_axis]) if shape and src_axis < len(shape) else None)
            if count is None:
                return None
            ext.append(count)
        else:
            index_ext = index_array_extent(ax, shape_table)
            if index_ext is not None:
                if idx_group_pos is None:
                    idx_group_pos = len(ext)
                idx_array_extents.append(index_ext)
        src_axis += 1
    if idx_array_extents:
        # Broadcast the index extents together; the LONGEST is not the numpy rule and is wrong
        # whenever the ranks tie.
        group = idx_array_extents[0]
        for other in idx_array_extents[1:]:
            group = broadcast_extents(group, other)
        ext[idx_group_pos:idx_group_pos] = list(group)
    if shape and src_axis < len(shape):
        for i in range(src_axis, len(shape)):
            ext.append(const_or_name(shape[i]))
    return tuple(ext) if ext else None


def ellipsis_extent(shape: tuple[str, ...] | None, src_axis: int, n_consumers: int) -> list[ast.expr] | None:
    """The full extents of the source axes an Ellipsis at ``src_axis`` covers (every axis the
    ``n_consumers`` explicit entries do not consume); None when the source rank is unknown or
    exceeded."""
    if not shape:
        return None
    out: list[ast.expr] = []
    for unused in range(max(len(shape) - n_consumers, 0)):
        if src_axis >= len(shape):
            return None
        out.append(const_or_name(shape[src_axis]))
        src_axis += 1
    return out


def subscript_base_shape(expr: ast.Subscript, shape_table: ShapeTable) -> tuple[str, ...] | None:
    """The shape the subscript's axes index: a Name's table entry; a chained scalar-indexed base
    ``A[i, j][outer]``'s residual shape; else any sized base, such as a CALL result indexed
    directly (``np.expand_dims(np.take(x, 0, axis=k), axis=k)`` after the frontend rewrites
    expand_dims to a newaxis index)."""
    name = name_id(expr.value)
    if name:
        return shape_table.get(name)
    shape = chained_base_shape(expr.value, shape_table)
    if shape is None:
        base_ext = iter_extent_of(expr.value, shape_table)
        shape = tuple(ast.unparse(e) for e in base_ext) if base_ext is not None else None
    return shape


def index_array_extent(ax: ast.expr, shape_table: ShapeTable) -> Extent | None:
    """The result extent a gather index contributes, or None for a scalar index: a known-shape Name
    (a scalar Name -- loop var/symbol -- has no shape) or an advanced-index EXPRESSION such as
    ``edge_idx[:, :, 0] - 1``."""
    if isinstance(ax, ast.Name) and shape_table.get(ax.id):
        return tuple(const_or_name(s) for s in shape_table[ax.id])
    if advanced_index_rank(ax, shape_table):
        ie = iter_extent_of(ax, shape_table)
        return None if ie is None else tuple(ie)
    return None


def slice_count(ax: ast.Slice, axis_len: ast.expr | None) -> ast.expr | None:
    """Element count of slice ``ax`` over an axis of length ``axis_len``: ``hi - lo`` (simplified), or
    ``ceil((hi - lo) / |k|)`` for a stride k (``== len(range(lo, hi, k))``). None when a bound does
    not resolve, and for a BOUNDED reverse slice (``a[lo::-1]`` / ``a[:hi:-1]``): numpy flips the
    bound defaults under a negative step, so ``hi - lo`` is not its element count."""
    lo = resolve_negative(ax.lower, axis_len) if ax.lower is not None else const_(0)
    hi = resolve_negative(ax.upper, axis_len) if ax.upper is not None else axis_len
    if hi is None or lo is None:
        return None
    if isinstance(hi, ast.Constant) and isinstance(lo, ast.Constant):
        raw: ast.expr = const_(hi.value - lo.value)
    elif isinstance(lo, ast.Constant) and lo.value == 0:
        raw = hi
    else:
        # ``(lo + K) - lo`` -> K (slice ``[i:i+K]``), ``(K + lo) - lo`` -> K, ``lo - lo`` -> 0
        simplified = simplify_sub(hi, lo)
        raw = simplified if simplified is not None else ast.BinOp(left=hi, op=ast.Sub(), right=lo)
    step = slice_step_any(ax)
    if step_is_negative(step) and (ax.lower is not None or ax.upper is not None):
        return None
    if isinstance(step, ast.expr):
        # Symbolic stride: ceil(raw / step), no abs() -- a bounded slice's step is positive or the
        # numpy source is empty.
        exact = span_multiple_of(raw, step)
        if exact is not None:
            return exact
        return ast.BinOp(
            left=ast.BinOp(left=ast.BinOp(left=raw, op=ast.Add(), right=step), op=ast.Sub(), right=const_(1)),
            op=ast.FloorDiv(),
            right=step,
        )
    if step is not None and step != 1:
        # A full-axis reverse spans as many elements as its positive magnitude.
        astep = abs(step)
        if isinstance(raw, ast.Constant):
            return const_((raw.value + astep - 1) // astep)
        return ast.BinOp(
            left=ast.BinOp(left=raw, op=ast.Add(), right=const_(astep - 1)),
            op=ast.FloorDiv(),
            right=const_(astep),
        )
    return raw


#: Extent sizer per expression node type.
EXPR_EXTENT: dict[type[ast.expr], Callable[..., Extent | None]] = {
    ast.Name: name_extent,
    ast.BinOp: binop_extent,
    ast.UnaryOp: unaryop_extent,
    ast.Compare: compare_extent,
    ast.BoolOp: boolop_extent,
    ast.IfExp: ifexp_extent,
    ast.Call: call_extent,
    ast.Subscript: subscript_extent,
    ast.Attribute: attribute_extent,
}

#: ``np.<attr>`` -> its sizer. ``np.<ufunc>.outer`` is matched by suffix in :func:`np_call_extent`.
NP_CALL_EXTENT: dict[str, Callable[[str, ast.Call, ShapeTable], CallSize]] = {
    "eye": eye_extent,
    "identity": eye_extent,
    "arange": arange_extent,
    "hstack": concat_call_extent,
    "vstack": concat_call_extent,
    "concatenate": concat_call_extent,
    "linalg.inv": first_operand_extent,
    "linalg.cholesky": first_operand_extent,
    "linalg.solve": second_operand_extent,
    **dict.fromkeys((*NP_ZEROS_ALIASES, "full", "full_like"), constructor_extent),
    "reshape": reshape_call_extent,
    "transpose": transpose_extent,
    "swapaxes": swapaxes_extent,
    "moveaxis": moveaxis_extent,
    "expand_dims": expand_dims_extent,
    "squeeze": squeeze_extent,
    "take": take_extent,
    "repeat": unsized_extent,
    "einsum": contraction_call_extent,
    "tensordot": contraction_call_extent,
    "inner": contraction_call_extent,
    "trace": unsized_extent,
    "vdot": unsized_extent,
    "median": unsized_extent,
    "diagonal": diagonal_extent,
    "diag": diag_extent,
    "searchsorted": second_operand_extent,
    "cumsum": scan_extent,
    "cumprod": scan_extent,
    "maximum.accumulate": scan_extent,
    "minimum.accumulate": scan_extent,
    "pad": pad_extent,
    "stack": stack_extent,
}


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
        ext = iter_extent_of(child, shape_table)
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
        ext = iter_extent_of(op, shape_table)
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
