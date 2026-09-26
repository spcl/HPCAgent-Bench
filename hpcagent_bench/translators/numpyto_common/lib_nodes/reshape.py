"""``np.reshape`` as a rank-aware copy nest."""

import ast

from hpcagent_bench.translators.numpyto_common.lib_nodes.dims import dims_agree
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import (
    const_or_name,
    name_,
    shape_total_product,
    store_,
)

__all__ = [
    "axis_stride",
    "decoded_source_axes",
    "expand_reshape",
    "flat_copy",
    "flat_index",
    "product_str",
    "reshape_axis_groups",
    "reshape_grouped_copy",
    "reshape_order",
    "token_product",
]


def product_str(tokens: list[str]) -> str:
    """``["a", "b"]`` -> ``"(a) * (b)"``; the empty group is the unit extent."""
    return " * ".join(f"({t})" for t in tokens) if tokens else "1"


def reshape_axis_groups(
    src_shape: tuple[str, ...],
    tgt_shape: tuple[str, ...],
    aliases: dict[str, str] | None,
    shape_table: dict[str, tuple[str, ...]] | None,
) -> list[tuple[list[int], list[int]]] | None:
    """Pair the two shapes of a C-order reshape into contiguous axis groups, or None.

    Axes that agree pairwise from the front and from the back are paired 1:1; whatever is left in
    the middle is ONE group. Sound with no algebra over the extents -- a reshape keeps element
    order and the two shapes multiply out to the same total by construction, so once the matched
    axes are equal the unmatched middles span the same extent. Proving that directly means proving
    ``num_groups * (c // num_groups) == c``, which sympy will not do, and every grouped-channel
    reshape in the ML track has exactly that shape.

    A middle that is N:M with both sides above one axis is a genuine re-layout and returns None.
    """
    ns, nt = len(src_shape), len(tgt_shape)
    head = 0
    while head < ns and head < nt and dims_agree(src_shape[head], tgt_shape[head], aliases, shape_table):
        head += 1
    tail = 0
    while (
        tail < ns - head
        and tail < nt - head
        and dims_agree(src_shape[ns - 1 - tail], tgt_shape[nt - 1 - tail], aliases, shape_table)
    ):
        tail += 1
    groups: list[tuple[list[int], list[int]]] = [([i], [i]) for i in range(head)]
    mid_src, mid_tgt = list(range(head, ns - tail)), list(range(head, nt - tail))
    if mid_src or mid_tgt:
        if len(mid_src) > 1 and len(mid_tgt) > 1:
            return None
        groups.append((mid_src, mid_tgt))
    groups.extend(([ns - 1 - k], [nt - 1 - k]) for k in reversed(range(tail)))
    return groups


def reshape_grouped_copy(
    tgt_name: str,
    src_name: str,
    src_shape: tuple[str, ...],
    tgt_shape: tuple[str, ...],
    groups: list[tuple[list[int], list[int]]],
) -> list[ast.stmt] | None:
    """The copy nest for a reshape whose axes pair up (:func:`reshape_axis_groups`), or None.

    Iterates the finer side of every group, so each index on the coarser side is a linear
    combination of those iterators and every subscript stays affine. A group that splits BOTH
    sides is not expressible this way and declines.
    """
    if not src_shape or not tgt_shape:
        return None
    src_index: list[str] = [""] * len(src_shape)
    tgt_index: list[str] = [""] * len(tgt_shape)
    loops: list[tuple[str, str]] = []
    for src_idxs, tgt_idxs in groups:
        if not src_idxs or not tgt_idxs:
            # One side spans nothing, so the other side is a run of size-1 axes indexed at 0.
            solo, index = (src_idxs, src_index) if tgt_idxs == [] else (tgt_idxs, tgt_index)
            shape = src_shape if tgt_idxs == [] else tgt_shape
            if any(str(shape[ax]) != "1" for ax in solo):
                return None
            for ax in solo:
                index[ax] = "0"
            continue
        if len(src_idxs) == len(tgt_idxs):
            for s_ax, t_ax in zip(src_idxs, tgt_idxs):
                it = f"__rs{len(loops)}"
                loops.append((it, src_shape[s_ax]))
                src_index[s_ax] = tgt_index[t_ax] = it
            continue
        fine_is_src = len(src_idxs) > len(tgt_idxs)
        fine_axes, fine_shape = (src_idxs, src_shape) if fine_is_src else (tgt_idxs, tgt_shape)
        coarse_axes = tgt_idxs if fine_is_src else src_idxs
        if len(coarse_axes) != 1:
            return None
        terms: list[str] = []
        its: list[str] = []
        for ax in fine_axes:
            it = f"__rs{len(loops)}"
            loops.append((it, fine_shape[ax]))
            its.append(it)
        for pos, (ax, it) in enumerate(zip(fine_axes, its)):
            stride = product_str([fine_shape[a] for a in fine_axes[pos + 1 :]])
            terms.append(it if stride == "1" else f"({it}) * ({stride})")
            (src_index if fine_is_src else tgt_index)[ax] = it
        (tgt_index if fine_is_src else src_index)[coarse_axes[0]] = " + ".join(terms)
    if not all(src_index) or not all(tgt_index):
        return None

    def subscript(name: str, parts: list[str], ctx: ast.expr_context) -> ast.expr:
        elts = [ast.parse(t, mode="eval").body for t in parts]
        sl = elts[0] if len(elts) == 1 else ast.Tuple(elts=elts, ctx=ast.Load())
        return ast.Subscript(value=name_(name), slice=sl, ctx=ctx)

    inner: ast.stmt = ast.Assign(
        targets=[subscript(tgt_name, tgt_index, ast.Store())], value=subscript(src_name, src_index, ast.Load())
    )
    for it, bound in reversed(loops):
        inner = ast.For(
            target=store_(it),
            iter=ast.Call(func=name_("range"), args=[const_or_name(bound)], keywords=[]),
            body=[inner],
            orelse=[],
        )
    return [inner]


def expand_reshape(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    dim_aliases: dict[str, str] | None = None,
) -> list[ast.stmt]:
    """``out = np.reshape(A, (m, n, ...))`` -> rank-aware loop-nest copy.

    Two lowerings. When the two shapes pair up into contiguous axis groups
    (:func:`reshape_axis_groups`) the nest runs over the finer side of each group and every
    subscript is affine. Otherwise it falls back to a nest over the **target** shape whose source
    multi-index comes from div/mod on a running flat index -- correct for both C (flat-indexes
    anyway) and Fortran (type-checks rank), but non-affine over a symbolic stride.

    ``order="F"`` (numpy column-major ravel/fill) is honoured: target flat
    index and source multi-index are both computed column-major, so a
    Fortran-order reshape lowers correctly (QE vexx_k uses ``order="F"``
    throughout its FFT band-pair convolution).
    """
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.reshape needs Name first arg")
    a = args[0]
    a_shape = shape_table.get(a.id)
    if not a_shape:
        raise NotImplementedError("np.reshape: source shape unknown")
    tgt_shape = shape_table.get(target.id)
    if not tgt_shape:
        # Target shape unknown: flat copy (Fortran rejects a rank mismatch here).
        return flat_copy(target.id, a.id, a_shape)

    # Build per-axis loop iters for the target shape.
    tgt_rank = len(tgt_shape)
    src_rank = len(a_shape)
    tgt_iters = [f"__r{i}" for i in range(tgt_rank)]
    fortran = reshape_order(kwargs) == "F"

    # Affine fast path: when the two shapes pair up into contiguous axis groups, iterate the finer
    # side and read the coarser index off a linear combination of its iterators. The flat-index
    # form below is correct either way, but its ``/`` and ``%`` are non-affine over a symbolic
    # stride, which drops the nest out of Pluto's model. ``order="F"`` keeps the flat form.
    if not fortran:
        groups = reshape_axis_groups(tuple(a_shape), tuple(tgt_shape), dim_aliases, shape_table)
        if groups is not None:
            grouped = reshape_grouped_copy(target.id, a.id, tuple(a_shape), tuple(tgt_shape), groups)
            if grouped is not None:
                return grouped

    flat_expr = flat_index(tgt_iters, tgt_shape, fortran)
    src_axes = decoded_source_axes(a_shape, flat_expr, fortran)

    # ``out[t0, t1, ...] = A[<computed-axes>]``.
    if tgt_rank == 1:
        lhs_slice = name_(tgt_iters[0])
    else:
        lhs_slice = ast.Tuple(elts=[name_(it) for it in tgt_iters], ctx=ast.Load())
    if src_rank == 1:
        rhs_slice = src_axes[0]
    else:
        rhs_slice = ast.Tuple(elts=src_axes, ctx=ast.Load())
    inner = ast.Assign(
        targets=[ast.Subscript(value=name_(target.id), slice=lhs_slice, ctx=ast.Store())],
        value=ast.Subscript(value=name_(a.id), slice=rhs_slice, ctx=ast.Load()),
    )

    # Wrap in target-shape loop nest (outermost first).
    current: ast.stmt = inner
    for it, bound in zip(reversed(tgt_iters), reversed(list(tgt_shape))):
        current = ast.For(
            target=store_(it),
            iter=ast.Call(func=name_("range"), args=[const_or_name(bound)], keywords=[]),
            body=[current],
            orelse=[],
        )
    return [current]


def flat_copy(target_id: str, source_id: str, a_shape: tuple[str, ...]) -> list[ast.stmt]:
    """``target[__r] = source[__r]`` over the source's element count."""
    total = shape_total_product(a_shape)
    body = [
        ast.Assign(
            targets=[ast.Subscript(value=name_(target_id), slice=name_("__r"), ctx=ast.Store())],
            value=ast.Subscript(value=name_(source_id), slice=name_("__r"), ctx=ast.Load()),
        )
    ]
    return [
        ast.For(
            target=store_("__r"),
            iter=ast.Call(func=name_("range"), args=[total], keywords=[]),
            body=body,
            orelse=[],
        )
    ]


def reshape_order(kwargs: list[ast.keyword] | None) -> str:
    """The reshape's memory order: numpy's default C (row-major), or ``order="F"``, which ravels the
    source and fills the target column-major. Flat position k of one maps to flat position k of the
    other in the SAME order, so both the target flat index and the source multi-index use it."""
    order = "C"
    for kw in kwargs or []:
        if vars(kw).get("arg") == "order" and isinstance(kw.value, ast.Constant):
            order = str(kw.value.value).upper()
    return order


def token_product(*toks: str) -> str:
    """Parenthesised product of the non-unit shape tokens (``"1"`` for none)."""
    toks = [t for t in toks if t and t != "1"]
    if not toks:
        return "1"
    if len(toks) == 1:
        return toks[0]
    return "(" + " * ".join(f"({t})" for t in toks) + ")"


def axis_stride(shape: tuple[str, ...], i: int, fortran: bool) -> str:
    """Stride of axis ``i`` = product of the FASTER-varying axes: the trailing axes in C order, the
    leading axes in F order."""
    faster = list(shape[:i]) if fortran else list(shape[i + 1 :])
    return token_product(*faster) if faster else "1"


def flat_index(tgt_iters: list[str], tgt_shape: tuple[str, ...], fortran: bool) -> str:
    """Flat index of the current target iteration in the reshape's order.

    The STRIDE is parenthesised too: it is re-parsed from text, and an unbracketed compound factor
    re-associates (``(i) * (w - k) / 1 + 1`` as ``(i*(w-k))/1 + 1``)."""
    flat_parts: list[str] = []
    for i, it in enumerate(tgt_iters):
        stride = axis_stride(tgt_shape, i, fortran)
        flat_parts.append(it if stride == "1" else f"({it}) * ({stride})")
    return " + ".join(flat_parts) if flat_parts else "0"


def decoded_source_axes(a_shape: tuple[str, ...], flat_expr: str, fortran: bool) -> list[ast.expr]:
    """The source multi-index decoded from the flat index via div/mod on the source strides (same
    order). The MOST-major axis (``i == 0`` in C, the last in F) needs no modulo. A size-1 source
    axis indexes to a constant 0, avoiding a degenerate ``flat % 1`` whose bare ``1`` literal clashes
    with the int64 flat index under Fortran ``-std=f2018``."""
    src_rank = len(a_shape)
    src_axes: list[ast.expr] = []
    for i in range(src_rank):
        if str(a_shape[i]) == "1":
            src_axes.append(ast.Constant(value=0))
            continue
        stride = axis_stride(a_shape, i, fortran)
        ax_expr = flat_expr if stride == "1" else f"(({flat_expr}) / ({stride}))"
        is_major = (i == src_rank - 1) if fortran else (i == 0)
        if not is_major:
            ax_expr = f"(({ax_expr}) % ({a_shape[i]}))"
        src_axes.append(ast.parse(ax_expr, mode="eval").body)
    return src_axes
