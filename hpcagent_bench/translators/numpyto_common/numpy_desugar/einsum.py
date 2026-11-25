"""``np.einsum`` lowered to an explicit loop nest."""

import ast

from hpcagent_bench.translators.numpyto_common.ast_build import const_int, name_
from hpcagent_bench.translators.numpyto_common.emit_helpers.numpy_names import numpy_call_attr
from hpcagent_bench.translators.numpyto_common.lib_nodes import parse_einsum_subscripts
from hpcagent_bench.translators.numpyto_common.numpy_desugar.hoist import HoistForm, ValueHoist
from hpcagent_bench.translators.numpyto_common.numpy_desugar.kinds import KIND_RANK, dtype_kind
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import expr_rank

__all__ = [
    "EINSUM_HOIST",
    "TENSORDOT_HOIST",
    "bound_operands",
    "einsum_inline_stmts",
    "hoist_einsum",
    "hoist_tensordot",
    "product_dtype",
    "tensordot_axes",
]


def einsum_inline_stmts(
    subs: str, operands: list[str], ctr: int, dtype: str, prefix: str = "__es"
) -> tuple[list[ast.stmt], str] | tuple[None, None]:
    """``(stmts, temp)`` computing ``np.einsum(subs, *operands)`` as a contraction loop nest.

    ``(None, None)`` for ellipsis or scalar output. The temp takes the dtype expression ``dtype``."""
    try:
        in_subs, out_sub = parse_einsum_subscripts(subs)
    except Exception:  # noqa: BLE001 -- ellipsis / malformed
        return None, None
    if not out_sub or len(in_subs) != len(operands):
        return None, None
    # index char -> (operand index, axis) of the first operand carrying it.
    char_src: dict[str, tuple] = {}
    for oi, sub in enumerate(in_subs):
        for ax, ch in enumerate(sub):
            char_src.setdefault(ch, (oi, ax))
    if any(ch not in char_src for ch in out_sub):
        return None, None

    def extent(ch: str) -> str:
        oi, ax = char_src[ch]
        return f"{operands[oi]}.shape[{ax}]"

    p = f"{prefix}{ctr}"
    out_chars = list(out_sub)
    contracted = [c for c in char_src if c not in out_sub]
    outshape = ", ".join(extent(c) for c in out_chars)
    src = [f"{p} = np.empty(({outshape},), {dtype})"]
    indent = ""
    for c in out_chars:
        src.append(f"{indent}for {p}_{c} in range({extent(c)}):")
        indent += "    "
    out_idx = ", ".join(f"{p}_{c}" for c in out_chars)
    src.append(f"{indent}{p}[{out_idx}] = 0")
    cind = indent
    for c in contracted:
        src.append(f"{cind}for {p}_{c} in range({extent(c)}):")
        cind += "    "
    terms = [f"{operands[oi]}[{', '.join(f'{p}_{ch}' for ch in sub)}]" for oi, sub in enumerate(in_subs)]
    src.append(f"{cind}{p}[{out_idx}] += {' * '.join(terms)}")
    return ast.parse("\n".join(src)).body, p


def product_dtype(names: list[str], kinds: list[str | None]) -> str:
    """The dtype expression of a product of the arrays ``names``: the widest one's dtype when every kind is known,
    else numpy's promotion of a one-element product (a float ``qr`` times a complex ``vc_box`` is complex)."""
    if None not in kinds:
        return f"{names[max(range(len(kinds)), key=lambda at: KIND_RANK.get(kinds[at] or '', -1))]}.dtype"
    return f"({' * '.join(f'np.zeros(1, {name}.dtype)' for name in names)}).dtype"


def bound_operands(operands: list[ast.expr], hoist: ValueHoist, prefix: str) -> list[str]:
    """Each operand's name, binding one that is not a Name (``np.conj(becphi_r)``) to ``<prefix><index>`` first."""
    names = []
    for at, operand in enumerate(operands):
        if isinstance(operand, ast.Name):
            names.append(operand.id)
            continue
        bound = f"{prefix}{at}"
        hoist.queue([f"{bound} = {ast.unparse(operand)}"])
        names.append(bound)
    return names


def hoist_einsum(node: ast.expr, hoist: ValueHoist) -> ast.expr | None:
    """``np.einsum("<subs>", *operands)`` -> the temp its contraction loop nest fills, also inside larger expressions."""
    if not isinstance(node, ast.Call) or numpy_call_attr(node) != "einsum" or not node.args:
        return None
    subs, operands = node.args[0], node.args[1:]
    if not (isinstance(subs, ast.Constant) and isinstance(subs.value, str)) or not operands:
        return None
    if any(isinstance(o, ast.Starred) for o in operands):
        return None
    try:
        in_subs, out_sub = parse_einsum_subscripts(subs.value)
    except Exception:  # noqa: BLE001 -- ellipsis / malformed: einsum_inline_stmts refuses it too
        return None
    # einsum_inline_stmts' own refusals, checked before any operand is bound: a refused einsum keeps its operands.
    if not out_sub or len(in_subs) != len(operands) or any(c not in "".join(in_subs) for c in out_sub):
        return None
    names = bound_operands(operands, hoist, f"__es{hoist.ctr}_o")
    dtype = product_dtype(names, [dtype_kind(operand, hoist.tables.dtypes) for operand in operands])
    stmts, temp = einsum_inline_stmts(subs.value, names, hoist.ctr, dtype)
    if stmts is None or temp is None:
        return None
    hoist.ctr += 1
    hoist.pre.extend(stmts)
    hoist.tables.ranks[temp] = len(out_sub)  # a later peel reads it (lulesh's ``coefficient[:, None] * __es4``)
    return name_(temp)


EINSUM_HOIST = HoistForm(frozenset({"einsum"}), (), hoist_einsum)


def tensordot_axes(value: ast.Call, rank_a: int, rank_b: int) -> tuple[list[int], list[int]] | None:
    """The axes of ``a`` and of ``b`` that ``np.tensordot(a, b, axes)`` contracts, non-negative; None when not literal."""
    kw = {k.arg: k.value for k in value.keywords}
    axes = kw["axes"] if "axes" in kw else (value.args[2] if len(value.args) > 2 else ast.Constant(value=2))
    count = const_int(axes)
    if count is not None:
        return list(range(rank_a - count, rank_a)), list(range(count))
    if not (isinstance(axes, (ast.Tuple, ast.List)) and len(axes.elts) == 2):
        return None
    pair = []
    for side, rank in zip(axes.elts, (rank_a, rank_b), strict=True):
        listed = side.elts if isinstance(side, (ast.Tuple, ast.List)) else [side]
        found = [const_int(e) for e in listed]
        if any(v is None or not -rank <= v < rank for v in found):
            return None
        pair.append([v % rank for v in found if v is not None])
    return (pair[0], pair[1]) if len(pair[0]) == len(pair[1]) else None


def hoist_tensordot(node: ast.expr, hoist: ValueHoist) -> ast.expr | None:
    """``np.tensordot(a, b, axes)`` with literal axes and ranked operands -> the einsum loop nest of the same
    contraction (numba only: numba has no ``np.tensordot``; chebyshev_filter_subspace applies its 1-D Laplacian along
    each axis this way). ``a``'s free axes come first, then ``b``'s, as numpy orders them; an operand that is no
    name is bound to a temp first."""
    if not isinstance(node, ast.Call) or numpy_call_attr(node) != "tensordot" or len(node.args) < 2:
        return None
    operands = node.args[:2]
    rank_a, rank_b = (expr_rank(o, hoist.tables.ranks) for o in operands)
    if rank_a is None or rank_b is None:
        return None
    contracted = tensordot_axes(node, rank_a, rank_b)
    if contracted is None or rank_a + rank_b > 26:
        return None
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    sub_a = [next(letters) for unused in range(rank_a)]
    sub_b = [next(letters) for unused in range(rank_b)]
    for axis_a, axis_b in zip(*contracted, strict=True):
        sub_b[axis_b] = sub_a[axis_a]
    out = [c for i, c in enumerate(sub_a) if i not in contracted[0]] + [
        c for i, c in enumerate(sub_b) if i not in contracted[1]
    ]
    names = bound_operands(operands, hoist, f"__td{hoist.ctr}_")
    dtype = product_dtype(names, [dtype_kind(o, hoist.tables.dtypes) for o in operands])
    subs = f"{''.join(sub_a)},{''.join(sub_b)}->{''.join(out)}"
    stmts, temp = einsum_inline_stmts(subs, names, hoist.ctr, dtype, "__tdo")
    if stmts is None or temp is None:
        return None
    hoist.ctr += 1
    hoist.pre.extend(stmts)
    hoist.tables.ranks[temp] = len(out)
    return name_(temp)


TENSORDOT_HOIST = HoistForm(frozenset({"tensordot"}), (), hoist_tensordot)
