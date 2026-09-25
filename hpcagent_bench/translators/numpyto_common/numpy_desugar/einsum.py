"""``np.einsum`` lowered to an explicit loop nest."""

import ast

from hpcagent_bench.translators.numpyto_common.lib_nodes import parse_einsum_subscripts
from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import np_attr
from hpcagent_bench.translators.numpyto_common.numpy_desugar.hoist import HoistForm, ValueHoist
from hpcagent_bench.translators.numpyto_common.numpy_desugar.kinds import KIND_RANK, dtype_kind


def einsum_inline_stmts(subs: str, operands: list[str], ctr: int, dtype_of: str):
    """Source statements computing ``np.einsum(subs, *operands)`` into a fresh
    temp via an explicit contraction loop nest (output indices outer, contracted
    indices inner-accumulate). Returns ``(stmts, temp_name)`` or ``(None, None)``
    when the form is unsupported (ellipsis / scalar output). numba and pythran
    compile this; neither supports ``np.einsum`` on these shapes. The temp takes
    ``dtype_of``'s dtype: the operand holding numpy's result type of them all."""
    try:
        in_subs, out_sub = parse_einsum_subscripts(subs)
    except Exception:  # noqa: BLE001 -- ellipsis / malformed -> caller bails
        return None, None
    if not out_sub or len(in_subs) != len(operands):
        return None, None  # scalar full-contraction not handled here
    # index char -> (operand index, axis): first operand carrying it.
    char_src: dict[str, tuple] = {}
    for oi, sub in enumerate(in_subs):
        for ax, ch in enumerate(sub):
            char_src.setdefault(ch, (oi, ax))
    if any(ch not in char_src for ch in out_sub):
        return None, None

    def extent(ch: str) -> str:
        oi, ax = char_src[ch]
        return f"{operands[oi]}.shape[{ax}]"

    p = f"__es{ctr}"
    out_chars = list(out_sub)
    contracted = [c for c in char_src if c not in out_sub]
    outshape = ", ".join(extent(c) for c in out_chars)
    src = [f"{p} = np.empty(({outshape},), {dtype_of}.dtype)"]
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


def hoist_einsum(node: ast.AST, hoist: ValueHoist) -> ast.expr | None:
    """``np.einsum("<subs>", *names)`` -> the temp its contraction loop nest fills; handles einsum nested in
    arithmetic (seissol's ``Q[:] = Q + np.einsum(...)``)."""
    if not isinstance(node, ast.Call) or np_attr(node) != "einsum" or not node.args:
        return None
    subs, operands = node.args[0], node.args[1:]
    if not (isinstance(subs, ast.Constant) and isinstance(subs.value, str)):
        return None
    names = [o.id for o in operands if isinstance(o, ast.Name)]
    if not operands or len(names) != len(operands):
        return None  # only bare-array operands -> else leave verbatim
    # numpy's result type is the widest operand kind. An unknown kind may be the widest one, so it
    # yields to the first operand unless a known operand is already complex, the widest there is.
    kinds = [dtype_kind(operand, hoist.tables.dtypes) for operand in operands]
    widest = max(range(len(kinds)), key=lambda at: KIND_RANK.get(kinds[at] or "", -1))
    dtype_of = names[widest] if None not in kinds or kinds[widest] == "complex" else names[0]
    stmts, temp = einsum_inline_stmts(subs.value, names, hoist.ctr, dtype_of)
    if stmts is None:
        return None
    hoist.ctr += 1
    hoist.pre.extend(stmts)
    return ast.Name(id=temp, ctx=ast.Load())


EINSUM_HOIST = HoistForm(frozenset({"einsum"}), (), hoist_einsum)
