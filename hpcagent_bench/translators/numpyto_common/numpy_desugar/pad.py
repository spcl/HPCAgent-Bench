"""``np.pad`` lowered to an allocation plus slice copies."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import const_int, np_attr
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import expr_rank


def const_pair_widths(pad_width: ast.AST, rank: int):
    """Per-axis ``(lo, hi)`` exprs from a scalar ``R`` or a ``((lo, hi), ...)`` ``pad_width``, else None."""
    if isinstance(pad_width, (ast.Name, ast.Constant)):
        return [(pad_width, copy.deepcopy(pad_width)) for unused in range(rank)]
    if isinstance(pad_width, (ast.Tuple, ast.List)) and len(pad_width.elts) == rank:
        out = []
        for e in pad_width.elts:
            if isinstance(e, (ast.Tuple, ast.List)) and len(e.elts) == 2:
                out.append((e.elts[0], e.elts[1]))
            else:
                return None
        return out
    return None


def pad_inline_stmts(target: str, arr: ast.AST, widths, rank: int, ctr: int) -> list[ast.stmt]:
    """``<target> = np.pad(arr, ..., mode="edge")`` as a clamped-index copy loop nest.

    Inlined rather than a helper call: numba needs an ``@njit`` callee, pythran a plain def."""
    p = f"__pd{ctr}"
    xv = f"{p}_x"
    src = [f"{xv} = {ast.unparse(arr)}"]
    dims = [f"{xv}.shape[{i}] + {ast.unparse(lo)} + {ast.unparse(hi)}" for i, (lo, hi) in enumerate(widths)]
    src.append(f"{target} = np.empty(({', '.join(dims)},), {xv}.dtype)")
    indent = ""
    for i, (lo, hi) in enumerate(widths):
        src.append(f"{indent}for {p}_i{i} in range({dims[i]}):")
        indent += "    "
        src.append(f"{indent}{p}_s{i} = min(max({p}_i{i} - {ast.unparse(lo)}, 0), {xv}.shape[{i}] - 1)")
    idx_o = ", ".join(f"{p}_i{i}" for i in range(rank))
    idx_s = ", ".join(f"{p}_s{i}" for i in range(rank))
    src.append(f"{indent}{target}[{idx_o}] = {xv}[{idx_s}]")
    return ast.parse("\n".join(src)).body


def widths_all_literal(widths) -> bool:
    """True iff every (lo, hi) pair is a compile-time int (dace's own ``np.pad`` handles those)."""
    return all(const_int(lo) is not None and const_int(hi) is not None for lo, hi in widths)


def pad_constant_inline_stmts(target: str, arr: ast.AST, widths, fill: ast.AST, ctr: int) -> list[ast.stmt]:
    """``<target> = np.pad(arr, ..., mode="constant")`` as ``np.full`` plus an interior slice copy.

    For symbolic widths: dace's own ``np.pad`` casts every width through ``int()`` and fails, while
    ``np.full`` and slice assignment take symbolic extents."""
    p = f"__pdc{ctr}"
    xv = f"{p}_x"
    src = [f"{xv} = {ast.unparse(arr)}"]
    dims = [f"{xv}.shape[{i}] + {ast.unparse(lo)} + {ast.unparse(hi)}" for i, (lo, hi) in enumerate(widths)]
    src.append(f"{target} = np.full(({', '.join(dims)},), {ast.unparse(fill)}, {xv}.dtype)")
    interior = ", ".join(
        f"{ast.unparse(lo)}:{ast.unparse(lo)} + {xv}.shape[{i}]" for i, (lo, unused) in enumerate(widths)
    )
    src.append(f"{target}[{interior}] = {xv}")
    return ast.parse("\n".join(src)).body


class PadInline(ast.NodeTransformer):
    """``name = np.pad(x, pad_width=..., mode="edge")`` -> an inline edge-pad loop nest.

    Only the bare-assign form is matched. ``mode="constant"`` is lowered only with
    ``lower_symbolic_constant`` (dace) and a non-literal width; see :func:`pad_constant_inline_stmts`.
    """

    def __init__(self, ranks: dict[str, int], lower_symbolic_constant: bool = False) -> None:
        self.ranks = ranks
        self.lower_symbolic_constant = lower_symbolic_constant
        self.changed = False
        self._ctr = 0

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name) or np_attr(node.value) != "pad":
            return node
        call = node.value
        kw = {k.arg: k.value for k in call.keywords}
        mode = kw.get("mode")
        mode_value = mode.value if isinstance(mode, ast.Constant) else None
        if mode_value not in ("edge", "constant") or not call.args:
            return node
        arr = call.args[0]
        rank = expr_rank(arr, self.ranks)
        if rank is None or rank < 1:
            return node
        pad_width = kw.get("pad_width") or (call.args[1] if len(call.args) > 1 else None)
        widths = const_pair_widths(pad_width, rank) if pad_width is not None else None
        if widths is None:
            return node
        if mode_value == "edge":
            self.changed = True
            stmts = pad_inline_stmts(node.targets[0].id, arr, widths, rank, self._ctr)
            self._ctr += 1
            return stmts
        if not self.lower_symbolic_constant or widths_all_literal(widths):
            return node
        self.changed = True
        fill = kw.get("constant_values", ast.Constant(value=0))
        stmts = pad_constant_inline_stmts(node.targets[0].id, arr, widths, fill, self._ctr)
        self._ctr += 1
        return stmts
