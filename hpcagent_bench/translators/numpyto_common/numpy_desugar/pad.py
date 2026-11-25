"""``np.pad`` lowered to an allocation plus slice copies."""

import ast
import copy
from collections.abc import Sequence

from hpcagent_bench.translators.numpyto_common.ast_build import const_int, walk_nodes
from hpcagent_bench.translators.numpyto_common.emit_helpers.numpy_names import numpy_call_attr
from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import name_store_counts
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import expr_rank

__all__ = [
    "PadInline",
    "const_pair_widths",
    "once_bound_tuples",
    "pad_constant_inline_stmts",
    "pad_inline_stmts",
    "widths_all_literal",
]


def once_bound_tuples(fn: ast.FunctionDef) -> dict[str, ast.Tuple]:
    """Names ``fn`` binds exactly once, to a tuple display."""
    counts = name_store_counts(fn)
    return {
        node.targets[0].id: node.value
        for node in walk_nodes(fn)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Tuple)
        and counts.get(node.targets[0].id) == 1
    }


def const_pair_widths(pad_width: ast.expr, rank: int) -> list[tuple[ast.expr, ast.expr]] | None:
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


def pad_inline_stmts(
    target: str, arr: ast.expr, widths: list[tuple[ast.expr, ast.expr]], rank: int, ctr: int
) -> list[ast.stmt]:
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


def widths_all_literal(widths: Sequence[tuple[ast.expr, ast.expr]]) -> bool:
    """True iff every (lo, hi) pair is a compile-time int (dace's own ``np.pad`` handles those)."""
    return all(const_int(lo) is not None and const_int(hi) is not None for lo, hi in widths)


def pad_constant_inline_stmts(
    target: str, arr: ast.expr, widths: Sequence[tuple[ast.expr, ast.expr]], fill: ast.expr, ctr: int
) -> list[ast.stmt]:
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

    Only the bare-assign form is matched. ``mode="constant"`` is lowered with ``lower_symbolic_constant`` (dace)
    and a non-literal width, and always with ``lower_constant`` (numba, which has no ``np.pad``); see
    :func:`pad_constant_inline_stmts`. ``tuples`` maps a once-bound name to the width tuple it holds (max_filter's
    ``pad_width = ((0, 0), (0, tail))``), which a bare name would otherwise read as one width for every side.
    """

    __slots__ = ("_ctr", "changed", "lower_constant", "lower_symbolic_constant", "ranks", "tuples")

    def __init__(
        self,
        ranks: dict[str, int],
        lower_symbolic_constant: bool = False,
        lower_constant: bool = False,
        tuples: dict[str, ast.Tuple] | None = None,
    ) -> None:
        self.ranks = ranks
        self.lower_symbolic_constant = lower_symbolic_constant
        self.lower_constant = lower_constant
        self.tuples = tuples or {}
        self.changed = False
        self._ctr = 0

    def visit_Assign(self, node: ast.Assign) -> ast.stmt | list[ast.stmt]:
        """Inline an edge-mode ``np.pad`` assignment as a loop nest."""
        self.generic_visit(node)
        call = node.value
        if (
            len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or not isinstance(call, ast.Call)
            or numpy_call_attr(call) != "pad"
        ):
            return node
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
        if isinstance(pad_width, ast.Name) and pad_width.id in self.tuples:
            pad_width = self.tuples[pad_width.id]
        widths = const_pair_widths(pad_width, rank) if pad_width is not None else None
        if widths is None:
            return node
        if mode_value == "edge":
            self.changed = True
            stmts = pad_inline_stmts(node.targets[0].id, arr, widths, rank, self._ctr)
            self._ctr += 1
            return stmts
        lower = self.lower_constant or (self.lower_symbolic_constant and not widths_all_literal(widths))
        if not lower:
            return node
        self.changed = True
        fill = kw.get("constant_values", ast.Constant(value=0))
        stmts = pad_constant_inline_stmts(node.targets[0].id, arr, widths, fill, self._ctr)
        self._ctr += 1
        return stmts
