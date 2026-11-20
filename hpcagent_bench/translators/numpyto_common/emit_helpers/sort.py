"""Decoding of the sort library marker (``lib_nodes.SORT_LIBRARY_MARKER``).

The lowering leaves ``np.sort`` / ``np.argsort`` along the last axis as a marker call; each native backend renders
it as one call to a ``npb_[arg]sort_<type>(out, src, rows, n)`` helper it defines with its library sort.
"""

import ast
import dataclasses

from hpcagent_bench.translators.numpyto_common.emit_helpers.fftw import marker_constant, marker_operands

__all__ = ["SortCall", "sort_call"]


@dataclasses.dataclass(frozen=True, slots=True)
class SortCall:
    """``n``-long rows of ``src``, each sorted (or argsorted) into the same row of ``out``."""

    out: str
    src: str
    argsort: bool
    #: The operand's leading extents, whose product is the row count (empty for a 1-D operand).
    rows: list[ast.expr]
    n: ast.expr

    @property
    def helper(self) -> str:
        """The helper's name stem: ``npb_sort`` or ``npb_argsort``."""
        return "npb_argsort" if self.argsort else "npb_sort"


def sort_call(node: ast.Call) -> SortCall:
    """``marker(out, src, argsort_flag, *extents)``."""
    out, src = marker_operands(node)
    *rows, n = node.args[3:]
    return SortCall(out, src, bool(marker_constant(node.args[2])), rows, n)
