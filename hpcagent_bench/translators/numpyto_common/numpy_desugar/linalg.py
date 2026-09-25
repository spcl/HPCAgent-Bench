"""``np.linalg`` cholesky / solve / inv lowered to loops for backends without them."""

import ast
from collections.abc import Callable

from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import DesugarError, np_submodule_attr
from hpcagent_bench.translators.numpyto_common.numpy_desugar.hoist import HoistForm, HoistTables, ValueHoist
from hpcagent_bench.translators.numpyto_common.numpy_desugar.kinds import dtype_kind
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import expr_rank


def cholesky_lines(temp: str, a: str, n: str, p: str, hermitian: bool = False) -> list[str]:
    """Lines computing ``np.linalg.cholesky(a)`` into a fresh ``temp`` (Cholesky-Banachiewicz).

    The ``np.zeros`` init supplies numpy's zero strict upper triangle. ``hermitian`` (complex ``a``)
    conjugates the second inner-product factor (``a = L L^H``) and takes the real part before ``sqrt``."""
    cj = "np.conj({0})" if hermitian else "{0}"
    diag = f"np.sqrt({p}_s.real)" if hermitian else f"np.sqrt({p}_s)"
    jk, ik = f"{temp}[{p}_j, {p}_k]", f"{temp}[{p}_i, {p}_k]"
    return [
        f"{temp} = np.zeros(({n}, {n}), {a}.dtype)",
        f"for {p}_j in range({n}):",
        f"    {p}_s = {a}[{p}_j, {p}_j]",
        f"    for {p}_k in range({p}_j):",
        f"        {p}_s -= {jk} * {cj.format(jk)}",
        f"    {temp}[{p}_j, {p}_j] = {diag}",
        f"    for {p}_i in range({p}_j + 1, {n}):",
        f"        {p}_t = {a}[{p}_i, {p}_j]",
        f"        for {p}_k in range({p}_j):",
        f"            {p}_t -= {ik} * {cj.format(jk)}",
        f"        {temp}[{p}_i, {p}_j] = {p}_t / {temp}[{p}_j, {p}_j]",
    ]


def gauss_jordan_lines(aw: str, o: str, n: str, m: str | None, p: str) -> list[str]:
    """Lines for in-place Gauss-Jordan with partial pivoting on ``(aw | o)``: ``o`` -> ``aw^-1 @ o``.

    ``m`` is non-None for a 2-D ``o``, ``None`` for 1-D. Core of ``solve`` (``o`` = copy of ``b``) and
    ``inv`` (``o`` = identity). Row-vectorized: a per-element loop's scalar temp aliasing an array
    element is mis-typed as an ndarray by pythran."""
    k, r = f"{p}_k", f"{p}_r"
    o_swap = [f"        {p}_to = {o}[{k}].copy()"] if m is not None else [f"        {p}_to = {o}[{k}]"]
    o_swap += [f"        {o}[{k}] = {o}[{p}_pv]", f"        {o}[{p}_pv] = {p}_to"]
    return (
        [
            f"for {k} in range({n}):",
            f"    {p}_pv = {k}",
            f"    for {r} in range({k} + 1, {n}):",
            f"        if np.abs({aw}[{r}, {k}]) > np.abs({aw}[{p}_pv, {k}]):",
            f"            {p}_pv = {r}",
            f"    if {p}_pv != {k}:",
            f"        {p}_tr = {aw}[{k}].copy()",
            f"        {aw}[{k}] = {aw}[{p}_pv]",
            f"        {aw}[{p}_pv] = {p}_tr",
        ]
        + o_swap
        + [
            f"    {p}_f = {aw}[{k}, {k}]",
            f"    {aw}[{k}] = {aw}[{k}] / {p}_f",
            f"    {o}[{k}] = {o}[{k}] / {p}_f",
            f"    for {r} in range({n}):",
            f"        if {r} != {k}:",
            f"            {p}_g = {aw}[{r}, {k}]",
            f"            {aw}[{r}] -= {p}_g * {aw}[{k}]",
            f"            {o}[{r}] -= {p}_g * {o}[{k}]",
        ]
    )


def linalg_operand(node: ast.expr, p: str, tag: str, hoist: ValueHoist) -> str:
    """A Name operand as is; any other expression materialised to a contiguous temp the loops can index."""
    if isinstance(node, ast.Name):
        return node.id
    nm = f"{p}_{tag}"
    hoist.queue([f"{nm} = np.ascontiguousarray({ast.unparse(node)})"])
    return nm


def hoist_cholesky(node: ast.Call, hoist: ValueHoist) -> ast.expr | None:
    a = node.args[0]
    ra = expr_rank(a, hoist.tables.ranks)
    if ra is None:
        return None
    if ra != 2:
        raise DesugarError(f"np.linalg.cholesky: only a 2-D operand is lowered (got ndim {ra})")
    p = f"__chol{hoist.ctr}"
    hoist.ctr += 1
    an = linalg_operand(a, p, "a", hoist)
    temp = f"{p}_o"
    hermitian = dtype_kind(a, hoist.tables.dtypes) == "complex"
    hoist.queue(cholesky_lines(temp, an, f"{an}.shape[0]", p, hermitian=hermitian))
    return ast.Name(id=temp, ctx=ast.Load())


def hoist_solve(node: ast.Call, hoist: ValueHoist) -> ast.expr | None:
    if len(node.args) < 2:
        return None
    a, b = node.args[0], node.args[1]
    tables = hoist.tables
    ra, rb = expr_rank(a, tables.ranks), expr_rank(b, tables.ranks)
    if ra is None or rb is None:
        return None
    # Gated here, not in ``hoist_linalg``: ownership depends on the rhs rank, and the raises
    # below must not fire for a call the backend handles natively.
    if "solve" not in tables.lower_ops and rb not in tables.solve_rhs_ranks:
        return None
    if ra != 2:
        raise DesugarError(f"np.linalg.solve: A must be 2-D (got ndim {ra})")
    if rb not in (1, 2):
        raise DesugarError(f"np.linalg.solve: b must be 1-D or 2-D (got ndim {rb})")
    p = f"__solv{hoist.ctr}"
    hoist.ctr += 1
    an, bn = linalg_operand(a, p, "a", hoist), linalg_operand(b, p, "b", hoist)
    temp = f"{p}_o"
    hoist.queue(
        [f"{p}_aw = {an}.copy()", f"{temp} = {bn}.copy()"]
        + gauss_jordan_lines(f"{p}_aw", temp, f"{an}.shape[0]", (f"{bn}.shape[1]" if rb == 2 else None), p)
    )
    return ast.Name(id=temp, ctx=ast.Load())


def hoist_inv(node: ast.Call, hoist: ValueHoist) -> ast.expr | None:
    a = node.args[0]
    ra = expr_rank(a, hoist.tables.ranks)
    if ra is None:
        return None
    if ra != 2:
        raise DesugarError(f"np.linalg.inv: only a 2-D operand is lowered (got ndim {ra})")
    p = f"__inv{hoist.ctr}"
    hoist.ctr += 1
    an = linalg_operand(a, p, "a", hoist)
    temp, n = f"{p}_o", f"{an}.shape[0]"
    hoist.queue(
        [
            f"{p}_aw = {an}.copy()",
            f"{temp} = np.zeros(({n}, {n}), {an}.dtype)",
            f"for {p}_d in range({n}):",
            f"    {temp}[{p}_d, {p}_d] = 1",
        ]
        + gauss_jordan_lines(f"{p}_aw", temp, n, n, p)
    )
    return ast.Name(id=temp, ctx=ast.Load())


LINALG_LOWERINGS: dict[str, Callable[[ast.Call, ValueHoist], ast.expr | None]] = {
    "cholesky": hoist_cholesky,
    "inv": hoist_inv,
}


def hoist_linalg(node: ast.AST, hoist: ValueHoist) -> ast.expr | None:
    """A lowerable ``np.linalg.cholesky/solve/inv(...)`` -> the temp its loop nest fills.

    The temp is filled before the statement runs, so ``A[:] = np.linalg.cholesky(A) + ...`` stays safe.
    Only ops in ``lower_ops`` and ``solve`` rhs ranks in ``solve_rhs_ranks`` are touched. A >2-D operand
    raises :class:`DesugarError`; an unknown-rank operand is left verbatim."""
    if not isinstance(node, ast.Call) or not node.args:
        return None
    op = np_submodule_attr(node, "linalg")
    if op == "solve":
        return hoist_solve(node, hoist)
    if op is None or op not in hoist.tables.lower_ops:
        return None
    return LINALG_LOWERINGS[op](node, hoist)


#: numpy.linalg ops the desugar can lower to loops.
LINALG_LOWERABLE = {"cholesky", "solve", "inv"}


#: numpy.linalg ops each backend compiles natively (left in place); pythran has no numpy.linalg.
NATIVE_LINALG: dict[str | None, set] = {
    "numba": {"cholesky", "solve", "inv"},
    "dace": {"cholesky", "solve", "inv"},
    "pythran": set(),
}


#: ``solve`` right-hand-side ranks lowered per backend even though :data:`NATIVE_LINALG` lists ``solve``.
#: DaCe's ``Solve`` library node reads ``shape_out[1]`` unconditionally, so a 1-D ``b`` fails at expansion
#: (after the frontend accepted it); the 2-D rhs keeps the native node.
LOWER_SOLVE_RHS_RANKS: dict[str | None, frozenset] = {"dace": frozenset({1})}


def lowers_linalg(tables: HoistTables) -> bool:
    """Whether the backend lowers any ``np.linalg`` call at all (pythran every op, dace a ``solve`` rhs rank)."""
    return bool(tables.lower_ops or tables.solve_rhs_ranks)


LINALG_HOIST = HoistForm(frozenset(LINALG_LOWERABLE), (), hoist_linalg, lowers_linalg)
