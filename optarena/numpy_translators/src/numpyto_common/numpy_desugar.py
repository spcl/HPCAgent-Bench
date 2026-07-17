"""Desugar numpy ops the verbatim Python backends (numba / pythran) cannot
compile into the equivalent plain-numpy loops they CAN.

The C / Fortran backends lower these constructs through the full IR pipeline,
but ``numpyto_numba`` and ``numpyto_pythran`` emit the kernel body verbatim --
so a kernel using a batched (>=3-D) ``@`` / ``np.matmul``, ``np.pad`` or
``np.einsum`` fails to type (numba) or template-instantiate (pythran). This
module rewrites those constructs at the source-AST level into framework-
compatible numpy, preserving semantics, so both backends emit a valid variant.

Entry point: :func:`desugar_for_python_backend` -- parse the kernel source,
expand the unsupported ops, return the rewritten source. Shape ranks come from
the parsed :class:`KernelIR` (declared arrays) plus a light local-allocation
walk, which is all the batched-``@`` rewrite needs to tell a batched matmul
(operand rank > 2) from an ordinary 2-D one (which numba / pythran handle).
"""
import ast
import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from numpyto_common import dtypes


class DesugarError(NotImplementedError):
    """A desugar pass matched a construct it OWNS but hit a variant it cannot
    lower correctly. Raised (never swallowed) so the emit fails loudly instead of
    producing a silently-wrong kernel -- a construct we do NOT own is left
    verbatim (a clean backend skip), only an owned-but-unhandled shape raises."""


# Constructors whose first arg is a shape tuple -> result rank = len(shape).
_SHAPE_CTORS = {"empty", "zeros", "ones", "full", "ndarray"}
_LIKE_CTORS = {"empty_like", "zeros_like", "ones_like"}

# dtype "kind" ordered by promotion rank (numpy-style: bool < int < float < complex).
_KIND_RANK = {"bool": 0, "int": 1, "float": 2, "complex": 3}
#: ``np.<name>`` dtype spellings -> kind.
_DTYPE_NAME_KIND = {
    "bool": "bool",
    "bool_": "bool",
    "int8": "int",
    "int16": "int",
    "int32": "int",
    "int64": "int",
    "intp": "int",
    "intc": "int",
    "uint8": "int",
    "uint16": "int",
    "uint32": "int",
    "uint64": "int",
    "float16": "float",
    "float32": "float",
    "float64": "float",
    "float": "float",
    "double": "float",
    "complex64": "complex",
    "complex128": "complex",
    "complex": "complex"
}


def _kind_of_dtype_str(dt: Optional[str]) -> Optional[str]:
    """A numpy dtype tag string (``"int64"``, ``"float32"``) -> its kind."""
    if not dt:
        return None
    return _DTYPE_NAME_KIND.get(dt) or ("int" if dt.startswith(("int", "uint")) else "float" if dt.startswith(
        "float") else "complex" if dt.startswith("complex") else "bool" if dt.startswith("bool") else None)


def _dtype_arg_kind(node: ast.AST) -> Optional[str]:
    """A dtype ARGUMENT (``np.int64`` / ``np.dtype('f8')`` / a bare name) -> kind."""
    if isinstance(node, ast.Attribute):
        return _DTYPE_NAME_KIND.get(node.attr)
    if isinstance(node, ast.Name):
        return _DTYPE_NAME_KIND.get(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _kind_of_dtype_str(node.value)
    return None


def _promote_kind(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """numpy-style promotion; ``None`` (unknown) is contagious so callers stay
    conservative (an unknown operand never masquerades as a known kind)."""
    if a is None or b is None:
        return None
    return a if _KIND_RANK[a] >= _KIND_RANK[b] else b


def _np_attr(node: ast.AST) -> Optional[str]:
    """``np.<attr>`` / ``numpy.<attr>`` call -> ``attr`` else None."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
            and node.func.value.id in ("np", "numpy")):
        return node.func.attr
    return None


def _np_fft_attr(node: ast.AST) -> Optional[str]:
    """``np.fft.<attr>(...)`` / ``numpy.fft.<attr>(...)`` call -> ``attr`` (one
    of ``fft``/``ifft``/``fft2``/``ifft2``/``fftn``/``ifftn``), else None. The
    call func is a two-level Attribute (``np.fft.fft``), so the single-level
    ``_np_attr`` misses it."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "fft"
            and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id in ("np", "numpy")):
        return node.func.attr
    return None


def _tuple_len(node: ast.AST) -> Optional[int]:
    if isinstance(node, (ast.Tuple, ast.List)):
        return len(node.elts)
    return None


def _is_newaxis(e: ast.AST) -> bool:
    """``None`` / ``np.newaxis`` / bare ``newaxis`` -- a subscript newaxis."""
    return ((isinstance(e, ast.Constant) and e.value is None) or (isinstance(e, ast.Attribute) and e.attr == "newaxis")
            or (isinstance(e, ast.Name) and e.id == "newaxis"))


def _is_ellipsis(e: ast.AST) -> bool:
    """A ``...`` subscript entry (``ast.Constant(Ellipsis)``). It expands to
    full slices over every otherwise-unindexed axis, so it drops NO axis."""
    return isinstance(e, ast.Constant) and e.value is Ellipsis


#: numpy reductions that take an ``axis`` (drops the reduced axes; no axis ->
#: scalar). Used only for ndim propagation, not rewriting.
_REDUCE_FNS = {"sum", "prod", "mean", "std", "var", "min", "max", "amin", "amax", "argmin", "argmax", "any", "all"}

#: ufuncs whose result is always a boolean array (regardless of operand dtype).
_BOOL_UFUNCS = {
    "logical_and", "logical_or", "logical_not", "logical_xor", "isnan", "isinf", "isfinite", "isclose", "equal",
    "not_equal", "less", "less_equal", "greater", "greater_equal"
}


def _expr_rank(value: ast.AST, ranks: Dict[str, int]) -> Optional[int]:
    """Best-effort ndim of an expression given the current rank table."""
    if isinstance(value, ast.Name):
        return ranks.get(value.id)
    if isinstance(value, ast.BinOp):
        if isinstance(value.op, ast.MatMult):
            lr = _expr_rank(value.left, ranks)
            rr = _expr_rank(value.right, ranks)
            if lr is None or rr is None:
                return None
            # matmul: 1-D operands contract to a scalar; otherwise the result
            # keeps the larger batch rank (numpy stacks/broadcasts leading axes).
            if lr == 1 and rr == 1:
                return 0
            return max(lr, rr)
        lr = _expr_rank(value.left, ranks)
        rr = _expr_rank(value.right, ranks)
        return max([r for r in (lr, rr) if r is not None], default=None)
    if isinstance(value, ast.UnaryOp):
        return _expr_rank(value.operand, ranks)
    if isinstance(value, ast.Compare):
        rs = [_expr_rank(value.left, ranks)] + [_expr_rank(c, ranks) for c in value.comparators]
        return max([r for r in rs if r is not None], default=None)  # bool mask keeps operand rank
    if isinstance(value, ast.BoolOp):
        rs = [_expr_rank(v, ranks) for v in value.values]
        return max([r for r in rs if r is not None], default=None)
    if isinstance(value, ast.Subscript):
        base = _expr_rank(value.value, ranks)
        if base is None:
            return None
        sl = value.slice
        if isinstance(sl, ast.Slice):
            return base  # a full/partial slice keeps the rank
        if _is_newaxis(sl):
            return base + 1  # a[None] / a[np.newaxis] -- a newaxis adds a dimension
        if _is_ellipsis(sl):
            return base  # a[...] keeps every axis
        if isinstance(sl, ast.Tuple):
            # slices keep a dim, newaxis adds one, an integer/array index removes
            # one, and an ellipsis (``a[..., i]``) expands to full slices over
            # all otherwise-unindexed axes -- it drops NOTHING.
            drop = 0
            for e in sl.elts:
                if isinstance(e, ast.Slice) or _is_ellipsis(e):
                    continue
                if _is_newaxis(e):
                    drop -= 1
                else:
                    drop += 1
            return base - drop
        return base - 1  # single integer/Name index
    if isinstance(value, ast.Call):
        if isinstance(value.func, ast.Name) and value.func.id == "abs" and value.args:
            return _expr_rank(value.args[0], ranks)  # builtin abs is elementwise
        if _np_fft_attr(value) and value.args:
            return _expr_rank(value.args[0], ranks)  # fft/ifft/fftn... preserve rank
        attr = _np_attr(value)
        if attr in ("arange", "linspace"):
            return 1  # always 1-D
        if attr in _REDUCE_FNS and value.args:
            base = _expr_rank(value.args[0], ranks)
            if base is None:
                return None
            kw = {k.arg: k.value for k in value.keywords}
            ax = kw.get("axis") or (value.args[1] if len(value.args) > 1 else None)
            if ax is None:
                return 0  # full reduction -> scalar
            if isinstance(ax, (ast.Tuple, ast.List)):
                return base - len(ax.elts)
            return base - 1  # single reduced axis
        if attr in _SHAPE_CTORS and value.args:
            n = _tuple_len(value.args[0])
            if n is not None:
                return n
            a0 = value.args[0]
            if (isinstance(a0, ast.Attribute) and a0.attr == "shape"):
                return _expr_rank(a0.value, ranks)  # np.zeros(C.shape, ...) keeps C's rank
            if isinstance(a0, (ast.Name, ast.Constant)):
                return 1  # 1-D length
        if attr in _LIKE_CTORS and value.args:
            return _expr_rank(value.args[0], ranks)
        if attr in ("reshape", ) and len(value.args) >= 2:
            n = _tuple_len(value.args[1])
            if n is not None:
                return n
        if attr in ("copy", "ascontiguousarray", "asarray", "array") and value.args:
            return _expr_rank(value.args[0], ranks)
        if attr == "matmul" and len(value.args) == 2:
            return _expr_rank(ast.BinOp(left=value.args[0], op=ast.MatMult(), right=value.args[1]), ranks)
        # rank-preserving methods ``x.astype(dt)`` / ``x.copy()`` (receiver's rank).
        if (isinstance(value.func, ast.Attribute) and value.func.attr in ("astype", "copy", "ravel")):
            return _expr_rank(value.func.value, ranks) if value.func.attr != "ravel" else 1
        # ``x.reshape((a, b))`` method form.
        if (isinstance(value.func, ast.Attribute) and value.func.attr == "reshape" and value.args):
            n = _tuple_len(value.args[0]) or (len(value.args) if all(
                isinstance(a, (ast.Name, ast.Constant)) for a in value.args) else None)
            if n is not None:
                return n
        # Fallback for any other ``np.<fn>(...)``: the remaining numpy functions
        # reaching here are elementwise / broadcasting ufuncs (abs, sqrt, exp,
        # less, greater, minimum, maximum, where, conj, ...), so the result rank
        # is the max of the argument ranks. (Rank-changing np.* -- constructors,
        # reductions, reshape, matmul -- returned above.)
        if attr is not None:
            rs = [_expr_rank(a, ranks) for a in value.args]
            return max([r for r in rs if r is not None], default=None)
    return None


def _call_return_rank(value: ast.AST, call_returns: Dict[str, int]) -> Optional[int]:
    """Rank of ``helper(...)`` when ``helper`` is a local function with a known
    return rank -- ``_expr_rank`` alone returns None for a call to a non-numpy
    Name, so ``x = relu(a @ b)`` would leave ``x`` untracked."""
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        return call_returns.get(value.func.id)
    return None


def _rank_table(tree: ast.AST, seed: Dict[str, int], call_returns: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Propagate ndim across straight-line assignments to a fixpoint. ``call_returns``
    (a ``{helper: return_ndim}`` map) lets a local bound to a helper call inherit
    that helper's return rank (the ML kernels thread arrays through relu/conv2d
    helpers, which ``_expr_rank`` cannot see into)."""
    ranks = dict(seed)
    for _ in range(8):
        changed = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
                r = _expr_rank(node.value, ranks)
                if r is None and call_returns is not None:
                    r = _call_return_rank(node.value, call_returns)
                if r is not None and ranks.get(node.targets[0].id) != r:
                    ranks[node.targets[0].id] = r
                    changed = True
        if not changed:
            break
    return ranks


def _dtype_kind(value: ast.AST, dtypes: Dict[str, str]) -> Optional[str]:
    """Best-effort dtype KIND (``bool``/``int``/``float``/``complex``) of an
    expression given the current name->kind table. ``None`` = unknown; callers
    must treat unknown conservatively (e.g. not desugar a matmul as integer)."""
    if isinstance(value, ast.Name):
        return dtypes.get(value.id)
    if isinstance(value, ast.Constant):
        v = value.value
        return ("bool" if isinstance(v, bool) else "int" if isinstance(v, int) else
                "float" if isinstance(v, float) else "complex" if isinstance(v, complex) else None)
    if isinstance(value, (ast.Compare, ast.BoolOp)):
        return "bool"
    if isinstance(value, ast.UnaryOp):
        return "bool" if isinstance(value.op, ast.Not) else _dtype_kind(value.operand, dtypes)
    if isinstance(value, ast.BinOp):
        lk, rk = _dtype_kind(value.left, dtypes), _dtype_kind(value.right, dtypes)
        if isinstance(value.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
            return "bool" if (lk == "bool" or rk == "bool") else _promote_kind(lk, rk)
        if isinstance(value.op, (ast.Div, ast.MatMult)):
            p = _promote_kind(lk, rk)
            return "float" if p in ("int", "bool") else p  # true division promotes to float
        return _promote_kind(lk, rk)
    if isinstance(value, ast.Subscript):
        return _dtype_kind(value.value, dtypes)  # indexing preserves dtype
    if isinstance(value, ast.Call):
        f = value.func
        if isinstance(f, ast.Attribute) and f.attr == "astype" and value.args:
            return _dtype_arg_kind(value.args[0])
        attr = _np_attr(value)
        if attr in _BOOL_UFUNCS:
            return "bool"  # logical_and / less / isnan ... always produce a bool array
        if attr in _DTYPE_NAME_KIND:
            return _DTYPE_NAME_KIND[attr]  # np.int64(x) scalar cast
        if attr in _SHAPE_CTORS:
            kw = {k.arg: k.value for k in value.keywords}
            dt = kw.get("dtype") or (value.args[1] if len(value.args) > 1 else None)
            return _dtype_arg_kind(dt) if dt is not None else "float"  # default float64
        if attr in _LIKE_CTORS and value.args:
            return _dtype_kind(value.args[0], dtypes)
        if attr in ("astype", "copy", "ascontiguousarray", "asarray", "array", "reshape") and value.args:
            return _dtype_kind(value.args[0], dtypes)
        if attr in ("where", "minimum", "maximum", "clip") and len(value.args) >= 2:
            return _promote_kind(_dtype_kind(value.args[-2], dtypes), _dtype_kind(value.args[-1], dtypes))
        if attr in ("abs", "sqrt", "exp", "sin", "cos", "conj", "real", "imag", "sum", "prod") and value.args:
            return _dtype_kind(value.args[0], dtypes)
    return None


def _dtype_table(tree: ast.AST, seed: Dict[str, str]) -> Dict[str, str]:
    """Propagate dtype kinds across straight-line assignments to a fixpoint."""
    dtypes = dict(seed)
    for _ in range(8):
        changed = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
                k = _dtype_kind(node.value, dtypes)
                if k is not None and dtypes.get(node.targets[0].id) != k:
                    dtypes[node.targets[0].id] = k
                    changed = True
        if not changed:
            break
    return dtypes


def _matmul_pairs(node: ast.AST) -> List[ast.AST]:
    """Every matmul (``@`` BinOp or ``np.matmul`` call) under ``node``."""
    out: List[ast.AST] = []
    for n in ast.walk(node):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult):
            out.append(n)
        elif _np_attr(n) == "matmul" and len(getattr(n, "args", [])) == 2:
            out.append(n)
    return out


def _matmul_operands(mm: ast.AST):
    if isinstance(mm, ast.BinOp):
        return mm.left, mm.right
    return mm.args[0], mm.args[1]


class _IndexLeadingAxis(ast.NodeTransformer):
    """Subscript every rank > 2 ``Name`` by ``[bv]`` (its leading/batch axis),
    dropping it to a 2-D operand. Names of rank <= 2 are left untouched (a
    shared 2-D right operand broadcasts across the batch)."""

    def __init__(self, bv: str, ranks: Dict[str, int]):
        self.bv = bv
        self.ranks = ranks

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and (self.ranks.get(node.id, 0) or 0) > 2:
            return ast.copy_location(
                ast.Subscript(value=ast.Name(id=node.id, ctx=ast.Load()),
                              slice=ast.Name(id=self.bv, ctx=ast.Load()),
                              ctx=ast.Load()), node)
        return node


class _BatchedMatmulToLoop(ast.NodeTransformer):
    """``Q[:] = Q + I @ star`` (I rank-3) -> a loop over the batch axis doing a
    2-D GEMM per element. numba / pythran support 2-D ``@`` but not the stacked
    (>=3-D) form -- and Fortran has no matmul-broadcast either, so this is the
    universal "batched GEMM = for-loop over GEMMs" lowering."""

    def __init__(self, ranks: Dict[str, int]):
        self.ranks = ranks
        self._ctr = 0
        self.changed = False

    def _batch_source(self, value: ast.AST) -> Optional[ast.Name]:
        """First rank > 2 Name feeding a batched matmul -- its leading axis is
        the batch extent."""
        for mm in _matmul_pairs(value):
            for op in _matmul_operands(mm):
                for n in ast.walk(op):
                    if isinstance(n, ast.Name) and (self.ranks.get(n.id, 0) or 0) > 2:
                        return n
        return None

    def _is_batched(self, value: ast.AST) -> bool:
        """True iff the statement carries a CLEANLY batched matmul: at least one
        operand has rank > 2 AND every operand is a bare ``Name``. The bare-Name
        guard is load-bearing -- a ``reshape`` / ``transpose`` wrapping the
        operand restructures axes, so indexing its leading axis (doitgen's
        ``np.reshape(A, (NR, NQ, 1, NP)) @ C4``) would be a miscompile, not a
        batched GEMM. Those stay verbatim (and skip on numba/pythran)."""
        for mm in _matmul_pairs(value):
            a, b = _matmul_operands(mm)
            if not (isinstance(a, ast.Name) and isinstance(b, ast.Name)):
                return False
            ra, rb = _expr_rank(a, self.ranks), _expr_rank(b, self.ranks)
            if (ra or 0) > 2 or (rb or 0) > 2:
                return True
        return False

    def _index_target(self, target: ast.AST, bv: str) -> Optional[ast.AST]:
        """``T[:]`` or bare rank > 2 ``T`` -> ``T[bv]``."""
        if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                and isinstance(target.slice, ast.Slice) and target.slice.lower is None and target.slice.upper is None):
            return ast.Subscript(value=ast.Name(id=target.value.id, ctx=ast.Load()),
                                 slice=ast.Name(id=bv, ctx=ast.Load()),
                                 ctx=ast.Store())
        if isinstance(target, ast.Name) and (self.ranks.get(target.id, 0) or 0) > 2:
            return ast.Subscript(value=ast.Name(id=target.id, ctx=ast.Load()),
                                 slice=ast.Name(id=bv, ctx=ast.Load()),
                                 ctx=ast.Store())
        return None

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if len(node.targets) != 1 or not self._is_batched(node.value):
            return node
        bsrc = self._batch_source(node.value)
        if bsrc is None:
            return node
        new_target = self._index_target(node.targets[0], "")  # probe form first
        if new_target is None:
            return node  # target not a recognised batched whole-array write
        bv = f"__bm{self._ctr}"
        self._ctr += 1
        self.changed = True
        new_target = self._index_target(node.targets[0], bv)
        new_value = _IndexLeadingAxis(bv, self.ranks).visit(copy.deepcopy(node.value))
        extent = ast.Subscript(value=ast.Attribute(value=ast.Name(id=bsrc.id, ctx=ast.Load()),
                                                   attr="shape",
                                                   ctx=ast.Load()),
                               slice=ast.Constant(value=0),
                               ctx=ast.Load())
        loop = ast.For(target=ast.Name(id=bv, ctx=ast.Store()),
                       iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[extent], keywords=[]),
                       body=[ast.Assign(targets=[new_target], value=new_value)],
                       orelse=[])
        return ast.copy_location(loop, node)


def _const_pair_widths(pad_width: ast.AST, rank: int):
    """Parse ``pad_width`` into a list of ``(lo, hi)`` AST exprs per axis, or
    None if the form is unsupported. Accepts a scalar ``R`` (symmetric, every
    axis) or a per-axis tuple ``((lo, hi), ...)``."""
    if isinstance(pad_width, (ast.Name, ast.Constant)):
        return [(pad_width, copy.deepcopy(pad_width)) for _ in range(rank)]
    if isinstance(pad_width, (ast.Tuple, ast.List)) and len(pad_width.elts) == rank:
        out = []
        for e in pad_width.elts:
            if isinstance(e, (ast.Tuple, ast.List)) and len(e.elts) == 2:
                out.append((e.elts[0], e.elts[1]))
            else:
                return None
        return out
    return None


def _pad_inline_stmts(target: str, arr: ast.AST, widths, rank: int, ctr: int) -> List[ast.stmt]:
    """Inline ``<target> = np.pad(arr, ..., mode="edge")`` as a clamped-index
    copy loop nest (numba / pythran compile this; neither supports np.pad).
    Inlining -- rather than a helper call -- sidesteps numba's rule that an
    ``@njit`` body may only call other ``@njit`` functions (pythran wants the
    opposite, plain defs), so one expansion serves both backends."""
    p = f"__pd{ctr}"
    xv = f"{p}_x"
    src = [f"{xv} = {ast.unparse(arr)}"]
    dims = [f"{xv}.shape[{i}] + {ast.unparse(lo)} + {ast.unparse(hi)}" for i, (lo, hi) in enumerate(widths)]
    src.append(f"{target} = np.empty(({', '.join(dims)},), {xv}.dtype)")
    indent = ""
    for i, (lo, hi) in enumerate(widths):
        src.append(f"{indent}for {p}_i{i} in range({dims[i]}):")
        indent += "    "
        src.append(f"{indent}{p}_s{i} = min(max({p}_i{i} - "
                   f"{ast.unparse(lo)}, 0), {xv}.shape[{i}] - 1)")
    idx_o = ", ".join(f"{p}_i{i}" for i in range(rank))
    idx_s = ", ".join(f"{p}_s{i}" for i in range(rank))
    src.append(f"{indent}{target}[{idx_o}] = {xv}[{idx_s}]")
    return ast.parse("\n".join(src)).body


class _PadInline(ast.NodeTransformer):
    """Replace ``name = np.pad(x, pad_width=..., mode="edge")`` with an inline
    edge-pad loop nest. Only the bare-assign form is handled; np.pad nested in a
    larger expression is left verbatim (no misfire)."""

    def __init__(self, ranks: Dict[str, int]):
        self.ranks = ranks
        self.changed = False
        self._ctr = 0

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if (len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name) or _np_attr(node.value) != "pad"):
            return node
        call = node.value
        kw = {k.arg: k.value for k in call.keywords}
        mode = kw.get("mode")
        if not (isinstance(mode, ast.Constant) and mode.value == "edge") or not call.args:
            return node  # only edge mode, array as first positional
        arr = call.args[0]
        rank = _expr_rank(arr, self.ranks)
        if rank is None or rank < 1:
            return node
        pad_width = kw.get("pad_width") or (call.args[1] if len(call.args) > 1 else None)
        widths = _const_pair_widths(pad_width, rank) if pad_width is not None else None
        if widths is None:
            return node
        self.changed = True
        stmts = _pad_inline_stmts(node.targets[0].id, arr, widths, rank, self._ctr)
        self._ctr += 1
        return stmts


def _einsum_inline_stmts(subs: str, operands: List[str], ctr: int):
    """Source statements computing ``np.einsum(subs, *operands)`` into a fresh
    temp via an explicit contraction loop nest (output indices outer, contracted
    indices inner-accumulate). Returns ``(stmts, temp_name)`` or ``(None, None)``
    when the form is unsupported (ellipsis / scalar output). numba and pythran
    compile this; neither supports ``np.einsum`` on these shapes."""
    from numpyto_common.lib_nodes import _parse_einsum_subscripts
    try:
        in_subs, out_sub = _parse_einsum_subscripts(subs)
    except Exception:  # noqa: BLE001 -- ellipsis / malformed -> caller bails
        return None, None
    if not out_sub or len(in_subs) != len(operands):
        return None, None  # scalar full-contraction not handled here
    # index char -> (operand index, axis): first operand carrying it.
    char_src: Dict[str, tuple] = {}
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
    src = [f"{p} = np.empty(({outshape},), {operands[0]}.dtype)"]
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


class _EinsumHoister(ast.NodeTransformer):
    """Replace each ``np.einsum(...)`` (bare-Name operands) inside one statement
    with a fresh temp Name, accumulating the temp's compute statements in
    ``self.pre`` to be spliced before the statement."""

    def __init__(self, ctr: int):
        self.ctr = ctr
        self.pre: List[ast.stmt] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)  # inner einsums first
        if _np_attr(node) != "einsum" or not node.args:
            return node
        if not (isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            return node
        operands = node.args[1:]
        if not operands or not all(isinstance(o, ast.Name) for o in operands):
            return node  # only bare-array operands -> else leave verbatim
        stmts, temp = _einsum_inline_stmts(node.args[0].value, [o.id for o in operands], self.ctr)
        if stmts is None:
            return node
        self.ctr += 1
        self.pre.extend(stmts)
        return ast.copy_location(ast.Name(id=temp, ctx=ast.Load()), node)


class _EinsumInline(ast.NodeTransformer):
    """Hoist ``np.einsum`` out of any value-bearing statement into a preceding
    contraction loop nest. Handles einsum nested in arithmetic (seissol's
    ``Q[:] = Q + np.einsum(...)``)."""

    def __init__(self):
        self.changed = False
        self._ctr = 0

    def _hoist(self, node):
        if getattr(node, "value", None) is None:
            return node
        h = _EinsumHoister(self._ctr)
        node.value = h.visit(node.value)
        self._ctr = h.ctr
        if h.pre:
            self.changed = True
            return h.pre + [node]
        return node

    def visit_Assign(self, node):
        return self._hoist(node)

    def visit_AugAssign(self, node):
        return self._hoist(node)

    def visit_Return(self, node):
        return self._hoist(node)

    def visit_Expr(self, node):
        return self._hoist(node)


def _fft_axes(fattr: str, call: ast.Call, rank: int):
    """``(transform_axes, inverse)`` for an ``np.fft.<fattr>`` call, or
    ``(None, inverse)`` when the axis spec is non-constant (caller bails, leaving
    the call verbatim). ``fft``/``ifft`` take one ``axis`` (default last);
    ``fftn``/``ifftn`` an ``axes`` sequence (default ALL); ``fft2``/``ifft2`` the
    last two axes. Negative axes wrap modulo ``rank``."""
    kwargs = {k.arg: k.value for k in call.keywords}
    inverse = fattr.startswith("i")
    base = fattr[1:] if inverse else fattr
    if base == "fft":
        ax = kwargs.get("axis") or (call.args[1] if len(call.args) > 1 else None)
        if ax is None:
            return [rank - 1], inverse
        if isinstance(ax, ast.Constant) and isinstance(ax.value, int):
            return [ax.value % rank], inverse
        return None, inverse
    if base == "fft2":
        return ([rank - 2, rank - 1] if rank >= 2 else None), inverse
    if base == "fftn":
        axes = kwargs.get("axes") or (call.args[1] if len(call.args) > 1 else None)
        if axes is None:
            return list(range(rank)), inverse
        if isinstance(axes, (ast.Tuple, ast.List)) and all(
                isinstance(e, ast.Constant) and isinstance(e.value, int) for e in axes.elts):
            return [e.value % rank for e in axes.elts], inverse
        return None, inverse
    return None, inverse


def _fft_inline_stmts(tname: str, sname: str, taxes: List[int], rank: int, inverse: bool, ctr: int,
                      alloc: bool) -> List[ast.stmt]:
    """Source statements computing ``np.fft.*`` into ``tname`` (shape == source)
    as a naive DFT loop nest -- the same O(prod(N_t)^2) transform the C/Fortran
    backends lower, but as plain numpy (``np.exp`` of a complex phase, complex
    ``+=``) that numba njit-compiles and pythran template-instantiates. Output
    indices iterate every axis; summation iterators only the transform axes
    ``taxes`` (batch axes ride the output iterator). Inverse uses ``+1j`` and
    divides by ``prod(N_t)``. ``alloc`` allocates ``tname`` (bare-Name target);
    a ``tname[:]`` slice target writes the existing buffer in place."""
    p = f"__ft{ctr}"
    sign = "1j" if inverse else "-1j"
    # Bind each axis size to an int local first. pythran otherwise forward-
    # substitutes ``sname.shape[i]`` into ``range(...)`` over a lazy numpy_expr
    # source and fails template type inference; an int local pins it to ``long``.
    d = [f"{p}_d{i}" for i in range(rank)]
    lines: List[str] = [f"{d[i]} = {sname}.shape[{i}]" for i in range(rank)]
    if alloc:
        lines.append(f"{tname} = np.zeros(({', '.join(d)},), np.complex128)")
    o = [f"{p}_k{i}" for i in range(rank)]
    ind = ""
    for i in range(rank):
        lines.append(f"{ind}for {o[i]} in range({d[i]}):")
        ind += "    "
    oidx = ", ".join(o)
    lines.append(f"{ind}{tname}[{oidx}] = 0j")
    n = {t: f"{p}_n{t}" for t in taxes}
    cind = ind
    for t in taxes:
        lines.append(f"{cind}for {n[t]} in range({d[t]}):")
        cind += "    "
    terms = [f"(2.0 * 3.141592653589793 * {o[t]} * {n[t]} / {d[t]})" for t in taxes]
    phase = " + ".join(terms)
    sidx = ", ".join((n[ax] if ax in taxes else o[ax]) for ax in range(rank))
    lines.append(f"{cind}{tname}[{oidx}] += {sname}[{sidx}] * np.exp({sign} * ({phase}))")
    if inverse:
        denom = " * ".join(d[t] for t in taxes)
        lines.append(f"{ind}{tname}[{oidx}] = {tname}[{oidx}] / ({denom})")
    return ast.parse("\n".join(lines)).body


class _FftInline(ast.NodeTransformer):
    """Replace ``out = np.fft.fft/ifft/fftn/ifftn/fft2/ifft2(x)`` (and the
    ``out[:] =`` slice-assign form) with a naive-DFT loop nest. numba supports no
    ``np.fft`` at all; pythran supports 1-D ``fft``/``ifft`` but not N-D
    ``fftn``/``ifftn`` -- lowering all variants uniformly keeps one code path
    (the loop DFT matches numpy to ~1e-15 at any realistic size). A non-Name
    argument (``ifftn(u1 * np.exp(...))``) is hoisted to a temp first so the loop
    body can index it; a non-constant axis spec leaves the call verbatim."""

    def __init__(self, ranks: Dict[str, int]):
        self.ranks = ranks
        self.changed = False
        self._ctr = 0

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        fattr = _np_fft_attr(node.value)
        if fattr is None or not node.value.args:
            return node
        tgt = node.targets[0]
        if isinstance(tgt, ast.Name):
            tname, alloc = tgt.id, True
        elif (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and isinstance(tgt.slice, ast.Slice)
              and tgt.slice.lower is None and tgt.slice.upper is None):
            tname, alloc = tgt.value.id, False
        else:
            return node
        arg = node.value.args[0]
        rank = _expr_rank(arg, self.ranks)
        if rank is None or rank < 1:
            return node
        taxes, inverse = _fft_axes(fattr, node.value, rank)
        if not taxes:
            return node
        pre: List[ast.stmt] = []
        if isinstance(arg, ast.Name):
            sname = arg.id
        else:
            sname = f"__fti{self._ctr}"
            pre = ast.parse(f"{sname} = {ast.unparse(arg)}").body
        stmts = _fft_inline_stmts(tname, sname, taxes, rank, inverse, self._ctr, alloc)
        self._ctr += 1
        self.changed = True
        return pre + stmts


def _mgrid_inline_stmts(tnames: List[str], slices: List[ast.AST], ctr: int) -> Optional[List[ast.stmt]]:
    """``i, j = np.mgrid[a0:b0, a1:b1]`` -> per-axis ``arange`` reshaped onto its
    own axis and broadcast-added to a full-shape int zeros. numba and pythran
    support neither ``np.mgrid``; both support ``arange`` + ``reshape`` +
    broadcasting. ``None`` when a slice has a step / open upper bound."""
    k = len(slices)
    if len(tnames) != k:
        return None
    los, his = [], []
    for sl in slices:
        if not isinstance(sl, ast.Slice) or sl.step is not None or sl.upper is None:
            return None
        los.append("0" if sl.lower is None else f"({ast.unparse(sl.lower)})")
        his.append(f"({ast.unparse(sl.upper)})")
    exts = [f"({his[m]} - {los[m]})" for m in range(k)]
    full = ", ".join(exts)
    lines = []
    for m in range(k):
        rshape = ", ".join(exts[mm] if mm == m else "1" for mm in range(k))
        lines.append(f"{tnames[m]} = np.arange({los[m]}, {his[m]}).reshape({rshape}) + "
                     f"np.zeros(({full},), np.int64)")
    return ast.parse("\n".join(lines)).body


class _MgridInline(ast.NodeTransformer):
    """Replace ``i, j = np.mgrid[s0, s1]`` with explicit ``arange`` broadcasts."""

    def __init__(self):
        self.changed = False
        self._ctr = 0

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        val = node.value
        if not (isinstance(val, ast.Subscript) and isinstance(val.value, ast.Attribute) and val.value.attr == "mgrid"
                and isinstance(val.value.value, ast.Name) and val.value.value.id in ("np", "numpy")):
            return node
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Tuple):
            return node
        elts = node.targets[0].elts
        if not all(isinstance(e, ast.Name) for e in elts):
            return node
        slices = val.slice.elts if isinstance(val.slice, ast.Tuple) else [val.slice]
        stmts = _mgrid_inline_stmts([e.id for e in elts], slices, self._ctr)
        if stmts is None:
            return node
        self._ctr += 1
        self.changed = True
        return stmts


class _FancyGatherHoister(ast.NodeTransformer):
    """Replace each multi-index fancy gather ``A[idx0, idx1, ...]`` (a Tuple index,
    one entry per axis, with >=1 index ARRAY entry) inside one statement with a
    fresh temp Name, accumulating the temp's gather loop in ``self.pre``. numba
    supports a single advanced index ``A[idx]`` but not the multi-index
    (``UniTuple``) point-wise gather -- neither all-1-D (fft_3d's ``u2[q,r,s]``)
    nor mixed 2-D-array + scalar (icon_gather's ``A[nbr[:,:,n]-1, jk,
    blk[:,:,n]-1]``). Array index entries (possibly expressions) are hoisted to
    temps; the driver is the first array entry, scalar axes ride each iteration.
    All array entries must share the driver rank (they broadcast to it)."""

    def __init__(self, ranks: Dict[str, int], ctr: int):
        self.ranks = ranks
        self.ctr = ctr
        self.pre: List[ast.stmt] = []

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        if not (isinstance(node.value, ast.Name) and isinstance(node.ctx, ast.Load)
                and isinstance(node.slice, ast.Tuple)):
            return node
        arr = node.value.id
        arank = self.ranks.get(arr)
        elts = node.slice.elts
        if not arank or len(elts) != arank:
            return node
        elt_ranks = [_expr_rank(e, self.ranks) for e in elts]
        arrs = [r for r in elt_ranks if r and r >= 1]
        if not arrs or any(r != arrs[0] for r in arrs):
            return node  # need >=1 index array; all arrays share the driver rank
        driver_rank = arrs[0]
        p = f"__gather{self.ctr}"
        self.ctr += 1
        iters = [f"{p}_i{k}" for k in range(driver_rank)]
        it = ", ".join(iters)
        pre, idx_exprs, first = [], [], None
        for j, e in enumerate(elts):
            if (elt_ranks[j] or 0) >= 1:
                t = f"{p}_x{j}"
                pre.append(f"{t} = {ast.unparse(e)}")
                idx_exprs.append(f"{t}[{it}]")
                first = first or t
            else:
                idx_exprs.append(ast.unparse(e))
        temp = f"{p}_o"
        lines = pre + [f"{temp} = np.empty({first}.shape, {arr}.dtype)"]
        deepen = ""
        for k in range(driver_rank):
            lines.append(f"{deepen}for {iters[k]} in range({first}.shape[{k}]):")
            deepen += "    "
        lines.append(f"{deepen}{temp}[{it}] = {arr}[{', '.join(idx_exprs)}]")
        self.pre.extend(ast.parse("\n".join(lines)).body)
        return ast.copy_location(ast.Name(id=temp, ctx=ast.Load()), node)


class _FancyGatherInline(ast.NodeTransformer):
    """Hoist multi-array fancy gathers out of any value-bearing statement into a
    preceding gather loop (handles ``chk[i] = np.sum(u2[q, r, s])``)."""

    def __init__(self, ranks: Dict[str, int]):
        self.ranks = ranks
        self.changed = False
        self._ctr = 0

    def _hoist(self, node):
        if getattr(node, "value", None) is None:
            return node
        h = _FancyGatherHoister(self.ranks, self._ctr)
        node.value = h.visit(node.value)
        self._ctr = h.ctr
        if h.pre:
            self.changed = True
            return h.pre + [node]
        return node

    def visit_Assign(self, node):
        return self._hoist(node)

    def visit_AugAssign(self, node):
        return self._hoist(node)

    def visit_Return(self, node):
        return self._hoist(node)

    def visit_Expr(self, node):
        return self._hoist(node)


#: Reductions numba does NOT accept an ``axis=`` kwarg for (unlike ``sum`` /
#: ``prod``, which it supports natively). ``mean`` additionally has no axis form.
_REDUCE_AXIS_OPS = {"sum", "prod", "mean", "std", "var", "min", "max", "amin", "amax", "argmin", "argmax", "any", "all"}


def _const_int(node: ast.AST) -> Optional[int]:
    """A constant integer literal, including a negated one (``axis=-1`` parses as
    ``UnaryOp(USub, Constant(1))``, NOT ``Constant(-1)``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _const_int(node.operand)
        return None if v is None else -v
    return None


def _axis_list(ax: Optional[ast.AST], rank: int) -> Optional[List[int]]:
    """Normalize an ``axis=`` node to a sorted list of non-negative axis indices,
    or None when it is not an all-constant int / tuple of ints. Handles the single
    ``axis=k`` / ``axis=-1`` form and the tuple ``axis=(1, 2, 3)`` form (pooling /
    conv reductions)."""
    if ax is None:
        return None

    def _in_range(v: Optional[int]) -> bool:
        # A valid numpy program's axis is in [-rank, rank); an out-of-range value
        # means our rank estimate is wrong, so bail (leave verbatim) rather than
        # wrap it into an over-reduction.
        return v is not None and -rank <= v < rank

    if isinstance(ax, (ast.Tuple, ast.List)):
        vals = [_const_int(e) for e in ax.elts]
        if not vals or any(not _in_range(v) for v in vals):
            return None
        return sorted({v % rank for v in vals})
    v = _const_int(ax)
    return [v % rank] if _in_range(v) else None


def _reduce_axis_stmts(tname: str,
                       sname: str,
                       op: str,
                       axes: List[int],
                       rank: int,
                       ctr: int,
                       keepdims: bool = False,
                       elem_is_float: bool = False,
                       ddof: int = 0) -> List[ast.stmt]:
    """Source statements reducing ``sname`` over ``axes`` (a sorted list of one or
    more axis indices) into a freshly allocated ``tname`` via an explicit loop nest
    -- the numba/pythran-compatible form of ``np.sum/prod/mean/min/max/argmin/
    argmax(x, axis=..., keepdims=...)`` (numba rejects ``keepdims`` and a tuple
    axis over a >4-D array; pythran rejects ``keepdims``). Output indices iterate
    every non-reduced axis; the reduction iterators run over ``axes`` (nested).
    ``keepdims`` keeps each reduced axis as a size-1 output dim (so the result
    broadcasts back against the input, as softmax's ``x - max`` needs). Mean/var
    divide by the reduced-element count; min/max/arg* compare against a seed."""
    p = f"__rd{ctr}"
    axset = set(axes)
    out_axes = [i for i in range(rank) if i not in axset]
    d = [f"{p}_d{i}" for i in range(rank)]
    lines: List[str] = [f"{d[i]} = {sname}.shape[{i}]" for i in range(rank)]
    is_arg = op in ("argmin", "argmax")
    # ``mean``/``std``/``var`` preserve a FLOAT input's dtype (float32 stays
    # float32); an integer / bool / unknown input upcasts to float64 (numpy's
    # rule). ``{sname}.dtype`` resolves the concrete width at compile time and is
    # numba/pythran-safe (already used by the sum/prod/min/max branch).
    float_res = f"{sname}.dtype" if elem_is_float else "np.float64"
    dtype = ("np.int64" if is_arg else
             "np.bool_" if op in ("any", "all") else float_res if op in ("mean", "std", "var") else f"{sname}.dtype")
    shape_dims = [(d[i] if i in out_axes else "1") for i in range(rank)] if keepdims else [d[i] for i in out_axes]
    lines.append(f"{tname} = np.empty(({''.join(s + ', ' for s in shape_dims)}), {dtype})")
    o = {i: f"{p}_k{i}" for i in out_axes}
    ind = ""
    for i in out_axes:
        lines.append(f"{ind}for {o[i]} in range({d[i]}):")
        ind += "    "
    tgt_idx = [(o[i] if i in out_axes else "0") for i in range(rank)] if keepdims else [o[i] for i in out_axes]
    tgt = f"{tname}[{', '.join(tgt_idx)}]"
    jv = {ax: f"{p}_j{ax}" for ax in axes}
    count = " * ".join(d[ax] for ax in axes)

    def elem(seed: bool = False):
        # index reduced axes with their loop var (or 0 for the comparison seed).
        return f"{sname}[{', '.join(('0' if seed else jv[i]) if i in axset else o[i] for i in range(rank))}]"

    def reduce_loops(base_ind: str, body: List[str]) -> None:
        cur = base_ind
        for ax in axes:
            lines.append(f"{cur}for {jv[ax]} in range({d[ax]}):")
            cur += "    "
        for b in body:
            lines.append(f"{cur}{b}")

    if op in ("any", "all"):
        lines.append(f"{ind}{tgt} = {'True' if op == 'all' else 'False'}")
        if op == "all":
            reduce_loops(ind, [f"if not {elem()}:", f"    {tgt} = False"])
        else:
            reduce_loops(ind, [f"if {elem()}:", f"    {tgt} = True"])
    elif op == "sum":
        lines.append(f"{ind}{tgt} = 0")
        reduce_loops(ind, [f"{tgt} += {elem()}"])
    elif op == "prod":
        lines.append(f"{ind}{tgt} = 1")
        reduce_loops(ind, [f"{tgt} *= {elem()}"])
    elif op == "mean":
        lines.append(f"{ind}{tgt} = 0.0")
        reduce_loops(ind, [f"{tgt} += {elem()}"])
        lines.append(f"{ind}{tgt} = {tgt} / ({count})")
    elif op in ("var", "std"):
        # two-pass: mean (always / N), then sum of squared deviations / (N - ddof)
        # -- matches numpy's np.var/np.std and the C/Fortran sequential lowering.
        # ``ddof`` defaults to 0; np.std(x, axis=k, ddof=1) divides by N-1.
        var_denom = f"({count}) - {ddof}" if ddof else f"({count})"
        m, dv = f"{p}_m", f"{p}_dv"
        lines.append(f"{ind}{m} = 0.0")
        reduce_loops(ind, [f"{m} += {elem()}"])
        lines.append(f"{ind}{m} = {m} / ({count})")
        lines.append(f"{ind}{tgt} = 0.0")
        reduce_loops(ind, [f"{dv} = {elem()} - {m}", f"{tgt} += {dv} * {dv}"])
        lines.append(f"{ind}{tgt} = {tgt} / ({var_denom})")
        if op == "std":
            lines.append(f"{ind}{tgt} = np.sqrt({tgt})")
    elif op in ("min", "amin", "max", "amax"):
        cmp = "<" if op in ("min", "amin") else ">"
        # numpy np.min/np.max PROPAGATE NaN (result is NaN if any element is NaN);
        # a plain `elem cmp tgt` drops it. `elem != elem` captures a NaN element
        # into tgt, and once tgt is NaN no finite element can displace it (every
        # NaN comparison is False), so NaN sticks -- matching the imperative path.
        e = elem()
        lines.append(f"{ind}{tgt} = {elem(seed=True)}")
        reduce_loops(ind, [f"if {e} != {e} or {e} {cmp} {tgt}:", f"    {tgt} = {e}"])
    else:  # argmin / argmax -- single axis (the hoister rejects tuple-axis arg*)
        ax = axes[0]
        cmp = "<" if op == "argmin" else ">"
        best = f"{p}_best"
        e = elem()
        lines.append(f"{ind}{best} = {elem(seed=True)}")
        lines.append(f"{ind}{tgt} = 0")
        lines.append(f"{ind}for {jv[ax]} in range(1, {d[ax]}):")
        # numpy argmin/argmax return the index of the FIRST NaN when one is
        # present. `best == best` is False once best is NaN, locking in that first
        # index; `elem != elem` lets a NaN element win over a finite running best.
        lines.append(f"{ind}    if {best} == {best} and ({e} != {e} or {e} {cmp} {best}):")
        lines.append(f"{ind}        {best} = {e}")
        lines.append(f"{ind}        {tgt} = {jv[ax]}")
    return ast.parse("\n".join(lines)).body


class _ReduceAxisHoister(ast.NodeTransformer):
    """Replace each ``np.mean/min/max/argmin/argmax/any/all(x, axis=<int>)`` --
    OR the method form ``x.mean(axis=<int>)`` (velocity's ``levmask.any(axis=0)``)
    -- inside one statement with a fresh temp Name, accumulating its reduction
    loop in ``self.pre``. A non-Name ``x`` (bellman_ford's ``dist[:, None] +
    graph``) is hoisted to a temp first; a non-constant axis or a rank<2
    (scalar-result) reduction is left verbatim (numba's no-axis scalar form)."""

    def __init__(self, ranks: Dict[str, int], ctr: int, dtypes: Optional[Dict[str, str]] = None):
        self.ranks = ranks
        self.ctr = ctr
        self.dtypes = dtypes or {}
        self.pre: List[ast.stmt] = []

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        kw = {k.arg: k.value for k in node.keywords}
        npop = _np_attr(node)
        if npop in _REDUCE_AXIS_OPS and node.args:  # np.mean(x, axis=k)
            op, arg = npop, node.args[0]
            ax = kw.get("axis") or (node.args[1] if len(node.args) > 1 else None)
        elif (isinstance(node.func, ast.Attribute) and node.func.attr in _REDUCE_AXIS_OPS
              and not (isinstance(node.func.value, ast.Name) and node.func.value.id in ("np", "numpy"))):
            op, arg = node.func.attr, node.func.value  # x.mean(axis=k) method form
            ax = kw.get("axis") or (node.args[0] if node.args else None)
        else:
            return node
        rank = _expr_rank(arg, self.ranks)
        if rank is None or rank < 2:
            return node
        axes = _axis_list(ax, rank)
        if not axes:
            return node
        kd = kw.get("keepdims")
        keepdims = isinstance(kd, ast.Constant) and kd.value is True
        if len(axes) == rank and not keepdims:
            return node  # every axis reduced -> a scalar; leave the backend's full reduction
        if op in ("argmin", "argmax") and len(axes) > 1:
            return node  # numpy itself rejects a tuple axis for argmin/argmax
        if isinstance(arg, ast.Name):
            sname = arg.id
        else:
            sname = f"__rsrc{self.ctr}"
            self.pre.extend(ast.parse(f"{sname} = {ast.unparse(arg)}").body)
        ddof = 0
        if op in ("var", "std"):
            dkw = kw.get("ddof")
            if isinstance(dkw, ast.Constant) and isinstance(dkw.value, int) and not isinstance(dkw.value, bool):
                ddof = dkw.value
            elif dkw is not None:
                return node  # non-constant ddof: cannot fold the divisor, leave verbatim
        temp = f"__rdo{self.ctr}"
        elem_is_float = _dtype_kind(arg, self.dtypes) == "float"
        self.pre.extend(_reduce_axis_stmts(temp, sname, op, axes, rank, self.ctr, keepdims, elem_is_float, ddof))
        self.ctr += 1
        return ast.copy_location(ast.Name(id=temp, ctx=ast.Load()), node)


class _ReduceAxisInline(ast.NodeTransformer):
    """Hoist axis reductions out of any value-bearing statement into preceding
    reduction loops (handles ``V = np.max(s, axis=0) + e`` and the bare
    ``mean = np.mean(data, axis=0)``)."""

    def __init__(self, ranks: Dict[str, int], dtypes: Optional[Dict[str, str]] = None):
        self.ranks = ranks
        self.dtypes = dtypes or {}
        self.changed = False
        self._ctr = 0

    def _hoist(self, node):
        if vars(node).get("value") is None:
            return node
        h = _ReduceAxisHoister(self.ranks, self._ctr, self.dtypes)
        node.value = h.visit(node.value)
        self._ctr = h.ctr
        if h.pre:
            self.changed = True
            return h.pre + [node]
        return node

    def visit_Assign(self, node):
        return self._hoist(node)

    def visit_AugAssign(self, node):
        return self._hoist(node)

    def visit_Return(self, node):
        return self._hoist(node)

    def visit_Expr(self, node):
        return self._hoist(node)


class _CallFixups(ast.NodeTransformer):
    """Small call-form fixups for numba's narrower numpy surface:
    ``np.ndarray(shape, dtype=D)`` -> ``np.empty(shape, D)`` (numba has no
    ``np.ndarray`` constructor); ``np.linspace(a, b, n, dtype=D)`` ->
    ``np.linspace(a, b, n).astype(D)`` (numba's linspace takes no dtype kwarg);
    builtin ``abs(<array>)`` -> ``np.abs(<array>)`` (numba's builtin ``abs``
    types scalars only, not arrays -- mandelbrot's ``abs(Z)`` on complex grids)."""

    def __init__(self, ranks: Dict[str, int]):
        self.ranks = ranks
        self.changed = False

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id == "abs" and len(node.args) == 1 and not node.keywords
                and (_expr_rank(node.args[0], self.ranks) or 0) >= 1):
            self.changed = True
            npabs = ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="abs", ctx=ast.Load())
            return ast.copy_location(ast.Call(func=npabs, args=node.args, keywords=[]), node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "issparse":
            # ``scipy.sparse.issparse(x)`` -> ``False``: the C/Fortran/dace ABI
            # only ever passes DENSE numpy arrays, so the sparse branch is dead
            # (banded_mmt's own comment: "the static dense backends prune this
            # branch"). numba/pythran cannot type scipy.sparse; folding to False
            # lets them dead-code-eliminate it and compile the dense path.
            self.changed = True
            return ast.copy_location(ast.Constant(value=False), node)
        attr = _np_attr(node)
        # numba/pythran want a TUPLE shape, not a list literal: ``np.empty([a, b],
        # ...)`` (a common ML-port idiom, lenet's maxpool) fails to type. Rewrite
        # the shape-carrying list argument to a tuple (data lists -- ``np.array([
        # ...])`` -- are not shape args, so ``array`` is not in this set).
        shape_pos = {"zeros": 0, "ones": 0, "empty": 0, "full": 0, "reshape": 1}.get(attr)
        if shape_pos is not None and len(node.args) > shape_pos and isinstance(node.args[shape_pos], ast.List):
            lst = node.args[shape_pos]
            node.args[shape_pos] = ast.copy_location(ast.Tuple(elts=lst.elts, ctx=ast.Load()), lst)
            self.changed = True
            return node
        if attr in ("zeros", "ones", "empty", "full") and any(k.arg == "order" for k in node.keywords):
            self.changed = True
            node.keywords = [k for k in node.keywords if k.arg != "order"]
            return node
        if attr == "ndarray" and node.args:
            kw = {k.arg: k.value for k in node.keywords}
            dt = kw.get("dtype") or (node.args[1] if len(node.args) > 1 else None)
            self.changed = True
            empty = ast.Attribute(value=node.func.value, attr="empty", ctx=ast.Load())
            return ast.copy_location(
                ast.Call(func=empty, args=[node.args[0]] + ([dt] if dt is not None else []), keywords=[]), node)
        if attr == "linspace":
            kw = {k.arg: k.value for k in node.keywords}
            if "dtype" in kw:
                self.changed = True
                base = ast.Call(func=node.func, args=node.args, keywords=[k for k in node.keywords if k.arg != "dtype"])
                cast = ast.Attribute(value=base, attr="astype", ctx=ast.Load())
                return ast.copy_location(ast.Call(func=cast, args=[kw["dtype"]], keywords=[]), node)
        if attr == "flip" and node.args:
            # np.flip(x[, axis]) -> a reverse-step slice (pythran's np.flip fails
            # type deduction -- durbin); no axis reverses every axis.
            x = node.args[0]
            kw = {k.arg: k.value for k in node.keywords}
            ax = kw.get("axis") or (node.args[1] if len(node.args) > 1 else None)
            rank = _expr_rank(x, self.ranks)
            if rank is None:
                return node
            if ax is None:
                axes = set(range(rank))
            elif isinstance(ax, ast.Constant) and isinstance(ax.value, int):
                axes = {ax.value % rank}
            else:
                return node
            slices = ", ".join("::-1" if d in axes else ":" for d in range(rank))
            self.changed = True
            return ast.copy_location(ast.parse(f"({ast.unparse(x)})[{slices}]", mode="eval").body, node)
        return node


_OUTER_OPS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "true_divide": "/"}


def _ufunc_outer_op(node: ast.AST) -> Optional[str]:
    """``np.<op>.outer(...)`` -> ``<op>`` (add/subtract/multiply/...) else None."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "outer"
            and isinstance(node.func.value, ast.Attribute) and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id in ("np", "numpy")):
        return node.func.value.attr
    return None


class _UfuncOuterHoister(ast.NodeTransformer):
    """Replace ``np.add.outer(a, b)`` (1-D operands) with a reshape+broadcast temp
    (numba has no ufunc.outer): ``a[:,None] op b[None,:]`` as a (len_a, len_b)
    grid. Non-Name operands are unparsed inline into hoisted temps first."""

    def __init__(self, ranks: Dict[str, int], ctr: int):
        self.ranks = ranks
        self.ctr = ctr
        self.pre: List[ast.stmt] = []

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        op = _ufunc_outer_op(node)
        if op not in _OUTER_OPS or len(node.args) != 2:
            return node
        a, b = node.args
        if _expr_rank(a, self.ranks) != 1 or _expr_rank(b, self.ranks) != 1:
            return node  # only the 1-D x 1-D outer grid
        p = f"__ao{self.ctr}"
        self.ctr += 1
        na, nb, sym = f"{p}_a", f"{p}_b", _OUTER_OPS[op]
        # ``.copy()`` -- a strided slice (floyd's ``path[:, k]`` column) is
        # non-contiguous, and numba's reshape requires a contiguous array.
        src = [
            f"{na} = ({ast.unparse(a)}).copy()", f"{nb} = ({ast.unparse(b)}).copy()",
            f"{p} = {na}.reshape({na}.shape[0], 1) {sym} {nb}.reshape(1, {nb}.shape[0])"
        ]
        self.pre.extend(ast.parse("\n".join(src)).body)
        return ast.copy_location(ast.Name(id=p, ctx=ast.Load()), node)


class _UfuncOuterInline(ast.NodeTransformer):
    """Hoist ufunc.outer out of any value-bearing statement (floyd_warshall's
    ``np.minimum(path, np.add.outer(path[:,k], path[k,:]))``)."""

    def __init__(self, ranks: Dict[str, int]):
        self.ranks = ranks
        self.changed = False
        self._ctr = 0

    def _hoist(self, node):
        if getattr(node, "value", None) is None:
            return node
        h = _UfuncOuterHoister(self.ranks, self._ctr)
        node.value = h.visit(node.value)
        self._ctr = h.ctr
        if h.pre:
            self.changed = True
            return h.pre + [node]
        return node

    def visit_Assign(self, node):
        return self._hoist(node)

    def visit_AugAssign(self, node):
        return self._hoist(node)

    def visit_Return(self, node):
        return self._hoist(node)

    def visit_Expr(self, node):
        return self._hoist(node)


class _ScalarizeMask(ast.NodeTransformer):
    """Index every same-shape array reference by the loop iterators ``idx_slice``:
    a masked read ``X[<mask>]`` -> ``X[i, j]`` and a bare full-shape array Name
    ``Z`` -> ``Z[i, j]``. Lower-rank operands / scalars (``horizon``) are left
    alone (they broadcast). Turns a whole-array masked expression into the
    per-element body of a guarded loop."""

    def __init__(self, maskdump: str, idx_slice: ast.AST, arank: int, ranks: Dict[str, int]):
        self.maskdump = maskdump
        self.idx_slice = idx_slice
        self.arank = arank
        self.ranks = ranks

    def _sub(self, value_node: ast.AST) -> ast.Subscript:
        return ast.Subscript(value=value_node, slice=copy.deepcopy(self.idx_slice), ctx=ast.Load())

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.ctx, ast.Load) and ast.dump(node.slice) == self.maskdump:
            return self._sub(node.value)  # X[mask] -> X[idx]; do not recurse into it
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and self.ranks.get(node.id) == self.arank:
            return self._sub(node)
        return node


class _MaskedAssignToLoop(ast.NodeTransformer):
    """``T[mask] = rhs`` -> a guarded loop nest ``for i,j: if mask[i,j]:
    T[i,j] = rhs[i,j]``. numba rejects multi-dimensional boolean-mask indexing
    (``r2inv[in_range]``, mandelbrot's ``Z[abs(Z) < h]``). A loop -- NOT
    ``np.where`` -- because the masked form computes the RHS ONLY on selected
    elements: mandelbrot freezes diverged points precisely so the squared term
    never overflows, and force_lj divides only where ``rsq > 0``. ``np.where``
    would evaluate the RHS everywhere and change those results. Restricted to a
    >=2-D mask that is a bool-array Name of the target's rank or an inline
    Compare / ``& | ^ ~`` logical combo of that rank. A same-rank INTEGER index
    Name is a fancy index, not a mask, so it is left verbatim (a clean skip)."""

    def __init__(self, ranks: Dict[str, int], dtypes: Dict[str, str]):
        self.ranks = ranks
        self.dtypes = dtypes
        self.changed = False
        self._ctr = 0

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and isinstance(tgt.ctx, ast.Store)):
            return node
        idx = tgt.slice
        arank = self.ranks.get(tgt.value.id)
        if not arank or arank < 2 or isinstance(idx, (ast.Tuple, ast.Slice)):
            return node
        struct_mask = (isinstance(idx, (ast.Compare, ast.BoolOp))
                       or (isinstance(idx, ast.BinOp) and isinstance(idx.op, (ast.BitAnd, ast.BitOr, ast.BitXor)))
                       or (isinstance(idx, ast.UnaryOp) and isinstance(idx.op, ast.Invert)))
        if isinstance(idx, ast.Name):
            # A full-shape index Name is a mask ONLY if boolean-kind; a same-rank
            # integer array is a fancy index (different semantics) -> leave verbatim.
            if self.ranks.get(idx.id) != arank or _dtype_kind(idx, self.dtypes) in ("int", "float", "complex"):
                return node
        elif struct_mask:
            if _expr_rank(idx, self.ranks) != arank:
                return node
        else:
            return node
        T = tgt.value.id
        p = f"__mi{self._ctr}"
        self._ctr += 1
        idx_vars = [f"{p}_{k}" for k in range(arank)]
        idx_slice = ast.parse(f"_x[{', '.join(idx_vars)}]", mode="eval").body.slice
        scal = _ScalarizeMask(ast.dump(idx), idx_slice, arank, self.ranks)
        mask_s = ast.unparse(scal.visit(copy.deepcopy(idx)))
        rhs_s = ast.unparse(
            _ScalarizeMask(ast.dump(idx), idx_slice, arank, self.ranks).visit(copy.deepcopy(node.value)))
        lines, deepen = [], ""
        for k in range(arank):
            lines.append(f"{deepen}for {idx_vars[k]} in range({T}.shape[{k}]):")
            deepen += "    "
        lines.append(f"{deepen}if {mask_s}:")
        lines.append(f"{deepen}    {T}[{', '.join(idx_vars)}] = {rhs_s}")
        self.changed = True
        return [ast.copy_location(s, node) for s in ast.parse("\n".join(lines)).body]


#: masked-gather reductions this lowers (a boolean-mask select feeding a full
#: reduction). ``mean`` matches numpy's mean-of-empty -> nan; extend as needed.
_MASKED_REDUCE_OPS = {"mean"}


def _masked_reduce_of(node: ast.AST, gathers: Dict[str, tuple]):
    """A supported reduction of a masked-gather name in ``gathers`` -> ``(op,
    name)``, else None. Handles the method form ``v.mean()`` and the function
    form ``np.mean(v)`` (v the whole operand -- a full reduction, no axis)."""
    if not isinstance(node, ast.Call) or node.keywords:
        return None
    f = node.func
    if (isinstance(f, ast.Attribute) and f.attr in _MASKED_REDUCE_OPS and not node.args
            and isinstance(f.value, ast.Name) and f.value.id in gathers):
        return f.attr, f.value.id
    if (_np_attr(node) in _MASKED_REDUCE_OPS and len(node.args) == 1 and isinstance(node.args[0], ast.Name)
            and node.args[0].id in gathers):
        return _np_attr(node), node.args[0].id
    return None


def _masked_reduce_lines(temp: str, a: str, mask: str, rank: int, op: str, p: str) -> List[str]:
    """Source lines reducing the boolean-masked selection ``a[mask]`` into scalar
    ``temp`` via an accumulate loop -- the numba/pythran/dace-compatible form of
    ``a[mask].mean()`` (the masked select alone is a dynamic-length array pythran
    cannot type and dace cannot shape). ``mean`` divides sum by the masked count
    and yields ``np.nan`` for an empty selection (numpy's mean-of-empty)."""
    idx = ", ".join(f"{p}_i{d}" for d in range(rank))
    lines = [f"{p}_s = 0.0", f"{p}_n = 0"]
    deep = ""
    for d in range(rank):
        lines.append(f"{deep}for {p}_i{d} in range({a}.shape[{d}]):")
        deep += "    "
    lines += [f"{deep}if {mask}[{idx}]:", f"{deep}    {p}_s += {a}[{idx}]", f"{deep}    {p}_n += 1"]
    # op == "mean" (the only supported reduction). Empty selection -> nan (numpy).
    lines += [f"if {p}_n > 0:", f"    {temp} = {p}_s / {p}_n", "else:", f"    {temp} = np.nan"]
    return lines


def _is_bool_mask(mask: ast.AST, a: ast.AST, ranks: Dict[str, int], dtypes: Dict[str, str]) -> bool:
    """True iff ``mask`` is a boolean array of ``a``'s rank -- a bool-kind Name or
    an inline Compare / BoolOp / logical_* combo -- i.e. ``a[mask]`` is a boolean
    select (not an integer fancy index or a scalar/slice index)."""
    if isinstance(mask, (ast.Tuple, ast.Slice)) or _is_newaxis(mask):
        return False
    ar = _expr_rank(a, ranks)
    if ar is None or _expr_rank(mask, ranks) != ar:
        return False
    return _dtype_kind(mask, dtypes) == "bool"


def _masked_reduce_map(fn: ast.AST, ranks: Dict[str, int], dtypes: Dict[str, str]) -> Dict[str, tuple]:
    """``{name: (a_Name, mask_ast)}`` for every ``name = a[mask]`` boolean-select
    whose EVERY load-use is a supported masked reduction -- so the select can be
    dropped and each reduction inlined as an accumulate loop. A name used any other
    way (indexed, returned, passed on) is excluded and left verbatim."""
    gathers: Dict[str, tuple] = {}
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name)
                and _is_bool_mask(node.value.slice, node.value.value, ranks, dtypes)):
            gathers[node.targets[0].id] = (node.value.value, node.value.slice)
    ok: Dict[str, tuple] = {}
    for v, am in gathers.items():
        loads = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == v and isinstance(n.ctx, ast.Load))
        reds = sum(1 for n in ast.walk(fn) if _masked_reduce_of(n, {v: am}) is not None)
        if reds and reds == loads:
            ok[v] = am
    return ok


class _MaskedReduceHoister(ast.NodeTransformer):
    """Replace each ``v.mean()`` / ``np.mean(v)`` (v a lowerable masked-gather
    name) inside one statement with a fresh temp Name, emitting its accumulate
    loop into ``self.pre``."""

    def __init__(self, gathers: Dict[str, tuple], ranks: Dict[str, int], ctr: int):
        self.gathers = gathers
        self.ranks = ranks
        self.ctr = ctr
        self.pre: List[ast.stmt] = []

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        hit = _masked_reduce_of(node, self.gathers)
        if hit is None:
            return node
        op, v = hit
        a, mask = self.gathers[v]
        p = f"__mr{self.ctr}"
        self.ctr += 1
        temp = f"{p}_o"
        lines = _masked_reduce_lines(temp, a.id, ast.unparse(mask), _expr_rank(a, self.ranks), op, p)
        self.pre.extend(ast.parse("\n".join(lines)).body)
        return ast.copy_location(ast.Name(id=temp, ctx=ast.Load()), node)


class _MaskedReduceInline(ast.NodeTransformer):
    """Drop each lowerable ``v = a[mask]`` boolean-select and inline its reductions
    (``res[i] = v.mean()`` -> accumulate loop) -- azimint_naive's ``values =
    data[mask]; res[i] = values.mean()``. The masked select is a dynamic-length
    array pythran cannot type (auto-before-deduction) and dace cannot shape; numba
    would DCE the now-unused select anyway. ``gathers`` is pre-vetted so every use
    of the name is a reduction, making the drop safe."""

    def __init__(self, gathers: Dict[str, tuple], ranks: Dict[str, int]):
        self.gathers = gathers
        self.ranks = ranks
        self.changed = False
        self._ctr = 0

    def _hoist(self, node):
        if node.value is None:
            return node
        h = _MaskedReduceHoister(self.gathers, self.ranks, self._ctr)
        node.value = h.visit(node.value)
        self._ctr = h.ctr
        if h.pre:
            self.changed = True
            return h.pre + [node]
        return node

    def visit_Assign(self, node: ast.Assign):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.gathers
                and isinstance(node.value, ast.Subscript)):
            self.changed = True
            return []  # drop the masked select; each reduction inlines its own loop
        return self._hoist(node)

    def visit_AugAssign(self, node):
        return self._hoist(node)

    def visit_Return(self, node):
        return self._hoist(node)

    def visit_Expr(self, node):
        return self._hoist(node)


_AT_OPS = {"add": "+=", "subtract": "-=", "multiply": "*="}


def _ufunc_at_op(node: ast.AST) -> Optional[str]:
    """``np.add.at(...)`` / ``np.subtract.at`` / ``np.multiply.at`` -> the op."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "at"
            and isinstance(node.func.value, ast.Attribute) and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id in ("np", "numpy")):
        return node.func.value.attr
    return None


class _AddAtInline(ast.NodeTransformer):
    """``np.add.at(A, idx, vals)`` -> an explicit scatter loop. numba has no
    ufunc.at; a sequential ``+=`` loop reproduces its defining property --
    duplicate indices accumulate (unlike ``A[idx] += vals``). Handles a single
    1-D index (edge_laplacian's ``np.add.at(Lx, src, flux)``) and a tuple of
    index arrays + scalar axes (icon_scatter's ``np.add.at(out, (i2d, jk, j2d),
    val)``); the driver is the first index array, scalars ride each iteration."""

    def __init__(self, ranks: Dict[str, int]):
        self.ranks = ranks
        self.changed = False
        self._ctr = 0

    def visit_Expr(self, node: ast.Expr):
        self.generic_visit(node)
        call = node.value
        op = _ufunc_at_op(call) if isinstance(call, ast.Call) else None
        if op not in _AT_OPS or len(call.args) < 2 or not isinstance(call.args[0], ast.Name):
            return node
        A = call.args[0].id
        elts = call.args[1].elts if isinstance(call.args[1], ast.Tuple) else [call.args[1]]
        vals = call.args[2] if len(call.args) > 2 else None
        driver_rank = next((r for e in elts if (r := _expr_rank(e, self.ranks)) and r >= 1), None)
        if driver_rank is None:
            return node
        p = f"__sc{self._ctr}"
        self._ctr += 1
        iters = [f"{p}_i{k}" for k in range(driver_rank)]
        it = ", ".join(iters)
        # ``np.ascontiguousarray`` materialises each hoisted index / value array:
        # pythran keeps ``nbr_idx[:, :, n] - 1`` as a lazy numpy_expr that cannot
        # be indexed by a tuple in the scatter loop (icon_scatter); it is a no-op
        # for an already-contiguous array under numba.
        pre, idx_exprs, first_arr = [], [], None
        for j, e in enumerate(elts):
            if (_expr_rank(e, self.ranks) or 0) >= 1:
                t = f"{p}_x{j}"
                pre.append(f"{t} = np.ascontiguousarray({ast.unparse(e)})")
                idx_exprs.append(f"{t}[{it}]")
                first_arr = first_arr or t
            else:
                idx_exprs.append(ast.unparse(e))
        if vals is None:
            rhs = "1"
        else:
            tv = f"{p}_v"
            pre.append(f"{tv} = np.ascontiguousarray({ast.unparse(vals)})")
            vr = _expr_rank(vals, self.ranks)
            if vr and vr not in (0, driver_rank):
                # vals broadcasts against the index shape; only a scalar or a
                # driver-shaped vals is unambiguous. Anything else would need
                # numpy broadcast alignment we do not model -> fail loudly.
                raise DesugarError(f"np.add.at values ndim {vr} != index ndim {driver_rank} "
                                   "(only scalar or matching-shape values are lowered)")
            rhs = f"{tv}[{it}]" if vr and vr >= 1 else tv
        lines, deepen = list(pre), ""
        for k in range(driver_rank):
            lines.append(f"{deepen}for {iters[k]} in range({first_arr}.shape[{k}]):")
            deepen += "    "
        lines.append(f"{deepen}{A}[{', '.join(idx_exprs)}] {_AT_OPS[op]} {rhs}")
        self.changed = True
        return [ast.copy_location(s, node) for s in ast.parse("\n".join(lines)).body]


class _HistogramHoister(ast.NodeTransformer):
    """Replace ``np.histogram(a, bins[, lo, hi][, weights=w])[0]`` with a fresh
    temp Name, emitting the numpy-histogram loop into ``self.pre``: a min/max scan
    for the default range, then per-element binning ``b = int((a-lo)*bins/(hi-lo))``
    clamped to ``[0, bins-1]`` accumulating ``1`` (or ``w[i]``). numba has no
    np.histogram; this is the same loop the C/Fortran backends lower (azimint_hist)."""

    def __init__(self, ctr: int):
        self.ctr = ctr
        self.pre: List[ast.stmt] = []

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        if not (isinstance(node.slice, ast.Constant) and node.slice.value == 0 and _np_attr(node.value) == "histogram"
                and len(node.value.args) >= 2):
            return node
        call = node.value
        a, bins = ast.unparse(call.args[0]), ast.unparse(call.args[1])
        kw = {k.arg: k.value for k in call.keywords}
        lo = hi = None
        if len(call.args) >= 4:
            lo, hi = call.args[2], call.args[3]
        rng = kw.get("range")
        if isinstance(rng, ast.Tuple) and len(rng.elts) == 2:
            lo, hi = rng.elts
        weights = kw.get("weights")
        p = f"__hist{self.ctr}"
        self.ctr += 1
        lines = []
        if lo is None or hi is None:
            lo_s, hi_s = f"{p}_lo", f"{p}_hi"
            lines += [
                f"{lo_s} = {a}[0]", f"{hi_s} = {a}[0]", f"for {p}_s in range({a}.shape[0]):",
                f"    if {a}[{p}_s] < {lo_s}: {lo_s} = {a}[{p}_s]", f"    if {a}[{p}_s] > {hi_s}: {hi_s} = {a}[{p}_s]"
            ]
        else:
            lo_s, hi_s = f"({ast.unparse(lo)})", f"({ast.unparse(hi)})"
        temp = f"{p}_o"
        add = f"{ast.unparse(weights)}[{p}_i]" if weights is not None else "1.0"
        lines += [
            f"{temp} = np.zeros({bins}, np.float64)", f"for {p}_i in range({a}.shape[0]):",
            f"    {p}_b = int(({a}[{p}_i] - {lo_s}) * {bins} / ({hi_s} - {lo_s}))", f"    if {p}_b < 0: {p}_b = 0",
            f"    if {p}_b > {bins} - 1: {p}_b = {bins} - 1", f"    {temp}[{p}_b] += {add}"
        ]
        self.pre.extend(ast.parse("\n".join(lines)).body)
        return ast.copy_location(ast.Name(id=temp, ctx=ast.Load()), node)


#: binary arithmetic ufuncs whose ``out=`` form maps to a plain BinOp.
_UFUNC_OUT_OPS = {
    "add": ast.Add,
    "subtract": ast.Sub,
    "multiply": ast.Mult,
    "divide": ast.Div,
    "true_divide": ast.Div,
    "power": ast.Pow,
    "floor_divide": ast.FloorDiv,
    "remainder": ast.Mod,
    "mod": ast.Mod,
}


class _UfuncOutInline(ast.NodeTransformer):
    """``np.multiply(a, b, out=c)`` (and the other binary arithmetic ufuncs) -> the
    explicit assignment ``c = a <op> b``. The C/Fortran backends have no ufunc
    dispatch, so the ``out=`` form must be lowered to a store (minife's axpby).
    ``c`` may be a slice (``wcoefs[:n]``) -- the assignment target is that slice."""

    def _rewrite(self, call: ast.AST):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id in ("np", "numpy")):
            return None
        op = _UFUNC_OUT_OPS.get(call.func.attr)
        if op is None or len(call.args) != 2:
            return None
        out = next((kw.value for kw in call.keywords if kw.arg == "out"), None)
        if out is None:
            return None
        target = copy.deepcopy(out)
        for n in ast.walk(target):
            if isinstance(n, (ast.Name, ast.Subscript, ast.Attribute)):
                n.ctx = ast.Store()
        return ast.copy_location(
            ast.Assign(targets=[target], value=ast.BinOp(left=call.args[0], op=op(), right=call.args[1])), call)

    def visit_Expr(self, node: ast.Expr):
        rw = self._rewrite(node.value)
        return rw if rw is not None else node


class _HistogramInline(ast.NodeTransformer):
    """Hoist ``np.histogram(...)[0]`` out of any value-bearing statement into its
    preceding binning loop (azimint's ``histw = np.histogram(r, n, weights=d)[0]``)."""

    def __init__(self, ranks: Dict[str, int]):
        self.changed = False
        self._ctr = 0

    def _hoist(self, node):
        if getattr(node, "value", None) is None:
            return node
        h = _HistogramHoister(self._ctr)
        node.value = h.visit(node.value)
        self._ctr = h.ctr
        if h.pre:
            self.changed = True
            return h.pre + [node]
        return node

    def visit_Assign(self, node):
        return self._hoist(node)

    def visit_AugAssign(self, node):
        return self._hoist(node)

    def visit_Return(self, node):
        return self._hoist(node)

    def visit_Expr(self, node):
        return self._hoist(node)


def _int_matmul_stmts(temp: str, a: str, b: str, ra: int, rb: int, ctr: int) -> List[str]:
    """Source lines for an INTEGER matmul (``a @ b``) as an explicit loop.
    An int64 accumulator holds exact integer sums (numba's BLAS-backed ``@`` is
    float-only). Raises for ranks numba could not express even after batching."""
    p = f"__mm{ctr}"
    if ra == 1 and rb == 1:  # dot -> scalar
        return [f"{temp} = 0", f"for {p}_k in range({a}.shape[0]):", f"    {temp} += {a}[{p}_k] * {b}[{p}_k]"]
    if ra == 1 and rb == 2:  # (K,) @ (K, N) -> (N,)
        return [
            f"{temp} = np.zeros({b}.shape[1], np.int64)", f"for {p}_j in range({b}.shape[1]):",
            f"    for {p}_k in range({a}.shape[0]):", f"        {temp}[{p}_j] += {a}[{p}_k] * {b}[{p}_k, {p}_j]"
        ]
    if ra == 2 and rb == 1:  # (M, K) @ (K,) -> (M,)
        return [
            f"{temp} = np.zeros({a}.shape[0], np.int64)", f"for {p}_i in range({a}.shape[0]):",
            f"    for {p}_k in range({a}.shape[1]):", f"        {temp}[{p}_i] += {a}[{p}_i, {p}_k] * {b}[{p}_k]"
        ]
    if ra == 2 and rb == 2:  # (M, K) @ (K, N) -> (M, N)
        return [
            f"{temp} = np.zeros(({a}.shape[0], {b}.shape[1]), np.int64)", f"for {p}_i in range({a}.shape[0]):",
            f"    for {p}_j in range({b}.shape[1]):", f"        for {p}_k in range({a}.shape[1]):",
            f"            {temp}[{p}_i, {p}_j] += {a}[{p}_i, {p}_k] * {b}[{p}_k, {p}_j]"
        ]
    raise DesugarError(f"integer matmul of ranks {ra}x{rb} is unsupported "
                       "(numba has no integer @; only <=2-D operands are lowered)")


class _IntMatmulHoister(ast.NodeTransformer):
    """Replace each INTEGER ``a @ b`` / ``np.matmul`` / ``np.dot`` (both operands
    integer/bool kind) with a temp Name, emitting an explicit loop into
    ``self.pre``. numba's ``@`` is BLAS-backed and float-only, so int matmul
    (bfs's ``frontier @ graph``) fails to type; float matmul is LEFT for numba's
    fast path. Owned-but-unhandled shapes (unknown rank, >2-D) raise DesugarError."""

    def __init__(self, ranks: Dict[str, int], dtypes: Dict[str, str], ctr: int):
        self.ranks = ranks
        self.dtypes = dtypes
        self.ctr = ctr
        self.pre: List[ast.stmt] = []

    def _lower(self, a: ast.expr, b: ast.expr):
        if _dtype_kind(a, self.dtypes) not in ("int", "bool") or _dtype_kind(b, self.dtypes) not in ("int", "bool"):
            return None  # not (definitely) an integer matmul -> leave for numba's float @
        ra, rb = _expr_rank(a, self.ranks), _expr_rank(b, self.ranks)
        if ra is None or rb is None:
            return None  # can't determine the shape -> leave verbatim (a clean skip),
            # NOT a raise: an unknown rank is an inference gap, not a known-unsupported shape.
        p = f"__mmi{self.ctr}"
        pre = []
        aid = a.id if isinstance(a, ast.Name) else f"{p}_a"
        bid = b.id if isinstance(b, ast.Name) else f"{p}_b"
        if not isinstance(a, ast.Name):
            pre.append(f"{aid} = {ast.unparse(a)}")
        if not isinstance(b, ast.Name):
            pre.append(f"{bid} = {ast.unparse(b)}")
        temp = f"{p}_o"
        self.pre.extend(ast.parse("\n".join(pre + _int_matmul_stmts(temp, aid, bid, ra, rb, self.ctr))).body)
        self.ctr += 1
        return ast.Name(id=temp, ctx=ast.Load())

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if isinstance(node.op, ast.MatMult):
            rep = self._lower(node.left, node.right)
            if rep is not None:
                return ast.copy_location(rep, node)
        return node

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if _np_attr(node) in ("matmul", "dot") and len(node.args) == 2:
            rep = self._lower(node.args[0], node.args[1])
            if rep is not None:
                return ast.copy_location(rep, node)
        return node


class _IntMatmulInline(ast.NodeTransformer):
    """Hoist integer matmuls out of any value-bearing statement (bfs's
    ``reach = frontier @ graph``)."""

    def __init__(self, ranks: Dict[str, int], dtypes: Dict[str, str]):
        self.ranks = ranks
        self.dtypes = dtypes
        self.changed = False
        self._ctr = 0

    def _hoist(self, node):
        if getattr(node, "value", None) is None:
            return node
        h = _IntMatmulHoister(self.ranks, self.dtypes, self._ctr)
        node.value = h.visit(node.value)
        self._ctr = h.ctr
        if h.pre:
            self.changed = True
            return h.pre + [node]
        return node

    def visit_Assign(self, node):
        return self._hoist(node)

    def visit_AugAssign(self, node):
        return self._hoist(node)

    def visit_Return(self, node):
        return self._hoist(node)

    def visit_Expr(self, node):
        return self._hoist(node)


def _is_transpose_expr(v: ast.AST) -> bool:
    """``np.transpose(x, ...)`` / ``x.transpose(...)`` / ``x.T`` -- these produce a
    non-contiguous view."""
    return (_np_attr(v) == "transpose" or (isinstance(v, ast.Attribute) and v.attr == "T")
            or (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "transpose"))


def _noncontig_names(tree: ast.AST) -> set:
    """Names bound to a non-contiguous view (a transpose, or a transpose chained
    through another such name) -- to a fixpoint."""
    nc: set = set()
    for _ in range(6):
        grew = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
                v = node.value
                bad = _is_transpose_expr(v) or (isinstance(v, ast.Name) and v.id in nc)
                if bad and node.targets[0].id not in nc:
                    nc.add(node.targets[0].id)
                    grew = True
        if not grew:
            break
    return nc


class _ReshapeContiguousInline(ast.NodeTransformer):
    """Wrap a reshape's array operand in ``np.ascontiguousarray`` when it is
    non-contiguous (a transpose or a transpose-derived name) -- numba's reshape
    requires a contiguous array (stockham's ``np.reshape(tmp_perm, (N,))`` where
    ``tmp_perm = np.transpose(yv, ...)``). A no-op for already-contiguous inputs."""

    def __init__(self, noncontig: set):
        self.noncontig = noncontig
        self.changed = False

    def _noncontig(self, x: ast.AST) -> bool:
        return (isinstance(x, ast.Name) and x.id in self.noncontig) or _is_transpose_expr(x)

    def _wrap(self, x: ast.AST) -> ast.Call:
        self.changed = True
        acont = ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="ascontiguousarray", ctx=ast.Load())
        return ast.Call(func=acont, args=[x], keywords=[])

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if _np_attr(node) == "reshape" and node.args and self._noncontig(node.args[0]):
            node.args[0] = self._wrap(node.args[0])
        elif (isinstance(node.func, ast.Attribute) and node.func.attr == "reshape"
              and self._noncontig(node.func.value)):
            node.func.value = self._wrap(node.func.value)
        return node


class _RepeatAxisHoister(ast.NodeTransformer):
    """Replace ``np.repeat(x, m, axis=k)`` (constant axis, scalar count) with a
    temp Name, emitting the gather loop ``out[..., j, ...] = x[..., j // m, ...]``
    into ``self.pre`` (numpy repeats each slice ``m`` times consecutively along
    ``axis``). numba rejects the ``axis=`` kwarg on np.repeat (stockham's
    ``np.repeat(reshape(tmp, (R, R**i, 1)), R**(K-i-1), axis=2)``)."""

    def __init__(self, ranks: Dict[str, int], ctr: int):
        self.ranks = ranks
        self.ctr = ctr
        self.pre: List[ast.stmt] = []

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if _np_attr(node) != "repeat" or len(node.args) < 2:
            return node
        kw = {k.arg: k.value for k in node.keywords}
        ax = kw.get("axis") or (node.args[2] if len(node.args) > 2 else None)
        if not (isinstance(ax, ast.Constant) and isinstance(ax.value, int)):
            return node  # no axis / non-constant -> leave verbatim (a clean skip)
        x, m = node.args[0], node.args[1]
        rank = _expr_rank(x, self.ranks)
        if rank is None:
            return node
        k = ax.value % rank
        p = f"__rp{self.ctr}"
        self.ctr += 1
        pre = []
        if isinstance(x, ast.Name):
            xid = x.id
        else:
            xid = f"{p}_x"
            pre.append(f"{xid} = {ast.unparse(x)}")
        ms = f"({ast.unparse(m)})"
        dims = [(f"{xid}.shape[{d}] * {ms}" if d == k else f"{xid}.shape[{d}]") for d in range(rank)]
        iters = [f"{p}_i{d}" for d in range(rank)]
        out = f"{p}_o"
        lines = pre + [f"{out} = np.empty(({', '.join(dims)},), {xid}.dtype)"]
        deep = ""
        for d in range(rank):
            lines.append(f"{deep}for {iters[d]} in range({dims[d]}):")
            deep += "    "
        src_idx = ", ".join((f"{iters[k]} // {ms}" if d == k else iters[d]) for d in range(rank))
        lines.append(f"{deep}{out}[{', '.join(iters)}] = {xid}[{src_idx}]")
        self.pre.extend(ast.parse("\n".join(lines)).body)
        return ast.copy_location(ast.Name(id=out, ctx=ast.Load()), node)


class _RepeatAxisInline(ast.NodeTransformer):
    """Hoist ``np.repeat(..., axis=k)`` out of any value-bearing statement."""

    def __init__(self, ranks: Dict[str, int]):
        self.ranks = ranks
        self.changed = False
        self._ctr = 0

    def _hoist(self, node):
        if getattr(node, "value", None) is None:
            return node
        h = _RepeatAxisHoister(self.ranks, self._ctr)
        node.value = h.visit(node.value)
        self._ctr = h.ctr
        if h.pre:
            self.changed = True
            return h.pre + [node]
        return node

    def visit_Assign(self, node):
        return self._hoist(node)

    def visit_AugAssign(self, node):
        return self._hoist(node)

    def visit_Return(self, node):
        return self._hoist(node)

    def visit_Expr(self, node):
        return self._hoist(node)


#: numpy ops whose RESULT keeps the operand's dimensionality, so a negative
#: ``axis=`` counts back from the operand's own rank (``flip``/``roll``/``cumsum``
#: /``concatenate``/... are all axis-preserving).
_AXIS_PRESERVING_OPS = {
    "flip", "roll", "cumsum", "cumprod", "nancumsum", "nancumprod", "sort", "argsort", "concatenate", "diff", "gradient"
}
#: ops that ADD one axis, so the axis-space is the operand rank PLUS one
#: (``np.stack((a, b), axis=-1)`` on rank-2 operands addresses axis 2).
_AXIS_ADDING_OPS = {"stack", "expand_dims"}


class _NormalizeNegativeAxis(ast.NodeTransformer):
    """Rewrite a NEGATIVE ``axis=`` literal to the positive index it denotes --
    ``np.flip(a, axis=-1)`` -> ``np.flip(a, axis=1)`` for a rank-2 ``a``.

    numpy counts a negative axis back from the last (``-1`` == ``rank - 1``), but
    some backend runtimes mishandle it: pythran's ``np.flip`` / ``np.stack`` with
    ``axis=-1`` silently return the wrong result. The native C/Fortran lowering
    and the reduction desugar already normalize the axis, so only the
    verbatim-body python backends (pythran especially) are affected; rewriting the
    literal to ``rank + axis`` makes every backend agree.

    The axis-space rank is the first array operand's rank for an axis-preserving
    op and that rank + 1 for an axis-ADDING op (``stack``/``expand_dims``, whose
    result has one more dimension than the operand). A negative axis whose op or
    operand rank cannot be determined is left verbatim (never guessed). Only the
    ``axis=`` keyword form is normalized -- a positional axis position differs per
    op (``np.roll``'s second positional arg is the shift, not the axis)."""

    def __init__(self, ranks: Dict[str, int]):
        self.ranks = ranks
        self.changed = False

    def _operand_rank(self, node: ast.Call) -> Optional[int]:
        # The first positional operand carries the rank; for stack/expand_dims it
        # is the SEQUENCE being stacked (a tuple/list), so take its first element.
        if not node.args:
            return None
        a0 = node.args[0]
        if isinstance(a0, (ast.Tuple, ast.List)):
            return _expr_rank(a0.elts[0], self.ranks) if a0.elts else None
        return _expr_rank(a0, self.ranks)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        op = _np_attr(node)
        adding = op in _AXIS_ADDING_OPS
        if not adding and op not in _AXIS_PRESERVING_OPS:
            return node
        kw = next((k for k in node.keywords if k.arg == "axis"), None)
        if kw is None:
            return node
        val = _const_int(kw.value)
        if val is None or val >= 0:
            return node
        base = self._operand_rank(node)
        if base is None:
            return node
        pos = base + (1 if adding else 0) + val  # val < 0
        if pos < 0:
            return node  # rank estimate off -> leave verbatim rather than wrap wrong
        kw.value = ast.copy_location(ast.Constant(value=pos), kw.value)
        self.changed = True
        return node


class _DropGuards(ast.NodeTransformer):
    """Replace ``raise ...`` / ``assert ...`` statements with ``pass``. These are
    input-validation guards (``if bad: raise ValueError(f"...")``); OptArena kernels
    run on oracle-validated inputs so the guard never fires. Dropping them also
    removes the f-string messages pythran cannot parse and the exception types
    numba/pythran/dace need not express. ``pass`` (not deletion) keeps an
    otherwise-empty ``if`` body syntactically valid."""

    def __init__(self):
        self.changed = False

    def visit_Raise(self, node: ast.Raise):
        self.changed = True
        return ast.copy_location(ast.Pass(), node)

    def visit_Assert(self, node: ast.Assert):
        self.changed = True
        return ast.copy_location(ast.Pass(), node)


class _DropValidationGuards(ast.NodeTransformer):
    """Remove an input-validation guard ``if <cond>: raise/assert`` ENTIRELY -- the
    condition included -- so a ``.ndim`` / ``.flags.c_contiguous`` / ``.dtype`` check
    the native backends cannot emit disappears with the guard, not just the raise
    (which the emitter already skips, leaving the unemittable condition behind).
    Fires only when the whole if-body is raise/assert/pass and there is no else, so a
    real branch is never touched. OptArena kernels run on oracle-validated inputs, so
    the guard never fires (minife's ``_require_float_vector`` rank/contiguity checks)."""

    def visit_If(self, node: ast.If):
        self.generic_visit(node)
        if (not node.orelse and node.body and all(isinstance(s, (ast.Raise, ast.Assert, ast.Pass)) for s in node.body)):
            return None
        return node










def _strip_docstring_stmts(body: List[ast.stmt]) -> List[ast.stmt]:
    return [
        s for s in body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))
    ]


#: ``np.<name>`` abstract dtype category -> the concrete dtype KINDS it covers.
#: Used to fold ``np.issubdtype(x.dtype, np.<name>)`` to a compile-time bool.
_ISSUBDTYPE_CATEGORY: Dict[str, set] = {
    "integer": {"int"},
    "signedinteger": {"int"},
    "unsignedinteger": {"int"},
    "floating": {"float"},
    "complexfloating": {"complex"},
    "inexact": {"float", "complex"},
    "number": {"int", "float", "complex"},
    "bool_": {"bool"},
    "bool": {"bool"},
}


class _IssubdtypeFold(ast.NodeTransformer):
    """Fold ``np.issubdtype(<expr>.dtype, np.<category>)`` -- and a bare concrete
    ``np.issubdtype(np.int32, np.integer)`` -- to a ``True``/``False`` constant from
    the known dtype KIND of the operand (bool/int/float/complex). numba/pythran/dace
    cannot evaluate ``np.issubdtype``, but the answer is a compile-time property of
    the (statically known) dtype, so the guard/branch it feeds resolves and (with
    dead-branch elim) disappears -- the isinstance-style check the C++/C backends
    would do. Left verbatim when the operand kind or the category is unknown."""

    def __init__(self, dtypes: Dict[str, str]):
        self.dtypes = dtypes
        self.changed = False

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if _np_attr(node) != "issubdtype" or len(node.args) != 2:
            return node
        a = node.args[0]
        kind = _dtype_kind(a.value, self.dtypes) if (isinstance(a, ast.Attribute)
                                                     and a.attr == "dtype") else _dtype_arg_kind(a)
        cat = node.args[1]
        catname = cat.attr if isinstance(cat, ast.Attribute) else (cat.id if isinstance(cat, ast.Name) else None)
        kinds = _ISSUBDTYPE_CATEGORY.get(catname)
        if kind is None or kinds is None:
            return node
        self.changed = True
        return ast.copy_location(ast.Constant(value=(kind in kinds)), node)


class _DeadBranchElim(ast.NodeTransformer):
    """Constant-fold boolean guards (``X and False`` -> ``False``, etc.) and drop
    the unreachable branch of ``if <const bool>:``. After the desugar folds a
    ``scipy.sparse.issparse(x)`` guard to ``False`` (dense-only ABI), this removes
    the dead sparse branch entirely -- numba DCEs it before typing, but pythran
    statically types it (``.toarray()`` on a dense array) and errors otherwise."""

    def __init__(self):
        self.changed = False

    def _const_bool(self, node: ast.AST):
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            v = self._const_bool(node.operand)
            return None if v is None else (not v)  # ``not issubdtype(...)`` folds too
        if isinstance(node, ast.BoolOp):
            vals = [self._const_bool(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return False if any(v is False for v in vals) else (True if all(v is True for v in vals) else None)
            return True if any(v is True for v in vals) else (False if all(v is False for v in vals) else None)
        return None

    def visit_If(self, node: ast.If):
        self.generic_visit(node)
        taken = self._const_bool(node.test)
        if taken is True:
            self.changed = True
            return node.body
        if taken is False:
            self.changed = True
            return node.orelse or [ast.copy_location(ast.Pass(), node)]
        return node


def _fd_step(precision: Optional[str] = None) -> str:
    """``sqrt(machine epsilon)`` of the WORKING float type, as a source literal.

    The forward-difference step scale MINPACK's ``fdjac2`` uses (``h = sqrt(epsfcn) *
    |p_j|``, ``epsfcn`` defaulting to the machine epsilon OF THE WORKING TYPE).
    :func:`_curve_fit_lm_lines` reuses the SAME rule so the emitted fit and the scipy
    reference share a Jacobian truncation error and converge to the same stationary
    point (see that function's docstring).

    It MUST track ``precision``: this is emitted as source text, so the fp64 literal
    ``sqrt(DBL_EPSILON) = 1.49e-08`` survives into an fp32 kernel unchanged (the desugar
    is an AST rewrite and ``apply_precision`` only remaps dtype tables, never body
    literals). There the step underflows -- raman_fitting fits an amplitude of ~1580,
    and ``1580 + 1.49e-08*1580`` is exactly ``1580`` in fp32, whose ulp is ~1.9e-04 --
    so every Jacobian column comes out identically zero, every LM trip is rejected, and
    the fit returns its initial guess having never moved. sqrt(eps) is what balances
    truncation against round-off; a merely representable step would still be swamped.

    Read off the registry rather than hardcoded per dtype. A storage-only float (fp8)
    has no numpy finfo, so it falls back to the fp64 rule -- curve_fit at fp8 is not a
    thing we emit, and a wrong-but-fp64 step is the status quo, not a regression.
    """
    dtype = dtypes.canonical(precision) if precision else "float64"
    try:
        eps = float(np.finfo(np.dtype(dtype)).eps)
    except TypeError:
        eps = float(np.finfo(np.float64).eps)
    return repr(math.sqrt(eps))


def _list_display_elts(node: ast.AST) -> Optional[List[ast.expr]]:
    """A 1-D ``[e0, e1, ...]`` display -> its element expressions, else None.

    Only a flat list of NON-display elements qualifies: a nested list / tuple
    element (``peaks = [(1580.0, 9.0), ...]``) is a 2-D literal this 1-D
    array fold would mis-shape, so it is refused.
    """
    if not isinstance(node, ast.List):
        return None
    if any(isinstance(e, (ast.List, ast.Tuple, ast.Starred)) for e in node.elts):
        return None
    return list(node.elts)


def _len_call_of(node: ast.AST, name: str) -> bool:
    """``len(<name>)`` -- the list's running length."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len"
            and len(node.args) == 1 and isinstance(node.args[0], ast.Name) and node.args[0].id == name)


class _SubstLenWithIndex(ast.NodeTransformer):
    """``len(<name>)`` -> ``<idx>`` inside an appended value expression.

    In ``while len(c) < K: c.append(1200.0 + 200.0 * len(c))`` the appended value
    reads the length AT append time, which for the element landing at index ``i``
    is exactly ``i`` -- so the loop-index substitution is semantics-preserving.
    """

    def __init__(self, name: str, idx: str):
        self.name = name
        self.idx = idx

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if _len_call_of(node, self.name):
            return ast.copy_location(ast.Name(id=self.idx, ctx=ast.Load()), node)
        return node


def _appended_elts(stmt: ast.stmt, name: str) -> Optional[List[ast.expr]]:
    """One list-grow statement -> the element expressions it appends, else None.

    Accepts ``name.append(e)``, ``name += [e...]`` and ``name = name + [e...]``
    -- the three spellings the corpus uses to build a parameter vector.
    """
    if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == "append"
            and isinstance(stmt.value.func.value, ast.Name) and stmt.value.func.value.id == name
            and len(stmt.value.args) == 1):
        return [stmt.value.args[0]]
    if (isinstance(stmt, ast.AugAssign) and isinstance(stmt.op, ast.Add) and isinstance(stmt.target, ast.Name)
            and stmt.target.id == name):
        return _list_display_elts(stmt.value)
    if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name and isinstance(stmt.value, ast.BinOp)
            and isinstance(stmt.value.op, ast.Add) and isinstance(stmt.value.left, ast.Name)
            and stmt.value.left.id == name):
        return _list_display_elts(stmt.value.right)
    return None


def _off_add(off: str, delta: str) -> str:
    """``off + delta`` as a source string, folding the literal + literal case so
    the common ``0 + 1 + 1`` prefix stays a plain ``2`` in the emitted index."""
    try:
        return str(int(off) + int(delta))
    except ValueError:
        return f"{off} + {delta}" if off != "0" else delta


def _range_bound(node: ast.AST) -> Optional[ast.expr]:
    """``range(E)`` -> ``E`` (single-argument form only), else None."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"
            and len(node.args) == 1):
        return node.args[0]
    return None


def _plan_list_build(body: List[ast.stmt], start: int, name: str):
    """Plan the fold of the list variable ``name`` built from ``body[start]``.

    Returns ``(stmts, next_index)`` -- the replacement statements and the index
    one past the last statement consumed -- or ``None`` to leave the build
    verbatim (an unrecognised mutation; the emitter's ``List`` guard then fails
    loudly, exactly as before this pass existed).

    Recognised build shape (raman_fitting's peak-centre and initial-guess
    preludes), in order::

        name = [<scalar>, ...]              # seed display (possibly empty)
        for v in range(E):                  # fixed number of elements per trip
            name += [<scalar>, ...]
        while len(name) < E:                # extend to a symbolic length
            name.append(<expr of len(name)>)
        name = name[:E]                     # truncate to the final length
        name += [<scalar>, ...]

    Each element's destination index is tracked as a symbolic offset, so the
    result is a plain ``np.zeros`` + indexed stores whose length is an
    expression in the kernel's size symbols -- ``npeaks`` is ``params.shape[0]``
    (a RUNTIME argument in the emitted C, not a compile-time constant), so a
    fixed-length literal array would be wrong for any preset with a different K.
    """
    seed = _list_display_elts(body[start].value)
    if seed is None:
        return None
    # (kind, base_offset, payload) segments in build order.
    segs: List[Tuple[str, str, object]] = [("lit", "0", seed)]
    off = _off_add("0", str(len(seed)))
    trunc: Optional[str] = None
    all_int = all(isinstance(e, ast.Constant) and isinstance(e.value, int) and not isinstance(e.value, bool)
                  for e in seed)
    j = start + 1
    while j < len(body):
        stmt = body[j]
        elts = _appended_elts(stmt, name)
        if elts is not None:  # straight-line append / extend
            if trunc is not None:
                return None  # a grow AFTER the truncate: length no longer well defined
            segs.append(("lit", off, elts))
            off = _off_add(off, str(len(elts)))
            all_int = all_int and all(isinstance(e, ast.Constant) and isinstance(e.value, int) for e in elts)
            j += 1
            continue
        # ``for v in range(E): name += [...]`` -- a fixed stride per trip.
        if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name) and not stmt.orelse:
            bound = _range_bound(stmt.iter)
            if bound is None:
                return None
            per: List[ast.expr] = []
            for sub in stmt.body:
                got = _appended_elts(sub, name)
                if got is None:
                    return None  # loop does more than grow the list -> not ours
                per.extend(got)
            if not per or trunc is not None:
                return None
            segs.append(("for", off, (stmt.target.id, ast.unparse(bound), per)))
            off = _off_add(off, f"{len(per)} * ({ast.unparse(bound)})")
            all_int = False
            j += 1
            continue
        # ``while len(name) < E: name.append(<expr>)`` -- extend to length E.
        if (isinstance(stmt, ast.While) and isinstance(stmt.test, ast.Compare) and len(stmt.test.ops) == 1
                and isinstance(stmt.test.ops[0], ast.Lt) and _len_call_of(stmt.test.left, name)
                and len(stmt.body) == 1):
            got = _appended_elts(stmt.body[0], name)
            if got is None or len(got) != 1 or trunc is not None:
                return None
            bound = ast.unparse(stmt.test.comparators[0])
            # The while alone leaves length max(len(seed), E); only a following
            # ``name = name[:E]`` on the SAME bound pins it to E. Without that the
            # length is a max() this fold does not model -- leave it verbatim.
            nxt = body[j + 1] if j + 1 < len(body) else None
            if not (isinstance(nxt, ast.Assign) and len(nxt.targets) == 1 and isinstance(nxt.targets[0], ast.Name)
                    and nxt.targets[0].id == name and isinstance(nxt.value, ast.Subscript)
                    and isinstance(nxt.value.value, ast.Name) and nxt.value.value.id == name
                    and isinstance(nxt.value.slice, ast.Slice) and nxt.value.slice.lower is None
                    and nxt.value.slice.step is None and nxt.value.slice.upper is not None
                    and ast.unparse(nxt.value.slice.upper) == bound):
                return None
            segs.append(("while", off, (bound, got[0])))
            trunc = bound
            all_int = False
            j += 2
            continue
        break
    length = trunc if trunc is not None else off
    if length == "0":
        return None  # an empty list nothing grows -- not an array build
    pfx = f"__lst_{name}"
    dtype = "np.int64" if all_int else "np.float64"
    lines = [f"{name} = np.zeros(({length},), dtype={dtype})"]
    for kind, base, payload in segs:
        if kind == "lit":
            for k, e in enumerate(payload):
                idx = _off_add(base, str(k))
                store = f"{name}[{idx}] = {ast.unparse(e)}"
                # A truncating build may drop seed elements (K < len(seed)), so a
                # direct store is guarded by the final length; an untruncated
                # build stores unconditionally.
                lines.append(f"if {idx} < {length}:\n    {store}" if trunc is not None else store)
        elif kind == "for":
            var, bound, per = payload
            lines.append(f"for {var} in range({bound}):")
            for k, e in enumerate(per):
                idx = _off_add(f"{len(per)} * {var}", str(k))
                lines.append(f"    {name}[{_off_add(base, idx)}] = {ast.unparse(e)}")
        else:  # "while" -- elements at index base .. length-1, value reads its own index
            bound, val = payload
            idx = f"{pfx}_i"
            filled = ast.unparse(_SubstLenWithIndex(name, idx).visit(copy.deepcopy(val)))
            lines.append(f"for {idx} in range({base}, {bound}):\n    {name}[{idx}] = {filled}")
    return ast.parse("\n".join(lines)).body, j


def _fold_list_preludes(fn: ast.FunctionDef) -> None:
    """Fold kernel-body Python list builds into ``np.zeros`` + indexed stores.

    The native emitters have no list type -- a surviving ``List`` display is a
    hard ``NotImplementedError`` at emit -- and a list whose length is a size
    SYMBOL cannot be spelled as a fixed literal array either. Only top-level
    statements of ``fn`` are considered: that is where the corpus builds its
    parameter vectors, and it keeps the offset bookkeeping (which must see the
    statements in execution order) straightforward. An unrecognised build is
    left untouched, so this pass can only turn an emit failure into a success.
    """
    out: List[ast.stmt] = []
    i = 0
    changed = False
    while i < len(fn.body):
        stmt = fn.body[i]
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.List)):
            plan = _plan_list_build(fn.body, i, stmt.targets[0].id)
            if plan is not None:
                out.extend(plan[0])
                i = plan[1]
                changed = True
                continue
        out.append(stmt)
        i += 1
    if changed:
        fn.body = out
        ast.fix_missing_locations(fn)


def _curve_fit_lm_lines(popt: str, f: str, x: str, y: str, p0: str, pfx: str, iters: int,
                        precision: Optional[str] = None) -> List[str]:
    """Source lines for a naive Levenberg-Marquardt fit replacing ``curve_fit``.

    ``scipy.optimize.curve_fit(f, x, y, p0=...)`` with no bounds / sigma is an
    unconstrained nonlinear least-squares fit; MINPACK's ``lmdif`` (what scipy
    calls) is a trust-region LM over a forward-difference Jacobian. This emits
    the textbook damped-normal-equations form of the same thing::

        r = f(x, p) - y                       # residual
        J[:, c] = (f(x, p + h e_c) - f(x, p)) / h   # forward-difference Jacobian
        (J^T J + lam * diag(J^T J)) dp = -J^T r
        accept dp if it lowers ||r||^2 (lam /= 10), else keep p (lam *= 10)

    A rejected step is not retried within the trip: the next trip simply re-solves
    at the same p with a larger lam (costing one redundant Jacobian), which keeps
    the emitted nest a flat fixed-trip loop with no inner convergence search.

    Two choices make the result agree with scipy's to the harness's 1e-9, rather
    than merely landing on the same optimum to fitting accuracy:

    * The step ``h = sqrt(eps) * |p_j|`` is MINPACK's own (:func:`_fd_step`, over the
      WORKING precision -- an fp64 step silently vanishes at fp32). A
      finite-difference Jacobian shifts the stationary point from ``J^T r = 0``
      to ``J~^T r = 0``; sharing the step makes both solvers inherit the SAME
      shift instead of two different ones.
    * A fixed trip count, iterating well past convergence, rather than a
      dynamic break -- the static backends want a static trip count, and the
      extra trips are no-ops once the step is rejected at ``lam`` ceiling.

    ``np.linalg.solve`` is left for the existing solve expander (Gauss-Jordan
    with partial pivoting) to lower; the damping keeps the system positive
    definite, so the pivoting never meets a singular column in practice.

    Every generated name differs from every other by more than CASE: Fortran
    identifiers are case-insensitive, so a ``_J`` Jacobian beside a ``_j`` loop
    index would collide into one symbol in the emitted module.
    """
    n, m = f"{p0}.shape[0]", f"{y}.shape[0]"
    i, c, a, b, it = f"{pfx}_i", f"{pfx}_c", f"{pfx}_a", f"{pfx}_b", f"{pfx}_it"
    return [
        f"{popt} = np.zeros(({n},), dtype=np.float64)",
        f"{pfx}_jac = np.zeros(({m}, {n}), dtype=np.float64)",
        f"{pfx}_ata = np.zeros(({n}, {n}), dtype=np.float64)",
        f"{pfx}_atad = np.zeros(({n}, {n}), dtype=np.float64)",
        f"{pfx}_grad = np.zeros(({n},), dtype=np.float64)",
        f"{pfx}_rhs = np.zeros(({n},), dtype=np.float64)",
        f"{pfx}_pt = np.zeros(({n},), dtype=np.float64)",
        f"{pfx}_r = np.zeros(({m},), dtype=np.float64)",
        f"for {i} in range({n}):",
        f"    {popt}[{i}] = {p0}[{i}]",
        f"{pfx}_lam = 0.001",
        f"{pfx}_f0 = {f}({x}, {popt})",
        f"{pfx}_ssq = 0.0",
        f"for {i} in range({m}):",
        f"    {pfx}_r[{i}] = {pfx}_f0[{i}] - {y}[{i}]",
        f"    {pfx}_ssq = {pfx}_ssq + {pfx}_r[{i}] * {pfx}_r[{i}]",
        f"for {it} in range({iters}):",
        f"    for {c} in range({n}):",
        f"        {pfx}_h = {_fd_step(precision)} * np.abs({popt}[{c}])",
        f"        if {pfx}_h == 0.0:",
        f"            {pfx}_h = {_fd_step(precision)}",
        f"        for {i} in range({n}):",
        f"            {pfx}_pt[{i}] = {popt}[{i}]",
        f"        {pfx}_pt[{c}] = {popt}[{c}] + {pfx}_h",
        f"        {pfx}_fp = {f}({x}, {pfx}_pt)",
        f"        for {i} in range({m}):",
        f"            {pfx}_jac[{i}, {c}] = ({pfx}_fp[{i}] - {pfx}_f0[{i}]) / {pfx}_h",
        f"    for {a} in range({n}):",
        f"        {pfx}_ga = 0.0",
        f"        for {i} in range({m}):",
        f"            {pfx}_ga = {pfx}_ga + {pfx}_jac[{i}, {a}] * {pfx}_r[{i}]",
        f"        {pfx}_grad[{a}] = {pfx}_ga",
        f"        for {b} in range({n}):",
        f"            {pfx}_ab = 0.0",
        f"            for {i} in range({m}):",
        f"                {pfx}_ab = {pfx}_ab + {pfx}_jac[{i}, {a}] * {pfx}_jac[{i}, {b}]",
        f"            {pfx}_ata[{a}, {b}] = {pfx}_ab",
        f"    for {a} in range({n}):",
        f"        for {b} in range({n}):",
        f"            {pfx}_atad[{a}, {b}] = {pfx}_ata[{a}, {b}]",
        f"        {pfx}_atad[{a}, {a}] = {pfx}_ata[{a}, {a}] + {pfx}_lam * {pfx}_ata[{a}, {a}]",
        f"        {pfx}_rhs[{a}] = -{pfx}_grad[{a}]",
        f"    {pfx}_step = np.linalg.solve({pfx}_atad, {pfx}_rhs)",
        f"    for {i} in range({n}):",
        f"        {pfx}_pt[{i}] = {popt}[{i}] + {pfx}_step[{i}]",
        f"    {pfx}_ft = {f}({x}, {pfx}_pt)",
        f"    {pfx}_ssqt = 0.0",
        f"    for {i} in range({m}):",
        f"        {pfx}_ssqt = {pfx}_ssqt + ({pfx}_ft[{i}] - {y}[{i}]) * ({pfx}_ft[{i}] - {y}[{i}])",
        f"    if {pfx}_ssqt < {pfx}_ssq:",
        f"        for {i} in range({n}):",
        f"            {popt}[{i}] = {pfx}_pt[{i}]",
        f"        for {i} in range({m}):",
        f"            {pfx}_f0[{i}] = {pfx}_ft[{i}]",
        f"            {pfx}_r[{i}] = {pfx}_ft[{i}] - {y}[{i}]",
        f"        {pfx}_ssq = {pfx}_ssqt",
        f"        {pfx}_lam = {pfx}_lam * 0.1",
        f"        if {pfx}_lam < 1e-14:",
        f"            {pfx}_lam = 1e-14",
        f"    else:",
        f"        {pfx}_lam = {pfx}_lam * 10.0",
        f"        if {pfx}_lam > 10000000000.0:",
        f"            {pfx}_lam = 10000000000.0",
    ]


#: ``curve_fit`` keywords the LM lowering reproduces exactly. ``maxfev`` bounds
#: MINPACK's function-evaluation budget; the emitted fit uses a fixed trip count
#: that converges far inside it, so honouring the number is meaningless -- but a
#: keyword that CHANGES the objective (bounds / sigma / absolute_sigma) or the
#: derivative (jac) would silently fit something else, so it is refused.
_CURVE_FIT_IGNORED_KW = frozenset({"maxfev", "p0", "method", "full_output"})

#: Fixed LM trip count. The corpus fit is converged well inside it -- raising the
#: count to 200 reproduces the 100-trip parameters bit-for-bit -- and each surplus
#: trip is a rejected step (lambda pinned at its ceiling), so the margin costs
#: time rather than accuracy and buys insensitivity to the starting guess.
_CURVE_FIT_ITERS = 100


def _curve_fit_call(node: ast.AST) -> Optional[ast.Call]:
    """A ``curve_fit(...)`` call -- bare, ``scipy.optimize.``- or ``optimize.``-
    qualified -- else None."""
    if not isinstance(node, ast.Call):
        return None
    fn = node.func
    if isinstance(fn, ast.Name) and fn.id == "curve_fit":
        return node
    if isinstance(fn, ast.Attribute) and fn.attr == "curve_fit":
        return node
    return None


class _CurveFitRewriter(ast.NodeTransformer):
    """``popt, pcov = curve_fit(f, x, y, p0=g)`` -> a naive LM loop nest.

    The static backends have no scipy; the fit is an unconstrained nonlinear
    least-squares problem over a smooth analytic model, so it lowers to plain
    loops + arithmetic (:func:`_curve_fit_lm_lines`) with the linear solve left
    to the existing ``np.linalg.solve`` expander.

    The model ``f`` is a nested / module-level ``def f(grid, *p)``: curve_fit
    calls it as ``f(x, *popt)``, so its varargs tuple IS the parameter vector.
    Rebinding ``*p`` to a single ndarray parameter is therefore exactly
    curve_fit's own contract, and turns ``f`` into an ordinary one-array-in,
    one-array-out helper the existing inliner already handles (its free
    ``npeaks`` resolves in the kernel scope it inlines into). Negative constant
    indices into ``p`` (``p[-1]`` -- the shared baseline) are rewritten against
    the now-known parameter count, which the emitters cannot fold themselves.

    ``pcov`` (the covariance estimate) is NOT computed: the corpus kernel binds
    it to ``_`` and never reads it. A live ``pcov`` raises rather than emitting
    a silently-absent output.
    """

    def __init__(self, tree: ast.Module, kernel: ast.FunctionDef, precision: Optional[str] = None):
        self.tree = tree
        self.kernel = kernel
        self.ctr = 0
        self.changed = False
        #: Working float precision, for the LM's finite-difference step (:func:`_fd_step`).
        self.precision = precision
        #: ``(popt_name, parameter-count expression)`` per lowered fit.
        self.fitted: List[Tuple[str, str]] = []

    def _find_model(self, name: str) -> Optional[ast.FunctionDef]:
        for scope in (self.kernel, self.tree):
            for node in ast.walk(scope):
                if isinstance(node, ast.FunctionDef) and node.name == name and node is not self.kernel:
                    return node
        return None

    @staticmethod
    def _rebind_varargs(fdef: ast.FunctionDef, nexpr: str) -> None:
        """``def f(grid, *p)`` -> ``def f(grid, p)`` with ``p[-k]`` -> ``p[nexpr - k]``."""
        vp = fdef.args.vararg
        if vp is None:
            return
        fdef.args.args.append(ast.arg(arg=vp.arg))
        fdef.args.vararg = None
        for sub in ast.walk(fdef):
            if (isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id == vp.arg
                    and isinstance(sub.slice, ast.UnaryOp) and isinstance(sub.slice.op, ast.USub)
                    and isinstance(sub.slice.operand, ast.Constant)):
                sub.slice = ast.parse(f"{nexpr} - {sub.slice.operand.value}", mode="eval").body
        ast.fix_missing_locations(fdef)

    def visit_Assign(self, node: ast.Assign):
        call = _curve_fit_call(node.value)
        if call is None or len(node.targets) != 1:
            return node
        tgt = node.targets[0]
        if isinstance(tgt, ast.Tuple):
            if len(tgt.elts) != 2 or not all(isinstance(e, ast.Name) for e in tgt.elts):
                raise DesugarError(f"curve_fit: unsupported target {ast.unparse(tgt)}")
            if tgt.elts[1].id != "_":
                raise DesugarError("curve_fit: the pcov covariance output is not computed by the LM "
                                   f"lowering, but {tgt.elts[1].id!r} binds it")
            popt = tgt.elts[0].id
        elif isinstance(tgt, ast.Name):
            popt = tgt.id
        else:
            raise DesugarError(f"curve_fit: unsupported target {ast.unparse(tgt)}")
        for kw in call.keywords:
            if kw.arg not in _CURVE_FIT_IGNORED_KW:
                raise DesugarError(f"curve_fit: keyword {kw.arg!r} changes the fit; the LM lowering "
                                   "only reproduces the unweighted, unbounded, FD-Jacobian form")
        p0 = next((kw.value for kw in call.keywords if kw.arg == "p0"), None)
        if p0 is None and len(call.args) >= 4:
            p0 = call.args[3]
        if len(call.args) < 3 or p0 is None:
            raise DesugarError("curve_fit: need f, xdata, ydata and an explicit p0")
        f, x, y = call.args[0], call.args[1], call.args[2]
        if not all(isinstance(v, ast.Name) for v in (f, x, y, p0)):
            raise DesugarError("curve_fit: f / xdata / ydata / p0 must be plain names")
        model = self._find_model(f.id)
        if model is None:
            raise DesugarError(f"curve_fit: model {f.id!r} is not a def in this module")
        pfx = f"__lm{self.ctr}"
        self.ctr += 1
        nexpr = f"{p0.id}.shape[0]"
        self._rebind_varargs(model, nexpr)
        self.fitted.append((popt, nexpr))
        self.changed = True
        lines = _curve_fit_lm_lines(popt, f.id, x.id, y.id, p0.id, pfx, _CURVE_FIT_ITERS, self.precision)
        return ast.parse("\n".join(lines)).body


class _NegParamIndexFold(ast.NodeTransformer):
    """``popt[-k]`` -> ``popt[<len> - k]`` for a fitted parameter vector.

    The kernel reads the fitted baseline as ``offset[0] = popt[-1]``. ``popt``
    is created by the LM lowering with a symbolic length, so the emitters (which
    fold a negative index only against a STATIC extent) cannot resolve it.
    """

    def __init__(self, name: str, nexpr: str):
        self.name = name
        self.nexpr = nexpr

    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)
        if (isinstance(node.value, ast.Name) and node.value.id == self.name
                and isinstance(node.slice, ast.UnaryOp) and isinstance(node.slice.op, ast.USub)
                and isinstance(node.slice.operand, ast.Constant)):
            node.slice = ast.parse(f"{self.nexpr} - {node.slice.operand.value}", mode="eval").body
            ast.fix_missing_locations(node)
        return node


def rewrite_curve_fit(tree: ast.Module, kernel: ast.FunctionDef, precision: Optional[str] = None) -> None:
    """Lower every ``curve_fit`` in ``kernel`` to a naive LM loop nest, in place.

    Runs BEFORE helper inlining (like the eigh rewriter) so the model ``def`` is
    still a distinct function to rebind, and the LM's calls to it are inlined by
    the ordinary helper machinery afterwards.

    ``precision`` is the working float type. It is needed HERE, at the source rewrite,
    because the LM's finite-difference step is a numerical-method constant baked into
    the emitted body -- the later ``apply_precision`` remaps dtype tables only and
    cannot reach a literal. See :func:`_fd_step`. ``None`` keeps the fp64 rule.
    """
    fits = [n for n in ast.walk(kernel) if _curve_fit_call(n) is not None]
    if not fits:
        return
    # The list preludes build the p0 vector this fit consumes, so they must be
    # arrays before the LM lines index them.
    _fold_list_preludes(kernel)
    rw = _CurveFitRewriter(tree, kernel, precision)
    kernel.body = [s for stmt in kernel.body for s in _as_stmts(rw.visit(stmt))]
    if not rw.changed:
        return
    # ``offset[0] = popt[-1]`` reads the fitted vector's tail; resolve it against
    # the parameter count now that the vector is an array of known length.
    for popt, nexpr in rw.fitted:
        _NegParamIndexFold(popt, nexpr).visit(kernel)
    ast.fix_missing_locations(kernel)


def _as_stmts(res) -> List[ast.stmt]:
    """A NodeTransformer result (one node, a list, or a dropped ``None``) as a list."""
    if res is None:
        return []
    return res if isinstance(res, list) else [res]


def _as_matmul(node: ast.AST):
    """``a @ b`` / ``np.matmul(a, b)`` / ``np.dot(a, b)`` -> ``(a, b)`` else None."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        return node.left, node.right
    if _np_attr(node) in ("matmul", "dot") and len(getattr(node, "args", [])) == 2:
        return node.args[0], node.args[1]
    return None


def _as_reshape(node: ast.AST):
    """``np.reshape(x, shape)`` / ``x.reshape(shape)`` -> ``(x, shape_node)`` else None."""
    if _np_attr(node) == "reshape" and len(node.args) >= 2:
        return node.args[0], node.args[1]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "reshape"
            and node.args):
        shape = node.args[0] if len(node.args) == 1 else ast.Tuple(elts=list(node.args), ctx=ast.Load())
        return node.func.value, shape
    return None


class _ReshapeMatmulInline(ast.NodeTransformer):
    """``T[:] = np.reshape(np.reshape(X, (*batch, 1, K)) @ Y, (*batch, N))`` -> a
    contraction loop ``r[*b, n] = sum_k X[*b, k] * Y[k, n]`` into a fresh full
    temp (so the ``A = f(A)`` WAR in doitgen is safe). numba cannot type the
    reshape-wrapped batched ``@`` (the existing batched-matmul pass deliberately
    refuses reshape-wrapped operands as a miscompile risk). Fires ONLY on the
    unit-dim-insertion form (``mid[-2] == 1``, ``len(mid) == X.ndim + 1``, Y 2-D);
    a genuinely different reshape is left verbatim. A matched-but-inconsistent
    shape (Y not 2-D) raises DesugarError rather than miscompiling."""

    def __init__(self, ranks: Dict[str, int]):
        self.ranks = ranks
        self.changed = False
        self._ctr = 0

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        tgt = node.targets[0]
        if not ((isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name)
                 and isinstance(tgt.slice, ast.Slice)) or isinstance(tgt, ast.Name)):
            return node
        outer = _as_reshape(node.value)
        if not outer:
            return node
        mm = _as_matmul(outer[0])
        if not mm:
            return node
        inner = _as_reshape(mm[0])
        if not inner or not isinstance(inner[0], ast.Name) or not isinstance(mm[1], ast.Name):
            return node
        X, Y, mid = inner[0], mm[1], inner[1]
        rX = _expr_rank(X, self.ranks)
        # Fire only on the unit-dim-insertion batched form; other reshapes -> verbatim.
        if not (isinstance(mid, ast.Tuple) and rX and len(mid.elts) == rX + 1
                and isinstance(mid.elts[-2], ast.Constant) and mid.elts[-2].value == 1):
            return node
        if _expr_rank(Y, self.ranks) != 2 or rX < 2:
            raise DesugarError(f"reshape-batched matmul: unit-dim form needs a 2-D right operand and a >=2-D "
                               f"left operand (got left ndim {rX}, right ndim {_expr_rank(Y, self.ranks)})")
        p = f"__dg{self._ctr}"
        self._ctr += 1
        batch = list(range(rX - 1))
        bi = [f"{p}_b{i}" for i in batch]
        bidx = ", ".join(bi)
        oshape = "".join(f"{X.id}.shape[{i}], " for i in batch) + f"{Y.id}.shape[1]"
        temp = f"{p}_o"
        lines = [f"{temp} = np.zeros(({oshape},), {X.id}.dtype)"]
        deep = ""
        for i in batch:
            lines.append(f"{deep}for {bi[i]} in range({X.id}.shape[{i}]):")
            deep += "    "
        lines.append(f"{deep}for {p}_n in range({Y.id}.shape[1]):")
        lines.append(f"{deep}    for {p}_k in range({X.id}.shape[{rX - 1}]):")
        lines.append(f"{deep}        {temp}[{bidx}, {p}_n] += {X.id}[{bidx}, {p}_k] * {Y.id}[{p}_k, {p}_n]")
        node.value = ast.Name(id=temp, ctx=ast.Load())
        self.changed = True
        return [ast.copy_location(s, node) for s in ast.parse("\n".join(lines)).body] + [node]


def _np_linalg_attr(node: ast.AST) -> Optional[str]:
    """``np.linalg.<attr>(...)`` / ``numpy.linalg.<attr>(...)`` call -> ``attr``
    (``cholesky``/``solve``/``inv``), else None. Like :func:`_np_fft_attr` but for
    the two-level ``np.linalg`` prefix the single-level ``_np_attr`` misses."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "linalg"
            and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id in ("np", "numpy")):
        return node.func.attr
    return None


def _cholesky_lines(temp: str, a: str, n: str, p: str, hermitian: bool = False) -> List[str]:
    """Source lines computing ``np.linalg.cholesky(a)`` into a freshly zeroed
    ``temp`` via the Cholesky-Banachiewicz triple loop (the same O(n^3) form the
    C/Fortran backends lower in :func:`lib_nodes.expand_cholesky`). ``temp`` is a
    fresh buffer, so a strict-upper-triangle of zeros (numpy's convention) comes
    for free from the ``np.zeros`` init.

    ``hermitian`` (a complex-Hermitian positive-definite ``a``, e.g. the metric
    ``b`` in a generalized eigenproblem) conjugates the second factor of every
    inner product (``a = L Lᴴ``) and takes the real part of the diagonal argument
    to ``sqrt`` (Hermitian diagonals are real up to roundoff); for a real ``a``
    both reduce to the plain form."""
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


def _gauss_jordan_lines(aw: str, o: str, n: str, m: Optional[str], p: str) -> List[str]:
    """Source lines for Gauss-Jordan elimination with partial pivoting reducing
    ``(aw | o)`` in place so ``aw`` -> identity and ``o`` -> ``aw^-1 @ o``. ``m``
    marks a 2-D ``o`` (whole-row ops on it) vs ``None`` for a 1-D vector ``o``
    (scalar ops). This is the shared core of both ``np.linalg.solve`` (``o`` = a
    copy of ``b``) and ``np.linalg.inv`` (``o`` = the identity).

    Row-vectorized (``aw[r] -= g * aw[k]`` etc.) rather than an inner column loop:
    the per-element form is arithmetically identical but leaves a scalar temp
    aliasing an array element, which pythran mis-types as an ndarray in a larger
    function. Whole-row ops sidestep that and match numpy's LU-with-partial-pivot
    solve/inv to rounding for well-conditioned systems (validated ~1e-17)."""
    k, r = f"{p}_k", f"{p}_r"
    # A 2-D ``o`` swaps whole rows (a fresh row .copy()); a 1-D ``o`` swaps a scalar.
    o_swap = [f"        {p}_to = {o}[{k}].copy()"] if m is not None else [f"        {p}_to = {o}[{k}]"]
    o_swap += [f"        {o}[{k}] = {o}[{p}_pv]", f"        {o}[{p}_pv] = {p}_to"]
    return [
        f"for {k} in range({n}):",
        f"    {p}_pv = {k}",
        f"    for {r} in range({k} + 1, {n}):",
        f"        if np.abs({aw}[{r}, {k}]) > np.abs({aw}[{p}_pv, {k}]):",
        f"            {p}_pv = {r}",
        f"    if {p}_pv != {k}:",
        f"        {p}_tr = {aw}[{k}].copy()",
        f"        {aw}[{k}] = {aw}[{p}_pv]",
        f"        {aw}[{p}_pv] = {p}_tr",
    ] + o_swap + [
        f"    {p}_f = {aw}[{k}, {k}]",
        f"    {aw}[{k}] = {aw}[{k}] / {p}_f",
        f"    {o}[{k}] = {o}[{k}] / {p}_f",
        f"    for {r} in range({n}):",
        f"        if {r} != {k}:",
        f"            {p}_g = {aw}[{r}, {k}]",
        f"            {aw}[{r}] -= {p}_g * {aw}[{k}]",
        f"            {o}[{r}] -= {p}_g * {o}[{k}]",
    ]


class _LinalgHoister(ast.NodeTransformer):
    """Replace each lowerable ``np.linalg.cholesky/solve/inv(...)`` inside one
    statement with a fresh temp Name, emitting its loop nest into ``self.pre``.
    Only ops in ``lower_ops`` are touched (a backend whose native ``np.linalg``
    handles an op leaves it verbatim). Owned-but-unhandled variants (a >2-D
    operand) raise :class:`DesugarError`; an unknown-rank operand is left verbatim
    (an inference gap, a clean backend skip). A non-Name operand is materialised
    to a temp first so the loop body can index it."""

    def __init__(self, ranks: Dict[str, int], dtypes: Dict[str, str], lower_ops: set, ctr: int):
        self.ranks = ranks
        self.dtypes = dtypes
        self.lower_ops = lower_ops
        self.ctr = ctr
        self.pre: List[ast.stmt] = []

    def _src_name(self, node: ast.AST, p: str, tag: str) -> str:
        """A Name operand is used directly; an expression is materialised to a
        contiguous temp (so the loop body can index it repeatedly)."""
        if isinstance(node, ast.Name):
            return node.id
        nm = f"{p}_{tag}"
        self.pre.append(ast.parse(f"{nm} = np.ascontiguousarray({ast.unparse(node)})").body[0])
        return nm

    def _emit(self, lines: List[str]) -> None:
        self.pre.extend(ast.parse("\n".join(lines)).body)

    def _chol(self, node: ast.Call):
        a = node.args[0]
        ra = _expr_rank(a, self.ranks)
        if ra is None:
            return node  # unknown rank -> verbatim (inference gap, not a raise)
        if ra != 2:
            raise DesugarError(f"np.linalg.cholesky: only a 2-D operand is lowered (got ndim {ra})")
        p = f"__chol{self.ctr}"
        self.ctr += 1
        an = self._src_name(a, p, "a")
        temp = f"{p}_o"
        self._emit(_cholesky_lines(temp, an, f"{an}.shape[0]", p, hermitian=_dtype_kind(a, self.dtypes) == "complex"))
        return ast.copy_location(ast.Name(id=temp, ctx=ast.Load()), node)

    def _solve(self, node: ast.Call):
        if len(node.args) < 2:
            return node
        a, b = node.args[0], node.args[1]
        ra, rb = _expr_rank(a, self.ranks), _expr_rank(b, self.ranks)
        if ra is None or rb is None:
            return node
        if ra != 2:
            raise DesugarError(f"np.linalg.solve: A must be 2-D (got ndim {ra})")
        if rb not in (1, 2):
            raise DesugarError(f"np.linalg.solve: b must be 1-D or 2-D (got ndim {rb})")
        p = f"__solv{self.ctr}"
        self.ctr += 1
        an, bn = self._src_name(a, p, "a"), self._src_name(b, p, "b")
        temp = f"{p}_o"
        self._emit([f"{p}_aw = {an}.copy()", f"{temp} = {bn}.copy()"] +
                   _gauss_jordan_lines(f"{p}_aw", temp, f"{an}.shape[0]", (f"{bn}.shape[1]" if rb == 2 else None), p))
        return ast.copy_location(ast.Name(id=temp, ctx=ast.Load()), node)

    def _inv(self, node: ast.Call):
        a = node.args[0]
        ra = _expr_rank(a, self.ranks)
        if ra is None:
            return node
        if ra != 2:
            raise DesugarError(f"np.linalg.inv: only a 2-D operand is lowered (got ndim {ra})")
        p = f"__inv{self.ctr}"
        self.ctr += 1
        an = self._src_name(a, p, "a")
        temp, n = f"{p}_o", f"{an}.shape[0]"
        self._emit([
            f"{p}_aw = {an}.copy()", f"{temp} = np.zeros(({n}, {n}), {an}.dtype)", f"for {p}_d in range({n}):",
            f"    {temp}[{p}_d, {p}_d] = 1"
        ] + _gauss_jordan_lines(f"{p}_aw", temp, n, n, p))
        return ast.copy_location(ast.Name(id=temp, ctx=ast.Load()), node)

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)  # inner linalg calls first
        op = _np_linalg_attr(node)
        if op not in self.lower_ops or not node.args:
            return node
        return {"cholesky": self._chol, "solve": self._solve, "inv": self._inv}[op](node)


class _LinalgInline(ast.NodeTransformer):
    """Hoist ``np.linalg.cholesky/solve/inv`` out of any value-bearing statement
    into its preceding loop nest (cholesky2's ``A[:] = np.linalg.cholesky(A) +
    np.triu(A, k=1)`` -- the cholesky is computed into a fresh temp BEFORE ``A`` is
    overwritten, so the in-place read is safe; contour_integral's ``X =
    np.linalg.solve(Tz, Y)`` nested in a for-loop). Only backends lacking a native
    ``np.linalg`` (pythran) enable this; numba / dace keep the intrinsic."""

    def __init__(self, ranks: Dict[str, int], dtypes: Dict[str, str], lower_ops: set):
        self.ranks = ranks
        self.dtypes = dtypes
        self.lower_ops = lower_ops
        self.changed = False
        self._ctr = 0

    def _hoist(self, node):
        if node.value is None:
            return node
        h = _LinalgHoister(self.ranks, self.dtypes, self.lower_ops, self._ctr)
        node.value = h.visit(node.value)
        self._ctr = h.ctr
        if h.pre:
            self.changed = True
            return h.pre + [node]
        return node

    def visit_Assign(self, node):
        return self._hoist(node)

    def visit_AugAssign(self, node):
        return self._hoist(node)

    def visit_Return(self, node):
        return self._hoist(node)

    def visit_Expr(self, node):
        return self._hoist(node)


def _eigh_jacobi_lines(w: str, y: str, c: str, n: str, p: str) -> List[str]:
    """Source lines diagonalising the ``n``-by-``n`` Hermitian matrix ``c`` by
    cyclic complex Jacobi into eigenvalues ``w`` (ascending real, shape ``(n,)``)
    and eigenvectors ``y`` (unitary columns). Each sweep rotates every off-diagonal
    pair ``(pp, qq)`` to zero with a unitary ``J`` (phase ``apq/|apq|`` then a real
    symmetric Jacobi angle); the two-sided update ``A = Jᴴ A J`` runs as explicit
    column/row loops. Selection-sort ascending at the end (numpy.linalg.eigh's
    order). ``n`` is passed explicitly (not ``c.shape[0]``) so the C/Fortran
    backends see the resolved dimension symbol, not a temp's ``.shape``. Validated
    against numpy to ~5e-15."""
    # Diagonalise ``c`` IN PLACE -- the caller always passes a fresh, disposable
    # matrix (the reduced ``L⁻¹ a L⁻ᴴ`` or an ``ascontiguousarray`` copy), so no
    # extra working copy is needed (and the C/Fortran backends need not infer a
    # copy-temp's complex dtype).
    a, v = c, f"{p}_jv"
    return [
        # eigenvector accumulator V = I, as zeros + a diagonal loop (``np.eye``'s
        # C/Fortran expansion does not carry the complex dtype the way ``np.zeros``
        # does, so the accumulator would otherwise declare real).
        f"{v} = np.zeros(({n}, {n}), {c}.dtype)",
        f"for {p}_di in range({n}):",
        f"    {v}[{p}_di, {p}_di] = 1",
        f"for {p}_sw in range(80):",
        f"    {p}_off = 0.0",
        f"    for {p}_pp in range({n}):",
        f"        for {p}_qq in range({p}_pp + 1, {n}):",
        f"            {p}_off += {a}[{p}_pp, {p}_qq].real * {a}[{p}_pp, {p}_qq].real "
        f"+ {a}[{p}_pp, {p}_qq].imag * {a}[{p}_pp, {p}_qq].imag",
        f"    if {p}_off <= 1e-30:",
        f"        break",
        f"    for {p}_pp in range({n}):",
        f"        for {p}_qq in range({p}_pp + 1, {n}):",
        f"            {p}_apq = {a}[{p}_pp, {p}_qq]",
        f"            {p}_m = np.hypot({p}_apq.real, {p}_apq.imag)",
        f"            if {p}_m == 0.0:",
        f"                continue",
        f"            {p}_app = {a}[{p}_pp, {p}_pp].real",
        f"            {p}_aqq = {a}[{p}_qq, {p}_qq].real",
        f"            {p}_ephi = {p}_apq / {p}_m",
        f"            {p}_tau = ({p}_aqq - {p}_app) / (2.0 * {p}_m)",
        f"            {p}_ts = 1.0 if {p}_tau >= 0.0 else -1.0",
        f"            {p}_t = {p}_ts / (abs({p}_tau) + np.sqrt({p}_tau * {p}_tau + 1.0))",
        f"            {p}_c = 1.0 / np.sqrt({p}_t * {p}_t + 1.0)",
        f"            {p}_s = {p}_t * {p}_c",
        f"            for {p}_k in range({n}):",  # A @ J : columns pp, qq
        f"                {p}_akp = {a}[{p}_k, {p}_pp]",
        f"                {p}_akq = {a}[{p}_k, {p}_qq]",
        f"                {a}[{p}_k, {p}_pp] = {p}_c * {p}_akp - {p}_s * np.conj({p}_ephi) * {p}_akq",
        f"                {a}[{p}_k, {p}_qq] = {p}_s * {p}_ephi * {p}_akp + {p}_c * {p}_akq",
        f"            for {p}_k in range({n}):",  # Jᴴ @ A : rows pp, qq
        f"                {p}_apk = {a}[{p}_pp, {p}_k]",
        f"                {p}_aqk = {a}[{p}_qq, {p}_k]",
        f"                {a}[{p}_pp, {p}_k] = {p}_c * {p}_apk - {p}_s * {p}_ephi * {p}_aqk",
        f"                {a}[{p}_qq, {p}_k] = {p}_s * np.conj({p}_ephi) * {p}_apk + {p}_c * {p}_aqk",
        f"            for {p}_k in range({n}):",  # V @ J : columns pp, qq
        f"                {p}_vkp = {v}[{p}_k, {p}_pp]",
        f"                {p}_vkq = {v}[{p}_k, {p}_qq]",
        f"                {v}[{p}_k, {p}_pp] = {p}_c * {p}_vkp - {p}_s * np.conj({p}_ephi) * {p}_vkq",
        f"                {v}[{p}_k, {p}_qq] = {p}_s * {p}_ephi * {p}_vkp + {p}_c * {p}_vkq",
        f"{w} = np.zeros({n}, np.float64)",
        f"for {p}_i in range({n}):",
        f"    {w}[{p}_i] = {a}[{p}_i, {p}_i].real",
        f"for {p}_i in range({n}):",  # selection-sort ascending, permuting eigenvectors
        f"    {p}_mn = {p}_i",
        f"    for {p}_j in range({p}_i + 1, {n}):",
        f"        if {w}[{p}_j] < {w}[{p}_mn]:",
        f"            {p}_mn = {p}_j",
        f"    if {p}_mn != {p}_i:",
        f"        {p}_tw = {w}[{p}_i]",
        f"        {w}[{p}_i] = {w}[{p}_mn]",
        f"        {w}[{p}_mn] = {p}_tw",
        f"        for {p}_k in range({n}):",
        f"            {p}_tv = {v}[{p}_k, {p}_i]",
        f"            {v}[{p}_k, {p}_i] = {v}[{p}_k, {p}_mn]",
        f"            {v}[{p}_k, {p}_mn] = {p}_tv",
        f"{y} = {v}",
    ]


def _eigh_stmts(w: str,
                v: str,
                a: str,
                b: Optional[str],
                lo: str,
                hi: str,
                p: str,
                native_std: bool = False) -> List[str]:
    """Source lines for ``w, v = eigh(a[, b])[subset lo:hi]`` (ascending). The
    generalized Hermitian problem ``a x = w b x`` is reduced to standard form via
    the Cholesky factor of ``b`` (``b = L Lᴴ``): ``C = L⁻¹ a L⁻ᴴ`` is Hermitian
    with the same eigenvalues, and its eigenvectors back-transform as ``x = L⁻ᴴ
    y``. ``cholesky``/``inv``/``@`` are left as ``np.linalg``/matmul for the native
    backends (numba/dace) and lowered by :class:`_LinalgInline` for pythran; the
    standard eigh is the self-contained Jacobi above, unless ``native_std`` (used
    by backends whose ``np.linalg.eigh`` handles the standard complex-Hermitian
    case natively -- jax's ``jnp.linalg.eigh``), in which case a single
    ``np.linalg.eigh`` call is emitted for it. Validated vs scipy ~1e-15."""
    if b is not None:
        pre = [
            f"{p}_L = np.linalg.cholesky({b})",
            f"{p}_Li = np.linalg.inv({p}_L)",
            f"{p}_C = {p}_Li @ {a} @ {p}_Li.conj().T",
        ]
        cname = f"{p}_C"
    else:
        pre = [f"{p}_C = {a}.copy()"]
        cname = f"{p}_C"
    std = ([f"{p}_wa, {p}_ya = np.linalg.eigh({cname})"] if native_std else _eigh_jacobi_lines(
        f"{p}_wa", f"{p}_ya", cname, f"{a}.shape[0]", p))
    lines = pre + std
    xname = f"{p}_xa" if b is not None else f"{p}_ya"
    if b is not None:
        lines.append(f"{p}_xa = {p}_Li.conj().T @ {p}_ya")
    if lo == "None":  # whole spectrum -> bare name (a ``[None:None]`` slice trips the C lowering)
        lines += [f"{w} = {p}_wa", f"{v} = {xname}"]
    else:
        lines += [f"{w} = {p}_wa[{lo}:{hi}]", f"{v} = {xname}[:, {lo}:{hi}]"]
    return lines


def _eigh_c_stmts(w: str,
                  v: Optional[str],
                  a: str,
                  b: Optional[str],
                  lo: str,
                  hi: str,
                  p: str,
                  eigenvalues_only: bool = False) -> List[str]:
    """Fully self-contained loop lowering of standard/generalized complex-Hermitian
    ``eigh`` for the C/Fortran backends, which have no ``np.linalg`` and no matmul
    lowering for the ``L⁻ᴴ`` conjugate-transpose operand. Emits explicit loops
    only: complex-Hermitian Cholesky ``b = L Lᴴ``, the lower-triangular inverse
    ``L⁻¹`` by forward substitution, the two matmuls ``C = L⁻¹ a L⁻ᴴ``, the cyclic
    complex Jacobi, and the back-transform ``x = L⁻ᴴ y``. Matmul outputs are
    pre-zeroed and ``+=``-accumulated (no complex literal needed); a complex zero
    is ``z - z``. Validated vs scipy ~1e-15.

    ``eigenvalues_only`` (``np.linalg.eigvalsh``) binds only the ascending eigenvalue
    vector ``w`` into a single Name target: the same Jacobi sweep runs, but the
    ``L⁻ᴴ`` back-transform and the ``v`` eigenvector output are dropped (``v`` is
    ``None``). numpy has no generalized eigvalsh, so this path always has ``b`` None."""
    n = f"{a}.shape[0]"
    lines: List[str] = []
    if b is not None:
        L, Li = f"{p}_L", f"{p}_Li"
        lines += _cholesky_lines(L, b, n, f"{p}c", hermitian=True)
        lines += [  # explicit lower-triangular inverse L^-1 by forward substitution
            f"{Li} = np.zeros(({n}, {n}), {b}.dtype)",
            f"for {p}_ij in range({n}):",
            f"    {Li}[{p}_ij, {p}_ij] = 1.0 / {L}[{p}_ij, {p}_ij]",
            f"    for {p}_ii in range({p}_ij + 1, {n}):",
            f"        {p}_acc = {L}[{p}_ii, {p}_ii] - {L}[{p}_ii, {p}_ii]",
            f"        for {p}_ik in range({p}_ij, {p}_ii):",
            f"            {p}_acc += {L}[{p}_ii, {p}_ik] * {Li}[{p}_ik, {p}_ij]",
            f"        {Li}[{p}_ii, {p}_ij] = -{p}_acc / {L}[{p}_ii, {p}_ii]",
        ]
        # ``Tm`` / ``Cm`` (not ``T`` / ``C``): Fortran is case-insensitive, so a
        # matrix named ``T`` would collide with the Jacobi rotation scalar ``t``
        # (tangent) and ``C`` with ``c`` (cosine) -- the emitter would silently
        # drop one declaration and the body would index a scalar.
        T, C = f"{p}_Tm", f"{p}_Cm"
        lines += [  # Tm = Li @ a
            f"{T} = np.zeros(({n}, {n}), {b}.dtype)",
            f"for {p}_ti in range({n}):",
            f"    for {p}_tj in range({n}):",
            f"        for {p}_tk in range({n}):",
            f"            {T}[{p}_ti, {p}_tj] += {Li}[{p}_ti, {p}_tk] * {a}[{p}_tk, {p}_tj]",
        ]
        lines += [  # C = T @ Li^H  (Li^H[k, l] = conj(Li[l, k]))
            f"{C} = np.zeros(({n}, {n}), {b}.dtype)",
            f"for {p}_ci in range({n}):",
            f"    for {p}_cj in range({n}):",
            f"        for {p}_ck in range({n}):",
            f"            {C}[{p}_ci, {p}_cj] += {T}[{p}_ci, {p}_ck] * np.conj({Li}[{p}_cj, {p}_ck])",
        ]
        cname = C
    else:
        # ``Cm`` (not ``C``): Fortran is case-insensitive, so a matrix named ``C``
        # collides with the Jacobi cosine scalar ``c`` -- the emitter would reject
        # the second declaration. (The generalized branch above uses ``Cm`` too.)
        cname = f"{p}_Cm"
        # Explicit ``np.zeros`` allocation + element copy (NOT ``np.ascontiguousarray``,
        # whose copy-loop lowering leaves the fresh RUNTIME-shaped target unallocated --
        # a NULL write in the Jacobi). Mirrors the generalized branch's ``np.zeros``
        # temps. The Jacobi mutates ``Cm`` in place, so the input ``a`` must not alias it.
        lines += [
            f"{cname} = np.zeros(({n}, {n}), {a}.dtype)",
            f"for {p}_ci in range({n}):",
            f"    for {p}_cj in range({n}):",
            f"        {cname}[{p}_ci, {p}_cj] = {a}[{p}_ci, {p}_cj]",
        ]
    lines += _eigh_jacobi_lines(f"{p}_wa", f"{p}_ya", cname, n, p)
    if eigenvalues_only:  # eigvalsh: only the eigenvalue vector, no back-transform / U output
        lines.append(f"{w} = {p}_wa" if lo == "None" else f"{w} = {p}_wa[{lo}:{hi}]")
        return lines
    if b is not None:
        X = f"{p}_X"
        lines += [  # back-transform x = Li^H @ ya
            f"{X} = np.zeros(({n}, {n}), {b}.dtype)",
            f"for {p}_xi in range({n}):",
            f"    for {p}_xj in range({n}):",
            f"        for {p}_xk in range({n}):",
            f"            {X}[{p}_xi, {p}_xj] += np.conj({Li}[{p}_xk, {p}_xi]) * {p}_ya[{p}_xk, {p}_xj]",
        ]
        xname = X
    else:
        xname = f"{p}_ya"
    if lo == "None":  # whole spectrum -> bare name (a ``[None:None]`` slice trips the C lowering)
        lines += [f"{w} = {p}_wa", f"{v} = {xname}"]
    else:
        lines += [f"{w} = {p}_wa[{lo}:{hi}]", f"{v} = {xname}[:, {lo}:{hi}]"]
    return lines


class _EighLoopRewriter(ast.NodeTransformer):
    """Rewrite ``w, v = eigh(a[, b], subset_by_index=[lo, hi])`` (np.linalg /
    scipy.linalg / an imported alias) to the fully self-contained loop lowering
    (:func:`_eigh_c_stmts`) for the C/Fortran frontend, which has no ``np.linalg``.
    Applied to the whole module tree (helpers included) BEFORE kernel inlining, so
    the ``_sci_eigh`` alias import is still in scope. A non-Name operand is
    materialised first."""

    def __init__(self, alias_names: set):
        self.alias_names = alias_names
        self._ctr = 0

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        hit = _eigh_call_kind(node.value, self.alias_names)
        if hit is None:
            return node
        kind, a_node, b_node, kw = hit
        tgt = node.targets[0]
        # ``w, v = eigh(...)`` (eigenpair) or ``w = eigvalsh(...)`` (a single Name
        # target -- eigenvalues only, no eigenvector back-transform / U output).
        if isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2 and all(isinstance(e, ast.Name) for e in tgt.elts):
            w, v = tgt.elts[0].id, tgt.elts[1].id
        elif kind == "eigvalsh" and isinstance(tgt, ast.Name):
            w, v = tgt.id, None
        else:
            return node
        p = f"__eigh{self._ctr}"
        self._ctr += 1
        pre: List[str] = []

        def name_of(nd, tag):
            if isinstance(nd, ast.Name):
                return nd.id
            pre.append(f"{p}_{tag} = np.ascontiguousarray({ast.unparse(nd)})")
            return f"{p}_{tag}"

        aname = name_of(a_node, "a")
        bname = name_of(b_node, "b") if b_node is not None else None
        s = kw.get("subset_by_index")
        if isinstance(s, (ast.List, ast.Tuple)) and len(s.elts) == 2:
            lo, hi = ast.unparse(s.elts[0]), f"({ast.unparse(s.elts[1])}) + 1"
        else:
            lo, hi = "None", "None"
        lines = pre + _eigh_c_stmts(w, v, aname, bname, lo, hi, p, eigenvalues_only=(v is None))
        return [ast.copy_location(st, node) for st in ast.parse("\n".join(lines)).body]


def _eigh_alias_names(tree: ast.AST) -> set:
    """Names that refer to ``scipy.linalg.eigh`` via ``from scipy.linalg import
    eigh [as X]`` (cegterg's ``_sci_eigh``). ``np.linalg.eigh`` / ``scipy.linalg.
    eigh`` attribute calls are recognised separately."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("scipy.linalg", "scipy"):
            for al in node.names:
                base = al.name if node.module == "scipy.linalg" else None
                if base == "eigh" or al.name == "linalg.eigh":
                    out.add(al.asname or al.name)
    return out


def _eigh_call_kind(node: ast.AST, alias_names: set):
    """``eigh(a[, b], ...)`` / ``eigvalsh(a, ...)`` -> ``(kind, a_node,
    b_node_or_None, kwargs)`` for a matching call (``np.linalg`` / ``scipy.linalg``
    / an imported ``eigh`` alias), else None. ``kind`` is ``"eigh"`` (returns an
    eigenpair ``(w, U)``) or ``"eigvalsh"`` (returns only the eigenvalue vector).
    numpy has no generalized ``eigvalsh``, so an ``eigvalsh`` call carries no metric
    ``b`` -- its second positional argument, if any, is ``UPLO`` not an operand."""
    if not isinstance(node, ast.Call) or not node.args:
        return None
    f = node.func
    linalg_attr = _np_linalg_attr(node)
    scipy_attr = (f.attr
                  if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute) and f.value.attr == "linalg"
                  and isinstance(f.value.value, ast.Name) and f.value.value.id == "scipy" else None)
    if linalg_attr == "eigvalsh" or scipy_attr == "eigvalsh":
        kind = "eigvalsh"
    elif linalg_attr == "eigh" or scipy_attr == "eigh" or (isinstance(f, ast.Name) and f.id in alias_names):
        kind = "eigh"
    else:
        return None
    kw = {k.arg: k.value for k in node.keywords}
    a = node.args[0]
    b = None if kind == "eigvalsh" else (node.args[1] if len(node.args) > 1 else kw.get("b"))
    return kind, a, b, kw


def _eigh_call_ab(node: ast.AST, alias_names: set):
    """``eigh(a[, b], ...)`` -> ``(a_node, b_node_or_None, kwargs)`` for a matching
    eigh/eigvalsh call, else None. A thin :func:`_eigh_call_kind` wrapper for callers
    that only need the operands (the jax rewriter, which lowers eigenpairs only)."""
    hit = _eigh_call_kind(node, alias_names)
    return None if hit is None else hit[1:]


def _is_eigh_assign_target(node: ast.AST, alias_names: set) -> bool:
    """``True`` when ``node`` is an assignment :class:`_EighLoopRewriter` lowers
    directly -- ``w, v = eigh(...)`` (eigenpair) or ``w = eigvalsh(...)`` (a single
    Name target). Such an assign must NOT have its RHS call hoisted out first, or the
    rewriter would no longer see the eigh/eigvalsh call as the statement's RHS."""
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
        return False
    hit = _eigh_call_kind(node.value, alias_names)
    if hit is None:
        return False
    tgt = node.targets[0]
    if isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2 and all(isinstance(e, ast.Name) for e in tgt.elts):
        return True
    return hit[0] == "eigvalsh" and isinstance(tgt, ast.Name)


class _EighCallHoister(ast.NodeTransformer):
    """Materialise an ``eigh`` / ``eigvalsh`` call that appears NESTED in an
    expression -- ``float(np.linalg.eigvalsh(T).max()) + beta`` in LS3DF's Lanczos
    upper-bound -- into its own ``__eigv<k> = <call>`` statement, so the direct-assign
    :class:`_EighLoopRewriter` can lower it. A call that is already the RHS of an
    eligible eigh-assign (:func:`_is_eigh_assign_target`) is left in place. Runs on the
    whole module (helpers included) BEFORE the loop rewriter, mirroring its scope."""

    def __init__(self, alias_names: set):
        self.alias_names = alias_names
        self.pre: List[ast.stmt] = []
        self._ctr = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if _eigh_call_kind(node, self.alias_names) is None:
            return node
        self._ctr += 1
        name = f"__eigv{self._ctr}"
        self.pre.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=node))
        return ast.Name(id=name, ctx=ast.Load())

    def _flush(self, node: ast.stmt):
        # An eligible direct eigh-assign stays whole -- descending would hoist its own
        # RHS call and hide it from the loop rewriter.
        if _is_eigh_assign_target(node, self.alias_names):
            return node
        saved = self.pre
        self.pre = []
        self.generic_visit(node)
        pre = self.pre
        self.pre = saved
        if not pre:
            return node
        for s in pre:
            ast.copy_location(s, node)
            ast.fix_missing_locations(s)
        return pre + [node]

    def _visit_stmts(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
        out: List[ast.stmt] = []
        for s in stmts:
            r = self.visit(s)
            if r is None:
                continue
            out.extend(r if isinstance(r, list) else [r])
        return out

    def visit_While(self, node: ast.While) -> ast.AST:
        # Do NOT hoist an eigh call out of the loop CONDITION: a ``__eigv`` temp
        # emitted before the loop would freeze a value the ``while`` test must
        # recompute each iteration (``while eigvalsh(A).max() > tol: A = update(A)``).
        # Leave ``node.test`` unvisited; only the body / else statements hoist locally.
        node.body = self._visit_stmts(node.body)
        node.orelse = self._visit_stmts(node.orelse)
        return node

    visit_Assign = _flush
    visit_AugAssign = _flush
    visit_Expr = _flush
    visit_Return = _flush
    visit_If = _flush
    visit_For = _flush


class _EighInline(ast.NodeTransformer):
    """Lower ``w, v = eigh(a[, b], subset_by_index=[lo, hi])`` -- standard or
    generalized complex-Hermitian ``eigh`` (numpy or scipy, incl. an imported
    alias) -- to a Cholesky-reduced complex Jacobi loop nest (see
    :func:`_eigh_stmts`). Handles the tuple-target eigenpair form and the
    eigenvalues-only single-target form (``np.linalg.eigvalsh`` or
    ``eigh(..., eigvals_only=True)``); a non-``Name`` operand is materialised first.
    Runs BEFORE :class:`_LinalgInline` so the cholesky/inv it emits are themselves
    lowered for pythran."""

    def __init__(self, ranks: Dict[str, int], alias_names: set):
        self.ranks = ranks
        self.alias_names = alias_names
        self.changed = False
        self._ctr = 0

    def _subset(self, kw) -> tuple:
        """``subset_by_index=[lo, hi]`` (inclusive) -> slice bounds ``(lo, hi+1)``
        strings; whole spectrum -> ``('None', 'None')`` (a full ``[:]`` slice)."""
        s = kw.get("subset_by_index")
        if isinstance(s, (ast.List, ast.Tuple)) and len(s.elts) == 2:
            return ast.unparse(s.elts[0]), f"({ast.unparse(s.elts[1])}) + 1"
        return "None", "None"

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            return node
        hit = _eigh_call_kind(node.value, self.alias_names)
        if hit is None:
            return node
        kind, a_node, b_node, kw = hit
        tgt = node.targets[0]
        evo = kw.get("eigvals_only")
        # ``eigvalsh`` (a distinct eigenvalues-only op) and ``eigh(..., eigvals_only=True)``
        # (scipy's flag) both bind a single eigenvalue vector -- no eigenvectors.
        eigvals_only = kind == "eigvalsh" or (isinstance(evo, ast.Constant) and evo.value is True)
        # Target: ``w, v = eigh(...)`` (tuple) or ``w = eigvalsh(...)`` / ``w = eigh(..., eigvals_only=True)``.
        if isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2 and all(isinstance(e, ast.Name) for e in tgt.elts):
            w, v = tgt.elts[0].id, tgt.elts[1].id
        elif isinstance(tgt, ast.Name) and eigvals_only:
            w, v = tgt.id, None
        else:
            return node
        if _expr_rank(a_node, self.ranks) not in (2, None) or (b_node is not None
                                                               and _expr_rank(b_node, self.ranks) not in (2, None)):
            return node
        p = f"__eigh{self._ctr}"
        self._ctr += 1
        pre: List[str] = []

        def name_of(nd, tag):
            if isinstance(nd, ast.Name):
                return nd.id
            pre.append(f"{p}_{tag} = np.ascontiguousarray({ast.unparse(nd)})")
            return f"{p}_{tag}"

        aname = name_of(a_node, "a")
        bname = name_of(b_node, "b") if b_node is not None else None
        lo, hi = self._subset(kw)
        vtmp = v if v is not None else f"{p}_vdrop"
        lines = pre + _eigh_stmts(w, vtmp, aname, bname, lo, hi, p)
        self.changed = True
        return [ast.copy_location(s, node) for s in ast.parse("\n".join(lines)).body]

    def visit_AugAssign(self, node):
        return node

    def visit_Return(self, node):
        return node

    def visit_Expr(self, node):
        return node


#: numpy.linalg ops each verbatim-body backend compiles NATIVELY (left in place);
#: any op NOT listed for the target backend is lowered to explicit loops by the
#: desugar. numba (numba.np.linalg) and dace (dace.libraries.linalg replacements)
#: implement cholesky/solve/inv directly; pythran has no numpy.linalg at all.
_LINALG_LOWERABLE = {"cholesky", "solve", "inv"}
_NATIVE_LINALG: Dict[Optional[str], set] = {
    "numba": {"cholesky", "solve", "inv"},
    "dace": {"cholesky", "solve", "inv"},
    "pythran": set(),
}


def _param_body_rank_evidence(fn: ast.FunctionDef) -> Dict[str, int]:
    """Lower bounds on a helper's param ranks from how the BODY uses each param,
    independent of call sites: ``p.shape[k]`` implies rank >= k+1, and a
    multi-axis subscript ``p[:, a:b, c:d, :]`` implies rank >= (non-newaxis index
    count). This is flow-insensitive-proof -- a call site that passes a local
    which was later reshaped to a smaller rank poisons the call-site inference
    (lenet reshapes ``x`` from 4-D to 2-D), but the body's own ``x.shape[3]`` /
    4-slice index pins the true rank."""
    params = {a.arg for a in fn.args.args}
    ev: Dict[str, int] = {}

    def bump(name: str, r: int) -> None:
        if name in params and r > ev.get(name, 0):
            ev[name] = r

    for node in ast.walk(fn):
        if isinstance(node, ast.Subscript):
            v = node.value
            if (isinstance(v, ast.Attribute) and v.attr == "shape" and isinstance(v.value, ast.Name)):
                k = _const_int(node.slice)
                if k is not None and k >= 0:
                    bump(v.value.id, k + 1)  # p.shape[k] -> rank >= k+1
            elif isinstance(v, ast.Name) and isinstance(node.slice, ast.Tuple):
                bump(v.id, sum(0 if _is_newaxis(e) else 1 for e in node.slice.elts))
    return ev


def _return_rank(fn: ast.FunctionDef, ranks: Dict[str, int]) -> Optional[int]:
    """Rank of ``fn``'s returned value (the max over its ``return`` statements),
    given a rank table for its body -- so a caller can propagate it."""
    rs = [_expr_rank(n.value, ranks) for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    rs = [r for r in rs if r is not None]
    return max(rs) if rs else None


def _infer_param_ranks(funcs: List[ast.FunctionDef], kernel_name: str,
                       kir_seed: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    """Per-function ``{param: ndim}`` seeds. The kernel's array params come from
    ``kir_seed``; a HELPER function's param ranks are inferred from its call sites
    -- ``getAcc(pos, ...)`` in the kernel tells ``getAcc`` that ``pos`` has the
    kernel's rank for ``pos`` -- unified with body-usage lower bounds
    (:func:`_param_body_rank_evidence`). Ranks merge by MAX: a param used at rank
    R somewhere is at least rank R, and a conflicting smaller value only ever comes
    from a flow-insensitive rank-table mix-up (a reshaped local passed to a
    helper), never from the param's real dimensionality. Iterated to a fixpoint so
    a helper calling another helper also resolves."""
    by_name = {fn.name: fn for fn in funcs}
    params = {fn.name: [a.arg for a in fn.args.args] for fn in funcs}
    seeds: Dict[str, Dict[str, int]] = {fn.name: dict(_param_body_rank_evidence(fn)) for fn in funcs}
    ret_rank: Dict[str, int] = {}
    for _ in range(6):
        changed = False
        for fn in funcs:
            base = dict(seeds[fn.name])
            if fn.name == kernel_name:
                base.update(kir_seed)
            ranks = _rank_table(fn, base, call_returns=ret_rank)
            rr = _return_rank(fn, ranks)
            if rr is not None and ret_rank.get(fn.name) != rr:
                ret_rank[fn.name] = rr
                changed = True
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in by_name:
                    callee = node.func.id
                    for i, arg in enumerate(node.args):
                        if i >= len(params[callee]):
                            break
                        r = _expr_rank(arg, ranks)
                        if r is None:
                            r = _call_return_rank(arg, ret_rank)
                        pname = params[callee][i]
                        if r is not None and r > seeds[callee].get(pname, -1):
                            seeds[callee][pname] = r
                            changed = True
        if not changed:
            break
    return seeds


class _DecomposeRollSlice(ast.NodeTransformer):
    """``T = np.roll(O, shift, axis)`` where the operand ``O`` or target ``T`` is a
    SLICE / subscript (not a bare array name) -- decompose into bare-name temps so
    the native ``expand_roll`` (which needs a bare Name) applies, and a sliced
    self-roll ``X[..] = np.roll(X[..], ..)`` reads a SNAPSHOT (the temp) so the
    in-place write is safe. numpy and the Python backends roll a slice verbatim, so
    this is native-only (the band-group circular shift in QE vexx negrp>1)."""

    def __init__(self):
        self.changed = False
        self._n = 0

    def _fresh(self) -> str:
        self._n += 1
        return f"__roll_{self._n}"

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        v = node.value
        # Require a POSITIONAL shift (args[0]=array, args[1]=shift): the native
        # ``expand_roll`` reads the shift from args[1], so a keyword ``shift=`` roll
        # is not lowerable -- don't decompose it into a dead snapshot temp, leave it
        # to fail loudly unchanged.
        if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "roll"
                and isinstance(v.func.value, ast.Name) and v.func.value.id in ("np", "numpy") and len(v.args) >= 2
                and len(node.targets) == 1):
            return node
        target = node.targets[0]
        op_bare = isinstance(v.args[0], ast.Name)
        tgt_bare = isinstance(target, ast.Name)
        if op_bare and tgt_bare:
            return node  # expand_roll handles the bare-Name form directly
        out: List[ast.stmt] = []
        if not op_bare:  # snapshot a sliced operand into a bare-name temp
            src = self._fresh()
            out.append(ast.Assign(targets=[ast.Name(id=src, ctx=ast.Store())], value=v.args[0]))
            v.args[0] = ast.Name(id=src, ctx=ast.Load())
        if tgt_bare:
            out.append(node)  # target bare -> roll writes it directly
        else:  # roll into a bare temp, then copy back to the sliced target
            dst = self._fresh()
            out.append(ast.Assign(targets=[ast.Name(id=dst, ctx=ast.Store())], value=v))
            out.append(ast.Assign(targets=[target], value=ast.Name(id=dst, ctx=ast.Load())))
        for s in out:
            ast.copy_location(s, node)
        self.changed = True
        return out


class _ComplexAccessorToFunc(ast.NodeTransformer):
    """Canonicalise every complex-accessor spelling to its ``np.*`` function form:
    ``z.real`` -> ``np.real(z)``, ``z.imag`` -> ``np.imag(z)``, ``z.conjugate()`` /
    ``z.conj()`` -> ``np.conj(z)``. One canonical spelling means one native emit
    handler per op (``creal`` / ``cimag`` / ``conj``) instead of parallel attribute,
    method, and function paths, and the Python backends run the standard ``np.*``
    ufuncs. Only the three complex accessors are rewritten -- ``.shape`` / ``.size``
    / ``.T`` / ``.dtype`` and every other attribute pass through untouched.

    ``conjugate_only`` restricts the rewrite to the ``.conjugate()`` / ``.conj()``
    method (used for the Python backends, which already run ``.real`` / ``.imag``
    verbatim but whose pythran path lacks the ``.conjugate()`` method)."""

    def __init__(self, conjugate_only: bool = False):
        self.changed = False
        self.conjugate_only = conjugate_only

    def _np_call(self, fn: str, arg: ast.expr) -> ast.expr:
        self.changed = True
        return ast.copy_location(
            ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=fn, ctx=ast.Load()),
                     args=[arg],
                     keywords=[]), arg)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # ``x.conjugate()`` / ``x.conj()`` (no args) -> ``np.conj(x)``. Handled at
        # the Call so ``.conjugate`` is not first mistaken for an accessor below;
        # ``np.conj(x)`` (a real call with args) is left as-is.
        if (isinstance(node.func, ast.Attribute) and not node.args and not node.keywords
                and node.func.attr in ("conjugate", "conj")):
            return self._np_call("conj", self.visit(node.func.value))
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        # ``x.real`` / ``x.imag`` accessor -> function form; but NOT the ``np.real``
        # / ``np.imag`` module attribute (that IS the function -- rewriting it would
        # nest ``np.real(np)``).
        if (not self.conjugate_only and isinstance(node.ctx, ast.Load) and node.attr in ("real", "imag")
                and not (isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy"))):
            return self._np_call(node.attr, node.value)
        return node


def _np_multi_call(fn: str, args: List[ast.expr]) -> ast.Call:
    """Build ``np.<fn>(*args)`` (the multi-argument sibling of
    ``_ComplexAccessorToFunc._np_call``)."""
    return ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=fn, ctx=ast.Load()),
                    args=args,
                    keywords=[])


def _cmp_zero(x: ast.expr, op: ast.cmpop) -> ast.Compare:
    """Build ``x <op> 0`` (heaviside's sign test against zero)."""
    return ast.Compare(left=x, ops=[op], comparators=[ast.Constant(value=0)])


#: ``np.<ufunc>.reduce`` -> the plain reducer it equals. A DaCe ``Reduce`` library node re-emits
#: reductions in ufunc form (``np.add.reduce(x, axis=k)``); the native backends have no ufunc
#: dispatch, so canonicalise to the reducer the loop-lowering already handles.
_UFUNC_REDUCE_TO_CALL = {
    "add": "sum",
    "multiply": "prod",
    "maximum": "max",
    "minimum": "min",
    "logical_and": "all",
    "logical_or": "any",
}


class _UfuncReduceToReducer(ast.NodeTransformer):
    """``np.add.reduce(x, axis=k)`` -> ``np.sum(x, axis=k)`` (and the prod / max / min / all / any
    ufuncs). ``ufunc.reduce`` defaults to ``axis=0`` while the reducer defaults to ``axis=None``
    (a full reduction), so inject an explicit ``axis=0`` when the call gave none -- preserving the
    ufunc semantics. Runs before the elementwise-ufunc desugars so ``np.add`` inside ``np.add.reduce``
    is never mistaken for an elementwise add."""

    def __init__(self):
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr == "reduce" and isinstance(f.value, ast.Attribute)
                and isinstance(f.value.value, ast.Name) and f.value.value.id in ("np", "numpy")
                and f.value.attr in _UFUNC_REDUCE_TO_CALL):
            self.changed = True
            has_axis = len(node.args) > 1 or any(k.arg == "axis" for k in node.keywords)
            keywords = list(node.keywords)
            if not has_axis:
                keywords.append(ast.keyword(arg="axis", value=ast.Constant(value=0)))
            return ast.copy_location(
                ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                                            attr=_UFUNC_REDUCE_TO_CALL[f.value.attr],
                                            ctx=ast.Load()),
                         args=node.args,
                         keywords=keywords), node)
        return node


class _ElementalUfuncToPrimitive(ast.NodeTransformer):
    """Rewrite the two-argument elemental numpy ufuncs that lack a direct native /
    JIT lowering into equivalent expressions over already-supported primitives, so
    every backend (C / C++ / Fortran + numba / pythran / jax) lowers them uniformly
    and the array / slice forms scalarise through the normal elementwise expander:

      * ``np.mod(a, b)`` / ``np.remainder(a, b)`` -> ``a % b`` -- numpy's floored
        modulo is exactly the ``%`` operator (result takes the sign of the divisor).
      * ``np.logaddexp(a, b)`` -> ``np.maximum(a, b) + np.log(1.0 + np.exp(-np.abs(a - b)))``
        -- numpy's stable log-sum-exp (``npy_logaddexp``). ``log1p`` would be the exact
        spelling but has no Fortran intrinsic; here ``exp(-|a-b|)`` is in ``(0, 1]`` so
        ``log(1 + .)`` is well-conditioned and agrees with numpy to a few ulp.
      * ``np.heaviside(a, b)`` -> ``np.where(a < 0, 0.0, np.where(a == 0, b, 1.0))``
        -- 0 below zero, the second arg exactly at zero, 1 above.

    numba has no ``np.heaviside`` and pythran no ``np.logaddexp``; expanding to the
    shared primitives every backend supports is the single uniform lowering, with no
    per-backend special-casing. Reused operands are deep-copied so no AST node is
    shared between two positions."""

    def __init__(self):
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        f = node.func
        if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("np", "numpy")
                and len(node.args) == 2 and not node.keywords):
            return node
        a, b = node.args
        if f.attr in ("mod", "remainder"):
            self.changed = True
            return ast.copy_location(ast.BinOp(left=a, op=ast.Mod(), right=b), node)
        if f.attr == "logaddexp":
            self.changed = True
            diff = ast.BinOp(left=a, op=ast.Sub(), right=copy.deepcopy(b))
            expterm = _np_multi_call("exp", [ast.UnaryOp(op=ast.USub(), operand=_np_multi_call("abs", [diff]))])
            onep = ast.BinOp(left=ast.Constant(value=1.0), op=ast.Add(), right=expterm)
            tail = _np_multi_call("log", [onep])
            head = _np_multi_call("maximum", [copy.deepcopy(a), copy.deepcopy(b)])
            return ast.copy_location(ast.BinOp(left=head, op=ast.Add(), right=tail), node)
        if f.attr == "heaviside":
            self.changed = True
            inner = _np_multi_call("where", [_cmp_zero(copy.deepcopy(a), ast.Eq()), b, ast.Constant(value=1.0)])
            outer = _np_multi_call("where", [_cmp_zero(a, ast.Lt()), ast.Constant(value=0.0), inner])
            return ast.copy_location(outer, node)
        return node


def desugar_for_python_backend(source: str, kir, backend: Optional[str] = None) -> str:
    """Rewrite ``source`` so numba / pythran / dace can compile it: expand the
    numpy ops they do not support (batched ``@`` / ``np.matmul``, ``np.pad``,
    ``np.einsum``, ``np.fft.*``, ``np.mgrid``, axis reductions, ufunc.outer,
    multi-array fancy gather, 2-D boolean-mask assignment, ``np.ndarray`` /
    ``np.linspace(dtype=)`` / ``abs(array)``) into plain loops / broadcasts /
    ``np.where``. EVERY function in the module is processed (helpers too -- nbody's
    masked updates live in getAcc/getEnergy), each with its own rank table seeded
    from the kernel arrays (kir) or inferred call-site param ranks. Every pass is
    pattern-guarded and the original ``source`` is returned byte-for-byte when none
    fire, so a kernel that needs no rewrite keeps its verbatim body.

    ``backend`` selects the target's native-``np.linalg`` capability: an op the
    backend implements natively (numba / dace do cholesky/solve/inv) is left
    verbatim; one it lacks (pythran has no np.linalg) is lowered to explicit loops.
    ``None`` (the default) lowers no linalg -- the safe backwards-compatible base."""
    lower_linalg = _LINALG_LOWERABLE - _NATIVE_LINALG.get(backend, _LINALG_LOWERABLE)
    tree = ast.parse(source)
    all_funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    kir_seed: Dict[str, int] = {a.name: len(a.shape) for a in kir.arrays}
    kir_dtype_seed: Dict[str, str] = {
        a.name: _kind_of_dtype_str(vars(a).get("dtype"))
        for a in kir.arrays if _kind_of_dtype_str(vars(a).get("dtype"))
    }
    param_ranks = _infer_param_ranks(all_funcs, kir.kernel_name, kir_seed)
    eigh_aliases = _eigh_alias_names(tree)
    changed = False
    for fn in (all_funcs or [tree]):
        is_kernel = getattr(fn, "name", None) == kir.kernel_name
        seed = dict(param_ranks.get(getattr(fn, "name", None), {}))
        if is_kernel:
            seed.update(kir_seed)
        ranks = _rank_table(fn, seed)
        dtypes = _dtype_table(fn, kir_dtype_seed if is_kernel else {})
        noncontig = _noncontig_names(fn)
        masked_gathers = _masked_reduce_map(fn, ranks, dtypes)
        passes = [
            _DropGuards(),
            _NormalizeNegativeAxis(ranks),
            _EighInline(ranks, eigh_aliases),
            _LinalgInline(ranks, dtypes, lower_linalg),
            _ReshapeMatmulInline(ranks),
            _BatchedMatmulToLoop(ranks),
            _PadInline(ranks),
            _EinsumInline(),
            _FftInline(ranks),
            _MgridInline(),
            _FancyGatherInline(ranks),
            _ReduceAxisInline(ranks, dtypes),
            _MaskedReduceInline(masked_gathers, ranks),
            _CallFixups(ranks),
            _IssubdtypeFold(dtypes),
            _DeadBranchElim(),
            _UfuncOuterInline(ranks),
            _MaskedAssignToLoop(ranks, dtypes),
            _AddAtInline(ranks),
            _HistogramInline(ranks),
            _RepeatAxisInline(ranks),
            _ReshapeContiguousInline(noncontig),
            _IntMatmulInline(ranks, dtypes),
            _ComplexAccessorToFunc(conjugate_only=True),
            _ElementalUfuncToPrimitive(),
        ]
        for p in passes:
            # Process THIS scope's own statements only; a nested def is its own
            # scope (its params carry different ranks) and is handled as its own
            # entry in ``all_funcs``, so skip it here to avoid a wrong-rank pass.
            new_body = []
            for stmt in fn.body:
                if isinstance(stmt, ast.FunctionDef):
                    new_body.append(stmt)
                    continue
                res = p.visit(stmt)
                if res is None:
                    continue
                new_body.extend(res if isinstance(res, list) else [res])
            fn.body = new_body
        changed = changed or any(p.changed for p in passes)
    if not changed:
        return source  # nothing matched -> leave the body verbatim
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
