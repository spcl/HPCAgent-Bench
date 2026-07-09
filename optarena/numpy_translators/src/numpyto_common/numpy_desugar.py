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
from typing import Dict, List, Optional


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
        if isinstance(sl, ast.Tuple):
            # slices keep a dim, newaxis adds one, an integer/array index removes one.
            drop = 0
            for e in sl.elts:
                if isinstance(e, ast.Slice):
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


def _rank_table(tree: ast.AST, seed: Dict[str, int]) -> Dict[str, int]:
    """Propagate ndim across straight-line assignments to a fixpoint."""
    ranks = dict(seed)
    for _ in range(8):
        changed = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
                r = _expr_rank(node.value, ranks)
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
_REDUCE_AXIS_OPS = {"mean", "std", "var", "min", "max", "amin", "amax", "argmin", "argmax", "any", "all"}


def _reduce_axis_stmts(tname: str, sname: str, op: str, ax: int, rank: int, ctr: int) -> List[ast.stmt]:
    """Source statements reducing ``sname`` over axis ``ax`` into a freshly
    allocated ``tname`` (rank-1+ result) via an explicit loop nest -- the
    numba/pythran-compatible form of ``np.mean/min/max/argmin/argmax(x, axis=k)``
    (numba rejects ``axis=`` on these). Output indices iterate every axis but
    ``ax``; the reduction iterator runs over ``ax``. Mean accumulates then
    divides (matches the C/Fortran sequential lowering); min/max/arg* compare."""
    p = f"__rd{ctr}"
    out_axes = [i for i in range(rank) if i != ax]
    d = [f"{p}_d{i}" for i in range(rank)]
    lines: List[str] = [f"{d[i]} = {sname}.shape[{i}]" for i in range(rank)]
    is_arg = op in ("argmin", "argmax")
    dtype = ("np.int64" if is_arg else
             "np.bool_" if op in ("any", "all") else "np.float64" if op in ("mean", "std", "var") else f"{sname}.dtype")
    lines.append(f"{tname} = np.empty(({''.join(d[i] + ', ' for i in out_axes)}), {dtype})")
    o = {i: f"{p}_k{i}" for i in out_axes}
    ind = ""
    for i in out_axes:
        lines.append(f"{ind}for {o[i]} in range({d[i]}):")
        ind += "    "
    tgt = f"{tname}[{', '.join(o[i] for i in out_axes)}]"

    def elem(jexpr):
        return f"{sname}[{', '.join((jexpr if i == ax else o[i]) for i in range(rank))}]"

    if op in ("any", "all"):
        lines.append(f"{ind}{tgt} = {'True' if op == 'all' else 'False'}")
        lines.append(f"{ind}for {p}_j in range({d[ax]}):")
        if op == "all":
            lines.append(f"{ind}    if not {elem(p + '_j')}:")
            lines.append(f"{ind}        {tgt} = False")
        else:
            lines.append(f"{ind}    if {elem(p + '_j')}:")
            lines.append(f"{ind}        {tgt} = True")
    elif op == "mean":
        lines.append(f"{ind}{tgt} = 0.0")
        lines.append(f"{ind}for {p}_j in range({d[ax]}):")
        lines.append(f"{ind}    {tgt} += {elem(p + '_j')}")
        lines.append(f"{ind}{tgt} = {tgt} / {d[ax]}")
    elif op in ("var", "std"):
        # two-pass (mean, then sum of squared deviations / N, ddof=0) -- matches
        # numpy's np.var/np.std default and the C/Fortran sequential lowering.
        m, dv = f"{p}_m", f"{p}_dv"
        lines.append(f"{ind}{m} = 0.0")
        lines.append(f"{ind}for {p}_j in range({d[ax]}):")
        lines.append(f"{ind}    {m} += {elem(p + '_j')}")
        lines.append(f"{ind}{m} = {m} / {d[ax]}")
        lines.append(f"{ind}{tgt} = 0.0")
        lines.append(f"{ind}for {p}_j in range({d[ax]}):")
        lines.append(f"{ind}    {dv} = {elem(p + '_j')} - {m}")
        lines.append(f"{ind}    {tgt} += {dv} * {dv}")
        lines.append(f"{ind}{tgt} = {tgt} / {d[ax]}")
        if op == "std":
            lines.append(f"{ind}{tgt} = np.sqrt({tgt})")
    elif op in ("min", "amin", "max", "amax"):
        cmp = "<" if op in ("min", "amin") else ">"
        lines.append(f"{ind}{tgt} = {elem('0')}")
        lines.append(f"{ind}for {p}_j in range(1, {d[ax]}):")
        lines.append(f"{ind}    if {elem(p + '_j')} {cmp} {tgt}:")
        lines.append(f"{ind}        {tgt} = {elem(p + '_j')}")
    else:  # argmin / argmax -- track best value + its index
        cmp = "<" if op == "argmin" else ">"
        best = f"{p}_best"
        lines.append(f"{ind}{best} = {elem('0')}")
        lines.append(f"{ind}{tgt} = 0")
        lines.append(f"{ind}for {p}_j in range(1, {d[ax]}):")
        lines.append(f"{ind}    if {elem(p + '_j')} {cmp} {best}:")
        lines.append(f"{ind}        {best} = {elem(p + '_j')}")
        lines.append(f"{ind}        {tgt} = {p}_j")
    return ast.parse("\n".join(lines)).body


class _ReduceAxisHoister(ast.NodeTransformer):
    """Replace each ``np.mean/min/max/argmin/argmax/any/all(x, axis=<int>)`` --
    OR the method form ``x.mean(axis=<int>)`` (velocity's ``levmask.any(axis=0)``)
    -- inside one statement with a fresh temp Name, accumulating its reduction
    loop in ``self.pre``. A non-Name ``x`` (bellman_ford's ``dist[:, None] +
    graph``) is hoisted to a temp first; a non-constant axis or a rank<2
    (scalar-result) reduction is left verbatim (numba's no-axis scalar form)."""

    def __init__(self, ranks: Dict[str, int], ctr: int):
        self.ranks = ranks
        self.ctr = ctr
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
        if not (isinstance(ax, ast.Constant) and isinstance(ax.value, int)):
            return node
        rank = _expr_rank(arg, self.ranks)
        if rank is None or rank < 2:
            return node
        if isinstance(arg, ast.Name):
            sname = arg.id
        else:
            sname = f"__rsrc{self.ctr}"
            self.pre.extend(ast.parse(f"{sname} = {ast.unparse(arg)}").body)
        temp = f"__rdo{self.ctr}"
        self.pre.extend(_reduce_axis_stmts(temp, sname, op, ax.value % rank, rank, self.ctr))
        self.ctr += 1
        return ast.copy_location(ast.Name(id=temp, ctx=ast.Load()), node)


class _ReduceAxisInline(ast.NodeTransformer):
    """Hoist axis reductions out of any value-bearing statement into preceding
    reduction loops (handles ``V = np.max(s, axis=0) + e`` and the bare
    ``mean = np.mean(data, axis=0)``)."""

    def __init__(self, ranks: Dict[str, int]):
        self.ranks = ranks
        self.changed = False
        self._ctr = 0

    def _hoist(self, node):
        if getattr(node, "value", None) is None:
            return node
        h = _ReduceAxisHoister(self.ranks, self._ctr)
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
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in self.gathers and isinstance(node.value, ast.Subscript)):
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
    "add": ast.Add, "subtract": ast.Sub, "multiply": ast.Mult, "divide": ast.Div,
    "true_divide": ast.Div, "power": ast.Pow, "floor_divide": ast.FloorDiv,
    "remainder": ast.Mod, "mod": ast.Mod,
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
        if (not node.orelse and node.body
                and all(isinstance(s, (ast.Raise, ast.Assert, ast.Pass)) for s in node.body)):
            return None
        return node


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
        self._emit([f"{p}_aw = {an}.copy()", f"{temp} = {bn}.copy()"]
                   + _gauss_jordan_lines(f"{p}_aw", temp, f"{an}.shape[0]",
                                         (f"{bn}.shape[1]" if rb == 2 else None), p))
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
            f"{p}_aw = {an}.copy()", f"{temp} = np.zeros(({n}, {n}), {an}.dtype)",
            f"for {p}_d in range({n}):", f"    {temp}[{p}_d, {p}_d] = 1"
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


def _eigh_stmts(w: str, v: str, a: str, b: Optional[str], lo: str, hi: str, p: str,
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


def _eigh_c_stmts(w: str, v: str, a: str, b: Optional[str], lo: str, hi: str, p: str) -> List[str]:
    """Fully self-contained loop lowering of standard/generalized complex-Hermitian
    ``eigh`` for the C/Fortran backends, which have no ``np.linalg`` and no matmul
    lowering for the ``L⁻ᴴ`` conjugate-transpose operand. Emits explicit loops
    only: complex-Hermitian Cholesky ``b = L Lᴴ``, the lower-triangular inverse
    ``L⁻¹`` by forward substitution, the two matmuls ``C = L⁻¹ a L⁻ᴴ``, the cyclic
    complex Jacobi, and the back-transform ``x = L⁻ᴴ y``. Matmul outputs are
    pre-zeroed and ``+=``-accumulated (no complex literal needed); a complex zero
    is ``z - z``. Validated vs scipy ~1e-15."""
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
        cname = f"{p}_C"
        lines += [f"{cname} = np.ascontiguousarray({a})"]
    lines += _eigh_jacobi_lines(f"{p}_wa", f"{p}_ya", cname, n, p)
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
        hit = _eigh_call_ab(node.value, self.alias_names)
        if hit is None:
            return node
        a_node, b_node, kw = hit
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2 and all(isinstance(e, ast.Name) for e in tgt.elts)):
            return node
        w, v = tgt.elts[0].id, tgt.elts[1].id
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
        lines = pre + _eigh_c_stmts(w, v, aname, bname, lo, hi, p)
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


def _eigh_call_ab(node: ast.AST, alias_names: set):
    """``eigh(a[, b], ...)`` -> ``(a_node, b_node_or_None, kwargs)`` for a matching
    eigh call (``np.linalg.eigh`` / ``scipy.linalg.eigh`` / an imported alias),
    else None."""
    if not isinstance(node, ast.Call) or not node.args:
        return None
    f = node.func
    is_eigh = (_np_linalg_attr(node) == "eigh" or (isinstance(f, ast.Name) and f.id in alias_names) or
               (isinstance(f, ast.Attribute) and f.attr == "eigh" and isinstance(f.value, ast.Attribute)
                and f.value.attr == "linalg" and isinstance(f.value.value, ast.Name) and f.value.value.id == "scipy"))
    if not is_eigh:
        return None
    kw = {k.arg: k.value for k in node.keywords}
    a = node.args[0]
    b = node.args[1] if len(node.args) > 1 else kw.get("b")
    return a, b, kw


class _EighInline(ast.NodeTransformer):
    """Lower ``w, v = eigh(a[, b], subset_by_index=[lo, hi])`` -- standard or
    generalized complex-Hermitian ``eigh`` (numpy or scipy, incl. an imported
    alias) -- to a Cholesky-reduced complex Jacobi loop nest (see
    :func:`_eigh_stmts`). Handles the tuple-target eigenpair form and the
    ``eigvals_only=True`` single-target form; a non-``Name`` operand is
    materialised first. Runs BEFORE :class:`_LinalgInline` so the cholesky/inv it
    emits are themselves lowered for pythran."""

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
        hit = _eigh_call_ab(node.value, self.alias_names)
        if hit is None:
            return node
        a_node, b_node, kw = hit
        tgt = node.targets[0]
        evo = kw.get("eigvals_only")
        eigvals_only = isinstance(evo, ast.Constant) and evo.value is True
        # Target: ``w, v = eigh(...)`` (tuple) or ``w = eigh(..., eigvals_only=True)``.
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


def _infer_param_ranks(funcs: List[ast.FunctionDef], kernel_name: str,
                       kir_seed: Dict[str, int]) -> Dict[str, Dict[str, int]]:
    """Per-function ``{param: ndim}`` seeds. The kernel's array params come from
    ``kir_seed``; a HELPER function's param ranks are inferred from its call sites
    -- ``getAcc(pos, ...)`` in the kernel tells ``getAcc`` that ``pos`` has the
    kernel's rank for ``pos``. Iterated to a fixpoint so a helper calling another
    helper also resolves (nbody's masked ops live in getAcc/getEnergy)."""
    by_name = {fn.name: fn for fn in funcs}
    params = {fn.name: [a.arg for a in fn.args.args] for fn in funcs}
    seeds: Dict[str, Dict[str, int]] = {name: {} for name in by_name}
    for _ in range(4):
        changed = False
        for fn in funcs:
            base = dict(seeds[fn.name])
            if fn.name == kernel_name:
                base.update(kir_seed)
            ranks = _rank_table(fn, base)
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in by_name:
                    callee = node.func.id
                    for i, arg in enumerate(node.args):
                        if i >= len(params[callee]):
                            break
                        r = _expr_rank(arg, ranks)
                        pname = params[callee][i]
                        if r is not None and seeds[callee].get(pname) != r:
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
                and isinstance(v.func.value, ast.Name) and v.func.value.id in ("np", "numpy")
                and len(v.args) >= 2 and len(node.targets) == 1):
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
                     args=[arg], keywords=[]), arg)

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
            _EighInline(ranks, eigh_aliases),
            _LinalgInline(ranks, dtypes, lower_linalg),
            _ReshapeMatmulInline(ranks),
            _BatchedMatmulToLoop(ranks),
            _PadInline(ranks),
            _EinsumInline(),
            _FftInline(ranks),
            _MgridInline(),
            _FancyGatherInline(ranks),
            _ReduceAxisInline(ranks),
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
