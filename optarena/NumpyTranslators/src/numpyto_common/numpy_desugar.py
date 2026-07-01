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

# Constructors whose first arg is a shape tuple -> result rank = len(shape).
_SHAPE_CTORS = {"empty", "zeros", "ones", "full", "ndarray"}
_LIKE_CTORS = {"empty_like", "zeros_like", "ones_like"}


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
    Compare / ``& | ^ ~`` logical combo of that rank."""

    def __init__(self, ranks: Dict[str, int]):
        self.ranks = ranks
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
            if self.ranks.get(idx.id) != arank:
                return node  # an int/index Name (not a full-shape bool mask)
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
        pre, idx_exprs, first_arr = [], [], None
        for j, e in enumerate(elts):
            if (_expr_rank(e, self.ranks) or 0) >= 1:
                t = f"{p}_x{j}"
                pre.append(f"{t} = {ast.unparse(e)}")
                idx_exprs.append(f"{t}[{it}]")
                first_arr = first_arr or t
            else:
                idx_exprs.append(ast.unparse(e))
        if vals is None:
            rhs = "1"
        else:
            tv = f"{p}_v"
            pre.append(f"{tv} = {ast.unparse(vals)}")
            rhs = tv if not (_expr_rank(vals, self.ranks) or 0) >= 1 else f"{tv}[{it}]"
        lines, deepen = list(pre), ""
        for k in range(driver_rank):
            lines.append(f"{deepen}for {iters[k]} in range({first_arr}.shape[{k}]):")
            deepen += "    "
        lines.append(f"{deepen}{A}[{', '.join(idx_exprs)}] {_AT_OPS[op]} {rhs}")
        self.changed = True
        return [ast.copy_location(s, node) for s in ast.parse("\n".join(lines)).body]


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


def desugar_for_python_backend(source: str, kir) -> str:
    """Rewrite ``source`` so numba / pythran can compile it: expand the numpy
    ops they do not support (batched ``@`` / ``np.matmul``, ``np.pad``,
    ``np.einsum``, ``np.fft.*``, ``np.mgrid``, axis reductions, ufunc.outer,
    multi-array fancy gather, 2-D boolean-mask assignment, ``np.ndarray`` /
    ``np.linspace(dtype=)`` / ``abs(array)``) into plain loops / broadcasts /
    ``np.where``. EVERY function in the module is processed (helpers too -- nbody's
    masked updates live in getAcc/getEnergy), each with its own rank table seeded
    from the kernel arrays (kir) or inferred call-site param ranks. Every pass is
    pattern-guarded and the original ``source`` is returned byte-for-byte when none
    fire, so a kernel that needs no rewrite keeps its verbatim body."""
    tree = ast.parse(source)
    all_funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    kir_seed: Dict[str, int] = {a.name: len(a.shape) for a in kir.arrays}
    param_ranks = _infer_param_ranks(all_funcs, kir.kernel_name, kir_seed)
    changed = False
    for fn in (all_funcs or [tree]):
        seed = dict(param_ranks.get(getattr(fn, "name", None), {}))
        if getattr(fn, "name", None) == kir.kernel_name:
            seed.update(kir_seed)
        ranks = _rank_table(fn, seed)
        passes = [
            _BatchedMatmulToLoop(ranks),
            _PadInline(ranks),
            _EinsumInline(),
            _FftInline(ranks),
            _MgridInline(),
            _FancyGatherInline(ranks),
            _ReduceAxisInline(ranks),
            _CallFixups(ranks),
            _UfuncOuterInline(ranks),
            _MaskedAssignToLoop(ranks),
            _AddAtInline(ranks),
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
