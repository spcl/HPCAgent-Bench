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


def _tuple_len(node: ast.AST) -> Optional[int]:
    if isinstance(node, (ast.Tuple, ast.List)):
        return len(node.elts)
    return None


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
    if isinstance(value, ast.Subscript):
        base = _expr_rank(value.value, ranks)
        if base is None:
            return None
        sl = value.slice
        if isinstance(sl, ast.Slice):
            return base  # a full/partial slice keeps the rank
        if isinstance(sl, ast.Tuple):
            # rank drops by the number of integer (non-slice) indices.
            drop = sum(0 if isinstance(e, ast.Slice) else 1 for e in sl.elts)
            return base - drop
        return base - 1  # single integer/Name index
    if isinstance(value, ast.Call):
        attr = _np_attr(value)
        if attr in _SHAPE_CTORS and value.args:
            n = _tuple_len(value.args[0])
            if n is not None:
                return n
            if isinstance(value.args[0], (ast.Name, ast.Constant)):
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


def desugar_for_python_backend(source: str, kir) -> str:
    """Rewrite ``source`` so numba / pythran can compile it: expand the numpy
    ops they do not support (currently batched ``@`` / ``np.matmul``) into plain
    loops. Returns the rewritten source; a no-op when nothing matches."""
    # Cheap pre-check: a source with none of the trigger tokens cannot match
    # any rewrite -> return it byte-for-byte so the (vast) majority of kernels
    # keep their verbatim body (no reparse / comment-stripping churn).
    if not any(tok in source for tok in ("@", "matmul", "pad", "einsum")):
        return source
    tree = ast.parse(source)
    seed: Dict[str, int] = {a.name: len(a.shape) for a in kir.arrays}
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == kir.kernel_name), None)
    scope = fn if fn is not None else tree
    ranks = _rank_table(scope, seed)
    mm = _BatchedMatmulToLoop(ranks)
    mm.visit(scope)
    pad = _PadInline(ranks)
    pad.visit(scope)
    es = _EinsumInline()
    es.visit(scope)
    if not (mm.changed or pad.changed or es.changed):
        return source  # nothing matched -> leave the body verbatim
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
