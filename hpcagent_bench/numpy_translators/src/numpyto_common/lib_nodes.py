"""Library-node registry: numpy idioms -> Python-loop AST expansions.

Single source of truth for lowering numpy library calls to loops: each entry
registers an expander returning replacement AST statements, so the C/Fortran
emitters walk plain loops with no knowledge of the original idiom.

Covers reductions (sum/max/min/mean/prod -> accumulator loop, mean divides by
count), allocation aliases (zeros_like/empty_like, alongside the existing
``_ZerosRewriter`` for ``np.zeros``), naive shape-aware matmul, and power
(catches the ``math.pow``/``np.power`` call forms; the ``**`` binop itself
already passes through to the emitter).

Registry keys are the POST-``_MathRewriter`` call shape: ``np.sum`` is still
``Attribute(Name('np'), 'sum')``; ``math.exp`` is already a bare ``exp`` Name.
"""

import ast
import copy
import re
from functools import lru_cache
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from numpyto_common import dtypes
from numpyto_common.ir import tag_numpy_origin


def _name(n: str) -> ast.Name:
    return ast.Name(id=n, ctx=ast.Load())


def _const(v) -> ast.Constant:
    # numpy scalars (np.int64(0)) aren't Python int, so Fortran emit misclassifies
    # a size symbol built from one as REAL, and numpy 2.0's repr unparses it as
    # ``np.int64(0)``, breaking dace's sympy range parse. Coerce so every backend
    # sees a bare ``0`` / ``0.0``.
    if type(v).__module__.startswith("numpy"):
        v = v.item()
    return ast.Constant(value=v)


def _store(n: str) -> ast.Name:
    return ast.Name(id=n, ctx=ast.Store())


def _attr_call(mod: str, attr: str, args: List[ast.expr]) -> ast.Call:
    return ast.Call(func=ast.Attribute(value=_name(mod), attr=attr, ctx=ast.Load()), args=args, keywords=[])


# Expanders -- each returns replacement statements for the original assignment.


def _make_iter_name(prefix: str, depth: int) -> str:
    return f"{prefix}{depth}"


def _wrap_for_loops(iters: List[str], bounds, body: List[ast.stmt]) -> List[ast.stmt]:
    """Wrap ``body`` in nested ``for v in range(bound):`` loops, outermost first.

    ``bounds`` entries are either a string (rendered via :func:`_const_or_name`)
    or an already-built AST expression (passed through unchanged).
    """
    out = body
    for var, bound in zip(reversed(iters), reversed(bounds)):
        bound_node = _const_or_name(bound) if isinstance(bound, str) else bound
        out = [
            ast.For(target=_store(var),
                    iter=ast.Call(func=_name("range"), args=[bound_node], keywords=[]),
                    body=out,
                    orelse=[])
        ]
    return out


def _ast_eq(a: ast.AST, b: ast.AST) -> bool:
    """Structural equality for two AST expressions -- only ``Name``/``Constant``
    ids/values and matching ``BinOp`` ops; anything else is False, so the
    algebraic simplifier just falls back to the unsimplified form."""
    if type(a) is not type(b):
        return False
    if isinstance(a, ast.Name):
        return a.id == b.id
    if isinstance(a, ast.Constant):
        return a.value == b.value
    if isinstance(a, ast.BinOp):
        return (type(a.op) is type(b.op) and _ast_eq(a.left, b.left) and _ast_eq(a.right, b.right))
    return False


def _simplify_sub(hi: ast.AST, lo: ast.AST) -> Optional[ast.AST]:
    """Algebraic simplification for ``hi - lo``. Returns ``None``
    when the form doesn't match a known simplifying pattern."""
    if _ast_eq(hi, lo):
        return ast.Constant(value=0)
    if isinstance(hi, ast.BinOp) and isinstance(hi.op, ast.Add):
        # ``(lo + K) - lo`` -> K, at ANY depth of the ``+`` chain. A convolution tap slice spells
        # its upper bound ``ky + (oh - 1) * stride + 1``, which puts ``lo`` one level down where the
        # top-level match missed it. The extent is the same number either way, but the unsimplified
        # form NAMES the tap loop's variable, and the buffer it sizes is declared outside that loop
        # -- so the emitted C does not compile (alexnet, lenet5).
        for near, far in ((hi.left, hi.right), (hi.right, hi.left)):
            if _ast_eq(near, lo):
                return far
            inner = _simplify_sub(near, lo)
            if inner is not None:
                # ``(near - lo) + far`` is ``(near + far) - lo``, which is ``hi - lo``.
                return ast.BinOp(left=inner, op=ast.Add(), right=far)
    # ``(K - lo) - lo`` and other forms don't simplify in general.
    return None


def _is_full_slice_elt(e: ast.AST) -> bool:
    """Return True when subscript element ``e`` is a bare ``:`` -- a WHOLE-axis
    selection, i.e. semantically the same as omitting the axis."""
    return isinstance(e, ast.Slice) and e.lower is None and e.upper is None and e.step is None


def _is_full_slice_subscript(node: ast.Subscript) -> bool:
    """Return True when ``node`` is a Subscript whose slice is a
    full slice ``:`` (or a tuple of full slices ``:, :``)."""
    sl = node.slice
    if isinstance(sl, ast.Slice):
        return _is_full_slice_elt(sl)
    if isinstance(sl, ast.Tuple):
        return all(_is_full_slice_elt(e) for e in sl.elts)
    return False


def _has_slice_subscript(expr: ast.AST) -> bool:
    """True when ``expr`` contains a Subscript with a literal ``ast.Slice`` axis.
    Tells the call hoister whether the temp it emits needs an explicit slice-LHS
    form so slice-fusion can lower the per-element copy."""
    for sub in ast.walk(expr):
        if isinstance(sub, ast.Subscript):
            sl = sub.slice
            if isinstance(sl, ast.Slice):
                return True
            if isinstance(sl, ast.Tuple) and any(isinstance(e, ast.Slice) for e in sl.elts):
                return True
    return False


def _const_or_name(token: str) -> ast.expr:
    """Render a shape token as the matching AST node: int literal -> ``Constant``,
    bare identifier -> ``Name``, compound expression (``"N * 2"``, ``"x.shape[3]"``)
    -> re-parsed via ``ast.parse(mode="eval")`` into real Subscript/BinOp nodes.

    Compound support matters because ``_resolve_shape_token`` stringifies BinOps
    and ``arr.shape[i]`` refs into the shape table; without re-parsing, downstream
    AST walkers (e.g. the source-order shape resolver) would only see an opaque
    ``Name(id=full-text)``.
    """
    try:
        return _const(int(token))
    except ValueError:
        pass
    if isinstance(token, str) and token.isidentifier():
        return _name(token)
    # Compound expression -- re-parse.
    try:
        return ast.parse(str(token), mode="eval").body
    except (SyntaxError, ValueError):
        return _name(str(token))


def _shape_total_product(shape: Tuple[str, ...]) -> ast.expr:
    """Return an AST for ``shape[0] * shape[1] * ...`` -- used by mean."""
    parts = [_const_or_name(s) for s in shape]
    expr = parts[0]
    for p in parts[1:]:
        expr = ast.BinOp(left=expr, op=ast.Mult(), right=p)
    return expr


def _slice_step_const(sl: ast.Slice) -> Optional[int]:
    """Return a Slice's constant integer step (``a[lo:hi:k]`` -> ``k``), or ``None`` when
    there is no step or it is not a nonzero integer constant. A NEGATIVE step (``a[::-1]``
    reverse, ``a[::-2]``) is returned as-is; callers handle the reverse index mapping and
    take ``abs`` for the element count. A symbolic step is unsupported (``None``)."""
    step = sl.step
    if step is None:
        return None
    v = _const_int(step)
    return v if v not in (None, 0) else None


def _slice_step_expr(sl: ast.Slice) -> Optional[ast.expr]:
    """A slice's step when it is SYMBOLIC -- an expression rather than a literal.

    ``None`` for no step and for a literal one; :func:`_slice_step_const` answers that question.
    The value is a runtime stride the kernel takes across the ABI (a conv/pool ``stride``), so it
    cannot be folded to a literal and the loop nest has to carry it.

    Only a BOUNDED slice reaches here (see ``_reject_unsupported_slices``), and that is what makes
    the positive-stride lowering ``start + pos * step`` sound without knowing the sign: under a
    negative step numpy flips the bound defaults, so ``lo:hi:k`` with ``lo < hi`` is EMPTY, and the
    assignment consuming it already fails in numpy. There is no correct run to preserve.
    """
    step = sl.step
    if step is None or _const_int(step) is not None:
        return None
    return step


def _slice_step_any(sl: ast.Slice):
    """A slice's step as a literal ``int``, as an ``ast.expr`` when it is symbolic, or ``None``.

    The one accessor the index and extent builders share, so a symbolic stride can never reach one
    of them while the other still reads the slice as contiguous -- the two must walk the same run.
    """
    const = _slice_step_const(sl)
    return const if const is not None else _slice_step_expr(sl)


def _step_is_negative(step) -> bool:
    """``step`` is a literal negative stride -- the numpy reverse. A symbolic step is never this:
    it is emitted as a positive stride, which is the only sign a bounded slice can carry."""
    return isinstance(step, int) and step < 0


def _step_node(step) -> ast.expr:
    """``step`` as an expression, whether it arrived as a literal int or already as one."""
    return _const(step) if isinstance(step, int) else step


def _is_shape_scalar(node: ast.AST) -> bool:
    """``True`` for a ``.shape`` read -- ``A.shape`` (a tuple of dimensions) or
    ``A.shape[i]`` (one dimension). Both are INTEGER-valued regardless of ``A``'s
    element dtype, so a value/dtype walk must not descend into them."""
    if isinstance(node, ast.Attribute) and node.attr == "shape":
        return True
    return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "shape")


def _reads_complex(expr: ast.AST, local_dtypes: Dict[str, str]) -> bool:
    """True iff evaluating ``expr`` reads a complex value: a ``Constant(complex)``
    or a ``Name`` tagged complex in ``local_dtypes``. ``.shape`` subtrees are
    skipped -- always integer dimensions even when the array is complex.
    Shared predicate for the dtype-propagation passes (``_CallHoister._infer_complex``,
    ``LibNodeRewriter.visit_Assign``)."""
    if _is_shape_scalar(expr):
        return False
    if isinstance(expr, ast.Constant):
        return isinstance(expr.value, complex)
    if isinstance(expr, ast.Name):
        dt = local_dtypes.get(expr.id)
        return bool(dt and dt.startswith("complex"))
    return any(_reads_complex(c, local_dtypes) for c in ast.iter_child_nodes(expr))


def _slice_axes(node: ast.AST) -> List[ast.AST]:
    """Flat list of per-axis index nodes for any Subscript: ``A[i]`` -> ``[i]``,
    ``A[i, j]`` -> ``[i, j]``. A Slice axis is returned as the Slice node itself
    so callers can decide whether to scalarize."""
    if not isinstance(node, ast.Subscript):
        return []
    sl = node.slice
    if isinstance(sl, ast.Tuple):
        return list(sl.elts)
    return [sl]


def _is_special_axis(elt: ast.expr) -> bool:
    """A rank-shifting subscript entry: numpy newaxis (``None``) or ``...``
    (Ellipsis). Neither consumes exactly one source axis the ordinary way -- a
    newaxis inserts a size-1 result axis; an Ellipsis fills the un-consumed
    source axes."""
    return isinstance(elt, ast.Constant) and (elt.value is None or elt.value is Ellipsis)


def _is_scalar_axis(elt: ast.expr) -> bool:
    """A subscript entry that consumes ONE source axis as a scalar index -- an
    int Constant or a bare Name (loop iter / symbol), never a Slice, Ellipsis or
    newaxis."""
    if isinstance(elt, ast.Constant):
        return elt.value is not Ellipsis and elt.value is not None
    return isinstance(elt, ast.Name)


def _operand_token_shape(node: ast.expr, shape_table):
    """Residual shape TOKENS (not AST nodes -- stays consistent with the shape
    table) of an einsum/contraction operand. Bare ``Name(A)`` -> A's declared
    shape. Anything else (a Subscript slice/index chain, a matmul, a
    shape-preserving call...) routes through the general ``_iter_extent_of``
    sizer, which already knows a partial slice's actual bound (``a[k:k+H]``
    -> ``H``, not the base axis's full extent) alongside every other form it
    resolves. ``None`` if unresolvable."""
    nm = _name_id(node)
    if nm:
        return shape_table.get(nm)
    ext = _iter_extent_of(node, shape_table)
    return tuple(ast.unparse(e) for e in ext) if ext is not None else None


def _chained_base_shape(node: ast.expr, shape_table):
    """Residual token-shape of a SCALAR-chained subscript base ``A[i, j][...]``,
    when every inner index is a single-axis scalar (int Constant / bare Name):
    numpy combined-basic-indexing drops one leading axis per scalar, e.g.
    ``psi_frag[f]`` on ``(F, X, Y, Z, K)`` -> ``(X, Y, Z, K)``. Lets
    :func:`_iter_extent_of` size a chained access like ``psi_frag[f][..., 0]``
    whose base isn't yet flattened to a single Name subscript. ``None`` for a
    non-Name base or a Slice/Ellipsis/newaxis inner index."""
    if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)):
        return None
    # A bare Name axis is scalar ONLY when it is not a known array: ``x[aj]`` with aj an index
    # ARRAY is a gather, which KEEPS the axis (broadcast to aj's shape) instead of dropping it.
    # Treating it as a scalar shortened the residual shape, and an outer subscript
    # (``x[aj][:, None, :, :]``) then ran off the end and gave up on the whole extent.
    for axis in _slice_axes(node):
        if not _is_scalar_axis(axis):
            return None
        if isinstance(axis, ast.Name) and shape_table.get(axis.id):
            return None
    return _operand_token_shape(node, shape_table)


def _contraction_result_extent(expr: ast.Call, shape_table):
    """Output iter-extent of an ``np.einsum``/``tensordot``/``inner`` call.
    einsum uses its subscript string directly; tensordot/inner are mapped to
    an equivalent einsum spec first. ``None`` if operand shapes don't resolve."""
    attr = expr.func.attr
    if attr == "einsum":
        if not (isinstance(expr.args[0], ast.Constant) and isinstance(expr.args[0].value, str)):
            return None
        try:
            inputs, output = _parse_einsum_subscripts(expr.args[0].value)
        except NotImplementedError:
            return None
        operand_nodes = expr.args[1:]
    else:
        # tensordot / inner: build the equivalent spec from operand ranks. Operands need not be
        # bare Names -- ``_operand_token_shape`` resolves a slice/index-chain's residual rank too
        # (conv2d's tensordot contracts a sliced 4-D input against a partially-indexed
        # ``weights[ki, kj]``).
        a, b = expr.args[0], expr.args[1]
        shape_a, shape_b = _operand_token_shape(a, shape_table), _operand_token_shape(b, shape_table)
        if not shape_a or not shape_b:
            return None
        ra, rb = len(shape_a), len(shape_b)
        letters = "abcdefghijklmnopqrstuvwxyz"
        if attr == "inner":
            a_spec, b_spec = list(letters[:ra]), list(letters[ra:ra + rb])
            b_spec[-1] = a_spec[-1]
            inputs = ["".join(a_spec), "".join(b_spec)]
            output = "".join(a_spec[:-1] + b_spec[:-1])
        else:  # tensordot default axes=2
            kwargs = expr.keywords
            axes_node = expr.args[2] if len(expr.args) > 2 else _axes_kwarg(kwargs)
            try:
                a_ax, b_ax = _tensordot_axes(axes_node, ra, rb)
            except NotImplementedError:
                return None  # sizer contract: an unresolved extent is None, never an exception
            a_spec = list(letters[:ra])
            b_spec = [None] * rb
            nxt = ra
            for ca, cb in zip(a_ax, b_ax):
                b_spec[cb] = a_spec[ca]
            for i in range(rb):
                if b_spec[i] is None:
                    b_spec[i] = letters[nxt]
                    nxt += 1
            inputs = ["".join(a_spec), "".join(b_spec)]
            output = "".join([c for i, c in enumerate(a_spec) if i not in a_ax] +
                             [c for i, c in enumerate(b_spec) if i not in b_ax])
        operand_nodes = [a, b]
    letter_extent: Dict[str, str] = {}
    for spec, node in zip(inputs, operand_nodes):
        shape = _operand_token_shape(node, shape_table)
        if shape is None or len(shape) != len(spec):
            return None
        for letter, dim in zip(spec, shape):
            letter_extent.setdefault(letter, dim)
    if not output:
        return None  # scalar
    return tuple(_const_or_name(letter_extent[c]) for c in output)


def _np_fft_attr(call: ast.Call) -> Optional[str]:
    """The ``<name>`` of an ``np.fft.<name>(...)`` / ``numpy.fft.<name>(...)`` call,
    else ``None``. ``np.fft.*`` is a two-level attribute (``func.value`` is the
    ``np.fft`` attribute), which the single-level ``np.<attr>`` matchers miss."""
    f = call.func
    if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute) and f.value.attr == "fft"
            and isinstance(f.value.value, ast.Name) and f.value.value.id in ("np", "numpy")):
        return f.attr
    return None


#: Array methods whose numpy function twin takes the array first and means the same thing, so the
#: method spelling can be routed to it instead of growing a second branch. ``reshape`` is absent on
#: purpose: it has a method branch of its own that resolves a ``-1`` against the receiver.
ARRAY_METHOD_SHAPE_OPS: FrozenSet[str] = frozenset({
    "copy", "transpose", "squeeze", "clip", "round", "conj", "conjugate", "cumsum", "cumprod", "take", "repeat",
    "diagonal", "swapaxes"
})

#: Array REDUCTION methods with the same property. Kept beside the shape ops rather than merged
#: into them: a reduction CHANGES the rank and a shape op does not, and the sizer routes the two
#: differently. ``sort`` is absent on purpose -- the METHOD sorts in place and ``np.sort`` returns
#: a copy, so they are not the same call.
ARRAY_METHOD_REDUCTIONS: FrozenSet[str] = frozenset(
    {"sum", "prod", "mean", "max", "min", "argmax", "argmin", "std", "var", "any", "all", "ptp"})


class ArrayMethodRewriter(ast.NodeTransformer):
    """Normalize ``X.<m>(...)`` to ``np.<m>(X, ...)`` for every array method whose numpy function
    twin takes the array first (:data:`lib_nodes.ARRAY_METHOD_REDUCTIONS`).

    The method spelling only ever reached the expanders when the RECEIVER was a bare Name; an
    expression receiver walked straight through to the emitter, where correlation's
    ``((data - mean) ** 2).sum(axis=0)`` and nbody's ``(mass[i, 0] * vel[i, j]).sum()`` both died
    as "call to <expr>.sum not supported". One rewrite here serves every backend and every
    receiver shape, and the reduction expanders keep their single function-form path.

    A LOGICAL SPARSE receiver keeps its method: its buffers are not a dense array, and
    ``np.sum(A)`` over them would index a CSR triple as a 2-D matrix.
    """

    def __init__(self, sparse_names=None):
        self.sparse_names = set(sparse_names or ())

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in ARRAY_METHOD_REDUCTIONS):
            return node
        recv = func.value
        if isinstance(recv, ast.Name) and (recv.id in ("np", "numpy") or recv.id in self.sparse_names):
            return node
        if isinstance(recv, ast.Subscript) and isinstance(recv.value, ast.Name) and recv.value.id in self.sparse_names:
            return node
        return ast.copy_location(
            ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=func.attr, ctx=ast.Load()),
                     args=[recv] + list(node.args),
                     keywords=list(node.keywords)), node)


def _np_call_attr(func: ast.expr) -> Optional[str]:
    """``np.foo`` -> ``"foo"``, ``np.foo.bar`` -> ``"foo.bar"``, anything else -> ``None``."""
    if not isinstance(func, ast.Attribute):
        return None
    if isinstance(func.value, ast.Name) and func.value.id == "np":
        return func.attr
    if (isinstance(func.value, ast.Attribute) and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "np"):
        return f"{func.value.attr}.{func.attr}"
    return None


def _iter_extent_of(expr: ast.expr, shape_table: Dict[str, Tuple[str, ...]]) -> Optional[Tuple[ast.expr, ...]]:
    """Iteration extent of an array-valued expression. ``Name(A)`` -> A's full
    shape; ``Subscript(A, axes)`` -> upper-minus-lower per Slice axis in order
    (non-Slice axes are scalar, contribute nothing); negative slice bounds
    resolve against the operand's declared shape. ``None`` for unsupported
    forms -- caller falls through to ``NotImplementedError``.
    """
    if isinstance(expr, ast.Name):
        shape = shape_table.get(expr.id)
        return None if shape is None else tuple(_const_or_name(s) for s in shape)
    if isinstance(expr, ast.BinOp):
        # numpy ``@`` (MatMult) treats the last two axes as the matrix; leading
        # axes are batched and broadcast per the numpy spec:
        #
        #   1-D @ 1-D                  -> scalar (None)
        #   2-D @ 1-D                  -> (M,)
        #   1-D @ 2-D                  -> (N,)
        #   2-D @ 2-D                  -> (M, N)
        #   (..., M, K) @ (..., K, N)  -> (..., M, N)  (batched)
        #   (..., M, K) @ (K, N)       -> (..., M, N)  (broadcast 2-D rhs)
        #   (M, K) @ (..., K, N)       -> (..., M, N)  (broadcast 2-D lhs)
        if isinstance(expr.op, ast.MatMult):
            l_ext = _iter_extent_of(expr.left, shape_table)
            r_ext = _iter_extent_of(expr.right, shape_table)
            if l_ext is None or r_ext is None:
                return None
            ll, rl = len(l_ext), len(r_ext)
            if ll == 1 and rl == 1:
                return None  # scalar
            if ll == 2 and rl == 1:
                return (l_ext[0], )
            if ll == 1 and rl == 2:
                return (r_ext[1], )
            if ll == 2 and rl == 2:
                return (l_ext[0], r_ext[1])
            # Batched matmul: last two dims are M,K and K,N; broadcast leading axes.
            if ll >= 2 and rl >= 2:
                l_batch = l_ext[:-2]
                r_batch = r_ext[:-2]
                l_mn = (l_ext[-2], l_ext[-1])
                r_mn = (r_ext[-2], r_ext[-1])
                batch = _broadcast_extents(l_batch, r_batch)
                return tuple(batch) + (l_mn[0], r_mn[1])
            # 1-D paired with batched 2-D (collapse leading axis):
            #   ``a (K,) @ B (..., K, N) -> (..., N)``
            #   ``A (..., M, K) @ b (K,) -> (..., M)``
            if ll == 1 and rl > 2:
                return tuple(r_ext[:-2]) + (r_ext[-1], )
            if rl == 1 and ll > 2:
                return tuple(l_ext[:-2]) + (l_ext[-2], )
            return None
        # Broadcast the two extents axis-by-axis, aligned from the right,
        # picking the non-1 axis at each position (numpy rules).
        # ``A + b`` with A:(N, M), b:(M,) -> (N, M).
        # ``X + Y[:, None]`` with X:(M,), Y[:, None]:(N, 1) -> (N, M).
        l_ext = _iter_extent_of(expr.left, shape_table)
        r_ext = _iter_extent_of(expr.right, shape_table)
        if l_ext is None: return r_ext
        if r_ext is None: return l_ext
        return _broadcast_extents(l_ext, r_ext)
    if isinstance(expr, ast.UnaryOp):
        return _iter_extent_of(expr.operand, shape_table)
    if isinstance(expr, ast.Call):
        # ``np.fft.*`` is a two-level attribute (``func.value`` is ``np.fft``),
        # missed by the single-level ``np.<attr>`` matcher below. ``fftfreq(n)``
        # builds a length-n 1-D array; complex transforms/shifts are shape-
        # preserving -- sizing them here lets the harvest resolve LS3DF's
        # ``kx = 2*pi*np.fft.fftfreq(N)`` -> ``np.meshgrid`` -> ``gsq`` chain before
        # scalarization. REAL-FFT variants change the transformed axis length
        # (``rfftfreq(n)`` -> ``n//2+1``; ``rfftn``/``irfftn`` resize the last
        # axis) and are NOT sized here -- bail to None rather than emit a wrong extent.
        _fft = _np_fft_attr(expr)
        if _fft is not None:
            if _fft == "fftfreq" and expr.args:
                return (copy.deepcopy(expr.args[0]), )
            if _fft in ("fftn", "ifftn", "fft", "ifft", "fft2", "ifft2", "fftshift", "ifftshift") and expr.args:
                return _iter_extent_of(expr.args[0], shape_table)
            return None
        # Method-form ``<expr>.reshape(...)``: receiver is the operand, not ``np``,
        # so the ``np.reshape`` branch below (and the harvest, which runs before
        # the method->function normaliser) misses it. Left unsized, a reshape
        # target like ``X = (Yf @ C).reshape(shp)`` poisons every derived shape
        # (LS3DF's Rayleigh-Ritz).
        if (isinstance(expr.func, ast.Attribute) and expr.func.attr == "reshape"
                and not (isinstance(expr.func.value, ast.Name) and expr.func.value.id in ("np", "numpy"))
                and expr.args):
            if len(expr.args) == 1 and isinstance(expr.args[0], (ast.Tuple, ast.List)):
                elts = list(expr.args[0].elts)
            elif len(expr.args) == 1 and isinstance(expr.args[0], (ast.Name, ast.Constant, ast.BinOp, ast.UnaryOp)):
                elts = [expr.args[0]]
            else:
                elts = list(expr.args)  # varargs ``.reshape(a, b, c)``
            neg1 = [i for i, e in enumerate(elts) if _const_int(e) == -1]
            if len(neg1) == 1:
                base = _iter_extent_of(expr.func.value, shape_table)
                if base is None:
                    return None
                others = [e for j, e in enumerate(elts) if j != neg1[0]]
                denom = _mul_exts(others) if others else _const(1)
                elts[neg1[0]] = ast.BinOp(left=_mul_exts(base), op=ast.Div(), right=denom)
            elif neg1:
                return None
            return tuple(elts)
        # Axis-aware reduction: ``np.sum(operand, axis=k)`` -> operand's extent
        # with axis k removed (size 1 if keepdims); axis=None collapses to a
        # scalar (None). Makes the reduction's result shape available to every
        # shape-propagation caller (gem's ``r = np.sqrt(np.sum(d * d, axis=2))``,
        # force_lj, kmeans).
        if _is_reduction_call(expr):
            # The METHOD spelling carries its operand in the RECEIVER, not in ``args[0]``:
            # ``m.any(axis=-1)`` resolved to None where ``np.any(m, axis=-1)`` resolved fine, so
            # every shape built on one silently fell back to a broadcast partner's extent --
            # cp2k_density_matrix_trs4's ``c_pos`` came out (nnz, 1) instead of (nnz, fanout), and
            # the scatter that reads it then ran over a third of the contributions.
            method_form = (isinstance(expr.func, ast.Attribute)
                           and not (isinstance(expr.func.value, ast.Name) and expr.func.value.id in ("np", "numpy")))
            # ``_read_axis_keepdims`` reads the axis from positional slot 1, so the method's args are
            # shifted by one to put them in the vocabulary it expects.
            red_args = ([expr.func.value] + list(expr.args)) if method_form else list(expr.args)
            if not red_args:
                return None
            axes, keepdims = _read_axis_keepdims(red_args, expr.keywords)
            if axes is None:
                return None
            base = _iter_extent_of(red_args[0], shape_table)
            if base is None:
                return None
            n = len(base)
            norm = {a % n for a in axes}
            if keepdims:
                return tuple(_const(1) if i in norm else base[i] for i in range(n))
            return tuple(base[i] for i in range(n) if i not in norm) or None
        # Shape-CHANGING ops: result extent is NOT the operand's. ``np.reshape(A,
        # newshape)`` -> newshape; treating it as elementwise would propagate the
        # wrong rank to an enclosing BinOp (stockham_fft's
        # ``tmp_twid = np.reshape(tmp_perm, (N,)) * np.reshape(D, (N,))`` must be
        # rank-1, not tmp_perm's rank-3). ``repeat``/``transpose`` aren't statically
        # resolvable here -- bail to None rather than report the wrong extent.
        # ``np.<x>`` and the two-level ufunc-method spellings ``np.<ufunc>.<method>`` alike. The
        # dotted form never reached here, so the ``maximum.accumulate`` branch below was
        # unreachable and every ``np.<op>.outer`` came back with its first operand's extent.
        # Method spelling of a shape op: the receiver IS the operand, so ``rho.copy()`` says what
        # ``np.copy(rho)`` says. Routed once, here, rather than as a method branch per op -- the
        # two spellings had already diverged for reshape and for the reductions above, each fixed
        # separately after a kernel came back sized off a broadcast partner instead. ``astype`` and
        # ``flatten`` have no numpy function twin to route to and answer directly.
        if (isinstance(expr.func, ast.Attribute)
                and not (isinstance(expr.func.value, ast.Name) and expr.func.value.id in ("np", "numpy"))):
            method = expr.func.attr
            if method == "astype":
                return _iter_extent_of(expr.func.value, shape_table)  # dtype only, never the shape
            if method in ("ravel", "flatten"):
                base = _iter_extent_of(expr.func.value, shape_table)
                return None if base is None else (_mul_exts(base), )
            if method in ARRAY_METHOD_SHAPE_OPS:
                routed = ast.Call(func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()),
                                                     attr=method,
                                                     ctx=ast.Load()),
                                  args=[expr.func.value] + list(expr.args),
                                  keywords=list(expr.keywords))
                return _iter_extent_of(ast.copy_location(routed, expr), shape_table)
        attr = _np_call_attr(expr.func)
        if attr is not None:
            # An array CONSTRUCTOR states its extent in its shape argument -- read it
            # directly, since an inline constructor (never assigned to a Name) is
            # never sized by the harvest. Otherwise the triu/tril first-arg hoist in
            # ``_CallHoister`` (gated on a resolvable extent) leaves it buried and the
            # bare-Name expander rejects it -- gpt2_block's causal-mask
            # ``np.triu(np.ones((seq, seq), np.float32), 1)``. ``*_like`` aliases take
            # an array, not a shape -- mirror that operand's extent instead.
            # ``np.full`` states its extent exactly as the zeros aliases do, but it is not one of
            # them: it carries a fill VALUE that the alias rewriter would drop, turning a -inf pad
            # into a zero pad. Sized here, aliased nowhere.
            # ``np.linalg`` is a TWO-level attribute, so every single-level branch below reads it
            # as nothing. ``inv``/``cholesky`` are shape-preserving factors; ``solve`` returns x
            # with b's shape. Sized here and not only in lowering's harvest, because a shape read
            # that resolves against a factorisation is asked at parse time -- ls3df_scf reaches its
            # whole Rayleigh-Ritz block through ``Linv``, and an unsized factor breaks that chain.
            # Square identities and the 1-D generators state their extent in an argument, same as
            # the zeros aliases do; unsized, ls3df_scf's ``s_sub`` chain broke on the ``np.eye(k)``
            # jitter term and took the whole Rayleigh-Ritz block with it.
            if attr in ("eye", "identity") and expr.args:
                n = copy.deepcopy(expr.args[0])
                return (n, copy.deepcopy(expr.args[1])) if attr == "eye" and len(
                    expr.args) >= 2 and not _const_int(expr.args[1]) is None else (n, copy.deepcopy(n))
            if attr == "arange" and len(expr.args) == 1:
                return (copy.deepcopy(expr.args[0]), )
            if attr in ("linalg.inv", "linalg.cholesky") and expr.args:
                return _iter_extent_of(expr.args[0], shape_table)
            if attr == "linalg.solve" and len(expr.args) >= 2:
                return _iter_extent_of(expr.args[1], shape_table)
            if (attr in NP_ZEROS_ALIASES or attr in ("full", "full_like")) and expr.args:
                if attr.endswith("_like"):
                    return _iter_extent_of(expr.args[0], shape_table)
                shape_arg = expr.args[0]
                if isinstance(shape_arg, (ast.Tuple, ast.List)):
                    return tuple(copy.deepcopy(e) for e in shape_arg.elts)
                return (copy.deepcopy(shape_arg), )
            if attr == "reshape" and len(expr.args) >= 2:
                newshape = expr.args[1]
                elts: Optional[List[ast.expr]] = None
                if isinstance(newshape, (ast.Tuple, ast.List)):
                    elts = list(newshape.elts)
                elif isinstance(newshape, (ast.Name, ast.Constant, ast.BinOp, ast.UnaryOp)):
                    elts = [newshape]
                if elts is not None:
                    # Resolve a ``-1`` placeholder (``x.reshape(batch, -1)``) to
                    # ``total_source_size // product(other target dims)``.
                    neg1 = [i for i, e in enumerate(elts) if _const_int(e) == -1]
                    if len(neg1) == 1:
                        base = _iter_extent_of(expr.args[0], shape_table)
                        if base is None:
                            return None
                        others = [e for j, e in enumerate(elts) if j != neg1[0]]
                        denom = _mul_exts(others) if others else _const(1)
                        # ``/`` (renders as integer division in C/Fortran for int dims),
                        # not ``//`` which is not valid C when the token is emitted.
                        elts[neg1[0]] = ast.BinOp(left=_mul_exts(base), op=ast.Div(), right=denom)
                    elif neg1:
                        return None  # more than one -1 is ambiguous
                    return tuple(elts)
            if attr == "transpose" and expr.args:
                # ``x.T``/``np.transpose(x)`` -> operand's extent with axes reversed
                # (or permuted by an explicit axes tuple). nbody's ``dx = x.T - x``
                # (x is (N, 1)) must broadcast to (N, N); ``None`` here would collapse
                # it to (N, 1) -- ``dx`` then allocates (N, 1) but is written (N, N)
                # (heap overflow), and the ``(dx*inv_r3) @ mass`` matmul mis-contracts.
                base = _iter_extent_of(expr.args[0], shape_table)
                if base is None:
                    return None
                if (len(expr.args) >= 2 and isinstance(expr.args[1], (ast.Tuple, ast.List))):
                    perm = [
                        e.value for e in expr.args[1].elts if isinstance(e, ast.Constant) and isinstance(e.value, int)
                    ]
                    if len(perm) == len(base):
                        return tuple(base[p] for p in perm)
                return tuple(reversed(base))
            # Shape aliases -> operand's extent with axes swapped / a unit axis
            # inserted or dropped, so an enclosing BinOp broadcasts and the fresh
            # target allocates at the right rank.
            if attr == "swapaxes" and len(expr.args) >= 3:
                base = _iter_extent_of(expr.args[0], shape_table)
                if base is None:
                    return None
                i, j = _const_axis(expr.args[1], len(base)), _const_axis(expr.args[2], len(base))
                if i is None or j is None:
                    return None
                out = list(base)
                out[i], out[j] = out[j], out[i]
                return tuple(out)
            if attr == "moveaxis" and len(expr.args) >= 3:
                # numpy's own algorithm: drop the source axis, reinsert it at the destination among
                # what is left. Without this the sizer returns None, the hoister declines, and the
                # call survives to the emitter as an unsupported ``np.moveaxis``.
                base = _iter_extent_of(expr.args[0], shape_table)
                if base is None:
                    return None
                src = _const_axis(expr.args[1], len(base))
                dst = _const_axis(expr.args[2], len(base))
                if src is None or dst is None:
                    return None
                rest = [e for n, e in enumerate(base) if n != src]
                return tuple(rest[:dst] + [base[src]] + rest[dst:])
            if attr == "expand_dims" and expr.args:
                base = _iter_extent_of(expr.args[0], shape_table)
                if base is None:
                    return None
                axis = _const_axis(_kwarg_or_pos(expr.args, expr.keywords, 1, "axis"), len(base) + 1)
                if axis is None:
                    return None
                out = list(base)
                out.insert(axis, _const(1))
                return tuple(out)
            if attr == "squeeze" and expr.args:
                base = _iter_extent_of(expr.args[0], shape_table)
                if base is None:
                    return None
                axis_node = _kwarg_or_pos(expr.args, expr.keywords, 1, "axis")
                if axis_node is not None:
                    axis = _const_axis(axis_node, len(base))
                    if axis is None or not (isinstance(base[axis], ast.Constant) and base[axis].value == 1):
                        return None
                    out = [e for k, e in enumerate(base) if k != axis]
                else:
                    out = [e for e in base if not (isinstance(e, ast.Constant) and e.value == 1)]
                return tuple(out) if out else (_const(1), )
            if attr == "take" and len(expr.args) >= 2:
                base = _iter_extent_of(expr.args[0], shape_table)
                # A LITERAL index takes one element off the axis, so numpy drops that axis
                # entirely -- distinct from an unresolvable index, which is what ``None`` from
                # ``_iter_extent_of`` otherwise means. Conflating the two left the enclosing
                # ``np.expand_dims`` / ``np.concatenate`` unsized and refused.
                lit_index = _const_int(expr.args[1])
                idx_ext = None if lit_index is not None else _iter_extent_of(expr.args[1], shape_table)
                if base is None or (lit_index is None and (idx_ext is None or len(idx_ext) != 1)):
                    return None
                if lit_index is not None:
                    axis_node = _kwarg_or_pos(expr.args, expr.keywords, 2, "axis")
                    if axis_node is None:
                        return None  # flat take on an N-D source: numpy ravels first
                    axis = _const_axis(axis_node, len(base))
                    if axis is None:
                        return None
                    out = [e for k, e in enumerate(base) if k != axis]
                    return tuple(out) or None
                axis_node = _kwarg_or_pos(expr.args, expr.keywords, 2, "axis")
                if axis_node is None:
                    return idx_ext if len(base) == 1 else None  # flat take on a 1-D source
                axis = _const_axis(axis_node, len(base))
                if axis is None:
                    return None
                out = list(base)
                out[axis] = idx_ext[0]
                return tuple(out)
            if attr == "repeat":
                return None
            # ``np.<op>.outer(a, b)`` pairs every element of a with every element of b: rank 2,
            # sized from the two operands' own extents.
            if attr.endswith(".outer") and len(expr.args) == 2:
                l_out = _iter_extent_of(expr.args[0], shape_table)
                r_out = _iter_extent_of(expr.args[1], shape_table)
                if l_out is None or r_out is None or len(l_out) != 1 or len(r_out) != 1:
                    return None
                return (l_out[0], r_out[0])
            # ``np.einsum(subscripts, *operands)`` -> the output extent: one axis per
            # output index letter, sized from the operand that introduces it. The
            # elementwise fallthrough below would wrongly take the first operand's
            # full rank.
            if attr in ("einsum", "tensordot", "inner") and len(expr.args) >= 2:
                ext = _contraction_result_extent(expr, shape_table)
                if ext is not None:
                    return ext
                return None
            if attr in ("trace", "vdot", "median"):
                return None  # scalar result
            if attr == "diagonal" and expr.args:
                base = _iter_extent_of(expr.args[0], shape_table)
                return (base[0], ) if base else None
            # ``np.diag(v [, k])`` -- a 1-D operand builds an ``(n+|k|, n+|k|)``
            # matrix (single source of truth for the constructed shape); a 2-D
            # operand extracts the main diagonal (length of its first axis).
            if attr == "diag" and expr.args:
                base = _iter_extent_of(expr.args[0], shape_table)
                if base is None:
                    return None
                if len(base) == 2:
                    return (base[0], )
                if len(base) != 1:
                    return None
                k_node = _kwarg_or_pos(expr.args, expr.keywords, 1, "k")
                if k_node is None:
                    off = 0
                else:
                    kc = _const_int(k_node)
                    if kc is None:
                        return None  # non-const offset can't size the result
                    off = abs(kc)
                side = (base[0] if off == 0 else ast.BinOp(left=base[0], op=ast.Add(), right=_const(off)))
                return (side, copy.deepcopy(side))
            # ``np.cumsum``/``np.cumprod`` and the ``maximum``/``minimum.accumulate`` running
            # extremes: a prefix scan is shape-preserving along its
            # axis, so the result takes the operand's extent. Skipping this leaves a
            # fresh LHS (histogram_equalization's ``cdf = np.cumsum(hist)``) unsized,
            # so ``_ELEMENT_WRITE_EXPANDERS`` (gated on the target already having a
            # shape) silently skips it and ``cdf[0] = ..`` stores through a NULL
            # pointer (SIGSEGV). numpy flattens an axis-less scan over an N-D operand,
            # which the cumulative expander rejects -- leave unresolved rather than
            # claim a wrong shape.
            # ``np.searchsorted(a, v)`` returns one index per element of ``v``, so the result takes
            # the VALUES operand's extent, not the sorted array's.
            if attr == "searchsorted" and len(expr.args) >= 2:
                return _iter_extent_of(expr.args[1], shape_table)
            if attr in ("cumsum", "cumprod", "maximum.accumulate", "minimum.accumulate") and expr.args:
                base = _iter_extent_of(expr.args[0], shape_table)
                if base is None:
                    return None
                if _kwarg_or_pos(expr.args, expr.keywords, 1, "axis") is not None or len(base) == 1:
                    return base
                return None
            # ``np.pad(src, pad_width, ...)`` -> each source axis grown by its
            # ``before + after`` width (scalar R or per-axis tuple). The stencil
            # ghost cells / the vector variants' unpadded component axis.
            if attr == "pad" and expr.args:
                base = _iter_extent_of(expr.args[0], shape_table)
                if base is None:
                    return None
                pad_arg = _kwarg_or_pos(expr.args, expr.keywords, 1, "pad_width")
                return _pad_output_extent(base, pad_arg)
            # ``np.concatenate((a, b, ...), axis=k)`` -> the operands' common
            # shape with axis ``k`` summed across operands (dwt2d Haar
            # recompose). Other axes are taken from the first operand.
            if attr == "concatenate" and expr.args:
                try:
                    names, shapes, axis = _concat_operands_axis(expr.args, expr.keywords, shape_table)
                except NotImplementedError:
                    return None
                base = list(shapes[0])
                summed = "(" + ") + (".join(s[axis] for s in shapes) + ")"
                base[axis] = summed
                return tuple(_const_or_name(t) for t in base)
            # ``np.stack((a, b, ...), axis=k)`` -> the operands' common shape with a NEW
            # size-N axis inserted at k (N = number of operands); out's rank = rank + 1.
            if attr == "stack" and expr.args:
                try:
                    names, shapes, _ = _concat_operands_axis(expr.args, expr.keywords, shape_table)
                    axis = _stack_axis(expr.args, expr.keywords, len(shapes[0]))
                except NotImplementedError:
                    return None
                out = [_const_or_name(t) for t in shapes[0]]
                out.insert(axis, _const(len(names)))
                return tuple(out)
        # A BINARY ufunc BROADCASTS its operands; the first-arg rule below is only right when one
        # operand carries the whole extent. ``np.equal(a[:, None, :], b[:, :, None])`` -- what the
        # frontend rewrites ``a[:, None, :] == b[:, :, None]`` into -- came back with the LEFT
        # side's (N, 1, F) rather than (N, F, F), so cp2k_density_matrix_trs4's match plane was
        # declared a third of its size and every reduction over it ran short. The Compare spelling
        # already broadcast; the function spelling now agrees with it.
        if isinstance(expr.func, ast.Attribute) and expr.func.attr in _BROADCASTING_UFUNCS and len(expr.args) >= 2:
            ext = _broadcast_children(list(expr.args), shape_table)
            if ext is not None:
                return ext
        # Elementwise/unary math functions (abs, sqrt, exp, sin, cos, log, ...)
        # preserve the operand's iter extent -- pick the first arg that resolves.
        for arg in expr.args:
            ext = _iter_extent_of(arg, shape_table)
            if ext is not None:
                return ext
        return None
    if isinstance(expr, ast.Compare):
        # ``a == b`` etc: broadcast the operand extents (numpy rules) so an outer
        # comparison ``a[:, None] == b[None, :]`` with (N, 1) and (1, N) yields
        # (N, N), not just the left side's (N, 1) (smith_waterman/needleman_wunsch
        # substitution matrix).
        return _broadcast_children([expr.left, *expr.comparators], shape_table)
    if isinstance(expr, ast.BoolOp):
        return _broadcast_children(expr.values, shape_table)
    if isinstance(expr, ast.Subscript):
        name = _name_id(expr.value)
        if name:
            shape = shape_table.get(name)
        else:
            # Chained scalar-indexed base ``A[i, j][outer]``: resolve the base's
            # residual shape so the outer axes below index it -- not yet flattened
            # to a single Name subscript at harvest time.
            shape = _chained_base_shape(expr.value, shape_table)
            if shape is None:
                # Any other sized base: a CALL result indexed directly, which is what
                # ``np.expand_dims(np.take(x, 0, axis=k), axis=k)`` becomes once the frontend
                # rewrites expand_dims to a newaxis index. Its own extent is the shape the
                # axes below index against.
                base_ext = _iter_extent_of(expr.value, shape_table)
                shape = tuple(ast.unparse(e) for e in base_ext) if base_ext is not None else None
        axes = _slice_axes(expr)
        ext: List[ast.expr] = []
        src_axis = 0  # source-axis pointer -- advances on Slice / scalar
        # axes, NOT on ``None`` (newaxis -- pure result-axis insertion).
        # Explicit (non-special) axes consumed; Ellipsis fills the rest.
        n_src_consumers = sum(1 for ax in axes if not _is_special_axis(ax))
        # numpy ADVANCED indexing: several integer-ARRAY indices together broadcast
        # into a SINGLE group of result axes, not one per array -- ``u2[q, r, s]``
        # with q/r/s all (J,) -> (J,), not (J,J,J) (fft_3d checksum gather). Collect
        # the index-array shapes and emit the broadcast once, at the first
        # index-array's result position.
        idx_array_extents: List[Tuple[ast.expr, ...]] = []
        idx_group_pos: Optional[int] = None
        for ax in axes:
            if isinstance(ax, ast.Constant) and ax.value is None:
                # numpy newaxis -- inserts a length-1 result axis without
                # consuming a source axis.
                ext.append(_const(1))
                continue
            if isinstance(ax, ast.Constant) and ax.value is Ellipsis:
                # numpy Ellipsis -- expands to the full slices of every source
                # axis the other (explicit) entries do not consume, each
                # contributing that axis's full extent. Needs the source rank.
                if not shape:
                    return None
                for _ in range(max(len(shape) - n_src_consumers, 0)):
                    if src_axis >= len(shape):
                        return None
                    ext.append(_const_or_name(shape[src_axis]))
                    src_axis += 1
                continue
            if isinstance(ax, ast.Slice):
                axis_len = (_const_or_name(shape[src_axis]) if shape and src_axis < len(shape) else None)
                lo = _resolve_negative(ax.lower, axis_len) if ax.lower is not None else _const(0)
                hi = _resolve_negative(ax.upper, axis_len) if ax.upper is not None else axis_len
                if hi is None or lo is None:
                    return None
                if isinstance(hi, ast.Constant) and isinstance(lo, ast.Constant):
                    raw: ast.expr = _const(hi.value - lo.value)
                elif isinstance(lo, ast.Constant) and lo.value == 0:
                    # ``hi - 0`` simplifies to ``hi``.
                    raw = hi
                else:
                    # ``hi - lo`` algebraic simplification:
                    #  ``(lo + K) - lo`` -> K  (slice ``[i:i+K]``)
                    #  ``(K + lo) - lo`` -> K
                    #  ``lo - lo``       -> 0
                    simplified = _simplify_sub(hi, lo)
                    raw = simplified if simplified is not None else ast.BinOp(left=hi, op=ast.Sub(), right=lo)
                # Strided slice ``a[lo:hi:k]`` has ``ceil((hi - lo) / k)``
                # elements (== ``len(range(lo, hi, k))``). dwt2d's Haar
                # ``b[:, 0::2]`` over an even axis ``s`` -> ``s // 2``.
                step = _slice_step_any(ax)
                if _step_is_negative(step) and (ax.lower is not None or ax.upper is not None):
                    # A BOUNDED reverse slice (``a[lo::-1]``/``a[:hi:-1]``): numpy flips
                    # the bound defaults under a negative step, so ``raw = hi - lo`` is
                    # NOT the element count and the ceil below would over-count, reading
                    # OOB. Only full-axis reverse (``a[::-k]``) is reliably counted --
                    # bail on the bounded form.
                    return None
                if isinstance(step, ast.expr):
                    # Symbolic stride: ceil(raw / step) with the divisor carried as an expression.
                    # No abs() -- a bounded slice's step is positive or the numpy source is empty.
                    ext.append(
                        ast.BinOp(left=ast.BinOp(left=ast.BinOp(left=raw, op=ast.Add(), right=step),
                                                 op=ast.Sub(),
                                                 right=_const(1)),
                                  op=ast.FloorDiv(),
                                  right=step))
                elif step is not None and step != 1:
                    # Element count is ceil(raw / |step|) -- a full-axis negative step
                    # (reverse) spans the same number of elements as its positive
                    # magnitude.
                    astep = abs(step)
                    if isinstance(raw, ast.Constant):
                        ext.append(_const((raw.value + astep - 1) // astep))
                    else:
                        ext.append(
                            ast.BinOp(left=ast.BinOp(left=raw, op=ast.Add(), right=_const(astep - 1)),
                                      op=ast.FloorDiv(),
                                      right=_const(astep)))
                else:
                    ext.append(raw)
            elif isinstance(ax, ast.Name) and shape_table.get(ax.id):
                # Fancy-index gather: ``arr[idx]``, ``idx`` a known-shape int array
                # (a scalar Name -- loop var/symbol -- has no shape, contributes
                # nothing). Multiple index arrays broadcast into ONE result-axis
                # group -- record, emit later.
                if idx_group_pos is None:
                    idx_group_pos = len(ext)
                idx_array_extents.append(tuple(_const_or_name(s) for s in shape_table[ax.id]))
            elif _advanced_index_rank(ax, shape_table):
                # Advanced-index EXPRESSION axis (``edge_idx[:, :, 0] - 1``): a
                # sliced/offset index array used as a gather index. Its result
                # extent is its own extent; joins the same broadcast group as any
                # bare-Name index.
                ie = _iter_extent_of(ax, shape_table)
                if ie is not None:
                    if idx_group_pos is None:
                        idx_group_pos = len(ext)
                    idx_array_extents.append(tuple(ie))
            # scalar axis: contributes nothing to result extent but advances
            # the source-axis pointer.
            src_axis += 1
        if idx_array_extents:
            # Broadcast the index extents together (numpy advanced index). Picking the
            # LONGEST is not the numpy rule and is wrong whenever the ranks tie:
            # ``nbfp[ti[None, :, None], tj[:, None, :], 0]`` with ti (1, I, 1) and tj (P, 1, J)
            # is (P, I, J), not the first operand's (1, I, 1).
            group = idx_array_extents[0]
            for other in idx_array_extents[1:]:
                group = _broadcast_extents(group, other)
            ext[idx_group_pos:idx_group_pos] = list(group)
        # Append any trailing axes that the Subscript didn't index --
        # numpy implicitly takes the full extent for omitted trailing
        # axes. ``path[:]`` on a 2-D ``path`` returns a 2-D extent.
        if shape and src_axis < len(shape):
            for i in range(src_axis, len(shape)):
                ext.append(_const_or_name(shape[i]))
        return tuple(ext) if ext else None
    if isinstance(expr, ast.Attribute):
        # ``A.T`` reverses the axes; ``.real`` / ``.imag`` pick a component and keep them. The
        # ``np.transpose`` branch above already claims to answer for ``x.T``, but an attribute is
        # never a Call and never reached it -- so a transpose spelled the short way resolved in the
        # RANK table and to nothing here, and the two disagreed about the same expression.
        if expr.attr == "T":
            base = _iter_extent_of(expr.value, shape_table)
            return None if base is None else tuple(reversed(base))
        if expr.attr in ("real", "imag"):
            return _iter_extent_of(expr.value, shape_table)
    return None


_REDUCTION_NAMES: Set[str] = {
    "sum",
    "mean",
    "prod",
    "std",
    "var",
    "min",
    "max",
    "argmin",
    "argmax",
    "any",
    "all",
    "count_nonzero",
    "median",
    # dot/vdot/matmul-like calls also collapse the operand to a scalar or
    # lower-rank array -- not preserved at the operand's iter extent. Same for
    # np.linalg.* (norm collapses to scalar; lstsq returns a tuple).
    "dot",
    "vdot",
    "inner",
    "norm",
    "det",
    "lstsq",
}


def _is_reduction_call(call: ast.Call) -> bool:
    """True for ``np.sum(...)``/``arr.sum(...)``/similar reduction calls --
    only the registry of names above counts."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr in _REDUCTION_NAMES
    if isinstance(call.func, ast.Name):
        return call.func.id in _REDUCTION_NAMES
    return False


def _is_const_one(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value == 1


def _broadcast_extents(l_ext: Tuple[ast.expr, ...], r_ext: Tuple[ast.expr, ...]) -> Tuple[ast.expr, ...]:
    """Numpy broadcasting on two extent tuples: align from the right, pad the
    shorter on the left with implicit 1. Per aligned axis, a literal ``1``
    stretches to the other side; otherwise the left side wins (shape mismatches
    surface later, at scalarise time)."""
    rank = max(len(l_ext), len(r_ext))
    l_pad = (_const(1), ) * (rank - len(l_ext)) + l_ext
    r_pad = (_const(1), ) * (rank - len(r_ext)) + r_ext
    out: List[ast.expr] = []
    for l, r in zip(l_pad, r_pad):
        # A size-1 axis on either side stretches to the other's extent -- a size-1
        # RIGHT axis must yield the LEFT extent, not silently keep the (already
        # equal) left, so ``B(N, M) * a(N, 1)`` broadcasts to ``M`` rather than
        # dropping it.
        if _extent_is_one(l):
            out.append(r)
        elif _extent_is_one(r):
            out.append(l)
        else:
            # Equal extents keep either; a genuine runtime-1 mismatch can't resolve
            # statically, so take the left (the scalarizer indexes each operand by
            # its own shape, so a per-operand size-1 axis still reads with a 0).
            out.append(l)
    return tuple(out)


def _extent_is_one(node: ast.expr) -> bool:
    """True when an extent is the literal ``1`` -- a bare ``Constant(1)`` or a
    node that unparses to ``"1"`` (a shape token ``"1"`` re-parsed via
    ``_const_or_name``). A symbolic extent that is only 1 at runtime cannot be
    detected here."""
    if _is_const_one(node):
        return True
    try:
        return ast.unparse(node).strip() == "1"
    except (AttributeError, ValueError):
        return False


def extent_is_scalar(ext: Optional[Tuple[ast.expr, ...]]) -> bool:
    """True when a broadcast extent is entirely size-1: such a value is a SCALAR
    in numpyto's model, not a ``T t[1]`` array (e.g. ``t = (a[i] > x)`` with ``x``
    shape ``(1,)``). Misregistering it as an array desyncs scalar uses (``t = 0``,
    ``if t``) from the array-style ``memset``/``t[__w0] = ...`` writes the extent
    would drive -- doesn't compile. Rank-0 (empty tuple) is already scalar."""
    return ext is not None and all(_extent_is_one(e) for e in ext)


def _is_integer_expr(node: ast.AST, local_dtypes: Dict[str, str], array_names: Set[str] = frozenset()) -> bool:
    """Best-effort: does ``node`` evaluate to an integer? Recognises int Constants,
    Names tagged integer in ``local_dtypes``, and ``+ - * % //`` over integer
    operands.

    An ARRAY Name (in ``array_names``) counts as integer only when
    ``local_dtypes`` explicitly tags it int -- untagged arrays default to float.
    A non-array Name (loop iter/shape symbol) is integer by default, so
    ``j % nx`` (j int-tagged, nx a symbol) stays integer."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        dt = local_dtypes.get(node.id)
        if dt is not None:
            return dtypes.is_integer(dt)
        return node.id not in array_names  # untagged array -> float default
    if isinstance(node, ast.Subscript):
        # An element/gather of an integer-typed array is itself integer
        # (QE index tables ``dfftt_nl[gki]``, ``igk_exx[:n, k]``); an untagged
        # base defaults to float.
        base = node.value
        if isinstance(base, ast.Name):
            dt = local_dtypes.get(base.id)
            return dt is not None and dtypes.is_integer(dt)
        return _is_integer_expr(base, local_dtypes, array_names)
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Mod, ast.FloorDiv)):
            return False
        return (_is_integer_expr(node.left, local_dtypes, array_names)
                and _is_integer_expr(node.right, local_dtypes, array_names))
    if isinstance(node, ast.UnaryOp):
        return _is_integer_expr(node.operand, local_dtypes, array_names)
    return False


def _as_float64(node: ast.expr) -> ast.expr:
    """Wrap ``node`` in ``np.float64(...)`` -- the cast both native emitters render
    (``(double)(x)`` / ``REAL(x, kind=c_double)``). Same spelling lowering's
    ``_TrueDivisionPromoter`` uses, so the two paths stay one convention."""
    return ast.Call(func=ast.Attribute(value=_name("np"), attr="float64", ctx=ast.Load()), args=[node], keywords=[])


def _provably_integer(node: ast.expr, local_dtypes: Dict[str, str]) -> bool:
    """True when ``node``'s VALUE is certainly integer: an int Constant, or a Name /
    element-of-Name tagged with an integer dtype.

    Deliberately stricter than :func:`_is_integer_expr`, which reads an UNTAGGED
    non-array Name as integer -- right when classifying size symbols, wrong here:
    a ufunc operand that is merely untagged (an undeclared float scalar, a float
    array) must not be taken for an integer, because that decides whether the
    result dtype is integral."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        dt = local_dtypes.get(node.id)
        return dt is not None and dtypes.is_integer(dt)
    if isinstance(node, ast.Subscript):
        return _provably_integer(node.value, local_dtypes)
    return False


def _all_integer_operands(args: List[ast.expr], local_dtypes: Optional[Dict[str, str]]) -> bool:
    """True when EVERY operand of a ufunc call is provably integer -- i.e. numpy would
    promote the result to an integer dtype. An absent dtype table answers False."""
    if not local_dtypes or not args:
        return False
    return all(_provably_integer(a, local_dtypes) for a in args)


#: Elementwise ufuncs whose numpy result dtype is the PROMOTED OPERAND dtype, so an
#: all-integer call returns an integer array. Their hoisted temp must be declared
#: integer, not the double default: a double temp rounds every value above 2**53
#: (``np.power(3, 39)`` came back 11 short). ``divide`` is NOT here -- it always
#: returns float, and its cast is applied in :func:`expand_divide`.
#: numpy ufuncs whose result extent is the BROADCAST of their operands, not the first one's. Only
#: functions whose every argument is an OPERAND belong here: a call whose second positional slot is
#: an axis, a shape or a tolerance must keep the first-arg rule, or that slot's extent would be
#: folded into the result.
_BROADCASTING_UFUNCS: Set[str] = {
    "add", "subtract", "multiply", "divide", "true_divide", "floor_divide", "power", "float_power", "mod", "remainder",
    "fmod", "maximum", "minimum", "fmax", "fmin", "hypot", "arctan2", "logaddexp", "logaddexp2", "copysign",
    "nextafter", "heaviside", "less", "less_equal", "greater", "greater_equal", "equal", "not_equal", "logical_and",
    "logical_or", "logical_xor", "bitwise_and", "bitwise_or", "bitwise_xor", "left_shift", "right_shift"
}

_INT_PRESERVING_ELEMENTWISE: Set[str] = {"add", "subtract", "multiply", "power", "maximum", "minimum"}


def _broadcast_children(children: List[ast.expr], shape_table: Dict[str, Tuple[str,
                                                                               ...]]) -> Optional[Tuple[ast.expr, ...]]:
    """Fold every child's iter extent through numpy broadcasting, skipping
    scalar (None-extent) children. Returns the broadcast extent, or None when
    no child has an extent. Shared by the Compare / BoolOp extent branches."""
    acc: Optional[Tuple[ast.expr, ...]] = None
    for child in children:
        ext = _iter_extent_of(child, shape_table)
        if ext is None:
            continue
        acc = ext if acc is None else _broadcast_extents(acc, ext)
    return acc


def _resolve_negative(node: ast.AST, axis_len: Optional[ast.expr]) -> Optional[ast.expr]:
    """Resolve a slice bound: negative int -> ``axis_len - K``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value < 0:
        if axis_len is None:
            return None
        return ast.BinOp(left=axis_len, op=ast.Sub(), right=_const(-node.value))
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int) and axis_len is not None):
        return ast.BinOp(left=axis_len, op=ast.Sub(), right=_const(node.operand.value))
    return node


def _name_id(node: ast.AST) -> Optional[str]:
    return node.id if isinstance(node, ast.Name) else None


def _advanced_index_rank(expr: ast.expr, shape_table: Dict[str, Tuple[str, ...]]) -> Optional[int]:
    """Broadcast rank of an advanced-index EXPRESSION used as one axis of an outer
    gather, or ``None`` if ``expr`` isn't one: a Subscript on a known array with
    >=1 Slice axis, possibly wrapped in arithmetic (ICON's ``edge_idx[:, :, 0] -
    1``). Rank = number of Slice axes. Lets the SliceFusion RHS scalarizer
    recognise ``w[idx[:, :, 0] - 1, jk, blk[:, :, 0] - 1]`` as an advanced-index
    group, not a plain scalar axis."""
    if isinstance(expr, ast.Subscript):
        name = _name_id(expr.value)
        if name and shape_table.get(name):
            n = sum(1 for a in _slice_axes(expr) if isinstance(a, ast.Slice))
            return n or None
        return None
    if isinstance(expr, ast.Name):
        # A bare index ARRAY is an advanced index of its own rank. Only the sliced spelling was
        # recognised, so an OFFSET gather (``coulomb_table_f[ri + 1]``) read as a scalar axis.
        shape = shape_table.get(expr.id)
        return len(shape) if shape else None
    if isinstance(expr, ast.BinOp):
        return (_advanced_index_rank(expr.left, shape_table) or _advanced_index_rank(expr.right, shape_table))
    if isinstance(expr, ast.UnaryOp):
        return _advanced_index_rank(expr.operand, shape_table)
    return None


def _slice_start(ax: ast.Slice, axis_len: Optional[ast.expr], step) -> Optional[ast.expr]:
    """First SOURCE index a slice reads. ``lower`` when given (negative resolved
    against ``axis_len``), else 0 -- except under a NEGATIVE step, where numpy
    flips the default and starts at the last element ``axis_len - 1``
    (``a[::-1]``). A reverse slice over an untracked axis cannot be indexed at
    all; refuse loudly rather than emit the forward ``a[i]`` that would silently
    drop the reversal (mirrors lowering's slice-assign rewriters).

    ``step`` is a literal int or a symbolic step expression; only a literal can be the reverse."""
    if ax.lower is not None:
        return _resolve_negative(ax.lower, axis_len)
    if _step_is_negative(step):
        if axis_len is None:
            raise NotImplementedError("reverse slice needs a known axis length (shape untracked)")
        return ast.BinOp(left=axis_len, op=ast.Sub(), right=_const(1))
    return _const(0)


def _strided_index(ivar: ast.expr, start: Optional[ast.expr], step) -> ast.expr:
    """Source index of result position ``ivar`` within a slice ``[start::step]``:
    ``start + ivar * step``. Must stay in lockstep with :func:`_iter_extent_of`,
    which counts ``ceil(extent / |step|)`` elements -- an index that ignored
    ``step`` would walk a DIFFERENT (contiguous) run of the same length.

    ``step`` is a literal int or a symbolic step expression; both multiply the position."""
    unit = step is None or (isinstance(step, int) and step == 1)
    pos: ast.expr = ivar if unit else ast.BinOp(left=ivar, op=ast.Mult(), right=_step_node(step))
    if isinstance(start, ast.Constant) and start.value == 0:
        return pos
    return ast.BinOp(left=pos, op=ast.Add(), right=start)


def _scalarize_at_iters(expr: ast.expr, iters: List[ast.expr], shape_table: Dict[str, Tuple[str, ...]]) -> ast.expr:
    """Render an array-valued expression at the given iter indices. Recursive
    structural lowering, independent of any one numpy op: ``Name(A)`` ->
    ``A[iters]``; ``Subscript(A, axes)`` -> walk axes, each Slice axis consumes
    one iter (offset by ``slice.lower``), scalar axes kept as-is;
    ``BinOp``/``UnaryOp``/``Call``/``IfExp`` recurse on children; ``Constant``
    unchanged.
    """
    if isinstance(expr, ast.Constant):
        return expr
    if isinstance(expr, ast.Name):
        shape = shape_table.get(expr.id)
        if shape is None:
            return expr
        if len(shape) > len(iters):
            return expr
        # Right-align the operand's axes against the iter nest (numpy broadcasts
        # along the leading axes); index any size-1 axis with constant 0, since a
        # length-1 axis broadcasts and must not consume the iter. softmax/mlp's
        # keepdims ``tmp_max`` is (N, H, SM, 1) and must read ``tmp_max[i, j, k,
        # 0]``, not ``tmp_max[..., r3]`` out of bounds.
        offset = len(iters) - len(shape)
        elts = [ast.Constant(value=0) if s == "1" else iters[offset + i] for i, s in enumerate(shape)]
        slot = elts[0] if len(elts) == 1 else ast.Tuple(elts=list(elts), ctx=ast.Load())
        return ast.Subscript(value=expr, slice=slot, ctx=ast.Load())
    if isinstance(expr, ast.Subscript):
        name = _name_id(expr.value)
        shape = shape_table.get(name) if name else None
        axes = _slice_axes(expr)
        new_axes: List[ast.expr] = []
        iter_idx = 0
        src_axis = 0  # source-axis pointer (see _iter_extent_of).
        group_iters: Optional[List[ast.expr]] = None  # shared advanced-index iters
        for ax in axes:
            if isinstance(ax, ast.Constant) and ax.value is None:
                # newaxis -- consume one iter from the result-axis side but
                # contribute no source index. The size-1 result axis maps
                # every read to ``source[...]`` (constant).
                if iter_idx < len(iters):
                    iter_idx += 1
                continue
            if isinstance(ax, ast.Slice):
                axis_len = (_const_or_name(shape[src_axis]) if shape and src_axis < len(shape) else None)
                step = _slice_step_any(ax)
                lo = _slice_start(ax, axis_len, step)
                if iter_idx >= len(iters):
                    return expr  # not enough iters supplied
                ivar = iters[iter_idx]
                iter_idx += 1
                new_axes.append(_strided_index(ivar, lo, step))
            elif isinstance(ax, ast.Name) and shape_table.get(ax.id):
                # Fancy-index gather: ``arr[idx]`` -> ``arr[idx[k]]``. Multiple index
                # arrays in one subscript form a numpy advanced-index GROUP that
                # broadcasts to a single result-axis set and SHARES the iters:
                # ``u2[q, r, s]`` -> ``u2[q[m], r[m], s[m]]`` (one iter ``m``, not
                # three). Consumed once, at the first index-array axis, reused after.
                idx_shape = shape_table[ax.id]
                if group_iters is None:
                    if iter_idx + len(idx_shape) > len(iters):
                        return expr  # not enough iters supplied
                    group_iters = iters[iter_idx:iter_idx + len(idx_shape)]
                    iter_idx += len(idx_shape)
                idx_iters = group_iters[-len(idx_shape):]
                if len(idx_iters) == 1:
                    new_axes.append(ast.Subscript(value=ax, slice=idx_iters[0], ctx=ast.Load()))
                else:
                    new_axes.append(
                        ast.Subscript(value=ax, slice=ast.Tuple(elts=list(idx_iters), ctx=ast.Load()), ctx=ast.Load()))
                src_axis += 1
                continue
            else:
                # Advanced-index EXPRESSION axis (``edge_idx[:, :, 0] - 1``):
                # part of the same broadcast group as any bare-Name index, with
                # shared iters. Recurse to scalarize its nested slices.
                adv_rank = _advanced_index_rank(ax, shape_table)
                if adv_rank:
                    if group_iters is None:
                        if iter_idx + adv_rank > len(iters):
                            return expr  # not enough iters supplied
                        group_iters = iters[iter_idx:iter_idx + adv_rank]
                        iter_idx += adv_rank
                    idx_iters = group_iters[-adv_rank:]
                    new_axes.append(_scalarize_at_iters(ax, idx_iters, shape_table))
                    src_axis += 1
                    continue
                # Concrete scalar index -- resolve a negative ``arr[-1]`` against
                # the axis length (C / Fortran have no negative indexing): the
                # stencil_*_vc ``w_dist[-1]`` last-weight read.
                axis_len = (_const_or_name(shape[src_axis]) if shape and src_axis < len(shape) else None)
                new_axes.append(_resolve_negative(ax, axis_len))
            src_axis += 1
        # If the source has more axes than the Subscript covered, the iter
        # nest may carry additional trailing iters that map straight to
        # the missing source axes.
        while src_axis < (len(shape) if shape else 0) and iter_idx < len(iters):
            new_axes.append(iters[iter_idx])
            iter_idx += 1
            src_axis += 1
        if not new_axes:
            return expr.value
        slot = new_axes[0] if len(new_axes) == 1 else ast.Tuple(elts=new_axes, ctx=ast.Load())
        return ast.Subscript(value=expr.value, slice=slot, ctx=ast.Load())
    if isinstance(expr, ast.BinOp):
        return ast.BinOp(left=_scalarize_at_iters(expr.left, iters, shape_table),
                         op=expr.op,
                         right=_scalarize_at_iters(expr.right, iters, shape_table))
    if isinstance(expr, ast.UnaryOp):
        return ast.UnaryOp(op=expr.op, operand=_scalarize_at_iters(expr.operand, iters, shape_table))
    if isinstance(expr, ast.Compare):
        return ast.Compare(left=_scalarize_at_iters(expr.left, iters, shape_table),
                           ops=expr.ops,
                           comparators=[_scalarize_at_iters(c, iters, shape_table) for c in expr.comparators])
    if isinstance(expr, ast.BoolOp):
        return ast.BoolOp(op=expr.op, values=[_scalarize_at_iters(v, iters, shape_table) for v in expr.values])
    if isinstance(expr, ast.IfExp):
        return ast.IfExp(test=_scalarize_at_iters(expr.test, iters, shape_table),
                         body=_scalarize_at_iters(expr.body, iters, shape_table),
                         orelse=_scalarize_at_iters(expr.orelse, iters, shape_table))
    if isinstance(expr, ast.Call):
        # An array CONSTRUCTOR is not elementwise: every element of ``np.zeros_like(a)`` is 0,
        # whatever ``a`` is. Recursing into the args instead emitted a per-element call to the
        # constructor itself (``__t[i, j] = np.zeros_like(...)``), which no backend can render.
        fill = _ctor_fill_element(expr)
        if fill is not None:
            return fill
        # Math intrinsics on array values fall through; the args are
        # array expressions to scalarize.
        return ast.Call(func=expr.func,
                        args=[_scalarize_at_iters(a, iters, shape_table) for a in expr.args],
                        keywords=expr.keywords)
    return expr


#: Element value of an array constructor, for the constructors whose fill is DEFINED. ``empty`` /
#: ``empty_like`` / ``ndarray`` are absent on purpose: their contents are whatever the allocation
#: held, and naming a value for them would put an invented number into the emitted kernel.
_CTOR_FILL: Dict[str, float] = {"zeros": 0.0, "zeros_like": 0.0, "ones": 1.0, "ones_like": 1.0}


def _ctor_fill_element(expr: ast.Call) -> Optional[ast.expr]:
    """The scalar every element of ``np.zeros(...)`` / ``np.ones_like(...)`` / ``np.full(...)``
    holds, or ``None`` when the call is not such a constructor."""
    if not (isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name)
            and expr.func.value.id in ("np", "numpy")):
        return None
    attr = expr.func.attr
    if attr in ("full", "full_like") and len(expr.args) >= 2:
        return copy.deepcopy(expr.args[1])
    value = _CTOR_FILL.get(attr)
    return None if value is None else _const(value)


def _eval_axes(node) -> Optional[List[int]]:
    """``[k]`` / ``[k1, k2, ...]`` for a literal axis spec, ``None`` when it is not one.

    ``None`` here means UNREADABLE, which is not the same as "no axis given" -- callers have to
    keep the two apart themselves, because only they know whether the slot they read is an axis.
    """
    value = _const_int(node)
    if value is not None:
        return [value]
    if isinstance(node, (ast.Tuple, ast.List)):
        out = []
        for elt in node.elts:
            element = _const_int(elt)
            if element is None:
                return None
            out.append(element)
        return out
    return None


def _read_axis_keepdims(args, kwargs):
    """Return ``(axes, keepdims)`` from a call, keyword or positional. ``axes``:
    ``None`` for full reduction (``np.X(arr)``); ``[k]`` for single-axis
    (``np.X(arr, axis=k)``, negative ``axis=-1`` accepted); ``[k1, k2, ...]`` for
    multi-axis (``axis=(1, 2, 3)``/``axis=[1, 2, 3]``, order preserved for the
    reduction loop nest, though kept-axes ordering follows the source array).
    """
    # ``axes = None`` means "reduce over EVERY axis" downstream, so it may only ever come from an
    # ABSENT argument. An ``axis=`` KEYWORD that is present but unreadable is refused instead:
    # reading it as None turned ``np.sum(x, axis=dim)`` into a full reduction that compiled clean.
    #
    # The POSITIONAL slot stays best-effort here on purpose. This reader is shared with calls whose
    # second positional argument is not an axis at all -- ``np.logaddexp(a, b)``, ``np.heaviside(a,
    # b)``, ``np.isclose(a, b, 1e-12)`` -- so refusing on it rejects the second OPERAND. A caller
    # that knows its slot 1 really is an axis checks it itself; see ``_expand_axis_reduction``.
    axes = None
    if len(args) >= 2:
        axes = _eval_axes(args[1])
    for kw in kwargs or []:
        if kw.arg == "axis":
            if isinstance(kw.value, ast.Constant) and kw.value.value is None:
                axes = None
                continue
            axes = _eval_axes(kw.value)
            if axes is None:
                raise NotImplementedError(f"axis {ast.unparse(kw.value)!r} must be a compile-time integer or "
                                          f"tuple of them (it selects the loop nest)")
    keepdims = False
    for kw in kwargs or []:
        if kw.arg == "keepdims":
            # Same rule: a non-literal keepdims changes the RESULT RANK, so it cannot default to False.
            if not isinstance(kw.value, ast.Constant):
                raise NotImplementedError(f"keepdims {ast.unparse(kw.value)!r} must be a compile-time "
                                          f"constant (it selects the result rank)")
            keepdims = bool(kw.value.value)
    return axes, keepdims


def _expand_axis_reduction(target, args, kwargs, shape_table, init, op_fn, post_fn=None, update_fn=None):
    """Generic axis-aware reduction. Lowers ``out = np.X(arr, axis=k,
    keepdims=True)`` into a nested loop, non-reduction axes outside and the
    reduction axis inside; writes through to ``out`` at the kept axes (axis
    ``k`` writes ``out[..., 0, ...]`` when ``keepdims=True``). Full reduction
    (``axis=None``) walks all axes and writes one scalar to ``target``.

    :param post_fn: optional ``(target_lvalue, divisor) -> ast.stmt`` invoked
        after the loop closes; used by mean to divide by the reduction size.
    :param update_fn: optional ``(store, load, src) -> ast.stmt`` overriding the
        default ``store = op_fn(load, src)``; used by if-guarded boolean
        reductions (any/all/count_nonzero), which can't rely on C's
        bool-as-int arithmetic (invalid in Fortran).

    A float sum accumulates in ONE chain, matching the source order rather than numpy's pairwise
    blocking: reassociation is sanctioned, so the extra block loop bought only a scop that pet
    refuses (POLYCC-008) and a scop-external block accumulator it silently drops (POLYCC-009).
    """
    arr = args[0]
    shape = _resolve_shape(arr, shape_table)
    # Here slot 1 REALLY is the axis (``np.sum(a, 1)``), unlike the shared reader's general case, so
    # an unreadable one is refused rather than silently becoming a reduction over every axis.
    if len(args) >= 2 and not (isinstance(args[1], ast.Constant) and args[1].value is None):
        if _eval_axes(args[1]) is None:
            raise NotImplementedError(f"axis {ast.unparse(args[1])!r} must be a compile-time integer or tuple "
                                      f"of them (it selects the loop nest)")
    axes, keepdims = _read_axis_keepdims(args, kwargs)
    n_dim = len(shape)

    # ``initial=`` (sum/prod/max/min) seeds the reduction instead of the default
    # identity/first element. The loop still walks every element (max/min are
    # idempotent, so re-including index 0 is harmless) -- numpy's
    # ``op(initial, *elements)``.
    initial = _read_kwarg(kwargs, "initial")
    if initial is not None:
        init = initial

    # ``where=`` masks which elements take part and ``dtype=`` pins the ACCUMULATOR type (the
    # classic case being a float64 accumulator over a float32 operand, which changes the result,
    # not just its storage). Neither is honoured by the loop below, so neither may be ignored.
    for dropped in ("where", "dtype", "out"):
        if _read_kwarg(kwargs, dropped) is not None:
            raise NotImplementedError(f"reduction {dropped}= is not lowered; it changes the result, "
                                      f"so it cannot be dropped")

    if axes is None:
        # Full reduction -- scalar target.
        iters = [_make_iter_name("__r", i) for i in range(n_dim)]
        subscript = ast.Subscript(
            value=_name(arr.id),
            slice=(_name(iters[0]) if n_dim == 1 else ast.Tuple(elts=[_name(i) for i in iters], ctx=ast.Load())),
            ctx=ast.Load())
        target_load = ast.Name(id=target.id, ctx=ast.Load())
        body = [
            update_fn(target, target_load, subscript) if update_fn else ast.Assign(targets=[target],
                                                                                   value=op_fn(target_load, subscript))
        ]
        loops = _wrap_for_loops(iters, shape, body)
        stmts = [ast.Assign(targets=[target], value=_init_for(init, arr, n_dim))]
        stmts.extend(loops)
        if post_fn is not None:
            stmts.append(post_fn(target, _shape_total_product(shape)))
        return stmts

    # Axis-aware reduction. ``axes`` is a list of one or more axis
    # indices to reduce. Negative axes resolve mod n_dim; duplicates
    # are rejected.
    axes_norm: List[int] = []
    for a in axes:
        na = a + n_dim if a < 0 else a
        if na < 0 or na >= n_dim:
            raise NotImplementedError(f"axis {a} out of range for ndim {n_dim}")
        if na in axes_norm:
            raise NotImplementedError(f"duplicate axis {a} in reduction tuple")
        axes_norm.append(na)
    axes_set = set(axes_norm)
    # Outer iter names walk the kept axes (those NOT in axes_set);
    # one inner iter per reduction axis.
    kept_axes = [k for k in range(n_dim) if k not in axes_set]
    outer_iter_names = [_make_iter_name("__ax", i) for i in range(len(kept_axes))]
    red_iter_names = [_make_iter_name("__rd", i) for i in range(len(axes_norm))]
    red_iter_map = dict(zip(axes_norm, red_iter_names))

    def _src_elts() -> List[ast.expr]:
        out = []
        outer_pos = 0
        for k in range(n_dim):
            if k in axes_set:
                out.append(_name(red_iter_map[k]))
            else:
                out.append(_name(outer_iter_names[outer_pos]))
                outer_pos += 1
        return out

    def _out_elts() -> List[ast.expr]:
        out = []
        outer_pos = 0
        for k in range(n_dim):
            if k in axes_set:
                if keepdims:
                    out.append(_const(0))
            else:
                out.append(_name(outer_iter_names[outer_pos]))
                outer_pos += 1
        return out

    src_elts = _src_elts()
    src_slot = src_elts[0] if n_dim == 1 else ast.Tuple(elts=src_elts, ctx=ast.Load())
    out_elts = _out_elts()
    if len(out_elts) == 0:
        # No kept axes and no keepdims -- scalar result. Falls back to
        # the full-reduction style.
        out_sub = target
        out_load = ast.Name(id=target.id, ctx=ast.Load())
    elif len(out_elts) == 1:
        out_sub = ast.Subscript(value=_name(target.id), slice=out_elts[0], ctx=ast.Store())
        out_load = ast.Subscript(value=_name(target.id), slice=out_elts[0], ctx=ast.Load())
    else:
        out_slot = ast.Tuple(elts=out_elts, ctx=ast.Load())
        out_sub = ast.Subscript(value=_name(target.id), slice=out_slot, ctx=ast.Store())
        out_load = ast.Subscript(value=_name(target.id), slice=out_slot, ctx=ast.Load())
    src_sub = ast.Subscript(value=_name(arr.id), slice=src_slot, ctx=ast.Load())
    # Init for axis-reductions: ``out[outer..] = init`` (or the
    # zero-th element of the reduction axes for max/min).
    if isinstance(init, ast.Subscript):
        init_src_elts = []
        outer_pos2 = 0
        for k in range(n_dim):
            if k in axes_set:
                init_src_elts.append(_const(0))
            else:
                init_src_elts.append(_name(outer_iter_names[outer_pos2]))
                outer_pos2 += 1
        init_slot = init_src_elts[0] if n_dim == 1 else ast.Tuple(elts=init_src_elts, ctx=ast.Load())
        init_node = ast.Subscript(value=_name(arr.id), slice=init_slot, ctx=ast.Load())
    else:
        init_node = init
    init_stmt = ast.Assign(targets=[out_sub], value=init_node)
    update_stmt = (update_fn(out_sub, out_load, src_sub) if update_fn else ast.Assign(targets=[out_sub],
                                                                                      value=op_fn(out_load, src_sub)))
    # Inner loop nest over the reduction axes, deepest first.
    inner_stmts: List[ast.stmt] = [update_stmt]
    for ax, rn in zip(reversed(axes_norm), reversed(red_iter_names)):
        inner_stmts = [
            ast.For(target=_store(rn),
                    iter=ast.Call(func=_name("range"), args=[_const_or_name(shape[ax])], keywords=[]),
                    body=inner_stmts,
                    orelse=[])
        ]
    if post_fn is not None:
        # Divisor for mean: product of the reduction-axis sizes.
        divisor = _const_or_name(shape[axes_norm[0]])
        for ax in axes_norm[1:]:
            divisor = ast.BinOp(left=divisor, op=ast.Mult(), right=_const_or_name(shape[ax]))
        inner_stmts.append(post_fn(out_sub, divisor))
    body = [init_stmt] + inner_stmts
    bounds = tuple(shape[k] for k in kept_axes)
    if not bounds:
        # No kept axes (all reduced; equivalent to full reduction).
        return body
    return _wrap_for_loops(outer_iter_names, bounds, body)


def _init_for(init, arr, n_dim):
    """Resolve init for full reduction: rewrite max/min first-element to
    a fully-zeroed subscript if needed."""
    if isinstance(init, ast.Subscript):
        # For full reduction we use arr[0, 0, ..., 0] as the init.
        if n_dim == 1:
            return ast.Subscript(value=_name(arr.id), slice=_const(0), ctx=ast.Load())
        return ast.Subscript(value=_name(arr.id),
                             slice=ast.Tuple(elts=[_const(0)] * n_dim, ctx=ast.Load()),
                             ctx=ast.Load())
    return init


def _read_kwarg(kwargs, name):
    """Return the AST value of keyword ``name`` in ``kwargs`` (list of
    ``ast.keyword``), or ``None`` when absent."""
    for kw in (kwargs or []):
        if kw.arg == name:
            return kw.value
    return None


def _reduction_elem_is_integer(args, local_dtypes):
    """True when the reduced array (first arg, a bare Name) is tagged an
    integer / boolean dtype -- numpy upcasts int8/16/32/bool to int64 for
    ``sum`` / ``prod``, so the accumulator must be an integer, not a float."""
    if not local_dtypes or not args or not isinstance(args[0], ast.Name):
        return False
    dt = local_dtypes.get(args[0].id)
    return dt is not None and dt.startswith(("int", "uint", "bool"))


def _nan_reduce_op(cmp):
    """Running max/min update that propagates NaN like numpy: ``x if (x <cmp> acc
    or x != x) else acc``. The ``x != x`` test lets a NaN element win, and once
    the accumulator is NaN it stays (nothing compares ``<cmp>`` against a NaN).
    Matches numpy (``np.max``/``np.min`` return NaN if any element is NaN),
    unlike C's ``fmax``/the ``max`` macro, which suppress NaN."""

    def _f(acc, x):
        return ast.IfExp(test=ast.BoolOp(op=ast.Or(),
                                         values=[
                                             ast.Compare(left=copy.deepcopy(x),
                                                         ops=[cmp()],
                                                         comparators=[copy.deepcopy(acc)]),
                                             ast.Compare(left=copy.deepcopy(x),
                                                         ops=[ast.NotEq()],
                                                         comparators=[copy.deepcopy(x)])
                                         ]),
                         body=copy.deepcopy(x),
                         orelse=copy.deepcopy(acc))

    return _f


def expand_sum(target, args, shape_table, kwargs=None, local_dtypes=None):
    is_int = _reduction_elem_is_integer(args, local_dtypes)
    if is_int and local_dtypes is not None and isinstance(target, ast.Name):
        local_dtypes[target.id] = "int64"
    return _expand_axis_reduction(target,
                                  args,
                                  kwargs,
                                  shape_table,
                                  init=_const(0) if is_int else _const(0.0),
                                  op_fn=lambda acc, x: ast.BinOp(left=acc, op=ast.Add(), right=x))


def expand_max(target, args, shape_table, kwargs=None):
    arr = args[0]
    _reject_zero_size_reduction(args, kwargs, shape_table)
    return _expand_axis_reduction(target,
                                  args,
                                  kwargs,
                                  shape_table,
                                  init=ast.Subscript(value=arr, slice=_const(0), ctx=ast.Load()),
                                  op_fn=_nan_reduce_op(ast.Gt))


def expand_min(target, args, shape_table, kwargs=None):
    arr = args[0]
    _reject_zero_size_reduction(args, kwargs, shape_table)
    return _expand_axis_reduction(target,
                                  args,
                                  kwargs,
                                  shape_table,
                                  init=ast.Subscript(value=arr, slice=_const(0), ctx=ast.Load()),
                                  op_fn=_nan_reduce_op(ast.Lt))


def _reject_zero_size_reduction(args, kwargs, shape_table):
    """Refuse to lower ``np.max``/``np.min`` over a statically zero-length
    reduction axis: numpy raises ``zero-size array to reduction ... which has no
    identity``, and the seed ``arr[..., 0]`` would read OOB. Raise
    ``NotImplementedError`` instead of emitting an OOB seed. (A symbolic extent
    that's 0 only at runtime can't be caught here.)"""
    if not args or not isinstance(args[0], ast.Name):
        return
    shape = shape_table.get(args[0].id)
    if not shape:
        return
    n_dim = len(shape)
    axes, _ = _read_axis_keepdims(args, kwargs)
    red_axes = range(n_dim) if axes is None else [a + n_dim if a < 0 else a for a in axes]
    for ax in red_axes:
        if 0 <= ax < n_dim and str(shape[ax]) == "0":
            raise NotImplementedError("zero-size array to reduction which has no identity")


def expand_mean(target, args, shape_table, kwargs=None):
    # Special form ``np.mean(arr[mask])``, ``mask`` a same-length boolean array:
    # boolean fancy indexing produces a dynamic-length compacted view we don't
    # materialise, so emit a masked-sum + count loop directly.
    if (args and isinstance(args[0], ast.Subscript) and isinstance(args[0].value, ast.Name)
            and isinstance(args[0].slice, ast.Name)):
        arr = args[0].value
        mask = args[0].slice
        a_shape = shape_table.get(arr.id)
        if a_shape and len(a_shape) == 1:
            n_ast = _const_or_name(a_shape[0])
            iter_name = "__mn_i"
            sum_name = "__mn_sum"
            cnt_name = "__mn_cnt"
            body = [
                ast.If(test=ast.Subscript(value=_name(mask.id), slice=_name(iter_name), ctx=ast.Load()),
                       body=[
                           ast.AugAssign(target=_store(sum_name),
                                         op=ast.Add(),
                                         value=ast.Subscript(value=_name(arr.id),
                                                             slice=_name(iter_name),
                                                             ctx=ast.Load())),
                           ast.AugAssign(target=_store(cnt_name), op=ast.Add(), value=_const(1)),
                       ],
                       orelse=[]),
            ]
            return [
                ast.Assign(targets=[_store(sum_name)], value=_const(0.0)),
                ast.Assign(targets=[_store(cnt_name)], value=_const(0)),
                ast.For(target=_store(iter_name),
                        iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                        body=body,
                        orelse=[]),
                ast.Assign(targets=[_store(target.id)],
                           value=ast.BinOp(left=_name(sum_name), op=ast.Div(), right=_name(cnt_name))),
            ]
    return _expand_axis_reduction(target,
                                  args,
                                  kwargs,
                                  shape_table,
                                  init=_const(0.0),
                                  op_fn=lambda acc, x: ast.BinOp(left=acc, op=ast.Add(), right=x),
                                  post_fn=lambda lvalue, divisor: ast.Assign(
                                      targets=[lvalue],
                                      value=ast.BinOp(left=(lvalue if isinstance(lvalue, ast.Name) else ast.Subscript(
                                          value=lvalue.value, slice=lvalue.slice, ctx=ast.Load())),
                                                      op=ast.Div(),
                                                      right=divisor)))


def expand_prod(target, args, shape_table, kwargs=None, local_dtypes=None):
    is_int = _reduction_elem_is_integer(args, local_dtypes)
    if is_int and local_dtypes is not None and isinstance(target, ast.Name):
        local_dtypes[target.id] = "int64"
    return _expand_axis_reduction(target,
                                  args,
                                  kwargs,
                                  shape_table,
                                  init=_const(1) if is_int else _const(1.0),
                                  op_fn=lambda acc, x: ast.BinOp(left=acc, op=ast.Mult(), right=x))


def _truthy(x):
    """``x != 0`` -- element truthiness. On a boolean mask the Fortran emitter
    folds ``<logical> /= 0`` back to the bare logical; C reads it as 0/1."""
    return ast.Compare(left=x, ops=[ast.NotEq()], comparators=[_const(0)])


def _falsy(x):
    return ast.Compare(left=x, ops=[ast.Eq()], comparators=[_const(0)])


def _if_set(test_fn, value_fn):
    """Build an ``update_fn`` that, per element, tests ``test_fn(src)`` and on
    hit assigns ``value_fn(load)`` to the accumulator. Keeps the accumulator
    INTEGER (0/1 or a count) so no backend needs bool-as-int arithmetic (which
    Fortran rejects)."""

    def _f(store, load, src):
        return ast.If(test=test_fn(src), body=[ast.Assign(targets=[store], value=value_fn(load))], orelse=[])

    return _f


def expand_any(target, args, shape_table, kwargs=None):
    """``s = np.any(A [, axis=k, keepdims=...])`` -- OR reduction. Init=0; each
    truthy element sets the (integer 0/1) accumulator to 1."""
    return _expand_axis_reduction(target,
                                  args,
                                  kwargs,
                                  shape_table,
                                  init=_const(0),
                                  op_fn=None,
                                  update_fn=_if_set(_truthy, lambda load: _const(1)))


def expand_all(target, args, shape_table, kwargs=None):
    """``s = np.all(A [, axis=k, keepdims=...])`` -- AND reduction. Init=1; each
    falsy element clears the (integer 0/1) accumulator to 0."""
    return _expand_axis_reduction(target,
                                  args,
                                  kwargs,
                                  shape_table,
                                  init=_const(1),
                                  op_fn=None,
                                  update_fn=_if_set(_falsy, lambda load: _const(0)))


def expand_count_nonzero(target, args, shape_table, kwargs=None):
    """``s = np.count_nonzero(A [, axis=k, keepdims=...])`` -- count of non-zero
    elements. Init=0; each truthy element increments the integer accumulator."""
    return _expand_axis_reduction(target,
                                  args,
                                  kwargs,
                                  shape_table,
                                  init=_const(0),
                                  op_fn=None,
                                  update_fn=_if_set(_truthy,
                                                    lambda load: ast.BinOp(left=load, op=ast.Add(), right=_const(1))))


def expand_argmax(target, args, shape_table, kwargs=None):
    """``i = np.argmax(A [, axis=k, keepdims=...])`` -- index of the maximum.
    Only ``axis=None`` (flat, scalar result) and ``axis=int`` are implemented;
    axis-tuple raises ``NotImplementedError`` (caller can express it via a
    reshape + flat argmax; TODO if needed)."""
    return _expand_arg_reduction(target, args, shape_table, kwargs, op="argmax")


def expand_argmin(target, args, shape_table, kwargs=None):
    return _expand_arg_reduction(target, args, shape_table, kwargs, op="argmin")


def _expand_arg_reduction(target, args, shape_table, kwargs, op: str):
    """``argmax``/``argmin`` shared scaffold. Supports the full ``axis = None /
    int / tuple / list`` matrix: ``None`` is a full reduction to a single flat
    index; an int reduces one axis, keeping the others at the input's extent,
    indexed 0..shape[axis]-1; a tuple ravels the chosen axes and returns the
    flat index across them (output keeps the axes not in the tuple; per
    kept-axes position, walk the reduction axes in source order tracking
    (best_val, best_flat_idx), with the flat index using row-major mapping over
    the reduction-axis sizes). ``keepdims=True`` adds a size-1 axis at each
    reduced position in all three forms.
    """
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError(f"np.{op} needs Name first arg")
    a = args[0]
    shape = _resolve_shape(a, shape_table)
    axes, keepdims = _read_axis_keepdims(args, kwargs)
    n_dim = len(shape)
    cmp_op = ast.Gt() if op == "argmax" else ast.Lt()
    # Normalise axes -> set + ordered list (for flat-index mapping).
    if axes is None:
        axes_norm = list(range(n_dim))
    else:
        axes_norm = []
        for ax in axes:
            na = ax + n_dim if ax < 0 else ax
            if na < 0 or na >= n_dim:
                raise NotImplementedError(f"np.{op} axis {ax} out of range for ndim {n_dim}")
            if na in axes_norm:
                raise NotImplementedError(f"np.{op} duplicate axis {ax}")
            axes_norm.append(na)
    axes_set = set(axes_norm)
    kept_axes = [k for k in range(n_dim) if k not in axes_set]
    outer_iter_names = [_make_iter_name("__aax", i) for i in range(len(kept_axes))]
    red_iter_names = [_make_iter_name("__ard", i) for i in range(len(axes_norm))]
    red_iter_map = dict(zip(axes_norm, red_iter_names))

    def _src_elts():
        out = []
        outer_pos = 0
        for k in range(n_dim):
            if k in axes_set:
                out.append(_name(red_iter_map[k]))
            else:
                out.append(_name(outer_iter_names[outer_pos]))
                outer_pos += 1
        return out

    def _out_elts():
        out = []
        outer_pos = 0
        for k in range(n_dim):
            if k in axes_set:
                if keepdims:
                    out.append(_const(0))
            else:
                out.append(_name(outer_iter_names[outer_pos]))
                outer_pos += 1
        return out

    def _init_src_elts():
        # First-element init: reduction axes pinned at 0, kept axes at
        # outer iter.
        out = []
        outer_pos = 0
        for k in range(n_dim):
            if k in axes_set:
                out.append(_const(0))
            else:
                out.append(_name(outer_iter_names[outer_pos]))
                outer_pos += 1
        return out

    src_elts = _src_elts()
    src_slot = (src_elts[0] if n_dim == 1 else ast.Tuple(elts=src_elts, ctx=ast.Load()))
    src_sub = ast.Subscript(value=_name(a.id), slice=src_slot, ctx=ast.Load())
    out_elts = _out_elts()
    init_src_elts = _init_src_elts()
    init_slot = (init_src_elts[0] if n_dim == 1 else ast.Tuple(elts=init_src_elts, ctx=ast.Load()))
    init_val = ast.Subscript(value=_name(a.id), slice=init_slot, ctx=ast.Load())
    if not out_elts:
        # Scalar target -- full reduction with no kept axes.
        out_sub: ast.expr = target
    elif len(out_elts) == 1:
        out_sub = ast.Subscript(value=_name(target.id), slice=out_elts[0], ctx=ast.Store())
    else:
        out_sub = ast.Subscript(value=_name(target.id), slice=ast.Tuple(elts=out_elts, ctx=ast.Load()), ctx=ast.Store())
    # The running-best temp is named after its TARGET. A fixed name collided across two argmax
    # expansions in one kernel, and when the two operands had different dtypes the second
    # expansion's temp inherited the first's declared type -- cp2k_density_matrix_trs4 compared an
    # integer temp with .eqv. because the other argmax in the same body ran over a boolean plane.
    best_val = f"__ar_val_{target.id}" if isinstance(target, ast.Name) else "__ar_val"
    init_stmts: List[ast.stmt] = [
        ast.Assign(targets=[_store(best_val)], value=init_val),
        ast.Assign(targets=[out_sub], value=_const(0)),
    ]
    # Flat index across the reduction axes (in source order):
    #   ((red_iter[0] * shape[axis1]) + red_iter[1]) * shape[axis2]
    #     + red_iter[2] + ...
    if len(axes_norm) == 1:
        flat_idx: ast.expr = _name(red_iter_map[axes_norm[0]])
    else:
        flat_idx = _name(red_iter_map[axes_norm[0]])
        for k in range(1, len(axes_norm)):
            flat_idx = ast.BinOp(left=ast.BinOp(left=flat_idx, op=ast.Mult(),
                                                right=_const_or_name(shape[axes_norm[k]])),
                                 op=ast.Add(),
                                 right=_name(red_iter_map[axes_norm[k]]))
    # NaN semantics (numpy): argmax/argmin return the index of the FIRST NaN.
    # Update rule ``(best == best) and (src != src or src <cmp> best)``:
    # ``best == best`` goes false once ``best`` is NaN, locking the index at the
    # first NaN; ``src != src`` lets a NaN element win (sets best = NaN); else the
    # ordinary ``src <cmp> best`` drives the arg. (Element 0 already NaN: the seed
    # ``best == best`` is false and the index stays 0.)
    best_not_nan = ast.Compare(left=_name(best_val), ops=[ast.Eq()], comparators=[_name(best_val)])
    src_is_nan = ast.Compare(left=copy.deepcopy(src_sub), ops=[ast.NotEq()], comparators=[copy.deepcopy(src_sub)])
    ordinary = ast.Compare(left=copy.deepcopy(src_sub), ops=[cmp_op], comparators=[_name(best_val)])
    update = ast.If(test=ast.BoolOp(op=ast.And(),
                                    values=[best_not_nan,
                                            ast.BoolOp(op=ast.Or(), values=[src_is_nan, ordinary])]),
                    body=[
                        ast.Assign(targets=[_store(best_val)], value=copy.deepcopy(src_sub)),
                        ast.Assign(targets=[out_sub], value=flat_idx),
                    ],
                    orelse=[])
    # Wrap the comparison in nested reduction loops, deepest first.
    inner_body: List[ast.stmt] = [update]
    for ax, rn in zip(reversed(axes_norm), reversed(red_iter_names)):
        inner_body = [
            ast.For(target=_store(rn),
                    iter=ast.Call(func=_name("range"), args=[_const_or_name(shape[ax])], keywords=[]),
                    body=inner_body,
                    orelse=[])
        ]
    body_stmts = init_stmts + inner_body
    if not kept_axes:
        return body_stmts
    bounds = tuple(shape[k] for k in kept_axes)
    return _wrap_for_loops(outer_iter_names, bounds, body_stmts)


def expand_matmul(target: ast.expr, lhs: ast.expr, rhs: ast.expr, shape_table: Dict[str, Tuple[str,
                                                                                               ...]]) -> List[ast.stmt]:
    """Lower ``C = A @ B`` to the naive ``M x K x N`` triple-loop GEMM. Both
    operands must be Name expressions with a declared shape in the shape table.
    """
    if not (isinstance(lhs, ast.Name) and isinstance(rhs, ast.Name)):
        raise NotImplementedError("matmul operand is not a bare Name")
    a_name, b_name = lhs.id, rhs.id
    a_shape = shape_table.get(a_name)
    b_shape = shape_table.get(b_name)
    if not a_shape or not b_shape:
        raise NotImplementedError("matmul shapes not resolvable from IR")
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise NotImplementedError("only 2-D matmul supported")
    m, k = a_shape
    k2, n = b_shape
    # ``k`` and ``k2`` should be the same symbol; emit the LHS one and let the
    # compiler / numpy oracle catch divergence.
    body = [
        ast.Assign(targets=[
            ast.Subscript(value=_name(target.id),
                          slice=ast.Tuple(elts=[_name("__i"), _name("__j")], ctx=ast.Load()),
                          ctx=ast.Store())
        ],
                   value=_const(0.0)),
        ast.For(target=_store("__l"),
                iter=ast.Call(func=_name("range"), args=[_const_or_name(k)], keywords=[]),
                body=[
                    ast.AugAssign(target=ast.Subscript(value=_name(target.id),
                                                       slice=ast.Tuple(elts=[_name("__i"), _name("__j")],
                                                                       ctx=ast.Load()),
                                                       ctx=ast.Store()),
                                  op=ast.Add(),
                                  value=ast.BinOp(left=ast.Subscript(value=_name(a_name),
                                                                     slice=ast.Tuple(elts=[_name("__i"),
                                                                                           _name("__l")],
                                                                                     ctx=ast.Load()),
                                                                     ctx=ast.Load()),
                                                  op=ast.Mult(),
                                                  right=ast.Subscript(value=_name(b_name),
                                                                      slice=ast.Tuple(elts=[_name("__l"),
                                                                                            _name("__j")],
                                                                                      ctx=ast.Load()),
                                                                      ctx=ast.Load())))
                ],
                orelse=[]),
    ]
    j_loop = ast.For(target=_store("__j"),
                     iter=ast.Call(func=_name("range"), args=[_const_or_name(n)], keywords=[]),
                     body=body,
                     orelse=[])
    i_loop = ast.For(target=_store("__i"),
                     iter=ast.Call(func=_name("range"), args=[_const_or_name(m)], keywords=[]),
                     body=[j_loop],
                     orelse=[])
    return [i_loop]


def _resolve_shape(arr_node: ast.expr, shape_table: Dict[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    if not isinstance(arr_node, ast.Name):
        raise NotImplementedError("reduction operand is not a bare Name")
    shape = shape_table.get(arr_node.id)
    if shape is None:
        raise NotImplementedError(f"shape of {arr_node.id!r} not in IR's shape table")
    return shape


# Registry


def expand_dot(target: ast.expr, args: List[ast.expr], shape_table: Dict[str, Tuple[str, ...]]) -> List[ast.stmt]:
    """``s = np.dot(a, b)`` -> accumulator loop.

    Each operand may be a bare Name (whose full shape drives the
    iteration) OR an array slice such as ``A[i, :j]`` (whose slice
    extent drives the iteration). The structural rule: derive the
    iteration extent from the first operand, then scalarize both
    operands at each iter index via :func:`_scalarize_at_iters`.
    """
    if len(args) != 2:
        raise NotImplementedError("np.dot needs 2 args")
    a, b = args
    extent = _iter_extent_of(a, shape_table)
    if extent is None:
        raise NotImplementedError("np.dot: cannot derive iteration extent")
    if len(extent) != 1:
        raise NotImplementedError("expand_dot expects 1-D iteration")
    iter_name = "__r0"
    iters = [_name(iter_name)]
    sa = _scalarize_at_iters(a, iters, shape_table)
    sb = _scalarize_at_iters(b, iters, shape_table)
    body = [
        ast.AugAssign(target=target if isinstance(target, ast.Subscript) else _store(target.id),
                      op=ast.Add(),
                      value=ast.BinOp(left=sa, op=ast.Mult(), right=sb))
    ]
    loop = [
        ast.For(target=_store(iter_name),
                iter=ast.Call(func=_name("range"), args=[extent[0]], keywords=[]),
                body=body,
                orelse=[])
    ]
    return [ast.Assign(targets=[target], value=_const(0.0))] + loop


def _read_fft_norm(args, kwargs) -> str:
    """``norm`` of an ``np.fft.*`` call -- ``'backward'`` (default)/``'forward'``/
    ``'ortho'`` -- from keyword ``norm=`` or positional slot 3. Missing /
    non-literal / ``None`` falls back to ``'backward'`` (unnormalized forward,
    ``1/prod(N)`` on the inverse)."""
    node = _kwarg_or_pos(args, kwargs, 3, "norm")
    if isinstance(node, ast.Constant) and node.value in ("backward", "forward", "ortho"):
        return node.value
    return "backward"


def _read_fft_axes(args, kwargs, rank: int, is_n: bool) -> List[int]:
    """Resolve the transform axes for an ``np.fft.*`` call. ``fft``/``ifft`` take
    a single ``axis`` (default last); ``fftn``/``ifftn`` take an ``axes``
    sequence (default all axes); ``fft2``/``ifft2`` are ``fftn`` over the last
    two axes. Negative axes wrap modulo ``rank``."""

    def _norm(a):
        # An axis outside the rank means the rank we resolved is not the operand's real rank -- vexx
        # reshapes through a ``.ndim``-conditional tuple that never folds, so the spilled operand is
        # recorded rank 1 and ``axes=(0, 1, 2)`` indexed past the iterator list. Declining is the
        # sizer's contract; the IndexError it used to raise escaped as a crash.
        a = int(a)
        pos = a + rank if a < 0 else a
        if not 0 <= pos < rank:
            raise NotImplementedError(f"np.fft.*: axis {a} is outside the operand rank {rank}")
        return pos

    if is_n:
        spec = _kwarg_or_pos(args, kwargs, 2, "axes")
        if isinstance(spec, (ast.Tuple, ast.List)):
            return [_norm(e.value) for e in spec.elts if isinstance(e, ast.Constant)]
        return list(range(rank))  # default: every axis
    spec = _kwarg_or_pos(args, kwargs, 2, "axis")
    if isinstance(spec, ast.Constant):
        return [_norm(spec.value)]
    return [rank - 1]  # default: last axis


def _expand_dftn(target: ast.expr,
                 args: List[ast.expr],
                 shape_table: Dict[str, Tuple[str, ...]],
                 inverse: bool,
                 is_n: bool = True,
                 kwargs=None) -> List[ast.stmt]:
    """``out = np.fft.fft/ifft/fft2/ifft2/fftn/ifftn(x)`` -> a naive DFT.
    Correctness-only (O(prod(N_t)^2) over the transform axes), kept tiny via the
    benchmark's small preset. Over transform-axis set ``T`` (remaining axes
    batched untouched), the forward transform is

        out[o] = sum_{n_t, t in T} x[src] * exp(-2j*pi * sum_{t in T} o_t n_t / N_t)

    ``src`` indexes transform axes by summation iterator ``n_t`` and batch axes
    by output iterator ``o``. The inverse uses ``+2j*pi`` and divides by
    ``prod(N_t)``. Both operands are complex; the phase numerator is float
    (``2.0 * pi * ...``) so ``/ N_t`` is real division, not C truncation."""
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.fft.* needs a bare Name operand")
    src = args[0]
    shape = shape_table.get(src.id)
    if not shape:
        raise NotImplementedError("np.fft.*: source shape unknown")
    rank = len(shape)
    taxes = _read_fft_axes(args, kwargs, rank, is_n)
    # Output index iterators (one per axis); summation iterators only for the
    # transform axes. The source index uses the summation iterator on transform
    # axes and the (fixed) output iterator on batch axes.
    o_iters = [f"__fk{i}" for i in range(rank)]
    n_iters = {t: f"__fn{t}" for t in taxes}
    o_slot = (_name(o_iters[0]) if rank == 1 else ast.Tuple(elts=[_name(o) for o in o_iters], ctx=ast.Load()))
    src_idx = [(_name(n_iters[d]) if d in taxes else _name(o_iters[d])) for d in range(rank)]
    src_slot = (src_idx[0] if rank == 1 else ast.Tuple(elts=src_idx, ctx=ast.Load()))
    out_k = ast.Subscript(value=_name(target.id), slice=o_slot, ctx=ast.Store())
    out_k_load = ast.Subscript(value=_name(target.id), slice=o_slot, ctx=ast.Load())
    # Emit pi as a numeric literal (backend-agnostic): this expander runs after
    # _MathRewriter, so an ``np.pi`` Attribute would reach the emitter unlowered.
    pi = _const(3.141592653589793)
    # total phase = sum_{t in T} (2.0 * pi * o_t * n_t) / N_t
    phase = None
    for t in taxes:
        num = ast.BinOp(left=ast.BinOp(left=ast.BinOp(left=_const(2.0), op=ast.Mult(), right=pi),
                                       op=ast.Mult(),
                                       right=_name(o_iters[t])),
                        op=ast.Mult(),
                        right=_name(n_iters[t]))
        term = ast.BinOp(left=num, op=ast.Div(), right=_const_or_name(shape[t]))
        phase = term if phase is None else ast.BinOp(left=phase, op=ast.Add(), right=term)
    sign = _const(1j) if inverse else _const(-1j)
    # Emit the already-lowered bare ``exp`` (not ``np.exp``): this expander runs
    # after ``_MathRewriter`` (np.exp -> exp), so ``np.exp`` here would reach the
    # emitter unlowered. The emitter routes ``exp`` of a complex operand to ``cexp``.
    twiddle = ast.Call(func=_name("exp"), args=[ast.BinOp(left=sign, op=ast.Mult(), right=phase)], keywords=[])
    src_n = ast.Subscript(value=_name(src.id), slice=src_slot, ctx=ast.Load())
    acc = ast.AugAssign(target=out_k, op=ast.Add(), value=ast.BinOp(left=src_n, op=ast.Mult(), right=twiddle))
    inner = _wrap_for_loops([n_iters[t] for t in taxes], [shape[t] for t in taxes], [acc])
    body: List[ast.stmt] = [ast.Assign(targets=[out_k], value=_const(0j))] + inner
    # numpy ``norm``: 'backward' (default) puts ``1/prod(N)`` on the INVERSE; 'forward' puts it on
    # the FORWARD; 'ortho' puts ``1/sqrt(prod(N))`` on BOTH. Divide when this direction carries it.
    norm = _read_fft_norm(args, kwargs)
    if norm == "ortho" or ((norm == "forward") != inverse):
        denom = None
        for t in taxes:
            ext = _const_or_name(shape[t])
            denom = ext if denom is None else ast.BinOp(left=denom, op=ast.Mult(), right=ext)
        if norm == "ortho":
            # ``prod(N_t)`` is an integer extent product; ``sqrt`` of an integer is
            # rejected by gfortran (must be REAL or COMPLEX). C/C++ promote silently;
            # coerce portably via ``* 1.0`` (Fortran emits the float literal at
            # double precision, so nothing is lost).
            denom = ast.BinOp(left=denom, op=ast.Mult(), right=_const(1.0))
            denom = ast.Call(func=_name("sqrt"), args=[denom], keywords=[])
        body.append(ast.Assign(targets=[out_k], value=ast.BinOp(left=out_k_load, op=ast.Div(), right=denom)))
    return _wrap_for_loops(o_iters, list(shape), body)


def expand_fftn(target, args, shape_table, kwargs=None):
    return _expand_dftn(target, args, shape_table, inverse=False, is_n=True, kwargs=kwargs)


def expand_ifftn(target, args, shape_table, kwargs=None):
    return _expand_dftn(target, args, shape_table, inverse=True, is_n=True, kwargs=kwargs)


def expand_fft(target, args, shape_table, kwargs=None):
    # 1-D DFT along a single ``axis`` (default last); for a 1-D input == fftn.
    return _expand_dftn(target, args, shape_table, inverse=False, is_n=False, kwargs=kwargs)


def expand_ifft(target, args, shape_table, kwargs=None):
    return _expand_dftn(target, args, shape_table, inverse=True, is_n=False, kwargs=kwargs)


def expand_fftfreq(target, args, shape_table, kwargs=None) -> List[ast.stmt]:
    """``np.fft.fftfreq(n, d=1.0)`` -> DFT sample frequencies (length ``n``):
    ``out[i] = (i if i <= (n - 1) // 2 else i - n) / (n * d)``. Indices up to
    ``(n - 1) // 2`` are non-negative frequencies, the rest wrap negative,
    scaled by sample spacing ``d``. ``n * d`` is real, so ``/`` is real division.
    Matches ``numpy.fft.fftfreq`` for even and odd ``n``."""
    if not args:
        raise NotImplementedError("np.fft.fftfreq needs the sample count n")
    n = args[0]
    d_node = _kwarg_or_pos(args, kwargs, 1, "d")
    if d_node is None:
        d_node = _const(1.0)
    it = "__ff"
    half = ast.BinOp(left=ast.BinOp(left=copy.deepcopy(n), op=ast.Sub(), right=_const(1)),
                     op=ast.FloorDiv(),
                     right=_const(2))
    numer = ast.IfExp(test=ast.Compare(left=_name(it), ops=[ast.LtE()], comparators=[half]),
                      body=_name(it),
                      orelse=ast.BinOp(left=_name(it), op=ast.Sub(), right=copy.deepcopy(n)))
    denom = ast.BinOp(left=copy.deepcopy(n), op=ast.Mult(), right=copy.deepcopy(d_node))
    body = [
        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=_name(it), ctx=ast.Store())],
                   value=ast.BinOp(left=numer, op=ast.Div(), right=denom))
    ]
    return _wrap_for_loops([it], [copy.deepcopy(n)], body)


def _alloc_marker(name: str) -> ast.Assign:
    """``<name> = __hpcagent_bench_zeros__()`` -- the allocation-SITE marker. A local
    whose extent depends on a body-computed scalar can't be malloc'd at fn-top
    (the scalar is garbage there), so the emitter declares it NULL and defers the
    malloc to this marker, placed after the scalar's assignment. An expander that
    consumes the original Assign must re-emit the marker or the buffer stays
    NULL; for a fn-top-malloc'd static local the marker is a no-op, so emitting
    it unconditionally is always safe."""
    return ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                      value=ast.Call(func=ast.Name(id="__hpcagent_bench_zeros__", ctx=ast.Load()), args=[],
                                     keywords=[]))


def expand_copy(target: ast.expr,
                args: List[ast.expr],
                shape_table: Dict[str, Tuple[str, ...]],
                local_dtypes: Optional[Dict[str, str]] = None) -> List[ast.stmt]:
    """``out = np.copy(a)`` -> elementwise copy loop into the LHS. Source may be
    a bare Name (``np.copy(a)``) or an array-valued Subscript (``grid[0].copy()``
    -- a row lowered into a fresh local); for the Subscript case the iteration
    extent comes from the *result* shape (un-indexed axes) via
    ``_iter_extent_of``, scalarized per element like the elementwise expanders.
    """
    if not args:
        raise NotImplementedError("np.copy needs an operand")
    src = args[0]
    shape = _iter_extent_of(src, shape_table)
    if not shape:
        raise NotImplementedError("np.copy: source shape unknown")
    # Register + allocate the fresh target, mirroring the matmul/linalg expanders.
    # Without the ``__hpcagent_bench_zeros__`` marker, a copy target with a RUNTIME
    # (symbolic) shape -- eigh Jacobi's ``Cm = np.ascontiguousarray(a)`` or
    # ``__eigh<k>_a = np.ascontiguousarray(Linv @ h_sub @ Linv.T)`` -- is declared
    # a null pointer and never malloc'd, so the copy loop writes through NULL.
    # The emitter treats a static-shape target's marker as a no-op stack
    # declaration, so emitting it unconditionally is safe for both.
    shape_table.setdefault(target.id, tuple(ast.unparse(e) for e in shape))
    if local_dtypes is not None:
        src_name = None
        if isinstance(src, ast.Name):
            src_name = src.id
        elif isinstance(src, ast.Subscript):
            base = src.value
            while isinstance(base, ast.Subscript):
                base = base.value
            if isinstance(base, ast.Name):
                src_name = base.id
        if src_name:
            src_dt = local_dtypes.get(src_name)
            if src_dt:
                local_dtypes[target.id] = src_dt
    iters = [f"__r{i}" for i in range(len(shape))]
    iter_nodes = [_name(i) for i in iters]
    idx = (iter_nodes[0] if len(iters) == 1 else ast.Tuple(elts=iter_nodes, ctx=ast.Load()))
    sub_src = _scalarize_at_iters(src, iter_nodes, shape_table)
    sub_dst = ast.Subscript(value=_name(target.id), slice=idx, ctx=ast.Store())
    body = [ast.Assign(targets=[sub_dst], value=sub_src)]
    return [_alloc_marker(target.id)] + _wrap_for_loops(iters, shape, body)


def expand_bincount(target: ast.expr,
                    args: List[ast.expr],
                    shape_table: Dict[str, Tuple[str, ...]],
                    kwargs=None) -> List[ast.stmt]:
    """``out = np.bincount(idx, weights=w, minlength=M)`` -> zero ``M`` slots, then scatter-add.

    numpy sizes the result ``max(minlength, idx.max() + 1)``, and the second term is data. ``M`` is
    the only sound choice here: every corpus caller assigns the result into a buffer of exactly that
    extent, so an index at or past ``M`` would make numpy return a LONGER array and the assignment
    itself would raise. Without ``minlength`` the extent is genuinely undecidable, so decline.

    Duplicate indices ACCUMULATE (this is the histogram, not a fancy store), which is why the body
    is ``+=`` and not a last-write-wins store.
    """
    if not args:
        raise NotImplementedError("np.bincount needs an index operand")
    idx = args[0]
    weights = _kwarg_or_pos(args, kwargs, 1, "weights")
    minlength = _kwarg_or_pos(args, kwargs, 2, "minlength")
    if minlength is None:
        raise NotImplementedError("np.bincount without minlength has a data-dependent extent")
    ext = _iter_extent_of(idx, shape_table)
    if (not ext or len(ext) != 1) and weights is not None:
        # spmv builds its index as ``np.repeat(np.arange(M), np.diff(A_indptr))`` -- a data-dependent
        # extent the sizer cannot resolve. numpy REQUIRES weights and index to be the same length, so
        # the weights' extent is that length, and it resolves (it is the declared nnz buffer).
        ext = _iter_extent_of(weights, shape_table)
    if not ext or len(ext) != 1:
        raise NotImplementedError("np.bincount needs a rank-1 index operand of known extent")
    out_len = ast.unparse(minlength)
    shape_table.setdefault(target.id, (out_len, ))
    zero_it = "__bcz0"
    zero_body = [
        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=_name(zero_it), ctx=ast.Store())],
                   value=_const(0.0))
    ]
    acc_it = "__bc0"
    idx_k = _scalarize_at_iters(idx, [_name(acc_it)], shape_table)
    val_k = (_const(1.0) if weights is None else _scalarize_at_iters(weights, [_name(acc_it)], shape_table))
    acc_body = [
        ast.AugAssign(target=ast.Subscript(value=_name(target.id), slice=idx_k, ctx=ast.Store()),
                      op=ast.Add(),
                      value=val_k)
    ]
    return ([_alloc_marker(target.id)] + _wrap_for_loops([zero_it], [_const_or_name(out_len)], zero_body) +
            _wrap_for_loops([acc_it], list(ext), acc_body))


def expand_outer(target: ast.expr,
                 args: List[ast.expr],
                 shape_table: Dict[str, Tuple[str, ...]],
                 op=None) -> List[ast.stmt]:
    """``out = np.outer(a, b)`` -> ``out[i, j] = a[i] * b[j]``. ``op`` defaults to
    ``Mult()``; pass ``ast.Add()`` for ``np.add.outer`` (sum-outer-product).
    """
    if op is None:
        op = ast.Mult()
    if len(args) != 2:
        raise NotImplementedError("np.outer needs 2 args")
    a, b = args
    a_ext = _iter_extent_of(a, shape_table)
    b_ext = _iter_extent_of(b, shape_table)
    if a_ext is None or b_ext is None or len(a_ext) != 1 or len(b_ext) != 1:
        raise NotImplementedError("only 1-D np.outer supported")
    iter_a, iter_b = _name("__i"), _name("__j")
    sa = _scalarize_at_iters(a, [iter_a], shape_table)
    sb = _scalarize_at_iters(b, [iter_b], shape_table)
    body = [
        ast.Assign(targets=[
            ast.Subscript(value=_name(target.id),
                          slice=ast.Tuple(elts=[iter_a, iter_b], ctx=ast.Load()),
                          ctx=ast.Store())
        ],
                   value=ast.BinOp(left=sa, op=op, right=sb))
    ]
    bounds = (a_ext[0], b_ext[0])
    return _wrap_for_loops(["__i", "__j"], bounds, body)


def expand_add_outer(target, args, shape_table):
    return expand_outer(target, args, shape_table, op=ast.Add())


def _expand_elementwise(target, args, shape_table, op_fn):
    """``out = op(a, b)`` -> per-element loop nest.

    Iteration extent comes from the first array-valued operand
    (Name or Subscript-with-Slice). Operands are scalarized via
    ``_scalarize_at_iters`` so slice arguments
    (``np.maximum(A[i, :], B[i, :])``) lower as cleanly as bare
    Names. Scalars on either operand broadcast -- ``np.maximum(0, x)``
    or ``np.minimum(x, 0)`` are equally fine.
    """
    if len(args) != 2:
        raise NotImplementedError("elementwise needs 2 args")
    a, b = args
    # The iteration extent is the numpy BROADCAST of both operands, not just the
    # first: ``np.maximum(a(M,), B(N, M))`` must iterate the full (N, M) output,
    # so fold both extents through ``_broadcast_extents``. ``_scalarize_at_iters``
    # then indexes each operand against the full iter nest, reading a size-1/
    # missing leading axis with a constant 0.
    ea = _iter_extent_of(a, shape_table)
    eb = _iter_extent_of(b, shape_table)
    if ea is None and eb is None:
        raise NotImplementedError("elementwise: extent unknown for both args")
    if ea is None:
        extent = eb
    elif eb is None:
        extent = ea
    else:
        extent = _broadcast_extents(ea, eb)
    iters = [_name(f"__r{i}") for i in range(len(extent))]

    # Constants / scalar Names broadcast; arrays scalarize.
    def maybe_scalar(node):
        if isinstance(node, ast.Constant):
            return node
        if isinstance(node, ast.Name) and not shape_table.get(node.id):
            return node
        return _scalarize_at_iters(node, iters, shape_table)

    sa = maybe_scalar(a)
    sb = maybe_scalar(b)
    idx = iters[0] if len(iters) == 1 else ast.Tuple(elts=list(iters), ctx=ast.Load())
    body = [
        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=idx, ctx=ast.Store())], value=op_fn(sa, sb))
    ]
    out = body
    for var, bound in zip(reversed([i.id for i in iters]), reversed(extent)):
        out = [
            ast.For(target=_store(var),
                    iter=ast.Call(func=_name("range"), args=[bound], keywords=[]),
                    body=out,
                    orelse=[])
        ]
    return out


def expand_minimum(t, a, s):
    return _expand_elementwise(t, a, s, lambda x, y: ast.Call(func=_name("min"), args=[x, y], keywords=[]))


def expand_maximum(t, a, s):
    return _expand_elementwise(t, a, s, lambda x, y: ast.Call(func=_name("max"), args=[x, y], keywords=[]))


def expand_add(t, a, s):
    return _expand_elementwise(t, a, s, lambda x, y: ast.BinOp(left=x, op=ast.Add(), right=y))


def expand_multiply(t, a, s):
    return _expand_elementwise(t, a, s, lambda x, y: ast.BinOp(left=x, op=ast.Mult(), right=y))


def expand_power(t: ast.expr, a: List[ast.expr], s: Dict[str, Tuple[str, ...]]) -> List[ast.stmt]:
    """``out = np.power(a, b)`` -> per-element ``a[i] ** b[i]``.

    A ``**`` BinOp, NOT a bare ``pow(...)`` call: each backend's ``**`` routing already
    dispatches on the operand type (C picks ``__npb_int_pow`` over libm's double ``pow``,
    Fortran's ``**`` is exact on integers). A ``pow`` Name call bypassed that and rounded
    every int64 result above 2**53 through a double."""
    return _expand_elementwise(t, a, s, lambda x, y: ast.BinOp(left=x, op=ast.Pow(), right=y))


def expand_subtract(t, a, s):
    return _expand_elementwise(t, a, s, lambda x, y: ast.BinOp(left=x, op=ast.Sub(), right=y))


def expand_divide(target: ast.expr,
                  args: List[ast.expr],
                  shape_table: Dict[str, Tuple[str, ...]],
                  local_dtypes: Optional[Dict[str, str]] = None) -> List[ast.stmt]:
    """``out = np.divide(a, b)`` -> per-element TRUE division.

    numpy ``divide`` / ``true_divide`` is always true division (int64 / int64 -> float64),
    but this expander runs at libnode-expand -- AFTER lowering's ``_TrueDivisionPromoter``
    phase, which never sees the ``Div`` synthesized here. So apply the promoter's own cast
    directly: an all-integer pair gets its left operand wrapped in ``np.float64(...)``,
    which both native emitters render as a floating divide. Anything not PROVABLY integer
    is left alone (an unwarranted cast would force fp64 into an fp32 kernel)."""
    int_div = _all_integer_operands(args, local_dtypes)

    def op_fn(x: ast.expr, y: ast.expr) -> ast.expr:
        return ast.BinOp(left=_as_float64(x) if int_div else x, op=ast.Div(), right=y)

    return _expand_elementwise(target, args, shape_table, op_fn)


def _cmp(op):
    """Return an op_fn that builds ``ast.Compare(left=x, ops=[op], comparators=[y])``."""
    return lambda x, y: ast.Compare(left=x, ops=[op()], comparators=[y])


def expand_less(t, a, s):
    return _expand_elementwise(t, a, s, _cmp(ast.Lt))


def expand_less_equal(t, a, s):
    return _expand_elementwise(t, a, s, _cmp(ast.LtE))


def expand_greater(t, a, s):
    return _expand_elementwise(t, a, s, _cmp(ast.Gt))


def expand_greater_equal(t, a, s):
    return _expand_elementwise(t, a, s, _cmp(ast.GtE))


def expand_equal(t, a, s):
    return _expand_elementwise(t, a, s, _cmp(ast.Eq))


def expand_not_equal(t, a, s):
    return _expand_elementwise(t, a, s, _cmp(ast.NotEq))


def expand_logical_and(t, a, s):
    return _expand_elementwise(t, a, s, lambda x, y: ast.BoolOp(op=ast.And(), values=[x, y]))


def expand_logical_or(t, a, s):
    return _expand_elementwise(t, a, s, lambda x, y: ast.BoolOp(op=ast.Or(), values=[x, y]))


def expand_logical_not(target, args, shape_table):
    """``out = np.logical_not(a)`` -> per-element ``out[i] = not a[i]``."""
    return _unary_elementwise(target, args, shape_table, lambda x: ast.UnaryOp(op=ast.Not(), operand=x))


def expand_negative(t, a, s):
    """``out = np.negative(a)`` -> ``out[i] = -a[i]``."""
    if not args_one_name(a):
        raise NotImplementedError("np.negative needs a Name arg")
    return _unary_elementwise(t, a, s, lambda x: ast.UnaryOp(op=ast.USub(), operand=x))


def expand_tanh(t, a, s):
    return _unary_elementwise(t, a, s, lambda x: ast.Call(func=_name("tanh"), args=[x], keywords=[]))


def expand_sin_arr(t, a, s):
    return _unary_elementwise(t, a, s, lambda x: ast.Call(func=_name("sin"), args=[x], keywords=[]))


def expand_cos_arr(t, a, s):
    return _unary_elementwise(t, a, s, lambda x: ast.Call(func=_name("cos"), args=[x], keywords=[]))


def expand_exp_arr(t, a, s):
    return _unary_elementwise(t, a, s, lambda x: ast.Call(func=_name("exp"), args=[x], keywords=[]))


def expand_log_arr(t, a, s):
    return _unary_elementwise(t, a, s, lambda x: ast.Call(func=_name("log"), args=[x], keywords=[]))


def expand_sqrt_arr(t, a, s):
    return _unary_elementwise(t, a, s, lambda x: ast.Call(func=_name("sqrt"), args=[x], keywords=[]))


def args_one_name(args):
    return args and isinstance(args[0], ast.Name)


def _unary_elementwise(target, args, shape_table, op_fn):
    """Common scaffold for ``out = np.<unary>(expr)`` -> per-element op.

    Accepts any array-valued expression: bare Name, slice subscript,
    or BinOp / UnaryOp / Call whose iteration extent is derivable.
    """
    if not args:
        raise NotImplementedError("unary elementwise needs an arg")
    a = args[0]
    extent = _iter_extent_of(a, shape_table)
    if extent is None:
        raise NotImplementedError("unary elementwise: extent unknown")
    iters = [_name(f"__r{i}") for i in range(len(extent))]
    sa = _scalarize_at_iters(a, iters, shape_table)
    idx = (iters[0] if len(iters) == 1 else ast.Tuple(elts=list(iters), ctx=ast.Load()))
    body = [ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=idx, ctx=ast.Store())], value=op_fn(sa))]
    out = body
    for var, bound in zip(reversed([i.id for i in iters]), reversed(extent)):
        out = [
            ast.For(target=_store(var),
                    iter=ast.Call(func=_name("range"), args=[bound], keywords=[]),
                    body=out,
                    orelse=[])
        ]
    return out


def expand_clip(target: ast.expr, args: List[ast.expr], shape_table: Dict[str, Tuple[str, ...]]) -> List[ast.stmt]:
    """``out = np.clip(a, lo, hi)`` -> ``out[i] = min(hi, max(lo, a[i]))``. numpy
    defines clip as ``minimum(a_max, maximum(a, a_min))``, so a degenerate
    ``lo > hi`` resolves to hi -- matched by the outer ``min`` (the reversed
    order would return ``lo`` instead).
    """
    if len(args) != 3 or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.clip needs Name + 2 scalar args")
    a = args[0]
    shape = shape_table.get(a.id)
    if not shape:
        raise NotImplementedError("np.clip: shape unknown")
    iters = [f"__r{i}" for i in range(len(shape))]
    idx = (_name(iters[0]) if len(iters) == 1 else ast.Tuple(elts=[_name(i) for i in iters], ctx=ast.Load()))
    a_sub = ast.Subscript(value=_name(a.id), slice=idx, ctx=ast.Load())
    clamped = ast.Call(func=_name("min"),
                       args=[args[2], ast.Call(func=_name("max"), args=[args[1], a_sub], keywords=[])],
                       keywords=[])
    body = [ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=idx, ctx=ast.Store())], value=clamped)]
    return _wrap_for_loops(iters, shape, body)


def expand_where(target: ast.expr, args: List[ast.expr], shape_table: Dict[str, Tuple[str, ...]]) -> List[ast.stmt]:
    """``out = np.where(cond, a, b)`` -> elementwise ternary. ``cond`` may be a
    bare mask Name or a whole-array comparison (lulesh's BC selection ``sel ==
    XI_M_SYMM``); ``a``/``b`` may be a Name or any whole-array expression (incl.
    a fancy gather ``delv[ielem]``). Every operand is scalarised recursively.
    """
    if len(args) != 3:
        raise NotImplementedError("np.where needs cond + 2 args")
    # Result shape: the hoister registered the target temp's shape; fall back to
    # the mask Name's own shape when ``where`` is assigned directly.
    shape = shape_table.get(target.id) if isinstance(target, ast.Name) else None
    if not shape and isinstance(args[0], ast.Name):
        shape = shape_table.get(args[0].id)
    if not shape:
        raise NotImplementedError("np.where: shape unknown")
    iters = [f"__r{i}" for i in range(len(shape))]
    idx = (_name(iters[0]) if len(iters) == 1 else ast.Tuple(elts=[_name(i) for i in iters], ctx=ast.Load()))
    iter_nodes = [_name(i) for i in iters]

    def maybe_sub(arg):
        return _scalarize_at_iters(arg, iter_nodes, shape_table)

    ternary = ast.IfExp(test=maybe_sub(args[0]), body=maybe_sub(args[1]), orelse=maybe_sub(args[2]))
    body = [ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=idx, ctx=ast.Store())], value=ternary)]
    return _wrap_for_loops(iters, shape, body)


def _kwarg_or_pos(args: List[ast.expr], kwargs, pos: int, name: str):
    """Resolve a numpy arg passed positionally OR by keyword: ``args[pos]`` if
    present, else the ``name=`` keyword value from ``kwargs``, else ``None``.
    Lets an expander accept both ``np.transpose(A, (1,0,2))`` and
    ``np.transpose(A, axes=(1,0,2))``. :func:`_call_expander` forwards
    ``node.keywords`` only to expanders declaring a ``kwargs`` parameter, so
    reading keywords requires that parameter.
    """
    if len(args) > pos:
        return args[pos]
    for kw in (kwargs or []):
        if kw.arg == name:
            return kw.value
    return None


def expand_transpose(target: ast.expr,
                     args: List[ast.expr],
                     shape_table: Dict[str, Tuple[str, ...]],
                     kwargs=None) -> List[ast.stmt]:
    """``out = np.transpose(A[, axes])`` -> nested per-element copy. Supports any
    rank N >= 1 with an explicit perm (Tuple/List of int constants, positional or
    ``axes=``); without one, defaults to reversing the axes. Output is a fresh
    array (already declared by the pre-pass harvester), copied element-by-element
    through the permuted index map.
    """
    if not args_one_name(args):
        raise NotImplementedError("np.transpose needs Name first arg")
    a = args[0]
    if isinstance(target, ast.Name) and target.id == a.id:
        # ``tap = np.moveaxis(tap, -1, 1)`` -- the permutation writes back into the buffer it is
        # reading. Element by element that overwrites source cells before they are read, so the
        # result is neither the input nor the permutation. Refuse rather than emit it: a wrong
        # answer that compiles is worse than a kernel that does not.
        raise NotImplementedError(f"np.transpose permutes {a.id!r} into itself; give the result its own name")
    shape = shape_table.get(a.id)
    if not shape:
        raise NotImplementedError("np.transpose: source shape unknown")
    n_dim = len(shape)
    perm_arg = _kwarg_or_pos(args, kwargs, 1, "axes")
    if perm_arg is not None:
        if not isinstance(perm_arg, (ast.Tuple, ast.List)):
            raise NotImplementedError("np.transpose: perm must be Tuple/List")
        perm = [e.value for e in perm_arg.elts if isinstance(e, ast.Constant) and isinstance(e.value, int)]
        if len(perm) != n_dim:
            raise NotImplementedError("np.transpose: perm size != ndim")
    else:
        perm = list(reversed(range(n_dim)))
    src_iters = [f"__t{i}" for i in range(n_dim)]
    # Output index in source axis order: out[src_iters[perm[0]], ..., src_iters[perm[-1]]]
    # i.e. axis ``i`` of out comes from source axis perm[i].
    out_slot_elts = [_name(src_iters[p]) for p in perm]
    src_slot_elts = [_name(v) for v in src_iters]
    if n_dim == 1:
        out_slot = out_slot_elts[0]
        src_slot = src_slot_elts[0]
    else:
        out_slot = ast.Tuple(elts=out_slot_elts, ctx=ast.Load())
        src_slot = ast.Tuple(elts=src_slot_elts, ctx=ast.Load())
    body = [
        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=out_slot, ctx=ast.Store())],
                   value=ast.Subscript(value=_name(a.id), slice=src_slot, ctx=ast.Load()))
    ]
    return _wrap_for_loops(src_iters, shape, body)


def _const_axis(node: Optional[ast.expr], rank: int) -> Optional[int]:
    """A (possibly negative) constant-int axis normalized to ``[0, rank)``; ``None`` when
    ``node`` is not a plain int constant or the axis is out of range."""
    val = _const_int(node)
    if val is None:
        return None
    ax = val + rank if val < 0 else val
    return ax if 0 <= ax < rank else None


def expand_swapaxes(target: ast.expr,
                    args: List[ast.expr],
                    shape_table: Dict[str, Tuple[str, ...]],
                    kwargs=None) -> List[ast.stmt]:
    """``out = np.swapaxes(a, i, j)`` -> ``np.transpose(a, perm)`` with ``perm`` the
    identity permutation with axes ``i`` and ``j`` exchanged (constant int axes). Reuses
    the transpose loop-lowering, so no new machinery -- the ML attention Q/K axis swap."""
    if not (args_one_name(args) and len(args) >= 3):
        raise NotImplementedError("np.swapaxes needs (Name, int, int)")
    a = args[0]
    shape = shape_table.get(a.id)
    if not shape:
        raise NotImplementedError("np.swapaxes: source shape unknown")
    rank = len(shape)
    i, j = _const_axis(args[1], rank), _const_axis(args[2], rank)
    if i is None or j is None:
        raise NotImplementedError("np.swapaxes: axes must be constant ints in range")
    perm = list(range(rank))
    perm[i], perm[j] = perm[j], perm[i]
    perm_tuple = ast.Tuple(elts=[_const(p) for p in perm], ctx=ast.Load())
    return expand_transpose(target, [a, perm_tuple], shape_table)


def expand_moveaxis(target: ast.expr,
                    args: List[ast.expr],
                    shape_table: Dict[str, Tuple[str, ...]],
                    kwargs=None) -> List[ast.stmt]:
    """``out = np.moveaxis(a, source, destination)`` -> ``np.transpose(a, perm)`` with
    ``perm`` numpy's own algorithm (drop axis ``source``, reinsert it at ``destination``
    among the rest). Reuses the transpose loop-lowering like ``expand_swapaxes``.
    Scalar ``source``/``destination`` only (constant ints); the tuple form of the numpy
    API is rare enough in this corpus to decline rather than support here."""
    if not args_one_name(args):
        raise NotImplementedError("np.moveaxis needs a Name first arg")
    a = args[0]
    shape = shape_table.get(a.id)
    if not shape:
        raise NotImplementedError("np.moveaxis: source shape unknown")
    rank = len(shape)
    src = _const_axis(_kwarg_or_pos(args, kwargs, 1, "source"), rank)
    dst = _const_axis(_kwarg_or_pos(args, kwargs, 2, "destination"), rank)
    if src is None or dst is None:
        raise NotImplementedError("np.moveaxis: source/destination must be constant ints in range")
    perm = [n for n in range(rank) if n != src]
    perm.insert(dst, src)
    perm_tuple = ast.Tuple(elts=[_const(p) for p in perm], ctx=ast.Load())
    return expand_transpose(target, [a, perm_tuple], shape_table)


def expand_expand_dims(target: ast.expr,
                       args: List[ast.expr],
                       shape_table: Dict[str, Tuple[str, ...]],
                       kwargs=None) -> List[ast.stmt]:
    """``out = np.expand_dims(a, axis)`` -> ``np.reshape(a, <a's shape with a size-1 axis
    inserted at axis>)`` -- a metadata view, lowered as the reshape flat-copy."""
    if not args_one_name(args):
        raise NotImplementedError("np.expand_dims needs a Name first arg")
    a = args[0]
    shape = shape_table.get(a.id)
    if not shape:
        raise NotImplementedError("np.expand_dims: source shape unknown")
    axis = _const_axis(_kwarg_or_pos(args, kwargs, 1, "axis"), len(shape) + 1)
    if axis is None:
        raise NotImplementedError("np.expand_dims: axis must be a constant int in range")
    new = list(shape)
    new.insert(axis, "1")
    newshape = ast.Tuple(elts=[_const_or_name(str(t)) for t in new], ctx=ast.Load())
    return expand_reshape(target, [a, newshape], shape_table)


def expand_squeeze(target: ast.expr,
                   args: List[ast.expr],
                   shape_table: Dict[str, Tuple[str, ...]],
                   kwargs=None) -> List[ast.stmt]:
    """``out = np.squeeze(a[, axis])`` -> ``np.reshape(a, <a's shape with the size-1
    axis / all size-1 axes dropped>)``. Without ``axis`` every unit dim is dropped; with
    ``axis`` that one axis (which must be size-1) is dropped."""
    if not args_one_name(args):
        raise NotImplementedError("np.squeeze needs a Name first arg")
    a = args[0]
    shape = shape_table.get(a.id)
    if not shape:
        raise NotImplementedError("np.squeeze: source shape unknown")
    axis_node = _kwarg_or_pos(args, kwargs, 1, "axis")
    if axis_node is not None:
        axis = _const_axis(axis_node, len(shape))
        if axis is None or str(shape[axis]) != "1":
            raise NotImplementedError("np.squeeze: axis must be a constant size-1 dim")
        new = [t for k, t in enumerate(shape) if k != axis]
    else:
        new = [t for t in shape if str(t) != "1"]
    new = new or ["1"]  # a fully-squeezed array is scalar-like -> keep a (1,) buffer
    newshape = ast.Tuple(elts=[_const_or_name(str(t)) for t in new], ctx=ast.Load())
    return expand_reshape(target, [a, newshape], shape_table)


def expand_take(target: ast.expr,
                args: List[ast.expr],
                shape_table: Dict[str, Tuple[str, ...]],
                kwargs=None) -> List[ast.stmt]:
    """``out = np.take(a, idx[, axis=k])`` -> a gather loop nest. With ``axis=k``,
    the k-th axis is indexed by 1-D ``idx`` (out's k-th extent = idx's length),
    other axes copied straight through: ``out[.., t, ..] = a[.., idx[t], ..]``
    (the ML embedding lookup). Without ``axis``, the flat take needs a 1-D
    source: ``out[t] = a[idx[t]]``."""
    if not (args_one_name(args) and len(args) >= 2 and isinstance(args[1], ast.Name)):
        raise NotImplementedError("np.take needs (Name a, Name idx)")
    a, idx = args[0], args[1]
    a_shape, idx_shape = shape_table.get(a.id), shape_table.get(idx.id)
    if not a_shape or not idx_shape:
        raise NotImplementedError("np.take: source / index shape unknown")
    if len(idx_shape) != 1:
        raise NotImplementedError("np.take: index must be 1-D")
    axis_node = _kwarg_or_pos(args, kwargs, 2, "axis")
    if axis_node is None:
        if len(a_shape) != 1:
            raise NotImplementedError("np.take without axis needs a 1-D source")
        axis = 0
    else:
        axis = _const_axis(axis_node, len(a_shape))
        if axis is None:
            raise NotImplementedError("np.take: axis must be a constant int in range")
    out_shape = list(a_shape)
    out_shape[axis] = idx_shape[0]  # the gathered axis takes the index length
    iters = [f"__tk{i}" for i in range(len(out_shape))]
    src_index = [_name(v) for v in iters]
    src_index[axis] = ast.Subscript(value=_name(idx.id), slice=_name(iters[axis]), ctx=ast.Load())
    out_slot = _name(iters[0]) if len(iters) == 1 else ast.Tuple(elts=[_name(v) for v in iters], ctx=ast.Load())
    src_slot = src_index[0] if len(src_index) == 1 else ast.Tuple(elts=src_index, ctx=ast.Load())
    body = [
        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=out_slot, ctx=ast.Store())],
                   value=ast.Subscript(value=_name(a.id), slice=src_slot, ctx=ast.Load()))
    ]
    return _wrap_for_loops(iters, out_shape, body)


def expand_linspace(target: ast.expr, args: List[ast.expr], shape_table: Dict[str, Tuple[str, ...]]) -> List[ast.stmt]:
    """``out = np.linspace(start, stop, n)`` -> ``for i in range(n): out[i] =
    (stop - start) / max(n - 1, 1) * i + start``, then ``out[n - 1] = stop`` when ``n > 1``.

    Both details are numpy's arithmetic, not cosmetics. numpy divides FIRST and scales the index by
    that step (``y = arange(num) * step; y += start``); folding the division to the end --
    ``start + span * i / div`` -- is a different rounding, and it was off by an ulp on the very
    first grid this expansion was used for. numpy then PINS the last sample to ``stop``, because
    the computed one need not land on it exactly. An ulp is not a rounding nit wherever the grid
    seeds an iteration: mandelbrot's escape-time map turned 4.4e-16 at the seed into 1.3 in the
    result. ``max(n - 1, 1)`` is numpy's divisor too, so ``np.linspace(start, stop, 1)`` returns
    ``[start]`` rather than dividing by zero -- which is also why the endpoint pin is guarded:
    at ``n == 1`` numpy keeps ``start`` and pinning would overwrite it with ``stop``.
    """
    if len(args) != 3:
        raise NotImplementedError("np.linspace needs (start, stop, n)")
    start, stop, n = args
    span = ast.BinOp(left=copy.deepcopy(stop), op=ast.Sub(), right=copy.deepcopy(start))
    denom = ast.Call(func=_name("max"),
                     args=[ast.BinOp(left=copy.deepcopy(n), op=ast.Sub(), right=_const(1)),
                           _const(1)],
                     keywords=[])
    step = ast.BinOp(left=span, op=ast.Div(), right=denom)
    expr = ast.BinOp(left=ast.BinOp(left=step, op=ast.Mult(), right=_name("__i")),
                     op=ast.Add(),
                     right=copy.deepcopy(start))
    body = [
        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=_name("__i"), ctx=ast.Store())], value=expr)
    ]
    last = ast.Subscript(value=_name(target.id),
                         slice=ast.BinOp(left=copy.deepcopy(n), op=ast.Sub(), right=_const(1)),
                         ctx=ast.Store())
    return [
        ast.For(target=_store("__i"),
                iter=ast.Call(func=_name("range"), args=[copy.deepcopy(n)], keywords=[]),
                body=body,
                orelse=[]),
        ast.If(test=ast.Compare(left=copy.deepcopy(n), ops=[ast.Gt()], comparators=[_const(1)]),
               body=[ast.Assign(targets=[last], value=copy.deepcopy(stop))],
               orelse=[]),
    ]


def arange_count(args: List[ast.expr]) -> ast.expr:
    """Element count of ``np.arange(*args)``: ``ceil(span / step)``, clamped at zero.

    The obvious ``(span + step - 1) // step`` is a POSITIVE-STEP identity. Under a negative step
    it over-counts -- ``np.arange(10, 0, -1)`` came out 12, so the iota loop wrote 12 elements
    into an array sized ``stop - start`` = -10: a negative C VLA (which does not compile) and a
    zero-length Fortran array the loop then ran off the end of, returning garbage that graded
    green. ``-((-span) // step)`` is ceil for either sign, given floor division.

    Constant arguments fold here, so the common literal ``arange`` keeps a plain integer extent
    rather than an expression a Fortran declaration would have to be able to evaluate.

    Shared with the shape inference in :meth:`_extent_to_shape_token` so the extent an array is
    declared with and the trip count the loop runs cannot disagree -- disagreeing is what turned
    a wrong count into an out-of-bounds write."""
    if len(args) == 1:
        span, step = args[0], None
    elif len(args) == 2:
        span, step = ast.BinOp(left=args[1], op=ast.Sub(), right=args[0]), None
    elif len(args) == 3:
        span, step = ast.BinOp(left=args[1], op=ast.Sub(), right=args[0]), args[2]
    else:
        raise NotImplementedError("np.arange needs 1-3 args")
    literals = [_const_int(a) for a in args]  # accepts a negated literal: step=-1 is UnaryOp(USub, 1)
    if all(v is not None for v in literals):
        start, stop = (0, literals[0]) if len(args) == 1 else (literals[0], literals[1])
        istep = literals[2] if len(args) == 3 else 1
        if istep == 0:
            raise NotImplementedError("np.arange step must be nonzero")
        return _const(max(0, -((start - stop) // istep)))
    # Unit step (the 1- and 2-arg forms, or an explicit step of 1): the count IS the span, so keep
    # the plain expression rather than wrapping it in the ceil form.
    if step is None or _const_int(step) == 1:
        return span
    # -((-span) // step): ceil for either sign, since // is emitted as the flooring int_floor.
    return ast.UnaryOp(op=ast.USub(),
                       operand=ast.BinOp(left=ast.UnaryOp(op=ast.USub(), operand=span), op=ast.FloorDiv(), right=step))


def expand_arange(target: ast.expr, args: List[ast.expr], shape_table: Dict[str, Tuple[str, ...]]) -> List[ast.stmt]:
    """``out = np.arange(stop)`` -> ``for i in range(stop): out[i] = i``;
    ``np.arange(start, stop)`` -> ``out[i] = start + i``;
    ``np.arange(start, stop, step)`` -> ``out[i] = start + i*step`` over
    :func:`arange_count` elements. The iota value casts to ``out``'s declared dtype on
    assignment. Mirrors :func:`expand_linspace`."""
    if len(args) == 1:
        start, step = _const(0), None
    elif len(args) == 2:
        start, step = args[0], None
    elif len(args) == 3:
        start, step = args[0], args[2]
    else:
        raise NotImplementedError("np.arange needs 1-3 args")
    count = arange_count(args)
    # value(i) = start + i*step  (step omitted -> +i)
    idx = _name("__i")
    scaled = idx if step is None else ast.BinOp(left=idx, op=ast.Mult(), right=step)
    value = scaled if (len(args) == 1) else ast.BinOp(left=start, op=ast.Add(), right=scaled)
    body = [
        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=_name("__i"), ctx=ast.Store())], value=value)
    ]
    return [
        ast.For(target=_store("__i"),
                iter=ast.Call(func=_name("range"), args=[count], keywords=[]),
                body=body,
                orelse=[])
    ]


class _RenameNames(ast.NodeTransformer):
    """Rename bare ``Name`` ids per a mapping (used to bind a fromfunction
    lambda's parameters to the loop iteration variables)."""

    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.mapping:
            return ast.copy_location(ast.Name(id=self.mapping[node.id], ctx=node.ctx), node)
        return node


def expand_fromfunction(target: ast.expr, args: List[ast.expr], shape_table: Dict[str, Tuple[str,
                                                                                             ...]]) -> List[ast.stmt]:
    """``out = np.fromfunction(lambda i, j: f(i, j), (N, M))`` -> ``for i in
    range(N): for j in range(M): out[i, j] = f(i, j)``. The lambda body is
    inlined per-element with its parameters bound to the loop iters, realising
    the lambda as the loop body rather than a separate callable. Captured free
    variables are left untouched."""
    if len(args) < 2 or not isinstance(args[0], ast.Lambda):
        raise NotImplementedError("np.fromfunction needs (lambda, shape)")
    lam, shape_node = args[0], args[1]
    params = [a.arg for a in lam.args.args]
    shape_elts = (list(shape_node.elts) if isinstance(shape_node, (ast.Tuple, ast.List)) else [shape_node])
    if len(params) != len(shape_elts):
        raise NotImplementedError("np.fromfunction: lambda arity != shape rank")
    iters = [f"__ff{i}" for i in range(len(params))]
    body_expr = _RenameNames(dict(zip(params, iters))).visit(copy.deepcopy(lam.body))
    slot_elts = [_name(v) for v in iters]
    slot = slot_elts[0] if len(iters) == 1 else ast.Tuple(elts=slot_elts, ctx=ast.Load())
    body = [ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=slot, ctx=ast.Store())], value=body_expr)]
    return _wrap_for_loops(iters, shape_elts, body)


#: Synthetic keyword the tuple-unpack in lowering.py attaches to each split
#: ``np.meshgrid`` call, telling this expander which output array it builds.
#: numpy's ``meshgrid`` has no such keyword, so the name is unambiguous.
MESHGRID_AXIS_KW = "__meshgrid_axis__"


def expand_meshgrid(target: ast.expr,
                    args: List[ast.expr],
                    shape_table: Dict[str, Tuple[str, ...]],
                    kwargs=None) -> List[ast.stmt]:
    """Emit ONE broadcast output of ``np.meshgrid(a0, a1, ..., a_{k-1})``. Given
    1-D inputs of lengths ``(N0, ..., N_{k-1})``: ``indexing='ij'`` gives every
    output shape ``(N0, ..., N_{k-1})`` with ``out_d[i0, ..., i_{k-1}] =
    a_d[i_d]``; ``indexing='xy'`` (numpy's default) swaps axes 0 and 1, so the
    output shape is ``(N1, N0, N2, ...)`` and output ``d`` varies with input axis
    ``perm[d]``.

    A meshgrid call returns a TUPLE of arrays; the lowering-side unpack splits it
    into one call per output and marks each with :data:`MESHGRID_AXIS_KW` so this
    expander builds that output's broadcast-copy loop nest. Each input must be a
    1-D array of known length."""
    kwargs = kwargs or []
    indexing = "xy"
    axis: Optional[int] = None
    for kw in kwargs:
        if kw.arg == "indexing" and isinstance(kw.value, ast.Constant):
            indexing = kw.value.value
        elif kw.arg == MESHGRID_AXIS_KW and isinstance(kw.value, ast.Constant):
            axis = kw.value.value
    if indexing not in ("ij", "xy"):
        raise NotImplementedError(f"np.meshgrid indexing={indexing!r} not supported")
    k = len(args)
    if axis is None or not (0 <= axis < k):
        raise NotImplementedError("np.meshgrid: output axis unresolved")
    # Length of each 1-D input array.
    lengths: List[ast.expr] = []
    for a in args:
        ext = _iter_extent_of(a, shape_table)
        if ext is None or len(ext) != 1:
            raise NotImplementedError("np.meshgrid needs 1-D inputs of known length")
        lengths.append(ext[0])
    # perm maps an OUTPUT axis to the INPUT axis whose length it takes; it is a
    # single swap of 0 and 1 for 'xy' (and its own inverse), identity for 'ij'.
    perm = list(range(k))
    if indexing == "xy" and k >= 2:
        perm[0], perm[1] = 1, 0
    out_dims = [lengths[perm[p]] for p in range(k)]
    iters = [f"__mgi{p}" for p in range(k)]
    # This output varies with input ``axis`` -> along output axis ``perm[axis]``
    # (perm is self-inverse, so the read iterator is ``iters[perm[axis]]``).
    read_iter = _name(iters[perm[axis]])
    src = _scalarize_at_iters(copy.deepcopy(args[axis]), [read_iter], shape_table)
    out_slot = (_name(iters[0]) if k == 1 else ast.Tuple(elts=[_name(v) for v in iters], ctx=ast.Load()))
    body = [ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=out_slot, ctx=ast.Store())], value=src)]
    return _wrap_for_loops(iters, [copy.deepcopy(d) for d in out_dims], body)


def expand_eye(target: ast.expr, args: List[ast.expr], shape_table: Dict[str, Tuple[str, ...]]) -> List[ast.stmt]:
    """``out = np.eye(n)`` -> ``out[i, j] = (i == j) ? 1.0 : 0.0``."""
    if not args:
        raise NotImplementedError("np.eye needs at least 1 arg")
    n = args[0]
    body = [
        ast.Assign(targets=[
            ast.Subscript(value=_name(target.id),
                          slice=ast.Tuple(elts=[_name("__i"), _name("__j")], ctx=ast.Load()),
                          ctx=ast.Store())
        ],
                   value=ast.IfExp(test=ast.Compare(left=_name("__i"), ops=[ast.Eq()], comparators=[_name("__j")]),
                                   body=_const(1.0),
                                   orelse=_const(0.0)))
    ]
    return [
        ast.For(target=_store("__i"),
                iter=ast.Call(func=_name("range"), args=[n], keywords=[]),
                body=[
                    ast.For(target=_store("__j"),
                            iter=ast.Call(func=_name("range"), args=[n], keywords=[]),
                            body=body,
                            orelse=[])
                ],
                orelse=[])
    ]


def _expand_triangular(target: ast.expr, args: List[ast.expr], shape_table: Dict[str, Tuple[str, ...]], kwargs,
                       lower: bool) -> List[ast.stmt]:
    """Shared triu/tril lowering: copy ``A[i, j]`` where it's on the kept side of
    the ``i + k`` diagonal, else 0. ``lower=False`` keeps ``j >= i + k`` (upper);
    ``lower=True`` keeps ``j <= i + k``. Optional ``k`` offset (positional or
    ``k=``) defaults to 0."""
    name = "np.tril" if lower else "np.triu"
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError(f"{name} needs Name first arg")
    a = args[0]
    shape = shape_table.get(a.id)
    if not shape or len(shape) != 2:
        raise NotImplementedError(f"{name}: only 2-D supported")
    m, n = shape
    k_arg: ast.expr = _const(0)
    _k = _kwarg_or_pos(args, kwargs, 1, "k")
    if _k is not None:
        k_arg = _k
    a_sub = ast.Subscript(value=_name(a.id),
                          slice=ast.Tuple(elts=[_name("__i"), _name("__j")], ctx=ast.Load()),
                          ctx=ast.Load())
    if isinstance(k_arg, ast.Constant) and k_arg.value == 0:
        threshold: ast.expr = _name("__i")
    else:
        threshold = ast.BinOp(left=_name("__i"), op=ast.Add(), right=k_arg)
    cmp_op = ast.LtE() if lower else ast.GtE()
    body = [
        ast.Assign(targets=[
            ast.Subscript(value=_name(target.id),
                          slice=ast.Tuple(elts=[_name("__i"), _name("__j")], ctx=ast.Load()),
                          ctx=ast.Store())
        ],
                   value=ast.IfExp(test=ast.Compare(left=_name("__j"), ops=[cmp_op], comparators=[threshold]),
                                   body=a_sub,
                                   orelse=_const(0.0)))
    ]
    return _wrap_for_loops(["__i", "__j"], (m, n), body)


def expand_triu(target: ast.expr,
                args: List[ast.expr],
                shape_table: Dict[str, Tuple[str, ...]],
                kwargs=None) -> List[ast.stmt]:
    """``out = np.triu(A [, k])`` -> ``out[i, j] = A[i, j] if j >= i+k else 0``.
    Optional ``k`` offset (default 0) selects the diagonal; ``k=1`` skips the
    main diagonal (strict upper-triangular).
    """
    return _expand_triangular(target, args, shape_table, kwargs, lower=False)


def expand_hstack(target: ast.expr, args: List[ast.expr], shape_table: Dict[str, Tuple[str, ...]]) -> List[ast.stmt]:
    """``out = np.hstack((a, b, c, ...))`` -- horizontal concatenation. 2-D
    operands ``(N, K_i)`` -> ``(N, sum K_i)``, each copied into ``out`` at its
    column offset. 1-D operands ``(K_i,)`` -> ``(sum K_i,)``, flat concat.
    """
    if not args:
        raise NotImplementedError("np.hstack needs at least one arg")
    if len(args) == 1 and isinstance(args[0], ast.Tuple):
        operands = list(args[0].elts)
    else:
        operands = list(args)
    names: List[str] = []
    shapes: List[Tuple[str, ...]] = []
    for op in operands:
        if not isinstance(op, ast.Name):
            raise NotImplementedError("np.hstack: operand must be a Name")
        s = shape_table.get(op.id)
        if s is None:
            raise NotImplementedError(f"np.hstack: shape of {op.id} unknown")
        names.append(op.id)
        shapes.append(tuple(s))
    rank = len(shapes[0])
    if any(len(s) != rank for s in shapes):
        raise NotImplementedError("np.hstack: mixed ranks unsupported")
    if rank not in (1, 2):
        raise NotImplementedError("np.hstack: only rank-1 / rank-2 supported")
    out: List[ast.stmt] = []
    if rank == 1:
        offset_tok = "0"
        for nm, s in zip(names, shapes):
            k_ast = _const_or_name(s[0])
            col_index: ast.expr
            if offset_tok == "0":
                col_index = _name("__hsj")
            else:
                col_index = ast.BinOp(left=_name("__hsj"), op=ast.Add(), right=_const_or_name(offset_tok))
            body = [
                ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=col_index, ctx=ast.Store())],
                           value=ast.Subscript(value=_name(nm), slice=_name("__hsj"), ctx=ast.Load()))
            ]
            out.append(
                ast.For(target=_store("__hsj"),
                        iter=ast.Call(func=_name("range"), args=[k_ast], keywords=[]),
                        body=body,
                        orelse=[]))
            offset_tok = (f"({offset_tok}) + ({s[0]})" if offset_tok != "0" else str(s[0]))
        return out
    # rank == 2
    n_tok = shapes[0][0]
    offset_tok = "0"
    for nm, s in zip(names, shapes):
        k_ast = _const_or_name(s[1])
        col_index: ast.expr
        if offset_tok == "0":
            col_index = _name("__hsj")
        else:
            col_index = ast.BinOp(left=_name("__hsj"), op=ast.Add(), right=_const_or_name(offset_tok))
        body = [
            ast.Assign(targets=[
                ast.Subscript(value=_name(target.id),
                              slice=ast.Tuple(elts=[_name("__hsi"), col_index], ctx=ast.Load()),
                              ctx=ast.Store())
            ],
                       value=ast.Subscript(value=_name(nm),
                                           slice=ast.Tuple(elts=[_name("__hsi"), _name("__hsj")], ctx=ast.Load()),
                                           ctx=ast.Load()))
        ]
        inner = ast.For(target=_store("__hsj"),
                        iter=ast.Call(func=_name("range"), args=[k_ast], keywords=[]),
                        body=body,
                        orelse=[])
        out.append(
            ast.For(target=_store("__hsi"),
                    iter=ast.Call(func=_name("range"), args=[_const_or_name(n_tok)], keywords=[]),
                    body=[inner],
                    orelse=[]))
        offset_tok = (f"({offset_tok}) + ({s[1]})" if offset_tok != "0" else str(s[1]))
    return out


def _concat_operands_axis(args, kwargs, shape_table):
    """Shared parse for ``np.concatenate`` / ``np.stack``-style calls: return
    ``(names, shapes, axis)``. The sequence is the first positional arg (a
    tuple/list of array Names); ``axis`` is a keyword or the 2nd positional
    (default 0, normalised against the operand rank)."""
    kwargs = kwargs or []
    if not args:
        raise NotImplementedError("np.concatenate needs a sequence arg")
    seq = args[0]
    if not isinstance(seq, (ast.Tuple, ast.List)):
        raise NotImplementedError("np.concatenate: sequence must be a tuple/list")
    # ``axis`` may be a plain or negated literal (``axis=-1`` parses as
    # ``UnaryOp(USub, Constant(1))``, not ``Constant(-1)``); ``_const_int``
    # accepts both, and the ``axis < 0`` fixup below resolves it mod rank.
    # ``axis=None`` is not "no axis": numpy FLATTENS every operand and concatenates the results,
    # which is a different output rank. Refuse rather than fall through to the axis-0 default.
    axis_node = _kwarg_or_pos(args, kwargs, 1, "axis")
    if isinstance(axis_node, ast.Constant) and axis_node.value is None:
        raise NotImplementedError("np.concatenate(axis=None) flattens every operand first; not lowered")
    axis = _axis_literal_or_refuse(axis_node, "np.concatenate", 0)
    names: List[Optional[str]] = []
    shapes: List[Tuple[str, ...]] = []
    for op in seq.elts:
        if isinstance(op, ast.Name):
            s = shape_table.get(op.id)
            if s is None:
                raise NotImplementedError(f"np.concatenate: shape of {op.id} unknown")
            names.append(op.id)
            shapes.append(tuple(s))
            continue
        # A non-Name operand -- dwt2d's rotate ``np.concatenate((e[:, 1:], e[:,
        # 0:1]), axis=1)`` -- still has a resolvable EXTENT, all the shape rule
        # needs. The expander materialises such operands into Names before
        # calling this, so ``names`` stays complete; reject a leftover None
        # rather than emit a nameless read.
        ext = _iter_extent_of(op, shape_table)
        if ext is None:
            raise NotImplementedError("np.concatenate: operand must be a Name")
        names.append(None)
        shapes.append(tuple(ast.unparse(e) for e in ext))
    rank = len(shapes[0])
    if any(len(s) != rank for s in shapes):
        raise NotImplementedError("np.concatenate: mixed ranks unsupported")
    if axis < 0:
        axis += rank
    return names, shapes, axis


def expand_concatenate(target: ast.expr,
                       args: List[ast.expr],
                       shape_table: Dict[str, Tuple[str, ...]],
                       kwargs=None,
                       local_dtypes=None,
                       fresh_local_allocs=None) -> List[ast.stmt]:
    """``out = np.concatenate((a, b, ...), axis=k)`` -- join along ``axis``. Each
    operand copies into ``out`` at its cumulative offset along ``axis`` (other
    axes index 1:1); generalises hstack/vstack to arbitrary axis and rank.

    A SLICED operand (dwt2d's rotate ``np.concatenate((e[:, 1:], e[:, 0:1]),
    axis=1)``) is spilled into a scratch buffer first -- the same
    materialisation einsum uses -- so the bare-Name join below is unchanged."""
    prelude: List[ast.stmt] = []
    if args and isinstance(args[0], (ast.Tuple, ast.List)):
        prelude, elts = _materialize_operands(args[0].elts,
                                              shape_table,
                                              "__cc_",
                                              local_dtypes=local_dtypes,
                                              fresh_local_allocs=fresh_local_allocs)
        args = [ast.Tuple(elts=list(elts), ctx=ast.Load())] + list(args[1:])
    names, shapes, axis = _concat_operands_axis(args, kwargs, shape_table)
    if any(nm is None for nm in names):  # materialisation above could not spill this operand
        raise NotImplementedError("np.concatenate: operand must be a Name")
    rank = len(shapes[0])
    iters = [_make_iter_name("__cc", d) for d in range(rank)]
    out: List[ast.stmt] = []
    offset_tok = "0"
    for nm, s in zip(names, shapes):
        tgt_elts: List[ast.expr] = []
        for d in range(rank):
            if d == axis and offset_tok != "0":
                tgt_elts.append(ast.BinOp(left=_name(iters[d]), op=ast.Add(), right=_const_or_name(offset_tok)))
            else:
                tgt_elts.append(_name(iters[d]))
        tgt_slot = tgt_elts[0] if rank == 1 else ast.Tuple(elts=tgt_elts, ctx=ast.Load())
        src_slot = (_name(iters[0]) if rank == 1 else ast.Tuple(elts=[_name(i) for i in iters], ctx=ast.Load()))
        body = [
            ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=tgt_slot, ctx=ast.Store())],
                       value=ast.Subscript(value=_name(nm), slice=src_slot, ctx=ast.Load()))
        ]
        out.extend(_wrap_for_loops(iters, s, body))
        offset_tok = (f"({offset_tok}) + ({s[axis]})" if offset_tok != "0" else str(s[axis]))
    return prelude + out


def _const_int(node: Optional[ast.expr]) -> Optional[int]:
    """A plain (possibly negative) int constant, or ``None``. A negative literal parses as
    ``UnaryOp(USub, Constant(n))`` -- not ``Constant(-n)`` -- so handle both."""
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)):
        return -node.operand.value
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _axis_literal_or_refuse(node: Optional[ast.expr], what: str, default: Optional[int] = None) -> int:
    """The literal axis in ``node``; ``default`` when the argument is ABSENT; a refusal otherwise.

    The distinction is the whole point. Reading an axis as ``default`` when it is merely
    UNREADABLE is how ``np.concatenate((a, b), axis=dim)`` came to concatenate along axis 0 and
    compile clean. An axis chooses the loop nest, so a static emitter has exactly two honest
    answers: the literal, or a refusal.
    """
    if node is None:
        if default is None:
            raise NotImplementedError(f"{what}: axis is required")
        return default
    axis = _const_int(node)
    if axis is None:
        raise NotImplementedError(f"{what}: axis {ast.unparse(node)!r} must be a compile-time integer "
                                  f"(it selects the loop nest, so there is no runtime form for it)")
    return axis


def _mul_exts(exprs) -> ast.expr:
    """Left-folded product of the given extent expressions (``1`` when empty) -- used to
    size a ``reshape(-1)`` dimension from the source extent and the other target dims."""
    exprs = list(exprs)
    if not exprs:
        return _const(1)
    prod = exprs[0]
    for e in exprs[1:]:
        prod = ast.BinOp(left=prod, op=ast.Mult(), right=e)
    return prod


def _stack_axis(args, kwargs, rank: int) -> int:
    """The (possibly negative) NEW-axis position for ``np.stack``, normalized to
    ``[0, rank]`` (an insert position, so ``rank`` -- append -- is valid, unlike
    concatenate's ``[0, rank)``)."""
    axis = _axis_literal_or_refuse(_kwarg_or_pos(args, kwargs, 1, "axis"), "np.stack", 0)
    if axis < 0:
        axis += rank + 1
    if not (0 <= axis <= rank):
        raise NotImplementedError("np.stack: axis out of range")
    return axis


def expand_stack(target: ast.expr,
                 args: List[ast.expr],
                 shape_table: Dict[str, Tuple[str, ...]],
                 kwargs=None) -> List[ast.stmt]:
    """``out = np.stack((a, b, ...), axis=k)`` -- join N same-shape operands along
    a NEW axis ``k`` (out's k-th extent = N, rank = operand rank + 1). Operand
    ``s`` copies to ``out`` at position ``s`` of the inserted axis, other axes
    1:1: ``out[.., s, ..] = operand_s[..]``. (concatenate joins an existing axis;
    stack inserts one.)"""
    names, shapes, _ = _concat_operands_axis(args, kwargs, shape_table)
    rank = len(shapes[0])
    axis = _stack_axis(args, kwargs, rank)
    iters = [_make_iter_name("__st", d) for d in range(rank)]
    src_slot = (_name(iters[0]) if rank == 1 else ast.Tuple(elts=[_name(i) for i in iters], ctx=ast.Load()))
    out: List[ast.stmt] = []
    for s_idx, (nm, s) in enumerate(zip(names, shapes)):
        tgt_elts = [_name(iters[d]) for d in range(rank)]
        tgt_elts.insert(axis, _const(s_idx))
        tgt_slot = tgt_elts[0] if len(tgt_elts) == 1 else ast.Tuple(elts=tgt_elts, ctx=ast.Load())
        body = [
            ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=tgt_slot, ctx=ast.Store())],
                       value=ast.Subscript(value=_name(nm), slice=src_slot, ctx=ast.Load()))
        ]
        out.extend(_wrap_for_loops(iters, s, body))
    return out


def expand_flip(target: ast.expr,
                args: List[ast.expr],
                shape_table: Dict[str, Tuple[str, ...]],
                kwargs=None) -> List[ast.stmt]:
    """``out = np.flip(A[, axis])`` -> reverse-order copy. Without ``axis`` EVERY axis is
    reversed (numpy's default); with ``axis=k`` only that axis. N-D: a loop nest over the
    shape where each flipped axis index ``i`` reads ``extent - 1 - i`` from the source."""
    if not args_one_name(args):
        raise NotImplementedError("np.flip needs Name arg")
    a = args[0]
    shape = shape_table.get(a.id)
    if not shape:
        raise NotImplementedError("np.flip: source shape unknown")
    rank = len(shape)
    axis_node = _kwarg_or_pos(args, kwargs, 1, "axis")
    if axis_node is None:
        flipped = set(range(rank))
    else:
        ax = _const_axis(axis_node, rank)
        if ax is None:
            raise NotImplementedError("np.flip: axis must be a constant int in range")
        flipped = {ax}
    iters = [f"__fl{d}" for d in range(rank)]
    src_elts: List[ast.expr] = []
    for d in range(rank):
        if d in flipped:
            ext = _const_or_name(shape[d])
            src_elts.append(
                ast.BinOp(left=ast.BinOp(left=ext, op=ast.Sub(), right=_const(1)), op=ast.Sub(), right=_name(iters[d])))
        else:
            src_elts.append(_name(iters[d]))
    out_slot = _name(iters[0]) if rank == 1 else ast.Tuple(elts=[_name(i) for i in iters], ctx=ast.Load())
    src_slot = src_elts[0] if rank == 1 else ast.Tuple(elts=src_elts, ctx=ast.Load())
    body = [
        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=out_slot, ctx=ast.Store())],
                   value=ast.Subscript(value=_name(a.id), slice=src_slot, ctx=ast.Load()))
    ]
    return _wrap_for_loops(iters, shape, body)


def expand_diff(target: ast.expr,
                args: List[ast.expr],
                shape_table: Dict[str, Tuple[str, ...]],
                kwargs=None) -> List[ast.stmt]:
    """``out = np.diff(A[, n][, axis])`` -> ``A[..., 1:, ...] - A[..., :-1, ...]`` along ``axis``.

    First difference only. ``n > 1`` is this applied again and needs a temporary per stage;
    ``prepend=``/``append=`` are a concatenate, which the caller can spell directly."""
    if not args_one_name(args):
        raise NotImplementedError("np.diff needs Name arg")
    a = args[0]
    shape = shape_table.get(a.id)
    if not shape:
        raise NotImplementedError("np.diff: source shape unknown")
    if {k.arg for k in (kwargs or [])} & {"prepend", "append"}:
        raise NotImplementedError("np.diff prepend=/append= is a concatenate; write the concatenate")
    rank = len(shape)
    n_node = _kwarg_or_pos(args, kwargs, 1, "n")
    if n_node is not None and _const_int(n_node) != 1:
        raise NotImplementedError("np.diff: only the first difference (n=1) is supported")
    ax_node = _kwarg_or_pos(args, kwargs, 2, "axis")
    ax = rank - 1 if ax_node is None else _const_axis(ax_node, rank)
    if ax is None:
        raise NotImplementedError("np.diff: axis must be a constant int in range")
    iters = [f"__df{d}" for d in range(rank)]
    bounds: List[Any] = list(shape)
    bounds[ax] = ast.BinOp(left=_const_or_name(shape[ax]), op=ast.Sub(), right=_const(1))

    def slot(offset: int) -> ast.expr:
        elts: List[ast.expr] = [_name(v) for v in iters]
        if offset:
            elts[ax] = ast.BinOp(left=_name(iters[ax]), op=ast.Add(), right=_const(offset))
        return elts[0] if rank == 1 else ast.Tuple(elts=elts, ctx=ast.Load())

    body = [
        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=slot(0), ctx=ast.Store())],
                   value=ast.BinOp(left=ast.Subscript(value=_name(a.id), slice=slot(1), ctx=ast.Load()),
                                   op=ast.Sub(),
                                   right=ast.Subscript(value=_name(a.id), slice=slot(0), ctx=ast.Load())))
    ]
    return _wrap_for_loops(iters, bounds, body)


def expand_std(target, args, shape_table, kwargs=None):
    """``s = np.std(A [, axis=k, keepdims=...])`` -- mean + sum of squared
    deviations + sqrt, over the unified axis-aware reduction scaffold (axis=None
    full reduction, axis=k vector along kept axes, keepdims preserves a size-1
    axis at k). Composed as a mean reduction (sum/count) then a
    sum-of-squared-deviations reduction over the same axes then sqrt(sum/count);
    goes through ``_expand_axis_reduction`` twice -- once into ``target`` for the
    mean, once into a scratch ``__sd`` for the squared deviations.
    """
    return _expand_var_or_std(target, args, shape_table, kwargs, finish="sqrt")


def expand_var(target, args, shape_table, kwargs=None):
    """``s = np.var(A [, axis=k, keepdims=...])`` -- mean + sum of
    squared deviations, no sqrt. Shares the scaffold with
    :func:`expand_std` (axis-tuple supported)."""
    return _expand_var_or_std(target, args, shape_table, kwargs, finish="none")


def _expand_var_or_std(target, args, shape_table, kwargs, finish: str):
    """Shared scaffold for ``np.var``/``np.std`` -- variance with optional sqrt
    finalisation. Supports the full ``axis = None/int/tuple/list`` and
    ``keepdims`` matrix: walks kept axes outside, reduces each reduction axis
    inside (deepest first), writes back ``sqrt(sum_of_squared_dev / divisor)``
    per kept-axis position (``divisor`` = product of the reduction-axis sizes).
    """
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError(f"np.{finish or 'var'} needs Name first arg")
    a = args[0]
    shape = _resolve_shape(a, shape_table)
    axes, keepdims = _read_axis_keepdims(args, kwargs)
    n_dim = len(shape)
    if axes is None:
        axes_norm = list(range(n_dim))
    else:
        axes_norm = []
        for ax in axes:
            na = ax + n_dim if ax < 0 else ax
            if na < 0 or na >= n_dim:
                raise NotImplementedError(f"np.{finish or 'var'} axis {ax} out of range")
            if na in axes_norm:
                raise NotImplementedError(f"np.{finish or 'var'} duplicate axis {ax}")
            axes_norm.append(na)
    axes_set = set(axes_norm)
    kept_axes = [k for k in range(n_dim) if k not in axes_set]

    # Step 1: compute the mean (axis-aware) into ``target``.
    mean_stmts = _expand_axis_reduction(
        target,
        args,
        kwargs,
        shape_table,
        init=_const(0.0),
        op_fn=lambda acc, x: ast.BinOp(left=acc, op=ast.Add(), right=x),
        post_fn=lambda lvalue, divisor: ast.Assign(targets=[lvalue],
                                                   value=ast.BinOp(left=(lvalue if isinstance(
                                                       lvalue, ast.Name) else ast.Subscript(
                                                           value=lvalue.value, slice=lvalue.slice, ctx=ast.Load())),
                                                                   op=ast.Div(),
                                                                   right=divisor)))

    # Step 2: accumulate squared deviations into ``__sd_acc`` (scalar per
    # kept-axes position), then finalise as ``out = (__sd_acc / divisor)``, sqrt
    # wrapped for std. The mean target slot stays alive through the inner
    # reduction loops; only the outer (kept) loop nest overwrites it.
    outer_iter_names = [_make_iter_name("__sax", i) for i in range(len(kept_axes))]
    red_iter_names = [_make_iter_name("__srd", i) for i in range(len(axes_norm))]
    red_iter_map = dict(zip(axes_norm, red_iter_names))

    def _src_elts():
        out = []
        outer_pos = 0
        for k in range(n_dim):
            if k in axes_set:
                out.append(_name(red_iter_map[k]))
            else:
                out.append(_name(outer_iter_names[outer_pos]))
                outer_pos += 1
        return out

    def _out_elts():
        out = []
        outer_pos = 0
        for k in range(n_dim):
            if k in axes_set:
                if keepdims:
                    out.append(_const(0))
            else:
                out.append(_name(outer_iter_names[outer_pos]))
                outer_pos += 1
        return out

    src_elts = _src_elts()
    src_slot = src_elts[0] if n_dim == 1 else ast.Tuple(elts=src_elts, ctx=ast.Load())
    out_elts = _out_elts()
    is_scalar_target = len(out_elts) == 0
    if is_scalar_target:
        out_sub: ast.expr = target
        out_load: ast.expr = ast.Name(id=target.id, ctx=ast.Load())
    elif len(out_elts) == 1:
        out_sub = ast.Subscript(value=_name(target.id), slice=out_elts[0], ctx=ast.Store())
        out_load = ast.Subscript(value=_name(target.id), slice=out_elts[0], ctx=ast.Load())
    else:
        slot = ast.Tuple(elts=out_elts, ctx=ast.Load())
        out_sub = ast.Subscript(value=_name(target.id), slice=slot, ctx=ast.Store())
        out_load = ast.Subscript(value=_name(target.id), slice=slot, ctx=ast.Load())
    src_sub = ast.Subscript(value=_name(a.id), slice=src_slot, ctx=ast.Load())
    sd_acc = "__sd_acc"
    diff = ast.BinOp(left=src_sub, op=ast.Sub(), right=out_load)
    sq = ast.BinOp(left=diff, op=ast.Mult(), right=diff)
    init_acc = ast.Assign(targets=[_store(sd_acc)], value=_const(0.0))
    add_acc = ast.AugAssign(target=_store(sd_acc), op=ast.Add(), value=sq)
    # Divisor = product of every reduction-axis size, minus ``ddof`` (numpy's
    # ``np.var``/``np.std`` divide by ``N - ddof``; ddof defaults to 0). The
    # mean above always divides by the full ``N`` -- only the variance honors
    # ddof.
    divisor: ast.expr = _const_or_name(shape[axes_norm[0]])
    for ax in axes_norm[1:]:
        divisor = ast.BinOp(left=divisor, op=ast.Mult(), right=_const_or_name(shape[ax]))
    ddof = _read_kwarg(kwargs, "ddof")
    if ddof is not None and not (isinstance(ddof, ast.Constant) and ddof.value == 0):
        divisor = ast.BinOp(left=divisor, op=ast.Sub(), right=copy.deepcopy(ddof))
    finalize_value: ast.expr = ast.BinOp(left=_name(sd_acc), op=ast.Div(), right=divisor)
    if finish == "sqrt":
        finalize_value = ast.Call(func=_name("sqrt"), args=[finalize_value], keywords=[])
    finalize = ast.Assign(targets=[out_sub], value=finalize_value)
    # Wrap inner reduction iters around add_acc, deepest first.
    inner_body: List[ast.stmt] = [add_acc]
    for ax, rn in zip(reversed(axes_norm), reversed(red_iter_names)):
        inner_body = [
            ast.For(target=_store(rn),
                    iter=ast.Call(func=_name("range"), args=[_const_or_name(shape[ax])], keywords=[]),
                    body=inner_body,
                    orelse=[])
        ]
    body_stmts = [init_acc, *inner_body, finalize]
    if is_scalar_target:
        return mean_stmts + body_stmts
    bounds = tuple(shape[k] for k in kept_axes)
    return mean_stmts + _wrap_for_loops(outer_iter_names, bounds, body_stmts)


def expand_dot_2d(target: ast.expr, args: List[ast.expr], shape_table: Dict[str, Tuple[str, ...]]) -> List[ast.stmt]:
    """``out = np.dot(A, b)`` -> matrix-vector / matrix-matrix. Routes 1-D x 1-D
    (both operands with 1-D iteration extent -- bare Names or slices like
    ``A[i, :j]``) to :func:`expand_dot`; the remaining branches handle
    matrix-vector/vector-matrix, requiring both operands to be bare Names with
    declared 2-D shape.
    """
    if len(args) != 2:
        raise NotImplementedError("np.dot needs 2 args")
    a, b = args
    a_ext = _iter_extent_of(a, shape_table)
    b_ext = _iter_extent_of(b, shape_table)
    if a_ext is not None and b_ext is not None and len(a_ext) == 1 and len(b_ext) == 1:
        return expand_dot(target, args, shape_table)
    if not (isinstance(a, ast.Name) and isinstance(b, ast.Name)):
        raise NotImplementedError("np.dot mv/vm/mm needs bare Name args")
    a_shape, b_shape = shape_table.get(a.id), shape_table.get(b.id)
    if not a_shape or not b_shape:
        raise NotImplementedError("np.dot: shapes unknown")
    if len(a_shape) == 1 and len(b_shape) == 1:
        return expand_dot(target, args, shape_table)
    if len(a_shape) == 2 and len(b_shape) == 1:
        m, k = a_shape
        return [
            ast.For(target=_store("__i"),
                    iter=ast.Call(func=_name("range"), args=[_const_or_name(m)], keywords=[]),
                    body=[
                        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=_name("__i"), ctx=ast.Store())],
                                   value=_const(0.0)),
                        ast.For(target=_store("__l"),
                                iter=ast.Call(func=_name("range"), args=[_const_or_name(k)], keywords=[]),
                                body=[
                                    ast.AugAssign(target=ast.Subscript(value=_name(target.id),
                                                                       slice=_name("__i"),
                                                                       ctx=ast.Store()),
                                                  op=ast.Add(),
                                                  value=ast.BinOp(left=ast.Subscript(
                                                      value=_name(a.id),
                                                      slice=ast.Tuple(elts=[_name("__i"), _name("__l")],
                                                                      ctx=ast.Load()),
                                                      ctx=ast.Load()),
                                                                  op=ast.Mult(),
                                                                  right=ast.Subscript(value=_name(b.id),
                                                                                      slice=_name("__l"),
                                                                                      ctx=ast.Load())))
                                ],
                                orelse=[]),
                    ],
                    orelse=[])
        ]
    if len(a_shape) == 1 and len(b_shape) == 2:
        k, n = b_shape
        return [
            ast.For(target=_store("__j"),
                    iter=ast.Call(func=_name("range"), args=[_const_or_name(n)], keywords=[]),
                    body=[
                        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=_name("__j"), ctx=ast.Store())],
                                   value=_const(0.0)),
                        ast.For(target=_store("__l"),
                                iter=ast.Call(func=_name("range"), args=[_const_or_name(k)], keywords=[]),
                                body=[
                                    ast.AugAssign(target=ast.Subscript(value=_name(target.id),
                                                                       slice=_name("__j"),
                                                                       ctx=ast.Store()),
                                                  op=ast.Add(),
                                                  value=ast.BinOp(left=ast.Subscript(value=_name(a.id),
                                                                                     slice=_name("__l"),
                                                                                     ctx=ast.Load()),
                                                                  op=ast.Mult(),
                                                                  right=ast.Subscript(
                                                                      value=_name(b.id),
                                                                      slice=ast.Tuple(elts=[_name("__l"),
                                                                                            _name("__j")],
                                                                                      ctx=ast.Load()),
                                                                      ctx=ast.Load())))
                                ],
                                orelse=[]),
                    ],
                    orelse=[])
        ]
    # 2-D x 2-D: delegate to matmul.
    return expand_matmul(target, a, b, shape_table)


# Einsum / tensor-contraction family.


def _parse_einsum_subscripts(spec: str):
    """Split ``"ij,jk->ik"`` into ``(["ij", "jk"], "ik")``. The explicit ``->``
    form is required; the implicit-output form (no ``->``) is synthesised as
    numpy does: every index appearing exactly once across all inputs, in
    alphabetical order. ``...`` ellipsis raises (unsupported)."""
    spec = spec.replace(" ", "")
    if "..." in spec:
        raise NotImplementedError("einsum ellipsis unsupported")
    if "->" in spec:
        lhs, rhs = spec.split("->")
    else:
        lhs = spec
        counts: Dict[str, int] = {}
        for ch in lhs.replace(",", ""):
            counts[ch] = counts.get(ch, 0) + 1
        rhs = "".join(sorted(c for c, n in counts.items() if n == 1))
    inputs = lhs.split(",")
    return inputs, rhs


def _expand_einsum_ellipsis(spec: str, ranks: List[int]) -> str:
    """Rewrite ``...`` in an einsum spec to explicit index letters using each
    operand's rank, so the plain-subscript lowering handles it:
    ``'...ij,...jk->...ik'`` on rank-3 operands -> ``'Aij,Ajk->Aik'`` (the
    broadcast axis becomes a fresh shared index ``A``). Requires an explicit
    ``->`` and that every ``...`` covers the same axis count (numpy's
    differing-rank ellipsis broadcasting isn't modelled) -- else raises.
    """
    spec = spec.replace(" ", "")
    if "->" not in spec:
        raise NotImplementedError("einsum: ellipsis requires an explicit -> output")
    lhs, rhs = spec.split("->")
    ins = lhs.split(",")
    if len(ins) != len(ranks):
        raise NotImplementedError("einsum: operand count != subscript count")
    ell_rank, seen = 0, False
    for sub, r in zip(ins, ranks):
        if "..." in sub:
            er = r - len(sub.replace("...", ""))
            if er < 0:
                raise NotImplementedError("einsum: too many indices for operand rank")
            if seen and er != ell_rank:
                raise NotImplementedError("einsum: differing ellipsis ranks (broadcast) unsupported")
            ell_rank, seen = er, True
    used = set(spec) - set(".,->")
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    fresh = [c for c in letters if c not in used]
    if len(fresh) < ell_rank:
        raise NotImplementedError("einsum: not enough free index letters for ellipsis")
    ell = "".join(fresh[:ell_rank])
    new_ins = [sub.replace("...", ell) for sub in ins]
    return ",".join(new_ins) + "->" + rhs.replace("...", ell)


#: Counter for the scratch buffers :func:`_materialize_operands` spills non-Name operands into.
#: Reset per translation unit by :func:`reset_temp_counters`.
_OP_SPILL_TEMP = [0]


def _materialize_operands(operands, shape_table, prefix: str, local_dtypes=None, fresh_local_allocs=None):
    """Spill every non-Name operand into a fresh scratch buffer. Several
    expanders (einsum, concatenate) are written against bare-Name operands, but
    a kernel may hand them a slice/call/arithmetic expression -- dwt2d's
    ``np.concatenate((e[:, 1:], e[:, 0:1]), axis=1)``. Copying each into a local
    first lets the bare-Name expansion run unchanged, instead of teaching every
    expander to index arbitrary sub-expressions.

    ``prefix`` is the caller's name stem (``__es_``/``__cc_``): the buffer is
    ``<prefix>op<N>``, its copy-loop iterators ``<prefix>c<i>``.

    Returns ``(prelude_stmts, operand_exprs)``: the copy loops to emit first,
    and the operand list with each spill replaced by its buffer Name. The
    buffer is registered in ``shape_table`` + ``fresh_local_allocs`` so the
    emitter declares and allocates it, inheriting the source array's dtype. An
    operand whose extent won't resolve passes through untouched, so the
    caller's own bare-Name check still reports it.
    """
    prelude: List[ast.stmt] = []
    out: List[ast.expr] = []
    for op in operands:
        if isinstance(op, ast.Name):
            out.append(op)
            continue
        op_ext = _iter_extent_of(op, shape_table)
        if op_ext is None:
            out.append(op)  # unresolved -- the caller's bare-Name check raises
            continue
        _OP_SPILL_TEMP[0] += 1
        tmp = f"{prefix}op{_OP_SPILL_TEMP[0]}"
        tmp_shape = tuple(_CallHoister._extent_to_shape_token(e) for e in op_ext)
        shape_table[tmp] = tmp_shape
        if fresh_local_allocs is not None:
            fresh_local_allocs[tmp] = tmp_shape
        if local_dtypes is not None:
            base = op.value if isinstance(op, ast.Subscript) else op
            base_name = _name_id(base)
            if base_name and local_dtypes.get(base_name):
                local_dtypes[tmp] = local_dtypes[base_name]
        cp_iters = [f"{prefix}c{i}" for i in range(len(op_ext))]
        cp_nodes = [_name(c) for c in cp_iters]
        cp_slot = cp_nodes[0] if len(cp_nodes) == 1 else ast.Tuple(elts=cp_nodes, ctx=ast.Load())
        cp_src = _scalarize_at_iters(op, cp_nodes, shape_table)
        cp_dst = ast.Subscript(value=_name(tmp), slice=cp_slot, ctx=ast.Store())
        prelude.append(_alloc_marker(tmp))
        prelude.extend(_wrap_for_loops(cp_iters, list(tmp_shape), [ast.Assign(targets=[cp_dst], value=cp_src)]))
        out.append(_name(tmp))
    return prelude, out


def expand_einsum(target: ast.expr,
                  args: List[ast.expr],
                  shape_table: Dict[str, Tuple[str, ...]],
                  local_dtypes=None,
                  fresh_local_allocs=None) -> List[ast.stmt]:
    """Lower ``np.einsum(subscripts, *operands)`` to a nested loop nest. Output
    indices become nested loops over the result; indices summed away (in the
    inputs but not the output) become inner accumulation loops. Body:
    ``out[out_idx] (+)= prod(operand[its idx letters])``.

    Handles N operands and arbitrary index letters, including a letter repeated
    within one operand (``ii`` -> diagonal ``A[i, i]``). One path subsumes
    matmul ``ij,jk->ik``, transpose ``ij->ji``, trace ``ii->``, diagonal
    ``ii->i``, outer ``i,j->ij`` and sum ``ij->``.

    A non-Name operand (Subscript/Call/BinOp like ``psi_frag[f]``) is first
    spilled into a fresh scratch buffer (the same copy-into-local pattern
    ``expand_copy``/``expand_median`` use), so the bare-Name expansion below
    contracts the buffer.
    """
    if not args or not isinstance(args[0], ast.Constant) or not isinstance(args[0].value, str):
        raise NotImplementedError("einsum needs a literal subscript string")
    spec = args[0].value
    operands = args[1:]
    # Materialize any non-Name operand into a fresh local so the bare-Name expansion (and
    # the ellipsis rank lookup) sees a Name.
    prelude, operands = _materialize_operands(operands,
                                              shape_table,
                                              "__es_",
                                              local_dtypes=local_dtypes,
                                              fresh_local_allocs=fresh_local_allocs)
    if "..." in spec:
        # Expand ``...`` to explicit letters from each operand's rank (needs the
        # shape table), then lower the plain form; the parser stays ellipsis-free.
        ranks = []
        for op in operands:
            if not isinstance(op, ast.Name) or shape_table.get(op.id) is None:
                raise NotImplementedError("einsum ellipsis needs bare-Name operands with known shape")
            ranks.append(len(shape_table[op.id]))
        spec = _expand_einsum_ellipsis(spec, ranks)
    inputs, output = _parse_einsum_subscripts(spec)
    if len(inputs) != len(operands):
        raise NotImplementedError("einsum operand count mismatches subscripts")
    operand_names: List[str] = []
    for op in operands:
        if not isinstance(op, ast.Name):
            raise NotImplementedError("einsum operands must be bare Names")
        operand_names.append(op.id)
    # Map every index letter to its extent symbol (first operand that uses it).
    letter_extent: Dict[str, str] = {}
    for spec, name in zip(inputs, operand_names):
        shape = shape_table.get(name)
        if shape is None or len(shape) != len(spec):
            raise NotImplementedError(f"einsum: shape of {name!r} unknown / rank mismatch")
        for letter, dim in zip(spec, shape):
            letter_extent.setdefault(letter, dim)
    out_letters = list(output)
    sum_letters = [c for c in letter_extent if c not in out_letters]
    # Per-letter loop variable.
    var_of = {c: f"__es_{c}" for c in letter_extent}

    def _subscript(name: str, spec: str) -> ast.expr:
        idx = [_name(var_of[c]) for c in spec]
        sl = idx[0] if len(idx) == 1 else ast.Tuple(elts=idx, ctx=ast.Load())
        return ast.Subscript(value=_name(name), slice=sl, ctx=ast.Load())

    # Product of every operand scalarised at its index letters.
    product: ast.expr = _subscript(operand_names[0], inputs[0])
    for name, spec in zip(operand_names[1:], inputs[1:]):
        product = ast.BinOp(left=product, op=ast.Mult(), right=_subscript(name, spec))

    # Output write target.
    if out_letters:
        out_idx = [_name(var_of[c]) for c in out_letters]
        out_sl = out_idx[0] if len(out_idx) == 1 else ast.Tuple(elts=out_idx, ctx=ast.Load())
        out_store = ast.Subscript(value=_name(target.id), slice=out_sl, ctx=ast.Store())
    else:
        out_store = _store(target.id)  # scalar result (trace / full sum)

    # Inner: accumulate the product over the summed letters.
    if sum_letters:
        body: List[ast.stmt] = [ast.AugAssign(target=out_store, op=ast.Add(), value=product)]
        body = _wrap_for_loops([var_of[c] for c in sum_letters], [letter_extent[c] for c in sum_letters], body)
        zero = ast.Assign(targets=[copy.deepcopy(out_store)], value=_const(0.0))
        inner: List[ast.stmt] = [zero] + body
    else:
        inner = [ast.Assign(targets=[copy.deepcopy(out_store)], value=product)]

    if out_letters:
        return prelude + _wrap_for_loops([var_of[c] for c in out_letters], [letter_extent[c]
                                                                            for c in out_letters], inner)
    return prelude + inner


def expand_tensordot(target: ast.expr,
                     args: List[ast.expr],
                     shape_table: Dict[str, Tuple[str, ...]],
                     kwargs=None,
                     local_dtypes=None,
                     fresh_local_allocs=None) -> List[ast.stmt]:
    """``np.tensordot(a, b, axes)`` -> an equivalent einsum.

    ``axes`` is an int K (contract the last K axes of ``a`` with the first K
    of ``b``) or a pair of axis lists. Default ``axes=2``.

    A non-Name operand (conv2d's sliced ``input[:, ki:ki+H_out, kj:kj+W_out,
    :]`` and partially-indexed ``weights[ki, kj]``) is spilled into a fresh
    scratch buffer first, the same ``_materialize_operands`` pattern
    :func:`expand_einsum` uses -- under a distinct ``__td_`` prefix so its
    temps/copy-iterators cannot collide with einsum's own ``__es_`` spill
    once the mapped call below runs.
    """
    if len(args) < 2:
        raise NotImplementedError("tensordot needs 2 array args")
    prelude, (a, b) = _materialize_operands(args[:2],
                                            shape_table,
                                            "__td_",
                                            local_dtypes=local_dtypes,
                                            fresh_local_allocs=fresh_local_allocs)
    if not (isinstance(a, ast.Name) and isinstance(b, ast.Name)):
        raise NotImplementedError("tensordot operands must be bare Names")
    ra = len(shape_table.get(a.id, ()))
    rb = len(shape_table.get(b.id, ()))
    if not ra or not rb:
        raise NotImplementedError("tensordot: operand shapes unknown")
    axes_node = args[2] if len(args) > 2 else _axes_kwarg(kwargs)
    a_ax, b_ax = _tensordot_axes(axes_node, ra, rb)
    letters = "abcdefghijklmnopqrstuvwxyz"
    a_spec = list(letters[:ra])
    b_spec = [None] * rb
    # Shared contraction letters: pair a_ax[i] <-> b_ax[i].
    nxt = ra
    for ca, cb in zip(a_ax, b_ax):
        b_spec[cb] = a_spec[ca]
    for i in range(rb):
        if b_spec[i] is None:
            b_spec[i] = letters[nxt]
            nxt += 1
    out_spec = [c for i, c in enumerate(a_spec) if i not in a_ax] + \
               [c for i, c in enumerate(b_spec) if i not in b_ax]
    spec = f"{''.join(a_spec)},{''.join(b_spec)}->{''.join(out_spec)}"
    return prelude + expand_einsum(
        target, [_const(spec), a, b], shape_table, local_dtypes=local_dtypes, fresh_local_allocs=fresh_local_allocs)


def _axes_kwarg(kwargs):
    for kw in kwargs or []:
        if kw.arg == "axes":
            return kw.value
    return _const(2)


def _tensordot_axes(node: ast.expr, ra: int, rb: int):
    """Resolve tensordot ``axes`` into ``(a_axes, b_axes)``, normalised against each operand's rank.

    An axis past the operand's rank means the rank we resolved is not the rank the kernel meant --
    cp2k_grid_integrate contracts axis 3 of a ``np.where`` result whose recorded shape is rank 3,
    because the broadcast against a 4-D operand was never folded into it. Declining is the only
    sound answer: a spec built from a wrong rank contracts the wrong axes and still emits.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        k = node.value
        if k > ra or k > rb:
            raise NotImplementedError(f"tensordot axes={k} exceeds operand rank ({ra}, {rb})")
        return list(range(ra - k, ra)), list(range(k))
    if isinstance(node, (ast.Tuple, ast.List)) and len(node.elts) == 2:

        def _axis_literal(x):
            # A negative axis parses as UnaryOp(USub), not Constant -- reading ``.value`` off it
            # raised AttributeError out of the sizer, the same escaped-exception class as the
            # out-of-range index below.
            try:
                value = ast.literal_eval(x)
            except (ValueError, SyntaxError, TypeError):
                raise NotImplementedError("tensordot axes entries must be integer literals")
            return value

        def _axis_list(e):
            if isinstance(e, (ast.Tuple, ast.List)):
                return [_axis_literal(x) for x in e.elts]
            return [_axis_literal(e)]

        def _normalised(axes, rank, side):
            out = []
            for ax in axes:
                if not isinstance(ax, int):
                    raise NotImplementedError("tensordot axes entries must be integer literals")
                pos = ax + rank if ax < 0 else ax
                if not 0 <= pos < rank:
                    raise NotImplementedError(f"tensordot {side} axis {ax} is outside rank {rank}")
                out.append(pos)
            return out

        a_ax = _normalised(_axis_list(node.elts[0]), ra, "a")
        b_ax = _normalised(_axis_list(node.elts[1]), rb, "b")
        if len(a_ax) != len(b_ax):
            raise NotImplementedError("tensordot axis lists must have equal length")
        return a_ax, b_ax
    raise NotImplementedError("tensordot axes must be an int or a 2-tuple of axis lists")


def expand_inner(target: ast.expr, args: List[ast.expr], shape_table: Dict[str, Tuple[str, ...]]) -> List[ast.stmt]:
    """``np.inner(a, b)`` -> contract the LAST axis of each operand.

    Rank-1 x rank-1 is the plain dot product (routes to :func:`expand_dot`)."""
    if len(args) != 2:
        raise NotImplementedError("np.inner needs 2 args")
    a, b = args
    if not (isinstance(a, ast.Name) and isinstance(b, ast.Name)):
        raise NotImplementedError("np.inner operands must be bare Names")
    ra = len(shape_table.get(a.id, ()))
    rb = len(shape_table.get(b.id, ()))
    if ra == 1 and rb == 1:
        return expand_dot(target, args, shape_table)
    letters = "abcdefghijklmnopqrstuvwxyz"
    a_spec = list(letters[:ra])
    b_spec = list(letters[ra:ra + rb])
    b_spec[-1] = a_spec[-1]  # contract the last axis of each
    out_spec = a_spec[:-1] + b_spec[:-1]
    spec = f"{''.join(a_spec)},{''.join(b_spec)}->{''.join(out_spec)}"
    return expand_einsum(target, [_const(spec), a, b], shape_table)


def expand_vdot(target: ast.expr,
                args: List[ast.expr],
                shape_table: Dict[str, Tuple[str, ...]],
                local_dtypes=None) -> List[ast.stmt]:
    """``np.vdot(a, b)`` -> ``sum(conj(a) * b)`` over the flattened operands.
    The conjugate is emitted only when the first operand is complex (a no-op
    for reals, and invalid on a real scalar); real operands reduce to the plain
    dot product."""
    if len(args) != 2:
        raise NotImplementedError("np.vdot needs 2 args")
    a, b = args
    if not (isinstance(a, ast.Name) and isinstance(b, ast.Name)):
        raise NotImplementedError("np.vdot operands must be bare Names")
    shape = shape_table.get(a.id)
    if shape is None:
        raise NotImplementedError("np.vdot: operand shape unknown")
    if len(shape) != 1:
        raise NotImplementedError("np.vdot only supports rank-1 operands")
    is_complex = bool(local_dtypes and str(local_dtypes.get(a.id, "")).startswith("complex"))
    it = "__vd"
    a_elem: ast.expr = ast.Subscript(value=_name(a.id), slice=_name(it), ctx=ast.Load())
    if is_complex:
        a_elem = _attr_call("np", "conj", [a_elem])
    prod = ast.BinOp(left=a_elem,
                     op=ast.Mult(),
                     right=ast.Subscript(value=_name(b.id), slice=_name(it), ctx=ast.Load()))
    body = [ast.AugAssign(target=_store(target.id), op=ast.Add(), value=prod)]
    return [ast.Assign(targets=[_store(target.id)], value=_const(0.0)), *_wrap_for_loops([it], [shape[0]], body)]


def _pad_widths(pad_arg: Optional[ast.expr], n_axes: int):
    """Per-axis ``(before, after)`` pad widths for an ``np.pad`` call. Accepts
    the two numpy spellings the corpus uses: a scalar ``R`` (int/Name), padding
    every axis ``(R, R)``; or a tuple of per-axis ``(before, after)`` pairs --
    ``((R, R), (R, R), (R, R), (0, 0))`` (vector stencils leave the component
    axis unpadded), where a bare int means ``(v, v)``. Returns a length-``n_axes``
    list of ``(before_node, after_node)`` AST exprs, or ``None`` if unresolvable."""
    if isinstance(pad_arg, (ast.Constant, ast.Name)):
        return [(pad_arg, pad_arg) for _ in range(n_axes)]
    if isinstance(pad_arg, (ast.Tuple, ast.List)) and len(pad_arg.elts) == n_axes:
        out = []
        for e in pad_arg.elts:
            if isinstance(e, (ast.Tuple, ast.List)) and len(e.elts) == 2:
                out.append((e.elts[0], e.elts[1]))
            elif isinstance(e, (ast.Constant, ast.Name)):
                out.append((e, e))
            else:
                return None
        return out
    return None


def _pad_output_extent(src_extent, pad_arg: Optional[ast.expr]):
    """Output extent of ``np.pad``: each source axis grown by ``before+after``.

    ``src_extent`` is the tuple of source-axis extent AST nodes; returns the
    per-axis output extent nodes, or ``None`` if ``pad_arg`` is unsupported."""
    widths = _pad_widths(pad_arg, len(src_extent))
    if widths is None:
        return None
    out = []
    for d, (before, after) in zip(src_extent, widths):
        total = ast.BinOp(left=copy.deepcopy(before), op=ast.Add(), right=copy.deepcopy(after))
        out.append(ast.BinOp(left=d, op=ast.Add(), right=total))
    return tuple(out)


def _pad_src_base_and_lead(src_node: ast.expr):
    """Split an ``np.pad`` source into ``(base_name, lead_scalar_indices)``. A
    bare ``Name`` pads the whole array (no lead). A ``Subscript`` with leading
    SCALAR indices -- ``in_grid[b]`` (stencil_4d) -- pads the sliced sub-array,
    so the lead scalars prepend to every generated source read. ``None`` for
    any other form (partial-slice subscripts etc.)."""
    if isinstance(src_node, ast.Name):
        return src_node.id, []
    if isinstance(src_node, ast.Subscript) and isinstance(src_node.value, ast.Name):
        sl = src_node.slice
        elts = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
        # Drop TRAILING whole-axis selections: ``in_grid[b]`` is normalized to
        # ``in_grid[b, :, :, :]`` before this runs, and a bare ``:`` selects the
        # same sub-array the lead-indexed form names. A PARTIAL slice (``a:b``)
        # still bails out below.
        while elts and _is_full_slice_elt(elts[-1]):
            elts.pop()
        if any(isinstance(e, ast.Slice) for e in elts):
            return None
        return src_node.value.id, elts
    return None


def _pad_mode_str(args: List[ast.expr], kwargs) -> str:
    """The ``mode`` string of an ``np.pad`` call (default numpy ``constant``)."""
    m = _kwarg_or_pos(args, kwargs or [], 2, "mode")
    if isinstance(m, ast.Constant) and isinstance(m.value, str):
        return m.value
    return "constant"


def _pad_fill(kwargs) -> ast.expr:
    """``np.pad``'s ``constant_values``, or numpy's own default of 0.

    A per-axis sequence is refused rather than guessed. Not cosmetic: max_filter pads its tail with
    ``-inf`` precisely so the running maximum never sees it, and filling zeros there instead
    returned a too-large maximum on every trailing block -- a wrong answer, not a refusal.
    """
    for keyword in (kwargs or []):
        if keyword.arg != "constant_values":
            continue
        if isinstance(keyword.value, (ast.Tuple, ast.List)):
            raise NotImplementedError("np.pad: per-axis constant_values unsupported")
        return copy.deepcopy(keyword.value)
    return _const(0.0)


def expand_pad(target: ast.expr,
               args: List[ast.expr],
               shape_table: Dict[str, Tuple[str, ...]],
               kwargs=None,
               local_dtypes=None,
               fresh_local_allocs=None) -> List[ast.stmt]:
    """``padded = np.pad(src, pad_width, mode=...)`` -> a ghost-cell fill loop.
    Each source axis grows by its ``before + after`` width (scalar ``R`` pads
    every axis ``(R, R)``; a per-axis tuple lets vector stencils leave the
    component axis ``(0, 0)``). Source may be a bare array or a
    leading-scalar-indexed sub-array (``in_grid[b]``). Two modes: ``edge``
    takes the nearest source edge value (``padded[i...] = src[clamp(i - before,
    0, d - 1)...]``, clamp emitted as one conditional-expression assign to a
    scalar ``__ps<k>`` local, no min/max in subscript position); ``constant``
    (numpy default) zeroes the buffer then copies the interior ``padded[i +
    before...] = src[i...]``. The halo-exchange idiom of the structured-grid
    stencils."""
    if not args:
        raise NotImplementedError("np.pad needs a source operand")
    base = _pad_src_base_and_lead(args[0])
    if base is None:
        raise NotImplementedError("np.pad source must be a Name or scalar-indexed sub-array")
    base_name, lead = base
    src_ext = _iter_extent_of(args[0], shape_table)
    if src_ext is None:
        raise NotImplementedError(f"np.pad: shape of {base_name!r} unknown")
    view = [_const_or_name(s) if isinstance(s, str) else s for s in src_ext]
    pad_arg = _kwarg_or_pos(args, kwargs or [], 1, "pad_width")
    widths = _pad_widths(pad_arg, len(view))
    if widths is None:
        raise NotImplementedError("np.pad needs scalar or per-axis tuple pad_width")
    mode = _pad_mode_str(args, kwargs)
    if mode not in ("edge", "constant", "reflect", "wrap", "symmetric"):
        raise NotImplementedError(f"np.pad mode={mode!r} unsupported")
    rank = len(view)

    def _before(k):
        return copy.deepcopy(widths[k][0])

    def _dim(k):
        return copy.deepcopy(view[k])

    out_bounds = [b for b in _pad_output_extent(tuple(_dim(k) for k in range(rank)), pad_arg)]

    def _store_target(idx_nodes):
        sl = idx_nodes[0] if rank == 1 else ast.Tuple(elts=idx_nodes, ctx=ast.Load())
        return ast.Subscript(value=_name(target.id), slice=sl, ctx=ast.Store())

    def _src_read(idx_nodes):
        full = [copy.deepcopy(e) for e in lead] + idx_nodes
        sl = full[0] if len(full) == 1 else ast.Tuple(elts=full, ctx=ast.Load())
        return ast.Subscript(value=_name(base_name), slice=sl, ctx=ast.Load())

    if mode == "constant":
        # Fill the whole padded buffer with the pad value, then copy the interior shifted by before.
        zero_iters = [f"__pz{k}" for k in range(rank)]
        zero_body = [ast.Assign(targets=[_store_target([_name(v) for v in zero_iters])], value=_pad_fill(kwargs))]
        stmts = _wrap_for_loops(zero_iters, out_bounds, zero_body)
        cp_iters = [f"__pc{k}" for k in range(rank)]
        dst_idx = [ast.BinOp(left=_name(cp_iters[k]), op=ast.Add(), right=_before(k)) for k in range(rank)]
        cp_body = [ast.Assign(targets=[_store_target(dst_idx)], value=_src_read([_name(v) for v in cp_iters]))]
        stmts += _wrap_for_loops(cp_iters, [_dim(k) for k in range(rank)], cp_body)
        return stmts

    # Boundary modes: each output cell reads the source cell whose index is a
    # mode-specific remap of ``q = out_iter - before`` back into ``[0, d-1]``
    # (edge = clamp, wrap = periodic, reflect/symmetric = mirror), emitted as
    # scalar ``__ps<k>`` locals so no min/max/mod sits in subscript position.
    # edge clamps in ONE conditional expression (see _remap); the fold/mod modes
    # keep their statement sequence, which reads sv back.
    out_iters = [f"__pp{k}" for k in range(rank)]
    src_idx_vars = [f"__ps{k}" for k in range(rank)]

    def _floor_mod(x: ast.expr, m: ast.expr) -> ast.expr:
        # ((x % m) + m) % m -- a floor modulo, correct whether the backend's
        # ``%`` truncates (C) or floors, keeping the index in [0, m).
        inner = ast.BinOp(left=x, op=ast.Mod(), right=copy.deepcopy(m))
        return ast.BinOp(left=ast.BinOp(left=inner, op=ast.Add(), right=copy.deepcopy(m)),
                         op=ast.Mod(),
                         right=copy.deepcopy(m))

    def _fold_high(sv: str, hi: ast.expr, d: ast.expr) -> ast.stmt:
        # ``if sv >= d: sv = hi - sv`` -- fold the period's upper half down.
        return ast.If(test=ast.Compare(left=_name(sv), ops=[ast.GtE()], comparators=[copy.deepcopy(d)]),
                      body=[ast.Assign(targets=[_store(sv)], value=ast.BinOp(left=hi, op=ast.Sub(), right=_name(sv)))],
                      orelse=[])

    def _remap(sv: str, d: ast.expr, pv: str, raw: ast.expr) -> List[ast.stmt]:
        if mode == "edge":
            # ONE conditional-expression assign, not two guard ifs: a polyhedral
            # extractor (pluto/pet) reads data-dependent control flow inside the
            # loop body as a statement it cannot schedule and drops the body.
            # Every arm recomputes the PRE-clamp ``raw``; reading sv back would
            # add a RAW dependence on top of the WAW the assign already carries.
            upper = ast.BinOp(left=copy.deepcopy(d), op=ast.Sub(), right=_const(1))
            hi = ast.IfExp(test=ast.Compare(left=copy.deepcopy(raw), ops=[ast.Gt()],
                                            comparators=[copy.deepcopy(upper)]),
                           body=copy.deepcopy(upper),
                           orelse=copy.deepcopy(raw))
            clamp = ast.IfExp(test=ast.Compare(left=copy.deepcopy(raw), ops=[ast.Lt()], comparators=[_const(0)]),
                              body=_const(0),
                              orelse=hi)
            return [ast.Assign(targets=[_store(sv)], value=clamp)]
        if mode == "wrap":  # periodic tiling: src[q mod d]
            return [ast.Assign(targets=[_store(sv)], value=_floor_mod(_name(sv), d))]
        # symmetric/reflect: mirror with period 2d (incl. edge) or 2(d-1) (excl.
        # edge). The modulus must share the int64 index kind: a literal extent
        # folds to a literal modulus (Fortran emitter kind-coerces it), but a
        # symbolic extent needs an int local ``pv`` -- an inline compound
        # modulus with a default-kind literal would clash with the int64 index
        # under Fortran's kind-strict MODULO.
        dv = _const_int(d)
        if mode == "symmetric":  # mirror INCLUDING the edge; period 2d
            if dv is not None:
                return [
                    ast.Assign(targets=[_store(sv)], value=_floor_mod(_name(sv), _const(2 * dv))),
                    _fold_high(sv, _const(2 * dv - 1), d)
                ]
            period = ast.BinOp(left=_const(2), op=ast.Mult(), right=copy.deepcopy(d))
            return [
                ast.Assign(targets=[_store(pv)], value=period),
                ast.Assign(targets=[_store(sv)], value=_floor_mod(_name(sv), _name(pv))),
                _fold_high(sv, ast.BinOp(left=_name(pv), op=ast.Sub(), right=_const(1)), d)
            ]
        # mode == "reflect": period 2(d-1); a size-1 axis just repeats element 0.
        if dv is not None:
            if dv == 1:
                return [ast.Assign(targets=[_store(sv)], value=_const(0))]
            m = 2 * (dv - 1)
            return [
                ast.Assign(targets=[_store(sv)], value=_floor_mod(_name(sv), _const(m))),
                _fold_high(sv, _const(m), d)
            ]
        period = ast.BinOp(left=_const(2),
                           op=ast.Mult(),
                           right=ast.BinOp(left=copy.deepcopy(d), op=ast.Sub(), right=_const(1)))
        reflect_body = [
            ast.Assign(targets=[_store(pv)], value=period),
            ast.Assign(targets=[_store(sv)], value=_floor_mod(_name(sv), _name(pv))),
            _fold_high(sv, _name(pv), d)
        ]
        return [
            ast.If(test=ast.Compare(left=copy.deepcopy(d), ops=[ast.Eq()], comparators=[_const(1)]),
                   body=[ast.Assign(targets=[_store(sv)], value=_const(0))],
                   orelse=reflect_body)
        ]

    pre: List[ast.stmt] = []
    for k in range(rank):
        sv = src_idx_vars[k]
        raw = ast.BinOp(left=_name(out_iters[k]), op=ast.Sub(), right=_before(k))
        # edge folds the raw index into its clamp expression; the fold/mod modes
        # read sv back, so they need the plain seeding assign first.
        if mode != "edge":
            pre.append(ast.Assign(targets=[_store(sv)], value=raw))
        pre.extend(_remap(sv, _dim(k), f"__pm{k}", raw))
    body = pre + [
        ast.Assign(targets=[_store_target([_name(v) for v in out_iters])],
                   value=_src_read([_name(v) for v in src_idx_vars]))
    ]
    return _wrap_for_loops(out_iters, out_bounds, body)


def expand_trace(target: ast.expr, args: List[ast.expr], shape_table: Dict[str, Tuple[str, ...]]) -> List[ast.stmt]:
    """``np.trace(A)`` -> ``sum_i A[i, i]`` (the diagonal sum)."""
    if len(args) != 1 or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.trace needs one bare-Name 2-D arg")
    shape = shape_table.get(args[0].id)
    if shape is None or len(shape) != 2:
        raise NotImplementedError("np.trace needs a 2-D array")
    it = "__tr"
    diag = ast.Subscript(value=_name(args[0].id),
                         slice=ast.Tuple(elts=[_name(it), _name(it)], ctx=ast.Load()),
                         ctx=ast.Load())
    body = [ast.AugAssign(target=_store(target.id), op=ast.Add(), value=diag)]
    return [ast.Assign(targets=[_store(target.id)], value=_const(0.0)), *_wrap_for_loops([it], [shape[0]], body)]


def expand_diagonal(target: ast.expr, args: List[ast.expr], shape_table: Dict[str, Tuple[str, ...]]) -> List[ast.stmt]:
    """``np.diagonal(A)`` -> ``out[i] = A[i, i]``."""
    if len(args) != 1 or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.diagonal needs one bare-Name 2-D arg")
    shape = shape_table.get(args[0].id)
    if shape is None or len(shape) != 2:
        raise NotImplementedError("np.diagonal needs a 2-D array")
    it = "__dg"
    diag = ast.Subscript(value=_name(args[0].id),
                         slice=ast.Tuple(elts=[_name(it), _name(it)], ctx=ast.Load()),
                         ctx=ast.Load())
    body = [ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=_name(it), ctx=ast.Store())], value=diag)]
    return _wrap_for_loops([it], [shape[0]], body)


def expand_diag(target: ast.expr,
                args: List[ast.expr],
                shape_table: Dict[str, Tuple[str, ...]],
                kwargs=None) -> List[ast.stmt]:
    """``np.diag(v [, k])`` -- construct a diagonal matrix, or extract one. 1-D
    ``v`` of shape ``(n,)`` -> an ``(n+|k|, n+|k|)`` matrix, all zeros except
    ``out[i, i+k] = v[i]`` for ``k >= 0`` (``out[i-k, i] = v[i]`` for ``k < 0``),
    matching ``numpy.diag``. 2-D operand -> extract the main diagonal (delegates
    to :func:`expand_diagonal`). ``k`` must be a constant int (sizes the
    result). The matrix is zeroed first, then the diagonal written, so no
    out-of-range read occurs."""
    if not args:
        raise NotImplementedError("np.diag needs an operand")
    v = args[0]
    ext = _iter_extent_of(v, shape_table)
    if ext is None:
        raise NotImplementedError("np.diag: operand shape unknown")
    if len(ext) == 2:
        return expand_diagonal(target, args, shape_table)  # extract-diagonal
    if len(ext) != 1:
        raise NotImplementedError("np.diag: only 1-D / 2-D operands supported")
    k_node = _kwarg_or_pos(args, kwargs, 1, "k")
    if k_node is None:
        k = 0
    else:
        k = _const_int(k_node)
        if k is None:
            raise NotImplementedError("np.diag: offset k must be a constant int")
    n_tok = _CallHoister._extent_to_shape_token(ext[0])
    side_tok = n_tok if k == 0 else f"({n_tok}) + {abs(k)}"
    # Zero the whole (side x side) matrix.
    zero_body = [
        ast.Assign(targets=[
            ast.Subscript(value=_name(target.id),
                          slice=ast.Tuple(elts=[_name("__dg_zi"), _name("__dg_zj")], ctx=ast.Load()),
                          ctx=ast.Store())
        ],
                   value=_const(0.0))
    ]
    zero_loops = _wrap_for_loops(["__dg_zi", "__dg_zj"], [side_tok, side_tok], zero_body)
    # Write v along the k-th diagonal.
    it = "__dg_i"
    v_elem = _scalarize_at_iters(v, [_name(it)], shape_table)
    if k >= 0:
        row: ast.expr = _name(it)
        col: ast.expr = (_name(it) if k == 0 else ast.BinOp(left=_name(it), op=ast.Add(), right=_const(k)))
    else:
        row = ast.BinOp(left=_name(it), op=ast.Add(), right=_const(-k))
        col = _name(it)
    set_body = [
        ast.Assign(targets=[
            ast.Subscript(value=_name(target.id), slice=ast.Tuple(elts=[row, col], ctx=ast.Load()), ctx=ast.Store())
        ],
                   value=v_elem)
    ]
    return zero_loops + _wrap_for_loops([it], [n_tok], set_body)


def _scan_target_offsets(target, ndim):
    """Resolve a cumulative-scan assignment target into ``(base_name, starts)``.
    ``starts[k]`` is the lower bound to add to the operand's index along axis
    ``k`` (``None`` for a zero/omitted lower bound). A bare ``Name`` target
    scans from 0 on every axis; a partial-slice target like ``out[1:]`` shifts
    the written region by its lower bound -- DBCSR's ``row_offsets[1:] =
    np.cumsum(m_sizes)``, whose target is one element longer than the operand."""
    if isinstance(target, ast.Name):
        return target.id, [None] * ndim
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        slc = target.slice
        parts = slc.elts if isinstance(slc, ast.Tuple) else [slc]
        if len(parts) != ndim:
            raise NotImplementedError("cumulative scan: slice-target rank mismatch")
        starts = []
        for p in parts:
            if not isinstance(p, ast.Slice):
                raise NotImplementedError("cumulative scan: non-slice index in target")
            lo = p.lower
            if lo is None or (isinstance(lo, ast.Constant) and lo.value == 0):
                starts.append(None)
            else:
                starts.append(lo)
        return target.value.id, starts
    raise NotImplementedError("cumulative scan: unsupported target")


def _expand_cumulative(target, args, shape_table, op, kwargs=None, combine=None):
    """Shared prefix-scan for cumsum/cumprod/maximum.accumulate/minimum.accumulate.
    1-D (or ``axis=None`` over a 1-D operand): ``out[0] = a[0]``, then ``out[i] =
    combine(out[i-1], a[i])``. N-D with ``axis=k``: same recurrence along axis
    ``k``, other axes as outer loops. ``combine`` builds the recurrence from
    ``(prev, cur)``; ``None`` means the binary ``op`` (Add for cumsum, Mult for
    cumprod)."""
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError("cumulative scan needs a bare-Name array")
    a = args[0]
    shape = shape_table.get(a.id)
    if shape is None:
        raise NotImplementedError("cumulative scan: operand shape unknown")
    axes, _ = _read_axis_keepdims(args[1:], kwargs)
    if axes is None:
        if len(shape) != 1:
            raise NotImplementedError("cumulative scan over >1-D needs an explicit axis")
        axis = 0
    else:
        axis = axes[0] % len(shape)
    n = len(shape)
    outer = [i for i in range(n) if i != axis]
    iters = {i: f"__cs{i}" for i in range(n)}
    sc = iters[axis]

    def _idx(scan_expr):
        elts = [scan_expr if i == axis else _name(iters[i]) for i in range(n)]
        return elts[0] if n == 1 else ast.Tuple(elts=elts, ctx=ast.Load())

    target_base, t_start = _scan_target_offsets(target, n)

    def _add_off(e, off):
        return e if off is None else ast.BinOp(left=e, op=ast.Add(), right=copy.deepcopy(off))

    def _tidx(scan_expr):
        # Target index space = operand index space shifted by the slice's
        # per-axis lower bound (``out[1:] = np.cumsum(a)`` writes ``out[1+i]``).
        elts = [_add_off(scan_expr if i == axis else _name(iters[i]), t_start[i]) for i in range(n)]
        return elts[0] if n == 1 else ast.Tuple(elts=elts, ctx=ast.Load())

    out_at = lambda e: ast.Subscript(value=_name(target_base), slice=_tidx(e), ctx=ast.Load())
    a_at = lambda e: ast.Subscript(value=_name(a.id), slice=_idx(e), ctx=ast.Load())
    sc_prev = ast.BinOp(left=_name(sc), op=ast.Sub(), right=_const(1))
    # out[..start+0..] = a[..0..]
    init = ast.Assign(targets=[ast.Subscript(value=_name(target_base), slice=_tidx(_const(0)), ctx=ast.Store())],
                      value=a_at(_const(0)))
    # for sc in 1..N: out[..start+sc..] = combine(out[..start+sc-1..], a[..sc..])
    recur_val = (combine(out_at(sc_prev), a_at(_name(sc)))
                 if combine is not None else ast.BinOp(left=out_at(sc_prev), op=op, right=a_at(_name(sc))))
    recur = ast.Assign(targets=[ast.Subscript(value=_name(target_base), slice=_tidx(_name(sc)), ctx=ast.Store())],
                       value=recur_val)
    scan_loop = ast.For(target=_store(sc),
                        iter=ast.Call(func=_name("range"), args=[_const(1), _const_or_name(shape[axis])], keywords=[]),
                        body=[recur],
                        orelse=[])
    inner: List[ast.stmt] = [init, scan_loop]
    return _wrap_for_loops([iters[i] for i in outer], [shape[i] for i in outer], inner)


def expand_cumsum(target, args, shape_table, kwargs=None):
    return _expand_cumulative(target, args, shape_table, ast.Add(), kwargs)


def expand_cumprod(target, args, shape_table, kwargs=None):
    return _expand_cumulative(target, args, shape_table, ast.Mult(), kwargs)


def _running_extreme_combine(cmp):
    """``prev if (prev cmp cur) else cur`` -- the scalar running-max/min a cumulative
    ``maximum``/``minimum`` accumulate needs (no numpy ufunc survives to the backend)."""

    def combine(prev, cur):
        return ast.IfExp(test=ast.Compare(left=copy.deepcopy(prev), ops=[cmp()], comparators=[copy.deepcopy(cur)]),
                         body=copy.deepcopy(prev),
                         orelse=copy.deepcopy(cur))

    return combine


def expand_cummax(target, args, shape_table, kwargs=None):
    """``np.maximum.accumulate(a)`` -> running maximum (``out[i] = max(out[i-1], a[i])``)."""
    return _expand_cumulative(target, args, shape_table, None, kwargs, combine=_running_extreme_combine(ast.GtE))


def expand_cummin(target, args, shape_table, kwargs=None):
    """``np.minimum.accumulate(a)`` -> running minimum (``out[i] = min(out[i-1], a[i])``)."""
    return _expand_cumulative(target, args, shape_table, None, kwargs, combine=_running_extreme_combine(ast.LtE))


def _make_sort_routine(buf: str, n: ast.expr, prefix: str) -> List[ast.stmt]:
    """In-place ascending insertion sort over ``buf[0:n]`` (rendered as plain
    loops -- every backend supports them; shared by median / future
    percentile / quantile). ``prefix`` namespaces the loop / temp vars."""
    i, j, key = f"{prefix}_i", f"{prefix}_j", f"{prefix}_key"
    # key = buf[i]; j = i - 1; while j >= 0 and buf[j] > key: buf[j+1]=buf[j]; j-=1; buf[j+1]=key
    inner = [
        ast.Assign(targets=[_store(key)], value=ast.Subscript(value=_name(buf), slice=_name(i), ctx=ast.Load())),
        ast.Assign(targets=[_store(j)], value=ast.BinOp(left=_name(i), op=ast.Sub(), right=_const(1))),
        ast.While(test=ast.BoolOp(op=ast.And(),
                                  values=[
                                      ast.Compare(left=_name(j), ops=[ast.GtE()], comparators=[_const(0)]),
                                      ast.Compare(left=ast.Subscript(value=_name(buf), slice=_name(j), ctx=ast.Load()),
                                                  ops=[ast.Gt()],
                                                  comparators=[_name(key)])
                                  ]),
                  body=[
                      ast.Assign(targets=[
                          ast.Subscript(value=_name(buf),
                                        slice=ast.BinOp(left=_name(j), op=ast.Add(), right=_const(1)),
                                        ctx=ast.Store())
                      ],
                                 value=ast.Subscript(value=_name(buf), slice=_name(j), ctx=ast.Load())),
                      ast.AugAssign(target=_store(j), op=ast.Sub(), value=_const(1))
                  ],
                  orelse=[]),
        ast.Assign(targets=[
            ast.Subscript(value=_name(buf),
                          slice=ast.BinOp(left=_name(j), op=ast.Add(), right=_const(1)),
                          ctx=ast.Store())
        ],
                   value=_name(key)),
    ]
    return [
        ast.For(target=_store(i),
                iter=ast.Call(func=_name("range"), args=[_const(1), n], keywords=[]),
                body=inner,
                orelse=[])
    ]


def expand_median(target, args, shape_table, kwargs=None, local_dtypes=None, fresh_local_allocs=None) -> List[ast.stmt]:
    """``np.median(a)`` (full, flattened) -> copy + insertion-sort + pick the
    middle element (mean of the two middles for an even count).

    A scratch buffer ``__md_buf`` of the operand's total size holds the sorted
    copy so the input is not mutated."""
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.median needs a bare-Name array")
    # ONLY the full flattened median. An axis was silently ignored: the expander flattened every
    # element and returned one scalar, so ``np.median(A, axis=1)`` computed a whole-array median
    # and reported no error at all.
    if _read_axis_keepdims(args, kwargs or [])[0] is not None:
        raise NotImplementedError("np.median(axis=...) is not implemented; only the flattened median is")
    a = args[0]
    shape = shape_table.get(a.id)
    if shape is None:
        raise NotImplementedError("np.median: operand shape unknown")
    total = shape[0] if len(shape) == 1 else "(" + ") * (".join(shape) + ")"
    buf = "__md_buf"
    if fresh_local_allocs is not None:
        fresh_local_allocs[buf] = (total, )
    n_node = _const_or_name(total)
    # Flat copy a -> buf.
    cp_iters = [f"__mdc{i}" for i in range(len(shape))]
    flat = _flat_index(cp_iters, shape)
    copy_body = [
        ast.Assign(targets=[ast.Subscript(value=_name(buf), slice=flat, ctx=ast.Store())],
                   value=ast.Subscript(value=_name(a.id),
                                       slice=(_name(cp_iters[0]) if len(shape) == 1 else ast.Tuple(
                                           elts=[_name(c) for c in cp_iters], ctx=ast.Load())),
                                       ctx=ast.Load()))
    ]
    copy_loops = _wrap_for_loops(cp_iters, list(shape), copy_body)
    sort = _make_sort_routine(buf, n_node, "__md")
    half = ast.BinOp(left=copy.deepcopy(n_node), op=ast.FloorDiv(), right=_const(2))
    mid = ast.Subscript(value=_name(buf), slice=copy.deepcopy(half), ctx=ast.Load())
    mid_lo = ast.Subscript(value=_name(buf),
                           slice=ast.BinOp(left=copy.deepcopy(half), op=ast.Sub(), right=_const(1)),
                           ctx=ast.Load())
    # even count -> mean of the two middles; odd -> the single middle.
    even = ast.Compare(left=ast.BinOp(left=copy.deepcopy(n_node), op=ast.Mod(), right=_const(2)),
                       ops=[ast.Eq()],
                       comparators=[_const(0)])
    pick = ast.IfExp(test=even,
                     body=ast.BinOp(left=ast.BinOp(left=mid_lo, op=ast.Add(), right=copy.deepcopy(mid)),
                                    op=ast.Div(),
                                    right=_const(2.0)),
                     orelse=mid)
    store = ast.Assign(targets=[_store(target.id)], value=pick)
    return [*copy_loops, *sort, store]


def expand_sort(target, args, shape_table, kwargs=None) -> List[ast.stmt]:
    """``np.sort(a)`` (1-D, ascending) -> copy ``a`` into the target buffer, then
    an in-place insertion sort of that buffer. Only the 1-D form is lowered. The
    target may be an output parameter (``out[:] = np.sort(a)``) or a fresh local
    (``t = np.sort(a)``, allocated via :data:`_ELEMENT_WRITE_EXPANDERS`);
    registering ``shape_table[target]`` here makes the sorted buffer's shape
    available to that auto-alloc and any later use of ``t``."""
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.sort needs a bare-Name array")
    if not isinstance(target, ast.Name):
        raise NotImplementedError("np.sort target must be a bare Name")
    a = args[0]
    shape = shape_table.get(a.id)
    if shape is None:
        raise NotImplementedError("np.sort: operand shape unknown")
    if len(shape) != 1:
        raise NotImplementedError("np.sort: only a 1-D array is lowered")
    out = target.id
    shape_table.setdefault(out, tuple(shape))
    it = "__srtc"
    copy_body = [
        ast.Assign(targets=[ast.Subscript(value=_name(out), slice=_name(it), ctx=ast.Store())],
                   value=ast.Subscript(value=_name(a.id), slice=_name(it), ctx=ast.Load()))
    ]
    copy_loops = _wrap_for_loops([it], [shape[0]], copy_body)
    sort = _make_sort_routine(out, _const_or_name(shape[0]), "__srt")
    return [*copy_loops, *sort]


def expand_searchsorted(target, args, shape_table, kwargs=None, local_dtypes=None) -> List[ast.stmt]:
    """``np.searchsorted(a, v, side=...)`` -> a binary search per element of ``v``.

    ``a`` is a sorted 1-D array; the result is an int64 array shaped like ``v``, holding for each
    element the index where it would be inserted to keep ``a`` sorted. ``side='left'`` counts the
    entries STRICTLY below the value, ``side='right'`` counts those at or below it -- one comparison
    apart, and the difference is exactly what a bin lookup's ``- 1`` relies on.

    A binary search, not a scan: numpy's is O(log n) per element and the corpus calls this with a
    grid of tens of thousands of edges. A linear count would return the same indices and turn the
    kernel's complexity class into something the reference never had.
    """
    if len(args) < 2 or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.searchsorted needs a bare-Name sorted array and a values operand")
    if not isinstance(target, ast.Name):
        raise NotImplementedError("np.searchsorted target must be a bare Name")
    sorted_shape = shape_table.get(args[0].id)
    if sorted_shape is None or len(sorted_shape) != 1:
        raise NotImplementedError("np.searchsorted: the sorted operand must be a 1-D array of known shape")
    values_shape = _iter_extent_of(args[1], shape_table)
    if values_shape is None:
        raise NotImplementedError("np.searchsorted: shape of the values operand unknown")
    side_node = _kwarg_or_pos(args, kwargs or [], 2, "side")
    side = "left" if side_node is None else (side_node.value if isinstance(side_node, ast.Constant) else None)
    if side not in ("left", "right"):
        raise NotImplementedError("np.searchsorted: side must be the literal 'left' or 'right'")
    if local_dtypes is not None:
        local_dtypes.setdefault(target.id, "int64")
    shape_table.setdefault(target.id, tuple(ast.unparse(e) for e in values_shape))

    iters = [f"__ss{k}" for k in range(len(values_shape))]
    index = _name(iters[0]) if len(iters) == 1 else ast.Tuple(elts=[_name(v) for v in iters], ctx=ast.Load())
    value = _scalarize_at_iters(copy.deepcopy(args[1]), [_name(v) for v in iters], shape_table)
    lo, hi, mid = "__ss_lo", "__ss_hi", "__ss_mid"
    # ``a[mid] <= v`` for side='right', ``a[mid] < v`` for side='left': the first is the count of
    # entries at or below the value, the second the count strictly below it.
    below = ast.Compare(left=ast.Subscript(value=_name(args[0].id), slice=_name(mid), ctx=ast.Load()),
                        ops=[ast.LtE() if side == "right" else ast.Lt()],
                        comparators=[value])
    search = [
        ast.Assign(targets=[_store(lo)], value=_const(0)),
        ast.Assign(targets=[_store(hi)], value=_const_or_name(sorted_shape[0])),
        ast.While(test=ast.Compare(left=_name(lo), ops=[ast.Lt()], comparators=[_name(hi)]),
                  body=[
                      ast.Assign(targets=[_store(mid)],
                                 value=ast.BinOp(left=ast.BinOp(left=_name(lo), op=ast.Add(), right=_name(hi)),
                                                 op=ast.FloorDiv(),
                                                 right=_const(2))),
                      ast.If(test=below,
                             body=[
                                 ast.Assign(targets=[_store(lo)],
                                            value=ast.BinOp(left=_name(mid), op=ast.Add(), right=_const(1)))
                             ],
                             orelse=[ast.Assign(targets=[_store(hi)], value=_name(mid))]),
                  ],
                  orelse=[]),
        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=index, ctx=ast.Store())], value=_name(lo)),
    ]
    return _wrap_for_loops(iters, list(values_shape), search)


def _flat_index(iters: List[str], shape) -> ast.expr:
    """Row-major flat index ``((i0*d1 + i1)*d2 + i2)...`` for ``iters`` over
    ``shape``."""
    idx: ast.expr = _name(iters[0])
    for k in range(1, len(iters)):
        idx = ast.BinOp(left=ast.BinOp(left=idx, op=ast.Mult(), right=_const_or_name(shape[k])),
                        op=ast.Add(),
                        right=_name(iters[k]))
    return idx


def expand_roll(target, args, shape_table, kwargs=None) -> List[ast.stmt]:
    """``np.roll(a, shift, axis)`` -> ``out[i] = a[(i - shift) % n]`` along the
    rolled axis (1-D, or N-D with an explicit axis)."""
    if len(args) < 2 or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.roll needs a bare-Name array and a shift")
    a = args[0]
    shift = args[1]
    shape = shape_table.get(a.id)
    if shape is None:
        raise NotImplementedError("np.roll: operand shape unknown")
    axis_node = args[2] if len(args) > 2 else _axis_kwarg(kwargs)
    n = len(shape)
    if axis_node is None:
        if n != 1:
            raise NotImplementedError("np.roll over >1-D needs an explicit axis")
        axis = 0
    elif isinstance(axis_node, ast.Constant) and isinstance(axis_node.value, int):
        axis = axis_node.value % n
    else:
        raise NotImplementedError("np.roll axis must be a literal int")
    iters = [f"__rl{i}" for i in range(n)]
    extent = _const_or_name(shape[axis])
    # roll shifts element i to i+shift, so out[i] sources from a[i-shift]. The
    # double mod ``((i - shift) % ext + ext) % ext`` keeps the index in [0, ext)
    # for a negative shift too (C/Fortran ``%`` keeps the dividend's sign, so a
    # bare mod could go negative -> OOB read).
    src_axis = ast.BinOp(left=ast.BinOp(left=ast.BinOp(left=ast.BinOp(left=_name(iters[axis]),
                                                                      op=ast.Sub(),
                                                                      right=shift),
                                                       op=ast.Mod(),
                                                       right=copy.deepcopy(extent)),
                                        op=ast.Add(),
                                        right=copy.deepcopy(extent)),
                         op=ast.Mod(),
                         right=copy.deepcopy(extent))
    src_elts = [src_axis if i == axis else _name(iters[i]) for i in range(n)]
    dst_elts = [_name(it) for it in iters]
    src_sl = src_elts[0] if n == 1 else ast.Tuple(elts=src_elts, ctx=ast.Load())
    dst_sl = dst_elts[0] if n == 1 else ast.Tuple(elts=dst_elts, ctx=ast.Load())
    body = [
        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=dst_sl, ctx=ast.Store())],
                   value=ast.Subscript(value=_name(a.id), slice=src_sl, ctx=ast.Load()))
    ]
    return _wrap_for_loops(iters, list(shape), body)


def _axis_kwarg(kwargs):
    for kw in kwargs or []:
        if kw.arg == "axis":
            return kw.value
    return None


def expand_tril(target: ast.expr,
                args: List[ast.expr],
                shape_table: Dict[str, Tuple[str, ...]],
                kwargs=None) -> List[ast.stmt]:
    """``np.tril(A, k=0)`` -> lower-triangular copy (zero where ``j > i + k``).

    Mirrors :func:`expand_triu` with the complementary mask."""
    return _expand_triangular(target, args, shape_table, kwargs, lower=True)


def product_str(tokens: List[str]) -> str:
    """``["a", "b"]`` -> ``"(a) * (b)"``; the empty group is the unit extent."""
    return " * ".join(f"({t})" for t in tokens) if tokens else "1"


def reshape_axis_groups(
        src_shape: Tuple[str, ...], tgt_shape: Tuple[str, ...], aliases: Optional[Dict[str, str]],
        shape_table: Optional[Dict[str, Tuple[str, ...]]]) -> Optional[List[Tuple[List[int], List[int]]]]:
    """Pair the two shapes of a C-order reshape into contiguous axis groups, or None.

    Axes that agree pairwise from the front and from the back are paired 1:1; whatever is left in
    the middle is ONE group. Sound with no algebra over the extents -- a reshape keeps element
    order and the two shapes multiply out to the same total by construction, so once the matched
    axes are equal the unmatched middles span the same extent. Proving that directly means proving
    ``num_groups * (c // num_groups) == c``, which sympy will not do, and every grouped-channel
    reshape in the ML track has exactly that shape.

    A middle that is N:M with both sides above one axis is a genuine re-layout and returns None.
    """
    ns, nt = len(src_shape), len(tgt_shape)
    head = 0
    while head < ns and head < nt and dims_agree(src_shape[head], tgt_shape[head], aliases, shape_table):
        head += 1
    tail = 0
    while (tail < ns - head and tail < nt - head
           and dims_agree(src_shape[ns - 1 - tail], tgt_shape[nt - 1 - tail], aliases, shape_table)):
        tail += 1
    groups: List[Tuple[List[int], List[int]]] = [([i], [i]) for i in range(head)]
    mid_src, mid_tgt = list(range(head, ns - tail)), list(range(head, nt - tail))
    if mid_src or mid_tgt:
        if len(mid_src) > 1 and len(mid_tgt) > 1:
            return None
        groups.append((mid_src, mid_tgt))
    groups.extend(([ns - 1 - k], [nt - 1 - k]) for k in reversed(range(tail)))
    return groups


def reshape_grouped_copy(tgt_name: str, src_name: str, src_shape: Tuple[str, ...], tgt_shape: Tuple[str, ...],
                         groups: List[Tuple[List[int], List[int]]]) -> Optional[List[ast.stmt]]:
    """The copy nest for a reshape whose axes pair up (:func:`reshape_axis_groups`), or None.

    Iterates the finer side of every group, so each index on the coarser side is a linear
    combination of those iterators and every subscript stays affine. A group that splits BOTH
    sides is not expressible this way and declines.
    """
    if not src_shape or not tgt_shape:
        return None
    src_index: List[str] = [""] * len(src_shape)
    tgt_index: List[str] = [""] * len(tgt_shape)
    loops: List[Tuple[str, str]] = []
    for src_idxs, tgt_idxs in groups:
        if not src_idxs or not tgt_idxs:
            # One side spans nothing, so the other side is a run of size-1 axes indexed at 0.
            solo, index = (src_idxs, src_index) if tgt_idxs == [] else (tgt_idxs, tgt_index)
            shape = src_shape if tgt_idxs == [] else tgt_shape
            if any(str(shape[ax]) != "1" for ax in solo):
                return None
            for ax in solo:
                index[ax] = "0"
            continue
        if len(src_idxs) == len(tgt_idxs):
            for s_ax, t_ax in zip(src_idxs, tgt_idxs):
                it = f"__rs{len(loops)}"
                loops.append((it, src_shape[s_ax]))
                src_index[s_ax] = tgt_index[t_ax] = it
            continue
        fine_is_src = len(src_idxs) > len(tgt_idxs)
        fine_axes, fine_shape = (src_idxs, src_shape) if fine_is_src else (tgt_idxs, tgt_shape)
        coarse_axes = tgt_idxs if fine_is_src else src_idxs
        if len(coarse_axes) != 1:
            return None
        terms: List[str] = []
        its: List[str] = []
        for ax in fine_axes:
            it = f"__rs{len(loops)}"
            loops.append((it, fine_shape[ax]))
            its.append(it)
        for pos, (ax, it) in enumerate(zip(fine_axes, its)):
            stride = product_str([fine_shape[a] for a in fine_axes[pos + 1:]])
            terms.append(it if stride == "1" else f"({it}) * ({stride})")
            (src_index if fine_is_src else tgt_index)[ax] = it
        (tgt_index if fine_is_src else src_index)[coarse_axes[0]] = " + ".join(terms)
    if not all(src_index) or not all(tgt_index):
        return None

    def subscript(name: str, parts: List[str], ctx: ast.expr_context) -> ast.expr:
        elts = [ast.parse(t, mode="eval").body for t in parts]
        sl = elts[0] if len(elts) == 1 else ast.Tuple(elts=elts, ctx=ast.Load())
        return ast.Subscript(value=_name(name), slice=sl, ctx=ctx)

    inner: ast.stmt = ast.Assign(targets=[subscript(tgt_name, tgt_index, ast.Store())],
                                 value=subscript(src_name, src_index, ast.Load()))
    for it, bound in reversed(loops):
        inner = ast.For(target=_store(it),
                        iter=ast.Call(func=_name("range"), args=[_const_or_name(bound)], keywords=[]),
                        body=[inner],
                        orelse=[])
    return [inner]


def expand_reshape(target: ast.expr,
                   args: List[ast.expr],
                   shape_table: Dict[str, Tuple[str, ...]],
                   kwargs=None,
                   dim_aliases: Optional[Dict[str, str]] = None) -> List[ast.stmt]:
    """``out = np.reshape(A, (m, n, ...))`` -> rank-aware loop-nest copy.

    Two lowerings. When the two shapes pair up into contiguous axis groups
    (:func:`reshape_axis_groups`) the nest runs over the finer side of each group and every
    subscript is affine. Otherwise it falls back to a nest over the **target** shape whose source
    multi-index comes from div/mod on a running flat index -- correct for both C (flat-indexes
    anyway) and Fortran (type-checks rank), but non-affine over a symbolic stride.

    ``order="F"`` (numpy column-major ravel/fill) is honoured: target flat
    index and source multi-index are both computed column-major, so a
    Fortran-order reshape lowers correctly (QE vexx_k uses ``order="F"``
    throughout its FFT band-pair convolution).
    """
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.reshape needs Name first arg")
    a = args[0]
    a_shape = shape_table.get(a.id)
    if not a_shape:
        raise NotImplementedError("np.reshape: source shape unknown")
    tgt_shape = shape_table.get(target.id)
    if not tgt_shape:
        # Target shape unknown -- fall back to the legacy flat-copy form.
        # Same risk as before (Fortran rejects rank mismatch); preserved
        # only so existing kernels don't regress mid-migration.
        total = _shape_total_product(a_shape)
        body = [
            ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=_name("__r"), ctx=ast.Store())],
                       value=ast.Subscript(value=_name(a.id), slice=_name("__r"), ctx=ast.Load()))
        ]
        return [
            ast.For(target=_store("__r"),
                    iter=ast.Call(func=_name("range"), args=[total], keywords=[]),
                    body=body,
                    orelse=[])
        ]

    # Build per-axis loop iters for the target shape.
    tgt_rank = len(tgt_shape)
    src_rank = len(a_shape)
    tgt_iters = [f"__r{i}" for i in range(tgt_rank)]

    # Memory order: numpy default is C (row-major); ``order="F"`` ravels the
    # source and fills the target column-major. Flat position k of one maps to
    # flat position k of the other in the SAME order, so both the target flat
    # index and the source multi-index are computed in ``order``.
    order = "C"
    for kw in (kwargs or []):
        if vars(kw).get("arg") == "order" and isinstance(kw.value, ast.Constant):
            order = str(kw.value.value).upper()
    fortran = order == "F"

    # Affine fast path: when the two shapes pair up into contiguous axis groups, iterate the finer
    # side and read the coarser index off a linear combination of its iterators. The flat-index
    # form below is correct either way, but its ``/`` and ``%`` are non-affine over a symbolic
    # stride, which drops the nest out of Pluto's model. ``order="F"`` keeps the flat form.
    if not fortran:
        groups = reshape_axis_groups(tuple(a_shape), tuple(tgt_shape), dim_aliases, shape_table)
        if groups is not None:
            grouped = reshape_grouped_copy(target.id, a.id, tuple(a_shape), tuple(tgt_shape), groups)
            if grouped is not None:
                return grouped

    def _mul(*toks: str) -> str:
        toks = [t for t in toks if t and t != "1"]
        if not toks:
            return "1"
        if len(toks) == 1:
            return toks[0]
        return "(" + " * ".join(f"({t})" for t in toks) + ")"

    def _stride(shape, i: int) -> str:
        # Stride of axis ``i`` = product of the FASTER-varying axes: the trailing
        # axes in C order, the leading axes in F order.
        faster = list(shape[:i]) if fortran else list(shape[i + 1:])
        return _mul(*faster) if faster else "1"

    # Flat index of the current target iteration in ``order``.
    flat_parts: List[str] = []
    for i, it in enumerate(tgt_iters):
        stride = _stride(tgt_shape, i)
        # Parenthesise the STRIDE too: it is re-parsed from text, and a single compound factor came
        # back unbracketed, so ``(i) * (w - k) / 1 + 1`` re-associated as ``(i*(w-k))/1 + 1``. Only
        # the second-to-last axis of a reshape is ever a lone compound factor, which is why square
        # 2-D cases were right and every conv shape gathered 75% of its elements from the wrong slot.
        flat_parts.append(it if stride == "1" else f"({it}) * ({stride})")
    flat_expr = " + ".join(flat_parts) if flat_parts else "0"

    # Decode the source multi-index from the flat index via div/mod on the source
    # strides (same ``order``). The MOST-major axis (largest stride: ``i == 0`` in
    # C, ``i == src_rank - 1`` in F) needs no modulo.
    src_axes: List[ast.expr] = []
    for i in range(src_rank):
        # A size-1 source axis indexes to a constant 0, avoiding a degenerate
        # ``flat % 1``/``flat / 1`` whose bare ``1`` literal clashes with the
        # int64 flat index under Fortran ``-std=f2018`` (GNU "Different type kinds").
        if str(a_shape[i]) == "1":
            src_axes.append(ast.Constant(value=0))
            continue
        stride = _stride(a_shape, i)
        ax_expr = flat_expr if stride == "1" else f"(({flat_expr}) / ({stride}))"
        is_major = (i == src_rank - 1) if fortran else (i == 0)
        if not is_major:
            ax_expr = f"(({ax_expr}) % ({a_shape[i]}))"
        src_axes.append(ast.parse(ax_expr, mode="eval").body)

    # ``out[t0, t1, ...] = A[<computed-axes>]``.
    if tgt_rank == 1:
        lhs_slice = _name(tgt_iters[0])
    else:
        lhs_slice = ast.Tuple(elts=[_name(it) for it in tgt_iters], ctx=ast.Load())
    if src_rank == 1:
        rhs_slice = src_axes[0]
    else:
        rhs_slice = ast.Tuple(elts=src_axes, ctx=ast.Load())
    inner = ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=lhs_slice, ctx=ast.Store())],
                       value=ast.Subscript(value=_name(a.id), slice=rhs_slice, ctx=ast.Load()))

    # Wrap in target-shape loop nest (outermost first).
    current: ast.stmt = inner
    for it, bound in zip(reversed(tgt_iters), reversed(list(tgt_shape))):
        current = ast.For(target=_store(it),
                          iter=ast.Call(func=_name("range"), args=[_const_or_name(bound)], keywords=[]),
                          body=[current],
                          orelse=[])
    return [current]


def _diff_operand(expr: ast.expr) -> Optional[ast.Name]:
    """``np.diff(p)`` -> ``p``, else ``None``.

    The one per-element ``np.repeat`` count form whose SUM telescopes without
    touching data: ``sum(np.diff(p)) == p[-1] - p[0]``. ``n``/``axis`` kwargs
    change which sum telescopes (or whether it does at all), so only the bare
    single-arg call is recognised.
    """
    if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute)
            and isinstance(expr.func.value, ast.Name) and expr.func.value.id == "np" and expr.func.attr == "diff"):
        return None
    if len(expr.args) != 1 or not isinstance(expr.args[0], ast.Name) or expr.keywords:
        return None
    return expr.args[0]


def _expand_repeat_prefix_sum(target: ast.expr, a: ast.Name, a_shape: Tuple[str, ...], k_arg: ast.expr,
                              shape_table: Dict[str, Tuple[str,
                                                           ...]], local_dtypes, fresh_local_allocs) -> List[ast.stmt]:
    """Per-element ``np.repeat`` count -- the destination offset is the
    RUNNING prefix sum of the counts, not ``outer * K`` (that formula reads
    the count array as a scalar multiplier and is wrong the moment two counts
    differ; it also silently skips a zero count instead of writing nothing)::

        pos = 0
        for i in range(<source extent>):
            for r in range(counts[i]):
                out[pos] = src[i]
                pos += 1

    Only a 1-D source is supported -- there is no per-axis running offset
    here for the axis-aware case. ``counts`` must be ``np.diff(p)`` so its
    sum (the result extent) telescopes to ``p[-1] - p[0]``; any other
    per-element form has a data-dependent sum with no static extent and is
    refused rather than guessed (an under-sized buffer is a heap overflow).
    ``np.diff`` is never materialised: ``counts[i]`` is expressed inline as
    ``p[i + 1] - p[i]``, so no auxiliary array is allocated at all.
    """
    if len(a_shape) != 1:
        raise NotImplementedError("np.repeat with a per-element count needs a 1-D source")
    diff_src = _diff_operand(k_arg)
    if diff_src is None:
        raise NotImplementedError("np.repeat with a per-element count needs a derivable sum "
                                  f"(got {ast.unparse(k_arg)}); only np.diff(p) telescopes to p[-1] - p[0]")
    p_shape = shape_table.get(diff_src.id)
    if not p_shape or len(p_shape) != 1:
        raise NotImplementedError("np.repeat: np.diff operand shape unknown")
    last_idx = ast.BinOp(left=_const_or_name(p_shape[0]), op=ast.Sub(), right=_const(1))
    total = ast.BinOp(left=ast.Subscript(value=copy.deepcopy(diff_src), slice=last_idx, ctx=ast.Load()),
                      op=ast.Sub(),
                      right=ast.Subscript(value=copy.deepcopy(diff_src), slice=_const(0), ctx=ast.Load()))
    extent_tok = ast.unparse(total)
    shape_table[target.id] = (extent_tok, )
    if fresh_local_allocs is not None:
        fresh_local_allocs[target.id] = (extent_tok, )
    pos, src_iter, cnt_iter = "__rep_pos0", "__rep_i0", "__rep_r0"
    if local_dtypes is not None:
        local_dtypes[pos] = "int64"
    count_at_i = ast.BinOp(left=ast.Subscript(value=copy.deepcopy(diff_src),
                                              slice=ast.BinOp(left=_name(src_iter), op=ast.Add(), right=_const(1)),
                                              ctx=ast.Load()),
                           op=ast.Sub(),
                           right=ast.Subscript(value=copy.deepcopy(diff_src), slice=_name(src_iter), ctx=ast.Load()))
    body = [
        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=_name(pos), ctx=ast.Store())],
                   value=ast.Subscript(value=copy.deepcopy(a), slice=_name(src_iter), ctx=ast.Load())),
        ast.AugAssign(target=_store(pos), op=ast.Add(), value=_const(1)),
    ]
    inner_loop = ast.For(target=_store(cnt_iter),
                         iter=ast.Call(func=_name("range"), args=[count_at_i], keywords=[]),
                         body=body,
                         orelse=[])
    outer_loop = ast.For(target=_store(src_iter),
                         iter=ast.Call(func=_name("range"), args=[_const_or_name(a_shape[0])], keywords=[]),
                         body=[inner_loop],
                         orelse=[])
    init_pos = ast.Assign(targets=[_store(pos)], value=_const(0))
    return [_alloc_marker(target.id), init_pos, outer_loop]


def expand_repeat(target: ast.expr,
                  args: List[ast.expr],
                  shape_table: Dict[str, Tuple[str, ...]],
                  kwargs=None,
                  local_dtypes=None,
                  fresh_local_allocs=None) -> List[ast.stmt]:
    """``out = np.repeat(A, K, axis=N)`` -> tile-and-write loop nest. Source
    ``A`` of shape ``(s0, ..., sN, ..., sM-1)`` becomes ``out`` of shape
    ``(s0, ..., sN*K, ..., sM-1)``. Per-element::

        out[i0, ..., iN_outer * K + iN_inner, ..., iM-1]
            = A[i0, ..., iN_outer, ..., iM-1]

    The broadcast-from-size-1 case (``sN == 1``, stockham_fft) still applies
    the formula unchanged: ``iN_outer`` only ranges over 0, so ``A`` reads
    ``[..., 0, ...]`` regardless of the inner index.

    ``axis`` may be int (positional or kwarg) or None (flat-axis repeat: result
    is a flat 1-D array of size ``prod(A.shape) * K``).
    """
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.repeat needs Name first arg")
    a = args[0]
    a_shape = shape_table.get(a.id)
    if not a_shape:
        raise NotImplementedError("np.repeat: source shape unknown")
    # ``K`` -- repetitions.
    if len(args) < 2:
        raise NotImplementedError("np.repeat needs repetitions arg")
    k_arg = args[1]
    # A PER-ELEMENT repeat count (``np.repeat(np.arange(M), np.diff(A_indptr))``) is a different
    # lowering: the destination offset is a prefix sum of the counts, not ``outer * K`` (that
    # formula reads the count array as a scalar multiplier and computes the wrong offsets).
    if _iter_extent_of(k_arg, shape_table) is not None:
        return _expand_repeat_prefix_sum(target, a, a_shape, k_arg, shape_table, local_dtypes, fresh_local_allocs)
    # ``axis`` -- positional [2] or kwarg. An ABSENT axis means numpy's flat repeat; an axis that is
    # merely unreadable must NOT fall into that branch, because flat repeat is a different output
    # shape and a different loop nest, not a degraded version of the same one.
    axis_node = _kwarg_or_pos(args, kwargs, 2, "axis")
    axis: Optional[int] = None
    if not (axis_node is None or (isinstance(axis_node, ast.Constant) and axis_node.value is None)):
        axis = _axis_literal_or_refuse(axis_node, "np.repeat")
    n_dim = len(a_shape)
    if axis is None:
        # Flat repeat: each scalar element repeated K times.
        # out[flat * K + r] = A[flat]
        iters = [_make_iter_name("__rp", i) for i in range(n_dim)]
        rep_iter = _make_iter_name("__rep", 0)
        # source subscript
        src_slot = (_name(iters[0]) if n_dim == 1 else ast.Tuple(elts=[_name(i) for i in iters], ctx=ast.Load()))
        # destination flat index = ((((i0)*s1 + i1)*s2 + ...) * K + r)
        flat_index: ast.expr = _name(iters[0])
        for k in range(1, n_dim):
            flat_index = ast.BinOp(left=ast.BinOp(left=flat_index, op=ast.Mult(), right=_const_or_name(a_shape[k])),
                                   op=ast.Add(),
                                   right=_name(iters[k]))
        dst_index = ast.BinOp(left=ast.BinOp(left=flat_index, op=ast.Mult(), right=k_arg),
                              op=ast.Add(),
                              right=_name(rep_iter))
        body = [
            ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=dst_index, ctx=ast.Store())],
                       value=ast.Subscript(value=_name(a.id), slice=src_slot, ctx=ast.Load()))
        ]
        # Wrap with the source loops and the repetition loop deepest.
        out = body
        out = [
            ast.For(target=_store(rep_iter),
                    iter=ast.Call(func=_name("range"), args=[k_arg], keywords=[]),
                    body=out,
                    orelse=[])
        ]
        for var, bound in zip(reversed(iters), reversed(a_shape)):
            out = [
                ast.For(target=_store(var),
                        iter=ast.Call(func=_name("range"), args=[_const_or_name(bound)], keywords=[]),
                        body=out,
                        orelse=[])
            ]
        return out
    # Axis-aware repeat: walk every axis; for axis ``N`` the dest
    # index is ``outer_N * K + inner_N`` while source still reads at
    # ``outer_N``.
    if axis < 0:
        axis += n_dim
    if axis < 0 or axis >= n_dim:
        raise NotImplementedError(f"np.repeat axis {axis} out of range for ndim {n_dim}")
    iters = [_make_iter_name("__rp", i) for i in range(n_dim)]
    rep_iter = _make_iter_name("__rep", 0)
    src_elts = [_name(iters[i]) for i in range(n_dim)]
    dst_elts: List[ast.expr] = []
    for i in range(n_dim):
        if i == axis:
            dst_elts.append(
                ast.BinOp(left=ast.BinOp(left=_name(iters[i]), op=ast.Mult(), right=k_arg),
                          op=ast.Add(),
                          right=_name(rep_iter)))
        else:
            dst_elts.append(_name(iters[i]))
    src_slot = (src_elts[0] if n_dim == 1 else ast.Tuple(elts=src_elts, ctx=ast.Load()))
    dst_slot = (dst_elts[0] if n_dim == 1 else ast.Tuple(elts=dst_elts, ctx=ast.Load()))
    body = [
        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=dst_slot, ctx=ast.Store())],
                   value=ast.Subscript(value=_name(a.id), slice=src_slot, ctx=ast.Load()))
    ]
    # Innermost = repetition loop.
    out = body
    out = [
        ast.For(target=_store(rep_iter),
                iter=ast.Call(func=_name("range"), args=[k_arg], keywords=[]),
                body=out,
                orelse=[])
    ]
    for var, bound in zip(reversed(iters), reversed(a_shape)):
        out = [
            ast.For(target=_store(var),
                    iter=ast.Call(func=_name("range"), args=[_const_or_name(bound)], keywords=[]),
                    body=out,
                    orelse=[])
        ]
    return out


def _classify_norm_ord(node: Optional[ast.expr]) -> Optional[str]:
    """Classify a ``np.linalg.norm`` ``ord`` argument.

    ``"l2"`` for the default (``None`` / 2 -> Euclidean vector norm or
    Frobenius matrix norm), ``"l1"`` for ``ord=1``, ``"inf"`` for ``np.inf``
    / ``math.inf`` / ``float("inf")``. Anything else (3, ``'nuc'``, ``'fro'``
    spelled out, ``-inf``, the matrix spectral 2-norm) -> ``None``, so the
    caller raises rather than emit a silently-wrong norm.
    """
    if node is None:
        return "l2"
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return None
        if v is None or v == 2:
            return "l2"
        if v == 1:
            return "l1"
        if isinstance(v, float) and v == float("inf"):
            return "inf"
        return None
    if isinstance(node, ast.Attribute) and node.attr in ("inf", "Inf", "PINF"):
        return "inf"
    # ``np.inf`` is rewritten to the bare Name ``INFINITY`` (the lowered
    # numeric-constant token, see lowering.py) before this expander runs.
    if isinstance(node, ast.Name) and node.id in ("INFINITY", "inf", "Inf"):
        return "inf"
    return None


def expand_linalg_norm(target: ast.expr,
                       args: List[ast.expr],
                       shape_table: Dict[str, Tuple[str, ...]],
                       kwargs=None) -> List[ast.stmt]:
    """``s = np.linalg.norm(v[, ord, axis=None, keepdims=False])``. numpy puts
    ``ord`` SECOND (positional) -- unlike a reduction, whose second positional
    is ``axis`` -- so it's parsed explicitly before delegating axis/keepdims
    handling (else a positional ``ord`` is misread as ``axis``: ``ord=1``
    spuriously raises, ``ord=np.inf`` silently returns the L2 norm). Supported:
    ``ord`` in {None, 2} is L2/Frobenius via squared-sum + sqrt, full or
    axis-aware; ``ord`` in {1, inf} for a 1-D operand is ``sum(|v|)``/``max(|v|)``.
    A matrix 1/inf norm, an ord+axis combination, or any other ``ord`` raises
    ``NotImplementedError`` -- never a silent wrong norm.
    """
    if not args:
        raise NotImplementedError("np.linalg.norm needs an operand")
    kwargs = list(kwargs or [])
    a = args[0]
    ord_node: Optional[ast.expr] = args[1] if len(args) >= 2 else None
    for kw in kwargs:
        if kw.arg == "ord":
            ord_node = kw.value
    kind = _classify_norm_ord(ord_node)
    if kind is None:
        raise NotImplementedError("np.linalg.norm: unsupported ord (only None/2, 1, inf)")
    # Strip ``ord`` (positional arg[1] or keyword) so the shared axis reader
    # sees the reduction layout (operand, axis, keepdims).
    reduction_args = [a] + list(args[2:])
    reduction_kwargs = [kw for kw in kwargs if kw.arg != "ord"]
    axes, keepdims = _read_axis_keepdims(reduction_args, reduction_kwargs)

    if kind == "l2":
        if axes is None:
            # Full reduction -- scalar accumulator + sqrt.
            extent = _iter_extent_of(a, shape_table)
            if extent is None:
                raise NotImplementedError("np.linalg.norm: cannot derive iteration extent")
            iters = [_make_iter_name("__nr", i) for i in range(len(extent))]
            sa = _scalarize_at_iters(a, [_name(it) for it in iters], shape_table)
            acc_init = ast.Assign(targets=[_store(target.id)], value=_const(0.0))
            inner = [
                ast.AugAssign(target=_store(target.id), op=ast.Add(), value=ast.BinOp(left=sa, op=ast.Mult(), right=sa))
            ]
            loops = _wrap_for_loops(iters, extent, inner)
            finish = ast.Assign(targets=[_store(target.id)],
                                value=ast.Call(func=_name("sqrt"), args=[_name(target.id)], keywords=[]))
            return [acc_init, *loops, finish]
        # Axis-aware -- reduce per axis kept, sum-of-squares, then sqrt.
        sq_op = lambda acc, x: ast.BinOp(left=acc, op=ast.Add(), right=ast.BinOp(left=x, op=ast.Mult(), right=x))
        sqrt_post = lambda lvalue, divisor: ast.Assign(targets=[
            lvalue
            if isinstance(lvalue, ast.Name) else ast.Subscript(value=lvalue.value, slice=lvalue.slice, ctx=ast.Store())
        ],
                                                       value=ast.Call(func=_name("sqrt"),
                                                                      args=[
                                                                          lvalue if isinstance(lvalue, ast.Name) else
                                                                          ast.Subscript(value=lvalue.value,
                                                                                        slice=lvalue.slice,
                                                                                        ctx=ast.Load())
                                                                      ],
                                                                      keywords=[]))
        return _expand_axis_reduction(target,
                                      reduction_args,
                                      reduction_kwargs,
                                      shape_table,
                                      init=_const(0.0),
                                      op_fn=sq_op,
                                      post_fn=sqrt_post)

    # ``ord`` in {1, inf}. A 1-D operand is a vector norm (sum|v| / max|v|); a
    # 2-D operand is a matrix norm (ord=1 = max column abs-sum, ord=inf = max row
    # abs-sum) -- the max over per-line abs-sums.
    if axes is not None:
        raise NotImplementedError("np.linalg.norm: ord=1/inf with axis= not supported")
    extent = _iter_extent_of(a, shape_table)
    if extent is None:
        raise NotImplementedError("np.linalg.norm: cannot derive iteration extent")
    if len(extent) == 1:
        it = _make_iter_name("__nr", 0)
        sa = _scalarize_at_iters(a, [_name(it)], shape_table)
        abs_sa = ast.Call(func=_name("abs"), args=[sa], keywords=[])
        acc_init = ast.Assign(targets=[_store(target.id)], value=_const(0.0))
        if kind == "l1":
            inner = [ast.AugAssign(target=_store(target.id), op=ast.Add(), value=abs_sa)]
        else:  # inf: running max of |v| (|v| >= 0, so 0 is a safe max identity)
            inner = [
                ast.Assign(targets=[_store(target.id)],
                           value=ast.IfExp(test=ast.Compare(left=copy.deepcopy(abs_sa),
                                                            ops=[ast.Gt()],
                                                            comparators=[_name(target.id)]),
                                           body=abs_sa,
                                           orelse=_name(target.id)))
            ]
        loops = _wrap_for_loops([it], extent, inner)
        return [acc_init, *loops]
    if len(extent) == 2 and isinstance(a, ast.Name):
        # Matrix ord=1 / ord=inf: accumulate each line's abs-sum into a scalar
        # ``__nmc`` then keep the running max. ord=1 sums down columns (outer j),
        # ord=inf sums across rows (outer i); the element a[i, j] is the same.
        m_ext, n_ext = extent
        i_it, j_it, csum = "__nmi", "__nmj", "__nmc"
        elem = ast.Call(func=_name("abs"),
                        args=[
                            ast.Subscript(value=_name(a.id),
                                          slice=ast.Tuple(elts=[_name(i_it), _name(j_it)], ctx=ast.Load()),
                                          ctx=ast.Load())
                        ],
                        keywords=[])
        if kind == "l1":  # outer over columns j, inner over rows i
            outer_it, outer_bound, inner_it, inner_bound = j_it, n_ext, i_it, m_ext
        else:  # inf: outer over rows i, inner over columns j
            outer_it, outer_bound, inner_it, inner_bound = i_it, m_ext, j_it, n_ext
        inner_loop = ast.For(target=_store(inner_it),
                             iter=ast.Call(func=_name("range"), args=[copy.deepcopy(inner_bound)], keywords=[]),
                             body=[ast.AugAssign(target=_store(csum), op=ast.Add(), value=elem)],
                             orelse=[])
        keep_max = ast.Assign(targets=[_store(target.id)],
                              value=ast.IfExp(test=ast.Compare(left=_name(csum),
                                                               ops=[ast.Gt()],
                                                               comparators=[_name(target.id)]),
                                              body=_name(csum),
                                              orelse=_name(target.id)))
        outer_loop = ast.For(target=_store(outer_it),
                             iter=ast.Call(func=_name("range"), args=[copy.deepcopy(outer_bound)], keywords=[]),
                             body=[ast.Assign(targets=[_store(csum)], value=_const(0.0)), inner_loop, keep_max],
                             orelse=[])
        return [ast.Assign(targets=[_store(target.id)], value=_const(0.0)), outer_loop]
    raise NotImplementedError("np.linalg.norm: ord=1/inf supported for a 1-D or 2-D operand")


def _guarded_div(num: ast.expr, denom: ast.expr) -> ast.expr:
    """``denom != 0 ? num / denom : 0`` -- guards the naive Gaussian-elimination
    solve against a zero pivot. A rank-deficient system (e.g. GMRES after an
    early break leaves a singular ``H``) makes a diagonal pivot exactly 0;
    numpy's SVD-based ``lstsq`` still returns a finite minimum-norm solution,
    but the unguarded division would emit NaN/inf. Inert for a full-rank system."""
    return ast.IfExp(test=ast.Compare(left=copy.deepcopy(denom), ops=[ast.NotEq()], comparators=[_const(0.0)]),
                     body=ast.BinOp(left=num, op=ast.Div(), right=denom),
                     orelse=_const(0.0))


def expand_lstsq(target: ast.expr,
                 args: List[ast.expr],
                 shape_table: Dict[str, Tuple[str, ...]],
                 kwargs=None,
                 fresh_local_allocs=None) -> List[ast.stmt]:
    """``y = np.linalg.lstsq(A, b, rcond=...)[0]`` -> in-place Gaussian
    elimination with partial pivoting, writing the solution into ``target``.

    Conservative scope: A must be a SQUARE M x M region, either a bare Name
    (shape (M, M)) or a Subscript like ``H[:m, :]`` whose first slice has a
    ``stop`` (m); b matches as a Name of shape (M,) or a Subscript ``e1[:m]``.
    M at codegen time is the slice's ``stop`` symbol or the array's first dim.
    A and b are mutated in place -- unlike numpy's side-effect-free ``lstsq``,
    but acceptable since the only caller in scope (gmres) uses the result once
    and never re-reads H or b after.

    Heuristic: detects the ``np.linalg.lstsq(...)[0]`` form via the caller
    (LibNodeRewriter handles the Subscript unwrap); takes only the positional
    args, ``rcond`` in ``kwargs`` is consumed and ignored.
    """
    if len(args) < 2:
        raise NotImplementedError("np.linalg.lstsq needs A and b args")
    a_node, b_node = args[0], args[1]
    a_size = _lstsq_first_axis_size(a_node, shape_table)
    if a_size is None:
        raise NotImplementedError("np.linalg.lstsq: cannot infer A's leading dimension")
    a_name, a_base = _lstsq_array_base(a_node)
    if a_name is None:
        raise NotImplementedError("np.linalg.lstsq: A must be a Name or simple slice")
    # ``b`` may be an expression (gmres passes ``beta * e1[:m]``); Gaussian
    # elimination needs an indexable, MUTABLE b, so materialize it into a fresh
    # length-M temp vector via a fill loop first. A bare Name/simple slice is
    # used in place.
    pre: List[ast.stmt] = []
    b_name, b_base = _lstsq_array_base(b_node)
    if b_name is None:
        b_name, b_base = "__lq_b", None
        _bi = "__lq_bi"
        # Allocation marker first (mirrors expand_solve/expand_inv): length-M
        # ``__lq_b``, whose M depends on a body-computed scalar (gmres ``m =
        # min(max_iter, n)``), is deferred-malloc'd here. Without it the NULL
        # pointer is dereferenced in the fill loop below.
        pre.append(
            ast.Assign(targets=[_store(b_name)],
                       value=ast.Call(func=_name("__hpcagent_bench_zeros__"), args=[], keywords=[])))
        _elem = _scalarize_at_iters(b_node, [_name(_bi)], shape_table)
        pre.append(
            ast.For(target=_store(_bi),
                    iter=ast.Call(func=_name("range"), args=[a_size], keywords=[]),
                    body=[
                        ast.Assign(targets=[ast.Subscript(value=_name(b_name), slice=_name(_bi), ctx=ast.Store())],
                                   value=_elem)
                    ],
                    orelse=[]))
        if fresh_local_allocs is not None:
            fresh_local_allocs[b_name] = (ast.unparse(a_size), )
    # The solution vector ``target`` is written element-wise by back
    # substitution; register its shape so the caller allocates it.
    if isinstance(target, ast.Name):
        shape_table[target.id] = (ast.unparse(a_size), )
    # M = A.shape[0]
    p_iter = "__lq_p"
    r_iter = "__lq_r"
    c_iter = "__lq_c"
    factor = "__lq_factor"
    sum_v = "__lq_sum"
    p_name = _name(p_iter)
    r_name = _name(r_iter)
    c_name = _name(c_iter)
    a_pp = _lstsq_index2d(a_name, p_name, p_name, a_base)
    a_rp = _lstsq_index2d(a_name, r_name, p_name, a_base)
    a_pc = _lstsq_index2d(a_name, p_name, c_name, a_base)
    a_rcol = _lstsq_index2d(a_name, r_name, c_name, a_base)
    a_rr = _lstsq_index2d(a_name, r_name, r_name, a_base)
    b_p = _lstsq_index1d(b_name, p_name, b_base)
    b_r = _lstsq_index1d(b_name, r_name, b_base)
    # Forward elimination over pivot p:
    #   for p in 0..M:
    #     for r in p+1..M:
    #       factor = A[r,p] / A[p,p]
    #       for c in p+1..M: A[r,c] -= factor * A[p,c]
    #       b[r] -= factor * b[p]
    inner_c = [
        ast.AugAssign(target=ast.Subscript(value=_name(a_name),
                                           slice=ast.Tuple(elts=[r_name, c_name], ctx=ast.Load()),
                                           ctx=ast.Store()),
                      op=ast.Sub(),
                      value=ast.BinOp(left=_name(factor), op=ast.Mult(), right=a_pc))
    ]
    inner_c_for = ast.For(target=_store(c_iter),
                          iter=ast.Call(func=_name("range"),
                                        args=[ast.BinOp(left=p_name, op=ast.Add(), right=_const(1)), a_size],
                                        keywords=[]),
                          body=inner_c,
                          orelse=[])
    factor_assign = ast.Assign(targets=[_store(factor)], value=_guarded_div(a_rp, a_pp))
    b_aug = ast.AugAssign(target=ast.Subscript(value=_name(b_name), slice=r_name, ctx=ast.Store()),
                          op=ast.Sub(),
                          value=ast.BinOp(left=_name(factor), op=ast.Mult(), right=b_p))
    inner_r = ast.For(target=_store(r_iter),
                      iter=ast.Call(func=_name("range"),
                                    args=[ast.BinOp(left=p_name, op=ast.Add(), right=_const(1)), a_size],
                                    keywords=[]),
                      body=[factor_assign, inner_c_for, b_aug],
                      orelse=[])
    fwd = ast.For(target=_store(p_iter),
                  iter=ast.Call(func=_name("range"), args=[a_size], keywords=[]),
                  body=[inner_r],
                  orelse=[])
    # Back substitution:
    #   for r in M-1..0 (reverse):
    #     sum = b[r]
    #     for c in r+1..M: sum -= A[r,c] * y[c]
    #     y[r] = sum / A[r,r]
    y_c = ast.Subscript(value=_name(target.id), slice=c_name, ctx=ast.Load())
    y_r = ast.Subscript(value=_name(target.id), slice=r_name, ctx=ast.Store())
    bs_inner = [
        ast.AugAssign(target=_store(sum_v), op=ast.Sub(), value=ast.BinOp(left=a_rcol, op=ast.Mult(), right=y_c))
    ]
    bs_inner_for = ast.For(target=_store(c_iter),
                           iter=ast.Call(func=_name("range"),
                                         args=[ast.BinOp(left=r_name, op=ast.Add(), right=_const(1)), a_size],
                                         keywords=[]),
                           body=bs_inner,
                           orelse=[])
    bs_sum_init = ast.Assign(targets=[_store(sum_v)], value=b_r)
    bs_y_assign = ast.Assign(targets=[y_r], value=_guarded_div(_name(sum_v), a_rr))
    # Reverse iteration via ``range(M-1, -1, -1)``.
    bs = ast.For(target=_store(r_iter),
                 iter=ast.Call(func=_name("range"),
                               args=[ast.BinOp(left=a_size, op=ast.Sub(), right=_const(1)),
                                     _const(-1),
                                     _const(-1)],
                               keywords=[]),
                 body=[bs_sum_init, bs_inner_for, bs_y_assign],
                 orelse=[])
    return pre + [fwd, bs]


def _lstsq_first_axis_size(node: ast.expr, shape_table: Dict[str, Tuple[str, ...]]):
    """First-axis size of ``node`` as an AST expression: bare Name -> shape_table
    lookup; Subscript ``H[:m, ...]`` -> the explicit ``m`` stop."""
    if isinstance(node, ast.Name):
        shape = shape_table.get(node.id)
        if shape:
            return _const_or_name(shape[0])
        return None
    if isinstance(node, ast.Subscript):
        sl = node.slice
        first = sl.elts[0] if isinstance(sl, ast.Tuple) else sl
        if isinstance(first, ast.Slice) and first.upper is not None:
            return first.upper
        if isinstance(first, ast.Slice) and first.upper is None:
            # Whole axis -- fall back to shape_table of the base.
            if isinstance(node.value, ast.Name):
                shape = shape_table.get(node.value.id)
                if shape:
                    return _const_or_name(shape[0])
        if not isinstance(first, ast.Slice) and isinstance(node.value, ast.Name):
            shape = shape_table.get(node.value.id)
            if shape:
                return _const_or_name(shape[0])
    return None


def _lstsq_array_base(node: ast.expr):
    """Return ``(name, base_offsets)`` for a Name or simple slice subscript.
    ``base_offsets`` is the lower-bound shift per axis (list of ast.expr) so the
    expander can rewrite ``A[i, j]`` as ``Name[i + base0, j + base1]``. Zero for
    a Name or ``H[:m, :]`` (lower=None defaults to 0); captured for ``H[2:m,
    :]`` (non-zero lower)."""
    if isinstance(node, ast.Name):
        return node.id, None
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        sl = node.slice
        slots = sl.elts if isinstance(sl, ast.Tuple) else [sl]
        bases = []
        for s in slots:
            if isinstance(s, ast.Slice):
                bases.append(s.lower if s.lower is not None else _const(0))
            else:
                bases.append(None)  # concrete index -- not a slice axis
        return node.value.id, bases
    return None, None


def _lstsq_index2d(name: str, i: ast.expr, j: ast.expr, base) -> ast.Subscript:
    """Build ``name[i, j]`` (or ``name[i + base0, j + base1]`` when
    a non-zero base is present)."""
    if base is not None:
        slot_i = (i if (isinstance(base[0], ast.Constant) and base[0].value == 0) else ast.BinOp(
            left=i, op=ast.Add(), right=base[0]))
        slot_j = (j if (isinstance(base[1], ast.Constant) and base[1].value == 0) else ast.BinOp(
            left=j, op=ast.Add(), right=base[1]))
    else:
        slot_i, slot_j = i, j
    return ast.Subscript(value=_name(name), slice=ast.Tuple(elts=[slot_i, slot_j], ctx=ast.Load()), ctx=ast.Load())


def _lstsq_index1d(name: str, i: ast.expr, base) -> ast.Subscript:
    if base is not None and not (isinstance(base[0], ast.Constant) and base[0].value == 0):
        slot = ast.BinOp(left=i, op=ast.Add(), right=base[0])
    else:
        slot = i
    return ast.Subscript(value=_name(name), slice=slot, ctx=ast.Load())


def expand_cholesky(target: ast.expr,
                    args: List[ast.expr],
                    shape_table: Dict[str, Tuple[str, ...]],
                    local_dtypes: Optional[Dict[str, str]] = None) -> List[ast.stmt]:
    """``L = np.linalg.cholesky(A)`` -> Cholesky-Banachiewicz triple loop.
    Computes ``L`` such that ``L @ L.T == A`` for symmetric positive-definite
    ``A``. Naive O(n^3), no blocking::

        for j in range(n):
            s = A[j, j]
            for k in range(j):
                s -= L[j, k] * L[j, k]
            L[j, j] = sqrt(s)
            for i in range(j + 1, n):
                s = A[i, j]
                for k in range(j):
                    s -= L[i, k] * L[j, k]
                L[i, j] = s / L[j, j]
    """
    if not args_one_name(args):
        raise NotImplementedError("np.linalg.cholesky needs Name arg")
    a = args[0]
    a_shape = shape_table.get(a.id)
    if not a_shape or len(a_shape) != 2:
        raise NotImplementedError("cholesky: only 2-D arg")
    if local_dtypes is not None:
        a_dt = local_dtypes.get(a.id)
        if a_dt:
            local_dtypes[target.id] = a_dt
            local_dtypes.setdefault("__s", a_dt)
    n = a_shape[0]
    n_ast = _const_or_name(n)
    inner_k = [
        ast.AugAssign(target=_store("__s"),
                      op=ast.Sub(),
                      value=ast.BinOp(left=ast.Subscript(value=_name(target.id),
                                                         slice=ast.Tuple(elts=[_name("__j"), _name("__k")],
                                                                         ctx=ast.Load()),
                                                         ctx=ast.Load()),
                                      op=ast.Mult(),
                                      right=ast.Subscript(value=_name(target.id),
                                                          slice=ast.Tuple(elts=[_name("__j"),
                                                                                _name("__k")],
                                                                          ctx=ast.Load()),
                                                          ctx=ast.Load()))),
    ]
    inner_i = [
        ast.Assign(targets=[_store("__s")],
                   value=ast.Subscript(value=_name(a.id),
                                       slice=ast.Tuple(elts=[_name("__i"), _name("__j")], ctx=ast.Load()),
                                       ctx=ast.Load())),
        ast.For(target=_store("__k"),
                iter=ast.Call(func=_name("range"), args=[_name("__j")], keywords=[]),
                body=[
                    ast.AugAssign(target=_store("__s"),
                                  op=ast.Sub(),
                                  value=ast.BinOp(left=ast.Subscript(value=_name(target.id),
                                                                     slice=ast.Tuple(elts=[_name("__i"),
                                                                                           _name("__k")],
                                                                                     ctx=ast.Load()),
                                                                     ctx=ast.Load()),
                                                  op=ast.Mult(),
                                                  right=ast.Subscript(value=_name(target.id),
                                                                      slice=ast.Tuple(elts=[_name("__j"),
                                                                                            _name("__k")],
                                                                                      ctx=ast.Load()),
                                                                      ctx=ast.Load())))
                ],
                orelse=[]),
        ast.Assign(targets=[
            ast.Subscript(value=_name(target.id),
                          slice=ast.Tuple(elts=[_name("__i"), _name("__j")], ctx=ast.Load()),
                          ctx=ast.Store())
        ],
                   value=ast.BinOp(left=_name("__s"),
                                   op=ast.Div(),
                                   right=ast.Subscript(value=_name(target.id),
                                                       slice=ast.Tuple(elts=[_name("__j"), _name("__j")],
                                                                       ctx=ast.Load()),
                                                       ctx=ast.Load()))),
    ]
    j_body = [
        ast.Assign(targets=[_store("__s")],
                   value=ast.Subscript(value=_name(a.id),
                                       slice=ast.Tuple(elts=[_name("__j"), _name("__j")], ctx=ast.Load()),
                                       ctx=ast.Load())),
        ast.For(target=_store("__k"),
                iter=ast.Call(func=_name("range"), args=[_name("__j")], keywords=[]),
                body=inner_k,
                orelse=[]),
        ast.Assign(targets=[
            ast.Subscript(value=_name(target.id),
                          slice=ast.Tuple(elts=[_name("__j"), _name("__j")], ctx=ast.Load()),
                          ctx=ast.Store())
        ],
                   value=ast.Call(func=_name("sqrt"), args=[_name("__s")], keywords=[])),
        ast.For(target=_store("__i"),
                iter=ast.Call(func=_name("range"),
                              args=[ast.BinOp(left=_name("__j"), op=ast.Add(), right=_const(1)), n_ast],
                              keywords=[]),
                body=inner_i,
                orelse=[]),
    ]
    # numpy's cholesky returns 0 in the strict upper triangle, but the
    # Banachiewicz loop below only writes the lower triangle + diagonal.
    # Pre-zero the upper triangle so unwritten cells aren't malloc garbage;
    # ``target`` is a fresh temp (!= ``a``), so this can't corrupt the source.
    zero_upper = ast.For(target=_store("__zi"),
                         iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                         body=[
                             ast.For(target=_store("__zj"),
                                     iter=ast.Call(
                                         func=_name("range"),
                                         args=[ast.BinOp(left=_name("__zi"), op=ast.Add(), right=_const(1)), n_ast],
                                         keywords=[]),
                                     body=[
                                         ast.Assign(targets=[
                                             ast.Subscript(value=_name(target.id),
                                                           slice=ast.Tuple(elts=[_name("__zi"),
                                                                                 _name("__zj")],
                                                                           ctx=ast.Load()),
                                                           ctx=ast.Store())
                                         ],
                                                    value=_const(0.0))
                                     ],
                                     orelse=[])
                         ],
                         orelse=[])
    return [
        zero_upper,
        ast.For(target=_store("__j"),
                iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                body=j_body,
                orelse=[])
    ]


def expand_histogram(target: ast.expr,
                     args: List[ast.expr],
                     shape_table: Dict[str, Tuple[str, ...]],
                     kwargs=None,
                     local_dtypes: Optional[Dict[str, str]] = None,
                     fresh_local_allocs: Optional[Dict[str, Tuple[str, ...]]] = None) -> List[ast.stmt]:
    """``hist = np.histogram(a, bins[, range=(lo, hi)][, weights=w])[0]``.
    Implements the numpy histogram contract: if ``range`` is None, lo/hi come
    from an inline min/max pass; if ``weights=w`` is given, each element
    contributes ``w[i]`` instead of 1.0. Bin index: ``min(bins - 1, max(0, (a[i]
    - lo) * bins // (hi - lo)))`` -- the clamp is required because numpy's last
    bin is closed (``[edges[-2], edges[-1]]`` includes the right endpoint,
    unlike every other half-open bin) -- then WALKED ONE STEP against the
    materialised bin edges, which is what makes it equal to numpy's answer
    rather than merely close to it (see the edge block below).

    Supported call shapes (positional + keyword): ``np.histogram(a, bins)``,
    ``np.histogram(a, bins, weights=w)``, ``np.histogram(a, bins, lo, hi)``,
    ``np.histogram(a, bins, lo, hi, weights=w)``.
    """
    kwargs = kwargs or []
    if len(args) < 2 or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.histogram needs (a, bins[, lo, hi])")
    a = args[0]
    bins = args[1]
    a_shape = shape_table.get(a.id)
    if not a_shape or len(a_shape) != 1:
        raise NotImplementedError("np.histogram: only 1-D input")
    n = a_shape[0]
    n_ast = _const_or_name(n)
    # ``range`` keyword (numpy: a 2-tuple) or positional lo / hi.
    lo: Optional[ast.expr] = None
    hi: Optional[ast.expr] = None
    if len(args) >= 4:
        lo, hi = args[2], args[3]
    for kw in kwargs:
        if kw.arg == "range" and isinstance(kw.value, ast.Tuple) and len(kw.value.elts) == 2:
            lo, hi = kw.value.elts[0], kw.value.elts[1]
    # ``weights`` keyword.
    weights: Optional[ast.expr] = None
    for kw in kwargs:
        if kw.arg == "weights" and isinstance(kw.value, ast.Name):
            weights = kw.value
    out: List[ast.stmt] = []
    # When range is unspecified, compute a.min() / a.max() inline.
    if lo is None or hi is None:
        out.append(
            ast.Assign(targets=[_store("__hlo")],
                       value=ast.Subscript(value=_name(a.id), slice=_const(0), ctx=ast.Load())))
        out.append(
            ast.Assign(targets=[_store("__hhi")],
                       value=ast.Subscript(value=_name(a.id), slice=_const(0), ctx=ast.Load())))
        scan_body = [
            ast.If(test=ast.Compare(left=ast.Subscript(value=_name(a.id), slice=_name("__hsi"), ctx=ast.Load()),
                                    ops=[ast.Lt()],
                                    comparators=[_name("__hlo")]),
                   body=[
                       ast.Assign(targets=[_store("__hlo")],
                                  value=ast.Subscript(value=_name(a.id), slice=_name("__hsi"), ctx=ast.Load()))
                   ],
                   orelse=[]),
            ast.If(test=ast.Compare(left=ast.Subscript(value=_name(a.id), slice=_name("__hsi"), ctx=ast.Load()),
                                    ops=[ast.Gt()],
                                    comparators=[_name("__hhi")]),
                   body=[
                       ast.Assign(targets=[_store("__hhi")],
                                  value=ast.Subscript(value=_name(a.id), slice=_name("__hsi"), ctx=ast.Load()))
                   ],
                   orelse=[]),
        ]
        out.append(
            ast.For(target=_store("__hsi"),
                    iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                    body=scan_body,
                    orelse=[]))
        lo, hi = _name("__hlo"), _name("__hhi")
    # Zero the target.
    zero_body = [
        ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=_name("__bi"), ctx=ast.Store())],
                   value=_const(0.0))
    ]
    out.append(
        ast.For(target=_store("__bi"),
                iter=ast.Call(func=_name("range"), args=[bins], keywords=[]),
                body=zero_body,
                orelse=[]))
    # numpy's bin is defined by its EDGE ARRAY, not by the closed form below: it truncates the
    # same index and then walks it one step against linspace's edges. The two round apart --
    # probing every edge and one ulp either side, the closed form alone puts 248 of 3005 probes
    # one bin over at bins=1000. The edges are therefore rebuilt with linspace's own arithmetic
    # and, crucially, in the SAMPLE dtype, one rounding per statement -- which is why ``step``
    # is parked in the buffer rather than left in a scalar a backend may evaluate wider. Only
    # edges 0..bins-1 are read (the walk up stops at bins-1), hence no top edge.
    #
    # Named after the histogram it belongs to rather than from a running counter: a counter is
    # process-global, so the name would depend on how many kernels the interpreter emitted first
    # and two emitter runs would not agree byte for byte.
    edges_name = f"__hedge_{target.id}"
    step_name = f"__hstep_{target.id}"
    bins_dim = ast.unparse(bins)
    shape_table[edges_name] = (bins_dim, )
    if local_dtypes is not None:
        a_dt = local_dtypes.get(a.id)
        if a_dt is not None:
            local_dtypes[edges_name] = a_dt
            local_dtypes[step_name] = a_dt
    if fresh_local_allocs is not None:
        fresh_local_allocs[edges_name] = (bins_dim, )

    def edge_at(idx: ast.expr, ctx) -> ast.Subscript:
        """``<edges>[idx]`` -- a FRESH Subscript per call; a shared node is renamed in place by
        the Fortran emitter's loop-variable uniquifier (see expand_linalg_inv)."""
        return ast.Subscript(value=_name(edges_name), slice=idx, ctx=ctx)

    out.append(_alloc_marker(edges_name))
    out.append(
        ast.Assign(targets=[edge_at(_const(0), ast.Store())],
                   value=ast.BinOp(left=ast.BinOp(left=copy.deepcopy(hi), op=ast.Sub(), right=copy.deepcopy(lo)),
                                   op=ast.Div(),
                                   right=copy.deepcopy(bins))))
    out.append(ast.Assign(targets=[_store(step_name)], value=edge_at(_const(0), ast.Load())))
    out.append(
        ast.For(target=_store("__hj"),
                iter=ast.Call(func=_name("range"), args=[copy.deepcopy(bins)], keywords=[]),
                body=[
                    ast.Assign(targets=[edge_at(_name("__hj"), ast.Store())],
                               value=ast.BinOp(left=_name("__hj"), op=ast.Mult(), right=_name(step_name))),
                    ast.Assign(targets=[edge_at(_name("__hj"), ast.Store())],
                               value=ast.BinOp(left=edge_at(_name("__hj"), ast.Load()),
                                               op=ast.Add(),
                                               right=copy.deepcopy(lo))),
                ],
                orelse=[]))
    # Per-element binning. Bin index (truncated via ``int()``):
    #   bidx = min(bins - 1, max(0, int((a[i] - lo) * bins / (hi - lo))))
    # FloorDiv is wrong here since numerator/denominator are real-valued; numpy
    # uses floor(real_div), same as int(positive_real_div) for nonneg values.
    a_i = ast.Subscript(value=_name(a.id), slice=_name("__hi"), ctx=ast.Load())
    bin_idx = ast.Call(func=_name("int"),
                       args=[
                           ast.BinOp(left=ast.BinOp(left=ast.BinOp(left=a_i, op=ast.Sub(), right=lo),
                                                    op=ast.Mult(),
                                                    right=bins),
                                     op=ast.Div(),
                                     right=ast.BinOp(left=hi, op=ast.Sub(), right=lo))
                       ],
                       keywords=[])
    # int() each clamp bound so every min/max operand is int64: bins-1 and 0 are otherwise
    # default-kind integers, and Fortran's min(default, INT(.., c_int64_t)) is a mixed-kind
    # GNU extension that -std=f2018 rejects (harmless (int64_t) casts in C).
    clamp = ast.Call(func=_name("min"),
                     args=[
                         ast.Call(func=_name("int"),
                                  args=[ast.BinOp(left=bins, op=ast.Sub(), right=_const(1))],
                                  keywords=[]),
                         ast.Call(func=_name("max"),
                                  args=[ast.Call(func=_name("int"), args=[_const(0)], keywords=[]), bin_idx],
                                  keywords=[])
                     ],
                     keywords=[])
    add_val: ast.expr
    if weights is not None:
        add_val = ast.Subscript(value=_name(weights.id), slice=_name("__hi"), ctx=ast.Load())
    else:
        add_val = _const(1.0)
    # numpy drops samples outside [lo, hi] (only the last bin is closed); guard the increment so
    # they are not folded into the edge bins. An auto lo/hi (a.min()/a.max()) makes this a no-op.
    in_range = ast.BoolOp(op=ast.And(),
                          values=[
                              ast.Compare(left=copy.deepcopy(lo), ops=[ast.LtE()], comparators=[copy.deepcopy(a_i)]),
                              ast.Compare(left=copy.deepcopy(a_i), ops=[ast.LtE()], comparators=[copy.deepcopy(hi)]),
                          ])
    # numpy's own two corrections, in its order: step down when the sample falls below its own
    # bin's lower edge, then up when it reaches the next edge (never off the last bin).
    walk_down = ast.If(test=ast.Compare(left=copy.deepcopy(a_i),
                                        ops=[ast.Lt()],
                                        comparators=[edge_at(_name("__bidx"), ast.Load())]),
                       body=[
                           ast.Assign(targets=[_store("__bidx")],
                                      value=ast.BinOp(left=_name("__bidx"), op=ast.Sub(), right=_const(1)))
                       ],
                       orelse=[])
    walk_up = ast.If(test=ast.BoolOp(
        op=ast.And(),
        values=[
            ast.Compare(left=_name("__bidx"),
                        ops=[ast.Lt()],
                        comparators=[ast.BinOp(left=copy.deepcopy(bins), op=ast.Sub(), right=_const(1))]),
            ast.Compare(
                left=copy.deepcopy(a_i),
                ops=[ast.GtE()],
                comparators=[edge_at(ast.BinOp(left=_name("__bidx"), op=ast.Add(), right=_const(1)), ast.Load())]),
        ]),
                     body=[
                         ast.Assign(targets=[_store("__bidx")],
                                    value=ast.BinOp(left=_name("__bidx"), op=ast.Add(), right=_const(1)))
                     ],
                     orelse=[])
    bin_body = [
        ast.Assign(targets=[_store("__bidx")], value=clamp),
        walk_down,
        walk_up,
        ast.If(test=in_range,
               body=[
                   ast.AugAssign(target=ast.Subscript(value=_name(target.id), slice=_name("__bidx"), ctx=ast.Store()),
                                 op=ast.Add(),
                                 value=add_val)
               ],
               orelse=[]),
    ]
    out.append(
        ast.For(target=_store("__hi"),
                iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                body=bin_body,
                orelse=[]))
    return out


def expand_linalg_solve(target: ast.expr,
                        args: List[ast.expr],
                        shape_table: Dict[str, Tuple[str, ...]],
                        kwargs=None,
                        local_dtypes: Optional[Dict[str, str]] = None,
                        fresh_local_allocs: Optional[Dict[str, Tuple[str, ...]]] = None) -> List[ast.stmt]:
    """``x = np.linalg.solve(A, b)`` solves ``Ax = b`` for square A. Implemented
    as Gauss-Jordan elimination on the augmented [A | b] matrix. Conservative: A
    must be Name (N, N), b must be Name (N,) or (N, K); x is written into
    target with the same shape as b.
    """
    if len(args) < 2 or not isinstance(args[0], ast.Name) \
            or not isinstance(args[1], ast.Name):
        raise NotImplementedError("np.linalg.solve needs Name args")
    a = args[0]
    b = args[1]
    a_shape = shape_table.get(a.id)
    b_shape = shape_table.get(b.id)
    if not a_shape or len(a_shape) != 2:
        raise NotImplementedError("np.linalg.solve: A must be 2-D")
    if not b_shape or len(b_shape) not in (1, 2):
        raise NotImplementedError("np.linalg.solve: b must be 1-D or 2-D")
    n = a_shape[0]
    n_ast = _const_or_name(n)
    # Same Gauss-Jordan body as expand_linalg_inv, but the row ops apply to
    # ``b`` (not the identity), giving x = A^-1 @ b. ``__sol_aw`` is the
    # working copy of A. A fixed name would alias across call sites with
    # different sizes, so number it like ``expand_linalg_inv`` does.
    aw_name = f"__sol_aw{_LINALG_AW[0]}"
    _LINALG_AW[0] += 1
    out: List[ast.stmt] = []
    aw = lambda r, c: ast.Subscript(value=_name(aw_name), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Load())
    aw_store = lambda r, c: ast.Subscript(
        value=_name(aw_name), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Store())
    # ``b`` indexing depends on rank.
    is_2d = len(b_shape) == 2

    def b_load(r, c=None):
        if is_2d:
            return ast.Subscript(value=_name(target.id), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Load())
        return ast.Subscript(value=_name(target.id), slice=r, ctx=ast.Load())

    def b_store(r, c=None):
        if is_2d:
            return ast.Subscript(value=_name(target.id), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Store())
        return ast.Subscript(value=_name(target.id), slice=r, ctx=ast.Store())

    # Publish working-buffer shape + dtype + fresh-local alloc so the emit
    # declares the buffer as a flat 2-D buffer of A's element dtype (same
    # logic as ``expand_linalg_inv``).
    shape_table[aw_name] = (n, n)
    a_dt = None
    if local_dtypes is not None:
        a_dt = local_dtypes.get(a.id) or local_dtypes.get(b.id)
        if a_dt is not None:
            local_dtypes[aw_name] = a_dt
            local_dtypes[target.id] = a_dt
            for nm in ("__sol_tmp", "__sol_factor"):
                local_dtypes.setdefault(nm, a_dt)
    if fresh_local_allocs is not None:
        fresh_local_allocs[aw_name] = (n, n)
    out.append(
        ast.Assign(targets=[_store(aw_name)],
                   value=ast.Call(func=_name("__hpcagent_bench_zeros__"), args=[], keywords=[])))
    # Init: copy A into __sol_aw and b into target.
    if is_2d:
        m_ast = _const_or_name(b_shape[1])
        copy_inner = ast.For(target=_store("__sol_j"),
                             iter=ast.Call(func=_name("range"), args=[m_ast], keywords=[]),
                             body=[
                                 ast.Assign(targets=[b_store(_name("__sol_i"), _name("__sol_j"))],
                                            value=ast.Subscript(value=_name(b.id),
                                                                slice=ast.Tuple(
                                                                    elts=[_name("__sol_i"),
                                                                          _name("__sol_j")],
                                                                    ctx=ast.Load()),
                                                                ctx=ast.Load()))
                             ],
                             orelse=[])
    else:
        copy_inner = ast.Assign(targets=[b_store(_name("__sol_i"))],
                                value=ast.Subscript(value=_name(b.id), slice=_name("__sol_i"), ctx=ast.Load()))
    out.append(
        ast.For(target=_store("__sol_i"),
                iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                body=[
                    ast.For(target=_store("__sol_j"),
                            iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                            body=[
                                ast.Assign(targets=[aw_store(_name("__sol_i"), _name("__sol_j"))],
                                           value=ast.Subscript(value=_name(a.id),
                                                               slice=ast.Tuple(
                                                                   elts=[_name("__sol_i"),
                                                                         _name("__sol_j")],
                                                                   ctx=ast.Load()),
                                                               ctx=ast.Load()))
                            ],
                            orelse=[]),
                    copy_inner if is_2d else copy_inner,
                ],
                orelse=[]))
    # Gauss-Jordan on (__sol_aw | target).
    K = _name("__sol_k")
    P = _name("__sol_p")
    R = _name("__sol_r")
    C = _name("__sol_c")
    F = _name("__sol_factor")
    T = _name("__sol_tmp")
    # Pivot search.
    pivot_init = ast.Assign(targets=[_store("__sol_p")], value=K)
    pivot_scan = ast.For(target=_store("__sol_r"),
                         iter=ast.Call(func=_name("range"),
                                       args=[ast.BinOp(left=K, op=ast.Add(), right=_const(1)), n_ast],
                                       keywords=[]),
                         body=[
                             ast.If(test=ast.Compare(
                                 left=ast.Call(func=_name("abs"), args=[aw(R, K)], keywords=[]),
                                 ops=[ast.Gt()],
                                 comparators=[ast.Call(func=_name("abs"), args=[aw(P, K)], keywords=[])]),
                                    body=[ast.Assign(targets=[_store("__sol_p")], value=R)],
                                    orelse=[])
                         ],
                         orelse=[])
    # Swap row p and row k in __sol_aw.
    swap_aw = [
        ast.Assign(targets=[_store("__sol_tmp")], value=aw(K, C)),
        ast.Assign(targets=[aw_store(K, C)], value=aw(P, C)),
        ast.Assign(targets=[aw_store(P, C)], value=T),
    ]
    swap_aw_loop = ast.For(target=_store("__sol_c"),
                           iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                           body=swap_aw,
                           orelse=[])
    # Swap row p and row k in target (the b-side).
    if is_2d:
        m_ast = _const_or_name(b_shape[1])
        swap_b = [
            ast.Assign(targets=[_store("__sol_tmp")], value=b_load(K, C)),
            ast.Assign(targets=[b_store(K, C)], value=b_load(P, C)),
            ast.Assign(targets=[b_store(P, C)], value=T),
        ]
        swap_b_loop = ast.For(target=_store("__sol_c"),
                              iter=ast.Call(func=_name("range"), args=[m_ast], keywords=[]),
                              body=swap_b,
                              orelse=[])
    else:
        swap_b_loop = ast.If(test=ast.Compare(left=P, ops=[ast.NotEq()], comparators=[K]),
                             body=[
                                 ast.Assign(targets=[_store("__sol_tmp")], value=b_load(K)),
                                 ast.Assign(targets=[b_store(K)], value=b_load(P)),
                                 ast.Assign(targets=[b_store(P)], value=T),
                             ],
                             orelse=[])
    # Divide pivot row by aw[k, k]. Stash divisor.
    pivot_div_stash = ast.Assign(targets=[_store("__sol_factor")], value=aw(K, K))
    pivot_div_aw_body = [
        ast.Assign(targets=[aw_store(K, C)], value=ast.BinOp(left=aw(K, C), op=ast.Div(), right=F)),
    ]
    pivot_div_aw = ast.For(target=_store("__sol_c"),
                           iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                           body=pivot_div_aw_body,
                           orelse=[])
    if is_2d:
        pivot_div_b_body = [
            ast.Assign(targets=[b_store(K, C)], value=ast.BinOp(left=b_load(K, C), op=ast.Div(), right=F)),
        ]
        pivot_div_b = ast.For(target=_store("__sol_c"),
                              iter=ast.Call(func=_name("range"), args=[_const_or_name(b_shape[1])], keywords=[]),
                              body=pivot_div_b_body,
                              orelse=[])
    else:
        pivot_div_b = ast.Assign(targets=[b_store(K)], value=ast.BinOp(left=b_load(K), op=ast.Div(), right=F))
    # Eliminate other rows.
    elim_factor = ast.Assign(targets=[_store("__sol_factor")], value=aw(R, K))
    elim_aw_inner = ast.For(target=_store("__sol_c"),
                            iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                            body=[
                                ast.Assign(targets=[aw_store(R, C)],
                                           value=ast.BinOp(left=aw(R, C),
                                                           op=ast.Sub(),
                                                           right=ast.BinOp(left=F, op=ast.Mult(), right=aw(K, C))))
                            ],
                            orelse=[])
    if is_2d:
        elim_b_inner = ast.For(target=_store("__sol_c"),
                               iter=ast.Call(func=_name("range"), args=[_const_or_name(b_shape[1])], keywords=[]),
                               body=[
                                   ast.Assign(targets=[b_store(R, C)],
                                              value=ast.BinOp(left=b_load(R, C),
                                                              op=ast.Sub(),
                                                              right=ast.BinOp(left=F, op=ast.Mult(), right=b_load(K,
                                                                                                                  C))))
                               ],
                               orelse=[])
    else:
        elim_b_inner = ast.Assign(targets=[b_store(R)],
                                  value=ast.BinOp(left=b_load(R),
                                                  op=ast.Sub(),
                                                  right=ast.BinOp(left=F, op=ast.Mult(), right=b_load(K))))
    elim_outer = ast.For(target=_store("__sol_r"),
                         iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                         body=[
                             ast.If(test=ast.Compare(left=R, ops=[ast.NotEq()], comparators=[K]),
                                    body=[elim_factor, elim_aw_inner, elim_b_inner],
                                    orelse=[])
                         ],
                         orelse=[])
    k_body = [pivot_init, pivot_scan, swap_aw_loop, swap_b_loop, pivot_div_stash, pivot_div_aw, pivot_div_b, elim_outer]
    out.append(
        ast.For(target=_store("__sol_k"),
                iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                body=k_body,
                orelse=[]))
    return out


#: Monotone counter for the ``inv`` scratch working-copy buffer. A fixed name
#: (``__inv_aw``) would alias across call sites: LS3DF's Rayleigh-Ritz runs
#: ``np.linalg.inv`` once per SCF sub-solve, each a different size, so the
#: second call's shape would overwrite the first's declaration and the first
#: inversion would malloc from an uninitialised dimension. A per-call suffix
#: keeps each buffer distinct. Reset per translation unit by :func:`reset_temp_counters`.
_LINALG_AW = [0]


def reset_temp_counters() -> None:
    """Zero the scratch-buffer name counters at the start of a translation unit.

    They only have to be unique WITHIN one kernel, but they are module state, so left
    running they number the second kernel emitted in a process from wherever the first
    stopped -- the emitted text then depends on what else the process translated, and
    re-emitting the same kernel twice yields two different sources. Called by
    :func:`numpyto_common.lowering.lower`, the single entry point of a translation unit."""
    _OP_SPILL_TEMP[0] = 0
    _LINALG_AW[0] = 0


def expand_linalg_inv(target: ast.expr,
                      args: List[ast.expr],
                      shape_table: Dict[str, Tuple[str, ...]],
                      kwargs=None,
                      local_dtypes: Optional[Dict[str, str]] = None,
                      fresh_local_allocs: Optional[Dict[str, Tuple[str, ...]]] = None) -> List[ast.stmt]:
    """``X = np.linalg.inv(A)`` -- in-place Gauss-Jordan elimination with
    partial pivoting on the augmented [A | I] matrix (textbook, Golub & Van
    Loan Algorithm 3.4.1):

    1. Form A_work = copy of A, X = I.
    2. For each pivot column k: find row with max |A_work[k:, k]|, swap rows k
       and pivot; divide pivot row by A_work[k, k]; for each row i != k,
       subtract A_work[i, k] * pivot row.
    3. Result: A_work becomes I, X becomes A^-1.

    Conservative: ``A`` must be a Name with a known square 2-D shape. A is
    preserved -- the elimination runs on a copy, ``__inv_aw``.
    """
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.linalg.inv needs Name first arg")
    a = args[0]
    shape = shape_table.get(a.id)
    if not shape or len(shape) != 2:
        raise NotImplementedError("np.linalg.inv: only 2-D square input supported")
    n = shape[0]
    n_ast = _const_or_name(n)
    aw_name = f"__inv_aw{_LINALG_AW[0]}"
    _LINALG_AW[0] += 1
    out: List[ast.stmt] = []
    # Publish the working buffer's shape so the emit flattens ``__inv_aw[i, j]``
    # as ``i*n + j`` instead of chained ``[i][j]``, and its dtype so the emit
    # declares the right element type (e.g. ``double _Complex __inv_aw[n*n]``).
    shape_table[aw_name] = (n, n)
    a_dt = None
    if local_dtypes is not None:
        a_dt = local_dtypes.get(a.id)
        if a_dt is not None:
            local_dtypes[aw_name] = a_dt
            local_dtypes[target.id] = a_dt
            # Scalar swap / pivot temps carry A's dtype too.
            for nm in ("__inv_tmp", "__inv_factor"):
                local_dtypes.setdefault(nm, a_dt)
    if fresh_local_allocs is not None:
        fresh_local_allocs[aw_name] = (n, n)
    out.append(
        ast.Assign(targets=[_store(aw_name)],
                   value=ast.Call(func=_name("__hpcagent_bench_zeros__"), args=[], keywords=[])))
    # Copy A to a working buffer ``__inv_aw[i, j]``; initialise target
    # as the identity I[i, j].
    out.append(
        ast.For(target=_store("__inv_i"),
                iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                body=[
                    ast.For(target=_store("__inv_j"),
                            iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                            body=[
                                ast.Assign(targets=[
                                    ast.Subscript(value=_name(aw_name),
                                                  slice=ast.Tuple(elts=[_name("__inv_i"),
                                                                        _name("__inv_j")],
                                                                  ctx=ast.Load()),
                                                  ctx=ast.Store())
                                ],
                                           value=ast.Subscript(value=_name(a.id),
                                                               slice=ast.Tuple(
                                                                   elts=[_name("__inv_i"),
                                                                         _name("__inv_j")],
                                                                   ctx=ast.Load()),
                                                               ctx=ast.Load())),
                                ast.Assign(targets=[
                                    ast.Subscript(value=_name(target.id),
                                                  slice=ast.Tuple(elts=[_name("__inv_i"),
                                                                        _name("__inv_j")],
                                                                  ctx=ast.Load()),
                                                  ctx=ast.Store())
                                ],
                                           value=ast.IfExp(test=ast.Compare(left=_name("__inv_i"),
                                                                            ops=[ast.Eq()],
                                                                            comparators=[_name("__inv_j")]),
                                                           body=_const(1.0),
                                                           orelse=_const(0.0))),
                            ],
                            orelse=[])
                ],
                orelse=[]))
    # Outer loop over pivot column k = 0..n:
    # 1) find pivot row p = k; for r in k+1..n: if |aw[r,k]| > |aw[p,k]|: p = r
    # 2) swap rows p and k in aw and target
    # 3) divide aw[k, :] and target[k, :] by aw[k, k]
    # 4) for r != k: factor = aw[r, k]; aw[r, :] -= factor * aw[k, :];
    #                target[r, :] -= factor * target[k, :]
    aw = lambda r, c: ast.Subscript(value=_name(aw_name), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Load())
    aw_store = lambda r, c: ast.Subscript(
        value=_name(aw_name), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Store())
    tgt = lambda r, c: ast.Subscript(
        value=_name(target.id), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Load())
    tgt_store = lambda r, c: ast.Subscript(
        value=_name(target.id), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Store())
    # Every use below is a FRESH node from this factory, never a shared one: the Fortran
    # emitter's loop-variable uniquifier mutates Name nodes in place, and __inv_r / __inv_c each
    # bind THREE separate sibling/nested loops here (pivot_scan+elim_outer; swap_loop+pivot_div+
    # elim_inner_loop). A shared node's rename from the first loop stuck on every later occurrence,
    # so elim_outer's body kept reading pivot_scan's row and swap_loop's column -- silently wrong
    # data (not a crash) that only showed up as a numeric mismatch, and one build compiled from
    # freed-then-realloc'd memory besides. See _FortranRenameTemps.visit_Name.
    nm = lambda tag: _name(f"__inv_{tag}")
    # Pivot search.
    pivot_init = ast.Assign(targets=[_store("__inv_p")], value=nm("k"))
    pivot_scan = ast.For(target=_store("__inv_r"),
                         iter=ast.Call(func=_name("range"),
                                       args=[ast.BinOp(left=nm("k"), op=ast.Add(), right=_const(1)), n_ast],
                                       keywords=[]),
                         body=[
                             ast.If(test=ast.Compare(
                                 left=ast.Call(func=_name("abs"), args=[aw(nm("r"), nm("k"))], keywords=[]),
                                 ops=[ast.Gt()],
                                 comparators=[ast.Call(func=_name("abs"), args=[aw(nm("p"), nm("k"))], keywords=[])]),
                                    body=[ast.Assign(targets=[_store("__inv_p")], value=nm("r"))],
                                    orelse=[])
                         ],
                         orelse=[])
    # Swap row p and row k in both aw and target.
    swap_body = [
        ast.Assign(targets=[_store("__inv_tmp")], value=aw(nm("k"), nm("c"))),
        ast.Assign(targets=[aw_store(nm("k"), nm("c"))], value=aw(nm("p"), nm("c"))),
        ast.Assign(targets=[aw_store(nm("p"), nm("c"))], value=nm("tmp")),
        ast.Assign(targets=[_store("__inv_tmp")], value=tgt(nm("k"), nm("c"))),
        ast.Assign(targets=[tgt_store(nm("k"), nm("c"))], value=tgt(nm("p"), nm("c"))),
        ast.Assign(targets=[tgt_store(nm("p"), nm("c"))], value=nm("tmp")),
    ]
    swap_loop = ast.For(target=_store("__inv_c"),
                        iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                        body=swap_body,
                        orelse=[])
    # Divide pivot row by aw[k, k] -- stash it first since the loop overwrites
    # aw[k, k] itself before every use.
    pivot_div_stash = ast.Assign(targets=[_store("__inv_factor")], value=aw(nm("k"), nm("k")))
    pivot_div_body_safe = [
        ast.Assign(targets=[tgt_store(nm("k"), nm("c"))],
                   value=ast.BinOp(left=tgt(nm("k"), nm("c")), op=ast.Div(), right=nm("factor"))),
        ast.Assign(targets=[aw_store(nm("k"), nm("c"))],
                   value=ast.BinOp(left=aw(nm("k"), nm("c")), op=ast.Div(), right=nm("factor"))),
    ]
    pivot_div = [
        pivot_div_stash,
        ast.For(target=_store("__inv_c"),
                iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                body=pivot_div_body_safe,
                orelse=[]),
    ]
    # Eliminate other rows.
    elim_factor = ast.Assign(targets=[_store("__inv_factor")], value=aw(nm("r"), nm("k")))
    elim_body_inner = [
        ast.Assign(targets=[tgt_store(nm("r"), nm("c"))],
                   value=ast.BinOp(left=tgt(nm("r"), nm("c")),
                                   op=ast.Sub(),
                                   right=ast.BinOp(left=nm("factor"), op=ast.Mult(), right=tgt(nm("k"), nm("c"))))),
        ast.Assign(targets=[aw_store(nm("r"), nm("c"))],
                   value=ast.BinOp(left=aw(nm("r"), nm("c")),
                                   op=ast.Sub(),
                                   right=ast.BinOp(left=nm("factor"), op=ast.Mult(), right=aw(nm("k"), nm("c"))))),
    ]
    elim_inner_loop = ast.For(target=_store("__inv_c"),
                              iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                              body=elim_body_inner,
                              orelse=[])
    elim_outer = ast.For(target=_store("__inv_r"),
                         iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                         body=[
                             ast.If(test=ast.Compare(left=nm("r"), ops=[ast.NotEq()], comparators=[nm("k")]),
                                    body=[elim_factor, elim_inner_loop],
                                    orelse=[])
                         ],
                         orelse=[])
    # K-loop body.
    k_body = [pivot_init, pivot_scan, swap_loop] + pivot_div + [elim_outer]
    out.append(
        ast.For(target=_store("__inv_k"),
                iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                body=k_body,
                orelse=[]))
    return out


def expand_linalg_det(target: ast.expr,
                      args: List[ast.expr],
                      shape_table: Dict[str, Tuple[str, ...]],
                      kwargs=None,
                      local_dtypes: Optional[Dict[str, str]] = None,
                      fresh_local_allocs: Optional[Dict[str, Tuple[str, ...]]] = None) -> List[ast.stmt]:
    """``s = np.linalg.det(A)`` -- LU factorisation (Gaussian elimination
    with partial pivoting) on a scratch copy ``__det_aw`` of A. The
    determinant is the product of the pivots (U's diagonal) times
    ``(-1) ** (# row swaps)``.

    ``target`` is a scalar (the hoister lifts a nested ``np.linalg.det``
    to a fresh scalar temp before this fires). Conservative: ``A`` must be
    a Name with a known square 2-D shape; ``A`` is preserved -- the
    elimination runs on the copy ``__det_aw``.
    """
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.linalg.det needs Name first arg")
    if not isinstance(target, ast.Name):
        raise NotImplementedError("np.linalg.det: scalar Name target expected")
    a = args[0]
    shape = shape_table.get(a.id)
    if not shape or len(shape) != 2:
        raise NotImplementedError("np.linalg.det: only 2-D square input supported")
    n = shape[0]
    n_ast = _const_or_name(n)
    out: List[ast.stmt] = []
    # Publish the working buffer's shape + dtype + fresh-local alloc so the
    # emit declares ``__det_aw`` as a flat 2-D buffer of A's element dtype
    # (same registration logic as ``expand_linalg_inv``).
    shape_table["__det_aw"] = (n, n)
    a_dt = None
    if local_dtypes is not None:
        a_dt = local_dtypes.get(a.id)
        if a_dt is not None:
            local_dtypes["__det_aw"] = a_dt
            for nm in ("__det_tmp", "__det_factor"):
                local_dtypes.setdefault(nm, a_dt)
            # The determinant of a complex matrix is complex; keep the
            # accumulator's dtype aligned with A's element dtype.
            local_dtypes.setdefault(target.id, a_dt)
    if fresh_local_allocs is not None:
        fresh_local_allocs["__det_aw"] = (n, n)
    aw = lambda r, c: ast.Subscript(
        value=_name("__det_aw"), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Load())
    aw_store = lambda r, c: ast.Subscript(
        value=_name("__det_aw"), slice=ast.Tuple(elts=[r, c], ctx=ast.Load()), ctx=ast.Store())
    out.append(
        ast.Assign(targets=[_store("__det_aw")],
                   value=ast.Call(func=_name("__hpcagent_bench_zeros__"), args=[], keywords=[])))
    # Copy A into the working buffer.
    out.append(
        ast.For(target=_store("__det_i"),
                iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                body=[
                    ast.For(target=_store("__det_j"),
                            iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                            body=[
                                ast.Assign(targets=[aw_store(_name("__det_i"), _name("__det_j"))],
                                           value=ast.Subscript(value=_name(a.id),
                                                               slice=ast.Tuple(
                                                                   elts=[_name("__det_i"),
                                                                         _name("__det_j")],
                                                                   ctx=ast.Load()),
                                                               ctx=ast.Load()))
                            ],
                            orelse=[])
                ],
                orelse=[]))
    # Accumulator starts at 1 (product of pivots * swap sign).
    out.append(ast.Assign(targets=[_store(target.id)], value=_const(1.0)))
    K = _name("__det_k")
    P = _name("__det_p")
    R = _name("__det_r")
    C = _name("__det_c")
    F = _name("__det_factor")
    T = _name("__det_tmp")
    # Pivot search: p = argmax_{r >= k} |aw[r, k]|.
    pivot_init = ast.Assign(targets=[_store("__det_p")], value=K)
    pivot_scan = ast.For(target=_store("__det_r"),
                         iter=ast.Call(func=_name("range"),
                                       args=[ast.BinOp(left=K, op=ast.Add(), right=_const(1)), n_ast],
                                       keywords=[]),
                         body=[
                             ast.If(test=ast.Compare(
                                 left=ast.Call(func=_name("abs"), args=[aw(R, K)], keywords=[]),
                                 ops=[ast.Gt()],
                                 comparators=[ast.Call(func=_name("abs"), args=[aw(P, K)], keywords=[])]),
                                    body=[ast.Assign(targets=[_store("__det_p")], value=R)],
                                    orelse=[])
                         ],
                         orelse=[])
    # Swap rows p and k (when distinct) and flip the running sign.
    swap_body = [
        ast.Assign(targets=[_store("__det_tmp")], value=aw(K, C)),
        ast.Assign(targets=[aw_store(K, C)], value=aw(P, C)),
        ast.Assign(targets=[aw_store(P, C)], value=T),
    ]
    swap_loop = ast.For(target=_store("__det_c"),
                        iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                        body=swap_body,
                        orelse=[])
    sign_flip = ast.Assign(targets=[_store(target.id)], value=ast.UnaryOp(op=ast.USub(), operand=_name(target.id)))
    swap_if = ast.If(test=ast.Compare(left=P, ops=[ast.NotEq()], comparators=[K]),
                     body=[swap_loop, sign_flip],
                     orelse=[])
    # Multiply the determinant by this pivot.
    pivot_mul = ast.Assign(targets=[_store(target.id)],
                           value=ast.BinOp(left=_name(target.id), op=ast.Mult(), right=aw(K, K)))
    # Eliminate rows below k (guard a zero pivot so a singular matrix
    # yields det == 0 rather than a NaN from divide-by-zero).
    elim_factor = ast.Assign(targets=[_store("__det_factor")],
                             value=ast.BinOp(left=aw(R, K), op=ast.Div(), right=aw(K, K)))
    elim_inner = ast.For(target=_store("__det_c"),
                         iter=ast.Call(func=_name("range"),
                                       args=[ast.BinOp(left=K, op=ast.Add(), right=_const(1)), n_ast],
                                       keywords=[]),
                         body=[
                             ast.Assign(targets=[aw_store(R, C)],
                                        value=ast.BinOp(left=aw(R, C),
                                                        op=ast.Sub(),
                                                        right=ast.BinOp(left=F, op=ast.Mult(), right=aw(K, C))))
                         ],
                         orelse=[])
    elim_outer = ast.For(target=_store("__det_r"),
                         iter=ast.Call(func=_name("range"),
                                       args=[ast.BinOp(left=K, op=ast.Add(), right=_const(1)), n_ast],
                                       keywords=[]),
                         body=[elim_factor, elim_inner],
                         orelse=[])
    elim_guard = ast.If(test=ast.Compare(left=ast.Call(func=_name("abs"), args=[aw(K, K)], keywords=[]),
                                         ops=[ast.Gt()],
                                         comparators=[_const(0.0)]),
                        body=[elim_outer],
                        orelse=[])
    k_body = [pivot_init, pivot_scan, swap_if, pivot_mul, elim_guard]
    out.append(
        ast.For(target=_store("__det_k"),
                iter=ast.Call(func=_name("range"), args=[n_ast], keywords=[]),
                body=k_body,
                orelse=[]))
    return out


#: Map of ``("np", attr) -> expander``. The expander signature is
#: ``(assign_target, call_args, shape_table) -> list[stmt]``.
NP_CALL_EXPANDERS: Dict[Tuple[str, str], Callable] = {
    # Reductions
    ("np", "sum"):
    expand_sum,
    # np.maximum/minimum.accumulate: a DaCe Scan re-emits cummax/cummin as these
    # (there's no np.cummax). ufunc.reduce forms are normalized to np.sum/...
    # upstream in native_desugar (_UfuncReduceToReducer), so they never reach here.
    ("np", "searchsorted"):
    expand_searchsorted,
    ("np", "maximum.accumulate"):
    expand_cummax,
    ("np", "minimum.accumulate"):
    expand_cummin,
    ("np", "sort"):
    expand_sort,
    ("np", "max"):
    expand_max,
    ("np", "min"):
    expand_min,
    ("np", "mean"):
    expand_mean,
    ("np", "prod"):
    expand_prod,
    ("np", "std"):
    expand_std,
    # Linear algebra
    ("np", "dot"):
    expand_dot_2d,
    ("np", "einsum"):
    expand_einsum,
    ("np", "tensordot"):
    expand_tensordot,
    ("np", "inner"):
    expand_inner,
    ("np", "vdot"):
    expand_vdot,
    ("np", "trace"):
    expand_trace,
    ("np", "diagonal"):
    expand_diagonal,
    ("np", "diag"):
    expand_diag,
    ("np", "cumsum"):
    expand_cumsum,
    ("np", "cumprod"):
    expand_cumprod,
    ("np", "median"):
    expand_median,
    ("np", "roll"):
    expand_roll,
    ("np", "tril"):
    expand_tril,
    ("np", "pad"):
    expand_pad,
    ("np", "outer"):
    expand_outer,
    ("np", "add.outer"):
    expand_add_outer,
    ("np", "transpose"):
    expand_transpose,
    ("np", "linalg.cholesky"):
    expand_cholesky,
    ("np", "linalg.norm"):
    expand_linalg_norm,
    ("np", "linalg.lstsq"):
    expand_lstsq,
    ("np", "linalg.inv"):
    expand_linalg_inv,
    ("np", "linalg.det"):
    expand_linalg_det,
    ("np", "linalg.solve"):
    expand_linalg_solve,
    ("np", "fft.fftn"):
    expand_fftn,
    ("np", "fft.ifftn"):
    expand_ifftn,
    ("np", "fft.fft"):
    expand_fft,
    ("np", "fft.ifft"):
    expand_ifft,
    ("np", "fft.fftfreq"):
    expand_fftfreq,
    ("np", "var"):
    expand_var,
    ("np", "argmax"):
    expand_argmax,
    ("np", "argmin"):
    expand_argmin,
    ("np", "any"):
    expand_any,
    ("np", "all"):
    expand_all,
    ("np", "count_nonzero"):
    expand_count_nonzero,
    # Memory / shape
    ("np", "copy"):
    expand_copy,
    # np.asarray/np.ascontiguousarray of an already-materialised array is a copy
    # (contiguity/dtype already hold for our buffers), so they lower exactly
    # like np.copy (dbcsr/minife pass inputs through np.asarray before indexing).
    ("np", "asarray"):
    expand_copy,
    ("np", "ascontiguousarray"):
    expand_copy,
    # ``np.array(<array expr>)`` is the same materialising copy. The literal-list and 0-d scalar
    # forms never reach here: the frontend rewrites both before lowering runs.
    ("np", "array"):
    expand_copy,
    ("np", "bincount"):
    expand_bincount,
    ("np", "reshape"):
    expand_reshape,
    ("np", "swapaxes"):
    expand_swapaxes,
    ("np", "moveaxis"):
    expand_moveaxis,
    ("np", "expand_dims"):
    expand_expand_dims,
    ("np", "squeeze"):
    expand_squeeze,
    ("np", "take"):
    expand_take,
    ("np", "repeat"):
    expand_repeat,
    ("np", "eye"):
    expand_eye,
    ("np", "meshgrid"):
    expand_meshgrid,
    ("np", "triu"):
    expand_triu,
    ("np", "hstack"):
    expand_hstack,
    ("np", "concatenate"):
    expand_concatenate,
    ("np", "stack"):
    expand_stack,
    ("np", "diff"):
    expand_diff,
    ("np", "flip"):
    expand_flip,
    ("np", "linspace"):
    expand_linspace,
    ("np", "arange"):
    expand_arange,
    ("np", "fromfunction"):
    expand_fromfunction,
    ("np", "histogram"):
    expand_histogram,
    # Elementwise
    ("np", "minimum"):
    expand_minimum,
    ("np", "maximum"):
    expand_maximum,
    ("np", "add"):
    expand_add,
    ("np", "multiply"):
    expand_multiply,
    # Comparison ops -> per-element Compare (boolean output array).
    ("np", "less"):
    expand_less,
    ("np", "less_equal"):
    expand_less_equal,
    ("np", "greater"):
    expand_greater,
    ("np", "greater_equal"):
    expand_greater_equal,
    ("np", "equal"):
    expand_equal,
    ("np", "not_equal"):
    expand_not_equal,
    # Logical ops -> per-element BoolOp / UnaryOp.
    ("np", "logical_and"):
    expand_logical_and,
    ("np", "logical_or"):
    expand_logical_or,
    ("np", "logical_not"):
    expand_logical_not,
    ("np", "subtract"):
    expand_subtract,
    ("np", "divide"):
    expand_divide,
    ("np", "true_divide"):
    expand_divide,
    ("np", "negative"):
    expand_negative,
    ("np", "power"):
    expand_power,
    ("np", "tanh"):
    expand_tanh,
    ("np", "clip"):
    expand_clip,
    ("np", "where"):
    expand_where,
    # Per-element math intrinsics. These also live in MATH_BUILTINS for
    # the scalar-arg form; the call-hoister catches the array form first.
    ("np", "exp"):
    expand_exp_arr,
    ("np", "log"):
    expand_log_arr,
    ("np", "sqrt"):
    expand_sqrt_arr,
    ("np", "sin"):
    expand_sin_arr,
    ("np", "cos"):
    expand_cos_arr,
}

# Elementwise transcendental/math ufuncs (ARRAY form). Mirrors the scalar
# _TRIG/_ALG_TRANS lists in lowering.py's MATH_BUILTINS so a function usable
# scalar-side is usable array-side too.
#
# Functions with a libm name (sin, atan2, rint, ...) emit a plain call --
# resolved through <math.h>/<cmath> -- via _unary_call_expander/
# _binary_call_expander. Functions without one (square, reciprocal, sign,
# degrees, radians) emit an inline expr (x*x, 1.0/x, ...) via
# _unary_expr_expander: language-agnostic, so no helper functions/macros
# needed beyond the prelude's min/max/int_floor.


def _unary_call_expander(c_name: str) -> Callable:
    """Elementwise expander for a unary numpy ufunc that maps directly to
    a libm function (``np.tan(arr)`` -> ``out[i] = tan(arr[i])``)."""
    return lambda t, a, s: _unary_elementwise(t, a, s, lambda x: ast.Call(func=_name(c_name), args=[x], keywords=[]))


def _unary_expr_expander(make: Callable[[ast.expr], ast.expr]) -> Callable:
    """Elementwise expander for a unary ufunc with no direct libm name --
    the result is an expression of the (scalarised) operand. ``make`` may
    use the operand twice; it is deep-copied per use to avoid sharing a
    single AST node across the tree."""
    return lambda t, a, s: _unary_elementwise(t, a, s, lambda x: make(x))


#: numpy unary ufuncs that map 1:1 to a libm call. The scalar form is already
#: handled by ``MATH_BUILTINS``; this registers the ARRAY form so ``out =
#: np.tan(arr)`` lowers to a loop. ``round``/``around`` map to ``rint``
#: (round-half-to-even, matching numpy; C ``round`` is half-away-from-zero).
UNARY_C_MATH: Dict[str, str] = {
    "tan": "tan",
    "sinh": "sinh",
    "cosh": "cosh",
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "arcsinh": "asinh",
    "arccosh": "acosh",
    "arctanh": "atanh",
    "exp2": "exp2",
    "expm1": "expm1",
    "log2": "log2",
    "log10": "log10",
    "log1p": "log1p",
    "cbrt": "cbrt",
    "floor": "floor",
    "ceil": "ceil",
    "trunc": "trunc",
    "rint": "rint",
    "round": "rint",
    "around": "rint",
    "fabs": "fabs",
    "erf": "erf",
    "erfc": "erfc",
    "tgamma": "tgamma",
    "lgamma": "lgamma",
}
for _np_name, _c_name in UNARY_C_MATH.items():
    NP_CALL_EXPANDERS[("np", _np_name)] = _unary_call_expander(_c_name)

# Inline-expression ufuncs valid in both C and Fortran: square -> x*x,
# reciprocal -> 1.0/x, degrees/radians via the exact double conversion factor
# (180/pi, pi/180) as a plain numeric literal -- no M_PI (C-only), no
# per-language divergence.
_DEG_PER_RAD = 57.29577951308232  # 180 / pi
_RAD_PER_DEG = 0.017453292519943295  # pi / 180
NP_CALL_EXPANDERS[(
    "np",
    "square")] = _unary_expr_expander(lambda x: ast.BinOp(left=copy.deepcopy(x), op=ast.Mult(), right=copy.deepcopy(x)))
NP_CALL_EXPANDERS[("np",
                   "reciprocal")] = _unary_expr_expander(lambda x: ast.BinOp(left=_const(1.0), op=ast.Div(), right=x))
NP_CALL_EXPANDERS[(
    "np", "degrees")] = _unary_expr_expander(lambda x: ast.BinOp(left=x, op=ast.Mult(), right=_const(_DEG_PER_RAD)))
NP_CALL_EXPANDERS[("np", "rad2deg")] = NP_CALL_EXPANDERS[("np", "degrees")]
NP_CALL_EXPANDERS[(
    "np", "radians")] = _unary_expr_expander(lambda x: ast.BinOp(left=x, op=ast.Mult(), right=_const(_RAD_PER_DEG)))
NP_CALL_EXPANDERS[("np", "deg2rad")] = NP_CALL_EXPANDERS[("np", "radians")]
# ``sign`` has no both-language inline form (C bool arithmetic vs Fortran
# logicals), so emit a ``__npb_sign(x)`` marker each backend specialises in its
# own _emit_call. Kept out of the promotion pass via the math intrinsic name set.
NP_CALL_EXPANDERS[("np", "sign")] = _unary_call_expander("__npb_sign")


def _binary_call_expander(c_name: str) -> Callable:
    """Elementwise expander for a binary numpy ufunc that maps to a libm
    call (``np.arctan2(a, b)`` -> ``out[i] = atan2(a[i], b[i])``).
    Broadcasts a scalar second operand. Mirrors :func:`expand_power`."""

    def _expand(target, args, shape_table):
        if len(args) != 2:
            raise NotImplementedError(f"np.{c_name} needs 2 args")
        a, b = args
        extent = _iter_extent_of(a, shape_table)
        if extent is None:
            extent = _iter_extent_of(b, shape_table)
        if extent is None:
            raise NotImplementedError(f"np.{c_name}: extent unknown")
        iters = [_name(f"__r{i}") for i in range(len(extent))]
        sa = _scalarize_at_iters(a, iters, shape_table)
        sb = _scalarize_at_iters(b, iters, shape_table)
        idx = (iters[0] if len(iters) == 1 else ast.Tuple(elts=list(iters), ctx=ast.Load()))
        body = [
            ast.Assign(targets=[ast.Subscript(value=_name(target.id), slice=idx, ctx=ast.Store())],
                       value=ast.Call(func=_name(c_name), args=[sa, sb], keywords=[]))
        ]
        out = body
        for var, bound in zip(reversed([i.id for i in iters]), reversed(extent)):
            out = [
                ast.For(target=_store(var),
                        iter=ast.Call(func=_name("range"), args=[bound], keywords=[]),
                        body=out,
                        orelse=[])
            ]
        return out

    return _expand


for _np_name, _c_name in {
        "arctan2": "atan2",
        "hypot": "hypot",
        "copysign": "copysign",
        "fmod": "fmod",
        "fmax": "fmax",
        "fmin": "fmin"
}.items():
    NP_CALL_EXPANDERS[("np", _np_name)] = _binary_call_expander(_c_name)

#: ``np.zeros_like`` etc. share a shape with another array. The
#: rewriter at the lower() level translates these into a local-array
#: declaration the existing ``_ZerosRewriter`` already understands.
NP_ZEROS_ALIASES: Tuple[str, ...] = (
    "zeros",
    "empty",
    "zeros_like",
    "empty_like",
    "ones",
    "ones_like",
    "ndarray",  # ``np.ndarray((I, J, K), dtype=...)`` -- raw uninitialised
    # allocator used by gt4py-derived weather kernels (vadv).
    # Same shape harvest as ``np.empty``.
)


def _static_shape_of(expr, axis, shape_table):
    """Static (loop-var-free) shape token for the given axis of an expression,
    or None if not derivable. ``Subscript(Name, ...)`` returns the source
    array's full axis size from its declared shape, regardless of slice
    bounds, so a temp can be declared at function scope without depending on a
    loop variable."""
    if isinstance(expr, ast.Name):
        shape = shape_table.get(expr.id)
        if shape and axis < len(shape):
            return shape[axis]
    if isinstance(expr, ast.Subscript):
        name = expr.value.id if isinstance(expr.value, ast.Name) else None
        shape = shape_table.get(name) if name else None
        if shape:
            # Skip non-Slice axes to align with the array's full rank.
            sl = expr.slice
            axes = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            slice_count = 0
            for i, ax in enumerate(axes):
                if isinstance(ax, ast.Slice):
                    if slice_count == axis and i < len(shape):
                        return shape[i]
                    slice_count += 1
    return None


def _call_to_str(node):
    """Render an extent AST node as a shape-table token string."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return str(node.value)
    if isinstance(node, ast.Name):
        return node.id
    return ast.unparse(node)


#: Single identifier inside a shape-token string, matched on word boundaries so substituting ``c``
#: never hits ``channels`` or ``__inl6_c``.
DIM_IDENT_RE = re.compile(r"[A-Za-z_]\w*")

#: ``arr.shape[i]`` inside a shape token -- a DIMENSION read, resolvable against the shape table.
SHAPE_READ_RE = re.compile(r"(\w+)\.shape\[(\d+)\]")

#: Cap on alternating alias/shape-read expansion rounds. Each round can expose new names, so the
#: two rewrites do not reach a joint fixpoint in one pass; a chain deeper than this is pathological
#: and stopping early only costs a declined matmul.
DIM_EXPAND_ROUNDS: int = 8


def substitute_dim_aliases(token: str,
                           aliases: Dict[str, str],
                           shape_table: Optional[Dict[str, Tuple[str, ...]]] = None) -> str:
    """Rewrite one shape token into the kernel's PARAMETER vocabulary, to a fixpoint.

    A kernel names its own dimensions (``batch, channels, h, w = x.shape``), so the SAME extent
    reaches a comparison spelled two ways: ``channels`` from a body local, ``embed_dim`` from
    ``init.shapes``. ``aliases`` maps each dimension local to its definition; expanding both sides
    puts them in one vocabulary. Cycle-guarded on the active substitution chain, so a
    self-referential def stops expanding instead of recursing forever.

    An inlined helper spells its dims as a read off a LOCAL array (``__inl91_c =
    __inl8_y.shape[3]``), which no alias can resolve on its own -- hence ``shape_table``, and hence
    the rounds: resolving a shape read exposes fresh names to alias-expand, and vice versa.
    """

    def expand(text: str, active: Tuple[str, ...]) -> str:

        def repl(m: "re.Match") -> str:
            ident = m.group(0)
            if ident not in aliases or ident in active:
                return ident
            return "(" + expand(aliases[ident], active + (ident, )) + ")"

        return DIM_IDENT_RE.sub(repl, text)

    def resolve_shape_reads(text: str) -> str:

        def repl(m: "re.Match") -> str:
            shape = shape_table.get(m.group(1))
            idx = int(m.group(2))
            if shape is None or idx >= len(shape):
                return m.group(0)
            return "(" + str(shape[idx]) + ")"

        return SHAPE_READ_RE.sub(repl, text)

    # Each round is "aliases to a fixpoint, then resolve shape reads", and a round only REPEATS
    # when a shape read actually resolved -- that is the only thing that can expose a name the
    # alias pass has not seen. Looping on any change instead would restart the per-chain cycle
    # guard, and a self-referential def would grow one level per round instead of stopping.
    text = str(token)
    for _ in range(DIM_EXPAND_ROUNDS):
        grown = expand(text, ())
        if not shape_table:
            return grown
        resolved = resolve_shape_reads(grown)
        if resolved == grown:
            return grown
        text = resolved
    return text


@lru_cache(maxsize=None, typed=True)
def sympify_shape(text: str):
    """``text`` as a sympy expression, or ``None`` when it does not parse.

    One shape token is compared against many others, so without this the same string is re-parsed
    once per PAIR. Keyed on the string and never on a sympy object: sympy equality is structural
    and would collapse distinct tokens onto one entry.

    Every non-call identifier is bound to a Symbol first. Bare ``sympify`` resolves a name against
    sympy's own namespace, where ``N`` is the numeric-evaluation FUNCTION, ``S`` the singleton
    registry, and ``E``/``I``/``O``/``Q``/``pi``/``beta``/``gamma``/``zeta`` are constants or
    functions -- so ``N`` came back uncomparable and every extent naming it answered "not equal"
    however plainly equal it was. Those are ordinary dimension names in this corpus.

    The symbols are integral and non-negative because an extent is: without that, a pooled
    ``(2 * oh) // 2`` stays a ``floor`` sympy cannot cancel against ``oh``, and the two agreeing
    extents read as two different shapes. The assumption only lets sympy PROVE equalities that
    already hold for the integer extents these symbols stand for.
    """
    import sympy  # Deferred: sympy costs ~100s of ms to import and most kernels never reach here.
    # An unresolved ``A.shape[i]`` is not sympy syntax, so the whole token used to fail to parse and
    # every compare naming one answered False. It is a fixed extent, not arithmetic: fold it to an
    # atom. Both sides mangle the same way, so the surrounding arithmetic still cancels.
    text = SHAPE_READ_RE.sub(lambda m: f"__shp_{m.group(1)}_{m.group(2)}", text)
    names = {}
    for m in DIM_IDENT_RE.finditer(text):
        if text[m.end():m.end() + 1] != "(":  # a call target is a function, not a dimension
            names[m.group()] = sympy.Symbol(m.group(), integer=True, nonnegative=True)
    try:
        return sympy.sympify(text, locals=names)
    except (SyntaxError, TypeError, AttributeError, ValueError, IndexError, sympy.SympifyError):
        # ``sympify`` EVALUATES the token, so any exception the expression can raise is on this
        # path: a token that reads past the end of a tuple literal arrives as a bare ``IndexError``
        # from inside sympy's parser. Unresolvable is None, which the callers read as "not equal"
        # and decline on -- the safe direction, per :func:`dims_agree`.
        return None


#: Integer points the numeric refutation evaluates at. Two of them, because a single point lets a
#: pair coincide by accident (``2 * a`` and ``a + 7`` both give 14 at ``a = 7``), and fixed rather
#: than random so a verdict never depends on the run.
DIM_PROBE_POINTS: Tuple[Tuple[int, ...], ...] = ((7, 11, 13, 17, 19, 23, 29, 31), (3, 41, 5, 37, 2, 43, 11, 47))


def shape_exprs_differ_numerically(ea, eb) -> bool:
    """``True`` when the two expressions disagree at one integer point, which REFUTES equality.

    Shape tokens agree when they agree as FUNCTIONS of their symbols, so one disagreeing
    assignment settles the question -- in microseconds, against the hundreds of milliseconds
    ``simplify`` spends reaching the same verdict on a conv extent like ``(h - k) // s + 1``.
    Agreement at a point proves nothing and falls through to the symbolic rung, so this can only
    ever answer False faster, never answer True.
    """
    symbols = sorted(ea.free_symbols | eb.free_symbols, key=str)
    if not symbols or len(symbols) > len(DIM_PROBE_POINTS[0]):
        return False
    for values in DIM_PROBE_POINTS:
        point = dict(zip(symbols, values))
        try:
            va, vb = ea.subs(point), eb.subs(point)
        except (TypeError, ValueError, ZeroDivisionError):
            return False
        if not (va.is_number and vb.is_number):
            return False
        if va != vb:
            return True
    return False


@lru_cache(maxsize=None, typed=True)
def shape_exprs_equal(sa: str, sb: str) -> bool:
    """``True`` when two already alias-substituted shape expressions denote the same extent.

    Split out of :func:`dims_agree` so the symbolic work is memoized. :func:`reshape_axis_groups`
    asks about the same pair once per axis of every reshape, and a densenet-sized model reaches
    this rung thousands of times over a vocabulary of a few dozen tokens -- profiled at 61% of the
    whole lowering before the cache. Pure in its arguments: the alias table that produced the two
    strings is already folded into them.

    Four rungs, and ``simplify`` -- measured at 784 ms on one conv-extent pair -- is the last.
    ``expand`` PROVES the linear cases (swin's ``4 * (4 * embed_dim)`` against ``16 * embed_dim``);
    a non-zero expansion of a POLYNOMIAL difference is already conclusive, so those need nothing
    further; and :func:`shape_exprs_differ_numerically` refutes the rest in microseconds. Every
    rung answers exactly what ``simplify`` alone answered -- this is a cost cut, not a semantic
    change.
    """
    import sympy  # Deferred, as in sympify_shape.
    ea, eb = sympify_shape(sa), sympify_shape(sb)
    if ea is None or eb is None:
        return False
    try:
        diff = ea - eb
        if sympy.expand(diff) == 0:
            return True
        if diff.is_polynomial():
            return False  # An expanded polynomial is canonical: non-zero here means non-zero.
        if shape_exprs_differ_numerically(ea, eb):
            return False
        return bool(sympy.simplify(diff) == 0)
    except (TypeError, AttributeError, ValueError):
        return False


def dims_agree(a: str,
               b: str,
               aliases: Optional[Dict[str, str]] = None,
               shape_table: Optional[Dict[str, Tuple[str, ...]]] = None) -> bool:
    """``True`` when two shape tokens denote the same extent.

    Three rungs, cheapest first, because the first answers nearly every call: literal string
    equality; equality after :func:`substitute_dim_aliases` puts both in the parameter vocabulary;
    and only then a symbolic compare, for the case where substitution leaves arithmetically-equal
    but textually different expressions (swin's ``4 * (4 * embed_dim)`` against ``16 * embed_dim``).

    Unresolvable is FALSE, never True: a wrong ``True`` here contracts over two different extents,
    which is a miscompile, while a wrong ``False`` only declines a matmul the hoister then refuses.
    """
    if a == b:
        return True
    if not aliases and not shape_table:
        return False
    sa = substitute_dim_aliases(a, aliases or {}, shape_table)
    sb = substitute_dim_aliases(b, aliases or {}, shape_table)
    if sa == sb:
        return True
    # Order-insensitive key: both rungs above are symmetric and so is a zero difference, so
    # ``(a, b)`` and ``(b, a)`` must not occupy two cache entries.
    return shape_exprs_equal(sa, sb) if sa <= sb else shape_exprs_equal(sb, sa)


def _matmul_result_shape(a_shape: Tuple[str, ...],
                         b_shape: Tuple[str, ...],
                         dim_aliases: Optional[Dict[str, str]] = None,
                         shape_table: Optional[Dict[str, Tuple[str, ...]]] = None) -> Optional[Tuple[str, ...]]:
    """``A @ B``'s result shape under numpy broadcasting rules. Supports 1-D x
    2-D / 2-D x 1-D / 2-D x 2-D; batched ``(*batch, m, k) @ (k, n) -> (*batch,
    m, n)`` (rank(a) >= 3, rank(b) == 2) and its mirror (rank(a) == 2, rank(b)
    >= 3); and both-batched ``(*batch, m, k) @ (*batch, k, n) -> (*batch, m,
    n)`` when both ranks are >= 3 and share the same leading batch dims.

    Dimension agreement goes through :func:`dims_agree` rather than ``==``: the two operands'
    tokens come from different vocabularies (a body local vs an ``init.shapes`` symbol), so string
    identity declines contractions whose extents match. ``dim_aliases`` is what reconciles them.
    """
    # Normalise first: a PARAMETER's shape arrives as a list and a hoisted temp's as a tuple, so the
    # ``a_shape[:-2] == b_shape[:-2]`` batch test below was comparing a list against a tuple and
    # always answering False -- every batched matmul mixing the two was silently declined.
    a_shape, b_shape = tuple(a_shape), tuple(b_shape)

    def agree(x: str, y: str) -> bool:
        return dims_agree(str(x), str(y), dim_aliases, shape_table)

    if len(a_shape) == 2 and len(b_shape) == 2:
        return (a_shape[0], b_shape[1])
    if len(a_shape) == 2 and len(b_shape) == 1:
        return (a_shape[0], )
    if len(a_shape) == 1 and len(b_shape) == 2:
        return (b_shape[1], )
    if len(a_shape) >= 3 and len(b_shape) >= 3:
        # (*batch, m, k) @ (*batch, k, n) -> (*batch, m, n): identical batch.
        batch_ok = (len(a_shape) == len(b_shape) and all(agree(x, y) for x, y in zip(a_shape[:-2], b_shape[:-2])))
        if batch_ok and agree(a_shape[-1], b_shape[-2]):
            return tuple(a_shape[:-2]) + (a_shape[-2], b_shape[-1])
        return None
    if len(a_shape) >= 3 and len(b_shape) == 2:
        # (*batch, m, k) @ (k, n) -> (*batch, m, n)
        if agree(a_shape[-1], b_shape[0]):
            return tuple(a_shape[:-1]) + (b_shape[1], )
    if len(a_shape) == 2 and len(b_shape) >= 3:
        # (m, k) @ (*batch, k, n) -> (*batch, m, n)
        if agree(a_shape[1], b_shape[-2]):
            return tuple(b_shape[:-2]) + (a_shape[0], b_shape[-1])
    return None


def _hoist_matmul(matmul: ast.BinOp,
                  shape_table: Dict[str, Tuple[str, ...]],
                  temp_arrays: Dict[str, Tuple[str, ...]],
                  temp_counter: List[int],
                  dim_aliases: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], List[ast.stmt]]:
    """Hoist a ``lhs @ rhs`` subexpression to a fresh temp array. Returns
    ``(temp_name, pre_stmts)``: caller substitutes ``temp_name`` for the matmul
    expression and prepends ``pre_stmts`` before the enclosing assignment.
    ``None`` signals an unsupported form (caller falls through to
    ``NotImplementedError``).

    Handles slice operands ``A[i, :j] @ A[:j, j]`` by lowering to a scalar
    accumulator loop (dot-product form) when both operands have 1-D iteration
    extent.
    """
    # Slice-aware matmuls via iteration extent, three forms:
    #   1-D x 1-D -> scalar dot (e.g. ``A[i, :j] @ A[:j, j]``).
    #   1-D x 2-D -> 1-D vector ``out[j] = sum_l a[l] * b[l, j]``.
    #   2-D x 1-D -> 1-D vector ``out[i] = sum_l a[i, l] * b[l]``.
    l_ext = _iter_extent_of(matmul.left, shape_table)
    r_ext = _iter_extent_of(matmul.right, shape_table)
    if (l_ext is not None and r_ext is not None and len(l_ext) == 1 and len(r_ext) == 1):
        temp_counter[0] += 1
        temp = f"__mm{temp_counter[0]}"
        # Scalar temp -- caller declares it as ``double``.
        iter_var = f"__mml{temp_counter[0]}"
        sa = _scalarize_at_iters(matmul.left, [_name(iter_var)], shape_table)
        sb = _scalarize_at_iters(matmul.right, [_name(iter_var)], shape_table)
        stmts = [
            ast.Assign(targets=[_store(temp)], value=_const(0.0)),
            ast.For(target=_store(iter_var),
                    iter=ast.Call(func=_name("range"), args=[l_ext[0]], keywords=[]),
                    body=[
                        ast.AugAssign(target=_store(temp),
                                      op=ast.Add(),
                                      value=ast.BinOp(left=sa, op=ast.Mult(), right=sb))
                    ],
                    orelse=[]),
        ]
        return temp, stmts
    # 1-D x 2-D / 2-D x 1-D slice-form matmul (matrix-vector).
    if (l_ext is not None and r_ext is not None and {len(l_ext), len(r_ext)} == {1, 2}):
        temp_counter[0] += 1
        temp = f"__mm{temp_counter[0]}"
        # Output is 1-D; the shared K axis is the matching extent.
        if len(l_ext) == 1:  # 1-D x 2-D: out[j] = sum_l a[l] * b[l, j]
            k_extent, n_extent = l_ext[0], r_ext[1]
            # Use the FULL extent of the RHS array as the temp shape so
            # the function-scope declaration doesn't depend on a loop
            # variable. The actual iteration uses the dynamic extent.
            shape = (_static_shape_of(matmul.right, 1, shape_table) or _call_to_str(n_extent), )
            temp_arrays[temp] = shape
            shape_table[temp] = shape
            l_iter = _name(f"__mml{temp_counter[0]}")  # k
            out_iter = _name(f"__mmj{temp_counter[0]}")  # j
            sa = _scalarize_at_iters(matmul.left, [l_iter], shape_table)
            sb = _scalarize_at_iters(matmul.right, [l_iter, out_iter], shape_table)
            stmts = [
                ast.For(target=_store(out_iter.id),
                        iter=ast.Call(func=_name("range"), args=[n_extent], keywords=[]),
                        body=[
                            ast.Assign(targets=[ast.Subscript(value=_name(temp), slice=out_iter, ctx=ast.Store())],
                                       value=_const(0.0)),
                            ast.For(target=_store(l_iter.id),
                                    iter=ast.Call(func=_name("range"), args=[k_extent], keywords=[]),
                                    body=[
                                        ast.AugAssign(target=ast.Subscript(value=_name(temp),
                                                                           slice=out_iter,
                                                                           ctx=ast.Store()),
                                                      op=ast.Add(),
                                                      value=ast.BinOp(left=sa, op=ast.Mult(), right=sb))
                                    ],
                                    orelse=[]),
                        ],
                        orelse=[])
            ]
        else:  # 2-D x 1-D
            m_extent, k_extent = l_ext[0], l_ext[1]
            shape = (_static_shape_of(matmul.left, 0, shape_table) or _call_to_str(m_extent), )
            temp_arrays[temp] = shape
            shape_table[temp] = shape
            out_iter = _name(f"__mmi{temp_counter[0]}")
            l_iter = _name(f"__mml{temp_counter[0]}")
            sa = _scalarize_at_iters(matmul.left, [out_iter, l_iter], shape_table)
            sb = _scalarize_at_iters(matmul.right, [l_iter], shape_table)
            stmts = [
                ast.For(target=_store(out_iter.id),
                        iter=ast.Call(func=_name("range"), args=[m_extent], keywords=[]),
                        body=[
                            ast.Assign(targets=[ast.Subscript(value=_name(temp), slice=out_iter, ctx=ast.Store())],
                                       value=_const(0.0)),
                            ast.For(target=_store(l_iter.id),
                                    iter=ast.Call(func=_name("range"), args=[k_extent], keywords=[]),
                                    body=[
                                        ast.AugAssign(target=ast.Subscript(value=_name(temp),
                                                                           slice=out_iter,
                                                                           ctx=ast.Store()),
                                                      op=ast.Add(),
                                                      value=ast.BinOp(left=sa, op=ast.Mult(), right=sb))
                                    ],
                                    orelse=[]),
                        ],
                        orelse=[])
            ]
        return temp, stmts
    # 2-D x 2-D scalarised form: either operand may be a BinOp /
    # Subscript expression instead of a bare Name. Recover their iter
    # extents and scalarise at the matmul loop indices (i, l) / (l, j).
    if (l_ext is not None and r_ext is not None and len(l_ext) == 2 and len(r_ext) == 2
            and not (isinstance(matmul.left, ast.Name) and isinstance(matmul.right, ast.Name))):
        temp_counter[0] += 1
        temp = f"__mm{temp_counter[0]}"
        m_extent, k_extent = l_ext[0], l_ext[1]
        _, n_extent = r_ext
        shape = (_static_shape_of(matmul.left, 0, shape_table)
                 or _call_to_str(m_extent), _static_shape_of(matmul.right, 1, shape_table) or _call_to_str(n_extent))
        temp_arrays[temp] = shape
        shape_table[temp] = shape
        i_iter = _name(f"__mmi{temp_counter[0]}")
        j_iter = _name(f"__mmj{temp_counter[0]}")
        l_iter = _name(f"__mml{temp_counter[0]}")
        sa = _scalarize_at_iters(matmul.left, [i_iter, l_iter], shape_table)
        sb = _scalarize_at_iters(matmul.right, [l_iter, j_iter], shape_table)
        out_sub = ast.Tuple(elts=[i_iter, j_iter], ctx=ast.Load())
        stmts = [
            ast.For(target=_store(i_iter.id),
                    iter=ast.Call(func=_name("range"), args=[m_extent], keywords=[]),
                    body=[
                        ast.For(target=_store(j_iter.id),
                                iter=ast.Call(func=_name("range"), args=[n_extent], keywords=[]),
                                body=[
                                    ast.Assign(
                                        targets=[ast.Subscript(value=_name(temp), slice=out_sub, ctx=ast.Store())],
                                        value=_const(0.0)),
                                    ast.For(target=_store(l_iter.id),
                                            iter=ast.Call(func=_name("range"), args=[k_extent], keywords=[]),
                                            body=[
                                                ast.AugAssign(target=ast.Subscript(value=_name(temp),
                                                                                   slice=out_sub,
                                                                                   ctx=ast.Store()),
                                                              op=ast.Add(),
                                                              value=ast.BinOp(left=sa, op=ast.Mult(), right=sb))
                                            ],
                                            orelse=[]),
                                ],
                                orelse=[])
                    ],
                    orelse=[])
        ]
        return temp, stmts
    # BATCHED scalarised form -- the batched counterpart of the 2-D x 2-D branch above. The
    # bare-Name batched path further down reads both operands' declared shapes, so it declines the
    # moment one is an expression: conv_transpose2d's ``xg_flat @ wg[:, :, ky, kx]`` is
    # (n, h*w, in_per_group) @ (in_per_group, out_per_group), and declining it left the contraction
    # to slice fusion, which refuses. Extents come from ``_iter_extent_of`` here, which reads
    # through the slice, so the operand's spelling stops mattering.
    if (l_ext is not None and r_ext is not None and max(len(l_ext), len(r_ext)) >= 3
            and min(len(l_ext), len(r_ext)) >= 2
            and not (isinstance(matmul.left, ast.Name) and isinstance(matmul.right, ast.Name))):
        # Both operands batched must agree on the batch RANK: numpy would broadcast a mismatch,
        # and the bare-Name path below does not model that either. Refuse rather than guess.
        if len(l_ext) >= 3 and len(r_ext) >= 3 and len(l_ext) != len(r_ext):
            return None, []
        temp_counter[0] += 1
        temp = f"__mm{temp_counter[0]}"
        ctr = temp_counter[0]
        batch_ext = (l_ext if len(l_ext) >= len(r_ext) else r_ext)[:-2]
        batched = matmul.left if len(l_ext) >= len(r_ext) else matmul.right
        m_extent, k_extent = l_ext[-2], l_ext[-1]
        n_extent = r_ext[-1]
        # Declare the temp from STATIC axis tokens where they exist, so the function-scope
        # declaration never names a loop variable (same rule as the branches above).
        shape = tuple(
            _static_shape_of(batched, axis, shape_table) or _call_to_str(ext) for axis, ext in enumerate(batch_ext)) + (
                _static_shape_of(matmul.left,
                                 len(l_ext) - 2, shape_table)
                or _call_to_str(m_extent), _static_shape_of(matmul.right,
                                                            len(r_ext) - 1, shape_table) or _call_to_str(n_extent))
        temp_arrays[temp] = shape
        shape_table[temp] = shape
        batch_iters = [_name(f"__mmb{ctr}_{i}") for i in range(len(batch_ext))]
        i_iter = _name(f"__mmi{ctr}")
        j_iter = _name(f"__mmj{ctr}")
        l_iter = _name(f"__mml{ctr}")
        left_iters = ([*batch_iters] if len(l_ext) >= 3 else []) + [i_iter, l_iter]
        right_iters = ([*batch_iters] if len(r_ext) >= 3 else []) + [l_iter, j_iter]
        sa = _scalarize_at_iters(matmul.left, left_iters, shape_table)
        sb = _scalarize_at_iters(matmul.right, right_iters, shape_table)
        out_sub = ast.Tuple(elts=[*batch_iters, i_iter, j_iter], ctx=ast.Load())
        out_ref = lambda ctx: ast.Subscript(value=_name(temp), slice=copy.deepcopy(out_sub), ctx=ctx)
        body: List[ast.stmt] = [
            ast.For(target=_store(j_iter.id),
                    iter=ast.Call(func=_name("range"), args=[n_extent], keywords=[]),
                    body=[
                        ast.Assign(targets=[out_ref(ast.Store())], value=_const(0.0)),
                        ast.For(target=_store(l_iter.id),
                                iter=ast.Call(func=_name("range"), args=[k_extent], keywords=[]),
                                body=[
                                    ast.AugAssign(target=out_ref(ast.Store()),
                                                  op=ast.Add(),
                                                  value=ast.BinOp(left=sa, op=ast.Mult(), right=sb))
                                ],
                                orelse=[]),
                    ],
                    orelse=[])
        ]
        body = [
            ast.For(target=_store(i_iter.id),
                    iter=ast.Call(func=_name("range"), args=[m_extent], keywords=[]),
                    body=body,
                    orelse=[])
        ]
        for iter_node, ext in zip(reversed(batch_iters), reversed(batch_ext)):
            body = [
                ast.For(target=_store(iter_node.id),
                        iter=ast.Call(func=_name("range"), args=[ext], keywords=[]),
                        body=body,
                        orelse=[])
            ]
        return temp, body
    if not (isinstance(matmul.left, ast.Name) and isinstance(matmul.right, ast.Name)):
        return None, []
    a_name, b_name = matmul.left.id, matmul.right.id
    a_shape = shape_table.get(a_name)
    b_shape = shape_table.get(b_name)
    if not a_shape or not b_shape:
        return None, []
    result_shape = _matmul_result_shape(a_shape, b_shape, dim_aliases, shape_table)
    if result_shape is None:
        return None, []

    temp_counter[0] += 1
    temp = f"__mm{temp_counter[0]}"
    temp_arrays[temp] = result_shape
    shape_table[temp] = result_shape

    # Batched matmul ``(*batch, m, k) @ (k, n) -> (*batch, m, n)``: wrap a plain
    # 2-D matmul body in a loop nest over the batch dims, indexing the LHS by
    # ``[*batch, m, k]`` and writing the temp by ``[*batch, m, n]``. Same shape
    # for ``(m, k) @ (*batch, k, n)``.
    if (len(a_shape) >= 3 and len(b_shape) == 2) or \
       (len(a_shape) == 2 and len(b_shape) >= 3) or \
       (len(a_shape) >= 3 and len(b_shape) >= 3):
        if not (isinstance(matmul.left, ast.Name) and isinstance(matmul.right, ast.Name)):
            return None, []
        a_name_b, b_name_b = matmul.left.id, matmul.right.id
        ctr = temp_counter[0]
        # Which side(s) carry the batch dims. Both-batched broadcasts the SAME
        # batch index into both operands; one-sided indexes only that operand.
        a_batch = len(a_shape) >= 3
        b_batch = len(b_shape) >= 3
        if a_batch:
            batch_shape = a_shape[:-2]
            m, k = a_shape[-2], a_shape[-1]
            n = b_shape[-1]
        else:
            batch_shape = b_shape[:-2]
            m, k = a_shape
            n = b_shape[-1]
        batch_iters = [f"__mmb{ctr}_{i}" for i in range(len(batch_shape))]
        i_iter, j_iter, l_iter = f"__mmi{ctr}", f"__mmj{ctr}", f"__mml{ctr}"
        batch_names = [_name(b) for b in batch_iters]
        # Each operand's subscript is prefixed with the batch iters iff that
        # operand is batched; the output is always batched.
        a_sub_elts = ((batch_names if a_batch else []) + [_name(i_iter), _name(l_iter)])
        b_sub_elts = ((batch_names if b_batch else []) + [_name(l_iter), _name(j_iter)])
        out_sub_elts = batch_names + [_name(i_iter), _name(j_iter)]
        out_sub = ast.Tuple(elts=out_sub_elts, ctx=ast.Load())
        a_sub = (ast.Tuple(elts=a_sub_elts, ctx=ast.Load()) if len(a_sub_elts) > 1 else a_sub_elts[0])
        b_sub = (ast.Tuple(elts=b_sub_elts, ctx=ast.Load()) if len(b_sub_elts) > 1 else b_sub_elts[0])
        # Innermost: out[*batch, i, j] = 0; for l: out += a[*] * b[*].
        zero_assign = ast.Assign(targets=[ast.Subscript(value=_name(temp), slice=out_sub, ctx=ast.Store())],
                                 value=_const(0.0))
        accum = ast.AugAssign(target=ast.Subscript(value=_name(temp), slice=out_sub, ctx=ast.Store()),
                              op=ast.Add(),
                              value=ast.BinOp(left=ast.Subscript(value=_name(a_name_b), slice=a_sub, ctx=ast.Load()),
                                              op=ast.Mult(),
                                              right=ast.Subscript(value=_name(b_name_b), slice=b_sub, ctx=ast.Load())))
        l_loop = ast.For(target=_store(l_iter),
                         iter=ast.Call(func=_name("range"), args=[_const_or_name(k)], keywords=[]),
                         body=[accum],
                         orelse=[])
        j_loop = ast.For(target=_store(j_iter),
                         iter=ast.Call(func=_name("range"), args=[_const_or_name(n)], keywords=[]),
                         body=[zero_assign, l_loop],
                         orelse=[])
        i_loop = ast.For(target=_store(i_iter),
                         iter=ast.Call(func=_name("range"), args=[_const_or_name(m)], keywords=[]),
                         body=[j_loop],
                         orelse=[])
        # Wrap with the batch loops, outermost first.
        current: ast.stmt = i_loop
        for bi, bdim in zip(reversed(batch_iters), reversed(list(batch_shape))):
            current = ast.For(target=_store(bi),
                              iter=ast.Call(func=_name("range"), args=[_const_or_name(bdim)], keywords=[]),
                              body=[current],
                              orelse=[])
        return temp, [current]

    # Emit the matmul loop nest that fills ``temp``.
    stmts: List[ast.stmt] = []
    if len(a_shape) == 2 and len(b_shape) == 2:
        m, k = a_shape
        _, n = b_shape
        stmts.append(
            ast.For(target=_store("__i"),
                    iter=ast.Call(func=_name("range"), args=[_const_or_name(m)], keywords=[]),
                    body=[
                        ast.For(
                            target=_store("__j"),
                            iter=ast.Call(func=_name("range"), args=[_const_or_name(n)], keywords=[]),
                            body=[
                                ast.Assign(targets=[
                                    ast.Subscript(value=_name(temp),
                                                  slice=ast.Tuple(elts=[_name("__i"), _name("__j")], ctx=ast.Load()),
                                                  ctx=ast.Store())
                                ],
                                           value=_const(0.0)),
                                ast.For(target=_store("__l"),
                                        iter=ast.Call(func=_name("range"), args=[_const_or_name(k)], keywords=[]),
                                        body=[
                                            ast.AugAssign(
                                                target=ast.Subscript(value=_name(temp),
                                                                     slice=ast.Tuple(elts=[_name("__i"),
                                                                                           _name("__j")],
                                                                                     ctx=ast.Load()),
                                                                     ctx=ast.Store()),
                                                op=ast.Add(),
                                                value=ast.BinOp(left=ast.Subscript(
                                                    value=_name(a_name),
                                                    slice=ast.Tuple(elts=[_name("__i"), _name("__l")], ctx=ast.Load()),
                                                    ctx=ast.Load()),
                                                                op=ast.Mult(),
                                                                right=ast.Subscript(value=_name(b_name),
                                                                                    slice=ast.Tuple(elts=[
                                                                                        _name("__l"),
                                                                                        _name("__j")
                                                                                    ],
                                                                                                    ctx=ast.Load()),
                                                                                    ctx=ast.Load())))
                                        ],
                                        orelse=[])
                            ],
                            orelse=[])
                    ],
                    orelse=[]))
    elif len(a_shape) == 2 and len(b_shape) == 1:
        m, k = a_shape
        stmts.append(
            ast.For(target=_store("__i"),
                    iter=ast.Call(func=_name("range"), args=[_const_or_name(m)], keywords=[]),
                    body=[
                        ast.Assign(targets=[ast.Subscript(value=_name(temp), slice=_name("__i"), ctx=ast.Store())],
                                   value=_const(0.0)),
                        ast.For(target=_store("__l"),
                                iter=ast.Call(func=_name("range"), args=[_const_or_name(k)], keywords=[]),
                                body=[
                                    ast.AugAssign(target=ast.Subscript(value=_name(temp),
                                                                       slice=_name("__i"),
                                                                       ctx=ast.Store()),
                                                  op=ast.Add(),
                                                  value=ast.BinOp(left=ast.Subscript(
                                                      value=_name(a_name),
                                                      slice=ast.Tuple(elts=[_name("__i"), _name("__l")],
                                                                      ctx=ast.Load()),
                                                      ctx=ast.Load()),
                                                                  op=ast.Mult(),
                                                                  right=ast.Subscript(value=_name(b_name),
                                                                                      slice=_name("__l"),
                                                                                      ctx=ast.Load())))
                                ],
                                orelse=[])
                    ],
                    orelse=[]))
    else:  # len(a)==1, len(b)==2
        k, n = b_shape
        stmts.append(
            ast.For(target=_store("__j"),
                    iter=ast.Call(func=_name("range"), args=[_const_or_name(n)], keywords=[]),
                    body=[
                        ast.Assign(targets=[ast.Subscript(value=_name(temp), slice=_name("__j"), ctx=ast.Store())],
                                   value=_const(0.0)),
                        ast.For(target=_store("__l"),
                                iter=ast.Call(func=_name("range"), args=[_const_or_name(k)], keywords=[]),
                                body=[
                                    ast.AugAssign(target=ast.Subscript(value=_name(temp),
                                                                       slice=_name("__j"),
                                                                       ctx=ast.Store()),
                                                  op=ast.Add(),
                                                  value=ast.BinOp(left=ast.Subscript(value=_name(a_name),
                                                                                     slice=_name("__l"),
                                                                                     ctx=ast.Load()),
                                                                  op=ast.Mult(),
                                                                  right=ast.Subscript(
                                                                      value=_name(b_name),
                                                                      slice=ast.Tuple(elts=[_name("__l"),
                                                                                            _name("__j")],
                                                                                      ctx=ast.Load()),
                                                                      ctx=ast.Load())))
                                ],
                                orelse=[])
                    ],
                    orelse=[]))
    return temp, stmts


class _MatmulHoister(ast.NodeTransformer):
    """Replace ``A @ B`` subexpressions with a fresh temp Name and record the
    matmul loop nest that fills the temp. Multiple matmuls in one expression
    each get their own temp (chained ``A @ B @ C`` lifts to two temps fused
    left-to-right)."""

    def __init__(self, shape_table, temp_arrays, temp_counter, local_dtypes=None, sparse=None, dim_aliases=None):
        self.shape_table = shape_table
        self.temp_arrays = temp_arrays
        self.temp_counter = temp_counter
        self.local_dtypes: Dict[str, str] = (local_dtypes if local_dtypes is not None else {})
        #: Dimension local -> its definition, so a contraction whose operands spell the same extent
        #: two ways (``channels`` vs ``embed_dim``) is recognised instead of declined.
        self.dim_aliases: Dict[str, str] = dim_aliases or {}
        #: Logical-name -> SparseArrayDesc (from KernelIR.sparse). When
        #: a matmul's operands are sparse, route to the sparse emitter.
        self.sparse: Dict[str, object] = sparse or {}
        self.pre_stmts: List[ast.stmt] = []

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.MatMult):
            # Sparse path: both operands are logical sparse arrays.
            sp = self._try_hoist_sparse_matmul(node)
            if sp is not None:
                temp, stmts = sp
                self.pre_stmts.extend(self._prepend_alloc_markers(stmts))
                return ast.Name(id=temp, ctx=ast.Load())
            node = self._materialise_call_operands(node)
            temp, stmts = _hoist_matmul(node, self.shape_table, self.temp_arrays, self.temp_counter, self.dim_aliases)
            if temp is not None:
                self.pre_stmts.extend(self._prepend_alloc_markers(stmts))
                # Propagate complex dtype across the matmul: if
                # either operand carries a complex tag (or a complex
                # Constant somewhere in the subtree), tag the matmul
                # temp ``__mm<n>`` so its decl is the right C type.
                for sub in ast.walk(node):
                    if (isinstance(sub, ast.Constant) and isinstance(sub.value, complex)):
                        self.local_dtypes[temp] = "complex128"
                        break
                    if isinstance(sub, ast.Name):
                        dt = self.local_dtypes.get(sub.id)
                        if dt and dt.startswith("complex"):
                            self.local_dtypes[temp] = "complex128"
                            break
                return ast.Name(id=temp, ctx=ast.Load())
        return node

    def _materialise_call_operands(self, node: ast.BinOp) -> ast.BinOp:
        """Spill a CALL-valued matmul operand to a temp array, so the hoister sees a bare Name.

        ``relu_self_attention`` writes ``np.maximum(scores, 0.0) @ v``. The elementwise call has a
        perfectly well-defined extent, but the loop nest below indexes its operands by name, so the
        matmul was declined -- and a declined matmul reaches slice fusion, where scalarising it
        would drop the contraction. Materialising is what numpy does anyway; the guard downstream
        stays exactly as strict.

        Two things are deliberately left alone, on the same principle -- do not reroute what
        already lowers. A ``Subscript`` operand has its own slice-aware path, and a RANK-1 operand
        reaches the scalar dot-product form, which reads a call operand happily via
        ``_iter_extent_of``; spilling either would trade a working lowering for an extra temp
        array and a copy loop.
        """
        left, right = node.left, node.right
        for side in ("left", "right"):
            operand = left if side == "left" else right
            if not isinstance(operand, ast.Call):
                continue
            ext = _iter_extent_of(operand, self.shape_table)
            if ext is None or len(ext) < 2:
                continue
            nm, stmts = self._materialise_dense_operand(operand)
            if nm is None:
                continue
            self.pre_stmts.extend(self._prepend_alloc_markers(stmts))
            if side == "left":
                left = _name(nm)
            else:
                right = _name(nm)
        if left is node.left and right is node.right:
            return node
        return ast.BinOp(left=left, op=ast.MatMult(), right=right)

    def _prepend_alloc_markers(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
        """Prepend a ``__hpcagent_bench_zeros__()`` allocation marker for each array
        temp written in ``stmts`` (first-write order). A matmul/column-slice
        temp whose shape depends on a body-computed scalar (gmres ``n``/``m``)
        can't be malloc'd at fn-top -- the marker defers its malloc to this
        site, which always follows the scalar's assignment. For a
        param-shaped temp already malloc'd at fn-top the marker is a no-op, so
        prepending one unconditionally is safe.
        """
        seen: List[str] = []
        for s in stmts:
            for sub in ast.walk(s):
                tgt = None
                if isinstance(sub, ast.Assign) and sub.targets:
                    tgt = sub.targets[0]
                elif isinstance(sub, ast.AugAssign):
                    tgt = sub.target
                if tgt is None:
                    continue
                while isinstance(tgt, ast.Subscript):
                    tgt = tgt.value
                if (isinstance(tgt, ast.Name) and tgt.id in self.temp_arrays and tgt.id not in seen):
                    seen.append(tgt.id)
        return [_alloc_marker(n) for n in seen] + stmts

    def _try_hoist_sparse_matmul(self, node: ast.BinOp):
        """Route ``A @ B`` through the sparse emitter when an operand carries a
        sparse layout. Returns ``(temp_name, stmts)`` for the fresh result
        temp, or ``None`` when neither operand is sparse (dense path handles
        it).

        Type algebra (raises ``NotImplementedError`` on unsupported combos, so
        a clear failure surfaces at lowering rather than silent wrong
        numerics): ``sparse @ dense``/``dense @ sparse`` -> **dense** (matvec
        if the dense operand is 1-D, matmat -- CSR only for now -- if 2-D);
        ``csr @ csr`` -> **dense** result temp (the surrounding ``alpha *
        (A@B) + beta * C`` densifies it anyway, matching scipy's ``sparse @
        sparse + dense``; a CSR-output Gustavson form exists separately for
        pure-SpGEMM kernels); every other ``sparse @ sparse`` combo errors.

        This is the C/Fortran realisation of
        :func:`numpyto_common.sparse_emit.result_layout`: every supported case
        here densifies, matching ``result_layout(..., target="c") == DENSE`` --
        the hoister runs in the dense-accumulation context, so it always
        densifies rather than emitting a CSR-output SpGEMM.
        """
        if not self.sparse:
            return None
        # A sparse operand is always a bare logical Name (slicing a CSR buffer
        # set is unsupported). When exactly one operand is sparse and the other
        # is a non-Name dense expression -- e.g. GMRES's column slice ``A @
        # Q[:, k]`` -- materialise the dense operand into a fresh temp array so
        # the SpMV/SpMM expanders (which require a declared dense array) can
        # consume it.
        pre: List[ast.stmt] = []
        # Sparse TRANSPOSE matvec ``A.T @ x`` (bicg's ``A.T @ p_tilde``): A's
        # CSR buffers are exactly A.T's CSC buffers (and vice versa), and COO
        # transposes by swapping row/col roles -- so a transpose reuses the
        # same physical buffers under the dual format, no extra data.
        tr = self._transpose_sparse_desc(node.left)
        if tr is not None and isinstance(node.right, ast.Name) and node.right.id not in self.sparse:
            td, transposed = tr
            dense_shape = self.shape_table.get(node.right.id)
            if dense_shape and len(dense_shape) == 1:
                self.temp_counter[0] += 1
                temp = f"__mm{self.temp_counter[0]}"
                n_rows = td.logical_shape[0] if td.logical_shape else "0"
                self.temp_arrays[temp] = (n_rows, )
                self.shape_table[temp] = (n_rows, )
                return temp, pre + self._sparse_matvec(td, node.right.id, temp, transposed=transposed)
        l_sparse = (isinstance(node.left, ast.Name) and node.left.id in self.sparse)
        r_sparse = (isinstance(node.right, ast.Name) and node.right.id in self.sparse)
        if not (l_sparse or r_sparse):
            return None  # neither operand is a sparse Name -- dense path
        if l_sparse and not isinstance(node.right, ast.Name):
            nm, stmts = self._materialise_dense_operand(node.right, max_rank=1)
            if nm is None:
                return None
            pre.extend(stmts)
            node = ast.BinOp(left=node.left, op=ast.MatMult(), right=_name(nm))
        elif r_sparse and not isinstance(node.left, ast.Name):
            nm, stmts = self._materialise_dense_operand(node.left, max_rank=1)
            if nm is None:
                return None
            pre.extend(stmts)
            node = ast.BinOp(left=_name(nm), op=ast.MatMult(), right=node.right)
        if not (isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name)):
            return None
        la = self.sparse.get(node.left.id)
        ra = self.sparse.get(node.right.id)
        if la is None and ra is None:
            return None  # purely dense -- not our path
        from numpyto_common import sparse_emit as _se

        # sparse @ sparse
        if la is not None and ra is not None:
            lfmt, rfmt = la.format, ra.format
            if lfmt == "csr" and rfmt == "csr":
                self.temp_counter[0] += 1
                temp = f"__mm{self.temp_counter[0]}"
                ni = la.logical_shape[0] if la.logical_shape else "0"
                nj = (ra.logical_shape[1] if len(ra.logical_shape) > 1 else
                      (ra.logical_shape[0] if ra.logical_shape else "0"))
                self.temp_arrays[temp] = (ni, nj)
                self.shape_table[temp] = (ni, nj)
                stmts = _se.expand_matmul_csr_csr_dense(temp, la.buffers, ra.buffers, ni, nj)
                return temp, pre + stmts
            raise NotImplementedError(f"sparse @ sparse only supports csr @ csr; got "
                                      f"{lfmt} @ {rfmt} ({node.left.id} @ {node.right.id}). "
                                      "Convert operands to CSR or split the kernel.")

        # sparse @ dense / dense @ sparse -- exactly one operand is sparse.
        if la is not None:
            sp_desc, dense_name, sp_on_left = la, node.right.id, True
        else:
            sp_desc, dense_name, sp_on_left = ra, node.left.id, False
        dense_shape = self.shape_table.get(dense_name)
        rank = len(dense_shape) if dense_shape else None
        if rank == 1:
            # matvec: sparse (M x N) @ dense (N,) -> dense (M,).
            if not sp_on_left:
                raise NotImplementedError("dense (1-D) @ sparse is a row-vector times matrix; "
                                          "not supported -- write it as sparse.T @ x.")
            self.temp_counter[0] += 1
            temp = f"__mm{self.temp_counter[0]}"
            n_rows = sp_desc.logical_shape[0] if sp_desc.logical_shape else "0"
            self.temp_arrays[temp] = (n_rows, )
            self.shape_table[temp] = (n_rows, )
            stmts = self._sparse_matvec(sp_desc, dense_name, temp)
            return temp, pre + stmts
        # matmat sparse @ dense (2-D) -> dense -- CSR only for now.
        if rank == 2 and sp_on_left and sp_desc.format == "csr":
            self.temp_counter[0] += 1
            temp = f"__mm{self.temp_counter[0]}"
            n_rows = sp_desc.logical_shape[0] if sp_desc.logical_shape else "0"
            n_cols = dense_shape[1]
            self.temp_arrays[temp] = (n_rows, n_cols)
            self.shape_table[temp] = (n_rows, n_cols)
            stmts = _se.expand_matmul_csr_dense_mat(temp, sp_desc.buffers, dense_name, n_rows, n_cols)
            return temp, pre + stmts
        raise NotImplementedError(f"sparse @ dense for format {sp_desc.format} with dense rank "
                                  f"{rank} not supported ({node.left.id} @ {node.right.id}).")

    def _materialise_dense_operand(self, expr: ast.expr, max_rank: Optional[int] = None):
        """Copy a non-Name dense matmul operand into a fresh temp array, so the consumer -- which
        requires a *declared* array -- sees a bare Name. Two callers want this: the SpMV/SpMM
        expanders, for a column slice like ``Q[:, k]`` in ``A @ Q[:, k]``, and the dense hoister,
        for a call-valued operand like ``np.maximum(scores, 0.0) @ v``. Returns ``(temp_name,
        stmts)`` filling the temp, or ``(None, [])`` when the extent doesn't resolve.

        ``max_rank`` bounds what is accepted. The sparse path passes 1 on purpose: a 2-D dense
        slice on the sparse side (SpMM with a sliced RHS) must fall through and fail loudly rather
        than emit wrong shapes.
        """
        ext = _iter_extent_of(expr, self.shape_table)
        if not ext:
            return None, []
        if max_rank is not None and len(ext) > max_rank:
            return None, []
        self.temp_counter[0] += 1
        n = self.temp_counter[0]
        temp = f"__spv{n}"
        # Carry a complex dtype tag from any complex base array so the
        # temp's C decl matches (real default otherwise).
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Name):
                dt = self.local_dtypes.get(sub.id)
                if dt and dt.startswith("complex"):
                    self.local_dtypes[temp] = "complex128"
                    break
        shape = tuple((_static_shape_of(expr, ax, self.shape_table) or _call_to_str(e)) for ax, e in enumerate(ext))
        self.temp_arrays[temp] = shape
        self.shape_table[temp] = shape
        iters = [_name(f"__spvi{n}_{ax}") for ax in range(len(ext))]
        elem = _scalarize_at_iters(expr, iters, self.shape_table)
        sub_slice = (ast.Tuple(elts=list(iters), ctx=ast.Load()) if len(iters) > 1 else iters[0])
        body: ast.stmt = ast.Assign(targets=[ast.Subscript(value=_name(temp), slice=sub_slice, ctx=ast.Store())],
                                    value=elem)
        for it, extent in zip(reversed(iters), reversed(list(ext))):
            body = ast.For(target=_store(it.id),
                           iter=ast.Call(func=_name("range"), args=[extent], keywords=[]),
                           body=[body],
                           orelse=[])
        return temp, [body]

    def _transpose_sparse_desc(self, operand):
        """If ``operand`` is ``A.T`` for a sparse ``A``, return ``(desc, transposed)`` describing
        ``A.T`` so the matvec dispatcher emits ``A.T @ x`` directly; ``None`` otherwise.

        Two kinds of transpose, and the flag says which:

        * **relabelled** (``transposed=False``) -- CSR's buffers ARE its transpose's CSC buffers
          (and vice versa), and COO transposes by swapping the row/col roles. The descriptor alone
          carries the transpose, so the forward per-format matvec is correct as-is.
        * **access-transposed** (``transposed=True``) -- DIA and BCSR have no dual descriptor over
          the same buffers (negating DIA's offsets re-keys its data columns; BCSR would need
          indptr/indices rebuilt over block columns AND every block transposed). The format stays
          put and the ``*_t`` matvec transposes the ACCESS instead.

        ``logical_shape`` is reversed either way, so the caller sizes the result temp off the
        transpose's own row count without knowing which kind it got.
        """
        if not (isinstance(operand, ast.Attribute) and operand.attr == "T" and isinstance(operand.value, ast.Name)
                and operand.value.id in self.sparse):
            return None
        from numpyto_common.ir import SparseArrayDesc
        d = self.sparse[operand.value.id]
        ls = list(d.logical_shape) if d.logical_shape else []
        swapped = tuple(reversed(ls)) if len(ls) >= 2 else tuple(ls)
        dual = {"csr": "csc", "csc": "csr"}.get(d.format)
        if dual is not None:
            return SparseArrayDesc(name=d.name, format=dual, logical_shape=swapped, buffers=dict(d.buffers)), False
        if d.format == "coo":
            b = dict(d.buffers)
            if "row" in b and "col" in b:
                b["row"], b["col"] = d.buffers["col"], d.buffers["row"]
            return SparseArrayDesc(name=d.name, format="coo", logical_shape=swapped, buffers=b), False
        if d.format in ("dia", "bcsr"):
            return SparseArrayDesc(name=d.name, format=d.format, logical_shape=swapped, buffers=dict(d.buffers)), True
        return None

    def _sparse_matvec(self, sp_desc, dense_name: str, temp: str, transposed: bool = False):
        """Build the per-format matvec loop nest filling 1-D ``temp``.

        Derives each format's extra size symbols from the sparse
        descriptor's logical shape + physical buffer shapes, then calls
        the matching dispatcher in ``sparse_emit``.

        ``transposed`` marks the DIA/BCSR ``A.T @ x`` case, where the descriptor is A's own
        (only its logical shape is reversed) and the ``*_t`` dispatcher transposes the access --
        see :meth:`_transpose_sparse_desc`. Every other format arrives already relabelled, so a
        forward matvec on the dual descriptor is the transpose.
        """
        from numpyto_common import sparse_emit as _se
        fmt = sp_desc.format
        bufs = sp_desc.buffers
        tgt = _name(temp)
        n_rows = sp_desc.logical_shape[0] if sp_desc.logical_shape else "0"
        n_cols = (sp_desc.logical_shape[1] if len(sp_desc.logical_shape) > 1 else "0")

        def _buf_shape(role, axis):
            """Shape token of the physical buffer for ``role`` at ``axis``,
            looked up from the shape table (physical buffers are declared
            arrays)."""
            phys = bufs.get(role)
            sh = self.shape_table.get(phys) if phys else None
            if sh and axis < len(sh):
                return sh[axis]
            return None

        if fmt == "csr":
            return _se.expand_matmul_csr_dense_vec(tgt, bufs, dense_name, n_rows)
        if fmt == "csc":
            return _se.expand_matmul_csc_dense_vec(tgt, bufs, dense_name, n_rows, n_cols)
        if fmt == "coo":
            nnz = _buf_shape("data", 0) or "0"
            return _se.expand_matmul_coo_dense_vec(tgt, bufs, dense_name, n_rows, nnz)
        if fmt == "dia":
            ndiag = _buf_shape("data", 0) or "0"
            # Transposed: the descriptor's shape is reversed, so A's own (rows, cols) are
            # (n_cols, n_rows) here -- the expanders take A's dims, not A.T's, to stay
            # readable side by side.
            if transposed:
                return _se.expand_matmul_dia_t_dense_vec(tgt, bufs, dense_name, n_cols, n_rows, ndiag)
            return _se.expand_matmul_dia_dense_vec(tgt, bufs, dense_name, n_rows, n_cols, ndiag)
        if fmt == "ell":
            maxnz = _buf_shape("data", 1) or "0"
            return _se.expand_matmul_ell_dense_vec(tgt, bufs, dense_name, n_rows, maxnz)
        if fmt == "jds":
            # njd = len(jd_ptr) - 1
            jdlen = _buf_shape("jd_ptr", 0)
            njd = f"({jdlen}) - 1" if jdlen else "0"
            # The dispatcher uses a sorted-order scratch accumulator but relies
            # on the caller to declare it (see expand_matmul_jds_* docstring):
            # register it as a fresh (n_rows,) local so the emitter allocates
            # it -- the dispatcher zeroes it itself.
            self.temp_arrays["__jds_y_perm"] = (n_rows, )
            self.shape_table["__jds_y_perm"] = (n_rows, )
            return _se.expand_matmul_jds_dense_vec(tgt, bufs, dense_name, n_rows, njd)
        if fmt == "bcsr":
            # block dims live on the descriptor's logical shape vs buffer
            # data shape [nnz_blk, R, C]; n_block_rows = len(indptr) - 1.
            iplen = _buf_shape("indptr", 0)
            nbr = f"({iplen}) - 1" if iplen else "0"
            R = _buf_shape("data", 1) or "1"
            C = _buf_shape("data", 2) or "1"
            if transposed:
                # ``n_rows`` is A.T's row count, i.e. A's COLUMN count -- exactly the length
                # the transposed matvec writes.
                return _se.expand_matmul_bcsr_t_dense_vec(tgt, bufs, dense_name, nbr, R, C, n_rows)
            return _se.expand_matmul_bcsr_dense_vec(tgt, bufs, dense_name, nbr, R, C, n_rows)
        if fmt == "bcoo":
            # block-COO: row[k]/col[k] hold block coords, data is
            # [n_blocks, R, C]; n_blocks = len(row); the total scalar
            # row count is n_rows (the descriptor's logical row dim).
            nblk = _buf_shape("row", 0) or _buf_shape("data", 0) or "0"
            R = _buf_shape("data", 1) or "1"
            C = _buf_shape("data", 2) or "1"
            return _se.expand_matmul_bcoo_dense_vec(tgt, bufs, dense_name, n_rows, nblk, R, C)
        if fmt == "sell_c_sigma":
            nsl = _buf_shape("slice_ptr", 0)
            nslices = f"({nsl}) - 1" if nsl else "0"
            # slice height C is a kernel parameter; default symbol "C".
            return _se.expand_matmul_sell_c_sigma_dense_vec(tgt, bufs, dense_name, n_rows, nslices, "C")
        raise NotImplementedError(f"sparse matvec for format {fmt!r} not supported.")


class _CallHoister(ast.NodeTransformer):
    """Hoist any registered ``np.*`` call buried in an expression to a fresh
    temp ``__cb<n>``; the expander then lowers ``__cb<n> = call(...)``. A
    scalar-returning call (reduction/dot/std) hoists to a scalar local; an
    array-returning call (copy/outer/transpose) hoists to an array temp whose
    shape is inferred from its arguments.
    """

    def __init__(self, shape_table, scalar_temps, array_temps, counter, local_dtypes=None, dim_aliases=None):
        self.shape_table = shape_table
        self.scalar_temps = scalar_temps
        self.array_temps = array_temps
        self.counter = counter
        #: Forwarded to the nested ``_MatmulHoister`` (see its docstring).
        self.dim_aliases: Dict[str, str] = dim_aliases or {}
        # Side-effect dtype table (shared with the lowering pipeline)
        # so a ``__cb<n>`` whose RHS contains complex literals or
        # complex-typed Name references is tagged ``complex128``.
        self.local_dtypes: Dict[str, str] = local_dtypes if local_dtypes is not None else {}
        self.pre_stmts: List[ast.stmt] = []

    def _infer_complex(self, expr: ast.AST) -> bool:
        """``True`` iff ``expr`` reads a complex value (skipping ``.shape`` reads)."""
        return _reads_complex(expr, self.local_dtypes)

    def _key_of(self, call: ast.Call):
        func = call.func
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name):
                return ("np" if func.value.id == "np" else func.value.id, func.attr)
            if (isinstance(func.value, ast.Attribute) and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "np"):
                return ("np", f"{func.value.attr}.{func.attr}")
        return None

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # ``np.repeat(src, np.diff(p))``: the count's telescoping sum (see
        # expand_repeat / _diff_operand) needs the ORIGINAL ``np.diff`` call
        # form. A plain ``generic_visit`` would recurse into it first -- ``np.diff``
        # is itself a registered call, so it would get hoisted into an opaque
        # ``__cb<n>`` temp before the repeat expander ever ran, losing the one
        # piece of syntax that proves the sum is derivable. Visit every other
        # child normally and leave that one argument untouched.
        if (self._key_of(node) == ("np", "repeat") and len(node.args) >= 2 and _diff_operand(node.args[1]) is not None):
            node.func = self.visit(node.func)
            node.args = [(a if i == 1 else self.visit(a)) for i, a in enumerate(node.args)]
            node.keywords = [self.visit(kw) for kw in node.keywords]
        else:
            self.generic_visit(node)
        # Hoist any matmul subexpressions inside the call args first:
        # ``np.maximum(input @ w1 + b1, 0)`` -> ``__mm1 = input @ w1; ...;
        # np.maximum(__mm1 + b1, 0)``, so the elementwise expander sees a bare
        # BinOp on Names, not a MatMult.
        mm = _MatmulHoister(self.shape_table,
                            self.array_temps,
                            self.counter,
                            local_dtypes=self.local_dtypes,
                            sparse=vars(self).get("sparse"),
                            dim_aliases=self.dim_aliases)
        node.args = [mm.visit(a) for a in node.args]
        self.pre_stmts.extend(mm.pre_stmts)
        # Hoist a non-Name first arg of an array reduction (sum/max/min/mean/
        # prod/std/argmax/argmin) into a fresh temp: ``np.mean(a * b)`` ->
        # ``__cb<n> = a * b; np.mean(__cb<n>)`` so the reduction expander sees
        # a Name operand -- likewise ``np.argmax(np.abs(v))`` spills
        # ``np.abs(v)`` before the arg-reduction scaffold (which requires a
        # Name) runs. Shape-preserving index ops (roll/flip/transpose/reshape)
        # join the set too, so ls3df _hpsi's ``np.roll(psi_frag[f], m, axis)``
        # spills ``psi_frag[f]`` and hoists as ``np.roll(__cb<n>, m, axis)`` --
        # otherwise the whole-array roll stays buried in the broadcast BinOp
        # and the per-element scalarizer mangles it into a scalar-arg roll.
        key = self._key_of(node)
        if (key in ({("np", k)
                     for k in {
                         "sum", "max", "min", "mean", "prod", "std", "var", "median", "any", "all", "count_nonzero",
                         "argmax", "argmin", "repeat", "transpose", "reshape", "triu", "tril", "flip", "roll", "copy",
                         "array", "bincount", "cumsum", "cumprod", "swapaxes", "expand_dims", "squeeze", "moveaxis"
                     }}
                    | {("np", "fft.fftn"), ("np", "fft.ifftn"), ("np", "fft.fft"), ("np", "fft.ifft")}) and node.args
                and not isinstance(node.args[0], ast.Name)):
            first = node.args[0]
            ext = _iter_extent_of(first, self.shape_table)
            if ext is not None:
                self.counter[0] += 1
                temp = f"__cb{self.counter[0]}"
                shape = tuple(self._extent_to_shape_token(e) for e in ext)
                self.array_temps[temp] = shape
                self.shape_table[temp] = shape
                if self._infer_complex(first):
                    self.local_dtypes[temp] = "complex128"
                # When ``first`` carries slice-bearing Subscripts (maxpool's
                # ``np.max(x[:, 2i:2i+2, :], axis=(1, 2))``), the
                # post-LibNodeRewriter lift can no longer recover x's
                # per-statement shape -- by then x is overwritten with its
                # final shape. Emit the slice-LHS form instead: marker +
                # ``__cb[:, ...] = first``; slice-fusion lowers this into a
                # per-element copy later.
                if _has_slice_subscript(first):
                    rank = len(shape)
                    slice_form = (ast.Slice(lower=None, upper=None, step=None) if rank == 1 else ast.Tuple(
                        elts=[ast.Slice(lower=None, upper=None, step=None) for _ in range(rank)], ctx=ast.Load()))
                    marker = ast.Assign(targets=[ast.Name(id=temp, ctx=ast.Store())],
                                        value=ast.Call(func=ast.Name(id="__hpcagent_bench_zeros__", ctx=ast.Load()),
                                                       args=[],
                                                       keywords=[]))
                    slice_lhs = ast.Subscript(value=ast.Name(id=temp, ctx=ast.Load()),
                                              slice=slice_form,
                                              ctx=ast.Store())
                    slice_assign = ast.Assign(targets=[slice_lhs], value=first)
                    self.pre_stmts.append(marker)
                    self.pre_stmts.append(slice_assign)
                else:
                    # Synth: ``__cb<n> = first``. The LibNodeRewriter's
                    # _lower_prelude_calls step then turns this into a
                    # per-element copy via _WholeArrayAssignRewriter.
                    self.pre_stmts.append(ast.Assign(targets=[ast.Name(id=temp, ctx=ast.Store())], value=first))
                node.args[0] = ast.Name(id=temp, ctx=ast.Load())
        key = self._key_of(node)
        if key is None or key not in NP_CALL_EXPANDERS:
            return node
        # Stash axis/keepdims kwargs for the reduction case so
        # _derive_output_shape can compute the correct array shape.
        if key == ("np", "linalg.norm"):
            # ``linalg.norm``'s positional layout is ``(v, ord, axis, keepdims)``,
            # unlike a reduction's 2nd-positional ``axis`` -- strip a
            # positional/keyword ``ord`` before reading the axis (mirroring
            # ``expand_linalg_norm``), else a positional ord (``norm(a, 1)``) is
            # misread as ``axis=1`` and an axis-less vector norm is wrongly
            # hoisted as an array.
            norm_args = [node.args[0]] + list(node.args[2:]) if node.args else []
            norm_kwargs = [kw for kw in node.keywords if kw.arg != "ord"]
            self._cur_axis, self._cur_keepdims = _read_axis_keepdims(norm_args, norm_kwargs)
        else:
            self._cur_axis, self._cur_keepdims = _read_axis_keepdims(node.args, node.keywords)
        self.counter[0] += 1
        temp = f"__cb{self.counter[0]}"
        # Classify: scalar return vs array return.
        is_scalar = key[1] in {
            "sum", "max", "min", "mean", "prod", "std", "var", "dot", "vdot", "inner", "linalg.norm", "linalg.det",
            "argmax", "argmin", "any", "all", "count_nonzero", "median", "trace"
        }
        # ``np.inner`` is scalar ONLY for rank-1 x rank-1; higher ranks
        # contract the last axes into an array result.
        if key[1] == "inner":
            ranks = [len(self.shape_table.get(a.id, ())) for a in node.args if isinstance(a, ast.Name)]
            if any(r > 1 for r in ranks):
                is_scalar = False
        # Axis-aware reductions with axis specified return an array. ``var``
        # belongs here for the same reason ``std`` does -- they're one op
        # (``_expand_var_or_std``, std is var plus a sqrt). Omitting it left
        # gpt2_block's layer-norm ``np.var(z, axis=-1, keepdims=True)``
        # classified scalar, so its temp was never sized or declared an array.
        if (is_scalar and key[1] in {
                "sum", "max", "min", "mean", "prod", "std", "var", "argmax", "argmin", "any", "all", "count_nonzero",
                "linalg.norm"
        } and self._cur_axis is not None):
            is_scalar = False
        if is_scalar and node.args and isinstance(node.args[0], ast.Subscript):
            # np.dot on 1-D slices is scalar.
            ext = _iter_extent_of(node.args[0], self.shape_table)
            if ext is not None and len(ext) == 1 and key[1] == "dot":
                is_scalar = True
        if not is_scalar:
            # Array-returning: try to determine the output shape from args.
            shape = self._derive_output_shape(key, node.args, node.keywords)
            if shape is None:
                return node
            self.array_temps[temp] = shape
            self.shape_table[temp] = shape
            # ``argmax``/``argmin`` produce an INDEX array -> int64, not the
            # default double (so the buffer + any store into an int target is
            # an integer, matching numpy's intp result).
            if key[1] in {"argmax", "argmin"}:
                self.local_dtypes[temp] = "int64"
            # Propagate complex dtype when the call's argument tree
            # contains complex literals / complex-Name references.
            # ``np.exp(-2.0j * np.pi * ...)`` etc. land here.
            # Every ``np.fft.*`` transform RETURNS complex even from a real
            # input, so force the output temp complex regardless of operand.
            if self._infer_complex(node) or key[1] in {"fft.fftn", "fft.ifftn", "fft.fft", "fft.ifft"}:
                self.local_dtypes[temp] = "complex128"
            # Shape-preserving ops (``reshape`` / ``repeat`` / ``copy``
            # / ``transpose`` / ``flip``) inherit the source array's
            # dtype: ``Xiv = np.reshape(Xi, (xn * yn,))`` where ``Xi``
            # is int64 must keep Xiv as int64, not the default double.
            SHAPE_PRESERVING = {
                "reshape", "repeat", "copy", "array", "asarray", "ascontiguousarray", "transpose", "flip"
            }
            if (key[1] in SHAPE_PRESERVING and node.args and temp not in self.local_dtypes):
                first = node.args[0]
                if isinstance(first, ast.Name):
                    src_dt = self.local_dtypes.get(first.id)
                    if src_dt:
                        self.local_dtypes[temp] = src_dt
            # An all-integer elementwise ufunc returns an INTEGER array in numpy --
            # declare the temp int64 so an exact int64 result is not round-tripped
            # through a double (which drops every bit above 2**53).
            if (key[1] in _INT_PRESERVING_ELEMENTWISE and temp not in self.local_dtypes
                    and _all_integer_operands(node.args, self.local_dtypes)):
                self.local_dtypes[temp] = "int64"
            # ``np.where`` promotes its two VALUE operands and ignores the condition's dtype, so an
            # integer select stays integer -- the last hop of bitonic_sort's comparator network,
            # whose int64 payload was otherwise handed back through a float temp.
            if (key[1] == "where" and len(node.args) == 3 and temp not in self.local_dtypes
                    and _all_integer_operands(node.args[1:], self.local_dtypes)):
                self.local_dtypes[temp] = "int64"
        else:
            self.scalar_temps[temp] = True
            if self._infer_complex(node):
                self.local_dtypes[temp] = "complex128"
            # A value-preserving scalar reduction (max / min / sum / prod) over an
            # INTEGER-tagged operand yields an integer -- inherit that dtype so the
            # accumulator temp is declared int, not the float default. Otherwise the
            # Fortran emit's running-max ``merge(int_elem, real_acc, ...)`` update is
            # a kind mismatch. mean/std/var/median are excluded: they produce a float
            # even from an int input.
            elif (key[1] in {"max", "min", "sum", "prod"} and node.args and isinstance(node.args[0], ast.Name)):
                src_dt = self.local_dtypes.get(node.args[0].id)
                if src_dt and dtypes.is_integer(src_dt):
                    self.local_dtypes[temp] = src_dt
        # Emit a ``__cb<n> = __hpcagent_bench_zeros__()`` marker first so the emit
        # walker can inline-declare the temp at the marker site -- required
        # when the temp's shape depends on an enclosing for-loop iter
        # (stockham_fft's ``R ** i``). The subsequent ``__cb<n> = call(...)``
        # then lowers into a per-element copy via the existing call-expansion path.
        if not is_scalar:
            self.pre_stmts.append(
                ast.Assign(targets=[ast.Name(id=temp, ctx=ast.Store())],
                           value=ast.Call(func=ast.Name(id="__hpcagent_bench_zeros__", ctx=ast.Load()),
                                          args=[],
                                          keywords=[])))
        # Synthesise an Assign that the LibNodeRewriter will lower.
        self.pre_stmts.append(ast.Assign(targets=[ast.Name(id=temp, ctx=ast.Store())], value=node))
        return ast.Name(id=temp, ctx=ast.Load())

    def _derive_output_shape(self, key, args, keywords=None):
        op = key[1]
        # Tensor contractions (einsum/tensordot/inner): reuse the shared
        # output-extent resolver so the hoister can lift a contraction out of a
        # BinOp -- seissol's batched-GEMM-as-einsum ``Q[:] = Q +
        # np.einsum('dkl,blq,dqp->bkp', ...)``. A scalar-result contraction
        # ('ii->') yields a None extent, handled by the direct-assign expander
        # path instead.
        if op in {"einsum", "tensordot", "inner"} and len(args) >= 2:
            # ``tensordot``'s ``axes`` is frequently passed by KEYWORD
            # (``axes=([2], [0])``); dropping it here defaults to ``axes=2``
            # and yields a truncated/scalar extent (the >2-D temp mis-sized as 1-D).
            call = _attr_call("np", op, list(args))
            call.keywords = list(keywords or [])
            ext = _iter_extent_of(call, self.shape_table)
            if ext is not None:
                return tuple(self._extent_to_shape_token(e) for e in ext)
        # ``np.bincount(idx, weights=w, minlength=M)`` -> a rank-1 result of exactly M slots (see
        # expand_bincount for why the data-dependent upper term is not the extent).
        if op == "bincount" and args:
            minlength = _kwarg_or_pos(args, keywords or [], 2, "minlength")
            if minlength is not None:
                return (self._extent_to_shape_token(minlength), )
        # ``np.searchsorted(a, v)`` -> one index per element of the VALUES operand, so the temp
        # takes ``v``'s extent and not the sorted array's.
        if op == "searchsorted" and len(args) >= 2:
            ext = _iter_extent_of(args[1], self.shape_table)
            if ext is not None:
                return tuple(self._extent_to_shape_token(e) for e in ext)
        # ``np.pad`` -> source shape with each axis grown by ``2 * pad_width``.
        if op == "pad" and args:
            call = _attr_call("np", "pad", list(args))
            call.keywords = list(keywords or [])
            ext = _iter_extent_of(call, self.shape_table)
            if ext is not None:
                return tuple(self._extent_to_shape_token(e) for e in ext)
        # Allocator-style calls: shape from the constructor arg.
        if op in {"linspace", "arange"}:
            # linspace(start, stop, n) -> (n,); arange(stop) -> (stop,);
            # arange(start, stop[, step]) -> its element count. The 3-arg form must go through
            # arange_count: `stop - start` ignores the step, which over-allocates for step > 1 and
            # is NEGATIVE for a step < 0 (see arange_count).
            if op == "linspace" and len(args) >= 3:
                return (self._extent_to_shape_token(args[2]), )
            if op == "arange":
                if len(args) == 1:
                    return (self._extent_to_shape_token(args[0]), )
                if len(args) == 2:
                    return (ast.unparse(ast.BinOp(left=args[1], op=ast.Sub(), right=args[0])), )
                if len(args) >= 3:
                    return (ast.unparse(arange_count(list(args[:3]))), )
        # ``np.fromfunction(lambda..., (N, M))`` -> the SECOND arg is the shape.
        if op == "fromfunction" and len(args) >= 2:
            sh = args[1]
            elts = (sh.elts if isinstance(sh, (ast.Tuple, ast.List)) else [sh])
            return tuple(self._extent_to_shape_token(e) for e in elts)
        # ``np.histogram(a, bins, ...)`` returns ``hist`` of length
        # ``bins`` (the ``[0]`` Subscript unwrap selects it).
        if op == "histogram" and len(args) >= 2:
            return (self._extent_to_shape_token(args[1]), )
        # ``np.linalg.inv(A)`` returns the square inverse with A's
        # shape.
        if op == "linalg.inv" and args and isinstance(args[0], ast.Name):
            shape = self.shape_table.get(args[0].id)
            if shape:
                return tuple(shape)
        # ``np.linalg.solve(A, b)`` returns x with b's shape.
        if op == "linalg.solve" and len(args) >= 2 \
                and isinstance(args[1], ast.Name):
            shape = self.shape_table.get(args[1].id)
            if shape:
                return tuple(shape)
        # Every ``np.fft.*`` transform is shape-preserving (the output has the
        # same shape as the input -- only the values change).
        if op in {"fft.fftn", "fft.ifftn", "fft.fft", "fft.ifft"} \
                and args and isinstance(args[0], ast.Name):
            shape = self.shape_table.get(args[0].id)
            if shape:
                return tuple(shape)
        # ``np.fft.fftfreq(n, d=...)`` -> a 1-D frequency array of length ``n``
        # (the first positional arg is the sample count, not an array operand).
        if op == "fft.fftfreq" and args:
            return (self._extent_to_shape_token(args[0]), )
        # ``np.diag(v [, k])`` -- 1-D operand builds an ``(n+|k|, n+|k|)`` matrix,
        # 2-D operand extracts the diagonal. Reuses the ``_iter_extent_of`` rule
        # so the constructed-shape logic lives in one place; lets a Lanczos
        # ``T = np.diag(alphas) + np.diag(betas[1:], 1) + np.diag(betas[1:], -1)``
        # hoist each ``np.diag`` out of the BinOp into a correctly sized temp.
        if op == "diag" and args:
            call = _attr_call("np", "diag", list(args))
            call.keywords = list(keywords or [])
            ext = _iter_extent_of(call, self.shape_table)
            if ext is not None:
                return tuple(self._extent_to_shape_token(e) for e in ext)
        # ``np.roll`` / ``np.linalg.cholesky`` / ``np.tril`` / ``np.triu`` all
        # return an array with the FIRST operand's shape -- so an inline
        # ``acc + np.roll(x, m, axis)`` (the periodic-stencil idiom) can be
        # hoisted out of the BinOp instead of reaching the emitter unlowered.
        if op in {"roll", "linalg.cholesky", "tril", "triu"} \
                and args and isinstance(args[0], ast.Name):
            shape = self.shape_table.get(args[0].id)
            if shape:
                return tuple(shape)
        # ``swapaxes`` / ``expand_dims`` / ``squeeze`` / ``moveaxis`` -- the operand's extent with axes
        # swapped, moved, or a unit axis inserted / dropped. ``_iter_extent_of`` already computes all three, so route to
        # it rather than restating the axis arithmetic; without a branch here they fall through to
        # the elementwise case, which skips them (they are NON_ELEMENTWISE), and the None return
        # silently DECLINES to hoist -- leaving ``q @ np.swapaxes(k, -1, -2)`` for the emitter.
        if op in {"swapaxes", "expand_dims", "squeeze", "moveaxis"} and args:
            call = _attr_call("np", op, list(args))
            call.keywords = list(keywords or [])
            ext = _iter_extent_of(call, self.shape_table)
            if ext is not None:
                return tuple(self._extent_to_shape_token(e) for e in ext)
        # ``np.reshape(a, shape)`` -- output extents are the shape arg, with a
        # single ``-1`` resolved to prod(source) / prod(other dims). Lets the
        # flattened-dot idiom ``a.ravel() @ a.ravel()`` (lowered to reshape)
        # hoist inline out of the matmul.
        if op == "reshape" and len(args) >= 2 and isinstance(args[0], ast.Name):
            src = self.shape_table.get(args[0].id)
            sh = args[1]
            elts = sh.elts if isinstance(sh, (ast.Tuple, ast.List)) else [sh]
            toks = [self._extent_to_shape_token(e) for e in elts]
            if src is not None:
                prod_src = "(" + ") * (".join(str(s) for s in src) + ")"
                if any(str(t).strip() == "-1" for t in toks):
                    others = [t for t in toks if str(t).strip() != "-1"]
                    if others:
                        denom = "(" + ") * (".join(str(t) for t in others) + ")"
                        neg = f"({prod_src}) / ({denom})"
                    else:
                        neg = f"({prod_src})"
                    toks = [neg if str(t).strip() == "-1" else str(t) for t in toks]
                return tuple(str(t) for t in toks)
        # ``np.concatenate((a, b, ...), axis=k)`` -> common shape, axis summed.
        if op == "concatenate" and args:
            try:
                _names, shapes, axis = _concat_operands_axis(args, keywords, self.shape_table)
            except NotImplementedError:
                shapes = None
            if shapes:
                base = list(shapes[0])
                base[axis] = "(" + ") + (".join(s[axis] for s in shapes) + ")"
                return tuple(base)
        # Elementwise unary / binary share the operand shape; first array
        # operand (Name or Subscript-with-Slice) wins.
        if op in ELEMENTWISE_SHAPE_OPS and args:
            # Broadcast the extents of ALL operands, not just the first: the
            # hoisted temp for ``np.maximum(a(M,), B(N, M))`` must be the full
            # broadcast shape ``(N, M)``, matching the elementwise expander's own
            # broadcast iteration -- else a lower-rank first operand under-sizes
            # the temp.
            acc: Optional[Tuple[ast.expr, ...]] = None
            for arg in args:
                ext = _iter_extent_of(arg, self.shape_table)
                if ext is None:
                    continue
                acc = ext if acc is None else _broadcast_extents(acc, ext)
            if acc is not None:
                return tuple(self._extent_to_shape_token(e) for e in acc)
        # ``np.hstack((a, b, c))`` -- horizontal stack along axis 1
        # for 2-D operands, axis 0 for 1-D operands. Sum the
        # concatenation-axis widths; the other axes are shared.
        if op == "hstack" and args:
            ops = (list(args[0].elts) if (len(args) == 1 and isinstance(args[0], ast.Tuple)) else list(args))
            shapes = []
            for op_arg in ops:
                if not isinstance(op_arg, ast.Name):
                    return None
                s = self.shape_table.get(op_arg.id)
                if not s:
                    return None
                shapes.append(s)
            if not shapes:
                return None
            rank = len(shapes[0])
            if rank == 1:
                widths = "+".join(s[0] for s in shapes)
                return (widths, )
            if rank == 2:
                widths = "+".join(s[1] for s in shapes)
                return (shapes[0][0], widths)
            return None
        if op == "diff" and args and isinstance(args[0], ast.Name):
            shape = self.shape_table.get(args[0].id)
            if not shape:
                return None
            rank = len(shape)
            n_node = _kwarg_or_pos(args, keywords, 1, "n")
            if n_node is not None and _const_int(n_node) != 1:
                return None
            ax_node = _kwarg_or_pos(args, keywords, 2, "axis")
            ax = rank - 1 if ax_node is None else _const_axis(ax_node, rank)
            if ax is None:
                return None
            out = list(shape)
            ext = out[ax]
            out[ax] = str(int(ext) - 1) if ext.strip().isdigit() else f"({ext}) - 1"
            return tuple(out)
        if op in {"transpose", "triu", "flip"} and args and isinstance(args[0], ast.Name):
            shape = self.shape_table.get(args[0].id)
            if not shape:
                return None
            if op != "transpose":
                return tuple(shape)
            # ``np.transpose(A, axes)`` honours the perm (positional or
            # via the ``axes=`` keyword); without it, reverse axes.
            perm_arg = _kwarg_or_pos(args, keywords, 1, "axes")
            if isinstance(perm_arg, (ast.Tuple, ast.List)):
                perm = [e.value for e in perm_arg.elts if isinstance(e, ast.Constant) and isinstance(e.value, int)]
                if len(perm) == len(shape):
                    return tuple(shape[p] for p in perm)
            return tuple(reversed(shape))
        # Axis-aware reductions: output shape = reduction axis removed (or size
        # 1 if keepdims). ``argmax``/``argmin`` with an axis return the index
        # array over the kept axes (same shape as a value reduction);
        # axis-aware ``linalg.norm`` is a per-line L2 reduction with the same
        # kept-axes shape; ``var`` sizes exactly like ``std`` (one op -- std is
        # var plus a sqrt).
        if op in {"sum", "max", "min", "mean", "prod", "std", "var", "argmax", "argmin", "linalg.norm"}:
            if args and isinstance(args[0], ast.Name):
                src_shape = self.shape_table.get(args[0].id)
                if src_shape:
                    # args doesn't carry keywords (those are on the parent
                    # call), so read the live axis/keepdims stash visit_Call set.
                    kw_axes, kw_keep = vars(self).get('_cur_axis'), vars(self).get('_cur_keepdims', False)
                    if kw_axes is None:
                        return None  # scalar -- not array-shape
                    # ``_read_axis_keepdims`` returns a list or None; normalise
                    # to a set of resolved positive axes.
                    if isinstance(kw_axes, int):
                        kw_axes = [kw_axes]
                    resolved = []
                    for a in kw_axes:
                        na = a + len(src_shape) if a < 0 else a
                        if 0 <= na < len(src_shape):
                            resolved.append(na)
                    axes_set = set(resolved)
                    if kw_keep:
                        return tuple("1" if i in axes_set else s for i, s in enumerate(src_shape))
                    return tuple(s for i, s in enumerate(src_shape) if i not in axes_set)
        if op == "reshape" and len(args) >= 2:
            shape_arg = args[1]
            if isinstance(shape_arg, ast.Tuple):
                parts = []
                for e in shape_arg.elts:
                    if _const_int(e) is not None:
                        parts.append(str(_const_int(e)))
                    elif isinstance(e, ast.Name):
                        parts.append(e.id)
                    else:
                        parts.append(ast.unparse(e))
                # Resolve a ``-1`` placeholder (``x.reshape(batch, -1)``) to the source
                # element count over the product of the other target dims. ``/`` renders
                # as integer division in C/Fortran (both dims are integers).
                neg1 = [i for i, p in enumerate(parts) if p.strip() == "-1"]
                src = args[0]
                src_shape = self.shape_table.get(src.id) if isinstance(src, ast.Name) else None
                if len(neg1) == 1 and src_shape:
                    total = " * ".join(f"({t})" for t in src_shape)
                    others = [p for j, p in enumerate(parts) if j != neg1[0]]
                    denom = " * ".join(f"({p})" for p in others) if others else "1"
                    parts[neg1[0]] = f"({total}) / ({denom})"
                return tuple(parts)
        if op in {"outer", "add.outer"} and len(args) == 2:
            a_ext = _iter_extent_of(args[0], self.shape_table)
            b_ext = _iter_extent_of(args[1], self.shape_table)
            if (a_ext is not None and b_ext is not None and len(a_ext) == 1 and len(b_ext) == 1):
                return (self._extent_to_shape_token(a_ext[0]), self._extent_to_shape_token(b_ext[0]))
        # ``np.diagonal(a)`` on a SQUARE rank-2 operand: one element per row. Without a size here the
        # hoister declines, so the diagonal stayed inline inside ``np.tanh(...)`` -- where the
        # elementwise scalariser has no cell to read and the call reached emit whole.
        if op == "diagonal" and len(args) == 1:
            d_ext = _iter_extent_of(args[0], self.shape_table)
            if d_ext is not None and len(d_ext) == 2 and ast.unparse(d_ext[0]) == ast.unparse(d_ext[1]):
                return (self._extent_to_shape_token(d_ext[0]), )
        # linalg ops that preserve their argument's shape.
        if op in {"linalg.cholesky", "linalg.inv"} \
                and args and isinstance(args[0], ast.Name):
            shape = self.shape_table.get(args[0].id)
            if shape:
                return tuple(shape)
        return None

    @staticmethod
    def _extent_to_shape_token(node):
        """Render an extent AST as a shape-table token (string): Constants ->
        int string; Names -> name; BinOps -> ``ast.unparse`` (e.g. ``N - 2``),
        which the emitter treats as a non-int symbolic shape.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return str(node.value)
        if isinstance(node, ast.Name):
            return node.id
        return ast.unparse(node)


import inspect


def _call_expander(expander,
                   target,
                   args,
                   keywords,
                   shape_table,
                   local_dtypes=None,
                   fresh_local_allocs=None,
                   dim_aliases=None):
    """Adapter: pass ``keywords``/``local_dtypes``/``fresh_local_allocs`` to
    expanders that accept them, else call with the legacy signature. The two
    extra tables let an expander register internal working buffers (shape +
    dtype) so the emit declares them correctly.
    """
    sig = inspect.signature(expander)
    params = sig.parameters
    extras: Dict[str, object] = {}
    if "kwargs" in params:
        extras["kwargs"] = keywords
    if "local_dtypes" in params and local_dtypes is not None:
        extras["local_dtypes"] = local_dtypes
    if "fresh_local_allocs" in params and fresh_local_allocs is not None:
        extras["fresh_local_allocs"] = fresh_local_allocs
    if "dim_aliases" in params and dim_aliases is not None:
        extras["dim_aliases"] = dim_aliases
    return expander(target, args, shape_table, **extras)


#: Expander keys that accept a partial-slice assignment target
#: (``row_offsets[1:] = np.cumsum(m_sizes)``) in addition to a bare Name. The
#: full-slice form is canonicalised to a Name in ``visit_Assign``; only a
#: shifted slice reaches here, and only the cumulative scans honour the
#: lower-bound offset (via :func:`_scan_target_offsets`).
#: Registered ops whose result shape is NOT the broadcast of its operands -- reductions,
#: constructors, shape-changers and contractions. Everything else that has an expander IS
#: elementwise, which is how :data:`ELEMENTWISE_SHAPE_OPS` below is derived.
#:
#: Derived, not restated, and computed AFTER every registration: the previous hand-written list was
#: missing `square`, `reciprocal`, `degrees`, `radians`, `asarray`, the comparison ufuncs and the
#: binary libm ufuncs, and each omission made ``_derive_output_shape`` return None -- which does not
#: fail, it silently DECLINES to hoist and leaves a bare ``np.square(...)`` for the emitter.
NON_ELEMENTWISE_SHAPE_OPS: Set[str] = set(_REDUCTION_NAMES) | {
    "reshape", "repeat", "transpose", "flip", "roll", "triu", "tril", "concatenate", "stack", "pad", "diag", "diagonal",
    "diff", "trace", "einsum", "tensordot", "inner", "outer", "dot", "vdot", "matmul", "cumsum", "cumprod", "linspace",
    "arange", "zeros", "ones", "empty", "full", "eye", "identity", "fromfunction", "meshgrid", "mgrid", "sort",
    "argsort", "histogram", "unique", "take", "interp", "searchsorted", "nonzero", "kron", "cross", "split",
    "expand_dims", "squeeze", "swapaxes", "moveaxis", "ravel", "flatten", "tile", "broadcast_to", "atleast_1d",
    "atleast_2d", "append", "insert", "delete"
}

ELEMENTWISE_SHAPE_OPS: FrozenSet[str] = frozenset(
    name for module, name in NP_CALL_EXPANDERS
    if module == "np" and "." not in name and name not in NON_ELEMENTWISE_SHAPE_OPS) | {
        "copy", "array", "where", "clip"
    }

_SLICE_TARGET_EXPANDERS = {("np", "cumsum"), ("np", "cumprod")}

#: Expander keys that write element-wise to ``target`` (no allocation);
#: ``target`` must already be declared at the C level. When the kernel body
#: uses ``X = np.linspace(...)`` as the first reference to ``X``, the
#: LibNodeRewriter registers ``X`` in :attr:`fresh_local_allocs` so the emitter
#: generates a local decl.
_ELEMENT_WRITE_EXPANDERS = {
    ("np", "linspace"),
    ("np", "arange"),
    ("np", "fromfunction"),
    # Elementwise functions that write to a fresh-local LHS need the
    # same auto-alloc treatment -- the original Assign is replaced by
    # the loop nest, leaving the target dangling without a decl.
    ("np", "less"),
    ("np", "less_equal"),
    ("np", "greater"),
    ("np", "greater_equal"),
    ("np", "equal"),
    ("np", "not_equal"),
    ("np", "logical_and"),
    ("np", "logical_or"),
    ("np", "logical_not"),
    *(("np", _name) for _name in UNARY_C_MATH),
    ("np", "maximum"),
    ("np", "minimum"),
    ("np", "add"),
    ("np", "subtract"),
    ("np", "multiply"),
    ("np", "divide"),
    ("np", "power"),
    ("np", "exp"),
    ("np", "log"),
    ("np", "sqrt"),
    ("np", "sin"),
    ("np", "cos"),
    ("np", "tan"),
    ("np", "tanh"),
    ("np", "abs"),
    ("np", "absolute"),
    ("np", "sort"),
    ("np", "histogram"),
    ("np", "linalg.inv"),
    ("np", "linalg.solve"),
    ("np", "linalg.lstsq"),
    # Contraction / scan / indexing ops that write element-wise to a fresh LHS.
    ("np", "einsum"),
    ("np", "tensordot"),
    ("np", "inner"),
    ("np", "trace"),
    ("np", "diagonal"),
    ("np", "diag"),
    ("np", "fft.fftfreq"),
    ("np", "cumsum"),
    ("np", "cumprod"),
    ("np", "searchsorted"),
    # Same shape as cumsum/cumprod -- a running max/min bound to a fresh local. Left out, the
    # expander consumed the Assign that carried the allocation marker and the scan's first store
    # went through a NULL pointer.
    ("np", "maximum.accumulate"),
    ("np", "minimum.accumulate"),
    # Same shape again: the tile-and-write nest replaces the Assign that carried the marker, so a
    # fresh ``row_of = np.repeat(...)`` was emitted with no declaration at all.
    ("np", "repeat"),
    # An AXIS reduction writes one element per kept-axes position, so its fresh LHS is an array and
    # needs the marker like any other element-write. Without it the Fortran backend declared
    # ``valid = match.any(axis=-1)`` allocatable -- its extent names a computed scalar -- and then
    # had no site to allocate it at, so the reduction's first store went into an unallocated array.
    ("np", "sum"),
    ("np", "prod"),
    ("np", "mean"),
    ("np", "any"),
    ("np", "all"),
    ("np", "count_nonzero"),
    ("np", "argmax"),
    ("np", "argmin"),
    ("np", "roll"),
    ("np", "tril"),
    ("np", "pad"),
    # ``out = np.concatenate((a, b), axis=k)`` copies each operand into ``out`` at its
    # offset -- an element-write like the rest, so its fresh LHS needs the same auto-alloc
    # (dwt2d's ``e1 = np.concatenate((e[:, 1:], e[:, 0:1]), axis=1)`` was emitted undeclared).
    ("np", "concatenate"),
}


def _index_axes(sub: ast.Subscript) -> Optional[Tuple[str, ...]]:
    """Per-axis index expressions of a fully scalar subscript, or ``None`` when
    any axis is a slice (no single cell to reason about)."""
    elts = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else [sub.slice]
    if any(isinstance(e, (ast.Slice, ast.Starred)) for e in elts):
        return None
    return tuple(ast.unparse(e) for e in elts)


def _reduction_misses_target(target: ast.Subscript, loop: ast.For, body: ast.expr) -> bool:
    """True when moving the reduction onto ``target`` cannot change what ``body``
    reads: either ``body`` never touches the target's array, or every read of it
    provably misses ``target``'s cell for all iterations of ``loop``.

    ``trmm``'s ``B[i, j] += sum_k A[k, i] * B[k, j]`` needs the second rung: ``k``
    runs from ``i + 1``, so the accumulating cell is never read back. ``symm``
    takes the first (``temp2`` is write-only in the body).
    """
    base = target.value.id
    mentions = [n for n in ast.walk(body) if isinstance(n, ast.Name) and n.id == base]
    if not mentions:
        return True
    reads = [n for n in ast.walk(body) if isinstance(n, ast.Subscript) and n.value in mentions]
    if len(reads) != len(mentions):
        return False  # a bare-Name (whole-array) mention -- no cell to compare
    axes = _index_axes(target)
    if axes is None:
        return False
    # The nonnegativity below is what proves the miss, so demand the 0-based
    # ``range(n)`` that makes it true rather than assuming every hoist emits one.
    if not (isinstance(loop.iter, ast.Call) and isinstance(loop.iter.func, ast.Name) and loop.iter.func.id == "range"
            and len(loop.iter.args) == 1):
        return False
    # Deferred: sympy import costs ~100s of ms and only this narrow shape needs it.
    import sympy
    iter_name = loop.target.id
    it = sympy.Symbol(iter_name, nonnegative=True)
    for read in reads:
        read_axes = _index_axes(read)
        if read_axes is None or len(read_axes) != len(axes):
            return False
        try:
            diffs = [
                sympy.sympify(r, locals={iter_name: it}) - sympy.sympify(t, locals={iter_name: it})
                for r, t in zip(read_axes, axes)
            ]
        except (SyntaxError, TypeError, AttributeError, ValueError, sympy.SympifyError):
            return False
        if not any(d.is_zero is False for d in diffs):
            return False
    return True


def _accumulation_addend(step: ast.stmt, scalar: str) -> Optional[ast.expr]:
    """What one ``+=`` step of ``scalar`` adds, in either spelling the lowerings emit
    (``s += x`` from the matmul hoist, ``s = s + x`` from the reduction expanders), else ``None``."""
    if isinstance(step, ast.AugAssign):
        if isinstance(step.op, ast.Add) and isinstance(step.target, ast.Name) and step.target.id == scalar:
            return step.value
        return None
    if (isinstance(step, ast.Assign) and len(step.targets) == 1 and isinstance(step.targets[0], ast.Name)
            and step.targets[0].id == scalar and isinstance(step.value, ast.BinOp)
            and isinstance(step.value.op, ast.Add) and isinstance(step.value.left, ast.Name)
            and step.value.left.id == scalar):
        return step.value.right
    return None


def _retarget_scalar_accumulator(node: ast.stmt, prelude: List[ast.stmt]) -> Optional[List[ast.stmt]]:
    """Fold ``s = 0.0; for k: s += f(k); T[idx] (+)= s`` into an in-place
    reduction on ``T[idx]``, dropping the scalar. Returns ``None`` when the
    statement is not that shape.

    ``emit_pluto`` hoists every scalar declaration above ``#pragma scop``, so
    ``s`` models to pet as a one-element array live across the whole iteration
    domain: the false WAW/WAR fragments the schedule, and Pluto's ``--parallel``
    privatises only its own tile counters, leaving ``s`` shared in the parallel
    band -- a race. An array cell carries the same reduction with no
    scop-external state, which is why ``syrk`` (reducing into ``C[i, si1]``) was
    never affected.

    pet goes further than racing on that scalar: a statement whose only write is one drops out of
    the transformed output entirely (POLYCC-009), so the same fold is what keeps a full
    ``np.sum`` into an array cell computing at all.
    """
    if len(prelude) < 2 or not isinstance(node.value, ast.Name):
        return None
    augmented = isinstance(node, ast.AugAssign)
    if augmented:
        if not isinstance(node.op, ast.Add):
            return None
        target = node.target
    else:
        if len(node.targets) != 1:
            return None
        target = node.targets[0]
    # A single cell is the whole point: a slice destination is a different lowering
    # (slice fusion) and would not give pet the affine reduction carrier we are after.
    if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
            and _index_axes(target) is not None):
        return None
    scalar = node.value.id
    init, loop = prelude[-2], prelude[-1]
    if not (isinstance(init, ast.Assign) and len(init.targets) == 1 and isinstance(init.targets[0], ast.Name)
            and init.targets[0].id == scalar and isinstance(init.value, ast.Constant) and init.value.value == 0.0):
        return None
    if not (isinstance(loop, ast.For) and not loop.orelse and len(loop.body) == 1
            and isinstance(loop.target, ast.Name)):
        return None
    addend = _accumulation_addend(loop.body[0], scalar)
    if addend is None:
        return None
    if any(isinstance(n, ast.Name) and n.id == scalar for n in ast.walk(addend)):
        return None  # self-referential accumulation -- not a plain reduction
    if not _reduction_misses_target(target, loop, addend):
        return None
    store = copy.deepcopy(target)
    store.ctx = ast.Store()
    loop.body = [ast.AugAssign(target=store, op=ast.Add(), value=addend)]
    # An ``AugAssign`` target already holds the running value: zero-initialising
    # it would drop what the reduction must add to.
    if augmented:
        return prelude[:-2] + [loop]
    zero = copy.deepcopy(target)
    zero.ctx = ast.Store()
    return prelude[:-2] + [ast.Assign(targets=[zero], value=_const(0.0)), loop]


class LibNodeRewriter(ast.NodeTransformer):
    """Single-pass rewriter for the library-node registry. Consumes
    :class:`KernelIR`'s array-shape table so reduction expanders know how many
    loop levels to emit and matmul knows the M/K/N bounds. Assignments whose
    RHS matches a registered idiom are replaced in-place by the expander's
    statement list; other statements pass through.

    Propagates shapes through whole-array aliases (``x = __cb2``): the LHS
    Name inherits the RHS's shape so subsequent uses of ``x`` can be
    matmul-hoisted/scalarised.
    """

    def __init__(self,
                 shape_table: Dict[str, Tuple[str, ...]],
                 known_arrays: Optional[Set[str]] = None,
                 local_dtypes: Optional[Dict[str, str]] = None,
                 sparse: Optional[Dict[str, object]] = None,
                 dim_aliases: Optional[Dict[str, str]] = None,
                 native_call: Optional[Callable[[Tuple[str, str], ast.Call, Dict, Dict], bool]] = None,
                 native_dtypes: Optional[Dict[str, str]] = None):
        self.shape_table = shape_table
        #: Target's "I render this numpy call myself" predicate. A call it claims is left
        #: UNEXPANDED so the emitter can use its own intrinsic -- Fortran's SUM/MAXVAL/NORM2 --
        #: instead of the loop nest every target would otherwise get. Default: claims nothing.
        self.native_call = native_call
        #: name -> element dtype, for the same predicate. Separate from ``local_dtypes``, which
        #: deliberately tags only integers: the predicate needs POSITIVE evidence of a float, and
        #: "untagged" there means "not an integer", which a boolean mask also satisfies.
        self.native_dtypes = native_dtypes or {}
        #: Dimension local -> its definition in parameter terms, threaded to the matmul hoister so
        #: a contraction dim spelled two ways still matches. See :func:`dims_agree`.
        self.dim_aliases: Dict[str, str] = dim_aliases or {}
        #: Logical-name -> SparseArrayDesc, threaded to the matmul hoister so
        #: ``A @ B`` on sparse operands routes to the per-format sparse emitter.
        self.sparse: Dict[str, object] = sparse or {}
        #: Names already known as signature-declared arrays (kernel
        #: parameters/outputs) -- the auto-alloc path skips these to avoid
        #: re-declaring an already-declared input.
        self.known_arrays: Set[str] = known_arrays or set()
        #: Per-local dtype table, shared with ``_CallHoister`` so the temp it
        #: synthesises for ``np.exp(-2j * ...)`` carries ``complex128`` through
        #: to the emit-time declaration.
        self.local_dtypes: Dict[str, str] = (local_dtypes if local_dtypes is not None else {})
        #: Filled-in temps the emitter must declare as local arrays (same dict
        #: shape as ``zeros_locals`` so the zeros rewriter picks them up once
        #: LibNodeRewriter has finished).
        self.matmul_temps: Dict[str, Tuple[str, ...]] = {}
        #: Scalar temps introduced by ``_CallHoister`` (e.g. ``A[i, j] -=
        #: np.dot(...)`` -> intermediate scalar).
        self.scalar_call_temps: Dict[str, bool] = {}
        #: Fresh locals introduced when an expander writes element-wise to a
        #: bare-Name LHS (linspace/arange/np.less etc). These need a C decl but
        #: the original Assign is consumed by the expander, so the emitter
        #: would otherwise miss the allocation. Mirrors ``zeros_locals`` --
        #: merged in by ``lower()``.
        self.fresh_local_allocs: Dict[str, Tuple[str, ...]] = {}
        self._counter = [0]

    def _hoist_value(self, value: ast.expr) -> Tuple[ast.expr, List[ast.stmt]]:
        # First hoist registered library-node calls, so e.g. ``A[i, j] -=
        # np.dot(A[i, :j], A[j, :j])`` becomes ``__cb1 = np.dot(...); A[i, j]
        # -= __cb1`` -- the next matmul hoist + expansion passes then handle
        # the synthetic assignment uniformly.
        call_hoister = _CallHoister(self.shape_table,
                                    self.scalar_call_temps,
                                    self.matmul_temps,
                                    self._counter,
                                    local_dtypes=self.local_dtypes,
                                    dim_aliases=self.dim_aliases)
        call_hoister.sparse = self.sparse
        value = call_hoister.visit(value)
        pre = list(call_hoister.pre_stmts)
        # Now hoist any matmul subexpressions.
        mm_hoister = _MatmulHoister(self.shape_table,
                                    self.matmul_temps,
                                    self._counter,
                                    local_dtypes=self.local_dtypes,
                                    sparse=self.sparse,
                                    dim_aliases=self.dim_aliases)
        new_value = mm_hoister.visit(value)
        pre.extend(mm_hoister.pre_stmts)
        return new_value, pre

    def _update_shape_for_assign(self, target_id: str, rhs: ast.AST) -> None:
        """Update ``shape_table[target_id]`` to reflect the broadcast extent of
        ``rhs``. Mirrors the post-pipeline source-order shape resolver but
        runs inside the LibNodeRewriter pass so the hoister sees the
        then-current shape of every reassigned local. Also propagates
        ``local_dtypes`` for complex-RHS so the next statement's hoister sees
        the up-to-date dtype tag."""
        # Name = Name alias.
        if isinstance(rhs, ast.Name):
            src = self.shape_table.get(rhs.id)
            if src is not None:
                self.shape_table[target_id] = tuple(src)
            rhs_dt = self.local_dtypes.get(rhs.id)
            if rhs_dt and target_id not in self.local_dtypes:
                self.local_dtypes[target_id] = rhs_dt
            return
        # np.zeros / empty / etc constructor -- the ZerosRewriter owns the ALLOCATION, and it
        # runs in a later phase. The _like forms still have to publish their EXTENT here: the
        # source array's shape may only become known during this pass (eigh_test's ``scaled =
        # np.zeros_like(bu)``, where ``bu`` is an eigh output this same rewriter expands), and a
        # local with no shape declines every matmul it feeds -- which slice fusion then refuses.
        if (isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Attribute) and isinstance(rhs.func.value, ast.Name)
                and rhs.func.value.id == "np" and rhs.func.attr in NP_ZEROS_ALIASES):
            if rhs.func.attr.endswith("_like") and rhs.args and isinstance(rhs.args[0], ast.Name):
                src = self.shape_table.get(rhs.args[0].id)
                if src is not None:
                    self.shape_table[target_id] = tuple(src)
            return
        # Shape-CHANGING ops (reshape/repeat/transpose) aren't elementwise, but
        # the generic ``_iter_extent_of`` Call branch would treat them as such
        # and return the source operand's extent -- the wrong shape for the LHS.
        if (isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Attribute) and isinstance(rhs.func.value, ast.Name)
                and rhs.func.value.id == "np" and rhs.func.attr in {"reshape", "repeat", "transpose"}):
            attr = rhs.func.attr
            if attr == "reshape" and len(rhs.args) >= 2:
                # ``yv = np.reshape(y, (R**i, R, ...))`` -- the result
                # shape is the explicit newshape arg, not y's extent.
                newshape = rhs.args[1]
                toks: Optional[Tuple[str, ...]] = None
                if isinstance(newshape, (ast.Tuple, ast.List)):
                    toks = tuple(ast.unparse(e) for e in newshape.elts)
                elif isinstance(newshape, ast.Name):
                    toks = (newshape.id, )
                elif _const_int(newshape) is not None:
                    toks = (str(_const_int(newshape)), )
                if toks is not None:
                    # Resolve a ``-1`` placeholder (``x.reshape(batch, -1)``) to the source
                    # element count divided by the product of the other target dims.
                    neg1 = [i for i, t in enumerate(toks) if t.strip() == "-1"]
                    src = rhs.args[0]
                    src_shape = self.shape_table.get(src.id) if isinstance(src, ast.Name) else None
                    if len(neg1) == 1 and src_shape:
                        total = " * ".join(f"({t})" for t in src_shape)
                        others = [t for j, t in enumerate(toks) if j != neg1[0]]
                        denom = " * ".join(f"({t})" for t in others) if others else "1"
                        toks = tuple(f"({total}) / ({denom})" if j == neg1[0] else t for j, t in enumerate(toks))
                    self.shape_table[target_id] = toks
            # For reshape with an unparsed newshape, and for repeat/transpose,
            # the dedicated expander plus the harvested declaration shape
            # (e.g. D's rank-3 ``np.empty((R, R**i, R**(K-i-1)))`` consumed by
            # ``D[:] = np.repeat(...)``) are authoritative -- never downgrade a
            # known shape from the source operand's extent.
            return
        # BinOp/UnaryOp/IfExp/Call/Subscript -- broadcast extent. ``Subscript``
        # covers ``cols = A_col[A_row[i]:A_row[i+1]]`` (dynamic-bound slice) and
        # ``y = arr[idx]`` (fancy gather), so the next statement sees the
        # local's shape when hoisting a matmul.
        if isinstance(rhs, (ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Call, ast.Subscript)):
            ext = _iter_extent_of(rhs, self.shape_table)
            if ext is not None:
                self.shape_table[target_id] = tuple(_CallHoister._extent_to_shape_token(e) for e in ext)
            # ``ngm = qgm.shape[0]`` reads a DIMENSION -- an integer regardless of
            # the array's dtype. Type it int64 and skip the complex walk below,
            # which would otherwise see complex base Name ``qgm`` and wrongly tag
            # the scalar bound complex (vexx_k's ``_addusxx_g``/``_newdxx_g``).
            if _is_shape_scalar(rhs):
                if target_id not in self.local_dtypes:
                    self.local_dtypes[target_id] = "int64"
                return
            # Complex-dtype propagation for BinOp/UnaryOp/Call: a subtree that
            # reads a complex Constant or Name promotes the LHS. ``.shape``
            # subtrees are skipped, so a dimension read off a complex array
            # (``ngm = qgm.shape[0] - 1``) isn't mis-tagged complex.
            if target_id not in self.local_dtypes and _reads_complex(rhs, self.local_dtypes):
                self.local_dtypes[target_id] = "complex128"
            return

    def _lookup(self, call: ast.Call):
        """Resolve a ``Call.func`` to a registry key. Recognises both
        ``np.<name>`` and ``np.linalg.<name>`` (the latter an Attribute whose
        value is itself an Attribute on a Name); the registry key encodes the
        qualified form as ``("np", "linalg.cholesky")``.
        """
        func = call.func
        if isinstance(func, ast.Attribute):
            if (isinstance(func.value, ast.Name)):
                return ("np" if func.value.id == "np" else func.value.id, func.attr)
            if (isinstance(func.value, ast.Attribute) and isinstance(func.value.value, ast.Name)
                    and func.value.value.id == "np"):
                # ``np.linalg.cholesky`` -> key ("np", "linalg.cholesky")
                return ("np", f"{func.value.attr}.{func.attr}")
        return None

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        # Captured BEFORE any rewriting: once the RHS is hoisted its arguments are temps, and the
        # note is meant to read as the numpy the kernel was written in.
        numpy_text = ast.unparse(node.value)
        self.generic_visit(node)
        # ``D[:] = np.repeat(...)``/``D[:, :] = np.transpose(...)``: canonicalise
        # the slice-LHS-with-call form to ``D = call(...)`` so the registered
        # call expander fires -- required for stockham_fft's ``y[:] =
        # np.reshape(...)`` and ``D[:] = np.repeat(...)``.
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name) and _is_full_slice_subscript(node.targets[0])
                and isinstance(node.value, ast.Call) and self._lookup(node.value) in NP_CALL_EXPANDERS):
            node.targets[0] = ast.Name(id=node.targets[0].value.id, ctx=ast.Store())
        # ``y = np.linalg.lstsq(A, b, rcond=...)[0]`` canonicalisation: strip
        # the trailing ``[0]`` subscript on a tuple-returning call so the
        # registered call expander fires on the bare call. Only lstsq/histogram
        # for now; extend as other tuple-returners (svd, eig, etc.) land.
        if (len(node.targets) == 1 and isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Call)
                and isinstance(node.value.slice, ast.Constant) and node.value.slice.value == 0):
            inner = node.value.value
            inner_key = self._lookup(inner)
            if inner_key in {("np", "linalg.lstsq"), ("np", "histogram")}:
                node.value = inner
        node.value, prelude = self._hoist_value(node.value)
        # Lower any prelude assigns that are themselves registered calls.
        prelude = self._lower_prelude_calls(prelude)
        # Reassigned local (lenet's ``x = relu(conv2d(x))`` chain): refresh shape_table[target] so
        # the NEXT statement sees the new shape. Strictly AFTER hoisting this RHS -- Python evaluates
        # the RHS against the OLD binding, and refreshing first made ``x = x @ w.T + b`` contract over
        # the RESULT's extent: out_features instead of in_features, silently dropping terms when
        # in > out and reading past the row when in < out.
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            self._update_shape_for_assign(node.targets[0].id, node.value)
        # Whole-array alias propagation: ``x = <Name>`` where the RHS is a Name
        # with a known shape gives ``x`` the same shape, so downstream visits
        # see ``x`` as an array. Also propagate ``local_dtypes``, else a
        # complex temp aliased to a fresh local loses its dtype tag and the
        # next call hoister synthesizes a non-complex temp.
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Name)
                and node.value.id in self.shape_table):
            self.shape_table[node.targets[0].id] = self.shape_table[node.value.id]
            rhs_dt = self.local_dtypes.get(node.value.id)
            if rhs_dt and node.targets[0].id not in self.local_dtypes:
                self.local_dtypes[node.targets[0].id] = rhs_dt
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            target = node.targets[0]
            if isinstance(node.value, ast.Call):
                key = self._lookup(node.value)
                expander = NP_CALL_EXPANDERS.get(key) if key else None
                if expander is not None and self._target_renders(key, node.value):
                    return node
                if expander is not None:
                    try:
                        expanded = _call_expander(expander,
                                                  target,
                                                  node.value.args,
                                                  node.value.keywords,
                                                  self.shape_table,
                                                  local_dtypes=self.local_dtypes,
                                                  fresh_local_allocs=self.fresh_local_allocs,
                                                  dim_aliases=self.dim_aliases)
                        # Linspace/arange/similar element-write expanders
                        # consume the original Assign, leaving the target
                        # dangling without a decl -- register a fresh-local
                        # allocation now so the emitter sees the shape downstream.
                        if (key in _ELEMENT_WRITE_EXPANDERS and target.id in self.shape_table
                                and target.id not in self.known_arrays):
                            # Not gated on the local being UNREGISTERED: a temp the call-hoister
                            # already spilled is registered as an array temp but still carries no
                            # allocation SITE, and the expander has just consumed the assignment
                            # that would have carried one. max_filter's hoisted running-max scan
                            # wrote its first element through a NULL pointer that way.
                            self.fresh_local_allocs.setdefault(target.id, tuple(self.shape_table[target.id]))
                            # ...and mark the allocation SITE. Registering the shape only gets the
                            # local DECLARED; a local whose extent depends on a body-computed scalar
                            # (histogram_equalization's ``cdf = np.cumsum(hist)``, shape ``(nbins,)``
                            # with ``nbins`` assigned in the body) is declared NULL at fn-top and
                            # malloc'd at its ``__hpcagent_bench_zeros__`` marker instead. The expander
                            # consumed the Assign that would have carried that marker, so without one
                            # the buffer stays NULL and the first store segfaults. The marker is a
                            # no-op for a fn-top-malloc'd local, so emitting it unconditionally is
                            # safe -- same rationale as ``_prepend_alloc_markers`` for matmul temps.
                            prelude = prelude + [_alloc_marker(target.id)]
                        # ``np.arange`` over integer bounds yields an integer
                        # iota (numpy intp) -- declare the local int64 so a
                        # gather index built from it (``q = j % nx``) is
                        # integer, not the float default (fft_3d).
                        if (key == ("np", "arange") and target.id not in self.local_dtypes
                                and all(_is_integer_expr(a, self.local_dtypes) for a in node.value.args)):
                            self.local_dtypes[target.id] = "int64"
                        tag_numpy_origin(expanded, numpy_text)
                        return prelude + expanded
                    except NotImplementedError:
                        pass
        # Partial-slice assignment target for a cumulative scan (``row_offsets[1:]
        # = np.cumsum(m_sizes)``): the full-slice case is canonicalised to a bare
        # Name above, but a shifted slice keeps its offset, routed to the
        # offset-aware cumulative expander.
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name) and isinstance(node.value, ast.Call)):
            key = self._lookup(node.value)
            if key in _SLICE_TARGET_EXPANDERS:
                try:
                    expanded = _call_expander(NP_CALL_EXPANDERS[key],
                                              node.targets[0],
                                              node.value.args,
                                              node.value.keywords,
                                              self.shape_table,
                                              local_dtypes=self.local_dtypes,
                                              fresh_local_allocs=self.fresh_local_allocs)
                    return prelude + expanded
                except NotImplementedError:
                    pass
        retargeted = _retarget_scalar_accumulator(node, prelude)
        if retargeted is not None:
            return retargeted
        if prelude:
            return prelude + [node]
        return node

    def _target_renders(self, key, call: ast.Call) -> bool:
        """The target claims this call as its own intrinsic, so leave it unexpanded.

        The shape table goes with it: the claim has to be decidable here, because past this point
        the loop nest the call would have become no longer exists to fall back to.
        """
        if self.native_call is None or key is None:
            return False
        return self.native_call(key, call, self.shape_table, self.native_dtypes)

    def _lower_prelude_calls(self, prelude: List[ast.stmt]) -> List[ast.stmt]:
        """Recursively lower any registered-call assigns inside the prelude
        that the call-hoister produced. The hoister synthesises ``__cb<n> =
        np.<op>(args)`` statements, each an Assign-to-Name with a registered
        call -- feed them through the same expander pipeline so the prelude
        emits as plain loops, not an unsupported np.<op> call.
        """
        out: List[ast.stmt] = []
        for stmt in prelude:
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Call)):
                key = self._lookup(stmt.value)
                expander = NP_CALL_EXPANDERS.get(key) if key else None
                if expander is not None and self._target_renders(key, stmt.value):
                    out.append(stmt)
                    continue
                if expander is not None:
                    try:
                        expanded = _call_expander(expander,
                                                  stmt.targets[0],
                                                  stmt.value.args,
                                                  stmt.value.keywords,
                                                  self.shape_table,
                                                  local_dtypes=self.local_dtypes,
                                                  fresh_local_allocs=self.fresh_local_allocs,
                                                  dim_aliases=self.dim_aliases)
                        # Same note as the direct path: the hoister split ``out = f(np.sum(a))`` into
                        # a temp assign, and it is THIS statement that becomes the loop nest.
                        tag_numpy_origin(expanded, ast.unparse(stmt.value))
                        # ...and the same auto-alloc. An expander that consumes the assign leaves
                        # its target with no allocation SITE, and a hoisted temp whose extent
                        # depends on a body-computed scalar is declared NULL at function top and
                        # malloc'd at its marker. Without one, max_filter's hoisted running-max
                        # scan wrote its first element through that NULL.
                        spilled = stmt.targets[0].id
                        if (key in _ELEMENT_WRITE_EXPANDERS and spilled in self.shape_table
                                and spilled not in self.known_arrays):
                            self.fresh_local_allocs.setdefault(spilled, tuple(self.shape_table[spilled]))
                            out.append(_alloc_marker(spilled))
                        out.extend(expanded)
                        # Integer-iota arange in the prelude (hoisted ``__cb =
                        # np.arange(1, 1025)``) keeps an int64 dtype so a derived
                        # gather index stays integer (fft_3d).
                        if (key == ("np", "arange") and stmt.targets[0].id not in self.local_dtypes
                                and all(_is_integer_expr(a, self.local_dtypes) for a in stmt.value.args)):
                            self.local_dtypes[stmt.targets[0].id] = "int64"
                        continue
                    except NotImplementedError:
                        pass
            out.append(stmt)
        return out

    def _flatten_visit_list(self, stmts):
        """Visit each stmt; flatten any nested lists returned by visits
        (visit_Assign can return ``[prelude..., assign]`` lists)."""
        out = []
        for s in stmts:
            r = self.visit(s)
            if isinstance(r, list):
                out.extend(r)
            else:
                out.append(r)
        return out

    def visit_If(self, node: ast.If) -> ast.AST:
        """Hoist any registered ``np.X(...)`` call inside the ``if`` test
        expression -- the common iterative-solver pattern ``if
        np.linalg.norm(r) < tol: break`` puts the call on the Compare LHS,
        where the Assign-only hoister never reaches it. Returns ``[prelude...,
        If]`` when hoisting happened."""
        node.body = self._flatten_visit_list(node.body)
        node.orelse = self._flatten_visit_list(node.orelse)
        node.test, prelude = self._hoist_value(node.test)
        prelude = self._lower_prelude_calls(prelude)
        if prelude:
            return prelude + [node]
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        node.body = self._flatten_visit_list(node.body)
        node.orelse = self._flatten_visit_list(node.orelse)
        node.test, prelude = self._hoist_value(node.test)
        prelude = self._lower_prelude_calls(prelude)
        if prelude:
            return prelude + [node]
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        node.value, prelude = self._hoist_value(node.value)
        prelude = self._lower_prelude_calls(prelude)
        retargeted = _retarget_scalar_accumulator(node, prelude)
        if retargeted is not None:
            return retargeted
        if prelude:
            return prelude + [node]
        return node


#: Public name for the extent oracle. The Fortran intrinsic gate has to ask the same question
#: lowering asks -- what rank does this operand actually have -- and a backend reaching across for a
#: leading-underscore name would be reaching for something this module never promised to keep.
iter_extent_of = _iter_extent_of
