"""Harvest local array shapes from constructor assignments."""

import ast
from types import NotImplementedType

from hpcagent_bench.translators.numpyto_common.frontend import collect_inlined_scalar_defs, dtype_from_constructor
from hpcagent_bench.translators.numpyto_common.lib_nodes.dims import (
    DIM_IDENT_RE,
    NP_ZEROS_ALIASES,
    SHAPE_READ_RE,
    shape_exprs_equal,
    substitute_dim_aliases,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import (
    broadcast_children,
    iter_extent_of,
    extent_is_scalar,
)
from hpcagent_bench.translators.numpyto_common.lowering.mathfuncs import NP_ELEMENTWISE
from hpcagent_bench.translators.numpyto_common.lowering.shape_reads import resolve_shape_token
from hpcagent_bench.translators.numpyto_common.numpy_desugar import np_submodule_attr


def branch_pin_(stmt: ast.stmt) -> tuple[dict[str, int], bool]:
    """The zero-pin an ``if`` puts on one of its two sides, and whether that side is the TAKEN one.

    ``if padding:`` / ``if pa == 0:`` / ``if tail:`` guard the conv and running-max ports'
    pad-or-alias pairs. Exactly one side of such a test runs with the scalar equal to zero, and
    that is what makes ``h + 2 * padding`` and ``h`` one buffer rather than the two shapes the
    rebinding guard refuses. Returns an empty pin for a test this cannot invert exactly: only a
    bare name, ``name != 0``, ``name == 0`` and ``name > 0`` (whose false side is zero because an
    extent knob is non-negative -- a negative padding describes no array).
    """
    if not isinstance(stmt, ast.If):
        return {}, False
    test = stmt.test
    if isinstance(test, ast.Name):
        return {test.id: 0}, False
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.left, ast.Name)):
        return {}, False
    right = test.comparators[0]
    if not (isinstance(right, ast.Constant) and not isinstance(right.value, bool) and right.value == 0):
        return {}, False
    op = test.ops[0]
    if isinstance(op, (ast.NotEq, ast.Gt)):
        return {test.left.id: 0}, False
    if isinstance(op, ast.Eq):
        return {test.left.id: 0}, True
    return {}, False


def substitute_ints(token: str, values: dict[str, int]) -> str:
    """``token`` with each named scalar replaced by its assumed integer value."""

    class Sub_(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id not in values:
                return node
            return ast.copy_location(ast.Constant(value=values[node.id]), node)

    return ast.unparse(ast.fix_missing_locations(Sub_().visit(ast.parse(token, mode="eval").body)))


def shapes_agree_under(
    known: tuple[str, ...],
    candidate: tuple[str, ...],
    assume: dict[str, int],
    aliases: dict[str, str],
    shapes: dict[str, tuple[str, ...]],
) -> bool:
    """Whether two shape token tuples denote the same extent once ``assume`` is substituted.

    The pin goes in FIRST, then ``aliases``: one branch spells the extent with the kernel's own
    dimension locals (``h`` off ``x.shape[2]``) and the other with the declared symbol
    (``height``), so without the expansion ``h + 2 * 0`` and ``height`` compare unequal -- but the
    pinned scalar is often an alias itself (``pad`` for ``(kernel_size - 1) // 2``), and expanding
    it away first would leave the pin with nothing to bind.
    """
    return all(
        shape_exprs_equal(
            substitute_dim_aliases(substitute_ints(a, assume), aliases, shapes),
            substitute_dim_aliases(substitute_ints(b, assume), aliases, shapes),
        )
        for a, b in zip(known, candidate)
    )


def ctor_shape_arg(call: ast.Call) -> ast.expr | None:
    """Return the shape argument of an array constructor call
    (``np.zeros/empty/ones/ndarray(...)``): the first positional arg, or the
    ``shape=`` keyword when there is none (``np.ndarray(shape=(nlev, klon))`` --
    cloudsc's local declarations)."""
    if call.args:
        return call.args[0]
    for kw in call.keywords:
        if kw.arg == "shape":
            return kw.value
    return None


def is_scalar_helper_call(node: ast.AST, scalar_helpers: set[str] | None) -> bool:
    """Whether ``node`` calls a kernel helper emitted as a by-value SCALAR function.

    Such a call is rank 0 whatever its arguments are. :func:`iter_extent_of` reads a call it does
    not recognise as ELEMENTWISE and answers with the broadcast join of the arguments, which sizes
    a reduction's scalar result like the array it reduces -- and the caller then broadcasts the
    call over that buffer, one invocation per element.
    """
    return (
        bool(scalar_helpers)
        and isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in scalar_helpers
    )


def harvest_local_shapes(
    tree: ast.AST,
    shape_table: dict[str, tuple[str, ...]],
    dtype_table: dict[str, str] | None = None,
    scalar_helpers: set[str] | None = None,
) -> None:
    """Pre-scan the body for ``name = np.<alloc>(...)`` and seed the
    shape table with the inferred output shapes.

    Recognises ``zeros / empty / ones / zeros_like / empty_like`` and
    ``np.copy / transpose / triu / flip / linalg.cholesky / outer`` --
    any registered allocator-style call that has a deterministic
    output shape from its args. Run before LibNodeRewriter so the
    downstream call-hoister / scalarizer see ``Q``'s shape when
    visiting ``Q[:, k]``.

    Also populates ``dtype_table`` (if provided) with the dtype hint
    from the constructor's ``dtype=`` kwarg, so the emitter can
    declare ``X = np.zeros((N,), dtype=np.complex128)`` as
    ``double _Complex X[N]``.
    """
    # A name bound BOTH ways -- ``padded = x`` in one branch, ``padded = np.zeros(...)`` in the
    # other -- must take the ALLOCATION's shape: the alias is derived, the allocation is the
    # declaration, and the allocation is the larger of the two wherever the branch exists to avoid
    # it (conv_standard_1d's zero-padded buffer). ast.walk is not source order, so aliases are
    # deferred and applied only to targets no allocation claimed.
    aliases: list[tuple[str, str]] = []
    for stmt in ast.walk(tree):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue
        rhs = optional_constructor(stmt.value)
        if isinstance(rhs, ast.Name):
            # Name = Name alias -- inherit shape and dtype from the source, after every allocation.
            aliases.append((target.id, rhs.id))
            continue
        harvest_assign(target.id, rhs, shape_table, dtype_table, scalar_helpers)
    for name, src in aliases:
        src_shape = shape_table.get(src)
        if src_shape and name not in shape_table:
            shape_table[name] = tuple(src_shape)
        if dtype_table is not None:
            src_dt = dtype_table.get(src)
            if src_dt is not None and name not in dtype_table:
                dtype_table[name] = src_dt


def optional_constructor(rhs: ast.expr) -> ast.expr:
    """``X = np.zeros(.., dtype=..) if cond else None`` (a buffer allocated only in one branch) read as
    its constructor, so the local is typed and sized like a direct ``X = np.zeros(..)``: untyped, a
    complex buffer would default to real, which C++ rejects at the complex accumulation."""
    if isinstance(rhs, ast.IfExp):
        ctor = [b for b in (rhs.body, rhs.orelse) if isinstance(b, ast.Call)]
        none_br = [b for b in (rhs.body, rhs.orelse) if isinstance(b, ast.Constant) and b.value is None]
        if len(ctor) == 1 and len(none_br) == 1:
            return ctor[0]
    return rhs


def harvest_assign(
    target_id: str,
    rhs: ast.expr,
    shape_table: dict[str, tuple[str, ...]],
    dtype_table: dict[str, str] | None,
    scalar_helpers: set[str] | None,
) -> None:
    """Seed the shape (and dtype) of ``target_id = rhs``."""
    # ``nxt = data[partner]`` -- a gather or a slice of an array carries the BASE's dtype, not the
    # sweep's float default (an int64 value round-tripped through a float temp).
    if isinstance(rhs, ast.Subscript) and isinstance(rhs.value, ast.Name) and dtype_table is not None:
        if target_id not in dtype_table:
            src_dtype = dtype_table.get(rhs.value.id)
            if src_dtype is not None:
                dtype_table[target_id] = src_dtype
    # ``np.linalg.<op>`` is a TWO-level attribute the single-level ``np.<attr>`` gate below never
    # matches: register what the solve / inv / cholesky expanders write -- ``solve`` returns x with
    # b's shape (not the square A's); ``inv`` / ``cholesky`` are shape-preserving.
    linalg_op = np_submodule_attr(rhs, "linalg")
    if linalg_op in ("solve", "inv", "cholesky"):
        source_arg = rhs.args[1] if linalg_op == "solve" and len(rhs.args) >= 2 else (rhs.args[0] if rhs.args else None)
        if isinstance(source_arg, ast.Name):
            linalg_source_shape = shape_table.get(source_arg.id)
            if linalg_source_shape:
                shape_table[target_id] = tuple(linalg_source_shape)
        return
    if (
        isinstance(rhs, ast.Call)
        and isinstance(rhs.func, ast.Attribute)
        and isinstance(rhs.func.value, ast.Name)
        and rhs.func.value.id == "np"
    ):
        harvest_np_call(target_id, rhs, shape_table, dtype_table)
        return
    if is_scalar_helper_call(rhs, scalar_helpers):
        return
    # Last-ditch: a BinOp / UnaryOp / Compare / BoolOp / Subscript whose operands have known shapes
    # mirrors the (broadcast / slice / gather) extent -- ``x = a + b``, a boolean mask, a gather local
    # ``nb = neigh[:, j]`` -- so the downstream rewriters resolve ``arr[nb]`` to ``arr[nb[i]]``.
    # ``ast.Call`` covers a method-form shape op the pre-normalise harvest sees before it becomes
    # ``np.<fn>`` (``X = (Yf @ C).reshape(shp)``). An all-size-1 broadcast is a SCALAR local (see
    # extent_is_scalar).
    if (
        isinstance(rhs, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Subscript, ast.Call))
        and target_id not in shape_table
    ):
        ext = iter_extent_of(rhs, shape_table)
        if ext is not None and not extent_is_scalar(ext):
            shape_table[target_id] = tuple(ast.unparse(e) for e in ext)


def harvest_np_call(
    target_id: str, rhs: ast.Call, shape_table: dict[str, tuple[str, ...]], dtype_table: dict[str, str] | None
) -> None:
    """``target = np.<attr>(...)``: its dtype from a ``dtype=`` hint, and the output shape the call
    determines from its args."""
    if dtype_table is not None:
        dt = dtype_from_constructor(rhs)
        if dt is not None:
            dtype_table[target_id] = dt
    attr = rhs.func.attr
    if attr in NP_ZEROS_ALIASES:
        harvest_zeros_like(target_id, rhs, shape_table)
    elif (counted := counted_constructor_shape(attr, rhs.args)) is not UNHANDLED:
        if counted is not None:
            shape_table[target_id] = counted
    elif attr in NP_ELEMENTWISE and rhs.args:
        # Elementwise broadcast ops: the broadcast of the args' extents.
        ext = broadcast_children(rhs.args, shape_table)
        if ext is not None:
            shape_table[target_id] = tuple(ast.unparse(e) for e in ext)
    elif (
        attr in {"copy", "asarray", "ascontiguousarray", "triu", "flip"}
        and rhs.args
        and isinstance(rhs.args[0], ast.Name)
    ):
        src_shape = shape_table.get(rhs.args[0].id)
        if src_shape:
            shape_table[target_id] = tuple(src_shape)
    elif attr == "transpose" and rhs.args and isinstance(rhs.args[0], ast.Name):
        harvest_transpose(target_id, rhs, shape_table)
    elif target_id not in shape_table:
        # Any other ``np.<func>(...)`` whose result shape ``iter_extent_of`` derives: axis-aware
        # reductions (``rsq = np.sum(dpos * dpos, axis=2)``) and elementwise math wrapping one.
        ext = iter_extent_of(rhs, shape_table)
        if ext is not None:
            shape_table[target_id] = tuple(ast.unparse(e) for e in ext)


#: What :func:`counted_constructor_shape` returns for a call it does not size.
UNHANDLED = NotImplemented


def counted_constructor_shape(attr: str, args: list[ast.expr]) -> tuple[str, ...] | None | NotImplementedType:
    """The shape a constructor states in its count arguments: ``np.eye(M[, N])`` -> ``(M, M | N)``,
    ``np.linspace(start, stop, n)`` -> ``(n,)`` (numpy's default of 50 is refused by the expander),
    ``np.arange(stop)`` -> ``(stop,)`` (None for the multi-argument form), ``np.identity(n)`` ->
    ``(n, n)``; :data:`UNHANDLED` for anything else."""
    if attr == "eye" and args:
        first_tok = ast.unparse(args[0])
        return (first_tok, ast.unparse(args[1]) if len(args) >= 2 else first_tok)
    if attr == "linspace" and len(args) >= 3:
        return (ast.unparse(args[2]),)
    if attr == "arange" and args:
        return (ast.unparse(args[0]),) if len(args) == 1 else None
    if attr == "identity" and args:
        tok = ast.unparse(args[0])
        return (tok, tok)
    return UNHANDLED


def harvest_zeros_like(target_id: str, rhs: ast.Call, shape_table: dict[str, tuple[str, ...]]) -> None:
    """``np.zeros_like(other)`` -> other's shape; ``np.zeros((N, M))`` / ``np.ndarray(shape=(N, M))`` ->
    the first positional arg or the ``shape=`` keyword: a tuple, a Name, an int, ``x.shape``, or a
    scalar ``arr.shape[i]`` (or arithmetic over it) resolved like a shape-tuple element."""
    attr = rhs.func.attr
    if attr.endswith("_like") and rhs.args and isinstance(rhs.args[0], ast.Name):
        src_shape = shape_table.get(rhs.args[0].id)
        if src_shape:
            shape_table[target_id] = tuple(src_shape)
        return
    shape_arg = ctor_shape_arg(rhs)
    if shape_arg is None:
        return
    if isinstance(shape_arg, (ast.Tuple, ast.List)):
        shape_table[target_id] = tuple(resolve_shape_token(e, shape_table) for e in shape_arg.elts)
    elif isinstance(shape_arg, ast.Name):
        shape_table[target_id] = (shape_arg.id,)
    elif isinstance(shape_arg, ast.Constant) and isinstance(shape_arg.value, int):
        shape_table[target_id] = (str(shape_arg.value),)
    elif isinstance(shape_arg, ast.Attribute) and shape_arg.attr == "shape":
        if isinstance(shape_arg.value, ast.Name):
            src = shape_table.get(shape_arg.value.id)
            if src is not None:
                shape_table[target_id] = tuple(src)
    elif isinstance(shape_arg, (ast.Subscript, ast.BinOp)):
        shape_table[target_id] = (resolve_shape_token(shape_arg, shape_table),)


def harvest_transpose(target_id: str, rhs: ast.Call, shape_table: dict[str, tuple[str, ...]]) -> None:
    """``np.transpose(A[, axes])``: A's shape permuted, or reversed without an axes tuple."""
    src_shape = shape_table.get(rhs.args[0].id)
    if not src_shape:
        return
    if len(rhs.args) >= 2 and isinstance(rhs.args[1], ast.Tuple):
        perm = [e.value for e in rhs.args[1].elts if isinstance(e, ast.Constant) and isinstance(e.value, int)]
        if len(perm) == len(src_shape):
            shape_table[target_id] = tuple(src_shape[p] for p in perm)
    else:
        shape_table[target_id] = tuple(reversed(src_shape))


def collect_dim_aliases(tree: ast.AST, array_names: set[str]) -> dict[str, str]:
    """Map each DIMENSION local to its definition, for :func:`lib_nodes.dims_agree`.

    A kernel names its own dimensions off a parameter's shape -- ``batch, channels, h, w =
    x.shape``, which the tuple desugar has already folded to ``batch = batch_size`` / ``channels =
    embed_dim``. Those locals then spell shape tokens the ``init.shapes`` side spells with the
    symbol, so two operands of one contraction disagree textually while denoting the same extent.

    ``collect_inlined_scalar_defs`` with no prefix over-collects for this purpose: its
    scalar-vs-array test is structural (a BinOp of Names looks like a dimension), so an ARRAY
    expression -- vision_attention's ``resid = attn_out + tokens`` -- comes back as a candidate.
    Substituting one into a shape token would be nonsense, so a name is kept only when neither it
    nor any identifier it reads is a known array.

    A ``.shape[i]`` read is the exception the filter must not eat: ``__inl91_c =
    __inl8_y.shape[3]`` names an array precisely to read a DIMENSION off it, and it is the only
    form swin's inlined stages have. Those reads are masked out before the array test and resolved
    later, against the then-current shape table, by :func:`lib_nodes.substitute_dim_aliases`.
    """
    candidates = collect_inlined_scalar_defs(tree, None)
    return {
        name: rhs
        for name, rhs in candidates.items()
        if name not in array_names and not (array_names & set(DIM_IDENT_RE.findall(SHAPE_READ_RE.sub("", rhs))))
    }
