"""Static rank (and tuple-length) inference over a function body."""

import ast
import re
from collections.abc import Callable
from collections.abc import Iterator

from hpcagent_bench.translators.numpyto_common.lib_nodes import (
    iter_extent_of,
    parse_einsum_subscripts,
    extent_is_scalar,
)
from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet
from hpcagent_bench.translators.numpyto_common.subscripts import is_ellipsis, is_newaxis
from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import (
    REDUCE_FNS,
    LIKE_CTORS,
    SHAPE_CTORS,
    const_int,
    np_attr,
    tuple_len,
    np_submodule_attr,
    reachable_functions,
)


#: Tuple-shape lengths known to :func:`expr_rank` while :func:`rank_table` iterates, so ``.reshape(name)``
#: reports the tuple's length instead of the rank-1 guess a bare Name gets.
active_tuple_lengths: dict[str, int | None] | None = None


def newaxis_singletons(value: ast.AST, rank: int) -> frozenset:
    """Axes of ``value`` a literal ``None`` in its own subscript pins to extent 1.

    Index arrays in one gather broadcast against each other, so an open mesh (``g_z[:, None, None]``,
    ``g_y[None, :, None]``) names different shapes of one rank; indexing each at the full iterator tuple
    reads past its singleton axes without raising. An expression whose axes cannot be placed (an ellipsis,
    an advanced index, not a subscript) reports no singleton.
    """
    if not isinstance(value, ast.Subscript):
        return frozenset()
    elts = list(value.slice.elts) if isinstance(value.slice, ast.Tuple) else [value.slice]
    if any(is_ellipsis(e) for e in elts):
        return frozenset()
    axes, out = [], 0
    for e in elts:
        if is_newaxis(e):
            axes.append(out)
            out += 1
        elif isinstance(e, ast.Slice):
            out += 1
        elif isinstance(e, ast.Constant) and isinstance(e.value, int) and not isinstance(e.value, bool):
            continue  # a scalar index consumes its base axis and adds none
        else:
            return frozenset()
    return frozenset(a for a in axes if a < rank)


def axis_count(args: list[ast.expr], keywords: list[ast.keyword]) -> int | None:
    """How many axes an ``axis=`` argument names. ``None`` when it is absent (numpy's "every
    size-1 axis", which is not a compile-time count) or not a literal."""
    kw = {k.arg: k.value for k in keywords}
    axis = kw.get("axis") or (args[0] if args else None)
    if axis is None:
        return None
    if isinstance(axis, (ast.Tuple, ast.List)):
        return len(axis.elts)
    return 1 if isinstance(axis, ast.Constant) and isinstance(axis.value, int) else None


def expr_rank(value: ast.AST, ranks: dict[str, int]) -> int | None:
    """Best-effort ndim of an expression given the current rank table."""
    handler = RANK_HANDLERS.get(type(value))
    return None if handler is None else handler(value, ranks)


def name_rank(value: ast.Name, ranks: dict[str, int]) -> int | None:
    return ranks.get(value.id)


def constant_rank(value: ast.Constant, ranks: dict[str, int]) -> int | None:
    return 0 if isinstance(value.value, (bool, int, float, complex)) else None


def attribute_rank(value: ast.Attribute, ranks: dict[str, int]) -> int | None:
    # ``A.T`` reverses the axes, keeping the rank.
    return expr_rank(value.value, ranks) if value.attr == "T" else None


def binop_rank(value: ast.BinOp, ranks: dict[str, int]) -> int | None:
    lr = expr_rank(value.left, ranks)
    rr = expr_rank(value.right, ranks)
    if isinstance(value.op, ast.MatMult):
        if lr is None or rr is None:
            return None
        # matmul: 1-D operands contract to a scalar; otherwise the result
        # keeps the larger batch rank (numpy stacks/broadcasts leading axes).
        return 0 if lr == 1 and rr == 1 else max(lr, rr)
    return max([r for r in (lr, rr) if r is not None], default=None)


def unaryop_rank(value: ast.UnaryOp, ranks: dict[str, int]) -> int | None:
    return expr_rank(value.operand, ranks)


def list_rank(value: ast.List, ranks: dict[str, int]) -> int | None:
    # ``np.array([a, b])``: a list literal adds one axis over its elements' rank.
    return 1 + max((expr_rank(e, ranks) or 0) for e in value.elts) if value.elts else 1


def compare_rank(value: ast.Compare, ranks: dict[str, int]) -> int | None:
    # A boolean mask keeps its operands' rank.
    rs = [expr_rank(value.left, ranks)] + [expr_rank(c, ranks) for c in value.comparators]
    return max([r for r in rs if r is not None], default=None)


def boolop_rank(value: ast.BoolOp, ranks: dict[str, int]) -> int | None:
    rs = [expr_rank(v, ranks) for v in value.values]
    return max([r for r in rs if r is not None], default=None)


def tuple_index_drop(elts: list[ast.expr], ranks: dict[str, int]) -> int:
    """Axes a tuple index removes from its base.

    A slice or an ellipsis drops nothing and a newaxis adds one. The advanced indices (scalars and index
    arrays) broadcast against each other: the block consumes one base axis per index and gives back ONE
    broadcast shape, a drop of ``n_adv - max(rank)``. So ``a[i, j]`` drops 2, ``A[ia, ib]`` with both rank 2
    drops 0, and ``x[:, idx]`` with a 1-D ``idx`` drops 0. An index of unknown rank counts as a scalar.
    """
    drop = 0
    adv_ranks = []
    for e in elts:
        if isinstance(e, ast.Slice) or is_ellipsis(e):
            continue
        if is_newaxis(e):
            drop -= 1
            continue
        idx_rank = expr_rank(e, ranks)
        adv_ranks.append(0 if idx_rank is None else idx_rank)
    if adv_ranks:
        drop += len(adv_ranks) - max(adv_ranks)
    return drop


def is_tuple_call(node: ast.expr) -> bool:
    """``tuple(...)``, the spelling of a whole index built from a sequence."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "tuple"


def name_index_rank(base: int, index: ast.Name, ranks: dict[str, int]) -> int | None:
    """Rank of ``a[name]``.

    A Name of unknown or zero rank is a scalar (a loop iterator) and drops one axis. An array index may be
    a gather (adds ``rank - 1`` axes) or a boolean mask (removes them); the two readings agree only for a
    rank-1 index, so any higher rank is unknown rather than guessed.
    """
    idx_rank = ranks.get(index.id)
    if idx_rank is None or idx_rank < 1:
        return base - 1
    return base if idx_rank == 1 else None


def subscript_rank(value: ast.Subscript, ranks: dict[str, int]) -> int | None:
    base = expr_rank(value.value, ranks)
    if base is None:
        return None
    sl = value.slice
    if isinstance(sl, ast.Slice) or is_ellipsis(sl):
        return base  # a slice or ``a[...]`` keeps every axis
    if is_newaxis(sl):
        return base + 1  # a[None] / a[np.newaxis] -- a newaxis adds a dimension
    if isinstance(sl, ast.Tuple):
        return base - tuple_index_drop(sl.elts, ranks)
    if is_tuple_call(sl):
        # ``A[tuple(axes)]`` is the WHOLE index, one entry per axis; what it drops depends on what the
        # sequence holds, which is not visible here.
        return None
    if isinstance(sl, ast.Name):
        return name_index_rank(base, sl, ranks)
    return base - 1  # single integer index


def call_rank(value: ast.Call, ranks: dict[str, int]) -> int | None:
    func = value.func
    is_abs = isinstance(func, ast.Name) and func.id == "abs"
    if value.args and (is_abs or np_submodule_attr(value, "fft")):
        return expr_rank(value.args[0], ranks)  # builtin abs is elementwise; fft/ifft/fftn... preserve rank
    attr = np_attr(value)
    if attr is None:
        return method_call_rank(value, ranks)
    handler = NP_CALL_RANKS.get(attr)
    return np_fallthrough_rank(value, attr, ranks) if handler is None else handler(value, attr, ranks)


def ufunc_rank(value: ast.Call, ranks: dict[str, int]) -> int | None:
    # Remaining np.<fn>(...) are elementwise/broadcasting ufuncs (abs, sqrt, exp, less, minimum,
    # where, conj, ...) -> max of arg ranks. Rank-changing ops have their own handler.
    rs = [expr_rank(a, ranks) for a in value.args]
    return max([r for r in rs if r is not None], default=None)


def np_fallthrough_rank(value: ast.Call, attr: str, ranks: dict[str, int]) -> int | None:
    """A numpy call no dedicated rule decided: the method spellings its name matches, else a ufunc."""
    if attr in ("astype", "copy"):
        return expr_rank(value.func.value, ranks) if isinstance(value.func, ast.Attribute) else None
    if attr == "ravel":
        return 1
    if attr == "reshape" and value.args:
        return reshape_args_rank(value, lambda: ufunc_rank(value, ranks))
    return ufunc_rank(value, ranks)


def method_call_rank(value: ast.Call, ranks: dict[str, int]) -> int | None:
    """``x.astype(dt)`` / ``x.copy()`` / ``x.conj()`` keep the receiver's rank; ``ravel``/``flatten``
    are one axis. The conjugate forms are method-only: ``np.conj(z)`` names ``np`` as its receiver."""
    func = value.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr in ("astype", "copy", "conj", "conjugate"):
        return expr_rank(func.value, ranks)
    if func.attr in ("ravel", "flatten"):
        return 1
    if func.attr == "reshape" and value.args:
        return reshape_args_rank(value, lambda: None)
    return None


def reshape_args_rank(value: ast.Call, undecided: Callable[[], int | None]) -> int | None:
    """``x.reshape((a, b))`` method form; ``undecided`` answers a shape argument this cannot count.

    The multi-arg spelling has one positional argument per dimension, so the rank is the argument count
    whatever each expression is (``X.reshape(-1, X.shape[-1])`` is rank 2). A lone argument counts as one
    axis only when it is a literal or a Name not bound to a tuple: any other expression may be a
    whole shape tuple.
    """
    a0 = value.args[0]
    n = tuple_len(a0)
    lengths = active_tuple_lengths or {}
    if n is None and len(value.args) == 1 and isinstance(a0, ast.Name) and a0.id in lengths:
        # A shape tuple whose length is not known yet has no rank to report: 1 would be a guess.
        return lengths[a0.id]
    # A lone integer literal, ``-1`` included, is one axis; ``-1`` parses as a UnaryOp, not a Constant.
    if n is None and (len(value.args) > 1 or isinstance(a0, (ast.Name, ast.Constant)) or const_int(a0) is not None):
        n = len(value.args)
    return undecided() if n is None else n


def one_axis_rank(value: ast.Call, attr: str, ranks: dict[str, int]) -> int | None:
    return 1  # np.arange / np.linspace are always 1-D


def reduce_call_rank(value: ast.Call, attr: str, ranks: dict[str, int]) -> int | None:
    if not value.args:
        return np_fallthrough_rank(value, attr, ranks)
    base = expr_rank(value.args[0], ranks)
    if base is None:
        return None
    kw = {k.arg: k.value for k in value.keywords}
    # keepdims=True keeps every reduced axis at extent 1, so the rank is unchanged.
    keep = kw.get("keepdims")
    if isinstance(keep, ast.Constant) and keep.value is True:
        return base
    ax = kw.get("axis") or (value.args[1] if len(value.args) > 1 else None)
    if ax is None:
        return 0  # full reduction -> scalar
    if isinstance(ax, (ast.Tuple, ast.List)):
        return base - len(ax.elts)
    return base - 1  # single reduced axis


def shape_ctor_rank(value: ast.Call, attr: str, ranks: dict[str, int]) -> int | None:
    if value.args:
        a0 = value.args[0]
        n = tuple_len(a0)
        if n is not None:
            return n
        if isinstance(a0, ast.Attribute) and a0.attr == "shape":
            return expr_rank(a0.value, ranks)  # np.zeros(C.shape, ...) keeps C's rank
        if isinstance(a0, (ast.Name, ast.Constant)) or expr_rank(a0, ranks) == 0:
            return 1  # 1-D length, including a scalar expression such as ``n_rows * block``
    return np_fallthrough_rank(value, attr, ranks)


def first_arg_rank(value: ast.Call, attr: str, ranks: dict[str, int]) -> int | None:
    """``np.zeros_like(a)`` / ``np.copy(a)`` / ``np.asarray(a)`` keep their operand's rank."""
    return expr_rank(value.args[0], ranks) if value.args else np_fallthrough_rank(value, attr, ranks)


def np_reshape_rank(value: ast.Call, attr: str, ranks: dict[str, int]) -> int | None:
    n = tuple_len(value.args[1]) if len(value.args) >= 2 else None
    if n is None and len(value.args) >= 2 and const_int(value.args[1]) is not None:
        n = 1  # ``np.reshape(a, -1)``: one integer extent is one axis
    return np_fallthrough_rank(value, attr, ranks) if n is None else n


def diag_rank(value: ast.Call, attr: str, ranks: dict[str, int]) -> int | None:
    # 2-D in extracts the diagonal (1-D out), 1-D in builds the matrix (2-D out).
    if not value.args:
        return np_fallthrough_rank(value, attr, ranks)
    return {1: 2, 2: 1}.get(expr_rank(value.args[0], ranks))


def take_rank(value: ast.Call, attr: str, ranks: dict[str, int]) -> int | None:
    # ``np.take(a, idx, axis=k)`` replaces axis k by the INDEX's own rank, so a scalar index drops it.
    if len(value.args) < 2:
        return np_fallthrough_rank(value, attr, ranks)
    base = expr_rank(value.args[0], ranks)
    idx = expr_rank(value.args[1], ranks)
    return None if base is None or idx is None else base - 1 + idx


def expand_dims_rank(value: ast.Call, attr: str, ranks: dict[str, int]) -> int | None:
    if not value.args:
        return np_fallthrough_rank(value, attr, ranks)
    base = expr_rank(value.args[0], ranks)
    return None if base is None else base + 1


def squeeze_rank(value: ast.Call, attr: str, ranks: dict[str, int]) -> int | None:
    if not value.args:
        return np_fallthrough_rank(value, attr, ranks)
    base = expr_rank(value.args[0], ranks)
    axes = axis_count(value.args[1:], value.keywords)
    return None if base is None or axes is None else base - axes


def matmul_call_rank(value: ast.Call, attr: str, ranks: dict[str, int]) -> int | None:
    if len(value.args) != 2:
        return np_fallthrough_rank(value, attr, ranks)
    return expr_rank(ast.BinOp(left=value.args[0], op=ast.MatMult(), right=value.args[1]), ranks)


def tensordot_rank(value: ast.Call, attr: str, ranks: dict[str, int]) -> int | None:
    # A contraction, not a broadcast: each operand loses its contracted axes.
    if len(value.args) < 2:
        return np_fallthrough_rank(value, attr, ranks)
    la, lb = expr_rank(value.args[0], ranks), expr_rank(value.args[1], ranks)
    n = tensordot_contracted(value)
    return None if la is None or lb is None or n is None else la + lb - 2 * n


def einsum_rank(value: ast.Call, attr: str, ranks: dict[str, int]) -> int | None:
    """``np.einsum('gai,ai->ga', ...)`` has its OUTPUT subscripts' rank, not its largest operand's.

    A spec that is not a literal string, or that holds an ellipsis, stays unranked.
    """
    spec = value.args[0] if value.args else None
    if not (isinstance(spec, ast.Constant) and isinstance(spec.value, str)) or "..." in spec.value:
        return None
    return len(parse_einsum_subscripts(spec.value)[1])


#: ``np.<attr>`` calls whose rank has its own rule; every other numpy call is elementwise.
NP_CALL_RANKS: dict[str, Callable[[ast.Call, str, dict[str, int]], int | None]] = {
    "einsum": einsum_rank,
    "arange": one_axis_rank,
    "linspace": one_axis_rank,
    **dict.fromkeys(sorted(REDUCE_FNS), reduce_call_rank),
    **dict.fromkeys(sorted(SHAPE_CTORS), shape_ctor_rank),
    **dict.fromkeys((*sorted(LIKE_CTORS), "copy", "ascontiguousarray", "asarray", "array"), first_arg_rank),
    "reshape": np_reshape_rank,
    "diag": diag_rank,
    "take": take_rank,
    "expand_dims": expand_dims_rank,
    "squeeze": squeeze_rank,
    "matmul": matmul_call_rank,
    "tensordot": tensordot_rank,
}


#: One rank rule per expression node type; a type absent here has no rank.
RANK_HANDLERS: dict[type, Callable[..., int | None]] = {
    ast.Name: name_rank,
    ast.Constant: constant_rank,
    ast.Attribute: attribute_rank,
    ast.BinOp: binop_rank,
    ast.UnaryOp: unaryop_rank,
    ast.List: list_rank,
    ast.Compare: compare_rank,
    ast.BoolOp: boolop_rank,
    ast.Subscript: subscript_rank,
    ast.Call: call_rank,
}


def tensordot_contracted(value: ast.Call) -> int | None:
    """Axes each operand of ``np.tensordot`` contracts: an integer ``axes`` contracts that many, a
    pair of axis sequences contracts one per listed axis, a pair of bare ints contracts one."""
    kw = {k.arg: k.value for k in value.keywords}
    axes = kw["axes"] if "axes" in kw else (value.args[2] if len(value.args) > 2 else None)
    if axes is None:
        return 2  # numpy's own default
    n = const_int(axes)
    if n is not None:
        return n
    if isinstance(axes, (ast.Tuple, ast.List)) and len(axes.elts) == 2:
        first = axes.elts[0]
        if isinstance(first, (ast.Tuple, ast.List)):
            return len(first.elts)
        return None if const_int(first) is None else 1
    return None


def call_return_rank(value: ast.AST, call_returns: dict[str, int]) -> int | None:
    """Rank of ``helper(...)`` for a local function with a known return rank, which ``expr_rank`` cannot see."""
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        return call_returns.get(value.func.id)
    return None


#: An identifier inside a shape token, so a declared extent can be told from an array name.
IDENT_RE = r"[A-Za-z_][A-Za-z0-9_]*"


def name_value_pairs(tree: ast.AST) -> Iterator[tuple[str, ast.expr]]:
    """Every ``name = <expr>`` binding, including the elements of a parallel tuple assignment.

    ``X, Y, sigma = Y, Ynew, sigma_new`` is three bindings; a loop-carried swap needs them as forward edges.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            yield target.id, node.value
        elif (
            isinstance(target, ast.Tuple)
            and isinstance(node.value, ast.Tuple)
            and len(target.elts) == len(node.value.elts)
        ):
            for t, v in zip(target.elts, node.value.elts):
                if isinstance(t, ast.Name):
                    yield t.id, v


def extent_tokens(
    value: ast.AST,
    table: dict[str, tuple[str, ...]],
    tuple_locals: frozenset[str] = frozenset(),
    arrays: frozenset[str] = frozenset(),
) -> tuple[str, ...] | None:
    """``value``'s shape as unparsed tokens, or ``None`` when this round cannot say.

    An extent still spelled through a ``.shape`` read is reported as unknown rather than recorded:
    it is the SAME extent as the declared one under a different name, and recording it would make
    the two bindings of one local look like a disagreement and drop the local entirely. The joint
    fixpoint in :func:`resolve_shape_reads` re-reads it once the read has been rewritten.

    ``tuple_locals`` names the locals bound to a tuple: ``x.reshape(shp)`` reads ``shp`` as one dimension,
    a whole rank off for a tuple, and a wrong answer propagates where no answer does not.
    """
    # An unsized ARRAY operand makes the whole expression unsized: ``iter_extent_of`` reports a broadcast's
    # known side, which is wrong for ``v[..., None] * X`` while ``X`` is unknown. Scalars contribute no axis
    # and are never in the table, so only arrays count.
    if any(n.id in arrays and n.id not in table for n in ast.walk(value) if isinstance(n, ast.Name)):
        return None
    ext = iter_extent_of(value, table)
    if ext is None or extent_is_scalar(ext):
        return None
    toks = tuple(ast.unparse(e) for e in ext)
    if any(".shape" in t for t in toks):
        return None
    return None if any(t in tuple_locals for t in toks) else toks


def shape_table(tree: ast.AST, seed: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    """Propagate full shapes across straight-line assignments to a fixpoint.

    The rank table's twin. A backward walk from a read to its definition cannot answer a rebound name or a
    cycle (``X, Y = Y, Ynew`` in a loop); a forward pass takes the extent in from the seed and closes the
    cycle on the next round.

    A name whose bindings do not AGREE is dropped, as in :func:`drop_rank_conflicts`. A declared array
    keeps its declared shape whatever a rebinding makes it.
    """
    tuple_locals = frozenset(
        node.targets[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, (ast.Tuple, ast.List))
    )
    pairs = list(name_value_pairs(tree))
    # The rank table only says which names are ARRAYS. A name it cannot rank counts as one: refusing to
    # size an expression costs a resolution, reporting the other operand's shape costs a wrong one. A
    # manifest symbol is a declared extent, never an array.
    ranks = rank_table(tree, {k: len(v) for k, v in seed.items()})
    symbols = {ident for shape in seed.values() for tok in shape for ident in re.findall(IDENT_RE, str(tok))}
    arrays = (frozenset(n for n, unused in pairs if ranks.get(n, 1) >= 1) | frozenset(seed)) - symbols
    table: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in seed.items()}
    for unused in range(8):
        changed = False
        for name, value in pairs:
            if name in seed:
                continue
            ext = extent_tokens(value, table, tuple_locals, arrays)
            if ext is not None and table.get(name) != ext:
                table[name] = ext
                changed = True
        if not changed:
            break
    per_name: dict[str, OrderedSet] = {}
    for name, value in pairs:
        ext = extent_tokens(value, table, tuple_locals, arrays)
        if ext is not None:
            per_name.setdefault(name, OrderedSet()).add(ext)
    for name, seen in per_name.items():
        if len(seen) > 1 and name not in seed:
            table.pop(name, None)
    return table


def rank_table(tree: ast.AST, seed: dict[str, int], call_returns: dict[str, int] | None = None) -> dict[str, int]:
    """Propagate ndim across straight-line assignments to a fixpoint. ``call_returns`` (``{helper:
    return_ndim}``) gives a local bound to a helper call that helper's return rank."""
    global active_tuple_lengths
    ranks = dict(seed)
    # The tree does not change while the table converges: index its bindings once, not every round.
    bindings, first_values = name_binding_index(tree)
    for unused in range(8):
        changed = False
        active_tuple_lengths = tuple_lengths(bindings, first_values, ranks, seed)
        for name, value in bindings:
            r = expr_rank(value, ranks)
            if r is None and call_returns is not None:
                r = call_return_rank(value, call_returns)
            if r is not None and ranks.get(name) != r:
                ranks[name] = r
                changed = True
        if not changed:
            break
    active_tuple_lengths = tuple_lengths(bindings, first_values, ranks, seed)
    drop_rank_conflicts(tree, ranks, seed)
    active_tuple_lengths = None
    return ranks


def drop_rank_conflicts(tree: ast.AST, ranks: dict[str, int], seed: dict[str, int]) -> None:
    """Forget any name whose assignments do not AGREE on a rank.

    The table is flow-insensitive: keeping the last writer would hand every consumer one rank at every
    program point. One rank per name or none; a caller that needs the rank AT a statement tracks it itself.
    """
    per_name: dict[str, OrderedSet] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            rank = expr_rank(node.value, ranks)
            if rank is not None:
                per_name.setdefault(node.targets[0].id, OrderedSet()).add(rank)
    for name, seen in per_name.items():
        if len(seen) <= 1:
            continue
        if name in seed:
            ranks[name] = seed[name]  # a declared array keeps its DECLARED rank, whatever a rebinding makes it
        else:
            ranks.pop(name, None)


def int_expr(value: ast.AST, ranks: dict[str, int], seed_ranks: dict[str, int] | None = None) -> int | None:
    """Evaluate a small integer expression used as a tuple length or repeat count: constants, names (array
    rank), ``arr.ndim``, unary minus and ``+ - * / //``.

    ``arr.ndim`` reads ``seed_ranks`` (the immutable seed) first, so a local cannot inflate its own rank
    through its own tuple-shape expression (``shape = (1, c) + (1,) * (x.ndim - 2)``).
    """
    if isinstance(value, ast.Constant) and isinstance(value.value, int):
        return value.value
    if isinstance(value, ast.Name):
        return ranks.get(value.id)
    if isinstance(value, ast.Attribute) and value.attr == "ndim" and isinstance(value.value, ast.Name):
        if seed_ranks is not None and value.value.id in seed_ranks:
            return seed_ranks[value.value.id]
        return ranks.get(value.value.id)
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
        v = int_expr(value.operand, ranks, seed_ranks)
        return None if v is None else -v
    if isinstance(value, ast.BinOp):
        lv = int_expr(value.left, ranks, seed_ranks)
        rv = int_expr(value.right, ranks, seed_ranks)
        if lv is None or rv is None:
            return None
        if isinstance(value.op, ast.Add):
            return lv + rv
        if isinstance(value.op, ast.Sub):
            return lv - rv
        if isinstance(value.op, ast.Mult):
            return lv * rv
        if isinstance(value.op, (ast.Div, ast.FloorDiv)):
            return None if rv == 0 else lv // rv
    return None


def tuple_expr_len(
    value: ast.AST,
    ranks: dict[str, int],
    assigns: dict[str, ast.expr],
    visited: set[str] | None = None,
    seed_ranks: dict[str, int] | None = None,
) -> int | None:
    """Length of a tuple-valued expression, if statically known.

    Covers tuple/list literals, ``arr.shape``, a name bound to one, concatenation ``A + B`` and repetition
    ``(1,) * n``.
    """
    if visited is None:
        visited = set()
    if isinstance(value, (ast.Tuple, ast.List)):
        return len(value.elts)
    if isinstance(value, ast.Attribute) and value.attr == "shape":
        return expr_rank(value.value, ranks)
    if isinstance(value, ast.Name) and value.id not in visited:
        visited.add(value.id)
        bound = assigns.get(value.id)
        if bound is None:
            return None
        return tuple_expr_len(bound, ranks, assigns, visited, seed_ranks)
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
        lv = tuple_expr_len(value.left, ranks, assigns, visited, seed_ranks)
        rv = tuple_expr_len(value.right, ranks, assigns, visited, seed_ranks)
        if lv is not None and rv is not None:
            return lv + rv
        return None
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult):
        # Tuple * int or int * tuple.
        lt = tuple_expr_len(value.left, ranks, assigns, visited, seed_ranks)
        rt = tuple_expr_len(value.right, ranks, assigns, visited, seed_ranks)
        li = int_expr(value.left, ranks, seed_ranks)
        ri = int_expr(value.right, ranks, seed_ranks)
        if lt is not None and ri is not None:
            return lt * ri
        if rt is not None and li is not None:
            return rt * li
        return None
    return None


def build_tuple_lengths(
    tree: ast.AST, ranks: dict[str, int], seed_ranks: dict[str, int] | None = None
) -> dict[str, int | None]:
    """Map each local bound to a tuple form to its length (``None`` when not yet known)."""
    bindings, first_values = name_binding_index(tree)
    return tuple_lengths(bindings, first_values, ranks, seed_ranks)


def name_binding_index(tree: ast.AST) -> tuple[list[tuple[str, ast.expr]], dict[str, ast.expr]]:
    """Every single-Name ``name = value`` under ``tree`` in ``ast.walk`` order, and each name's first one
    in a module, function or class body.

    One walk: breadth-first meets a body's statements after their owner, grouped in owner order.
    """
    bindings: list[tuple[str, ast.expr]] = []
    first: dict[str, ast.expr] = {}
    body_statements: OrderedSet[int] = OrderedSet()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module, ast.ClassDef)):
            body_statements.update(id(stmt) for stmt in node.body)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            bindings.append((node.targets[0].id, node.value))
            if id(node) in body_statements:
                first.setdefault(node.targets[0].id, node.value)
    return bindings, first


def tuple_lengths(
    bindings: list[tuple[str, ast.expr]],
    first_values: dict[str, ast.expr],
    ranks: dict[str, int],
    seed_ranks: dict[str, int] | None,
) -> dict[str, int | None]:
    """Each bound name whose value is a tuple form, with its length when statically known, else None."""
    lengths: dict[str, int | None] = {}
    for name, value in bindings:
        length = tuple_expr_len(value, ranks, first_values, seed_ranks=seed_ranks)
        if length is not None:
            lengths[name] = length
        elif tuple_valued(value, first_values):
            lengths.setdefault(name, None)
    return lengths


def tuple_valued(value: ast.AST, assigns: dict[str, ast.expr], visited: frozenset[str] = frozenset()) -> bool:
    """True iff ``value`` is one of the tuple forms :func:`tuple_expr_len` sizes, whether or not its
    length is known yet: ``shp = Y.shape`` is a tuple before ``Y`` has a rank, never one dimension.
    """
    if isinstance(value, (ast.Tuple, ast.List)):
        return True
    if isinstance(value, ast.Attribute):
        return value.attr == "shape"
    if isinstance(value, ast.Name):
        bound = assigns.get(value.id)
        return bound is not None and value.id not in visited and tuple_valued(bound, assigns, visited | {value.id})
    if isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Mult)):
        return tuple_valued(value.left, assigns, visited) or tuple_valued(value.right, assigns, visited)
    return False


def param_body_rank_evidence(fn: ast.FunctionDef) -> dict[str, int]:
    """Lower bounds on a helper's param ranks from how its BODY uses each param: ``p.shape[k]`` implies
    rank >= k+1, a tuple subscript ``p[:, a:b, c]`` rank >= its non-newaxis entry count. Unlike call-site
    ranks, these cannot be poisoned by a flow-insensitive table passing a later-reshaped local."""
    params = {a.arg for a in fn.args.args}
    ev: dict[str, int] = {}

    def bump(name: str, r: int) -> None:
        if name in params and r > ev.get(name, 0):
            ev[name] = r

    for node in ast.walk(fn):
        if isinstance(node, ast.Subscript):
            v = node.value
            if isinstance(v, ast.Attribute) and v.attr == "shape" and isinstance(v.value, ast.Name):
                k = const_int(node.slice)
                if k is not None and k >= 0:
                    bump(v.value.id, k + 1)  # p.shape[k] -> rank >= k+1
            elif isinstance(v, ast.Name) and isinstance(node.slice, ast.Tuple):
                bump(v.id, sum(0 if is_newaxis(e) else 1 for e in node.slice.elts))
    return ev


def return_rank(fn: ast.FunctionDef, ranks: dict[str, int], seed_ranks: dict[str, int] | None = None) -> int | None:
    """Rank of ``fn``'s returned value (the max over its ``return`` statements),
    given a rank table for its body -- so a caller can propagate it."""
    global active_tuple_lengths
    prev = active_tuple_lengths
    active_tuple_lengths = build_tuple_lengths(fn, ranks, seed_ranks=seed_ranks)
    try:
        rs = [expr_rank(n.value, ranks) for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
        rs = [r for r in rs if r is not None]
        return max(rs) if rs else None
    finally:
        active_tuple_lengths = prev


def infer_param_ranks(
    funcs: list[ast.FunctionDef], kernel_name: str, kir_seed: dict[str, int]
) -> dict[str, dict[str, int]]:
    """Per-function ``{param: ndim}`` seeds, iterated to a fixpoint so helper-to-helper calls resolve.

    The kernel's params come from ``kir_seed``; a helper's from its positional call sites and the body
    lower bounds of :func:`param_body_rank_evidence`, merged by MAX: a smaller conflicting value only
    comes from a flow-insensitive mix-up (a reshaped local passed on), never from the param's real rank."""
    by_name = {fn.name: fn for fn in funcs}
    params = {fn.name: [a.arg for a in fn.args.args] for fn in funcs}
    seeds: dict[str, dict[str, int]] = {fn.name: dict(param_body_rank_evidence(fn)) for fn in funcs}
    ret_rank: dict[str, int] = {}
    for unused in range(6):
        changed = False
        for fn in funcs:
            base = dict(seeds[fn.name])
            if fn.name == kernel_name:
                base.update(kir_seed)
            ranks = rank_table(fn, base, call_returns=ret_rank)
            rr = return_rank(fn, ranks, seed_ranks=base)
            if rr is not None and ret_rank.get(fn.name) != rr:
                ret_rank[fn.name] = rr
                changed = True
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in by_name:
                    callee = node.func.id
                    for i, arg in enumerate(node.args):
                        if i >= len(params[callee]):
                            break
                        r = expr_rank(arg, ranks)
                        if r is None:
                            r = call_return_rank(arg, ret_rank)
                        pname = params[callee][i]
                        if r is not None and r > seeds[callee].get(pname, -1):
                            seeds[callee][pname] = r
                            changed = True
        if not changed:
            break
    return seeds


def helper_return_ranks(
    funcs: list[ast.FunctionDef], param_ranks: dict[str, dict[str, int]], kernel_name: str, kernel_seed: dict[str, int]
) -> dict[str, int]:
    """``{function: rank of what it returns}`` over the call-site parameter ranks, two rounds so a helper
    returning another helper's result resolves; gives a name bound to a helper call a rank."""
    returns: dict[str, int] = {}
    for unused in range(2):
        for fn in funcs:
            seed = dict(param_ranks.get(fn.name, {}))
            if fn.name == kernel_name:
                seed.update(kernel_seed)
            rank = return_rank(fn, rank_table(fn, seed, call_returns=returns), seed_ranks=seed)
            if rank is not None:
                returns[fn.name] = rank
    return returns


def agreed_param_ranks(
    funcs: list[ast.FunctionDef],
    kernel_name: str,
    param_ranks: dict[str, dict[str, int]],
    kernel_seed: dict[str, int],
    returns: dict[str, int],
) -> dict[str, dict[str, int]]:
    """Parameter ranks that hold at EVERY call site the kernel reaches, plus the kernel's own.

    :func:`infer_param_ranks` merges call sites by MAX, which is right for sizing a table and wrong for
    deciding a branch: a helper called with a vector at one site and a matrix at another would have its
    ``x.ndim == 2`` test answered for both. A parameter only reaches here when every reachable site
    passes it one known rank, and never for a helper whose name escapes as a value."""
    by_name = {fn.name: fn for fn in funcs}
    reachable = reachable_functions(funcs, kernel_name)
    callee_names: set[int] = set()
    observed: dict[tuple[str, str], set[int | None]] = {}
    for fn in funcs:
        if fn.name not in reachable:
            continue
        seed = dict(param_ranks.get(fn.name, {}))
        if fn.name == kernel_name:
            seed.update(kernel_seed)
        ranks = rank_table(fn, seed, call_returns=returns)
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in by_name):
                continue
            callee_names.add(id(node.func))
            unreadable = bool(node.keywords) or any(isinstance(a, ast.Starred) for a in node.args)
            params = [a.arg for a in by_name[node.func.id].args.args]
            for i, pname in enumerate(params):
                rank = None if unreadable or i >= len(node.args) else expr_rank(node.args[i], ranks)
                observed.setdefault((node.func.id, pname), set()).add(rank)
    escaped = {n.id for n in ast.walk(ast.Module(body=list(funcs), type_ignores=[])) if isinstance(n, ast.Name)}
    escaped = {name for name in escaped if name in by_name} - {
        n.func.id
        for fn in funcs
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and id(n.func) in callee_names
    }
    agreed: dict[str, dict[str, int]] = {fn.name: {} for fn in funcs}
    if kernel_name in agreed:
        agreed[kernel_name] = dict(kernel_seed)
    for (callee, pname), ranks_seen in observed.items():
        rank = next(iter(ranks_seen)) if len(ranks_seen) == 1 else None
        if rank is not None and callee != kernel_name and callee not in escaped:
            agreed[callee][pname] = rank
    return agreed
