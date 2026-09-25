"""``arr.shape[k]`` reads resolved against the shape table."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.lib_nodes.dims import NP_ZEROS_ALIASES
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of_, extent_is_scalar
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import const_or_name


def negative_literal_offset(node: ast.AST) -> int | None:
    """``K`` for a negative integer literal ``-K`` -- a signed ``Constant`` or ``UnaryOp(USub, Constant)``, the
    form numpy source parses to -- else None. Numpy counts such an index from the end of its axis."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value < 0:
        return -node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return node.operand.value
    return None


def const_int_index(node: ast.AST) -> int | None:
    """Return the integer value of a constant subscript index (``arr[3]`` /
    ``arr[-1]``), or ``None`` for a non-constant one. Numpy spells a negative
    literal index as ``UnaryOp(USub, Constant)``, not a signed ``Constant``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -node.operand.value
    return None


def is_newaxis_result_axis(sub: ast.Subscript, k: int) -> bool:
    """True when result axis ``k`` of ``sub`` is a ``np.newaxis``, whatever the base's rank.

    ``X[:, None, :].shape[1]`` is 1 for every ``X``: a newaxis inserts a unit axis and consumes
    no source axis. That makes the rank of an ``np.expand_dims`` operand irrelevant, which is
    what lets the extent resolve before the operand's own shape is harvested.

    Restricted to an all-``:``/newaxis subscript and a non-negative ``k``, so result axis ``k`` IS
    element ``k``: a scalar index, a gather, an Ellipsis, or an unnamed trailing axis each shift
    that correspondence by an amount only the base's rank fixes.
    """
    if k < 0:
        return False
    elts = list(sub.slice.elts) if isinstance(sub.slice, ast.Tuple) else [sub.slice]
    if k >= len(elts) or not all(isinstance(e, ast.Slice) or is_newaxis(e) for e in elts):
        return False
    return is_newaxis(elts[k])


def is_newaxis(elt: ast.expr) -> bool:
    """``np.newaxis`` in a subscript, which parses as a ``None`` constant."""
    return isinstance(elt, ast.Constant) and elt.value is None


class ShapeMidExpressionRewriter(ast.NodeTransformer):
    """Replace ``arr.shape[k]`` (and bare ``arr.shape``) anywhere in
    the body with the matching shape symbol from the IR's shape table.

    Legacy HPCAgent-Bench kernels read array extents inline -- e.g.
    ``for i in range(A.shape[0]):`` or ``a = np.zeros(A.shape)`` --
    which the C / Fortran emitter cannot lower directly. Resolve them
    at the AST level to the names declared on the array's shape tuple
    (parsed from ``bench_info.init.shapes`` or recovered via
    ``shapes_from_initialize``).

    A ``.shape`` / ``.shape[k]`` read whose base is a rank-shifting
    SUBSCRIPT rather than a bare Name (``v[..., None].shape[-1]`` in an
    inlined ``X.reshape(-1, X.shape[-1])``) resolves via
    :func:`iter_extent_of_`, which is Ellipsis / newaxis-aware -- so a
    broadcast subscript's static extent folds the same way a declared
    array's does. The Name-base path is unchanged.
    """

    def __init__(self, arrays_shapes) -> None:
        self.arrays_shapes = arrays_shapes

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # Check the ``arr.shape[k]`` pattern BEFORE descending into
        # the children -- otherwise ``visit_Attribute`` would rewrite
        # the ``arr.shape`` inner node to a Tuple and the pattern
        # match below would miss.
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "shape"
            and isinstance(node.value.value, ast.Name)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            shape = self.arrays_shapes.get(node.value.value.id)
            if shape and 0 <= node.slice.value < len(shape):
                return const_or_name(shape[node.slice.value])
        # ``<array-expr>.shape[k]`` on a Subscript or CALL base (``v[..., None]``,
        # ``np.maximum(h, 0.0)``): resolve the base's static extent (newaxis / Ellipsis
        # aware) and pick axis ``k`` (negative indices allowed). Unresolvable -> left intact.
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "shape"
            and isinstance(node.value.value, (ast.Subscript, ast.Call))
        ):
            k = const_int_index(node.slice)
            if k is not None:
                ext = iter_extent_of_(node.value.value, self.arrays_shapes)
                if ext is not None and -len(ext) <= k < len(ext):
                    return copy.deepcopy(ext[k])
                # The base's own shape is not knowable everywhere this runs (the first pass
                # has only the DECLARED arrays), but a newaxis is 1 at every rank. Subscript
                # bases only -- a call has no index list to read a newaxis out of.
                if isinstance(node.value.value, ast.Subscript) and is_newaxis_result_axis(node.value.value, k):
                    return ast.Constant(value=1)
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        # ``len(arr)`` -> the array's FIRST-dim size symbol (numpy ``len`` is
        # ``shape[0]``). C / C++ have no array ``len`` and Fortran's ``len`` is
        # the CHARACTER-length intrinsic, so the literal call fails to compile;
        # the python backends (numba / pythran / jax) run the body verbatim and
        # keep the builtin, so they never reach this native-only rewriter.
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
        ):
            shape = self.arrays_shapes.get(node.args[0].id)
            if shape:
                return const_or_name(shape[0])
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.value, ast.Name):
            # Bare ``<array-expr>.shape`` on a Subscript or CALL base (``Y.shape`` where
            # ``Y`` inlined to ``psi_frag[f]`` or to ``np.maximum(__hcall1, 0.0)``, or
            # ``v[..., None].shape``) -> the tuple of static extents from
            # :func:`iter_extent_of_`, which sizes an elementwise call from its operands.
            # Downstream tuple-subscript / reshape folding then consumes the literal tuple.
            if node.attr == "shape" and isinstance(node.value, (ast.Subscript, ast.Call)):
                ext = iter_extent_of_(node.value, self.arrays_shapes)
                if ext is not None:
                    return ast.Tuple(elts=[copy.deepcopy(e) for e in ext], ctx=ast.Load())
            return node
        shape = self.arrays_shapes.get(node.value.id)
        if not shape:
            return node
        if node.attr == "shape":
            # ``const_or_name`` (not a bare ``Name(id=token)``) so a COMPOUND shape
            # token -- a slice extent like ``"__inl2_na - 1"`` (LS3DF's Lanczos
            # ``off = betas[:na - 1]``) -- re-parses into a real BinOp rather than a
            # malformed Name whose id is source text (which the int-context / logical
            # analyses then misclassify).
            return ast.Tuple(elts=[const_or_name(s) for s in shape], ctx=ast.Load())
        if node.attr == "size":
            # ``arr.size`` -> product of shape symbols (each token re-parsed).
            if len(shape) == 1:
                return const_or_name(shape[0])
            expr = const_or_name(shape[0])
            for s in shape[1:]:
                expr = ast.BinOp(left=expr, op=ast.Mult(), right=const_or_name(s))
            return expr
        if node.attr == "ndim":
            return ast.Constant(value=len(shape))
        # ``arr.dtype`` -- leave intact; downstream emit drops the dtype
        # kwarg via the builtin-cast / math rewriters as appropriate.
        return node


def fold_shape_reads_in_table(shapes: dict[str, object]) -> None:
    """Fold ``<expr>.shape[k]`` inside the shape TABLE's own tokens, exactly as
    :class:`ShapeMidExpressionRewriter` folds them in the body.

    Inlining substitutes a helper's argument EXPRESSION at every use, so an
    ``np.expand_dims(x, 1)`` argument (already rewritten to ``x[:, None, :]``) leaves the
    helper's output shape as ``('x[:, None, :].shape[0]', 'x[:, None, :].shape[1]', ...)``.
    The regex resolver only matches a Name base, so those tokens survive; the body's copies
    of them get folded but the table's do not, and the two then disagree. Anything reading
    the table for a CONSTANT sees source text: ``expand_squeeze`` asked to drop axis 1 finds
    ``'x[:, None, :].shape[1]'`` instead of ``'1'``, cannot prove the axis is a unit axis,
    and declines -- leaving ``np.squeeze`` for the emitter to reject.

    Never-worse: a token is replaced only when the fold resolves every ``.shape`` read in
    it, so a self-referential or unknown base keeps the original text for the downstream
    source-order resolvers.
    """
    rewriter = ShapeMidExpressionRewriter(shapes)
    for name in list(shapes):
        tokens = shapes[name]
        folded = []
        for tok in tokens:
            text = str(tok)
            if ".shape" in text:
                try:
                    new = ast.unparse(rewriter.visit(ast.parse(text, mode="eval").body))
                except SyntaxError:
                    new = text
                if ".shape" not in new:
                    text = new
            folded.append(text)
        shapes[name] = folded if isinstance(tokens, list) else tuple(folded)


def resolve_shape_token(node: ast.AST, shape_table: dict[str, tuple[str, ...]]) -> str:
    """Stringify a shape-tuple element, resolving ``arr.shape[i]``
    references against the known shape of ``arr``.

    * ``arr.shape[i]`` -> the ``i``-th token of ``shape_table[arr]``
      (constant ``i``) so a helper-inlined ``np.empty([x.shape[0],
      x.shape[1] // 2, ...])`` becomes a concrete shape tuple.
    * ``arr.shape[i] // K`` -> the resolved token wrapped in the
      ``//`` BinOp (still printable, still a valid C extent).
    * Anything else -> ``ast.unparse(node)`` (existing behaviour).
    """
    resolved = resolve_arr_shape_subscript(node, shape_table)
    if resolved is not None:
        return resolved
    if isinstance(node, ast.BinOp):
        left = resolve_shape_token(node.left, shape_table)
        right = resolve_shape_token(node.right, shape_table)
        op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%"}.get(
            type(node.op)
        )
        if op is not None:
            return f"({left} {op} {right})"
    return ast.unparse(node)


def resolve_arr_shape_subscript(node: ast.AST, shape_table: dict[str, tuple[str, ...]]) -> str | None:
    """Return the resolved shape token for ``arr.shape[i]``, or None
    if the form does not match or the source array is unknown."""
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "shape"
        and isinstance(node.value.value, ast.Name)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
    ):
        return None
    src = shape_table.get(node.value.value.id)
    if src is None or node.slice.value >= len(src):
        return None
    return src[node.slice.value]


class ResolveArrShape(ast.NodeTransformer):
    """Replace ``arr.shape[i]`` (where ``i`` is a constant int and
    ``arr`` is in the shape table) with the corresponding token.

    Walks the function body in source order so a reassigned local
    (``x = relu(...); x = maxpool2d(x); ...``) has the THEN-current
    shape used at each reference point. Cross-statement state is
    tracked via :attr:`current` which is forked at branches.

    The token may be a plain identifier (``N``), an integer literal
    (``5``), or a compound source expression (``H - K + 1``). Plain
    identifiers and ints parse back to the appropriate AST; compound
    forms are re-parsed via :func:`ast.parse` so the result is a real
    expression node, not an unparsable string.
    """

    def __init__(
        self,
        shapes: dict[str, list[str]],
        param_shapes: dict[str, tuple[str, ...]] | None = None,
        zeros_locals: dict[str, tuple[str, ...]] | None = None,
        reassign_shapes: dict[str, list[tuple[str, ...]]] | None = None,
    ) -> None:
        # ``shapes`` is the harvest's final-state table (used as a
        # fallback for purely-static lookups). ``current`` is the
        # WORKING table -- it is seeded ONLY with bench-info
        # parameter shapes (and any harvested locals that are never
        # reassigned) and gets updated as we walk statements in
        # source order. This way ``x.shape[i]`` at line K resolves
        # against the value of ``x`` AT line K, not the final value.
        self.shapes = shapes
        # ``zeros_locals`` carries the harvested shapes of every
        # ``Name = __hpcagent_bench_zeros__()`` marker so the resolver can
        # populate ``current`` when it hits one of those markers
        # without having to look at the np.zeros original call.
        self.zeros_locals = zeros_locals or {}
        # ``reassign_shapes`` is a per-name FIFO of shapes recorded
        # by ``WholeArrayAssignRewriter`` for every reassignment.
        # When we hit the Nth marker for a given name we pop the Nth
        # shape from this list (a name reassigned 3 times will have
        # 3 entries here, consumed in source order).
        self._reassign_shapes: dict[str, list[tuple[str, ...]]] = {
            k: list(v) for k, v in (reassign_shapes or {}).items()
        }
        if param_shapes is not None:
            self.current: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in param_shapes.items()}
        else:
            self.current = {k: tuple(v) for k, v in shapes.items()}

    def reresolve_token(self, tok: str) -> str:
        """Re-resolve any ``arr.shape[i]`` references inside ``tok``
        against the live ``self.current`` shape table. Tokens are
        re-parsed and substituted axis-wise; the returned string is
        always re-emittable (passes back through
        :func:`const_or_name` correctly)."""
        try:
            tree = ast.parse(str(tok), mode="eval").body
        except (SyntaxError, ValueError):
            return tok

        class Sub_(ast.NodeTransformer):
            def __init__(self_inner, current) -> None:
                self_inner.current = current

            def visit_Subscript(self_inner, node):
                self_inner.generic_visit(node)
                if not (
                    isinstance(node.value, ast.Attribute)
                    and node.value.attr == "shape"
                    and isinstance(node.value.value, ast.Name)
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, int)
                ):
                    return node
                src = self_inner.current.get(node.value.value.id)
                if not src or node.slice.value >= len(src):
                    return node
                return const_or_name(src[node.slice.value])

        tree = Sub_(self.current).visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = self.visit_stmt_list(node.body)
        # After walking the body, write the resolver's updated
        # zeros_locals back to the tree-attribute the emitter reads.
        if "zeros_locals" in vars(node):
            node.zeros_locals.update(self.zeros_locals)  # type: ignore[attr-defined]
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        node.iter = self.visit(node.iter)
        node.body = self.visit_stmt_list(node.body)
        node.orelse = self.visit_stmt_list(node.orelse)
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        node.test = self.visit(node.test)
        node.body = self.visit_stmt_list(node.body)
        node.orelse = self.visit_stmt_list(node.orelse)
        return node

    visit_While = visit_If

    def visit_stmt_list(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in stmts:
            new_stmt = self.visit(stmt)
            self.update_shape_for(stmt)
            if isinstance(new_stmt, list):
                out.extend(new_stmt)
            else:
                out.append(new_stmt)
        return out

    def update_shape_for(self, stmt: ast.stmt) -> None:
        """Update ``self.current`` to reflect the shape of an Assign
        target. Recognises:

        * ``Name = Name`` -- alias, inherit source shape.
        * ``Name = np.zeros((N, M), ...)`` / ``np.empty([...])`` /
          ``np.empty_like(other)`` -- shape from the constructor.
        * ``Name = BinOp/UnaryOp/IfExp`` -- shape from broadcast via
          :func:`iter_extent_of_`.
        * ``Name = Call(...)`` -- if the call is an elementwise math
          intrinsic, propagate the first array operand's shape.
        * Anything else -- leave the existing entry alone (or unset
          a non-broadcast result).
        """
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            return
        target = stmt.targets[0].id
        rhs = stmt.value
        # ``Name = __hpcagent_bench_zeros__()`` marker -- could be from the
        # ZerosRewriter (single shape per name in ``zeros_locals``)
        # OR from ``WholeArrayAssignRewriter`` (one marker per
        # reassignment, shape FIFO in ``_reassign_shapes``).
        if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and rhs.func.id == "__hpcagent_bench_zeros__":
            if target in self._reassign_shapes and self._reassign_shapes[target]:
                self.current[target] = self._reassign_shapes[target].pop(0)
                return
            if target in self.zeros_locals:
                # Re-resolve any ``arr.shape[i]`` references inside the
                # stored shape tokens against ``self.current`` so a
                # reassigned source array (lenet's ``x``) contributes
                # the correct THEN-current axis lengths. The
                # ``self.zeros_locals`` entry is also updated so the
                # emitter's decl block uses the fresh tokens.
                fresh = tuple(self.reresolve_token(t) for t in self.zeros_locals[target])
                self.current[target] = fresh
                self.zeros_locals[target] = fresh
                return
        if isinstance(rhs, ast.Name):
            src = self.current.get(rhs.id)
            if src is not None:
                self.current[target] = src
            return
        if (
            isinstance(rhs, ast.Call)
            and isinstance(rhs.func, ast.Attribute)
            and isinstance(rhs.func.value, ast.Name)
            and rhs.func.value.id == "np"
        ):
            attr = rhs.func.attr
            if attr in NP_ZEROS_ALIASES and rhs.args:
                if attr.endswith("_like") and isinstance(rhs.args[0], ast.Name):
                    src = self.current.get(rhs.args[0].id)
                    if src is not None:
                        self.current[target] = src
                    return
                shape_arg = rhs.args[0]
                if isinstance(shape_arg, (ast.Tuple, ast.List)):
                    self.current[target] = tuple(resolve_shape_token(e, self.current) for e in shape_arg.elts)
                    return
                if (
                    isinstance(shape_arg, ast.Attribute)
                    and shape_arg.attr == "shape"
                    and isinstance(shape_arg.value, ast.Name)
                ):
                    src = self.current.get(shape_arg.value.id)
                    if src is not None:
                        self.current[target] = tuple(src)
                    return
            if attr == "linspace" and len(rhs.args) >= 3:
                tok = ast.unparse(rhs.args[2])
                self.current[target] = (tok,)
                return
        # An all-size-1 result is a scalar, not a broadcast shape (see extent_is_scalar).
        if isinstance(rhs, (ast.BinOp, ast.UnaryOp, ast.IfExp)):
            ext = iter_extent_of_(rhs, self.current)
            if ext is not None and not extent_is_scalar(ext):
                self.current[target] = tuple(ast.unparse(e) for e in ext)
            return
        if isinstance(rhs, ast.Call):
            ext = iter_extent_of_(rhs, self.current)
            if ext is not None and not extent_is_scalar(ext):
                self.current[target] = tuple(ast.unparse(e) for e in ext)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "shape"
            and isinstance(node.value.value, ast.Name)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            return node
        src = self.current.get(node.value.value.id)
        if not src or node.slice.value >= len(src):
            return node
        tok = src[node.slice.value]
        # Plain literal int / identifier shortcut.
        try:
            return ast.Constant(value=int(tok))
        except (TypeError, ValueError):
            pass
        # Try parsing as a pure expression -- ``H - K + 1`` / ``N`` /
        # ``(N + 1)``. Strip any surrounding parens for cleanliness.
        try:
            parsed = ast.parse(str(tok), mode="eval").body
            return parsed
        except (SyntaxError, ValueError):
            return ast.Name(id=str(tok), ctx=ast.Load())
