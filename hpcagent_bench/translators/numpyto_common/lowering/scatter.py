"""``np.<ufunc>.at`` unbuffered scatters lowered to explicit loops."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import broadcast_extents, iter_extent_of_
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import const_int
from hpcagent_bench.translators.numpyto_common.lib_nodes.scalarize import scalarize_at_iters
from hpcagent_bench.translators.numpyto_common.lowering.calls import const_or_name_token
from hpcagent_bench.translators.numpyto_common.subscripts import is_full_slice


class ScatterAtRewriter(ast.NodeTransformer):
    """Lower ``np.<op>.at(target, idx, vals)`` (unbuffered scatter) into an
    explicit indexed loop -- the only correct sequential form, since repeated
    indices must accumulate (plain ``target[idx] += vals`` would not).

        np.add.at(Lx, src, flux)      ->  for __k in range(E): Lx[src[__k]] += flux[__k]
        np.subtract.at(Lx, dst, flux) ->  for __k in range(E): Lx[dst[__k]] -= flux[__k]
        np.maximum.at(M, idx, v)      ->  for __k in range(E): M[idx[__k]] = max(M[idx[__k]], v[__k])

    Every binary ufunc exposes ``.at``; we cover the realistic scatter ops:
    arithmetic (add/subtract/multiply/divide -> compound assign) and
    maximum/minimum (no compound operator -> ``t[i] = max(t[i], v)``). ``idx``
    is either a bare index-array Name (its shape gives the trip count) or any
    array-valued EXPRESSION whose extent :func:`iter_extent_of_` can resolve
    (a ``.reshape(-1)`` flatten, an offset ``ikb - 1``, ...); ``vals`` is an
    array Name / expression (subscripted per element), its unary negation, or
    a scalar constant broadcast to every iteration (azimint's counting
    ``np.add.at(counts, bin_id, 1)``). Anything unresolvable is refused rather
    than mis-lowered. Used by edge_laplacian, vexx_k, azimint_naive.
    """

    #: arithmetic ufuncs -> the compound-assign operator (``t[i] op= v``).
    AUG = {"add": ast.Add, "subtract": ast.Sub, "multiply": ast.Mult, "divide": ast.Div, "true_divide": ast.Div}
    #: max/min ufuncs -> a builtin folded into ``t[i] = fn(t[i], v)``.
    FOLD = {"maximum": "max", "minimum": "min"}

    def __init__(
        self,
        shapes: dict[str, list[str]],
        bool_names: set[str] | None = None,
        wrapper_defs: dict[str, ast.expr] | None = None,
    ) -> None:
        self.shapes = shapes
        #: Names proven boolean (:func:`collect_bool_names`) -- a boolean array
        #: used as the index of a ``.at`` scatter is a MASK, not a gather; letting
        #: it fall through the generalised expression path would silently scatter
        #: through 0/1 truth values instead of refusing. Empty by default so the
        #: unit tests that build this rewriter directly (no bool-name harvest)
        #: keep their prior bare-Name-only behaviour.
        self.bool_names = bool_names or frozenset()
        #: name -> its ``.reshape(-1)`` / ``np.broadcast_to(...)`` RHS, for a local
        #: alias assigned once then read (possibly more than once) bare inside
        #: ``.at()`` -- icon_scatter's ``vals = np.broadcast_to(...)``. Looking
        #: through the alias lets :meth:`peel_flatten` reach the wrapped operand
        #: the same way it does when the call sits inline at the ``.at()`` site.
        self.wrapper_defs = wrapper_defs or {}
        self._n = 0

    @staticmethod
    def is_ufunc_at(func: ast.AST):
        # Attribute chain ``np.<op>.at`` -> Attribute(Attribute(Name('np'), op), 'at')
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "at"
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id in ("np", "numpy")
        ):
            return func.value.attr
        return None

    @staticmethod
    def unwrap_wrapper_call(expr: ast.expr) -> ast.expr | None:
        """The wrapped BASE operand if ``expr`` is a ``<base>.reshape(-1)`` flatten
        (method OR the ``np.reshape(base, -1)`` function form the ``reshape``
        normaliser rewrites method calls to earlier in the same LibNode-expand
        phase) or a ``np.broadcast_to(base, shape)`` call; else ``None``.

        The two "transparent" wrapper idioms :meth:`peel_flatten` strips.
        Exposed so a pre-pass can find candidate ``name = <wrapper>`` aliases
        before this rewriter runs (see :func:`lp_scatter_at`)."""
        if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and not expr.keywords):
            return None
        is_np = isinstance(expr.func.value, ast.Name) and expr.func.value.id in ("np", "numpy")
        if expr.func.attr == "reshape":
            if is_np and len(expr.args) == 2:
                base, shape_arg = expr.args
            elif not is_np and len(expr.args) == 1:
                base, shape_arg = expr.func.value, expr.args[0]
            else:
                return None
            elt = (
                shape_arg.elts[0]
                if isinstance(shape_arg, (ast.Tuple, ast.List)) and len(shape_arg.elts) == 1
                else shape_arg
            )
            return base if const_int(elt) == -1 else None
        if expr.func.attr == "broadcast_to" and is_np and len(expr.args) == 2:
            return expr.args[0]
        return None

    def peel_flatten(self, expr: ast.expr) -> ast.expr:
        """Strip a bare ``<base>.reshape(-1)`` flatten (method or function form)
        or a ``np.broadcast_to(<base>, shape)`` wrapper, returning ``base``
        (looking through one level of local-alias indirection via
        :attr:`wrapper_defs` first).

        ``scalarize_at_iters`` scalarises a Subscript/Name/BinOp structurally but
        has no notion of a ``reshape`` call, so a flattened index/value would
        reach it unindexed. Rather than reimplement flat-index unravelling, let
        the scatter loop iterate the array's OWN (pre-flatten) axes instead --
        exactly the multi-axis nest already used for lulesh's 2-D ``nodelist``
        index. ``np.<op>.at``'s accumulate/fold ops are commutative over repeated
        indices, so visiting (index, value) pairs in nested-axis order instead of
        flat order changes nothing about the result.

        ``np.broadcast_to(operand, shape)`` reads IDENTICALLY to ``operand`` once
        scalarised structurally: a size-1 (or omitted/newaxis) source axis already
        reads index 0 under ``scalarize_at_iters``'s standard broadcast rule, so
        the explicit target shape carries no information the scalariser needs
        (icon_scatter's ``vals = np.broadcast_to(val[:, :, :, None], (nproma,
        nlev, nblks, nnbr))``)."""
        if isinstance(expr, ast.Name) and expr.id in self.wrapper_defs:
            return self.peel_flatten(self.wrapper_defs[expr.id])
        base = self.unwrap_wrapper_call(expr)
        return expr if base is None else base

    def refuse_boolean_index(self, idx: ast.expr, op: str) -> None:
        for n in ast.walk(idx):
            if isinstance(n, ast.Name) and n.id in self.bool_names:
                raise NotImplementedError(
                    f"np.{op}.at index {ast.unparse(idx)!r} reads boolean {n.id!r} -- "
                    "a boolean array there is a MASK, not a gather"
                )

    def index_extent(self, idx: ast.expr, op: str) -> tuple[ast.expr, tuple]:
        """The peeled index expression and its per-axis extent (shape tokens)."""
        peeled = self.peel_flatten(idx)
        if isinstance(peeled, ast.Name):
            bound = self.shapes.get(peeled.id)
            if not bound:
                raise NotImplementedError(f"np.{op}.at: unknown extent for index '{peeled.id}'")
            return peeled, tuple(bound)
        ext = iter_extent_of_(peeled, self.shapes)
        if ext is None:
            raise NotImplementedError(
                f"np.{op}.at: cannot determine scatter extent for index expression {ast.unparse(idx)!r}"
            )
        return peeled, tuple(ast.unparse(e) for e in ext)

    def val_at(self, vals: ast.expr, iters: list[ast.expr]) -> ast.expr:
        if isinstance(vals, ast.UnaryOp) and isinstance(vals.op, ast.USub):
            return ast.UnaryOp(op=ast.USub(), operand=self.val_at(vals.operand, iters))
        if isinstance(vals, ast.Constant):
            # A scalar fill: every iteration adds/folds the SAME literal, not a
            # per-element gather (azimint's ``np.add.at(counts, bin_id, 1)``).
            return vals
        peeled = self.peel_flatten(vals)
        if iter_extent_of_(peeled, self.shapes) is not None:
            return scalarize_at_iters(peeled, iters, self.shapes)
        if isinstance(peeled, ast.Name):
            # Untracked-shape Name: the original bare-Name contract -- read
            # elementwise at the SAME iters the index uses (edge_laplacian's
            # ``flux``, whose shape this rewriter never needed to know).
            slot = iters[0] if len(iters) == 1 else ast.Tuple(elts=list(iters), ctx=ast.Load())
            return ast.Subscript(value=ast.Name(id=peeled.id, ctx=ast.Load()), slice=slot, ctx=ast.Load())
        raise NotImplementedError(
            "np.<op>.at value must be an array name, its negation, a scalar constant, or a resolvable array expression"
        )

    def validate_target(self, target: ast.expr, op: str) -> None:
        """A target is a bare Name, or a slice VIEW of one -- ``base[:, ii]``
        (vexx_k's ``deexx[:, ii]``), numpy's own scatter-through-a-view
        semantics, since a basic-indexing slice is a view onto the same
        buffer. The view's lead must be full slices and scalars with EXACTLY
        one full slice: that is the single axis the index array writes
        through (:meth:`write_through_target`); anything else (a
        partial/strided slice, a fancy index, more than one full-slice axis)
        is refused by naming the form rather than mis-lowered."""
        if isinstance(target, ast.Name):
            return
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            lead = list(target.slice.elts) if isinstance(target.slice, ast.Tuple) else [target.slice]
            if sum(1 for e in lead if is_full_slice(e)) == 1 and all(
                is_full_slice(e) or not isinstance(e, ast.Slice) for e in lead
            ):
                return
        raise NotImplementedError(
            f"np.{op}.at needs a Name target or a slice view of one with exactly one "
            f"full-slice axis, not {ast.unparse(target)!r}"
        )

    def write_through_target(self, target: ast.expr, idx_expr: ast.expr, ctx: ast.expr_context) -> ast.Subscript:
        """``target``'s element-write Subscript with ``idx_expr`` substituted at
        its (single, validated) full-slice axis; every other lead component
        (a scalar like ``ii``) passes through unchanged. For a bare-Name
        target this is just ``target[idx_expr]``."""
        if isinstance(target, ast.Name):
            return ast.Subscript(value=ast.Name(id=target.id, ctx=ast.Load()), slice=copy.deepcopy(idx_expr), ctx=ctx)
        lead = list(target.slice.elts) if isinstance(target.slice, ast.Tuple) else [target.slice]
        new_lead = [copy.deepcopy(idx_expr) if is_full_slice(e) else copy.deepcopy(e) for e in lead]
        slot = new_lead[0] if len(new_lead) == 1 else ast.Tuple(elts=new_lead, ctx=ast.Load())
        return ast.Subscript(value=ast.Name(id=target.value.id, ctx=ast.Load()), slice=slot, ctx=ctx)

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        call = node.value
        if not isinstance(call, ast.Call):
            return node
        op = self.is_ufunc_at(call.func)
        if op is None:
            return node
        if (op not in self.AUG and op not in self.FOLD) or len(call.args) != 3:
            raise NotImplementedError(f"unsupported np.{op}.at form")
        target, idx, vals = call.args
        # MULTI-index scatter -- the unstructured / semi-structured ICON form
        # ``np.add.at(out, (idx2d - 1, jk, blk2d - 1), val[:, jk, :])``: the
        # index is a TUPLE of mixed indirect-array / scalar axes. Lower to an
        # accumulation loop nest over the (broadcast) value plane. Restricted
        # to a Name target -- no kernel in this corpus scatters a multi-index
        # tuple through a slice VIEW, so that combination stays refused.
        if isinstance(idx, ast.Tuple):
            if not isinstance(target, ast.Name):
                raise NotImplementedError(f"np.{op}.at needs a Name target for a multi-index scatter")
            return self.multi_index_scatter(node, op, target, idx, vals)
        self.validate_target(target, op)
        self.refuse_boolean_index(idx, op)
        idx_peeled, bound = self.index_extent(idx, op)
        self._n += 1
        # Iterate EVERY axis of the (peeled) index array (lulesh's nodelist is 2-D
        # ``(numelem, 8)``), so the scatter is a scalar ``target[idx[k0,k1]] op=
        # vals[k0,k1]`` -- not a leading-axis-only loop that leaves the trailing
        # axes as unlowered slices. ``vals`` is indexed with the same iters
        # (it broadcasts to the index shape for a 1-D target).
        # 1-D index keeps the flat ``__sat{n}`` name (the common edge_laplacian
        # case); a multi-D index (lulesh nodelist, or a flattened ``.reshape(-1)``
        # peeled back to its 2-D base) suffixes one iter per axis.
        iters = [f"__sat{self._n}"] if len(bound) == 1 else [f"__sat{self._n}_{d}" for d in range(len(bound))]
        iter_nodes = [ast.Name(id=i, ctx=ast.Load()) for i in iters]
        # A scatter whose target rows are BLOCKS rather than scalars: the index picks ONE leading
        # axis and every remaining axis belongs to the target and the value alike. Iterated over the
        # index alone, the body is a whole-block assignment that the emitters scalarise from the
        # TARGET's shape, which leaves the value operand as a bare pointer with no subscript --
        # cp2k_density_matrix_trs4's (nnz * fanout, bs, bs) contribution buffer. Extending the nest
        # over the trailing axes keeps both sides at the same rank.
        trail: tuple[str, ...] = ()
        if isinstance(target, ast.Name):
            tshape = tuple(self.shapes.get(target.id) or ())
            val_ext = iter_extent_of_(self.peel_flatten(vals), self.shapes)
            if len(tshape) > 1 and val_ext is not None and len(val_ext) == len(bound) + len(tshape) - 1:
                trail = tuple(str(t) for t in tshape[1:])
        trail_iters = [f"__sat{self._n}_t{d}" for d in range(len(trail))]
        trail_nodes = [ast.Name(id=i, ctx=ast.Load()) for i in trail_iters]
        idx_k = scalarize_at_iters(idx_peeled, iter_nodes, self.shapes)
        val_k = self.val_at(vals, iter_nodes + trail_nodes)

        def cell(ctx: ast.expr_context) -> ast.expr:
            base = self.write_through_target(target, idx_k, ctx)
            if not trail_nodes:
                return base
            lead = list(base.slice.elts) if isinstance(base.slice, ast.Tuple) else [base.slice]
            elts = [copy.deepcopy(e) for e in lead] + [copy.deepcopy(t) for t in trail_nodes]
            return ast.Subscript(
                value=ast.Name(id=base.value.id, ctx=ast.Load()), slice=ast.Tuple(elts=elts, ctx=ast.Load()), ctx=ctx
            )

        if op in self.AUG:
            stmt: ast.stmt = ast.AugAssign(target=cell(ast.Store()), op=self.AUG[op](), value=val_k)
        else:  # maximum / minimum -> t[i] = fn(t[i], v)
            stmt = ast.Assign(
                targets=[cell(ast.Store())],
                value=ast.Call(
                    func=ast.Name(id=self.FOLD[op], ctx=ast.Load()), args=[cell(ast.Load()), val_k], keywords=[]
                ),
            )
        body: list[ast.stmt] = [stmt]
        for it, ext in zip(reversed(iters + trail_iters), reversed(tuple(bound) + trail)):  # nest deepest-last
            body = [
                ast.For(
                    target=ast.Name(id=it, ctx=ast.Store()),
                    iter=ast.Call(
                        func=ast.Name(id="range", ctx=ast.Load()), args=[const_or_name_token(ext)], keywords=[]
                    ),
                    body=body,
                    orelse=[],
                )
            ]
        return ast.copy_location(body[0], node)

    def multi_index_scatter(self, node, op, target: ast.Name, idx_tuple: ast.Tuple, vals: ast.expr) -> ast.AST:
        """Lower a TUPLE-index ``np.<op>.at(out, (i0, i1, ...), vals)`` scatter.

        Each tuple component is an INDIRECT axis (a 2-D index array slice such
        as ``nbr_idx[:, :, n] - 1``) or a STRUCTURED axis (a scalar loop var /
        constant). The value ``vals`` (e.g. ``val[:, jk, :]``) defines the
        broadcast plane; we loop over that plane and, at each point, scalarize
        every index component and the value via :func:`scalarize_at_iters`
        (Slice axes consume an iter; scalar axes pass through), then accumulate
        ``out[idx0, idx1, ...] op= val`` -- the only sequentially-correct form
        when distinct neighbours hit the same target (duplicate-index sum)."""
        # The iteration plane is the numpy BROADCAST of the value and every
        # array-valued index component (icon_scatter's ``lev``/``idx``/``blk``
        # each carry only PART of the plane -- ``lev`` alone is missing the
        # nproma/nblks/nnbr axes an index component supplies, and vice versa --
        # so folding every resolvable extent together, not just the first one
        # that resolves, is required to recover the full (nproma, nlev, nblks,
        # nnbr) plane).
        self.refuse_boolean_index(vals, op)
        for comp in idx_tuple.elts:
            self.refuse_boolean_index(comp, op)
        # Peel a ``.reshape(-1)`` flatten / ``np.broadcast_to`` wrapper (directly,
        # or through one local-alias indirection) off every component up front,
        # same as the single-index path -- ``scalarize_at_iters`` cannot
        # structurally decompose either wrapper call.
        vals_p = self.peel_flatten(vals)
        idx_p = [self.peel_flatten(c) for c in idx_tuple.elts]
        ext = None
        for comp in (vals_p, *idx_p):
            comp_ext = iter_extent_of_(comp, self.shapes)
            if comp_ext is None:
                continue
            ext = comp_ext if ext is None else broadcast_extents(ext, comp_ext)
        if ext is None:
            raise NotImplementedError("multi-index np.<op>.at: cannot determine scatter extent")
        self._n += 1
        iters = [f"__sat{self._n}_{d}" for d in range(len(ext))]
        iter_nodes = [ast.Name(id=i, ctx=ast.Load()) for i in iters]
        idx_scalars = [scalarize_at_iters(c, iter_nodes, self.shapes) for c in idx_p]
        val_s = scalarize_at_iters(vals_p, iter_nodes, self.shapes)
        slot = ast.Tuple(elts=idx_scalars, ctx=ast.Load())
        lhs = ast.Subscript(value=ast.Name(id=target.id, ctx=ast.Load()), slice=slot, ctx=ast.Store())
        if op in self.AUG:
            stmt: ast.stmt = ast.AugAssign(target=lhs, op=self.AUG[op](), value=val_s)
        else:  # maximum / minimum -> t[i] = fn(t[i], v)
            cur = ast.Subscript(
                value=ast.Name(id=target.id, ctx=ast.Load()),
                slice=ast.Tuple(elts=list(idx_scalars), ctx=ast.Load()),
                ctx=ast.Load(),
            )
            stmt = ast.Assign(
                targets=[lhs],
                value=ast.Call(func=ast.Name(id=self.FOLD[op], ctx=ast.Load()), args=[cur, val_s], keywords=[]),
            )
        body: list[ast.stmt] = [stmt]
        for d in reversed(range(len(ext))):
            # ``iter_extent_of_`` already returns each extent as an AST node
            # (a Name like ``nproma`` or a computed length), so it is the
            # loop's ``range`` bound directly.
            body = [
                ast.For(
                    target=ast.Name(id=iters[d], ctx=ast.Store()),
                    iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[copy.deepcopy(ext[d])], keywords=[]),
                    body=body,
                    orelse=[],
                )
            ]
        return ast.copy_location(body[0], node)
