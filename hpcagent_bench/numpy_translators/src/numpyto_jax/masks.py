"""Jit-mode rewrites of boolean masks and data-dependent slices into fixed-shape masked forms."""

import ast
from collections.abc import Callable

from numpyto_common.parallelism import is_timestep_loop
from numpyto_common.subscripts import is_full_slice

from numpyto_jax.errors import EmitError
from numpyto_jax.jnp import is_bool_expr
from numpyto_jax.loops import index_in_shape, is_static_iterable, loop_vars, range_args_static
from numpyto_jax.names import deep_copy, is_np_attr, names_loaded, names_stored


def boolean_mask_transform(fn: ast.FunctionDef) -> None:
    """Lower boolean-mask indexing (no static shape under jit) to ``where``:

    * ``A[m] = rhs``      -> ``A = np.where(m, rhs|A[m]->A, A)``
    * ``A[m] <op>= rhs``  -> ``A = np.where(m, A <op> rhs|.., A)``
    * ``A[m].mean()``     -> ``np.sum(np.where(m, A, 0)) / np.sum(m)``
    * ``A[m].sum()``      -> ``np.sum(np.where(m, A, 0))``

    ``m`` is a comparison/``np.logical_*`` result (or a name bound to one).
    Masked-out lanes are the identity, so this is exact -- powers
    mandelbrot1's escape update and nbody's ``inv_r3[inv_r3>0]**-1.5``."""
    bool_names = set()
    for s in ast.walk(fn):
        if isinstance(s, ast.Assign) and is_bool_expr(s.value):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    bool_names.add(t.id)

    def is_mask(idx: ast.expr) -> bool:
        return is_bool_expr(idx) or (isinstance(idx, ast.Name) and idx.id in bool_names)

    inline_masked_subsets(fn, is_mask)
    MaskToWhere(is_mask).visit(fn)
    ast.fix_missing_locations(fn)


def inline_masked_subsets(fn: ast.FunctionDef, is_mask: Callable[[ast.expr], bool]) -> None:
    """``v = data[mask]; ... v.mean()`` (azimint_naive) -> drop the definition and substitute
    ``data[mask]``, so the masked-reduction rewrite applies. Only single-assignment names qualify."""
    store_counts: dict = {}
    for s in ast.walk(fn):
        for t in s.targets if isinstance(s, ast.Assign) else []:
            if isinstance(t, ast.Name):
                store_counts[t.id] = store_counts.get(t.id, 0) + 1
    subset_map = {}
    for s in ast.walk(fn):
        if (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
            and isinstance(s.value, ast.Subscript)
            and is_mask(s.value.slice)
            and store_counts.get(s.targets[0].id) == 1
        ):
            subset_map[s.targets[0].id] = s.value
    if not subset_map:
        return

    class Substituter(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in subset_map:
                return None  # drop the now-inlined definition
            self.generic_visit(node)
            return node

        def visit_Name(self, n):
            if isinstance(n.ctx, ast.Load) and n.id in subset_map:
                return deep_copy(subset_map[n.id])
            return n

    Substituter().visit(fn)
    ast.fix_missing_locations(fn)


class MaskToWhere(ast.NodeTransformer):
    """Masked reductions and masked (augmented) stores -> ``np.where`` forms."""

    def __init__(self, is_mask: Callable[[ast.expr], bool]) -> None:
        self.is_mask = is_mask

    def widen(self, node: ast.expr) -> ast.expr:
        """``node`` with every masked subscript ``X[m]`` replaced by the whole ``X``."""
        is_mask = self.is_mask

        class Walker(ast.NodeTransformer):
            def visit_Subscript(self, n):
                self.generic_visit(n)
                return n.value if is_mask(n.slice) else n

        return Walker().visit(deep_copy(node))

    def visit_Call(self, node):
        self.generic_visit(node)
        # ``X[m].mean()`` / ``.sum()``  and ``np.sum(X[m])`` / ``np.mean``
        red = None
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in ("mean", "sum")
            and isinstance(node.func.value, ast.Subscript)
            and self.is_mask(node.func.value.slice)
        ):
            red, sub = node.func.attr, node.func.value
        elif (
            (is_np_attr(node.func, "sum") or is_np_attr(node.func, "mean"))
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Subscript)
            and self.is_mask(node.args[0].slice)
        ):
            red, sub = node.func.attr, node.args[0]
        if red is None:
            return node
        m, arr = sub.slice, sub.value
        masked = np_call("where", [deep_copy(m), arr, ast.Constant(value=0)])
        total = np_call("sum", [masked])
        if red == "sum":
            return ast.copy_location(total, node)
        return ast.copy_location(ast.BinOp(left=total, op=ast.Div(), right=np_call("sum", [deep_copy(m)])), node)

    def visit_Assign(self, node):
        self.generic_visit(node)
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and self.is_mask(node.targets[0].slice)
        ):
            tgt = node.targets[0]
            m, arr = tgt.slice, tgt.value
            new = np_call("where", [deep_copy(m), self.widen(node.value), deep_copy(arr)])
            return ast.copy_location(ast.Assign(targets=[deep_copy(arr)], value=new), node)
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.target, ast.Subscript) and self.is_mask(node.target.slice):
            tgt = node.target
            m, arr = tgt.slice, tgt.value
            rhs = ast.BinOp(left=deep_copy(arr), op=node.op, right=self.widen(node.value))
            new = np_call("where", [deep_copy(m), rhs, deep_copy(arr)])
            return ast.copy_location(ast.Assign(targets=[deep_copy(arr)], value=new), node)
        return node


def np_call(name: str, args: list[ast.AST]) -> ast.Call:
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=name, ctx=ast.Load()), args=args, keywords=[]
    )


def mask_reduction_slices(fn: ast.FunctionDef) -> None:
    """Rewrite a variable-width slice feeding a reduction into a masked
    full-width operand, so the reduction needs no dynamic shape.

    Triangular linalg kernels reduce over a prefix/suffix (``A[i, :j] @
    A[:j, j]``, ``np.dot(A[i, :k], A[j, :k])``). Masked-out entries are 0 --
    the identity for ``@``/``sum`` -- so ``X[.., :j, ..]`` -> ``np.where(
    np.arange(n) < j, X[.., :, ..], 0)`` is exact. Operands that aren't a
    clean one-sided dynamic slice are left alone (may be rejected later)."""
    lv = loop_vars(fn)

    class Rewriter(ast.NodeTransformer):
        def visit_BinOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, ast.MatMult):
                node.left = maybe_mask(node.left, lv)
                node.right = maybe_mask(node.right, lv)
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if is_np_attr(node.func, "dot") and len(node.args) == 2:
                node.args = [maybe_mask(a, lv) for a in node.args]
            return node

    Rewriter().visit(fn)
    ast.fix_missing_locations(fn)


def dyn_slice_info(node: ast.AST, lv: set[str]):
    """For ``Arr[.., dynamic-slice, ..]`` return ``(arr, axis, lower, upper)``
    when a bound depends on a loop var, else None. Covers one-sided (``:j``,
    ``i:``) and two-sided-but-one-dynamic (``i:M``) slices. ``arr`` must be a
    plain Name and the slice the only ``ast.Slice`` in the subscript."""
    if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)):
        return None
    sl = node.slice
    elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]

    def is_dyn(e):
        return (
            isinstance(e, ast.Slice)
            and e.step is None
            and (
                (e.lower is not None and bool(names_loaded(e.lower) & lv))
                or (e.upper is not None and bool(names_loaded(e.upper) & lv))
            )
        )

    dyn_positions = [k for k, e in enumerate(elts) if is_dyn(e)]
    if len(dyn_positions) != 1:
        return None
    # Other axes must be plain int indices or full ``:`` slices (the only ones
    # the 1-D axis mask broadcasts cleanly against).
    p = dyn_positions[0]
    for k, e in enumerate(elts):
        if k != p and isinstance(e, ast.Slice) and not is_full_slice(e):
            return None
    s = elts[p]
    return node.value, p, s.lower, s.upper


def axis_mask(arr: ast.AST, p: int, lower: ast.AST | None, upper: ast.AST | None) -> ast.AST:
    """``np.arange(arr.shape[p])`` constrained by the present bounds:
    ``(arange >= lower) & (arange < upper)``."""
    shape_p = ast.Subscript(
        value=ast.Attribute(value=arr, attr="shape", ctx=ast.Load()), slice=ast.Constant(value=p), ctx=ast.Load()
    )
    arange = ast.Call(
        func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="arange", ctx=ast.Load()),
        args=[shape_p],
        keywords=[],
    )
    terms = []
    if lower is not None:
        terms.append(ast.Compare(left=arange, ops=[ast.GtE()], comparators=[deep_copy(lower)]))
    if upper is not None:
        terms.append(ast.Compare(left=arange, ops=[ast.Lt()], comparators=[deep_copy(upper)]))
    mask = terms[0]
    for t in terms[1:]:
        mask = ast.BinOp(left=mask, op=ast.BitAnd(), right=t)
    return mask


def widen_to_full(node: ast.Subscript, p: int) -> ast.AST:
    """Replace the dynamic slice axis ``p`` of a subscript with full ``:``."""
    if isinstance(node.slice, ast.Tuple):
        new_elts = list(node.slice.elts)
        new_elts[p] = ast.Slice(lower=None, upper=None, step=None)
        return ast.Subscript(value=node.value, slice=ast.Tuple(elts=new_elts, ctx=ast.Load()), ctx=ast.Load())
    return node.value  # ``v[:k]`` -> whole vector ``v``


def maybe_mask(node: ast.AST, lv: set[str]) -> ast.AST:
    info = dyn_slice_info(node, lv)
    if info is None:
        return node
    arr, p, lower, upper = info
    full = widen_to_full(node, p)
    mask = axis_mask(arr, p, lower, upper)
    where = ast.Call(
        func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="where", ctx=ast.Load()),
        args=[mask, full, ast.Constant(value=0)],
        keywords=[],
    )
    return ast.copy_location(where, node)


def widen_dynamic_slices(node: ast.AST, lv: set[str]) -> ast.AST:
    """Drop one-sided dynamic-slice bounds to full ``:`` (no zeroing -- a write
    mask does the truncation). Used on the RHS of a masked dynamic write."""

    class Walker(ast.NodeTransformer):
        def visit_Subscript(self, n):
            self.generic_visit(n)
            info = dyn_slice_info(n, lv)
            if info is None:
                return n
            p = info[1]
            if isinstance(n.slice, ast.Tuple):
                elts = list(n.slice.elts)
                elts[p] = ast.Slice(lower=None, upper=None, step=None)
                return ast.copy_location(
                    ast.Subscript(value=n.value, slice=ast.Tuple(elts=elts, ctx=ast.Load()), ctx=ast.Load()), n
                )
            return n.value  # ``v[:k]`` -> ``v``

    return Walker().visit(node)


def mask_dynamic_writes(fn: ast.FunctionDef) -> None:
    """Rewrite a write to a variable-width prefix/suffix into a masked write
    over the full axis: ``C[i, :i+1] += rhs`` becomes ``C[i, :] = np.where(
    np.arange(n) < i+1, C[i, :] + rhs_widened, C[i, :])`` (functionalised to
    ``.at[i, :].set(..)`` downstream). Covers syrk/syr2k/symm rank-updates."""
    lv = loop_vars(fn)

    class Rewriter(ast.NodeTransformer):
        def rewrite(self, target, value):
            info = dyn_slice_info(target, lv)
            if info is None:
                return None
            arr, p, lower, upper = info
            full = widen_dynamic_slices(target, lv)
            mask = axis_mask(arr, p, lower, upper)
            keep = widen_dynamic_slices(target, lv)
            where = ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="where", ctx=ast.Load()),
                args=[mask, value, keep],
                keywords=[],
            )
            return ast.Assign(targets=[full], value=where)

        def visit_AugAssign(self, node):
            self.generic_visit(node)
            if dyn_slice_info(node.target, lv) is None:
                return node
            old = widen_dynamic_slices(deep_copy(node.target), lv)
            rhs = ast.BinOp(left=old, op=node.op, right=widen_dynamic_slices(node.value, lv))
            out = self.rewrite(node.target, rhs)
            return ast.copy_location(out, node) if out else node

        def visit_Assign(self, node):
            self.generic_visit(node)
            if len(node.targets) != 1 or dyn_slice_info(node.targets[0], lv) is None:
                return node
            out = self.rewrite(node.targets[0], widen_dynamic_slices(node.value, lv))
            return ast.copy_location(out, node) if out else node

    Rewriter().visit(fn)
    ast.fix_missing_locations(fn)


def rewrite_flip_prefix(fn: ast.FunctionDef) -> None:
    """``np.flip(arr[:k])`` (reverse of a dynamic prefix) -> the reversal
    gather ``arr[np.clip(k-1 - np.arange(n), 0, n-1)]``. Surrounding
    reduction/write masking then truncates to the first ``k`` lanes, so
    durbin's flip-prefix ops lower without a data-dependent shape."""
    lv = loop_vars(fn)

    class Rewriter(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if not (is_np_attr(node.func, "flip") and len(node.args) == 1):
                return node
            arg = node.args[0]
            if not (
                isinstance(arg, ast.Subscript)
                and isinstance(arg.value, ast.Name)
                and isinstance(arg.slice, ast.Slice)
                and arg.slice.lower is None
                and arg.slice.upper is not None
                and (names_loaded(arg.slice.upper) & lv)
            ):
                return node
            arr, k = arg.value, arg.slice.upper
            n = ast.Subscript(
                value=ast.Attribute(value=arr, attr="shape", ctx=ast.Load()),
                slice=ast.Constant(value=0),
                ctx=ast.Load(),
            )
            arange = np_call("arange", [n])
            km1 = ast.BinOp(left=deep_copy(k), op=ast.Sub(), right=ast.Constant(value=1))
            idx = ast.BinOp(left=km1, op=ast.Sub(), right=arange)
            hi = ast.BinOp(
                left=ast.Subscript(
                    value=ast.Attribute(value=arr, attr="shape", ctx=ast.Load()),
                    slice=ast.Constant(value=0),
                    ctx=ast.Load(),
                ),
                op=ast.Sub(),
                right=ast.Constant(value=1),
            )
            clipped = np_call("clip", [idx, ast.Constant(value=0), hi])
            return ast.copy_location(ast.Subscript(value=arr, slice=clipped, ctx=ast.Load()), node)

    Rewriter().visit(fn)
    ast.fix_missing_locations(fn)


def mask_slice_reads(fn: ast.FunctionDef) -> None:
    """Mask a dynamic-slice read bound to an intermediate: ``cols = A_col[
    A_row[i]:A_row[i+1]]`` -> ``cols = np.where(mask, A_col, 0)`` (0-filled).
    CSR SpMV's ``vals @ x[cols]`` then works: masked ``vals`` lanes are 0 (the
    ``@`` identity), so the gather at 0-filled ``cols`` contributes nothing."""
    lv = loop_vars(fn)

    class Rewriter(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Subscript)
                and dyn_slice_info(node.value, lv) is not None
            ):
                node.value = maybe_mask(node.value, lv)
            return node

    Rewriter().visit(fn)
    ast.fix_missing_locations(fn)


def dynamic_window_slices(fn: ast.FunctionDef) -> None:
    """Rewrite a fixed-width sliding window ``arr[.., i:i+K, ..]`` (K static,
    i a loop var) to ``lax.dynamic_slice_in_dim(arr, i, K, axis)`` -- the conv
    kernels' ``input[:, i:i+K, j:j+K, :, None]``. Multiple windowed axes nest;
    the residual int/full/newaxis indices apply afterwards."""
    lv = loop_vars(fn)

    def window(s):
        # ``lo:lo+W`` with lo dynamic and W static -> (lo, W); else None.
        if not (isinstance(s, ast.Slice) and s.lower is not None and s.upper is not None and s.step is None):
            return None
        if not (names_loaded(s.lower) & lv):
            return None
        u = s.upper
        if isinstance(u, ast.BinOp) and isinstance(u.op, ast.Add):
            if ast.unparse(u.left) == ast.unparse(s.lower) and not (names_loaded(u.right) & lv):
                return s.lower, u.right
            if ast.unparse(u.right) == ast.unparse(s.lower) and not (names_loaded(u.left) & lv):
                return s.lower, u.left
        return None

    class Rewriter(ast.NodeTransformer):
        def visit_Subscript(self, node):
            self.generic_visit(node)
            if not isinstance(node.value, ast.Name):
                return node
            elts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            wins = [(k, window(e)) for k, e in enumerate(elts)]
            wins = [(k, w) for k, w in wins if w is not None]
            if not wins:
                return node
            arr = node.value
            for k, (start, width) in wins:
                arr = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="lax", ctx=ast.Load()), attr="dynamic_slice_in_dim", ctx=ast.Load()
                    ),
                    args=[arr, start, width, ast.Constant(value=k)],
                    keywords=[],
                )
            resid = list(elts)
            for k, unused in wins:
                resid[k] = ast.Slice(lower=None, upper=None, step=None)
            new_slice = ast.Tuple(elts=resid, ctx=ast.Load()) if isinstance(node.slice, ast.Tuple) else resid[0]
            return ast.copy_location(ast.Subscript(value=arr, slice=new_slice, ctx=ast.Load()), node)

    Rewriter().visit(fn)
    ast.fix_missing_locations(fn)


def for_is_unrolled(node: ast.For) -> bool:
    """Will ``emit_for`` UNROLL this ``for`` (index/carry stay concrete),
    rather than lower to a rolled ``fori_loop``/``while_loop`` (carry becomes
    a tracer)? True for a static-iterable/literal-sequence loop, or a
    static-range loop whose index feeds a shape (non-time-stepping). Used by
    :func:`loop_index_tainted` to find which scalars become tracers post-loop."""
    rng = node.iter
    if not (isinstance(rng, ast.Call) and isinstance(rng.func, ast.Name) and rng.func.id == "range"):
        return is_static_iterable(rng) or isinstance(rng, (ast.Tuple, ast.List))
    i = node.target.id if isinstance(node.target, ast.Name) else "_i"
    return index_in_shape(node, i) and range_args_static(rng) and not is_timestep_loop(node)


def rolled_loop_writes(fn: ast.FunctionDef) -> set[str]:
    """Names written inside a ROLLED loop (a ``while``, or a non-unrolled
    ``for``) -- such a name is a loop carry, a tracer once lowered to
    ``fori_loop``/``while_loop``, so slicing an array by it after the loop
    (ls3df's ``alphas[:na]`` with ``na += 1``) is as data-dependent as the
    index itself."""
    out: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.While) or isinstance(n, ast.For) and not for_is_unrolled(n):
            out |= names_stored(ast.Module(body=n.body, type_ignores=[]))
    return out


def loop_index_tainted(fn: ast.FunctionDef) -> set[str]:
    """Loop-index vars, scalars carried through a rolled loop
    (:func:`rolled_loop_writes`), plus every scalar TRANSITIVELY derived by a
    plain ``name = <expr(tainted)>`` assign (dwt2d's ``s = n >> lvl``). A
    slice bound built from such a name is as data-dependent as the index."""
    tainted = set(loop_vars(fn)) | rolled_loop_writes(fn)
    changed = True
    while changed:
        changed = False
        for n in ast.walk(fn):
            if (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id not in tainted
                and (names_loaded(n.value) & tainted)
            ):
                tainted.add(n.targets[0].id)
                changed = True
    return tainted


def reject_dynamic_slices(fn: ast.FunctionDef) -> None:
    """Raise if a ``ast.Slice`` bound depends on a loop-index variable
    (cholesky's ``A[i, :j]``). Such variable-length slices have no static
    shape and can't be traced -- honest fallback beats broken output.
    Unrolled-loop indices are excluded (concrete -> static slices). The taint
    is transitive, so a derived scalar (dwt2d's ``s = n >> lvl`` then
    ``out[:s, :s]``) is caught too, rather than crashing at run time."""
    tainted = loop_index_tainted(fn)
    for n in ast.walk(fn):
        if isinstance(n, ast.Slice):
            for part in (n.lower, n.upper, n.step):
                if part is not None and (names_loaded(part) & tainted):
                    raise EmitError("data-dependent slice bound (needs masking/padding)")
