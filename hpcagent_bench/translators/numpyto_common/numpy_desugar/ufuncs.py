"""ufunc forms: ``outer``, ``out=``, elemental binary ufuncs, complex accessors, numba call fixups."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet
from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import RankedRewritePass, RewritePass, np_attr
from hpcagent_bench.translators.numpyto_common.numpy_desugar.hoist import HoistForm, ValueHoist
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import expr_rank


class CallFixups(RankedRewritePass):
    """Small call-form fixups for the narrower numba/pythran numpy surface, among them
    ``np.ndarray(shape, dtype=D)`` -> ``np.empty(shape, D)`` (numba has no
    ``np.ndarray`` constructor); ``np.linspace(a, b, n, dtype=D)`` ->
    ``np.linspace(a, b, n).astype(D)`` (numba's linspace takes no dtype kwarg);
    builtin ``abs(<array>)`` -> ``np.abs(<array>)`` (numba's builtin ``abs``
    types scalars only)."""

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "abs"
            and len(node.args) == 1
            and not node.keywords
            and (expr_rank(node.args[0], self.ranks) or 0) >= 1
        ):
            self.changed = True
            npabs = ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="abs", ctx=ast.Load())
            return ast.copy_location(ast.Call(func=npabs, args=node.args, keywords=[]), node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "issparse":
            # ``scipy.sparse.issparse(x)`` -> ``False``: the kernel ABI only passes dense numpy
            # arrays, and numba/pythran cannot type scipy.sparse, so the sparse branch must go.
            self.changed = True
            return ast.copy_location(ast.Constant(value=False), node)
        attr = np_attr(node)
        # numba/pythran want a tuple shape, not a list literal. ``array`` is not in this set: its
        # list is data, not a shape.
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
                ast.Call(func=empty, args=[node.args[0]] + ([dt] if dt is not None else []), keywords=[]), node
            )
        if attr == "linspace":
            kw = {k.arg: k.value for k in node.keywords}
            if "dtype" in kw:
                self.changed = True
                base = ast.Call(func=node.func, args=node.args, keywords=[k for k in node.keywords if k.arg != "dtype"])
                cast = ast.Attribute(value=base, attr="astype", ctx=ast.Load())
                return ast.copy_location(ast.Call(func=cast, args=[kw["dtype"]], keywords=[]), node)
        if attr == "flip" and node.args:
            # np.flip(x[, axis]) -> a reverse-step slice (pythran's np.flip fails
            # type deduction); no axis reverses every axis.
            x = node.args[0]
            kw = {k.arg: k.value for k in node.keywords}
            ax = kw.get("axis") or (node.args[1] if len(node.args) > 1 else None)
            rank = expr_rank(x, self.ranks)
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
        if attr == "swapaxes" and len(node.args) == 3 and not node.keywords:
            # np.swapaxes -> np.transpose: no backend implements swapaxes, dace makes it a callback.
            rank = expr_rank(node.args[0], self.ranks)
            axes = [a.value for a in node.args[1:] if isinstance(a, ast.Constant) and isinstance(a.value, int)]
            if rank is None or len(axes) != 2:
                return node
            perm = list(range(rank))
            i, j = axes[0] % rank, axes[1] % rank
            perm[i], perm[j] = perm[j], perm[i]
            self.changed = True
            order = ", ".join(str(p) for p in perm)
            return ast.copy_location(
                ast.parse(f"np.transpose({ast.unparse(node.args[0])}, ({order}))", mode="eval").body, node
            )
        return node


OUTER_OPS = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "true_divide": "/"}


def ufunc_method_op(node: ast.AST, method: str) -> str | None:
    """``np.<op>.<method>(...)`` (``np.add.outer`` / ``np.subtract.at``) -> ``<op>``, else None."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id in ("np", "numpy")
    ):
        return node.func.value.attr
    return None


def hoist_ufunc_outer(node: ast.AST, hoist: ValueHoist) -> ast.expr | None:
    """``np.add.outer(a, b)`` (1-D operands) -> a reshape+broadcast temp ``a[:, None] op b[None, :]``
    (numba has no ufunc.outer). Both operands are copied into hoisted temps first."""
    if not isinstance(node, ast.Call):
        return None
    op = ufunc_method_op(node, "outer")
    if op is None or op not in OUTER_OPS or len(node.args) != 2:
        return None
    a, b = node.args
    ranks = hoist.tables.ranks
    if expr_rank(a, ranks) != 1 or expr_rank(b, ranks) != 1:
        return None  # only the 1-D x 1-D outer grid
    p = f"__ao{hoist.ctr}"
    hoist.ctr += 1
    na, nb, sym = f"{p}_a", f"{p}_b", OUTER_OPS[op]
    # ``.copy()``: a strided slice (a column) is non-contiguous, and numba's reshape needs a contiguous array.
    hoist.queue(
        [
            f"{na} = ({ast.unparse(a)}).copy()",
            f"{nb} = ({ast.unparse(b)}).copy()",
            f"{p} = {na}.reshape({na}.shape[0], 1) {sym} {nb}.reshape(1, {nb}.shape[0])",
        ]
    )
    return ast.Name(id=p, ctx=ast.Load())


UFUNC_OUTER_HOIST = HoistForm(frozenset({"outer"}), (), hoist_ufunc_outer)


#: binary arithmetic ufuncs whose ``out=`` form maps to a plain BinOp.
UFUNC_OUT_OPS = {
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


#: binary ufuncs with no Python operator whose ``out=`` form keeps the call and only drops the
#: keyword -- the plain call already lowers through the generic elementwise-Call path.
UFUNC_OUT_CALLS = OrderedSet(("maximum", "minimum", "fmax", "fmin"))


class FillDiagonalInline(ast.NodeTransformer):
    """``np.fill_diagonal(A, v)`` -> ``for __fd in range(min(A.shape[0], A.shape[1])): A[__fd, __fd] = v``.

    numpy's rule for a 2-D target, written out. A higher-rank target keeps the call and is refused
    downstream: numpy then fills ``A[i, i, ..., i]`` and requires every axis to be equal, and
    ``wrap=True`` fills a different set of cells again, so neither may be assumed here.
    """

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        call = node.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id in ("np", "numpy")
            and call.func.attr == "fill_diagonal"
        ):
            return node
        if len(call.args) != 2 or call.keywords or not isinstance(call.args[0], ast.Name):
            return node
        arr, val = call.args[0], call.args[1]
        it = "__fd0"

        def axis(k: int) -> ast.expr:
            return ast.Subscript(
                value=ast.Attribute(value=ast.Name(id=arr.id, ctx=ast.Load()), attr="shape", ctx=ast.Load()),
                slice=ast.Constant(value=k),
                ctx=ast.Load(),
            )

        bound = ast.Call(func=ast.Name(id="min", ctx=ast.Load()), args=[axis(0), axis(1)], keywords=[])
        store = ast.Assign(
            targets=[
                ast.Subscript(
                    value=ast.Name(id=arr.id, ctx=ast.Load()),
                    slice=ast.Tuple(
                        elts=[ast.Name(id=it, ctx=ast.Load()), ast.Name(id=it, ctx=ast.Load())], ctx=ast.Load()
                    ),
                    ctx=ast.Store(),
                )
            ],
            value=val,
        )
        loop = ast.For(
            target=ast.Name(id=it, ctx=ast.Store()),
            iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=[bound], keywords=[]),
            body=[store],
            orelse=[],
        )
        return ast.fix_missing_locations(ast.copy_location(loop, node))


class UfuncOutInline(ast.NodeTransformer):
    """``np.multiply(a, b, out=c)`` (the binary arithmetic ufuncs) -> the explicit assignment
    ``c = a <op> b``; ``np.maximum(a, b, out=c)`` (and the other ``UFUNC_OUT_CALLS`` members, which
    have no BinOp form) -> ``c = np.maximum(a, b)``. The C/Fortran backends have no ufunc dispatch,
    so the ``out=`` form must become a store. ``c`` may be a slice; the target is that slice."""

    def rewrite_(self, call: ast.AST):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and len(call.args) == 2):
            return None
        # ``np.<ufunc>.outer(a, b, out=c)``: an outer product has no BinOp spelling, so the call is
        # kept and only the ``out=`` becomes a target.
        outer_form = (
            isinstance(call.func.value, ast.Attribute)
            and isinstance(call.func.value.value, ast.Name)
            and call.func.value.value.id in ("np", "numpy")
            and call.func.attr == "outer"
        )
        if outer_form:
            op = None
        else:
            if not (isinstance(call.func.value, ast.Name) and call.func.value.id in ("np", "numpy")):
                return None
            attr = call.func.attr
            op = UFUNC_OUT_OPS.get(attr)
            if op is None and attr not in UFUNC_OUT_CALLS:
                return None
        out = next((kw.value for kw in call.keywords if kw.arg == "out"), None)
        if out is None:
            return None
        target = copy.deepcopy(out)
        for n in ast.walk(target):
            if isinstance(n, (ast.Name, ast.Subscript, ast.Attribute)):
                n.ctx = ast.Store()
        if op is not None:
            value: ast.expr = ast.BinOp(left=call.args[0], op=op(), right=call.args[1])
        else:
            value = ast.Call(func=copy.deepcopy(call.func), args=[call.args[0], call.args[1]], keywords=[])
        return ast.copy_location(ast.Assign(targets=[target], value=value), call)

    def visit_Expr(self, node: ast.Expr):
        rw = self.rewrite_(node.value)
        return rw if rw is not None else node


class ComplexAccessorToFunc(ast.NodeTransformer):
    """``z.real`` -> ``np.real(z)``, ``z.imag`` -> ``np.imag(z)``, ``z.conjugate()``/``z.conj()`` ->
    ``np.conj(z)``: one canonical spelling means one native emit handler per op.

    ``conjugate_only`` restricts the rewrite to ``.conjugate()``/``.conj()``, for Python backends that
    run ``.real``/``.imag`` verbatim but whose pythran path lacks the ``.conjugate()`` method."""

    def __init__(self, conjugate_only: bool = False) -> None:
        self.changed = False
        self.conjugate_only = conjugate_only

    def np_call(self, fn: str, arg: ast.expr) -> ast.expr:
        self.changed = True
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=fn, ctx=ast.Load()),
                args=[arg],
                keywords=[],
            ),
            arg,
        )

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Handled at the Call so ``.conjugate`` is not first mistaken for an accessor below;
        # ``np.conj(x)`` (a call with args) is left as-is.
        if (
            isinstance(node.func, ast.Attribute)
            and not node.args
            and not node.keywords
            and node.func.attr in ("conjugate", "conj")
        ):
            return self.np_call("conj", self.visit(node.func.value))
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        # Not the ``np.real`` / ``np.imag`` module attribute: that is the function itself.
        if (
            not self.conjugate_only
            and isinstance(node.ctx, ast.Load)
            and node.attr in ("real", "imag")
            and not (isinstance(node.value, ast.Name) and node.value.id in ("np", "numpy"))
        ):
            return self.np_call(node.attr, node.value)
        return node


def np_multi_call(fn: str, args: list[ast.expr]) -> ast.Call:
    """Build ``np.<fn>(*args)``."""
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=fn, ctx=ast.Load()), args=args, keywords=[]
    )


def cmp_zero(x: ast.expr, op: ast.cmpop) -> ast.Compare:
    """Build ``x <op> 0``."""
    return ast.Compare(left=x, ops=[op], comparators=[ast.Constant(value=0)])


class ElementalUfuncToPrimitive(RewritePass):
    """Rewrite two-argument elemental numpy ufuncs with no direct native/JIT
    lowering (numba has no ``np.heaviside``, pythran no ``np.logaddexp``) into
    already-supported primitives, which every backend lowers through the normal
    elementwise expander:

      * ``np.mod(a, b)``/``np.remainder(a, b)`` -> ``a % b`` -- numpy's floored
        modulo is exactly the ``%`` operator (sign of the divisor).
      * ``np.logaddexp(a, b)`` -> ``np.maximum(a, b) + np.log(1.0 + np.exp(-np.abs(a - b)))``
        -- ``log1p`` has no Fortran intrinsic; ``exp(-|a-b|)`` is in ``(0, 1]``
        so ``log(1 + .)`` is well-conditioned, agreeing with numpy to a few ulp.
      * ``np.heaviside(a, b)`` -> ``np.where(a < 0, 0.0, np.where(a == 0, b, 1.0))``.

    Reused operands are deep-copied so no AST node is shared between two positions."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        f = node.func
        if not (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id in ("np", "numpy")
            and len(node.args) == 2
            and not node.keywords
        ):
            return node
        a, b = node.args
        if f.attr in ("mod", "remainder"):
            self.changed = True
            return ast.copy_location(ast.BinOp(left=a, op=ast.Mod(), right=b), node)
        if f.attr == "logaddexp":
            self.changed = True
            diff = ast.BinOp(left=a, op=ast.Sub(), right=copy.deepcopy(b))
            expterm = np_multi_call("exp", [ast.UnaryOp(op=ast.USub(), operand=np_multi_call("abs", [diff]))])
            onep = ast.BinOp(left=ast.Constant(value=1.0), op=ast.Add(), right=expterm)
            tail = np_multi_call("log", [onep])
            head = np_multi_call("maximum", [copy.deepcopy(a), copy.deepcopy(b)])
            return ast.copy_location(ast.BinOp(left=head, op=ast.Add(), right=tail), node)
        if f.attr == "heaviside":
            self.changed = True
            inner = np_multi_call("where", [cmp_zero(copy.deepcopy(a), ast.Eq()), b, ast.Constant(value=1.0)])
            outer = np_multi_call("where", [cmp_zero(a, ast.Lt()), ast.Constant(value=0.0), inner])
            return ast.copy_location(outer, node)
        return node
