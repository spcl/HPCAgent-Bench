"""``np.*`` / ``math.*`` -> ``jnp.*`` rewrite of expressions."""

import ast

from hpcagent_bench.translators.numpyto_jax.names import is_identity_test, is_np_attr, names_loaded
from hpcagent_bench.translators.numpyto_jax.state import STATE
from hpcagent_bench.translators.numpyto_jax.vocab import BOOL_FUNCS

# Bare ``math.f`` (sin, sqrt, ...) works scalar-eagerly but raises once
# vectorised or traced (needs a host Python float) -- map to the jnp ufunc.
# Most names match; inverse-trig and power differ.
MATH_TO_JNP = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "exp": "exp",
    "log": "log",
    "log2": "log2",
    "log10": "log10",
    "sqrt": "sqrt",
    "pow": "power",
    "floor": "floor",
    "ceil": "ceil",
    "fabs": "fabs",
}


class JnpRewriter(ast.NodeTransformer):
    """``np.<x>`` -> ``jnp.<x>`` plus the jnp spellings of what numpy and jax name differently."""

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id == "np":
            return ast.copy_location(ast.Name(id="jnp", ctx=node.ctx), node)
        # np_float/np_complex are framework globals resolving to the
        # 64-bit dtypes under the x64 config.
        if node.id == "np_float":
            return ast.copy_location(
                ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="float64", ctx=node.ctx), node
            )
        if node.id == "np_complex":
            return ast.copy_location(
                ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="complex128", ctx=node.ctx), node
            )
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        self.generic_visit(node)
        # np.ndarray(shape, dtype=..) is a bare ctor; jnp has none -- use jnp.empty.
        if node.attr == "ndarray":
            return ast.copy_location(ast.Attribute(value=node.value, attr="empty", ctx=node.ctx), node)
        # jax arrays have no C/F layout distinction (always contiguous), so
        # ascontiguousarray/asfortranarray (the latter from rewrite_eigh's
        # non-Name eigh operand, and gromacs_nbnxm) map to plain jnp.asarray.
        if node.attr in ("ascontiguousarray", "asfortranarray"):
            return ast.copy_location(ast.Attribute(value=node.value, attr="asarray", ctx=node.ctx), node)
        # np.intp/uintp (lulesh's gather-index dtype) have no jnp spelling;
        # under x64 they're the 64-bit integer types.
        if node.attr in ("intp", "uintp"):
            return ast.copy_location(
                ast.Attribute(value=node.value, attr="int64" if node.attr == "intp" else "uint64", ctx=node.ctx),
                node,
            )
        return node

    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        # jnp ctors reject numpy's order= (C/F layout; jax has none) -- drop
        # it from ctors only (jnp.reshape DOES honour order=, left intact).
        if isinstance(node.func, ast.Attribute) and node.func.attr in (
            "zeros",
            "ones",
            "empty",
            "full",
            "zeros_like",
            "ones_like",
            "empty_like",
            "full_like",
        ):
            node.keywords = [k for k in node.keywords if k.arg != "order"]
        # max/min over 2+ scalar args (needleman_wunsch/smith_waterman's DP
        # recurrences) fold into a left-nested jnp.maximum/minimum chain --
        # traced scalars can't compare with Python max. A single-iterable
        # max(seq) is untouched.
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in ("max", "min")
            and len(node.args) >= 2
            and not node.keywords
        ):
            attr = "maximum" if node.func.id == "max" else "minimum"
            expr = node.args[0]
            for rhs in node.args[1:]:
                expr = ast.Call(
                    func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr=attr, ctx=ast.Load()),
                    args=[expr, rhs],
                    keywords=[],
                )
            return ast.copy_location(expr, node)
        # Bare math fns (sin(b), sqrt(b[jg])) -> jnp ufuncs so a vectorised/
        # traced arg works (see MATH_TO_JNP).
        if isinstance(node.func, ast.Name) and node.func.id in MATH_TO_JNP:
            return ast.copy_location(
                ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="jnp", ctx=ast.Load()), attr=MATH_TO_JNP[node.func.id], ctx=ast.Load()
                    ),
                    args=node.args,
                    keywords=node.keywords,
                ),
                node,
            )
        # float(x) forces host concretisation, which a traced value can't
        # give (TSVC argmax's checksum needs this) -- route through a
        # traceable jnp cast; safe since the result only ever feeds
        # arithmetic/comparison, never a range/shape. int(x) is NOT
        # rewritten: it can feed range(int(x))/a shape needing a concrete
        # Python int, which a traced cast would break.
        if isinstance(node.func, ast.Name) and node.func.id == "float" and len(node.args) == 1:
            return ast.copy_location(
                ast.Call(
                    func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="asarray", ctx=ast.Load()),
                    args=[
                        node.args[0],
                        ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="float64", ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
                node,
            )
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.expr:
        self.generic_visit(node)
        # Data-dependent ternary (lulesh's Courant limit) can't yield a
        # concrete bool under trace -> jnp.where(cond, a, b). A static
        # ternary (fv3_dycore's ``8 if hord == 10 else hord``) stays a real
        # Python ternary since its arms may differ in shape. An identity
        # test (``x if v is not None else None``) also stays Python --
        # jnp.where can't select None. jit-only; eager runs it as-is.
        none_branch = any(isinstance(b, ast.Constant) and b.value is None for b in (node.body, node.orelse))
        if (
            not STATE.jit_mode
            or is_identity_test(node.test)
            or none_branch
            or names_loaded(node.test) <= (STATE.emit_static | STATE.module_consts)
        ):
            return node
        where = ast.Call(
            func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="where", ctx=ast.Load()),
            args=[bool_cond_ast(node.test), node.body, node.orelse],
            keywords=[],
        )
        return ast.copy_location(where, node)

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:
        self.generic_visit(node)
        # a and b / a or b: when every operand is provably boolean (compare
        # or np.logical_*), this is a mask combine -> elementwise & / | (a
        # traced array has no Python bool, and bitwise-on-bool is exact).
        # For non-boolean operands, and/or return one OPERAND by truthiness
        # (``n or N`` -> N when n is falsy); &/| would bit-combine VALUES
        # instead, a miscompile with no general traceable fix (would need
        # jnp.where(bool(first), ...), itself untraceable + loses short-
        # circuiting). Left verbatim: eager mode's concrete host scalars
        # keep exact numpy semantics; a jit trace of this then raises
        # honestly instead of silently miscompiling.
        if all(is_bool_expr(v) for v in node.values):
            bitop = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
            expr = node.values[0]
            for rhs in node.values[1:]:
                expr = ast.BinOp(left=expr, op=bitop, right=rhs)
            return ast.copy_location(expr, node)
        return node


def np_to_jnp(tree: ast.AST) -> ast.AST:
    """Rewrite ``np.<x>`` -> ``jnp.<x>`` (and bare ``np`` -> ``jnp``)."""
    return JnpRewriter().visit(tree)


def unparse_jnp(node: ast.AST) -> str:
    """Unparse with np->jnp already applied."""
    return ast.unparse(np_to_jnp(ast.fix_missing_locations(node)))


def is_bool_expr(node: ast.AST) -> bool:
    """A provably-boolean array expression: a comparison, ``np.logical_*``/
    predicate call, or bitwise combo of such (force_lj's ``(rsq < cutoffsq) &
    (rsq > 0.0)``) -- ``boolean_mask_transform`` needs this to lower
    ``A[mask] = ..`` to ``where`` rather than an untraceable boolean index."""
    if isinstance(node, ast.Compare):
        return True
    if isinstance(node, ast.Call):
        return any(is_np_attr(node.func, f) for f in BOOL_FUNCS)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
        return is_bool_expr(node.left) and is_bool_expr(node.right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
        return is_bool_expr(node.operand)
    return False


def bool_cond_ast(node: ast.AST) -> ast.AST:
    """Coerce a Python-truthiness condition into an explicit boolean-array
    expression a jit trace can evaluate: ``a and b`` -> ``mask(a) & mask(b)``,
    ``a or b`` -> ``mask(a) | mask(b)``, ``not a`` -> ``~mask(a)``. A
    non-boolean operand (cloudsc's ``if ldcum[jl-1] and plude > rlmin: ..``)
    becomes ``x != 0`` (numpy truthiness) so the whole condition traces
    instead of raising ``TracerBoolConversionError``. Already-boolean operands
    are left untouched."""
    if isinstance(node, ast.BoolOp):
        op = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        expr = bool_cond_ast(node.values[0])
        for v in node.values[1:]:
            expr = ast.BinOp(left=expr, op=op, right=bool_cond_ast(v))
        return expr
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return ast.UnaryOp(op=ast.Invert(), operand=bool_cond_ast(node.operand))
    if is_bool_expr(node):
        return node
    return ast.Compare(left=node, ops=[ast.NotEq()], comparators=[ast.Constant(value=0)])


def cond_str(test: ast.AST) -> str:
    """Unparse a condition (np->jnp) as a traceable boolean-array expression."""
    return unparse_jnp(bool_cond_ast(test))
