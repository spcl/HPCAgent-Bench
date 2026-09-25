"""Scalar casts and numpy true division."""

import ast

from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import is_integer_expr


class BuiltinCastRewriter(ast.NodeTransformer):
    """Drop Python's ``float(x)`` cast on the kernel body.

    In C / Fortran the surrounding operation promotes int -> double
    automatically, so the ``float(x)`` Python uses for division
    semantics is a genuine no-op (and the emitter would otherwise
    produce ``float(x)`` literally, which is not valid C).

    ``int(x)`` is NOT dropped: it is a value-changing TRUNCATION, not a
    no-op. Dropping it relied on the target being int-declared so the
    assignment truncated implicitly -- but that is fragile (``y =
    int(x) + 0.5`` with a double ``y`` would silently keep the fraction)
    and, worse, it erases the barrier that keeps int-ness from
    propagating BACKWARD into a float source: after ``ri = int(rs)``
    became ``ri = rs``, the used-as-int analysis walked from the
    index ``ri`` into ``rs`` and mistyped the whole GROMACS distance
    chain (``rsq``/``rinv``/``dx``) as int, truncating every force to
    zero. Every native emitter already renders a bare ``int(x)`` (C/C++
    ``(int)(x)``, Fortran ``INT(x, kind)``), so leaving it in place is
    both correct and faithful.
    """

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        # True-division barrier: a ``float(x)`` operand of ``/`` must be
        # PRESERVED, not dropped -- numpy ``/`` is true division, so an int/int
        # ``float(a) / b`` must stay floating. Rewrite the ``float(x)`` cast to a
        # real ``np.float64(x)`` cast (which the C / Fortran emitters render)
        # BEFORE ``generic_visit`` reaches the inner ``float`` call and drops it.
        if isinstance(node.op, ast.Div):
            node.left = self.keep_float_as_cast(node.left)
            node.right = self.keep_float_as_cast(node.right)
        self.generic_visit(node)
        return node

    @staticmethod
    def keep_float_as_cast(operand: ast.AST) -> ast.AST:
        if (
            isinstance(operand, ast.Call)
            and isinstance(operand.func, ast.Name)
            and operand.func.id == "float"
            and len(operand.args) == 1
        ):
            return ast.copy_location(
                ast.Call(
                    func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="float64", ctx=ast.Load()),
                    args=list(operand.args),
                    keywords=[],
                ),
                operand,
            )
        return operand

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "float" and len(node.args) == 1:
            return node.args[0]
        return node


class ScalarFloatTagger(ast.NodeVisitor):
    """Tag a local scalar float when its DEFINING expression is provably non-integer.

    :func:`is_integer_expr` reads an untagged non-array Name as INTEGER -- right for the
    symbols (``N`` / ``k`` / ``m``) it exists to classify, wrong for a float scalar, which
    reaches it untagged: ``local_dtypes`` carries the arrays, and a scalar derived in the
    body (``e = 0.5 * (b - a)``) is in no table at all. Reading those as integer makes
    :class:`TrueDivisionPromoter` fire on a float ``/`` and bake in an fp64 cast.

    Visits in source order so a tag is available to the statements that follow it, and only
    ever ADDS float tags it can prove (an integer expression stays untagged and keeps the
    old reading). So this can only turn a false promotion OFF -- never a new one on."""

    def __init__(self, tags: dict[str, str], array_names: set[str]) -> None:
        self.tags = tags
        self.array_names = array_names

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return
        name = node.targets[0].id
        if name in self.tags or name in self.array_names:
            return
        if not is_integer_expr(node.value, self.tags, self.array_names):
            self.tags[name] = "float64"


class TrueDivisionPromoter(ast.NodeTransformer):
    """numpy ``/`` is TRUE division: int / int -> float64. C ``/`` and Fortran
    ``/`` do INTEGER division on integer operands, so wrap BOTH operands of an
    all-integer division in an ``np.float64(...)`` cast (which both emitters
    render as ``(double)(x)`` / ``REAL(x, kind=c_double)``) to force a floating
    divide -- matching numpy. Float / complex operands are left untouched (the
    surrounding arithmetic already promotes); ``//`` (FloorDiv) is a distinct op
    handled by the emitters' integer floor macro and is NOT touched here.

    ``np.float64`` is deliberate and stays fp64 even on an fp32 emit: numpy's int/int
    IS float64 regardless of the kernel's float precision, so this cast is faithful. It
    is only correct while the operands really are integers, though -- hence the dtype
    table this is handed must be complete (see :class:`ScalarFloatTagger`). Firing on a
    float divide silently promotes the surrounding expression to double, which fp64
    cannot reveal because there double IS the precision.

    Casting the RIGHT operand as well is what keeps this case distinguishable downstream:
    the C emitter narrows an integer divisor to the KERNEL's float type (numpy's mixed
    float/int rule), and a bare integer left here would read as that case and pull an
    int/int divide down to float32 on an fp32 emit. It also leaves no implicit int -> double
    for the conversion gate; the divide's value is unchanged either way."""

    def __init__(self, local_dtypes, array_names) -> None:
        self.local_dtypes = local_dtypes or {}
        self.array_names = array_names or set()

    @staticmethod
    def as_f64(node: ast.expr) -> ast.expr:
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="float64", ctx=ast.Load()),
                args=[node],
                keywords=[],
            ),
            node,
        )

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.op, ast.Div)
            and is_integer_expr(node.left, self.local_dtypes, self.array_names)
            and is_integer_expr(node.right, self.local_dtypes, self.array_names)
        ):
            node.left = self.as_f64(node.left)
            node.right = self.as_f64(node.right)
        return node
