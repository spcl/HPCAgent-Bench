"""Control-flow cleanup: validation guards, ``errstate``, ``issubdtype``, dead branches, boolean-op ifs."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import np_attr
from hpcagent_bench.translators.numpyto_common.numpy_desugar.kinds import dtype_arg_kind, dtype_kind


class SpliceErrstate(ast.NodeTransformer):
    """``with np.errstate(<flags>): <body>`` -> ``<body>``.

    ``errstate`` changes how numpy REPORTS an invalid operation (warn / raise / ignore); it never
    changes the value produced -- azimint's empty bin still divides 0 by 0 and still yields nan.
    The native backends do not report at all, so the context is already what ``ignore`` asks for,
    and pythran refuses the statement outright ("With statements not supported").

    Only ``np.errstate`` is spliced. Any other context manager is left standing: a ``with`` that
    owns a resource is not a no-op, and silently dropping it would be a different program.
    """

    def __init__(self) -> None:
        self.changed = False

    def visit_With(self, node: ast.With) -> ast.AST:
        self.generic_visit(node)
        if len(node.items) != 1 or node.items[0].optional_vars is not None:
            return node
        call = node.items[0].context_expr
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "errstate"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id in ("np", "numpy")
        ):
            return node
        self.changed = True
        return node.body


class DropGuards(ast.NodeTransformer):
    """Replace ``raise ...`` / ``assert ...`` statements with ``pass``. These are
    input-validation guards (``if bad: raise ValueError(f"...")``); HPCAgent-Bench kernels
    run on oracle-validated inputs so the guard never fires. Dropping them also
    removes the f-string messages pythran cannot parse and the exception types
    numba/pythran/dace need not express. ``pass`` (not deletion) keeps an
    otherwise-empty ``if`` body syntactically valid."""

    def __init__(self) -> None:
        self.changed = False

    def visit_Raise(self, node: ast.Raise):
        self.changed = True
        return ast.copy_location(ast.Pass(), node)

    def visit_Assert(self, node: ast.Assert):
        self.changed = True
        return ast.copy_location(ast.Pass(), node)


class DropValidationGuards(ast.NodeTransformer):
    """Remove an input-validation guard ``if <cond>: raise/assert`` ENTIRELY -- the
    condition included -- so a ``.ndim`` / ``.flags.c_contiguous`` / ``.dtype`` check
    the native backends cannot emit disappears with the guard, not just the raise
    (which the emitter already skips, leaving the unemittable condition behind).
    Fires only when the whole if-body is raise/assert/pass and there is no else, so a
    real branch is never touched. HPCAgent-Bench kernels run on oracle-validated inputs, so
    the guard never fires (minife's ``_require_float_vector`` rank/contiguity checks)."""

    def visit_If(self, node: ast.If):
        self.generic_visit(node)
        if not node.orelse and node.body and all(isinstance(s, (ast.Raise, ast.Assert, ast.Pass)) for s in node.body):
            return None
        return node


#: ``np.<name>`` abstract dtype category -> the concrete dtype KINDS it covers.
#: Used to fold ``np.issubdtype(x.dtype, np.<name>)`` to a compile-time bool.
ISSUBDTYPE_CATEGORY: dict[str, set] = {
    "integer": {"int"},
    "signedinteger": {"int"},
    "unsignedinteger": {"int"},
    "floating": {"float"},
    "complexfloating": {"complex"},
    "inexact": {"float", "complex"},
    "number": {"int", "float", "complex"},
    "bool_": {"bool"},
    "bool": {"bool"},
}


class IssubdtypeFold(ast.NodeTransformer):
    """Fold ``np.issubdtype(<expr>.dtype, np.<category>)`` -- and the bare
    ``np.issubdtype(np.int32, np.integer)`` -- to a ``True``/``False`` constant
    from the operand's known dtype KIND (bool/int/float/complex). numba/pythran/
    dace cannot evaluate ``np.issubdtype``, but the answer is a compile-time
    property of a statically known dtype, so the branch it feeds resolves and
    (with dead-branch elim) disappears -- the isinstance-style check C/C++
    backends would do natively. Left verbatim when the kind or category is
    unknown."""

    def __init__(self, dtypes: dict[str, str]) -> None:
        self.dtypes = dtypes
        self.changed = False

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if np_attr(node) != "issubdtype" or len(node.args) != 2:
            return node
        a = node.args[0]
        kind = (
            dtype_kind(a.value, self.dtypes)
            if (isinstance(a, ast.Attribute) and a.attr == "dtype")
            else dtype_arg_kind(a)
        )
        cat = node.args[1]
        catname = cat.attr if isinstance(cat, ast.Attribute) else (cat.id if isinstance(cat, ast.Name) else None)
        kinds = ISSUBDTYPE_CATEGORY.get(catname)
        if kind is None or kinds is None:
            return node
        self.changed = True
        return ast.copy_location(ast.Constant(value=(kind in kinds)), node)


class DeadBranchElim(ast.NodeTransformer):
    """Constant-fold boolean guards (``X and False`` -> ``False``, etc.) and drop
    the unreachable branch of ``if <const bool>:``. After the desugar folds a
    ``scipy.sparse.issparse(x)`` guard to ``False`` (dense-only ABI), this removes
    the dead sparse branch entirely -- numba DCEs it before typing, but pythran
    statically types it (``.toarray()`` on a dense array) and errors otherwise."""

    def __init__(self) -> None:
        self.changed = False

    def const_bool(self, node: ast.AST):
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            v = self.const_bool(node.operand)
            return None if v is None else (not v)  # ``not issubdtype(...)`` folds too
        if isinstance(node, ast.BoolOp):
            vals = [self.const_bool(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return False if any(v is False for v in vals) else (True if all(v is True for v in vals) else None)
            return True if any(v is True for v in vals) else (False if all(v is False for v in vals) else None)
        return None

    def visit_If(self, node: ast.If):
        self.generic_visit(node)
        taken = self.const_bool(node.test)
        if taken is True:
            self.changed = True
            return node.body
        if taken is False:
            self.changed = True
            return node.orelse or [ast.copy_location(ast.Pass(), node)]
        return node


#: An ``or`` rewrite CLONES the body once per operand, so a wide disjunction trades one
#: frontend refusal for a source blow-up; the runtime-axis dispatch that needs it has two.
BOOLOP_CLONE_MAX = 4


def symbolic_eq_test(node: ast.expr) -> bool:
    """A single-comparator ``==`` / ``!=`` -- the one shape dace replaces with a bare
    sympy object, and so the only one that has to leave a ``BoolOp`` list field."""
    return isinstance(node, ast.Compare) and len(node.comparators) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq))


class BoolOpIfToChain(ast.NodeTransformer):
    """``if A or B: body`` -> ``if A: body elif B: body``; ``if A and B: body`` ->
    ``if A: (if B: body)``.

    dace's ``RewriteSympyEquality.visit_Compare`` returns a bare ``sympy.Eq``/``Ne`` for a
    comparison with a SYMBOL operand, which breaks the ``ast.NodeTransformer`` contract:
    stock ``generic_visit`` reads a non-AST return from a LIST field as a list of
    replacement nodes and ``.extend()``s it, so a symbolic ``==`` inside ``BoolOp.values``
    refuses the program with ``'Equality' object is not iterable``. The SAME comparison as
    ``If.test`` -- a single field -- is a plain ``setattr`` and works. The rewrite is
    therefore exactly "no such comparison stays a direct child of a ``BoolOp`` list", which
    is what the frontend's runtime-axis dispatch (``if dim == 0 or dim == -2:``) needs.

    Both forms preserve short-circuit evaluation EXACTLY -- every operand is evaluated at
    most once, in source order, under the same condition as before -- so an operand with a
    side effect stays safe. Only the BODY is cloned, into branches that are mutually
    exclusive, so at most one copy ever runs and the tests all stay symbolic (a scalar
    flag would have turned a branch dace specialises at compile time into a
    data-dependent one).

    Left alone: an ``and`` carrying an ``else`` (that one would have to clone the ELSE into
    every level), a disjunction too wide to clone, and a test with no bare ``==``/``!=`` in
    it -- nothing there trips the dace bug, and splitting one would cost
    :class:`DeadBranchElim` its whole-BoolOp constant fold."""

    def __init__(self) -> None:
        self.changed = False

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        test = node.test
        if not isinstance(test, ast.BoolOp) or not any(symbolic_eq_test(v) for v in test.values):
            return node
        if isinstance(test.op, ast.And):
            if node.orelse:
                return node
            inner: list[ast.stmt] = node.body
            for value in reversed(test.values):
                inner = [ast.If(test=value, body=inner, orelse=[])]
        else:
            if len(test.values) > BOOLOP_CLONE_MAX:
                return node
            inner = node.orelse
            for value in reversed(test.values):
                inner = [ast.If(test=value, body=copy.deepcopy(node.body), orelse=inner)]
        self.changed = True
        # A rebuilt test may itself be a BoolOp (``(a or b) and c``); each rewrite strictly
        # shrinks the test's nesting, so re-visiting terminates.
        return ast.copy_location(self.visit(inner[0]), node)
