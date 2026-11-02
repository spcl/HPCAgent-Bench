"""Control-flow cleanup: validation guards, ``errstate``, ``issubdtype``, dead branches, boolean-op ifs."""

import ast
import builtins
import copy

from hpcagent_bench.translators.numpyto_common.ast_build import walk_nodes
from hpcagent_bench.translators.numpyto_common.emit_helpers.numpy_names import is_numpy_module, numpy_call_attr
from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import RewritePass
from hpcagent_bench.translators.numpyto_common.numpy_desugar.kinds import dtype_arg_kind, dtype_kind

__all__ = [
    "BOOLOP_CLONE_MAX",
    "GUARD_MESSAGE",
    "ISSUBDTYPE_CATEGORY",
    "METADATA_ATTRS",
    "METADATA_CALLS",
    "BoolOpIfToChain",
    "DeadBranchElim",
    "DropGuards",
    "DropValidationGuards",
    "IssubdtypeFold",
    "SpliceErrstate",
    "symbolic_eq_test",
    "validates_metadata",
]


class SpliceErrstate(RewritePass):
    """``with np.errstate(<flags>): <body>`` -> ``<body>``.

    ``errstate`` changes only how numpy reports an invalid operation, never the value; the native
    backends do not report at all, and pythran rejects ``with``. Other context managers may own a
    resource and are left standing.
    """

    __slots__ = ()

    def visit_With(self, node: ast.With) -> ast.AST | list[ast.stmt]:
        self.generic_visit(node)
        if len(node.items) != 1 or node.items[0].optional_vars is not None:
            return node
        call = node.items[0].context_expr
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "errstate"
            and is_numpy_module(call.func.value)
        ):
            return node
        self.changed = True
        return node.body


#: What a kernel guard raises when its own message is not a constant: numba cannot format one at run time.
GUARD_MESSAGE = "kernel guard"

#: Reads of an array's metadata; a test holding one validates its input's shape or dtype, never its values.
METADATA_ATTRS = frozenset({"dtype", "flags", "ndim", "shape"})
METADATA_CALLS = frozenset({"isinstance", "issubdtype"})


def validates_metadata(test: ast.expr) -> bool:
    """Whether a guard's ``test`` checks shape or dtype rather than data."""
    for node in walk_nodes(test):
        if isinstance(node, ast.Attribute) and node.attr in METADATA_ATTRS | METADATA_CALLS:
            return True
        if isinstance(node, ast.Name) and node.id in METADATA_CALLS:
            return True
    return False


class DropGuards(RewritePass):
    """``raise ...`` / ``assert ...`` -> ``pass``; ``pass`` keeps an otherwise-empty ``if`` body valid.

    Sound for a validation guard: kernels run on oracle-validated inputs. A guard on data (a CG breakdown, a sanity
    check) can fire on such inputs, so ``keep_data`` keeps it, with a constant message, for a backend that can raise."""

    __slots__ = ("data_tests", "keep_data")

    def __init__(self, keep_data: bool = False) -> None:
        super().__init__()
        self.keep_data = keep_data
        self.data_tests = 0

    def visit_If(self, node: ast.If) -> ast.AST:
        data = self.keep_data and not validates_metadata(node.test)
        self.data_tests += data
        self.generic_visit(node)
        self.data_tests -= data
        return node

    def visit_Raise(self, node: ast.Raise) -> ast.stmt:
        exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        builtin = getattr(builtins, exc.id, None) if isinstance(exc, ast.Name) else None
        if not (self.data_tests and isinstance(builtin, type) and issubclass(builtin, BaseException)):
            self.changed = True
            return ast.copy_location(ast.Pass(), node)
        if isinstance(node.exc, ast.Call) and not all(isinstance(a, ast.Constant) for a in node.exc.args):
            self.changed = True
            node.exc.args = [ast.Constant(value=GUARD_MESSAGE)]
        return node

    def visit_Assert(self, node: ast.Assert) -> ast.stmt:
        if not (self.keep_data and not validates_metadata(node.test)):
            self.changed = True
            return ast.copy_location(ast.Pass(), node)
        if node.msg is not None and not isinstance(node.msg, ast.Constant):
            self.changed = True
            node.msg = ast.Constant(value=GUARD_MESSAGE)
        return node


class DropValidationGuards(ast.NodeTransformer):
    """Remove ``if <cond>: raise/assert`` entirely, condition included.

    The condition (``.ndim`` / ``.flags.c_contiguous`` / ``.dtype``) is often unemittable. Fires only
    when the body is all raise/assert/pass with no else; inputs are oracle-validated, so it never fires."""

    __slots__ = ()

    def visit_If(self, node: ast.If) -> ast.stmt | None:
        self.generic_visit(node)
        if not node.orelse and node.body and all(isinstance(s, (ast.Raise, ast.Assert, ast.Pass)) for s in node.body):
            return None
        return node


#: ``np.<name>`` abstract dtype category -> the concrete dtype kinds it covers.
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
    """``np.issubdtype(<expr>.dtype | np.<dtype>, np.<category>)`` -> ``True``/``False`` from the known kind.

    The backends cannot evaluate ``np.issubdtype``; folding lets :class:`DeadBranchElim` drop the branch.
    Left verbatim when the kind or category is unknown."""

    __slots__ = ("changed", "dtypes")

    def __init__(self, dtypes: dict[str, str]) -> None:
        self.dtypes = dtypes
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        if numpy_call_attr(node) != "issubdtype" or len(node.args) != 2:
            return node
        a = node.args[0]
        kind = (
            dtype_kind(a.value, self.dtypes)
            if (isinstance(a, ast.Attribute) and a.attr == "dtype")
            else dtype_arg_kind(a)
        )
        cat = node.args[1]
        catname = cat.attr if isinstance(cat, ast.Attribute) else (cat.id if isinstance(cat, ast.Name) else None)
        kinds = ISSUBDTYPE_CATEGORY.get(catname) if catname is not None else None
        if kind is None or kinds is None:
            return node
        self.changed = True
        return ast.copy_location(ast.Constant(value=(kind in kinds)), node)


class DeadBranchElim(RewritePass):
    """Drop the unreachable branch of ``if <const bool>:``, folding ``not`` / ``and`` / ``or`` of constants.

    pythran types dead branches too (e.g. ``.toarray()`` under a folded ``issparse``) and would reject them."""

    __slots__ = ()

    def const_bool(self, node: ast.expr) -> bool | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            v = self.const_bool(node.operand)
            return None if v is None else (not v)
        if isinstance(node, ast.BoolOp):
            vals = [self.const_bool(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return False if any(v is False for v in vals) else (True if all(v is True for v in vals) else None)
            return True if any(v is True for v in vals) else (False if all(v is False for v in vals) else None)
        return None

    def visit_If(self, node: ast.If) -> ast.stmt | list[ast.stmt]:
        self.generic_visit(node)
        taken = self.const_bool(node.test)
        if taken is True:
            self.changed = True
            return node.body
        if taken is False:
            self.changed = True
            return node.orelse or [ast.copy_location(ast.Pass(), node)]
        return node


#: An ``or`` rewrite clones the body once per operand; wider disjunctions are left alone.
BOOLOP_CLONE_MAX = 4


def symbolic_eq_test(node: ast.expr) -> bool:
    """A single-comparator ``==`` / ``!=``: the shape dace replaces with a bare sympy object."""
    return isinstance(node, ast.Compare) and len(node.comparators) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq))


class BoolOpIfToChain(RewritePass):
    """``if A or B: body`` -> ``if A: body elif B: body``; ``if A and B: body`` ->
    ``if A: (if B: body)``.

    dace's ``RewriteSympyEquality`` returns a bare ``sympy.Eq``/``Ne`` for a symbolic comparison; inside
    a ``BoolOp.values`` list ``generic_visit`` tries to iterate it and fails, while as ``If.test`` it
    works. So no such comparison may stay a direct child of a ``BoolOp``.

    Sound: short-circuit order is preserved (each operand evaluated at most once, in order, under the
    same condition), and the cloned bodies sit in mutually exclusive branches, so at most one runs.

    Left alone: an ``and`` with an ``else`` (would clone the else per level), a disjunction wider than
    :data:`BOOLOP_CLONE_MAX`, and a test with no bare ``==``/``!=`` (keeps :class:`DeadBranchElim`'s
    whole-BoolOp fold)."""

    __slots__ = ()

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
        # A rebuilt test may itself be a BoolOp; each rewrite shrinks the nesting, so this terminates.
        return ast.copy_location(self.visit(inner[0]), node)
