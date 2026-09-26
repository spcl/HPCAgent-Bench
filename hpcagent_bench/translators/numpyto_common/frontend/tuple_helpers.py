"""Tuple-returning helpers folded into templated expressions spliced at each call."""

import ast
import copy
from collections.abc import Sequence

from hpcagent_bench.translators.numpyto_common import dtypes
from hpcagent_bench.translators.numpyto_common.ir import ArrayDesc, KernelIR, ScalarDesc, SymbolDesc
from hpcagent_bench.translators.numpyto_common.numpy_desugar import rank_table
from hpcagent_bench.translators.numpyto_common.tuple_desugar import desugar_tuples
from hpcagent_bench.translators.numpyto_common.frontend.axes import AxisReshapeToIndexing
from hpcagent_bench.translators.numpyto_common.frontend.body_rewrites import native_desugar
from hpcagent_bench.translators.numpyto_common.frontend.helper_params import infer_helper_params
from hpcagent_bench.translators.numpyto_common.frontend.helper_specialize import bind_call_constants, substitute_names
from hpcagent_bench.translators.numpyto_common.frontend.inlining import strip_docstrings_
from hpcagent_bench.translators.numpyto_common.frontend.module_constants import inline_module_constants

__all__ = [
    "InlineTupleHelperCalls",
    "desugar_helper_tuples",
    "fold_call_arg_constant",
    "folded_straight_line",
    "return_expression",
    "rewrite_helper_axes",
    "tuple_leaves",
    "tuple_template_for_call",
]


def desugar_helper_tuples(
    hfn: ast.FunctionDef, arrays: list[ArrayDesc], scalars: list[ScalarDesc], symbols: Sequence[SymbolDesc] = ()
) -> None:
    """Run :func:`desugar_tuples` on a helper that survived inlining, against ITS OWN param ranks.

    The kernel body gets this pass once, inside ``parse_kernel`` (ranks from its declared array
    args). A helper built here by :func:`build_helper_kirs` is a second, separate ``KernelIR`` --
    without its own call, ``axes = tuple(range(2, x.ndim))`` (the instance-norm idiom) never folds
    and reaches the structural-axis guard below as a runtime ``Call``, not a literal tuple.

    ``symbols`` (always integer, see :class:`SymbolDesc`) count as int scalars too: a call-site
    argument classified as a size symbol rather than a plain scalar (``_as_tuple(pool_kernel_size,
    3)`` where the sizer reads ``pool_kernel_size`` as a dimension) otherwise has no known KIND, so
    ``isinstance(value, tuple)`` cannot decide and the dead guard branch survives -- which then
    disqualifies the tuple-returning shape :func:`tuple_template_for_call` looks for.
    """
    ranks = {a.name: len(a.shape) for a in arrays}
    # ScalarDesc.dtype is already canonicalized (see ScalarDesc.__post_init__), so a plain name-shape
    # check is exact here -- same split the emitters use, not a guess.
    int_scalars = frozenset(s.name for s in scalars if dtypes.is_integer(s.dtype)) | frozenset(s.name for s in symbols)
    float_scalars = frozenset(s.name for s in scalars if s.dtype.startswith("float"))
    desugar_tuples(hfn, int_scalars=int_scalars, float_scalars=float_scalars, arrays=frozenset(ranks), ranks=ranks)


def fold_call_arg_constant(
    arg: ast.expr, arrays: list[ArrayDesc], scalars: list[ScalarDesc], symbols: Sequence[SymbolDesc] = ()
) -> ast.Constant | None:
    """``arg`` reduced to a literal against the kernel's own rank/scalar tables, or ``None``.

    A call-site argument is not always spelled as a bare literal -- ``_as_tuple(v, x.ndim - 2)``
    picks its count off the operand's rank, same as the ``(1,) * (x.ndim - 2)`` broadcast idiom
    :mod:`tuple_desugar` already folds. Reuses that SAME fold (via :func:`desugar_helper_tuples`
    on a throwaway one-line probe) rather than a second constant-arithmetic implementation.
    """
    probe = ast.parse("def __probe():\n return __ARG__\n").body[0]
    probe.body[0].value = copy.deepcopy(arg)
    ast.fix_missing_locations(probe)
    desugar_helper_tuples(probe, arrays, scalars, symbols)
    folded = probe.body[0].value
    return folded if isinstance(folded, ast.Constant) else None


def rewrite_helper_axes(hfn: ast.FunctionDef, arrays: list[ArrayDesc], scalars: list[ScalarDesc]) -> None:
    """The axis-to-indexing rewrite the kernel body gets, against the helper's OWN param ranks.

    ``expand_dims`` / ``squeeze`` / ``swapaxes`` / ``moveaxis`` are pure index rewrites, but each
    needs the operand's rank. A helper that survives inlining never went through the kernel's own
    pass, so squeezenet's ``np.moveaxis(x, 1, -1)`` on a helper parameter reached lowering as an
    unsupported call while the SAME line inside an inlined helper folded.
    """
    ranks = {a.name: len(a.shape) for a in arrays}
    AxisReshapeToIndexing(rank_table(hfn, ranks), frozenset(s.name for s in scalars)).visit(hfn)
    ast.fix_missing_locations(hfn)


def folded_straight_line(body: list[ast.stmt]) -> list[ast.stmt] | None:
    """``body`` with its leading single-assignment locals folded into the statements that read them.

    A tuple-returning helper has no ABI to be called across, so it is spliced into each call site as
    ONE expression -- which needs a body that only ever returns. The useful ones compute a few index
    locals first: ``_tap_span`` binds ``offset``, ``rhs`` and its four bounds, then picks between
    three 4-tuples on guards over them. :func:`return_expression` saw an ``Assign`` at the head,
    declined, and three conv_transpose kernels emitted no DaCe program at all.

    Only a name bound ONCE is folded, and only across a run of such assignments. A rebinding needs
    the value live at each read, which one substitution cannot express, so it declines instead.
    """
    folded: dict[str, ast.expr] = {}
    out: list[ast.stmt] = []
    for stmt in strip_docstrings_(body):  # a helper's docstring is an Expr, and it is the FIRST statement
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            if name in folded:
                return None
            value = substitute_names(copy.deepcopy(stmt.value), folded)  # a later local may read an earlier one
            folded[name] = value
            continue
        if not isinstance(stmt, (ast.If, ast.Return)):
            return None
        # The locals are interleaved WITH the guards, not merely ahead of them: ``_tap_span``
        # bails, computes ``iz_hi``, bails again, then computes the bounds it returns. A guard that
        # rebound a folded name would make one substitution stand for two values, so decline.
        if any(
            isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store) and sub.id in folded for sub in ast.walk(stmt)
        ):
            return None
        kept = copy.deepcopy(stmt)
        substitute_names(kept, folded)
        out.append(kept)
    return out


def return_expression(body: list[ast.stmt]) -> ast.expr | None:
    """A body that only ever returns, collapsed into ONE expression, or ``None``.

    A guard the helper-level fold could not decide stays in that expression as an ``IfExp``, for
    the SPLICE SITE to decide. ``_as_tuple(value, dims)`` returns ``value`` untouched when it
    already is a tuple, so the guard's answer belongs to each call site, not to whichever site
    happened to be inspected first.
    """
    if not body:
        return None
    head = body[0]
    if isinstance(head, ast.Return):
        return head.value
    if not isinstance(head, ast.If):
        return None
    taken = return_expression(head.body)
    other = return_expression(head.orelse if head.orelse else body[1:])
    if taken is None or other is None:
        return None
    return ast.IfExp(test=head.test, body=taken, orelse=other)


def tuple_leaves(expr: ast.expr) -> list[ast.expr]:
    """The values an ``IfExp`` chain can evaluate to, in branch order."""
    if isinstance(expr, ast.IfExp):
        return tuple_leaves(expr.body) + tuple_leaves(expr.orelse)
    return [expr]


def tuple_template_for_call(
    hdef: ast.FunctionDef,
    call: ast.Call,
    tree: ast.Module,
    parent: KernelIR,
    arr_by: dict[str, ArrayDesc],
    sca_by: dict[str, ScalarDesc],
    sym_by: dict[str, SymbolDesc],
    kernel_fn: ast.FunctionDef,
) -> ast.expr | None:
    """``hdef`` folded against THIS call's own arguments as one spliceable expression, or ``None``.

    A helper whose every branch yields a fixed-length tuple has no C/Fortran ABI at all -- there is
    no tuple return value -- so the correct lowering is to splice its result into the call site
    rather than emit it as a function.

    Only the ARRAY parameters carry a descriptor into the fold. A scalar one would hand the guard a
    kind to decide on, and the kind that decides it is the argument's at the splice site, which
    this helper's own tables cannot see: :func:`infer_param_desc` falls back to "float64 scalar"
    for a kernel local, and that fallback is what answered ``isinstance(stride, tuple)`` with False
    for a ``stride`` the line above had already tupled.
    """
    hfn = copy.deepcopy(hdef)
    pnames = [a.arg for a in hfn.args.args]
    inline_module_constants(tree, hfn, pnames)
    native_desugar(hfn)
    consts: dict[str, ast.expr] = {}
    for pname, arg in zip(pnames, call.args):
        folded = (
            arg
            if isinstance(arg, ast.Constant)
            else fold_call_arg_constant(arg, parent.arrays, parent.scalars, parent.symbols)
        )
        if folded is not None:
            consts[pname] = folded
    bind_call_constants(hfn, consts)
    arrays, unused, unused = infer_helper_params(pnames, call.args, arr_by, sca_by, sym_by, kernel_fn)
    desugar_helper_tuples(hfn, arrays, [], [])
    straight = folded_straight_line(hfn.body)
    expr = return_expression(straight) if straight is not None else None
    if expr is None:
        return None
    leaves = tuple_leaves(expr)
    if not any(isinstance(leaf, ast.Tuple) for leaf in leaves):
        return None
    if not all(isinstance(leaf, ast.Tuple) or (isinstance(leaf, ast.Name) and leaf.id in pnames) for leaf in leaves):
        return None
    return expr


class InlineTupleHelperCalls(ast.NodeTransformer):
    """Replace each call to one tuple-returning helper with THAT CALL'S OWN templated result, each
    parameter substituted by that call's argument expression.

    One template per call site, never one for the whole helper: ``_as_tuple``'s result depends on
    its argument's TYPE, so a template resolved against the first call site is wrong at any site
    passing a different kind. That is how a second ``_as_tuple(stride, 2)`` on an already-tupled
    ``stride`` became ``(stride, stride)`` -- a self-referential binding that dropped the statement
    holding the real value and left every later ``stride[i]`` reading the ``None`` it was seeded
    with, which is a wrong loop nest rather than a refusal.
    """

    def __init__(self, pnames: list[str], templates: dict[int, ast.expr]) -> None:
        self.pnames = pnames
        self.templates = templates

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        template = self.templates.get(id(node))
        if template is None:
            return node
        substituted = substitute_names(copy.deepcopy(template), dict(zip(self.pnames, node.args)))
        return ast.copy_location(substituted, node)
