"""One :class:`KernelIR` per helper that survives inlining."""

import ast
import copy
import dataclasses
from collections.abc import Sequence

from hpcagent_bench.translators.numpyto_common.ir import ArrayDesc, KernelIR, ScalarDesc, SymbolDesc
from hpcagent_bench.translators.numpyto_common.ordered import OrderedSet
from hpcagent_bench.translators.numpyto_common.frontend.axes import reject_symbolic_axis, reject_unsupported_slices
from hpcagent_bench.translators.numpyto_common.frontend.body_rewrites import native_desugar
from hpcagent_bench.translators.numpyto_common.frontend.callsite import (
    ReplaceStmts,
    build_callsite_stmts,
    caller_side_shape,
    caller_side_symbol,
    held_before,
    reorder_helper_call_args,
    shape_symbols,
)
from hpcagent_bench.translators.numpyto_common.frontend.helper_params import (
    DescKey,
    infer_helper_params,
    mark_written_outputs,
    reject_subscripted_scalar_params,
    widen_counting_scalar_params,
)
from hpcagent_bench.translators.numpyto_common.frontend.helper_shapes import (
    desc_key,
    helper_return_array_shape,
    helper_return_shape_from_body,
    structure_key,
    call_specialized_body,
    helper_call_local_arrays,
    helper_returns_rank0,
    target_shape_is_the_call_itself,
)
from hpcagent_bench.translators.numpyto_common.frontend.helper_specialize import (
    bind_call_constants,
    literal_call_arg,
    literal_key,
    rewrite_returns_to_outparam,
    specialise_helpers_by_call_signature,
)
from hpcagent_bench.translators.numpyto_common.frontend.inlining import HoistMultiStmtHelpers, unroll_const_list_loops
from hpcagent_bench.translators.numpyto_common.frontend.module_constants import inline_module_constants
from hpcagent_bench.translators.numpyto_common.frontend.shapes import local_array_def, conflicting_rebind_shapes
from hpcagent_bench.translators.numpyto_common.frontend.tuple_helpers import (
    InlineTupleHelperCalls,
    desugar_helper_tuples,
    fold_call_arg_constant,
    rewrite_helper_axes,
    tuple_template_for_call,
)

__all__ = [
    "ArraySpec",
    "HelperKirBuilder",
    "Scope",
    "Site",
    "abi_hostile_arguments",
    "build_helper_kirs",
    "collect_called_helper_defs",
    "helper_call_sites",
    "helpers_callers_first",
]


def collect_called_helper_defs(tree: ast.Module, kernel_fn: ast.FunctionDef) -> list[ast.FunctionDef]:
    """Top-level helper ``FunctionDef``s still CALLED after inlining -- the ones
    inlining could not absorb (an early ``return`` / recursion). Collected
    transitively: a captured helper may call another non-inlinable helper, which
    must be emitted too. Returned in definition order (a callee defined above its
    caller emits first, so no forward declaration is needed)."""
    defs_by_name: dict[str, ast.FunctionDef] = {
        n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n is not kernel_fn
    }
    captured: dict[str, ast.FunctionDef] = {}
    frontier: list[ast.AST] = [kernel_fn]
    while frontier:
        node = frontier.pop()
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id in defs_by_name
                and sub.func.id not in captured
            ):
                d = defs_by_name[sub.func.id]
                captured[sub.func.id] = d
                frontier.append(d)
    # Definition order (as they appear in the module), so a helper that calls
    # another emits after its callee.
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in captured]


def helper_call_sites(
    fn: ast.FunctionDef,
) -> tuple[dict[str, ast.Call], dict[str, ast.Assign], dict[str, list[ast.Assign]]]:
    """Helper call sites inside ONE scope: the first call of each name, its enclosing assignment
    (``X = h(...)`` / ``X[:, j] = h(...)`` -- the LHS classifies the return, array out-param vs
    by-value scalar, and sizes it), and EVERY ``X = h(...)`` site.

    Rewriting only the first left the others spelling the helper's ORIGINAL signature while the
    definition had moved to the out-param ABI; the arity happened to still match, so the reorder
    permuted unrelated slots and emitted a call that does not compile (vgg16's
    ``_maxpool2d(2, h[...], 2)``)."""
    call_of: dict[str, ast.Call] = {}
    assign_of: dict[str, ast.Assign] = {}
    assigns_of: dict[str, list[ast.Assign]] = {}
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ):
            assigns_of.setdefault(node.value.func.id, []).append(node)
            if node.value.func.id not in call_of:
                call_of[node.value.func.id] = node.value
                assign_of[node.value.func.id] = node
    for node in ast.walk(fn):  # plain-call fallback (scalar helper in an expression)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in call_of:
            call_of[node.func.id] = node
    return call_of, assign_of, assigns_of


def helpers_callers_first(helper_defs: list[ast.FunctionDef], kernel_fn: ast.FunctionDef) -> list[ast.FunctionDef]:
    """``helper_defs`` reordered so a helper is visited before every helper it calls.

    A helper's parameters are inferred from a call site, and when the only site sits in a SIBLING
    that sibling's descriptor tables are the resolution scope -- which exist only once the sibling
    itself has been built. Independent helpers keep definition order. The emission order the C
    backend needs (callee defined above its caller, so no forward declaration) is a property of
    ``out``, restored by sorting it back at the end of :func:`build_helper_kirs`."""
    names = {h.name: h for h in helper_defs}
    calls = {
        h.name: sorted(
            {
                n.func.id
                for n in ast.walk(h)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in names
                and n.func.id != h.name
            }
        )
        for h in helper_defs
    }
    ordered: list[ast.FunctionDef] = []
    placed: OrderedSet[str] = OrderedSet()

    def visit(h: ast.FunctionDef) -> None:
        if h.name in placed:
            return
        placed.add(h.name)
        ordered.append(h)
        for callee in calls[h.name]:
            visit(names[callee])

    # Seeded from what the KERNEL calls, not from definition order: helpers are conventionally
    # written callee-above-caller, so walking definition order visits a sibling-only callee before
    # the caller whose tables resolve it, and it is skipped for having no reachable call site --
    # which is the very refusal this ordering exists to remove.
    for node in ast.walk(kernel_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names:
            visit(names[node.func.id])
    for h in helper_defs:
        visit(h)
    return ordered


def abi_hostile_arguments(tree: ast.Module, hname: str) -> list[str]:
    """Constants a call site binds to ``hname`` that no C or Fortran parameter table has a type for.

    ``None`` is not a value in either language and neither is a string selector, so a helper called
    as ``_reduce(v, mode='total')`` or ``_default_stride(None, k)`` has no standalone ABI however
    well its body lowers -- the argument only exists to be resolved at the call site. Both forms are
    decidable exactly BECAUSE the site pins them to a literal, which is what the inline path does
    with them; refusing here hands the kernel to that path rather than emitting a signature naming a
    parameter the caller cannot pass.
    """
    seen: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == hname):
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Constant) and (arg.value is None or isinstance(arg.value, str)):
                spelling = repr(arg.value)
                if spelling not in seen:
                    seen.append(spelling)
    return seen


#: A scope a helper call can live in, with the descriptors its arguments resolve against.
Scope = tuple[ast.FunctionDef, list[ArrayDesc], list[ScalarDesc], list[SymbolDesc]]


@dataclasses.dataclass(slots=True, kw_only=True)
class Site:
    """One helper being built: its first call site, the owning scope, and its working copy."""

    hdef: ast.FunctionDef
    hidx: int
    owner_fn: ast.FunctionDef
    oarrays: list[ArrayDesc]
    oscalars: list[ScalarDesc]
    osymbols: list[SymbolDesc]
    #: Owner descriptors by name; arrays include the locals bound from other helpers' calls.
    oarr_by: dict[str, ArrayDesc]
    osca_by: dict[str, ScalarDesc]
    osym_by: dict[str, SymbolDesc]
    call: ast.Call
    assign: ast.Assign | None
    assigns: list[ast.Assign]
    lhs: ast.expr | None
    hfn: ast.FunctionDef
    pnames: list[str]
    hconsts: set[str]

    def held_names(self) -> set[str]:
        """Names the owner can pass: its symbols, arrays, scalars and every shape symbol they name."""
        return (
            {sy.name for sy in self.osymbols}
            | {a.name for a in self.oarrays}
            | {sc.name for sc in self.oscalars}
            | set(shape_symbols(self.oarrays))
        )


@dataclasses.dataclass(slots=True, kw_only=True)
class ArraySpec:
    """What every call site of one array-returning helper is checked against and rewritten with."""

    decl_pnames: list[str]
    pnames: list[str]
    kept_args: list[ast.expr]
    call_consts: dict[str, ast.expr]
    param_info: dict[str, tuple[tuple[str, ...], str]]
    extra_syms: list[str]
    hret_shape: list[str]
    hret_dtype: str | None
    inout: bool


class HelperKirBuilder:
    """Builds one :class:`KernelIR` per helper that survives inlining, rewriting its call sites.

    Helpers are visited callers-first, so a helper called only from a sibling resolves its arguments
    against that sibling's descriptors (registered in ``scopes`` once the sibling is built).
    """

    __slots__ = (
        "callsite_rewrites",
        "generation",
        "helper_defs",
        "keep_helpers",
        "kernel_fn",
        "local_arrays",
        "out",
        "parent",
        "scopes",
        "tree",
    )

    def __init__(self, tree: ast.Module, kernel_fn: ast.FunctionDef, parent: KernelIR, keep_helpers: bool) -> None:
        self.tree = tree
        self.kernel_fn = kernel_fn
        self.parent = parent
        self.keep_helpers = keep_helpers
        self.helper_defs: list[ast.FunctionDef] = []
        self.out: list[KernelIR] = []
        #: {id(owner tree): {id(Assign): replacement stmts}}; call sites are not all in the kernel body.
        self.callsite_rewrites: dict[int, dict[int, list[ast.stmt]]] = {}
        self.scopes: list[Scope] = [(kernel_fn, parent.arrays, parent.scalars, parent.symbols)]
        #: Memo of :func:`helper_call_local_arrays`; ``generation`` retires entries whose trees changed.
        self.local_arrays: dict[tuple[str, int, DescKey, DescKey, DescKey], dict[str, ArrayDesc]] = {}
        self.generation = 0

    def build(self) -> list[KernelIR]:
        self.helper_defs = collect_called_helper_defs(self.tree, self.kernel_fn)
        if not self.helper_defs:
            return []
        # Lift helper calls nested in expressions into assignments: the assignment target is what
        # classifies the return as an array.
        HoistMultiStmtHelpers({h.name: h for h in self.helper_defs}).visit(self.kernel_fn)
        ast.fix_missing_locations(self.kernel_fn)
        arr_by = {a.name: a for a in self.parent.arrays}
        if specialise_helpers_by_call_signature(self.tree, self.kernel_fn, self.helper_defs, arr_by):
            self.helper_defs = collect_called_helper_defs(self.tree, self.kernel_fn)
        hidx_of = {id(h): i for i, h in enumerate(self.helper_defs)}
        for hdef in helpers_callers_first(self.helper_defs, self.kernel_fn):
            site = self.first_site(hdef, hidx_of[id(hdef)])
            if site is not None:
                self.build_helper(site)
        for owner_fn, unused, unused, unused in self.scopes:
            rewrites = self.callsite_rewrites.get(id(owner_fn))
            if rewrites:
                ReplaceStmts(rewrites).visit(owner_fn)
                ast.fix_missing_locations(owner_fn)
        self.refuse_unemitted_callees()
        # Last, once every helper's param_order() is final and the call sites are rewritten.
        reorder_helper_call_args([self.kernel_fn] + [h.tree for h in self.out], self.out)
        # Built callers-first, emitted callee-first so a C caller sees a definition.
        defn_order = {h.name: i for i, h in enumerate(self.helper_defs)}
        self.out.sort(key=lambda ir: defn_order[ir.kernel_name])
        return self.out

    def rewrote(self, owner: ast.FunctionDef) -> None:
        """``owner``'s tree changed: renumber it and retire the memo."""
        self.generation += 1
        ast.fix_missing_locations(owner)

    def first_site(self, hdef: ast.FunctionDef, hidx: int) -> Site | None:
        """The first call of ``hdef`` in the first scope that calls it, or ``None`` when none does."""
        for owner_fn, oarrays, oscalars, osymbols in self.scopes:
            call_of, assign_of, assigns_of = helper_call_sites(owner_fn)
            if hdef.name in call_of:
                break
        else:
            return None
        # Only while building the kept form: on the inlined retry a refusal has no fallback left.
        hostile = abi_hostile_arguments(self.tree, hdef.name) if self.keep_helpers else []
        if hostile:
            raise NotImplementedError(
                f"helper {hdef.name!r} is called with {hostile}, which no ABI carries; "
                f"it must be inlined into its caller"
            )
        oarr_by = {a.name: a for a in oarrays}
        osca_by = {s.name: s for s in oscalars}
        osym_by = {s.name: s for s in osymbols}
        # Locals bound from another helper's call carry a shape only the callee's return gives.
        # Resolution only: they never join ``oarrays`` (the emitted ABI).
        oarr_by.update(copy.deepcopy(self.chased_locals(owner_fn, oarr_by, osca_by, osym_by)))
        assign = assign_of.get(hdef.name)
        hfn = copy.deepcopy(hdef)
        pnames = [a.arg for a in hfn.args.args]
        return Site(
            hdef=hdef,
            hidx=hidx,
            owner_fn=owner_fn,
            oarrays=oarrays,
            oscalars=oscalars,
            osymbols=osymbols,
            oarr_by=oarr_by,
            osca_by=osca_by,
            osym_by=osym_by,
            call=call_of[hdef.name],
            assign=assign,
            assigns=assigns_of.get(hdef.name, [assign] if assign is not None else []),
            lhs=assign.targets[0] if assign is not None else None,
            hfn=hfn,
            pnames=pnames,
            hconsts=set(),
        )

    def chased_locals(
        self,
        owner_fn: ast.FunctionDef,
        oarr_by: dict[str, ArrayDesc],
        osca_by: dict[str, ScalarDesc],
        osym_by: dict[str, SymbolDesc],
    ) -> dict[str, ArrayDesc]:
        key = (structure_key(owner_fn), self.generation, desc_key(oarr_by), desc_key(osca_by), desc_key(osym_by))
        chased = self.local_arrays.get(key)
        if chased is None:
            chased = helper_call_local_arrays(owner_fn, self.helper_defs, oarr_by, osca_by, osym_by)
            self.local_arrays[key] = chased
        return chased

    def build_helper(self, site: Site) -> None:
        hret_shape, hret_dtype = helper_return_array_shape(site.lhs, site.oarr_by, site.owner_fn)
        # Every extent comes off the first call site, so an argument rebound to another shape
        # elsewhere makes the inference unsound.
        for node in ([site.lhs] if site.lhs is not None else []) + list(site.call.args):
            clash = conflicting_rebind_shapes(site.owner_fn, node, site.oarr_by, ignore=site.assign)
            if clash is not None:
                raise NotImplementedError(
                    f"helper {site.hdef.name!r} is called on {node.id!r}, which is rebound to both {clash[0]} and "
                    f"{clash[1]}; the helper's extents are emitted as constants and cannot serve both"
                )
        # The parent's folded names carry over: array params reuse the parent's folded shapes.
        site.hconsts = set(self.parent.inlined_consts) | set(inline_module_constants(self.tree, site.hfn, site.pnames))
        # The kernel body's desugars and const-list unroll; before mark_written_outputs so a
        # ufunc-out / roll rewrite counts as a write.
        native_desugar(site.hfn)
        unroll_const_list_loops(site.hfn)
        if hret_shape is None or target_shape_is_the_call_itself(site.owner_fn, site.lhs, site.oarr_by, site.hdef.name):
            hret_shape, hret_dtype = self.return_shape_from_body(site, hret_shape, hret_dtype)
        if hret_shape is None:
            self.build_by_value(site)
        else:
            self.build_array_return(site, hret_shape, hret_dtype)

    @staticmethod
    def return_shape_from_body(
        site: Site, hret_shape: list[str] | None, hret_dtype: str | None
    ) -> tuple[list[str] | None, str | None]:
        """The helper body's own answer when the call-site target says nothing: it wins when it sizes
        the return, when the target gave nothing, or when every return is provably rank 0 (a
        reduction must stay by-value even though the target guess broadcast it)."""
        probe = call_specialized_body(site.hfn, site.pnames, site.call.args)
        body_shape, body_dtype = helper_return_shape_from_body(
            probe, site.pnames, site.call.args, site.oarr_by, site.osca_by, site.osym_by, site.owner_fn
        )
        if (
            body_shape is not None
            or hret_shape is None
            or helper_returns_rank0(
                probe, site.pnames, site.call.args, site.oarr_by, site.osca_by, site.osym_by, site.owner_fn
            )
        ):
            return body_shape, body_dtype
        return hret_shape, hret_dtype

    def build_by_value(self, site: Site) -> None:
        """A scalar (by-value) or void helper. Compile-time call arguments are bound into the body
        but every parameter slot stays: the call is left where the caller put it."""
        hfn, pnames = site.hfn, site.pnames
        call_consts: dict[str, ast.expr] = {}
        for pn, a in zip(pnames, site.call.args):
            if literal_call_arg(a):
                call_consts[pn] = a
                continue
            folded = fold_call_arg_constant(a, site.oarrays, site.oscalars, site.osymbols)
            if folded is not None:
                call_consts[pn] = folded
        bind_call_constants(hfn, call_consts)
        arrays, scalars, symbols = infer_helper_params(
            pnames, site.call.args, site.oarr_by, site.osca_by, site.osym_by, site.owner_fn
        )
        # The helper's own compile-time tuples, against its param ranks, before the axis guards.
        rewrite_helper_axes(hfn, arrays, scalars)
        desugar_helper_tuples(hfn, arrays, scalars, symbols)
        calls = self.calls_of(site.hdef)
        if self.splice_tuple_returns(site, calls):
            return
        self.refuse_by_value_forms(site, arrays)
        # A kept helper is lowered as its own kernel, so it needs the kernel body's guards.
        reject_symbolic_axis(hfn)
        reject_unsupported_slices(hfn)
        reject_subscripted_scalar_params(hfn, scalars, site.hdef.name)
        widen_counting_scalar_params(hfn, scalars)
        mark_written_outputs(hfn, arrays)
        extra_syms = sorted(s for s in shape_symbols(arrays) if s not in set(pnames))
        if extra_syms:
            self.thread_extra_symbols(site, calls, extra_syms)
            symbols.extend(SymbolDesc(name=s) for s in extra_syms)
        returns_value = any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(hfn))
        self.out.append(
            KernelIR(
                tree=hfn,
                kernel_name=site.hdef.name,
                short_name=site.hdef.name,
                input_args=list(pnames) + extra_syms,
                symbols=symbols,
                arrays=arrays,
                scalars=scalars,
                source_path=self.parent.source_path,
                inlined_consts=site.hconsts,
                # Void when nothing is returned: a "scalar" helper gets a result slot in C/Fortran.
                return_kind="scalar" if returns_value else None,
            )
        )
        self.scopes.append((hfn, arrays, scalars, symbols))

    def calls_of(self, hdef: ast.FunctionDef) -> list[tuple[ast.FunctionDef, ast.Call]]:
        """Every call of ``hdef`` with its owning tree. An already-built sibling is represented by its
        built copy, since splicing into the original lands in a tree nothing emits."""
        built = {ir.kernel_name: ir.tree for ir in self.out}
        owners = [self.kernel_fn] + [built.get(h.name, h) for h in self.helper_defs if h is not hdef]
        return [
            (owner, n)
            for owner in owners
            for n in ast.walk(owner)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == hdef.name
        ]

    def scope_of(self, owner: ast.FunctionDef) -> tuple[list[ArrayDesc], list[ScalarDesc], list[SymbolDesc]]:
        for fn, a, sc, sy in self.scopes:
            if fn is owner:
                return a, sc, sy
        return self.parent.arrays, self.parent.scalars, self.parent.symbols

    def splice_tuple_returns(self, site: Site, calls: list[tuple[ast.FunctionDef, ast.Call]]) -> bool:
        """Splice a helper whose folded body only returns a tuple into every call site (a tuple has no
        C/Fortran return), then fold the spliced tuples in each owner. ``False`` when some call site
        does not fit the template, in which case the helper must stay a function."""
        pnames = site.pnames
        templates: dict[int, ast.expr] = {}
        if calls and all(not c.keywords and len(c.args) == len(pnames) for unused, c in calls):
            for owner, call in calls:
                oa, osc, osy = self.scope_of(owner)
                template = tuple_template_for_call(
                    site.hdef,
                    call,
                    self.tree,
                    self.parent,
                    {a.name: a for a in oa},
                    {s.name: s for s in osc},
                    {s.name: s for s in osy},
                    owner,
                )
                if template is None:
                    templates.clear()
                    break
                templates[id(call)] = template
        if not templates:
            return False
        for owner in {id(o): o for o, unused in calls}.values():
            InlineTupleHelperCalls(pnames, templates).visit(owner)
            # A built owner never passes through the loop again, so its spliced tuples fold here.
            oa, osc, osy = self.scope_of(owner)
            desugar_helper_tuples(owner, oa, osc, osy)
            reject_symbolic_axis(owner)
            reject_unsupported_slices(owner)
            self.rewrote(owner)
        return True

    @staticmethod
    def refuse_by_value_forms(site: Site, arrays: list[ArrayDesc]) -> None:
        """Refuse what a by-value helper cannot emit: a tuple return (one C return slot), or a return
        of its own allocation (the value belongs in an out-param the call site does not have)."""
        hfn, name = site.hfn, site.hdef.name
        if isinstance(site.lhs, ast.Tuple) or any(
            isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple) for n in ast.walk(hfn)
        ):
            raise NotImplementedError(
                f"helper {name!r} returns a tuple; it has no standalone ABI and must be inlined into its caller"
            )
        # The helper's own params are the scope its local allocations resolve against.
        harr_by = {a.name: a for a in arrays}
        for n in ast.walk(hfn):
            if (
                isinstance(n, ast.Return)
                and isinstance(n.value, ast.Name)
                and local_array_def(hfn, n.value.id, harr_by) is not None
            ):
                raise NotImplementedError(
                    f"helper {name!r} returns its own array {n.value.id!r} by value; the call site's "
                    f"target did not resolve to an array, so there is no out-param to write it into"
                )

    def thread_extra_symbols(
        self, site: Site, calls: list[tuple[ast.FunctionDef, ast.Call]], extra_syms: list[str]
    ) -> None:
        """Shape symbols the helper's array params name but the call does not pass become parameters,
        appended to every call. A caller can only pass a symbol it holds (its own shapes included)."""
        for owner, unused in calls:
            oa, osc, osy = self.scope_of(owner)
            held = {sy.name for sy in osy} | {a.name for a in oa} | {sc.name for sc in osc} | set(shape_symbols(oa))
            absent = [sy for sy in extra_syms if sy not in held]
            if absent:
                raise NotImplementedError(
                    f"helper {site.hdef.name!r} needs shape symbols {absent}, which its caller does not hold; "
                    f"it must be inlined into its caller"
                )
        for owner, call in calls:
            call.args.extend(ast.Name(id=s, ctx=ast.Load()) for s in extra_syms)
            self.rewrote(owner)

    def build_array_return(self, site: Site, hret_shape: list[str], hret_dtype: str | None) -> None:
        """An array-returning helper: specialised on the first site's literal arguments, unused params
        dropped, the return written into an out-param (or into the argument it already updates)."""
        hfn, hdef = site.hfn, site.hdef
        call_consts = {pn: a for pn, a in zip(site.pnames, site.call.args) if literal_call_arg(a)}
        # Also prunes what the substitution makes dead, so ``used`` below sees no dead reads.
        bind_call_constants(hfn, call_consts)
        used = {n.id for n in ast.walk(hfn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        keep = [(pn, a) for pn, a in zip(site.pnames, site.call.args) if pn in used]
        decl_pnames = list(site.pnames)
        pnames = [pn for pn, unused in keep]
        kept_args = [a for unused, a in keep]
        hfn.args.args = [a for a in hfn.args.args if a.arg in used]
        hfn.args.defaults = []
        arrays, scalars, symbols = infer_helper_params(
            pnames, kept_args, site.oarr_by, site.osca_by, site.osym_by, site.owner_fn
        )
        # Before ``hret`` joins ``arrays``: fold the helper's tuples, then the kernel-body guards.
        rewrite_helper_axes(hfn, arrays, scalars)
        desugar_helper_tuples(hfn, arrays, scalars, symbols)
        reject_symbolic_axis(hfn)
        reject_unsupported_slices(hfn)
        reject_subscripted_scalar_params(hfn, scalars, hdef.name)
        widen_counting_scalar_params(hfn, scalars)
        mark_written_outputs(hfn, arrays)
        inout_param = self.inout_param_(site, pnames, kept_args, arrays, hret_shape)
        if inout_param is not None:
            hret = inout_param
        else:
            hret = f"__hret_{site.hidx}"
            arrays.append(ArrayDesc(name=hret, dtype=hret_dtype, shape=tuple(hret_shape), is_output=True))
        # Shape symbols the array params name that no argument passes; a sibling caller must hold them.
        extra_syms = sorted(s for s in shape_symbols(arrays) if s not in set(pnames))
        if extra_syms and site.owner_fn is not self.kernel_fn:
            held = site.held_names()
            absent = [sy for sy in extra_syms if sy not in held]
            if absent:
                raise NotImplementedError(
                    f"helper {hdef.name!r} needs shape symbols {absent}, which its calling "
                    f"helper does not hold; it must be inlined into its caller"
                )
        symbols.extend(SymbolDesc(name=s) for s in extra_syms)
        rewrite_returns_to_outparam(hfn, hret)
        self.out.append(
            KernelIR(
                tree=hfn,
                kernel_name=hdef.name,
                short_name=hdef.name,
                input_args=list(pnames) + extra_syms + ([] if inout_param is not None else [hret]),
                symbols=symbols,
                arrays=arrays,
                scalars=scalars,
                source_path=self.parent.source_path,
                inlined_consts=site.hconsts,
                return_kind=hret,
            )
        )
        self.scopes.append((hfn, arrays, scalars, symbols))
        if site.assign is not None:
            spec = ArraySpec(
                decl_pnames=decl_pnames,
                pnames=pnames,
                kept_args=kept_args,
                call_consts=call_consts,
                param_info={a.name: (a.shape, a.dtype) for a in arrays if a.name != hret},
                extra_syms=extra_syms,
                hret_shape=hret_shape,
                hret_dtype=hret_dtype,
                inout=inout_param is not None,
            )
            for sidx, assign in enumerate(site.assigns):
                if isinstance(assign.value, ast.Call):
                    self.rewrite_array_call_site(site, spec, sidx, assign)

    @staticmethod
    def inout_param_(
        site: Site, pnames: list[str], kept_args: list[ast.expr], arrays: list[ArrayDesc], hret_shape: list[str]
    ) -> str | None:
        """``X = h(X, ...)``: the parameter the result is written back into, which then takes the one
        ABI slot (a second out-param would alias it under ``restrict``). Both extents must agree."""
        if not isinstance(site.lhs, ast.Name):
            return None
        lhs_id = site.lhs.id
        inout_param = next((pn for pn, a in zip(pnames, kept_args) if isinstance(a, ast.Name) and a.id == lhs_id), None)
        if inout_param is None:
            return None
        desc = next((a for a in arrays if a.name == inout_param), None)
        if desc is None or tuple(str(s) for s in desc.shape) != tuple(str(s) for s in hret_shape):
            got = tuple(desc.shape) if desc is not None else None
            raise NotImplementedError(
                f"helper {site.hdef.name!r} returns into its own argument {inout_param!r}, but the argument is "
                f"{got} and the result is {tuple(hret_shape)}; one in-out pointer cannot carry both extents "
                f"while a helper's dimensions are emitted as constants"
            )
        desc.is_output = True
        return inout_param

    def rewrite_array_call_site(self, site: Site, spec: ArraySpec, sidx: int, assign: ast.Assign) -> None:
        """Queue the caller-side statements for one ``X = h(...)`` site. The body is specialised on the
        first site's literal arguments and shapes, so a site disagreeing on either is refused."""
        name = site.hdef.name
        assert isinstance(assign.value, ast.Call)
        site_args = assign.value.args
        if len(site_args) != len(spec.decl_pnames):
            raise NotImplementedError(
                f"helper {name!r} is called with {len(site_args)} args at one "
                f"site and declares {len(spec.decl_pnames)}; the call sites disagree"
            )
        site_consts = {pn: literal_key(a) for pn, a in zip(spec.decl_pnames, site_args) if literal_call_arg(a)}
        first_consts = {pn: literal_key(a) for pn, a in spec.call_consts.items() if literal_call_arg(a)}
        if site_consts != first_consts:
            raise NotImplementedError(
                f"helper {name!r} is specialized on {first_consts} but another "
                f"call site passes {site_consts}; give the two calls their own helper"
            )
        site_kept = [a for pn, a in zip(spec.decl_pnames, site_args) if pn in spec.pnames]
        for first_a, site_a in zip(spec.kept_args, site_kept):
            if not (isinstance(first_a, ast.Name) and isinstance(site_a, ast.Name)):
                continue
            first_d, site_d = site.oarr_by.get(first_a.id), site.oarr_by.get(site_a.id)
            if first_d is not None and site_d is not None and tuple(first_d.shape) != tuple(site_d.shape):
                raise NotImplementedError(
                    f"helper {name!r} is specialized on {first_a.id}{tuple(first_d.shape)} but another "
                    f"call site passes {site_a.id}{tuple(site_d.shape)}; a helper cannot serve two shapes "
                    f"while its dimensions are emitted as constants"
                )
        # What the caller can name at this site (see caller_side_symbol); resolved per site.
        owner_held = site.held_names() | {a.arg for a in site.owner_fn.args.args} | held_before(site.owner_fn, assign)

        def respell(shape: Sequence[str]) -> list[str]:
            return caller_side_shape(shape, owner_held, spec.decl_pnames, site_args, site.hfn, name, site.oarr_by)

        extra_srcs = [
            caller_side_symbol(sy, owner_held, spec.decl_pnames, site_args, site.hfn, name, 0, site.oarr_by)
            for sy in spec.extra_syms
        ]
        site_param_info = {pn: (respell(shp), dt) for pn, (shp, dt) in spec.param_info.items()}
        self.callsite_rewrites.setdefault(id(site.owner_fn), {})[id(assign)] = build_callsite_stmts(
            assign.targets[0],
            name,
            spec.pnames,
            site_kept,
            extra_srcs,
            site_param_info,
            respell(spec.hret_shape),
            spec.hret_dtype,
            f"{site.hidx}_{sidx}" if sidx else site.hidx,
            inout=spec.inout,
            live_buffers=frozenset(a.name for a in site.oarrays),
        )

    def refuse_unemitted_callees(self) -> None:
        """A kept helper calling a sibling that got no KernelIR would call a function nothing emits."""
        emitted = {h.kernel_name for h in self.out}
        known = {h.name for h in self.helper_defs}
        for h in self.out:
            missing = sorted(
                {
                    n.func.id
                    for n in ast.walk(h.tree)
                    if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id in known
                    and n.func.id not in emitted
                }
            )
            if missing:
                raise NotImplementedError(
                    f"helper {h.kernel_name!r} calls {missing}, which are reached only from "
                    f"another helper and are not emitted as functions of their own"
                )


def build_helper_kirs(
    tree: ast.Module, kernel_fn: ast.FunctionDef, parent: KernelIR, keep_helpers: bool
) -> list[KernelIR]:
    """One :class:`KernelIR` per helper still called after inlining (see :class:`HelperKirBuilder`)."""
    return HelperKirBuilder(tree, kernel_fn, parent, keep_helpers).build()
