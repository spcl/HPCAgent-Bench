"""The lowered kernel body rewritten for ONE cooperative GPU launch (the uniform execution model).

Every thread of the grid runs the whole body. Control flow and scalar statements stay UNIFORM: each thread computes
the same scalars from the same values. Three statement kinds leave that rule:

* a loop the numba translator proves parallel (:func:`~hpcagent_bench.translators.numpyto_numba.prange.
  outermost_safe_loops`) is distributed over the grid, one iteration per thread with a grid-stride loop; a
  perfectly nested rectangular map nest is linearized first (:func:`~hpcagent_bench.translators.numpyto_common.
  parallelism.collapsible_depth`), and a reduction combines per-thread partials through a block tree and one atomic
  per block into a frame slot;
* any other statement that writes an array, or any compound statement without a parallel loop inside, runs on the
  lead thread only, and the scalars it assigns are broadcast through the frame afterwards;
* a grid barrier closes every such write epoch and opens it, so no uniform read races a write.

Local arrays (``np.zeros`` temporaries) become kernel parameters the host entry allocates in device memory; their
fills become parallel loop nests at the points the C emitter fills them. What this model cannot express is refused
with :class:`GpuUnsupported`."""

import ast
import dataclasses

from hpcagent_bench.translators.numpyto_c.emit import LocalGroups, collect_implicit_locals, default_float_dtype
from hpcagent_bench.translators.numpyto_common.ast_build import walk_nodes
from hpcagent_bench.translators.numpyto_common.ir import COMPLEX_DTYPES, ArrayDesc, KernelIR
from hpcagent_bench.translators.numpyto_common.parallelism import (
    collapsible_depth,
    loop_index,
    loop_is_parallel_safe,
    loop_reduction,
    range_args,
    scalars_live_after,
)
from hpcagent_bench.translators.numpyto_numba.prange import (
    numba_reduction,
    outermost_safe_loops,
    private_scalars,
    statement_blocks,
    unit_step,
    without_updates,
    written_parameters,
)

__all__ = [
    "ALLOC_MARKER",
    "FRAME",
    "GPU_NAMES",
    "GpuUnsupported",
    "Kernelized",
    "REDUCTION_NAMES",
    "SYNC",
    "kernelize",
]

#: The lowering's allocation marker (``t = __hpcagent_bench_zeros__(...)``).
ALLOC_MARKER = "__hpcagent_bench_zeros__"
#: Names the GPU prelude defines and the rewritten body reads: the grid barrier, the frame, the thread ids.
SYNC = "__npb_sync"
FRAME = "__npb_frame"
LEAD = "__npb_lead"
GTID = "__npb_gtid"
GSIZE = "__npb_gsize"
SEED = "__npb_seed"
LOAD = "__npb_load"
GPU_NAMES = frozenset({SYNC, FRAME, LEAD, GTID, GSIZE, SEED, LOAD})
#: Reduction operator (as :func:`loop_reduction` spells it) -> the prelude's combine and the partial's start value
#: (``None``: an idempotent operator starts from the accumulator's own value).
REDUCTION_NAMES: dict[str, tuple[str, int | None]] = {
    "+": ("__npb_combine_add", 0),
    "*": ("__npb_combine_mul", 1),
    "max": ("__npb_combine_max", None),
    "min": ("__npb_combine_min", None),
}


#: Constructor kinds that leave a local's contents undefined: no fill.
SCRATCH_KINDS = frozenset({"empty", "empty_like", "ndarray"})


def fill_value(kind: str) -> int:
    """The element a ``kind`` constructor fills with (``zero_fill_stmt``'s rule: ones -> 1, else 0)."""
    return 1 if kind in ("ones", "ones_like") else 0


class GpuUnsupported(NotImplementedError):
    """A kernel construct the GPU translator does not emit: the refusal names it."""


@dataclasses.dataclass(frozen=True, slots=True)
class Kernelized:
    """What the host entry needs from the rewrite: the hoisted locals it allocates, whether the body has a parallel
    loop (else one thread runs it), and the frame slots the body uses."""

    hoisted: tuple[ArrayDesc, ...]
    parallel: bool
    frame_slots: int


def call(name: str, *args: ast.expr) -> ast.Call:
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=list(args), keywords=[])


def stmt_call(name: str, *args: ast.expr) -> ast.Expr:
    return ast.Expr(value=call(name, *args))


def load(name: str) -> ast.Name:
    return ast.Name(id=name, ctx=ast.Load())


def store(name: str) -> ast.Name:
    return ast.Name(id=name, ctx=ast.Store())


def is_call_to(stmt: ast.stmt, name: str) -> bool:
    """``stmt`` is the bare call statement ``name(...)``."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name)
        and stmt.value.func.id == name
    )


def is_docstring(stmt: ast.stmt) -> bool:
    """A bare constant expression: no code."""
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)


def is_sync(stmt: ast.stmt) -> bool:
    return is_call_to(stmt, SYNC)


def marker_of(stmt: ast.stmt) -> ast.Call | None:
    """The allocation marker call ``stmt`` is (as an assignment or a bare expression), else ``None``."""
    value = stmt.value if isinstance(stmt, (ast.Assign, ast.Expr)) else None
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == ALLOC_MARKER:
        return value
    return None


def is_reassign(marker: ast.Call) -> bool:
    """A ``__reassign__`` marker: the local is rebound to values computed from its own, never refilled."""
    return bool(marker.args) and isinstance(marker.args[0], ast.Constant) and marker.args[0].value == "__reassign__"


def range_of(loop: ast.For) -> tuple[ast.expr, ast.expr]:
    """``(lo, hi)`` of a unit-step ``range`` loop."""
    args = range_args(loop)
    return (ast.Constant(0), args[0]) if len(args) == 1 else (args[0], args[1])


def fill_nest(desc: ArrayDesc, value: int, index: int) -> ast.For:
    """``desc[...] = value`` over every element: a perfect rectangular map nest the rewrite distributes."""
    names = [f"__npb_f{index}_{axis}" for axis in range(len(desc.shape))]
    constant: ast.expr = ast.Constant(float(value) if desc.dtype.startswith(("float", "complex")) else value)
    target = ast.Subscript(
        value=load(desc.name),
        slice=ast.Tuple(elts=[load(n) for n in names], ctx=ast.Load()) if len(names) > 1 else load(names[0]),
        ctx=ast.Store(),
    )
    nest: ast.stmt = ast.Assign(targets=[target], value=constant)
    for name, extent in reversed(list(zip(names, desc.shape, strict=True))):
        bound = ast.parse(str(extent), mode="eval").body
        nest = ast.For(target=store(name), iter=call("range", bound), body=[nest], orelse=[])
    assert isinstance(nest, ast.For)
    return nest


def refuse_unsupported_types(kir: KernelIR) -> None:
    """Complex values (no device ``std::complex`` in the frame or the atomics) and kept helpers are refused."""
    if kir.helpers:
        raise GpuUnsupported(f"{kir.kernel_name}: a helper that cannot be inlined ({kir.helpers[0].kernel_name})")
    typed = [a.dtype for a in kir.arrays] + [s.dtype for s in kir.scalars] + list(kir.local_dtypes.values())
    if any(dtype in COMPLEX_DTYPES for dtype in typed):
        raise GpuUnsupported(f"{kir.kernel_name}: complex values")


def hoist_locals(kir: KernelIR) -> tuple[ArrayDesc, ...]:
    """Turn every local array into a kernel parameter the host allocates, and its fills into loop nests.

    A local sized by a loop index or by a body-computed scalar has no size the host knows before the launch, and a
    reassigned local whose extent changes has no single size: both are refused. The C emitter fills a ``zeros`` /
    ``ones`` local at function top and again at each of its markers but a ``__reassign__``, and an output parameter
    an ``np.zeros`` aliases once at function top; the nests land at those same points."""
    if not kir.zeros_locals:
        return ()
    groups = LocalGroups.of(kir, collect_implicit_locals(kir))
    if groups.inline or groups.deferred:
        name = next(iter({**groups.inline, **groups.deferred}))
        raise GpuUnsupported(f"{kir.kernel_name}: local array {name!r} is sized by a loop index or a computed value")
    for name, shapes in kir.reassign_shapes.items():
        if any(tuple(shape) != tuple(kir.zeros_locals.get(name, shape)) for shape in shapes):
            raise GpuUnsupported(f"{kir.kernel_name}: local array {name!r} changes extent")
    default_float = default_float_dtype(kir)
    hoisted = tuple(
        ArrayDesc(name, kir.local_dtypes.get(name, default_float), tuple(shape) or ("1",), is_output=True)
        for name, shape in groups.fn_top.items()
    )
    params = {a.name: a for a in kir.arrays}
    # (desc, fill value) at function top; a hoisted local is refilled at each of its markers too.
    top_fills: dict[str, tuple[ArrayDesc, int]] = {}
    for name in groups.param_inits:
        kind = kir.zeros_fills.get(name, "zeros")
        if name in params and kind not in SCRATCH_KINDS:
            top_fills[name] = (params[name], fill_value(kind))
    for desc in hoisted:
        kind = kir.zeros_fills.get(desc.name)
        if kind is not None and kind not in SCRATCH_KINDS:
            top_fills[desc.name] = (desc, fill_value(kind))
    refills = {d.name for d in hoisted} & set(top_fills)
    counter = iter(range(1 << 30))

    def rewrite(stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in stmts:
            marker = marker_of(stmt)
            if marker is None:
                for block in statement_blocks(stmt):
                    block[:] = rewrite(block)
                out.append(stmt)
                continue
            target = stmt.targets[0] if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 else None
            if isinstance(target, ast.Name) and target.id in refills and not is_reassign(marker):
                out.append(fill_nest(*top_fills[target.id], next(counter)))
        return out

    body = rewrite(kir.tree.body)
    top = [fill_nest(desc, value, next(counter)) for desc, value in top_fills.values()]
    kir.tree.body = top + body
    kir.arrays.extend(hoisted)
    kir.tree.args.args.extend(ast.arg(arg=d.name) for d in hoisted)
    kir.zeros_locals = {}
    kir.reassign_shapes = {}
    return hoisted


def escapes(stmt: ast.stmt) -> bool:
    """Whether ``stmt`` holds a ``return``, or a ``break`` / ``continue`` of a loop enclosing it: control that must
    stay uniform, so ``stmt`` cannot run on the lead thread alone."""

    def walk(stmts: list[ast.stmt], in_own_loop: bool) -> bool:
        for one in stmts:
            if isinstance(one, ast.Return) or (isinstance(one, (ast.Break, ast.Continue)) and not in_own_loop):
                return True
            inner = in_own_loop or isinstance(one, (ast.For, ast.While))
            if any(walk(block, inner) for block in statement_blocks(one)):
                return True
        return False

    return walk([stmt], False)


def assigned_names(stmts: list[ast.stmt]) -> list[str]:
    """Names ``stmts`` bind by assignment (not loop targets, which C scopes to their loop), in first-bind order."""
    seen: dict[str, None] = {}
    for stmt in stmts:
        for node in walk_nodes(stmt):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, (ast.AugAssign, ast.AnnAssign))
                else []
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    seen.setdefault(target.id, None)
    return list(seen)


@dataclasses.dataclass(frozen=True, slots=True)
class Reduction:
    """A proven loop that is a reduction: its accumulator and :func:`loop_reduction`'s operator."""

    acc: str
    op: str


def loop_kind(loop: ast.For, fn: ast.FunctionDef, kernel: str) -> Reduction | None:
    """``None`` for a proven loop whose iterations are independent, else its :class:`Reduction` -- the two cases
    :func:`~hpcagent_bench.translators.numpyto_numba.prange.prange_safe` accepts, told apart its own way."""
    if not scalars_live_after(fn, loop) and loop_is_parallel_safe(without_updates(loop, private_scalars(loop))):
        return None
    acc, found = numba_reduction(loop, fn), loop_reduction(loop)
    if acc is None or found is None or found[0] not in REDUCTION_NAMES:
        raise GpuUnsupported(f"{kernel}: proven loop over {loop_index(loop)!r} has no GPU form")
    return Reduction(acc, found[0])


@dataclasses.dataclass(slots=True)
class Rewriter:
    """The uniform-model rewrite of one kernel body (see the module docstring)."""

    kir: KernelIR
    #: id() of each loop the prange analysis proved -> its kind (:func:`loop_kind`).
    chosen: dict[int, Reduction | None]
    slots: int = 0
    sites: int = 0

    def block(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        """``stmts`` rewritten; consecutive lead-only statements share one epoch."""
        out: list[ast.stmt] = []
        pending: list[ast.stmt] = []
        for stmt in stmts:
            if self.must_stay_uniform(stmt):
                self.flush(pending, out)
                out += self.uniform(stmt, out)
            elif pending or self.lead_only(stmt):
                pending.append(stmt)
            else:
                out.append(stmt)
        self.flush(pending, out)
        return out

    def must_stay_uniform(self, stmt: ast.stmt) -> bool:
        """Control every thread takes: a jump, a parallel loop or a statement holding one, a statement leaving
        itself."""
        if isinstance(stmt, (ast.Break, ast.Continue, ast.Return, ast.Pass)):
            return True
        if not statement_blocks(stmt):
            return False
        return any(id(node) in self.chosen for node in walk_nodes(stmt)) or escapes(stmt)

    @staticmethod
    def lead_only(stmt: ast.stmt) -> bool:
        """What only the lead thread runs: a compound statement, an array store, a call statement. A scalar
        assignment runs on every thread."""
        if isinstance(stmt, ast.Expr):
            return not is_docstring(stmt)
        if statement_blocks(stmt):
            return True
        if isinstance(stmt, ast.Assign):
            return not all(isinstance(t, ast.Name) for t in stmt.targets)
        if isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
            return not isinstance(stmt.target, ast.Name)
        return True

    @staticmethod
    def open_epoch(out: list[ast.stmt]) -> None:
        if not (out and is_sync(out[-1])):
            out.append(stmt_call(SYNC))

    def flush(self, pending: list[ast.stmt], out: list[ast.stmt]) -> None:
        """Run ``pending`` on the lead thread, then broadcast the scalars it assigned."""
        if not pending:
            return
        names = assigned_names(pending)
        self.slots = max(self.slots, len(names))
        self.open_epoch(out)
        seeds = [stmt_call(SEED, load(FRAME), ast.Constant(k), load(n)) for k, n in enumerate(names)]
        out.append(ast.If(test=load(LEAD), body=[*pending, *seeds], orelse=[]))
        out.append(stmt_call(SYNC))
        out += [stmt_call(LOAD, load(FRAME), ast.Constant(k), load(n)) for k, n in enumerate(names)]
        pending.clear()

    def uniform(self, stmt: ast.stmt, out: list[ast.stmt]) -> list[ast.stmt]:
        if isinstance(stmt, ast.For) and id(stmt) in self.chosen:
            return self.parallel(stmt, self.chosen[id(stmt)], out)
        for name in ("body", "orelse"):
            value = getattr(stmt, name, None)
            if isinstance(value, list) and value:
                setattr(stmt, name, self.block(value))
        return [stmt]

    def parallel(self, loop: ast.For, reduction: Reduction | None, out: list[ast.stmt]) -> list[ast.stmt]:
        """The proven loop distributed over the grid, between two barriers."""
        self.open_epoch(out)
        if reduction is None:
            return [*self.map_nest(loop), stmt_call(SYNC)]
        combine, start = REDUCTION_NAMES[reduction.op]
        acc = reduction.acc
        self.slots = max(self.slots, 1)
        lo, hi = range_of(loop)
        strided = ast.For(target=loop.target, iter=grid_range(lo, hi), body=loop.body, orelse=[])
        reset = [] if start is None else [ast.Assign(targets=[store(acc)], value=ast.Constant(start))]
        return [
            stmt_call(SEED, load(FRAME), ast.Constant(0), load(acc)),
            stmt_call(SYNC),
            *reset,
            strided,
            stmt_call(combine, load(FRAME), ast.Constant(0), load(acc)),
            stmt_call(SYNC),
            stmt_call(LOAD, load(FRAME), ast.Constant(0), load(acc)),
        ]

    def map_nest(self, loop: ast.For) -> list[ast.stmt]:
        """``loop`` and the rectangular unit-step levels :func:`collapsible_depth` folds in, as one grid-stride loop
        over their linearized space, the innermost index fastest so neighbouring threads touch neighbouring
        elements; the extents are bound first."""
        nest = [loop]
        for unused in range(collapsible_depth(loop) - 1):
            inner = nest[-1].body[0]
            if not (isinstance(inner, ast.For) and isinstance(inner.iter, ast.Call) and unit_step(inner.iter)):
                break
            nest.append(inner)
        if len(nest) == 1:
            lo, hi = range_of(loop)
            return [ast.For(target=loop.target, iter=grid_range(lo, hi), body=loop.body, orelse=[])]
        site = self.sites
        self.sites += 1
        flat = f"__npb_k{site}"
        extents = [f"__npb_e{site}_{level}" for level in range(len(nest))]
        bind: list[ast.stmt] = []
        for name, level in zip(extents, nest, strict=True):
            lo, hi = range_of(level)
            span = ast.BinOp(left=hi, op=ast.Sub(), right=lo)
            bind.append(ast.Assign(targets=[store(name)], value=call("max", span, ast.Constant(0))))
        total: ast.expr = load(extents[0])
        for name in extents[1:]:
            total = ast.BinOp(left=total, op=ast.Mult(), right=load(name))
        unpack: list[ast.stmt] = []
        rest: ast.expr = load(flat)
        for name, level in reversed(list(zip(extents, nest, strict=True))):
            lo, unused_hi = range_of(level)
            here = rest if level is nest[0] else ast.BinOp(left=rest, op=ast.Mod(), right=load(name))
            unpack.insert(0, ast.Assign(targets=[store(loop_index(level))], value=ast.BinOp(left=lo, op=ast.Add(), right=here)))
            rest = ast.BinOp(left=rest, op=ast.FloorDiv(), right=load(name))
        declared = set(self.kir.int_locals)
        self.kir.int_locals.extend(n for n in [*extents, *map(loop_index, nest)] if n not in declared)
        strided = ast.For(
            target=store(flat), iter=call("range", load(GTID), total, load(GSIZE)), body=[*unpack, *nest[-1].body], orelse=[]
        )
        return [*bind, strided]


def grid_range(lo: ast.expr, hi: ast.expr) -> ast.Call:
    """``range(lo + gtid, hi, gsize)``: this thread's share of ``range(lo, hi)``."""
    return call("range", ast.BinOp(left=lo, op=ast.Add(), right=load(GTID)), hi, load(GSIZE))


def kernelize(kir: KernelIR) -> Kernelized:
    """Rewrite ``kir`` (lowered, helpers inlined, C-spelled) in place for the uniform model; see the module
    docstring. Raises :class:`GpuUnsupported` for what the model cannot express."""
    refuse_unsupported_types(kir)
    hoisted = hoist_locals(kir)
    fn = kir.tree
    chosen: list[ast.For] = []
    if any(isinstance(node, ast.For) for node in walk_nodes(fn)):
        ranks = {a.name: len(a.shape) for a in kir.arrays}
        mutates, params = written_parameters(ast.Module(body=[fn], type_ignores=[]))
        chosen = outermost_safe_loops(fn.body, fn, ranks, mutates, params)
    rewriter = Rewriter(kir, {id(loop): loop_kind(loop, fn, kir.kernel_name) for loop in chosen})
    body = rewriter.block(fn.body)
    # The launch is the kernel's first barrier and its end the last: a barrier or a broadcast load at either end of
    # the body orders nothing.
    while body and (is_sync(body[-1]) or is_call_to(body[-1], LOAD)):
        body.pop()
    first = next((k for k, stmt in enumerate(body) if not is_docstring(stmt)), None)
    if first is not None and is_sync(body[first]):
        body.pop(first)
    fn.body = body or [ast.Pass()]
    ast.fix_missing_locations(fn)
    return Kernelized(hoisted=hoisted, parallel=bool(chosen), frame_slots=max(rewriter.slots, 1))
