"""SSA-style renaming of names rebound with different shapes."""

import ast
from dataclasses import dataclass

from hpcagent_bench.translators.numpyto_common.lib_nodes.dims import shape_exprs_equal
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import iter_extent_of
from hpcagent_bench.translators.numpyto_common.lowering.shape_harvest import (
    branch_pin_,
    shapes_agree_under,
    collect_dim_aliases,
)

__all__ = [
    "SsaRenamer",
    "SsaScope",
    "apply_renames",
    "binds_name",
    "live_on_loop_reentry",
    "read_in",
    "rebinds",
    "rename_reads",
    "ssa_rename_reassigned",
]


def read_in(name: str, blocks: "tuple[list[ast.stmt], ...]") -> bool:
    """True when ``name`` is READ anywhere in ``blocks`` (the code that runs after a nested block).

    Liveness is what decides whether a shape re-binding inside an if/loop body is ambiguous. A name
    that is written and fully consumed inside the block is unambiguous no matter how many sibling
    blocks re-bind it to other shapes -- that is the ordinary "two independent temporaries in two
    loop nests" shape, and refusing it would reject working kernels.
    """
    for block in blocks:
        for stmt in block:
            for sub in ast.walk(stmt):
                if not (isinstance(sub, ast.Name) and sub.id == name):
                    continue
                # Load is the obvious read. An AugAssign target and a `del` also READ the binding
                # they act on, but both carry ctx=Store/Del -- so a whole-array `e += 1.0` was
                # neither a kill (correctly) nor a read (wrongly), and liveness answered "dead".
                if isinstance(sub.ctx, (ast.Load, ast.Del)):
                    return True
                if isinstance(sub.ctx, ast.Store) and any(
                    isinstance(anc, ast.AugAssign) and anc.target is sub for anc in ast.walk(stmt)
                ):
                    return True
    return False


def rebinds(stmt: ast.stmt, name: str) -> bool:
    """True when ``stmt`` unconditionally re-binds ``name`` at its own statement level.

    Only forms that ALWAYS execute when the statement is reached count, so a
    re-binding buried in an ``if`` does not qualify -- it does not dominate what
    follows. ``AugAssign`` is a read plus a write of the SAME buffer, so it never
    kills a binding.
    """
    if isinstance(stmt, ast.Assign):
        for tgt in stmt.targets:
            if binds_name(tgt, name):
                return True
    # A For target is NOT a kill: `for e in range(k)` with a runtime k == 0 never binds it, so the
    # previous binding survives the loop. Treating it as one dropped every read before it from the
    # liveness scan (may-define, not must-define).
    return False


def binds_name(target: ast.AST, name: str) -> bool:
    """True when an assignment TARGET binds ``name``, through any nesting.

    Covers ``x``, ``a, x = ...``, ``a, *x = ...`` and ``(a, (x, b)) = ...``. Missing a form here is
    not symmetric: an unrecognised target means the kill is not seen, the prefix is not truncated,
    extra reads are counted, and a working kernel is REFUSED.
    """
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, ast.Starred):
        return binds_name(target.value, name)
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(binds_name(el, name) for el in target.elts)
    return False


def live_on_loop_reentry(stmts: list[ast.stmt], i: int, name: str) -> "tuple[list[ast.stmt], ...]":
    """The loop-body prefix whose reads of ``name`` can see the binding made at
    ``stmts[i]``.

    Re-entering a loop body runs ``stmts[:i]`` again AFTER ``stmts[i]``, so those
    reads are part of the liveness question -- but only up to the first statement
    that re-binds ``name``. Past that point the read sees the fresh binding, not
    the one minted at ``stmts[i]``.

    Ignoring the kill is what made the refusal reject working kernels:
    daubechies_dwt2d assigns ``e`` twice per level (rows then columns) and
    ls3df_scf re-binds ``X`` twice per SCF iteration, both re-assigning at the top
    of the next iteration before any read. Both had always emitted correct code.
    """
    prefix = stmts[:i]
    for j, stmt in enumerate(prefix):
        if rebinds(stmt, name):
            # Stop AT the kill, not before it: the killing statement's own RHS still reads the old
            # binding (`X = X * 2.0`, `X, Y = Y, X`) and that read happens on re-entry. Dropping the
            # whole statement skipped it, so a genuinely ambiguous rebinding was accepted.
            return (prefix[:j], [stmt.value] if isinstance(stmt, ast.Assign) else [])
    return (prefix,)


def ssa_rename_reassigned(tree: ast.AST, arrays_shapes: dict[str, list[str]]) -> None:
    """SSA-style rename for Names reassigned with different broadcast
    extents.

    Walks every function body's statement list in source order. For
    each ``Name = expr`` whose RHS has a derivable iteration extent,
    tracks the current active extent per Name. When a reassignment
    yields a different extent, mints ``<name>__v<n>`` and rewrites
    forward Load-context references to ``<name>`` to the new version
    until the next reassignment.

    The first occurrence keeps the original name. Recurses into
    ``For`` / ``If`` / ``While`` bodies but treats each as an
    independent scope -- nested writes do not poison the outer
    version map (the outer scope's name stays bound to its outer
    extent across the nested block).

    Unblocks the canonical hdiff / vadv kernels where ``res`` is
    reassigned twice with different shapes; without renaming the
    ``LiftFreshArrayFromSlices`` lifter bails on the shape mismatch.
    """
    renamer = SsaRenamer(arrays_shapes, collect_dim_aliases(tree, set(arrays_shapes)))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Per function, like ``version``: a helper's locals are not the kernel's.
            renamer.shape_toks_of.clear()
            renamer.shape_rank.clear()
            renamer.walk(node.body, {}, {}, {}, SsaScope())


@dataclass(frozen=True, slots=True)
class SsaScope:
    """Where a statement list sits: ``nested`` inside if/loop bodies; ``live_after`` the code that
    runs after it; ``loop_body`` a loop body (whose earlier statements run again after each one);
    ``reentry`` every enclosing loop's re-entry point; ``pin`` the scalars an enclosing branch fixes
    and ``general_side`` whether this is the side holding the general spelling of an extent."""

    nested: bool = False
    live_after: tuple[list[ast.stmt], ...] = ()
    loop_body: bool = False
    reentry: tuple[tuple[list[ast.stmt], int], ...] = ()
    pin: dict[str, int] | None = None
    general_side: bool = True


class SsaRenamer:
    """The state of one :func:`ssa_rename_reassigned` walk.

    A single function-scope rename_map / shape map -- Python does not have block scope for
    assignments, so a ``bcol = ...`` inside sibling for-loops at function scope is the SAME local
    being reassigned; sharing ``version`` across nested scopes mints a fresh version for each shape
    change even when the assignments live in different loop bodies.
    """

    __slots__ = ("dim_aliases", "shape_rank", "shape_toks_of", "shapes")

    def __init__(self, arrays_shapes: dict[str, list[str]], dim_aliases: dict[str, str]) -> None:
        self.shapes: dict[str, tuple[str, ...]] = {name: tuple(shape) for name, shape in arrays_shapes.items()}
        self.dim_aliases = dim_aliases
        #: Which shape each buffer is currently DECLARED with, and how general each recorded shape is.
        self.shape_toks_of: dict[str, tuple[str, ...]] = {}
        self.shape_rank: dict[str, dict[tuple[str, ...], int]] = {}

    def register_alloc(self, target_id: str, rhs: ast.AST) -> None:
        """Register the shape of ``np.zeros((...))`` / ``np.empty((...))``
        style allocators so subsequent reads see the allocated extent
        when the SSA pass computes broadcast extents inside loop
        bodies. Conservative -- only handles the Tuple-shape form."""
        if not isinstance(rhs, ast.Call):
            return
        func = rhs.func
        attr = None
        if isinstance(func, ast.Attribute):
            attr = func.attr
        elif isinstance(func, ast.Name):
            attr = func.id
        if attr not in {"zeros", "empty", "ones", "ndarray", "zeros_like", "empty_like", "ones_like"}:
            return
        if attr.endswith("_like") and rhs.args and isinstance(rhs.args[0], ast.Name):
            src = self.shapes.get(rhs.args[0].id)
            if src:
                self.shapes[target_id] = src
            return
        if not rhs.args:
            return
        sh = rhs.args[0]
        if isinstance(sh, (ast.Tuple, ast.List)):
            self.shapes[target_id] = tuple(ast.unparse(e) for e in sh.elts)
        elif isinstance(sh, ast.Constant) and isinstance(sh.value, int):
            self.shapes[target_id] = (str(sh.value),)
        elif isinstance(sh, ast.Name):
            self.shapes[target_id] = (sh.id,)

    def walk(
        self,
        stmts: list[ast.stmt],
        rename_map: dict[str, str],
        last_shape: dict[str, tuple[str, ...]],
        version: dict[str, dict[tuple[str, ...], str]],
        scope: SsaScope,
    ) -> None:
        for i, stmt in enumerate(stmts):
            # Rewrite Load-context Names on the RHS / iter / test BEFORE the version-mint decision
            # (the assignment's RHS reads the old version's storage).
            rename_reads(stmt, rename_map)
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                self.rebind(stmts, i, rename_map, last_shape, version, scope)
            if isinstance(stmt, (ast.For, ast.If, ast.While)):
                self.walk_branches(stmts, i, rename_map, last_shape, version, scope)

    def rebind(
        self,
        stmts: list[ast.stmt],
        i: int,
        rename_map: dict[str, str],
        last_shape: dict[str, tuple[str, ...]],
        version: dict[str, dict[tuple[str, ...], str]],
        scope: SsaScope,
    ) -> None:
        """Decide the name a plain ``Name = expr`` binds: the original, the version already holding
        this shape, or a freshly minted ``<name>__v<n>``."""
        stmt = stmts[i]
        orig = stmt.targets[0].id
        # Allocator-style RHS -- register the allocated shape so later reads inside loop bodies
        # resolve their extents. Allocations themselves don't trigger a rename.
        self.register_alloc(orig, stmt.value)
        ext = iter_extent_of(stmt.value, self.shapes)
        if ext is None:
            return
        shape_toks = tuple(ast.unparse(e) for e in ext)
        # ``version[orig]`` is ``{shape_tuple -> active_name}``: each distinct shape gets its own
        # active name, reused by every later assignment with that shape so sibling-loop
        # reassignments stay on the same buffer.
        if not isinstance(version.get(orig), dict):
            version[orig] = {}
        # A binding on the side where the pinned scalar is NOT zero holds the general spelling of
        # the extent; the other side, and any binding outside the branch, is the special case it
        # collapses to. Whichever is general is what the buffer must be DECLARED with -- the other
        # one only has to be reachable under the pin.
        rank = 1 if (scope.pin and scope.general_side) else 0
        name_for_shape = version[orig].get(shape_toks)
        merged_under_pin = False
        if name_for_shape is None:
            name_for_shape, merged_under_pin = self.equal_extent_version(version[orig], shape_toks, scope.pin)
            if name_for_shape is not None:
                version[orig][shape_toks] = name_for_shape
        if name_for_shape is None:
            name_for_shape = self.mint(stmts, i, orig, version[orig], shape_toks, scope)
            version[orig][shape_toks] = name_for_shape
        if name_for_shape != orig:
            stmt.targets[0].id = name_for_shape
            rename_map[orig] = name_for_shape
        else:
            rename_map.pop(orig, None)
        last_shape[orig] = shape_toks
        if merged_under_pin:
            if rank > self.shape_rank.setdefault(orig, {}).get(self.shape_toks_of[name_for_shape], 0):
                self.shapes[name_for_shape] = shape_toks
                self.shape_toks_of[name_for_shape] = shape_toks
                self.shape_rank[orig][shape_toks] = rank
        else:
            self.shapes[name_for_shape] = shape_toks
            self.shape_toks_of[name_for_shape] = shape_toks
            self.shape_rank.setdefault(orig, {})[shape_toks] = rank

    def equal_extent_version(
        self, versions: dict[tuple[str, ...], str], shape_toks: tuple[str, ...], pin: dict[str, int] | None
    ) -> tuple[str | None, bool]:
        """The version already bound to an EQUAL extent under another spelling, and whether it was only
        equal under the enclosing branch's pin.

        The key is the extent TEXT, so two spellings of one extent -- ``N`` against the ``R + N - r -
        (R - r)`` a slice pair unparses to -- look like a second shape. Extent equality is asked the
        way the rest of the lowering asks it; unresolvable answers False, so a genuine second shape
        is never merged onto one buffer. ``padded`` is ``(n, c, h + 2*padding, w + 2*padding)`` in the
        padding branch and ``(n, c, h, w)`` in the else -- two shapes only until the branch's own
        ``padding == 0`` is applied, exactly the condition under which the else binding can run."""
        for known_toks, known_name in tuple(versions.items()):
            if len(known_toks) != len(shape_toks):
                continue
            if all(a == b or shape_exprs_equal(a, b) for a, b in zip(known_toks, shape_toks)):
                return known_name, False
            if pin and shapes_agree_under(known_toks, shape_toks, pin, self.dim_aliases, self.shapes):
                return known_name, True
        return None, False

    def mint(
        self,
        stmts: list[ast.stmt],
        i: int,
        orig: str,
        versions: dict[tuple[str, ...], str],
        shape_toks: tuple[str, ...],
        scope: SsaScope,
    ) -> str:
        """The name for a NEW shape of ``orig``: the original name on first occurrence, else
        ``<orig>__v<n>``.

        Minting a second buffer inside an if/loop body is only a problem when the name is LIVE AFTER
        that body: the rename cannot survive the block, so a later read would bind to a buffer the
        untaken path never wrote. Inside a loop body the statements BEFORE this one also run after
        it, on the next iteration, and every enclosing loop re-enters, not just the innermost."""
        if not versions:
            return orig
        after = scope.live_after
        for blk, idx in scope.reentry:
            after = after + live_on_loop_reentry(blk, idx, orig)
        if scope.loop_body:
            after = after + live_on_loop_reentry(stmts, i, orig)
        if scope.nested and read_in(orig, after):
            raise NotImplementedError(
                f"{orig!r} (line {stmts[i].lineno}) is re-bound to a different shape "
                f"inside conditional control flow and read again afterwards; "
                f"which buffer that read sees is not decidable statically. Hoist "
                f"the re-binding to function scope, or give the two shapes "
                f"separate names. Bound as {list(versions)}, now {shape_toks}."
            )
        return f"{orig}__v{len(versions)}"

    def walk_branches(
        self,
        stmts: list[ast.stmt],
        i: int,
        rename_map: dict[str, str],
        last_shape: dict[str, tuple[str, ...]],
        version: dict[str, dict[tuple[str, ...], str]],
        scope: SsaScope,
    ) -> None:
        """Recurse into a For / If / While's body and orelse with COPIED maps, so a rename minted
        inside does not stay active after the block closes. What executes after the body is
        everything later in this block plus whatever follows the enclosing blocks; a While re-tests
        its condition and both loop forms may run an ``else`` after the body. The enclosing loops'
        RE-ENTRY POINTS are carried down (not a pre-truncated prefix: the truncation depends on the
        name being minted, known only at the mint site)."""
        stmt = stmts[i]
        inner_after = (stmts[i + 1 :],) + scope.live_after
        if isinstance(stmt, ast.While):
            inner_after = ([ast.Expr(value=stmt.test)],) + inner_after
        if stmt.orelse and isinstance(stmt, (ast.For, ast.While)):
            inner_after = (stmt.orelse,) + inner_after
        is_loop = isinstance(stmt, (ast.For, ast.While))
        inner_reentry = (scope.reentry + ((stmts, i),)) if scope.loop_body else scope.reentry
        branch_pin, zero_on_taken = branch_pin_(stmt)
        for branch, in_loop, taken in ((stmt.body, is_loop, True), (stmt.orelse, False, False)):
            inner_scope = SsaScope(
                nested=True,
                live_after=inner_after,
                loop_body=in_loop,
                reentry=inner_reentry,
                pin={**(scope.pin or {}), **branch_pin},
                general_side=bool(branch_pin) and taken is not zero_on_taken,
            )
            self.walk(branch, dict(rename_map), dict(last_shape), version, inner_scope)


def rename_reads(stmt: ast.stmt, rename_map: dict[str, str]) -> None:
    """Rename the Load-context Names ``stmt`` reads before it binds anything: an assignment's value
    (and an AugAssign's target), an if / while test, a for's iter, an expression, a return value.
    A plain ``arr[idx] = val`` / ``obj.attr = val`` target carries the buffer's base Name in Load
    context and is renamed too, so a fill that follows a reassignment writes the new version; the
    Store-context Name of a ``name = ...`` target is left to the version-mint decision."""
    if isinstance(stmt, (ast.Assign, ast.AugAssign)):
        apply_renames(stmt.value, rename_map)
        if isinstance(stmt, ast.AugAssign):
            apply_renames(stmt.target, rename_map)
        else:
            for tgt in stmt.targets:
                if not isinstance(tgt, ast.Name):
                    apply_renames(tgt, rename_map)
    elif isinstance(stmt, (ast.If, ast.While)):
        apply_renames(stmt.test, rename_map)
    elif isinstance(stmt, ast.For):
        apply_renames(stmt.iter, rename_map)
    elif isinstance(stmt, ast.Expr):
        apply_renames(stmt.value, rename_map)
    elif isinstance(stmt, ast.Return):
        if stmt.value is not None:
            apply_renames(stmt.value, rename_map)


def apply_renames(node: ast.AST, rename_map: dict[str, str]) -> None:
    if not rename_map:
        return
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id in rename_map:
            sub.id = rename_map[sub.id]
