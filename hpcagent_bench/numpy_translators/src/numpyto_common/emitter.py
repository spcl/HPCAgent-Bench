"""Shared statement-dispatch skeleton for the imperative body emitters.

Only the genuinely target-agnostic part of the walk lives here: ``emit_block``
and the ``emit_stmt`` dispatch (For / While / If / Assign / AugAssign / Expr /
Break / Continue / Pass / Return). The leaves that differ per target are small
hooks (statement terminator, break / continue keyword, return handling).

Also shared: the ``emit_expr`` wrapper that re-rounds fp8 results and re-wraps narrow
integers (targets supply the fp8 function names and the narrowing spelling), and the
tuple-target splitter both imperative backends run before emitting.

The body of each statement/expression *form* (loops, subscripts, calls, the
type system) is legitimately language-specific -- C flattens N-D subscripts and
runs an int-typing pass, Fortran is 1-based / column-major with kind inference,
their indent step differs (2 vs 4 spaces), control flow is braces vs
``do/end do`` -- so those stay overridden in the subclass rather than forced
through a leaky hook surface. A subclass is free to override ``emit_stmt``
wholesale if a target ever needs a different dispatch.
"""

import ast
import copy
from collections.abc import Mapping
from functools import cached_property
from typing import List, NamedTuple, Optional, Sequence

from numpyto_common import dtypes, narrow_int
from numpyto_common.ir import KernelIR, numpy_origin


def index_rank_error(name: str, shape: Optional[Sequence[str]], n_indices: int) -> str:
    """The one diagnostic both backends raise for an index the target cannot express.

    WHEN to raise it is language-specific and stays in the backend: C flattens row-major onto a
    flat pointer, so the axis count must MATCH the declared rank; Fortran emits a genuine
    multidimensional reference, where fewer axes is a valid array section but more is not. WHAT
    to say is not -- both mean the array's rank is unknown or disagrees with the source, almost
    always a missing ``init.shapes`` declaration or a numpy construct with no static rank (a
    boolean-mask gather). Neither may emit anyway: the result does not compile.
    """
    rank = "unknown" if shape is None else list(shape)
    return (
        f"cannot index {name!r} with {n_indices} axes: its shape is {rank} "
        f"(rank {0 if shape is None else len(shape)}). "
        f"Declare init.shapes[{name!r}] with the matching rank."
    )


class Fp8Fns(NamedTuple):
    """The three prelude entry points for one fp8 format."""

    promote: str  # storage byte -> float
    demote: str  # float -> storage byte
    round: str  # float -> float, rounded to the fp8 grid


#: BinOp ops that are never fp8 arithmetic (bit/shift work is integer); the fp8 round-to-grid wrap skips them.
FP8_NON_ARITH_OPS = (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)


def fp8_function_names(prefix: str) -> dict[str, Fp8Fns]:
    """The fp8 prelude function names per canonical registry dtype, each spelled ``<prefix><function>``."""
    return {
        f"float8_{fmt}": Fp8Fns(f"{prefix}{fmt}_to_f32", f"{prefix}f32_to_{fmt}", f"{prefix}rn_{fmt}")
        for fmt in ("e4m3", "e5m2")
    }


def fp8_functions(dtype: str, names: Mapping[str, Fp8Fns]) -> Fp8Fns | None:
    """:class:`Fp8Fns` for a storage-only (fp8) dtype, else None (gated on the registry)."""
    if not dtype or not dtypes.is_storage_only(dtype):
        return None
    return names[dtypes.canonical(dtype)]


def fp8_dtypes_used(kir: KernelIR) -> list[str]:
    """The canonical storage-only (fp8) dtypes this kernel mentions, deduped; drives prelude injection + promote/demote."""
    seen: list[str] = []
    for dt in (
        *(a.dtype for a in kir.arrays),
        *(s.dtype for s in kir.scalars),
        *kir.local_dtypes.values(),
        kir.float_precision or "",
    ):
        if dt and dtypes.is_storage_only(dt):
            canon = dtypes.canonical(dt)
            if canon not in seen:
                seen.append(canon)
    return seen


def tuple_element(node: ast.AST, i: int, n: int) -> ast.expr | None:
    """Element ``i`` of an ``n``-wide tuple-valued expression, or None when it is not one.

    A conditional over tuples is projected by pushing the index through it, so the guards are
    duplicated and the tuples disappear.
    """
    if isinstance(node, ast.Tuple):
        return copy.deepcopy(node.elts[i]) if len(node.elts) == n else None
    if isinstance(node, ast.IfExp):
        body = tuple_element(node.body, i, n)
        orelse = tuple_element(node.orelse, i, n)
        if body is None or orelse is None:
            return None
        return ast.IfExp(test=copy.deepcopy(node.test), body=body, orelse=orelse)
    return None


class TupleTargetSplitter(ast.NodeTransformer):
    """Rewrite ``a, b, c = <tuple-valued expr>`` into one scalar assignment per element.

    Neither C nor Fortran has a tuple. The frontend splices a tuple-returning helper into its call
    site as a SINGLE expression -- a conditional selecting between tuple literals -- which the
    lowering splitter (matching a bare tuple RHS) leaves alone. Every element repeats the guards.

    Declines when a target name is read by the RHS: python binds every target from the OLD values,
    and a sequential split would read one already updated.
    """

    def visit_Assign(self, node: ast.Assign) -> object:
        self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Tuple):
            return node
        targets = node.targets[0].elts
        if not all(isinstance(t, ast.Name) for t in targets):
            return node
        names = {t.id for t in targets}
        if any(isinstance(sub, ast.Name) and sub.id in names for sub in ast.walk(node.value)):
            return node
        parts = [tuple_element(node.value, i, len(targets)) for i in range(len(targets))]
        if any(p is None for p in parts):
            return node
        out: list[ast.stmt] = []
        for target, part in zip(targets, parts):
            stmt = ast.Assign(targets=[copy.deepcopy(target)], value=part)
            ast.copy_location(stmt, node)
            ast.fix_missing_locations(stmt)
            out.append(stmt)
        return out


class BaseEmitter:
    """Target-agnostic statement walk. Subclasses provide the per-form emit
    methods (``_emit_for`` etc.), the ``emit_expr`` expression walk, and the
    leaf hooks below."""

    #: Statement terminator appended to a bare expression statement
    #: (C: ``";"``; Fortran: ``""``).
    _STMT_TERM: str = ""
    #: ``break`` / ``continue`` rendered for the target.
    _KW_BREAK: str = "break"
    _KW_CONTINUE: str = "continue"
    #: How the target opens and closes a one-line comment. Empty disables provenance notes.
    _COMMENT: tuple = ()
    #: Canonical fp8 dtype -> this target's promote / demote / round function names.
    fp8_names: Mapping[str, Fp8Fns]
    kir: KernelIR

    @staticmethod
    def static_step_sign(step_node: Optional[ast.AST]) -> Optional[int]:
        """+1 / -1 when a range step's sign is decidable from the AST, else None.

        None means the sign is a RUNTIME fact and the loop direction cannot be baked in. Both
        backends used to fall back to a textual ``startswith("-")`` on the emitted step, which is
        only ever right for a literal: with ``s = -1`` held in a variable the text is ``s``, so C
        emitted a forward loop that ran zero times and Fortran adjusted the inclusive bound the
        wrong way and overran it. Neither failed loudly.
        """
        if step_node is None:
            return 1
        if isinstance(step_node, ast.UnaryOp) and isinstance(step_node.op, ast.USub):
            inner = BaseEmitter.static_step_sign(step_node.operand)
            return None if inner is None else -inner
        if isinstance(step_node, ast.Constant) and isinstance(step_node.value, (int, float)):
            return -1 if step_node.value < 0 else 1
        return None

    def numpy_note(self, node: ast.stmt, indent: str) -> str:
        """The comment line naming the numpy expression ``node`` was lowered from, or ``""``.

        Emitted only where the operation became an explicit loop nest, so the generated source says
        what it is doing. Where a target renders the operation as a named intrinsic instead
        (Fortran's ``MATMUL``, ``SUM``) no note is attached, because the intrinsic never reaches
        this path -- the name is already the documentation.
        """
        if not self._COMMENT:
            return ""
        text = numpy_origin(node)
        if not text:
            return ""
        open_, close = self._COMMENT
        return f"{indent}{open_} numpy: {text}{' ' + close if close else ''}\n"

    def emit_stmt_with_note(self, node: ast.stmt, indent: str) -> str:
        """``emit_stmt`` prefixed by its numpy provenance note, when it has one.

        Read the note only after the statement emits to something: a dropped statement (a bare
        return temp, an input-validation raise) would otherwise leave its comment behind with
        nothing under it.
        """
        text = self.emit_stmt(node, indent)
        return (self.numpy_note(node, indent) + text) if text else text

    def emit_block(self, stmts: List[ast.stmt], indent: str) -> str:
        out = [self.emit_stmt_with_note(s, indent) for s in stmts]
        return "\n".join(line for line in out if line)

    def emit_stmt(self, node: ast.stmt, indent: str) -> str:
        if isinstance(node, ast.For):
            return self._emit_for(node, indent)
        if isinstance(node, ast.While):
            return self._emit_while(node, indent)
        if isinstance(node, ast.If):
            return self._emit_if(node, indent)
        if isinstance(node, ast.Assign):
            return self._emit_assign(node, indent)
        if isinstance(node, ast.AugAssign):
            return self._emit_augassign(node, indent)
        if isinstance(node, ast.Expr):
            v = node.value
            # Drop bare docstrings AND no-op bare-name / constant expression
            # statements: an inlined in-place helper leaves its unused return temp
            # as ``x_hcall1`` on its own line -- a harmless no-op in C but an
            # unclassifiable statement in Fortran (minife's ``_matvec_std_arrays``).
            # A Call statement (real side effect) still falls through and is emitted.
            if isinstance(v, (ast.Constant, ast.Name)):
                return ""
            return f"{indent}{self.emit_expr(v)}{self._STMT_TERM}"
        if isinstance(node, ast.Break):
            return f"{indent}{self._KW_BREAK}"
        if isinstance(node, ast.Continue):
            return f"{indent}{self._KW_CONTINUE}"
        if isinstance(node, ast.Pass):
            return ""
        if isinstance(node, (ast.Raise, ast.Assert)):
            # Input-validation guards (``if bad: raise ValueError(...)`` /
            # ``assert n > 0``). HPCAgent-Bench kernels run on oracle-validated inputs, so
            # the guard never fires; drop it (an empty ``if`` body is valid C/
            # Fortran). Dropping -- not lowering -- because the message is a Python
            # string/f-string the backends cannot express and need not.
            return ""
        if isinstance(node, ast.Return):
            return self._emit_return(node, indent)
        raise NotImplementedError(
            f"unsupported statement: {type(node).__name__} (line {vars(node).get('lineno', '?')})"
        )

    def _emit_return(self, node: ast.Return, indent: str) -> str:
        """HPCAgent-Bench kernels are void -- outputs are written through array
        parameters, so ``return x`` is dropped by default. Fortran overrides to
        emit a bare ``return`` statement."""
        return ""

    def emit_expr_inner(self, node: ast.AST) -> str:
        raise NotImplementedError

    def name_dtype(self, name: str) -> str | None:
        raise NotImplementedError

    def wrap_narrow(self, text: str, wrap: str) -> str:
        """``text`` (computed in the wide int64) narrowed back to the ``wrap`` element dtype."""
        raise NotImplementedError

    def emit_expr(self, node: ast.AST) -> str:
        """Emit an expression, re-rounding a float BinOp result to the fp8 grid (per-op in numpy)
        and re-wrapping a narrow-int +/-/* result back to its element width (numpy wraps there;
        the promoting read computes wide, so an intermediate that overflows would not)."""
        text = self.emit_expr_inner(node)
        if isinstance(node, ast.BinOp):
            text = self.fp8_round(node, text)
        wrap = narrow_int.wrap_dtype(node, self.wrap_name_dtype)
        if wrap is not None:
            text = self.wrap_narrow(text, wrap)
        return text

    def wrap_name_dtype(self, name: str) -> str | None:
        """Name -> numpy dtype for the narrow-int wrap oracle; a shape symbol is the wide int64."""
        for s in self.kir.symbols:
            if s.name == name:
                return "int64"
        return self.name_dtype(name)

    def fp8_fns(self, dtype: str) -> Fp8Fns | None:
        return fp8_functions(dtype, self.fp8_names)

    def fp8_round(self, node: ast.BinOp, text: str) -> str:
        """Wrap a float BinOp result in the fp8 round-to-grid helper (per-op rounding is load-bearing, not decorative)."""
        if isinstance(node.op, FP8_NON_ARITH_OPS):
            return text
        fns = self.kernel_fp8_fns
        if fns is None or not self.touches_fp8(node):
            return text
        return f"{fns.round}({text})"

    @cached_property
    def kernel_fp8_fns(self) -> Fp8Fns | None:
        """The kernel's single fp8 format's helpers, or None if it uses none (mixing both formats is refused)."""
        used = fp8_dtypes_used(self.kir)
        if len(used) > 1:
            raise NotImplementedError(
                f"kernel {self.kir.kernel_name!r} mixes fp8 formats {used}: the grid each "
                f"intermediate rounds to is ambiguous"
            )
        return self.fp8_fns(used[0]) if used else None

    def touches_fp8(self, node: ast.AST) -> bool:
        """True when the subtree reads an fp8 array/scalar/local, so the enclosing op yields an fp8 float to re-round."""
        for sub in ast.walk(node):
            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name):
                name = sub.value.id
            elif isinstance(sub, ast.Name):
                name = sub.id
            else:
                continue
            if self.fp8_fns(self.name_dtype(name) or "") is not None:
                return True
        return False

    def promote_name_read(self, node: ast.Name, access: str) -> str:
        """Promote a bare fp8 Name to float on READ. Store ctx falls through to the store seam."""
        if not isinstance(node.ctx, ast.Load):
            return access
        fns = self.fp8_fns(self.name_dtype(node.id) or "")
        return f"{fns.promote}({access})" if fns is not None else access

    def store_fns(self, target: ast.AST) -> Fp8Fns | None:
        """:class:`Fp8Fns` when an assignment target is an fp8 element/name, else None (the store half of promote/demote)."""
        base = target
        while isinstance(base, ast.Subscript):
            base = base.value
        if not isinstance(base, ast.Name):
            return None
        return self.fp8_fns(self.name_dtype(base.id) or "")
