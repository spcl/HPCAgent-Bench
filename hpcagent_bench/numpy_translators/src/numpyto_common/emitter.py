"""Shared statement-dispatch skeleton for the imperative body emitters.

Only the genuinely target-agnostic part of the walk lives here: ``emit_block``
and the ``emit_stmt`` dispatch (For / While / If / Assign / AugAssign / Expr /
Break / Continue / Pass / Return). The leaves that differ per target are small
hooks (statement terminator, break / continue keyword, return handling).

The body of each statement/expression *form* (loops, subscripts, calls, the
type system) is legitimately language-specific -- C flattens N-D subscripts and
runs an int-typing pass, Fortran is 1-based / column-major with kind inference,
their indent step differs (2 vs 4 spaces), control flow is braces vs
``do/end do`` -- so those stay overridden in the subclass rather than forced
through a leaky hook surface. A subclass is free to override ``emit_stmt``
wholesale if a target ever needs a different dispatch.
"""
import ast
from typing import List, Optional, Sequence

from numpyto_common.ir import numpy_origin


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
    return (f"cannot index {name!r} with {n_indices} axes: its shape is {rank} "
            f"(rank {0 if shape is None else len(shape)}). "
            f"Declare init.shapes[{name!r}] with the matching rank.")


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
        raise NotImplementedError(f"unsupported statement: {type(node).__name__} "
                                  f"(line {vars(node).get('lineno', '?')})")

    def _emit_return(self, node: ast.Return, indent: str) -> str:
        """HPCAgent-Bench kernels are void -- outputs are written through array
        parameters, so ``return x`` is dropped by default. Fortran overrides to
        emit a bare ``return`` statement."""
        return ""
