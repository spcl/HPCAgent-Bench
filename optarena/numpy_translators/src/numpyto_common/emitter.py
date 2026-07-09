"""Shared statement-dispatch skeleton for the imperative body emitters.

Only the target-agnostic walk lives here: ``emit_block`` and the ``emit_stmt``
dispatch (For / While / If / Assign / AugAssign / Expr / Break / Continue /
Pass / Return). Per-target leaves are small hooks (statement terminator,
break / continue keyword, return handling).

Each statement/expression *form* (loops, subscripts, calls, the type system) is
language-specific -- C flattens N-D subscripts and int-types, Fortran is 1-based
column-major with kind inference, indent steps differ -- so those stay
overridden in the subclass. A subclass may override ``emit_stmt`` wholesale.
"""
import ast
from typing import List


class BaseEmitter:
    """Target-agnostic statement walk. Subclasses provide the per-form emit
    methods (``_emit_for`` etc.), ``emit_expr``, and the leaf hooks below."""

    #: Statement terminator appended to a bare expression statement
    #: (C: ``";"``; Fortran: ``""``).
    _STMT_TERM: str = ""
    #: ``break`` / ``continue`` rendered for the target.
    _KW_BREAK: str = "break"
    _KW_CONTINUE: str = "continue"

    def emit_block(self, stmts: List[ast.stmt], indent: str) -> str:
        out = [self.emit_stmt(s, indent) for s in stmts]
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
            # ``assert n > 0``). OptArena kernels run on oracle-validated inputs, so
            # the guard never fires; drop it (an empty ``if`` body is valid C/
            # Fortran). Dropping -- not lowering -- because the message is a Python
            # string/f-string the backends cannot express and need not.
            return ""
        if isinstance(node, ast.Return):
            return self._emit_return(node, indent)
        raise NotImplementedError(
            f"unsupported statement: {type(node).__name__} "
            f"(line {getattr(node, 'lineno', '?')})")

    def _emit_return(self, node: ast.Return, indent: str) -> str:
        """OptArena kernels are void -- outputs are written through array
        parameters, so ``return x`` is dropped by default. Fortran overrides to
        emit a bare ``return`` statement."""
        return ""
