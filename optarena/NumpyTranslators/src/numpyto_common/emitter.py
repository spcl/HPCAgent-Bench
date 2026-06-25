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
from typing import List


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
            # Drop bare docstrings.
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                return ""
            return f"{indent}{self.emit_expr(v)}{self._STMT_TERM}"
        if isinstance(node, ast.Break):
            return f"{indent}{self._KW_BREAK}"
        if isinstance(node, ast.Continue):
            return f"{indent}{self._KW_CONTINUE}"
        if isinstance(node, ast.Pass):
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
