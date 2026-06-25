"""Sanitize emitted Python source before it crosses into a container / mounted
work folder (directive #4).

Two operations, both driven through ``ast`` so they are faithful (never a regex
guess):

* **strip comments** -- ``ast.parse`` -> ``ast.unparse`` drops every ``#``
  comment (comments are not part of the AST); docstrings are string-expression
  *statements* and survive unparse, so they are removed explicitly.
* **mangle identifiers** -- rename locals per a ``{original: mangled}`` registry
  via a ``NodeTransformer`` (off by default; opt-in).

Scope: the *Python-emitting* backends (CuPy / Numba / Pythran, and later
JAX / DaCe) plus the assembled runnable module. The C / Fortran emitters never
carry Python comments (they reconstruct the program from the AST), so they do
not route through here.

Comment stripping is automatic for any backend that already round-trips through
``ast.unparse``; this module is what gives the *textual passthrough* backends
the same guarantee without forcing them onto the IR.
"""
import ast
from typing import Dict, Optional


class Rename(ast.NodeTransformer):
    """Rename bound identifiers per a registry. Conservative: only ``Name``,
    function/argument names and keyword-argument names -- never attribute
    members (so ``cp.zeros`` keeps ``zeros``)."""

    def __init__(self, registry: Dict[str, str]):
        self._r = registry

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._r.get(node.id, node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._r.get(node.arg, node.arg)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self._r.get(node.name, node.name)
        self.generic_visit(node)
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        if node.arg is not None:
            node.arg = self._r.get(node.arg, node.arg)
        self.generic_visit(node)
        return node


def _strip_docstrings(tree: ast.AST) -> None:
    """Drop the leading string-expression statement of the module and of every
    function / class definition."""
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]


def sanitize(py_src: str, *, strip_docstrings: bool = True,
             name_registry: Optional[Dict[str, str]] = None) -> str:
    """Return ``py_src`` with ``#`` comments removed (and, by default,
    docstrings), optionally mangled per ``name_registry``.

    ``py_src`` must be valid Python (the Python-emitting backends' output).
    """
    tree = ast.parse(py_src)
    if name_registry:
        tree = Rename(name_registry).visit(tree)
    if strip_docstrings:
        _strip_docstrings(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"
