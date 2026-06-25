"""Canonical NumPy Form (CNF) validator.

AST-walks a kernel's ``<short>_numpy.py`` and reports violations of the
three CNF invariants documented in ``docs/canonical_numpy_form.md``:

  INV1  Static shape, known at declaration. No reassignment changes an
        array's rank/shape; want a different shape -> a new named buffer.
  INV2  Explicit indexing. No chained subscripts ``a[i][j]`` and no
        fancy indexing ``a[index_array]`` (outside the sparse-gather
        form). Index higher-rank arrays with a full index.
  INV3  Declare-then-fill, never grow. No list / set / dict / collections
        containers; no ``.append`` / ``.extend`` / ``np.concatenate`` of
        varying length inside the kernel body.

Each violation is a :class:`CnfViolation` naming the kernel, the line,
the invariant, and the canonical-rewrite hint (a cookbook-entry pointer).
The walker is conservative: it only flags patterns it is confident are
non-canonical, so a clean report is a strong signal, while a flagged
pattern is worth an author's look even if occasionally a false positive
(documented per-rule below).

Run:  ``python -m optarena.validate_cnf <kernel_numpy.py> [...]``
   or import :func:`validate_cnf_file` / :func:`scan_violations`.
"""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


# Containers forbidden by INV3.
_FORBIDDEN_CTORS = {"list", "set", "dict", "frozenset", "tuple"}
_FORBIDDEN_COLLECTIONS = {
    "deque", "defaultdict", "OrderedDict", "Counter", "namedtuple",
}
# Growth / dynamic-shape method calls forbidden by INV3.
_FORBIDDEN_METHODS = {"append", "extend", "insert", "pop", "remove", "add"}
# numpy calls that grow / change rank dynamically (INV1/INV3 hints).
_DYNAMIC_NP = {"concatenate", "hstack", "vstack", "stack", "append",
                "resize", "vsplit", "hsplit", "split"}


@dataclass(frozen=True)
class CnfViolation:
    kernel: str
    lineno: int
    col: int
    invariant: str          # "INV1" | "INV2" | "INV3"
    message: str
    hint: str               # cookbook pointer


def _kernel_funcs(tree: ast.AST) -> List[ast.FunctionDef]:
    """Return top-level (non-nested) function defs — the kernel bodies."""
    return [n for n in tree.body if isinstance(n, ast.FunctionDef)] \
        if isinstance(tree, ast.Module) else []


def _array_decl_rank(value: ast.AST) -> Optional[int]:
    """If ``value`` is ``np.zeros/empty/ones/full((s0, s1, ...), ...)``,
    return the declared rank; else None. A shape arg that is a bare Name
    or single int yields rank 1."""
    if not isinstance(value, ast.Call):
        return None
    func = value.func
    attr = None
    if isinstance(func, ast.Attribute):
        attr = func.attr
    elif isinstance(func, ast.Name):
        attr = func.id
    if attr not in {"zeros", "empty", "ones", "full", "ndarray",
                     "zeros_like", "empty_like", "ones_like"}:
        return None
    if attr.endswith("_like"):
        return None  # rank follows the source; treat as opaque
    if not value.args:
        return None
    shape_arg = value.args[0]
    if isinstance(shape_arg, (ast.Tuple, ast.List)):
        return len(shape_arg.elts)
    # bare int / Name / arithmetic -> rank 1
    return 1


def scan_violations(source: str, kernel: str) -> List[CnfViolation]:
    """Parse ``source`` and return CNF violations for the kernel funcs."""
    tree = ast.parse(source)
    out: List[CnfViolation] = []
    for fn in _kernel_funcs(tree):
        out.extend(_scan_function(fn, kernel))
    return out


def _scan_function(fn: ast.FunctionDef, kernel: str) -> List[CnfViolation]:
    out: List[CnfViolation] = []

    # --- INV1: track declared array ranks; flag rank-changing reassign --
    # Map name -> declared rank (from np.zeros/empty/... or reshape arg).
    decl_rank: Dict[str, int] = {}

    def _record_and_check_assign(target_id: str, value: ast.AST,
                                  node: ast.AST) -> None:
        new_rank = _array_decl_rank(value)
        # reshape(x, (a, b, ...)) declares the target's new rank.
        if new_rank is None and isinstance(value, ast.Call):
            f = value.func
            if (isinstance(f, ast.Attribute) and f.attr == "reshape"
                    and len(value.args) >= 2):
                sh = value.args[1]
                if isinstance(sh, (ast.Tuple, ast.List)):
                    new_rank = len(sh.elts)
                else:
                    new_rank = 1
        if new_rank is None:
            return
        prev = decl_rank.get(target_id)
        if prev is not None and prev != new_rank:
            out.append(CnfViolation(
                kernel, node.lineno, node.col_offset, "INV1",
                f"'{target_id}' reassigned from rank-{prev} to rank-{new_rank}; "
                "an array's rank must not change.",
                "cookbook: whole-array reassign with shape change -> named buffers"))
        decl_rank[target_id] = new_rank

    for node in ast.walk(fn):
        # INV1 — assignment rank tracking
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            _record_and_check_assign(node.targets[0].id, node.value, node)

        # INV2 — chained subscript a[i][j]
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Subscript):
            # Only flag when the inner subscript's base is a Name (an
            # array), not e.g. a tuple-shape constant fold.
            inner = node.value
            if isinstance(inner.value, ast.Name):
                out.append(CnfViolation(
                    kernel, node.lineno, node.col_offset, "INV2",
                    f"chained subscript on '{inner.value.id}' "
                    "(a[i][j]); use a single full index a[i, j].",
                    "cookbook: chained subscript -> full index"))

        # INV2 — fancy indexing a[index_array] (a Name slot that is an
        # array, not a scalar loop var). We can't resolve dtype here, so
        # flag a Subscript whose slot is a Name AND whose base is a Name
        # only when the slot name looks array-ish is too weak; instead we
        # flag the well-known gather shape `a[b]` where `b` is itself
        # later subscripted as an array. Conservative: flag a Subscript
        # slot that is a Name appearing elsewhere as a Subscript base.
        # (Handled in a second pass below.)

        # INV3 — forbidden container constructors / collections
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in _FORBIDDEN_CTORS:
                # Allow `tuple()`-free: bare tuple literals are fine; only
                # the `tuple(...)` / `list(...)` ctor calls are flagged.
                out.append(CnfViolation(
                    kernel, node.lineno, node.col_offset, "INV3",
                    f"'{f.id}(...)' container constructor; CNF allows only "
                    "static-shape tensors.",
                    "cookbook: dynamic growth -> pre-declared buffer"))
            if isinstance(f, ast.Attribute):
                # Is the receiver a module alias (np / numpy / cp / cupy /
                # math)? ``np.add(Z, C, Z)`` is an elementwise ufunc, NOT
                # a set ``.add`` — only flag .add/.append/etc. on a
                # NON-module receiver.
                recv_is_module = (
                    isinstance(f.value, ast.Name)
                    and f.value.id in {"np", "numpy", "cp", "cupy", "math"})
                if f.attr in _FORBIDDEN_METHODS and not recv_is_module:
                    out.append(CnfViolation(
                        kernel, node.lineno, node.col_offset, "INV3",
                        f"'.{f.attr}(...)' grows/mutates a container; "
                        "CNF forbids dynamic resize.",
                        "cookbook: .append -> pre-declared worst-case buffer"))
                if f.attr in _DYNAMIC_NP:
                    out.append(CnfViolation(
                        kernel, node.lineno, node.col_offset, "INV1",
                        f"'np.{f.attr}(...)' changes shape/rank dynamically.",
                        "cookbook: reshape/concat -> declared buffer + copy"))
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                    and f.value.id == "collections" \
                    and f.attr in _FORBIDDEN_COLLECTIONS:
                out.append(CnfViolation(
                    kernel, node.lineno, node.col_offset, "INV3",
                    f"'collections.{f.attr}' is not a tensor; CNF forbids it.",
                    "cookbook: dynamic growth -> pre-declared buffer"))

    # INV2 — fancy gather a[idx] where idx is used elsewhere as an array.
    # Build the set of names ever used as a Subscript BASE (i.e. arrays).
    array_names: Set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            array_names.add(node.value.id)
    for node in ast.walk(fn):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            sl = node.slice
            # a[idx] single-axis where idx is a Name that is itself an array
            if isinstance(sl, ast.Name) and sl.id in array_names:
                out.append(CnfViolation(
                    kernel, node.lineno, node.col_offset, "INV2",
                    f"fancy index '{node.value.id}[{sl.id}]' where '{sl.id}' "
                    "is an array; use an explicit gather loop (or the sparse "
                    "layout system).",
                    "cookbook: fancy gather -> explicit loop"))

    # Stable order by line then column.
    out.sort(key=lambda v: (v.lineno, v.col))
    return out


def validate_cnf_file(path: str) -> List[CnfViolation]:
    """Validate one ``<short>_numpy.py`` file. Returns the violations."""
    import pathlib
    p = pathlib.Path(path)
    short = p.stem.removesuffix("_numpy")
    return scan_violations(p.read_text(), short)


def main(argv: Optional[List[str]] = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m optarena.validate_cnf <kernel_numpy.py> [...]",
              file=sys.stderr)
        return 2
    total = 0
    for path in args:
        violations = validate_cnf_file(path)
        for v in violations:
            print(f"{path}:{v.lineno}:{v.col}: [{v.invariant}] {v.message} "
                  f"({v.hint})")
        total += len(violations)
    if total:
        print(f"\n{total} CNF violation(s) across {len(args)} file(s).",
              file=sys.stderr)
        return 1
    print(f"CNF clean: {len(args)} file(s) checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
