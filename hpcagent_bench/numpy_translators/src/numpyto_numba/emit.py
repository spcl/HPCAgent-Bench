"""Emit a Numba-compiled version of a numpy kernel.

Numba supports a large subset of numpy plus pure-Python loops; the
translation is simply wrapping the function in
``@numba.njit(parallel=True)`` and leaving the body alone. There is ONE
numba build and it is the parallel one -- it is also the
``scientific_computing`` speedup denominator, and a serial denominator on a
multi-core box measures the wrong thing.

The two bodies that do NOT get ``parallel=True`` are:

1. one calling a :data:`PARFOR_UNSAFE_CALLS` op, which numba's parfor rewriter -- the thing the
   flag turns on -- answers differently from numpy.
2. one performing an in-place slice assignment whose RHS reads the same array (e.g.
   ``a[i, 1:-1] += a[i, 2:]``). Numba rewrites that whole-array update into a parallel loop, which
   races on the overlapping slice.

The emit additionally rewrites the loop variable of ONE ``range``
for-loop to ``numba.prange`` -- but ONLY a loop a
conservative dependency check can prove has no loop-carried dependency.
A blind rewrite of the first ``range`` loop (the old behaviour) silently
raced a scan / prefix-sum loop (``a[i] = a[i-1] + x[i]``) or a scalar /
same-cell reduction (``out[0] += x[i]``): prange runs the iterations out
of order, so the read of a previously-written cell sees a garbage value.
Correctness wins over speed here -- a loop that cannot be PROVEN
independent is left serial (plain ``range``), never guessed parallel.
"""

import ast
import re

from numpyto_common.parallelism import loop_is_parallel_safe

#: Whole-array numpy calls numba's parfor rewriter -- what ``parallel=True`` turns on -- answers
#: differently from numpy, each measured on numba 0.65.1: ``max`` / ``min`` (also spelled ``amax`` /
#: ``amin``) SUPPRESS NaN where numpy propagates it, and a rectangular ``eye(m, n)`` fused with a
#: following prange is read back as if square, so ``eye(3, 5)`` copies rows 0, 3 and 6 of its own
#: flat buffer. A body calling one loses ``parallel=True``, the trade ``fastmath`` already makes.
PARFOR_UNSAFE_CALLS = frozenset({"max", "min", "amax", "amin", "eye", "identity"})


def emit_numba(numpy_source: str, fastmath: bool = False, kir=None) -> str:
    """Translate one numpy kernel source into its Numba sibling.

    :param numpy_source: contents of ``<short>_numpy.py``.
    :param fastmath: opt into ``fastmath=True``. Off by default: it lets LLVM reassociate
        reductions and assume no-nan/no-inf, which diverges from numpy's exact semantics, and it
        miscompiles some gather/while-loop reductions into a SIGSEGV on numba 0.65 + LLVM.
    :param kir: parsed :class:`KernelIR`; when supplied, ops numba cannot
        type verbatim (batched >=3-D ``@``) are desugared into plain loops.
    :returns: Python source code.
    """
    if kir is not None:
        from numpyto_common.numpy_desugar import desugar_for_python_backend
        numpy_source = desugar_for_python_backend(numpy_source, kir, backend="numba")
    parallel = not (calls_a_parfor_unsafe_op(numpy_source) or _has_inplace_slice_self_dependency(numpy_source))
    opts = ["parallel=True"] if parallel else []
    if fastmath:
        opts.append("fastmath=True")
    opts.append("cache=True")
    decorator = f"@nb.njit({', '.join(opts)})"

    # 1. Make sure ``import numba as nb`` is present, and silence the one warning that
    #    ALWAYS emitting parallel=True guarantees: numba raises NumbaPerformanceWarning on a
    #    kernel where nothing could be parallelised (a scan, a scalar reduction). That is a
    #    statement about the kernel, not a defect in the emit, and it would otherwise fire once
    #    per such kernel across the whole corpus sweep. Scoped to the category, never a bare
    #    ignore -- a typing or lowering warning must still be heard.
    out = numpy_source
    if "import numba" not in out:
        out = ("import warnings\n"
               "import numba as nb\n"
               "from numba.core.errors import NumbaPerformanceWarning\n"
               "warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)\n") + out

    # 2. Inject the decorator on EVERY top-level ``def`` (``(?m)^`` anchors to
    #    column 0, so indented / nested defs are skipped). Decorating only the
    #    first def silently left the real kernel un-njit'd whenever a helper was
    #    defined first (lenet's ``relu`` before ``lenet5``) -- numba then ran the
    #    kernel as plain numpy, a false pass. numba's njit is lazy, so an unused
    #    helper is never compiled; a called one must be njit to be callable from
    #    nopython code, so decorating all top-level defs is both correct and free.
    out = re.sub(r"(?m)^(def\s+\w+\()", f"{decorator}\n\\1", out)

    # 3. Rewrite ONE ``range`` loop to ``nb.prange`` -- the
    #    first (in source order) that ``parallelism.loop_is_parallel_safe`` (the
    #    shared source-of-truth predicate the C / Fortran OpenMP emitters use too)
    #    proves carries no cross-iteration dependency. A loop that fails the check (scan,
    #    scalar/same-cell reduction, index-shifted stencil, data-dependent
    #    scatter) stays serial: prange would reorder its iterations and read a
    #    not-yet-written / already-overwritten cell -> silent miscompile.
    #    The rewrite splices only the ``range`` identifier of the chosen loop
    #    (located via the AST) so the body is otherwise preserved verbatim.
    out = _parallelize_one_range_loop(out)

    header = (f'"""Auto-generated by NumpyToNumba (njit{" parallel=True" if parallel else ""}). '
              'Decorator added; body preserved verbatim."""\n')
    return header + out


def calls_a_parfor_unsafe_op(src: str) -> bool:
    """True if ``src`` calls a :data:`PARFOR_UNSAFE_CALLS` name, in either the ``np.max(a)`` or the
    ``a.max()`` spelling -- both reach the same numba implementation."""
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in PARFOR_UNSAFE_CALLS
        for n in ast.walk(ast.parse(src)))


def _has_inplace_slice_self_dependency(src: str) -> bool:
    """True if the body does an in-place slice assignment whose RHS reads the same array.

    ``a[i, 1:-1] += a[i, 2:]`` is the canonical case: numba's parfor pass turns the
    whole-array update into a parallel loop, but the LHS and RHS slices overlap, so
    the read races the write. A scalar subscript like ``a[i] = a[i - 1] + x[i]`` is
    NOT caught here; that dependency is handled by the prange-rewrite check instead.
    """
    def _base_name(node: ast.AST) -> str | None:
        while isinstance(node, ast.Subscript):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    def _contains_slice(node: ast.AST) -> bool:
        return any(isinstance(child, ast.Slice) for child in ast.walk(node))

    for stmt in ast.walk(ast.parse(src)):
        if isinstance(stmt, (ast.AugAssign, ast.Assign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                if not isinstance(target, ast.Subscript) or not _contains_slice(target):
                    continue
                lhs_name = _base_name(target)
                if lhs_name is None:
                    continue
                for rhs in ast.walk(stmt.value):
                    if isinstance(rhs, ast.Subscript) and _base_name(rhs) == lhs_name:
                        return True
    return False


def _abs_offset(src: str, lineno: int, col: int) -> int:
    """Absolute character offset of (1-based ``lineno``, 0-based ``col``). The
    source is ASCII, so ``col_offset`` (a UTF-8 byte offset) equals the char offset."""
    lines = src.splitlines(keepends=True)
    return sum(len(line) for line in lines[:lineno - 1]) + col


def _parallelize_one_range_loop(src: str) -> str:
    """Rewrite the ``range`` identifier of the first (source-order) provably
    independent ``range`` for-loop to ``nb.prange``. If none qualify, return
    ``src`` unchanged (fully serial -- correct, just not parallel)."""
    tree = ast.parse(src)
    range_fors = sorted((n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.iter, ast.Call)
                         and isinstance(n.iter.func, ast.Name) and n.iter.func.id == "range"),
                        key=lambda n: (n.lineno, n.col_offset))
    target = next((f for f in range_fors if loop_is_parallel_safe(f)), None)
    if target is None:
        return src
    fn = target.iter.func
    off = _abs_offset(src, fn.lineno, fn.col_offset)
    if src[off:off + 5] != "range":
        return src  # position drift (should not happen); leave serial rather than corrupt.
    return src[:off] + "nb.prange" + src[off + 5:]
