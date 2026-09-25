"""Emit a Numba-compiled version of a numpy kernel.

Numba types a large subset of numpy plus plain loops, so a dense kernel keeps its body and gains
``@nb.njit(parallel=True)``. The body changes only where numba cannot follow numpy:

* a sparse ``A @ x`` against a live ``scipy.sparse`` matrix is lowered onto the unpacked buffer ABI
  (:mod:`numpyto_numba.sparse`);
* a 1-D ``np.fft.fft``/``ifft`` runs in ``objmode`` (:mod:`numpyto_numba.objmode_fft`);
* ``parallel=True`` is dropped for a body numba's parfor pass would answer differently from numpy,
  and at most one provably independent ``range`` loop becomes ``nb.prange``
  (:mod:`numpyto_numba.parfor`). A loop that cannot be proven independent stays serial.
"""

import ast
import re

from numpyto_common.frontend import PruneSparseDispatch
from numpyto_common.ir import KernelIR

from numpyto_numba.objmode_fft import rewrite_fft_to_objmode
from numpyto_numba.parfor import calls_a_parfor_unsafe_op, has_inplace_slice_self_dependency, parallelize_one_range_loop
from numpyto_numba.sparse import rewrite_sparse_matmuls


def without_sparse_dispatch(source: str) -> str:
    """``source`` with its sparse dispatch branch dropped (:class:`PruneSparseDispatch`).

    numba cannot type ``isinstance(x, np.ndarray)``, and the numba build only ever runs on dense
    arrays, so the branch is dead here as in the static backends (banded_mmt). Returned verbatim,
    comments kept, when nothing is pruned."""
    tree = ast.parse(source)
    before = ast.dump(tree)
    PruneSparseDispatch().visit(tree)
    return source if ast.dump(tree) == before else ast.unparse(ast.fix_missing_locations(tree))


def emit_numba(numpy_source: str, fastmath: bool = False, kir: KernelIR | None = None) -> str:
    """Translate one numpy kernel source into its Numba sibling.

    :param numpy_source: contents of ``<short>_numpy.py``.
    :param fastmath: opt into ``fastmath=True``. Off by default: it lets LLVM reassociate
        reductions and assume no-nan/no-inf, which diverges from numpy's exact semantics, and it
        miscompiles some gather/while-loop reductions into a SIGSEGV on numba 0.65 + LLVM.
    :param kir: parsed :class:`KernelIR`; when supplied, ops numba cannot
        type verbatim (batched >=3-D ``@``) are desugared into plain loops.
    :returns: Python source code.
    """
    sparse = False
    numpy_source = without_sparse_dispatch(numpy_source)
    if kir is not None:
        from numpyto_common.numpy_desugar import desugar_for_python_backend

        numpy_source = desugar_for_python_backend(numpy_source, kir, backend="numba")
        unpacked = rewrite_sparse_matmuls(numpy_source, kir)
        if unpacked is not None:
            numpy_source, sparse = unpacked, True
    numpy_source, uses_objmode = rewrite_fft_to_objmode(numpy_source)
    parallel = not (calls_a_parfor_unsafe_op(numpy_source) or has_inplace_slice_self_dependency(numpy_source))
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
        out = (
            "import warnings\n"
            "import numba as nb\n"
            "from numba.core.errors import NumbaPerformanceWarning\n"
            "warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)\n"
        ) + out
    if uses_objmode:
        out = "from numba import objmode\n" + out
    # The sparse rewrite allocates its result temps with np.zeros; spmm's reference imports nothing.
    if sparse and "import numpy" not in out:
        out = "import numpy as np\n" + out

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
    out = parallelize_one_range_loop(out)

    body = "sparse operands lowered onto the unpacked buffer ABI" if sparse else "body preserved verbatim"
    header = (
        f'"""Auto-generated by NumpyToNumba (njit{" parallel=True" if parallel else ""}). Decorator added; {body}."""\n'
    )
    return header + out
