"""The library-node rewriter: replaces each registered numpy call assignment with its loop expansion."""

import ast
import copy
import inspect
from typing import TYPE_CHECKING
from collections.abc import Callable

from hpcagent_bench.translators.numpyto_common.ir import tag_numpy_origin
from hpcagent_bench.translators.numpyto_common.lib_nodes.call_hoist import CallHoister, numpy_call_key
from hpcagent_bench.translators.numpyto_common.lib_nodes.dims import NP_ZEROS_ALIASES, call_to_str
from hpcagent_bench.translators.numpyto_common.lib_nodes.elementwise import UNARY_C_MATH
from hpcagent_bench.translators.numpyto_common.lib_nodes.extents import is_integer_expr, iter_extent_of_
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import (
    alloc_marker,
    const_,
    const_int,
    is_full_slice_subscript,
    is_shape_scalar,
    reads_complex,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.matmul_hoist import MatmulHoister
from hpcagent_bench.translators.numpyto_common.lib_nodes.registry import NP_CALL_EXPANDERS

if TYPE_CHECKING:
    pass


def call_expander(
    expander: Callable,
    target: ast.expr,
    args: list[ast.expr],
    keywords: list[ast.keyword],
    shape_table: dict[str, tuple[str, ...]],
    local_dtypes: dict[str, str] | None = None,
    fresh_local_allocs: dict[str, tuple[str, ...]] | None = None,
    dim_aliases: dict[str, str] | None = None,
    library: bool = False,
    library_nd: bool = False,
) -> list[ast.stmt]:
    """Adapter: pass ``keywords``/``local_dtypes``/``fresh_local_allocs``/``library`` to
    expanders that accept them, else call with the legacy signature. The two
    extra tables let an expander register internal working buffers (shape +
    dtype) so the emit declares them correctly. ``library`` is the same "prefer a real library
    over a loop nest" signal :data:`BLAS_GEMM_MARKER` uses for matmul (``expand_fft``/
    ``expand_ifft``/``expand_fftn``/``expand_ifftn`` are its other consumer, see
    FFT_LIBRARY_MARKER).
    """
    sig = inspect.signature(expander)
    params = sig.parameters
    extras: dict[str, object] = {}
    if "kwargs" in params:
        extras["kwargs"] = keywords
    if "local_dtypes" in params and local_dtypes is not None:
        extras["local_dtypes"] = local_dtypes
    if "fresh_local_allocs" in params and fresh_local_allocs is not None:
        extras["fresh_local_allocs"] = fresh_local_allocs
    if "dim_aliases" in params and dim_aliases is not None:
        extras["dim_aliases"] = dim_aliases
    if "library" in params:
        extras["library"] = library
    if "library_nd" in params:
        extras["library_nd"] = library_nd
    return expander(target, args, shape_table, **extras)


SLICE_TARGET_EXPANDERS = {("np", "cumsum"), ("np", "cumprod")}

#: Expander keys that write element-wise to ``target`` (no allocation);
#: ``target`` must already be declared at the C level. When the kernel body
#: uses ``X = np.linspace(...)`` as the first reference to ``X``, the
#: LibNodeRewriter registers ``X`` in :attr:`fresh_local_allocs` so the emitter
#: generates a local decl.
ELEMENT_WRITE_EXPANDERS = {
    ("np", "linspace"),
    ("np", "arange"),
    ("np", "fromfunction"),
    # Elementwise functions that write to a fresh-local LHS need the
    # same auto-alloc treatment -- the original Assign is replaced by
    # the loop nest, leaving the target dangling without a decl.
    ("np", "less"),
    ("np", "less_equal"),
    ("np", "greater"),
    ("np", "greater_equal"),
    ("np", "equal"),
    ("np", "not_equal"),
    ("np", "logical_and"),
    ("np", "logical_or"),
    ("np", "logical_not"),
    *(("np", name_) for name_ in UNARY_C_MATH),
    ("np", "maximum"),
    ("np", "minimum"),
    ("np", "add"),
    ("np", "subtract"),
    ("np", "multiply"),
    ("np", "divide"),
    ("np", "power"),
    ("np", "exp"),
    ("np", "log"),
    ("np", "sqrt"),
    ("np", "sin"),
    ("np", "cos"),
    ("np", "tan"),
    ("np", "tanh"),
    ("np", "abs"),
    ("np", "absolute"),
    ("np", "sort"),
    ("np", "histogram"),
    ("np", "linalg.inv"),
    ("np", "linalg.solve"),
    ("np", "linalg.lstsq"),
    # Contraction / scan / indexing ops that write element-wise to a fresh LHS.
    ("np", "einsum"),
    ("np", "tensordot"),
    ("np", "inner"),
    ("np", "trace"),
    ("np", "diagonal"),
    ("np", "diag"),
    ("np", "fft.fftfreq"),
    ("np", "cumsum"),
    ("np", "cumprod"),
    ("np", "searchsorted"),
    # Same shape as cumsum/cumprod -- a running max/min bound to a fresh local. Left out, the
    # expander consumed the Assign that carried the allocation marker and the scan's first store
    # went through a NULL pointer.
    ("np", "maximum.accumulate"),
    ("np", "minimum.accumulate"),
    # Same shape again: the tile-and-write nest replaces the Assign that carried the marker, so a
    # fresh ``row_of = np.repeat(...)`` was emitted with no declaration at all.
    ("np", "repeat"),
    # An AXIS reduction writes one element per kept-axes position, so its fresh LHS is an array and
    # needs the marker like any other element-write. Without it the Fortran backend declared
    # ``valid = match.any(axis=-1)`` allocatable -- its extent names a computed scalar -- and then
    # had no site to allocate it at, so the reduction's first store went into an unallocated array.
    ("np", "sum"),
    ("np", "prod"),
    ("np", "mean"),
    ("np", "any"),
    ("np", "all"),
    ("np", "count_nonzero"),
    ("np", "argmax"),
    ("np", "argmin"),
    ("np", "roll"),
    ("np", "tril"),
    ("np", "pad"),
    # ``out = np.concatenate((a, b), axis=k)`` copies each operand into ``out`` at its
    # offset -- an element-write like the rest, so its fresh LHS needs the same auto-alloc
    # (dwt2d's ``e1 = np.concatenate((e[:, 1:], e[:, 0:1]), axis=1)`` was emitted undeclared).
    ("np", "concatenate"),
}


def index_axes(sub: ast.Subscript) -> tuple[str, ...] | None:
    """Per-axis index expressions of a fully scalar subscript, or ``None`` when
    any axis is a slice (no single cell to reason about)."""
    elts = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else [sub.slice]
    if any(isinstance(e, (ast.Slice, ast.Starred)) for e in elts):
        return None
    return tuple(ast.unparse(e) for e in elts)


def reduction_misses_target(target: ast.Subscript, loop: ast.For, body: ast.expr) -> bool:
    """True when moving the reduction onto ``target`` cannot change what ``body``
    reads: either ``body`` never touches the target's array, or every read of it
    provably misses ``target``'s cell for all iterations of ``loop``.

    ``trmm``'s ``B[i, j] += sum_k A[k, i] * B[k, j]`` needs the second rung: ``k``
    runs from ``i + 1``, so the accumulating cell is never read back. ``symm``
    takes the first (``temp2`` is write-only in the body).
    """
    base = target.value.id
    mentions = [n for n in ast.walk(body) if isinstance(n, ast.Name) and n.id == base]
    if not mentions:
        return True
    reads = [n for n in ast.walk(body) if isinstance(n, ast.Subscript) and n.value in mentions]
    if len(reads) != len(mentions):
        return False  # a bare-Name (whole-array) mention -- no cell to compare
    axes = index_axes(target)
    if axes is None:
        return False
    # The nonnegativity below is what proves the miss, so demand the 0-based
    # ``range(n)`` that makes it true rather than assuming every hoist emits one.
    if not (
        isinstance(loop.iter, ast.Call)
        and isinstance(loop.iter.func, ast.Name)
        and loop.iter.func.id == "range"
        and len(loop.iter.args) == 1
    ):
        return False
    # Deferred: sympy import costs ~100s of ms and only this narrow shape needs it.
    import sympy

    iter_name = loop.target.id
    it = sympy.Symbol(iter_name, nonnegative=True)
    for read in reads:
        read_axes = index_axes(read)
        if read_axes is None or len(read_axes) != len(axes):
            return False
        try:
            diffs = [
                sympy.sympify(r, locals={iter_name: it}) - sympy.sympify(t, locals={iter_name: it})
                for r, t in zip(read_axes, axes)
            ]
        except (SyntaxError, TypeError, AttributeError, ValueError, sympy.SympifyError):
            return False
        if not any(d.is_zero is False for d in diffs):
            return False
    return True


def accumulation_addend(step: ast.stmt, scalar: str) -> ast.expr | None:
    """What one ``+=`` step of ``scalar`` adds, in either spelling the lowerings emit
    (``s += x`` from the matmul hoist, ``s = s + x`` from the reduction expanders), else ``None``."""
    if isinstance(step, ast.AugAssign):
        if isinstance(step.op, ast.Add) and isinstance(step.target, ast.Name) and step.target.id == scalar:
            return step.value
        return None
    if (
        isinstance(step, ast.Assign)
        and len(step.targets) == 1
        and isinstance(step.targets[0], ast.Name)
        and step.targets[0].id == scalar
        and isinstance(step.value, ast.BinOp)
        and isinstance(step.value.op, ast.Add)
        and isinstance(step.value.left, ast.Name)
        and step.value.left.id == scalar
    ):
        return step.value.right
    return None


def retarget_scalar_accumulator(node: ast.stmt, prelude: list[ast.stmt]) -> list[ast.stmt] | None:
    """Fold ``s = 0.0; for k: s += f(k); T[idx] (+)= s`` into an in-place
    reduction on ``T[idx]``, dropping the scalar. Returns ``None`` when the
    statement is not that shape.

    ``emit_pluto`` hoists every scalar declaration above ``#pragma scop``, so
    ``s`` models to pet as a one-element array live across the whole iteration
    domain: the false WAW/WAR fragments the schedule, and Pluto's ``--parallel``
    privatises only its own tile counters, leaving ``s`` shared in the parallel
    band -- a race. An array cell carries the same reduction with no
    scop-external state, which is why ``syrk`` (reducing into ``C[i, si1]``) was
    never affected.

    pet goes further than racing on that scalar: a statement whose only write is one drops out of
    the transformed output entirely (POLYCC-009), so the same fold is what keeps a full
    ``np.sum`` into an array cell computing at all.
    """
    if len(prelude) < 2 or not isinstance(node.value, ast.Name):
        return None
    augmented = isinstance(node, ast.AugAssign)
    if augmented:
        if not isinstance(node.op, ast.Add):
            return None
        target = node.target
    else:
        if len(node.targets) != 1:
            return None
        target = node.targets[0]
    # A single cell is the whole point: a slice destination is a different lowering
    # (slice fusion) and would not give pet the affine reduction carrier we are after.
    if not (
        isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and index_axes(target) is not None
    ):
        return None
    scalar = node.value.id
    init, loop = prelude[-2], prelude[-1]
    if not (
        isinstance(init, ast.Assign)
        and len(init.targets) == 1
        and isinstance(init.targets[0], ast.Name)
        and init.targets[0].id == scalar
        and isinstance(init.value, ast.Constant)
        and init.value.value == 0.0
    ):
        return None
    if not (
        isinstance(loop, ast.For) and not loop.orelse and len(loop.body) == 1 and isinstance(loop.target, ast.Name)
    ):
        return None
    addend = accumulation_addend(loop.body[0], scalar)
    if addend is None:
        return None
    if any(isinstance(n, ast.Name) and n.id == scalar for n in ast.walk(addend)):
        return None  # self-referential accumulation -- not a plain reduction
    if not reduction_misses_target(target, loop, addend):
        return None
    store = copy.deepcopy(target)
    store.ctx = ast.Store()
    loop.body = [ast.AugAssign(target=store, op=ast.Add(), value=addend)]
    # An ``AugAssign`` target already holds the running value: zero-initialising
    # it would drop what the reduction must add to.
    if augmented:
        return prelude[:-2] + [loop]
    zero = copy.deepcopy(target)
    zero.ctx = ast.Store()
    return prelude[:-2] + [ast.Assign(targets=[zero], value=const_(0.0)), loop]


class LibNodeRewriter(ast.NodeTransformer):
    """Single-pass rewriter for the library-node registry. Consumes
    :class:`KernelIR`'s array-shape table so reduction expanders know how many
    loop levels to emit and matmul knows the M/K/N bounds. Assignments whose
    RHS matches a registered idiom are replaced in-place by the expander's
    statement list; other statements pass through.

    Propagates shapes through whole-array aliases (``x = __cb2``): the LHS
    Name inherits the RHS's shape so subsequent uses of ``x`` can be
    matmul-hoisted/scalarised.
    """

    def __init__(
        self,
        shape_table: dict[str, tuple[str, ...]],
        known_arrays: set[str] | None = None,
        local_dtypes: dict[str, str] | None = None,
        sparse: dict[str, object] | None = None,
        dim_aliases: dict[str, str] | None = None,
        native_call: Callable[[tuple[str, str], ast.Call, dict, dict], bool] | None = None,
        native_dtypes: dict[str, str] | None = None,
        blas: bool = False,
        fft_library: bool = False,
        scalar_helpers: set[str] | None = None,
        fft_library_nd: bool = False,
    ) -> None:
        self.shape_table = shape_table
        #: Kernel helpers emitted as by-value SCALAR functions. A call to one is rank 0 whatever
        #: its array arguments are, and the generic extent sizer reads an unrecognised call as
        #: elementwise -- which sizes a reduction's result like the array it reduces.
        self.scalar_helpers: set[str] = set(scalar_helpers or ())
        #: Target renders a dense 2-D float GEMM as a BLAS call. Threaded to the matmul hoister;
        #: every other matmul shape (batched, transposed, matvec, sparse, non-float) keeps its loops.
        self.blas = blas
        #: Target renders a whole-array 1-D np.fft.fft/ifft/fftn/ifftn as FFT_LIBRARY_MARKER
        #: instead of the naive O(N^2) loop. SEPARATE from ``blas`` (not reused): a target that
        #: cannot render BLAS_GEMM_MARKER (numpyto_fortran/numpyto_numba's emitters have no
        #: ``_emit_blas_gemm`` equivalent) must still be able to opt into FFT library lowering
        #: without also being handed an unrenderable BLAS marker on its next matmul.
        self.fft_library = fft_library
        #: Target also renders a batched / N-D np.fft.* as FFTN_LIBRARY_MARKER (numpyto_c only).
        self.fft_library_nd = fft_library_nd
        #: Target's "I render this numpy call myself" predicate. A call it claims is left
        #: UNEXPANDED so the emitter can use its own intrinsic -- Fortran's SUM/MAXVAL/NORM2 --
        #: instead of the loop nest every target would otherwise get. Default: claims nothing.
        self.native_call = native_call
        #: name -> element dtype, for the same predicate. Separate from ``local_dtypes``, which
        #: deliberately tags only integers: the predicate needs POSITIVE evidence of a float, and
        #: "untagged" there means "not an integer", which a boolean mask also satisfies.
        self.native_dtypes = native_dtypes or {}
        #: Dimension local -> its definition in parameter terms, threaded to the matmul hoister so
        #: a contraction dim spelled two ways still matches. See :func:`dims_agree`.
        self.dim_aliases: dict[str, str] = dim_aliases or {}
        #: Logical-name -> SparseArrayDesc, threaded to the matmul hoister so
        #: ``A @ B`` on sparse operands routes to the per-format sparse emitter.
        self.sparse: dict[str, object] = sparse or {}
        #: Names already known as signature-declared arrays (kernel
        #: parameters/outputs) -- the auto-alloc path skips these to avoid
        #: re-declaring an already-declared input.
        self.known_arrays: set[str] = known_arrays or set()
        #: Per-local dtype table, shared with ``CallHoister`` so the temp it
        #: synthesises for ``np.exp(-2j * ...)`` carries ``complex128`` through
        #: to the emit-time declaration.
        self.local_dtypes: dict[str, str] = local_dtypes if local_dtypes is not None else {}
        #: Filled-in temps the emitter must declare as local arrays (same dict
        #: shape as ``zeros_locals`` so the zeros rewriter picks them up once
        #: LibNodeRewriter has finished).
        self.matmul_temps: dict[str, tuple[str, ...]] = {}
        #: Scalar temps introduced by ``CallHoister`` (e.g. ``A[i, j] -=
        #: np.dot(...)`` -> intermediate scalar).
        self.scalar_call_temps: dict[str, bool] = {}
        #: Fresh locals introduced when an expander writes element-wise to a
        #: bare-Name LHS (linspace/arange/np.less etc). These need a C decl but
        #: the original Assign is consumed by the expander, so the emitter
        #: would otherwise miss the allocation. Mirrors ``zeros_locals`` --
        #: merged in by ``lower()``.
        self.fresh_local_allocs: dict[str, tuple[str, ...]] = {}
        self._counter = [0]

    def hoist_value(self, value: ast.expr) -> tuple[ast.expr, list[ast.stmt]]:
        # First hoist registered library-node calls, so e.g. ``A[i, j] -=
        # np.dot(A[i, :j], A[j, :j])`` becomes ``__cb1 = np.dot(...); A[i, j]
        # -= __cb1`` -- the next matmul hoist + expansion passes then handle
        # the synthetic assignment uniformly.
        call_hoister = CallHoister(
            self.shape_table,
            self.scalar_call_temps,
            self.matmul_temps,
            self._counter,
            local_dtypes=self.local_dtypes,
            dim_aliases=self.dim_aliases,
            blas=self.blas,
        )
        call_hoister.sparse = self.sparse
        value = call_hoister.visit(value)
        pre = list(call_hoister.pre_stmts)
        # Now hoist any matmul subexpressions.
        mm_hoister = MatmulHoister(
            self.shape_table,
            self.matmul_temps,
            self._counter,
            local_dtypes=self.local_dtypes,
            sparse=self.sparse,
            dim_aliases=self.dim_aliases,
            blas=self.blas,
        )
        new_value = mm_hoister.visit(value)
        pre.extend(mm_hoister.pre_stmts)
        return new_value, pre

    def update_shape_for_assign(self, target_id: str, rhs: ast.AST) -> None:
        """Update ``shape_table[target_id]`` to reflect the broadcast extent of
        ``rhs``. Mirrors the post-pipeline source-order shape resolver but
        runs inside the LibNodeRewriter pass so the hoister sees the
        then-current shape of every reassigned local. Also propagates
        ``local_dtypes`` for complex-RHS so the next statement's hoister sees
        the up-to-date dtype tag."""
        # Name = Name alias.
        if isinstance(rhs, ast.Name):
            src = self.shape_table.get(rhs.id)
            if src is not None:
                self.shape_table[target_id] = tuple(src)
            rhs_dt = self.local_dtypes.get(rhs.id)
            if rhs_dt and target_id not in self.local_dtypes:
                self.local_dtypes[target_id] = rhs_dt
            return
        # np.zeros / empty / etc constructor -- the ZerosRewriter owns the ALLOCATION, and it
        # runs in a later phase. The _like forms still have to publish their EXTENT here: the
        # source array's shape may only become known during this pass (eigh_test's ``scaled =
        # np.zeros_like(bu)``, where ``bu`` is an eigh output this same rewriter expands), and a
        # local with no shape declines every matmul it feeds -- which slice fusion then refuses.
        if (
            isinstance(rhs, ast.Call)
            and isinstance(rhs.func, ast.Attribute)
            and isinstance(rhs.func.value, ast.Name)
            and rhs.func.value.id == "np"
            and rhs.func.attr in NP_ZEROS_ALIASES
        ):
            if rhs.func.attr.endswith("_like") and rhs.args and isinstance(rhs.args[0], ast.Name):
                src = self.shape_table.get(rhs.args[0].id)
                if src is not None:
                    self.shape_table[target_id] = tuple(src)
            return
        # Shape-CHANGING ops (reshape/repeat/transpose) aren't elementwise, but
        # the generic ``iter_extent_of_`` Call branch would treat them as such
        # and return the source operand's extent -- the wrong shape for the LHS.
        if (
            isinstance(rhs, ast.Call)
            and isinstance(rhs.func, ast.Attribute)
            and isinstance(rhs.func.value, ast.Name)
            and rhs.func.value.id == "np"
            and rhs.func.attr in {"reshape", "repeat", "transpose"}
        ):
            attr = rhs.func.attr
            if attr == "reshape" and len(rhs.args) >= 2:
                # ``yv = np.reshape(y, (R**i, R, ...))`` -- the result
                # shape is the explicit newshape arg, not y's extent.
                newshape = rhs.args[1]
                toks: tuple[str, ...] | None = None
                if isinstance(newshape, (ast.Tuple, ast.List)):
                    toks = tuple(ast.unparse(e) for e in newshape.elts)
                elif isinstance(newshape, ast.Name):
                    toks = (newshape.id,)
                elif const_int(newshape) is not None:
                    toks = (str(const_int(newshape)),)
                if toks is not None:
                    # Resolve a ``-1`` placeholder (``x.reshape(batch, -1)``) to the source
                    # element count divided by the product of the other target dims.
                    neg1 = [i for i, t in enumerate(toks) if t.strip() == "-1"]
                    src = rhs.args[0]
                    src_shape = self.shape_table.get(src.id) if isinstance(src, ast.Name) else None
                    if len(neg1) == 1 and src_shape:
                        total = " * ".join(f"({t})" for t in src_shape)
                        others = [t for j, t in enumerate(toks) if j != neg1[0]]
                        denom = " * ".join(f"({t})" for t in others) if others else "1"
                        toks = tuple(f"({total}) / ({denom})" if j == neg1[0] else t for j, t in enumerate(toks))
                    self.shape_table[target_id] = toks
            # For reshape with an unparsed newshape, and for repeat/transpose,
            # the dedicated expander plus the harvested declaration shape
            # (e.g. D's rank-3 ``np.empty((R, R**i, R**(K-i-1)))`` consumed by
            # ``D[:] = np.repeat(...)``) are authoritative -- never downgrade a
            # known shape from the source operand's extent.
            return
        # BinOp/UnaryOp/IfExp/Call/Subscript -- broadcast extent. ``Subscript``
        # covers ``cols = A_col[A_row[i]:A_row[i+1]]`` (dynamic-bound slice) and
        # ``y = arr[idx]`` (fancy gather), so the next statement sees the
        # local's shape when hoisting a matmul.
        # A by-value helper call is rank 0; see :attr:`scalar_helpers`.
        if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name) and rhs.func.id in self.scalar_helpers:
            return
        if isinstance(rhs, (ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Call, ast.Subscript)):
            ext = iter_extent_of_(rhs, self.shape_table)
            if ext is not None:
                self.shape_table[target_id] = tuple(call_to_str(e) for e in ext)
            # ``ngm = qgm.shape[0]`` reads a DIMENSION -- an integer regardless of
            # the array's dtype. Type it int64 and skip the complex walk below,
            # which would otherwise see complex base Name ``qgm`` and wrongly tag
            # the scalar bound complex (vexx_k's ``_addusxx_g``/``_newdxx_g``).
            if is_shape_scalar(rhs):
                if target_id not in self.local_dtypes:
                    self.local_dtypes[target_id] = "int64"
                return
            # Complex-dtype propagation for BinOp/UnaryOp/Call: a subtree that
            # reads a complex Constant or Name promotes the LHS. ``.shape``
            # subtrees are skipped, so a dimension read off a complex array
            # (``ngm = qgm.shape[0] - 1``) isn't mis-tagged complex.
            if target_id not in self.local_dtypes and reads_complex(rhs, self.local_dtypes):
                self.local_dtypes[target_id] = "complex128"
            return

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        # Captured BEFORE any rewriting: once the RHS is hoisted its arguments are temps, and the
        # note is meant to read as the numpy the kernel was written in.
        numpy_text = ast.unparse(node.value)
        self.generic_visit(node)
        # ``D[:] = np.repeat(...)``/``D[:, :] = np.transpose(...)``: canonicalise
        # the slice-LHS-with-call form to ``D = call(...)`` so the registered
        # call expander fires -- required for stockham_fft's ``y[:] =
        # np.reshape(...)`` and ``D[:] = np.repeat(...)``.
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and is_full_slice_subscript(node.targets[0])
            and isinstance(node.value, ast.Call)
            and numpy_call_key(node.value) in NP_CALL_EXPANDERS
        ):
            node.targets[0] = ast.Name(id=node.targets[0].value.id, ctx=ast.Store())
        # ``y = np.linalg.lstsq(A, b, rcond=...)[0]`` canonicalisation: strip
        # the trailing ``[0]`` subscript on a tuple-returning call so the
        # registered call expander fires on the bare call. Only lstsq/histogram
        # for now; extend as other tuple-returners (svd, eig, etc.) land.
        if (
            len(node.targets) == 1
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Call)
            and isinstance(node.value.slice, ast.Constant)
            and node.value.slice.value == 0
        ):
            inner = node.value.value
            inner_key = numpy_call_key(inner)
            if inner_key in {("np", "linalg.lstsq"), ("np", "histogram")}:
                node.value = inner
        node.value, prelude = self.hoist_value(node.value)
        # Lower any prelude assigns that are themselves registered calls.
        prelude = self.lower_prelude_calls(prelude)
        # Reassigned local (lenet's ``x = relu(conv2d(x))`` chain): refresh shape_table[target] so
        # the NEXT statement sees the new shape. Strictly AFTER hoisting this RHS -- Python evaluates
        # the RHS against the OLD binding, and refreshing first made ``x = x @ w.T + b`` contract over
        # the RESULT's extent: out_features instead of in_features, silently dropping terms when
        # in > out and reading past the row when in < out.
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.update_shape_for_assign(node.targets[0].id, node.value)
        # Whole-array alias propagation: ``x = <Name>`` where the RHS is a Name
        # with a known shape gives ``x`` the same shape, so downstream visits
        # see ``x`` as an array. Also propagate ``local_dtypes``, else a
        # complex temp aliased to a fresh local loses its dtype tag and the
        # next call hoister synthesizes a non-complex temp.
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.shape_table
        ):
            self.shape_table[node.targets[0].id] = self.shape_table[node.value.id]
            rhs_dt = self.local_dtypes.get(node.value.id)
            if rhs_dt and node.targets[0].id not in self.local_dtypes:
                self.local_dtypes[node.targets[0].id] = rhs_dt
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0]
            if isinstance(node.value, ast.Call):
                key = numpy_call_key(node.value)
                expander = NP_CALL_EXPANDERS.get(key) if key else None
                if expander is not None and self.target_renders(key, node.value):
                    return node
                if expander is not None:
                    try:
                        expanded = call_expander(
                            expander,
                            target,
                            node.value.args,
                            node.value.keywords,
                            self.shape_table,
                            local_dtypes=self.local_dtypes,
                            fresh_local_allocs=self.fresh_local_allocs,
                            dim_aliases=self.dim_aliases,
                            library=self.fft_library,
                            library_nd=self.fft_library_nd,
                        )
                        # Linspace/arange/similar element-write expanders
                        # consume the original Assign, leaving the target
                        # dangling without a decl -- register a fresh-local
                        # allocation now so the emitter sees the shape downstream.
                        if (
                            key in ELEMENT_WRITE_EXPANDERS
                            and target.id in self.shape_table
                            and target.id not in self.known_arrays
                        ):
                            # Not gated on the local being UNREGISTERED: a temp the call-hoister
                            # already spilled is registered as an array temp but still carries no
                            # allocation SITE, and the expander has just consumed the assignment
                            # that would have carried one. max_filter's hoisted running-max scan
                            # wrote its first element through a NULL pointer that way.
                            self.fresh_local_allocs.setdefault(target.id, tuple(self.shape_table[target.id]))
                            # ...and mark the allocation SITE. Registering the shape only gets the
                            # local DECLARED; a local whose extent depends on a body-computed scalar
                            # (histogram_equalization's ``cdf = np.cumsum(hist)``, shape ``(nbins,)``
                            # with ``nbins`` assigned in the body) is declared NULL at fn-top and
                            # malloc'd at its ``__hpcagent_bench_zeros__`` marker instead. The expander
                            # consumed the Assign that would have carried that marker, so without one
                            # the buffer stays NULL and the first store segfaults. The marker is a
                            # no-op for a fn-top-malloc'd local, so emitting it unconditionally is
                            # safe -- same rationale as ``prepend_alloc_markers`` for matmul temps.
                            prelude = prelude + [alloc_marker(target.id)]
                        # ``np.arange`` over integer bounds yields an integer
                        # iota (numpy intp) -- declare the local int64 so a
                        # gather index built from it (``q = j % nx``) is
                        # integer, not the float default (fft_3d).
                        if (
                            key == ("np", "arange")
                            and target.id not in self.local_dtypes
                            and all(is_integer_expr(a, self.local_dtypes) for a in node.value.args)
                        ):
                            self.local_dtypes[target.id] = "int64"
                        tag_numpy_origin(expanded, numpy_text)
                        return prelude + expanded
                    except NotImplementedError:
                        pass
        # Partial-slice assignment target for a cumulative scan (``row_offsets[1:]
        # = np.cumsum(m_sizes)``): the full-slice case is canonicalised to a bare
        # Name above, but a shifted slice keeps its offset, routed to the
        # offset-aware cumulative expander.
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and isinstance(node.value, ast.Call)
        ):
            key = numpy_call_key(node.value)
            if key in SLICE_TARGET_EXPANDERS:
                try:
                    expanded = call_expander(
                        NP_CALL_EXPANDERS[key],
                        node.targets[0],
                        node.value.args,
                        node.value.keywords,
                        self.shape_table,
                        local_dtypes=self.local_dtypes,
                        fresh_local_allocs=self.fresh_local_allocs,
                        library=self.fft_library,
                        library_nd=self.fft_library_nd,
                    )
                    return prelude + expanded
                except NotImplementedError:
                    pass
        retargeted = retarget_scalar_accumulator(node, prelude)
        if retargeted is not None:
            return retargeted
        if prelude:
            return prelude + [node]
        return node

    def target_renders(self, key: tuple[str, str] | None, call: ast.Call) -> bool:
        """The target claims this call as its own intrinsic, so leave it unexpanded.

        The shape table goes with it: the claim has to be decidable here, because past this point
        the loop nest the call would have become no longer exists to fall back to.
        """
        if self.native_call is None or key is None:
            return False
        return self.native_call(key, call, self.shape_table, self.native_dtypes)

    def lower_prelude_calls(self, prelude: list[ast.stmt]) -> list[ast.stmt]:
        """Recursively lower any registered-call assigns inside the prelude
        that the call-hoister produced. The hoister synthesises ``__cb<n> =
        np.<op>(args)`` statements, each an Assign-to-Name with a registered
        call -- feed them through the same expander pipeline so the prelude
        emits as plain loops, not an unsupported np.<op> call.
        """
        out: list[ast.stmt] = []
        for stmt in prelude:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Call)
            ):
                key = numpy_call_key(stmt.value)
                expander = NP_CALL_EXPANDERS.get(key) if key else None
                if expander is not None and self.target_renders(key, stmt.value):
                    out.append(stmt)
                    continue
                if expander is not None:
                    try:
                        expanded = call_expander(
                            expander,
                            stmt.targets[0],
                            stmt.value.args,
                            stmt.value.keywords,
                            self.shape_table,
                            local_dtypes=self.local_dtypes,
                            fresh_local_allocs=self.fresh_local_allocs,
                            dim_aliases=self.dim_aliases,
                            library=self.fft_library,
                            library_nd=self.fft_library_nd,
                        )
                        # Same note as the direct path: the hoister split ``out = f(np.sum(a))`` into
                        # a temp assign, and it is THIS statement that becomes the loop nest.
                        tag_numpy_origin(expanded, ast.unparse(stmt.value))
                        # ...and the same auto-alloc. An expander that consumes the assign leaves
                        # its target with no allocation SITE, and a hoisted temp whose extent
                        # depends on a body-computed scalar is declared NULL at function top and
                        # malloc'd at its marker. Without one, max_filter's hoisted running-max
                        # scan wrote its first element through that NULL.
                        spilled = stmt.targets[0].id
                        if (
                            key in ELEMENT_WRITE_EXPANDERS
                            and spilled in self.shape_table
                            and spilled not in self.known_arrays
                        ):
                            self.fresh_local_allocs.setdefault(spilled, tuple(self.shape_table[spilled]))
                            out.append(alloc_marker(spilled))
                        out.extend(expanded)
                        # Integer-iota arange in the prelude (hoisted ``__cb =
                        # np.arange(1, 1025)``) keeps an int64 dtype so a derived
                        # gather index stays integer (fft_3d).
                        if (
                            key == ("np", "arange")
                            and stmt.targets[0].id not in self.local_dtypes
                            and all(is_integer_expr(a, self.local_dtypes) for a in stmt.value.args)
                        ):
                            self.local_dtypes[stmt.targets[0].id] = "int64"
                        continue
                    except NotImplementedError:
                        pass
            out.append(stmt)
        return out

    def flatten_visit_list(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        """Visit each stmt; flatten any nested lists returned by visits
        (visit_Assign can return ``[prelude..., assign]`` lists)."""
        out = []
        for s in stmts:
            r = self.visit(s)
            if isinstance(r, list):
                out.extend(r)
            else:
                out.append(r)
        return out

    def visit_If(self, node: ast.If) -> ast.AST:
        """Hoist any registered ``np.X(...)`` call inside the ``if`` test
        expression -- the common iterative-solver pattern ``if
        np.linalg.norm(r) < tol: break`` puts the call on the Compare LHS,
        where the Assign-only hoister never reaches it. Returns ``[prelude...,
        If]`` when hoisting happened."""
        node.body = self.flatten_visit_list(node.body)
        node.orelse = self.flatten_visit_list(node.orelse)
        node.test, prelude = self.hoist_value(node.test)
        prelude = self.lower_prelude_calls(prelude)
        if prelude:
            return prelude + [node]
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        node.body = self.flatten_visit_list(node.body)
        node.orelse = self.flatten_visit_list(node.orelse)
        node.test, prelude = self.hoist_value(node.test)
        prelude = self.lower_prelude_calls(prelude)
        if prelude:
            return prelude + [node]
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        self.generic_visit(node)
        node.value, prelude = self.hoist_value(node.value)
        prelude = self.lower_prelude_calls(prelude)
        retargeted = retarget_scalar_accumulator(node, prelude)
        if retargeted is not None:
            return retargeted
        if prelude:
            return prelude + [node]
        return node


#: Public name for the extent oracle. The Fortran intrinsic gate has to ask the same question
#: lowering asks -- what rank does this operand actually have -- and a backend reaching across for a
#: leading-underscore name would be reaching for something this module never promised to keep.
iter_extent_of = iter_extent_of_
