"""In-memory representation: the Python AST + a layout side-table.

The IR follows the same pattern as :mod:`affinepython.ir` -- the AST
is the canonical form (round-trips via :func:`ast.unparse` for free),
and three small dataclasses carry the layout / shape information the
backends need to emit typed C signatures and resolve subscripts.

The design is deliberately reusable: when ``NumpyToDaCe`` and friends
land, this module hoists to ``numpyto_common.ir`` unchanged. Until
then, NumpyToC consumes it locally.
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from numpyto_common import dtypes

#: Float / complex dtypes a precision sweep remaps; everything else
#: (int / uint / bool) keeps its dtype so a mixed kernel's index arrays
#: stay integer when the floating precision changes.
_FLOAT_DTYPES = frozenset({"float64", "float32", "float16", "float128", "double", "float8_e4m3", "float8_e5m2"})
_COMPLEX_DTYPES = frozenset({"complex128", "complex64", "complex256"})
_COMPLEX_FOR_FLOAT = {"float64": "complex128", "float32": "complex64",
                      "float16": "complex64", "float128": "complex256"}


def _apply_precision(dtype: str, precision: Optional[str]) -> str:
    """Selectively remap a single dtype to the target floating ``precision``.

    Float and complex dtypes become ``precision`` (and its complex
    counterpart); int / uint / bool are left unchanged. A blanket remap
    would turn an int32 index array into ``float`` (s4114's ``ip``).
    """
    if not precision:
        return dtype
    if dtype in _FLOAT_DTYPES:
        return precision
    if dtype in _COMPLEX_DTYPES:
        return _COMPLEX_FOR_FLOAT.get(precision, dtype)
    return dtype


def apply_precision(kir: "KernelIR", precision: Optional[str]) -> "KernelIR":
    """Set the kernel's floating precision ON THE IR, so every emitter
    just reads ``arr.dtype`` -- no per-emit override. Remaps float/complex
    array, scalar and local dtypes to ``precision`` (ints untouched) and
    records :attr:`KernelIR.float_precision` so the emitter's default for a
    temp not listed in ``local_dtypes`` (e.g. a matmul scratch) matches.

    ``precision`` of ``None``/empty is a no-op (each dtype keeps its
    declared value -- the natural fp64 path).

    The spelling is normalized to the canonical registry key first, so the
    enum-style ``fp8_e4m3`` and the numpy-style ``float8_e4m3`` are ONE leg and
    every downstream ``dtype.startswith("float")`` float test still fires.

    Recurses into :attr:`KernelIR.helpers`. Each helper is a full KernelIR with its
    own dtype tables that the emitter writes as its own native function signature, so
    narrowing only the caller leaves every callee declared at its stale fp64 and the
    emitted call does not typecheck (``passed REAL(4) to REAL(8)``; ``expected 'const
    double * restrict' but argument is of type 'const float *'``). Recursion, not a flat
    loop: helpers can nest inside helpers, and each one needs its own float_precision so
    the emitter's default for an unlisted temp matches inside the helper body too.
    """
    if not precision:
        return kir
    precision = dtypes.canonical(precision)
    for helper in kir.helpers:
        apply_precision(helper, precision)
    for arr in kir.arrays:
        arr.dtype = _apply_precision(arr.dtype, precision)
    for sca in kir.scalars:
        sca.dtype = _apply_precision(sca.dtype, precision)
    for name, dt in list(kir.local_dtypes.items()):
        kir.local_dtypes[name] = _apply_precision(dt, precision)
    kir.float_precision = precision
    return kir


@dataclass
class SymbolDesc:
    """One scalar shape / scale parameter (always integer-typed).

    :ivar name: source-level name (``"N"``, ``"LEN_1D"``, ``"ITERATIONS"``).
    """
    name: str


@dataclass
class ArrayDesc:
    """One array parameter.

    :ivar name: source-level name.
    :ivar dtype: numpy-style dtype tag (e.g. ``"float64"``, ``"int32"``);
        each backend maps it to its own concrete type.
    :ivar shape: list of source-level symbol names making up the
        logical shape, slow-to-fast. ``("N", "N")`` for a square
        matrix; ``("LEN_1D",)`` for a vector. Entries may also be
        integer literals (``("1",)`` for a 1-element output buffer).
    :ivar is_output: ``True`` when the array appears on the LHS of an
        assignment in the kernel body. Drives ``const`` qualification.
    """
    name: str
    dtype: str
    shape: Tuple[str, ...]
    is_output: bool = False


@dataclass
class ScalarDesc:
    """One scalar (non-shape) parameter -- e.g. ``alpha`` in GEMM."""
    name: str
    dtype: str
    is_output: bool = False


@dataclass
class SparseArrayDesc:
    """A logical sparse array that expands into physical buffer arrays.

    The kernel body references the logical name (``A`` in ``A @ B``);
    the matmul hoister consults this descriptor to emit the per-format
    loop nest reading the physical buffers. The buffers themselves are
    registered as ordinary :class:`ArrayDesc` entries so they appear in
    the C / Fortran signature.

    :ivar name: logical array name as it appears in the kernel body.
    :ivar format: one of ``optarena.spec.SUPPORTED_SPARSE_FORMATS``.
    :ivar logical_shape: dense-equivalent shape (``("NI", "NK")``).
    :ivar buffers: ``{role: physical_name}`` -- e.g.
        ``{"indptr": "A_indptr", "indices": "A_indices", "data": "A_data"}``.
    """
    name: str
    format: str
    logical_shape: Tuple[str, ...]
    buffers: Dict[str, str]


@dataclass
class KernelIR:
    """The full kernel: function-def AST + parameter tables.

    Field order matches the way every emitter walks the program --
    backend code never needs to ask anything else about the source.

    :ivar tree: the function-def's AST node (the body is what we
        lower and emit).
    :ivar kernel_name: function-symbol the backends use (``"s111"``).
    :ivar input_args: ordered parameter names exactly as
        :data:`bench_info.input_args` lists them. The C signature is
        emitted in this order so positional ctypes calls line up.
    :ivar symbols: per-name lookup for shape / iteration-count params.
    :ivar arrays: per-name lookup for array params.
    :ivar scalars: per-name lookup for non-shape, non-array scalars.
    :ivar source_path: pathlib.Path of the input file, for diagnostics.
    """
    tree: ast.FunctionDef
    kernel_name: str
    #: Stable short name used for file paths and symbol prefixes.
    #: Equal to ``kernel_name`` for Foundation kernels (which all
    #: have distinct ``func_name``) and to ``bench_info.short_name``
    #: for legacy kernels (where every kernel has ``func_name = 'kernel'``
    #: and the short name disambiguates).
    short_name: str = ""
    input_args: List[str] = field(default_factory=list)
    symbols: List[SymbolDesc] = field(default_factory=list)
    arrays: List[ArrayDesc] = field(default_factory=list)
    scalars: List[ScalarDesc] = field(default_factory=list)
    source_path: Optional[str] = None
    #: Logical-name -> SparseArrayDesc for arrays carrying a sparse
    #: layout (CSR / CSC / ...). Empty for dense kernels. Consumed by
    #: the matmul hoister to route ``A @ B`` through the sparse path.
    sparse: Dict[str, "SparseArrayDesc"] = field(default_factory=dict)
    #: One sub-:class:`KernelIR` per top-level helper that is CALLED in the kernel
    #: body but could not be inlined (an early ``return`` / recursion). Built by
    #: :func:`parse_kernel` (params inferred from the call site) and lowered by
    #: :func:`lower`; every emitter emits each as its own native function, so the
    #: early ``return`` is just a native ``return``.
    helpers: List["KernelIR"] = field(default_factory=list)
    #: When this KernelIR IS a helper: how its value comes back --
    #: ``None`` for a void/in-place helper, ``"scalar"`` for a by-value return
    #: (the return dtype is the sole :attr:`scalars` entry marked ``is_output``),
    #: or the out-parameter array name for an array return.
    return_kind: Optional[str] = None
    # ------------------------------------------------------------------
    # Lowering side-tables. Populated by :func:`numpyto_common.lowering.lower`
    # (empty on a freshly-parsed IR) and consumed by every emitter. They live
    # on the IR -- not monkey-patched onto ``tree.__dict__`` -- so a reader is a
    # typed field access (``kir.local_dtypes``) with a sane empty default, never
    # a ``getattr(kir.tree, ...)`` with a hand-repeated fallback.
    # ------------------------------------------------------------------
    #: Loop-index / tuple-unpack integer locals the emitter must declare ``int``.
    int_locals: List[str] = field(default_factory=list)
    #: Local-name -> numpy dtype tag for body locals (``"complex128"``, ``"int64"``,
    #: ``"bool_"``) the signature does not carry. Drives temp declarations.
    local_dtypes: Dict[str, str] = field(default_factory=dict)
    #: Local-name -> shape tuple of token strings for harvested / lifted local
    #: arrays (``np.zeros`` temps, matmul scratch, slice-fusion lifts).
    zeros_locals: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    #: Local-name -> constructor kind (``"zeros"`` / ``"ones"`` / ``"empty"`` /
    #: ...) so the emitter re-initialises a local that aliases an output buffer.
    zeros_fills: Dict[str, str] = field(default_factory=dict)
    #: Scalar call-hoist temp names (declared as plain float locals by the emit
    #: walker's implicit-local logic; kept for completeness / diagnostics).
    scalar_call_temps: List[str] = field(default_factory=list)
    #: Local-name -> FIFO of per-reassignment shapes (SSA-versioned locals whose
    #: broadcast extent changes between writes), consumed in source order at emit.
    reassign_shapes: Dict[str, List[Tuple[str, ...]]] = field(default_factory=dict)
    #: Floating precision the sweep pinned (``"float32"`` / ...); the emitter's
    #: default dtype for a temp not in ``local_dtypes``. ``None`` = natural fp64.
    float_precision: Optional[str] = None

    def param_order(self) -> List[str]:
        """Return the argument names in **ABI order**.

        The ABI convention (one source of truth for the emitted C / Fortran
        signature *and* the binding JSON the harness calls through): all
        **references** (array / pointer params) sorted alphabetically, then
        all **scalars** (everything passed by value -- the integer shape
        ``symbols`` and the value ``scalars``) sorted alphabetically.

        This deliberately ignores ``input_args`` for *ordering* (it still
        defines membership): the order is derived purely from each param's
        ABI kind so it is stable and caller-independent. The caller side
        (:meth:`Framework.call_args` for the C/C++ backends) reads the same
        binding order, so the positional ctypes call always lines up.
        """
        refs = sorted(a.name for a in self.arrays)
        scalars = sorted([s.name for s in self.symbols] + [s.name for s in self.scalars])
        return refs + scalars
