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

#: Float / complex dtypes a precision sweep remaps; everything else
#: (int / uint / bool) keeps its dtype so a mixed kernel's index arrays
#: stay integer when the floating precision changes.
_FLOAT_DTYPES = frozenset({"float64", "float32", "float16", "float128", "double"})
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
    records ``tree.float_precision`` so the emitter's default for a temp
    not listed in ``local_dtypes`` (e.g. a matmul scratch) matches.

    ``precision`` of ``None``/empty is a no-op (each dtype keeps its
    declared value -- the natural fp64 path).
    """
    if not precision:
        return kir
    for arr in kir.arrays:
        arr.dtype = _apply_precision(arr.dtype, precision)
    for sca in kir.scalars:
        sca.dtype = _apply_precision(sca.dtype, precision)
    local_dtypes = vars(kir.tree).get("local_dtypes")
    if local_dtypes:
        for name, dt in list(local_dtypes.items()):
            local_dtypes[name] = _apply_precision(dt, precision)
    kir.tree.float_precision = precision   # type: ignore[attr-defined]
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

    def is_array(self, name: str) -> bool:
        return any(a.name == name for a in self.arrays)

    def is_symbol(self, name: str) -> bool:
        return any(s.name == name for s in self.symbols)

    def find_array(self, name: str) -> Optional[ArrayDesc]:
        for a in self.arrays:
            if a.name == name:
                return a
        return None
