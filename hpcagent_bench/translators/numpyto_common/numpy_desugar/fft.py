"""``np.fft.*`` lowered to explicit DFT loops."""

import ast
import copy
from collections.abc import Callable

from hpcagent_bench.translators.numpyto_common import dtypes
from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import np_submodule_attr
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import expr_rank


def fft_axes(fattr: str, call: ast.Call, rank: int):
    """``(transform_axes, inverse)`` for an ``np.fft.<fattr>`` call, or
    ``(None, inverse)`` when the axis spec is non-constant (caller bails, leaving
    the call verbatim). ``fft``/``ifft`` take one ``axis`` (default last);
    ``fftn``/``ifftn`` an ``axes`` sequence (default ALL); ``fft2``/``ifft2`` the
    last two axes. Negative axes wrap modulo ``rank``."""
    kwargs = {k.arg: k.value for k in call.keywords}
    inverse = fattr.startswith("i")
    base = fattr[1:] if inverse else fattr
    if base == "fft":
        ax = kwargs.get("axis") or (call.args[1] if len(call.args) > 1 else None)
        if ax is None:
            return [rank - 1], inverse
        if isinstance(ax, ast.Constant) and isinstance(ax.value, int):
            return [ax.value % rank], inverse
        return None, inverse
    if base == "fft2":
        return ([rank - 2, rank - 1] if rank >= 2 else None), inverse
    if base == "fftn":
        axes = kwargs.get("axes") or (call.args[1] if len(call.args) > 1 else None)
        if axes is None:
            return list(range(rank)), inverse
        if isinstance(axes, (ast.Tuple, ast.List)) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, int) for e in axes.elts
        ):
            return [e.value % rank for e in axes.elts], inverse
        return None, inverse
    return None, inverse


def fft_inline_stmts(
    tname: str, sname: str, taxes: list[int], rank: int, inverse: bool, ctr: int, alloc: bool, real_dtype: str
) -> list[ast.stmt]:
    """Source statements computing ``np.fft.*`` into ``tname`` (shape == source)
    as a naive DFT loop nest -- the same O(prod(N_t)^2) transform the C/Fortran
    backends lower, but as plain numpy (``np.exp`` of a complex phase, complex
    ``+=``) that numba njit-compiles and pythran template-instantiates. Output
    indices iterate every axis; summation iterators only the transform axes
    ``taxes`` (batch axes ride the output iterator). Inverse uses ``+1j`` and
    divides by ``prod(N_t)``. ``alloc`` allocates ``tname`` (bare-Name target);
    a ``tname[:]`` slice target writes the existing buffer in place.

    ``real_dtype`` (the transform's complex dtype's real half, e.g. ``float64`` for
    ``complex128``) casts the phase divisor: dace constant-folds the phase's leading
    ``1j`` into the product chain and codegens a raw ``complex/int64`` division, which
    dace/runtime/include/dace/complex.h has no ``operator/`` for; a same-precision REAL
    divisor resolves to the native ``std::complex`` ``operator/`` instead. Must track the
    transform's actual precision -- this is emitted as source text, so a hardcoded fp64
    cast would silently double the working precision of an fp32 build."""
    p = f"__ft{ctr}"
    sign = "1j" if inverse else "-1j"
    # Bind each axis size to an int local first. pythran otherwise forward-
    # substitutes ``sname.shape[i]`` into ``range(...)`` over a lazy numpy_expr
    # source and fails template type inference; an int local pins it to ``long``.
    d = [f"{p}_d{i}" for i in range(rank)]
    lines: list[str] = [f"{d[i]} = {sname}.shape[{i}]" for i in range(rank)]
    if alloc:
        # The transform's OWN complex width, from the same real_dtype the phase divisor uses --
        # a hardcoded complex128 doubles an fp32 port's working precision and turns the store
        # back into its complex64 target into a narrowing copy.
        lines.append(f"{tname} = np.zeros(({', '.join(d)},), np.{dtypes.complex_dtype_for(real_dtype)})")
    o = [f"{p}_k{i}" for i in range(rank)]
    ind = ""
    for i in range(rank):
        lines.append(f"{ind}for {o[i]} in range({d[i]}):")
        ind += "    "
    oidx = ", ".join(o)
    lines.append(f"{ind}{tname}[{oidx}] = 0j")
    n = {t: f"{p}_n{t}" for t in taxes}
    cind = ind
    for t in taxes:
        lines.append(f"{cind}for {n[t]} in range({d[t]}):")
        cind += "    "
    terms = [f"(2.0 * 3.141592653589793 * {o[t]} * {n[t]} / np.{real_dtype}({d[t]}))" for t in taxes]
    phase = " + ".join(terms)
    sidx = ", ".join((n[ax] if ax in taxes else o[ax]) for ax in range(rank))
    lines.append(f"{cind}{tname}[{oidx}] += {sname}[{sidx}] * np.exp({sign} * ({phase}))")
    if inverse:
        denom = " * ".join(d[t] for t in taxes)
        lines.append(f"{ind}{tname}[{oidx}] = {tname}[{oidx}] / ({denom})")
    return ast.parse("\n".join(lines)).body


def fft_real_dtype(sname: str, tname: str, array_dtypes: dict[str, str]) -> str:
    """Real dtype backing the FFT phase divisor's cast (:func:`fft_inline_stmts`): the
    REAL half of the transform's OWN complex dtype, read off whichever of the source /
    target array names is in ``array_dtypes``. Falls back to float64 only when neither
    resolves (a hoisted, non-Name transform argument) -- the same fp64-when-unknown rule
    :func:`fd_step` uses, not a default the normal (Name-argument) path takes."""
    dtype = array_dtypes.get(sname) or array_dtypes.get(tname)
    if dtype is not None:
        try:
            return dtypes.real_component_dtype(dtype)
        except KeyError:
            pass
    return "float64"


class FftInline(ast.NodeTransformer):
    """Replace ``out = np.fft.fft/ifft/fftn/ifftn/fft2/ifft2(x)`` (and the
    ``out[:] =`` slice-assign form) with a naive-DFT loop nest. numba supports no
    ``np.fft`` at all; pythran supports 1-D ``fft``/``ifft`` but not N-D
    ``fftn``/``ifftn`` -- lowering all variants uniformly keeps one code path
    (the loop DFT matches numpy to ~1e-15 at any realistic size). A non-Name
    argument (``ifftn(u1 * np.exp(...))``) is hoisted to a temp first so the loop
    body can index it; a non-constant axis spec leaves the call verbatim.

    A transform that is one OPERAND of a larger right-hand side
    (``np.fft.ifftn(g) * nnr``, QE's unscaled backward transform) is hoisted the
    same way and then lowered, because the whole point is that no ``np.fft`` call
    survives into the emitted program. dace is not lowered at all
    (:data:`NATIVE_FFT_BACKENDS`): its FFT library nodes are the real transform."""

    def __init__(self, ranks: dict[str, int], array_dtypes: dict[str, str]) -> None:
        self.ranks = ranks
        self.array_dtypes = array_dtypes
        self.changed = False
        self._ctr = 0

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        fattr = np_submodule_attr(node.value, "fft")
        if fattr is None or not node.value.args:
            return self.hoist_operand_transform(node)
        tgt = node.targets[0]
        if isinstance(tgt, ast.Name):
            tname, alloc = tgt.id, True
        elif (
            isinstance(tgt, ast.Subscript)
            and isinstance(tgt.value, ast.Name)
            and isinstance(tgt.slice, ast.Slice)
            and tgt.slice.lower is None
            and tgt.slice.upper is None
        ):
            tname, alloc = tgt.value.id, False
        else:
            return node
        arg = node.value.args[0]
        rank = expr_rank(arg, self.ranks)
        if rank is None or rank < 1:
            return node
        taxes, inverse = fft_axes(fattr, node.value, rank)
        if not taxes:
            return node
        pre: list[ast.stmt] = []
        if isinstance(arg, ast.Name):
            sname = arg.id
        else:
            sname = f"__fti{self._ctr}"
            pre = ast.parse(f"{sname} = {ast.unparse(arg)}").body
        real_dtype = fft_real_dtype(sname, tname, self.array_dtypes)
        stmts = fft_inline_stmts(tname, sname, taxes, rank, inverse, self._ctr, alloc, real_dtype)
        self._ctr += 1
        self.changed = True
        return pre + stmts

    def hoist_operand_transform(self, node: ast.Assign) -> ast.Assign | list[ast.stmt]:
        """``out = <expr with np.fft.X(a) inside>`` -> bind each transform to its own temp first.

        :meth:`visit_Assign` matches a BARE transform call, so the emitted program kept the call
        whenever the reference wrapped it -- the QE normalization ``np.fft.ifftn(g) * nnr`` is one.
        Each hoisted binding is re-fed through :meth:`visit_Assign`, which lowers it to the loop
        DFT, so the statement list this returns carries no ``np.fft`` call either.
        """
        found: list[tuple[str, ast.Call]] = []

        def bind(call: ast.Call) -> ast.Name | None:
            if np_submodule_attr(call, "fft") is None or not call.args:
                return None
            rank = expr_rank(call.args[0], self.ranks)
            if rank is None or rank < 1:
                return None
            name = f"__fth{len(found)}_{self._ctr}"
            self.ranks[name] = rank
            found.append((name, call))
            return ast.Name(id=name, ctx=ast.Load())

        value = SubstituteFftCalls(bind).visit(node.value)
        if not found:
            return node
        out: list[ast.stmt] = []
        for name, call in found:
            binding = ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=call)
            ast.copy_location(binding, node)
            ast.fix_missing_locations(binding)
            lowered = self.visit_Assign(binding)
            out.extend(lowered if isinstance(lowered, list) else [lowered])
        node.value = value
        ast.fix_missing_locations(node)
        out.append(node)
        return out

    def visit_Return(self, node: ast.Return) -> ast.Return | list[ast.stmt]:
        """``return <expr carrying np.fft.X(a)>`` -> bind the value, lower the binding, return the name.

        The Assign forms above are where the loop DFT is built, so a transform the reference returns
        directly (cegterg's ``return np.fft.ifftn(...).reshape(...)``) survived into a numba body, which
        has no ``np.fft`` at all. A binding the Assign lowering declines leaves the return untouched."""
        if node.value is None or not any(np_submodule_attr(n, "fft") is not None for n in ast.walk(node.value)):
            return node
        name = f"__fret{self._ctr}"
        binding = ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=copy.deepcopy(node.value))
        ast.copy_location(binding, node)
        ast.fix_missing_locations(binding)
        lowered = self.visit_Assign(binding)
        stmts = lowered if isinstance(lowered, list) else [lowered]
        if any(np_submodule_attr(n, "fft") is not None for s in stmts for n in ast.walk(s)):
            return node
        return [*stmts, ast.copy_location(ast.Return(value=ast.Name(id=name, ctx=ast.Load())), node)]


class SubstituteFftCalls(ast.NodeTransformer):
    """Replace every ``np.fft.*`` call ``visitor`` accepts with the Name it returns."""

    def __init__(self, visitor: Callable[[ast.Call], ast.Name | None]) -> None:
        self.visitor = visitor

    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        return self.visitor(node) or node


#: Backends that compile ``np.fft.*`` NATIVELY, so :class:`FftInline` leaves the call in place. dace's
#: frontend binds its FFT/IFFT library nodes, which the canonicalize finalize lowers to FFTW3 on the
#: CPU and cuFFT/hipFFT on the GPU; the inlined loop DFT is O(N^2) per axis and never finishes at
#: benchmark sizes (fft_1d's N ~ 7e7).
NATIVE_FFT_BACKENDS = frozenset({"dace"})
