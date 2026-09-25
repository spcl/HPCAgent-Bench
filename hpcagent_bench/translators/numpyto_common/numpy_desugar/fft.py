"""``np.fft.*`` lowered to explicit DFT loops."""

import ast
import copy
from collections.abc import Callable

from hpcagent_bench.translators.numpyto_common import dtypes
from hpcagent_bench.translators.numpyto_common.numpy_desugar.common import np_submodule_attr
from hpcagent_bench.translators.numpyto_common.numpy_desugar.ranks import expr_rank


def fft_axes(fattr: str, call: ast.Call, rank: int):
    """``(transform_axes, inverse)`` for an ``np.fft.<fattr>`` call; axes are ``None`` when non-constant.

    ``fft``: one ``axis`` (default last); ``fftn``: ``axes`` (default all); ``fft2``: last two.
    Negative axes wrap modulo ``rank``."""
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
    """Statements computing ``np.fft.*`` of ``sname`` into ``tname`` as a naive DFT loop nest.

    Output indices iterate every axis, summation indices only ``taxes``; inverse uses ``+1j`` and
    divides by ``prod(N_t)``. ``alloc`` allocates ``tname``; otherwise the existing buffer is written.

    ``real_dtype`` (real half of the transform's complex dtype) casts the phase divisor: dace folds the
    leading ``1j`` into the product and emits a ``complex/int64`` division that its complex.h has no
    ``operator/`` for, while a same-precision real divisor resolves. It must match the transform's
    precision, or an fp32 build silently computes in fp64."""
    p = f"__ft{ctr}"
    sign = "1j" if inverse else "-1j"
    # Int locals for the extents: pythran otherwise forward-substitutes ``sname.shape[i]`` of a lazy
    # numpy_expr into ``range(...)`` and fails type inference.
    d = [f"{p}_d{i}" for i in range(rank)]
    lines: list[str] = [f"{d[i]} = {sname}.shape[{i}]" for i in range(rank)]
    if alloc:
        # The transform's own complex width; complex128 here would widen an fp32 port.
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
    """Real half of the source's or target's complex dtype; float64 when neither is known."""
    dtype = array_dtypes.get(sname) or array_dtypes.get(tname)
    if dtype is not None:
        try:
            return dtypes.real_component_dtype(dtype)
        except KeyError:
            pass
    return "float64"


class FftInline(ast.NodeTransformer):
    """``out = np.fft.fft/ifft/fftn/ifftn/fft2/ifft2(x)`` (or ``out[:] = ...``) -> a naive-DFT loop nest.

    numba has no ``np.fft``; pythran lacks N-D ``fftn``, so all variants take one code path. A non-Name
    argument, or a transform nested in a larger expression or a ``return``, is bound to a temp first;
    a non-constant axis spec leaves the call verbatim. Not applied to :data:`NATIVE_FFT_BACKENDS`."""

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
        """``out = f(np.fft.X(a))`` -> ``__fth = np.fft.X(a); out = f(__fth)``, lowering each binding."""
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
        """``return <expr with np.fft.X(a)>`` -> bind the value, lower the binding, return the name.

        Left untouched when the lowering leaves any ``np.fft`` call in the binding."""
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


#: Backends that compile ``np.fft.*`` natively, so :class:`FftInline` leaves the call in place. dace binds
#: FFT library nodes (FFTW3 / cuFFT / hipFFT); the O(N^2) loop DFT would not finish at benchmark sizes.
NATIVE_FFT_BACKENDS = frozenset({"dace"})
