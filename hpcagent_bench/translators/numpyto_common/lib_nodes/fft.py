"""``np.fft.*`` lowering: naive DFT loops, or FFTW-backed markers when the target asks for ``library``."""

import ast
import copy

from hpcagent_bench.translators.numpyto_common.lib_nodes.call_args import kwarg_or_pos
from hpcagent_bench.translators.numpyto_common.lib_nodes.helpers import const_, const_or_name, name_, wrap_for_loops

__all__ = [
    "FFTN_LIBRARY_MARKER",
    "FFT_LIBRARY_MARKER",
    "NORM_KIND",
    "expand_dft_1d_library",
    "expand_dftn",
    "expand_dftn_library",
    "expand_fft",
    "expand_fftfreq",
    "expand_fftn",
    "expand_ifft",
    "expand_ifftn",
    "read_fft_axes",
    "read_fft_norm",
]

#: Pseudo-call a library-capable target's emitter renders as a whole-array 1-D DFT: FFTW3
#: (fftw_plan_dft_1d/fftwf_plan_dft_1d) for C/C++/Fortran, a numba ``objmode`` call into
#: numpy.fft for numba. Emitted ONLY when the caller asked for ``library``; DaCe's driver never
#: does, so it never sees this name and keeps the naive loop. Args are
#: ``(out, src, n, inverse_flag, norm_kind)`` -- ``norm_kind`` is 0/1/2 for backward/forward/ortho
#: (see :func:`read_fft_norm`), needed by numba's objmode path (``np.fft.fft(..., norm=...)``);
#: the C/C++/Fortran path applies norm itself via the SAME division loop the naive path uses
#: (:func:`expand_dft_1d_library`), so it ignores the arg.
FFT_LIBRARY_MARKER = "__fft_1d_library"

#: Pseudo-call the C/C++ emitter renders as ONE FFTW3 ``fftw_plan_many_dft``: an N-D transform over
#: a CONTIGUOUS block of axes that is either leading (the trailing axes are the batch, interleaved)
#: or trailing (the leading axes are the batch, one transform after another). Emitted only under
#: ``library_nd`` (numpyto_c's own lowering); every other target keeps the naive
#: O(prod(N_t)^2) loop. Args are ``(out, src, inverse_flag, norm_kind, n_transform_axes,
#: leading_flag, *extents)`` with ``extents`` the operand's full shape.
FFTN_LIBRARY_MARKER = "__fftn_library"


def read_fft_norm(args: list[ast.expr], kwargs: list[ast.keyword] | None) -> str:
    """``norm`` of an ``np.fft.*`` call -- ``'backward'`` (default)/``'forward'``/
    ``'ortho'`` -- from keyword ``norm=`` or positional slot 3. Missing /
    non-literal / ``None`` falls back to ``'backward'`` (unnormalized forward,
    ``1/prod(N)`` on the inverse)."""
    node = kwarg_or_pos(args, kwargs, 3, "norm")
    if isinstance(node, ast.Constant) and node.value in ("backward", "forward", "ortho"):
        return node.value
    return "backward"


def read_fft_axes(args: list[ast.expr], kwargs: list[ast.keyword] | None, rank: int, is_n: bool) -> list[int]:
    """Resolve the transform axes for an ``np.fft.*`` call. ``fft``/``ifft`` take
    a single ``axis`` (default last); ``fftn``/``ifftn`` take an ``axes``
    sequence (default all axes); ``fft2``/``ifft2`` are ``fftn`` over the last
    two axes. Negative axes wrap modulo ``rank``."""

    def norm_(a: int) -> int:
        # An axis outside the rank means the rank we resolved is not the operand's real rank -- vexx
        # reshapes through a ``.ndim``-conditional tuple that never folds, so the spilled operand is
        # recorded rank 1 and ``axes=(0, 1, 2)`` indexed past the iterator list. Declining is the
        # sizer's contract.
        a = int(a)
        pos = a + rank if a < 0 else a
        if not 0 <= pos < rank:
            raise NotImplementedError(f"np.fft.*: axis {a} is outside the operand rank {rank}")
        return pos

    if is_n:
        spec = kwarg_or_pos(args, kwargs, 2, "axes")
        if isinstance(spec, (ast.Tuple, ast.List)):
            return [norm_(e.value) for e in spec.elts if isinstance(e, ast.Constant)]
        return list(range(rank))  # default: every axis
    spec = kwarg_or_pos(args, kwargs, 2, "axis")
    if isinstance(spec, ast.Constant):
        return [norm_(spec.value)]
    return [rank - 1]  # default: last axis


#: ``norm=`` encoded as a small int for :data:`FFT_LIBRARY_MARKER`'s last arg -- an emitter
#: renders C code from an AST, so the norm STRING itself (not spellable as a bare identifier
#: without quoting machinery every target would need its own copy of) never needs to cross that
#: boundary; the int does.
NORM_KIND = {"backward": 0, "forward": 1, "ortho": 2}


def expand_dft_1d_library(target: ast.expr, src: ast.expr, n: str, inverse: bool, norm: str) -> list[ast.stmt]:
    """Whole-array 1-D DFT via :data:`FFT_LIBRARY_MARKER` -- O(N log N). One call does the WHOLE transform
    (norm included -- each backend's marker renderer applies it, see FFT_LIBRARY_MARKER's own
    docstring), so this returns a single statement, never wrapped in a per-element loop."""
    call = ast.Call(
        func=name_(FFT_LIBRARY_MARKER),
        args=[
            name_(target.id),
            name_(src.id),
            const_or_name(n),
            const_(1 if inverse else 0),
            const_(NORM_KIND[norm]),
        ],
        keywords=[],
    )
    return [ast.Expr(value=call)]


def expand_dftn_library(
    target: ast.expr, src: ast.expr, shape: tuple[str, ...], taxes: list[int], inverse: bool, norm: str
) -> list[ast.stmt]:
    """N-D (or batched 1-D) DFT via :data:`FFTN_LIBRARY_MARKER` -- O(P log P) per transform. The
    transform axes must be one contiguous run touching either end of the shape; the caller checks."""
    leading = taxes[0] == 0
    call = ast.Call(
        func=name_(FFTN_LIBRARY_MARKER),
        args=[
            name_(target.id),
            name_(src.id),
            const_(1 if inverse else 0),
            const_(NORM_KIND[norm]),
            const_(len(taxes)),
            const_(1 if leading else 0),
            *[const_or_name(e) for e in shape],
        ],
        keywords=[],
    )
    return [ast.Expr(value=call)]


def expand_dftn(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    inverse: bool,
    is_n: bool = True,
    kwargs: list[ast.keyword] | None = None,
    library: bool = False,
    library_nd: bool = False,
) -> list[ast.stmt]:
    """``out = np.fft.fft/ifft/fft2/ifft2/fftn/ifftn(x)`` -> a DFT.

    ``library`` (a single, whole-array 1-D transform -- ``rank == 1``) emits
    :data:`FFT_LIBRARY_MARKER` instead of the naive loop below: each C/C++/Fortran emitter
    renders it as an FFTW3 call (O(N log N)); numba's emitter renders it as an ``objmode`` call
    into ``numpy.fft`` (numba's nopython mode cannot type ``np.fft.*`` at all -- see
    ``frameworks/test.py``'s ``njit_reference`` compile-stage fallback). DaCe is untouched: its
    driver never sets ``library``, so it receives this function's naive body (see
    numpyto_c/dace_emit.py). A batched / N-D transform (``rank > 1``: fft_3d, ls3df_scf,
    vloc_psi_k_acc, bout_hasegawa_wakatani, cegterg, vexx_k) keeps the naive loop on every target:
    batching a library plan over non-transform axes is unimplemented there. ``library_nd``
    (numpyto_c only) lifts that for C/C++: a transform over a contiguous run of axes touching
    either end of the shape emits :data:`FFTN_LIBRARY_MARKER`, one ``fftw_plan_many_dft``.

    The naive path is O(prod(N_t)^2) over the transform axes -- correctness-only, kept tiny via
    the benchmark's small preset. Over transform-axis set ``T`` (remaining axes batched
    untouched), the forward transform is

        out[o] = sum_{n_t, t in T} x[src] * exp(-2j*pi * sum_{t in T} o_t n_t / N_t)

    ``src`` indexes transform axes by summation iterator ``n_t`` and batch axes
    by output iterator ``o``. The inverse uses ``+2j*pi`` and divides by
    ``prod(N_t)``. Both operands are complex; the phase numerator is float
    (``2.0 * pi * ...``) so ``/ N_t`` is real division, not C truncation."""
    if not args or not isinstance(args[0], ast.Name):
        raise NotImplementedError("np.fft.* needs a bare Name operand")
    src = args[0]
    shape = shape_table.get(src.id)
    if not shape:
        raise NotImplementedError("np.fft.*: source shape unknown")
    rank = len(shape)
    taxes = read_fft_axes(args, kwargs, rank, is_n)
    if library and rank == 1:
        return expand_dft_1d_library(target, src, shape[0], inverse, read_fft_norm(args, kwargs))
    contiguous = taxes == list(range(taxes[0], taxes[0] + len(taxes))) if taxes else False
    if library_nd and contiguous and (taxes[0] == 0 or taxes[-1] == rank - 1):
        return expand_dftn_library(target, src, tuple(shape), taxes, inverse, read_fft_norm(args, kwargs))
    # Output index iterators (one per axis); summation iterators only for the
    # transform axes. The source index uses the summation iterator on transform
    # axes and the (fixed) output iterator on batch axes.
    o_iters = [f"__fk{i}" for i in range(rank)]
    n_iters = {t: f"__fn{t}" for t in taxes}
    o_slot = name_(o_iters[0]) if rank == 1 else ast.Tuple(elts=[name_(o) for o in o_iters], ctx=ast.Load())
    src_idx = [(name_(n_iters[d]) if d in taxes else name_(o_iters[d])) for d in range(rank)]
    src_slot = src_idx[0] if rank == 1 else ast.Tuple(elts=src_idx, ctx=ast.Load())
    out_k = ast.Subscript(value=name_(target.id), slice=o_slot, ctx=ast.Store())
    out_k_load = ast.Subscript(value=name_(target.id), slice=o_slot, ctx=ast.Load())
    # Emit pi as a numeric literal (backend-agnostic): this expander runs after
    # MathRewriter, so an ``np.pi`` Attribute would reach the emitter unlowered.
    pi = const_(3.141592653589793)
    # total phase = sum_{t in T} (2.0 * pi * o_t * n_t) / N_t
    phase = None
    for t in taxes:
        num = ast.BinOp(
            left=ast.BinOp(
                left=ast.BinOp(left=const_(2.0), op=ast.Mult(), right=pi), op=ast.Mult(), right=name_(o_iters[t])
            ),
            op=ast.Mult(),
            right=name_(n_iters[t]),
        )
        term = ast.BinOp(left=num, op=ast.Div(), right=const_or_name(shape[t]))
        phase = term if phase is None else ast.BinOp(left=phase, op=ast.Add(), right=term)
    sign = const_(1j) if inverse else const_(-1j)
    # Emit the already-lowered bare ``exp`` (not ``np.exp``): this expander runs
    # after ``MathRewriter`` (np.exp -> exp), so ``np.exp`` here would reach the
    # emitter unlowered. The emitter routes ``exp`` of a complex operand to ``cexp``.
    twiddle = ast.Call(func=name_("exp"), args=[ast.BinOp(left=sign, op=ast.Mult(), right=phase)], keywords=[])
    src_n = ast.Subscript(value=name_(src.id), slice=src_slot, ctx=ast.Load())
    acc = ast.AugAssign(target=out_k, op=ast.Add(), value=ast.BinOp(left=src_n, op=ast.Mult(), right=twiddle))
    inner = wrap_for_loops([n_iters[t] for t in taxes], [shape[t] for t in taxes], [acc])
    body: list[ast.stmt] = [ast.Assign(targets=[out_k], value=const_(0j))] + inner
    # numpy ``norm``: 'backward' (default) puts ``1/prod(N)`` on the INVERSE; 'forward' puts it on
    # the FORWARD; 'ortho' puts ``1/sqrt(prod(N))`` on BOTH. Divide when this direction carries it.
    norm = read_fft_norm(args, kwargs)
    if norm == "ortho" or ((norm == "forward") != inverse):
        denom = None
        for t in taxes:
            ext = const_or_name(shape[t])
            denom = ext if denom is None else ast.BinOp(left=denom, op=ast.Mult(), right=ext)
        if norm == "ortho":
            # ``prod(N_t)`` is an integer extent product; ``sqrt`` of an integer is
            # rejected by gfortran (must be REAL or COMPLEX). C/C++ promote silently;
            # coerce portably via ``* 1.0`` (Fortran emits the float literal at
            # double precision, so nothing is lost).
            denom = ast.BinOp(left=denom, op=ast.Mult(), right=const_(1.0))
            denom = ast.Call(func=name_("sqrt"), args=[denom], keywords=[])
        body.append(ast.Assign(targets=[out_k], value=ast.BinOp(left=out_k_load, op=ast.Div(), right=denom)))
    return wrap_for_loops(o_iters, list(shape), body)


def expand_fftn(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    library: bool = False,
    library_nd: bool = False,
) -> list[ast.stmt]:
    return expand_dftn(
        target, args, shape_table, inverse=False, is_n=True, kwargs=kwargs, library=library, library_nd=library_nd
    )


def expand_ifftn(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    library: bool = False,
    library_nd: bool = False,
) -> list[ast.stmt]:
    return expand_dftn(
        target, args, shape_table, inverse=True, is_n=True, kwargs=kwargs, library=library, library_nd=library_nd
    )


def expand_fft(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    library: bool = False,
    library_nd: bool = False,
) -> list[ast.stmt]:
    # 1-D DFT along a single ``axis`` (default last); for a 1-D input == fftn.
    return expand_dftn(
        target, args, shape_table, inverse=False, is_n=False, kwargs=kwargs, library=library, library_nd=library_nd
    )


def expand_ifft(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
    library: bool = False,
    library_nd: bool = False,
) -> list[ast.stmt]:
    return expand_dftn(
        target, args, shape_table, inverse=True, is_n=False, kwargs=kwargs, library=library, library_nd=library_nd
    )


def expand_fftfreq(
    target: ast.expr,
    args: list[ast.expr],
    shape_table: dict[str, tuple[str, ...]],
    kwargs: list[ast.keyword] | None = None,
) -> list[ast.stmt]:
    """``np.fft.fftfreq(n, d=1.0)`` -> DFT sample frequencies (length ``n``):
    ``out[i] = (i if i <= (n - 1) // 2 else i - n) / (n * d)``. Indices up to
    ``(n - 1) // 2`` are non-negative frequencies, the rest wrap negative,
    scaled by sample spacing ``d``. ``n * d`` is real, so ``/`` is real division.
    Matches ``numpy.fft.fftfreq`` for even and odd ``n``."""
    if not args:
        raise NotImplementedError("np.fft.fftfreq needs the sample count n")
    n = args[0]
    d_node = kwarg_or_pos(args, kwargs, 1, "d")
    if d_node is None:
        d_node = const_(1.0)
    it = "__ff"
    half = ast.BinOp(
        left=ast.BinOp(left=copy.deepcopy(n), op=ast.Sub(), right=const_(1)), op=ast.FloorDiv(), right=const_(2)
    )
    numer = ast.IfExp(
        test=ast.Compare(left=name_(it), ops=[ast.LtE()], comparators=[half]),
        body=name_(it),
        orelse=ast.BinOp(left=name_(it), op=ast.Sub(), right=copy.deepcopy(n)),
    )
    denom = ast.BinOp(left=copy.deepcopy(n), op=ast.Mult(), right=copy.deepcopy(d_node))
    body = [
        ast.Assign(
            targets=[ast.Subscript(value=name_(target.id), slice=name_(it), ctx=ast.Store())],
            value=ast.BinOp(left=numer, op=ast.Div(), right=denom),
        )
    ]
    return wrap_for_loops([it], [copy.deepcopy(n)], body)
