"""``_fold_slice_view_aliases`` -- fold a name bound to a partial/strided VIEW of
an array into every subscripted use, composing offsets/strides.

``x_g = padded[:, g*ipg:(g+1)*ipg]`` then ``x_g[:, :, iy0:iy0+h:s]`` (grouped
conv's per-group input slab) left a bare ``:`` reaching the C/Fortran emitter as
a value expression -- ``NotImplementedError: expression Slice`` -- across ~32
machine_learning benchmarks. :func:`_fold_subarray_aliases` already folds a
SCALAR index prefix plus dropped trailing full slices (xsbench's ``low = A[i,
j]``); this pass handles a Slice with real bounds/step at ANY axis position,
including a chain of views (``window = x_g[...]`` where ``x_g`` is itself a
view), and refuses to fold whenever it cannot prove the fold is sound.
"""
import ast

import numpy as np
from _op_oracle import run_op

from numpyto_common.lowering import _fold_slice_view_aliases
from numpyto_common.ordered import OrderedSet


def _fold(src: str, shapes) -> str:
    tree = ast.parse(src).body[0]
    _fold_slice_view_aliases(tree, {k: list(v) for k, v in shapes.items()})
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# --------------------------------------------------------------------------- #
# structural: the composed subscript / dropped alias, asserted on exact text  #
# --------------------------------------------------------------------------- #


def test_view_folds_into_a_plain_subscript():
    # ``x_g = padded[:, g*ipg:(g+1)*ipg]`` (padded rank 4) then a full 4-index
    # use -- the untouched axes pass straight through, the offset axis composes
    # to ``g * ipg + 1``. This is the exact grouped-conv shape from
    # conv2d_batch_norm_scaling.
    src = ("def f(padded, g, ipg, out):\n"
           "    x_g = padded[:, g * ipg:(g + 1) * ipg]\n"
           "    out[0] = x_g[0, 1, 2, 3]\n")
    lowered = _fold(src, {"padded": ("N", "C", "H", "W")})
    assert "x_g" not in lowered
    assert "out[0] = padded[0, g * ipg + 1, 2, 3]" in lowered


def test_further_slice_composes_offset_and_stride():
    # ``row = arr[0:20:2]`` is itself a STRIDED view; slicing it again
    # (``row[a:b:c]``) must compose both the offset (``step*inner_start``) and
    # the stride (``step_outer*step_inner``), not just reuse ``a:b:c`` verbatim.
    # ``win`` is passed bare to ``consume`` so it is not itself folded away,
    # leaving its RHS visible for inspection.
    src = ("def f(arr, a, b, c, out):\n"
           "    row = arr[0:20:2]\n"
           "    win = row[a:b:c]\n"
           "    consume(win)\n")
    lowered = _fold(src, {"arr": ("N", )})
    assert "row" not in lowered
    assert "win = arr[2 * a:2 * b:2 * c]" in lowered


def test_integer_view_index_drops_the_axis():
    # ``row = mat[i, a:b]`` -- ``i`` is an INTEGER index and drops axis 0 (numpy
    # squeeze); axis 1 stays a real bounded slice (not trimmed to nothing, so
    # :func:`_fold_subarray_aliases` -- which bails on any surviving Slice --
    # does not touch this). ``row[j]`` composes to ``mat[i, a + j]``: the
    # dropped axis's original scalar index passes through unchanged, the kept
    # axis's offset composes.
    src = ("def f(mat, i, a, b, j, out):\n"
           "    row = mat[i, a:b]\n"
           "    out[0] = row[j]\n")
    lowered = _fold(src, {"mat": ("M", "N")})
    assert "row" not in lowered
    assert "out[0] = mat[i, a + j]" in lowered


def test_implicit_trailing_dimensions_are_padded():
    # ``arr[:, a:b]`` on a 4-D array means ``arr[:, a:b, :, :]`` -- the pass must
    # pad the missing trailing axes itself (this AST predates any external
    # padding pass) so a 4-index use still composes correctly.
    src = ("def f(arr, a, b, out):\n"
           "    view = arr[:, a:b]\n"
           "    out[0] = view[0, 1, 2, 3]\n")
    lowered = _fold(src, {"arr": ("N", "C", "H", "W")})
    assert "view" not in lowered
    assert "out[0] = arr[0, a + 1, 2, 3]" in lowered


# --------------------------------------------------------------------------- #
# negative: unsound folds must NOT fire -- a correct refusal beats a wrong one #
# --------------------------------------------------------------------------- #


def test_view_written_through_is_not_folded():
    # ``view[0, 0] = 5`` writes THROUGH the alias -- it is a genuine alias, not a
    # private copy, so folding would silently redirect that store onto ``arr``
    # at the wrong composed offset instead of leaving the (correct) alias write
    # alone. The whole statement sequence must survive untouched.
    src = ("def f(arr, i, out):\n"
           "    view = arr[:, i:i + 2]\n"
           "    view[0, 0] = 5\n"
           "    out[0] = view[0, 1]\n")
    lowered = _fold(src, {"arr": ("N", "M")})
    assert lowered == ast.unparse(ast.parse(src))


def test_negative_step_view_is_not_folded():
    # ``a[::-2]`` is a numpy REVERSE: the implicit start is the LAST element, not
    # 0, which the ``start + step*index`` composition assumes. A literal negative
    # step is provably wrong to compose (unlike a symbolic step, always emitted
    # positive per this file's own convention), so it must be refused, not folded
    # with the wrong (0-based) start.
    src = ("def f(a, out):\n"
           "    b = a[::-2]\n"
           "    out[0] = b[0]\n")
    lowered = _fold(src, {"a": ("N", )})
    assert lowered == ast.unparse(ast.parse(src))


def test_base_rewritten_between_bind_and_use_is_not_folded():
    # ``arr`` is reassigned after ``view`` captures a slice of the ORIGINAL
    # ``arr`` -- composing ``view[0, 1]`` against the (rebound) name ``arr``
    # would read whatever ``something_else`` returns, not the array ``view``
    # actually sliced. Must be left alone.
    src = ("def f(arr, i, out):\n"
           "    view = arr[:, i:i + 2]\n"
           "    arr = something_else(arr)\n"
           "    out[0] = view[0, 1]\n")
    lowered = _fold(src, {"arr": ("N", "M")})
    assert lowered == ast.unparse(ast.parse(src))


# --------------------------------------------------------------------------- #
# numeric: the composed access matches numpy, through the real C backend      #
# --------------------------------------------------------------------------- #


def test_grouped_slab_view_matches_numpy_through_c():
    # A minimal grouped-conv-shaped kernel: a per-group offset view (``x_g``),
    # then a FURTHER strided sub-window of it (``window``) feeding an
    # accumulation -- the exact ``view-of-a-view`` chain conv2d_batch_norm_scaling
    # hits, run through the real C emitter and compared bit-for-bit to numpy.
    src = ("import numpy as np\n"
           "def f(x, out):\n"
           "    groups = 2\n"
           "    ipg = x.shape[1] // groups\n"
           "    for g in range(groups):\n"
           "        x_g = x[:, g * ipg:(g + 1) * ipg]\n"
           "        window = x_g[:, :, 0:4:2]\n"
           "        for n in range(x.shape[0]):\n"
           "            for c in range(ipg):\n"
           "                for k in range(2):\n"
           "                    out[n, g * ipg + c, k] = window[n, c, k]\n")
    N, C, H, groups = 2, 4, 6, 2
    ipg = C // groups
    rng = np.random.default_rng(3)
    x = rng.standard_normal((N, C, H))
    res = run_op(src,
                 "f", {"x": x}, {"out": (N, C, 2)}, {
                     "N": N,
                     "C": C,
                     "H": H
                 },
                 shapes={
                     "x": "(N,C,H)",
                     "out": "(N,C,2)"
                 },
                 backends=("c", ))
    assert res["c"] == "ok", res


def test_conv2d_batch_norm_scaling_c_matches_numpy():
    # The actual benchmark this cause was reported against -- a further check that
    # the fold is not just structurally plausible but numerically exact once
    # compiled and run.
    import numerical_oracle as no
    status = no.run_kernel("conv2d_batch_norm_scaling", preset="S", precision="fp64", seed=0, only_backends={"c"})
    assert status.get("c") == "ok", status


def test_a_folded_staging_local_is_reported_dead():
    # A folded alias whose name no longer appears anywhere is reported back so the
    # caller can drop its entry from ``zeros_locals``: the slice lifter's ``__hcall``
    # staging copy is exactly this shape, and leaving it allocated emits a
    # malloc/free pair for a buffer nothing writes or reads (and, with no use left to
    # place it against, hoisted to function top -- see
    # test_runtime_axis_dispatch.test_a_branch_allocates_only_its_own_buffers).
    src = ("def f(x, out):\n"
           "    __hcall1 = x[0:3, :]\n"
           "    out[0] = __hcall1[1, 2]\n")
    tree = ast.parse(src).body[0]
    dead = _fold_slice_view_aliases(tree, {"x": ["n", "m"], "__hcall1": ["3", "m"]})
    assert dead == {"__hcall1"}
    assert "__hcall1" not in ast.unparse(tree)


def test_a_declined_fold_reports_nothing_dead():
    # The write-through alias of test_view_written_through_is_not_folded: the name
    # stays live, so nothing may be pruned from the allocation table.
    src = ("def f(x, out):\n"
           "    v = x[0:3, :]\n"
           "    v[1, 2] = 5.0\n"
           "    out[0] = v[1, 2]\n")
    tree = ast.parse(src).body[0]
    assert _fold_slice_view_aliases(tree, {"x": ["n", "m"], "v": ["3", "m"]}) == OrderedSet()
