# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Chained-subscript collapse, config-flag typing, and slice-fusion gather offset.

These are the general translator features the QE exact-exchange kernel
(``vexx_all_paths``) needs to emit natively for EVERY configuration (the
translation is orthogonal to the config flags -- one binary handles all of them):

  * ``tabxx_qr[ia][:, ijtoh[ih, jh]]`` -- a CHAINED subscript (a scalar row index,
    then a full slice + a scalar column). numpy basic indexing associates, so it
    collapses to a single ``tabxx_qr[ia, :, col]`` that the shape harvest /
    scalarizer / dot-product operand path handle uniformly.
  * ``okvan`` / ``tqr`` / ... -- boolean CONFIG FLAGS. A bool-valued preset scalar
    is a ``bool`` parameter (C ``bool`` / Fortran ``logical``), not an integer
    size symbol, so the ``if okvan and tqr:`` conditionals type-check.
  * ``big_result[ip*n:ip*n+n] -= rg[nlg]`` (the noncolin ``npol=2`` finalise) --
    a slice-assign into a NON-zero-start destination gathering through a
    length-matched index array: the gather index must be read at the LOCAL offset
    ``si0 - ip*n``, not the absolute ``si0`` (which runs off ``nlg``).
"""

import ast
import dataclasses
import json
import pathlib
from collections.abc import Mapping

import numpy as np
import pytest

from hpcagent_bench.translators.numpyto_common import lowering
from hpcagent_bench.translators.numpyto_common.frontend import collect_bool_preset_names, parse_kernel
from hpcagent_bench.translators.numpyto_common.lib_nodes import reads_complex
from hpcagent_bench.translators.numpyto_common.lowering import ChainedSubscriptFlattener, lower
from tests.translators.op_oracle import bench_info_ as synthesize_bench_info
from tests.translators.op_oracle import run_op

ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")


def ok_(res):
    assert any(v == "ok" for v in res.values()), f"every backend skipped; the comparison never ran: {res}"
    return all(v == "ok" or v.startswith("skip") for v in res.values()), res


def expr_(s: str) -> ast.AST:
    return ast.parse(s, mode="eval").body


def unparse_(node: ast.AST) -> str:
    return ast.unparse(ast.fix_missing_locations(node))


# structural: chained subscript collapses to a single subscript


def collapse(expr: str, shapes) -> str:
    # The pre-harvest configuration: every base axis the rank names is spelled out.
    node = ast.parse(expr, mode="eval").body
    table = {k: tuple(v) for k, v in shapes.items()}
    new = ChainedSubscriptFlattener(table, explicit_trailing_axes=True).visit(node)
    return ast.unparse(ast.fix_missing_locations(new))


def test_collapse_scalar_row_then_slice_column() -> None:
    # ``tabxx_qr[ia][:, c]`` -- scalar row consumes axis 0; the slice + column
    # apply to the remaining axes -> ``tabxx_qr[ia, :, c]``.
    assert collapse("A[ia][:, c]", {"A": ("nat", "K", "nij")}) == "A[ia, :, c]"


def test_collapse_slice_then_trailing_scalar() -> None:
    # ``becxx[:, j, k][m]`` -- the trailing index selects the surviving full-slice
    # axis 0 -> ``becxx[m, j, k]``.
    assert collapse("A[:, j, k][m]", {"A": ("nkb", "nb", "nks")}) == "A[m, j, k]"


def test_collapse_of_an_unsized_base_continues_on_the_axes_after_the_inner() -> None:
    # No shape for the base: the outer entries still land on the axes right after the scalar
    # inner, whatever the rank, so ``A[i][:, c]`` is ``A[i, :, c]``.
    assert collapse("A[i][:, c]", {}) == "A[i, :, c]"


def test_collapse_rebases_an_index_through_a_partial_inner_slice() -> None:
    # ``A[1:3][j]`` is ``A[1 + j]``: the bounded slice shifts the origin of the axis ``j`` indexes.
    assert collapse("A[1:3][j]", {"A": ("n", "m")}) == "A[1 + j, :]"


def test_collapse_indexes_into_a_fancy_inner_index() -> None:
    # ``A[idx][j]`` with ``idx`` an ARRAY is a fancy GATHER (== ``A[idx[j]]``), not
    # a scalar associate -- ``A[idx, j]`` would read a column instead of row ``idx[j]``.
    assert collapse("A[idx][j]", {"A": ("n", "m"), "idx": ("k",)}) == "A[idx[j], :]"


def test_collapse_keeps_scalar_inner_when_a_sibling_name_is_an_array() -> None:
    # ``A[i][j]`` -- ``i`` is a plain scalar (absent from the shape table) even
    # though some OTHER name ``idx`` is an array: the scalar associate still fires.
    assert collapse("A[i][j]", {"A": ("n", "m"), "idx": ("k",)}) == "A[i, j]"


def test_collapse_expands_an_inner_ellipsis_against_the_base_rank() -> None:
    # ``A[..., j]`` on a rank-3 base is ``A[:, :, j]``, so ``[k]`` lands on axis 0.
    assert collapse("A[..., j][k]", {"A": ("n", "m", "p")}) == "A[k, :, j]"


# ---- chained index arrays: composed into one subscript, two-step where numpy transposes ----

ABI_BACKENDS = ("c", "fortran")


def test_an_outer_row_and_column_land_on_the_gathered_row_and_the_base_column() -> None:
    # ``A[idx][j, k]``: ``j`` picks gathered row ``idx[j]``, ``k`` the base column after it.
    A = np.arange(20.0).reshape(4, 5)
    idx = np.array([3, 0, 2])
    assert A[idx][1, 4] == A[idx[1], 4]
    node = ast.parse("A[idx][j, k]", mode="eval").body
    new = ChainedSubscriptFlattener({"A": ("n", "m"), "idx": ("p",)}).visit(node)
    assert unparse_(new) == "A[idx[j], k]"


def test_a_gather_then_a_row_index_reads_the_gathered_row() -> None:
    # ``A[idx][j]`` is row ``idx[j]`` of A. The last flattening phase used to emit ``A[idx, j]``, a
    # column read that compiles on a square A and returns wrong numbers.
    src = "import numpy as np\ndef f(A, idx, out):\n    for j in range(idx.shape[0]):\n        out[j, :] = A[idx][j]\n"
    M = 4
    A = np.random.default_rng(3).standard_normal((M, M))
    idx = np.array([3, 1, 0, 2], dtype=np.int64)
    res = run_op(
        src,
        "f",
        {"A": A, "idx": idx},
        {"out": (M, M)},
        {"M": M},
        shapes={"A": "(M,M)", "idx": "(M,)", "out": "(M,M)"},
        backends=ABI_BACKENDS,
    )
    ok, r = ok_(res)
    assert ok, r


def test_a_gather_through_a_partial_slice_offsets_the_index_array() -> None:
    # ``a[1:5][jdx]`` is ``a[1 + jdx]``. Every flattening phase used to decline it, and the chain
    # reached the emitter as an index with more axes than ``a`` has.
    src = "import numpy as np\ndef f(a, jdx, out):\n    out[:] = a[1:5][jdx]\n"
    N, P = 6, 4
    a = np.random.default_rng(4).standard_normal(N)
    jdx = np.array([3, 1, 0, 2], dtype=np.int64)
    res = run_op(
        src,
        "f",
        {"a": a, "jdx": jdx},
        {"out": (P,)},
        {"N": N, "P": P},
        shapes={"a": "(N,)", "jdx": "(P,)", "out": "(P,)"},
        backends=ABI_BACKENDS,
    )
    ok, r = ok_(res)
    assert ok, r


def lowered_source(
    src: str,
    arrays: Mapping[str, np.ndarray],
    outputs: Mapping[str, tuple[int, ...]],
    shapes: Mapping[str, str],
    syms: Mapping[str, int],
    workdir: pathlib.Path,
) -> str:
    """Kernel ``gather`` after lowering, as source, through the real file-reading entry point."""
    kernel = workdir / "gather_numpy.py"
    kernel.write_text(src)
    info = workdir / "bench_info.json"
    dtypes = {name: str(array.dtype) for name, array in arrays.items() if array.dtype.kind == "i"}
    info.write_text(
        json.dumps(synthesize_bench_info("gather", list(arrays), list(outputs), dict(shapes), dict(syms), dtypes))
    )
    return ast.unparse(lower(parse_kernel(kernel, info)).tree)


@dataclasses.dataclass(frozen=True, slots=True)
class ChainCase:
    """One kernel body over named arrays, the statement lowering must emit, and the output numpy writes."""

    body: str
    arrays: dict[str, np.ndarray]
    shapes: dict[str, str]
    out: tuple[int, ...]
    lowered: str


def check_chain_case(case: ChainCase, syms: Mapping[str, int], workdir: pathlib.Path) -> None:
    src = f"import numpy as np\ndef gather({', '.join(case.arrays)}, out):\n    {case.body}\n"
    shapes = {**case.shapes, "out": "(" + ",".join(str(extent) for extent in case.out) + ")"}
    assert case.lowered in lowered_source(src, case.arrays, {"out": case.out}, shapes, syms, workdir)
    dtypes = {name: str(array.dtype) for name, array in case.arrays.items() if array.dtype.kind == "i"}
    res = run_op(
        src,
        "gather",
        dict(case.arrays),
        {"out": case.out},
        dict(syms),
        shapes=shapes,
        backends=ABI_BACKENDS,
        dtypes=dtypes,
    )
    ok, r = ok_(res)
    assert ok, r


VIEW_RNG = np.random.default_rng(5)
VIEW_A = VIEW_RNG.standard_normal((3, 5, 7))
VIEW_B = VIEW_RNG.standard_normal((3, 5, 7, 6))
VIEW_IDX = np.array([0, 2], dtype=np.int64)
VIEW_PAIR = np.array([[6], [1], [3], [0]], dtype=np.int64)
VIEW_JDX = np.array([5, 0, 2, 4, 1], dtype=np.int64)
VIEW_SYMS = {"F": 3, "X": 5, "Y": 7, "Z": 6, "P": 2, "Q": 4, "R": 5}
A_IDX_SHAPES = {"A": "(F,X,Y)", "idx": "(P,)"}

#: ``(chained, naive flat)`` numpy shapes: each chain below would transpose if merged into one subscript.
VIEW_PREMISES = {
    "integer-before-slice": (VIEW_A[2][:3, VIEW_IDX].shape, VIEW_A[2, :3, VIEW_IDX].shape),
    "loop-integer-partial-slice": (VIEW_A[1][1:4, VIEW_IDX].shape, VIEW_A[1, 1:4, VIEW_IDX].shape),
    "integer-after-slice": (VIEW_B[:, 2][1:3, :, VIEW_IDX].shape, VIEW_B[1:3, 2, :, VIEW_IDX].shape),
    "two-index-arrays": (VIEW_B[2][:3, VIEW_PAIR, VIEW_JDX].shape, VIEW_B[2, :3, VIEW_PAIR, VIEW_JDX].shape),
    "length-one-slice": (VIEW_A[2][1:2, VIEW_IDX].shape, VIEW_A[2, 1:2, VIEW_IDX].shape),
    "newaxis-before": (VIEW_A[2][None, :3, VIEW_IDX].shape, VIEW_A[2, None, :3, VIEW_IDX].shape),
    "newaxis-after": (VIEW_A[2][:3, VIEW_IDX, None].shape, VIEW_A[2, :3, VIEW_IDX, None].shape),
}

VIEW_GATHERS = {
    "integer-before-slice": ChainCase(
        "out[:, :] = A[2][:3, idx]",
        {"A": VIEW_A, "idx": VIEW_IDX},
        A_IDX_SHAPES,
        (3, 2),
        "out[si0, si1] = A[2, si0, idx[si1]]",
    ),
    "loop-integer-partial-slice": ChainCase(
        "for i in range(3):\n        out[i, :, :] = A[i][1:4, idx]",
        {"A": VIEW_A, "idx": VIEW_IDX},
        A_IDX_SHAPES,
        (3, 3, 2),
        "out[i, si1, si2] = A[i, si1 + 1, idx[si2]]",
    ),
    "integer-after-slice": ChainCase(
        "out[:, :, :] = B[:, 2][1:3, :, idx]",
        {"B": VIEW_B, "idx": VIEW_IDX},
        {"B": "(F,X,Y,Z)", "idx": "(P,)"},
        (2, 7, 2),
        "out[si0, si1, si2] = B[si0 + 1, 2, si1, idx[si2]]",
    ),
    "two-index-arrays": ChainCase(
        "out[:, :, :] = B[2][:3, pair, jdx]",
        {"B": VIEW_B, "pair": VIEW_PAIR, "jdx": VIEW_JDX},
        {"B": "(F,X,Y,Z)", "pair": "(Q,1)", "jdx": "(R,)"},
        (3, 4, 5),
        "out[si0, si1, si2] = B[2, si0, pair[si1, 0], jdx[si2]]",
    ),
    "length-one-slice": ChainCase(
        "out[:, :] = A[2][1:2, idx]",
        {"A": VIEW_A, "idx": VIEW_IDX},
        A_IDX_SHAPES,
        (1, 2),
        "out[si0, si1] = A[2, 0 + 1, idx[si1]]",
    ),
    "newaxis-before": ChainCase(
        "out[:, :, :] = A[2][None, :3, idx]",
        {"A": VIEW_A, "idx": VIEW_IDX},
        A_IDX_SHAPES,
        (1, 3, 2),
        "out[si0, si1, si2] = A[2, si1, idx[si2]]",
    ),
    "newaxis-after": ChainCase(
        "out[:, :, :] = A[2][:3, idx, None]",
        {"A": VIEW_A, "idx": VIEW_IDX},
        A_IDX_SHAPES,
        (3, 2, 1),
        "out[si0, si1, si2] = A[2, si0, idx[si1]]",
    ),
}


@pytest.mark.parametrize("name", list(VIEW_GATHERS))
def test_an_index_array_split_from_a_scalar_by_a_slice_keeps_its_axis_order(name: str, tmp_path: pathlib.Path) -> None:
    # ``A[2][:3, idx]`` is (3, P), the flat ``A[2, :3, idx]`` is (P, 3). The pre-harvest phase used to emit
    # the flat form (SIG11 in C); the two-step form then reached the emitter as a 5-axis index. Lowering now
    # reads the view's axes at the statement iterators and composes them onto the base, one element read.
    chained, flat = VIEW_PREMISES[name]
    assert chained != flat
    check_chain_case(VIEW_GATHERS[name], VIEW_SYMS, tmp_path)


def test_adjacent_index_arrays_behind_a_slice_share_one_broadcast_block(tmp_path: pathlib.Path) -> None:
    # ``C[:3, pair, jdx]``: pair (Q, 1) and jdx (R,) broadcast to ONE (Q, R) block after the slice axis.
    # Each array used to take iterators of its own, so ``jdx`` was left unindexed.
    case = ChainCase(
        "out[:, :, :] = C[:3, pair, jdx]",
        {"C": VIEW_B[0], "pair": VIEW_PAIR, "jdx": VIEW_JDX},
        {"C": "(X,Y,Z)", "pair": "(Q,1)", "jdx": "(R,)"},
        VIEW_B[0][:3, VIEW_PAIR, VIEW_JDX].shape,
        "out[si0, si1, si2] = C[si0, pair[si1, 0], jdx[si2]]",
    )
    check_chain_case(case, VIEW_SYMS, tmp_path)


GATHER_RNG = np.random.default_rng(11)
COUNTS = np.array([1.0, 4.0, 2.0])
MAT = np.array([2, 0, 1, 2], dtype=np.int64)
LIM = np.arange(5, dtype=np.float64)
TABLE = GATHER_RNG.standard_normal((3, 2, 5))
GATHER_SYMS = {"M": 3, "P": 4, "J": 5, "X": 2, "Y": 5, "Q": 3}
COUNT_SHAPES = {"counts": "(M,)", "mat": "(P,)", "lim": "(J,)"}

NEWAXIS_GATHERS = {
    "xsbench-two-statements": ChainCase(
        "valid = lim[None, :] < counts[mat][:, None]\n    out[:, :] = np.where(valid, conc, 0.0)",
        {"counts": COUNTS, "mat": MAT, "lim": LIM, "conc": GATHER_RNG.standard_normal((4, 5))},
        {**COUNT_SHAPES, "conc": "(P,J)"},
        (4, 5),
        "valid[si0, si1] = lim[si1] < counts[mat[si0]]",
    ),
    "rank-3-base": ChainCase(
        "out[:, :, :, :] = B[mat][:, None] + 1.0",
        {"B": TABLE, "mat": MAT},
        {"B": "(M,X,Y)", "mat": "(P,)"},
        (4, 1, 2, 5),
        "out[si0, si1, si2, si3] = B[mat[si0], si2, si3] + 1.0",
    ),
    "rank-3-base-on-where": ChainCase(
        "out[:, :, :, :] = np.where(B[mat][:, None] > 0.0, 1.0, -1.0)",
        {"B": TABLE, "mat": MAT},
        {"B": "(M,X,Y)", "mat": "(P,)"},
        (4, 1, 2, 5),
        "1.0 if B[mat[__r0], __r2, __r3] > 0.0 else -1.0",
    ),
    "inside-a-2d-index-on-where": ChainCase(
        "out[:, :, :] = np.where(x[aj][:, None, :] > 0.5, 1.0, 0.0)",
        {"x": np.array([0.2, 0.9, 0.6]), "aj": np.array([[2, 0, 1], [1, 1, 0], [0, 2, 2], [2, 1, 0]], dtype=np.int64)},
        {"x": "(M,)", "aj": "(P,Q)"},
        (4, 1, 3),
        "1.0 if x[aj[__r0, __r2]] > 0.5 else 0.0",
    ),
    "inside-the-index-on-where": ChainCase(
        "out[:, :] = np.where(lim[None, :] < counts[mat[:, None]], 1.0, 0.0)",
        {"counts": COUNTS, "mat": MAT, "lim": LIM},
        COUNT_SHAPES,
        (4, 5),
        "1.0 if lim[__r1] < counts[mat[__r0]] else 0.0",
    ),
    "newaxis-before-the-gathered-axis": ChainCase(
        "out[:, :] = counts[mat][None, :] + lim[:, None]",
        {"counts": COUNTS, "mat": MAT, "lim": LIM},
        COUNT_SHAPES,
        (5, 4),
        "out[si0, si1] = counts[mat[si1]] + lim[si0]",
    ),
    "newaxes-separating-advanced-entries": ChainCase(
        "out[:, :, :] = G[None, mat, None, 2]",
        {"G": GATHER_RNG.standard_normal((3, 5)), "mat": MAT},
        {"G": "(M,J)", "mat": "(P,)"},
        (4, 1, 1),
        "out[si0, si1, si2] = G[mat[si0], 2]",
    ),
}


@pytest.mark.parametrize("name", list(NEWAXIS_GATHERS))
def test_a_newaxis_beside_a_gathered_axis_reads_each_gathered_row(name: str, tmp_path: pathlib.Path) -> None:
    # xsbench's ``num_nucs[mat][:, None]`` now flattens to ``num_nucs[mat, None]``. A newaxis in a slice-free
    # gather inserts a unit axis and reads no source axis, and one inside an index array is a result axis of
    # that array; both used to shift the gather onto the column iterator.
    check_chain_case(NEWAXIS_GATHERS[name], GATHER_SYMS, tmp_path)


# pure: a boolean preset value is a config-flag name (typed bool)


def test_bool_preset_names_picks_boolean_flags_not_int_symbols() -> None:
    params = {
        "S": {"N": 6, "okvan": False, "tqr": False, "negrp": 1},
        "fuzzed": {"N": [6, 16], "okvan": {"set": [False, True]}, "negrp": {"set": [1, 2]}},
    }
    # ``okvan`` is boolean everywhere it is pinned; ``N`` / ``negrp`` are integers.
    assert collect_bool_preset_names(params) == {"okvan", "tqr"}


# numeric: bit-close to numpy across every backend


def test_chained_column_dot_matches_numpy() -> None:
    # ``np.dot(A[ia][:, 1], v[box[ia]])`` -- the collapsed chained column dotted
    # with a materialised-box gather (vexx_k ``_newdxx_r``).
    src = (
        "import numpy as np\n"
        "def f(A, box, v, out):\n"
        "    nat = A.shape[0]\n"
        "    for ia in range(nat):\n"
        "        bx = box[ia]\n"
        "        col = A[ia][:, 1]\n"
        "        out[ia] = np.dot(col, v[bx])\n"
    )
    nat, K, ncol, N = 3, 4, 2, 10
    rng = np.random.default_rng(0)
    A = rng.standard_normal((nat, K, ncol))
    box = np.stack([np.sort(rng.choice(N, K, replace=False)) for unused in range(nat)]).astype(np.int64)
    v = rng.standard_normal(N)
    res = run_op(
        src,
        "f",
        {"A": A, "box": box, "v": v},
        {"out": (nat,)},
        {"nat": nat, "K": K, "ncol": ncol, "N": N},
        shapes={"A": "(nat,K,ncol)", "box": "(nat,K)", "v": "(N,)", "out": "(nat,)"},
        backends=ALL,
    )
    ok, r = ok_(res)
    assert ok, r


def test_slice_assign_gather_offset_matches_numpy() -> None:
    # ``out[k*n:k*n+n] -= r[idx]`` for k in 0,1 -- the length-``n`` gather index
    # ``idx`` must be read at the LOCAL slice offset, so k=1 does not run off it
    # (the vexx_k noncolin npol=2 finalise OOB).
    src = (
        "import numpy as np\n"
        "def f(r, idx, out):\n"
        "    n = idx.shape[0]\n"
        "    for k in range(2):\n"
        "        out[k * n:k * n + n] -= r[idx]\n"
    )
    n, M, P = 5, 12, 10
    rng = np.random.default_rng(1)
    r = rng.standard_normal(M)
    idx = rng.integers(0, M, size=n).astype(np.int64)
    res = run_op(
        src,
        "f",
        {"r": r, "idx": idx},
        {"out": (P,)},
        {"n": n, "M": M, "P": P},
        shapes={"r": "(M,)", "idx": "(n,)", "out": "(P,)"},
        backends=ALL,
    )
    ok, rr = ok_(res)
    assert ok, rr


def test_shape_of_complex_array_is_integer_bound() -> None:
    # ``n = z.shape[0]`` reads a DIMENSION (int) even though ``z`` is complex --
    # a complex-typed loop bound would make ``for i in range(n)`` a type error
    # (vexx_k ``ngm = qgm.shape[0]``).
    src = (
        "import numpy as np\n"
        "def f(z, out):\n"
        "    n = z.shape[0]\n"
        "    for i in range(n):\n"
        "        out[i] = z[i].real + z[i].imag\n"
    )
    N = 6
    rng = np.random.default_rng(2)
    z = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex128)
    res = run_op(src, "f", {"z": z}, {"out": (N,)}, {"N": N}, shapes={"z": "(N,)", "out": "(N,)"}, backends=ALL)
    ok, r = ok_(res)
    assert ok, r


# .shape / len / .size resolve to the SYMBOLIC dims that were provided
# ``ShapeMidExpressionRewriter`` rewrites an inline extent read against the array's
# declared shape tuple, so ``A.shape[k]`` emits the k-th shape SYMBOL (not a native
# ``.shape`` the C / Fortran backends cannot lower).


def rewrite_shape(expr: str, shapes) -> str:
    node = expr_(expr)
    new = lowering.ShapeMidExpressionRewriter({k: tuple(v) for k, v in shapes.items()}).visit(node)
    return unparse_(new)


def test_shape_index_maps_to_declared_symbol() -> None:
    # ``A.shape[k]`` -> the k-th token of the provided shape tuple.
    assert rewrite_shape("A.shape[0]", {"A": ("nat", "K", "nij")}) == "nat"
    assert rewrite_shape("A.shape[1]", {"A": ("nat", "K", "nij")}) == "K"
    assert rewrite_shape("A.shape[2]", {"A": ("nat", "K", "nij")}) == "nij"


def test_shape_index_in_range_bound_maps_to_symbol() -> None:
    # The common ``for i in range(A.shape[0])`` -> ``range(nat)``.
    assert rewrite_shape("range(A.shape[0])", {"A": ("nat", "K")}) == "range(nat)"


def test_shape_index_numeric_dim_is_a_literal() -> None:
    # A concrete (pinned) dimension resolves to an integer literal, not a symbol.
    assert rewrite_shape("A.shape[1]", {"A": ("N", "3")}) == "3"


def test_bare_shape_maps_to_symbol_tuple() -> None:
    # ``A.shape`` (e.g. ``np.zeros(A.shape)``) -> the full symbol tuple.
    assert rewrite_shape("A.shape", {"A": ("N", "M")}) == "(N, M)"


def test_len_and_size_map_to_symbols() -> None:
    # ``len(A)`` == ``A.shape[0]``; ``A.size`` == product of the shape symbols.
    assert rewrite_shape("len(A)", {"A": ("N", "M")}) == "N"
    assert rewrite_shape("A.size", {"A": ("N", "M")}) == "N * M"


def test_shape_of_unknown_array_is_left_untouched() -> None:
    # No declared shape -> cannot resolve -> the read is left as-is.
    assert rewrite_shape("A.shape[0]", {}) == "A.shape[0]"


# .shape read is INTEGER even off a complex array (skips the complex walk)


def test_reads_complex_skips_shape_subtree_bare_and_compound() -> None:
    # ``qgm`` is complex, but a ``.shape`` read yields integer DIMENSIONS. The
    # complex-dtype predicate must skip the ``.shape`` subtree for the bare form
    # AND the compound arithmetic form, else the integer bound is tagged complex
    # (vexx_k ``ngm = qgm.shape[0]`` / ``qgm.shape[0] - 1``).
    dt = {"qgm": "complex128"}
    assert reads_complex(expr_("qgm.shape[0]"), dt) is False
    assert reads_complex(expr_("qgm.shape[0] - 1"), dt) is False
    assert reads_complex(expr_("2 * qgm.shape[0] + 1"), dt) is False


def test_reads_complex_still_detects_a_genuine_complex_value_read() -> None:
    # A real VALUE read of the complex array (not its shape) is still complex.
    dt = {"qgm": "complex128"}
    assert reads_complex(expr_("qgm[i] + 1.0"), dt) is True
    assert reads_complex(expr_("qgm.shape[0] * qgm[0]"), dt) is True
    assert reads_complex(expr_("(1 + 2j)"), dt) is True


# trailing implicit-axis pad reads at the LOCAL slice offset (iter - start)


def pad_trailing(rhs_expr: str, start, source_shape):
    # Drive SliceToScalarRewriter for a single-slice LHS assignment whose slice
    # starts at ``start``, on a partial-scalar RHS of a higher-rank source.
    iv = ast.Name(id="si", ctx=ast.Load())
    lhs_slice = ast.Slice(lower=(None if start == 0 else ast.Constant(start)), upper=None, step=None)
    rw = lowering.SliceToScalarRewriter(
        array_shapes={"dH": tuple(source_shape)},
        iter_vars=[iv],
        lhs_ranges=[(ast.Constant(start), ast.Constant(start + 3))],
        lhs_name="out",
        lhs_dims=[lhs_slice],
    )
    return unparse_(rw.visit(expr_(rhs_expr)))


def test_trailing_pad_reads_local_offset_for_nonzero_start() -> None:
    # ``out[2:2+M] = dH[a, b]`` -- dH rank 3, RHS names 2 axes; the implicit
    # trailing axis is padded with the LHS slice iter at its LOCAL position
    # ``si - 2``, so a non-zero-start destination spans dH's length-M axis from 0
    # (the absolute ``si`` would run off the axis).
    assert pad_trailing("dH[a, b]", 2, ("A", "B", "M")) == "dH[a, b, si - 2]"


def test_trailing_pad_no_offset_for_zero_start() -> None:
    # A zero-start slice needs no correction: local offset == absolute index.
    assert pad_trailing("dH[a, b]", 0, ("A", "B", "M")) == "dH[a, b, si]"


# iter-start offset copies the shared start node (no AST aliasing)


def test_iter_minus_start_copies_shared_start_node() -> None:
    # ``start`` is the SAME node object as the loop-header ``range`` lower bound, so
    # ``iter_minus_start`` must embed a COPY -- else one mutable subtree lives in two
    # tree positions and a later in-place rewrite of the loop bound corrupts the
    # gather offset (and vice versa).
    start = expr_("ip * n")  # a shared BinOp: the slice lower bound / loop start
    iv = ast.Name(id="si", ctx=ast.Load())
    out = lowering.SliceToScalarRewriter.iter_minus_start(iv, start)
    assert unparse_(out) == "si - ip * n"
    embedded = [b for b in ast.walk(out) if isinstance(b, ast.BinOp) and isinstance(b.op, ast.Mult)]
    assert embedded and embedded[0] is not start  # a copy, not the aliased node


def test_iter_minus_start_zero_start_is_bare_fresh_iter() -> None:
    # start == 0 -> bare iter (no offset), and a FRESH Name (not the passed object).
    iv = ast.Name(id="si", ctx=ast.Load())
    out = lowering.SliceToScalarRewriter.iter_minus_start(iv, ast.Constant(value=0))
    assert unparse_(out) == "si"
    assert out is not iv
