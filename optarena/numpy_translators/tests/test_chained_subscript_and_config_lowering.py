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

import numpy as np
from _op_oracle import run_op

from numpyto_common.frontend import _collect_bool_preset_names
from numpyto_common.lowering import _CollapseChainedSubscripts

_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")


def _ok(res):
    return all(v == "ok" or v.startswith("skip") for v in res.values()), res


# ---- structural: chained subscript collapses to a single subscript ----


def _collapse(expr: str, shapes) -> str:
    node = ast.parse(expr, mode="eval").body
    new = _CollapseChainedSubscripts({k: tuple(v) for k, v in shapes.items()}).visit(node)
    return ast.unparse(ast.fix_missing_locations(new))


def test_collapse_scalar_row_then_slice_column():
    # ``tabxx_qr[ia][:, c]`` -- scalar row consumes axis 0; the slice + column
    # apply to the remaining axes -> ``tabxx_qr[ia, :, c]``.
    assert _collapse("A[ia][:, c]", {"A": ("nat", "K", "nij")}) == "A[ia, :, c]"


def test_collapse_slice_then_trailing_scalar():
    # ``becxx[:, j, k][m]`` -- the trailing index selects the surviving full-slice
    # axis 0 -> ``becxx[m, j, k]``.
    assert _collapse("A[:, j, k][m]", {"A": ("nkb", "nb", "nks")}) == "A[m, j, k]"


def test_collapse_bails_on_unknown_base_shape():
    # No shape for the base array -> left chained (cannot resolve trailing axes).
    assert _collapse("A[i][:, c]", {}) == "A[i][:, c]"


def test_collapse_bails_on_partial_inner_slice():
    # A bounded inner slice is not a plain associate -> left untouched.
    assert _collapse("A[1:3][j]", {"A": ("n", "m")}) == "A[1:3][j]"


# ---- pure: a boolean preset value is a config-flag name (typed bool) ----


def test_bool_preset_names_picks_boolean_flags_not_int_symbols():
    params = {
        "S": {"N": 6, "okvan": False, "tqr": False, "negrp": 1},
        "fuzzed": {"N": [6, 16], "okvan": {"set": [False, True]}, "negrp": {"set": [1, 2]}},
    }
    # ``okvan`` is boolean everywhere it is pinned; ``N`` / ``negrp`` are integers.
    assert _collect_bool_preset_names(params) == {"okvan", "tqr"}


# ---- numeric: bit-close to numpy across every backend ----


def test_chained_column_dot_matches_numpy():
    # ``np.dot(A[ia][:, 1], v[box[ia]])`` -- the collapsed chained column dotted
    # with a materialised-box gather (vexx_k ``_newdxx_r``).
    src = ("import numpy as np\n"
           "def f(A, box, v, out):\n"
           "    nat = A.shape[0]\n"
           "    for ia in range(nat):\n"
           "        bx = box[ia]\n"
           "        col = A[ia][:, 1]\n"
           "        out[ia] = np.dot(col, v[bx])\n")
    nat, K, ncol, N = 3, 4, 2, 10
    rng = np.random.default_rng(0)
    A = rng.standard_normal((nat, K, ncol))
    box = np.stack([np.sort(rng.choice(N, K, replace=False)) for _ in range(nat)]).astype(np.int64)
    v = rng.standard_normal(N)
    res = run_op(src, "f", {"A": A, "box": box, "v": v}, {"out": (nat, )},
                 {"nat": nat, "K": K, "ncol": ncol, "N": N},
                 shapes={"A": "(nat,K,ncol)", "box": "(nat,K)", "v": "(N,)", "out": "(nat,)"},
                 backends=_ALL)
    ok, r = _ok(res)
    assert ok, r


def test_slice_assign_gather_offset_matches_numpy():
    # ``out[k*n:k*n+n] -= r[idx]`` for k in 0,1 -- the length-``n`` gather index
    # ``idx`` must be read at the LOCAL slice offset, so k=1 does not run off it
    # (the vexx_k noncolin npol=2 finalise OOB).
    src = ("import numpy as np\n"
           "def f(r, idx, out):\n"
           "    n = idx.shape[0]\n"
           "    for k in range(2):\n"
           "        out[k * n:k * n + n] -= r[idx]\n")
    n, M, P = 5, 12, 10
    rng = np.random.default_rng(1)
    r = rng.standard_normal(M)
    idx = rng.integers(0, M, size=n).astype(np.int64)
    res = run_op(src, "f", {"r": r, "idx": idx}, {"out": (P, )},
                 {"n": n, "M": M, "P": P},
                 shapes={"r": "(M,)", "idx": "(n,)", "out": "(P,)"}, backends=_ALL)
    ok, rr = _ok(res)
    assert ok, rr


def test_shape_of_complex_array_is_integer_bound():
    # ``n = z.shape[0]`` reads a DIMENSION (int) even though ``z`` is complex --
    # a complex-typed loop bound would make ``for i in range(n)`` a type error
    # (vexx_k ``ngm = qgm.shape[0]``).
    src = ("import numpy as np\n"
           "def f(z, out):\n"
           "    n = z.shape[0]\n"
           "    for i in range(n):\n"
           "        out[i] = z[i].real + z[i].imag\n")
    N = 6
    rng = np.random.default_rng(2)
    z = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex128)
    res = run_op(src, "f", {"z": z}, {"out": (N, )},
                 {"N": N}, shapes={"z": "(N,)", "out": "(N,)"}, backends=_ALL)
    ok, r = _ok(res)
    assert ok, r
