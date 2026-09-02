"""``np.diff`` is the first difference, so it lowers to ``A[..., 1:, ...] - A[..., :-1, ...]``.

It reached the native emitters unexpanded and died there with ``call to np.diff not supported`` --
a refusal, not a miscompile, but one that pushed kernels into writing the subtraction by hand.
cp2k_density_matrix_trs4 hit it on ``np.repeat(np.arange(n), np.diff(row_ptr))``, the standard CSR
row-length idiom.

The axis shortens by exactly one, which is the part a shape-preserving assumption gets wrong: the
output is NOT the operand's shape, so ``diff`` also has to leave ``ELEMENTWISE_SHAPE_OPS``. Only the
first difference expands -- ``n > 1`` needs a temporary per stage, and ``prepend=``/``append=`` are a
concatenate the caller can spell.
"""

import ast

import numpy as np
import pytest
from _op_oracle import run_op

from numpyto_common.lib_nodes import NP_CALL_EXPANDERS, expand_diff

_NATIVE = ("c", "cpp", "fortran")


def _expand(src: str, shapes: dict) -> str:
    """Run the registered expander over the single ``out = np.diff(...)`` statement in ``src``."""
    assign = ast.parse(src).body[0]
    stmts = expand_diff(assign.targets[0], assign.value.args, shapes, assign.value.keywords)
    return "\n".join(ast.unparse(ast.fix_missing_locations(s)) for s in stmts)


def _assert_ok(res: dict) -> None:
    for backend, status in res.items():
        assert status == "ok" or status.startswith("skip"), f"{backend}: {status}"
    assert any(status == "ok" for status in res.values()), f"all skipped (vacuous): {res}"


def test_registered():
    assert NP_CALL_EXPANDERS[("np", "diff")] is expand_diff


def test_1d_expands_to_neighbour_subtraction():
    got = _expand("out = np.diff(a)", {"a": ("n",)})
    assert got == "for __df0 in range(n - 1):\n    out[__df0] = a[__df0 + 1] - a[__df0]"


def test_2d_defaults_to_last_axis():
    got = _expand("out = np.diff(a)", {"a": ("r", "c")})
    assert got == (
        "for __df0 in range(r):\n"
        "    for __df1 in range(c - 1):\n"
        "        out[__df0, __df1] = a[__df0, __df1 + 1] - a[__df0, __df1]"
    )


def test_2d_axis_zero_walks_rows():
    got = _expand("out = np.diff(a, axis=0)", {"a": ("r", "c")})
    assert got == (
        "for __df0 in range(r - 1):\n"
        "    for __df1 in range(c):\n"
        "        out[__df0, __df1] = a[__df0 + 1, __df1] - a[__df0, __df1]"
    )


def test_negative_axis_normalizes():
    assert _expand("out = np.diff(a, axis=-1)", {"a": ("n",)}) == _expand("out = np.diff(a)", {"a": ("n",)})


def test_literal_extent_folds_the_bound():
    got = _expand("out = np.diff(a)", {"a": ("8",)})
    assert "range(8 - 1)" in got


def test_n_greater_than_one_refused():
    with pytest.raises(NotImplementedError, match="first difference"):
        _expand("out = np.diff(a, 2)", {"a": ("n",)})


def test_prepend_refused_as_concatenate():
    with pytest.raises(NotImplementedError, match="concatenate"):
        _expand("out = np.diff(a, prepend=0)", {"a": ("n",)})


def test_nonconstant_axis_refused():
    with pytest.raises(NotImplementedError, match="constant int"):
        _expand("out = np.diff(a, axis=k)", {"a": ("r", "c")})


def test_numeric_1d_matches_numpy():
    a = np.array([3.0, 5.0, 4.0, 9.0, 9.0, 1.0])
    src = "import numpy as np\n\ndef k(a, out):\n    out[:] = np.diff(a)\n"
    _assert_ok(
        run_op(src, "k", {"a": a}, {"out": (5,)}, {"n": 6}, shapes={"a": "(6,)", "out": "(5,)"}, backends=_NATIVE)
    )


def test_numeric_row_lengths_from_csr_row_ptr():
    """The idiom that found the gap: row lengths out of a CSR ``row_ptr``."""
    row_ptr = np.array([0, 2, 2, 5, 9], dtype=np.int64)
    src = "import numpy as np\n\ndef k(row_ptr, out):\n    out[:] = np.diff(row_ptr)\n"
    _assert_ok(
        run_op(
            src,
            "k",
            {"row_ptr": row_ptr},
            {"out": (4,)},
            {"n": 5},
            shapes={"row_ptr": "(5,)", "out": "(4,)"},
            backends=_NATIVE,
            dtypes={"row_ptr": "int64", "out": "int64"},
        )
    )


def test_numeric_2d_axis_zero_matches_numpy():
    a = np.arange(12, dtype=np.float64).reshape(3, 4) ** 2
    src = "import numpy as np\n\ndef k(a, out):\n    out[:, :] = np.diff(a, axis=0)\n"
    _assert_ok(run_op(src, "k", {"a": a}, {"out": (2, 4)}, {}, shapes={"a": "(3,4)", "out": "(2,4)"}, backends=_NATIVE))
