# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Two ways the rank table answered with a rank it could not know.

* An index with more entries than its base has axes is an IndexError in numpy, so the base's rank is
  what is wrong. ``base - indices`` handed every consumer a negative or shifted rank; the table now
  answers unknown. The three imprecisions that produced such bases now rank correctly: an allocation
  from a shape-tuple local, a ``slice(...)`` index, and an allocation sized by a scalar expression.
* A binary op with one operand of unknown rank reported the other operand's rank. The unknown side may
  be the higher-rank one, so the result is unknown. The manifest scalars and size symbols are seeded at
  rank 0, so an array-scalar expression keeps its rank.
"""

import ast
from types import SimpleNamespace

import numpy as np
import pytest

from numpyto_common.numpy_desugar import desugar_for_python_backend, expr_rank, rank_table


def ranks_of(src: str, seed: dict[str, int]) -> dict[str, int]:
    return rank_table(ast.parse(src), seed)


def numpy_ndims(src: str, scope: dict[str, object], names: tuple[str, ...]) -> dict[str, int]:
    """The ``.ndim`` numpy itself gives each name after running ``src``."""
    exec(src, scope)  # noqa: S102
    return {name: np.ndim(scope[name]) for name in names}


def desugared_kernel(src: str, kir: SimpleNamespace, backend: str = "pythran") -> dict[str, object]:
    """Desugar ``src`` for ``backend`` and run it, returning the module namespace."""
    namespace: dict[str, object] = {}
    exec(desugar_for_python_backend(src, kir, backend=backend), namespace)  # noqa: S102
    return namespace


def kernel_ir(arrays: dict[str, tuple[str, ...]], scalars: tuple[str, ...] = ()) -> SimpleNamespace:
    """The fields ``desugar_for_python_backend`` reads off a KernelIR, for a kernel named ``k``."""
    return SimpleNamespace(
        kernel_name="k",
        arrays=[SimpleNamespace(name=n, shape=s) for n, s in arrays.items()],
        scalars=[SimpleNamespace(name=n) for n in scalars],
        symbols=[],
    )


@pytest.mark.parametrize(
    ("src", "ranks"),
    [("s[0]", {"s": 0}), ("v[i, j]", {"v": 1}), ("m[:, i, j]", {"m": 2}), ("m[i, j, ..., k]", {"m": 2})],
)
def test_more_indices_than_axes_has_no_rank(src: str, ranks: dict[str, int]) -> None:
    """numpy raises IndexError on each of these, so no rank the table reports for them can be right."""
    assert expr_rank(ast.parse(src, mode="eval").body, ranks) is None


def test_an_array_allocated_from_a_shape_tuple_local_has_the_tuples_rank() -> None:
    """Max-pool's ``padded = np.full(padded_shape, fill)``: the bare Name read as ONE length made
    ``padded`` rank 1, and the four-index window over it came out rank -1."""
    src = "shape = (n, c, h, w)\npadded = np.full(shape, -np.inf)\nwindow = padded[:, :, i, j]\n"
    scope: dict[str, object] = {"np": np, "n": 2, "c": 3, "h": 4, "w": 5, "i": 1, "j": 2}
    want = numpy_ndims(src, scope, ("padded", "window"))
    got = ranks_of(src, dict.fromkeys(("n", "c", "h", "w", "i", "j"), 0))
    assert want == {"padded": 4, "window": 2}
    assert {"padded": got["padded"], "window": got["window"]} == want


def test_a_slice_object_index_keeps_its_axis() -> None:
    """``slice(lo, hi)`` in an index is a slice, not a scalar index: max-pool 3-D's window
    ``padded[b, ch, slice(sz, sz + k), ...]`` keeps its spatial axes."""
    src = "xg = x[:, slice(lo, hi)]\nwindow = x[b, ch, slice(lo, hi)]\n"
    scope: dict[str, object] = {"x": np.zeros((2, 3, 4)), "lo": 0, "hi": 2, "b": 1, "ch": 2}
    want = numpy_ndims(src, scope, ("xg", "window"))
    got = ranks_of(src, {"x": 3, "lo": 0, "hi": 0, "b": 0, "ch": 0})
    assert want == {"xg": 3, "window": 1}
    assert {"xg": got["xg"], "window": got["window"]} == want


def test_an_allocation_sized_by_a_scalar_expression_is_one_dimensional() -> None:
    """gmres's ``e1 = np.zeros(m + 1, b.dtype)``: the elementwise fallback ranked the LENGTH, and
    called the vector a scalar."""
    src = "e1 = np.zeros(m + 1, b.dtype)\nwork = np.empty(hi - lo + 1)\n"
    scope: dict[str, object] = {"np": np, "b": np.zeros(3), "m": 2, "hi": 4, "lo": 1}
    want = numpy_ndims(src, scope, ("e1", "work"))
    got = ranks_of(src, {"b": 1, "m": 0, "hi": 0, "lo": 0})
    assert want == {"e1": 1, "work": 1}
    assert {"e1": got["e1"], "work": got["work"]} == want


def test_a_binary_op_with_an_unranked_operand_has_no_rank() -> None:
    """``y = a * z`` where ``z`` comes back from a helper the table cannot see into: ``z`` may be a
    matrix, so ``y``'s rank is unknown, not ``a``'s rank 1."""
    assert "y" not in ranks_of("z = helper(b)\ny = a * z\n", {"a": 1, "b": 1})


def test_a_negative_axis_is_not_normalized_against_a_guessed_rank() -> None:
    """``np.flip(a * c, axis=-1)`` with ``c`` a pseudo-inverse the table cannot rank: reporting ``a``'s
    rank 1 for the product normalized the axis to 0, and the desugared kernel reversed the rows instead
    of the columns."""
    src = "import numpy as np\ndef k(a, m, out):\n    c = np.linalg.pinv(m)\n    out[:, :] = np.flip(a * c, axis=-1)\n"
    kir = kernel_ir({"a": ("N",), "m": ("N", "N"), "out": ("N", "N")})
    a, out = np.arange(1.0, 4.0), np.zeros((3, 3))
    m = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]])
    desugared_kernel(src, kir, backend="numba")["k"](a, m, out)
    assert "np.flip(a * c, axis=-1)" in desugar_for_python_backend(src, kir, backend="numba")
    assert np.array_equal(out, np.flip(a * np.linalg.pinv(m), axis=-1))


def test_a_local_bound_to_a_helper_call_takes_the_helpers_return_rank() -> None:
    """``c = helper(b)`` is the helper's matrix. The desugar computed every helper's return rank and
    never handed it to the kernel's table, so ``a * c`` was ranked by ``a`` alone: the negative axis
    went to 0 and the pythran variant reversed the rows instead of the columns."""
    src = (
        "import numpy as np\n"
        "def helper(b):\n"
        "    return b[:, None] * b[None, :]\n"
        "def k(a, b, out):\n"
        "    c = helper(b)\n"
        "    out[:, :] = np.flip(a * c, axis=-1)\n"
    )
    kir = kernel_ir({"a": ("N",), "b": ("N",), "out": ("N", "N")})
    a, b, out = np.arange(1.0, 4.0), np.array([1.0, 2.0, 5.0]), np.zeros((3, 3))
    desugared_kernel(src, kir)["k"](a, b, out)
    assert "out[:, :] = (a * c)[:, ::-1]" in desugar_for_python_backend(src, kir, backend="pythran")
    assert np.array_equal(out, np.flip(a * (b[:, None] * b[None, :]), axis=-1))


def test_a_manifest_scalar_operand_keeps_the_arrays_rank_for_the_desugar() -> None:
    """``x * alpha`` with ``alpha`` a manifest scalar is ``x``'s rank, so its negative axis is still
    rewritten to the positive slice pythran handles. Without the rank-0 seed the strict table could not
    rank it, and ``np.flip(..., axis=-1)`` reached pythran verbatim."""
    src = "import numpy as np\ndef k(x, alpha, out):\n    out[:, :] = np.flip(x * alpha, axis=-1)\n"
    kir = kernel_ir({"x": ("N", "N"), "out": ("N", "N")}, scalars=("alpha",))
    x, out = np.arange(9.0).reshape(3, 3), np.zeros((3, 3))
    desugared_kernel(src, kir)["k"](x, 2.0, out)
    assert "out[:, :] = (x * alpha)[:, ::-1]" in desugar_for_python_backend(src, kir, backend="pythran")
    assert np.array_equal(out, np.flip(x * 2.0, axis=-1))


def test_a_range_counter_is_an_integer_scalar() -> None:
    """``icg`` in the depthwise conv's ``padded[:, ic_of_oc + icg, :, :]`` counts a ``range`` loop. A
    strict table with no rank for it could not rank the index sum, nor anything computed from ``icg``."""
    src = "for icg in range(groups):\n    gathered = padded[:, ic_of_oc + icg, :, :]\n    j = icg + 1\n"
    scope: dict[str, object] = {"padded": np.zeros((2, 6, 3, 3)), "ic_of_oc": np.array([0, 1]), "groups": 2}
    want = numpy_ndims(src, scope, ("icg", "gathered", "j"))
    got = ranks_of(src, {"padded": 4, "ic_of_oc": 1, "groups": 0})
    assert want == {"icg": 0, "gathered": 4, "j": 0}
    assert {"icg": got["icg"], "gathered": got["gathered"], "j": got["j"]} == want


def test_a_shape_entry_and_the_scalar_builtins_are_integer_scalars() -> None:
    """blasst's ``scale = 1.0 / np.sqrt(query.shape[-1])`` and a pooling helper's ``k = int(size)``:
    a strict binary op needs both ranked, or every product with them is unknown."""
    src = (
        "scale = 1.0 / np.sqrt(q.shape[-1])\nk = int(size)\nscores = (q @ q.T) * scale / (k * k)\nn = len(q) + q.ndim\n"
    )
    scope: dict[str, object] = {"np": np, "q": np.ones((3, 3)), "size": 2.0}
    want = numpy_ndims(src, scope, ("scale", "k", "scores", "n"))
    got = ranks_of(src, {"q": 2})
    assert want == {"scale": 0, "k": 0, "scores": 2, "n": 0}
    assert {"scale": got["scale"], "k": got["k"], "scores": got["scores"], "n": got["n"]} == want


def test_a_reduction_over_more_axes_than_its_operand_has_no_rank() -> None:
    """``np.max(s, axis=1)`` over a rank-0 ``s`` is an AxisError in numpy; the table answered -1."""
    assert "m" not in ranks_of("m = np.max(s, axis=1)\n", {"s": 0})
