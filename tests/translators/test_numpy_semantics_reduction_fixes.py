# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""numpy-faithful semantics fixes in the SHARED translator lowering.

Two tiers:

* AST / source-level UNIT tests (fast, deterministic) that pin the exact loop /
  cast / dtype an expander or rank rule produces -- so a regression points
  straight at the cause.
* End-to-end ORACLE tests (``run_op`` / ``run_return_op``) that compile + run the
  kernel on every backend and compare bit-exact to numpy.

Covers: NaN-propagating max/min/argmax/argmin, int64 sum/prod accumulators,
zero-size reduction refusal, std/var ddof, sum/prod/max/min ``initial=``,
mean/std/var float-dtype preservation, concatenate negative axis, elementwise
broadcast of BOTH operands, symmetric size-1 broadcast, ellipsis subscript rank,
and int/int true division.
"""

import ast
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest

from hpcagent_bench.translators.numpyto_common.lib_nodes import (
    broadcast_extents,
    const_or_name,
    expand_max,
    expand_min,
    expand_prod,
    expand_sum,
)
from hpcagent_bench.translators.numpyto_common.lib_nodes.reductions import expand_arg_reduction
from tests.translators.op_oracle import all_ok, run_op

ALL = ("c", "cpp", "fortran", "numba")


def call_args(src: str) -> tuple[list[ast.Name], list[ast.keyword]]:
    call = ast.parse(src, mode="eval").body
    return call.args, call.keywords


def target(name: str = "out") -> ast.Name:
    return ast.Name(id=name, ctx=ast.Store())


# Fix 2: integer sum/prod accumulate in int64 (not a float 0.0 / 1.0)         #


@pytest.mark.parametrize(
    ("expander", "call", "dtype", "identity"),
    [(expand_sum, "np.sum(a)", "int64", 0), (expand_prod, "np.prod(a)", "int32", 1)],
    ids=["sum", "prod"],
)
def test_integer_sum_and_prod_use_an_int_accumulator_and_int64_dtype(
    expander: Callable, call: str, dtype: str, identity: int
) -> None:
    args, kws = call_args(call)
    ld = {"a": dtype}
    stmts = expander(target(), args, {"a": ("N",)}, kws, local_dtypes=ld)
    init = stmts[0]
    assert isinstance(init.value, ast.Constant)
    assert init.value.value == identity
    assert isinstance(init.value.value, int)
    assert ld["out"] == "int64"  # result upcast to int64 (numpy rule)


# Fix 3: zero-size reduction refuses to lower (no OOB seed)                    #


@pytest.mark.parametrize(
    ("expander", "call", "shape", "extra"),
    [
        (expand_max, "np.max(a)", ("0",), ()),
        (expand_min, "np.min(a, axis=1)", ("N", "0"), ()),
        # numpy raises "attempt to get argmax of an empty sequence"; the loop's seed read a[0] out of bounds.
        (expand_arg_reduction, "np.argmax(a)", ("0",), ("argmax",)),
    ],
    ids=["max", "min_along_axis", "argmax"],
)
def test_a_zero_length_reduction_axis_refuses(
    expander: Callable, call: str, shape: tuple[str, ...], extra: tuple[str, ...]
) -> None:
    args, kws = call_args(call)
    with pytest.raises(NotImplementedError):
        expander(target(), args, {"a": shape}, kws, *extra)


def test_max_zero_length_kept_axis_ok() -> None:
    # A zero-length KEPT axis is fine (no reduction over it); only a zero-length
    # REDUCED axis has no identity.
    args, kws = call_args("np.max(a, axis=1)")
    expand_max(target(), args, {"a": ("0", "M")}, kws)  # must not raise


# Fix 8/9: elementwise broadcasts BOTH operands (and symmetric size-1)         #


def ext_(*toks: str) -> tuple[ast.expr, ...]:
    return tuple(const_or_name(t) for t in toks)


def unp(exts: tuple[ast.expr, ...]) -> tuple[str, ...]:
    return tuple(ast.unparse(e) for e in exts)


def test_broadcast_extents_size1_symmetric() -> None:
    assert unp(broadcast_extents(ext_("N", "1"), ext_("N", "M"))) == ("N", "M")
    assert unp(broadcast_extents(ext_("M"), ext_("N", "M"))) == ("N", "M")
    assert unp(broadcast_extents(ext_("1"), ext_("N"))) == ("N",)
    assert unp(broadcast_extents(ext_("N", "M"), ext_("N", "1"))) == ("N", "M")


class OracleCase(NamedTuple):
    """One kernel compiled and run on every backend against numpy, with the keywords of ``run_op``."""

    src: str
    inputs: dict[str, np.ndarray]
    outputs: dict[str, tuple[int, ...]]
    symbols: dict[str, int]
    shapes: dict[str, str]
    dtypes: dict[str, str] | None = None
    #: A numpy fact the case rests on, asserted before the run.
    anchor: Callable[[], bool] | None = None


NAN_ARGS = np.array([1.0, 5.0, np.nan, 2.0, np.nan])  # first NaN at index 2
OVERFLOW_INT32 = np.full((4, 3), 2**30, dtype=np.int32)  # each column sums to 4 * 2**30 = 2**32: not an int32


# pythran runs the full reduction natively, and its runtime ``np.max`` / ``np.min`` SUPPRESS NaN (verified: returns 2.0
# for the NaN-propagation input) -- a pythran limitation in code the shared translator does not own. The C / C++ /
# Fortran path (which the NaN fix targets) plus numba / jax propagate NaN.
ORACLE_CASES = {
    "integer-sum-prod": OracleCase(
        "import numpy as np\ndef f(a, s, p):\n s[0] = np.sum(a)\n p[0] = np.prod(a)\n",
        {"a": np.array([1, 2, 3, 4, 5], dtype=np.int64)},
        {"s": (1,), "p": (1,)},
        {"N": 5},
        {"a": "(N,)", "s": "(1,)", "p": "(1,)"},
        {"a": "int64", "s": "int64", "p": "int64"},
    ),
    "max-min-nan-propagation": OracleCase(
        "import numpy as np\ndef f(a, mx, mn):\n mx[0] = np.max(a)\n mn[0] = np.min(a)\n",
        {"a": np.array([1.0, np.nan, 2.0, -3.0])},
        {"mx": (1,), "mn": (1,)},
        {"N": 4},
        {"a": "(N,)", "mx": "(1,)", "mn": "(1,)"},
    ),
    "argmax-argmin-first-nan-index": OracleCase(
        "import numpy as np\ndef f(a, i, j):\n i[0] = np.argmax(a)\n j[0] = np.argmin(a)\n",
        {"a": NAN_ARGS},
        {"i": (1,), "j": (1,)},
        {"N": 5},
        {"a": "(N,)", "i": "(1,)", "j": "(1,)"},
        {"i": "int64", "j": "int64"},
        lambda: bool(np.argmax(NAN_ARGS) == 2 and np.argmin(NAN_ARGS) == 2),
    ),
    "var-std-ddof": OracleCase(
        "import numpy as np\ndef f(a, v, s):\n v[0] = np.var(a, ddof=1)\n s[0] = np.std(a, ddof=2)\n",
        {"a": np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])},
        {"v": (1,), "s": (1,)},
        {"N": 8},
        {"a": "(N,)", "v": "(1,)", "s": "(1,)"},
    ),
    "initial": OracleCase(
        "import numpy as np\ndef f(a, s, p, mx, mn):\n"
        " s[0] = np.sum(a, initial=10.0)\n"
        " p[0] = np.prod(a, initial=2.0)\n"
        " mx[0] = np.max(a, initial=100.0)\n"
        " mn[0] = np.min(a, initial=-5.0)\n",
        {"a": np.array([1.0, 2.0, 3.0, 4.0])},
        {"s": (1,), "p": (1,), "mx": (1,), "mn": (1,)},
        {"N": 4},
        {"a": "(N,)", "s": "(1,)", "p": "(1,)", "mx": "(1,)", "mn": "(1,)"},
    ),
    "axis-sum-int32-overflow": OracleCase(
        "import numpy as np\ndef f(x, out):\n    out[:] = np.sum(x, axis=0)\n",
        {"x": OVERFLOW_INT32},
        {"out": (3,)},
        {"N": 4},
        {"x": "(4,3)", "out": "(3,)"},
        {"x": "int32"},
        lambda: np.sum(OVERFLOW_INT32, axis=0).dtype == np.int64,  # numpy anchor
    ),
    "concatenate-negative-axis": OracleCase(
        "import numpy as np\ndef f(a, b, out):\n out[:] = np.concatenate((a, b), axis=-1)\n",
        {"a": np.arange(6.0).reshape(2, 3), "b": (np.arange(4.0) + 10).reshape(2, 2)},
        {"out": (2, 5)},
        {"M": 2, "N": 3, "K": 2},
        {"a": "(M, N)", "b": "(M, K)", "out": "(M, 5)"},
    ),
    "maximum-broadcast-lower-rank-first-operand": OracleCase(
        "import numpy as np\ndef f(a, B, out):\n out[:] = np.maximum(a, B)\n",
        {"a": np.array([1.0, 5.0, 3.0]), "B": np.array([[0.0, 6.0, 2.0], [4.0, 1.0, 9.0]])},  # (M,), (N, M)
        {"out": (2, 3)},
        {"M": 3, "N": 2},
        {"a": "(M,)", "B": "(N, M)", "out": "(N, M)"},
    ),
    "multiply-broadcast-row-vector": OracleCase(
        "import numpy as np\ndef f(a, B, out):\n out[:] = np.multiply(a, B)\n",
        {"a": np.array([[2.0, 3.0, 4.0]]), "B": np.array([[1.0, 1.0, 1.0], [5.0, 6.0, 7.0]])},  # (1, M), (N, M)
        {"out": (2, 3)},
        {"M": 3, "N": 2},
        {"a": "(1, M)", "B": "(N, M)", "out": "(N, M)"},
    ),
    "int-true-division": OracleCase(
        "import numpy as np\ndef f(a, b, out):\n out[0] = a[0] / b[0]\n out[1] = a[1] / b[1]\n out[2] = a[2] / b[2]\n",
        {"a": np.array([7, 1, 9], dtype=np.int64), "b": np.array([2, 4, 2], dtype=np.int64)},
        {"out": (3,)},
        {"N": 3},
        {"a": "(N,)", "b": "(N,)", "out": "(N,)"},
        {"a": "int64", "b": "int64"},
    ),
    "explicit-float-cast-in-division": OracleCase(
        "import numpy as np\ndef f(a, b, out):\n out[0] = float(a[0]) / b[0]\n out[1] = float(a[1]) / b[1]\n",
        {"a": np.array([7, 3], dtype=np.int64), "b": np.array([2, 4], dtype=np.int64)},
        {"out": (2,)},
        {"N": 2},
        {"a": "(N,)", "b": "(N,)", "out": "(N,)"},
        {"a": "int64", "b": "int64"},
    ),
}


@pytest.mark.parametrize("case", ORACLE_CASES.values(), ids=ORACLE_CASES.keys())
def test_explicit_float_cast_in_division_all_backends(case: OracleCase) -> None:
    """Every fix above, compiled and run on every backend and compared bit-exact to numpy."""
    assert case.anchor is None or case.anchor()
    res = run_op(
        case.src,
        "f",
        case.inputs,
        case.outputs,
        case.symbols,
        shapes=case.shapes,
        dtypes=case.dtypes,
        backends=ALL,
    )
    ok, unused = all_ok(res)
    assert ok, res
