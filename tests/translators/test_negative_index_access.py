# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Negative CONSTANT index ``a[-k]`` -> explicit ``a[N-k]`` for the C / C++ ABI.

numpy (and fortran / numba) wrap a negative index from the end
(``a[-1]`` is the last element), but C has no negative indexing -- ``a[-1]``
underflows the pointer and reads garbage. The C emitter therefore normalizes a
negative constant index against the array's known dimension size, at both read
and write positions. A negative SLICE bound (``a[:-1]``) is a different construct
and is left to the slice lowering (it already works).
"""

import numpy as np

from tests.translators.op_oracle import all_ok, parse_source, run_op

ALL = ("c", "cpp", "fortran", "numba")


# negative_literal_offset: -1 is a UnaryOp, not a Constant                   #


# the C emit turns ``a[-1]`` into ``a[N - 1]`` (no literal ``[-1]``)          #


def emit_c_(src: str, inputs: list[str], shapes: dict[str, str], syms: dict[str, int]) -> str:
    from hpcagent_bench.translators.numpyto_c.emit import emit_c
    from hpcagent_bench.translators.numpyto_common.lowering import lower

    return emit_c(lower(parse_source(src, "f", inputs, [], shapes, syms)), fn_name="f")


def test_c_emit_leaves_positive_index_alone() -> None:
    c = emit_c_(
        "import numpy as np\ndef f(a, out):\n out[0] = a[2]\n", ["a", "out"], {"a": "(N,)", "out": "(2,)"}, {"N": 6}
    )
    assert "a[2]" in c


# numerical: bit-exact vs numpy across every backend                          #


def test_bare_negative_index_read() -> None:
    a = np.arange(6, dtype=np.float64)
    ok, res = all_ok(
        run_op(
            "import numpy as np\ndef f(a, out):\n out[0] = a[-1]\n out[1] = a[-2]\n",
            "f",
            {"a": a},
            {"out": (2,)},
            {"N": 6},
            shapes={"a": "(N,)", "out": "(2,)"},
            backends=ALL,
        )
    )
    assert ok, res


def test_negative_index_write() -> None:
    a = np.arange(6, dtype=np.float64)
    ok, res = all_ok(
        run_op(
            "import numpy as np\ndef f(a, out):\n out[:] = a\n out[-1] = 99.0\n",
            "f",
            {"a": a},
            {"out": (6,)},
            {"N": 6},
            shapes={"a": "(N,)", "out": "(N,)"},
            backends=ALL,
        )
    )
    assert ok, res


def test_negative_index_2d_mixed_axes() -> None:
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    ok, res = all_ok(
        run_op(
            "import numpy as np\ndef f(a, out):\n out[0] = a[-1, -1]\n out[1] = a[1, -1]\n out[2] = a[-1, 2]\n",
            "f",
            {"a": a},
            {"out": (3,)},
            {"M": 3, "N": 4},
            shapes={"a": "(M, N)", "out": "(3,)"},
            backends=ALL,
        )
    )
    assert ok, res


def test_negative_slice_bound_still_works() -> None:
    # ``a[:-1]`` is a slice bound, NOT an index -- left to the slice lowering.
    a = np.arange(6, dtype=np.float64)
    ok, res = all_ok(
        run_op(
            "import numpy as np\ndef f(a, out):\n out[:] = a[1:] - a[:-1]\n",
            "f",
            {"a": a},
            {"out": (5,)},
            {"N": 6},
            shapes={"a": "(N,)", "out": "(5,)"},
            backends=ALL,
        )
    )
    assert ok, res
