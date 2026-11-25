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
import pytest

from tests.translators.op_oracle import all_ok, parse_source, run_op

ALL = ("c", "cpp", "fortran", "numba")


# negative_literal_offset: -1 is a UnaryOp, not a Constant                   #


# the C emit turns ``a[-1]`` into ``a[N - 1]`` (no literal ``[-1]``)          #


def emit_c_(src: str, inputs: list[str], shapes: dict[str, str], syms: dict[str, int]) -> str:
    from hpcagent_bench.translators.numpyto_c.emit import emit_c
    from hpcagent_bench.translators.numpyto_common.lowering import lower

    return emit_c(lower(parse_source(src, "f", inputs, [], shapes, syms)), fn_name="f")


# numerical: bit-exact vs numpy across every backend                          #


@pytest.mark.parametrize(
    ("src", "a", "out_shape", "syms", "shapes"),
    [
        (
            "import numpy as np\ndef f(a, out):\n out[0] = a[-1]\n out[1] = a[-2]\n",
            np.arange(6, dtype=np.float64),
            (2,),
            {"N": 6},
            {"a": "(N,)", "out": "(2,)"},
        ),
        (
            "import numpy as np\ndef f(a, out):\n out[:] = a\n out[-1] = 99.0\n",
            np.arange(6, dtype=np.float64),
            (6,),
            {"N": 6},
            {"a": "(N,)", "out": "(N,)"},
        ),
        (
            "import numpy as np\ndef f(a, out):\n out[0] = a[-1, -1]\n out[1] = a[1, -1]\n out[2] = a[-1, 2]\n",
            np.arange(12, dtype=np.float64).reshape(3, 4),
            (3,),
            {"M": 3, "N": 4},
            {"a": "(M, N)", "out": "(3,)"},
        ),
        # ``a[:-1]`` is a slice bound, NOT an index -- left to the slice lowering.
        (
            "import numpy as np\ndef f(a, out):\n out[:] = a[1:] - a[:-1]\n",
            np.arange(6, dtype=np.float64),
            (5,),
            {"N": 6},
            {"a": "(N,)", "out": "(5,)"},
        ),
    ],
    ids=["bare_read", "write", "2d_mixed_axes", "negative_slice_bound_still_works"],
)
def test_negative_index_matches_numpy_on_every_backend(
    src: str, a: np.ndarray, out_shape: tuple[int, ...], syms: dict[str, int], shapes: dict[str, str]
) -> None:
    ok, res = all_ok(run_op(src, "f", {"a": a}, {"out": out_shape}, syms, shapes=shapes, backends=ALL))
    assert ok, res
