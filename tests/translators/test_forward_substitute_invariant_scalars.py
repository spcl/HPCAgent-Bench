# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A loop-invariant scalar is replayed at its deeper use sites, and the assign is gone.

pet drops such an assign and the deeper consumers then read an uninitialised value (POLYCC-001);
substituting it also makes scalar-laundered indirection literal (POLYCC-006). Asserted on the
emitted C, since the emitted text is what polycc reads.
"""

import re

from hpcagent_bench.translators.numpyto_c.emit import emit_c
from hpcagent_bench.translators.numpyto_common.lowering import lower
from tests.translators.op_oracle import parse_source

#: conv_2d's own ``w = w_box[di + R, dj + R]``, with the pad already materialised.
CONV = (
    "import numpy as np\n"
    "def conv_op(w_box, padded, out_grid, K, N):\n"
    "    for di in range(K):\n"
    "        for dj in range(K):\n"
    "            w = w_box[di, dj]\n"
    "            for i in range(N):\n"
    "                for j in range(N):\n"
    "                    out_grid[i, j] += w * padded[i + di, j + dj]\n"
)


def emit_(src: str, func: str, extra_args: tuple[str, ...] = (), dtypes: dict[str, str] | None = None) -> str:
    """Emit C for a throwaway kernel over ``w_box`` / ``padded`` / ``out_grid``."""
    shapes = {"w_box": "(K, K)", "padded": "(N + K, N + K)", "out_grid": "(N, N)"}
    shapes.update(dict.fromkeys(extra_args, "(N,)"))
    kir = parse_source(src, func, ["w_box", "padded", *extra_args], ["out_grid"], shapes, {"K": 3, "N": 8}, dtypes)
    return emit_c(lower(kir), fn_name=func)


def assigns_to(text: str, name: str) -> list[str]:
    """Whole-statement ``<name> = ...;`` lines, the form the substitution deletes."""
    return re.findall(rf"^\s*{name} = .*;$", text, re.MULTILINE)


def test_declines_when_the_source_array_is_written_in_the_nest() -> None:
    # Condition 4: replaying the read would cross a store to its own buffer.
    body = emit_(
        CONV.replace("conv_op", "alias_op").replace(
            "                    out_grid[i, j] += w * padded[i + di, j + dj]\n",
            "                    out_grid[i, j] += w * padded[i + di, j + dj]\n"
            "                    w_box[di, dj] = out_grid[i, j]\n",
        ),
        "alias_op",
    )
    assert assigns_to(body, "w"), f"a written source array must decline the substitution:\n{body}"


def test_declines_when_the_scalar_is_read_after_the_loop() -> None:
    # Condition 6: a read past the loop sees the last iteration's value.
    body = emit_(
        CONV.replace("conv_op(w_box, padded, out_grid, K, N)", "live_op(w_box, padded, tail, out_grid, K, N)").replace(
            "def conv_op", "def live_op"
        )
        + "    tail[0] = w\n",
        "live_op",
        extra_args=("tail",),
    )
    assert assigns_to(body, "w"), f"a scalar live past its loop must decline:\n{body}"


def test_declines_when_an_operand_is_reassigned() -> None:
    # Condition 5: ``k`` is written twice, so a deeper replay reads the second binding.
    body = emit_(
        CONV.replace("conv_op", "unstable_op").replace(
            "            w = w_box[di, dj]\n",
            "            k = dj\n            w = w_box[di, k]\n            k = dj + 1\n",
        ),
        "unstable_op",
    )
    assert assigns_to(body, "w"), f"an unstable operand must decline the substitution:\n{body}"


def test_declines_when_no_read_is_deeper() -> None:
    # Condition 7: a same-depth read is not the defect, so replaying is pure growth.
    body = emit_(
        CONV.replace("conv_op", "flat_op").replace(
            "            w = w_box[di, dj]\n            for i in range(N):\n                for j in range(N):\n",
            "            for i in range(N):\n"
            "                for j in range(N):\n"
            "                    w = w_box[di, dj]\n",
        ),
        "flat_op",
    )
    assert assigns_to(body, "w"), f"a same-depth read must decline the substitution:\n{body}"
