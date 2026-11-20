# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""``arr[idx] = v`` is a boolean-MASK select or an integer-index SCATTER, and only the index
array's DTYPE separates them -- shape equality cannot.

Accepting a bare ``Name`` index on shape equality alone would lower an int64 index array whose
declared shape matches the target to ``if (idx[i]) arr[i] = v``: the values are read as truth, at
the wrong positions, and the loop runs to the target's extent rather than the index set's --
reading off the end whenever the index set is shorter (lulesh's ``xdd[symmX] = 0.0``: ``symmX`` is
a node-index set of length ``edgeNodes**2`` declared ``(numNode,)``). ``collect_bool_names`` is
the one criterion both mask rewriters use.
"""

from hpcagent_bench.translators.numpyto_c.emit import emit_c
from hpcagent_bench.translators.numpyto_common.lowering import lower
from tests.translators.op_oracle import parse_source


def emit_c_(src: str, inputs: list[str], shapes: dict[str, str], syms: dict[str, int], dtypes: dict[str, str]) -> str:
    return emit_c(lower(parse_source(src, "f", inputs, [], shapes, syms, dtypes)), fn_name="f")


SRC = "import numpy as np\ndef f(out, idx):\n out[idx] = 0.0\n"


def test_the_scatter_runs_over_the_index_set_not_the_target() -> None:
    """The loop bound is the INDEX array's extent. Taking the target's would read past the end
    of any index set shorter than the array it writes into."""
    c = emit_c_(SRC, ["out", "idx"], {"out": "(N,)", "idx": "(M,)"}, {"N": 8, "M": 3}, {"idx": "int64"})
    scatter = [ln for ln in c.splitlines() if "out[idx[" in ln]
    assert scatter, c
    bound = [ln for ln in c.splitlines() if "for (" in ln and "< M;" in ln]
    assert bound, f"scatter loop is not bounded by the index extent M:\n{c}"


def test_a_mask_computed_in_the_kernel_is_still_a_mask() -> None:
    """The mask PRODUCER is lowered to an explicit loop before the mask CONSUMERS run, so a set
    collected at the consumer sees only ``m[i] = ...`` and cannot prove ``m`` boolean. Harvest
    once, off the source-shaped tree (LoweringContext), or mandelbrot1's ``N[I] = n`` survives as
    a raw array subscript and the C will not compile."""
    src = "import numpy as np\ndef f(out, src, horizon):\n m = np.less(src, horizon)\n out[m] = 0.0\n"
    c = emit_c_(src, ["out", "src", "horizon"], {"out": "(N,)", "src": "(N,)"}, {"N": 8}, {})
    assert "if (m[" in c, f"a kernel-computed np.less mask did not lower to a guard:\n{c}"
    assert "out[m]" not in c, f"mask left as a raw array subscript (will not compile):\n{c}"
