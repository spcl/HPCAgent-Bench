# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The DaCe backend keeps a float axis reduction as the numpy call, which canon lowers to a library reduction."""

import ast

from hpcagent_bench.translators.numpyto_c.dace_emit import body_allocated_shape


class Kir:
    """The fields ``desugar_for_python_backend`` reads off a KernelIR."""

    class Arr:
        def __init__(self, name: str, shape: tuple[str, ...], dtype: str) -> None:
            self.name, self.shape, self.dtype = name, shape, dtype

    sparse = None
    kernel_name = "rows"

    def __init__(self, dtype: str) -> None:
        self.arrays = [Kir.Arr("a", ("N", "M"), dtype), Kir.Arr("out", ("N",), dtype)]
        self.scalars = [Kir.Arr("dt", (), dtype)]


def test_a_kept_reduction_sizes_the_helper_out_param_from_its_operand() -> None:
    """lenet's pooling helper: the store's extent is the operand's allocation minus the reduced axes."""
    body = ast.parse("split = np.reshape(x, (N, h // 2, 2, w // 2, 2, 6))\nhret[:] = np.max(split, axis=(2, 4))\n").body
    assert body_allocated_shape(body, "hret", {"N", "h", "w"}) == ["N", "h // 2", "w // 2", "6"]
