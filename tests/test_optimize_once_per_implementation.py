# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``Test.run`` optimizes each implementation ONCE: the first/validation execution optimizes it, and
the timed median execution measures the handle that call returned. Optimizing again in the median
execution re-ran a DaCe column's whole search (parse, pipeline, compile, reference, verify, score)
a second time per kernel."""

import pytest

from hpcagent_bench.frameworks import Benchmark, Test, generate_framework
from hpcagent_bench.frameworks.framework import ArgValue, BenchData, KernelImpl, KernelResult


def test_run_optimizes_each_implementation_once_and_times_the_optimized_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    """One ``optimize`` call per implementation, and every timed run goes through the handle it returned."""
    frmwrk = generate_framework("numpy")
    optimized_from: list[KernelImpl] = []
    handle_calls: list[int] = []

    def optimize(program: KernelImpl, bench: Benchmark, bdata: BenchData) -> KernelImpl:
        optimized_from.append(program)

        def handle(*args: ArgValue, **kwargs: ArgValue) -> KernelResult:
            handle_calls.append(1)
            return program(*args, **kwargs)

        return handle

    monkeypatch.setattr(frmwrk, "optimize", optimize)
    test = Test(Benchmark("gemm"), frmwrk, generate_framework("numpy"))
    res = test.run(preset="S", validate=True, repeat=3, timeout=300.0, datatype=None, ignore_errors=True)

    assert list(res) == ["numpy"], res
    assert res["numpy"]["validated"], res
    assert len(optimized_from) == 1, f"optimize ran {len(optimized_from)} times for one implementation"
    # first/validation: 1 rep + 1 capture run; median: 3 reps + 1 capture run -- all on the optimized handle.
    assert len(handle_calls) == 6, handle_calls
