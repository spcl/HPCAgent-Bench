# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``sdfg.instrument`` has to be set BEFORE ``sdfg.compile()``, not after.

``dace.codegen.codegen`` reads ``sdfg.instrument`` while it builds the ``.so`` (it picks an
instrumentation PROVIDER off it and only that provider emits the timing code into the generated
C++/CUDA/HIP). ``DaceFramework.create_timer`` used to set it AFTER ``compile_variants`` had
already compiled every variant -- a no-op for the already-built binary, so
``sdfg.get_latest_report()`` always came back ``None`` and every dace_gpu* column silently fell
back to the host python series (DaCe's own ``__call__`` marshalling overhead included in every
"native" number that was never native).

Nothing here needs a real dace build: what is under test is WHEN the attribute is set relative to
``compile()``, which a fake SDFG that records its own ``instrument`` at ``compile()`` time answers
without a C++ toolchain.
"""

import types

import dace
import pytest

from hpcagent_bench.frameworks import dace_framework

#: ``compile_variants`` reads ``ctx.opt`` unconditionally before the loop, even though a FINALIZED
#: pipeline (every entry in ``DACE_PIPELINES``) never calls a method on it -- an attribute access
#: that a bare ``None`` context would fail on.
FAKE_CTX = types.SimpleNamespace(opt=None)


class FakeCompiledSDFG:
    """Stands in for what ``sdfg.compile()`` returns; ``TimedCompiledSDFG`` only stores it."""


class FakeSDFG:
    """A minimal stand-in for ``dace.SDFG``: carries ``_name`` and ``instrument`` the same way,
    and ``compile()`` snapshots ``instrument`` the same moment ``dace.codegen.codegen`` reads it."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.instrument = dace.InstrumentationType.No_Instrumentation
        #: What ``instrument`` held at the instant ``compile()`` ran -- ``None`` if never compiled.
        self.instrument_at_compile_time: dace.InstrumentationType | None = None

    def compile(self) -> FakeCompiledSDFG:
        self.instrument_at_compile_time = self.instrument
        return FakeCompiledSDFG()


def make_framework(arch: str, pipeline: str) -> dace_framework.DaceFramework:
    fw = dace_framework.DaceFramework.__new__(dace_framework.DaceFramework)
    fw.info = {"arch": arch, "pipelines": (pipeline,)}
    return fw


@pytest.mark.parametrize(
    "arch, pipeline, expected",
    [
        ("gpu", "parallel_gpu", dace.InstrumentationType.GPU_Events),
        ("cpu", "parallel_cpu", dace.InstrumentationType.Timer),
    ],
)
def test_instrumentation_is_set_before_compile(arch: str, pipeline: str, expected: dace.InstrumentationType) -> None:
    """Written to fail on the shipped behaviour: ``compile()`` must observe the FINAL
    instrumentation type, not ``No_Instrumentation`` (what a post-compile assignment would leave
    it seeing)."""
    fw = make_framework(arch, pipeline)
    sdfg = FakeSDFG(pipeline)
    compiled = fw.compile_variants({pipeline: sdfg}, FAKE_CTX)
    assert sdfg.instrument_at_compile_time == expected, (
        f"compile() saw instrument={sdfg.instrument_at_compile_time!r}, expected {expected!r} -- "
        "instrumentation was set too late for codegen to see it"
    )
    assert compiled[pipeline].sdfg is sdfg


def test_gpu_uses_events_not_the_cpu_timer() -> None:
    """The pre-fix code set ``Timer`` unconditionally (even on a GPU pipeline) -- itself moot
    while the assignment was a no-op, but wrong on its own terms once it starts taking effect:
    DaCe's ``Timer`` provider brackets a HOST clock around the kernel launch, which undercounts an
    asynchronous GPU kernel the same way an unsynchronized python timer does."""
    fw = make_framework("gpu", "parallel_gpu")
    sdfg = FakeSDFG("parallel_gpu")
    fw.compile_variants({"parallel_gpu": sdfg}, FAKE_CTX)
    assert sdfg.instrument_at_compile_time == dace.InstrumentationType.GPU_Events
    assert sdfg.instrument_at_compile_time != dace.InstrumentationType.Timer


class FakeReportSDFG:
    """Records whether ``clear_instrumentation_reports`` was called, which is what stops
    ``create_timer`` from letting :meth:`DaceFramework.stop_timer` read a STALE report left in a
    reused build folder's ``perf/`` directory."""

    def __init__(self) -> None:
        self.cleared = False

    def clear_instrumentation_reports(self) -> None:
        self.cleared = True


def test_create_timer_clears_any_stale_report() -> None:
    program = dace_framework.TimedCompiledSDFG.__new__(dace_framework.TimedCompiledSDFG)
    program.sdfg = FakeReportSDFG()
    program.name = "parallel_gpu"
    fw = dace_framework.DaceFramework.__new__(dace_framework.DaceFramework)
    fw.create_timer(program)
    assert program.sdfg.cleared, "create_timer must clear perf/ before a new measurement reads it"


def test_create_timer_is_a_noop_for_a_non_compiled_program() -> None:
    """A program that never became a ``TimedCompiledSDFG`` (a plain ``@dace.program`` handle, or a
    pipeline that failed to compile) has no ``.sdfg`` to clear; create_timer must not raise."""
    fw = dace_framework.DaceFramework.__new__(dace_framework.DaceFramework)
    timer = fw.create_timer(lambda: None)
    assert timer.program is not None
