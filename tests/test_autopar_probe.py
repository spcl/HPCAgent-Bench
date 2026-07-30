# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``flags.probe_autopar`` is the only thing in the tree allowed to call an autopar column
"working": it compiles a real SCoP and inspects the object with ``nm``, never trusting mere
flag ACCEPTANCE (see ``flags.POLLY_PAR`` for the measured bug this guards against: Ubuntu clang
accepts ``-mllvm -polly-parallel`` and outlines nothing). These tests exercise the probe itself
plus the ``cpp_runtime`` gate built on it -- not the polly/gcc_autopar frameworks end to end
(``tests/test_frameworks.py`` covers those)."""
import shutil

import pytest

from hpcagent_bench import flags
from hpcagent_bench.benchmarks import cpp_runtime
from hpcagent_bench.frameworks.errors import NotSupportedByFramework


def test_probe_rejected_for_a_nonexistent_compiler():
    """No compiler on PATH -> REJECTED, not an exception, and no subprocess is even spawned."""
    probe = flags.probe_autopar("definitely-not-a-real-compiler-xyz", "-O3", "anything")
    assert probe.verdict is flags.AutoparVerdict.REJECTED
    assert "not found" in probe.detail


@pytest.mark.integration
def test_probe_is_vacuous_when_flags_carry_no_autopar():
    """Plain -O3 (no ``-ftree-parallelize-loops`` / Polly at all) compiles cleanly on any host but
    must never read as OK: nothing outlines a parallel loop without an autopar flag. This is the
    portable VACUOUS case (task requirement: "a flag set that compiles but outlines nothing") --
    unlike Polly's vacuousness, it does not depend on a specific broken clang build.
    """
    if shutil.which("gcc") is None:
        pytest.skip("gcc not installed")
    probe = flags.probe_autopar("gcc", flags.CPU_BASELINE_GCC, flags.GCC_AUTOPAR_OUTLINE_PATTERN)
    assert probe.verdict is flags.AutoparVerdict.VACUOUS, probe


@pytest.mark.integration
def test_probe_is_ok_for_gcc_tree_parallelize_loops():
    """Non-vacuity floor: gcc's ``-ftree-parallelize-loops`` genuinely outlines a parallel loop on
    this box (measured: GOMP>0 and a ``_loopfn``/``_omp_fn`` symbol). A probe test that never sees
    a positive verdict would be worthless -- this is the one that proves OK is reachable at all.
    """
    if shutil.which("gcc") is None:
        pytest.skip("gcc not installed")
    probe = flags.gcc_autopar_capability()
    assert probe.verdict is flags.AutoparVerdict.OK, probe
    assert "GOMP=0" not in probe.detail


def test_probe_is_lru_cached():
    """``@lru_cache(typed=True)``: identical args must not re-invoke the compiler a second time."""
    flags.probe_autopar.cache_clear()
    first = flags.probe_autopar("definitely-not-a-real-compiler-xyz", "-O3", "x")
    second = flags.probe_autopar("definitely-not-a-real-compiler-xyz", "-O3", "x")
    assert first == second
    info = flags.probe_autopar.cache_info()
    assert info.hits >= 1


def test_gate_declines_polly_when_the_probe_is_vacuous(monkeypatch):
    """``cpp_runtime.assert_autopar_capable`` must raise :class:`NotSupportedByFramework` --
    the framework's existing "deliberate, correct decline" mechanism -- when the probe is not OK.

    Forced via monkeypatch rather than relying on THIS host's clang: the point is the gate's
    wiring, not today's measured verdict, so the test stays correct even after a future clang
    build genuinely wires up Polly's auto-pipeline.
    """
    monkeypatch.setattr(flags, "polly_capability",
                        lambda: flags.AutoparProbe(flags.AutoparVerdict.VACUOUS, "forced for test"))
    with pytest.raises(NotSupportedByFramework, match="vacuous"):
        cpp_runtime.assert_autopar_capable("polly", "gemm")


def test_gate_allows_polly_when_the_probe_is_ok(monkeypatch):
    """Symmetric case: an OK verdict must not raise."""
    monkeypatch.setattr(flags, "polly_capability", lambda: flags.AutoparProbe(flags.AutoparVerdict.OK, "forced"))
    cpp_runtime.assert_autopar_capable("polly", "gemm")  # must not raise


@pytest.mark.parametrize("framework", ["cc", "llvm", "fortran", "cc_autopar", "fortran_autopar", "pluto"])
def test_gate_is_a_no_op_for_ungated_frameworks(framework):
    """Only ``polly`` is wired into :data:`cpp_runtime.AUTOPAR_GATED` today (task scope); every
    other native flavor must pass through regardless of any probe's verdict."""
    cpp_runtime.assert_autopar_capable(framework, "gemm")  # must not raise
