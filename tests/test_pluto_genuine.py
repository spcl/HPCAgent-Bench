# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Pluto column times what polycc wrote, or it times nothing.

The bug these pin is one the column shipped with: ``pluto`` built the same ``<base>_fpNN.cpp`` as
``llvm``, with the same clang, and never ran polycc -- so every pluto-vs-llvm number in the results
DB was llvm-vs-llvm under a polyhedral label. What makes that hard to keep fixed is that each way of
reintroducing it is silent. A fallback to the untransformed source compiles and runs. A positional
call in the canonical ABI order against polycc's symbols-first VLA signature returns numbers. A
cached ``.so`` from before the column was rebuilt loads. None of them raise.

So these tests are mostly about what the column REFUSES to do. ``tests/test_native_autogen.py``
covers the emitted binding's order against the canonical one; here it is the build path, the
marshalling that consumes that order, the polycc invocation, and -- with a real toolchain -- that
the transformed library computes the right answer.
"""
import ctypes
import os
import pathlib
import shutil
import subprocess
import types
from typing import Any, Dict

import numpy as np
import pytest

from hpcagent_bench import flags, pluto_transform
from hpcagent_bench.benchmarks import cpp_runtime
from hpcagent_bench.frameworks.benchmark import Benchmark
from hpcagent_bench.frameworks.errors import NotSupportedByFramework
from hpcagent_bench.frameworks.pluto_framework import PlutoFramework
from hpcagent_bench.harness import preflight

#: An affine matmul in the shape the translator emits for polycc: ``int64_t`` counters (which is why
#: the invocation needs ``--pet``; the default clan extractor rejects them) and rank-2 arrays as VLA
#: parameters, whose extents C requires declared BEFORE them -- the reason polycc's signature puts
#: the size symbol first and the canonical C ABI's does not.
SCOP = """#include <stdint.h>
#include <math.h>
void mm_fp64(const int64_t N, double (*restrict A)[N], double (*restrict B)[N], double (*restrict C)[N]) {
#pragma scop
  for (int64_t i = 0; i < N; i++)
    for (int64_t j = 0; j < N; j++)
      for (int64_t k = 0; k < N; k++)
        C[i][j] += A[i][k] * B[k][j];
#pragma endscop
}
"""

#: The Pluto column gates on this exactly as ``cpp_runtime.assert_autopar_capable`` does: polycc has
#: already written ``#pragma omp parallel for`` into the source, so a clang that generates no OpenMP
#: for it would time Pluto's parallel output single-threaded (see ``flags.PLUTO_PAR``).
PLUTO_CAPABILITY = flags.pluto_capability()

#: Why an absent polycc is a genuine environment gap rather than a weakened test: Pluto has no wheel
#: and no distro package here, it is built from source by CI and by the container recipe.
NO_POLYCC = "polycc absent: the Pluto toolchain is built from source, see containers/pluto.Dockerfile"


def write_scop(cpp_backend: pathlib.Path, base: str = "mm", fptype: str = "fp64", text: str = SCOP) -> pathlib.Path:
    """Emit ``<base>_<fptype>_pluto_input.c`` where the transform step looks for it."""
    cpp_backend.mkdir(parents=True, exist_ok=True)
    scop = cpp_backend / f"{base}_{fptype}_pluto_input.c"
    scop.write_text(text)
    return scop


class ManifestFreeBench(Benchmark):
    """A ``Benchmark`` that skips the manifest load.

    ``call_args`` reads ``bname`` and nothing else -- the paths come through ``_cpp_backend`` and
    ``_native_base``, which these tests point at a tmp dir. Loading a real corpus entry would tie an
    argument-ordering test to a kernel whose manifest says nothing about argument ordering.
    """

    def __init__(self, bname: str = "mm") -> None:
        self.bname = bname
        self.bdata: Dict[Any, Any] = {}


def no_impl(*args: Any, **kwargs: Any) -> None:
    """``call_args`` marshals values for a callable without invoking it; this stands in for it."""
    raise AssertionError("call_args must not invoke the kernel")


def fake_arg(name: str, kind: str) -> Any:
    """A stand-in for one ``_abi_args`` descriptor: ``call_args`` reads only ``name`` and ``kind``."""
    return types.SimpleNamespace(name=name, kind=kind, shape=(), dtype="float64")


def failing_polycc(cmd: Any, **kwargs: Any) -> subprocess.CompletedProcess:
    """A polycc that rejects the scop, with the diagnostic pet actually prints for a non-affine one."""
    return subprocess.CompletedProcess(cmd, 1, "", "pet: data dependent conditions not supported")


# --------------------------------------------------------------------------------------------------
# Decline, never fall back. Each of these has a tempting "just build the C++ instead" answer, and
# taking it is how the column reported clang numbers under Pluto's name for as long as it did.
# --------------------------------------------------------------------------------------------------


def test_absent_polycc_declines_instead_of_building_the_untransformed_source(tmp_path, monkeypatch) -> None:
    """No polycc means no Pluto source. The failure mode being refused is compiling the emitted C++
    and calling the result a Pluto measurement."""
    write_scop(tmp_path)
    monkeypatch.setattr(pluto_transform, "polycc_exe", lambda: None)
    with pytest.raises(NotSupportedByFramework) as excinfo:
        pluto_transform.transformed_sources(tmp_path, "mm")
    assert "polycc" in str(excinfo.value)


def test_no_emitted_scop_declines(tmp_path) -> None:
    """A kernel the translator emitted no ``#pragma scop`` for has nothing for polycc to transform."""
    with pytest.raises(NotSupportedByFramework) as excinfo:
        pluto_transform.transformed_sources(tmp_path, "mm")
    assert "scop" in str(excinfo.value)


def test_nonaffine_scop_declines_because_polycc_may_miscompile_it(tmp_path, monkeypatch) -> None:
    """polycc may silently MISCOMPILE a scop outside its affine model rather than reject it, so
    "exited 0" is not evidence the transform was sound. The gate is the detector, not the exit code."""
    write_scop(tmp_path)
    monkeypatch.setattr(pluto_transform, "polycc_exe", lambda: "/usr/bin/polycc")
    monkeypatch.setattr(pluto_transform, "scop_nonaffine_reason", lambda text: "data-dependent-bound")
    with pytest.raises(NotSupportedByFramework) as excinfo:
        pluto_transform.transformed_sources(tmp_path, "mm")
    assert "affine" in str(excinfo.value)
    assert "data-dependent-bound" in str(excinfo.value)


def test_polycc_rejection_declines_and_surfaces_its_own_diagnostic(tmp_path, monkeypatch) -> None:
    """A rejected scop must name WHY in the decline; an opaque "polycc failed" sends the reader back
    to a run they cannot reproduce."""
    write_scop(tmp_path)
    monkeypatch.setattr(pluto_transform, "polycc_exe", lambda: "/usr/bin/polycc")
    monkeypatch.setattr(pluto_transform.subprocess, "run", failing_polycc)
    with pytest.raises(NotSupportedByFramework) as excinfo:
        pluto_transform.transformed_sources(tmp_path, "mm")
    assert "data dependent conditions not supported" in str(excinfo.value)


def test_a_failed_polycc_leaves_no_partial_output_to_be_compiled(tmp_path, monkeypatch) -> None:
    """polycc writes as it goes. A truncated translation unit whose mtime is newer than the scop's is
    exactly the "fresh enough, reuse it" condition the next build tests, so it would compile half a
    kernel and time it."""
    scop = write_scop(tmp_path)
    out = pluto_transform.transformed_path(scop)
    out.write_text("/* half a kernel */\n")
    monkeypatch.setattr(pluto_transform, "polycc_exe", lambda: "/usr/bin/polycc")
    monkeypatch.setattr(pluto_transform.subprocess, "run", failing_polycc)

    _cmd, proc = pluto_transform.run_polycc(scop, out)

    assert proc.returncode == 1
    assert not out.exists(), "a partial polycc output survived a failed run"


def test_call_args_declines_when_the_binding_is_absent(tmp_path, monkeypatch) -> None:
    """A positional ctypes call cannot detect a permuted argument list -- it runs and returns numbers.
    With no binding there is no safe default, so the only correct answer is to decline."""
    framework = PlutoFramework.__new__(PlutoFramework)
    monkeypatch.setattr(PlutoFramework, "_cpp_backend", lambda self, bench: tmp_path)
    monkeypatch.setattr(PlutoFramework, "_native_base", lambda self, bench: "mm")
    with pytest.raises(NotSupportedByFramework):
        framework.call_args(ManifestFreeBench(), no_impl, {}, {})


# --------------------------------------------------------------------------------------------------
# What the timed library is built FROM, and how it is invoked.
# --------------------------------------------------------------------------------------------------


def test_the_build_selects_polyccs_output_over_the_emitted_cpp(tmp_path, monkeypatch) -> None:
    """With both on disk, the pluto column must take ``<base>_fpNN_pluto.c``. The emitted ``.cpp``
    sitting beside it is what the llvm column builds, and picking it up is the original bug."""
    scop = write_scop(tmp_path)
    (tmp_path / "mm_fp64.cpp").write_text("// what the llvm column compiles\n")
    monkeypatch.setattr(pluto_transform, "polycc_exe", lambda: "/usr/bin/polycc")

    def fake_polycc(cmd: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        pathlib.Path(cmd[cmd.index("-o") + 1]).write_text("/* transformed */\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(pluto_transform.subprocess, "run", fake_polycc)

    sources = cpp_runtime._native_sources(tmp_path, "mm", "pluto")

    assert [p.name for p in sources] == ["mm_fp64_pluto.c"]
    assert sources[0] != scop, "the column compiled polycc's INPUT rather than its output"


def test_the_column_compiles_c_with_a_c_driver() -> None:
    """polycc's output is C and only C: VLA parameters, the ``restrict`` keyword and ``register``,
    none of which survive C++ -- and it prepends its own ``#define min(x,y)``, which detonates inside
    libstdc++. A ``clangpp`` here would not be a style choice, it would not compile."""
    assert cpp_runtime.FRAMEWORK_LANG["pluto"] == "c"
    assert cpp_runtime.FRAMEWORK_COMPILER["pluto"] != "clangpp"
    assert cpp_runtime.FRAMEWORK_LANG["pluto"] != cpp_runtime.FRAMEWORK_LANG["llvm"]


def test_polycc_is_invoked_with_pet_and_the_report_only_adds_verbosity() -> None:
    """``--pet`` because the emitted scop uses ``int64_t`` counters the default clan extractor
    rejects. The report's args are defined as an EXTENSION of the build's so the two are structurally
    incapable of describing different transforms."""
    assert "--pet" in pluto_transform.POLYCC_ARGS
    assert pluto_transform.POLYCC_REPORT_ARGS[:len(pluto_transform.POLYCC_ARGS)] == pluto_transform.POLYCC_ARGS
    assert set(pluto_transform.POLYCC_REPORT_ARGS) - set(pluto_transform.POLYCC_ARGS) == {"--debug"}


def test_polycc_runs_under_the_pet_parse_shim(tmp_path, monkeypatch) -> None:
    """pet extracts the scop with a flag-less libclang. On aarch64 its default target has no ``neon``
    feature, so glibc's ``<bits/math-vector.h>`` -- reached through the preamble's ``<math.h>`` --
    rejects the whole translation unit before any scop is seen. Wired into ``run_polycc`` so the
    timed build and the transformation report parse the scop identically."""
    scop = write_scop(tmp_path)
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(pluto_transform, "polycc_exe", lambda: "/usr/bin/polycc")

    def capture(cmd: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        shim = pathlib.Path(kwargs["env"]["C_INCLUDE_PATH"].split(":")[0])
        seen["shadowed"] = (shim / "bits" / "math-vector.h").read_text()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(pluto_transform.subprocess, "run", capture)
    pluto_transform.run_polycc(scop, pluto_transform.transformed_path(scop))

    assert "libm-simd-decl-stubs.h" in seen["shadowed"], "math-vector.h was not shadowed for the pet parse"


def test_the_pet_shim_shadows_one_header_and_leads_the_rest_of_the_path(tmp_path, monkeypatch) -> None:
    """Scoped as narrowly as it can be: one header, prepended to an existing ``C_INCLUDE_PATH``
    rather than replacing it, and only for the polycc subprocess -- polycc invokes no compiler, so
    nothing that gets MEASURED is built under this environment."""
    monkeypatch.setenv("C_INCLUDE_PATH", "/opt/site/include")
    env = pluto_transform.pet_parse_env(tmp_path)
    first, _, rest = env["C_INCLUDE_PATH"].partition(":")
    assert rest == "/opt/site/include", "the shim replaced the caller's include path instead of leading it"
    assert sorted(p.name for p in (pathlib.Path(first) / "bits").iterdir()) == ["math-vector.h"]


# --------------------------------------------------------------------------------------------------
# Marshalling: consuming polycc's order. test_native_autogen pins that the emitted order DIFFERS from
# the canonical one; this pins that call_args actually reorders the values by it.
# --------------------------------------------------------------------------------------------------


def test_call_args_marshals_in_polyccs_order_not_the_canonical_abis(tmp_path, monkeypatch) -> None:
    """The transformed function takes each size SYMBOL before the VLA array it dimensions, so the
    canonical order would hand a pointer to an ``int64_t`` parameter -- measured elsewhere in this
    tree as an immediate SIGSEGV, which is the good outcome; the bad one is plausible numbers."""
    (tmp_path /
     "mm_fp64_pluto_binding.json").write_text('{"args": [{"name": "N"}, {"name": "A"}, {"name": "B"}, {"name": "C"}]}')
    # Canonical C ABI order: sorted pointers, then sorted scalars. Deliberately NOT polycc's.
    canonical = [fake_arg("A", "ptr"), fake_arg("B", "ptr"), fake_arg("C", "ptr"), fake_arg("N", "scalar")]
    assert [a.name for a in canonical] != ["N", "A", "B", "C"], "the orders must differ, or this proves nothing"
    monkeypatch.setattr(PlutoFramework, "_cpp_backend", lambda self, bench: tmp_path)
    monkeypatch.setattr(PlutoFramework, "_native_base", lambda self, bench: "mm")
    monkeypatch.setattr(PlutoFramework, "_abi_args", lambda self, bench: canonical)
    framework = PlutoFramework.__new__(PlutoFramework)

    resolved = {"A": "a", "B": "b", "C": "c"}
    args, kwargs = framework.call_args(ManifestFreeBench(), no_impl, resolved, {"N": 8})

    assert kwargs == {}
    assert args == [8, "a", "b", "c"], "the size symbol must lead; a leading pointer means the binding was not read"


def test_call_args_still_allocates_declared_outputs_after_reordering(tmp_path, monkeypatch) -> None:
    """Reordering must not drop the base class's output allocation: a kernel whose numpy reference
    RETURNS a buffer has no init-provided value, and the C signature still declares the pointer."""
    (tmp_path / "mm_fp64_pluto_binding.json").write_text('{"args": [{"name": "N"}, {"name": "C"}]}')
    declared = [fake_arg("C", "ptr"), fake_arg("N", "scalar")]
    monkeypatch.setattr(PlutoFramework, "_cpp_backend", lambda self, bench: tmp_path)
    monkeypatch.setattr(PlutoFramework, "_native_base", lambda self, bench: "mm")
    monkeypatch.setattr(PlutoFramework, "_abi_args", lambda self, bench: declared)
    monkeypatch.setattr(PlutoFramework, "_alloc_output", staticmethod(lambda arg, bdata: f"allocated:{arg.name}"))
    framework = PlutoFramework.__new__(PlutoFramework)

    args, _kwargs = framework.call_args(ManifestFreeBench(), no_impl, {}, {"N": 8})

    assert args == [8, "allocated:C"]


# --------------------------------------------------------------------------------------------------
# Preflight: report the cause once, up front, rather than once per declined kernel.
# --------------------------------------------------------------------------------------------------


def test_preflight_is_fatal_for_a_pluto_job_without_polycc(monkeypatch) -> None:
    """With polycc absent the column declines every kernel, so the job burns its allocation producing
    nothing but skips. Fatal for the same reason a dace without the fork's pipeline is."""
    monkeypatch.setattr(pluto_transform, "polycc_exe", lambda: None)
    code, report, env = preflight.run(["pluto"], print_env=True)
    assert code == 1
    assert any("polycc" in line for line in report)
    assert env == [], "a refused job must hand the caller nothing to eval"


def test_preflight_passes_and_says_so_when_polycc_is_present(monkeypatch) -> None:
    monkeypatch.setattr(pluto_transform, "polycc_exe", lambda: "/usr/bin/polycc")
    code, report, _env = preflight.run(["pluto"])
    assert code == 0
    assert any("polycc present" in line for line in report)


def test_preflight_does_not_gate_columns_that_never_run_polycc(monkeypatch) -> None:
    """Only the column that COMPILES polycc's output needs it; gating a numpy job on Pluto's
    toolchain would refuse a run that is perfectly valid without it."""
    monkeypatch.setattr(pluto_transform, "polycc_exe", lambda: None)
    assert preflight.needs_polycc(["numpy", "cc", "llvm"]) == []
    assert preflight.run(["numpy"])[0] == 0


# --------------------------------------------------------------------------------------------------
# End to end on the real toolchain: transform, compile, call, and check the arithmetic.
# --------------------------------------------------------------------------------------------------


@pytest.mark.skipif(pluto_transform.polycc_exe() is None, reason=NO_POLYCC)
@pytest.mark.skipif(shutil.which("clang") is None, reason="clang absent: the pluto column compiles polycc's C with it")
@pytest.mark.skipif(PLUTO_CAPABILITY.verdict is not flags.AutoparVerdict.OK,
                    reason=f"this host's clang emits no OpenMP for Pluto's pragma: {PLUTO_CAPABILITY.detail}")
def test_the_transformed_library_computes_the_right_answer(tmp_path) -> None:
    """The whole path with nothing faked: polycc transforms the scop, the column compiles ITS output
    as C, and the resulting symbol -- called through polycc's symbols-first signature -- agrees with
    numpy. The transformed source is checked for polycc's marks first, so a library that happened to
    be built from the untransformed input could not pass this by computing the right answer."""
    write_scop(tmp_path)

    so_path = cpp_runtime._ensure_built(tmp_path, "mm", "pluto")

    assert so_path.name == "libmm_pluto.so"
    transformed = (tmp_path / "mm_fp64_pluto.c").read_text()
    assert "#pragma omp parallel for" in transformed, "polycc marked no loop parallel"
    assert "#pragma scop" not in transformed, "this is the untransformed input, not polycc's output"

    lib = ctypes.CDLL(str(so_path))
    kernel = lib["mm_fp64"]
    ptr = ctypes.POINTER(ctypes.c_double)
    kernel.argtypes = [ctypes.c_int64, ptr, ptr, ptr]
    kernel.restype = None

    n = 64
    rng = np.random.default_rng(0)
    a = np.ascontiguousarray(rng.random((n, n)))
    b = np.ascontiguousarray(rng.random((n, n)))
    c = np.zeros((n, n))
    kernel(n, *(arr.ctypes.data_as(ptr) for arr in (a, b, c)))

    np.testing.assert_allclose(c, a @ b, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(pluto_transform.polycc_exe() is None, reason=NO_POLYCC)
@pytest.mark.skipif(shutil.which("clang") is None, reason="clang absent: the pluto column compiles polycc's C with it")
@pytest.mark.skipif(PLUTO_CAPABILITY.verdict is not flags.AutoparVerdict.OK,
                    reason=f"this host's clang emits no OpenMP for Pluto's pragma: {PLUTO_CAPABILITY.detail}")
def test_a_stale_library_is_rebuilt_rather_than_timed(tmp_path) -> None:
    """The ``.so`` name says which framework built it and nothing about which SOURCES it compiled, so
    a tree holding one from before this column compiled polycc's output would be loaded, timed and
    recorded as a Pluto number while being a clang one."""
    write_scop(tmp_path)
    build = tmp_path / "build"
    build.mkdir()
    stale = build / "libmm_pluto.so"
    stale.write_bytes(b"not a library")
    os.utime(stale, (0, 0))  # unambiguously older than the transform, whatever the mtime granularity

    so_path = cpp_runtime._ensure_built(tmp_path, "mm", "pluto")

    assert so_path == stale
    assert stale.read_bytes()[:4] == b"\x7fELF", "a stale .so was returned instead of rebuilt"
