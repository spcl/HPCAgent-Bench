# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Framework flavor-grouping regression tests (pure metadata, no compile/run).

Pins the consolidated registry: one Framework subclass per ``base`` flavor family,
the native backend split into its base languages, each language's autopar variant,
and polly, vs Pluto as its own toolchain, and APPy fully removed.
"""

import subprocess
import sys
import types

import pytest

from hpcagent_bench import frameworks
from hpcagent_bench.frameworks import NativeFramework, PlutoFramework
from hpcagent_bench.frameworks import framework as framework_module
from hpcagent_bench.frameworks.framework import (
    FRAMEWORK_META,
    Framework,
    framework_bases,
    framework_class,
    framework_flavors,
    generate_framework,
)
from hpcagent_bench.languages import LANG_TARGET, gpu_backend

#: The C family, one flavor per (vendor, autopar) pair. Seven, not eight: icx has no
#: auto-parallelizer (icc-classic's ``-parallel`` is accepted with warning #10430 and outlines
#: nothing), so there is deliberately no ``cc_oneapi_autopar``. Pinned here so a vendor arm cannot
#: be dropped, or a serial one added back, without this test saying so.
C_FAMILY = ["cc", "cc_autopar", "cc_llvm", "cc_llvm_autopar", "cc_oneapi", "cc_nvhpc", "cc_nvhpc_autopar"]


def test_native_family_is_the_base_languages_their_autopar_and_polly() -> None:
    # Each base language (c/cpp/fortran) plus its auto-parallelizing variant, plus polly.
    # The C family spans four vendors (C_FAMILY); cc_autopar/fortran_autopar are the gcc autopar
    # route; flang is LLVM Fortran; llvm/polly are the C++ clang pair and ``cpp`` its gcc half, so
    # a C-vs-C++ reading is within one compiler family instead of across two. All build through
    # the one NativeFramework wrapper.
    assert framework_flavors("native") == C_FAMILY + ["llvm", "cpp", "fortran", "fortran_autopar", "flang", "polly"]
    for name in framework_flavors("native"):
        assert type(generate_framework(name)) is NativeFramework


def test_the_oneapi_arm_has_no_autopar_flavor() -> None:
    """icx has no auto-parallelizer, so registering one would publish serial numbers under a
    parallel name. Pinned separately from the inventory above so the reason survives a rename."""
    from hpcagent_bench import flags

    assert "cc_oneapi_autopar" not in FRAMEWORK_META
    assert not hasattr(flags, "ICX_AUTOPAR"), (
        "an ICX_AUTOPAR constant is back; icx accepts -parallel with warning #10430 and outlines "
        "nothing, so any column built on it would be silently serial"
    )


def test_pluto_is_its_own_base_and_a_native_subclass() -> None:
    # Pluto is a separate toolchain (polycc source-to-source), not a native flavor, and the base
    # carries two arch flavors: polycc on the CPU and PPCG, the polyhedral GPU generator.
    assert framework_flavors("pluto") == ["pluto", "ppcg", "ppcg_cuda", "ppcg_hip"]
    for name in framework_flavors("pluto"):
        fw = generate_framework(name)
        assert type(fw) is PlutoFramework
        assert isinstance(fw, NativeFramework)  # reuses the C-ABI wrapper machinery
        assert fw.kernel_attr == f"kernel_{name}"
    assert FRAMEWORK_META["pluto"]["arch"] == "cpu" and FRAMEWORK_META["ppcg"]["arch"] == "gpu"
    # PPCG emits CUDA and the LOCAL toolchain decides what that compiles as (hipify runs in between
    # on ROCm). Pinned against ``gpu_backend()`` rather than a literal, because a literal here is
    # what left the entry claiming nvcc on an AMD node.
    assert FRAMEWORK_META["ppcg"]["language"] == gpu_backend()
    # ...and the two flavors of that column state theirs instead, so they do not move with the host.
    assert FRAMEWORK_META["ppcg_cuda"]["language"] == "cuda"
    assert FRAMEWORK_META["ppcg_hip"]["language"] == "hip"


def test_native_flavors_carry_language_and_compiler() -> None:
    expect = {
        "cc": ("c", "gcc"),
        "cc_autopar": ("c", "gcc"),
        "llvm": ("cpp", "clang"),
        "cpp": ("cpp", "gpp"),
        "fortran": ("fortran", "gfortran"),
        "fortran_autopar": ("fortran", "gfortran"),
        "flang": ("fortran", "flang"),
        "polly": ("cpp", "clang"),
        "pluto": ("c", "clang"),
    }
    for name, (lang, comp) in expect.items():
        assert FRAMEWORK_META[name]["language"] == lang
        assert FRAMEWORK_META[name]["compiler"] == comp


def test_arch_families_share_one_class() -> None:
    # Two PARENT columns (the searching flavors, fastest of the SDFG pipelines they name) plus one
    # flavor per individual pipeline, which is what lets a pipeline be measured on the kernels
    # where it LOSES. ``dace_cpu_simplify`` has no ``dace_gpu_simplify`` twin -- listed so adding
    # one stays a deliberate edit here rather than a silent asymmetry.
    assert sorted(framework_flavors("dace")) == [
        "dace_cpu",
        "dace_cpu_autoopt",
        "dace_cpu_canonicalize",
        "dace_gpu",
        "dace_gpu_autoopt",
        "dace_gpu_canonicalize",
    ]
    # The shape that makes the inventory mean something: a parent column names no ``column`` and no
    # ``flavor`` of its own, and every other flavor names its parent AND exactly one pipeline. A
    # per-pipeline column that searched two would report the fastest of them under one pipeline's
    # name, which is the measurement these columns exist to avoid.
    parents = {n for n in framework_flavors("dace") if FRAMEWORK_META[n].get("column") is None}
    assert parents == {"dace_cpu", "dace_gpu"}
    for name in framework_flavors("dace"):
        meta = FRAMEWORK_META[name]
        if name in parents:
            assert meta.get("flavor") is None, name
        else:
            assert meta["column"] in parents and len(meta["pipelines"]) == 1, name
    assert framework_flavors("tvm") == ["tvm", "tvm_cpu"]
    assert {type(generate_framework(n)).__name__ for n in framework_flavors("dace")} == {"DaceFramework"}
    assert {type(generate_framework(n)).__name__ for n in framework_flavors("tvm")} == {"TVMFramework"}


def test_appy_removed() -> None:
    assert "appy" not in FRAMEWORK_META
    import hpcagent_bench.frameworks as infra

    assert "APPyFramework" not in vars(infra)


#: The adapter class every base resolves to, by name: ``framework_class`` finds it through the
#: ``<base>_framework.py`` convention, and a lookup change must not hand a column another adapter.
BASE_CLASS_NAMES = {
    "numpy": "Framework",
    "numba": "NumbaFramework",
    "cupy": "CupyFramework",
    "jax": "JaxFramework",
    "pythran": "PythranFramework",
    "dace": "DaceFramework",
    "native": "NativeFramework",
    "pluto": "PlutoFramework",
    "triton": "TritonFramework",
    "tvm": "TVMFramework",
}


def test_every_framework_resolves_to_its_adapter_class_and_package_export() -> None:
    assert set(framework_bases()) == set(BASE_CLASS_NAMES)
    for name, meta in FRAMEWORK_META.items():
        cls = framework_class(name)
        assert issubclass(cls, Framework) and cls.__name__ == BASE_CLASS_NAMES[meta["base"]], name
        assert getattr(frameworks, cls.__name__) is cls
        assert cls.__name__ in frameworks.__all__


def test_a_misspelled_class_name_is_not_exported() -> None:
    """The convention matches case-insensitively to FIND a class, never to invent a second public name."""
    assert not hasattr(frameworks, "TvmFramework")


def test_a_base_without_its_module_names_the_expected_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(FRAMEWORK_META, "probe_missing", {**FRAMEWORK_META["numba"], "base": "nosuchbase"})
    with pytest.raises(ModuleNotFoundError, match="hpcagent_bench/frameworks/nosuchbase_framework.py"):
        framework_class("probe_missing")


def test_a_backend_missing_its_own_dependency_keeps_that_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the adapter module's own absence is reworded; a missing optional dependency inside it is not."""

    def absent_dependency(name: str) -> types.ModuleType:
        raise ModuleNotFoundError("No module named 'cupy'", name="cupy")

    monkeypatch.setattr(framework_module.importlib, "import_module", absent_dependency)
    with pytest.raises(ModuleNotFoundError) as excinfo:
        framework_class("cupy")
    assert excinfo.value.name == "cupy"


def test_a_module_without_the_conventional_class_names_it(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "hpcagent_bench.frameworks.probeonly_framework"
    monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))
    monkeypatch.setitem(FRAMEWORK_META, "probe_empty", {**FRAMEWORK_META["numba"], "base": "probeonly"})
    with pytest.raises(ImportError, match="probeonly_framework.py defines no Framework subclass"):
        framework_class("probe_empty")


def test_the_native_tables_are_projections_of_the_registry() -> None:
    from hpcagent_bench.autogen import NATIVE_FRAMEWORKS
    from hpcagent_bench.benchmarks.cpp_runtime import FRAMEWORK_LANG

    columns = [name for name, meta in FRAMEWORK_META.items() if meta["base"] in ("native", "pluto")]
    assert list(NATIVE_FRAMEWORKS) == columns and list(FRAMEWORK_LANG) == columns
    for name in columns:
        meta = FRAMEWORK_META[name]
        assert FRAMEWORK_LANG[name] == meta.get("language")
        assert NATIVE_FRAMEWORKS[name] == meta.get("emit_language", meta.get("language"))


def test_the_polyhedral_columns_emit_c_and_compile_what_their_tool_writes() -> None:
    """polycc and ppcg both read the C target's ``_pluto_input.c``; polycc writes C and ppcg writes CUDA."""
    from hpcagent_bench.autogen import NATIVE_FRAMEWORKS
    from hpcagent_bench.benchmarks.cpp_runtime import FRAMEWORK_LANG

    for name in ("pluto", "ppcg", "ppcg_cuda", "ppcg_hip"):
        assert NATIVE_FRAMEWORKS[name] == "c" and LANG_TARGET[NATIVE_FRAMEWORKS[name]] == "c"
    assert FRAMEWORK_LANG["pluto"] == "c"
    assert (FRAMEWORK_LANG["ppcg_cuda"], FRAMEWORK_LANG["ppcg_hip"]) == ("cuda", "hip")


@pytest.mark.parametrize(
    "module", ["hpcagent_bench.benchmarks.cpp_runtime", "hpcagent_bench.autogen", "hpcagent_bench.frameworks"]
)
def test_each_native_table_module_imports_first_in_a_fresh_interpreter(module: str) -> None:
    """A registry check that read ``cpp_runtime.FRAMEWORK_LANG`` while framework.py loaded made
    ``import hpcagent_bench.benchmarks.cpp_runtime`` a circular ImportError when it came first."""
    proc = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
