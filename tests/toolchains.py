# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Toolchain markers: one marker per family of things not every supported host has.

A test that needs such a toolchain carries its marker and may name the requirements it needs
(``@pytest.mark.mpi("mpi4py")``); a bare marker needs every requirement of the family. Selection
goes through ``-m`` and nothing else:

* a run whose ``-m`` expression does not name the marker DESELECTS those tests, and the summary
  counts them as deselected, never as skipped;
* a run that names it keeps them, and a missing requirement is a setup error carrying the probe's
  diagnosis. ``-m "mpi or not mpi"`` runs everything, the mpi tests included.

``HPCAGENT_BENCH_NO_SKIP=1`` fails a session that skipped anything and lists the skipped node ids.

Every probe lives here and imports what it needs only when a marked test is selected: the repo-root
conftest loads this plugin for the translator tree too, which imports nothing from hpcagent_bench.
"""

import ctypes.util
import functools
import importlib
import importlib.util
import os
import pathlib
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import ModuleType

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

#: ``1`` fails any session that skipped a test.
NO_SKIP_ENV = "HPCAGENT_BENCH_NO_SKIP"

#: The words of a ``-m`` expression that are operators rather than marker names.
EXPRESSION_OPERATORS = frozenset({"and", "or", "not"})


def import_diagnosis(module: str) -> str:
    """Empty when ``module`` imports, else why. ``OSError`` counts: a wheel whose shared library
    will not dlopen raises it instead of ``ImportError`` (the apache-tvm ABI break did)."""
    try:
        importlib.import_module(module)
    except (ImportError, OSError) as exc:
        return f"import {module} raised {type(exc).__name__}: {exc}"
    return ""


def path_diagnosis(*names: str) -> str:
    """Empty when every executable in ``names`` is on PATH, else the missing ones."""
    missing = [name for name in names if shutil.which(name) is None]
    return f"not on PATH: {', '.join(missing)}" if missing else ""


def load_path(name: str, path: pathlib.Path) -> ModuleType:
    """A helper module that is not importable by package name (a sibling of a port test)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=1, typed=True)
def mpi_c() -> str:
    from tests import mpi_launch_helpers

    return mpi_launch_helpers.c_toolchain_diagnosis()


@functools.lru_cache(maxsize=1, typed=True)
def mpi4py_launcher() -> str:
    from tests import mpi_launch_helpers

    return mpi_launch_helpers.mpi4py_launcher_diagnosis()


@functools.lru_cache(maxsize=1, typed=True)
def gpu_device() -> str:
    missing = import_diagnosis("cupy")
    if missing:
        return missing
    runtime = importlib.import_module("cupy.cuda.runtime")
    try:
        count = runtime.getDeviceCount()
    except Exception as exc:  # noqa: BLE001 -- each GPU runtime raises its own error class
        return f"cupy.cuda.runtime.getDeviceCount() raised {type(exc).__name__}: {exc}"
    return "" if count > 0 else "cupy reports no GPU device"


@functools.lru_cache(maxsize=1, typed=True)
def hip_cupy() -> str:
    missing = import_diagnosis("cupy._environment")
    if missing:
        return missing
    if "_get_hipcc_include_dirs" in vars(importlib.import_module("cupy._environment")):
        return ""
    return "this cupy is not a ROCm build: cupy._environment has no _get_hipcc_include_dirs"


@functools.lru_cache(maxsize=1, typed=True)
def papi_library() -> str:
    from tests import papi_probe

    return "" if papi_probe.PAPI_LIBRARY else "ctypes.util.find_library('papi') found no libpapi on this host"


@functools.lru_cache(maxsize=1, typed=True)
def papi_counting() -> str:
    from hpcagent_bench.harness import papi
    from tests import papi_probe

    if papi_probe.CAN_COUNT:
        return ""
    if not papi_probe.PAPI_LIBRARY:
        return "no libpapi on this host, so no hardware counter can be armed"
    gate = papi.perf_event_reason()
    return gate[1] if gate else "PAPI loads but arms no countable event on this CPU"


@functools.lru_cache(maxsize=1, typed=True)
def perf_sampling() -> str:
    from hpcagent_bench import perf_reports

    try:
        perf_reports.perf_check()
    except perf_reports.PerfUnavailable as exc:
        return f"{exc.cause}: {exc}"
    return ""


@functools.lru_cache(maxsize=1, typed=True)
def polycc() -> str:
    from hpcagent_bench import pluto_transform

    if pluto_transform.polycc_exe():
        return ""
    return "polycc absent: the Pluto toolchain is built from source, see containers/pluto.Dockerfile"


@functools.lru_cache(maxsize=1, typed=True)
def pluto_openmp() -> str:
    from hpcagent_bench import flags

    probe = flags.pluto_capability()
    if probe.verdict is flags.AutoparVerdict.OK:
        return ""
    return f"this host's clang emits no OpenMP for Pluto's pragma: {probe.detail}"


@functools.lru_cache(maxsize=None, typed=True)
def compiler_driver(driver: str) -> str:
    from hpcagent_bench import languages

    return "" if languages.resolve_compiler(driver) else f"toolchain absent: {driver} is not on PATH"


@functools.lru_cache(maxsize=1, typed=True)
def docker_daemon() -> str:
    missing = path_diagnosis("docker")
    if missing:
        return missing
    try:
        done = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=60, check=False)
    except subprocess.TimeoutExpired:
        return "docker info did not answer within 60 s"
    if done.returncode == 0:
        return ""
    tail = (done.stderr or done.stdout).strip().splitlines()
    return f"docker info exited {done.returncode}: {tail[-1] if tail else 'no output'}"


def rodinia_hotspot_source() -> pathlib.Path | None:
    """Rodinia's OpenMP HotSpot source: ``RODINIA_ROOT``, else the sibling checkout the port came from."""
    roots = [pathlib.Path(root)] if (root := os.environ.get("RODINIA_ROOT")) else []
    roots.append(REPO.parent / "HPC" / "rodinia")
    for candidate in (base / "openmp" / "hotspot" / "hotspot_openmp.cpp" for base in roots):
        if candidate.is_file():
            return candidate
    return None


@functools.lru_cache(maxsize=1, typed=True)
def rodinia() -> str:
    if rodinia_hotspot_source() is not None:
        return ""
    return "no Rodinia checkout (set RODINIA_ROOT); Rodinia is not vendored in this repository"


@functools.lru_cache(maxsize=1, typed=True)
def tsvc_source() -> str:
    port = load_path("port_tsvc_cpp_references", REPO / "scripts" / "port_tsvc_cpp_references.py")
    roots = [port.DEFAULT_CPP_ROOT / family[0] for family in port.FAMILIES.values()]
    missing = [str(root) for root in roots if not root.is_dir()]
    return f"the TSVC C++ source of record is not on this machine: {', '.join(missing)}" if missing else ""


@functools.lru_cache(maxsize=1, typed=True)
def cloudsc_data() -> str:
    root = os.environ.get("CLOUDSC_DATA_DIR", "")
    if root and pathlib.Path(root).is_dir():
        return ""
    return "dwarf-p-cloudsc serialbox data is not present: set CLOUDSC_DATA_DIR to its directory"


@functools.lru_cache(maxsize=1, typed=True)
def distro_openblas() -> str:
    if ctypes.util.find_library("openblas") and ctypes.util.find_library("blas"):
        return ""
    return "no distro OpenBLAS on this host: install libopenblas-dev (CI gets it from .github/actions/setup)"


@functools.lru_cache(maxsize=1, typed=True)
def multiarch_libgomp() -> str:
    resident = pathlib.Path("/usr/lib/x86_64-linux-gnu/libgomp.so")
    return "" if resident.exists() else f"{resident} is not installed on this host"


@functools.lru_cache(maxsize=1, typed=True)
def fftw_lapack() -> str:
    helper = load_path("cegterg_reference_ctypes", REPO / "tests" / "ports" / "cegterg" / "cegterg_reference_ctypes.py")
    if helper.toolchain_available():
        return ""
    return "g++ with the FFTW3 / LAPACK / BLAS headers and libraries is unavailable (apt libfftw3-dev liblapacke-dev)"


@dataclass(frozen=True, slots=True)
class Toolchain:
    """What a marker needs, and one probe per requirement that answers empty when it is present."""

    description: str
    requirements: Mapping[str, Callable[[], str]]


#: Every toolchain marker. ``CI:`` names the jobs that provision the family and select its marker;
#: tests/test_toolchain_markers.py holds the workflow to that.
TOOLCHAINS: dict[str, Toolchain] = {
    "mpi": Toolchain(
        "an MPI C wrapper whose launcher starts a real 2-rank world (c), and mpi4py bootstrapping under "
        "a launcher (mpi4py). CI: mpi.",
        {"c": mpi_c, "mpi4py": mpi4py_launcher},
    ),
    "mpich": Toolchain(
        "MPICH's own mpicc.mpich and mpiexec.mpich, the distributed track's default; the mpi job runs "
        "OpenMPI instead. CI: none.",
        {"mpich": functools.partial(path_diagnosis, "mpicc.mpich", "mpiexec.mpich")},
    ),
    "gpu": Toolchain(
        "a GPU software stack: a CUDA or ROCm device cupy can see (device), nvcc (nvcc), cupy (cupy), a "
        "ROCm cupy build (hip_cupy), triton (triton). The gpu job is disabled and gpu-agentbench is a "
        "manual self-hosted run. CI: gpu, gpu-agentbench.",
        {
            "device": gpu_device,
            "nvcc": functools.partial(path_diagnosis, "nvcc"),
            "cupy": functools.partial(import_diagnosis, "cupy"),
            "hip_cupy": hip_cupy,
            "triton": functools.partial(import_diagnosis, "triton"),
        },
    ),
    "papi": Toolchain(
        "libpapi loads (ctypes.util.find_library('papi')); apt libpapi-dev. CI: unit.",
        {"papi": papi_library},
    ),
    "hw_counters": Toolchain(
        "hardware performance counters reachable from user space: PAPI arms a countable event (papi), "
        "perf can sample (perf). GitHub-hosted runners pass no PMU through. CI: none.",
        {"papi": papi_counting, "perf": perf_sampling},
    ),
    "pluto": Toolchain(
        "polycc from the source-built Pluto (polycc), and a clang that turns Pluto's OpenMP pragma into "
        "runtime calls (openmp). CI: frameworks-pluto.",
        {"polycc": polycc, "openmp": pluto_openmp},
    ),
    "oneapi": Toolchain(
        "Intel oneAPI icpx, from containers/install-extra-toolchains.sh. CI: unit, integration.",
        {"icpx": functools.partial(compiler_driver, "icpx")},
    ),
    "nvhpc": Toolchain(
        "NVIDIA HPC SDK nvc++, from containers/install-extra-toolchains.sh with INSTALL_NVHPC=1 (unit "
        "shard 0 only). CI: unit.",
        {"nvc++": functools.partial(compiler_driver, "nvc++")},
    ),
    "apptainer": Toolchain(
        "an unprivileged apptainer on PATH. CI: container-image.",
        {"apptainer": functools.partial(path_diagnosis, "apptainer")},
    ),
    "docker": Toolchain(
        "a reachable docker daemon for the image-build tests. CI: none.",
        {"daemon": docker_daemon},
    ),
    "gt4py": Toolchain(
        "the gt4py extra (GTScript and its numpy backend). CI: mpi.",
        {"gt4py": functools.partial(import_diagnosis, "gt4py.cartesian.gtscript")},
    ),
    "latex": Toolchain(
        "latex and dvipng, which matplotlib's usetex rendering shells out to. CI: none.",
        {"latex": functools.partial(path_diagnosis, "latex", "dvipng")},
    ),
    "agent_extras": Toolchain(
        "the agent-optimas extra (optimas) and the harbor extra (harbor). CI: unit.",
        {
            "optimas": functools.partial(import_diagnosis, "optimas"),
            "harbor": functools.partial(import_diagnosis, "harbor.models.task.config"),
        },
    ),
    "hf": Toolchain(
        "the hf extra's pyarrow parquet writer. CI: none.",
        {"pyarrow": functools.partial(import_diagnosis, "pyarrow.parquet")},
    ),
    "upstream_sources": Toolchain(
        "a non-vendored upstream checkout or dataset: Rodinia via RODINIA_ROOT (rodinia), the TSVC C++ "
        "source of record (tsvc), dwarf-p-cloudsc serialbox data via CLOUDSC_DATA_DIR (cloudsc_data). "
        "CI: none.",
        {"rodinia": rodinia, "tsvc": tsvc_source, "cloudsc_data": cloudsc_data},
    ),
    "distro": Toolchain(
        "the Ubuntu layout .github/actions/setup provides and a spack host lays out differently: distro "
        "OpenBLAS (openblas), /usr/lib/x86_64-linux-gnu/libgomp.so (libgomp), versioned drivers (gcc-16, "
        "clang-22), the hpcagent-bench console script of the editable install (console_script), g++ with "
        "FFTW3 and LAPACK (fftw). CI: unit, integration, mpi, ports-cegterg.",
        {
            "openblas": distro_openblas,
            "libgomp": multiarch_libgomp,
            "gcc-16": functools.partial(path_diagnosis, "gcc-16"),
            "clang-22": functools.partial(path_diagnosis, "clang-22"),
            "console_script": functools.partial(path_diagnosis, "hpcagent-bench"),
            "fftw": fftw_lapack,
        },
    ),
}


#: Vendor compiler drivers, by the toolchain marker a case parametrized over compilers carries.
VENDOR_DRIVERS: dict[str, str] = {
    "icx": "oneapi",
    "icpx": "oneapi",
    "ifx": "oneapi",
    "nvc": "nvhpc",
    "nvc++": "nvhpc",
    "nvfortran": "nvhpc",
}


def driver_marks(driver: str) -> tuple[pytest.MarkDecorator, ...]:
    """The marker a case compiled by ``driver`` carries: its vendor family, or none for GNU/LLVM."""
    family = VENDOR_DRIVERS.get(driver)
    if family == "oneapi":
        return (pytest.mark.oneapi,)
    if family == "nvhpc":
        return (pytest.mark.nvhpc,)
    return ()


def named_markers(expression: str) -> frozenset[str]:
    """The marker names a ``-m`` expression mentions, whatever it does with them."""
    return frozenset(re.findall(r"[A-Za-z_]\w*", expression)) - EXPRESSION_OPERATORS


def pytest_configure(config: pytest.Config) -> None:
    for name, toolchain in TOOLCHAINS.items():
        config.addinivalue_line("markers", f"{name}(requirement, ...): {toolchain.description}")
    if os.environ.get(NO_SKIP_ENV) == "1" and "PYTEST_XDIST_WORKER" not in os.environ:
        config.pluginmanager.register(SkipGuard(), "hpcagent-bench-no-skip")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect every test whose toolchain marker the ``-m`` expression does not name."""
    named = named_markers(str(config.getoption("markexpr")))
    kept: list[pytest.Item] = []
    dropped: list[pytest.Item] = []
    for item in items:
        needs = set()
        for marker in item.iter_markers():
            toolchain = TOOLCHAINS.get(marker.name)
            if toolchain is None:
                continue
            unknown = sorted(set(map(str, marker.args)) - set(toolchain.requirements))
            if unknown:
                raise pytest.UsageError(
                    f"{item.nodeid}: {marker.name} has no requirement {unknown}; known: {sorted(toolchain.requirements)}"
                )
            needs.add(marker.name)
        (kept if needs <= named else dropped).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """A selected toolchain test whose requirement is absent errors before any fixture is built."""
    for marker in item.iter_markers():
        toolchain = TOOLCHAINS.get(marker.name)
        if toolchain is None:
            continue
        for requirement in marker.args or tuple(toolchain.requirements):
            diagnosis = toolchain.requirements[requirement]()
            if diagnosis:
                pytest.fail(
                    f"-m selected {marker.name}; its requirement {requirement!r} is not met: {diagnosis}", pytrace=False
                )


@dataclass(slots=True)
class SkipGuard:
    """Registered under ``HPCAGENT_BENCH_NO_SKIP=1``: a runtime skip fails the session."""

    skipped: list[str] = field(default_factory=list)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.skipped and "wasxfail" not in vars(report):
            self.skipped.append(report.nodeid)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if self.skipped and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter: pytest.TerminalReporter) -> None:
        if not self.skipped:
            return
        terminalreporter.section(f"{NO_SKIP_ENV}=1: {len(self.skipped)} skipped test(s) fail the session", red=True)
        for nodeid in self.skipped:
            terminalreporter.line(nodeid)
