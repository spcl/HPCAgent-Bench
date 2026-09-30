# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The context gate (containers/lib/omp_context_gate.py) on contexts built from this host's own tools.

The image builds real contexts under /opt/omp (containers/lib/omp_contexts.sh) and verify_image.py runs the gate
in each. These integration tests build the same layout in a temporary directory from what this host has:
gnu = gcc's libgomp, llvm = the libomp clang links, with ``libgomp.so.1`` a link to libomp inside the llvm
directory only. In ONE process each then maps exactly one runtime while gcc, gfortran, clang, numba and BLAS
all run multi-threaded, and the gate refuses a second runtime and a serial team. They need gcc, gfortran, numpy,
scipy and numba (images and CI carry them and FAIL without); the llvm test needs clang, so it is skipped
where clang is absent (a login node) and runs in every image and in CI, which install it.
"""

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import types

import pytest

from hpcagent_bench import omp_context

REPO = pathlib.Path(__file__).resolve().parents[1]
GATE_PATH = REPO / "containers" / "lib" / "omp_context_gate.py"
LLVM_AVAILABLE = shutil.which("clang") is not None


#: Environment variables that pin a thread count.
THREAD_PINS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


def load_gate() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("omp_context_gate_under_test", GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(GATE_PATH.parent))
    spec.loader.exec_module(module)
    return module


def real(driver: str, name: str) -> pathlib.Path:
    answer = subprocess.run([driver, f"-print-file-name={name}"], capture_output=True, text=True, check=True)
    return pathlib.Path(answer.stdout.strip()).resolve()


def build_contexts(root: pathlib.Path, llvm: bool) -> None:
    """The layout containers/lib/omp_contexts.sh writes, from this host's libgomp and (llvm) libomp."""
    gomp = real(os.environ.get("CC", "gcc"), "libgomp.so.1")
    (root / "gnu" / "lib").mkdir(parents=True)
    for name in ("libgomp.so.1", "libgomp.so.1.0.0", "libgomp.so"):
        (root / "gnu" / "lib" / name).symlink_to(gomp)
    if llvm:
        libomp = real("clang", "libomp.so")
        (root / "llvm" / "lib").mkdir(parents=True)
        for name in ("libomp.so", "libgomp.so.1", "libgomp.so.1.0.0", "libgomp.so"):
            (root / "llvm" / "lib" / name).symlink_to(libomp)
        soname = subprocess.run(["objdump", "-p", str(libomp)], capture_output=True, text=True, check=True).stdout
        for line in soname.splitlines():
            if "SONAME" in line and not (root / "llvm" / "lib" / line.split()[-1]).exists():
                (root / "llvm" / "lib" / line.split()[-1]).symlink_to(libomp)


def run_gate(root: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE_PATH), "--root", str(root), *args],
        capture_output=True,
        text=True,
        # The gate measures threading: a thread count pinned by this process (numerical_oracle pins one
        # BLAS thread on import) must not reach it.
        env={key: value for key, value in os.environ.items() if key not in (omp_context.CONTEXT_ENV, *THREAD_PINS)},
        timeout=1800,
        check=False,
    )


@pytest.mark.integration
def test_the_gnu_context_maps_libgomp_alone_while_gcc_gfortran_numba_and_blas_run_multithreaded(
    tmp_path: pathlib.Path,
) -> None:
    build_contexts(tmp_path, llvm=False)
    done = run_gate(tmp_path, "--context", "gnu", "--require", "gcc", "gfortran", "--any-runtime")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "gcc: teams" in done.stdout and "gfortran: teams" in done.stdout and "numba prange + BLAS" in done.stdout
    assert "one OpenMP runtime mapped" in done.stdout


@pytest.mark.integration
@pytest.mark.skipif(not LLVM_AVAILABLE, reason="clang absent: images and CI install it, where this test runs")
def test_the_llvm_context_maps_libomp_alone_while_clang_and_numbas_gomp_abi_pool_run_multithreaded(
    tmp_path: pathlib.Path,
) -> None:
    """clang code (``__kmpc_*``) and numba's omppool (``GOMP_parallel``, resolved to libomp through the
    context's ``libgomp.so.1`` link) share one runtime file, both multi-threaded."""
    build_contexts(tmp_path, llvm=True)
    done = run_gate(tmp_path, "--context", "llvm", "--require", "clang", "--any-runtime")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "clang: teams" in done.stdout and "numba prange + BLAS: layer omp" in done.stdout
    assert "one OpenMP runtime mapped" in done.stdout and str(real("clang", "libomp.so")) in done.stdout


@pytest.mark.integration
def test_the_gate_names_a_runtime_that_is_not_the_contexts_and_exits_3(tmp_path: pathlib.Path) -> None:
    build_contexts(tmp_path, llvm=False)
    other = tmp_path / "libother.so.1"
    other.write_bytes(b"")
    done = run_gate(tmp_path, "--context", "gnu", "--require", "gcc", "--no-numpy", "--expect-runtime", str(other))
    assert done.returncode == 3 and "expected" in done.stderr, done.stdout + done.stderr


@pytest.mark.integration
def test_a_missing_required_probe_compiler_fails_the_gate_and_an_absent_optional_one_does_not(
    tmp_path: pathlib.Path,
) -> None:
    build_contexts(tmp_path, llvm=False)
    missing = run_gate(
        tmp_path,
        "--context",
        "gnu",
        "--require",
        "gcc",
        "gfortran",
        "hpcagent-no-such-probe",
        "--no-numpy",
        "--any-runtime",
    )
    assert missing.returncode == 0, "only the probes the context defines are looked up"
    env = {key: value for key, value in os.environ.items() if key != omp_context.CONTEXT_ENV}
    env["PATH"] = str(pathlib.Path(sys.executable).parent)  # no compilers at all
    absent = subprocess.run(
        [sys.executable, str(GATE_PATH), "--root", str(tmp_path), "--context", "gnu", "--no-numpy"],
        capture_output=True,
        text=True,
        env={**env, "CC": "hpcagent-no-such-gcc", "FC": "hpcagent-no-such-gfortran"},
        check=False,
    )
    assert absent.returncode != 0 and "required" in absent.stderr + absent.stdout


@pytest.mark.integration
def test_a_pragma_compiled_to_serial_code_reports_a_team_of_one_and_the_gate_calls_it_serial() -> None:
    """The silent failure: OpenMP source built WITHOUT -fopenmp links the runtime for omp_* calls and
    runs every region on one thread."""
    gate = load_gate()
    with pytest.raises(gate.SerialTeam, match="team of one"):
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            serial = gate.Probe(
                "serial", (os.environ.get("CC", "gcc"),), "openmp_probe.c", ("-Wl,--no-as-needed", "-lgomp")
            )
            lib = gate.build_probe(serial, pathlib.Path(raw))
            assert lib is not None
            gate.run_probe(serial, lib)


@pytest.mark.parametrize("context", ["gnu", "llvm", "nvhpc"])
def test_the_gates_environment_is_the_judges_grading_child_environment(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, context: str
) -> None:
    """omp_context_gate.py cannot import hpcagent_bench (it runs at image build): its own copy of
    ``context_env`` must stay the judge's."""
    from hpcagent_bench import config

    (tmp_path / context / "lib").mkdir(parents=True)
    config.set_override(omp_context.ROOT_KEY, str(tmp_path))
    base = {"LD_LIBRARY_PATH": "/opt/view/lib:/usr/lib"}
    assert load_gate().context_environment(tmp_path, context, base) == omp_context.context_env(context, base)
