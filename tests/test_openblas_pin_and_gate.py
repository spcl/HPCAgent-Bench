# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every image builds the one OpenBLAS the gate proved, through the overlay, and runs the gate.

OpenBLAS 0.3.34 segfaults a tall row-major dgemm (M >= 8192, K >= 512) in its Haswell/Zen kernels, the
builtin spack recipe adds NO_AVX512 under +dynamic_dispatch (which routes a Zen 4 host onto those
kernels), and Ubuntu's MAX_THREADS=64 OpenBLAS crashes past 64 concurrent callers. The images pin
0.3.30, register the hpcagent overlay ahead of builtin, and run containers/lib/blas_gate.sh.
"""

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
IMAGES = REPO / "containers" / "images"
LIB = REPO / "containers" / "lib"
DOCKERFILES = ("judge-agent-amd", "judge-agent-cpu", "judge-agent-cuda")
OVERLAY_PACKAGE = LIB / "spack-overlay" / "spack_repo" / "hpcagent" / "packages" / "openblas" / "package.py"


@pytest.mark.parametrize("image", DOCKERFILES)
def test_the_image_builds_the_pinned_openblas_with_runtime_dispatch(image: str) -> None:
    docker = (IMAGES / image / "Dockerfile").read_text(encoding="utf-8")
    specs = re.findall(r"'  - (openblas[^']*)'", docker)
    assert specs and all(spec.startswith("openblas@0.3.30") for spec in specs), specs


@pytest.mark.parametrize("image", DOCKERFILES)
def test_the_overlay_is_registered_ahead_of_builtin(image: str) -> None:
    docker = (IMAGES / image / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY containers/lib/spack-overlay /opt/spack-overlay" in docker
    hpcagent = re.search(r"'  hpcagent: [^']*'|\"  hpcagent: [^\"]*\"", docker)
    builtin = re.search(r"'  builtin: /opt/spack-packages/repos/spack_repo/builtin'", docker)
    assert hpcagent and builtin and hpcagent.start() < builtin.start(), "the overlay must shadow builtin"


@pytest.mark.parametrize("image", DOCKERFILES)
def test_the_image_runs_the_blas_gate_on_its_view(image: str) -> None:
    docker = (IMAGES / image / "Dockerfile").read_text(encoding="utf-8")
    assert "RUN sh /tmp/blas_gate.sh /opt/view/include /opt/view/lib" in docker


def test_the_cpu_image_installs_no_distribution_openblas() -> None:
    docker = (IMAGES / "judge-agent-cpu" / "Dockerfile").read_text(encoding="utf-8")
    assert "libopenblas-openmp-dev" not in docker
    assert "! dpkg -l | grep -E '^ii +libopenblas'" in docker


def test_the_overlay_drops_no_avx512_only_under_dynamic_dispatch() -> None:
    source = OVERLAY_PACKAGE.read_text(encoding="utf-8")
    assert 'if self.spec.satisfies("+dynamic_dispatch"):' in source
    assert 'defs = [d for d in defs if d != "NO_AVX512=1"]' in source


def test_the_gate_covers_the_failing_shapes_and_the_caller_count() -> None:
    gate = (LIB / "blas_gate.sh").read_text(encoding="utf-8")
    assert '"100000 406 815"' in gate and '"8192 406 512"' in gate
    assert "Haswell SkylakeX Zen" in gate
    assert '"${work}/callers" "$(nproc)"' in gate


@pytest.mark.parametrize("image", DOCKERFILES)
def test_numpy_and_scipy_are_rebuilt_on_the_view_openblas_under_omp_numba(image: str) -> None:
    """The wheels' bundled scipy-openblas (pthreads, MAX_THREADS=64) crashed under numba prange; the
    image rebuilds numpy/scipy against /opt/view's OpenBLAS after the python stack is installed, and
    numba runs its OpenMP layer so every BLAS caller shares one runtime."""
    docker = (IMAGES / image / "Dockerfile").read_text(encoding="utf-8")
    assert "ENV NUMBA_THREADING_LAYER=omp" in docker
    run = docker.index("RUN sh /tmp/one-openmp/numpy_on_openblas.sh /opt/view")
    install = docker.index("--group /opt/hpcagent-bench/pyproject.toml:judge-proxy")
    assert install < run, "the rebuild must follow the install that brings the wheels"


def test_the_numpy_rebuild_keeps_the_versions_and_gates_concurrent_callers() -> None:
    script = (LIB / "numpy_on_openblas.sh").read_text(encoding="utf-8")
    assert "--no-binary numpy" in script and "--no-binary scipy" in script
    assert "-Dblas=openblas" in script and "-Dlapack=openblas" in script
    assert '"numpy==${numpy_v}"' in script and '"scipy==${scipy_v}"' in script
    assert "numba.prange" in script and "2 * (os.cpu_count() or 1)" in script
