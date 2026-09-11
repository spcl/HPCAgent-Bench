# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The EDF is part of the image contract, so its PATH is worth a test.

The image's build gates assert that the toolchain it installed is the one on PATH. None of that
survives into a run: the CSCS Container Engine does not reliably preserve the image's own ENV, so
the EDF restates PATH absolutely, and anything the EDF omits is silently gone no matter what the
build proved. The failure has no diagnostic naming the cause -- a missing prefix just resolves to
the distro copy and the run continues with the wrong compiler, the wrong MPI, or an interpreter
that imports none of what was installed.

Each entry below is here because dropping it produced a real, silent failure:

* ``/opt/venv/bin`` -- the rocm/pytorch base ships a venv on PATH, so every ``python3 -m pip
  install`` in the Dockerfile (torch, cupy, the editable dace) lands in ``/opt/venv/lib``.
  ``PIP_BREAK_SYSTEM_PACKAGES=1`` on those lines defeats PEP 668; it does not redirect the install.
  Without this entry ``python3`` is ``/usr/bin/python3`` and the judge dies at ``import dace``.
* ``/opt/view/bin`` -- the spack MPICH built ``+rocm device=ch4 netmod=ofi``. Without it ``mpicc``
  and ``mpiexec`` come from two different MPIs and every rank becomes its own COMM_WORLD of size 1:
  P processes each solving the whole problem, and nothing reports an error.
* ``/opt/gcc/bin`` -- the pinned gcc. A bare ``c++`` otherwise finds the distro gcc 13.
* ``/opt/rocm/bin`` -- hipcc and the profilers.

Ordering matters as much as presence: all of them must precede ``/usr/bin``, or the distro copy
wins and the entry is decoration.
"""

from __future__ import annotations

import pathlib
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EDF = ROOT / "containers" / "cluster" / "ce-images" / "judge-agent-amd" / "edf.toml.example"

#: Every prefix that must be on PATH ahead of the distro, and what silently breaks without it.
REQUIRED_PREFIXES = {
    "/opt/gcc/bin": "the pinned gcc; a bare c++ finds the distro gcc 13 instead",
    "/opt/view/bin": "the spack GPU-aware MPICH; a split wrapper/launcher gives P singleton ranks",
    "/opt/rocm/bin": "hipcc and the ROCm profilers",
    "/opt/venv/bin": "the base image venv holding torch, cupy and dace; without it import dace fails",
}


@pytest.fixture(scope="module")
def env() -> dict:
    return tomllib.loads(EDF.read_text())["env"]


def test_path_entries_are_present(env) -> None:
    missing = {p: why for p, why in REQUIRED_PREFIXES.items() if p not in env["PATH"].split(":")}
    assert not missing, f"EDF PATH is missing load-bearing prefixes: {missing}"


def test_path_entries_precede_the_distro(env) -> None:
    entries = env["PATH"].split(":")
    distro = entries.index("/usr/bin")
    late = [p for p in REQUIRED_PREFIXES if p in entries and entries.index(p) > distro]
    assert not late, f"these resolve to the distro copy because they follow /usr/bin: {late}"


def test_toolchain_is_named_not_left_to_path(env) -> None:
    # A stale configure cache beats PATH, so a toolchain that looks selected can still be ignored.
    for var in ("CC", "CXX", "FC"):
        assert env[var].startswith("/opt/gcc/bin/"), f"{var}={env[var]!r} does not name the pinned gcc"


def test_rocm_libs_precede_the_distro_libdir(env) -> None:
    # The distro libdir carries an ancient libhsa-runtime64 that leaves ROCR_1 symbols undefined.
    entries = env["LD_LIBRARY_PATH"].split(":")
    assert entries.index("/opt/rocm/lib") < entries.index("/usr/lib/x86_64-linux-gnu")


def test_cwd_is_off_sys_path(env) -> None:
    """The image's own dace must win over anything mounted from the host.

    dace is installed editable, so `import dace` resolves through a finder -- and a plain
    DIRECTORY named `dace` on sys.path beats that finder, importing as an empty namespace
    package instead. sys.path starts with the CWD and the EDF's workdir is ${SCRATCH}, which
    holds the live extended checkout, so `import dace` there SUCCEEDS and returns a module with
    __file__ None and no SDFG. Measured on v6: unusable from ${SCRATCH} and from /opt, usable
    from /tmp. PYTHONSAFEPATH drops the CWD, so the container stops depending on where the job
    happened to start.
    """
    assert env.get("PYTHONSAFEPATH") == "1", (
        "PYTHONSAFEPATH=1 is missing: import dace from the workdir returns a broken namespace "
        "package shadowed by ${SCRATCH}/dace"
    )
