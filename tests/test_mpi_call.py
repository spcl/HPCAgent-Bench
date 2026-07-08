# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The distributed build + runner (``sandbox.build_mpi``, ``languages.build_mpi_executable_commands``,
``agent_bench/mpi_call.run``).

``mpi_call.run`` is the 5th runner: it must return the SAME ``(outputs, native_ns)`` shape as the
single-node ``_call_isolated`` so the grading + metric core is reused verbatim. These tests pin
the build command shape (pure, no compiler) and -- gated on a working MPI toolchain -- the full
build -> scatter -> launch -> gather round-trip, asserting the reconstructed global output equals
the reference and that a launcher failure is a SCORED ``RuntimeError``.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from optarena.agent_bench import mpi_call
from optarena.agent_bench.envelope import Submission
from optarena.agent_bench.mpi_descriptor import ArrayDist, AxisDist, Descriptor, Grid
from optarena.agent_bench.sandbox import Sandbox
from optarena.agent_bench.task import Task
from optarena.bindings.contract import Arg, Binding
from optarena.bindings.stubs import LANGS
from optarena.languages import build_mpi_executable_commands
from tests.mpi_launch_helpers import c_toolchain, cc_override_for

RANKS = 4
_C_KERNEL = """
#include <mpi.h>
#include <stdint.h>
void yax_mpi(const double *restrict x, double *restrict y, const int64_t N, const double a,
             MPI_Fint comm, uint8_t *restrict workspace, const int64_t workspace_size) {
    for (int64_t i = 0; i < N; i++) y[i] = a * x[i];
}
"""


def _yax_binding() -> Binding:
    args = (
        Arg(name="x", kind="ptr", dtype="float64", is_const=True),
        Arg(name="y", kind="ptr", dtype="float64", is_const=False, role="output"),
        Arg(name="N", kind="scalar", dtype="int64", is_const=True, role="symbol"),
        Arg(name="a", kind="scalar", dtype="float64", is_const=True),
    )
    return Binding(kernel="yax", config="dense", args=args, symbols={lang: "yax_fp64" for lang in LANGS})


def _descriptor(locations=None) -> Descriptor:
    block0 = ArrayDist(axes=(AxisDist(grid_dim=0, scheme="block"), ))
    return Descriptor(grid=Grid((RANKS, )),
                      arrays={
                          "x": block0,
                          "y": block0
                      },
                      symbol_axes={"N": [("x", 0)]},
                      locations=locations or {})


# --------------------------------------------------------------------------------------- #
# build_mpi_executable_commands -- pure command shape (no compiler needed)
# --------------------------------------------------------------------------------------- #
def test_build_commands_compile_each_source_and_link_executable():
    cmds = build_mpi_executable_commands([("c", Path("k.c"))], Path("d.c"), Path("bench"))
    assert len(cmds) == 3  # compile kernel, compile driver, link
    # the MPICH C wrapper is the default; each source compiles to an object.
    assert cmds[0][0] == "mpicc.mpich" and "-c" in cmds[0] and str(Path("k.c")) in cmds[0]
    assert cmds[1][0] == "mpicc.mpich" and str(Path("d.c")) in cmds[1]
    # the link produces an EXECUTABLE (never -shared) and names the exe.
    link = cmds[-1]
    assert "-shared" not in " ".join(link)
    assert "-o" in link and str(Path("bench")) in link


def test_build_commands_cc_override_swaps_wrapper():
    cmds = build_mpi_executable_commands([("c", Path("k.c"))], Path("d.c"), Path("bench"), cc_override={"c": "mpicc"})
    assert all(argv[0] == "mpicc" for argv in cmds)  # OpenMPI wrapper for a matching launcher


def test_build_commands_fortran_kernel_links_with_fortran_driver():
    # A Fortran kernel + the always-C driver -> link with the Fortran wrapper (pulls libgfortran).
    cmds = build_mpi_executable_commands([("fortran", Path("k.f90"))], Path("d.c"), Path("bench"))
    assert cmds[0][0] == "mpifort.mpich"  # kernel compiled with the Fortran wrapper
    assert cmds[1][0] == "mpicc.mpich"  # driver always C
    assert cmds[-1][0] == "mpifort.mpich"  # link driver = Fortran


def test_build_commands_baseline_flows_from_matrix_no_literal_flags():
    cmds = build_mpi_executable_commands([("c", Path("k.c"))], Path("d.c"), Path("bench"))
    # -O3/-march come from the matrix baseline, never hard-coded in the yaml template.
    assert any("-O3" in tok for tok in cmds[0])


def test_build_commands_empty_sources_raises():
    with pytest.raises(ValueError, match="no kernel sources"):
        build_mpi_executable_commands([], Path("d.c"), Path("bench"))


def test_build_commands_device_routes_driver_and_link_to_gpu_compiler():
    # Device residency: a CUDA kernel + the (device) driver both compile with nvcc, and the link
    # is nvcc too (it auto-links libcudart) -- not the C wrapper.
    cmds = build_mpi_executable_commands([("cuda", Path("k.cu"))], Path("d.cu"), Path("bench"), driver_lang="cuda")
    assert cmds[0][0] == "nvcc" and str(Path("k.cu")) in cmds[0]  # kernel via nvcc
    assert cmds[1][0] == "nvcc" and str(Path("d.cu")) in cmds[1]  # driver via nvcc (not mpicc)
    assert cmds[-1][0] == "nvcc" and "-shared" not in " ".join(cmds[-1])  # link exe with nvcc


def test_mpi_wrapper_flags_extracts_include_and_link():
    # MPICH's `-show` line carries -I<mpi include> (compile) and -L/-l<mpi lib> (link); the helper
    # keeps exactly those so nvcc/hipcc (not MPI wrappers) can build MPI code. Gated on the wrapper.
    from optarena.languages import mpi_wrapper_flags
    if shutil.which("mpicc.mpich") is None:
        pytest.skip("mpicc.mpich unavailable")
    inc, link = mpi_wrapper_flags("mpicc.mpich")
    assert inc and all(t.startswith("-I") for t in inc)
    assert any(t.startswith("-l") for t in link) and all(t.startswith(("-L", "-l")) for t in link)
    assert not any(t.startswith("-Wl,") for t in link)  # wrapper hardening dropped (nvcc rejects it)


def test_mpi_wrapper_flags_missing_wrapper_is_empty():
    from optarena.languages import mpi_wrapper_flags
    assert mpi_wrapper_flags("definitely-not-a-real-compiler-xyz") == ([], [])


# --------------------------------------------------------------------------------------- #
# with_oversubscribe -- family-aware, idempotent launcher rewrite (pure, no launch)
# --------------------------------------------------------------------------------------- #
def test_oversubscribe_no_op_for_mpich_hydra():
    # Hydra oversubscribes by default AND rejects --oversubscribe (an OpenMPI-only flag), so the
    # config-default launcher must come back untouched.
    assert mpi_call.with_oversubscribe(["mpiexec.mpich", "-n"]) == ["mpiexec.mpich", "-n"]
    assert mpi_call.with_oversubscribe(["mpiexec", "-n"]) == ["mpiexec", "-n"]


def test_oversubscribe_adds_flag_for_openmpi_mpirun():
    # OpenMPI's mpirun REFUSES to oversubscribe without the flag; insert it before the -n tail.
    assert mpi_call.with_oversubscribe(["mpirun", "-n"]) == ["mpirun", "--oversubscribe", "-n"]
    assert mpi_call.with_oversubscribe(["mpirun.openmpi", "-n"]) == ["mpirun.openmpi", "--oversubscribe", "-n"]


def test_oversubscribe_is_idempotent():
    already = ["mpirun", "--oversubscribe", "-n"]
    assert mpi_call.with_oversubscribe(already) == already


def test_oversubscribe_leaves_srun_to_the_scheduler():
    # srun oversubscription is a site allocation concern (--overcommit), not this runner's job.
    assert mpi_call.with_oversubscribe(["srun", "--mpi=pmi2", "-n"]) == ["srun", "--mpi=pmi2", "-n"]
    assert mpi_call.with_oversubscribe([]) == []


# --------------------------------------------------------------------------------------- #
# Sandbox.build_mpi -- delivery handling
# --------------------------------------------------------------------------------------- #
def test_build_mpi_any_delivery_unsupported():
    b = _yax_binding()
    sub = Submission(language="c", library="/tmp/does-not-matter.so")
    with Sandbox(Task(kernel="yax"), b) as sb:
        res = sb.build_mpi(sub, _descriptor())
    assert not res.ok and "not supported" in res.log


def test_build_mpi_python_delivery_stashes_module():
    b = _yax_binding()
    sub = Submission(language="python", source="def kernel_mpi(*a, **k): pass\n")
    with Sandbox(Task(kernel="yax"), b) as sb:
        res = sb.build_mpi(sub, _descriptor())
        assert res.ok and res.exe is None and res.lib is not None
        assert res.lib.read_text().startswith("def kernel_mpi")


def test_build_mpi_device_rejects_non_gpu_kernel():
    # A GPU-located array (descriptor location=device) delivers GPU-pointer tiles, so a plain C
    # kernel_mpi (which would deref a device pointer on the host) is a clean build failure, not a
    # wrong build. Needs no compiler.
    b = _yax_binding()
    sub = Submission(language="c", source=_C_KERNEL)
    with Sandbox(Task(kernel="yax"), b) as sb:
        res = sb.build_mpi(sub, _descriptor(locations={"x": "device", "y": "device"}))
    assert not res.ok and "cuda/hip" in res.log


# --------------------------------------------------------------------------------------- #
# End to end: build -> scatter -> launch -> gather (gated on a working MPI toolchain)
# --------------------------------------------------------------------------------------- #
def test_build_mpi_and_run_round_trip(tmp_path):
    tc = c_toolchain()
    if tc is None:
        pytest.skip("no working MPI C compiler + launcher in this environment")
    cc, launch = tc
    b, desc = _yax_binding(), _descriptor()
    sub = Submission(language="c", source=_C_KERNEL)
    N = 20
    x = np.arange(N, dtype=np.float64) + 1.0
    data = {"x": x, "y": np.zeros(N), "N": N, "a": 3.0}

    with Sandbox(Task(kernel="yax"), b) as sb:
        built = sb.build_mpi(sub, desc, cc_override=cc_override_for(cc))
        assert built.ok, built.log
        assert built.exe is not None and built.exe.exists()
        outputs, native_ns = mpi_call.run(built.exe,
                                          b,
                                          desc,
                                          data,
                                          is_python=False,
                                          launcher=launch,
                                          k_repeats=5,
                                          timeout=60)
    assert set(outputs) == {"y"}  # only the output pointer is gathered
    assert np.allclose(outputs["y"], 3.0 * x)
    assert native_ns >= 0


def test_run_nonzero_exit_is_scored_runtimeerror(tmp_path):
    tc = c_toolchain()
    if tc is None:
        pytest.skip("no working MPI launcher in this environment")
    _cc, launch = tc
    b, desc = _yax_binding(), _descriptor()
    data = {"x": np.arange(8.0), "y": np.zeros(8), "N": 8, "a": 2.0}
    # A nonexistent executable -> the launcher exits non-zero -> a scored RuntimeError.
    with pytest.raises(RuntimeError):
        mpi_call.run(tmp_path / "no_such_bench",
                     b,
                     desc,
                     data,
                     is_python=False,
                     launcher=launch,
                     k_repeats=1,
                     timeout=30)


# --- device residency (E1): the launch argv + the H2D/D2H staging ---------------------------------


def _cuda_available() -> bool:
    """A usable NVIDIA device + cupy attached to it (the device-residency e2e gate)."""
    import importlib.util
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 -- no usable device
        return False


def test_program_argv_python_forwards_device_mask_only_for_device():
    """The launcher's program tail: ``--device-mask <csv>`` (the GPU-located pointer indices) rides
    the mpi4py driver invocation ONLY when some array is device (per-array residency); an empty mask
    adds nothing, and the C ``bench`` tail (exe infile outfile) never carries it (the C driver bakes
    the mask at build time)."""
    art, inf, out = Path("/x/bench"), Path("/t/in.bin"), Path("/t/out.bin")
    host = mpi_call._program_argv(art, inf, out, is_python=True, python_exe="py", grid_dims=(4, ), device_mask=())
    dev = mpi_call._program_argv(art, inf, out, is_python=True, python_exe="py", grid_dims=(2, 2), device_mask=(0, 2))
    assert host[:3] == ["py", "-m", mpi_call.PY_DRIVER_MODULE] and "--device-mask" not in host
    assert dev[6] == "2,2" and dev[-2:] == ["--device-mask", "0,2"]  # grid forwarded, then the mask
    c = mpi_call._program_argv(art, inf, out, is_python=False, python_exe="py", grid_dims=(4, ), device_mask=(0, ))
    assert c == ["/x/bench", "/t/in.bin", "/t/out.bin"]  # the mask never leaks into the C program tail


def test_stage_host_returns_numpy_and_sizes_workspace():
    """``_stage`` all-host path (empty on_device): the compute tiles are the scattered host arrays
    and the workspace is None for a 0-byte request or an uninitialised ``uint8`` buffer."""
    from optarena.agent_bench import mpi_py_driver
    tiles = [np.arange(4, dtype=np.float64)]
    compute, ws = mpi_py_driver._stage(tiles, 0, frozenset())
    assert compute[0] is tiles[0] and ws is None
    _c, ws2 = mpi_py_driver._stage(tiles, 32, frozenset())
    assert ws2.shape == (32, ) and ws2.dtype == np.uint8


def test_stage_device_mask_copies_only_selected_tiles():
    """``_stage`` per-array path: only the tiles in ``on_device`` become cupy (H2D); a host-located
    tile stays numpy. Gated on a usable GPU; reuses the single-node device marshalling contract."""
    if not _cuda_available():
        pytest.skip("no CUDA device / cupy")
    import cupy as cp

    from optarena.agent_bench import mpi_py_driver
    tiles = [np.arange(4, dtype=np.float64), np.arange(4, 8, dtype=np.float64)]
    compute, ws = mpi_py_driver._stage(tiles, 16, frozenset({0}))  # only tile 0 on device
    assert isinstance(compute[0], cp.ndarray) and isinstance(compute[1], np.ndarray)  # mixed residency
    assert isinstance(ws, cp.ndarray)  # any device tile -> device scratch
    assert cp.asnumpy(compute[0]).tolist() == [0, 1, 2, 3]  # H2D preserved the values
