# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The EDF and the image's ENV PATH are one contract, so both are tested here.

At run time PATH is the image's own ENV (the CE reads the /etc/environment build_common.ce_squash_mounted
writes) with the launch venv in front: the ENTRYPOINT (containers/lib/launch_venv.sh) prepends it and names
its python in HPCAGENT_BENCH_IMAGE_PYTHON. The CE applies an EDF's [env] AFTER the ENTRYPOINT, so an EDF that
sets PATH, VIRTUAL_ENV or HPCAGENT_BENCH_IMAGE_PYTHON silently undoes the venv: every step then runs the image
python, which imports none of the packages installed at launch.

A missing PATH prefix has no diagnostic naming the cause -- it resolves to the distro copy and the run
continues with the wrong compiler or the wrong MPI. Each prefix below is here because dropping it produced
a real, silent failure. Without the spack MPICH in ``/opt/view/bin``, ``mpicc`` and ``mpiexec`` come from two
different MPIs and every rank becomes its own COMM_WORLD of size 1: P processes each solving the whole
problem, and nothing reports an error.
"""

import pathlib
import re

import pytest
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMAGES = ROOT / "containers" / "images"
#: Every judge and agent EDF of the images that start through the launch hook.
LAUNCH_EDFS = sorted(IMAGES.glob("judge-agent-*/*.edf.toml.in"))

#: Per image: the prefixes its ENV PATH must put ahead of the base image's (or of ``/usr/bin`` when it
#: spells the whole PATH), why, and the directory its named toolchain (CC/CXX/FC) lives in.
IMAGE_TOOLCHAIN: dict[str, tuple[dict[str, str], str]] = {
    "judge-agent-amd": (
        {
            "/opt/gcc/bin": "the pinned gcc; a bare c++ finds the distro gcc 13 instead",
            "/opt/view/bin": "the spack GPU-aware MPICH; a split wrapper/launcher gives P singleton ranks",
            "/opt/rocm/bin": "hipcc and the ROCm profilers",
        },
        "/opt/gcc/bin/",
    ),
    "judge-agent-cuda": (
        {
            "/opt/gcc/bin": "the pinned gcc; a bare c++ finds the distro gcc instead",
            "/opt/view/bin": "the spack MPICH; the base image's HPC-X Open MPI pairs with no MPICH mpiexec",
            "/usr/local/cuda/bin": "nvcc and the CUDA profilers",
        },
        "/opt/gcc/bin/",
    ),
    "judge-agent-cpu": (
        {
            "/usr/local/bin": "the gcc 16 symlinks, ahead of the distro gcc 13",
            "/opt/view/bin": "the spack MPICH",
            "/usr/lib/llvm-${LLVM_MAJOR}/bin": "clang and flang",
        },
        "/usr/bin/",
    ),
}


def image_path(image: str) -> list[str]:
    found = re.search(r"^ENV PATH=(\S+)", (IMAGES / image / "Dockerfile").read_text(), re.MULTILINE)
    assert found is not None, f"{image}/Dockerfile sets no ENV PATH"
    return found.group(1).split(":")


def edf_env(edf: pathlib.Path) -> dict[str, str]:
    return tomllib.loads(edf.read_text())["env"]


def edf_id(edf: pathlib.Path) -> str:
    return f"{edf.parent.name}/{edf.name}"


def test_every_launch_image_has_a_toolchain_entry() -> None:
    assert {edf.parent.name for edf in LAUNCH_EDFS} == set(IMAGE_TOOLCHAIN)


@pytest.mark.parametrize("image", sorted(IMAGE_TOOLCHAIN))
def test_the_image_path_puts_its_toolchain_ahead_of_the_base(image: str) -> None:
    entries = image_path(image)
    required = IMAGE_TOOLCHAIN[image][0]
    missing = {p: why for p, why in required.items() if p not in entries}
    assert not missing, f"{image}: the image PATH is missing load-bearing prefixes: {missing}"
    base = entries.index("${PATH}") if "${PATH}" in entries else entries.index("/usr/bin")
    assert all(entries.index(p) < base for p in required), entries


def test_the_gh200_image_keeps_the_base_images_open_mpi_off_path() -> None:
    """The NGC base ships HPC-X Open MPI in /usr/local/mpi/bin; on PATH it pairs an Open MPI mpicc
    with an MPICH mpiexec, and P ranks each come up as their own COMM_WORLD of size 1."""
    assert "/usr/local/mpi/bin" not in image_path("judge-agent-cuda")


@pytest.mark.parametrize("edf", LAUNCH_EDFS, ids=edf_id)
def test_an_edf_starts_the_launch_hook_and_never_undoes_it(edf: pathlib.Path) -> None:
    config = tomllib.loads(edf.read_text())
    assert config.get("entrypoint") is True
    assert not {"PATH", "VIRTUAL_ENV", "HPCAGENT_BENCH_IMAGE_PYTHON"} & set(config["env"]), edf


@pytest.mark.parametrize("edf", LAUNCH_EDFS, ids=edf_id)
def test_an_edf_binds_the_node_shm_with_exec_for_the_launch_venv(edf: pathlib.Path) -> None:
    """The CE mounts the container's own /dev/shm noexec; the launch venv lives on the node's."""
    assert "/dev/shm:/opt/node-shm" in tomllib.loads(edf.read_text())["mounts"], edf


@pytest.mark.parametrize("edf", LAUNCH_EDFS, ids=edf_id)
def test_toolchain_is_named_not_left_to_path(edf: pathlib.Path) -> None:
    # A stale configure cache beats PATH, so a toolchain that looks selected can still be ignored.
    env, toolchain = edf_env(edf), IMAGE_TOOLCHAIN[edf.parent.name][1]
    for var in ("CC", "CXX", "FC"):
        assert env[var].startswith(toolchain), f"{edf_id(edf)}: {var}={env[var]!r} is not under {toolchain}"


@pytest.mark.parametrize("edf", LAUNCH_EDFS, ids=edf_id)
def test_mpich_does_not_inherit_a_libfabric_provider(edf: pathlib.Path) -> None:
    """MPICH inherits FI_PROVIDER and MPI_Init aborts."""
    assert "FI_PROVIDER" not in edf_env(edf), edf


@pytest.mark.parametrize("edf", sorted(IMAGES.glob("judge-agent-amd/*.edf.toml.in")), ids=edf_id)
def test_rocm_libs_precede_the_distro_libdir(edf: pathlib.Path) -> None:
    # The distro libdir carries an ancient libhsa-runtime64 that leaves ROCR_1 symbols undefined.
    entries = edf_env(edf)["LD_LIBRARY_PATH"].split(":")
    assert entries.index("/opt/rocm/lib") < entries.index("/usr/lib/x86_64-linux-gnu")


@pytest.mark.parametrize("edf", LAUNCH_EDFS, ids=edf_id)
def test_cwd_is_off_sys_path(edf: pathlib.Path) -> None:
    """The image's own dace must win over anything mounted from the host.

    dace is installed editable, so `import dace` resolves through a finder -- and a plain
    DIRECTORY named `dace` on sys.path beats that finder, importing as an empty namespace
    package instead. sys.path starts with the CWD and the EDF's workdir is ${SCRATCH}, which
    holds the live extended checkout, so `import dace` there SUCCEEDS and returns a module with
    __file__ None and no SDFG (measured: unusable from ${SCRATCH} and from /opt, usable from
    /tmp). PYTHONSAFEPATH drops the CWD, so the container does not depend on where the job starts.
    """
    assert edf_env(edf).get("PYTHONSAFEPATH") == "1", (
        f"{edf_id(edf)}: PYTHONSAFEPATH=1 is missing: import dace from the workdir returns a broken "
        "namespace package shadowed by ${SCRATCH}/dace"
    )


@pytest.mark.parametrize(
    "template",
    sorted(IMAGES.glob("*/edf*.toml.example")),
    ids=lambda p: p.parent.name + "/" + p.name,
)
def test_every_image_names_its_interpreter_absolutely(template: pathlib.Path) -> None:
    """run_cluster.sh runs every role's Python through HPCAGENT_BENCH_IMAGE_PYTHON, never a PATH lookup."""
    python = tomllib.loads(template.read_text())["env"]["HPCAGENT_BENCH_IMAGE_PYTHON"]
    assert pathlib.PurePosixPath(python).is_absolute() and pathlib.PurePosixPath(python).name.startswith("python")
