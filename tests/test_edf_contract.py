# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The EDF is part of the image contract, so its PATH is worth a test.

The image's build gates assert that the toolchain it installed is the one on PATH. At run time PATH is
the image's own ENV (the CE reads the /etc/environment build_common.ce_squash_mounted writes) with the
launch venv in front: the ENTRYPOINT (containers/lib/launch_venv.sh) prepends it and names its python in
HPCAGENT_BENCH_IMAGE_PYTHON. The CE applies an EDF's [env] AFTER the ENTRYPOINT, so an EDF that sets
PATH, VIRTUAL_ENV or HPCAGENT_BENCH_IMAGE_PYTHON silently undoes the venv: every step then runs the image
python, which imports none of the packages installed at launch (verify job 667867).

The failure a missing PATH prefix causes has no diagnostic naming the cause -- it just resolves to the
distro copy and the run continues with the wrong compiler or the wrong MPI. Each entry below is here
because dropping it produced a real, silent failure:

* ``/opt/view/bin`` -- the spack MPICH built ``+rocm device=ch4 netmod=ofi``. Without it ``mpicc``
  and ``mpiexec`` come from two different MPIs and every rank becomes its own COMM_WORLD of size 1:
  P processes each solving the whole problem, and nothing reports an error.
* ``/opt/gcc/bin`` -- the pinned gcc. A bare ``c++`` otherwise finds the distro gcc 13.
* ``/opt/rocm/bin`` -- hipcc and the profilers.

The Dockerfile prepends them to the base image's PATH, so all of them precede ``/usr/bin``.
"""

import pathlib
import re

import pytest
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMAGES = ROOT / "containers" / "images"
EDF = IMAGES / "judge-agent-amd" / "agent.edf.toml.in"
#: Every judge and agent EDF of the images that start through the launch hook.
LAUNCH_EDFS = sorted(IMAGES.glob("judge-agent-*/*.edf.toml.in"))

#: Every prefix the AMD image's ENV PATH must put ahead of the base image's, and what silently breaks without it.
REQUIRED_PREFIXES = {
    "/opt/gcc/bin": "the pinned gcc; a bare c++ finds the distro gcc 13 instead",
    "/opt/view/bin": "the spack GPU-aware MPICH; a split wrapper/launcher gives P singleton ranks",
    "/opt/rocm/bin": "hipcc and the ROCm profilers",
}


@pytest.fixture(scope="module")
def env() -> dict[str, str]:
    return tomllib.loads(EDF.read_text())["env"]


def test_the_image_path_puts_its_toolchain_ahead_of_the_base() -> None:
    found = re.search(r"^ENV PATH=(\S+)", (IMAGES / "judge-agent-amd" / "Dockerfile").read_text(), re.MULTILINE)
    assert found is not None
    entries = found.group(1).split(":")
    missing = {p: why for p, why in REQUIRED_PREFIXES.items() if p not in entries}
    assert not missing, f"the image PATH is missing load-bearing prefixes: {missing}"
    assert all(entries.index(p) < entries.index("${PATH}") for p in REQUIRED_PREFIXES), entries


@pytest.mark.parametrize("edf", LAUNCH_EDFS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_an_edf_starts_the_launch_hook_and_never_undoes_it(edf: pathlib.Path) -> None:
    config = tomllib.loads(edf.read_text())
    assert config.get("entrypoint") is True
    assert not {"PATH", "VIRTUAL_ENV", "HPCAGENT_BENCH_IMAGE_PYTHON"} & set(config["env"]), edf


def test_toolchain_is_named_not_left_to_path(env: dict[str, str]) -> None:
    # A stale configure cache beats PATH, so a toolchain that looks selected can still be ignored.
    for var in ("CC", "CXX", "FC"):
        assert env[var].startswith("/opt/gcc/bin/"), f"{var}={env[var]!r} does not name the pinned gcc"


def test_rocm_libs_precede_the_distro_libdir(env: dict[str, str]) -> None:
    # The distro libdir carries an ancient libhsa-runtime64 that leaves ROCR_1 symbols undefined.
    entries = env["LD_LIBRARY_PATH"].split(":")
    assert entries.index("/opt/rocm/lib") < entries.index("/usr/lib/x86_64-linux-gnu")


def test_cwd_is_off_sys_path(env: dict[str, str]) -> None:
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


@pytest.mark.parametrize(
    "template",
    sorted((ROOT / "containers" / "images").glob("*/edf*.toml.example")),
    ids=lambda p: p.parent.name + "/" + p.name,
)
def test_every_image_names_its_interpreter_absolutely(template: pathlib.Path) -> None:
    """run_cluster.sh runs every role's Python through HPCAGENT_BENCH_IMAGE_PYTHON, never a PATH lookup."""
    python = tomllib.loads(template.read_text())["env"]["HPCAGENT_BENCH_IMAGE_PYTHON"]
    assert pathlib.PurePosixPath(python).is_absolute() and pathlib.PurePosixPath(python).name.startswith("python")
