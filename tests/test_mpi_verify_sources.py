# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The MPI / RCCL verification suite (experiments/mpi/verify) stays runnable between GPU runs.

The suite's real checks need mi300 nodes (verify.sbatch). What a CPU host can hold: both scripts
parse, the flag resolver prints every assignment verify.sh evals, the C probe parses, every probe
still asks the question it exists for, and every verdict a program prints is one the results table
can carry. The ``amd`` group adds the HIP probes' syntax check against the image's own mpi.h with
the harness's resolved flags.
"""

import functools
import importlib.util
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from hpcagent_bench import languages

VERIFY = Path(__file__).resolve().parents[1] / "experiments" / "mpi" / "verify"
HIP_PROBES = ("gpuaware.hip", "gpu_initiated.hip", "rccl_allreduce.hip")

#: The MPI surface mpi_hello.c uses, declared just far enough for a syntax-only parse on a host
#: without MPI. The real header is exercised by verify.sh's build step on the judge image.
MPI_STUB = """
typedef int MPI_Comm; typedef int MPI_Datatype; typedef int MPI_Op; typedef int MPI_Info;
#define MPI_COMM_WORLD 0
#define MPI_COMM_TYPE_SHARED 1
#define MPI_INFO_NULL 0
#define MPI_INT 1
#define MPI_SUM 1
#define MPI_MAX_PROCESSOR_NAME 256
#define MPI_MAX_LIBRARY_VERSION_STRING 8192
int MPI_Init(int *, char ***); int MPI_Finalize(void);
int MPI_Comm_rank(MPI_Comm, int *); int MPI_Comm_size(MPI_Comm, int *); int MPI_Comm_free(MPI_Comm *);
int MPI_Get_processor_name(char *, int *); int MPI_Get_library_version(char *, int *);
int MPI_Comm_split_type(MPI_Comm, int, int, MPI_Info, MPI_Comm *);
int MPI_Allreduce(const void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm); int MPI_Barrier(MPI_Comm);
"""


@functools.lru_cache(maxsize=1)
def resolver() -> ModuleType:
    """experiments/mpi/verify/resolve_flags.py, loaded by path (experiments/ is not a package)."""
    spec = importlib.util.spec_from_file_location("mpi_verify_resolve_flags", VERIFY / "resolve_flags.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", ["verify.sh", "verify.sbatch"])
def test_the_driver_scripts_parse_as_bash(script: str) -> None:
    proc = subprocess.run(["bash", "-n", str(VERIFY / script)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_the_resolver_prints_every_assignment_verify_sh_evals() -> None:
    """verify.sh evals these lines and reads each name; a missing one would compile with an empty flag."""
    lines = resolver().assignments()
    keys = {line.split("=", 1)[0] for line in lines}
    expected = set()
    for lang, libs in resolver().LIBRARIES.items():
        expected |= {f"{lang.upper()}_CC", f"{lang.upper()}_FLAGS"}
        expected |= {
            f"{lang.upper()}_{lib.upper()}_{field}" for lib in libs for field in ("OFFERED", "COMPILE", "LINK")
        }
    assert keys == expected
    for line in lines:
        assert len(shlex.split(line)) == 1, line  # one shell word per line: a safe `eval`


def test_the_resolver_reports_what_the_harness_offers() -> None:
    """OFFERED mirrors languages.library_offered, so an offered_* FAIL means the harness refuses the library."""
    values = dict(line.split("=", 1) for line in resolver().assignments())
    assert values["HIP_RCCL_OFFERED"] == str(int(languages.library_offered("rccl", "hip")))
    assert values["HIP_MPI_OFFERED"] == str(int(languages.library_offered("mpi", "hip")))
    assert values["C_MPI_OFFERED"] == str(int(languages.library_offered("mpi", "c")))


def test_the_c_probe_parses(tmp_path: Path) -> None:
    (tmp_path / "mpi.h").write_text(MPI_STUB)
    cmd = [
        "gcc",
        "-fsyntax-only",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{tmp_path}",
        str(VERIFY / "mpi_hello.c"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    ("source", "must_use"),
    [
        ("gpuaware.hip", ("MPIX_GPU_query_support(MPIX_GPU_SUPPORT_HIP", "hipPointerGetAttributes", "MPI_T_cvar_read")),
        (
            "gpu_initiated.hip",
            (
                '"hipStream_t"',
                "MPIX_Stream_create",
                "MPIX_Stream_comm_create",
                "MPIX_Send_enqueue",
                "MPIX_Recv_enqueue",
                "MPIX_Allreduce_enqueue",
                "V_UNSUPPORTED",
            ),
        ),
        ("rccl_allreduce.hip", ("ncclBfloat16", "ncclFloat", "MPI_Bcast(&id", "busbw")),
    ],
)
def test_each_probe_still_asks_its_question(source: str, must_use: tuple[str, ...]) -> None:
    text = (VERIFY / source).read_text()
    missing = [token for token in must_use if token not in text]
    assert not missing, f"{source} no longer uses {missing}"


def test_every_check_prints_a_one_word_verdict_name() -> None:
    """report splits `VERDICT <test> <result> <detail>` on whitespace, so each check's name must be one word."""
    names = set()
    for path in sorted(VERIFY.glob("*.c")) + sorted(VERIFY.glob("*.hip")):
        names |= set(re.findall(r"VERDICT (\w+)", path.read_text()))
    names |= set(re.findall(r"verdict \"?(\w+)", (VERIFY / "verify.sh").read_text()))
    assert {"mpi_hello", "gpuaware", "gpuaware_control", "gpu_initiated", "rccl_"} <= names
    assert {"nested_srun", "nested_mpi_gang", "compile_", "offered_", "ldd_"} <= names


@pytest.mark.amd
@pytest.mark.parametrize("source", HIP_PROBES)
def test_the_hip_probes_parse_with_the_resolved_flags(source: str) -> None:
    """On the judge image: the harness's own hip driver + mpi / rccl tokens parse every probe."""
    driver = languages.submission_toolchain("hip").driver
    compile_tokens = languages.library_build_flags("hip", ["mpi", "rccl"])[0]
    cmd = [shutil.which(driver) or driver, "-fsyntax-only", languages.std_flag("hip"), f"-I{VERIFY}", *compile_tokens]
    proc = subprocess.run([*cmd, str(VERIFY / source)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
