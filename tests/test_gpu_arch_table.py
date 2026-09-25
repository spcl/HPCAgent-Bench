# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One partition table names the AMD GPU arch; builds, gates, EDF templates and the harness take it from there.

containers/images/gpu_arch.env maps a Slurm partition to the gfx arch of its GPUs. These pin
the table and its shell lookup, the absence of gfx literals wherever an arch could be spelled instead,
the rendered EDF arch variables, the runtime three-way check on a stub srun, the device-code gate on
stand-in binaries, and detect_gfx refusing to guess.
"""

import ast
import pathlib
import re
import subprocess
import tomllib
from collections.abc import Iterator

import pytest

from hpcagent_bench import flags, languages

ROOT = pathlib.Path(__file__).resolve().parents[1]
CE = ROOT / "containers" / "images"
TABLE = CE / "gpu_arch.env"
GATE = ROOT / "containers" / "lib" / "device_arch_gate.sh"
CHECK = CE / "gpu_arch_check.sh"
SHELL_PATH = "/usr/bin:/bin"
#: A gfx arch spelled out.
GFX_LITERAL = re.compile(r"\bgfx[0-9a-f]{3,4}\b")
#: The AMD image directories; each builds with ROCM_ARCH from the table.
AMD_IMAGES = ("judge-agent-amd", "sglang", "sglang-mi200", "vllm")
#: Image directories outside the table, with the reason.
NOT_AMD = {
    "judge-agent-cuda": "GH200 image built on another Alps cluster; its arch is a CUDA capability",
    "vllm-cuda": "GH200 inference image built on another Alps cluster; its arch is a CUDA capability",
    "judge-agent-cpu": "CPU-only image with no GPU code at all, built on whichever host architecture",
}
#: The arch variables an image ENV sets and an EDF template may restate.
ARCH_VARS = ("HCC_AMDGPU_TARGET", "PYTORCH_ROCM_ARCH", "GPU_ARCHS", "GPU_ARCH_LIST")
#: Non-comment gfx literals that must stay, keyed by (file, stripped line), with the reason.
LITERAL_EXCEPTIONS = {
    (
        "containers/images/sglang-mi200/Dockerfile",
        r"""ALLOW = 'if amdgpu_target not in ["gfx942", "gfx950", "gfx1250"]:\n'""",
    ): "upstream setup_rocm.py allow-list line, matched verbatim so the edit fails when upstream changes it",
}
#: rocminfo with a CPU agent first and one GPU agent, whose arch is filled in.
ROCMINFO = """\
*******
Agent 1
*******
  Name:                    AMD EPYC 7A53 64-Core Processor
  Marketing Name:          AMD EPYC 7A53 64-Core Processor
*******
Agent 2
*******
  Name:                    {arch}
  Marketing Name:          AMD Instinct
      Name:                amdgcn-amd-amdhsa--{arch}:sramecc+:xnack-
"""
#: rocprim's arch-name table as it sits in a rocprim-using .so: metadata, not device code.
NAME_TABLE = ("gfx803 gfx900 gfx906 gfx908 gfx90a gfx942 gfx950 gfx1030", "gfx90a", "gfx942")
#: srun that answers the two in-image commands gpu_arch_check.sh runs, and logs each call.
STUB_SRUN = """#!/bin/sh
echo "$*" >> "$STUB_LOG"
case "$*" in
    *gpu-arch*) [ -z "$STUB_STAMP" ] || printf '%s\\n' "$STUB_STAMP" ;;
    *rocminfo*) printf '%s' "$STUB_ROCMINFO" ;;
    *) exit 99 ;;
esac
"""


def table() -> dict[str, str]:
    """gpu_arch.env as {partition: arch}."""
    rows: dict[str, str] = {}
    for line in TABLE.read_text(encoding="ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        assert sep and key.startswith("GPU_ARCH_"), f"not a GPU_ARCH_<partition>=<arch> row: {line!r}"
        assert key.removeprefix("GPU_ARCH_") not in rows, f"partition named twice: {line!r}"
        rows[key.removeprefix("GPU_ARCH_")] = value
    return rows


def build_partitions(image: str) -> list[str]:
    """The partition column of the images.env row named after an image directory: the hardware its
    build.sbatch builds for (the Slurm partition itself comes from the site layer)."""
    done = run(["bash", "-c", 'source "$1"; printf "%s\\n" "${CE_IMAGE_TABLE}"', "bash", str(CE / "images.env")], {})
    assert done.returncode == 0, done.stderr
    rows = [line.split() for line in done.stdout.splitlines() if line.strip()]
    return [row[4] for row in rows if row[0] == image and row[4] != "-"]


def run(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False, env={"PATH": SHELL_PATH, **env})


def code_lines(path: pathlib.Path) -> str:
    """The file without its comment lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def test_every_table_row_maps_a_partition_to_one_gfx_arch() -> None:
    rows = table()
    assert {"mi300", "mi200"} <= set(rows), rows
    bad = {partition: arch for partition, arch in rows.items() if not re.fullmatch(r"gfx[0-9a-f]{3,4}", arch)}
    assert not bad, f"not a gfx arch: {bad}"


CE_GPU_ARCH = 'source "$1"; ce_gpu_arch || exit $?; printf "ROCM_ARCH=%s\\n" "${ROCM_ARCH}"'


@pytest.mark.parametrize("variable", ["SLURM_JOB_PARTITION", "ROCM_PARTITION"])
def test_ce_gpu_arch_exports_the_table_arch_of_the_job_partition(variable: str) -> None:
    for partition, arch in table().items():
        done = run(["bash", "-c", CE_GPU_ARCH, "bash", str(CE / "build_common.sh")], {variable: partition})
        assert done.returncode == 0, done.stderr
        assert f"ROCM_ARCH={arch}" in done.stdout.splitlines()


@pytest.mark.parametrize(
    ("env", "reason"),
    [
        ({}, "no SLURM_JOB_PARTITION"),
        ({"SLURM_JOB_PARTITION": "normal"}, "names no GPU arch for partition 'normal'"),
        ({"SLURM_JOB_PARTITION": "mi300", "ROCM_PARTITION": "mi200"}, "but this job runs on mi300"),
        ({"SLURM_JOB_PARTITION": "mi300", "ROCM_ARCH": "gfx000"}, "disagrees with gpu_arch.env"),
    ],
)
def test_ce_gpu_arch_refuses_a_missing_unknown_or_contradicted_partition(env: dict[str, str], reason: str) -> None:
    done = run(["bash", "-c", CE_GPU_ARCH, "bash", str(CE / "build_common.sh")], env)
    assert done.returncode == 2
    assert reason in done.stderr
    assert "ROCM_ARCH=" not in done.stdout


def test_every_amd_image_builds_on_exactly_one_partition_the_table_names() -> None:
    images = {path.parent.name for path in CE.glob("*/build.sbatch")}
    assert images == set(AMD_IMAGES) | set(NOT_AMD), images
    rows = table()
    for image in AMD_IMAGES:
        partitions = build_partitions(image)
        assert len(partitions) == 1 and partitions[0] in rows, (image, partitions)
    for image in NOT_AMD:
        assert "ROCM_ARCH" not in (CE / image / "Dockerfile").read_text(encoding="utf-8"), image


@pytest.mark.parametrize("image", AMD_IMAGES)
def test_every_amd_image_takes_rocm_arch_from_the_table_refuses_none_stamps_it_and_overrides_the_base_env(
    image: str,
) -> None:
    build = code_lines(CE / image / "build.sh")
    assert re.search(r"^ce_gpu_arch$", build, re.M), f"{image}/build.sh never looks the arch up"
    assert '--build-arg "ROCM_ARCH=${ROCM_ARCH}"' in build
    docker = code_lines(CE / image / "Dockerfile")
    assert re.findall(r"^ARG ROCM_ARCH\b.*$", docker, re.M) == ["ARG ROCM_ARCH"]
    assert 'test -n "${ROCM_ARCH:-}" ||' in docker
    assert "printf '%s\\n' \"${ROCM_ARCH}\" > /opt/gpu-arch" in docker
    for var in ARCH_VARS:
        values = {value.strip('"') for value in re.findall(rf"\b{var}=(\S+)", docker)}
        assert values == {"${ROCM_ARCH}"}, (image, var, values)
    assert "COPY containers/lib/device_arch_gate.sh /usr/local/bin/device_arch_gate.sh" in docker
    assert '/usr/local/bin/device_arch_gate.sh --exact "${ROCM_ARCH}"' in docker


def test_no_dockerfile_gives_rocm_arch_a_default() -> None:
    dockerfiles = sorted((ROOT / "containers").rglob("Dockerfile"))
    assert dockerfiles
    pattern = re.compile(r"^\s*ARG\s+ROCM_ARCH\s*=", re.M)
    assert [str(path) for path in dockerfiles if pattern.search(path.read_text(encoding="utf-8"))] == []


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
    return [name for name in out.decode().split("\0") if name]


def in_literal_scope(rel: str) -> bool:
    """The files where a spelled-out arch would be a second source of truth."""
    name = rel.rsplit("/", 1)[-1]
    parts = rel.split("/")
    if rel == "containers/images/gpu_arch.env" or name.endswith(".md") or {"skills", "tests"} & set(parts):
        return False
    if name in ("Dockerfile", "build.sh") or name.endswith(".sbatch") or re.fullmatch(r"edf.*\.toml\.example", name):
        return True
    if parts[0] == "hpcagent_bench":
        return name.endswith(".py")
    return parts[0] in ("experiments", "scripts", "reproducibility")


def python_literals(text: str) -> list[str]:
    """String constants that are not docstrings or other bare string statements."""
    tree = ast.parse(text)
    bare = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Expr)}
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in bare
    ]


def code_pieces(rel: str, text: str) -> list[str]:
    """What a file says outside its comments: string literals for Python, comment-stripped lines otherwise."""
    if rel.endswith(".py"):
        return python_literals(text)
    return [re.sub(r"(^|\s)#.*$", "", line).strip() for line in text.splitlines()]


def test_no_gfx_arch_is_spelled_outside_the_table_in_builds_launchers_scripts_or_the_harness() -> None:
    found: list[str] = []
    scanned = 0
    for rel in tracked_files():
        path = ROOT / rel
        if not in_literal_scope(rel) or not path.is_file():
            continue
        scanned += 1
        text = path.read_bytes().decode("utf-8", "replace")
        if not GFX_LITERAL.search(text):
            continue
        for piece in code_pieces(rel, text):
            if GFX_LITERAL.search(piece) and (rel, piece) not in LITERAL_EXCEPTIONS:
                found.append(f"{rel}: {piece}")
    assert scanned > 100, f"only {scanned} files in scope; the scan checks nothing"
    assert not found, "gfx arch literals outside gpu_arch.env:\n" + "\n".join(found)


def rendered_edfs(tmp_path: pathlib.Path) -> dict[str, dict[str, str]]:
    """install_edfs.sh run on stand-in images, as {template: rendered [env]}."""
    roles = ("JUDGE_AGENT_AMD", "JUDGE_AMD", "INFERENCE_SGLANG", "INFERENCE_VLLM", "INFERENCE_SGLANG_MI200")
    names = 'source "$1"; shift; for r in "$@"; do for s in SQSH EDF_LATEST TEMPLATE; do n="${r}_${s}"; echo "${!n}"; done; done'
    listed = run(["bash", "-c", names, "bash", str(CE / "images.env"), *roles], {})
    assert listed.returncode == 0, listed.stderr
    fields = listed.stdout.splitlines()
    ce, edf_dir = tmp_path / "ce", tmp_path / "edf"
    ce.mkdir()
    for sqsh in fields[0::3]:
        (ce / sqsh).write_bytes(b"sqsh")
    env = {"HOME": str(tmp_path), "SCRATCH": str(tmp_path), "CE_IMAGES": str(ce), "EDF_DIR": str(edf_dir)}
    done = run(["bash", str(CE / "install_edfs.sh")], env)
    assert done.returncode == 0, done.stderr
    return {
        template: tomllib.loads((edf_dir / f"{edf}.toml").read_text(encoding="utf-8"))["env"]
        for edf, template in zip(fields[1::3], fields[2::3])
    }


def test_rendered_edf_arch_variables_equal_the_table_arch_of_the_partition_their_image_builds_on(
    tmp_path: pathlib.Path,
) -> None:
    rows = table()
    checked = 0
    for template, env in rendered_edfs(tmp_path).items():
        source = (CE / template).read_text(encoding="utf-8")
        arch = rows[build_partitions(template.split("/")[0])[0]]
        for var in ARCH_VARS:
            if var in env:
                assert f'{var} = "${{GPU_ARCH}}"' in source, (template, var)
                assert env[var] == arch, (template, var, env[var])
                checked += 1
    assert checked >= 4, "no rendered EDF restates an arch variable; the test checks nothing"


def run_check(tmp_path: pathlib.Path, partition: str, stamp: str, gpu: str) -> subprocess.CompletedProcess[str]:
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    (stub / "srun").write_text(STUB_SRUN, encoding="utf-8")
    (stub / "srun").chmod(0o755)
    env = {
        "PATH": f"{stub}:{SHELL_PATH}",
        "SLURM_JOB_PARTITION": partition,
        "STUB_LOG": str(tmp_path / "srun.log"),
        "STUB_STAMP": stamp,
        "STUB_ROCMINFO": ROCMINFO.format(arch=gpu),
    }
    return subprocess.run(["bash", str(CHECK), "the-edf"], capture_output=True, text=True, check=False, env=env)


def test_the_runtime_check_passes_when_table_stamp_and_gpu_agree(tmp_path: pathlib.Path) -> None:
    arch = table()["mi200"]
    done = run_check(tmp_path, "mi200", arch, arch)
    assert done.returncode == 0, done.stderr
    assert f"the-edf on mi200: {arch}" in done.stdout
    calls = (tmp_path / "srun.log").read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2 and all("--overlap" in call and "--environment=the-edf" in call for call in calls), calls


@pytest.mark.parametrize("wrong", ["stamp", "gpu"])
def test_the_runtime_check_exits_2_and_prints_all_three_when_one_source_disagrees(
    tmp_path: pathlib.Path, wrong: str
) -> None:
    rows = table()
    right, other = rows["mi300"], rows["mi200"]
    stamp, gpu = (other, right) if wrong == "stamp" else (right, other)
    done = run_check(tmp_path, "mi300", stamp, gpu)
    assert done.returncode == 2
    assert f"gpu_arch.env[mi300] = {right}" in done.stderr
    assert f"/opt/gpu-arch = {stamp}" in done.stderr
    assert f"rocminfo = {gpu}" in done.stderr


def test_the_runtime_check_warns_and_passes_an_image_built_before_the_stamp(tmp_path: pathlib.Path) -> None:
    done = run_check(tmp_path, "mi300", "", table()["mi200"])
    assert done.returncode == 0
    assert "WARNING" in done.stderr and "no /opt/gpu-arch" in done.stderr
    assert len((tmp_path / "srun.log").read_text(encoding="utf-8").splitlines()) == 1, "rocminfo ran without a stamp"


def test_the_runtime_check_refuses_a_partition_the_table_does_not_name(tmp_path: pathlib.Path) -> None:
    arch = table()["mi300"]
    done = run_check(tmp_path, "normal", arch, arch)
    assert done.returncode == 2
    assert "names no GPU arch for partition 'normal'" in done.stderr
    assert not (tmp_path / "srun.log").exists()


@pytest.mark.parametrize(
    ("launcher", "check", "first_gpu_step"),
    [
        ("experiments/run_cluster.sh", "\ncheck_gpu_arch\n", 'role_srun "${INFERENCE_NODES}"'),
        (
            "containers/images/verify_image.sbatch",
            'gpu_arch_check.sh" "${EDF}"',
            'srun --environment="${EDF}"',
        ),
        (
            "containers/inference/serve-private.sbatch",
            'gpu_arch_check.sh" "${EDF}"',
            'make_private_dir "${RUN_DIR}"',
        ),
        (
            "experiments/mpi/smoke-mlscale-e2e.sbatch",
            'gpu_arch_check.sh" "${EDF}"',
            'srun --overlap --nodes=1 --ntasks=1 --nodelist="${NODE}"',
        ),
        (
            "experiments/mpi/smoke-mpi-judge.sbatch",
            'gpu_arch_check.sh" "${EDF}"',
            'srun --ntasks=1 --cpus-per-task=24 --environment="${EDF}"',
        ),
    ],
)
def test_every_gpu_launcher_checks_the_arch_before_its_first_gpu_step(
    launcher: str, check: str, first_gpu_step: str
) -> None:
    text = (ROOT / launcher).read_text(encoding="utf-8")
    assert text.count(check) == 1, f"{launcher} must check the GPU arch exactly once"
    assert text.find(first_gpu_step) > text.find(check), f"{launcher} starts a GPU step before checking the arch"


def fake_so(path: pathlib.Path, *strings: str) -> pathlib.Path:
    """A stand-in shared object: an ELF magic and NUL-separated strings, the way `strings -a` sees them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"".join(s.encode("ascii") + b"\x00\x01\x00" for s in strings))
    return path


def gate(mode: str, arch: str, *paths: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return run(["bash", str(GATE), mode, arch, *map(str, paths)], {})


def test_the_gate_passes_device_code_for_exactly_the_arch_beside_host_objects_and_the_name_table(
    tmp_path: pathlib.Path,
) -> None:
    cupy = tmp_path / "cupy"
    fake_so(cupy / "cuda" / "cub.so", "hipv4-amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-", *NAME_TABLE)
    fake_so(cupy / "random" / "_generator_api.so", "amdgcn-amd-amdhsa--gfx942")
    fake_so(cupy / "_core" / "core.so", "PyInit_core", "cupy._core.core")
    done = gate("--exact", "gfx942", cupy)
    assert done.returncode == 0, done.stderr
    assert "device_arch_gate --exact gfx942: OK" in done.stdout


def test_the_gate_names_the_file_that_carries_device_code_for_another_arch(tmp_path: pathlib.Path) -> None:
    right = fake_so(tmp_path / "ops" / "right.so", "amdgcn-amd---gfx90a", *NAME_TABLE)
    wrong = fake_so(tmp_path / "ops" / "wrong.so", "hipv4-amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-")
    done = gate("--exact", "gfx90a", tmp_path / "ops")
    assert done.returncode == 1
    assert f"{wrong} carries device code for gfx942 expected only gfx90a" in done.stderr
    assert f"{right} carries" not in done.stderr


@pytest.mark.parametrize("mode", ["--exact", "--contains"])
def test_the_rocprim_name_table_alone_is_not_device_code(tmp_path: pathlib.Path, mode: str) -> None:
    table_only = fake_so(tmp_path / "thrust.so", *NAME_TABLE, *NAME_TABLE)
    done = gate(mode, "gfx90a", table_only)
    assert done.returncode == 1
    assert f"no device code for gfx90a in: {table_only}" in done.stderr


def test_contains_accepts_a_vendor_fat_binary_that_exact_refuses(tmp_path: pathlib.Path) -> None:
    fat = fake_so(
        tmp_path / "librocvendor.so.1.0",
        "amdgcn-amd-amdhsa--gfx90a",
        "amdgcn-amd-amdhsa--gfx942",
        "amdgcn-amd-amdhsa--gfx1030",
    )
    assert gate("--contains", "gfx90a", fat).returncode == 0
    assert gate("--contains", "gfx950", fat).returncode == 1
    exact = gate("--exact", "gfx90a", fat)
    assert exact.returncode == 1 and "gfx1030 gfx942" in exact.stderr


@pytest.mark.parametrize(
    ("argv", "rc", "message"),
    [
        (["--any", "gfx942", "{so}"], 2, "usage:"),
        (["--exact", "942", "{so}"], 2, "usage:"),
        (["--exact", "gfx942"], 2, "usage:"),
        (["--exact", "gfx942", "{missing}"], 1, "no such file or directory"),
        (["--exact", "gfx942", "{empty}"], 1, "no shared objects under"),
    ],
)
def test_the_gate_refuses_a_bad_mode_arch_or_path(
    tmp_path: pathlib.Path, argv: list[str], rc: int, message: str
) -> None:
    so = fake_so(tmp_path / "a.so", "amdgcn-amd-amdhsa--gfx942")
    (tmp_path / "empty").mkdir()
    paths = {"so": str(so), "missing": str(tmp_path / "missing.so"), "empty": str(tmp_path / "empty")}
    done = run(["bash", str(GATE), *(arg.format(**paths) for arg in argv)], {})
    assert done.returncode == rc
    assert message in done.stderr


def fake_rocminfo(monkeypatch: pytest.MonkeyPatch, stdout: str | None, stamp: pathlib.Path) -> list[list[str]]:
    """rocminfo answers ``stdout``, or is absent when it is None; the image stamp is ``stamp``."""
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if stdout is None:
            raise FileNotFoundError(2, "No such file or directory", "rocminfo")
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(flags.subprocess, "run", fake_run)
    monkeypatch.setattr(flags, "IMAGE_GPU_ARCH", stamp)
    monkeypatch.delenv("HPCAGENT_BENCH_GFX", raising=False)
    return calls


def test_detect_gfx_raises_instead_of_guessing_when_rocminfo_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    fake_rocminfo(monkeypatch, None, tmp_path / "gpu-arch")
    with pytest.raises(RuntimeError, match="rocminfo failed.*HPCAGENT_BENCH_GFX"):
        flags.detect_gfx()


def test_detect_gfx_raises_when_rocminfo_lists_only_a_cpu_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    fake_rocminfo(monkeypatch, ROCMINFO.split("*******\nAgent 2")[0], tmp_path / "gpu-arch")
    with pytest.raises(RuntimeError, match="no gfx GPU agent"):
        flags.detect_gfx()


def test_detect_gfx_returns_the_first_gpu_agent_with_no_stamp_or_a_matching_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    arch = table()["mi300"]
    stamp = tmp_path / "gpu-arch"
    calls = fake_rocminfo(monkeypatch, ROCMINFO.format(arch=arch), stamp)
    assert flags.detect_gfx() == arch
    stamp.write_text(f"{arch}\n", encoding="ascii")
    assert flags.detect_gfx() == arch
    assert calls == [["rocminfo"], ["rocminfo"]]


def test_detect_gfx_raises_when_the_gpu_is_not_the_arch_the_image_was_built_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    rows = table()
    stamp = tmp_path / "gpu-arch"
    stamp.write_text(f"{rows['mi200']}\n", encoding="ascii")
    fake_rocminfo(monkeypatch, ROCMINFO.format(arch=rows["mi300"]), stamp)
    with pytest.raises(RuntimeError, match=f"built for {rows['mi200']}.*GPU is {rows['mi300']}"):
        flags.detect_gfx()


def test_hpcagent_bench_gfx_overrides_the_probe_without_running_rocminfo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    calls = fake_rocminfo(monkeypatch, None, tmp_path / "gpu-arch")
    monkeypatch.setenv("HPCAGENT_BENCH_GFX", "gfx1103")
    assert flags.detect_gfx() == "gfx1103"
    assert calls == []


def test_a_failed_amd_arch_detection_is_not_cached_as_an_offload_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    arch = table()["mi300"]
    answers: Iterator[str | RuntimeError] = iter([RuntimeError("no gfx GPU agent"), arch])

    def detect() -> str:
        answer = next(answers)
        if isinstance(answer, RuntimeError):
            raise answer
        return answer

    monkeypatch.setattr(flags, "detect_gfx", detect)
    monkeypatch.setattr(languages, "offload_probe", lambda model, vendor, arch, *, run: True)
    monkeypatch.delenv(languages.OFFLOAD_ARCH_ENV.format(vendor="AMD"), raising=False)
    languages.offload_arch.cache_clear()
    with pytest.raises(RuntimeError, match="no gfx GPU agent"):
        languages.offload_arch("openmp", "amd")
    assert languages.offload_arch("openmp", "amd") == arch
    languages.offload_arch.cache_clear()
