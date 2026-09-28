# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The AMD judge + agent images are ONE portable build for every AMD partition.

Device code for every gpu_arch.env AMD target and cpu_target.env's baseline CPU target, so the
image built on an mi300 node also runs on mi200. The per-partition part is only the EDF, which
renders the partition's arch for run-time JIT builds. The GPU arch table and the runtime check are
tests/test_gpu_arch_table.py.
"""

import pathlib
import re
import shutil
import subprocess
import tomllib
from typing import Any

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CE = ROOT / "containers" / "images"
RECIPE = CE / "judge-agent-amd"
SHELL_PATH = "/usr/bin:/bin"
CANDIDATES = {"agent": "hpcagent-bench-agent-amd-candidate.sqsh", "judge": "hpcagent-bench-judge-amd-candidate.sqsh"}
LIVE = {"agent": "hpcagent-bench-agent-amd.sqsh", "judge": "hpcagent-bench-judge-amd.sqsh"}
HWLOC = "/usr/lib/x86_64-linux-gnu/libhwloc.so.15"
ARCH_VARS = ("HCC_AMDGPU_TARGET", "PYTORCH_ROCM_ARCH")
AMD_ROLES = ("JUDGE_AGENT_AMD_SQSH", "JUDGE_AMD_SQSH", "INFERENCE_SGLANG_SQSH", "INFERENCE_VLLM_SQSH")


def run(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False, env={"PATH": SHELL_PATH, **env})


def common(snippet: str, build_common: pathlib.Path = CE / "build_common.sh") -> subprocess.CompletedProcess[str]:
    """Run ``snippet`` in bash with build_common.sh sourced."""
    return run(["bash", "-c", f'source "$1"; {snippet}', "bash", str(build_common)], {})


def env_value(name: str, key: str) -> str:
    rows = (CE / name).read_text(encoding="ascii").splitlines()
    found = [row.split("=", 1)[1] for row in rows if row.startswith(f"{key}=")]
    assert len(found) == 1, (name, key, found)
    return found[0]


def test_ce_amd_targets_exports_the_table_list() -> None:
    done = common('ce_amd_targets >/dev/null && printf "%s" "${ROCM_ARCH}"')
    assert done.returncode == 0, done.stderr
    assert done.stdout == env_value("gpu_arch.env", "AMD_GPU_TARGETS")


def test_every_partition_arch_is_a_target_of_the_one_amd_image() -> None:
    targets = env_value("gpu_arch.env", "AMD_GPU_TARGETS").split(";")
    rows = (CE / "gpu_arch.env").read_text(encoding="ascii").splitlines()
    partition_archs = [row.split("=", 1)[1] for row in rows if row.startswith("GPU_ARCH_")]
    assert partition_archs
    assert set(partition_archs) <= set(targets), (partition_archs, targets)


def test_ce_amd_targets_refuses_a_malformed_list(tmp_path: pathlib.Path) -> None:
    fake = tmp_path / "images"
    fake.mkdir()
    shutil.copy2(CE / "build_common.sh", fake / "build_common.sh")
    (fake / "gpu_arch.env").write_text("AMD_GPU_TARGETS=gfx942,gfx90a\n", encoding="ascii")
    done = common("ce_amd_targets", fake / "build_common.sh")
    assert done.returncode == 2
    assert "not a ;-separated list" in done.stderr


def test_ce_spack_target_defaults_to_this_cpu_family() -> None:
    family = subprocess.run(["uname", "-m"], capture_output=True, text=True, check=True).stdout.strip()
    done = common('ce_spack_target >/dev/null && printf "%s" "${SPACK_TARGET}"')
    assert done.returncode == 0, done.stderr
    assert done.stdout == env_value("cpu_target.env", f"SPACK_TARGET_{family}")


def test_ce_spack_target_refuses_a_family_the_table_does_not_name(tmp_path: pathlib.Path) -> None:
    fake = tmp_path / "images"
    fake.mkdir()
    shutil.copy2(CE / "build_common.sh", fake / "build_common.sh")
    (fake / "cpu_target.env").write_text("SPACK_TARGET_riscv64=rv64gc\n", encoding="ascii")
    done = common("ce_spack_target", fake / "build_common.sh")
    assert done.returncode == 2
    assert "names no spack target for CPU family" in done.stderr


@pytest.mark.parametrize("target", ["agent", "judge"])
def test_each_build_target_has_one_candidate_name(target: str) -> None:
    done = common(f'source "{CE}/images.env"; ce_amd_candidate {target}')
    assert done.returncode == 0, done.stderr
    assert done.stdout == CANDIDATES[target]


def test_ce_amd_candidate_refuses_an_unknown_target() -> None:
    done = common(f'source "{CE}/images.env"; ce_amd_candidate sglang')
    assert done.returncode == 2
    assert "unknown build target 'sglang'" in done.stderr


def test_no_script_spells_a_judge_agent_amd_candidate_name_outside_images_env() -> None:
    """images.env is the one place the names are spelled; a script spelling one by hand drifts."""
    literal = re.compile(r"hpcagent-bench-(agent|judge)-amd-candidate")
    for path in sorted(CE.rglob("*")):
        if path.is_file() and path.suffix in {".sh", ".sbatch", ".env", ""} and path.name != "images.env":
            assert not literal.search(path.read_text(encoding="utf-8", errors="replace")), path
    for rel in ("judge-agent-amd/build.sh", "judge-agent-amd/build.sbatch"):
        assert "ce_amd_candidate" in (CE / rel).read_text(encoding="utf-8"), rel
    assert len(literal.findall((CE / "images.env").read_text(encoding="utf-8"))) == 2


def spack_target_block(spack_target: str, tmp_path: pathlib.Path) -> str:
    """The Dockerfile's site packages.yaml SPACK_TARGET step, run on an empty file; returns the file."""
    text = (RECIPE / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(
        r'^    if \[ -n "\$\{SPACK_TARGET\}" \]; then \\\n +printf .*?^    fi; \\$', text, re.MULTILINE | re.DOTALL
    )
    assert match, "the Dockerfile lost its SPACK_TARGET packages.yaml step"
    packages = tmp_path / "etc" / "spack" / "packages.yaml"
    packages.parent.mkdir(parents=True)
    packages.write_text("packages:\n", encoding="utf-8")
    step = match.group(0).removesuffix("; \\")
    done = run(["sh", "-euc", step], {"SPACK_ROOT": str(tmp_path), "SPACK_TARGET": spack_target})
    assert done.returncode == 0, done.stderr
    return packages.read_text(encoding="utf-8")


def test_a_from_scratch_build_keeps_spack_host_detection(tmp_path: pathlib.Path) -> None:
    """No SPACK_TARGET (a plain podman build): the build machine's own ISA, i.e. -march=native."""
    assert spack_target_block("", tmp_path) == "packages:\n"


def test_the_published_baseline_is_required_for_every_package(tmp_path: pathlib.Path) -> None:
    assert spack_target_block("x86_64_v3", tmp_path) == 'packages:\n  all:\n    require: ["target=x86_64_v3"]\n'


def test_the_build_passes_both_targets_and_the_dockerfile_defaults_to_native() -> None:
    build = (RECIPE / "build.sh").read_text(encoding="utf-8")
    assert re.search(r"^ce_amd_targets$", build, re.MULTILINE)
    assert '--build-arg "SPACK_TARGET=${SPACK_TARGET}"' in build
    docker = (RECIPE / "Dockerfile").read_text(encoding="utf-8")
    assert re.findall(r"^ARG SPACK_TARGET\b.*$", docker, re.MULTILINE) == ["ARG SPACK_TARGET="]
    assert 'grep -vx -e bin -e "linux-${SPACK_TARGET}"' in docker, "the stray-target gate is gone"
    assert "amdgpu_target=${ROCM_ARCH}" not in docker, "spack takes the list ,-separated"
    assert "openblas threads=openmp +fortran +dynamic_dispatch" in docker, "BLAS lost its run-time ISA dispatch"


def test_the_pip_wheel_cache_is_keyed_by_the_target_list() -> None:
    """pip keys a built cupy wheel by its sdist, not by HCC_AMDGPU_TARGET."""
    assert 'ce_cache_args spack-buildcache "pip-cache/${ROCM_ARCH//;/-}"' in (RECIPE / "build.sh").read_text(
        encoding="utf-8"
    )


def images_env(*names: str) -> list[str]:
    script = 'source "$1"; shift; for n in "$@"; do echo "${!n}"; done'
    done = run(["bash", "-c", script, "bash", str(CE / "images.env"), *names], {})
    assert done.returncode == 0, done.stderr
    return done.stdout.splitlines()


def install_edfs(tmp_path: pathlib.Path, images: list[str]) -> tuple[subprocess.CompletedProcess[str], pathlib.Path]:
    ce, edf_dir = tmp_path / "ce", tmp_path / "edf"
    ce.mkdir()
    for image in images:
        (ce / image).write_bytes(b"sqsh")
    env = {"HOME": str(tmp_path), "SCRATCH": str(tmp_path), "CE_IMAGES": str(ce), "EDF_DIR": str(edf_dir)}
    return run(["bash", str(CE / "install_edfs.sh")], env), edf_dir


def edf(edf_dir: pathlib.Path, name: str) -> dict[str, Any]:
    return tomllib.loads((edf_dir / f"{name}.toml").read_text(encoding="utf-8"))


def test_images_env_points_every_amd_partition_edf_at_the_one_image() -> None:
    assert images_env("JUDGE_AGENT_AMD_SQSH", "JUDGE_AGENT_AMD_MI200_SQSH") == [LIVE["agent"]] * 2
    assert images_env("JUDGE_AMD_SQSH", "JUDGE_AMD_MI200_SQSH") == [LIVE["judge"]] * 2
    assert images_env("INFERENCE_VLLM_SQSH", "INFERENCE_VLLM_MI200_SQSH") == ["hpcagent-bench-vllm-amd.sqsh"] * 2


def test_install_edfs_renders_each_partition_edf_with_its_own_arch(tmp_path: pathlib.Path) -> None:
    done, edf_dir = install_edfs(tmp_path, images_env(*AMD_ROLES))
    assert done.returncode == 0, done.stderr
    for partition in ("mi200", "mi300"):
        arch = env_value("gpu_arch.env", f"GPU_ARCH_{partition}")
        for name, image in (
            (f"hpcagent-bench-agent-{partition}-latest", LIVE["agent"]),
            (f"hpcagent-bench-judge-{partition}-latest", LIVE["judge"]),
            (f"hpcagent-bench-judge-{partition}-mlscale", LIVE["judge"]),
        ):
            rendered = edf(edf_dir, name)
            assert rendered["image"] == str(tmp_path / "ce" / image), name
            assert {var: rendered["env"][var] for var in ARCH_VARS} == dict.fromkeys(ARCH_VARS, arch), name


def test_the_mlscale_edf_is_the_judge_edf_plus_the_hwloc_preload(tmp_path: pathlib.Path) -> None:
    done, edf_dir = install_edfs(tmp_path, images_env(*AMD_ROLES))
    assert done.returncode == 0, done.stderr
    judge = (edf_dir / "hpcagent-bench-judge-mi200-latest.toml").read_text(encoding="utf-8").splitlines()
    mlscale = (edf_dir / "hpcagent-bench-judge-mi200-mlscale.toml").read_text(encoding="utf-8").splitlines()
    differ = [(a, b) for a, b in zip(judge, mlscale, strict=True) if a != b]
    preload = str(edf(edf_dir, "hpcagent-bench-judge-mi200-latest")["env"]["LD_PRELOAD"])
    assert differ == [(f'LD_PRELOAD = "{preload}"', f'LD_PRELOAD = "{preload}:{HWLOC}"')], differ


@pytest.mark.parametrize(("role", "target"), [("judge-agent-amd", "agent"), ("judge", "judge")])
def test_promote_image_moves_each_candidate_over_its_live_name(tmp_path: pathlib.Path, role: str, target: str) -> None:
    ce, edf_dir = tmp_path / "ce", tmp_path / "edf"
    ce.mkdir()
    edf_dir.mkdir()
    candidate = CANDIDATES[target]
    (ce / candidate).write_bytes(b"sqsh")
    (ce / f"{candidate}.digest").write_text("sha256:abc\n", encoding="utf-8")
    (ce / f"{candidate}.verified").write_text(f"verified profile={target} job=1 digest=sha256:abc\n", encoding="utf-8")
    env = {"SCRATCH": str(tmp_path), "CE_IMAGES": str(ce), "EDF_DIR": str(edf_dir), "DRY_RUN": "1"}
    done = run(["bash", str(CE / "promote_image.sh"), role], env)
    assert done.returncode == 0, done.stderr
    assert f"{candidate}\n  -> {LIVE[target]}" in done.stdout


def test_verify_only_reverifies_the_candidates_without_building(tmp_path: pathlib.Path) -> None:
    """A verifier fix must not cost a rebuild: VERIFY_ONLY=1 re-runs stage 2 on what is there."""
    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    ce = repo / "containers" / "images"
    (ce / "judge-agent-amd").mkdir(parents=True)
    (scratch / "ce-images").mkdir(parents=True)
    for name in ("build_common.sh", "images.env", "gpu_arch.env", "cpu_target.env"):
        shutil.copy2(CE / name, ce / name)
    (ce / "judge-agent-amd" / "build.sbatch").write_text(f'touch "{tmp_path}/built"\n', encoding="utf-8")
    (ce / "verify_image.sbatch").write_text(f'echo "$PROFILE $IMAGE" >> "{tmp_path}/verified"\n', encoding="utf-8")
    for name in CANDIDATES.values():
        (scratch / "ce-images" / name).write_bytes(b"sqsh")
        (scratch / "ce-images" / f"{name}.digest").write_text("sha256:abc", encoding="utf-8")
    env = {
        "SCRATCH": str(scratch),
        "REPO": str(repo),
        "IMAGE_DIR": "containers/images/judge-agent-amd",
        "SLURM_JOB_PARTITION": "mi300",
        "SLURM_JOB_ID": "7",
        "VERIFY_ONLY": "1",
    }
    done = run(["bash", str(CE / "build_and_verify.sbatch")], env)
    assert done.returncode == 0, done.stderr
    assert not (tmp_path / "built").exists()
    images = {target: scratch / "ce-images" / name for target, name in CANDIDATES.items()}
    assert (tmp_path / "verified").read_text(encoding="utf-8").splitlines() == [
        f"judge-agent-amd {images['agent']}",
        f"judge {images['judge']}",
    ]
    for target, profile in (("agent", "judge-agent-amd"), ("judge", "judge")):
        marker = pathlib.Path(f"{images[target]}.verified").read_text(encoding="utf-8")
        assert marker == f"verified profile={profile} job=7 digest=sha256:abc\n", target
