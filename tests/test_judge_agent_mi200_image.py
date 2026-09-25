# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The mi200 (MI250X gfx90a, EPYC 7A53 Zen 3) judge + agent pair built from the judge-agent-amd recipe.

The mi300 images cannot start on mi200: their spack stack is zen4 and the preloaded mimalloc dies on
SIGILL. These pin the partition parameter of the build (candidate names, spack target, pip cache),
the mi300 inputs staying what they were, and the mi200 EDF renders and promotion. The GPU arch table and the runtime check are tests/test_gpu_arch_table.py.
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
#: The candidate names every mi300 build wrote before the partition became a parameter.
MI300_CANDIDATES = {
    "agent": "hpcagent-bench-ce-amd-mi300-candidate.sqsh",
    "judge": "hpcagent-bench-ce-judge-amd-mi300-candidate.sqsh",
}
MI200_CANDIDATES = {
    "agent": "hpcagent-bench-ce-amd-mi200-candidate.sqsh",
    "judge": "hpcagent-bench-ce-judge-amd-mi200-candidate.sqsh",
}
MI200_LIVE = {"agent": "hpcagent-bench-agent-mi200.sqsh", "judge": "hpcagent-bench-judge-mi200.sqsh"}
HWLOC = "/usr/lib/x86_64-linux-gnu/libhwloc.so.15"
ARCH_VARS = ("HCC_AMDGPU_TARGET", "PYTORCH_ROCM_ARCH")


def run(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False, env={"PATH": SHELL_PATH, **env})


def common(
    snippet: str, env: dict[str, str], build_common: pathlib.Path = CE / "build_common.sh"
) -> subprocess.CompletedProcess[str]:
    """Run ``snippet`` in bash with build_common.sh sourced."""
    return run(["bash", "-c", f'source "$1"; {snippet}', "bash", str(build_common)], env)


def partition_arch(partition: str) -> str:
    rows = (CE / "gpu_arch.env").read_text(encoding="ascii").splitlines()
    found = [row.split("=", 1)[1] for row in rows if row.startswith(f"GPU_ARCH_{partition}=")]
    assert len(found) == 1, found
    return found[0]


@pytest.mark.parametrize(
    ("partition", "target"),
    [("mi300", ""), ("mi200", "zen3")],
)
def test_ce_spack_target_pins_zen3_on_mi200_and_leaves_mi300_on_host_detection(partition: str, target: str) -> None:
    """An mi300 build passes no target, so its spack config is byte-identical to before the table."""
    done = common(
        'ce_gpu_arch >/dev/null && ce_spack_target && printf "TARGET=[%s]\\n" "${SPACK_TARGET}"',
        {"SLURM_JOB_PARTITION": partition},
    )
    assert done.returncode == 0, done.stderr
    assert f"TARGET=[{target}]" in done.stdout.splitlines(), done.stdout


def test_every_cpu_target_row_names_a_partition_the_gpu_table_knows() -> None:
    rows = [
        line
        for line in (CE / "cpu_target.env").read_text(encoding="ascii").splitlines()
        if line and not line.startswith("#")
    ]
    assert rows == ["SPACK_TARGET_mi200=zen3"], rows
    assert partition_arch("mi200")


def test_ce_spack_target_refuses_a_value_that_is_not_a_spack_target(tmp_path: pathlib.Path) -> None:
    for name in ("build_common.sh", "gpu_arch.env"):
        shutil.copy2(CE / name, tmp_path / name)
    (tmp_path / "cpu_target.env").write_text("SPACK_TARGET_mi200=zen3 target=zen4\n", encoding="ascii")
    done = common(
        "ce_gpu_arch >/dev/null && ce_spack_target", {"SLURM_JOB_PARTITION": "mi200"}, tmp_path / "build_common.sh"
    )
    assert done.returncode == 2
    assert "is not a spack target for partition mi200" in done.stderr


@pytest.mark.parametrize(
    ("partition", "names"),
    [("mi300", MI300_CANDIDATES), ("mi200", MI200_CANDIDATES)],
)
def test_the_candidate_names_are_per_partition_and_mi300_keeps_its_old_names(
    partition: str, names: dict[str, str]
) -> None:
    for target, name in names.items():
        done = common(f"ce_amd_candidate {target} {partition}", {})
        assert (done.returncode, done.stdout) == (0, name), done.stderr


def test_ce_amd_candidate_refuses_an_unknown_target() -> None:
    done = common("ce_amd_candidate sglang mi300", {})
    assert done.returncode == 2 and "unknown build target 'sglang'" in done.stderr


def test_no_script_spells_a_judge_agent_amd_candidate_name_outside_images_env() -> None:
    """A script that spells a candidate by hand writes an mi200 build over the mi300 candidate or
    verifies the wrong file: images.env is the one place the names are spelled."""
    literal = re.compile(r"hpcagent-bench-ce-(judge-)?amd-mi[0-9]+-candidate")
    for path in sorted(CE.rglob("*")):
        if path.is_file() and path.suffix in {".sh", ".sbatch", ".env", ""} and path.name != "images.env":
            assert not literal.search(path.read_text(encoding="utf-8", errors="replace")), path
    for rel in ("judge-agent-amd/build.sh", "judge-agent-amd/build.sbatch"):
        assert "ce_amd_candidate" in (CE / rel).read_text(encoding="utf-8"), rel
    assert len(literal.findall((CE / "images.env").read_text(encoding="utf-8"))) == 4


def spack_target_block(spack_target: str, tmp_path: pathlib.Path) -> str:
    """The Dockerfile's site packages.yaml SPACK_TARGET step, run on an empty file; returns the file."""
    text = (RECIPE / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(
        r'^    if \[ -n "\$\{SPACK_TARGET:-\}" \]; then \\\n +printf .*?^    fi; \\$', text, re.MULTILINE | re.DOTALL
    )
    assert match, "the Dockerfile lost its SPACK_TARGET packages.yaml step"
    step = match.group(0).removesuffix("; \\")
    packages = tmp_path / "etc" / "spack" / "packages.yaml"
    packages.parent.mkdir(parents=True)
    packages.write_text("packages:\n", encoding="utf-8")
    done = run(["sh", "-euc", step], {"SPACK_ROOT": str(tmp_path), "SPACK_TARGET": spack_target})
    assert done.returncode == 0, done.stderr
    return packages.read_text(encoding="utf-8")


def test_an_unset_spack_target_leaves_the_site_packages_yaml_untouched(tmp_path: pathlib.Path) -> None:
    assert spack_target_block("", tmp_path) == "packages:\n"


def test_a_spack_target_requires_it_for_every_package(tmp_path: pathlib.Path) -> None:
    assert spack_target_block("zen3", tmp_path) == 'packages:\n  all:\n    require: ["target=zen3"]\n'


def test_the_spack_target_reaches_the_build_only_as_an_optional_build_arg() -> None:
    build = (RECIPE / "build.sh").read_text(encoding="utf-8")
    assert re.search(r"^ce_spack_target$", build, re.MULTILINE)
    assert '[[ -z "${SPACK_TARGET}" ]] || SPACK_TARGET_ARGS=(--build-arg "SPACK_TARGET=${SPACK_TARGET}")' in build
    assert '      "${SPACK_TARGET_ARGS[@]}" \\\n' in build
    docker = (RECIPE / "Dockerfile").read_text(encoding="utf-8")
    assert re.findall(r"^ARG SPACK_TARGET\b.*$", docker, re.MULTILINE) == ["ARG SPACK_TARGET="]
    assert 'grep -vx -e bin -e "linux-${SPACK_TARGET}"' in docker, "the stray-target gate is gone"


def test_the_pip_wheel_cache_is_per_gpu_arch() -> None:
    """pip keys a built cupy wheel by its sdist, not by HCC_AMDGPU_TARGET."""
    assert 'PIP_CACHE="${PIP_CACHE:-${SCRATCH:?}/pip-cache/${ROCM_ARCH}}"' in (RECIPE / "build.sh").read_text(
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


MI300_ROLES = ("JUDGE_AGENT_AMD_SQSH", "JUDGE_AMD_SQSH", "INFERENCE_SGLANG_SQSH", "INFERENCE_VLLM_SQSH")


def edf(edf_dir: pathlib.Path, name: str) -> dict[str, Any]:
    return tomllib.loads((edf_dir / f"{name}.toml").read_text(encoding="utf-8"))


def test_images_env_names_the_mi200_pair_apart_from_every_mi300_name() -> None:
    names = images_env(
        "JUDGE_AGENT_AMD_MI200_SQSH",
        "JUDGE_AGENT_AMD_MI200_EDF_LATEST",
        "JUDGE_AMD_MI200_SQSH",
        "JUDGE_AMD_MI200_EDF_LATEST",
        "JUDGE_AMD_MI200_MLSCALE_EDF_LATEST",
    )
    assert names == [
        MI200_LIVE["agent"],
        "hpcagent-bench-agent-mi200-latest",
        MI200_LIVE["judge"],
        "hpcagent-bench-judge-mi200-latest",
        "hpcagent-bench-judge-mi200-mlscale",
    ]
    assert images_env("JUDGE_AGENT_AMD_SQSH", "JUDGE_AMD_SQSH") == [
        "hpcagent-bench-agent-mi300.sqsh",
        "hpcagent-bench-judge-mi300.sqsh",
    ]


def test_install_edfs_renders_the_mi200_pair_with_the_mi200_arch(tmp_path: pathlib.Path) -> None:
    done, edf_dir = install_edfs(tmp_path, [*images_env(*MI300_ROLES), *MI200_LIVE.values()])
    assert done.returncode == 0, done.stderr
    arch = partition_arch("mi200")
    for name, image in (
        ("hpcagent-bench-agent-mi200-latest", MI200_LIVE["agent"]),
        ("hpcagent-bench-judge-mi200-latest", MI200_LIVE["judge"]),
        ("hpcagent-bench-judge-mi200-mlscale", MI200_LIVE["judge"]),
    ):
        rendered = edf(edf_dir, name)
        assert rendered["image"] == str(tmp_path / "ce" / image), name
        assert {var: rendered["env"][var] for var in ARCH_VARS} == dict.fromkeys(ARCH_VARS, arch), name
    mi300 = edf(edf_dir, "hpcagent-bench-judge-mi300-latest")
    assert {var: mi300["env"][var] for var in ARCH_VARS} == dict.fromkeys(ARCH_VARS, partition_arch("mi300"))


def test_the_mlscale_edf_is_the_judge_edf_plus_the_hwloc_preload(tmp_path: pathlib.Path) -> None:
    done, edf_dir = install_edfs(tmp_path, [*images_env(*MI300_ROLES), MI200_LIVE["judge"]])
    assert done.returncode == 0, done.stderr
    judge = (edf_dir / "hpcagent-bench-judge-mi200-latest.toml").read_text(encoding="utf-8").splitlines()
    mlscale = (edf_dir / "hpcagent-bench-judge-mi200-mlscale.toml").read_text(encoding="utf-8").splitlines()
    differ = [(a, b) for a, b in zip(judge, mlscale, strict=True) if a != b]
    preload = str(edf(edf_dir, "hpcagent-bench-judge-mi200-latest")["env"]["LD_PRELOAD"])
    assert differ == [(f'LD_PRELOAD = "{preload}"', f'LD_PRELOAD = "{preload}:{HWLOC}"')], differ


def test_without_mi200_images_install_edfs_writes_exactly_the_mi300_set(tmp_path: pathlib.Path) -> None:
    done, edf_dir = install_edfs(tmp_path, images_env(*MI300_ROLES))
    assert done.returncode == 0, done.stderr
    assert sorted(path.stem for path in edf_dir.glob("*.toml")) == [
        "hpcagent-bench-agent-mi300-latest",
        "hpcagent-bench-judge-mi300-latest",
        "hpcagent-bench-sglang-mi300-latest",
        "hpcagent-bench-vllm-mi300-latest",
    ]


@pytest.mark.parametrize(("role", "target"), [("judge-agent-amd-mi200", "agent"), ("judge-mi200", "judge")])
def test_promote_image_moves_each_mi200_candidate_over_its_own_live_name(
    tmp_path: pathlib.Path, role: str, target: str
) -> None:
    ce, edf_dir = tmp_path / "ce", tmp_path / "edf"
    ce.mkdir()
    edf_dir.mkdir()
    candidate = MI200_CANDIDATES[target]
    (ce / candidate).write_bytes(b"sqsh")
    (ce / f"{candidate}.digest").write_text("sha256:abc\n", encoding="utf-8")
    (ce / f"{candidate}.verified").write_text(f"verified profile={target} job=1 digest=sha256:abc\n", encoding="utf-8")
    env = {"SCRATCH": str(tmp_path), "CE_IMAGES": str(ce), "EDF_DIR": str(edf_dir), "DRY_RUN": "1"}
    done = run(["bash", str(CE / "promote_image.sh"), role], env)
    assert done.returncode == 0, done.stderr
    assert f"{candidate}\n  -> {MI200_LIVE[target]}" in done.stdout


def test_promote_image_keeps_the_mi300_candidates_and_live_names(tmp_path: pathlib.Path) -> None:
    script = 'source "$1"; for r in judge-agent-amd judge; do echo "$(role_candidate $r) $(role_live $r)"; done'
    text = (CE / "promote_image.sh").read_text(encoding="utf-8")
    functions = re.search(r"^role_candidate\(\) \{.*?^\}\nrole_live\(\) \{.*?^\}\n", text, re.MULTILINE | re.DOTALL)
    assert functions, "promote_image.sh lost role_candidate/role_live"
    lib = tmp_path / "roles.sh"
    lib.write_text(f'source "{CE}/images.env"\nsource "{CE}/build_common.sh"\n{functions.group(0)}', encoding="utf-8")
    done = run(["bash", "-c", script, "bash", str(lib)], {})
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        f"{MI300_CANDIDATES['agent']} hpcagent-bench-agent-mi300.sqsh",
        f"{MI300_CANDIDATES['judge']} hpcagent-bench-judge-mi300.sqsh",
    ]


def test_verify_only_reverifies_the_partitions_candidates_without_building(tmp_path: pathlib.Path) -> None:
    """A verifier fix must not cost a 4 h rebuild: VERIFY_ONLY=1 re-runs stage 2 on what is there."""
    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    ce = repo / "containers" / "images"
    (ce / "judge-agent-amd").mkdir(parents=True)
    (scratch / "ce-images").mkdir(parents=True)
    for name in ("build_common.sh", "images.env", "gpu_arch.env", "cpu_target.env"):
        shutil.copy2(CE / name, ce / name)
    (ce / "judge-agent-amd" / "build.sbatch").write_text(f'touch "{tmp_path}/built"\n', encoding="utf-8")
    (ce / "verify_image.sbatch").write_text(f'echo "$PROFILE $IMAGE" >> "{tmp_path}/verified"\n', encoding="utf-8")
    for name in MI200_CANDIDATES.values():
        (scratch / "ce-images" / name).write_bytes(b"sqsh")
        (scratch / "ce-images" / f"{name}.digest").write_text("sha256:abc", encoding="utf-8")
    env = {
        "SCRATCH": str(scratch),
        "REPO": str(repo),
        "IMAGE_DIR": "containers/images/judge-agent-amd",
        "SLURM_JOB_PARTITION": "mi200",
        "SLURM_JOB_ID": "7",
        "VERIFY_ONLY": "1",
    }
    done = run(["bash", str(CE / "build_and_verify.sbatch")], env)
    assert done.returncode == 0, done.stderr
    assert not (tmp_path / "built").exists()
    images = {target: scratch / "ce-images" / name for target, name in MI200_CANDIDATES.items()}
    assert (tmp_path / "verified").read_text(encoding="utf-8").splitlines() == [
        f"judge-agent-amd {images['agent']}",
        f"judge {images['judge']}",
    ]
    for target, profile in (("agent", "judge-agent-amd"), ("judge", "judge")):
        marker = pathlib.Path(f"{images[target]}.verified").read_text(encoding="utf-8")
        assert marker == f"verified profile={profile} job=7 digest=sha256:abc\n", target
