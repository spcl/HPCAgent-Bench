# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The mi200 (MI250X, gfx90a) SGLang image, its pipeline entries and the serving gate's key file.

MI250X needs its own image: sgl_kernel and cupy carry device code for one gfx arch, the base's
common_ops is gfx942-only, and aiter has no gfx90a kernels. These pin the recipe to the ROCM_ARCH build
arg the mi200 row of gpu_arch.env sets, pin every pipeline map to one profile name, run the recipe's
setup_rocm.py edit on a stand-in file, and check that verify-tools-reasoning.py sends the key it reads.
The launcher is tests/test_serve_private.py; the table and the device gate are tests/test_gpu_arch_table.py.
"""

import importlib.util
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tomllib
import types
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CE = ROOT / "containers" / "images"
MI200 = CE / "sglang-mi200"
PROFILE = "sglang-mi200"
CANDIDATE = "hpcagent-bench-sglang-mi200-candidate.sqsh"
LIVE = "hpcagent-bench-sglang-mi200.sqsh"
KEY = "0123456789abcdef" * 4
SETUP_PATH = "/sgl-workspace/sglang/python/sglang/kernels/aot/setup_rocm.py"
#: The setup_rocm.py lines the recipe edits, verbatim from the pinned base image.
SETUP_ROCM = """\
default_target = "gfx942"
amdgpu_target = os.environ.get("AMDGPU_TARGET", default_target)

if torch.cuda.is_available():
    try:
        amdgpu_target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception as e:
        print(f"Warning: Failed to detect GPU properties: {e}")

if amdgpu_target not in ["gfx942", "gfx950", "gfx1250"]:
    sys.exit(1)

hipcc_flags = [
    f"--amdgpu-target={amdgpu_target}",
    "-DENABLE_BF16",
    "-DENABLE_FP8",
    fp8_macro,
]
"""


def load_module(path: pathlib.Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_lines(path: pathlib.Path) -> str:
    """The file without its comment lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def mi200_arch() -> str:
    """The arch gpu_arch.env names for partition mi200."""
    rows = (CE / "gpu_arch.env").read_text(encoding="ascii").splitlines()
    found = [row.removeprefix("GPU_ARCH_mi200=") for row in rows if row.startswith("GPU_ARCH_mi200=")]
    assert len(found) == 1, found
    return found[0]


def dockerfile_base(path: pathlib.Path) -> str:
    match = re.search(
        r"^ARG BASE_IMAGE=(\S+)\\\n(@sha256:[0-9a-f]{64})$", path.read_text(encoding="utf-8"), re.MULTILINE
    )
    assert match, path
    return match.group(1) + match.group(2)


def build_sh_base(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8")
    repo = re.search(r'^BASE_REPO="([^"]+)"$', text, re.MULTILINE)
    digest = re.search(r'^BASE_DIGEST="([^"]+)"$', text, re.MULTILINE)
    assert repo and digest, path
    return f"{repo.group(1)}@{digest.group(1)}"


def test_the_mi200_recipe_pins_the_same_base_digest_as_sglang() -> None:
    bases = {
        dockerfile_base(CE / "sglang" / "Dockerfile"),
        dockerfile_base(MI200 / "Dockerfile"),
        build_sh_base(CE / "sglang" / "build.sh"),
        build_sh_base(MI200 / "build.sh"),
    }
    assert len(bases) == 1, bases


def test_the_mi200_recipe_builds_every_device_artifact_for_the_rocm_arch_build_arg_only() -> None:
    code = code_lines(MI200 / "Dockerfile")
    assert re.findall(r"^ARG ROCM_ARCH\b.*$", code, re.MULTILINE) == ["ARG ROCM_ARCH"]
    assert 'AMDGPU_TARGET="${ROCM_ARCH}"' in code
    assert 'HCC_AMDGPU_TARGET="${ROCM_ARCH}"' in code
    assert re.findall(r'(?:ROCM_ARCH|AMDGPU_TARGET|GPU_ARCHS)="?gfx', code) == []


def test_the_mi200_build_runs_on_mi200_and_passes_the_table_arch_as_the_build_arg() -> None:
    sbatch = (MI200 / "build.sbatch").read_text(encoding="utf-8")
    assert re.findall(r"^#SBATCH --partition=(\S+)$", sbatch, re.MULTILINE) == ["mi200"]
    assert '"${SLURM_JOB_PARTITION:-}" != mi200' in sbatch
    build = code_lines(MI200 / "build.sh")
    assert re.search(r"^ce_gpu_arch$", build, re.MULTILINE)
    assert '--build-arg "ROCM_ARCH=${ROCM_ARCH}"' in build


def test_the_mi200_build_fails_unless_cupy_and_common_ops_carry_device_code_for_exactly_the_build_arch() -> None:
    code = code_lines(MI200 / "Dockerfile")
    gated = re.findall(r'/usr/local/bin/device_arch_gate\.sh --exact "\$\{ROCM_ARCH\}" "\$\{(\w+)\}"', code)
    assert gated == ["d", "so"], gated
    assert "COPY containers/lib/device_arch_gate.sh /usr/local/bin/device_arch_gate.sh" in code


def test_the_mi200_image_ships_no_aiter_prebuild_and_serves_with_aiter_off() -> None:
    code = code_lines(MI200 / "Dockerfile")
    assert "import aiter" not in code and "AITER_JIT_DIR" not in code
    # GPU_ARCHS appears only as the override of the base's gfx942 ENV, never as an aiter arch setting.
    assert re.findall(r"GPU_ARCHS=\S+", code) == ["GPU_ARCHS=${ROCM_ARCH}"]
    assert "SGLANG_USE_AITER=0" in code
    env = tomllib.loads((MI200 / "edf.toml.example").read_text(encoding="utf-8"))["env"]
    assert env["SGLANG_USE_AITER"] == "0"
    assert [name for name in env if name.startswith("AITER_")] == []


def setup_edit_block() -> str:
    check = load_module(ROOT / "scripts" / "check_dockerfile_python.py", "check_dockerfile_python")
    found = [source for _, source in check.blocks(MI200 / "Dockerfile") if SETUP_PATH in source]
    assert len(found) == 1, "the recipe must carry exactly one setup_rocm.py edit block"
    return found[0]


def run_setup_edit(tmp_path: pathlib.Path, setup_text: str) -> tuple[subprocess.CompletedProcess[str], str]:
    setup = tmp_path / "setup_rocm.py"
    setup.write_text(setup_text, encoding="utf-8")
    script = tmp_path / "edit.py"
    script.write_text(setup_edit_block().replace(SETUP_PATH, str(setup)), encoding="utf-8")
    env = {**os.environ, "ROCM_ARCH": mi200_arch()}
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False, env=env)
    return done, setup.read_text(encoding="utf-8")


def test_the_setup_rocm_edit_admits_the_mi200_arch_lets_amdgpu_target_win_and_drops_enable_fp8(
    tmp_path: pathlib.Path,
) -> None:
    done, edited = run_setup_edit(tmp_path, SETUP_ROCM)
    assert done.returncode == 0, done.stderr
    assert 'if "AMDGPU_TARGET" not in os.environ and torch.cuda.is_available():' in edited
    assert f'["gfx942", "gfx950", "gfx1250", "{mi200_arch()}"]' in edited
    assert "-DENABLE_FP8" not in edited
    assert "fp8_macro," in edited and "-DENABLE_BF16" in edited


def test_the_setup_rocm_edit_fails_the_build_on_a_setup_it_does_not_recognise(tmp_path: pathlib.Path) -> None:
    unknown = SETUP_ROCM.replace('"gfx950", ', "")
    done, edited = run_setup_edit(tmp_path, unknown)
    assert done.returncode != 0
    assert "gfx1250" in done.stderr and "re-derive" in done.stderr
    assert edited == unknown


def images_env() -> list[str]:
    names = (
        "INFERENCE_SGLANG_MI200_SQSH INFERENCE_SGLANG_MI200_EDF_LATEST INFERENCE_SGLANG_MI200_TEMPLATE "
        "JUDGE_AGENT_AMD_SQSH JUDGE_AMD_SQSH INFERENCE_SGLANG_SQSH INFERENCE_VLLM_SQSH"
    )
    script = f'source "$1"; for n in {names}; do printf "%s\\n" "${{!n}}"; done'
    done = subprocess.run(
        ["bash", "-c", script, "_", str(CE / "images.env")],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    return done.stdout.splitlines()


def test_images_env_names_the_mi200_live_image_edf_and_template() -> None:
    sqsh, edf, template = images_env()[:3]
    assert (sqsh, edf, template) == (LIVE, "hpcagent-bench-sglang-mi200-latest", "sglang-mi200/edf.toml.example")
    assert (CE / template).is_file()


def test_build_and_verify_maps_the_profile_to_the_candidate_its_build_writes(tmp_path: pathlib.Path) -> None:
    """VERIFY_ONLY on mi200 verifies exactly the candidate sglang-mi200/build.sbatch writes."""
    assert "${SCRATCH:?}/ce-images/${INFERENCE_SGLANG_MI200_CANDIDATE}" in (MI200 / "build.sbatch").read_text(
        encoding="utf-8"
    )
    repo, scratch = tmp_path / "repo", tmp_path / "scratch"
    ce = repo / "containers" / "images"
    (ce / PROFILE).mkdir(parents=True)
    (scratch / "ce-images").mkdir(parents=True)
    for name in ("build_common.sh", "images.env", "gpu_arch.env", "cpu_target.env"):
        shutil.copy2(CE / name, ce / name)
    (ce / "verify_image.sbatch").write_text(f'echo "$PROFILE $IMAGE" >> "{tmp_path}/verified"\n', encoding="utf-8")
    (scratch / "ce-images" / CANDIDATE).write_bytes(b"sqsh")
    env = {
        "PATH": "/usr/bin:/bin",
        "SCRATCH": str(scratch),
        "REPO": str(repo),
        "IMAGE_DIR": f"containers/images/{PROFILE}",
        "SLURM_JOB_PARTITION": "mi200",
        "SLURM_JOB_ID": "7",
        "VERIFY_ONLY": "1",
    }
    done = subprocess.run(
        ["bash", str(CE / "build_and_verify.sbatch")], capture_output=True, text=True, check=False, env=env
    )
    assert done.returncode == 0, done.stderr
    verified = (tmp_path / "verified").read_text(encoding="utf-8").splitlines()
    assert verified == [f"{PROFILE} {scratch / 'ce-images' / CANDIDATE}"]


def test_verify_image_py_checks_sgl_kernel_not_aiter_for_mi200_and_keeps_the_sglang_surface() -> None:
    verify = load_module(CE / "verify_image.py", "verify_image")
    mi200 = {check.name for check in verify.checks(PROFILE)}
    assert {"sglang", "sgl_kernel", "triton", "libfabric", "libcxi", "rocBLAS"} <= mi200
    assert {"aiter", "flydsl"}.isdisjoint(mi200)
    sglang = {check.name: check.required for check in verify.checks("sglang")}
    assert sglang["aiter"] and sglang["flydsl"] and "sgl_kernel" not in sglang
    helped = subprocess.run(
        [sys.executable, str(CE / "verify_image.py"), "--help"], capture_output=True, text=True, check=True
    )
    assert PROFILE in helped.stdout


def test_verify_image_sbatch_gives_mi200_the_venv_path_its_modules_and_a_counted_launch_check() -> None:
    text = (CE / "verify_image.sbatch").read_text(encoding="utf-8")
    assert re.search(r'^\s+sglang\|sglang-mi200\)\s+ce_path="/opt/venv/bin" ;;$', text, re.MULTILINE)
    assert re.search(r"^\s+sglang\|sglang-mi200\|vllm\)\s+ce_fi_provider=", text, re.MULTILINE)
    modules = re.search(r'^\s+sglang-mi200\)\s+sc_modules="([^"]+)" ;;$', text, re.MULTILINE)
    assert modules and "sgl_kernel" in modules.group(1) and "flydsl" not in modules.group(1)
    assert "inference/sglang_kernel_launch_check.py" in text
    assert (ROOT / "containers" / "inference" / "sglang_kernel_launch_check.py").is_file()
    assert 'if [ "${launch_rc}" -ne 0 ]; then' in text


def test_promote_image_moves_the_mi200_candidate_over_its_live_name(tmp_path: pathlib.Path) -> None:
    ce, edf_dir = tmp_path / "ce", tmp_path / "edf"
    ce.mkdir()
    edf_dir.mkdir()
    (ce / CANDIDATE).write_bytes(b"sqsh")
    (ce / f"{CANDIDATE}.digest").write_text("sha256:abc\n", encoding="utf-8")
    (ce / f"{CANDIDATE}.verified").write_text(f"verified profile={PROFILE} job=1 digest=sha256:abc\n", encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin", "SCRATCH": str(tmp_path), "CE_IMAGES": str(ce), "EDF_DIR": str(edf_dir)}
    done = subprocess.run(
        ["bash", str(CE / "promote_image.sh"), PROFILE],
        capture_output=True,
        text=True,
        check=False,
        env={**env, "DRY_RUN": "1"},
    )
    assert done.returncode == 0, done.stderr
    assert f"{CANDIDATE}\n  -> {LIVE}" in done.stdout
    assert (ce / CANDIDATE).is_file(), "a dry run moved the candidate"


def install_edfs(tmp_path: pathlib.Path, images: list[str]) -> tuple[subprocess.CompletedProcess[str], pathlib.Path]:
    ce, edf_dir = tmp_path / "ce", tmp_path / "edf"
    ce.mkdir()
    for image in images:
        (ce / image).write_bytes(b"sqsh")
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "SCRATCH": str(tmp_path),
        "CE_IMAGES": str(ce),
        "EDF_DIR": str(edf_dir),
    }
    done = subprocess.run(["bash", str(CE / "install_edfs.sh")], capture_output=True, text=True, check=False, env=env)
    return done, edf_dir


def test_install_edfs_renders_sglang_mi200_latest_onto_the_mi200_image(tmp_path: pathlib.Path) -> None:
    _, edf_dir = install_edfs(tmp_path, [LIVE])
    edf = tomllib.loads((edf_dir / "hpcagent-bench-sglang-mi200-latest.toml").read_text(encoding="utf-8"))
    assert edf["image"] == str(tmp_path / "ce" / LIVE)
    assert edf["workdir"] == str(tmp_path)
    assert edf["env"]["SGLANG_USE_AITER"] == "0"
    # Unquoted dotted TOML keys nest: com.hooks.aws_ofi_nccl.enabled is com -> hooks -> aws_ofi_nccl.
    # The mi300 sglang EDF's pinned netstack artifact, not "host": host mode's rocm6 RCCL plugin needs
    # libamdhip64.so.6, which a ROCm 7.2 image lacks, so tp8 init dies with no NET plugin (649811).
    hooks = edf["annotations"]["com"]["hooks"]
    mi300 = tomllib.loads((CE / "sglang" / "edf.toml.example").read_text(encoding="utf-8"))["annotations"]
    assert hooks == mi300["com"]["hooks"]
    assert (hooks["netstack"]["source"], hooks["cxi"]["enabled"], hooks["aws_ofi_nccl"]["enabled"]) == (
        "artifact",
        "true",
        "true",
    )


def test_a_missing_mi200_image_does_not_fail_install_edfs_for_the_other_roles(tmp_path: pathlib.Path) -> None:
    done, edf_dir = install_edfs(tmp_path, images_env()[3:])
    assert done.returncode == 0, done.stderr
    assert not (edf_dir / "hpcagent-bench-sglang-mi200-latest.toml").exists()


GATE = load_module(ROOT / "containers" / "inference" / "verify-tools-reasoning.py", "verify_tools_reasoning")
#: One response that satisfies both the tool-call and the reasoning check.
RESPONSE = {
    "choices": [
        {
            "message": {
                "content": "10",
                "reasoning_content": "12 + 3 - 5",
                "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city": "Zurich"}'}}],
            }
        }
    ]
}


def run_gate(monkeypatch: pytest.MonkeyPatch, extra_argv: list[str]) -> tuple[int, list[urllib.request.Request]]:
    seen: list[urllib.request.Request] = []

    def fake_urlopen(req: urllib.request.Request, timeout: int) -> io.BytesIO:
        seen.append(req)
        return io.BytesIO(json.dumps(RESPONSE).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    argv = ["verify-tools-reasoning.py", "--base", "http://127.0.0.1:1", "--model", "m", *extra_argv]
    monkeypatch.setattr(sys, "argv", argv)
    return GATE.main(), seen


def test_the_gate_sends_no_authorization_header_without_an_api_key_file(monkeypatch: pytest.MonkeyPatch) -> None:
    rc, seen = run_gate(monkeypatch, [])
    assert rc == 0
    assert len(seen) == 2
    assert [req.get_header("Authorization") for req in seen] == [None, None]
    assert [req.get_header("Content-type") for req in seen] == ["application/json"] * 2


def test_the_gate_sends_the_bearer_key_read_from_the_api_key_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    key_file = tmp_path / "key"
    key_file.write_text(KEY + "\n", encoding="utf-8")
    rc, seen = run_gate(monkeypatch, ["--api-key-file", str(key_file)])
    assert rc == 0
    assert [req.get_header("Authorization") for req in seen] == [f"Bearer {KEY}"] * 2


def test_the_gate_refuses_an_empty_api_key_file(tmp_path: pathlib.Path) -> None:
    key_file = tmp_path / "key"
    key_file.write_text("\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="empty"):
        GATE.read_api_key(str(key_file))
