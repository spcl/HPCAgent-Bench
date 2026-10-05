# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The image recipes and the tooling around them install Python only through uv sync from uv.lock.

uv.lock decides every version. An image installs only what it builds from source (the image-python groups); every
other locked package is installed when a job starts (containers/lib/launch_venv.sh). Static: the checks read files.
"""

import pathlib
import re
import tomllib

import pytest

ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
IMAGES: pathlib.Path = ROOT / "containers" / "images"
LIB: pathlib.Path = ROOT / "containers" / "lib"
JUDGE_AGENT: tuple[str, ...] = ("judge-agent-amd", "judge-agent-cpu", "judge-agent-cuda")
EXTRA_OF: dict[str, str] = {"judge-agent-amd": "amdgpu", "judge-agent-cpu": "cpu", "judge-agent-cuda": "nvgpu"}
HARNESS_GROUPS: tuple[str, ...] = ("harness-miniswe", "harness-openhands")
PIP_INTERFACE: re.Pattern[str] = re.compile(
    r"uv pip|pip install|image-pins|UV_CONSTRAINT|PIP_[A-Z]|--break-system-packages|--no-sources|--excludes"
)
#: Every file that installs Python for an image, a job or a release smoke.
INSTALLERS: tuple[pathlib.Path, ...] = (
    *IMAGES.glob("*/Dockerfile"),
    *LIB.glob("*.sh"),
    ROOT / "scripts" / "do_release.sh",
    ROOT / ".github" / "actions" / "setup" / "action.yml",
    ROOT / "tests" / "test_container_launch.py",
    ROOT / "tests" / "test_packaging.py",
)


def recipe(image: str) -> str:
    return (IMAGES / image / "Dockerfile").read_text(encoding="utf-8")


def code_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


def pyproject(path: pathlib.Path = ROOT / "pyproject.toml") -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_no_installer_uses_the_pip_interface() -> None:
    hits = {
        str(path.relative_to(ROOT)): [
            line.strip() for line in code_lines(path.read_text(encoding="utf-8")) if PIP_INTERFACE.search(line)
        ]
        for path in INSTALLERS
    }
    assert {path: lines for path, lines in hits.items() if lines} == {}


#: What each image builds from source and so never takes from the launch venv.
IMAGE_BUILT: dict[str, tuple[str, ...]] = {
    "judge-agent-amd": ("numpy", "scipy", "mpi4py", "cupy"),
    "judge-agent-cpu": ("numpy", "scipy", "mpi4py"),
    "judge-agent-cuda": ("numpy", "scipy", "mpi4py"),
}


def launch_args(text: str) -> str:
    """The agent stage's /opt/launch/sync.args, as the Dockerfile writes it."""
    found = re.search(r'echo "([^"]*)" > /opt/launch/sync\.args', text, re.DOTALL)
    assert found is not None
    return " ".join(found.group(1).replace("\\\n", " ").split())


@pytest.mark.parametrize("image", JUDGE_AGENT)
def test_a_judge_agent_image_installs_only_what_it_builds_from_source(image: str) -> None:
    """Every other locked package is installed when a job starts; an image sync never selects a framework extra."""
    text = recipe(image)
    assert "COPY pyproject.toml uv.lock /opt/hpcagent-bench/" in text
    assert "COPY agent/pyproject.toml /opt/hpcagent-bench-agent/pyproject.toml" in text
    sync = re.search(
        r"uv sync --frozen --(?:exact|inexact)[^;]*--no-default-groups --group image-python "
        r"--no-binary-package mpi4py;",
        text,
        re.DOTALL,
    )
    assert sync is not None, image
    image_syncs = [one for one in re.findall(r"uv sync [^;]*?(?=; \\)", text, re.DOTALL) if "/opt/rocprof" not in one]
    assert [one for one in image_syncs if "--extra" in one] == [], image


@pytest.mark.parametrize("image", JUDGE_AGENT)
def test_the_launch_venv_installs_the_extra_and_the_proxy_and_never_an_image_build(image: str) -> None:
    text = recipe(image)
    args = launch_args(text)
    assert args.startswith(f"--no-install-project --extra {EXTRA_OF[image]} --group judge-proxy"), args
    assert {f"--no-install-package {name}" for name in IMAGE_BUILT[image]} <= {
        f"--no-install-package {word}" for word in args.split("--no-install-package ")[1:] for word in [word.strip()]
    }, args
    assert 'ENTRYPOINT ["/opt/launch/launch_venv.sh"]' in text
    assert "COPY containers/lib/launch_venv.sh containers/lib/one_openmp.sh /opt/launch/" in text
    judge = text[text.index("FROM agent AS judge") :]
    assert "sed -i 's/^--no-install-project //' /opt/launch/sync.args" in judge, "a judge job installs hpcagent_bench"
    assert "package_hook.sh" not in judge
    for name in ("DACE_COMMIT", "direct_url.json", "/opt/dace", "constraint.txt", "image-pins"):
        assert name not in "\n".join(code_lines(text)), name


@pytest.mark.parametrize("image", JUDGE_AGENT)
def test_every_wheel_gate_runs_in_a_launch_venv_the_image_does_not_keep(image: str) -> None:
    text = recipe(image)
    gate = re.search(
        r"RUN HPCAGENT_BENCH_LAUNCH_ROOT=/opt/launch-gate [^\n]*/opt/launch/launch_venv\.sh sh -eux -c '(.*?)' \\\n"
        r"    && rm -rf /opt/launch-gate",
        text,
        re.DOTALL,
    )
    assert gate is not None, image
    for check in (
        "import torch",
        "playwright install",
        "one_openmp.sh /opt/view",
        "omp_contexts.sh",
        "openmp_gate.py context --context gnu --wheels --torch",
        "HAVE_ISL",
    ):
        assert check in gate.group(1), check
    after = text[gate.end() :]
    assert 'python3 -c "import torch' not in after and "import dace" not in after


@pytest.mark.parametrize("image", JUDGE_AGENT)
def test_the_package_hook_serves_only_the_harness_venvs(image: str) -> None:
    text = recipe(image)
    assert "ln -s /opt/hpcagent-bench-agent /opt/hpcagent-bench/agent" in text
    hooks = re.findall(r"sh /tmp/package_hook\.sh [^;]*", text)
    assert hooks and all('"/opt/harness/${venv}"' in hook and '--group "harness-${venv}"' in hook for hook in hooks)


@pytest.mark.parametrize("image", JUDGE_AGENT)
def test_the_openblas_rebuild_syncs_the_image_group_only(image: str) -> None:
    """A sync that selects an extra would put the framework wheels into the image (they install at launch)."""
    rebuild = re.search(r"numpy_on_openblas\.sh /opt/view /opt/hpcagent-bench ([^&]*)&&", recipe(image))
    assert rebuild is not None, image
    assert rebuild.group(1).split() == ["--no-default-groups", "--group", "image-python"]


def test_the_amd_image_builds_cupy_from_its_group_after_the_numpy_rebuild() -> None:
    text = recipe("judge-agent-amd")
    assert text.index("numpy_on_openblas.sh") < text.index("--group image-cupy-rocm --no-binary-package cupy")
    cupy = text[text.index("CUPY_INSTALL_USE_HIP=1") : text.index("--no-binary-package cupy")]
    assert "--no-install-package numpy --no-install-package scipy" in cupy


def test_the_launch_hook_builds_one_locked_venv_per_pin_beside_the_image_python() -> None:
    hook = (LIB / "launch_venv.sh").read_text(encoding="utf-8")
    assert 'cat "${workspace}/uv.lock" "${launch}/sync.args" "${launch}/image.id" | sha256sum' in hook
    assert "flock 9" in hook and 'touch "${home}/ready"' in hook
    assert "zz-image-site.pth" in hook, "the image's source builds stay visible after the venv's own packages"
    assert "uv sync -q --frozen --inexact --no-install-project" not in hook, "the judge installs the project"
    assert 'one_openmp.sh" --link-only' in hook
    assert 'HPCAGENT_BENCH_IMAGE_PYTHON="${venv}/bin/python3"' in hook and 'exec "$@"' in hook


def test_the_rocprof_compute_environment_is_a_locked_project_synced_into_its_venv() -> None:
    project = IMAGES / "judge-agent-amd" / "rocprof-compute"
    deps = pyproject(project / "pyproject.toml")["project"]["dependencies"]  # type: ignore[index]
    assert "pandas==2.2.3" in deps and "astunparse==1.6.2" in deps
    assert (project / "uv.lock").is_file()
    text = recipe("judge-agent-amd")
    assert "/opt/rocprof-compute/" in text and "UV_PROJECT_ENVIRONMENT=/opt/rocprof-compute-venv" in text


def test_the_extras_are_three_exclusive_framework_sets_that_each_carry_dev() -> None:
    project = pyproject()
    extras = project["project"]["optional-dependencies"]  # type: ignore[index]
    assert {"cpu", "amdgpu", "nvgpu"} <= set(extras)
    assert {"amd", "nvidia"}.isdisjoint(extras)
    for name in ("cpu", "amdgpu", "nvgpu"):
        assert "hpcagent_bench[dev]" in extras[name], name
    conflicts = project["tool"]["uv"]["conflicts"]  # type: ignore[index]
    assert [{"extra": "cpu"}, {"extra": "amdgpu"}, {"extra": "nvgpu"}] in conflicts
    sources = project["tool"]["uv"]["sources"]  # type: ignore[index]
    assert {entry["extra"] for entry in sources["torch"]} == {"cpu", "amdgpu", "nvgpu"}
    indexes = {index["name"]: index["url"] for index in project["tool"]["uv"]["index"]}  # type: ignore[index]
    assert {entry["index"] for entry in sources["torch"]} <= set(indexes)
    assert "rocm-rel-7." in indexes["rocm-radeon"], "AMD's torch links the image's ROCm (no bundled HIP runtime)"
    assert {entry["index"] for entry in sources["torch"] if entry["extra"] == "amdgpu"} == {"rocm-radeon"}
    assert indexes["pytorch-cuda"].rstrip("/").rsplit("/", 1)[1].startswith("cu13")


def test_rocm_jax_and_hip_cupy_are_part_of_the_amdgpu_extra() -> None:
    amdgpu = pyproject()["project"]["optional-dependencies"]["amdgpu"]  # type: ignore[index]
    names = {re.split(r"[=;\s\[<>]", item, maxsplit=1)[0] for item in amdgpu}
    assert {"jaxlib", "jax-rocm7-plugin", "jax-rocm7-pjrt", "cupy", "torch", "triton"} <= names
    assert "cupy-rocm" not in pyproject()["dependency-groups"]  # type: ignore[operator]


def test_the_harness_groups_live_in_the_agent_project_and_conflict_with_the_proxy_stack() -> None:
    agent = pyproject(ROOT / "agent" / "pyproject.toml")
    root = pyproject()
    assert sorted(name for name in agent["dependency-groups"] if name.startswith("harness-")) == list(HARNESS_GROUPS)  # type: ignore[attr-defined]
    assert [name for name in root["dependency-groups"] if name.startswith("harness-")] == []  # type: ignore[attr-defined]
    conflicts = [item for group in root["tool"]["uv"]["conflicts"] for item in group]  # type: ignore[index]
    for name in HARNESS_GROUPS:
        assert {"package": "hpcagent-agent", "group": name} in conflicts, name


def test_the_harbor_adapter_is_a_workspace_member_not_an_editable_install() -> None:
    members = pyproject()["tool"]["uv"]["workspace"]["members"]  # type: ignore[index]
    assert "adapters/hpcagent_bench" in members
    action = (ROOT / ".github" / "actions" / "setup" / "action.yml").read_text(encoding="utf-8")
    assert "editable" not in action


def test_the_openblas_build_tools_are_a_locked_group() -> None:
    groups = pyproject()["dependency-groups"]
    assert {"meson-python", "Cython", "pybind11", "patchelf"} <= set(groups["openblas-build"])  # type: ignore[index]


def test_the_sglang_image_syncs_its_tiny_locked_project_into_the_vendor_venv() -> None:
    project = pyproject(IMAGES / "sglang" / "pyproject.toml")
    assert project["project"]["dependencies"] == ["flydsl==0.3.2", "cupy==14.2.0"]  # type: ignore[index]
    assert (IMAGES / "sglang" / "uv.lock").is_file()
    text = recipe("sglang")
    assert "COPY containers/images/sglang/pyproject.toml containers/images/sglang/uv.lock /opt/sglang-extras/" in text
    assert "UV_PROJECT_ENVIRONMENT=/opt/venv" in text
    assert "uv sync --frozen --inexact" in text and "--no-install-package numpy" in text
    assert "--no-binary-package cupy" in text


if __name__ == "__main__":
    test_no_installer_uses_the_pip_interface()
    for name in JUDGE_AGENT:
        test_a_judge_agent_image_installs_only_what_it_builds_from_source(name)
        test_the_launch_venv_installs_the_extra_and_the_proxy_and_never_an_image_build(name)
        test_every_wheel_gate_runs_in_a_launch_venv_the_image_does_not_keep(name)
        test_the_package_hook_serves_only_the_harness_venvs(name)
        test_the_openblas_rebuild_syncs_the_image_group_only(name)
    test_the_amd_image_builds_cupy_from_its_group_after_the_numpy_rebuild()
    test_the_launch_hook_builds_one_locked_venv_per_pin_beside_the_image_python()
    test_the_rocprof_compute_environment_is_a_locked_project_synced_into_its_venv()
    test_the_extras_are_three_exclusive_framework_sets_that_each_carry_dev()
    test_rocm_jax_and_hip_cupy_are_part_of_the_amdgpu_extra()
    test_the_harness_groups_live_in_the_agent_project_and_conflict_with_the_proxy_stack()
    test_the_harbor_adapter_is_a_workspace_member_not_an_editable_install()
    test_the_openblas_build_tools_are_a_locked_group()
    test_the_sglang_image_syncs_its_tiny_locked_project_into_the_vendor_venv()
