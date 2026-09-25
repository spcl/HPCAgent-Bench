# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The GH200 (Daint) and CPU-only images are reached, promoted, verified and served by contract.

Nothing here builds an image. The properties are the ones that fail a campaign silently when they
drift: which EDFs a platform renders and onto which image, which toolchain the EDF PATH resolves to,
which candidate a promotion moves, what the verifier asks of each profile, and whether the Daint
serve command keeps the served model name, window and parsers the agent side keys on.
"""

import importlib.util
import os
import pathlib
import subprocess
import sys
import tomllib
from types import ModuleType

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CE = ROOT / "containers" / "images"
SERVE = ROOT / "containers" / "inference" / "serve-daint.sbatch"
ARCH = os.uname().machine

#: Platform -> (EDF name, template, image) it renders, as images.env names them.
RENDERED = {
    "gh200": [
        ("hpcagent-bench-agent-gh200-latest", "judge-agent-cuda/edf.toml.example", "hpcagent-bench-agent-gh200.sqsh"),
        (
            "hpcagent-bench-judge-gh200-latest",
            "judge-agent-cuda/edf.judge.toml.example",
            "hpcagent-bench-judge-gh200.sqsh",
        ),
        ("hpcagent-bench-vllm-gh200-latest", "vllm-cuda/edf.toml.example", "hpcagent-bench-vllm-gh200.sqsh"),
    ],
    "cpu": [
        (
            f"hpcagent-bench-agent-cpu-{ARCH}-latest",
            "judge-agent-cpu/edf.toml.example",
            f"hpcagent-bench-agent-cpu-{ARCH}.sqsh",
        ),
        (
            f"hpcagent-bench-judge-cpu-{ARCH}-latest",
            "judge-agent-cpu/edf.judge.toml.example",
            f"hpcagent-bench-judge-cpu-{ARCH}.sqsh",
        ),
    ],
}

#: judge-agent template -> the prefixes its PATH must put ahead of /usr/bin, and the toolchain it names.
TOOLCHAIN_EDFS = {
    "judge-agent-cuda/edf.toml.example": (("/opt/gcc/bin", "/opt/view/bin", "/usr/local/cuda/bin"), "/opt/gcc/bin/"),
    "judge-agent-cuda/edf.judge.toml.example": (
        ("/opt/gcc/bin", "/opt/view/bin", "/usr/local/cuda/bin"),
        "/opt/gcc/bin/",
    ),
    "judge-agent-cpu/edf.toml.example": (("/opt/venv/bin", "/usr/local/bin", "/usr/lib/llvm-22/bin"), "/usr/bin/"),
    "judge-agent-cpu/edf.judge.toml.example": (
        ("/opt/venv/bin", "/usr/local/bin", "/usr/lib/llvm-22/bin"),
        "/usr/bin/",
    ),
}

#: Daint serve: model -> (default nodes, served window, tool-call parser, reasoning parser), the beverin
#: configs' windows and parsers; kimi's default width is beverin's four nodes.
SERVED = {
    "qwen38": (1, 262144, "qwen3_coder", "qwen3"),
    "oss120b": (1, 131072, "openai", "openai_gptoss"),
    "kimi": (4, 262144, "kimi_k2", "kimi_k2"),
}


def load(path: pathlib.Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install(tmp_path: pathlib.Path, platform: str, images: list[str]) -> subprocess.CompletedProcess[str]:
    """install_edfs.sh for ``platform`` against stand-in images under a throwaway SCRATCH."""
    ce, edf_dir = tmp_path / "ce", tmp_path / "edf"
    ce.mkdir(exist_ok=True)
    for image in images:
        (ce / image).write_bytes(b"sqsh")
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "SCRATCH": str(tmp_path),
        "CE_IMAGES": str(ce),
        "EDF_DIR": str(edf_dir),
        "CE_PLATFORM": platform,
    }
    return subprocess.run(["bash", str(CE / "install_edfs.sh")], capture_output=True, text=True, check=False, env=env)


@pytest.mark.parametrize("platform", sorted(RENDERED))
def test_a_platform_renders_its_roles_onto_its_own_images(tmp_path: pathlib.Path, platform: str) -> None:
    done = install(tmp_path, platform, [image for _, _, image in RENDERED[platform]])
    assert done.returncode == 0, done.stderr
    for edf, _, image in RENDERED[platform]:
        rendered = tomllib.loads((tmp_path / "edf" / f"{edf}.toml").read_text(encoding="utf-8"))
        assert rendered["image"] == str(tmp_path / "ce" / image), (edf, rendered["image"])
        assert "<hpcagent_bench_edf_mounts>" not in rendered["mounts"], edf


@pytest.mark.parametrize("platform", sorted(RENDERED))
def test_a_platform_renders_nothing_of_another(tmp_path: pathlib.Path, platform: str) -> None:
    """A Daint checkout has no beverin images; rendering them there would only report failures."""
    install(tmp_path, platform, [image for _, _, image in RENDERED[platform]])
    names = {path.stem for path in (tmp_path / "edf").glob("*.toml")}
    assert names == {edf for edf, _, _ in RENDERED[platform]}, names


def test_the_default_platform_still_renders_no_gh200_or_cpu_name(tmp_path: pathlib.Path) -> None:
    """Beverin's install must stay what it was before the switch existed."""
    images = [image for roles in RENDERED.values() for _, _, image in roles]
    install(tmp_path, "amd", images)
    names = {path.stem for path in (tmp_path / "edf").glob("*.toml")}
    assert names.isdisjoint(edf for roles in RENDERED.values() for edf, _, _ in roles), names


def test_an_unknown_platform_is_refused(tmp_path: pathlib.Path) -> None:
    done = install(tmp_path, "mi250", [])
    assert done.returncode == 2
    assert "CE_PLATFORM must be amd, gh200 or cpu" in done.stderr


@pytest.mark.parametrize("template", sorted(TOOLCHAIN_EDFS))
def test_a_judge_agent_edf_resolves_the_image_toolchain_before_the_distro(template: str) -> None:
    prefixes, toolchain = TOOLCHAIN_EDFS[template]
    env = tomllib.loads((CE / template).read_text(encoding="utf-8"))["env"]
    path = env["PATH"].split(":")
    late = [prefix for prefix in prefixes if prefix not in path or path.index(prefix) > path.index("/usr/bin")]
    assert late == [], late
    assert all(env[var].startswith(toolchain) for var in ("CC", "CXX", "FC")), env
    assert env["PYTHONSAFEPATH"] == "1"


@pytest.mark.parametrize("template", ["judge-agent-cuda/edf.toml.example", "judge-agent-cuda/edf.judge.toml.example"])
def test_the_gh200_edfs_keep_the_base_images_open_mpi_off_path(template: str) -> None:
    """The NGC base ships HPC-X Open MPI in /usr/local/mpi/bin; on PATH it pairs an Open MPI mpicc
    with an MPICH mpiexec, and P ranks each come up as their own COMM_WORLD of size 1."""
    env = tomllib.loads((CE / template).read_text(encoding="utf-8"))["env"]
    assert "/usr/local/mpi/bin" not in env["PATH"].split(":"), env["PATH"]
    assert "FI_PROVIDER" not in env, "MPICH inherits FI_PROVIDER and MPI_Init aborts (629966)"


@pytest.mark.parametrize("platform", sorted(RENDERED))
def test_promotion_moves_exactly_the_candidates_the_builds_write(tmp_path: pathlib.Path, platform: str) -> None:
    """build.sh writes <live>-candidate.sqsh; a map that disagrees promotes nothing, or the wrong file."""
    ce = tmp_path / "ce"
    ce.mkdir()
    for _, _, image in RENDERED[platform]:
        candidate = ce / image.replace(".sqsh", "-candidate.sqsh")
        candidate.write_bytes(b"sqsh")
        (ce / f"{candidate.name}.digest").write_text("sha256:x\n", encoding="utf-8")
        (ce / f"{candidate.name}.verified").write_text("verified digest=sha256:x\n", encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "SCRATCH": str(tmp_path), "CE_IMAGES": str(ce)}
    env |= {"CE_PLATFORM": platform, "DRY_RUN": "1"}
    done = subprocess.run(
        ["bash", str(CE / "promote_image.sh"), "--all"], capture_output=True, text=True, check=False, env=env
    )
    assert done.returncode == 0, done.stdout + done.stderr
    for _, _, image in RENDERED[platform]:
        assert f"-> {image}" in done.stdout, done.stdout


@pytest.fixture(name="verify", scope="module")
def verify_fixture() -> ModuleType:
    return load(CE / "verify_image.py", "gh200_cpu_verify_image")


@pytest.mark.parametrize("profile", ["judge-agent-cuda", "judge-cuda", "judge-agent-cpu", "judge-cpu"])
def test_every_new_judge_agent_profile_requires_the_agent_runtimes(verify: ModuleType, profile: str) -> None:
    agent = {check.name: check.required for check in verify.checks(profile) if check.group == "agent"}
    want = {"openai-agents SDK", "claude CLI", *(f"{name} interpreter" for name in verify.HARNESS_RUNTIMES)}
    assert set(agent) == want and all(agent.values()), agent


@pytest.mark.parametrize(("profile", "required"), [("judge", True), ("judge-cuda", False), ("judge-cpu", False)])
def test_the_library_registry_is_held_to_a_record_only_where_one_was_measured(
    verify: ModuleType, profile: str, required: bool
) -> None:
    registry = [check for check in verify.checks(profile) if check.kind == "library-registry"]
    assert [check.required for check in registry] == [required], registry


@pytest.mark.parametrize("gpu_only", ["ppcg", "cupy", "triton", "hipcc", "nvcc", "rocprofv3", "ncu"])
def test_the_cpu_profile_asks_for_nothing_gpu_only(verify: ModuleType, gpu_only: str) -> None:
    targets = {check.target for check in verify.checks("judge-agent-cpu")}
    assert gpu_only not in targets, gpu_only


def test_the_gh200_serving_profile_checks_the_engine_and_the_hook_fabric(verify: ModuleType) -> None:
    names = {check.name for check in verify.checks("vllm-cuda")}
    assert {"vllm", "triton", "libfabric", "libcxi", "torch"} <= names, names
    assert names.isdisjoint({"aiter", "flydsl", "rocBLAS"}), names


def serve(model: str, **extra: str) -> subprocess.CompletedProcess[str]:
    """serve-daint.sbatch in DRY_RUN, with nothing of the caller's job or node choice leaking in.

    SCRATCH is a fixed path, never the caller's: the script names its run dir and (through
    scripts/cache_env.sh) its JIT cache under it, and a host with no SCRATCH -- a CI runner --
    otherwise refuses before printing the command this test reads. A dry run creates neither."""
    inherited = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_") and k != "SERVE_NODES"}
    env = inherited | {"MODEL": model, "DRY_RUN": "1", "SCRATCH": "/serve-daint-dry-run", **extra}
    return subprocess.run(["bash", str(SERVE)], capture_output=True, text=True, check=False, env=env, cwd=ROOT)


def argv(done: subprocess.CompletedProcess[str]) -> str:
    return next(line for line in done.stdout.splitlines() if line.startswith("argv: "))


@pytest.mark.parametrize("model", sorted(SERVED))
def test_the_daint_serve_keeps_the_served_name_window_and_parsers(model: str) -> None:
    _, window, tool, reasoning = SERVED[model]
    done = serve(model)
    assert done.returncode == 0, done.stderr
    line = argv(done)
    for words in (
        "--served-model-name hpcagent-bench-vllm",
        f"--max-model-len {window}",
        f"--tool-call-parser {tool}",
        f"--reasoning-parser {reasoning}",
        "--enable-auto-tool-choice",
    ):
        assert words in line, (words, line)


@pytest.mark.parametrize("model", sorted(SERVED))
def test_the_agent_driver_reads_the_daint_window_off_the_serve_command(model: str) -> None:
    """claude-code compacts against agent_driver.served_context; a spelling it cannot read falls back to
    the 262144 cap and overflows a 131072 server."""
    _, window, _, _ = SERVED[model]
    driver = load(ROOT / "experiments" / "agent_driver.py", "gh200_cpu_agent_driver")
    assert driver.served_context({"VLLM_EXTRA_ARGS": argv(serve(model))}) == window


@pytest.mark.parametrize("model", sorted(SERVED))
def test_a_daint_serve_is_one_pipeline_across_its_default_width(model: str) -> None:
    nodes = SERVED[model][0]
    line = argv(serve(model))
    assert "--tensor-parallel-size 4" in line, line
    if nodes == 1:
        assert "--pipeline-parallel-size" not in line and "--nnodes" not in line, line
    else:
        assert f"--pipeline-parallel-size {nodes}" in line and f"--nnodes {nodes}" in line, line


def test_kimi_may_run_on_the_two_node_floor_when_asked() -> None:
    line = argv(serve("kimi", SERVE_NODES="2"))
    assert "--pipeline-parallel-size 2" in line and "--nnodes 2" in line, line


def test_kimi_is_refused_on_a_node_that_cannot_hold_its_weights() -> None:
    done = serve("kimi", SERVE_NODES="1")
    assert done.returncode == 2
    assert "kimi needs at least 2 node(s)" in done.stderr


def test_a_job_whose_node_count_is_not_the_serve_width_is_refused() -> None:
    """-N and the engine's PP must agree: fewer nodes than PP never comes up, more sit idle."""
    done = serve("kimi", SLURM_JOB_ID="1", SLURM_JOB_NUM_NODES="2")
    assert done.returncode == 2
    assert "submit with -N 4, or set SERVE_NODES" in done.stderr


@pytest.mark.parametrize("word", ["--host=0.0.0.0", "--port", "--api-key=x"])
def test_the_daint_serve_refuses_extra_args_that_rebind_or_rekey_it(word: str) -> None:
    done = serve("qwen38", EXTRA_ARGS=word)
    assert done.returncode == 2
    assert f"EXTRA_ARGS may not set {word.split('=')[0]}" in done.stderr
