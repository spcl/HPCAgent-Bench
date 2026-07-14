# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the container-launch factory (optarena/containers.py) -- argv assembly,
backend resolution, the Harbor provider name, and the CE launcher. Pure argv assertions:
no real container, GPU, or LLM, so this runs on any CI runner.

The bash<->Python golden parity test lives in test_container_launch_parity.py."""
import os

import pytest

from optarena import containers


@pytest.fixture(autouse=True)
def clean_backend_env(monkeypatch):
    """Drop every ambient container/runtime var so a developer's shell cannot skew the
    argv assertions; each test sets only what it needs."""
    for key in list(os.environ):
        if key.startswith("OPTARENA_") or key in ("OLLAMA_HOST", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(key, raising=False)
    yield


def test_load_backends_lists_the_four_exec_wrappers():
    spellings, passthrough = containers.load_backends()
    assert set(spellings) == {"apptainer", "docker", "podman", "udocker"}
    assert "ANTHROPIC_API_KEY" in passthrough
    assert spellings["apptainer"].verb == ("exec", )
    assert spellings["docker"].verb == ("run", "--rm", "--network", "host")


def test_resolve_backend_precedence(monkeypatch):
    assert containers.resolve_backend("podman") == "podman"  # explicit wins
    monkeypatch.setenv("OPTARENA_RUNTIME_BACKEND", "docker")
    assert containers.resolve_backend() == "docker"  # canonical env next
    monkeypatch.delenv("OPTARENA_RUNTIME_BACKEND")
    assert containers.resolve_backend() == "apptainer"  # config/code default


def test_resolve_backend_ignores_the_legacy_bash_var(monkeypatch):
    # The decouple fix: $OPTARENA_CONTAINER_RUNTIME is the shell launcher's own knob and
    # must NOT steer the Python path (else a Harbor run would crash for a user who set it
    # for a local bash run). Only $OPTARENA_RUNTIME_BACKEND is shared.
    monkeypatch.setenv("OPTARENA_CONTAINER_RUNTIME", "podman")
    assert containers.resolve_backend() == "apptainer"


def test_resolve_backend_rejects_unknown():
    with pytest.raises(ValueError):
        containers.resolve_backend("singularity")  # dropped as a selectable backend


def test_local_run_command_apptainer_cpu():
    argv = containers.local_run_command(["python", "-m", "optarena.cli", "agent"],
                                        backend="apptainer",
                                        hardware="cpu",
                                        repo_root="/repo")
    assert argv == [
        "apptainer", "exec", "--env", "OPTARENA_IMAGE=cpu", "--bind", "/repo:/repo", "--pwd", "/repo",
        "/repo/optarena-cpu.sif", "python", "-m", "optarena.cli", "agent"
    ]


def test_local_run_command_docker_nvidia_gpu_tokens():
    argv = containers.local_run_command(["run"], backend="docker", hardware="nvidia", repo_root="/r")
    # docker run --rm --network host --gpus all ...
    assert argv[:6] == ["docker", "run", "--rm", "--network", "host", "--gpus"]
    assert "all" in argv
    assert argv[-2:] == ["optarena:nvidia", "run"]


def test_local_run_command_podman_amd_gpu_tokens():
    argv = containers.local_run_command(["x"], backend="podman", hardware="amd", repo_root="/r")
    assert "/dev/kfd" in argv and "--group-add" in argv and "keep-groups" in argv


def test_udocker_gpu_token_is_empty_by_design():
    # udocker enables NVIDIA at SETUP time (`udocker setup --nvidia`), never as a run flag,
    # so the fold adds no GPU token -- correct for a properly-set-up udocker.
    cpu = containers.local_run_command(["x"], backend="udocker", hardware="cpu", repo_root="/r")
    nv = containers.local_run_command(["x"], backend="udocker", hardware="nvidia", repo_root="/r")
    assert len(nv) == len(cpu)  # nvidia adds no GPU token (only the hw-dependent image/env differ)
    assert not any(t in nv for t in ("--gpus", "--nv", "--rocm", "--device"))


def test_local_run_command_rejects_ce():
    with pytest.raises(ValueError):
        containers.local_run_command(["x"], backend="ce")


def test_default_image_sif_tag_and_overrides(monkeypatch):
    assert containers.default_image("apptainer", "cpu", repo_root="/r") == "/r/optarena-cpu.sif"
    assert containers.default_image("docker", "nvidia") == "optarena:nvidia"
    monkeypatch.setenv("OPTARENA_SIF", "/scratch/my.sif")
    assert containers.default_image("apptainer", "cpu", repo_root="/r") == "/scratch/my.sif"
    monkeypatch.setenv("OPTARENA_DOCKER_IMAGE", "reg/img:tag")
    assert containers.default_image("docker", "cpu") == "reg/img:tag"


def test_collect_env_order_is_pinned(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk")  # a passthrough (non-OPTARENA) var
    monkeypatch.setenv("OPTARENA_ZED", "z")  # dynamic OPTARENA_*, sorts last
    monkeypatch.setenv("OPTARENA_ABC", "a")  # dynamic OPTARENA_*, sorts before ZED
    pairs = containers.collect_env("cpu")
    assert pairs[0] == ("OPTARENA_IMAGE", "cpu")  # image first
    assert ("ANTHROPIC_API_KEY", "sk") in pairs
    keys = [k for k, _ in pairs]
    assert keys.index("OPTARENA_ABC") < keys.index("OPTARENA_ZED")  # sorted
    assert keys.count("OPTARENA_IMAGE") == 1  # no duplicate


def test_harbor_env_for_maps_and_raises():
    assert containers.harbor_env_for("apptainer") == "singularity"
    assert containers.harbor_env_for("docker") == "docker"
    for backend in ("podman", "udocker", "ce"):
        with pytest.raises(ValueError):
            containers.harbor_env_for(backend)


def test_ce_launcher_default_and_alps_form():
    from optarena.agent_bench.judge_scheduler import DEFAULT_JUDGE_LAUNCHER
    assert containers.ce_launcher() == DEFAULT_JUDGE_LAUNCHER  # plain judge launcher
    assert containers.ce_launcher() == ("srun", "--nodelist", "{node}", "--gpus", "1", "-n", "1")
    alps = containers.ce_launcher(environment="/e.toml", overlap=True)
    assert alps[-2:] == ("--overlap", "--environment=/e.toml")


def test_require_ce_environment():
    containers.require_ce_environment(["srun", "--environment=/e.toml"])  # glued: ok
    containers.require_ce_environment(["srun", "--environment", "/e.toml"])  # space: ok
    containers.require_ce_environment(["srun"], image="/e.toml")  # explicit image: ok
    with pytest.raises(ValueError):
        containers.require_ce_environment(["srun", "--nodelist", "n0"])  # neither: raise
